use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use chrono::Utc;
use rdkafka::consumer::StreamConsumer;
use rdkafka::message::Message;
use rdkafka::producer::{FutureProducer, FutureRecord};
use tokio::sync::{Semaphore, broadcast};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::agent::AgentOutput;
use crate::agent::events::AgentEvent;
use crate::config::DaemonConfig;

use super::store::TaskStore;
use super::types::{DaemonCommand, DaemonTask, TaskState};

/// Cloneable handle for producing commands and reading state.
#[derive(Clone)]
pub struct DaemonHandle {
    producer: FutureProducer,
    commands_topic: String,
    store: Arc<dyn TaskStore>,
    event_channels: Arc<std::sync::RwLock<HashMap<uuid::Uuid, broadcast::Sender<AgentEvent>>>>,
}

impl DaemonHandle {
    /// Submit a task: create in store as Pending, produce `SubmitTask` to Kafka.
    pub async fn submit_task(
        &self,
        task: impl Into<String>,
        source: impl Into<String>,
        story_id: Option<String>,
    ) -> Result<uuid::Uuid, Error> {
        let id = uuid::Uuid::new_v4();
        let task_str = task.into();
        let source_str = source.into();

        let daemon_task = DaemonTask::new(id, &task_str, &source_str);
        self.store.insert(daemon_task)?;

        let cmd = DaemonCommand::SubmitTask {
            id,
            task: task_str,
            source: source_str,
            story_id,
        };
        let payload = serde_json::to_vec(&cmd)
            .map_err(|e| Error::Daemon(format!("failed to serialize command: {e}")))?;

        self.producer
            .send(
                FutureRecord::to(&self.commands_topic)
                    .key(&id.to_string())
                    .payload(&payload),
                rdkafka::util::Timeout::Never,
            )
            .await
            .map_err(|(e, _)| Error::Daemon(format!("failed to produce command: {e}")))?;

        Ok(id)
    }

    /// Read task from store.
    pub fn get_task(&self, id: uuid::Uuid) -> Result<Option<DaemonTask>, Error> {
        self.store.get(id)
    }

    /// List tasks from store.
    pub fn list_tasks(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<DaemonTask>, usize), Error> {
        self.store.list(limit, offset)
    }

    /// Subscribe to real-time events for a task (for SSE).
    pub fn subscribe_events(&self, id: uuid::Uuid) -> Option<broadcast::Receiver<AgentEvent>> {
        let channels = self.event_channels.read().ok()?;
        channels.get(&id).map(|tx| tx.subscribe())
    }

    /// Produce a `CancelTask` command.
    pub async fn cancel_task(&self, id: uuid::Uuid) -> Result<(), Error> {
        let cmd = DaemonCommand::CancelTask { id };
        let payload = serde_json::to_vec(&cmd)
            .map_err(|e| Error::Daemon(format!("failed to serialize command: {e}")))?;

        self.producer
            .send(
                FutureRecord::to(&self.commands_topic)
                    .key(&id.to_string())
                    .payload(&payload),
                rdkafka::util::Timeout::Never,
            )
            .await
            .map_err(|(e, _)| Error::Daemon(format!("failed to produce cancel: {e}")))?;

        Ok(())
    }
}

/// The daemon event loop. Consumes commands from Kafka, executes tasks.
pub struct DaemonCore {
    consumer: StreamConsumer,
    producer: FutureProducer,
    events_topic: String,
    store: Arc<dyn TaskStore>,
    semaphore: Arc<Semaphore>,
    event_channels: Arc<std::sync::RwLock<HashMap<uuid::Uuid, broadcast::Sender<AgentEvent>>>>,
    task_cancels: Arc<std::sync::RwLock<HashMap<uuid::Uuid, CancellationToken>>>,
    active_tasks: JoinSet<()>,
    cancel: CancellationToken,
}

impl DaemonCore {
    pub fn new(
        config: &DaemonConfig,
        consumer: StreamConsumer,
        producer: FutureProducer,
        store: Arc<dyn TaskStore>,
        cancel: CancellationToken,
    ) -> (Self, DaemonHandle) {
        let event_channels = Arc::new(std::sync::RwLock::new(HashMap::new()));
        let handle = DaemonHandle {
            producer: producer.clone(),
            commands_topic: config.kafka.commands_topic.clone(),
            store: store.clone(),
            event_channels: event_channels.clone(),
        };
        let core = Self {
            consumer,
            producer,
            events_topic: config.kafka.events_topic.clone(),
            store,
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_tasks)),
            event_channels,
            task_cancels: Arc::new(std::sync::RwLock::new(HashMap::new())),
            active_tasks: JoinSet::new(),
            cancel,
        };
        (core, handle)
    }

    /// Run the Kafka consumer loop. Blocks until cancellation.
    ///
    /// `build_runner` is called for each submitted task. It receives the task ID,
    /// task text, and an event callback, and returns a future that produces the
    /// agent output.
    pub async fn run<F, Fut>(mut self, build_runner: F) -> Result<(), Error>
    where
        F: Fn(uuid::Uuid, String, Option<String>, Arc<dyn Fn(AgentEvent) + Send + Sync>) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = Result<AgentOutput, Error>> + Send + 'static,
    {
        use futures::StreamExt;

        let build_runner = Arc::new(build_runner);

        let mut stream = self.consumer.stream();

        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => {
                    tracing::info!("daemon core shutting down, draining active tasks");
                    while self.active_tasks.join_next().await.is_some() {}
                    break;
                }
                msg = stream.next() => {
                    let Some(msg_result) = msg else {
                        tracing::warn!("kafka consumer stream ended unexpectedly");
                        break;
                    };
                    let msg = match msg_result {
                        Ok(m) => m,
                        Err(e) => {
                            tracing::error!(error = %e, "kafka consumer error");
                            continue;
                        }
                    };
                    let payload = match msg.payload() {
                        Some(p) => p,
                        None => continue,
                    };
                    let cmd: DaemonCommand = match serde_json::from_slice(payload) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!(error = %e, "failed to deserialize daemon command");
                            continue;
                        }
                    };

                    match cmd {
                        DaemonCommand::SubmitTask { id, task, source, story_id } => {
                            // Re-insert task if missing (e.g. after restart with message replay)
                            if let Ok(None) = self.store.get(id) {
                                let _ = self.store.insert(DaemonTask::new(id, &task, &source));
                            }

                            let permit = match self.semaphore.clone().acquire_owned().await {
                                Ok(p) => p,
                                Err(_) => break, // semaphore closed
                            };

                            let (tx, _) = broadcast::channel(1024);
                            if let Ok(mut channels) = self.event_channels.write() {
                                channels.insert(id, tx.clone());
                            }

                            // Per-task cancellation token
                            let task_cancel = CancellationToken::new();
                            if let Ok(mut cancels) = self.task_cancels.write() {
                                cancels.insert(id, task_cancel.clone());
                            }

                            // Build on_event that produces to both Kafka and broadcast
                            let event_producer = self.producer.clone();
                            let events_topic = self.events_topic.clone();
                            let on_event: Arc<dyn Fn(AgentEvent) + Send + Sync> =
                                Arc::new(move |event: AgentEvent| {
                                    let _ = tx.send(event.clone());
                                    // Fire-and-forget produce to Kafka
                                    let json = serde_json::to_vec(&event).unwrap_or_default();
                                    drop(event_producer.send(
                                        FutureRecord::to(&events_topic)
                                            .key(&id.to_string())
                                            .payload(&json),
                                        rdkafka::util::Timeout::Never,
                                    ));
                                });

                            let store = self.store.clone();
                            let channels = self.event_channels.clone();
                            let task_cancels = self.task_cancels.clone();
                            let build_runner = build_runner.clone();

                            self.active_tasks.spawn(async move {
                                store
                                    .update(id, &|t| {
                                        t.state = TaskState::Running;
                                        t.started_at = Some(Utc::now());
                                    })
                                    .ok();

                                let runner = build_runner(id, task, story_id, on_event);
                                tokio::select! {
                                    result = runner => {
                                        match result {
                                            Ok(output) => {
                                                store
                                                    .update(id, &|t| {
                                                        t.state = TaskState::Completed;
                                                        t.completed_at = Some(Utc::now());
                                                        t.result = Some(output.result.clone());
                                                        t.tokens_used = output.tokens_used;
                                                        t.tool_calls_made = output.tool_calls_made;
                                                        t.estimated_cost_usd = output.estimated_cost_usd;
                                                    })
                                                    .ok();
                                            }
                                            Err(e) => {
                                                store
                                                    .update(id, &|t| {
                                                        t.state = TaskState::Failed;
                                                        t.completed_at = Some(Utc::now());
                                                        t.error = Some(e.to_string());
                                                    })
                                                    .ok();
                                            }
                                        }
                                    }
                                    _ = task_cancel.cancelled() => {
                                        store
                                            .update(id, &|t| {
                                                t.state = TaskState::Cancelled;
                                                t.completed_at = Some(Utc::now());
                                            })
                                            .ok();
                                    }
                                }

                                if let Ok(mut ch) = channels.write() {
                                    ch.remove(&id);
                                }
                                if let Ok(mut tc) = task_cancels.write() {
                                    tc.remove(&id);
                                }
                                drop(permit);
                            });
                        }
                        DaemonCommand::CancelTask { id } => {
                            // Cancel the running task if it exists
                            if let Ok(cancels) = self.task_cancels.read()
                                && let Some(token) = cancels.get(&id)
                            {
                                token.cancel();
                            }
                            // If task isn't running, just mark cancelled in store
                            if let Ok(Some(task)) = self.store.get(id)
                                && task.state == TaskState::Pending
                            {
                                self.store
                                    .update(id, &|t| {
                                        t.state = TaskState::Cancelled;
                                        t.completed_at = Some(Utc::now());
                                    })
                                    .ok();
                            }
                            if let Ok(mut ch) = self.event_channels.write() {
                                ch.remove(&id);
                            }
                        }
                    }
                }
                Some(result) = self.active_tasks.join_next() => {
                    if let Err(e) = result {
                        tracing::error!("task panicked: {e}");
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::store::InMemoryTaskStore;

    fn test_config() -> DaemonConfig {
        crate::config::DaemonConfig {
            kafka: crate::config::KafkaConfig {
                brokers: "localhost:9092".into(),
                consumer_group: "test".into(),
                commands_topic: "test.commands".into(),
                events_topic: "test.events".into(),
                dead_letter_topic: "test.dead-letter".into(),
            },
            bind: "127.0.0.1:0".into(),
            max_concurrent_tasks: 4,
            schedules: vec![],
            metrics: None,
            sensors: None,
            ws: None,
            #[cfg(feature = "telegram")]
            telegram: None,
            database_url: None,
        }
    }

    fn test_producer() -> FutureProducer {
        crate::daemon::kafka::create_producer(&test_config().kafka).unwrap()
    }

    fn test_handle() -> DaemonHandle {
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let event_channels = Arc::new(std::sync::RwLock::new(HashMap::new()));
        DaemonHandle {
            producer: test_producer(),
            commands_topic: "test.commands".into(),
            store,
            event_channels,
        }
    }

    #[tokio::test]
    async fn daemon_core_new_returns_handle() {
        let config = test_config();
        let producer = test_producer();
        let consumer = crate::daemon::kafka::create_commands_consumer(&config.kafka).unwrap();
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let cancel = CancellationToken::new();

        let (_core, handle) = DaemonCore::new(&config, consumer, producer, store.clone(), cancel);

        // Handle should be usable — get_task returns None for unknown ID
        assert!(handle.get_task(uuid::Uuid::new_v4()).unwrap().is_none());
    }

    #[test]
    fn handle_get_task_returns_inserted() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        let task = DaemonTask::new(id, "test", "api");
        handle.store.insert(task).unwrap();

        let fetched = handle.get_task(id).unwrap().unwrap();
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.task, "test");
    }

    #[test]
    fn handle_get_task_not_found() {
        let handle = test_handle();
        assert!(handle.get_task(uuid::Uuid::new_v4()).unwrap().is_none());
    }

    #[test]
    fn handle_list_tasks_returns_stored() {
        let handle = test_handle();
        for i in 0..3 {
            handle
                .store
                .insert(DaemonTask::new(
                    uuid::Uuid::new_v4(),
                    format!("task {i}"),
                    "api",
                ))
                .unwrap();
        }

        let (tasks, total) = handle.list_tasks(10, 0).unwrap();
        assert_eq!(total, 3);
        assert_eq!(tasks.len(), 3);
    }

    #[test]
    fn handle_list_tasks_pagination() {
        let handle = test_handle();
        for i in 0..5 {
            handle
                .store
                .insert(DaemonTask::new(
                    uuid::Uuid::new_v4(),
                    format!("task {i}"),
                    "api",
                ))
                .unwrap();
        }

        let (tasks, total) = handle.list_tasks(2, 1).unwrap();
        assert_eq!(total, 5);
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn handle_subscribe_events_none_when_no_channel() {
        let handle = test_handle();
        assert!(handle.subscribe_events(uuid::Uuid::new_v4()).is_none());
    }

    #[test]
    fn handle_subscribe_events_returns_receiver() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();

        // Insert a broadcast channel
        let (tx, _) = broadcast::channel(16);
        handle.event_channels.write().unwrap().insert(id, tx);

        let rx = handle.subscribe_events(id);
        assert!(rx.is_some());
    }

    #[tokio::test]
    async fn daemon_core_new_semaphore_matches_config() {
        let mut config = test_config();
        config.max_concurrent_tasks = 2;
        let producer = test_producer();
        let consumer = crate::daemon::kafka::create_commands_consumer(&config.kafka).unwrap();
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let cancel = CancellationToken::new();

        let (core, _handle) = DaemonCore::new(&config, consumer, producer, store, cancel);

        // Semaphore should have 2 permits
        let p1 = core.semaphore.clone().try_acquire_owned();
        let p2 = core.semaphore.clone().try_acquire_owned();
        let p3 = core.semaphore.clone().try_acquire_owned();
        assert!(p1.is_ok());
        assert!(p2.is_ok());
        assert!(p3.is_err()); // third should fail — only 2 permits
    }
}
