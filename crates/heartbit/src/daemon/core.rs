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

use super::notify::{OnTaskComplete, TaskOutcome};
use super::store::TaskStore;
use super::types::{DaemonCommand, DaemonTask, TaskState, TaskStats};

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
            trust_level: None,
            user_id: None,
            tenant_id: None,
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

    /// Submit a task with user context for multi-tenant isolation.
    ///
    /// Like `submit_task`, but attaches user/tenant identity to the command
    /// and creates the task record with user context.
    pub async fn submit_task_with_user(
        &self,
        task: impl Into<String>,
        source: impl Into<String>,
        story_id: Option<String>,
        user_context: &super::types::UserContext,
    ) -> Result<uuid::Uuid, Error> {
        let id = uuid::Uuid::new_v4();
        let task_str = task.into();
        let source_str = source.into();

        let daemon_task = DaemonTask::new_with_user(
            id,
            &task_str,
            &source_str,
            &user_context.user_id,
            &user_context.tenant_id,
        );
        self.store.insert(daemon_task)?;

        let cmd = DaemonCommand::SubmitTask {
            id,
            task: task_str,
            source: source_str,
            story_id,
            trust_level: None,
            user_id: Some(user_context.user_id.clone()),
            tenant_id: Some(user_context.tenant_id.clone()),
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
        let channels = match self.event_channels.read() {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, "event_channels lock poisoned in subscribe_events");
                return None;
            }
        };
        channels.get(&id).map(|tx| tx.subscribe())
    }

    /// Register a task directly in the store (for non-Kafka execution paths like Telegram/WS).
    pub fn register_task(
        &self,
        id: uuid::Uuid,
        task: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<(), Error> {
        let daemon_task = DaemonTask::new(id, task, source);
        self.store.insert(daemon_task)
    }

    /// Register a task with user context for multi-tenant isolation.
    pub fn register_task_with_user(
        &self,
        id: uuid::Uuid,
        task: impl Into<String>,
        source: impl Into<String>,
        user_id: impl Into<String>,
        tenant_id: impl Into<String>,
    ) -> Result<(), Error> {
        let daemon_task = DaemonTask::new_with_user(id, task, source, user_id, tenant_id);
        self.store.insert(daemon_task)
    }

    /// Update a registered task's state (for non-Kafka execution paths).
    pub fn update_task(&self, id: uuid::Uuid, f: &dyn Fn(&mut DaemonTask)) -> Result<(), Error> {
        self.store.update(id, f)
    }

    /// List tasks with optional source/state/tenant filters.
    pub fn list_tasks_filtered(
        &self,
        limit: usize,
        offset: usize,
        source: Option<&str>,
        state: Option<TaskState>,
        tenant_id: Option<&str>,
    ) -> Result<(Vec<DaemonTask>, usize), Error> {
        self.store
            .list_filtered(limit, offset, source, state, tenant_id)
    }

    /// Aggregate stats, optionally scoped to a tenant.
    pub fn stats(&self, tenant_id: Option<&str>) -> Result<TaskStats, Error> {
        self.store.stats(tenant_id)
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
    /// task text, source tag, optional story ID, optional trust level, and an
    /// event callback, and returns a future that produces the agent output.
    ///
    /// `on_complete` is an optional callback fired when any task reaches a
    /// terminal state (Completed, Failed, Cancelled). Used for proactive
    /// notifications (e.g. Telegram).
    pub async fn run<F, Fut>(
        mut self,
        build_runner: F,
        on_complete: Option<Arc<OnTaskComplete>>,
    ) -> Result<(), Error>
    where
        F: Fn(
                uuid::Uuid,
                String,
                String,
                Option<String>,
                Option<crate::config::TrustLevel>,
                Arc<dyn Fn(AgentEvent) + Send + Sync>,
                Option<String>, // user_id
                Option<String>, // tenant_id
            ) -> Fut
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
                        DaemonCommand::SubmitTask { id, task, source, story_id, trust_level, user_id, tenant_id } => {
                            // Re-insert task if missing (e.g. after restart with message replay)
                            if let Ok(None) = self.store.get(id) {
                                if let (Some(uid), Some(tid)) = (&user_id, &tenant_id) {
                                    let _ = self.store.insert(DaemonTask::new_with_user(
                                        id, &task, &source, uid, tid,
                                    ));
                                } else {
                                    let _ = self.store.insert(DaemonTask::new(id, &task, &source));
                                }
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
                                    let json = match serde_json::to_vec(&event) {
                                        Ok(j) => j,
                                        Err(e) => {
                                            tracing::error!(error = %e, "failed to serialize agent event for kafka");
                                            return;
                                        }
                                    };
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
                            let on_complete = on_complete.clone();
                            let outcome_story_id = story_id.clone();

                            self.active_tasks.spawn(async move {
                                store
                                    .update(id, &|t| {
                                        t.state = TaskState::Running;
                                        t.started_at = Some(Utc::now());
                                    })
                                    .ok();

                                let start = std::time::Instant::now();
                                let runner = build_runner(id, task, source.clone(), story_id, trust_level, on_event, user_id, tenant_id);
                                tokio::select! {
                                    result = runner => {
                                        let duration_secs = start.elapsed().as_secs_f64();
                                        match result {
                                            Ok(output) => {
                                                let tokens = output.tokens_used;
                                                let cost = output.estimated_cost_usd;
                                                let result_text = output.result.clone();
                                                store
                                                    .update(id, &|t| {
                                                        t.state = TaskState::Completed;
                                                        t.completed_at = Some(Utc::now());
                                                        t.result = Some(result_text.clone());
                                                        t.tokens_used = tokens;
                                                        t.tool_calls_made = output.tool_calls_made;
                                                        t.estimated_cost_usd = cost;
                                                    })
                                                    .ok();
                                                if let Some(ref cb) = on_complete {
                                                    cb(TaskOutcome {
                                                        id,
                                                        source: source.clone(),
                                                        state: TaskState::Completed,
                                                        result_summary: Some(result_text),
                                                        error: None,
                                                        duration_secs,
                                                        tokens,
                                                        cost,
                                                        story_id: outcome_story_id.clone(),
                                                    });
                                                }
                                            }
                                            Err(e) => {
                                                let error_str = e.to_string();
                                                store
                                                    .update(id, &|t| {
                                                        t.state = TaskState::Failed;
                                                        t.completed_at = Some(Utc::now());
                                                        t.error = Some(error_str.clone());
                                                    })
                                                    .ok();
                                                if let Some(ref cb) = on_complete {
                                                    cb(TaskOutcome {
                                                        id,
                                                        source: source.clone(),
                                                        state: TaskState::Failed,
                                                        result_summary: None,
                                                        error: Some(error_str),
                                                        duration_secs,
                                                        tokens: Default::default(),
                                                        cost: None,
                                                        story_id: outcome_story_id.clone(),
                                                    });
                                                }
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
                                        if let Some(ref cb) = on_complete {
                                            cb(TaskOutcome {
                                                id,
                                                source: source.clone(),
                                                state: TaskState::Cancelled,
                                                result_summary: None,
                                                error: None,
                                                duration_secs: start.elapsed().as_secs_f64(),
                                                tokens: Default::default(),
                                                cost: None,
                                                story_id: outcome_story_id.clone(),
                                            });
                                        }
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
            heartbit_pulse: None,
            auth: None,
            owner_emails: vec![],
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

    #[test]
    fn register_task_appears_in_store() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle
            .register_task(id, "do something", "telegram")
            .unwrap();

        let task = handle.get_task(id).unwrap().unwrap();
        assert_eq!(task.id, id);
        assert_eq!(task.task, "do something");
        assert_eq!(task.source, "telegram");
        assert_eq!(task.state, TaskState::Pending);
    }

    #[test]
    fn register_task_duplicate_rejected() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle.register_task(id, "first", "api").unwrap();
        let err = handle.register_task(id, "second", "api").unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn update_task_modifies_state() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle.register_task(id, "test", "ws").unwrap();

        handle
            .update_task(id, &|t| {
                t.state = TaskState::Running;
                t.started_at = Some(chrono::Utc::now());
            })
            .unwrap();

        let task = handle.get_task(id).unwrap().unwrap();
        assert_eq!(task.state, TaskState::Running);
        assert!(task.started_at.is_some());
    }

    #[test]
    fn update_task_nonexistent_returns_error() {
        let handle = test_handle();
        let err = handle
            .update_task(uuid::Uuid::new_v4(), &|_| {})
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn handle_list_tasks_filtered_by_source() {
        let handle = test_handle();
        for i in 0..3 {
            let source = if i < 2 { "telegram" } else { "api" };
            handle
                .register_task(uuid::Uuid::new_v4(), format!("task {i}"), source)
                .unwrap();
        }

        let (tasks, total) = handle
            .list_tasks_filtered(10, 0, Some("telegram"), None, None)
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().all(|t| t.source == "telegram"));
    }

    #[test]
    fn handle_list_tasks_filtered_by_state() {
        let handle = test_handle();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        handle.register_task(id1, "a", "api").unwrap();
        handle.register_task(id2, "b", "api").unwrap();
        handle
            .update_task(id1, &|t| t.state = TaskState::Running)
            .unwrap();

        let (tasks, total) = handle
            .list_tasks_filtered(10, 0, None, Some(TaskState::Running), None)
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(tasks[0].id, id1);
    }

    #[test]
    fn handle_stats_aggregates() {
        let handle = test_handle();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let id3 = uuid::Uuid::new_v4();
        handle.register_task(id1, "a", "api").unwrap();
        handle.register_task(id2, "b", "telegram").unwrap();
        handle.register_task(id3, "c", "api").unwrap();
        handle
            .update_task(id2, &|t| t.state = TaskState::Running)
            .unwrap();
        handle
            .update_task(id3, &|t| {
                t.state = TaskState::Completed;
                t.tokens_used.input_tokens = 100;
                t.estimated_cost_usd = Some(0.01);
            })
            .unwrap();

        let stats = handle.stats(None).unwrap();
        assert_eq!(stats.total_tasks, 3);
        assert_eq!(stats.active_tasks, 1);
        assert_eq!(stats.tasks_by_source.get("api"), Some(&2));
        assert_eq!(stats.tasks_by_source.get("telegram"), Some(&1));
        assert_eq!(stats.tasks_by_state.get("running"), Some(&1));
        assert_eq!(stats.tasks_by_state.get("completed"), Some(&1));
        assert_eq!(stats.total_input_tokens, 100);
        assert!((stats.total_estimated_cost_usd - 0.01).abs() < 1e-9);
    }

    #[test]
    fn handle_list_filtered_by_tenant() {
        let handle = test_handle();
        let task1 = DaemonTask::new_with_user(uuid::Uuid::new_v4(), "a", "api", "alice", "acme");
        let task2 = DaemonTask::new_with_user(uuid::Uuid::new_v4(), "b", "api", "bob", "globex");
        let task3 =
            DaemonTask::new_with_user(uuid::Uuid::new_v4(), "c", "telegram", "carol", "acme");
        handle.store.insert(task1).unwrap();
        handle.store.insert(task2).unwrap();
        handle.store.insert(task3).unwrap();

        let (tasks, total) = handle
            .list_tasks_filtered(10, 0, None, None, Some("acme"))
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().all(|t| t.tenant_id.as_deref() == Some("acme")));
    }

    #[test]
    fn handle_stats_filtered_by_tenant() {
        let handle = test_handle();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let task1 = DaemonTask::new_with_user(id1, "a", "api", "alice", "acme");
        let task2 = DaemonTask::new_with_user(id2, "b", "api", "bob", "globex");
        handle.store.insert(task1).unwrap();
        handle.store.insert(task2).unwrap();
        handle
            .update_task(id1, &|t| t.tokens_used.input_tokens = 50)
            .unwrap();
        handle
            .update_task(id2, &|t| t.tokens_used.input_tokens = 100)
            .unwrap();

        let stats = handle.stats(Some("acme")).unwrap();
        assert_eq!(stats.total_tasks, 1);
        assert_eq!(stats.total_input_tokens, 50);

        let stats = handle.stats(None).unwrap();
        assert_eq!(stats.total_tasks, 2);
        assert_eq!(stats.total_input_tokens, 150);
    }

    #[test]
    fn register_task_with_user_stores_user_context() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        let task = DaemonTask::new_with_user(id, "check deals", "api", "alice", "acme");
        handle.store.insert(task).unwrap();

        let fetched = handle.get_task(id).unwrap().unwrap();
        assert_eq!(fetched.user_id.as_deref(), Some("alice"));
        assert_eq!(fetched.tenant_id.as_deref(), Some("acme"));
        assert_eq!(fetched.task, "check deals");
    }

    #[test]
    fn register_task_without_user_has_none_context() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle.register_task(id, "basic task", "api").unwrap();

        let task = handle.get_task(id).unwrap().unwrap();
        assert!(task.user_id.is_none());
        assert!(task.tenant_id.is_none());
    }

    #[test]
    fn register_task_with_user_method_stores_context() {
        let handle = test_handle();
        let id = uuid::Uuid::new_v4();
        handle
            .register_task_with_user(id, "user task", "ws", "bob", "globex")
            .unwrap();

        let task = handle.get_task(id).unwrap().unwrap();
        assert_eq!(task.user_id.as_deref(), Some("bob"));
        assert_eq!(task.tenant_id.as_deref(), Some("globex"));
        assert_eq!(task.source, "ws");
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
