use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use rdkafka::admin::{AdminClient, AdminOptions, NewTopic, TopicReplication};
use rdkafka::client::DefaultClientContext;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::stream_consumer::StreamConsumer;
use rdkafka::producer::{FutureProducer, FutureRecord};

use crate::Error;
use crate::config::KafkaConfig;

use super::CommandProducer;

/// Create topics if they don't exist. Called once at daemon startup.
pub async fn ensure_topics(config: &KafkaConfig) -> Result<(), Error> {
    let admin: AdminClient<DefaultClientContext> = ClientConfig::new()
        .set("bootstrap.servers", &config.brokers)
        .create()
        .map_err(|e| Error::Daemon(format!("failed to create admin client: {e}")))?;

    let retention = (7 * 24 * 60 * 60 * 1000i64).to_string();
    let commands = NewTopic::new(&config.commands_topic, 4, TopicReplication::Fixed(1))
        .set("cleanup.policy", "delete")
        .set("retention.ms", &retention);

    let events = NewTopic::new(&config.events_topic, 8, TopicReplication::Fixed(1))
        .set("cleanup.policy", "delete")
        .set("retention.ms", &retention);

    let opts = AdminOptions::new().operation_timeout(Some(Duration::from_secs(10)));

    let results = admin
        .create_topics(&[commands, events], &opts)
        .await
        .map_err(|e| Error::Daemon(format!("topic creation failed: {e}")))?;

    for result in results {
        match result {
            Ok(_) => {}
            Err((topic, code)) => {
                // TopicAlreadyExists is fine — we want idempotency
                if code != rdkafka::types::RDKafkaErrorCode::TopicAlreadyExists {
                    return Err(Error::Daemon(format!(
                        "failed to create topic '{topic}': {code}"
                    )));
                }
            }
        }
    }

    Ok(())
}

/// Build a `FutureProducer` from config.
pub fn create_producer(config: &KafkaConfig) -> Result<FutureProducer, Error> {
    ClientConfig::new()
        .set("bootstrap.servers", &config.brokers)
        .set("message.timeout.ms", "30000")
        .set("linger.ms", "5")
        .create()
        .map_err(|e| Error::Daemon(format!("failed to create producer: {e}")))
}

/// [`CommandProducer`] backed by a real Kafka [`FutureProducer`].
pub struct KafkaCommandProducer(FutureProducer);

impl KafkaCommandProducer {
    pub fn new(producer: FutureProducer) -> Self {
        Self(producer)
    }
}

impl CommandProducer for KafkaCommandProducer {
    fn send_command<'a>(
        &'a self,
        topic: &'a str,
        key: &'a str,
        payload: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            self.0
                .send(
                    FutureRecord::to(topic).key(key).payload(payload),
                    rdkafka::util::Timeout::Never,
                )
                .await
                .map(|_| ())
                .map_err(|(e, _)| Error::Daemon(format!("kafka send failed: {e}")))
        })
    }
}

/// Build a `StreamConsumer` subscribed to the commands topic.
pub fn create_commands_consumer(config: &KafkaConfig) -> Result<StreamConsumer, Error> {
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", &config.brokers)
        .set("group.id", &config.consumer_group)
        .set("auto.offset.reset", "earliest")
        .set("enable.auto.commit", "true")
        .create()
        .map_err(|e| Error::Daemon(format!("failed to create consumer: {e}")))?;

    rdkafka::consumer::Consumer::subscribe(&consumer, &[&config.commands_topic]).map_err(|e| {
        Error::Daemon(format!(
            "failed to subscribe to {}: {e}",
            config.commands_topic
        ))
    })?;

    Ok(consumer)
}

/// Sensor topic definitions.
pub struct SensorTopicConfig {
    pub topic: &'static str,
    pub partitions: i32,
    pub retention_ms: i64,
    pub cleanup_policy: &'static str,
}

/// All sensor topics with their default configurations.
pub const SENSOR_TOPICS: &[SensorTopicConfig] = &[
    SensorTopicConfig {
        topic: "hb.sensor.email",
        partitions: 4,
        retention_ms: 7 * 24 * 60 * 60 * 1000,
        cleanup_policy: "delete",
    },
    SensorTopicConfig {
        topic: "hb.sensor.image",
        partitions: 2,
        retention_ms: 14 * 24 * 60 * 60 * 1000,
        cleanup_policy: "delete",
    },
    SensorTopicConfig {
        topic: "hb.sensor.audio",
        partitions: 2,
        retention_ms: 7 * 24 * 60 * 60 * 1000,
        cleanup_policy: "delete",
    },
    SensorTopicConfig {
        topic: "hb.sensor.rss",
        partitions: 2,
        retention_ms: 3 * 24 * 60 * 60 * 1000,
        cleanup_policy: "delete",
    },
    SensorTopicConfig {
        topic: "hb.sensor.weather",
        partitions: 1,
        retention_ms: 48 * 60 * 60 * 1000,
        cleanup_policy: "delete",
    },
    SensorTopicConfig {
        topic: "hb.sensor.webhook",
        partitions: 4,
        retention_ms: 3 * 24 * 60 * 60 * 1000,
        cleanup_policy: "delete",
    },
    SensorTopicConfig {
        topic: "hb.stories",
        partitions: 1,
        retention_ms: -1, // unlimited
        cleanup_policy: "compact",
    },
    SensorTopicConfig {
        topic: "hb.dead-letter",
        partitions: 2,
        retention_ms: 30 * 24 * 60 * 60 * 1000,
        cleanup_policy: "delete",
    },
];

/// Ensure sensor-related Kafka topics exist. Called at daemon startup when
/// the sensor layer is enabled. Idempotent — existing topics are not modified.
pub async fn ensure_sensor_topics(config: &KafkaConfig) -> Result<(), Error> {
    let admin: AdminClient<DefaultClientContext> = ClientConfig::new()
        .set("bootstrap.servers", &config.brokers)
        .create()
        .map_err(|e| Error::Sensor(format!("failed to create admin client: {e}")))?;

    let opts = AdminOptions::new().operation_timeout(Some(Duration::from_secs(10)));

    // Process topics in batches to avoid lifetime issues with retention strings.
    for topic_cfg in SENSOR_TOPICS {
        let retention = topic_cfg.retention_ms.to_string();
        let mut nt = NewTopic::new(
            topic_cfg.topic,
            topic_cfg.partitions,
            TopicReplication::Fixed(1),
        )
        .set("cleanup.policy", topic_cfg.cleanup_policy);
        if topic_cfg.retention_ms >= 0 {
            nt = nt.set("retention.ms", &retention);
        }

        let results = admin
            .create_topics(&[nt], &opts)
            .await
            .map_err(|e| Error::Sensor(format!("sensor topic creation failed: {e}")))?;

        for result in results {
            match result {
                Ok(_) => {}
                Err((topic, code)) => {
                    if code != rdkafka::types::RDKafkaErrorCode::TopicAlreadyExists {
                        return Err(Error::Sensor(format!(
                            "failed to create sensor topic '{topic}': {code}"
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Build a `StreamConsumer` subscribed to a specific sensor topic.
pub fn create_sensor_consumer(
    config: &KafkaConfig,
    topic: &str,
    group_suffix: &str,
) -> Result<StreamConsumer, Error> {
    let group_id = format!("{}-sensor-{}", config.consumer_group, group_suffix);
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", &config.brokers)
        .set("group.id", &group_id)
        .set("auto.offset.reset", "earliest")
        .set("enable.auto.commit", "true")
        .create()
        .map_err(|e| Error::Sensor(format!("failed to create sensor consumer: {e}")))?;

    rdkafka::consumer::Consumer::subscribe(&consumer, &[topic])
        .map_err(|e| Error::Sensor(format!("failed to subscribe to {topic}: {e}")))?;

    Ok(consumer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_producer_invalid_brokers_fails() {
        // We can't easily test success without a running Kafka broker,
        // but we can verify the config builder works.
        let config = KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        // Producer creation succeeds even with unreachable brokers
        // (connection is lazy).
        let result = create_producer(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn sensor_topics_config_valid() {
        // Verify sensor topic definitions are well-formed
        assert_eq!(SENSOR_TOPICS.len(), 8);
        for t in SENSOR_TOPICS {
            assert!(!t.topic.is_empty());
            assert!(t.partitions > 0);
            assert!(
                t.cleanup_policy == "delete" || t.cleanup_policy == "compact",
                "invalid cleanup_policy for {}: {}",
                t.topic,
                t.cleanup_policy
            );
        }
    }

    #[test]
    fn sensor_topics_include_stories_compacted() {
        let stories = SENSOR_TOPICS
            .iter()
            .find(|t| t.topic == "hb.stories")
            .expect("hb.stories topic should exist");
        assert_eq!(stories.cleanup_policy, "compact");
        assert_eq!(stories.retention_ms, -1);
    }

    #[tokio::test]
    async fn create_sensor_consumer_succeeds() {
        let config = KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        // Consumer creation is lazy — succeeds without running broker
        let result = create_sensor_consumer(&config, "hb.sensor.rss", "rss");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn create_consumer_succeeds_with_config() {
        let config = KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test-group".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        // Consumer creation and subscription succeeds even without
        // a running broker (connection is lazy).
        let result = create_commands_consumer(&config);
        assert!(result.is_ok());
    }

    // --- Kafka integration tests (require a running broker) ---
    //
    // Run with: cargo test -p heartbit kafka -- --ignored
    // Requires: Kafka broker at localhost:9092

    fn test_kafka_config(prefix: &str) -> KafkaConfig {
        let id = uuid::Uuid::new_v4();
        KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: format!("test-{prefix}-{id}"),
            commands_topic: format!("test.{prefix}.commands.{id}"),
            events_topic: format!("test.{prefix}.events.{id}"),
            dead_letter_topic: "test.dead-letter".into(),
        }
    }

    async fn cleanup_test_topics(config: &KafkaConfig) {
        let admin: AdminClient<DefaultClientContext> = ClientConfig::new()
            .set("bootstrap.servers", &config.brokers)
            .create()
            .expect("admin client for cleanup");
        let _ = admin
            .delete_topics(
                &[&config.commands_topic, &config.events_topic],
                &AdminOptions::new(),
            )
            .await;
    }

    fn test_consumer(config: &KafkaConfig) -> StreamConsumer {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("bootstrap.servers", &config.brokers)
            .set("group.id", &config.consumer_group)
            .set("auto.offset.reset", "earliest")
            .set("enable.auto.commit", "true")
            .create()
            .expect("create test consumer");
        rdkafka::consumer::Consumer::subscribe(&consumer, &[&config.commands_topic])
            .expect("subscribe to test topic");
        consumer
    }

    #[tokio::test]
    #[ignore] // Requires Kafka broker at localhost:9092
    async fn kafka_command_producer_round_trip() {
        use rdkafka::Message as _;

        let config = test_kafka_config("rt");
        ensure_topics(&config).await.expect("ensure test topics");

        // Send a command via KafkaCommandProducer
        let producer =
            KafkaCommandProducer::new(create_producer(&config).expect("create producer"));

        let id = uuid::Uuid::new_v4();
        let cmd = super::super::types::DaemonCommand::SubmitTask {
            id,
            task: "round trip test".into(),
            source: "test".into(),
            story_id: None,
        };
        let payload = serde_json::to_vec(&cmd).expect("serialize");

        producer
            .send_command(&config.commands_topic, &id.to_string(), &payload)
            .await
            .expect("send_command failed");

        // Consume it back and verify
        let consumer = test_consumer(&config);
        let msg = tokio::time::timeout(Duration::from_secs(10), consumer.recv())
            .await
            .expect("timed out waiting for message")
            .expect("consume error");

        let key = std::str::from_utf8(msg.key().expect("no key")).expect("key not utf8");
        assert_eq!(key, id.to_string());

        let received: super::super::types::DaemonCommand =
            serde_json::from_slice(msg.payload().expect("no payload")).expect("deserialize");
        match received {
            super::super::types::DaemonCommand::SubmitTask {
                id: recv_id,
                task,
                source,
                ..
            } => {
                assert_eq!(recv_id, id);
                assert_eq!(task, "round trip test");
                assert_eq!(source, "test");
            }
            other => panic!("unexpected command: {other:?}"),
        }

        cleanup_test_topics(&config).await;
    }

    #[tokio::test]
    #[ignore] // Requires Kafka broker at localhost:9092
    async fn kafka_pulse_scheduler_produces_command() {
        use rdkafka::Message as _;

        let config = test_kafka_config("pulse");
        ensure_topics(&config).await.expect("ensure test topics");

        let producer = std::sync::Arc::new(KafkaCommandProducer::new(
            create_producer(&config).expect("create producer"),
        ));

        // Create scheduler with 1-second interval
        let dir = tempfile::tempdir().unwrap();
        let pulse_config = crate::config::HeartbitPulseConfig {
            enabled: true,
            interval_seconds: 1,
            active_hours: None,
            prompt: None,
            idle_backoff_threshold: 6,
        };
        let scheduler = super::super::heartbit_pulse::HeartbitPulseScheduler::new(
            &pulse_config,
            dir.path(),
            producer,
            &config.commands_topic,
        )
        .expect("create scheduler");

        // Add a todo so the scheduler fires
        scheduler
            .todo_store()
            .add(super::super::todo::TodoEntry::new(
                "Kafka pulse test",
                "test",
            ))
            .expect("add todo");

        let cancel = tokio_util::sync::CancellationToken::new();
        let cancel2 = cancel.clone();
        tokio::spawn(async move { scheduler.run(cancel2).await });

        // Consume the command from Kafka
        let consumer = test_consumer(&config);
        let msg = tokio::time::timeout(Duration::from_secs(15), consumer.recv())
            .await
            .expect("scheduler did not produce a command within 15s")
            .expect("consume error");

        let received: super::super::types::DaemonCommand =
            serde_json::from_slice(msg.payload().expect("no payload")).expect("deserialize");
        match received {
            super::super::types::DaemonCommand::SubmitTask { source, task, .. } => {
                assert_eq!(source, "heartbit");
                assert!(task.contains("Kafka pulse test"), "task: {task}");
            }
            other => panic!("unexpected command: {other:?}"),
        }

        cancel.cancel();
        cleanup_test_topics(&config).await;
    }

    /// End-to-end: pulse scheduler → Kafka → DaemonCore → runner → store updated.
    #[tokio::test]
    #[ignore] // Requires Kafka broker at localhost:9092
    async fn kafka_daemon_processes_heartbit_pulse() {
        let config = test_kafka_config("e2e");
        ensure_topics(&config).await.expect("ensure test topics");

        let producer = create_producer(&config).expect("create producer");
        let consumer = create_commands_consumer(&config).expect("create consumer");

        let daemon_config = crate::config::DaemonConfig {
            kafka: config.clone(),
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
        };

        let store: std::sync::Arc<dyn super::super::store::TaskStore> =
            std::sync::Arc::new(super::super::store::InMemoryTaskStore::new());
        let cancel = tokio_util::sync::CancellationToken::new();

        let (core, _handle) = super::super::core::DaemonCore::new(
            &daemon_config,
            consumer,
            producer.clone(),
            store.clone(),
            cancel.clone(),
        );

        // Create pulse scheduler with 1-second interval
        let dir = tempfile::tempdir().unwrap();
        let pulse_config = crate::config::HeartbitPulseConfig {
            enabled: true,
            interval_seconds: 1,
            active_hours: None,
            prompt: None,
            idle_backoff_threshold: 6,
        };
        let pulse_producer = std::sync::Arc::new(KafkaCommandProducer::new(
            create_producer(&config).expect("create pulse producer"),
        ));
        let scheduler = super::super::heartbit_pulse::HeartbitPulseScheduler::new(
            &pulse_config,
            dir.path(),
            pulse_producer,
            &config.commands_topic,
        )
        .expect("create scheduler");

        // Add a todo so the pulse fires
        scheduler
            .todo_store()
            .add(super::super::todo::TodoEntry::new(
                "Deploy new API version",
                "user",
            ))
            .expect("add todo");

        // Channel to signal when the runner has been called
        let (runner_tx, mut runner_rx) = tokio::sync::mpsc::channel::<String>(1);

        // Mock build_runner: captures task text, returns a simple AgentOutput
        let build_runner = move |_id: uuid::Uuid,
                                 task: String,
                                 _source: String,
                                 _story_id: Option<String>,
                                 _on_event: std::sync::Arc<
            dyn Fn(crate::agent::events::AgentEvent) + Send + Sync,
        >| {
            let tx = runner_tx.clone();
            async move {
                let _ = tx.send(task).await;
                Ok(crate::agent::AgentOutput {
                    result: "HEARTBIT_OK".into(),
                    tool_calls_made: 0,
                    tokens_used: crate::llm::types::TokenUsage::default(),
                    structured: None,
                    estimated_cost_usd: None,
                })
            }
        };

        // Spawn daemon core (consumes from Kafka, calls runner)
        let core_handle = tokio::spawn(async move {
            let _ = core.run(build_runner).await;
        });

        // Spawn pulse scheduler (produces to Kafka after 1s)
        let pulse_cancel = cancel.clone();
        tokio::spawn(async move { scheduler.run(pulse_cancel).await });

        // Wait for the runner to receive the task
        let task_text = tokio::time::timeout(Duration::from_secs(15), runner_rx.recv())
            .await
            .expect("timed out waiting for daemon to process heartbit pulse")
            .expect("runner channel closed");

        // Verify the task text contains our todo
        assert!(
            task_text.contains("Deploy new API version"),
            "pulse prompt should contain the todo text, got: {task_text}"
        );
        assert!(
            task_text.contains("HEARTBIT PULSE mode"),
            "pulse prompt should contain the heartbit header, got: {task_text}"
        );

        // Poll store until the task is marked Completed
        let mut completed_task = None;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if let Ok((tasks, _)) = store.list(10, 0) {
                if let Some(t) = tasks.iter().find(|t| {
                    t.source == "heartbit" && t.state == super::super::types::TaskState::Completed
                }) {
                    completed_task = Some(t.clone());
                    break;
                }
            }
        }

        let task = completed_task.expect("heartbit task should reach Completed state");
        assert_eq!(task.result.as_deref(), Some("HEARTBIT_OK"));
        assert_eq!(task.source, "heartbit");

        // Cleanup
        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(5), core_handle).await;
        cleanup_test_topics(&config).await;
    }
}
