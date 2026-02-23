use std::time::Duration;

use rdkafka::admin::{AdminClient, AdminOptions, NewTopic, TopicReplication};
use rdkafka::client::DefaultClientContext;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::stream_consumer::StreamConsumer;
use rdkafka::producer::FutureProducer;

use crate::Error;
use crate::config::KafkaConfig;

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
}
