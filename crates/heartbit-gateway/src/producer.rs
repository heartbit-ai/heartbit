use std::sync::Arc;

use heartbit::{CommandProducer, DaemonCommand, KafkaCommandProducer};

/// Thin wrapper around `KafkaCommandProducer` that serializes `DaemonCommand`
/// and sends it to a fixed Kafka topic.
pub struct GatewayProducer {
    inner: Arc<KafkaCommandProducer>,
    topic: String,
}

impl GatewayProducer {
    pub fn new(producer: KafkaCommandProducer, topic: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(producer),
            topic: topic.into(),
        }
    }

    /// Serialize and produce a `DaemonCommand` to the configured topic.
    #[allow(dead_code)] // Will be used by webhook endpoints in next phase
    pub async fn submit_task(&self, key: &str, command: &DaemonCommand) -> anyhow::Result<()> {
        let payload =
            serde_json::to_vec(command).map_err(|e| anyhow::anyhow!("serialize error: {e}"))?;
        self.inner
            .send_command(&self.topic, key, &payload)
            .await
            .map_err(|e| anyhow::anyhow!("kafka produce error: {e}"))
    }

    /// Access the underlying producer (e.g. for cron scheduler).
    pub fn inner(&self) -> &Arc<KafkaCommandProducer> {
        &self.inner
    }

    /// The Kafka topic this producer writes to.
    pub fn topic(&self) -> &str {
        &self.topic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_producer_accessors() {
        let kafka_config = heartbit::KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        let fp = heartbit::daemon::kafka::create_producer(&kafka_config).unwrap();
        let producer = GatewayProducer::new(KafkaCommandProducer::new(fp), "my-topic");

        assert_eq!(producer.topic(), "my-topic");
        // inner() returns an Arc
        let _arc = producer.inner().clone();
    }
}
