use std::sync::Arc;

use heartbit::KafkaConfig;
use heartbit::llm::DynLlmProvider;
use heartbit_sensors::{SensorConfig, SensorManager, SensorMetrics};
use rdkafka::producer::FutureProducer;
use tokio_util::sync::CancellationToken;

/// Start the sensor manager in a background task.
///
/// Returns the `SensorMetrics` handle (for Prometheus exposition) if sensors
/// were actually started, or `None` if the sensor layer is disabled/empty.
pub async fn start_sensors(
    sensor_config: &SensorConfig,
    producer: FutureProducer,
    slm_provider: Arc<dyn DynLlmProvider>,
    kafka_config: &KafkaConfig,
    commands_topic: &str,
    dead_letter_topic: &str,
    cancel: CancellationToken,
) -> anyhow::Result<Option<Arc<SensorMetrics>>> {
    if !sensor_config.enabled || sensor_config.sources.is_empty() {
        tracing::info!("sensor layer disabled or no sources configured");
        return Ok(None);
    }

    heartbit::daemon::kafka::ensure_sensor_topics(kafka_config)
        .await
        .map_err(|e| anyhow::anyhow!("ensure sensor topics: {e}"))?;

    let metrics = SensorMetrics::new().map_err(|e| anyhow::anyhow!("sensor metrics init: {e}"))?;
    let metrics = Arc::new(metrics);

    let manager = SensorManager::new(
        sensor_config.clone(),
        producer,
        slm_provider,
        Some(metrics.clone()),
        commands_topic,
        dead_letter_topic,
    );

    let sensor_kafka = kafka_config.clone();
    tokio::spawn(async move {
        if let Err(e) = manager.run(&sensor_kafka, cancel).await {
            tracing::error!(error = %e, "sensor manager failed");
        }
    });

    tracing::info!(
        sources = sensor_config.sources.len(),
        "sensor manager started"
    );
    Ok(Some(metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit::llm::types::{CompletionRequest, CompletionResponse};

    struct DummyProvider;

    impl DynLlmProvider for DummyProvider {
        fn complete<'a>(
            &'a self,
            _request: CompletionRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<CompletionResponse, heartbit::Error>>
                    + Send
                    + 'a,
            >,
        > {
            Box::pin(async { Err(heartbit::Error::Agent("dummy".into())) })
        }

        fn stream_complete<'a>(
            &'a self,
            _request: CompletionRequest,
            _on_text: &'a heartbit::OnText,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<CompletionResponse, heartbit::Error>>
                    + Send
                    + 'a,
            >,
        > {
            Box::pin(async { Err(heartbit::Error::Agent("dummy".into())) })
        }

        fn model_name(&self) -> Option<&str> {
            None
        }
    }

    #[tokio::test]
    async fn start_sensors_disabled_returns_none() {
        let config = SensorConfig {
            enabled: false,
            routing: None,
            salience: None,
            token_budget: None,
            stories: None,
            sources: vec![],
        };
        let kafka_config = KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        let fp = heartbit::daemon::kafka::create_producer(&kafka_config).unwrap();
        let provider: Arc<dyn DynLlmProvider> = Arc::new(DummyProvider);
        let cancel = CancellationToken::new();

        let result = start_sensors(
            &config,
            fp,
            provider,
            &kafka_config,
            "test.commands",
            "test.dead-letter",
            cancel,
        )
        .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn start_sensors_empty_sources_returns_none() {
        let config = SensorConfig {
            enabled: true,
            routing: None,
            salience: None,
            token_budget: None,
            stories: None,
            sources: vec![],
        };
        let kafka_config = KafkaConfig {
            brokers: "localhost:9092".into(),
            consumer_group: "test".into(),
            commands_topic: "test.commands".into(),
            events_topic: "test.events".into(),
            dead_letter_topic: "test.dead-letter".into(),
        };
        let fp = heartbit::daemon::kafka::create_producer(&kafka_config).unwrap();
        let provider: Arc<dyn DynLlmProvider> = Arc::new(DummyProvider);
        let cancel = CancellationToken::new();

        let result = start_sensors(
            &config,
            fp,
            provider,
            &kafka_config,
            "test.commands",
            "test.dead-letter",
            cancel,
        )
        .await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }
}
