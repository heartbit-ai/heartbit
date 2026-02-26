use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rdkafka::Message as KafkaMessage;
use rdkafka::consumer::StreamConsumer;
use rdkafka::producer::FutureProducer;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::config::{KafkaConfig, SensorConfig, SensorSourceConfig};
use crate::daemon::kafka;
use crate::llm::DynLlmProvider;
use crate::sensor::metrics::SensorMetrics;
use crate::sensor::sources::audio::AudioSensor;
use crate::sensor::sources::image::ImageSensor;
use crate::sensor::sources::jmap::JmapEmailSensor;
use crate::sensor::sources::mcp::{McpSensor, McpTriageProcessor};
use crate::sensor::sources::rss::RssSensor;
use crate::sensor::sources::weather::WeatherSensor;
use crate::sensor::sources::webhook::WebhookSensor;
use crate::sensor::stories::StoryCorrelator;
use crate::sensor::triage::audio::AudioTriageProcessor;
use crate::sensor::triage::context::{TaskContext, TrustLevel};
use crate::sensor::triage::email::EmailTriageProcessor;
use crate::sensor::triage::image::ImageTriageProcessor;
use crate::sensor::triage::rss::RssTriageProcessor;
use crate::sensor::triage::structured::StructuredTriageProcessor;
use crate::sensor::triage::webhook::WebhookTriageProcessor;
use crate::sensor::triage::{TriageDecision, TriageProcessor};
use crate::sensor::{Sensor, SensorEvent};

/// Manages the lifecycle of all sensors, triage consumers, and story correlation.
///
/// `SensorManager` is the top-level coordinator for the sensor pipeline:
/// 1. Spawns sensor tasks that produce to per-modality Kafka topics.
/// 2. Spawns triage consumers that read from those topics and classify events.
/// 3. Correlates promoted events into stories.
/// 4. Forwards story-enriched commands to the main daemon topic.
pub struct SensorManager {
    config: SensorConfig,
    producer: FutureProducer,
    slm_provider: Arc<dyn DynLlmProvider>,
    metrics: Option<Arc<SensorMetrics>>,
    commands_topic: String,
    dead_letter_topic: String,
    /// Owner emails from daemon config for trust resolution.
    owner_emails: Vec<String>,
}

impl SensorManager {
    /// Create a new sensor manager.
    ///
    /// - `config`: Sensor layer configuration.
    /// - `producer`: Shared Kafka producer for all sensors.
    /// - `slm_provider`: LLM provider used for SLM-powered triage.
    /// - `metrics`: Optional Prometheus metrics.
    /// - `commands_topic`: The main daemon commands topic for promoted events.
    /// - `owner_emails`: Owner email addresses for trust resolution.
    pub fn new(
        config: SensorConfig,
        producer: FutureProducer,
        slm_provider: Arc<dyn DynLlmProvider>,
        metrics: Option<Arc<SensorMetrics>>,
        commands_topic: impl Into<String>,
        dead_letter_topic: impl Into<String>,
    ) -> Self {
        Self::with_owner_emails(
            config,
            producer,
            slm_provider,
            metrics,
            commands_topic,
            dead_letter_topic,
            vec![],
        )
    }

    /// Create a new sensor manager with owner emails for trust resolution.
    pub fn with_owner_emails(
        config: SensorConfig,
        producer: FutureProducer,
        slm_provider: Arc<dyn DynLlmProvider>,
        metrics: Option<Arc<SensorMetrics>>,
        commands_topic: impl Into<String>,
        dead_letter_topic: impl Into<String>,
        owner_emails: Vec<String>,
    ) -> Self {
        Self {
            config,
            producer,
            slm_provider,
            metrics,
            commands_topic: commands_topic.into(),
            dead_letter_topic: dead_letter_topic.into(),
            owner_emails,
        }
    }

    /// Run all sensors and triage consumers until cancellation.
    ///
    /// This spawns each sensor and triage consumer as independent tasks and
    /// blocks until `cancel` is triggered or all tasks complete.
    pub async fn run(
        &self,
        kafka_config: &KafkaConfig,
        cancel: CancellationToken,
    ) -> Result<(), Error> {
        if !self.config.enabled {
            tracing::info!("sensor layer disabled by config");
            return Ok(());
        }

        let mut tasks = JoinSet::new();

        // Build and spawn sensors
        let sensors = self.build_sensors();
        for sensor in sensors {
            let producer = self.producer.clone();
            let cancel = cancel.clone();
            let name = sensor.name().to_string();
            let topic = sensor.kafka_topic().to_string();

            tracing::info!(sensor = %name, topic = %topic, "starting sensor");

            tasks.spawn(async move {
                let result = sensor.run(producer, cancel).await;
                if let Err(ref e) = result {
                    tracing::error!(sensor = %name, error = %e, "sensor task failed");
                }
                (name, result)
            });
        }

        // Build and spawn triage consumers
        // Pair each processor with per-source sender lists for trust resolution.
        let triage_processors = self.build_triage_processors();
        let sender_lists: Vec<_> = self
            .config
            .sources
            .iter()
            .map(|s| {
                let (p, b) = s.sender_lists();
                (p.to_vec(), b.to_vec())
            })
            .collect();
        for (processor, (priority_senders, blocked_senders)) in
            triage_processors.into_iter().zip(sender_lists)
        {
            let topic = processor.source_topic().to_string();
            let consumer =
                kafka::create_sensor_consumer(kafka_config, &topic, &topic).map_err(|e| {
                    Error::Sensor(format!("failed to create consumer for {topic}: {e}"))
                })?;

            let cancel = cancel.clone();
            let metrics = self.metrics.clone();
            let producer = self.producer.clone();
            let commands_topic = self.commands_topic.clone();
            let dead_letter_topic = self.dead_letter_topic.clone();
            let owner_emails = self.owner_emails.clone();

            let correlation_window = self
                .config
                .stories
                .as_ref()
                .map(|s| Duration::from_secs(s.correlation_window_hours * 3600))
                .unwrap_or(Duration::from_secs(4 * 3600));
            let correlator = std::sync::Mutex::new(StoryCorrelator::new(correlation_window));

            tracing::info!(topic = %topic, "starting triage consumer");

            tasks.spawn(async move {
                run_triage_consumer(
                    consumer,
                    processor,
                    &correlator,
                    &producer,
                    &commands_topic,
                    &dead_letter_topic,
                    metrics.as_deref(),
                    cancel,
                    &owner_emails,
                    &priority_senders,
                    &blocked_senders,
                )
                .await
            });
        }

        // Wait for all tasks (they run until cancellation)
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok((name, Err(e))) => {
                    tracing::warn!(sensor = %name, error = %e, "sensor task exited with error");
                }
                Err(e) => {
                    tracing::error!(error = %e, "sensor task panicked");
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Default directory for sensor dedup state (`~/.heartbit/sensors`).
    fn default_sensor_state_dir() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(format!("{home}/.heartbit/sensors"))
    }

    /// Build sensor instances from config.
    fn build_sensors(&self) -> Vec<Box<dyn Sensor>> {
        let mut sensors: Vec<Box<dyn Sensor>> = Vec::new();

        for source in &self.config.sources {
            match source {
                SensorSourceConfig::JmapEmail {
                    name,
                    server,
                    username,
                    password_env,
                    priority_senders,
                    poll_interval_seconds,
                    ..
                } => {
                    let password = std::env::var(password_env).unwrap_or_default();
                    sensors.push(Box::new(JmapEmailSensor::new(
                        name.clone(),
                        server.clone(),
                        username.clone(),
                        password,
                        priority_senders.clone(),
                        Duration::from_secs(*poll_interval_seconds),
                    )));
                }
                SensorSourceConfig::Rss {
                    name,
                    feeds,
                    poll_interval_seconds,
                    ..
                } => {
                    sensors.push(Box::new(RssSensor::new(
                        name.clone(),
                        feeds.clone(),
                        Duration::from_secs(*poll_interval_seconds),
                    )));
                }
                SensorSourceConfig::Image {
                    name,
                    watch_directory,
                    poll_interval_seconds,
                } => {
                    sensors.push(Box::new(ImageSensor::new(
                        name.clone(),
                        watch_directory.clone(),
                        Duration::from_secs(*poll_interval_seconds),
                    )));
                }
                SensorSourceConfig::Audio {
                    name,
                    watch_directory,
                    whisper_model,
                    poll_interval_seconds,
                    ..
                } => {
                    sensors.push(Box::new(AudioSensor::new(
                        name.clone(),
                        watch_directory.clone(),
                        whisper_model.clone(),
                        Duration::from_secs(*poll_interval_seconds),
                    )));
                }
                SensorSourceConfig::Weather {
                    name,
                    api_key_env,
                    locations,
                    poll_interval_seconds,
                    alert_only,
                } => {
                    let api_key = std::env::var(api_key_env).unwrap_or_default();
                    sensors.push(Box::new(WeatherSensor::new(
                        name.clone(),
                        api_key,
                        locations.clone(),
                        Duration::from_secs(*poll_interval_seconds),
                        *alert_only,
                    )));
                }
                SensorSourceConfig::Webhook {
                    name,
                    path,
                    secret_env,
                } => {
                    let secret = secret_env.as_ref().and_then(|env| std::env::var(env).ok());
                    sensors.push(Box::new(WebhookSensor::new(
                        name.clone(),
                        path.clone(),
                        secret,
                    )));
                }
                SensorSourceConfig::Mcp {
                    name,
                    server,
                    tool_name,
                    tool_args,
                    kafka_topic,
                    modality,
                    poll_interval_seconds,
                    id_field,
                    content_field,
                    items_field,
                    enrich_tool,
                    enrich_id_param,
                    dedup_ttl_seconds,
                    ..
                } => {
                    let state_dir = Self::default_sensor_state_dir();
                    sensors.push(Box::new(McpSensor::new(
                        name.clone(),
                        (**server).clone(),
                        tool_name.clone(),
                        tool_args.clone(),
                        kafka_topic.clone(),
                        *modality,
                        Duration::from_secs(*poll_interval_seconds),
                        id_field.clone(),
                        content_field.clone(),
                        items_field.clone(),
                        enrich_tool.clone(),
                        enrich_id_param.clone(),
                        Duration::from_secs(*dedup_ttl_seconds),
                        Some(state_dir),
                    )));
                }
            }
        }

        sensors
    }

    /// Build triage processors from config.
    fn build_triage_processors(&self) -> Vec<Box<dyn TriageProcessor>> {
        let mut processors: Vec<Box<dyn TriageProcessor>> = Vec::new();

        for source in &self.config.sources {
            match source {
                SensorSourceConfig::JmapEmail {
                    priority_senders,
                    blocked_senders,
                    ..
                } => {
                    processors.push(Box::new(EmailTriageProcessor::new(
                        Arc::clone(&self.slm_provider),
                        priority_senders.clone(),
                        blocked_senders.clone(),
                    )));
                }
                SensorSourceConfig::Rss {
                    interest_keywords, ..
                } => {
                    processors.push(Box::new(RssTriageProcessor::new(
                        Arc::clone(&self.slm_provider),
                        interest_keywords.clone(),
                    )));
                }
                SensorSourceConfig::Image { .. } => {
                    processors.push(Box::new(ImageTriageProcessor::new(Arc::clone(
                        &self.slm_provider,
                    ))));
                }
                SensorSourceConfig::Audio { known_contacts, .. } => {
                    processors.push(Box::new(AudioTriageProcessor::new(
                        Arc::clone(&self.slm_provider),
                        known_contacts.clone(),
                    )));
                }
                SensorSourceConfig::Weather { .. } => {
                    processors.push(Box::new(StructuredTriageProcessor));
                }
                SensorSourceConfig::Webhook { .. } => {
                    processors.push(Box::new(WebhookTriageProcessor::new(Arc::clone(
                        &self.slm_provider,
                    ))));
                }
                SensorSourceConfig::Mcp {
                    kafka_topic,
                    modality,
                    priority_senders,
                    blocked_senders,
                    ..
                } => {
                    if kafka_topic == "hb.sensor.email" {
                        processors.push(Box::new(EmailTriageProcessor::new(
                            Arc::clone(&self.slm_provider),
                            priority_senders.clone(),
                            blocked_senders.clone(),
                        )));
                    } else {
                        processors.push(Box::new(McpTriageProcessor::new(
                            kafka_topic.clone(),
                            *modality,
                        )));
                    }
                }
            }
        }

        processors
    }
}

/// Run a single triage consumer loop: read events from a sensor topic,
/// run triage, correlate into stories, and forward promoted events to
/// the daemon commands topic.
///
/// Returns `(topic_name, result)` so the caller can identify which consumer
/// exited and whether it was clean or an error.
#[allow(clippy::too_many_arguments)]
async fn run_triage_consumer(
    consumer: StreamConsumer,
    processor: Box<dyn TriageProcessor>,
    correlator: &std::sync::Mutex<StoryCorrelator>,
    producer: &FutureProducer,
    commands_topic: &str,
    dead_letter_topic: &str,
    metrics: Option<&SensorMetrics>,
    cancel: CancellationToken,
    owner_emails: &[String],
    priority_senders: &[String],
    blocked_senders: &[String],
) -> (String, Result<(), Error>) {
    let topic = processor.source_topic().to_string();
    // In-memory dedup: event ID → first-seen time. Events within the TTL window are
    // dropped as duplicates. Periodically cleaned to avoid unbounded growth.
    let dedup_ttl = Duration::from_secs(2 * 3600); // 2 hours
    let mut seen: HashMap<String, Instant> = HashMap::new();
    let mut last_cleanup = Instant::now();
    let cleanup_interval = Duration::from_secs(300); // 5 minutes

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                return (topic, Ok(()));
            }
            msg_result = consumer.recv() => {
                let msg = match msg_result {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(topic = %topic, error = %e, "Kafka recv error");
                        continue;
                    }
                };

                let payload = match msg.payload() {
                    Some(p) => p,
                    None => continue,
                };

                let event: SensorEvent = match serde_json::from_slice(payload) {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::warn!(topic = %topic, error = %e, "failed to deserialize sensor event");
                        continue;
                    }
                };

                // Periodic cleanup of expired dedup entries.
                let now = Instant::now();
                if now.duration_since(last_cleanup) >= cleanup_interval {
                    seen.retain(|_, first_seen| now.duration_since(*first_seen) < dedup_ttl);
                    last_cleanup = now;
                }

                // Dedup: skip if we've seen this event ID within the TTL window.
                if let Some(first_seen) = seen.get(&event.id)
                    && now.duration_since(*first_seen) < dedup_ttl
                {
                    tracing::debug!(
                        topic = %topic,
                        event_id = %event.id,
                        "duplicate event skipped"
                    );
                    if let Some(m) = metrics {
                        m.record_event_dropped(&event.sensor_name);
                    }
                    continue;
                }
                seen.insert(event.id.clone(), now);

                if let Some(m) = metrics {
                    m.record_event_received(&event.sensor_name);
                }

                let start = std::time::Instant::now();
                let decision = match processor.process(&event).await {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::warn!(
                            sensor = %event.sensor_name,
                            error = %e,
                            "triage processing failed"
                        );
                        TriageDecision::DeadLetter {
                            error: e.to_string(),
                        }
                    }
                };
                let duration = start.elapsed().as_secs_f64();

                let decision_str = match &decision {
                    TriageDecision::Promote { .. } => "promote",
                    TriageDecision::Drop { .. } => "drop",
                    TriageDecision::DeadLetter { .. } => "dead_letter",
                };

                if let Some(m) = metrics {
                    m.record_triage_decision(&event.sensor_name, decision_str, duration);
                }

                match decision {
                    TriageDecision::Promote {
                        priority,
                        summary,
                        extracted_entities,
                        estimated_tokens: _,
                        action_categories,
                        action_hints,
                        has_attachments,
                        sender,
                        subject,
                        message_ref,
                    } => {
                        let entities: HashSet<String> =
                            extracted_entities.iter().cloned().collect();

                        let story_id = {
                            let mut corr = correlator.lock().unwrap_or_else(|e| e.into_inner());
                            let id = corr.correlate_with_links(
                                &event.id,
                                &event.sensor_name,
                                &summary,
                                &entities,
                                priority,
                                &event.related_ids,
                            );
                            if let Some(m) = metrics {
                                m.set_stories_active(corr.active_stories().len() as i64);
                            }
                            id
                        };

                        // Resolve sender trust level from config lists
                        let trust_level = TrustLevel::resolve(
                            sender.as_deref(),
                            owner_emails,
                            priority_senders,
                            blocked_senders,
                        );

                        // Build rich task context for the agent
                        let context = TaskContext {
                            summary: summary.clone(),
                            action_categories,
                            action_hints,
                            sender,
                            subject,
                            message_ref,
                            has_attachments,
                            entities: extracted_entities,
                            priority,
                            story_id: story_id.clone(),
                            sensor: event.sensor_name.clone(),
                            source_id: event.source_id.clone(),
                            trust_level,
                        };

                        // Build a proper DaemonCommand::SubmitTask
                        let cmd = crate::daemon::types::DaemonCommand::SubmitTask {
                            id: uuid::Uuid::new_v4(),
                            task: context.to_task_prompt(),
                            source: format!("sensor:{}", event.sensor_name),
                            story_id: Some(story_id.clone()),
                            trust_level: Some(context.trust_level),
                        };

                        let payload = match serde_json::to_vec(&cmd) {
                            Ok(p) => p,
                            Err(e) => {
                                tracing::error!(
                                    sensor = %event.sensor_name,
                                    error = %e,
                                    "failed to serialize promoted event, skipping"
                                );
                                continue;
                            }
                        };

                        use rdkafka::producer::FutureRecord;
                        if let Err((e, _)) = producer
                            .send(
                                FutureRecord::to(commands_topic)
                                    .key(&story_id)
                                    .payload(&payload),
                                rdkafka::util::Timeout::After(Duration::from_secs(5)),
                            )
                            .await
                        {
                            tracing::warn!(
                                story = %story_id,
                                error = %e,
                                "failed to produce promoted event to commands topic"
                            );
                        } else {
                            tracing::info!(
                                sensor = %event.sensor_name,
                                source_id = %event.source_id,
                                story = %story_id,
                                priority = ?priority,
                                "promoted sensor event to commands topic"
                            );
                        }
                    }
                    TriageDecision::Drop { reason } => {
                        tracing::info!(
                            sensor = %event.sensor_name,
                            source_id = %event.source_id,
                            reason = %reason,
                            "dropped sensor event"
                        );
                    }
                    TriageDecision::DeadLetter { error } => {
                        tracing::warn!(
                            sensor = %event.sensor_name,
                            error = %error,
                            "sensor event sent to dead letter"
                        );

                        let dl_payload = serde_json::json!({
                            "event": event,
                            "error": error,
                            "topic": topic,
                            "timestamp": chrono::Utc::now().to_rfc3339(),
                        });
                        if let Ok(bytes) = serde_json::to_vec(&dl_payload) {
                            use rdkafka::producer::FutureRecord;
                            if let Err((e, _)) = producer
                                .send(
                                    FutureRecord::to(dead_letter_topic)
                                        .key(&event.id)
                                        .payload(&bytes),
                                    rdkafka::util::Timeout::After(Duration::from_secs(5)),
                                )
                                .await
                            {
                                tracing::error!(
                                    error = %e,
                                    "failed to produce to dead-letter topic"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::DynLlmProvider;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };

    struct MockProvider;

    impl DynLlmProvider for MockProvider {
        fn complete<'a>(
            &'a self,
            _req: CompletionRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<CompletionResponse, Error>> + Send + 'a>,
        > {
            Box::pin(async {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text { text: "{}".into() }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            })
        }

        fn stream_complete<'a>(
            &'a self,
            _req: CompletionRequest,
            _on_text: &'a crate::llm::OnText,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<CompletionResponse, Error>> + Send + 'a>,
        > {
            Box::pin(async { Err(Error::Sensor("streaming not supported in mock".into())) })
        }

        fn model_name(&self) -> Option<&str> {
            Some("mock")
        }
    }

    fn make_manager(sources: Vec<SensorSourceConfig>) -> SensorManager {
        let config = SensorConfig {
            enabled: true,
            routing: None,
            salience: None,
            token_budget: None,
            stories: None,
            sources,
        };

        let producer = rdkafka::ClientConfig::new()
            .set("bootstrap.servers", "localhost:9092")
            .create::<FutureProducer>()
            .unwrap();

        SensorManager::new(
            config,
            producer,
            Arc::new(MockProvider),
            None,
            "heartbit.commands",
            "heartbit.dead-letter",
        )
    }

    #[test]
    fn build_sensors_from_rss_config() {
        let manager = make_manager(vec![SensorSourceConfig::Rss {
            name: "test_rss".into(),
            feeds: vec!["https://example.com/feed".into()],
            interest_keywords: vec!["rust".into()],
            poll_interval_seconds: 60,
        }]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 1);
        assert_eq!(sensors[0].name(), "test_rss");
        assert_eq!(sensors[0].kafka_topic(), "hb.sensor.rss");
    }

    #[test]
    fn build_sensors_rss_and_weather() {
        let manager = make_manager(vec![
            SensorSourceConfig::Rss {
                name: "rss1".into(),
                feeds: vec!["https://example.com/feed".into()],
                interest_keywords: vec![],
                poll_interval_seconds: 60,
            },
            SensorSourceConfig::Weather {
                name: "weather1".into(),
                api_key_env: "KEY".into(),
                locations: vec!["Paris,FR".into()],
                poll_interval_seconds: 1800,
                alert_only: true,
            },
        ]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 2, "both RSS and Weather should be built");
        assert_eq!(sensors[0].name(), "rss1");
        assert_eq!(sensors[1].name(), "weather1");
    }

    #[test]
    fn build_triage_processors_from_config() {
        let manager = make_manager(vec![SensorSourceConfig::Rss {
            name: "rss1".into(),
            feeds: vec!["https://example.com/feed".into()],
            interest_keywords: vec!["rust".into(), "ai".into()],
            poll_interval_seconds: 60,
        }]);

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 1);
        assert_eq!(processors[0].source_topic(), "hb.sensor.rss");
    }

    #[test]
    fn sensor_manager_disabled_builds_no_sensors() {
        let config = SensorConfig {
            enabled: false,
            routing: None,
            salience: None,
            token_budget: None,
            stories: None,
            sources: vec![],
        };

        let producer = rdkafka::ClientConfig::new()
            .set("bootstrap.servers", "localhost:9092")
            .create::<FutureProducer>()
            .unwrap();

        let manager = SensorManager::new(
            config,
            producer,
            Arc::new(MockProvider),
            None,
            "heartbit.commands",
            "heartbit.dead-letter",
        );

        assert!(manager.build_sensors().is_empty());
    }

    #[test]
    fn build_sensors_empty_config() {
        let manager = make_manager(vec![]);
        assert!(manager.build_sensors().is_empty());
    }

    #[test]
    fn build_triage_processors_empty_config() {
        let manager = make_manager(vec![]);
        assert!(manager.build_triage_processors().is_empty());
    }

    #[test]
    fn build_sensors_from_jmap_email_config() {
        // SAFETY: test-only; no concurrent access to this env var.
        unsafe { std::env::set_var("TEST_JMAP_PASS_MGR", "secret") };
        let manager = make_manager(vec![SensorSourceConfig::JmapEmail {
            name: "work_email".into(),
            server: "https://jmap.example.com".into(),
            username: "user@example.com".into(),
            password_env: "TEST_JMAP_PASS_MGR".into(),
            priority_senders: vec!["boss@company.com".into()],
            blocked_senders: vec![],
            poll_interval_seconds: 60,
        }]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 1);
        assert_eq!(sensors[0].name(), "work_email");
        assert_eq!(sensors[0].kafka_topic(), "hb.sensor.email");
        assert_eq!(sensors[0].modality(), crate::sensor::SensorModality::Text);
    }

    #[test]
    fn build_triage_processors_from_jmap_email_config() {
        let manager = make_manager(vec![SensorSourceConfig::JmapEmail {
            name: "work_email".into(),
            server: "https://jmap.example.com".into(),
            username: "user@example.com".into(),
            password_env: "TEST_JMAP_PASS".into(),
            priority_senders: vec!["boss@company.com".into()],
            blocked_senders: vec![],
            poll_interval_seconds: 60,
        }]);

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 1);
        assert_eq!(processors[0].source_topic(), "hb.sensor.email");
        assert_eq!(
            processors[0].modality(),
            crate::sensor::SensorModality::Text
        );
    }

    #[test]
    fn build_sensors_mixed_rss_and_email() {
        // SAFETY: test-only; no concurrent access to this env var.
        unsafe { std::env::set_var("TEST_JMAP_PASS_MIXED", "secret") };
        let manager = make_manager(vec![
            SensorSourceConfig::JmapEmail {
                name: "work_email".into(),
                server: "https://jmap.example.com".into(),
                username: "user@example.com".into(),
                password_env: "TEST_JMAP_PASS_MIXED".into(),
                priority_senders: vec![],
                blocked_senders: vec![],
                poll_interval_seconds: 60,
            },
            SensorSourceConfig::Rss {
                name: "tech_news".into(),
                feeds: vec!["https://example.com/feed".into()],
                interest_keywords: vec!["rust".into()],
                poll_interval_seconds: 900,
            },
        ]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 2);
        assert_eq!(sensors[0].name(), "work_email");
        assert_eq!(sensors[1].name(), "tech_news");

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 2);
        assert_eq!(processors[0].source_topic(), "hb.sensor.email");
        assert_eq!(processors[1].source_topic(), "hb.sensor.rss");
    }

    #[test]
    fn build_sensors_from_image_config() {
        let manager = make_manager(vec![SensorSourceConfig::Image {
            name: "doc_scanner".into(),
            watch_directory: "/tmp/inbox/images".into(),
            poll_interval_seconds: 30,
        }]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 1);
        assert_eq!(sensors[0].name(), "doc_scanner");
        assert_eq!(sensors[0].kafka_topic(), "hb.sensor.image");
        assert_eq!(sensors[0].modality(), crate::sensor::SensorModality::Image);
    }

    #[test]
    fn build_triage_processors_from_image_config() {
        let manager = make_manager(vec![SensorSourceConfig::Image {
            name: "doc_scanner".into(),
            watch_directory: "/tmp/inbox/images".into(),
            poll_interval_seconds: 30,
        }]);

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 1);
        assert_eq!(processors[0].source_topic(), "hb.sensor.image");
        assert_eq!(
            processors[0].modality(),
            crate::sensor::SensorModality::Image
        );
    }

    #[test]
    fn build_sensors_from_audio_config() {
        let manager = make_manager(vec![SensorSourceConfig::Audio {
            name: "voice_notes".into(),
            watch_directory: "/tmp/inbox/audio".into(),
            whisper_model: "base".into(),
            known_contacts: vec![],
            poll_interval_seconds: 30,
        }]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 1);
        assert_eq!(sensors[0].name(), "voice_notes");
        assert_eq!(sensors[0].kafka_topic(), "hb.sensor.audio");
        assert_eq!(sensors[0].modality(), crate::sensor::SensorModality::Audio);
    }

    #[test]
    fn build_triage_processors_from_audio_config() {
        let manager = make_manager(vec![SensorSourceConfig::Audio {
            name: "voice_notes".into(),
            watch_directory: "/tmp/inbox/audio".into(),
            whisper_model: "base".into(),
            known_contacts: vec![],
            poll_interval_seconds: 30,
        }]);

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 1);
        assert_eq!(processors[0].source_topic(), "hb.sensor.audio");
        assert_eq!(
            processors[0].modality(),
            crate::sensor::SensorModality::Audio
        );
    }

    #[test]
    fn build_sensors_from_weather_config() {
        // SAFETY: test-only; no concurrent access to this env var.
        unsafe { std::env::set_var("TEST_WEATHER_KEY", "test-key") };
        let manager = make_manager(vec![SensorSourceConfig::Weather {
            name: "local_weather".into(),
            api_key_env: "TEST_WEATHER_KEY".into(),
            locations: vec!["Paris,FR".into()],
            poll_interval_seconds: 1800,
            alert_only: true,
        }]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 1);
        assert_eq!(sensors[0].name(), "local_weather");
        assert_eq!(sensors[0].kafka_topic(), "hb.sensor.weather");
        assert_eq!(
            sensors[0].modality(),
            crate::sensor::SensorModality::Structured
        );
    }

    #[test]
    fn build_triage_processors_from_weather_config() {
        let manager = make_manager(vec![SensorSourceConfig::Weather {
            name: "local_weather".into(),
            api_key_env: "WEATHER_KEY".into(),
            locations: vec!["Paris,FR".into()],
            poll_interval_seconds: 1800,
            alert_only: false,
        }]);

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 1);
        assert_eq!(processors[0].source_topic(), "hb.sensor.weather");
        assert_eq!(
            processors[0].modality(),
            crate::sensor::SensorModality::Structured
        );
    }

    #[test]
    fn build_sensors_from_webhook_config() {
        let manager = make_manager(vec![SensorSourceConfig::Webhook {
            name: "github_events".into(),
            path: "/webhooks/github".into(),
            secret_env: None,
        }]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 1);
        assert_eq!(sensors[0].name(), "github_events");
        assert_eq!(sensors[0].kafka_topic(), "hb.sensor.webhook");
        assert_eq!(
            sensors[0].modality(),
            crate::sensor::SensorModality::Structured
        );
    }

    #[test]
    fn build_triage_processors_from_webhook_config() {
        let manager = make_manager(vec![SensorSourceConfig::Webhook {
            name: "github_events".into(),
            path: "/webhooks/github".into(),
            secret_env: None,
        }]);

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 1);
        assert_eq!(processors[0].source_topic(), "hb.sensor.webhook");
        assert_eq!(
            processors[0].modality(),
            crate::sensor::SensorModality::Structured
        );
    }

    #[test]
    fn build_sensors_all_seven_types() {
        // SAFETY: test-only; no concurrent access to these env vars.
        unsafe {
            std::env::set_var("TEST_JMAP_PASS_ALL7", "secret");
            std::env::set_var("TEST_WEATHER_KEY_ALL7", "key");
        }
        let manager = make_manager(vec![
            SensorSourceConfig::JmapEmail {
                name: "email".into(),
                server: "https://jmap.example.com".into(),
                username: "user@example.com".into(),
                password_env: "TEST_JMAP_PASS_ALL7".into(),
                priority_senders: vec![],
                blocked_senders: vec![],
                poll_interval_seconds: 60,
            },
            SensorSourceConfig::Rss {
                name: "rss".into(),
                feeds: vec!["https://example.com/feed".into()],
                interest_keywords: vec![],
                poll_interval_seconds: 900,
            },
            SensorSourceConfig::Image {
                name: "images".into(),
                watch_directory: "/tmp/images".into(),
                poll_interval_seconds: 30,
            },
            SensorSourceConfig::Audio {
                name: "audio".into(),
                watch_directory: "/tmp/audio".into(),
                whisper_model: "small".into(),
                known_contacts: vec![],
                poll_interval_seconds: 30,
            },
            SensorSourceConfig::Weather {
                name: "weather".into(),
                api_key_env: "TEST_WEATHER_KEY_ALL7".into(),
                locations: vec!["Paris,FR".into()],
                poll_interval_seconds: 1800,
                alert_only: true,
            },
            SensorSourceConfig::Webhook {
                name: "webhook".into(),
                path: "/webhooks/github".into(),
                secret_env: None,
            },
            SensorSourceConfig::Mcp {
                name: "mcp_calendar".into(),
                server: Box::new(crate::config::McpServerEntry::Simple(
                    "http://localhost:3000/mcp".into(),
                )),
                tool_name: "list_events".into(),
                tool_args: serde_json::json!({}),
                kafka_topic: "hb.sensor.calendar".into(),
                modality: crate::sensor::SensorModality::Structured,
                poll_interval_seconds: 300,
                id_field: "eventId".into(),
                content_field: Some("summary".into()),
                items_field: None,
                priority_senders: vec![],
                blocked_senders: vec![],
                enrich_tool: None,
                enrich_id_param: None,
                dedup_ttl_seconds: 604800,
            },
        ]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 7);
        assert_eq!(sensors[0].name(), "email");
        assert_eq!(sensors[1].name(), "rss");
        assert_eq!(sensors[2].name(), "images");
        assert_eq!(sensors[3].name(), "audio");
        assert_eq!(sensors[4].name(), "weather");
        assert_eq!(sensors[5].name(), "webhook");
        assert_eq!(sensors[6].name(), "mcp_calendar");

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 7);
        assert_eq!(processors[0].source_topic(), "hb.sensor.email");
        assert_eq!(processors[1].source_topic(), "hb.sensor.rss");
        assert_eq!(processors[2].source_topic(), "hb.sensor.image");
        assert_eq!(processors[3].source_topic(), "hb.sensor.audio");
        assert_eq!(processors[4].source_topic(), "hb.sensor.weather");
        assert_eq!(processors[5].source_topic(), "hb.sensor.webhook");
        assert_eq!(processors[6].source_topic(), "hb.sensor.calendar");
    }

    #[test]
    fn promoted_event_produces_valid_daemon_command() {
        // Verify the command format matches DaemonCommand::SubmitTask
        let cmd = crate::daemon::types::DaemonCommand::SubmitTask {
            id: uuid::Uuid::new_v4(),
            task: "[sensor:rss] Test summary\n\nStory: story-123\nPriority: High\nEstimated tokens: 500\nSource: feed-1".into(),
            source: "sensor:rss".into(),
            story_id: Some("story-123".into()),
            trust_level: None,
        };
        let payload = serde_json::to_vec(&cmd).unwrap();
        let parsed: crate::daemon::types::DaemonCommand = serde_json::from_slice(&payload).unwrap();
        match parsed {
            crate::daemon::types::DaemonCommand::SubmitTask { task, source, .. } => {
                assert!(task.contains("[sensor:rss]"));
                assert_eq!(source, "sensor:rss");
            }
            _ => panic!("expected SubmitTask"),
        }
    }

    #[test]
    fn promoted_event_includes_story_id() {
        let cmd = crate::daemon::types::DaemonCommand::SubmitTask {
            id: uuid::Uuid::new_v4(),
            task: "test task".into(),
            source: "sensor:rss".into(),
            story_id: Some("story-cve-2026-001".into()),
            trust_level: None,
        };
        let payload = serde_json::to_vec(&cmd).unwrap();
        let parsed: crate::daemon::types::DaemonCommand = serde_json::from_slice(&payload).unwrap();
        match parsed {
            crate::daemon::types::DaemonCommand::SubmitTask { story_id, .. } => {
                assert_eq!(story_id.as_deref(), Some("story-cve-2026-001"));
            }
            _ => panic!("expected SubmitTask"),
        }
    }

    // --- MCP sensor manager tests ---

    #[test]
    fn build_sensors_from_mcp_config() {
        let manager = make_manager(vec![SensorSourceConfig::Mcp {
            name: "gmail_inbox".into(),
            server: Box::new(crate::config::McpServerEntry::Simple(
                "http://localhost:3000/mcp".into(),
            )),
            tool_name: "search_emails".into(),
            tool_args: serde_json::json!({"query": "is:unread"}),
            kafka_topic: "hb.sensor.email".into(),
            modality: crate::sensor::SensorModality::Text,
            poll_interval_seconds: 60,
            id_field: "messageId".into(),
            content_field: Some("snippet".into()),
            items_field: None,
            priority_senders: vec!["boss@company.com".into()],
            blocked_senders: vec![],
            enrich_tool: None,
            enrich_id_param: None,
            dedup_ttl_seconds: 604800,
        }]);

        let sensors = manager.build_sensors();
        assert_eq!(sensors.len(), 1);
        assert_eq!(sensors[0].name(), "gmail_inbox");
        assert_eq!(sensors[0].kafka_topic(), "hb.sensor.email");
        assert_eq!(sensors[0].modality(), crate::sensor::SensorModality::Text);
    }

    #[test]
    fn build_triage_mcp_email_topic() {
        let manager = make_manager(vec![SensorSourceConfig::Mcp {
            name: "mcp_email".into(),
            server: Box::new(crate::config::McpServerEntry::Simple(
                "http://localhost:3000/mcp".into(),
            )),
            tool_name: "search_emails".into(),
            tool_args: serde_json::json!({}),
            kafka_topic: "hb.sensor.email".into(),
            modality: crate::sensor::SensorModality::Text,
            poll_interval_seconds: 60,
            id_field: "id".into(),
            content_field: None,
            items_field: None,
            priority_senders: vec!["boss@company.com".into()],
            blocked_senders: vec!["spam@example.com".into()],
            enrich_tool: None,
            enrich_id_param: None,
            dedup_ttl_seconds: 604800,
        }]);

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 1);
        // When kafka_topic is "hb.sensor.email", should use EmailTriageProcessor.
        assert_eq!(processors[0].source_topic(), "hb.sensor.email");
        assert_eq!(
            processors[0].modality(),
            crate::sensor::SensorModality::Text
        );
    }

    #[test]
    fn task_context_built_from_promote_decision_fields() {
        use crate::sensor::triage::{ActionCategory, Priority};

        // Simulate what the Promote arm in run_triage_consumer does:
        let summary = "Invoice from Acme Corp";
        let action_categories = vec![ActionCategory::PayOrProcess, ActionCategory::StoreOrFile];
        let action_hints = vec!["Download invoice PDF".to_string()];
        let sender = Some("billing@acme.com".to_string());
        let subject = Some("Invoice #2024-387".to_string());
        let message_ref = Some("19abc123".to_string());
        let has_attachments = true;
        let extracted_entities = vec!["Acme Corp".to_string(), "consulting".to_string()];
        let priority = Priority::High;

        let context = TaskContext {
            summary: summary.to_string(),
            action_categories,
            action_hints,
            sender,
            subject,
            message_ref,
            has_attachments,
            entities: extracted_entities,
            priority,
            story_id: "story-invoice-001".to_string(),
            sensor: "gmail_inbox".to_string(),
            source_id: "msg-19abc123@gmail.com".to_string(),
            trust_level: TrustLevel::default(),
        };

        let prompt = context.to_task_prompt();

        // Verify the prompt contains all structured sections
        assert!(prompt.contains("[sensor:gmail_inbox]"));
        assert!(prompt.contains("## Email Triage Summary"));
        assert!(prompt.contains("Invoice from Acme Corp"));
        assert!(prompt.contains("**From:** billing@acme.com"));
        assert!(prompt.contains("**Subject:** Invoice #2024-387"));
        assert!(prompt.contains("**Priority:** high"));
        assert!(prompt.contains("**Attachments:** Yes"));
        assert!(prompt.contains("pay_or_process"));
        assert!(prompt.contains("store_or_file"));
        assert!(prompt.contains("## Suggested Actions"));
        assert!(prompt.contains("- Download invoice PDF"));
        assert!(prompt.contains("## How to Access"));
        assert!(prompt.contains("`gmail_get_message`"));
        assert!(prompt.contains("`19abc123`"));
        assert!(prompt.contains("**Story:** story-invoice-001"));
        assert!(prompt.contains("**Source:** msg-19abc123@gmail.com"));
    }

    #[test]
    fn build_triage_mcp_other_topic() {
        let manager = make_manager(vec![SensorSourceConfig::Mcp {
            name: "calendar".into(),
            server: Box::new(crate::config::McpServerEntry::Simple(
                "http://localhost:3000/mcp".into(),
            )),
            tool_name: "list_events".into(),
            tool_args: serde_json::json!({}),
            kafka_topic: "hb.sensor.calendar".into(),
            modality: crate::sensor::SensorModality::Structured,
            poll_interval_seconds: 300,
            id_field: "eventId".into(),
            content_field: Some("summary".into()),
            items_field: None,
            priority_senders: vec![],
            blocked_senders: vec![],
            enrich_tool: None,
            enrich_id_param: None,
            dedup_ttl_seconds: 604800,
        }]);

        let processors = manager.build_triage_processors();
        assert_eq!(processors.len(), 1);
        // Non-email topics should use McpTriageProcessor with the configured topic.
        assert_eq!(processors[0].source_topic(), "hb.sensor.calendar");
        assert_eq!(
            processors[0].modality(),
            crate::sensor::SensorModality::Structured
        );
    }
}
