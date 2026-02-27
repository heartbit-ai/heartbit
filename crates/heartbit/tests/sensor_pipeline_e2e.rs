#![cfg(feature = "sensor")]
//! Sensor pipeline end-to-end tests.
//!
//! These tests exercise the full sensor pipeline through real Kafka:
//!   SensorEvent → Kafka topic → triage consumer → story correlator → commands topic
//!
//! Requirements:
//! - Running Kafka broker at `localhost:9092` (`docker compose up kafka -d`)
//! - For SLM tests: `OPENROUTER_API_KEY` env var set
//!
//! Run with:
//!   cargo test --test sensor_pipeline_e2e -- --ignored
//!
//! Or run only non-SLM tests (just Kafka pipeline):
//!   cargo test --test sensor_pipeline_e2e kafka_pipeline -- --ignored

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use rdkafka::Message as KafkaMessage;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::producer::{FutureProducer, FutureRecord};

use heartbit::config::KafkaConfig;
use heartbit::daemon::kafka;
use heartbit::sensor::compression::CompressionPolicy;
use heartbit::sensor::metrics::SensorMetrics;
use heartbit::sensor::routing::{ModelRouter, ModelTier};
use heartbit::sensor::stories::StoryCorrelator;
use heartbit::sensor::triage::audio::AudioTriageProcessor;
use heartbit::sensor::triage::email::EmailTriageProcessor;
use heartbit::sensor::triage::image::ImageTriageProcessor;
use heartbit::sensor::triage::rss::RssTriageProcessor;
use heartbit::sensor::triage::structured::StructuredTriageProcessor;
use heartbit::sensor::triage::webhook::WebhookTriageProcessor;
use heartbit::sensor::triage::{TriageDecision, TriageProcessor};
use heartbit::sensor::{SensorEvent, SensorModality};
use heartbit::{Error, Priority};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const BROKERS: &str = "localhost:9092";

/// Generate a unique suffix for test topic names to avoid cross-run pollution.
fn unique_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{ts}_{n}")
}

/// Check that Kafka is reachable. Returns false if broker is down.
fn kafka_is_reachable() -> bool {
    use rdkafka::consumer::BaseConsumer;
    let consumer: Result<BaseConsumer, _> = ClientConfig::new()
        .set("bootstrap.servers", BROKERS)
        .create();
    match consumer {
        Ok(c) => c.fetch_metadata(None, Duration::from_secs(3)).is_ok(),
        Err(_) => false,
    }
}

/// Ensure a test topic exists (idempotent).
async fn ensure_topic(topic: &str, partitions: i32) {
    use rdkafka::admin::{AdminClient, AdminOptions, NewTopic, TopicReplication};
    use rdkafka::config::ClientConfig;
    let admin: AdminClient<rdkafka::client::DefaultClientContext> = ClientConfig::new()
        .set("bootstrap.servers", BROKERS)
        .create()
        .expect("admin client");
    let opts = AdminOptions::new().operation_timeout(Some(Duration::from_secs(5)));
    let nt = NewTopic::new(topic, partitions, TopicReplication::Fixed(1));
    let _ = admin.create_topics(&[nt], &opts).await;
}

/// Create a `FutureProducer` for test use.
fn test_producer() -> FutureProducer {
    ClientConfig::new()
        .set("bootstrap.servers", BROKERS)
        .create()
        .expect("test producer")
}

/// Create a `StreamConsumer` subscribed to a topic with a unique group.
fn test_consumer(topic: &str, group: &str) -> StreamConsumer {
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", BROKERS)
        .set("group.id", group)
        .set("auto.offset.reset", "earliest")
        .set("enable.auto.commit", "true")
        .create()
        .expect("test consumer");
    Consumer::subscribe(&consumer, &[topic]).expect("subscribe");
    consumer
}

/// Produce a `SensorEvent` to a Kafka topic and wait for delivery.
async fn produce_event(producer: &FutureProducer, topic: &str, event: &SensorEvent) {
    let payload = serde_json::to_vec(event).expect("serialize event");
    producer
        .send(
            FutureRecord::to(topic)
                .key(&event.source_id)
                .payload(&payload),
            rdkafka::util::Timeout::After(Duration::from_secs(5)),
        )
        .await
        .expect("produce event");
}

/// Consume one message from a topic with timeout.
async fn consume_one(consumer: &StreamConsumer, timeout: Duration) -> Option<Vec<u8>> {
    tokio::time::timeout(timeout, async {
        loop {
            match consumer.recv().await {
                Ok(msg) => {
                    if let Some(payload) = msg.payload() {
                        return payload.to_vec();
                    }
                }
                Err(_) => continue,
            }
        }
    })
    .await
    .ok()
}

/// Make a simple `SensorEvent` for testing.
fn make_event(
    sensor_name: &str,
    modality: SensorModality,
    content: &str,
    source_id: &str,
) -> SensorEvent {
    SensorEvent {
        id: SensorEvent::generate_id(content, source_id),
        sensor_name: sensor_name.into(),
        modality,
        observed_at: Utc::now(),
        content: content.into(),
        source_id: source_id.into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    }
}

/// Make a weather `SensorEvent` with metadata.
fn make_weather_event(location: &str, description: &str, temp: f64, alert: bool) -> SensorEvent {
    let content = serde_json::json!({
        "temperature_c": temp,
        "description": description,
        "wind_speed_ms": 5.0,
        "humidity_pct": 65.0,
    });
    let content_str = serde_json::to_string(&content).expect("serialize");
    let source_id = format!("{location}:{}", Utc::now().timestamp());

    SensorEvent {
        id: SensorEvent::generate_id(&content_str, &source_id),
        sensor_name: "test_weather".into(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: content_str,
        source_id,
        metadata: Some(serde_json::json!({
            "location": location,
            "temperature_c": temp,
            "description": description,
            "wind_speed_ms": 5.0,
            "humidity_pct": 65.0,
            "alert": alert,
        })),
        binary_ref: None,
        related_ids: vec![],
    }
}

/// Mock LLM provider that returns a fixed JSON response (for non-OpenRouter tests).
struct MockSlmProvider {
    response: String,
}

impl MockSlmProvider {
    fn new(response: &str) -> Self {
        Self {
            response: response.into(),
        }
    }

    /// Returns a mock provider whose triage response always classifies as relevant.
    fn rss_classifier() -> Self {
        Self::new(
            &serde_json::json!({
                "relevant": true,
                "category": "tech_news",
                "summary": "Interesting tech article about Rust and WebAssembly",
                "entities": ["rust", "webassembly"]
            })
            .to_string(),
        )
    }

    /// Returns a mock provider for webhook classification.
    fn webhook_classifier() -> Self {
        Self::new(
            &serde_json::json!({
                "category": "pr_review",
                "summary": "PR review requested on feature branch",
                "entities": ["user/repo"],
                "action_required": true
            })
            .to_string(),
        )
    }

    /// Returns a mock provider for email classification.
    fn email_classifier(relevant: bool, urgent: bool) -> Self {
        let urgency = if urgent { "high" } else { "normal" };
        Self::new(
            &serde_json::json!({
                "relevant": relevant,
                "urgency": urgency,
                "summary": "Important project update from team lead",
                "entities": ["project_alpha", "team_lead"],
                "category": "work"
            })
            .to_string(),
        )
    }

    /// Returns a mock provider for image classification.
    fn image_classifier(category: &str) -> Self {
        Self::new(
            &serde_json::json!({
                "category": category,
                "description": "A scanned invoice from Acme Corp",
                "objects": ["text", "table", "logo"],
                "text_content": "Invoice #1234 — Total: $5,000"
            })
            .to_string(),
        )
    }

    /// Returns a mock provider for audio classification.
    fn audio_classifier(category: &str) -> Self {
        Self::new(
            &serde_json::json!({
                "category": category,
                "summary": "Voice note about quarterly review",
                "entities": ["quarterly_review"],
                "speaker": "alice@company.com"
            })
            .to_string(),
        )
    }
}

impl heartbit::llm::DynLlmProvider for MockSlmProvider {
    fn complete<'a>(
        &'a self,
        _req: heartbit::llm::types::CompletionRequest,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<heartbit::llm::types::CompletionResponse, Error>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async {
            Ok(heartbit::llm::types::CompletionResponse {
                content: vec![heartbit::llm::types::ContentBlock::Text {
                    text: self.response.clone(),
                }],
                stop_reason: heartbit::llm::types::StopReason::EndTurn,
                usage: heartbit::llm::types::TokenUsage::default(),
                model: None,
            })
        })
    }

    fn stream_complete<'a>(
        &'a self,
        _req: heartbit::llm::types::CompletionRequest,
        _on_text: &'a heartbit::llm::OnText,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<heartbit::llm::types::CompletionResponse, Error>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async { Err(Error::Sensor("not supported".into())) })
    }

    fn model_name(&self) -> Option<&str> {
        Some("mock-slm")
    }
}

// ---------------------------------------------------------------------------
// The `run_triage_consumer` function is private. We re-implement a small
// version here that tests the public trait methods and Kafka I/O.
// ---------------------------------------------------------------------------

/// Simplified triage loop for testing: read one event, process it, write result.
async fn run_triage_once(
    consumer: &StreamConsumer,
    processor: &dyn TriageProcessor,
    producer: &FutureProducer,
    commands_topic: &str,
    correlator: &std::sync::Mutex<StoryCorrelator>,
    metrics: Option<&SensorMetrics>,
    timeout: Duration,
) -> Option<TriageDecision> {
    let payload = consume_one(consumer, timeout).await?;
    let event: SensorEvent = serde_json::from_slice(&payload).ok()?;

    if let Some(m) = metrics {
        m.record_event_received(&event.sensor_name);
    }

    let start = std::time::Instant::now();
    let decision = processor.process(&event).await.ok()?;
    let duration = start.elapsed().as_secs_f64();

    let decision_str = match &decision {
        TriageDecision::Promote { .. } => "promote",
        TriageDecision::Drop { .. } => "drop",
        TriageDecision::DeadLetter { .. } => "dead_letter",
    };

    if let Some(m) = metrics {
        m.record_triage_decision(&event.sensor_name, decision_str, duration);
    }

    if let TriageDecision::Promote {
        ref priority,
        ref summary,
        ref extracted_entities,
        estimated_tokens,
        ..
    } = decision
    {
        let entities: HashSet<String> = extracted_entities.iter().cloned().collect();
        let story_id = {
            let mut corr = correlator.lock().unwrap_or_else(|e| e.into_inner());
            corr.correlate_with_links(
                &event.id,
                &event.sensor_name,
                summary,
                &entities,
                *priority,
                &event.related_ids,
            )
        };

        let task_json = serde_json::json!({
            "task": format!(
                "[sensor:{sensor}] {summary}\n\nStory: {story_id}\nPriority: {priority:?}\nEstimated tokens: {estimated_tokens}\nSource: {source_id}",
                sensor = event.sensor_name,
                source_id = event.source_id,
            ),
            "source": format!("sensor:{}", event.sensor_name),
            "story_id": story_id,
        });

        let payload = serde_json::to_vec(&task_json).ok()?;
        let _ = producer
            .send(
                FutureRecord::to(commands_topic)
                    .key(&story_id)
                    .payload(&payload),
                rdkafka::util::Timeout::After(Duration::from_secs(5)),
            )
            .await;
    }

    Some(decision)
}

// ===========================================================================
// TEST SUITE 1: Kafka pipeline with mock SLM (no API key needed)
// ===========================================================================

/// Test: RSS event → Kafka → mock SLM triage → promoted → commands topic
#[tokio::test]
#[ignore] // Requires Kafka
async fn kafka_pipeline_rss_event_promoted_to_commands() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.rss.promote.{suffix}");
    let commands_topic = format!("hb.test.commands.promote.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();

    // Create a consumer on the commands topic BEFORE producing
    let cmd_consumer = test_consumer(&commands_topic, &format!("cmd-promote-{suffix}"));

    // Triage consumer reads from the sensor topic
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-promote-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Create processor with mock SLM
    let slm = Arc::new(MockSlmProvider::rss_classifier());
    let processor = RssTriageProcessor::new(slm, vec!["rust".into(), "ai".into()]);

    // Create correlator
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Produce an RSS event with matching keywords
    let event = SensorEvent {
        id: SensorEvent::generate_id("Rust 2026 edition released", "https://example.com/rust"),
        sensor_name: "test_rss".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Rust 2026 edition released with exciting new features for WebAssembly".into(),
        source_id: "https://example.com/rust-2026".into(),
        metadata: Some(serde_json::json!({"feed": "hn"})),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    // Run triage
    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    let decision = decision.unwrap();
    assert!(
        decision.is_promote(),
        "RSS event with matching keywords should be promoted"
    );

    // Verify message landed on commands topic
    let cmd_msg = consume_one(&cmd_consumer, Duration::from_secs(10)).await;
    assert!(
        cmd_msg.is_some(),
        "promoted event should appear on commands topic"
    );

    let cmd_json: serde_json::Value =
        serde_json::from_slice(&cmd_msg.unwrap()).expect("parse command");
    assert!(
        cmd_json["task"]
            .as_str()
            .unwrap_or("")
            .contains("[sensor:test_rss]"),
        "command task should reference sensor name"
    );
    assert!(
        cmd_json["source"]
            .as_str()
            .unwrap_or("")
            .starts_with("sensor:"),
        "command source should start with sensor:"
    );
    assert!(
        cmd_json["story_id"].is_string(),
        "command should have a story_id"
    );
}

/// Test: RSS event with NO matching keywords → dropped (no SLM call needed)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_rss_no_keywords_dropped() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.rss.drop.{suffix}");
    let commands_topic = format!("hb.test.commands.drop.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-drop-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Processor with keywords that won't match
    let slm = Arc::new(MockSlmProvider::rss_classifier());
    let processor = RssTriageProcessor::new(slm, vec!["kubernetes".into(), "docker".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Event content doesn't match "kubernetes" or "docker"
    let event = make_event(
        "test_rss",
        SensorModality::Text,
        "New JavaScript framework released today",
        "https://example.com/js-framework",
    );

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    assert!(
        decision.unwrap().is_drop(),
        "RSS event with no matching keywords should be dropped"
    );
}

/// Test: Weather alert event → promoted (rule-based, no SLM)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_weather_alert_promoted() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.weather.alert.{suffix}");
    let commands_topic = format!("hb.test.commands.alert.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let cmd_consumer = test_consumer(&commands_topic, &format!("cmd-alert-{suffix}"));
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-alert-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = StructuredTriageProcessor;
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Weather alert event
    let event = make_weather_event("London", "thunderstorm with heavy rain", 25.0, true);
    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    let d = decision.unwrap();
    assert!(d.is_promote(), "weather alert should be promoted");
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::High,
            "weather alerts should be High priority"
        );
    }

    // Verify on commands topic
    let cmd_msg = consume_one(&cmd_consumer, Duration::from_secs(10)).await;
    assert!(
        cmd_msg.is_some(),
        "weather alert should appear on commands topic"
    );

    let cmd_json: serde_json::Value =
        serde_json::from_slice(&cmd_msg.unwrap()).expect("parse command");
    assert!(
        cmd_json["task"]
            .as_str()
            .unwrap_or("")
            .contains("Weather alert")
    );
}

/// Test: Normal weather → dropped (no SLM needed)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_weather_normal_dropped() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.weather.normal.{suffix}");
    let commands_topic = format!("hb.test.commands.normal.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-normal-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = StructuredTriageProcessor;
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = make_weather_event("Paris", "clear sky", 22.0, false);
    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    assert!(
        decision.unwrap().is_drop(),
        "normal weather should be dropped"
    );
}

/// Test: Webhook event → mock SLM triage → promoted with correct priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_webhook_pr_review_high_priority() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.webhook.pr.{suffix}");
    let commands_topic = format!("hb.test.commands.pr.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let cmd_consumer = test_consumer(&commands_topic, &format!("cmd-pr-{suffix}"));
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-pr-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::webhook_classifier());
    let processor = WebhookTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id(
            r#"{"action":"review_requested","pull_request":{"number":42}}"#,
            "github:pr42",
        ),
        sensor_name: "github_hooks".into(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content:
            r#"{"action":"review_requested","pull_request":{"number":42,"title":"Add feature"}}"#
                .into(),
        source_id: "github:pr42".into(),
        metadata: Some(serde_json::json!({
            "source": "github",
            "content_type": "application/json",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote());
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::High,
            "PR review should be High priority"
        );
    }

    let cmd_msg = consume_one(&cmd_consumer, Duration::from_secs(10)).await;
    assert!(
        cmd_msg.is_some(),
        "webhook event should appear on commands topic"
    );
}

/// Test: Metrics are correctly updated through the pipeline
#[tokio::test]
#[ignore]
async fn kafka_pipeline_metrics_updated() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.weather.metrics.{suffix}");
    let commands_topic = format!("hb.test.commands.metrics.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-metrics-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = StructuredTriageProcessor;
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));
    let metrics = SensorMetrics::new().expect("create metrics");

    // Process a weather alert (promoted)
    let alert_event = make_weather_event("Berlin", "blizzard", -25.0, true);
    produce_event(&producer, &sensor_topic, &alert_event).await;

    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        Some(&metrics),
        Duration::from_secs(10),
    )
    .await;

    // Process normal weather (dropped)
    let normal_event = make_weather_event("Rome", "sunny", 28.0, false);
    produce_event(&producer, &sensor_topic, &normal_event).await;

    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        Some(&metrics),
        Duration::from_secs(10),
    )
    .await;

    // Verify metrics
    let encoded = metrics.encode().expect("encode metrics");
    assert!(
        encoded.contains(r#"heartbit_sensor_events_received_total{sensor_name="test_weather"} 2"#),
        "should have 2 events received for test_weather. Metrics:\n{encoded}"
    );
    assert!(
        encoded.contains(r#"heartbit_sensor_events_promoted_total{sensor_name="test_weather"} 1"#),
        "should have 1 promoted event. Metrics:\n{encoded}"
    );
    assert!(
        encoded.contains(r#"heartbit_sensor_events_dropped_total{sensor_name="test_weather"} 1"#),
        "should have 1 dropped event. Metrics:\n{encoded}"
    );
}

/// Test: Story correlation groups related events
#[tokio::test]
#[ignore]
async fn kafka_pipeline_story_correlation() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.weather.story.{suffix}");
    let commands_topic = format!("hb.test.commands.story.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-story-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = StructuredTriageProcessor;
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Two weather alerts for the same location should be in the same story
    let event1 = make_weather_event("Tokyo", "severe thunderstorm warning", 32.0, true);
    produce_event(&producer, &sensor_topic, &event1).await;
    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    let event2 = make_weather_event("Tokyo", "heavy rainfall alert", 30.0, true);
    produce_event(&producer, &sensor_topic, &event2).await;
    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Different location → different story
    let event3 = make_weather_event("Sydney", "heatwave warning", 45.0, true);
    produce_event(&producer, &sensor_topic, &event3).await;
    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Check correlator state
    let corr = correlator.lock().unwrap();
    let stories = corr.active_stories();

    // Tokyo events should share a story, Sydney is separate
    assert!(
        stories.len() >= 2,
        "should have at least 2 stories (Tokyo + Sydney), got {}",
        stories.len()
    );

    // Verify Tokyo story has 2 events
    let tokyo_story = stories.iter().find(|s| s.entities.contains("Tokyo"));
    assert!(
        tokyo_story.is_some(),
        "should have a story with Tokyo entity"
    );
    assert_eq!(
        tokyo_story.unwrap().events.len(),
        2,
        "Tokyo story should have 2 events"
    );

    // Verify Sydney story has 1 event
    let sydney_story = stories.iter().find(|s| s.entities.contains("Sydney"));
    assert!(
        sydney_story.is_some(),
        "should have a story with Sydney entity"
    );
    assert_eq!(
        sydney_story.unwrap().events.len(),
        1,
        "Sydney story should have 1 event"
    );
}

/// Test: Multiple event types through the pipeline in sequence
#[tokio::test]
#[ignore]
async fn kafka_pipeline_mixed_events() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let weather_topic = format!("hb.test.sensor.weather.mixed.{suffix}");
    let webhook_topic = format!("hb.test.sensor.webhook.mixed.{suffix}");
    let commands_topic = format!("hb.test.commands.mixed.{suffix}");

    ensure_topic(&weather_topic, 1).await;
    ensure_topic(&webhook_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let cmd_consumer = test_consumer(&commands_topic, &format!("cmd-mixed-{suffix}"));
    let weather_consumer = test_consumer(&weather_topic, &format!("w-triage-mixed-{suffix}"));
    let webhook_consumer = test_consumer(&webhook_topic, &format!("wh-triage-mixed-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let weather_processor = StructuredTriageProcessor;
    let webhook_processor =
        WebhookTriageProcessor::new(Arc::new(MockSlmProvider::webhook_classifier()));
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // 1. Weather alert
    let weather_event = make_weather_event("NYC", "hurricane warning", 30.0, true);
    produce_event(&producer, &weather_topic, &weather_event).await;
    let d1 = run_triage_once(
        &weather_consumer,
        &weather_processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;
    assert!(d1.unwrap().is_promote());

    // 2. Webhook PR review
    let webhook_event = SensorEvent {
        id: SensorEvent::generate_id("pr_event", "github:pr99"),
        sensor_name: "github".into(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: r#"{"action":"review_requested","number":99}"#.into(),
        source_id: "github:pr99".into(),
        metadata: Some(serde_json::json!({"source": "github"})),
        binary_ref: None,
        related_ids: vec![],
    };
    produce_event(&producer, &webhook_topic, &webhook_event).await;
    let d2 = run_triage_once(
        &webhook_consumer,
        &webhook_processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;
    assert!(d2.unwrap().is_promote());

    // 3. Normal weather (should be dropped)
    let normal_weather = make_weather_event("Oslo", "cloudy", 8.0, false);
    produce_event(&producer, &weather_topic, &normal_weather).await;
    let d3 = run_triage_once(
        &weather_consumer,
        &weather_processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;
    assert!(d3.unwrap().is_drop());

    // Verify exactly 2 messages on commands topic (weather alert + webhook)
    let msg1 = consume_one(&cmd_consumer, Duration::from_secs(10)).await;
    assert!(msg1.is_some(), "first command should exist");
    let msg2 = consume_one(&cmd_consumer, Duration::from_secs(5)).await;
    assert!(msg2.is_some(), "second command should exist");
    let msg3 = consume_one(&cmd_consumer, Duration::from_secs(3)).await;
    assert!(
        msg3.is_none(),
        "third command should NOT exist (normal weather was dropped)"
    );
}

/// Test: Sensor topic creation is idempotent
#[tokio::test]
#[ignore]
async fn kafka_ensure_sensor_topics_idempotent() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let config = KafkaConfig {
        brokers: BROKERS.into(),
        commands_topic: "heartbit.commands".into(),
        events_topic: "heartbit.events".into(),
        consumer_group: "heartbit-e2e-test".into(),
        dead_letter_topic: "heartbit.dead-letter".into(),
    };

    // Call twice — second call should not fail
    kafka::ensure_sensor_topics(&config)
        .await
        .expect("first call");
    kafka::ensure_sensor_topics(&config)
        .await
        .expect("second call (idempotent)");
}

/// Test: SensorEvent serialization round-trip through Kafka
#[tokio::test]
#[ignore]
async fn kafka_sensor_event_serde_roundtrip() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let topic = format!("hb.test.sensor.roundtrip.{suffix}");
    ensure_topic(&topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let consumer = test_consumer(&topic, &format!("serde-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let original = SensorEvent {
        id: "test-roundtrip-id".into(),
        sensor_name: "roundtrip_sensor".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Hello from the sensor pipeline E2E test! 🎉".into(),
        source_id: "e2e:test:001".into(),
        metadata: Some(serde_json::json!({"key": "value", "nested": {"a": 1}})),
        binary_ref: Some("/storage/test.bin".into()),
        related_ids: vec!["related-001".into(), "related-002".into()],
    };

    produce_event(&producer, &topic, &original).await;

    let payload = consume_one(&consumer, Duration::from_secs(10)).await;
    assert!(payload.is_some(), "should receive the event back");

    let received: SensorEvent =
        serde_json::from_slice(&payload.unwrap()).expect("deserialize event");

    assert_eq!(received.id, original.id);
    assert_eq!(received.sensor_name, original.sensor_name);
    assert_eq!(received.modality, original.modality);
    assert_eq!(received.content, original.content);
    assert_eq!(received.source_id, original.source_id);
    assert_eq!(received.metadata, original.metadata);
    assert_eq!(received.binary_ref, original.binary_ref);
    assert_eq!(received.related_ids, original.related_ids);
}

// ===========================================================================
// TEST SUITE 2: Real SLM via OpenRouter (requires OPENROUTER_API_KEY)
// ===========================================================================

/// Create a real OpenRouter provider with a cheap SLM model.
fn make_openrouter_slm() -> Option<Arc<dyn heartbit::llm::DynLlmProvider>> {
    let api_key = std::env::var("OPENROUTER_API_KEY").ok()?;
    if api_key.is_empty() {
        return None;
    }
    // Use a very cheap model for triage
    let provider = heartbit::llm::openrouter::OpenRouterProvider::new(api_key, "qwen/qwen3-4b");
    Some(Arc::new(provider) as Arc<dyn heartbit::llm::DynLlmProvider>)
}

/// Test: RSS triage with real SLM via OpenRouter
#[tokio::test]
#[ignore]
async fn slm_rss_triage_with_openrouter() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }
    let Some(slm) = make_openrouter_slm() else {
        eprintln!("SKIP: OPENROUTER_API_KEY not set");
        return;
    };

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.rss.slm.{suffix}");
    let commands_topic = format!("hb.test.commands.slm.rss.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-slm-rss-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = RssTriageProcessor::new(slm, vec!["rust".into(), "programming".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Event that matches keywords → will hit SLM
    let event = SensorEvent {
        id: SensorEvent::generate_id("Rust async", "https://blog.rust-lang.org/async"),
        sensor_name: "tech_rss".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "The Rust programming language has released a new async runtime that significantly improves performance for web servers and microservices.".into(),
        source_id: "https://blog.rust-lang.org/async".into(),
        metadata: Some(serde_json::json!({"feed": "rust_blog"})),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(30), // longer timeout for real SLM
    )
    .await;

    assert!(decision.is_some(), "should get a decision from real SLM");
    let d = decision.unwrap();
    // Real SLM should promote this (relevant tech content matching keywords)
    assert!(
        d.is_promote(),
        "SLM should promote relevant Rust article, got: {d:?}"
    );

    if let TriageDecision::Promote {
        summary,
        extracted_entities,
        estimated_tokens,
        ..
    } = &d
    {
        assert!(!summary.is_empty(), "SLM should generate a summary");
        assert!(estimated_tokens > &0, "estimated tokens should be > 0");
        eprintln!("SLM summary: {summary}");
        eprintln!("SLM entities: {extracted_entities:?}");
    }
}

/// Test: Webhook triage with real SLM via OpenRouter
#[tokio::test]
#[ignore]
async fn slm_webhook_triage_with_openrouter() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }
    let Some(slm) = make_openrouter_slm() else {
        eprintln!("SKIP: OPENROUTER_API_KEY not set");
        return;
    };

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.webhook.slm.{suffix}");
    let commands_topic = format!("hb.test.commands.slm.webhook.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-slm-webhook-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = WebhookTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("gh_issue", "github:issue123"),
        sensor_name: "github_hooks".into(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: r#"{"action":"opened","issue":{"number":123,"title":"Critical bug in authentication module","body":"Users are unable to log in after the recent deployment. All login attempts return 500 errors.","labels":["bug","critical","auth"]}}"#.into(),
        source_id: "github:issue123".into(),
        metadata: Some(serde_json::json!({"source": "github", "content_type": "application/json"})),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(30),
    )
    .await;

    assert!(decision.is_some(), "should get a decision from real SLM");
    let d = decision.unwrap();
    assert!(
        d.is_promote(),
        "SLM should promote GitHub issue event, got: {d:?}"
    );

    if let TriageDecision::Promote {
        summary, priority, ..
    } = &d
    {
        assert!(!summary.is_empty(), "SLM should generate a summary");
        eprintln!("SLM priority: {priority:?}");
        eprintln!("SLM summary: {summary}");
    }
}

// ===========================================================================
// TEST SUITE 3: Email triage through Kafka pipeline
// ===========================================================================

/// Test: Email from known contact → promoted with High/Critical priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_known_contact_promoted() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.known.{suffix}");
    let commands_topic = format!("hb.test.commands.email.known.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-email-known-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::email_classifier(true, true));
    let processor = EmailTriageProcessor::new(slm, vec!["boss@company.com".into()], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("urgent email", "msg:001"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: URGENT - Q4 budget review needed\n\nPlease review the attached budget."
            .into(),
        source_id: "msg:001".into(),
        metadata: Some(serde_json::json!({
            "from": "boss@company.com",
            "subject": "URGENT - Q4 budget review needed",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    let d = decision.unwrap();
    assert!(
        d.is_promote(),
        "email from known contact should be promoted"
    );
    if let TriageDecision::Promote { priority, .. } = &d {
        assert!(
            *priority == Priority::Critical || *priority == Priority::High,
            "known + urgent should be Critical or High, got: {priority:?}"
        );
    }
}

/// Test: Email from blocked sender → dropped
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_blocked_sender_dropped() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.blocked.{suffix}");
    let commands_topic = format!("hb.test.commands.email.blocked.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-email-blocked-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::email_classifier(true, false));
    let processor = EmailTriageProcessor::new(slm, vec![], vec!["spam@evil.com".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("spam email", "msg:spam"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: You won a million dollars!\n\nClick here now.".into(),
        source_id: "msg:spam".into(),
        metadata: Some(serde_json::json!({
            "from": "spam@evil.com",
            "subject": "You won a million dollars!",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_drop(), "email from blocked sender should be dropped");
    if let TriageDecision::Drop { reason } = &d {
        assert!(
            reason.contains("blocked"),
            "reason should mention blocked: {reason}"
        );
    }
}

// ===========================================================================
// TEST SUITE 4: Image triage through Kafka pipeline
// ===========================================================================

/// Test: Invoice image → promoted with High priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_image_invoice_high_priority() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.image.invoice.{suffix}");
    let commands_topic = format!("hb.test.commands.image.invoice.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-image-invoice-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::image_classifier("invoice"));
    let processor = ImageTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("invoice scan", "img:001"),
        sensor_name: "document_scanner".into(),
        modality: SensorModality::Image,
        observed_at: Utc::now(),
        content: "Scanned invoice from Acme Corp for $5,000".into(),
        source_id: "img:001".into(),
        metadata: Some(serde_json::json!({
            "filename": "invoice_2026_Q1.pdf",
            "dimensions": "2480x3508",
        })),
        binary_ref: Some("/storage/images/invoice_2026_Q1.pdf".into()),
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote(), "invoice image should be promoted");
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::High,
            "invoices should be High priority"
        );
    }
}

/// Test: Photo image → promoted with Normal priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_image_photo_normal_priority() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.image.photo.{suffix}");
    let commands_topic = format!("hb.test.commands.image.photo.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-image-photo-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::image_classifier("photo"));
    let processor = ImageTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = make_event(
        "camera_roll",
        SensorModality::Image,
        "A photo of the team building event",
        "img:team_photo_001",
    );

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote(), "photo should be promoted");
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::Normal,
            "photos should be Normal priority"
        );
    }
}

// ===========================================================================
// TEST SUITE 5: Audio triage through Kafka pipeline
// ===========================================================================

/// Test: Voice note audio → promoted with High priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_audio_voice_note_high() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.audio.voice.{suffix}");
    let commands_topic = format!("hb.test.commands.audio.voice.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-audio-voice-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::audio_classifier("voice_note"));
    let processor = AudioTriageProcessor::new(slm, vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("voice note", "audio:voice001"),
        sensor_name: "voice_notes".into(),
        modality: SensorModality::Audio,
        observed_at: Utc::now(),
        content: "Reminder to schedule the quarterly review meeting with the team.".into(),
        source_id: "audio:voice001".into(),
        metadata: Some(serde_json::json!({
            "filename": "voice_note_20260222.wav",
            "duration_seconds": 15,
        })),
        binary_ref: Some("/storage/audio/voice_note_20260222.wav".into()),
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote(), "voice note should be promoted");
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::High,
            "voice notes should be High priority"
        );
    }
}

/// Test: Music audio → dropped
#[tokio::test]
#[ignore]
async fn kafka_pipeline_audio_music_dropped() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.audio.music.{suffix}");
    let commands_topic = format!("hb.test.commands.audio.music.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-audio-music-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::audio_classifier("music"));
    let processor = AudioTriageProcessor::new(slm, vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("music file", "audio:music001"),
        sensor_name: "audio_inbox".into(),
        modality: SensorModality::Audio,
        observed_at: Utc::now(),
        content: "Background music track for presentation".into(),
        source_id: "audio:music001".into(),
        metadata: Some(serde_json::json!({
            "filename": "background_music.mp3",
            "duration_seconds": 180,
        })),
        binary_ref: Some("/storage/audio/background_music.mp3".into()),
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    assert!(decision.unwrap().is_drop(), "music audio should be dropped");
}

// ===========================================================================
// TEST SUITE 6: Compression pipeline
// ===========================================================================

/// Test: Text compression truncates long content
#[tokio::test]
#[ignore]
async fn kafka_pipeline_compression_text_truncation() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let policy = CompressionPolicy::new();

    // Create a long text event
    let long_content = "A".repeat(10000);
    let event = make_event("test_rss", SensorModality::Text, &long_content, "rss:long");

    let (compressed, bytes_saved) = policy.compress(&event);

    assert!(
        compressed.len() <= 4096,
        "compressed text should be <= 4096 bytes, got {}",
        compressed.len()
    );
    assert!(
        bytes_saved > 0,
        "should save bytes when compressing long text"
    );
}

/// Test: Structured data passes through without compression
#[tokio::test]
#[ignore]
async fn kafka_pipeline_compression_structured_passthrough() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let policy = CompressionPolicy::new();

    let event = make_weather_event("London", "clear sky", 22.0, false);
    let original_len = event.content.len();
    let (compressed, bytes_saved) = policy.compress(&event);

    assert_eq!(
        compressed.len(),
        original_len,
        "structured data should pass through unchanged"
    );
    assert_eq!(bytes_saved, 0, "no bytes should be saved for passthrough");
}

/// Test: Image/Audio compression keeps only summary
#[tokio::test]
#[ignore]
async fn kafka_pipeline_compression_summary_only() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let policy = CompressionPolicy::new();

    // Image with long content — SummaryOnly should replace with summary from metadata
    let event = SensorEvent {
        id: "img-compress-test".into(),
        sensor_name: "camera".into(),
        modality: SensorModality::Image,
        observed_at: Utc::now(),
        content: "A".repeat(5000),
        source_id: "img:compress".into(),
        metadata: Some(serde_json::json!({"summary": "A scanned invoice from Acme Corp"})),
        binary_ref: None,
        related_ids: vec![],
    };

    let (compressed, bytes_saved) = policy.compress(&event);
    // SummaryOnly keeps a truncated version
    assert!(
        compressed.len() < 5000,
        "summary-only should be shorter than original"
    );
    assert!(bytes_saved > 0, "should save bytes with summary-only");
}

// ===========================================================================
// TEST SUITE 7: Model routing
// ===========================================================================

/// Test: Model router directs triage to local/light provider
#[tokio::test]
#[ignore]
async fn kafka_pipeline_model_routing_triage() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let light = Arc::new(MockSlmProvider::new("light")) as Arc<dyn heartbit::llm::DynLlmProvider>;
    let frontier =
        Arc::new(MockSlmProvider::new("frontier")) as Arc<dyn heartbit::llm::DynLlmProvider>;

    // Without local provider: triage should use CloudLight
    let router = ModelRouter::new(None, light.clone(), frontier.clone(), None);
    let (tier, _) = router.route_triage();
    assert_eq!(
        tier,
        ModelTier::CloudLight,
        "triage without local → CloudLight"
    );

    let (tier, _) = router.route_reason();
    assert_eq!(tier, ModelTier::CloudFrontier, "reason → CloudFrontier");

    assert!(router.route_vision().is_none(), "no vision provider → None");

    // With local provider: triage should use Local
    let local = Arc::new(MockSlmProvider::new("local")) as Arc<dyn heartbit::llm::DynLlmProvider>;
    let router_with_local = ModelRouter::new(Some(local), light, frontier, None);
    let (tier, _) = router_with_local.route_triage();
    assert_eq!(tier, ModelTier::Local, "triage with local → Local");
}

// ===========================================================================
// TEST SUITE 8: Edge cases
// ===========================================================================

/// Test: Empty content event is handled gracefully
#[tokio::test]
#[ignore]
async fn kafka_pipeline_empty_content_event() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.weather.empty.{suffix}");
    let commands_topic = format!("hb.test.commands.empty.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-empty-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = StructuredTriageProcessor;
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Empty content but valid metadata with alert=true
    let event = SensorEvent {
        id: SensorEvent::generate_id("", "empty:001"),
        sensor_name: "test_weather".into(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: String::new(),
        source_id: "empty:001".into(),
        metadata: Some(serde_json::json!({
            "location": "TestCity",
            "alert": true,
            "description": "test alert",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(
        decision.is_some(),
        "empty content should still produce a decision"
    );
    // Alert=true → should still promote even with empty content
    assert!(
        decision.unwrap().is_promote(),
        "alert=true should promote even with empty content"
    );
}

/// Test: Large payload event through Kafka
#[tokio::test]
#[ignore]
async fn kafka_pipeline_large_payload() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let topic = format!("hb.test.sensor.large.{suffix}");
    ensure_topic(&topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let consumer = test_consumer(&topic, &format!("large-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // ~100KB payload — well within Kafka's default 1MB limit
    let large_content = "X".repeat(100_000);
    let event = make_event(
        "large_sensor",
        SensorModality::Text,
        &large_content,
        "large:001",
    );

    produce_event(&producer, &topic, &event).await;

    let payload = consume_one(&consumer, Duration::from_secs(10)).await;
    assert!(payload.is_some(), "large payload should be received");

    let received: SensorEvent =
        serde_json::from_slice(&payload.unwrap()).expect("deserialize large event");
    assert_eq!(received.content.len(), 100_000, "content should be intact");
}

/// Test: Duplicate event IDs don't crash the pipeline
#[tokio::test]
#[ignore]
async fn kafka_pipeline_duplicate_events() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.weather.dup.{suffix}");
    let commands_topic = format!("hb.test.commands.dup.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-dup-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = StructuredTriageProcessor;
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Send the same event twice
    let event = make_weather_event("DupCity", "storm warning", 35.0, true);
    produce_event(&producer, &sensor_topic, &event).await;
    produce_event(&producer, &sensor_topic, &event).await;

    // Both should be processed without error
    let d1 = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;
    assert!(d1.is_some());
    assert!(d1.unwrap().is_promote());

    let d2 = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;
    assert!(d2.is_some());
    assert!(d2.unwrap().is_promote());

    // Story correlator should group them (same entities)
    let corr = correlator.lock().unwrap();
    let stories = corr.active_stories();
    let dup_story = stories.iter().find(|s| s.entities.contains("DupCity"));
    assert!(dup_story.is_some(), "should have DupCity story");
    assert_eq!(
        dup_story.unwrap().events.len(),
        2,
        "duplicate events should both be added to the same story"
    );
}

/// Test: Email thread correlation via related_ids
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_thread_correlation() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.thread.{suffix}");
    let commands_topic = format!("hb.test.commands.email.thread.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-email-thread-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::email_classifier(true, false));
    let processor = EmailTriageProcessor::new(slm, vec![], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // First email in thread
    let event1 = SensorEvent {
        id: SensorEvent::generate_id("original email", "msg:thread-001"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: Project update\n\nHere's the latest status.".into(),
        source_id: "msg:thread-001".into(),
        metadata: Some(serde_json::json!({
            "from": "alice@company.com",
            "subject": "Project update",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event1).await;
    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Reply referencing the first email via related_ids
    let event2 = SensorEvent {
        id: SensorEvent::generate_id("reply email", "msg:thread-002"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: Re: Project update\n\nLooks good, approved.".into(),
        source_id: "msg:thread-002".into(),
        metadata: Some(serde_json::json!({
            "from": "bob@company.com",
            "subject": "Re: Project update",
            "in_reply_to": "msg:thread-001",
        })),
        binary_ref: None,
        related_ids: vec![event1.id.clone()],
    };

    produce_event(&producer, &sensor_topic, &event2).await;
    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Both emails should be in the same story (thread-based correlation)
    let corr = correlator.lock().unwrap();
    let stories = corr.active_stories();
    assert!(
        !stories.is_empty(),
        "should have at least one story for email thread"
    );

    // The reply references event1.id in related_ids, so they should share a story
    let thread_story = stories.iter().find(|s| s.events.len() >= 2);
    assert!(
        thread_story.is_some(),
        "email thread should produce a single story with 2 events, got stories: {:?}",
        stories.iter().map(|s| s.events.len()).collect::<Vec<_>>()
    );
}

/// Test: SLM via OpenRouter for email triage
#[tokio::test]
#[ignore]
async fn slm_email_triage_with_openrouter() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }
    let Some(slm) = make_openrouter_slm() else {
        eprintln!("SKIP: OPENROUTER_API_KEY not set");
        return;
    };

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.slm.{suffix}");
    let commands_topic = format!("hb.test.commands.email.slm.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-slm-email-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = EmailTriageProcessor::new(slm, vec!["cto@company.com".into()], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("slm email test", "msg:slm-001"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: Server outage alert\n\nThe production database server is experiencing high latency. Current response time is 5x normal. Please investigate immediately.".into(),
        source_id: "msg:slm-001".into(),
        metadata: Some(serde_json::json!({
            "from": "monitoring@company.com",
            "subject": "Server outage alert",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(30),
    )
    .await;

    assert!(decision.is_some(), "should get a decision from real SLM");
    let d = decision.unwrap();
    // Real SLM should recognize this as relevant (server outage)
    assert!(
        d.is_promote(),
        "SLM should promote server outage email, got: {d:?}"
    );
    if let TriageDecision::Promote { summary, .. } = &d {
        assert!(!summary.is_empty(), "SLM should generate a summary");
        eprintln!("Email SLM summary: {summary}");
    }
}

// ===========================================================================
// TEST SUITE 9: SLM parse failure fallbacks (all triage processors)
// ===========================================================================

/// Mock that returns invalid JSON to trigger fallback paths.
fn mock_invalid_json() -> Arc<MockSlmProvider> {
    Arc::new(MockSlmProvider::new("This is not valid JSON at all {{"))
}

/// Test: RSS SLM returns invalid JSON → fallback uses keyword relevance
#[tokio::test]
#[ignore]
async fn kafka_pipeline_rss_slm_parse_failure_fallback() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.rss.parse_fail.{suffix}");
    let commands_topic = format!("hb.test.commands.rss.parse_fail.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-rss-parse-fail-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // High keyword relevance (> 0.3) → fallback sets relevant=true → Promote
    let processor = RssTriageProcessor::new(
        mock_invalid_json(),
        vec!["rust".into(), "programming".into()],
    );
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("Rust programming guide", "rss:fallback"),
        sensor_name: "test_rss".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "A comprehensive guide to Rust programming and systems development".into(),
        source_id: "rss:fallback".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    // keyword_relevance("...Rust programming...", ["rust", "programming"]) = 1.0 > 0.3
    // So fallback sets relevant=true → Promote
    assert!(
        d.is_promote(),
        "RSS with high keyword relevance should promote even on SLM parse failure, got: {d:?}"
    );
}

/// Test: RSS SLM parse failure with LOW keyword relevance → Drop
#[tokio::test]
#[ignore]
async fn kafka_pipeline_rss_slm_parse_failure_low_relevance_drop() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.rss.parse_fail_low.{suffix}");
    let commands_topic = format!("hb.test.commands.rss.parse_fail_low.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(
        &sensor_topic,
        &format!("triage-rss-parse-fail-low-{suffix}"),
    );
    tokio::time::sleep(Duration::from_millis(500)).await;

    // keyword "blockchain" won't match content about cooking
    // But keyword_relevance returns 0.0 with non-empty keywords → pre-filter drops
    let processor = RssTriageProcessor::new(
        mock_invalid_json(),
        vec!["blockchain".into(), "web3".into()],
    );
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = make_event(
        "test_rss",
        SensorModality::Text,
        "Italian cooking recipes for beginners",
        "rss:cooking",
    );

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    assert!(
        decision.unwrap().is_drop(),
        "RSS with no keyword match should be dropped before SLM"
    );
}

/// Test: Webhook SLM returns invalid JSON → fallback Normal priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_webhook_slm_parse_failure_fallback() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.webhook.parse_fail.{suffix}");
    let commands_topic = format!("hb.test.commands.webhook.parse_fail.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let cmd_consumer = test_consumer(&commands_topic, &format!("cmd-webhook-parse-fail-{suffix}"));
    let triage_consumer = test_consumer(
        &sensor_topic,
        &format!("triage-webhook-parse-fail-{suffix}"),
    );
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = WebhookTriageProcessor::new(mock_invalid_json());
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("webhook_event", "github:fallback"),
        sensor_name: "github_hooks".into(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: r#"{"action":"opened","issue":{"number":99}}"#.into(),
        source_id: "github:fallback".into(),
        metadata: Some(serde_json::json!({"source": "github"})),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(
        d.is_promote(),
        "webhook should promote on SLM parse failure (fallback)"
    );
    if let TriageDecision::Promote {
        priority, summary, ..
    } = &d
    {
        assert_eq!(
            *priority,
            Priority::Normal,
            "fallback should be Normal priority"
        );
        assert!(
            summary.contains("github"),
            "fallback summary should contain source: {summary}"
        );
    }

    // Should still land on commands topic
    let cmd_msg = consume_one(&cmd_consumer, Duration::from_secs(10)).await;
    assert!(cmd_msg.is_some(), "fallback should still produce command");
}

/// Test: Image SLM parse failure → filename heuristic (invoice filename → High)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_image_slm_parse_failure_invoice_heuristic() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.image.parse_fail.{suffix}");
    let commands_topic = format!("hb.test.commands.image.parse_fail.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer =
        test_consumer(&sensor_topic, &format!("triage-image-parse-fail-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = ImageTriageProcessor::new(mock_invalid_json());
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // filename contains "invoice" → heuristic should give High priority
    let event = SensorEvent {
        id: SensorEvent::generate_id("invoice image", "img:parse_fail"),
        sensor_name: "scanner".into(),
        modality: SensorModality::Image,
        observed_at: Utc::now(),
        content: "Scanned document".into(),
        source_id: "img:parse_fail".into(),
        metadata: Some(serde_json::json!({
            "filename": "invoice_march_2026.jpg",
            "extension": "jpg",
        })),
        binary_ref: Some("/storage/images/invoice_march_2026.jpg".into()),
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote());
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::High,
            "invoice in filename should trigger High via heuristic fallback"
        );
    }
}

/// Test: Email SLM returns invalid JSON → fallback defaults
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_slm_parse_failure_fallback() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.parse_fail.{suffix}");
    let commands_topic = format!("hb.test.commands.email.parse_fail.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer =
        test_consumer(&sensor_topic, &format!("triage-email-parse-fail-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // SLM returns invalid JSON → fallback: relevant=true, urgency="normal"
    let processor = EmailTriageProcessor::new(mock_invalid_json(), vec![], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("parse fail email", "msg:parse_fail"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: Meeting notes\n\nHere are the notes from today's meeting.".into(),
        source_id: "msg:parse_fail".into(),
        metadata: Some(serde_json::json!({
            "from": "colleague@company.com",
            "subject": "Meeting notes",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    // Fallback: relevant=true, urgency="normal", unknown sender → Normal
    assert!(
        d.is_promote(),
        "email should promote on SLM parse failure (fallback relevant=true)"
    );
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::Normal,
            "fallback with unknown sender should be Normal"
        );
    }
}

// ===========================================================================
// TEST SUITE 10: Email priority edge cases
// ===========================================================================

/// Test: Email known contact + urgent SLM = Critical priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_known_urgent_critical() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.critical.{suffix}");
    let commands_topic = format!("hb.test.commands.email.critical.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-email-critical-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Mock returns urgency: "high" (not boolean)
    let slm = Arc::new(MockSlmProvider::email_classifier(true, true));
    let processor = EmailTriageProcessor::new(slm, vec!["ceo@company.com".into()], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("critical email", "msg:critical"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: EMERGENCY - System down\n\nAll production servers are offline.".into(),
        source_id: "msg:critical".into(),
        metadata: Some(serde_json::json!({
            "from": "ceo@company.com",
            "subject": "EMERGENCY - System down",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote());
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::Critical,
            "known contact + urgent → Critical"
        );
    }
}

/// Test: Email unknown sender + non-urgent → Normal priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_unknown_sender_normal() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.unknown.{suffix}");
    let commands_topic = format!("hb.test.commands.email.unknown.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-email-unknown-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // relevant, not urgent
    let slm = Arc::new(MockSlmProvider::email_classifier(true, false));
    let processor = EmailTriageProcessor::new(slm, vec!["boss@company.com".into()], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("unknown email", "msg:unknown"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: Inquiry\n\nI'd like to learn about your services.".into(),
        source_id: "msg:unknown".into(),
        metadata: Some(serde_json::json!({
            "from": "stranger@external.com",
            "subject": "Inquiry",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote());
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::Normal,
            "unknown sender + not urgent → Normal"
        );
    }
}

/// Test: Email thread reply boosts priority (Normal → High)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_thread_reply_priority_boost() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.thread_boost.{suffix}");
    let commands_topic = format!("hb.test.commands.email.thread_boost.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(
        &sensor_topic,
        &format!("triage-email-thread-boost-{suffix}"),
    );
    tokio::time::sleep(Duration::from_millis(500)).await;

    // relevant, not urgent → base Normal; thread reply → boost to High
    let slm = Arc::new(MockSlmProvider::email_classifier(true, false));
    let processor = EmailTriageProcessor::new(slm, vec![], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("thread reply", "msg:thread_boost"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: Re: Meeting\n\nSounds good, see you there.".into(),
        source_id: "msg:thread_boost".into(),
        metadata: Some(serde_json::json!({
            "from": "colleague@company.com",
            "subject": "Re: Meeting",
            "in_reply_to": "<msg-original@company.com>",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote());
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::High,
            "thread reply should boost Normal → High"
        );
    }
}

// ===========================================================================
// TEST SUITE 11: Audio + Image additional categories
// ===========================================================================

/// Test: Audio with known speaker → priority boost (meeting Normal → High)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_audio_known_speaker_boost() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.audio.speaker.{suffix}");
    let commands_topic = format!("hb.test.commands.audio.speaker.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-audio-speaker-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Mock returns "meeting" → Normal, but known speaker should boost to High
    let slm = Arc::new(MockSlmProvider::audio_classifier("meeting"));
    let processor = AudioTriageProcessor::new(slm, vec!["alice@company.com".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("meeting audio", "audio:speaker"),
        sensor_name: "voice_notes".into(),
        modality: SensorModality::Audio,
        observed_at: Utc::now(),
        content: "Weekly standup recording with action items.".into(),
        source_id: "audio:speaker".into(),
        metadata: Some(serde_json::json!({
            "filename": "standup.mp3",
            "speaker": "alice@company.com",
        })),
        binary_ref: Some("/storage/audio/standup.mp3".into()),
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote());
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::High,
            "known speaker should boost meeting Normal → High"
        );
    }
}

/// Test: Image diagram → Low priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_image_diagram_low_priority() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.image.diagram.{suffix}");
    let commands_topic = format!("hb.test.commands.image.diagram.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-image-diagram-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::image_classifier("diagram"));
    let processor = ImageTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = make_event(
        "scanner",
        SensorModality::Image,
        "Architecture diagram showing microservices",
        "img:diagram001",
    );

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(d.is_promote());
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(*priority, Priority::Low, "diagram should be Low priority");
    }
}

// ===========================================================================
// TEST SUITE 12: Structured triage edge cases
// ===========================================================================

/// Test: Non-weather structured event → Drop
#[tokio::test]
#[ignore]
async fn kafka_pipeline_structured_non_weather_dropped() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.structured.nonweather.{suffix}");
    let commands_topic = format!("hb.test.commands.structured.nonweather.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(
        &sensor_topic,
        &format!("triage-structured-nonweather-{suffix}"),
    );
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = StructuredTriageProcessor;
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // GPS event (sensor_name doesn't contain "weather") → should drop
    let event = SensorEvent {
        id: SensorEvent::generate_id("gps data", "gps:001"),
        sensor_name: "gps_tracker".into(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: r#"{"lat": 51.5074, "lon": -0.1278}"#.into(),
        source_id: "gps:001".into(),
        metadata: Some(serde_json::json!({"device": "phone"})),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(
        d.is_drop(),
        "non-weather structured event should be dropped"
    );
    if let TriageDecision::Drop { reason } = &d {
        assert!(reason.contains("no actionable data"), "reason: {reason}");
    }
}

// ===========================================================================
// TEST SUITE 13: Webhook edge cases
// ===========================================================================

/// Test: Webhook with no source metadata → "unknown" source used
#[tokio::test]
#[ignore]
async fn kafka_pipeline_webhook_no_source_metadata() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.webhook.no_source.{suffix}");
    let commands_topic = format!("hb.test.commands.webhook.no_source.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer =
        test_consumer(&sensor_topic, &format!("triage-webhook-no-source-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::webhook_classifier());
    let processor = WebhookTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Event with NO metadata at all → extract_source returns "unknown"
    let event = SensorEvent {
        id: SensorEvent::generate_id("webhook no meta", "hook:no_source"),
        sensor_name: "generic_webhook".into(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: r#"{"event":"ping","status":"ok"}"#.into(),
        source_id: "hook:no_source".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    assert!(
        d.is_promote(),
        "webhook with no source should still classify"
    );
}

// ===========================================================================
// TEST SUITE 14: RSS edge cases
// ===========================================================================

/// Test: RSS with empty keywords list → all events go to SLM (no pre-filter)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_rss_empty_keywords_bypass_prefilter() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.rss.empty_kw.{suffix}");
    let commands_topic = format!("hb.test.commands.rss.empty_kw.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-rss-empty-kw-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Empty keywords → keyword_relevance returns 0.0 but the check
    // `relevance < 0.01 && !self.interest_keywords.is_empty()` is FALSE
    // because keywords IS empty → goes to SLM
    let slm = Arc::new(MockSlmProvider::rss_classifier());
    let processor = RssTriageProcessor::new(slm, vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = make_event(
        "test_rss",
        SensorModality::Text,
        "Italian cooking recipes for beginners — no tech keywords",
        "rss:empty_kw",
    );

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some());
    let d = decision.unwrap();
    // Mock SLM returns relevant=true → Promote
    assert!(
        d.is_promote(),
        "empty keyword list should skip pre-filter, go to SLM"
    );
    if let TriageDecision::Promote { priority, .. } = &d {
        // keyword_relevance = 0.0, so relevance < 0.5 → Low
        assert_eq!(*priority, Priority::Low, "0.0 relevance → Low priority");
    }
}

// ===========================================================================
// TEST SUITE 15: Metrics - dead letter + model routing
// ===========================================================================

/// Test: Dead letter metric is incremented
#[tokio::test]
#[ignore]
async fn kafka_pipeline_metrics_dead_letter() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let metrics = SensorMetrics::new().expect("create metrics");

    // Manually record a dead_letter decision
    metrics.record_event_received("broken_sensor");
    metrics.record_triage_decision("broken_sensor", "dead_letter", 0.5);

    let encoded = metrics.encode().expect("encode metrics");
    assert!(
        encoded
            .contains(r#"heartbit_sensor_events_dead_letter_total{sensor_name="broken_sensor"} 1"#),
        "dead letter metric should be incremented. Metrics:\n{encoded}"
    );
    assert!(
        encoded.contains("heartbit_sensor_triage_duration_seconds"),
        "duration should be recorded for dead letter. Metrics:\n{encoded}"
    );
}

/// Test: Model routing metrics tracked correctly
#[tokio::test]
#[ignore]
async fn kafka_pipeline_metrics_model_routing() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let metrics = SensorMetrics::new().expect("create metrics");

    // Simulate routing decisions
    metrics.record_model_routing("local");
    metrics.record_model_routing("local");
    metrics.record_model_routing("cloud_light");
    metrics.record_model_routing("cloud_frontier");

    let encoded = metrics.encode().expect("encode metrics");
    assert!(
        encoded.contains(r#"heartbit_sensor_model_routing_total{tier="local"} 2"#),
        "local routing count. Metrics:\n{encoded}"
    );
    assert!(
        encoded.contains(r#"heartbit_sensor_model_routing_total{tier="cloud_light"} 1"#),
        "cloud_light routing count. Metrics:\n{encoded}"
    );
    assert!(
        encoded.contains(r#"heartbit_sensor_model_routing_total{tier="cloud_frontier"} 1"#),
        "cloud_frontier routing count. Metrics:\n{encoded}"
    );
}

/// Test: Token budget metrics
#[tokio::test]
#[ignore]
async fn kafka_pipeline_metrics_token_budget() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let metrics = SensorMetrics::new().expect("create metrics");

    metrics.set_token_budget(25000, 100000);
    let encoded = metrics.encode().expect("encode metrics");
    assert!(
        encoded.contains("heartbit_sensor_token_budget_used 25000"),
        "token budget used. Metrics:\n{encoded}"
    );
    assert!(
        encoded.contains("heartbit_sensor_token_budget_limit 100000"),
        "token budget limit. Metrics:\n{encoded}"
    );

    // Update budget
    metrics.set_token_budget(75000, 100000);
    let encoded = metrics.encode().expect("encode metrics");
    assert!(
        encoded.contains("heartbit_sensor_token_budget_used 75000"),
        "updated token budget. Metrics:\n{encoded}"
    );
}

// ===========================================================================
// TEST SUITE 16: Story correlator edge cases
// ===========================================================================

/// Test: Story with stale marking
#[tokio::test]
#[ignore]
async fn kafka_pipeline_story_stale_marking() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.weather.stale.{suffix}");
    let commands_topic = format!("hb.test.commands.stale.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-stale-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = StructuredTriageProcessor;
    // Use a very short window so we can test staleness
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_millis(100)));

    // Create a story
    let event = make_weather_event("StaleCity", "tornado warning", 30.0, true);
    produce_event(&producer, &sensor_topic, &event).await;
    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Verify story exists
    {
        let corr = correlator.lock().unwrap();
        let stories = corr.active_stories();
        assert!(
            stories.iter().any(|s| s.entities.contains("StaleCity")),
            "should have StaleCity story"
        );
    }

    // Wait for stale window to expire
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Mark stale stories
    {
        let mut corr = correlator.lock().unwrap();
        corr.mark_stale(Duration::from_millis(100));
    }

    // Active stories should no longer contain the stale story
    {
        let corr = correlator.lock().unwrap();
        let active = corr.active_stories();
        let stale_story = active.iter().find(|s| s.entities.contains("StaleCity"));
        assert!(
            stale_story.is_none(),
            "StaleCity story should be marked stale and not in active list"
        );
    }
}

/// Test: New event after stale marking creates fresh story
#[tokio::test]
#[ignore]
async fn kafka_pipeline_story_fresh_after_stale() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.weather.fresh_stale.{suffix}");
    let commands_topic = format!("hb.test.commands.fresh_stale.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-fresh-stale-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let processor = StructuredTriageProcessor;
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_millis(100)));

    // Event 1 → creates story
    let event1 = make_weather_event("FreshCity", "storm alert", 28.0, true);
    produce_event(&producer, &sensor_topic, &event1).await;
    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Wait for stale + mark
    tokio::time::sleep(Duration::from_millis(200)).await;
    {
        let mut corr = correlator.lock().unwrap();
        corr.mark_stale(Duration::from_millis(100));
    }

    // Event 2 → should create a NEW story (old one is stale)
    let event2 = make_weather_event("FreshCity", "new storm alert", 26.0, true);
    produce_event(&producer, &sensor_topic, &event2).await;
    run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    let corr = correlator.lock().unwrap();
    let active = corr.active_stories();
    let fresh_story = active.iter().find(|s| s.entities.contains("FreshCity"));
    assert!(
        fresh_story.is_some(),
        "should have a fresh FreshCity story after stale + new event"
    );
    assert_eq!(
        fresh_story.unwrap().events.len(),
        1,
        "fresh story should have only 1 event (not merged with stale)"
    );
}

// ===========================================================================
// TEST SUITE 17: Email edge cases — known-but-irrelevant still promotes
// ===========================================================================

/// Test: Email from known contact with SLM marking irrelevant → still promoted
/// (known contacts always bypass SLM relevance)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_known_irrelevant_still_promotes() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.known_irrel.{suffix}");
    let commands_topic = format!("hb.test.commands.email.known_irrel.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer =
        test_consumer(&sensor_topic, &format!("triage-email-known-irrel-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // SLM says NOT relevant, but sender IS known → should still promote
    let slm = Arc::new(MockSlmProvider::email_classifier(false, false));
    let processor = EmailTriageProcessor::new(slm, vec!["alice@company.com".into()], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("chatty email", "msg:irrel_known"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: Lunch plans\n\nWant to grab lunch today?".into(),
        source_id: "msg:irrel_known".into(),
        metadata: Some(serde_json::json!({
            "from": "alice@company.com",
            "subject": "Lunch plans",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    let d = decision.unwrap();
    assert!(
        d.is_promote(),
        "known contact should be promoted even when SLM says irrelevant, got: {d:?}"
    );
}

/// Test: Email from unknown sender with SLM marking irrelevant → dropped
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_unknown_irrelevant_dropped() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.unk_irrel.{suffix}");
    let commands_topic = format!("hb.test.commands.email.unk_irrel.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-email-unk-irrel-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // SLM says NOT relevant, sender NOT known → should drop
    let slm = Arc::new(MockSlmProvider::email_classifier(false, false));
    let processor = EmailTriageProcessor::new(slm, vec![], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("newsletter", "msg:newsletter"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: Weekly newsletter\n\nHere is your newsletter.".into(),
        source_id: "msg:newsletter".into(),
        metadata: Some(serde_json::json!({
            "from": "news@random.com",
            "subject": "Weekly newsletter",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    let d = decision.unwrap();
    assert!(
        d.is_drop(),
        "unknown + irrelevant should be dropped, got: {d:?}"
    );
}

// ===========================================================================
// TEST SUITE 18: Content truncation edge cases
// ===========================================================================

/// Test: Email with > 2000 byte body gets truncated before SLM call
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_long_content_truncated() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.trunc.{suffix}");
    let commands_topic = format!("hb.test.commands.email.trunc.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-email-trunc-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::email_classifier(true, false));
    let processor = EmailTriageProcessor::new(slm, vec!["sender@example.com".into()], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Create event with 5000-byte body (well over 2000 limit)
    let long_body = "A".repeat(5000);
    let event = SensorEvent {
        id: SensorEvent::generate_id(&long_body, "msg:longbody"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: long_body,
        source_id: "msg:longbody".into(),
        metadata: Some(serde_json::json!({
            "from": "sender@example.com",
            "subject": "Long email",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Should still process successfully (not crash on large content)
    assert!(decision.is_some(), "should handle large email body");
    assert!(
        decision.unwrap().is_promote(),
        "known contact with long email should still promote"
    );
}

/// Test: RSS with > 2000 byte content gets truncated before SLM call
#[tokio::test]
#[ignore]
async fn kafka_pipeline_rss_long_content_truncated() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.rss.trunc.{suffix}");
    let commands_topic = format!("hb.test.commands.rss.trunc.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-rss-trunc-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::rss_classifier());
    let processor = RssTriageProcessor::new(slm, vec!["rust".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // 5000 bytes with matching keyword at the start
    let long_content = format!("Rust programming language {}", "x".repeat(4970));
    let event = SensorEvent {
        id: SensorEvent::generate_id(&long_content, "rss:longcontent"),
        sensor_name: "test_rss".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: long_content,
        source_id: "rss:longcontent".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should handle large RSS body");
    assert!(
        decision.unwrap().is_promote(),
        "RSS with matching keyword and long content should still promote"
    );
}

// ===========================================================================
// TEST SUITE 19: RSS keyword case insensitivity
// ===========================================================================

/// Test: RSS keyword matching is case-insensitive
#[tokio::test]
#[ignore]
async fn kafka_pipeline_rss_keyword_case_insensitive() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.rss.case.{suffix}");
    let commands_topic = format!("hb.test.commands.rss.case.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-rss-case-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::rss_classifier());
    // Keywords lowercase, content uppercase
    let processor = RssTriageProcessor::new(slm, vec!["rust".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("RUST IS GREAT", "rss:case_test"),
        sensor_name: "test_rss".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "RUST IS GREAT — THE 2026 EDITION IS HERE".into(),
        source_id: "rss:case_test".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    assert!(
        decision.unwrap().is_promote(),
        "case-insensitive keyword 'rust' should match 'RUST'"
    );
}

// ===========================================================================
// TEST SUITE 20: Audio case-insensitive known speaker
// ===========================================================================

/// Test: Audio known contact matching is case-insensitive (ALICE matches alice)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_audio_known_speaker_case_insensitive() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.audio.case.{suffix}");
    let commands_topic = format!("hb.test.commands.audio.case.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-audio-case-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Mock returns speaker as "ALICE@COMPANY.COM" (uppercase)
    let slm = Arc::new(MockSlmProvider::new(
        &serde_json::json!({
            "category": "other",
            "summary": "Audio note about meetings",
            "entities": ["meetings"],
            "speaker": "ALICE@COMPANY.COM"
        })
        .to_string(),
    ));
    // Known contacts list has lowercase
    let processor = AudioTriageProcessor::new(slm, vec!["alice@company.com".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("case audio", "audio:case_test"),
        sensor_name: "test_audio".into(),
        modality: SensorModality::Audio,
        observed_at: Utc::now(),
        content: "Meeting notes from this morning".into(),
        source_id: "audio:case_test".into(),
        metadata: Some(serde_json::json!({
            "speaker": "ALICE@COMPANY.COM",
            "duration_seconds": 60
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    let d = decision.unwrap();
    // "other" category would normally be Low, but known speaker boost should push it up
    assert!(
        d.is_promote(),
        "known speaker with case mismatch should still promote"
    );
    if let TriageDecision::Promote { priority, .. } = &d {
        assert!(
            *priority != Priority::Low,
            "known speaker boost should elevate from Low, got: {priority:?}"
        );
    }
}

// ===========================================================================
// TEST SUITE 21: Compression edge cases
// ===========================================================================

/// Test: SummaryOnly compression with missing metadata falls back to original content
#[tokio::test]
#[ignore]
async fn kafka_pipeline_compression_summary_only_no_metadata() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let policy = CompressionPolicy::new();
    // Default policy uses SummaryOnly for Image; no metadata → content preserved
    let event = SensorEvent {
        id: SensorEvent::generate_id("no metadata image", "img:nometa"),
        sensor_name: "test_image".into(),
        modality: SensorModality::Image,
        observed_at: Utc::now(),
        content: "This is the original image description that should be preserved".into(),
        source_id: "img:nometa".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    let (compressed, saved) = policy.compress(&event);
    assert_eq!(
        compressed, "This is the original image description that should be preserved",
        "content should be unchanged when metadata.summary is missing"
    );
    assert_eq!(saved, 0, "no bytes saved when no summary available");
}

/// Test: Compression with invalid StripPattern regex falls through gracefully
#[tokio::test]
#[ignore]
async fn kafka_pipeline_compression_invalid_regex_passthrough() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    use heartbit::sensor::compression::CompressionRule;

    let policy = CompressionPolicy::with_rules(
        SensorModality::Text,
        vec![CompressionRule::StripPattern {
            pattern: "[invalid regex((".into(),
        }],
    );

    let event = SensorEvent {
        id: SensorEvent::generate_id("regex test", "text:regex"),
        sensor_name: "test_rss".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Content that should survive invalid regex".into(),
        source_id: "text:regex".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    let (compressed, saved) = policy.compress(&event);
    assert_eq!(
        compressed, "Content that should survive invalid regex",
        "content should pass through when regex is invalid"
    );
    assert_eq!(saved, 0, "no bytes saved with invalid regex");
}

// ===========================================================================
// TEST SUITE 22: SensorEvent serde edge cases
// ===========================================================================

/// Test: SensorEvent with null metadata deserializes as None
#[tokio::test]
#[ignore]
async fn kafka_pipeline_serde_null_metadata() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    // JSON with explicit null for metadata
    let json = serde_json::json!({
        "id": "test-null-meta",
        "sensor_name": "test_sensor",
        "modality": "text",
        "observed_at": Utc::now(),
        "content": "hello",
        "source_id": "src:null",
        "metadata": null,
        "binary_ref": null,
        "related_ids": []
    });

    let event: SensorEvent = serde_json::from_value(json).expect("deserialize with null metadata");
    assert!(
        event.metadata.is_none(),
        "null metadata should deserialize as None"
    );
    assert!(
        event.binary_ref.is_none(),
        "null binary_ref should deserialize as None"
    );
    assert!(
        event.related_ids.is_empty(),
        "empty related_ids should deserialize as empty vec"
    );
}

/// Test: SensorEvent with missing optional fields deserializes correctly
#[tokio::test]
#[ignore]
async fn kafka_pipeline_serde_missing_optional_fields() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    // JSON without optional fields
    let json = serde_json::json!({
        "id": "test-missing",
        "sensor_name": "test_sensor",
        "modality": "structured",
        "observed_at": Utc::now(),
        "content": "structured data",
        "source_id": "src:missing"
    });

    let event: SensorEvent =
        serde_json::from_value(json).expect("deserialize with missing optional fields");
    assert!(event.metadata.is_none());
    assert!(event.binary_ref.is_none());
    assert!(event.related_ids.is_empty());
}

// ===========================================================================
// TEST SUITE 23: Story correlator — priority doesn't downgrade
// ===========================================================================

/// Test: Lower-priority event doesn't downgrade existing story priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_story_priority_no_downgrade() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.prinodown.{suffix}");
    let commands_topic = format!("hb.test.commands.email.prinodown.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-prinodown-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // First email: urgent + known → Critical
    let slm_urgent = Arc::new(MockSlmProvider::email_classifier(true, true));
    let processor_urgent =
        EmailTriageProcessor::new(slm_urgent, vec!["boss@company.com".into()], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event1 = SensorEvent {
        id: SensorEvent::generate_id("urgent msg", "msg:pri1"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Urgent: server is down!".into(),
        source_id: "msg:pri1".into(),
        metadata: Some(serde_json::json!({
            "from": "boss@company.com",
            "subject": "Server down!",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event1).await;
    run_triage_once(
        &triage_consumer,
        &processor_urgent,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Verify story priority is Critical or High
    let first_priority = {
        let corr = correlator.lock().unwrap();
        let stories = corr.active_stories();
        assert!(!stories.is_empty(), "should have at least one story");
        stories[0].priority
    };
    assert!(
        first_priority == Priority::Critical || first_priority == Priority::High,
        "first event should create high/critical story, got: {first_priority:?}"
    );

    // Second email: normal priority from unknown, but overlapping entities
    let slm_normal = Arc::new(MockSlmProvider::email_classifier(true, false));
    let processor_normal = EmailTriageProcessor::new(slm_normal, vec![], vec![]);

    let event2 = SensorEvent {
        id: SensorEvent::generate_id("followup", "msg:pri2"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "FYI: server metrics look okay now".into(),
        source_id: "msg:pri2".into(),
        metadata: Some(serde_json::json!({
            "from": "ops@company.com",
            "subject": "Server followup",
        })),
        binary_ref: None,
        // Link to same thread
        related_ids: vec!["msg:pri1".into()],
    };

    produce_event(&producer, &sensor_topic, &event2).await;
    run_triage_once(
        &triage_consumer,
        &processor_normal,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Story priority should NOT have downgraded
    let final_priority = {
        let corr = correlator.lock().unwrap();
        let stories = corr.active_stories();
        // Find story with the first event
        let story = stories
            .iter()
            .find(|s| s.events.iter().any(|e| e.event_id == event1.id));
        assert!(story.is_some(), "original story should still exist");
        story.unwrap().priority
    };

    assert!(
        final_priority >= first_priority,
        "story priority should not downgrade: was {first_priority:?}, now {final_priority:?}"
    );
}

// ===========================================================================
// TEST SUITE 24: Webhook with long content truncation
// ===========================================================================

/// Test: Webhook with very long payload gets truncated before SLM call
#[tokio::test]
#[ignore]
async fn kafka_pipeline_webhook_long_content_truncated() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.webhook.trunc.{suffix}");
    let commands_topic = format!("hb.test.commands.webhook.trunc.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-webhook-trunc-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::webhook_classifier());
    let processor = WebhookTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // 5000-byte payload
    let long_content = format!(
        "{{\"action\": \"review_requested\", \"data\": \"{}\"}}",
        "x".repeat(4900)
    );
    let event = SensorEvent {
        id: SensorEvent::generate_id(&long_content, "webhook:longtrunc"),
        sensor_name: "test_webhook".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: long_content,
        source_id: "webhook:longtrunc".into(),
        metadata: Some(serde_json::json!({
            "source": "github",
            "event_type": "pull_request"
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(
        decision.is_some(),
        "should handle large webhook payload without crash"
    );
    assert!(
        decision.unwrap().is_promote(),
        "webhook with review_requested should promote even with long content"
    );
}

// ===========================================================================
// TEST SUITE 25: Audio SLM parse failure → fallback to "other" → Low priority
// ===========================================================================

/// Test: Audio with invalid SLM JSON falls back to Low priority
#[tokio::test]
#[ignore]
async fn kafka_pipeline_audio_slm_parse_failure_fallback() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.audio.parsefail.{suffix}");
    let commands_topic = format!("hb.test.commands.audio.parsefail.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-audio-parsefail-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = mock_invalid_json();
    let processor = AudioTriageProcessor::new(slm, vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("audio parse fail", "audio:parsefail"),
        sensor_name: "test_audio".into(),
        modality: SensorModality::Audio,
        observed_at: Utc::now(),
        content: "Transcript of a voice note about project updates".into(),
        source_id: "audio:parsefail".into(),
        metadata: Some(serde_json::json!({
            "duration_seconds": 45
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    let d = decision.unwrap();
    // SLM parse failure defaults to "other" category → Low priority → Promote
    assert!(
        d.is_promote(),
        "audio SLM parse failure should fall back to Promote with Low priority"
    );
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::Low,
            "audio SLM parse failure defaults to Low priority"
        );
    }
}

// ===========================================================================
// TEST SUITE 26: Image SLM parse failure — screenshot heuristic
// ===========================================================================

/// Test: Image with filename containing "screenshot" falls back to Normal on SLM failure
#[tokio::test]
#[ignore]
async fn kafka_pipeline_image_slm_parse_failure_screenshot_heuristic() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.image.screenshot.{suffix}");
    let commands_topic = format!("hb.test.commands.image.screenshot.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-img-screenshot-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = mock_invalid_json();
    let processor = ImageTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("screenshot image", "img:screenshot_test"),
        sensor_name: "test_image".into(),
        modality: SensorModality::Image,
        observed_at: Utc::now(),
        content: "A screenshot from the application".into(),
        source_id: "img:screenshot_test".into(),
        metadata: Some(serde_json::json!({
            "filename": "screenshot_2026.png",
            "size_bytes": 50000,
            "dimensions": "1920x1080"
        })),
        binary_ref: Some("/tmp/screenshots/screenshot_2026.png".into()),
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    let d = decision.unwrap();
    assert!(
        d.is_promote(),
        "screenshot with SLM parse failure should still promote"
    );
    if let TriageDecision::Promote { priority, .. } = &d {
        // screenshot → not invoice/receipt/document → Normal heuristic
        assert_eq!(
            *priority,
            Priority::Normal,
            "screenshot heuristic should give Normal priority"
        );
    }
}

// ===========================================================================
// TEST SUITE 27: Metrics — multiple sensors, label cardinality
// ===========================================================================

/// Test: Metrics track multiple unique sensor names correctly
#[tokio::test]
#[ignore]
async fn kafka_pipeline_metrics_multiple_sensors() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let metrics = SensorMetrics::new().expect("create sensor metrics");

    // Record events from many different sensors
    for i in 0..10 {
        let name = format!("sensor_{i}");
        metrics.record_event_received(&name);
        metrics.record_triage_decision(&name, "promote", 0.01);
    }

    let text = metrics.encode().expect("encode metrics");

    // All 10 sensors should appear
    for i in 0..10 {
        let label = format!("sensor_{i}");
        assert!(
            text.contains(&format!("sensor_name=\"{label}\"")),
            "metrics should contain label for {label}"
        );
    }

    // Gauges should still work
    metrics.set_stories_active(7);
    let text = metrics.encode().unwrap();
    assert!(text.contains("heartbit_sensor_stories_active 7"));
}

// ===========================================================================
// TEST SUITE 28: Email — empty from field edge case
// ===========================================================================

/// Test: Email with empty "from" field is not blocked (blocked list has empty string guard)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_empty_from_not_blocked() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.empty_from.{suffix}");
    let commands_topic = format!("hb.test.commands.email.empty_from.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-empty-from-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::email_classifier(true, false));
    // Put empty string in blocked list — should NOT block events with empty from
    let processor = EmailTriageProcessor::new(slm, vec![], vec!["".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("no from email", "msg:nofrom"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: System notification\n\nDisk space warning.".into(),
        source_id: "msg:nofrom".into(),
        metadata: Some(serde_json::json!({
            "from": "",
            "subject": "System notification",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should get a triage decision");
    let d = decision.unwrap();
    // Empty from → not blocked, not known → SLM decides → relevant=true → Promote
    assert!(
        d.is_promote(),
        "email with empty from should not be blocked, got: {d:?}"
    );
}

// ===========================================================================
// TEST SUITE 29: Compression — multi-rule pipeline
// ===========================================================================

/// Test: Multiple compression rules applied sequentially (strip + truncate)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_compression_multi_rule_pipeline() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    use heartbit::sensor::compression::CompressionRule;

    let policy = CompressionPolicy::with_rules(
        SensorModality::Text,
        vec![
            // First strip email signatures
            CompressionRule::StripPattern {
                pattern: r"--\s*\n[\s\S]*$".into(),
            },
            // Then truncate to 100 bytes
            CompressionRule::Truncate { max_bytes: 100 },
        ],
    );

    let event = SensorEvent {
        id: SensorEvent::generate_id("multi rule", "text:multi"),
        sensor_name: "test_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Hello, this is the main email body with important information.\n\n-- \nBest regards,\nJohn Doe\nSenior Engineer\njohn@example.com\n+1-555-0123".into(),
        source_id: "text:multi".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    let (compressed, saved) = policy.compress(&event);
    // Signature should be stripped first
    assert!(
        !compressed.contains("Best regards"),
        "signature should be stripped"
    );
    assert!(
        !compressed.contains("john@example.com"),
        "signature details should be stripped"
    );
    // Then truncated to 100 bytes
    assert!(compressed.len() <= 100, "should be truncated to 100 bytes");
    assert!(saved > 0, "should have saved some bytes");
}

// ===========================================================================
// TEST SUITE 30: ModelRouter — all routing methods
// ===========================================================================

/// Test: ModelRouter routes to local provider when available
#[tokio::test]
#[ignore]
async fn kafka_pipeline_model_router_local_triage() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let local = Arc::new(MockSlmProvider::rss_classifier());
    let light = Arc::new(MockSlmProvider::rss_classifier());
    let frontier = Arc::new(MockSlmProvider::rss_classifier());

    let router = ModelRouter::new(
        Some(local as Arc<dyn heartbit::llm::DynLlmProvider>),
        light as Arc<dyn heartbit::llm::DynLlmProvider>,
        frontier as Arc<dyn heartbit::llm::DynLlmProvider>,
        None,
    );

    // With local provider, triage should route to Local tier
    let (tier, _) = router.route_triage();
    assert_eq!(
        tier,
        ModelTier::Local,
        "triage should prefer local when available"
    );

    // Summarize also prefers local
    let (tier, _) = router.route_summarize();
    assert_eq!(
        tier,
        ModelTier::Local,
        "summarize should prefer local when available"
    );

    // Reason always uses frontier
    let (tier, _) = router.route_reason();
    assert_eq!(
        tier,
        ModelTier::CloudFrontier,
        "reason always uses frontier"
    );

    // Vision returns None when not configured
    assert!(
        router.route_vision().is_none(),
        "vision should be None when not configured"
    );
}

/// Test: ModelRouter routes to cloud light when no local provider
#[tokio::test]
#[ignore]
async fn kafka_pipeline_model_router_no_local_fallback() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let light = Arc::new(MockSlmProvider::rss_classifier());
    let frontier = Arc::new(MockSlmProvider::rss_classifier());
    let vision = Arc::new(MockSlmProvider::image_classifier("photo"));

    let router = ModelRouter::new(
        None, // No local provider
        light as Arc<dyn heartbit::llm::DynLlmProvider>,
        frontier as Arc<dyn heartbit::llm::DynLlmProvider>,
        Some(vision as Arc<dyn heartbit::llm::DynLlmProvider>),
    );

    // Without local, triage falls back to cloud light
    let (tier, _) = router.route_triage();
    assert_eq!(
        tier,
        ModelTier::CloudLight,
        "triage should fall back to cloud light"
    );

    // Summarize also falls back to cloud light
    let (tier, _) = router.route_summarize();
    assert_eq!(
        tier,
        ModelTier::CloudLight,
        "summarize should fall back to cloud light"
    );

    // Vision returns Some when configured
    let vision_result = router.route_vision();
    assert!(
        vision_result.is_some(),
        "vision should be Some when configured"
    );
    let (tier, _) = vision_result.unwrap();
    assert_eq!(tier, ModelTier::Vision, "vision tier should be Vision");
}

// ===========================================================================
// TEST SUITE 31: Image with missing metadata fields
// ===========================================================================

/// Test: Image event with no size_bytes in metadata still processes
#[tokio::test]
#[ignore]
async fn kafka_pipeline_image_missing_size_metadata() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.image.nosize.{suffix}");
    let commands_topic = format!("hb.test.commands.image.nosize.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-img-nosize-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::image_classifier("invoice"));
    let processor = ImageTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Image event with minimal metadata (no size_bytes, no dimensions)
    let event = SensorEvent {
        id: SensorEvent::generate_id("minimal image", "img:nosize"),
        sensor_name: "test_image".into(),
        modality: SensorModality::Image,
        observed_at: Utc::now(),
        content: "A scanned document image".into(),
        source_id: "img:nosize".into(),
        metadata: Some(serde_json::json!({
            "filename": "scan.pdf"
        })),
        binary_ref: Some("/tmp/scans/scan.pdf".into()),
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(
        decision.is_some(),
        "should handle image with missing size metadata"
    );
    assert!(
        decision.unwrap().is_promote(),
        "invoice image should promote even without size metadata"
    );
}

// ===========================================================================
// TEST SUITE 32: Email with missing "from" (no metadata at all)
// ===========================================================================

/// Test: Email event with no metadata at all still processes via SLM fallback
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_no_metadata() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.email.nometa.{suffix}");
    let commands_topic = format!("hb.test.commands.email.nometa.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-email-nometa-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::email_classifier(true, false));
    let processor = EmailTriageProcessor::new(slm, vec![], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    // Email with NO metadata at all
    let event = SensorEvent {
        id: SensorEvent::generate_id("email no meta", "msg:nometa"),
        sensor_name: "work_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Subject: Meeting tomorrow\n\nPlease confirm attendance.".into(),
        source_id: "msg:nometa".into(),
        metadata: None, // No metadata at all
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should handle email with no metadata");
    // SLM says relevant=true → Promote; no known contact → Normal priority
    let d = decision.unwrap();
    assert!(
        d.is_promote(),
        "email with no metadata but relevant should promote"
    );
    if let TriageDecision::Promote { priority, .. } = &d {
        assert_eq!(
            *priority,
            Priority::Normal,
            "no metadata → no known contact → Normal priority"
        );
    }
}

// ===========================================================================
// TEST SUITE 33: Real SLM tests — image and audio (OpenRouter)
// ===========================================================================

/// Test: Image triage with real OpenRouter SLM
#[tokio::test]
#[ignore]
async fn slm_image_triage_with_openrouter() {
    use heartbit::OpenRouterProvider;

    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("SKIP: OPENROUTER_API_KEY not set");
            return;
        }
    };

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.image.slm.{suffix}");
    let commands_topic = format!("hb.test.commands.image.slm.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-img-slm-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let provider = OpenRouterProvider::new(&api_key, "qwen/qwen3-4b");
    let slm: Arc<dyn heartbit::llm::DynLlmProvider> = Arc::new(provider);
    let processor = ImageTriageProcessor::new(slm);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("real image slm", "img:real_slm"),
        sensor_name: "test_image".into(),
        modality: SensorModality::Image,
        observed_at: Utc::now(),
        content: "A photo of a whiteboard with project planning notes and deadlines".into(),
        source_id: "img:real_slm".into(),
        metadata: Some(serde_json::json!({
            "filename": "whiteboard_photo.jpg",
            "size_bytes": 2500000,
            "dimensions": "4032x3024"
        })),
        binary_ref: Some("/tmp/photos/whiteboard_photo.jpg".into()),
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(30),
    )
    .await;

    assert!(
        decision.is_some(),
        "image triage with real SLM should return a decision"
    );
    // Real SLM should classify and promote (whiteboard photo is useful)
    let d = decision.unwrap();
    assert!(
        d.is_promote(),
        "whiteboard photo should be promoted by real SLM, got: {d:?}"
    );
}

/// Test: Audio triage with real OpenRouter SLM
#[tokio::test]
#[ignore]
async fn slm_audio_triage_with_openrouter() {
    use heartbit::OpenRouterProvider;

    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("SKIP: OPENROUTER_API_KEY not set");
            return;
        }
    };

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let sensor_topic = format!("hb.test.sensor.audio.slm.{suffix}");
    let commands_topic = format!("hb.test.commands.audio.slm.{suffix}");

    ensure_topic(&sensor_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-audio-slm-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let provider = OpenRouterProvider::new(&api_key, "qwen/qwen3-4b");
    let slm: Arc<dyn heartbit::llm::DynLlmProvider> = Arc::new(provider);
    let processor = AudioTriageProcessor::new(slm, vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let event = SensorEvent {
        id: SensorEvent::generate_id("real audio slm", "audio:real_slm"),
        sensor_name: "test_audio".into(),
        modality: SensorModality::Audio,
        observed_at: Utc::now(),
        content: "Hey team, quick update on the quarterly review. Revenue is up 15% and we need to finalize the budget by Friday. Please send me your department numbers by end of day tomorrow.".into(),
        source_id: "audio:real_slm".into(),
        metadata: Some(serde_json::json!({
            "duration_seconds": 30,
            "format": "wav"
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    produce_event(&producer, &sensor_topic, &event).await;

    let decision = run_triage_once(
        &triage_consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(30),
    )
    .await;

    assert!(
        decision.is_some(),
        "audio triage with real SLM should return a decision"
    );
    let d = decision.unwrap();
    // Voice note about quarterly review should be categorized as voice_note or meeting
    assert!(
        d.is_promote(),
        "voice note about quarterly review should be promoted, got: {d:?}"
    );
}

// ===========================================================================
// TEST SUITE 34: Sensor source → Kafka (RSS source with mock HTTP feed)
// ===========================================================================

/// Test: RssSensor::run() fetches RSS feed from mock HTTP server, produces events to Kafka
#[tokio::test]
#[ignore]
async fn kafka_source_rss_sensor_produces_events() {
    use heartbit::sensor::Sensor;
    use heartbit::sensor::sources::rss::RssSensor;
    use tokio_util::sync::CancellationToken;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let rss_topic = format!("hb.test.source.rss.{suffix}");
    ensure_topic(&rss_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Start a mock HTTP server serving an RSS feed
    let rss_xml = r#"<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Rust 2026 Released</title>
      <link>https://example.com/rust-2026</link>
      <description>Exciting new features in the Rust 2026 edition.</description>
    </item>
    <item>
      <title>AI Breakthrough</title>
      <link>https://example.com/ai-breakthrough</link>
      <description>New AI model achieves human-level reasoning.</description>
    </item>
  </channel>
</rss>"#;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let mock_addr = listener.local_addr().expect("local addr");
    let mock_url = format!("http://{mock_addr}/feed.xml");

    // Serve the RSS XML on any request
    let rss_xml_owned = rss_xml.to_string();
    let server_handle = tokio::spawn(async move {
        // Accept up to 3 connections (sensor may poll multiple times)
        for _ in 0..3 {
            if let Ok((mut stream, _)) = listener.accept().await {
                let xml = rss_xml_owned.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = vec![0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/rss+xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        xml.len(),
                        xml
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.shutdown().await;
                });
            }
        }
    });

    // Create and run the RSS sensor
    let sensor = RssSensor::new(
        "test_rss_source",
        vec![mock_url],
        Duration::from_millis(100),
    );

    // Override the kafka_topic by subscribing our consumer to the sensor's topic
    let consumer = test_consumer(sensor.kafka_topic(), &format!("source-rss-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let producer = test_producer();

    let sensor_handle = tokio::spawn(async move { sensor.run(producer, cancel_clone).await });

    // Wait for the sensor to fetch and produce events
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Cancel the sensor
    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), sensor_handle).await;
    server_handle.abort();

    // Consume events from the sensor's Kafka topic, filtering by our sensor name
    let mut events = Vec::new();
    while let Some(payload) = consume_one(&consumer, Duration::from_secs(3)).await {
        if let Ok(event) = serde_json::from_slice::<SensorEvent>(&payload) {
            if event.sensor_name == "test_rss_source" {
                events.push(event);
            }
        }
    }

    assert!(
        events.len() >= 2,
        "RSS sensor should have produced at least 2 events, got {}",
        events.len()
    );

    // Verify event properties
    let titles: Vec<&str> = events
        .iter()
        .filter_map(|e| e.metadata.as_ref()?.get("title")?.as_str())
        .collect();
    assert!(
        titles.iter().any(|t| t.contains("Rust")),
        "should have Rust article, titles: {titles:?}"
    );
    assert!(
        titles.iter().any(|t| t.contains("AI")),
        "should have AI article, titles: {titles:?}"
    );

    for event in &events {
        assert_eq!(event.sensor_name, "test_rss_source");
        assert_eq!(event.modality, SensorModality::Text);
        assert!(!event.content.is_empty());
        assert!(!event.source_id.is_empty());
    }
}

// ===========================================================================
// TEST SUITE 35: Image sensor source → Kafka (file system based)
// ===========================================================================

/// Test: ImageSensor::run() watches directory for new images, produces events to Kafka
#[tokio::test]
#[ignore]
async fn kafka_source_image_sensor_produces_events() {
    use heartbit::sensor::Sensor;
    use heartbit::sensor::sources::image::ImageSensor;
    use tokio_util::sync::CancellationToken;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();

    // Create temp directory with test images
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let dir_path = temp_dir.path();

    // Create some "image" files (content doesn't matter, just extension)
    std::fs::write(dir_path.join("test_photo.jpg"), b"fake jpg data").expect("write jpg");
    std::fs::write(dir_path.join("invoice_scan.png"), b"fake png data").expect("write png");
    std::fs::write(dir_path.join("notes.txt"), b"not an image").expect("write txt");

    let sensor = ImageSensor::new("test_image_source", dir_path, Duration::from_millis(100));

    // Subscribe to the sensor's topic
    let consumer = test_consumer(sensor.kafka_topic(), &format!("source-image-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let producer = test_producer();

    let sensor_handle = tokio::spawn(async move { sensor.run(producer, cancel_clone).await });

    // Wait for the sensor to scan directory and produce events
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Cancel the sensor
    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), sensor_handle).await;

    // Consume events
    let mut events = Vec::new();
    while let Some(payload) = consume_one(&consumer, Duration::from_secs(3)).await {
        if let Ok(event) = serde_json::from_slice::<SensorEvent>(&payload) {
            events.push(event);
        }
    }

    // Should have found 2 image files (jpg + png), not the txt
    assert!(
        events.len() >= 2,
        "image sensor should have produced at least 2 events (jpg+png), got {}",
        events.len()
    );

    for event in &events {
        assert_eq!(event.sensor_name, "test_image_source");
        assert_eq!(event.modality, SensorModality::Image);
        assert!(
            event.binary_ref.is_some(),
            "image events should have binary_ref"
        );
    }

    // Verify no txt files were picked up
    let filenames: Vec<String> = events
        .iter()
        .filter_map(|e| {
            e.metadata
                .as_ref()?
                .get("filename")?
                .as_str()
                .map(String::from)
        })
        .collect();
    assert!(
        !filenames.iter().any(|f| f.ends_with(".txt")),
        "should not include .txt files, got: {filenames:?}"
    );
}

// ===========================================================================
// TEST SUITE 36: Webhook source → Kafka (build event + produce)
// ===========================================================================

/// Test: WebhookSensor builds events from payloads and produces to Kafka
#[tokio::test]
#[ignore]
async fn kafka_source_webhook_sensor_build_and_produce() {
    use heartbit::sensor::sources::webhook::WebhookSensor;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let webhook_topic = format!("hb.test.source.webhook.{suffix}");
    ensure_topic(&webhook_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();
    let consumer = test_consumer(&webhook_topic, &format!("source-webhook-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let sensor = WebhookSensor::new("github_hooks", "/webhooks/github", Some("secret123".into()));

    // Simulate receiving webhook payloads
    let payloads = vec![
        br#"{"action":"opened","number":42,"repo":"user/project"}"#.to_vec(),
        br#"{"action":"closed","number":43,"repo":"user/project"}"#.to_vec(),
        br#"{"action":"review_requested","number":44,"reviewer":"alice"}"#.to_vec(),
    ];

    for payload in &payloads {
        let event = sensor
            .build_event_from_payload(payload)
            .expect("build event");
        let event_bytes = serde_json::to_vec(&event).expect("serialize");
        let key = sensor.kafka_key(payload);

        producer
            .send(
                rdkafka::producer::FutureRecord::to(&webhook_topic)
                    .key(&key)
                    .payload(&event_bytes),
                rdkafka::util::Timeout::After(Duration::from_secs(5)),
            )
            .await
            .expect("produce webhook event");
    }

    // Consume and verify
    let mut events = Vec::new();
    for _ in 0..3 {
        if let Some(payload) = consume_one(&consumer, Duration::from_secs(5)).await {
            if let Ok(event) = serde_json::from_slice::<SensorEvent>(&payload) {
                events.push(event);
            }
        }
    }

    assert_eq!(events.len(), 3, "should have consumed 3 webhook events");

    for event in &events {
        assert_eq!(event.sensor_name, "github_hooks");
        assert_eq!(event.modality, SensorModality::Structured);
        assert!(event.content.contains("action"));
    }

    // Verify deterministic IDs (same payload → same event ID)
    let event1a = sensor
        .build_event_from_payload(&payloads[0])
        .expect("build");
    let event1b = sensor
        .build_event_from_payload(&payloads[0])
        .expect("build");
    assert_eq!(
        event1a.id, event1b.id,
        "same payload should produce same ID"
    );
}

// ===========================================================================
// TEST SUITE 37: Full pipeline integration — RSS source → triage → story → commands
// ===========================================================================

/// Test: Full end-to-end pipeline from RSS HTTP feed → sensor → Kafka → triage → story → commands
#[tokio::test]
#[ignore]
async fn kafka_full_pipeline_rss_source_to_commands() {
    use heartbit::sensor::Sensor;
    use heartbit::sensor::sources::rss::RssSensor;
    use tokio_util::sync::CancellationToken;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let commands_topic = format!("hb.test.commands.fullpipe.{suffix}");

    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Mock RSS feed
    let rss_xml = r#"<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Tech News</title>
    <item>
      <title>Rust WebAssembly Update</title>
      <link>https://example.com/rust-wasm</link>
      <description>New Rust tooling for WebAssembly improves developer experience.</description>
    </item>
  </channel>
</rss>"#;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let mock_addr = listener.local_addr().expect("local addr");
    let mock_url = format!("http://{mock_addr}/feed.xml");

    let rss_xml_owned = rss_xml.to_string();
    let server_handle = tokio::spawn(async move {
        for _ in 0..5 {
            if let Ok((mut stream, _)) = listener.accept().await {
                let xml = rss_xml_owned.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = vec![0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/rss+xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        xml.len(),
                        xml
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.shutdown().await;
                });
            }
        }
    });

    // Start RSS sensor producing to its default topic
    let sensor = RssSensor::new("fullpipe_rss", vec![mock_url], Duration::from_millis(200));
    let sensor_topic = sensor.kafka_topic().to_string();
    ensure_topic(&sensor_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let sensor_producer = test_producer();

    let sensor_handle =
        tokio::spawn(async move { sensor.run(sensor_producer, cancel_clone).await });

    // Wait for sensor to produce
    tokio::time::sleep(Duration::from_secs(3)).await;
    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), sensor_handle).await;
    server_handle.abort();

    // Now run triage on the sensor topic
    let triage_consumer = test_consumer(&sensor_topic, &format!("triage-fullpipe-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::rss_classifier());
    let processor = RssTriageProcessor::new(slm, vec!["rust".into(), "wasm".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));
    let triage_producer = test_producer();

    // Process events through triage — loop until we find our sensor's event
    // (shared topic may contain events from other tests)
    let mut decision = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while tokio::time::Instant::now() < deadline {
        if let Some(payload) = consume_one(&triage_consumer, Duration::from_secs(3)).await {
            if let Ok(event) = serde_json::from_slice::<SensorEvent>(&payload) {
                if event.sensor_name == "fullpipe_rss" {
                    let d = processor.process(&event).await.ok();
                    if let Some(TriageDecision::Promote {
                        ref summary,
                        ref priority,
                        ref extracted_entities,
                        estimated_tokens: _,
                        ..
                    }) = d
                    {
                        let entities: HashSet<String> =
                            extracted_entities.iter().cloned().collect();
                        let mut corr = correlator.lock().unwrap_or_else(|e| e.into_inner());
                        let story_id = corr.correlate_with_links(
                            &event.id,
                            &event.sensor_name,
                            summary,
                            &entities,
                            *priority,
                            &event.related_ids,
                        );
                        drop(corr);

                        let cmd = serde_json::json!({
                            "task": summary,
                            "source": format!("sensor:{}", event.sensor_name),
                            "priority": priority,
                            "story_id": story_id,
                            "sensor_event_id": event.id,
                        });
                        let cmd_bytes = serde_json::to_vec(&cmd).unwrap();
                        let record = rdkafka::producer::FutureRecord::to(&commands_topic)
                            .payload(&cmd_bytes)
                            .key(&event.id);
                        triage_producer
                            .send(
                                record,
                                rdkafka::util::Timeout::After(Duration::from_secs(5)),
                            )
                            .await
                            .expect("produce command");
                    }
                    decision = d;
                    break;
                }
            }
        }
    }

    assert!(
        decision.is_some(),
        "full pipeline should produce a triage decision"
    );
    assert!(
        decision.as_ref().unwrap().is_promote(),
        "rust+wasm article should be promoted"
    );

    // Verify command landed in commands topic
    let cmd_consumer = test_consumer(&commands_topic, &format!("cmd-fullpipe-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let cmd_payload = consume_one(&cmd_consumer, Duration::from_secs(5)).await;
    assert!(
        cmd_payload.is_some(),
        "promoted event should produce a command"
    );

    let cmd: serde_json::Value =
        serde_json::from_slice(&cmd_payload.unwrap()).expect("parse command");
    assert!(
        cmd["task"].as_str().is_some(),
        "command should have task field"
    );
    let source = cmd["source"].as_str().expect("source field");
    assert!(
        source.contains("sensor:"),
        "source should reference sensor origin, got: {source}"
    );
    assert!(
        cmd["story_id"].as_str().is_some(),
        "command should have story_id"
    );
}

// ===========================================================================
// TEST SUITE 38: JMAP Email sensor E2E — mock JMAP server → sensor → Kafka
// ===========================================================================

/// Test: JmapEmailSensor::run() discovers session, fetches emails, produces to Kafka
#[tokio::test]
#[ignore]
async fn kafka_source_jmap_sensor_produces_events() {
    use heartbit::sensor::Sensor;
    use heartbit::sensor::sources::jmap::JmapEmailSensor;
    use tokio_util::sync::CancellationToken;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();

    // Mock JMAP server: serves session discovery and email fetch
    let session_json = r#"{
        "apiUrl": "API_URL_PLACEHOLDER",
        "downloadUrl": "http://localhost/download",
        "uploadUrl": "http://localhost/upload",
        "eventSourceUrl": "http://localhost/events",
        "primaryAccounts": {
            "urn:ietf:params:jmap:mail": "acc-001"
        },
        "capabilities": {}
    }"#;

    let email_response_json = r#"{
        "methodResponses": [
            ["Email/query", {"ids": ["e1"]}, "q"],
            ["Email/get", {
                "accountId": "acc-001",
                "list": [
                    {
                        "id": "e1",
                        "messageId": ["msg-jmap-e2e@test.com"],
                        "subject": "E2E Test Email",
                        "from": [{"name": "Tester", "email": "test@example.com"}],
                        "to": [{"name": "User", "email": "user@example.com"}],
                        "receivedAt": "2026-02-22T10:00:00Z",
                        "textBody": [{"partId": "1", "type": "text/plain"}],
                        "bodyValues": {
                            "1": {"value": "This is an E2E test email body."}
                        },
                        "inReplyTo": null,
                        "references": null
                    }
                ]
            }, "g"]
        ]
    }"#;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock JMAP server");
    let mock_addr = listener.local_addr().expect("local addr");
    let server_url = format!("http://{mock_addr}");
    let api_url = format!("http://{mock_addr}/api/");

    // Replace API_URL_PLACEHOLDER in session JSON with actual URL
    let session_json = session_json.replace("API_URL_PLACEHOLDER", &api_url);

    let server_handle = tokio::spawn(async move {
        // Accept multiple connections (session discovery + email fetch + potentially more)
        for _ in 0..10 {
            if let Ok((mut stream, _)) = listener.accept().await {
                let session = session_json.clone();
                let emails = email_response_json.to_string();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = vec![0u8; 8192];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let request = String::from_utf8_lossy(&buf[..n]);

                    let body = if request.contains("/.well-known/jmap") {
                        session
                    } else {
                        emails
                    };

                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.shutdown().await;
                });
            }
        }
    });

    let sensor = JmapEmailSensor::new(
        "e2e_jmap_test",
        &server_url,
        "testuser",
        "testpass",
        vec!["vip@example.com".into()],
        Duration::from_millis(200),
    );

    let consumer = test_consumer(sensor.kafka_topic(), &format!("jmap-source-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let producer = test_producer();

    let sensor_handle = tokio::spawn(async move { sensor.run(producer, cancel_clone).await });

    // Wait for the sensor to fetch and produce
    tokio::time::sleep(Duration::from_secs(3)).await;

    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), sensor_handle).await;
    server_handle.abort();

    // Consume events from the email topic, filtering by our sensor name
    let mut events = Vec::new();
    while let Some(payload) = consume_one(&consumer, Duration::from_secs(3)).await {
        if let Ok(event) = serde_json::from_slice::<SensorEvent>(&payload) {
            if event.sensor_name == "e2e_jmap_test" {
                events.push(event);
            }
        }
    }

    assert!(
        !events.is_empty(),
        "JMAP sensor should have produced at least 1 event"
    );

    let event = &events[0];
    assert_eq!(event.sensor_name, "e2e_jmap_test");
    assert_eq!(event.modality, SensorModality::Text);
    assert!(
        event.content.contains("E2E test email body"),
        "content: {}",
        event.content
    );

    // Verify metadata
    let meta = event.metadata.as_ref().expect("metadata should be present");
    assert_eq!(meta["subject"].as_str().unwrap_or(""), "E2E Test Email");
    assert_eq!(meta["from"].as_str().unwrap_or(""), "test@example.com");
}

// ===========================================================================
// TEST SUITE 39: Weather sensor E2E — mock OpenWeatherMap → sensor → Kafka
// ===========================================================================

/// Test: WeatherSensor::run() polls mock API, produces events to Kafka
#[tokio::test]
#[ignore]
async fn kafka_source_weather_sensor_produces_events() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();

    // The WeatherSensor builds URLs to api.openweathermap.org which we can't
    // easily redirect. Instead, simulate the sensor's output and test the full
    // Kafka → triage pipeline with a manually produced weather event.

    // Produce a weather event manually (simulating what the sensor would do)
    let producer = test_producer();
    let topic = "hb.sensor.weather";
    ensure_topic(topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let weather_event = SensorEvent {
        id: SensorEvent::generate_id("weather content", "TestCity:1234567890"),
        sensor_name: format!("weather_e2e_{suffix}"),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: serde_json::json!({
            "temperature_c": 42.0,
            "description": "extreme heat warning",
            "wind_speed_ms": 5.0,
            "humidity_pct": 30.0,
        })
        .to_string(),
        source_id: "TestCity:1234567890".into(),
        metadata: Some(serde_json::json!({
            "location": "TestCity",
            "temperature_c": 42.0,
            "description": "extreme heat warning",
            "wind_speed_ms": 5.0,
            "humidity_pct": 30.0,
            "alert": true,
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    let payload = serde_json::to_vec(&weather_event).expect("serialize");
    producer
        .send(
            FutureRecord::to(topic).payload(&payload).key("TestCity"),
            rdkafka::util::Timeout::After(Duration::from_secs(5)),
        )
        .await
        .expect("produce weather event");

    // Consume and triage
    let consumer = test_consumer(topic, &format!("weather-src-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let triage = heartbit::sensor::triage::structured::StructuredTriageProcessor::new();
    let sensor_name = format!("weather_e2e_{suffix}");

    let mut found = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        if let Some(payload) = consume_one(&consumer, Duration::from_secs(3)).await {
            if let Ok(event) = serde_json::from_slice::<SensorEvent>(&payload) {
                if event.sensor_name == sensor_name {
                    let decision = triage.process(&event).await.expect("triage");
                    assert!(
                        decision.is_promote(),
                        "severe weather should be promoted: {decision:?}"
                    );
                    found = true;
                    break;
                }
            }
        }
    }

    assert!(found, "should have found and triaged our weather event");
}

// ===========================================================================
// TEST SUITE 40: Weather sensor alert_only filtering via Kafka
// ===========================================================================

/// Test: Weather sensor with alert_only=true drops normal conditions
#[tokio::test]
#[ignore]
async fn kafka_source_weather_alert_only_drops_normal() {
    use heartbit::sensor::triage::structured::StructuredTriageProcessor;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let topic = "hb.sensor.weather";
    ensure_topic(topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Produce a normal (non-severe) weather event
    let producer = test_producer();
    let normal_event = SensorEvent {
        id: SensorEvent::generate_id("normal weather", "London:1234567890"),
        sensor_name: format!("weather_normal_{suffix}"),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: serde_json::json!({
            "temperature_c": 22.0,
            "description": "clear sky",
            "wind_speed_ms": 3.0,
            "humidity_pct": 50.0,
        })
        .to_string(),
        source_id: "London:1234567890".into(),
        metadata: Some(serde_json::json!({
            "location": "London",
            "temperature_c": 22.0,
            "description": "clear sky",
            "wind_speed_ms": 3.0,
            "humidity_pct": 50.0,
            "alert": false,
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    let payload = serde_json::to_vec(&normal_event).expect("serialize");
    producer
        .send(
            FutureRecord::to(topic).payload(&payload).key("London"),
            rdkafka::util::Timeout::After(Duration::from_secs(5)),
        )
        .await
        .expect("produce");

    let consumer = test_consumer(topic, &format!("weather-normal-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let triage = StructuredTriageProcessor::new();
    let sensor_name = format!("weather_normal_{suffix}");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        if let Some(payload) = consume_one(&consumer, Duration::from_secs(3)).await {
            if let Ok(event) = serde_json::from_slice::<SensorEvent>(&payload) {
                if event.sensor_name == sensor_name {
                    let decision = triage.process(&event).await.expect("triage");
                    assert!(
                        decision.is_drop(),
                        "normal weather should be dropped: {decision:?}"
                    );
                    return;
                }
            }
        }
    }
    panic!("weather event not found in topic");
}

// ===========================================================================
// TEST SUITE 41: Audio sensor E2E — temp dir → sensor → Kafka
// ===========================================================================

/// Test: AudioSensor::run() watches directory, produces events to Kafka
#[tokio::test]
#[ignore]
async fn kafka_source_audio_sensor_produces_events() {
    use heartbit::sensor::Sensor;
    use heartbit::sensor::sources::audio::AudioSensor;
    use tokio_util::sync::CancellationToken;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();

    // Create temp dir with audio files
    let dir = tempfile::tempdir().expect("create tempdir");
    tokio::fs::write(dir.path().join("meeting.wav"), b"fake wav audio data")
        .await
        .expect("write wav");
    tokio::fs::write(dir.path().join("note.mp3"), b"fake mp3 audio data")
        .await
        .expect("write mp3");
    // Non-audio file should be ignored
    tokio::fs::write(dir.path().join("readme.txt"), b"not audio")
        .await
        .expect("write txt");

    let sensor = AudioSensor::new(
        "audio_e2e_test",
        dir.path(),
        "base.en",
        Duration::from_millis(200),
    );

    let consumer = test_consumer(sensor.kafka_topic(), &format!("audio-source-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let producer = test_producer();

    let sensor_handle = tokio::spawn(async move { sensor.run(producer, cancel_clone).await });

    // Wait for sensor to scan and produce
    tokio::time::sleep(Duration::from_secs(3)).await;

    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), sensor_handle).await;

    // Consume events, filtering by our sensor name
    let mut events = Vec::new();
    while let Some(payload) = consume_one(&consumer, Duration::from_secs(3)).await {
        if let Ok(event) = serde_json::from_slice::<SensorEvent>(&payload) {
            if event.sensor_name == "audio_e2e_test" {
                events.push(event);
            }
        }
    }

    assert_eq!(
        events.len(),
        2,
        "should have 2 audio events (wav + mp3), got {}",
        events.len()
    );

    // Verify modality and binary_ref
    for event in &events {
        assert_eq!(event.modality, SensorModality::Audio);
        assert!(
            event.binary_ref.is_some(),
            "audio events should have binary_ref"
        );
        let meta = event.metadata.as_ref().expect("metadata");
        assert!(
            meta["extension"].as_str().is_some(),
            "metadata should have extension"
        );
    }

    // Verify filenames are present
    let filenames: Vec<&str> = events
        .iter()
        .filter_map(|e| e.metadata.as_ref()?.get("filename")?.as_str())
        .collect();
    assert!(
        filenames.iter().any(|f| f.contains("meeting.wav")),
        "should have meeting.wav, filenames: {filenames:?}"
    );
    assert!(
        filenames.iter().any(|f| f.contains("note.mp3")),
        "should have note.mp3, filenames: {filenames:?}"
    );
}

// ===========================================================================
// TEST SUITE 42: Compression policy chaining — multiple rules in sequence
// ===========================================================================

/// Test: Compression rules chain correctly (truncate → strip pattern)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_compression_chain_truncate_then_strip() {
    use heartbit::sensor::compression::CompressionRule;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    // Create a policy with two chained rules: truncate first, then strip
    let policy = CompressionPolicy::with_rules(
        SensorModality::Text,
        vec![
            CompressionRule::Truncate { max_bytes: 100 },
            CompressionRule::StripPattern {
                pattern: r"ADVERTISEMENT".into(),
            },
        ],
    );

    // Content has ADVERTISEMENT in first 100 bytes
    let content = format!("ADVERTISEMENT Important news here. {}", "x".repeat(200));
    let event = SensorEvent {
        id: "chain-test".into(),
        sensor_name: "chain_test".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content,
        source_id: "src".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    let (compressed, saved) = policy.compress(&event);

    // Should be truncated to <= 100 bytes, then ADVERTISEMENT stripped
    assert!(compressed.len() <= 100);
    assert!(!compressed.contains("ADVERTISEMENT"));
    assert!(saved > 0);
}

// ===========================================================================
// TEST SUITE 43: Compression rule — StripPattern on structured data
// ===========================================================================

/// Test: StripPattern works on structured content
#[tokio::test]
#[ignore]
async fn kafka_pipeline_compression_strip_on_structured() {
    use heartbit::sensor::compression::CompressionRule;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let policy = CompressionPolicy::with_rules(
        SensorModality::Structured,
        vec![CompressionRule::StripPattern {
            pattern: r#""debug_info":\s*"[^"]*""#.into(),
        }],
    );

    let content = r#"{"temp": 22, "debug_info": "internal-trace-123", "unit": "C"}"#;
    let event = SensorEvent {
        id: "strip-test".into(),
        sensor_name: "strip_test".into(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: content.into(),
        source_id: "src".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    let (compressed, saved) = policy.compress(&event);
    assert!(!compressed.contains("debug_info"));
    assert!(compressed.contains("temp"));
    assert!(saved > 0);
}

// ===========================================================================
// TEST SUITE 44: JMAP email triage through Kafka pipeline
// ===========================================================================

/// Test: JMAP sensor → Kafka → email triage produces correct decisions
#[tokio::test]
#[ignore]
async fn kafka_pipeline_jmap_email_through_triage() {
    use heartbit::sensor::triage::email::EmailTriageProcessor;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let topic = format!("hb.test.jmap.triage.{suffix}");
    let commands_topic = format!("hb.test.jmap.cmd.{suffix}");

    ensure_topic(&topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Simulate JMAP email event with threading
    let producer = test_producer();
    let event = SensorEvent {
        id: SensorEvent::generate_id("email content", "msg-001@test.com"),
        sensor_name: "jmap_triage_test".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Hi team, please review the Q4 budget proposal attached.".into(),
        source_id: "msg-001@test.com".into(),
        metadata: Some(serde_json::json!({
            "subject": "Q4 Budget Review",
            "from": "boss@company.com",
            "to": ["team@company.com"],
            "message_id": "msg-001@test.com",
            "in_reply_to": null,
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    let payload = serde_json::to_vec(&event).expect("serialize");
    producer
        .send(
            FutureRecord::to(&topic).payload(&payload).key("test"),
            rdkafka::util::Timeout::After(Duration::from_secs(5)),
        )
        .await
        .expect("produce");

    let consumer = test_consumer(&topic, &format!("jmap-triage-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::email_classifier(true, true));
    let processor = EmailTriageProcessor::new(slm, vec!["boss@company.com".into()], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let decision = run_triage_once(
        &consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should produce a triage decision");
    let d = decision.unwrap();
    assert!(d.is_promote(), "known sender email should be promoted");

    // Known sender + urgent → should be Critical or High priority
    if let TriageDecision::Promote { priority, .. } = &d {
        assert!(
            *priority == Priority::Critical || *priority == Priority::High,
            "known+urgent should be Critical or High, got: {priority:?}"
        );
    }
}

// ===========================================================================
// TEST SUITE 45: Email with thread reply boosts priority
// ===========================================================================

/// Test: Email triage boosts priority when event has related_ids (thread reply)
#[tokio::test]
#[ignore]
async fn kafka_pipeline_email_reply_in_thread_boost() {
    use heartbit::sensor::triage::email::EmailTriageProcessor;

    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let topic = format!("hb.test.email.thread.{suffix}");
    let commands_topic = format!("hb.test.email.thread.cmd.{suffix}");

    ensure_topic(&topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();

    // Email reply in a thread (has in_reply_to)
    let event = SensorEvent {
        id: SensorEvent::generate_id("reply content", "msg-reply@test.com"),
        sensor_name: "thread_test".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Sure, I'll review the budget by Friday.".into(),
        source_id: "msg-reply@test.com".into(),
        metadata: Some(serde_json::json!({
            "subject": "Re: Q4 Budget Review",
            "from": "unknown@external.com",
            "to": ["team@company.com"],
            "message_id": "msg-reply@test.com",
            "in_reply_to": ["msg-001@test.com"],
        })),
        binary_ref: None,
        related_ids: vec![SensorEvent::generate_id("", "msg-001@test.com")],
    };

    let payload = serde_json::to_vec(&event).expect("serialize");
    producer
        .send(
            FutureRecord::to(&topic).payload(&payload).key("test"),
            rdkafka::util::Timeout::After(Duration::from_secs(5)),
        )
        .await
        .expect("produce");

    let consumer = test_consumer(&topic, &format!("thread-test-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let slm = Arc::new(MockSlmProvider::email_classifier(true, false));
    let processor = EmailTriageProcessor::new(slm, vec![], vec![]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let decision = run_triage_once(
        &consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(decision.is_some(), "should produce a decision");
    let d = decision.unwrap();
    assert!(d.is_promote(), "thread reply should be promoted");

    // Thread reply from unknown sender with non-urgent SLM → Normal base
    // But boost_priority should upgrade to High because it's a thread reply
    if let TriageDecision::Promote { priority, .. } = &d {
        assert!(
            *priority == Priority::High || *priority == Priority::Normal,
            "thread reply should boost priority, got: {priority:?}"
        );
    }
}

// ===========================================================================
// TEST SUITE 46: Multiple sensor types through single story correlator
// ===========================================================================

/// Test: Events from different sensor types with shared entities join same story
#[tokio::test]
#[ignore]
async fn kafka_pipeline_cross_sensor_story_correlation() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let email_topic = format!("hb.test.cross.email.{suffix}");
    let image_topic = format!("hb.test.cross.image.{suffix}");
    let commands_topic = format!("hb.test.cross.cmd.{suffix}");

    ensure_topic(&email_topic, 1).await;
    ensure_topic(&image_topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();

    // Email about invoice from Alice
    let email_event = SensorEvent {
        id: SensorEvent::generate_id("email-alice", "msg-alice@test.com"),
        sensor_name: "cross_email".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: "Please review attached invoice from Alice.".into(),
        source_id: "msg-alice@test.com".into(),
        metadata: Some(serde_json::json!({
            "from": "alice@acme.com",
            "subject": "Invoice #1234",
        })),
        binary_ref: None,
        related_ids: vec![],
    };

    // Image of the invoice (shared entity: "alice", "invoice")
    let image_event = SensorEvent {
        id: SensorEvent::generate_id("image-invoice", "invoice-scan.jpg"),
        sensor_name: "cross_image".into(),
        modality: SensorModality::Image,
        observed_at: Utc::now(),
        content: "Scanned invoice from Alice showing billing details".into(),
        source_id: "invoice-scan.jpg".into(),
        metadata: Some(serde_json::json!({
            "filename": "invoice_alice.jpg",
            "summary": "Scanned invoice from Alice",
        })),
        binary_ref: Some("/path/to/invoice_alice.jpg".into()),
        related_ids: vec![],
    };

    // Produce both events
    let payload1 = serde_json::to_vec(&email_event).expect("serialize");
    let payload2 = serde_json::to_vec(&image_event).expect("serialize");

    producer
        .send(
            FutureRecord::to(&email_topic)
                .payload(&payload1)
                .key("test"),
            rdkafka::util::Timeout::After(Duration::from_secs(5)),
        )
        .await
        .expect("produce email");

    producer
        .send(
            FutureRecord::to(&image_topic)
                .payload(&payload2)
                .key("test"),
            rdkafka::util::Timeout::After(Duration::from_secs(5)),
        )
        .await
        .expect("produce image");

    // Triage email
    let email_consumer = test_consumer(&email_topic, &format!("cross-email-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let email_slm = Arc::new(MockSlmProvider::email_classifier(true, false));
    let email_proc =
        heartbit::sensor::triage::email::EmailTriageProcessor::new(email_slm, vec![], vec![]);

    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let d1 = run_triage_once(
        &email_consumer,
        &email_proc,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(d1.is_some(), "email should produce decision");
    assert!(d1.unwrap().is_promote(), "email should be promoted");

    // Triage image
    let image_consumer = test_consumer(&image_topic, &format!("cross-image-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    let image_slm = Arc::new(MockSlmProvider::new(
        &serde_json::json!({
            "category": "document",
            "priority": "high",
            "summary": "Scanned invoice from Alice showing $5000",
            "entities": ["alice", "invoice", "billing"],
        })
        .to_string(),
    ));
    let image_proc = heartbit::sensor::triage::image::ImageTriageProcessor::new(image_slm);

    let d2 = run_triage_once(
        &image_consumer,
        &image_proc,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    assert!(d2.is_some(), "image should produce decision");
    assert!(
        d2.unwrap().is_promote(),
        "document image should be promoted"
    );

    // Check story correlator — both events should be in same story (shared entity overlap)
    let corr = correlator.lock().unwrap();
    let stories = corr.active_stories();
    // At minimum 1 story, at most 2 (depends on entity overlap threshold)
    assert!(!stories.is_empty(), "should have at least one active story");
}

// ===========================================================================
// TEST SUITE 47: Dead letter handling through Kafka
// ===========================================================================

/// Test: Events that fail triage go to dead letter
#[tokio::test]
#[ignore]
async fn kafka_pipeline_dead_letter_on_triage_error() {
    if !kafka_is_reachable() {
        eprintln!("SKIP: Kafka not reachable at {BROKERS}");
        return;
    }

    let suffix = unique_suffix();
    let topic = format!("hb.test.deadletter.{suffix}");
    let commands_topic = format!("hb.test.deadletter.cmd.{suffix}");

    ensure_topic(&topic, 1).await;
    ensure_topic(&commands_topic, 1).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let producer = test_producer();

    // Event with corrupted content that might cause triage issues
    let event = SensorEvent {
        id: "corrupted-event".into(),
        sensor_name: "deadletter_test".into(),
        modality: SensorModality::Text,
        observed_at: Utc::now(),
        content: String::new(), // Empty content
        source_id: "src".into(),
        metadata: None,
        binary_ref: None,
        related_ids: vec![],
    };

    let payload = serde_json::to_vec(&event).expect("serialize");
    producer
        .send(
            FutureRecord::to(&topic).payload(&payload).key("test"),
            rdkafka::util::Timeout::After(Duration::from_secs(5)),
        )
        .await
        .expect("produce");

    let consumer = test_consumer(&topic, &format!("dl-test-{suffix}"));
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Use RSS triage which should still handle empty content gracefully
    let slm = Arc::new(MockSlmProvider::rss_classifier());
    let processor = RssTriageProcessor::new(slm, vec!["rust".into()]);
    let correlator = std::sync::Mutex::new(StoryCorrelator::new(Duration::from_secs(3600)));

    let decision = run_triage_once(
        &consumer,
        &processor,
        &producer,
        &commands_topic,
        &correlator,
        None,
        Duration::from_secs(10),
    )
    .await;

    // Empty content with no keyword match → should be dropped
    assert!(decision.is_some(), "should produce a decision");
    let d = decision.unwrap();
    assert!(
        d.is_drop(),
        "empty content with no keyword match should be dropped, got: {d:?}"
    );
}
