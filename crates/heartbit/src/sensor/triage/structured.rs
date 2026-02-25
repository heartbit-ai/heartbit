use std::future::Future;
use std::pin::Pin;

use crate::Error;
use crate::sensor::triage::{Priority, TriageDecision, TriageProcessor};
use crate::sensor::{SensorEvent, SensorModality};

/// Rule-based triage processor for structured sensor data.
///
/// Unlike the text/image/audio processors, this processor does NOT use an SLM.
/// Structured data (weather JSON, GPS coordinates, API responses) follows
/// deterministic rules that can be evaluated without language model inference.
///
/// Processing pipeline:
/// 1. Parse event metadata for known fields.
/// 2. Weather events: `alert: true` -> Promote High; no alert -> Drop (informational).
/// 3. GPS events (future): geofence triggers -> Promote Normal.
/// 4. Fallback: Drop with "no actionable data".
pub struct StructuredTriageProcessor;

impl StructuredTriageProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StructuredTriageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract a string field from event metadata.
fn meta_str<'a>(metadata: Option<&'a serde_json::Value>, key: &str) -> Option<&'a str> {
    metadata.and_then(|m| m.get(key)).and_then(|v| v.as_str())
}

/// Extract a boolean field from event metadata.
fn meta_bool(metadata: Option<&serde_json::Value>, key: &str) -> Option<bool> {
    metadata.and_then(|m| m.get(key)).and_then(|v| v.as_bool())
}

/// Build a weather triage summary string.
fn weather_summary(is_alert: bool, description: &str, location: &str) -> String {
    if is_alert {
        format!("Weather alert: {description} in {location}")
    } else {
        format!("Weather: {description} in {location}")
    }
}

/// Estimate token cost for downstream LLM processing.
///
/// Rough heuristic: content length / 4 (chars-to-tokens) + 256 (system overhead).
fn estimate_tokens(content: &str) -> usize {
    content.len() / 4 + 256
}

/// Process a weather event based on metadata rules.
fn triage_weather(event: &SensorEvent) -> TriageDecision {
    let metadata = event.metadata.as_ref();

    let is_alert = meta_bool(metadata, "alert").unwrap_or(false);
    let description = meta_str(metadata, "description").unwrap_or("unknown");
    let location = meta_str(metadata, "location").unwrap_or("unknown");

    let summary = weather_summary(is_alert, description, location);
    let entities = vec![location.to_string()];
    let estimated_tokens = estimate_tokens(&event.content);

    if is_alert {
        TriageDecision::Promote {
            priority: Priority::High,
            summary,
            extracted_entities: entities,
            estimated_tokens,
            action_categories: vec![],
            action_hints: vec![],
            has_attachments: false,
            sender: None,
            subject: None,
            message_ref: None,
        }
    } else {
        TriageDecision::Drop {
            reason: format!("informational weather reading for {location}"),
        }
    }
}

impl TriageProcessor for StructuredTriageProcessor {
    fn modality(&self) -> SensorModality {
        SensorModality::Structured
    }

    fn source_topic(&self) -> &str {
        "hb.sensor.weather"
    }

    fn process(
        &self,
        event: &SensorEvent,
    ) -> Pin<Box<dyn Future<Output = Result<TriageDecision, Error>> + Send + '_>> {
        let decision = if event.sensor_name.contains("weather") {
            triage_weather(event)
        } else {
            // Future: GPS geofence, API response triage, etc.
            TriageDecision::Drop {
                reason: "no actionable data in structured event".into(),
            }
        };

        Box::pin(async move { Ok(decision) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_weather_event(
        location: &str,
        description: &str,
        temp: f64,
        alert: bool,
    ) -> SensorEvent {
        let content = serde_json::json!({
            "temperature_c": temp,
            "description": description,
            "wind_speed_ms": 5.0,
            "humidity_pct": 65.0,
        });
        let content_str = serde_json::to_string(&content).expect("serializable");

        SensorEvent {
            id: SensorEvent::generate_id(&content_str, &format!("{location}:123")),
            sensor_name: "weather_sensor".into(),
            modality: SensorModality::Structured,
            observed_at: Utc::now(),
            content: content_str,
            source_id: format!("{location}:123"),
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

    // --- Trait property tests ---

    #[test]
    fn processor_modality_is_structured() {
        let processor = StructuredTriageProcessor::new();
        assert_eq!(processor.modality(), SensorModality::Structured);
    }

    #[test]
    fn processor_source_topic() {
        let processor = StructuredTriageProcessor::new();
        assert_eq!(processor.source_topic(), "hb.sensor.weather");
    }

    #[test]
    fn processor_default_trait() {
        let processor = StructuredTriageProcessor::default();
        assert_eq!(processor.modality(), SensorModality::Structured);
    }

    // --- Weather alert triage tests ---

    #[tokio::test]
    async fn weather_alert_promotes_high() {
        let processor = StructuredTriageProcessor::new();
        let event = make_weather_event("London", "thunderstorm", 25.0, true);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote(), "alert should be promoted");
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::High);
        }
    }

    #[tokio::test]
    async fn normal_weather_dropped() {
        let processor = StructuredTriageProcessor::new();
        let event = make_weather_event("Paris", "clear sky", 22.0, false);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_drop(), "normal weather should be dropped");
        if let TriageDecision::Drop { reason } = &decision {
            assert!(
                reason.contains("informational"),
                "drop reason should mention informational, got: {reason}"
            );
        }
    }

    #[tokio::test]
    async fn alert_summary_format() {
        let processor = StructuredTriageProcessor::new();
        let event = make_weather_event("Berlin", "heavy rain", 18.0, true);

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { summary, .. } = &decision {
            assert_eq!(summary, "Weather alert: heavy rain in Berlin");
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    #[tokio::test]
    async fn normal_weather_summary_format_via_helper() {
        // Test the summary helper directly for non-alert case
        let summary = weather_summary(false, "clear sky", "Tokyo");
        assert_eq!(summary, "Weather: clear sky in Tokyo");
    }

    #[tokio::test]
    async fn entities_contain_location() {
        let processor = StructuredTriageProcessor::new();
        let event = make_weather_event("NYC", "storm warning", 30.0, true);

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            extracted_entities, ..
        } = &decision
        {
            assert_eq!(extracted_entities, &["NYC"]);
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    #[tokio::test]
    async fn estimated_tokens_reasonable() {
        let processor = StructuredTriageProcessor::new();
        let event = make_weather_event("Oslo", "blizzard", 5.0, true);

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            estimated_tokens, ..
        } = &decision
        {
            // content is ~80 bytes, so tokens ~ 80/4 + 256 = 276
            assert!(
                *estimated_tokens > 256,
                "estimated tokens should include base overhead, got: {estimated_tokens}"
            );
            assert!(
                *estimated_tokens < 500,
                "estimated tokens should be reasonable, got: {estimated_tokens}"
            );
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    // --- Missing metadata tests ---

    #[tokio::test]
    async fn missing_metadata_drops_with_reason() {
        let processor = StructuredTriageProcessor::new();
        let event = SensorEvent {
            id: "test-id".into(),
            sensor_name: "weather_sensor".into(),
            modality: SensorModality::Structured,
            observed_at: Utc::now(),
            content: "{}".into(),
            source_id: "unknown:0".into(),
            metadata: None,
            binary_ref: None,
            related_ids: vec![],
        };

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_drop(),
            "missing metadata should result in drop (alert defaults to false)"
        );
    }

    #[tokio::test]
    async fn missing_alert_field_defaults_to_false() {
        let processor = StructuredTriageProcessor::new();
        let event = SensorEvent {
            id: "test-id".into(),
            sensor_name: "weather_sensor".into(),
            modality: SensorModality::Structured,
            observed_at: Utc::now(),
            content: "{}".into(),
            source_id: "London:0".into(),
            metadata: Some(serde_json::json!({
                "location": "London",
                "temperature_c": 20.0,
                "description": "clear",
            })),
            binary_ref: None,
            related_ids: vec![],
        };

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_drop(),
            "missing alert field should default to false (drop)"
        );
    }

    // --- Non-weather structured event tests ---

    #[tokio::test]
    async fn non_weather_structured_event_dropped() {
        let processor = StructuredTriageProcessor::new();
        let event = SensorEvent {
            id: "gps-001".into(),
            sensor_name: "gps_tracker".into(),
            modality: SensorModality::Structured,
            observed_at: Utc::now(),
            content: r#"{"lat": 51.5, "lon": -0.1}"#.into(),
            source_id: "gps:device1".into(),
            metadata: Some(serde_json::json!({"device": "phone"})),
            binary_ref: None,
            related_ids: vec![],
        };

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_drop());
        if let TriageDecision::Drop { reason } = &decision {
            assert!(reason.contains("no actionable data"), "reason: {reason}");
        }
    }

    // --- No LLM provider verification ---

    #[test]
    fn processor_has_no_provider_field() {
        // StructuredTriageProcessor is a unit struct with no fields.
        // This test verifies it requires no LLM provider by construction.
        let _processor = StructuredTriageProcessor;
        // If this compiles, the processor has no fields (including no provider).
        assert_eq!(std::mem::size_of::<StructuredTriageProcessor>(), 0);
    }

    // --- Helper function unit tests ---

    #[test]
    fn meta_str_extracts_field() {
        let meta = serde_json::json!({"location": "London"});
        assert_eq!(meta_str(Some(&meta), "location"), Some("London"));
    }

    #[test]
    fn meta_str_returns_none_for_missing() {
        let meta = serde_json::json!({"location": "London"});
        assert_eq!(meta_str(Some(&meta), "description"), None);
    }

    #[test]
    fn meta_str_returns_none_when_no_metadata() {
        assert_eq!(meta_str(None, "location"), None);
    }

    #[test]
    fn meta_bool_extracts_true() {
        let meta = serde_json::json!({"alert": true});
        assert_eq!(meta_bool(Some(&meta), "alert"), Some(true));
    }

    #[test]
    fn meta_bool_extracts_false() {
        let meta = serde_json::json!({"alert": false});
        assert_eq!(meta_bool(Some(&meta), "alert"), Some(false));
    }

    #[test]
    fn meta_bool_returns_none_for_non_bool() {
        let meta = serde_json::json!({"alert": "yes"});
        assert_eq!(meta_bool(Some(&meta), "alert"), None);
    }

    #[test]
    fn meta_bool_returns_none_when_no_metadata() {
        assert_eq!(meta_bool(None, "alert"), None);
    }

    #[test]
    fn weather_summary_alert() {
        let s = weather_summary(true, "heavy snow", "Moscow");
        assert_eq!(s, "Weather alert: heavy snow in Moscow");
    }

    #[test]
    fn weather_summary_normal() {
        let s = weather_summary(false, "partly cloudy", "Sydney");
        assert_eq!(s, "Weather: partly cloudy in Sydney");
    }

    #[test]
    fn estimate_tokens_formula() {
        // 400 bytes / 4 + 256 = 356
        let content = "A".repeat(400);
        assert_eq!(estimate_tokens(&content), 356);
    }

    #[test]
    fn estimate_tokens_empty_content() {
        // 0 / 4 + 256 = 256
        assert_eq!(estimate_tokens(""), 256);
    }

    #[test]
    fn triage_weather_alert_directly() {
        let event = make_weather_event("Rome", "tornado warning", 28.0, true);
        let decision = triage_weather(&event);
        assert!(decision.is_promote());
        if let TriageDecision::Promote {
            priority,
            summary,
            extracted_entities,
            ..
        } = &decision
        {
            assert_eq!(*priority, Priority::High);
            assert!(summary.contains("tornado warning"));
            assert!(summary.contains("Rome"));
            assert!(extracted_entities.contains(&"Rome".to_string()));
        }
    }

    #[test]
    fn triage_weather_normal_directly() {
        let event = make_weather_event("Madrid", "sunny", 30.0, false);
        let decision = triage_weather(&event);
        assert!(decision.is_drop());
        if let TriageDecision::Drop { reason } = &decision {
            assert!(
                reason.contains("Madrid"),
                "reason should mention location: {reason}"
            );
        }
    }
}
