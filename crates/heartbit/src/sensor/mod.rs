pub mod compression;
pub mod manager;
pub mod metrics;
pub mod perception;
pub mod routing;
pub mod sources;
pub mod stories;
pub mod triage;

use std::future::Future;
use std::pin::Pin;

use chrono::{DateTime, Utc};
use rdkafka::producer::FutureProducer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::Error;

/// Sensory modality — the type of information a sensor captures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorModality {
    /// Email body, RSS content, chat messages.
    Text,
    /// Photos, screenshots, documents-as-images.
    Image,
    /// Voice notes, calls, podcasts.
    Audio,
    /// Weather JSON, GPS coordinates, API responses.
    Structured,
}

impl std::fmt::Display for SensorModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensorModality::Text => write!(f, "text"),
            SensorModality::Image => write!(f, "image"),
            SensorModality::Audio => write!(f, "audio"),
            SensorModality::Structured => write!(f, "structured"),
        }
    }
}

/// A single event observed by a sensor.
///
/// This is the fundamental data unit flowing through the sensor pipeline.
/// Each sensor produces `SensorEvent` values to its dedicated Kafka topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorEvent {
    /// Unique ID for dedup (FNV-1a hash of content + source_id).
    pub id: String,
    /// Human name of the sensor that produced this event ("work_email", "tech_rss").
    pub sensor_name: String,
    /// Modality of the observed data.
    pub modality: SensorModality,
    /// When the event was observed.
    pub observed_at: DateTime<Utc>,
    /// Text content (or transcription/description for non-text modalities).
    pub content: String,
    /// Provenance identifier: URL, Message-ID, file hash, etc.
    pub source_id: String,
    /// Sensor-specific structured data (email headers, EXIF data, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Storage path for large binary data (images, audio). Not inlined in Kafka.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_ref: Option<String>,
    /// Links to other events (email thread references, etc.).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_ids: Vec<String>,
}

impl SensorEvent {
    /// Generate a deterministic ID from content and source_id using FNV-1a.
    pub fn generate_id(content: &str, source_id: &str) -> String {
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in content.as_bytes().iter().chain(source_id.as_bytes()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("{hash:016x}")
    }
}

/// A sensor is a long-running async task that polls or subscribes to an
/// external source and produces `SensorEvent` values to its Kafka topic.
pub trait Sensor: Send + Sync {
    /// Human-readable name for this sensor instance.
    fn name(&self) -> &str;

    /// The modality of data this sensor produces.
    fn modality(&self) -> SensorModality;

    /// The Kafka topic this sensor writes to (e.g., "hb.sensor.rss").
    fn kafka_topic(&self) -> &str;

    /// Run the sensor loop. Blocks until cancellation.
    fn run(
        &self,
        producer: FutureProducer,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_modality_serde_roundtrip() {
        for modality in [
            SensorModality::Text,
            SensorModality::Image,
            SensorModality::Audio,
            SensorModality::Structured,
        ] {
            let json = serde_json::to_string(&modality).unwrap();
            let back: SensorModality = serde_json::from_str(&json).unwrap();
            assert_eq!(back, modality);
        }
    }

    #[test]
    fn sensor_modality_snake_case() {
        assert_eq!(
            serde_json::to_string(&SensorModality::Text).unwrap(),
            r#""text""#
        );
        assert_eq!(
            serde_json::to_string(&SensorModality::Image).unwrap(),
            r#""image""#
        );
        assert_eq!(
            serde_json::to_string(&SensorModality::Audio).unwrap(),
            r#""audio""#
        );
        assert_eq!(
            serde_json::to_string(&SensorModality::Structured).unwrap(),
            r#""structured""#
        );
    }

    #[test]
    fn sensor_modality_display() {
        assert_eq!(SensorModality::Text.to_string(), "text");
        assert_eq!(SensorModality::Image.to_string(), "image");
        assert_eq!(SensorModality::Audio.to_string(), "audio");
        assert_eq!(SensorModality::Structured.to_string(), "structured");
    }

    #[test]
    fn sensor_event_serde_roundtrip() {
        let event = SensorEvent {
            id: "abc123".into(),
            sensor_name: "tech_rss".into(),
            modality: SensorModality::Text,
            observed_at: Utc::now(),
            content: "Rust 2026 edition released".into(),
            source_id: "https://example.com/article/1".into(),
            metadata: Some(serde_json::json!({"feed": "hn"})),
            binary_ref: None,
            related_ids: vec![],
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: SensorEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "abc123");
        assert_eq!(back.sensor_name, "tech_rss");
        assert_eq!(back.modality, SensorModality::Text);
        assert_eq!(back.content, "Rust 2026 edition released");
    }

    #[test]
    fn sensor_event_optional_fields_omitted() {
        let event = SensorEvent {
            id: "def456".into(),
            sensor_name: "test".into(),
            modality: SensorModality::Structured,
            observed_at: Utc::now(),
            content: "{}".into(),
            source_id: "test-source".into(),
            metadata: None,
            binary_ref: None,
            related_ids: vec![],
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("metadata"), "json: {json}");
        assert!(!json.contains("binary_ref"), "json: {json}");
        assert!(!json.contains("related_ids"), "json: {json}");
    }

    #[test]
    fn sensor_event_with_binary_ref() {
        let event = SensorEvent {
            id: "img001".into(),
            sensor_name: "scanner".into(),
            modality: SensorModality::Image,
            observed_at: Utc::now(),
            content: "Photo of invoice".into(),
            source_id: "file:///inbox/invoice.jpg".into(),
            metadata: None,
            binary_ref: Some("/storage/images/invoice.jpg".into()),
            related_ids: vec![],
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("binary_ref"), "json: {json}");
        let back: SensorEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.binary_ref.as_deref(),
            Some("/storage/images/invoice.jpg")
        );
    }

    #[test]
    fn sensor_event_with_related_ids() {
        let event = SensorEvent {
            id: "email001".into(),
            sensor_name: "work_email".into(),
            modality: SensorModality::Text,
            observed_at: Utc::now(),
            content: "Re: Project update".into(),
            source_id: "msg-id-001@example.com".into(),
            metadata: None,
            binary_ref: None,
            related_ids: vec!["msg-id-000@example.com".into()],
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("related_ids"), "json: {json}");
        let back: SensorEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.related_ids, vec!["msg-id-000@example.com"]);
    }

    #[test]
    fn generate_id_deterministic() {
        let id1 = SensorEvent::generate_id("hello", "source1");
        let id2 = SensorEvent::generate_id("hello", "source1");
        assert_eq!(id1, id2);
    }

    #[test]
    fn generate_id_different_inputs() {
        let id1 = SensorEvent::generate_id("hello", "source1");
        let id2 = SensorEvent::generate_id("world", "source1");
        let id3 = SensorEvent::generate_id("hello", "source2");
        assert_ne!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn generate_id_hex_format() {
        let id = SensorEvent::generate_id("test", "src");
        assert_eq!(id.len(), 16);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn backward_compat_missing_optional_fields() {
        let json = r#"{
            "id": "test",
            "sensor_name": "s",
            "modality": "text",
            "observed_at": "2026-01-01T00:00:00Z",
            "content": "hello",
            "source_id": "src"
        }"#;
        let event: SensorEvent = serde_json::from_str(json).unwrap();
        assert!(event.metadata.is_none());
        assert!(event.binary_ref.is_none());
        assert!(event.related_ids.is_empty());
    }
}
