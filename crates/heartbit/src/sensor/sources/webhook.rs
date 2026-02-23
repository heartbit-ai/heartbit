use std::future::Future;
use std::pin::Pin;

use chrono::Utc;
use hmac::{Hmac, Mac};
use rdkafka::producer::FutureProducer;
use sha2::Sha256;
use subtle::ConstantTimeEq;
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::sensor::{Sensor, SensorEvent, SensorModality};

type HmacSha256 = Hmac<Sha256>;

/// Webhook sensor. Unlike polling sensors (RSS, weather), this sensor is
/// push-based: it provides an Axum route handler that receives HTTP POST
/// requests. The `run()` method simply blocks until cancellation, keeping the
/// sensor alive for the lifetime of the daemon.
///
/// Event production happens via `build_event_from_payload()`, which is called
/// from the Axum handler that routes incoming POST requests to the correct
/// webhook sensor instance.
pub struct WebhookSensor {
    name: String,
    path: String,
    secret: Option<String>,
}

/// Kafka topic for all webhook sensor events.
const WEBHOOK_TOPIC: &str = "hb.sensor.webhook";

impl WebhookSensor {
    /// Create a new webhook sensor.
    ///
    /// - `name`: Human-readable name for this webhook (e.g., "github_webhooks").
    /// - `path`: URL path this webhook listens on (e.g., "/webhooks/github").
    /// - `secret`: Optional HMAC-SHA256 secret for signature verification.
    pub fn new(name: impl Into<String>, path: impl Into<String>, secret: Option<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            secret,
        }
    }

    /// The URL path this webhook listens on.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The HMAC-SHA256 secret, if configured.
    pub fn secret(&self) -> Option<&str> {
        self.secret.as_deref()
    }

    /// Build a `SensorEvent` from a raw webhook payload.
    ///
    /// - `payload`: the raw HTTP request body bytes.
    ///
    /// Returns an error if the payload is not valid UTF-8 or cannot be
    /// processed.
    pub fn build_event_from_payload(&self, payload: &[u8]) -> Result<SensorEvent, Error> {
        let content = std::str::from_utf8(payload)
            .map_err(|e| Error::Sensor(format!("webhook payload is not valid UTF-8: {e}")))?
            .to_string();

        let event_hash = SensorEvent::generate_id(&content, &self.name);
        let source_id = format!("{}:{}", self.name, event_hash);

        let id = SensorEvent::generate_id(&content, &source_id);

        Ok(SensorEvent {
            id,
            sensor_name: self.name.clone(),
            modality: SensorModality::Structured,
            observed_at: Utc::now(),
            content,
            source_id,
            metadata: Some(serde_json::json!({
                "source": self.name,
                "content_type": "application/json",
            })),
            binary_ref: None,
            related_ids: vec![],
        })
    }

    /// Verify a webhook signature against the payload.
    ///
    /// Expected format: `sha256=<hex_digest>` (GitHub webhook style).
    /// Computes HMAC-SHA256 with the given secret and compares the hex digest
    /// in constant time to prevent timing attacks.
    pub fn verify_signature(payload: &[u8], signature: &str, secret: &str) -> bool {
        // Validate format: must start with "sha256=" followed by a hex string
        let Some(hex_part) = signature.strip_prefix("sha256=") else {
            return false;
        };

        // Hex string must be non-empty and contain only hex digits
        if hex_part.is_empty() || !hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
            return false;
        }

        // SHA-256 produces 32 bytes = 64 hex chars
        if hex_part.len() != 64 {
            return false;
        }

        // Compute expected HMAC-SHA256
        let Ok(mut mac) = HmacSha256::new_from_slice(secret.as_bytes()) else {
            return false;
        };
        mac.update(payload);
        let expected = hex::encode(mac.finalize().into_bytes());

        // Constant-time comparison to prevent timing attacks
        expected.as_bytes().ct_eq(hex_part.as_bytes()).into()
    }

    /// Compute the Kafka key for this event.
    ///
    /// Format: `{webhook_name}:{event_hash}`
    ///
    /// For UTF-8 payloads, the hash is computed from the text content.
    /// For binary payloads, the hash is computed from hex-encoded bytes
    /// to ensure distinct keys for distinct binary inputs.
    pub fn kafka_key(&self, payload: &[u8]) -> String {
        let content_for_hash = match std::str::from_utf8(payload) {
            Ok(s) => s.to_string(),
            Err(_) => {
                // For binary payloads, use a hex prefix + length as hash input
                // to ensure distinct keys without allocating a full hex encoding.
                let prefix: String = payload
                    .iter()
                    .take(64)
                    .map(|b| format!("{b:02x}"))
                    .collect();
                format!("binary:{prefix}:len={}", payload.len())
            }
        };
        let event_hash = SensorEvent::generate_id(&content_for_hash, &self.name);
        format!("{}:{}", self.name, event_hash)
    }
}

impl Sensor for WebhookSensor {
    fn name(&self) -> &str {
        &self.name
    }

    fn modality(&self) -> SensorModality {
        SensorModality::Structured
    }

    fn kafka_topic(&self) -> &str {
        WEBHOOK_TOPIC
    }

    fn run(
        &self,
        _producer: FutureProducer,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        // Push-based sensor: just wait for cancellation.
        // The actual event production happens via build_event_from_payload(),
        // called from the Axum HTTP handler.
        Box::pin(async move {
            cancel.cancelled().await;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Sensor trait property tests ---

    #[test]
    fn sensor_name() {
        let sensor = WebhookSensor::new("github_hooks", "/webhooks/github", None);
        assert_eq!(sensor.name(), "github_hooks");
    }

    #[test]
    fn sensor_modality_is_structured() {
        let sensor = WebhookSensor::new("test", "/test", None);
        assert_eq!(sensor.modality(), SensorModality::Structured);
    }

    #[test]
    fn sensor_kafka_topic() {
        let sensor = WebhookSensor::new("test", "/test", None);
        assert_eq!(sensor.kafka_topic(), "hb.sensor.webhook");
    }

    // --- Accessor tests ---

    #[test]
    fn path_accessor() {
        let sensor = WebhookSensor::new("gh", "/webhooks/github", None);
        assert_eq!(sensor.path(), "/webhooks/github");
    }

    #[test]
    fn secret_accessor_some() {
        let sensor = WebhookSensor::new("gh", "/webhooks/github", Some("mysecret".into()));
        assert_eq!(sensor.secret(), Some("mysecret"));
    }

    #[test]
    fn secret_accessor_none() {
        let sensor = WebhookSensor::new("gh", "/webhooks/github", None);
        assert_eq!(sensor.secret(), None);
    }

    // --- build_event_from_payload tests ---

    #[test]
    fn build_event_basic() {
        let sensor = WebhookSensor::new("github", "/webhooks/github", None);
        let payload = br#"{"action":"opened","number":42}"#;

        let event = sensor
            .build_event_from_payload(payload)
            .expect("should build event");

        assert_eq!(event.sensor_name, "github");
        assert_eq!(event.modality, SensorModality::Structured);
        assert_eq!(event.content, r#"{"action":"opened","number":42}"#);
        assert!(event.binary_ref.is_none());
        assert!(event.related_ids.is_empty());
    }

    #[test]
    fn build_event_metadata_fields() {
        let sensor = WebhookSensor::new("slack", "/webhooks/slack", None);
        let payload = br#"{"type":"message","text":"hello"}"#;

        let event = sensor
            .build_event_from_payload(payload)
            .expect("should build event");

        let meta = event.metadata.expect("metadata should be present");
        assert_eq!(meta["source"], "slack");
        assert_eq!(meta["content_type"], "application/json");
    }

    #[test]
    fn build_event_source_id_format() {
        let sensor = WebhookSensor::new("gh_hooks", "/hooks/gh", None);
        let payload = b"test_payload";

        let event = sensor
            .build_event_from_payload(payload)
            .expect("should build event");

        // source_id format: "{webhook_name}:{event_hash}"
        assert!(
            event.source_id.starts_with("gh_hooks:"),
            "source_id should start with webhook name: {}",
            event.source_id
        );
        // The hash part should be a 16-char hex string
        let hash_part = event.source_id.strip_prefix("gh_hooks:").unwrap_or("");
        assert_eq!(hash_part.len(), 16, "hash should be 16 hex chars");
        assert!(
            hash_part.chars().all(|c| c.is_ascii_hexdigit()),
            "hash should be hex: {hash_part}"
        );
    }

    #[test]
    fn build_event_deterministic_id() {
        let sensor = WebhookSensor::new("test", "/test", None);
        let payload = br#"{"key":"value"}"#;

        let event1 = sensor
            .build_event_from_payload(payload)
            .expect("should build event");
        let event2 = sensor
            .build_event_from_payload(payload)
            .expect("should build event");

        assert_eq!(event1.id, event2.id, "same payload should produce same ID");
        assert_eq!(
            event1.source_id, event2.source_id,
            "same payload should produce same source_id"
        );
    }

    #[test]
    fn build_event_different_payloads_different_ids() {
        let sensor = WebhookSensor::new("test", "/test", None);

        let event1 = sensor
            .build_event_from_payload(b"payload_one")
            .expect("should build event");
        let event2 = sensor
            .build_event_from_payload(b"payload_two")
            .expect("should build event");

        assert_ne!(
            event1.id, event2.id,
            "different payloads should produce different IDs"
        );
    }

    #[test]
    fn build_event_empty_payload() {
        let sensor = WebhookSensor::new("test", "/test", None);
        let payload = b"";

        let event = sensor
            .build_event_from_payload(payload)
            .expect("empty payload should be valid");

        assert_eq!(event.content, "");
        assert_eq!(event.sensor_name, "test");
    }

    #[test]
    fn build_event_invalid_utf8() {
        let sensor = WebhookSensor::new("test", "/test", None);
        let payload: &[u8] = &[0xff, 0xfe, 0xfd];

        let result = sensor.build_event_from_payload(payload);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not valid UTF-8"),
            "error should mention UTF-8: {err}"
        );
    }

    // --- verify_signature tests ---

    /// Helper: compute HMAC-SHA256 and return `sha256=<hex>` signature.
    fn compute_signature(payload: &[u8], secret: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(payload);
        format!("sha256={}", hex::encode(mac.finalize().into_bytes()))
    }

    #[test]
    fn verify_signature_correct_hmac() {
        let payload = b"Hello, World!";
        let secret = "my_webhook_secret";
        let sig = compute_signature(payload, secret);
        assert!(WebhookSensor::verify_signature(payload, &sig, secret));
    }

    #[test]
    fn verify_signature_wrong_secret_rejected() {
        let payload = b"Hello, World!";
        let sig = compute_signature(payload, "correct_secret");
        assert!(!WebhookSensor::verify_signature(
            payload,
            &sig,
            "wrong_secret"
        ));
    }

    #[test]
    fn verify_signature_wrong_payload_rejected() {
        let secret = "my_secret";
        let sig = compute_signature(b"original payload", secret);
        assert!(!WebhookSensor::verify_signature(
            b"tampered payload",
            &sig,
            secret
        ));
    }

    #[test]
    fn verify_signature_empty_payload() {
        let secret = "my_secret";
        let sig = compute_signature(b"", secret);
        assert!(WebhookSensor::verify_signature(b"", &sig, secret));
    }

    #[test]
    fn verify_signature_empty_secret() {
        let payload = b"data";
        let sig = compute_signature(payload, "");
        assert!(WebhookSensor::verify_signature(payload, &sig, ""));
    }

    #[test]
    fn verify_signature_github_style_format() {
        // Ensure the signature format matches GitHub webhook style
        let sig = compute_signature(b"test", "secret");
        assert!(sig.starts_with("sha256="));
        assert_eq!(sig.len(), 7 + 64); // "sha256=" + 64 hex chars
    }

    #[test]
    fn verify_signature_invalid_prefix() {
        let sig = format!("md5={}", "a".repeat(64));
        assert!(!WebhookSensor::verify_signature(b"payload", &sig, "secret"));
    }

    #[test]
    fn verify_signature_no_prefix() {
        let sig = "a".repeat(64);
        assert!(!WebhookSensor::verify_signature(b"payload", &sig, "secret"));
    }

    #[test]
    fn verify_signature_empty_hex() {
        assert!(!WebhookSensor::verify_signature(
            b"payload", "sha256=", "secret"
        ));
    }

    #[test]
    fn verify_signature_wrong_hex_length() {
        let sig = format!("sha256={}", "a".repeat(32));
        assert!(!WebhookSensor::verify_signature(b"payload", &sig, "secret"));
    }

    #[test]
    fn verify_signature_non_hex_chars() {
        let sig = format!("sha256={}gg", "a".repeat(62));
        assert!(!WebhookSensor::verify_signature(b"payload", &sig, "secret"));
    }

    #[test]
    fn verify_signature_empty_string() {
        assert!(!WebhookSensor::verify_signature(b"payload", "", "secret"));
    }

    #[test]
    fn verify_signature_forged_hex_rejected() {
        // Valid format but wrong digest — must be rejected
        let sig = format!("sha256={}", "00".repeat(32));
        assert!(!WebhookSensor::verify_signature(b"payload", &sig, "secret"));
    }

    // --- kafka_key tests ---

    #[test]
    fn kafka_key_format() {
        let sensor = WebhookSensor::new("github", "/webhooks/github", None);
        let key = sensor.kafka_key(b"test_payload");
        assert!(
            key.starts_with("github:"),
            "kafka key should start with webhook name: {key}"
        );
    }

    #[test]
    fn kafka_key_deterministic() {
        let sensor = WebhookSensor::new("test", "/test", None);
        let key1 = sensor.kafka_key(b"payload");
        let key2 = sensor.kafka_key(b"payload");
        assert_eq!(key1, key2);
    }

    #[test]
    fn kafka_key_binary_payloads_distinct() {
        let sensor = WebhookSensor::new("test", "/test", None);
        let key1 = sensor.kafka_key(&[0xff, 0xfe, 0x01]);
        let key2 = sensor.kafka_key(&[0xff, 0xfe, 0x02]);
        assert_ne!(
            key1, key2,
            "different binary payloads should produce different keys"
        );
        assert!(
            key1.starts_with("test:"),
            "binary key should still have sensor prefix"
        );
    }

    #[test]
    fn kafka_key_binary_deterministic() {
        let sensor = WebhookSensor::new("test", "/test", None);
        let payload: &[u8] = &[0xff, 0xfe, 0xfd];
        let key1 = sensor.kafka_key(payload);
        let key2 = sensor.kafka_key(payload);
        assert_eq!(key1, key2, "same binary payload should produce same key");
    }

    // --- serde roundtrip ---

    #[test]
    fn produced_event_serde_roundtrip() {
        let sensor = WebhookSensor::new("github", "/webhooks/github", None);
        let payload = br#"{"action":"closed","number":99}"#;

        let event = sensor
            .build_event_from_payload(payload)
            .expect("should build event");

        let json = serde_json::to_string(&event).expect("serialize");
        let back: SensorEvent = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(back.id, event.id);
        assert_eq!(back.sensor_name, "github");
        assert_eq!(back.modality, SensorModality::Structured);
        assert_eq!(back.content, r#"{"action":"closed","number":99}"#);
        assert_eq!(back.source_id, event.source_id);
    }

    // --- run() test ---

    #[tokio::test]
    async fn run_returns_on_cancellation() {
        let sensor = WebhookSensor::new("test", "/test", None);
        let cancel = CancellationToken::new();

        let cancel_clone = cancel.clone();
        let handle = tokio::spawn(async move {
            // Create a dummy producer config (won't actually connect)
            let producer: FutureProducer = rdkafka::config::ClientConfig::new()
                .set("bootstrap.servers", "localhost:9092")
                .create()
                .expect("create producer config");
            sensor.run(producer, cancel_clone).await
        });

        // Cancel immediately
        cancel.cancel();

        let result = handle.await.expect("task should complete");
        assert!(result.is_ok(), "run() should return Ok on cancellation");
    }
}
