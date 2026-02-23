//! POMDP-inspired observation compression for the sensor pipeline.
//!
//! Compresses raw `SensorEvent` content before forwarding to downstream agents,
//! reducing token usage and improving signal-to-noise ratio.
//!
//! ## Maturity levels (ACON paper, arXiv:2510.0061)
//!
//! **Level 1 — Rule-Based (current):**
//! Static per-modality rules (truncate, strip patterns, summary-only, passthrough).
//! No LLM calls, deterministic, zero latency overhead. Good enough for MVP where
//! sensors produce well-structured data with predictable content shapes.
//!
//! **Level 2 — SLM-Guided (TODO):**
//! Use a small language model (local or cloud-light tier) to dynamically decide
//! compression strategy per event. The SLM scores information density of each
//! section and selectively preserves high-value segments. Requires:
//! - Integration with `ModelRouter::route_summarize()` for SLM access.
//! - Learned compression policies from user feedback (which compressions lost
//!   critical info vs. which were fine).
//! - Latency budget: must complete within triage SLA (~100ms local, ~500ms cloud).
//!
//! **Level 3 — Distilled (TODO):**
//! Train a tiny distilled model specifically for compression decisions, using
//! traces from Level 2 as training data. The distilled model runs locally with
//! sub-10ms latency and near-zero cost. Requires:
//! - Sufficient Level 2 trace data (thousands of compression decisions).
//! - ONNX or similar runtime for inference without full LLM stack.
//! - A/B testing framework to validate distilled model quality vs. SLM baseline.

pub mod rules;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::sensor::{SensorEvent, SensorModality};
use crate::tool::builtins::floor_char_boundary;

/// A single compression rule applied to event content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompressionRule {
    /// Truncate content at a UTF-8 boundary.
    Truncate { max_bytes: usize },
    /// Strip substrings matching a regex pattern.
    StripPattern { pattern: String },
    /// Discard raw content, keep only the summary from metadata.
    SummaryOnly,
    /// Pass content through unchanged.
    Passthrough,
}

/// Per-modality compression policy.
///
/// Each `SensorModality` maps to an ordered list of `CompressionRule`s.
/// Rules are applied sequentially: the output of one rule feeds the next.
pub struct CompressionPolicy {
    rules: HashMap<SensorModality, Vec<CompressionRule>>,
}

impl Default for CompressionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionPolicy {
    /// Create a policy with custom rules for a single modality.
    pub fn with_rules(modality: SensorModality, rules: Vec<CompressionRule>) -> Self {
        let mut map = HashMap::new();
        map.insert(modality, rules);
        Self { rules: map }
    }

    /// Create a default policy with sensible rules per modality.
    ///
    /// - `Text` -> Truncate at 4096 bytes
    /// - `Image` -> SummaryOnly (raw pixels are in `binary_ref`, content is description)
    /// - `Audio` -> SummaryOnly (transcription may be huge)
    /// - `Structured` -> Passthrough (JSON is already compact)
    pub fn new() -> Self {
        let mut rules = HashMap::new();
        rules.insert(
            SensorModality::Text,
            vec![CompressionRule::Truncate { max_bytes: 4096 }],
        );
        rules.insert(SensorModality::Image, vec![CompressionRule::SummaryOnly]);
        rules.insert(SensorModality::Audio, vec![CompressionRule::SummaryOnly]);
        rules.insert(
            SensorModality::Structured,
            vec![CompressionRule::Passthrough],
        );
        Self { rules }
    }

    /// Compress a sensor event's content according to its modality rules.
    ///
    /// Returns `(compressed_content, estimated_bytes_saved)`.
    pub fn compress(&self, event: &SensorEvent) -> (String, usize) {
        let rules = match self.rules.get(&event.modality) {
            Some(r) => r,
            None => return (event.content.clone(), 0),
        };

        let original_len = event.content.len();
        let mut content = event.content.clone();

        for rule in rules {
            match rule {
                CompressionRule::Truncate { max_bytes } => {
                    if content.len() > *max_bytes {
                        let boundary = floor_char_boundary(&content, *max_bytes);
                        content.truncate(boundary);
                    }
                }
                CompressionRule::StripPattern { pattern } => {
                    if let Ok(re) = regex::Regex::new(pattern) {
                        content = re.replace_all(&content, "").into_owned();
                    }
                }
                CompressionRule::SummaryOnly => {
                    if let Some(metadata) = &event.metadata
                        && let Some(summary) = metadata.get("summary").and_then(|v| v.as_str())
                    {
                        content = summary.to_owned();
                    }
                }
                CompressionRule::Passthrough => {}
            }
        }

        let saved = original_len.saturating_sub(content.len());
        (content, saved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_event(modality: SensorModality, content: &str) -> SensorEvent {
        SensorEvent {
            id: "test".into(),
            sensor_name: "test_sensor".into(),
            modality,
            observed_at: Utc::now(),
            content: content.into(),
            source_id: "src".into(),
            metadata: None,
            binary_ref: None,
            related_ids: vec![],
        }
    }

    #[test]
    fn default_policy_has_all_modalities() {
        let policy = CompressionPolicy::new();
        assert!(policy.rules.contains_key(&SensorModality::Text));
        assert!(policy.rules.contains_key(&SensorModality::Image));
        assert!(policy.rules.contains_key(&SensorModality::Audio));
        assert!(policy.rules.contains_key(&SensorModality::Structured));
    }

    #[test]
    fn compress_truncates_long_text() {
        let policy = CompressionPolicy::new();
        let long_content = "a".repeat(8000);
        let event = make_event(SensorModality::Text, &long_content);

        let (compressed, saved) = policy.compress(&event);
        assert!(compressed.len() <= 4096);
        assert!(saved > 0);
        assert_eq!(saved, 8000 - compressed.len());
    }

    #[test]
    fn compress_short_text_unchanged() {
        let policy = CompressionPolicy::new();
        let event = make_event(SensorModality::Text, "short text");

        let (compressed, saved) = policy.compress(&event);
        assert_eq!(compressed, "short text");
        assert_eq!(saved, 0);
    }

    #[test]
    fn compress_passthrough_for_structured() {
        let policy = CompressionPolicy::new();
        let json = r#"{"temp": 72, "unit": "F"}"#;
        let event = make_event(SensorModality::Structured, json);

        let (compressed, saved) = policy.compress(&event);
        assert_eq!(compressed, json);
        assert_eq!(saved, 0);
    }

    #[test]
    fn compress_summary_only_with_metadata() {
        let policy = CompressionPolicy::new();
        let mut event = make_event(SensorModality::Image, "very long image description here");
        event.metadata = Some(serde_json::json!({"summary": "A photo of a cat"}));

        let (compressed, saved) = policy.compress(&event);
        assert_eq!(compressed, "A photo of a cat");
        assert!(saved > 0);
    }

    #[test]
    fn compress_summary_only_without_metadata_keeps_content() {
        let policy = CompressionPolicy::new();
        let event = make_event(SensorModality::Image, "image description");

        let (compressed, saved) = policy.compress(&event);
        assert_eq!(compressed, "image description");
        assert_eq!(saved, 0);
    }

    #[test]
    fn compress_returns_bytes_saved() {
        let policy = CompressionPolicy::new();
        let content = "x".repeat(5000);
        let event = make_event(SensorModality::Text, &content);

        let (compressed, saved) = policy.compress(&event);
        assert_eq!(saved, 5000 - compressed.len());
    }

    #[test]
    fn compress_truncate_respects_utf8_boundary() {
        let policy = CompressionPolicy::new();
        // Each emoji is 4 bytes; 1025 emojis = 4100 bytes, should truncate to <= 4096
        let content: String = std::iter::repeat_n('\u{1F600}', 1025).collect();
        assert_eq!(content.len(), 4100);
        let event = make_event(SensorModality::Text, &content);

        let (compressed, saved) = policy.compress(&event);
        assert!(compressed.len() <= 4096);
        // Must be valid UTF-8 (String guarantees this, but verify length is on boundary)
        assert_eq!(compressed.len() % 4, 0);
        assert!(saved > 0);
    }

    #[test]
    fn strip_pattern_removes_matches() {
        let mut rules = HashMap::new();
        rules.insert(
            SensorModality::Text,
            vec![CompressionRule::StripPattern {
                pattern: r"\b(ADVERTISEMENT|SPONSORED)\b".into(),
            }],
        );
        let policy = CompressionPolicy { rules };
        let event = make_event(
            SensorModality::Text,
            "News ADVERTISEMENT content SPONSORED end",
        );

        let (compressed, saved) = policy.compress(&event);
        assert!(!compressed.contains("ADVERTISEMENT"));
        assert!(!compressed.contains("SPONSORED"));
        assert!(saved > 0);
    }

    #[test]
    fn unknown_modality_returns_content_unchanged() {
        // Policy with empty rules map
        let policy = CompressionPolicy {
            rules: HashMap::new(),
        };
        let event = make_event(SensorModality::Text, "hello");

        let (compressed, saved) = policy.compress(&event);
        assert_eq!(compressed, "hello");
        assert_eq!(saved, 0);
    }
}
