pub mod audio;
pub mod email;
pub mod image;
pub mod rss;
pub mod structured;
pub mod webhook;

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::sensor::{SensorEvent, SensorModality};

/// Priority level assigned by triage.
///
/// Higher priority events get faster processing and richer agent squads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Background/informational — may be batched or deferred.
    Low,
    /// Standard processing within normal SLA.
    Normal,
    /// Important but not urgent — process promptly.
    High,
    /// Immediate attention required — bypass queues.
    Critical,
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Priority::Low => write!(f, "low"),
            Priority::Normal => write!(f, "normal"),
            Priority::High => write!(f, "high"),
            Priority::Critical => write!(f, "critical"),
        }
    }
}

/// Triage decision for a sensor event.
///
/// Each triage processor evaluates a sensor event and produces one of three
/// outcomes: promote (forward for processing), drop (discard), or dead-letter
/// (processing error).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TriageDecision {
    /// Promote to story correlator and then to the daemon for agent processing.
    Promote {
        priority: Priority,
        /// SLM-generated one-sentence summary.
        summary: String,
        /// Extracted entities: people, organizations, topics.
        extracted_entities: Vec<String>,
        /// Estimated token cost for downstream frontier LLM processing.
        estimated_tokens: usize,
    },
    /// Drop silently (spam, duplicate, irrelevant).
    Drop { reason: String },
    /// Send to dead-letter topic (processing error, malformed input).
    DeadLetter { error: String },
}

impl TriageDecision {
    /// Returns `true` if this decision promotes the event for processing.
    pub fn is_promote(&self) -> bool {
        matches!(self, TriageDecision::Promote { .. })
    }

    /// Returns `true` if this decision drops the event.
    pub fn is_drop(&self) -> bool {
        matches!(self, TriageDecision::Drop { .. })
    }

    /// Returns `true` if this decision sends to dead-letter.
    pub fn is_dead_letter(&self) -> bool {
        matches!(self, TriageDecision::DeadLetter { .. })
    }
}

/// Per-modality triage processor.
///
/// Each sensor type gets a dedicated triage consumer with modality-specific
/// processing logic. SLMs handle the heavy lifting (classification, extraction,
/// summarization).
pub trait TriageProcessor: Send + Sync {
    /// The modality this processor handles.
    fn modality(&self) -> SensorModality;

    /// The source Kafka topic this processor consumes from.
    fn source_topic(&self) -> &str;

    /// Process a single sensor event and produce a triage decision.
    fn process(
        &self,
        event: &SensorEvent,
    ) -> Pin<Box<dyn Future<Output = Result<TriageDecision, Error>> + Send + '_>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_serde_roundtrip() {
        for p in [
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Critical,
        ] {
            let json = serde_json::to_string(&p).unwrap();
            let back: Priority = serde_json::from_str(&json).unwrap();
            assert_eq!(back, p);
        }
    }

    #[test]
    fn priority_snake_case() {
        assert_eq!(serde_json::to_string(&Priority::Low).unwrap(), r#""low""#);
        assert_eq!(
            serde_json::to_string(&Priority::Normal).unwrap(),
            r#""normal""#
        );
        assert_eq!(serde_json::to_string(&Priority::High).unwrap(), r#""high""#);
        assert_eq!(
            serde_json::to_string(&Priority::Critical).unwrap(),
            r#""critical""#
        );
    }

    #[test]
    fn priority_ordering() {
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Critical);
    }

    #[test]
    fn priority_display() {
        assert_eq!(Priority::Low.to_string(), "low");
        assert_eq!(Priority::Normal.to_string(), "normal");
        assert_eq!(Priority::High.to_string(), "high");
        assert_eq!(Priority::Critical.to_string(), "critical");
    }

    #[test]
    fn triage_decision_promote_roundtrip() {
        let decision = TriageDecision::Promote {
            priority: Priority::High,
            summary: "Important email from client".into(),
            extracted_entities: vec!["Acme Corp".into(), "billing".into()],
            estimated_tokens: 500,
        };
        let json = serde_json::to_string(&decision).unwrap();
        assert!(json.contains(r#""type":"promote""#));
        let back: TriageDecision = serde_json::from_str(&json).unwrap();
        assert!(back.is_promote());
    }

    #[test]
    fn triage_decision_drop_roundtrip() {
        let decision = TriageDecision::Drop {
            reason: "spam detected".into(),
        };
        let json = serde_json::to_string(&decision).unwrap();
        assert!(json.contains(r#""type":"drop""#));
        let back: TriageDecision = serde_json::from_str(&json).unwrap();
        assert!(back.is_drop());
    }

    #[test]
    fn triage_decision_dead_letter_roundtrip() {
        let decision = TriageDecision::DeadLetter {
            error: "failed to parse XML".into(),
        };
        let json = serde_json::to_string(&decision).unwrap();
        assert!(json.contains(r#""type":"dead_letter""#));
        let back: TriageDecision = serde_json::from_str(&json).unwrap();
        assert!(back.is_dead_letter());
    }

    #[test]
    fn triage_decision_predicates() {
        let promote = TriageDecision::Promote {
            priority: Priority::Normal,
            summary: "s".into(),
            extracted_entities: vec![],
            estimated_tokens: 0,
        };
        assert!(promote.is_promote());
        assert!(!promote.is_drop());
        assert!(!promote.is_dead_letter());

        let drop = TriageDecision::Drop { reason: "r".into() };
        assert!(!drop.is_promote());
        assert!(drop.is_drop());
        assert!(!drop.is_dead_letter());

        let dl = TriageDecision::DeadLetter { error: "e".into() };
        assert!(!dl.is_promote());
        assert!(!dl.is_drop());
        assert!(dl.is_dead_letter());
    }
}
