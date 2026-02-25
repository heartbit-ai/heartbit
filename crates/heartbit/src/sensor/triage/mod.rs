pub mod audio;
pub mod context;
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

/// Action category for email triage — what the email needs from the agent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionCategory {
    /// Email needs a reply.
    Respond,
    /// Attachment or document to save/file.
    StoreOrFile,
    /// Invoice, billing, payment to process.
    PayOrProcess,
    /// Inform user (e.g., via Telegram) but no direct action needed.
    Notify,
    /// Unclear or sensitive — needs human judgment.
    Escalate,
    /// FYI only — no action required.
    Informational,
}

impl std::fmt::Display for ActionCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionCategory::Respond => write!(f, "respond"),
            ActionCategory::StoreOrFile => write!(f, "store_or_file"),
            ActionCategory::PayOrProcess => write!(f, "pay_or_process"),
            ActionCategory::Notify => write!(f, "notify"),
            ActionCategory::Escalate => write!(f, "escalate"),
            ActionCategory::Informational => write!(f, "informational"),
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
        /// What the email needs — action-oriented classification.
        #[serde(default)]
        action_categories: Vec<ActionCategory>,
        /// SLM-generated concrete next steps for the agent.
        #[serde(default)]
        action_hints: Vec<String>,
        /// Whether the email has attachments (from SLM + metadata inspection).
        #[serde(default)]
        has_attachments: bool,
        /// Sender email address extracted from metadata.
        #[serde(default)]
        sender: Option<String>,
        /// Email subject line extracted from metadata.
        #[serde(default)]
        subject: Option<String>,
        /// Gmail message ID for MCP `gmail_get_message` tool.
        #[serde(default)]
        message_ref: Option<String>,
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
            action_categories: vec![ActionCategory::Respond, ActionCategory::PayOrProcess],
            action_hints: vec!["Reply to confirm receipt".into()],
            has_attachments: true,
            sender: Some("billing@acme.com".into()),
            subject: Some("Invoice #2024-387".into()),
            message_ref: Some("19abc123".into()),
        };
        let json = serde_json::to_string(&decision).unwrap();
        assert!(json.contains(r#""type":"promote""#));
        let back: TriageDecision = serde_json::from_str(&json).unwrap();
        assert!(back.is_promote());
        if let TriageDecision::Promote {
            action_categories,
            action_hints,
            has_attachments,
            sender,
            subject,
            message_ref,
            ..
        } = &back
        {
            assert_eq!(action_categories.len(), 2);
            assert_eq!(action_categories[0], ActionCategory::Respond);
            assert_eq!(action_categories[1], ActionCategory::PayOrProcess);
            assert_eq!(action_hints.len(), 1);
            assert!(*has_attachments);
            assert_eq!(sender.as_deref(), Some("billing@acme.com"));
            assert_eq!(subject.as_deref(), Some("Invoice #2024-387"));
            assert_eq!(message_ref.as_deref(), Some("19abc123"));
        }
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
            action_categories: vec![],
            action_hints: vec![],
            has_attachments: false,
            sender: None,
            subject: None,
            message_ref: None,
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

    // --- ActionCategory tests ---

    #[test]
    fn action_category_serde_roundtrip() {
        for cat in [
            ActionCategory::Respond,
            ActionCategory::StoreOrFile,
            ActionCategory::PayOrProcess,
            ActionCategory::Notify,
            ActionCategory::Escalate,
            ActionCategory::Informational,
        ] {
            let json = serde_json::to_string(&cat).unwrap();
            let back: ActionCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(back, cat);
        }
    }

    #[test]
    fn action_category_snake_case() {
        assert_eq!(
            serde_json::to_string(&ActionCategory::Respond).unwrap(),
            r#""respond""#
        );
        assert_eq!(
            serde_json::to_string(&ActionCategory::StoreOrFile).unwrap(),
            r#""store_or_file""#
        );
        assert_eq!(
            serde_json::to_string(&ActionCategory::PayOrProcess).unwrap(),
            r#""pay_or_process""#
        );
        assert_eq!(
            serde_json::to_string(&ActionCategory::Notify).unwrap(),
            r#""notify""#
        );
        assert_eq!(
            serde_json::to_string(&ActionCategory::Escalate).unwrap(),
            r#""escalate""#
        );
        assert_eq!(
            serde_json::to_string(&ActionCategory::Informational).unwrap(),
            r#""informational""#
        );
    }

    #[test]
    fn action_category_display() {
        assert_eq!(ActionCategory::Respond.to_string(), "respond");
        assert_eq!(ActionCategory::StoreOrFile.to_string(), "store_or_file");
        assert_eq!(ActionCategory::PayOrProcess.to_string(), "pay_or_process");
        assert_eq!(ActionCategory::Notify.to_string(), "notify");
        assert_eq!(ActionCategory::Escalate.to_string(), "escalate");
        assert_eq!(ActionCategory::Informational.to_string(), "informational");
    }

    #[test]
    fn promote_backward_compat_old_json_without_new_fields() {
        // Old-format JSON without the new action fields should deserialize
        // with defaults (empty vecs, false, None).
        let json = r#"{"type":"promote","priority":"high","summary":"test","extracted_entities":["x"],"estimated_tokens":100}"#;
        let decision: TriageDecision = serde_json::from_str(json).unwrap();
        assert!(decision.is_promote());
        if let TriageDecision::Promote {
            action_categories,
            action_hints,
            has_attachments,
            sender,
            subject,
            message_ref,
            ..
        } = &decision
        {
            assert!(action_categories.is_empty());
            assert!(action_hints.is_empty());
            assert!(!has_attachments);
            assert!(sender.is_none());
            assert!(subject.is_none());
            assert!(message_ref.is_none());
        }
    }

    #[test]
    fn promote_with_all_new_fields_roundtrip() {
        let decision = TriageDecision::Promote {
            priority: Priority::Critical,
            summary: "Invoice from Acme".into(),
            extracted_entities: vec!["Acme Corp".into()],
            estimated_tokens: 300,
            action_categories: vec![
                ActionCategory::PayOrProcess,
                ActionCategory::StoreOrFile,
                ActionCategory::Notify,
            ],
            action_hints: vec!["Download invoice PDF".into(), "Store in workspace".into()],
            has_attachments: true,
            sender: Some("billing@acme.com".into()),
            subject: Some("Invoice #2024-387 — January consulting".into()),
            message_ref: Some("19abc123def".into()),
        };

        let json = serde_json::to_string(&decision).unwrap();
        let back: TriageDecision = serde_json::from_str(&json).unwrap();
        if let TriageDecision::Promote {
            priority,
            action_categories,
            action_hints,
            has_attachments,
            sender,
            subject,
            message_ref,
            ..
        } = &back
        {
            assert_eq!(*priority, Priority::Critical);
            assert_eq!(action_categories.len(), 3);
            assert_eq!(action_hints.len(), 2);
            assert!(*has_attachments);
            assert_eq!(sender.as_deref(), Some("billing@acme.com"));
            assert_eq!(
                subject.as_deref(),
                Some("Invoice #2024-387 — January consulting")
            );
            assert_eq!(message_ref.as_deref(), Some("19abc123def"));
        } else {
            panic!("expected Promote, got {back:?}");
        }
    }
}
