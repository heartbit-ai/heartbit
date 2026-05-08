//! Per-post data types — what one line of a writer's JSONL corpus contains.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Engagement metrics from the source platform.
///
/// Best-effort: missing fields default to zero on parse, and the type is
/// stored only when the JSONL line included it. Posts without engagement
/// data (e.g., manually authored corpora) carry [`CorpusEntry::engagement`]
/// = `None` and never construct this struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Engagement {
    /// Like / heart count.
    #[serde(default)]
    pub likes: u64,
    /// Repost / retweet / quote count.
    #[serde(default)]
    pub reposts: u64,
    /// Reply count.
    #[serde(default)]
    pub replies: u64,
}

/// One post in a writer's reference corpus. The minimal schema requires only
/// `id` and `post_text`; everything else is optional.
///
/// `id` is typically the source platform's post id (e.g., the X tweet id as
/// a string). It is the dedup key on re-import (see
/// [`crate::corpus::Corpus::append_from_jsonl`]).
///
/// The writer handle is **not** stored on the entry — it is implicit from
/// the file the entry lives in (`<writer>.jsonl`). Storing it per entry
/// would be redundant and would let imports drift.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusEntry {
    /// Stable identifier — typically the X tweet id as a string.
    /// Used for dedup on re-import.
    pub id: String,

    /// The post text (no markdown stripping; stored verbatim).
    pub post_text: String,

    /// Original posting time; RFC3339 in JSONL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posted_at: Option<DateTime<Utc>>,

    /// Engagement metrics from the source (best-effort; may be absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engagement: Option<Engagement>,

    /// Manual tags: `["thread_opener", "hot_take", "self_deprecating"]`.
    /// Empty by default; absent vs. empty are stored identically (`Vec::new`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,

    /// Pre-computed embedding. P1.2b stores but does not generate
    /// embeddings (P1.4 wires the local-embedding pipeline through this
    /// field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_minimal_round_trip_via_json() {
        let entry = CorpusEntry {
            id: "1857234567890".to_string(),
            post_text: "the bitter lesson keeps winning".to_string(),
            posted_at: None,
            engagement: None,
            tags: Vec::new(),
            embedding: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        let back: CorpusEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, entry);
    }

    #[test]
    fn entry_full_round_trip_via_json() {
        let entry = CorpusEntry {
            id: "1857234567891".to_string(),
            post_text: "compute + scale + simple objective".to_string(),
            posted_at: Some(
                "2025-04-12T14:32:00Z"
                    .parse::<DateTime<Utc>>()
                    .expect("rfc3339 parses"),
            ),
            engagement: Some(Engagement {
                likes: 1234,
                reposts: 56,
                replies: 12,
            }),
            tags: vec!["hot_take".to_string(), "thread_opener".to_string()],
            embedding: Some(vec![0.1, 0.2, 0.3]),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        let back: CorpusEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, entry);
    }

    #[test]
    fn engagement_default_is_all_zero() {
        let e = Engagement::default();
        assert_eq!(e.likes, 0);
        assert_eq!(e.reposts, 0);
        assert_eq!(e.replies, 0);
    }

    #[test]
    fn engagement_partial_fields_default_remaining_to_zero() {
        let parsed: Engagement = serde_json::from_str(r#"{"likes": 42}"#).expect("parses");
        assert_eq!(parsed.likes, 42);
        assert_eq!(parsed.reposts, 0);
        assert_eq!(parsed.replies, 0);
    }

    #[test]
    fn entry_optional_fields_omitted_when_none_or_empty() {
        let entry = CorpusEntry {
            id: "1".to_string(),
            post_text: "hi".to_string(),
            posted_at: None,
            engagement: None,
            tags: Vec::new(),
            embedding: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        // Only id + post_text should appear in the wire form.
        assert!(json.contains("\"id\""));
        assert!(json.contains("\"post_text\""));
        assert!(!json.contains("posted_at"));
        assert!(!json.contains("engagement"));
        assert!(!json.contains("tags"));
        assert!(!json.contains("embedding"));
    }

    #[test]
    fn entry_unknown_field_rejected_via_deny_unknown_fields() {
        let json = r#"{"id":"1","post_text":"hi","bogus":"oops"}"#;
        let err = serde_json::from_str::<CorpusEntry>(json).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("bogus") || msg.contains("unknown"));
    }

    #[test]
    fn engagement_unknown_field_rejected_via_deny_unknown_fields() {
        let json = r#"{"likes":1,"shares":99}"#;
        let err = serde_json::from_str::<Engagement>(json).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("shares") || msg.contains("unknown"));
    }
}
