//! Proactive posting pipeline — generates a topic, drafts candidates
//! via the existing review pipeline, gates through Telegram, posts
//! the chosen draft, records outcome.
//!
//! See spec §6 for the storage shape; the runtime lives in the daemon
//! umbrella's `handle_persona_post`.

use chrono::{DateTime, Utc};

pub mod engagement;
pub mod history;
pub mod topic_context;

pub use engagement::{
    EngagementSnapshot, EngagementStore, EngagementStoreError, InMemoryEngagementStore,
    JsonlEngagementStore,
};
pub use history::{InMemoryPostHistoryStore, JsonlPostHistoryStore, PostHistoryStore, StoreError};
pub use topic_context::{
    HeartbitRsXTopicContext, TopicContextDeps, TopicContextProvider, XGhostTopicContext,
};

/// One historical post (or skip / time-out / no_topic) recorded by the
/// daemon's persona post handler.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PostHistoryEntry {
    /// When the tick fired (regardless of outcome).
    pub posted_at: DateTime<Utc>,
    /// Topic the generator proposed (empty when outcome is `NoTopic`).
    pub topic: String,
    /// What ultimately happened.
    pub outcome: PostOutcome,
    /// Tweet id when `outcome` is `Posted`; else `None`.
    pub tweet_id: Option<String>,
}

/// What happened in one persona-post tick.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PostOutcome {
    /// Topic generator returned the literal "no_topic" — pipeline NOT called.
    NoTopic,
    /// Topic was already posted within the lookback window — pipeline NOT called.
    SkippedDuplicate,
    /// Pipeline ran, user picked a draft, post succeeded.
    Posted {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Public URL of the published thread.
        url: String,
    },
    /// User pressed Skip on Telegram review.
    Skipped,
    /// Telegram review timed out without a pick.
    TimedOut,
    /// Pipeline's publish gate rejected the chosen candidate.
    GateRejected {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Reason from the publish gate.
        reason: String,
    },
    /// User picked but the X API call failed.
    PublishFailed {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Reason for failure.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_outcome_no_topic_distinct_from_skipped() {
        let a = PostOutcome::NoTopic;
        let b = PostOutcome::Skipped;
        assert_ne!(a, b);
    }

    #[test]
    fn post_history_entry_round_trips_through_serde() {
        let entry = PostHistoryEntry {
            posted_at: Utc::now(),
            topic: "calibrated abstention".into(),
            outcome: PostOutcome::Posted {
                chosen_index: 1,
                url: "https://x.com/i/web/status/123".into(),
            },
            tweet_id: Some("123".into()),
        };
        let s = serde_json::to_string(&entry).unwrap();
        let parsed: PostHistoryEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.topic, entry.topic);
        assert_eq!(parsed.tweet_id, entry.tweet_id);
        assert_eq!(parsed.outcome, entry.outcome);
    }
}
