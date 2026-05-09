//! Reply pipeline — drafts a single short reply to a specific mention,
//! routes to Telegram for review, posts via `twitter_reply` on user pick.
//!
//! See spec §2/§5 for the architecture; this file holds the value types
//! and the public surface. The runtime lives in [`run_reply_pipeline`]
//! once Task 5 lands it.

use chrono::{DateTime, Utc};

pub mod prompts;

/// A mention of the operator's account fetched from `twitter_mentions`.
#[derive(Debug, Clone)]
pub struct Mention {
    /// X tweet ID of the mention itself.
    pub id: String,
    /// Plain text of the mention.
    pub text: String,
    /// X user ID of the mentioner.
    pub author_id: String,
    /// Public handle of the mentioner (sans `@`).
    pub author_handle: String,
    /// When the mention was posted.
    pub posted_at: DateTime<Utc>,
    /// Tweet ID this mention is replying to (None when it's a top-level
    /// `@operator …` mention rather than a reply on an operator's tweet).
    pub in_reply_to_tweet_id: Option<String>,
}

/// A small snapshot of a tweet (text + timing). Used as a parent-tweet
/// context for the reply researcher.
#[derive(Debug, Clone)]
pub struct TweetSnapshot {
    /// X tweet ID.
    pub id: String,
    /// Plain text of the tweet.
    pub text: String,
    /// When the tweet was posted.
    pub posted_at: DateTime<Utc>,
}

/// Tone-calibration context about the mentioner. None of these are
/// strictly required; the writer degrades gracefully if missing.
#[derive(Debug, Clone, Default)]
pub struct MentionerContext {
    /// Public handle of the mentioner (sans `@`).
    pub handle: String,
    /// Bio text from the mentioner's profile, if available.
    pub bio: Option<String>,
    /// Up to 3 recent tweets, abridged.
    pub recent_tweets: Vec<TweetSnapshot>,
    /// Follower count of the mentioner, if available.
    pub follower_count: Option<u64>,
}

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::llm::types::TokenUsage;
use thiserror::Error;

use crate::pipeline::{PipelineError, ProgressCallback, ResearcherOverride};

/// Configuration for one reply-pipeline run.
#[derive(Clone)]
pub struct ReplyConfig<'a> {
    /// Persona name.
    pub persona_name: &'a str,
    /// LLM provider used for every sub-agent in this run.
    pub provider: Arc<BoxedProvider>,
    /// Root directory containing per-persona corpora.
    pub corpora_root: &'a Path,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: &'a Path,
    /// Optional progress callback.
    pub on_progress: Option<ProgressCallback>,
    /// The mention being replied to.
    pub mention: Mention,
    /// The operator's tweet the mention is replying to (when the mention
    /// is a thread reply rather than a top-level @-mention).
    pub parent: Option<TweetSnapshot>,
    /// Optional bio + recent tweets for tone calibration.
    pub mentioner_context: Option<MentionerContext>,
    /// Number of distinct candidate replies to generate (1..=3).
    /// 1 = no judge; 2 or 3 = judge picks.
    pub candidates_per_reply: usize,
    /// Persona-specific mode addendum.
    pub mode_addendum: Option<&'a str>,
    /// Optional researcher override (same semantics as PipelineConfig).
    pub researcher_override: Option<ResearcherOverride>,
    /// Telegram-or-mock delivery layer for the reply review.
    pub delivery: Arc<dyn ReplyReviewDelivery>,
    /// `twitter_reply` tool — production wires `Arc::new(TwitterReplyTool::new())`;
    /// tests wire a mock.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver for `twitter_tool`.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Output of a successful reply-pipeline run.
#[derive(Debug, Clone)]
pub struct ReplyOutput {
    /// X tweet ID of the mention this run was for.
    pub mention_id: String,
    /// One generated reply candidate per slot (`candidates_per_reply`).
    pub candidates: Vec<ReplyCandidateRecord>,
    /// Aggregated token usage across all sub-agents.
    pub usage_summary: TokenUsage,
    /// Final outcome of the run.
    pub outcome: ReplyOutcome,
}

/// One generated reply draft (post-style/fact, pre-publish_gate).
#[derive(Debug, Clone)]
pub struct ReplyCandidateRecord {
    /// The composed reply text.
    pub draft: String,
    /// Voice-match score from the style critic, 0.0..=1.0.
    pub style_match_score: f32,
    /// Fact-check verdict: "verified" | "unverifiable: reason" | "rejected: reason".
    pub fact_check_verdict: String,
}

/// What happened in this reply run.
#[derive(Debug, Clone)]
pub enum ReplyOutcome {
    /// User picked candidate `chosen_index` and the reply was published.
    Posted {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// X tweet ID of the published reply.
        reply_tweet_id: String,
        /// Public URL of the published reply.
        reply_url: String,
    },
    /// User pressed Skip.
    Skipped,
    /// Telegram review timed out without a pick.
    TimedOut,
    /// User picked `chosen_index` but `publish_gate` rejected it.
    GateRejected {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Reason from the publish gate.
        reason: String,
    },
    /// User picked `chosen_index` but the X API call failed.
    PublishFailed {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Reason for failure.
        reason: String,
    },
    /// All candidates returned the literal "no_reply" string — the
    /// writer chose not to engage. No Telegram review was sent.
    NoReply,
}

/// Errors raised by `run_reply_pipeline`.
#[derive(Debug, Error)]
pub enum ReplyError {
    /// Underlying pipeline error.
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Telegram delivery error.
    #[error("delivery: {0}")]
    Delivery(#[from] crate::review::ReviewDeliveryError),
    /// Misconfigured `ReplyConfig`.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Telegram-or-mock review delivery for reply messages. Mirrors
/// [`crate::review::ReviewDelivery`] but with a parent-quoted message
/// shape. See spec §9 for the full text layout.
///
/// Methods use the project's `Pin<Box<dyn Future>>` desugaring to stay
/// object-safe without the `async-trait` crate (matches `ReviewDelivery`).
pub trait ReplyReviewDelivery: Send + Sync {
    /// Deliver a reply review to the user (Telegram bot or mock).
    fn deliver<'a>(
        &'a self,
        msg: ReplyReviewMessage,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        crate::review::DeliveredReview,
                        crate::review::ReviewDeliveryError,
                    >,
                > + Send
                + 'a,
        >,
    >;

    /// Report final outcome back to the delivery layer (edits the original
    /// message, etc.). Non-fatal; runtime ignores errors.
    fn report<'a>(
        &'a self,
        receipt: crate::review::DeliveryReceipt,
        outcome: ReplyOutcome,
    ) -> Pin<Box<dyn Future<Output = Result<(), crate::review::ReviewDeliveryError>> + Send + 'a>>;
}

/// Message body for a reply review. The Telegram impl renders this as
/// the parent + mention + drafts layout from spec §9.1.
#[derive(Debug, Clone)]
pub struct ReplyReviewMessage {
    /// The mention being replied to.
    pub mention: Mention,
    /// The operator's parent tweet, if applicable.
    pub parent: Option<TweetSnapshot>,
    /// Optional mentioner context for tone calibration.
    pub mentioner_context: Option<MentionerContext>,
    /// One drafted reply per slot.
    pub candidates: Vec<String>,
    /// How long to wait for a user pick (Telegram impl uses this).
    pub interaction_timeout_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mention_round_trips_through_clone() {
        let m = Mention {
            id: "1".into(),
            text: "hi".into(),
            author_id: "12".into(),
            author_handle: "alice".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: Some("99".into()),
        };
        let copy = m.clone();
        assert_eq!(copy.id, m.id);
        assert_eq!(copy.text, m.text);
        assert_eq!(copy.in_reply_to_tweet_id, m.in_reply_to_tweet_id);
    }

    #[test]
    fn mentioner_context_default_has_empty_handle_and_no_recent_tweets() {
        let m = MentionerContext::default();
        assert!(m.handle.is_empty());
        assert!(m.bio.is_none());
        assert!(m.recent_tweets.is_empty());
        assert!(m.follower_count.is_none());
    }

    #[test]
    fn reply_outcome_no_reply_is_distinct_from_skipped() {
        let a = ReplyOutcome::Skipped;
        let b = ReplyOutcome::NoReply;
        // No Eq derive (CredentialResolver isn't Eq); rely on debug
        // representation as a stand-in for "these are different variants".
        assert_ne!(format!("{a:?}"), format!("{b:?}"));
    }

    #[test]
    fn reply_error_display_round_trips() {
        let e = ReplyError::InvalidConfig("test".to_string());
        assert!(format!("{e}").contains("invalid config"));
    }
}
