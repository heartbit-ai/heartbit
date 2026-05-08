//! Review-mode pipeline — sends N candidate drafts to the user via a
//! [`ReviewDelivery`] (Telegram in production), awaits the user's pick,
//! then posts the chosen draft to X via the `twitter_thread` tool.
//!
//! Public entry: [`run_review_pipeline`] (lands in Task 2).

use std::path::Path;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::llm::types::TokenUsage;
use heartbit_core::tool::Tool;
use thiserror::Error;

use crate::pipeline::{CandidateRecord, PipelineError, ProgressCallback};

pub mod delivery;
pub mod prompts;
pub mod tweet_split;

pub use delivery::{
    DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReportableOutcome, ReviewDelivery,
    ReviewDeliveryError, ReviewMessage,
};
pub use prompts::{build_report_message, build_review_message};
pub use tweet_split::parse_thread_tweets;

/// Configuration for one review-mode pipeline run.
#[derive(Clone)]
pub struct ReviewConfig<'a> {
    /// Persona instance name (used to load StyleProfile snapshot).
    pub persona_name: &'a str,
    /// Topic / prompt for this run.
    pub topic: &'a str,
    /// LLM provider (shared across sub-agents).
    pub provider: Arc<BoxedProvider>,
    /// Corpora root (currently unused; reserved for P1.3e few-shot retrieval).
    pub corpora_root: &'a Path,
    /// Profiles root (passed to SnapshotStore::open).
    pub profiles_root: &'a Path,
    /// Optional progress callback. Called with a short status string at
    /// each pipeline stage start.
    pub on_progress: Option<ProgressCallback>,
    /// Number of distinct candidate drafts to generate (1..=10).
    /// Same semantics as `PipelineConfig.candidates_per_draft`.
    pub candidates_per_draft: usize,
    /// Telegram-or-mock delivery layer.
    pub delivery: Arc<dyn ReviewDelivery>,
    /// Twitter-or-mock posting tool. Production wires
    /// `Arc::new(TwitterThreadTool::new())`; tests wire a mock Tool.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver for `twitter_tool`. Threaded into
    /// `ExecutionContext::credentials` at execute-time.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Output of a successful review-mode run.
#[derive(Debug, Clone)]
pub struct ReviewOutput {
    /// All distinct candidate drafts (1..=`candidates_per_draft` after dedup).
    pub candidates: Vec<CandidateRecord>,
    /// Researcher's digest text.
    pub research_digest: String,
    /// Accumulated token usage across sub-agent calls.
    pub usage_summary: TokenUsage,
    /// What happened.
    pub outcome: ReviewOutcome,
}

/// Outcome of the review interaction.
#[derive(Debug, Clone)]
pub enum ReviewOutcome {
    /// User picked candidate `chosen_index` and the post was published.
    Posted {
        /// 0-based index into `candidates`.
        chosen_index: usize,
        /// Final URL of the first tweet in the (possibly single-tweet) thread.
        tweet_url: String,
        /// IDs of all tweets posted (1 for single, N for thread).
        tweet_ids: Vec<String>,
    },
    /// User pressed Skip.
    Skipped,
    /// Timeout elapsed before user responded.
    TimedOut,
    /// User picked candidate `chosen_index` but `publish_gate` rejected it.
    GateRejected {
        /// 0-based index of the rejected draft.
        chosen_index: usize,
        /// Reason from `PublishGateError`'s display.
        reason: String,
    },
    /// User picked candidate `chosen_index` but the X API call failed.
    PublishFailed {
        /// 0-based index of the draft that failed to post.
        chosen_index: usize,
        /// Failure reason.
        reason: String,
    },
}

/// Errors raised by `run_review_pipeline`. Note: `ReviewDelivery::report()`
/// failures are intentionally NOT a `ReviewError` variant. They're
/// non-fatal (post may have succeeded; only the after-the-fact message
/// edit failed) and are logged via the `on_progress` callback.
#[derive(Debug, Error)]
pub enum ReviewError {
    /// Candidate generation failed (delegates to `PipelineError`).
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Telegram (or mock) delivery failed.
    #[error("delivery: {0}")]
    Delivery(#[from] ReviewDeliveryError),
    /// Config validation at run start.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_error_pipeline_renders_inner_message() {
        let e = ReviewError::Pipeline(PipelineError::InvalidConfig(
            "candidates_per_draft must be in 1..=10 (got 0)".to_string(),
        ));
        let s = format!("{e}");
        assert!(s.contains("pipeline:"), "got: {s}");
        assert!(s.contains("invalid config"), "got: {s}");
    }

    #[test]
    fn review_error_delivery_renders_transport_error() {
        let e = ReviewError::Delivery(ReviewDeliveryError::Transport("bot offline".to_string()));
        let s = format!("{e}");
        assert!(s.contains("delivery:"), "got: {s}");
        assert!(s.contains("bot offline"), "got: {s}");
    }

    #[test]
    fn review_error_invalid_config_renders_with_string() {
        let e = ReviewError::InvalidConfig("no profile snapshot".to_string());
        let s = format!("{e}");
        assert!(s.contains("invalid config"), "got: {s}");
        assert!(s.contains("no profile snapshot"), "got: {s}");
    }
}
