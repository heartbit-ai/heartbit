//! X self-amplification of pascal.heartbit.ai blog posts.
//!
//! When `handle_persona_blog` reports `BlogOutcome::Posted` and the
//! optional `deploy_command` succeeded (or was absent), the daemon
//! enqueues a `DaemonCommand::BlogAnnounceX` command. The handler in
//! `heartbit::daemon::blog_announce_x_handler` then calls
//! [`run_x_announcement_pipeline`] which drafts a thread → length
//! normalize → Telegram review → publish via the existing
//! `twitter_tool`.
//!
//! No researcher / no fact_check: the source is the operator's own
//! blog, already fact-checked through the blog pipeline.

#![allow(dead_code)] // Pipeline impl arrives in Task 5; types are public for the handler.

use std::path::Path;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::tool::Tool;

use crate::pipeline::ProgressCallback;
use crate::review::ReviewDelivery;

/// Configuration for one X announcement tick.
pub struct XAnnouncementConfig<'a> {
    /// Persona name (e.g. `"heartbit-ghost:x"`).
    pub persona_name: &'a str,
    /// Default LLM provider.
    pub provider: Arc<BoxedProvider>,
    /// Optional writer-stage provider override (falls back to `provider`).
    pub writer_provider: Option<Arc<BoxedProvider>>,
    /// Persona corpora root.
    pub corpora_root: &'a Path,
    /// Persona voice profiles root.
    pub profiles_root: &'a Path,
    /// Optional progress callback for tracing.
    pub on_progress: Option<ProgressCallback>,
    /// Title of the blog post being announced.
    pub title: &'a str,
    /// One-line excerpt (≤160 chars).
    pub excerpt: &'a str,
    /// First ~500 chars of the blog post body for context.
    pub body_snippet: &'a str,
    /// Canonical URL of the blog post.
    pub post_url: &'a str,
    /// Telegram review delivery.
    pub delivery: Arc<dyn ReviewDelivery>,
    /// X publish tool (existing `twitter_tool` from the daemon).
    pub twitter_tool: Arc<dyn Tool>,
    /// X API credentials.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Terminal state of one X announcement tick.
#[derive(Debug, Clone)]
pub enum XAnnouncementOutcome {
    /// Pipeline ran, operator picked the draft, X publish succeeded.
    Posted {
        /// IDs of the tweets in the thread (head-first).
        tweet_ids: Vec<String>,
        /// Public URL of the head tweet.
        head_url: String,
    },
    /// Operator pressed Skip on Telegram.
    Skipped,
    /// Telegram review timed out.
    TimedOut,
    /// Operator picked but X publish failed (non-OK tool result).
    PublishFailed {
        /// Reason from the X tool's error.
        reason: String,
    },
}

/// Result of one announcement pipeline tick.
#[derive(Debug, Clone)]
pub struct XAnnouncementOutput {
    /// Final outcome.
    pub outcome: XAnnouncementOutcome,
    /// Token usage across writer + length_normalize calls.
    pub usage_summary: heartbit_core::llm::types::TokenUsage,
}

/// Errors emitted by [`run_x_announcement_pipeline`].
#[derive(Debug, thiserror::Error)]
pub enum XAnnouncementError {
    /// Writer stage failure.
    #[error("writer: {0}")]
    Writer(String),
    /// Length normalization fatally rejected the draft (rare).
    #[error("length normalize: {0}")]
    LengthNormalize(String),
    /// Telegram delivery error.
    #[error("delivery: {0}")]
    Delivery(#[from] crate::review::ReviewDeliveryError),
    /// Invalid config.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// System prompt for the X-announcement writer.
pub const X_ANNOUNCE_WRITER_PROMPT: &str = "You are writing an X (Twitter) announcement thread for a long-form essay the operator just published on their personal blog.\n\n\
Rules:\n\
- Produce 3-5 tweets. Each tweet ≤280 characters (hard cap).\n\
- Tweet 1 = the hook. Lead with the most surprising claim from the essay.\n\
- Middle tweets = the substance. Distill the argument into bite-sized chunks.\n\
- Final tweet MUST include the canonical blog URL. Format the URL on its own line.\n\
- Maintain the operator's voice (dhh/mitsuhiko-leaning, opinionated, no marketing-speak).\n\
- NO emojis. NO hashtags. NO 'Read more here'. Just substance + link.\n\
- Do NOT quote the essay verbatim. Re-state the argument in tweet-native form.\n\
- ZERO TOLERANCE FOR INVENTION: every claim in the thread MUST be supported by the body_snippet provided. If you can't say something from the snippet, omit it.\n\n\
Output format: one tweet per line. Empty lines between tweets are ignored.";

/// Build the user message for the X announcement writer.
pub fn build_x_announce_user_message(
    title: &str,
    excerpt: &str,
    body_snippet: &str,
    post_url: &str,
) -> String {
    format!(
        "Announce this essay on X. Use a 3-5 tweet thread.\n\n\
TITLE: {title}\n\n\
EXCERPT: {excerpt}\n\n\
BODY SNIPPET (only source of truth — do not invent beyond this):\n{body_snippet}\n\n\
CANONICAL URL (must appear in final tweet): {post_url}\n\n\
Write the thread now. One tweet per line. ≤280 chars each."
    )
}

/// Entry point for the X announcement pipeline (implementation in Task 5).
///
/// Drafts a tweet thread from the blog post context in `cfg`, sends it for
/// Telegram review, and publishes via `cfg.twitter_tool` on approval.
pub async fn run_x_announcement_pipeline(
    _cfg: XAnnouncementConfig<'_>,
) -> Result<XAnnouncementOutput, XAnnouncementError> {
    // TODO(Task 5): implement full pipeline.
    Err(XAnnouncementError::InvalidConfig(
        "run_x_announcement_pipeline not yet implemented".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_prompt_pins_load_bearing_rules() {
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("3-5 tweets"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("≤280"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("ZERO TOLERANCE FOR INVENTION"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("canonical blog URL"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("NO emojis"));
    }

    #[test]
    fn user_message_includes_url_and_title() {
        let msg = build_x_announce_user_message(
            "Agent loops cost money",
            "Why background loops compound costs.",
            "When you wrap a model in a loop, every tick is a separate API call...",
            "https://pascal.heartbit.ai/agent-loops/",
        );
        assert!(msg.contains("Agent loops cost money"));
        assert!(msg.contains("https://pascal.heartbit.ai/agent-loops/"));
        assert!(msg.contains("only source of truth"));
    }
}
