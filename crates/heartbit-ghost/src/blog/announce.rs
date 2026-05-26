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

use std::path::Path;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::llm::types::TokenUsage;
use heartbit_core::tool::Tool;

use crate::agents::writer_recipe;
use crate::pipeline::{ProgressCallback, normalize_tweet_length, runner_from_recipe};
use crate::review::{
    DeliveryOutcome, ReportableOutcome, ReviewDelivery, ReviewMessage, parse_thread_tweets,
};

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
- Produce 3-5 tweets TOTAL, and that count INCLUDES the final link tweet. Each tweet ≤280 characters (hard cap).\n\
- Tweet 1 = the hook. Lead with the most surprising claim from the essay.\n\
- Middle tweets = the substance. Distill the argument into bite-sized chunks.\n\
- The thread ends with EXACTLY ONE tweet that pairs a short call-to-read with the canonical blog URL (e.g. \"The full essay: <url>\"). The URL appears exactly ONCE in the whole thread — only in this final tweet.\n\
- NEVER emit a tweet that is just a bare URL with no sentence. Do NOT repeat the URL across multiple tweets.\n\
- Maintain the operator's voice (dhh/mitsuhiko-leaning, opinionated, no marketing-speak).\n\
- NO emojis. NO hashtags. NO 'Read more here'. Just substance + one closing link line.\n\
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

/// Run the X announcement pipeline end-to-end.
///
/// Drafts a tweet thread from the blog post context in `cfg`, length-normalizes
/// it to 280 chars/segment, sends it for Telegram review, and publishes via
/// `cfg.twitter_tool` on approval. Mirrors `run_review_pipeline` minus the
/// researcher / fact_check / publish_gate / image stages (the blog post is the
/// pre-vetted source of truth).
pub async fn run_x_announcement_pipeline(
    cfg: XAnnouncementConfig<'_>,
) -> Result<XAnnouncementOutput, XAnnouncementError> {
    let progress = |msg: &str| {
        if let Some(cb) = cfg.on_progress.as_ref() {
            cb(msg);
        }
    };

    let mut total_usage = TokenUsage::default();

    // 1. Build the writer with the X-announcement-specific system prompt.
    progress("Drafting announcement thread...");
    let voice_provider = cfg
        .writer_provider
        .clone()
        .unwrap_or_else(|| cfg.provider.clone());
    let mut recipe = writer_recipe();
    recipe.system_prompt = X_ANNOUNCE_WRITER_PROMPT.to_string();
    // Allow more tokens than the default writer (1024) — a 3-5 tweet thread
    // plus reasoning headroom benefits from a higher cap.
    recipe.max_tokens = Some(2048);
    let writer = runner_from_recipe(voice_provider, recipe, Vec::new())
        .map_err(|e| XAnnouncementError::Writer(e.to_string()))?;

    // 2. Run the writer.
    let user_msg =
        build_x_announce_user_message(cfg.title, cfg.excerpt, cfg.body_snippet, cfg.post_url);
    let writer_out = writer
        .execute(&user_msg)
        .await
        .map_err(|e| XAnnouncementError::Writer(e.to_string()))?;
    total_usage += writer_out.tokens_used;
    let raw_draft = writer_out.result;

    // 3. Length-normalize each segment to 280 chars.
    let normalized = normalize_tweet_length(&raw_draft, 280);

    // 4. Telegram review (single candidate — the blog post is the source of
    // truth, so a multi-candidate fanout adds noise without information).
    progress("Sending to Telegram for review...");
    let review_msg = ReviewMessage {
        persona_name: cfg.persona_name.to_string(),
        topic: format!("Announcement: {}", cfg.title),
        candidates: vec![normalized.clone()],
        interaction_id: uuid::Uuid::new_v4(),
    };
    let delivered = cfg.delivery.deliver_and_await(&review_msg).await?;
    let receipt = delivered.receipt;

    // 5. Branch on outcome.
    let (outcome, report) = match delivered.outcome {
        DeliveryOutcome::Skip => {
            progress("User skipped.");
            (XAnnouncementOutcome::Skipped, ReportableOutcome::Skipped)
        }
        DeliveryOutcome::TimedOut => {
            progress("Review timed out.");
            (XAnnouncementOutcome::TimedOut, ReportableOutcome::TimedOut)
        }
        DeliveryOutcome::Pick(_chosen_index) => {
            // 6. Publish via twitter_tool.
            progress("Publishing thread to X...");
            let tweets = parse_thread_tweets(&normalized);
            let exec_ctx = heartbit_core::ExecutionContext {
                credentials: Some(cfg.credentials.clone()),
                ..Default::default()
            };
            let input = serde_json::json!({ "tweets": tweets });
            match cfg.twitter_tool.execute(&exec_ctx, input).await {
                Err(e) => {
                    let reason = format!("{e}");
                    progress(&format!("twitter_tool errored: {reason}"));
                    (
                        XAnnouncementOutcome::PublishFailed {
                            reason: reason.clone(),
                        },
                        ReportableOutcome::PublishFailed {
                            chosen_index: 0,
                            reason,
                        },
                    )
                }
                Ok(tool_out) if tool_out.is_error => {
                    let reason = tool_out.content.clone();
                    progress(&format!("twitter_tool returned is_error=true: {reason}"));
                    (
                        XAnnouncementOutcome::PublishFailed {
                            reason: reason.clone(),
                        },
                        ReportableOutcome::PublishFailed {
                            chosen_index: 0,
                            reason,
                        },
                    )
                }
                Ok(tool_out) => {
                    let (tweet_ids, head_url) = parse_thread_output(&tool_out.content);
                    (
                        XAnnouncementOutcome::Posted {
                            tweet_ids,
                            head_url: head_url.clone(),
                        },
                        ReportableOutcome::Posted {
                            chosen_index: 0,
                            tweet_url: head_url,
                        },
                    )
                }
            }
        }
    };

    // 7. Report final state back to delivery (best-effort).
    if let Err(e) = cfg.delivery.report(receipt, report).await {
        progress(&format!("report failed (non-fatal): {e}"));
    }

    progress("Done.");
    Ok(XAnnouncementOutput {
        outcome,
        usage_summary: total_usage,
    })
}

/// Parse `twitter_tool` success output JSON: `{thread_root_id, tweet_ids, urls}`.
/// Returns `(tweet_ids, head_url)`. On parse failure returns
/// `(vec![], "<unknown>")` — the caller has already accepted the post as
/// successful; missing structure is a non-fatal observability gap.
fn parse_thread_output(content: &str) -> (Vec<String>, String) {
    #[derive(serde::Deserialize)]
    struct Parsed {
        tweet_ids: Vec<String>,
        urls: Vec<String>,
    }
    match serde_json::from_str::<Parsed>(content) {
        Ok(p) => {
            let head_url = p.urls.first().cloned().unwrap_or_else(|| {
                p.tweet_ids
                    .first()
                    .map(|id| format!("https://twitter.com/i/web/status/{id}"))
                    .unwrap_or_else(|| "<unknown>".to_string())
            });
            (p.tweet_ids, head_url)
        }
        Err(_) => (Vec::new(), "<unknown>".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::review::{DeliveredReview, DeliveryReceipt};
    use heartbit_core::ExecutionContext;
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::CredentialResolver as CredentialResolverTrait;
    use heartbit_core::execution_context::Secret;
    use heartbit_core::llm::LlmProvider;
    use heartbit_core::llm::types::ToolDefinition;
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason,
    };
    use heartbit_core::tool::ToolOutput;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;

    // ── Mocks (mirrors crates/heartbit-ghost/src/review/mod.rs::tests) ────────

    /// MockReviewDelivery returns a pre-canned outcome and records report() calls.
    struct MockReviewDelivery {
        outcome: DeliveryOutcome,
        reports: Mutex<Vec<ReportableOutcome>>,
    }

    impl MockReviewDelivery {
        fn arc(outcome: DeliveryOutcome) -> Arc<dyn ReviewDelivery> {
            Arc::new(MockReviewDelivery {
                outcome,
                reports: Mutex::new(Vec::new()),
            })
        }
    }

    impl ReviewDelivery for MockReviewDelivery {
        fn deliver_and_await<'a>(
            &'a self,
            _message: &'a ReviewMessage,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<DeliveredReview, crate::review::ReviewDeliveryError>>
                    + Send
                    + 'a,
            >,
        > {
            let outcome = self.outcome.clone();
            Box::pin(async move {
                Ok(DeliveredReview {
                    outcome,
                    receipt: DeliveryReceipt {
                        data: serde_json::Value::Null,
                    },
                })
            })
        }

        fn report<'a>(
            &'a self,
            _receipt: DeliveryReceipt,
            outcome: ReportableOutcome,
        ) -> Pin<Box<dyn Future<Output = Result<(), crate::review::ReviewDeliveryError>> + Send + 'a>>
        {
            self.reports.lock().unwrap().push(outcome);
            Box::pin(async move { Ok(()) })
        }
    }

    /// MockTwitterTool returns a canned ToolOutput and records the last input.
    struct MockTwitterTool {
        canned: Mutex<Option<ToolOutput>>,
        last_input: Mutex<Option<serde_json::Value>>,
    }

    impl MockTwitterTool {
        fn success(thread_json: &str) -> Arc<Self> {
            Arc::new(MockTwitterTool {
                canned: Mutex::new(Some(ToolOutput::success(thread_json))),
                last_input: Mutex::new(None),
            })
        }

        fn errored(reason: &str) -> Arc<Self> {
            Arc::new(MockTwitterTool {
                canned: Mutex::new(Some(ToolOutput::error(reason))),
                last_input: Mutex::new(None),
            })
        }

        fn last_input(&self) -> Option<serde_json::Value> {
            self.last_input.lock().unwrap().clone()
        }
    }

    impl Tool for MockTwitterTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "twitter_thread".to_string(),
                description: "mock".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, CoreError>> + Send + '_>> {
            *self.last_input.lock().unwrap() = Some(input);
            let canned = self.canned.lock().unwrap().take();
            Box::pin(async move {
                canned.ok_or_else(|| CoreError::Agent("mock twitter tool exhausted".into()))
            })
        }
    }

    /// MockProvider returns canned completion text (one entry per writer call).
    struct MockProvider {
        responses: Mutex<VecDeque<String>>,
    }

    impl MockProvider {
        fn arc(responses: Vec<&str>) -> Arc<BoxedProvider> {
            let p = MockProvider {
                responses: Mutex::new(responses.into_iter().map(String::from).collect()),
            };
            Arc::new(BoxedProvider::new(p))
        }
    }

    impl LlmProvider for MockProvider {
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CoreError>> + Send {
            let response = self.responses.lock().unwrap().pop_front();
            let has_respond = request
                .tools
                .iter()
                .any(|t| t.name == heartbit_core::llm::types::RESPOND_TOOL_NAME);
            async move {
                let text =
                    response.ok_or_else(|| CoreError::Agent("mock exhausted".to_string()))?;
                let content = if has_respond {
                    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                        CoreError::Agent(format!("mock: canned response is not valid JSON: {e}"))
                    })?;
                    vec![ContentBlock::ToolUse {
                        id: "respond_1".to_string(),
                        name: "__respond__".to_string(),
                        input: value,
                    }]
                } else {
                    vec![ContentBlock::Text { text }]
                };
                Ok(CompletionResponse {
                    content,
                    usage: TokenUsage::default(),
                    stop_reason: if has_respond {
                        StopReason::ToolUse
                    } else {
                        StopReason::EndTurn
                    },
                    model: None,
                })
            }
        }
    }

    /// Stub credential resolver — never called in mock tests.
    struct StubCredentialResolver;

    impl CredentialResolverTrait for StubCredentialResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(Secret::new("stub")) })
        }
    }

    /// A 3-tweet thread draft (one tweet per line). The final tweet carries
    /// the canonical URL, mirroring the X_ANNOUNCE_WRITER_PROMPT contract.
    const DRAFT_THREAD: &str = "Background loops quietly compound API cost.\n\
Every tick is a fresh model call — the loop never amortizes.\n\
Full argument: https://pascal.heartbit.ai/agent-loops/";

    /// Build an `XAnnouncementConfig` with the supplied collaborators.
    fn mk_cfg<'a>(
        profiles_root: &'a std::path::Path,
        provider: Arc<BoxedProvider>,
        delivery: Arc<dyn ReviewDelivery>,
        twitter_tool: Arc<dyn Tool>,
    ) -> XAnnouncementConfig<'a> {
        XAnnouncementConfig {
            persona_name: "x",
            provider,
            writer_provider: None,
            corpora_root: profiles_root,
            profiles_root,
            on_progress: None,
            title: "Agent loops cost money",
            excerpt: "Why background loops compound costs.",
            body_snippet: "When you wrap a model in a loop, every tick is a separate API call.",
            post_url: "https://pascal.heartbit.ai/agent-loops/",
            delivery,
            twitter_tool,
            credentials: Arc::new(StubCredentialResolver),
        }
    }

    #[tokio::test]
    async fn pipeline_pick_publishes_and_returns_posted() {
        let provider = MockProvider::arc(vec![DRAFT_THREAD]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool: Arc<dyn Tool> = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123","124"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );
        let cfg = mk_cfg(
            std::path::Path::new("/tmp"),
            provider,
            delivery,
            twitter_tool,
        );
        let out = run_x_announcement_pipeline(cfg).await.expect("happy path");
        match out.outcome {
            XAnnouncementOutcome::Posted {
                tweet_ids,
                head_url,
            } => {
                assert_eq!(tweet_ids, vec!["123".to_string(), "124".to_string()]);
                assert_eq!(head_url, "https://twitter.com/i/web/status/123");
            }
            other => panic!("expected Posted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pipeline_skip_returns_skipped_no_publish() {
        let provider = MockProvider::arc(vec![DRAFT_THREAD]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Skip);
        // Concrete handle so we can assert the tool was never invoked.
        let twitter_tool = MockTwitterTool::errored("should not be called");
        let twitter_tool_dyn: Arc<dyn Tool> = twitter_tool.clone();
        let cfg = mk_cfg(
            std::path::Path::new("/tmp"),
            provider,
            delivery,
            twitter_tool_dyn,
        );
        let out = run_x_announcement_pipeline(cfg)
            .await
            .expect("skip is a success outcome");
        assert!(matches!(out.outcome, XAnnouncementOutcome::Skipped));
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be invoked on Skip"
        );
    }

    #[tokio::test]
    async fn pipeline_timeout_returns_timed_out() {
        let provider = MockProvider::arc(vec![DRAFT_THREAD]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::TimedOut);
        let twitter_tool = MockTwitterTool::errored("should not be called");
        let twitter_tool_dyn: Arc<dyn Tool> = twitter_tool.clone();
        let cfg = mk_cfg(
            std::path::Path::new("/tmp"),
            provider,
            delivery,
            twitter_tool_dyn,
        );
        let out = run_x_announcement_pipeline(cfg)
            .await
            .expect("timeout is a success outcome");
        assert!(matches!(out.outcome, XAnnouncementOutcome::TimedOut));
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be invoked on TimedOut"
        );
    }

    #[tokio::test]
    async fn pipeline_publish_failure_returns_publish_failed() {
        let provider = MockProvider::arc(vec![DRAFT_THREAD]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool: Arc<dyn Tool> =
            MockTwitterTool::errored("X auth failed (401): Unauthorized");
        let cfg = mk_cfg(
            std::path::Path::new("/tmp"),
            provider,
            delivery,
            twitter_tool,
        );
        let out = run_x_announcement_pipeline(cfg)
            .await
            .expect("publish failure is a success outcome");
        match out.outcome {
            XAnnouncementOutcome::PublishFailed { reason } => {
                assert!(reason.contains("401"), "got: {reason}");
            }
            other => panic!("expected PublishFailed, got {other:?}"),
        }
    }

    #[test]
    fn writer_prompt_pins_load_bearing_rules() {
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("3-5 tweets"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("≤280"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("ZERO TOLERANCE FOR INVENTION"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("canonical blog URL"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("NO emojis"));
        // Pin the single-link rule that prevents trailing bare-link tweets.
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("EXACTLY ONE tweet"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("bare URL"));
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

    #[test]
    fn parse_thread_output_extracts_ids_and_head_url() {
        let content = r#"{"thread_root_id":"1","tweet_ids":["1","2"],"urls":["https://x.com/i/web/status/1"]}"#;
        let (ids, url) = parse_thread_output(content);
        assert_eq!(ids, vec!["1".to_string(), "2".to_string()]);
        assert_eq!(url, "https://x.com/i/web/status/1");
    }

    #[test]
    fn parse_thread_output_falls_back_to_twitter_url_when_urls_missing() {
        // Empty `urls` array → synthesize from first tweet_id.
        let content = r#"{"thread_root_id":"1","tweet_ids":["42"],"urls":[]}"#;
        let (ids, url) = parse_thread_output(content);
        assert_eq!(ids, vec!["42".to_string()]);
        assert_eq!(url, "https://twitter.com/i/web/status/42");
    }

    #[test]
    fn parse_thread_output_invalid_json_returns_unknown() {
        let (ids, url) = parse_thread_output("not json");
        assert!(ids.is_empty());
        assert_eq!(url, "<unknown>");
    }

    #[test]
    fn parse_thread_output_missing_fields_returns_unknown() {
        // Missing `urls` field → serde rejects → fallback path.
        let (ids, url) = parse_thread_output(r#"{"thread_root_id":"1","tweet_ids":["1"]}"#);
        assert!(ids.is_empty());
        assert_eq!(url, "<unknown>");
    }
}
