//! Review-mode pipeline — sends N candidate drafts to the user via a
//! [`ReviewDelivery`] (Telegram in production), awaits the user's pick,
//! then posts the chosen draft to X via the `twitter_thread` tool.
//!
//! Public entry: [`run_review_pipeline`].

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
    /// Persona-specific mode addendum surfaced in the writer's user
    /// message after voice_guidelines. None for personas that don't
    /// have one (heartbit-ghost:x).
    pub mode_addendum: Option<&'a str>,
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
    /// Whether `image_generator` produced an image that was attached to
    /// the head tweet. `false` when the recipe returned `"no_image"`,
    /// when the call failed, when the marker was absent, or when the
    /// outcome was Skipped/TimedOut/GateRejected/PublishFailed (i.e.,
    /// no post happened).
    pub head_image_attached: bool,
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

/// Parse the `twitter_thread` tool's success output into `(tweet_ids, head_url)`.
///
/// The tool's `ToolOutput.content` on success is the JSON serialization
/// of `ThreadOutput { thread_root_id, tweet_ids, urls }` (see
/// `crates/heartbit-ghost/src/tools/thread.rs`). On parse failure (e.g.,
/// the tool was mocked with a non-JSON string), returns `(vec![], "<unknown>")`
/// — the caller has already accepted that the post succeeded; treat the
/// missing structure as a non-fatal observability gap.
pub(crate) fn parse_twitter_thread_output(content: &str) -> (Vec<String>, String) {
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

/// Extract the base64 image data from `image_generator`'s output.
///
/// `ImageGenerateTool` emits matched output with the prefix
/// `[IMAGE:base64:` and a closing `]`. The body inside the brackets
/// can be one of three shapes (see `crates/heartbit-core/src/tool/builtins/image_generate.rs`):
///
/// 1. data-URL fragment: `image/png;base64,iVBOR...` (the `images[]`
///    and `image_url` paths)
/// 2. mime-prefixed without `base64,`: `image/png;iVBOR...` (the
///    `inline_data` fallback path)
/// 3. pure base64: `iVBOR...` (synthetic / future paths)
///
/// This helper strips any `mime[;base64],` prefix and returns the bare
/// base64 string for downstream decoding.
///
/// Returns `None` when:
/// - the marker is absent
/// - the recipe returned the literal `"no_image"` (case-insensitive)
/// - input is empty / whitespace
/// - the body between brackets is empty after prefix stripping
///
/// Neither base64 alphabet (`A–Z a–z 0–9 + / =`) nor the data-URL
/// prefix (`mime/subtype;base64,`) contains `]`, so `find(']')` after
/// the prefix is safe.
pub(crate) fn extract_image_marker(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.eq_ignore_ascii_case("no_image") {
        return None;
    }
    let prefix = "[IMAGE:base64:";
    let start = trimmed.find(prefix)?;
    let after_prefix = &trimmed[start + prefix.len()..];
    let end = after_prefix.find(']')?;
    let body = &after_prefix[..end];
    if body.is_empty() {
        return None;
    }
    // Normalize to pure base64. Strip any leading `mime[;base64],` or
    // `mime;` prefix; otherwise return body as-is.
    let b64 = if let Some((_, rest)) = body.split_once(";base64,") {
        rest
    } else if let Some((_, rest)) = body.split_once(';') {
        rest
    } else {
        body
    };
    if b64.is_empty() {
        return None;
    }
    Some(b64.to_string())
}

/// Execute one review-mode pipeline run.
///
/// Flow: snapshot load → research → N parallel writer→critic→fact_check
/// → dedup → ReviewDelivery::deliver_and_await → on Pick: publish_gate
/// → twitter_tool.execute → ReviewDelivery::report → return.
pub async fn run_review_pipeline(cfg: ReviewConfig<'_>) -> Result<ReviewOutput, ReviewError> {
    let progress = |msg: &str| {
        if let Some(cb) = cfg.on_progress.as_ref() {
            cb(msg);
        }
    };

    // 1. Validate config.
    if !(1..=10).contains(&cfg.candidates_per_draft) {
        return Err(ReviewError::InvalidConfig(format!(
            "candidates_per_draft must be in 1..=10 (got {})",
            cfg.candidates_per_draft,
        )));
    }
    // persona_pick_buttons asserts 1..=9. Constrain further here.
    if cfg.candidates_per_draft > 9 {
        return Err(ReviewError::InvalidConfig(format!(
            "review mode requires candidates_per_draft <= 9 \
             (Telegram inline-keyboard limit; got {})",
            cfg.candidates_per_draft,
        )));
    }

    // 2. Load StyleProfile snapshot — same as run_pipeline.
    progress("Loading profile snapshot...");
    let store = crate::voice::SnapshotStore::open(cfg.profiles_root, cfg.persona_name)
        .map_err(PipelineError::from)?;
    let snapshot = store
        .load_latest()
        .map_err(PipelineError::from)?
        .ok_or_else(|| PipelineError::NoProfileSnapshot {
            persona: cfg.persona_name.to_string(),
            profiles_dir: cfg.profiles_root.join(cfg.persona_name),
        })?;
    let profile = snapshot.profile;

    // 3. Build agents (researcher / writer / critic / fact_check ONLY —
    // no judge or image_generator in review mode).
    use crate::agents::{fact_check_recipe, researcher_recipe, style_critic_recipe, writer_recipe};
    use heartbit_core::tool::builtins::{WebFetchTool, WebSearchTool};

    let researcher_tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(WebSearchTool::new()),
        Arc::new(WebFetchTool::new()),
    ];
    let researcher = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        researcher_recipe(),
        researcher_tools,
    )
    .map_err(|e| PipelineError::Builder {
        stage: "researcher".to_string(),
        source: e,
    })?;
    let writer =
        crate::pipeline::runner_from_recipe(cfg.provider.clone(), writer_recipe(), Vec::new())
            .map_err(|e| PipelineError::Builder {
                stage: "writer".to_string(),
                source: e,
            })?;
    let critic = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        style_critic_recipe(),
        Vec::new(),
    )
    .map_err(|e| PipelineError::Builder {
        stage: "style_critic".to_string(),
        source: e,
    })?;
    let fact =
        crate::pipeline::runner_from_recipe(cfg.provider.clone(), fact_check_recipe(), Vec::new())
            .map_err(|e| PipelineError::Builder {
                stage: "fact_check".to_string(),
                source: e,
            })?;

    let mut total_usage = TokenUsage::default();

    // 4. Run researcher.
    progress("Researching topic...");
    let researcher_out = researcher
        .execute(cfg.topic)
        .await
        .map_err(|e| PipelineError::Agent {
            stage: "researcher".to_string(),
            source: e,
        })?;
    let research_digest = researcher_out.result.clone();
    total_usage += researcher_out.tokens_used;

    // 5. Render voice guidelines.
    let voice_guidelines = crate::pipeline::render_style_profile_as_english(&profile);

    // 6. Parallel candidate generation — same shape as run_pipeline.
    let n = cfg.candidates_per_draft;
    progress(&format!("Generating {n} candidate(s) in parallel..."));

    let writer = std::sync::Arc::new(writer);
    let critic = std::sync::Arc::new(critic);
    let fact = std::sync::Arc::new(fact);
    let topic_owned: String = cfg.topic.to_string();
    let digest_owned = std::sync::Arc::new(research_digest.clone());
    let voice_owned = std::sync::Arc::new(voice_guidelines.clone());
    // Convert to owned Arc<str> so the spawn closures are 'static.
    let mode_addendum_owned: Option<std::sync::Arc<str>> =
        cfg.mode_addendum.map(std::sync::Arc::from);

    let mut joinset: tokio::task::JoinSet<Result<CandidateRecord, PipelineError>> =
        tokio::task::JoinSet::new();
    for i in 0..n {
        let writer = writer.clone();
        let critic = critic.clone();
        let fact = fact.clone();
        let topic = topic_owned.clone();
        let digest = digest_owned.clone();
        let voice = voice_owned.clone();
        let mode_addendum = mode_addendum_owned.clone();
        joinset.spawn(async move {
            crate::pipeline::generate_candidate(
                i,
                n,
                &topic,
                &digest,
                &voice,
                &writer,
                &critic,
                &fact,
                mode_addendum.as_deref(),
            )
            .await
        });
    }

    let mut candidates: Vec<CandidateRecord> = Vec::with_capacity(n);
    let mut errors: Vec<PipelineError> = Vec::new();
    while let Some(res) = joinset.join_next().await {
        match res {
            Ok(Ok(rec)) => candidates.push(rec),
            Ok(Err(e)) => {
                progress(&format!("candidate failed: {e}"));
                errors.push(e);
            }
            Err(joinerr) => {
                progress(&format!("candidate task panicked: {joinerr}"));
            }
        }
    }
    candidates.sort_by_key(|c| c.variant_index);

    if candidates.is_empty() {
        if errors.len() == 1 {
            return Err(ReviewError::Pipeline(errors.swap_remove(0)));
        }
        return Err(ReviewError::Pipeline(PipelineError::AllCandidatesFailed {
            errors,
            n,
        }));
    }

    // 7. Dedup. (Skip the retry-once pass — review mode is OK with
    // ship-with-fewer; the user may pick from the surviving distinct set.)
    let candidates = crate::pipeline::dedup_candidates(candidates);

    // Sum per-candidate usage.
    for c in &candidates {
        total_usage += c.usage;
    }

    // 8. Build ReviewMessage.
    let interaction_id = uuid::Uuid::new_v4();
    let candidate_drafts: Vec<String> = candidates.iter().map(|c| c.draft.clone()).collect();
    let review_msg = ReviewMessage {
        persona_name: cfg.persona_name.to_string(),
        topic: cfg.topic.to_string(),
        candidates: candidate_drafts,
        interaction_id,
    };

    // 9. Deliver and await pick.
    progress("Sending review to user...");
    let delivered = cfg.delivery.deliver_and_await(&review_msg).await?;

    // 10. Branch on outcome.
    let (outcome, report, head_image_attached) = match delivered.outcome {
        DeliveryOutcome::Skip => {
            progress("User skipped.");
            (ReviewOutcome::Skipped, ReportableOutcome::Skipped, false)
        }
        DeliveryOutcome::TimedOut => {
            progress("Review timed out.");
            (ReviewOutcome::TimedOut, ReportableOutcome::TimedOut, false)
        }
        DeliveryOutcome::Pick(chosen_index) => {
            if chosen_index >= candidates.len() {
                return Err(ReviewError::InvalidConfig(format!(
                    "delivery returned out-of-range pick index {chosen_index} \
                     (candidates.len() = {})",
                    candidates.len()
                )));
            }
            let chosen = &candidates[chosen_index];

            // 10a. publish_gate.
            match crate::pipeline::check_publish_gate(&chosen.draft, &profile) {
                Err(gate_err) => {
                    let reason = format!("{gate_err}");
                    progress(&format!("publish_gate rejected pick: {reason}"));
                    (
                        ReviewOutcome::GateRejected {
                            chosen_index,
                            reason: reason.clone(),
                        },
                        ReportableOutcome::GateRejected {
                            chosen_index,
                            reason,
                        },
                        false,
                    )
                }
                Ok(()) => {
                    // 10b. NEW (P1.3f): run image_generator on the
                    // chosen draft. Failure is non-blocking — text-only
                    // post on any error.
                    progress("Generating optional image...");
                    let head_image_b64: Option<String> = {
                        let recipe = crate::agents::image_generator_recipe();
                        let image_tools: Vec<Arc<dyn Tool>> = vec![Arc::new(
                            heartbit_core::tool::builtins::ImageGenerateTool::new(),
                        )];
                        match crate::pipeline::runner_from_recipe(
                            cfg.provider.clone(),
                            recipe,
                            image_tools,
                        ) {
                            Ok(image_runner) => {
                                let voice =
                                    crate::pipeline::render_style_profile_as_english(&profile);
                                let msg = format!(
                                    "Approved draft:\n{}\n\n{}\n\n\
                                     Decide whether to attach an image. \
                                     If no, output the literal string \"no_image\". \
                                     If yes, call image_generate with a concise visual prompt and return the result.",
                                    chosen.draft, voice,
                                );
                                match image_runner.execute(&msg).await {
                                    Ok(out) => {
                                        total_usage += out.tokens_used;
                                        // P1.3g: prefer the raw `image_generate`
                                        // tool output (full marker, untruncated)
                                        // over the model's text response. The
                                        // tool result that re-entered the
                                        // conversation was redacted to a tiny
                                        // placeholder, so `out.result` would not
                                        // recover the base64 payload. The
                                        // fallback to `out.result` covers the
                                        // case where the agent decided
                                        // "no_image" without calling the tool.
                                        let raw_tool_output: Option<String> = out
                                            .tool_call_results
                                            .iter()
                                            .find(|r| {
                                                r.tool_name == "image_generate" && !r.is_error
                                            })
                                            .map(|r| r.output.clone());
                                        match raw_tool_output {
                                            Some(raw) => extract_image_marker(&raw),
                                            None => extract_image_marker(&out.result),
                                        }
                                    }
                                    Err(e) => {
                                        progress(&format!(
                                            "image_generator failed (non-blocking): {e}"
                                        ));
                                        None
                                    }
                                }
                            }
                            Err(e) => {
                                progress(&format!(
                                    "image_generator builder failed (non-blocking): {e}"
                                ));
                                None
                            }
                        }
                    };
                    let head_image_attached = head_image_b64.is_some();

                    // 10c. Post via twitter_tool (with optional head image).
                    progress(&format!("Posting candidate {chosen_index}..."));
                    let tweets = parse_thread_tweets(&chosen.draft);
                    let exec_ctx = heartbit_core::ExecutionContext {
                        credentials: Some(cfg.credentials.clone()),
                        ..Default::default()
                    };
                    let mut input = serde_json::json!({"tweets": tweets});
                    if let Some(b64) = head_image_b64.as_ref() {
                        input["head_image_b64"] = serde_json::Value::String(b64.clone());
                    }
                    let post_outcome = match cfg.twitter_tool.execute(&exec_ctx, input).await {
                        Err(e) => {
                            let reason = format!("{e}");
                            progress(&format!("twitter_tool errored: {reason}"));
                            (
                                ReviewOutcome::PublishFailed {
                                    chosen_index,
                                    reason: reason.clone(),
                                },
                                ReportableOutcome::PublishFailed {
                                    chosen_index,
                                    reason,
                                },
                                false, // head_image_attached
                            )
                        }
                        Ok(tool_out) if tool_out.is_error => {
                            let reason = tool_out.content.clone();
                            progress(&format!("twitter_tool returned is_error=true: {reason}"));
                            (
                                ReviewOutcome::PublishFailed {
                                    chosen_index,
                                    reason: reason.clone(),
                                },
                                ReportableOutcome::PublishFailed {
                                    chosen_index,
                                    reason,
                                },
                                false,
                            )
                        }
                        Ok(tool_out) => {
                            let (tweet_ids, tweet_url) =
                                parse_twitter_thread_output(&tool_out.content);
                            (
                                ReviewOutcome::Posted {
                                    chosen_index,
                                    tweet_url: tweet_url.clone(),
                                    tweet_ids,
                                },
                                ReportableOutcome::Posted {
                                    chosen_index,
                                    tweet_url,
                                },
                                head_image_attached, // true iff image actually generated AND post succeeded
                            )
                        }
                    };
                    (post_outcome.0, post_outcome.1, post_outcome.2)
                }
            }
        }
    };

    // 11. Report outcome (non-fatal on error).
    if let Err(e) = cfg.delivery.report(delivered.receipt, report).await {
        progress(&format!("report failed (non-fatal): {e}"));
    }

    progress("Done.");
    Ok(ReviewOutput {
        candidates,
        research_digest,
        usage_summary: total_usage,
        outcome,
        head_image_attached,
    })
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

    use heartbit_core::ExecutionContext;
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::CredentialResolver as CredentialResolverTrait;
    use heartbit_core::execution_context::Secret;
    use heartbit_core::llm::types::ToolDefinition;
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use heartbit_core::tool::ToolOutput;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use tempfile::TempDir;

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
        ) -> Pin<Box<dyn Future<Output = Result<DeliveredReview, ReviewDeliveryError>> + Send + 'a>>
        {
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
        ) -> Pin<Box<dyn Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>> {
            self.reports.lock().unwrap().push(outcome);
            Box::pin(async move { Ok(()) })
        }
    }

    /// MockTwitterTool returns a canned ToolOutput. Tests can configure
    /// the success body or set is_error=true.
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

        pub fn last_input(&self) -> Option<serde_json::Value> {
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

    /// MockProvider — same shape as pipeline::tests but local copy
    /// (the pipeline tests' MockProvider is `pub(super)`-scoped).
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

    /// Snapshot fixture — same shape as pipeline::tests::seed_snapshot.
    fn seed_snapshot(persona: &str) -> (TempDir, std::path::PathBuf) {
        use crate::voice::{
            BlendEntry, BlendRecipe, EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency,
            HashtagPolicy, LineBreaks, OpeningPattern, PartialStyleProfile, PeriodsPolicy,
            QuotationMarks, SentenceLengthTarget, SnapshotStore, SpecificityTarget, StyleProfile,
            ThreadRhythm,
        };
        let dir = TempDir::new().unwrap();
        let profile = StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![OpeningPattern::ClaimFirst],
            opening_pattern_weights: vec![1.0],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::Never,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec!["specific".to_string()],
            ai_tells_to_avoid: vec!["delve".to_string()],
            thread_rhythm: ThreadRhythm::Linear,
            thread_max_length: 5,
            thread_opener_must_hook: false,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        };
        let recipe = BlendRecipe {
            version: 1,
            blend: vec![BlendEntry {
                writer: "k".to_string(),
                weight: 1.0,
            }],
            overrides: PartialStyleProfile::default(),
        };
        let store = SnapshotStore::open(dir.path(), persona).unwrap();
        store.save_new(profile, &recipe).unwrap();
        let root = dir.path().to_path_buf();
        (dir, root)
    }

    /// Boilerplate: build a single-candidate ReviewConfig (saves repetition
    /// across tests). `provider` and `delivery` and `twitter_tool` are
    /// caller-provided.
    fn mk_review_cfg<'a>(
        profiles_root: &'a std::path::Path,
        provider: Arc<BoxedProvider>,
        delivery: Arc<dyn ReviewDelivery>,
        twitter_tool: Arc<dyn Tool>,
    ) -> ReviewConfig<'a> {
        ReviewConfig {
            persona_name: "x",
            topic: "agent harness",
            provider,
            corpora_root: profiles_root, // unused in tests; reuse path
            profiles_root,
            on_progress: None,
            candidates_per_draft: 1,
            delivery,
            twitter_tool,
            credentials: Arc::new(StubCredentialResolver),
            mode_addendum: None,
        }
    }

    #[tokio::test]
    async fn run_review_pipeline_pick_index_0_posts_to_twitter() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- AI is moving fast", // researcher
            "concrete short post",                   // writer iter 1
            r#"{"verdict": "pass", "style_match_score": 0.92}"#, // critic
            r#"{"verdict": "verified"}"#,            // fact_check
            "no_image",                              // image_generator (P1.3f)
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool: Arc<dyn Tool> = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg).await.expect("happy path");
        match out.outcome {
            ReviewOutcome::Posted {
                chosen_index,
                tweet_url,
                tweet_ids,
            } => {
                assert_eq!(chosen_index, 0);
                assert_eq!(tweet_url, "https://twitter.com/i/web/status/123");
                assert_eq!(tweet_ids, vec!["123".to_string()]);
            }
            other => panic!("expected Posted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_review_pipeline_skip_returns_skipped_no_post() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            "draft text",
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Skip);
        // twitter_tool should never be called — set it up to return an
        // error so we'd notice if it was invoked.
        let twitter_tool: Arc<dyn Tool> = MockTwitterTool::errored("should not be called");
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg).await.expect("skip is success");
        assert!(matches!(out.outcome, ReviewOutcome::Skipped));
    }

    #[tokio::test]
    async fn run_review_pipeline_timed_out_returns_timed_out_no_post() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            "draft text",
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::TimedOut);
        let twitter_tool: Arc<dyn Tool> = MockTwitterTool::errored("should not be called");
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg).await.expect("timeout is success");
        assert!(matches!(out.outcome, ReviewOutcome::TimedOut));
    }

    #[tokio::test]
    async fn run_review_pipeline_pick_publish_gate_rejects_long_draft() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let too_long = "x".repeat(290);
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            too_long.as_str(),
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool: Arc<dyn Tool> = MockTwitterTool::errored("should not be called");
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg)
            .await
            .expect("gate rejection is success");
        match out.outcome {
            ReviewOutcome::GateRejected {
                chosen_index,
                reason,
            } => {
                assert_eq!(chosen_index, 0);
                assert!(reason.contains("280"), "got: {reason}");
            }
            other => panic!("expected GateRejected, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_review_pipeline_pick_twitter_api_error_returns_publish_failed() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            "short post",
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
            "no_image", // image_generator (P1.3f)
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool: Arc<dyn Tool> =
            MockTwitterTool::errored("X auth failed (401): Unauthorized");
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg)
            .await
            .expect("publish failure is success");
        match out.outcome {
            ReviewOutcome::PublishFailed {
                chosen_index,
                reason,
            } => {
                assert_eq!(chosen_index, 0);
                assert!(reason.contains("401"), "got: {reason}");
            }
            other => panic!("expected PublishFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_review_pipeline_invalid_candidates_per_draft_rejected() {
        // Build a minimal cfg with candidates_per_draft = 0 (invalid).
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Skip);
        let twitter_tool: Arc<dyn Tool> = MockTwitterTool::errored("never called");
        let mut cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        cfg.candidates_per_draft = 0;
        let err = run_review_pipeline(cfg).await.unwrap_err();
        match err {
            ReviewError::InvalidConfig(msg) => {
                assert!(msg.contains("candidates_per_draft"), "got: {msg}");
                assert!(msg.contains("1..=10"), "got: {msg}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    // --- P1.3f: extract_image_marker unit tests ---

    #[test]
    fn extract_image_marker_happy_path() {
        let raw = "preamble [IMAGE:base64:iVBORw0KGgo=] suffix";
        let out = extract_image_marker(raw);
        assert_eq!(out.as_deref(), Some("iVBORw0KGgo="));
    }

    #[test]
    fn extract_image_marker_no_image_returns_none() {
        assert!(extract_image_marker("no_image").is_none());
        assert!(extract_image_marker("NO_IMAGE").is_none());
        assert!(extract_image_marker("  no_image  ").is_none());
    }

    #[test]
    fn extract_image_marker_absent_marker_returns_none() {
        assert!(extract_image_marker("just some text").is_none());
        assert!(extract_image_marker("").is_none());
        // Marker prefix without closing bracket → None.
        assert!(extract_image_marker("[IMAGE:base64:abc").is_none());
        // Empty body between brackets → None.
        assert!(extract_image_marker("[IMAGE:base64:]").is_none());
    }

    #[test]
    fn extract_image_marker_strips_data_url_prefix() {
        // Production marker format from ImageGenerateTool's `images[]` /
        // `image_url` paths (data: prefix already stripped by the tool):
        // body = "image/png;base64,iVBOR..."
        let raw = "[IMAGE:base64:image/png;base64,iVBORw0KGgo=]\n\nGenerated image for: test";
        assert_eq!(extract_image_marker(raw).as_deref(), Some("iVBORw0KGgo="));
    }

    #[test]
    fn extract_image_marker_strips_mime_only_prefix() {
        // Production marker format from ImageGenerateTool's `inline_data`
        // fallback path: body = "image/png;iVBOR..." (no `base64,` separator).
        let raw = "[IMAGE:base64:image/png;iVBORw0KGgo=]";
        assert_eq!(extract_image_marker(raw).as_deref(), Some("iVBORw0KGgo="));
    }

    #[test]
    fn extract_image_marker_leaves_pure_base64_unchanged() {
        // Already-canonical body (no mime prefix).
        let raw = "[IMAGE:base64:iVBORw0KGgo=]";
        assert_eq!(extract_image_marker(raw).as_deref(), Some("iVBORw0KGgo="));
    }

    // --- P1.3f: integration tests covering image_generator flow ---

    #[tokio::test]
    async fn run_review_pipeline_pick_with_image_generator_attaches_head_image() {
        let (_dir, profiles_root) = seed_snapshot("x");
        // Response queue: researcher, writer, critic, fact_check, then
        // image_generator (returns base64 marker), then... twitter_tool
        // (mocked separately, doesn't consume from the LLM queue).
        let provider = MockProvider::arc(vec![
            "Research digest:\n- AI is moving fast", // researcher
            "concrete short post",                   // writer iter 1
            r#"{"verdict": "pass", "style_match_score": 0.92}"#, // critic iter 1
            r#"{"verdict": "verified"}"#,            // fact_check
            "[IMAGE:base64:iVBORw0KGgo=]",           // image_generator
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );
        let cfg = mk_review_cfg(
            &profiles_root,
            provider,
            delivery,
            twitter_tool.clone() as Arc<dyn Tool>,
        );
        let out = run_review_pipeline(cfg)
            .await
            .expect("happy path with image");
        assert!(matches!(out.outcome, ReviewOutcome::Posted { .. }));
        assert!(
            out.head_image_attached,
            "image_generator returned a marker; head_image_attached must be true"
        );
        // Assert the twitter_tool received head_image_b64 in its input.
        let last_input = twitter_tool.last_input().expect("twitter_tool was called");
        assert_eq!(
            last_input.get("head_image_b64").and_then(|v| v.as_str()),
            Some("iVBORw0KGgo="),
            "twitter_tool input did not carry head_image_b64; got: {last_input}"
        );
    }

    #[tokio::test]
    async fn run_review_pipeline_pick_image_generator_no_image_posts_text_only() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            "draft text",
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
            "no_image", // image_generator opts out
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"456","tweet_ids":["456"],"urls":["https://twitter.com/i/web/status/456"]}"#,
        );
        let cfg = mk_review_cfg(
            &profiles_root,
            provider,
            delivery,
            twitter_tool.clone() as Arc<dyn Tool>,
        );
        let out = run_review_pipeline(cfg).await.expect("no_image path");
        assert!(matches!(out.outcome, ReviewOutcome::Posted { .. }));
        assert!(
            !out.head_image_attached,
            "image_generator returned no_image; head_image_attached must be false"
        );
        let last_input = twitter_tool.last_input().expect("twitter_tool was called");
        assert!(
            last_input.get("head_image_b64").is_none(),
            "no_image path must NOT pass head_image_b64; got: {last_input}"
        );
    }
}
