//! Quote-tweet pipeline — polls curated source accounts, drafts
//! opinionated-but-charitable quote-tweets via the `quote_writer`
//! agent, routes through Telegram review, posts via `twitter_quote`.
//!
//! The runtime is [`run_quote_pipeline`]; helper types and the
//! [`QuoteReviewDelivery`] trait live here too. Mirrors
//! [`crate::reply::run_reply_pipeline`] closely — the major deviations are
//! (a) the writer output is length-normalized BEFORE the critic / fact
//! pass and (b) a pre-filter sweep drops Unverifiable + gate-failing
//! candidates before delivery.

pub(crate) mod prompts;
pub mod sources;

pub use sources::{
    InMemoryQuoteSeenStore, JsonlQuoteSeenStore, QuoteCandidate, QuoteSeenStore, QuoteSource,
    QuoteStoreError, XUserTimelineSource,
};

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::llm::types::TokenUsage;
use thiserror::Error;

use crate::pipeline::{FactVerdict, PipelineError, ProgressCallback};

/// Configuration for one quote-pipeline run.
#[derive(Clone)]
pub struct QuoteConfig<'a> {
    /// Persona name.
    pub persona_name: &'a str,
    /// LLM provider used for every sub-agent in this run by default.
    pub provider: Arc<BoxedProvider>,
    /// Optional override for the writer + critic stages. When `Some`, the
    /// writer and critic route through this provider instead of
    /// `provider`. Researcher + fact_check always use `provider`.
    pub writer_provider: Option<Arc<BoxedProvider>>,
    /// Root directory containing per-persona corpora.
    pub corpora_root: &'a Path,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: &'a Path,
    /// Optional progress callback.
    pub on_progress: Option<ProgressCallback>,
    /// The source tweet being quoted.
    pub source: QuoteCandidate,
    /// Number of distinct candidate quote-tweets to generate (1..=3).
    pub candidates_per_quote: usize,
    /// Telegram-or-mock delivery layer for the quote review.
    pub delivery: Arc<dyn QuoteReviewDelivery>,
    /// `twitter_quote` tool — production wires `Arc::new(TwitterQuoteTool::new())`;
    /// tests wire a mock.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver for `twitter_tool`.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Output of a successful quote-pipeline run.
#[derive(Debug, Clone)]
pub struct QuoteOutput {
    /// X tweet ID of the source tweet this run was for.
    pub source_id: String,
    /// One generated quote-tweet candidate per surviving slot.
    pub candidates: Vec<QuoteCandidateRecord>,
    /// Researcher digest produced during this run. Empty string when the
    /// pipeline short-circuits before the researcher stage (e.g. invalid
    /// config), or when no research was performed.
    pub research_digest: String,
    /// Aggregated token usage across all sub-agents.
    pub usage_summary: TokenUsage,
    /// Final outcome of the run.
    pub outcome: QuoteOutcome,
}

/// One generated quote-tweet draft (post-style/fact, post-length-normalize,
/// pre-publish_gate).
#[derive(Debug, Clone)]
pub struct QuoteCandidateRecord {
    /// The composed quote-tweet text (already length-normalized).
    pub draft: String,
    /// Voice-match score from the style critic, 0.0..=1.0.
    pub style_match_score: f32,
    /// Parsed fact-check verdict. `Unverifiable` candidates are dropped
    /// by the pre-filter sweep before delivery.
    pub fact_check_verdict: FactVerdict,
}

/// What happened in this quote run.
#[derive(Debug, Clone)]
pub enum QuoteOutcome {
    /// User picked candidate `chosen_index` and the quote-tweet was published.
    Posted {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// X tweet ID of the published quote-tweet.
        quote_tweet_id: String,
        /// Public URL of the published quote-tweet.
        quote_url: String,
    },
    /// User pressed Skip.
    Skipped,
    /// Telegram review timed out without a pick.
    TimedOut,
    /// User picked `chosen_index` but the post-pick `publish_gate`
    /// rejected it. The pre-filter sweep should have caught this, so
    /// this variant is reached only via defense-in-depth.
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
    /// All candidates returned the literal "no_quote" string — the
    /// writer chose not to engage. No Telegram review was sent.
    NoQuote,
    /// Every candidate was dropped by the pre-filter sweep (Unverifiable
    /// fact verdict and/or `publish_gate` rejection). No Telegram
    /// review was sent.
    AllCandidatesGateRejected {
        /// Per-candidate drop reasons, parallel to the order they were
        /// generated (post-no_quote-filter).
        reasons: Vec<String>,
    },
}

/// Errors raised by `run_quote_pipeline`.
#[derive(Debug, Error)]
pub enum QuoteError {
    /// Underlying pipeline error.
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Telegram delivery error.
    #[error("delivery: {0}")]
    Delivery(#[from] crate::review::ReviewDeliveryError),
    /// Misconfigured `QuoteConfig`.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Telegram-or-mock review delivery for quote-tweet messages. Mirrors
/// [`crate::reply::ReplyReviewDelivery`].
///
/// Methods use the project's `Pin<Box<dyn Future>>` desugaring to stay
/// object-safe without the `async-trait` crate.
pub trait QuoteReviewDelivery: Send + Sync {
    /// Deliver a quote review to the user (Telegram bot or mock).
    fn deliver<'a>(
        &'a self,
        msg: QuoteReviewMessage,
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
        outcome: QuoteOutcome,
    ) -> Pin<Box<dyn Future<Output = Result<(), crate::review::ReviewDeliveryError>> + Send + 'a>>;
}

/// Message body for a quote review.
#[derive(Debug, Clone)]
pub struct QuoteReviewMessage {
    /// The source tweet being quoted.
    pub source: QuoteCandidate,
    /// One drafted quote-tweet per slot.
    pub candidates: Vec<String>,
    /// How long to wait for a user pick (Telegram impl uses this).
    pub interaction_timeout_seconds: u64,
}

/// Execute one quote-tweet pipeline. Returns when the user picks (and the
/// quote-tweet posts), skips, times out, all candidates return "no_quote",
/// or every candidate is dropped by the pre-filter sweep.
pub async fn run_quote_pipeline(cfg: QuoteConfig<'_>) -> Result<QuoteOutput, QuoteError> {
    use crate::agents::{
        fact_check_recipe, quote_writer_recipe, researcher_recipe, style_critic_recipe,
    };
    use heartbit_core::llm::types::TokenUsage;
    use heartbit_core::tool::builtins::{WebFetchTool, WebSearchTool};

    // 1. Validate.
    if !(1..=3).contains(&cfg.candidates_per_quote) {
        return Err(QuoteError::InvalidConfig(format!(
            "candidates_per_quote must be in 1..=3 (got {})",
            cfg.candidates_per_quote,
        )));
    }

    let progress = |s: &str| {
        if let Some(cb) = cfg.on_progress.as_ref() {
            cb(s);
        }
    };

    // 2. Load profile snapshot.
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

    // 3. Build the 4 sub-agent runners.
    //    - researcher + fact_check use `cfg.provider` (canonical).
    //    - writer + critic optionally use `cfg.writer_provider` (override).
    let writer_provider = cfg
        .writer_provider
        .clone()
        .unwrap_or_else(|| cfg.provider.clone());

    // The quote pipeline does NOT support a researcher_override (unlike
    // the proactive `posts` path). Quote research always uses the public
    // web via WebSearchTool + WebFetchTool.
    let researcher_tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(WebSearchTool::new()),
        Arc::new(WebFetchTool::new()),
    ];
    let researcher = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        researcher_recipe(),
        researcher_tools,
    )
    .map_err(|source| PipelineError::Builder {
        stage: "researcher".to_string(),
        source,
    })?;
    let writer = crate::pipeline::runner_from_recipe(
        writer_provider.clone(),
        quote_writer_recipe(),
        Vec::new(),
    )
    .map_err(|source| PipelineError::Builder {
        stage: "quote_writer".to_string(),
        source,
    })?;
    let critic = crate::pipeline::runner_from_recipe(
        writer_provider.clone(),
        style_critic_recipe(),
        Vec::new(),
    )
    .map_err(|source| PipelineError::Builder {
        stage: "style_critic".to_string(),
        source,
    })?;
    let fact =
        crate::pipeline::runner_from_recipe(cfg.provider.clone(), fact_check_recipe(), Vec::new())
            .map_err(|source| PipelineError::Builder {
                stage: "fact_check".to_string(),
                source,
            })?;

    let mut total_usage = TokenUsage::default();
    let voice_guidelines = crate::pipeline::render_style_profile_as_english(&profile);

    // 4. Run researcher.
    progress("Researching source tweet...");
    let research_msg = prompts::build_quote_research_user_message(&cfg.source);
    let researcher_out =
        researcher
            .execute(&research_msg)
            .await
            .map_err(|source| PipelineError::Agent {
                stage: "researcher".to_string(),
                source,
            })?;
    let digest = researcher_out.result.clone();
    total_usage += researcher_out.tokens_used;

    // 5. Generate N quote candidates in parallel via tokio::JoinSet.
    progress(&format!(
        "Generating {} candidate(s) in parallel...",
        cfg.candidates_per_quote
    ));
    let writer = Arc::new(writer);
    let critic = Arc::new(critic);
    let fact = Arc::new(fact);
    let voice_owned: Arc<str> = voice_guidelines.clone().into();
    let digest_owned: Arc<str> = digest.clone().into();
    let source_owned = Arc::new(cfg.source.clone());
    // Detect the source tweet's language so the writer can mirror it.
    let quote_language = Arc::new(crate::reply::language::detect_reply_language(
        &cfg.source.text,
    ));
    tracing::info!(
        author = %cfg.source.author_handle,
        lang = %quote_language.code,
        lang_name = %quote_language.english_name,
        "detected quote language"
    );

    type CandidateResult = (String, f32, FactVerdict, TokenUsage);
    let mut joinset: tokio::task::JoinSet<Result<CandidateResult, PipelineError>> =
        tokio::task::JoinSet::new();
    for _ in 0..cfg.candidates_per_quote {
        let writer = writer.clone();
        let critic = critic.clone();
        let fact = fact.clone();
        let voice = voice_owned.clone();
        let digest = digest_owned.clone();
        let source = source_owned.clone();
        let lang = quote_language.clone();
        joinset.spawn(async move {
            let writer_msg =
                prompts::build_quote_writer_user_message(&digest, &source, &voice, &lang);
            let writer_out =
                writer
                    .execute(&writer_msg)
                    .await
                    .map_err(|source| PipelineError::Agent {
                        stage: "quote_writer".to_string(),
                        source,
                    })?;
            let raw_draft = writer_out.result.trim().to_string();
            // Writer-driven no_quote short-circuit (case-insensitive, as
            // with reply's `no_reply`).
            if raw_draft.eq_ignore_ascii_case("no_quote") {
                return Ok((
                    raw_draft,
                    0.0_f32,
                    FactVerdict::Verified, // unused for no_quote
                    writer_out.tokens_used,
                ));
            }
            // Deterministic length normalization BEFORE critic + fact_check
            // (matches `pipeline::generate_candidate`'s placement). This
            // ensures the critic/fact see the text we'll actually publish.
            let draft = crate::pipeline::normalize_tweet_length(
                &raw_draft,
                crate::pipeline::MAX_TWEET_CHARS,
            );
            // Style critic.
            let critic_msg = prompts::build_quote_critic_user_message(&draft, &voice);
            let critic_out =
                critic
                    .execute(&critic_msg)
                    .await
                    .map_err(|source| PipelineError::Agent {
                        stage: "style_critic".to_string(),
                        source,
                    })?;
            let style_score = parse_style_match_score(&critic_out.result).unwrap_or(0.5);
            // Fact check.
            let fact_msg = prompts::build_quote_fact_user_message(&draft, &digest);
            let fact_out =
                fact.execute(&fact_msg)
                    .await
                    .map_err(|source| PipelineError::Agent {
                        stage: "fact_check".to_string(),
                        source,
                    })?;
            let fact_verdict = crate::pipeline::parse_fact_verdict(&fact_out.result)
                .map_err(|source| PipelineError::FactCheckParseFailed { source })?;
            let mut usage = writer_out.tokens_used;
            usage += critic_out.tokens_used;
            usage += fact_out.tokens_used;
            Ok((draft, style_score, fact_verdict, usage))
        });
    }
    let mut survivors: Vec<QuoteCandidateRecord> = Vec::new();
    let mut saw_no_quote = false;
    while let Some(handle) = joinset.join_next().await {
        let (draft, style_score, fact_verdict, usage) =
            handle.map_err(|e| PipelineError::Agent {
                stage: "candidate".to_string(),
                source: heartbit_core::error::Error::Agent(format!("join: {e}")),
            })??;
        total_usage += usage;
        if draft.eq_ignore_ascii_case("no_quote") {
            saw_no_quote = true;
            continue;
        }
        survivors.push(QuoteCandidateRecord {
            draft,
            style_match_score: style_score,
            fact_check_verdict: fact_verdict,
        });
    }

    // 6. If every candidate was no_quote, return early without delivery.
    if survivors.is_empty() && saw_no_quote {
        return Ok(QuoteOutput {
            source_id: cfg.source.id.clone(),
            candidates: Vec::new(),
            research_digest: digest.clone(),
            usage_summary: total_usage,
            outcome: QuoteOutcome::NoQuote,
        });
    }

    // 7. Dedup near-duplicate drafts (lower index wins). Operates on the
    //    draft text via `pipeline::dedup::distinct_indices`.
    let survivors = dedup_quote_records(survivors);

    // 8. Pre-filter sweep — drop Unverifiable + publish_gate failures
    //    BEFORE delivery so the operator never sees a draft we can't ship.
    let mut filtered: Vec<QuoteCandidateRecord> = Vec::with_capacity(survivors.len());
    let mut drop_reasons: Vec<String> = Vec::new();
    for c in survivors {
        if let FactVerdict::Unverifiable { reason } = &c.fact_check_verdict {
            drop_reasons.push(format!("unverifiable: {reason}"));
            continue;
        }
        if let Err(e) = crate::pipeline::check_publish_gate(&c.draft, &profile) {
            drop_reasons.push(format!("publish_gate: {e}"));
            continue;
        }
        filtered.push(c);
    }

    if filtered.is_empty() {
        if saw_no_quote && drop_reasons.is_empty() {
            // (Defensive: should be unreachable — caught above.)
            return Ok(QuoteOutput {
                source_id: cfg.source.id.clone(),
                candidates: Vec::new(),
                research_digest: digest.clone(),
                usage_summary: total_usage,
                outcome: QuoteOutcome::NoQuote,
            });
        }
        return Ok(QuoteOutput {
            source_id: cfg.source.id.clone(),
            candidates: Vec::new(),
            research_digest: digest.clone(),
            usage_summary: total_usage,
            outcome: QuoteOutcome::AllCandidatesGateRejected {
                reasons: drop_reasons,
            },
        });
    }

    let survivors = filtered;

    // 9. Telegram review delivery.
    progress("Sending review to user...");
    let drafts_for_review: Vec<String> = survivors.iter().map(|c| c.draft.clone()).collect();
    let msg = QuoteReviewMessage {
        source: cfg.source.clone(),
        candidates: drafts_for_review,
        interaction_timeout_seconds: 300,
    };
    let delivered = cfg.delivery.deliver(msg).await?;
    let outcome = match delivered.outcome {
        crate::review::DeliveryOutcome::Pick(idx) if idx < survivors.len() => {
            // 10. Defense-in-depth: re-check publish_gate on the picked
            //     draft. The pre-filter should have removed any failures,
            //     but mirror reply/mod.rs's discipline.
            let chosen_draft = survivors[idx].draft.clone();
            if let Err(e) = crate::pipeline::check_publish_gate(&chosen_draft, &profile) {
                QuoteOutcome::GateRejected {
                    chosen_index: idx,
                    reason: format!("{e}"),
                }
            } else {
                // 11. twitter_quote tool call.
                progress(&format!("Posting candidate {idx}..."));
                let exec_ctx = heartbit_core::ExecutionContext {
                    credentials: Some(cfg.credentials.clone()),
                    ..Default::default()
                };
                let tool_input = serde_json::json!({
                    "text": chosen_draft,
                    "quote_tweet_id": cfg.source.id,
                });
                match cfg.twitter_tool.execute(&exec_ctx, tool_input).await {
                    Ok(out) if !out.is_error => {
                        let (tweet_id, url) = parse_quote_tool_output(&out.content);
                        QuoteOutcome::Posted {
                            chosen_index: idx,
                            quote_tweet_id: tweet_id,
                            quote_url: url,
                        }
                    }
                    Ok(out) => QuoteOutcome::PublishFailed {
                        chosen_index: idx,
                        reason: out.content,
                    },
                    Err(e) => QuoteOutcome::PublishFailed {
                        chosen_index: idx,
                        reason: format!("{e}"),
                    },
                }
            }
        }
        crate::review::DeliveryOutcome::Pick(_) => QuoteOutcome::Skipped, // unreachable
        crate::review::DeliveryOutcome::Skip => QuoteOutcome::Skipped,
        crate::review::DeliveryOutcome::TimedOut => QuoteOutcome::TimedOut,
    };

    // 12. Optional report-back to delivery (non-fatal).
    let _ = cfg
        .delivery
        .report(delivered.receipt, outcome.clone())
        .await;

    Ok(QuoteOutput {
        source_id: cfg.source.id.clone(),
        candidates: survivors,
        research_digest: digest,
        usage_summary: total_usage,
        outcome,
    })
}

// Helpers — pure functions ----------------------------------------------

fn parse_style_match_score(raw: &str) -> Option<f32> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    v.get("style_match_score")?.as_f64().map(|x| x as f32)
}

fn parse_quote_tool_output(content: &str) -> (String, String) {
    #[derive(serde::Deserialize)]
    struct Parsed {
        tweet_id: String,
        url: String,
    }
    serde_json::from_str::<Parsed>(content)
        .map(|p| (p.tweet_id, p.url))
        .unwrap_or_else(|_| (String::new(), "<unknown>".to_string()))
}

/// Drop near-duplicate quote drafts via the shared Levenshtein dedup
/// helper. Lower index wins on collision (declaration-order tiebreak).
fn dedup_quote_records(candidates: Vec<QuoteCandidateRecord>) -> Vec<QuoteCandidateRecord> {
    if candidates.len() <= 1 {
        return candidates;
    }
    let drafts: Vec<&str> = candidates.iter().map(|c| c.draft.as_str()).collect();
    let kept = crate::pipeline::dedup::distinct_indices(
        &drafts,
        crate::pipeline::dedup::LEVENSHTEIN_DUPLICATE_THRESHOLD,
    );
    kept.into_iter().map(|i| candidates[i].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::Utc;
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

    /// MockQuoteReviewDelivery — returns a pre-canned outcome. An optional
    /// error mode simulates delivery failure (used to assert delivery is
    /// never called in no_quote / gate-reject tests).
    struct MockQuoteReviewDelivery {
        outcome: Option<crate::review::DeliveryOutcome>,
        error_msg: Option<String>,
        reports: Mutex<Vec<QuoteOutcome>>,
    }

    impl MockQuoteReviewDelivery {
        fn arc(outcome: crate::review::DeliveryOutcome) -> Arc<MockQuoteReviewDelivery> {
            Arc::new(MockQuoteReviewDelivery {
                outcome: Some(outcome),
                error_msg: None,
                reports: Mutex::new(Vec::new()),
            })
        }

        fn errored(reason: &str) -> Arc<MockQuoteReviewDelivery> {
            Arc::new(MockQuoteReviewDelivery {
                outcome: None,
                error_msg: Some(reason.to_string()),
                reports: Mutex::new(Vec::new()),
            })
        }
    }

    impl QuoteReviewDelivery for MockQuoteReviewDelivery {
        fn deliver<'a>(
            &'a self,
            _msg: QuoteReviewMessage,
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
        > {
            let outcome = self.outcome.clone();
            let error_msg = self.error_msg.clone();
            Box::pin(async move {
                if let Some(msg) = error_msg {
                    return Err(crate::review::ReviewDeliveryError::Transport(msg));
                }
                Ok(crate::review::DeliveredReview {
                    outcome: outcome.expect("either outcome or error_msg must be set"),
                    receipt: crate::review::DeliveryReceipt {
                        data: serde_json::Value::Null,
                    },
                })
            })
        }

        fn report<'a>(
            &'a self,
            _receipt: crate::review::DeliveryReceipt,
            outcome: QuoteOutcome,
        ) -> Pin<Box<dyn Future<Output = Result<(), crate::review::ReviewDeliveryError>> + Send + 'a>>
        {
            self.reports.lock().unwrap().push(outcome);
            Box::pin(async move { Ok(()) })
        }
    }

    /// MockQuoteTool — captures `last_input` so tests can assert the body
    /// sent to `twitter_quote` includes `quote_tweet_id` (not `in_reply_to`).
    struct MockQuoteTool {
        canned: Mutex<Option<ToolOutput>>,
        last_input: Mutex<Option<serde_json::Value>>,
    }

    impl MockQuoteTool {
        fn success(body: &str) -> Arc<Self> {
            Arc::new(MockQuoteTool {
                canned: Mutex::new(Some(ToolOutput::success(body))),
                last_input: Mutex::new(None),
            })
        }

        fn errored(reason: &str) -> Arc<Self> {
            Arc::new(MockQuoteTool {
                canned: Mutex::new(Some(ToolOutput::error(reason))),
                last_input: Mutex::new(None),
            })
        }

        fn last_input(&self) -> Option<serde_json::Value> {
            self.last_input.lock().unwrap().clone()
        }
    }

    impl Tool for MockQuoteTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "twitter_quote".to_string(),
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
                canned.ok_or_else(|| CoreError::Agent("mock quote tool exhausted".into()))
            })
        }
    }

    /// MockProvider — same shape as reply/mod.rs::tests::MockProvider.
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

    /// Stub credential resolver — never invoked in mock tests.
    struct StubCredentialResolver;

    impl CredentialResolverTrait for StubCredentialResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(Secret::new("stub")) })
        }
    }

    /// Snapshot fixture — mirrors reply/mod.rs::tests::seed_snapshot.
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

    /// Fixture source tweet.
    fn fixture_source() -> QuoteCandidate {
        QuoteCandidate {
            id: "9001".into(),
            text: "Microservices solve every problem".into(),
            author_id: "42".into(),
            author_handle: "shipit".into(),
            posted_at: Utc::now(),
        }
    }

    /// Boilerplate builder for QuoteConfig (single candidate by default).
    #[allow(clippy::too_many_arguments)]
    fn mk_quote_cfg<'a>(
        profiles_root: &'a std::path::Path,
        provider: Arc<BoxedProvider>,
        delivery: Arc<dyn QuoteReviewDelivery>,
        twitter_tool: Arc<dyn Tool>,
        candidates_per_quote: usize,
        source: QuoteCandidate,
    ) -> QuoteConfig<'a> {
        QuoteConfig {
            persona_name: "x",
            provider,
            writer_provider: None,
            corpora_root: profiles_root,
            profiles_root,
            on_progress: None,
            source,
            candidates_per_quote,
            delivery,
            twitter_tool,
            credentials: Arc::new(StubCredentialResolver),
        }
    }

    // --- Test 1: happy path, user picks idx 0, twitter_quote posts ---

    #[tokio::test]
    async fn run_quote_pipeline_pick_index_0_posts_to_twitter() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "research digest about microservices tradeoffs",
            "concrete short quote comment",
            r#"{"verdict":"pass","style_match_score":0.92}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery_concrete =
            MockQuoteReviewDelivery::arc(crate::review::DeliveryOutcome::Pick(0));
        let delivery_trait: Arc<dyn QuoteReviewDelivery> = delivery_concrete.clone();
        let twitter_tool = MockQuoteTool::success(
            r#"{"tweet_id":"quote123","url":"https://x.com/i/web/status/quote123"}"#,
        );
        let cfg = mk_quote_cfg(
            &profiles_root,
            provider,
            delivery_trait,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_source(),
        );
        let out = run_quote_pipeline(cfg).await.expect("happy path");
        match out.outcome {
            QuoteOutcome::Posted {
                chosen_index,
                quote_tweet_id,
                quote_url,
            } => {
                assert_eq!(chosen_index, 0);
                assert_eq!(quote_tweet_id, "quote123");
                assert_eq!(quote_url, "https://x.com/i/web/status/quote123");
            }
            other => panic!("expected Posted, got {other:?}"),
        }
        assert_eq!(out.candidates.len(), 1);
        assert_eq!(out.source_id, "9001");

        // Verify the X tool received quote_tweet_id (NOT in_reply_to).
        let body = twitter_tool.last_input().expect("twitter tool was called");
        assert_eq!(
            body.get("quote_tweet_id").and_then(|v| v.as_str()),
            Some("9001"),
            "expected quote_tweet_id=9001 in tool input; got: {body}"
        );
        assert!(
            body.get("in_reply_to").is_none(),
            "in_reply_to must NOT be present in quote-tweet tool input; got: {body}"
        );
        assert!(
            body.get("text").and_then(|v| v.as_str()).is_some(),
            "text must be present in tool input; got: {body}"
        );

        // Verify delivery.report() was called with the Posted outcome.
        let reports = delivery_concrete.reports.lock().unwrap();
        assert_eq!(reports.len(), 1, "report() should be called exactly once");
        match &reports[0] {
            QuoteOutcome::Posted { chosen_index, .. } => assert_eq!(*chosen_index, 0),
            other => panic!("expected report() to receive Posted, got {other:?}"),
        }
    }

    // --- Test 2: user presses Skip, no post call ---

    #[tokio::test]
    async fn run_quote_pipeline_skip_returns_skipped_no_post() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "research digest",
            "short quote candidate",
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockQuoteReviewDelivery::arc(crate::review::DeliveryOutcome::Skip);
        let twitter_tool = MockQuoteTool::errored("twitter must not be called on skip");
        let cfg = mk_quote_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn QuoteReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_source(),
        );
        let out = run_quote_pipeline(cfg).await.expect("skip is success");
        assert!(
            matches!(out.outcome, QuoteOutcome::Skipped),
            "expected Skipped, got {:?}",
            out.outcome
        );
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be called on skip"
        );
    }

    // --- Test 3: review times out, no post call ---

    #[tokio::test]
    async fn run_quote_pipeline_timed_out_returns_timed_out_no_post() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "research digest",
            "short quote candidate",
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockQuoteReviewDelivery::arc(crate::review::DeliveryOutcome::TimedOut);
        let twitter_tool = MockQuoteTool::errored("twitter must not be called on timeout");
        let cfg = mk_quote_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn QuoteReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_source(),
        );
        let out = run_quote_pipeline(cfg).await.expect("timeout is success");
        assert!(
            matches!(out.outcome, QuoteOutcome::TimedOut),
            "expected TimedOut, got {:?}",
            out.outcome
        );
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be called on timeout"
        );
    }

    // --- Test 4: every candidate is no_quote — NoQuote outcome, delivery not called ---

    #[tokio::test]
    async fn run_quote_pipeline_all_no_quote_returns_no_quote_no_delivery() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "research digest", // researcher
            "no_quote",        // writer short-circuits; critic+fact not called
        ]);
        // Delivery should NEVER be called — set it to error if it is.
        let delivery = MockQuoteReviewDelivery::errored("delivery must not be called");
        let twitter_tool = MockQuoteTool::errored("twitter must not be called");
        let cfg = mk_quote_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn QuoteReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_source(),
        );
        let out = run_quote_pipeline(cfg).await.expect("no_quote is success");
        assert!(
            matches!(out.outcome, QuoteOutcome::NoQuote),
            "expected NoQuote, got {:?}",
            out.outcome
        );
        assert!(
            out.candidates.is_empty(),
            "no candidates when writer says no_quote"
        );
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be called"
        );
    }

    // --- Test 5: every candidate fails the publish gate (pre-filter sweep) ---

    #[tokio::test]
    async fn run_quote_pipeline_all_candidates_gate_rejected_skips_delivery() {
        let (_dir, profiles_root) = seed_snapshot("x");
        // A 281-char unbroken run-on (no `. ` / `! ` / `? ` boundaries)
        // survives `normalize_tweet_length` intact, then fails the publish
        // gate. The pre-filter sweep drops it before delivery.
        let too_long = "x".repeat(281);
        let provider = MockProvider::arc(vec![
            "research digest",
            too_long.as_str(),
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockQuoteReviewDelivery::errored("delivery must not be called");
        let twitter_tool = MockQuoteTool::errored("twitter must not be called");
        let cfg = mk_quote_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn QuoteReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_source(),
        );
        let out = run_quote_pipeline(cfg)
            .await
            .expect("all-gate-rejected is success");
        match out.outcome {
            QuoteOutcome::AllCandidatesGateRejected { reasons } => {
                assert!(!reasons.is_empty(), "expected drop reasons");
                assert!(
                    reasons[0].contains("publish_gate"),
                    "first reason should reference publish_gate; got: {reasons:?}"
                );
            }
            other => panic!("expected AllCandidatesGateRejected, got {other:?}"),
        }
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be called when all candidates dropped"
        );
    }

    // --- Test 6: every candidate is Unverifiable — AllCandidatesGateRejected, delivery not called ---

    #[tokio::test]
    async fn run_quote_pipeline_all_unverifiable_returns_all_candidates_gate_rejected() {
        let (_dir, profiles_root) = seed_snapshot("x");
        // Short draft so publish_gate passes — only the Unverifiable verdict
        // should trigger the pre-filter drop. The response sequence for a
        // single-candidate run is: researcher → writer → critic → fact_check.
        let provider = MockProvider::arc(vec![
            "research digest about microservices tradeoffs", // researcher
            "concrete short quote comment",                  // writer (short, passes publish_gate)
            r#"{"verdict":"pass","style_match_score":0.85}"#, // critic
            r#"{"verdict":"unverifiable","reason":"no data"}"#, // fact_check → Unverifiable
        ]);
        // Delivery should NEVER be called — set it to error if it is.
        let delivery = MockQuoteReviewDelivery::errored("delivery must not be called");
        let twitter_tool = MockQuoteTool::errored("twitter must not be called");
        let cfg = mk_quote_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn QuoteReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_source(),
        );
        let out = run_quote_pipeline(cfg)
            .await
            .expect("all-unverifiable is success");
        match out.outcome {
            QuoteOutcome::AllCandidatesGateRejected { reasons } => {
                assert!(!reasons.is_empty(), "expected at least one drop reason");
                assert!(
                    reasons[0].starts_with("unverifiable:"),
                    "first reason should start with 'unverifiable:'; got: {reasons:?}"
                );
            }
            other => panic!("expected AllCandidatesGateRejected, got {other:?}"),
        }
        assert!(
            twitter_tool.last_input().is_none(),
            "twitter_tool must not be called when all candidates dropped"
        );
    }

    // --- Test 7: user picks but Twitter API errors → PublishFailed ---

    #[tokio::test]
    async fn run_quote_pipeline_pick_twitter_api_error_returns_publish_failed() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "research digest",
            "short quote candidate",
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockQuoteReviewDelivery::arc(crate::review::DeliveryOutcome::Pick(0));
        let twitter_tool = MockQuoteTool::errored("rate limited (429)");
        let cfg = mk_quote_cfg(
            &profiles_root,
            provider,
            delivery as Arc<dyn QuoteReviewDelivery>,
            twitter_tool.clone() as Arc<dyn Tool>,
            1,
            fixture_source(),
        );
        let out = run_quote_pipeline(cfg)
            .await
            .expect("publish failure is success");
        match out.outcome {
            QuoteOutcome::PublishFailed {
                chosen_index,
                reason,
            } => {
                assert_eq!(chosen_index, 0);
                assert!(
                    reason.contains("rate limited") || reason.contains("429"),
                    "got: {reason}"
                );
            }
            other => panic!("expected PublishFailed, got {other:?}"),
        }
    }
}
