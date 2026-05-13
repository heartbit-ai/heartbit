//! Handler for `DaemonCommand::PersonaQuote` — picks un-seen source
//! tweets from one of the configured `source_user_ids`, drafts a
//! quote-tweet via `run_quote_pipeline`, records the seen state.
//!
//! Mirrors [`crate::daemon::persona_post_handler::handle_persona_post`]
//! in shape (deps struct + free function), with the trigger being a
//! curated source tweet rather than a generated topic.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::{PersonaParams, PersonaRegistry};

use heartbit_ghost::quote::sources::{QuoteCandidate, QuoteSeenStore, QuoteSource};
use heartbit_ghost::quote::{QuoteConfig, QuoteOutcome, QuoteReviewDelivery, run_quote_pipeline};

/// Dependencies for one `handle_persona_quote` invocation. Grouped so
/// the function stays within Clippy's `too_many_arguments` threshold
/// (mirrors `PersonaPostDeps` + `ReplyDraftDeps`).
pub struct PersonaQuoteDeps<'a> {
    /// Persona name to run (e.g. `"heartbit-ghost:x"`).
    pub persona_name: &'a str,
    /// Persona registry populated at daemon startup.
    pub registry: &'a PersonaRegistry,
    /// Source-tweet fetcher (`XUserTimelineSource` in prod; mock in tests).
    pub source: &'a dyn QuoteSource,
    /// Already-quoted dedup store.
    pub seen_store: &'a dyn QuoteSeenStore,
    /// Curated source X user IDs (numeric strings) to poll.
    pub source_user_ids: &'a [String],
    /// Maximum age in hours of a source tweet for it to be quote-able.
    /// `0` disables the age filter.
    pub max_age_hours: i64,
    /// Maximum number of source tweets to draft+review per scheduler tick.
    pub max_candidates_per_tick: usize,
    /// LLM provider for sub-agents (researcher + fact_check; also
    /// writer + critic when `writer_provider` is None).
    pub provider: Arc<BoxedProvider>,
    /// Optional override LLM provider for `quote_writer` + style critic.
    pub writer_provider: Option<Arc<BoxedProvider>>,
    /// Telegram (or mock) quote-review delivery.
    pub delivery: Arc<dyn QuoteReviewDelivery>,
    /// `twitter_quote` tool — used by `run_quote_pipeline` to post.
    pub twitter_quote_tool: Arc<dyn Tool>,
    /// Credential resolver shared with `twitter_quote_tool`.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Number of candidate quote-tweets to draft per chosen source tweet.
    pub candidates_per_draft: usize,
    /// Root directory containing per-persona corpora.
    pub corpora_root: &'a Path,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: &'a Path,
}

/// Run one `PersonaQuote` handler invocation.
///
/// Steps:
/// 1. Expand the persona (validates it's registered).
/// 2. For each `source_user_id`, fetch recent tweets via `source.recent`.
/// 3. Filter out already-seen + over-age tweets.
/// 4. Take the first `max_candidates_per_tick` candidates (source order).
/// 5. Record `seen_store.record` BEFORE running the pipeline so a panic
///    or restart mid-run doesn't cause us to re-draft the same source.
/// 6. Run `run_quote_pipeline`. Stop on first `Posted`; on any other
///    outcome, continue with the next candidate this tick.
pub async fn handle_persona_quote(deps: PersonaQuoteDeps<'_>) -> Result<()> {
    let persona = deps
        .registry
        .get(deps.persona_name)
        .ok_or_else(|| anyhow::anyhow!("persona '{}' not registered", deps.persona_name))?;
    let _expansion = persona
        .expand(&PersonaParams::default())
        .map_err(|e| anyhow::anyhow!("expand persona '{}': {e}", deps.persona_name))?;

    let now = Utc::now();
    let max_age = ChronoDuration::hours(deps.max_age_hours);

    // 1+2: collect candidates across all sources, filter age + seen.
    let mut candidates: Vec<QuoteCandidate> = Vec::new();
    for user_id in deps.source_user_ids {
        let fetched = match deps.source.recent(user_id).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    user_id = %user_id,
                    error = %e,
                    "failed to fetch source timeline; skipping"
                );
                continue;
            }
        };
        for c in fetched {
            if deps.max_age_hours > 0 && (now - c.posted_at) > max_age {
                continue;
            }
            match deps.seen_store.was_seen(deps.persona_name, &c.id).await {
                Ok(true) => continue,
                Ok(false) => {}
                Err(e) => {
                    tracing::warn!(
                        tweet_id = %c.id,
                        error = %e,
                        "seen_store.was_seen failed; treating as un-seen"
                    );
                }
            }
            candidates.push(c);
        }
    }

    if candidates.is_empty() {
        tracing::info!(persona = %deps.persona_name, "no quote candidates available this tick");
        return Ok(());
    }

    // 3+4: pick first N, draft each.
    let to_process: Vec<QuoteCandidate> = candidates
        .into_iter()
        .take(deps.max_candidates_per_tick)
        .collect();
    for source in to_process {
        // Record BEFORE running so a panic/restart doesn't re-draft.
        if let Err(e) = deps.seen_store.record(deps.persona_name, &source.id).await {
            tracing::warn!(error = %e, "seen_store.record failed before pipeline");
        }
        let cfg = QuoteConfig {
            persona_name: deps.persona_name,
            provider: deps.provider.clone(),
            corpora_root: deps.corpora_root,
            profiles_root: deps.profiles_root,
            on_progress: Some(Arc::new(|s: &str| tracing::info!("quote: {s}"))),
            source: source.clone(),
            // Field name in QuoteConfig is `candidates_per_quote`; the
            // operator-facing config field is `candidates_per_draft`.
            candidates_per_quote: deps.candidates_per_draft,
            delivery: deps.delivery.clone(),
            twitter_tool: deps.twitter_quote_tool.clone(),
            credentials: deps.credentials.clone(),
            writer_provider: deps.writer_provider.clone(),
        };
        match run_quote_pipeline(cfg).await {
            Ok(out) => {
                tracing::info!(
                    persona = %deps.persona_name,
                    source_id = %source.id,
                    outcome = ?out.outcome,
                    "quote pipeline complete"
                );
                // Stop on first Posted; continue if any other outcome so
                // we try the next candidate this tick.
                if matches!(out.outcome, QuoteOutcome::Posted { .. }) {
                    break;
                }
            }
            Err(e) => {
                tracing::error!(
                    persona = %deps.persona_name,
                    source_id = %source.id,
                    error = %e,
                    "quote pipeline failed"
                );
            }
        }
    }
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;

    use heartbit_core::ExecutionContext;
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::{CredentialResolver as CredentialResolverTrait, Secret};
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage, ToolDefinition,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use heartbit_core::persona::{Persona, PersonaExpansion, PersonaParams, PersonaRegistry};
    use heartbit_core::tool::ToolOutput;

    use heartbit_ghost::quote::sources::{InMemoryQuoteSeenStore, QuoteCandidate, QuoteSource};
    use heartbit_ghost::quote::{QuoteOutcome, QuoteReviewDelivery, QuoteReviewMessage};
    use heartbit_ghost::review::{
        DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReviewDeliveryError,
    };

    use heartbit_ghost::tools::XApiError;

    use chrono::Utc;
    use tempfile::TempDir;

    // ─── StubTestPersona ─────────────────────────────────────────────────────

    struct StubTestPersona {
        name: String,
    }

    impl Persona for StubTestPersona {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "stub persona for tests"
        }
        fn version(&self) -> &str {
            "0.0.1"
        }
        fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, CoreError> {
            Ok(PersonaExpansion::default())
        }
    }

    // ─── MockQuoteSource ─────────────────────────────────────────────────────

    struct MockQuoteSource {
        canned: Mutex<Vec<QuoteCandidate>>,
    }

    impl MockQuoteSource {
        fn with(canned: Vec<QuoteCandidate>) -> Self {
            Self {
                canned: Mutex::new(canned),
            }
        }
    }

    impl QuoteSource for MockQuoteSource {
        fn recent<'a>(
            &'a self,
            _user_id: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<QuoteCandidate>, XApiError>> + Send + 'a>>
        {
            let cs = self.canned.lock().unwrap().clone();
            Box::pin(async move { Ok(cs) })
        }
    }

    // ─── MockQuoteReviewDelivery ─────────────────────────────────────────────

    struct MockQuoteReviewDelivery {
        outcome: DeliveryOutcome,
    }

    impl MockQuoteReviewDelivery {
        fn arc(outcome: DeliveryOutcome) -> Arc<dyn QuoteReviewDelivery> {
            Arc::new(Self { outcome })
        }
    }

    impl QuoteReviewDelivery for MockQuoteReviewDelivery {
        fn deliver<'a>(
            &'a self,
            _msg: QuoteReviewMessage,
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
            _outcome: QuoteOutcome,
        ) -> Pin<Box<dyn Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    // ─── MockQuoteTool ───────────────────────────────────────────────────────

    struct MockQuoteTool {
        canned: Mutex<Option<ToolOutput>>,
    }

    impl MockQuoteTool {
        fn success(body: &str) -> Arc<dyn Tool> {
            Arc::new(MockQuoteTool {
                canned: Mutex::new(Some(ToolOutput::success(body))),
            })
        }

        fn errored(reason: &str) -> Arc<dyn Tool> {
            Arc::new(MockQuoteTool {
                canned: Mutex::new(Some(ToolOutput::error(reason))),
            })
        }
    }

    impl Tool for MockQuoteTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "twitter_quote".into(),
                description: "mock".into(),
                input_schema: serde_json::json!({"type":"object"}),
            }
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            _input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, CoreError>> + Send + '_>> {
            let canned = self.canned.lock().unwrap().take();
            Box::pin(async move {
                canned.ok_or_else(|| CoreError::Agent("mock quote tool exhausted".into()))
            })
        }
    }

    // ─── MockProvider ────────────────────────────────────────────────────────

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
                        id: "respond_1".into(),
                        name: "__respond__".into(),
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

    // ─── StubCredentialResolver ──────────────────────────────────────────────

    struct StubCredentialResolver;

    impl CredentialResolverTrait for StubCredentialResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(Secret::new("stub")) })
        }
    }

    // ─── seed_snapshot ───────────────────────────────────────────────────────

    fn seed_snapshot(persona: &str) -> (TempDir, std::path::PathBuf) {
        use heartbit_ghost::voice::{
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
            voice_traits: vec!["specific".into()],
            ai_tells_to_avoid: vec!["delve".into()],
            thread_rhythm: ThreadRhythm::Linear,
            thread_max_length: 5,
            thread_opener_must_hook: false,
            topical_obsessions: vec!["AI".into()],
            topical_avoidances: vec!["politics".into()],
        };
        let recipe = BlendRecipe {
            version: 1,
            blend: vec![BlendEntry {
                writer: "k".into(),
                weight: 1.0,
            }],
            overrides: PartialStyleProfile::default(),
        };
        let store = SnapshotStore::open(dir.path(), persona).unwrap();
        store.save_new(profile, &recipe).unwrap();
        let root = dir.path().to_path_buf();
        (dir, root)
    }

    fn fixture_source(id: &str) -> QuoteCandidate {
        QuoteCandidate {
            id: id.into(),
            text: "Microservices solve every problem".into(),
            author_id: "42".into(),
            author_handle: "shipit".into(),
            posted_at: Utc::now(),
        }
    }

    // ─── Test: unknown persona returns Err ───────────────────────────────────

    #[tokio::test]
    async fn unknown_persona_returns_err() {
        let (_dir, profiles_root) = seed_snapshot("any");
        let registry = PersonaRegistry::new();
        let source = MockQuoteSource::with(vec![]);
        let seen = InMemoryQuoteSeenStore::new();
        let provider = MockProvider::arc(vec![]);
        let delivery = MockQuoteReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockQuoteTool::errored("should not be called");
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);

        let source_ids = vec!["123".to_string()];
        let deps = PersonaQuoteDeps {
            persona_name: "missing-persona",
            registry: &registry,
            source: &source,
            seen_store: &seen,
            source_user_ids: &source_ids,
            max_age_hours: 12,
            max_candidates_per_tick: 1,
            provider,
            writer_provider: None,
            delivery,
            twitter_quote_tool: twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
        };
        let err = handle_persona_quote(deps)
            .await
            .expect_err("expected error for unknown persona");
        assert!(err.to_string().contains("not registered"), "got: {err}");
    }

    // ─── Test: no candidates available is a no-op ────────────────────────────

    #[tokio::test]
    async fn empty_source_is_noop() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona { name: "x".into() }));
        let source = MockQuoteSource::with(vec![]);
        let seen = InMemoryQuoteSeenStore::new();
        let provider = MockProvider::arc(vec![]);
        let delivery = MockQuoteReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockQuoteTool::errored("should not be called");
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);

        let source_ids = vec!["123".to_string()];
        let deps = PersonaQuoteDeps {
            persona_name: "x",
            registry: &registry,
            source: &source,
            seen_store: &seen,
            source_user_ids: &source_ids,
            max_age_hours: 12,
            max_candidates_per_tick: 1,
            provider,
            writer_provider: None,
            delivery,
            twitter_quote_tool: twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
        };
        handle_persona_quote(deps)
            .await
            .expect("noop should not error");
    }

    // ─── Test: already-seen candidates are skipped ───────────────────────────

    #[tokio::test]
    async fn already_seen_candidates_are_skipped() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona { name: "x".into() }));
        let candidate = fixture_source("9001");
        let source = MockQuoteSource::with(vec![candidate.clone()]);
        let seen = InMemoryQuoteSeenStore::new();
        // Mark as already seen — pipeline must NOT be invoked.
        seen.record("x", "9001").await.expect("seed seen");

        let provider = MockProvider::arc(vec![]); // never called
        let delivery = MockQuoteReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockQuoteTool::errored("should not be called");
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);

        let source_ids = vec!["42".to_string()];
        let deps = PersonaQuoteDeps {
            persona_name: "x",
            registry: &registry,
            source: &source,
            seen_store: &seen,
            source_user_ids: &source_ids,
            max_age_hours: 0, // disable age filter
            max_candidates_per_tick: 1,
            provider,
            writer_provider: None,
            delivery,
            twitter_quote_tool: twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
        };
        handle_persona_quote(deps)
            .await
            .expect("should not error on all-seen");
    }

    // ─── Test: happy path records seen + reaches pipeline ────────────────────

    #[tokio::test]
    async fn happy_path_records_seen_before_pipeline() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona { name: "x".into() }));
        let candidate = fixture_source("777");
        let source = MockQuoteSource::with(vec![candidate.clone()]);
        let seen = InMemoryQuoteSeenStore::new();

        // 4 canned responses cover: researcher, writer, critic, fact_check.
        let provider = MockProvider::arc(vec![
            "research digest about microservices tradeoffs",
            "concrete short quote comment",
            r#"{"verdict":"pass","style_match_score":0.92}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockQuoteReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockQuoteTool::success(
            r#"{"tweet_id":"quote777","url":"https://x.com/i/web/status/quote777"}"#,
        );
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);

        let source_ids = vec!["42".to_string()];
        let deps = PersonaQuoteDeps {
            persona_name: "x",
            registry: &registry,
            source: &source,
            seen_store: &seen,
            source_user_ids: &source_ids,
            max_age_hours: 0,
            max_candidates_per_tick: 1,
            provider,
            writer_provider: None,
            delivery,
            twitter_quote_tool: twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
        };
        handle_persona_quote(deps).await.expect("happy path");

        // The seen_store must have been updated BEFORE the pipeline ran.
        let was_seen: bool = seen
            .was_seen("x", "777")
            .await
            .expect("was_seen should not error");
        assert!(was_seen, "seen_store must record the source id");
    }
}
