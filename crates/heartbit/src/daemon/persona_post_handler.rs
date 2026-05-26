//! PersonaPost dispatcher handler — runs the proactive-post pipeline
//! for one scheduler tick: pre-fetch context → topic generator →
//! duplicate-check → review pipeline → record outcome.
//!
//! See P1.6 spec §8 for the algorithm.

use std::path::Path;
use std::sync::Arc;

use chrono::Duration;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::{PersonaParams, PersonaRegistry};

use heartbit_ghost::posts::{PostHistoryEntry, PostHistoryStore, PostOutcome, TopPostsProvider};
use heartbit_ghost::review::{
    ReviewConfig, ReviewDelivery, ReviewError, ReviewOutcome, parse_thread_tweets,
    run_review_pipeline,
};

/// Dependencies for one `handle_persona_post` invocation.
///
/// Groups all parameters so the function stays within Clippy's
/// `too_many_arguments` threshold (mirrors `ReplyDraftDeps`).
pub struct PersonaPostDeps<'a> {
    /// Persona name to run (e.g. `"heartbit-ghost:x"`).
    pub persona_name: &'a str,
    /// Persona registry populated at daemon startup.
    pub registry: &'a PersonaRegistry,
    /// Post history store (for de-dup + recording outcome).
    pub history: &'a dyn PostHistoryStore,
    /// Lookback window for the duplicate check.
    pub history_lookback: Duration,
    /// Optional fallback brief from config (appended to topic generator input).
    pub topic_brief: Option<&'a str>,
    /// Operator's X user_id (passed to the topic context provider).
    pub operator_user_id: &'a str,
    /// LLM provider for sub-agents (topic generator + review pipeline).
    pub provider: Arc<BoxedProvider>,
    /// Telegram (or mock) review delivery.
    pub delivery: Arc<dyn ReviewDelivery>,
    /// `twitter_thread` tool — used by `run_review_pipeline` to post.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver shared with `twitter_tool` + topic context provider.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Number of candidate threads to draft.
    pub candidates_per_draft: usize,
    /// Root directory containing per-persona corpora.
    pub corpora_root: &'a Path,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: &'a Path,
    /// Optional ranker for top-engaged posts. When `Some` and `top_n > 0`
    /// and the ranker returns ≥3 exemplars, an `EXEMPLARS —` block is
    /// prepended to the writer's user_message. Cold start (<3 exemplars)
    /// silently no-injects so pre-P2.0 behavior is preserved.
    pub top_posts_provider: Option<&'a dyn TopPostsProvider>,
    /// How many exemplars to request from `top_posts_provider`. `0`
    /// disables injection. The min-3 threshold is applied inside the
    /// handler, not here.
    pub top_n: usize,
    /// Optional override provider for the writer + style-critic agents.
    /// `None` falls back to `provider`. Built by the CLI startup from
    /// `[daemon.persona_posts.writer_provider]` and threaded onto the
    /// matching `PersonaPostEntry`.
    pub writer_provider: Option<Arc<BoxedProvider>>,
    /// How to produce the optional head-tweet image. Threaded onto the
    /// review pipeline's `ReviewConfig.image_source`. `Online` (default)
    /// searches Openverse, `Ai` generates, `None` skips.
    pub image_source: heartbit_core::config::ImageSource,
}

/// Run one `PersonaPost` handler invocation.
///
/// Steps:
/// 1. Look up and expand the persona.
/// 2. Build topic-generator user message from recent history + `TopicContextProvider` output.
/// 3. Run the topic generator.
/// 4. Short-circuit on `"no_topic"` (records `PostOutcome::NoTopic` to store).
/// 5. Duplicate-check via `PostHistoryStore::was_posted_recently`.
/// 6. Run `run_review_pipeline` on the selected topic.
/// 7. Map `ReviewOutcome` → `PostOutcome` and record to store.
///
/// On any terminal outcome, records to history so the cron poll won't
/// retry without an explicit gap. Returns the outcome for caller
/// introspection.
pub async fn handle_persona_post(deps: PersonaPostDeps<'_>) -> Result<PostOutcome, anyhow::Error> {
    let persona = deps
        .registry
        .get(deps.persona_name)
        .ok_or_else(|| anyhow::anyhow!("persona '{}' not registered", deps.persona_name))?;
    let expansion = persona
        .expand(&PersonaParams::default())
        .map_err(|e| anyhow::anyhow!("expand persona '{}': {e}", deps.persona_name))?;

    // Researcher override (heartbit-rs:x uses repo_researcher).
    let researcher_override = expansion
        .agents
        .iter()
        .find(|a| a.name == "repo_researcher")
        .map(|recipe| {
            let recipe = std::sync::Arc::new(recipe.clone_config());
            let tools: Vec<std::sync::Arc<dyn Tool>> = expansion
                .tools
                .iter()
                .filter(|t| t.definition().name == "repo_inspect")
                .cloned()
                .collect();
            (recipe, tools)
        });

    // 1. Build the topic generator's input context.
    let recent_history = deps
        .history
        .recent(deps.persona_name, 5)
        .await
        .map_err(|e| anyhow::anyhow!("history.recent: {e}"))?;
    let recent_history_json =
        serde_json::to_string(&recent_history).unwrap_or_else(|_| "[]".into());
    let mut user_message = String::new();
    if let Some(provider) = expansion.topic_context_provider.as_ref() {
        let ctx_result = provider
            .build_context(
                deps.operator_user_id,
                &recent_history_json,
                deps.credentials.clone(),
            )
            .await;
        match ctx_result {
            Ok(s) => user_message.push_str(&s),
            Err(e) => {
                tracing::warn!(error = %e, "topic context provider failed; continuing with history only");
                user_message.push_str(&render_history_block(&recent_history));
            }
        }
    } else {
        user_message.push_str(&render_history_block(&recent_history));
    }
    if let Some(brief) = deps.topic_brief {
        user_message.push_str("\nTOPIC AREA (from config):\n");
        user_message.push_str(brief);
        user_message.push('\n');
    }
    user_message.push_str("\nPropose ONE topic per the OUTPUT spec, or 'no_topic'.\n");

    // 2. Run the topic generator.
    let topic_runner = heartbit_ghost::pipeline::runner_from_recipe(
        deps.provider.clone(),
        heartbit_ghost::agents::topic_generator_recipe(),
        Vec::new(),
    )
    .map_err(|e| anyhow::anyhow!("topic generator builder: {e}"))?;
    let gen_out = topic_runner
        .execute(&user_message)
        .await
        .map_err(|e| anyhow::anyhow!("topic generator exec: {e}"))?;
    let topic_raw = gen_out.result.trim();

    // 3. Handle the no_topic short-circuit.
    if topic_raw.eq_ignore_ascii_case("no_topic") || topic_raw.is_empty() {
        let entry = PostHistoryEntry {
            posted_at: chrono::Utc::now(),
            topic: String::new(),
            outcome: PostOutcome::NoTopic,
            tweet_id: None,
            text: None,
        };
        if let Err(e) = deps.history.record(deps.persona_name, entry).await {
            tracing::warn!(error = %e, "history.record (NoTopic) failed");
        }
        return Ok(PostOutcome::NoTopic);
    }
    let topic = topic_raw.to_string();

    // 4. Duplicate check.
    let is_dupe = deps
        .history
        .was_posted_recently(deps.persona_name, &topic, deps.history_lookback)
        .await
        .unwrap_or(false);
    if is_dupe {
        let entry = PostHistoryEntry {
            posted_at: chrono::Utc::now(),
            topic: topic.clone(),
            outcome: PostOutcome::SkippedDuplicate,
            tweet_id: None,
            text: None,
        };
        if let Err(e) = deps.history.record(deps.persona_name, entry).await {
            tracing::warn!(error = %e, "history.record (SkippedDuplicate) failed");
        }
        return Ok(PostOutcome::SkippedDuplicate);
    }

    // 5. Build the engagement few-shot block (P2.0). Cold start (<3
    // exemplars) silently no-injects so pre-P2.0 behavior is preserved.
    let exemplar_block =
        build_exemplar_block(deps.top_posts_provider, deps.top_n, chrono::Utc::now()).await;

    // 6. Run the review pipeline.
    let cfg = ReviewConfig {
        persona_name: deps.persona_name,
        topic: &topic,
        provider: deps.provider.clone(),
        corpora_root: deps.corpora_root,
        profiles_root: deps.profiles_root,
        on_progress: Some(std::sync::Arc::new(|s: &str| tracing::info!("post: {s}"))),
        candidates_per_draft: deps.candidates_per_draft,
        delivery: deps.delivery.clone(),
        twitter_tool: deps.twitter_tool.clone(),
        credentials: deps.credentials.clone(),
        mode_addendum: expansion.mode_addendum,
        researcher_override,
        exemplar_block: if exemplar_block.is_empty() {
            None
        } else {
            Some(exemplar_block.as_str())
        },
        writer_provider: deps.writer_provider.clone(),
        image_source: deps.image_source,
    };
    let review_out = run_review_pipeline(cfg)
        .await
        .map_err(|e: ReviewError| anyhow::anyhow!("review pipeline: {e}"))?;

    // 6. Map ReviewOutcome → PostOutcome and record.
    let post_outcome = map_review_outcome(&review_out.outcome);
    let tweet_id = match &review_out.outcome {
        ReviewOutcome::Posted { tweet_ids, .. } => tweet_ids.first().cloned(),
        _ => None,
    };
    // On the Posted path, capture the first tweet's text so future runs of
    // TopPostsProvider can render exemplars without round-tripping the X API.
    // Non-Posted outcomes (Skipped/TimedOut/GateRejected/PublishFailed) leave
    // `text` as None.
    let text = match &review_out.outcome {
        ReviewOutcome::Posted { chosen_index, .. } => review_out
            .candidates
            .get(*chosen_index)
            .and_then(|c| parse_thread_tweets(&c.draft).into_iter().next()),
        _ => None,
    };
    let entry = PostHistoryEntry {
        posted_at: chrono::Utc::now(),
        topic: topic.clone(),
        outcome: post_outcome.clone(),
        tweet_id,
        text,
    };
    if let Err(e) = deps.history.record(deps.persona_name, entry).await {
        tracing::warn!(error = %e, "history.record (terminal) failed");
    }
    Ok(post_outcome)
}

/// Build the writer-prompt EXEMPLARS block from the `TopPostsProvider`.
///
/// Returns an empty string when:
/// - no provider is configured;
/// - `top_n == 0` (operator disabled the feature);
/// - the provider call fails (warn-logged, fail-open);
/// - fewer than 3 exemplars are available (cold start — silent no-op).
///
/// The em dash in `EXEMPLARS \u{2014}` is intentional — it matches the
/// spec and is asserted byte-for-byte in the integration tests.
async fn build_exemplar_block(
    provider: Option<&dyn TopPostsProvider>,
    top_n: usize,
    now: chrono::DateTime<chrono::Utc>,
) -> String {
    let Some(provider) = provider else {
        return String::new();
    };
    if top_n == 0 {
        return String::new();
    }
    let top = match provider.top_n(top_n).await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(error = %e, "top_posts_provider failed; skipping exemplar injection");
            return String::new();
        }
    };
    if top.len() < 3 {
        // Cold-start: too few exemplars to be useful as few-shot signal.
        return String::new();
    }
    let mut s = String::from(
        "EXEMPLARS \u{2014} your highest-engaged posts from the last 30 days.\n\
         Study the voice, structure, and angle. Do NOT copy literally.\n\n",
    );
    for p in &top {
        let age = (now - p.posted_at).num_days();
        s.push_str(&format!(
            "[{} days ago, engagement score {:.0}]\n{}\n\n",
            age, p.engagement_score, p.text
        ));
    }
    s.push_str("---\n\n");
    s
}

fn map_review_outcome(o: &ReviewOutcome) -> PostOutcome {
    match o {
        ReviewOutcome::Posted {
            chosen_index,
            tweet_url,
            ..
        } => PostOutcome::Posted {
            chosen_index: *chosen_index,
            url: tweet_url.clone(),
        },
        ReviewOutcome::Skipped => PostOutcome::Skipped,
        ReviewOutcome::TimedOut => PostOutcome::TimedOut,
        ReviewOutcome::GateRejected {
            chosen_index,
            reason,
        } => PostOutcome::GateRejected {
            chosen_index: *chosen_index,
            reason: reason.clone(),
        },
        ReviewOutcome::PublishFailed {
            chosen_index,
            reason,
        } => PostOutcome::PublishFailed {
            chosen_index: *chosen_index,
            reason: reason.clone(),
        },
        ReviewOutcome::AllCandidatesGateRejected { reasons } => {
            PostOutcome::AllCandidatesGateRejected {
                reasons: reasons.clone(),
            }
        }
    }
}

fn render_history_block(history: &[PostHistoryEntry]) -> String {
    let mut out = String::new();
    out.push_str("RECENT POST HISTORY (last 5 from store):\n");
    if history.is_empty() {
        out.push_str("(none)\n");
    } else {
        for entry in history.iter().take(5) {
            let when = entry.posted_at.format("%Y-%m-%d");
            let outcome = match &entry.outcome {
                PostOutcome::Posted { .. } => "Posted",
                PostOutcome::Skipped => "Skipped",
                PostOutcome::TimedOut => "TimedOut",
                PostOutcome::NoTopic => "NoTopic",
                PostOutcome::SkippedDuplicate => "SkippedDuplicate",
                PostOutcome::GateRejected { .. } => "GateRejected",
                PostOutcome::PublishFailed { .. } => "PublishFailed",
                PostOutcome::AllCandidatesGateRejected { .. } => "AllCandidatesGateRejected",
            };
            let topic = if entry.topic.is_empty() {
                "(no topic)"
            } else {
                entry.topic.as_str()
            };
            out.push_str(&format!("- [{when}] {outcome}: {topic}\n"));
        }
    }
    out
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
    use std::sync::{Arc, Mutex};

    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::{CredentialResolver as CredentialResolverTrait, Secret};
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage, ToolDefinition,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use heartbit_core::persona::{Persona, PersonaExpansion, PersonaParams, PersonaRegistry};
    use heartbit_core::tool::ToolOutput;
    use heartbit_core::{ExecutionContext, Tool};
    use heartbit_ghost::posts::{InMemoryPostHistoryStore, PostHistoryEntry, PostOutcome};
    use heartbit_ghost::review::{
        DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReportableOutcome, ReviewDelivery,
        ReviewDeliveryError, ReviewMessage,
    };
    use tempfile::TempDir;

    // ─── StubTestPersona ─────────────────────────────────────────────────────

    struct StubTestPersona {
        name: String,
    }

    impl StubTestPersona {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
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
            // Minimal expansion: no agents, no tools, no topic_context_provider.
            Ok(PersonaExpansion::default())
        }
    }

    // ─── MockReviewDelivery ───────────────────────────────────────────────────

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

    // ─── MockTwitterTool ──────────────────────────────────────────────────────

    struct MockTwitterTool {
        canned: Mutex<Option<ToolOutput>>,
    }

    impl MockTwitterTool {
        fn success(thread_json: &str) -> Arc<dyn Tool> {
            Arc::new(MockTwitterTool {
                canned: Mutex::new(Some(ToolOutput::success(thread_json))),
            })
        }

        fn errored(reason: &str) -> Arc<dyn Tool> {
            Arc::new(MockTwitterTool {
                canned: Mutex::new(Some(ToolOutput::error(reason))),
            })
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
            _input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, CoreError>> + Send + '_>> {
            let canned = self.canned.lock().unwrap().take();
            Box::pin(async move {
                canned.ok_or_else(|| CoreError::Agent("mock twitter tool exhausted".into()))
            })
        }
    }

    // ─── MockProvider ─────────────────────────────────────────────────────────

    /// Mock LlmProvider that pops canned response strings off a VecDeque
    /// and (when constructed via [`MockProvider::arc_capturing`]) records
    /// every [`CompletionRequest`] it sees. The captured requests are
    /// what the engagement-injection integration tests assert against.
    struct MockProvider {
        responses: Mutex<VecDeque<String>>,
        /// `Some` when callers want to inspect what was sent to the LLM.
        /// Populated by `complete()` in arrival order.
        captured: Option<Arc<Mutex<Vec<CompletionRequest>>>>,
    }

    impl MockProvider {
        fn arc(responses: Vec<&str>) -> Arc<BoxedProvider> {
            let p = MockProvider {
                responses: Mutex::new(responses.into_iter().map(String::from).collect()),
                captured: None,
            };
            Arc::new(BoxedProvider::new(p))
        }

        /// Same as [`MockProvider::arc`] but returns a handle the caller
        /// can read after the run to inspect every request that was sent.
        fn arc_capturing(
            responses: Vec<&str>,
        ) -> (Arc<BoxedProvider>, Arc<Mutex<Vec<CompletionRequest>>>) {
            let captured: Arc<Mutex<Vec<CompletionRequest>>> = Arc::new(Mutex::new(Vec::new()));
            let p = MockProvider {
                responses: Mutex::new(responses.into_iter().map(String::from).collect()),
                captured: Some(captured.clone()),
            };
            (Arc::new(BoxedProvider::new(p)), captured)
        }
    }

    impl LlmProvider for MockProvider {
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CoreError>> + Send {
            if let Some(captured) = self.captured.as_ref() {
                captured.lock().unwrap().push(request.clone());
            }
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

    // ─── StubCredentialResolver ───────────────────────────────────────────────

    struct StubCredentialResolver;

    impl CredentialResolverTrait for StubCredentialResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(Secret::new("stub")) })
        }
    }

    // ─── seed_snapshot ────────────────────────────────────────────────────────

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

    // ─── Test 1: happy path ───────────────────────────────────────────────────

    #[tokio::test]
    async fn happy_path_runs_pipeline_and_records() {
        let persona_name = "test-persona";
        let (_dir, profiles_root) = seed_snapshot(persona_name);

        // Provider canned: topic_generator first, then review pipeline.
        // Single-candidate order: researcher → writer → critic → fact_check → image_generator.
        let provider = MockProvider::arc(vec![
            "calibrated abstention",                         // topic_generator
            "Research digest:\n- AI",                        // researcher
            "concrete short post",                           // writer iter 1
            r#"{"verdict":"pass","style_match_score":0.9}"#, // critic
            r#"{"verdict":"verified"}"#,                     // fact_check
            "no_image",                                      // image_generator
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );

        let history = InMemoryPostHistoryStore::new();
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);

        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona::new(persona_name)));

        let deps = PersonaPostDeps {
            persona_name,
            registry: &registry,
            history: &history,
            history_lookback: Duration::days(30),
            topic_brief: None,
            operator_user_id: "12345",
            provider,
            delivery,
            twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            top_posts_provider: None,
            top_n: 0,
            writer_provider: None,
            // Reaches the post arm; AI path consumes the canned image slot.
            image_source: heartbit_core::config::ImageSource::Ai,
        };

        let outcome = handle_persona_post(deps).await.expect("happy path");
        match outcome {
            PostOutcome::Posted { chosen_index, url } => {
                assert_eq!(chosen_index, 0);
                assert!(url.contains("123"), "url should contain tweet id 123");
            }
            other => panic!("expected Posted, got {other:?}"),
        }

        // Verify history records the Posted outcome.
        let recent = history.recent(persona_name, 5).await.unwrap();
        assert_eq!(recent.len(), 1);
        assert!(matches!(recent[0].outcome, PostOutcome::Posted { .. }));
        assert_eq!(recent[0].tweet_id.as_deref(), Some("123"));
    }

    // ─── Test 2: no_topic short-circuit ──────────────────────────────────────

    #[tokio::test]
    async fn no_topic_short_circuits() {
        let persona_name = "test-persona";
        let (_dir, profiles_root) = seed_snapshot(persona_name);

        // Provider: just one canned response — "no_topic".
        let provider = MockProvider::arc(vec!["no_topic"]);
        // Delivery + twitter must NEVER be called.
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::errored("should not be called");

        let history = InMemoryPostHistoryStore::new();
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona::new(persona_name)));

        let deps = PersonaPostDeps {
            persona_name,
            registry: &registry,
            history: &history,
            history_lookback: Duration::days(30),
            topic_brief: None,
            operator_user_id: "12345",
            provider,
            delivery,
            twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            top_posts_provider: None,
            top_n: 0,
            writer_provider: None,
            // no_topic short-circuit — never reaches the image stage.
            image_source: heartbit_core::config::ImageSource::None,
        };

        let outcome = handle_persona_post(deps).await.expect("no_topic");
        assert_eq!(outcome, PostOutcome::NoTopic);

        let recent = history.recent(persona_name, 5).await.unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].outcome, PostOutcome::NoTopic);
        assert!(recent[0].topic.is_empty());
    }

    // ─── Test 3: duplicate topic ──────────────────────────────────────────────

    #[tokio::test]
    async fn duplicate_topic_skips_pipeline() {
        let persona_name = "test-persona";
        let (_dir, profiles_root) = seed_snapshot(persona_name);
        let history = InMemoryPostHistoryStore::new();

        // Pre-seed a recent post with the same topic the generator will return.
        history
            .record(
                persona_name,
                PostHistoryEntry {
                    posted_at: chrono::Utc::now() - chrono::Duration::hours(12),
                    topic: "calibrated abstention".into(),
                    outcome: PostOutcome::Posted {
                        chosen_index: 0,
                        url: "https://x.com/i/web/status/100".into(),
                    },
                    tweet_id: Some("100".into()),
                    text: None,
                },
            )
            .await
            .unwrap();

        // Provider: just topic_generator — pipeline must not be called.
        let provider = MockProvider::arc(vec!["calibrated abstention"]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::errored("should not be called");

        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona::new(persona_name)));

        let deps = PersonaPostDeps {
            persona_name,
            registry: &registry,
            history: &history,
            history_lookback: Duration::hours(24),
            topic_brief: None,
            operator_user_id: "12345",
            provider,
            delivery,
            twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            top_posts_provider: None,
            top_n: 0,
            writer_provider: None,
            // Duplicate-topic skip — never reaches the image stage.
            image_source: heartbit_core::config::ImageSource::None,
        };

        let outcome = handle_persona_post(deps).await.expect("dup");
        assert_eq!(outcome, PostOutcome::SkippedDuplicate);

        let recent = history.recent(persona_name, 5).await.unwrap();
        assert_eq!(recent.len(), 2);
        // Most recent is the SkippedDuplicate (just recorded).
        assert_eq!(recent[0].outcome, PostOutcome::SkippedDuplicate);
    }

    // ─── Test 4: telegram skip ────────────────────────────────────────────────

    #[tokio::test]
    async fn telegram_skip_records_skipped() {
        let persona_name = "test-persona";
        let (_dir, profiles_root) = seed_snapshot(persona_name);

        let provider = MockProvider::arc(vec![
            "calibrated abstention",                         // topic_generator
            "Research digest:\n- AI",                        // researcher
            "concrete short post",                           // writer iter 1
            r#"{"verdict":"pass","style_match_score":0.9}"#, // critic
            r#"{"verdict":"verified"}"#,                     // fact_check
            "no_image",                                      // image_generator
        ]);
        // User presses Skip on Telegram.
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Skip);
        let twitter_tool = MockTwitterTool::errored("should not be called");

        let history = InMemoryPostHistoryStore::new();
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona::new(persona_name)));

        let deps = PersonaPostDeps {
            persona_name,
            registry: &registry,
            history: &history,
            history_lookback: Duration::days(30),
            topic_brief: None,
            operator_user_id: "12345",
            provider,
            delivery,
            twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            top_posts_provider: None,
            top_n: 0,
            writer_provider: None,
            // Telegram Skip — never reaches the image stage.
            image_source: heartbit_core::config::ImageSource::None,
        };

        let outcome = handle_persona_post(deps).await.expect("skip");
        assert_eq!(outcome, PostOutcome::Skipped);

        let recent = history.recent(persona_name, 5).await.unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].outcome, PostOutcome::Skipped);
    }

    // ─── Test 5: unknown persona returns Err ──────────────────────────────────

    #[tokio::test]
    async fn unknown_persona_returns_err() {
        let (_dir, profiles_root) = seed_snapshot("any");
        let provider = MockProvider::arc(vec![]); // never called
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::errored("should not be called");

        let history = InMemoryPostHistoryStore::new();
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);
        let registry = PersonaRegistry::new(); // empty

        let deps = PersonaPostDeps {
            persona_name: "missing-persona",
            registry: &registry,
            history: &history,
            history_lookback: Duration::days(30),
            topic_brief: None,
            operator_user_id: "12345",
            provider,
            delivery,
            twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            top_posts_provider: None,
            top_n: 0,
            writer_provider: None,
            // Unknown-persona error path — never reaches the image stage.
            image_source: heartbit_core::config::ImageSource::None,
        };

        let err = handle_persona_post(deps)
            .await
            .expect_err("expected error for unknown persona");
        assert!(
            err.to_string().contains("not registered"),
            "error should say 'not registered', got: {err}"
        );

        // Store should be empty — no recording on persona lookup failure.
        let recent = history.recent("missing-persona", 5).await.unwrap();
        assert!(recent.is_empty());
    }

    // ─── Test 6: Posted entry records first-tweet text (Task 4) ──────────────

    #[tokio::test]
    async fn happy_path_records_text_on_posted_entry() {
        let persona_name = "test-persona";
        let (_dir, profiles_root) = seed_snapshot(persona_name);

        // Provider canned: topic_generator first, then review pipeline.
        // The writer's draft is the source of the first-tweet text exemplar.
        let provider = MockProvider::arc(vec![
            "calibrated abstention",                         // topic_generator
            "Research digest:\n- AI",                        // researcher
            "concrete short post",                           // writer iter 1
            r#"{"verdict":"pass","style_match_score":0.9}"#, // critic
            r#"{"verdict":"verified"}"#,                     // fact_check
            "no_image",                                      // image_generator
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );

        let history = InMemoryPostHistoryStore::new();
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);

        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona::new(persona_name)));

        let deps = PersonaPostDeps {
            persona_name,
            registry: &registry,
            history: &history,
            history_lookback: Duration::days(30),
            topic_brief: None,
            operator_user_id: "12345",
            provider,
            delivery,
            twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            top_posts_provider: None,
            top_n: 0,
            writer_provider: None,
            // Reaches the post arm; AI path consumes the canned image slot.
            image_source: heartbit_core::config::ImageSource::Ai,
        };

        let outcome = handle_persona_post(deps).await.expect("happy path");
        assert!(matches!(outcome, PostOutcome::Posted { .. }));

        let recent = history.recent(persona_name, 5).await.unwrap();
        let posted = recent
            .iter()
            .find(|e| matches!(e.outcome, PostOutcome::Posted { .. }))
            .expect("one Posted entry");
        assert!(
            posted.text.is_some(),
            "Posted entry must carry the first-tweet text"
        );
        // Writer emitted a single-tweet draft "concrete short post", which
        // is the first (and only) tweet of the chosen thread.
        assert_eq!(posted.text.as_deref(), Some("concrete short post"));
    }

    // ─── Engagement injection (P2.0 Task 5) ──────────────────────────────────

    use heartbit_core::llm::types::Role;
    use heartbit_ghost::posts::{
        EngagementSnapshot, EngagementStore, InMemoryEngagementStore, JoinedTopPostsProvider,
    };

    /// Pull every user_message text the writer agent saw. The writer is
    /// identified by its message prefix: `Topic: ` on its own, or
    /// `EXEMPLARS \u{2014}` when the engagement block was prepended.
    /// All other pipeline stages (researcher/critic/fact_check) start
    /// with different prefixes.
    fn writer_user_messages(captured: &[CompletionRequest]) -> Vec<String> {
        captured
            .iter()
            .filter_map(|req| {
                // The user_message lives on the LAST user-role Message in
                // the conversation (the model's prior turns and tool
                // results may precede it).
                let last_user = req.messages.iter().rev().find(|m| m.role == Role::User)?;
                let text: String = last_user
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if text.starts_with("Topic: ") || text.starts_with("EXEMPLARS \u{2014}") {
                    Some(text)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Seed `n` Posted entries + matching engagement snapshots so the
    /// provider returns `n` exemplars. Returns the provider Arc.
    async fn seed_engagement(
        persona: &str,
        history: Arc<heartbit_ghost::posts::InMemoryPostHistoryStore>,
        engagement: Arc<InMemoryEngagementStore>,
        count: usize,
    ) -> Arc<dyn heartbit_ghost::posts::TopPostsProvider> {
        for i in 0..count {
            let id = format!("tw_{i}");
            history
                .record(
                    persona,
                    PostHistoryEntry {
                        posted_at: chrono::Utc::now() - chrono::Duration::hours(48 + i as i64),
                        topic: format!("topic_{i}"),
                        outcome: PostOutcome::Posted {
                            chosen_index: 0,
                            url: format!("https://x.com/i/web/status/{id}"),
                        },
                        tweet_id: Some(id.clone()),
                        text: Some(format!("exemplar_text_{i}")),
                    },
                )
                .await
                .unwrap();
            engagement
                .record(EngagementSnapshot {
                    tweet_id: id,
                    captured_at: chrono::Utc::now(),
                    // Distinct scores so ordering is deterministic.
                    likes: (10 + i * 10) as u64,
                    replies: 0,
                    retweets: 0,
                    quotes: 0,
                    bookmarks: 0,
                    impressions: None,
                })
                .await
                .unwrap();
        }
        Arc::new(JoinedTopPostsProvider::new(
            history,
            engagement,
            persona.to_string(),
        ))
    }

    #[tokio::test]
    async fn writer_receives_exemplar_block_when_top_posts_present() {
        let persona_name = "test-persona";
        let (_dir, profiles_root) = seed_snapshot(persona_name);

        // Capturing provider — 6 canned responses, same as happy path.
        let (provider, captured) = MockProvider::arc_capturing(vec![
            "calibrated abstention",                         // topic_generator
            "Research digest:\n- AI",                        // researcher
            "concrete short post",                           // writer iter 1
            r#"{"verdict":"pass","style_match_score":0.9}"#, // critic
            r#"{"verdict":"verified"}"#,                     // fact_check
            "no_image",                                      // image_generator
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );

        // Use Arc<InMemoryPostHistoryStore> so we can both seed it AND pass
        // it to the provider (which needs Arc<dyn PostHistoryStore>).
        let history = Arc::new(heartbit_ghost::posts::InMemoryPostHistoryStore::new());
        let engagement = Arc::new(InMemoryEngagementStore::new());
        let top_provider = seed_engagement(persona_name, history.clone(), engagement, 3).await;

        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona::new(persona_name)));

        let deps = PersonaPostDeps {
            persona_name,
            registry: &registry,
            history: history.as_ref(),
            history_lookback: Duration::days(30),
            topic_brief: None,
            operator_user_id: "12345",
            provider,
            delivery,
            twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            top_posts_provider: Some(top_provider.as_ref()),
            top_n: 5,
            writer_provider: None,
            // Reaches the post arm; AI path consumes the canned image slot.
            image_source: heartbit_core::config::ImageSource::Ai,
        };

        let outcome = handle_persona_post(deps).await.expect("happy path");
        assert!(matches!(outcome, PostOutcome::Posted { .. }));

        let requests = captured.lock().unwrap().clone();
        let writer_msgs = writer_user_messages(&requests);
        assert!(
            !writer_msgs.is_empty(),
            "writer must have been invoked at least once; got {} captured requests",
            requests.len()
        );
        let writer_msg = &writer_msgs[0];
        assert!(
            writer_msg.starts_with("EXEMPLARS \u{2014}"),
            "writer message must be prefixed with the em-dash EXEMPLARS block; \
             got first 200 chars: {:?}",
            &writer_msg[..writer_msg.len().min(200)]
        );
        // The exemplar text from seed_engagement must appear verbatim.
        assert!(
            writer_msg.contains("exemplar_text_"),
            "writer message must contain the exemplar text bodies"
        );
    }

    #[tokio::test]
    async fn writer_unaffected_when_fewer_than_three_exemplars() {
        let persona_name = "test-persona";
        let (_dir, profiles_root) = seed_snapshot(persona_name);

        let (provider, captured) = MockProvider::arc_capturing(vec![
            "calibrated abstention",
            "Research digest:\n- AI",
            "concrete short post",
            r#"{"verdict":"pass","style_match_score":0.9}"#,
            r#"{"verdict":"verified"}"#,
            "no_image",
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );

        let history = Arc::new(heartbit_ghost::posts::InMemoryPostHistoryStore::new());
        let engagement = Arc::new(InMemoryEngagementStore::new());
        // Only 1 exemplar — below the 3-min threshold → no injection.
        let top_provider = seed_engagement(persona_name, history.clone(), engagement, 1).await;

        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona::new(persona_name)));

        let deps = PersonaPostDeps {
            persona_name,
            registry: &registry,
            history: history.as_ref(),
            history_lookback: Duration::days(30),
            topic_brief: None,
            operator_user_id: "12345",
            provider,
            delivery,
            twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            top_posts_provider: Some(top_provider.as_ref()),
            top_n: 5,
            writer_provider: None,
            // Reaches the post arm; AI path consumes the canned image slot.
            image_source: heartbit_core::config::ImageSource::Ai,
        };

        let outcome = handle_persona_post(deps).await.expect("cold start");
        assert!(matches!(outcome, PostOutcome::Posted { .. }));

        let requests = captured.lock().unwrap().clone();
        // No request anywhere should contain "EXEMPLARS" — cold start
        // means no injection, full stop.
        for req in &requests {
            let text: String = req
                .messages
                .iter()
                .flat_map(|m| {
                    m.content.iter().filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                })
                .collect();
            assert!(
                !text.contains("EXEMPLARS"),
                "no EXEMPLARS expected on cold start, but found one. Message: {:?}",
                &text[..text.len().min(200)]
            );
        }
    }

    #[tokio::test]
    async fn writer_unaffected_when_top_n_is_zero() {
        // Even when the provider would return >=3, top_n = 0 disables
        // injection. This is the operator's "kill switch".
        let persona_name = "test-persona";
        let (_dir, profiles_root) = seed_snapshot(persona_name);

        let (provider, captured) = MockProvider::arc_capturing(vec![
            "calibrated abstention",
            "Research digest:\n- AI",
            "concrete short post",
            r#"{"verdict":"pass","style_match_score":0.9}"#,
            r#"{"verdict":"verified"}"#,
            "no_image",
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );

        let history = Arc::new(heartbit_ghost::posts::InMemoryPostHistoryStore::new());
        let engagement = Arc::new(InMemoryEngagementStore::new());
        // 5 exemplars available — yet top_n=0 must block injection.
        let top_provider = seed_engagement(persona_name, history.clone(), engagement, 5).await;

        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona::new(persona_name)));

        let deps = PersonaPostDeps {
            persona_name,
            registry: &registry,
            history: history.as_ref(),
            history_lookback: Duration::days(30),
            topic_brief: None,
            operator_user_id: "12345",
            provider,
            delivery,
            twitter_tool,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            top_posts_provider: Some(top_provider.as_ref()),
            top_n: 0,
            writer_provider: None,
            // Reaches the post arm; AI path consumes the canned image slot.
            image_source: heartbit_core::config::ImageSource::Ai,
        };

        let outcome = handle_persona_post(deps).await.expect("disabled");
        assert!(matches!(outcome, PostOutcome::Posted { .. }));

        let requests = captured.lock().unwrap().clone();
        for req in &requests {
            let text: String = req
                .messages
                .iter()
                .flat_map(|m| {
                    m.content.iter().filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                })
                .collect();
            assert!(
                !text.contains("EXEMPLARS"),
                "top_n=0 must block injection, but found one. Message: {:?}",
                &text[..text.len().min(200)]
            );
        }
    }
}
