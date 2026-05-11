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

use heartbit_ghost::posts::{PostHistoryEntry, PostHistoryStore, PostOutcome};
use heartbit_ghost::review::{
    ReviewConfig, ReviewDelivery, ReviewError, ReviewOutcome, run_review_pipeline,
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

    // 5. Run the review pipeline.
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
    let entry = PostHistoryEntry {
        posted_at: chrono::Utc::now(),
        topic: topic.clone(),
        outcome: post_outcome.clone(),
        tweet_id,
        // Task 4 will populate this from the first tweet's text on the
        // Posted path; Task 2 keeps it None to focus on the API helper.
        text: None,
    };
    if let Err(e) = deps.history.record(deps.persona_name, entry).await {
        tracing::warn!(error = %e, "history.record (terminal) failed");
    }
    Ok(post_outcome)
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
}
