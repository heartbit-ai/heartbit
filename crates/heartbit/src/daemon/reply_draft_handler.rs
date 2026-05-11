//! Free-function handler for [`super::types::DaemonCommand::ReplyDraft`].
//!
//! Builds a [`heartbit_ghost::reply::ReplyConfig`] from the injected
//! dependencies and mention payload, calls
//! [`heartbit_ghost::reply::run_reply_pipeline`], then — on any non-error
//! outcome — marks the mention replied and records per-author rate so the
//! cron poller does not retry. Errors from the pipeline propagate to the
//! caller; store failures after a successful pipeline run are logged as
//! warnings rather than propagated (best-effort idempotency).
//!
//! The daemon dispatch arm in `core.rs` keeps a stub `tracing::warn` with a
//! `TODO(P1.5c task 11b)` comment; wiring the full lifecycle (instantiating
//! registries, providers, deliveries at startup) is deferred to a follow-up
//! integration step mirroring Task 10's lifecycle pattern.

use std::path::Path;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::PersonaRegistry;
use heartbit_ghost::reply::{
    DailyTokenBudget, Mention, MentionStore, MentionerContext, ReplyConfig, ReplyError,
    ReplyOutcome, ReplyReviewDelivery, ScamJudge, TweetSnapshot, run_reply_pipeline,
};

use crate::Error;

/// Dependencies for one `handle_reply_draft` invocation.
///
/// Groups parameters so the function stays within Clippy's `too_many_arguments`
/// threshold (mirrors `MentionPollDeps` from `mention_poll_handler`).
pub struct ReplyDraftDeps<'a> {
    /// Persona registry (used for `expand()` calls).
    pub registry: &'a PersonaRegistry,
    /// Mention store (for `mark_replied` + `record_reply_to_author`).
    pub store: &'a dyn MentionStore,
    /// LLM provider used for every sub-agent in the pipeline.
    pub provider: Arc<BoxedProvider>,
    /// Telegram (or mock) review delivery.
    pub delivery: Arc<dyn ReplyReviewDelivery>,
    /// `twitter_reply` tool instance.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver for the twitter tool.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Number of distinct candidate replies to generate (1..=3).
    pub candidates_per_reply: usize,
    /// Root directory containing per-persona corpora.
    pub corpora_root: &'a Path,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: &'a Path,
    /// Daily token-budget tracker — updated after every pipeline run.
    pub budget_tracker: Arc<dyn DailyTokenBudget>,
    /// Optional content-aware LLM scam judge. Runs after the
    /// Kafka-redelivery dedup and before the full reply pipeline. A
    /// non-OK verdict marks the mention replied and returns `Skipped`,
    /// short-circuiting expensive multi-agent generation for crypto
    /// scams / spam / ads that the structural guards can't catch.
    /// `None` skips this stage entirely (preserves V1 behavior).
    pub scam_judge: Option<Arc<ScamJudge>>,
}

/// Run one `ReplyDraft` handler invocation.
///
/// On success (including `Skipped` / `TimedOut` / `NoReply` / `GateRejected`),
/// marks the mention replied and records per-author rate so the cron poller
/// does not retry on the next tick. Pipeline errors propagate as `Err` without
/// touching the store — the mention will be retried on the next poll.
pub async fn handle_reply_draft<'a>(
    deps: ReplyDraftDeps<'a>,
    persona_name: &'a str,
    mention: Mention,
    parent: Option<TweetSnapshot>,
    mentioner_context: Option<MentionerContext>,
) -> Result<ReplyOutcome, Error> {
    // Kafka redelivery dedup: with `enable.auto.commit=true`, a crash in the
    // ~5s window after a Posted outcome but before the offset commits causes
    // the broker to redeliver the same `ReplyDraft`. Without this check, the
    // handler would re-run the full pipeline → second Telegram review → if
    // operator clicks approve again, duplicate X post. Filed in task #41.
    if deps.store.was_posted(&mention.id).await.unwrap_or(false) {
        tracing::info!(
            persona_name,
            mention_id = %mention.id,
            "reply skipped — mention already posted (Kafka redelivery dedup)"
        );
        return Ok(ReplyOutcome::Skipped);
    }

    // Content-aware scam/spam/ad filter. Runs ONE cheap LLM call before
    // the multi-agent pipeline. Fail-open: a judge error is treated as OK.
    // The structural guards catch handle patterns; this catches the
    // content patterns they miss (crypto pumps with clean-looking handles,
    // engagement farming with normal account age, etc.).
    if let Some(ref judge) = deps.scam_judge {
        let verdict = judge.evaluate(&mention.text, &mention.author_handle).await;
        if !verdict.is_ok() {
            tracing::info!(
                persona_name,
                mention_id = %mention.id,
                author_handle = %mention.author_handle,
                verdict = verdict.label(),
                reason = verdict.reason().unwrap_or(""),
                "reply skipped by scam judge"
            );
            if let Err(e) = deps.store.mark_replied(&mention.id).await {
                tracing::warn!(
                    mention_id = %mention.id,
                    error = %e,
                    "mark_replied failed after scam-judge skip (best-effort)"
                );
            }
            return Ok(ReplyOutcome::Skipped);
        }
    }

    let persona = deps
        .registry
        .get(persona_name)
        .ok_or_else(|| Error::Daemon(format!("persona '{persona_name}' not registered")))?;

    let expansion = persona
        .expand(&heartbit_core::persona::PersonaParams::default())
        .map_err(|e| Error::Daemon(format!("expand persona '{persona_name}': {e}")))?;

    // If the expansion supplies a `repo_researcher` agent, thread it through
    // as the researcher override so the pipeline uses the persona-specific recipe.
    let researcher_override = expansion
        .agents
        .iter()
        .find(|a| a.name == "repo_researcher")
        .map(|recipe| {
            let recipe = Arc::new(recipe.clone_config());
            let tools: Vec<Arc<dyn Tool>> = expansion
                .tools
                .iter()
                .filter(|t| t.definition().name == "repo_inspect")
                .cloned()
                .collect();
            (recipe, tools)
        });

    // Clone ids before mention is moved into cfg.
    let mention_id = mention.id.clone();
    let author_id = mention.author_id.clone();
    let conversation_id_opt = mention.conversation_id.clone();
    let now = chrono::Utc::now();

    let cfg = ReplyConfig {
        persona_name,
        provider: deps.provider,
        corpora_root: deps.corpora_root,
        profiles_root: deps.profiles_root,
        on_progress: Some(Arc::new(|s: &str| tracing::info!("reply: {s}"))),
        mention,
        parent,
        mentioner_context,
        candidates_per_reply: deps.candidates_per_reply,
        mode_addendum: expansion.mode_addendum,
        researcher_override,
        delivery: deps.delivery,
        twitter_tool: deps.twitter_tool,
        credentials: deps.credentials,
    };

    let output = run_reply_pipeline(cfg)
        .await
        .map_err(|e: ReplyError| Error::Daemon(format!("reply pipeline: {e}")))?;

    // Always mark the mention replied — even on Skipped / TimedOut / NoReply /
    // GateRejected — so the cron poller does not re-dispatch on the next tick.
    // Record the per-author rate (conservative: Skipped counts).
    if let Err(e) = deps.store.mark_replied(&mention_id).await {
        tracing::warn!(
            mention_id,
            error = %e,
            "handle_reply_draft: mark_replied failed (best-effort)"
        );
    }
    if let Err(e) = deps.store.record_reply_to_author(&author_id, now).await {
        tracing::warn!(
            author_id,
            error = %e,
            "handle_reply_draft: record_reply_to_author failed (best-effort)"
        );
    }

    // On a Posted outcome, mark the mention so a Kafka redelivery short-circuits
    // (paired with the early `was_posted` check above), and update
    // conversation-depth accounting so the ConversationDepthGuard can cap future
    // replies in the same thread.
    if matches!(output.outcome, ReplyOutcome::Posted { .. }) {
        if let Err(e) = deps.store.mark_posted(&mention_id).await {
            tracing::warn!(
                mention_id,
                error = %e,
                "mark_posted failed (Kafka redelivery dedup will not protect this mention)"
            );
        }
        if let Some(ref conv_id) = conversation_id_opt
            && let Err(e) = deps.store.record_reply_in_conversation(conv_id).await
        {
            tracing::warn!(
                conversation_id = %conv_id,
                error = %e,
                "record_reply_in_conversation failed (best-effort)"
            );
        }
    }

    // Record token usage against the daily budget regardless of outcome
    // (tokens were consumed by the LLM even for Skipped/GateRejected runs).
    let total_tokens = output.usage_summary.input_tokens as u64
        + output.usage_summary.output_tokens as u64
        + output.usage_summary.reasoning_tokens as u64;
    if total_tokens > 0
        && let Err(e) = deps
            .budget_tracker
            .record_usage(persona_name, total_tokens)
            .await
    {
        tracing::warn!(
            persona_name,
            error = %e,
            "budget record_usage failed (best-effort)"
        );
    }

    Ok(output.outcome)
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

    use chrono::Utc;
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::{CredentialResolver as CredentialResolverTrait, Secret};
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage, ToolDefinition,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use heartbit_core::persona::{Persona, PersonaExpansion, PersonaParams};
    use heartbit_core::tool::ToolOutput;
    use heartbit_core::{ExecutionContext, Tool};
    use heartbit_ghost::reply::{
        InMemoryDailyBudget, InMemoryMentionStore, MentionStore, ReplyOutcome, ReplyReviewDelivery,
        ReplyReviewMessage,
    };
    use heartbit_ghost::review::{DeliveredReview, DeliveryOutcome, DeliveryReceipt};
    use tempfile::TempDir;

    // ─── Stub Persona ────────────────────────────────────────────────────────

    struct StubPersona {
        key: String,
    }

    impl StubPersona {
        fn ok(key: &str) -> Arc<Self> {
            Arc::new(StubPersona {
                key: key.to_string(),
            })
        }
    }

    impl Persona for StubPersona {
        fn name(&self) -> &str {
            &self.key
        }
        fn description(&self) -> &str {
            "stub"
        }
        fn version(&self) -> &str {
            "0.0.1"
        }
        fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, CoreError> {
            Ok(PersonaExpansion::default())
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

        fn failing() -> Arc<BoxedProvider> {
            let p = MockProvider {
                responses: Mutex::new(VecDeque::new()),
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

    // ─── MockReplyReviewDelivery ─────────────────────────────────────────────

    struct MockReplyReviewDelivery {
        outcome: Option<DeliveryOutcome>,
        error_msg: Option<String>,
    }

    impl MockReplyReviewDelivery {
        fn arc(outcome: DeliveryOutcome) -> Arc<Self> {
            Arc::new(MockReplyReviewDelivery {
                outcome: Some(outcome),
                error_msg: None,
            })
        }
    }

    impl ReplyReviewDelivery for MockReplyReviewDelivery {
        fn deliver<'a>(
            &'a self,
            _msg: ReplyReviewMessage,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            DeliveredReview,
                            heartbit_ghost::review::ReviewDeliveryError,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            let outcome = self.outcome.clone();
            let error_msg = self.error_msg.clone();
            Box::pin(async move {
                if let Some(msg) = error_msg {
                    return Err(heartbit_ghost::review::ReviewDeliveryError::Transport(msg));
                }
                Ok(DeliveredReview {
                    outcome: outcome.expect("either outcome or error_msg must be set"),
                    receipt: DeliveryReceipt {
                        data: serde_json::Value::Null,
                    },
                })
            })
        }

        fn report<'a>(
            &'a self,
            _receipt: DeliveryReceipt,
            _outcome: ReplyOutcome,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<(), heartbit_ghost::review::ReviewDeliveryError>>
                    + Send
                    + 'a,
            >,
        > {
            Box::pin(async move { Ok(()) })
        }
    }

    // ─── MockTwitterTool ─────────────────────────────────────────────────────

    struct MockTwitterTool {
        canned: Mutex<Option<ToolOutput>>,
    }

    impl MockTwitterTool {
        fn success(body: &str) -> Arc<Self> {
            Arc::new(MockTwitterTool {
                canned: Mutex::new(Some(ToolOutput::success(body))),
            })
        }
    }

    impl Tool for MockTwitterTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "twitter_reply".to_string(),
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

    // ─── seed_snapshot (mirrors reply/mod.rs::tests::seed_snapshot) ─────────

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

    // ─── fixture mention ─────────────────────────────────────────────────────

    fn fixture_mention() -> Mention {
        Mention {
            id: "mention_42".into(),
            text: "how does heartbit compare to LangChain?".into(),
            author_id: "author_999".into(),
            author_handle: "curious_dev".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: None,
            conversation_id: None,
        }
    }

    // ─── build_deps helper ───────────────────────────────────────────────────

    fn build_deps<'a>(
        registry: &'a heartbit_core::persona::PersonaRegistry,
        store: &'a dyn MentionStore,
        provider: Arc<BoxedProvider>,
        delivery: Arc<dyn ReplyReviewDelivery>,
        twitter_tool: Arc<dyn Tool>,
        profiles_root: &'a Path,
    ) -> ReplyDraftDeps<'a> {
        ReplyDraftDeps {
            registry,
            store,
            provider,
            delivery,
            twitter_tool,
            credentials: Arc::new(StubCredentialResolver),
            candidates_per_reply: 1,
            corpora_root: profiles_root,
            profiles_root,
            budget_tracker: Arc::new(InMemoryDailyBudget::new()),
            // Most tests don't exercise the judge — leave it off.
            scam_judge: None,
        }
    }

    // ─── Test 1: happy path ──────────────────────────────────────────────────

    #[tokio::test]
    async fn happy_path_runs_pipeline_marks_replied_and_records_author() {
        let persona_key = "stub:x";
        let (_dir, profiles_root) = seed_snapshot(persona_key);
        let store = InMemoryMentionStore::new();
        let provider = MockProvider::arc(vec![
            "research digest",
            "short concrete reply",
            r#"{"verdict":"pass","style_match_score":0.90}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockReplyReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"tweet_id":"reply1","url":"https://x.com/i/web/status/reply1"}"#,
        );
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        registry.register(StubPersona::ok(persona_key));

        let mention = fixture_mention();
        let mention_id = mention.id.clone();
        let author_id = mention.author_id.clone();

        let deps = build_deps(
            &registry,
            &store,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool as Arc<dyn Tool>,
            &profiles_root,
        );

        let outcome = handle_reply_draft(deps, persona_key, mention, None, None)
            .await
            .expect("happy path should succeed");

        match outcome {
            ReplyOutcome::Posted { chosen_index, .. } => {
                assert_eq!(chosen_index, 0);
            }
            other => panic!("expected Posted, got {other:?}"),
        }

        // Mention should be marked as replied.
        assert!(
            store.was_replied(&mention_id).await.unwrap(),
            "mention must be marked replied after successful pipeline"
        );

        // Per-author record should be present.
        let count = store
            .replies_to_author_since(&author_id, Utc::now() - chrono::Duration::minutes(1))
            .await
            .unwrap();
        assert_eq!(count, 1, "should have recorded one reply to author");
    }

    // ─── Test 2: skipped outcome still marks replied ─────────────────────────

    #[tokio::test]
    async fn skipped_outcome_still_marks_replied() {
        let persona_key = "stub:x";
        let (_dir, profiles_root) = seed_snapshot(persona_key);
        let store = InMemoryMentionStore::new();
        let provider = MockProvider::arc(vec![
            "research digest",
            "short reply candidate",
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        // User presses Skip — delivery returns DeliveryOutcome::Skip.
        let delivery = MockReplyReviewDelivery::arc(DeliveryOutcome::Skip);
        let twitter_tool = MockTwitterTool::success(r#"{"tweet_id":"unused","url":"unused"}"#);
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        registry.register(StubPersona::ok(persona_key));

        let mention = fixture_mention();
        let mention_id = mention.id.clone();
        let author_id = mention.author_id.clone();

        let deps = build_deps(
            &registry,
            &store,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool as Arc<dyn Tool>,
            &profiles_root,
        );

        let outcome = handle_reply_draft(deps, persona_key, mention, None, None)
            .await
            .expect("skip is a valid outcome, not an error");

        assert!(
            matches!(outcome, ReplyOutcome::Skipped),
            "expected Skipped, got {outcome:?}"
        );

        // Even on Skip: mark replied (so cron doesn't retry).
        assert!(
            store.was_replied(&mention_id).await.unwrap(),
            "skipped mention must be marked replied"
        );

        // Even on Skip: record per-author rate (conservative).
        let count = store
            .replies_to_author_since(&author_id, Utc::now() - chrono::Duration::minutes(1))
            .await
            .unwrap();
        assert_eq!(
            count, 1,
            "skipped reply should still count for per-author rate"
        );
    }

    // ─── Test 3: unknown persona returns Err, store unchanged ────────────────

    #[tokio::test]
    async fn unknown_persona_returns_err() {
        let (_dir, profiles_root) = seed_snapshot("stub:x");
        let store = InMemoryMentionStore::new();
        let provider = MockProvider::failing();
        let delivery = MockReplyReviewDelivery::arc(DeliveryOutcome::Skip);
        let twitter_tool = MockTwitterTool::success(r#"{"tweet_id":"unused","url":"unused"}"#);
        // Registry is empty — "stub:x" not registered.
        let registry = heartbit_core::persona::PersonaRegistry::new();

        let mention = fixture_mention();
        let mention_id = mention.id.clone();

        let deps = build_deps(
            &registry,
            &store,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool as Arc<dyn Tool>,
            &profiles_root,
        );

        let result = handle_reply_draft(deps, "stub:x", mention, None, None).await;
        assert!(result.is_err(), "unknown persona should return Err");
        assert!(
            result.unwrap_err().to_string().contains("not registered"),
            "error should say 'not registered'"
        );

        // Store must be untouched.
        assert!(
            !store.was_replied(&mention_id).await.unwrap(),
            "mention must not be marked replied when persona lookup fails"
        );
    }

    // ─── Test 4: pipeline failure propagates, store unchanged ────────────────

    #[tokio::test]
    async fn pipeline_failure_propagates_err() {
        let persona_key = "stub:x";
        let (_dir, profiles_root) = seed_snapshot(persona_key);
        let store = InMemoryMentionStore::new();
        // Empty provider — first LLM call will fail ("mock exhausted").
        let provider = MockProvider::failing();
        let delivery = MockReplyReviewDelivery::arc(DeliveryOutcome::Skip);
        let twitter_tool = MockTwitterTool::success(r#"{"tweet_id":"unused","url":"unused"}"#);
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        registry.register(StubPersona::ok(persona_key));

        let mention = fixture_mention();
        let mention_id = mention.id.clone();

        let deps = build_deps(
            &registry,
            &store,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool as Arc<dyn Tool>,
            &profiles_root,
        );

        let result = handle_reply_draft(deps, persona_key, mention, None, None).await;
        assert!(result.is_err(), "pipeline failure should propagate as Err");

        // The function must fail BEFORE mark_replied — store must be untouched.
        assert!(
            !store.was_replied(&mention_id).await.unwrap(),
            "mention must NOT be marked replied when the pipeline fails"
        );
    }

    // ─── Test 5: posted-mention redelivery is skipped without pipeline ──────
    //
    // Kafka uses `enable.auto.commit=true` (~5s interval). A daemon crash
    // after a Posted outcome but before the offset commits causes the
    // broker to redeliver the same `ReplyDraft`. The handler must check
    // `was_posted` at the start and short-circuit; otherwise the pipeline
    // re-runs → second Telegram review → potential duplicate X post.
    #[tokio::test]
    async fn already_posted_mention_skips_pipeline_on_redelivery() {
        let persona_key = "stub:x";
        let (_dir, profiles_root) = seed_snapshot(persona_key);
        let store = InMemoryMentionStore::new();

        let mention = fixture_mention();
        let mention_id = mention.id.clone();

        // Simulate a prior successful run: the mention is already marked posted.
        store.mark_posted(&mention_id).await.unwrap();

        // Failing provider — if the handler runs ANY part of the pipeline,
        // it'll exhaust the mock and the test will fail with a non-Ok result.
        let provider = MockProvider::failing();
        let delivery = MockReplyReviewDelivery::arc(DeliveryOutcome::Skip);
        let twitter_tool = MockTwitterTool::success(r#"{"tweet_id":"unused","url":"unused"}"#);
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        registry.register(StubPersona::ok(persona_key));

        let deps = build_deps(
            &registry,
            &store,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool as Arc<dyn Tool>,
            &profiles_root,
        );

        let outcome = handle_reply_draft(deps, persona_key, mention, None, None)
            .await
            .expect("redelivery must short-circuit successfully, not error");

        assert!(
            matches!(outcome, ReplyOutcome::Skipped),
            "redelivered already-posted mention should return Skipped, got {outcome:?}"
        );
    }

    // ─── Test 6: happy path now also marks the mention as posted ────────────
    //
    // Sibling to Test 1 (happy_path_…). Test 1 already exists and verifies
    // `mark_replied` + `record_reply_to_author`; this one verifies the
    // additional `mark_posted` call introduced for redelivery dedup.
    #[tokio::test]
    async fn posted_outcome_sets_was_posted_flag() {
        let persona_key = "stub:x";
        let (_dir, profiles_root) = seed_snapshot(persona_key);
        let store = InMemoryMentionStore::new();
        let provider = MockProvider::arc(vec![
            "research digest",
            "short concrete reply",
            r#"{"verdict":"pass","style_match_score":0.90}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockReplyReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"tweet_id":"reply1","url":"https://x.com/i/web/status/reply1"}"#,
        );
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        registry.register(StubPersona::ok(persona_key));

        let mention = fixture_mention();
        let mention_id = mention.id.clone();

        let deps = build_deps(
            &registry,
            &store,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool as Arc<dyn Tool>,
            &profiles_root,
        );

        let outcome = handle_reply_draft(deps, persona_key, mention, None, None)
            .await
            .expect("happy path should succeed");

        assert!(matches!(outcome, ReplyOutcome::Posted { .. }));
        assert!(
            store.was_posted(&mention_id).await.unwrap(),
            "Posted outcome must set was_posted for redelivery dedup"
        );
    }

    // ─── Test 8: scam judge short-circuits before the pipeline ────────────
    //
    // The scam judge runs after the Kafka-redelivery dedup and before the
    // multi-agent pipeline. A non-OK verdict marks the mention replied
    // and returns Skipped without burning any pipeline tokens.
    //
    // This is the operator's gap from the smoke-test review: structural
    // guards let through crypto-pump mentions with clean handles; the
    // judge catches them on content alone.
    #[tokio::test]
    async fn scam_judge_skips_classified_scam_without_running_pipeline() {
        let persona_key = "stub:x";
        let (_dir, profiles_root) = seed_snapshot(persona_key);
        let store = InMemoryMentionStore::new();
        // Provider for the JUDGE — returns one verdict, then is exhausted.
        let judge_provider = MockProvider::arc(vec!["VERDICT: SCAM: crypto pump bait"]);
        // Provider for the PIPELINE — exhausted from the start. If the
        // pipeline runs at all, it'll fail with "mock exhausted" and the
        // handler will return Err. Test asserts handler returns Ok(Skipped),
        // proving the pipeline did NOT run.
        let pipeline_provider = MockProvider::failing();
        let delivery = MockReplyReviewDelivery::arc(DeliveryOutcome::Skip);
        let twitter_tool = MockTwitterTool::success(r#"{"tweet_id":"unused","url":"unused"}"#);
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        registry.register(StubPersona::ok(persona_key));

        let mention = fixture_mention();
        let mention_id = mention.id.clone();

        let mut deps = build_deps(
            &registry,
            &store,
            pipeline_provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool as Arc<dyn Tool>,
            &profiles_root,
        );
        deps.scam_judge = Some(Arc::new(heartbit_ghost::reply::ScamJudge::new(
            judge_provider,
        )));

        let outcome = handle_reply_draft(deps, persona_key, mention, None, None)
            .await
            .expect("scam-judge short-circuit must return Ok(Skipped), not Err");

        assert!(
            matches!(outcome, ReplyOutcome::Skipped),
            "non-OK verdict must return Skipped, got {outcome:?}"
        );

        // Mention is marked replied so the next poll doesn't retry.
        assert!(store.was_replied(&mention_id).await.unwrap());
        // was_posted is NOT set — nothing reached the X API.
        assert!(!store.was_posted(&mention_id).await.unwrap());
    }

    // ─── Test 9: scam judge OK verdict lets the pipeline proceed ──────────
    #[tokio::test]
    async fn scam_judge_ok_verdict_runs_pipeline_normally() {
        let persona_key = "stub:x";
        let (_dir, profiles_root) = seed_snapshot(persona_key);
        let store = InMemoryMentionStore::new();
        let judge_provider = MockProvider::arc(vec!["VERDICT: OK"]);
        // Pipeline provider: full happy-path canned responses.
        let pipeline_provider = MockProvider::arc(vec![
            "research digest",
            "short concrete reply",
            r#"{"verdict":"pass","style_match_score":0.90}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockReplyReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool =
            MockTwitterTool::success(r#"{"tweet_id":"r1","url":"https://x.com/i/web/status/r1"}"#);
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        registry.register(StubPersona::ok(persona_key));

        let mention = fixture_mention();
        let mention_id = mention.id.clone();

        let mut deps = build_deps(
            &registry,
            &store,
            pipeline_provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool as Arc<dyn Tool>,
            &profiles_root,
        );
        deps.scam_judge = Some(Arc::new(heartbit_ghost::reply::ScamJudge::new(
            judge_provider,
        )));

        let outcome = handle_reply_draft(deps, persona_key, mention, None, None)
            .await
            .expect("OK verdict + happy pipeline should produce a Posted outcome");

        assert!(
            matches!(outcome, ReplyOutcome::Posted { .. }),
            "OK verdict must NOT block the pipeline, got {outcome:?}"
        );
        assert!(store.was_posted(&mention_id).await.unwrap());
    }

    // ─── Test 7: Skipped outcome does NOT set was_posted ───────────────────
    #[tokio::test]
    async fn skipped_outcome_does_not_set_was_posted() {
        let persona_key = "stub:x";
        let (_dir, profiles_root) = seed_snapshot(persona_key);
        let store = InMemoryMentionStore::new();
        let provider = MockProvider::arc(vec![
            "research digest",
            "short reply candidate",
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockReplyReviewDelivery::arc(DeliveryOutcome::Skip);
        let twitter_tool = MockTwitterTool::success(r#"{"tweet_id":"unused","url":"unused"}"#);
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        registry.register(StubPersona::ok(persona_key));

        let mention = fixture_mention();
        let mention_id = mention.id.clone();

        let deps = build_deps(
            &registry,
            &store,
            provider,
            delivery as Arc<dyn ReplyReviewDelivery>,
            twitter_tool as Arc<dyn Tool>,
            &profiles_root,
        );

        let outcome = handle_reply_draft(deps, persona_key, mention, None, None)
            .await
            .expect("skip is a valid outcome");
        assert!(matches!(outcome, ReplyOutcome::Skipped));

        // was_replied is set (existing behavior — prevents re-poll-dispatch).
        assert!(store.was_replied(&mention_id).await.unwrap());
        // was_posted is NOT set — nothing was actually posted to X.
        assert!(
            !store.was_posted(&mention_id).await.unwrap(),
            "Skipped outcome must not mark posted — only real X posts do"
        );
    }
}
