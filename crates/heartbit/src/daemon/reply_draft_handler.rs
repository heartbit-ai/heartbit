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
    Mention, MentionStore, MentionerContext, ReplyConfig, ReplyError, ReplyOutcome,
    ReplyReviewDelivery, TweetSnapshot, run_reply_pipeline,
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
        InMemoryMentionStore, MentionStore, ReplyOutcome, ReplyReviewDelivery, ReplyReviewMessage,
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
}
