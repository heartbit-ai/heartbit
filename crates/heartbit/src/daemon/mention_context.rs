//! Resolved per-persona mention context injected into [`super::core::DaemonCore`].
//!
//! The CLI constructs [`MentionContext`] from `[[daemon.persona_mentions]]` TOML
//! entries before handing control to `DaemonCore::run()`. `DaemonCore` uses it to:
//!
//! 1. Spawn one [`super::mention_poll::MentionPollScheduler`] per entry.
//! 2. Dispatch `DaemonCommand::MentionPoll` to [`super::mention_poll_handler::handle_mention_poll`].
//! 3. Dispatch `DaemonCommand::ReplyDraft` to [`super::reply_draft_handler::handle_reply_draft`].

use std::path::PathBuf;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::ExecutionContext;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::PersonaRegistry;
use heartbit_ghost::reply::{
    BotHeuristicConfig, DailyTokenBudget, EnrichmentCache, InMemoryDailyBudget, MentionStore,
    ReplyReviewDelivery, ScamJudge, SpamGuard, SpamGuardConfig,
};
use heartbit_ghost::tools::client::XClient;

/// All dependencies resolved for a single `[[daemon.persona_mentions]]` entry.
///
/// Constructed by the CLI from a [`crate::config::PersonaMentionsConfig`] block
/// and injected into [`MentionContext`].
pub struct PersonaMentionEntry {
    /// Persona slug (e.g. `"heartbit-ghost:x"`).
    pub persona: String,
    /// X/Twitter user-id of the operator account.
    pub user_id: String,
    /// Poll interval in seconds (forwarded to `MentionPollScheduler`).
    pub poll_interval_seconds: u64,
    /// Number of distinct candidate replies to generate (1..=3).
    pub candidates_per_reply: usize,
    /// Mention store backend for `since_id` tracking, rate-limit bookkeeping, and
    /// spam-guard lookups. Created from `PersonaMentionsConfig.mention_store`.
    pub store: Arc<dyn MentionStore>,
    /// Spam rules for early-exit of self-replies and low-quality mentions.
    pub spam_guard: SpamGuard,
    /// Execution context carrying credentials for the `twitter_mentions` tool.
    pub exec_ctx: ExecutionContext,
    /// Maximum mentions to fetch per poll cycle (passed to the tool as `max_results`).
    pub max_results: u32,

    // ── P1.7 loop-protection guards ─────────────────────────────────────────
    /// When `true`, the thread-depth guard skips thread continuations.
    pub enable_thread_depth_guard: bool,
    /// Bot-heuristic guard config. `None` disables the guard.
    pub bot_heuristic: Option<BotHeuristicConfig>,
    /// Maximum replies per unique conversation (0 = unlimited).
    pub per_conversation_max_replies: usize,
    /// Shared daily token-budget tracker (in-memory or JSONL).
    pub budget_tracker: Arc<dyn DailyTokenBudget>,
    /// Budget cap in tokens per UTC day. `None` = unlimited.
    pub daily_token_budget: Option<u64>,
}

impl PersonaMentionEntry {
    /// Construct from the raw config fields with an already-resolved store.
    ///
    /// `credential_resolver` is wrapped in an [`ExecutionContext`] for the tool call path.
    pub fn new(
        persona: impl Into<String>,
        user_id: impl Into<String>,
        poll_interval_seconds: u64,
        candidates_per_reply: usize,
        store: Arc<dyn MentionStore>,
        credential_resolver: Arc<dyn CredentialResolver>,
        max_results: u32,
    ) -> Self {
        let user_id = user_id.into();
        let spam_guard = SpamGuard::new(SpamGuardConfig::defaults_for(user_id.clone()));
        let exec_ctx = ExecutionContext {
            credentials: Some(credential_resolver),
            ..ExecutionContext::default()
        };
        Self {
            persona: persona.into(),
            user_id,
            poll_interval_seconds,
            candidates_per_reply,
            store,
            spam_guard,
            exec_ctx,
            max_results,
            enable_thread_depth_guard: true,
            bot_heuristic: None,
            per_conversation_max_replies: 0,
            budget_tracker: Arc::new(InMemoryDailyBudget::new()),
            daily_token_budget: None,
        }
    }
}

/// Shared dependencies for `ReplyDraft` handling — created once per daemon boot.
pub struct ReplySharedContext {
    /// Persona registry populated by `heartbit_ghost::register`.
    /// Stored as `Arc` so it can be shared across spawned tasks without cloning.
    pub registry: Arc<PersonaRegistry>,
    /// LLM provider for the reply pipeline.
    pub provider: Arc<BoxedProvider>,
    /// Telegram (or mock) review delivery.
    pub delivery: Arc<dyn ReplyReviewDelivery>,
    /// `twitter_reply` tool instance.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver used by the twitter tool.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Root directory containing per-persona corpora.
    pub corpora_root: PathBuf,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: PathBuf,
    /// Optional content-aware scam/spam/ad classifier. When present, the
    /// reply handler runs this BEFORE the multi-agent pipeline; a non-OK
    /// verdict short-circuits to `ReplyOutcome::Skipped`. Single shared
    /// instance reused across all persona entries.
    pub scam_judge: Option<Arc<ScamJudge>>,
}

/// Bundle of all mention-polling context, passed to
/// [`super::core::DaemonCore::with_mention_context`].
pub struct MentionContext {
    /// One entry per enabled `[[daemon.persona_mentions]]` block.
    pub entries: Vec<PersonaMentionEntry>,
    /// Shared reply context (registry, provider, delivery, tools).
    pub reply: ReplySharedContext,
    /// `twitter_mentions` tool shared across all entries.
    pub mentions_tool: Arc<dyn Tool>,
    /// Shared X API client used by the poll handler to enrich each
    /// surviving mention with `author_handle` + `MentionerContext`
    /// (activates the bot-heuristic guard) and the parent tweet text
    /// (gives the reply writer real thread context). `None` preserves
    /// V1 behavior (no enrichment, bot guard inert).
    pub enricher: Option<Arc<XClient>>,
    /// Shared in-memory cache for enrichment calls. Dedups repeated
    /// `GET /2/users/:id` lookups (same author mentioning multiple times)
    /// and `GET /2/tweets/:id` parent fetches (replies in the same
    /// thread). `None` makes every enrichment call hit the API.
    pub enrichment_cache: Option<Arc<EnrichmentCache>>,
}

impl MentionContext {
    pub fn new(
        entries: Vec<PersonaMentionEntry>,
        reply: ReplySharedContext,
        mentions_tool: Arc<dyn Tool>,
    ) -> Self {
        Self {
            entries,
            reply,
            mentions_tool,
            enricher: None,
            enrichment_cache: None,
        }
    }

    /// Attach a shared X API client used for per-mention enrichment.
    pub fn with_enricher(mut self, enricher: Arc<XClient>) -> Self {
        self.enricher = Some(enricher);
        self
    }

    /// Attach a shared in-memory cache for enrichment dedup.
    pub fn with_enrichment_cache(mut self, cache: Arc<EnrichmentCache>) -> Self {
        self.enrichment_cache = Some(cache);
        self
    }
}

#[cfg(test)]
mod tests {
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::{CredentialResolver as CredResolverTrait, Secret};
    use heartbit_core::llm::types::{CompletionRequest, CompletionResponse, ToolDefinition};
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use heartbit_core::tool::ToolOutput;
    use heartbit_core::{ExecutionContext, Tool};
    use heartbit_ghost::reply::{
        InMemoryMentionStore, MentionStore, ReplyOutcome, ReplyReviewDelivery, ReplyReviewMessage,
    };
    use heartbit_ghost::review::{
        DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReviewDeliveryError,
    };

    use super::*;

    struct NopCreds;
    impl CredResolverTrait for NopCreds {
        fn resolve(
            &self,
            _: &str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Secret, CoreError>> + Send + '_>,
        > {
            Box::pin(async { Err(CoreError::Daemon("nop".into())) })
        }
    }

    struct NopTool;
    impl Tool for NopTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "nop".into(),
                description: "nop".into(),
                input_schema: serde_json::json!({}),
            }
        }
        fn execute(
            &self,
            _: &ExecutionContext,
            _: serde_json::Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ToolOutput, CoreError>> + Send + '_>,
        > {
            Box::pin(async { Ok(ToolOutput::success(String::new())) })
        }
    }

    struct NopDelivery;
    impl ReplyReviewDelivery for NopDelivery {
        fn deliver<'a>(
            &'a self,
            _: ReplyReviewMessage,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<DeliveredReview, ReviewDeliveryError>>
                    + Send
                    + 'a,
            >,
        > {
            Box::pin(async {
                Ok(DeliveredReview {
                    outcome: DeliveryOutcome::Pick(0),
                    receipt: DeliveryReceipt {
                        data: serde_json::Value::Null,
                    },
                })
            })
        }
        fn report<'a>(
            &'a self,
            _: DeliveryReceipt,
            _: ReplyOutcome,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>,
        > {
            Box::pin(async { Ok(()) })
        }
    }

    struct NopProvider;
    impl LlmProvider for NopProvider {
        async fn complete(&self, _: CompletionRequest) -> Result<CompletionResponse, CoreError> {
            Err(CoreError::Daemon("nop".into()))
        }
    }

    /// `PersonaMentionEntry::new` sets every field correctly.
    #[test]
    fn persona_mention_entry_new_sets_fields() {
        let store: Arc<dyn MentionStore> = Arc::new(InMemoryMentionStore::new());
        let creds: Arc<dyn CredentialResolver> = Arc::new(NopCreds);

        let entry =
            PersonaMentionEntry::new("heartbit-ghost:x", "user_42", 300, 2, store, creds, 50);

        assert_eq!(entry.persona, "heartbit-ghost:x");
        assert_eq!(entry.user_id, "user_42");
        assert_eq!(entry.poll_interval_seconds, 300);
        assert_eq!(entry.candidates_per_reply, 2);
        assert_eq!(entry.max_results, 50);
        // spam_guard operator_user_id mirrors the user_id passed in.
        assert_eq!(entry.spam_guard.config().operator_user_id, "user_42");
    }

    /// `MentionContext::new` stores entries and tools correctly.
    #[test]
    fn mention_context_new_stores_parts() {
        let mentions_tool: Arc<dyn Tool> = Arc::new(NopTool);
        let reply = ReplySharedContext {
            registry: Arc::new(PersonaRegistry::new()),
            provider: Arc::new(BoxedProvider::new(NopProvider)),
            delivery: Arc::new(NopDelivery),
            twitter_tool: Arc::new(NopTool),
            credentials: Arc::new(NopCreds),
            corpora_root: std::path::PathBuf::from("/tmp"),
            profiles_root: std::path::PathBuf::from("/tmp"),
            scam_judge: None,
        };

        let mc = MentionContext::new(vec![], reply, mentions_tool);
        assert!(mc.entries.is_empty());
    }
}
