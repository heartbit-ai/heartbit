//! # Heartbit
//!
//! Multi-agent enterprise runtime with LLM orchestration, MCP tools, and durable execution.
//!
//! Heartbit provides a complete framework for building LLM-powered agents in Rust:
//! an orchestrator that dispatches tasks to sub-agents, each running a ReAct loop
//! with parallel tool execution via `tokio::JoinSet`. Three execution paths cover
//! different deployment needs: standalone (zero infra), durable (Restate SDK), and
//! daemon (Kafka-backed with HTTP API).
//!
//! ## Feature Flags
//!
//! | Feature | What it enables |
//! |---------|-----------------|
//! | `core` (default) | Agent runner, orchestrator, LLM providers, tools, memory, config |
//! | `kafka` | Kafka consumer/producer |
//! | `daemon` | Daemon with HTTP API, cron scheduling, metrics |
//! | `restate` | Durable workflow execution via Restate SDK 0.8 |
//! | `postgres` | PostgreSQL-backed memory and task store (pgvector) |
//! | `a2a` | Agent-to-Agent protocol |
//! | `local-embedding` | Local ONNX embeddings via fastembed (no API keys) |
//! | `full` | All of the above (except `local-embedding`) |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use heartbit::{
//!     AnthropicProvider, BoxedProvider, RetryingProvider,
//!     AgentRunner,
//! };
//!
//! # async fn run() -> Result<(), heartbit::Error> {
//! let provider = Arc::new(BoxedProvider::new(
//!     RetryingProvider::with_defaults(
//!         AnthropicProvider::new("sk-...", "claude-sonnet-4-20250514")
//!     )
//! ));
//!
//! let mut agent = AgentRunner::builder(provider)
//!     .system_prompt("You are a helpful assistant.")
//!     .on_text(Arc::new(|text| print!("{text}")))
//!     .build()?;
//!
//! let output = agent.execute("What is Rust?").await?;
//! println!("Tokens: {} in / {} out",
//!     output.tokens_used.input_tokens,
//!     output.tokens_used.output_tokens);
//! # Ok(())
//! # }
//! ```
//!
//! ## Key Types
//!
//! - [`AgentRunner`] / [`AgentRunnerBuilder`] — single agent ReAct loop
//! - [`Orchestrator`] / [`OrchestratorBuilder`] — multi-agent dispatch
//! - [`AnthropicProvider`] / [`OpenRouterProvider`] — LLM backends
//! - [`Tool`] — trait for agent-callable tools
//! - [`Memory`] / [`InMemoryStore`] — persistent agent memory
//! - [`Guardrail`] — pre/post LLM and tool hooks
//! - [`EvalRunner`] / [`EvalCase`] — evaluation framework

extern crate self as heartbit;

// All core modules and flat re-exports flow through this glob. Local module
// declarations below (channel, memory, store, auth, daemon, workflow,
// sandbox) shadow the glob-imported names — that's the point: the umbrella's
// versions add platform-specific extensions on top of core.
pub use heartbit_core::*;

// --- Umbrella-side platform modules ---
// `auth` is unconditional: `auth::ct` is a foundational constant-time helper
// that all builds need access to. `auth::jwt` and `auth::vault` remain gated
// inside `auth/mod.rs`.
pub mod auth;
pub mod channel;
pub mod memory;
pub mod sandbox;
pub mod store;

#[cfg(feature = "daemon")]
pub mod daemon;
#[cfg(feature = "restate")]
pub mod workflow;

// --- Channel re-exports (always available — lightweight traits) ---
#[cfg(feature = "postgres")]
pub use channel::PostgresSessionStore;
pub use channel::bridge::{InteractionBridge, OutboundMessage};
pub use channel::session::{
    InMemorySessionStore, Session, SessionMessage, SessionRole, SessionStore,
    format_session_context,
};
pub use channel::types::WsFrame;
pub use channel::{ChannelBridge, ConsolidateSession, MediaAttachment, RunTask, RunTaskInput};

#[cfg(feature = "discord")]
pub use channel::discord::{
    DiscordBridge, DiscordConfig, chunk_message as discord_chunk_message, get_gateway_url,
    send_typing as discord_send_typing, strip_mention as discord_strip_mention,
};

#[cfg(feature = "slack")]
pub use channel::slack::{
    SlackBridge, SlackConfig, SlackEnvelope, SlackEvent, SocketModeAck,
    chunk_message as slack_chunk_message, get_socket_url, strip_mention as slack_strip_mention,
    validate_bot_token,
};

// --- Config re-exports (always available — just data structs) ---
pub use config::{
    ActionBudgetConfig, ActionBudgetRuleConfig, ActiveHoursConfig, AgentConfig,
    AgentProviderConfig, AuthConfig, BehavioralConfig, BehavioralRuleConfig, CascadeConfig,
    CascadeGateConfig, CascadeTierConfig, ContextStrategyConfig, DaemonAuditConfig, DaemonConfig,
    DaemonMcpServerConfig, DispatchMode, EmbeddingConfig, GuardrailsConfig, HeartbitConfig,
    HeartbitPulseConfig, InjectionConfig, InputConstraintConfig, KNOWN_BUILTINS, KafkaConfig,
    KnowledgeConfig, KnowledgeSourceConfig, LspConfig, McpResourceMode, McpServerEntry,
    MemoryConfig, MetricsConfig, OrchestratorConfig, PersonaMentionsConfig, PiiConfig,
    RetryProviderConfig, SalienceConfig, SandboxConfig, ScheduleEntry, SecretPatternConfig,
    SecretScanConfig, SensorConfig, SensorModality, SensorRoutingConfig, SensorSourceConfig,
    SessionPruneConfigToml, SpawnConfig, StoryCorrelationConfig, TokenBudgetConfig,
    TokenExchangeConfig, ToolPolicyConfig, ToolPolicyRuleConfig, TrustLevel, WorkspaceConfig,
    WsConfig, parse_reasoning_effort, parse_tool_profile, parse_workflow_type,
};

// --- Auth re-exports ---
pub use auth::TenantScope;
#[cfg(feature = "vault")]
pub use auth::vault::{CredentialResolver, CredentialVault};
#[cfg(feature = "daemon")]
pub use auth::{JwksClient, JwtValidator};

// --- Daemon re-exports (feature-gated) ---
#[cfg(all(feature = "daemon", feature = "postgres"))]
pub use daemon::PostgresTaskStore;
#[cfg(feature = "daemon")]
pub use daemon::openai_compat;
#[cfg(feature = "daemon")]
pub use daemon::{
    CommandProducer, CronScheduler, DaemonCommand, DaemonCore, DaemonHandle, DaemonMetrics,
    DaemonTask, EdgeConditionPattern, EdgeConditionSpec, EdgeTransform, FileTodoStore,
    HeartbitPulseScheduler, InMemoryTaskStore, KafkaCommandProducer, MentionContext,
    MentionPollDeps, MentionPollScheduler, OnTaskComplete, PersonaMentionEntry, ReplyDraftDeps,
    ReplySharedContext, RuntimeAdvancedConfig, RuntimeAgentConfig, RuntimeEvalRequest,
    RuntimeEvalResponse, RuntimeEvalSseEvent, RuntimeGuardrailConfig, RuntimeMcpServer,
    RuntimeMemoryConfig, RuntimeOrchestratorConfig, RuntimeProviderConfig, RuntimeProviderType,
    RuntimeRequest, RuntimeResponse, RuntimeScorerConfig, RuntimeSpawnConfig, RuntimeSseEvent,
    RuntimeSubAgentConfig, RuntimeTwitterCredentials, RuntimeWorkflowConfig, RuntimeWorkflowEdge,
    RuntimeWorkflowNode, TaskOutcome, TaskState, TaskStats, TaskStore, TodoEntry, TodoList,
    TodoManageTool, UsageGroupBy, UsageQuery, UsageRow, UserContext, format_notification,
    handle_mention_poll, handle_reply_draft,
};

// --- Eval re-exports ---
pub use eval::{
    CaseComparison, CostScorer, EvalCase, EvalComparison, EvalResult, EvalRunner, EvalScorer,
    EvalSummary, EventCollector, ExpectedToolCall, KeywordScorer, LatencyScorer, SafetyScorer,
    ScorerResult, SimilarityScorer, ToolCallCountScorer, TrajectoryScorer, build_eval_agent,
    clear_events,
};

// --- Memory re-exports (feature-gated platform impls only — core impls flow through glob) ---
#[cfg(feature = "local-embedding")]
pub use memory::LocalEmbeddingProvider;
#[cfg(feature = "postgres")]
pub use memory::postgres::PostgresMemoryStore;

// --- Template re-exports ---
pub use template::registry::{known_templates, resolve_template};
pub use template::skills::{SkillContent, known_skills, load_skill, load_skills};
pub use template::{AgentTemplate, PartialAgentConfig, TemplateMeta, resolve_agent_config};

// --- Macro re-exports (feature-gated) ---
#[cfg(feature = "macro")]
pub use heartbit_macro::heartbit_tool;

// --- Store re-exports (feature-gated platform impls only) ---
#[cfg(feature = "postgres")]
pub use store::PostgresStore;
#[cfg(feature = "postgres")]
pub use store::postgres::PostgresAuditTrail;

// --- Sandbox re-exports ---
// CorePathPolicy/Builder are always available (pure Rust, no Linux/sandbox gate).
pub use sandbox::{CorePathPolicy, CorePathPolicyBuilder};
// SandboxPolicy (Landlock kernel enforcement) is Linux + sandbox-feature only.
#[cfg(all(target_os = "linux", feature = "sandbox"))]
pub use sandbox::SandboxPolicy;
