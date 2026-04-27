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
//! | `sensor` | 7 sensor sources, triage pipeline, story correlation |
//! | `restate` | Durable workflow execution via Restate SDK 0.8 |
//! | `postgres` | PostgreSQL-backed memory and task store (pgvector) |
//! | `a2a` | Agent-to-Agent protocol |
//! | `telegram` | Telegram bot adapter |
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

// --- Core modules (always available) ---
pub mod agent;
pub mod channel;
pub mod config;
pub mod error;
pub mod eval;
pub mod knowledge;
pub mod llm;
pub mod memory;
pub mod signal;
pub mod store;
pub mod template;
pub mod tool;
pub(crate) mod util;
pub mod workspace;

#[cfg(all(target_os = "linux", feature = "sandbox"))]
pub mod sandbox;

pub mod lsp;

// --- Feature-gated modules ---
// `auth` is unconditional: `auth::ct` is a foundational constant-time helper
// that all builds need access to. `auth::jwt` and `auth::vault` remain gated
// inside `auth/mod.rs`.
pub mod auth;
#[cfg(feature = "daemon")]
pub mod daemon;
#[cfg(feature = "sensor")]
pub mod sensor;
#[cfg(feature = "restate")]
pub mod workflow;

// --- Channel re-exports (always available — lightweight traits) ---
pub use channel::bridge::{InteractionBridge, OutboundMessage};
#[cfg(feature = "postgres")]
pub use channel::session::PostgresSessionStore;
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

#[cfg(feature = "telegram")]
pub use channel::telegram::{
    AccessControl as TelegramAccessControl, CallbackAction, ChatSessionMap, DmPolicy,
    RateLimiter as TelegramRateLimiter, StreamBuffer, TelegramAdapter, TelegramBridge,
    TelegramConfig, approval_buttons, chunk_message, markdown_to_telegram_html,
    parse_callback_data, question_buttons,
};

// --- Agent re-exports ---
pub use agent::audit::{AuditMode, AuditRecord, AuditTrail, InMemoryAuditTrail};
pub use agent::batch::{BatchConfig, BatchExecutor, BatchExecutorBuilder, BatchResult};
pub use agent::blackboard::{Blackboard, InMemoryBlackboard};
pub use agent::cache::ResponseCache;
pub use agent::context::ContextStrategy;
pub use agent::dag::{DagAgent, DagAgentBuilder};
pub use agent::debate::{DebateAgent, DebateAgentBuilder};
pub use agent::events::{AgentEvent, OnEvent};
pub use agent::guardrail::{GuardAction, Guardrail};
#[cfg(feature = "sensor")]
pub use agent::guardrails::SensorSecurityGuardrail;
pub use agent::guardrails::tool_policy::{InputConstraint, ToolRule};
pub use agent::guardrails::{
    ActionBudgetGuardrail, ActionBudgetGuardrailBuilder, BehaviorRule, BehavioralMonitorGuardrail,
    BehavioralMonitorGuardrailBuilder, BudgetRule, ConditionalGuardrail, ContentFenceGuardrail,
    GuardrailChain, GuardrailMode, InjectionClassifierGuardrail, LlmJudgeGuardrail,
    LlmJudgeGuardrailBuilder, PiiAction, PiiDetector, PiiGuardrail, SecretAction,
    SecretScannerGuardrail, SecretScannerGuardrailBuilder, ToolPolicyGuardrail, WarnToDeny,
};
pub use agent::instructions::{
    discover_instruction_files, load_instructions, prepend_instructions,
};
pub use agent::mixture::{MixtureOfAgentsAgent, MixtureOfAgentsAgentBuilder};
pub use agent::observability::ObservabilityMode;
pub use agent::orchestrator::{Orchestrator, OrchestratorBuilder, SubAgentConfig};
pub use agent::permission::{
    LearnedPermissions, PermissionAction, PermissionRule, PermissionRuleset,
};
pub use agent::prompts::MULTI_AGENT_COLLAB_PROMPT;
pub use agent::pruner::SessionPruneConfig;
pub use agent::routing::{
    AgentCapability, ComplexitySignals, KeywordRoutingStrategy, RoutingDecision, RoutingMode,
    RoutingStrategy, TaskComplexityAnalyzer, resolve_routing_mode, should_escalate,
};
pub use agent::tool_filter::ToolProfile;
pub use agent::voting::{VoteResult, VotingAgent, VotingAgentBuilder};
pub use agent::workflow::{
    LoopAgent, LoopAgentBuilder, ParallelAgent, ParallelAgentBuilder, SequentialAgent,
    SequentialAgentBuilder, WorkflowRouter, WorkflowType,
};
pub use agent::{AgentOutput, AgentRunner, AgentRunnerBuilder, OnInput};

// --- Config re-exports (always available — just data structs) ---
pub use config::{
    ActionBudgetConfig, ActionBudgetRuleConfig, ActiveHoursConfig, AgentConfig,
    AgentProviderConfig, AuthConfig, BehavioralConfig, BehavioralRuleConfig, CascadeConfig,
    CascadeGateConfig, CascadeTierConfig, ContextStrategyConfig, DaemonConfig,
    DaemonMcpServerConfig, DispatchMode, EmbeddingConfig, GuardrailsConfig, HeartbitConfig,
    HeartbitPulseConfig, InjectionConfig, InputConstraintConfig, KNOWN_BUILTINS, KafkaConfig,
    KnowledgeConfig, KnowledgeSourceConfig, LspConfig, McpResourceMode, McpServerEntry,
    MemoryConfig, MetricsConfig, OrchestratorConfig, PiiConfig, RetryProviderConfig,
    SalienceConfig, ScheduleEntry, SecretPatternConfig, SecretScanConfig, SensorConfig,
    SensorModality, SensorRoutingConfig, SensorSourceConfig, SessionPruneConfigToml, SpawnConfig,
    StoryCorrelationConfig, TokenBudgetConfig, TokenExchangeConfig, ToolPolicyConfig,
    ToolPolicyRuleConfig, WorkspaceConfig, WsConfig, parse_reasoning_effort, parse_tool_profile,
    parse_workflow_type,
};

// --- Auth re-exports (feature-gated) ---
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
    HeartbitPulseScheduler, InMemoryTaskStore, KafkaCommandProducer, OnTaskComplete,
    RuntimeAdvancedConfig, RuntimeAgentConfig, RuntimeEvalRequest, RuntimeEvalResponse,
    RuntimeEvalSseEvent, RuntimeGuardrailConfig, RuntimeMcpServer, RuntimeMemoryConfig,
    RuntimeOrchestratorConfig, RuntimeProviderConfig, RuntimeProviderType, RuntimeRequest,
    RuntimeResponse, RuntimeScorerConfig, RuntimeSpawnConfig, RuntimeSseEvent,
    RuntimeSubAgentConfig, RuntimeTwitterCredentials, RuntimeWorkflowConfig, RuntimeWorkflowEdge,
    RuntimeWorkflowNode, TaskOutcome, TaskState, TaskStats, TaskStore, TodoEntry, TodoList,
    TodoManageTool, UsageGroupBy, UsageQuery, UsageRow, UserContext, format_notification,
};

// --- Error re-exports ---
pub use error::Error;

// --- Eval re-exports ---
pub use eval::{
    CaseComparison, CostScorer, EvalCase, EvalComparison, EvalResult, EvalRunner, EvalScorer,
    EvalSummary, EventCollector, ExpectedToolCall, KeywordScorer, LatencyScorer, SafetyScorer,
    ScorerResult, SimilarityScorer, ToolCallCountScorer, TrajectoryScorer, build_eval_agent,
    clear_events,
};

// --- Knowledge re-exports ---
pub use knowledge::in_memory::InMemoryKnowledgeBase;
pub use knowledge::{Chunk, DocumentSource, KnowledgeBase, KnowledgeQuery, SearchResult};

// --- LLM re-exports ---
pub use llm::ApprovalDecision;
pub use llm::LlmProvider;
pub use llm::OnApproval;
pub use llm::OnText;
pub use llm::anthropic::AnthropicProvider;
pub use llm::cascade::{CascadingProvider, ConfidenceGate, HeuristicGate};
pub use llm::error_class::{ErrorClass, classify as classify_error};
pub use llm::gemini::GeminiProvider;
pub use llm::openai_compat::{AuthStyle, OpenAiCompatProvider};
pub use llm::openrouter::OpenRouterProvider;
pub use llm::pricing::estimate_cost;
pub use llm::registry::{
    ProviderInfo, detect_available_provider, get_provider, known_providers as known_llm_providers,
    resolve_api_key,
};
pub use llm::retry::{OnRetry, RetryConfig, RetryingProvider};
pub use llm::types::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, ReasoningEffort, Role,
    StopReason, TokenUsage, ToolCall, ToolChoice, ToolDefinition, ToolResult,
};
pub use llm::{BoxedProvider, DynLlmProvider};

// --- LSP re-exports ---
pub use lsp::{Diagnostic as LspDiagnostic, LspManager};

// --- Memory re-exports ---
pub use memory::Confidentiality;
pub use memory::consolidation::{ConsolidationPipeline, cluster_by_keywords};
#[cfg(feature = "local-embedding")]
pub use memory::embedding::LocalEmbeddingProvider;
pub use memory::embedding::{EmbeddingMemory, EmbeddingProvider, NoopEmbedding, OpenAiEmbedding};
pub use memory::hybrid::{cosine_similarity, rrf_fuse};
pub use memory::in_memory::InMemoryStore;
pub use memory::namespaced::NamespacedMemory;
#[cfg(feature = "postgres")]
pub use memory::postgres::PostgresMemoryStore;
pub use memory::pruning::{DEFAULT_MIN_STRENGTH, default_min_age, prune_weak_entries};
pub use memory::reflection::ReflectionTracker;
pub use memory::scoring::ScoringWeights;
pub use memory::{Memory, MemoryEntry, MemoryQuery, MemoryType};

// --- Sensor re-exports (feature-gated) ---
#[cfg(feature = "sensor")]
pub use sensor::manager::SensorManager;
#[cfg(feature = "sensor")]
pub use sensor::metrics::SensorMetrics;
#[cfg(feature = "sensor")]
pub use sensor::routing::{ModelRouter, ModelTier};
#[cfg(feature = "sensor")]
pub use sensor::stories::{Story, StoryCorrelator, StoryStatus, SubjectType};
#[cfg(feature = "sensor")]
pub use sensor::triage::context::TaskContext;
// TrustLevel is always available (defined in config.rs).
pub use config::TrustLevel;
#[cfg(feature = "sensor")]
pub use sensor::triage::{ActionCategory, Priority, TriageDecision, TriageProcessor};
#[cfg(feature = "sensor")]
pub use sensor::{Sensor, SensorEvent};

// --- Tool re-exports ---
pub use template::registry::{known_templates, resolve_template};
pub use template::skills::{SkillContent, known_skills, load_skill, load_skills};
pub use template::{AgentTemplate, PartialAgentConfig, TemplateMeta, resolve_agent_config};
#[cfg(feature = "a2a")]
pub use tool::a2a::A2aClient;
pub use tool::builtins::{
    BuiltinToolsConfig, FileTracker, OnQuestion, Question, QuestionOption, QuestionRequest,
    QuestionResponse, TodoPriority, TodoStatus, TodoStore, ToolRisk, TwitterCredentials,
    builtin_tools,
};
pub use tool::mcp::{
    AuthProvider, AuthResolver, DirectAuthProvider, DynamicAuthResolver, McpClient,
    McpPromptArgument, McpPromptDef, McpPromptMessage, McpPromptMessageContent, McpResourceContent,
    McpResourceDef, McpRoot, McpTransportPool, SamplingContent, SamplingHandler, SamplingMessage,
    SamplingModelHint, SamplingModelPreferences, SamplingRequest, StaticAuthProvider,
    StaticAuthResolver, TokenExchangeAuthProvider,
};
pub use tool::mcp_presets::{McpPreset, check_preset_env, known_presets, resolve_preset};
pub use tool::mcp_server::{McpServer, McpServerConfig, ServerResource};
pub use tool::{Tool, ToolOutput, validate_tool_input};

// --- Macro re-exports (feature-gated) ---
#[cfg(feature = "macro")]
pub use heartbit_macro::heartbit_tool;

// --- Store re-exports ---
#[cfg(feature = "postgres")]
pub use store::PostgresStore;
#[cfg(feature = "postgres")]
pub use store::postgres::PostgresAuditTrail;

// --- Workspace re-exports ---
#[cfg(all(target_os = "linux", feature = "sandbox"))]
pub use sandbox::SandboxPolicy;
pub use workspace::Workspace;
