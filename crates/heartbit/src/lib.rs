// --- Core modules (always available) ---
pub mod agent;
pub mod channel;
pub mod config;
pub mod error;
pub mod eval;
pub mod knowledge;
pub mod llm;
pub mod memory;
pub mod store;
pub mod tool;
pub(crate) mod util;
pub mod workspace;

pub mod lsp;

// --- Feature-gated modules ---
#[cfg(feature = "daemon")]
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

#[cfg(feature = "telegram")]
pub use channel::telegram::{
    AccessControl as TelegramAccessControl, CallbackAction, ChatSessionMap, DmPolicy,
    RateLimiter as TelegramRateLimiter, StreamBuffer, TelegramAdapter, TelegramBridge,
    TelegramConfig, approval_buttons, chunk_message, markdown_to_telegram_html,
    parse_callback_data, question_buttons,
};

// --- Agent re-exports ---
pub use agent::audit::{AuditRecord, AuditTrail, InMemoryAuditTrail};
pub use agent::blackboard::{Blackboard, InMemoryBlackboard};
pub use agent::context::ContextStrategy;
pub use agent::events::{AgentEvent, OnEvent};
pub use agent::guardrail::{GuardAction, Guardrail, GuardrailMeta};
#[cfg(feature = "sensor")]
pub use agent::guardrails::SensorSecurityGuardrail;
pub use agent::guardrails::tool_policy::{InputConstraint, ToolRule};
pub use agent::guardrails::{
    ConditionalGuardrail, ContentFenceGuardrail, GuardrailChain, GuardrailMode,
    InjectionClassifierGuardrail, LlmJudgeGuardrail, LlmJudgeGuardrailBuilder, PiiAction,
    PiiDetector, PiiGuardrail, ToolPolicyGuardrail, WarnToDeny,
};
pub use agent::instructions::{
    discover_instruction_files, load_instructions, prepend_instructions,
};
pub use agent::observability::ObservabilityMode;
pub use agent::orchestrator::{Orchestrator, OrchestratorBuilder, SubAgentConfig};
pub use agent::permission::{
    LearnedPermissions, PermissionAction, PermissionRule, PermissionRuleset,
};
pub use agent::pruner::SessionPruneConfig;
pub use agent::routing::{
    AgentCapability, ComplexitySignals, KeywordRoutingStrategy, RoutingDecision, RoutingMode,
    RoutingStrategy, TaskComplexityAnalyzer, resolve_routing_mode, should_escalate,
};
pub use agent::tool_filter::ToolProfile;
pub use agent::workflow::{
    LoopAgent, LoopAgentBuilder, ParallelAgent, ParallelAgentBuilder, SequentialAgent,
    SequentialAgentBuilder,
};
pub use agent::{AgentOutput, AgentRunner, AgentRunnerBuilder, OnInput};

// --- Config re-exports (always available — just data structs) ---
pub use config::{
    ActiveHoursConfig, AgentConfig, AgentProviderConfig, AuthConfig, CascadeConfig,
    CascadeGateConfig, CascadeTierConfig, ContextStrategyConfig, DaemonConfig, DispatchMode,
    EmbeddingConfig, GuardrailsConfig, HeartbitConfig, HeartbitPulseConfig, InjectionConfig,
    InputConstraintConfig, KafkaConfig, KnowledgeConfig, KnowledgeSourceConfig, LspConfig,
    McpServerEntry, MemoryConfig, MetricsConfig, OrchestratorConfig, PiiConfig,
    RetryProviderConfig, SalienceConfig, ScheduleEntry, SensorConfig, SensorModality,
    SensorRoutingConfig, SensorSourceConfig, SessionPruneConfigToml, StoryCorrelationConfig,
    TokenBudgetConfig, ToolPolicyConfig, ToolPolicyRuleConfig, WorkspaceConfig, WsConfig,
    parse_reasoning_effort, parse_tool_profile,
};

// --- Auth re-exports (feature-gated) ---
#[cfg(feature = "daemon")]
pub use auth::{JwksClient, JwtValidator};

// --- Daemon re-exports (feature-gated) ---
#[cfg(all(feature = "daemon", feature = "postgres"))]
pub use daemon::PostgresTaskStore;
#[cfg(feature = "daemon")]
pub use daemon::{
    CommandProducer, CronScheduler, DaemonCommand, DaemonCore, DaemonHandle, DaemonMetrics,
    DaemonTask, FileTodoStore, HeartbitPulseScheduler, InMemoryTaskStore, KafkaCommandProducer,
    OnTaskComplete, TaskOutcome, TaskState, TaskStats, TaskStore, TodoEntry, TodoList,
    TodoManageTool, UserContext, format_notification,
};

// --- Error re-exports ---
pub use error::Error;

// --- Eval re-exports ---
pub use eval::{
    EvalCase, EvalResult, EvalRunner, EvalScorer, EvalSummary, EventCollector, ExpectedToolCall,
    KeywordScorer, ScorerResult, SimilarityScorer, TrajectoryScorer, build_eval_agent,
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
pub use llm::openrouter::OpenRouterProvider;
pub use llm::pricing::estimate_cost;
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
#[cfg(feature = "a2a")]
pub use tool::a2a::A2aClient;
pub use tool::builtins::{
    BuiltinToolsConfig, FileTracker, OnQuestion, Question, QuestionOption, QuestionRequest,
    QuestionResponse, TodoPriority, TodoStatus, TodoStore, builtin_tools,
};
pub use tool::mcp::{AuthProvider, McpClient, StaticAuthProvider, TokenExchangeAuthProvider};
pub use tool::{Tool, ToolOutput, validate_tool_input};

// --- Store re-exports ---
#[cfg(feature = "postgres")]
pub use store::PostgresStore;
#[cfg(feature = "postgres")]
pub use store::postgres::PostgresAuditTrail;

// --- Workspace re-exports ---
pub use workspace::Workspace;
