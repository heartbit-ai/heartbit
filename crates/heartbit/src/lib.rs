pub mod agent;
pub mod channel;
pub mod config;
pub mod daemon;
pub mod error;
pub mod knowledge;
pub mod llm;
pub mod lsp;
pub mod memory;
pub mod sensor;
pub mod store;
pub mod tool;
pub(crate) mod util;
pub mod workflow;
pub mod workspace;

pub use channel::bridge::{InteractionBridge, OutboundMessage};
pub use channel::session::{
    InMemorySessionStore, PostgresSessionStore, Session, SessionMessage, SessionRole, SessionStore,
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

pub use agent::blackboard::{Blackboard, InMemoryBlackboard};
pub use agent::context::ContextStrategy;
pub use agent::events::{AgentEvent, OnEvent};
pub use agent::guardrail::{GuardAction, Guardrail};
pub use agent::guardrails::ContentFenceGuardrail;
pub use agent::guardrails::SensorSecurityGuardrail;
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
    AgentCapability, ComplexitySignals, RoutingDecision, RoutingMode, TaskComplexityAnalyzer,
    resolve_routing_mode, should_escalate,
};
pub use agent::tool_filter::ToolProfile;
pub use agent::{AgentOutput, AgentRunner, AgentRunnerBuilder, OnInput};
pub use config::{
    ActiveHoursConfig, AgentConfig, AgentProviderConfig, AuthConfig, CascadeConfig,
    CascadeGateConfig, CascadeTierConfig, ContextStrategyConfig, DaemonConfig, DispatchMode,
    EmbeddingConfig, HeartbitConfig, HeartbitPulseConfig, KafkaConfig, KnowledgeConfig,
    KnowledgeSourceConfig, LspConfig, McpServerEntry, MemoryConfig, MetricsConfig,
    OrchestratorConfig, RetryProviderConfig, SalienceConfig, ScheduleEntry, SensorConfig,
    SensorRoutingConfig, SensorSourceConfig, SessionPruneConfigToml, StoryCorrelationConfig,
    TokenBudgetConfig, WorkspaceConfig, WsConfig, parse_reasoning_effort, parse_tool_profile,
};
pub use daemon::{
    CommandProducer, CronScheduler, DaemonCommand, DaemonCore, DaemonHandle, DaemonMetrics,
    DaemonTask, FileTodoStore, HeartbitPulseScheduler, InMemoryTaskStore, KafkaCommandProducer,
    OnTaskComplete, PostgresTaskStore, TaskOutcome, TaskState, TaskStats, TaskStore, TodoEntry,
    TodoList, TodoManageTool, format_notification,
};
pub use error::Error;
pub use knowledge::in_memory::InMemoryKnowledgeBase;
pub use knowledge::{Chunk, DocumentSource, KnowledgeBase, KnowledgeQuery, SearchResult};
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
pub use lsp::{Diagnostic as LspDiagnostic, LspManager};
pub use memory::Confidentiality;
pub use memory::consolidation::{ConsolidationPipeline, cluster_by_keywords};
pub use memory::embedding::{EmbeddingMemory, EmbeddingProvider, NoopEmbedding, OpenAiEmbedding};
pub use memory::hybrid::{cosine_similarity, rrf_fuse};
pub use memory::in_memory::InMemoryStore;
pub use memory::namespaced::NamespacedMemory;
pub use memory::postgres::PostgresMemoryStore;
pub use memory::pruning::{DEFAULT_MIN_STRENGTH, default_min_age, prune_weak_entries};
pub use memory::reflection::ReflectionTracker;
pub use memory::scoring::ScoringWeights;
pub use memory::{Memory, MemoryEntry, MemoryQuery, MemoryType};
pub use sensor::manager::SensorManager;
pub use sensor::metrics::SensorMetrics;
pub use sensor::routing::{ModelRouter, ModelTier};
pub use sensor::stories::{Story, StoryCorrelator, StoryStatus, SubjectType};
pub use sensor::triage::context::TaskContext;
pub use sensor::triage::context::TrustLevel;
pub use sensor::triage::{ActionCategory, Priority, TriageDecision, TriageProcessor};
pub use sensor::{Sensor, SensorEvent, SensorModality};
pub use tool::a2a::A2aClient;
pub use tool::builtins::{
    BuiltinToolsConfig, FileTracker, OnQuestion, Question, QuestionOption, QuestionRequest,
    QuestionResponse, TodoPriority, TodoStatus, TodoStore, builtin_tools,
};
pub use tool::mcp::McpClient;
pub use tool::{Tool, ToolOutput, validate_tool_input};
pub use workspace::Workspace;
