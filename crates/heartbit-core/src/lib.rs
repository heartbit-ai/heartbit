//! # heartbit-core
//!
//! The Rust agentic framework — agents, tools, LLM providers, memory, evaluation.
//!
//! Documentation lands here as the crate's docs.rs preamble. The README
//! is rendered above this on docs.rs.

#![allow(unexpected_cfgs)]

// Modules are added one at a time as subsequent tasks move them in.
pub mod agent;
pub mod auth;
pub mod channel;
pub mod error;
pub mod http;
pub mod knowledge;
pub mod llm;
pub mod lsp;
pub mod memory;
pub mod signal;
pub mod store;
pub mod tool;
pub mod types;
pub mod util;
pub mod workspace;

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

// --- Error re-exports ---
pub use error::Error;

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
pub use memory::embedding::{EmbeddingMemory, EmbeddingProvider, NoopEmbedding, OpenAiEmbedding};
pub use memory::hybrid::{cosine_similarity, rrf_fuse};
pub use memory::in_memory::InMemoryStore;
pub use memory::namespaced::NamespacedMemory;
pub use memory::pruning::{DEFAULT_MIN_STRENGTH, default_min_age, prune_weak_entries};
pub use memory::reflection::ReflectionTracker;
pub use memory::scoring::ScoringWeights;
pub use memory::{Memory, MemoryEntry, MemoryQuery, MemoryType};

// --- Tool re-exports ---
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

// --- Workspace re-exports ---
pub use workspace::Workspace;
