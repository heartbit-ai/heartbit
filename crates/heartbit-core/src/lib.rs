//! # heartbit-core
//!
//! The Rust agentic framework — agents, tools, LLM providers, memory, evaluation.
//!
//! Documentation lands here as the crate's docs.rs preamble. The README
//! is rendered above this on docs.rs.

#![deny(missing_docs)]

// Modules are added one at a time as subsequent tasks move them in.
pub mod agent;
pub mod auth;
pub mod channel;
pub mod config;
pub mod error;
pub mod eval;
pub mod execution_context;
pub mod http;
pub mod knowledge;
pub mod llm;
pub mod lsp;
pub mod memory;
pub mod persona;
pub mod sandbox;
#[cfg(unix)]
pub mod signal;
pub mod store;
pub mod template;
pub mod tool;
pub(crate) mod types;
pub(crate) mod util;
pub mod workspace;

// --- Agent re-exports ---
pub use agent::audit::{AuditMode, AuditRecord, AuditTrail, InMemoryAuditTrail};
pub use agent::batch::{BatchConfig, BatchExecutor, BatchExecutorBuilder, BatchResult};
pub use agent::blackboard::{Blackboard, InMemoryBlackboard};
pub use agent::cache::ResponseCache;
pub use agent::context::ContextStrategy;
pub use agent::dag::{DagAgent, DagAgentBuilder};
pub use agent::debate::{DebateAgent, DebateAgentBuilder};
pub use agent::evaluator::{EvaluatorOptimizerAgent, EvaluatorOptimizerAgentBuilder};
pub use agent::events::{AgentEvent, OnEvent};
pub use agent::guardrail::{GuardAction, Guardrail};
#[allow(deprecated)]
pub use agent::guardrails::ContentFenceGuardrail;
pub use agent::guardrails::tool_policy::{InputConstraint, ToolRule};
pub use agent::guardrails::{
    ActionBudgetGuardrail, ActionBudgetGuardrailBuilder, BehaviorRule, BehavioralMonitorGuardrail,
    BehavioralMonitorGuardrailBuilder, BudgetRule, ConditionalGuardrail, GuardrailChain,
    GuardrailMode, InjectionClassifierGuardrail, LlmJudgeGuardrail, LlmJudgeGuardrailBuilder,
    PiiAction, PiiDetector, PiiGuardrail, SecretAction, SecretScannerGuardrail,
    SecretScannerGuardrailBuilder, SensorSecurityGuardrail, ToolPolicyGuardrail, WarnToDeny,
};
pub use agent::handoff::{HandoffRunner, HandoffRunnerBuilder, make_handoff_tool};
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
pub use agent::tenant_tracker::{TenantTokenState, TenantTokenTracker, TokenReservation};
pub use agent::tool_filter::ToolProfile;
pub use agent::voting::{VoteResult, VotingAgent, VotingAgentBuilder};
pub use agent::workflow::{
    LoopAgent, LoopAgentBuilder, ParallelAgent, ParallelAgentBuilder, SequentialAgent,
    SequentialAgentBuilder, WorkflowRouter, WorkflowType,
};
pub use agent::{AgentOutput, AgentRunner, AgentRunnerBuilder, OnInput};

// --- Error re-exports ---
pub use error::Error;

// --- Execution context re-exports ---
pub use execution_context::{AuditSink, CredentialResolver, ExecutionContext, Secret};

// --- Eval re-exports ---
pub use eval::{
    CaseComparison, CostScorer, EvalCase, EvalComparison, EvalResult, EvalRunner, EvalScorer,
    EvalSummary, EventCollector, KeywordScorer, LatencyScorer, SafetyScorer, ScorerResult,
    SimilarityScorer, ToolCallCountScorer, TrajectoryScorer, build_eval_agent, clear_events,
};

// --- Sandbox re-exports ---
pub use sandbox::{CorePathPolicy, CorePathPolicyBuilder};

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
pub use llm::circuit::{
    CircuitBreakerProvider, CircuitConfig, CircuitKey, CircuitPermit, CircuitTracker,
    ProviderCircuit, is_circuit_failure,
};
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
pub use memory::consolidation::{
    ConsolidationPipeline, ConsolidationResult, DEFAULT_SUMMARY_MAX_TOKENS, cluster_by_keywords,
};
pub use memory::embedding::{EmbeddingMemory, EmbeddingProvider, NoopEmbedding, OpenAiEmbedding};
pub use memory::hybrid::{cosine_similarity, rrf_fuse};
pub use memory::in_memory::InMemoryStore;
pub use memory::namespaced::NamespacedMemory;
pub use memory::pruning::{DEFAULT_MIN_STRENGTH, default_min_age, prune_weak_entries};
pub use memory::reflection::ReflectionTracker;
pub use memory::scoring::ScoringWeights;
pub use memory::{Memory, MemoryEntry, MemoryQuery, MemoryType};

// --- Persona re-exports ---
pub use persona::{
    AuthorshipMode, Persona, PersonaExpansion, PersonaParams, PersonaRegistry, ReviewSpec,
    TopicContextProvider, TriggerSpec,
};

// --- Tool re-exports ---
#[cfg(feature = "a2a")]
pub use tool::a2a::A2aClient;
#[cfg(feature = "ghost-domain-config")]
pub use tool::builtins::TwitterCredentials;
pub use tool::builtins::{
    BuiltinToolsConfig, FileTracker, OnQuestion, Question, QuestionOption, QuestionRequest,
    QuestionResponse, TodoPriority, TodoStatus, TodoStore, ToolRisk, builtin_tools,
};
pub use tool::handoff::{HandoffContextMode, HandoffTarget, HandoffTool};
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

// --- Benchmark-only helpers ---
//
// Thin wrappers over crate-internal hot paths exposed exclusively for
// criterion benchmarks under `crates/heartbit-core/benches/`. Gated
// behind the `bench-internals` feature so downstream consumers cannot
// accidentally rely on this surface; the wrappers allocate their own
// internal state and never leak crate-private types across the boundary.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod __bench {
    #![allow(missing_docs)]

    /// Feed `chunk` to a fresh `SseParser` and return the number of
    /// emitted events. Used to benchmark per-chunk allocation overhead
    /// in the Anthropic SSE hot path (P-LLM-2, P-LLM-14).
    pub fn sse_parse_chunk(chunk: &str) -> usize {
        let mut parser = crate::llm::anthropic::SseParser::new();
        parser.feed(chunk).len()
    }

    /// Mock LLM provider that returns the same canned response on every
    /// call (no draining). Designed for the agent-ReAct-turn bench
    /// (Bench-NEW-1 in `tasks/perf-audit-v2-bench-gaps.md`) where each
    /// criterion sample needs a fresh `execute()` against an identical
    /// provider response — different from the test-only `MockProvider`
    /// which drains a queue.
    pub struct BenchMockProvider {
        response: crate::llm::types::CompletionResponse,
    }

    impl BenchMockProvider {
        /// Build a provider that always returns a single text response.
        pub fn new_text(text: impl Into<String>) -> Self {
            use crate::llm::types::{CompletionResponse, ContentBlock, StopReason, TokenUsage};
            Self {
                response: CompletionResponse {
                    content: vec![ContentBlock::Text { text: text.into() }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage {
                        input_tokens: 64,
                        output_tokens: 16,
                        ..Default::default()
                    },
                    model: None,
                },
            }
        }
    }

    impl crate::llm::LlmProvider for BenchMockProvider {
        async fn complete(
            &self,
            _request: crate::llm::types::CompletionRequest,
        ) -> Result<crate::llm::types::CompletionResponse, crate::error::Error> {
            Ok(self.response.clone())
        }

        fn model_name(&self) -> Option<&str> {
            Some("bench-mock")
        }
    }
}
