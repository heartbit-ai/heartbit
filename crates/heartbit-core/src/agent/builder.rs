use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::error::Error;
use crate::knowledge::KnowledgeBase;
use crate::llm::LlmProvider;
use crate::llm::types::ToolDefinition;
use crate::memory::Memory;
use crate::tool::Tool;
use crate::tool::builtins::OnQuestion;

use super::audit;
use super::cache;
use super::context::ContextStrategy;
use super::events::OnEvent;
use super::guardrail::Guardrail;
use super::instructions;
use super::observability;
use super::permission;
use super::pruner;
use super::runner::{AgentRunner, OnInput, RESOURCEFULNESS_GUIDELINES};
use super::tool_filter;

/// Appended to the system prompt when restore-on-demand is enabled, so the model
/// knows to act on a pruned marker.
const CONTEXT_RECALL_HINT: &str = "\n\nNote: old tool outputs may be truncated with a \
    '[pruned: … id=<ref>]' marker. Call fetch_full_output(<ref>) to restore the exact content, \
    or recall_context(<query>) to find older outputs by meaning.";

/// Builder for [`AgentRunner`].
///
/// Construct via [`AgentRunner::builder`], configure the agent with chainable
/// setter methods, then call [`build`](AgentRunnerBuilder::build) to produce
/// the runner. Build validates several invariants: `on_input` and
/// `structured_schema` are mutually exclusive, `max_tool_calls_per_turn = 0` is
/// rejected, and turn/token limits must be non-zero. For multi-agent scenarios
/// use [`OrchestratorBuilder`](crate::agent::orchestrator::OrchestratorBuilder)
/// instead, which internally wraps an `AgentRunner` with sub-agent delegation.
pub struct AgentRunnerBuilder<P: LlmProvider> {
    pub(super) provider: Arc<P>,
    pub(super) name: String,
    pub(super) system_prompt: String,
    pub(super) tools: Vec<Arc<dyn Tool>>,
    pub(super) max_turns: usize,
    pub(super) max_tokens: u32,
    pub(super) context_strategy: Option<ContextStrategy>,
    pub(super) summarize_threshold: Option<u32>,
    pub(super) context_window_tokens: Option<u32>,
    pub(super) compaction_threshold_fraction: f32,
    pub(super) memory: Option<Arc<dyn Memory>>,
    pub(super) knowledge_base: Option<Arc<dyn KnowledgeBase>>,
    pub(super) on_text: Option<Arc<crate::llm::OnText>>,
    pub(super) on_reasoning: Option<Arc<crate::llm::OnReasoning>>,
    pub(super) on_approval: Option<Arc<crate::llm::OnApproval>>,
    pub(super) tool_timeout: Option<Duration>,
    pub(super) max_tool_output_bytes: Option<usize>,
    pub(super) structured_schema: Option<serde_json::Value>,
    pub(super) on_event: Option<Arc<OnEvent>>,
    pub(super) guardrails: Vec<Arc<dyn Guardrail>>,
    pub(super) on_question: Option<Arc<OnQuestion>>,
    pub(super) on_input: Option<Arc<OnInput>>,
    pub(super) interrupt: Option<super::interrupt::InterruptHandle>,
    pub(super) run_timeout: Option<Duration>,
    pub(super) reasoning_effort: Option<crate::llm::types::ReasoningEffort>,
    pub(super) enable_reflection: bool,
    pub(super) tool_output_compression_threshold: Option<usize>,
    pub(super) max_tools_per_turn: Option<usize>,
    pub(super) tool_profile: Option<tool_filter::ToolProfile>,
    pub(super) max_identical_tool_calls: Option<u32>,
    pub(super) max_fuzzy_identical_tool_calls: Option<u32>,
    /// Hard cap on the number of tool invocations the LLM may emit per turn.
    /// Distinct from `max_tools_per_turn` (which limits tool *definitions* offered
    /// to the LLM). `None` = unlimited. Zero is rejected at build time.
    pub(super) max_tool_calls_per_turn: Option<u32>,
    pub(super) permission_rules: permission::PermissionRuleset,
    /// Instruction file contents to prepend to the system prompt.
    pub(super) instruction_text: Option<String>,
    pub(super) learned_permissions: Option<Arc<std::sync::Mutex<permission::LearnedPermissions>>>,
    pub(super) lsp_manager: Option<Arc<crate::lsp::LspManager>>,
    pub(super) session_prune_config: Option<pruner::SessionPruneConfig>,
    pub(super) enable_recursive_summarization: bool,
    pub(super) reflection_threshold: Option<u32>,
    pub(super) consolidate_on_exit: bool,
    pub(super) observability_mode: Option<observability::ObservabilityMode>,
    /// Optional workspace root for file tool path resolution and system prompt.
    pub(super) workspace: Option<std::path::PathBuf>,
    /// Hard limit on cumulative tokens (input + output) across all turns.
    pub(super) max_total_tokens: Option<u64>,
    /// Optional persistent goal gated by an independent judge.
    pub(super) goal: Option<super::goal::GoalCondition>,
    /// Controls whether audit records include full content or metadata only.
    pub(super) audit_mode: audit::AuditMode,
    /// Optional audit trail for recording untruncated agent decisions.
    pub(super) audit_trail: Option<Arc<dyn audit::AuditTrail>>,
    /// Optional user context for multi-tenant audit enrichment.
    pub(super) audit_user_id: Option<String>,
    pub(super) audit_tenant_id: Option<String>,
    /// Delegation chain for audit records (e.g., `["heartbit-agent"]` when acting OBO user).
    pub(super) audit_delegation_chain: Vec<String>,
    /// Optional LRU response cache size. When set, builds a `ResponseCache`.
    pub(super) response_cache_size: Option<usize>,
    /// Optional per-tenant in-flight token tracker. When set, the runner calls
    /// `tracker.adjust(&scope, delta)` after each LLM response to reconcile
    /// actual usage against the estimate. Has no effect when `audit_tenant_id`
    /// is unset.
    pub(super) tenant_tracker: Option<Arc<crate::agent::tenant_tracker::TenantTokenTracker>>,
    /// Optional per-run context recall store. When set, every tool output is
    /// indexed by `tool_call_id` so pruned results can be restored on demand.
    pub(super) context_recall_store: Option<Arc<crate::agent::context_recall::ContextRecallStore>>,
}

impl<P: LlmProvider> AgentRunnerBuilder<P> {
    /// Set the agent's name (used in logs, traces, and the orchestrator's prompt).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the agent's system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Register a single tool.
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Register a batch of tools.
    ///
    /// SECURITY (F-MCP-2): when MCP-discovered tools and builtins coexist,
    /// **register the trusted builtins first**. The runner deduplicates by
    /// name with first-wins semantics, so a hostile MCP server that exports a
    /// tool named `bash` will be shadowed by the local `bash` builtin only if
    /// the builtin was added before. The collision is logged at `error!` and
    /// emits a `tool_name_collision` audit signal.
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Cap on agent loop iterations before the runner aborts.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// `max_tokens` cap on each LLM completion the runner issues.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Context window management strategy.
    pub fn context_strategy(mut self, strategy: ContextStrategy) -> Self {
        self.context_strategy = Some(strategy);
        self
    }

    /// Set the token threshold at which to trigger automatic summarization.
    pub fn summarize_threshold(mut self, threshold: u32) -> Self {
        self.summarize_threshold = Some(threshold);
        self
    }

    /// Set the model context window (tokens) to enable the proactive compaction
    /// backstop (triggers on real prompt tokens crossing the threshold fraction).
    pub fn context_window_tokens(mut self, tokens: u32) -> Self {
        self.context_window_tokens = Some(tokens);
        self
    }

    /// Fraction of the context window at which to proactively compact (default 0.70).
    pub fn compaction_threshold_fraction(mut self, fraction: f32) -> Self {
        self.compaction_threshold_fraction = fraction;
        self
    }

    /// Attach a memory store to the agent. Memory tools (store, recall, update,
    /// forget, consolidate) are created at `build()` time using the builder's `name`.
    ///
    /// Call `.name()` before or after `.memory()` — the agent name is resolved at build.
    pub fn memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Attach a knowledge base to the agent. The `knowledge_search` tool is
    /// added at `build()` time.
    pub fn knowledge(mut self, kb: Arc<dyn KnowledgeBase>) -> Self {
        self.knowledge_base = Some(kb);
        self
    }

    /// Set a callback for streaming text output. When set, the agent uses
    /// `stream_complete` instead of `complete`, calling the callback for each
    /// text delta as it arrives from the LLM.
    ///
    /// The callback must not panic. A panic inside the callback will propagate
    /// through the agent loop and abort the run.
    pub fn on_text(mut self, callback: Arc<crate::llm::OnText>) -> Self {
        self.on_text = Some(callback);
        self
    }

    /// Set a callback invoked with each reasoning (chain-of-thought) delta during
    /// streaming, for reasoning models. Separate from [`on_text`](Self::on_text)
    /// so a UI can render thinking live and distinctly from the answer. Only
    /// honored on the streaming path (when `on_text` is also set).
    pub fn on_reasoning(mut self, callback: Arc<crate::llm::OnReasoning>) -> Self {
        self.on_reasoning = Some(callback);
        self
    }

    /// Set a callback for human-in-the-loop approval before tool execution.
    ///
    /// When set, the callback is invoked with the list of tool calls before
    /// each execution round. If it returns `false`, tool execution is denied
    /// and the agent receives error results, allowing the LLM to adjust.
    pub fn on_approval(mut self, callback: Arc<crate::llm::OnApproval>) -> Self {
        self.on_approval = Some(callback);
        self
    }

    /// Set a timeout for individual tool executions. If a tool does not
    /// complete within this duration, the execution is cancelled and an
    /// error result is returned to the LLM.
    ///
    /// Default: `None` (no timeout).
    pub fn tool_timeout(mut self, timeout: Duration) -> Self {
        self.tool_timeout = Some(timeout);
        self
    }

    /// Set a maximum byte size for individual tool output content.
    ///
    /// Tool results exceeding this limit are truncated with a
    /// `[truncated: N bytes omitted]` suffix, preventing oversized results
    /// from blowing out the context window.
    ///
    /// Default: `None` (no truncation).
    pub fn max_tool_output_bytes(mut self, max: usize) -> Self {
        self.max_tool_output_bytes = Some(max);
        self
    }

    /// Set a JSON Schema for structured output. The agent will receive a
    /// synthetic `__respond__` tool with this schema. When the LLM calls
    /// `__respond__`, its input is extracted as `AgentOutput::structured`.
    ///
    /// The agent can still use regular tools before producing output.
    pub fn structured_schema(mut self, schema: serde_json::Value) -> Self {
        self.structured_schema = Some(schema);
        self
    }

    /// Set a callback for structured agent events. Events are emitted at key
    /// points in the agent loop: run start/end, turn transitions, LLM responses,
    /// tool call start/completion, approval decisions, and context summarization.
    pub fn on_event(mut self, callback: Arc<OnEvent>) -> Self {
        self.on_event = Some(callback);
        self
    }

    /// Add a single guardrail. Multiple guardrails are evaluated in order;
    /// first `Deny` wins.
    pub fn guardrail(mut self, guardrail: Arc<dyn Guardrail>) -> Self {
        self.guardrails.push(guardrail);
        self
    }

    /// Add multiple guardrails at once.
    pub fn guardrails(mut self, guardrails: Vec<Arc<dyn Guardrail>>) -> Self {
        self.guardrails.extend(guardrails);
        self
    }

    /// Set a callback for structured questions to the user. When set, a
    /// `question` tool is added at `build()` time allowing the agent to
    /// ask the user structured questions with predefined options.
    pub fn on_question(mut self, callback: Arc<OnQuestion>) -> Self {
        self.on_question = Some(callback);
        self
    }

    /// Set a callback for interactive mode. When set and the LLM returns
    /// text without tool calls, the callback is invoked to get the next
    /// user message. Return `Some(message)` to continue the conversation
    /// or `None` to end the session.
    pub fn on_input(mut self, callback: Arc<OnInput>) -> Self {
        self.on_input = Some(callback);
        self
    }

    /// Attach a re-armable interrupt handle. Triggering it abandons the in-flight
    /// turn (aborting LLM generation) and returns to awaiting input, preserving the
    /// session. Only meaningful with `on_input` (the interactive path).
    pub fn interrupt(mut self, handle: super::interrupt::InterruptHandle) -> Self {
        self.interrupt = Some(handle);
        self
    }

    /// Set a wall-clock deadline for the entire run. If the agent does not
    /// complete within this duration, `Error::RunTimeout` is returned.
    ///
    /// Default: `None` (no deadline).
    pub fn run_timeout(mut self, timeout: Duration) -> Self {
        self.run_timeout = Some(timeout);
        self
    }

    /// Set the reasoning/thinking effort level. Enables extended thinking
    /// on models that support it (e.g., Qwen3 via OpenRouter, Claude).
    ///
    /// Default: `None` (no reasoning).
    pub fn reasoning_effort(mut self, effort: crate::llm::types::ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Enable post-tool reflection prompts (Reflexion/CRITIC pattern).
    pub fn enable_reflection(mut self, enabled: bool) -> Self {
        self.enable_reflection = enabled;
        self
    }

    /// Byte threshold above which tool output is compressed via an LLM call.
    pub fn tool_output_compression_threshold(mut self, threshold: usize) -> Self {
        self.tool_output_compression_threshold = Some(threshold);
        self
    }

    /// Cap on tool definitions sent to the LLM per turn (dynamic tool selection).
    pub fn max_tools_per_turn(mut self, max: usize) -> Self {
        self.max_tools_per_turn = Some(max);
        self
    }

    /// Set a static tool profile to pre-filter tools before dynamic selection.
    ///
    /// When set, tool definitions are filtered to the profile's subset before
    /// `max_tools_per_turn` scoring applies. Use `ToolProfile::Conversational`
    /// for chat-only agents, `Standard` for code agents, `Full` for all tools.
    pub fn tool_profile(mut self, profile: tool_filter::ToolProfile) -> Self {
        self.tool_profile = Some(profile);
        self
    }

    /// Set the maximum number of consecutive identical tool-call turns before
    /// the agent receives an error result instead of executing the tools.
    ///
    /// This detects "doom loops" where the LLM keeps repeating the exact same
    /// tool calls. After `max` consecutive identical turns, all tool calls in
    /// the turn receive an error result asking the LLM to try a different approach.
    ///
    /// Default: `None` (no detection).
    pub fn max_identical_tool_calls(mut self, max: u32) -> Self {
        self.max_identical_tool_calls = Some(max);
        self
    }

    /// Set the maximum number of consecutive fuzzy-identical tool-call turns
    /// before the agent receives an error result. Fuzzy matching compares sorted
    /// tool names (ignoring inputs), catching loops where the agent retries the
    /// same tools with different arguments.
    ///
    /// Default: `None` (no fuzzy detection).
    pub fn max_fuzzy_identical_tool_calls(mut self, max: u32) -> Self {
        self.max_fuzzy_identical_tool_calls = Some(max);
        self
    }

    /// Cap the number of tool *invocations* the LLM may emit per turn.
    /// When the LLM returns more tool_use blocks than `cap`, the run
    /// returns `Error::Agent` (wrapped in `Error::WithPartialUsage`) and
    /// no tools are dispatched.
    ///
    /// **Distinct from `max_tools_per_turn`**: that one limits the *tool
    /// definitions* offered to the LLM before it responds (pre-filter).
    /// This one caps the *invocations* in the LLM's actual response
    /// (post-response). Both can be set independently.
    ///
    /// Default: `None` (unlimited). Recommended for production: 8.
    /// Zero is rejected at build time.
    pub fn max_tool_calls_per_turn(mut self, cap: u32) -> Self {
        self.max_tool_calls_per_turn = Some(cap);
        self
    }

    /// Set declarative permission rules for tool calls.
    ///
    /// Rules are evaluated per tool call before the `on_approval` callback.
    /// `Allow` executes without asking, `Deny` returns an error result,
    /// `Ask` falls through to the `on_approval` callback.
    pub fn permission_rules(mut self, rules: permission::PermissionRuleset) -> Self {
        self.permission_rules = rules;
        self
    }

    /// Set learned permissions for persisting AlwaysAllow/AlwaysDeny decisions.
    ///
    /// When set, approval decisions with `AlwaysAllow` or `AlwaysDeny` are
    /// saved to disk and injected into the live permission ruleset.
    pub fn learned_permissions(
        mut self,
        learned: Arc<std::sync::Mutex<permission::LearnedPermissions>>,
    ) -> Self {
        self.learned_permissions = Some(learned);
        self
    }

    /// Set an LSP manager for collecting diagnostics after file-modifying tools.
    ///
    /// When set, after any tool named `write`, `edit`, or `patch` completes,
    /// the manager reads the modified file and collects diagnostics from the
    /// language server. Diagnostics are appended to the tool result so the
    /// LLM sees compilation errors immediately.
    pub fn lsp_manager(mut self, manager: Arc<crate::lsp::LspManager>) -> Self {
        self.lsp_manager = Some(manager);
        self
    }

    /// Enable session pruning to reduce token usage by truncating old tool results.
    pub fn session_prune_config(mut self, config: pruner::SessionPruneConfig) -> Self {
        self.session_prune_config = Some(config);
        self
    }

    /// Enable recursive (cluster-then-summarize) summarization for long conversations.
    pub fn enable_recursive_summarization(mut self, enable: bool) -> Self {
        self.enable_recursive_summarization = enable;
        self
    }

    /// Set cumulative importance threshold for memory reflection triggers.
    /// When the sum of stored memory importance values exceeds this threshold,
    /// the store tool appends a reflection hint to guide the agent.
    pub fn reflection_threshold(mut self, threshold: u32) -> Self {
        self.reflection_threshold = Some(threshold);
        self
    }

    /// Enable automatic memory consolidation at session end.
    ///
    /// When enabled, clusters related episodic memories by keyword overlap
    /// and merges them into semantic summaries. Requires memory to be configured.
    /// Adds LLM calls at session end (one per cluster).
    pub fn consolidate_on_exit(mut self, enable: bool) -> Self {
        self.consolidate_on_exit = enable;
        self
    }

    /// Set the observability verbosity mode for this agent.
    ///
    /// Controls how much detail is recorded in tracing spans:
    /// - `Production`: span names + durations only (near-zero overhead)
    /// - `Analysis`: + metrics (tokens, latencies, costs)
    /// - `Debug`: + full payloads (truncated to 4KB)
    ///
    /// When not set, resolved via `HEARTBIT_OBSERVABILITY` env var or default (`Production`).
    pub fn observability_mode(mut self, mode: observability::ObservabilityMode) -> Self {
        self.observability_mode = Some(mode);
        self
    }

    /// Provide pre-loaded instruction text to prepend to the system prompt.
    ///
    /// Use [`instructions::load_instructions`] to load from file paths, or
    /// [`instructions::discover_instruction_files`] to auto-discover them.
    pub fn instruction_text(mut self, text: impl Into<String>) -> Self {
        let text = text.into();
        if !text.is_empty() {
            self.instruction_text = Some(text);
        }
        self
    }

    /// Set a hard limit on cumulative tokens (input + output) across all turns.
    ///
    /// When the total tokens consumed exceed this limit, the agent returns
    /// `Error::BudgetExceeded` with partial usage data.
    ///
    /// Default: `None` (no budget).
    pub fn max_total_tokens(mut self, max: u64) -> Self {
        self.max_total_tokens = Some(max);
        self
    }

    /// Set a persistent [`GoalCondition`](super::goal::GoalCondition): after the
    /// agent would naturally finish, an INDEPENDENT judge decides whether the
    /// objective is met. If not, the judge's reason is re-injected and the agent
    /// continues (bounded by the goal's `max_continuations` and the agent's
    /// `max_turns`). [`AgentOutput::goal_met`](super::AgentOutput::goal_met)
    /// records the outcome.
    pub fn goal(mut self, goal: super::goal::GoalCondition) -> Self {
        self.goal = Some(goal);
        self
    }

    /// Set the audit mode controlling what data is stored in audit records.
    ///
    /// - `Full` (default): all content is recorded.
    /// - `MetadataOnly`: user content fields are replaced with `[stripped]`.
    pub fn audit_mode(mut self, mode: audit::AuditMode) -> Self {
        self.audit_mode = mode;
        self
    }

    /// Attach an audit trail for recording untruncated agent decisions.
    ///
    /// When set, every LLM response, tool call, tool result, run completion,
    /// run failure, and guardrail denial is recorded with full payloads.
    /// Recording is best-effort: failures are logged, never abort the agent.
    pub fn audit_trail(mut self, trail: Arc<dyn audit::AuditTrail>) -> Self {
        self.audit_trail = Some(trail);
        self
    }

    /// Set user context for multi-tenant audit enrichment.
    /// When set, all `AuditRecord` entries include the user and tenant IDs.
    pub fn audit_user_context(
        mut self,
        user_id: impl Into<String>,
        tenant_id: impl Into<String>,
    ) -> Self {
        self.audit_user_id = Some(user_id.into());
        self.audit_tenant_id = Some(tenant_id.into());
        self
    }

    /// Set the delegation chain for audit records.
    ///
    /// Populated when the daemon acts on behalf of a user via RFC 8693 token exchange.
    /// The chain records which agent(s) are in the delegation path.
    pub fn audit_delegation_chain(mut self, chain: Vec<String>) -> Self {
        self.audit_delegation_chain = chain;
        self
    }

    /// Enable an LRU response cache with the given maximum number of entries.
    /// Identical requests (same system prompt, messages, and tool names) return
    /// cached responses without calling the LLM. Only non-streaming calls are cached.
    /// Size must be at least 1.
    pub fn response_cache_size(mut self, size: usize) -> Self {
        self.response_cache_size = Some(size);
        self
    }

    /// Set the agent's workspace directory. When set, file tools resolve
    /// relative paths against this directory, BashTool starts here, and a
    /// workspace hint is appended to the system prompt.
    pub fn workspace(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.workspace = Some(path.into());
        self
    }

    /// Optional per-tenant in-flight token tracker. When set, the runner
    /// calls `tracker.adjust(&scope, delta)` after each LLM response,
    /// reconciling the per-tenant `in_flight` counter against the
    /// estimated reservation made at submit time. Has no effect when
    /// `audit_tenant_id` is unset.
    pub fn tenant_tracker(
        mut self,
        tracker: Arc<crate::agent::tenant_tracker::TenantTokenTracker>,
    ) -> Self {
        self.tenant_tracker = Some(tracker);
        self
    }

    /// Enable restore-on-demand: share a `ContextRecallStore` so tool outputs are
    /// indexed for `fetch_full_output` / `recall_context`. Pass the SAME store
    /// into `BuiltinToolsConfig.context_recall_store` so the tools are registered.
    pub fn context_recall_store(
        mut self,
        store: Arc<crate::agent::context_recall::ContextRecallStore>,
    ) -> Self {
        self.context_recall_store = Some(store);
        self
    }

    /// Validate the builder and produce a ready-to-run [`AgentRunner`].
    ///
    /// Returns [`Error::Config`] for mis-configured runners (empty name, zero `max_turns`,
    /// mutually-exclusive options like `on_input` + `structured_schema`, etc.).
    pub fn build(self) -> Result<AgentRunner<P>, Error> {
        if self.name.is_empty() {
            return Err(Error::Config("agent name must not be empty".into()));
        }
        if self.max_turns == 0 {
            return Err(Error::Config("max_turns must be at least 1".into()));
        }
        if self.max_tokens == 0 {
            return Err(Error::Config("max_tokens must be at least 1".into()));
        }
        if matches!(
            self.context_strategy,
            Some(ContextStrategy::SlidingWindow { .. })
        ) && self.summarize_threshold.is_some()
        {
            return Err(Error::Config(
                "cannot use summarize_threshold with SlidingWindow context strategy".into(),
            ));
        }
        if self.on_input.is_some() && self.structured_schema.is_some() {
            return Err(Error::Config(
                "on_input (interactive mode) and structured_schema are mutually exclusive".into(),
            ));
        }
        // A goal gates only the text natural-completion exit; with structured
        // output the run returns via the `__respond__` path, so a goal would be a
        // silent no-op. Reject it at build time rather than mislead the caller.
        if self.goal.is_some() && self.structured_schema.is_some() {
            return Err(Error::Config(
                "goal and structured_schema are mutually exclusive (a goal gates the text \
                 completion exit, which structured output bypasses)"
                    .into(),
            ));
        }
        if self.max_tools_per_turn == Some(0) {
            return Err(Error::Config(
                "max_tools_per_turn must be at least 1".into(),
            ));
        }
        if self.tool_output_compression_threshold == Some(0) {
            return Err(Error::Config(
                "tool_output_compression_threshold must be at least 1".into(),
            ));
        }
        if self.max_identical_tool_calls == Some(0) {
            return Err(Error::Config(
                "max_identical_tool_calls must be at least 1".into(),
            ));
        }
        if self.max_fuzzy_identical_tool_calls == Some(0) {
            return Err(Error::Config(
                "max_fuzzy_identical_tool_calls must be at least 1".into(),
            ));
        }
        if self.max_tool_calls_per_turn == Some(0) {
            return Err(Error::Config(
                "max_tool_calls_per_turn must be > 0 if set".into(),
            ));
        }
        if self.max_total_tokens == Some(0) {
            return Err(Error::Config("max_total_tokens must be at least 1".into()));
        }
        if self.response_cache_size == Some(0) {
            return Err(Error::Config(
                "response_cache_size must be at least 1".into(),
            ));
        }

        // Collect all tools, including memory and knowledge tools
        let mut all_tools = self.tools;
        let memory_scope = crate::auth::TenantScope::from_audit_fields(
            self.audit_tenant_id.as_deref(),
            self.audit_user_id.as_deref(),
        );
        let memory_ref = self.memory.clone();
        if let Some(memory) = self.memory {
            all_tools.extend(crate::memory::tools::memory_tools_with_reflection(
                memory,
                &self.name,
                memory_scope,
                self.reflection_threshold,
            ));
        }
        if let Some(kb) = self.knowledge_base {
            // SECURITY (F-KB-1): scope the KB tool to this runner's tenant.
            let kb_scope = crate::auth::TenantScope::from_audit_fields(
                self.audit_tenant_id.as_deref(),
                self.audit_user_id.as_deref(),
            );
            all_tools.extend(crate::knowledge::tools::knowledge_tools(kb, kb_scope));
        }
        if let Some(on_question) = self.on_question {
            all_tools.push(Arc::new(crate::tool::builtins::QuestionTool::new(
                on_question,
            )));
        }

        let mut tools: HashMap<String, Arc<dyn Tool>> = HashMap::with_capacity(all_tools.len());
        let mut tool_defs: Vec<ToolDefinition> = Vec::with_capacity(all_tools.len());

        for t in all_tools {
            let def = t.definition();
            if tools.contains_key(&def.name) {
                // SECURITY (F-MCP-2): elevate the log level — a duplicate tool
                // name is a potential MCP-shadowing attempt. The existing
                // first-wins behavior is preserved (so trusted builtins added
                // first take precedence) but the event is now auditable.
                tracing::error!(
                    tool = %def.name,
                    "duplicate tool name (potential MCP-shadowing); keeping first registration"
                );
                continue;
            }
            tool_defs.push(def.clone());
            tools.insert(def.name, t);
        }

        // Inject the synthetic __respond__ tool for structured output.
        // Only the ToolDefinition is added — there's no Tool impl because
        // the execute loop intercepts __respond__ calls before tool dispatch.
        if let Some(ref schema) = self.structured_schema {
            tool_defs.push(ToolDefinition {
                name: crate::llm::types::RESPOND_TOOL_NAME.into(),
                description: crate::llm::types::RESPOND_TOOL_DESCRIPTION.into(),
                input_schema: schema.clone(),
            });
        }

        // Prepend instruction text to the system prompt if provided.
        let mut system_prompt = match self.instruction_text {
            Some(ref text) => instructions::prepend_instructions(&self.system_prompt, text),
            None => self.system_prompt,
        };

        // Append workspace hint to the system prompt if configured.
        if let Some(ref ws) = self.workspace {
            system_prompt.push_str(&format!(
                "\n\nYour workspace directory is {}. You can freely create, organize, and manage \
                 files there. Use it for notes, intermediate results, generated artifacts, and \
                 anything you want to persist. Paths can be relative (resolved against workspace) \
                 or absolute.",
                ws.display()
            ));
        }

        // Append resourcefulness guidelines only when the agent has power tools
        // (bash, write, patch, edit) that make the guidance relevant. Saves ~180
        // tokens for conversational-only agents.
        let has_power_tools = tool_defs
            .iter()
            .any(|t| matches!(t.name.as_str(), "bash" | "write" | "patch" | "edit"));
        if has_power_tools {
            system_prompt.push_str(RESOURCEFULNESS_GUIDELINES);
        }

        // Inject current date/time so the model knows "today".
        use chrono::Utc;
        system_prompt.push_str(&format!(
            "\n\nCurrent date and time: {} UTC",
            Utc::now().format("%A, %B %-d, %Y %H:%M")
        ));

        // When restore-on-demand is enabled, tell the model it can fetch pruned results.
        if self.context_recall_store.is_some() {
            system_prompt.push_str(CONTEXT_RECALL_HINT);
        }

        Ok(AgentRunner {
            provider: self.provider,
            name: self.name,
            system_prompt,
            tools,
            tool_defs,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            context_strategy: self.context_strategy.unwrap_or(ContextStrategy::Unlimited),
            summarize_threshold: self.summarize_threshold,
            context_window_tokens: self.context_window_tokens,
            compaction_threshold_fraction: self.compaction_threshold_fraction,
            on_text: self.on_text,
            on_reasoning: self.on_reasoning,
            on_approval: self.on_approval,
            tool_timeout: self.tool_timeout,
            max_tool_output_bytes: self.max_tool_output_bytes,
            structured_schema: self.structured_schema,
            on_event: self.on_event,
            guardrails: self.guardrails,
            on_input: self.on_input,
            interrupt: self.interrupt,
            run_timeout: self.run_timeout,
            reasoning_effort: self.reasoning_effort,
            enable_reflection: self.enable_reflection,
            tool_output_compression_threshold: self.tool_output_compression_threshold,
            max_tools_per_turn: self.max_tools_per_turn,
            tool_profile: self.tool_profile,
            max_identical_tool_calls: self.max_identical_tool_calls,
            max_fuzzy_identical_tool_calls: self.max_fuzzy_identical_tool_calls,
            max_tool_calls_per_turn: self.max_tool_calls_per_turn,
            permission_rules: parking_lot::RwLock::new(self.permission_rules),
            learned_permissions: self.learned_permissions,
            lsp_manager: self.lsp_manager,
            session_prune_config: self.session_prune_config,
            memory: memory_ref,
            enable_recursive_summarization: self.enable_recursive_summarization,
            consolidate_on_exit: self.consolidate_on_exit,
            observability_mode: observability::ObservabilityMode::resolve(
                observability::OBSERVABILITY_ENV_KEY,
                None,
                self.observability_mode,
            ),
            max_total_tokens: self.max_total_tokens,
            goal: self.goal,
            audit_mode: self.audit_mode,
            audit_trail: self.audit_trail,
            audit_user_id: self.audit_user_id,
            audit_tenant_id: self.audit_tenant_id,
            audit_delegation_chain: self.audit_delegation_chain,
            response_cache: self.response_cache_size.map(cache::ResponseCache::new),
            tenant_tracker: self.tenant_tracker,
            context_recall_store: self.context_recall_store,
            cumulative_actual_tokens: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}
