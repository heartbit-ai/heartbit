use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{Instrument, debug, info_span};

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{
    CompletionRequest, ContentBlock, Message, StopReason, TokenUsage, ToolCall, ToolDefinition,
    ToolResult,
};
use crate::memory::Memory;
use crate::tool::{Tool, ToolOutput, validate_tool_input};
use crate::util::levenshtein;

use super::audit::{AuditRecord, AuditTrail};
use super::builder::AgentRunnerBuilder;
use super::cache;
use super::context::{AgentContext, ContextStrategy};
use super::doom_loop::DoomLoopTracker;
use super::events::{AgentEvent, EVENT_MAX_PAYLOAD_BYTES, OnEvent, truncate_for_event};
use super::guardrail::{GuardAction, Guardrail};
use super::observability;
use super::permission;
use super::pruner;
use super::tool_filter;

/// Callback for interactive mode. Called when the agent needs more user input
/// (i.e., the LLM returned text without tool calls). Returns `Some(message)`
/// to continue the conversation, or `None` to end the session.
pub type OnInput = dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
    + Send
    + Sync;

/// Behavioral guidelines appended to every agent's system prompt.
/// Ensures agents proactively discover capabilities and exhaust options
/// before claiming they cannot do something.
pub(crate) const RESOURCEFULNESS_GUIDELINES: &str = "\n\n\
## Resourcefulness\n\
Before claiming you cannot do something or lack access to a tool:\n\
- Use bash to check for installed CLIs (`which <tool>`, `command -v <tool>`).\n\
- Search for files, configs, and resources before saying they don't exist.\n\
- Read documentation, help output (`<tool> --help`), and man pages when unsure.\n\
- Try alternative approaches when the first attempt fails.\n\
Never say \"I don't have access\" or \"I can't\" without evidence. Investigate first.";

/// One tool execution record. Captures the full input + untruncated output
/// of a single tool call.
///
/// Populated by [`AgentRunner`] as tools execute; read via
/// [`AgentOutput::tool_call_results`]. The output here is the raw
/// post-guardrail content, BEFORE [`Tool::redact_for_history`] is applied
/// for conversation-history inclusion. Callers that need the original
/// (e.g., to extract a base64 image marker) should read this field rather
/// than rely on the agent's textual `result`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Name of the tool that was invoked.
    pub tool_name: String,
    /// Tool-call identifier issued by the LLM (used for tool-result pairing).
    pub tool_call_id: String,
    /// Raw input arguments passed to the tool.
    pub input: serde_json::Value,
    /// FULL, untruncated output content. May be large (base64 images,
    /// research dumps). Use [`Tool::redact_for_history`] to get the
    /// conversation-safe variant.
    pub output: String,
    /// Whether the tool produced an error.
    pub is_error: bool,
    /// Wall-clock duration of the tool's `execute` call.
    pub duration_ms: u64,
}

/// Output of a completed agent run.
///
/// Returned by [`AgentRunner::execute`] on success. Contains the agent's
/// final text response and usage accounting for the entire run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    /// The agent's final text response.
    pub result: String,
    /// Total number of tool calls made during the run.
    pub tool_calls_made: usize,
    /// Aggregate token usage for the entire run.
    pub tokens_used: TokenUsage,
    /// Structured output when the agent was configured with a response schema.
    /// Contains the validated JSON conforming to the schema.
    pub structured: Option<serde_json::Value>,
    /// Estimated cost in USD based on model pricing. `None` if the model is
    /// unknown or cost estimation is not available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_cost_usd: Option<f64>,
    /// The model name used for this run. For cascading providers, this is the
    /// last model that produced a response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Per-tool-call records with full input + untruncated output. Empty
    /// when no tools were called or for composite agents that don't track
    /// per-tool detail. Order matches dispatch order (which may differ
    /// from completion order due to parallel execution). Counts only
    /// tools that were actually executed — denied/blocked calls do not
    /// appear here.
    #[serde(default)]
    pub tool_call_results: Vec<ToolCallRecord>,
}

impl AgentOutput {
    /// Accumulate this output's usage, tool calls, and cost into running totals.
    pub(crate) fn accumulate_into(
        &self,
        total_usage: &mut TokenUsage,
        total_tool_calls: &mut usize,
        total_cost: &mut Option<f64>,
    ) {
        *total_usage += self.tokens_used;
        *total_tool_calls += self.tool_calls_made;
        if let Some(cost) = self.estimated_cost_usd {
            *total_cost.get_or_insert(0.0) += cost;
        }
    }
}

/// Runs an agent loop: LLM call → tool execution → repeat until done.
pub struct AgentRunner<P: LlmProvider> {
    pub(super) provider: Arc<P>,
    pub(super) name: String,
    pub(super) system_prompt: String,
    pub(super) tools: HashMap<String, Arc<dyn Tool>>,
    pub(super) tool_defs: Vec<ToolDefinition>,
    pub(super) max_turns: usize,
    pub(super) max_tokens: u32,
    pub(super) context_strategy: ContextStrategy,
    /// Token threshold at which to trigger summarization. `None` = no summarization.
    pub(super) summarize_threshold: Option<u32>,
    /// Optional callback for streaming text output.
    pub(super) on_text: Option<Arc<crate::llm::OnText>>,
    /// Optional callback for human-in-the-loop approval before tool execution.
    pub(super) on_approval: Option<Arc<crate::llm::OnApproval>>,
    /// Optional timeout for individual tool executions.
    pub(super) tool_timeout: Option<Duration>,
    /// Optional maximum byte size for tool output content. Oversized results
    /// are truncated with a `[truncated: N bytes omitted]` suffix.
    pub(super) max_tool_output_bytes: Option<usize>,
    /// When set, a synthetic `respond` tool is injected with this JSON Schema.
    /// The agent calls `respond` to produce structured output conforming to the schema.
    pub(super) structured_schema: Option<serde_json::Value>,
    /// Optional callback for structured agent events.
    pub(super) on_event: Option<Arc<OnEvent>>,
    /// Guardrails applied to LLM calls and tool executions.
    pub(super) guardrails: Vec<Arc<dyn Guardrail>>,
    /// Optional callback for interactive mode. When set and the LLM returns
    /// text without tool calls, the callback is invoked to get the next user
    /// message instead of returning immediately.
    pub(super) on_input: Option<Arc<OnInput>>,
    /// Optional wall-clock deadline for the entire run. When set, the full
    /// `execute` call (all turns) is wrapped in `tokio::time::timeout`.
    pub(super) run_timeout: Option<Duration>,
    /// Optional reasoning/thinking effort level for models that support it.
    pub(super) reasoning_effort: Option<crate::llm::types::ReasoningEffort>,
    /// When true, inject a reflection prompt after tool results to encourage
    /// the agent to assess results before the next action (Reflexion/CRITIC pattern).
    pub(super) enable_reflection: bool,
    /// When set, tool outputs exceeding this byte threshold are compressed
    /// via an LLM call that preserves factual content while removing redundancy.
    pub(super) tool_output_compression_threshold: Option<usize>,
    /// When set, limits the number of tool definitions sent per LLM turn.
    /// Tools are selected based on recent usage and keyword relevance.
    pub(super) max_tools_per_turn: Option<usize>,
    /// When set, pre-filters tool definitions based on query classification
    /// before dynamic selection. Reduces token usage for simple queries.
    pub(super) tool_profile: Option<tool_filter::ToolProfile>,
    /// Maximum number of consecutive identical tool-call turns before the
    /// agent receives an error result instead of executing the tools. `None`
    /// disables doom loop detection.
    pub(super) max_identical_tool_calls: Option<u32>,
    /// Maximum number of consecutive fuzzy-identical tool-call turns before
    /// doom loop detection triggers. Fuzzy matching compares sorted tool names
    /// (ignoring inputs). `None` disables fuzzy detection.
    pub(super) max_fuzzy_identical_tool_calls: Option<u32>,
    /// Hard cap on the number of tool invocations per LLM turn. When the LLM
    /// emits more tool_use blocks than this limit, the run fails with
    /// `Error::Agent` (wrapped in `Error::WithPartialUsage`). `None` = unlimited.
    pub(super) max_tool_calls_per_turn: Option<u32>,
    /// Declarative permission rules evaluated per tool call before the
    /// `on_approval` callback. `Allow` → execute, `Deny` → error result,
    /// `Ask` → fall through to `on_approval`.
    ///
    /// Wrapped in `RwLock` for interior mutability: learned rules from
    /// `AlwaysAllow`/`AlwaysDeny` are injected at runtime via `&self`.
    /// Lock is never held across `.await`.
    pub(super) permission_rules: parking_lot::RwLock<permission::PermissionRuleset>,
    /// Optional learned permissions for persisting AlwaysAllow/AlwaysDeny decisions.
    pub(super) learned_permissions: Option<Arc<std::sync::Mutex<permission::LearnedPermissions>>>,
    /// Optional LSP manager for collecting diagnostics after file-modifying tools.
    pub(super) lsp_manager: Option<Arc<crate::lsp::LspManager>>,
    /// Optional session pruning config. When set, old tool results are truncated
    /// before each LLM call to reduce token usage.
    pub(super) session_prune_config: Option<pruner::SessionPruneConfig>,
    /// Optional memory store reference for pre-compaction flush.
    pub(super) memory: Option<Arc<dyn Memory>>,
    /// When true, use recursive (cluster-then-summarize) summarization for
    /// long conversations instead of single-shot.
    pub(super) enable_recursive_summarization: bool,
    /// When true, run memory consolidation at session end.
    pub(super) consolidate_on_exit: bool,
    /// Observability verbosity level controlling span attribute recording.
    pub(super) observability_mode: observability::ObservabilityMode,
    /// Hard limit on cumulative tokens (input + output) across all turns.
    /// When exceeded, the agent returns `Error::BudgetExceeded`.
    pub(super) max_total_tokens: Option<u64>,
    /// Controls whether audit records include full content or metadata only.
    pub(super) audit_mode: super::audit::AuditMode,
    /// Optional audit trail for recording untruncated agent decisions.
    pub(super) audit_trail: Option<Arc<dyn AuditTrail>>,
    /// Optional user context for multi-tenant audit enrichment.
    pub(super) audit_user_id: Option<String>,
    pub(super) audit_tenant_id: Option<String>,
    /// Delegation chain for audit records (e.g., `["heartbit-agent"]` when acting on behalf of user).
    pub(super) audit_delegation_chain: Vec<String>,
    /// Optional LRU cache for LLM completion responses. Skips the LLM call
    /// when an identical request (system prompt + messages + tool names) is found.
    pub(super) response_cache: Option<cache::ResponseCache>,
    /// Optional per-tenant in-flight token tracker. When set, `adjust()` is called
    /// after each LLM response to reconcile actual vs. estimated usage.
    pub(super) tenant_tracker: Option<Arc<crate::agent::tenant_tracker::TenantTokenTracker>>,
    /// Cumulative actual tokens (input + output) across all turns for this runner.
    /// Used to compute signed deltas for `tenant_tracker.adjust()` and to release
    /// the full amount on `Drop`.
    pub(super) cumulative_actual_tokens: std::sync::atomic::AtomicUsize,
}

impl<P: LlmProvider> AgentRunner<P> {
    /// Create a new [`AgentRunnerBuilder`] for an agent backed by `provider`.
    ///
    /// The builder uses sensible defaults (10 turns, 4096 tokens) so the
    /// minimum required configuration is just a system prompt.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::sync::Arc;
    /// use heartbit_core::{AgentRunner, AnthropicProvider, BoxedProvider};
    ///
    /// # async fn run() -> Result<(), heartbit_core::Error> {
    /// let provider = Arc::new(BoxedProvider::new(AnthropicProvider::new(
    ///     "sk-...",
    ///     "claude-sonnet-4-20250514",
    /// )));
    /// let agent = AgentRunner::builder(provider)
    ///     .system_prompt("You are a helpful assistant.")
    ///     .build()?;
    /// # let _ = agent;
    /// # Ok(()) }
    /// ```
    pub fn builder(provider: Arc<P>) -> AgentRunnerBuilder<P> {
        AgentRunnerBuilder {
            provider,
            name: "agent".into(),
            system_prompt: String::new(),
            tools: Vec::new(),
            max_turns: 10,
            max_tokens: 4096,
            context_strategy: None,
            summarize_threshold: None,
            memory: None,
            knowledge_base: None,
            on_text: None,
            on_approval: None,
            tool_timeout: None,
            max_tool_output_bytes: None,
            structured_schema: None,
            on_event: None,
            guardrails: Vec::new(),
            on_question: None,
            on_input: None,
            run_timeout: None,
            reasoning_effort: None,
            enable_reflection: false,
            tool_output_compression_threshold: None,
            max_tools_per_turn: None,
            tool_profile: None,
            max_identical_tool_calls: None,
            max_fuzzy_identical_tool_calls: None,
            max_tool_calls_per_turn: None,
            permission_rules: permission::PermissionRuleset::default(),
            instruction_text: None,
            learned_permissions: None,
            lsp_manager: None,
            session_prune_config: None,
            enable_recursive_summarization: false,
            reflection_threshold: None,
            consolidate_on_exit: false,
            observability_mode: None,
            workspace: None,
            max_total_tokens: None,
            audit_mode: super::audit::AuditMode::Full,
            audit_trail: None,
            audit_user_id: None,
            audit_tenant_id: None,
            audit_delegation_chain: Vec::new(),
            response_cache_size: None,
            tenant_tracker: None,
        }
    }

    /// Returns the agent's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Read-access to the permission rules (acquires read lock).
    fn eval_permission(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
    ) -> Option<permission::PermissionAction> {
        self.permission_rules.read().evaluate(tool_name, input)
    }

    /// Check if the permission ruleset has any rules.
    fn has_permission_rules(&self) -> bool {
        !self.permission_rules.read().is_empty()
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(ref cb) = self.on_event {
            cb(event);
        }
    }

    /// Record an audit entry (best-effort). Failures are logged, never abort the agent.
    async fn audit(&self, mut record: AuditRecord) {
        if let Some(ref trail) = self.audit_trail {
            if self.audit_mode == super::audit::AuditMode::MetadataOnly {
                // Owned variant skips the top-level + per-scalar clones
                // (P-CROSS-7) — ~1 ms saved per record on 100 KB payloads.
                let payload = std::mem::take(&mut record.payload);
                record.payload = super::audit::strip_content_owned(payload);
            }
            if let Err(e) = trail.record(record).await {
                tracing::warn!(error = %e, "audit record failed");
            }
        }
    }

    /// Persist an AlwaysAllow/AlwaysDeny decision as a learned permission rule.
    ///
    /// For each distinct tool name in the tool calls, a tool-level rule is created
    /// (`pattern: "*"`). The rule is added to both the in-memory ruleset and the
    /// on-disk learned permissions file.
    fn persist_approval_decision(
        &self,
        tool_calls: &[ToolCall],
        decision: crate::llm::ApprovalDecision,
    ) {
        let action = if decision.is_allowed() {
            permission::PermissionAction::Allow
        } else {
            permission::PermissionAction::Deny
        };
        // Collect distinct tool names
        let mut seen = std::collections::HashSet::new();
        let mut new_rules = Vec::new();
        for tc in tool_calls {
            if seen.insert(tc.name.clone()) {
                new_rules.push(permission::PermissionRule {
                    tool: tc.name.clone(),
                    pattern: "*".into(),
                    action,
                });
            }
        }
        // Inject into the live ruleset so the rule takes effect immediately
        // within this session (not just after restart).
        self.permission_rules.write().append_rules(&new_rules);
        // Persist to disk if learned permissions are configured
        if let Some(ref learned) = self.learned_permissions {
            for rule in new_rules {
                if let Ok(mut guard) = learned.lock()
                    && let Err(e) = guard.add_rule(rule)
                {
                    tracing::warn!(
                        error = %e,
                        "failed to persist learned permission rule"
                    );
                }
            }
        }
    }

    /// Estimate cost in USD based on model pricing and accumulated token usage.
    fn estimate_cost(&self, usage: &TokenUsage) -> Option<f64> {
        self.provider
            .model_name()
            .and_then(|model| crate::llm::pricing::estimate_cost(model, usage))
    }

    /// Run the agent on `task` and return the final output.
    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let ctx = AgentContext::new(&self.system_prompt, task, self.tool_defs.clone())
            .with_max_turns(self.max_turns)
            .with_max_tokens(self.max_tokens)
            .with_context_strategy(self.context_strategy.clone())
            .with_reasoning_effort(self.reasoning_effort);
        self.execute_with_context(ctx, task).await
    }

    /// Execute with pre-built multimodal content blocks (e.g., text + images).
    pub async fn execute_with_content(
        &self,
        content: Vec<ContentBlock>,
    ) -> Result<AgentOutput, Error> {
        // Extract text for event/span descriptions
        let task_summary: String = content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");

        let ctx = AgentContext::from_content(&self.system_prompt, content, self.tool_defs.clone())
            .with_max_turns(self.max_turns)
            .with_max_tokens(self.max_tokens)
            .with_context_strategy(self.context_strategy.clone())
            .with_reasoning_effort(self.reasoning_effort);
        self.execute_with_context(ctx, &task_summary).await
    }

    async fn execute_with_context(
        &self,
        ctx: AgentContext,
        task_description: &str,
    ) -> Result<AgentOutput, Error> {
        // Shared accumulator so we can retrieve partial usage even when the
        // future is dropped by tokio::time::timeout.
        let usage_acc = Arc::new(std::sync::Mutex::new(TokenUsage::default()));
        let fut = {
            let acc = usage_acc.clone();
            async move {
                match self.execute_inner(ctx, task_description, acc).await {
                    Ok(output) => Ok(output),
                    Err((e, usage)) => Err(e.with_partial_usage(usage)),
                }
            }
        };
        let mut result = match self.run_timeout {
            Some(timeout) => match tokio::time::timeout(timeout, fut).await {
                Ok(result) => result,
                Err(_) => {
                    let usage = *usage_acc.lock().expect("usage lock poisoned");
                    Err(Error::RunTimeout(timeout).with_partial_usage(usage))
                }
            },
            None => fut.await,
        };

        // Audit: run failed
        if let Err(ref e) = result {
            self.audit(AuditRecord {
                agent: self.name.clone(),
                turn: 0,
                event_type: "run_failed".into(),
                payload: serde_json::json!({
                    "error": e.to_string(),
                }),
                usage: e.partial_usage(),
                timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
            })
            .await;
        }

        // Session-end maintenance (best-effort, errors logged but not propagated).
        if let Ok(ref mut output) = result {
            // Consolidate related episodic memories into semantic summaries (opt-in).
            let consolidation_usage = self.consolidate_memory_on_exit().await;
            if consolidation_usage.input_tokens > 0 || consolidation_usage.output_tokens > 0 {
                output.tokens_used += consolidation_usage;
                // Add consolidation cost increment (uses static model name — consolidation
                // always runs through the same provider, not cascade tiers).
                if let Some(consolidation_cost) = self.estimate_cost(&consolidation_usage) {
                    output.estimated_cost_usd =
                        Some(output.estimated_cost_usd.unwrap_or(0.0) + consolidation_cost);
                }
            }

            // Prune weak/old memories.
            self.prune_memory_on_exit().await;
        }

        result
    }

    async fn execute_inner(
        &self,
        initial_ctx: AgentContext,
        task: &str,
        usage_acc: Arc<std::sync::Mutex<TokenUsage>>,
    ) -> Result<AgentOutput, (Error, TokenUsage)> {
        let mode = self.observability_mode;
        let run_span = info_span!(
            "heartbit.agent.run",
            agent = %self.name,
            max_turns = self.max_turns,
            task = tracing::field::Empty,
            model = tracing::field::Empty,
            total_input_tokens = tracing::field::Empty,
            total_output_tokens = tracing::field::Empty,
            estimated_cost_usd = tracing::field::Empty,
        );
        if mode.includes_metrics()
            && let Some(model) = self.provider.model_name()
        {
            run_span.record("model", model);
        }
        if mode.includes_payloads() {
            run_span.record(
                "task",
                truncate_for_event(task, EVENT_MAX_PAYLOAD_BYTES).as_str(),
            );
        } else if mode.includes_metrics() {
            let cut = crate::tool::builtins::floor_char_boundary(task, 256);
            run_span.record("task", &task[..cut]);
        }

        let result = async {
            self.emit(AgentEvent::RunStarted {
                agent: self.name.clone(),
                task: task.to_string(),
            });

            let mut ctx = initial_ctx;

            let mut total_tool_calls = 0usize;
            // P1.3g: per-tool-call records (full input/output) accumulated
            // across all turns; moved into AgentOutput.tool_call_results on
            // run completion. Only includes tools that actually executed —
            // not denied/permission-rejected calls.
            let mut tool_call_records: Vec<ToolCallRecord> = Vec::new();
            let mut total_usage = TokenUsage::default();
            // Accumulate cost per-turn for accurate cascade pricing.
            let mut total_cost: f64 = 0.0;
            // Track recently used tool names (last 2 turns) for dynamic tool selection
            let mut recently_used_tools: Vec<String> = Vec::new();
            let mut doom_tracker = DoomLoopTracker::new();
            let mut last_model_name: Option<String> = None;
            // Prevents infinite compaction loops: set true after compaction,
            // cleared at the start of each normal iteration.
            let mut compacted_last_turn = false;

            loop {
                if ctx.current_turn() >= ctx.max_turns() {
                    self.emit(AgentEvent::RunFailed {
                        agent: self.name.clone(),
                        error: format!("Max turns ({}) exceeded", ctx.max_turns()),
                        partial_usage: total_usage,
                    });
                    return Err((Error::MaxTurnsExceeded(ctx.max_turns()), total_usage));
                }

                ctx.increment_turn();
                let can_compact = !compacted_last_turn;
                compacted_last_turn = false;
                debug!(agent = %self.name, turn = ctx.current_turn(), "executing turn");
                self.emit(AgentEvent::TurnStarted {
                    agent: self.name.clone(),
                    turn: ctx.current_turn(),
                    max_turns: ctx.max_turns(),
                });

                // Provide turn context to stateful guardrails
                for g in &self.guardrails {
                    g.set_turn(ctx.current_turn());
                }

                // Session pruning: create a pruned view of messages for this LLM call
                let mut request = if let Some(ref prune_config) = self.session_prune_config {
                    let mut req = ctx.to_request();
                    let (pruned_msgs, prune_stats) =
                        pruner::prune_old_tool_results(&req.messages, prune_config);
                    req.messages = pruned_msgs;
                    if prune_stats.did_prune() {
                        debug!(
                            agent = %self.name,
                            turn = ctx.current_turn(),
                            pruned = prune_stats.tool_results_pruned,
                            total = prune_stats.tool_results_total,
                            bytes_saved = prune_stats.bytes_saved,
                            "session pruning applied"
                        );
                        self.emit(AgentEvent::SessionPruned {
                            agent: self.name.clone(),
                            turn: ctx.current_turn(),
                            tool_results_pruned: prune_stats.tool_results_pruned,
                            bytes_saved: prune_stats.bytes_saved,
                            tool_results_total: prune_stats.tool_results_total,
                        });
                    }
                    req
                } else {
                    ctx.to_request()
                };

                // Tool profile pre-filter: narrow tool set based on query classification
                if let Some(profile) = self.tool_profile {
                    request.tools = tool_filter::filter_tools(&request.tools, profile);
                }

                // Dynamic tool selection: filter tools when there are too many
                if let Some(max_tools) = self.max_tools_per_turn {
                    request.tools = self.select_tools_for_turn(
                        &request.tools,
                        &request.messages,
                        &recently_used_tools,
                        max_tools,
                    );
                }

                for g in &self.guardrails {
                    if let Err(e) = g.pre_llm(&mut request).await {
                        self.emit(AgentEvent::RunFailed {
                            agent: self.name.clone(),
                            error: e.to_string(),
                            partial_usage: total_usage,
                        });
                        return Err((e, total_usage));
                    }
                }
                // Response cache: compute key for non-streaming requests.
                // SECURITY (F-AGENT-3): scope the cache by tenant_id+user_id
                // when known. Otherwise a runner shared across tenants could
                // serve tenant A's cached response to tenant B if their
                // (system_prompt, messages, tools) tuple coincides.
                let cache_key = if self.response_cache.is_some() && self.on_text.is_none() {
                    let tool_names: Vec<&str> =
                        request.tools.iter().map(|t| t.name.as_str()).collect();
                    let namespace = match (&self.audit_tenant_id, &self.audit_user_id) {
                        (Some(t), Some(u)) => Some(format!("{t}:{u}")),
                        (Some(t), None) => Some(t.clone()),
                        (None, Some(u)) => Some(format!(":{u}")),
                        (None, None) => None,
                    };
                    Some(cache::ResponseCache::compute_key_scoped(
                        &request.system,
                        &request.messages,
                        &tool_names,
                        namespace.as_deref(),
                    ))
                } else {
                    None
                };
                // Check cache before calling LLM
                let cache_hit = cache_key
                    .and_then(|k| self.response_cache.as_ref().and_then(|c| c.get(k)));
                let llm_start = Instant::now();
                let llm_span = info_span!(
                    "heartbit.agent.llm_call",
                    agent = %self.name,
                    turn = ctx.current_turn(),
                    { observability::GEN_AI_REQUEST_MODEL } = tracing::field::Empty,
                    latency_ms = tracing::field::Empty,
                    { observability::GEN_AI_USAGE_INPUT_TOKENS } = tracing::field::Empty,
                    { observability::GEN_AI_USAGE_OUTPUT_TOKENS } = tracing::field::Empty,
                    { observability::GEN_AI_RESPONSE_FINISH_REASON } = tracing::field::Empty,
                    tool_call_count = tracing::field::Empty,
                    ttft_ms = tracing::field::Empty,
                    response_text = tracing::field::Empty,
                    cache_hit = tracing::field::Empty,
                );
                let llm_result = if let Some(cached) = cache_hit {
                    tracing::debug!(
                        agent = %self.name,
                        turn = ctx.current_turn(),
                        "response cache hit, skipping LLM call"
                    );
                    if mode.includes_metrics() {
                        llm_span.record("cache_hit", true);
                    }
                    Ok(cached)
                } else {
                    // TTFT: wrap on_text to capture time-to-first-token
                    let ttft_ms_inner = Arc::new(std::sync::atomic::AtomicU64::new(0));
                    let ttft_ref = ttft_ms_inner.clone();
                    let result = async {
                        match &self.on_text {
                            Some(cb) => {
                                let ttft_ref = ttft_ref.clone();
                                let start = llm_start;
                                let inner_cb = cb.clone();
                                let wrapper: Box<crate::llm::OnText> =
                                    Box::new(move |text: &str| {
                                        ttft_ref
                                            .compare_exchange(
                                                0,
                                                start.elapsed().as_millis() as u64,
                                                std::sync::atomic::Ordering::Relaxed,
                                                std::sync::atomic::Ordering::Relaxed,
                                            )
                                            .ok();
                                        inner_cb(text);
                                    });
                                self.provider.stream_complete(request, &*wrapper).await
                            }
                            None => self.provider.complete(request).await,
                        }
                    }
                    .instrument(llm_span.clone())
                    .await;
                    // Store successful non-streaming responses in cache.
                    // Only cache EndTurn responses — ToolUse responses trigger
                    // side-effecting tool execution and must not be replayed.
                    if let (Ok(resp), Some(key)) = (&result, cache_key)
                        && resp.stop_reason == crate::llm::types::StopReason::EndTurn
                        && let Some(ref c) = self.response_cache
                    {
                        c.put(key, resp.clone());
                    }
                    if mode.includes_metrics() {
                        let ttft = ttft_ms_inner.load(std::sync::atomic::Ordering::Relaxed);
                        llm_span.record("ttft_ms", ttft);
                        llm_span.record("cache_hit", false);
                    }
                    result
                };
                let llm_latency_ms = llm_start.elapsed().as_millis() as u64;
                // Record LLM call span attributes
                if mode.includes_metrics() {
                    llm_span.record("latency_ms", llm_latency_ms);
                    if let Ok(ref r) = llm_result {
                        if let Some(ref model) = r.model {
                            llm_span.record(observability::GEN_AI_REQUEST_MODEL, model.as_str());
                        } else if let Some(model) = self.provider.model_name() {
                            llm_span.record(observability::GEN_AI_REQUEST_MODEL, model);
                        }
                    } else if let Some(model) = self.provider.model_name() {
                        llm_span.record(observability::GEN_AI_REQUEST_MODEL, model);
                    }
                    if let Ok(ref r) = llm_result {
                        llm_span.record(
                            observability::GEN_AI_USAGE_INPUT_TOKENS,
                            r.usage.input_tokens,
                        );
                        llm_span.record(
                            observability::GEN_AI_USAGE_OUTPUT_TOKENS,
                            r.usage.output_tokens,
                        );
                        llm_span.record(
                            observability::GEN_AI_RESPONSE_FINISH_REASON,
                            format!("{:?}", r.stop_reason).as_str(),
                        );
                        llm_span.record("tool_call_count", r.tool_calls().len());
                    }
                }
                if mode.includes_payloads()
                    && let Ok(ref r) = llm_result
                {
                    llm_span.record(
                        "response_text",
                        truncate_for_event(&r.text(), EVENT_MAX_PAYLOAD_BYTES).as_str(),
                    );
                }
                let mut response = match llm_result {
                    Ok(r) => r,
                    Err(e) => {
                        // Auto-compaction: on context overflow, summarize and retry
                        if crate::llm::error_class::classify(&e)
                            == crate::llm::error_class::ErrorClass::ContextOverflow
                            && can_compact
                            && ctx.message_count() > 5
                        {
                            tracing::warn!(
                                agent = %self.name,
                                error = %e,
                                "context overflow detected, attempting auto-compaction"
                            );
                            match self.generate_summary(&ctx).await {
                                Ok((Some(summary), summary_usage)) => {
                                    total_usage += summary_usage;
                                    if let Some(c) = self.estimate_cost(&summary_usage) {
                                        total_cost += c;
                                    }
                                    *usage_acc.lock().expect("usage lock poisoned") = total_usage;
                                    self.flush_to_memory_before_compaction(&ctx, 4).await;
                                    ctx.inject_summary(summary, 4);
                                    self.emit(AgentEvent::AutoCompactionTriggered {
                                        agent: self.name.clone(),
                                        turn: ctx.current_turn(),
                                        success: true,
                                        usage: summary_usage,
                                    });
                                    self.emit(AgentEvent::ContextSummarized {
                                        agent: self.name.clone(),
                                        turn: ctx.current_turn(),
                                        usage: summary_usage,
                                    });
                                    compacted_last_turn = true;
                                    continue;
                                }
                                Ok((None, summary_usage)) => {
                                    total_usage += summary_usage;
                                    *usage_acc.lock().expect("usage lock poisoned") = total_usage;
                                    self.emit(AgentEvent::AutoCompactionTriggered {
                                        agent: self.name.clone(),
                                        turn: ctx.current_turn(),
                                        success: false,
                                        usage: summary_usage,
                                    });
                                    tracing::warn!(
                                        agent = %self.name,
                                        "auto-compaction summary was truncated, cannot compact"
                                    );
                                }
                                Err(summary_err) => {
                                    self.emit(AgentEvent::AutoCompactionTriggered {
                                        agent: self.name.clone(),
                                        turn: ctx.current_turn(),
                                        success: false,
                                        usage: TokenUsage::default(),
                                    });
                                    tracing::warn!(
                                        agent = %self.name,
                                        error = %summary_err,
                                        "auto-compaction summary failed"
                                    );
                                }
                            }
                        }
                        self.emit(AgentEvent::RunFailed {
                            agent: self.name.clone(),
                            error: e.to_string(),
                            partial_usage: total_usage,
                        });
                        return Err((e, total_usage));
                    }
                };
                total_usage += response.usage;

                // Reconcile per-tenant in-flight token estimate with actual usage.
                // Uses cumulative `total_usage` (not per-turn) so the tracker always
                // reflects the true running total and multi-turn deltas are correct.
                if let (Some(tracker), Some(tid)) =
                    (&self.tenant_tracker, &self.audit_tenant_id)
                {
                    let actual =
                        (total_usage.input_tokens + total_usage.output_tokens) as usize;
                    let prev = self
                        .cumulative_actual_tokens
                        .swap(actual, std::sync::atomic::Ordering::SeqCst);
                    let delta = actual as i64 - prev as i64;
                    let scope = crate::auth::TenantScope::new(tid.clone());
                    tracker.adjust(&scope, delta);
                }

                // Per-turn cost: prefer response.model (cascade) over static model_name()
                let turn_model = response
                    .model
                    .as_deref()
                    .or_else(|| self.provider.model_name());
                if let Some(model) = turn_model {
                    last_model_name = Some(model.to_string());
                    if let Some(cost) =
                        crate::llm::pricing::estimate_cost(model, &response.usage)
                    {
                        total_cost += cost;
                    }
                }
                // Update shared accumulator so RunTimeout can retrieve partial usage
                *usage_acc.lock().expect("usage lock poisoned") = total_usage;

                // Check token budget
                if let Some(max) = self.max_total_tokens {
                    let used = total_usage.total();
                    if used > max {
                        self.emit(AgentEvent::BudgetExceeded {
                            agent: self.name.clone(),
                            used,
                            limit: max,
                            partial_usage: total_usage,
                        });
                        return Err((
                            Error::BudgetExceeded { used, limit: max },
                            total_usage,
                        ));
                    }
                }

                let mut tool_calls = response.tool_calls();

                // SECURITY (F-AGENT-1): repair Levenshtein-close typos in tool names
                // BEFORE permissions and pre_tool guardrails see them. Otherwise an
                // LLM could emit `bask` to bypass a `bash` deny-rule and have it
                // silently dispatched to `bash` later. We mutate `call.name` here
                // and emit a `ToolNameRepaired` event so the audit trail records
                // the substitution. The repair only fires for unknown names; exact
                // matches are untouched.
                for call in tool_calls.iter_mut() {
                    if !self.tools.contains_key(&call.name)
                        && let Some(repaired) = self.find_closest_tool(&call.name, 2)
                    {
                        let repaired = repaired.to_string();
                        tracing::warn!(
                            agent = %self.name,
                            original = %call.name,
                            repaired = %repaired,
                            "tool name repaired via Levenshtein match (pre-policy)"
                        );
                        self.emit(AgentEvent::ToolNameRepaired {
                            agent: self.name.clone(),
                            original: call.name.clone(),
                            repaired: repaired.clone(),
                        });
                        call.name = repaired;
                    }
                }

                // Tool-call cap: reject turns that exceed max_tool_calls_per_turn.
                // Checked before dispatch so no tools are executed on a capped turn.
                if let Some(cap) = self.max_tool_calls_per_turn
                    && tool_calls.len() as u32 > cap
                {
                    let err = Error::Agent(format!(
                        "tool-call cap exceeded: turn produced {} calls, max is {cap}",
                        tool_calls.len()
                    ));
                    self.emit(AgentEvent::RunFailed {
                        agent: self.name.clone(),
                        error: err.to_string(),
                        partial_usage: total_usage,
                    });
                    return Err((err, total_usage));
                }

                self.emit(AgentEvent::LlmResponse {
                    agent: self.name.clone(),
                    turn: ctx.current_turn(),
                    usage: response.usage,
                    stop_reason: response.stop_reason,
                    tool_call_count: tool_calls.len(),
                    text: truncate_for_event(&response.text(), EVENT_MAX_PAYLOAD_BYTES),
                    latency_ms: llm_latency_ms,
                    model: response
                        .model
                        .clone()
                        .or_else(|| self.provider.model_name().map(|s| s.to_string())),
                    time_to_first_token_ms: 0,
                });

                // Audit: LLM response (untruncated)
                self.audit(AuditRecord {
                    agent: self.name.clone(),
                    turn: ctx.current_turn(),
                    event_type: "llm_response".into(),
                    payload: serde_json::json!({
                        "text": response.text(),
                        "stop_reason": format!("{:?}", response.stop_reason),
                        "tool_call_count": tool_calls.len(),
                        "latency_ms": llm_latency_ms,
                        "model": response.model.as_deref()
                            .or_else(|| self.provider.model_name()),
                    }),
                    usage: response.usage,
                    timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
                })
                .await;

                // post_llm guardrail: inspect response, first Deny discards it.
                // When denied, we insert a synthetic assistant message before the
                // denial feedback to maintain the alternating user/assistant message
                // invariant required by the Anthropic API.
                let mut post_llm_denied = false;
                for g in &self.guardrails {
                    match g
                        .post_llm(&mut response)
                        .await
                        .map_err(|e| (e, total_usage))?
                    {
                        GuardAction::Allow => {}
                        GuardAction::Warn { reason } => {
                            self.emit(AgentEvent::GuardrailWarned {
                                agent: self.name.clone(),
                                hook: "post_llm".into(),
                                reason: reason.clone(),
                                tool_name: None,
                            });
                            self.audit(AuditRecord {
                                agent: self.name.clone(),
                                turn: ctx.current_turn(),
                                event_type: "guardrail_warned".into(),
                                payload: serde_json::json!({
                                    "hook": "post_llm",
                                    "reason": reason,
                                }),
                                usage: TokenUsage::default(),
                                timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
                            })
                            .await;
                            // Continue — do NOT discard the response
                        }
                        GuardAction::Deny { reason } => {
                            self.emit(AgentEvent::GuardrailDenied {
                                agent: self.name.clone(),
                                hook: "post_llm".into(),
                                reason: reason.clone(),
                                tool_name: None,
                            });
                            // Audit: guardrail denied
                            self.audit(AuditRecord {
                                agent: self.name.clone(),
                                turn: ctx.current_turn(),
                                event_type: "guardrail_denied".into(),
                                payload: serde_json::json!({
                                    "hook": "post_llm",
                                    "reason": reason,
                                }),
                                usage: TokenUsage::default(),
                                timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
                            })
                            .await;
                            // Maintain alternating roles: assistant placeholder, then user denial
                            ctx.add_assistant_message(Message {
                                role: crate::llm::types::Role::Assistant,
                                content: vec![ContentBlock::Text {
                                    text: "[Response denied by guardrail]".into(),
                                }],
                            });
                            ctx.add_user_message(format!(
                            "[Guardrail denied your previous response: {reason}. Please try again.]"
                        ));
                            post_llm_denied = true;
                            break;
                        }
                        GuardAction::Kill { reason } => {
                            self.emit(AgentEvent::KillSwitchActivated {
                                agent: self.name.clone(),
                                reason: reason.clone(),
                                guardrail_name: g.name().to_string(),
                            });
                            self.audit(AuditRecord {
                                agent: self.name.clone(),
                                turn: ctx.current_turn(),
                                event_type: "guardrail_killed".into(),
                                payload: serde_json::json!({
                                    "hook": "post_llm",
                                    "reason": reason,
                                }),
                                usage: TokenUsage::default(),
                                timestamp: chrono::Utc::now(),
                                user_id: self.audit_user_id.clone(),
                                tenant_id: self.audit_tenant_id.clone(),
                                delegation_chain: self.audit_delegation_chain.clone(),
                            })
                            .await;
                            return Err((
                                Error::KillSwitch(reason),
                                total_usage,
                            ));
                        }
                    }
                }
                if post_llm_denied {
                    continue;
                }

                // Add assistant message to context (move content, avoid clone)
                ctx.add_assistant_message(Message {
                    role: crate::llm::types::Role::Assistant,
                    content: response.content,
                });

                // Evict base64 media from older messages to prevent context bloat.
                ctx.evict_media();

                // Check for structured output: if the LLM called the synthetic `__respond__` tool,
                // validate its input against the schema, then extract as structured result.
                // Count ALL tool calls in this turn (including co-submitted ones) for parity
                // with the Restate path, even though non-__respond__ calls are not executed.
                if let Some(ref schema) = self.structured_schema
                    && let Some(respond_call) = tool_calls
                        .iter()
                        .find(|tc| tc.name == crate::llm::types::RESPOND_TOOL_NAME)
                {
                    let structured = respond_call.input.clone();

                    // Validate against the caller's schema before accepting.
                    if let Err(validation_error) =
                        crate::tool::validate_tool_input(schema, &structured)
                    {
                        // Count the failed attempt and feed the validation error
                        // back to the LLM so it can self-correct on the next turn.
                        total_tool_calls += tool_calls.len();
                        tracing::warn!(
                            agent = %self.name,
                            error = %validation_error,
                            "structured output failed schema validation, retrying"
                        );
                        ctx.add_tool_results(vec![ToolResult {
                            tool_use_id: respond_call.id.clone(),
                            content: format!(
                                "Structured output validation failed: {validation_error}. \
                                 Please fix the output to match the schema and call __respond__ again."
                            ),
                            is_error: true,
                        }]);
                        continue;
                    }

                    total_tool_calls += tool_calls.len();
                    let text = serde_json::to_string_pretty(&structured)
                        .unwrap_or_else(|_| structured.to_string());
                    self.emit(AgentEvent::RunCompleted {
                        agent: self.name.clone(),
                        total_usage,
                        tool_calls_made: total_tool_calls,
                    });
                    // Audit: run completed (structured)
                    let preview_end =
                        crate::tool::builtins::floor_char_boundary(&text, 1000);
                    self.audit(AuditRecord {
                        agent: self.name.clone(),
                        turn: ctx.current_turn(),
                        event_type: "run_completed".into(),
                        payload: serde_json::json!({
                            "total_tool_calls": total_tool_calls,
                            "result_preview": &text[..preview_end],
                        }),
                        usage: total_usage,
                        timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
                    })
                    .await;
                    return Ok(AgentOutput {
                        result: text,
                        tool_calls_made: total_tool_calls,
                        tokens_used: total_usage,
                        structured: Some(structured),
                        estimated_cost_usd: if total_cost > 0.0 {
                            Some(total_cost)
                        } else {
                            self.estimate_cost(&total_usage)
                        },
                        model_name: last_model_name.clone(),
                        tool_call_results: std::mem::take(&mut tool_call_records),
                    });
                }

                if tool_calls.is_empty() {
                    // Check for truncation
                    if response.stop_reason == StopReason::MaxTokens {
                        self.emit(AgentEvent::RunFailed {
                            agent: self.name.clone(),
                            error: "Response truncated (max_tokens reached)".into(),
                            partial_usage: total_usage,
                        });
                        return Err((Error::Truncated, total_usage));
                    }

                    // Structured output was requested but LLM returned text without
                    // calling __respond__. This is a contract violation — the caller
                    // expects structured output but would get None silently.
                    if self.structured_schema.is_some() {
                        self.emit(AgentEvent::RunFailed {
                            agent: self.name.clone(),
                            error: "LLM returned text without calling __respond__".into(),
                            partial_usage: total_usage,
                        });
                        return Err((
                            Error::Agent(
                                "LLM returned text without calling __respond__; \
                             structured output was not produced"
                                    .into(),
                            ),
                            total_usage,
                        ));
                    }

                    // Interactive mode: if on_input is set, ask for more input
                    // instead of returning. This enables multi-turn conversations.
                    if let Some(ref on_input) = self.on_input
                        && let Some(next_message) = on_input().await
                        && !next_message.trim().is_empty()
                    {
                        ctx.add_user_message(next_message);
                        continue;
                    }

                    self.emit(AgentEvent::RunCompleted {
                        agent: self.name.clone(),
                        total_usage,
                        tool_calls_made: total_tool_calls,
                    });
                    let result_text =
                        ctx.last_assistant_text().unwrap_or_default().to_string();
                    // Audit: run completed
                    let preview_end =
                        crate::tool::builtins::floor_char_boundary(&result_text, 1000);
                    self.audit(AuditRecord {
                        agent: self.name.clone(),
                        turn: ctx.current_turn(),
                        event_type: "run_completed".into(),
                        payload: serde_json::json!({
                            "total_tool_calls": total_tool_calls,
                            "result_preview": &result_text[..preview_end],
                        }),
                        usage: total_usage,
                        timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
                    })
                    .await;
                    return Ok(AgentOutput {
                        result: result_text,
                        tool_calls_made: total_tool_calls,
                        tokens_used: total_usage,
                        structured: None,
                        estimated_cost_usd: if total_cost > 0.0 {
                            Some(total_cost)
                        } else {
                            self.estimate_cost(&total_usage)
                        },
                        model_name: last_model_name.clone(),
                        tool_call_results: std::mem::take(&mut tool_call_records),
                    });
                }

                // Permission rules + human-in-the-loop approval.
                //
                // When permission rules are set, each call is evaluated individually:
                //   Allow → execute without asking
                //   Deny  → error result
                //   Ask   → deferred to `on_approval` callback
                // Calls with no matching rule are also deferred to `on_approval`.
                //
                // When no rules are set, the legacy behavior applies: if `on_approval`
                // is set, the entire batch is sent for approval.
                let (tool_calls, permission_denied_results) = if self.has_permission_rules() {
                    let mut allowed = Vec::new();
                    let mut denied = Vec::new();
                    let mut needs_approval = Vec::new();

                    for call in tool_calls {
                        match self.eval_permission(&call.name, &call.input) {
                            Some(permission::PermissionAction::Allow) => {
                                allowed.push(call);
                            }
                            Some(permission::PermissionAction::Deny) => {
                                debug!(
                                    agent = %self.name,
                                    tool = %call.name,
                                    "tool call denied by permission rule"
                                );
                                denied.push(ToolResult::error(
                                    call.id.clone(),
                                    format!("Permission denied for tool '{}'", call.name),
                                ));
                            }
                            Some(permission::PermissionAction::Ask) | None => {
                                needs_approval.push(call);
                            }
                        }
                    }

                    // Ask for the remaining calls via the on_approval callback
                    if !needs_approval.is_empty() {
                        if let Some(ref cb) = self.on_approval {
                            self.emit(AgentEvent::ApprovalRequested {
                                agent: self.name.clone(),
                                turn: ctx.current_turn(),
                                tool_names: needs_approval
                                    .iter()
                                    .map(|tc| tc.name.clone())
                                    .collect(),
                            });
                            let decision = cb(&needs_approval);
                            self.emit(AgentEvent::ApprovalDecision {
                                agent: self.name.clone(),
                                turn: ctx.current_turn(),
                                approved: decision.is_allowed(),
                            });
                            // Persist AlwaysAllow / AlwaysDeny as learned rules
                            if decision.is_persistent() {
                                self.persist_approval_decision(&needs_approval, decision);
                            }
                            if decision.is_allowed() {
                                allowed.extend(needs_approval);
                            } else {
                                for call in &needs_approval {
                                    denied.push(ToolResult::error(
                                        call.id.clone(),
                                        "Tool execution denied by human reviewer".to_string(),
                                    ));
                                }
                            }
                        } else {
                            // No callback → allow
                            allowed.extend(needs_approval);
                        }
                    }

                    // If ALL calls were denied, add results and continue
                    if allowed.is_empty() && !denied.is_empty() {
                        total_tool_calls += denied.len();
                        ctx.add_tool_results(denied);
                        continue;
                    }

                    (allowed, denied)
                } else if let Some(ref cb) = self.on_approval {
                    // Legacy path: no permission rules, batch approval callback
                    self.emit(AgentEvent::ApprovalRequested {
                        agent: self.name.clone(),
                        turn: ctx.current_turn(),
                        tool_names: tool_calls.iter().map(|tc| tc.name.clone()).collect(),
                    });
                    let decision = cb(&tool_calls);
                    self.emit(AgentEvent::ApprovalDecision {
                        agent: self.name.clone(),
                        turn: ctx.current_turn(),
                        approved: decision.is_allowed(),
                    });
                    // Persist AlwaysAllow / AlwaysDeny as learned rules
                    if decision.is_persistent() {
                        self.persist_approval_decision(&tool_calls, decision);
                    }
                    if !decision.is_allowed() {
                        debug!(
                            agent = %self.name,
                            "tool execution denied by approval callback"
                        );
                        let results: Vec<ToolResult> = tool_calls
                            .iter()
                            .map(|tc| {
                                ToolResult::error(
                                    tc.id.clone(),
                                    "Tool execution denied by human reviewer".to_string(),
                                )
                            })
                            .collect();
                        total_tool_calls += tool_calls.len();
                        ctx.add_tool_results(results);
                        continue;
                    }
                    (tool_calls, Vec::new())
                } else {
                    (tool_calls, Vec::new())
                };

                // Doom loop detection: if the same set of tool calls is repeated
                // for N consecutive turns, return error results instead of executing.
                if let Some(threshold) = self.max_identical_tool_calls {
                    let (exact, fuzzy) = doom_tracker.record(
                        &tool_calls,
                        threshold,
                        self.max_fuzzy_identical_tool_calls,
                    );
                    if exact {
                        debug!(
                            agent = %self.name,
                            count = doom_tracker.count(),
                            "doom loop detected, returning error results"
                        );
                        self.emit(AgentEvent::DoomLoopDetected {
                            agent: self.name.clone(),
                            turn: ctx.current_turn(),
                            consecutive_count: doom_tracker.count(),
                            tool_names: tool_calls
                                .iter()
                                .map(|tc| tc.name.clone())
                                .collect(),
                        });
                        let results: Vec<ToolResult> = tool_calls
                            .iter()
                            .map(|tc| {
                                ToolResult::error(
                                    tc.id.clone(),
                                    format!(
                                        "Doom loop detected: identical tool calls repeated {} \
                                         times consecutively. Try a different approach.",
                                        doom_tracker.count()
                                    ),
                                )
                            })
                            .collect();
                        total_tool_calls += tool_calls.len();
                        ctx.add_tool_results(results);
                        continue;
                    } else if fuzzy {
                        debug!(
                            agent = %self.name,
                            count = doom_tracker.fuzzy_count(),
                            "fuzzy doom loop detected, returning error results"
                        );
                        self.emit(AgentEvent::FuzzyDoomLoopDetected {
                            agent: self.name.clone(),
                            turn: ctx.current_turn(),
                            consecutive_count: doom_tracker.fuzzy_count(),
                            tool_names: tool_calls
                                .iter()
                                .map(|tc| tc.name.clone())
                                .collect(),
                        });
                        let results: Vec<ToolResult> = tool_calls
                            .iter()
                            .map(|tc| {
                                ToolResult::error(
                                    tc.id.clone(),
                                    format!(
                                        "Fuzzy doom loop detected: same tools with different \
                                         inputs repeated {} times consecutively. Try a \
                                         completely different approach.",
                                        doom_tracker.fuzzy_count()
                                    ),
                                )
                            })
                            .collect();
                        total_tool_calls += tool_calls.len();
                        ctx.add_tool_results(results);
                        continue;
                    }
                }

                // pre_tool guardrail: per-call fine-grained filter
                let (allowed_calls, denied_results) = if self.guardrails.is_empty() {
                    (tool_calls, Vec::new())
                } else {
                    let mut allowed = Vec::new();
                    let mut denied = Vec::new();
                    for call in tool_calls {
                        let mut call_denied = false;
                        for g in &self.guardrails {
                            match g.pre_tool(&call).await.map_err(|e| (e, total_usage))? {
                                GuardAction::Allow => {}
                                GuardAction::Warn { reason } => {
                                    self.emit(AgentEvent::GuardrailWarned {
                                        agent: self.name.clone(),
                                        hook: "pre_tool".into(),
                                        reason: reason.clone(),
                                        tool_name: Some(call.name.clone()),
                                    });
                                    self.audit(AuditRecord {
                                        agent: self.name.clone(),
                                        turn: ctx.current_turn(),
                                        event_type: "guardrail_warned".into(),
                                        payload: serde_json::json!({
                                            "hook": "pre_tool",
                                            "reason": reason,
                                            "tool_name": call.name,
                                        }),
                                        usage: TokenUsage::default(),
                                        timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
                                    })
                                    .await;
                                    // Continue — do NOT deny the tool call
                                }
                                GuardAction::Deny { reason } => {
                                    self.emit(AgentEvent::GuardrailDenied {
                                        agent: self.name.clone(),
                                        hook: "pre_tool".into(),
                                        reason: reason.clone(),
                                        tool_name: Some(call.name.clone()),
                                    });
                                    // Audit: pre_tool guardrail denied
                                    self.audit(AuditRecord {
                                        agent: self.name.clone(),
                                        turn: ctx.current_turn(),
                                        event_type: "guardrail_denied".into(),
                                        payload: serde_json::json!({
                                            "hook": "pre_tool",
                                            "reason": reason,
                                            "tool_name": call.name,
                                        }),
                                        usage: TokenUsage::default(),
                                        timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
                                    })
                                    .await;
                                    denied.push(ToolResult::error(
                                        call.id.clone(),
                                        format!("Guardrail denied: {reason}"),
                                    ));
                                    call_denied = true;
                                    break;
                                }
                                GuardAction::Kill { reason } => {
                                    self.emit(AgentEvent::KillSwitchActivated {
                                        agent: self.name.clone(),
                                        reason: reason.clone(),
                                        guardrail_name: g.name().to_string(),
                                    });
                                    self.audit(AuditRecord {
                                        agent: self.name.clone(),
                                        turn: ctx.current_turn(),
                                        event_type: "guardrail_killed".into(),
                                        payload: serde_json::json!({
                                            "hook": "pre_tool",
                                            "reason": reason,
                                            "tool_name": call.name,
                                        }),
                                        usage: TokenUsage::default(),
                                        timestamp: chrono::Utc::now(),
                                        user_id: self.audit_user_id.clone(),
                                        tenant_id: self.audit_tenant_id.clone(),
                                        delegation_chain: self.audit_delegation_chain.clone(),
                                    })
                                    .await;
                                    return Err((
                                        Error::KillSwitch(reason),
                                        total_usage,
                                    ));
                                }
                            }
                        }
                        if !call_denied {
                            allowed.push(call);
                        }
                    }
                    (allowed, denied)
                };

                total_tool_calls +=
                    allowed_calls.len() + denied_results.len() + permission_denied_results.len();
                // Update recently-used tool list for dynamic tool selection
                recently_used_tools = allowed_calls.iter().map(|c| c.name.clone()).collect();
                let tool_batch_span = info_span!(
                    "heartbit.agent.tool_batch",
                    agent = %self.name,
                    turn = ctx.current_turn(),
                    tool_count = allowed_calls.len(),
                );
                let (mut results, batch_records) = self
                    .execute_tools_parallel(&allowed_calls, ctx.current_turn())
                    .instrument(tool_batch_span)
                    .await;
                tool_call_records.extend(batch_records);
                results.extend(denied_results);
                results.extend(permission_denied_results);

                // LSP diagnostics: after file-modifying tools, collect diagnostics
                // and append to the tool result so the LLM sees errors immediately.
                if let Some(ref lsp) = self.lsp_manager {
                    self.append_lsp_diagnostics(lsp, &allowed_calls, &mut results)
                        .await;
                }

                // Compress oversized tool outputs via LLM call
                if let Some(threshold) = self.tool_output_compression_threshold {
                    for result in &mut results {
                        if !result.is_error && result.content.len() > threshold {
                            let compressed = self
                                .compress_tool_output(&result.content, threshold, &mut total_usage)
                                .await;
                            result.content = compressed;
                        }
                    }
                    *usage_acc.lock().expect("usage lock poisoned") = total_usage;
                }

                ctx.add_tool_results(results);

                // Reflection: inject a user-role prompt that nudges the LLM to assess
                // tool results before deciding the next action (Reflexion/CRITIC pattern).
                if self.enable_reflection {
                    ctx.add_user_message(
                        "Before proceeding, briefly reflect on the tool results above:\n\
                     1. Did you get the information you needed?\n\
                     2. Are there any errors or unexpected results?\n\
                     3. What is the best next step?"
                            .to_string(),
                    );
                }

                // Summarization: if threshold is set and context exceeds it, compress.
                // Guard on message count: inject_summary(keep_last_n=4) is a no-op
                // when total messages <= 5 (1 first + 4 kept), so skip the LLM call.
                if let Some(threshold) = self.summarize_threshold
                    && ctx.message_count() > 5
                    && ctx.needs_compaction(threshold)
                {
                    debug!(agent = %self.name, "context exceeds threshold, summarizing");
                    let summarize_span = info_span!(
                        "heartbit.agent.summarize",
                        agent = %self.name,
                        turn = ctx.current_turn(),
                    );
                    let (summary, summary_usage) =
                        match self.generate_summary(&ctx).instrument(summarize_span).await {
                            Ok(r) => r,
                            Err(e) => {
                                self.emit(AgentEvent::RunFailed {
                                    agent: self.name.clone(),
                                    error: e.to_string(),
                                    partial_usage: total_usage,
                                });
                                return Err((e, total_usage));
                            }
                        };
                    total_usage += summary_usage;
                    *usage_acc.lock().expect("usage lock poisoned") = total_usage;
                    if let Some(summary) = summary {
                        self.flush_to_memory_before_compaction(&ctx, 4).await;
                        ctx.inject_summary(summary, 4);
                        self.emit(AgentEvent::ContextSummarized {
                            agent: self.name.clone(),
                            turn: ctx.current_turn(),
                            usage: summary_usage,
                        });
                    }
                }
            }
        }
        .instrument(run_span.clone())
        .await;

        // Record final metrics on the run span
        if mode.includes_metrics() {
            let usage = match &result {
                Ok(output) => &output.tokens_used,
                Err((_, usage)) => usage,
            };
            run_span.record("total_input_tokens", usage.input_tokens);
            run_span.record("total_output_tokens", usage.output_tokens);
            if let Ok(ref output) = result
                && let Some(cost) = output.estimated_cost_usd
            {
                run_span.record("estimated_cost_usd", cost);
            }
        }

        result
    }

    /// Generate a summary of the conversation so far using the LLM.
    ///
    /// Returns `(Option<summary_text>, token_usage)`. The summary is `None` if
    /// truncated (MaxTokens), in which case the caller should skip compaction.
    /// Token usage is always returned so the caller can accumulate it.
    async fn generate_summary(
        &self,
        ctx: &AgentContext,
    ) -> Result<(Option<String>, TokenUsage), Error> {
        let text = ctx.conversation_text();
        let lines: Vec<&str> = text.lines().collect();

        // Use recursive summarization for long conversations (>20 lines)
        const CLUSTER_SIZE: usize = 10;
        if self.enable_recursive_summarization && lines.len() > CLUSTER_SIZE * 2 {
            return self.generate_recursive_summary(&lines, CLUSTER_SIZE).await;
        }

        self.summarize_text(&text).await
    }

    /// Single-shot summarization of a text block.
    async fn summarize_text(&self, text: &str) -> Result<(Option<String>, TokenUsage), Error> {
        let summary_request = CompletionRequest {
            system: "You are a summarization assistant. Summarize the following conversation \
                     concisely, preserving key facts, decisions, and tool results. \
                     Focus on information that would be needed to continue the conversation."
                .into(),
            messages: vec![Message::user(text.to_string())],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };

        let response = self.provider.complete(summary_request).await?;
        let usage = response.usage;
        if response.stop_reason == StopReason::MaxTokens {
            tracing::warn!(
                agent = %self.name,
                "summarization truncated (max_tokens reached), skipping compaction"
            );
            return Ok((None, usage));
        }
        Ok((Some(response.text()), usage))
    }

    /// Recursive summarization: chunk messages into clusters, summarize each,
    /// then summarize the combined cluster summaries.
    ///
    /// Preserves 3-5x more detail than single-shot for long conversations.
    async fn generate_recursive_summary(
        &self,
        lines: &[&str],
        cluster_size: usize,
    ) -> Result<(Option<String>, TokenUsage), Error> {
        let mut total_usage = TokenUsage::default();
        let mut cluster_summaries = Vec::new();

        // Phase 1: Summarize each cluster
        for chunk in lines.chunks(cluster_size) {
            let cluster_text = chunk.join("\n");
            let (summary, usage) = self.summarize_text(&cluster_text).await?;
            total_usage += usage;
            match summary {
                Some(s) => cluster_summaries.push(s),
                None => {
                    // If any cluster summary is truncated, fall back to single-shot
                    let full_text = lines.join("\n");
                    let (summary, usage) = self.summarize_text(&full_text).await?;
                    total_usage += usage;
                    return Ok((summary, total_usage));
                }
            }
        }

        // Phase 2: Combine cluster summaries into final summary
        let combined = format!(
            "Summarize the following section summaries into one cohesive summary:\n\n{}",
            cluster_summaries
                .iter()
                .enumerate()
                .map(|(i, s)| format!("Section {}:\n{}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n\n")
        );
        let (final_summary, combine_usage) = self.summarize_text(&combined).await?;
        total_usage += combine_usage;
        Ok((final_summary, total_usage))
    }

    /// Build a `TenantScope` from the agent's audit identity fields.
    ///
    /// Falls back to single-tenant (empty `tenant_id`) when no audit context is set.
    fn memory_scope(&self) -> crate::auth::TenantScope {
        crate::auth::TenantScope::from_audit_fields(
            self.audit_tenant_id.as_deref(),
            self.audit_user_id.as_deref(),
        )
    }

    /// Flush key tool results to memory before compaction.
    ///
    /// Extracts non-error tool results exceeding a minimum length from messages
    /// that are about to be compacted, storing them as episodic memories.
    async fn flush_to_memory_before_compaction(&self, ctx: &AgentContext, keep_last_n: usize) {
        let Some(ref memory) = self.memory else {
            return;
        };

        let messages = ctx.messages_to_be_compacted(keep_last_n);
        let now = chrono::Utc::now();

        for msg in messages {
            if msg.role != crate::llm::types::Role::User {
                continue;
            }
            for block in &msg.content {
                if let ContentBlock::ToolResult {
                    content, is_error, ..
                } = block
                {
                    // Skip errors and very short results
                    if *is_error || content.len() < 50 {
                        continue;
                    }
                    // Truncate very long results to a reasonable size
                    let stored_content = if content.len() > 500 {
                        format!(
                            "{}...",
                            &content[..crate::tool::builtins::floor_char_boundary(content, 500)]
                        )
                    } else {
                        content.clone()
                    };
                    let id = uuid::Uuid::new_v4().to_string();
                    let entry = crate::memory::MemoryEntry {
                        id,
                        agent: self.name.clone(),
                        content: stored_content,
                        category: "fact".into(),
                        tags: vec!["auto-flush".into()],
                        created_at: now,
                        last_accessed: now,
                        access_count: 0,
                        importance: 3,
                        memory_type: crate::memory::MemoryType::Episodic,
                        keywords: vec![],
                        summary: None,
                        strength: 0.8,
                        related_ids: vec![],
                        source_ids: vec![],
                        embedding: None,
                        confidentiality: crate::memory::Confidentiality::default(),
                        author_user_id: None,
                        author_tenant_id: None,
                    };
                    let scope = self.memory_scope();
                    if let Err(e) = memory.store(&scope, entry).await {
                        tracing::warn!(
                            agent = %self.name,
                            error = %e,
                            "failed to flush tool result to memory before compaction"
                        );
                    }
                }
            }
        }
    }

    /// Prune weak memory entries at session end.
    ///
    /// Runs Ebbinghaus-based pruning with default thresholds. Errors are logged
    /// but do not fail the session — pruning is best-effort maintenance.
    async fn prune_memory_on_exit(&self) {
        let Some(ref memory) = self.memory else {
            return;
        };
        let scope = self.memory_scope();
        match crate::memory::pruning::prune_weak_entries(
            memory,
            &scope,
            crate::memory::pruning::DEFAULT_MIN_STRENGTH,
            crate::memory::pruning::default_min_age(),
        )
        .await
        {
            Ok(0) => {}
            Ok(n) => {
                tracing::debug!(agent = %self.name, pruned = n, "pruned weak memory entries at session end");
            }
            Err(e) => {
                tracing::warn!(agent = %self.name, error = %e, "memory pruning failed at session end");
            }
        }
    }

    /// Run memory consolidation at session end (opt-in).
    ///
    /// Clusters related episodic memories by keyword overlap and merges them
    /// into semantic summaries via LLM. Returns accumulated token usage.
    async fn consolidate_memory_on_exit(&self) -> TokenUsage {
        if !self.consolidate_on_exit {
            return TokenUsage::default();
        }
        let Some(ref memory) = self.memory else {
            return TokenUsage::default();
        };
        let pipeline = crate::memory::consolidation::ConsolidationPipeline::new(
            memory.clone(),
            self.provider.clone(),
            &self.name,
        );
        let scope = self.memory_scope();
        match pipeline.run(&scope).await {
            Ok((0, _, usage)) => usage,
            Ok((clusters, entries, usage)) => {
                tracing::debug!(
                    agent = %self.name,
                    clusters,
                    entries,
                    "consolidated memories at session end"
                );
                usage
            }
            Err(e) => {
                tracing::warn!(
                    agent = %self.name,
                    error = %e,
                    "memory consolidation failed at session end"
                );
                TokenUsage::default()
            }
        }
    }

    /// Select the most relevant tools for the current turn.
    ///
    /// Strategy:
    /// 1. Always include tools used in the last 2 turns (momentum)
    /// 2. Score remaining tools by keyword overlap with recent messages
    /// 3. Cap at `max_tools`
    pub(super) fn select_tools_for_turn(
        &self,
        all_tools: &[ToolDefinition],
        messages: &[Message],
        recently_used: &[String],
        max_tools: usize,
    ) -> Vec<ToolDefinition> {
        if all_tools.len() <= max_tools {
            return all_tools.to_vec();
        }

        // Collect text from last 2 user/assistant messages for keyword matching
        let recent_text: String = messages
            .iter()
            .rev()
            .take(4)
            .flat_map(|m| m.content.iter())
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        let keywords: Vec<&str> = recent_text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() > 2)
            .collect();

        // Partition into pinned (always included) and candidates.
        // Pinned: recently-used tools + __respond__ (structured output must never be dropped).
        let mut selected: Vec<ToolDefinition> = Vec::new();
        let mut candidates: Vec<(ToolDefinition, usize)> = Vec::new();

        for tool in all_tools {
            if recently_used.contains(&tool.name)
                || tool.name == crate::llm::types::RESPOND_TOOL_NAME
            {
                selected.push(tool.clone());
            } else {
                // Score by keyword overlap with tool name + description
                let tool_text = format!("{} {}", tool.name, tool.description).to_lowercase();
                let score = keywords
                    .iter()
                    .filter(|kw| tool_text.contains(**kw))
                    .count();
                candidates.push((tool.clone(), score));
            }
        }

        // Sort candidates by score descending
        candidates.sort_by_key(|c| std::cmp::Reverse(c.1));

        // Fill remaining slots (cap total at max_tools)
        let remaining = max_tools.saturating_sub(selected.len());
        selected.extend(candidates.into_iter().take(remaining).map(|(t, _)| t));

        selected.truncate(max_tools);
        selected
    }

    /// Compress a tool output using the LLM when it exceeds the threshold.
    ///
    /// Returns the original content if below threshold or on compression error.
    /// On success, returns the compressed text with a byte-count annotation.
    async fn compress_tool_output(
        &self,
        content: &str,
        threshold: usize,
        usage_acc: &mut TokenUsage,
    ) -> String {
        if content.len() < threshold {
            return content.to_string();
        }
        let original_len = content.len();
        let request = CompletionRequest {
            system: "Compress the following tool output, preserving all factual content, \
                     key values, and actionable information. Remove redundancy and formatting \
                     noise. Return ONLY the compressed content."
                .into(),
            messages: vec![Message::user(content.to_string())],
            tools: vec![],
            max_tokens: (self.max_tokens / 3).max(256),
            tool_choice: None,
            reasoning_effort: None,
        };
        match self.provider.complete(request).await {
            Ok(resp) => {
                *usage_acc += resp.usage;
                let compressed = resp.text();
                if compressed.is_empty() {
                    content.to_string()
                } else {
                    format!("{compressed}\n[compressed from {original_len} bytes]")
                }
            }
            Err(e) => {
                debug!(agent = %self.name, error = %e, "tool output compression failed, using original");
                content.to_string()
            }
        }
    }

    /// Find the closest tool name match within a maximum edit distance.
    /// Returns the matching tool name if found within `max_distance`.
    pub(super) fn find_closest_tool(&self, name: &str, max_distance: usize) -> Option<&str> {
        self.tools
            .keys()
            .map(|k| (k.as_str(), levenshtein(name, k)))
            .filter(|(_, d)| *d <= max_distance && *d > 0)
            .min_by_key(|(_, d)| *d)
            .map(|(name, _)| name)
    }

    /// After file-modifying tools, collect LSP diagnostics and append them
    /// to the corresponding tool results.
    async fn append_lsp_diagnostics(
        &self,
        lsp: &crate::lsp::LspManager,
        calls: &[ToolCall],
        results: &mut [ToolResult],
    ) {
        for (idx, call) in calls.iter().enumerate() {
            if !crate::lsp::is_file_modifying_tool(&call.name) {
                continue;
            }
            // Skip LSP diagnostics for failed tool calls — the file wasn't modified
            if idx < results.len() && results[idx].is_error {
                continue;
            }
            // Extract the file path from the tool input
            let path_str = match call
                .input
                .get("path")
                .or_else(|| call.input.get("file_path"))
            {
                Some(serde_json::Value::String(s)) => s.clone(),
                _ => continue,
            };
            let path = std::path::Path::new(&path_str);
            let diagnostics = lsp.notify_file_changed(path).await;
            if diagnostics.is_empty() {
                tracing::debug!(
                    agent = %self.name,
                    path = %path_str,
                    "lsp: no diagnostics for file"
                );
            } else {
                let formatted = crate::lsp::format_diagnostics(&path_str, &diagnostics);
                tracing::info!(
                    agent = %self.name,
                    path = %path_str,
                    count = diagnostics.len(),
                    "lsp-diagnostics appended to tool result"
                );
                if idx < results.len() {
                    results[idx].content.push('\n');
                    results[idx].content.push_str(&formatted);
                }
            }
        }
    }

    /// Execute tools in parallel via JoinSet, returning results in original call order.
    ///
    /// Panicked tasks produce an error `ToolResult` so the LLM always gets a
    /// result for every `tool_use_id` it sent.
    async fn execute_tools_parallel(
        &self,
        calls: &[ToolCall],
        turn: usize,
    ) -> (Vec<ToolResult>, Vec<ToolCallRecord>) {
        let call_ids: Vec<String> = calls.iter().map(|c| c.id.clone()).collect();
        let call_names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
        let mut join_set = tokio::task::JoinSet::new();

        // Construct per-turn ExecutionContext from runner's audit fields.
        // Phase 0: workspace, credentials, audit_sink are not yet populated on
        // AgentRunner — leave them None until persona/credential plumbing lands.
        let exec_ctx = crate::ExecutionContext {
            tenant_id: self.audit_tenant_id.clone(),
            user_id: self.audit_user_id.clone(),
            workspace: None,
            credentials: None,
            audit_sink: None,
        };

        for (idx, call) in calls.iter().enumerate() {
            // SECURITY (F-AGENT-1): names are already repaired upstream of the
            // permission and pre_tool guardrails. If the lookup fails here, the
            // name was unknown AND not Levenshtein-close to any tool — return a
            // "Tool not found" error and let the LLM correct itself. Repairing
            // at dispatch time would bypass the policy that just ran.
            let tool = self.tools.get(&call.name).cloned();
            let input = call.input.clone();
            let call_name = call.name.clone();
            let timeout = self.tool_timeout;

            self.emit(AgentEvent::ToolCallStarted {
                agent: self.name.clone(),
                tool_name: call.name.clone(),
                tool_call_id: call.id.clone(),
                input: truncate_for_event(
                    &serde_json::to_string(&call.input).unwrap_or_default(),
                    EVENT_MAX_PAYLOAD_BYTES,
                ),
            });

            // Audit: tool call (untruncated input)
            self.audit(AuditRecord {
                agent: self.name.clone(),
                turn,
                event_type: "tool_call".into(),
                payload: serde_json::json!({
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "input": call.input,
                }),
                usage: TokenUsage::default(),
                timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
            })
            .await;

            // Validate input against the tool's declared schema before dispatching.
            // On failure, produce an error result without executing the tool.
            if let Some(ref t) = tool {
                let schema = &t.definition().input_schema;
                if let Err(msg) = validate_tool_input(schema, &input) {
                    join_set.spawn(async move { (idx, Ok(ToolOutput::error(msg)), 0u64) });
                    continue;
                }
            }

            let tool_span = info_span!(
                "heartbit.agent.tool_call",
                agent = %self.name,
                tool_name = %call.name,
            );
            let task_ctx = exec_ctx.clone();
            join_set.spawn(
                async move {
                    let start = std::time::Instant::now();
                    let output = match tool {
                        Some(t) => match timeout {
                            Some(dur) => {
                                match tokio::time::timeout(dur, t.execute(&task_ctx, input)).await {
                                    Ok(result) => result,
                                    Err(_) => Ok(ToolOutput::error(format!(
                                        "Tool execution timed out after {}s",
                                        dur.as_secs_f64()
                                    ))),
                                }
                            }
                            None => t.execute(&task_ctx, input).await,
                        },
                        None => Ok(ToolOutput::error(format!("Tool not found: {call_name}"))),
                    };
                    let duration_ms = start.elapsed().as_millis() as u64;
                    (idx, output, duration_ms)
                }
                .instrument(tool_span),
            );
        }

        // Collect (idx, output, duration) tuples from JoinSet
        let mut outputs: Vec<Option<(ToolOutput, u64)>> = vec![None; calls.len()];
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((idx, Ok(output), duration_ms)) => {
                    let output = match self.max_tool_output_bytes {
                        Some(max) => output.truncated(max),
                        None => output,
                    };
                    outputs[idx] = Some((output, duration_ms));
                }
                Ok((idx, Err(e), duration_ms)) => {
                    outputs[idx] = Some((ToolOutput::error(e.to_string()), duration_ms));
                }
                Err(join_err) => {
                    tracing::error!(error = %join_err, "tool task panicked");
                }
            }
        }

        // Apply post_tool guardrails and convert to ToolResult
        let mut results_vec = Vec::with_capacity(calls.len());
        let mut records_vec: Vec<ToolCallRecord> = Vec::with_capacity(calls.len());
        for (idx, slot) in outputs.into_iter().enumerate() {
            let (mut output, duration_ms) = slot
                .unwrap_or_else(|| (ToolOutput::error("Tool execution panicked".to_string()), 0));

            // post_tool guardrail: each guardrail can mutate the output
            for g in &self.guardrails {
                if let Err(e) = g.post_tool(&calls[idx], &mut output).await {
                    self.emit(AgentEvent::GuardrailDenied {
                        agent: self.name.clone(),
                        hook: "post_tool".into(),
                        reason: e.to_string(),
                        tool_name: Some(call_names[idx].clone()),
                    });
                    // Audit: post_tool guardrail denied
                    self.audit(AuditRecord {
                        agent: self.name.clone(),
                        turn,
                        event_type: "guardrail_denied".into(),
                        payload: serde_json::json!({
                            "hook": "post_tool",
                            "reason": e.to_string(),
                            "tool_name": call_names[idx],
                        }),
                        usage: TokenUsage::default(),
                        timestamp: chrono::Utc::now(),
                        // SECURITY (F-AGENT-5): attribute the deny to the
                        // identity the rest of the run is attributed to. All
                        // other AuditRecord sites in this file pass these
                        // fields; this one used to set them to None, leaving
                        // post_tool denials unattributable cross-tenant.
                        user_id: self.audit_user_id.clone(),
                        tenant_id: self.audit_tenant_id.clone(),
                        delegation_chain: self.audit_delegation_chain.clone(),
                    })
                    .await;
                    // post_tool error: convert to error output instead of aborting
                    // the entire run (consistent with tool execution errors)
                    output = ToolOutput::error(format!("Guardrail error: {e}"));
                    break;
                }
            }

            let is_error = output.is_error;
            self.emit(AgentEvent::ToolCallCompleted {
                agent: self.name.clone(),
                tool_name: call_names[idx].clone(),
                tool_call_id: call_ids[idx].clone(),
                is_error,
                duration_ms,
                output: truncate_for_event(&output.content, EVENT_MAX_PAYLOAD_BYTES),
            });
            // Audit: tool result (untruncated output)
            self.audit(AuditRecord {
                agent: self.name.clone(),
                turn,
                event_type: "tool_result".into(),
                payload: serde_json::json!({
                    "tool_name": call_names[idx],
                    "tool_call_id": call_ids[idx],
                    "output": output.content,
                    "is_error": is_error,
                    "duration_ms": duration_ms,
                }),
                usage: TokenUsage::default(),
                timestamp: chrono::Utc::now(),
                user_id: self.audit_user_id.clone(),
                tenant_id: self.audit_tenant_id.clone(),
                delegation_chain: self.audit_delegation_chain.clone(),
            })
            .await;

            // Capture FULL post-guardrail content for AgentOutput.tool_call_results.
            // This is the raw output as the caller would expect to see it; the
            // redacted variant below is only what gets fed back to the LLM.
            records_vec.push(ToolCallRecord {
                tool_name: call_names[idx].clone(),
                tool_call_id: call_ids[idx].clone(),
                input: calls[idx].input.clone(),
                output: output.content.clone(),
                is_error,
                duration_ms,
            });

            // Compute the conversation-history-safe variant via the tool's
            // optional `redact_for_history` override. Tools without an
            // override return the content verbatim. Re-look-up by name —
            // the original `Arc<dyn Tool>` was moved into the JoinSet
            // earlier and is no longer in scope here. On miss (which can
            // happen for "Tool not found" stub outputs), fall through to
            // the original content.
            let redacted_content = match self.tools.get(&calls[idx].name) {
                Some(t) => t.redact_for_history(&output.content),
                None => output.content.clone(),
            };
            let redacted_output = if is_error {
                ToolOutput::error(redacted_content)
            } else {
                ToolOutput::success(redacted_content)
            };
            results_vec.push(tool_output_to_result(
                call_ids[idx].clone(),
                redacted_output,
            ));
        }

        (results_vec, records_vec)
    }
}

impl<P: LlmProvider> Drop for AgentRunner<P> {
    fn drop(&mut self) {
        if let (Some(tracker), Some(tid)) =
            (self.tenant_tracker.as_ref(), self.audit_tenant_id.as_ref())
        {
            let actual = self
                .cumulative_actual_tokens
                .load(std::sync::atomic::Ordering::SeqCst) as i64;
            if actual > 0 {
                let scope = crate::auth::TenantScope::new(tid.clone());
                tracker.adjust(&scope, -actual);
            }
        }
    }
}

pub(super) fn tool_output_to_result(tool_use_id: String, output: ToolOutput) -> ToolResult {
    if output.is_error {
        ToolResult::error(tool_use_id, output.content)
    } else {
        ToolResult::success(tool_use_id, output.content)
    }
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::Arc;

    use crate::agent::tenant_tracker::TenantTokenTracker;
    use crate::auth::TenantScope;
    use crate::error::Error;
    use crate::llm::types::{
        CompletionResponse, ContentBlock, StopReason, TokenUsage, ToolDefinition,
    };
    use crate::tool::{Tool, ToolOutput};

    use super::super::test_helpers::MockProvider;
    use super::AgentRunner;

    /// Trivial no-op tool so the runner can dispatch a tool_use response.
    struct NoopTool;

    impl Tool for NoopTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "noop".into(),
                description: "Does nothing.".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            }
        }

        fn execute(
            &self,
            _ctx: &crate::ExecutionContext,
            _input: serde_json::Value,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
        {
            Box::pin(async { Ok(ToolOutput::success("ok".to_string())) })
        }
    }

    /// Build a tool-use response so the runner loops back for a second LLM call.
    fn tool_use_response(input_tokens: u32, output_tokens: u32) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "noop".into(),
                input: serde_json::json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage {
                input_tokens,
                output_tokens,
                ..Default::default()
            },
            model: None,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_runner_adjusts_tenant_tracker_per_turn() {
        let tracker = Arc::new(TenantTokenTracker::new(1_000_000));
        let scope = TenantScope::new("acme");
        // Simulate the daemon's submit-time admission check (Task 7) — drop
        // the reservation immediately, matching admission-only semantics.
        drop(tracker.reserve(&scope, 5000).unwrap());
        assert_eq!(tracker.snapshot()[0].1.in_flight, 0);

        // Build a mock provider that returns known TokenUsage in one turn.
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "done", 100, 200,
        )]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("test")
            .audit_user_context("test-user", "acme")
            .tenant_tracker(tracker.clone())
            .max_turns(1)
            .build()
            .unwrap();
        let _output = runner.execute("hello").await.unwrap();

        // After one turn: cumulative_actual_tokens = 300, so adjust(+300).
        let snap = tracker.snapshot();
        assert_eq!(snap[0].1.in_flight, 300);

        // After runner Drop: in_flight returns to 0.
        drop(runner);
        let snap = tracker.snapshot();
        assert_eq!(snap[0].1.in_flight, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_runner_adjusts_tracker_cumulatively_across_turns() {
        // Two-turn test: verifies cumulative semantics (not per-turn deltas).
        // Turn 1: tool_use response (300 tokens) → runner loops.
        // Turn 2: text response (200 tokens) → runner stops.
        // Expected: in_flight = 500 (cumulative), zeroed on Drop.
        let tracker = Arc::new(TenantTokenTracker::new(1_000_000));
        let scope = TenantScope::new("acme");
        drop(tracker.reserve(&scope, 5000).unwrap());

        let provider = Arc::new(MockProvider::new(vec![
            tool_use_response(100, 200), // turn 1: +300 → 300 cumulative
            MockProvider::text_response("done", 50, 150), // turn 2: +200 → 500 cumulative
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("test")
            .audit_user_context("test-user", "acme")
            .tenant_tracker(tracker.clone())
            .max_turns(2)
            .tool(Arc::new(NoopTool))
            .build()
            .unwrap();
        let _output = runner.execute("hello").await.unwrap();

        // After two turns: cumulative = 300 + 200 = 500.
        let snap = tracker.snapshot();
        assert_eq!(snap[0].1.in_flight, 500);

        drop(runner);
        assert_eq!(tracker.snapshot()[0].1.in_flight, 0);
    }

    #[tokio::test]
    async fn execution_context_propagates_to_tool() {
        use std::sync::Mutex;

        use crate::ExecutionContext;
        use crate::llm::types::ToolCall;

        struct CtxCapturingTool {
            captured_tenant: Arc<Mutex<Option<String>>>,
        }

        impl Tool for CtxCapturingTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "ctx_capture".into(),
                    description: "Captures the tenant_id from ExecutionContext.".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }
            }

            fn execute(
                &self,
                ctx: &ExecutionContext,
                _input: serde_json::Value,
            ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
            {
                let captured = self.captured_tenant.clone();
                let tenant = ctx.tenant_id.clone();
                Box::pin(async move {
                    *captured.lock().unwrap() = tenant;
                    Ok(ToolOutput::success("ok"))
                })
            }
        }

        let captured = Arc::new(Mutex::new(None));
        let tool = Arc::new(CtxCapturingTool {
            captured_tenant: captured.clone(),
        });

        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("test")
            .max_turns(1)
            .tools(vec![tool as Arc<dyn Tool>])
            .audit_user_context("test-user", "test-tenant")
            .build()
            .unwrap();

        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "ctx_capture".into(),
            input: serde_json::json!({}),
        }];
        let (_results, _records) = runner.execute_tools_parallel(&calls, 0).await;

        assert_eq!(
            captured.lock().unwrap().as_deref(),
            Some("test-tenant"),
            "tool did not receive the tenant_id from ExecutionContext"
        );
    }

    /// P1.3g: a single-tool agent run populates `AgentOutput.tool_call_results`
    /// with one record carrying the right tool_name, input, and output.
    #[tokio::test]
    async fn agent_output_tool_call_results_populated_after_tool_call() {
        // Turn 1: LLM asks for `noop` tool. Turn 2: LLM returns final text.
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_response(10, 20),
            MockProvider::text_response("done", 5, 5),
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("test")
            .max_turns(2)
            .tool(Arc::new(NoopTool))
            .build()
            .unwrap();
        let output = runner.execute("hello").await.unwrap();

        assert_eq!(output.tool_call_results.len(), 1, "expected one record");
        let rec = &output.tool_call_results[0];
        assert_eq!(rec.tool_name, "noop");
        assert_eq!(rec.tool_call_id, "call-1");
        assert_eq!(rec.input, serde_json::json!({}));
        assert_eq!(rec.output, "ok");
        assert!(!rec.is_error);
    }

    /// P1.3g: when a tool overrides `redact_for_history`, `tool_call_results`
    /// must contain the FULL untruncated output, while the conversation
    /// history sent to the next LLM turn carries the redacted variant.
    #[tokio::test]
    async fn agent_output_tool_call_results_uses_full_output_not_redacted() {
        use crate::llm::types::Role;

        /// Tool that returns a long output and redacts to "REDACTED".
        struct BlobTool;
        impl Tool for BlobTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "blob".into(),
                    description: "Returns a large blob.".into(),
                    input_schema: serde_json::json!({"type": "object", "properties": {}}),
                }
            }

            fn execute(
                &self,
                _ctx: &crate::ExecutionContext,
                _input: serde_json::Value,
            ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
            {
                Box::pin(async { Ok(ToolOutput::success("FULL_BLOB_DATA_XYZ".to_string())) })
            }

            fn redact_for_history(&self, _output: &str) -> String {
                "REDACTED".to_string()
            }
        }

        // Turn 1: LLM calls `blob`. Turn 2: LLM returns final text.
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-blob".into(),
                    name: "blob".into(),
                    input: serde_json::json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 20,
                    ..Default::default()
                },
                model: None,
            },
            MockProvider::text_response("done", 5, 5),
        ]));

        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("test")
            .max_turns(2)
            .tool(Arc::new(BlobTool))
            .build()
            .unwrap();
        let output = runner.execute("hello").await.unwrap();

        // 1) AgentOutput records the FULL untruncated output.
        assert_eq!(output.tool_call_results.len(), 1);
        assert_eq!(output.tool_call_results[0].output, "FULL_BLOB_DATA_XYZ");
        assert!(!output.tool_call_results[0].output.contains("REDACTED"));

        // 2) The conversation history sent on turn 2 contains the REDACTED
        //    variant in the ContentBlock::ToolResult fed back to the LLM.
        let captured = provider
            .captured_requests
            .lock()
            .expect("capture lock poisoned");
        assert!(
            captured.len() >= 2,
            "expected at least 2 LLM calls, got {}",
            captured.len()
        );
        let turn2 = &captured[1];
        let mut found_redacted = false;
        let mut found_full = false;
        for msg in &turn2.messages {
            if msg.role == Role::User {
                for block in &msg.content {
                    if let ContentBlock::ToolResult { content, .. } = block {
                        if content.contains("REDACTED") {
                            found_redacted = true;
                        }
                        if content.contains("FULL_BLOB_DATA_XYZ") {
                            found_full = true;
                        }
                    }
                }
            }
        }
        assert!(
            found_redacted,
            "expected REDACTED tool result in turn-2 conversation history"
        );
        assert!(
            !found_full,
            "FULL_BLOB_DATA_XYZ should NOT be in conversation history sent to LLM"
        );
    }
}
