pub mod audit;
pub mod blackboard;
pub(crate) mod blackboard_tools;
pub mod context;
pub mod events;
pub mod guardrail;
pub mod guardrails;
pub mod instructions;
pub mod observability;
pub mod orchestrator;
pub mod permission;
pub mod pruner;
pub mod routing;
pub(crate) mod token_estimator;
pub mod tool_filter;
pub mod workflow;

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
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
use crate::tool::{Tool, ToolOutput, validate_tool_input};

use crate::knowledge::KnowledgeBase;
use crate::memory::Memory;

use crate::tool::builtins::OnQuestion;

use self::audit::{AuditRecord, AuditTrail};
use self::context::{AgentContext, ContextStrategy};
use self::events::{AgentEvent, EVENT_MAX_PAYLOAD_BYTES, OnEvent, truncate_for_event};
use self::guardrail::{GuardAction, Guardrail};

/// Callback for interactive mode. Called when the agent needs more user input
/// (i.e., the LLM returned text without tool calls). Returns `Some(message)`
/// to continue the conversation, or `None` to end the session.
pub type OnInput = dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
    + Send
    + Sync;

/// Behavioral guidelines appended to every agent's system prompt.
/// Ensures agents proactively discover capabilities and exhaust options
/// before claiming they cannot do something.
const RESOURCEFULNESS_GUIDELINES: &str = "\n\n\
## Resourcefulness\n\
Before claiming you cannot do something or lack access to a tool:\n\
- Use bash to check for installed CLIs (`which <tool>`, `command -v <tool>`).\n\
- Search for files, configs, and resources before saying they don't exist.\n\
- Read documentation, help output (`<tool> --help`), and man pages when unsure.\n\
- Try alternative approaches when the first attempt fails.\n\
Never say \"I don't have access\" or \"I can't\" without evidence. Investigate first.";

/// Tracks consecutive identical tool-call turns to detect doom loops.
///
/// Each turn's tool calls are hashed as a sorted set of `(name, input_json)` pairs.
/// When N consecutive turns produce the same hash, the tracker signals a doom loop.
struct DoomLoopTracker {
    /// Hash of the previous turn's tool calls, and its consecutive count.
    last_hash: Option<u64>,
    count: u32,
}

impl DoomLoopTracker {
    fn new() -> Self {
        Self {
            last_hash: None,
            count: 0,
        }
    }

    /// Hash a set of tool calls for the current turn. Tool calls are sorted by
    /// name so that ordering differences don't produce different hashes.
    fn hash_tool_calls(calls: &[ToolCall]) -> u64 {
        let mut sorted: Vec<(String, String)> = calls
            .iter()
            .map(|tc| (tc.name.clone(), tc.input.to_string()))
            .collect();
        sorted.sort();
        let mut hasher = DefaultHasher::new();
        for (name, input) in &sorted {
            name.hash(&mut hasher);
            input.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Record the current turn's tool calls hash. Returns `true` if a doom loop
    /// is detected (count >= threshold).
    fn record(&mut self, calls: &[ToolCall], threshold: u32) -> bool {
        let hash = Self::hash_tool_calls(calls);
        match self.last_hash {
            Some(prev) if prev == hash => {
                self.count += 1;
            }
            _ => {
                self.last_hash = Some(hash);
                self.count = 1;
            }
        }
        self.count >= threshold
    }
}

/// Output of an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    pub result: String,
    pub tool_calls_made: usize,
    pub tokens_used: TokenUsage,
    /// Structured output when the agent was configured with a response schema.
    /// Contains the validated JSON conforming to the schema.
    pub structured: Option<serde_json::Value>,
    /// Estimated cost in USD based on model pricing. `None` if the model is
    /// unknown or cost estimation is not available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_cost_usd: Option<f64>,
}

/// Runs an agent loop: LLM call → tool execution → repeat until done.
pub struct AgentRunner<P: LlmProvider> {
    provider: Arc<P>,
    name: String,
    system_prompt: String,
    tools: HashMap<String, Arc<dyn Tool>>,
    tool_defs: Vec<ToolDefinition>,
    max_turns: usize,
    max_tokens: u32,
    context_strategy: ContextStrategy,
    /// Token threshold at which to trigger summarization. `None` = no summarization.
    summarize_threshold: Option<u32>,
    /// Optional callback for streaming text output.
    on_text: Option<Arc<crate::llm::OnText>>,
    /// Optional callback for human-in-the-loop approval before tool execution.
    on_approval: Option<Arc<crate::llm::OnApproval>>,
    /// Optional timeout for individual tool executions.
    tool_timeout: Option<Duration>,
    /// Optional maximum byte size for tool output content. Oversized results
    /// are truncated with a `[truncated: N bytes omitted]` suffix.
    max_tool_output_bytes: Option<usize>,
    /// When set, a synthetic `respond` tool is injected with this JSON Schema.
    /// The agent calls `respond` to produce structured output conforming to the schema.
    structured_schema: Option<serde_json::Value>,
    /// Optional callback for structured agent events.
    on_event: Option<Arc<OnEvent>>,
    /// Guardrails applied to LLM calls and tool executions.
    guardrails: Vec<Arc<dyn Guardrail>>,
    /// Optional callback for interactive mode. When set and the LLM returns
    /// text without tool calls, the callback is invoked to get the next user
    /// message instead of returning immediately.
    on_input: Option<Arc<OnInput>>,
    /// Optional wall-clock deadline for the entire run. When set, the full
    /// `execute` call (all turns) is wrapped in `tokio::time::timeout`.
    run_timeout: Option<Duration>,
    /// Optional reasoning/thinking effort level for models that support it.
    reasoning_effort: Option<crate::llm::types::ReasoningEffort>,
    /// When true, inject a reflection prompt after tool results to encourage
    /// the agent to assess results before the next action (Reflexion/CRITIC pattern).
    enable_reflection: bool,
    /// When set, tool outputs exceeding this byte threshold are compressed
    /// via an LLM call that preserves factual content while removing redundancy.
    tool_output_compression_threshold: Option<usize>,
    /// When set, limits the number of tool definitions sent per LLM turn.
    /// Tools are selected based on recent usage and keyword relevance.
    max_tools_per_turn: Option<usize>,
    /// When set, pre-filters tool definitions based on query classification
    /// before dynamic selection. Reduces token usage for simple queries.
    tool_profile: Option<tool_filter::ToolProfile>,
    /// Maximum number of consecutive identical tool-call turns before the
    /// agent receives an error result instead of executing the tools. `None`
    /// disables doom loop detection.
    max_identical_tool_calls: Option<u32>,
    /// Declarative permission rules evaluated per tool call before the
    /// `on_approval` callback. `Allow` → execute, `Deny` → error result,
    /// `Ask` → fall through to `on_approval`.
    ///
    /// Wrapped in `RwLock` for interior mutability: learned rules from
    /// `AlwaysAllow`/`AlwaysDeny` are injected at runtime via `&self`.
    /// Lock is never held across `.await`.
    permission_rules: std::sync::RwLock<permission::PermissionRuleset>,
    /// Optional learned permissions for persisting AlwaysAllow/AlwaysDeny decisions.
    learned_permissions: Option<Arc<std::sync::Mutex<permission::LearnedPermissions>>>,
    /// Optional LSP manager for collecting diagnostics after file-modifying tools.
    lsp_manager: Option<Arc<crate::lsp::LspManager>>,
    /// Optional session pruning config. When set, old tool results are truncated
    /// before each LLM call to reduce token usage.
    session_prune_config: Option<pruner::SessionPruneConfig>,
    /// Optional memory store reference for pre-compaction flush.
    memory: Option<Arc<dyn Memory>>,
    /// When true, use recursive (cluster-then-summarize) summarization for
    /// long conversations instead of single-shot.
    enable_recursive_summarization: bool,
    /// When true, run memory consolidation at session end.
    consolidate_on_exit: bool,
    /// Observability verbosity level controlling span attribute recording.
    observability_mode: observability::ObservabilityMode,
    /// Hard limit on cumulative tokens (input + output) across all turns.
    /// When exceeded, the agent returns `Error::BudgetExceeded`.
    max_total_tokens: Option<u64>,
    /// Optional audit trail for recording untruncated agent decisions.
    audit_trail: Option<Arc<dyn AuditTrail>>,
    /// Optional user context for multi-tenant audit enrichment.
    audit_user_id: Option<String>,
    audit_tenant_id: Option<String>,
    /// Delegation chain for audit records (e.g., `["heartbit-agent"]` when acting on behalf of user).
    audit_delegation_chain: Vec<String>,
}

impl<P: LlmProvider> AgentRunner<P> {
    pub fn builder(provider: Arc<P>) -> AgentRunnerBuilder<P> {
        AgentRunnerBuilder {
            provider,
            name: String::new(),
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
            audit_trail: None,
            audit_user_id: None,
            audit_tenant_id: None,
            audit_delegation_chain: Vec::new(),
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
        self.permission_rules
            .read()
            .expect("permission rules lock poisoned")
            .evaluate(tool_name, input)
    }

    /// Check if the permission ruleset has any rules.
    fn has_permission_rules(&self) -> bool {
        !self
            .permission_rules
            .read()
            .expect("permission rules lock poisoned")
            .is_empty()
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(ref cb) = self.on_event {
            cb(event);
        }
    }

    /// Record an audit entry (best-effort). Failures are logged, never abort the agent.
    async fn audit(&self, record: AuditRecord) {
        if let Some(ref trail) = self.audit_trail
            && let Err(e) = trail.record(record).await
        {
            tracing::warn!(error = %e, "audit record failed");
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
        self.permission_rules
            .write()
            .expect("permission rules lock poisoned")
            .append_rules(&new_rules);
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
            let mut total_usage = TokenUsage::default();
            // Accumulate cost per-turn for accurate cascade pricing.
            let mut total_cost: f64 = 0.0;
            // Track recently used tool names (last 2 turns) for dynamic tool selection
            let mut recently_used_tools: Vec<String> = Vec::new();
            let mut doom_tracker = DoomLoopTracker::new();
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
                );
                // TTFT: wrap on_text to capture time-to-first-token
                let ttft_ms = Arc::new(std::sync::atomic::AtomicU64::new(0));
                let llm_result = async {
                    match &self.on_text {
                        Some(cb) => {
                            let ttft_ref = ttft_ms.clone();
                            let start = llm_start;
                            let inner_cb = cb.clone();
                            let wrapper: Box<crate::llm::OnText> = Box::new(move |text: &str| {
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
                let llm_latency_ms = llm_start.elapsed().as_millis() as u64;
                let time_to_first_token_ms = ttft_ms.load(std::sync::atomic::Ordering::Relaxed);
                // Record LLM call span attributes
                if mode.includes_metrics() {
                    llm_span.record("latency_ms", llm_latency_ms);
                    llm_span.record("ttft_ms", time_to_first_token_ms);
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
                let response = match llm_result {
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
                // Per-turn cost: prefer response.model (cascade) over static model_name()
                let turn_model = response
                    .model
                    .as_deref()
                    .or_else(|| self.provider.model_name());
                if let Some(model) = turn_model
                    && let Some(cost) =
                        crate::llm::pricing::estimate_cost(model, &response.usage)
                {
                    total_cost += cost;
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

                let tool_calls = response.tool_calls();

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
                    time_to_first_token_ms,
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
                    match g.post_llm(&response).await.map_err(|e| (e, total_usage))? {
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
                if let Some(threshold) = self.max_identical_tool_calls
                    && doom_tracker.record(&tool_calls, threshold)
                {
                    debug!(
                        agent = %self.name,
                        count = doom_tracker.count,
                        "doom loop detected, returning error results"
                    );
                    self.emit(AgentEvent::DoomLoopDetected {
                        agent: self.name.clone(),
                        turn: ctx.current_turn(),
                        consecutive_count: doom_tracker.count,
                        tool_names: tool_calls.iter().map(|tc| tc.name.clone()).collect(),
                    });
                    let results: Vec<ToolResult> = tool_calls
                        .iter()
                        .map(|tc| {
                            ToolResult::error(
                                tc.id.clone(),
                                format!(
                                    "Doom loop detected: identical tool calls repeated {} times \
                                 consecutively. Try a different approach.",
                                    doom_tracker.count
                                ),
                            )
                        })
                        .collect();
                    total_tool_calls += tool_calls.len();
                    ctx.add_tool_results(results);
                    continue;
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
                let mut results = self
                    .execute_tools_parallel(&allowed_calls, ctx.current_turn())
                    .instrument(tool_batch_span)
                    .await;
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
                    };
                    if let Err(e) = memory.store(entry).await {
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
        match crate::memory::pruning::prune_weak_entries(
            memory,
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
        match pipeline.run().await {
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
    fn select_tools_for_turn(
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
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

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
    fn find_closest_tool(&self, name: &str, max_distance: usize) -> Option<&str> {
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
    async fn execute_tools_parallel(&self, calls: &[ToolCall], turn: usize) -> Vec<ToolResult> {
        let call_ids: Vec<String> = calls.iter().map(|c| c.id.clone()).collect();
        let call_names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, call) in calls.iter().enumerate() {
            let tool = self.tools.get(&call.name).cloned().or_else(|| {
                self.find_closest_tool(&call.name, 2)
                    .and_then(|repaired_name| {
                        tracing::warn!(
                            agent = %self.name,
                            original = %call.name,
                            repaired = %repaired_name,
                            "tool name repaired via Levenshtein match"
                        );
                        self.tools.get(repaired_name).cloned()
                    })
            });
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
            join_set.spawn(
                async move {
                    let start = std::time::Instant::now();
                    let output = match tool {
                        Some(t) => match timeout {
                            Some(dur) => match tokio::time::timeout(dur, t.execute(input)).await {
                                Ok(result) => result,
                                Err(_) => Ok(ToolOutput::error(format!(
                                    "Tool execution timed out after {}s",
                                    dur.as_secs_f64()
                                ))),
                            },
                            None => t.execute(input).await,
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
                        user_id: None,
                        tenant_id: None,
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
            results_vec.push(tool_output_to_result(call_ids[idx].clone(), output));
        }

        results_vec
    }
}

fn tool_output_to_result(tool_use_id: String, output: ToolOutput) -> ToolResult {
    if output.is_error {
        ToolResult::error(tool_use_id, output.content)
    } else {
        ToolResult::success(tool_use_id, output.content)
    }
}

pub struct AgentRunnerBuilder<P: LlmProvider> {
    provider: Arc<P>,
    name: String,
    system_prompt: String,
    tools: Vec<Arc<dyn Tool>>,
    max_turns: usize,
    max_tokens: u32,
    context_strategy: Option<ContextStrategy>,
    summarize_threshold: Option<u32>,
    memory: Option<Arc<dyn Memory>>,
    knowledge_base: Option<Arc<dyn KnowledgeBase>>,
    on_text: Option<Arc<crate::llm::OnText>>,
    on_approval: Option<Arc<crate::llm::OnApproval>>,
    tool_timeout: Option<Duration>,
    max_tool_output_bytes: Option<usize>,
    structured_schema: Option<serde_json::Value>,
    on_event: Option<Arc<OnEvent>>,
    guardrails: Vec<Arc<dyn Guardrail>>,
    on_question: Option<Arc<OnQuestion>>,
    on_input: Option<Arc<OnInput>>,
    run_timeout: Option<Duration>,
    reasoning_effort: Option<crate::llm::types::ReasoningEffort>,
    enable_reflection: bool,
    tool_output_compression_threshold: Option<usize>,
    max_tools_per_turn: Option<usize>,
    tool_profile: Option<tool_filter::ToolProfile>,
    max_identical_tool_calls: Option<u32>,
    permission_rules: permission::PermissionRuleset,
    /// Instruction file contents to prepend to the system prompt.
    instruction_text: Option<String>,
    learned_permissions: Option<Arc<std::sync::Mutex<permission::LearnedPermissions>>>,
    lsp_manager: Option<Arc<crate::lsp::LspManager>>,
    session_prune_config: Option<pruner::SessionPruneConfig>,
    enable_recursive_summarization: bool,
    reflection_threshold: Option<u32>,
    consolidate_on_exit: bool,
    observability_mode: Option<observability::ObservabilityMode>,
    /// Optional workspace root for file tool path resolution and system prompt.
    workspace: Option<std::path::PathBuf>,
    /// Hard limit on cumulative tokens (input + output) across all turns.
    max_total_tokens: Option<u64>,
    /// Optional audit trail for recording untruncated agent decisions.
    audit_trail: Option<Arc<dyn AuditTrail>>,
    /// Optional user context for multi-tenant audit enrichment.
    audit_user_id: Option<String>,
    audit_tenant_id: Option<String>,
    /// Delegation chain for audit records (e.g., `["heartbit-agent"]` when acting OBO user).
    audit_delegation_chain: Vec<String>,
}

impl<P: LlmProvider> AgentRunnerBuilder<P> {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn context_strategy(mut self, strategy: ContextStrategy) -> Self {
        self.context_strategy = Some(strategy);
        self
    }

    /// Set the token threshold at which to trigger automatic summarization.
    pub fn summarize_threshold(mut self, threshold: u32) -> Self {
        self.summarize_threshold = Some(threshold);
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

    pub fn enable_reflection(mut self, enabled: bool) -> Self {
        self.enable_reflection = enabled;
        self
    }

    pub fn tool_output_compression_threshold(mut self, threshold: usize) -> Self {
        self.tool_output_compression_threshold = Some(threshold);
        self
    }

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

    /// Attach an audit trail for recording untruncated agent decisions.
    ///
    /// When set, every LLM response, tool call, tool result, run completion,
    /// run failure, and guardrail denial is recorded with full payloads.
    /// Recording is best-effort: failures are logged, never abort the agent.
    pub fn audit_trail(mut self, trail: Arc<dyn AuditTrail>) -> Self {
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

    /// Set the agent's workspace directory. When set, file tools resolve
    /// relative paths against this directory, BashTool starts here, and a
    /// workspace hint is appended to the system prompt.
    pub fn workspace(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.workspace = Some(path.into());
        self
    }

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
        if self.max_total_tokens == Some(0) {
            return Err(Error::Config("max_total_tokens must be at least 1".into()));
        }

        // Collect all tools, including memory and knowledge tools
        let mut all_tools = self.tools;
        let memory_ref = self.memory.clone();
        if let Some(memory) = self.memory {
            all_tools.extend(crate::memory::tools::memory_tools_with_reflection(
                memory,
                &self.name,
                self.reflection_threshold,
            ));
        }
        if let Some(kb) = self.knowledge_base {
            all_tools.extend(crate::knowledge::tools::knowledge_tools(kb));
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
                tracing::warn!(tool = %def.name, "duplicate tool name, keeping first registration");
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
            on_text: self.on_text,
            on_approval: self.on_approval,
            tool_timeout: self.tool_timeout,
            max_tool_output_bytes: self.max_tool_output_bytes,
            structured_schema: self.structured_schema,
            on_event: self.on_event,
            guardrails: self.guardrails,
            on_input: self.on_input,
            run_timeout: self.run_timeout,
            reasoning_effort: self.reasoning_effort,
            enable_reflection: self.enable_reflection,
            tool_output_compression_threshold: self.tool_output_compression_threshold,
            max_tools_per_turn: self.max_tools_per_turn,
            tool_profile: self.tool_profile,
            max_identical_tool_calls: self.max_identical_tool_calls,
            permission_rules: std::sync::RwLock::new(self.permission_rules),
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
            audit_trail: self.audit_trail,
            audit_user_id: self.audit_user_id,
            audit_tenant_id: self.audit_tenant_id,
            audit_delegation_chain: self.audit_delegation_chain,
        })
    }
}

/// Levenshtein edit distance between two strings (char-based, not byte-based).
use crate::util::levenshtein;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use serde_json::json;
    use std::sync::Mutex;

    // --- Mock LlmProvider ---

    struct MockProvider {
        responses: Mutex<Vec<CompletionResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<CompletionResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl LlmProvider for MockProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            let mut responses = self.responses.lock().expect("mock lock poisoned");
            if responses.is_empty() {
                return Err(Error::Agent("no more mock responses".into()));
            }
            Ok(responses.remove(0))
        }
    }

    // --- Mock Tool ---

    struct MockTool {
        def: ToolDefinition,
        response: String,
        is_error: bool,
    }

    impl MockTool {
        fn new(name: &str, response: &str) -> Self {
            Self {
                def: ToolDefinition {
                    name: name.into(),
                    description: format!("Mock tool {name}"),
                    input_schema: json!({"type": "object"}),
                },
                response: response.into(),
                is_error: false,
            }
        }

        fn failing(name: &str, error_msg: &str) -> Self {
            Self {
                def: ToolDefinition {
                    name: name.into(),
                    description: format!("Failing mock tool {name}"),
                    input_schema: json!({"type": "object"}),
                },
                response: error_msg.into(),
                is_error: true,
            }
        }
    }

    impl Tool for MockTool {
        fn definition(&self) -> ToolDefinition {
            self.def.clone()
        }

        fn execute(
            &self,
            _input: serde_json::Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>,
        > {
            let response = self.response.clone();
            let is_error = self.is_error;
            Box::pin(async move {
                if is_error {
                    Ok(ToolOutput::error(response))
                } else {
                    Ok(ToolOutput::success(response))
                }
            })
        }
    }

    #[tokio::test]
    async fn agent_returns_text_on_end_turn() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Hello!".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("You are helpful.")
            .build()
            .unwrap();

        let output = runner.execute("say hello").await.unwrap();
        assert_eq!(output.result, "Hello!");
        assert_eq!(output.tool_calls_made, 0);
        assert_eq!(output.tokens_used.input_tokens, 10);
    }

    #[tokio::test]
    async fn estimated_cost_usd_populated_for_known_model() {
        // A mock provider that returns a known Anthropic model name
        struct CostMockProvider;
        impl LlmProvider for CostMockProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "response".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage {
                        input_tokens: 1000,
                        output_tokens: 500,
                        ..Default::default()
                    },
                    model: None,
                })
            }
            fn model_name(&self) -> Option<&str> {
                Some("claude-sonnet-4-20250514")
            }
        }

        let provider = Arc::new(CostMockProvider);
        let runner = AgentRunner::builder(provider)
            .name("cost-test")
            .system_prompt("sys")
            .build()
            .unwrap();

        let output = runner.execute("task").await.unwrap();
        assert!(
            output.estimated_cost_usd.is_some(),
            "expected cost estimate for known model"
        );
        let cost = output.estimated_cost_usd.unwrap();
        // 1000 input @ $3/M = $0.003, 500 output @ $15/M = $0.0075 => $0.0105
        assert!(
            (cost - 0.0105).abs() < 0.001,
            "expected ~$0.0105, got: {cost}"
        );
    }

    #[tokio::test]
    async fn estimated_cost_usd_none_for_unknown_model() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text { text: "hi".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

        let output = runner.execute("task").await.unwrap();
        assert!(
            output.estimated_cost_usd.is_none(),
            "expected None for mock provider without model_name"
        );
    }

    #[tokio::test]
    async fn agent_executes_tool_and_continues() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "search".into(),
                    input: json!({"q": "rust"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 20,
                    output_tokens: 10,
                    ..Default::default()
                },
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Found it!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 30,
                    output_tokens: 15,
                    ..Default::default()
                },
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("You are helpful.")
            .tool(Arc::new(MockTool::new("search", "search results here")))
            .build()
            .unwrap();

        let output = runner.execute("find rust info").await.unwrap();
        assert_eq!(output.result, "Found it!");
        assert_eq!(output.tool_calls_made, 1);
        assert_eq!(output.tokens_used.input_tokens, 50);
        assert_eq!(output.tokens_used.output_tokens, 25);
    }

    #[tokio::test]
    async fn agent_errors_on_max_turns() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .max_turns(2)
            .build()
            .unwrap();

        let err = runner.execute("loop forever").await.unwrap_err();
        assert!(
            matches!(
                err,
                Error::WithPartialUsage {
                    ref source,
                    ..
                } if matches!(**source, Error::MaxTurnsExceeded(2))
            ),
            "expected MaxTurnsExceeded(2), got: {err:?}"
        );
    }

    #[tokio::test]
    async fn agent_error_carries_partial_token_usage() {
        // When max_turns is exceeded, the error should carry accumulated tokens.
        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: tool call → tool result loop
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cache_creation_input_tokens: 30,
                    cache_read_input_tokens: 0,
                    reasoning_tokens: 0,
                },
                model: None,
            },
            // Turn 2: another tool call → exceeds max_turns
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 120,
                    output_tokens: 60,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 25,
                    reasoning_tokens: 0,
                },
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .max_turns(2)
            .build()
            .unwrap();

        let err = runner.execute("loop forever").await.unwrap_err();
        let partial = err.partial_usage();
        assert_eq!(partial.input_tokens, 220, "100 + 120");
        assert_eq!(partial.output_tokens, 110, "50 + 60");
        assert_eq!(partial.cache_creation_input_tokens, 30);
        assert_eq!(partial.cache_read_input_tokens, 25);
    }

    #[tokio::test]
    async fn agent_returns_error_for_unknown_tool() {
        // Unknown tool now returns error as tool result (not hard crash)
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "nonexistent".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Sorry about that.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

        // No longer errors — sends error back to LLM, which recovers
        let output = runner.execute("use unknown tool").await.unwrap();
        assert_eq!(output.result, "Sorry about that.");
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn agent_executes_parallel_tool_calls() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "search".into(),
                        input: json!({"q": "a"}),
                    },
                    ContentBlock::ToolUse {
                        id: "c2".into(),
                        name: "read".into(),
                        input: json!({"path": "/tmp"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "found")))
            .tool(Arc::new(MockTool::new("read", "file content")))
            .build()
            .unwrap();

        let output = runner.execute("do both").await.unwrap();
        assert_eq!(output.result, "Done!");
        assert_eq!(output.tool_calls_made, 2);
    }

    #[tokio::test]
    async fn agent_errors_on_max_tokens() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "truncated...".into(),
            }],
            stop_reason: StopReason::MaxTokens,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

        let err = runner.execute("write a long essay").await.unwrap_err();
        assert!(
            matches!(
                err,
                Error::WithPartialUsage {
                    ref source,
                    ..
                } if matches!(**source, Error::Truncated)
            ),
            "expected Truncated, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn agent_handles_tool_error_result() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "failing".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Tool failed, but I recovered.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::failing("failing", "something broke")))
            .build()
            .unwrap();

        let output = runner.execute("try the tool").await.unwrap();
        assert_eq!(output.result, "Tool failed, but I recovered.");
    }

    #[tokio::test]
    async fn max_tokens_is_configurable() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text { text: "ok".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .max_tokens(8192)
            .build()
            .unwrap();

        // Just verify it builds and runs without error
        let output = runner.execute("test").await.unwrap();
        assert_eq!(output.result, "ok");
    }

    #[test]
    fn build_errors_on_empty_name() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider).system_prompt("sys").build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string().contains("agent name must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn build_errors_on_zero_max_turns() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .max_turns(0)
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string().contains("max_turns must be at least 1"),
            "error: {err}"
        );
    }

    #[test]
    fn build_errors_on_zero_max_tokens() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .max_tokens(0)
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string().contains("max_tokens must be at least 1"),
            "error: {err}"
        );
    }

    #[test]
    fn build_errors_on_sliding_window_with_summarize_threshold() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .context_strategy(ContextStrategy::SlidingWindow { max_tokens: 50000 })
            .summarize_threshold(8000)
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string()
                .contains("cannot use summarize_threshold with SlidingWindow"),
            "error: {err}"
        );
    }

    #[test]
    fn build_errors_on_on_input_with_structured_schema() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let on_input: Arc<OnInput> = Arc::new(|| Box::pin(async { None }));
        let result = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .on_input(on_input)
            .structured_schema(serde_json::json!({"type": "object"}))
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string().contains(
                "on_input (interactive mode) and structured_schema are mutually exclusive"
            ),
            "error: {err}"
        );
    }

    #[tokio::test]
    async fn instruction_text_prepended_to_system_prompt() {
        // CapturingProvider records the system prompt from the first LLM call.
        struct CapturingProvider {
            captured_system: Mutex<Option<String>>,
        }
        impl LlmProvider for CapturingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                *self.captured_system.lock().expect("lock") = Some(request.system.clone());
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "done".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            }
        }

        let provider = Arc::new(CapturingProvider {
            captured_system: Mutex::new(None),
        });
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("You are an agent.")
            .instruction_text("Be careful with files.")
            .build()
            .unwrap();
        let _output = runner.execute("task").await.unwrap();
        let system = provider
            .captured_system
            .lock()
            .expect("lock")
            .clone()
            .expect("system prompt should have been captured");
        assert!(
            system.contains("# Project Instructions"),
            "system prompt should contain instruction header: {system}"
        );
        assert!(
            system.contains("Be careful with files."),
            "system prompt should contain instruction text: {system}"
        );
        assert!(
            system.contains("You are an agent."),
            "system prompt should contain original prompt: {system}"
        );
        // Instructions come before the original prompt
        let instruction_pos = system.find("Be careful with files.").unwrap();
        let prompt_pos = system.find("You are an agent.").unwrap();
        assert!(
            instruction_pos < prompt_pos,
            "instructions should precede the original system prompt"
        );
    }

    #[test]
    fn instruction_text_empty_is_noop() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "done".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));
        // Empty instruction text should not modify the system prompt
        let builder = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("You are an agent.")
            .instruction_text(""); // empty → should be ignored
        // The internal instruction_text should be None (empty string filtered out)
        assert!(
            builder.instruction_text.is_none(),
            "empty instruction text should not be stored"
        );
        let _runner = builder.build().unwrap();
    }

    #[tokio::test]
    async fn context_strategy_builder_sets_sliding_window() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text { text: "ok".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .context_strategy(ContextStrategy::SlidingWindow { max_tokens: 50000 })
            .build()
            .unwrap();

        assert_eq!(
            runner.context_strategy,
            ContextStrategy::SlidingWindow { max_tokens: 50000 }
        );
    }

    #[tokio::test]
    async fn agent_uses_stream_complete_when_on_text_set() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct StreamTrackingProvider {
            stream_called: Arc<AtomicBool>,
        }

        impl LlmProvider for StreamTrackingProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "non-stream".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            }

            async fn stream_complete(
                &self,
                _request: CompletionRequest,
                on_text: &crate::llm::OnText,
            ) -> Result<CompletionResponse, Error> {
                self.stream_called.store(true, Ordering::SeqCst);
                on_text("streamed ");
                on_text("text");
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "streamed text".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            }
        }

        let stream_called = Arc::new(AtomicBool::new(false));
        let provider = Arc::new(StreamTrackingProvider {
            stream_called: stream_called.clone(),
        });

        let received = Arc::new(Mutex::new(Vec::<String>::new()));
        let received_clone = received.clone();
        let callback: Arc<crate::llm::OnText> = Arc::new(move |text: &str| {
            received_clone.lock().expect("lock").push(text.to_string());
        });

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .on_text(callback)
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert!(
            stream_called.load(Ordering::SeqCst),
            "stream_complete should have been called"
        );
        assert_eq!(output.result, "streamed text");

        let texts = received.lock().expect("lock");
        assert_eq!(*texts, vec!["streamed ", "text"]);
    }

    #[tokio::test]
    async fn context_strategy_defaults_to_unlimited() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text { text: "ok".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

        assert_eq!(runner.context_strategy, ContextStrategy::Unlimited);
    }

    #[tokio::test]
    async fn approval_callback_approves_tool_execution() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let approved = Arc::new(AtomicBool::new(false));
        let approved_clone = approved.clone();

        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({"q": "rust"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Found it!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let callback: Arc<crate::llm::OnApproval> = Arc::new(move |_calls| {
            approved_clone.store(true, Ordering::SeqCst);
            crate::llm::ApprovalDecision::Allow
        });

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "results")))
            .on_approval(callback)
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert!(
            approved.load(Ordering::SeqCst),
            "approval callback was called"
        );
        assert_eq!(output.result, "Found it!");
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn approval_callback_denies_tool_execution() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({"q": "rust"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // After denial, LLM responds with text instead
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "I understand, I won't execute that.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let callback: Arc<crate::llm::OnApproval> =
            Arc::new(|_calls| crate::llm::ApprovalDecision::Deny);

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "results")))
            .on_approval(callback)
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert_eq!(output.result, "I understand, I won't execute that.");
        // Tool call is counted even though denied (the LLM made the call)
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn approval_callback_receives_correct_tool_calls() {
        let received_calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let received_clone = received_calls.clone();

        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "search".into(),
                        input: json!({"q": "rust"}),
                    },
                    ContentBlock::ToolUse {
                        id: "c2".into(),
                        name: "read".into(),
                        input: json!({"path": "/tmp"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let callback: Arc<crate::llm::OnApproval> = Arc::new(move |calls| {
            let names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
            received_clone.lock().expect("lock").extend(names);
            crate::llm::ApprovalDecision::Allow
        });

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "found")))
            .tool(Arc::new(MockTool::new("read", "content")))
            .on_approval(callback)
            .build()
            .unwrap();

        runner.execute("test").await.unwrap();

        let calls = received_calls.lock().expect("lock");
        assert_eq!(*calls, vec!["search", "read"]);
    }

    #[tokio::test]
    async fn tool_timeout_returns_error_to_llm() {
        // A slow tool should time out and return an error result to the LLM
        struct SlowTool;
        impl Tool for SlowTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "slow_tool".into(),
                    description: "Takes forever".into(),
                    input_schema: json!({"type": "object"}),
                }
            }
            fn execute(
                &self,
                _input: serde_json::Value,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>,
            > {
                Box::pin(async {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    Ok(ToolOutput::success("should never reach here"))
                })
            }
        }

        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "slow_tool".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Tool timed out, moving on.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(SlowTool))
            .tool_timeout(std::time::Duration::from_millis(50))
            .build()
            .unwrap();

        let output = runner.execute("run slow tool").await.unwrap();
        assert_eq!(output.result, "Tool timed out, moving on.");
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn tool_timeout_does_not_affect_fast_tools() {
        // A fast tool should complete normally even with a timeout set
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Got results!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "search results")))
            .tool_timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        let output = runner.execute("search").await.unwrap();
        assert_eq!(output.result, "Got results!");
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn no_tool_timeout_allows_unlimited_execution() {
        // Without tool_timeout, tools run without a timeout (backward compatible)
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .build()
            .unwrap();

        // No tool_timeout set — should work as before
        let output = runner.execute("test").await.unwrap();
        assert_eq!(output.result, "Done!");
    }

    #[tokio::test]
    async fn no_approval_callback_executes_tools_directly() {
        // Without on_approval, tools execute without any gate
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert_eq!(output.result, "Done!");
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn schema_validation_rejects_bad_input() {
        // Tool with a strict schema requiring a "query" string
        struct StrictTool;
        impl Tool for StrictTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "search".into(),
                    description: "Search".into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }),
                }
            }
            fn execute(
                &self,
                _input: serde_json::Value,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>,
            > {
                Box::pin(async { Ok(ToolOutput::success("should not be called")) })
            }
        }

        // LLM sends invalid input (missing required "query"), then corrects
        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: LLM tries to call search with bad input
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({"wrong_field": 42}), // Missing "query"
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: LLM sees validation error, responds with text
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "I see the validation error.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(StrictTool))
            .build()
            .unwrap();

        let output = runner.execute("search for something").await.unwrap();
        // Agent gets error from validation, then LLM responds
        assert_eq!(output.result, "I see the validation error.");
        assert_eq!(output.tool_calls_made, 1); // The call was counted
    }

    #[tokio::test]
    async fn large_tool_output_is_truncated() {
        // Tool that returns a very large result
        struct BigTool;
        impl Tool for BigTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "big".into(),
                    description: "Returns big output".into(),
                    input_schema: json!({"type": "object"}),
                }
            }
            fn execute(
                &self,
                _input: serde_json::Value,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>,
            > {
                Box::pin(async { Ok(ToolOutput::success("x".repeat(10_000))) })
            }
        }

        // Capture what the LLM receives by checking the second response
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "big".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Got truncated result.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(BigTool))
            .max_tool_output_bytes(500)
            .build()
            .unwrap();

        let output = runner.execute("get big data").await.unwrap();
        assert_eq!(output.result, "Got truncated result.");
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn structured_output_extracts_respond_tool() {
        // When structured_schema is set, __respond__ tool call returns structured output
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer", "confidence"]
        });

        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "c1".into(),
                name: "__respond__".into(),
                input: json!({"answer": "42", "confidence": 0.95}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage {
                input_tokens: 20,
                output_tokens: 15,
                ..Default::default()
            },
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("You are helpful.")
            .structured_schema(schema)
            .build()
            .unwrap();

        let output = runner.execute("what is the answer?").await.unwrap();
        assert!(output.structured.is_some());
        let structured = output.structured.unwrap();
        assert_eq!(structured["answer"], "42");
        assert_eq!(structured["confidence"], 0.95);
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn structured_output_none_without_schema() {
        // Without structured_schema, output.structured is always None
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Hello!".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert!(output.structured.is_none());
    }

    #[tokio::test]
    async fn structured_output_allows_real_tools_first() {
        // Agent uses a regular tool, then calls __respond__ on the next turn
        let schema = json!({
            "type": "object",
            "properties": { "result": {"type": "string"} },
            "required": ["result"]
        });

        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: Use a real tool
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({"q": "data"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: Call __respond__ with structured output
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "__respond__".into(),
                    input: json!({"result": "found it"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "search results")))
            .structured_schema(schema)
            .build()
            .unwrap();

        let output = runner.execute("find data").await.unwrap();
        assert!(output.structured.is_some());
        assert_eq!(output.structured.unwrap()["result"], "found it");
        // 1 real tool call + 1 __respond__ call
        assert_eq!(output.tool_calls_made, 2);
    }

    #[test]
    fn structured_schema_injects_respond_tool_definition() {
        let schema = json!({
            "type": "object",
            "properties": { "answer": {"type": "string"} }
        });

        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .structured_schema(schema.clone())
            .build()
            .unwrap();

        // __respond__ should be in tool_defs but NOT in the tools HashMap
        assert!(runner.tool_defs.iter().any(|d| d.name == "__respond__"));
        assert!(!runner.tools.contains_key("__respond__"));
        let respond_def = runner
            .tool_defs
            .iter()
            .find(|d| d.name == "__respond__")
            .unwrap();
        assert_eq!(respond_def.input_schema, schema);
    }

    #[tokio::test]
    async fn structured_output_counts_all_tool_calls_in_respond_turn() {
        // When __respond__ appears alongside other tool calls, ALL are counted
        let schema = json!({
            "type": "object",
            "properties": { "result": {"type": "string"} }
        });

        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![
                ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({"q": "data"}),
                },
                ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "__respond__".into(),
                    input: json!({"result": "done"}),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "results")))
            .structured_schema(schema)
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert!(output.structured.is_some());
        // Both tool calls in the turn are counted (search + __respond__)
        assert_eq!(output.tool_calls_made, 2);
    }

    #[tokio::test]
    async fn structured_output_max_turns_when_respond_never_called() {
        // When structured_schema is set but LLM never calls __respond__,
        // the agent exhausts turns and returns MaxTurnsExceeded
        let schema = json!({
            "type": "object",
            "properties": { "result": {"type": "string"} }
        });

        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "results")))
            .structured_schema(schema)
            .max_turns(2)
            .build()
            .unwrap();

        let err = runner.execute("test").await.unwrap_err();
        assert!(
            matches!(
                err,
                Error::WithPartialUsage {
                    ref source,
                    ..
                } if matches!(**source, Error::MaxTurnsExceeded(2))
            ),
            "expected MaxTurnsExceeded(2), got: {err:?}"
        );
    }

    #[test]
    fn no_respond_tool_without_schema() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

        assert!(!runner.tool_defs.iter().any(|d| d.name == "__respond__"));
    }

    #[tokio::test]
    async fn small_tool_output_not_truncated_with_limit() {
        // When max_tool_output_bytes is set but output is small, no truncation
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "small result")))
            .max_tool_output_bytes(1000)
            .build()
            .unwrap();

        let output = runner.execute("search").await.unwrap();
        assert_eq!(output.result, "Done!");
    }

    #[test]
    fn agent_output_roundtrips() {
        let output = AgentOutput {
            result: "Hello!".into(),
            tool_calls_made: 3,
            tokens_used: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
            structured: Some(json!({"answer": "42"})),
            estimated_cost_usd: Some(0.0342),
        };
        let json_str = serde_json::to_string(&output).unwrap();
        let parsed: AgentOutput = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.result, "Hello!");
        assert_eq!(parsed.tool_calls_made, 3);
        assert_eq!(parsed.tokens_used.input_tokens, 100);
        assert_eq!(parsed.structured, Some(json!({"answer": "42"})));
        assert_eq!(parsed.estimated_cost_usd, Some(0.0342));
    }

    #[test]
    fn agent_output_structured_none_serializes() {
        let output = AgentOutput {
            result: "ok".into(),
            tool_calls_made: 0,
            tokens_used: TokenUsage::default(),
            structured: None,
            estimated_cost_usd: None,
        };
        let json_str = serde_json::to_string(&output).unwrap();
        let parsed: AgentOutput = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.structured.is_none());
    }

    #[tokio::test]
    async fn structured_output_errors_when_llm_ignores_respond() {
        // When structured_schema is set but LLM returns text without calling
        // __respond__, the agent should return an error (not silently succeed
        // with structured: None).
        let schema = json!({
            "type": "object",
            "properties": { "answer": {"type": "string"} },
            "required": ["answer"]
        });

        let provider = Arc::new(MockProvider::new(vec![
            // LLM ignores __respond__ and returns plain text
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Here is the answer.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .structured_schema(schema)
            .build()
            .unwrap();

        let err = runner.execute("test").await.unwrap_err();
        assert!(
            err.to_string().contains("__respond__"),
            "error should mention __respond__: {err}"
        );
    }

    #[tokio::test]
    async fn structured_output_does_not_force_tool_choice() {
        // Verify that structured_schema does NOT force ToolChoice::Any —
        // the LLM should freely choose tools. The __respond__ tool injection
        // plus the guard against plain-text responses is sufficient.
        use std::sync::atomic::{AtomicBool, Ordering};

        struct ToolChoiceTracker {
            tool_choice_any_seen: Arc<AtomicBool>,
        }

        impl LlmProvider for ToolChoiceTracker {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                if request.tool_choice == Some(crate::llm::types::ToolChoice::Any) {
                    self.tool_choice_any_seen.store(true, Ordering::SeqCst);
                }
                Ok(CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "__respond__".into(),
                        input: json!({"answer": "42"}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                    model: None,
                })
            }
        }

        let seen = Arc::new(AtomicBool::new(false));
        let provider = Arc::new(ToolChoiceTracker {
            tool_choice_any_seen: seen.clone(),
        });

        let schema = json!({
            "type": "object",
            "properties": { "answer": {"type": "string"} }
        });

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .structured_schema(schema)
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert!(
            !seen.load(Ordering::SeqCst),
            "tool_choice should NOT be forced to Any"
        );
        assert!(
            output.structured.is_some(),
            "structured output should still work"
        );
    }

    #[tokio::test]
    async fn respond_tool_skips_co_submitted_real_tools() {
        // When __respond__ appears alongside a real tool call in the same turn,
        // the real tool should NOT be executed (early return on __respond__).
        use std::sync::atomic::{AtomicBool, Ordering};

        let tool_executed = Arc::new(AtomicBool::new(false));
        let tool_executed_clone = tool_executed.clone();

        struct TrackingTool {
            executed: Arc<AtomicBool>,
        }
        impl Tool for TrackingTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "real_tool".into(),
                    description: "A real tool".into(),
                    input_schema: json!({"type": "object"}),
                }
            }
            fn execute(
                &self,
                _input: serde_json::Value,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>,
            > {
                self.executed.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(ToolOutput::success("done")) })
            }
        }

        let provider = Arc::new(MockProvider::new(vec![
            // LLM returns both __respond__ and real_tool in same turn
            CompletionResponse {
                content: vec![
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "real_tool".into(),
                        input: json!({}),
                    },
                    ContentBlock::ToolUse {
                        id: "c2".into(),
                        name: "__respond__".into(),
                        input: json!({"answer": "42"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let schema = json!({
            "type": "object",
            "properties": { "answer": {"type": "string"} }
        });

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(TrackingTool {
                executed: tool_executed_clone,
            }))
            .structured_schema(schema)
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();

        assert!(
            output.structured.is_some(),
            "should return structured output"
        );
        assert_eq!(output.tool_calls_made, 2, "should count both tool calls");
        assert!(
            !tool_executed.load(Ordering::SeqCst),
            "real_tool should NOT have been executed when __respond__ is present"
        );
    }

    #[tokio::test]
    async fn structured_output_validated_against_schema() {
        // When __respond__ output doesn't match schema, the agent should feed
        // a validation error back to the LLM so it can self-correct.
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["answer", "confidence"]
        });

        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: LLM responds with invalid output (missing "confidence")
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "__respond__".into(),
                    input: json!({"answer": "42"}), // missing required "confidence"
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: LLM corrects itself with valid output
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "__respond__".into(),
                    input: json!({"answer": "42", "confidence": 0.95}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .structured_schema(schema)
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert!(output.structured.is_some());
        assert_eq!(output.structured.unwrap()["confidence"], 0.95);
        // First invalid call + second valid call
        assert_eq!(output.tool_calls_made, 2);
    }

    #[tokio::test]
    async fn structured_output_validation_wrong_type() {
        // Validate that type mismatches are caught (string where number expected)
        let schema = json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            },
            "required": ["count"]
        });

        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: Wrong type
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "__respond__".into(),
                    input: json!({"count": "not a number"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: Corrected
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "__respond__".into(),
                    input: json!({"count": 42}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .structured_schema(schema)
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert_eq!(output.structured.unwrap()["count"], 42);
    }

    #[tokio::test]
    async fn structured_output_valid_on_first_try() {
        // When __respond__ output matches schema on first try, no retry needed
        let schema = json!({
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            },
            "required": ["result"]
        });

        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "c1".into(),
                name: "__respond__".into(),
                input: json!({"result": "hello"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .structured_schema(schema)
            .build()
            .unwrap();

        let output = runner.execute("test").await.unwrap();
        assert_eq!(output.structured.unwrap()["result"], "hello");
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn summarization_tokens_accumulated_in_total_usage() {
        // Verify that when summarization triggers, the summary LLM call's tokens
        // are included in the final AgentOutput.tokens_used.
        //
        // Setup: summarize_threshold=1 forces summarization after every tool round.
        // We need: turn 1 (tool call) → summarization → turn 2 (text response).
        // The mock provides 3 responses: tool call, summary, final text.
        //
        // However, the summarization guard requires ctx.message_count() > 5 and
        // ctx.needs_compaction(threshold). With threshold=1, needs_compaction
        // will be true as soon as any tokens accumulate. But message_count() > 5
        // requires at least 6 messages: initial user + (assistant + tool_results) * N.
        // After turn 1: user + assistant + tool_results = 3 messages. Not enough.
        // After turn 2: 3 + assistant + tool_results = 5 messages. Not enough (need >5).
        // After turn 3: 5 + assistant + tool_results = 7 messages. This triggers it!
        //
        // So we need: 3 tool rounds + 1 summary + 1 final text = 5 mock responses.
        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: tool call
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                model: None,
            },
            // Turn 2: tool call
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                model: None,
            },
            // Turn 3: tool call (after this, message_count > 5, triggers summarization)
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c3".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                model: None,
            },
            // Summary LLM call (triggered by summarize_threshold)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Summary of conversation so far.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cache_creation_input_tokens: 25,
                    cache_read_input_tokens: 10,
                    reasoning_tokens: 0,
                },
                model: None,
            },
            // Turn 4: final text response
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Final answer.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .summarize_threshold(1) // trigger summarization at minimal threshold
            .max_turns(10)
            .build()
            .unwrap();

        let output = runner.execute("test task").await.unwrap();
        assert_eq!(output.result, "Final answer.");
        // Total: 4 agent turns (10 each) + 1 summary call (100)
        assert_eq!(output.tokens_used.input_tokens, 10 + 10 + 10 + 100 + 10);
        // Total: 4 agent turns (5 each) + 1 summary call (50)
        assert_eq!(output.tokens_used.output_tokens, 5 + 5 + 5 + 50 + 5);
        // Cache tokens come only from the summary call
        assert_eq!(output.tokens_used.cache_creation_input_tokens, 25);
        assert_eq!(output.tokens_used.cache_read_input_tokens, 10);
    }

    #[test]
    fn knowledge_base_adds_search_tool() {
        use crate::knowledge::in_memory::InMemoryKnowledgeBase;

        let kb: Arc<dyn crate::knowledge::KnowledgeBase> = Arc::new(InMemoryKnowledgeBase::new());
        let provider = Arc::new(MockProvider::new(vec![]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .knowledge(kb)
            .build()
            .unwrap();

        assert!(
            runner
                .tool_defs
                .iter()
                .any(|d| d.name == "knowledge_search"),
            "agent should have knowledge_search tool"
        );
    }

    #[tokio::test]
    async fn on_event_emits_run_started_and_completed() {
        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let events_clone = events.clone();

        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Done.".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test-agent")
            .system_prompt("sys")
            .on_event(Arc::new(move |e| {
                events_clone.lock().unwrap().push(e);
            }))
            .build()
            .unwrap();

        runner.execute("hello").await.unwrap();

        let events = events.lock().unwrap();
        assert!(
            events.len() >= 4,
            "expected at least 4 events, got {}",
            events.len()
        );

        // First event: RunStarted
        match &events[0] {
            AgentEvent::RunStarted { agent, task } => {
                assert_eq!(agent, "test-agent");
                assert_eq!(task, "hello");
            }
            other => panic!("expected RunStarted, got: {other:?}"),
        }

        // Second event: TurnStarted
        match &events[1] {
            AgentEvent::TurnStarted { agent, turn, .. } => {
                assert_eq!(agent, "test-agent");
                assert_eq!(*turn, 1);
            }
            other => panic!("expected TurnStarted, got: {other:?}"),
        }

        // Third event: LlmResponse
        match &events[2] {
            AgentEvent::LlmResponse {
                agent,
                turn,
                tool_call_count,
                ..
            } => {
                assert_eq!(agent, "test-agent");
                assert_eq!(*turn, 1);
                assert_eq!(*tool_call_count, 0);
            }
            other => panic!("expected LlmResponse, got: {other:?}"),
        }

        // Last event: RunCompleted
        match events.last().unwrap() {
            AgentEvent::RunCompleted {
                agent,
                tool_calls_made,
                ..
            } => {
                assert_eq!(agent, "test-agent");
                assert_eq!(*tool_calls_made, 0);
            }
            other => panic!("expected RunCompleted, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn on_event_emits_tool_call_events() {
        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let events_clone = events.clone();

        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Result.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("worker")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "found it")))
            .on_event(Arc::new(move |e| {
                events_clone.lock().unwrap().push(e);
            }))
            .build()
            .unwrap();

        runner.execute("find stuff").await.unwrap();

        let events = events.lock().unwrap();
        let tool_started: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolCallStarted { .. }))
            .collect();
        let tool_completed: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolCallCompleted { .. }))
            .collect();

        assert_eq!(tool_started.len(), 1, "expected 1 ToolCallStarted");
        assert_eq!(tool_completed.len(), 1, "expected 1 ToolCallCompleted");

        match &tool_started[0] {
            AgentEvent::ToolCallStarted {
                tool_name,
                tool_call_id,
                ..
            } => {
                assert_eq!(tool_name, "search");
                assert_eq!(tool_call_id, "call-1");
            }
            _ => unreachable!(),
        }

        match &tool_completed[0] {
            AgentEvent::ToolCallCompleted {
                tool_name,
                is_error,
                ..
            } => {
                assert_eq!(tool_name, "search");
                assert!(!is_error);
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn on_event_emits_run_failed_on_max_turns() {
        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let events_clone = events.clone();

        // Provider that always returns tool calls — will exceed max_turns
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "search".into(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("limited")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "found")))
            .max_turns(1)
            .on_event(Arc::new(move |e| {
                events_clone.lock().unwrap().push(e);
            }))
            .build()
            .unwrap();

        let result = runner.execute("go").await;
        assert!(result.is_err());

        let events = events.lock().unwrap();
        let run_failed: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::RunFailed { .. }))
            .collect();
        assert_eq!(run_failed.len(), 1, "expected 1 RunFailed event");

        match &run_failed[0] {
            AgentEvent::RunFailed { agent, error, .. } => {
                assert_eq!(agent, "limited");
                assert!(error.contains("Max turns"), "error: {error}");
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn no_events_when_callback_not_set() {
        // Just verify execution works fine without on_event
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Done.".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("quiet")
            .system_prompt("sys")
            .build()
            .unwrap();

        let output = runner.execute("hello").await.unwrap();
        assert_eq!(output.result, "Done.");
    }

    // --- Guardrail tests ---

    use crate::agent::guardrail::{GuardAction, Guardrail};

    struct SystemPromptInjector {
        suffix: String,
    }

    impl Guardrail for SystemPromptInjector {
        fn pre_llm(
            &self,
            request: &mut CompletionRequest,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + '_>>
        {
            request.system = format!("{} {}", request.system, self.suffix);
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    async fn pre_llm_guardrail_modifies_request() {
        struct CapturingProvider {
            captured_system: Mutex<Option<String>>,
        }

        impl LlmProvider for CapturingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                *self.captured_system.lock().unwrap() = Some(request.system);
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text { text: "ok".into() }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            }
        }

        let provider = Arc::new(CapturingProvider {
            captured_system: Mutex::new(None),
        });

        let guardrail: Arc<dyn Guardrail> = Arc::new(SystemPromptInjector {
            suffix: "SAFETY_NOTICE".into(),
        });

        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("You are helpful.")
            .guardrail(guardrail)
            .build()
            .unwrap();

        runner.execute("hello").await.unwrap();

        let captured = provider.captured_system.lock().unwrap().clone().unwrap();
        assert!(
            captured.contains("SAFETY_NOTICE"),
            "system prompt should contain injected suffix: {captured}"
        );
    }

    #[tokio::test]
    async fn post_llm_guardrail_denies_response() {
        // First response denied, second allowed. Should consume 2 turns.
        struct CountingProvider {
            call_count: Mutex<usize>,
        }

        impl LlmProvider for CountingProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                let mut count = self.call_count.lock().unwrap();
                *count += 1;
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: format!("Response #{count}"),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            }
        }

        // Deny only the first call
        struct DenyOnce {
            denied: Mutex<bool>,
        }

        impl Guardrail for DenyOnce {
            fn post_llm(
                &self,
                _response: &crate::llm::types::CompletionResponse,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<GuardAction, Error>> + Send + '_>,
            > {
                Box::pin(async {
                    let mut denied = self.denied.lock().unwrap();
                    if !*denied {
                        *denied = true;
                        Ok(GuardAction::deny("unsafe content"))
                    } else {
                        Ok(GuardAction::Allow)
                    }
                })
            }
        }

        let provider = Arc::new(CountingProvider {
            call_count: Mutex::new(0),
        });

        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .guardrail(Arc::new(DenyOnce {
                denied: Mutex::new(false),
            }))
            .max_turns(3)
            .build()
            .unwrap();

        let output = runner.execute("hello").await.unwrap();
        // Second response should be the result (first was denied)
        assert_eq!(output.result, "Response #2");
        // LLM called twice
        assert_eq!(*provider.call_count.lock().unwrap(), 2);
    }

    #[tokio::test]
    async fn post_llm_denial_maintains_alternating_roles() {
        // Verify that post_llm denial inserts assistant placeholder before
        // user feedback, so the Anthropic alternating-roles invariant holds.
        use crate::llm::types::{CompletionResponse, Role};

        struct RecordingProvider {
            call_count: Mutex<usize>,
            last_messages: Mutex<Vec<Role>>,
        }

        impl LlmProvider for RecordingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                let mut count = self.call_count.lock().unwrap();
                *count += 1;
                // Record message roles from each request
                let roles: Vec<Role> = request.messages.iter().map(|m| m.role.clone()).collect();
                *self.last_messages.lock().unwrap() = roles;
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: format!("Response #{count}"),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            }
        }

        struct DenyOnce {
            denied: Mutex<bool>,
        }

        impl Guardrail for DenyOnce {
            fn post_llm(
                &self,
                _response: &CompletionResponse,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<GuardAction, Error>> + Send + '_>,
            > {
                Box::pin(async {
                    let mut denied = self.denied.lock().unwrap();
                    if !*denied {
                        *denied = true;
                        Ok(GuardAction::deny("blocked"))
                    } else {
                        Ok(GuardAction::Allow)
                    }
                })
            }
        }

        let provider = Arc::new(RecordingProvider {
            call_count: Mutex::new(0),
            last_messages: Mutex::new(vec![]),
        });

        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .guardrail(Arc::new(DenyOnce {
                denied: Mutex::new(false),
            }))
            .max_turns(3)
            .build()
            .unwrap();

        let output = runner.execute("hello").await.unwrap();
        assert_eq!(output.result, "Response #2");

        // The second LLM call should have alternating user/assistant/user roles
        let roles = provider.last_messages.lock().unwrap();
        for pair in roles.windows(2) {
            assert_ne!(
                pair[0],
                pair[1],
                "Found consecutive messages with same role: {:?}",
                roles.as_slice()
            );
        }
    }

    struct DenyingPreTool {
        blocked_tool: String,
        reason: String,
    }

    impl Guardrail for DenyingPreTool {
        fn pre_tool(
            &self,
            call: &crate::llm::types::ToolCall,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<GuardAction, Error>> + Send + '_>,
        > {
            let result = if call.name == self.blocked_tool {
                GuardAction::deny(&self.reason)
            } else {
                GuardAction::Allow
            };
            Box::pin(async move { Ok(result) })
        }
    }

    #[tokio::test]
    async fn pre_tool_guardrail_denies_specific_tool() {
        // Provider: turn 1 calls "dangerous" tool, gets denied error, turn 2 answers
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "dangerous".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "OK, skipping dangerous tool.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("dangerous", "should not run")))
            .guardrail(Arc::new(DenyingPreTool {
                blocked_tool: "dangerous".into(),
                reason: "tool is blocked".into(),
            }))
            .build()
            .unwrap();

        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "OK, skipping dangerous tool.");
        assert_eq!(output.tool_calls_made, 1); // denied call is counted
    }

    struct RedactingPostTool;

    impl Guardrail for RedactingPostTool {
        fn post_tool(
            &self,
            _call: &crate::llm::types::ToolCall,
            output: &mut crate::tool::ToolOutput,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + '_>>
        {
            output.content = output.content.replace("SECRET", "[REDACTED]");
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    async fn post_tool_guardrail_redacts_output() {
        // Provider: turn 1 calls a tool, turn 2 uses the (redacted) result
        struct CapturingProvider {
            responses: Mutex<Vec<CompletionResponse>>,
            tool_results_seen: Mutex<Vec<String>>,
        }

        impl LlmProvider for CapturingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                // Capture tool results from messages
                for msg in &request.messages {
                    for block in &msg.content {
                        if let ContentBlock::ToolResult { content, .. } = block {
                            self.tool_results_seen.lock().unwrap().push(content.clone());
                        }
                    }
                }

                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Err(Error::Agent("no more responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let provider = Arc::new(CapturingProvider {
            responses: Mutex::new(vec![
                CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "search".into(),
                        input: json!({}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                    model: None,
                },
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                },
            ]),
            tool_results_seen: Mutex::new(vec![]),
        });

        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "Found SECRET data")))
            .guardrail(Arc::new(RedactingPostTool))
            .build()
            .unwrap();

        runner.execute("search").await.unwrap();

        let results = provider.tool_results_seen.lock().unwrap();
        assert!(
            results.iter().any(|r| r.contains("[REDACTED]")),
            "tool result should be redacted: {results:?}"
        );
        assert!(
            !results.iter().any(|r| r.contains("SECRET")),
            "tool result should not contain SECRET: {results:?}"
        );
    }

    #[tokio::test]
    async fn multiple_guardrails_compose() {
        // First guardrail allows, second denies
        struct AllowGuardrail;
        impl Guardrail for AllowGuardrail {}

        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Denied.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .guardrail(Arc::new(AllowGuardrail))
            .guardrail(Arc::new(DenyingPreTool {
                blocked_tool: "search".into(),
                reason: "blocked by second guardrail".into(),
            }))
            .build()
            .unwrap();

        let output = runner.execute("search").await.unwrap();
        assert_eq!(output.result, "Denied.");
    }

    #[tokio::test]
    async fn guardrail_error_aborts_run() {
        struct ErrorGuardrail;
        impl Guardrail for ErrorGuardrail {
            fn pre_llm(
                &self,
                _request: &mut CompletionRequest,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + '_>>
            {
                Box::pin(async { Err(Error::Guardrail("fatal check failed".into())) })
            }
        }

        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "should not reach".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .guardrail(Arc::new(ErrorGuardrail))
            .build()
            .unwrap();

        let err = runner.execute("hello").await.unwrap_err();
        assert!(
            err.to_string().contains("fatal check failed"),
            "error should contain guardrail message: {err}"
        );
    }

    #[tokio::test]
    async fn on_approval_and_pre_tool_compose() {
        // on_approval approves, but pre_tool denies a specific tool
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "safe".into(),
                        input: json!({}),
                    },
                    ContentBlock::ToolUse {
                        id: "c2".into(),
                        name: "dangerous".into(),
                        input: json!({}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Used safe, dangerous blocked.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let approval: Arc<crate::llm::OnApproval> =
            Arc::new(|_calls: &[_]| crate::llm::ApprovalDecision::Allow);

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("safe", "safe result")))
            .tool(Arc::new(MockTool::new("dangerous", "should not run")))
            .on_approval(approval)
            .guardrail(Arc::new(DenyingPreTool {
                blocked_tool: "dangerous".into(),
                reason: "blocked".into(),
            }))
            .build()
            .unwrap();

        let output = runner.execute("do both").await.unwrap();
        assert_eq!(output.result, "Used safe, dangerous blocked.");
        assert_eq!(output.tool_calls_made, 2);
    }

    #[tokio::test]
    async fn no_guardrails_unchanged_behavior() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Found it.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .build()
            .unwrap();

        let output = runner.execute("search").await.unwrap();
        assert_eq!(output.result, "Found it.");
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn on_input_continues_conversation() {
        // First LLM response: text (triggers on_input)
        // Second LLM response: text (triggers on_input, returns None → end)
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Hello! How can I help?".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Sure, here you go.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let on_input: Arc<OnInput> = Arc::new(move || {
            let count = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move {
                match count {
                    0 => Some("Tell me more.".into()),
                    _ => None, // End the session
                }
            })
        });

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .max_turns(10)
            .on_input(on_input)
            .build()
            .unwrap();

        let output = runner.execute("Hi").await.unwrap();
        // The final response text should be from the second LLM call
        assert_eq!(output.result, "Sure, here you go.");
        // on_input was called twice: first returned Some, second returned None
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn on_input_empty_string_ends_session() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Response.".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let on_input: Arc<OnInput> = Arc::new(|| {
            Box::pin(async { Some("   ".into()) }) // whitespace-only → treated as empty
        });

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .max_turns(10)
            .on_input(on_input)
            .build()
            .unwrap();

        let output = runner.execute("Hi").await.unwrap();
        assert_eq!(output.result, "Response.");
    }

    #[tokio::test]
    async fn post_tool_guardrail_error_emits_event() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct FailingPostTool;
        impl Guardrail for FailingPostTool {
            fn post_tool(
                &self,
                _call: &crate::llm::types::ToolCall,
                _output: &mut ToolOutput,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + '_>>
            {
                Box::pin(async { Err(Error::Guardrail("output too large".into())) })
            }
        }

        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text { text: "OK.".into() }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let saw_post_tool_event = Arc::new(AtomicBool::new(false));
        let saw_clone = saw_post_tool_event.clone();
        let on_event: Arc<OnEvent> = Arc::new(move |event| {
            if let AgentEvent::GuardrailDenied { hook, .. } = &event {
                if hook == "post_tool" {
                    saw_clone.store(true, Ordering::SeqCst);
                }
            }
        });

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .guardrail(Arc::new(FailingPostTool))
            .on_event(on_event)
            .build()
            .unwrap();

        runner.execute("search").await.unwrap();
        assert!(
            saw_post_tool_event.load(Ordering::SeqCst),
            "should have emitted GuardrailDenied event with hook=post_tool"
        );
    }

    #[tokio::test]
    async fn without_on_input_returns_immediately() {
        // Without on_input, a text response ends the run (existing behavior)
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Done.".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

        let output = runner.execute("Hi").await.unwrap();
        assert_eq!(output.result, "Done.");
    }

    #[tokio::test]
    async fn run_timeout_preserves_partial_usage() {
        // Provider that returns a tool call on first response (accumulating usage),
        // then hangs indefinitely on the second call so timeout fires.
        struct SlowProvider;
        impl LlmProvider for SlowProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                // First call: has only the user message. Return a tool call.
                if request.messages.len() <= 1 {
                    return Ok(CompletionResponse {
                        content: vec![ContentBlock::ToolUse {
                            id: "tc1".into(),
                            name: "echo".into(),
                            input: json!({}),
                        }],
                        stop_reason: StopReason::ToolUse,
                        usage: TokenUsage {
                            input_tokens: 100,
                            output_tokens: 50,
                            ..Default::default()
                        },
                        model: None,
                    });
                }
                // Second call: sleep forever (simulates slow LLM)
                tokio::time::sleep(Duration::from_secs(3600)).await;
                unreachable!()
            }
        }

        let provider = Arc::new(SlowProvider);
        let tool = Arc::new(MockTool::new("echo", "echoed"));
        let runner = AgentRunner::builder(provider)
            .name("timeout-test")
            .system_prompt("sys")
            .tool(tool)
            .max_turns(10)
            .run_timeout(Duration::from_millis(100))
            .build()
            .unwrap();

        let err = runner.execute("go").await.unwrap_err();
        assert!(
            matches!(&err, Error::WithPartialUsage { source, .. }
                if matches!(**source, Error::RunTimeout(_))),
            "expected WithPartialUsage(RunTimeout), got: {err}"
        );
        let usage = err.partial_usage();
        assert_eq!(usage.input_tokens, 100, "should preserve input tokens");
        assert_eq!(usage.output_tokens, 50, "should preserve output tokens");
    }

    #[tokio::test]
    async fn run_timeout_without_accumulated_usage() {
        // Provider that immediately hangs — no usage accumulated.
        struct ImmediatelySlowProvider;
        impl LlmProvider for ImmediatelySlowProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                unreachable!()
            }
        }

        let provider = Arc::new(ImmediatelySlowProvider);
        let runner = AgentRunner::builder(provider)
            .name("timeout-test")
            .system_prompt("sys")
            .run_timeout(Duration::from_millis(50))
            .build()
            .unwrap();

        let err = runner.execute("go").await.unwrap_err();
        assert!(
            matches!(&err, Error::WithPartialUsage { source, .. }
                if matches!(**source, Error::RunTimeout(_))),
            "expected WithPartialUsage(RunTimeout), got: {err}"
        );
        let usage = err.partial_usage();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }

    #[tokio::test]
    async fn llm_error_mid_run_preserves_partial_usage() {
        // Provider that returns a tool call on turn 1 (accumulating tokens),
        // then errors on turn 2 with an API error.
        struct FailOnSecondCall;
        impl LlmProvider for FailOnSecondCall {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                if request.messages.len() <= 1 {
                    return Ok(CompletionResponse {
                        content: vec![ContentBlock::ToolUse {
                            id: "tc1".into(),
                            name: "echo".into(),
                            input: json!({}),
                        }],
                        stop_reason: StopReason::ToolUse,
                        usage: TokenUsage {
                            input_tokens: 200,
                            output_tokens: 80,
                            ..Default::default()
                        },
                        model: None,
                    });
                }
                Err(Error::Api {
                    status: 500,
                    message: "internal server error".into(),
                })
            }
        }

        let provider = Arc::new(FailOnSecondCall);
        let tool = Arc::new(MockTool::new("echo", "echoed"));
        let runner = AgentRunner::builder(provider)
            .name("mid-error-test")
            .system_prompt("sys")
            .tool(tool)
            .max_turns(10)
            .build()
            .unwrap();

        let err = runner.execute("go").await.unwrap_err();
        assert!(
            matches!(&err, Error::WithPartialUsage { source, .. }
                if matches!(**source, Error::Api { status: 500, .. })),
            "expected WithPartialUsage(Api{{500}}), got: {err}"
        );
        let usage = err.partial_usage();
        assert_eq!(
            usage.input_tokens, 200,
            "should preserve input tokens from turn 1"
        );
        assert_eq!(
            usage.output_tokens, 80,
            "should preserve output tokens from turn 1"
        );
    }

    // --- Reflection tests ---

    #[tokio::test]
    async fn reflection_prompt_injected_after_tool_results() {
        // Provider that captures all user-role messages to verify the reflection prompt
        struct ReflectionCapture {
            responses: Mutex<Vec<CompletionResponse>>,
            user_messages: Mutex<Vec<String>>,
        }
        impl LlmProvider for ReflectionCapture {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                for msg in &request.messages {
                    if msg.role == crate::llm::types::Role::User {
                        for block in &msg.content {
                            if let ContentBlock::Text { text } = block {
                                self.user_messages.lock().unwrap().push(text.clone());
                            }
                        }
                    }
                }
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Err(Error::Agent("no more responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let provider = Arc::new(ReflectionCapture {
            responses: Mutex::new(vec![
                // Turn 1: tool call
                CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "t1".into(),
                        name: "search".into(),
                        input: json!({}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                    model: None,
                },
                // Turn 2: final answer
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                },
            ]),
            user_messages: Mutex::new(vec![]),
        });

        let tool = Arc::new(MockTool::new("search", "found results"));
        let runner = AgentRunner::builder(provider.clone())
            .name("reflector")
            .system_prompt("sys")
            .tool(tool)
            .enable_reflection(true)
            .build()
            .unwrap();

        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "Done.");

        let msgs = provider.user_messages.lock().unwrap();
        // Should contain: initial task + reflection prompt
        assert!(
            msgs.iter()
                .any(|m| m.contains("Before proceeding, briefly reflect")),
            "expected reflection prompt in user messages, got: {msgs:?}"
        );
    }

    #[tokio::test]
    async fn reflection_not_injected_when_disabled() {
        struct ReflectionCapture {
            responses: Mutex<Vec<CompletionResponse>>,
            user_messages: Mutex<Vec<String>>,
        }
        impl LlmProvider for ReflectionCapture {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                for msg in &request.messages {
                    if msg.role == crate::llm::types::Role::User {
                        for block in &msg.content {
                            if let ContentBlock::Text { text } = block {
                                self.user_messages.lock().unwrap().push(text.clone());
                            }
                        }
                    }
                }
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Err(Error::Agent("no more responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let provider = Arc::new(ReflectionCapture {
            responses: Mutex::new(vec![
                CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "t1".into(),
                        name: "search".into(),
                        input: json!({}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                    model: None,
                },
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                },
            ]),
            user_messages: Mutex::new(vec![]),
        });

        let tool = Arc::new(MockTool::new("search", "found results"));
        // Note: enable_reflection is NOT called (default false)
        let runner = AgentRunner::builder(provider.clone())
            .name("no-reflect")
            .system_prompt("sys")
            .tool(tool)
            .build()
            .unwrap();

        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "Done.");

        let msgs = provider.user_messages.lock().unwrap();
        assert!(
            !msgs.iter().any(|m| m.contains("reflect")),
            "should not contain reflection prompt, got: {msgs:?}"
        );
    }

    #[tokio::test]
    async fn reflection_not_injected_when_no_tool_calls() {
        struct ReflectionCapture {
            responses: Mutex<Vec<CompletionResponse>>,
            user_messages: Mutex<Vec<String>>,
        }
        impl LlmProvider for ReflectionCapture {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                for msg in &request.messages {
                    if msg.role == crate::llm::types::Role::User {
                        for block in &msg.content {
                            if let ContentBlock::Text { text } = block {
                                self.user_messages.lock().unwrap().push(text.clone());
                            }
                        }
                    }
                }
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Err(Error::Agent("no more responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let provider = Arc::new(ReflectionCapture {
            responses: Mutex::new(vec![CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Direct answer.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            }]),
            user_messages: Mutex::new(vec![]),
        });

        // Even with reflection enabled, no tool calls means no reflection prompt
        let runner = AgentRunner::builder(provider.clone())
            .name("no-tools")
            .system_prompt("sys")
            .enable_reflection(true)
            .build()
            .unwrap();

        let output = runner.execute("just answer").await.unwrap();
        assert_eq!(output.result, "Direct answer.");

        let msgs = provider.user_messages.lock().unwrap();
        assert!(
            !msgs.iter().any(|m| m.contains("reflect")),
            "no reflection when no tool calls, got: {msgs:?}"
        );
    }

    // --- Tool output compression tests ---

    #[tokio::test]
    async fn compress_short_output_unchanged() {
        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: tool call
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: final answer
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let tool = Arc::new(MockTool::new("search", "short result"));
        let runner = AgentRunner::builder(provider)
            .name("compressor")
            .system_prompt("sys")
            .tool(tool)
            // Threshold higher than tool output → no compression
            .tool_output_compression_threshold(10000)
            .build()
            .unwrap();

        let output = runner.execute("search something").await.unwrap();
        assert_eq!(output.result, "Done.");
        // Only 2 LLM calls (tool turn + final answer), no compression call
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn compress_long_output_calls_llm() {
        // Provider that returns: tool call, then compression response, then final answer
        struct CompressionProvider {
            responses: Mutex<Vec<CompletionResponse>>,
            call_count: Mutex<usize>,
        }
        impl LlmProvider for CompressionProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                let mut count = self.call_count.lock().unwrap();
                *count += 1;
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Err(Error::Agent("no more responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let provider = Arc::new(CompressionProvider {
            responses: Mutex::new(vec![
                // Turn 1: tool call
                CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "t1".into(),
                        name: "read".into(),
                        input: json!({}),
                    }],
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                    model: None,
                },
                // Compression call response
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Compressed summary of large file.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage {
                        input_tokens: 50,
                        output_tokens: 10,
                        ..Default::default()
                    },
                    model: None,
                },
                // Turn 2: final answer
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Here's the result.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                },
            ]),
            call_count: Mutex::new(0),
        });

        // Tool returns a large output (>50 bytes threshold)
        let large_output = "x".repeat(200);
        let tool = Arc::new(MockTool::new("read", &large_output));
        let runner = AgentRunner::builder(provider.clone())
            .name("compressor")
            .system_prompt("sys")
            .tool(tool)
            .tool_output_compression_threshold(50)
            .build()
            .unwrap();

        let output = runner.execute("read the file").await.unwrap();
        assert_eq!(output.result, "Here's the result.");
        // 3 LLM calls: tool call + compression + final answer
        let calls = *provider.call_count.lock().unwrap();
        assert_eq!(calls, 3, "expected 3 LLM calls (tool + compress + answer)");
        // Compression tokens should be accumulated
        assert_eq!(output.tokens_used.input_tokens, 50);
        assert_eq!(output.tokens_used.output_tokens, 10);
    }

    #[tokio::test]
    async fn compression_preserves_error_status() {
        // Error tool outputs should NOT be compressed
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "failing_tool".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Tool failed.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let large_error = "e".repeat(200);
        let tool = Arc::new(MockTool::failing("failing_tool", &large_error));
        let runner = AgentRunner::builder(provider)
            .name("compressor")
            .system_prompt("sys")
            .tool(tool)
            .tool_output_compression_threshold(50)
            .build()
            .unwrap();

        let output = runner.execute("try something").await.unwrap();
        assert_eq!(output.result, "Tool failed.");
        // Only 2 LLM calls — error outputs are not compressed
        assert_eq!(output.tool_calls_made, 1);
    }

    // --- Dynamic tool selection tests ---

    #[test]
    fn select_tools_returns_all_when_below_max() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("selector")
            .system_prompt("sys")
            .max_tools_per_turn(10)
            .build()
            .unwrap();

        let tools = vec![
            ToolDefinition {
                name: "a".into(),
                description: "Tool A".into(),
                input_schema: json!({"type": "object"}),
            },
            ToolDefinition {
                name: "b".into(),
                description: "Tool B".into(),
                input_schema: json!({"type": "object"}),
            },
        ];

        let selected = runner.select_tools_for_turn(&tools, &[], &[], 10);
        assert_eq!(selected.len(), 2, "should return all when below max");
    }

    #[test]
    fn select_tools_includes_recently_used() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("selector")
            .system_prompt("sys")
            .max_tools_per_turn(2)
            .build()
            .unwrap();

        let tools: Vec<ToolDefinition> = (0..5)
            .map(|i| ToolDefinition {
                name: format!("tool_{i}"),
                description: format!("Tool number {i}"),
                input_schema: json!({"type": "object"}),
            })
            .collect();

        // tool_3 was recently used
        let recently_used = vec!["tool_3".to_string()];
        let selected = runner.select_tools_for_turn(&tools, &[], &recently_used, 2);

        assert_eq!(selected.len(), 2, "should cap at max");
        assert!(
            selected.iter().any(|t| t.name == "tool_3"),
            "recently used tool must be included"
        );
    }

    #[test]
    fn select_tools_keyword_match_ranking() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("selector")
            .system_prompt("sys")
            .max_tools_per_turn(2)
            .build()
            .unwrap();

        let tools = vec![
            ToolDefinition {
                name: "web_search".into(),
                description: "Search the web".into(),
                input_schema: json!({"type": "object"}),
            },
            ToolDefinition {
                name: "read_file".into(),
                description: "Read a file from disk".into(),
                input_schema: json!({"type": "object"}),
            },
            ToolDefinition {
                name: "write_file".into(),
                description: "Write a file to disk".into(),
                input_schema: json!({"type": "object"}),
            },
            ToolDefinition {
                name: "run_command".into(),
                description: "Run a shell command".into(),
                input_schema: json!({"type": "object"}),
            },
        ];

        // User message mentions "search" and "web"
        let messages = vec![Message::user(
            "Please search the web for information.".to_string(),
        )];
        let selected = runner.select_tools_for_turn(&tools, &messages, &[], 2);

        assert_eq!(selected.len(), 2);
        // web_search should be selected (matches "search" and "web")
        assert!(
            selected.iter().any(|t| t.name == "web_search"),
            "web_search should be selected by keyword match, got: {:?}",
            selected.iter().map(|t| &t.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn select_tools_caps_at_max() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("selector")
            .system_prompt("sys")
            .max_tools_per_turn(3)
            .build()
            .unwrap();

        let tools: Vec<ToolDefinition> = (0..10)
            .map(|i| ToolDefinition {
                name: format!("tool_{i}"),
                description: format!("Tool number {i}"),
                input_schema: json!({"type": "object"}),
            })
            .collect();

        let selected = runner.select_tools_for_turn(&tools, &[], &[], 3);
        assert_eq!(selected.len(), 3, "should cap at max_tools");
    }

    #[test]
    fn select_tools_caps_when_recently_used_exceeds_max() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("selector")
            .system_prompt("sys")
            .build()
            .unwrap();

        let tools: Vec<ToolDefinition> = (0..5)
            .map(|i| ToolDefinition {
                name: format!("tool_{i}"),
                description: format!("Tool {i}"),
                input_schema: json!({"type": "object"}),
            })
            .collect();

        // recently_used has 4 tools, max_tools is 2 — should truncate to 2
        let recently_used: Vec<String> = (0..4).map(|i| format!("tool_{i}")).collect();
        let selected = runner.select_tools_for_turn(&tools, &[], &recently_used, 2);
        assert_eq!(
            selected.len(),
            2,
            "should cap at max_tools even when recently_used exceeds it"
        );
    }

    #[test]
    fn select_tools_preserves_respond_tool() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

        let tools: Vec<ToolDefinition> = vec![
            ToolDefinition {
                name: "bash".into(),
                description: "Run commands".into(),
                input_schema: json!({"type": "object"}),
            },
            ToolDefinition {
                name: "read".into(),
                description: "Read files".into(),
                input_schema: json!({"type": "object"}),
            },
            ToolDefinition {
                name: "write".into(),
                description: "Write files".into(),
                input_schema: json!({"type": "object"}),
            },
            ToolDefinition {
                name: crate::llm::types::RESPOND_TOOL_NAME.into(),
                description: "Structured output".into(),
                input_schema: json!({"type": "object"}),
            },
        ];

        // max_tools=2, no recently_used — __respond__ must still be included
        let selected = runner.select_tools_for_turn(&tools, &[], &[], 2);
        assert!(
            selected.iter().any(|t| t.name == "__respond__"),
            "__respond__ must always survive select_tools_for_turn"
        );
    }

    #[test]
    fn levenshtein_identical_strings() {
        assert_eq!(super::levenshtein("read_file", "read_file"), 0);
    }

    #[test]
    fn levenshtein_single_substitution() {
        assert_eq!(super::levenshtein("reed_file", "read_file"), 1);
    }

    #[test]
    fn levenshtein_empty_strings() {
        assert_eq!(super::levenshtein("", ""), 0);
        assert_eq!(super::levenshtein("abc", ""), 3);
        assert_eq!(super::levenshtein("", "xyz"), 3);
    }

    #[test]
    fn find_closest_tool_exact_match_returns_none() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("read_file", "ok")))
            .build()
            .unwrap();
        assert!(runner.find_closest_tool("read_file", 2).is_none());
    }

    #[test]
    fn find_closest_tool_within_distance() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("read_file", "ok")))
            .build()
            .unwrap();
        assert_eq!(runner.find_closest_tool("reed_file", 2), Some("read_file"));
    }

    #[test]
    fn find_closest_tool_too_far() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("read_file", "ok")))
            .build()
            .unwrap();
        assert!(runner.find_closest_tool("completely_wrong", 2).is_none());
    }

    #[test]
    fn find_closest_tool_prefers_closest() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("read_fil", "ok")))
            .tool(Arc::new(MockTool::new("read_file", "ok")))
            .build()
            .unwrap();
        assert_eq!(runner.find_closest_tool("read_fi", 2), Some("read_fil"));
    }

    #[tokio::test]
    async fn tool_name_repair_executes_correct_tool() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "tc1".into(),
                    name: "reed_file".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 15,
                    output_tokens: 3,
                    ..Default::default()
                },
                model: None,
            },
        ]));
        let runner = AgentRunner::builder(provider)
            .name("repair-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("read_file", "file contents here")))
            .build()
            .unwrap();
        let output = runner.execute("read the file").await.unwrap();
        assert_eq!(output.result, "Done!");
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn tool_name_too_far_returns_error() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "tc1".into(),
                    name: "completely_wrong".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Error handled".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 15,
                    output_tokens: 3,
                    ..Default::default()
                },
                model: None,
            },
        ]));
        let runner = AgentRunner::builder(provider)
            .name("repair-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("read_file", "file contents here")))
            .build()
            .unwrap();
        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "Error handled");
        assert_eq!(output.tool_calls_made, 1);
    }

    // --- FallibleMockProvider: returns Result<CompletionResponse, Error> per call ---

    struct FallibleMockProvider {
        responses: Mutex<Vec<Result<CompletionResponse, Error>>>,
    }

    impl FallibleMockProvider {
        fn new(responses: Vec<Result<CompletionResponse, Error>>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl LlmProvider for FallibleMockProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            let mut responses = self.responses.lock().expect("mock lock poisoned");
            if responses.is_empty() {
                return Err(Error::Agent("no more mock responses".into()));
            }
            responses.remove(0)
        }
    }

    fn overflow_error() -> Error {
        Error::Api {
            status: 400,
            message: "prompt is too long: 250000 tokens > 200000 maximum".into(),
        }
    }

    fn success_response(text: &str) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
            model: None,
        }
    }

    fn tool_use_response(id: &str, tool_name: &str) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: tool_name.into(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
            model: None,
        }
    }

    #[tokio::test]
    async fn auto_compaction_on_context_overflow() {
        // Scenario: agent does 3 tool-use turns (builds up 7 messages > 5),
        // then LLM returns overflow error. Auto-compaction summarizes and retries.
        //
        // Messages after 3 tool turns: initial_user + 3*(assistant + tool_result) = 7
        //
        // Call sequence:
        // 1-3. Turns 1-3: tool use responses (success)
        // 4. Turn 4: overflow error (triggers compaction)
        // 5. Summary LLM call (success via generate_summary)
        // 6. Turn 4 retry: success text response
        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let events_clone = events.clone();

        let provider = Arc::new(FallibleMockProvider::new(vec![
            Ok(tool_use_response("c1", "search")), // Turn 1
            Ok(tool_use_response("c2", "search")), // Turn 2
            Ok(tool_use_response("c3", "search")), // Turn 3
            Err(overflow_error()),                 // Turn 4: overflow
            Ok(success_response("Summary of conversation so far")), // Summary call
            Ok(success_response("Final answer after compaction")), // Retry
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test-compact")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .max_turns(10)
            .on_event(Arc::new(move |e| {
                events_clone.lock().unwrap().push(e);
            }))
            .build()
            .unwrap();

        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "Final answer after compaction");

        let events = events.lock().unwrap();
        let summarized = events
            .iter()
            .any(|e| matches!(e, AgentEvent::ContextSummarized { .. }));
        assert!(summarized, "expected ContextSummarized event");
    }

    #[tokio::test]
    async fn auto_compaction_not_attempted_twice() {
        // After compaction, if overflow recurs the agent fails (no infinite loop).
        // 3 tool turns build up 7 messages (> 5 threshold).
        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let events_clone = events.clone();

        let provider = Arc::new(FallibleMockProvider::new(vec![
            Ok(tool_use_response("c1", "search")),
            Ok(tool_use_response("c2", "search")),
            Ok(tool_use_response("c3", "search")),
            Err(overflow_error()),
            Ok(success_response("Summary")),
            Err(overflow_error()), // Retry still overflows
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test-compact")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .max_turns(10)
            .on_event(Arc::new(move |e| {
                events_clone.lock().unwrap().push(e);
            }))
            .build()
            .unwrap();

        let err = runner.execute("do something").await.unwrap_err();
        let inner = match &err {
            Error::WithPartialUsage { source, .. } => source.as_ref(),
            other => other,
        };
        assert!(
            matches!(inner, Error::Api { status: 400, .. }),
            "expected overflow error, got: {err:?}"
        );

        let events = events.lock().unwrap();
        let count = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ContextSummarized { .. }))
            .count();
        assert_eq!(count, 1, "compaction attempted exactly once");
    }

    #[tokio::test]
    async fn auto_compaction_skipped_when_too_few_messages() {
        // Overflow on first call with only 1 message — no compaction attempted.
        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let events_clone = events.clone();

        let provider = Arc::new(FallibleMockProvider::new(vec![Err(overflow_error())]));

        let runner = AgentRunner::builder(provider)
            .name("test-compact")
            .system_prompt("sys")
            .max_turns(10)
            .on_event(Arc::new(move |e| {
                events_clone.lock().unwrap().push(e);
            }))
            .build()
            .unwrap();

        let err = runner.execute("short task").await.unwrap_err();
        let inner = match &err {
            Error::WithPartialUsage { source, .. } => source.as_ref(),
            other => other,
        };
        assert!(
            matches!(inner, Error::Api { status: 400, .. }),
            "expected overflow error, got: {err:?}"
        );

        let events = events.lock().unwrap();
        let count = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ContextSummarized { .. }))
            .count();
        assert_eq!(count, 0, "no compaction with too few messages");
    }

    // --- Doom Loop Detection Tests ---

    #[test]
    fn doom_loop_tracker_detects_repeated_calls() {
        let mut tracker = DoomLoopTracker::new();
        let calls = vec![ToolCall {
            id: "call-1".into(),
            name: "search".into(),
            input: json!({"query": "rust"}),
        }];
        assert!(!tracker.record(&calls, 3));
        assert!(!tracker.record(&calls, 3));
        assert!(tracker.record(&calls, 3)); // 3rd time triggers
    }

    #[test]
    fn doom_loop_tracker_resets_on_different_call() {
        let mut tracker = DoomLoopTracker::new();
        let calls_a = vec![ToolCall {
            id: "call-1".into(),
            name: "search".into(),
            input: json!({"query": "rust"}),
        }];
        let calls_b = vec![ToolCall {
            id: "call-2".into(),
            name: "search".into(),
            input: json!({"query": "python"}),
        }];
        assert!(!tracker.record(&calls_a, 3));
        assert!(!tracker.record(&calls_a, 3));
        // Different input resets
        assert!(!tracker.record(&calls_b, 3));
        assert!(!tracker.record(&calls_b, 3));
        assert!(tracker.record(&calls_b, 3)); // 3rd consecutive of calls_b
    }

    #[test]
    fn doom_loop_tracker_ignores_call_id_differences() {
        // The call ID changes each turn but name+input are the same
        let mut tracker = DoomLoopTracker::new();
        let calls_1 = vec![ToolCall {
            id: "call-1".into(),
            name: "read".into(),
            input: json!({"file": "foo.txt"}),
        }];
        let calls_2 = vec![ToolCall {
            id: "call-2".into(),
            name: "read".into(),
            input: json!({"file": "foo.txt"}),
        }];
        assert!(!tracker.record(&calls_1, 2));
        assert!(tracker.record(&calls_2, 2)); // Same name+input, different ID
    }

    #[test]
    fn doom_loop_tracker_multi_tool_turn() {
        let mut tracker = DoomLoopTracker::new();
        let calls = vec![
            ToolCall {
                id: "a".into(),
                name: "search".into(),
                input: json!({"q": "x"}),
            },
            ToolCall {
                id: "b".into(),
                name: "read".into(),
                input: json!({"file": "y"}),
            },
        ];
        assert!(!tracker.record(&calls, 2));
        assert!(tracker.record(&calls, 2));
    }

    #[tokio::test]
    async fn doom_loop_detected_after_threshold() {
        // Mock provider returns the same tool call 4 times, then text.
        // With threshold 3, the 3rd turn should get error results.
        let tool_response = |id: &str| CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: "my_tool".into(),
                input: json!({"key": "same_value"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: None,
        };

        let provider = Arc::new(MockProvider::new(vec![
            tool_response("c1"),
            tool_response("c2"),
            tool_response("c3"), // 3rd identical turn => doom loop detected
            // After error result, LLM returns text (adapted)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "I'll try something different.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let tool = MockTool::new("my_tool", "tool result");
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(tool))
            .max_turns(10)
            .max_identical_tool_calls(3)
            .build()
            .unwrap();

        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "I'll try something different.");
        // 2 real tool calls + 1 doom-loop error result + 0 (text response)
        assert_eq!(output.tool_calls_made, 3);
    }

    #[tokio::test]
    async fn doom_loop_resets_on_different_call() {
        // 2 identical calls, then a different call, then 2 more of the different.
        // With threshold 3, no doom loop should be detected.
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "my_tool".into(),
                    input: json!({"key": "value_a"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "my_tool".into(),
                    input: json!({"key": "value_a"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Different input resets tracker
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c3".into(),
                    name: "my_tool".into(),
                    input: json!({"key": "value_b"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c4".into(),
                    name: "my_tool".into(),
                    input: json!({"key": "value_b"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let tool = MockTool::new("my_tool", "result");
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(tool))
            .max_turns(10)
            .max_identical_tool_calls(3)
            .build()
            .unwrap();

        let output = runner.execute("task").await.unwrap();
        assert_eq!(output.result, "done");
        // All 4 tool calls executed normally (no doom loop triggered)
        assert_eq!(output.tool_calls_made, 4);
    }

    #[tokio::test]
    async fn doom_loop_disabled_by_default() {
        // Without setting max_identical_tool_calls, no detection occurs
        // even with many identical calls.
        let tool_response = |id: &str| CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: "my_tool".into(),
                input: json!({"key": "same"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: None,
        };

        let provider = Arc::new(MockProvider::new(vec![
            tool_response("c1"),
            tool_response("c2"),
            tool_response("c3"),
            tool_response("c4"),
            tool_response("c5"),
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let tool = MockTool::new("my_tool", "result");
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(tool))
            .max_turns(10)
            // NOT setting max_identical_tool_calls
            .build()
            .unwrap();

        let output = runner.execute("task").await.unwrap();
        assert_eq!(output.result, "done");
        // All 5 tool calls executed without doom loop detection
        assert_eq!(output.tool_calls_made, 5);
    }

    #[test]
    fn builder_rejects_zero_max_identical_tool_calls() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .max_identical_tool_calls(0)
            .build();
        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("max_identical_tool_calls must be at least 1"),
                    "error: {msg}"
                );
            }
            Ok(_) => panic!("expected error for max_identical_tool_calls(0)"),
        }
    }

    #[test]
    fn builder_rejects_zero_max_total_tokens() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .max_total_tokens(0)
            .build();
        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("max_total_tokens must be at least 1"),
                    "error: {msg}"
                );
            }
            Ok(_) => panic!("expected error for max_total_tokens(0)"),
        }
    }

    // --- Permission Rules Integration Tests ---

    #[tokio::test]
    async fn permission_allow_bypasses_approval() {
        // With permission rules, allowed tools don't trigger the approval callback.
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "read_file".into(),
                    input: json!({"path": "src/main.rs"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let rules = permission::PermissionRuleset::new(vec![permission::PermissionRule {
            tool: "read_file".into(),
            pattern: "*".into(),
            action: permission::PermissionAction::Allow,
        }]);

        let approval_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let approval_called_clone = approval_called.clone();

        let runner = AgentRunner::builder(provider)
            .name("perm-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("read_file", "file contents")))
            .on_approval(Arc::new(move |_: &[ToolCall]| {
                approval_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                crate::llm::ApprovalDecision::Deny // Would deny if called
            }))
            .permission_rules(rules)
            .build()
            .unwrap();

        let output = runner.execute("read something").await.unwrap();
        assert_eq!(output.result, "done");
        assert!(!approval_called.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn permission_deny_returns_error_result() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "bash".into(),
                    input: json!({"command": "rm -rf /"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "ok i won't do that".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let rules = permission::PermissionRuleset::new(vec![permission::PermissionRule {
            tool: "bash".into(),
            pattern: "rm *".into(),
            action: permission::PermissionAction::Deny,
        }]);

        let runner = AgentRunner::builder(provider)
            .name("perm-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("bash", "executed")))
            .permission_rules(rules)
            .build()
            .unwrap();

        let output = runner.execute("delete everything").await.unwrap();
        assert_eq!(output.result, "ok i won't do that");
        // 1 denied + 0 executed
        assert_eq!(output.tool_calls_made, 1);
    }

    #[tokio::test]
    async fn permission_ask_falls_through_to_approval() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "bash".into(),
                    input: json!({"command": "cargo test"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "tests passed".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let rules = permission::PermissionRuleset::new(vec![
            permission::PermissionRule {
                tool: "bash".into(),
                pattern: "rm *".into(),
                action: permission::PermissionAction::Deny,
            },
            permission::PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: permission::PermissionAction::Ask,
            },
        ]);

        let approval_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let approval_called_clone = approval_called.clone();

        let runner = AgentRunner::builder(provider)
            .name("perm-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("bash", "ok")))
            .on_approval(Arc::new(move |_: &[ToolCall]| {
                approval_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                crate::llm::ApprovalDecision::Allow // Approve
            }))
            .permission_rules(rules)
            .build()
            .unwrap();

        let output = runner.execute("run tests").await.unwrap();
        assert_eq!(output.result, "tests passed");
        assert!(approval_called.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn permission_mixed_allow_and_deny() {
        // Two tool calls in one turn: one allowed, one denied.
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "read_file".into(),
                        input: json!({"path": "src/main.rs"}),
                    },
                    ContentBlock::ToolUse {
                        id: "c2".into(),
                        name: "read_file".into(),
                        input: json!({"path": ".env"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "got it".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let rules = permission::PermissionRuleset::new(vec![
            permission::PermissionRule {
                tool: "*".into(),
                pattern: "*.env*".into(),
                action: permission::PermissionAction::Deny,
            },
            permission::PermissionRule {
                tool: "read_file".into(),
                pattern: "*".into(),
                action: permission::PermissionAction::Allow,
            },
        ]);

        let runner = AgentRunner::builder(provider)
            .name("perm-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("read_file", "contents")))
            .permission_rules(rules)
            .build()
            .unwrap();

        let output = runner.execute("read files").await.unwrap();
        assert_eq!(output.result, "got it");
        // Both counted: 1 executed + 1 denied
        assert_eq!(output.tool_calls_made, 2);
    }

    #[tokio::test]
    async fn permission_no_rules_uses_legacy_approval() {
        // Without permission rules, the legacy on_approval callback is used.
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "denied".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("perm-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("bash", "ok")))
            .on_approval(Arc::new(|_: &[ToolCall]| {
                crate::llm::ApprovalDecision::Deny
            }))
            .build()
            .unwrap();

        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "denied");
    }

    #[tokio::test]
    async fn always_allow_injects_rule_into_live_ruleset() {
        // No permission rules → legacy callback path on turn 1.
        // AlwaysAllow injects a learned rule → turn 2 auto-allowed (no callback).
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: LLM calls bash → legacy path → callback
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: LLM calls bash again → learned rule matches → auto-allowed
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 3: done
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("perm-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("bash", "ok")))
            .on_approval(Arc::new(move |_: &[ToolCall]| {
                call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                crate::llm::ApprovalDecision::AlwaysAllow
            }))
            .build()
            .unwrap();

        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "done");
        // Callback should have been called exactly once (turn 1, legacy path).
        // Turn 2 uses the per-call path (has_permission_rules is now true),
        // and the injected Allow rule matches → no callback.
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn always_deny_injects_rule_into_live_ruleset() {
        // No permission rules → legacy callback path on turn 1.
        // AlwaysDeny injects a learned Deny rule → turn 2 auto-denied.
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: LLM calls bash → legacy path → callback
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "bash".into(),
                    input: json!({"command": "rm -rf /"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: LLM calls bash again → learned Deny rule matches → auto-denied
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "bash".into(),
                    input: json!({"command": "rm -rf /"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 3: done
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "gave up".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("perm-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("bash", "ok")))
            .on_approval(Arc::new(move |_: &[ToolCall]| {
                call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                crate::llm::ApprovalDecision::AlwaysDeny
            }))
            .build()
            .unwrap();

        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "gave up");
        // Callback should have been called exactly once (turn 1, legacy path).
        // Turn 2 uses the per-call path, injected Deny rule matches → no callback.
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn config_deny_overrides_learned_allow() {
        // Config has specific deny rule for "bash rm *".
        // User says AlwaysAllow for bash → learned Allow rule added.
        // Next call with "rm" should still be denied by the config rule
        // (config rules are evaluated before learned rules).
        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: bash ls → Ask → callback → AlwaysAllow
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: bash rm → config deny matches first (before learned allow)
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "bash".into(),
                    input: json!({"command": "rm -rf /"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 3: done
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "blocked".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let rules = permission::PermissionRuleset::new(vec![permission::PermissionRule {
            tool: "bash".into(),
            pattern: "rm *".into(),
            action: permission::PermissionAction::Deny,
        }]);

        let runner = AgentRunner::builder(provider)
            .name("perm-test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("bash", "ok")))
            .on_approval(Arc::new(|_: &[ToolCall]| {
                crate::llm::ApprovalDecision::AlwaysAllow
            }))
            .permission_rules(rules)
            .build()
            .unwrap();

        let output = runner.execute("do something").await.unwrap();
        assert_eq!(output.result, "blocked");
        // Turn 2's "rm -rf /" should be denied by config rule even though
        // we have a learned Allow rule for bash
    }

    #[tokio::test]
    async fn workspace_injects_system_prompt_hint() {
        let provider = MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "done".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]);
        let runner = AgentRunner::builder(Arc::new(provider))
            .name("test")
            .system_prompt("base prompt")
            .workspace("/test/workspace")
            .build()
            .unwrap();

        // The system prompt should contain the workspace path
        assert!(runner.system_prompt.contains("/test/workspace"));
        assert!(runner.system_prompt.contains("base prompt"));
        assert!(runner.system_prompt.contains("workspace directory"));
    }

    #[tokio::test]
    async fn no_workspace_no_prompt_hint() {
        let provider = MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "done".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }]);
        let runner = AgentRunner::builder(Arc::new(provider))
            .name("test")
            .system_prompt("base prompt")
            .tool(Arc::new(MockTool::new("bash", "ok")))
            .build()
            .unwrap();

        assert!(runner.system_prompt.starts_with("base prompt"));
        assert!(runner.system_prompt.contains("Resourcefulness"));
        assert!(!runner.system_prompt.contains("workspace"));
    }

    #[test]
    fn resourcefulness_guidelines_included_with_power_tools() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("prompt")
            .tool(Arc::new(MockTool::new("bash", "ok")))
            .build()
            .unwrap();
        assert!(
            runner.system_prompt.contains("Resourcefulness"),
            "should include guidelines when bash tool is present"
        );
    }

    #[test]
    fn resourcefulness_guidelines_excluded_without_power_tools() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("prompt")
            .tool(Arc::new(MockTool::new("memory_recall", "ok")))
            .build()
            .unwrap();
        assert!(
            !runner.system_prompt.contains("Resourcefulness"),
            "should not include guidelines when only memory tools are present"
        );
    }

    #[test]
    fn system_prompt_contains_current_date() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("prompt")
            .build()
            .unwrap();
        assert!(
            runner.system_prompt.contains("Current date and time:"),
            "system prompt should contain current date/time"
        );
        // Verify it contains the current year
        let year = chrono::Utc::now().format("%Y").to_string();
        assert!(
            runner.system_prompt.contains(&year),
            "system prompt should contain current year"
        );
    }

    #[tokio::test]
    async fn budget_exceeded_returns_error() {
        // Mock: first call returns 60k tokens, second call also returns 60k tokens
        // Budget is 100k, so the second call should trigger BudgetExceeded
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "echo".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 30000,
                    output_tokens: 30000,
                    ..Default::default()
                },
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 30000,
                    output_tokens: 30000,
                    ..Default::default()
                },
                model: None,
            },
        ]));
        let tool = MockTool::new("echo", "ok");
        let runner = AgentRunner::builder(provider)
            .name("budget-test")
            .system_prompt("test")
            .tool(Arc::new(tool))
            .max_total_tokens(100000) // Budget: 100k total tokens
            .build()
            .unwrap();

        let result = runner.execute("test task").await;
        match result {
            Err(Error::WithPartialUsage { source, usage }) => {
                assert!(
                    matches!(
                        *source,
                        Error::BudgetExceeded {
                            used: 120000,
                            limit: 100000
                        }
                    ),
                    "expected BudgetExceeded, got: {source}"
                );
                assert_eq!(usage.total(), 120000);
            }
            Err(e) => panic!("expected BudgetExceeded, got: {e}"),
            Ok(output) => panic!("expected error, got success: {}", output.result),
        }
    }

    #[tokio::test]
    async fn budget_not_exceeded_when_under_limit() {
        // Single LLM call with 100 tokens, budget is 1000
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "done".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 50,
                output_tokens: 50,
                ..Default::default()
            },
            model: None,
        }]));
        let runner = AgentRunner::builder(provider)
            .name("budget-ok-test")
            .system_prompt("test")
            .max_total_tokens(1000)
            .build()
            .unwrap();

        let output = runner.execute("test task").await.unwrap();
        assert_eq!(output.tokens_used.total(), 100);
    }

    #[tokio::test]
    async fn budget_event_emitted_on_exceeded() {
        let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "done".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 100,
                ..Default::default()
            },
            model: None,
        }]));
        let runner = AgentRunner::builder(provider)
            .name("budget-event-test")
            .system_prompt("test")
            .max_total_tokens(50) // Way below what the response will use
            .on_event(Arc::new(move |event| {
                events_clone.lock().unwrap().push(event);
            }))
            .build()
            .unwrap();

        let _ = runner.execute("test task").await;
        let events = events.lock().unwrap();
        let budget_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::BudgetExceeded { .. }))
            .collect();
        assert_eq!(
            budget_events.len(),
            1,
            "expected exactly one BudgetExceeded event"
        );
        match &budget_events[0] {
            AgentEvent::BudgetExceeded { used, limit, .. } => {
                assert_eq!(*used, 200);
                assert_eq!(*limit, 50);
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn agent_runner_records_audit_trail() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Done!".into(),
            }],
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
            stop_reason: StopReason::EndTurn,
            model: Some("test-model".into()),
        }]));

        let trail = Arc::new(crate::agent::audit::InMemoryAuditTrail::new());
        let runner = AgentRunner::builder(provider)
            .name("audit-test")
            .system_prompt("You help.")
            .max_turns(5)
            .audit_trail(trail.clone())
            .build()
            .unwrap();

        let output = runner.execute("hello").await.unwrap();
        assert_eq!(output.result, "Done!");

        let entries = trail.entries().await.unwrap();
        let event_types: Vec<&str> = entries.iter().map(|e| e.event_type.as_str()).collect();
        assert!(
            event_types.contains(&"llm_response"),
            "expected llm_response, got: {event_types:?}"
        );
        assert!(
            event_types.contains(&"run_completed"),
            "expected run_completed, got: {event_types:?}"
        );
    }

    #[tokio::test]
    async fn audit_trail_captures_tool_calls() {
        let tool = Arc::new(MockTool::new("greet", "Hello!"));
        let provider = Arc::new(MockProvider::new(vec![
            // Turn 1: call the tool
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "greet".into(),
                    input: json!({"name": "world"}),
                }],
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                stop_reason: StopReason::ToolUse,
                model: None,
            },
            // Turn 2: final text
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "All done.".into(),
                }],
                usage: TokenUsage {
                    input_tokens: 15,
                    output_tokens: 3,
                    ..Default::default()
                },
                stop_reason: StopReason::EndTurn,
                model: None,
            },
        ]));

        let trail = Arc::new(crate::agent::audit::InMemoryAuditTrail::new());
        let runner = AgentRunner::builder(provider)
            .name("tool-audit-test")
            .system_prompt("You help.")
            .tool(tool)
            .max_turns(5)
            .audit_trail(trail.clone())
            .build()
            .unwrap();

        runner.execute("greet the world").await.unwrap();

        let entries = trail.entries().await.unwrap();
        let event_types: Vec<&str> = entries.iter().map(|e| e.event_type.as_str()).collect();
        assert!(
            event_types.contains(&"tool_call"),
            "expected tool_call, got: {event_types:?}"
        );
        assert!(
            event_types.contains(&"tool_result"),
            "expected tool_result, got: {event_types:?}"
        );

        // Verify tool_result has untruncated output
        let tool_result = entries
            .iter()
            .find(|e| e.event_type == "tool_result")
            .unwrap();
        assert_eq!(tool_result.payload["output"], "Hello!");

        // Verify tool_call has correct turn (not 0)
        let tool_call_entry = entries
            .iter()
            .find(|e| e.event_type == "tool_call")
            .unwrap();
        assert!(
            tool_call_entry.turn > 0,
            "tool_call turn should be > 0, got: {}",
            tool_call_entry.turn
        );
        // Verify tool_call has full input
        assert_eq!(tool_call_entry.payload["input"]["name"], "world");
    }

    #[tokio::test]
    async fn audit_trail_none_by_default() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text { text: "OK".into() }],
            usage: TokenUsage::default(),
            stop_reason: StopReason::EndTurn,
            model: None,
        }]));

        // No audit trail set — should not panic
        let runner = AgentRunner::builder(provider)
            .name("no-audit")
            .system_prompt("You help.")
            .max_turns(5)
            .build()
            .unwrap();

        let output = runner.execute("hello").await.unwrap();
        assert_eq!(output.result, "OK");
    }

    #[test]
    fn audit_user_context_builder_sets_fields() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test-agent")
            .system_prompt("prompt")
            .max_turns(5)
            .audit_user_context("alice", "acme")
            .build()
            .unwrap();

        assert_eq!(runner.audit_user_id.as_deref(), Some("alice"));
        assert_eq!(runner.audit_tenant_id.as_deref(), Some("acme"));
    }

    #[test]
    fn audit_user_context_defaults_to_none() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test-agent")
            .system_prompt("prompt")
            .max_turns(5)
            .build()
            .unwrap();

        assert!(runner.audit_user_id.is_none());
        assert!(runner.audit_tenant_id.is_none());
    }
}
