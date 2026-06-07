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

/// System prompt for context compaction. Unlike a generic summary, it pins the
/// load-bearing state an agent needs to keep working — replacing the old
/// positional "keep the last N messages" heuristic, which drops the weakest
/// attention position (the middle) where that state often lives. Kept compact
/// (bullets, no padding) so the summary itself doesn't reintroduce bloat.
pub(crate) const COMPACTION_SUMMARY_SYSTEM: &str = "You are compacting an agent's working \
conversation to free up context WITHOUT losing anything the agent needs to continue. Produce a \
summary that preserves, explicitly and completely:\n\
1. GOAL: the original task/objective and any refinements to it.\n\
2. FILES: every file created, modified, or deleted (exact paths) and what changed in each.\n\
3. TODOS: all tasks still open or in progress.\n\
4. UNRESOLVED: errors, test failures, blockers, and open questions not yet resolved.\n\
5. DECISIONS: key decisions and WHY, including approaches tried and abandoned (and the reason).\n\
Then a brief narrative of progress so far. Be COMPLETE on items 1-5 — omitting one loses work the \
agent must redo. Be concise elsewhere: compact bullet points, no preamble, no padding.";

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

/// Whether `input_tokens` has reached `fraction` of the context `window`.
/// Returns false for a zero window (unknown → no trigger). `fraction` is
/// clamped to [0.0, 1.0] defensively.
fn over_window_fraction(input_tokens: u32, window: u32, fraction: f32) -> bool {
    window > 0 && input_tokens as f32 >= fraction.clamp(0.0, 1.0) * window as f32
}

/// Default hard cap (bytes) on each FRESH tool result entering the context.
/// 256KB matches the `read` builtin's own `MAX_FILE_SIZE` — a legitimate
/// full-file read must never be truncated by this net. A single uncapped
/// result (e.g. a grep sweeping build artifacts) once reached ~1MB and blew a
/// 262K-token window in one turn. When the model window is known, the
/// effective cap is additionally clamped to `window_tokens` bytes (≈ ¼ of the
/// window in tokens) so small-window models stay protected.
pub(super) const DEFAULT_TOOL_RESULT_INGEST_CAP: usize = 256 * 1024;

/// Emergency per-result bound applied on a context-overflow error, before
/// retrying. Deliberately aggressive: the window is already blown, and full
/// content stays restorable via `fetch_full_output` when a recall store is set.
const EMERGENCY_TOOL_RESULT_MAX_BYTES: usize = 4_096;

/// True when `text` is a multi-question prose battery aimed at the user —
/// at least two non-empty lines ending with `?`. The threshold deliberately
/// ignores single trailing questions ("Anything else?") and rhetorical asides;
/// code blocks rarely produce two `?`-terminated lines.
fn is_prose_question_battery(text: &str) -> bool {
    text.lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty() && l.ends_with('?'))
        .count()
        >= 2
}

/// Harness-barrier tools: they mutate the guard/goal state sibling calls are
/// checked against, so a batch containing one is split — barriers execute
/// FIRST, serially, before the rest is guard-checked and dispatched (TOCTOU
/// fix, live finding 2026-06-07).
const BARRIER_TOOLS: &[&str] = &["set_scope", "set_goal"];

/// Fallback bound (bytes) for the summarization transcript when the model's
/// context window is unknown. 256KB ≈ 64K tokens.
const DEFAULT_SUMMARY_INPUT_MAX_BYTES: usize = 262_144;

/// Deterministic delegation nudge. Prompt-only routing has repeatedly failed
/// to make mid-tier models delegate organically — they grind through
/// substantive work solo (live trace evidence, 2026-06-07: ~30 sessions, zero
/// organic delegations across three models). After `after_tool_calls` direct
/// tool calls on ONE user request with none of `tool_names` used, the runner
/// injects a one-shot reminder that the squad exists. Deterministic and
/// model-agnostic, like the doom-loop and replan gates.
#[derive(Debug, Clone)]
pub struct DelegationNudge {
    /// Direct-tool-call count (per user request) at which the nudge fires.
    pub after_tool_calls: u32,
    /// Tool names that count as delegation — using any of them suppresses
    /// the nudge for the rest of the request.
    pub tool_names: Vec<String>,
}

/// Head+tail slice of an oversized summarization transcript: keeps the task
/// (head) and the most recent activity (tail), drops the middle.
fn bound_transcript(text: &str, budget: usize) -> String {
    const MARKER: &str = "\n[transcript abridged for summary — middle omitted]\n";
    if text.len() <= budget {
        return text.to_string();
    }
    let head_budget = budget / 4;
    let tail_budget = budget
        .saturating_sub(head_budget)
        .saturating_sub(MARKER.len());
    let head_end = crate::tool::builtins::floor_char_boundary(text, head_budget);
    let mut tail_start = text.len().saturating_sub(tail_budget);
    while tail_start < text.len() && !text.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    format!("{}{}{}", &text[..head_end], MARKER, &text[tail_start..])
}

/// Output of a completed agent run.
///
/// Returned by [`AgentRunner::execute`] on success. Contains the agent's
/// final text response and usage accounting for the entire run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
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
    /// Whether the run's [`GoalCondition`](super::goal::GoalCondition) was met,
    /// as decided by the independent goal judge. `None` when no goal was set;
    /// `Some(true)` when the judge confirmed the objective; `Some(false)` when
    /// the continuation cap was exhausted without the objective being met.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_met: Option<bool>,
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
    /// Model context window (tokens) for the proactive-compaction backstop. When
    /// `Some`, compaction triggers on real `usage.input_tokens` crossing
    /// `compaction_threshold_fraction * window`. `None` -> fall back to the
    /// `summarize_threshold` estimate path.
    pub(super) context_window_tokens: Option<u32>,
    /// Fraction of the context window at which the backstop fires (default 0.70).
    pub(super) compaction_threshold_fraction: f32,
    /// Optional callback for streaming text output.
    pub(super) on_text: Option<Arc<crate::llm::OnText>>,
    /// Optional callback for streaming reasoning (chain-of-thought) output.
    pub(super) on_reasoning: Option<Arc<crate::llm::OnReasoning>>,
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
    /// Optional re-armable per-turn interrupt. When set and triggered, the
    /// in-flight LLM generation is aborted and the turn ends cleanly (the session
    /// continues, awaiting the next `on_input` message).
    pub(super) interrupt: Option<super::interrupt::InterruptHandle>,
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
    /// Hard cap (bytes) applied to each fresh tool result at ingestion into
    /// the conversation context. Defaults to [`DEFAULT_TOOL_RESULT_INGEST_CAP`]
    /// (64KB). Full output is preserved in `context_recall_store` (when set)
    /// and in `AgentOutput::tool_call_records` regardless.
    pub(super) tool_result_ingest_cap: Option<usize>,
    /// Optional deterministic delegation nudge (see [`DelegationNudge`]).
    pub(super) delegation_nudge: Option<DelegationNudge>,
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
    /// Optional per-run context recall store. When set, every tool output is
    /// indexed by `tool_call_id` so pruned results can be restored on demand.
    pub(super) context_recall_store: Option<Arc<crate::agent::context_recall::ContextRecallStore>>,
    /// Optional shared to-do store. When set, the runner recites the open
    /// (Pending/InProgress) items at the context tail each turn — the
    /// long-horizon-planning "recitation" mechanism that keeps the live plan in
    /// recent attention and lets it survive compaction (re-recited from the
    /// store, not a lossy summary).
    pub(super) todo_store: Option<Arc<crate::tool::builtins::TodoStore>>,
    /// When true, a RED verification (`VERIFY_RESULT: FAIL` as the latest
    /// canonical sentinel in the transcript) blocks natural completion: the
    /// runner re-injects a corrective nudge and continues (bounded by
    /// `MAX_VERIFY_REPLANS`) instead of finishing on red. The long-horizon
    /// "replan on out-of-plan" signal for the no-goal path (a `GoalCondition`,
    /// if present, already gates on the same evidence via its judge).
    pub(super) replan_on_verify_fail: bool,
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
    /// Optional persistent goal: an independent judge gates EVERY natural stop
    /// and the agent keeps working (bounded by `max_continuations` and
    /// `max_turns`) until the objective is met. A shared slot so the
    /// `set_goal` tool can install/replace the goal at runtime (e.g. from
    /// `intake` acceptance criteria); a met or budget-exhausted goal
    /// auto-clears (per-request semantics). `None` inside = no goal gating.
    pub(super) goal: super::goal::GoalSlot,
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
            context_window_tokens: None,
            compaction_threshold_fraction: 0.70,
            memory: None,
            knowledge_base: None,
            on_text: None,
            on_reasoning: None,
            on_approval: None,
            tool_timeout: None,
            max_tool_output_bytes: None,
            structured_schema: None,
            on_event: None,
            guardrails: Vec::new(),
            on_question: None,
            on_input: None,
            interrupt: None,
            run_timeout: None,
            reasoning_effort: None,
            enable_reflection: false,
            tool_output_compression_threshold: None,
            tool_result_ingest_cap: Some(DEFAULT_TOOL_RESULT_INGEST_CAP),
            delegation_nudge: None,
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
            goal: None,
            goal_slot: None,
            audit_mode: super::audit::AuditMode::Full,
            audit_trail: None,
            audit_user_id: None,
            audit_tenant_id: None,
            audit_delegation_chain: Vec::new(),
            response_cache_size: None,
            tenant_tracker: None,
            context_recall_store: None,
            todo_store: None,
            replan_on_verify_fail: false,
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
            // Goal gating: how many extra continuations the independent judge has
            // granted so far (bounded by the goal's `max_continuations`). The
            // turn counter on `ctx` is the other bound — a goal continuation goes
            // through the loop top, so it consumes a turn and respects max_turns.
            let mut goal_continuations_used: u32 = 0;
            // Last settled goal verdict (the gate clears the slot when a goal
            // settles, so the verdict must outlive it for the final output).
            let mut last_goal_met: Option<bool> = None;
            // Long-horizon "replan on out-of-plan": bounded count of verify-fail
            // continuations so a permanently-red verify can't loop forever.
            let mut verify_replans_used: u32 = 0;
            const MAX_VERIFY_REPLANS: u32 = 8;
            // Track recently used tool names (last 2 turns) for dynamic tool selection
            let mut recently_used_tools: Vec<String> = Vec::new();
            let mut doom_tracker = DoomLoopTracker::new();
            let mut last_model_name: Option<String> = None;
            // Prevents infinite compaction loops: set true after compaction,
            // cleared at the start of each normal iteration.
            let mut compacted_last_turn = false;
            // Anti-thrash guard for proactive compaction: never fire two turns running.
            let mut proactive_compacted_last_turn = false;
            // Delegation-nudge state, scoped to ONE user request (reset when
            // `on_input` delivers the next message in chat mode).
            let mut nudge_tool_calls: u32 = 0;
            let mut nudge_delegated = false;
            let mut nudge_sent = false;
            // Ask-gate: one prose-battery→question-tool redirect per request.
            let mut prose_question_nudged = false;

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

                // Long-horizon planning (recitation): re-surface the live plan
                // (open todos, read from the actual store) at the context tail
                // each turn. This keeps the plan in recent attention (counters
                // lost-in-the-middle) and means it survives compaction — the
                // next turn re-recites from the store, not a lossy summary.
                // Self-gating: no open todos (trivial/chat tasks) → no block.
                // Appended to the LAST message (a user/tool-result), NOT the
                // system prompt or a new message, so prompt-cache prefixes and
                // role alternation are untouched.
                if let Some(ref store) = self.todo_store
                    && let Some(block) =
                        crate::tool::builtins::recite_open_todos(&store.open_items())
                    && let Some(last) = request.messages.last_mut()
                {
                    last.content.push(ContentBlock::Text { text: block });
                }

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
                // TTFT: captured by the on_text wrapper below; ALSO read at the
                // LlmResponse event emission (not just the tracing span) — the
                // event is what the TUI trace and /stats consume. Stays 0 for
                // cache hits and non-streaming calls.
                let ttft_ms = Arc::new(std::sync::atomic::AtomicU64::new(0));
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
                    let ttft_ms_inner = ttft_ms.clone();
                    let ttft_ref = ttft_ms_inner.clone();
                    let llm_future = async {
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
                                // Reasoning models: stream chain-of-thought live via
                                // the dedicated channel when a callback is wired.
                                match &self.on_reasoning {
                                    Some(rcb) => {
                                        let rcb = rcb.clone();
                                        let reasoning_wrapper: Box<crate::llm::OnReasoning> =
                                            Box::new(move |r: &str| rcb(r));
                                        self.provider
                                            .stream_complete_with_reasoning(
                                                request,
                                                &*wrapper,
                                                &*reasoning_wrapper,
                                            )
                                            .await
                                    }
                                    None => self.provider.stream_complete(request, &*wrapper).await,
                                }
                            }
                            None => self.provider.complete(request).await,
                        }
                    }
                    .instrument(llm_span.clone());
                    // A triggered interrupt aborts the in-flight generation: race the
                    // LLM call against the per-turn token. On interrupt, synthesize a
                    // clean end-of-turn (non-empty text — providers reject empty
                    // assistant content), rearm for the next turn, and let the
                    // existing no-tool-calls path await the next `on_input` message.
                    let mut interrupted = false;
                    let result = match self.interrupt.as_ref() {
                        Some(handle) => {
                            let token = handle.token();
                            tokio::select! {
                                biased;
                                _ = token.cancelled() => {
                                    handle.rearm();
                                    interrupted = true;
                                    Ok(crate::llm::types::CompletionResponse {
                                        content: vec![crate::llm::types::ContentBlock::Text {
                                            text: "[interrupted by user]".into(),
                                        }],
                                        stop_reason: crate::llm::types::StopReason::EndTurn,
                                        reasoning: None,
                                        usage: TokenUsage::default(),
                                        model: None,
                                    })
                                }
                                r = llm_future => r,
                            }
                        }
                        None => llm_future.await,
                    };
                    // Store successful non-streaming responses in cache (never the
                    // synthetic interrupt response). Only EndTurn responses are
                    // cached — ToolUse responses trigger side-effecting execution.
                    if !interrupted
                        && let (Ok(resp), Some(key)) = (&result, cache_key)
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
                        // Context-overflow recovery. No message-count gate: the
                        // 2026-06-07 incident overflowed at EXACTLY 5 messages
                        // (one giant fresh tool result) and a `> 5` gate here
                        // skipped recovery entirely.
                        if crate::llm::error_class::classify(&e)
                            == crate::llm::error_class::ErrorClass::ContextOverflow
                            && can_compact
                        {
                            tracing::warn!(
                                agent = %self.name,
                                error = %e,
                                "context overflow detected, attempting recovery"
                            );
                            // Deterministic first: hard-truncate oversized tool
                            // results and retry WITHOUT an LLM call — a
                            // summarization request would resend the very
                            // context that just overflowed.
                            let emergency_cap = self
                                .session_prune_config
                                .as_ref()
                                .map(|c| c.pruned_tool_result_max_bytes)
                                .unwrap_or(EMERGENCY_TOOL_RESULT_MAX_BYTES);
                            let saved = ctx.truncate_oversized_tool_results(
                                emergency_cap,
                                self.context_recall_store.is_some(),
                            );
                            if saved > 0 {
                                tracing::warn!(
                                    agent = %self.name,
                                    bytes_saved = saved,
                                    emergency_cap,
                                    "oversized tool results truncated, retrying"
                                );
                                self.emit(AgentEvent::AutoCompactionTriggered {
                                    agent: self.name.clone(),
                                    turn: ctx.current_turn(),
                                    success: true,
                                    usage: TokenUsage::default(),
                                });
                                compacted_last_turn = true;
                                continue;
                            }
                            // Nothing oversized (aggregate bloat): fall back to
                            // LLM summarization — `generate_summary` bounds its
                            // transcript, so it cannot itself overflow.
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
                // Real post-prune input token count for the proactive compaction backstop.
                let last_input_tokens = response.usage.input_tokens;

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

                // Surface the model's chain-of-thought (reasoning models only)
                // as a distinct event, ahead of the answer.
                if let Some(reasoning) = &response.reasoning
                    && !reasoning.is_empty()
                {
                    self.emit(AgentEvent::Reasoning {
                        agent: self.name.clone(),
                        turn: ctx.current_turn(),
                        text: truncate_for_event(reasoning, EVENT_MAX_PAYLOAD_BYTES),
                    });
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
                    time_to_first_token_ms: ttft_ms.load(std::sync::atomic::Ordering::Relaxed),
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
                        goal_met: None,
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

                    // STOP GATES come BEFORE awaiting the next user message —
                    // in chat mode `on_input().await` blocks until the user
                    // types again, so gates placed after it would only ever
                    // fire at session end (found inert on the TUI path).

                    // Ask-gate: a stop that is a multi-question prose battery
                    // is a CLARIFICATION, not a completion — and the user
                    // can't answer options efficiently in free text. When the
                    // structured `question` tool is registered, redirect ONCE
                    // per request (live finding 6a254624: the model reliably
                    // asks, but in prose; prompt rules alone don't move it).
                    if !prose_question_nudged
                        && self.tools.contains_key("question")
                        && ctx
                            .last_assistant_text()
                            .is_some_and(|t| is_prose_question_battery(&t))
                    {
                        prose_question_nudged = true;
                        debug!(agent = %self.name, "prose question battery; redirecting to the question tool");
                        ctx.add_user_message(
                            "[ask gate] You just asked the user several questions in free \
                             text — they cannot answer options efficiently that way. Re-ask \
                             NOW with the `question` tool: 1-4 batched questions, 2-4 \
                             concrete options each (multiple=false unless choices combine). \
                             For open questions, offer your best concrete proposals as the \
                             options."
                                .to_string(),
                        );
                        continue;
                    }

                    // Long-horizon "replan on out-of-plan": a RED verification is
                    // the canonical out-of-plan signal. Before allowing natural
                    // completion, if opted in and the latest canonical
                    // VERIFY_RESULT is FAIL, re-inject a corrective nudge and
                    // continue (bounded) instead of finishing on red. Deterministic
                    // — no judge call; a green/absent verify falls through. A
                    // GoalCondition, if present, gates on the same evidence via its
                    // judge, so this is the cheap pre-gate for the no-goal path.
                    if self.replan_on_verify_fail
                        && verify_replans_used < MAX_VERIFY_REPLANS
                        && let Some(outcome) =
                            crate::codegen::parse_latest_verify(&ctx.conversation_text())
                        && !outcome.passed
                    {
                        verify_replans_used += 1;
                        debug!(
                            agent = %self.name,
                            replan = verify_replans_used,
                            "verification RED; replanning before completion"
                        );
                        ctx.add_user_message(
                            "Verification is RED (VERIFY_RESULT: FAIL) — do NOT finish yet. \
                             Update your plan/todos, fix the underlying failure, then re-run the \
                             verify tool until it reports VERIFY_RESULT: PASS before completing."
                                .to_string(),
                        );
                        continue;
                    }

                    // Goal gating: an INDEPENDENT judge decides whether the
                    // objective is met before this natural stop is allowed (anti
                    // over-report). Not-met re-injects the judge's reason and
                    // continues (bounded by max_continuations AND the loop's
                    // max_turns guard). Met OR cap-exhausted records the verdict
                    // and CLEARS the slot (per-request semantics: a settled goal
                    // must not bill a judge call on every later chat turn).
                    let goal_now = self
                        .goal
                        .read()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone();
                    if let Some(goal) = goal_now {
                        // The judge sees the whole conversation rendered to text —
                        // including tool results (the EVIDENCE) — not just the
                        // agent's final claim, so it grades what actually happened.
                        let transcript = ctx.conversation_text();
                        let (verdict, judge_usage) = goal.evaluate(&transcript).await;
                        // Account the judge's tokens against the run's usage.
                        total_usage += judge_usage;
                        if verdict.satisfied {
                            last_goal_met = Some(true);
                            *self
                                .goal
                                .write()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
                        } else if goal_continuations_used < goal.max_continuations() {
                            goal_continuations_used += 1;
                            debug!(
                                agent = %self.name,
                                continuation = goal_continuations_used,
                                reason = %verdict.reason,
                                "goal not yet met; continuing"
                            );
                            ctx.add_user_message(goal.continuation_message(&verdict.reason));
                            continue;
                        } else {
                            // Continuation budget exhausted without meeting the
                            // goal: report not-met, stop gating.
                            debug!(
                                agent = %self.name,
                                "goal continuation budget exhausted; goal cleared (not met)"
                            );
                            last_goal_met = Some(false);
                            *self
                                .goal
                                .write()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
                        }
                    }

                    // Interactive mode: if on_input is set, ask for more input
                    // instead of returning. This enables multi-turn conversations.
                    if let Some(ref on_input) = self.on_input
                        && let Some(next_message) = on_input().await
                        && !next_message.trim().is_empty()
                    {
                        ctx.add_user_message(next_message);
                        // New user request: the delegation-nudge window restarts.
                        nudge_tool_calls = 0;
                        nudge_delegated = false;
                        nudge_sent = false;
                        prose_question_nudged = false;
                        continue;
                    }

                    let goal_met: Option<bool> = last_goal_met;

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
                        goal_met,
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

                // Harness-barrier tools (set_scope / set_goal) mutate the very
                // state their sibling calls are checked against. Executing them
                // inside the parallel batch is a TOCTOU: live session 6a251f55
                // emitted set_scope + an out-of-scope write in ONE batch, and
                // the write's pre_tool ran on the still-empty allowlist. Hoist
                // barriers out and execute them FIRST (serially, in emission
                // order) so siblings are checked against the updated state.
                // Approval/permissions already ran for the whole batch above;
                // barrier tools mutate in-process harness state only, so the
                // pre_tool guardrail pass on them is deliberately skipped.
                let mut barrier_results: Vec<ToolResult> = Vec::new();
                let mut barrier_records: Vec<ToolCallRecord> = Vec::new();
                let tool_calls: Vec<crate::llm::types::ToolCall> = if tool_calls.len() > 1
                    && tool_calls
                        .iter()
                        .any(|c| BARRIER_TOOLS.contains(&c.name.as_str()))
                {
                    let (barriers, rest): (Vec<_>, Vec<_>) = tool_calls
                        .into_iter()
                        .partition(|c| BARRIER_TOOLS.contains(&c.name.as_str()));
                    for call in barriers {
                        let (mut r, mut rec) = self
                            .execute_tools_parallel(
                                std::slice::from_ref(&call),
                                ctx.current_turn(),
                                None,
                            )
                            .await;
                        barrier_results.append(&mut r);
                        barrier_records.append(&mut rec);
                    }
                    rest
                } else {
                    tool_calls
                };

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
                // A triggered interrupt abandons the in-flight tool batch: race
                // the batch against the per-turn token. On interrupt, synthesize a
                // result for EVERY allowed call (so no tool_use is left without a
                // tool_result — providers reject that), drop the batch future (its
                // JoinSet drop kills any in-flight subprocess via kill_on_drop), and
                // leave the token CANCELLED so the next LLM call's own race ends the
                // turn cleanly → await `on_input` (history preserved).
                let mut tool_interrupted = false;
                let (mut results, batch_records) = match self.interrupt.as_ref() {
                    Some(handle) => {
                        let token = handle.token();
                        tracing::info!(
                            target: "heartbit::interrupt",
                            checkpoint = "CP4_before_tool_select",
                            is_cancelled = token.is_cancelled(),
                            turn = ctx.current_turn(),
                            tool_count = allowed_calls.len(),
                            "tool-batch interrupt race armed"
                        );
                        // Snapshot once per batch — introspection tools (the
                        // advisor) read it from ExecutionContext.transcript.
                        let transcript = Some(Arc::new(ctx.messages().to_vec()));
                        tokio::select! {
                            biased;
                            _ = token.cancelled() => {
                                tracing::info!(
                                    target: "heartbit::interrupt",
                                    checkpoint = "CP3_tool_cancel_arm_fired",
                                    turn = ctx.current_turn(),
                                    "tool-batch interrupted: synthesizing results, abandoning batch"
                                );
                                tool_interrupted = true;
                                self.synthesize_interrupted_tool_batch(&allowed_calls)
                            }
                            r = self
                                .execute_tools_parallel(&allowed_calls, ctx.current_turn(), transcript)
                                .instrument(tool_batch_span) => r,
                        }
                    }
                    None => {
                        let transcript = Some(Arc::new(ctx.messages().to_vec()));
                        self.execute_tools_parallel(&allowed_calls, ctx.current_turn(), transcript)
                            .instrument(tool_batch_span)
                            .await
                    }
                };
                tool_call_records.extend(batch_records);
                tool_call_records.extend(barrier_records);
                results.extend(barrier_results);
                results.extend(denied_results);
                results.extend(permission_denied_results);

                // LSP diagnostics: after file-modifying tools, collect diagnostics
                // and append to the tool result so the LLM sees errors immediately.
                if !tool_interrupted
                    && let Some(ref lsp) = self.lsp_manager
                {
                    self.append_lsp_diagnostics(lsp, &allowed_calls, &mut results)
                        .await;
                }

                // Compress oversized tool outputs via LLM call
                if !tool_interrupted
                    && let Some(threshold) = self.tool_output_compression_threshold
                {
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

                // Error-aware fuzzy doom reset: a batch where every call
                // succeeded is normal sequential work — only consecutive
                // ERRORING same-name batches indicate a loop.
                if !tool_interrupted && self.max_identical_tool_calls.is_some() {
                    doom_tracker.note_batch_outcome(results.iter().any(|r| r.is_error));
                }

                ctx.add_tool_results(results);

                // Hard ingestion cap: a single fresh tool result must never be
                // able to blow the context window (pruning only trims OLD
                // results; the proactive trigger only sees the PREVIOUS call's
                // usage). Full content is already in the recall store (when
                // set) and in `tool_call_records`.
                if let Some(cap) = self.tool_result_ingest_cap {
                    // Clamp to the model window when known: bytes ≈ tokens*4,
                    // so `window_tokens` BYTES bounds one result to ~¼ of the
                    // window in TOKENS (256KB default vs a 32K-token model
                    // would otherwise still overflow it).
                    let cap = match self.context_window_tokens {
                        Some(window) => cap.min(window as usize),
                        None => cap,
                    };
                    let saved =
                        ctx.cap_last_tool_results(cap, self.context_recall_store.is_some());
                    if saved > 0 {
                        debug!(
                            agent = %self.name,
                            turn = ctx.current_turn(),
                            bytes_saved = saved,
                            cap,
                            "fresh tool results capped at ingestion"
                        );
                    }
                }

                // Reflection: inject a user-role prompt that nudges the LLM to assess
                // tool results before deciding the next action (Reflexion/CRITIC pattern).
                if !tool_interrupted && self.enable_reflection {
                    ctx.add_user_message(
                        "Before proceeding, briefly reflect on the tool results above:\n\
                     1. Did you get the information you needed?\n\
                     2. Are there any errors or unexpected results?\n\
                     3. What is the best next step?"
                            .to_string(),
                    );
                }

                // Deterministic delegation nudge: after N direct tool calls on
                // one user request with no delegation tool used, remind ONCE
                // that the squad exists (prompt guidance alone has proven
                // insufficient on mid-tier models — same rationale as the
                // doom-loop and replan gates).
                if let Some(ref nudge) = self.delegation_nudge {
                    nudge_tool_calls += allowed_calls.len() as u32;
                    if allowed_calls
                        .iter()
                        .any(|c| nudge.tool_names.contains(&c.name))
                    {
                        nudge_delegated = true;
                    }
                    if !tool_interrupted
                        && !nudge_sent
                        && !nudge_delegated
                        && nudge_tool_calls >= nudge.after_tool_calls
                    {
                        nudge_sent = true;
                        debug!(
                            agent = %self.name,
                            tool_calls = nudge_tool_calls,
                            "delegation nudge injected"
                        );
                        ctx.add_user_message(format!(
                            "[delegation check] You have made {nudge_tool_calls} direct tool \
                             calls on this request without delegating. If meaningful work \
                             remains — especially independent parts — delegate it now ({}) \
                             instead of continuing alone. If the task is nearly finished, \
                             ignore this and complete it.",
                            nudge.tool_names.join(" / ")
                        ));
                    }
                }

                // Proactive compaction backstop. Prefer the REAL post-prune token
                // count vs the window fraction; fall back to the chars/4 estimate
                // vs summarize_threshold when no window is known. Never compact two
                // turns running (anti-thrash).
                let proactive_trigger = match self.context_window_tokens {
                    // max(real, estimate): the REAL count is from the PREVIOUS
                    // response and is blind to tool results that landed since;
                    // the chars/4 estimate of the live ctx catches fresh bloat.
                    Some(window) => over_window_fraction(
                        last_input_tokens.max(ctx.total_tokens()),
                        window,
                        self.compaction_threshold_fraction,
                    ),
                    None => self
                        .summarize_threshold
                        .is_some_and(|t| ctx.needs_compaction(t)),
                };
                let do_proactive_compact = !tool_interrupted
                    && ctx.message_count() > 5
                    && !proactive_compacted_last_turn
                    && proactive_trigger;
                if do_proactive_compact {
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
                proactive_compacted_last_turn = do_proactive_compact;
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
        // Bound the transcript BEFORE summarizing: the summary call must never
        // itself overflow the window it is trying to rescue. chars ≈ tokens*4,
        // so window*2 bytes targets ~half the window; generous fixed fallback
        // when the window is unknown.
        let budget = self
            .context_window_tokens
            .map(|w| (w as usize).saturating_mul(2).max(2_048))
            .unwrap_or(DEFAULT_SUMMARY_INPUT_MAX_BYTES);
        let text = bound_transcript(&ctx.conversation_text(), budget);
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
            system: COMPACTION_SUMMARY_SYSTEM.into(),
            messages: vec![Message::user(text.to_string())],
            tools: vec![],
            // Headroom for the structured schema (goal/files/todos/errors/decisions
            // + narrative). A summary that hits MaxTokens is dropped (returns None
            // below), silently skipping compaction — so don't starve it.
            max_tokens: 2048,
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

    /// Synthesize results for a tool batch abandoned by a user interrupt.
    ///
    /// Emits a `ToolCallCompleted` (so a TUI's in-flight ⏳ cell finalizes) and
    /// records an error `ToolResult`/`ToolCallRecord` for EVERY call — leaving no
    /// `tool_use` without a matching `tool_result`, which providers reject. The
    /// real batch future is dropped by the caller, killing any in-flight
    /// subprocess via `kill_on_drop`.
    fn synthesize_interrupted_tool_batch(
        &self,
        calls: &[ToolCall],
    ) -> (Vec<ToolResult>, Vec<ToolCallRecord>) {
        const MSG: &str = "Interrupted by user before completion.";
        let mut results = Vec::with_capacity(calls.len());
        let mut records = Vec::with_capacity(calls.len());
        for call in calls {
            self.emit(AgentEvent::ToolCallCompleted {
                agent: self.name.clone(),
                tool_name: call.name.clone(),
                tool_call_id: call.id.clone(),
                is_error: true,
                duration_ms: 0,
                output: MSG.to_string(),
            });
            results.push(ToolResult::error(call.id.clone(), MSG.to_string()));
            records.push(ToolCallRecord {
                tool_name: call.name.clone(),
                tool_call_id: call.id.clone(),
                input: call.input.clone(),
                output: MSG.to_string(),
                is_error: true,
                duration_ms: 0,
            });
        }
        (results, records)
    }

    /// Execute tools in parallel via JoinSet, returning results in original call order.
    ///
    /// Panicked tasks produce an error `ToolResult` so the LLM always gets a
    /// result for every `tool_use_id` it sent.
    async fn execute_tools_parallel(
        &self,
        calls: &[ToolCall],
        turn: usize,
        transcript: Option<Arc<Vec<Message>>>,
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
            transcript,
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

            // Index every tool output into the context recall store so pruned
            // results can be restored on demand via `fetch_full_output`.
            if let Some(store) = &self.context_recall_store {
                store
                    .index(&call_ids[idx], &call_names[idx], &output.content)
                    .await;
            }

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
    use super::{AgentRunner, DelegationNudge};

    // Long-horizon planning (recitation): when a todo_store has open items,
    // the runner appends a plan block to the tail of the last message.
    #[tokio::test(flavor = "multi_thread")]
    async fn recites_open_todos_at_context_tail() {
        let store = Arc::new(crate::tool::builtins::TodoStore::new());
        // Populate via the tool so we exercise the real store path.
        let tools = crate::tool::builtins::todo_tools(store.clone());
        let write = tools
            .iter()
            .find(|t| t.definition().name == "todowrite")
            .unwrap();
        write
            .execute(
                &crate::ExecutionContext::default(),
                serde_json::json!({"todos": [
                    {"content": "finish the parser", "status": "in_progress", "priority": "high"},
                    {"content": "write the docs", "status": "pending", "priority": "medium"}
                ]}),
            )
            .await
            .unwrap();

        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "done", 1, 1,
        )]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .todo_store(store)
            .max_turns(1)
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        let last_msg = reqs[0].messages.last().expect("at least one message");
        let tail_text: String = last_msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            tail_text.contains("[plan — open items"),
            "recitation block missing from tail: {tail_text:?}"
        );
        assert!(tail_text.contains("[>] finish the parser"), "{tail_text:?}");
        assert!(tail_text.contains("[ ] write the docs"), "{tail_text:?}");
    }

    // No open todos → no recitation block (trivial tasks pay nothing).
    #[tokio::test(flavor = "multi_thread")]
    async fn no_recitation_when_no_open_todos() {
        let store = Arc::new(crate::tool::builtins::TodoStore::new()); // empty
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "done", 1, 1,
        )]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .todo_store(store)
            .max_turns(1)
            .build()
            .unwrap();
        runner.execute("hi").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        let has_plan = reqs[0].messages.iter().any(|m| {
            m.content.iter().any(
                |b| matches!(b, ContentBlock::Text { text } if text.contains("[plan — open items")),
            )
        });
        assert!(
            !has_plan,
            "no plan block expected when there are no open todos"
        );
    }

    /// A "verify" tool that always reports RED (for the replan gate tests).
    struct FailingVerifyTool;
    impl Tool for FailingVerifyTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "verify".into(),
                description: "Runs verification.".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            }
        }
        fn execute(
            &self,
            _ctx: &crate::ExecutionContext,
            _input: serde_json::Value,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
        {
            Box::pin(async {
                Ok(ToolOutput::success(
                    "VERIFY_RESULT: FAIL exit_code=1 command=cargo test".to_string(),
                ))
            })
        }
    }

    fn verify_tool_call() -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "v1".into(),
                name: "verify".into(),
                input: serde_json::json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            reasoning: None,
            usage: TokenUsage::default(),
            model: None,
        }
    }

    // Long-horizon "replan on out-of-plan": with the gate ON, a RED verify
    // blocks natural completion — the runner re-injects a nudge and continues,
    // BOUNDED (≤ MAX_VERIFY_REPLANS = 8 replans) so it can't loop forever.
    #[tokio::test(flavor = "multi_thread")]
    async fn replan_on_verify_fail_blocks_completion_but_is_bounded() {
        let mut responses = vec![verify_tool_call()];
        for _ in 0..12 {
            responses.push(MockProvider::text_response("done", 1, 1));
        }
        let provider = Arc::new(MockProvider::new(responses));
        let runner = AgentRunner::builder(provider.clone())
            .name("t")
            .system_prompt("s")
            .tools(vec![Arc::new(FailingVerifyTool)])
            .replan_on_verify_fail(true)
            .max_turns(50)
            .build()
            .unwrap();
        let out = runner.execute("do it").await.unwrap();
        assert_eq!(out.result, "done", "completes after the bound is hit");
        // 1 verify tool call + 9 completion attempts (8 replans, then fall-through).
        let n = provider.captured_requests.lock().unwrap().len();
        assert_eq!(
            n, 10,
            "expected 8 bounded replans before completion; got {n} provider calls"
        );
    }

    // With the gate OFF (default), a RED verify does NOT block completion: the
    // agent finishes on the first EndTurn.
    #[tokio::test(flavor = "multi_thread")]
    async fn no_replan_when_gate_disabled() {
        let mut responses = vec![verify_tool_call()];
        for _ in 0..12 {
            responses.push(MockProvider::text_response("done", 1, 1));
        }
        let provider = Arc::new(MockProvider::new(responses));
        let runner = AgentRunner::builder(provider.clone())
            .name("t")
            .system_prompt("s")
            .tools(vec![Arc::new(FailingVerifyTool)])
            .max_turns(50)
            .build()
            .unwrap();
        let out = runner.execute("do it").await.unwrap();
        assert_eq!(out.result, "done");
        let n = provider.captured_requests.lock().unwrap().len();
        assert_eq!(
            n, 2,
            "without the gate, completes on the first EndTurn (got {n})"
        );
    }

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
            reasoning: None,
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

    #[tokio::test]
    async fn interrupt_aborts_turn_then_continues_with_next_input() {
        use crate::agent::interrupt::InterruptHandle;

        // A single real response — it must only be consumed by the turn AFTER the
        // interrupted one (proving the interrupted turn never called the provider).
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "real answer",
            3,
            2,
        )]));

        // Pre-trigger: the biased select takes the cancel arm on the first turn.
        let interrupt = InterruptHandle::new();
        interrupt.interrupt();

        // on_input yields one follow-up message, then ends the session.
        let inputs = Arc::new(std::sync::Mutex::new(vec![
            Some("follow up".to_string()),
            None,
        ]));
        let on_input: Arc<crate::agent::OnInput> = {
            let inputs = inputs.clone();
            Arc::new(move || {
                let inputs = inputs.clone();
                Box::pin(async move {
                    let mut g = inputs.lock().expect("lock");
                    if g.is_empty() { None } else { g.remove(0) }
                })
            })
        };

        let runner = AgentRunner::builder(provider)
            .name("interruptible")
            .max_turns(10)
            .on_input(on_input)
            .interrupt(interrupt)
            .build()
            .unwrap();

        let out = runner.execute("hello").await.unwrap();

        // The interrupted first turn produced no real assistant content; the
        // follow-up turn produced the real answer, and the run ended cleanly.
        assert_eq!(out.result, "real answer");
        // Only the real turn's tokens count (the synthetic interrupt added none).
        assert_eq!(out.tokens_used.input_tokens, 3);
        assert_eq!(out.tokens_used.output_tokens, 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn interrupt_during_tool_batch_abandons_it_and_ends_turn() {
        use crate::agent::interrupt::InterruptHandle;
        use std::sync::atomic::{AtomicBool, Ordering};

        // A tool that signals when it starts running, sleeps, then records that
        // it ran to completion. If the batch is interrupted, `finished` stays false.
        struct SlowTool {
            started: tokio::sync::mpsc::UnboundedSender<()>,
            finished: Arc<AtomicBool>,
        }
        impl Tool for SlowTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "slow".into(),
                    description: "Sleeps for a while.".into(),
                    input_schema: serde_json::json!({"type": "object", "properties": {}}),
                }
            }
            fn execute(
                &self,
                _ctx: &crate::ExecutionContext,
                _input: serde_json::Value,
            ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
            {
                let started = self.started.clone();
                let finished = self.finished.clone();
                Box::pin(async move {
                    let _ = started.send(());
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    finished.store(true, Ordering::SeqCst);
                    Ok(ToolOutput::success("slept".to_string()))
                })
            }
        }

        let (start_tx, mut start_rx) = tokio::sync::mpsc::unbounded_channel();
        let finished = Arc::new(AtomicBool::new(false));

        let provider = Arc::new(MockProvider::new(vec![
            // turn 1: ask for the slow tool.
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-slow".into(),
                    name: "slow".into(),
                    input: serde_json::json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            // turn 2 (after the follow-up message): the real answer.
            MockProvider::text_response("real answer", 3, 2),
        ]));

        let interrupt = InterruptHandle::new();
        // Cancel the instant the slow tool starts — i.e. mid-batch.
        let canceller = {
            let interrupt = interrupt.clone();
            tokio::spawn(async move {
                start_rx.recv().await;
                interrupt.interrupt();
            })
        };

        // on_input yields one follow-up message, then ends the session.
        let inputs = Arc::new(std::sync::Mutex::new(vec![
            Some("follow up".to_string()),
            None,
        ]));
        let on_input: Arc<crate::agent::OnInput> = {
            let inputs = inputs.clone();
            Arc::new(move || {
                let inputs = inputs.clone();
                Box::pin(async move {
                    let mut g = inputs.lock().expect("lock");
                    if g.is_empty() { None } else { g.remove(0) }
                })
            })
        };

        let runner = AgentRunner::builder(provider)
            .name("interruptible")
            .max_turns(10)
            .tool(Arc::new(SlowTool {
                started: start_tx,
                finished: finished.clone(),
            }))
            .on_input(on_input)
            .interrupt(interrupt)
            .build()
            .unwrap();

        let out = runner.execute("please run slow").await.unwrap();
        canceller.await.unwrap();

        // The interrupt abandoned the running batch and ended the turn; the
        // follow-up message was then answered normally (history preserved).
        assert_eq!(out.result, "real answer");
        assert!(
            !finished.load(Ordering::SeqCst),
            "the tool batch must be abandoned, not awaited to completion"
        );
        // Every interrupted tool_use still gets a (synthetic) result, so the
        // conversation stays valid (no orphan tool_use).
        assert!(
            out.tool_call_results
                .iter()
                .any(|r| r.is_error && r.output.contains("Interrupted")),
            "the interrupted tool must record a synthetic result"
        );
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
        let (_results, _records) = runner.execute_tools_parallel(&calls, 0, None).await;

        assert_eq!(
            captured.lock().unwrap().as_deref(),
            Some("test-tenant"),
            "tool did not receive the tenant_id from ExecutionContext"
        );
    }

    #[tokio::test]
    async fn transcript_snapshot_reaches_the_tool() {
        use std::sync::Mutex;

        use crate::ExecutionContext;
        use crate::llm::types::ToolCall;

        struct TranscriptCapturingTool {
            seen: Arc<Mutex<Option<usize>>>,
        }

        impl Tool for TranscriptCapturingTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "advisor_probe".into(),
                    description: "Counts the transcript messages it can see.".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }
            }

            fn execute(
                &self,
                ctx: &ExecutionContext,
                _input: serde_json::Value,
            ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
            {
                let seen = self.seen.clone();
                let n = ctx.transcript.as_ref().map(|t| t.len());
                Box::pin(async move {
                    *seen.lock().unwrap() = n;
                    Ok(ToolOutput::success("ok"))
                })
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let tool = Arc::new(TranscriptCapturingTool { seen: seen.clone() });
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("test")
            .max_turns(1)
            .tools(vec![tool as Arc<dyn Tool>])
            .build()
            .unwrap();

        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "advisor_probe".into(),
            input: serde_json::json!({}),
        }];
        let transcript = Arc::new(vec![
            crate::llm::types::Message::user("hello"),
            crate::llm::types::Message::user("second message"),
        ]);
        let (_r, _rec) = runner
            .execute_tools_parallel(&calls, 0, Some(transcript))
            .await;
        assert_eq!(
            *seen.lock().unwrap(),
            Some(2),
            "the tool must see the 2-message transcript snapshot"
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
                reasoning: None,
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

    #[test]
    fn over_window_fraction_triggers_at_or_above_budget() {
        assert!(super::over_window_fraction(700, 1000, 0.70));
        assert!(super::over_window_fraction(800, 1000, 0.70));
        assert!(!super::over_window_fraction(699, 1000, 0.70));
        assert!(!super::over_window_fraction(10, 0, 0.70)); // unknown window -> no trigger
    }

    #[test]
    fn compaction_summary_prompt_pins_the_preservation_schema() {
        // Guards the structured schema against accidental gutting. (Content guard;
        // summary QUALITY is verified live, not by a unit test.)
        let p = super::COMPACTION_SUMMARY_SYSTEM;
        for marker in ["GOAL", "FILES", "TODOS", "UNRESOLVED", "DECISIONS"] {
            assert!(p.contains(marker), "compaction prompt must pin {marker}");
        }
    }

    #[tokio::test]
    async fn compaction_sends_the_structured_summary_prompt() {
        // Trigger one proactive compaction and confirm the summary request carried
        // the structured preservation prompt. Both compaction paths (single-shot +
        // recursive) route through `summarize_text`, so this covers both. Proves
        // PLUMBING (the prompt is sent), not summary quality.
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_with_tokens(800),
            tool_use_with_tokens(800),
            tool_use_with_tokens(800), // fires compaction
            MockProvider::text_response("summary text", 1, 1), // the summary call
            MockProvider::text_response("done", 800, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("test")
            .tool(Arc::new(NoopTool))
            .context_window_tokens(1000)
            .max_turns(10)
            .build()
            .unwrap();
        let _ = runner.execute("do things").await.unwrap();

        let reqs = provider.captured_requests.lock().expect("lock");
        assert!(
            reqs.iter()
                .any(|r| r.system.contains("GOAL") && r.system.contains("DECISIONS")),
            "the summary call must use the structured preservation prompt"
        );
    }

    /// A tool that returns `size` bytes of output (for context-size tests).
    struct BigTool {
        size: usize,
    }
    impl Tool for BigTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "big".into(),
                description: "Returns a large output.".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            }
        }
        fn execute(
            &self,
            _ctx: &crate::ExecutionContext,
            _input: serde_json::Value,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
        {
            let size = self.size;
            Box::pin(async move { Ok(ToolOutput::success("x".repeat(size))) })
        }
    }

    /// Tool-use response for tool `name` reporting `input_tokens`.
    fn tool_use_named(name: &str, input_tokens: u32) -> crate::llm::types::CompletionResponse {
        crate::llm::types::CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: name.into(),
                input: serde_json::json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            reasoning: None,
            usage: TokenUsage {
                input_tokens,
                output_tokens: 1,
                ..Default::default()
            },
            model: None,
        }
    }

    /// Extract all ToolResult contents from a request's messages.
    fn tool_result_contents(req: &crate::llm::types::CompletionRequest) -> Vec<String> {
        req.messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                ContentBlock::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            })
            .collect()
    }

    fn overflow_error() -> Error {
        Error::Api {
            status: 400,
            message: "Prompt contains 325070 tokens and 0 draft tokens, too large for model \
                      with 262144 maximum context length"
                .into(),
        }
    }

    // --- Layer 2: hard ingestion cap on fresh tool results ---

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_cap_truncates_giant_fresh_tool_result_by_default() {
        // A 200KB fresh tool result must be capped at ingestion (default 64KB)
        // so it can never blow the next request — the 2026-06-07 incident.
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_named("big", 100),
            MockProvider::text_response("done", 100, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(BigTool { size: 400_000 }))
            .max_turns(5)
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        let contents = tool_result_contents(&reqs[1]);
        assert_eq!(contents.len(), 1);
        assert!(
            contents[0].len() <= super::DEFAULT_TOOL_RESULT_INGEST_CAP + 64,
            "fresh result must be capped: got {} bytes",
            contents[0].len()
        );
        assert!(contents[0].contains("[truncated:"), "non-restorable marker");
        assert!(
            !contents[0].contains("fetch_full_output"),
            "no restore promise without a recall store"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_cap_marker_is_restorable_with_recall_store() {
        let store = Arc::new(crate::agent::context_recall::ContextRecallStore::new());
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_named("big", 100),
            MockProvider::text_response("done", 100, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(BigTool { size: 400_000 }))
            .context_recall_store(store.clone())
            .max_turns(5)
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        {
            let reqs = provider.captured_requests.lock().unwrap();
            let contents = tool_result_contents(&reqs[1]);
            assert!(
                contents[0].contains("fetch_full_output(\"call-1\")"),
                "restorable marker must name the ref: {}",
                &contents[0][contents[0].len().saturating_sub(200)..]
            );
        }
        // Full content must be restorable from the store.
        let full = store.get("call-1").await.expect("full output indexed");
        assert_eq!(full.len(), 400_000);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_cap_clamped_by_model_window() {
        // The static default (256KB) would blow a small model window on its
        // own — when the window is known, the per-result cap clamps to
        // window-tokens bytes (≈ ¼ of the window in tokens).
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_named("big", 100),
            MockProvider::text_response("done", 100, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(BigTool { size: 10_000 }))
            .context_window_tokens(1000)
            .max_turns(5)
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        let contents = tool_result_contents(&reqs[1]);
        assert!(
            contents[0].len() <= 1_000 + 64,
            "cap must clamp to the window: got {} bytes",
            contents[0].len()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_cap_configurable_via_builder() {
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_named("big", 100),
            MockProvider::text_response("done", 100, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(BigTool { size: 50_000 }))
            .tool_result_ingest_cap(1_000)
            .max_turns(5)
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        let contents = tool_result_contents(&reqs[1]);
        assert!(
            contents[0].len() <= 1_000 + 64,
            "custom cap applies: got {} bytes",
            contents[0].len()
        );
    }

    // --- Layer 3: proactive trigger must see a fresh (uncounted) bloated ctx ---

    #[tokio::test(flavor = "multi_thread")]
    async fn proactive_compaction_sees_fresh_context_estimate() {
        // The provider reports LOW input tokens (100 < 700 budget) but the
        // accumulated fresh tool results push the chars/4 estimate of the ctx
        // far over the window fraction — compaction must still fire.
        let events: Arc<std::sync::Mutex<Vec<crate::agent::events::AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let provider = Arc::new(MockProvider::new(vec![
            tool_use_named("big", 100),
            tool_use_named("big", 100),
            tool_use_named("big", 100), // turn 3: message_count > 5, estimate >> 700
            MockProvider::text_response("summary text", 1, 1),
            MockProvider::text_response("done", 100, 1),
        ]));
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(BigTool { size: 10_000 }))
            .context_window_tokens(1000)
            .max_turns(10)
            .on_event(Arc::new(move |ev| {
                events_clone.lock().expect("lock").push(ev);
            }))
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        let summarized = events
            .lock()
            .expect("lock")
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::agent::events::AgentEvent::ContextSummarized { .. }
                )
            })
            .count();
        assert_eq!(
            summarized, 1,
            "estimate-driven proactive compaction must fire exactly once"
        );
    }

    // --- Layer 4: bounded summary transcript ---

    #[tokio::test(flavor = "multi_thread")]
    async fn summary_transcript_is_bounded() {
        // The summary request must never resend an unbounded transcript —
        // otherwise compaction itself overflows the window it's trying to save.
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_named("big", 800),
            tool_use_named("big", 800),
            tool_use_named("big", 800), // fires proactive compaction (800 >= 700)
            MockProvider::text_response("summary text", 1, 1),
            MockProvider::text_response("done", 800, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(BigTool { size: 5_000 }))
            .context_window_tokens(1000)
            .max_turns(10)
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        let summary_req = reqs
            .iter()
            .find(|r| r.system.contains("GOAL"))
            .expect("summary request captured");
        let user_text: String = summary_req.messages[0]
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        // window=1000 tokens → budget = 1000*2 bytes (half the window in chars)
        assert!(
            user_text.len() <= 2_000 + 128,
            "summary transcript must be bounded: got {} bytes",
            user_text.len()
        );
        assert!(
            user_text.contains("[transcript abridged"),
            "abridge marker present"
        );
    }

    // --- Layer 5: reactive overflow recovery ---

    #[tokio::test(flavor = "multi_thread")]
    async fn reactive_overflow_truncates_oversized_results_and_retries_without_summary() {
        // On a classified context-overflow error, the runner must FIRST
        // hard-truncate oversized tool results (deterministic, no LLM) and
        // retry — NOT summarize (the summary call would itself overflow).
        let provider = Arc::new(MockProvider::new_with_results(vec![
            Ok(tool_use_named("big", 100)),
            Err(overflow_error()),
            Ok(MockProvider::text_response("done", 100, 1)),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(BigTool { size: 60_000 }))
            .max_turns(5)
            .build()
            .unwrap();
        let output = runner.execute("go").await.expect("run must recover");
        assert_eq!(output.result, "done");

        let reqs = provider.captured_requests.lock().unwrap();
        assert!(
            !reqs.iter().any(|r| r.system.contains("GOAL")),
            "deterministic recovery must not call the summarizer"
        );
        // The retried request carries the emergency-truncated result.
        let contents = tool_result_contents(&reqs[2]);
        assert!(
            contents[0].len() <= 4_096 + 64,
            "retried result emergency-truncated: got {} bytes",
            contents[0].len()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reactive_overflow_recovers_with_few_messages_via_summary() {
        // Overflow on the FIRST call (message_count == 1): nothing to truncate,
        // so the runner falls back to summarization and retries. The incident
        // failed here because of a `message_count > 5` gate — pinned removed.
        let provider = Arc::new(MockProvider::new_with_results(vec![
            Err(overflow_error()),
            Ok(MockProvider::text_response("summary text", 1, 1)),
            Ok(MockProvider::text_response("done", 100, 1)),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .max_turns(5)
            .build()
            .unwrap();
        let output = runner.execute("go").await.expect("run must recover");
        assert_eq!(output.result, "done");

        let reqs = provider.captured_requests.lock().unwrap();
        assert!(
            reqs.iter().any(|r| r.system.contains("GOAL")),
            "summary fallback must have been attempted"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn goal_gates_before_next_input_in_chat_mode() {
        // Chat mode (on_input set): the goal judge must gate EACH natural stop
        // BEFORE the runner awaits the next user message — otherwise the gate
        // only fires at session end (inert mid-session). Once met, the goal
        // auto-clears (per-request semantics): later stops pay no judge call.
        use std::sync::atomic::{AtomicUsize, Ordering};

        let main = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("attempt 1", 10, 1),
            MockProvider::text_response("attempt 2", 10, 1),
            MockProvider::text_response("final", 10, 1),
        ]));
        let judge = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("GOAL_MET: NO: no evidence yet", 1, 1),
            MockProvider::text_response("GOAL_MET: YES", 1, 1),
        ]));
        let inputs = Arc::new(AtomicUsize::new(0));
        let inputs_c = inputs.clone();
        let on_input: Arc<crate::agent::runner::OnInput> = Arc::new(move || {
            let n = inputs_c.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if n == 0 {
                    Some("another request".to_string())
                } else {
                    None
                }
            })
        });
        let runner = AgentRunner::builder(main.clone())
            .name("test")
            .system_prompt("sys")
            .goal(crate::agent::goal::GoalCondition::new(
                "demonstrate the result",
                Arc::new(crate::llm::BoxedProvider::from_arc(judge.clone())),
            ))
            .on_input(on_input)
            .max_turns(10)
            .build()
            .unwrap();
        let out = runner.execute("do the thing").await.unwrap();

        assert_eq!(out.result, "final");
        assert_eq!(out.goal_met, Some(true));
        let judge_calls = judge.captured_requests.lock().unwrap().len();
        assert_eq!(
            judge_calls, 2,
            "judge gates each stop until met, then auto-clears (no 3rd call)"
        );
        let main_reqs = main.captured_requests.lock().unwrap();
        assert_eq!(main_reqs.len(), 3, "continuation + chat turn both happened");
        let texts: Vec<String> = main_reqs[1]
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("not yet complete")),
            "the judge's continuation reached the agent BEFORE any input wait: {texts:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn barrier_tool_seeds_guard_before_sibling_mutations() {
        // LIVE FINDING (session 6a251f55, 2026-06-07): the model emitted
        // set_scope + an out-of-scope write in ONE parallel batch — the
        // write's pre_tool ran on the still-empty allowlist (TOCTOU) and the
        // mutation went through. Harness-barrier tools must execute FIRST,
        // before sibling calls are guard-checked or dispatched.
        let guard = Arc::new(crate::agent::guardrails::ScopeGuard::new(vec![]));
        let provider = Arc::new(MockProvider::new(vec![
            crate::llm::types::CompletionResponse {
                content: vec![
                    ContentBlock::ToolUse {
                        id: "c-scope".into(),
                        name: "set_scope".into(),
                        input: serde_json::json!({"paths": ["/tmp/x/utils.py"]}),
                    },
                    ContentBlock::ToolUse {
                        id: "c-ok".into(),
                        name: "write".into(),
                        input: serde_json::json!({"file_path": "/tmp/x/utils.py", "content": "a"}),
                    },
                    ContentBlock::ToolUse {
                        id: "c-deny".into(),
                        name: "write".into(),
                        input: serde_json::json!({"file_path": "/tmp/x/notes.txt", "content": "b"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            MockProvider::text_response("done", 10, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(crate::tool::set_scope::SetScopeTool::new(
                guard.clone(),
            )))
            .tool(Arc::new(NamedTool { name: "write" }))
            .guardrail(guard)
            .max_turns(5)
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        // The 2nd request carries the batch's tool results — find them by id.
        let results: Vec<(String, String, bool)> = reqs[1]
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => Some((tool_use_id.clone(), content.clone(), *is_error)),
                _ => None,
            })
            .collect();
        let by_id = |id: &str| {
            results
                .iter()
                .find(|(i, _, _)| i == id)
                .unwrap_or_else(|| panic!("missing result {id}: {results:?}"))
        };
        let (_, scope_out, scope_err) = by_id("c-scope");
        assert!(!scope_err, "set_scope itself succeeds: {scope_out}");
        let (_, ok_out, ok_err) = by_id("c-ok");
        assert!(!ok_err, "in-scope sibling write allowed: {ok_out}");
        let (_, deny_out, deny_err) = by_id("c-deny");
        assert!(
            *deny_err && deny_out.contains("scope guard"),
            "out-of-scope sibling write must be DENIED by the freshly-seeded \
             guard (TOCTOU fix): err={deny_err} out={deny_out}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn prose_question_battery_triggers_ask_gate() {
        // Live finding (session 6a254624): the model asks its clarification
        // battery in PROSE — the user can't answer options efficiently and
        // the structured channel sits unused. When the stop is a multi-
        // question prose battery AND the question tool is registered, the
        // runner deterministically redirects to the tool (once per request).
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response(
                "Avant de commencer :\n1. Quel langage ?\n2. Interface web ou CLI ?\n3. Quelle persistance ?",
                10,
                5,
            ),
            tool_use_named("question", 10), // the redirect worked
            MockProvider::text_response("done", 10, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(NamedTool { name: "question" }))
            .max_turns(6)
            .build()
            .unwrap();
        let out = runner.execute("crée un CRM").await.unwrap();
        assert_eq!(out.result, "done");
        let reqs = provider.captured_requests.lock().unwrap();
        let texts: Vec<String> = reqs[1]
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("[ask gate]")),
            "the prose battery must be redirected to the question tool: {texts:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn single_trailing_question_does_not_trigger_ask_gate() {
        // "Anything else?" endings are not clarification batteries.
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "Voilà, c'est fait. Autre chose ?",
            10,
            5,
        )]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(NamedTool { name: "question" }))
            .max_turns(4)
            .build()
            .unwrap();
        runner.execute("petite tâche").await.unwrap();
        assert_eq!(
            provider.captured_requests.lock().unwrap().len(),
            1,
            "no redirect turn for a single trailing question"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ask_gate_inactive_without_question_tool() {
        let provider = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "1. Quel langage ?\n2. Quelle interface ?",
            10,
            5,
        )]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .max_turns(4)
            .build()
            .unwrap();
        runner.execute("crée un CRM").await.unwrap();
        assert_eq!(
            provider.captured_requests.lock().unwrap().len(),
            1,
            "no question tool registered → prose questions pass through"
        );
    }

    // --- Deterministic delegation nudge ---

    /// A no-op tool with a configurable name (stands in for delegate_task).
    struct NamedTool {
        name: &'static str,
    }
    impl Tool for NamedTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.into(),
                description: format!("Mock {}", self.name),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            }
        }
        fn execute(
            &self,
            _ctx: &crate::ExecutionContext,
            _input: serde_json::Value,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
        {
            Box::pin(async { Ok(ToolOutput::success("ok")) })
        }
    }

    fn nudge_text_in(req: &crate::llm::types::CompletionRequest) -> usize {
        req.messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(
                |b| matches!(b, ContentBlock::Text { text } if text.contains("[delegation check]")),
            )
            .count()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delegation_nudge_fires_once_after_threshold() {
        // 3 direct tool calls with after_tool_calls=2 → the nudge is injected
        // exactly ONCE (after the 2nd call), visible in later requests.
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_named("work", 100),
            tool_use_named("work", 100),
            tool_use_named("work", 100),
            MockProvider::text_response("done", 100, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(NamedTool { name: "work" }))
            .delegation_nudge(DelegationNudge {
                after_tool_calls: 2,
                tool_names: vec!["delegate_task".into()],
            })
            .max_turns(10)
            .build()
            .unwrap();
        runner.execute("substantive task").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        assert_eq!(
            nudge_text_in(&reqs[2]),
            1,
            "nudge present after the threshold"
        );
        assert_eq!(
            nudge_text_in(reqs.last().unwrap()),
            1,
            "nudge injected exactly once, not repeated"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delegation_nudge_suppressed_when_delegation_used() {
        // The first call IS a delegation tool → no nudge ever fires.
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_named("delegate_task", 100),
            tool_use_named("work", 100),
            tool_use_named("work", 100),
            tool_use_named("work", 100),
            MockProvider::text_response("done", 100, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(NamedTool { name: "work" }))
            .tool(Arc::new(NamedTool {
                name: "delegate_task",
            }))
            .delegation_nudge(DelegationNudge {
                after_tool_calls: 2,
                tool_names: vec!["delegate_task".into()],
            })
            .max_turns(10)
            .build()
            .unwrap();
        runner.execute("substantive task").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        assert_eq!(
            nudge_text_in(reqs.last().unwrap()),
            0,
            "delegating suppresses the nudge"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn delegation_nudge_absent_by_default() {
        let provider = Arc::new(MockProvider::new(vec![
            tool_use_named("work", 100),
            tool_use_named("work", 100),
            tool_use_named("work", 100),
            MockProvider::text_response("done", 100, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(NamedTool { name: "work" }))
            .max_turns(10)
            .build()
            .unwrap();
        runner.execute("task").await.unwrap();

        let reqs = provider.captured_requests.lock().unwrap();
        assert_eq!(
            nudge_text_in(reqs.last().unwrap()),
            0,
            "no nudge when not configured"
        );
    }

    /// Helper to build a tool-use response where the LLM reports `input_tokens`.
    fn tool_use_with_tokens(input_tokens: u32) -> crate::llm::types::CompletionResponse {
        crate::llm::types::CompletionResponse {
            content: vec![crate::llm::types::ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "noop".into(),
                input: serde_json::json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            reasoning: None,
            usage: TokenUsage {
                input_tokens,
                output_tokens: 1,
                ..Default::default()
            },
            model: None,
        }
    }

    #[tokio::test]
    async fn proactive_compaction_fires_when_real_tokens_cross_window_fraction() {
        // window=1000, fraction=0.70 → budget=700. Three tool-use turns drive
        // message_count to 7 (>5) while reporting 800 input tokens (>= 700).
        // Proactive compaction fires on turn 3, consuming the summary response.
        // Turn 4 is a terminal text response.
        let events: Arc<std::sync::Mutex<Vec<crate::agent::events::AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let provider = Arc::new(MockProvider::new(vec![
            tool_use_with_tokens(800),                         // turn 1 main LLM
            tool_use_with_tokens(800),                         // turn 2 main LLM
            tool_use_with_tokens(800),                         // turn 3 main LLM — fires compaction
            MockProvider::text_response("summary text", 1, 1), // summary call
            MockProvider::text_response("done", 800, 1),       // turn 4 main LLM
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("test")
            .tool(Arc::new(NoopTool))
            .context_window_tokens(1000)
            .max_turns(10)
            .on_event(Arc::new(move |ev| {
                events_clone.lock().expect("lock").push(ev);
            }))
            .build()
            .unwrap();

        let _output = runner.execute("do things").await.unwrap();

        let summarized = events
            .lock()
            .expect("lock")
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::agent::events::AgentEvent::ContextSummarized { .. }
                )
            })
            .count();
        assert_eq!(
            summarized, 1,
            "expected exactly 1 ContextSummarized event, got {summarized}"
        );
    }

    #[tokio::test]
    async fn proactive_compaction_does_not_fire_below_fraction() {
        // Same shape but input_tokens=600 (< 700 budget) → ZERO ContextSummarized.
        // Three tool-use turns still needed so message_count crosses 5.
        let events: Arc<std::sync::Mutex<Vec<crate::agent::events::AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let provider = Arc::new(MockProvider::new(vec![
            tool_use_with_tokens(600),                   // turn 1
            tool_use_with_tokens(600),                   // turn 2
            tool_use_with_tokens(600),                   // turn 3 — 600 < 700, no compact
            MockProvider::text_response("done", 600, 1), // turn 4 terminal
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("test")
            .tool(Arc::new(NoopTool))
            .context_window_tokens(1000)
            .max_turns(10)
            .on_event(Arc::new(move |ev| {
                events_clone.lock().expect("lock").push(ev);
            }))
            .build()
            .unwrap();

        let _output = runner.execute("do things").await.unwrap();

        let summarized = events
            .lock()
            .expect("lock")
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::agent::events::AgentEvent::ContextSummarized { .. }
                )
            })
            .count();
        assert_eq!(
            summarized, 0,
            "expected 0 ContextSummarized events below fraction, got {summarized}"
        );
    }

    #[tokio::test]
    async fn proactive_compaction_does_not_thrash_two_turns_running() {
        // Turns 1-3 drive message_count >5. Turn 3 fires compaction (consuming
        // the summary response). Turn 4 is also a high-token tool-use turn but
        // is suppressed by the anti-thrash flag. Turn 5 is a terminal text
        // response. Net result: exactly 1 ContextSummarized.
        let events: Arc<std::sync::Mutex<Vec<crate::agent::events::AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let provider = Arc::new(MockProvider::new(vec![
            tool_use_with_tokens(800),                         // turn 1 main LLM
            tool_use_with_tokens(800),                         // turn 2 main LLM
            tool_use_with_tokens(800),                         // turn 3 main LLM — fires compaction
            MockProvider::text_response("summary text", 1, 1), // summary call (turn 3)
            tool_use_with_tokens(800), // turn 4 main LLM — suppressed by anti-thrash
            MockProvider::text_response("done", 800, 1), // turn 5 terminal
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("test")
            .tool(Arc::new(NoopTool))
            .context_window_tokens(1000)
            .max_turns(10)
            .on_event(Arc::new(move |ev| {
                events_clone.lock().expect("lock").push(ev);
            }))
            .build()
            .unwrap();

        let _output = runner.execute("do things").await.unwrap();

        let summarized = events
            .lock()
            .expect("lock")
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::agent::events::AgentEvent::ContextSummarized { .. }
                )
            })
            .count();
        assert_eq!(
            summarized, 1,
            "anti-thrash guard must cap at exactly 1 ContextSummarized, got {summarized}"
        );
    }

    /// A streaming mock that emits its first token after a measurable delay —
    /// instant mocks would legitimately record a 0ms TTFT.
    struct SlowStreamingProvider;

    impl crate::llm::LlmProvider for SlowStreamingProvider {
        async fn complete(
            &self,
            _request: crate::llm::types::CompletionRequest,
        ) -> Result<CompletionResponse, Error> {
            Ok(MockProvider::text_response("hi", 1, 1))
        }

        async fn stream_complete(
            &self,
            _request: crate::llm::types::CompletionRequest,
            on_text: &crate::llm::OnText,
        ) -> Result<CompletionResponse, Error> {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            on_text("hi");
            Ok(MockProvider::text_response("hi", 1, 1))
        }
    }

    // TTFT regression (found by the TUI's own /analyze on a live trace): the
    // LlmResponse EVENT hardcoded time_to_first_token_ms: 0 even though the
    // on_text wrapper captured it — the value only ever reached the tracing
    // span. The event is what the TUI trace and /stats consume.
    #[tokio::test(flavor = "multi_thread")]
    async fn llm_response_event_carries_streaming_ttft() {
        let events: Arc<std::sync::Mutex<Vec<crate::agent::events::AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let ev = events.clone();
        let runner = AgentRunner::builder(Arc::new(SlowStreamingProvider))
            .name("t")
            .system_prompt("s")
            .max_turns(1)
            .on_text(Arc::new(|_| {}))
            .on_event(Arc::new(move |e| ev.lock().expect("lock").push(e)))
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        let evs = events.lock().expect("lock");
        let ttft = evs
            .iter()
            .find_map(|e| match e {
                crate::agent::events::AgentEvent::LlmResponse {
                    time_to_first_token_ms,
                    ..
                } => Some(*time_to_first_token_ms),
                _ => None,
            })
            .expect("LlmResponse event emitted");
        assert!(
            ttft >= 10,
            "streaming TTFT must reach the LlmResponse event, got {ttft}ms"
        );
    }
}
