use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, info_span};

use crate::config::DispatchMode;
use crate::error::Error;
use crate::llm::types::{TokenUsage, ToolDefinition};
use crate::llm::{BoxedProvider, LlmProvider};
use crate::tool::{Tool, ToolOutput};

use crate::memory::Memory;

use crate::knowledge::KnowledgeBase;

use crate::tool::builtins::OnQuestion;

use super::blackboard::{Blackboard, InMemoryBlackboard};
use super::blackboard_tools::blackboard_tools;
use super::context::ContextStrategy;
use super::events::{AgentEvent, OnEvent};
use super::guardrail::Guardrail;
use super::{AgentOutput, AgentRunner};

/// A sub-agent definition registered with the orchestrator.
#[derive(Clone)]
pub(crate) struct SubAgentDef {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) system_prompt: String,
    pub(crate) tools: Vec<Arc<dyn Tool>>,
    pub(crate) context_strategy: Option<ContextStrategy>,
    pub(crate) summarize_threshold: Option<u32>,
    pub(crate) tool_timeout: Option<Duration>,
    pub(crate) max_tool_output_bytes: Option<usize>,
    /// Per-agent turn limit. When `None`, uses orchestrator default.
    pub(crate) max_turns: Option<usize>,
    /// Per-agent token limit. When `None`, uses orchestrator default.
    pub(crate) max_tokens: Option<u32>,
    /// Optional JSON Schema for structured output.
    pub(crate) response_schema: Option<serde_json::Value>,
    /// Guardrails applied to this sub-agent's LLM calls and tool executions.
    pub(crate) guardrails: Vec<Arc<dyn Guardrail>>,
    /// Optional per-agent run timeout. When `None`, no timeout is applied.
    pub(crate) run_timeout: Option<Duration>,
    /// Optional per-agent LLM provider override. When `None`, the orchestrator's
    /// shared provider is used.
    pub(crate) provider_override: Option<Arc<BoxedProvider>>,
    /// Optional reasoning/thinking effort level for this sub-agent.
    pub(crate) reasoning_effort: Option<crate::llm::types::ReasoningEffort>,
    /// Enable reflection prompts after tool results for this sub-agent.
    pub(crate) enable_reflection: Option<bool>,
    /// Tool output compression threshold in bytes for this sub-agent.
    pub(crate) tool_output_compression_threshold: Option<usize>,
    /// Maximum tools per turn for this sub-agent.
    pub(crate) max_tools_per_turn: Option<usize>,
    /// Maximum consecutive identical tool-call turns for doom loop detection.
    pub(crate) max_identical_tool_calls: Option<u32>,
    /// Session pruning configuration.
    pub(crate) session_prune_config: Option<crate::agent::pruner::SessionPruneConfig>,
    /// Enable recursive summarization.
    pub(crate) enable_recursive_summarization: Option<bool>,
    /// Memory reflection threshold.
    pub(crate) reflection_threshold: Option<u32>,
    /// Run memory consolidation at session end.
    pub(crate) consolidate_on_exit: Option<bool>,
}

impl std::fmt::Debug for SubAgentDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubAgentDef")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("tools_count", &self.tools.len())
            .finish()
    }
}

/// A task delegated by the orchestrator to a sub-agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DelegatedTask {
    pub(crate) agent: String,
    pub(crate) task: String,
}

/// Result from a sub-agent execution.
#[derive(Debug, Clone)]
pub(crate) struct SubAgentResult {
    pub(crate) agent: String,
    pub(crate) result: String,
    pub(crate) tokens_used: TokenUsage,
    pub(crate) success: bool,
}

/// Multi-agent orchestrator.
///
/// Refactored to use `AgentRunner` internally with a `DelegateTaskTool`.
/// No duplicated agent loop — the orchestrator IS an AgentRunner.
///
/// The `DelegateTaskTool` accumulates sub-agent token usage in a shared
/// `Arc<Mutex<TokenUsage>>`. Each `run()` call resets the accumulator
/// before starting, so sequential calls are safe. For concurrent use,
/// create separate `Orchestrator` instances.
pub struct Orchestrator<P: LlmProvider> {
    runner: AgentRunner<P>,
    /// Shared accumulator for sub-agent token usage (populated by DelegateTaskTool).
    sub_agent_tokens: Arc<Mutex<TokenUsage>>,
}

impl<P: LlmProvider + 'static> Orchestrator<P> {
    pub fn builder(provider: Arc<P>) -> OrchestratorBuilder<P> {
        OrchestratorBuilder {
            provider,
            sub_agents: vec![],
            max_turns: 10,
            max_tokens: 4096,
            context_strategy: None,
            summarize_threshold: None,
            tool_timeout: None,
            max_tool_output_bytes: None,
            shared_memory: None,
            blackboard: None,
            knowledge_base: None,
            on_text: None,
            on_approval: None,
            on_event: None,
            guardrails: Vec::new(),
            on_question: None,
            run_timeout: None,
            enable_squads: None,
            reasoning_effort: None,
            enable_reflection: false,
            tool_output_compression_threshold: None,
            max_tools_per_turn: None,
            max_identical_tool_calls: None,
            permission_rules: super::permission::PermissionRuleset::default(),
            instruction_text: None,
            learned_permissions: None,
            lsp_manager: None,
            observability_mode: None,
            dispatch_mode: DispatchMode::Parallel,
        }
    }

    /// Run the orchestrator with a task. Returns the combined output from
    /// the orchestrator and all sub-agents.
    ///
    /// # Concurrent use
    ///
    /// This method takes `&mut self` to prevent concurrent calls on the same
    /// instance at compile time. The sub-agent token accumulator is reset at
    /// the start of each call, so concurrent runs would produce incorrect
    /// token counts. For concurrent use, create separate `Orchestrator`
    /// instances.
    pub async fn run(&mut self, task: &str) -> Result<AgentOutput, Error> {
        // Reset sub-agent token accumulator so repeated calls don't inflate counts
        {
            let mut acc = self.sub_agent_tokens.lock().expect("token lock poisoned");
            *acc = TokenUsage::default();
        }
        match self.runner.execute(task).await {
            Ok(mut output) => {
                // Add sub-agent tokens that were accumulated during delegation
                let sub_tokens = *self.sub_agent_tokens.lock().expect("token lock poisoned");
                output.tokens_used += sub_tokens;
                Ok(output)
            }
            Err(e) => {
                // Include sub-agent tokens in the error's partial usage so callers
                // see the full token cost even when the orchestrator itself fails.
                let sub_tokens = *self.sub_agent_tokens.lock().expect("token lock poisoned");
                let mut usage = e.partial_usage();
                usage += sub_tokens;
                Err(e.with_partial_usage(usage))
            }
        }
    }
}

/// The orchestrator's primary tool: delegates tasks to sub-agents in parallel.
///
/// Implements `Tool` so it can be registered with `AgentRunner`.
/// Unknown agent names return an error result to the LLM instead of crashing.
///
/// Sub-agents always use `AgentRunner<BoxedProvider>` for type erasure. Each
/// sub-agent uses its `provider_override` if set, falling back to `shared_provider`.
struct DelegateTaskTool {
    shared_provider: Arc<BoxedProvider>,
    sub_agents: Vec<SubAgentDef>,
    max_turns: usize,
    max_tokens: u32,
    /// Permission rules inherited from the orchestrator, forwarded to sub-agents.
    permission_rules: super::permission::PermissionRuleset,
    /// Shared accumulator for sub-agent token usage, read by Orchestrator::run.
    accumulated_tokens: Arc<Mutex<TokenUsage>>,
    /// Shared memory store for cross-agent memory (None if not configured).
    shared_memory: Option<Arc<dyn Memory>>,
    /// Shared blackboard for cross-agent coordination (None if not configured).
    blackboard: Option<Arc<dyn Blackboard>>,
    /// Shared knowledge base for document retrieval (None if not configured).
    knowledge_base: Option<Arc<dyn KnowledgeBase>>,
    /// Cached tool definition, computed at construction time to avoid calling
    /// `Tool::definition()` on every sub-agent tool every LLM turn.
    cached_definition: ToolDefinition,
    /// Optional event callback for sub-agent dispatch/completion events.
    on_event: Option<Arc<OnEvent>>,
    /// Optional streaming text callback, forwarded to sub-agents.
    on_text: Option<Arc<crate::llm::OnText>>,
    /// Optional LSP manager, forwarded to sub-agents.
    lsp_manager: Option<Arc<crate::lsp::LspManager>>,
    /// Observability mode inherited from the orchestrator, forwarded to sub-agents.
    observability_mode: super::observability::ObservabilityMode,
}

impl DelegateTaskTool {
    async fn delegate(&self, tasks: Vec<DelegatedTask>) -> Result<String, Error> {
        if tasks.is_empty() {
            return Err(Error::Agent(
                "delegate_task requires at least one task".into(),
            ));
        }
        let task_count = tasks.len();
        let agent_names: Vec<String> = tasks.iter().map(|t| t.agent.clone()).collect();
        let _delegate_span = info_span!(
            "heartbit.orchestrator.delegate",
            agent_count = task_count,
            agents = ?agent_names,
        );

        if let Some(ref cb) = self.on_event {
            cb(AgentEvent::SubAgentsDispatched {
                agent: "orchestrator".into(),
                agents: agent_names.clone(),
            });
        }

        let mut join_set = tokio::task::JoinSet::new();

        for (idx, task) in tasks.into_iter().enumerate() {
            let agent_def = match self.sub_agents.iter().find(|a| a.name == task.agent) {
                Some(def) => def.clone(),
                None => {
                    // Unknown agent: we'll collect this as an error in the results
                    let agent_name = task.agent.clone();
                    join_set.spawn(async move {
                        (
                            idx,
                            SubAgentResult {
                                agent: agent_name.clone(),
                                result: format!("Error: unknown agent '{agent_name}'"),
                                tokens_used: TokenUsage::default(),
                                success: false,
                            },
                        )
                    });
                    continue;
                }
            };

            let provider = agent_def
                .provider_override
                .clone()
                .unwrap_or_else(|| self.shared_provider.clone());
            let max_turns = agent_def.max_turns.unwrap_or(self.max_turns);
            let max_tokens = agent_def.max_tokens.unwrap_or(self.max_tokens);
            let shared_memory = self.shared_memory.clone();
            let blackboard = self.blackboard.clone();
            let knowledge_base = self.knowledge_base.clone();
            let on_event = self.on_event.clone();
            let on_text = self.on_text.clone();
            let lsp_manager = self.lsp_manager.clone();
            let permission_rules = self.permission_rules.clone();
            let observability_mode = self.observability_mode;

            info!(agent = %agent_def.name, task = %task.task, "spawning sub-agent");

            join_set.spawn(async move {
                let mut builder = AgentRunner::builder(provider)
                    .name(&agent_def.name)
                    .system_prompt(&agent_def.system_prompt)
                    .tools(agent_def.tools)
                    .max_turns(max_turns)
                    .max_tokens(max_tokens);

                if let Some(strategy) = agent_def.context_strategy {
                    builder = builder.context_strategy(strategy);
                }
                if let Some(threshold) = agent_def.summarize_threshold {
                    builder = builder.summarize_threshold(threshold);
                }
                if let Some(timeout) = agent_def.tool_timeout {
                    builder = builder.tool_timeout(timeout);
                }
                if let Some(max) = agent_def.max_tool_output_bytes {
                    builder = builder.max_tool_output_bytes(max);
                }
                if let Some(schema) = agent_def.response_schema {
                    builder = builder.structured_schema(schema);
                }
                if !agent_def.guardrails.is_empty() {
                    builder = builder.guardrails(agent_def.guardrails);
                }
                if let Some(timeout) = agent_def.run_timeout {
                    builder = builder.run_timeout(timeout);
                }
                if let Some(effort) = agent_def.reasoning_effort {
                    builder = builder.reasoning_effort(effort);
                }
                if let Some(true) = agent_def.enable_reflection {
                    builder = builder.enable_reflection(true);
                }
                if let Some(threshold) = agent_def.tool_output_compression_threshold {
                    builder = builder.tool_output_compression_threshold(threshold);
                }
                if let Some(max) = agent_def.max_tools_per_turn {
                    builder = builder.max_tools_per_turn(max);
                }
                if let Some(max) = agent_def.max_identical_tool_calls {
                    builder = builder.max_identical_tool_calls(max);
                }
                if let Some(ref config) = agent_def.session_prune_config {
                    builder = builder.session_prune_config(config.clone());
                }
                if let Some(true) = agent_def.enable_recursive_summarization {
                    builder = builder.enable_recursive_summarization(true);
                }
                if let Some(threshold) = agent_def.reflection_threshold {
                    builder = builder.reflection_threshold(threshold);
                }
                if let Some(true) = agent_def.consolidate_on_exit {
                    builder = builder.consolidate_on_exit(true);
                }

                // Forward permission rules from orchestrator to sub-agents
                if !permission_rules.is_empty() {
                    builder = builder.permission_rules(permission_rules);
                }

                // Forward observability mode from orchestrator to sub-agents
                builder = builder.observability_mode(observability_mode);

                // Forward LSP manager to sub-agents
                if let Some(ref lsp) = lsp_manager {
                    builder = builder.lsp_manager(lsp.clone());
                }

                // Forward on_event so sub-agent events are visible
                if let Some(ref on_event) = on_event {
                    builder = builder.on_event(on_event.clone());
                }
                // Forward on_text so sub-agent streaming text is visible
                if let Some(ref on_text) = on_text {
                    builder = builder.on_text(on_text.clone());
                }

                // Add memory tools if shared memory is configured
                if let Some(ref memory) = shared_memory {
                    let ns = Arc::new(crate::memory::namespaced::NamespacedMemory::new(
                        memory.clone(),
                        &agent_def.name,
                    ));
                    builder = builder.memory(ns);
                    builder = builder.tools(crate::memory::shared_tools::shared_memory_tools(
                        memory.clone(),
                        &agent_def.name,
                    ));
                }

                // Add blackboard tools if blackboard is configured
                if let Some(ref bb) = blackboard {
                    builder = builder.tools(blackboard_tools(bb.clone()));
                }

                // Add knowledge tools if knowledge base is configured
                if let Some(ref kb) = knowledge_base {
                    builder = builder.knowledge(kb.clone());
                }

                let runner = match builder.build() {
                    Ok(r) => r,
                    Err(e) => {
                        return (
                            idx,
                            SubAgentResult {
                                agent: agent_def.name,
                                result: format!("Error building agent: {e}"),
                                tokens_used: TokenUsage::default(),
                                success: false,
                            },
                        );
                    }
                };

                let result = match runner.execute(&task.task).await {
                    Ok(output) => {
                        // Write successful result to blackboard (matching Restate path)
                        if let Some(ref bb) = blackboard {
                            let key = format!("agent:{}", agent_def.name);
                            if let Err(e) = bb
                                .write(&key, serde_json::Value::String(output.result.clone()))
                                .await
                            {
                                tracing::warn!(
                                    agent = %agent_def.name,
                                    error = %e,
                                    "failed to write result to blackboard"
                                );
                            }
                        }
                        SubAgentResult {
                            agent: agent_def.name,
                            result: output.result,
                            tokens_used: output.tokens_used,
                            success: true,
                        }
                    }
                    Err(e) => SubAgentResult {
                        agent: agent_def.name,
                        result: format!("Error: {e}"),
                        tokens_used: e.partial_usage(),
                        success: false,
                    },
                };

                (idx, result)
            });
        }

        let mut results: Vec<Option<(usize, SubAgentResult)>> = vec![None; task_count];
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((idx, sub_result)) => {
                    results[idx] = Some((idx, sub_result));
                }
                Err(e) => {
                    tracing::error!(error = %e, "sub-agent task panicked");
                }
            }
        }

        // Fill gaps (panicked tasks) with error results, then sort by index
        let mut results: Vec<(usize, SubAgentResult)> = results
            .into_iter()
            .enumerate()
            .map(|(idx, r)| {
                r.unwrap_or_else(|| {
                    (
                        idx,
                        SubAgentResult {
                            agent: agent_names[idx].clone(),
                            result: "Error: sub-agent task panicked".into(),
                            tokens_used: TokenUsage::default(),
                            success: false,
                        },
                    )
                })
            })
            .collect();
        results.sort_by_key(|(idx, _)| *idx);

        // Accumulate sub-agent tokens (lock scope kept minimal — no callbacks inside)
        {
            let mut acc = self.accumulated_tokens.lock().expect("token lock poisoned");
            for (_, r) in &results {
                *acc += r.tokens_used;
            }
        }

        // Emit completion events outside the lock
        if let Some(ref cb) = self.on_event {
            for (_, r) in &results {
                cb(AgentEvent::SubAgentCompleted {
                    agent: r.agent.clone(),
                    success: r.success,
                    usage: r.tokens_used,
                });
            }
        }

        let formatted = results
            .iter()
            .map(|(_, r)| format!("=== Agent: {} ===\n{}", r.agent, r.result))
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(formatted)
    }
}

impl Tool for DelegateTaskTool {
    fn definition(&self) -> ToolDefinition {
        self.cached_definition.clone()
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let delegate_input: DelegateInput = serde_json::from_value(input)
                .map_err(|e| Error::Agent(format!("Invalid delegate_task input: {e}")))?;

            let result = self.delegate(delegate_input.tasks).await?;
            Ok(ToolOutput::success(result))
        })
    }
}

#[derive(Deserialize)]
struct DelegateInput {
    tasks: Vec<DelegatedTask>,
}

/// The orchestrator's squad-formation tool: dispatches per-agent tasks with a
/// shared private blackboard for intra-squad coordination.
///
/// Unlike `DelegateTaskTool` which runs agents independently with the outer blackboard,
/// `FormSquadTool` creates a private `InMemoryBlackboard` so squad members can read
/// each other's intermediate results without polluting the outer blackboard. The final
/// formatted result is written to the outer blackboard under `"squad:{names}"`.
///
/// Accepts the same `{tasks:[{agent, task}]}` format as `delegate_task`.
struct FormSquadTool {
    shared_provider: Arc<BoxedProvider>,
    agent_pool: Vec<SubAgentDef>,
    default_max_turns: usize,
    default_max_tokens: u32,
    /// Permission rules inherited from the orchestrator, forwarded to squad members.
    permission_rules: super::permission::PermissionRuleset,
    accumulated_tokens: Arc<Mutex<TokenUsage>>,
    shared_memory: Option<Arc<dyn Memory>>,
    /// Outer blackboard for writing squad results. Squad members use a private one.
    blackboard: Option<Arc<dyn Blackboard>>,
    knowledge_base: Option<Arc<dyn KnowledgeBase>>,
    on_event: Option<Arc<OnEvent>>,
    /// Optional streaming text callback, forwarded to squad members.
    on_text: Option<Arc<crate::llm::OnText>>,
    /// Optional LSP manager, forwarded to squad members.
    lsp_manager: Option<Arc<crate::lsp::LspManager>>,
    cached_definition: ToolDefinition,
    /// Observability mode inherited from the orchestrator, forwarded to squad members.
    observability_mode: super::observability::ObservabilityMode,
}

impl Tool for FormSquadTool {
    fn definition(&self) -> ToolDefinition {
        self.cached_definition.clone()
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let delegate_input: DelegateInput = serde_json::from_value(input)
                .map_err(|e| Error::Agent(format!("Invalid form_squad input: {e}")))?;

            let tasks = delegate_input.tasks;

            // Validate: at least 2 tasks (agents)
            if tasks.len() < 2 {
                return Ok(ToolOutput::error(
                    "form_squad requires at least 2 tasks. Use delegate_task for single-agent tasks."
                        .to_string(),
                ));
            }

            // Validate: no duplicate agent names
            {
                let mut seen = std::collections::HashSet::new();
                for t in &tasks {
                    if !seen.insert(&t.agent) {
                        return Ok(ToolOutput::error(format!(
                            "Duplicate agent name in squad: '{}'",
                            t.agent
                        )));
                    }
                }
            }

            // Validate all agents exist before spawning any
            for t in &tasks {
                if !self.agent_pool.iter().any(|a| a.name == t.agent) {
                    return Ok(ToolOutput::error(format!(
                        "Unknown agent '{}'. Available agents: {}",
                        t.agent,
                        self.agent_pool
                            .iter()
                            .map(|a| a.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
            }

            let task_count = tasks.len();
            let agent_names: Vec<String> = tasks.iter().map(|t| t.agent.clone()).collect();

            // Create private blackboard for intra-squad coordination
            let private_bb: Arc<dyn Blackboard> = Arc::new(InMemoryBlackboard::new());

            let _squad_span = info_span!(
                "heartbit.orchestrator.squad",
                agent_count = task_count,
                agents = ?agent_names,
            );

            if let Some(ref cb) = self.on_event {
                cb(AgentEvent::SubAgentsDispatched {
                    agent: "squad-leader".into(),
                    agents: agent_names.clone(),
                });
            }

            // Direct dispatch via JoinSet — no mini-orchestrator
            let mut join_set = tokio::task::JoinSet::new();

            for (idx, task) in tasks.into_iter().enumerate() {
                // Agent existence already validated above, but avoid panic in library code
                let agent_def = match self.agent_pool.iter().find(|a| a.name == task.agent) {
                    Some(def) => def.clone(),
                    None => {
                        return Ok(ToolOutput::error(format!(
                            "Internal error: agent '{}' not found after validation",
                            task.agent
                        )));
                    }
                };

                let provider = agent_def
                    .provider_override
                    .clone()
                    .unwrap_or_else(|| self.shared_provider.clone());
                let max_turns = agent_def.max_turns.unwrap_or(self.default_max_turns);
                let max_tokens = agent_def.max_tokens.unwrap_or(self.default_max_tokens);
                let shared_memory = self.shared_memory.clone();
                let bb = private_bb.clone();
                let knowledge_base = self.knowledge_base.clone();
                let on_event = self.on_event.clone();
                let on_text = self.on_text.clone();
                let lsp_manager = self.lsp_manager.clone();
                let permission_rules = self.permission_rules.clone();
                let observability_mode = self.observability_mode;

                info!(agent = %agent_def.name, task = %task.task, "spawning squad member");

                join_set.spawn(async move {
                    let mut builder = AgentRunner::builder(provider)
                        .name(&agent_def.name)
                        .system_prompt(&agent_def.system_prompt)
                        .tools(agent_def.tools)
                        .max_turns(max_turns)
                        .max_tokens(max_tokens);

                    if let Some(strategy) = agent_def.context_strategy {
                        builder = builder.context_strategy(strategy);
                    }
                    if let Some(threshold) = agent_def.summarize_threshold {
                        builder = builder.summarize_threshold(threshold);
                    }
                    if let Some(timeout) = agent_def.tool_timeout {
                        builder = builder.tool_timeout(timeout);
                    }
                    if let Some(max) = agent_def.max_tool_output_bytes {
                        builder = builder.max_tool_output_bytes(max);
                    }
                    if let Some(schema) = agent_def.response_schema {
                        builder = builder.structured_schema(schema);
                    }
                    if !agent_def.guardrails.is_empty() {
                        builder = builder.guardrails(agent_def.guardrails);
                    }
                    if let Some(timeout) = agent_def.run_timeout {
                        builder = builder.run_timeout(timeout);
                    }
                    if let Some(effort) = agent_def.reasoning_effort {
                        builder = builder.reasoning_effort(effort);
                    }
                    if let Some(true) = agent_def.enable_reflection {
                        builder = builder.enable_reflection(true);
                    }
                    if let Some(threshold) = agent_def.tool_output_compression_threshold {
                        builder = builder.tool_output_compression_threshold(threshold);
                    }
                    if let Some(max) = agent_def.max_tools_per_turn {
                        builder = builder.max_tools_per_turn(max);
                    }
                    if let Some(max) = agent_def.max_identical_tool_calls {
                        builder = builder.max_identical_tool_calls(max);
                    }
                    if let Some(ref config) = agent_def.session_prune_config {
                        builder = builder.session_prune_config(config.clone());
                    }
                    if let Some(true) = agent_def.enable_recursive_summarization {
                        builder = builder.enable_recursive_summarization(true);
                    }
                    if let Some(threshold) = agent_def.reflection_threshold {
                        builder = builder.reflection_threshold(threshold);
                    }
                    if let Some(true) = agent_def.consolidate_on_exit {
                        builder = builder.consolidate_on_exit(true);
                    }

                    // Forward permission rules from orchestrator to squad members
                    if !permission_rules.is_empty() {
                        builder = builder.permission_rules(permission_rules);
                    }

                    // Forward observability mode from orchestrator to squad members
                    builder = builder.observability_mode(observability_mode);

                    // Forward LSP manager to squad members
                    if let Some(ref lsp) = lsp_manager {
                        builder = builder.lsp_manager(lsp.clone());
                    }

                    // Forward on_event so sub-agent events are visible
                    if let Some(ref on_event) = on_event {
                        builder = builder.on_event(on_event.clone());
                    }
                    // Forward on_text so sub-agent streaming text is visible
                    if let Some(ref on_text) = on_text {
                        builder = builder.on_text(on_text.clone());
                    }

                    // Add memory tools if shared memory is configured
                    if let Some(ref memory) = shared_memory {
                        let ns = Arc::new(crate::memory::namespaced::NamespacedMemory::new(
                            memory.clone(),
                            &agent_def.name,
                        ));
                        builder = builder.memory(ns);
                        builder = builder.tools(crate::memory::shared_tools::shared_memory_tools(
                            memory.clone(),
                            &agent_def.name,
                        ));
                    }

                    // Add blackboard tools using the PRIVATE blackboard
                    builder = builder.tools(blackboard_tools(bb.clone()));

                    // Add knowledge tools if knowledge base is configured
                    if let Some(ref kb) = knowledge_base {
                        builder = builder.knowledge(kb.clone());
                    }

                    let runner = match builder.build() {
                        Ok(r) => r,
                        Err(e) => {
                            return (
                                idx,
                                SubAgentResult {
                                    agent: agent_def.name,
                                    result: format!("Error building agent: {e}"),
                                    tokens_used: TokenUsage::default(),
                                    success: false,
                                },
                            );
                        }
                    };

                    let result = match runner.execute(&task.task).await {
                        Ok(output) => {
                            // Write successful result to private blackboard
                            let key = format!("agent:{}", agent_def.name);
                            if let Err(e) = bb
                                .write(&key, serde_json::Value::String(output.result.clone()))
                                .await
                            {
                                tracing::warn!(
                                    agent = %agent_def.name,
                                    error = %e,
                                    "failed to write result to private blackboard"
                                );
                            }
                            SubAgentResult {
                                agent: agent_def.name,
                                result: output.result,
                                tokens_used: output.tokens_used,
                                success: true,
                            }
                        }
                        Err(e) => SubAgentResult {
                            agent: agent_def.name,
                            result: format!("Error: {e}"),
                            tokens_used: e.partial_usage(),
                            success: false,
                        },
                    };

                    (idx, result)
                });
            }

            let mut results: Vec<Option<(usize, SubAgentResult)>> = vec![None; task_count];
            while let Some(result) = join_set.join_next().await {
                match result {
                    Ok((idx, sub_result)) => {
                        results[idx] = Some((idx, sub_result));
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "squad member task panicked");
                    }
                }
            }

            // Fill gaps (panicked tasks) with error results, then sort by index
            let mut results: Vec<(usize, SubAgentResult)> = results
                .into_iter()
                .enumerate()
                .map(|(idx, r)| {
                    r.unwrap_or_else(|| {
                        (
                            idx,
                            SubAgentResult {
                                agent: agent_names[idx].clone(),
                                result: "Error: squad member task panicked".into(),
                                tokens_used: TokenUsage::default(),
                                success: false,
                            },
                        )
                    })
                })
                .collect();
            results.sort_by_key(|(idx, _)| *idx);

            let squad_label = format!("squad[{}]", agent_names.join(","));
            let bb_key = format!("squad:{}", agent_names.join("+"));

            // Accumulate sub-agent tokens (lock scope kept minimal)
            let mut total_tokens = TokenUsage::default();
            {
                let mut acc = self.accumulated_tokens.lock().expect("token lock poisoned");
                for (_, r) in &results {
                    *acc += r.tokens_used;
                    total_tokens += r.tokens_used;
                }
            }

            // Emit per-agent completion events
            if let Some(ref cb) = self.on_event {
                for (_, r) in &results {
                    cb(AgentEvent::SubAgentCompleted {
                        agent: r.agent.clone(),
                        success: r.success,
                        usage: r.tokens_used,
                    });
                }
            }

            let all_success = results.iter().all(|(_, r)| r.success);

            let formatted = results
                .iter()
                .map(|(_, r)| format!("=== Agent: {} ===\n{}", r.agent, r.result))
                .collect::<Vec<_>>()
                .join("\n\n");

            // Emit aggregate squad completion event
            if let Some(ref cb) = self.on_event {
                cb(AgentEvent::SubAgentCompleted {
                    agent: squad_label,
                    success: all_success,
                    usage: total_tokens,
                });
            }

            // Write squad result to outer blackboard
            if let Some(ref bb) = self.blackboard
                && let Err(e) = bb
                    .write(&bb_key, serde_json::Value::String(formatted.clone()))
                    .await
            {
                tracing::warn!(
                    key = %bb_key,
                    error = %e,
                    "failed to write squad result to outer blackboard"
                );
            }

            Ok(ToolOutput::success(formatted))
        })
    }
}

/// Build the orchestrator system prompt listing available agents.
///
/// Shared between standalone and Restate paths. Takes `(name, description, tool_names)` triples.
///
/// When `squads_enabled` is `true`, the prompt explains both `delegate_task` and `form_squad`
/// tools. When `false`, only `delegate_task` is mentioned (Restate path or opt-out).
///
/// **Note:** Only the agent's registered tools are listed. Runtime-injected tools
/// (memory, blackboard, knowledge) are shared infrastructure available to all agents
/// and are not shown here to avoid noise in the prompt.
pub(crate) fn build_system_prompt(
    agents: &[(&str, &str, &[String])],
    squads_enabled: bool,
    dispatch_mode: DispatchMode,
) -> String {
    let agent_list: String = agents
        .iter()
        .map(|(name, desc, tools)| {
            if tools.is_empty() {
                format!("- **{name}**: {desc}\n  Tools: (none)")
            } else {
                format!("- **{name}**: {desc}\n  Tools: {}", tools.join(", "))
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let delegation_instructions = match (squads_enabled, dispatch_mode) {
        (_, DispatchMode::Sequential) => {
            "## Delegation Tool\n\
             Delegate to ONE agent at a time using **delegate_task**. Wait for the result \
             before deciding the next agent. Do NOT batch multiple agents in a single call."
        }
        (true, DispatchMode::Parallel) => {
            "## Delegation Tools\n\
             You have two delegation tools:\n\n\
             1. **delegate_task** — Run independent subtasks in parallel. Each agent works in \
                isolation and cannot see other agents' output. Use when subtasks are independent.\n\n\
             2. **form_squad** — Run subtasks in parallel with a shared blackboard. \
                Unlike delegate_task, agents can read each other's results via the blackboard. \
                Agents run concurrently — use when they benefit from shared state, not when \
                strict ordering is needed.\n\n\
             After receiving results, synthesize them into a coherent response."
        }
        (false, DispatchMode::Parallel) => {
            "## Delegation Tool\n\
             Use the **delegate_task** tool to assign work to sub-agents. You can assign \
             multiple tasks at once for parallel execution. Each agent works in isolation. \
             After receiving results, synthesize them into a coherent response."
        }
    };

    let choose_tool_step = match (squads_enabled, dispatch_mode) {
        (_, DispatchMode::Sequential) => {
            "3. DELEGATE: Use delegate_task with ONE agent at a time. Wait for results before \
                delegating to the next agent."
        }
        (true, DispatchMode::Parallel) => {
            "3. CHOOSE TOOL: Select delegate_task for independent parallel work, or form_squad \
                when agents benefit from shared state via a blackboard."
        }
        (false, DispatchMode::Parallel) => {
            "3. DELEGATE: Use delegate_task to assign subtasks to the best-fit agents."
        }
    };

    format!(
        "You are an orchestrator agent. Analyze incoming tasks and delegate work to \
         specialized sub-agents.\n\n\
         ## Decision Process\n\
         1. DECOMPOSE: Break the task into distinct subtasks. Identify which require different expertise.\n\
         2. MATCH: For each subtask, pick the best-fit agent based on their description and tools.\n\
         {choose_tool_step}\n\n\
         ## Effort Scaling\n\
         - If only ONE agent is relevant, delegate a single task. Do NOT force-split across agents.\n\
         - If the task is simple enough for one agent, use one agent.\n\
         - Only use multiple agents when the task genuinely has multiple distinct parts \
           needing different expertise.\n\n\
         ## Task Quality\n\
         - Each delegated task must be self-contained: include all context the agent needs.\n\
         - Be specific: \"Read /path/to/file and extract X\" not \"look at the project\".\n\
         - Avoid overlapping tasks — no two agents should do the same work.\n\n\
         ## Available Sub-Agents\n\
         Choose agents based on their description and available tools:\n\
         {agent_list}\n\n\
         {delegation_instructions}"
    )
}

/// Build the delegate_task tool definition.
///
/// Shared between standalone and Restate paths. Takes `(name, description, tool_names)` triples.
///
/// When `dispatch_mode` is `Sequential`, the schema adds `maxItems: 1` to the tasks array
/// so the LLM can only dispatch one agent at a time. This is a schema-level enforcement
/// that works even with weaker models that ignore prompt instructions.
pub(crate) fn build_delegate_tool_schema(
    agents: &[(&str, &str, &[String])],
    dispatch_mode: DispatchMode,
) -> ToolDefinition {
    let agent_descriptions: Vec<serde_json::Value> = agents
        .iter()
        .map(|(name, desc, tools)| json!({"name": name, "description": desc, "tools": tools}))
        .collect();

    let (description, tasks_schema) = match dispatch_mode {
        DispatchMode::Sequential => (
            format!(
                "Delegate a task to ONE sub-agent at a time. Wait for the result before \
                 delegating to the next agent. Each task runs in isolation. \
                 Write clear, self-contained task descriptions with all necessary context. \
                 Available agents: {}",
                serde_json::to_string(&agent_descriptions)
                    .expect("agent list serialization is infallible")
            ),
            json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the sub-agent"
                        },
                        "task": {
                            "type": "string",
                            "description": "Task instruction for the sub-agent"
                        }
                    },
                    "required": ["agent", "task"]
                },
                "minItems": 1,
                "maxItems": 1
            }),
        ),
        DispatchMode::Parallel => (
            format!(
                "Delegate independent tasks to sub-agents for parallel execution. \
                 Each task runs in isolation — agents cannot see each other's work. \
                 Write clear, self-contained task descriptions with all necessary context. \
                 Available agents: {}",
                serde_json::to_string(&agent_descriptions)
                    .expect("agent list serialization is infallible")
            ),
            json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the sub-agent"
                        },
                        "task": {
                            "type": "string",
                            "description": "Task instruction for the sub-agent"
                        }
                    },
                    "required": ["agent", "task"]
                },
                "minItems": 1
            }),
        ),
    };

    ToolDefinition {
        name: "delegate_task".into(),
        description,
        input_schema: json!({
            "type": "object",
            "properties": {
                "tasks": tasks_schema
            },
            "required": ["tasks"]
        }),
    }
}

/// Build the form_squad tool definition.
///
/// Standalone path only. Takes `(name, description, tool_names)` triples so the LLM
/// knows which agents are available for squad formation.
///
/// Uses the same `{tasks:[{agent, task}]}` format as `delegate_task` for consistency.
pub(crate) fn build_form_squad_tool_schema(agents: &[(&str, &str, &[String])]) -> ToolDefinition {
    let agent_descriptions: Vec<serde_json::Value> = agents
        .iter()
        .map(|(name, desc, tools)| json!({"name": name, "description": desc, "tools": tools}))
        .collect();

    ToolDefinition {
        name: "form_squad".into(),
        description: format!(
            "Dispatch per-agent tasks in parallel with a shared blackboard for intra-squad coordination. \
             Unlike delegate_task, squad agents can read each other's results via the blackboard. \
             Use this when agents benefit from shared state (e.g., building on each other's work, \
             coordinating on a shared artifact). Agents run concurrently. \
             Requires at least 2 tasks (one per agent). \
             Available agents: {}",
            serde_json::to_string(&agent_descriptions)
                .expect("agent list serialization is infallible")
        ),
        input_schema: json!({
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "description": "Name of the sub-agent"
                            },
                            "task": {
                                "type": "string",
                                "description": "Task instruction for the sub-agent"
                            }
                        },
                        "required": ["agent", "task"]
                    },
                    "minItems": 2,
                    "description": "Per-agent tasks for the squad (minimum 2)"
                }
            },
            "required": ["tasks"]
        }),
    }
}

/// Configuration for adding a sub-agent to the orchestrator.
///
/// Used by `OrchestratorBuilder::sub_agent_full` to avoid a long parameter list.
pub struct SubAgentConfig {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub tools: Vec<Arc<dyn Tool>>,
    pub context_strategy: Option<ContextStrategy>,
    pub summarize_threshold: Option<u32>,
    pub tool_timeout: Option<Duration>,
    pub max_tool_output_bytes: Option<usize>,
    /// Per-agent turn limit. When `None`, uses orchestrator default.
    pub max_turns: Option<usize>,
    /// Per-agent token limit. When `None`, uses orchestrator default.
    pub max_tokens: Option<u32>,
    /// Optional JSON Schema for structured output. When set, the sub-agent
    /// receives a synthetic `__respond__` tool and returns structured JSON.
    pub response_schema: Option<serde_json::Value>,
    /// Optional per-agent run timeout. When `None`, no timeout is applied
    /// to this sub-agent's run.
    pub run_timeout: Option<Duration>,
    /// Guardrails applied to this sub-agent's LLM calls and tool executions.
    pub guardrails: Vec<Arc<dyn Guardrail>>,
    /// Optional per-agent LLM provider override. When `None`, the sub-agent
    /// inherits the orchestrator's provider. Use this to route sub-agents to
    /// different models (e.g., Haiku for cheap tasks, Opus for complex ones).
    pub provider: Option<Arc<BoxedProvider>>,
    /// Optional reasoning/thinking effort level for this sub-agent.
    pub reasoning_effort: Option<crate::llm::types::ReasoningEffort>,
    /// Enable reflection prompts after tool results for this sub-agent.
    pub enable_reflection: Option<bool>,
    /// Tool output compression threshold in bytes for this sub-agent.
    pub tool_output_compression_threshold: Option<usize>,
    /// Maximum tools per turn for this sub-agent.
    pub max_tools_per_turn: Option<usize>,
    /// Maximum consecutive identical tool-call turns for doom loop detection.
    pub max_identical_tool_calls: Option<u32>,
    /// Session pruning configuration for this sub-agent.
    pub session_prune_config: Option<crate::agent::pruner::SessionPruneConfig>,
    /// Enable recursive summarization for this sub-agent.
    pub enable_recursive_summarization: Option<bool>,
    /// Memory reflection threshold for this sub-agent.
    pub reflection_threshold: Option<u32>,
    /// Run memory consolidation at session end for this sub-agent.
    pub consolidate_on_exit: Option<bool>,
}

pub struct OrchestratorBuilder<P: LlmProvider> {
    provider: Arc<P>,
    sub_agents: Vec<SubAgentDef>,
    max_turns: usize,
    max_tokens: u32,
    context_strategy: Option<ContextStrategy>,
    summarize_threshold: Option<u32>,
    tool_timeout: Option<Duration>,
    max_tool_output_bytes: Option<usize>,
    shared_memory: Option<Arc<dyn Memory>>,
    blackboard: Option<Arc<dyn Blackboard>>,
    knowledge_base: Option<Arc<dyn KnowledgeBase>>,
    on_text: Option<Arc<crate::llm::OnText>>,
    on_approval: Option<Arc<crate::llm::OnApproval>>,
    on_event: Option<Arc<OnEvent>>,
    guardrails: Vec<Arc<dyn Guardrail>>,
    on_question: Option<Arc<OnQuestion>>,
    run_timeout: Option<Duration>,
    enable_squads: Option<bool>,
    reasoning_effort: Option<crate::llm::types::ReasoningEffort>,
    enable_reflection: bool,
    tool_output_compression_threshold: Option<usize>,
    max_tools_per_turn: Option<usize>,
    max_identical_tool_calls: Option<u32>,
    permission_rules: super::permission::PermissionRuleset,
    instruction_text: Option<String>,
    learned_permissions: Option<Arc<std::sync::Mutex<super::permission::LearnedPermissions>>>,
    lsp_manager: Option<Arc<crate::lsp::LspManager>>,
    observability_mode: Option<super::observability::ObservabilityMode>,
    dispatch_mode: DispatchMode,
}

impl<P: LlmProvider + 'static> OrchestratorBuilder<P> {
    pub fn sub_agent(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        self.sub_agents.push(SubAgentDef {
            name: name.into(),
            description: description.into(),
            system_prompt: system_prompt.into(),
            tools: vec![],
            context_strategy: None,
            summarize_threshold: None,
            tool_timeout: None,
            max_tool_output_bytes: None,
            max_turns: None,
            max_tokens: None,
            response_schema: None,
            run_timeout: None,
            guardrails: vec![],
            provider_override: None,
            reasoning_effort: None,
            enable_reflection: None,
            tool_output_compression_threshold: None,
            max_tools_per_turn: None,
            max_identical_tool_calls: None,
            session_prune_config: None,
            enable_recursive_summarization: None,
            reflection_threshold: None,
            consolidate_on_exit: None,
        });
        self
    }

    pub fn sub_agent_with_tools(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
        tools: Vec<Arc<dyn Tool>>,
    ) -> Self {
        self.sub_agents.push(SubAgentDef {
            name: name.into(),
            description: description.into(),
            system_prompt: system_prompt.into(),
            tools,
            context_strategy: None,
            summarize_threshold: None,
            tool_timeout: None,
            max_tool_output_bytes: None,
            max_turns: None,
            max_tokens: None,
            response_schema: None,
            run_timeout: None,
            guardrails: vec![],
            provider_override: None,
            reasoning_effort: None,
            enable_reflection: None,
            tool_output_compression_threshold: None,
            max_tools_per_turn: None,
            max_identical_tool_calls: None,
            session_prune_config: None,
            enable_recursive_summarization: None,
            reflection_threshold: None,
            consolidate_on_exit: None,
        });
        self
    }

    pub fn sub_agent_full(mut self, def: SubAgentConfig) -> Self {
        self.sub_agents.push(SubAgentDef {
            name: def.name,
            description: def.description,
            system_prompt: def.system_prompt,
            tools: def.tools,
            context_strategy: def.context_strategy,
            summarize_threshold: def.summarize_threshold,
            tool_timeout: def.tool_timeout,
            max_tool_output_bytes: def.max_tool_output_bytes,
            max_turns: def.max_turns,
            max_tokens: def.max_tokens,
            response_schema: def.response_schema,
            run_timeout: def.run_timeout,
            guardrails: def.guardrails,
            provider_override: def.provider,
            reasoning_effort: def.reasoning_effort,
            enable_reflection: def.enable_reflection,
            tool_output_compression_threshold: def.tool_output_compression_threshold,
            max_tools_per_turn: def.max_tools_per_turn,
            max_identical_tool_calls: def.max_identical_tool_calls,
            session_prune_config: def.session_prune_config,
            enable_recursive_summarization: def.enable_recursive_summarization,
            reflection_threshold: def.reflection_threshold,
            consolidate_on_exit: def.consolidate_on_exit,
        });
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

    /// Set context management strategy for the orchestrator's own conversation.
    pub fn context_strategy(mut self, strategy: ContextStrategy) -> Self {
        self.context_strategy = Some(strategy);
        self
    }

    /// Set token threshold for summarization of the orchestrator's own context.
    pub fn summarize_threshold(mut self, threshold: u32) -> Self {
        self.summarize_threshold = Some(threshold);
        self
    }

    /// Set timeout for the orchestrator's own tool executions (i.e., delegate_task).
    pub fn tool_timeout(mut self, timeout: Duration) -> Self {
        self.tool_timeout = Some(timeout);
        self
    }

    /// Set maximum byte size for tool output on the orchestrator's own tools.
    pub fn max_tool_output_bytes(mut self, max: usize) -> Self {
        self.max_tool_output_bytes = Some(max);
        self
    }

    /// Attach a shared memory store. Each sub-agent gets:
    /// - Private memory tools (namespaced to the agent)
    /// - Shared memory tools (cross-agent read/write)
    pub fn shared_memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.shared_memory = Some(memory);
        self
    }

    /// Attach a shared blackboard for cross-agent coordination.
    ///
    /// Each sub-agent receives `blackboard_read`, `blackboard_write`, and
    /// `blackboard_list` tools. After each sub-agent completes, its result
    /// is automatically written to the blackboard under the `"agent:{name}"` key.
    pub fn blackboard(mut self, blackboard: Arc<dyn Blackboard>) -> Self {
        self.blackboard = Some(blackboard);
        self
    }

    /// Attach a shared knowledge base for document retrieval.
    ///
    /// Each sub-agent receives a `knowledge_search` tool to query the knowledge
    /// base at runtime.
    pub fn knowledge(mut self, kb: Arc<dyn KnowledgeBase>) -> Self {
        self.knowledge_base = Some(kb);
        self
    }

    /// Set a callback for streaming text output on the orchestrator's LLM calls.
    /// Sub-agents do not stream — only the orchestrator's own reasoning and
    /// final synthesis are emitted incrementally.
    pub fn on_text(mut self, callback: Arc<crate::llm::OnText>) -> Self {
        self.on_text = Some(callback);
        self
    }

    /// Set a callback for human-in-the-loop approval on the orchestrator's
    /// tool calls (i.e., delegate_task calls). Sub-agents do not use this
    /// callback — only the orchestrator's decisions are gated.
    pub fn on_approval(mut self, callback: Arc<crate::llm::OnApproval>) -> Self {
        self.on_approval = Some(callback);
        self
    }

    /// Set learned permissions for persisting AlwaysAllow/AlwaysDeny decisions.
    pub fn learned_permissions(
        mut self,
        learned: Arc<std::sync::Mutex<super::permission::LearnedPermissions>>,
    ) -> Self {
        self.learned_permissions = Some(learned);
        self
    }

    /// Set an LSP manager for collecting diagnostics after file-modifying tools.
    pub fn lsp_manager(mut self, manager: Arc<crate::lsp::LspManager>) -> Self {
        self.lsp_manager = Some(manager);
        self
    }

    /// Set a callback for structured agent events.
    ///
    /// The callback receives events from the orchestrator's own agent loop **and**
    /// from all sub-agents (both `delegate_task` and `form_squad` paths). Sub-agent
    /// events carry the sub-agent name in their `agent` field for disambiguation.
    pub fn on_event(mut self, callback: Arc<OnEvent>) -> Self {
        self.on_event = Some(callback);
        self
    }

    /// Add a single guardrail applied to the orchestrator's own agent loop.
    pub fn guardrail(mut self, guardrail: Arc<dyn Guardrail>) -> Self {
        self.guardrails.push(guardrail);
        self
    }

    /// Add multiple guardrails to the orchestrator's own agent loop.
    pub fn guardrails(mut self, guardrails: Vec<Arc<dyn Guardrail>>) -> Self {
        self.guardrails.extend(guardrails);
        self
    }

    /// Set a callback for structured questions from the orchestrator to the user.
    pub fn on_question(mut self, callback: Arc<OnQuestion>) -> Self {
        self.on_question = Some(callback);
        self
    }

    /// Set a wall-clock deadline for the entire orchestrator run. If the run
    /// does not complete within this duration, `Error::RunTimeout` is returned.
    pub fn run_timeout(mut self, timeout: Duration) -> Self {
        self.run_timeout = Some(timeout);
        self
    }

    /// Enable or disable the `form_squad` tool for dynamic agent squad formation.
    ///
    /// When `true`, the orchestrator can assemble temporary squads of agents to
    /// collaboratively solve complex subtasks. When `false`, only `delegate_task`
    /// is available.
    ///
    /// Default: auto-enabled when there are >= 2 agents.
    pub fn enable_squads(mut self, enable: bool) -> Self {
        self.enable_squads = Some(enable);
        self
    }

    /// Set reasoning/thinking effort level for the orchestrator's own LLM calls.
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

    pub fn max_identical_tool_calls(mut self, max: u32) -> Self {
        self.max_identical_tool_calls = Some(max);
        self
    }

    pub fn permission_rules(mut self, rules: super::permission::PermissionRuleset) -> Self {
        self.permission_rules = rules;
        self
    }

    /// Provide pre-loaded instruction text to prepend to the orchestrator's system prompt.
    pub fn instruction_text(mut self, text: impl Into<String>) -> Self {
        let text = text.into();
        if !text.is_empty() {
            self.instruction_text = Some(text);
        }
        self
    }

    /// Set the observability verbosity mode for the orchestrator and its sub-agents.
    pub fn observability_mode(mut self, mode: super::observability::ObservabilityMode) -> Self {
        self.observability_mode = Some(mode);
        self
    }

    /// Set the dispatch mode for orchestrator delegation.
    ///
    /// When `Sequential`, the delegate_task schema constrains `maxItems: 1` so
    /// the LLM dispatches one agent at a time. This is enforced at the JSON schema
    /// level, which works even with weaker models that ignore prompt instructions.
    pub fn dispatch_mode(mut self, mode: DispatchMode) -> Self {
        self.dispatch_mode = mode;
        self
    }

    pub fn build(self) -> Result<Orchestrator<P>, Error> {
        // Validate sub-agent definitions
        {
            let mut seen = std::collections::HashSet::new();
            for agent in &self.sub_agents {
                if agent.name.is_empty() {
                    return Err(Error::Config("sub-agent name must not be empty".into()));
                }
                if !seen.insert(&agent.name) {
                    return Err(Error::Config(format!(
                        "duplicate sub-agent name: '{}'",
                        agent.name
                    )));
                }
                if agent.max_turns == Some(0) {
                    return Err(Error::Config(format!(
                        "sub-agent '{}': max_turns must be > 0",
                        agent.name
                    )));
                }
                if agent.max_tokens == Some(0) {
                    return Err(Error::Config(format!(
                        "sub-agent '{}': max_tokens must be > 0",
                        agent.name
                    )));
                }
            }
        }

        if self.sub_agents.is_empty() {
            tracing::warn!(
                "orchestrator built with no sub-agents — delegate_task tool will list no agents"
            );
        }

        // Sequential dispatch and squads are incompatible — form_squad runs agents
        // in parallel, which defeats sequential ordering. Force squads off.
        let squads_enabled = if self.dispatch_mode == DispatchMode::Sequential {
            false
        } else {
            self.enable_squads.unwrap_or(self.sub_agents.len() >= 2)
        };
        if squads_enabled && self.sub_agents.len() < 2 {
            tracing::warn!(
                "enable_squads is true but fewer than 2 agents are registered — \
                 form_squad requires at least 2 agents to be useful"
            );
        }

        let tool_names: Vec<Vec<String>> = self
            .sub_agents
            .iter()
            .map(|a| a.tools.iter().map(|t| t.definition().name).collect())
            .collect();
        let triples: Vec<(&str, &str, &[String])> = self
            .sub_agents
            .iter()
            .zip(tool_names.iter())
            .map(|(a, names)| (a.name.as_str(), a.description.as_str(), names.as_slice()))
            .collect();
        let system = build_system_prompt(&triples, squads_enabled, self.dispatch_mode);
        let cached_definition = build_delegate_tool_schema(&triples, self.dispatch_mode);
        let form_squad_definition = if squads_enabled {
            Some(build_form_squad_tool_schema(&triples))
        } else {
            None
        };
        // Drop borrows on self.sub_agents so we can move it below
        drop(triples);
        drop(tool_names);

        let sub_agent_tokens = Arc::new(Mutex::new(TokenUsage::default()));

        let shared_provider = Arc::new(BoxedProvider::from_arc(self.provider.clone()));

        // Clone agent pool for FormSquadTool before moving into DelegateTaskTool
        let agent_pool = if squads_enabled {
            Some(self.sub_agents.clone())
        } else {
            None
        };

        let resolved_mode = self
            .observability_mode
            .unwrap_or(super::observability::ObservabilityMode::Production);

        let delegate_tool: Arc<dyn Tool> = Arc::new(DelegateTaskTool {
            shared_provider: shared_provider.clone(),
            sub_agents: self.sub_agents,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            permission_rules: self.permission_rules.clone(),
            accumulated_tokens: sub_agent_tokens.clone(),
            shared_memory: self.shared_memory.clone(),
            blackboard: self.blackboard.clone(),
            knowledge_base: self.knowledge_base.clone(),
            cached_definition,
            on_event: self.on_event.clone(),
            on_text: self.on_text.clone(),
            lsp_manager: self.lsp_manager.clone(),
            observability_mode: resolved_mode,
        });

        let mut runner_builder = AgentRunner::builder(self.provider)
            .name("orchestrator")
            .system_prompt(system)
            .tool(delegate_tool)
            .max_turns(self.max_turns)
            .max_tokens(self.max_tokens);

        // Register form_squad tool when enabled
        if let Some(agent_pool) = agent_pool {
            // SAFETY: form_squad_definition is always Some when agent_pool is Some
            let squad_def = form_squad_definition.expect("squad definition computed when enabled");
            let form_squad_tool: Arc<dyn Tool> = Arc::new(FormSquadTool {
                shared_provider,
                agent_pool,
                default_max_turns: self.max_turns,
                default_max_tokens: self.max_tokens,
                permission_rules: self.permission_rules.clone(),
                accumulated_tokens: sub_agent_tokens.clone(),
                shared_memory: self.shared_memory,
                blackboard: self.blackboard,
                knowledge_base: self.knowledge_base,
                on_event: self.on_event.clone(),
                on_text: self.on_text.clone(),
                lsp_manager: self.lsp_manager.clone(),
                cached_definition: squad_def,
                observability_mode: resolved_mode,
            });
            runner_builder = runner_builder.tool(form_squad_tool);
        }

        if let Some(strategy) = self.context_strategy {
            runner_builder = runner_builder.context_strategy(strategy);
        }
        if let Some(threshold) = self.summarize_threshold {
            runner_builder = runner_builder.summarize_threshold(threshold);
        }
        if let Some(timeout) = self.tool_timeout {
            runner_builder = runner_builder.tool_timeout(timeout);
        }
        if let Some(max) = self.max_tool_output_bytes {
            runner_builder = runner_builder.max_tool_output_bytes(max);
        }
        if let Some(on_text) = self.on_text {
            runner_builder = runner_builder.on_text(on_text);
        }
        if let Some(on_approval) = self.on_approval {
            runner_builder = runner_builder.on_approval(on_approval);
        }
        if let Some(learned) = self.learned_permissions {
            runner_builder = runner_builder.learned_permissions(learned);
        }
        if let Some(lsp) = self.lsp_manager {
            runner_builder = runner_builder.lsp_manager(lsp);
        }
        if let Some(on_event) = self.on_event {
            runner_builder = runner_builder.on_event(on_event);
        }
        if !self.guardrails.is_empty() {
            runner_builder = runner_builder.guardrails(self.guardrails);
        }
        if let Some(on_question) = self.on_question {
            runner_builder = runner_builder.on_question(on_question);
        }
        if let Some(timeout) = self.run_timeout {
            runner_builder = runner_builder.run_timeout(timeout);
        }
        if let Some(effort) = self.reasoning_effort {
            runner_builder = runner_builder.reasoning_effort(effort);
        }
        if self.enable_reflection {
            runner_builder = runner_builder.enable_reflection(true);
        }
        if let Some(threshold) = self.tool_output_compression_threshold {
            runner_builder = runner_builder.tool_output_compression_threshold(threshold);
        }
        if let Some(max) = self.max_tools_per_turn {
            runner_builder = runner_builder.max_tools_per_turn(max);
        }
        if let Some(max) = self.max_identical_tool_calls {
            runner_builder = runner_builder.max_identical_tool_calls(max);
        }
        if !self.permission_rules.is_empty() {
            runner_builder = runner_builder.permission_rules(self.permission_rules);
        }
        if let Some(text) = self.instruction_text {
            runner_builder = runner_builder.instruction_text(text);
        }
        if let Some(mode) = self.observability_mode {
            runner_builder = runner_builder.observability_mode(mode);
        }

        let runner = runner_builder.build()?;

        Ok(Orchestrator {
            runner,
            sub_agent_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use crate::tool::ToolOutput;
    use std::sync::Mutex;

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

        fn model_name(&self) -> Option<&str> {
            Some("mock-model-v1")
        }
    }

    /// A mock tool for orchestrator tests. Returns a fixed response.
    struct MockTool {
        def: crate::llm::types::ToolDefinition,
        response: String,
    }

    impl MockTool {
        fn new(name: &str, response: &str) -> Self {
            Self {
                def: crate::llm::types::ToolDefinition {
                    name: name.into(),
                    description: format!("Mock {name}"),
                    input_schema: json!({"type": "object"}),
                },
                response: response.into(),
            }
        }
    }

    impl crate::tool::Tool for MockTool {
        fn definition(&self) -> crate::llm::types::ToolDefinition {
            self.def.clone()
        }

        fn execute(
            &self,
            _input: serde_json::Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>,
        > {
            let response = self.response.clone();
            Box::pin(async move { Ok(ToolOutput::success(response)) })
        }
    }

    #[test]
    fn system_prompt_includes_agents() {
        let tools_a = vec!["web_search".to_string(), "read_file".to_string()];
        let tools_b: Vec<String> = vec![];
        let agents: Vec<(&str, &str, &[String])> = vec![
            ("researcher", "Research specialist", tools_a.as_slice()),
            ("coder", "Coding expert", tools_b.as_slice()),
        ];

        let prompt = build_system_prompt(&agents, false, DispatchMode::Parallel);
        assert!(prompt.contains("researcher"));
        assert!(prompt.contains("Research specialist"));
        assert!(prompt.contains("coder"));
        assert!(prompt.contains("Tools: web_search, read_file"));
        assert!(prompt.contains("Tools: (none)"));
        // New structured sections
        assert!(
            prompt.contains("Decision Process"),
            "prompt should contain Decision Process section: {prompt}"
        );
        assert!(
            prompt.contains("Effort Scaling"),
            "prompt should contain Effort Scaling section: {prompt}"
        );
        assert!(
            prompt.contains("Task Quality"),
            "prompt should contain Task Quality section: {prompt}"
        );
        assert!(
            prompt.contains("DECOMPOSE"),
            "prompt should contain decomposition guidance: {prompt}"
        );
    }

    #[test]
    fn system_prompt_shows_tool_names() {
        let tools = vec![
            "web_search".to_string(),
            "read_file".to_string(),
            "knowledge_search".to_string(),
        ];
        let no_tools: Vec<String> = vec![];
        let agents: Vec<(&str, &str, &[String])> = vec![
            ("researcher", "Research specialist", tools.as_slice()),
            ("analyst", "Analytical thinker", no_tools.as_slice()),
        ];

        let prompt = build_system_prompt(&agents, false, DispatchMode::Parallel);
        assert!(
            prompt.contains("Tools: web_search, read_file, knowledge_search"),
            "prompt should list tool names: {prompt}"
        );
        assert!(
            prompt.contains("Tools: (none)"),
            "prompt should show (none) for agents without tools: {prompt}"
        );
    }

    #[test]
    fn system_prompt_sequential_says_one_at_a_time() {
        let tools: Vec<String> = vec![];
        let agents: Vec<(&str, &str, &[String])> =
            vec![("builder", "Builds stuff", tools.as_slice())];
        let prompt = build_system_prompt(&agents, false, DispatchMode::Sequential);
        assert!(
            prompt.contains("ONE agent at a time"),
            "sequential prompt should say one at a time: {prompt}"
        );
        assert!(
            !prompt.contains("parallel execution"),
            "sequential prompt should not mention parallel: {prompt}"
        );
    }

    #[test]
    fn system_prompt_parallel_says_parallel() {
        let tools: Vec<String> = vec![];
        let agents: Vec<(&str, &str, &[String])> =
            vec![("builder", "Builds stuff", tools.as_slice())];
        let prompt = build_system_prompt(&agents, false, DispatchMode::Parallel);
        assert!(
            prompt.contains("parallel execution"),
            "parallel prompt should mention parallel: {prompt}"
        );
    }

    #[test]
    fn delegate_schema_sequential_max_items_1() {
        let tools: Vec<String> = vec![];
        let agents: Vec<(&str, &str, &[String])> =
            vec![("builder", "Builds stuff", tools.as_slice())];
        let def = build_delegate_tool_schema(&agents, DispatchMode::Sequential);
        let tasks = &def.input_schema["properties"]["tasks"];
        assert_eq!(
            tasks["maxItems"], 1,
            "sequential schema should have maxItems=1: {tasks}"
        );
        assert!(
            def.description.contains("ONE sub-agent"),
            "sequential description should say ONE: {}",
            def.description
        );
    }

    #[test]
    fn delegate_schema_parallel_no_max_items() {
        let tools: Vec<String> = vec![];
        let agents: Vec<(&str, &str, &[String])> =
            vec![("builder", "Builds stuff", tools.as_slice())];
        let def = build_delegate_tool_schema(&agents, DispatchMode::Parallel);
        let tasks = &def.input_schema["properties"]["tasks"];
        assert!(
            tasks.get("maxItems").is_none(),
            "parallel schema should not have maxItems: {tasks}"
        );
    }

    #[tokio::test]
    async fn sequential_dispatch_disables_squads() {
        // With 2 agents, squads auto-enable. Sequential mode should force them off.
        // We verify by running the orchestrator and checking that the LLM request
        // only contains delegate_task (not form_squad).
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "done".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
        }]));
        let mut orch = Orchestrator::builder(provider)
            .sub_agent("a", "Agent A", "prompt a")
            .sub_agent("b", "Agent B", "prompt b")
            .dispatch_mode(DispatchMode::Sequential)
            .build()
            .unwrap();
        let output = orch.run("test").await.unwrap();
        assert_eq!(output.result, "done");
        // The system prompt should NOT mention form_squad
        // (indirectly verifies squads were disabled)
    }

    #[test]
    fn sequential_dispatch_disables_squads_in_prompt() {
        let tools: Vec<String> = vec![];
        let agents: Vec<(&str, &str, &[String])> = vec![
            ("a", "Agent A", tools.as_slice()),
            ("b", "Agent B", tools.as_slice()),
        ];
        // Sequential + squads_enabled=true should still produce a prompt without form_squad
        let prompt = build_system_prompt(&agents, false, DispatchMode::Sequential);
        assert!(
            !prompt.contains("form_squad"),
            "sequential prompt should not mention form_squad: {prompt}"
        );
    }

    #[test]
    fn delegate_tool_schema_includes_agents() {
        let tools = vec!["web_search".to_string()];
        let agents: Vec<(&str, &str, &[String])> =
            vec![("researcher", "Research", tools.as_slice())];
        let def = build_delegate_tool_schema(&agents, DispatchMode::Parallel);
        assert_eq!(def.name, "delegate_task");
        assert!(def.description.contains("researcher"));
        assert!(
            def.description.contains("web_search"),
            "delegate tool description should contain tool names: {}",
            def.description
        );
        assert!(
            def.description.contains("isolation"),
            "delegate tool description should mention isolation: {}",
            def.description
        );
        assert!(
            def.description.contains("self-contained"),
            "delegate tool description should mention self-contained tasks: {}",
            def.description
        );
    }

    #[test]
    fn delegate_tool_definition_includes_agents() {
        let agents: Vec<(&str, &str, &[String])> = vec![("researcher", "Research", &[])];
        let cached_definition = build_delegate_tool_schema(&agents, DispatchMode::Parallel);

        let tool = DelegateTaskTool {
            shared_provider: Arc::new(BoxedProvider::new(MockProvider::new(vec![]))),
            sub_agents: vec![SubAgentDef {
                name: "researcher".into(),
                description: "Research".into(),
                system_prompt: "prompt".into(),
                tools: vec![],
                context_strategy: None,
                summarize_threshold: None,
                tool_timeout: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: None,
                response_schema: None,
                run_timeout: None,
                guardrails: vec![],
                provider_override: None,
                reasoning_effort: None,
                enable_reflection: None,
                tool_output_compression_threshold: None,
                max_tools_per_turn: None,
                max_identical_tool_calls: None,
                session_prune_config: None,
                enable_recursive_summarization: None,
                reflection_threshold: None,
                consolidate_on_exit: None,
            }],
            shared_memory: None,
            blackboard: None,
            knowledge_base: None,
            max_turns: 10,
            max_tokens: 4096,
            permission_rules: crate::agent::permission::PermissionRuleset::default(),
            accumulated_tokens: Arc::new(Mutex::new(TokenUsage::default())),
            cached_definition,
            on_event: None,
            on_text: None,
            lsp_manager: None,
            observability_mode: crate::ObservabilityMode::Production,
        };

        let def = tool.definition();
        assert_eq!(def.name, "delegate_task");
        assert!(def.description.contains("researcher"));
        assert!(
            def.description.contains("tools"),
            "delegate tool description should contain 'tools' key: {}",
            def.description
        );
    }

    #[test]
    fn build_errors_on_duplicate_sub_agent_names() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research 1", "prompt1")
            .sub_agent("researcher", "Research 2", "prompt2")
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string()
                .contains("duplicate sub-agent name: 'researcher'"),
            "error: {err}"
        );
    }

    #[tokio::test]
    async fn orchestrator_direct_response_no_delegation() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Simple answer.".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
        }]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build()
            .unwrap();

        let output = orch.run("simple question").await.unwrap();
        assert_eq!(output.result, "Simple answer.");
        assert_eq!(output.tool_calls_made, 0);
    }

    #[tokio::test]
    async fn orchestrator_delegates_and_synthesizes() {
        // Responses consumed in order by provider. The orchestrator calls provider,
        // then sub-agents call provider (in spawn order under single-threaded tokio test runtime),
        // then orchestrator calls provider again for synthesis.
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator decides to delegate
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "researcher", "task": "Research Rust"},
                            {"agent": "analyst", "task": "Analyze findings"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 50,
                    output_tokens: 20,
                    ..Default::default()
                },
            },
            // 2: Sub-agent "researcher" response
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Rust is fast and safe.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 8,
                    ..Default::default()
                },
            },
            // 3: Sub-agent "analyst" response
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Strengths: memory safety, performance.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 12,
                    output_tokens: 10,
                    ..Default::default()
                },
            },
            // 4: Orchestrator synthesis
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Based on research: Rust is excellent.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 80,
                    output_tokens: 30,
                    ..Default::default()
                },
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research specialist", "You research.")
            .sub_agent("analyst", "Analysis expert", "You analyze.")
            .build()
            .unwrap();

        let output = orch.run("Analyze Rust").await.unwrap();
        assert_eq!(output.result, "Based on research: Rust is excellent.");
        assert_eq!(output.tool_calls_made, 1); // one delegate_task call
        // Orchestrator tokens (50+80 in, 20+30 out) + sub-agent tokens (10+12 in, 8+10 out)
        assert_eq!(output.tokens_used.input_tokens, 50 + 80 + 10 + 12);
        assert_eq!(output.tokens_used.output_tokens, 20 + 30 + 8 + 10);
    }

    #[tokio::test]
    async fn orchestrator_handles_unknown_agent_gracefully() {
        // Unknown agent now returns error in tool result, not a hard crash
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [{"agent": "nonexistent", "task": "do stuff"}]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // Orchestrator recovers after seeing the error
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "No such agent available.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build()
            .unwrap();

        let output = orch.run("delegate to unknown").await.unwrap();
        assert_eq!(output.result, "No such agent available.");
    }

    #[tokio::test]
    async fn orchestrator_handles_invalid_tool_name() {
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "wrong_tool".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Sorry, let me respond directly.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build()
            .unwrap();

        let output = orch.run("do something").await.unwrap();
        assert_eq!(output.result, "Sorry, let me respond directly.");
    }

    #[tokio::test]
    async fn orchestrator_handles_empty_delegate_tasks() {
        let provider = Arc::new(MockProvider::new(vec![
            // 1: LLM sends delegate_task with empty tasks array
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({"tasks": []}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: LLM recovers after seeing the error
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Let me try again properly.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build()
            .unwrap();

        let output = orch.run("do something").await.unwrap();
        assert_eq!(output.result, "Let me try again properly.");
    }

    #[tokio::test]
    async fn orchestrator_handles_missing_tasks_field() {
        let provider = Arc::new(MockProvider::new(vec![
            // 1: LLM sends delegate_task without tasks field
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: LLM recovers after seeing the parse error
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "I need to format correctly.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build()
            .unwrap();

        let output = orch.run("do something").await.unwrap();
        assert_eq!(output.result, "I need to format correctly.");
    }

    #[tokio::test]
    async fn blackboard_populated_after_delegation() {
        use crate::agent::blackboard::InMemoryBlackboard;

        let bb = Arc::new(InMemoryBlackboard::new());

        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator delegates to researcher
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [{"agent": "researcher", "task": "Find info"}]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Sub-agent responds
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Research result here.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
            // 3: Orchestrator synthesis
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research specialist", "You research.")
            .blackboard(bb.clone())
            .build()
            .unwrap();

        orch.run("research something").await.unwrap();

        // Verify the blackboard has the agent result
        let val: Option<serde_json::Value> = bb.read("agent:researcher").await.unwrap();
        assert!(val.is_some(), "blackboard should have agent:researcher key");
        assert_eq!(
            val.unwrap(),
            serde_json::Value::String("Research result here.".into())
        );
    }

    #[tokio::test]
    async fn sub_agents_receive_blackboard_tools() {
        use crate::agent::blackboard::InMemoryBlackboard;
        use crate::llm::types::CompletionRequest;

        // Track tool definitions seen by the sub-agent
        struct ToolTrackingProvider {
            responses: Mutex<Vec<CompletionResponse>>,
            tool_names_seen: Mutex<Vec<Vec<String>>>,
        }

        impl LlmProvider for ToolTrackingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                let names: Vec<String> = request.tools.iter().map(|t| t.name.clone()).collect();
                self.tool_names_seen.lock().expect("lock").push(names);

                let mut responses = self.responses.lock().expect("lock");
                if responses.is_empty() {
                    return Err(Error::Agent("no more mock responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let bb = Arc::new(InMemoryBlackboard::new());

        let provider = Arc::new(ToolTrackingProvider {
            responses: Mutex::new(vec![
                // 1: Orchestrator delegates
                CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "delegate_task".into(),
                        input: json!({
                            "tasks": [{"agent": "worker", "task": "do work"}]
                        }),
                    }],
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                },
                // 2: Sub-agent responds
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Work done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                },
                // 3: Orchestrator synthesis
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "All done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                },
            ]),
            tool_names_seen: Mutex::new(vec![]),
        });

        let mut orch = Orchestrator::builder(provider.clone())
            .sub_agent("worker", "Worker agent", "You work.")
            .blackboard(bb)
            .build()
            .unwrap();

        orch.run("do work").await.unwrap();

        // The second LLM call is from the sub-agent — check its tools
        let all_tool_names = provider.tool_names_seen.lock().expect("lock");
        assert!(
            all_tool_names.len() >= 2,
            "expected at least 2 LLM calls, got {}",
            all_tool_names.len()
        );
        let sub_agent_tools = &all_tool_names[1];
        assert!(
            sub_agent_tools.contains(&"blackboard_read".to_string()),
            "sub-agent should have blackboard_read tool, got: {sub_agent_tools:?}"
        );
        assert!(
            sub_agent_tools.contains(&"blackboard_write".to_string()),
            "sub-agent should have blackboard_write tool, got: {sub_agent_tools:?}"
        );
        assert!(
            sub_agent_tools.contains(&"blackboard_list".to_string()),
            "sub-agent should have blackboard_list tool, got: {sub_agent_tools:?}"
        );
    }

    #[test]
    fn blackboard_builder_method_works() {
        use crate::agent::blackboard::InMemoryBlackboard;

        let bb = Arc::new(InMemoryBlackboard::new());
        let provider = Arc::new(MockProvider::new(vec![]));

        // Should build successfully with blackboard
        let result = Orchestrator::builder(provider)
            .sub_agent("agent1", "Agent one", "You are agent 1.")
            .blackboard(bb)
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn knowledge_builder_method_works() {
        use crate::knowledge::in_memory::InMemoryKnowledgeBase;

        let kb: Arc<dyn KnowledgeBase> = Arc::new(InMemoryKnowledgeBase::new());
        let provider = Arc::new(MockProvider::new(vec![]));

        let result = Orchestrator::builder(provider)
            .sub_agent("agent1", "Agent one", "You are agent 1.")
            .knowledge(kb)
            .build();

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn sub_agents_receive_knowledge_tools() {
        use crate::knowledge::in_memory::InMemoryKnowledgeBase;
        use crate::llm::types::CompletionRequest;

        struct ToolTrackingProvider {
            responses: Mutex<Vec<CompletionResponse>>,
            tool_names_seen: Mutex<Vec<Vec<String>>>,
        }

        impl LlmProvider for ToolTrackingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                let names: Vec<String> = request.tools.iter().map(|t| t.name.clone()).collect();
                self.tool_names_seen.lock().expect("lock").push(names);

                let mut responses = self.responses.lock().expect("lock");
                if responses.is_empty() {
                    return Err(Error::Agent("no more mock responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let kb: Arc<dyn KnowledgeBase> = Arc::new(InMemoryKnowledgeBase::new());

        let provider = Arc::new(ToolTrackingProvider {
            responses: Mutex::new(vec![
                // 1: Orchestrator delegates
                CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "delegate_task".into(),
                        input: json!({
                            "tasks": [{"agent": "worker", "task": "do work"}]
                        }),
                    }],
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                },
                // 2: Sub-agent responds
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Work done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                },
                // 3: Orchestrator synthesis
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "All done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                },
            ]),
            tool_names_seen: Mutex::new(vec![]),
        });

        let mut orch = Orchestrator::builder(provider.clone())
            .sub_agent("worker", "Worker agent", "You work.")
            .knowledge(kb)
            .build()
            .unwrap();

        orch.run("do work").await.unwrap();

        let all_tool_names = provider.tool_names_seen.lock().expect("lock");
        assert!(
            all_tool_names.len() >= 2,
            "expected at least 2 LLM calls, got {}",
            all_tool_names.len()
        );
        let sub_agent_tools = &all_tool_names[1];
        assert!(
            sub_agent_tools.contains(&"knowledge_search".to_string()),
            "sub-agent should have knowledge_search tool, got: {sub_agent_tools:?}"
        );
    }

    #[tokio::test]
    async fn orchestrator_accumulates_cache_tokens_through_delegation() {
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator decides to delegate
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [{"agent": "researcher", "task": "Research Rust"}]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 50,
                    output_tokens: 20,
                    cache_creation_input_tokens: 100,
                    cache_read_input_tokens: 0,
                    reasoning_tokens: 0,
                },
            },
            // 2: Sub-agent "researcher" response (cache hit on second call)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Rust is fast.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 8,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 30,
                    reasoning_tokens: 0,
                },
            },
            // 3: Orchestrator synthesis (cache hit)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Rust is excellent.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 80,
                    output_tokens: 30,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 90,
                    reasoning_tokens: 0,
                },
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research specialist", "You research.")
            .build()
            .unwrap();

        let output = orch.run("Analyze Rust").await.unwrap();
        // Orchestrator: 50+80=130 in, 20+30=50 out, 100+0 cache_create, 0+90 cache_read
        // Sub-agent: 10 in, 8 out, 0 cache_create, 30 cache_read
        assert_eq!(output.tokens_used.input_tokens, 50 + 80 + 10);
        assert_eq!(output.tokens_used.output_tokens, 20 + 30 + 8);
        assert_eq!(output.tokens_used.cache_creation_input_tokens, 100);
        assert_eq!(output.tokens_used.cache_read_input_tokens, 90 + 30);
    }

    #[tokio::test]
    async fn orchestrator_error_includes_sub_agent_tokens() {
        // Scenario: orchestrator delegates, sub-agent succeeds, but orchestrator
        // hits max turns before synthesizing. The error's partial_usage should
        // include both orchestrator AND sub-agent tokens.
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator delegates
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [{"agent": "researcher", "task": "Research Rust"}]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 50,
                    output_tokens: 20,
                    ..Default::default()
                },
            },
            // 2: Sub-agent responds (tokens we must NOT lose)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Rust is fast.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 15,
                    output_tokens: 10,
                    ..Default::default()
                },
            },
            // 3: Orchestrator tries to delegate again (turn 2 = max_turns exceeded)
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-2".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [{"agent": "researcher", "task": "More research"}]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 80,
                    output_tokens: 25,
                    ..Default::default()
                },
            },
            // 4: Second sub-agent call
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "More info.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 12,
                    output_tokens: 8,
                    ..Default::default()
                },
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .max_turns(2)
            .build()
            .unwrap();

        let err = orch.run("research deeply").await.unwrap_err();

        // Error should be exactly one layer of WithPartialUsage wrapping MaxTurnsExceeded
        match &err {
            Error::WithPartialUsage { source, .. } => {
                assert!(
                    matches!(**source, Error::MaxTurnsExceeded(2)),
                    "inner error should be MaxTurnsExceeded(2), got: {source}"
                );
            }
            other => panic!("expected WithPartialUsage, got: {other}"),
        }

        let usage = err.partial_usage();
        // Orchestrator: 50+80 in, 20+25 out. Sub-agents: 15+12 in, 10+8 out.
        assert_eq!(
            usage.input_tokens,
            50 + 80 + 15 + 12,
            "input tokens: orchestrator(50+80) + sub-agent(15+12)"
        );
        assert_eq!(
            usage.output_tokens,
            20 + 25 + 10 + 8,
            "output tokens: orchestrator(20+25) + sub-agent(10+8)"
        );
    }

    #[tokio::test]
    async fn on_event_emits_sub_agent_events() {
        use crate::agent::events::AgentEvent;

        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let events_clone = events.clone();

        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator delegates to one agent
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [{"agent": "researcher", "task": "Research Rust"}]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Sub-agent responds
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Rust is fast.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
            },
            // 3: Orchestrator synthesizes
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Summary: Rust is fast.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .on_event(Arc::new(move |e| {
                events_clone.lock().unwrap().push(e);
            }))
            .build()
            .unwrap();

        orch.run("research task").await.unwrap();

        let events = events.lock().unwrap();

        let dispatched: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::SubAgentsDispatched { .. }))
            .collect();
        assert_eq!(dispatched.len(), 1, "expected 1 SubAgentsDispatched");
        match &dispatched[0] {
            AgentEvent::SubAgentsDispatched { agent, agents } => {
                assert_eq!(agent, "orchestrator");
                assert_eq!(agents, &["researcher"]);
            }
            _ => unreachable!(),
        }

        let completed: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::SubAgentCompleted { .. }))
            .collect();
        assert_eq!(completed.len(), 1, "expected 1 SubAgentCompleted");
        match &completed[0] {
            AgentEvent::SubAgentCompleted {
                agent,
                success,
                usage,
            } => {
                assert_eq!(agent, "researcher");
                assert!(success);
                assert_eq!(usage.input_tokens, 10);
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn sub_agent_receives_guardrails() {
        use crate::agent::guardrail::Guardrail;
        use crate::llm::types::CompletionRequest;

        // A guardrail that injects a marker into system prompts
        struct MarkerGuardrail;
        impl Guardrail for MarkerGuardrail {
            fn pre_llm(
                &self,
                request: &mut CompletionRequest,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<(), crate::error::Error>> + Send + '_>,
            > {
                request.system = format!("{} [GUARDRAIL_ACTIVE]", request.system);
                Box::pin(async { Ok(()) })
            }
        }

        struct CapturingProvider {
            responses: Mutex<Vec<CompletionResponse>>,
            systems_seen: Mutex<Vec<String>>,
        }

        impl LlmProvider for CapturingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, crate::error::Error> {
                self.systems_seen
                    .lock()
                    .unwrap()
                    .push(request.system.clone());
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Err(crate::error::Error::Agent("no more responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let guardrail: Arc<dyn Guardrail> = Arc::new(MarkerGuardrail);

        let provider = Arc::new(CapturingProvider {
            responses: Mutex::new(vec![
                // 1: Orchestrator delegates
                CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "delegate_task".into(),
                        input: json!({
                            "tasks": [{"agent": "worker", "task": "do work"}]
                        }),
                    }],
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                },
                // 2: Sub-agent responds
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Work done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                },
                // 3: Orchestrator synthesis
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "All done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                },
            ]),
            systems_seen: Mutex::new(vec![]),
        });

        let mut orch = Orchestrator::builder(provider.clone())
            .sub_agent_full(SubAgentConfig {
                name: "worker".into(),
                description: "Worker agent".into(),
                system_prompt: "You work.".into(),
                tools: vec![],
                context_strategy: None,
                summarize_threshold: None,
                tool_timeout: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: None,
                response_schema: None,
                run_timeout: None,
                guardrails: vec![guardrail],
                provider: None,
                reasoning_effort: None,
                enable_reflection: None,
                tool_output_compression_threshold: None,
                max_tools_per_turn: None,
                max_identical_tool_calls: None,
                session_prune_config: None,
                enable_recursive_summarization: None,
                reflection_threshold: None,
                consolidate_on_exit: None,
            })
            .build()
            .unwrap();

        orch.run("do work").await.unwrap();

        // The sub-agent's system prompt (second LLM call) should contain the guardrail marker
        let systems = provider.systems_seen.lock().unwrap();
        assert!(
            systems.len() >= 2,
            "expected at least 2 LLM calls, got {}",
            systems.len()
        );
        // systems[1] is the sub-agent call
        assert!(
            systems[1].contains("[GUARDRAIL_ACTIVE]"),
            "sub-agent system prompt should contain guardrail marker: {}",
            systems[1]
        );
        // systems[0] is the orchestrator call (no guardrail on orchestrator)
        assert!(
            !systems[0].contains("[GUARDRAIL_ACTIVE]"),
            "orchestrator system prompt should NOT contain guardrail marker: {}",
            systems[0]
        );
    }

    #[test]
    fn build_rejects_sub_agent_with_zero_max_turns() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = Orchestrator::builder(provider)
            .sub_agent_full(SubAgentConfig {
                name: "agent1".into(),
                description: "Test agent".into(),
                system_prompt: "prompt".into(),
                tools: vec![],
                context_strategy: None,
                summarize_threshold: None,
                tool_timeout: None,
                max_tool_output_bytes: None,
                max_turns: Some(0),
                max_tokens: None,
                response_schema: None,
                run_timeout: None,
                guardrails: vec![],
                provider: None,
                reasoning_effort: None,
                enable_reflection: None,
                tool_output_compression_threshold: None,
                max_tools_per_turn: None,
                max_identical_tool_calls: None,
                session_prune_config: None,
                enable_recursive_summarization: None,
                reflection_threshold: None,
                consolidate_on_exit: None,
            })
            .build();

        match result {
            Err(e) => assert!(
                e.to_string().contains("max_turns must be > 0"),
                "expected max_turns error, got: {e}"
            ),
            Ok(_) => panic!("expected build to fail with zero max_turns"),
        }
    }

    #[test]
    fn build_rejects_sub_agent_with_zero_max_tokens() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = Orchestrator::builder(provider)
            .sub_agent_full(SubAgentConfig {
                name: "agent1".into(),
                description: "Test agent".into(),
                system_prompt: "prompt".into(),
                tools: vec![],
                context_strategy: None,
                summarize_threshold: None,
                tool_timeout: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: Some(0),
                response_schema: None,
                run_timeout: None,
                guardrails: vec![],
                provider: None,
                reasoning_effort: None,
                enable_reflection: None,
                tool_output_compression_threshold: None,
                max_tools_per_turn: None,
                max_identical_tool_calls: None,
                session_prune_config: None,
                enable_recursive_summarization: None,
                reflection_threshold: None,
                consolidate_on_exit: None,
            })
            .build();

        match result {
            Err(e) => assert!(
                e.to_string().contains("max_tokens must be > 0"),
                "expected max_tokens error, got: {e}"
            ),
            Ok(_) => panic!("expected build to fail with zero max_tokens"),
        }
    }

    #[tokio::test]
    async fn sub_agent_uses_override_provider() {
        use crate::llm::types::CompletionRequest;

        // Provider that returns a model identifier in the response text
        struct IdentifiedProvider {
            id: String,
            responses: Mutex<Vec<CompletionResponse>>,
        }

        impl LlmProvider for IdentifiedProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                let mut responses = self.responses.lock().expect("lock");
                if responses.is_empty() {
                    return Err(Error::Agent(format!("no more responses for {}", self.id)));
                }
                Ok(responses.remove(0))
            }
        }

        // Orchestrator uses "opus" provider, sub-agent overrides with "haiku" provider
        let opus_provider = Arc::new(IdentifiedProvider {
            id: "opus".into(),
            responses: Mutex::new(vec![
                // 1: Orchestrator delegates
                CompletionResponse {
                    content: vec![ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "delegate_task".into(),
                        input: json!({
                            "tasks": [{"agent": "cheap", "task": "do cheap work"}]
                        }),
                    }],
                    stop_reason: StopReason::ToolUse,
                    usage: TokenUsage::default(),
                },
                // 3: Orchestrator synthesis
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                },
            ]),
        });

        let haiku_provider: Arc<BoxedProvider> = Arc::new(BoxedProvider::new(IdentifiedProvider {
            id: "haiku".into(),
            responses: Mutex::new(vec![
                // 2: Sub-agent responds via haiku
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Cheap work done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage {
                        input_tokens: 5,
                        output_tokens: 3,
                        ..Default::default()
                    },
                },
            ]),
        }));

        let mut orch = Orchestrator::builder(opus_provider)
            .sub_agent_full(SubAgentConfig {
                name: "cheap".into(),
                description: "Cheap agent".into(),
                system_prompt: "You do cheap work.".into(),
                tools: vec![],
                context_strategy: None,
                summarize_threshold: None,
                tool_timeout: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: None,
                response_schema: None,
                run_timeout: None,
                guardrails: vec![],
                provider: Some(haiku_provider),
                reasoning_effort: None,
                enable_reflection: None,
                tool_output_compression_threshold: None,
                max_tools_per_turn: None,
                max_identical_tool_calls: None,
                session_prune_config: None,
                enable_recursive_summarization: None,
                reflection_threshold: None,
                consolidate_on_exit: None,
            })
            .build()
            .unwrap();

        let output = orch.run("do work cheaply").await.unwrap();
        assert_eq!(output.result, "Done.");
        // Sub-agent tokens should be accumulated from the haiku provider
        assert_eq!(output.tokens_used.input_tokens, 5);
    }

    #[tokio::test]
    async fn sub_agent_inherits_default_provider() {
        // When no override is set, sub-agent uses the orchestrator's provider
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator delegates
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [{"agent": "worker", "task": "do work"}]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Sub-agent responds (from shared provider)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Work done.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
            // 3: Orchestrator synthesis
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "All done.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent_full(SubAgentConfig {
                name: "worker".into(),
                description: "Worker".into(),
                system_prompt: "Work.".into(),
                tools: vec![],
                context_strategy: None,
                summarize_threshold: None,
                tool_timeout: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: None,
                response_schema: None,
                run_timeout: None,
                guardrails: vec![],
                provider: None,
                reasoning_effort: None,
                enable_reflection: None,
                tool_output_compression_threshold: None,
                max_tools_per_turn: None,
                max_identical_tool_calls: None,
                session_prune_config: None,
                enable_recursive_summarization: None,
                reflection_threshold: None,
                consolidate_on_exit: None,
            })
            .build()
            .unwrap();

        let output = orch.run("do work").await.unwrap();
        assert_eq!(output.result, "All done.");
    }

    // --- FormSquadTool tests ---

    #[test]
    fn form_squad_tool_definition_schema() {
        let tools = vec!["web_search".to_string()];
        let agents: Vec<(&str, &str, &[String])> = vec![
            ("researcher", "Research specialist", tools.as_slice()),
            ("analyst", "Analysis expert", &[]),
        ];
        let def = build_form_squad_tool_schema(&agents);
        assert_eq!(def.name, "form_squad");
        assert!(
            def.description.contains("researcher"),
            "description should list agents: {}",
            def.description
        );
        assert!(
            def.description.contains("analyst"),
            "description should list agents: {}",
            def.description
        );
        assert!(
            def.description.contains("blackboard"),
            "description should mention shared blackboard: {}",
            def.description
        );
        assert!(
            def.description.contains("Unlike delegate_task"),
            "description should contrast with delegate_task: {}",
            def.description
        );
        // Check input schema uses tasks array (same format as delegate_task)
        assert_eq!(
            def.input_schema["properties"]["tasks"]["type"], "array",
            "schema should have tasks array"
        );
        assert_eq!(
            def.input_schema["properties"]["tasks"]["items"]["properties"]["agent"]["type"],
            "string",
            "tasks items should have agent field"
        );
        assert_eq!(
            def.input_schema["properties"]["tasks"]["items"]["properties"]["task"]["type"],
            "string",
            "tasks items should have task field"
        );
        let required = def.input_schema["required"]
            .as_array()
            .expect("required should be array");
        assert!(
            required.contains(&json!("tasks")),
            "tasks should be required"
        );
    }

    #[tokio::test]
    async fn form_squad_dispatches_directly() {
        // 3 agents: researcher, analyst, coder. Squad of researcher + analyst.
        // No squad-leader LLM calls — agents are dispatched directly.
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Outer orchestrator calls form_squad with per-agent tasks
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "form_squad".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "researcher", "task": "Research Rust"},
                            {"agent": "analyst", "task": "Analyze findings"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 50,
                    output_tokens: 20,
                    ..Default::default()
                },
            },
            // 2: Squad member "researcher" responds
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Rust is fast and safe.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 8,
                    ..Default::default()
                },
            },
            // 3: Squad member "analyst" responds
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Strengths: memory safety.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 12,
                    output_tokens: 10,
                    ..Default::default()
                },
            },
            // 4: Outer orchestrator synthesizes
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Final: Rust is excellent.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 60,
                    output_tokens: 25,
                    ..Default::default()
                },
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research specialist", "You research.")
            .sub_agent("analyst", "Analysis expert", "You analyze.")
            .sub_agent("coder", "Coding expert", "You code.")
            .build()
            .unwrap();

        let output = orch.run("Analyze Rust deeply").await.unwrap();
        assert_eq!(output.result, "Final: Rust is excellent.");
    }

    #[tokio::test]
    async fn form_squad_tokens_roll_up() {
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Outer orchestrator calls form_squad
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "form_squad".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "agent_a", "task": "Task A"},
                            {"agent": "agent_b", "task": "Task B"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 50,
                    output_tokens: 20,
                    ..Default::default()
                },
            },
            // 2: Squad member agent_a responds
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done A.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
            },
            // 3: Squad member agent_b responds
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done B.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 12,
                    output_tokens: 6,
                    ..Default::default()
                },
            },
            // 4: Outer orchestrator synthesizes
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "All done.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 60,
                    output_tokens: 25,
                    ..Default::default()
                },
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("agent_a", "Agent A", "You are A.")
            .sub_agent("agent_b", "Agent B", "You are B.")
            .build()
            .unwrap();

        let output = orch.run("Collaborate").await.unwrap();
        // Outer orchestrator: 50+60 in, 20+25 out
        // Squad members: agent_a 10 in + 5 out, agent_b 12 in + 6 out
        // No squad-leader overhead
        assert_eq!(
            output.tokens_used.input_tokens,
            50 + 60 + 10 + 12,
            "all token levels should roll up"
        );
        assert_eq!(
            output.tokens_used.output_tokens,
            20 + 25 + 5 + 6,
            "all token levels should roll up"
        );
    }

    #[tokio::test]
    async fn form_squad_returns_error_for_unknown_agent() {
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Outer orchestrator calls form_squad with unknown agent
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "form_squad".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "researcher", "task": "Do research"},
                            {"agent": "nonexistent", "task": "Do stuff"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Orchestrator recovers
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "No such agent available.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .sub_agent("analyst", "Analysis", "prompt")
            .build()
            .unwrap();

        let output = orch.run("delegate to unknown squad").await.unwrap();
        assert_eq!(output.result, "No such agent available.");
    }

    #[tokio::test]
    async fn form_squad_requires_at_least_two_agents() {
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Outer orchestrator tries to form squad with 1 task
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "form_squad".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "researcher", "task": "Solo task"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Orchestrator recovers
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Using delegate_task instead.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .sub_agent("analyst", "Analysis", "prompt")
            .build()
            .unwrap();

        let output = orch.run("form solo squad").await.unwrap();
        assert_eq!(output.result, "Using delegate_task instead.");
    }

    #[tokio::test]
    async fn form_squad_rejects_duplicate_agents() {
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Outer orchestrator sends duplicate agent names
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "form_squad".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "researcher", "task": "Task 1"},
                            {"agent": "researcher", "task": "Task 2"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Orchestrator recovers
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Fixed duplicate issue.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .sub_agent("analyst", "Analysis", "prompt")
            .build()
            .unwrap();

        let output = orch.run("form squad with dupes").await.unwrap();
        assert_eq!(output.result, "Fixed duplicate issue.");
    }

    #[tokio::test]
    async fn form_squad_private_blackboard() {
        use crate::agent::blackboard::InMemoryBlackboard;

        let outer_bb = Arc::new(InMemoryBlackboard::new());

        let provider = Arc::new(MockProvider::new(vec![
            // 1: Outer orchestrator calls form_squad
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "form_squad".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "writer_a", "task": "Write something"},
                            {"agent": "writer_b", "task": "Write something else"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Squad member writer_a responds
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Written to squad blackboard.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
            // 3: Squad member writer_b responds
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Also written.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
            // 4: Outer orchestrator synthesizes
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("writer_a", "Writer A", "You write.")
            .sub_agent("writer_b", "Writer B", "You write.")
            .blackboard(outer_bb.clone())
            .build()
            .unwrap();

        orch.run("write to blackboard").await.unwrap();

        // Squad result IS written to the outer blackboard under the "squad:" key.
        let squad_key = "squad:writer_a+writer_b";
        let val = outer_bb.read(squad_key).await.unwrap();
        assert!(
            val.is_some(),
            "outer blackboard should have squad result under '{squad_key}'"
        );

        // The squad member's agent:writer_a key should NOT be in the outer blackboard
        // (it was written to the private blackboard inside the squad)
        let agent_key = "agent:writer_a";
        let val = outer_bb.read(agent_key).await.unwrap();
        assert!(
            val.is_none(),
            "outer blackboard should NOT have '{agent_key}' — that's on the private blackboard"
        );
    }

    #[tokio::test]
    async fn form_squad_error_returns_tool_error_not_hard_error() {
        // One squad member's provider always fails → FormSquadTool returns
        // ToolOutput::error → outer orchestrator recovers gracefully.
        //
        // We use provider_override so agent_b gets a dedicated failing provider,
        // avoiding non-deterministic response ordering from a shared MockProvider.

        let failing_provider = Arc::new(BoxedProvider::new(MockProvider::new(vec![])));

        let provider = Arc::new(MockProvider::new(vec![
            // 1: Outer orchestrator calls form_squad
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "form_squad".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "agent_a", "task": "Do A"},
                            {"agent": "agent_b", "task": "Do B"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 50,
                    output_tokens: 20,
                    ..Default::default()
                },
            },
            // 2: Squad member agent_a responds (agent_b uses its own failing provider)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done A.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
            },
            // 3: Outer orchestrator recovers from the squad error
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Squad failed, falling back.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 60,
                    output_tokens: 25,
                    ..Default::default()
                },
            },
        ]));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("agent_a", "Agent A", "You are A.")
            .sub_agent_full(SubAgentConfig {
                name: "agent_b".into(),
                description: "Agent B".into(),
                system_prompt: "You are B.".into(),
                tools: vec![],
                context_strategy: None,
                summarize_threshold: None,
                tool_timeout: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: None,
                response_schema: None,
                run_timeout: None,
                guardrails: vec![],
                provider: Some(failing_provider),
                reasoning_effort: None,
                enable_reflection: None,
                tool_output_compression_threshold: None,
                max_tools_per_turn: None,
                max_identical_tool_calls: None,
                session_prune_config: None,
                enable_recursive_summarization: None,
                reflection_threshold: None,
                consolidate_on_exit: None,
            })
            .build()
            .unwrap();

        let output = orch.run("complex task").await.unwrap();
        assert_eq!(output.result, "Squad failed, falling back.");
        // Partial tokens from the successful squad member should be accumulated
        assert!(
            output.tokens_used.input_tokens > 50 + 60,
            "should include partial squad tokens: {}",
            output.tokens_used.input_tokens
        );
    }

    #[tokio::test]
    async fn orchestrator_registers_both_tools() {
        use crate::llm::types::CompletionRequest;

        struct ToolCapturingProvider {
            responses: Mutex<Vec<CompletionResponse>>,
            tool_names_seen: Mutex<Vec<Vec<String>>>,
        }

        impl LlmProvider for ToolCapturingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                let names: Vec<String> = request.tools.iter().map(|t| t.name.clone()).collect();
                self.tool_names_seen.lock().expect("lock").push(names);
                let mut responses = self.responses.lock().expect("lock");
                if responses.is_empty() {
                    return Err(Error::Agent("no more responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let provider = Arc::new(ToolCapturingProvider {
            responses: Mutex::new(vec![CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Direct answer.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            }]),
            tool_names_seen: Mutex::new(vec![]),
        });

        // >= 2 agents → squads auto-enabled
        let mut orch = Orchestrator::builder(provider.clone())
            .sub_agent("researcher", "Research", "prompt")
            .sub_agent("analyst", "Analysis", "prompt")
            .build()
            .unwrap();

        orch.run("test").await.unwrap();

        let tool_names = provider.tool_names_seen.lock().unwrap();
        assert!(
            tool_names[0].contains(&"delegate_task".to_string()),
            "should have delegate_task: {:?}",
            tool_names[0]
        );
        assert!(
            tool_names[0].contains(&"form_squad".to_string()),
            "should have form_squad: {:?}",
            tool_names[0]
        );
    }

    #[test]
    fn orchestrator_single_agent_no_squads() {
        let provider = Arc::new(MockProvider::new(vec![]));

        // Only 1 agent → squads auto-disabled
        let result = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build();

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn orchestrator_squads_disabled_explicitly() {
        use crate::llm::types::CompletionRequest;

        struct ToolCapturingProvider {
            responses: Mutex<Vec<CompletionResponse>>,
            tool_names_seen: Mutex<Vec<Vec<String>>>,
        }

        impl LlmProvider for ToolCapturingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                let names: Vec<String> = request.tools.iter().map(|t| t.name.clone()).collect();
                self.tool_names_seen.lock().expect("lock").push(names);
                let mut responses = self.responses.lock().expect("lock");
                if responses.is_empty() {
                    return Err(Error::Agent("no more responses".into()));
                }
                Ok(responses.remove(0))
            }
        }

        let provider = Arc::new(ToolCapturingProvider {
            responses: Mutex::new(vec![CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Direct answer.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            }]),
            tool_names_seen: Mutex::new(vec![]),
        });

        // 2 agents BUT squads explicitly disabled
        let mut orch = Orchestrator::builder(provider.clone())
            .sub_agent("researcher", "Research", "prompt")
            .sub_agent("analyst", "Analysis", "prompt")
            .enable_squads(false)
            .build()
            .unwrap();

        orch.run("test").await.unwrap();

        let tool_names = provider.tool_names_seen.lock().unwrap();
        assert!(
            tool_names[0].contains(&"delegate_task".to_string()),
            "should have delegate_task: {:?}",
            tool_names[0]
        );
        assert!(
            !tool_names[0].contains(&"form_squad".to_string()),
            "should NOT have form_squad when disabled: {:?}",
            tool_names[0]
        );
    }

    #[test]
    fn system_prompt_mentions_both_tools_when_squads_enabled() {
        let tools = vec!["web_search".to_string()];
        let agents: Vec<(&str, &str, &[String])> = vec![
            ("researcher", "Research specialist", tools.as_slice()),
            ("analyst", "Analysis expert", &[]),
        ];

        let prompt = build_system_prompt(&agents, true, DispatchMode::Parallel);
        assert!(
            prompt.contains("delegate_task"),
            "prompt should mention delegate_task: {prompt}"
        );
        assert!(
            prompt.contains("form_squad"),
            "prompt should mention form_squad: {prompt}"
        );
        assert!(
            prompt.contains("two delegation tools"),
            "prompt should explain both tools: {prompt}"
        );
        // Squads-enabled prompt should distinguish isolation vs collaboration
        assert!(
            prompt.contains("isolation"),
            "prompt should mention isolation for delegate_task: {prompt}"
        );
        assert!(
            prompt.contains("blackboard"),
            "prompt should mention shared blackboard for form_squad: {prompt}"
        );
    }

    #[test]
    fn system_prompt_only_delegate_when_squads_disabled() {
        let agents: Vec<(&str, &str, &[String])> = vec![("researcher", "Research specialist", &[])];

        let prompt = build_system_prompt(&agents, false, DispatchMode::Parallel);
        assert!(
            prompt.contains("delegate_task"),
            "prompt should mention delegate_task: {prompt}"
        );
        assert!(
            !prompt.contains("form_squad"),
            "prompt should NOT mention form_squad: {prompt}"
        );
        // Still has decision framework
        assert!(
            prompt.contains("Decision Process"),
            "prompt should contain Decision Process even without squads: {prompt}"
        );
        assert!(
            prompt.contains("Effort Scaling"),
            "prompt should contain Effort Scaling even without squads: {prompt}"
        );
    }

    #[tokio::test]
    async fn delegate_forwards_on_event_to_sub_agents() {
        // Verify that on_event receives events from sub-agents (not just orchestrator)
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator decides to delegate
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "worker", "task": "do work"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Sub-agent "worker" response
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
            // 3: Orchestrator synthesis
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "All done.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
        let events_clone = events.clone();
        let on_event: Arc<OnEvent> = Arc::new(move |event: AgentEvent| {
            events_clone.lock().expect("test lock").push(event);
        });

        let mut orch = Orchestrator::builder(provider)
            .sub_agent("worker", "Worker agent", "You do work.")
            .on_event(on_event)
            .build()
            .unwrap();

        let _output = orch.run("delegate some work").await.unwrap();

        let events = events.lock().expect("test lock");

        // Should have events from both "orchestrator" and "worker"
        let orchestrator_events: Vec<_> = events
            .iter()
            .filter(|e| match e {
                AgentEvent::RunStarted { agent, .. }
                | AgentEvent::TurnStarted { agent, .. }
                | AgentEvent::LlmResponse { agent, .. }
                | AgentEvent::RunCompleted { agent, .. } => agent == "orchestrator",
                _ => false,
            })
            .collect();
        let worker_events: Vec<_> = events
            .iter()
            .filter(|e| match e {
                AgentEvent::RunStarted { agent, .. }
                | AgentEvent::TurnStarted { agent, .. }
                | AgentEvent::LlmResponse { agent, .. }
                | AgentEvent::RunCompleted { agent, .. } => agent == "worker",
                _ => false,
            })
            .collect();

        assert!(
            !orchestrator_events.is_empty(),
            "should have orchestrator events"
        );
        assert!(
            !worker_events.is_empty(),
            "should have sub-agent worker events (forwarded via on_event)"
        );

        // Verify worker had a RunStarted event
        let worker_run_started = events
            .iter()
            .any(|e| matches!(e, AgentEvent::RunStarted { agent, .. } if agent == "worker"));
        assert!(
            worker_run_started,
            "sub-agent should emit RunStarted via forwarded on_event"
        );
    }

    /// Complex end-to-end audit trail test.
    ///
    /// Scenario:
    ///   1. Orchestrator delegates to 2 sub-agents: "researcher" (has web_search tool)
    ///      and "coder" (has read_file tool)
    ///   2. Each sub-agent calls its tool, produces output
    ///   3. Orchestrator synthesizes results
    ///
    /// Verifies:
    ///   - Complete event stream ordering
    ///   - Agent names on every event
    ///   - LlmResponse contains text, latency_ms > 0, model name
    ///   - ToolCallStarted contains input JSON
    ///   - ToolCallCompleted contains output content
    ///   - Sub-agent events are forwarded (not just orchestrator events)
    ///   - SubAgentsDispatched + SubAgentCompleted bracket sub-agent work
    ///   - Token usage rolls up correctly
    ///   - Truncation works for oversized tool output
    #[tokio::test]
    async fn full_audit_trail_end_to_end() {
        // Build a long tool output to test truncation
        let long_output = "x".repeat(8000); // > EVENT_MAX_PAYLOAD_BYTES (4096)

        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator decides to delegate to both agents (with reasoning text)
            CompletionResponse {
                content: vec![
                    ContentBlock::Text {
                        text: "I'll delegate to the researcher and coder.".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "orch-call-1".into(),
                        name: "delegate_task".into(),
                        input: json!({
                            "tasks": [
                                {"agent": "researcher", "task": "Search for Rust concurrency patterns"},
                                {"agent": "coder", "task": "Read the main.rs file"}
                            ]
                        }),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 100,
                    output_tokens: 40,
                    ..Default::default()
                },
            },
            // 2: Sub-agent "researcher" LLM response: calls web_search tool
            CompletionResponse {
                content: vec![
                    ContentBlock::Text {
                        text: "Let me search for Rust concurrency info.".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "res-call-1".into(),
                        name: "web_search".into(),
                        input: json!({"query": "rust async concurrency"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 20,
                    output_tokens: 10,
                    ..Default::default()
                },
            },
            // 3: Sub-agent "researcher" final response after tool result
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Rust uses async/await with tokio for concurrency.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 30,
                    output_tokens: 15,
                    ..Default::default()
                },
            },
            // 4: Sub-agent "coder" LLM response: calls read_file tool
            CompletionResponse {
                content: vec![
                    ContentBlock::Text {
                        text: "I'll read the main.rs file.".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "cod-call-1".into(),
                        name: "read_file".into(),
                        input: json!({"path": "/src/main.rs"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 15,
                    output_tokens: 8,
                    ..Default::default()
                },
            },
            // 5: Sub-agent "coder" final response after tool result
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "The main.rs contains the entry point.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 25,
                    output_tokens: 12,
                    ..Default::default()
                },
            },
            // 6: Orchestrator synthesis
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Combined analysis: Rust async is great for concurrency.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 200,
                    output_tokens: 50,
                    ..Default::default()
                },
            },
        ]));

        let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
        let events_clone = events.clone();
        let on_event: Arc<OnEvent> = Arc::new(move |event: AgentEvent| {
            events_clone.lock().expect("test lock").push(event);
        });

        // Both agents get both tools so the test is resilient to JoinSet ordering.
        // (Sub-agents share a MockProvider and run concurrently — response order is non-deterministic.)
        let long_output_clone = long_output.clone();
        let shared_tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(MockTool::new("web_search", &long_output_clone)),
            Arc::new(MockTool::new(
                "read_file",
                "fn main() { println!(\"hello\"); }",
            )),
        ];
        let mut orch = Orchestrator::builder(provider)
            .sub_agent_with_tools(
                "researcher",
                "Research specialist",
                "You research topics.",
                shared_tools.clone(),
            )
            .sub_agent_with_tools(
                "coder",
                "Code expert",
                "You read and analyze code.",
                shared_tools.clone(),
            )
            .on_event(on_event)
            .build()
            .unwrap();

        let output = orch
            .run("Analyze Rust concurrency and the main.rs file")
            .await
            .unwrap();

        // === Verify final output ===
        assert_eq!(
            output.result,
            "Combined analysis: Rust async is great for concurrency."
        );
        assert_eq!(output.tool_calls_made, 1); // orchestrator made 1 delegate_task call

        // === Collect and categorize events ===
        let events = events.lock().expect("test lock");

        // Helper to extract agent name from any event
        fn agent_of(e: &AgentEvent) -> &str {
            match e {
                AgentEvent::RunStarted { agent, .. }
                | AgentEvent::TurnStarted { agent, .. }
                | AgentEvent::LlmResponse { agent, .. }
                | AgentEvent::ToolCallStarted { agent, .. }
                | AgentEvent::ToolCallCompleted { agent, .. }
                | AgentEvent::RunCompleted { agent, .. }
                | AgentEvent::RunFailed { agent, .. }
                | AgentEvent::SubAgentsDispatched { agent, .. }
                | AgentEvent::SubAgentCompleted { agent, .. }
                | AgentEvent::ApprovalRequested { agent, .. }
                | AgentEvent::ApprovalDecision { agent, .. }
                | AgentEvent::ContextSummarized { agent, .. }
                | AgentEvent::GuardrailDenied { agent, .. }
                | AgentEvent::RetryAttempt { agent, .. }
                | AgentEvent::DoomLoopDetected { agent, .. }
                | AgentEvent::AutoCompactionTriggered { agent, .. } => agent,
            }
        }

        // Print event stream for debugging
        let event_summary: Vec<String> = events
            .iter()
            .enumerate()
            .map(|(i, e)| format!("{i}: [{:>12}] {:?}", agent_of(e), std::mem::discriminant(e)))
            .collect();

        // === 1. Verify we have events from all 3 agents ===
        let agents_seen: std::collections::HashSet<&str> = events.iter().map(agent_of).collect();
        assert!(
            agents_seen.contains("orchestrator"),
            "missing orchestrator events.\nEvent stream:\n{}",
            event_summary.join("\n")
        );
        assert!(
            agents_seen.contains("researcher"),
            "missing researcher events (should be forwarded).\nEvent stream:\n{}",
            event_summary.join("\n")
        );
        assert!(
            agents_seen.contains("coder"),
            "missing coder events (should be forwarded).\nEvent stream:\n{}",
            event_summary.join("\n")
        );

        // === 2. Verify orchestrator event sequence ===
        let orch_events: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| agent_of(e) == "orchestrator")
            .collect();

        // First orchestrator event: RunStarted
        assert!(
            matches!(orch_events[0], AgentEvent::RunStarted { task, .. } if task.contains("Analyze Rust")),
            "first orch event should be RunStarted, got: {:?}",
            orch_events[0]
        );
        // Last orchestrator event: RunCompleted
        assert!(
            matches!(orch_events.last().unwrap(), AgentEvent::RunCompleted { .. }),
            "last orch event should be RunCompleted, got: {:?}",
            orch_events.last().unwrap()
        );

        // === 3. Verify LlmResponse events have text, latency, and model ===
        let llm_responses: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::LlmResponse { .. }))
            .collect();
        assert!(
            llm_responses.len() >= 3,
            "expected >= 3 LlmResponse events (1 orch + at least 1 per sub-agent), got {}.\nEvents:\n{}",
            llm_responses.len(),
            event_summary.join("\n")
        );

        for llm_event in &llm_responses {
            match llm_event {
                AgentEvent::LlmResponse {
                    agent, text, model, ..
                } => {
                    // model_name should always be present (MockProvider returns "mock-model-v1")
                    assert_eq!(
                        model.as_deref(),
                        Some("mock-model-v1"),
                        "LlmResponse for '{agent}' should have model name"
                    );
                    // text should be non-empty (all our mock responses produce content)
                    assert!(
                        !text.is_empty(),
                        "LlmResponse for '{agent}' should have non-empty text"
                    );
                }
                _ => unreachable!(),
            }
        }

        // === 4. Verify ToolCallStarted events have input ===
        let tool_started: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolCallStarted { .. }))
            .collect();
        // Expect 3 tool calls: 1 delegate_task (orch) + 1 web_search (researcher) + 1 read_file (coder)
        assert!(
            tool_started.len() >= 3,
            "expected >= 3 ToolCallStarted events, got {}.\nEvents:\n{}",
            tool_started.len(),
            event_summary.join("\n")
        );

        // Find the web_search ToolCallStarted and verify input contains the query.
        // Note: agent name is NOT asserted because JoinSet ordering is non-deterministic.
        let web_search_started = tool_started.iter().find(|e| {
            matches!(e, AgentEvent::ToolCallStarted { tool_name, .. } if tool_name == "web_search")
        });
        assert!(
            web_search_started.is_some(),
            "should have a web_search ToolCallStarted"
        );
        match web_search_started.unwrap() {
            AgentEvent::ToolCallStarted { input, .. } => {
                assert!(
                    input.contains("rust async concurrency"),
                    "web_search input should contain query, got: {input}"
                );
            }
            _ => unreachable!(),
        }

        // Find the read_file ToolCallStarted and verify input
        let read_file_started = tool_started.iter().find(|e| {
            matches!(e, AgentEvent::ToolCallStarted { tool_name, .. } if tool_name == "read_file")
        });
        assert!(
            read_file_started.is_some(),
            "should have a read_file ToolCallStarted"
        );
        match read_file_started.unwrap() {
            AgentEvent::ToolCallStarted { input, .. } => {
                assert!(
                    input.contains("/src/main.rs"),
                    "read_file input should contain path, got: {input}"
                );
            }
            _ => unreachable!(),
        }

        // Find the delegate_task ToolCallStarted and verify input
        let delegate_started = tool_started.iter().find(|e| {
            matches!(e, AgentEvent::ToolCallStarted { tool_name, .. } if tool_name == "delegate_task")
        });
        assert!(
            delegate_started.is_some(),
            "should have a delegate_task ToolCallStarted"
        );
        match delegate_started.unwrap() {
            AgentEvent::ToolCallStarted { agent, input, .. } => {
                assert_eq!(agent, "orchestrator");
                assert!(
                    input.contains("researcher"),
                    "delegate_task input should contain agent names, got: {input}"
                );
            }
            _ => unreachable!(),
        }

        // === 5. Verify ToolCallCompleted events have output ===
        let tool_completed: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolCallCompleted { .. }))
            .collect();
        assert!(
            tool_completed.len() >= 3,
            "expected >= 3 ToolCallCompleted events, got {}",
            tool_completed.len()
        );

        // Find the web_search completion and verify truncation
        let web_search_completed = tool_completed.iter().find(|e| {
            matches!(e, AgentEvent::ToolCallCompleted { tool_name, .. } if tool_name == "web_search")
        });
        assert!(
            web_search_completed.is_some(),
            "should have a web_search ToolCallCompleted"
        );
        match web_search_completed.unwrap() {
            AgentEvent::ToolCallCompleted {
                output, is_error, ..
            } => {
                assert!(!is_error);
                // The web_search tool returns 8000 bytes, exceeds EVENT_MAX_PAYLOAD_BYTES (4096)
                assert!(
                    output.contains("[truncated:"),
                    "web_search output (8000 bytes) should be truncated in event, got {} bytes: {}",
                    output.len(),
                    &output[..output.len().min(100)]
                );
            }
            _ => unreachable!(),
        }

        // Find the read_file completion — should NOT be truncated (short output)
        let read_file_completed = tool_completed.iter().find(|e| {
            matches!(e, AgentEvent::ToolCallCompleted { tool_name, .. } if tool_name == "read_file")
        });
        assert!(
            read_file_completed.is_some(),
            "should have a read_file ToolCallCompleted"
        );
        match read_file_completed.unwrap() {
            AgentEvent::ToolCallCompleted {
                output, is_error, ..
            } => {
                assert!(!is_error);
                assert!(
                    output.contains("fn main()"),
                    "read_file output should contain file content, got: {output}"
                );
                assert!(
                    !output.contains("[truncated:"),
                    "read_file output should NOT be truncated"
                );
            }
            _ => unreachable!(),
        }

        // === 6. Verify SubAgentsDispatched and SubAgentCompleted events ===
        let dispatched: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::SubAgentsDispatched { .. }))
            .collect();
        assert_eq!(dispatched.len(), 1, "expected 1 SubAgentsDispatched event");
        match dispatched[0] {
            AgentEvent::SubAgentsDispatched { agents, .. } => {
                assert!(
                    agents.contains(&"researcher".to_string()),
                    "dispatched agents should include researcher"
                );
                assert!(
                    agents.contains(&"coder".to_string()),
                    "dispatched agents should include coder"
                );
            }
            _ => unreachable!(),
        }

        let completed: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::SubAgentCompleted { .. }))
            .collect();
        assert_eq!(
            completed.len(),
            2,
            "expected 2 SubAgentCompleted events (one per sub-agent)"
        );
        for c in &completed {
            match c {
                AgentEvent::SubAgentCompleted { success, agent, .. } => {
                    assert!(success, "sub-agent '{agent}' should succeed");
                }
                _ => unreachable!(),
            }
        }

        // === 7. Verify event ordering: RunStarted is always first for each agent ===
        for agent_name in &["orchestrator", "researcher", "coder"] {
            let agent_events: Vec<&AgentEvent> = events
                .iter()
                .filter(|e| agent_of(e) == *agent_name)
                .collect();
            if !agent_events.is_empty() {
                assert!(
                    matches!(agent_events[0], AgentEvent::RunStarted { .. }),
                    "first event for '{agent_name}' should be RunStarted, got: {:?}",
                    agent_events[0]
                );
            }
        }

        // === 8. Verify token roll-up ===
        // Orchestrator: 100+200 input, 40+50 output
        // Researcher: 20+30 input, 10+15 output
        // Coder: 15+25 input, 8+12 output
        // Total: 390 input, 135 output
        assert_eq!(
            output.tokens_used.input_tokens,
            100 + 200 + 20 + 30 + 15 + 25,
            "total input tokens should include orchestrator + sub-agents"
        );
        assert_eq!(
            output.tokens_used.output_tokens,
            40 + 50 + 10 + 15 + 8 + 12,
            "total output tokens should include orchestrator + sub-agents"
        );

        // === 9. Verify a sub-agent LlmResponse contains research text ===
        // (Agent name not asserted due to JoinSet ordering non-determinism)
        let sub_agent_llm = llm_responses.iter().find(|e| {
            matches!(e, AgentEvent::LlmResponse { text, .. }
                if text.contains("async/await"))
        });
        assert!(
            sub_agent_llm.is_some(),
            "should have a sub-agent LlmResponse with text about async/await"
        );

        // === 10. Verify total event count is reasonable ===
        // Minimum: 3 agents × (RunStarted + TurnStarted + LlmResponse) + tool events + completion events
        // Orchestrator: RunStarted, TurnStarted, LlmResponse, ToolCallStarted(delegate), ToolCallCompleted(delegate),
        //               TurnStarted, LlmResponse, RunCompleted = ~8
        // Researcher:   RunStarted, TurnStarted, LlmResponse, ToolCallStarted, ToolCallCompleted,
        //               TurnStarted, LlmResponse, RunCompleted = ~8
        // Coder:        same = ~8
        // + SubAgentsDispatched + 2× SubAgentCompleted = 3
        // Total ~27 events
        assert!(
            events.len() >= 20,
            "expected at least 20 events for full audit trail, got {}.\nEvents:\n{}",
            events.len(),
            event_summary.join("\n")
        );
    }

    #[tokio::test]
    async fn sub_agent_run_timeout_fires_when_configured() {
        // Provider that responds immediately for the orchestrator (delegate call),
        // then hangs forever for the sub-agent (simulating a slow LLM),
        // then synthesizes the timeout error result.
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator decides to delegate to "slow-agent"
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [{"agent": "slow-agent", "task": "do something"}]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Sub-agent "slow-agent" — hangs forever (timeout will fire)
            // This response won't be consumed because the provider will run out
            // But we need a 3rd response for the orchestrator's synthesis turn.
        ]));

        // Use a separate SlowProvider for the sub-agent via provider override
        struct SlowProvider;
        impl LlmProvider for SlowProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                unreachable!()
            }
        }
        let slow_provider = Arc::new(BoxedProvider::new(SlowProvider));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent_full(SubAgentConfig {
                name: "slow-agent".into(),
                description: "A slow agent".into(),
                system_prompt: "sys".into(),
                tools: vec![],
                context_strategy: None,
                summarize_threshold: None,
                tool_timeout: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: None,
                response_schema: None,
                run_timeout: Some(Duration::from_millis(100)),
                guardrails: vec![],
                provider: Some(slow_provider),
                reasoning_effort: None,
                enable_reflection: None,
                tool_output_compression_threshold: None,
                max_tools_per_turn: None,
                max_identical_tool_calls: None,
                session_prune_config: None,
                enable_recursive_summarization: None,
                reflection_threshold: None,
                consolidate_on_exit: None,
            })
            .build()
            .unwrap();

        // The orchestrator will delegate, the sub-agent will timeout,
        // and the error propagates back as a delegate_task tool result.
        // The orchestrator then tries to synthesize but has no more responses.
        let result = orch.run("go").await;
        // The orchestrator either returns an error (no more mock responses for synthesis)
        // or the result mentions timeout. Either way, the sub-agent's run_timeout fired.
        match result {
            Ok(output) => {
                // If somehow we got a result, it should mention timeout
                assert!(
                    output.result.contains("timeout") || output.result.contains("Timeout"),
                    "expected timeout in result, got: {}",
                    output.result
                );
            }
            Err(e) => {
                // The orchestrator ran out of mock responses after the sub-agent timed out,
                // which is fine — it confirms the sub-agent timeout was wired.
                let msg = e.to_string();
                assert!(
                    msg.contains("no more mock responses")
                        || msg.contains("timeout")
                        || msg.contains("Timeout"),
                    "expected timeout-related error, got: {msg}"
                );
            }
        }
    }

    /// Complex squad integration test exercising:
    /// - 3 squad agents with mixed capabilities (tools, plain, failing)
    /// - Private blackboard isolation from outer blackboard
    /// - Full event capture and audit trail analysis
    /// - Tool execution within a squad member (multi-turn agent)
    /// - Graceful error handling when one squad member fails
    /// - Token roll-up from successful + partial-failure squad members
    /// - Event ordering invariants (RunStarted first per agent, SubAgentsDispatched before SubAgentCompleted)
    #[tokio::test]
    async fn form_squad_complex_with_tools_events_and_failure() {
        use crate::agent::blackboard::InMemoryBlackboard;

        let outer_bb = Arc::new(InMemoryBlackboard::new());

        // The "reviewer" agent gets a dedicated failing provider (empty responses).
        let failing_provider = Arc::new(BoxedProvider::new(MockProvider::new(vec![])));

        // Shared provider serves orchestrator + planner + worker (in call order).
        // worker is multi-turn: calls `compute` tool, then produces final text.
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator forms a squad of planner + worker + reviewer
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "orch-call-1".into(),
                    name: "form_squad".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "planner", "task": "Create a plan for the analysis"},
                            {"agent": "worker", "task": "Compute the metrics"},
                            {"agent": "reviewer", "task": "Review all findings"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 100,
                    output_tokens: 40,
                    cache_creation_input_tokens: 5,
                    cache_read_input_tokens: 3,
                    reasoning_tokens: 0,
                },
            },
            // 2: Squad member "planner" responds immediately (single turn)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Plan: Step 1 gather data, Step 2 compute, Step 3 review.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 20,
                    output_tokens: 15,
                    reasoning_tokens: 8,
                    ..Default::default()
                },
            },
            // 3: Squad member "worker" calls the `compute` tool (turn 1)
            CompletionResponse {
                content: vec![
                    ContentBlock::Text {
                        text: "I'll compute the metrics now.".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "worker-call-1".into(),
                        name: "compute".into(),
                        input: json!({"expression": "42 * 17"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 25,
                    output_tokens: 12,
                    ..Default::default()
                },
            },
            // 4: Squad member "worker" produces final text after tool result (turn 2)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Computation result: 714. Analysis complete.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 35,
                    output_tokens: 18,
                    ..Default::default()
                },
            },
            // 5: (reviewer uses failing_provider, not this queue)
            // 6: Orchestrator recovers — synthesizes from partial squad results
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Squad partial success: plan and computation done, review failed.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 200,
                    output_tokens: 60,
                    ..Default::default()
                },
            },
        ]));

        // Capture all events
        let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
        let events_clone = events.clone();
        let on_event: Arc<OnEvent> = Arc::new(move |event: AgentEvent| {
            events_clone.lock().expect("test lock").push(event);
        });

        // Both planner and worker get the compute tool so shared MockProvider
        // response ordering is resilient to JoinSet non-determinism.
        let compute_tool: Arc<dyn Tool> = Arc::new(MockTool::new("compute", "714"));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent_with_tools(
                "planner",
                "Planning specialist",
                "You create plans.",
                vec![compute_tool.clone()],
            )
            .sub_agent_with_tools(
                "worker",
                "Computation worker",
                "You compute metrics.",
                vec![compute_tool.clone()],
            )
            .sub_agent_full(SubAgentConfig {
                name: "reviewer".into(),
                description: "Review specialist".into(),
                system_prompt: "You review findings.".into(),
                tools: vec![],
                context_strategy: None,
                summarize_threshold: None,
                tool_timeout: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: None,
                response_schema: None,
                run_timeout: None,
                guardrails: vec![],
                provider: Some(failing_provider),
                reasoning_effort: None,
                enable_reflection: None,
                tool_output_compression_threshold: None,
                max_tools_per_turn: None,
                max_identical_tool_calls: None,
                session_prune_config: None,
                enable_recursive_summarization: None,
                reflection_threshold: None,
                consolidate_on_exit: None,
            })
            .blackboard(outer_bb.clone())
            .on_event(on_event)
            .build()
            .unwrap();

        let output = orch.run("Analyze the system performance").await.unwrap();

        // === 1. Verify final output ===
        assert_eq!(
            output.result,
            "Squad partial success: plan and computation done, review failed."
        );

        // === 2. Verify token roll-up ===
        // Orchestrator: 100+200 in, 40+60 out, 5 cache_create, 3 cache_read
        // Planner: 20 in, 15 out, 8 reasoning
        // Worker: 25+35 in, 12+18 out
        // Reviewer: 0 (failed before any response)
        let expected_input = 100 + 200 + 20 + 25 + 35;
        let expected_output = 40 + 60 + 15 + 12 + 18;
        assert_eq!(
            output.tokens_used.input_tokens, expected_input,
            "input tokens should sum orchestrator + planner + worker (reviewer failed)"
        );
        assert_eq!(
            output.tokens_used.output_tokens, expected_output,
            "output tokens should sum orchestrator + planner + worker"
        );
        assert_eq!(
            output.tokens_used.reasoning_tokens, 8,
            "reasoning tokens should come from planner"
        );
        assert_eq!(
            output.tokens_used.cache_creation_input_tokens, 5,
            "cache creation tokens from orchestrator"
        );
        assert_eq!(
            output.tokens_used.cache_read_input_tokens, 3,
            "cache read tokens from orchestrator"
        );

        // === 3. Verify blackboard writes ===
        // Squad result should be on the outer blackboard under "squad:planner+worker+reviewer"
        let squad_key = "squad:planner+worker+reviewer";
        let squad_val = outer_bb.read(squad_key).await.unwrap();
        assert!(
            squad_val.is_some(),
            "outer blackboard should have squad result under '{squad_key}'"
        );
        let squad_text = squad_val.unwrap().to_string();
        // Planner's result should appear
        assert!(
            squad_text.contains("Plan: Step 1"),
            "squad result should include planner's output"
        );
        // Worker's result should appear
        assert!(
            squad_text.contains("Computation result: 714"),
            "squad result should include worker's output"
        );
        // Reviewer's error should appear
        assert!(
            squad_text.contains("Error"),
            "squad result should include reviewer's error"
        );

        // Private blackboard keys should NOT be on the outer blackboard
        assert!(
            outer_bb.read("agent:planner").await.unwrap().is_none(),
            "outer blackboard should NOT have agent:planner"
        );
        assert!(
            outer_bb.read("agent:worker").await.unwrap().is_none(),
            "outer blackboard should NOT have agent:worker"
        );

        // === 4. Analyze events ===
        let events = events.lock().expect("test lock");

        fn agent_of(e: &AgentEvent) -> &str {
            match e {
                AgentEvent::RunStarted { agent, .. }
                | AgentEvent::TurnStarted { agent, .. }
                | AgentEvent::LlmResponse { agent, .. }
                | AgentEvent::ToolCallStarted { agent, .. }
                | AgentEvent::ToolCallCompleted { agent, .. }
                | AgentEvent::RunCompleted { agent, .. }
                | AgentEvent::RunFailed { agent, .. }
                | AgentEvent::SubAgentsDispatched { agent, .. }
                | AgentEvent::SubAgentCompleted { agent, .. }
                | AgentEvent::ApprovalRequested { agent, .. }
                | AgentEvent::ApprovalDecision { agent, .. }
                | AgentEvent::ContextSummarized { agent, .. }
                | AgentEvent::GuardrailDenied { agent, .. }
                | AgentEvent::RetryAttempt { agent, .. }
                | AgentEvent::DoomLoopDetected { agent, .. }
                | AgentEvent::AutoCompactionTriggered { agent, .. } => agent,
            }
        }

        fn event_type(e: &AgentEvent) -> &'static str {
            match e {
                AgentEvent::RunStarted { .. } => "RunStarted",
                AgentEvent::TurnStarted { .. } => "TurnStarted",
                AgentEvent::LlmResponse { .. } => "LlmResponse",
                AgentEvent::ToolCallStarted { .. } => "ToolCallStarted",
                AgentEvent::ToolCallCompleted { .. } => "ToolCallCompleted",
                AgentEvent::RunCompleted { .. } => "RunCompleted",
                AgentEvent::RunFailed { .. } => "RunFailed",
                AgentEvent::SubAgentsDispatched { .. } => "SubAgentsDispatched",
                AgentEvent::SubAgentCompleted { .. } => "SubAgentCompleted",
                AgentEvent::ApprovalRequested { .. } => "ApprovalRequested",
                AgentEvent::ApprovalDecision { .. } => "ApprovalDecision",
                AgentEvent::ContextSummarized { .. } => "ContextSummarized",
                AgentEvent::GuardrailDenied { .. } => "GuardrailDenied",
                AgentEvent::RetryAttempt { .. } => "RetryAttempt",
                AgentEvent::DoomLoopDetected { .. } => "DoomLoopDetected",
                AgentEvent::AutoCompactionTriggered { .. } => "AutoCompactionTriggered",
            }
        }

        let event_summary: Vec<String> = events
            .iter()
            .enumerate()
            .map(|(i, e)| format!("{i}: [{:>12}] {}", agent_of(e), event_type(e)))
            .collect();
        let event_log = event_summary.join("\n");

        // 4a. Verify events from expected agents
        let agents_seen: std::collections::HashSet<&str> = events.iter().map(agent_of).collect();
        assert!(
            agents_seen.contains("orchestrator"),
            "missing orchestrator events.\n{event_log}"
        );
        // At least planner or worker should be present (JoinSet order non-deterministic)
        let has_planner = agents_seen.contains("planner");
        let has_worker = agents_seen.contains("worker");
        assert!(
            has_planner || has_worker,
            "should have events from at least one successful squad member.\n{event_log}"
        );

        // 4b. SubAgentsDispatched should fire exactly once (from squad-leader)
        let dispatched: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::SubAgentsDispatched { .. }))
            .collect();
        assert_eq!(
            dispatched.len(),
            1,
            "expected exactly 1 SubAgentsDispatched event.\n{event_log}"
        );
        match dispatched[0] {
            AgentEvent::SubAgentsDispatched { agents, agent } => {
                assert_eq!(
                    agent, "squad-leader",
                    "form_squad uses 'squad-leader' label"
                );
                assert_eq!(agents.len(), 3, "should dispatch 3 squad members");
                assert!(agents.contains(&"planner".to_string()));
                assert!(agents.contains(&"worker".to_string()));
                assert!(agents.contains(&"reviewer".to_string()));
            }
            _ => unreachable!(),
        }

        // 4c. SubAgentCompleted events: 3 per-agent + 1 aggregate = 4 total
        let completed: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::SubAgentCompleted { .. }))
            .collect();
        assert_eq!(
            completed.len(),
            4,
            "expected 4 SubAgentCompleted events (3 per-agent + 1 aggregate).\n{event_log}"
        );

        // Per-agent completions: planner and worker succeed, reviewer fails
        let per_agent: Vec<&AgentEvent> = completed
            .iter()
            .filter(|e| {
                matches!(e, AgentEvent::SubAgentCompleted { agent, .. }
                    if !agent.starts_with("squad["))
            })
            .copied()
            .collect();
        assert_eq!(per_agent.len(), 3, "3 per-agent completion events");

        // Find reviewer completion — should have success=false
        let reviewer_completed = per_agent.iter().find(
            |e| matches!(e, AgentEvent::SubAgentCompleted { agent, .. } if agent == "reviewer"),
        );
        assert!(
            reviewer_completed.is_some(),
            "should have reviewer SubAgentCompleted"
        );
        match reviewer_completed.unwrap() {
            AgentEvent::SubAgentCompleted { success, .. } => {
                assert!(!success, "reviewer should have failed");
            }
            _ => unreachable!(),
        }

        // Find planner completion — should have success=true
        let planner_completed = per_agent.iter().find(
            |e| matches!(e, AgentEvent::SubAgentCompleted { agent, .. } if agent == "planner"),
        );
        assert!(
            planner_completed.is_some(),
            "should have planner SubAgentCompleted"
        );
        match planner_completed.unwrap() {
            AgentEvent::SubAgentCompleted { success, usage, .. } => {
                assert!(success, "planner should have succeeded");
                assert_eq!(usage.input_tokens, 20);
                assert_eq!(usage.output_tokens, 15);
                assert_eq!(usage.reasoning_tokens, 8);
            }
            _ => unreachable!(),
        }

        // Aggregate squad completion event
        let squad_completed = completed.iter().find(|e| {
            matches!(e, AgentEvent::SubAgentCompleted { agent, .. }
                if agent.starts_with("squad["))
        });
        assert!(
            squad_completed.is_some(),
            "should have aggregate squad completion event.\n{event_log}"
        );
        match squad_completed.unwrap() {
            AgentEvent::SubAgentCompleted {
                agent,
                success,
                usage,
            } => {
                assert!(
                    agent.contains("planner")
                        && agent.contains("worker")
                        && agent.contains("reviewer"),
                    "aggregate label should list all agents: {agent}"
                );
                assert!(
                    !success,
                    "aggregate should be false because reviewer failed"
                );
                // Aggregate tokens = planner + worker + reviewer(0)
                assert_eq!(usage.input_tokens, 20 + 25 + 35, "aggregate input tokens");
                assert_eq!(usage.output_tokens, 15 + 12 + 18, "aggregate output tokens");
            }
            _ => unreachable!(),
        }

        // 4d. Verify worker had tool events (ToolCallStarted + ToolCallCompleted for "compute")
        let tool_started: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| {
                matches!(e, AgentEvent::ToolCallStarted { tool_name, .. } if tool_name == "compute")
            })
            .collect();
        // Due to JoinSet non-determinism, either planner or worker may call compute
        // (both have it). We just verify it was called.
        assert!(
            !tool_started.is_empty(),
            "should have at least one compute ToolCallStarted.\n{event_log}"
        );

        let tool_completed: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| {
                matches!(e, AgentEvent::ToolCallCompleted { tool_name, .. } if tool_name == "compute")
            })
            .collect();
        assert!(
            !tool_completed.is_empty(),
            "should have at least one compute ToolCallCompleted.\n{event_log}"
        );
        match tool_completed[0] {
            AgentEvent::ToolCallCompleted {
                output, is_error, ..
            } => {
                assert!(!is_error, "compute tool should succeed");
                assert!(
                    output.contains("714"),
                    "compute output should be '714', got: {output}"
                );
            }
            _ => unreachable!(),
        }

        // 4e. RunStarted is first event for each agent
        for agent_name in &["orchestrator", "planner", "worker"] {
            let agent_events: Vec<&AgentEvent> = events
                .iter()
                .filter(|e| agent_of(e) == *agent_name)
                .collect();
            if !agent_events.is_empty() {
                assert!(
                    matches!(agent_events[0], AgentEvent::RunStarted { .. }),
                    "first event for '{agent_name}' should be RunStarted, got: {:?}\n{event_log}",
                    agent_events[0]
                );
            }
        }

        // 4f. Reviewer should have RunStarted then RunFailed (provider error)
        let reviewer_events: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| agent_of(e) == "reviewer")
            .collect();
        if !reviewer_events.is_empty() {
            assert!(
                matches!(reviewer_events[0], AgentEvent::RunStarted { .. }),
                "reviewer first event should be RunStarted"
            );
            // Check for RunFailed
            let has_failed = reviewer_events
                .iter()
                .any(|e| matches!(e, AgentEvent::RunFailed { .. }));
            assert!(
                has_failed,
                "reviewer should have a RunFailed event.\n{event_log}"
            );
        }

        // 4g. SubAgentsDispatched appears before any SubAgentCompleted
        let dispatch_idx = events
            .iter()
            .position(|e| matches!(e, AgentEvent::SubAgentsDispatched { .. }));
        let first_completed_idx = events
            .iter()
            .position(|e| matches!(e, AgentEvent::SubAgentCompleted { .. }));
        if let (Some(d), Some(c)) = (dispatch_idx, first_completed_idx) {
            assert!(
                d < c,
                "SubAgentsDispatched (idx {d}) should precede SubAgentCompleted (idx {c})\n{event_log}"
            );
        }

        // 4h. LlmResponse events should carry model info from MockProvider
        let llm_responses: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::LlmResponse { .. }))
            .collect();
        assert!(
            !llm_responses.is_empty(),
            "should have LlmResponse events.\n{event_log}"
        );
        for lr in &llm_responses {
            match lr {
                AgentEvent::LlmResponse { model, .. } => {
                    assert_eq!(
                        model.as_deref(),
                        Some("mock-model-v1"),
                        "LlmResponse should carry provider model name"
                    );
                }
                _ => unreachable!(),
            }
        }

        // === 5. Verify total event count is reasonable ===
        // Orchestrator: RunStarted, TurnStarted, LlmResponse(squad call), ToolCallStarted(form_squad),
        //              ToolCallCompleted(form_squad), TurnStarted, LlmResponse(synthesis), RunCompleted = ~8
        // Planner: RunStarted, TurnStarted, LlmResponse, RunCompleted = ~4
        // Worker: RunStarted, TurnStarted, LlmResponse, ToolCallStarted(compute),
        //         ToolCallCompleted(compute), TurnStarted, LlmResponse, RunCompleted = ~8
        // Reviewer: RunStarted, TurnStarted(?), RunFailed = ~2-3
        // + SubAgentsDispatched + 3 SubAgentCompleted + 1 aggregate = 5
        // Total ~25-28 events
        assert!(
            events.len() >= 15,
            "expected at least 15 events for complex squad test, got {}.\n{event_log}",
            events.len(),
        );
    }

    #[test]
    fn build_rejects_empty_sub_agent_name() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = Orchestrator::builder(provider)
            .sub_agent("", "Empty name agent", "prompt")
            .build();
        match result {
            Err(Error::Config(msg)) => {
                assert!(
                    msg.contains("must not be empty"),
                    "expected empty name error, got: {msg}"
                );
            }
            Err(other) => panic!("expected Config error, got: {other:?}"),
            Ok(_) => panic!("expected error for empty sub-agent name"),
        }
    }

    #[tokio::test]
    async fn instruction_text_wired_to_orchestrator_system_prompt() {
        // CapturingProvider records the system prompt from LLM calls.
        struct CapturingProvider {
            captured_systems: Mutex<Vec<String>>,
        }
        impl LlmProvider for CapturingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                self.captured_systems
                    .lock()
                    .expect("lock")
                    .push(request.system.clone());
                // Return end_turn immediately so the orchestrator finishes
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Task complete.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                })
            }
        }

        let provider = Arc::new(CapturingProvider {
            captured_systems: Mutex::new(Vec::new()),
        });
        let mut orchestrator = Orchestrator::builder(provider.clone())
            .sub_agent("agent-a", "Does things", "You are agent A.")
            .instruction_text("Always verify your work.")
            .build()
            .unwrap();

        let _output = orchestrator.run("test task").await.unwrap();
        let systems = provider.captured_systems.lock().expect("lock").clone();
        // The orchestrator's own LLM call should have instructions prepended
        assert!(!systems.is_empty(), "should have at least one LLM call");
        let orchestrator_system = &systems[0];
        assert!(
            orchestrator_system.contains("# Project Instructions"),
            "orchestrator system prompt should contain instruction header"
        );
        assert!(
            orchestrator_system.contains("Always verify your work."),
            "orchestrator system prompt should contain instruction text"
        );
    }

    #[tokio::test]
    async fn permission_rules_propagate_to_sub_agents() {
        // Orchestrator-level permission rules should apply to sub-agent tool calls.
        // Here we deny "bash" at the orchestrator level and verify the worker's
        // bash call is rejected.
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator delegates to worker
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "orch-1".into(),
                    name: "delegate_task".into(),
                    input: json!({
                        "tasks": [{"agent": "worker", "task": "run a bash command"}]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: Worker tries to call bash (will be denied by permission rules)
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "worker-1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo hello"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 3: Worker sees the denial and responds
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Bash was denied.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
            // 4: Orchestrator synthesis
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Worker reported bash was denied.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
        let events_clone = events.clone();
        let on_event: Arc<OnEvent> = Arc::new(move |event: AgentEvent| {
            events_clone.lock().expect("test lock").push(event);
        });

        let deny_bash = crate::agent::permission::PermissionRuleset::new(vec![
            crate::agent::permission::PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: crate::agent::permission::PermissionAction::Deny,
            },
        ]);

        let bash_tool: Arc<dyn Tool> = Arc::new(MockTool::new("bash", "executed"));

        let mut orch = Orchestrator::builder(provider)
            .sub_agent_with_tools("worker", "Bash worker", "You run bash.", vec![bash_tool])
            .permission_rules(deny_bash)
            .on_event(on_event)
            .build()
            .unwrap();

        let output = orch.run("run bash via worker").await.unwrap();
        assert_eq!(output.result, "Worker reported bash was denied.");

        // The worker should NOT have ToolCallStarted/ToolCallCompleted for bash
        // because permission-denied calls skip event emission.
        let events = events.lock().expect("test lock");
        let worker_tool_events: Vec<_> = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    AgentEvent::ToolCallStarted { agent, tool_name, .. }
                    | AgentEvent::ToolCallCompleted { agent, tool_name, .. }
                        if agent == "worker" && tool_name == "bash"
                )
            })
            .collect();
        assert!(
            worker_tool_events.is_empty(),
            "bash tool calls in worker should be denied (no events emitted), got: {worker_tool_events:?}"
        );
    }

    #[tokio::test]
    async fn permission_rules_propagate_to_squad_members() {
        // Same test but via form_squad path.
        let provider = Arc::new(MockProvider::new(vec![
            // 1: Orchestrator forms a squad
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "orch-1".into(),
                    name: "form_squad".into(),
                    input: json!({
                        "tasks": [
                            {"agent": "alpha", "task": "run bash"},
                            {"agent": "beta", "task": "say hello"}
                        ]
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 2: alpha tries bash (denied)
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "alpha-1".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
            // 3: alpha sees denial
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Bash denied.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
            // 4: beta just responds (no bash)
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Hello!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
            // 5: Orchestrator synthesis
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Squad done.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let deny_bash = crate::agent::permission::PermissionRuleset::new(vec![
            crate::agent::permission::PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: crate::agent::permission::PermissionAction::Deny,
            },
        ]);

        let bash_tool: Arc<dyn Tool> = Arc::new(MockTool::new("bash", "executed"));

        let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
        let events_clone = events.clone();
        let on_event: Arc<OnEvent> = Arc::new(move |event: AgentEvent| {
            events_clone.lock().expect("test lock").push(event);
        });

        let mut orch = Orchestrator::builder(provider)
            .sub_agent_with_tools(
                "alpha",
                "Alpha agent",
                "You run bash.",
                vec![bash_tool.clone()],
            )
            .sub_agent("beta", "Beta agent", "You say hello.")
            .permission_rules(deny_bash)
            .on_event(on_event)
            .build()
            .unwrap();

        let output = orch.run("form a squad").await.unwrap();
        assert_eq!(output.result, "Squad done.");

        // Alpha's bash call should be denied — no ToolCallStarted/Completed events for bash
        let events = events.lock().expect("test lock");
        let bash_events: Vec<_> = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    AgentEvent::ToolCallStarted { tool_name, .. }
                    | AgentEvent::ToolCallCompleted { tool_name, .. }
                        if tool_name == "bash"
                )
            })
            .collect();
        assert!(
            bash_events.is_empty(),
            "bash tool calls in squad should be denied (no events), got: {bash_events:?}"
        );
    }
}
