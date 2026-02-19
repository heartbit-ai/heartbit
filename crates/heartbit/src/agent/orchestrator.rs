use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{TokenUsage, ToolDefinition};
use crate::tool::{Tool, ToolOutput};

use crate::memory::Memory;

use super::blackboard::Blackboard;
use super::blackboard_tools::blackboard_tools;
use super::context::ContextStrategy;
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
            shared_memory: None,
            blackboard: None,
            on_text: None,
            on_approval: None,
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
        let mut output = self.runner.execute(task).await?;
        // Add sub-agent tokens that were accumulated during delegation
        let sub_tokens = self.sub_agent_tokens.lock().expect("token lock poisoned");
        output.tokens_used.input_tokens += sub_tokens.input_tokens;
        output.tokens_used.output_tokens += sub_tokens.output_tokens;
        Ok(output)
    }
}

/// The orchestrator's primary tool: delegates tasks to sub-agents in parallel.
///
/// Implements `Tool` so it can be registered with `AgentRunner`.
/// Unknown agent names return an error result to the LLM instead of crashing.
struct DelegateTaskTool<P: LlmProvider> {
    provider: Arc<P>,
    sub_agents: Vec<SubAgentDef>,
    max_turns: usize,
    max_tokens: u32,
    /// Shared accumulator for sub-agent token usage, read by Orchestrator::run.
    accumulated_tokens: Arc<Mutex<TokenUsage>>,
    /// Shared memory store for cross-agent memory (None if not configured).
    shared_memory: Option<Arc<dyn Memory>>,
    /// Shared blackboard for cross-agent coordination (None if not configured).
    blackboard: Option<Arc<dyn Blackboard>>,
}

impl<P: LlmProvider + 'static> DelegateTaskTool<P> {
    async fn delegate(&self, tasks: Vec<DelegatedTask>) -> Result<String, Error> {
        let task_count = tasks.len();
        let agent_names: Vec<String> = tasks.iter().map(|t| t.agent.clone()).collect();
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
                            },
                        )
                    });
                    continue;
                }
            };

            let provider = self.provider.clone();
            let max_turns = agent_def.max_turns.unwrap_or(self.max_turns);
            let max_tokens = agent_def.max_tokens.unwrap_or(self.max_tokens);
            let shared_memory = self.shared_memory.clone();
            let blackboard = self.blackboard.clone();

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

                let runner = match builder.build() {
                    Ok(r) => r,
                    Err(e) => {
                        return (
                            idx,
                            SubAgentResult {
                                agent: agent_def.name,
                                result: format!("Error building agent: {e}"),
                                tokens_used: TokenUsage::default(),
                            },
                        );
                    }
                };

                let result = match runner.execute(&task.task).await {
                    Ok(output) => SubAgentResult {
                        agent: agent_def.name,
                        result: output.result,
                        tokens_used: output.tokens_used,
                    },
                    Err(e) => SubAgentResult {
                        agent: agent_def.name,
                        result: format!("Error: {e}"),
                        tokens_used: TokenUsage::default(),
                    },
                };

                // Write agent result to blackboard (matching Restate path's "agent:{name}" key)
                if let Some(ref bb) = blackboard {
                    let key = format!("agent:{}", result.agent);
                    if let Err(e) = bb
                        .write(&key, serde_json::Value::String(result.result.clone()))
                        .await
                    {
                        tracing::warn!(
                            agent = %result.agent,
                            error = %e,
                            "failed to write result to blackboard"
                        );
                    }
                }

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
                        },
                    )
                })
            })
            .collect();
        results.sort_by_key(|(idx, _)| *idx);

        // Accumulate sub-agent tokens
        {
            let mut acc = self.accumulated_tokens.lock().expect("token lock poisoned");
            for (_, r) in &results {
                acc.input_tokens += r.tokens_used.input_tokens;
                acc.output_tokens += r.tokens_used.output_tokens;
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

impl<P: LlmProvider + 'static> Tool for DelegateTaskTool<P> {
    fn definition(&self) -> ToolDefinition {
        let pairs: Vec<(&str, &str)> = self
            .sub_agents
            .iter()
            .map(|a| (a.name.as_str(), a.description.as_str()))
            .collect();
        build_delegate_tool_schema(&pairs)
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
    #[serde(default)]
    tasks: Vec<DelegatedTask>,
}

/// Build the orchestrator system prompt listing available agents.
///
/// Shared between standalone and Restate paths. Takes `(name, description)` pairs.
pub(crate) fn build_system_prompt(agents: &[(&str, &str)]) -> String {
    let agent_list: String = agents
        .iter()
        .map(|(name, desc)| format!("- **{name}**: {desc}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are an orchestrator agent. Your job is to break down complex tasks and \
         delegate them to specialized sub-agents.\n\n\
         Available sub-agents:\n{agent_list}\n\n\
         Use the delegate_task tool to assign work to sub-agents. You can assign \
         multiple tasks at once for parallel execution. After receiving results, \
         synthesize them into a coherent response."
    )
}

/// Build the delegate_task tool definition.
///
/// Shared between standalone and Restate paths. Takes `(name, description)` pairs.
pub(crate) fn build_delegate_tool_schema(agents: &[(&str, &str)]) -> ToolDefinition {
    let agent_descriptions: Vec<serde_json::Value> = agents
        .iter()
        .map(|(name, desc)| json!({"name": name, "description": desc}))
        .collect();

    ToolDefinition {
        name: "delegate_task".into(),
        description: format!(
            "Delegate tasks to sub-agents for parallel execution. Available agents: {}",
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
                    }
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
}

pub struct OrchestratorBuilder<P: LlmProvider> {
    provider: Arc<P>,
    sub_agents: Vec<SubAgentDef>,
    max_turns: usize,
    max_tokens: u32,
    shared_memory: Option<Arc<dyn Memory>>,
    blackboard: Option<Arc<dyn Blackboard>>,
    on_text: Option<Arc<crate::llm::OnText>>,
    on_approval: Option<Arc<crate::llm::OnApproval>>,
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

    pub fn build(self) -> Result<Orchestrator<P>, Error> {
        if self.sub_agents.is_empty() {
            tracing::warn!(
                "orchestrator built with no sub-agents — delegate_task tool will list no agents"
            );
        }

        let pairs: Vec<(&str, &str)> = self
            .sub_agents
            .iter()
            .map(|a| (a.name.as_str(), a.description.as_str()))
            .collect();
        let system = build_system_prompt(&pairs);

        let sub_agent_tokens = Arc::new(Mutex::new(TokenUsage::default()));

        let delegate_tool: Arc<dyn Tool> = Arc::new(DelegateTaskTool {
            provider: self.provider.clone(),
            sub_agents: self.sub_agents,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            accumulated_tokens: sub_agent_tokens.clone(),
            shared_memory: self.shared_memory,
            blackboard: self.blackboard,
        });

        let mut runner_builder = AgentRunner::builder(self.provider)
            .name("orchestrator")
            .system_prompt(system)
            .tool(delegate_tool)
            .max_turns(self.max_turns)
            .max_tokens(self.max_tokens);

        if let Some(on_text) = self.on_text {
            runner_builder = runner_builder.on_text(on_text);
        }
        if let Some(on_approval) = self.on_approval {
            runner_builder = runner_builder.on_approval(on_approval);
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
    }

    #[test]
    fn system_prompt_includes_agents() {
        let agents = vec![
            ("researcher", "Research specialist"),
            ("coder", "Coding expert"),
        ];

        let prompt = build_system_prompt(&agents);
        assert!(prompt.contains("researcher"));
        assert!(prompt.contains("Research specialist"));
        assert!(prompt.contains("coder"));
    }

    #[test]
    fn delegate_tool_schema_includes_agents() {
        let agents = vec![("researcher", "Research")];
        let def = build_delegate_tool_schema(&agents);
        assert_eq!(def.name, "delegate_task");
        assert!(def.description.contains("researcher"));
    }

    #[test]
    fn delegate_tool_definition_includes_agents() {
        let tool: DelegateTaskTool<MockProvider> = DelegateTaskTool {
            provider: Arc::new(MockProvider::new(vec![])),
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
            }],
            shared_memory: None,
            blackboard: None,
            max_turns: 10,
            max_tokens: 4096,
            accumulated_tokens: Arc::new(Mutex::new(TokenUsage::default())),
        };

        let def = tool.definition();
        assert_eq!(def.name, "delegate_task");
        assert!(def.description.contains("researcher"));
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
}
