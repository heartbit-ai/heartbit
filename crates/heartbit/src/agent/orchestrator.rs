use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{Message, TokenUsage, ToolDefinition, ToolResult};

use super::context::AgentContext;
use super::{AgentOutput, AgentRunner};

/// A sub-agent definition registered with the orchestrator.
#[derive(Debug, Clone)]
pub struct SubAgentDef {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
}

/// A task delegated by the orchestrator to a sub-agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegatedTask {
    pub agent: String,
    pub task: String,
}

/// Result from a sub-agent execution.
#[derive(Debug, Clone)]
pub struct SubAgentResult {
    pub agent: String,
    pub result: String,
    pub tokens_used: TokenUsage,
}

/// Multi-agent orchestrator.
///
/// The orchestrator is itself an agent whose primary tool is `delegate_task`.
/// It asks the LLM which sub-agents to spawn and with what instructions,
/// then runs them in parallel and aggregates results.
pub struct Orchestrator<P: LlmProvider> {
    provider: Arc<P>,
    sub_agents: Vec<SubAgentDef>,
    model: String,
    max_turns: usize,
}

impl<P: LlmProvider + 'static> Orchestrator<P> {
    pub fn builder(provider: Arc<P>) -> OrchestratorBuilder<P> {
        OrchestratorBuilder {
            provider,
            sub_agents: vec![],
            model: "claude-sonnet-4-20250514".into(),
            max_turns: 10,
        }
    }

    pub async fn run(&self, task: &str) -> Result<AgentOutput, Error> {
        let system = self.build_system_prompt();
        let tools = vec![self.delegate_tool_definition()];
        let mut ctx =
            AgentContext::new(&system, task, tools, &self.model).with_max_turns(self.max_turns);

        let mut total_tool_calls = 0usize;
        let mut total_usage = TokenUsage::default();

        loop {
            if ctx.current_turn() >= ctx.max_turns() {
                return Err(Error::MaxTurnsExceeded(ctx.max_turns()));
            }

            ctx.increment_turn();
            let response = self.provider.complete(ctx.to_request()).await?;
            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            let tool_calls = response.tool_calls();

            ctx.add_assistant_message(Message {
                role: crate::llm::types::Role::Assistant,
                content: response.content.clone(),
            });

            if tool_calls.is_empty() {
                return Ok(AgentOutput {
                    result: response.text(),
                    tool_calls_made: total_tool_calls,
                    tokens_used: total_usage,
                });
            }

            // Process each tool call (should be delegate_task calls)
            let mut tool_results = Vec::with_capacity(tool_calls.len());

            for call in &tool_calls {
                if call.name != "delegate_task" {
                    tool_results.push(ToolResult::error(
                        &call.id,
                        format!(
                            "Unknown tool: {}. Only delegate_task is available.",
                            call.name
                        ),
                    ));
                    continue;
                }

                total_tool_calls += 1;

                match serde_json::from_value::<DelegateInput>(call.input.clone()) {
                    Ok(input) => {
                        let results = self.delegate(input.tasks).await?;
                        let aggregated = self.format_results(&results);

                        // Accumulate sub-agent token usage
                        for r in &results {
                            total_usage.input_tokens += r.tokens_used.input_tokens;
                            total_usage.output_tokens += r.tokens_used.output_tokens;
                        }

                        tool_results.push(ToolResult::success(&call.id, aggregated));
                    }
                    Err(e) => {
                        tool_results.push(ToolResult::error(
                            &call.id,
                            format!("Invalid delegate_task input: {e}"),
                        ));
                    }
                }
            }

            ctx.add_tool_results(tool_results);
        }
    }

    async fn delegate(&self, tasks: Vec<DelegatedTask>) -> Result<Vec<SubAgentResult>, Error> {
        let mut join_set = tokio::task::JoinSet::new();

        for task in tasks {
            let provider = self.provider.clone();
            let agent_def = self
                .sub_agents
                .iter()
                .find(|a| a.name == task.agent)
                .ok_or_else(|| Error::Agent(format!("Unknown sub-agent: {}", task.agent)))?
                .clone();
            let model = self.model.clone();

            info!(agent = %agent_def.name, task = %task.task, "spawning sub-agent");

            join_set.spawn(async move {
                let runner = AgentRunner::builder(provider)
                    .name(&agent_def.name)
                    .system_prompt(&agent_def.system_prompt)
                    .model(&model)
                    .build();

                let output = runner.execute(&task.task).await?;
                Ok::<_, Error>(SubAgentResult {
                    agent: agent_def.name,
                    result: output.result,
                    tokens_used: output.tokens_used,
                })
            });
        }

        let mut results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            results.push(result.map_err(|e| Error::Agent(e.to_string()))??);
        }
        Ok(results)
    }

    fn delegate_tool_definition(&self) -> ToolDefinition {
        let agent_descriptions: Vec<serde_json::Value> = self
            .sub_agents
            .iter()
            .map(|a| {
                json!({
                    "name": a.name,
                    "description": a.description,
                })
            })
            .collect();

        ToolDefinition {
            name: "delegate_task".into(),
            description: format!(
                "Delegate tasks to sub-agents for parallel execution. Available agents: {}",
                serde_json::to_string(&agent_descriptions).unwrap_or_default()
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
                                    "description": "Name of the sub-agent to delegate to"
                                },
                                "task": {
                                    "type": "string",
                                    "description": "The task instruction for the sub-agent"
                                }
                            },
                            "required": ["agent", "task"]
                        },
                        "description": "List of tasks to delegate to sub-agents"
                    }
                },
                "required": ["tasks"]
            }),
        }
    }

    fn build_system_prompt(&self) -> String {
        let agent_list: String = self
            .sub_agents
            .iter()
            .map(|a| format!("- **{}**: {}", a.name, a.description))
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

    fn format_results(&self, results: &[SubAgentResult]) -> String {
        results
            .iter()
            .map(|r| format!("=== Agent: {} ===\n{}", r.agent, r.result))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[derive(Deserialize)]
struct DelegateInput {
    tasks: Vec<DelegatedTask>,
}

pub struct OrchestratorBuilder<P: LlmProvider> {
    provider: Arc<P>,
    sub_agents: Vec<SubAgentDef>,
    model: String,
    max_turns: usize,
}

impl<P: LlmProvider> OrchestratorBuilder<P> {
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
        });
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn build(self) -> Orchestrator<P> {
        Orchestrator {
            provider: self.provider,
            sub_agents: self.sub_agents,
            model: self.model,
            max_turns: self.max_turns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{CompletionRequest, CompletionResponse, ContentBlock, StopReason};
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
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Err(Error::Agent("no more mock responses".into()));
            }
            Ok(responses.remove(0))
        }
    }

    #[test]
    fn orchestrator_builds_system_prompt_with_agents() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research specialist", "You research things.")
            .sub_agent("coder", "Coding expert", "You write code.")
            .build();

        let prompt = orch.build_system_prompt();
        assert!(prompt.contains("researcher"));
        assert!(prompt.contains("Research specialist"));
        assert!(prompt.contains("coder"));
        assert!(prompt.contains("Coding expert"));
    }

    #[test]
    fn delegate_tool_definition_includes_agents() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build();

        let tool = orch.delegate_tool_definition();
        assert_eq!(tool.name, "delegate_task");
        assert!(tool.description.contains("researcher"));
    }

    #[test]
    fn format_results_aggregates() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let orch = Orchestrator::builder(provider).build();

        let results = vec![
            SubAgentResult {
                agent: "researcher".into(),
                result: "Found X".into(),
                tokens_used: TokenUsage::default(),
            },
            SubAgentResult {
                agent: "coder".into(),
                result: "Wrote Y".into(),
                tokens_used: TokenUsage::default(),
            },
        ];

        let formatted = orch.format_results(&results);
        assert!(formatted.contains("=== Agent: researcher ==="));
        assert!(formatted.contains("Found X"));
        assert!(formatted.contains("=== Agent: coder ==="));
        assert!(formatted.contains("Wrote Y"));
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

        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build();

        let output = orch.run("simple question").await.unwrap();
        assert_eq!(output.result, "Simple answer.");
        assert_eq!(output.tool_calls_made, 0);
    }

    #[tokio::test]
    async fn orchestrator_delegates_and_synthesizes() {
        let provider = Arc::new(MockProvider::new(vec![
            // Orchestrator's first response: delegate to sub-agents
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
            // Sub-agent "researcher" response
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
            // Sub-agent "analyst" response
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
            // Orchestrator's final synthesis
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

        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research specialist", "You research.")
            .sub_agent("analyst", "Analysis expert", "You analyze.")
            .build();

        let output = orch.run("Analyze Rust").await.unwrap();
        assert_eq!(output.result, "Based on research: Rust is excellent.");
        assert_eq!(output.tool_calls_made, 1); // one delegate_task call
    }

    #[tokio::test]
    async fn orchestrator_handles_unknown_agent() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "delegate_task".into(),
                input: json!({
                    "tasks": [{"agent": "nonexistent", "task": "do stuff"}]
                }),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
        }]));

        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build();

        let err = orch.run("delegate to unknown").await.unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
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
            // After error, orchestrator responds directly
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Sorry, let me respond directly.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build();

        let output = orch.run("do something").await.unwrap();
        assert_eq!(output.result, "Sorry, let me respond directly.");
    }
}
