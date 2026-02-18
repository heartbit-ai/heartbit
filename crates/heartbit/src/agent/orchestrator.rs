use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{TokenUsage, ToolDefinition};
use crate::tool::{Tool, ToolOutput};

use super::{AgentOutput, AgentRunner};

/// A sub-agent definition registered with the orchestrator.
#[derive(Clone)]
pub struct SubAgentDef {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub tools: Vec<Arc<dyn Tool>>,
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
/// Refactored to use `AgentRunner` internally with a `DelegateTaskTool`.
/// No duplicated agent loop — the orchestrator IS an AgentRunner.
pub struct Orchestrator<P: LlmProvider> {
    runner: AgentRunner<P>,
}

impl<P: LlmProvider + 'static> Orchestrator<P> {
    pub fn builder(provider: Arc<P>) -> OrchestratorBuilder<P> {
        OrchestratorBuilder {
            provider,
            sub_agents: vec![],
            max_turns: 10,
            max_tokens: 4096,
        }
    }

    pub async fn run(&self, task: &str) -> Result<AgentOutput, Error> {
        self.runner.execute(task).await
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
}

impl<P: LlmProvider + 'static> DelegateTaskTool<P> {
    async fn delegate(&self, tasks: Vec<DelegatedTask>) -> Result<String, Error> {
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
            let max_turns = self.max_turns;
            let max_tokens = self.max_tokens;

            info!(agent = %agent_def.name, task = %task.task, "spawning sub-agent");

            join_set.spawn(async move {
                let runner = AgentRunner::builder(provider)
                    .name(&agent_def.name)
                    .system_prompt(&agent_def.system_prompt)
                    .tools(agent_def.tools)
                    .max_turns(max_turns)
                    .max_tokens(max_tokens)
                    .build();

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

                (idx, result)
            });
        }

        let mut results: Vec<(usize, SubAgentResult)> = Vec::new();
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(indexed_result) => results.push(indexed_result),
                Err(e) => {
                    tracing::error!(error = %e, "sub-agent task panicked");
                }
            }
        }

        // Sort by original index to preserve order
        results.sort_by_key(|(idx, _)| *idx);

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
        let agent_descriptions: Vec<serde_json::Value> = self
            .sub_agents
            .iter()
            .map(|a| json!({"name": a.name, "description": a.description}))
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

fn build_system_prompt(sub_agents: &[SubAgentDef]) -> String {
    let agent_list: String = sub_agents
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

pub struct OrchestratorBuilder<P: LlmProvider> {
    provider: Arc<P>,
    sub_agents: Vec<SubAgentDef>,
    max_turns: usize,
    max_tokens: u32,
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

    pub fn build(self) -> Orchestrator<P> {
        let system = build_system_prompt(&self.sub_agents);

        let delegate_tool: Arc<dyn Tool> = Arc::new(DelegateTaskTool {
            provider: self.provider.clone(),
            sub_agents: self.sub_agents,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
        });

        let runner = AgentRunner::builder(self.provider)
            .name("orchestrator")
            .system_prompt(system)
            .tool(delegate_tool)
            .max_turns(self.max_turns)
            .max_tokens(self.max_tokens)
            .build();

        Orchestrator { runner }
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
            SubAgentDef {
                name: "researcher".into(),
                description: "Research specialist".into(),
                system_prompt: "You research.".into(),
                tools: vec![],
            },
            SubAgentDef {
                name: "coder".into(),
                description: "Coding expert".into(),
                system_prompt: "You code.".into(),
                tools: vec![],
            },
        ];

        let prompt = build_system_prompt(&agents);
        assert!(prompt.contains("researcher"));
        assert!(prompt.contains("Research specialist"));
        assert!(prompt.contains("coder"));
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
            }],
            max_turns: 10,
            max_tokens: 4096,
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

        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build();

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

        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research specialist", "You research.")
            .sub_agent("analyst", "Analysis expert", "You analyze.")
            .build();

        let output = orch.run("Analyze Rust").await.unwrap();
        assert_eq!(output.result, "Based on research: Rust is excellent.");
        assert_eq!(output.tool_calls_made, 1); // one delegate_task call
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

        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build();

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

        let orch = Orchestrator::builder(provider)
            .sub_agent("researcher", "Research", "prompt")
            .build();

        let output = orch.run("do something").await.unwrap();
        assert_eq!(output.result, "Sorry, let me respond directly.");
    }
}
