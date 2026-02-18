pub mod context;
pub mod orchestrator;

use std::sync::Arc;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{Message, TokenUsage, ToolCall, ToolDefinition, ToolResult};
use crate::tool::{Tool, ToolOutput};

use self::context::AgentContext;

/// Output of an agent run.
#[derive(Debug, Clone)]
pub struct AgentOutput {
    pub result: String,
    pub tool_calls_made: usize,
    pub tokens_used: TokenUsage,
}

/// Runs an agent loop: LLM call → tool execution → repeat until done.
pub struct AgentRunner<P: LlmProvider> {
    provider: Arc<P>,
    name: String,
    system_prompt: String,
    tools: Vec<Arc<dyn Tool>>,
    model: String,
    max_turns: usize,
}

impl<P: LlmProvider> AgentRunner<P> {
    pub fn builder(provider: Arc<P>) -> AgentRunnerBuilder<P> {
        AgentRunnerBuilder {
            provider,
            name: String::new(),
            system_prompt: String::new(),
            tools: vec![],
            model: "claude-sonnet-4-20250514".into(),
            max_turns: 10,
        }
    }

    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let tool_defs: Vec<ToolDefinition> = self.tools.iter().map(|t| t.definition()).collect();
        let mut ctx = AgentContext::new(&self.system_prompt, task, tool_defs, &self.model)
            .with_max_turns(self.max_turns);

        let mut total_tool_calls = 0usize;
        let mut total_usage = TokenUsage::default();

        loop {
            if ctx.current_turn() >= ctx.max_turns() {
                return Err(Error::MaxTurnsExceeded(ctx.max_turns()));
            }

            ctx.increment_turn();
            tracing::debug!(agent = %self.name, turn = ctx.current_turn(), "executing turn");
            let response = self.provider.complete(ctx.to_request()).await?;
            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            let tool_calls = response.tool_calls();

            // Add assistant message to context
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

            total_tool_calls += tool_calls.len();
            let results = self.execute_tools_parallel(&tool_calls).await?;
            ctx.add_tool_results(results);
        }
    }

    async fn execute_tools_parallel(&self, calls: &[ToolCall]) -> Result<Vec<ToolResult>, Error> {
        let mut join_set = tokio::task::JoinSet::new();

        for call in calls {
            let tool = self.find_tool(&call.name)?;
            let input = call.input.clone();
            let call_id = call.id.clone();

            join_set.spawn(async move {
                let output = tool.execute(input).await;
                (call_id, output)
            });
        }

        let mut results = Vec::with_capacity(calls.len());
        while let Some(result) = join_set.join_next().await {
            let (call_id, output) = result.map_err(|e| Error::Agent(e.to_string()))?;
            let output = output?;
            results.push(tool_output_to_result(call_id, output));
        }
        Ok(results)
    }

    fn find_tool(&self, name: &str) -> Result<Arc<dyn Tool>, Error> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .cloned()
            .ok_or_else(|| Error::ToolNotFound(name.to_string()))
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
    model: String,
    max_turns: usize,
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

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn build(self) -> AgentRunner<P> {
        AgentRunner {
            provider: self.provider,
            name: self.name,
            system_prompt: self.system_prompt,
            tools: self.tools,
            model: self.model,
            max_turns: self.max_turns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{CompletionResponse, ContentBlock, StopReason};
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
        async fn complete(
            &self,
            _request: crate::llm::types::CompletionRequest,
        ) -> Result<CompletionResponse, Error> {
            let mut responses = self.responses.lock().unwrap();
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
            Box::pin(async move { Ok(ToolOutput::success(response)) })
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
            },
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("You are helpful.")
            .build();

        let output = runner.execute("say hello").await.unwrap();
        assert_eq!(output.result, "Hello!");
        assert_eq!(output.tool_calls_made, 0);
        assert_eq!(output.tokens_used.input_tokens, 10);
    }

    #[tokio::test]
    async fn agent_executes_tool_and_continues() {
        let provider = Arc::new(MockProvider::new(vec![
            // First response: tool call
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
                },
            },
            // Second response: final text
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Found it!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 30,
                    output_tokens: 15,
                },
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("You are helpful.")
            .tool(Arc::new(MockTool::new("search", "search results here")))
            .build();

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
            },
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: TokenUsage::default(),
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .max_turns(2)
            .build();

        let err = runner.execute("loop forever").await.unwrap_err();
        assert!(matches!(err, Error::MaxTurnsExceeded(2)));
    }

    #[tokio::test]
    async fn agent_errors_on_unknown_tool() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "c1".into(),
                name: "nonexistent".into(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build();

        let err = runner.execute("use unknown tool").await.unwrap_err();
        assert!(matches!(err, Error::ToolNotFound(_)));
    }

    #[tokio::test]
    async fn agent_executes_parallel_tool_calls() {
        let provider = Arc::new(MockProvider::new(vec![
            // Two tool calls in one response
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
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "found")))
            .tool(Arc::new(MockTool::new("read", "file content")))
            .build();

        let output = runner.execute("do both").await.unwrap();
        assert_eq!(output.result, "Done!");
        assert_eq!(output.tool_calls_made, 2);
    }
}
