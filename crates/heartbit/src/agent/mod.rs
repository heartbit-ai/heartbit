pub mod context;
pub mod orchestrator;

use std::collections::HashMap;
use std::sync::Arc;

use tracing::debug;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{Message, StopReason, TokenUsage, ToolCall, ToolDefinition, ToolResult};
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
    tools: HashMap<String, Arc<dyn Tool>>,
    tool_defs: Vec<ToolDefinition>,
    max_turns: usize,
    max_tokens: u32,
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
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let mut ctx = AgentContext::new(&self.system_prompt, task, self.tool_defs.clone())
            .with_max_turns(self.max_turns)
            .with_max_tokens(self.max_tokens);

        let mut total_tool_calls = 0usize;
        let mut total_usage = TokenUsage::default();

        loop {
            if ctx.current_turn() >= ctx.max_turns() {
                return Err(Error::MaxTurnsExceeded(ctx.max_turns()));
            }

            ctx.increment_turn();
            debug!(agent = %self.name, turn = ctx.current_turn(), "executing turn");
            let response = self.provider.complete(ctx.to_request()).await?;
            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            let tool_calls = response.tool_calls();

            // Add assistant message to context (move content, avoid clone)
            ctx.add_assistant_message(Message {
                role: crate::llm::types::Role::Assistant,
                content: response.content,
            });

            if tool_calls.is_empty() {
                // Check for truncation
                if response.stop_reason == StopReason::MaxTokens {
                    return Err(Error::Truncated);
                }

                return Ok(AgentOutput {
                    result: ctx.last_assistant_text().unwrap_or_default().to_string(),
                    tool_calls_made: total_tool_calls,
                    tokens_used: total_usage,
                });
            }

            total_tool_calls += tool_calls.len();
            let results = self.execute_tools_parallel(&tool_calls).await;
            ctx.add_tool_results(results);
        }
    }

    /// Execute tools in parallel via JoinSet, returning results in original call order.
    async fn execute_tools_parallel(&self, calls: &[ToolCall]) -> Vec<ToolResult> {
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, call) in calls.iter().enumerate() {
            let tool = self.tools.get(&call.name).cloned();
            let input = call.input.clone();
            let call_id = call.id.clone();
            let call_name = call.name.clone();

            join_set.spawn(async move {
                let output = match tool {
                    Some(t) => t.execute(input).await,
                    None => Ok(ToolOutput::error(format!("Tool not found: {call_name}"))),
                };
                (idx, call_id, output)
            });
        }

        let mut results: Vec<(usize, ToolResult)> = Vec::with_capacity(calls.len());
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((idx, call_id, Ok(output))) => {
                    results.push((idx, tool_output_to_result(call_id, output)));
                }
                Ok((idx, call_id, Err(e))) => {
                    results.push((idx, ToolResult::error(call_id, e.to_string())));
                }
                Err(join_err) => {
                    tracing::error!(error = %join_err, "tool task panicked");
                }
            }
        }

        // Sort by original index to preserve call order
        results.sort_by_key(|(idx, _)| *idx);
        results.into_iter().map(|(_, r)| r).collect()
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

    pub fn build(self) -> AgentRunner<P> {
        let tool_defs: Vec<ToolDefinition> = self.tools.iter().map(|t| t.definition()).collect();
        let tools: HashMap<String, Arc<dyn Tool>> = self
            .tools
            .into_iter()
            .map(|t| (t.definition().name, t))
            .collect();

        AgentRunner {
            provider: self.provider,
            name: self.name,
            system_prompt: self.system_prompt,
            tools,
            tool_defs,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
        }
    }
}

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
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Sorry about that.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build();

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

    #[tokio::test]
    async fn agent_errors_on_max_tokens() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "truncated...".into(),
            }],
            stop_reason: StopReason::MaxTokens,
            usage: TokenUsage::default(),
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build();

        let err = runner.execute("write a long essay").await.unwrap_err();
        assert!(matches!(err, Error::Truncated));
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
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Tool failed, but I recovered.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::failing("failing", "something broke")))
            .build();

        let output = runner.execute("try the tool").await.unwrap();
        assert_eq!(output.result, "Tool failed, but I recovered.");
    }

    #[tokio::test]
    async fn max_tokens_is_configurable() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text { text: "ok".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .max_tokens(8192)
            .build();

        // Just verify it builds and runs without error
        let output = runner.execute("test").await.unwrap();
        assert_eq!(output.result, "ok");
    }
}
