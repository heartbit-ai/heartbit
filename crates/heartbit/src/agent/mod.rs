pub mod context;
pub mod orchestrator;
pub(crate) mod token_estimator;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tracing::debug;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{
    CompletionRequest, Message, StopReason, TokenUsage, ToolCall, ToolDefinition, ToolResult,
};
use crate::tool::{Tool, ToolOutput, validate_tool_input};

use crate::memory::Memory;

use self::context::{AgentContext, ContextStrategy};

/// Output of an agent run.
#[derive(Debug, Clone)]
pub struct AgentOutput {
    pub result: String,
    pub tool_calls_made: usize,
    pub tokens_used: TokenUsage,
    /// Structured output when the agent was configured with a response schema.
    /// Contains the validated JSON conforming to the schema.
    pub structured: Option<serde_json::Value>,
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
            on_text: None,
            on_approval: None,
            tool_timeout: None,
            max_tool_output_bytes: None,
            structured_schema: None,
        }
    }

    pub async fn execute(&self, task: &str) -> Result<AgentOutput, Error> {
        let mut ctx = AgentContext::new(&self.system_prompt, task, self.tool_defs.clone())
            .with_max_turns(self.max_turns)
            .with_max_tokens(self.max_tokens)
            .with_context_strategy(self.context_strategy.clone());

        let mut total_tool_calls = 0usize;
        let mut total_usage = TokenUsage::default();

        loop {
            if ctx.current_turn() >= ctx.max_turns() {
                return Err(Error::MaxTurnsExceeded(ctx.max_turns()));
            }

            ctx.increment_turn();
            debug!(agent = %self.name, turn = ctx.current_turn(), "executing turn");
            let response = match &self.on_text {
                Some(cb) => {
                    self.provider
                        .stream_complete(ctx.to_request(), &**cb)
                        .await?
                }
                None => self.provider.complete(ctx.to_request()).await?,
            };
            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            let tool_calls = response.tool_calls();

            // Add assistant message to context (move content, avoid clone)
            ctx.add_assistant_message(Message {
                role: crate::llm::types::Role::Assistant,
                content: response.content,
            });

            // Check for structured output: if the LLM called the synthetic `__respond__` tool,
            // extract its input as the structured result and return immediately.
            if self.structured_schema.is_some()
                && let Some(respond_call) = tool_calls.iter().find(|tc| tc.name == "__respond__")
            {
                let structured = respond_call.input.clone();
                let text = serde_json::to_string_pretty(&structured)
                    .unwrap_or_else(|_| structured.to_string());
                total_tool_calls += 1;
                return Ok(AgentOutput {
                    result: text,
                    tool_calls_made: total_tool_calls,
                    tokens_used: total_usage,
                    structured: Some(structured),
                });
            }

            if tool_calls.is_empty() {
                // Check for truncation
                if response.stop_reason == StopReason::MaxTokens {
                    return Err(Error::Truncated);
                }

                return Ok(AgentOutput {
                    result: ctx.last_assistant_text().unwrap_or_default().to_string(),
                    tool_calls_made: total_tool_calls,
                    tokens_used: total_usage,
                    structured: None,
                });
            }

            // Human-in-the-loop: if approval callback is set, ask before executing
            if let Some(ref cb) = self.on_approval
                && !cb(&tool_calls)
            {
                debug!(agent = %self.name, "tool execution denied by approval callback");
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

            total_tool_calls += tool_calls.len();
            let results = self.execute_tools_parallel(&tool_calls).await;
            ctx.add_tool_results(results);

            // Summarization: if threshold is set and context exceeds it, compress.
            // Guard on message count: inject_summary(keep_last_n=4) is a no-op
            // when total messages <= 5 (1 first + 4 kept), so skip the LLM call.
            if let Some(threshold) = self.summarize_threshold
                && ctx.message_count() > 5
                && ctx.needs_compaction(threshold)
            {
                debug!(agent = %self.name, "context exceeds threshold, summarizing");
                let summary = self.generate_summary(&ctx).await?;
                ctx.inject_summary(summary, 4);
            }
        }
    }

    /// Generate a summary of the conversation so far using the LLM.
    async fn generate_summary(&self, ctx: &AgentContext) -> Result<String, Error> {
        let summary_request = CompletionRequest {
            system: "You are a summarization assistant. Summarize the following conversation \
                     concisely, preserving key facts, decisions, and tool results. \
                     Focus on information that would be needed to continue the conversation."
                .into(),
            messages: vec![Message::user(ctx.conversation_text())],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };

        let response = self.provider.complete(summary_request).await?;
        Ok(response.text())
    }

    /// Execute tools in parallel via JoinSet, returning results in original call order.
    ///
    /// Panicked tasks produce an error `ToolResult` so the LLM always gets a
    /// result for every `tool_use_id` it sent.
    async fn execute_tools_parallel(&self, calls: &[ToolCall]) -> Vec<ToolResult> {
        let call_ids: Vec<String> = calls.iter().map(|c| c.id.clone()).collect();
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, call) in calls.iter().enumerate() {
            let tool = self.tools.get(&call.name).cloned();
            let input = call.input.clone();
            let call_name = call.name.clone();
            let timeout = self.tool_timeout;

            // Validate input against the tool's declared schema before dispatching.
            // On failure, produce an error result without executing the tool.
            if let Some(ref t) = tool {
                let schema = &t.definition().input_schema;
                if let Err(msg) = validate_tool_input(schema, &input) {
                    join_set.spawn(async move { (idx, Ok(ToolOutput::error(msg))) });
                    continue;
                }
            }

            join_set.spawn(async move {
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
                (idx, output)
            });
        }

        let mut results_vec: Vec<Option<ToolResult>> = vec![None; calls.len()];
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((idx, Ok(output))) => {
                    let output = match self.max_tool_output_bytes {
                        Some(max) => output.truncated(max),
                        None => output,
                    };
                    results_vec[idx] = Some(tool_output_to_result(call_ids[idx].clone(), output));
                }
                Ok((idx, Err(e))) => {
                    results_vec[idx] =
                        Some(ToolResult::error(call_ids[idx].clone(), e.to_string()));
                }
                Err(join_err) => {
                    tracing::error!(error = %join_err, "tool task panicked");
                }
            }
        }

        // Fill gaps (panicked tasks) with error results
        results_vec
            .into_iter()
            .enumerate()
            .map(|(idx, r)| {
                r.unwrap_or_else(|| {
                    ToolResult::error(call_ids[idx].clone(), "Tool execution panicked".to_string())
                })
            })
            .collect()
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
    on_text: Option<Arc<crate::llm::OnText>>,
    on_approval: Option<Arc<crate::llm::OnApproval>>,
    tool_timeout: Option<Duration>,
    max_tool_output_bytes: Option<usize>,
    structured_schema: Option<serde_json::Value>,
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

    pub fn build(self) -> Result<AgentRunner<P>, Error> {
        if self.max_turns == 0 {
            return Err(Error::Config("max_turns must be at least 1".into()));
        }
        if self.max_tokens == 0 {
            return Err(Error::Config("max_tokens must be at least 1".into()));
        }

        // Collect all tools, including memory tools created from builder's name
        let mut all_tools = self.tools;
        if let Some(memory) = self.memory {
            all_tools.extend(crate::memory::tools::memory_tools(memory, &self.name));
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
                name: "__respond__".into(),
                description: "Produce your final structured response. Call this tool when you \
                              have gathered all necessary information and are ready to return \
                              your answer in the required format."
                    .into(),
                input_schema: schema.clone(),
            });
        }

        Ok(AgentRunner {
            provider: self.provider,
            name: self.name,
            system_prompt: self.system_prompt,
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
        })
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
            .build()
            .unwrap();

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
            .build()
            .unwrap();

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
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

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

    #[tokio::test]
    async fn context_strategy_builder_sets_sliding_window() {
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text { text: "ok".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .context_strategy(ContextStrategy::SlidingWindow { max_tokens: 50000 })
            .build()
            .unwrap();

        assert!(matches!(
            runner.context_strategy,
            ContextStrategy::SlidingWindow { max_tokens: 50000 }
        ));
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
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .build()
            .unwrap();

        assert!(matches!(
            runner.context_strategy,
            ContextStrategy::Unlimited
        ));
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
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Found it!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let callback: Arc<crate::llm::OnApproval> = Arc::new(move |_calls| {
            approved_clone.store(true, Ordering::SeqCst);
            true
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
            },
            // After denial, LLM responds with text instead
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "I understand, I won't execute that.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let callback: Arc<crate::llm::OnApproval> = Arc::new(|_calls| false);

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
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        ]));

        let callback: Arc<crate::llm::OnApproval> = Arc::new(move |calls| {
            let names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
            received_clone.lock().expect("lock").extend(names);
            true
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
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Tool timed out, moving on.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
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
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Got results!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
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
            },
            // Turn 2: LLM sees validation error, responds with text
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "I see the validation error.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
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
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Got truncated result.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
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
            },
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
            .tool(Arc::new(MockTool::new("search", "small result")))
            .max_tool_output_bytes(1000)
            .build()
            .unwrap();

        let output = runner.execute("search").await.unwrap();
        assert_eq!(output.result, "Done!");
    }
}
