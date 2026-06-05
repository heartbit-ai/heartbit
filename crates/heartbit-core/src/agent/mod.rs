//! Agent runtime — `AgentRunner`, `Orchestrator`, workflow agents, guardrails, and supporting types.

pub mod audit;
pub mod batch;
pub mod blackboard;
pub(crate) mod blackboard_tools;
mod builder;
pub mod cache;
pub mod context;
pub mod context_recall;
pub mod dag;
pub mod debate;
mod doom_loop;
pub mod evaluator;
pub mod events;
pub mod flow;
pub mod goal;
#[cfg(test)]
mod goal_live;
pub mod guardrail;
pub mod guardrails;
pub mod handoff;
pub mod instructions;
pub mod interrupt;
pub mod mixture;
pub mod observability;
pub mod orchestrator;
pub mod permission;
pub mod prompts;
pub mod pruner;
pub mod routing;
mod runner;
pub mod tenant_tracker;
#[cfg(test)]
mod tetris_live;
pub mod token_estimator;
pub mod tool_filter;
pub mod voting;
pub mod workflow;

#[cfg(test)]
pub(crate) mod test_helpers;

// Re-exports for backward compatibility
pub use builder::AgentRunnerBuilder;
pub use interrupt::InterruptHandle;
pub use runner::{AgentOutput, AgentRunner, OnInput};
// Imports used by the test module via `use super::*`
#[cfg(test)]
use crate::error::Error;
#[cfg(test)]
use crate::llm::LlmProvider;
#[cfg(test)]
use crate::llm::types::{Message, ToolCall, ToolDefinition};
#[cfg(test)]
use crate::tool::{Tool, ToolOutput};
#[cfg(test)]
use audit::AuditTrail;
#[cfg(test)]
use context::ContextStrategy;
#[cfg(test)]
use doom_loop::DoomLoopTracker;
#[cfg(test)]
use events::{AgentEvent, OnEvent};
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;

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
            _ctx: &crate::ExecutionContext,
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
            reasoning: None,
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
                    reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Sorry about that.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Tool failed, but I recovered.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
            reasoning: None,
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
    fn build_errors_on_explicit_empty_name() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider)
            .name("")
            .system_prompt("sys")
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string().contains("agent name must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn build_succeeds_with_default_name() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = AgentRunner::builder(provider)
            .system_prompt("sys")
            .build()
            .expect("minimal builder chain must succeed without an explicit name");
        assert_eq!(runner.name(), "agent");
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
                    reasoning: None,
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
            reasoning: None,
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
            reasoning: None,
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
                    reasoning: None,
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
                    reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Found it!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            // After denial, LLM responds with text instead
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "I understand, I won't execute that.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                _ctx: &crate::ExecutionContext,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Tool timed out, moving on.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Got results!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                _ctx: &crate::ExecutionContext,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: LLM sees validation error, responds with text
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "I see the validation error.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                _ctx: &crate::ExecutionContext,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Got truncated result.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
            reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done!".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
            model_name: Some("claude-sonnet-4-6-20250610".into()),
            tool_call_results: Vec::new(),
            goal_met: None,
        };
        let json_str = serde_json::to_string(&output).unwrap();
        let parsed: AgentOutput = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.result, "Hello!");
        assert_eq!(parsed.tool_calls_made, 3);
        assert_eq!(parsed.tokens_used.input_tokens, 100);
        assert_eq!(parsed.structured, Some(json!({"answer": "42"})));
        assert_eq!(parsed.estimated_cost_usd, Some(0.0342));
        assert_eq!(
            parsed.model_name.as_deref(),
            Some("claude-sonnet-4-6-20250610")
        );
    }

    #[test]
    fn agent_output_structured_none_serializes() {
        let output = AgentOutput {
            result: "ok".into(),
            tool_calls_made: 0,
            tokens_used: TokenUsage::default(),
            structured: None,
            estimated_cost_usd: None,
            model_name: None,
            tool_call_results: Vec::new(),
            goal_met: None,
        };
        let json_str = serde_json::to_string(&output).unwrap();
        let parsed: AgentOutput = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.structured.is_none());
        assert!(parsed.model_name.is_none());
    }

    #[test]
    fn agent_output_backward_compat_no_model_name() {
        // Old JSON without model_name field should deserialize with None
        let json = r#"{"result":"ok","tool_calls_made":0,"tokens_used":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"reasoning_tokens":0}}"#;
        let parsed: AgentOutput = serde_json::from_str(json).unwrap();
        assert!(parsed.model_name.is_none());
        assert_eq!(parsed.result, "ok");
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
                reasoning: None,
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
                    reasoning: None,
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
                _ctx: &crate::ExecutionContext,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
    async fn reasoning_response_emits_reasoning_event_before_llm_response() {
        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let events_clone = events.clone();

        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "The answer is 42.".into(),
            }],
            stop_reason: StopReason::EndTurn,
            reasoning: Some("Let me think... 42.".into()),
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("thinker")
            .system_prompt("sys")
            .on_event(Arc::new(move |e| {
                events_clone.lock().unwrap().push(e);
            }))
            .build()
            .unwrap();

        runner.execute("what is the answer?").await.unwrap();

        let events = events.lock().unwrap();
        let reasoning_idx = events
            .iter()
            .position(|e| matches!(e, AgentEvent::Reasoning { .. }))
            .expect("a Reasoning event must be emitted");
        match &events[reasoning_idx] {
            AgentEvent::Reasoning { agent, text, .. } => {
                assert_eq!(agent, "thinker");
                assert_eq!(text, "Let me think... 42.");
            }
            other => panic!("expected Reasoning, got: {other:?}"),
        }
        // It must come BEFORE the matching LlmResponse for that turn.
        let llm_idx = events
            .iter()
            .position(|e| matches!(e, AgentEvent::LlmResponse { .. }))
            .expect("an LlmResponse event must be emitted");
        assert!(
            reasoning_idx < llm_idx,
            "Reasoning must precede LlmResponse"
        );
    }

    #[tokio::test]
    async fn non_reasoning_response_emits_no_reasoning_event() {
        let events: Arc<std::sync::Mutex<Vec<AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let events_clone = events.clone();
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "plain answer".into(),
            }],
            stop_reason: StopReason::EndTurn,
            reasoning: None,
            usage: TokenUsage::default(),
            model: None,
        }]));
        let runner = AgentRunner::builder(provider)
            .name("plain")
            .system_prompt("sys")
            .on_event(Arc::new(move |e| {
                events_clone.lock().unwrap().push(e);
            }))
            .build()
            .unwrap();
        runner.execute("hi").await.unwrap();
        assert!(
            !events
                .lock()
                .unwrap()
                .iter()
                .any(|e| matches!(e, AgentEvent::Reasoning { .. })),
            "no Reasoning event when the model returns none"
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
            reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Result.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
            reasoning: None,
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
            reasoning: None,
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
                    reasoning: None,
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
                    reasoning: None,
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
                _response: &mut crate::llm::types::CompletionResponse,
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
                    reasoning: None,
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
                _response: &mut CompletionResponse,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "OK, skipping dangerous tool.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                    reasoning: None,
                    usage: TokenUsage::default(),
                    model: None,
                },
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Denied.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Used safe, dangerous blocked.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Found it.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Sure, here you go.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text { text: "OK.".into() }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let saw_post_tool_event = Arc::new(AtomicBool::new(false));
        let saw_clone = saw_post_tool_event.clone();
        let on_event: Arc<OnEvent> = Arc::new(move |event| {
            if let AgentEvent::GuardrailDenied { hook, .. } = &event
                && hook == "post_tool"
            {
                saw_clone.store(true, Ordering::SeqCst);
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

    /// SECURITY (F-AGENT-5): the audit record emitted when a `post_tool`
    /// guardrail returns `Err` MUST carry the runner's `tenant_id` and
    /// `user_id`. Before the fix this site set them to `None`, leaving
    /// post_tool denials orphaned of tenant context — bad for forensic
    /// correlation in a multi-tenant SOC pipeline.
    #[tokio::test]
    async fn post_tool_guardrail_error_audit_carries_tenant_user() {
        struct FailingPostTool;
        impl Guardrail for FailingPostTool {
            fn post_tool(
                &self,
                _call: &crate::llm::types::ToolCall,
                _output: &mut ToolOutput,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + '_>>
            {
                Box::pin(async { Err(Error::Guardrail("denied by policy".into())) })
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text { text: "OK.".into() }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let trail = Arc::new(crate::agent::audit::InMemoryAuditTrail::new());
        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "result")))
            .guardrail(Arc::new(FailingPostTool))
            .audit_trail(trail.clone())
            .audit_user_context("alice", "tenant-1")
            .build()
            .unwrap();

        runner.execute("search").await.unwrap();

        let entries = trail.entries_unscoped(usize::MAX).await.unwrap();
        let denial = entries
            .iter()
            .find(|e| e.event_type == "guardrail_denied")
            .expect("expected a guardrail_denied audit record");
        assert_eq!(
            denial.user_id.as_deref(),
            Some("alice"),
            "audit record should carry user_id: {denial:?}"
        );
        assert_eq!(
            denial.tenant_id.as_deref(),
            Some("tenant-1"),
            "audit record should carry tenant_id: {denial:?}"
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
            reasoning: None,
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
                        reasoning: None,
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
                        reasoning: None,
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
                    reasoning: None,
                    usage: TokenUsage::default(),
                    model: None,
                },
                // Turn 2: final answer
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    reasoning: None,
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
                    reasoning: None,
                    usage: TokenUsage::default(),
                    model: None,
                },
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Done.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 2: final answer
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                    reasoning: None,
                    usage: TokenUsage::default(),
                    model: None,
                },
                // Compression call response
                CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Compressed summary of large file.".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    reasoning: None,
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
                    reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Tool failed.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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

    // levenshtein tests live in util.rs (canonical location)

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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
            reasoning: None,
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
            reasoning: None,
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
        assert!(!tracker.record(&calls, 3, None).0);
        assert!(!tracker.record(&calls, 3, None).0);
        assert!(tracker.record(&calls, 3, None).0); // 3rd time triggers
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
        assert!(!tracker.record(&calls_a, 3, None).0);
        assert!(!tracker.record(&calls_a, 3, None).0);
        // Different input resets
        assert!(!tracker.record(&calls_b, 3, None).0);
        assert!(!tracker.record(&calls_b, 3, None).0);
        assert!(tracker.record(&calls_b, 3, None).0); // 3rd consecutive of calls_b
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
        assert!(!tracker.record(&calls_1, 2, None).0);
        assert!(tracker.record(&calls_2, 2, None).0); // Same name+input, different ID
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
        assert!(!tracker.record(&calls, 2, None).0);
        assert!(tracker.record(&calls, 2, None).0);
    }

    #[test]
    fn fuzzy_doom_loop_same_tools_different_inputs() {
        let mut tracker = DoomLoopTracker::new();
        let calls_a = vec![ToolCall {
            id: "c1".into(),
            name: "search".into(),
            input: json!({"query": "rust"}),
        }];
        let calls_b = vec![ToolCall {
            id: "c2".into(),
            name: "search".into(),
            input: json!({"query": "python"}),
        }];
        let calls_c = vec![ToolCall {
            id: "c3".into(),
            name: "search".into(),
            input: json!({"query": "go"}),
        }];
        // All have same tool name "search" but different inputs
        let (exact, fuzzy) = tracker.record(&calls_a, 5, Some(3));
        assert!(!exact && !fuzzy, "first call: no detection");
        let (exact, fuzzy) = tracker.record(&calls_b, 5, Some(3));
        assert!(!exact && !fuzzy, "second call: no detection yet");
        let (exact, fuzzy) = tracker.record(&calls_c, 5, Some(3));
        assert!(!exact && fuzzy, "third call: fuzzy triggered");
    }

    #[test]
    fn fuzzy_doom_loop_different_tools_no_trigger() {
        let mut tracker = DoomLoopTracker::new();
        let calls_a = vec![ToolCall {
            id: "c1".into(),
            name: "search".into(),
            input: json!({"query": "rust"}),
        }];
        let calls_b = vec![ToolCall {
            id: "c2".into(),
            name: "read".into(),
            input: json!({"file": "foo.txt"}),
        }];
        let calls_c = vec![ToolCall {
            id: "c3".into(),
            name: "write".into(),
            input: json!({"file": "bar.txt"}),
        }];
        // Different tool names each turn
        let (_, fuzzy) = tracker.record(&calls_a, 5, Some(3));
        assert!(!fuzzy);
        let (_, fuzzy) = tracker.record(&calls_b, 5, Some(3));
        assert!(!fuzzy);
        let (_, fuzzy) = tracker.record(&calls_c, 5, Some(3));
        assert!(!fuzzy);
    }

    #[test]
    fn fuzzy_doom_loop_disabled_by_default() {
        let mut tracker = DoomLoopTracker::new();
        let calls_a = vec![ToolCall {
            id: "c1".into(),
            name: "search".into(),
            input: json!({"query": "rust"}),
        }];
        let calls_b = vec![ToolCall {
            id: "c2".into(),
            name: "search".into(),
            input: json!({"query": "python"}),
        }];
        // fuzzy_threshold is None — no fuzzy detection
        let (_, fuzzy) = tracker.record(&calls_a, 5, None);
        assert!(!fuzzy);
        let (_, fuzzy) = tracker.record(&calls_b, 5, None);
        assert!(!fuzzy);
    }

    #[test]
    fn exact_match_does_not_double_trigger_fuzzy() {
        let mut tracker = DoomLoopTracker::new();
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "search".into(),
            input: json!({"query": "rust"}),
        }];
        // Exact same calls 3 times with both thresholds set to 3
        let (exact, fuzzy) = tracker.record(&calls, 3, Some(3));
        assert!(!exact && !fuzzy);
        let (exact, fuzzy) = tracker.record(&calls, 3, Some(3));
        assert!(!exact && !fuzzy);
        // Third time: exact triggers, fuzzy should NOT
        let (exact, fuzzy) = tracker.record(&calls, 3, Some(3));
        assert!(exact, "exact should trigger");
        assert!(!fuzzy, "fuzzy should not trigger when exact fires");
    }

    #[test]
    fn exact_match_resets_fuzzy_count() {
        let mut tracker = DoomLoopTracker::new();
        // Two different-input calls build fuzzy count
        let calls_a = vec![ToolCall {
            id: "c1".into(),
            name: "search".into(),
            input: json!({"query": "a"}),
        }];
        let calls_b = vec![ToolCall {
            id: "c2".into(),
            name: "search".into(),
            input: json!({"query": "b"}),
        }];
        let calls_c = vec![ToolCall {
            id: "c3".into(),
            name: "read".into(),
            input: json!({"file": "x"}),
        }];
        tracker.record(&calls_a, 5, Some(3));
        tracker.record(&calls_b, 5, Some(3));
        // Different tool resets fuzzy count
        tracker.record(&calls_c, 5, Some(3));
        assert_eq!(
            tracker.fuzzy_count(),
            1,
            "fuzzy count reset on different tools"
        );
    }

    #[test]
    fn builder_rejects_zero_max_fuzzy_identical_tool_calls() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .max_fuzzy_identical_tool_calls(0)
            .build();
        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("max_fuzzy_identical_tool_calls must be at least 1"),
                    "error: {msg}"
                );
            }
            Ok(_) => panic!("expected error for max_fuzzy_identical_tool_calls(0)"),
        }
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
            reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
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

    #[test]
    fn builder_rejects_zero_response_cache_size() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("test")
            .response_cache_size(0)
            .build();
        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("response_cache_size must be at least 1"),
                    "error: {msg}"
                );
            }
            Ok(_) => panic!("expected error for response_cache_size(0)"),
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "ok i won't do that".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "tests passed".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "got it".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "denied".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 3: done
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 3: done
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "gave up".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            // Turn 3: done
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "blocked".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
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
            reasoning: None,
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
            reasoning: None,
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
                reasoning: None,
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
                reasoning: None,
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
            reasoning: None,
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
            reasoning: None,
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
            reasoning: None,
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

        let entries = trail.entries_unscoped(usize::MAX).await.unwrap();
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
                reasoning: None,
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
                reasoning: None,
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

        let entries = trail.entries_unscoped(usize::MAX).await.unwrap();
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
            reasoning: None,
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

    #[tokio::test]
    async fn post_llm_warn_does_not_block_execution() {
        // A Warn guardrail should emit an event but NOT discard the response.
        use std::sync::atomic::{AtomicBool, Ordering};

        struct WarnAlways;
        impl Guardrail for WarnAlways {
            fn post_llm(
                &self,
                _response: &mut crate::llm::types::CompletionResponse,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<GuardAction, Error>> + Send + '_>,
            > {
                Box::pin(async { Ok(GuardAction::warn("suspicious but allowed")) })
            }
        }

        let warned = Arc::new(AtomicBool::new(false));
        let warned_clone = warned.clone();

        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "answer".into(),
            }],
            stop_reason: StopReason::EndTurn,
            reasoning: None,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .guardrail(Arc::new(WarnAlways))
            .on_event(Arc::new(move |event| {
                if matches!(event, AgentEvent::GuardrailWarned { .. }) {
                    warned_clone.store(true, Ordering::Relaxed);
                }
            }))
            .build()
            .unwrap();

        let output = runner.execute("hello").await.unwrap();
        // Response was NOT discarded — Warn allows execution to continue
        assert_eq!(output.result, "answer");
        // GuardrailWarned event was emitted
        assert!(
            warned.load(Ordering::Relaxed),
            "GuardrailWarned event should have fired"
        );
    }

    #[tokio::test]
    async fn pre_tool_warn_does_not_block_tool_execution() {
        // A Warn on pre_tool should emit event but still execute the tool.
        use std::sync::atomic::{AtomicBool, Ordering};

        struct WarnPreTool;
        impl Guardrail for WarnPreTool {
            fn pre_tool(
                &self,
                _call: &crate::llm::types::ToolCall,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<GuardAction, Error>> + Send + '_>,
            > {
                Box::pin(async { Ok(GuardAction::warn("risky tool usage")) })
            }
        }

        let warned = Arc::new(AtomicBool::new(false));
        let warned_clone = warned.clone();

        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Done with search.".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("search", "search result")))
            .guardrail(Arc::new(WarnPreTool))
            .on_event(Arc::new(move |event| {
                if matches!(event, AgentEvent::GuardrailWarned { .. }) {
                    warned_clone.store(true, Ordering::Relaxed);
                }
            }))
            .build()
            .unwrap();

        let output = runner.execute("search something").await.unwrap();
        // Tool was executed despite the Warn
        assert_eq!(output.result, "Done with search.");
        assert_eq!(output.tool_calls_made, 1);
        // GuardrailWarned event was emitted
        assert!(
            warned.load(Ordering::Relaxed),
            "GuardrailWarned event should have fired"
        );
    }

    #[tokio::test]
    async fn max_tool_calls_per_turn_caps_excess_dispatch() {
        // Canned response with 3 tool_use blocks; cap is 2.
        let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
            content: vec![
                ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "a".into(),
                    input: json!({}),
                },
                ContentBlock::ToolUse {
                    id: "c2".into(),
                    name: "b".into(),
                    input: json!({}),
                },
                ContentBlock::ToolUse {
                    id: "c3".into(),
                    name: "c".into(),
                    input: json!({}),
                },
            ],
            stop_reason: StopReason::ToolUse,
            reasoning: None,
            usage: TokenUsage::default(),
            model: None,
        }]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("a", "x")))
            .tool(Arc::new(MockTool::new("b", "y")))
            .tool(Arc::new(MockTool::new("c", "z")))
            .max_tool_calls_per_turn(2)
            .build()
            .unwrap();

        let err = runner.execute("go").await.unwrap_err();
        let s = err.to_string();
        assert!(s.contains("tool-call cap exceeded"), "got: {s}");
        // The error should carry partial usage (Error::WithPartialUsage shape)
        assert!(
            matches!(err, Error::WithPartialUsage { .. }),
            "got: {err:?}"
        );
    }

    #[test]
    fn max_tool_calls_per_turn_zero_is_rejected_at_build() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let result = AgentRunner::builder(provider)
            .name("t")
            .system_prompt("p")
            .max_tool_calls_per_turn(0)
            .build();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string()
                .contains("max_tool_calls_per_turn must be > 0"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn max_tool_calls_per_turn_at_cap_is_allowed() {
        // Exactly at the cap (2 calls, cap=2) should NOT error — only > cap errors.
        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        name: "a".into(),
                        input: json!({}),
                    },
                    ContentBlock::ToolUse {
                        id: "c2".into(),
                        name: "b".into(),
                        input: json!({}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("a", "x")))
            .tool(Arc::new(MockTool::new("b", "y")))
            .max_tool_calls_per_turn(2)
            .build()
            .unwrap();

        let output = runner.execute("go").await.unwrap();
        assert_eq!(output.tool_calls_made, 2);
    }

    /// SECURITY (F-AGENT-1): a Levenshtein-close tool name (e.g. `bask` for `bash`)
    /// must not bypass a `ToolPolicyGuardrail` deny rule on the canonical name.
    /// Before the fix, repair happened in `execute_tools_parallel` AFTER pre_tool —
    /// so a typo passed the policy and dispatched to the real tool.
    #[tokio::test]
    async fn levenshtein_repair_runs_before_tool_policy() {
        use crate::agent::guardrails::tool_policy::{ToolPolicyGuardrail, ToolRule};

        let provider = Arc::new(MockProvider::new(vec![
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".into(),
                    // typo: should match `bash` via Levenshtein distance 1
                    name: "bask".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
            CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            },
        ]));

        // Policy denies the canonical name `bash`. If repair were post-policy,
        // the typo `bask` would pass the policy and execute the bash tool.
        let policy = Arc::new(ToolPolicyGuardrail::new(
            vec![ToolRule {
                tool_pattern: "bash".into(),
                action: crate::GuardAction::deny("blocked"),
                input_constraints: vec![],
            }],
            crate::GuardAction::Allow,
        ));

        let runner = AgentRunner::builder(provider)
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(MockTool::new("bash", "DANGEROUS_OUTPUT")))
            .guardrails(vec![policy])
            .build()
            .unwrap();

        let output = runner.execute("run shell").await.unwrap();

        // The deny must have fired. The tool result returned to the LLM should
        // be the deny-message, NOT "DANGEROUS_OUTPUT".
        assert!(
            !output.result.contains("DANGEROUS_OUTPUT"),
            "tool result leaked despite policy deny: {}",
            output.result
        );
        assert_eq!(
            output.tool_calls_made, 1,
            "exactly one tool call should be recorded (denied)"
        );
    }
}
