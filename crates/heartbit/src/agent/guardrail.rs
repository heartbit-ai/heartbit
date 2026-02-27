use std::future::Future;
use std::pin::Pin;

use crate::error::Error;
use crate::llm::types::{CompletionRequest, CompletionResponse, ToolCall};
use crate::tool::ToolOutput;

/// Action returned by guardrail hooks that can deny operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardAction {
    /// Allow the operation to proceed.
    Allow,
    /// Deny the operation with a reason.
    Deny { reason: String },
    /// Log the concern but allow the operation to proceed.
    ///
    /// The agent loop treats `Warn` like `Allow` but emits
    /// `AgentEvent::GuardrailWarned` and an audit record. This enables
    /// monitoring mode (shadow enforcement) without blocking production.
    Warn { reason: String },
}

impl GuardAction {
    /// Create a `Deny` action with the given reason.
    pub fn deny(reason: impl Into<String>) -> Self {
        GuardAction::Deny {
            reason: reason.into(),
        }
    }

    /// Create a `Warn` action with the given reason.
    pub fn warn(reason: impl Into<String>) -> Self {
        GuardAction::Warn {
            reason: reason.into(),
        }
    }

    /// Returns `true` if this action blocks the operation (`Deny`).
    pub fn is_denied(&self) -> bool {
        matches!(self, GuardAction::Deny { .. })
    }
}

/// Optional metadata for guardrail identification in events and audit records.
///
/// All guardrails auto-implement with `"unnamed"` via the blanket default.
/// Override `name()` to attribute which guardrail fired in logs.
pub trait GuardrailMeta {
    /// Human-readable name for this guardrail, used in events and audit.
    fn name(&self) -> &str {
        "unnamed"
    }
}

/// Interceptor hooks for LLM calls and tool executions.
///
/// All methods have default no-op implementations so guardrails only need to
/// override the hooks they care about. Methods use `Pin<Box<dyn Future>>` for
/// dyn-compatibility (same pattern as the `Tool` trait).
///
/// Multiple guardrails are registered via `Vec<Arc<dyn Guardrail>>` — first
/// `Deny` wins.
pub trait Guardrail: Send + Sync {
    /// Called before each LLM call. Can mutate the request (e.g., inject safety
    /// instructions, redact content). `Err` aborts the run.
    fn pre_llm(
        &self,
        _request: &mut CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }

    /// Called after each LLM response. Can inspect the response and deny it.
    /// `Deny` discards the response and injects the denial reason as a user
    /// message (consumes a turn). `Err` aborts the run.
    fn post_llm(
        &self,
        _response: &CompletionResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        Box::pin(async { Ok(GuardAction::Allow) })
    }

    /// Called before each tool execution. Can deny individual tool calls.
    /// `Deny` returns a `ToolResult::error` for that call. `Err` aborts the run.
    fn pre_tool(
        &self,
        _call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        Box::pin(async { Ok(GuardAction::Allow) })
    }

    /// Called after each tool execution (after truncation). Can mutate the
    /// output (e.g., redact sensitive data). `Err` converts to a tool error
    /// (consistent with tool execution errors — the agent loop continues).
    fn post_tool(
        &self,
        _call: &ToolCall,
        _output: &mut ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_action_deny_constructor() {
        let action = GuardAction::deny("PII detected");
        match action {
            GuardAction::Deny { reason } => assert_eq!(reason, "PII detected"),
            _ => panic!("expected Deny"),
        }
    }

    #[test]
    fn guard_action_warn_constructor() {
        let action = GuardAction::warn("suspicious pattern");
        match action {
            GuardAction::Warn { reason } => assert_eq!(reason, "suspicious pattern"),
            _ => panic!("expected Warn"),
        }
    }

    #[test]
    fn guard_action_is_denied() {
        assert!(GuardAction::deny("blocked").is_denied());
        assert!(!GuardAction::Allow.is_denied());
        assert!(!GuardAction::warn("suspicious").is_denied());
    }

    #[test]
    fn guardrail_meta_default_name() {
        struct MyGuardrail;
        impl GuardrailMeta for MyGuardrail {}
        assert_eq!(MyGuardrail.name(), "unnamed");
    }

    #[test]
    fn guardrail_meta_custom_name() {
        struct NamedGuardrail;
        impl GuardrailMeta for NamedGuardrail {
            fn name(&self) -> &str {
                "pii_detector"
            }
        }
        assert_eq!(NamedGuardrail.name(), "pii_detector");
    }

    #[tokio::test]
    async fn default_guardrail_allows_everything() {
        struct NoOpGuardrail;
        impl Guardrail for NoOpGuardrail {}

        let g = NoOpGuardrail;

        let mut request = CompletionRequest {
            system: "sys".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        g.pre_llm(&mut request).await.unwrap();

        let response = CompletionResponse {
            content: vec![],
            stop_reason: crate::llm::types::StopReason::EndTurn,
            usage: crate::llm::types::TokenUsage::default(),
            model: None,
        };
        let action = g.post_llm(&response).await.unwrap();
        assert!(matches!(action, GuardAction::Allow));

        let call = ToolCall {
            id: "c1".into(),
            name: "test".into(),
            input: serde_json::json!({}),
        };
        let action = g.pre_tool(&call).await.unwrap();
        assert!(matches!(action, GuardAction::Allow));

        let mut output = ToolOutput::success("result".to_string());
        g.post_tool(&call, &mut output).await.unwrap();
        assert_eq!(output.content, "result");
    }

    #[tokio::test]
    async fn post_tool_can_mutate_output() {
        struct RedactGuardrail;
        impl Guardrail for RedactGuardrail {
            fn post_tool(
                &self,
                _call: &ToolCall,
                output: &mut ToolOutput,
            ) -> Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + '_>>
            {
                // Mutation is synchronous; the future just returns Ok(())
                output.content = output.content.replace("secret", "[REDACTED]");
                Box::pin(async { Ok(()) })
            }
        }

        let g = RedactGuardrail;
        let call = ToolCall {
            id: "c1".into(),
            name: "test".into(),
            input: serde_json::json!({}),
        };
        let mut output = ToolOutput::success("the secret is 42".to_string());
        g.post_tool(&call, &mut output).await.unwrap();
        assert_eq!(output.content, "the [REDACTED] is 42");
    }

    #[tokio::test]
    async fn pre_tool_deny_returns_deny_action() {
        struct BlockBashGuardrail;
        impl Guardrail for BlockBashGuardrail {
            fn pre_tool(
                &self,
                call: &ToolCall,
            ) -> Pin<Box<dyn std::future::Future<Output = Result<GuardAction, Error>> + Send + '_>>
            {
                let name = call.name.clone();
                Box::pin(async move {
                    if name == "bash" {
                        Ok(GuardAction::deny("bash tool is disabled"))
                    } else {
                        Ok(GuardAction::Allow)
                    }
                })
            }
        }

        let g = BlockBashGuardrail;
        let bash_call = ToolCall {
            id: "c1".into(),
            name: "bash".into(),
            input: serde_json::json!({"command": "rm -rf /"}),
        };
        let action = g.pre_tool(&bash_call).await.unwrap();
        assert!(
            matches!(action, GuardAction::Deny { reason } if reason == "bash tool is disabled")
        );

        let read_call = ToolCall {
            id: "c2".into(),
            name: "read".into(),
            input: serde_json::json!({"path": "/tmp/test.txt"}),
        };
        let action = g.pre_tool(&read_call).await.unwrap();
        assert!(matches!(action, GuardAction::Allow));
    }
}
