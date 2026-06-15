//! Function-call validity guardrail (function-call-hallucination detection).
//!
//! Granite Guardian (arXiv 2412.07724) highlights **function-calling
//! hallucination** — a model inventing a tool that doesn't exist, or emitting
//! arguments that don't satisfy the tool's schema — as the most agent-relevant
//! safety check. The runner already *repairs* near-miss tool names (Levenshtein
//! ≤ 2) and validates schemas before `execute`, but that path silently fixes or
//! errors; it offers no composable **policy** hook and no audit signal.
//!
//! [`FunctionCallGuard`] is that hook: a `pre_tool` guardrail that **denies** a
//! call to an unknown tool (a genuine hallucination — not a typo) or one whose
//! arguments fail the tool's JSON Schema, and raises a guardrail event so the
//! detection shows up in observability and evals.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use crate::agent::guardrail::{GuardAction, Guardrail};
use crate::error::Error;
use crate::llm::types::ToolCall;

/// Denies tool calls that name a tool outside the known set, or whose arguments
/// violate the tool's JSON Schema. See the module docs.
pub struct FunctionCallGuard {
    /// Known tool name → its input JSON Schema (for argument validation).
    schemas: HashMap<String, serde_json::Value>,
    /// When `true`, also validate arguments against the schema; when `false`,
    /// only the unknown-tool check runs.
    validate_args: bool,
}

impl FunctionCallGuard {
    /// Build from `(tool_name, input_schema)` pairs (typically the agent's tool
    /// definitions). Argument validation is on by default.
    pub fn new(schemas: impl IntoIterator<Item = (String, serde_json::Value)>) -> Self {
        Self {
            schemas: schemas.into_iter().collect(),
            validate_args: true,
        }
    }

    /// Toggle JSON-Schema argument validation (the unknown-tool check always runs).
    #[must_use]
    pub fn validate_args(mut self, on: bool) -> Self {
        self.validate_args = on;
        self
    }
}

impl Guardrail for FunctionCallGuard {
    fn name(&self) -> &str {
        "function_call_validity"
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        let action = match self.schemas.get(&call.name) {
            None => GuardAction::Deny {
                reason: format!(
                    "hallucinated tool call: '{}' is not an available tool. Available: [{}]",
                    call.name,
                    self.schemas.keys().cloned().collect::<Vec<_>>().join(", ")
                ),
            },
            Some(schema) if self.validate_args => {
                match crate::tool::validate_tool_input(schema, &call.input) {
                    Ok(()) => GuardAction::Allow,
                    Err(msg) => GuardAction::Deny {
                        reason: format!("invalid arguments for '{}': {msg}", call.name),
                    },
                }
            }
            Some(_) => GuardAction::Allow,
        };
        Box::pin(async move { Ok(action) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_obj() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "x": { "type": "integer" } },
            "required": ["x"]
        })
    }

    fn call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn denies_unknown_tool() {
        let guard = FunctionCallGuard::new([("known".into(), schema_obj())]);
        let action = guard
            .pre_tool(&call("imaginary_tool", serde_json::json!({})))
            .await
            .unwrap();
        match action {
            GuardAction::Deny { reason } => assert!(reason.contains("hallucinated")),
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn allows_known_tool_with_valid_args() {
        let guard = FunctionCallGuard::new([("known".into(), schema_obj())]);
        let action = guard
            .pre_tool(&call("known", serde_json::json!({"x": 5})))
            .await
            .unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn denies_known_tool_with_invalid_args() {
        let guard = FunctionCallGuard::new([("known".into(), schema_obj())]);
        let action = guard
            .pre_tool(&call("known", serde_json::json!({"x": "not-an-int"})))
            .await
            .unwrap();
        match action {
            GuardAction::Deny { reason } => assert!(reason.contains("invalid arguments")),
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn arg_validation_can_be_disabled() {
        let guard = FunctionCallGuard::new([("known".into(), schema_obj())]).validate_args(false);
        // Bad args, but validation off → only the unknown-tool check runs → Allow.
        let action = guard
            .pre_tool(&call("known", serde_json::json!({"x": "bad"})))
            .await
            .unwrap();
        assert_eq!(action, GuardAction::Allow);
    }
}
