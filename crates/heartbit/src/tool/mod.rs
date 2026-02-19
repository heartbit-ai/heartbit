pub mod mcp;

use std::future::Future;
use std::pin::Pin;

use crate::error::Error;
use crate::llm::types::ToolDefinition;

/// Output of a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Trait for tools that agents can invoke.
///
/// Uses `Pin<Box<dyn Future>>` return type for dyn-compatibility,
/// allowing tools to be stored as `Arc<dyn Tool>`.
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>>;
}

/// Validate tool input against the tool's declared JSON Schema.
///
/// Returns `Ok(())` if valid, `Err(error_message)` if the input
/// does not conform. The error message is suitable for sending back
/// to the LLM so it can self-correct.
pub fn validate_tool_input(
    schema: &serde_json::Value,
    input: &serde_json::Value,
) -> Result<(), String> {
    let validator = match jsonschema::validator_for(schema) {
        Ok(v) => v,
        Err(e) => {
            // If the schema itself is invalid, skip validation rather than
            // rejecting every call. Log a warning for the operator.
            tracing::warn!(error = %e, "invalid tool schema, skipping validation");
            return Ok(());
        }
    };

    let errors: Vec<String> = validator
        .iter_errors(input)
        .map(|e| e.to_string())
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("Input validation failed: {}", errors.join("; ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_output_success() {
        let output = ToolOutput::success("result data");
        assert_eq!(output.content, "result data");
        assert!(!output.is_error);
    }

    #[test]
    fn tool_output_error() {
        let output = ToolOutput::error("something failed");
        assert_eq!(output.content, "something failed");
        assert!(output.is_error);
    }

    #[test]
    fn validate_accepts_valid_input() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        });
        let input = json!({"query": "test"});
        assert!(validate_tool_input(&schema, &input).is_ok());
    }

    #[test]
    fn validate_rejects_missing_required() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        });
        let input = json!({});
        let err = validate_tool_input(&schema, &input).unwrap_err();
        assert!(err.contains("validation failed"), "got: {err}");
    }

    #[test]
    fn validate_rejects_wrong_type() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        });
        let input = json!({"query": 42});
        let err = validate_tool_input(&schema, &input).unwrap_err();
        assert!(err.contains("validation failed"), "got: {err}");
    }

    #[test]
    fn validate_accepts_any_for_minimal_schema() {
        let schema = json!({"type": "object"});
        let input = json!({});
        assert!(validate_tool_input(&schema, &input).is_ok());
    }

    #[test]
    fn validate_skips_on_invalid_schema() {
        // An invalid schema should not block tool execution
        let schema = json!({"type": "not-a-real-type"});
        let input = json!({"anything": true});
        // Should not fail even though schema is invalid — skips validation
        assert!(validate_tool_input(&schema, &input).is_ok());
    }

    #[test]
    fn validate_accepts_extra_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        });
        // Extra properties are allowed by default in JSON Schema
        let input = json!({"query": "test", "extra": true});
        assert!(validate_tool_input(&schema, &input).is_ok());
    }
}
