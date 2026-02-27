#[cfg(feature = "a2a")]
pub mod a2a;
pub mod builtins;
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

    /// Truncate content if it exceeds `max_bytes`, preserving UTF-8 validity.
    ///
    /// When truncated, appends a `[truncated: N bytes omitted]` suffix so the
    /// LLM knows data was cut. Content within the limit is returned unchanged.
    /// A `max_bytes` of 0 is treated as no-op (returns content unchanged).
    ///
    /// Note: the suffix itself is not counted toward `max_bytes`, so the
    /// result may slightly exceed the limit.
    pub fn truncated(mut self, max_bytes: usize) -> Self {
        if max_bytes == 0 {
            return self;
        }
        if self.content.len() > max_bytes {
            let mut cut = max_bytes;
            while cut > 0 && !self.content.is_char_boundary(cut) {
                cut -= 1;
            }
            let omitted = self.content.len() - cut;
            self.content.truncate(cut);
            self.content
                .push_str(&format!("\n\n[truncated: {omitted} bytes omitted]"));
        }
        self
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
    fn tool_output_truncated_noop_when_within_limit() {
        let output = ToolOutput::success("short text");
        let truncated = output.truncated(100);
        assert_eq!(truncated.content, "short text");
        assert!(!truncated.is_error);
    }

    #[test]
    fn tool_output_truncated_cuts_long_content() {
        let output = ToolOutput::success("a".repeat(1000));
        let truncated = output.truncated(100);
        assert!(truncated.content.len() < 1000);
        assert!(truncated.content.starts_with("aaaa"));
        assert!(truncated.content.contains("[truncated:"));
        assert!(truncated.content.contains("bytes omitted]"));
        assert!(!truncated.is_error); // preserves is_error flag
    }

    #[test]
    fn tool_output_truncated_preserves_utf8() {
        // "é" is 2 bytes in UTF-8. A cut at byte 5 would split a char boundary.
        let output = ToolOutput::success("ééééé"); // 10 bytes
        let truncated = output.truncated(5);
        // Should cut at char boundary (4 bytes = 2 chars), not mid-char
        assert!(truncated.content.starts_with("éé"));
        assert!(truncated.content.contains("[truncated:"));
    }

    #[test]
    fn tool_output_truncated_exact_boundary_noop() {
        let output = ToolOutput::success("hello"); // 5 bytes
        let truncated = output.truncated(5);
        assert_eq!(truncated.content, "hello");
    }

    #[test]
    fn tool_output_truncated_zero_is_noop() {
        let output = ToolOutput::success("some content");
        let truncated = output.truncated(0);
        assert_eq!(truncated.content, "some content"); // unchanged
    }

    #[test]
    fn tool_output_truncated_error_also_truncates() {
        let output = ToolOutput::error("e".repeat(200));
        let truncated = output.truncated(50);
        assert!(truncated.content.contains("[truncated:"));
        assert!(truncated.is_error); // preserves error flag
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
