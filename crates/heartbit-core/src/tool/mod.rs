#[cfg(feature = "a2a")]
pub mod a2a;
pub mod builtins;
pub mod handoff;
pub mod mcp;
pub mod mcp_presets;
pub mod mcp_server;

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
///
/// # Example
///
/// Implementing a simple synchronous tool that echoes its input:
///
/// ```rust
/// use std::future::Future;
/// use std::pin::Pin;
/// use heartbit_core::{Tool, ToolOutput};
/// use heartbit_core::llm::types::ToolDefinition;
///
/// struct EchoTool;
///
/// impl Tool for EchoTool {
///     fn definition(&self) -> ToolDefinition {
///         ToolDefinition {
///             name: "echo".into(),
///             description: "Echo back the input string.".into(),
///             input_schema: serde_json::json!({
///                 "type": "object",
///                 "properties": { "text": { "type": "string" } },
///                 "required": ["text"]
///             }),
///         }
///     }
///
///     fn execute(
///         &self,
///         input: serde_json::Value,
///     ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
///         Box::pin(async move {
///             let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");
///             Ok(ToolOutput::success(text.to_string()))
///         })
///     }
/// }
/// ```
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

#[cfg(all(test, feature = "macro"))]
mod macro_tests {
    use super::*;
    use crate::Error;
    use serde_json::json;

    #[heartbit_macro::heartbit_tool(description = "Greet a user by name")]
    async fn greet_user(
        /// The user's name
        name: String,
    ) -> Result<ToolOutput, Error> {
        Ok(ToolOutput::success(format!("Hello, {name}!")))
    }

    #[heartbit_macro::heartbit_tool(description = "Add two integers")]
    async fn add_numbers(
        /// First number
        a: i32,
        /// Second number
        b: i32,
    ) -> Result<ToolOutput, Error> {
        Ok(ToolOutput::success(format!("{}", a + b)))
    }

    #[heartbit_macro::heartbit_tool(description = "Search with optional limit")]
    async fn search_items(
        /// The query string
        query: String,
        /// Max results
        #[tool(default = 10)]
        limit: Option<u32>,
    ) -> Result<ToolOutput, Error> {
        let l = limit.unwrap_or(10);
        Ok(ToolOutput::success(format!("query={query}, limit={l}")))
    }

    #[heartbit_macro::heartbit_tool(description = "Tool with various types")]
    async fn typed_params(
        text: String,
        count: u64,
        ratio: f64,
        flag: bool,
        items: Vec<String>,
        data: serde_json::Value,
    ) -> Result<ToolOutput, Error> {
        let _ = (text, count, ratio, flag, items, data);
        Ok(ToolOutput::success("ok"))
    }

    #[tokio::test]
    async fn macro_tool_name_is_snake_case() {
        let tool = GreetUser;
        let def = tool.definition();
        assert_eq!(def.name, "greet_user");
    }

    #[tokio::test]
    async fn macro_tool_description() {
        let tool = GreetUser;
        let def = tool.definition();
        assert_eq!(def.description, "Greet a user by name");
    }

    #[tokio::test]
    async fn macro_required_params_in_schema() {
        let tool = GreetUser;
        let def = tool.definition();
        let required = def.input_schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "name");
    }

    #[tokio::test]
    async fn macro_optional_not_in_required() {
        let tool = SearchItems;
        let def = tool.definition();
        let required = def.input_schema["required"].as_array().unwrap();
        // Only "query" should be required, not "limit"
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "query");
    }

    #[tokio::test]
    async fn macro_param_description_in_schema() {
        let tool = GreetUser;
        let def = tool.definition();
        let name_prop = &def.input_schema["properties"]["name"];
        assert_eq!(name_prop["description"], "The user's name");
        assert_eq!(name_prop["type"], "string");
    }

    #[tokio::test]
    async fn macro_default_value_in_schema() {
        let tool = SearchItems;
        let def = tool.definition();
        let limit_prop = &def.input_schema["properties"]["limit"];
        assert_eq!(limit_prop["default"], 10);
    }

    #[tokio::test]
    async fn macro_execute_valid_input() {
        let tool = GreetUser;
        let result = tool.execute(json!({"name": "Alice"})).await.unwrap();
        assert_eq!(result.content, "Hello, Alice!");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn macro_execute_multiple_params() {
        let tool = AddNumbers;
        let result = tool.execute(json!({"a": 3, "b": 7})).await.unwrap();
        assert_eq!(result.content, "10");
    }

    #[tokio::test]
    async fn macro_execute_missing_required_field() {
        let tool = GreetUser;
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("missing required field"), "got: {err}");
    }

    #[tokio::test]
    async fn macro_execute_wrong_type() {
        let tool = GreetUser;
        let result = tool.execute(json!({"name": 42})).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("invalid value"), "got: {err}");
    }

    #[tokio::test]
    async fn macro_execute_optional_with_default() {
        let tool = SearchItems;
        // Omit optional param — should use default
        let result = tool.execute(json!({"query": "rust"})).await.unwrap();
        assert_eq!(result.content, "query=rust, limit=10");
    }

    #[tokio::test]
    async fn macro_execute_optional_provided() {
        let tool = SearchItems;
        let result = tool
            .execute(json!({"query": "rust", "limit": 5}))
            .await
            .unwrap();
        assert_eq!(result.content, "query=rust, limit=5");
    }

    #[tokio::test]
    async fn macro_type_mapping_schema() {
        let tool = TypedParams;
        let def = tool.definition();
        let props = &def.input_schema["properties"];
        assert_eq!(props["text"]["type"], "string");
        assert_eq!(props["count"]["type"], "integer");
        assert_eq!(props["ratio"]["type"], "number");
        assert_eq!(props["flag"]["type"], "boolean");
        assert_eq!(props["items"]["type"], "array");
        assert_eq!(props["items"]["items"]["type"], "string");
        // serde_json::Value maps to {} (any)
        assert!(
            props["data"].as_object().unwrap().is_empty()
                || !props["data"].as_object().unwrap().contains_key("type")
        );
    }

    #[tokio::test]
    async fn macro_all_required_for_non_optional() {
        let tool = TypedParams;
        let def = tool.definition();
        let required = def.input_schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 6);
    }
}
