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

#[cfg(test)]
mod tests {
    use super::*;

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
}
