use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::agent::context_recall::ContextRecallStore;
use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

/// Restores the exact untruncated content of a past tool result by its ref
/// (the `tool_call_id`), e.g. after the pruner truncated it.
// Will be wired into `builtin_tools` in Task 7; suppress dead-code until then.
#[allow(dead_code)]
pub(crate) struct FetchFullOutputTool {
    pub(crate) store: Arc<ContextRecallStore>,
}

impl Tool for FetchFullOutputTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_full_output".into(),
            description: "Restore the full, untruncated content of an earlier tool result by \
                its ref. Use when an old tool output shows a '[pruned: … id=<ref>]' marker and \
                you need the complete content back."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "The ref (tool_call_id) shown in the pruned marker."
                    }
                },
                "required": ["ref"]
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let r = input
                .get("ref")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("ref is required".into()))?;
            match self.store.get(r).await {
                Some(content) => Ok(ToolOutput::success(content)),
                None => Ok(ToolOutput::error(format!(
                    "no stored output for ref '{r}' — it may have been evicted"
                ))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fetches_known_ref_and_errors_on_unknown() {
        let store = Arc::new(ContextRecallStore::new());
        store.index("tc_1", "bash", "FULL OUTPUT BODY").await;
        let tool = FetchFullOutputTool {
            store: store.clone(),
        };
        let ctx = crate::ExecutionContext::default();

        let ok = tool.execute(&ctx, json!({ "ref": "tc_1" })).await.unwrap();
        assert_eq!(ok.content, "FULL OUTPUT BODY");
        assert!(!ok.is_error);

        let miss = tool.execute(&ctx, json!({ "ref": "nope" })).await.unwrap();
        assert!(miss.is_error);
        assert!(miss.content.contains("nope"));
    }

    #[test]
    fn definition_declares_ref_param() {
        let tool = FetchFullOutputTool {
            store: Arc::new(ContextRecallStore::new()),
        };
        let def = tool.definition();
        assert_eq!(def.name, "fetch_full_output");
        assert!(def.input_schema["properties"].get("ref").is_some());
    }
}
