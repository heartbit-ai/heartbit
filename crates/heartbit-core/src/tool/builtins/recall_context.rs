use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::agent::context_recall::ContextRecallStore;
use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

/// Semantically find earlier tool outputs by meaning (when you don't remember
/// the exact ref). Returns ranked refs + snippets; restore the full body with
/// fetch_full_output(ref).
// Will be wired into `builtin_tools` in Task 7; suppress dead-code until then.
#[allow(dead_code)]
pub(crate) struct RecallContextTool {
    pub(crate) store: Arc<ContextRecallStore>,
}

impl Tool for RecallContextTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall_context".into(),
            description: "Find earlier tool outputs by meaning when you don't recall the exact \
                ref. Returns ranked {ref, tool, snippet}; call fetch_full_output(ref) to restore \
                the full body of the one you want."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to search for, in words." },
                    "limit": { "type": "integer", "description": "Max results (default 5)." }
                },
                "required": ["query"]
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let query = input
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("query is required".into()))?;
            let limit = input
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(5);

            let hits = self.store.recall(query, limit).await;
            if hits.is_empty() {
                return Ok(ToolOutput::success(
                    "No matching earlier outputs.".to_string(),
                ));
            }
            let mut out = format!(
                "{} match(es) — call fetch_full_output(ref) to restore:\n",
                hits.len()
            );
            for (i, h) in hits.iter().enumerate() {
                out.push_str(&format!(
                    "[{}] ref={} tool={} — {}\n",
                    i + 1,
                    h.r#ref,
                    h.tool_name,
                    h.snippet.replace('\n', " ")
                ));
            }
            Ok(ToolOutput::success(out))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_ranked_refs_for_a_query() {
        let store = Arc::new(ContextRecallStore::new());
        store
            .index(
                "tc_match",
                "bash",
                "cargo test failed: parser assertion error",
            )
            .await;
        store
            .index("tc_noise", "read", "lorem ipsum dolor sit amet")
            .await;
        let tool = RecallContextTool { store };
        let ctx = crate::ExecutionContext::default();

        let out = tool
            .execute(&ctx, json!({ "query": "test failure parser" }))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(
            out.content.contains("tc_match"),
            "ranked refs must include the match:\n{}",
            out.content
        );
        assert!(
            out.content.contains("fetch_full_output"),
            "must tell the model how to restore"
        );
    }

    #[test]
    fn definition_declares_query_param() {
        let tool = RecallContextTool {
            store: Arc::new(ContextRecallStore::new()),
        };
        let def = tool.definition();
        assert_eq!(def.name, "recall_context");
        assert!(def.input_schema["properties"].get("query").is_some());
    }
}
