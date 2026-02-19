use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

use super::KnowledgeBase;

/// Create knowledge tools for agent access to the knowledge base.
///
/// Returns 1 tool:
/// - `knowledge_search` — search the knowledge base for relevant documentation
pub fn knowledge_tools(kb: Arc<dyn KnowledgeBase>) -> Vec<Arc<dyn Tool>> {
    vec![Arc::new(KnowledgeSearchTool { kb })]
}

fn default_limit() -> usize {
    5
}

struct KnowledgeSearchTool {
    kb: Arc<dyn KnowledgeBase>,
}

#[derive(Deserialize)]
struct SearchInput {
    query: String,
    source_filter: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
}

impl Tool for KnowledgeSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "knowledge_search".into(),
            description: "Search the knowledge base for relevant documentation, code examples, \
                          and reference material. Use this when you need to find specific \
                          information from project docs, API references, or other indexed sources."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query describing what you're looking for"
                    },
                    "source_filter": {
                        "type": "string",
                        "description": "Optional URI prefix to restrict results to specific sources (e.g. 'docs/' or 'https://api.example.com')"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: SearchInput =
                serde_json::from_value(input).map_err(|e| Error::Agent(e.to_string()))?;

            let limit = input.limit.clamp(1, 20);

            let results = self
                .kb
                .search(super::KnowledgeQuery {
                    text: input.query,
                    source_filter: input.source_filter,
                    limit,
                })
                .await?;

            if results.is_empty() {
                return Ok(ToolOutput::success(
                    "No matching documents found in the knowledge base.",
                ));
            }

            let formatted = results
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    format!(
                        "--- Result {} (source: {}, matches: {}) ---\n{}",
                        i + 1,
                        r.chunk.source.uri,
                        r.match_count,
                        r.chunk.content,
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n");

            Ok(ToolOutput::success(format!(
                "Found {} result(s):\n\n{}",
                results.len(),
                formatted,
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::in_memory::InMemoryKnowledgeBase;
    use crate::knowledge::{Chunk, DocumentSource};

    fn setup() -> (Arc<dyn KnowledgeBase>, Vec<Arc<dyn Tool>>) {
        let kb: Arc<dyn KnowledgeBase> = Arc::new(InMemoryKnowledgeBase::new());
        let tools = knowledge_tools(kb.clone());
        (kb, tools)
    }

    fn find_tool<'a>(tools: &'a [Arc<dyn Tool>], name: &str) -> &'a Arc<dyn Tool> {
        tools
            .iter()
            .find(|t| t.definition().name == name)
            .unwrap_or_else(|| panic!("tool {name} not found"))
    }

    #[test]
    fn creates_one_tool() {
        let (_kb, tools) = setup();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition().name, "knowledge_search");
    }

    #[test]
    fn tool_definition_has_valid_schema() {
        let (_kb, tools) = setup();
        let def = tools[0].definition();
        assert!(!def.name.is_empty());
        assert!(!def.description.is_empty());
        assert!(def.input_schema.is_object());
        assert_eq!(def.input_schema["type"], "object");
        assert!(def.input_schema["properties"]["query"].is_object());
        let required = def.input_schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("query")));
    }

    #[tokio::test]
    async fn search_returns_formatted_results() {
        let (kb, tools) = setup();
        kb.index(Chunk {
            id: "c1".into(),
            content: "Rust provides memory safety without garbage collection.".into(),
            source: DocumentSource {
                uri: "docs/rust.md".into(),
                title: "Rust Guide".into(),
            },
            chunk_index: 0,
        })
        .await
        .unwrap();

        let search = find_tool(&tools, "knowledge_search");
        let result = search
            .execute(json!({"query": "rust memory"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Found 1 result"));
        assert!(result.content.contains("docs/rust.md"));
        assert!(result.content.contains("memory safety"));
    }

    #[tokio::test]
    async fn search_empty_results_returns_message() {
        let (_kb, tools) = setup();
        let search = find_tool(&tools, "knowledge_search");
        let result = search
            .execute(json!({"query": "nonexistent topic xyz"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("No matching documents"));
    }

    #[tokio::test]
    async fn search_with_source_filter() {
        let (kb, tools) = setup();
        kb.index(Chunk {
            id: "c1".into(),
            content: "Rust API reference".into(),
            source: DocumentSource {
                uri: "api/rust.md".into(),
                title: "API".into(),
            },
            chunk_index: 0,
        })
        .await
        .unwrap();
        kb.index(Chunk {
            id: "c2".into(),
            content: "Rust tutorial docs".into(),
            source: DocumentSource {
                uri: "docs/tutorial.md".into(),
                title: "Tutorial".into(),
            },
            chunk_index: 0,
        })
        .await
        .unwrap();

        let search = find_tool(&tools, "knowledge_search");
        let result = search
            .execute(json!({"query": "rust", "source_filter": "api/"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("api/rust.md"));
        assert!(!result.content.contains("docs/tutorial.md"));
    }

    #[tokio::test]
    async fn search_with_limit() {
        let (kb, tools) = setup();
        for i in 0..10 {
            kb.index(Chunk {
                id: format!("c{i}"),
                content: format!("Rust document {i}"),
                source: DocumentSource {
                    uri: "docs/rust.md".into(),
                    title: "Rust".into(),
                },
                chunk_index: i,
            })
            .await
            .unwrap();
        }

        let search = find_tool(&tools, "knowledge_search");
        let result = search
            .execute(json!({"query": "rust", "limit": 3}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Found 3 result"));
    }

    #[tokio::test]
    async fn search_rejects_missing_query() {
        let (_kb, tools) = setup();
        let search = find_tool(&tools, "knowledge_search");
        let result = search.execute(json!({})).await;
        assert!(result.is_err(), "should fail on missing required 'query'");
    }

    #[tokio::test]
    async fn search_default_limit_is_five() {
        let (kb, tools) = setup();
        for i in 0..10 {
            kb.index(Chunk {
                id: format!("c{i}"),
                content: format!("Rust item {i}"),
                source: DocumentSource {
                    uri: "f.md".into(),
                    title: "F".into(),
                },
                chunk_index: i,
            })
            .await
            .unwrap();
        }

        let search = find_tool(&tools, "knowledge_search");
        let result = search.execute(json!({"query": "rust"})).await.unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Found 5 result"));
    }
}
