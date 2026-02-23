use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use chrono::Utc;
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

use super::{Memory, MemoryEntry, MemoryQuery};

/// Create shared memory tools for cross-agent memory access.
///
/// - `shared_memory_read`: read memories from any agent's namespace
/// - `shared_memory_write`: write to a shared namespace visible to all agents
pub fn shared_memory_tools(memory: Arc<dyn Memory>, agent_name: &str) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(SharedMemoryReadTool {
            memory: memory.clone(),
        }),
        Arc::new(SharedMemoryWriteTool {
            memory,
            agent_name: agent_name.into(),
        }),
    ]
}

// --- shared_memory_read ---

struct SharedMemoryReadTool {
    memory: Arc<dyn Memory>,
}

#[derive(Deserialize)]
struct SharedReadInput {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "super::default_recall_limit")]
    limit: usize,
}

impl Tool for SharedMemoryReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "shared_memory_read".into(),
            description: "Read memories from any agent's namespace. Use this to access \
                          knowledge that other agents have stored."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for"
                    },
                    "agent": {
                        "type": "string",
                        "description": "Filter by agent name (omit for all agents)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)"
                    }
                }
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: SharedReadInput =
                serde_json::from_value(input).map_err(|e| Error::Memory(e.to_string()))?;

            let results = self
                .memory
                .recall(MemoryQuery {
                    text: input.query,
                    category: input.category,
                    tags: input.tags,
                    agent: input.agent, // None = all agents
                    limit: input.limit,
                    ..Default::default()
                })
                .await?;

            if results.is_empty() {
                return Ok(ToolOutput::success("No shared memories found."));
            }

            let formatted: Vec<String> = results
                .iter()
                .map(|e| {
                    let mt = match e.memory_type {
                        crate::memory::MemoryType::Episodic => "episodic",
                        crate::memory::MemoryType::Semantic => "semantic",
                        crate::memory::MemoryType::Reflection => "reflection",
                    };
                    format!(
                        "- [{}] @{} ({}, {}, importance:{}, strength:{:.2}) {}",
                        e.id, e.agent, e.category, mt, e.importance, e.strength, e.content,
                    )
                })
                .collect();

            let count = results.len();
            let noun = if count == 1 { "memory" } else { "memories" };
            Ok(ToolOutput::success(format!(
                "Found {count} shared {noun}:\n{}",
                formatted.join("\n")
            )))
        })
    }
}

// --- shared_memory_write ---

struct SharedMemoryWriteTool {
    memory: Arc<dyn Memory>,
    agent_name: String,
}

#[derive(Deserialize)]
struct SharedWriteInput {
    content: String,
    #[serde(default = "super::default_category")]
    category: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "super::default_importance")]
    importance: u8,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    summary: Option<String>,
}

impl Tool for SharedMemoryWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "shared_memory_write".into(),
            description: "Write a memory to the shared namespace, visible to all agents. \
                          Use this to share important findings with other agents."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to share"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["fact", "observation", "preference", "procedure"],
                        "description": "Category (default: fact)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for organization"
                    },
                    "importance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Importance score 1-10 (default: 5)"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords for improved retrieval (BM25 boost)"
                    },
                    "summary": {
                        "type": "string",
                        "description": "One-sentence summary for context"
                    }
                },
                "required": ["content"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: SharedWriteInput =
                serde_json::from_value(input).map_err(|e| Error::Memory(e.to_string()))?;

            let id = format!("shared:{}", Uuid::new_v4());
            let now = Utc::now();
            let entry = MemoryEntry {
                id: id.clone(),
                agent: self.agent_name.clone(),
                content: input.content,
                category: input.category,
                tags: input.tags,
                created_at: now,
                last_accessed: now,
                access_count: 0,
                importance: input.importance.clamp(1, 10),
                memory_type: super::MemoryType::default(),
                keywords: input.keywords,
                summary: input.summary,
                strength: 1.0,
                related_ids: vec![],
                source_ids: vec![],
                embedding: None,
            };

            self.memory.store(entry).await?;
            Ok(ToolOutput::success(format!(
                "Shared memory stored with id: {id}"
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::in_memory::InMemoryStore;

    fn setup() -> (Arc<dyn Memory>, Vec<Arc<dyn Tool>>) {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let tools = shared_memory_tools(store.clone(), "agent_a");
        (store, tools)
    }

    fn find_tool<'a>(tools: &'a [Arc<dyn Tool>], name: &str) -> &'a Arc<dyn Tool> {
        tools
            .iter()
            .find(|t| t.definition().name == name)
            .unwrap_or_else(|| panic!("tool {name} not found"))
    }

    #[test]
    fn creates_two_tools() {
        let (_store, tools) = setup();
        assert_eq!(tools.len(), 2);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert!(names.contains(&"shared_memory_read".to_string()));
        assert!(names.contains(&"shared_memory_write".to_string()));
    }

    #[tokio::test]
    async fn write_and_read_shared_memory() {
        let (_store, tools) = setup();
        let write_tool = find_tool(&tools, "shared_memory_write");
        let read_tool = find_tool(&tools, "shared_memory_read");

        let result = write_tool
            .execute(json!({
                "content": "Important finding",
                "category": "fact",
                "tags": ["important"]
            }))
            .await
            .unwrap();
        assert!(!result.is_error);

        let result = read_tool.execute(json!({})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Important finding"));
        assert!(result.content.contains("agent_a")); // provenance
    }

    #[tokio::test]
    async fn read_empty_shared_memory() {
        let (_store, tools) = setup();
        let read_tool = find_tool(&tools, "shared_memory_read");

        let result = read_tool.execute(json!({})).await.unwrap();
        assert_eq!(result.content, "No shared memories found.");
    }

    #[tokio::test]
    async fn shared_memory_visible_to_all_agents() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let tools_a = shared_memory_tools(store.clone(), "agent_a");
        let tools_b = shared_memory_tools(store.clone(), "agent_b");

        // Agent A writes
        let write_a = find_tool(&tools_a, "shared_memory_write");
        write_a
            .execute(json!({"content": "shared from A"}))
            .await
            .unwrap();

        // Agent B can read it
        let read_b = find_tool(&tools_b, "shared_memory_read");
        let result = read_b.execute(json!({})).await.unwrap();
        assert!(result.content.contains("shared from A"));
    }

    #[tokio::test]
    async fn filter_by_agent() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let tools_a = shared_memory_tools(store.clone(), "agent_a");
        let tools_b = shared_memory_tools(store.clone(), "agent_b");

        let write_a = find_tool(&tools_a, "shared_memory_write");
        let write_b = find_tool(&tools_b, "shared_memory_write");

        write_a
            .execute(json!({"content": "data from A"}))
            .await
            .unwrap();
        write_b
            .execute(json!({"content": "data from B"}))
            .await
            .unwrap();

        // Filter by agent_a only
        let read_a = find_tool(&tools_a, "shared_memory_read");
        let result = read_a.execute(json!({"agent": "agent_a"})).await.unwrap();
        assert!(result.content.contains("data from A"));
        assert!(!result.content.contains("data from B"));
    }

    #[tokio::test]
    async fn write_with_keywords_and_summary() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let tools = shared_memory_tools(store.clone(), "agent_a");
        let write_tool = find_tool(&tools, "shared_memory_write");

        write_tool
            .execute(json!({
                "content": "Rust has zero-cost abstractions",
                "keywords": ["rust", "performance", "abstractions"],
                "summary": "Key Rust language feature"
            }))
            .await
            .unwrap();

        // Verify keywords and summary were stored
        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].keywords,
            vec!["rust", "performance", "abstractions"]
        );
        assert_eq!(
            entries[0].summary.as_deref(),
            Some("Key Rust language feature")
        );
    }
}
