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

use super::scoring::{ScoringWeights, composite_score};
use super::{Memory, MemoryEntry, MemoryQuery};

/// Create the 5 memory tools bound to a specific memory store and agent name.
pub fn memory_tools(memory: Arc<dyn Memory>, agent_name: &str) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(MemoryStoreTool {
            memory: memory.clone(),
            agent_name: agent_name.into(),
        }),
        Arc::new(MemoryRecallTool {
            memory: memory.clone(),
            agent_name: agent_name.into(),
        }),
        Arc::new(MemoryUpdateTool {
            memory: memory.clone(),
        }),
        Arc::new(MemoryForgetTool {
            memory: memory.clone(),
        }),
        Arc::new(MemoryConsolidateTool {
            memory,
            agent_name: agent_name.into(),
        }),
    ]
}

// --- memory_store ---

struct MemoryStoreTool {
    memory: Arc<dyn Memory>,
    agent_name: String,
}

#[derive(Deserialize)]
struct StoreInput {
    content: String,
    #[serde(default = "super::default_category")]
    category: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "super::default_importance")]
    importance: u8,
}

impl Tool for MemoryStoreTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_store".into(),
            description: "Store a new memory. Use this to remember important facts, \
                          observations, preferences, or procedures for later recall."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to memorize"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["fact", "observation", "preference", "procedure"],
                        "description": "Category of memory (default: fact)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for organization"
                    },
                    "importance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Importance score 1-10 (default: 5). Higher = more likely to surface in recall."
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
            let input: StoreInput =
                serde_json::from_value(input).map_err(|e| Error::Memory(e.to_string()))?;

            let id = Uuid::new_v4().to_string();
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
            };

            self.memory.store(entry).await?;
            Ok(ToolOutput::success(format!("Stored memory with id: {id}")))
        })
    }
}

// --- memory_recall ---

struct MemoryRecallTool {
    memory: Arc<dyn Memory>,
    agent_name: String,
}

#[derive(Deserialize)]
struct RecallInput {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "super::default_recall_limit")]
    limit: usize,
}

impl Tool for MemoryRecallTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_recall".into(),
            description: "Search and retrieve stored memories. Filter by text query, \
                          category, or tags."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for in memories"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["fact", "observation", "preference", "procedure"],
                        "description": "Filter by category"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (matches any)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 10)"
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
            let input: RecallInput =
                serde_json::from_value(input).map_err(|e| Error::Memory(e.to_string()))?;

            let has_text_query = input.query.is_some();
            let results = self
                .memory
                .recall(MemoryQuery {
                    text: input.query,
                    category: input.category,
                    tags: input.tags,
                    agent: Some(self.agent_name.clone()),
                    limit: input.limit,
                })
                .await?;

            if results.is_empty() {
                return Ok(ToolOutput::success("No memories found."));
            }

            let now = Utc::now();
            let weights = ScoringWeights::default();

            let formatted: Vec<String> = results
                .iter()
                .map(|e| {
                    let relevance = if has_text_query { 1.0 } else { 0.0 };
                    let score =
                        composite_score(&weights, e.created_at, now, e.importance, relevance);
                    let display_content = if e.content.len() > 200 {
                        format!("{}...", &e.content[..200])
                    } else {
                        e.content.clone()
                    };
                    format!(
                        "- [{}] ({}, importance:{}) score:{:.2} {}\n  Tags: {:?} | Accessed: {} times",
                        e.id,
                        e.category,
                        e.importance,
                        score,
                        display_content,
                        e.tags,
                        e.access_count,
                    )
                })
                .collect();

            let count = results.len();
            let noun = if count == 1 { "memory" } else { "memories" };
            Ok(ToolOutput::success(format!(
                "Found {count} {noun}:\n{}",
                formatted.join("\n")
            )))
        })
    }
}

// --- memory_update ---

struct MemoryUpdateTool {
    memory: Arc<dyn Memory>,
}

#[derive(Deserialize)]
struct UpdateInput {
    id: String,
    content: String,
}

impl Tool for MemoryUpdateTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_update".into(),
            description: "Update an existing memory entry by ID with new content.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "ID of the memory to update"
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for the memory"
                    }
                },
                "required": ["id", "content"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: UpdateInput =
                serde_json::from_value(input).map_err(|e| Error::Memory(e.to_string()))?;

            self.memory.update(&input.id, input.content).await?;
            Ok(ToolOutput::success(format!("Updated memory: {}", input.id)))
        })
    }
}

// --- memory_forget ---

struct MemoryForgetTool {
    memory: Arc<dyn Memory>,
}

#[derive(Deserialize)]
struct ForgetInput {
    id: String,
}

impl Tool for MemoryForgetTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_forget".into(),
            description: "Delete a memory entry by ID.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "ID of the memory to delete"
                    }
                },
                "required": ["id"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: ForgetInput =
                serde_json::from_value(input).map_err(|e| Error::Memory(e.to_string()))?;

            let removed = self.memory.forget(&input.id).await?;
            if removed {
                Ok(ToolOutput::success(format!("Deleted memory: {}", input.id)))
            } else {
                Ok(ToolOutput::error(format!("Memory not found: {}", input.id)))
            }
        })
    }
}

// --- memory_consolidate ---

struct MemoryConsolidateTool {
    memory: Arc<dyn Memory>,
    agent_name: String,
}

#[derive(Deserialize)]
struct ConsolidateInput {
    source_ids: Vec<String>,
    content: String,
    #[serde(default = "super::default_category")]
    category: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "super::default_importance")]
    importance: u8,
}

impl Tool for MemoryConsolidateTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "memory_consolidate".into(),
            description: "Consolidate multiple memories into one. Provide the IDs of source \
                          memories to merge and the new consolidated content. Source memories \
                          are deleted and replaced with the new entry."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "description": "IDs of memories to consolidate (minimum 2)"
                    },
                    "content": {
                        "type": "string",
                        "description": "The consolidated content summarizing the source memories"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["fact", "observation", "preference", "procedure"],
                        "description": "Category for the consolidated memory (default: fact)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for the consolidated memory"
                    },
                    "importance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Importance score 1-10 (default: 5)"
                    }
                },
                "required": ["source_ids", "content"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: ConsolidateInput =
                serde_json::from_value(input).map_err(|e| Error::Memory(e.to_string()))?;

            if input.source_ids.len() < 2 {
                return Ok(ToolOutput::error(
                    "Consolidation requires at least 2 source memory IDs.",
                ));
            }

            // Create consolidated entry FIRST to prevent data loss.
            // If store fails, no sources are deleted.
            let new_id = Uuid::new_v4().to_string();
            let now = Utc::now();
            let entry = MemoryEntry {
                id: new_id.clone(),
                agent: self.agent_name.clone(),
                content: input.content,
                category: input.category,
                tags: input.tags,
                created_at: now,
                last_accessed: now,
                access_count: 0,
                importance: input.importance.clamp(1, 10),
            };

            self.memory.store(entry).await?;

            // Delete source memories, track which were found
            let total = input.source_ids.len();
            let mut deleted = 0;
            let mut not_found = Vec::new();
            for id in &input.source_ids {
                match self.memory.forget(id).await? {
                    true => deleted += 1,
                    false => not_found.push(id.clone()),
                }
            }

            if deleted == 0 {
                return Ok(ToolOutput::error(format!(
                    "None of the source memories were found. \
                     Consolidated entry {new_id} was created but no sources were removed."
                )));
            }

            let mut msg = format!("Consolidated {deleted} memories into new memory: {new_id}");
            if !not_found.is_empty() {
                msg.push_str(&format!(
                    "\nWarning: {} of {} source memories not found: {:?}",
                    not_found.len(),
                    total,
                    not_found
                ));
            }
            Ok(ToolOutput::success(msg))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::in_memory::InMemoryStore;

    fn setup() -> (Arc<dyn Memory>, Vec<Arc<dyn Tool>>) {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let tools = memory_tools(store.clone(), "test-agent");
        (store, tools)
    }

    fn find_tool<'a>(tools: &'a [Arc<dyn Tool>], name: &str) -> &'a Arc<dyn Tool> {
        tools
            .iter()
            .find(|t| t.definition().name == name)
            .unwrap_or_else(|| panic!("tool {name} not found"))
    }

    #[test]
    fn creates_five_tools() {
        let (_store, tools) = setup();
        assert_eq!(tools.len(), 5);

        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert!(names.contains(&"memory_store".to_string()));
        assert!(names.contains(&"memory_recall".to_string()));
        assert!(names.contains(&"memory_update".to_string()));
        assert!(names.contains(&"memory_forget".to_string()));
        assert!(names.contains(&"memory_consolidate".to_string()));
    }

    #[tokio::test]
    async fn store_tool_creates_memory() {
        let (store, tools) = setup();
        let tool = find_tool(&tools, "memory_store");

        let result = tool
            .execute(json!({
                "content": "Rust is memory-safe",
                "category": "fact",
                "tags": ["rust", "safety"]
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Stored memory with id:"));

        // Verify it's in the store
        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "Rust is memory-safe");
        assert_eq!(entries[0].agent, "test-agent");
    }

    #[tokio::test]
    async fn recall_tool_finds_memories() {
        let (_store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let recall_tool = find_tool(&tools, "memory_recall");

        // Store some memories
        store_tool
            .execute(json!({"content": "Rust is fast", "category": "fact"}))
            .await
            .unwrap();
        store_tool
            .execute(json!({"content": "Python is slow", "category": "observation"}))
            .await
            .unwrap();

        // Recall all
        let result = recall_tool.execute(json!({})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Found 2 memories"));

        // Recall by query
        let result = recall_tool.execute(json!({"query": "rust"})).await.unwrap();
        assert!(result.content.contains("Found 1 memory:"));
        assert!(result.content.contains("Rust is fast"));
    }

    #[tokio::test]
    async fn recall_tool_empty_result() {
        let (_store, tools) = setup();
        let recall_tool = find_tool(&tools, "memory_recall");

        let result = recall_tool.execute(json!({})).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content, "No memories found.");
    }

    #[tokio::test]
    async fn update_tool_modifies_memory() {
        let (store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let update_tool = find_tool(&tools, "memory_update");

        store_tool
            .execute(json!({"content": "original"}))
            .await
            .unwrap();

        // Get the ID
        let entries = store
            .recall(MemoryQuery {
                limit: 1,
                ..Default::default()
            })
            .await
            .unwrap();
        let id = &entries[0].id;

        let result = update_tool
            .execute(json!({"id": id, "content": "updated"}))
            .await
            .unwrap();
        assert!(!result.is_error);

        let entries = store
            .recall(MemoryQuery {
                limit: 1,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries[0].content, "updated");
    }

    #[tokio::test]
    async fn forget_tool_deletes_memory() {
        let (store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let forget_tool = find_tool(&tools, "memory_forget");

        store_tool
            .execute(json!({"content": "to delete"}))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 1,
                ..Default::default()
            })
            .await
            .unwrap();
        let id = &entries[0].id;

        let result = forget_tool.execute(json!({"id": id})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Deleted"));

        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn forget_tool_nonexistent() {
        let (_store, tools) = setup();
        let forget_tool = find_tool(&tools, "memory_forget");

        let result = forget_tool
            .execute(json!({"id": "nonexistent"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    // --- importance tests ---

    #[tokio::test]
    async fn store_tool_default_importance() {
        let (store, tools) = setup();
        let tool = find_tool(&tools, "memory_store");

        tool.execute(json!({"content": "test"})).await.unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 1,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries[0].importance, 5);
    }

    #[tokio::test]
    async fn store_tool_custom_importance() {
        let (store, tools) = setup();
        let tool = find_tool(&tools, "memory_store");

        tool.execute(json!({"content": "critical", "importance": 9}))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 1,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries[0].importance, 9);
    }

    #[tokio::test]
    async fn store_tool_clamps_importance() {
        let (store, tools) = setup();
        let tool = find_tool(&tools, "memory_store");

        // Value > 10 should clamp to 10
        tool.execute(json!({"content": "over", "importance": 15}))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 1,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries[0].importance, 10);
    }

    // --- recall output format tests ---

    #[tokio::test]
    async fn recall_tool_shows_score() {
        let (_store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let recall_tool = find_tool(&tools, "memory_recall");

        store_tool
            .execute(json!({"content": "scored memory"}))
            .await
            .unwrap();

        let result = recall_tool.execute(json!({})).await.unwrap();
        assert!(result.content.contains("score:"));
    }

    #[tokio::test]
    async fn recall_tool_shows_importance() {
        let (_store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let recall_tool = find_tool(&tools, "memory_recall");

        store_tool
            .execute(json!({"content": "important thing", "importance": 8}))
            .await
            .unwrap();

        let result = recall_tool.execute(json!({})).await.unwrap();
        assert!(result.content.contains("importance:8"));
    }

    // --- consolidation tests ---

    #[tokio::test]
    async fn consolidate_tool_merges_memories() {
        let (store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let consolidate_tool = find_tool(&tools, "memory_consolidate");

        store_tool
            .execute(json!({"content": "fact A"}))
            .await
            .unwrap();
        store_tool
            .execute(json!({"content": "fact B"}))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries.len(), 2);
        let id_a = entries[0].id.clone();
        let id_b = entries[1].id.clone();

        let result = consolidate_tool
            .execute(json!({
                "source_ids": [id_a, id_b],
                "content": "Combined A and B",
                "category": "fact"
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Consolidated 2 memories"));

        // Should now have 1 entry
        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "Combined A and B");
    }

    #[tokio::test]
    async fn consolidate_tool_requires_minimum_two() {
        let (_store, tools) = setup();
        let consolidate_tool = find_tool(&tools, "memory_consolidate");

        let result = consolidate_tool
            .execute(json!({
                "source_ids": ["only-one"],
                "content": "merged"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("at least 2"));
    }

    #[tokio::test]
    async fn consolidate_tool_partial_not_found() {
        let (store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let consolidate_tool = find_tool(&tools, "memory_consolidate");

        store_tool
            .execute(json!({"content": "exists"}))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 1,
                ..Default::default()
            })
            .await
            .unwrap();
        let real_id = entries[0].id.clone();

        let result = consolidate_tool
            .execute(json!({
                "source_ids": [real_id, "nonexistent-id"],
                "content": "partial consolidation"
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Consolidated 1 memories"));
        assert!(result.content.contains("Warning"));
        assert!(result.content.contains("nonexistent-id"));
    }

    #[tokio::test]
    async fn consolidate_tool_all_not_found() {
        let (_store, tools) = setup();
        let consolidate_tool = find_tool(&tools, "memory_consolidate");

        let result = consolidate_tool
            .execute(json!({
                "source_ids": ["fake1", "fake2"],
                "content": "nope"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result
                .content
                .contains("None of the source memories were found")
        );
    }

    #[tokio::test]
    async fn consolidate_tool_preserves_importance() {
        let (store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let consolidate_tool = find_tool(&tools, "memory_consolidate");

        store_tool
            .execute(json!({"content": "a", "importance": 3}))
            .await
            .unwrap();
        store_tool
            .execute(json!({"content": "b", "importance": 7}))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        let id_a = entries[0].id.clone();
        let id_b = entries[1].id.clone();

        consolidate_tool
            .execute(json!({
                "source_ids": [id_a, id_b],
                "content": "merged",
                "importance": 9
            }))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 1,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries[0].importance, 9);
    }

    #[tokio::test]
    async fn consolidate_tool_default_importance() {
        let (store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let consolidate_tool = find_tool(&tools, "memory_consolidate");

        store_tool.execute(json!({"content": "x"})).await.unwrap();
        store_tool.execute(json!({"content": "y"})).await.unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        let id_x = entries[0].id.clone();
        let id_y = entries[1].id.clone();

        consolidate_tool
            .execute(json!({
                "source_ids": [id_x, id_y],
                "content": "consolidated"
            }))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 1,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(entries[0].importance, 5); // default
    }
}
