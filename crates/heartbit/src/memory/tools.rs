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

use super::reflection::ReflectionTracker;
use super::{Memory, MemoryEntry, MemoryQuery};

/// Create the 5 memory tools bound to a specific memory store and agent name.
///
/// If `reflection_threshold` is set, the store tool will include a reflection
/// hint when cumulative importance exceeds the threshold.
pub fn memory_tools_with_reflection(
    memory: Arc<dyn Memory>,
    agent_name: &str,
    reflection_threshold: Option<u32>,
) -> Vec<Arc<dyn Tool>> {
    let tracker = reflection_threshold.map(|t| Arc::new(ReflectionTracker::new(t)));
    vec![
        Arc::new(MemoryStoreTool {
            memory: memory.clone(),
            agent_name: agent_name.into(),
            reflection_tracker: tracker,
        }),
        Arc::new(MemoryRecallTool {
            memory: memory.clone(),
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
    reflection_tracker: Option<Arc<ReflectionTracker>>,
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
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    summary: Option<String>,
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
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords for improved retrieval (BM25 scoring). Provide 3-5 key terms."
                    },
                    "summary": {
                        "type": "string",
                        "description": "One-sentence summary providing context for this memory."
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
                memory_type: super::MemoryType::default(),
                keywords: input.keywords,
                summary: input.summary,
                strength: 1.0,
                related_ids: vec![],
                source_ids: vec![],
                embedding: None,
            };

            let importance = entry.importance;
            let keywords = entry.keywords.clone();
            self.memory.store(entry).await?;

            // Link evolution: find related entries by keyword overlap
            if !keywords.is_empty()
                && let Ok(existing) = self
                    .memory
                    .recall(MemoryQuery {
                        limit: 20,
                        ..Default::default()
                    })
                    .await
            {
                for e in &existing {
                    if e.id == id || e.keywords.is_empty() {
                        continue;
                    }
                    let jaccard = super::consolidation::jaccard_similarity(&keywords, &e.keywords);
                    if jaccard >= 0.2 {
                        let _ = self.memory.add_link(&id, &e.id).await;
                    }
                }
            }

            let mut msg = format!("Stored memory with id: {id}");
            if let Some(ref tracker) = self.reflection_tracker
                && tracker.record(importance)
            {
                msg.push_str(super::reflection::REFLECTION_HINT);
            }
            Ok(ToolOutput::success(msg))
        })
    }
}

// --- memory_recall ---

struct MemoryRecallTool {
    memory: Arc<dyn Memory>,
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

            let results = self
                .memory
                .recall(MemoryQuery {
                    text: input.query,
                    category: input.category,
                    tags: input.tags,
                    // agent: None lets NamespacedMemory default to the correct
                    // compound namespace (e.g. "tg:123:assistant"). Passing the
                    // plain agent_name would bypass NamespacedMemory's scoping.
                    agent: None,
                    limit: input.limit,
                    ..Default::default()
                })
                .await?;

            if results.is_empty() {
                return Ok(ToolOutput::success("No memories found."));
            }

            // Results are pre-sorted by the store using its configured scoring
            // weights. We display rank order rather than recomputing scores
            // (which may use different weights than the store).
            let formatted: Vec<String> = results
                .iter()
                .enumerate()
                .map(|(rank, e)| {
                    let display_content = if e.content.len() > 200 {
                        let truncated: String = e.content.chars().take(200).collect();
                        format!("{truncated}...")
                    } else {
                        e.content.clone()
                    };
                    let mt = match e.memory_type {
                        crate::memory::MemoryType::Episodic => "episodic",
                        crate::memory::MemoryType::Semantic => "semantic",
                        crate::memory::MemoryType::Reflection => "reflection",
                    };
                    let mut line = format!(
                        "- #{} [{}] ({}, {}, importance:{}, strength:{:.2}) {}\n  Tags: {:?} | Accessed: {} times",
                        rank + 1,
                        e.id,
                        e.category,
                        mt,
                        e.importance,
                        e.strength,
                        display_content,
                        e.tags,
                        e.access_count,
                    );
                    if !e.keywords.is_empty() {
                        line.push_str(&format!(" | Keywords: {:?}", e.keywords));
                    }
                    line
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

            // Fetch source memories to merge their keywords and tags.
            // Use a generous limit — consolidation is not a hot path.
            let sources = self
                .memory
                .recall(MemoryQuery {
                    limit: 1000,
                    ..Default::default()
                })
                .await
                .unwrap_or_default();

            let source_set: std::collections::HashSet<&str> =
                input.source_ids.iter().map(|s| s.as_str()).collect();

            let mut merged_keywords: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            let mut merged_tags: std::collections::HashSet<String> =
                input.tags.iter().cloned().collect();

            for entry in &sources {
                if source_set.contains(entry.id.as_str()) {
                    merged_keywords.extend(entry.keywords.iter().cloned());
                    merged_tags.extend(entry.tags.iter().cloned());
                }
            }

            let keywords: Vec<String> = merged_keywords.into_iter().collect();
            let tags: Vec<String> = merged_tags.into_iter().collect();

            // Create consolidated entry FIRST to prevent data loss.
            // If store fails, no sources are deleted.
            let new_id = Uuid::new_v4().to_string();
            let now = Utc::now();
            let entry = MemoryEntry {
                id: new_id.clone(),
                agent: self.agent_name.clone(),
                content: input.content,
                category: input.category,
                tags,
                created_at: now,
                last_accessed: now,
                access_count: 0,
                importance: input.importance.clamp(1, 10),
                memory_type: super::MemoryType::Semantic,
                keywords,
                summary: None,
                strength: 1.0,
                related_ids: vec![],
                source_ids: input.source_ids.clone(),
                embedding: None,
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
                // Clean up the orphaned consolidated entry (best-effort)
                if let Err(e) = self.memory.forget(&new_id).await {
                    tracing::warn!(id = %new_id, error = %e, "failed to clean up orphaned consolidation entry");
                }
                return Ok(ToolOutput::error(
                    "None of the source memories were found. No consolidation performed.",
                ));
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
        let tools = memory_tools_with_reflection(store.clone(), "test-agent", None);
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
    async fn recall_tool_shows_rank() {
        let (_store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let recall_tool = find_tool(&tools, "memory_recall");

        store_tool
            .execute(json!({"content": "first memory"}))
            .await
            .unwrap();
        store_tool
            .execute(json!({"content": "second memory"}))
            .await
            .unwrap();

        let result = recall_tool.execute(json!({})).await.unwrap();
        assert!(result.content.contains("#1"), "should show rank #1");
        assert!(result.content.contains("#2"), "should show rank #2");
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

    #[tokio::test]
    async fn recall_tool_truncates_long_content_safely() {
        let (_store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let recall_tool = find_tool(&tools, "memory_recall");

        // Content with multi-byte UTF-8 characters (each emoji is 4 bytes)
        let content = "🦀".repeat(100); // 400 bytes, 100 chars
        store_tool
            .execute(json!({"content": content}))
            .await
            .unwrap();

        // Should not panic on non-ASCII content
        let result = recall_tool.execute(json!({})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Found 1 memory"));
    }

    #[tokio::test]
    async fn recall_tool_truncates_very_long_content() {
        let (_store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let recall_tool = find_tool(&tools, "memory_recall");

        // 500 ASCII chars — should be truncated to 200 + "..."
        let content = "a".repeat(500);
        store_tool
            .execute(json!({"content": content}))
            .await
            .unwrap();

        let result = recall_tool.execute(json!({})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("..."));
        // Should NOT contain all 500 'a's
        assert!(!result.content.contains(&"a".repeat(500)));
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
        let (store, tools) = setup();
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

        // Verify the orphaned consolidated entry was cleaned up
        let all = store
            .recall(MemoryQuery {
                limit: 100,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(all.is_empty(), "orphaned entry should have been cleaned up");
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

    #[tokio::test]
    async fn consolidate_tool_merges_source_keywords_and_tags() {
        let (store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let consolidate_tool = find_tool(&tools, "memory_consolidate");

        store_tool
            .execute(json!({
                "content": "first fact",
                "keywords": ["rust", "performance"],
                "tags": ["lang"]
            }))
            .await
            .unwrap();
        store_tool
            .execute(json!({
                "content": "second fact",
                "keywords": ["safety", "rust"],
                "tags": ["design"]
            }))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        let ids: Vec<String> = entries.iter().map(|e| e.id.clone()).collect();

        consolidate_tool
            .execute(json!({
                "source_ids": ids,
                "content": "merged fact",
                "tags": ["summary"]
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
        assert_eq!(entries.len(), 1);

        // Keywords merged from both sources (deduplicated)
        let kw = &entries[0].keywords;
        assert!(kw.contains(&"rust".to_string()));
        assert!(kw.contains(&"performance".to_string()));
        assert!(kw.contains(&"safety".to_string()));

        // Tags merged from sources + explicit input
        let tags = &entries[0].tags;
        assert!(tags.contains(&"summary".to_string()));
        assert!(tags.contains(&"lang".to_string()));
        assert!(tags.contains(&"design".to_string()));
    }

    // --- keywords and summary tests ---

    #[tokio::test]
    async fn store_tool_accepts_keywords() {
        let (store, tools) = setup();
        let tool = find_tool(&tools, "memory_store");

        tool.execute(json!({
            "content": "Rust has zero-cost abstractions",
            "keywords": ["rust", "zero-cost", "abstractions"]
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
        assert_eq!(
            entries[0].keywords,
            vec!["rust", "zero-cost", "abstractions"]
        );
    }

    #[tokio::test]
    async fn store_tool_accepts_summary() {
        let (store, tools) = setup();
        let tool = find_tool(&tools, "memory_store");

        tool.execute(json!({
            "content": "Detailed technical analysis of Rust ownership model",
            "summary": "Rust ownership analysis"
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
        assert_eq!(
            entries[0].summary.as_deref(),
            Some("Rust ownership analysis")
        );
    }

    #[tokio::test]
    async fn store_tool_keywords_improve_recall() {
        let (_store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let recall_tool = find_tool(&tools, "memory_recall");

        // Store with keyword "performance" not in content
        store_tool
            .execute(json!({
                "content": "Rust is great for systems programming",
                "keywords": ["performance", "speed", "systems"]
            }))
            .await
            .unwrap();

        // Search for "performance" — should find via keyword match
        let result = recall_tool
            .execute(json!({"query": "performance"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("Rust is great"),
            "keywords should enable finding the memory"
        );
    }

    #[tokio::test]
    async fn recall_tool_shows_keywords_in_output() {
        let (_store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");
        let recall_tool = find_tool(&tools, "memory_recall");

        store_tool
            .execute(json!({
                "content": "test content",
                "keywords": ["test-keyword"]
            }))
            .await
            .unwrap();

        let result = recall_tool.execute(json!({})).await.unwrap();
        assert!(
            result.content.contains("test-keyword"),
            "recall output should show keywords"
        );
    }

    // --- reflection tests ---

    fn setup_with_reflection(threshold: u32) -> (Arc<dyn Memory>, Vec<Arc<dyn Tool>>) {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let tools = memory_tools_with_reflection(store.clone(), "test-agent", Some(threshold));
        (store, tools)
    }

    #[tokio::test]
    async fn store_includes_reflection_hint_when_triggered() {
        let (_store, tools) = setup_with_reflection(10);
        let store_tool = find_tool(&tools, "memory_store");

        // Store with importance 10 — should trigger immediately
        let result = store_tool
            .execute(json!({"content": "very important", "importance": 10}))
            .await
            .unwrap();
        assert!(
            result.content.contains("Reflection suggested"),
            "should include reflection hint"
        );
    }

    #[tokio::test]
    async fn store_no_reflection_below_threshold() {
        let (_store, tools) = setup_with_reflection(20);
        let store_tool = find_tool(&tools, "memory_store");

        let result = store_tool
            .execute(json!({"content": "minor fact", "importance": 3}))
            .await
            .unwrap();
        assert!(
            !result.content.contains("Reflection suggested"),
            "should not trigger reflection below threshold"
        );
    }

    #[tokio::test]
    async fn store_reflection_accumulates() {
        let (_store, tools) = setup_with_reflection(10);
        let store_tool = find_tool(&tools, "memory_store");

        // importance 5 + 5 = 10, second should trigger
        let r1 = store_tool
            .execute(json!({"content": "fact A", "importance": 5}))
            .await
            .unwrap();
        assert!(!r1.content.contains("Reflection suggested"));

        let r2 = store_tool
            .execute(json!({"content": "fact B", "importance": 5}))
            .await
            .unwrap();
        assert!(
            r2.content.contains("Reflection suggested"),
            "should trigger after accumulation"
        );
    }

    // --- linking tests ---

    #[tokio::test]
    async fn store_creates_links_for_keyword_overlap() {
        let (store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");

        // Store first entry with keywords
        store_tool
            .execute(json!({
                "content": "Rust is fast",
                "keywords": ["rust", "performance", "speed"]
            }))
            .await
            .unwrap();

        // Store second entry with overlapping keywords
        store_tool
            .execute(json!({
                "content": "Rust has great perf",
                "keywords": ["rust", "performance"]
            }))
            .await
            .unwrap();

        // Check that entries are linked
        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();

        let has_links = entries.iter().any(|e| !e.related_ids.is_empty());
        assert!(
            has_links,
            "entries with overlapping keywords should be linked"
        );
    }

    #[tokio::test]
    async fn store_no_links_without_keywords() {
        let (store, tools) = setup();
        let store_tool = find_tool(&tools, "memory_store");

        store_tool
            .execute(json!({"content": "no keywords A"}))
            .await
            .unwrap();
        store_tool
            .execute(json!({"content": "no keywords B"}))
            .await
            .unwrap();

        let entries = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();

        let has_links = entries.iter().any(|e| !e.related_ids.is_empty());
        assert!(!has_links, "entries without keywords should not be linked");
    }
}
