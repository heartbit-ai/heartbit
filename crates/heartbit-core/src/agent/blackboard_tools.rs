use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

use super::blackboard::Blackboard;

/// Create blackboard tools for sub-agent access to the shared key-value store.
///
/// Returns 3 tools:
/// - `blackboard_read` — read a key
/// - `blackboard_write` — write a key-value pair
/// - `blackboard_list` — list all keys
///
/// Note: `Blackboard::clear()` is intentionally not exposed as a tool to prevent
/// sub-agents from wiping shared coordination data.
pub fn blackboard_tools(blackboard: Arc<dyn Blackboard>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(BlackboardReadTool {
            blackboard: blackboard.clone(),
        }),
        Arc::new(BlackboardWriteTool {
            blackboard: blackboard.clone(),
        }),
        Arc::new(BlackboardListTool { blackboard }),
    ]
}

// --- blackboard_read ---

struct BlackboardReadTool {
    blackboard: Arc<dyn Blackboard>,
}

#[derive(Deserialize)]
struct ReadInput {
    key: String,
}

impl Tool for BlackboardReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "blackboard_read".into(),
            description: "Read a value from the shared blackboard by key. Use this to access \
                          results from other agents or previously stored coordination data."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to read (e.g. 'agent:researcher' for agent results)"
                    }
                },
                "required": ["key"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: ReadInput =
                serde_json::from_value(input).map_err(|e| Error::Agent(e.to_string()))?;

            match self.blackboard.read(&input.key).await? {
                Some(serde_json::Value::String(s)) => Ok(ToolOutput::success(s)),
                Some(value) => {
                    let text =
                        serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
                    Ok(ToolOutput::success(text))
                }
                None => Ok(ToolOutput::success(format!(
                    "Key '{}' not found on blackboard.",
                    input.key
                ))),
            }
        })
    }
}

// --- blackboard_write ---

struct BlackboardWriteTool {
    blackboard: Arc<dyn Blackboard>,
}

#[derive(Deserialize)]
struct WriteInput {
    key: String,
    value: serde_json::Value,
}

impl Tool for BlackboardWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "blackboard_write".into(),
            description: "Write a key-value pair to the shared blackboard. Use this to store \
                          intermediate results or data for other agents to consume."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to write"
                    },
                    "value": {
                        "description": "The JSON value to store"
                    }
                },
                "required": ["key", "value"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: WriteInput =
                serde_json::from_value(input).map_err(|e| Error::Agent(e.to_string()))?;

            self.blackboard.write(&input.key, input.value).await?;
            Ok(ToolOutput::success(format!(
                "Written to blackboard key '{}'.",
                input.key
            )))
        })
    }
}

// --- blackboard_list ---

struct BlackboardListTool {
    blackboard: Arc<dyn Blackboard>,
}

impl Tool for BlackboardListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "blackboard_list".into(),
            description: "List all keys currently on the shared blackboard. Use this to discover \
                          what data is available from other agents."
                .into(),
            input_schema: json!({
                "type": "object"
            }),
        }
    }

    fn execute(
        &self,
        _input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let keys = self.blackboard.list_keys().await?;
            if keys.is_empty() {
                return Ok(ToolOutput::success("Blackboard is empty."));
            }
            let formatted = keys
                .iter()
                .map(|k| format!("- {k}"))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(ToolOutput::success(format!(
                "Blackboard keys ({}):\n{}",
                keys.len(),
                formatted
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::blackboard::InMemoryBlackboard;

    fn setup() -> (Arc<dyn Blackboard>, Vec<Arc<dyn Tool>>) {
        let bb: Arc<dyn Blackboard> = Arc::new(InMemoryBlackboard::new());
        let tools = blackboard_tools(bb.clone());
        (bb, tools)
    }

    fn find_tool<'a>(tools: &'a [Arc<dyn Tool>], name: &str) -> &'a Arc<dyn Tool> {
        tools
            .iter()
            .find(|t| t.definition().name == name)
            .unwrap_or_else(|| panic!("tool {name} not found"))
    }

    #[test]
    fn creates_three_tools() {
        let (_bb, tools) = setup();
        assert_eq!(tools.len(), 3);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert!(names.contains(&"blackboard_read".to_string()));
        assert!(names.contains(&"blackboard_write".to_string()));
        assert!(names.contains(&"blackboard_list".to_string()));
    }

    #[tokio::test]
    async fn read_tool_returns_value() {
        let (bb, tools) = setup();
        bb.write("test-key", json!({"data": "hello"}))
            .await
            .unwrap();

        let read = find_tool(&tools, "blackboard_read");
        let result = read.execute(json!({"key": "test-key"})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
    }

    #[tokio::test]
    async fn read_tool_returns_not_found() {
        let (_bb, tools) = setup();

        let read = find_tool(&tools, "blackboard_read");
        let result = read.execute(json!({"key": "missing"})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn write_tool_stores_value() {
        let (bb, tools) = setup();

        let write = find_tool(&tools, "blackboard_write");
        let result = write
            .execute(json!({"key": "my-key", "value": {"result": 42}}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("my-key"));

        // Verify via blackboard directly
        let val = bb.read("my-key").await.unwrap();
        assert_eq!(val, Some(json!({"result": 42})));
    }

    #[tokio::test]
    async fn list_tool_returns_keys() {
        let (bb, tools) = setup();
        bb.write("agent:alpha", json!("result-a")).await.unwrap();
        bb.write("agent:beta", json!("result-b")).await.unwrap();

        let list = find_tool(&tools, "blackboard_list");
        let result = list.execute(json!({})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("agent:alpha"));
        assert!(result.content.contains("agent:beta"));
        assert!(result.content.contains("2")); // count
    }

    #[tokio::test]
    async fn list_tool_returns_empty_message() {
        let (_bb, tools) = setup();

        let list = find_tool(&tools, "blackboard_list");
        let result = list.execute(json!({})).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content, "Blackboard is empty.");
    }

    #[tokio::test]
    async fn read_tool_returns_plain_text_for_strings() {
        let (bb, tools) = setup();
        // Simulate auto-written agent result (stored as Value::String)
        bb.write("agent:researcher", json!("Research findings here."))
            .await
            .unwrap();

        let read = find_tool(&tools, "blackboard_read");
        let result = read
            .execute(json!({"key": "agent:researcher"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        // Should be plain text, NOT JSON-quoted like "\"Research findings here.\""
        assert_eq!(result.content, "Research findings here.");
    }

    #[tokio::test]
    async fn read_tool_pretty_prints_structured_values() {
        let (bb, tools) = setup();
        bb.write("data", json!({"count": 42, "items": ["a", "b"]}))
            .await
            .unwrap();

        let read = find_tool(&tools, "blackboard_read");
        let result = read.execute(json!({"key": "data"})).await.unwrap();
        assert!(!result.is_error);
        // Should be pretty-printed JSON for structured data
        assert!(result.content.contains("\"count\": 42"));
        assert!(result.content.contains("\"items\""));
    }

    #[tokio::test]
    async fn read_tool_rejects_missing_key() {
        let (_bb, tools) = setup();
        let read = find_tool(&tools, "blackboard_read");
        let result = read.execute(json!({})).await;
        assert!(result.is_err(), "should fail on missing required 'key'");
    }

    #[tokio::test]
    async fn write_tool_rejects_missing_fields() {
        let (_bb, tools) = setup();
        let write = find_tool(&tools, "blackboard_write");

        // Missing both key and value
        let result = write.execute(json!({})).await;
        assert!(result.is_err(), "should fail on missing required fields");

        // Missing value
        let result = write.execute(json!({"key": "k"})).await;
        assert!(result.is_err(), "should fail on missing 'value'");
    }

    #[test]
    fn tool_definitions_have_valid_schemas() {
        let (_bb, tools) = setup();
        for tool in &tools {
            let def = tool.definition();
            assert!(!def.name.is_empty());
            assert!(!def.description.is_empty());
            // Schema should be a JSON object with "type"
            assert!(def.input_schema.is_object(), "tool {} schema", def.name);
            assert_eq!(
                def.input_schema["type"], "object",
                "tool {} schema type",
                def.name
            );
        }
    }
}
