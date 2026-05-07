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
/// - `blackboard_write` — write a key-value pair (caller-namespaced; `agent:` prefix is reserved)
/// - `blackboard_list` — list all keys
///
/// SECURITY (F-AGENT-7): the write tool is caller-namespaced. A sub-agent
/// named `worker` that writes key `notes` actually writes to
/// `caller:worker/notes`. The `agent:` prefix is reserved for the
/// orchestrator's automatic per-agent result writes; sub-agents cannot
/// shadow each other's results by writing `agent:other_worker`. The read
/// tool stays unnamespaced so sub-agents can read peers' published results.
pub fn blackboard_tools(blackboard: Arc<dyn Blackboard>, caller: &str) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(BlackboardReadTool {
            blackboard: blackboard.clone(),
        }),
        Arc::new(BlackboardWriteTool {
            blackboard: blackboard.clone(),
            caller: caller.to_string(),
        }),
        Arc::new(BlackboardListTool { blackboard }),
    ]
}

/// Reserved key prefix for orchestrator-managed per-agent result entries.
/// Sub-agents cannot write keys starting with this prefix via the
/// `blackboard_write` tool (F-AGENT-7).
const RESERVED_AGENT_PREFIX: &str = "agent:";

/// Compose a caller-scoped namespace from the caller agent name.
fn caller_namespace(caller: &str) -> String {
    format!("caller:{caller}/")
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
        _ctx: &crate::ExecutionContext,
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
    caller: String,
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
                          intermediate results or data for other agents to consume. \
                          Note: keys starting with 'agent:' are reserved for the orchestrator."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to write (will be namespaced under your agent name)"
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
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let input: WriteInput =
                serde_json::from_value(input).map_err(|e| Error::Agent(e.to_string()))?;

            // SECURITY (F-AGENT-7): refuse writes to the reserved orchestrator
            // prefix. Without this, a compromised sub-agent could write
            // `agent:other_worker = {fake_result}` and shadow another peer's
            // legitimate result that the orchestrator later reads back.
            if input.key.starts_with(RESERVED_AGENT_PREFIX) {
                return Ok(ToolOutput::error(format!(
                    "key prefix '{RESERVED_AGENT_PREFIX}' is reserved for the orchestrator; \
                     pick a different key (it will be namespaced under your agent name)"
                )));
            }

            // Caller-namespacing: keep the user-visible key intact for
            // discovery via blackboard_list, but prepend the caller scope so
            // two agents writing the same logical key don't collide.
            let key = format!("{}{}", caller_namespace(&self.caller), input.key);
            self.blackboard.write(&key, input.value).await?;
            Ok(ToolOutput::success(format!(
                "Written to blackboard key '{key}'."
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
        _ctx: &crate::ExecutionContext,
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
        let tools = blackboard_tools(bb.clone(), "test_agent");
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
        let result = read
            .execute(
                &crate::ExecutionContext::default(),
                json!({"key": "test-key"}),
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
    }

    #[tokio::test]
    async fn read_tool_returns_not_found() {
        let (_bb, tools) = setup();

        let read = find_tool(&tools, "blackboard_read");
        let result = read
            .execute(
                &crate::ExecutionContext::default(),
                json!({"key": "missing"}),
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn write_tool_stores_value() {
        let (bb, tools) = setup();

        let write = find_tool(&tools, "blackboard_write");
        let result = write
            .execute(
                &crate::ExecutionContext::default(),
                json!({"key": "my-key", "value": {"result": 42}}),
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("my-key"));

        // SECURITY (F-AGENT-7): the write must be caller-namespaced.
        // The plain key "my-key" should NOT be a top-level entry; the actual
        // entry is under "caller:test_agent/my-key".
        let plain = bb.read("my-key").await.unwrap();
        assert!(plain.is_none(), "plain key must not be the storage key");
        let scoped = bb.read("caller:test_agent/my-key").await.unwrap();
        assert_eq!(scoped, Some(json!({"result": 42})));
    }

    /// SECURITY (F-AGENT-7): a sub-agent must NOT be able to write keys with
    /// the reserved `agent:` prefix — that would let it shadow another peer's
    /// orchestrator-managed result entry.
    #[tokio::test]
    async fn write_tool_refuses_reserved_agent_prefix() {
        let (bb, tools) = setup();
        let write = find_tool(&tools, "blackboard_write");

        let result = write
            .execute(
                &crate::ExecutionContext::default(),
                json!({"key": "agent:other_worker", "value": "fake_result"}),
            )
            .await
            .unwrap();
        assert!(result.is_error, "agent: prefix must be rejected");
        assert!(
            result.content.contains("reserved"),
            "error should mention reservation: {}",
            result.content
        );
        // Storage must remain empty for that key.
        let val = bb.read("agent:other_worker").await.unwrap();
        assert!(val.is_none(), "blackboard must not have been written");
    }

    #[tokio::test]
    async fn list_tool_returns_keys() {
        let (bb, tools) = setup();
        bb.write("agent:alpha", json!("result-a")).await.unwrap();
        bb.write("agent:beta", json!("result-b")).await.unwrap();

        let list = find_tool(&tools, "blackboard_list");
        let result = list
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("agent:alpha"));
        assert!(result.content.contains("agent:beta"));
        assert!(result.content.contains("2")); // count
    }

    #[tokio::test]
    async fn list_tool_returns_empty_message() {
        let (_bb, tools) = setup();

        let list = find_tool(&tools, "blackboard_list");
        let result = list
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await
            .unwrap();
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
            .execute(
                &crate::ExecutionContext::default(),
                json!({"key": "agent:researcher"}),
            )
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
        let result = read
            .execute(&crate::ExecutionContext::default(), json!({"key": "data"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        // Should be pretty-printed JSON for structured data
        assert!(result.content.contains("\"count\": 42"));
        assert!(result.content.contains("\"items\""));
    }

    #[tokio::test]
    async fn read_tool_rejects_missing_key() {
        let (_bb, tools) = setup();
        let read = find_tool(&tools, "blackboard_read");
        let result = read
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await;
        assert!(result.is_err(), "should fail on missing required 'key'");
    }

    #[tokio::test]
    async fn write_tool_rejects_missing_fields() {
        let (_bb, tools) = setup();
        let write = find_tool(&tools, "blackboard_write");

        // Missing both key and value
        let result = write
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await;
        assert!(result.is_err(), "should fail on missing required fields");

        // Missing value
        let result = write
            .execute(&crate::ExecutionContext::default(), json!({"key": "k"}))
            .await;
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
