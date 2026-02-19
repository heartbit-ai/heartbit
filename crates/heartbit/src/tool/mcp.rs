use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const PROTOCOL_VERSION: &str = "2025-03-26";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

// --- JSON-RPC types ---

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
    id: u64,
}

#[derive(Debug, Serialize)]
struct JsonRpcNotification {
    jsonrpc: &'static str,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

// --- MCP types ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct McpToolDef {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct McpToolsListResult {
    tools: Vec<McpToolDef>,
}

#[derive(Debug, Deserialize)]
struct McpContent {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct McpCallToolResult {
    content: Vec<McpContent>,
    #[serde(default)]
    is_error: bool,
}

// --- Pure helper functions ---

/// Parse all SSE data payloads from a `text/event-stream` body.
///
/// Handles multi-line `data:` concatenation per the SSE spec and
/// returns all events in order. Use `find_rpc_response` to locate the
/// JSON-RPC response matching a specific request ID.
fn extract_sse_events(body: &str) -> Result<Vec<String>, Error> {
    let mut events: Vec<String> = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();

    for line in body.lines() {
        if line.trim().is_empty() {
            // Blank line = end of event
            if !current_lines.is_empty() {
                events.push(current_lines.join("\n"));
                current_lines.clear();
            }
        } else if let Some(rest) = line.strip_prefix("data:") {
            // SSE spec: strip exactly one leading space after the colon
            let data = rest.strip_prefix(' ').unwrap_or(rest);
            current_lines.push(data);
        }
    }

    // Handle body with no trailing blank line
    if !current_lines.is_empty() {
        events.push(current_lines.join("\n"));
    }

    if events.is_empty() {
        return Err(Error::Mcp("No data field in SSE response".into()));
    }
    Ok(events)
}

/// Find the JSON-RPC response matching `expected_id` in a list of SSE payloads.
///
/// Falls back to the last event if no ID matches (handles servers that omit
/// or null the ID in error responses).
fn find_rpc_response(events: &[String], expected_id: u64) -> Result<String, Error> {
    for event in events {
        if let Ok(value) = serde_json::from_str::<Value>(event)
            && value.get("id").and_then(|v| v.as_u64()) == Some(expected_id)
        {
            return Ok(event.clone());
        }
    }
    // Fallback: no event matched the ID — return the last payload
    events
        .last()
        .cloned()
        .ok_or_else(|| Error::Mcp("No events in SSE response".into()))
}

fn mcp_result_to_tool_output(result: McpCallToolResult) -> ToolOutput {
    let text: String = result
        .content
        .iter()
        .filter_map(|c| {
            if c.content_type == "text" {
                c.text.as_deref()
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    if result.is_error {
        ToolOutput::error(text)
    } else {
        ToolOutput::success(text)
    }
}

fn mcp_tool_to_definition(tool: &McpToolDef) -> ToolDefinition {
    ToolDefinition {
        name: tool.name.clone(),
        description: tool.description.clone().unwrap_or_default(),
        input_schema: tool
            .input_schema
            .clone()
            .unwrap_or_else(|| serde_json::json!({"type": "object"})),
    }
}

// --- McpSession ---

struct McpSession {
    client: reqwest::Client,
    endpoint: String,
    session_id: RwLock<Option<String>>,
    next_id: AtomicU64,
    auth_header: Option<String>,
}

impl McpSession {
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Read the current session ID (cloned out of the lock).
    fn read_session_id(&self) -> Result<Option<String>, Error> {
        Ok(self
            .session_id
            .read()
            .map_err(|e| Error::Mcp(format!("Lock poisoned: {e}")))?
            .clone())
    }

    /// Update session ID from response header if the server provides one.
    fn update_session_id(&self, response: &reqwest::Response) -> Result<(), Error> {
        if let Some(new_sid) = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|v| v.to_str().ok())
        {
            *self
                .session_id
                .write()
                .map_err(|e| Error::Mcp(format!("Lock poisoned: {e}")))? =
                Some(new_sid.to_string());
        }
        Ok(())
    }

    async fn rpc(&self, method: &str, params: Option<Value>) -> Result<Value, Error> {
        let id = self.next_id();
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
            id,
        };

        let mut builder = self
            .client
            .post(&self.endpoint)
            .header("Accept", "application/json, text/event-stream")
            .json(&request);

        if let Some(sid) = self.read_session_id()? {
            builder = builder.header("Mcp-Session-Id", sid);
        }
        if let Some(auth) = &self.auth_header {
            builder = builder.header("Authorization", auth);
        }

        let response = builder.send().await?;
        self.update_session_id(&response)?;

        let status = response.status();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(Error::Mcp(format!("HTTP {}: {}", status.as_u16(), body)));
        }

        let json_str = if content_type.contains("text/event-stream") {
            let events = extract_sse_events(&body)?;
            find_rpc_response(&events, id)?
        } else {
            body
        };

        let rpc_response: JsonRpcResponse = serde_json::from_str(&json_str)?;

        if let Some(err) = rpc_response.error {
            return Err(Error::Mcp(format!(
                "JSON-RPC error {}: {}",
                err.code, err.message
            )));
        }

        rpc_response
            .result
            .ok_or_else(|| Error::Mcp("Response missing both result and error".into()))
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), Error> {
        let notification = JsonRpcNotification {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
        };

        let mut builder = self
            .client
            .post(&self.endpoint)
            .header("Accept", "application/json, text/event-stream")
            .json(&notification);

        if let Some(sid) = self.read_session_id()? {
            builder = builder.header("Mcp-Session-Id", sid);
        }
        if let Some(auth) = &self.auth_header {
            builder = builder.header("Authorization", auth);
        }

        let response = builder.send().await?;
        self.update_session_id(&response)?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(Error::Mcp(format!(
                "Notification HTTP {}: {}",
                status.as_u16(),
                body
            )));
        }

        Ok(())
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolOutput, Error> {
        // MCP servers expect arguments to be an object, never null.
        // LLMs sometimes send null/empty for tools with no required params.
        let arguments = if arguments.is_null() {
            serde_json::json!({})
        } else {
            arguments
        };
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });

        let result_value = self.rpc("tools/call", Some(params)).await?;
        let result: McpCallToolResult = serde_json::from_value(result_value)?;
        Ok(mcp_result_to_tool_output(result))
    }
}

// --- McpTool ---

struct McpTool {
    session: Arc<McpSession>,
    def: ToolDefinition,
}

impl Tool for McpTool {
    fn definition(&self) -> ToolDefinition {
        self.def.clone()
    }

    fn execute(
        &self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            match self.session.call_tool(&self.def.name, input).await {
                Ok(output) => Ok(output),
                Err(e) => {
                    tracing::warn!(
                        tool = %self.def.name,
                        error = %e,
                        "MCP tool call failed"
                    );
                    Ok(ToolOutput::error(e.to_string()))
                }
            }
        })
    }
}

// --- McpClient ---

/// Client for the Model Context Protocol (MCP).
///
/// Connects to an MCP server, performs the handshake, discovers tools,
/// and produces `Vec<Arc<dyn Tool>>` that plug into `AgentRunnerBuilder::tools()`.
pub struct McpClient {
    session: Arc<McpSession>,
    tools: Vec<McpToolDef>,
}

impl McpClient {
    /// Connect to an MCP server and discover available tools.
    ///
    /// Performs the full handshake: initialize → notifications/initialized → tools/list.
    pub async fn connect(endpoint: &str) -> Result<Self, Error> {
        Self::connect_internal(endpoint, None).await
    }

    /// Connect to an MCP server with an authorization header.
    ///
    /// Use this for agentgateway or other authenticated MCP proxies.
    /// The `auth_header` is sent as the `Authorization` header value
    /// (e.g., `"Bearer <token>"`).
    pub async fn connect_with_auth(
        endpoint: &str,
        auth_header: impl Into<String>,
    ) -> Result<Self, Error> {
        Self::connect_internal(endpoint, Some(auth_header.into())).await
    }

    async fn connect_internal(endpoint: &str, auth_header: Option<String>) -> Result<Self, Error> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()?;

        let session = Arc::new(McpSession {
            client,
            endpoint: endpoint.to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header,
        });

        // Initialize — rpc() captures Mcp-Session-Id from the response automatically
        session
            .rpc(
                "initialize",
                Some(serde_json::json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "heartbit",
                        "version": "0.1.0"
                    }
                })),
            )
            .await?;

        session.notify("notifications/initialized", None).await?;

        let tools_result = session.rpc("tools/list", None).await?;
        let tools_list: McpToolsListResult = serde_json::from_value(tools_result)?;

        Ok(Self {
            session,
            tools: tools_list.tools,
        })
    }

    /// Get tool definitions without consuming the client.
    ///
    /// Useful when you only need the schemas (e.g., for Restate task payloads)
    /// and don't need the executable tool instances.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(mcp_tool_to_definition).collect()
    }

    /// Convert discovered MCP tools into `Arc<dyn Tool>` instances.
    pub fn into_tools(self) -> Vec<Arc<dyn Tool>> {
        let session = self.session;
        self.tools
            .into_iter()
            .map(|t| {
                let tool: Arc<dyn Tool> = Arc::new(McpTool {
                    session: Arc::clone(&session),
                    def: mcp_tool_to_definition(&t),
                });
                tool
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- JSON-RPC tests ---

    #[test]
    fn jsonrpc_request_serialization() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            method: "tools/list".to_string(),
            params: Some(json!({"cursor": null})),
            id: 42,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["method"], "tools/list");
        assert_eq!(json["id"], 42);
        assert!(json.get("params").is_some());
    }

    #[test]
    fn jsonrpc_request_null_params_omitted() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            method: "tools/list".to_string(),
            params: None,
            id: 1,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("params").is_none());
    }

    #[test]
    fn jsonrpc_notification_has_no_id() {
        let notif = JsonRpcNotification {
            jsonrpc: "2.0",
            method: "notifications/initialized".to_string(),
            params: None,
        };
        let json = serde_json::to_value(&notif).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["method"], "notifications/initialized");
        assert!(json.get("id").is_none());
        assert!(json.get("params").is_none());
    }

    #[test]
    fn jsonrpc_response_parses_result() {
        let json_str = r#"{"jsonrpc":"2.0","result":{"tools":[]},"id":1}"#;
        let response: JsonRpcResponse = serde_json::from_str(json_str).unwrap();
        assert!(response.result.is_some());
        assert!(response.error.is_none());
        assert_eq!(response.result.unwrap(), json!({"tools": []}));
    }

    #[test]
    fn jsonrpc_response_parses_error() {
        let json_str =
            r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"Method not found"},"id":1}"#;
        let response: JsonRpcResponse = serde_json::from_str(json_str).unwrap();
        assert!(response.result.is_none());
        let err = response.error.unwrap();
        assert_eq!(err.code, -32601);
        assert_eq!(err.message, "Method not found");
    }

    // --- SSE tests ---

    #[test]
    fn sse_basic_extraction() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"result\":{},\"id\":1}\n\n";
        let events = extract_sse_events(body).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], r#"{"jsonrpc":"2.0","result":{},"id":1}"#);
    }

    #[test]
    fn sse_no_data_field_errors() {
        let body = "event: message\n\n";
        let err = extract_sse_events(body).unwrap_err();
        assert!(matches!(err, Error::Mcp(_)));
        assert!(err.to_string().contains("No data field"));
    }

    #[test]
    fn sse_no_space_after_colon() {
        let body = "data:{\"result\":\"ok\"}\n";
        let events = extract_sse_events(body).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], r#"{"result":"ok"}"#);
    }

    #[test]
    fn sse_multiple_events_extracted() {
        let body =
            "event: message\ndata: {\"first\": true}\n\nevent: message\ndata: {\"last\": true}\n\n";
        let events = extract_sse_events(body).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], r#"{"first": true}"#);
        assert_eq!(events[1], r#"{"last": true}"#);
    }

    #[test]
    fn sse_multi_line_data_concatenated() {
        let body = "data: first line\ndata: second line\n\n";
        let events = extract_sse_events(body).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], "first line\nsecond line");
    }

    // --- find_rpc_response tests ---

    #[test]
    fn find_response_matches_by_id() {
        let events = vec![
            r#"{"jsonrpc":"2.0","method":"notifications/progress","params":{}}"#.to_string(),
            r#"{"jsonrpc":"2.0","result":{"tools":[]},"id":5}"#.to_string(),
        ];
        let result = find_rpc_response(&events, 5).unwrap();
        assert!(result.contains(r#""id":5"#));
        assert!(result.contains(r#""result""#));
    }

    #[test]
    fn find_response_falls_back_to_last() {
        let events = vec![r#"{"jsonrpc":"2.0","result":{},"id":99}"#.to_string()];
        // Looking for id=1, but only id=99 exists — falls back to last
        let result = find_rpc_response(&events, 1).unwrap();
        assert!(result.contains("99"));
    }

    // --- MCP types tests ---

    #[test]
    fn mcp_tools_list_parsing() {
        let json = json!({
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file from disk",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                },
                {
                    "name": "list_dir",
                    "description": "List directory contents",
                    "inputSchema": {"type": "object"}
                }
            ]
        });

        let result: McpToolsListResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.tools.len(), 2);
        assert_eq!(result.tools[0].name, "read_file");
        assert_eq!(
            result.tools[0].description.as_deref(),
            Some("Read a file from disk")
        );
        assert!(result.tools[0].input_schema.is_some());
        assert_eq!(result.tools[1].name, "list_dir");
    }

    #[test]
    fn mcp_tool_to_definition_mapping() {
        let mcp_def = McpToolDef {
            name: "search".into(),
            description: Some("Search for files".into()),
            input_schema: Some(json!({
                "type": "object",
                "properties": {"query": {"type": "string"}}
            })),
        };

        let def = mcp_tool_to_definition(&mcp_def);
        assert_eq!(def.name, "search");
        assert_eq!(def.description, "Search for files");
        assert_eq!(
            def.input_schema,
            json!({"type": "object", "properties": {"query": {"type": "string"}}})
        );
    }

    #[test]
    fn mcp_tool_defaults_for_missing_fields() {
        let json = json!({"name": "minimal"});
        let mcp_def: McpToolDef = serde_json::from_value(json).unwrap();
        assert!(mcp_def.description.is_none());
        assert!(mcp_def.input_schema.is_none());

        let def = mcp_tool_to_definition(&mcp_def);
        assert_eq!(def.name, "minimal");
        assert_eq!(def.description, "");
        assert_eq!(def.input_schema, json!({"type": "object"}));
    }

    // --- Tool result tests ---

    #[test]
    fn tool_result_success() {
        let result = McpCallToolResult {
            content: vec![McpContent {
                content_type: "text".into(),
                text: Some("file contents here".into()),
            }],
            is_error: false,
        };

        let output = mcp_result_to_tool_output(result);
        assert_eq!(output.content, "file contents here");
        assert!(!output.is_error);
    }

    #[test]
    fn tool_result_error() {
        let result = McpCallToolResult {
            content: vec![McpContent {
                content_type: "text".into(),
                text: Some("permission denied".into()),
            }],
            is_error: true,
        };

        let output = mcp_result_to_tool_output(result);
        assert_eq!(output.content, "permission denied");
        assert!(output.is_error);
    }

    #[test]
    fn tool_result_multi_text_joined() {
        let result = McpCallToolResult {
            content: vec![
                McpContent {
                    content_type: "text".into(),
                    text: Some("line one".into()),
                },
                McpContent {
                    content_type: "text".into(),
                    text: Some("line two".into()),
                },
                McpContent {
                    content_type: "text".into(),
                    text: Some("line three".into()),
                },
            ],
            is_error: false,
        };

        let output = mcp_result_to_tool_output(result);
        assert_eq!(output.content, "line one\nline two\nline three");
    }

    #[test]
    fn tool_result_images_skipped() {
        let result = McpCallToolResult {
            content: vec![
                McpContent {
                    content_type: "text".into(),
                    text: Some("caption".into()),
                },
                McpContent {
                    content_type: "image".into(),
                    text: None,
                },
                McpContent {
                    content_type: "text".into(),
                    text: Some("more text".into()),
                },
            ],
            is_error: false,
        };

        let output = mcp_result_to_tool_output(result);
        assert_eq!(output.content, "caption\nmore text");
    }

    #[test]
    fn tool_result_parses_from_json() {
        let json = json!({
            "content": [
                {"type": "text", "text": "hello from mcp"}
            ],
            "isError": false
        });

        let result: McpCallToolResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.content[0].text.as_deref(), Some("hello from mcp"));
        assert!(!result.is_error);
    }

    #[test]
    fn tool_result_is_error_defaults_false() {
        let json = json!({
            "content": [
                {"type": "text", "text": "ok"}
            ]
        });

        let result: McpCallToolResult = serde_json::from_value(json).unwrap();
        assert!(!result.is_error);
    }

    // --- McpSession tests ---

    #[test]
    fn session_next_id_is_monotonic() {
        let session = McpSession {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        };

        assert_eq!(session.next_id(), 0);
        assert_eq!(session.next_id(), 1);
        assert_eq!(session.next_id(), 2);
    }

    // --- McpTool tests ---

    #[test]
    fn mcp_tool_returns_correct_definition() {
        let session = Arc::new(McpSession {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        });

        let expected_def = ToolDefinition {
            name: "read_file".into(),
            description: "Read a file".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}}
            }),
        };

        let tool = McpTool {
            session,
            def: expected_def.clone(),
        };

        let def = tool.definition();
        assert_eq!(def, expected_def);
    }

    #[tokio::test]
    async fn mcp_tool_execute_catches_network_errors() {
        let session = Arc::new(McpSession {
            client: reqwest::Client::new(),
            endpoint: "http://127.0.0.1:1".to_string(), // nothing listening
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        });

        let tool = McpTool {
            session,
            def: ToolDefinition {
                name: "test_tool".into(),
                description: "test".into(),
                input_schema: json!({"type": "object"}),
            },
        };

        // execute() should catch the connection error and return ToolOutput::error,
        // not propagate it as Err
        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.is_error);
        assert!(!result.content.is_empty());
    }
}
