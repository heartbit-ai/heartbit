#![allow(missing_docs)]
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::{Tool, ToolOutput};
use crate::error::Error;

const PROTOCOL_VERSION: &str = "2025-11-25";

// --- JSON-RPC types ---

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
    id: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

impl JsonRpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        }
    }

    fn error(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
            }),
            id,
        }
    }
}

// JSON-RPC error codes
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

// --- MCP Server ---

/// Configuration for the MCP server.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub name: String,
    pub version: String,
    pub expose_tools: bool,
    pub expose_resources: bool,
    pub expose_prompts: bool,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            name: "heartbit".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            expose_tools: true,
            expose_resources: true,
            expose_prompts: false,
        }
    }
}

/// A resource exposed by the MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerResource {
    pub uri: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Callback to read resource content. Returns `(mime_type, text)`.
pub type ResourceReader =
    Arc<dyn Fn(&str) -> Result<Vec<(Option<String>, String)>, Error> + Send + Sync>;

/// Callback to authorize a JSON-RPC call. Receives the parsed `method`, the
/// session id (if the client provided one), and any HTTP-level credentials
/// the integrator wants to forward (typically a Bearer token extracted from
/// the `Authorization` header upstream). Return `true` to allow, `false` to
/// reject with HTTP-401-equivalent JSON-RPC error.
pub type AuthCallback = Arc<dyn Fn(&str, Option<&str>, Option<&str>) -> bool + Send + Sync>;

/// JSON-RPC error code used when [`McpServer`] rejects a call via its
/// `auth_callback`. Mirrors HTTP 401 semantics in the MCP transport layer.
const UNAUTHORIZED: i64 = -32001;

/// Maximum number of MCP sessions retained simultaneously.
///
/// SECURITY (F-MCP-3): without a cap, a hostile/unauth client can issue
/// distinct `Mcp-Session-Id` values indefinitely and balloon the in-memory
/// `sessions` map. The cap drops older entries (insertion order best-effort)
/// once the limit is reached, keeping memory bounded.
const MAX_SESSIONS: usize = 256;

/// MCP server that handles JSON-RPC requests.
///
/// Exposes heartbit tools, resources, and prompts to external MCP clients.
/// Designed to be mounted on an existing Axum router via `handle_request()`.
///
/// # Security
///
/// **The caller is responsible for authentication.** This server does not
/// authenticate JSON-RPC requests by default. When the server is reachable
/// over an untrusted network, the integrator MUST either:
///
/// 1. Mount the server behind an HTTP middleware that rejects unauth'd
///    requests before they reach `handle_request`, **or**
/// 2. Wire an [`AuthCallback`] via [`McpServer::with_auth_callback`] which is
///    consulted on every JSON-RPC call.
///
/// Without one of these, any network-reachable client can call `tools/call`
/// with arbitrary tool names — including potentially destructive builtins
/// like `bash`, `write`, or `patch` if those have been registered.
pub struct McpServer {
    config: McpServerConfig,
    tools: Vec<Arc<dyn Tool>>,
    resources: Vec<ServerResource>,
    resource_reader: Option<ResourceReader>,
    sessions: parking_lot::RwLock<HashMap<String, ()>>,
    auth_callback: Option<AuthCallback>,
}

impl McpServer {
    pub fn new(config: McpServerConfig) -> Self {
        Self {
            config,
            tools: Vec::new(),
            resources: Vec::new(),
            resource_reader: None,
            // parking_lot adopted on this hot path (every MCP request);
            // see T2 in `tasks/performance-audit-heartbit-core-2026-05-06.md`.
            sessions: parking_lot::RwLock::new(HashMap::new()),
            auth_callback: None,
        }
    }

    /// Register tools to expose via MCP.
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Register resources to expose via MCP.
    pub fn with_resources(
        mut self,
        resources: Vec<ServerResource>,
        reader: ResourceReader,
    ) -> Self {
        self.resources = resources;
        self.resource_reader = Some(reader);
        self
    }

    /// Install an authentication callback (`fn(method, session_id, auth_header) -> bool`).
    ///
    /// SECURITY (F-MCP-3): when set, every `handle_request` call is gated by
    /// this callback. The integrator should extract the relevant credentials
    /// from the HTTP layer (e.g. Authorization header) and pass them through
    /// [`McpServer::handle_request_with_auth`].
    pub fn with_auth_callback(mut self, callback: AuthCallback) -> Self {
        self.auth_callback = Some(callback);
        self
    }

    /// Create or validate a session ID.
    fn ensure_session(&self, session_id: Option<&str>) -> String {
        if let Some(sid) = session_id
            && self.sessions.read().contains_key(sid)
        {
            return sid.to_string();
        }
        let new_sid = Uuid::new_v4().to_string();
        let mut sessions = self.sessions.write();
        // SECURITY (F-MCP-3): bound the session map. When at capacity,
        // drop one existing entry before inserting the new one. Best-effort
        // FIFO via HashMap iteration order; for higher-traffic deployments
        // an LRU cache would be more appropriate.
        if sessions.len() >= MAX_SESSIONS
            && let Some(victim) = sessions.keys().next().cloned()
        {
            sessions.remove(&victim);
        }
        sessions.insert(new_sid.clone(), ());
        new_sid
    }

    /// Handle a JSON-RPC request and return a response with session ID.
    ///
    /// If an [`AuthCallback`] is installed, this method calls it without an
    /// auth header (`None`). Use [`McpServer::handle_request_with_auth`] when
    /// the integrator wants to forward HTTP-level credentials.
    pub async fn handle_request(&self, body: &str, session_id: Option<&str>) -> (String, String) {
        self.handle_request_with_auth(body, session_id, None).await
    }

    /// Handle a JSON-RPC request with an explicit auth header (e.g. extracted
    /// from the upstream HTTP Authorization header). When an
    /// [`AuthCallback`] is installed, it receives this value.
    pub async fn handle_request_with_auth(
        &self,
        body: &str,
        session_id: Option<&str>,
        auth_header: Option<&str>,
    ) -> (String, String) {
        let sid = self.ensure_session(session_id);

        let response = match serde_json::from_str::<JsonRpcRequest>(body) {
            Ok(req) => {
                // SECURITY (F-MCP-3): authentication gate. Run BEFORE routing
                // so unauthorized callers cannot probe or exercise tools.
                if let Some(ref cb) = self.auth_callback
                    && !cb(&req.method, session_id, auth_header)
                {
                    let id = req.id.clone().unwrap_or(Value::Null);
                    let err = JsonRpcResponse::error(id, UNAUTHORIZED, "Unauthorized");
                    serde_json::to_string(&err).unwrap_or_default()
                } else {
                    self.route(req).await
                }
            }
            Err(e) => {
                let err = JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}"));
                serde_json::to_string(&err).unwrap_or_default()
            }
        };

        (response, sid)
    }

    async fn route(&self, req: JsonRpcRequest) -> String {
        let id = req.id.clone().unwrap_or(Value::Null);
        let result = match req.method.as_str() {
            "initialize" => self.handle_initialize(&id),
            "ping" => Ok(JsonRpcResponse::success(id.clone(), serde_json::json!({}))),
            "tools/list" => self.handle_tools_list(&id, req.params.as_ref()),
            "tools/call" => self.handle_tools_call(&id, req.params.as_ref()).await,
            "resources/list" => self.handle_resources_list(&id, req.params.as_ref()),
            "resources/read" => self.handle_resources_read(&id, req.params.as_ref()),
            _ if req.method.starts_with("notifications/") => {
                // Notifications don't require a response, but we return empty for HTTP.
                return String::new();
            }
            _ => Ok(JsonRpcResponse::error(
                id.clone(),
                METHOD_NOT_FOUND,
                format!("Method not found: {}", req.method),
            )),
        };

        match result {
            Ok(resp) => serde_json::to_string(&resp).unwrap_or_default(),
            Err(e) => {
                let resp = JsonRpcResponse::error(id, INTERNAL_ERROR, e.to_string());
                serde_json::to_string(&resp).unwrap_or_default()
            }
        }
    }

    fn handle_initialize(&self, id: &Value) -> Result<JsonRpcResponse, Error> {
        let mut capabilities = serde_json::json!({});

        if self.config.expose_tools && !self.tools.is_empty() {
            capabilities["tools"] = serde_json::json!({ "listChanged": false });
        }
        if self.config.expose_resources && !self.resources.is_empty() {
            capabilities["resources"] =
                serde_json::json!({ "subscribe": false, "listChanged": false });
        }

        Ok(JsonRpcResponse::success(
            id.clone(),
            serde_json::json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": capabilities,
                "serverInfo": {
                    "name": self.config.name,
                    "version": self.config.version
                }
            }),
        ))
    }

    fn handle_tools_list(
        &self,
        id: &Value,
        _params: Option<&Value>,
    ) -> Result<JsonRpcResponse, Error> {
        if !self.config.expose_tools {
            return Ok(JsonRpcResponse::success(
                id.clone(),
                serde_json::json!({ "tools": [] }),
            ));
        }

        let tools: Vec<Value> = self
            .tools
            .iter()
            .map(|t| {
                let def = t.definition();
                serde_json::json!({
                    "name": def.name,
                    "description": def.description,
                    "inputSchema": def.input_schema,
                })
            })
            .collect();

        Ok(JsonRpcResponse::success(
            id.clone(),
            serde_json::json!({ "tools": tools }),
        ))
    }

    async fn handle_tools_call(
        &self,
        id: &Value,
        params: Option<&Value>,
    ) -> Result<JsonRpcResponse, Error> {
        let params = params.ok_or_else(|| Error::Mcp("Missing params for tools/call".into()))?;
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Mcp("Missing 'name' in tools/call params".into()))?;
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        let tool = self
            .tools
            .iter()
            .find(|t| t.definition().name == name)
            .ok_or_else(|| Error::Mcp(format!("Tool not found: {name}")))?;

        // TODO(phase-1): derive ExecutionContext from MCP session / clientInfo when
        // multi-tenant MCP integration lands (likely heartbit-ghost Phase 1). Default
        // placeholder is safe: pre-trait-change there was no context at all, so blast
        // radius is unchanged.
        match tool
            .execute(&crate::ExecutionContext::default(), arguments)
            .await
        {
            Ok(output) => Ok(JsonRpcResponse::success(
                id.clone(),
                tool_output_to_mcp(output),
            )),
            Err(e) => Ok(JsonRpcResponse::success(
                id.clone(),
                serde_json::json!({
                    "content": [{"type": "text", "text": e.to_string()}],
                    "isError": true
                }),
            )),
        }
    }

    fn handle_resources_list(
        &self,
        id: &Value,
        _params: Option<&Value>,
    ) -> Result<JsonRpcResponse, Error> {
        if !self.config.expose_resources {
            return Ok(JsonRpcResponse::success(
                id.clone(),
                serde_json::json!({ "resources": [] }),
            ));
        }

        let resources: Vec<Value> = self
            .resources
            .iter()
            .map(|r| serde_json::to_value(r).unwrap_or_default())
            .collect();

        Ok(JsonRpcResponse::success(
            id.clone(),
            serde_json::json!({ "resources": resources }),
        ))
    }

    fn handle_resources_read(
        &self,
        id: &Value,
        params: Option<&Value>,
    ) -> Result<JsonRpcResponse, Error> {
        let params =
            params.ok_or_else(|| Error::Mcp("Missing params for resources/read".into()))?;
        let uri = params
            .get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Mcp("Missing 'uri' in resources/read params".into()))?;

        // Validate the URI exists
        if !self.resources.iter().any(|r| r.uri == uri) {
            return Ok(JsonRpcResponse::error(
                id.clone(),
                INVALID_PARAMS,
                format!("Resource not found: {uri}"),
            ));
        }

        let reader = self
            .resource_reader
            .as_ref()
            .ok_or_else(|| Error::Mcp("No resource reader configured".into()))?;

        match reader(uri) {
            Ok(contents) => {
                let content_values: Vec<Value> = contents
                    .into_iter()
                    .map(|(mime, text)| {
                        let mut obj = serde_json::json!({
                            "uri": uri,
                            "text": text,
                        });
                        if let Some(m) = mime {
                            obj["mimeType"] = Value::String(m);
                        }
                        obj
                    })
                    .collect();
                Ok(JsonRpcResponse::success(
                    id.clone(),
                    serde_json::json!({ "contents": content_values }),
                ))
            }
            Err(e) => Ok(JsonRpcResponse::error(
                id.clone(),
                INTERNAL_ERROR,
                e.to_string(),
            )),
        }
    }
}

fn tool_output_to_mcp(output: ToolOutput) -> Value {
    serde_json::json!({
        "content": [{"type": "text", "text": output.content}],
        "isError": output.is_error
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;

    use crate::llm::types::ToolDefinition;
    use serde_json::json;

    struct EchoTool;

    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".into(),
                description: "Echo input".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }),
            }
        }

        fn execute(
            &self,
            _ctx: &crate::ExecutionContext,
            input: Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
            Box::pin(async move {
                let text = input
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("no text");
                Ok(ToolOutput::success(text))
            })
        }
    }

    struct FailTool;

    impl Tool for FailTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "fail".into(),
                description: "Always fails".into(),
                input_schema: json!({"type": "object"}),
            }
        }

        fn execute(
            &self,
            _ctx: &crate::ExecutionContext,
            _input: Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
            Box::pin(async move { Err(Error::Mcp("intentional failure".into())) })
        }
    }

    fn make_server() -> McpServer {
        let echo: Arc<dyn Tool> = Arc::new(EchoTool);
        let fail: Arc<dyn Tool> = Arc::new(FailTool);

        McpServer::new(McpServerConfig::default())
            .with_tools(vec![echo, fail])
            .with_resources(
                vec![
                    ServerResource {
                        uri: "heartbit://tasks/123".into(),
                        name: "task_123".into(),
                        description: Some("Task result".into()),
                        mime_type: Some("text/plain".into()),
                    },
                    ServerResource {
                        uri: "heartbit://config".into(),
                        name: "config".into(),
                        description: None,
                        mime_type: None,
                    },
                ],
                Arc::new(|uri: &str| match uri {
                    "heartbit://tasks/123" => {
                        Ok(vec![(Some("text/plain".into()), "Task completed!".into())])
                    }
                    "heartbit://config" => Ok(vec![(None, "key=value".into())]),
                    _ => Err(Error::Mcp(format!("Unknown resource: {uri}"))),
                }),
            )
    }

    // --- Initialize ---

    #[tokio::test]
    async fn initialize_returns_capabilities() {
        let server = make_server();
        let req = json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            },
            "id": 1
        });

        let (resp, sid) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();

        assert_eq!(parsed["result"]["protocolVersion"], "2025-11-25");
        assert!(parsed["result"]["capabilities"]["tools"].is_object());
        assert!(parsed["result"]["capabilities"]["resources"].is_object());
        assert_eq!(parsed["result"]["serverInfo"]["name"], "heartbit");
        assert!(!sid.is_empty());
    }

    #[tokio::test]
    async fn initialize_no_tools_capability_when_empty() {
        let server = McpServer::new(McpServerConfig::default());
        let req = json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 1
        });

        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();

        assert!(parsed["result"]["capabilities"]["tools"].is_null());
        assert!(parsed["result"]["capabilities"]["resources"].is_null());
    }

    // --- Ping ---

    #[tokio::test]
    async fn ping_returns_empty_result() {
        let server = make_server();
        let req = json!({"jsonrpc": "2.0", "method": "ping", "id": 42});
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["result"], json!({}));
        assert_eq!(parsed["id"], 42);
    }

    // --- Tools/list ---

    #[tokio::test]
    async fn tools_list_returns_all_tools() {
        let server = make_server();
        let req = json!({"jsonrpc": "2.0", "method": "tools/list", "id": 1});
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();

        let tools = parsed["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"], "echo");
        assert_eq!(tools[1]["name"], "fail");
        assert!(tools[0]["inputSchema"]["properties"]["text"].is_object());
    }

    #[tokio::test]
    async fn tools_list_empty_when_disabled() {
        let server = McpServer::new(McpServerConfig {
            expose_tools: false,
            ..Default::default()
        })
        .with_tools(vec![Arc::new(EchoTool)]);

        let req = json!({"jsonrpc": "2.0", "method": "tools/list", "id": 1});
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["result"]["tools"].as_array().unwrap().len(), 0);
    }

    // --- Tools/call ---

    #[tokio::test]
    async fn tools_call_echo() {
        let server = make_server();
        let req = json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hello world"}},
            "id": 1
        });
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();

        let content = &parsed["result"]["content"][0];
        assert_eq!(content["type"], "text");
        assert_eq!(content["text"], "hello world");
        assert_eq!(parsed["result"]["isError"], false);
    }

    #[tokio::test]
    async fn tools_call_fail_returns_error_content() {
        let server = make_server();
        let req = json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "fail", "arguments": {}},
            "id": 1
        });
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();

        assert_eq!(parsed["result"]["isError"], true);
        assert!(
            parsed["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("intentional failure")
        );
    }

    #[tokio::test]
    async fn tools_call_not_found() {
        let server = make_server();
        let req = json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
            "id": 1
        });
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert!(
            parsed["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not found")
        );
    }

    #[tokio::test]
    async fn tools_call_missing_params() {
        let server = make_server();
        let req = json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1
        });
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert!(parsed["error"].is_object());
    }

    // --- Resources/list ---

    #[tokio::test]
    async fn resources_list_returns_all() {
        let server = make_server();
        let req = json!({"jsonrpc": "2.0", "method": "resources/list", "id": 1});
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();

        let resources = parsed["result"]["resources"].as_array().unwrap();
        assert_eq!(resources.len(), 2);
        assert_eq!(resources[0]["uri"], "heartbit://tasks/123");
        assert_eq!(resources[0]["name"], "task_123");
        assert_eq!(resources[0]["mimeType"], "text/plain");
    }

    #[tokio::test]
    async fn resources_list_empty_when_disabled() {
        let server = McpServer::new(McpServerConfig {
            expose_resources: false,
            ..Default::default()
        })
        .with_resources(
            vec![ServerResource {
                uri: "test://x".into(),
                name: "x".into(),
                description: None,
                mime_type: None,
            }],
            Arc::new(|_| Ok(vec![])),
        );

        let req = json!({"jsonrpc": "2.0", "method": "resources/list", "id": 1});
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["result"]["resources"].as_array().unwrap().len(), 0);
    }

    // --- Resources/read ---

    #[tokio::test]
    async fn resources_read_success() {
        let server = make_server();
        let req = json!({
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": "heartbit://tasks/123"},
            "id": 1
        });
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();

        let contents = parsed["result"]["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["uri"], "heartbit://tasks/123");
        assert_eq!(contents[0]["text"], "Task completed!");
        assert_eq!(contents[0]["mimeType"], "text/plain");
    }

    #[tokio::test]
    async fn resources_read_not_found() {
        let server = make_server();
        let req = json!({
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": "heartbit://nonexistent"},
            "id": 1
        });
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert!(
            parsed["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not found")
        );
    }

    #[tokio::test]
    async fn resources_read_missing_uri() {
        let server = make_server();
        let req = json!({
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {},
            "id": 1
        });
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert!(parsed["error"].is_object());
    }

    // --- Method not found ---

    #[tokio::test]
    async fn unknown_method_returns_error() {
        let server = make_server();
        let req = json!({"jsonrpc": "2.0", "method": "foobar", "id": 1});
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["error"]["code"], METHOD_NOT_FOUND);
    }

    // --- Notifications ---

    #[tokio::test]
    async fn notification_returns_empty_string() {
        let server = make_server();
        let req = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        let (resp, _) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        assert!(resp.is_empty());
    }

    // --- Parse error ---

    #[tokio::test]
    async fn invalid_json_returns_parse_error() {
        let server = make_server();
        let (resp, _) = server.handle_request("not json", None).await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["error"]["code"], -32700);
    }

    // --- Session management ---

    #[tokio::test]
    async fn session_id_created_on_first_request() {
        let server = make_server();
        let req = json!({"jsonrpc": "2.0", "method": "ping", "id": 1});
        let (_, sid1) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), None)
            .await;
        assert!(!sid1.is_empty());
        // Reuse the session
        let (_, sid2) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), Some(&sid1))
            .await;
        assert_eq!(sid1, sid2);
    }

    #[tokio::test]
    async fn unknown_session_creates_new() {
        let server = make_server();
        let req = json!({"jsonrpc": "2.0", "method": "ping", "id": 1});
        let (_, sid) = server
            .handle_request(&serde_json::to_string(&req).unwrap(), Some("bad-session"))
            .await;
        assert_ne!(sid, "bad-session");
    }

    // --- tool_output_to_mcp ---

    #[test]
    fn tool_output_success_to_mcp() {
        let output = ToolOutput::success("hello");
        let mcp = tool_output_to_mcp(output);
        assert_eq!(mcp["content"][0]["type"], "text");
        assert_eq!(mcp["content"][0]["text"], "hello");
        assert_eq!(mcp["isError"], false);
    }

    #[test]
    fn tool_output_error_to_mcp() {
        let output = ToolOutput::error("bad");
        let mcp = tool_output_to_mcp(output);
        assert_eq!(mcp["content"][0]["text"], "bad");
        assert_eq!(mcp["isError"], true);
    }

    // --- Config defaults ---

    #[test]
    fn config_defaults() {
        let config = McpServerConfig::default();
        assert_eq!(config.name, "heartbit");
        assert!(config.expose_tools);
        assert!(config.expose_resources);
        assert!(!config.expose_prompts);
    }

    // --- ServerResource serde ---

    #[test]
    fn server_resource_serde_roundtrip() {
        let r = ServerResource {
            uri: "heartbit://tasks/1".into(),
            name: "task_1".into(),
            description: Some("A task".into()),
            mime_type: Some("application/json".into()),
        };
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["uri"], "heartbit://tasks/1");
        assert_eq!(json["mimeType"], "application/json");
        let parsed: ServerResource = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.name, "task_1");
    }

    #[test]
    fn server_resource_minimal() {
        let json = json!({"uri": "test://x", "name": "x"});
        let r: ServerResource = serde_json::from_value(json).unwrap();
        assert!(r.description.is_none());
        assert!(r.mime_type.is_none());
    }

    /// SECURITY (F-MCP-3): an installed `auth_callback` returning `false` must
    /// produce a JSON-RPC unauthorized response and **not** route to the tool.
    #[tokio::test]
    async fn auth_callback_rejects_when_returning_false() {
        let echo: Arc<dyn Tool> = Arc::new(EchoTool);
        let server = McpServer::new(McpServerConfig::default())
            .with_tools(vec![echo])
            .with_auth_callback(Arc::new(|_method, _sid, _auth| false));

        let req = json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 7,
            "params": {"name": "echo", "arguments": {"text": "should not run"}}
        });
        let (resp, _sid) = server.handle_request(&req.to_string(), None).await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert!(parsed["error"].is_object(), "expected error response");
        let code = parsed["error"]["code"].as_i64().unwrap_or_default();
        assert_eq!(code, UNAUTHORIZED, "expected 'Unauthorized' code");
        assert!(
            parsed["result"].is_null(),
            "result must be absent on auth failure"
        );
    }

    /// SECURITY (F-MCP-3): an `auth_callback` returning `true` allows the call
    /// to route normally. Confirms we did not introduce a regression.
    #[tokio::test]
    async fn auth_callback_allows_when_returning_true() {
        let echo: Arc<dyn Tool> = Arc::new(EchoTool);
        let server = McpServer::new(McpServerConfig::default())
            .with_tools(vec![echo])
            .with_auth_callback(Arc::new(|_method, _sid, _auth| true));

        let req = json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 8,
            "params": {"name": "echo", "arguments": {"text": "ok"}}
        });
        let (resp, _sid) = server.handle_request(&req.to_string(), None).await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert!(parsed["error"].is_null(), "expected success: {parsed}");
        assert!(
            parsed["result"]["content"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .contains("ok")
        );
    }

    /// SECURITY (F-MCP-3): the session map MUST be bounded so unauth'd clients
    /// cannot exhaust memory by minting fresh `Mcp-Session-Id` values.
    #[tokio::test]
    async fn session_map_is_bounded() {
        let server = McpServer::new(McpServerConfig::default());
        // Force the cap to fill — we do this by manipulating the lock directly.
        {
            let mut sessions = server.sessions.write();
            for i in 0..MAX_SESSIONS {
                sessions.insert(format!("sid-{i}"), ());
            }
            assert_eq!(sessions.len(), MAX_SESSIONS);
        }
        // Issue another `ensure_session` with a new id — should evict and stay bounded.
        let _ = server.ensure_session(None);
        let sessions = server.sessions.read();
        assert!(
            sessions.len() <= MAX_SESSIONS,
            "session map exceeded MAX_SESSIONS = {MAX_SESSIONS}: {}",
            sessions.len()
        );
    }
}
