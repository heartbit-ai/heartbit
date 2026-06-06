//! MCP server — exposes heartbit tools/resources to external MCP clients.
//!
//! Designed to be mounted on an existing Axum router via `handle_request()`. See
//! [`McpServer`] for security caveats; the caller is responsible for authentication.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    /// Server name reported in the `initialize` response.
    pub name: String,
    /// Server version reported in the `initialize` response.
    pub version: String,
    /// Expose registered tools via `tools/list` and `tools/call`.
    pub expose_tools: bool,
    /// Expose registered resources via `resources/list` and `resources/read`.
    pub expose_resources: bool,
    /// Expose prompts via `prompts/list` and `prompts/get` (currently a no-op).
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
    /// Resource URI (e.g. `file:///docs/readme.md`).
    pub uri: String,
    /// Display name for the resource.
    pub name: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional MIME type hint.
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

/// Idle timeout after which an inactive session is evicted.
///
/// SECURITY (F-MCP-3): sessions that have not sent a request within this
/// window are considered stale and removed from the session map, limiting
/// the blast radius of session-fixation or session-hijack attacks and
/// keeping memory bounded independently of the MAX_SESSIONS cap.
const SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60); // 30 minutes

/// MCP server that handles JSON-RPC requests.
///
/// Exposes heartbit tools, resources, and prompts to external MCP clients.
/// Designed to be mounted on an existing Axum router via `handle_request()`.
///
/// # Security — Fail-Closed Auth (F-MCP-3)
///
/// **This server fails closed: without authentication, every JSON-RPC request
/// returns an `Unauthorized` error by default.** To permit unauthenticated
/// access (e.g., for single-process local testing), call
/// [`McpServer::allow_unauthenticated`] explicitly on the builder.
///
/// For production deployments reachable over a network, the integrator MUST
/// either:
///
/// 1. Wire an [`AuthCallback`] via [`McpServer::with_auth_callback`] which is
///    consulted on every JSON-RPC call and passes the HTTP-level credentials
///    extracted by the enclosing handler, **or**
/// 2. Mount the server behind an HTTP middleware that rejects unauth'd
///    requests *before* they reach `handle_request` **and** call
///    [`McpServer::allow_unauthenticated`] to signal that outer-layer auth
///    is the only gate.
///
/// Without one of these, any network-reachable client is rejected with
/// UNAUTHORIZED (-32001).
pub struct McpServer {
    config: McpServerConfig,
    tools: Vec<Arc<dyn Tool>>,
    resources: Vec<ServerResource>,
    resource_reader: Option<ResourceReader>,
    /// Session map: session ID → last-active `Instant`.
    ///
    /// SECURITY (F-MCP-3): bounded by `MAX_SESSIONS`; entries older than
    /// `SESSION_IDLE_TIMEOUT` are evicted to prevent stale-session accumulation.
    sessions: parking_lot::RwLock<HashMap<String, Instant>>,
    auth_callback: Option<AuthCallback>,
    /// SECURITY (F-MCP-3): when `false` (the default) the server rejects any
    /// request that arrives with no `auth_callback` installed. Operators who
    /// deliberately want open access (local testing, inner-loop dev) must set
    /// this flag explicitly via [`McpServer::allow_unauthenticated`].
    allow_unauthenticated: bool,
}

impl McpServer {
    /// Create a fresh server with the given config. No tools/resources are registered until
    /// [`McpServer::with_tools`] / [`McpServer::with_resources`] are called.
    ///
    /// # Security
    ///
    /// By default the server fails **closed**: every request returns
    /// `Unauthorized` until either an [`AuthCallback`] is installed (via
    /// [`McpServer::with_auth_callback`]) or unauthenticated access is
    /// explicitly opted into (via [`McpServer::allow_unauthenticated`]).
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
            allow_unauthenticated: false,
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

    /// Opt into permitting requests without authentication.
    ///
    /// SECURITY (F-MCP-3): the default is **fail-closed** — any request that
    /// arrives when no [`AuthCallback`] is installed is rejected with
    /// `Unauthorized`. Call this method *only* when:
    ///
    /// - The process is single-tenant and runs locally (e.g. integration tests,
    ///   local CLI dev loop), **or**
    /// - An outer HTTP-middleware layer already enforces authentication and the
    ///   inner MCP layer should treat all forwarded requests as pre-authorized.
    ///
    /// Do **not** call this on a server that is directly reachable over an
    /// untrusted network without an [`AuthCallback`].
    pub fn allow_unauthenticated(mut self) -> Self {
        self.allow_unauthenticated = true;
        self
    }

    /// Create or validate a session ID, refreshing the last-active timestamp.
    ///
    /// SECURITY (F-MCP-3): evicts sessions idle longer than
    /// [`SESSION_IDLE_TIMEOUT`] before inserting a new one, so both the
    /// count cap and the idle-timeout cap apply independently.
    fn ensure_session(&self, session_id: Option<&str>) -> String {
        let now = Instant::now();

        // Fast path: re-use an existing, non-expired session.
        if let Some(sid) = session_id {
            let mut sessions = self.sessions.write();
            if let Some(last_active) = sessions.get_mut(sid) {
                if now.duration_since(*last_active) < SESSION_IDLE_TIMEOUT {
                    *last_active = now;
                    return sid.to_string();
                }
                // Session expired — fall through to create a fresh one.
                sessions.remove(sid);
            }
        }

        let new_sid = Uuid::new_v4().to_string();
        let mut sessions = self.sessions.write();

        // SECURITY (F-MCP-3): evict all TTL-expired sessions first.
        sessions.retain(|_, last_active| now.duration_since(*last_active) < SESSION_IDLE_TIMEOUT);

        // Then apply the count cap as a last-resort backstop.
        if sessions.len() >= MAX_SESSIONS
            && let Some(victim) = sessions.keys().next().cloned()
        {
            sessions.remove(&victim);
        }

        sessions.insert(new_sid.clone(), now);
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
    ///
    /// SECURITY (F-MCP-3): this method fails **closed**. If no
    /// [`AuthCallback`] is installed and [`McpServer::allow_unauthenticated`]
    /// was not called, every request is rejected with `Unauthorized` regardless
    /// of method. Install an auth callback or explicitly opt into open access
    /// via the builder before wiring this server into a network-accessible
    /// endpoint.
    pub async fn handle_request_with_auth(
        &self,
        body: &str,
        session_id: Option<&str>,
        auth_header: Option<&str>,
    ) -> (String, String) {
        let sid = self.ensure_session(session_id);

        let response = match serde_json::from_str::<JsonRpcRequest>(body) {
            Ok(req) => {
                // SECURITY (F-MCP-3): authentication gate — fail closed.
                // Decision matrix:
                //   auth_callback=Some(cb) → call cb; reject if it returns false
                //   auth_callback=None + allow_unauthenticated=true  → allow
                //   auth_callback=None + allow_unauthenticated=false → reject
                let authorized = match &self.auth_callback {
                    Some(cb) => cb(&req.method, session_id, auth_header),
                    None => self.allow_unauthenticated,
                };
                if !authorized {
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
            // Tests run in-process — explicitly opt into unauthenticated access
            // so the default fail-closed posture does not break the test suite.
            .allow_unauthenticated()
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
        let server = McpServer::new(McpServerConfig::default()).allow_unauthenticated();
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
        .allow_unauthenticated()
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
        .allow_unauthenticated()
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
            let now = Instant::now();
            for i in 0..MAX_SESSIONS {
                sessions.insert(format!("sid-{i}"), now);
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

    /// SECURITY (F-MCP-3): a server with no auth_callback and no
    /// allow_unauthenticated must return UNAUTHORIZED for any request.
    #[tokio::test]
    async fn no_auth_callback_and_no_allow_unauth_returns_unauthorized() {
        let echo: Arc<dyn Tool> = Arc::new(EchoTool);
        // Default server: auth_callback=None, allow_unauthenticated=false → fail closed.
        let server = McpServer::new(McpServerConfig::default()).with_tools(vec![echo]);

        let req = json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {"name": "echo", "arguments": {"text": "should not run"}}
        });
        let (resp, _) = server.handle_request(&req.to_string(), None).await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert!(
            parsed["error"].is_object(),
            "expected error when no auth installed; got: {parsed}"
        );
        assert_eq!(
            parsed["error"]["code"].as_i64().unwrap_or_default(),
            UNAUTHORIZED,
            "expected UNAUTHORIZED code; got: {parsed}"
        );
    }

    /// SECURITY (F-MCP-3): allow_unauthenticated() permits requests without
    /// an auth_callback. This is the opt-in escape hatch for local/test usage.
    #[tokio::test]
    async fn allow_unauthenticated_permits_requests() {
        let echo: Arc<dyn Tool> = Arc::new(EchoTool);
        let server = McpServer::new(McpServerConfig::default())
            .allow_unauthenticated()
            .with_tools(vec![echo]);

        let req = json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 2,
            "params": {"name": "echo", "arguments": {"text": "allowed"}}
        });
        let (resp, _) = server.handle_request(&req.to_string(), None).await;
        let parsed: Value = serde_json::from_str(&resp).unwrap();
        assert!(parsed["error"].is_null(), "expected success; got: {parsed}");
        assert_eq!(
            parsed["result"]["content"][0]["text"]
                .as_str()
                .unwrap_or_default(),
            "allowed"
        );
    }

    /// SECURITY (F-MCP-3): sessions past their idle TTL must be evicted so
    /// they cannot be reused after expiry (session fixation / long-lived orphan).
    #[test]
    fn expired_session_is_evicted() {
        let server = McpServer::new(McpServerConfig::default());
        let expired_sid = "old-session-id";
        // Inject a session whose last-active timestamp is far in the past.
        {
            let mut sessions = server.sessions.write();
            // Simulate a session created SESSION_IDLE_TIMEOUT + 1s ago.
            let expired_at = Instant::now()
                .checked_sub(SESSION_IDLE_TIMEOUT + Duration::from_secs(1))
                .expect("subtraction should not underflow on any sane platform");
            sessions.insert(expired_sid.to_string(), expired_at);
        }
        // ensure_session with the expired id must produce a *new* id.
        let new_sid = server.ensure_session(Some(expired_sid));
        assert_ne!(
            new_sid, expired_sid,
            "expired session must not be reused; got same sid back"
        );
        // The expired entry must no longer be in the map.
        assert!(
            !server.sessions.read().contains_key(expired_sid),
            "expired session must be evicted from the map"
        );
    }
}
