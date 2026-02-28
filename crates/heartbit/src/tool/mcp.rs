use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

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
    #[serde(default, rename = "nextCursor")]
    next_cursor: Option<String>,
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
    let non_text_count = result
        .content
        .iter()
        .filter(|c| c.content_type != "text")
        .count();
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

    let output = if text.is_empty() && non_text_count > 0 {
        format!(
            "[MCP server returned {non_text_count} non-text content block(s) that cannot be displayed]"
        )
    } else {
        text
    };

    if result.is_error {
        ToolOutput::error(output)
    } else {
        ToolOutput::success(output)
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

/// Process a raw JSON-RPC response string into the result value.
///
/// Shared between HTTP and stdio transports.
fn process_rpc_response(json_str: &str) -> Result<Value, Error> {
    let rpc_response: JsonRpcResponse = serde_json::from_str(json_str)?;

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

/// Read a JSON-RPC response from a stdio stream, skipping notifications.
///
/// MCP stdio protocol sends newline-delimited JSON. Notifications (no `id` field
/// or null id) are skipped. Returns the raw JSON string of the first response
/// matching `expected_id`.
async fn read_stdio_response<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
    expected_id: u64,
) -> Result<String, Error> {
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader
            .read_line(&mut buf)
            .await
            .map_err(|e| Error::Mcp(format!("stdio read error: {e}")))?;
        if n == 0 {
            return Err(Error::Mcp("MCP stdio server closed unexpectedly".into()));
        }
        let trimmed = buf.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Try to parse as JSON; skip non-JSON lines (e.g., debug output on stdout).
        let value: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Notifications have no "id" or null id — skip them.
        match value.get("id") {
            None | Some(&Value::Null) => continue,
            _ => {}
        }

        if value.get("id").and_then(|v| v.as_u64()) == Some(expected_id) {
            return Ok(trimmed.to_string());
        }
        // Different ID — skip (shouldn't happen with serialized access, but safe).
    }
}

// --- Auth providers ---

/// Provides authorization headers for MCP requests on a per-user basis.
///
/// Implementations can fetch tokens dynamically (e.g., via RFC 8693 token exchange)
/// instead of using a single static auth header for all requests.
pub trait AuthProvider: Send + Sync {
    /// Return the Authorization header value for the given user/tenant context.
    /// Returns `None` if no auth is needed.
    fn auth_header_for<'a>(
        &'a self,
        user_id: &'a str,
        tenant_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, Error>> + Send + 'a>>;
}

/// Auth provider that always returns the same static auth header.
pub struct StaticAuthProvider {
    header: Option<String>,
}

impl StaticAuthProvider {
    pub fn new(header: Option<String>) -> Self {
        Self { header }
    }
}

impl AuthProvider for StaticAuthProvider {
    fn auth_header_for<'a>(
        &'a self,
        _user_id: &'a str,
        _tenant_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, Error>> + Send + 'a>> {
        Box::pin(async move { Ok(self.header.clone()) })
    }
}

/// Auth provider that exchanges a subject token for a user-scoped delegated token
/// via RFC 8693 Token Exchange.
pub struct TokenExchangeAuthProvider {
    client: reqwest::Client,
    exchange_url: String,
    client_id: String,
    client_secret: String,
    agent_token: String,
    user_tokens: Arc<RwLock<HashMap<String, String>>>,
    /// Cache of exchanged tokens: user_id -> (access_token, expires_at).
    token_cache: RwLock<HashMap<String, (String, std::time::Instant)>>,
}

/// Token exchange response per RFC 8693.
#[derive(Deserialize)]
struct TokenExchangeResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    token_type: Option<String>,
}

/// Request timeout for token exchange calls.
const TOKEN_EXCHANGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

impl TokenExchangeAuthProvider {
    pub fn new(
        exchange_url: impl Into<String>,
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        agent_token: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(TOKEN_EXCHANGE_TIMEOUT)
                .build()
                .unwrap_or_default(),
            exchange_url: exchange_url.into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            agent_token: agent_token.into(),
            user_tokens: Arc::new(RwLock::new(HashMap::new())),
            token_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Set the user tokens map (user_id -> subject_token).
    pub fn with_user_tokens(mut self, tokens: Arc<RwLock<HashMap<String, String>>>) -> Self {
        self.user_tokens = tokens;
        self
    }
}

impl AuthProvider for TokenExchangeAuthProvider {
    fn auth_header_for<'a>(
        &'a self,
        user_id: &'a str,
        _tenant_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, Error>> + Send + 'a>> {
        Box::pin(async move {
            // Check cache first
            if let Ok(cache) = self.token_cache.read()
                && let Some((token, expires_at)) = cache.get(user_id)
                && std::time::Instant::now() < *expires_at
            {
                return Ok(Some(format!("Bearer {token}")));
            }

            let subject_token = {
                let tokens = self
                    .user_tokens
                    .read()
                    .map_err(|e| Error::Mcp(format!("user_tokens lock poisoned: {e}")))?;
                tokens.get(user_id).cloned().ok_or_else(|| {
                    Error::Mcp(format!("No subject token found for user '{user_id}'"))
                })?
            };

            let response = self
                .client
                .post(&self.exchange_url)
                .form(&[
                    (
                        "grant_type",
                        "urn:ietf:params:oauth:grant-type:token-exchange",
                    ),
                    ("subject_token", &subject_token),
                    (
                        "subject_token_type",
                        "urn:ietf:params:oauth:token-type:access_token",
                    ),
                    ("actor_token", &self.agent_token),
                    (
                        "actor_token_type",
                        "urn:ietf:params:oauth:token-type:access_token",
                    ),
                    ("client_id", &self.client_id),
                    ("client_secret", &self.client_secret),
                ])
                .send()
                .await
                .map_err(|e| Error::Mcp(format!("Token exchange request failed: {e}")))?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                // Truncate error body to avoid leaking sensitive IdP details in logs
                let truncated = if body.len() > 512 {
                    &body[..512]
                } else {
                    &body
                };
                return Err(Error::Mcp(format!(
                    "Token exchange failed (HTTP {status}): {truncated}"
                )));
            }

            let token_response: TokenExchangeResponse = response
                .json()
                .await
                .map_err(|e| Error::Mcp(format!("Token exchange response parse error: {e}")))?;

            // Cache the exchanged token with expiry (default 5 minutes if not specified)
            let ttl = token_response.expires_in.unwrap_or(300);
            // Expire 30 seconds early to avoid using nearly-expired tokens
            let expires_at =
                std::time::Instant::now() + std::time::Duration::from_secs(ttl.saturating_sub(30));
            if let Ok(mut cache) = self.token_cache.write() {
                cache.insert(
                    user_id.to_string(),
                    (token_response.access_token.clone(), expires_at),
                );
            }

            let token_type = token_response.token_type.as_deref().unwrap_or("Bearer");
            Ok(Some(format!(
                "{token_type} {}",
                token_response.access_token
            )))
        })
    }
}

// --- HTTP transport ---

/// HTTP-based transport for Streamable HTTP MCP servers.
struct HttpTransport {
    client: reqwest::Client,
    endpoint: String,
    session_id: RwLock<Option<String>>,
    next_id: AtomicU64,
    auth_header: Option<String>,
}

impl HttpTransport {
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

        process_rpc_response(&json_str)
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

        // Consume the response body to allow HTTP connection reuse
        let _ = response.bytes().await;

        Ok(())
    }
}

// --- Stdio transport ---

/// I/O handles for an MCP server running as a child process.
///
/// Fields are dropped in declaration order: stdin first (signals EOF to child),
/// then reader, then the process handle.
struct StdioIo {
    stdin: tokio::process::ChildStdin,
    reader: tokio::io::BufReader<tokio::process::ChildStdout>,
    _process: tokio::process::Child,
}

/// Stdio-based transport for MCP servers spawned as child processes.
///
/// Communication uses newline-delimited JSON-RPC on stdin/stdout.
/// Access is serialized via a tokio `Mutex` to prevent interleaved I/O.
struct StdioTransport {
    io: tokio::sync::Mutex<StdioIo>,
    next_id: AtomicU64,
}

impl StdioTransport {
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    async fn rpc(&self, method: &str, params: Option<Value>) -> Result<Value, Error> {
        let id = self.next_id();
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
            id,
        };
        let line = serde_json::to_string(&request)? + "\n";

        // Timeout covers the entire write+read cycle to prevent hangs from
        // both unresponsive writes (server stopped reading stdin) and slow reads.
        let mut io = self.io.lock().await;
        let json_str = tokio::time::timeout(REQUEST_TIMEOUT, async {
            io.stdin
                .write_all(line.as_bytes())
                .await
                .map_err(|e| Error::Mcp(format!("stdio write error: {e}")))?;
            io.stdin
                .flush()
                .await
                .map_err(|e| Error::Mcp(format!("stdio flush error: {e}")))?;
            read_stdio_response(&mut io.reader, id).await
        })
        .await
        .map_err(|_| {
            Error::Mcp(format!(
                "MCP stdio server timed out after {}s for request {id}",
                REQUEST_TIMEOUT.as_secs()
            ))
        })??;
        process_rpc_response(&json_str)
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), Error> {
        let notification = JsonRpcNotification {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
        };
        let line = serde_json::to_string(&notification)? + "\n";

        let mut io = self.io.lock().await;
        tokio::time::timeout(REQUEST_TIMEOUT, async {
            io.stdin
                .write_all(line.as_bytes())
                .await
                .map_err(|e| Error::Mcp(format!("stdio write error: {e}")))?;
            io.stdin
                .flush()
                .await
                .map_err(|e| Error::Mcp(format!("stdio flush error: {e}")))?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|_| {
            Error::Mcp(format!(
                "MCP stdio notification timed out after {}s",
                REQUEST_TIMEOUT.as_secs()
            ))
        })??;
        Ok(())
    }
}

// --- Unified transport ---

/// Unified MCP transport supporting both Streamable HTTP and stdio protocols.
enum Transport {
    Http(HttpTransport),
    Stdio(Box<StdioTransport>),
}

impl Transport {
    async fn rpc(&self, method: &str, params: Option<Value>) -> Result<Value, Error> {
        match self {
            Transport::Http(t) => t.rpc(method, params).await,
            Transport::Stdio(t) => t.rpc(method, params).await,
        }
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), Error> {
        match self {
            Transport::Http(t) => t.notify(method, params).await,
            Transport::Stdio(t) => t.notify(method, params).await,
        }
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
    transport: Arc<Transport>,
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
            match self.transport.call_tool(&self.def.name, input).await {
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
/// Connects to an MCP server via Streamable HTTP or stdio, performs the
/// handshake, discovers tools, and produces `Vec<Arc<dyn Tool>>` that plug
/// into `AgentRunnerBuilder::tools()`.
pub struct McpClient {
    transport: Arc<Transport>,
    tools: Vec<McpToolDef>,
}

impl McpClient {
    /// Connect to an MCP server over Streamable HTTP and discover available tools.
    ///
    /// Performs the full handshake: initialize → notifications/initialized → tools/list.
    pub async fn connect(endpoint: &str) -> Result<Self, Error> {
        Self::connect_http(endpoint, None).await
    }

    /// Connect to an MCP server over Streamable HTTP with an authorization header.
    ///
    /// Use this for agentgateway or other authenticated MCP proxies.
    /// The `auth_header` is sent as the `Authorization` header value
    /// (e.g., `"Bearer <token>"`).
    pub async fn connect_with_auth(
        endpoint: &str,
        auth_header: impl Into<String>,
    ) -> Result<Self, Error> {
        Self::connect_http(endpoint, Some(auth_header.into())).await
    }

    /// Connect to an MCP server via stdio (spawns a child process).
    ///
    /// The child process communicates using newline-delimited JSON-RPC
    /// on stdin/stdout (MCP stdio transport). The process is killed
    /// when the client is dropped.
    pub async fn connect_stdio(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self, Error> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args)
            .envs(env.iter())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().map_err(|e| {
            Error::Mcp(format!("Failed to spawn MCP stdio server '{command}': {e}"))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Mcp("Failed to capture stdin of MCP server".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Mcp("Failed to capture stdout of MCP server".into()))?;

        // Drain stderr in background to prevent pipe buffer deadlocks and log debug output.
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut reader = tokio::io::BufReader::new(stderr);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                tracing::debug!(
                                    target: "mcp_stdio_stderr",
                                    "{}",
                                    trimmed
                                );
                            }
                        }
                    }
                }
            });
        }

        let transport = Arc::new(Transport::Stdio(Box::new(StdioTransport {
            io: tokio::sync::Mutex::new(StdioIo {
                stdin,
                reader: tokio::io::BufReader::new(stdout),
                _process: child,
            }),
            next_id: AtomicU64::new(0),
        })));

        Self::handshake_and_discover(transport).await
    }

    async fn connect_http(endpoint: &str, auth_header: Option<String>) -> Result<Self, Error> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()?;

        let transport = Arc::new(Transport::Http(HttpTransport {
            client,
            endpoint: endpoint.to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header,
        }));

        Self::handshake_and_discover(transport).await
    }

    /// Perform MCP handshake and tool discovery on the given transport.
    async fn handshake_and_discover(transport: Arc<Transport>) -> Result<Self, Error> {
        // Initialize — for HTTP, rpc() captures Mcp-Session-Id automatically.
        transport
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

        transport.notify("notifications/initialized", None).await?;

        // Paginate tools/list — collect all pages via nextCursor
        let mut all_tools = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = cursor.as_ref().map(|c| serde_json::json!({"cursor": c}));
            let tools_result = transport.rpc("tools/list", params).await?;
            let page: McpToolsListResult = serde_json::from_value(tools_result)?;
            all_tools.extend(page.tools);
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }

        Ok(Self {
            transport,
            tools: all_tools,
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
        let transport = self.transport;
        self.tools
            .into_iter()
            .map(|t| {
                let tool: Arc<dyn Tool> = Arc::new(McpTool {
                    transport: Arc::clone(&transport),
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

    #[test]
    fn tool_result_non_text_only_shows_placeholder() {
        let result = McpCallToolResult {
            content: vec![
                McpContent {
                    content_type: "image".into(),
                    text: None,
                },
                McpContent {
                    content_type: "resource".into(),
                    text: None,
                },
            ],
            is_error: false,
        };

        let output = mcp_result_to_tool_output(result);
        assert!(output.content.contains("2 non-text content block(s)"));
        assert!(!output.is_error);
    }

    #[test]
    fn tool_result_mixed_text_and_non_text_returns_text() {
        // When there's both text and non-text, only text is returned (no placeholder)
        let result = McpCallToolResult {
            content: vec![
                McpContent {
                    content_type: "text".into(),
                    text: Some("real text".into()),
                },
                McpContent {
                    content_type: "image".into(),
                    text: None,
                },
            ],
            is_error: false,
        };

        let output = mcp_result_to_tool_output(result);
        assert_eq!(output.content, "real text");
    }

    // --- process_rpc_response tests ---

    #[test]
    fn process_rpc_response_success() {
        let json_str = r#"{"jsonrpc":"2.0","result":{"tools":[]},"id":1}"#;
        let value = process_rpc_response(json_str).unwrap();
        assert_eq!(value, json!({"tools": []}));
    }

    #[test]
    fn process_rpc_response_error() {
        let json_str =
            r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"Method not found"},"id":1}"#;
        let err = process_rpc_response(json_str).unwrap_err();
        assert!(err.to_string().contains("JSON-RPC error -32601"));
        assert!(err.to_string().contains("Method not found"));
    }

    #[test]
    fn process_rpc_response_missing_both() {
        let json_str = r#"{"jsonrpc":"2.0","id":1}"#;
        let err = process_rpc_response(json_str).unwrap_err();
        assert!(err.to_string().contains("missing both result and error"));
    }

    // --- read_stdio_response tests ---

    #[tokio::test]
    async fn read_stdio_response_finds_matching_id() {
        let (mut tx, rx) = tokio::io::duplex(4096);
        let mut reader = tokio::io::BufReader::new(rx);

        tokio::spawn(async move {
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"result\":{\"ok\":true},\"id\":1}\n")
                .await
                .unwrap();
        });

        let response = read_stdio_response(&mut reader, 1).await.unwrap();
        assert!(response.contains("\"id\":1"));
        assert!(response.contains("\"ok\":true"));
    }

    #[tokio::test]
    async fn read_stdio_response_skips_notifications() {
        let (mut tx, rx) = tokio::io::duplex(4096);
        let mut reader = tokio::io::BufReader::new(rx);

        tokio::spawn(async move {
            // Server sends a notification first, then the actual response.
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n")
                .await
                .unwrap();
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"result\":{\"tools\":[]},\"id\":1}\n")
                .await
                .unwrap();
        });

        let response = read_stdio_response(&mut reader, 1).await.unwrap();
        assert!(response.contains("\"id\":1"));
        assert!(response.contains("\"tools\""));
    }

    #[tokio::test]
    async fn read_stdio_response_skips_null_id() {
        let (mut tx, rx) = tokio::io::duplex(4096);
        let mut reader = tokio::io::BufReader::new(rx);

        tokio::spawn(async move {
            // Response with null ID (notification-like), then actual response.
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"result\":{},\"id\":null}\n")
                .await
                .unwrap();
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"result\":{\"found\":true},\"id\":2}\n")
                .await
                .unwrap();
        });

        let response = read_stdio_response(&mut reader, 2).await.unwrap();
        assert!(response.contains("\"id\":2"));
        assert!(response.contains("\"found\":true"));
    }

    #[tokio::test]
    async fn read_stdio_response_skips_non_json() {
        let (mut tx, rx) = tokio::io::duplex(4096);
        let mut reader = tokio::io::BufReader::new(rx);

        tokio::spawn(async move {
            // Server emits debug text before JSON response.
            tx.write_all(b"[DEBUG] initializing server...\n")
                .await
                .unwrap();
            tx.write_all(b"\n").await.unwrap(); // empty line
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"result\":{},\"id\":0}\n")
                .await
                .unwrap();
        });

        let response = read_stdio_response(&mut reader, 0).await.unwrap();
        assert!(response.contains("\"id\":0"));
    }

    #[tokio::test]
    async fn read_stdio_response_eof_errors() {
        let (tx, rx) = tokio::io::duplex(4096);
        let mut reader = tokio::io::BufReader::new(rx);

        // Close the write side immediately — simulates process exit.
        drop(tx);

        let err = read_stdio_response(&mut reader, 0).await.unwrap_err();
        assert!(
            err.to_string().contains("closed unexpectedly"),
            "error: {err}"
        );
    }

    #[tokio::test]
    async fn read_stdio_response_skips_wrong_id() {
        let (mut tx, rx) = tokio::io::duplex(4096);
        let mut reader = tokio::io::BufReader::new(rx);

        tokio::spawn(async move {
            // Response for a different request ID, then the correct one.
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"result\":{\"wrong\":true},\"id\":99}\n")
                .await
                .unwrap();
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"result\":{\"right\":true},\"id\":3}\n")
                .await
                .unwrap();
        });

        let response = read_stdio_response(&mut reader, 3).await.unwrap();
        assert!(response.contains("\"right\":true"));
    }

    #[tokio::test]
    async fn read_stdio_response_timeout_prevents_hang() {
        // Simulate a server that never responds — without timeout this would hang forever.
        let (_tx, rx) = tokio::io::duplex(4096);
        let mut reader = tokio::io::BufReader::new(rx);

        let result = tokio::time::timeout(
            Duration::from_millis(50),
            read_stdio_response(&mut reader, 0),
        )
        .await;

        assert!(result.is_err(), "should have timed out");
    }

    // --- HttpTransport tests ---

    #[test]
    fn http_transport_next_id_is_monotonic() {
        let transport = HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        };

        assert_eq!(transport.next_id(), 0);
        assert_eq!(transport.next_id(), 1);
        assert_eq!(transport.next_id(), 2);
    }

    // --- McpTool tests ---

    #[test]
    fn mcp_tool_returns_correct_definition() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));

        let expected_def = ToolDefinition {
            name: "read_file".into(),
            description: "Read a file".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}}
            }),
        };

        let tool = McpTool {
            transport,
            def: expected_def.clone(),
        };

        let def = tool.definition();
        assert_eq!(def, expected_def);
    }

    // --- AuthProvider tests ---

    #[tokio::test]
    async fn static_auth_provider_returns_header() {
        let provider = StaticAuthProvider::new(Some("Bearer xyz".to_string()));
        let result = provider.auth_header_for("user1", "tenant1").await.unwrap();
        assert_eq!(result, Some("Bearer xyz".to_string()));
    }

    #[tokio::test]
    async fn static_auth_provider_returns_none() {
        let provider = StaticAuthProvider::new(None);
        let result = provider.auth_header_for("user1", "tenant1").await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn static_auth_provider_ignores_user_tenant() {
        let provider = StaticAuthProvider::new(Some("Bearer abc".to_string()));
        let r1 = provider.auth_header_for("alice", "acme").await.unwrap();
        let r2 = provider.auth_header_for("bob", "globex").await.unwrap();
        assert_eq!(r1, r2);
        assert_eq!(r1, Some("Bearer abc".to_string()));
    }

    #[tokio::test]
    async fn token_exchange_provider_missing_user_token() {
        let user_tokens = Arc::new(std::sync::RwLock::new(HashMap::<String, String>::new()));
        let provider = TokenExchangeAuthProvider::new(
            "https://idp.example.com/token",
            "client-id",
            "client-secret",
            "agent-token-xyz",
        )
        .with_user_tokens(user_tokens);

        let result = provider.auth_header_for("unknown-user", "tenant1").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("unknown-user"),
            "error should mention the user_id: {err_msg}"
        );
    }

    #[tokio::test]
    async fn mcp_tool_execute_catches_network_errors() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://127.0.0.1:1".to_string(), // nothing listening
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));

        let tool = McpTool {
            transport,
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
