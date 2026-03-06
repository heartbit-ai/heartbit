use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const PROTOCOL_VERSION: &str = "2025-11-25";
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

// --- Server capabilities (parsed from initialize response) ---

#[derive(Debug, Default, Deserialize)]
#[allow(dead_code)]
struct ServerCapabilities {
    #[serde(default)]
    resources: Option<ResourcesCapability>,
    #[serde(default)]
    prompts: Option<PromptsCapability>,
    #[serde(default)]
    logging: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct ResourcesCapability {
    #[serde(default)]
    subscribe: bool,
    #[serde(default)]
    list_changed: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct PromptsCapability {
    #[serde(default)]
    list_changed: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct InitializeResult {
    #[serde(default)]
    capabilities: ServerCapabilities,
    #[serde(default)]
    server_info: Option<Value>,
}

// --- Resource types ---

/// A resource definition from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResourceDef {
    pub uri: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct McpResourcesListResult {
    resources: Vec<McpResourceDef>,
    #[serde(default, rename = "nextCursor")]
    next_cursor: Option<String>,
}

/// Content returned by `resources/read`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResourceContent {
    pub uri: String,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub blob: Option<String>,
}

#[derive(Debug, Deserialize)]
struct McpResourceReadResult {
    contents: Vec<McpResourceContent>,
}

// --- Prompt types ---

/// A prompt definition from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<McpPromptArgument>,
}

/// An argument for an MCP prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

#[derive(Debug, Deserialize)]
struct McpPromptsListResult {
    prompts: Vec<McpPromptDef>,
    #[serde(default, rename = "nextCursor")]
    next_cursor: Option<String>,
}

/// A message returned by `prompts/get`.
#[derive(Debug, Clone, Deserialize)]
pub struct McpPromptMessage {
    pub role: String,
    pub content: McpPromptMessageContent,
}

/// Content of a prompt message.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpPromptMessageContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct McpPromptGetResult {
    #[serde(default)]
    description: Option<String>,
    messages: Vec<McpPromptMessage>,
}

// --- MCP Logging support ---

/// Forward an MCP server log notification to tracing.
///
/// Called by stdio and HTTP transports when they encounter a notification
/// with `method: "notifications/message"`.
fn handle_log_notification(value: &Value) {
    if let Some(params) = value.get("params") {
        let level = params
            .get("level")
            .and_then(|v| v.as_str())
            .unwrap_or("info");
        let logger = params
            .get("logger")
            .and_then(|v| v.as_str())
            .unwrap_or("mcp");
        let data = params.get("data").and_then(|v| v.as_str()).unwrap_or("");
        match level {
            "error" | "critical" | "alert" | "emergency" => {
                tracing::error!(target: "mcp_server", logger = logger, "{data}");
            }
            "warning" => {
                tracing::warn!(target: "mcp_server", logger = logger, "{data}");
            }
            "debug" => {
                tracing::debug!(target: "mcp_server", logger = logger, "{data}");
            }
            _ => {
                tracing::info!(target: "mcp_server", logger = logger, "{data}");
            }
        }
    }
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
        if let Ok(value) = serde_json::from_str::<Value>(event) {
            // Forward log notifications from SSE events
            if value.get("method").and_then(|m| m.as_str()) == Some("notifications/message") {
                handle_log_notification(&value);
                continue;
            }
            if value.get("id").and_then(|v| v.as_u64()) == Some(expected_id) {
                return Ok(event.clone());
            }
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

        // Notifications have no "id" or null id — handle logging, skip others.
        match value.get("id") {
            None | Some(&Value::Null) => {
                if value.get("method").and_then(|m| m.as_str()) == Some("notifications/message") {
                    handle_log_notification(&value);
                }
                continue;
            }
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

/// HTTP header name used to pass the tenant ID to the IdP/MCP authorization server.
const TENANT_ID_HEADER: &str = "X-Tenant-ID";

/// Auth provider that exchanges a subject token for a user-scoped delegated token
/// via RFC 8693 Token Exchange.
pub struct TokenExchangeAuthProvider {
    client: reqwest::Client,
    exchange_url: String,
    client_id: String,
    client_secret: String,
    /// NHI tenant ID for `client_credentials` grant. When set, `agent_token` is
    /// auto-fetched and cached; the static `agent_token` field is ignored.
    tenant_id: Option<String>,
    /// Static fallback agent token. Used only when `tenant_id` is absent.
    agent_token: String,
    /// OAuth scopes for the `client_credentials` agent token grant.
    /// Defaults to `["openid"]` when empty.
    scopes: Vec<String>,
    /// Cache for the auto-fetched agent token: (access_token, expires_at).
    /// Uses std::sync::RwLock because the lock is never held across `.await`.
    agent_token_cache: RwLock<Option<(String, Instant)>>,
    /// Subject tokens for token exchange: key is `"{tenant_id}:{user_id}"`.
    /// Populated externally (e.g. by the daemon HTTP handler when a user submits a task).
    user_tokens: Arc<RwLock<HashMap<String, String>>>,
    /// Cache of exchanged tokens: (tenant_id, user_id) -> (access_token, expires_at).
    /// Keyed by both tenant and user to prevent cross-tenant token leakage.
    token_cache: RwLock<HashMap<(String, String), (String, Instant)>>,
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
            tenant_id: None,
            agent_token: agent_token.into(),
            scopes: Vec::new(),
            agent_token_cache: RwLock::new(None),
            user_tokens: Arc::new(RwLock::new(HashMap::new())),
            token_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Set the NHI tenant ID for automatic `client_credentials` agent token fetch.
    pub fn with_tenant_id(mut self, tenant_id: Option<String>) -> Self {
        self.tenant_id = tenant_id;
        self
    }

    /// Set the OAuth scopes for the `client_credentials` agent token grant.
    pub fn with_scopes(mut self, scopes: Vec<String>) -> Self {
        self.scopes = scopes;
        self
    }

    /// Set the user tokens map (`"{tenant_id}:{user_id}"` -> subject_token).
    pub fn with_user_tokens(mut self, tokens: Arc<RwLock<HashMap<String, String>>>) -> Self {
        self.user_tokens = tokens;
        self
    }

    /// Get a reference to the shared user tokens map for external population.
    pub fn user_tokens(&self) -> &Arc<RwLock<HashMap<String, String>>> {
        &self.user_tokens
    }

    /// Returns a valid agent token, fetching a fresh one via `client_credentials` if needed.
    ///
    /// When `tenant_id` is configured, auto-fetches and caches the token using
    /// `client_credentials` grant (AWS/GCP SDK pattern). Falls back to the static
    /// `agent_token` when `tenant_id` is absent.
    async fn ensure_valid_agent_token(&self) -> Result<String, Error> {
        // Check cache (read lock — not held across .await per codebase convention)
        {
            let cache = self
                .agent_token_cache
                .read()
                .map_err(|e| Error::Mcp(format!("agent_token_cache lock poisoned: {e}")))?;
            if let Some((token, expires_at)) = &*cache
                && Instant::now() < *expires_at
            {
                return Ok(token.clone());
            }
        }
        // Auto-fetch via client_credentials when tenant_id is configured
        if let Some(tenant_id) = &self.tenant_id {
            let scope = if self.scopes.is_empty() {
                "openid".to_string()
            } else {
                self.scopes.join(" ")
            };
            let response = self
                .client
                .post(&self.exchange_url)
                .header(TENANT_ID_HEADER, tenant_id)
                .form(&[
                    ("grant_type", "client_credentials"),
                    ("client_id", &self.client_id),
                    ("client_secret", &self.client_secret),
                    ("scope", &scope),
                ])
                .send()
                .await
                .map_err(|e| Error::Mcp(format!("Agent token fetch failed: {e}")))?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let cut = crate::tool::builtins::floor_char_boundary(&body, 512);
                return Err(Error::Mcp(format!(
                    "Agent token fetch failed (HTTP {status}): {}",
                    &body[..cut]
                )));
            }

            let resp: TokenExchangeResponse = response
                .json()
                .await
                .map_err(|e| Error::Mcp(format!("Agent token response parse error: {e}")))?;

            let ttl = resp.expires_in.unwrap_or(300).min(3600).saturating_sub(30);
            let expires_at = Instant::now() + Duration::from_secs(ttl);
            *self
                .agent_token_cache
                .write()
                .map_err(|e| Error::Mcp(format!("agent_token_cache lock poisoned: {e}")))? =
                Some((resp.access_token.clone(), expires_at));
            return Ok(resp.access_token);
        }
        // Fallback: static token from config
        Ok(self.agent_token.clone())
    }
}

impl AuthProvider for TokenExchangeAuthProvider {
    fn auth_header_for<'a>(
        &'a self,
        user_id: &'a str,
        tenant_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, Error>> + Send + 'a>> {
        Box::pin(async move {
            // Check cache first — keyed by (tenant_id, user_id) to prevent cross-tenant leakage
            let cache_key = (tenant_id.to_string(), user_id.to_string());
            if let Ok(cache) = self.token_cache.read()
                && let Some((token, expires_at)) = cache.get(&cache_key)
                && Instant::now() < *expires_at
            {
                return Ok(Some(format!("Bearer {token}")));
            }

            let token_key = format!("{tenant_id}:{user_id}");
            let subject_token = {
                let tokens = self
                    .user_tokens
                    .read()
                    .map_err(|e| Error::Mcp(format!("user_tokens lock poisoned: {e}")))?;
                tokens.get(&token_key).cloned().ok_or_else(|| {
                    Error::Mcp(format!(
                        "No subject token found for user '{user_id}' in tenant '{tenant_id}'"
                    ))
                })?
            };

            let agent_token = self.ensure_valid_agent_token().await?;
            let response = self
                .client
                .post(&self.exchange_url)
                .header(TENANT_ID_HEADER, tenant_id)
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
                    ("actor_token", &agent_token),
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
                let cut = crate::tool::builtins::floor_char_boundary(&body, 512);
                return Err(Error::Mcp(format!(
                    "Token exchange failed (HTTP {status}): {}",
                    &body[..cut]
                )));
            }

            let token_response: TokenExchangeResponse = response
                .json()
                .await
                .map_err(|e| Error::Mcp(format!("Token exchange response parse error: {e}")))?;

            // Cache the exchanged token with expiry (default 5 min, max 1 hour)
            let ttl = token_response.expires_in.unwrap_or(300).min(3600);
            // Expire 30 seconds early to avoid using nearly-expired tokens
            let now = Instant::now();
            let expires_at = now + Duration::from_secs(ttl.saturating_sub(30));
            if let Ok(mut cache) = self.token_cache.write() {
                // Prune expired entries to prevent unbounded growth in multi-tenant deployments
                cache.retain(|_, (_, exp)| now < *exp);
                cache.insert(cache_key, (token_response.access_token.clone(), expires_at));
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

// --- McpResourceTool ---

/// Bridge that exposes an MCP resource as a callable tool.
///
/// Tool name: `mcp_resource_{sanitized_name}`. Calling it reads the resource
/// and returns its content as text.
struct McpResourceTool {
    transport: Arc<Transport>,
    resource: McpResourceDef,
    tool_name: String,
}

impl Tool for McpResourceTool {
    fn definition(&self) -> ToolDefinition {
        let desc = self
            .resource
            .description
            .clone()
            .unwrap_or_else(|| format!("Read MCP resource: {}", self.resource.uri));
        ToolDefinition {
            name: self.tool_name.clone(),
            description: desc,
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
            }),
        }
    }

    fn execute(
        &self,
        _input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let params = serde_json::json!({ "uri": self.resource.uri });
            match self.transport.rpc("resources/read", Some(params)).await {
                Ok(value) => {
                    let result: McpResourceReadResult = serde_json::from_value(value)?;
                    let text: String = result
                        .contents
                        .iter()
                        .filter_map(|c| c.text.as_deref())
                        .collect::<Vec<_>>()
                        .join("\n");
                    if text.is_empty() {
                        Ok(ToolOutput::success(format!(
                            "[Resource {} returned no text content]",
                            self.resource.uri
                        )))
                    } else {
                        Ok(ToolOutput::success(text))
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        resource = %self.resource.uri,
                        error = %e,
                        "MCP resource read failed"
                    );
                    Ok(ToolOutput::error(e.to_string()))
                }
            }
        })
    }
}

// --- McpPromptTool ---

/// Bridge that exposes an MCP prompt as a callable tool.
struct McpPromptTool {
    transport: Arc<Transport>,
    prompt: McpPromptDef,
    tool_name: String,
}

impl Tool for McpPromptTool {
    fn definition(&self) -> ToolDefinition {
        let desc = self
            .prompt
            .description
            .clone()
            .unwrap_or_else(|| format!("Get MCP prompt: {}", self.prompt.name));
        // Build input schema from prompt arguments
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for arg in &self.prompt.arguments {
            let mut prop = serde_json::Map::new();
            prop.insert("type".into(), serde_json::json!("string"));
            if let Some(desc) = &arg.description {
                prop.insert("description".into(), serde_json::json!(desc));
            }
            properties.insert(arg.name.clone(), Value::Object(prop));
            if arg.required {
                required.push(serde_json::json!(arg.name));
            }
        }
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": properties,
        });
        if !required.is_empty() {
            schema["required"] = Value::Array(required);
        }
        ToolDefinition {
            name: self.tool_name.clone(),
            description: desc,
            input_schema: schema,
        }
    }

    fn execute(
        &self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let arguments = if input.is_null() || input.as_object().is_some_and(|m| m.is_empty()) {
                None
            } else {
                Some(input)
            };
            let mut params = serde_json::json!({ "name": self.prompt.name });
            if let Some(args) = arguments {
                params["arguments"] = args;
            }
            match self.transport.rpc("prompts/get", Some(params)).await {
                Ok(value) => {
                    let result: McpPromptGetResult = serde_json::from_value(value)?;
                    let text: String = result
                        .messages
                        .iter()
                        .map(|m| {
                            let content = m.content.text.as_deref().unwrap_or("");
                            format!("[{}] {}", m.role, content)
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    Ok(ToolOutput::success(text))
                }
                Err(e) => {
                    tracing::warn!(
                        prompt = %self.prompt.name,
                        error = %e,
                        "MCP prompt get failed"
                    );
                    Ok(ToolOutput::error(e.to_string()))
                }
            }
        })
    }
}

// --- McpClient ---

// --- Sampling types (Phase 3) ---

/// Request from MCP server asking the client to create an LLM completion.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingRequest {
    pub messages: Vec<SamplingMessage>,
    #[serde(default)]
    pub model_preferences: Option<SamplingModelPreferences>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingMessage {
    pub role: String,
    pub content: SamplingContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingModelPreferences {
    #[serde(default)]
    pub hints: Vec<SamplingModelHint>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SamplingModelHint {
    #[serde(default)]
    pub name: Option<String>,
}

/// Response to a `sampling/createMessage` request.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct SamplingResponse {
    role: String,
    content: SamplingContent,
    model: String,
}

/// Callback for handling sampling requests from MCP servers.
///
/// Takes a `SamplingRequest` and returns the model's response text and model name.
pub type SamplingHandler = Arc<
    dyn Fn(SamplingRequest) -> Pin<Box<dyn Future<Output = Result<(String, String), Error>> + Send>>
        + Send
        + Sync,
>;

/// Sanitize a name into a valid tool identifier (alphanumeric + underscores).
fn sanitize_tool_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Client for the Model Context Protocol (MCP).
///
/// Connects to an MCP server via Streamable HTTP or stdio, performs the
/// handshake, discovers tools/resources/prompts, and produces `Vec<Arc<dyn Tool>>`
/// that plug into `AgentRunnerBuilder::tools()`.
/// A root directory exposed to MCP servers via the `roots` capability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRoot {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

pub struct McpClient {
    transport: Arc<Transport>,
    tools: Vec<McpToolDef>,
    resources: Vec<McpResourceDef>,
    prompts: Vec<McpPromptDef>,
    capabilities: ServerCapabilities,
    sampling_handler: Option<SamplingHandler>,
    /// Root directories exposed to the MCP server via `roots/list` requests.
    /// Populated via `with_roots()`.
    roots: Vec<McpRoot>,
}

impl McpClient {
    /// Get the configured roots.
    pub fn roots(&self) -> &[McpRoot] {
        &self.roots
    }

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

    /// Set a sampling handler to respond to `sampling/createMessage` requests
    /// from the MCP server. This enables the server to request LLM completions
    /// from the client.
    pub fn with_sampling(mut self, handler: SamplingHandler) -> Self {
        self.sampling_handler = Some(handler);
        self
    }

    /// Set root directories to expose to MCP servers via the `roots` capability.
    pub fn with_roots(mut self, roots: Vec<McpRoot>) -> Self {
        self.roots = roots;
        self
    }

    /// Notify the server that the list of roots has changed.
    pub async fn send_roots_changed(&self) -> Result<(), Error> {
        self.transport
            .notify("notifications/roots/list_changed", None)
            .await
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

    /// Perform MCP handshake and tool/resource/prompt discovery on the given transport.
    async fn handshake_and_discover(transport: Arc<Transport>) -> Result<Self, Error> {
        // Initialize — for HTTP, rpc() captures Mcp-Session-Id automatically.
        let init_result = transport
            .rpc(
                "initialize",
                Some(serde_json::json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "sampling": {},
                        "roots": { "listChanged": true }
                    },
                    "clientInfo": {
                        "name": "heartbit",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                })),
            )
            .await?;

        let init: InitializeResult = serde_json::from_value(init_result).unwrap_or_default();

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

        // Discover resources if the server advertises support
        let mut all_resources = Vec::new();
        if init.capabilities.resources.is_some() {
            let mut cursor: Option<String> = None;
            loop {
                let params = cursor.as_ref().map(|c| serde_json::json!({"cursor": c}));
                match transport.rpc("resources/list", params).await {
                    Ok(value) => {
                        let page: McpResourcesListResult = serde_json::from_value(value)?;
                        all_resources.extend(page.resources);
                        cursor = page.next_cursor;
                        if cursor.is_none() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "resources/list failed, skipping resource discovery");
                        break;
                    }
                }
            }
        }

        // Discover prompts if the server advertises support
        let mut all_prompts = Vec::new();
        if init.capabilities.prompts.is_some() {
            let mut cursor: Option<String> = None;
            loop {
                let params = cursor.as_ref().map(|c| serde_json::json!({"cursor": c}));
                match transport.rpc("prompts/list", params).await {
                    Ok(value) => {
                        let page: McpPromptsListResult = serde_json::from_value(value)?;
                        all_prompts.extend(page.prompts);
                        cursor = page.next_cursor;
                        if cursor.is_none() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "prompts/list failed, skipping prompt discovery");
                        break;
                    }
                }
            }
        }

        Ok(Self {
            transport,
            tools: all_tools,
            resources: all_resources,
            prompts: all_prompts,
            capabilities: init.capabilities,
            sampling_handler: None,
            roots: Vec::new(),
        })
    }

    /// Get tool definitions without consuming the client.
    ///
    /// Useful when you only need the schemas (e.g., for Restate task payloads)
    /// and don't need the executable tool instances.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(mcp_tool_to_definition).collect()
    }

    /// Get discovered resource definitions.
    pub fn resource_definitions(&self) -> &[McpResourceDef] {
        &self.resources
    }

    /// Get discovered prompt definitions.
    pub fn prompt_definitions(&self) -> &[McpPromptDef] {
        &self.prompts
    }

    /// Whether the server supports resource subscriptions.
    pub fn supports_resource_subscribe(&self) -> bool {
        self.capabilities
            .resources
            .as_ref()
            .is_some_and(|r| r.subscribe)
    }

    /// Read a specific resource by URI.
    pub async fn resource_read(&self, uri: &str) -> Result<Vec<McpResourceContent>, Error> {
        let params = serde_json::json!({ "uri": uri });
        let value = self.transport.rpc("resources/read", Some(params)).await?;
        let result: McpResourceReadResult = serde_json::from_value(value)?;
        Ok(result.contents)
    }

    /// Set the server's log level via `logging/setLevel`.
    pub async fn set_log_level(&self, level: &str) -> Result<(), Error> {
        let params = serde_json::json!({ "level": level });
        self.transport.rpc("logging/setLevel", Some(params)).await?;
        Ok(())
    }

    /// Subscribe to resource change notifications.
    pub async fn resource_subscribe(&self, uri: &str) -> Result<(), Error> {
        let params = serde_json::json!({ "uri": uri });
        self.transport
            .rpc("resources/subscribe", Some(params))
            .await?;
        Ok(())
    }

    /// Get a prompt by name with optional arguments.
    pub async fn prompt_get(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<Vec<McpPromptMessage>, Error> {
        let mut params = serde_json::json!({ "name": name });
        if let Some(args) = arguments {
            params["arguments"] = args;
        }
        let value = self.transport.rpc("prompts/get", Some(params)).await?;
        let result: McpPromptGetResult = serde_json::from_value(value)?;
        Ok(result.messages)
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

    /// Convert discovered MCP resources into callable `Arc<dyn Tool>` instances.
    ///
    /// Each resource becomes a tool named `mcp_resource_{sanitized_name}`.
    pub fn into_resource_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.resources
            .iter()
            .map(|r| {
                let tool_name = format!("mcp_resource_{}", sanitize_tool_name(&r.name));
                let tool: Arc<dyn Tool> = Arc::new(McpResourceTool {
                    transport: Arc::clone(&self.transport),
                    resource: r.clone(),
                    tool_name,
                });
                tool
            })
            .collect()
    }

    /// Convert discovered MCP prompts into callable `Arc<dyn Tool>` instances.
    ///
    /// Each prompt becomes a tool named `mcp_prompt_{sanitized_name}`.
    pub fn into_prompt_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.prompts
            .iter()
            .map(|p| {
                let tool_name = format!("mcp_prompt_{}", sanitize_tool_name(&p.name));
                let tool: Arc<dyn Tool> = Arc::new(McpPromptTool {
                    transport: Arc::clone(&self.transport),
                    prompt: p.clone(),
                    tool_name,
                });
                tool
            })
            .collect()
    }

    /// Convert all discovered capabilities (tools + resources + prompts) into `Arc<dyn Tool>`.
    pub fn into_all_tools(self) -> Vec<Arc<dyn Tool>> {
        let transport = &self.transport;
        let mut all: Vec<Arc<dyn Tool>> = self
            .tools
            .iter()
            .map(|t| -> Arc<dyn Tool> {
                Arc::new(McpTool {
                    transport: Arc::clone(transport),
                    def: mcp_tool_to_definition(t),
                })
            })
            .collect();
        for r in &self.resources {
            let tool_name = format!("mcp_resource_{}", sanitize_tool_name(&r.name));
            all.push(Arc::new(McpResourceTool {
                transport: Arc::clone(transport),
                resource: r.clone(),
                tool_name,
            }));
        }
        for p in &self.prompts {
            let tool_name = format!("mcp_prompt_{}", sanitize_tool_name(&p.name));
            all.push(Arc::new(McpPromptTool {
                transport: Arc::clone(transport),
                prompt: p.clone(),
                tool_name,
            }));
        }
        all
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

    // --- Server capabilities parsing ---

    #[test]
    fn server_capabilities_parses_full() {
        let json = json!({
            "capabilities": {
                "resources": { "subscribe": true, "listChanged": true },
                "prompts": { "listChanged": false },
                "logging": {},
                "tools": { "listChanged": true }
            },
            "serverInfo": { "name": "test-server", "version": "1.0" }
        });
        let result: InitializeResult = serde_json::from_value(json).unwrap();
        assert!(result.capabilities.resources.is_some());
        let res = result.capabilities.resources.unwrap();
        assert!(res.subscribe);
        assert!(res.list_changed);
        assert!(result.capabilities.prompts.is_some());
    }

    #[test]
    fn server_capabilities_parses_empty() {
        let json = json!({
            "capabilities": {},
        });
        let result: InitializeResult = serde_json::from_value(json).unwrap();
        assert!(result.capabilities.resources.is_none());
        assert!(result.capabilities.prompts.is_none());
    }

    #[test]
    fn server_capabilities_defaults_on_missing() {
        let json = json!({});
        let result: InitializeResult = serde_json::from_value(json).unwrap();
        assert!(result.capabilities.resources.is_none());
        assert!(result.capabilities.prompts.is_none());
    }

    #[test]
    fn server_capabilities_resources_only() {
        let json = json!({
            "capabilities": {
                "resources": {}
            }
        });
        let result: InitializeResult = serde_json::from_value(json).unwrap();
        assert!(result.capabilities.resources.is_some());
        let res = result.capabilities.resources.unwrap();
        assert!(!res.subscribe); // defaults to false
        assert!(!res.list_changed);
        assert!(result.capabilities.prompts.is_none());
    }

    // --- Resource types ---

    #[test]
    fn resource_def_serde_roundtrip() {
        let def = McpResourceDef {
            uri: "file:///README.md".into(),
            name: "README".into(),
            description: Some("Project readme".into()),
            mime_type: Some("text/markdown".into()),
        };
        let json = serde_json::to_value(&def).unwrap();
        assert_eq!(json["uri"], "file:///README.md");
        assert_eq!(json["name"], "README");
        let parsed: McpResourceDef = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.uri, "file:///README.md");
        assert_eq!(parsed.mime_type.as_deref(), Some("text/markdown"));
    }

    #[test]
    fn resource_def_minimal() {
        let json = json!({"uri": "test://x", "name": "x"});
        let def: McpResourceDef = serde_json::from_value(json).unwrap();
        assert_eq!(def.uri, "test://x");
        assert!(def.description.is_none());
        assert!(def.mime_type.is_none());
    }

    #[test]
    fn resources_list_result_parsing() {
        let json = json!({
            "resources": [
                {
                    "uri": "file:///config.toml",
                    "name": "config",
                    "description": "App configuration",
                    "mimeType": "application/toml"
                },
                {
                    "uri": "db://users/schema",
                    "name": "users_schema"
                }
            ]
        });
        let result: McpResourcesListResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.resources.len(), 2);
        assert_eq!(result.resources[0].uri, "file:///config.toml");
        assert_eq!(result.resources[0].name, "config");
        assert_eq!(
            result.resources[0].mime_type.as_deref(),
            Some("application/toml")
        );
        assert_eq!(result.resources[1].name, "users_schema");
        assert!(result.next_cursor.is_none());
    }

    #[test]
    fn resources_list_with_cursor() {
        let json = json!({
            "resources": [{"uri": "a://1", "name": "one"}],
            "nextCursor": "page2"
        });
        let result: McpResourcesListResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.resources.len(), 1);
        assert_eq!(result.next_cursor.as_deref(), Some("page2"));
    }

    #[test]
    fn resource_content_parsing() {
        let json = json!({
            "uri": "file:///README.md",
            "mimeType": "text/markdown",
            "text": "# Hello World"
        });
        let content: McpResourceContent = serde_json::from_value(json).unwrap();
        assert_eq!(content.uri, "file:///README.md");
        assert_eq!(content.mime_type.as_deref(), Some("text/markdown"));
        assert_eq!(content.text.as_deref(), Some("# Hello World"));
        assert!(content.blob.is_none());
    }

    #[test]
    fn resource_read_result_parsing() {
        let json = json!({
            "contents": [
                {"uri": "file:///a.txt", "text": "content A"},
                {"uri": "file:///b.txt", "text": "content B"}
            ]
        });
        let result: McpResourceReadResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.contents.len(), 2);
        assert_eq!(result.contents[0].text.as_deref(), Some("content A"));
    }

    // --- Prompt types ---

    #[test]
    fn prompt_def_serde_roundtrip() {
        let def = McpPromptDef {
            name: "summarize".into(),
            description: Some("Summarize text".into()),
            arguments: vec![McpPromptArgument {
                name: "text".into(),
                description: Some("Text to summarize".into()),
                required: true,
            }],
        };
        let json = serde_json::to_value(&def).unwrap();
        assert_eq!(json["name"], "summarize");
        let parsed: McpPromptDef = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.arguments.len(), 1);
        assert!(parsed.arguments[0].required);
    }

    #[test]
    fn prompt_def_minimal() {
        let json = json!({"name": "greet"});
        let def: McpPromptDef = serde_json::from_value(json).unwrap();
        assert_eq!(def.name, "greet");
        assert!(def.description.is_none());
        assert!(def.arguments.is_empty());
    }

    #[test]
    fn prompts_list_result_parsing() {
        let json = json!({
            "prompts": [
                {
                    "name": "code_review",
                    "description": "Review code for issues",
                    "arguments": [
                        {"name": "code", "description": "Code to review", "required": true},
                        {"name": "language", "description": "Programming language", "required": false}
                    ]
                }
            ]
        });
        let result: McpPromptsListResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.prompts.len(), 1);
        assert_eq!(result.prompts[0].name, "code_review");
        assert_eq!(result.prompts[0].arguments.len(), 2);
        assert!(result.prompts[0].arguments[0].required);
        assert!(!result.prompts[0].arguments[1].required);
    }

    #[test]
    fn prompt_get_result_parsing() {
        let json = json!({
            "description": "A helpful prompt",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": "Please help me with this code"}
                },
                {
                    "role": "assistant",
                    "content": {"type": "text", "text": "I'd be happy to help!"}
                }
            ]
        });
        let result: McpPromptGetResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(
            result.messages[0].content.text.as_deref(),
            Some("Please help me with this code")
        );
        assert_eq!(result.messages[1].role, "assistant");
    }

    // --- sanitize_tool_name ---

    #[test]
    fn sanitize_tool_name_alphanumeric() {
        assert_eq!(sanitize_tool_name("hello_world"), "hello_world");
        assert_eq!(sanitize_tool_name("test123"), "test123");
    }

    #[test]
    fn sanitize_tool_name_special_chars() {
        assert_eq!(sanitize_tool_name("my-resource"), "my_resource");
        assert_eq!(sanitize_tool_name("path/to/thing"), "path_to_thing");
        assert_eq!(sanitize_tool_name("file.txt"), "file_txt");
        assert_eq!(sanitize_tool_name("a b c"), "a_b_c");
    }

    // --- McpResourceTool ---

    #[test]
    fn resource_tool_definition() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));

        let tool = McpResourceTool {
            transport,
            resource: McpResourceDef {
                uri: "file:///README.md".into(),
                name: "readme".into(),
                description: Some("Project readme".into()),
                mime_type: None,
            },
            tool_name: "mcp_resource_readme".into(),
        };

        let def = tool.definition();
        assert_eq!(def.name, "mcp_resource_readme");
        assert_eq!(def.description, "Project readme");
        assert_eq!(
            def.input_schema,
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn resource_tool_definition_default_description() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));

        let tool = McpResourceTool {
            transport,
            resource: McpResourceDef {
                uri: "db://users".into(),
                name: "users".into(),
                description: None,
                mime_type: None,
            },
            tool_name: "mcp_resource_users".into(),
        };

        let def = tool.definition();
        assert!(def.description.contains("db://users"));
    }

    // --- McpPromptTool ---

    #[test]
    fn prompt_tool_definition_with_args() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));

        let tool = McpPromptTool {
            transport,
            prompt: McpPromptDef {
                name: "review".into(),
                description: Some("Code review".into()),
                arguments: vec![
                    McpPromptArgument {
                        name: "code".into(),
                        description: Some("Code to review".into()),
                        required: true,
                    },
                    McpPromptArgument {
                        name: "language".into(),
                        description: None,
                        required: false,
                    },
                ],
            },
            tool_name: "mcp_prompt_review".into(),
        };

        let def = tool.definition();
        assert_eq!(def.name, "mcp_prompt_review");
        assert_eq!(def.description, "Code review");
        let schema = &def.input_schema;
        assert!(schema["properties"]["code"].is_object());
        assert_eq!(
            schema["properties"]["code"]["description"],
            "Code to review"
        );
        assert_eq!(schema["required"], json!(["code"]));
        // language is not required, shouldn't be in required array
        assert!(
            !schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("language"))
        );
    }

    #[test]
    fn prompt_tool_definition_no_args() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));

        let tool = McpPromptTool {
            transport,
            prompt: McpPromptDef {
                name: "greet".into(),
                description: None,
                arguments: vec![],
            },
            tool_name: "mcp_prompt_greet".into(),
        };

        let def = tool.definition();
        assert_eq!(def.name, "mcp_prompt_greet");
        assert!(def.description.contains("greet"));
        // No required array when no required args
        assert!(def.input_schema.get("required").is_none());
    }

    // --- into_resource_tools / into_prompt_tools ---

    #[test]
    fn into_resource_tools_creates_correct_names() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));

        let client = McpClient {
            transport,
            tools: vec![],
            resources: vec![
                McpResourceDef {
                    uri: "file:///a.txt".into(),
                    name: "readme-file".into(),
                    description: None,
                    mime_type: None,
                },
                McpResourceDef {
                    uri: "db://schema".into(),
                    name: "db schema".into(),
                    description: Some("Database schema".into()),
                    mime_type: None,
                },
            ],
            prompts: vec![],
            capabilities: ServerCapabilities::default(),
            sampling_handler: None,
            roots: Vec::new(),
        };

        let tools = client.into_resource_tools();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].definition().name, "mcp_resource_readme_file");
        assert_eq!(tools[1].definition().name, "mcp_resource_db_schema");
        assert_eq!(tools[1].definition().description, "Database schema");
    }

    #[test]
    fn into_prompt_tools_creates_correct_names() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));

        let client = McpClient {
            transport,
            tools: vec![],
            resources: vec![],
            prompts: vec![McpPromptDef {
                name: "code-review".into(),
                description: Some("Review code".into()),
                arguments: vec![],
            }],
            capabilities: ServerCapabilities::default(),
            sampling_handler: None,
            roots: Vec::new(),
        };

        let tools = client.into_prompt_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition().name, "mcp_prompt_code_review");
    }

    #[test]
    fn into_all_tools_combines_everything() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));

        let client = McpClient {
            transport,
            tools: vec![McpToolDef {
                name: "read_file".into(),
                description: Some("Read a file".into()),
                input_schema: Some(json!({"type": "object"})),
            }],
            resources: vec![McpResourceDef {
                uri: "file:///a.txt".into(),
                name: "readme".into(),
                description: None,
                mime_type: None,
            }],
            prompts: vec![McpPromptDef {
                name: "greet".into(),
                description: None,
                arguments: vec![],
            }],
            capabilities: ServerCapabilities::default(),
            sampling_handler: None,
            roots: Vec::new(),
        };

        let all = client.into_all_tools();
        assert_eq!(all.len(), 3);
        let names: Vec<String> = all.iter().map(|t| t.definition().name).collect();
        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"mcp_resource_readme".to_string()));
        assert!(names.contains(&"mcp_prompt_greet".to_string()));
    }

    #[test]
    fn supports_resource_subscribe_false_by_default() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));
        let client = McpClient {
            transport,
            tools: vec![],
            resources: vec![],
            prompts: vec![],
            capabilities: ServerCapabilities::default(),
            sampling_handler: None,
            roots: Vec::new(),
        };
        assert!(!client.supports_resource_subscribe());
    }

    #[test]
    fn supports_resource_subscribe_when_advertised() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));
        let client = McpClient {
            transport,
            tools: vec![],
            resources: vec![],
            prompts: vec![],
            capabilities: ServerCapabilities {
                resources: Some(ResourcesCapability {
                    subscribe: true,
                    list_changed: false,
                }),
                ..Default::default()
            },
            sampling_handler: None,
            roots: Vec::new(),
        };
        assert!(client.supports_resource_subscribe());
    }

    // --- Sampling types ---

    #[test]
    fn sampling_request_parsing() {
        let json = json!({
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": "What is 2+2?"}
                }
            ],
            "modelPreferences": {
                "hints": [{"name": "claude-sonnet-4-6-20250610"}]
            },
            "systemPrompt": "You are a math helper",
            "maxTokens": 100
        });
        let req: SamplingRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(
            req.messages[0].content.text.as_deref(),
            Some("What is 2+2?")
        );
        assert_eq!(req.system_prompt.as_deref(), Some("You are a math helper"));
        assert_eq!(req.max_tokens, Some(100));
        let hints = &req.model_preferences.unwrap().hints;
        assert_eq!(hints[0].name.as_deref(), Some("claude-sonnet-4-6-20250610"));
    }

    #[test]
    fn sampling_request_minimal() {
        let json = json!({
            "messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}]
        });
        let req: SamplingRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert!(req.model_preferences.is_none());
        assert!(req.system_prompt.is_none());
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn sampling_response_serialization() {
        let resp = SamplingResponse {
            role: "assistant".into(),
            content: SamplingContent {
                content_type: "text".into(),
                text: Some("4".into()),
            },
            model: "claude-sonnet-4-6-20250610".into(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"]["type"], "text");
        assert_eq!(json["content"]["text"], "4");
        assert_eq!(json["model"], "claude-sonnet-4-6-20250610");
    }

    #[test]
    fn sampling_message_serde_roundtrip() {
        let msg = SamplingMessage {
            role: "user".into(),
            content: SamplingContent {
                content_type: "text".into(),
                text: Some("hello".into()),
            },
        };
        let json = serde_json::to_value(&msg).unwrap();
        let parsed: SamplingMessage = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.role, "user");
        assert_eq!(parsed.content.text.as_deref(), Some("hello"));
    }

    #[test]
    fn with_sampling_sets_handler() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));
        let client = McpClient {
            transport,
            tools: vec![],
            resources: vec![],
            prompts: vec![],
            capabilities: ServerCapabilities::default(),
            sampling_handler: None,
            roots: Vec::new(),
        };
        assert!(client.sampling_handler.is_none());

        let handler: SamplingHandler =
            Arc::new(|_req| Box::pin(async move { Ok(("response".into(), "model".into())) }));
        let client = client.with_sampling(handler);
        assert!(client.sampling_handler.is_some());
    }

    // --- Logging ---

    #[test]
    fn handle_log_notification_info() {
        // Should not panic; just forwards to tracing
        let value = json!({
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {"level": "info", "logger": "test-server", "data": "Server started"}
        });
        handle_log_notification(&value);
    }

    #[test]
    fn handle_log_notification_error() {
        let value = json!({
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {"level": "error", "data": "Something went wrong"}
        });
        handle_log_notification(&value);
    }

    #[test]
    fn handle_log_notification_missing_params() {
        let value = json!({"jsonrpc": "2.0", "method": "notifications/message"});
        handle_log_notification(&value); // should not panic
    }

    #[test]
    fn find_rpc_response_skips_log_notifications() {
        let events = vec![
            r#"{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info","data":"log"}}"#.to_string(),
            r#"{"jsonrpc":"2.0","result":{"ok":true},"id":1}"#.to_string(),
        ];
        let result = find_rpc_response(&events, 1).unwrap();
        assert!(result.contains("\"id\":1"));
    }

    // --- Roots ---

    #[test]
    fn mcp_root_serde_roundtrip() {
        let root = McpRoot {
            uri: "file:///workspace/project".into(),
            name: Some("project".into()),
        };
        let json = serde_json::to_value(&root).unwrap();
        assert_eq!(json["uri"], "file:///workspace/project");
        assert_eq!(json["name"], "project");
        let parsed: McpRoot = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.uri, "file:///workspace/project");
    }

    #[test]
    fn mcp_root_minimal() {
        let json = json!({"uri": "file:///tmp"});
        let root: McpRoot = serde_json::from_value(json).unwrap();
        assert_eq!(root.uri, "file:///tmp");
        assert!(root.name.is_none());
    }

    #[test]
    fn mcp_root_name_omitted_when_none() {
        let root = McpRoot {
            uri: "file:///x".into(),
            name: None,
        };
        let json = serde_json::to_string(&root).unwrap();
        assert!(!json.contains("name"));
    }

    #[test]
    fn with_roots_sets_roots() {
        let transport = Arc::new(Transport::Http(HttpTransport {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            session_id: RwLock::new(None),
            next_id: AtomicU64::new(0),
            auth_header: None,
        }));
        let client = McpClient {
            transport,
            tools: vec![],
            resources: vec![],
            prompts: vec![],
            capabilities: ServerCapabilities::default(),
            sampling_handler: None,
            roots: Vec::new(),
        };
        assert!(client.roots().is_empty());

        let client = client.with_roots(vec![McpRoot {
            uri: "file:///workspace".into(),
            name: Some("workspace".into()),
        }]);
        assert_eq!(client.roots().len(), 1);
        assert_eq!(client.roots()[0].uri, "file:///workspace");
    }

    #[tokio::test]
    async fn read_stdio_response_forwards_log_notifications() {
        let (mut tx, rx) = tokio::io::duplex(4096);
        let mut reader = tokio::io::BufReader::new(rx);

        tokio::spawn(async move {
            // Server sends a log notification, then the actual response.
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\",\"data\":\"test log\"}}\n")
                .await
                .unwrap();
            tx.write_all(b"{\"jsonrpc\":\"2.0\",\"result\":{\"ok\":true},\"id\":1}\n")
                .await
                .unwrap();
        });

        let response = read_stdio_response(&mut reader, 1).await.unwrap();
        assert!(response.contains("\"id\":1"));
        assert!(response.contains("\"ok\":true"));
    }
}
