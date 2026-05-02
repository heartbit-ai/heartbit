//! Runtime types for dynamic agent execution via `POST /v1/execute`.
//!
//! These types define the API contract between heartbit-cloud (or any external caller)
//! and the daemon's runtime execution endpoint. Secret fields (`api_key`, `auth_header`)
//! are serialized normally for wire transport but redacted in `Debug` output.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

use crate::Error;
use crate::agent::AgentOutput;
use crate::agent::events::AgentEvent;
use crate::llm::types::TokenUsage;
use crate::memory::Memory;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Role for a message in the session history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMessageRole {
    User,
    Assistant,
}

/// A single message in session history, sent alongside the current prompt.
///
/// When `messages` is non-empty in a `RuntimeRequest`, the runtime converts
/// them to LLM `Message` types and pre-seeds the conversation history.
/// The `prompt` field then becomes the current (latest) user message only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMessage {
    pub role: RuntimeMessageRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

/// Top-level request for dynamic agent execution.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeRequest {
    pub task_id: uuid::Uuid,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    /// Tenant identifier for audit logging and isolation.
    #[serde(default)]
    pub tenant_id: Option<uuid::Uuid>,
    /// User identifier for audit logging.
    #[serde(default)]
    pub user_id: Option<String>,
    pub agent: RuntimeAgentConfig,
    pub provider: RuntimeProviderConfig,
    #[serde(default)]
    pub mcp_servers: Vec<RuntimeMcpServer>,
    #[serde(default)]
    pub builtin_tools: Vec<String>,
    #[serde(default)]
    pub guardrails: Option<RuntimeGuardrailConfig>,
    #[serde(default)]
    pub memory: Option<RuntimeMemoryConfig>,
    /// Session history messages. When non-empty, these are converted to LLM
    /// messages and pre-seed the conversation. `prompt` becomes the current
    /// message only. Backward compatible: defaults to empty vec.
    #[serde(default)]
    pub messages: Vec<RuntimeMessage>,
    /// Session identifier for correlating messages across requests.
    #[serde(default)]
    pub session_id: Option<uuid::Uuid>,
}

/// Agent configuration embedded in the runtime request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeAgentConfig {
    pub name: String,
    pub system_prompt: String,
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub advanced: RuntimeAdvancedConfig,
}

fn default_max_turns() -> usize {
    50
}

fn default_max_tokens() -> u32 {
    4096
}

/// Advanced agent settings (all optional with sensible defaults).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RuntimeAdvancedConfig {
    pub reasoning_effort: Option<String>,
    pub tool_profile: Option<String>,
    pub max_total_tokens: Option<u64>,
    pub run_timeout_seconds: Option<u64>,
    pub tool_timeout_seconds: Option<u64>,
    pub max_identical_tool_calls: Option<u32>,
    pub context_strategy: Option<String>,
    pub summarize_threshold: Option<u32>,
    pub enable_reflection: Option<bool>,
    pub session_prune: Option<bool>,
    pub recursive_summarization: Option<bool>,
    pub consolidate_on_exit: Option<bool>,
    pub max_tools_per_turn: Option<usize>,
    pub max_tool_output_bytes: Option<usize>,
    pub response_cache_size: Option<usize>,
}

/// Supported LLM provider types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProviderType {
    Anthropic,
    Openrouter,
}

/// LLM provider configuration.
///
/// `Debug` is manually implemented to redact `api_key`.
#[derive(Clone, Deserialize, Serialize)]
pub struct RuntimeProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: RuntimeProviderType,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub prompt_caching: bool,
}

impl std::fmt::Debug for RuntimeProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeProviderConfig")
            .field("provider_type", &self.provider_type)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("prompt_caching", &self.prompt_caching)
            .finish()
    }
}

/// MCP server to connect for tool access.
///
/// `Debug` is manually implemented to redact `auth_header`.
#[derive(Clone, Deserialize, Serialize)]
pub struct RuntimeMcpServer {
    pub url: String,
    #[serde(default)]
    pub auth_header: Option<String>,
}

impl std::fmt::Debug for RuntimeMcpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeMcpServer")
            .field("url", &self.url)
            .field(
                "auth_header",
                &self.auth_header.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Guardrail configuration for runtime requests.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RuntimeGuardrailConfig {
    #[serde(default)]
    pub injection: bool,
    #[serde(default)]
    pub pii: bool,
    #[serde(default = "default_injection_threshold")]
    pub injection_threshold: f32,
}

fn default_injection_threshold() -> f32 {
    0.5
}

/// Memory configuration for runtime requests.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RuntimeMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub reflection_threshold: Option<u32>,
    #[serde(default)]
    pub consolidate_on_exit: bool,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Response for non-streaming execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeResponse {
    pub task_id: uuid::Uuid,
    pub result: String,
    pub usage: TokenUsage,
    pub tool_calls_made: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_cost_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
}

/// SSE event types for streaming execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeSseEvent {
    Delta { content: String },
    Event { event: AgentEvent },
    Done(RuntimeResponse),
    Error { message: String },
}

// ---------------------------------------------------------------------------
// Runtime config (daemon section)
// ---------------------------------------------------------------------------

/// Configuration for the `/v1/execute` runtime endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeConfig {
    /// Maximum concurrent runtime executions. Default: 50.
    #[serde(default = "default_runtime_max_concurrent")]
    pub max_concurrent: usize,
    /// Maximum request body size in bytes. Default: 1 MB.
    #[serde(default = "default_runtime_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Allowed provider types. Empty = all allowed.
    #[serde(default)]
    pub allowed_providers: Vec<RuntimeProviderType>,
    /// Allowed MCP server URL prefixes for SSRF prevention. Empty = all allowed.
    #[serde(default)]
    pub allowed_mcp_prefixes: Vec<String>,
    /// Memory store configuration for tenant-scoped agent memory.
    #[serde(default)]
    pub memory_store: Option<RuntimeMemoryStoreConfig>,
    /// MCP connection cache TTL in seconds. Default: 300 (5 min). Set to 0 to disable.
    #[serde(default = "default_mcp_cache_ttl")]
    pub mcp_cache_ttl_seconds: u64,
}

fn default_runtime_max_concurrent() -> usize {
    50
}

fn default_runtime_max_body_bytes() -> usize {
    1_048_576 // 1 MB
}

fn default_mcp_cache_ttl() -> u64 {
    300 // 5 minutes
}

/// Memory store backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStoreType {
    #[default]
    InMemory,
    Postgres,
}

/// Memory store configuration for the runtime endpoint.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RuntimeMemoryStoreConfig {
    #[serde(default)]
    pub store_type: MemoryStoreType,
    /// PostgreSQL URL (required when store_type = Postgres).
    #[serde(default)]
    pub database_url: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_concurrent: default_runtime_max_concurrent(),
            max_body_bytes: default_runtime_max_body_bytes(),
            allowed_providers: Vec::new(),
            allowed_mcp_prefixes: Vec::new(),
            memory_store: None,
            mcp_cache_ttl_seconds: default_mcp_cache_ttl(),
        }
    }
}

// ---------------------------------------------------------------------------
// MCP Connection Cache
// ---------------------------------------------------------------------------

/// Cache for MCP server connections to avoid reconnecting on every request.
///
/// Key: `(url, auth_header_hash)` where auth_header_hash is 0 for no auth.
/// Value: `(tools, last_used)` — tools discovered from the server + timestamp.
///
/// Uses `std::sync::RwLock` (not tokio) because we never hold it across `.await`.
type McpCacheEntry = (Vec<Arc<dyn crate::tool::Tool>>, Instant);

pub struct McpConnectionCache {
    entries: std::sync::RwLock<HashMap<(String, u64), McpCacheEntry>>,
    ttl: std::time::Duration,
}

impl McpConnectionCache {
    pub fn new(ttl: std::time::Duration) -> Self {
        Self {
            entries: std::sync::RwLock::new(HashMap::new()),
            ttl,
        }
    }

    /// Look up cached tools for a server. Returns `None` if not cached or expired.
    /// Expired entries are lazily removed to prevent unbounded growth.
    fn get(&self, url: &str, auth_hash: u64) -> Option<Vec<Arc<dyn crate::tool::Tool>>> {
        let key = (url.to_string(), auth_hash);
        // Fast path: read lock only
        {
            let entries = self.entries.read().ok()?;
            match entries.get(&key) {
                Some((tools, last_used)) if last_used.elapsed() <= self.ttl => {
                    return Some(tools.clone());
                }
                Some(_) => {} // expired — fall through to remove
                None => return None,
            }
        }
        // Slow path: upgrade to write lock and remove expired entry
        if let Ok(mut entries) = self.entries.write() {
            entries.remove(&key);
        }
        None
    }

    /// Insert or update a cache entry.
    fn insert(&self, url: &str, auth_hash: u64, tools: Vec<Arc<dyn crate::tool::Tool>>) {
        if let Ok(mut entries) = self.entries.write() {
            entries.insert((url.to_string(), auth_hash), (tools, Instant::now()));
        }
    }

    /// Evict expired entries. Called lazily during lookups.
    pub fn evict_expired(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.retain(|_, (_, last_used)| last_used.elapsed() <= self.ttl);
        }
    }
}

impl std::fmt::Debug for McpConnectionCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.entries.read().map(|e| e.len()).unwrap_or(0);
        f.debug_struct("McpConnectionCache")
            .field("entries", &count)
            .field("ttl", &self.ttl)
            .finish()
    }
}

/// Hash an optional auth header for use as cache key.
fn hash_auth(auth: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    auth.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Agent builder from RuntimeRequest
// ---------------------------------------------------------------------------

/// Cloud-safe builtin tool names that don't expose filesystem access.
const CLOUD_SAFE_BUILTINS: &[&str] = &["websearch", "webfetch", "todo"];

/// Build an `AgentRunner` from a `RuntimeRequest`.
///
/// This consolidates dynamic agent construction into the lib crate:
/// 1. Instantiate provider from `RuntimeProviderConfig`
/// 2. Connect MCP servers in parallel (10s timeout, fail-open)
/// 3. Filter builtin tools by allowed list (cloud-safe only)
/// 4. Apply advanced config (reasoning_effort, tool_profile, etc.)
/// 5. Apply guardrails (PII + injection classifier)
/// 6. Return configured `AgentRunner`
pub async fn build_agent_from_request(
    req: &RuntimeRequest,
    on_text: Option<Arc<crate::llm::OnText>>,
    on_event: Option<Arc<crate::agent::events::OnEvent>>,
    runtime_config: Option<&RuntimeConfig>,
    memory_store: Option<Arc<dyn Memory>>,
    mcp_cache: Option<&McpConnectionCache>,
) -> Result<crate::agent::AgentRunner<crate::llm::BoxedProvider>, Error> {
    // 1. Validate provider type
    if let Some(rc) = runtime_config
        && !rc.allowed_providers.is_empty()
        && !rc.allowed_providers.contains(&req.provider.provider_type)
    {
        return Err(Error::Config(format!(
            "provider type '{:?}' not in allowed list",
            req.provider.provider_type
        )));
    }

    let provider = build_provider(&req.provider)?;
    let provider = Arc::new(provider);

    // 2. Connect MCP servers in parallel (10s timeout, fail-open, with optional cache)
    let mcp_tools = connect_mcp_servers(&req.mcp_servers, runtime_config, mcp_cache).await;

    // 3. Filter builtin tools
    let builtins = filtered_builtins(&req.builtin_tools);

    // Combine tools
    let mut tools: Vec<Arc<dyn crate::tool::Tool>> = builtins;
    tools.extend(mcp_tools);

    // 3b. Wire tenant-scoped memory tools when enabled
    let namespaced_memory = if let (Some(mc), Some(store)) = (&req.memory, memory_store.as_ref()) {
        if mc.enabled {
            let namespace = match &req.tenant_id {
                Some(tid) => format!("tenant:{tid}:{}", req.agent.name),
                None => req.agent.name.clone(),
            };
            let ns_mem = Arc::new(crate::memory::namespaced::NamespacedMemory::new(
                store.clone(),
                &namespace,
            ));
            let mem_tools = crate::memory::tools::memory_tools_with_reflection(
                ns_mem.clone() as Arc<dyn Memory>,
                &req.agent.name,
                mc.reflection_threshold,
            );
            tools.extend(mem_tools);
            Some(ns_mem as Arc<dyn Memory>)
        } else {
            None
        }
    } else {
        None
    };

    // 4. Build agent runner
    let mut rb = crate::agent::AgentRunner::builder(provider)
        .name(&req.agent.name)
        .system_prompt(&req.agent.system_prompt)
        .tools(tools)
        .max_turns(req.agent.max_turns)
        .max_tokens(req.agent.max_tokens);

    // Pre-seed conversation history from session messages
    if !req.messages.is_empty() {
        let initial: Vec<crate::llm::types::Message> = req
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    RuntimeMessageRole::User => crate::llm::types::Role::User,
                    RuntimeMessageRole::Assistant => crate::llm::types::Role::Assistant,
                };
                crate::llm::types::Message {
                    role,
                    content: vec![crate::llm::types::ContentBlock::Text {
                        text: m.content.clone(),
                    }],
                }
            })
            .collect();
        rb = rb.initial_messages(initial);
    }

    if let Some(on_text) = on_text {
        rb = rb.on_text(on_text);
    }
    if let Some(on_event) = on_event {
        rb = rb.on_event(on_event);
    }

    // Wire memory store if enabled
    if let Some(mem) = namespaced_memory {
        rb = rb.memory(mem);
    }

    // Wire tenant/user identity for audit logging and system prompt attribution
    match (&req.user_id, &req.tenant_id) {
        (Some(uid), Some(tid)) => {
            rb = rb.audit_user_context(uid.as_str(), tid.to_string());
        }
        (Some(uid), None) => {
            rb = rb.audit_user_context(uid.as_str(), "unknown");
        }
        (None, Some(tid)) => {
            rb = rb.audit_user_context("unknown", tid.to_string());
        }
        (None, None) => {}
    }

    // Apply advanced config
    let adv = &req.agent.advanced;
    if let Some(ref effort) = adv.reasoning_effort {
        rb = rb.reasoning_effort(crate::config::parse_reasoning_effort(effort)?);
    }
    if let Some(ref profile) = adv.tool_profile {
        rb = rb.tool_profile(crate::config::parse_tool_profile(profile)?);
    }
    if let Some(budget) = adv.max_total_tokens {
        rb = rb.max_total_tokens(budget);
    }
    if let Some(secs) = adv.run_timeout_seconds {
        rb = rb.run_timeout(std::time::Duration::from_secs(secs));
    }
    if let Some(secs) = adv.tool_timeout_seconds {
        rb = rb.tool_timeout(std::time::Duration::from_secs(secs));
    }
    if let Some(m) = adv.max_identical_tool_calls {
        rb = rb.max_identical_tool_calls(m);
    }
    if let Some(true) = adv.enable_reflection {
        rb = rb.enable_reflection(true);
    }
    if let Some(true) = adv.session_prune {
        rb = rb.session_prune_config(crate::agent::pruner::SessionPruneConfig::default());
    }
    if let Some(true) = adv.recursive_summarization {
        rb = rb.enable_recursive_summarization(true);
    }
    if let Some(true) = adv.consolidate_on_exit {
        rb = rb.consolidate_on_exit(true);
    }
    if let Some(max) = adv.max_tools_per_turn {
        rb = rb.max_tools_per_turn(max);
    }
    if let Some(max) = adv.max_tool_output_bytes {
        rb = rb.max_tool_output_bytes(max);
    }
    if let Some(ref strategy) = adv.context_strategy {
        match strategy.as_str() {
            "sliding_window" => {
                let threshold = adv.summarize_threshold.unwrap_or(100_000);
                rb = rb.context_strategy(crate::agent::context::ContextStrategy::SlidingWindow {
                    max_tokens: threshold,
                });
            }
            "summarize" => {
                if let Some(t) = adv.summarize_threshold {
                    rb = rb.summarize_threshold(t);
                }
            }
            "unlimited" => {} // explicit no-op
            other => {
                tracing::warn!(strategy = %other, "unknown context_strategy, ignoring");
            }
        }
    } else if let Some(t) = adv.summarize_threshold {
        rb = rb.summarize_threshold(t);
    }

    // 5. Apply guardrails
    if let Some(ref gc) = req.guardrails {
        let mut guardrails: Vec<Arc<dyn crate::agent::guardrail::Guardrail>> = Vec::new();
        if gc.injection {
            guardrails.push(Arc::new(
                crate::agent::guardrails::InjectionClassifierGuardrail::new(
                    gc.injection_threshold,
                    crate::agent::guardrails::GuardrailMode::Deny,
                ),
            ));
        }
        if gc.pii {
            guardrails.push(Arc::new(
                crate::agent::guardrails::PiiGuardrail::all_builtin(
                    crate::agent::guardrails::PiiAction::Redact,
                ),
            ));
        }
        if !guardrails.is_empty() {
            rb = rb.guardrails(guardrails);
        }
    }

    rb.build()
}

/// Build a `BoxedProvider` from runtime provider config, wrapped with retry logic.
fn build_provider(config: &RuntimeProviderConfig) -> Result<crate::llm::BoxedProvider, Error> {
    match config.provider_type {
        RuntimeProviderType::Anthropic => {
            let base = if config.prompt_caching {
                crate::llm::anthropic::AnthropicProvider::with_prompt_caching(
                    &config.api_key,
                    &config.model,
                )
            } else {
                crate::llm::anthropic::AnthropicProvider::new(&config.api_key, &config.model)
            };
            let retrying = crate::llm::retry::RetryingProvider::with_defaults(base);
            Ok(crate::llm::BoxedProvider::new(retrying))
        }
        RuntimeProviderType::Openrouter => {
            let base =
                crate::llm::openrouter::OpenRouterProvider::new(&config.api_key, &config.model);
            let retrying = crate::llm::retry::RetryingProvider::with_defaults(base);
            Ok(crate::llm::BoxedProvider::new(retrying))
        }
    }
}

/// Connect to MCP servers in parallel with 10s timeout per server, fail-open.
///
/// When `cache` is provided, already-connected servers are served from cache.
/// New connections are cached for future requests.
async fn connect_mcp_servers(
    servers: &[RuntimeMcpServer],
    runtime_config: Option<&RuntimeConfig>,
    cache: Option<&McpConnectionCache>,
) -> Vec<Arc<dyn crate::tool::Tool>> {
    if servers.is_empty() {
        return Vec::new();
    }

    let allowed_prefixes: Option<&[String]> = runtime_config
        .map(|rc| rc.allowed_mcp_prefixes.as_slice())
        .filter(|s| !s.is_empty());

    let mut all_tools = Vec::new();
    let mut uncached_servers = Vec::new();

    for server in servers {
        // SSRF check: validate URL against allowed prefixes.
        if let Some(prefixes) = allowed_prefixes
            && !prefixes.iter().any(|p| {
                server.url.starts_with(p)
                    && (server.url.len() == p.len()
                        || p.ends_with('/')
                        || server.url.as_bytes().get(p.len()) == Some(&b'/'))
            })
        {
            tracing::warn!(
                url = %server.url,
                "MCP server URL not in allowed prefixes, skipping"
            );
            continue;
        }

        // Check cache first
        let auth_h = hash_auth(server.auth_header.as_deref());
        if let Some(cache) = cache
            && let Some(tools) = cache.get(&server.url, auth_h)
        {
            tracing::debug!(url = %server.url, tools = tools.len(), "MCP cache hit");
            all_tools.extend(tools);
            continue;
        }
        uncached_servers.push((server.url.clone(), server.auth_header.clone(), auth_h));
    }

    if uncached_servers.is_empty() {
        return all_tools;
    }

    // Connect uncached servers in parallel
    let mut set = JoinSet::new();
    for (url, auth, auth_h) in uncached_servers {
        set.spawn(async move {
            let connect_fut = async {
                if let Some(ref header) = auth {
                    crate::tool::mcp::McpClient::connect_with_auth(&url, header.as_str()).await
                } else {
                    crate::tool::mcp::McpClient::connect(&url).await
                }
            };
            let connect_result =
                tokio::time::timeout(std::time::Duration::from_secs(10), connect_fut).await;

            match connect_result {
                Ok(Ok(client)) => {
                    let tools: Vec<Arc<dyn crate::tool::Tool>> = client.into_tools();
                    tracing::info!(url = %url, tool_count = tools.len(), "connected MCP server");
                    (url, auth_h, tools)
                }
                Ok(Err(e)) => {
                    tracing::warn!(url = %url, error = %e, "failed to connect MCP server");
                    (url, auth_h, Vec::new())
                }
                Err(_) => {
                    tracing::warn!(url = %url, "MCP server connection timed out (10s)");
                    (url, auth_h, Vec::new())
                }
            }
        });
    }

    while let Some(result) = set.join_next().await {
        if let Ok((url, auth_h, tools)) = result {
            if !tools.is_empty() {
                // Cache the successful connection
                if let Some(cache) = cache {
                    cache.insert(&url, auth_h, tools.clone());
                }
            }
            all_tools.extend(tools);
        }
    }
    all_tools
}

/// Filter builtin tools to only include cloud-safe tools.
fn filtered_builtins(requested: &[String]) -> Vec<Arc<dyn crate::tool::Tool>> {
    if requested.is_empty() {
        return Vec::new();
    }

    // Only allow cloud-safe builtins
    let allowed: Vec<&str> = requested
        .iter()
        .filter_map(|name| {
            let lower = name.to_lowercase();
            CLOUD_SAFE_BUILTINS.iter().find(|&&b| b == lower).copied()
        })
        .collect();

    if allowed.is_empty() {
        return Vec::new();
    }

    let btc = crate::tool::builtins::BuiltinToolsConfig::default();
    crate::tool::builtins::builtin_tools(btc)
        .into_iter()
        .filter(|tool| {
            let name = tool.definition().name.to_lowercase();
            // "todo" matches "todoread"/"todowrite"; "websearch" matches exactly.
            allowed.iter().any(|&a| name.starts_with(a))
        })
        .collect()
}

/// Build a `RuntimeResponse` from an `AgentOutput`.
pub fn runtime_response_from_output(
    task_id: uuid::Uuid,
    output: &AgentOutput,
    model: Option<String>,
) -> RuntimeResponse {
    let cost = crate::llm::pricing::estimate_cost(
        model.as_deref().unwrap_or("unknown"),
        &output.tokens_used,
    );

    RuntimeResponse {
        task_id,
        result: output.result.clone(),
        usage: output.tokens_used,
        tool_calls_made: output.tool_calls_made as u32,
        estimated_cost_usd: cost,
        model_name: model,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_request_serde_roundtrip() {
        let req = RuntimeRequest {
            task_id: uuid::Uuid::new_v4(),
            prompt: "Hello".into(),
            stream: true,
            tenant_id: None,
            user_id: None,
            agent: RuntimeAgentConfig {
                name: "test-agent".into(),
                system_prompt: "You are a test agent.".into(),
                max_turns: 10,
                max_tokens: 2048,
                advanced: RuntimeAdvancedConfig::default(),
            },
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Anthropic,
                api_key: "sk-secret".into(),
                model: "claude-sonnet-4-20250514".into(),
                prompt_caching: false,
            },
            mcp_servers: vec![RuntimeMcpServer {
                url: "https://mcp.example.com/sse".into(),
                auth_header: Some("Bearer tok_123".into()),
            }],
            builtin_tools: vec!["websearch".into()],
            guardrails: Some(RuntimeGuardrailConfig {
                injection: true,
                pii: false,
                injection_threshold: 0.85,
            }),
            memory: None,
            messages: vec![],
            session_id: None,
        };

        // Full roundtrip: serialize includes api_key for wire transport
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            json.contains("sk-secret"),
            "api_key must be serialized for wire transport"
        );
        assert!(
            json.contains("tok_123"),
            "auth_header must be serialized for wire transport"
        );

        let deserialized: RuntimeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.task_id, req.task_id);
        assert_eq!(deserialized.provider.api_key, "sk-secret");
        assert_eq!(
            deserialized.mcp_servers[0].auth_header.as_deref(),
            Some("Bearer tok_123")
        );

        // Debug output must redact secrets
        let debug = format!("{:?}", req);
        assert!(
            !debug.contains("sk-secret"),
            "api_key leaked in Debug output"
        );
        assert!(
            !debug.contains("tok_123"),
            "auth_header leaked in Debug output"
        );
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn runtime_response_serde_roundtrip() {
        let resp = RuntimeResponse {
            task_id: uuid::Uuid::new_v4(),
            result: "Hello world".into(),
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                reasoning_tokens: 0,
            },
            tool_calls_made: 2,
            estimated_cost_usd: Some(0.003),
            model_name: Some("claude-sonnet-4-20250514".into()),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: RuntimeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.task_id, resp.task_id);
        assert_eq!(deserialized.result, resp.result);
        assert_eq!(deserialized.tool_calls_made, 2);
    }

    #[test]
    fn runtime_sse_event_variants() {
        let delta = RuntimeSseEvent::Delta {
            content: "hello".into(),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains(r#""type":"delta"#));

        let error = RuntimeSseEvent::Error {
            message: "boom".into(),
        };
        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains(r#""type":"error"#));
    }

    #[test]
    fn default_values() {
        let json = r#"{
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "test",
            "agent": {
                "name": "a",
                "system_prompt": "sp"
            },
            "provider": {
                "type": "anthropic",
                "api_key": "sk-test",
                "model": "claude-sonnet-4-20250514"
            }
        }"#;
        let req: RuntimeRequest = serde_json::from_str(json).unwrap();
        assert!(!req.stream);
        assert_eq!(req.agent.max_turns, 50);
        assert_eq!(req.agent.max_tokens, 4096);
        assert!(req.mcp_servers.is_empty());
        assert!(req.builtin_tools.is_empty());
        assert!(req.guardrails.is_none());
    }

    #[test]
    fn runtime_config_defaults() {
        let config = RuntimeConfig::default();
        assert_eq!(config.max_concurrent, 50);
        assert_eq!(config.max_body_bytes, 1_048_576);
        assert!(config.allowed_providers.is_empty());
        assert!(config.allowed_mcp_prefixes.is_empty());
    }

    #[test]
    fn provider_type_rejects_unknown_at_serde_level() {
        let json = r#"{"type": "unsupported", "api_key": "key", "model": "m"}"#;
        let result: Result<RuntimeProviderConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn filtered_builtins_cloud_safe_only() {
        // "bash" and "read" are not cloud-safe and should be filtered out
        let tools = filtered_builtins(&[
            "websearch".into(),
            "bash".into(),
            "read".into(),
            "webfetch".into(),
        ]);
        for tool in &tools {
            let name = tool.definition().name.to_lowercase();
            assert!(
                !name.contains("bash") && !name.contains("read"),
                "non-cloud-safe tool leaked through: {name}"
            );
        }
    }

    #[test]
    fn filtered_builtins_empty_request() {
        let tools = filtered_builtins(&[]);
        assert!(tools.is_empty());
    }

    #[test]
    fn guardrail_config_defaults() {
        // Test serde defaults (used when deserializing from JSON with missing fields)
        let gc: RuntimeGuardrailConfig = serde_json::from_str("{}").unwrap();
        assert!(!gc.injection);
        assert!(!gc.pii);
        assert!((gc.injection_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn mcp_server_debug_redacts_auth() {
        let server = RuntimeMcpServer {
            url: "https://example.com".into(),
            auth_header: Some("Bearer secret".into()),
        };
        // Serialization includes auth for wire transport
        let json = serde_json::to_string(&server).unwrap();
        assert!(
            json.contains("secret"),
            "auth_header must be serialized for wire transport"
        );

        // Debug must redact
        let debug = format!("{:?}", server);
        assert!(!debug.contains("secret"), "auth_header leaked in Debug");
        assert!(debug.contains("<redacted>"));
    }

    #[tokio::test]
    async fn build_agent_rejects_disallowed_provider() {
        let req = RuntimeRequest {
            task_id: uuid::Uuid::new_v4(),
            prompt: "test".into(),
            stream: false,
            tenant_id: None,
            user_id: None,
            agent: RuntimeAgentConfig {
                name: "test".into(),
                system_prompt: "sp".into(),
                max_turns: 5,
                max_tokens: 100,
                advanced: RuntimeAdvancedConfig::default(),
            },
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Openrouter,
                api_key: "key".into(),
                model: "model".into(),
                prompt_caching: false,
            },
            mcp_servers: vec![],
            builtin_tools: vec![],
            guardrails: None,
            memory: None,
            messages: vec![],
            session_id: None,
        };

        let rc = RuntimeConfig {
            allowed_providers: vec![RuntimeProviderType::Anthropic],
            ..Default::default()
        };

        let result = build_agent_from_request(&req, None, None, Some(&rc), None, None).await;
        assert!(result.is_err());
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("not in allowed list")
        );
    }

    #[test]
    fn runtime_response_from_output_works() {
        let output = AgentOutput {
            result: "done".into(),
            tokens_used: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                reasoning_tokens: 0,
            },
            tool_calls_made: 3,
            structured: None,
            estimated_cost_usd: None,
            model_name: None,
        };
        let resp = runtime_response_from_output(
            uuid::Uuid::new_v4(),
            &output,
            Some("claude-sonnet-4-20250514".into()),
        );
        assert_eq!(resp.result, "done");
        assert_eq!(resp.tool_calls_made, 3);
        assert_eq!(resp.usage.input_tokens, 100);
    }

    #[tokio::test]
    async fn ssrf_prefix_rejects_subdomain_bypass() {
        // "https://good.com" must NOT match "https://good.com.evil.com"
        let servers = vec![RuntimeMcpServer {
            url: "https://good.com.evil.com/path".into(),
            auth_header: None,
        }];
        let rc = RuntimeConfig {
            allowed_mcp_prefixes: vec!["https://good.com".into()],
            ..Default::default()
        };
        let tools = connect_mcp_servers(&servers, Some(&rc), None).await;
        assert!(tools.is_empty(), "subdomain bypass should be blocked");
    }

    #[test]
    fn ssrf_prefix_boundary_logic() {
        // Test the SSRF prefix matching logic directly without network.
        // The logic: url.starts_with(prefix) AND
        //   (exact length match OR prefix ends with '/' OR url has '/' at prefix boundary)
        let check = |url: &str, prefix: &str| -> bool {
            url.starts_with(prefix)
                && (url.len() == prefix.len()
                    || prefix.ends_with('/')
                    || url.as_bytes().get(prefix.len()) == Some(&b'/'))
        };

        // Valid: exact match
        assert!(check("https://good.com", "https://good.com"));
        // Valid: path under prefix
        assert!(check("https://good.com/mcp", "https://good.com"));
        // Valid: prefix ends with slash
        assert!(check("https://good.com/mcp", "https://good.com/"));
        // BLOCKED: subdomain spoof
        assert!(!check("https://good.com.evil.com", "https://good.com"));
        // BLOCKED: port-based bypass
        assert!(!check("https://good.com:8080/x", "https://good.com"));
        // BLOCKED: query-based bypass
        assert!(!check("https://good.com?evil=1", "https://good.com"));
    }

    #[test]
    fn runtime_memory_config_serde() {
        let mc = RuntimeMemoryConfig {
            enabled: true,
            reflection_threshold: Some(50),
            consolidate_on_exit: true,
        };
        let json = serde_json::to_string(&mc).unwrap();
        let parsed: RuntimeMemoryConfig = serde_json::from_str(&json).unwrap();
        assert!(parsed.enabled);
        assert_eq!(parsed.reflection_threshold, Some(50));
        assert!(parsed.consolidate_on_exit);
    }

    #[test]
    fn runtime_memory_config_defaults() {
        let mc: RuntimeMemoryConfig = serde_json::from_str("{}").unwrap();
        assert!(!mc.enabled);
        assert!(mc.reflection_threshold.is_none());
        assert!(!mc.consolidate_on_exit);
    }

    #[test]
    fn runtime_request_with_memory() {
        let json = r#"{
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "test",
            "agent": {"name": "a", "system_prompt": "sp"},
            "provider": {"type": "anthropic", "api_key": "sk-test", "model": "claude-sonnet-4-20250514"},
            "memory": {"enabled": true, "reflection_threshold": 30}
        }"#;
        let req: RuntimeRequest = serde_json::from_str(json).unwrap();
        let mem = req.memory.unwrap();
        assert!(mem.enabled);
        assert_eq!(mem.reflection_threshold, Some(30));
        assert!(!mem.consolidate_on_exit);
    }

    #[test]
    fn runtime_request_backward_compat_no_memory() {
        // Existing JSON without memory field — must still deserialize
        let json = r#"{
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "test",
            "agent": {"name": "a", "system_prompt": "sp"},
            "provider": {"type": "anthropic", "api_key": "sk-test", "model": "claude-sonnet-4-20250514"}
        }"#;
        let req: RuntimeRequest = serde_json::from_str(json).unwrap();
        assert!(req.memory.is_none());
    }

    #[test]
    fn advanced_config_new_fields_default() {
        let adv: RuntimeAdvancedConfig = serde_json::from_str("{}").unwrap();
        assert!(adv.enable_reflection.is_none());
        assert!(adv.session_prune.is_none());
        assert!(adv.recursive_summarization.is_none());
        assert!(adv.consolidate_on_exit.is_none());
        assert!(adv.max_tools_per_turn.is_none());
        assert!(adv.max_tool_output_bytes.is_none());
        assert!(adv.response_cache_size.is_none());
    }

    #[test]
    fn runtime_memory_store_config_defaults() {
        let mc = RuntimeMemoryStoreConfig::default();
        assert_eq!(mc.store_type, MemoryStoreType::InMemory);
        assert!(mc.database_url.is_none());
    }

    #[test]
    fn memory_store_type_serde() {
        assert_eq!(
            serde_json::to_string(&MemoryStoreType::InMemory).unwrap(),
            "\"in_memory\""
        );
        assert_eq!(
            serde_json::to_string(&MemoryStoreType::Postgres).unwrap(),
            "\"postgres\""
        );
        let parsed: MemoryStoreType = serde_json::from_str("\"postgres\"").unwrap();
        assert_eq!(parsed, MemoryStoreType::Postgres);
    }

    #[test]
    fn memory_store_type_rejects_unknown() {
        let result: Result<MemoryStoreType, _> = serde_json::from_str("\"redis\"");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn build_agent_with_memory_enabled() {
        let store = Arc::new(crate::memory::in_memory::InMemoryStore::new());
        let req = RuntimeRequest {
            task_id: uuid::Uuid::new_v4(),
            prompt: "test".into(),
            stream: false,
            tenant_id: Some(uuid::Uuid::new_v4()),
            user_id: Some("user1".into()),
            agent: RuntimeAgentConfig {
                name: "test-agent".into(),
                system_prompt: "sp".into(),
                max_turns: 5,
                max_tokens: 100,
                advanced: RuntimeAdvancedConfig::default(),
            },
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Anthropic,
                api_key: "key".into(),
                model: "model".into(),
                prompt_caching: false,
            },
            mcp_servers: vec![],
            builtin_tools: vec![],
            guardrails: None,
            memory: Some(RuntimeMemoryConfig {
                enabled: true,
                reflection_threshold: Some(50),
                consolidate_on_exit: false,
            }),
            messages: vec![],
            session_id: None,
        };

        let agent = build_agent_from_request(&req, None, None, None, Some(store), None).await;
        // Should succeed — agent is built with memory tools
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn build_agent_memory_disabled_no_tools() {
        let store = Arc::new(crate::memory::in_memory::InMemoryStore::new());
        let req = RuntimeRequest {
            task_id: uuid::Uuid::new_v4(),
            prompt: "test".into(),
            stream: false,
            tenant_id: None,
            user_id: None,
            agent: RuntimeAgentConfig {
                name: "test-agent".into(),
                system_prompt: "sp".into(),
                max_turns: 5,
                max_tokens: 100,
                advanced: RuntimeAdvancedConfig::default(),
            },
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Anthropic,
                api_key: "key".into(),
                model: "model".into(),
                prompt_caching: false,
            },
            mcp_servers: vec![],
            builtin_tools: vec![],
            guardrails: None,
            memory: Some(RuntimeMemoryConfig {
                enabled: false,
                reflection_threshold: None,
                consolidate_on_exit: false,
            }),
            messages: vec![],
            session_id: None,
        };

        // Memory disabled — should still build without memory tools
        let agent = build_agent_from_request(&req, None, None, None, Some(store), None).await;
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn build_agent_no_memory_store_ignores_config() {
        let req = RuntimeRequest {
            task_id: uuid::Uuid::new_v4(),
            prompt: "test".into(),
            stream: false,
            tenant_id: None,
            user_id: None,
            agent: RuntimeAgentConfig {
                name: "test-agent".into(),
                system_prompt: "sp".into(),
                max_turns: 5,
                max_tokens: 100,
                advanced: RuntimeAdvancedConfig::default(),
            },
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Anthropic,
                api_key: "key".into(),
                model: "model".into(),
                prompt_caching: false,
            },
            mcp_servers: vec![],
            builtin_tools: vec![],
            guardrails: None,
            memory: Some(RuntimeMemoryConfig {
                enabled: true,
                reflection_threshold: None,
                consolidate_on_exit: false,
            }),
            messages: vec![],
            session_id: None,
        };

        // No store provided — memory config is ignored gracefully
        let agent = build_agent_from_request(&req, None, None, None, None, None).await;
        assert!(agent.is_ok());
    }

    #[test]
    fn mcp_cache_hit_and_miss() {
        let cache = McpConnectionCache::new(std::time::Duration::from_secs(60));
        // Miss
        assert!(cache.get("https://example.com/mcp", 0).is_none());
        // Insert
        cache.insert("https://example.com/mcp", 0, vec![]);
        // Hit (empty tools)
        assert!(cache.get("https://example.com/mcp", 0).is_some());
        // Different auth hash = miss
        assert!(cache.get("https://example.com/mcp", 42).is_none());
    }

    #[test]
    fn mcp_cache_ttl_expiry() {
        let cache = McpConnectionCache::new(std::time::Duration::from_millis(1));
        cache.insert("https://example.com", 0, vec![]);
        std::thread::sleep(std::time::Duration::from_millis(5));
        // Expired
        assert!(cache.get("https://example.com", 0).is_none());
    }

    #[test]
    fn mcp_cache_evict_expired() {
        let cache = McpConnectionCache::new(std::time::Duration::from_millis(1));
        cache.insert("https://a.com", 0, vec![]);
        cache.insert("https://b.com", 0, vec![]);
        std::thread::sleep(std::time::Duration::from_millis(5));
        cache.evict_expired();
        let entries = cache.entries.read().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn hash_auth_deterministic() {
        let h1 = hash_auth(Some("Bearer token123"));
        let h2 = hash_auth(Some("Bearer token123"));
        assert_eq!(h1, h2);
        let h3 = hash_auth(Some("Bearer other"));
        assert_ne!(h1, h3);
        let h4 = hash_auth(None);
        assert_ne!(h1, h4);
    }

    #[test]
    fn mcp_cache_debug() {
        let cache = McpConnectionCache::new(std::time::Duration::from_secs(60));
        let debug = format!("{:?}", cache);
        assert!(debug.contains("McpConnectionCache"));
        assert!(debug.contains("entries: 0"));
    }

    #[test]
    fn runtime_config_mcp_cache_ttl_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.mcp_cache_ttl_seconds, 300);
    }

    #[test]
    fn provider_debug_redacts_api_key() {
        let config = RuntimeProviderConfig {
            provider_type: RuntimeProviderType::Anthropic,
            api_key: "sk-super-secret".into(),
            model: "claude-sonnet-4-20250514".into(),
            prompt_caching: false,
        };
        let debug = format!("{:?}", config);
        assert!(!debug.contains("sk-super-secret"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn runtime_message_role_serde() {
        assert_eq!(
            serde_json::to_string(&RuntimeMessageRole::User).unwrap(),
            "\"user\""
        );
        assert_eq!(
            serde_json::to_string(&RuntimeMessageRole::Assistant).unwrap(),
            "\"assistant\""
        );
        let parsed: RuntimeMessageRole = serde_json::from_str("\"user\"").unwrap();
        assert_eq!(parsed, RuntimeMessageRole::User);
    }

    #[test]
    fn runtime_message_serde_roundtrip() {
        let msg = RuntimeMessage {
            role: RuntimeMessageRole::User,
            content: "Hello agent".into(),
            timestamp: Some(chrono::Utc::now()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: RuntimeMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content, "Hello agent");
        assert!(parsed.timestamp.is_some());
    }

    #[test]
    fn runtime_message_timestamp_optional() {
        let json = r#"{"role":"assistant","content":"hi"}"#;
        let msg: RuntimeMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, RuntimeMessageRole::Assistant);
        assert!(msg.timestamp.is_none());
    }

    #[test]
    fn runtime_request_backward_compat_no_messages() {
        let json = r#"{
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "test",
            "agent": {"name": "a", "system_prompt": "sp"},
            "provider": {"type": "anthropic", "api_key": "sk-test", "model": "claude-sonnet-4-20250514"}
        }"#;
        let req: RuntimeRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages.is_empty());
        assert!(req.session_id.is_none());
    }

    #[test]
    fn runtime_request_with_messages() {
        let json = r#"{
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "What was I saying?",
            "session_id": "660e8400-e29b-41d4-a716-446655440001",
            "agent": {"name": "a", "system_prompt": "sp"},
            "provider": {"type": "anthropic", "api_key": "sk-test", "model": "claude-sonnet-4-20250514"},
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"}
            ]
        }"#;
        let req: RuntimeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, RuntimeMessageRole::User);
        assert_eq!(req.messages[1].role, RuntimeMessageRole::Assistant);
        assert!(req.session_id.is_some());
    }

    #[tokio::test]
    async fn build_agent_with_session_messages() {
        let req = RuntimeRequest {
            task_id: uuid::Uuid::new_v4(),
            prompt: "continue".into(),
            stream: false,
            tenant_id: None,
            user_id: None,
            agent: RuntimeAgentConfig {
                name: "test-agent".into(),
                system_prompt: "sp".into(),
                max_turns: 5,
                max_tokens: 100,
                advanced: RuntimeAdvancedConfig::default(),
            },
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Anthropic,
                api_key: "key".into(),
                model: "model".into(),
                prompt_caching: false,
            },
            mcp_servers: vec![],
            builtin_tools: vec![],
            guardrails: None,
            memory: None,
            messages: vec![
                RuntimeMessage {
                    role: RuntimeMessageRole::User,
                    content: "Hello".into(),
                    timestamp: None,
                },
                RuntimeMessage {
                    role: RuntimeMessageRole::Assistant,
                    content: "Hi there!".into(),
                    timestamp: None,
                },
            ],
            session_id: Some(uuid::Uuid::new_v4()),
        };

        let agent = build_agent_from_request(&req, None, None, None, None, None).await;
        assert!(agent.is_ok());
    }
}
