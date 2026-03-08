use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::http::StatusCode;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use heartbit::{
    DaemonHandle, DaemonMetrics, HeartbitConfig, JwtValidator, Memory, ObservabilityMode,
};

// --- Request / Response types ---

#[derive(Deserialize)]
pub(crate) struct SubmitRequest {
    pub task: String,
    #[serde(default)]
    pub story_id: Option<String>,
    /// Optional user context passed in the request body (used when the caller
    /// authenticates with a service API key rather than a user JWT). When a JWT
    /// middleware UserContext is already present, that takes precedence.
    #[serde(default)]
    pub user_context: Option<SubmitUserContext>,
    /// Optional CRM entity context injected by the frontend when the user is on
    /// an entity detail page. Prepended to the task text so the agent starts
    /// with full page context without needing an MCP prefetch round-trip.
    #[serde(default)]
    pub entity_context: Option<serde_json::Value>,
    /// Per-request MCP OAuth tokens from cloud/gateway.
    /// Key: MCP server URL, Value: bearer token.
    #[serde(default)]
    #[allow(dead_code)]
    pub mcp_auth_tokens: Option<HashMap<String, String>>,
}

/// User context embedded in the task submission body.
/// Mirrors the fields of `UserContext` so CRM can pass user identity alongside
/// a service-level API key, enabling per-user memory/workspace isolation and
/// RFC 8693 OBO token exchange without requiring user JWT on every request.
#[derive(Deserialize)]
pub(crate) struct SubmitUserContext {
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
    /// The user's original JWT — stored in the shared `user_tokens` map so that
    /// `TokenExchangeAuthProvider` can exchange it for a per-user OBO token
    /// when calling MCP servers on behalf of this user.
    pub user_token: Option<String>,
    #[serde(default)]
    pub roles: Vec<String>,
}

#[derive(Serialize)]
pub(crate) struct SubmitResponse {
    pub id: uuid::Uuid,
    pub state: heartbit::TaskState,
}

#[derive(Deserialize)]
pub(crate) struct ListQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
}

pub(crate) fn default_limit() -> usize {
    50
}

#[derive(Deserialize)]
pub(crate) struct UsageQueryParams {
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
    #[serde(default)]
    pub group_by: Option<heartbit::UsageGroupBy>,
    #[serde(default)]
    pub agent_name: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

/// Parse an optional RFC 3339 date string into `DateTime<Utc>`.
/// Returns `Err` with a 400 response if the string is present but malformed.
pub(crate) fn parse_rfc3339(
    s: &Option<String>,
    field: &str,
) -> Result<Option<chrono::DateTime<chrono::Utc>>, (StatusCode, Json<serde_json::Value>)> {
    match s.as_deref() {
        None | Some("") => Ok(None),
        Some(v) => chrono::DateTime::parse_from_rfc3339(v)
            .map(|dt| Some(dt.with_timezone(&chrono::Utc)))
            .map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": format!("invalid RFC 3339 date for '{field}': {v}")
                    })),
                )
            }),
    }
}

#[derive(Serialize)]
pub(crate) struct ListResponse {
    pub tasks: Vec<heartbit::DaemonTask>,
    pub total: usize,
}

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    pub status: String,
    pub uptime_seconds: u64,
}

#[derive(Serialize)]
pub(crate) struct ReadinessResponse {
    pub ready: bool,
    pub checks: Vec<ReadinessCheck>,
}

#[derive(Serialize)]
pub(crate) struct ReadinessCheck {
    pub name: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

// --- Axum state ---

pub(crate) type PendingApprovals =
    Arc<std::sync::Mutex<HashMap<uuid::Uuid, std::sync::mpsc::Sender<heartbit::ApprovalDecision>>>>;

/// Marker extension set by `auth_middleware` when a valid static bearer token is present.
/// Handlers use this to distinguish "authenticated service caller" (user_context = None, ServiceAuth = Some)
/// from "authenticated user" (user_context = Some) and "truly unauthenticated" (neither).
#[derive(Clone, Copy)]
pub(crate) struct ServiceAuth;

/// Runtime application state shared via Axum.
///
/// Several fields are consumed by the runner closure in `mod.rs` rather than
/// by HTTP handlers directly; they're kept here to extend their lifetimes.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct AppState {
    pub handle: DaemonHandle,
    pub start_time: Instant,
    pub metrics: Option<Arc<DaemonMetrics>>,
    pub cancel: CancellationToken,
    pub kafka_brokers: Option<String>,
    pub config: Arc<HeartbitConfig>,
    pub observability_mode: ObservabilityMode,
    pub shared_memory: Option<Arc<dyn Memory>>,
    pub workspace_dir: Option<PathBuf>,
    pub tool_cache: Arc<HashMap<String, Vec<Arc<dyn heartbit::tool::Tool>>>>,
    pub jwt_validator: Option<Arc<JwtValidator>>,
    pub user_tokens: Arc<std::sync::RwLock<HashMap<String, String>>>,
    pub auth_provider: Option<Arc<dyn heartbit::AuthProvider>>,
    pub transport_pool: Arc<heartbit::McpTransportPool>,
    pub pending_approvals: PendingApprovals,
}
