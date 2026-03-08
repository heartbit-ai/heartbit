use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{MatchedPath, Path, Query, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, Opts};
use serde::Deserialize;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use heartbit::daemon::core::KAFKA_REQUIRED;
use heartbit::{AgentEvent, HeartbitConfig, UserContext};

use super::types::{
    AppState, HealthResponse, ListQuery, ListResponse, ReadinessCheck, ReadinessResponse,
    ServiceAuth, SubmitRequest, SubmitResponse, UsageQueryParams, parse_rfc3339,
};

use super::auth::task_tenant_allowed;

/// Resolve per-user MCP tools via the transport pool (shared connections, per-user auth).
///
/// For each HTTP MCP server, stamps tools from the cached transport with a
/// `DynamicAuthResolver` carrying the user's identity. Stdio servers and A2A
/// agents are loaded fresh (they don't support per-request auth headers).
///
/// Falls back to `None` when auth is unavailable, letting callers use the
/// static `tool_cache` instead.
pub(crate) async fn mcp_tools_for_user(
    config: &HeartbitConfig,
    auth_provider: &Arc<dyn heartbit::AuthProvider>,
    user_id: &str,
    tenant_id: &str,
    transport_pool: &heartbit::McpTransportPool,
) -> Option<HashMap<String, Vec<Arc<dyn heartbit::tool::Tool>>>> {
    if !auth_provider.has_credentials(user_id, tenant_id) {
        tracing::debug!("no credentials for user, using cached MCP tools");
        return None;
    }
    tracing::debug!(
        user_id = %user_id,
        tenant_id = %tenant_id,
        "resolved per-user auth for MCP tools via transport pool"
    );
    let mut cache = HashMap::new();
    for agent in &config.agents {
        let mut agent_tools: Vec<Arc<dyn heartbit::tool::Tool>> = Vec::new();
        for entry in &agent.mcp_servers {
            if entry.is_stdio() {
                // Stdio transports don't use auth headers
                agent_tools
                    .extend(crate::load_mcp_tools(&agent.name, std::slice::from_ref(entry)).await);
            } else {
                // Use pool: stamp cached transport with per-user resolver
                let resolver: Arc<dyn heartbit::AuthResolver> = Arc::new(
                    heartbit::DynamicAuthResolver::new(
                        Arc::clone(auth_provider),
                        user_id,
                        tenant_id,
                    )
                    .with_resource(entry.resource().map(String::from))
                    .with_scopes(entry.scopes().map(|s| s.to_vec())),
                );
                match transport_pool.tools_for_user(entry.url(), resolver) {
                    Ok(Some(tools)) => agent_tools.extend(tools),
                    Ok(None) => {
                        // Pool miss — server wasn't warmed at startup, connect now
                        tracing::debug!(
                            server = %entry.display_name(),
                            "pool miss, connecting on demand"
                        );
                        match heartbit::McpClient::connect_with_auth(
                            entry.url(),
                            // Use a fresh per-user token for the handshake
                            auth_provider
                                .auth_header_for(user_id, tenant_id)
                                .await
                                .unwrap_or(None)
                                .unwrap_or_default(),
                        )
                        .await
                        {
                            Ok(client) => agent_tools.extend(client.into_tools()),
                            Err(e) => tracing::warn!(
                                agent = %agent.name,
                                server = %entry.display_name(),
                                error = %e,
                                "failed to connect MCP with user auth"
                            ),
                        }
                    }
                    Err(e) => tracing::warn!(
                        agent = %agent.name,
                        server = %entry.display_name(),
                        error = %e,
                        "transport pool error, falling back"
                    ),
                }
            }
        }
        // Also load A2A tools (these use their own auth)
        agent_tools.extend(crate::load_a2a_tools(&agent.name, &agent.a2a_agents).await);
        cache.insert(agent.name.clone(), agent_tools);
    }
    Some(cache)
}

// --- Handlers ---

pub(crate) async fn handle_submit(
    State(state): State<AppState>,
    user_context: Option<axum::Extension<UserContext>>,
    Json(body): Json<SubmitRequest>,
) -> impl IntoResponse {
    // Resolve user context: JWT middleware takes precedence; fall back to body user_context
    // (used when the caller authenticates with a service API key, e.g. CRM->Heartbit).
    let body_ctx: Option<UserContext> = user_context
        .is_none()
        .then(|| {
            body.user_context.as_ref().and_then(|bc| {
                let uid = bc.user_id.clone()?;
                let tid = bc.tenant_id.clone()?;
                Some(UserContext {
                    user_id: uid,
                    tenant_id: tid,
                    roles: bc.roles.clone(),
                    raw_token: bc.user_token.clone(),
                })
            })
        })
        .flatten();

    let effective_ctx: Option<&UserContext> = user_context
        .as_ref()
        .map(|axum::Extension(ctx)| ctx as &UserContext)
        .or(body_ctx.as_ref());

    // If entity_context was provided, prepend it to the task text so the agent
    // starts with full page context (L0 grounding) without an MCP round-trip.
    let task_text = if let Some(ref ctx) = body.entity_context {
        let ctx_str = serde_json::to_string(ctx).unwrap_or_default();
        format!(
            "<entity_context>\n{ctx_str}\n</entity_context>\n\n{}",
            body.task
        )
    } else {
        body.task
    };

    let result = if let Some(ctx) = effective_ctx {
        // Stash the raw JWT for token exchange (consumed by TokenExchangeAuthProvider).
        if let Some(ref token) = ctx.raw_token {
            let key = format!("{}:{}", ctx.tenant_id, ctx.user_id);
            if let Ok(mut tokens) = state.user_tokens.write() {
                tokens.insert(key, token.clone());
            }
        }
        state
            .handle
            .submit_task_with_user(task_text, "api", body.story_id, ctx)
            .await
    } else {
        state
            .handle
            .submit_task(task_text, "api", body.story_id)
            .await
    };

    match result {
        Ok(id) => {
            if let Some(ref m) = state.metrics {
                m.record_task_submitted(effective_ctx.map(|c| c.tenant_id.as_str()), "api");
            }
            (
                StatusCode::CREATED,
                Json(SubmitResponse {
                    id,
                    state: heartbit::TaskState::Pending,
                }),
            )
                .into_response()
        }
        Err(e) => {
            let status = if e.to_string().contains(KAFKA_REQUIRED) {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(serde_json::json!({ "error": e.to_string() }))).into_response()
        }
    }
}

pub(crate) async fn handle_list(
    State(state): State<AppState>,
    user_context: Option<axum::Extension<UserContext>>,
    Query(query): Query<ListQuery>,
) -> impl IntoResponse {
    let state_filter = query
        .state
        .as_deref()
        .and_then(heartbit::TaskState::from_db_str);
    let tenant_filter = user_context
        .as_ref()
        .map(|axum::Extension(ctx)| ctx.tenant_id.clone());
    match state.handle.list_tasks_filtered(
        query.limit,
        query.offset,
        query.source.as_deref(),
        state_filter,
        tenant_filter.as_deref(),
    ) {
        Ok((tasks, total)) => Json(ListResponse { tasks, total }).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub(crate) async fn handle_stats(
    State(state): State<AppState>,
    user_context: Option<axum::Extension<UserContext>>,
    Query(params): Query<UsageQueryParams>,
) -> impl IntoResponse {
    let tenant_filter = user_context
        .as_ref()
        .map(|axum::Extension(ctx)| ctx.tenant_id.clone());

    // If from/to provided, use usage_stats for filtered totals
    if params.from.is_some() || params.to.is_some() {
        let from = match parse_rfc3339(&params.from, "from") {
            Ok(v) => v,
            Err(e) => return e.into_response(),
        };
        let to = match parse_rfc3339(&params.to, "to") {
            Ok(v) => v,
            Err(e) => return e.into_response(),
        };
        let query = heartbit::UsageQuery {
            from,
            to,
            tenant_id: tenant_filter,
            ..Default::default()
        };
        match state.handle.usage_stats(&query) {
            Ok(rows) => {
                let row = rows.first().cloned().unwrap_or_default();
                let uptime_seconds = state.start_time.elapsed().as_secs();
                Json(serde_json::json!({
                    "total_tasks": row.task_count,
                    "completed_tasks": row.completed_count,
                    "failed_tasks": row.failed_count,
                    "total_tokens": {
                        "input_tokens": row.input_tokens,
                        "output_tokens": row.output_tokens,
                        "cache_read_tokens": row.cache_read_tokens,
                        "cache_creation_tokens": row.cache_creation_tokens,
                    },
                    "total_estimated_cost_usd": row.estimated_cost_usd,
                    "uptime_seconds": uptime_seconds,
                }))
                .into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response(),
        }
    } else {
        match state.handle.stats(tenant_filter.as_deref()) {
            Ok(stats) => {
                let uptime_seconds = state.start_time.elapsed().as_secs();
                let mut json = serde_json::json!({
                    "total_tasks": stats.total_tasks,
                    "tasks_by_state": stats.tasks_by_state,
                    "tasks_by_source": stats.tasks_by_source,
                    "active_tasks": stats.active_tasks,
                    "total_tokens": {
                        "input_tokens": stats.total_input_tokens,
                        "output_tokens": stats.total_output_tokens,
                        "cache_read_tokens": stats.total_cache_read_tokens,
                        "cache_creation_tokens": stats.total_cache_creation_tokens,
                    },
                    "total_estimated_cost_usd": stats.total_estimated_cost_usd,
                    "uptime_seconds": uptime_seconds,
                });
                // Include per-tenant breakdown when no tenant filter is applied (admin view)
                if tenant_filter.is_none()
                    && let Ok(tenant_rows) = state.handle.usage_stats(&heartbit::UsageQuery {
                        group_by: Some(heartbit::UsageGroupBy::Tenant),
                        ..Default::default()
                    })
                {
                    let by_tenant: serde_json::Map<String, serde_json::Value> = tenant_rows
                        .into_iter()
                        .filter_map(|r| {
                            let key = r.group_key?;
                            Some((
                                key,
                                serde_json::json!({
                                    "task_count": r.task_count,
                                    "input_tokens": r.input_tokens,
                                    "output_tokens": r.output_tokens,
                                    "estimated_cost_usd": r.estimated_cost_usd,
                                }),
                            ))
                        })
                        .collect();
                    if !by_tenant.is_empty() {
                        json["by_tenant"] = serde_json::Value::Object(by_tenant);
                    }
                }
                Json(json).into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response(),
        }
    }
}

pub(crate) async fn handle_usage(
    State(state): State<AppState>,
    user_context: Option<axum::Extension<UserContext>>,
    Query(params): Query<UsageQueryParams>,
) -> impl IntoResponse {
    let from = match parse_rfc3339(&params.from, "from") {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    let to = match parse_rfc3339(&params.to, "to") {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    // If JWT present, force tenant_id from JWT (tenant isolation)
    let tenant_id = user_context
        .as_ref()
        .map(|axum::Extension(ctx)| ctx.tenant_id.clone());

    let query = heartbit::UsageQuery {
        from,
        to,
        tenant_id,
        user_id: params.user_id,
        agent_name: params.agent_name,
        model_name: params.model_name,
        source: params.source,
        group_by: params.group_by,
    };

    match state.handle.usage_stats(&query) {
        Ok(rows) => Json(serde_json::json!({ "rows": rows })).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub(crate) async fn handle_get(
    State(state): State<AppState>,
    user_context: Option<axum::Extension<UserContext>>,
    service_auth: Option<axum::Extension<ServiceAuth>>,
    Path(id): Path<uuid::Uuid>,
) -> impl IntoResponse {
    let not_found = || {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "task not found" })),
        )
            .into_response()
    };
    match state.handle.get_task(id) {
        Ok(Some(task)) => {
            if !task_tenant_allowed(
                task.tenant_id.as_deref(),
                user_context.as_ref(),
                service_auth.as_ref(),
            ) {
                return not_found();
            }
            Json(task).into_response()
        }
        Ok(None) => not_found(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub(crate) async fn handle_cancel(
    State(state): State<AppState>,
    user_context: Option<axum::Extension<UserContext>>,
    service_auth: Option<axum::Extension<ServiceAuth>>,
    Path(id): Path<uuid::Uuid>,
) -> impl IntoResponse {
    let not_found = || {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "task not found" })),
        )
            .into_response()
    };
    // Verify tenant ownership before cancelling
    match state.handle.get_task(id) {
        Ok(Some(task)) => {
            if !task_tenant_allowed(
                task.tenant_id.as_deref(),
                user_context.as_ref(),
                service_auth.as_ref(),
            ) {
                return not_found();
            }
        }
        Ok(None) => return not_found(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    }
    let cancel_tenant = user_context
        .as_ref()
        .map(|axum::Extension(ctx)| ctx.tenant_id.clone());
    match state.handle.cancel_task(id).await {
        Ok(()) => {
            if let Some(ref m) = state.metrics {
                m.record_task_cancelled(cancel_tenant.as_deref());
            }
            StatusCode::NO_CONTENT.into_response()
        }
        Err(e) => {
            let status = if e.to_string().contains(KAFKA_REQUIRED) {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(serde_json::json!({ "error": e.to_string() }))).into_response()
        }
    }
}

/// REST approval endpoint for SSE-based tasks.
///
/// CRM (and other REST callers) send `{ "approved": true/false }` to resolve
/// a pending HITL approval gate. The corresponding `on_approval` callback
/// unblocks and the agent resumes or aborts the tool call.
#[derive(Deserialize)]
pub(crate) struct ApprovalBody {
    approved: bool,
}

pub(crate) async fn handle_approval(
    State(state): State<AppState>,
    user_context: Option<axum::Extension<UserContext>>,
    service_auth: Option<axum::Extension<ServiceAuth>>,
    Path(id): Path<uuid::Uuid>,
    Json(body): Json<ApprovalBody>,
) -> impl IntoResponse {
    let not_found = || {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "task not found" })),
        )
            .into_response()
    };
    match state.handle.get_task(id) {
        Ok(Some(task)) => {
            if !task_tenant_allowed(
                task.tenant_id.as_deref(),
                user_context.as_ref(),
                service_auth.as_ref(),
            ) {
                return not_found();
            }
        }
        Ok(None) => return not_found(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    }

    let decision = if body.approved {
        heartbit::ApprovalDecision::Allow
    } else {
        heartbit::ApprovalDecision::Deny
    };

    // Take the sender out of the pending map and send the decision
    let sender = {
        let mut pending = state
            .pending_approvals
            .lock()
            .expect("pending_approvals lock");
        pending.remove(&id)
    };
    match sender {
        Some(tx) => {
            let _ = tx.send(decision);
            Json(serde_json::json!({ "ok": true })).into_response()
        }
        None => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "no pending approval for this task" })),
        )
            .into_response(),
    }
}

pub(crate) async fn handle_stream(
    State(state): State<AppState>,
    user_context: Option<axum::Extension<UserContext>>,
    service_auth: Option<axum::Extension<ServiceAuth>>,
    Path(id): Path<uuid::Uuid>,
) -> impl IntoResponse {
    // Verify tenant ownership before subscribing to events
    let not_found = || {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "task not found" })),
        )
            .into_response()
    };
    match state.handle.get_task(id) {
        Ok(Some(ref task)) => {
            if !task_tenant_allowed(
                task.tenant_id.as_deref(),
                user_context.as_ref(),
                service_auth.as_ref(),
            ) {
                return not_found();
            }
        }
        Ok(None) => return not_found(),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    }
    match state.handle.subscribe_events(id) {
        Some(rx) => {
            let stream = BroadcastStream::new(rx).filter_map(
                |result: Result<
                    AgentEvent,
                    tokio_stream::wrappers::errors::BroadcastStreamRecvError,
                >| {
                    result.ok().and_then(|event| {
                        serde_json::to_string(&event)
                            .ok()
                            .map(|data| Ok::<_, Infallible>(Event::default().data(data)))
                    })
                },
            );
            Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "task not found or already completed" })),
        )
            .into_response(),
    }
}

/// Liveness probe -- returns 200 unless shutting down.
pub(crate) async fn handle_healthz(State(state): State<AppState>) -> impl IntoResponse {
    if state.cancel.is_cancelled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse {
                status: "shutting_down".into(),
                uptime_seconds: state.start_time.elapsed().as_secs(),
            }),
        )
            .into_response();
    }
    Json(HealthResponse {
        status: "ok".into(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
    })
    .into_response()
}

/// Readiness probe -- checks Kafka broker connectivity + not shutting down.
pub(crate) async fn handle_readyz(State(state): State<AppState>) -> impl IntoResponse {
    let mut checks = Vec::new();
    let mut all_ok = true;

    // Check: not shutting down
    let shutdown_ok = !state.cancel.is_cancelled();
    if !shutdown_ok {
        all_ok = false;
    }
    checks.push(ReadinessCheck {
        name: "shutdown".into(),
        ok: shutdown_ok,
        message: if shutdown_ok {
            None
        } else {
            Some("daemon is shutting down".into())
        },
    });

    // Check: Kafka broker reachable (skipped in HTTP-only mode)
    if let Some(ref brokers) = state.kafka_brokers {
        let brokers = brokers.clone();
        let kafka_ok = tokio::task::spawn_blocking(move || kafka_health_check(&brokers))
            .await
            .unwrap_or(false);
        if !kafka_ok {
            all_ok = false;
        }
        checks.push(ReadinessCheck {
            name: "kafka".into(),
            ok: kafka_ok,
            message: if kafka_ok {
                None
            } else {
                Some("kafka broker unreachable".into())
            },
        });
    }

    let status = if all_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status,
        Json(ReadinessResponse {
            ready: all_ok,
            checks,
        }),
    )
        .into_response()
}

/// Check Kafka broker connectivity via metadata fetch with a 5s timeout.
fn kafka_health_check(brokers: &str) -> bool {
    use rdkafka::ClientConfig;
    use rdkafka::config::RDKafkaLogLevel;
    use rdkafka::consumer::{BaseConsumer, Consumer};

    let consumer: Result<BaseConsumer, _> = ClientConfig::new()
        .set("bootstrap.servers", brokers)
        .set_log_level(RDKafkaLogLevel::Emerg)
        .create();

    match consumer {
        Ok(c) => c.fetch_metadata(None, Duration::from_secs(5)).is_ok(),
        Err(_) => false,
    }
}

/// Prometheus metrics endpoint.
pub(crate) async fn handle_metrics(State(state): State<AppState>) -> impl IntoResponse {
    match state.metrics {
        Some(ref m) => {
            let encoder = prometheus::TextEncoder::new();
            let families = m.registry().gather();

            match encoder.encode_to_string(&families) {
                Ok(body) => (
                    StatusCode::OK,
                    [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
                    body,
                )
                    .into_response(),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response(),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "metrics not enabled" })),
        )
            .into_response(),
    }
}

// --- HTTP metrics middleware ---

/// HTTP request metrics registered on the same Prometheus `Registry` as `DaemonMetrics`.
#[derive(Clone)]
pub(crate) struct HttpMetrics {
    requests_total: IntCounterVec,
    request_duration_seconds: HistogramVec,
}

impl HttpMetrics {
    pub(crate) fn register(
        registry: &prometheus::Registry,
    ) -> anyhow::Result<Self, prometheus::Error> {
        let requests_total = IntCounterVec::new(
            Opts::new(
                "heartbit_http_requests_total",
                "Total HTTP requests by method, path, and status",
            ),
            &["method", "path", "status"],
        )?;
        let request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "heartbit_http_request_duration_seconds",
                "HTTP request duration in seconds",
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]),
            &["method", "path"],
        )?;
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(request_duration_seconds.clone()))?;
        Ok(Self {
            requests_total,
            request_duration_seconds,
        })
    }
}

/// Axum middleware that records HTTP request metrics.
pub(crate) async fn http_metrics_middleware(
    State(http_metrics): State<HttpMetrics>,
    matched_path: Option<MatchedPath>,
    request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    let method = request.method().to_string();
    let path = matched_path
        .map(|mp| mp.as_str().to_string())
        .unwrap_or_else(|| "unknown".into());
    let start = Instant::now();

    let response = next.run(request).await;

    let status = response.status().as_u16().to_string();
    let duration = start.elapsed().as_secs_f64();

    http_metrics
        .requests_total
        .with_label_values(&[&method, &path, &status])
        .inc();
    http_metrics
        .request_duration_seconds
        .with_label_values(&[&method, &path])
        .observe(duration);

    response
}

/// Permissive CORS middleware for local dashboard access.
pub(crate) async fn cors_middleware(
    request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    // Handle preflight OPTIONS requests.
    if request.method() == axum::http::Method::OPTIONS {
        return axum::http::Response::builder()
            .status(204)
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            .header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization",
            )
            .header("Access-Control-Max-Age", "3600")
            .body(axum::body::Body::empty())
            .unwrap()
            .into_response();
    }

    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert("Access-Control-Allow-Origin", "*".parse().unwrap());
    response
}

/// Validate that a string is safe to use as a filesystem path component.
///
/// Rejects empty strings, path separators, `..` traversal, and absolute paths.
pub(crate) fn validate_path_component(s: &str) -> Result<(), String> {
    if s.is_empty() {
        return Err("empty path component".into());
    }
    if s.contains('/') || s.contains('\\') {
        return Err(format!("path separator in component: {s:?}"));
    }
    if s == "." || s == ".." || s.contains("..") {
        return Err(format!("path traversal in component: {s:?}"));
    }
    if std::path::Path::new(s).is_absolute() {
        return Err(format!("absolute path in component: {s:?}"));
    }
    Ok(())
}
