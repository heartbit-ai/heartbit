use std::collections::HashMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::Router;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{MatchedPath, Path, Query, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{delete, get, post};
use futures::SinkExt;
use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, Opts};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tokio_util::sync::CancellationToken;

use heartbit::channel::bridge::{InteractionBridge, OutboundMessage};
use heartbit::channel::session::{
    InMemorySessionStore, SessionMessage, SessionRole, SessionStore, format_session_context,
};
use heartbit::channel::types::{self, WsFrame};
use heartbit::daemon::kafka;
use heartbit::{
    AgentEvent, AgentOutput, ConsolidateSession, CronScheduler, DaemonCore, DaemonHandle,
    DaemonMetrics, Error as HeartbitError, HeartbitConfig, HeartbitPulseScheduler,
    InMemoryTaskStore, KafkaCommandProducer, Memory, ObservabilityMode,
};

use crate::{build_on_retry, build_provider_from_config, init_tracing_from_config};

// --- Request / Response types ---

#[derive(Deserialize)]
pub struct SubmitRequest {
    pub task: String,
    #[serde(default)]
    pub story_id: Option<String>,
}

#[derive(Serialize)]
pub struct SubmitResponse {
    pub id: uuid::Uuid,
    pub state: heartbit::TaskState,
}

#[derive(Deserialize)]
pub struct ListQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    50
}

#[derive(Serialize)]
pub struct ListResponse {
    pub tasks: Vec<heartbit::DaemonTask>,
    pub total: usize,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_seconds: u64,
}

#[derive(Serialize)]
pub struct ReadinessResponse {
    pub ready: bool,
    pub checks: Vec<ReadinessCheck>,
}

#[derive(Serialize)]
pub struct ReadinessCheck {
    pub name: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

// --- Axum state ---

#[derive(Clone)]
struct AppState {
    handle: DaemonHandle,
    start_time: Instant,
    metrics: Option<Arc<DaemonMetrics>>,
    sensor_metrics: Option<Arc<heartbit::SensorMetrics>>,
    cancel: CancellationToken,
    kafka_brokers: String,
    sessions: Arc<dyn SessionStore>,
    ws_interaction_timeout: Duration,
    ws_semaphore: Arc<Semaphore>,
    config: Arc<HeartbitConfig>,
    observability_mode: ObservabilityMode,
    todo_store: Option<Arc<heartbit::FileTodoStore>>,
    shared_memory: Option<Arc<dyn Memory>>,
    workspace_dir: Option<PathBuf>,
    tool_cache: Arc<HashMap<String, Vec<Arc<dyn heartbit::tool::Tool>>>>,
}

// --- Handlers ---

async fn handle_submit(
    State(state): State<AppState>,
    Json(body): Json<SubmitRequest>,
) -> impl IntoResponse {
    if let Some(ref m) = state.metrics {
        m.record_task_submitted();
    }
    match state
        .handle
        .submit_task(body.task, "api", body.story_id)
        .await
    {
        Ok(id) => (
            StatusCode::CREATED,
            Json(SubmitResponse {
                id,
                state: heartbit::TaskState::Pending,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn handle_list(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> impl IntoResponse {
    match state.handle.list_tasks(query.limit, query.offset) {
        Ok((tasks, total)) => Json(ListResponse { tasks, total }).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn handle_get(
    State(state): State<AppState>,
    Path(id): Path<uuid::Uuid>,
) -> impl IntoResponse {
    match state.handle.get_task(id) {
        Ok(Some(task)) => Json(task).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "task not found" })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn handle_cancel(
    State(state): State<AppState>,
    Path(id): Path<uuid::Uuid>,
) -> impl IntoResponse {
    match state.handle.cancel_task(id).await {
        Ok(()) => {
            if let Some(ref m) = state.metrics {
                m.record_task_cancelled();
            }
            StatusCode::NO_CONTENT.into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn handle_events(
    State(state): State<AppState>,
    Path(id): Path<uuid::Uuid>,
) -> impl IntoResponse {
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

/// Liveness probe — returns 200 unless shutting down.
async fn handle_healthz(State(state): State<AppState>) -> impl IntoResponse {
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

/// Readiness probe — checks Kafka broker connectivity + not shutting down.
async fn handle_readyz(State(state): State<AppState>) -> impl IntoResponse {
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

    // Check: Kafka broker reachable
    let brokers = state.kafka_brokers.clone();
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
///
/// Combines metrics from the daemon registry and the sensor registry (if present)
/// into a single response.
async fn handle_metrics(State(state): State<AppState>) -> impl IntoResponse {
    match state.metrics {
        Some(ref m) => {
            let encoder = prometheus::TextEncoder::new();

            // Gather from daemon registry
            let mut families = m.registry().gather();

            // Append sensor metrics if present
            if let Some(ref sm) = state.sensor_metrics {
                families.extend(sm.registry().gather());
            }

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

/// Read the current persistent todo list (heartbit pulse).
async fn handle_todo(State(state): State<AppState>) -> impl IntoResponse {
    match state.todo_store {
        Some(ref store) => {
            let list = store.get_list();
            Json(list).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "heartbit pulse not enabled" })),
        )
            .into_response(),
    }
}

// --- HTTP metrics middleware ---

/// HTTP request metrics registered on the same Prometheus `Registry` as `DaemonMetrics`.
#[derive(Clone)]
struct HttpMetrics {
    requests_total: IntCounterVec,
    request_duration_seconds: HistogramVec,
}

impl HttpMetrics {
    fn register(registry: &prometheus::Registry) -> Result<Self, prometheus::Error> {
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
async fn http_metrics_middleware(
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
async fn cors_middleware(
    request: axum::http::Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    // Handle preflight OPTIONS requests.
    if request.method() == axum::http::Method::OPTIONS {
        return axum::http::Response::builder()
            .status(204)
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            .header("Access-Control-Allow-Headers", "Content-Type")
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

// --- Daemon startup ---

pub async fn run_daemon(
    config_path: &std::path::Path,
    bind_override: Option<&str>,
    verbose: bool,
    observability_flag: Option<&str>,
) -> Result<()> {
    let config = HeartbitConfig::from_file(config_path)
        .with_context(|| format!("failed to load config from {}", config_path.display()))?;

    init_tracing_from_config(&config)?;

    let daemon_config = config
        .daemon
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("[daemon] section required in config for daemon mode"))?
        .clone();

    let bind = bind_override
        .map(String::from)
        .unwrap_or_else(|| daemon_config.bind.clone());

    // Create metrics if enabled (default: enabled when no config or when enabled=true)
    let metrics_enabled = daemon_config.metrics.as_ref().is_none_or(|m| m.enabled);
    let metrics = if metrics_enabled {
        let m = DaemonMetrics::new().context("failed to create Prometheus metrics")?;
        tracing::info!("Prometheus metrics enabled");
        Some(Arc::new(m))
    } else {
        None
    };

    // Ensure Kafka topics exist
    tracing::info!("ensuring Kafka topics exist");
    kafka::ensure_topics(&daemon_config.kafka)
        .await
        .context("failed to ensure Kafka topics")?;

    // Create producer + consumer
    let producer =
        kafka::create_producer(&daemon_config.kafka).context("failed to create Kafka producer")?;
    let consumer = kafka::create_commands_consumer(&daemon_config.kafka)
        .context("failed to create Kafka consumer")?;

    // Build task store: PostgreSQL if configured, in-memory otherwise
    let store: Arc<dyn heartbit::TaskStore> = if let Some(ref db_url) = daemon_config.database_url {
        let pg_store = heartbit::PostgresTaskStore::connect(db_url)
            .await
            .context("failed to connect to task database")?;
        pg_store
            .run_migration()
            .await
            .context("failed to run task migration")?;
        tracing::info!("task store: PostgreSQL");
        Arc::new(pg_store)
    } else {
        tracing::info!("task store: in-memory (tasks lost on restart)");
        Arc::new(InMemoryTaskStore::new())
    };

    // Create cancellation token
    let cancel = CancellationToken::new();

    // Create DaemonCore + DaemonHandle
    let (core, handle) = DaemonCore::new(
        &daemon_config,
        consumer,
        producer.clone(),
        store,
        cancel.clone(),
    );

    // Start cron scheduler if schedules configured
    if !daemon_config.schedules.is_empty() {
        let cron = CronScheduler::new(
            &daemon_config.schedules,
            Arc::new(KafkaCommandProducer::new(producer.clone())),
            &daemon_config.kafka.commands_topic,
        )
        .context("failed to create cron scheduler")?;
        let cron_cancel = cancel.clone();
        tokio::spawn(async move {
            cron.run(cron_cancel).await;
        });
        tracing::info!(
            schedules = daemon_config.schedules.len(),
            "cron scheduler started"
        );
    }

    // Start heartbit pulse scheduler if configured and enabled
    let pulse_todo_store: Option<Arc<heartbit::FileTodoStore>> = if let Some(ref pulse_config) =
        daemon_config.heartbit_pulse
        && pulse_config.enabled
    {
        let ws_root = crate::workspace_root_from_config(&config);
        let ws = heartbit::Workspace::open(&ws_root)
            .context("failed to open workspace for heartbit pulse")?;
        let pulse = HeartbitPulseScheduler::new(
            pulse_config,
            ws.root(),
            Arc::new(KafkaCommandProducer::new(producer.clone())),
            &daemon_config.kafka.commands_topic,
        )
        .context("failed to create heartbit pulse scheduler")?;
        let store = pulse.todo_store().clone();
        let pulse_cancel = cancel.clone();
        tokio::spawn(async move {
            pulse.run(pulse_cancel).await;
        });
        tracing::info!(
            interval_secs = pulse_config.interval_seconds,
            "heartbit pulse scheduler started"
        );
        Some(store)
    } else {
        None
    };

    // Create shared memory store (one store for all execution paths)
    let shared_memory: Option<Arc<dyn Memory>> = if let Some(ref mem_config) = config.memory {
        match crate::create_memory_store(mem_config).await {
            Ok(store) => {
                tracing::info!("daemon shared memory: enabled");
                Some(store)
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to create shared memory, continuing without");
                None
            }
        }
    } else {
        None
    };

    // Provision workspace once (idempotent, reused by all execution paths)
    let daemon_workspace_dir =
        crate::provision_workspace(&crate::workspace_root_from_config(&config));

    // Pre-load MCP + A2A tools once for all agents (daemon reuses across tasks)
    let tool_cache: Arc<HashMap<String, Vec<Arc<dyn heartbit::tool::Tool>>>> = {
        let mut cache = HashMap::new();
        for agent in &config.agents {
            let mut agent_tools = crate::load_mcp_tools(&agent.name, &agent.mcp_servers).await;
            agent_tools.extend(crate::load_a2a_tools(&agent.name, &agent.a2a_agents).await);
            if !agent_tools.is_empty() {
                tracing::info!(
                    agent = %agent.name,
                    tools = agent_tools.len(),
                    "cached MCP/A2A tools"
                );
            }
            cache.insert(agent.name.clone(), agent_tools);
        }
        Arc::new(cache)
    };

    // Create sensor metrics (shared with AppState for /metrics endpoint)
    let sensor_metrics: Option<Arc<heartbit::SensorMetrics>> = if metrics_enabled
        && daemon_config
            .sensors
            .as_ref()
            .is_some_and(|s| s.enabled && !s.sources.is_empty())
    {
        let m =
            heartbit::SensorMetrics::new().context("failed to create sensor Prometheus metrics")?;
        Some(Arc::new(m))
    } else {
        None
    };

    // Start sensor manager if sensors configured and enabled
    if let Some(ref sensor_config) = daemon_config.sensors
        && sensor_config.enabled
        && !sensor_config.sources.is_empty()
    {
        // Ensure sensor-specific Kafka topics exist
        kafka::ensure_sensor_topics(&daemon_config.kafka)
            .await
            .context("failed to ensure sensor Kafka topics")?;

        // Build SLM provider for triage
        let slm_provider = build_provider_from_config(&config, None)
            .context("failed to build SLM provider for sensor triage")?;

        let sensor_manager = heartbit::SensorManager::new(
            sensor_config.clone(),
            producer.clone(),
            slm_provider,
            sensor_metrics.clone(),
            &daemon_config.kafka.commands_topic,
            &daemon_config.kafka.dead_letter_topic,
        );

        let sensor_cancel = cancel.clone();
        let sensor_kafka_config = daemon_config.kafka.clone();
        tokio::spawn(async move {
            if let Err(e) = sensor_manager
                .run(&sensor_kafka_config, sensor_cancel)
                .await
            {
                tracing::error!(error = %e, "sensor manager failed");
            }
        });

        tracing::info!(
            sources = sensor_config.sources.len(),
            "sensor manager started"
        );
    }

    // Signal handler
    let signal_cancel = cancel.clone();
    tokio::spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigint = signal(SignalKind::interrupt()).expect("SIGINT handler");
        let mut sigterm = signal(SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = sigint.recv() => tracing::info!("SIGINT received"),
            _ = sigterm.recv() => tracing::info!("SIGTERM received"),
        }
        signal_cancel.cancel();
    });

    // Resolve observability mode
    let config_obs = config
        .telemetry
        .as_ref()
        .and_then(|t| t.observability_mode.as_deref());
    let mode = crate::resolve_observability(observability_flag, config_obs, verbose);

    // Build the runner closure that creates an Orchestrator per task
    let config_arc = Arc::new(config);
    let config_for_state = config_arc.clone();
    let runner_metrics = metrics.clone();
    let runner_todo_store = pulse_todo_store.clone();
    let runner_memory = shared_memory.clone();
    let runner_workspace = daemon_workspace_dir.clone();
    let runner_tools = tool_cache.clone();
    let build_runner = move |_task_id: uuid::Uuid,
                             task_text: String,
                             source: String,
                             story_id: Option<String>,
                             on_event_fn: Arc<dyn Fn(AgentEvent) + Send + Sync>|
          -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AgentOutput, HeartbitError>> + Send>,
    > {
        let config = config_arc.clone();
        let task_metrics = runner_metrics.clone();
        let todo_store = runner_todo_store.clone();
        let memory = runner_memory.clone();
        let workspace_dir = runner_workspace.clone();
        let tools = runner_tools.clone();
        Box::pin(async move {
            // Wrap on_event to also record metrics
            let on_event: Arc<heartbit::OnEvent> = if let Some(ref m) = task_metrics {
                let inner = on_event_fn;
                let metrics = m.clone();
                Arc::new(move |event: AgentEvent| {
                    metrics.record_event(&event);
                    inner(event);
                })
            } else {
                on_event_fn
            };

            let on_retry = build_on_retry(&on_event);
            let provider = build_provider_from_config(&config, Some(on_retry.clone()))
                .map_err(|e| HeartbitError::Daemon(e.to_string()))?;

            // Track active tasks (after provider creation to avoid gauge leak on error)
            if let Some(ref m) = task_metrics {
                m.tasks_active().inc();
            }
            let start = Instant::now();

            let on_text: Arc<heartbit::OnText> = Arc::new(|_: &str| {});

            let result = crate::build_orchestrator_from_config(
                provider,
                &config,
                &task_text,
                on_text,
                None, // no approval in daemon mode
                Some(on_event),
                mode,
                story_id.as_deref(),
                None,   // no interactive question callback in daemon mode
                memory, // shared daemon memory
                workspace_dir,
                todo_store, // persistent daemon todo store
                Some(&tools),
            )
            .await
            .map_err(|e| HeartbitError::Daemon(e.to_string()));

            let duration_secs = start.elapsed().as_secs_f64();
            if let Some(ref m) = task_metrics {
                m.tasks_active().dec();
                m.record_task_by_source(&source);
                match &result {
                    Ok(_) => m.record_task_completed(duration_secs),
                    Err(_) => m.record_task_failed(duration_secs),
                }
                // Record pulse-specific metrics when source is "heartbit"
                if source == "heartbit" {
                    m.record_pulse_run();
                    match &result {
                        Ok(output) if output.result.contains("HEARTBIT_OK") => {
                            m.record_pulse_ok();
                        }
                        Ok(_) => m.record_pulse_action(),
                        Err(_) => {} // failures already tracked by record_task_failed
                    }
                }
            }

            result
        })
    };

    // Build Kafka brokers string for readiness check
    let kafka_brokers = daemon_config.kafka.brokers.clone();

    // WebSocket configuration
    let ws_timeout = daemon_config
        .ws
        .as_ref()
        .map(|ws| Duration::from_secs(ws.interaction_timeout_seconds))
        .unwrap_or(Duration::from_secs(120));
    let ws_max_connections = daemon_config
        .ws
        .as_ref()
        .map(|ws| ws.max_connections)
        .unwrap_or(100);
    let ws_enabled = daemon_config.ws.as_ref().is_none_or(|ws| ws.enabled);

    // Build session store: PostgreSQL if configured, in-memory otherwise
    let sessions: Arc<dyn SessionStore> = if let Some(ref db_url) = daemon_config
        .ws
        .as_ref()
        .and_then(|ws| ws.database_url.clone())
    {
        let pg_store = heartbit::PostgresSessionStore::connect(db_url)
            .await
            .context("failed to connect to session database")?;
        pg_store
            .run_migration()
            .await
            .context("failed to run session migration")?;
        tracing::info!("session store: PostgreSQL");
        Arc::new(pg_store)
    } else {
        tracing::info!("session store: in-memory (sessions lost on restart)");
        Arc::new(InMemorySessionStore::new())
    };

    // Start HTTP server
    let app_state = AppState {
        handle,
        start_time: Instant::now(),
        metrics: metrics.clone(),
        sensor_metrics,
        cancel: cancel.clone(),
        kafka_brokers,
        sessions,
        ws_interaction_timeout: ws_timeout,
        ws_semaphore: Arc::new(Semaphore::new(ws_max_connections)),
        config: config_for_state,
        observability_mode: mode,
        todo_store: pulse_todo_store,
        shared_memory: shared_memory.clone(),
        workspace_dir: daemon_workspace_dir,
        tool_cache: tool_cache.clone(),
    };

    let mut routes = Router::new()
        .route("/tasks", post(handle_submit))
        .route("/tasks", get(handle_list))
        .route("/tasks/{id}", get(handle_get))
        .route("/tasks/{id}", delete(handle_cancel))
        .route("/tasks/{id}/events", get(handle_events))
        .route("/healthz", get(handle_healthz))
        .route("/readyz", get(handle_readyz))
        .route("/health", get(handle_healthz))
        .route("/metrics", get(handle_metrics))
        .route("/todo", get(handle_todo));

    if ws_enabled {
        routes = routes.route("/ws", get(handle_ws_upgrade));
        tracing::info!("WebSocket endpoint enabled on /ws");
    }

    // Start Telegram adapter if configured
    if let Some(ref tg_config) = daemon_config.telegram
        && tg_config.enabled
    {
        let token = tg_config
            .token
            .clone()
            .or_else(|| std::env::var("HEARTBIT_TELEGRAM_TOKEN").ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Telegram enabled but no token: set daemon.telegram.token or HEARTBIT_TELEGRAM_TOKEN"
                )
            })?;

        let bot = teloxide::Bot::new(&token);
        let tg_sessions = app_state.sessions.clone();
        let tg_config = tg_config.clone();
        let tg_cancel = app_state.cancel.clone();
        let tg_app_config = app_state.config.clone();
        let tg_metrics = app_state.metrics.clone();

        // Capture shared resources for the Telegram RunTask closure
        let tg_todo_store = app_state.todo_store.clone();
        let tg_workspace = app_state.workspace_dir.clone();
        let tg_tools = app_state.tool_cache.clone();

        // Build the RunTask closure that reuses build_orchestrator_from_config
        let run_task: Arc<heartbit::channel::telegram::RunTask> = Arc::new(move |input| {
            let config = tg_app_config.clone();
            let bridge = input.bridge;
            let task_text = input.task_text;
            let user_ns = input.user_namespace;
            let m = tg_metrics.clone();
            let todo_store = tg_todo_store.clone();
            let workspace_dir = tg_workspace.clone();
            let tools = tg_tools.clone();
            Box::pin(async move {
                // No streaming preview for Telegram — only send the final
                // HTML-formatted result to avoid a raw-markdown flash.
                let on_text: Arc<heartbit::OnText> = Arc::new(|_: &str| {});
                let on_event = bridge.make_on_event();
                let on_question = bridge.make_on_question();

                let on_retry = build_on_retry(&on_event);
                let provider = build_provider_from_config(&config, Some(on_retry))
                    .map_err(|e| HeartbitError::Daemon(e.to_string()))?;

                // Track active tasks (after provider creation to avoid gauge leak on error)
                if let Some(ref m) = m {
                    m.tasks_active().inc();
                }
                let start = Instant::now();
                let result = crate::build_orchestrator_from_config(
                    provider,
                    &config,
                    &task_text,
                    on_text,
                    None, // no approval for Telegram — user already initiated the task
                    Some(on_event),
                    mode,
                    user_ns.as_deref(),
                    Some(on_question),
                    input.memory,
                    workspace_dir,
                    todo_store,
                    Some(&tools),
                )
                .await
                .map_err(|e| HeartbitError::Daemon(e.to_string()));

                let duration_secs = start.elapsed().as_secs_f64();
                if let Some(ref m) = m {
                    m.tasks_active().dec();
                    m.record_task_by_source("telegram");
                    match &result {
                        Ok(_) => m.record_task_completed(duration_secs),
                        Err(_) => m.record_task_failed(duration_secs),
                    }
                }

                Ok(result?.result)
            })
        });

        // Use shared daemon memory for Telegram (avoids duplicate stores)
        let tg_memory = shared_memory.clone();
        if tg_memory.is_some() {
            tracing::info!("Telegram memory: using shared daemon store");
        }

        // Wire consolidation callback: prune weak memories on idle sessions.
        //
        // NOTE: `Memory::prune()` is namespace-blind — it scans all entries in the
        // underlying store, not just the user's namespace. This is acceptable because
        // pruning only removes entries that are both weak AND old (entries that should
        // be cleaned up regardless of which user triggered the sweep).
        let consolidation_cb: Option<Arc<ConsolidateSession>> = tg_memory.as_ref().map(|mem| {
            let memory = mem.clone();
            let cb: Arc<ConsolidateSession> = Arc::new(move |user_id: i64| {
                let memory = memory.clone();
                Box::pin(async move {
                    let pruned = heartbit::prune_weak_entries(
                        &memory,
                        heartbit::DEFAULT_MIN_STRENGTH,
                        heartbit::default_min_age(),
                    )
                    .await?;
                    if pruned > 0 {
                        tracing::info!(user_id, pruned, "pruned weak memories on idle");
                    }
                    Ok(())
                })
                    as Pin<Box<dyn std::future::Future<Output = Result<(), HeartbitError>> + Send>>
            });
            cb
        });

        let adapter = Arc::new(heartbit::channel::telegram::TelegramAdapter::new(
            bot,
            tg_config,
            tg_sessions,
            tg_memory,
            run_task,
            consolidation_cb,
        ));

        tokio::spawn(async move {
            adapter.run(tg_cancel).await;
        });

        tracing::info!("Telegram bot started");
    }

    let mut app = routes.with_state(app_state);

    // Permissive CORS for local dashboard access (file:// or localhost origins).
    app = app.layer(middleware::from_fn(cors_middleware));

    // Add HTTP metrics middleware when metrics are enabled
    if let Some(ref m) = metrics {
        let http_metrics =
            HttpMetrics::register(m.registry()).context("failed to register HTTP metrics")?;
        app = app.layer(middleware::from_fn_with_state(
            http_metrics,
            http_metrics_middleware,
        ));
    }

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("failed to bind to {bind}"))?;
    tracing::info!(bind = %bind, "daemon HTTP server started");

    let http_cancel = cancel.clone();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                http_cancel.cancelled().await;
            })
            .await
            .ok();
    });

    // Run the DaemonCore consumer loop (blocks until cancellation)
    tracing::info!("daemon core started, consuming from Kafka");
    core.run(build_runner).await.context("daemon core error")?;

    tracing::info!("daemon shut down gracefully");
    Ok(())
}

// --- WebSocket handler ---

async fn handle_ws_upgrade(
    State(state): State<AppState>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    // Acquire connection permit
    let permit = match state.ws_semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "max WebSocket connections reached",
            )
                .into_response();
        }
    };

    ws.on_upgrade(move |socket| async move {
        handle_ws_connection(socket, state).await;
        drop(permit);
    })
    .into_response()
}

async fn handle_ws_connection(socket: WebSocket, state: AppState) {
    let (mut ws_tx, mut ws_rx) = futures::StreamExt::split(socket);

    // Per-connection outbound channel
    let (outbound_tx, mut outbound_rx) = tokio::sync::mpsc::channel::<OutboundMessage>(256);

    // Per-connection tracking of running tasks for abort
    let running_tasks: Arc<std::sync::RwLock<HashMap<uuid::Uuid, CancellationToken>>> =
        Arc::new(std::sync::RwLock::new(HashMap::new()));

    // Single bridge per connection — interaction_ids are globally unique so one
    // bridge can serve multiple concurrent tasks on the same connection.
    let bridge = Arc::new(InteractionBridge::new(
        outbound_tx.clone(),
        state.ws_interaction_timeout,
    ));

    // Outbound pump: reads from bridge outbound channel, converts to WS frames, sends
    let seq = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let cancel = state.cancel.clone();
    let pump_handle = tokio::spawn(async move {
        loop {
            tokio::select! {
                msg = outbound_rx.recv() => {
                    let Some(msg) = msg else { break; };
                    let frame = outbound_to_frame(msg, &seq);
                    let json = match serde_json::to_string(&frame) {
                        Ok(j) => j,
                        Err(_) => continue,
                    };
                    if ws_tx.send(WsMessage::Text(json.into())).await.is_err() {
                        break;
                    }
                }
                _ = cancel.cancelled() => break,
            }
        }
        let _ = ws_tx.close().await;
    });

    // Inbound loop: reads WS frames, dispatches to handlers
    while let Some(msg_result) = futures::StreamExt::next(&mut ws_rx).await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(error = %e, "WebSocket receive error");
                break;
            }
        };

        let text = match msg {
            WsMessage::Text(t) => t,
            WsMessage::Close(_) => break,
            WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
            WsMessage::Binary(_) => continue,
        };

        let frame: WsFrame = match serde_json::from_str(&text) {
            Ok(f) => f,
            Err(e) => {
                let err_frame = WsFrame::err("unknown", format!("invalid frame: {e}"));
                let _ = outbound_tx.try_send(OutboundMessage::RawFrame(err_frame));
                continue;
            }
        };

        match frame {
            WsFrame::Req { id, method, params } => {
                let response = dispatch_method(
                    &id,
                    &method,
                    params,
                    &state,
                    &outbound_tx,
                    &bridge,
                    &running_tasks,
                )
                .await;
                let _ = outbound_tx.try_send(OutboundMessage::RawFrame(response));
            }
            // Clients should only send Req frames; ignore others
            _ => continue,
        }
    }

    // Cleanup: cancel running tasks and abort pump
    if let Ok(tasks) = running_tasks.read() {
        for token in tasks.values() {
            token.cancel();
        }
    }
    pump_handle.abort();
}

/// Dispatch a WS request to the appropriate handler.
async fn dispatch_method(
    id: &str,
    method: &str,
    params: serde_json::Value,
    state: &AppState,
    outbound_tx: &tokio::sync::mpsc::Sender<OutboundMessage>,
    bridge: &Arc<InteractionBridge>,
    running_tasks: &Arc<std::sync::RwLock<HashMap<uuid::Uuid, CancellationToken>>>,
) -> WsFrame {
    match method {
        types::method::SESSION_CREATE => handle_ws_session_create(id, params, state),
        types::method::SESSION_LIST => handle_ws_session_list(id, state),
        types::method::SESSION_DELETE => handle_ws_session_delete(id, params, state),
        types::method::CHAT_HISTORY => handle_ws_chat_history(id, params, state),
        types::method::CHAT_SEND => {
            handle_ws_chat_send(id, params, state, outbound_tx, bridge, running_tasks).await
        }
        types::method::CHAT_ABORT => handle_ws_chat_abort(id, params, running_tasks),
        types::method::APPROVAL_RESOLVE => handle_ws_approval_resolve(id, params, bridge),
        types::method::INPUT_RESOLVE => handle_ws_input_resolve(id, params, bridge),
        types::method::QUESTION_RESOLVE => handle_ws_question_resolve(id, params, bridge),
        _ => WsFrame::err(id, format!("unknown method: {method}")),
    }
}

fn handle_ws_session_create(id: &str, params: serde_json::Value, state: &AppState) -> WsFrame {
    let title = params
        .get("title")
        .and_then(|v| v.as_str())
        .map(String::from);
    match state.sessions.create(title) {
        Ok(session) => {
            let result = types::SessionCreateResult {
                session_id: session.id,
            };
            WsFrame::ok(id, serde_json::to_value(result).unwrap_or_default())
        }
        Err(e) => WsFrame::err(id, e.to_string()),
    }
}

fn handle_ws_session_list(id: &str, state: &AppState) -> WsFrame {
    match state.sessions.list() {
        Ok(sessions) => {
            let summaries: Vec<types::SessionSummary> = sessions
                .iter()
                .map(|s| types::SessionSummary {
                    id: s.id,
                    title: s.title.clone(),
                    created_at: s.created_at,
                    message_count: s.messages.len(),
                })
                .collect();
            let result = types::SessionListResult {
                sessions: summaries,
            };
            WsFrame::ok(id, serde_json::to_value(result).unwrap_or_default())
        }
        Err(e) => WsFrame::err(id, e.to_string()),
    }
}

fn handle_ws_session_delete(id: &str, params: serde_json::Value, state: &AppState) -> WsFrame {
    let parsed: types::SessionDeleteParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return WsFrame::err(id, format!("invalid params: {e}")),
    };
    match state.sessions.delete(parsed.session_id) {
        Ok(deleted) => WsFrame::ok(id, serde_json::json!({ "deleted": deleted })),
        Err(e) => WsFrame::err(id, e.to_string()),
    }
}

fn handle_ws_chat_history(id: &str, params: serde_json::Value, state: &AppState) -> WsFrame {
    let parsed: types::ChatHistoryParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return WsFrame::err(id, format!("invalid params: {e}")),
    };
    match state.sessions.get(parsed.session_id) {
        Ok(Some(session)) => WsFrame::ok(
            id,
            serde_json::to_value(&session.messages).unwrap_or_default(),
        ),
        Ok(None) => WsFrame::err(id, "session not found"),
        Err(e) => WsFrame::err(id, e.to_string()),
    }
}

async fn handle_ws_chat_send(
    id: &str,
    params: serde_json::Value,
    state: &AppState,
    outbound_tx: &tokio::sync::mpsc::Sender<OutboundMessage>,
    bridge: &Arc<InteractionBridge>,
    running_tasks: &Arc<std::sync::RwLock<HashMap<uuid::Uuid, CancellationToken>>>,
) -> WsFrame {
    let parsed: types::ChatSendParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return WsFrame::err(id, format!("invalid params: {e}")),
    };

    let session_id = parsed.session_id;

    // Load session with prior history (before recording new message)
    let session = match state.sessions.get(session_id) {
        Ok(Some(s)) => s,
        Ok(None) => return WsFrame::err(id, "session not found"),
        Err(e) => return WsFrame::err(id, e.to_string()),
    };

    // Build task text with conversation history for multi-turn context
    let task_text = format_session_context(&session.messages, &parsed.message);

    // Record user message in session
    let user_msg = SessionMessage {
        role: SessionRole::User,
        content: parsed.message.clone(),
        timestamp: chrono::Utc::now(),
    };
    if let Err(e) = state.sessions.add_message(session_id, user_msg) {
        return WsFrame::err(id, e.to_string());
    }

    // Create per-task cancellation token
    let task_id = uuid::Uuid::new_v4();
    let task_cancel = CancellationToken::new();
    {
        let mut rt = running_tasks.write().expect("running tasks lock");
        rt.insert(task_id, task_cancel.clone());
    }

    // Build callbacks from connection-level bridge
    let on_text = bridge.make_on_text(session_id);
    let on_event = bridge.make_on_event(session_id);
    let on_approval = bridge.make_on_approval(session_id);
    let on_question = bridge.make_on_question(session_id);

    // Spawn agent task
    let config = state.config.clone();
    let sessions = state.sessions.clone();
    let running_tasks_clone = running_tasks.clone();
    let outbound = outbound_tx.clone();
    let mode = state.observability_mode;
    let metrics = state.metrics.clone();
    let ws_memory = state.shared_memory.clone();
    let ws_workspace = state.workspace_dir.clone();
    let ws_todo = state.todo_store.clone();
    let ws_tools = state.tool_cache.clone();

    tokio::spawn(async move {
        let result = run_interactive_task(
            &config,
            &task_text,
            InteractiveTaskParams {
                on_text,
                on_approval,
                on_event,
                on_question,
                mode,
                cancel: task_cancel,
            },
            metrics.as_deref(),
            ws_memory,
            ws_workspace,
            ws_todo,
            Some(&ws_tools),
            "ws",
        )
        .await;

        // Record result in session
        let (final_text, is_err) = match &result {
            Ok(output) => (output.result.clone(), false),
            Err(e) => (e.to_string(), true),
        };

        let assistant_msg = SessionMessage {
            role: SessionRole::Assistant,
            content: final_text.clone(),
            timestamp: chrono::Utc::now(),
        };
        let _ = sessions.add_message(session_id, assistant_msg);

        // Send final/error event via proper OutboundMessage variants
        if is_err {
            let _ = outbound.try_send(OutboundMessage::ChatError {
                session_id,
                error: final_text,
            });
        } else {
            let _ = outbound.try_send(OutboundMessage::ChatFinal {
                session_id,
                result: final_text,
            });
        }

        // Cleanup
        if let Ok(mut rt) = running_tasks_clone.write() {
            rt.remove(&task_id);
        }
    });

    // Return task_id to client
    let result = types::ChatSendResult { task_id };
    WsFrame::ok(id, serde_json::to_value(result).unwrap_or_default())
}

fn handle_ws_chat_abort(
    id: &str,
    params: serde_json::Value,
    running_tasks: &Arc<std::sync::RwLock<HashMap<uuid::Uuid, CancellationToken>>>,
) -> WsFrame {
    let parsed: types::ChatAbortParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return WsFrame::err(id, format!("invalid params: {e}")),
    };
    let cancelled = if let Ok(tasks) = running_tasks.read() {
        if let Some(token) = tasks.get(&parsed.task_id) {
            token.cancel();
            true
        } else {
            false
        }
    } else {
        false
    };
    WsFrame::ok(id, serde_json::json!({ "cancelled": cancelled }))
}

fn handle_ws_approval_resolve(
    id: &str,
    params: serde_json::Value,
    bridge: &Arc<InteractionBridge>,
) -> WsFrame {
    let parsed: types::ApprovalResolveParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return WsFrame::err(id, format!("invalid params: {e}")),
    };
    let decision = match parsed.parse_decision() {
        Some(d) => d,
        None => return WsFrame::err(id, format!("invalid decision: {}", parsed.decision)),
    };
    match bridge.resolve_approval(parsed.interaction_id, decision) {
        Ok(()) => WsFrame::ok(id, serde_json::json!({})),
        Err(e) => WsFrame::err(id, e.to_string()),
    }
}

fn handle_ws_input_resolve(
    id: &str,
    params: serde_json::Value,
    bridge: &Arc<InteractionBridge>,
) -> WsFrame {
    let parsed: types::InputResolveParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return WsFrame::err(id, format!("invalid params: {e}")),
    };
    match bridge.resolve_input(parsed.interaction_id, parsed.message) {
        Ok(()) => WsFrame::ok(id, serde_json::json!({})),
        Err(e) => WsFrame::err(id, e.to_string()),
    }
}

fn handle_ws_question_resolve(
    id: &str,
    params: serde_json::Value,
    bridge: &Arc<InteractionBridge>,
) -> WsFrame {
    let parsed: types::QuestionResolveParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return WsFrame::err(id, format!("invalid params: {e}")),
    };
    let response = heartbit::QuestionResponse {
        answers: parsed.answers,
    };
    match bridge.resolve_question(parsed.interaction_id, response) {
        Ok(()) => WsFrame::ok(id, serde_json::json!({})),
        Err(e) => WsFrame::err(id, e.to_string()),
    }
}

/// Convert an outbound bridge message to a WS frame.
fn outbound_to_frame(msg: OutboundMessage, seq: &std::sync::atomic::AtomicU64) -> WsFrame {
    match msg {
        // RawFrame is already a complete WsFrame (e.g., method responses) — pass through.
        OutboundMessage::RawFrame(frame) => frame,
        // All event variants get a monotonic sequence number.
        other => {
            let next_seq = seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            match other {
                OutboundMessage::TextDelta { session_id, text } => {
                    let payload = types::ChatDeltaPayload { session_id, text };
                    WsFrame::push(
                        types::event::CHAT_DELTA,
                        serde_json::to_value(payload).unwrap_or_default(),
                        next_seq,
                    )
                }
                OutboundMessage::AgentEvent { session_id, event } => {
                    let payload = types::AgentEventPayload {
                        session_id,
                        event: serde_json::to_value(event).unwrap_or_default(),
                    };
                    WsFrame::push(
                        types::event::AGENT_EVENT,
                        serde_json::to_value(payload).unwrap_or_default(),
                        next_seq,
                    )
                }
                OutboundMessage::InputNeeded {
                    session_id,
                    interaction_id,
                } => {
                    let payload = types::InteractionNeededPayload {
                        session_id,
                        interaction_id,
                        data: serde_json::Value::Null,
                    };
                    WsFrame::push(
                        types::event::INPUT_NEEDED,
                        serde_json::to_value(payload).unwrap_or_default(),
                        next_seq,
                    )
                }
                OutboundMessage::ApprovalNeeded {
                    session_id,
                    interaction_id,
                    tool_calls,
                } => {
                    let payload = types::InteractionNeededPayload {
                        session_id,
                        interaction_id,
                        data: tool_calls,
                    };
                    WsFrame::push(
                        types::event::APPROVAL_NEEDED,
                        serde_json::to_value(payload).unwrap_or_default(),
                        next_seq,
                    )
                }
                OutboundMessage::QuestionNeeded {
                    session_id,
                    interaction_id,
                    request,
                } => {
                    let payload = types::InteractionNeededPayload {
                        session_id,
                        interaction_id,
                        data: serde_json::to_value(request).unwrap_or_default(),
                    };
                    WsFrame::push(
                        types::event::QUESTION_NEEDED,
                        serde_json::to_value(payload).unwrap_or_default(),
                        next_seq,
                    )
                }
                OutboundMessage::ChatFinal { session_id, result } => {
                    let payload = types::ChatFinalPayload { session_id, result };
                    WsFrame::push(
                        types::event::CHAT_FINAL,
                        serde_json::to_value(payload).unwrap_or_default(),
                        next_seq,
                    )
                }
                OutboundMessage::ChatError { session_id, error } => {
                    let payload = types::ChatErrorPayload { session_id, error };
                    WsFrame::push(
                        types::event::CHAT_ERROR,
                        serde_json::to_value(payload).unwrap_or_default(),
                        next_seq,
                    )
                }
                OutboundMessage::RawFrame(_) => unreachable!(),
            }
        }
    }
}

/// Parameters for an interactive WS task.
struct InteractiveTaskParams {
    on_text: Arc<heartbit::OnText>,
    on_approval: Arc<heartbit::OnApproval>,
    on_event: Arc<heartbit::OnEvent>,
    on_question: Arc<heartbit::OnQuestion>,
    mode: ObservabilityMode,
    cancel: CancellationToken,
}

/// Run an interactive task using `build_orchestrator_from_config` with bridge callbacks.
#[allow(clippy::too_many_arguments)]
async fn run_interactive_task(
    config: &HeartbitConfig,
    task: &str,
    params: InteractiveTaskParams,
    metrics: Option<&DaemonMetrics>,
    external_memory: Option<Arc<dyn Memory>>,
    workspace_dir: Option<PathBuf>,
    daemon_todo_store: Option<Arc<heartbit::FileTodoStore>>,
    pre_loaded_tools: Option<&HashMap<String, Vec<Arc<dyn heartbit::tool::Tool>>>>,
    source: &str,
) -> std::result::Result<AgentOutput, HeartbitError> {
    let on_retry = build_on_retry(&params.on_event);
    let provider = build_provider_from_config(config, Some(on_retry))
        .map_err(|e| HeartbitError::Daemon(e.to_string()))?;

    if let Some(m) = metrics {
        m.tasks_active().inc();
    }
    let start = Instant::now();

    let result = tokio::select! {
        res = crate::build_orchestrator_from_config(
            provider,
            config,
            task,
            params.on_text,
            Some(params.on_approval),
            Some(params.on_event),
            params.mode,
            None, // no story_id for interactive sessions
            Some(params.on_question),
            external_memory,
            workspace_dir,
            daemon_todo_store,
            pre_loaded_tools,
        ) => {
            res.map_err(|e| HeartbitError::Daemon(e.to_string()))
        }
        _ = params.cancel.cancelled() => {
            Err(HeartbitError::Daemon("task cancelled".into()))
        }
    };

    let duration_secs = start.elapsed().as_secs_f64();
    if let Some(m) = metrics {
        m.tasks_active().dec();
        m.record_task_by_source(source);
        match &result {
            Ok(_) => m.record_task_completed(duration_secs),
            Err(_) => m.record_task_failed(duration_secs),
        }
    }

    result
}
