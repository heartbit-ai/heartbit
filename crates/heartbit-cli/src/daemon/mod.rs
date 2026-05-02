mod auth;
mod eval;
mod execute;
mod handlers;
mod memory;
mod types;

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::Router;
use axum::middleware;
use axum::routing::{get, post};
use tokio_util::sync::CancellationToken;

use heartbit::daemon::kafka;
use heartbit::{
    AgentEvent, AgentOutput, DaemonCore, DaemonMetrics, Error as HeartbitError, HeartbitConfig,
    InMemoryTaskStore, JwtValidator, Memory, PostgresStore,
};

use crate::{build_on_retry, build_provider_from_config, init_tracing_from_config};

use self::auth::{JwtMiddlewareState, auth_middleware, jwt_auth_middleware, resolve_auth_tokens};
use self::handlers::{
    HttpMetrics, cors_middleware, handle_approval, handle_cancel, handle_get, handle_healthz,
    handle_list, handle_metrics, handle_readyz, handle_stats, handle_stream, handle_submit,
    handle_usage, http_metrics_middleware, mcp_tools_for_user, validate_path_component,
};
use self::memory::build_institutional_entry;
use self::types::{AppState, PendingApprovals};

// --- Daemon startup ---

pub async fn run_daemon(
    config_path: &std::path::Path,
    bind_override: Option<&str>,
    verbose: bool,
    observability_flag: Option<&str>,
) -> Result<()> {
    let mut config = HeartbitConfig::from_file(config_path)
        .with_context(|| format!("failed to load config from {}", config_path.display()))?;

    // Resolve agent templates, skills, and variables.
    let variables = config.variables.clone();
    for i in 0..config.agents.len() {
        if config.agents[i].template.is_some() || !config.agents[i].skills.is_empty() {
            let resolved = heartbit::resolve_agent_config(&config.agents[i], &variables)
                .with_context(|| {
                    format!(
                        "failed to resolve template for agent '{}'",
                        config.agents[i].name
                    )
                })?;
            config.agents[i] = resolved;
        }
    }

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

    // Create a shared PgPool when a database URL is configured.
    // Reused by both the task store and the execute handler (for persistent memory).
    let db_pool: Option<sqlx::PgPool> = if let Some(ref db_url) = daemon_config.database_url {
        let pool = sqlx::PgPool::connect(db_url)
            .await
            .context("failed to connect to database")?;
        Some(pool)
    } else {
        None
    };

    // Build task store: PostgreSQL if configured, in-memory otherwise
    let store: Arc<dyn heartbit::TaskStore> = if let Some(ref pool) = db_pool {
        let pg_store = heartbit::PostgresTaskStore::new(pool.clone());
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

    // Run memory table migration when database is available.
    // This is done once at startup so the execute handler can create
    // PostgresMemoryStore instances without per-request migration overhead.
    if let Some(ref pool) = db_pool {
        let mut pg_mem = heartbit::PostgresMemoryStore::new(pool.clone());
        pg_mem
            .run_migration()
            .await
            .context("failed to run memory migration")?;
        tracing::info!("memory store: PostgreSQL migration complete");
    }

    // Create cancellation token
    let cancel = CancellationToken::new();

    // Keep a reference for background tasks that may need the store after it's moved.
    let store_for_tasks = store.clone();

    // Create DaemonCore + DaemonHandle — Kafka is optional
    let (core, handle) = if let Some(ref kafka_config) = daemon_config.kafka {
        tracing::info!("ensuring Kafka topics exist");
        kafka::ensure_topics(kafka_config)
            .await
            .context("failed to ensure Kafka topics")?;

        let producer =
            kafka::create_producer(kafka_config).context("failed to create Kafka producer")?;
        let consumer = kafka::create_commands_consumer(kafka_config)
            .context("failed to create Kafka consumer")?;

        let (core, handle) =
            DaemonCore::new(&daemon_config, consumer, producer, store, cancel.clone());
        // Attach Postgres store for audit retention (used by HEARTBIT_AUDIT_RETAIN_DAYS).
        let core = if let Some(ref pool) = db_pool {
            core.with_postgres_store(std::sync::Arc::new(PostgresStore::new(pool.clone())))
        } else {
            core
        };
        tracing::info!("Kafka consumer/producer initialized");
        (Some(core), handle)
    } else {
        tracing::info!("no [daemon.kafka] configured — running in HTTP-only mode");
        let handle = heartbit::DaemonHandle::http_only(store);
        (None, handle)
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

    // Pre-load MCP + A2A tools once for all agents (daemon reuses across tasks).
    // Also populate the transport pool for per-user auth stamping.
    let transport_pool = Arc::new(heartbit::McpTransportPool::new());
    let tool_cache: Arc<HashMap<String, Vec<Arc<dyn heartbit::tool::Tool>>>> = {
        let mut cache = HashMap::new();
        for agent in &config.agents {
            let mut agent_tools = crate::load_mcp_tools(&agent.name, &agent.mcp_servers).await;
            agent_tools.extend(crate::load_a2a_tools(&agent.name, &agent.a2a_agents).await);
            // Warm the transport pool for HTTP MCP servers (enables per-user auth stamping)
            for entry in &agent.mcp_servers {
                if !entry.is_stdio()
                    && !transport_pool.contains(entry.url())
                    && let Err(e) = transport_pool
                        .get_or_connect(entry.url(), entry.auth_header().map(String::from))
                        .await
                {
                    tracing::warn!(
                        agent = %agent.name,
                        server = %entry.display_name(),
                        error = %e,
                        "failed to warm transport pool (tools still loaded via McpClient)"
                    );
                }
            }
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

    // Signal handler
    heartbit::signal::spawn_shutdown_handler(cancel.clone());

    // Resolve observability mode
    let config_obs = config
        .telemetry
        .as_ref()
        .and_then(|t| t.observability_mode.as_deref());
    let mode = crate::resolve_observability(observability_flag, config_obs, verbose);

    // Shared map for subject tokens: "{tenant_id}:{user_id}" -> raw JWT.
    // Populated by HTTP submit handler, consumed by TokenExchangeAuthProvider.
    let user_tokens: Arc<std::sync::RwLock<HashMap<String, String>>> =
        Arc::new(std::sync::RwLock::new(HashMap::new()));

    // Create per-user auth provider from token_exchange config (dynamic MCP auth)
    let auth_provider: Option<Arc<dyn heartbit::AuthProvider>> = config
        .daemon
        .as_ref()
        .and_then(|d| d.auth.as_ref())
        .and_then(|a| a.token_exchange.as_ref())
        .map(|te| {
            tracing::info!(
                exchange_url = %te.exchange_url,
                client_id = %te.client_id,
                "token exchange auth provider configured for per-user MCP delegation"
            );
            Arc::new(
                heartbit::TokenExchangeAuthProvider::new(
                    &te.exchange_url,
                    &te.client_id,
                    &te.client_secret,
                    &te.agent_token,
                )
                .with_tenant_id(te.tenant_id.clone())
                .with_scopes(te.scopes.clone())
                .with_user_tokens(user_tokens.clone()),
            ) as Arc<dyn heartbit::AuthProvider>
        });

    // Build the runner closure that creates an Orchestrator per task
    let config_arc = Arc::new(config);
    let config_for_state = config_arc.clone();
    let runner_metrics = metrics.clone();
    let runner_memory = shared_memory.clone();
    let runner_workspace = daemon_workspace_dir.clone();
    let runner_tools = tool_cache.clone();
    let runner_auth_provider = auth_provider;
    let state_auth_provider = runner_auth_provider.clone();
    let runner_transport_pool = transport_pool.clone();
    let runner_user_tokens = user_tokens.clone();
    // Shared pending approvals map: REST endpoint writes decisions, on_approval callback reads them.
    let pending_approvals: PendingApprovals = Arc::new(std::sync::Mutex::new(HashMap::new()));
    let runner_pending_approvals = pending_approvals.clone();
    let build_runner = move |task_id: uuid::Uuid,
                             task_text: String,
                             source: String,
                             story_id: Option<String>,
                             trust_level: Option<heartbit::TrustLevel>,
                             on_event_fn: Arc<dyn Fn(AgentEvent) + Send + Sync>,
                             user_id: Option<String>,
                             tenant_id: Option<String>,
                             user_roles: Vec<String>,
                             mcp_auth_tokens: Option<HashMap<String, String>>|
          -> Pin<
        Box<dyn std::future::Future<Output = Result<AgentOutput, HeartbitError>> + Send>,
    > {
        let config = config_arc.clone();
        let task_metrics = runner_metrics.clone();
        let memory = runner_memory.clone();
        let workspace_dir = runner_workspace.clone();
        let tools = runner_tools.clone();
        let task_auth_provider = runner_auth_provider.clone();
        let task_transport_pool = runner_transport_pool.clone();
        let task_user_tokens = runner_user_tokens.clone();
        let task_pending_approvals = runner_pending_approvals.clone();
        Box::pin(async move {
            // Wrap on_event to also record metrics
            let on_event: Arc<heartbit::OnEvent> = if let Some(ref m) = task_metrics {
                let inner = on_event_fn;
                let metrics = m.clone();
                let tenant_label = tenant_id.clone();
                Arc::new(move |event: AgentEvent| {
                    metrics.record_event(&event, tenant_label.as_deref());
                    inner(event);
                })
            } else {
                on_event_fn
            };

            let on_retry = build_on_retry(&on_event);
            let provider = build_provider_from_config(&config, Some(on_retry.clone()))
                .map_err(|e| HeartbitError::Daemon(e.to_string()))?;

            // Record submission and track active tasks (after provider creation to avoid gauge leak on error)
            if let Some(ref m) = task_metrics {
                m.record_task_submitted(tenant_id.as_deref(), &source);
                m.tasks_active().inc();
            }
            let start = Instant::now();

            let on_text: Arc<heartbit::OnText> = Arc::new(|_: &str| {});

            // Build on_approval callback for REST/SSE tasks: blocks on mpsc channel,
            // resolved by POST /v1/tasks/{id}/approve endpoint.
            let approval_task_id = task_id;
            let approval_map = task_pending_approvals;
            let on_approval: Arc<heartbit::OnApproval> = Arc::new(move |_tool_calls| {
                let (tx, rx) = std::sync::mpsc::channel();
                {
                    let mut pending = approval_map.lock().expect("pending_approvals lock");
                    pending.insert(approval_task_id, tx);
                }
                // block_in_place tells tokio this thread is about to block, so it can
                // compensate by spawning additional worker threads.
                tokio::task::block_in_place(|| match rx.recv_timeout(Duration::from_secs(300)) {
                    Ok(decision) => decision,
                    Err(_) => {
                        let mut pending = approval_map.lock().expect("pending_approvals lock");
                        pending.remove(&approval_task_id);
                        heartbit::ApprovalDecision::Deny
                    }
                })
            });

            // Wire SensorSecurityGuardrail for sensor-sourced tasks
            let (mut guardrails, memory_confidentiality_cap): (
                Vec<Arc<dyn heartbit::Guardrail>>,
                Option<heartbit::Confidentiality>,
            ) = if source.starts_with("sensor:") {
                let trust = trust_level.unwrap_or(heartbit::TrustLevel::Unknown);
                let guardrail =
                    heartbit::SensorSecurityGuardrail::new(source.clone(), trust, vec![]);
                // Memory confidentiality cap based on trust level
                let cap = match trust {
                    heartbit::TrustLevel::Owner | heartbit::TrustLevel::Verified => None,
                    _ => Some(heartbit::Confidentiality::Public),
                };
                (vec![Arc::new(guardrail)], cap)
            } else {
                (vec![], None)
            };

            // Append config-based guardrails (injection, PII, tool policy, LLM judge)
            if let Some(ref gc) = config.guardrails {
                match gc.build() {
                    Ok(config_guardrails) => guardrails.extend(config_guardrails),
                    Err(e) => {
                        tracing::error!(error = %e, "failed to build config guardrails for daemon task, skipping");
                    }
                }
            }

            // Per-user memory namespace: wrap shared memory with user-scoped prefix
            let task_memory: Option<Arc<dyn heartbit::Memory>> = if let Some(ref uid) = user_id
                && let Some(ref tid) = tenant_id
                && let Some(ref mem) = memory
            {
                let ns_prefix = format!("tenant:{tid}:user:{uid}");
                let ns = heartbit::NamespacedMemory::new(mem.clone(), ns_prefix.clone());
                tracing::debug!(namespace = %ns_prefix, "memory namespaced to tenant/user");
                Some(Arc::new(ns))
            } else {
                memory
            };

            // Per-user workspace isolation: scope to {workspace}/{tenant_id}/{user_id}/
            // Sanitize IDs to prevent path traversal (reject /, \, .., absolute paths)
            let task_workspace = if let (Some(base), Some(tid), Some(uid)) =
                (&workspace_dir, &tenant_id, &user_id)
            {
                if let Err(e) =
                    validate_path_component(tid).and_then(|_| validate_path_component(uid))
                {
                    tracing::error!(
                        tenant_id = %tid,
                        user_id = %uid,
                        error = %e,
                        "rejected unsafe tenant/user ID in workspace path"
                    );
                    return Err(HeartbitError::Daemon(format!(
                        "invalid tenant/user ID for workspace: {e}"
                    )));
                }
                let scoped = base.join(tid).join(uid);
                if let Err(e) = tokio::fs::create_dir_all(&scoped).await {
                    tracing::warn!(
                        path = %scoped.display(),
                        error = %e,
                        "failed to create per-user workspace"
                    );
                }
                tracing::debug!(
                    path = %scoped.display(),
                    "workspace scoped to tenant/user"
                );
                Some(scoped)
            } else {
                workspace_dir
            };

            // Dynamic MCP auth -- resolve per-user tools via transport pool.
            // When mcp_auth_tokens are provided (from cloud/gateway), use DirectAuthProvider.
            // Otherwise fall back to TokenExchangeAuthProvider.
            let effective_auth: Option<Arc<dyn heartbit::AuthProvider>> =
                if let Some(tokens) = mcp_auth_tokens {
                    Some(Arc::new(heartbit::DirectAuthProvider::new(tokens)))
                } else {
                    task_auth_provider
                };

            let user_tools: Option<HashMap<String, Vec<Arc<dyn heartbit::tool::Tool>>>> =
                if let (Some(ap), Some(uid), Some(tid)) = (&effective_auth, &user_id, &tenant_id) {
                    mcp_tools_for_user(&config, ap, uid, tid, &task_transport_pool).await
                } else {
                    None
                };

            // Clean up subject token after exchange -- prevent unbounded growth
            // of the shared user_tokens map across long-lived daemon sessions.
            if let (Some(uid), Some(tid)) = (&user_id, &tenant_id) {
                let key = format!("{tid}:{uid}");
                if let Ok(mut tokens) = task_user_tokens.write() {
                    tokens.remove(&key);
                }
            }

            let effective_tools = user_tools.as_ref().unwrap_or(&tools);

            // Compute role-gated shared memory write permission.
            let allow_shared_write = config
                .daemon
                .as_ref()
                .map(|d| &d.memory.shared_write_roles)
                .is_none_or(|roles| {
                    roles.is_empty() || user_roles.iter().any(|r| roles.contains(r))
                });

            // Check if any agent has dangerous_tools enabled.
            let dangerous_tools = config.agents.iter().any(|a| a.dangerous_tools);

            let result = crate::RuntimeBuilder::new(provider, &config, &task_text, on_text)
                .on_approval(Some(on_approval))
                .on_event(Some(on_event))
                .observability_mode(mode)
                .story_id(story_id.as_deref())
                .external_memory(task_memory)
                .workspace_dir(task_workspace)
                .pre_loaded_tools(Some(effective_tools))
                .guardrails(guardrails)
                .memory_confidentiality_cap(memory_confidentiality_cap)
                .audit_user_id(user_id.as_deref())
                .audit_tenant_id(tenant_id.as_deref())
                .allow_shared_write(allow_shared_write)
                .dangerous_tools(dangerous_tools)
                .run()
                .await
                .map_err(|e| HeartbitError::Daemon(e.to_string()));

            let duration_secs = start.elapsed().as_secs_f64();
            if let Some(ref m) = task_metrics {
                m.tasks_active().dec();
                m.record_task_by_source(&source);
                match &result {
                    Ok(_) => m.record_task_completed(duration_secs, tenant_id.as_deref()),
                    Err(_) => m.record_task_failed(duration_secs, tenant_id.as_deref()),
                }
            }

            result
        })
    };

    // Build Kafka brokers string for readiness check
    let kafka_brokers = daemon_config.kafka.as_ref().map(|k| k.brokers.clone());

    // Resolve auth tokens from config + env
    let auth_tokens = resolve_auth_tokens(
        daemon_config
            .auth
            .as_ref()
            .map(|a| a.bearer_tokens.as_slice())
            .unwrap_or_default(),
        std::env::var("HEARTBIT_API_KEY").ok(),
    );

    if auth_tokens.is_some() {
        tracing::info!("HTTP API authentication enabled (bearer token)");
    } else {
        tracing::warn!("HTTP API authentication disabled — all routes are public");
    }

    // Build JWT validator from config (for multi-tenant auth)
    let jwt_validator = daemon_config
        .auth
        .as_ref()
        .and_then(|auth| auth.jwks_url.as_ref())
        .map(|jwks_url| {
            let auth = daemon_config.auth.as_ref().unwrap();
            let mut validator =
                JwtValidator::new(jwks_url.clone(), auth.issuer.clone(), auth.audience.clone());
            if let Some(ref claim) = auth.user_id_claim {
                validator = validator.with_user_id_claim(claim.clone());
            }
            if let Some(ref claim) = auth.tenant_id_claim {
                validator = validator.with_tenant_id_claim(claim.clone());
            }
            if let Some(ref claim) = auth.roles_claim {
                validator = validator.with_roles_claim(claim.clone());
            }
            tracing::info!("JWT/JWKS authentication enabled ({})", jwks_url);
            Arc::new(validator)
        });

    // In HTTP-only mode, spawn the audit retention prune task here.
    // (DaemonCore::run handles the Kafka-mode case.)
    if core.is_none()
        && let Some(ref pool) = db_pool
    {
        let retain_days = daemon_config
            .audit
            .retain_days
            .or_else(|| {
                std::env::var("HEARTBIT_AUDIT_RETAIN_DAYS")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .filter(|&d| d > 0);
        let interval_minutes = daemon_config
            .audit
            .prune_interval_minutes
            .filter(|&m| m > 0)
            .unwrap_or(60);

        if let Some(days) = retain_days {
            let store = std::sync::Arc::new(PostgresStore::new(pool.clone()));
            let cancel_prune = cancel.clone();
            let retain = chrono::Duration::days(days as i64);
            let interval = std::time::Duration::from_secs(interval_minutes * 60);
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(interval);
                tick.tick().await;
                loop {
                    tokio::select! {
                        _ = cancel_prune.cancelled() => {
                            tracing::info!("audit prune task: cancellation received, exiting");
                            break;
                        }
                        _ = tick.tick() => {
                            match store.prune_audit(retain).await {
                                Ok(n) => tracing::info!(removed = n, "audit retention prune"),
                                Err(e) => tracing::warn!(error = %e, "audit prune failed"),
                            }
                        }
                    }
                }
            });
            tracing::info!(
                retain_days = days,
                interval_minutes = interval_minutes,
                "audit retention prune task started (HTTP-only mode)"
            );
        }
    }

    // B5b: idempotency-key TTL sweep task (HTTP-only mode).
    // In Kafka mode, DaemonCore::run handles this.
    if core.is_none()
        && let Some(ttl_hours) = daemon_config.idempotency.ttl_hours
    {
        let sweep_store = store_for_tasks.clone();
        let cancel_sweep = cancel.clone();
        let interval_min = daemon_config
            .idempotency
            .sweep_interval_minutes
            .unwrap_or(60);
        let interval = std::time::Duration::from_secs(u64::from(interval_min) * 60);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.tick().await; // skip immediate fire
            loop {
                tokio::select! {
                    _ = cancel_sweep.cancelled() => {
                        tracing::info!("idempotency sweep: cancellation received, exiting");
                        break;
                    }
                    _ = tick.tick() => {
                        let cutoff = chrono::Utc::now()
                            - chrono::Duration::hours(i64::from(ttl_hours));
                        match sweep_store.sweep_expired_idempotency_keys(cutoff) {
                            Ok(n) if n > 0 => {
                                tracing::info!(swept = n, "idempotency keys expired")
                            }
                            Ok(_) => {}
                            Err(e) => {
                                tracing::warn!(error = %e, "idempotency sweep failed")
                            }
                        }
                    }
                }
            }
        });
        tracing::info!(
            ttl_hours,
            interval_minutes = interval_min,
            "idempotency key TTL sweep task started (HTTP-only mode)"
        );
    }

    // Start HTTP server
    let app_state = AppState {
        handle,
        start_time: Instant::now(),
        metrics: metrics.clone(),
        cancel: cancel.clone(),
        kafka_brokers,
        config: config_for_state,
        observability_mode: mode,
        shared_memory: shared_memory.clone(),
        workspace_dir: daemon_workspace_dir,
        tool_cache: tool_cache.clone(),
        jwt_validator: jwt_validator.clone(),
        user_tokens: user_tokens.clone(),
        auth_provider: state_auth_provider,
        transport_pool: transport_pool.clone(),
        pending_approvals,
        db_pool,
    };

    // Public routes -- health, readiness, metrics -- never require auth
    let public_routes = Router::new()
        .route("/v1/health", get(handle_healthz))
        .route("/v1/ready", get(handle_readyz))
        .route("/v1/metrics", get(handle_metrics));

    // Protected routes -- tasks, SSE, stats
    let protected_routes = Router::new()
        .route("/v1/tasks", post(handle_submit))
        .route("/v1/tasks", get(handle_list))
        .route("/v1/tasks/{id}", get(handle_get))
        .route("/v1/tasks/{id}/cancel", post(handle_cancel))
        .route("/v1/tasks/{id}/stream", get(handle_stream))
        .route("/v1/tasks/{id}/approve", post(handle_approval))
        .route("/v1/tasks/execute", post(execute::handle_execute))
        .route("/v1/tasks/eval", post(eval::handle_eval))
        .route("/v1/stats", get(handle_stats))
        .route("/v1/usage", get(handle_usage));

    // Apply auth middleware layers.
    // JWT middleware (innermost): validates JWT, injects UserContext into extensions.
    // Bearer token middleware (outermost): gates access by static token.
    // Layers execute outer-to-inner, so bearer auth runs first, then JWT enrichment.
    let jwt_is_sole_auth = jwt_validator.is_some() && auth_tokens.is_none();
    let mut protected_routes = protected_routes;
    if let Some(ref validator) = jwt_validator {
        let jwt_state = JwtMiddlewareState {
            validator: validator.clone(),
            // When JWT is the only auth, it must reject unauthenticated requests.
            // When bearer tokens are also configured, JWT only enriches.
            required: jwt_is_sole_auth,
        };
        protected_routes = protected_routes.layer(middleware::from_fn_with_state(
            jwt_state,
            jwt_auth_middleware,
        ));
    }

    let routes = if let Some(ref tokens) = auth_tokens {
        let authed = protected_routes.layer(middleware::from_fn_with_state(
            tokens.clone(),
            auth_middleware,
        ));
        public_routes.merge(authed)
    } else {
        // Either JWT-only auth (required=true rejects unauthenticated) or no auth at all
        public_routes.merge(protected_routes)
    };

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
    tracing::info!(bind = %bind, "runtime HTTP server started");

    let http_cancel = cancel.clone();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                http_cancel.cancelled().await;
            })
            .await
            .ok();
    });

    // Build on_complete callback for institutional memory persistence
    let on_complete: Option<Arc<heartbit::OnTaskComplete>> = if shared_memory.is_some() {
        let persist_memory = shared_memory.clone();
        Some(Arc::new(move |outcome: heartbit::TaskOutcome| {
            // Auto-persist completed task results to institutional memory
            if outcome.state == heartbit::TaskState::Completed
                && let Some(ref result) = outcome.result_summary
                && !result.is_empty()
                && let Some(ref memory) = persist_memory
            {
                let memory = memory.clone();
                let entry = build_institutional_entry(
                    &outcome.id,
                    &outcome.source,
                    result,
                    outcome.story_id.as_deref(),
                    outcome.user_id.as_deref(),
                    outcome.tenant_id.as_deref(),
                );
                let scope =
                    heartbit::TenantScope::new(outcome.tenant_id.clone().unwrap_or_default());
                tokio::spawn(async move {
                    if let Err(e) = memory.store(&scope, entry).await {
                        tracing::warn!(
                            error = %e,
                            "failed to persist institutional memory"
                        );
                    }
                });
            }
        }))
    } else {
        None
    };

    // Run the DaemonCore consumer loop (blocks until cancellation), or
    // wait for shutdown in HTTP-only mode.
    if let Some(core) = core {
        tracing::info!("runtime core started, consuming from Kafka");
        core.run(build_runner, on_complete)
            .await
            .context("daemon core error")?;
    } else {
        tracing::info!("runtime started in HTTP-only mode (no Kafka consumer)");
        cancel.cancelled().await;
    }

    tracing::info!("runtime shut down gracefully");
    Ok(())
}
