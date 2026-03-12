//! Cloud-to-runtime execution endpoint.
//!
//! Accepts a `RuntimeRequest` with full agent configuration, builds
//! an agent runner on the fly, and executes the task. This enables
//! heartbit-cloud to delegate agent execution with per-tenant
//! provider keys, MCP installations, and guardrails.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use regex::RegexBuilder;

use heartbit::{
    AgentEvent, AgentRunner, AnthropicProvider, BoxedProvider, BuiltinToolsConfig, DagAgent,
    DebateAgent, Guardrail, GuardrailMode, InjectionClassifierGuardrail, LoopAgent, McpClient,
    MixtureOfAgentsAgent, NamespacedMemory, OpenRouterProvider, Orchestrator, ParallelAgent,
    PiiGuardrail, PostgresMemoryStore, RetryingProvider, RuntimeProviderType, RuntimeRequest,
    RuntimeResponse, RuntimeSseEvent, SequentialAgent, SubAgentConfig, VotingAgent, WorkflowType,
    config::SpawnConfig,
};

use super::types::AppState;

/// Resolve a per-tenant workspace directory from the daemon's base workspace.
///
/// Returns `None` if no base workspace is configured. When a tenant_id is present,
/// scopes to `{base}/{tenant_id}/` to prevent cross-tenant file access.
async fn resolve_task_workspace(
    base: Option<&std::path::Path>,
    tenant_id: Option<&uuid::Uuid>,
) -> Option<std::path::PathBuf> {
    let base = base?;
    match tenant_id {
        Some(tid) => {
            let scoped = base.join(tid.to_string());
            // create_dir_all is safe — tenant_id is a UUID (no path traversal)
            if let Err(e) = tokio::fs::create_dir_all(&scoped).await {
                tracing::debug!(path = %scoped.display(), "workspace dir creation failed: {e}");
            }
            Some(scoped)
        }
        None => Some(base.to_path_buf()),
    }
}

/// Build a memory namespace string from tenant ID and agent name.
fn memory_namespace(tenant_id: Option<&uuid::Uuid>, agent_name: &str) -> String {
    match tenant_id {
        Some(tid) => format!("{tid}:{agent_name}"),
        None => agent_name.to_string(),
    }
}

/// Resolve a memory store from shared memory, database pool, or in-memory fallback.
///
/// Priority: shared_memory (with embeddings) > PostgresMemoryStore > InMemoryStore.
fn resolve_memory(
    shared_memory: &Option<Arc<dyn heartbit::Memory>>,
    db_pool: Option<&sqlx::PgPool>,
    tenant_id: Option<&uuid::Uuid>,
    agent_name: &str,
) -> Arc<dyn heartbit::Memory> {
    let ns = memory_namespace(tenant_id, agent_name);
    if let Some(sm) = shared_memory {
        tracing::debug!(namespace = %ns, "using shared memory store (with embeddings)");
        Arc::new(NamespacedMemory::new(sm.clone(), ns))
    } else if let Some(pool) = db_pool {
        let pg_mem = PostgresMemoryStore::new(pool.clone());
        tracing::debug!(namespace = %ns, "using PostgresMemoryStore (no embedding wrapper)");
        Arc::new(NamespacedMemory::new(Arc::new(pg_mem), ns))
    } else {
        Arc::new(heartbit::InMemoryStore::new())
    }
}

/// Handle a cloud-delegated execution request.
///
/// Builds an ephemeral agent runner from the request's config and executes the task.
/// Returns JSON for sync requests, SSE for streaming requests.
pub(crate) async fn handle_execute(
    State(state): State<AppState>,
    Json(req): Json<RuntimeRequest>,
) -> impl IntoResponse {
    if req.stream {
        handle_execute_stream(state, req).await.into_response()
    } else {
        handle_execute_sync(state, req).await.into_response()
    }
}

/// Run an agent with either a text prompt or pre-built content blocks.
/// Choosing between the two paths at a single site avoids repeating the
/// if/else dispatch in every handler that builds a standalone runner.
async fn run_agent(
    runner: AgentRunner<BoxedProvider>,
    prompt: &str,
    initial_content: Vec<heartbit::llm::types::ContentBlock>,
) -> Result<heartbit::AgentOutput, String> {
    if initial_content.is_empty() {
        runner.execute(prompt).await.map_err(|e| e.to_string())
    } else {
        runner
            .execute_with_content(initial_content)
            .await
            .map_err(|e| e.to_string())
    }
}

async fn handle_execute_sync(state: AppState, req: RuntimeRequest) -> impl IntoResponse {
    // Collect events during execution
    let events = Arc::new(std::sync::Mutex::new(Vec::<serde_json::Value>::new()));
    let events_clone = events.clone();
    let on_event: Arc<heartbit::OnEvent> = Arc::new(move |event: AgentEvent| {
        if let Ok(data) = serde_json::to_value(&event)
            && let Ok(mut evts) = events_clone.lock()
        {
            evts.push(data);
        }
    });

    let task_ws =
        resolve_task_workspace(state.workspace_dir.as_deref(), req.tenant_id.as_ref()).await;
    let task_id = req.task_id;

    let result = if req.workflow.is_some() {
        execute_workflow_inner(req, None, Some(on_event), task_ws).await
    } else if !req.sub_agents.is_empty() {
        execute_orchestrator_inner(
            req,
            None,
            Some(on_event),
            state.shared_memory.clone(),
            state.db_pool.as_ref(),
            task_ws,
        )
        .await
    } else {
        match build_runner_from_request(
            &req,
            None,
            state.shared_memory.clone(),
            state.db_pool.as_ref(),
            Some(on_event),
            task_ws,
        )
        .await
        {
            Ok(runner) => run_agent(runner, &req.prompt, req.initial_content).await,
            Err(e) => Err(e),
        }
    };

    match result {
        Ok(output) => {
            let collected_events =
                std::mem::take(&mut *events.lock().unwrap_or_else(|e| e.into_inner()));
            let resp = RuntimeResponse {
                result: output.result,
                usage: output.tokens_used,
                model_name: output.model_name,
                events: collected_events,
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            tracing::error!(task_id = %task_id, "execute failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

async fn handle_execute_stream(state: AppState, req: RuntimeRequest) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel::<RuntimeSseEvent>(256);

    let prompt = req.prompt.clone();
    let task_id = req.task_id;
    let is_multi_agent = !req.sub_agents.is_empty();
    let shared_memory = state.shared_memory.clone();
    let db_pool = state.db_pool.clone();
    let task_ws =
        resolve_task_workspace(state.workspace_dir.as_deref(), req.tenant_id.as_ref()).await;

    tokio::spawn(async move {
        // Wire on_text callback to send Delta events during execution
        let delta_tx = tx.clone();
        let on_text: Arc<heartbit::OnText> = Arc::new(move |text: &str| {
            let _ = delta_tx.try_send(RuntimeSseEvent::Delta {
                content: text.to_owned(),
            });
        });

        // Wire on_event callback to send Event SSE events during execution
        let event_tx = tx.clone();
        let on_event: Arc<heartbit::OnEvent> = Arc::new(move |event: AgentEvent| {
            let name = event.type_name().to_string();
            if let Ok(data) = serde_json::to_value(&event) {
                let _ = event_tx.try_send(RuntimeSseEvent::Event { name, data });
            }
        });

        let result = if req.workflow.is_some() {
            execute_workflow_inner(req, Some(on_text), Some(on_event), task_ws).await
        } else if is_multi_agent {
            execute_orchestrator_inner(
                req,
                Some(on_text),
                Some(on_event),
                shared_memory,
                db_pool.as_ref(),
                task_ws,
            )
            .await
        } else {
            match build_runner_from_request(
                &req,
                Some(on_text),
                shared_memory,
                db_pool.as_ref(),
                Some(on_event),
                task_ws,
            )
            .await
            {
                Ok(runner) => run_agent(runner, &prompt, req.initial_content).await,
                Err(e) => Err(e),
            }
        };

        match result {
            Ok(output) => {
                let resp = RuntimeResponse {
                    result: output.result,
                    usage: output.tokens_used,
                    model_name: output.model_name,
                    events: vec![],
                };
                let _ = tx.send(RuntimeSseEvent::Done(resp)).await;
            }
            Err(e) => {
                tracing::error!(task_id = %task_id, "stream execute failed: {e}");
                let _ = tx.send(RuntimeSseEvent::Error { message: e }).await;
            }
        }
    });

    let stream: ReceiverStream<RuntimeSseEvent> = ReceiverStream::new(rx);
    let sse_stream = futures::StreamExt::map(stream, |event| {
        let data = serde_json::to_string(&event).unwrap_or_default();
        Ok::<_, Infallible>(Event::default().data(data))
    });

    Sse::new(sse_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Build a retrying provider from runtime config.
pub(super) fn build_provider(config: &heartbit::RuntimeProviderConfig) -> Arc<BoxedProvider> {
    let base: BoxedProvider = match config.provider_type {
        RuntimeProviderType::Anthropic => {
            let p = if config.prompt_caching {
                AnthropicProvider::with_prompt_caching(&config.api_key, &config.model)
            } else {
                AnthropicProvider::new(&config.api_key, &config.model)
            };
            BoxedProvider::new(p)
        }
        RuntimeProviderType::Openrouter => {
            BoxedProvider::new(OpenRouterProvider::new(&config.api_key, &config.model))
        }
    };
    Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(base)))
}

/// Build guardrails from runtime config.
pub(super) fn build_guardrails(gc: &heartbit::RuntimeGuardrailConfig) -> Vec<Arc<dyn Guardrail>> {
    let mut guardrails: Vec<Arc<dyn Guardrail>> = Vec::new();
    if gc.injection {
        guardrails.push(Arc::new(InjectionClassifierGuardrail::new(
            gc.injection_threshold,
            GuardrailMode::Deny,
        )));
    }
    if gc.pii {
        guardrails.push(Arc::new(PiiGuardrail::all_builtin(gc.pii_action)));
    }
    guardrails
}

/// Build an `AgentRunner` from a `RuntimeRequest`.
///
/// Creates an ephemeral provider + agent from the request's configuration,
/// connects to MCP servers, wires guardrails, and returns a ready-to-execute runner.
///
/// When `db_pool` is provided and memory is enabled with a tenant ID,
/// uses `PostgresMemoryStore` for persistent cross-session memory.
pub(super) async fn build_runner_from_request(
    req: &RuntimeRequest,
    on_text: Option<Arc<heartbit::OnText>>,
    shared_memory: Option<Arc<dyn heartbit::Memory>>,
    db_pool: Option<&sqlx::PgPool>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<std::path::PathBuf>,
) -> Result<AgentRunner<BoxedProvider>, String> {
    let provider = build_provider(&req.provider);

    let mut builder = AgentRunner::builder(provider)
        .name(&req.agent.name)
        .max_turns(req.agent.max_turns)
        .max_tokens(req.agent.max_tokens);

    if let Some(ref prompt) = req.agent.system_prompt {
        builder = builder.system_prompt(prompt);
    }

    if let Some(callback) = on_text {
        builder = builder.on_text(callback);
    }

    if let (Some(user_id), Some(tenant_id)) = (&req.user_id, &req.tenant_id) {
        builder = builder.audit_user_context(user_id.as_str(), tenant_id.to_string());
    }

    // Apply advanced config
    let adv = &req.agent.advanced;
    if let Some(ref effort) = adv.reasoning_effort
        && let Ok(parsed) = heartbit::parse_reasoning_effort(effort)
    {
        builder = builder.reasoning_effort(parsed);
    }
    if let Some(ref profile) = adv.tool_profile
        && let Ok(parsed) = heartbit::parse_tool_profile(profile)
    {
        builder = builder.tool_profile(parsed);
    }
    if let Some(threshold) = adv.summarize_threshold {
        builder = builder.summarize_threshold(threshold);
    }
    if let Some(max) = adv.max_identical_tool_calls {
        builder = builder.max_identical_tool_calls(max);
    }
    if let Some(max_tools) = adv.max_tools_per_turn {
        builder = builder.max_tools_per_turn(max_tools);
    }

    // Wire memory if enabled.
    // Prefer shared_memory (already wrapped with EmbeddingMemory at daemon startup)
    // so cloud-submitted tasks get the same embedding-capable store as daemon agents.
    // Fall back to bare PostgresMemoryStore (BM25-only) if no shared memory but db_pool exists.
    // Last resort: ephemeral InMemoryStore.
    if let Some(ref mc) = req.memory
        && mc.enabled
    {
        let mem = resolve_memory(
            &shared_memory,
            db_pool,
            req.tenant_id.as_ref(),
            &req.agent.name,
        );
        builder = builder.memory(mem);
        if let Some(threshold) = mc.reflection_threshold {
            builder = builder.reflection_threshold(threshold);
        }
        builder = builder.consolidate_on_exit(mc.consolidate_on_exit);
    }

    // Collect tools (MCP + builtins) with workspace isolation
    let tools = collect_tools(&req.mcp_servers, &req.builtin_tools, workspace.as_deref()).await;
    if !tools.is_empty() {
        builder = builder.tools(tools);
    }

    // Wire on_event callback for structured event emission
    if let Some(on_event) = on_event {
        builder = builder.on_event(on_event);
    }

    // Apply guardrails
    if let Some(ref gc) = req.guardrails {
        let guardrails = build_guardrails(gc);
        if !guardrails.is_empty() {
            builder = builder.guardrails(guardrails);
        }
    }

    builder.build().map_err(|e| e.to_string())
}

/// Connect to MCP servers (in parallel) and collect builtin tools.
///
/// When `workspace` is provided, builtin tools (especially bash) are jailed
/// to that directory with a restrictive environment variable allowlist.
pub(super) async fn collect_tools(
    mcp_servers: &[heartbit::RuntimeMcpServer],
    builtin_tool_names: &[String],
    workspace: Option<&std::path::Path>,
) -> Vec<Arc<dyn heartbit::tool::Tool>> {
    let mut tools: Vec<Arc<dyn heartbit::tool::Tool>> = Vec::new();

    // Connect to MCP servers concurrently
    let mut join_set = tokio::task::JoinSet::new();
    for server in mcp_servers {
        let url = server.url.clone();
        let auth = server.auth_header.clone();
        join_set.spawn(async move {
            let result = match auth.as_deref() {
                Some(a) => McpClient::connect_with_auth(&url, a).await,
                None => McpClient::connect(&url).await,
            };
            (url, result)
        });
    }
    while let Some(Ok((url, result))) = join_set.join_next().await {
        match result {
            Ok(client) => tools.extend(client.into_tools()),
            Err(e) => {
                tracing::warn!(url = %url, error = %e, "failed to connect MCP server");
            }
        }
    }

    // Add builtin tools (filtered to requested names)
    if !builtin_tool_names.is_empty() {
        let env_policy = if workspace.is_some() {
            // Use restrictive allowlist in runtime mode to prevent env var leaks
            heartbit::workspace::EnvPolicy::Allowlist(
                heartbit::workspace::DAEMON_ENV_ALLOWLIST
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
            )
        } else {
            heartbit::workspace::EnvPolicy::Inherit
        };
        let bt_config = BuiltinToolsConfig {
            dangerous_tools: builtin_tool_names.iter().any(|t| t == "bash"),
            workspace: workspace.map(|p| p.to_path_buf()),
            env_policy,
            ..Default::default()
        };
        let all_builtins = heartbit::builtin_tools(bt_config);
        let requested: std::collections::HashSet<&str> =
            builtin_tool_names.iter().map(|s| s.as_str()).collect();
        for tool in all_builtins {
            if requested.contains(tool.definition().name.as_str()) {
                tools.push(tool);
            }
        }
    }

    tools
}

/// Shared orchestrator execution logic for both sync and stream paths.
async fn execute_orchestrator_inner(
    req: RuntimeRequest,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    shared_memory: Option<Arc<dyn heartbit::Memory>>,
    db_pool: Option<&sqlx::PgPool>,
    workspace: Option<std::path::PathBuf>,
) -> Result<heartbit::AgentOutput, String> {
    let provider = build_provider(&req.provider);

    const DEFAULT_ORCH_MAX_TURNS: usize = 10;
    const DEFAULT_MAX_TOKENS: u32 = 4096;
    let orch_config = req.orchestrator.as_ref();
    let max_turns = orch_config.map_or(DEFAULT_ORCH_MAX_TURNS, |c| c.max_turns);
    let max_tokens = orch_config.map_or(DEFAULT_MAX_TOKENS, |c| c.max_tokens);

    let mut builder = Orchestrator::builder(provider)
        .max_turns(max_turns)
        .max_tokens(max_tokens);

    if let Some(callback) = on_text {
        builder = builder.on_text(callback);
    }

    if let Some(callback) = on_event {
        builder = builder.on_event(callback);
    }

    // Wire memory into the orchestrator so it (and sub-agents) get memory tools
    if let Some(ref mc) = req.memory
        && mc.enabled
    {
        let mem = resolve_memory(
            &shared_memory,
            db_pool,
            req.tenant_id.as_ref(),
            &req.agent.name,
        );
        builder = builder.shared_memory(mem);
    }

    if let Some(ref oc) = req.orchestrator {
        if oc.enable_squads {
            builder = builder.enable_squads(true);
        }
        if let Some(mode) = oc.dispatch_mode {
            builder = builder.dispatch_mode(mode);
        }
        if let Some(ref spawn) = oc.spawn {
            let spawn_cfg = SpawnConfig {
                max_spawned_agents: spawn.max_spawned_agents,
                tool_allowlist: spawn.tool_allowlist.clone(),
                max_turns: spawn.max_turns,
                max_tokens: spawn.max_tokens,
                max_total_tokens: spawn.max_total_tokens,
            };
            let spawn_bt_config = BuiltinToolsConfig {
                workspace: workspace.clone(),
                ..Default::default()
            };
            let builtin_tools = heartbit::builtin_tools(spawn_bt_config);
            builder = builder.spawn_config(spawn_cfg, builtin_tools);
        }
    }

    if let (Some(user_id), Some(tenant_id)) = (&req.user_id, &req.tenant_id) {
        builder = builder.audit_user_context(user_id.as_str(), tenant_id.to_string());
    }

    if let Some(ref gc) = req.guardrails {
        let guardrails = build_guardrails(gc);
        if !guardrails.is_empty() {
            builder = builder.guardrails(guardrails);
        }
    }

    // Register sub-agents (with workspace isolation matching the parent)
    for sub in &req.sub_agents {
        let tools = collect_tools(&sub.mcp_servers, &sub.builtin_tools, workspace.as_deref()).await;

        let config = SubAgentConfig {
            name: sub.name.clone(),
            description: sub.description.clone(),
            system_prompt: sub.system_prompt.clone(),
            tools,
            max_turns: Some(sub.max_turns),
            max_tokens: Some(sub.max_tokens),
            ..Default::default()
        };

        builder = builder.sub_agent_full(config);
    }

    // 5. Execute
    let mut orchestrator = builder.build().map_err(|e| e.to_string())?;
    orchestrator
        .run(&req.prompt)
        .await
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Workflow execution
// ---------------------------------------------------------------------------

/// Maximum number of nodes allowed in a workflow definition.
/// Must match heartbit-cloud routes/workflows.rs MAX_WORKFLOW_NODES so that
/// a workflow created via the cloud API (validated at 100) never fails at runtime.
const MAX_WORKFLOW_NODES: usize = 100;
/// Maximum number of edges allowed in a workflow definition.
/// Must match heartbit-cloud routes/workflows.rs MAX_WORKFLOW_EDGES.
const MAX_WORKFLOW_EDGES: usize = 500;
/// Maximum byte length for workflow node names and sub-agent names.
/// Must match heartbit-cloud routes/workflows.rs MAX_NODE_NAME_BYTES.
const MAX_NAME_BYTES: usize = 100;

/// Shared workflow execution logic for both sync and stream paths.
async fn execute_workflow_inner(
    req: RuntimeRequest,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<std::path::PathBuf>,
) -> Result<heartbit::AgentOutput, String> {
    let wf = req
        .workflow
        .as_ref()
        .ok_or_else(|| "workflow config missing".to_string())?;

    if wf.nodes.is_empty() {
        return Err("workflow must have at least one node".into());
    }
    if wf.nodes.len() > MAX_WORKFLOW_NODES {
        return Err(format!(
            "workflow has {} nodes, exceeding maximum of {}",
            wf.nodes.len(),
            MAX_WORKFLOW_NODES
        ));
    }
    if wf.edges.len() > MAX_WORKFLOW_EDGES {
        return Err(format!(
            "workflow has {} edges, exceeding maximum of {}",
            wf.edges.len(),
            MAX_WORKFLOW_EDGES
        ));
    }

    // Validate node names and agent names before building the agent map.
    // heartbit-cloud validates full format; the daemon adds a lighter guard for
    // the case where it is called directly (bypassing the cloud API).
    for node in &wf.nodes {
        contains_unsafe_chars(&node.name).map_err(|e| format!("workflow node name: {e}"))?;
        contains_unsafe_chars(&node.agent_name)
            .map_err(|e| format!("workflow node '{}' agent_name: {e}", node.name))?;
    }
    let provider = build_provider(&req.provider);

    // Validate sub-agent names and build the lookup map in a single pass.
    let mut agent_map: std::collections::HashMap<&str, &heartbit::RuntimeSubAgentConfig> =
        std::collections::HashMap::with_capacity(req.sub_agents.len());
    for sa in &req.sub_agents {
        contains_unsafe_chars(&sa.name).map_err(|e| format!("sub-agent name: {e}"))?;
        agent_map.insert(sa.name.as_str(), sa);
    }

    let ws = workspace.as_deref();
    match wf.workflow_type {
        WorkflowType::Dag => execute_dag(&req, &agent_map, provider, on_text, on_event, ws).await,
        WorkflowType::Sequential => {
            execute_sequential(&req, &agent_map, provider, on_text, on_event, ws).await
        }
        WorkflowType::Parallel => {
            execute_parallel(&req, &agent_map, provider, on_text, on_event, ws).await
        }
        WorkflowType::Loop => execute_loop(&req, &agent_map, provider, on_text, on_event, ws).await,
        WorkflowType::Debate => {
            execute_debate(&req, &agent_map, provider, on_text, on_event, ws).await
        }
        WorkflowType::Voting => {
            execute_voting(&req, &agent_map, provider, on_text, on_event, ws).await
        }
        WorkflowType::Mixture => {
            execute_mixture(&req, &agent_map, provider, on_text, on_event, ws).await
        }
    }
}

type AgentMap<'a> = std::collections::HashMap<&'a str, &'a heartbit::RuntimeSubAgentConfig>;
type EdgeConditionFn = Box<dyn Fn(&str) -> bool + Send + Sync>;
type EdgeTransformFn = Box<dyn Fn(&str) -> String + Send + Sync>;

/// Build all per-node AgentRunners concurrently, returning them in node order.
///
/// Resolving nodes is synchronous (HashMap lookup); the async work is tool setup
/// (MCP server connections, builtin tool wiring).  Running all builds in parallel
/// reduces startup latency from O(N × max_build_time) to O(max_build_time).
async fn build_all_agents(
    nodes: &[heartbit::RuntimeWorkflowNode],
    agent_map: &AgentMap<'_>,
    provider: Arc<BoxedProvider>,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<&std::path::Path>,
) -> Result<Vec<(String, AgentRunner<BoxedProvider>)>, String> {
    // Resolve all nodes synchronously first (cheap HashMap lookups).
    // Clone each sub-agent config so the async closures can own their data.
    let resolved: Vec<(
        String,
        heartbit::RuntimeSubAgentConfig,
        Option<Arc<heartbit::OnEvent>>,
    )> = nodes
        .iter()
        .map(|node| {
            let sub = resolve_node(node, agent_map)?.clone();
            let node_event = wrap_node_event(node.name.clone(), on_event.clone());
            Ok((node.name.clone(), sub, node_event))
        })
        .collect::<Result<_, String>>()?;

    // Build the workspace path once as an owned `PathBuf` so async closures can use it.
    let ws_buf = workspace.map(|p| p.to_path_buf());

    let futs = resolved.into_iter().map(|(name, sub, node_event)| {
        let provider = provider.clone();
        let on_text = on_text.clone();
        let ws = ws_buf.clone();
        async move {
            let runner =
                build_workflow_agent(&sub, provider, on_text, node_event, ws.as_deref()).await?;
            Ok::<_, String>((name, runner))
        }
    });

    futures::future::try_join_all(futs).await
}

/// Build a single AgentRunner from a sub-agent config.
async fn build_workflow_agent(
    sub: &heartbit::RuntimeSubAgentConfig,
    provider: Arc<BoxedProvider>,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<&std::path::Path>,
) -> Result<AgentRunner<BoxedProvider>, String> {
    let tools = collect_tools(&sub.mcp_servers, &sub.builtin_tools, workspace).await;

    let mut builder = AgentRunner::builder(provider)
        .name(&sub.name)
        .system_prompt(&sub.system_prompt)
        .max_turns(sub.max_turns)
        .max_tokens(sub.max_tokens);

    if !tools.is_empty() {
        builder = builder.tools(tools);
    }
    if let Some(cb) = on_text {
        builder = builder.on_text(cb);
    }
    if let Some(cb) = on_event {
        builder = builder.on_event(cb);
    }

    builder.build().map_err(|e| e.to_string())
}

/// Wrap an `on_event` callback to emit WorkflowNodeStarted/Completed events
/// when the underlying agent's RunStarted/RunCompleted events fire.
fn wrap_node_event(
    node_name: String,
    on_event: Option<Arc<heartbit::OnEvent>>,
) -> Option<Arc<heartbit::OnEvent>> {
    on_event.map(|cb| {
        Arc::new(move |event: heartbit::AgentEvent| {
            match &event {
                heartbit::AgentEvent::RunStarted { .. } => {
                    cb(heartbit::AgentEvent::WorkflowNodeStarted {
                        node: node_name.clone(),
                    });
                }
                heartbit::AgentEvent::RunCompleted { .. } => {
                    cb(heartbit::AgentEvent::WorkflowNodeCompleted {
                        node: node_name.clone(),
                    });
                }
                heartbit::AgentEvent::RunFailed { .. } => {
                    cb(heartbit::AgentEvent::WorkflowNodeFailed {
                        node: node_name.clone(),
                    });
                }
                _ => {}
            }
            cb(event);
        }) as Arc<heartbit::OnEvent>
    })
}

/// Return an error if `s` contains ASCII control characters or Unicode format
/// characters (e.g. zero-width spaces, bidirectional overrides) that could be
/// used for log injection or terminal escape attacks.
///
/// Length is checked first (O(1)) so oversized inputs are rejected before the
/// O(n) Unicode scan.  The limit matches heartbit-cloud's `MAX_NODE_NAME_BYTES`.
fn contains_unsafe_chars(s: &str) -> Result<(), String> {
    if s.len() > MAX_NAME_BYTES {
        return Err(format!(
            "name too long ({} bytes, max {MAX_NAME_BYTES})",
            s.len()
        ));
    }
    if s.chars().any(|c| {
        c.is_control()
            || matches!(c,
                // Zero-width and joining characters
                '\u{200B}'..='\u{200F}' |
                // Bidirectional override and embedding
                '\u{202A}'..='\u{202E}' |
                // Additional directional/invisible formatting
                '\u{2060}'..='\u{2069}' |
                // Byte order mark
                '\u{FEFF}'
            )
    }) {
        return Err("contains control or unsafe Unicode format characters".into());
    }
    Ok(())
}

/// Resolve a workflow node to its sub-agent config.
fn resolve_node<'a>(
    node: &heartbit::RuntimeWorkflowNode,
    agent_map: &AgentMap<'a>,
) -> Result<&'a heartbit::RuntimeSubAgentConfig, String> {
    agent_map
        .get(node.agent_name.as_str())
        .copied()
        .ok_or_else(|| {
            format!(
                "workflow node '{}' references unknown agent '{}'",
                node.name, node.agent_name
            )
        })
}

/// Build a `Regex` with bounded NFA/DFA sizes to prevent regex-based DoS.
fn build_bounded_regex(pattern: &str) -> Result<regex::Regex, regex::Error> {
    RegexBuilder::new(pattern)
        .size_limit(1_000_000)
        .dfa_size_limit(10_000_000)
        .build()
}

/// Build an edge condition closure from a spec.
fn build_edge_condition(spec: &heartbit::EdgeConditionSpec) -> Result<EdgeConditionFn, String> {
    use heartbit::EdgeConditionPattern;
    let value = spec.value.clone();
    match spec.pattern {
        EdgeConditionPattern::Contains => Ok(Box::new(move |s: &str| s.contains(&value))),
        EdgeConditionPattern::NotContains => Ok(Box::new(move |s: &str| !s.contains(&value))),
        EdgeConditionPattern::StartsWith => Ok(Box::new(move |s: &str| s.starts_with(&value))),
        EdgeConditionPattern::Regex => {
            let re = build_bounded_regex(&value)
                .map_err(|e| format!("invalid regex '{}': {}", value, e))?;
            Ok(Box::new(move |s: &str| re.is_match(s)))
        }
    }
}

/// Build a named edge transform.
fn build_edge_transform(transform: &heartbit::EdgeTransform) -> EdgeTransformFn {
    match transform {
        heartbit::EdgeTransform::Uppercase => Box::new(|s: &str| s.to_uppercase()),
        heartbit::EdgeTransform::Lowercase => Box::new(|s: &str| s.to_lowercase()),
        heartbit::EdgeTransform::ExtractJson => Box::new(|s: &str| {
            // Try to parse the first valid JSON object or array from the text
            for delim_open in ['{', '['] {
                if let Some(start) = s.find(delim_open) {
                    let mut iter = serde_json::Deserializer::from_str(&s[start..])
                        .into_iter::<serde_json::Value>();
                    if let Some(Ok(value)) = iter.next() {
                        return value.to_string();
                    }
                }
            }
            s.to_string()
        }),
        heartbit::EdgeTransform::Trim => Box::new(|s: &str| s.trim().to_string()),
    }
}

async fn execute_dag(
    req: &RuntimeRequest,
    agent_map: &AgentMap<'_>,
    provider: Arc<BoxedProvider>,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<&std::path::Path>,
) -> Result<heartbit::AgentOutput, String> {
    let wf = req.workflow.as_ref().unwrap();
    let mut builder = DagAgent::<BoxedProvider>::builder();

    let agents =
        build_all_agents(&wf.nodes, agent_map, provider, on_text, on_event, workspace).await?;
    for (name, runner) in agents {
        builder = builder.node(&name, runner);
    }

    for edge in &wf.edges {
        if let Some(ref cond) = edge.condition {
            let condition = build_edge_condition(cond)?;
            builder = builder.conditional_edge(&edge.from, &edge.to, condition);
        } else if let Some(ref t) = edge.transform {
            let transform = build_edge_transform(t);
            builder = builder.edge_with_transform(&edge.from, &edge.to, transform);
        } else {
            builder = builder.edge(&edge.from, &edge.to);
        }
    }

    let dag = builder.build().map_err(|e| e.to_string())?;
    dag.execute(&req.prompt).await.map_err(|e| e.to_string())
}

async fn execute_sequential(
    req: &RuntimeRequest,
    agent_map: &AgentMap<'_>,
    provider: Arc<BoxedProvider>,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<&std::path::Path>,
) -> Result<heartbit::AgentOutput, String> {
    let wf = req.workflow.as_ref().unwrap();
    let mut builder = SequentialAgent::<BoxedProvider>::builder();

    // Build agents in parallel (setup cost), then add in node order (execution is sequential).
    let agents =
        build_all_agents(&wf.nodes, agent_map, provider, on_text, on_event, workspace).await?;
    for (_, runner) in agents {
        builder = builder.agent(runner);
    }

    let agent = builder.build().map_err(|e| e.to_string())?;
    agent.execute(&req.prompt).await.map_err(|e| e.to_string())
}

async fn execute_parallel(
    req: &RuntimeRequest,
    agent_map: &AgentMap<'_>,
    provider: Arc<BoxedProvider>,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<&std::path::Path>,
) -> Result<heartbit::AgentOutput, String> {
    let wf = req.workflow.as_ref().unwrap();
    let mut builder = ParallelAgent::<BoxedProvider>::builder();

    let agents =
        build_all_agents(&wf.nodes, agent_map, provider, on_text, on_event, workspace).await?;
    for (_, runner) in agents {
        builder = builder.agent(runner);
    }

    let agent = builder.build().map_err(|e| e.to_string())?;
    agent.execute(&req.prompt).await.map_err(|e| e.to_string())
}

async fn execute_loop(
    req: &RuntimeRequest,
    agent_map: &AgentMap<'_>,
    provider: Arc<BoxedProvider>,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<&std::path::Path>,
) -> Result<heartbit::AgentOutput, String> {
    let wf = req.workflow.as_ref().unwrap();
    let sub = resolve_node(&wf.nodes[0], agent_map)?;
    let node_event = wrap_node_event(wf.nodes[0].name.clone(), on_event);
    let runner = build_workflow_agent(sub, provider, on_text, node_event, workspace).await?;

    let mut builder = LoopAgent::builder().agent(runner);

    if let Some(max) = wf.max_iterations {
        builder = builder.max_iterations(max as usize);
    } else {
        // LoopAgent requires max_iterations — default to 10 if omitted
        builder = builder.max_iterations(10);
    }

    if let Some(ref pattern) = wf.stop_pattern {
        let re = build_bounded_regex(pattern)
            .map_err(|e| format!("invalid stop_pattern regex '{}': {}", pattern, e))?;
        builder = builder.should_stop(move |text: &str| re.is_match(text));
    } else {
        // LoopAgent requires should_stop — default to never-stop (rely on max_iterations)
        builder = builder.should_stop(|_: &str| false);
    }

    let agent = builder.build().map_err(|e| e.to_string())?;
    agent.execute(&req.prompt).await.map_err(|e| e.to_string())
}

async fn execute_debate(
    req: &RuntimeRequest,
    agent_map: &AgentMap<'_>,
    provider: Arc<BoxedProvider>,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<&std::path::Path>,
) -> Result<heartbit::AgentOutput, String> {
    let wf = req.workflow.as_ref().unwrap();
    let mut builder = DebateAgent::<BoxedProvider>::builder();

    let agents =
        build_all_agents(&wf.nodes, agent_map, provider, on_text, on_event, workspace).await?;
    // Zip agents back with nodes to recover role assignments (try_join_all preserves order).
    for ((_, runner), node) in agents.into_iter().zip(&wf.nodes) {
        let role = node.role.as_deref().unwrap_or("debater");
        match role {
            "judge" => builder = builder.judge(runner),
            _ => builder = builder.debater(runner),
        }
    }

    let rounds = wf.rounds.unwrap_or(2) as usize;
    builder = builder.max_rounds(rounds);

    let agent = builder.build().map_err(|e| e.to_string())?;
    agent.execute(&req.prompt).await.map_err(|e| e.to_string())
}

async fn execute_voting(
    req: &RuntimeRequest,
    agent_map: &AgentMap<'_>,
    provider: Arc<BoxedProvider>,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<&std::path::Path>,
) -> Result<heartbit::AgentOutput, String> {
    let wf = req.workflow.as_ref().unwrap();
    let mut builder = VotingAgent::<BoxedProvider>::builder();

    let agents =
        build_all_agents(&wf.nodes, agent_map, provider, on_text, on_event, workspace).await?;
    for (_, runner) in agents {
        builder = builder.voter(runner);
    }

    // VotingAgent requires a vote_extractor — default: trim entire output as the vote
    builder = builder.vote_extractor(|text: &str| text.trim().to_string());

    let agent = builder.build().map_err(|e| e.to_string())?;
    let vote_result = agent
        .execute(&req.prompt)
        .await
        .map_err(|e| e.to_string())?;
    Ok(vote_result.output)
}

async fn execute_mixture(
    req: &RuntimeRequest,
    agent_map: &AgentMap<'_>,
    provider: Arc<BoxedProvider>,
    on_text: Option<Arc<heartbit::OnText>>,
    on_event: Option<Arc<heartbit::OnEvent>>,
    workspace: Option<&std::path::Path>,
) -> Result<heartbit::AgentOutput, String> {
    let wf = req.workflow.as_ref().unwrap();
    let mut builder = MixtureOfAgentsAgent::<BoxedProvider>::builder();

    let agents =
        build_all_agents(&wf.nodes, agent_map, provider, on_text, on_event, workspace).await?;
    for ((_, runner), node) in agents.into_iter().zip(&wf.nodes) {
        let role = node.role.as_deref().unwrap_or("proposer");
        match role {
            "synthesizer" => builder = builder.synthesizer(runner),
            _ => builder = builder.proposer(runner),
        }
    }

    if let Some(layers) = wf.layers {
        builder = builder.layers(layers as usize);
    }

    let agent = builder.build().map_err(|e| e.to_string())?;
    agent.execute(&req.prompt).await.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit::{
        RuntimeAdvancedConfig, RuntimeAgentConfig, RuntimeGuardrailConfig, RuntimeMemoryConfig,
        RuntimeProviderConfig, RuntimeProviderType,
    };

    fn make_test_request() -> RuntimeRequest {
        RuntimeRequest {
            task_id: uuid::Uuid::new_v4(),
            prompt: "test".into(),
            stream: false,
            tenant_id: None,
            user_id: None,
            memory: None,
            agent: RuntimeAgentConfig {
                name: "test-agent".into(),
                system_prompt: Some("You are a test agent.".into()),
                max_turns: 5,
                max_tokens: 1024,
                advanced: RuntimeAdvancedConfig::default(),
            },
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Anthropic,
                api_key: "sk-test".into(),
                model: "claude-sonnet-4-20250514".into(),
                prompt_caching: false,
            },
            mcp_servers: vec![],
            builtin_tools: vec![],
            guardrails: None,
            messages: vec![],
            session_id: None,
            sub_agents: vec![],
            orchestrator: None,
            workflow: None,
            initial_content: vec![],
        }
    }

    #[tokio::test]
    async fn build_runner_from_request_anthropic() {
        let req = make_test_request();
        let result = build_runner_from_request(&req, None, None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_from_request_openrouter() {
        let mut req = make_test_request();
        req.provider.provider_type = RuntimeProviderType::Openrouter;
        let result = build_runner_from_request(&req, None, None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_guardrails() {
        let mut req = make_test_request();
        req.guardrails = Some(RuntimeGuardrailConfig {
            injection: true,
            pii: true,
            pii_action: Default::default(),
            injection_threshold: 0.5,
        });
        let result = build_runner_from_request(&req, None, None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_builtins() {
        let mut req = make_test_request();
        req.builtin_tools = vec!["todo_read".into(), "todo_write".into()];
        let result = build_runner_from_request(&req, None, None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_advanced_config() {
        let mut req = make_test_request();
        req.agent.advanced.summarize_threshold = Some(20);
        req.agent.advanced.max_identical_tool_calls = Some(3);
        req.agent.advanced.reasoning_effort = Some("high".into());
        let result = build_runner_from_request(&req, None, None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_prompt_caching() {
        let mut req = make_test_request();
        req.provider.prompt_caching = true;
        let result = build_runner_from_request(&req, None, None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_on_text() {
        let req = make_test_request();
        let on_text: Arc<heartbit::OnText> = Arc::new(|_: &str| {});
        let result = build_runner_from_request(&req, Some(on_text), None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_memory() {
        let mut req = make_test_request();
        req.memory = Some(RuntimeMemoryConfig {
            enabled: true,
            reflection_threshold: Some(50),
            consolidate_on_exit: true,
        });
        let result = build_runner_from_request(&req, None, None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_audit_context() {
        let mut req = make_test_request();
        req.user_id = Some("user-123".into());
        req.tenant_id = Some(uuid::Uuid::new_v4());
        let result = build_runner_from_request(&req, None, None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_on_event() {
        let req = make_test_request();
        let events = Arc::new(std::sync::Mutex::new(Vec::<AgentEvent>::new()));
        let events_clone = events.clone();
        let on_event: Arc<heartbit::OnEvent> = Arc::new(move |event: AgentEvent| {
            events_clone.lock().unwrap().push(event);
        });
        let result = build_runner_from_request(&req, None, None, None, Some(on_event), None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_memory_falls_back_to_in_memory_without_pool() {
        let mut req = make_test_request();
        req.memory = Some(RuntimeMemoryConfig {
            enabled: true,
            reflection_threshold: None,
            consolidate_on_exit: false,
        });
        req.tenant_id = Some(uuid::Uuid::new_v4());
        // No db_pool and no shared_memory → should fall back to InMemoryStore
        let result = build_runner_from_request(&req, None, None, None, None, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_uses_shared_memory_when_provided() {
        let mut req = make_test_request();
        req.memory = Some(RuntimeMemoryConfig {
            enabled: true,
            reflection_threshold: None,
            consolidate_on_exit: false,
        });
        req.tenant_id = Some(uuid::Uuid::new_v4());
        // Provide shared memory — should be used instead of creating bare PostgresMemoryStore
        let shared: Arc<dyn heartbit::Memory> = Arc::new(heartbit::InMemoryStore::new());
        let result = build_runner_from_request(&req, None, Some(shared), None, None, None).await;
        assert!(result.is_ok());
    }

    // --- Workflow dispatch tests ---

    #[test]
    fn build_edge_condition_contains() {
        let spec = heartbit::EdgeConditionSpec {
            pattern: heartbit::EdgeConditionPattern::Contains,
            value: "success".into(),
        };
        let cond = super::build_edge_condition(&spec).unwrap();
        assert!(cond("the operation was a success"));
        assert!(!cond("it failed"));
    }

    #[test]
    fn build_edge_condition_not_contains() {
        let spec = heartbit::EdgeConditionSpec {
            pattern: heartbit::EdgeConditionPattern::NotContains,
            value: "error".into(),
        };
        let cond = super::build_edge_condition(&spec).unwrap();
        assert!(cond("all good"));
        assert!(!cond("got an error"));
    }

    #[test]
    fn build_edge_condition_starts_with() {
        let spec = heartbit::EdgeConditionSpec {
            pattern: heartbit::EdgeConditionPattern::StartsWith,
            value: "OK".into(),
        };
        let cond = super::build_edge_condition(&spec).unwrap();
        assert!(cond("OK done"));
        assert!(!cond("Not OK"));
    }

    #[test]
    fn build_edge_condition_regex() {
        let spec = heartbit::EdgeConditionSpec {
            pattern: heartbit::EdgeConditionPattern::Regex,
            value: r"\d{3}".into(),
        };
        let cond = super::build_edge_condition(&spec).unwrap();
        assert!(cond("code 200"));
        assert!(!cond("no numbers"));
    }

    #[test]
    fn build_edge_condition_invalid_regex() {
        let spec = heartbit::EdgeConditionSpec {
            pattern: heartbit::EdgeConditionPattern::Regex,
            value: "[invalid".into(),
        };
        assert!(super::build_edge_condition(&spec).is_err());
    }

    #[test]
    fn build_edge_transform_uppercase() {
        let t = super::build_edge_transform(&heartbit::EdgeTransform::Uppercase);
        assert_eq!(t("hello"), "HELLO");
    }

    #[test]
    fn build_edge_transform_lowercase() {
        let t = super::build_edge_transform(&heartbit::EdgeTransform::Lowercase);
        assert_eq!(t("HELLO"), "hello");
    }

    #[test]
    fn build_edge_transform_extract_json() {
        let t = super::build_edge_transform(&heartbit::EdgeTransform::ExtractJson);
        assert_eq!(t(r#"result: {"key": "val"} end"#), r#"{"key":"val"}"#);
    }

    #[test]
    fn build_edge_transform_trim() {
        let t = super::build_edge_transform(&heartbit::EdgeTransform::Trim);
        assert_eq!(t("  hello  "), "hello");
    }

    #[test]
    fn resolve_node_missing_agent() {
        let node = heartbit::RuntimeWorkflowNode {
            name: "a".into(),
            agent_name: "nonexistent".into(),
            role: None,
        };
        let map: super::AgentMap = std::collections::HashMap::new();
        assert!(super::resolve_node(&node, &map).is_err());
    }

    #[tokio::test]
    async fn workflow_exceeding_max_nodes_rejected() {
        let nodes: Vec<heartbit::RuntimeWorkflowNode> = (0..=super::MAX_WORKFLOW_NODES)
            .map(|i| heartbit::RuntimeWorkflowNode {
                name: format!("n{i}"),
                agent_name: "a".into(),
                role: None,
            })
            .collect();
        let mut req = make_test_request();
        req.workflow = Some(heartbit::RuntimeWorkflowConfig {
            workflow_type: heartbit::WorkflowType::Sequential,
            nodes,
            edges: vec![],
            max_iterations: None,
            stop_pattern: None,
            rounds: None,
            layers: None,
        });
        let result = super::execute_workflow_inner(req, None, None, None).await;
        let err = result.unwrap_err();
        assert!(err.contains("exceeding maximum"), "got: {err}");
    }

    #[tokio::test]
    async fn workflow_node_name_with_control_char_rejected() {
        let mut req = make_test_request();
        req.workflow = Some(heartbit::RuntimeWorkflowConfig {
            workflow_type: heartbit::WorkflowType::Sequential,
            nodes: vec![heartbit::RuntimeWorkflowNode {
                name: "bad\x0aname".into(), // newline — log injection
                agent_name: "valid-agent".into(),
                role: None,
            }],
            edges: vec![],
            max_iterations: None,
            stop_pattern: None,
            rounds: None,
            layers: None,
        });
        let err = super::execute_workflow_inner(req, None, None, None)
            .await
            .unwrap_err();
        assert!(err.contains("control or unsafe Unicode"), "got: {err}");
    }

    #[tokio::test]
    async fn workflow_sub_agent_name_with_control_char_rejected() {
        let mut req = make_test_request();
        req.sub_agents = vec![heartbit::RuntimeSubAgentConfig {
            name: "bad\x1bname".into(), // ESC — ANSI injection
            description: String::new(),
            system_prompt: String::new(),
            max_turns: 10,
            max_tokens: 1024,
            builtin_tools: vec![],
            mcp_servers: vec![],
        }];
        req.workflow = Some(heartbit::RuntimeWorkflowConfig {
            workflow_type: heartbit::WorkflowType::Sequential,
            nodes: vec![heartbit::RuntimeWorkflowNode {
                name: "n1".into(),
                agent_name: "bad\x1bname".into(),
                role: None,
            }],
            edges: vec![],
            max_iterations: None,
            stop_pattern: None,
            rounds: None,
            layers: None,
        });
        let err = super::execute_workflow_inner(req, None, None, None)
            .await
            .unwrap_err();
        assert!(err.contains("control or unsafe Unicode"), "got: {err}");
    }

    #[tokio::test]
    async fn workflow_exceeding_max_edges_rejected() {
        let edges: Vec<heartbit::RuntimeWorkflowEdge> = (0..=super::MAX_WORKFLOW_EDGES)
            .map(|i| heartbit::RuntimeWorkflowEdge {
                from: format!("n{i}"),
                to: "n0".into(),
                condition: None,
                transform: None,
            })
            .collect();
        let mut req = make_test_request();
        req.workflow = Some(heartbit::RuntimeWorkflowConfig {
            workflow_type: heartbit::WorkflowType::Sequential,
            nodes: vec![heartbit::RuntimeWorkflowNode {
                name: "n0".into(),
                agent_name: "a".into(),
                role: None,
            }],
            edges,
            max_iterations: None,
            stop_pattern: None,
            rounds: None,
            layers: None,
        });
        let result = super::execute_workflow_inner(req, None, None, None).await;
        let err = result.unwrap_err();
        assert!(err.contains("exceeding maximum"), "got: {err}");
    }

    #[tokio::test]
    async fn build_workflow_agent_basic() {
        let sub = heartbit::RuntimeSubAgentConfig {
            name: "test".into(),
            description: "test".into(),
            system_prompt: "You help.".into(),
            max_turns: 5,
            max_tokens: 1024,
            builtin_tools: vec![],
            mcp_servers: vec![],
        };
        let provider = super::build_provider(&heartbit::RuntimeProviderConfig {
            provider_type: heartbit::RuntimeProviderType::Anthropic,
            api_key: "sk-test".into(),
            model: "claude-sonnet-4-20250514".into(),
            prompt_caching: false,
        });
        let result = super::build_workflow_agent(&sub, provider, None, None, None).await;
        assert!(result.is_ok());
    }
}
