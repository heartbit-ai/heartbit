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

use heartbit::{
    AgentRunner, AnthropicProvider, BoxedProvider, BuiltinToolsConfig, Guardrail, GuardrailMode,
    InjectionClassifierGuardrail, McpClient, OpenRouterProvider, Orchestrator, PiiGuardrail,
    RetryingProvider, RuntimeProviderType, RuntimeRequest, RuntimeResponse, RuntimeSseEvent,
    SubAgentConfig,
};

use super::types::AppState;

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

async fn handle_execute_sync(_state: AppState, req: RuntimeRequest) -> impl IntoResponse {
    if !req.sub_agents.is_empty() {
        return execute_orchestrator_sync(req).await.into_response();
    }

    let runner = match build_runner_from_request(&req, None).await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e })),
            )
                .into_response();
        }
    };

    match runner.execute(&req.prompt).await {
        Ok(output) => {
            let resp = RuntimeResponse {
                result: output.result,
                usage: output.tokens_used,
                model_name: output.model_name,
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            tracing::error!(task_id = %req.task_id, "execute failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    }
}

async fn handle_execute_stream(_state: AppState, req: RuntimeRequest) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel::<RuntimeSseEvent>(256);

    let prompt = req.prompt.clone();
    let task_id = req.task_id;
    let is_multi_agent = !req.sub_agents.is_empty();

    tokio::spawn(async move {
        // Wire on_text callback to send Delta events during execution
        let delta_tx = tx.clone();
        let on_text: Arc<heartbit::OnText> = Arc::new(move |text: &str| {
            let _ = delta_tx.try_send(RuntimeSseEvent::Delta {
                content: text.to_owned(),
            });
        });

        let result = if is_multi_agent {
            execute_orchestrator_inner(req, Some(on_text)).await
        } else {
            match build_runner_from_request(&req, Some(on_text)).await {
                Ok(runner) => runner.execute(&prompt).await.map_err(|e| e.to_string()),
                Err(e) => Err(e),
            }
        };

        match result {
            Ok(output) => {
                let resp = RuntimeResponse {
                    result: output.result,
                    usage: output.tokens_used,
                    model_name: output.model_name,
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
fn build_provider(config: &heartbit::RuntimeProviderConfig) -> Arc<BoxedProvider> {
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
fn build_guardrails(gc: &heartbit::RuntimeGuardrailConfig) -> Vec<Arc<dyn Guardrail>> {
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
async fn build_runner_from_request(
    req: &RuntimeRequest,
    on_text: Option<Arc<heartbit::OnText>>,
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

    // Wire memory if enabled
    if let Some(ref mc) = req.memory
        && mc.enabled
    {
        let mem: Arc<dyn heartbit::Memory> = Arc::new(heartbit::InMemoryStore::new());
        builder = builder.memory(mem);
        if let Some(threshold) = mc.reflection_threshold {
            builder = builder.reflection_threshold(threshold);
        }
        builder = builder.consolidate_on_exit(mc.consolidate_on_exit);
    }

    // Collect tools (MCP + builtins)
    let tools = collect_tools(&req.mcp_servers, &req.builtin_tools).await;
    if !tools.is_empty() {
        builder = builder.tools(tools);
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

/// Connect to MCP servers and collect tools for a sub-agent.
/// Connect to MCP servers (in parallel) and collect builtin tools.
async fn collect_tools(
    mcp_servers: &[heartbit::RuntimeMcpServer],
    builtin_tool_names: &[String],
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
        let bt_config = BuiltinToolsConfig {
            dangerous_tools: builtin_tool_names.iter().any(|t| t == "bash"),
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

/// Build and execute an Orchestrator from a multi-agent RuntimeRequest (sync path).
async fn execute_orchestrator_sync(req: RuntimeRequest) -> impl IntoResponse {
    let task_id = req.task_id;
    match execute_orchestrator_inner(req, None).await {
        Ok(output) => {
            let resp = RuntimeResponse {
                result: output.result,
                usage: output.tokens_used,
                model_name: output.model_name,
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            tracing::error!(task_id = %task_id, "orchestrator execute failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e })),
            )
                .into_response()
        }
    }
}

/// Shared orchestrator execution logic for both sync and stream paths.
async fn execute_orchestrator_inner(
    req: RuntimeRequest,
    on_text: Option<Arc<heartbit::OnText>>,
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

    if let Some(ref oc) = req.orchestrator {
        if oc.enable_squads {
            builder = builder.enable_squads(true);
        }
        if let Some(mode) = oc.dispatch_mode {
            builder = builder.dispatch_mode(mode);
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

    // Register sub-agents
    for sub in &req.sub_agents {
        let tools = collect_tools(&sub.mcp_servers, &sub.builtin_tools).await;

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
        }
    }

    #[tokio::test]
    async fn build_runner_from_request_anthropic() {
        let req = make_test_request();
        let result = build_runner_from_request(&req, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_from_request_openrouter() {
        let mut req = make_test_request();
        req.provider.provider_type = RuntimeProviderType::Openrouter;
        let result = build_runner_from_request(&req, None).await;
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
        let result = build_runner_from_request(&req, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_builtins() {
        let mut req = make_test_request();
        req.builtin_tools = vec!["todo_read".into(), "todo_write".into()];
        let result = build_runner_from_request(&req, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_advanced_config() {
        let mut req = make_test_request();
        req.agent.advanced.summarize_threshold = Some(20);
        req.agent.advanced.max_identical_tool_calls = Some(3);
        req.agent.advanced.reasoning_effort = Some("high".into());
        let result = build_runner_from_request(&req, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_prompt_caching() {
        let mut req = make_test_request();
        req.provider.prompt_caching = true;
        let result = build_runner_from_request(&req, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_on_text() {
        let req = make_test_request();
        let on_text: Arc<heartbit::OnText> = Arc::new(|_: &str| {});
        let result = build_runner_from_request(&req, Some(on_text)).await;
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
        let result = build_runner_from_request(&req, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn build_runner_with_audit_context() {
        let mut req = make_test_request();
        req.user_id = Some("user-123".into());
        req.tenant_id = Some(uuid::Uuid::new_v4());
        let result = build_runner_from_request(&req, None).await;
        assert!(result.is_ok());
    }
}
