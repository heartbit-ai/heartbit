use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use restate_sdk::prelude::*;

use heartbit::DynLlmProvider;
use heartbit::HeartbitConfig;
use heartbit::tool::Tool;
use heartbit::workflow::agent_service::{AgentService, AgentServiceImpl};
use heartbit::workflow::agent_workflow::{AgentWorkflow, AgentWorkflowImpl};
use heartbit::workflow::blackboard::{BlackboardObject, BlackboardObjectImpl};
use heartbit::workflow::budget::{TokenBudgetObject, TokenBudgetObjectImpl};
use heartbit::workflow::circuit_breaker::{CircuitBreakerObject, CircuitBreakerObjectImpl};
use heartbit::workflow::orchestrator_workflow::{OrchestratorWorkflow, OrchestratorWorkflowImpl};
use heartbit::workflow::scheduler::{SchedulerObject, SchedulerObjectImpl};

/// Start the Restate-compatible HTTP worker.
///
/// Loads config, creates the LLM provider and tools, optionally sets up
/// OpenTelemetry, then starts an HTTP server that Restate calls.
pub async fn run_worker(config_path: &Path, bind: &str) -> Result<()> {
    let config = HeartbitConfig::from_file(config_path)
        .with_context(|| format!("failed to load config from {}", config_path.display()))?;

    crate::init_tracing_from_config(&config)?;

    let provider = build_provider(&config)?;
    let tools = build_tools(&config).await?;

    tracing::info!(
        bind = %bind,
        provider = %config.provider.name,
        model = %config.provider.model,
        tools = tools.len(),
        "starting heartbit worker"
    );

    let addr: std::net::SocketAddr = bind.parse().context("invalid bind address")?;

    let agent_service = AgentServiceImpl::new(provider, &config.provider.name, tools);

    HttpServer::new(
        Endpoint::builder()
            .bind(agent_service.serve())
            .bind(AgentWorkflowImpl.serve())
            .bind(OrchestratorWorkflowImpl.serve())
            .bind(BlackboardObjectImpl.serve())
            .bind(TokenBudgetObjectImpl.serve())
            .bind(CircuitBreakerObjectImpl.serve())
            .bind(SchedulerObjectImpl.serve())
            .build(),
    )
    .listen_and_serve(addr)
    .await;

    Ok(())
}

pub(crate) fn setup_telemetry(otlp_endpoint: &str, service_name: &str) -> Result<()> {
    use opentelemetry::trace::TracerProvider;
    use opentelemetry_otlp::WithExportConfig;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(otlp_endpoint)
        .build()
        .context("failed to create OTLP exporter")?;

    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name(service_name.to_string())
                .build(),
        )
        .build();

    let tracer = provider.tracer("heartbit");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .init();

    Ok(())
}

fn build_provider(config: &HeartbitConfig) -> Result<Arc<dyn DynLlmProvider>> {
    // Reuse the shared provider construction from main.rs.
    // BoxedProvider implements DynLlmProvider, so the Arc coercion is safe.
    Ok(crate::build_provider_from_config(config, None)?)
}

async fn build_tools(config: &HeartbitConfig) -> Result<HashMap<String, Arc<dyn Tool>>> {
    let mut tools: HashMap<String, Arc<dyn Tool>> = HashMap::new();

    // Connect to MCP servers declared in agent configs
    for agent in &config.agents {
        for entry in &agent.mcp_servers {
            let url = entry.url();
            tracing::info!(agent = %agent.name, url = %url, "connecting to MCP server");
            let result = match entry.auth_header() {
                Some(auth) => heartbit::McpClient::connect_with_auth(url, auth).await,
                None => heartbit::McpClient::connect(url).await,
            };
            match result {
                Ok(client) => {
                    for tool in client.into_tools() {
                        let def = tool.definition();
                        if tools.contains_key(&def.name) {
                            tracing::warn!(
                                tool = %def.name,
                                agent = %agent.name,
                                url = %url,
                                "duplicate MCP tool name, keeping first registration"
                            );
                            continue;
                        }
                        tracing::info!(tool = %def.name, "registered MCP tool");
                        tools.insert(def.name.clone(), tool);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        agent = %agent.name,
                        url = %url,
                        error = %e,
                        "failed to connect to MCP server, skipping"
                    );
                }
            }
        }
    }

    // Connect to A2A agents declared in agent configs
    for agent in &config.agents {
        for entry in &agent.a2a_agents {
            let url = entry.url();
            tracing::info!(agent = %agent.name, url = %url, "discovering A2A agent");
            let result = match entry.auth_header() {
                Some(auth) => heartbit::A2aClient::connect_with_auth(url, auth).await,
                None => heartbit::A2aClient::connect(url).await,
            };
            match result {
                Ok(client) => {
                    for tool in client.into_tools() {
                        let def = tool.definition();
                        if tools.contains_key(&def.name) {
                            tracing::warn!(
                                tool = %def.name,
                                agent = %agent.name,
                                url = %url,
                                "duplicate A2A tool name, keeping first registration"
                            );
                            continue;
                        }
                        tracing::info!(tool = %def.name, "registered A2A agent tool");
                        tools.insert(def.name.clone(), tool);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        agent = %agent.name,
                        url = %url,
                        error = %e,
                        "failed to discover A2A agent, skipping"
                    );
                }
            }
        }
    }

    // Memory tools are not yet supported in the Restate worker path.
    // The shared AgentService has a single global tool registry, but memory
    // tools require per-agent namespacing (each agent needs its own
    // NamespacedMemory instance). In the standalone path, each sub-agent has
    // its own AgentRunner with isolated tools, so this works naturally.
    // TODO: Add per-agent tool scoping to AgentService for Restate memory support.
    if config.memory.is_some() {
        tracing::warn!(
            "memory configuration detected but memory tools are not yet supported \
             in the Restate worker path; use standalone mode ('heartbit run') for \
             memory-enabled agents"
        );
    }

    if config.knowledge.is_some() {
        tracing::warn!(
            "knowledge configuration detected but knowledge tools are not yet supported \
             in the Restate worker path; use standalone mode ('heartbit run') for \
             knowledge-enabled agents"
        );
    }

    Ok(tools)
}
