use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use restate_sdk::prelude::*;

use heartbit::llm::anthropic::AnthropicProvider;
use heartbit::llm::openrouter::OpenRouterProvider;
use heartbit::tool::Tool;
use heartbit::workflow::agent_service::{AgentService, AgentServiceImpl};
use heartbit::workflow::agent_workflow::{AgentWorkflow, AgentWorkflowImpl};
use heartbit::workflow::blackboard::{BlackboardObject, BlackboardObjectImpl};
use heartbit::workflow::budget::{TokenBudgetObject, TokenBudgetObjectImpl};
use heartbit::workflow::circuit_breaker::{CircuitBreakerObject, CircuitBreakerObjectImpl};
use heartbit::workflow::orchestrator_workflow::{OrchestratorWorkflow, OrchestratorWorkflowImpl};
use heartbit::workflow::scheduler::{SchedulerObject, SchedulerObjectImpl};
use heartbit::workflow::types::DynLlmProvider;
use heartbit::{HeartbitConfig, RetryConfig, RetryingProvider};

/// Start the Restate-compatible HTTP worker.
///
/// Loads config, creates the LLM provider and tools, optionally sets up
/// OpenTelemetry, then starts an HTTP server that Restate calls.
pub async fn run_worker(config_path: &Path, bind: &str) -> Result<()> {
    let config = HeartbitConfig::from_file(config_path)
        .with_context(|| format!("failed to load config from {}", config_path.display()))?;

    // Set up tracing: OTel-aware if configured, simple fmt otherwise
    if let Some(telemetry) = &config.telemetry {
        setup_telemetry(&telemetry.otlp_endpoint, &telemetry.service_name)?;
        tracing::info!(
            "OpenTelemetry configured, exporting to {}",
            telemetry.otlp_endpoint
        );
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .init();
    }

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

fn setup_telemetry(otlp_endpoint: &str, service_name: &str) -> Result<()> {
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
    let retry = config.provider.retry.as_ref().map(RetryConfig::from);

    match config.provider.name.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY env var required for anthropic provider")?;
            let base = AnthropicProvider::new(api_key, &config.provider.model);
            match retry {
                Some(rc) => Ok(Arc::new(RetryingProvider::new(base, rc))),
                None => Ok(Arc::new(base)),
            }
        }
        "openrouter" => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY env var required for openrouter provider")?;
            let base = OpenRouterProvider::new(api_key, &config.provider.model);
            match retry {
                Some(rc) => Ok(Arc::new(RetryingProvider::new(base, rc))),
                None => Ok(Arc::new(base)),
            }
        }
        other => bail!("Unknown provider: {other}. Use 'anthropic' or 'openrouter'."),
    }
}

async fn build_tools(config: &HeartbitConfig) -> Result<HashMap<String, Arc<dyn Tool>>> {
    let mut tools: HashMap<String, Arc<dyn Tool>> = HashMap::new();

    // Connect to MCP servers declared in agent configs
    for agent in &config.agents {
        for server_url in &agent.mcp_servers {
            tracing::info!(agent = %agent.name, url = %server_url, "connecting to MCP server");
            match heartbit::McpClient::connect(server_url).await {
                Ok(client) => {
                    for tool in client.into_tools() {
                        let def = tool.definition();
                        if tools.contains_key(&def.name) {
                            bail!(
                                "duplicate MCP tool name '{}' from agent '{}' server '{}' — \
                                 rename or remove the conflicting tool",
                                def.name,
                                agent.name,
                                server_url
                            );
                        }
                        tracing::info!(tool = %def.name, "registered MCP tool");
                        tools.insert(def.name.clone(), tool);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        agent = %agent.name,
                        url = %server_url,
                        error = %e,
                        "failed to connect to MCP server, skipping"
                    );
                }
            }
        }
    }

    Ok(tools)
}
