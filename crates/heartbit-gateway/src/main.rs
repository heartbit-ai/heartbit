use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use tokio_util::sync::CancellationToken;

mod config;
mod error;
mod producer;
mod server;
mod sources;

use config::GatewayConfig;
use producer::GatewayProducer;

#[derive(Parser)]
#[command(name = "heartbit-gateway", about = "Heartbit ingestion gateway")]
struct Cli {
    /// Path to gateway TOML config file
    #[arg(short, long, default_value = "gateway.toml")]
    config: std::path::PathBuf,
    /// Override bind address
    #[arg(short, long)]
    bind: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let config = GatewayConfig::from_file(&cli.config)
        .with_context(|| format!("failed to load config from {}", cli.config.display()))?;

    let bind = cli
        .bind
        .unwrap_or_else(|| config.server.listen_addr.clone());
    let cancel = CancellationToken::new();

    // Ensure Kafka topics exist
    heartbit::daemon::kafka::ensure_topics(&config.kafka)
        .await
        .context("failed to ensure Kafka topics")?;

    // Create Kafka producer
    let producer = heartbit::daemon::kafka::create_producer(&config.kafka)
        .context("failed to create Kafka producer")?;

    let gateway_producer = GatewayProducer::new(
        heartbit::KafkaCommandProducer::new(producer.clone()),
        &config.kafka.commands_topic,
    );

    // Start cron scheduler
    sources::cron::start_cron(
        &config.schedules,
        gateway_producer.inner().clone(),
        &config.kafka.commands_topic,
        cancel.clone(),
    )?;

    // Start sensor manager (if configured)
    if let Some(ref sensor_config) = config.sensors {
        // Build LLM provider for sensor triage from environment.
        // ANTHROPIC_API_KEY or OPENROUTER_API_KEY must be set for sensor triage.
        let slm_provider =
            build_slm_provider().context("failed to build SLM provider for sensor triage")?;

        sources::sensors::start_sensors(
            sensor_config,
            producer.clone(),
            slm_provider,
            &config.kafka,
            &config.kafka.commands_topic,
            &config.kafka.dead_letter_topic,
            cancel.clone(),
        )
        .await
        .context("failed to start sensors")?;
    }

    tracing::info!(topic = gateway_producer.topic(), "gateway producer ready");

    // Signal handler
    heartbit::signal::spawn_shutdown_handler(cancel.clone());

    // Start HTTP server
    let state = server::GatewayState {
        start_time: Instant::now(),
        cancel: cancel.clone(),
    };
    let app = server::build_router(state);

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("failed to bind to {bind}"))?;
    tracing::info!(bind = %bind, "gateway HTTP server started");

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            cancel.cancelled().await;
        })
        .await
        .context("gateway HTTP server error")?;

    tracing::info!("gateway shut down gracefully");
    Ok(())
}

/// Build the LLM provider for sensor triage from environment variables.
///
/// Tries `ANTHROPIC_API_KEY` first, then `OPENROUTER_API_KEY`. Both paths
/// currently wire `claude-sonnet-4` — NOT a haiku-class SLM. Triage runs per
/// ingested event, so model cost multiplies; switching to a cheaper
/// haiku-class model (or making this configurable) is a deliberate follow-up,
/// not something this function does today.
fn build_slm_provider() -> Result<Arc<dyn heartbit::llm::DynLlmProvider>> {
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
        let provider = heartbit::AnthropicProvider::new(&api_key, "claude-sonnet-4-20250514");
        return Ok(Arc::new(provider));
    }

    if let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") {
        let provider =
            heartbit::OpenRouterProvider::new(&api_key, "anthropic/claude-sonnet-4-20250514");
        return Ok(Arc::new(provider));
    }

    anyhow::bail!(
        "sensor triage requires an LLM provider: set ANTHROPIC_API_KEY or OPENROUTER_API_KEY"
    )
}
