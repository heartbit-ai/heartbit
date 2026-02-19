mod serve;
mod submit;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use heartbit::tool::Tool;
use heartbit::{
    AgentOutput, AnthropicProvider, ContextStrategy, ContextStrategyConfig, HeartbitConfig,
    InMemoryStore, LlmProvider, McpClient, Memory, MemoryConfig, OpenRouterProvider, Orchestrator,
    RetryConfig, RetryingProvider,
};

#[derive(Parser)]
#[command(name = "heartbit", about = "Multi-agent enterprise runtime")]
struct Cli {
    /// Path to heartbit.toml config file
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,

    /// Task to execute (when no subcommand is given)
    #[arg(trailing_var_arg = true)]
    task: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute a task directly (standalone mode, no Restate)
    Run {
        /// The task to execute
        #[arg(trailing_var_arg = true)]
        task: Vec<String>,
    },
    /// Start the Restate-compatible HTTP worker
    Serve {
        /// Address to bind the worker
        #[arg(long, default_value = "0.0.0.0:9080")]
        bind: String,
    },
    /// Submit a task to Restate for durable execution
    Submit {
        /// The task to execute
        #[arg(trailing_var_arg = true)]
        task: Vec<String>,

        /// Restate ingress URL (overrides config; defaults to http://localhost:8080)
        #[arg(long)]
        restate_url: Option<String>,
    },
    /// Query workflow status
    Status {
        /// Workflow ID to check
        workflow_id: String,

        /// Restate ingress URL (overrides config; defaults to http://localhost:8080)
        #[arg(long)]
        restate_url: Option<String>,
    },
    /// Send human approval signal to a workflow
    Approve {
        /// Workflow ID to approve
        workflow_id: String,

        /// Restate ingress URL (overrides config; defaults to http://localhost:8080)
        #[arg(long)]
        restate_url: Option<String>,
    },
}

const DEFAULT_RESTATE_URL: &str = "http://localhost:8080";

/// Resolve restate URL: CLI flag → config file → default.
fn resolve_restate_url(cli_url: Option<String>, config_path: Option<&std::path::Path>) -> String {
    if let Some(url) = cli_url {
        return url;
    }
    if let Some(path) = config_path
        && let Ok(config) = HeartbitConfig::from_file(path)
        && let Some(restate) = config.restate
    {
        return restate.endpoint;
    }
    DEFAULT_RESTATE_URL.into()
}

/// Initialize the simple fmt tracing subscriber (for non-serve commands).
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run { task }) => {
            init_tracing();
            let task_str = task.join(" ");
            if task_str.is_empty() {
                bail!("Usage: heartbit run <task>");
            }
            run_standalone(cli.config.as_deref(), &task_str).await
        }
        Some(Commands::Serve { bind }) => {
            // serve::run_worker handles its own tracing init (with optional OTel)
            let config_path = cli
                .config
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("heartbit.toml"));
            serve::run_worker(config_path, &bind).await
        }
        Some(Commands::Submit { task, restate_url }) => {
            init_tracing();
            let task_str = task.join(" ");
            if task_str.is_empty() {
                bail!("Usage: heartbit submit <task>");
            }
            let config_path = cli
                .config
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("heartbit.toml"));
            let url = resolve_restate_url(restate_url, Some(config_path));
            submit::submit_task(config_path, &task_str, &url).await
        }
        Some(Commands::Status {
            workflow_id,
            restate_url,
        }) => {
            init_tracing();
            let url = resolve_restate_url(restate_url, cli.config.as_deref());
            submit::query_status(&workflow_id, &url).await
        }
        Some(Commands::Approve {
            workflow_id,
            restate_url,
        }) => {
            init_tracing();
            let url = resolve_restate_url(restate_url, cli.config.as_deref());
            submit::send_approval(&workflow_id, &url).await
        }
        None => {
            init_tracing();
            // Backward-compatible: bare task args without subcommand
            let task_str = cli.task.join(" ");
            if task_str.is_empty() {
                bail!(
                    "Usage: heartbit [run|serve|submit|status|approve] <args>\n       heartbit <task>  (shorthand for 'run')"
                );
            }
            run_standalone(cli.config.as_deref(), &task_str).await
        }
    }
}

async fn run_standalone(config_path: Option<&std::path::Path>, task: &str) -> Result<()> {
    match config_path {
        Some(path) => run_from_config(path, task).await,
        None => run_from_env(task).await,
    }
}

/// Build a `RetryConfig` from the provider config, if retry is configured.
fn retry_config_from(config: &HeartbitConfig) -> Option<RetryConfig> {
    config.provider.retry.as_ref().map(|r| RetryConfig {
        max_retries: r.max_retries,
        base_delay: std::time::Duration::from_millis(r.base_delay_ms),
        max_delay: std::time::Duration::from_millis(r.max_delay_ms),
    })
}

async fn run_from_config(path: &std::path::Path, task: &str) -> Result<()> {
    let config = HeartbitConfig::from_file(path)
        .with_context(|| format!("failed to load config from {}", path.display()))?;

    let retry = retry_config_from(&config);

    match config.provider.name.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY env var required for anthropic provider")?;
            let base = AnthropicProvider::new(api_key, &config.provider.model);
            match retry {
                Some(rc) => {
                    let provider = Arc::new(RetryingProvider::new(base, rc));
                    let output = build_orchestrator_from_config(provider, &config, task).await?;
                    print_output(&output);
                }
                None => {
                    let provider = Arc::new(base);
                    let output = build_orchestrator_from_config(provider, &config, task).await?;
                    print_output(&output);
                }
            }
        }
        "openrouter" => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY env var required for openrouter provider")?;
            let base = OpenRouterProvider::new(api_key, &config.provider.model);
            match retry {
                Some(rc) => {
                    let provider = Arc::new(RetryingProvider::new(base, rc));
                    let output = build_orchestrator_from_config(provider, &config, task).await?;
                    print_output(&output);
                }
                None => {
                    let provider = Arc::new(base);
                    let output = build_orchestrator_from_config(provider, &config, task).await?;
                    print_output(&output);
                }
            }
        }
        other => bail!("Unknown provider: {other}. Use 'anthropic' or 'openrouter'."),
    }

    Ok(())
}

async fn build_orchestrator_from_config<P: LlmProvider + 'static>(
    provider: Arc<P>,
    config: &HeartbitConfig,
    task: &str,
) -> Result<AgentOutput> {
    let mut builder = Orchestrator::builder(provider)
        .max_turns(config.orchestrator.max_turns)
        .max_tokens(config.orchestrator.max_tokens);

    for agent in &config.agents {
        let tools = load_mcp_tools(&agent.name, &agent.mcp_servers).await;
        let (ctx_strategy, summarize_threshold) = match &agent.context_strategy {
            Some(ContextStrategyConfig::SlidingWindow { max_tokens }) => (
                Some(ContextStrategy::SlidingWindow {
                    max_tokens: *max_tokens,
                }),
                None,
            ),
            Some(ContextStrategyConfig::Summarize { max_tokens }) => (None, Some(*max_tokens)),
            Some(ContextStrategyConfig::Unlimited) | None => (None, None),
        };

        builder = builder.sub_agent_full(
            &agent.name,
            &agent.description,
            &agent.system_prompt,
            tools,
            ctx_strategy,
            summarize_threshold,
        );
    }

    // Wire shared memory if configured
    if let Some(ref memory_config) = config.memory {
        let memory: Arc<dyn Memory> = match memory_config {
            MemoryConfig::InMemory => Arc::new(InMemoryStore::new()),
            MemoryConfig::Postgres { .. } => {
                anyhow::bail!(
                    "Postgres memory store is not yet implemented. Use type = \"in_memory\" instead."
                );
            }
        };
        builder = builder.shared_memory(memory);
    }

    let orchestrator = builder.build()?;
    let output = orchestrator.run(task).await?;
    Ok(output)
}

/// Connect to MCP servers and collect tools. Failures are logged and skipped.
async fn load_mcp_tools(agent_name: &str, mcp_servers: &[String]) -> Vec<Arc<dyn Tool>> {
    let mut tools = Vec::new();
    for server_url in mcp_servers {
        tracing::info!(agent = %agent_name, url = %server_url, "connecting to MCP server");
        match McpClient::connect(server_url).await {
            Ok(client) => {
                for tool in client.into_tools() {
                    let def = tool.definition();
                    tracing::info!(tool = %def.name, "registered MCP tool");
                    tools.push(tool);
                }
            }
            Err(e) => {
                tracing::warn!(
                    agent = %agent_name,
                    url = %server_url,
                    error = %e,
                    "failed to connect to MCP server, skipping"
                );
            }
        }
    }
    tools
}

async fn run_from_env(task: &str) -> Result<()> {
    let provider_name = std::env::var("HEARTBIT_PROVIDER").unwrap_or_else(|_| {
        if std::env::var("OPENROUTER_API_KEY").is_ok() {
            "openrouter".into()
        } else {
            "anthropic".into()
        }
    });

    match provider_name.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY env var required for anthropic provider")?;
            let model = std::env::var("HEARTBIT_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
            let provider = Arc::new(RetryingProvider::with_defaults(AnthropicProvider::new(
                api_key, model,
            )));
            let output = run_default_orchestrator(provider, task).await?;
            print_output(&output);
        }
        "openrouter" => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY env var required for openrouter provider")?;
            let model = std::env::var("HEARTBIT_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-sonnet-4".into());
            let provider = Arc::new(RetryingProvider::with_defaults(OpenRouterProvider::new(
                api_key, model,
            )));
            let output = run_default_orchestrator(provider, task).await?;
            print_output(&output);
        }
        other => bail!("Unknown provider: {other}. Use 'anthropic' or 'openrouter'."),
    }

    Ok(())
}

async fn run_default_orchestrator<P: LlmProvider + 'static>(
    provider: Arc<P>,
    task: &str,
) -> Result<AgentOutput> {
    let orchestrator = Orchestrator::builder(provider)
        .sub_agent(
            "researcher",
            "Research specialist who gathers information and facts",
            "You are a research specialist. Your job is to gather relevant information, \
             facts, and data about the topic you're given. Be thorough and cite your \
             reasoning. Focus on accuracy and completeness.",
        )
        .sub_agent(
            "analyst",
            "Analytical thinker who evaluates and synthesizes information",
            "You are an analytical expert. Your job is to evaluate information critically, \
             identify patterns, weigh pros and cons, and provide structured analysis. \
             Be objective and thorough in your reasoning.",
        )
        .sub_agent(
            "writer",
            "Clear communicator who produces well-structured output",
            "You are a writing specialist. Your job is to take information and analysis \
             and produce clear, well-structured, and engaging output. Focus on clarity, \
             organization, and readability.",
        )
        .build()?;

    let output = orchestrator.run(task).await?;
    Ok(output)
}

fn print_output(output: &AgentOutput) {
    println!("{}", output.result);
    eprintln!(
        "\n---\nTokens used: {} in / {} out | Tool calls: {}",
        output.tokens_used.input_tokens, output.tokens_used.output_tokens, output.tool_calls_made,
    );
}
