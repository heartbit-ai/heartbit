mod serve;
mod submit;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use heartbit::tool::Tool;
use heartbit::{
    AgentOutput, AnthropicProvider, Blackboard, BoxedProvider, ContextStrategy,
    ContextStrategyConfig, HeartbitConfig, InMemoryBlackboard, InMemoryKnowledgeBase,
    InMemoryStore, KnowledgeBase, KnowledgeSourceConfig, McpClient, Memory, MemoryConfig,
    OnApproval, OnText, OpenRouterProvider, Orchestrator, PostgresMemoryStore, RetryConfig,
    RetryingProvider, SubAgentConfig, ToolCall,
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
        /// Require human approval before each tool execution round
        #[arg(long)]
        approve: bool,
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

        /// Require human approval before each tool execution round
        #[arg(long)]
        approve: bool,

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
    /// Send human approval signal to a child agent workflow.
    /// Use 'heartbit status <orchestrator-id>' to find child workflow IDs.
    Approve {
        /// Child agent workflow ID to approve (from 'status' output)
        workflow_id: String,

        /// Restate ingress URL (overrides config; defaults to http://localhost:8080)
        #[arg(long)]
        restate_url: Option<String>,
    },
    /// Get the result of a completed workflow
    Result {
        /// Workflow ID to get results from
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
        Some(Commands::Run { task, approve }) => {
            init_tracing();
            let task_str = task.join(" ");
            if task_str.is_empty() {
                bail!("Usage: heartbit run <task>");
            }
            run_standalone(cli.config.as_deref(), &task_str, approve).await
        }
        Some(Commands::Serve { bind }) => {
            // serve::run_worker handles its own tracing init (with optional OTel)
            let config_path = cli
                .config
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("heartbit.toml"));
            serve::run_worker(config_path, &bind).await
        }
        Some(Commands::Submit {
            task,
            approve,
            restate_url,
        }) => {
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
            submit::submit_task(config_path, &task_str, &url, approve).await
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
        Some(Commands::Result {
            workflow_id,
            restate_url,
        }) => {
            init_tracing();
            let url = resolve_restate_url(restate_url, cli.config.as_deref());
            submit::get_result(&workflow_id, &url).await
        }
        None => {
            init_tracing();
            // Backward-compatible: bare task args without subcommand
            let task_str = cli.task.join(" ");
            if task_str.is_empty() {
                bail!(
                    "Usage: heartbit [run|serve|submit|status|approve|result] <args>\n       heartbit <task>  (shorthand for 'run')"
                );
            }
            run_standalone(cli.config.as_deref(), &task_str, false).await
        }
    }
}

async fn run_standalone(
    config_path: Option<&std::path::Path>,
    task: &str,
    approve: bool,
) -> Result<()> {
    match config_path {
        Some(path) => run_from_config(path, task, approve).await,
        None => run_from_env(task, approve).await,
    }
}

/// Build a `RetryConfig` from the provider config, if retry is configured.
fn retry_config_from(config: &HeartbitConfig) -> Option<RetryConfig> {
    config.provider.retry.as_ref().map(RetryConfig::from)
}

/// Construct a type-erased LLM provider from config.
fn build_provider_from_config(config: &HeartbitConfig) -> Result<Arc<BoxedProvider>> {
    let retry = retry_config_from(config);
    match config.provider.name.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY env var required for anthropic provider")?;
            let base = if config.provider.prompt_caching {
                AnthropicProvider::with_prompt_caching(api_key, &config.provider.model)
            } else {
                AnthropicProvider::new(api_key, &config.provider.model)
            };
            match retry {
                Some(rc) => Ok(Arc::new(BoxedProvider::new(RetryingProvider::new(
                    base, rc,
                )))),
                None => Ok(Arc::new(BoxedProvider::new(base))),
            }
        }
        "openrouter" => {
            if config.provider.prompt_caching {
                tracing::warn!(
                    "prompt_caching is only effective with the 'anthropic' provider; \
                     ignored for 'openrouter'"
                );
            }
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY env var required for openrouter provider")?;
            let base = OpenRouterProvider::new(api_key, &config.provider.model);
            match retry {
                Some(rc) => Ok(Arc::new(BoxedProvider::new(RetryingProvider::new(
                    base, rc,
                )))),
                None => Ok(Arc::new(BoxedProvider::new(base))),
            }
        }
        other => bail!("Unknown provider: {other}. Use 'anthropic' or 'openrouter'."),
    }
}

async fn run_from_config(path: &std::path::Path, task: &str, approve: bool) -> Result<()> {
    let config = HeartbitConfig::from_file(path)
        .with_context(|| format!("failed to load config from {}", path.display()))?;

    let provider = build_provider_from_config(&config)?;
    let on_text = streaming_callback();
    let on_approval = if approve {
        Some(approval_callback())
    } else {
        None
    };

    let output =
        build_orchestrator_from_config(provider, &config, task, on_text, on_approval).await?;
    print_streaming_stats(&output);
    Ok(())
}

/// Load and index knowledge sources from config into an in-memory knowledge base.
async fn load_knowledge_base(config: &heartbit::KnowledgeConfig) -> Result<Arc<dyn KnowledgeBase>> {
    use heartbit::knowledge::chunker::ChunkConfig;
    use heartbit::knowledge::loader;

    let kb = Arc::new(InMemoryKnowledgeBase::new());
    let chunk_config = ChunkConfig {
        chunk_size: config.chunk_size,
        chunk_overlap: config.chunk_overlap,
    };

    for source in &config.sources {
        match source {
            KnowledgeSourceConfig::File { path } => {
                let path = std::path::Path::new(path);
                match loader::load_file(&*kb, path, &chunk_config).await {
                    Ok(count) => {
                        tracing::info!(path = %path.display(), chunks = count, "indexed knowledge file");
                    }
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "failed to load knowledge file, skipping");
                    }
                }
            }
            KnowledgeSourceConfig::Glob { pattern } => {
                match loader::load_glob(&*kb, pattern, &chunk_config).await {
                    Ok(count) => {
                        tracing::info!(pattern = %pattern, chunks = count, "indexed knowledge glob");
                    }
                    Err(e) => {
                        tracing::warn!(pattern = %pattern, error = %e, "failed to load knowledge glob, skipping");
                    }
                }
            }
            KnowledgeSourceConfig::Url { url } => {
                match loader::load_url(&*kb, url, &chunk_config).await {
                    Ok(count) => {
                        tracing::info!(url = %url, chunks = count, "indexed knowledge URL");
                    }
                    Err(e) => {
                        tracing::warn!(url = %url, error = %e, "failed to load knowledge URL, skipping");
                    }
                }
            }
        }
    }

    let total = kb.chunk_count().await.unwrap_or(0);
    tracing::info!(total_chunks = total, "knowledge base loaded");

    Ok(kb)
}

async fn build_orchestrator_from_config(
    provider: Arc<BoxedProvider>,
    config: &HeartbitConfig,
    task: &str,
    on_text: Arc<OnText>,
    on_approval: Option<Arc<OnApproval>>,
) -> Result<AgentOutput> {
    let mut builder = Orchestrator::builder(provider)
        .max_turns(config.orchestrator.max_turns)
        .max_tokens(config.orchestrator.max_tokens)
        .on_text(on_text);

    if let Some(cb) = on_approval {
        builder = builder.on_approval(cb);
    }

    for agent in &config.agents {
        let tools = load_mcp_tools(&agent.name, &agent.mcp_servers).await;
        let (ctx_strategy, summarize_threshold) = match &agent.context_strategy {
            Some(ContextStrategyConfig::SlidingWindow { max_tokens }) => (
                Some(ContextStrategy::SlidingWindow {
                    max_tokens: *max_tokens,
                }),
                None,
            ),
            Some(ContextStrategyConfig::Summarize { threshold }) => (None, Some(*threshold)),
            Some(ContextStrategyConfig::Unlimited) | None => (None, None),
        };

        builder = builder.sub_agent_full(SubAgentConfig {
            name: agent.name.clone(),
            description: agent.description.clone(),
            system_prompt: agent.system_prompt.clone(),
            tools,
            context_strategy: ctx_strategy,
            summarize_threshold,
            tool_timeout: agent
                .tool_timeout_seconds
                .map(std::time::Duration::from_secs),
            max_tool_output_bytes: agent.max_tool_output_bytes,
            max_turns: agent.max_turns,
            max_tokens: agent.max_tokens,
            response_schema: agent.response_schema.clone(),
        });
    }

    // Wire shared memory if configured
    if let Some(ref memory_config) = config.memory {
        let memory: Arc<dyn Memory> = match memory_config {
            MemoryConfig::InMemory => Arc::new(InMemoryStore::new()),
            MemoryConfig::Postgres { database_url } => {
                let store = PostgresMemoryStore::connect(database_url)
                    .await
                    .context("failed to connect to PostgreSQL for memory store")?;
                store
                    .run_migration()
                    .await
                    .context("failed to run memory store migration")?;
                Arc::new(store)
            }
        };
        builder = builder.shared_memory(memory);
    }

    // Always attach an in-memory blackboard for cross-agent coordination
    let blackboard: Arc<dyn Blackboard> = Arc::new(InMemoryBlackboard::new());
    builder = builder.blackboard(blackboard);

    // Wire knowledge base if configured
    if let Some(ref knowledge_config) = config.knowledge {
        let kb = load_knowledge_base(knowledge_config).await?;
        builder = builder.knowledge(kb);
    }

    let mut orchestrator = builder.build()?;
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

async fn run_from_env(task: &str, approve: bool) -> Result<()> {
    let provider_name = std::env::var("HEARTBIT_PROVIDER").unwrap_or_else(|_| {
        if std::env::var("OPENROUTER_API_KEY").is_ok() {
            "openrouter".into()
        } else {
            "anthropic".into()
        }
    });

    let provider: Arc<BoxedProvider> = match provider_name.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY env var required for anthropic provider")?;
            let model = std::env::var("HEARTBIT_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
            let prompt_caching = std::env::var("HEARTBIT_PROMPT_CACHING")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let base = if prompt_caching {
                AnthropicProvider::with_prompt_caching(api_key, model)
            } else {
                AnthropicProvider::new(api_key, model)
            };
            Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(base)))
        }
        "openrouter" => {
            if std::env::var("HEARTBIT_PROMPT_CACHING").is_ok() {
                tracing::warn!(
                    "HEARTBIT_PROMPT_CACHING is only effective with the 'anthropic' provider; \
                     ignored for 'openrouter'"
                );
            }
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY env var required for openrouter provider")?;
            let model = std::env::var("HEARTBIT_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-sonnet-4".into());
            Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(
                OpenRouterProvider::new(api_key, model),
            )))
        }
        other => bail!("Unknown provider: {other}. Use 'anthropic' or 'openrouter'."),
    };

    let on_text = streaming_callback();
    let on_approval = if approve {
        Some(approval_callback())
    } else {
        None
    };

    let output = run_default_orchestrator(provider, task, on_text, on_approval).await?;
    print_streaming_stats(&output);
    Ok(())
}

async fn run_default_orchestrator(
    provider: Arc<BoxedProvider>,
    task: &str,
    on_text: Arc<OnText>,
    on_approval: Option<Arc<OnApproval>>,
) -> Result<AgentOutput> {
    let mut builder = Orchestrator::builder(provider).on_text(on_text);

    if let Some(cb) = on_approval {
        builder = builder.on_approval(cb);
    }

    let blackboard: Arc<dyn Blackboard> = Arc::new(InMemoryBlackboard::new());

    let mut orchestrator = builder
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
        .blackboard(blackboard)
        .build()?;

    let output = orchestrator.run(task).await?;
    Ok(output)
}

/// Create a streaming text callback that writes deltas to stdout immediately.
fn streaming_callback() -> Arc<OnText> {
    Arc::new(|text: &str| {
        use std::io::Write;
        print!("{text}");
        std::io::stdout().flush().ok();
    })
}

/// Create an interactive approval callback that prompts on stderr.
///
/// The `OnApproval` callback is sync by design (`dyn Fn(&[ToolCall]) -> bool`).
/// The blocking stdin read runs on one tokio worker thread, but this is
/// intentional: the orchestrator is waiting for the human decision anyway,
/// and the multi-threaded runtime (`#[tokio::main]`) allows other tasks to
/// progress on remaining worker threads.
fn approval_callback() -> Arc<OnApproval> {
    Arc::new(|tool_calls: &[ToolCall]| {
        use std::io::Write;
        eprintln!("\n--- Approval Required ---");
        for tc in tool_calls {
            eprintln!("  Tool: {}", tc.name);
            if let Ok(pretty) = serde_json::to_string_pretty(&tc.input) {
                for line in pretty.lines() {
                    eprintln!("    {line}");
                }
            }
        }
        eprint!("Approve? [Y/n] ");
        std::io::stderr().flush().ok();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            return false;
        }
        let trimmed = input.trim().to_lowercase();
        trimmed.is_empty() || trimmed == "y" || trimmed == "yes"
    })
}

/// Print stats after a streaming run (text already printed via callback).
fn print_streaming_stats(output: &AgentOutput) {
    // Ensure stats start on a new line after streamed text
    let u = &output.tokens_used;
    if u.cache_creation_input_tokens > 0 || u.cache_read_input_tokens > 0 {
        eprintln!(
            "\n---\nTokens used: {} in / {} out (cache: {} created, {} read) | Tool calls: {}",
            u.input_tokens,
            u.output_tokens,
            u.cache_creation_input_tokens,
            u.cache_read_input_tokens,
            output.tool_calls_made,
        );
    } else {
        eprintln!(
            "\n---\nTokens used: {} in / {} out | Tool calls: {}",
            u.input_tokens, u.output_tokens, output.tool_calls_made,
        );
    }
}
