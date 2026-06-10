mod daemon;
mod persona;
mod persona_review;
mod serve;
mod submit;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use heartbit::tool::Tool;
use heartbit::{
    A2aClient, AgentEvent, AgentOutput, AgentRunner, AnthropicProvider, Blackboard, BoxedProvider,
    BuiltinToolsConfig, CascadeConfig, CascadeGateConfig, CascadingProvider, ContextStrategy,
    ContextStrategyConfig, HeartbitConfig, HeuristicGate, InMemoryBlackboard,
    InMemoryKnowledgeBase, InMemoryStore, KnowledgeBase, KnowledgeSourceConfig, McpClient,
    McpServerEntry, Memory, MemoryConfig, MemoryQuery, NamespacedMemory, ObservabilityMode,
    OnApproval, OnEvent, OnQuestion, OnRetry, OnText, Orchestrator, PostgresMemoryStore,
    QuestionRequest, QuestionResponse, RetryConfig, RetryingProvider, SubAgentConfig, TenantScope,
    ToolCall, Workspace, builtin_tools,
};

#[derive(Parser)]
#[command(name = "heartbit", about = "Multi-agent enterprise runtime", version)]
struct Cli {
    /// Path to heartbit.toml config file
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Observability verbosity: production, analysis, or debug
    #[arg(long, global = true)]
    observability: Option<String>,

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
        /// Print structured agent events to stderr as one-line JSON
        #[arg(long, short)]
        verbose: bool,
        /// Write the final AgentOutput as pretty JSON to this path (headless
        /// benchmark trace; env path only). Failure to write logs a warning
        /// and does not fail the run.
        #[arg(long)]
        trace_file: Option<PathBuf>,
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
    /// Start an interactive chat session (REPL)
    Chat {
        /// Require human approval before each tool execution round
        #[arg(long)]
        approve: bool,
        /// Print structured agent events to stderr as one-line JSON
        #[arg(long, short)]
        verbose: bool,
    },
    /// Run the daemon: long-running Kafka-backed task execution with HTTP API
    Daemon {
        /// Address to bind the HTTP API (overrides config)
        #[arg(long)]
        bind: Option<String>,
        /// Print structured agent events to stderr as one-line JSON
        #[arg(long, short)]
        verbose: bool,
        /// Validate config and exit (no Kafka, no HTTP bind, no DB connect)
        #[arg(long)]
        validate_config: bool,
    },
    /// Manage personas (list, run, configure, audit).
    Persona {
        #[command(subcommand)]
        sub: persona::PersonaCommand,
    },
    /// Manage agent templates
    Templates {
        #[command(subcommand)]
        action: TemplateAction,
    },
    /// Manage agent skills
    Skills {
        #[command(subcommand)]
        action: SkillAction,
    },
    /// Generate a starter config from a template
    Init {
        /// Template name to use as base
        template: String,
        /// Output file path (prints to stdout if not set)
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum TemplateAction {
    /// List all available templates
    List,
    /// Show details for a specific template
    Show {
        /// Template name
        name: String,
    },
}

#[derive(Subcommand)]
enum SkillAction {
    /// List all available skills
    List,
    /// Show content of a specific skill
    Show {
        /// Skill name
        name: String,
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

/// Resolve the observability mode from CLI flag, config, env, and verbose flag.
///
/// Priority: CLI `--observability` → config `[telemetry].observability_mode` →
/// `HEARTBIT_OBSERVABILITY` env var → `--verbose` implies Analysis → Production.
pub(crate) fn resolve_observability(
    cli_flag: Option<&str>,
    config_str: Option<&str>,
    verbose: bool,
) -> ObservabilityMode {
    // CLI flag has highest priority — check before resolve() which uses env > config > builder.
    if let Some(flag) = cli_flag {
        if let Some(mode) = ObservabilityMode::from_str_loose(flag) {
            return mode;
        }
        tracing::warn!(value = flag, "unknown --observability value, falling back");
    }

    let resolved = ObservabilityMode::resolve("HEARTBIT_OBSERVABILITY", config_str, None);

    // If nothing was explicitly set and --verbose is on, use Analysis
    if config_str.is_none()
        && std::env::var("HEARTBIT_OBSERVABILITY").is_err()
        && verbose
        && resolved == ObservabilityMode::Production
    {
        return ObservabilityMode::Analysis;
    }

    resolved
}

/// Initialize the simple fmt tracing subscriber (for non-serve commands).
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

/// Initialize tracing from config: OTel-aware if `[telemetry]` is set, simple fmt otherwise.
pub(crate) fn init_tracing_from_config(config: &HeartbitConfig) -> Result<()> {
    if let Some(telemetry) = &config.telemetry {
        serve::setup_telemetry(&telemetry.otlp_endpoint, &telemetry.service_name)?;
        tracing::info!(
            "OpenTelemetry configured, exporting to {}",
            telemetry.otlp_endpoint
        );
    } else {
        init_tracing();
    }
    Ok(())
}

/// Discover and load `HEARTBIT.md` instruction files from the current working directory.
///
/// Walks up the directory tree to the git root, then checks the global config
/// directory. Returns the combined instruction text, or `None` if no files found.
pub(crate) fn load_instruction_text() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let paths = heartbit::discover_instruction_files(&cwd);
    if paths.is_empty() {
        return None;
    }
    for path in &paths {
        tracing::info!(path = %path.display(), "discovered instruction file");
    }
    match heartbit::load_instructions(&paths) {
        Ok(text) if !text.is_empty() => Some(text),
        Ok(_) => None,
        Err(e) => {
            tracing::warn!(error = %e, "failed to load instruction files, skipping");
            None
        }
    }
}

/// Load learned permissions from the default path (`~/.config/heartbit/permissions.toml`).
///
/// Returns `None` if the default path cannot be determined. Logs a warning
/// and returns `None` if the file exists but fails to parse.
pub(crate) fn load_learned_permissions()
-> Option<Arc<std::sync::Mutex<heartbit::LearnedPermissions>>> {
    let path = heartbit::LearnedPermissions::default_path()?;
    match heartbit::LearnedPermissions::load(&path) {
        Ok(learned) => {
            if !learned.rules().is_empty() {
                tracing::info!(
                    path = %path.display(),
                    rules = learned.rules().len(),
                    "loaded learned permission rules"
                );
            }
            Some(Arc::new(std::sync::Mutex::new(learned)))
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to load learned permissions, ignoring");
            None
        }
    }
}

/// Provision a workspace directory, creating it if needed.
///
/// Returns `Some(path)` on success, logs a warning and returns `None` on failure.
pub(crate) fn provision_workspace(root: &std::path::Path) -> Option<PathBuf> {
    match Workspace::open(root) {
        Ok(ws) => Some(ws.root().to_path_buf()),
        Err(e) => {
            tracing::warn!(path = %root.display(), error = %e, "failed to create workspace, continuing without");
            None
        }
    }
}

/// Resolve workspace root from config, falling back to the default.
pub(crate) fn workspace_root_from_config(config: &HeartbitConfig) -> PathBuf {
    config
        .workspace
        .as_ref()
        .map(|ws| PathBuf::from(&ws.root))
        .unwrap_or_else(default_workspace_root_path)
}

/// Resolve the workspace root from an optional `HEARTBIT_WORKSPACE` override and
/// the `HOME` directory. When `workspace_env` is set and non-empty, it is used
/// verbatim; otherwise falls back to `{home}/.heartbit/workspaces`.
///
/// Pure helper so it can be unit-tested without env mutation.
fn resolve_workspace_root(workspace_env: Option<&str>, home: &str) -> PathBuf {
    match workspace_env.map(str::trim).filter(|v| !v.is_empty()) {
        Some(ws) => PathBuf::from(ws),
        None => PathBuf::from(format!("{home}/.heartbit/workspaces")),
    }
}

/// Default workspace root when no config is available. Honors the
/// `HEARTBIT_WORKSPACE` env override first, else `~/.heartbit/workspaces`.
fn default_workspace_root_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let workspace_env = std::env::var("HEARTBIT_WORKSPACE").ok();
    resolve_workspace_root(workspace_env.as_deref(), &home)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present (silently ignore if missing)
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run {
            task,
            approve,
            verbose,
            trace_file,
        }) => {
            let task_str = task.join(" ");
            if task_str.is_empty() {
                bail!("Usage: heartbit run <task>");
            }
            run_standalone(
                cli.config.as_deref(),
                &task_str,
                approve,
                verbose,
                cli.observability.as_deref(),
                trace_file,
            )
            .await
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
        Some(Commands::Chat { approve, verbose }) => {
            run_chat(
                cli.config.as_deref(),
                approve,
                verbose,
                cli.observability.as_deref(),
            )
            .await
        }
        Some(Commands::Daemon {
            bind,
            verbose,
            validate_config,
        }) => {
            let config_path = cli
                .config
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("heartbit.toml"));
            if validate_config {
                daemon::validate_config_only(config_path).await
            } else {
                daemon::run_daemon(
                    config_path,
                    bind.as_deref(),
                    verbose,
                    cli.observability.as_deref(),
                )
                .await
            }
        }
        Some(Commands::Persona { sub }) => persona::run(sub).await,
        Some(Commands::Templates { action }) => run_template_command(action),
        Some(Commands::Skills { action }) => run_skill_command(action),
        Some(Commands::Init { template, output }) => run_init_command(&template, output.as_deref()),
        None => {
            // Backward-compatible: bare task args without subcommand
            let task_str = cli.task.join(" ");
            if task_str.is_empty() {
                bail!(
                    "Usage: heartbit [run|chat|serve|submit|status|approve|result] <args>\n       heartbit <task>  (shorthand for 'run')"
                );
            }
            run_standalone(
                cli.config.as_deref(),
                &task_str,
                false,
                false,
                cli.observability.as_deref(),
                None,
            )
            .await
        }
    }
}

async fn run_standalone(
    config_path: Option<&std::path::Path>,
    task: &str,
    approve: bool,
    verbose: bool,
    observability_flag: Option<&str>,
    trace_file: Option<PathBuf>,
) -> Result<()> {
    match config_path {
        Some(path) => run_from_config(path, task, approve, verbose, observability_flag).await,
        None => run_from_env(task, approve, verbose, observability_flag, trace_file).await,
    }
}

/// Build a `RetryConfig` from the provider config, if retry is configured.
pub(crate) fn retry_config_from(config: &HeartbitConfig) -> Option<RetryConfig> {
    config.provider.retry.as_ref().map(RetryConfig::from)
}

/// Build an on_retry callback that emits `AgentEvent::RetryAttempt` via the event callback.
pub(crate) fn build_on_retry(on_event: &Arc<OnEvent>) -> Arc<OnRetry> {
    let cb = on_event.clone();
    Arc::new(
        move |attempt: u32, max_retries: u32, delay_ms: u64, error_class: &str| {
            cb(AgentEvent::RetryAttempt {
                agent: "(provider)".into(), // Provider-level: specific agent name unavailable
                attempt,
                max_retries,
                delay_ms,
                error_class: error_class.to_string(),
            });
        },
    )
}

/// Build a base `BoxedProvider` from provider name, model, and prompt caching flag.
///
/// Does NOT apply retry or cascade — those are layered on by callers.
fn build_base_provider(
    provider_name: &str,
    model: &str,
    prompt_caching: bool,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Result<BoxedProvider> {
    // Special case: Anthropic has a native driver
    if provider_name == "anthropic" {
        let key = match api_key {
            Some(k) => k.to_string(),
            None => std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY env var required for anthropic provider")?,
        };
        let base = if prompt_caching {
            AnthropicProvider::with_prompt_caching(key, model)
        } else {
            AnthropicProvider::new(key, model)
        };
        return Ok(BoxedProvider::new(base));
    }

    // Special case: Gemini has a native driver
    if provider_name == "gemini" {
        let key = match api_key {
            Some(k) => k.to_string(),
            None => std::env::var("GEMINI_API_KEY")
                .context("GEMINI_API_KEY env var required for gemini provider")?,
        };
        let base = match base_url {
            Some(url) => heartbit::GeminiProvider::with_base_url(key, model, url),
            None => heartbit::GeminiProvider::new(key, model),
        };
        return Ok(BoxedProvider::new(base));
    }

    // Registry-based: look up provider info
    if let Some(info) = heartbit::get_provider(provider_name) {
        let url = base_url
            .map(|s| s.to_string())
            .or_else(|| info.base_url.map(|s| s.to_string()))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "provider '{provider_name}' requires a base_url (native driver not available)"
                )
            })?;

        let key = match api_key {
            Some(k) => k.to_string(),
            None => heartbit::resolve_api_key(provider_name, info)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        };

        let auth_style = if info.auth_required() {
            heartbit::AuthStyle::Bearer
        } else {
            heartbit::AuthStyle::None
        };

        return Ok(BoxedProvider::new(heartbit::OpenAiCompatProvider::new(
            key, model, url, auth_style,
        )));
    }

    // Unknown provider: if base_url is provided, try OpenAI-compatible
    if let Some(url) = base_url {
        let key = api_key.unwrap_or("").to_string();
        let auth_style = if key.is_empty() {
            heartbit::AuthStyle::None
        } else {
            heartbit::AuthStyle::Bearer
        };
        return Ok(BoxedProvider::new(heartbit::OpenAiCompatProvider::new(
            key, model, url, auth_style,
        )));
    }

    bail!(
        "Unknown provider: '{provider_name}'. Known providers: {}. \
         Or provide a base_url for custom OpenAI-compatible endpoints.",
        heartbit::known_llm_providers().join(", ")
    );
}

/// Build a `HeuristicGate` from cascade gate configuration.
fn build_gate_from_config(gate_config: &CascadeGateConfig) -> HeuristicGate {
    match gate_config {
        CascadeGateConfig::Heuristic {
            min_output_tokens,
            accept_tool_calls,
            escalate_on_max_tokens,
        } => HeuristicGate {
            min_output_tokens: *min_output_tokens,
            accept_tool_calls: *accept_tool_calls,
            escalate_on_max_tokens: *escalate_on_max_tokens,
            ..Default::default()
        },
    }
}

/// Build a `CorePathPolicy` from an optional `[sandbox]` config section.
///
/// Returns `Ok(None)` when `sandbox` is absent (no restriction applied).
/// Returns `Ok(Some(Arc<CorePathPolicy>))` when the sandbox config is present.
fn build_path_policy(
    sandbox: Option<&heartbit::SandboxConfig>,
) -> Result<Option<Arc<heartbit::CorePathPolicy>>> {
    let Some(sb) = sandbox else {
        return Ok(None);
    };
    let mut b = heartbit::CorePathPolicy::builder();
    for d in &sb.allowed_dirs {
        b = b.allow_dir(d);
    }
    for g in &sb.deny_globs {
        b = b.deny_glob(g);
    }
    let policy = b
        .build()
        .with_context(|| "failed to build CorePathPolicy from [sandbox] config")?;
    Ok(Some(Arc::new(policy)))
}

/// Wrap a provider with retry if configured, applying the `on_retry` callback.
fn wrap_with_retry(
    provider: BoxedProvider,
    retry: Option<RetryConfig>,
    on_retry: Option<Arc<OnRetry>>,
) -> BoxedProvider {
    match retry {
        Some(rc) => {
            let mut retrying = RetryingProvider::new(provider, rc);
            if let Some(cb) = on_retry {
                retrying = retrying.with_on_retry(cb);
            }
            BoxedProvider::new(retrying)
        }
        None => provider,
    }
}

/// Wrap a provider with cascade if configured.
///
/// Builds a `CascadingProvider` with the configured tiers (cheapest first)
/// and the main provider as the final (most expensive) tier.
#[allow(clippy::too_many_arguments)]
fn wrap_with_cascade(
    main_provider: BoxedProvider,
    main_model: &str,
    provider_name: &str,
    prompt_caching: bool,
    cascade: &CascadeConfig,
    retry: Option<RetryConfig>,
    on_retry: Option<&Arc<OnRetry>>,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Result<BoxedProvider> {
    let mut builder = CascadingProvider::builder();

    for tier_cfg in &cascade.tiers {
        let tier_base = build_base_provider(
            provider_name,
            &tier_cfg.model,
            prompt_caching,
            base_url,
            api_key,
        )?;
        let tier_provider = wrap_with_retry(tier_base, retry.clone(), on_retry.cloned());
        builder = builder.add_tier(&tier_cfg.model, tier_provider);
    }

    // Main model is the final (most expensive) tier
    builder = builder.add_tier(main_model, main_provider);
    builder = builder.gate(build_gate_from_config(&cascade.gate));

    let tier_labels: Vec<&str> = cascade.tiers.iter().map(|t| t.model.as_str()).collect();
    tracing::info!(tiers = ?tier_labels, final_tier = main_model, "cascade provider enabled");
    let cascading = builder.build().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(BoxedProvider::new(cascading))
}

/// Construct a type-erased LLM provider from config.
///
/// When `on_retry` is provided, `RetryingProvider` will invoke it before each retry,
/// enabling `AgentEvent::RetryAttempt` emission through the event callback system.
pub(crate) fn build_provider_from_config(
    config: &HeartbitConfig,
    on_retry: Option<Arc<OnRetry>>,
) -> Result<Arc<BoxedProvider>> {
    if config.provider.prompt_caching && config.provider.name != "anthropic" {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "prompt_caching is only effective with the 'anthropic' provider; \
                 ignored for '{}'",
                config.provider.name
            );
        });
    }
    let retry = retry_config_from(config);
    let base = build_base_provider(
        &config.provider.name,
        &config.provider.model,
        config.provider.prompt_caching,
        config.provider.base_url.as_deref(),
        config.provider.api_key.as_deref(),
    )?;

    // Cascade wrapping (tiers get their own retry; main provider is the final tier)
    if let Some(ref cascade) = config.provider.cascade
        && cascade.enabled
        && !cascade.tiers.is_empty()
    {
        let main_with_retry = wrap_with_retry(base, retry.clone(), on_retry.clone());
        let cascaded = wrap_with_cascade(
            main_with_retry,
            &config.provider.model,
            &config.provider.name,
            config.provider.prompt_caching,
            cascade,
            retry,
            on_retry.as_ref(),
            config.provider.base_url.as_deref(),
            config.provider.api_key.as_deref(),
        )?;
        return Ok(Arc::new(cascaded));
    }

    Ok(Arc::new(wrap_with_retry(base, retry, on_retry)))
}

/// Wrap a type-erased provider with a per-(tenant, provider) circuit breaker.
///
/// The layering is `CircuitBreakerProvider<BoxedProvider>` — the circuit sits
/// outside retry so a persistent failure after all retries trips the circuit.
/// The result is re-boxed so callers keep the `Arc<BoxedProvider>` signature.
pub(crate) fn wrap_with_circuit(
    provider: Arc<BoxedProvider>,
    tracker: Arc<heartbit::CircuitTracker>,
    provider_name: &str,
    scope: heartbit::TenantScope,
) -> Arc<BoxedProvider> {
    // BoxedProvider::from_arc converts Arc<BoxedProvider> → BoxedProvider (via ArcAdapter).
    // CircuitBreakerProvider<BoxedProvider> then wraps that, and we re-box for the caller.
    let inner = BoxedProvider::from_arc(provider);
    Arc::new(BoxedProvider::new(heartbit::CircuitBreakerProvider::new(
        inner,
        tracker,
        provider_name,
        scope,
    )))
}

/// Build a type-erased LLM provider for a per-agent override.
///
/// Reads the API key from the environment (same as `build_provider_from_config`).
/// Uses the global retry config if provided, otherwise wraps with retry defaults.
pub(crate) fn build_agent_provider(
    config: &heartbit::AgentProviderConfig,
    retry: Option<RetryConfig>,
    on_retry: Option<Arc<OnRetry>>,
) -> Result<Arc<BoxedProvider>> {
    if config.prompt_caching && config.name != "anthropic" {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "prompt_caching is only effective with the 'anthropic' provider; \
                 ignored for '{}'",
                config.name
            );
        });
    }
    let base = build_base_provider(
        &config.name,
        &config.model,
        config.prompt_caching,
        config.base_url.as_deref(),
        config.api_key.as_deref(),
    )?;

    // Cascade wrapping for per-agent provider
    if let Some(ref cascade) = config.cascade
        && cascade.enabled
        && !cascade.tiers.is_empty()
    {
        let main_with_retry = wrap_with_retry(base, retry.clone(), on_retry.clone());
        let cascaded = wrap_with_cascade(
            main_with_retry,
            &config.model,
            &config.name,
            config.prompt_caching,
            cascade,
            retry,
            on_retry.as_ref(),
            config.base_url.as_deref(),
            config.api_key.as_deref(),
        )?;
        return Ok(Arc::new(cascaded));
    }

    Ok(Arc::new(wrap_with_retry(base, retry, on_retry)))
}

async fn run_from_config(
    path: &std::path::Path,
    task: &str,
    approve: bool,
    verbose: bool,
    observability_flag: Option<&str>,
) -> Result<()> {
    let mut config = HeartbitConfig::from_file(path)
        .with_context(|| format!("failed to load config from {}", path.display()))?;

    // Resolve agent templates, skills, and variables before building agents.
    let variables = config.variables.clone();
    for i in 0..config.agents.len() {
        if config.agents[i].template.is_some()
            || !config.agents[i].skills.is_empty()
            || !config.agents[i].skill_dirs.is_empty()
        {
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

    let config_obs = config
        .telemetry
        .as_ref()
        .and_then(|t| t.observability_mode.as_deref());
    let mode = resolve_observability(observability_flag, config_obs, verbose);

    let on_event = if verbose {
        Some(event_callback())
    } else {
        None
    };
    let on_retry = on_event.as_ref().map(build_on_retry);
    let base_provider = build_provider_from_config(&config, on_retry)?;

    // B5b: build circuit tracker and wrap provider
    let circuit_tracker = Arc::new(heartbit::CircuitTracker::new(
        heartbit::CircuitConfig::from(&config.provider.circuit),
    ));
    let provider = wrap_with_circuit(
        base_provider,
        Arc::clone(&circuit_tracker),
        &config.provider.name,
        heartbit::TenantScope::default(),
    );

    // B5b: build tenant tracker from config
    let tenant_tracker = config
        .orchestrator
        .max_tokens_in_flight_per_tenant
        .map(|cap| Arc::new(heartbit::TenantTokenTracker::new(cap)));

    let on_text = streaming_callback();
    let on_approval = if approve {
        Some(approval_callback())
    } else {
        None
    };

    let ws_root = workspace_root_from_config(&config);
    let workspace_dir = provision_workspace(&ws_root);

    let mut output = RuntimeBuilder::new(provider, &config, task, on_text)
        .on_approval(on_approval)
        .on_event(on_event)
        .observability_mode(mode)
        .workspace_dir(workspace_dir)
        .dangerous_tools(true)
        .tenant_tracker(tenant_tracker)
        .circuit_tracker(Some(circuit_tracker))
        .run()
        .await?;
    // Cost estimate is only accurate when all agents use the same model.
    // With per-agent provider overrides, tokens from different models are
    // mixed in total_usage, making a single-model estimate incorrect.
    let has_overrides = config.agents.iter().any(|a| a.provider.is_some());
    if !has_overrides {
        output.estimated_cost_usd =
            heartbit::estimate_cost(&config.provider.model, &output.tokens_used);
    }
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
        // SECURITY (F-KB-1): single-tenant CLI uses the default scope (`""`).
        let kb_scope = heartbit::TenantScope::default();
        match source {
            KnowledgeSourceConfig::File { path } => {
                let path = std::path::Path::new(path);
                match loader::load_file(&*kb, &kb_scope, path, &chunk_config).await {
                    Ok(count) => {
                        tracing::info!(path = %path.display(), chunks = count, "indexed knowledge file");
                    }
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "failed to load knowledge file, skipping");
                    }
                }
            }
            KnowledgeSourceConfig::Glob { pattern } => {
                match loader::load_glob(&*kb, &kb_scope, pattern, &chunk_config).await {
                    Ok(count) => {
                        tracing::info!(pattern = %pattern, chunks = count, "indexed knowledge glob");
                    }
                    Err(e) => {
                        tracing::warn!(pattern = %pattern, error = %e, "failed to load knowledge glob, skipping");
                    }
                }
            }
            KnowledgeSourceConfig::Url { url } => {
                match loader::load_url(&*kb, &kb_scope, url, &chunk_config).await {
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

    let total = kb
        .chunk_count(&heartbit::TenantScope::default())
        .await
        .unwrap_or(0);
    tracing::info!(total_chunks = total, "knowledge base loaded");

    Ok(kb)
}

pub(crate) struct RuntimeBuilder<'a> {
    provider: Arc<BoxedProvider>,
    config: &'a HeartbitConfig,
    task: &'a str,
    on_text: Arc<OnText>,
    on_approval: Option<Arc<OnApproval>>,
    on_event: Option<Arc<OnEvent>>,
    observability_mode: ObservabilityMode,
    story_id: Option<&'a str>,
    on_question: Option<Arc<OnQuestion>>,
    external_memory: Option<Arc<dyn Memory>>,
    workspace_dir: Option<std::path::PathBuf>,
    pre_loaded_tools: Option<&'a HashMap<String, Vec<Arc<dyn Tool>>>>,
    content_blocks: Option<Vec<heartbit::ContentBlock>>,
    guardrails: Vec<Arc<dyn heartbit::Guardrail>>,
    memory_confidentiality_cap: Option<heartbit::Confidentiality>,
    memory_default_confidentiality: Option<heartbit::Confidentiality>,
    audit_user_id: Option<&'a str>,
    audit_tenant_id: Option<&'a str>,
    allow_shared_write: bool,
    dangerous_tools: bool,
    /// B5b: optional tenant token tracker for in-flight token enforcement
    tenant_tracker: Option<Arc<heartbit::TenantTokenTracker>>,
    /// B5b: shared circuit tracker — wraps per-agent provider overrides with circuit breaker
    circuit_tracker: Option<Arc<heartbit::CircuitTracker>>,
}

impl<'a> RuntimeBuilder<'a> {
    pub(crate) fn new(
        provider: Arc<BoxedProvider>,
        config: &'a HeartbitConfig,
        task: &'a str,
        on_text: Arc<OnText>,
    ) -> Self {
        Self {
            provider,
            config,
            task,
            on_text,
            on_approval: None,
            on_event: None,
            observability_mode: ObservabilityMode::Production,
            story_id: None,
            on_question: None,
            external_memory: None,
            workspace_dir: None,
            pre_loaded_tools: None,
            content_blocks: None,
            guardrails: vec![],
            memory_confidentiality_cap: None,
            memory_default_confidentiality: None,
            audit_user_id: None,
            audit_tenant_id: None,
            allow_shared_write: true,
            dangerous_tools: false,
            tenant_tracker: None,
            circuit_tracker: None,
        }
    }

    pub(crate) fn on_approval(mut self, v: Option<Arc<OnApproval>>) -> Self {
        self.on_approval = v;
        self
    }

    pub(crate) fn on_event(mut self, v: Option<Arc<OnEvent>>) -> Self {
        self.on_event = v;
        self
    }

    pub(crate) fn observability_mode(mut self, v: ObservabilityMode) -> Self {
        self.observability_mode = v;
        self
    }

    pub(crate) fn story_id(mut self, v: Option<&'a str>) -> Self {
        self.story_id = v;
        self
    }

    #[allow(dead_code)]
    pub(crate) fn on_question(mut self, v: Option<Arc<OnQuestion>>) -> Self {
        self.on_question = v;
        self
    }

    pub(crate) fn external_memory(mut self, v: Option<Arc<dyn Memory>>) -> Self {
        self.external_memory = v;
        self
    }

    pub(crate) fn workspace_dir(mut self, v: Option<std::path::PathBuf>) -> Self {
        self.workspace_dir = v;
        self
    }

    pub(crate) fn pre_loaded_tools(
        mut self,
        v: Option<&'a HashMap<String, Vec<Arc<dyn Tool>>>>,
    ) -> Self {
        self.pre_loaded_tools = v;
        self
    }

    #[allow(dead_code)]
    pub(crate) fn content_blocks(mut self, v: Option<Vec<heartbit::ContentBlock>>) -> Self {
        self.content_blocks = v;
        self
    }

    pub(crate) fn guardrails(mut self, v: Vec<Arc<dyn heartbit::Guardrail>>) -> Self {
        self.guardrails = v;
        self
    }

    pub(crate) fn memory_confidentiality_cap(
        mut self,
        v: Option<heartbit::Confidentiality>,
    ) -> Self {
        self.memory_confidentiality_cap = v;
        self
    }

    #[allow(dead_code)]
    pub(crate) fn memory_default_confidentiality(
        mut self,
        v: Option<heartbit::Confidentiality>,
    ) -> Self {
        self.memory_default_confidentiality = v;
        self
    }

    pub(crate) fn audit_user_id(mut self, v: Option<&'a str>) -> Self {
        self.audit_user_id = v;
        self
    }

    pub(crate) fn audit_tenant_id(mut self, v: Option<&'a str>) -> Self {
        self.audit_tenant_id = v;
        self
    }

    pub(crate) fn allow_shared_write(mut self, v: bool) -> Self {
        self.allow_shared_write = v;
        self
    }

    pub(crate) fn dangerous_tools(mut self, v: bool) -> Self {
        self.dangerous_tools = v;
        self
    }

    pub(crate) fn tenant_tracker(mut self, v: Option<Arc<heartbit::TenantTokenTracker>>) -> Self {
        self.tenant_tracker = v;
        self
    }

    pub(crate) fn circuit_tracker(mut self, v: Option<Arc<heartbit::CircuitTracker>>) -> Self {
        self.circuit_tracker = v;
        self
    }

    pub(crate) async fn run(self) -> Result<AgentOutput> {
        let on_retry = self.on_event.as_ref().map(build_on_retry);

        // ── User identity context for system prompt ──
        // When user context is present, append identity block so the agent knows who it serves.
        let user_context_suffix = match (self.audit_user_id, self.audit_tenant_id) {
            (Some(uid), Some(tid)) => Some(format!(
                "\n---\nYou are operating on behalf of **{uid}** in organization **{tid}**.\nKeep this user's information private. Do not share their data with other users."
            )),
            _ => None,
        };

        // Build path policy from [sandbox] config if present.
        let path_policy = build_path_policy(self.config.sandbox.as_ref())?;

        // Create shared built-in tools (FileTracker, TodoStore shared across all agents).
        let builtins = {
            let mut btc = BuiltinToolsConfig::default();
            btc.on_question = self
                .on_question
                .clone()
                .or_else(|| Some(question_callback()));
            btc.workspace = self.workspace_dir.clone();
            btc.dangerous_tools = self.dangerous_tools;
            btc.path_policy = path_policy.clone();
            #[cfg(all(target_os = "linux", feature = "sandbox"))]
            if let Some(ref pp) = path_policy {
                btc.sandbox_policy =
                    Some(heartbit::SandboxPolicy::from_path_policy(Arc::clone(pp)));
            }
            builtin_tools(btc)
        };

        // ── Shared memory + story context pre-loading ──
        let mut task_text = self.task.to_string();
        let base_memory = if let Some(ext) = self.external_memory {
            Some(ext)
        } else if let Some(ref memory_config) = self.config.memory {
            Some(create_memory_store(memory_config).await?)
        } else {
            None
        };
        if let Some(ref base_memory) = base_memory
            && let Some(sid) = self.story_id
        {
            let scope = TenantScope::from_audit_fields(self.audit_tenant_id, self.audit_user_id);
            let prior = base_memory
                .recall(
                    &scope,
                    MemoryQuery {
                        limit: 10,
                        agent_prefix: Some(sid.to_string()),
                        ..Default::default()
                    },
                )
                .await
                .unwrap_or_default();

            if !prior.is_empty() {
                let ctx: String = prior
                    .iter()
                    .map(|e| format!("- [{}] {}", e.category, e.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                task_text =
                    format!("## Prior story context\n{ctx}\n\n## Current task\n{task_text}");
                tracing::info!(
                    story_id = sid,
                    prior_memories = prior.len(),
                    "loaded story context into task"
                );
            }
        }

        // When story context was prepended and content_blocks exist, update the
        // text block so the LLM sees the enriched context alongside images.
        let mut content_blocks = self.content_blocks;
        if task_text != self.task
            && let Some(ref mut blocks) = content_blocks
        {
            // Replace the first Text block with enriched task_text, or prepend one
            let updated = blocks.iter_mut().any(|b| {
                if let heartbit::ContentBlock::Text { text } = b {
                    *text = task_text.clone();
                    true
                } else {
                    false
                }
            });
            if !updated {
                blocks.insert(
                    0,
                    heartbit::ContentBlock::Text {
                        text: task_text.clone(),
                    },
                );
            }
        }

        // ── Routing decision ──
        // Replace static agent-count check with complexity-based routing.
        // Three modes: Auto (heuristic + capability match), AlwaysOrchestrate, SingleAgent.
        let routing_mode = heartbit::resolve_routing_mode(self.config.orchestrator.routing);

        // Event collector for Tier 3 escalation decisions.
        // When routing in auto mode with escalation enabled, we need to capture
        // events from the single-agent run to check for doom loops / compactions.
        let event_collector: Arc<std::sync::Mutex<Vec<heartbit::AgentEvent>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        let (route_to_single, selected_agent_index) = match routing_mode {
            heartbit::RoutingMode::AlwaysOrchestrate => (false, None),
            heartbit::RoutingMode::SingleAgent => (true, Some(0)),
            heartbit::RoutingMode::Auto => {
                if self.config.agents.len() == 1 {
                    // Single agent configured → skip analysis, zero overhead
                    (true, Some(0))
                } else if self.config.agents.is_empty() {
                    (false, None)
                } else {
                    // Build capability list from agent descriptions.
                    // Tool names aren't known until MCP connection, so we rely on
                    // the agent description for domain extraction.
                    let capabilities: Vec<heartbit::AgentCapability> = self
                        .config
                        .agents
                        .iter()
                        .map(|a| {
                            heartbit::AgentCapability::from_config(&a.name, &a.description, &[])
                        })
                        .collect();
                    let analyzer = heartbit::TaskComplexityAnalyzer::new(&capabilities);
                    let (decision, signals) = analyzer.analyze(&task_text);

                    // Emit TaskRouted event
                    let (is_single, idx) = match &decision {
                        heartbit::RoutingDecision::SingleAgent { agent_index, .. } => {
                            (true, Some(*agent_index))
                        }
                        heartbit::RoutingDecision::Orchestrate { .. } => (false, None),
                    };
                    if let Some(ref on_ev) = self.on_event {
                        let selected_name = idx.map(|i| self.config.agents[i].name.clone());
                        on_ev(heartbit::AgentEvent::TaskRouted {
                            decision: if is_single {
                                "single_agent".into()
                            } else {
                                "orchestrate".into()
                            },
                            reason: match &decision {
                                heartbit::RoutingDecision::SingleAgent { reason, .. }
                                | heartbit::RoutingDecision::Orchestrate { reason } => {
                                    (*reason).to_string()
                                }
                            },
                            selected_agent: selected_name,
                            complexity_score: signals.complexity_score,
                            escalated: false,
                        });
                    }
                    tracing::info!(
                        routing = if is_single { "single_agent" } else { "orchestrate" },
                        score = signals.complexity_score,
                        domains = ?signals.domain_signals,
                        "task routing decision"
                    );
                    (is_single, idx)
                }
            }
        };

        // Force single-agent routing when multimodal content is present.
        // The orchestrator only supports text — images would be silently lost.
        let (route_to_single, selected_agent_index) =
            if content_blocks.is_some() && !route_to_single {
                tracing::info!(
                    "forcing single-agent routing: multimodal content not supported by orchestrator"
                );
                (true, Some(0))
            } else {
                (route_to_single, selected_agent_index)
            };

        if route_to_single {
            let agent_index = selected_agent_index.unwrap_or(0);
            if agent_index >= self.config.agents.len() {
                anyhow::bail!(
                    "routing selected agent index {agent_index} but only {} agents configured",
                    self.config.agents.len()
                );
            }
            let agent = &self.config.agents[agent_index];

            // Load tools: builtins (filtered per-agent) + (cached or fresh) MCP + A2A
            let mut tools = filter_builtins_for_agent(&builtins, agent.builtin_tools.as_deref());
            match self.pre_loaded_tools.and_then(|c| c.get(&agent.name)) {
                Some(cached) => tools.extend(cached.iter().cloned()),
                None => {
                    tools.extend(load_mcp_tools(&agent.name, &agent.mcp_servers).await);
                    tools.extend(load_a2a_tools(&agent.name, &agent.a2a_agents).await);
                }
            }

            // Provider: agent-specific or global.
            // When the agent has its own provider config, wrap it with the circuit breaker
            // so per-agent overrides are not exempt from failure tracking.
            let agent_provider: Arc<BoxedProvider> = match &agent.provider {
                Some(p) => {
                    let built =
                        build_agent_provider(p, retry_config_from(self.config), on_retry.clone())?;
                    if let Some(ref ct) = self.circuit_tracker {
                        let scope = heartbit::TenantScope::from_audit_fields(
                            self.audit_tenant_id,
                            self.audit_user_id,
                        );
                        wrap_with_circuit(built, Arc::clone(ct), &p.name, scope)
                    } else {
                        built
                    }
                }
                None => Arc::clone(&self.provider),
            };

            // Settings: agent overrides, falling back to orchestrator defaults
            let max_turns = agent
                .max_turns
                .unwrap_or(self.config.orchestrator.max_turns);
            let max_tokens = agent
                .max_tokens
                .unwrap_or(self.config.orchestrator.max_tokens);

            let agent_provider_for_judge = Arc::clone(&agent_provider);
            let system_prompt = if let Some(ref suffix) = user_context_suffix {
                format!("{}{suffix}", agent.system_prompt)
            } else {
                agent.system_prompt.clone()
            };
            let mut rb = AgentRunner::builder(agent_provider)
                .name(&agent.name)
                .system_prompt(&system_prompt)
                .tools(tools)
                .max_turns(max_turns)
                .max_tokens(max_tokens)
                .on_text(Arc::clone(&self.on_text))
                .observability_mode(self.observability_mode);

            // Wire audit user context and delegation chain for multi-tenant enrichment
            if let Some(uid) = self.audit_user_id
                && let Some(tid) = self.audit_tenant_id
            {
                rb = rb
                    .audit_user_context(uid, tid)
                    .audit_delegation_chain(vec![agent.name.clone()]);
            }

            // B5b: wire tenant token tracker into single-agent runner
            if let Some(ref tracker) = self.tenant_tracker {
                rb = rb.tenant_tracker(tracker.clone());
            }

            // Context strategy: agent-level, then orchestrator-level fallback
            match &agent.context_strategy {
                Some(ContextStrategyConfig::SlidingWindow { max_tokens }) => {
                    rb = rb.context_strategy(ContextStrategy::SlidingWindow {
                        max_tokens: *max_tokens,
                    });
                }
                Some(ContextStrategyConfig::Summarize { threshold }) => {
                    rb = rb.summarize_threshold(*threshold);
                }
                Some(ContextStrategyConfig::Unlimited) | None => {
                    if let Some(st) = agent.summarize_threshold {
                        rb = rb.summarize_threshold(st);
                    } else {
                        match &self.config.orchestrator.context_strategy {
                            Some(ContextStrategyConfig::SlidingWindow { max_tokens }) => {
                                rb = rb.context_strategy(ContextStrategy::SlidingWindow {
                                    max_tokens: *max_tokens,
                                });
                            }
                            Some(ContextStrategyConfig::Summarize { threshold }) => {
                                rb = rb.summarize_threshold(*threshold);
                            }
                            _ => {
                                if let Some(threshold) =
                                    self.config.orchestrator.summarize_threshold
                                {
                                    rb = rb.summarize_threshold(threshold);
                                }
                            }
                        }
                    }
                }
            }

            // Timeouts: agent-level, then orchestrator-level fallback
            let tool_timeout = agent
                .tool_timeout_seconds
                .or(self.config.orchestrator.tool_timeout_seconds)
                .map(std::time::Duration::from_secs);
            if let Some(t) = tool_timeout {
                rb = rb.tool_timeout(t);
            }
            let max_tool_output = agent
                .max_tool_output_bytes
                .or(self.config.orchestrator.max_tool_output_bytes);
            if let Some(m) = max_tool_output {
                rb = rb.max_tool_output_bytes(m);
            }
            let run_timeout = agent
                .run_timeout_seconds
                .or(self.config.orchestrator.run_timeout_seconds)
                .map(std::time::Duration::from_secs);
            if let Some(t) = run_timeout {
                rb = rb.run_timeout(t);
            }

            // Reasoning / reflection / compression: agent-level, then orchestrator-level
            let reasoning_effort = agent.reasoning_effort.as_deref().or(self
                .config
                .orchestrator
                .reasoning_effort
                .as_deref());
            if let Some(effort) = reasoning_effort {
                rb = rb.reasoning_effort(heartbit::config::parse_reasoning_effort(effort)?);
            }
            let reflection = agent
                .enable_reflection
                .or(self.config.orchestrator.enable_reflection);
            if let Some(true) = reflection {
                rb = rb.enable_reflection(true);
            }
            let compression = agent
                .tool_output_compression_threshold
                .or(self.config.orchestrator.tool_output_compression_threshold);
            if let Some(t) = compression {
                rb = rb.tool_output_compression_threshold(t);
            }
            let max_tools = agent
                .max_tools_per_turn
                .or(self.config.orchestrator.max_tools_per_turn);
            if let Some(m) = max_tools {
                rb = rb.max_tools_per_turn(m);
            }
            let tool_profile_str = agent.tool_profile.as_deref().or(self
                .config
                .orchestrator
                .tool_profile
                .as_deref());
            if let Some(p) = tool_profile_str {
                rb = rb.tool_profile(heartbit::parse_tool_profile(p)?);
            }
            let doom_loop = agent
                .max_identical_tool_calls
                .or(self.config.orchestrator.max_identical_tool_calls);
            if let Some(m) = doom_loop {
                rb = rb.max_identical_tool_calls(m);
            }
            let fuzzy_doom = agent
                .max_fuzzy_identical_tool_calls
                .or(self.config.orchestrator.max_fuzzy_identical_tool_calls);
            if let Some(m) = fuzzy_doom {
                rb = rb.max_fuzzy_identical_tool_calls(m);
            }
            let tool_call_cap = agent
                .max_tool_calls_per_turn
                .or(self.config.orchestrator.max_tool_calls_per_turn);
            if let Some(cap) = tool_call_cap {
                rb = rb.max_tool_calls_per_turn(cap);
            }
            if let Some(budget) = agent.max_total_tokens {
                rb = rb.max_total_tokens(budget);
            }
            if let Some(size) = agent.response_cache_size {
                rb = rb.response_cache_size(size);
            }
            if let Some(ref am) = agent.audit_mode {
                let mode = match am.as_str() {
                    "metadata_only" => heartbit::AuditMode::MetadataOnly,
                    _ => heartbit::AuditMode::Full,
                };
                rb = rb.audit_mode(mode);
            }

            // Session prune config
            if let Some(ref sp) = agent.session_prune {
                rb = rb.session_prune_config(heartbit::SessionPruneConfig {
                    keep_recent_n: sp.keep_recent_n,
                    pruned_tool_result_max_bytes: sp.pruned_tool_result_max_bytes,
                    preserve_task: sp.preserve_task,
                });
            }
            if let Some(true) = agent.recursive_summarization {
                rb = rb.enable_recursive_summarization(true);
            }
            if let Some(t) = agent.reflection_threshold {
                rb = rb.reflection_threshold(t);
            }
            if let Some(true) = agent.consolidate_on_exit {
                rb = rb.consolidate_on_exit(true);
            }

            // Callbacks
            if let Some(ref cb) = self.on_approval {
                rb = rb.on_approval(Arc::clone(cb));
            }
            // Wrap on_event to also collect events for escalation decisions.
            {
                let collector = Arc::clone(&event_collector);
                let inner_cb = self.on_event.clone();
                let collecting_cb: Arc<heartbit::OnEvent> =
                    Arc::new(move |event: heartbit::AgentEvent| {
                        {
                            let mut events = collector.lock().expect("event collector lock");
                            events.push(event.clone());
                        }
                        if let Some(ref cb) = inner_cb {
                            cb(event);
                        }
                    });
                rb = rb.on_event(collecting_cb);
            }

            // Permission rules + learned permissions
            {
                let mut ruleset = heartbit::PermissionRuleset::new(self.config.permissions.clone());
                if let Some(ref learned) = load_learned_permissions() {
                    let guard = learned.lock().expect("learned permissions lock");
                    ruleset.append_rules(guard.rules());
                    rb = rb.learned_permissions(learned.clone());
                }
                if !ruleset.is_empty() {
                    rb = rb.permission_rules(ruleset);
                }
            }

            // Instruction text (HEARTBIT.md)
            if let Some(text) = load_instruction_text() {
                rb = rb.instruction_text(text);
            }

            // Workspace
            if let Some(ref ws) = self.workspace_dir {
                rb = rb.workspace(ws.clone());
            }

            // Memory with namespace
            if let Some(ref base_memory) = base_memory {
                let agent_ns = match self.story_id {
                    Some(sid) => format!("{sid}:{}", agent.name),
                    None => agent.name.clone(),
                };
                let mut ns = NamespacedMemory::new(Arc::clone(base_memory), agent_ns)
                    .with_max_confidentiality(self.memory_confidentiality_cap);
                if let Some(default) = self.memory_default_confidentiality {
                    ns = ns.with_default_store_confidentiality(default);
                }
                let namespaced: Arc<dyn Memory> = Arc::new(ns);
                rb = rb.memory(namespaced);
            }

            // Knowledge base
            if let Some(ref knowledge_config) = self.config.knowledge {
                let kb = load_knowledge_base(knowledge_config).await?;
                rb = rb.knowledge(kb);
            }

            // LSP integration
            if let Some(ref lsp_config) = self.config.lsp
                && lsp_config.enabled
            {
                let workspace_root = std::env::current_dir().unwrap_or_default();
                rb = rb.lsp_manager(Arc::new(heartbit::LspManager::new(workspace_root)));
            }

            // Wire guardrails: external (e.g., SensorSecurityGuardrail) + config-based
            {
                let mut all_guardrails = self.guardrails.clone();
                // Per-agent guardrails config overrides top-level
                let guardrails_config = agent
                    .guardrails
                    .as_ref()
                    .or(self.config.guardrails.as_ref());
                if let Some(gc) = guardrails_config {
                    let judge_provider = gc
                        .llm_judge
                        .as_ref()
                        .map(|_| Arc::clone(&agent_provider_for_judge));
                    all_guardrails.extend(gc.build_with_judge(judge_provider)?);
                }
                if !all_guardrails.is_empty() {
                    rb = rb.guardrails(all_guardrails);
                }
            }

            tracing::info!(
                agent = %agent.name,
                max_turns,
                routing = ?routing_mode,
                "single-agent path: bypassing orchestrator"
            );

            let runner = rb.build()?;
            let run_result = if let Some(blocks) = content_blocks {
                runner.execute_with_content(blocks).await
            } else {
                runner.execute(&task_text).await
            };
            match run_result {
                Ok(output) => return Ok(output),
                Err(err) => {
                    // Tier 3: escalation — if enabled and the failure warrants it,
                    // fall through to orchestrator with partial context.
                    let collected_events = event_collector
                        .lock()
                        .expect("event collector lock")
                        .clone();
                    let should_esc = self.config.orchestrator.escalation
                        && self.config.agents.len() > 1
                        && heartbit::should_escalate(&err, &collected_events);
                    if should_esc {
                        if let Some(ref on_ev) = self.on_event {
                            on_ev(heartbit::AgentEvent::TaskRouted {
                                decision: "orchestrate".into(),
                                reason: format!("escalated after single-agent failure: {err}"),
                                selected_agent: None,
                                complexity_score: 0.0,
                                escalated: true,
                            });
                        }
                        tracing::warn!(
                            error = %err,
                            "single-agent failed, escalating to orchestrator"
                        );
                        // Fall through to orchestrator path below
                    } else {
                        return Err(err.into());
                    }
                }
            }
        }

        // ── Multi-agent orchestrator path ──

        let provider_for_judge = Arc::clone(&self.provider);
        let mut builder = Orchestrator::builder(self.provider)
            .max_turns(self.config.orchestrator.max_turns)
            .max_tokens(self.config.orchestrator.max_tokens)
            .on_text(self.on_text)
            .observability_mode(self.observability_mode)
            .allow_shared_write(self.allow_shared_write);

        // Wire audit user context and delegation chain for multi-tenant enrichment
        if let Some(uid) = self.audit_user_id
            && let Some(tid) = self.audit_tenant_id
        {
            builder = builder
                .audit_user_context(uid, tid)
                .audit_delegation_chain(vec!["heartbit-daemon".into()]);
        }

        // B5b: wire tenant token tracker into orchestrator
        if let Some(tracker) = self.tenant_tracker.clone() {
            builder = builder.tenant_tracker(tracker);
        }

        // Wire guardrails: external (e.g., SensorSecurityGuardrail) + config-based
        {
            let mut all_guardrails = self.guardrails.clone();
            if let Some(gc) = &self.config.guardrails {
                let judge_provider = gc
                    .llm_judge
                    .as_ref()
                    .map(|_| Arc::clone(&provider_for_judge));
                all_guardrails.extend(gc.build_with_judge(judge_provider)?);
            }
            if !all_guardrails.is_empty() {
                builder = builder.guardrails(all_guardrails);
            }
        }

        // Wire workspace directory for sub-agents
        if let Some(ref ws) = self.workspace_dir {
            builder = builder.workspace(ws.clone());
        }

        // Wire squad formation opt-out from config
        if let Some(enable) = self.config.orchestrator.enable_squads {
            builder = builder.enable_squads(enable);
        }
        // Wire dispatch mode from config
        if let Some(mode) = self.config.orchestrator.dispatch_mode {
            builder = builder.dispatch_mode(mode);
        }
        // Wire multi-agent collaboration prompt from config
        if let Some(enabled) = self.config.orchestrator.multi_agent_prompt {
            builder = builder.multi_agent_prompt(enabled);
        }

        // Wire dynamic agent spawning from config
        if let Some(spawn_cfg) = self.config.orchestrator.spawn.clone() {
            builder = builder.spawn_config(spawn_cfg, builtins.clone());
        }

        // Wire orchestrator-level reasoning effort from config
        if let Some(ref effort) = self.config.orchestrator.reasoning_effort {
            builder = builder.reasoning_effort(heartbit::config::parse_reasoning_effort(effort)?);
        }
        // Wire orchestrator-level reflection from config
        if let Some(true) = self.config.orchestrator.enable_reflection {
            builder = builder.enable_reflection(true);
        }
        // Wire orchestrator-level tool output compression threshold from config
        if let Some(threshold) = self.config.orchestrator.tool_output_compression_threshold {
            builder = builder.tool_output_compression_threshold(threshold);
        }
        // Wire orchestrator-level max tools per turn from config
        if let Some(max) = self.config.orchestrator.max_tools_per_turn {
            builder = builder.max_tools_per_turn(max);
        }
        // Wire orchestrator-level doom loop detection from config
        if let Some(max) = self.config.orchestrator.max_identical_tool_calls {
            builder = builder.max_identical_tool_calls(max);
        }
        if let Some(max) = self.config.orchestrator.max_fuzzy_identical_tool_calls {
            builder = builder.max_fuzzy_identical_tool_calls(max);
        }
        if let Some(cap) = self.config.orchestrator.max_tool_calls_per_turn {
            builder = builder.max_tool_calls_per_turn(cap);
        }
        // Wire permission rules from config + learned permissions
        {
            let mut ruleset = heartbit::PermissionRuleset::new(self.config.permissions.clone());
            if let Some(ref learned) = load_learned_permissions() {
                let guard = learned.lock().expect("learned permissions lock");
                ruleset.append_rules(guard.rules());
                builder = builder.learned_permissions(learned.clone());
            }
            if !ruleset.is_empty() {
                builder = builder.permission_rules(ruleset);
            }
        }
        // Wire hierarchical instruction files (HEARTBIT.md)
        if let Some(text) = load_instruction_text() {
            builder = builder.instruction_text(text);
        }

        // Wire orchestrator-level context management.
        match &self.config.orchestrator.context_strategy {
            Some(ContextStrategyConfig::SlidingWindow { max_tokens }) => {
                builder = builder.context_strategy(ContextStrategy::SlidingWindow {
                    max_tokens: *max_tokens,
                });
            }
            Some(ContextStrategyConfig::Summarize { threshold }) => {
                builder = builder.summarize_threshold(*threshold);
            }
            Some(ContextStrategyConfig::Unlimited) | None => {
                if let Some(threshold) = self.config.orchestrator.summarize_threshold {
                    builder = builder.summarize_threshold(threshold);
                }
            }
        }
        if let Some(secs) = self.config.orchestrator.tool_timeout_seconds {
            builder = builder.tool_timeout(std::time::Duration::from_secs(secs));
        }
        if let Some(max) = self.config.orchestrator.max_tool_output_bytes {
            builder = builder.max_tool_output_bytes(max);
        }
        if let Some(secs) = self.config.orchestrator.run_timeout_seconds {
            builder = builder.run_timeout(std::time::Duration::from_secs(secs));
        }

        if let Some(cb) = self.on_approval {
            builder = builder.on_approval(cb);
        }
        if let Some(cb) = self.on_event {
            builder = builder.on_event(cb);
        }

        for agent in &self.config.agents {
            let mut tools = filter_builtins_for_agent(&builtins, agent.builtin_tools.as_deref());
            match self.pre_loaded_tools.and_then(|c| c.get(&agent.name)) {
                Some(cached) => tools.extend(cached.iter().cloned()),
                None => {
                    tools.extend(load_mcp_tools(&agent.name, &agent.mcp_servers).await);
                    tools.extend(load_a2a_tools(&agent.name, &agent.a2a_agents).await);
                }
            }
            let (ctx_strategy, summarize_threshold) = match &agent.context_strategy {
                Some(ContextStrategyConfig::SlidingWindow { max_tokens }) => (
                    Some(ContextStrategy::SlidingWindow {
                        max_tokens: *max_tokens,
                    }),
                    None,
                ),
                Some(ContextStrategyConfig::Summarize { threshold }) => (None, Some(*threshold)),
                Some(ContextStrategyConfig::Unlimited) | None => (None, agent.summarize_threshold),
            };

            // Build per-agent provider override if configured.
            // Per-agent providers inherit the global retry config.
            // Wrap with circuit breaker so overrides are not exempt from failure tracking.
            let agent_provider = match &agent.provider {
                Some(p) => {
                    let built =
                        build_agent_provider(p, retry_config_from(self.config), on_retry.clone())?;
                    let wrapped = if let Some(ref ct) = self.circuit_tracker {
                        let scope = heartbit::TenantScope::from_audit_fields(
                            self.audit_tenant_id,
                            self.audit_user_id,
                        );
                        wrap_with_circuit(built, Arc::clone(ct), &p.name, scope)
                    } else {
                        built
                    };
                    Some(wrapped)
                }
                None => None,
            };

            builder = builder.sub_agent_full(SubAgentConfig {
                name: agent.name.clone(),
                description: agent.description.clone(),
                system_prompt: if let Some(ref suffix) = user_context_suffix {
                    format!("{}{suffix}", agent.system_prompt)
                } else {
                    agent.system_prompt.clone()
                },
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
                run_timeout: agent
                    .run_timeout_seconds
                    .map(std::time::Duration::from_secs),
                guardrails: {
                    let mut agent_guardrails = self.guardrails.clone();
                    // Per-agent guardrails config overrides top-level
                    let gc = agent
                        .guardrails
                        .as_ref()
                        .or(self.config.guardrails.as_ref());
                    if let Some(gc) = gc {
                        let judge_provider = gc.llm_judge.as_ref().map(|_| {
                            agent_provider
                                .as_ref()
                                .map_or_else(|| Arc::clone(&provider_for_judge), Arc::clone)
                        });
                        agent_guardrails.extend(gc.build_with_judge(judge_provider)?);
                    }
                    agent_guardrails
                },
                provider: agent_provider,
                reasoning_effort: agent
                    .reasoning_effort
                    .as_deref()
                    .map(heartbit::config::parse_reasoning_effort)
                    .transpose()?,
                enable_reflection: agent.enable_reflection,
                tool_output_compression_threshold: agent.tool_output_compression_threshold,
                max_tools_per_turn: agent.max_tools_per_turn,
                tool_profile: agent
                    .tool_profile
                    .as_deref()
                    .map(heartbit::parse_tool_profile)
                    .transpose()?,
                max_identical_tool_calls: agent.max_identical_tool_calls,
                max_fuzzy_identical_tool_calls: agent.max_fuzzy_identical_tool_calls,
                max_tool_calls_per_turn: agent
                    .max_tool_calls_per_turn
                    .or(self.config.orchestrator.max_tool_calls_per_turn),
                session_prune_config: agent.session_prune.as_ref().map(|sp| {
                    heartbit::SessionPruneConfig {
                        keep_recent_n: sp.keep_recent_n,
                        pruned_tool_result_max_bytes: sp.pruned_tool_result_max_bytes,
                        preserve_task: sp.preserve_task,
                    }
                }),
                enable_recursive_summarization: agent.recursive_summarization,
                reflection_threshold: agent.reflection_threshold,
                consolidate_on_exit: agent.consolidate_on_exit,
                workspace: None,
                max_total_tokens: agent.max_total_tokens,
                audit_trail: None,
                audit_user_id: self.audit_user_id.map(String::from),
                audit_tenant_id: self.audit_tenant_id.map(String::from),
                audit_delegation_chain: Vec::new(),
                context: Default::default(),
            });
        }

        // Wire shared memory (story context already pre-loaded above)
        if let Some(base_memory) = base_memory {
            if let Some(sid) = self.story_id {
                builder = builder.memory_namespace_prefix(sid);
            }
            builder = builder.shared_memory(base_memory);
        }

        // Always attach an in-memory blackboard for cross-agent coordination
        let blackboard: Arc<dyn Blackboard> = Arc::new(InMemoryBlackboard::new());
        builder = builder.blackboard(blackboard);

        // Wire knowledge base if configured
        if let Some(ref knowledge_config) = self.config.knowledge {
            let kb = load_knowledge_base(knowledge_config).await?;
            builder = builder.knowledge(kb);
        }

        // Wire LSP integration if configured
        if let Some(ref lsp_config) = self.config.lsp
            && lsp_config.enabled
        {
            let workspace_root = std::env::current_dir().unwrap_or_default();
            builder = builder.lsp_manager(Arc::new(heartbit::LspManager::new(workspace_root)));
        }

        let mut orchestrator = builder.build()?;
        let output = orchestrator.run(&task_text).await?;
        Ok(output)
    }
}

/// Filter builtins per-agent based on an optional allowlist.
/// When `None`, all builtins are returned (backward compatible).
/// When `Some(list)`, only named tools are included. Empty list → no builtins.
fn filter_builtins_for_agent(
    builtins: &[Arc<dyn Tool>],
    allowlist: Option<&[String]>,
) -> Vec<Arc<dyn Tool>> {
    match allowlist {
        Some([]) => Vec::new(),
        Some(allowed) => {
            let set: std::collections::HashSet<&str> = allowed.iter().map(|s| s.as_str()).collect();
            builtins
                .iter()
                .filter(|t| set.contains(t.definition().name.as_str()))
                .cloned()
                .collect()
        }
        None => builtins.to_vec(),
    }
}

/// Connect to MCP servers and collect tools. Failures are logged and skipped.
pub(crate) async fn load_mcp_tools(
    agent_name: &str,
    mcp_servers: &[McpServerEntry],
) -> Vec<Arc<dyn Tool>> {
    let mut tools = Vec::new();
    for entry in mcp_servers {
        let server_label = entry.display_name();
        tracing::info!(agent = %agent_name, server = %server_label, "connecting to MCP server");
        let result = match entry {
            McpServerEntry::Stdio { command, args, env } => {
                McpClient::connect_stdio(command, args, env).await
            }
            _ => match entry.auth_header() {
                Some(auth) => McpClient::connect_with_auth(entry.url(), auth).await,
                None => McpClient::connect(entry.url()).await,
            },
        };
        match result {
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
                    server = %server_label,
                    error = %e,
                    "failed to connect to MCP server, skipping"
                );
            }
        }
    }
    tools
}

/// Discover A2A agents and collect tools. Failures are logged and skipped.
pub(crate) async fn load_a2a_tools(
    agent_name: &str,
    a2a_agents: &[McpServerEntry],
) -> Vec<Arc<dyn Tool>> {
    let mut tools = Vec::new();
    for entry in a2a_agents {
        let url = entry.url();
        tracing::info!(agent = %agent_name, url = %url, "discovering A2A agent");
        let result = match entry.auth_header() {
            Some(auth) => A2aClient::connect_with_auth(url, auth).await,
            None => A2aClient::connect(url).await,
        };
        match result {
            Ok(client) => {
                for tool in client.into_tools() {
                    let def = tool.definition();
                    tracing::info!(tool = %def.name, "registered A2A agent tool");
                    tools.push(tool);
                }
            }
            Err(e) => {
                tracing::warn!(
                    agent = %agent_name,
                    url = %url,
                    error = %e,
                    "failed to discover A2A agent, skipping"
                );
            }
        }
    }
    tools
}

/// Construct a type-erased LLM provider from environment variables.
///
/// Reads `HEARTBIT_PROVIDER` (default: auto-detect from available API keys),
/// `HEARTBIT_MODEL`, and `HEARTBIT_PROMPT_CACHING`. Always wraps with retry.
pub(crate) fn build_provider_from_env(
    on_retry: Option<Arc<OnRetry>>,
) -> Result<Arc<BoxedProvider>> {
    let provider_name = std::env::var("HEARTBIT_PROVIDER").unwrap_or_else(|_| {
        // Auto-detect from available API keys
        if let Some((name, _)) = heartbit::detect_available_provider() {
            name.to_string()
        } else {
            "anthropic".into() // default fallback
        }
    });

    let model = std::env::var("HEARTBIT_MODEL").unwrap_or_else(|_| {
        heartbit::get_provider(&provider_name)
            .map(|info| info.default_model.to_string())
            .unwrap_or_else(|| "claude-sonnet-4-20250514".into())
    });

    let prompt_caching = std::env::var("HEARTBIT_PROMPT_CACHING")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if prompt_caching && provider_name != "anthropic" {
        tracing::warn!(
            "HEARTBIT_PROMPT_CACHING is only effective with the 'anthropic' provider; \
             ignored for '{provider_name}'"
        );
    }

    let base_url = std::env::var("HEARTBIT_BASE_URL").ok();
    let api_key = std::env::var("HEARTBIT_API_KEY").ok();

    let base = build_base_provider(
        &provider_name,
        &model,
        prompt_caching,
        base_url.as_deref(),
        api_key.as_deref(),
    )?;

    let mut retrying = RetryingProvider::with_defaults(base);
    if let Some(cb) = on_retry {
        retrying = retrying.with_on_retry(cb);
    }
    Ok(Arc::new(BoxedProvider::new(retrying)))
}

/// Create a memory store from config.
pub(crate) async fn create_memory_store(config: &MemoryConfig) -> Result<Arc<dyn Memory>> {
    match config {
        MemoryConfig::InMemory => Ok(Arc::new(InMemoryStore::new())),
        MemoryConfig::Postgres {
            database_url,
            embedding,
        } => {
            let mut store = PostgresMemoryStore::connect(database_url)
                .await
                .context("failed to connect to PostgreSQL for memory store")?;
            store
                .run_migration()
                .await
                .context("failed to run memory store migration")?;
            let memory: Arc<dyn Memory> = Arc::new(store);

            // Wrap with EmbeddingMemory if embedding provider is configured
            if let Some(emb_config) = embedding
                && emb_config.provider != "none"
            {
                if emb_config.provider == "local" {
                    #[cfg(feature = "local-embedding")]
                    {
                        // Treat the OpenAI default model as "unset" for local provider
                        let model_name = if emb_config.model == "text-embedding-3-small" {
                            None
                        } else {
                            Some(emb_config.model.as_str())
                        };
                        let provider = heartbit::LocalEmbeddingProvider::new(
                            model_name,
                            emb_config.cache_dir.as_deref().map(std::path::Path::new),
                        )?;
                        return Ok(Arc::new(heartbit::EmbeddingMemory::new(
                            memory,
                            Arc::new(provider),
                        )));
                    }
                    #[cfg(not(feature = "local-embedding"))]
                    bail!("local embedding provider requires the 'local-embedding' feature");
                }

                let api_key = std::env::var(&emb_config.api_key_env).unwrap_or_default();
                if !api_key.is_empty() {
                    let mut provider = heartbit::OpenAiEmbedding::new(&api_key, &emb_config.model);
                    if let Some(ref base_url) = emb_config.base_url {
                        provider = provider.with_base_url(base_url);
                    }
                    if let Some(dim) = emb_config.dimension {
                        provider = provider.with_dimension(dim);
                    }
                    return Ok(Arc::new(heartbit::EmbeddingMemory::new(
                        memory,
                        Arc::new(provider),
                    )));
                }
            }
            Ok(memory)
        }
    }
}

async fn run_from_env(
    task: &str,
    approve: bool,
    verbose: bool,
    observability_flag: Option<&str>,
    trace_file: Option<PathBuf>,
) -> Result<()> {
    init_tracing();
    let mode = resolve_observability(observability_flag, None, verbose);
    let on_event = if verbose {
        Some(event_callback())
    } else {
        None
    };
    let on_retry = on_event.as_ref().map(build_on_retry);
    let provider = build_provider_from_env(on_retry)?;
    let model = resolve_model_from_env();

    let on_text = streaming_callback();
    let on_approval = if approve {
        Some(approval_callback())
    } else {
        None
    };

    let ws_root = default_workspace_root_path();
    let workspace_dir = provision_workspace(&ws_root);

    let mut output = run_default_agent(
        provider,
        task,
        on_text,
        on_approval,
        on_event,
        mode,
        workspace_dir,
    )
    .await?;
    output.estimated_cost_usd = heartbit::estimate_cost(&model, &output.tokens_used);
    // Headless-benchmark trace: serialize the final AgentOutput for the Harbor
    // adapter. The agent already did its work in the container, so a write
    // failure must NOT fail the run — log a warning and continue.
    if let Some(ref path) = trace_file
        && let Err(e) = write_trace(&output, path)
    {
        tracing::warn!(path = %path.display(), error = %e, "failed to write trace file");
    }
    print_streaming_stats(&output);
    Ok(())
}

/// Serialize an [`AgentOutput`] as pretty JSON to `path` (create/truncate).
///
/// Used by the headless-benchmark trace-file affordance. Pure helper so it can
/// be unit-tested without touching the env or the agent loop.
fn write_trace(output: &AgentOutput, path: &std::path::Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(output).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Whether non-interactive mode is enabled, given the raw `HEARTBIT_NONINTERACTIVE`
/// value. `"1"` or `"true"` (case-insensitive) => true; unset/anything else => false.
///
/// When enabled, the blocking `question` tool is dropped so a headless benchmark
/// never stalls on stdin. Pure helper for unit testing without env mutation.
fn noninteractive_enabled(val: Option<&str>) -> bool {
    matches!(
        val.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true")
    )
}

/// Resolve the model name from environment variables (mirrors build_provider_from_env logic).
fn resolve_model_from_env() -> String {
    let provider_name = std::env::var("HEARTBIT_PROVIDER").unwrap_or_else(|_| {
        if std::env::var("OPENROUTER_API_KEY").is_ok() {
            "openrouter".into()
        } else {
            "anthropic".into()
        }
    });
    std::env::var("HEARTBIT_MODEL").unwrap_or_else(|_| {
        if provider_name == "openrouter" {
            "anthropic/claude-sonnet-4".into()
        } else {
            "claude-sonnet-4-20250514".into()
        }
    })
}

/// Run a single agent with built-in tools (default no-config mode).
///
/// More efficient than the multi-agent orchestrator for most tasks:
/// single LLM context, no delegation overhead, direct tool access.
async fn run_default_agent(
    provider: Arc<BoxedProvider>,
    task: &str,
    on_text: Arc<OnText>,
    on_approval: Option<Arc<OnApproval>>,
    on_event: Option<Arc<OnEvent>>,
    observability_mode: ObservabilityMode,
    workspace_dir: Option<PathBuf>,
) -> Result<AgentOutput> {
    // Opt-in Agent Skills discovery: only when the operator explicitly sets
    // HEARTBIT_SKILL_DIRS (comma-separated). This avoids auto-injecting skill
    // descriptions from ancestor/home `.claude/skills` dirs into the system
    // prompt (a trust-boundary / prompt-injection risk). Empty => no skills.
    let skill_dirs: Vec<PathBuf> = std::env::var("HEARTBIT_SKILL_DIRS")
        .ok()
        .map(|v| parse_csv_env(&v).into_iter().map(PathBuf::from).collect())
        .unwrap_or_default();

    let mut tools = {
        let mut btc = BuiltinToolsConfig::default();
        // Drop the blocking `question` tool in non-interactive mode so a headless
        // benchmark never stalls on stdin (builtins only add it when on_question
        // is Some). Default (unset) is unchanged.
        if !noninteractive_enabled(std::env::var("HEARTBIT_NONINTERACTIVE").ok().as_deref()) {
            btc.on_question = Some(question_callback());
        }
        btc.workspace = workspace_dir.clone();
        btc.dangerous_tools = true; // CLI mode: enable bash for backward compat
        btc.skill_dirs = skill_dirs.clone();
        builtin_tools(btc)
    };

    // Load MCP tools from env: HEARTBIT_MCP_SERVERS=url1,url2
    if let Ok(servers) = std::env::var("HEARTBIT_MCP_SERVERS") {
        let entries: Vec<McpServerEntry> = parse_csv_env(&servers)
            .into_iter()
            .map(McpServerEntry::Simple)
            .collect();
        let mcp_tools = load_mcp_tools("heartbit", &entries).await;
        tools.extend(mcp_tools);
    }

    // Load A2A agent tools from env: HEARTBIT_A2A_AGENTS=url1,url2
    if let Ok(agents) = std::env::var("HEARTBIT_A2A_AGENTS") {
        let entries: Vec<McpServerEntry> = parse_csv_env(&agents)
            .into_iter()
            .map(McpServerEntry::Simple)
            .collect();
        let a2a_tools = load_a2a_tools("heartbit", &entries).await;
        tools.extend(a2a_tools);
    }

    let max_turns: usize = parse_env("HEARTBIT_MAX_TURNS").unwrap_or(50);
    let summarize_threshold: u32 = parse_env("HEARTBIT_SUMMARIZE_THRESHOLD").unwrap_or(80_000);
    let max_tool_output_bytes: usize =
        parse_env("HEARTBIT_MAX_TOOL_OUTPUT_BYTES").unwrap_or(32_768);
    let tool_timeout_secs: u64 = parse_env("HEARTBIT_TOOL_TIMEOUT").unwrap_or(120);

    // Base instructions + Level-1 Agent Skills catalog (progressive disclosure):
    // advertise the OPT-IN, explicitly-configured skills (name: description) so
    // the model knows which to load on demand via the `skill` tool. Uses the same
    // explicit dirs the skill tool searches, so the catalog never lists a skill
    // that cannot be loaded. No ancestor/$HOME auto-discovery (security).
    let mut env_system_prompt = String::from(
        "You are a skilled software engineer. Use the available tools to accomplish \
         the task. Work methodically: read relevant files before making changes, \
         verify your work by running tests, and explain your reasoning. \
         When modifying code, always read the file first.",
    );
    if let Some(catalog) = heartbit::skill::catalog_from_dirs(&skill_dirs) {
        env_system_prompt.push_str("\n\n");
        env_system_prompt.push_str(&catalog);
    }

    let mut builder = AgentRunner::builder(provider)
        .name("heartbit")
        .system_prompt(&env_system_prompt)
        .tools(tools)
        .max_turns(max_turns)
        .summarize_threshold(summarize_threshold)
        .max_tool_output_bytes(max_tool_output_bytes)
        .tool_timeout(std::time::Duration::from_secs(tool_timeout_secs))
        .on_text(on_text)
        .observability_mode(observability_mode);

    if let Some(ref ws) = workspace_dir {
        builder = builder.workspace(ws.clone());
    }

    if let Some(cb) = on_approval {
        builder = builder.on_approval(cb);
    }
    if let Some(cb) = on_event {
        builder = builder.on_event(cb);
    }
    // Wire learned permissions from disk
    if let Some(learned) = load_learned_permissions() {
        let guard = learned.lock().expect("learned permissions lock");
        if !guard.rules().is_empty() {
            let mut ruleset = heartbit::PermissionRuleset::default();
            ruleset.append_rules(guard.rules());
            builder = builder.permission_rules(ruleset);
        }
        drop(guard);
        builder = builder.learned_permissions(learned);
    }
    if let Ok(effort_str) = std::env::var("HEARTBIT_REASONING_EFFORT") {
        builder = builder.reasoning_effort(heartbit::config::parse_reasoning_effort(&effort_str)?);
    }
    if std::env::var("HEARTBIT_ENABLE_REFLECTION")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        builder = builder.enable_reflection(true);
    }
    if let Ok(threshold) = std::env::var("HEARTBIT_COMPRESSION_THRESHOLD")
        && let Ok(n) = threshold.parse::<usize>()
    {
        builder = builder.tool_output_compression_threshold(n);
    }
    if let Ok(max) = std::env::var("HEARTBIT_MAX_TOOLS_PER_TURN")
        && let Ok(n) = max.parse::<usize>()
    {
        builder = builder.max_tools_per_turn(n);
    }
    if let Ok(profile_str) = std::env::var("HEARTBIT_TOOL_PROFILE") {
        builder = builder.tool_profile(heartbit::parse_tool_profile(&profile_str)?);
    }
    if let Ok(max) = std::env::var("HEARTBIT_MAX_IDENTICAL_TOOL_CALLS")
        && let Ok(n) = max.parse::<u32>()
    {
        builder = builder.max_identical_tool_calls(n);
    }
    if let Ok(max) = std::env::var("HEARTBIT_MAX_TOOL_CALLS_PER_TURN")
        && let Ok(n) = max.parse::<u32>()
    {
        builder = builder.max_tool_calls_per_turn(n);
    }
    if let Ok(size) = std::env::var("HEARTBIT_RESPONSE_CACHE_SIZE")
        && let Ok(n) = size.parse::<usize>()
    {
        builder = builder.response_cache_size(n);
    }
    if std::env::var("HEARTBIT_SESSION_PRUNE")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        builder = builder.session_prune_config(heartbit::SessionPruneConfig::default());
    }
    if std::env::var("HEARTBIT_RECURSIVE_SUMMARIZATION")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        builder = builder.enable_recursive_summarization(true);
    }
    if let Ok(threshold) = std::env::var("HEARTBIT_REFLECTION_THRESHOLD")
        && let Ok(n) = threshold.parse::<u32>()
    {
        builder = builder.reflection_threshold(n);
    }
    if std::env::var("HEARTBIT_CONSOLIDATE_ON_EXIT")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        builder = builder.consolidate_on_exit(true);
    }
    // Wire hierarchical instruction files (HEARTBIT.md)
    if let Some(text) = load_instruction_text() {
        builder = builder.instruction_text(text);
    }
    // Wire memory from env: HEARTBIT_MEMORY=in_memory or postgres://...
    if let Ok(value) = std::env::var("HEARTBIT_MEMORY") {
        let config = if value == "in_memory" {
            MemoryConfig::InMemory
        } else if value.starts_with("postgres://") || value.starts_with("postgresql://") {
            MemoryConfig::Postgres {
                database_url: value,
                embedding: None,
            }
        } else {
            bail!("HEARTBIT_MEMORY must be 'in_memory' or a PostgreSQL URL, got: {value}");
        };
        let memory = create_memory_store(&config).await?;
        builder = builder.memory(memory);
    }
    // Wire LSP integration if enabled via env
    if std::env::var("HEARTBIT_LSP_ENABLED")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        let workspace_root = std::env::current_dir().unwrap_or_default();
        builder = builder.lsp_manager(Arc::new(heartbit::LspManager::new(workspace_root)));
    }

    let runner = builder.build()?;
    let output = runner.execute(task).await?;
    Ok(output)
}

/// Run an interactive chat session (REPL mode).
///
/// Prompts the user for an initial message, then loops: whenever the agent
/// produces a text response (no tool calls), the user is prompted again.
/// The session ends when the user sends an empty line or EOF.
///
/// When `config_path` is provided, the provider and settings are loaded from
/// the config file. MCP servers from all configured agents are loaded.
/// Otherwise falls back to environment variables.
async fn run_chat(
    config_path: Option<&std::path::Path>,
    approve: bool,
    verbose: bool,
    observability_flag: Option<&str>,
) -> Result<()> {
    match config_path {
        Some(path) => run_chat_from_config(path, approve, verbose, observability_flag).await,
        None => run_chat_from_env(approve, verbose, observability_flag).await,
    }
}

/// Chat session backed by a config file.
async fn run_chat_from_config(
    path: &std::path::Path,
    approve: bool,
    verbose: bool,
    observability_flag: Option<&str>,
) -> Result<()> {
    let mut config = HeartbitConfig::from_file(path)
        .with_context(|| format!("failed to load config from {}", path.display()))?;

    // Resolve agent templates, skills, and variables.
    let variables = config.variables.clone();
    for i in 0..config.agents.len() {
        if config.agents[i].template.is_some()
            || !config.agents[i].skills.is_empty()
            || !config.agents[i].skill_dirs.is_empty()
        {
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

    let config_obs = config
        .telemetry
        .as_ref()
        .and_then(|t| t.observability_mode.as_deref());
    let mode = resolve_observability(observability_flag, config_obs, verbose);

    let on_event = if verbose {
        Some(event_callback())
    } else {
        None
    };
    let on_retry = on_event.as_ref().map(build_on_retry);
    let base_provider = build_provider_from_config(&config, on_retry)?;

    // B5b: circuit breaker wrapper for chat path
    let circuit_tracker = Arc::new(heartbit::CircuitTracker::new(
        heartbit::CircuitConfig::from(&config.provider.circuit),
    ));
    let provider = wrap_with_circuit(
        base_provider,
        circuit_tracker,
        &config.provider.name,
        heartbit::TenantScope::default(),
    );

    // B5b: tenant token tracker for chat path
    let tenant_tracker = config
        .orchestrator
        .max_tokens_in_flight_per_tenant
        .map(|cap| Arc::new(heartbit::TenantTokenTracker::new(cap)));

    let initial = read_initial_chat_message().await?;
    let Some(initial) = initial else {
        return Ok(());
    };

    let ws_root = workspace_root_from_config(&config);
    let workspace_dir = provision_workspace(&ws_root);

    // Build path policy from [sandbox] config if present.
    let path_policy = build_path_policy(config.sandbox.as_ref())?;

    // Load MCP + A2A tools from all configured agents
    let mut tools = {
        let mut btc = BuiltinToolsConfig::default();
        btc.on_question = Some(question_callback());
        btc.workspace = workspace_dir.clone();
        btc.dangerous_tools = true; // CLI mode: enable bash for backward compat
        btc.path_policy = path_policy.clone();
        #[cfg(all(target_os = "linux", feature = "sandbox"))]
        if let Some(ref pp) = path_policy {
            btc.sandbox_policy = Some(heartbit::SandboxPolicy::from_path_policy(Arc::clone(pp)));
        }
        builtin_tools(btc)
    };
    for agent in &config.agents {
        let mcp_tools = load_mcp_tools(&agent.name, &agent.mcp_servers).await;
        let a2a_tools = load_a2a_tools(&agent.name, &agent.a2a_agents).await;
        tools.extend(mcp_tools);
        tools.extend(a2a_tools);
    }

    // Use orchestrator settings with chat-friendly max_turns default (200)
    let max_turns = parse_env("HEARTBIT_MAX_TURNS")
        .unwrap_or_else(|| std::cmp::max(config.orchestrator.max_turns, 200));

    let on_text = streaming_callback();
    let on_approval = if approve {
        Some(approval_callback())
    } else {
        None
    };

    // Request-intent router (mode contracts): chat inherits the core-side
    // request fidelity (answer/execute/study/clarify). Classifier reuses the
    // session provider. Opt out with HEARTBIT_REQUEST_ROUTER=0.
    let request_router = if std::env::var("HEARTBIT_REQUEST_ROUTER").as_deref() != Ok("0") {
        Some(std::sync::Arc::new(
            heartbit_core::agent::router::RequestRouter::new(Some(std::sync::Arc::new(
                heartbit_core::BoxedProvider::from_arc(provider.clone()),
            ))),
        ))
    } else {
        None
    };

    let mut builder = AgentRunner::builder(provider)
        .name("heartbit")
        .system_prompt(
            "You are a skilled software engineer in an interactive chat session. \
             Use the available tools to accomplish tasks. Work methodically: read \
             relevant files before making changes, verify your work by running tests, \
             and explain your reasoning. When modifying code, always read the file first.",
        )
        .tools(tools)
        .max_turns(max_turns)
        .max_tokens(config.orchestrator.max_tokens)
        .on_text(on_text)
        .on_input(input_callback())
        .observability_mode(mode);
    if let Some(router) = request_router {
        builder = builder.request_router(router);
    }

    if let Some(ref ws) = workspace_dir {
        builder = builder.workspace(ws.clone());
    }

    // Wire orchestrator-level context management
    match &config.orchestrator.context_strategy {
        Some(ContextStrategyConfig::SlidingWindow { max_tokens }) => {
            builder = builder.context_strategy(ContextStrategy::SlidingWindow {
                max_tokens: *max_tokens,
            });
        }
        Some(ContextStrategyConfig::Summarize { threshold }) => {
            builder = builder.summarize_threshold(*threshold);
        }
        Some(ContextStrategyConfig::Unlimited) | None => {
            // Default summarize threshold for interactive chat (80k tokens)
            let threshold = config.orchestrator.summarize_threshold.unwrap_or(80_000);
            builder = builder.summarize_threshold(threshold);
        }
    }
    // Defaults matching run_chat_from_env: 120s tool timeout, 32KB output limit
    let tool_timeout_secs = config.orchestrator.tool_timeout_seconds.unwrap_or(120);
    builder = builder.tool_timeout(std::time::Duration::from_secs(tool_timeout_secs));
    let max_output = config.orchestrator.max_tool_output_bytes.unwrap_or(32_768);
    builder = builder.max_tool_output_bytes(max_output);
    // Note: run_timeout_seconds is intentionally NOT wired for interactive chat
    // because it spans user think-time (time the user spends typing).

    if let Some(cb) = on_approval {
        builder = builder.on_approval(cb);
    }
    if let Some(cb) = on_event {
        builder = builder.on_event(cb);
    }
    if let Some(ref effort) = config.orchestrator.reasoning_effort {
        builder = builder.reasoning_effort(heartbit::config::parse_reasoning_effort(effort)?);
    }
    if let Some(true) = config.orchestrator.enable_reflection {
        builder = builder.enable_reflection(true);
    }
    if let Some(threshold) = config.orchestrator.tool_output_compression_threshold {
        builder = builder.tool_output_compression_threshold(threshold);
    }
    if let Some(max) = config.orchestrator.max_tools_per_turn {
        builder = builder.max_tools_per_turn(max);
    }
    if let Some(max) = config.orchestrator.max_identical_tool_calls {
        builder = builder.max_identical_tool_calls(max);
    }
    if let Some(max) = config.orchestrator.max_fuzzy_identical_tool_calls {
        builder = builder.max_fuzzy_identical_tool_calls(max);
    }
    if let Some(cap) = config.orchestrator.max_tool_calls_per_turn {
        builder = builder.max_tool_calls_per_turn(cap);
    }
    {
        let mut ruleset = heartbit::PermissionRuleset::new(config.permissions.clone());
        if let Some(ref learned) = load_learned_permissions() {
            let guard = learned.lock().expect("learned permissions lock");
            ruleset.append_rules(guard.rules());
            builder = builder.learned_permissions(learned.clone());
        }
        if !ruleset.is_empty() {
            builder = builder.permission_rules(ruleset);
        }
    }
    // Wire hierarchical instruction files (HEARTBIT.md)
    if let Some(text) = load_instruction_text() {
        builder = builder.instruction_text(text);
    }
    // Wire memory SOTA features from first agent config (chat uses a single agent)
    if let Some(first_agent) = config.agents.first() {
        if let Some(ref sp) = first_agent.session_prune {
            builder = builder.session_prune_config(heartbit::SessionPruneConfig {
                keep_recent_n: sp.keep_recent_n,
                pruned_tool_result_max_bytes: sp.pruned_tool_result_max_bytes,
                preserve_task: sp.preserve_task,
            });
        }
        if let Some(true) = first_agent.recursive_summarization {
            builder = builder.enable_recursive_summarization(true);
        }
        if let Some(threshold) = first_agent.reflection_threshold {
            builder = builder.reflection_threshold(threshold);
        }
        if let Some(true) = first_agent.consolidate_on_exit {
            builder = builder.consolidate_on_exit(true);
        }
        if let Some(budget) = first_agent.max_total_tokens {
            builder = builder.max_total_tokens(budget);
        }
    }
    // Wire memory if configured
    if let Some(ref memory_config) = config.memory {
        let memory = create_memory_store(memory_config).await?;
        builder = builder.memory(memory);
    }
    // Wire LSP integration if configured
    if let Some(ref lsp_config) = config.lsp
        && lsp_config.enabled
    {
        let workspace_root = std::env::current_dir().unwrap_or_default();
        builder = builder.lsp_manager(Arc::new(heartbit::LspManager::new(workspace_root)));
    }
    // Wire guardrails from config (top-level, since chat uses a single agent)
    if let Some(ref gc) = config.guardrails {
        let judge_provider: Option<Arc<heartbit::BoxedProvider>> = if gc.llm_judge.is_some() {
            // Use main provider as judge for chat (no separate judge configured)
            match build_provider_from_config(&config, None) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "failed to build judge provider for chat guardrails, skipping LLM judge");
                    None
                }
            }
        } else {
            None
        };
        let guardrails = gc.build_with_judge(judge_provider)?;
        if !guardrails.is_empty() {
            builder = builder.guardrails(guardrails);
        }
    }
    // B5b: wire tenant token tracker into chat agent runner
    if let Some(tracker) = tenant_tracker {
        builder = builder.tenant_tracker(tracker);
    }

    let runner = builder.build()?;
    let mut output = runner.execute(&initial).await?;
    // Cost estimate is only accurate when all agents use the same model.
    let has_overrides = config.agents.iter().any(|a| a.provider.is_some());
    if !has_overrides {
        output.estimated_cost_usd =
            heartbit::estimate_cost(&config.provider.model, &output.tokens_used);
    }
    print_streaming_stats(&output);
    Ok(())
}

/// Chat session backed by environment variables.
async fn run_chat_from_env(
    approve: bool,
    verbose: bool,
    observability_flag: Option<&str>,
) -> Result<()> {
    init_tracing();
    let mode = resolve_observability(observability_flag, None, verbose);
    let on_event = if verbose {
        Some(event_callback())
    } else {
        None
    };
    let on_retry = on_event.as_ref().map(build_on_retry);
    let provider = build_provider_from_env(on_retry)?;

    let initial = read_initial_chat_message().await?;
    let Some(initial) = initial else {
        return Ok(());
    };

    let ws_root = default_workspace_root_path();
    let workspace_dir = provision_workspace(&ws_root);

    let mut tools = {
        let mut btc = BuiltinToolsConfig::default();
        // Same non-interactive guard as the run env path: skip the blocking
        // `question` tool when HEARTBIT_NONINTERACTIVE is set. Default unchanged.
        if !noninteractive_enabled(std::env::var("HEARTBIT_NONINTERACTIVE").ok().as_deref()) {
            btc.on_question = Some(question_callback());
        }
        btc.workspace = workspace_dir.clone();
        btc.dangerous_tools = true; // CLI mode: enable bash for backward compat
        builtin_tools(btc)
    };
    if let Ok(servers) = std::env::var("HEARTBIT_MCP_SERVERS") {
        let entries: Vec<McpServerEntry> = parse_csv_env(&servers)
            .into_iter()
            .map(McpServerEntry::Simple)
            .collect();
        let mcp_tools = load_mcp_tools("heartbit", &entries).await;
        tools.extend(mcp_tools);
    }
    if let Ok(agents) = std::env::var("HEARTBIT_A2A_AGENTS") {
        let entries: Vec<McpServerEntry> = parse_csv_env(&agents)
            .into_iter()
            .map(McpServerEntry::Simple)
            .collect();
        let a2a_tools = load_a2a_tools("heartbit", &entries).await;
        tools.extend(a2a_tools);
    }

    let max_turns: usize = parse_env("HEARTBIT_MAX_TURNS").unwrap_or(200);
    let summarize_threshold: u32 = parse_env("HEARTBIT_SUMMARIZE_THRESHOLD").unwrap_or(80_000);
    let max_tool_output_bytes: usize =
        parse_env("HEARTBIT_MAX_TOOL_OUTPUT_BYTES").unwrap_or(32_768);
    let tool_timeout_secs: u64 = parse_env("HEARTBIT_TOOL_TIMEOUT").unwrap_or(120);

    let on_text = streaming_callback();
    let on_approval = if approve {
        Some(approval_callback())
    } else {
        None
    };

    let mut builder = AgentRunner::builder(provider)
        .name("heartbit")
        .system_prompt(
            "You are a skilled software engineer in an interactive chat session. \
             Use the available tools to accomplish tasks. Work methodically: read \
             relevant files before making changes, verify your work by running tests, \
             and explain your reasoning. When modifying code, always read the file first.",
        )
        .tools(tools)
        .max_turns(max_turns)
        .summarize_threshold(summarize_threshold)
        .max_tool_output_bytes(max_tool_output_bytes)
        .tool_timeout(std::time::Duration::from_secs(tool_timeout_secs))
        .on_text(on_text)
        .on_input(input_callback())
        .observability_mode(mode);

    if let Some(ref ws) = workspace_dir {
        builder = builder.workspace(ws.clone());
    }

    if let Some(cb) = on_approval {
        builder = builder.on_approval(cb);
    }
    if let Some(cb) = on_event {
        builder = builder.on_event(cb);
    }
    // Wire learned permissions from disk
    if let Some(learned) = load_learned_permissions() {
        let guard = learned.lock().expect("learned permissions lock");
        if !guard.rules().is_empty() {
            let mut ruleset = heartbit::PermissionRuleset::default();
            ruleset.append_rules(guard.rules());
            builder = builder.permission_rules(ruleset);
        }
        drop(guard);
        builder = builder.learned_permissions(learned);
    }
    if let Ok(effort_str) = std::env::var("HEARTBIT_REASONING_EFFORT") {
        builder = builder.reasoning_effort(heartbit::config::parse_reasoning_effort(&effort_str)?);
    }
    if std::env::var("HEARTBIT_ENABLE_REFLECTION")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        builder = builder.enable_reflection(true);
    }
    if let Ok(threshold) = std::env::var("HEARTBIT_COMPRESSION_THRESHOLD")
        && let Ok(n) = threshold.parse::<usize>()
    {
        builder = builder.tool_output_compression_threshold(n);
    }
    if let Ok(max) = std::env::var("HEARTBIT_MAX_TOOLS_PER_TURN")
        && let Ok(n) = max.parse::<usize>()
    {
        builder = builder.max_tools_per_turn(n);
    }
    if let Ok(profile_str) = std::env::var("HEARTBIT_TOOL_PROFILE") {
        builder = builder.tool_profile(heartbit::parse_tool_profile(&profile_str)?);
    }
    if let Ok(max) = std::env::var("HEARTBIT_MAX_IDENTICAL_TOOL_CALLS")
        && let Ok(n) = max.parse::<u32>()
    {
        builder = builder.max_identical_tool_calls(n);
    }
    if let Ok(max) = std::env::var("HEARTBIT_MAX_TOOL_CALLS_PER_TURN")
        && let Ok(n) = max.parse::<u32>()
    {
        builder = builder.max_tool_calls_per_turn(n);
    }
    if let Ok(size) = std::env::var("HEARTBIT_RESPONSE_CACHE_SIZE")
        && let Ok(n) = size.parse::<usize>()
    {
        builder = builder.response_cache_size(n);
    }
    if std::env::var("HEARTBIT_SESSION_PRUNE")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        builder = builder.session_prune_config(heartbit::SessionPruneConfig::default());
    }
    if std::env::var("HEARTBIT_RECURSIVE_SUMMARIZATION")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        builder = builder.enable_recursive_summarization(true);
    }
    if let Ok(threshold) = std::env::var("HEARTBIT_REFLECTION_THRESHOLD")
        && let Ok(n) = threshold.parse::<u32>()
    {
        builder = builder.reflection_threshold(n);
    }
    if std::env::var("HEARTBIT_CONSOLIDATE_ON_EXIT")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        builder = builder.consolidate_on_exit(true);
    }
    // Wire hierarchical instruction files (HEARTBIT.md)
    if let Some(text) = load_instruction_text() {
        builder = builder.instruction_text(text);
    }
    // Wire memory from env: HEARTBIT_MEMORY=in_memory or postgres://...
    if let Ok(value) = std::env::var("HEARTBIT_MEMORY") {
        let config = if value == "in_memory" {
            MemoryConfig::InMemory
        } else if value.starts_with("postgres://") || value.starts_with("postgresql://") {
            MemoryConfig::Postgres {
                database_url: value,
                embedding: None,
            }
        } else {
            bail!("HEARTBIT_MEMORY must be 'in_memory' or a PostgreSQL URL, got: {value}");
        };
        let memory = create_memory_store(&config).await?;
        builder = builder.memory(memory);
    }
    // Wire LSP integration if enabled via env
    if std::env::var("HEARTBIT_LSP_ENABLED")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        let workspace_root = std::env::current_dir().unwrap_or_default();
        builder = builder.lsp_manager(Arc::new(heartbit::LspManager::new(workspace_root)));
    }
    // Wire guardrails from env: HEARTBIT_GUARDRAILS_INJECTION=deny (or warn)
    {
        let mut env_guardrails: Vec<Arc<dyn heartbit::Guardrail>> = Vec::new();
        if let Ok(mode_str) = std::env::var("HEARTBIT_GUARDRAILS_INJECTION") {
            let mode = match mode_str.as_str() {
                "warn" => heartbit::GuardrailMode::Warn,
                "deny" => heartbit::GuardrailMode::Deny,
                other => {
                    bail!("HEARTBIT_GUARDRAILS_INJECTION must be 'warn' or 'deny', got: {other}")
                }
            };
            let threshold: f32 =
                parse_env("HEARTBIT_GUARDRAILS_INJECTION_THRESHOLD").unwrap_or(0.5);
            env_guardrails.push(Arc::new(heartbit::InjectionClassifierGuardrail::new(
                threshold, mode,
            )));
        }
        if let Ok(action_str) = std::env::var("HEARTBIT_GUARDRAILS_PII") {
            let action = match action_str.as_str() {
                "redact" => heartbit::PiiAction::Redact,
                "warn" => heartbit::PiiAction::Warn,
                "deny" => heartbit::PiiAction::Deny,
                other => {
                    bail!(
                        "HEARTBIT_GUARDRAILS_PII must be 'redact', 'warn', or 'deny', got: {other}"
                    )
                }
            };
            env_guardrails.push(Arc::new(heartbit::PiiGuardrail::all_builtin(action)));
        }
        if !env_guardrails.is_empty() {
            builder = builder.guardrails(env_guardrails);
        }
    }

    let runner = builder.build()?;
    let mut output = runner.execute(&initial).await?;
    let model = resolve_model_from_env();
    output.estimated_cost_usd = heartbit::estimate_cost(&model, &output.tokens_used);
    print_streaming_stats(&output);
    Ok(())
}

/// Print the chat banner and read the initial message from stdin.
///
/// Returns `None` if the user sends an empty line or EOF.
async fn read_initial_chat_message() -> Result<Option<String>> {
    eprintln!("Heartbit interactive chat. Type your message, then press Enter.");
    eprintln!("Send an empty line or Ctrl-D to exit.\n");

    eprint!("> ");
    std::io::Write::flush(&mut std::io::stderr()).ok();
    let initial = tokio::task::spawn_blocking(|| {
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).ok()?;
        let trimmed = buf.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
    .await
    .ok()
    .flatten();
    Ok(initial)
}

/// Create an input callback for interactive chat mode.
///
/// Prompts the user on stderr, reads a line from stdin. Returns `None`
/// on empty input or EOF to end the session.
fn input_callback() -> Arc<heartbit::OnInput> {
    Arc::new(|| {
        Box::pin(async {
            // Use spawn_blocking to avoid blocking a tokio worker thread.
            // Without this, a current_thread runtime would deadlock.
            tokio::task::spawn_blocking(|| {
                use std::io::Write;
                eprintln!(); // newline after streamed text
                eprint!("> ");
                std::io::stderr().flush().ok();

                let mut input = String::new();
                if std::io::stdin().read_line(&mut input).is_err() {
                    return None;
                }
                let trimmed = input.trim();
                if trimmed.is_empty() {
                    return None;
                }
                Some(trimmed.to_string())
            })
            .await
            .ok()
            .flatten()
        })
    })
}

/// Parse an env var into a type. Returns `None` if unset or unparseable.
fn parse_env<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| match v.parse() {
        Ok(parsed) => Some(parsed),
        Err(_) => {
            tracing::warn!("invalid value for {key}={v:?}, ignoring");
            None
        }
    })
}

/// Parse a comma-separated env var into a Vec of trimmed, non-empty strings.
fn parse_csv_env(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
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
/// Supported responses:
/// - `Y` / `y` / empty → Allow (this time)
/// - `n` / `N` → Deny (this time)
/// - `Y!` → AlwaysAllow (persist as learned rule)
/// - `N!` → AlwaysDeny (persist as learned rule)
///
/// The `OnApproval` callback is sync by design. The blocking stdin read runs
/// on one tokio worker thread, but this is intentional: the orchestrator is
/// waiting for the human decision anyway, and the multi-threaded runtime
/// allows other tasks to progress on remaining worker threads.
fn approval_callback() -> Arc<OnApproval> {
    Arc::new(|tool_calls: &[ToolCall]| {
        use heartbit::ApprovalDecision;
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
        eprint!("Approve? [Y/n/Y!/N!] ");
        std::io::stderr().flush().ok();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            return ApprovalDecision::Deny;
        }
        let trimmed = input.trim();
        match trimmed {
            "Y!" => ApprovalDecision::AlwaysAllow,
            "N!" => ApprovalDecision::AlwaysDeny,
            "" | "y" | "Y" | "yes" | "Yes" => ApprovalDecision::Allow,
            _ => ApprovalDecision::Deny,
        }
    })
}

/// Create a structured question callback that prompts on stderr.
///
/// Presents numbered options and reads the user's selection from stdin.
/// For single-select questions, the user enters one number.
/// For multi-select questions, the user enters comma-separated numbers.
/// Empty input defaults to the first option.
///
/// The blocking `stdin.read_line()` inside the async block runs on a tokio
/// worker thread. This is intentional: the agent loop is waiting for the
/// answer anyway, and the multi-threaded runtime allows other tasks to
/// progress on remaining worker threads.
fn question_callback() -> Arc<OnQuestion> {
    Arc::new(|request: QuestionRequest| {
        Box::pin(async move {
            // Use spawn_blocking to avoid blocking a tokio worker thread.
            // Without this, a current_thread runtime would deadlock.
            let result = tokio::task::spawn_blocking(move || {
                use std::io::Write;
                let mut answers = Vec::new();

                for q in &request.questions {
                    eprintln!("\n--- {} ---", q.header);
                    eprintln!("{}", q.question);
                    for (i, opt) in q.options.iter().enumerate() {
                        eprintln!("  {}. {} — {}", i + 1, opt.label, opt.description);
                    }
                    if q.multiple {
                        eprint!("Select (comma-separated numbers, default 1): ");
                    } else {
                        eprint!("Select (number, default 1): ");
                    }
                    std::io::stderr().flush().ok();

                    let mut input = String::new();
                    if std::io::stdin().read_line(&mut input).is_err() {
                        // Default to first option on read error
                        answers.push(vec![
                            q.options
                                .first()
                                .map(|o| o.label.clone())
                                .unwrap_or_default(),
                        ]);
                        continue;
                    }

                    let selected: Vec<String> = input
                        .trim()
                        .split(',')
                        .filter_map(|s| {
                            let n: usize = s.trim().parse().ok()?;
                            q.options.get(n.checked_sub(1)?).map(|o| o.label.clone())
                        })
                        .collect();

                    if selected.is_empty() {
                        // Default to first option
                        answers.push(vec![
                            q.options
                                .first()
                                .map(|o| o.label.clone())
                                .unwrap_or_default(),
                        ]);
                    } else {
                        answers.push(selected);
                    }
                }

                QuestionResponse { answers }
            })
            .await
            .map_err(|e| heartbit::Error::Agent(format!("Question callback panicked: {e}")))?;

            Ok(result)
        })
    })
}

/// Create an event callback that writes each event as one-line JSON to stderr.
pub(crate) fn event_callback() -> Arc<OnEvent> {
    Arc::new(|event: AgentEvent| {
        let json = serde_json::to_string(&event).expect("AgentEvent serialization is infallible");
        eprintln!("[event] {json}");
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
    if let Some(cost) = output.estimated_cost_usd {
        eprintln!("Estimated cost: ${cost:.4}");
    }
}

// ── Template / Skill / Init CLI commands ──

fn run_template_command(action: TemplateAction) -> Result<()> {
    match action {
        TemplateAction::List => {
            let templates = heartbit::known_templates();
            println!("Available templates ({}):\n", templates.len());
            for name in templates {
                match heartbit::resolve_template(name) {
                    Ok(t) => println!("  {:<20} {}", name, t.meta.description),
                    Err(_) => println!("  {:<20} (failed to load)", name),
                }
            }
            Ok(())
        }
        TemplateAction::Show { name } => {
            let template = heartbit::resolve_template(&name)
                .with_context(|| format!("failed to resolve template '{name}'"))?;
            println!("Template: {name}");
            println!("Description: {}", template.meta.description);
            println!("Version: {}", template.meta.version);
            if !template.meta.tags.is_empty() {
                println!("Tags: {}", template.meta.tags.join(", "));
            }
            if let Some(ref parent) = template.meta.extends {
                println!("Extends: {parent}");
            }
            println!("\n--- Agent Defaults ---");
            if let Some(mt) = template.agent.max_tokens {
                println!("max_tokens: {mt}");
            }
            if let Some(mt) = template.agent.max_turns {
                println!("max_turns: {mt}");
            }
            if let Some(ref tp) = template.agent.tool_profile {
                println!("tool_profile: {tp}");
            }
            if let Some(dt) = template.agent.dangerous_tools {
                println!("dangerous_tools: {dt}");
            }
            if let Some(ref prompt) = template.agent.system_prompt {
                println!("\n--- System Prompt ---\n{prompt}");
            }
            Ok(())
        }
    }
}

fn run_skill_command(action: SkillAction) -> Result<()> {
    match action {
        SkillAction::List => {
            let skills = heartbit::known_skills();
            println!("Available skills ({}):\n", skills.len());
            for name in skills {
                match heartbit::load_skill(name) {
                    Ok(s) => {
                        let desc = s.description.as_deref().unwrap_or("(no description)");
                        println!("  {:<20} {}", name, desc);
                    }
                    Err(_) => println!("  {:<20} (failed to load)", name),
                }
            }
            Ok(())
        }
        SkillAction::Show { name } => {
            let skill = heartbit::load_skill(&name)
                .with_context(|| format!("failed to load skill '{name}'"))?;
            println!("Skill: {}", skill.name);
            if let Some(ref desc) = skill.description {
                println!("Description: {desc}");
            }
            if let Some(max) = skill.max_inject_tokens {
                println!("Max inject tokens: {max}");
            }
            println!("\n--- Content ---\n{}", skill.content);
            Ok(())
        }
    }
}

fn run_init_command(template_name: &str, output: Option<&std::path::Path>) -> Result<()> {
    let template = heartbit::resolve_template(template_name)
        .with_context(|| format!("failed to resolve template '{template_name}'"))?;

    let config_toml = format!(
        r#"[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
template = "{template_name}"
name = "{template_name}"
description = "{description}"
skills = []
# system_prompt appended to template prompt:
# system_prompt = "Additional context here"
"#,
        description = template.meta.description
    );

    match output {
        Some(path) => {
            std::fs::write(path, &config_toml)
                .with_context(|| format!("failed to write {}", path.display()))?;
            println!("Config written to {}", path.display());
        }
        None => {
            print!("{config_toml}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Change 1: --trace-file write helper ---

    #[test]
    fn write_trace_emits_result_and_token_fields() {
        // Build an AgentOutput with the fields the Python adapter parses.
        // AgentOutput is #[non_exhaustive], so construct via Default + assign.
        let mut output = AgentOutput::default();
        output.result = "all done".to_string();
        output.tokens_used.input_tokens = 1234;
        output.tokens_used.output_tokens = 56;
        output.estimated_cost_usd = Some(0.0042);

        // Unique path in the temp dir (reading temp_dir is not env mutation).
        let path = std::env::temp_dir().join(format!(
            "heartbit-trace-test-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        write_trace(&output, &path).expect("write_trace should succeed");

        let bytes = std::fs::read_to_string(&path).expect("trace file should exist");
        let _ = std::fs::remove_file(&path);

        let parsed: serde_json::Value =
            serde_json::from_str(&bytes).expect("trace file should be valid JSON");
        assert_eq!(parsed["result"], "all done");
        assert_eq!(parsed["tokens_used"]["input_tokens"], 1234);
        assert_eq!(parsed["tokens_used"]["output_tokens"], 56);
        // estimated_cost_usd is skip_serializing_if Option::is_none — must be
        // present here since we set it (the adapter relies on this).
        assert_eq!(parsed["estimated_cost_usd"], 0.0042);
    }

    #[test]
    fn run_subcommand_parses_trace_file_flag_before_task() {
        // Locks the clap wiring: --trace-file (flags-first) is captured into the
        // Run arm and the trailing task is collected separately.
        let cli = Cli::try_parse_from(["heartbit", "run", "--trace-file", "/x.json", "do", "it"])
            .expect("should parse");
        match cli.command {
            Some(Commands::Run {
                task, trace_file, ..
            }) => {
                assert_eq!(trace_file, Some(PathBuf::from("/x.json")));
                assert_eq!(task, vec!["do".to_string(), "it".to_string()]);
            }
            _ => panic!("expected Run subcommand"),
        }
    }

    // --- Change 2: HEARTBIT_NONINTERACTIVE guard ---

    #[test]
    fn noninteractive_enabled_recognizes_truthy_values() {
        assert!(noninteractive_enabled(Some("1")));
        assert!(noninteractive_enabled(Some("true")));
        assert!(noninteractive_enabled(Some("TRUE")));
        assert!(noninteractive_enabled(Some(" true ")));
    }

    #[test]
    fn noninteractive_disabled_by_default_and_falsey() {
        assert!(!noninteractive_enabled(None));
        assert!(!noninteractive_enabled(Some("0")));
        assert!(!noninteractive_enabled(Some("false")));
        assert!(!noninteractive_enabled(Some("")));
        assert!(!noninteractive_enabled(Some("yes")));
    }

    // --- Change 3: HEARTBIT_WORKSPACE override ---

    #[test]
    fn resolve_workspace_root_honors_override() {
        assert_eq!(
            resolve_workspace_root(Some("/task/dir"), "/home/u"),
            PathBuf::from("/task/dir")
        );
    }

    #[test]
    fn resolve_workspace_root_falls_back_when_unset_or_empty() {
        assert_eq!(
            resolve_workspace_root(None, "/home/u"),
            PathBuf::from("/home/u/.heartbit/workspaces")
        );
        // Empty / whitespace-only override is treated as unset.
        assert_eq!(
            resolve_workspace_root(Some("   "), "/home/u"),
            PathBuf::from("/home/u/.heartbit/workspaces")
        );
    }

    #[test]
    fn verbose_implies_analysis_mode() {
        // When --verbose is set without any explicit observability flag, env, or config,
        // the resolved mode should be Analysis.
        // Remove env var if set to ensure clean state.
        unsafe {
            std::env::remove_var("HEARTBIT_OBSERVABILITY");
        }
        let mode = resolve_observability(None, None, true);
        assert_eq!(mode, ObservabilityMode::Analysis);
    }

    #[test]
    fn explicit_observability_overrides_verbose() {
        // When --observability is explicitly set, it takes precedence over --verbose.
        unsafe {
            std::env::remove_var("HEARTBIT_OBSERVABILITY");
        }
        let mode = resolve_observability(Some("production"), None, true);
        assert_eq!(mode, ObservabilityMode::Production);
    }

    #[test]
    fn no_verbose_defaults_to_production() {
        // When neither --verbose nor --observability is set, default to Production.
        unsafe {
            std::env::remove_var("HEARTBIT_OBSERVABILITY");
        }
        let mode = resolve_observability(None, None, false);
        assert_eq!(mode, ObservabilityMode::Production);
    }

    #[test]
    fn config_observability_used_when_no_cli_flag() {
        // Config telemetry observability_mode is used when no CLI flag is provided.
        unsafe {
            std::env::remove_var("HEARTBIT_OBSERVABILITY");
        }
        let mode = resolve_observability(None, Some("debug"), false);
        assert_eq!(mode, ObservabilityMode::Debug);
    }

    #[test]
    fn cli_flag_overrides_config() {
        // CLI --observability flag takes precedence over config.
        unsafe {
            std::env::remove_var("HEARTBIT_OBSERVABILITY");
        }
        let mode = resolve_observability(Some("debug"), Some("production"), false);
        assert_eq!(mode, ObservabilityMode::Debug);
    }

    /// B5b regression test: lock in the per-(tenant, provider) circuit registry
    /// behavior that the per-agent provider override path depends on. The earlier
    /// fix-up commit `ae00217` was needed because no test exercised the override
    /// wiring; this test prevents the silent-regression class of bug — so the
    /// tracker's identity invariants (same scope+provider → same Arc; different
    /// scopes → different Arcs) are pinned even if `wrap_with_circuit` is later
    /// refactored.
    #[test]
    fn wrap_with_circuit_threads_scope_and_provider_name_to_tracker() {
        use heartbit::{
            BoxedProvider, CircuitConfig, CircuitTracker, CompletionRequest, CompletionResponse,
            Error as HbError, LlmProvider, OnText, TenantScope,
        };

        // Minimal stub: the test does not invoke any provider method; it only
        // needs a value that satisfies `LlmProvider` so we can build BoxedProvider.
        struct StubProvider;
        impl LlmProvider for StubProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, HbError> {
                unreachable!("stub provider must not be called from this test")
            }
            async fn stream_complete(
                &self,
                _request: CompletionRequest,
                _on_text: &OnText,
            ) -> Result<CompletionResponse, HbError> {
                unreachable!("stub provider must not be called from this test")
            }
        }

        let tracker = Arc::new(CircuitTracker::new(CircuitConfig::default()));
        let inner: Arc<BoxedProvider> = Arc::new(BoxedProvider::new(StubProvider));

        let scope_a = TenantScope::new("tenant-a");
        let scope_b = TenantScope::new("tenant-b");

        // Wrapping must succeed for distinct (scope, provider) pairs and must
        // not panic — this is the smoke test for the helper itself.
        let _wrapped_a = wrap_with_circuit(
            Arc::clone(&inner),
            Arc::clone(&tracker),
            "anthropic",
            scope_a.clone(),
        );
        let _wrapped_b = wrap_with_circuit(
            Arc::clone(&inner),
            Arc::clone(&tracker),
            "anthropic",
            scope_b.clone(),
        );

        // Same scope+provider must resolve to the SAME Arc (the registry must
        // not create a fresh circuit per call — that would defeat shared
        // failure tracking across the per-agent override and global paths).
        let circuit_a1 = tracker.circuit_for(&scope_a, "anthropic");
        let circuit_a2 = tracker.circuit_for(&scope_a, "anthropic");
        assert!(
            Arc::ptr_eq(&circuit_a1, &circuit_a2),
            "same scope+provider should resolve to same circuit"
        );

        // Different scopes (multi-tenant isolation) must resolve to distinct circuits.
        let circuit_b = tracker.circuit_for(&scope_b, "anthropic");
        assert!(
            !Arc::ptr_eq(&circuit_a1, &circuit_b),
            "different scopes should resolve to different circuits"
        );

        // Different provider names within the same scope must also be distinct.
        let circuit_a_other = tracker.circuit_for(&scope_a, "openai");
        assert!(
            !Arc::ptr_eq(&circuit_a1, &circuit_a_other),
            "different provider names should resolve to different circuits"
        );
    }
}
