use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use slint::Weak;

use heartbit::tool::Tool;
use heartbit::{
    builtin_tools, A2aClient, AgentOutput, AgentRunner, AnthropicProvider, Blackboard,
    BoxedProvider, BuiltinToolsConfig, ContextStrategy, ContextStrategyConfig, HeartbitConfig,
    InMemoryBlackboard, InMemoryKnowledgeBase, InMemoryStore, KnowledgeBase, KnowledgeSourceConfig,
    McpClient, McpServerEntry, Memory, MemoryConfig, OpenRouterProvider, Orchestrator,
    PostgresMemoryStore, RetryConfig, RetryingProvider, SubAgentConfig,
};

use crate::bridge::command::CockpitCommand;
use crate::callbacks::{
    build_on_approval, build_on_event, build_on_input, build_on_question, build_on_text,
    SharedState,
};
use crate::MainWindow;

/// Main runtime loop running on the background tokio thread.
///
/// Receives commands from the UI and executes agent/orchestrator runs.
pub async fn runtime_loop(
    mut cmd_rx: tokio::sync::mpsc::UnboundedReceiver<CockpitCommand>,
    ui_handle: Weak<MainWindow>,
    shared: Arc<SharedState>,
    config_path: Option<std::path::PathBuf>,
    approve: bool,
) {
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            CockpitCommand::SubmitTask { task } => {
                tracing::info!(%task, "received SubmitTask command");
                // Reset state for fresh run and show user message
                {
                    let mut proc = shared.processor.lock().expect("processor lock poisoned");
                    proc.reset();
                    proc.push_user_message(&task);
                }
                // Clear any stale question/approval/input state
                shared.question_tx.lock().expect("lock").take();
                {
                    let mut pq = shared.pending_question.lock().expect("lock");
                    *pq = None;
                }
                let ui = ui_handle.clone();
                let shared_reset = Arc::clone(&shared);
                ui.upgrade_in_event_loop(move |w| {
                    w.set_run_status("running".into());
                    w.set_question_pending(false);
                    crate::callbacks::sync_state_to_ui(&shared_reset, &w);
                })
                .ok();

                // Spawn task so we can listen for Cancel commands concurrently
                let task_owned = task.clone();
                let config_owned = config_path.clone();
                let ui_task = ui_handle.clone();
                let shared_task = Arc::clone(&shared);
                let mut task_handle = tokio::spawn(async move {
                    run_task(
                        &task_owned,
                        config_owned.as_deref(),
                        ui_task,
                        shared_task,
                        approve,
                    )
                    .await
                });

                // Listen for Cancel/Stop while the task runs, with 1s heartbeat
                let result = loop {
                    tokio::select! {
                        res = &mut task_handle => {
                            break match res {
                                Ok(inner) => Some(inner),
                                Err(e) if e.is_cancelled() => None,
                                Err(e) => Some(Err(e.into())),
                            };
                        }
                        Some(cmd) = cmd_rx.recv() => {
                            match cmd {
                                CockpitCommand::Cancel => {
                                    tracing::info!("cancelling running task");
                                    // Drop oneshot senders BEFORE aborting to unblock
                                    // any callback blocked in block_in_place.
                                    // build_on_approval uses block_in_place(|| rx.blocking_recv())
                                    // which doesn't yield to the runtime — abort() alone
                                    // would deadlock because the task can't reach an .await point.
                                    shared.approval_tx.lock().expect("lock").take();
                                    shared.input_tx.lock().expect("lock").take();
                                    shared.question_tx.lock().expect("lock").take();
                                    task_handle.abort();
                                }
                                CockpitCommand::Stop => {
                                    // Drop senders to unblock approval callback (see Cancel)
                                    shared.approval_tx.lock().expect("lock").take();
                                    shared.input_tx.lock().expect("lock").take();
                                    shared.question_tx.lock().expect("lock").take();
                                    task_handle.abort();
                                    return;
                                }
                                CockpitCommand::SubmitTask { .. } => {
                                    tracing::warn!("ignoring SubmitTask while task is running");
                                }
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_secs(1)) => {
                            // Heartbeat: update elapsed time in UI
                            let shared_hb = Arc::clone(&shared);
                            ui_handle.clone().upgrade_in_event_loop(move |w| {
                                crate::callbacks::sync_state_to_ui(&shared_hb, &w);
                            }).ok();
                        }
                    }
                };

                let ui = ui_handle.clone();
                let status = match result {
                    None => {
                        // Task was cancelled — clean up pending state
                        tracing::info!("task cancelled by user");
                        {
                            let mut proc =
                                shared.processor.lock().expect("processor lock poisoned");
                            proc.push_system_message("Task cancelled by user");
                            proc.clear_pending_approval();
                            proc.cancel_active_agents();
                            proc.freeze_elapsed();
                        }
                        "cancelled"
                    }
                    Some(Ok(output)) => {
                        tracing::info!(
                            tokens_in = output.tokens_used.input_tokens,
                            tokens_out = output.tokens_used.output_tokens,
                            tool_calls = output.tool_calls_made,
                            cost = ?output.estimated_cost_usd,
                            "task completed successfully"
                        );
                        {
                            let mut proc =
                                shared.processor.lock().expect("processor lock poisoned");
                            if let Some(cost) = output.estimated_cost_usd {
                                proc.set_estimated_cost(cost);
                            }
                            proc.freeze_elapsed();
                        }
                        "completed"
                    }
                    Some(Err(e)) => {
                        tracing::error!(error = %e, "task failed");
                        let error_event = heartbit::AgentEvent::RunFailed {
                            agent: "heartbit".into(),
                            error: format!("{e:#}"),
                            partial_usage: heartbit::TokenUsage::default(),
                        };
                        {
                            let mut proc =
                                shared.processor.lock().expect("processor lock poisoned");
                            proc.process_event(&error_event);
                            proc.freeze_elapsed();
                        }
                        "failed"
                    }
                };

                // Post-task cleanup: drop any orphaned oneshot senders and clear
                // question/approval/input state. This prevents stale callbacks from
                // blocking the next run (e.g., error during mid-approval).
                cleanup_post_task(&shared);

                let shared_snap = Arc::clone(&shared);
                let status_owned: slint::SharedString = status.into();
                ui.upgrade_in_event_loop(move |w| {
                    w.set_run_status(status_owned);
                    w.set_input_requested(false);
                    w.set_question_pending(false);
                    crate::callbacks::sync_state_to_ui(&shared_snap, &w);
                    w.invoke_focus_input();
                })
                .ok();
            }
            CockpitCommand::Cancel => {
                tracing::debug!("Cancel received but no task is running");
            }
            CockpitCommand::Stop => break,
        }
    }
}

/// Drop orphaned oneshot senders and clear pending question state.
///
/// Called after every task completes (success, error, or cancellation) to ensure
/// no stale callbacks block the next run.
fn cleanup_post_task(shared: &SharedState) {
    shared.approval_tx.lock().expect("lock").take();
    shared.input_tx.lock().expect("lock").take();
    shared.question_tx.lock().expect("lock").take();
    let mut pq = shared.pending_question.lock().expect("lock");
    *pq = None;
}

/// Execute a task using either config-based orchestrator or env-based single agent.
async fn run_task(
    task: &str,
    config_path: Option<&Path>,
    ui_handle: Weak<MainWindow>,
    shared: Arc<SharedState>,
    approve: bool,
) -> Result<AgentOutput> {
    match config_path {
        Some(path) => run_from_config(path, task, ui_handle, shared, approve).await,
        None => run_from_env(task, ui_handle, shared, approve).await,
    }
}

/// Run with a config file — multi-agent orchestrator.
async fn run_from_config(
    path: &Path,
    task: &str,
    ui_handle: Weak<MainWindow>,
    shared: Arc<SharedState>,
    approve: bool,
) -> Result<AgentOutput> {
    let config = HeartbitConfig::from_file(path)
        .with_context(|| format!("failed to load config from {}", path.display()))?;

    let provider = build_provider_from_config(&config)?;

    let on_event = build_on_event(Arc::clone(&shared), ui_handle.clone());
    let on_text = build_on_text(Arc::clone(&shared), ui_handle.clone());
    let on_question = build_on_question(Arc::clone(&shared), ui_handle.clone());

    let mut builder = Orchestrator::builder(provider)
        .max_turns(config.orchestrator.max_turns)
        .max_tokens(config.orchestrator.max_tokens)
        .on_text(on_text)
        .on_event(on_event);

    if approve {
        let on_approval = build_on_approval(Arc::clone(&shared));
        builder = builder.on_approval(on_approval);
    }

    if let Some(enable) = config.orchestrator.enable_squads {
        builder = builder.enable_squads(enable);
    }

    // Wire orchestrator-level reasoning effort from config
    if let Some(ref effort) = config.orchestrator.reasoning_effort {
        builder = builder.reasoning_effort(heartbit::config::parse_reasoning_effort(effort)?);
    }
    // Wire orchestrator-level reflection from config
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
    // Wire permission rules from config + learned permissions
    {
        let mut ruleset = heartbit::PermissionRuleset::new(config.permissions.clone());
        if let Some(learned) = load_learned_permissions() {
            if let Ok(guard) = learned.lock() {
                ruleset.append_rules(guard.rules());
            }
            builder = builder.learned_permissions(learned);
        }
        if !ruleset.is_empty() {
            builder = builder.permission_rules(ruleset);
        }
    }

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
            if let Some(threshold) = config.orchestrator.summarize_threshold {
                builder = builder.summarize_threshold(threshold);
            }
        }
    }
    if let Some(secs) = config.orchestrator.tool_timeout_seconds {
        builder = builder.tool_timeout(std::time::Duration::from_secs(secs));
    }
    if let Some(max) = config.orchestrator.max_tool_output_bytes {
        builder = builder.max_tool_output_bytes(max);
    }
    if let Some(secs) = config.orchestrator.run_timeout_seconds {
        builder = builder.run_timeout(std::time::Duration::from_secs(secs));
    }

    let builtins = create_builtin_tools(Some(on_question));
    let retry = retry_config_from(&config);

    for agent in &config.agents {
        let mcp_tools = load_mcp_tools(&agent.name, &agent.mcp_servers).await;
        let a2a_tools = load_a2a_tools(&agent.name, &agent.a2a_agents).await;
        let mut tools = builtins.clone();
        tools.extend(mcp_tools);
        tools.extend(a2a_tools);

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

        let agent_provider = match &agent.provider {
            Some(p) => Some(build_agent_provider(p, retry.clone())?),
            None => None,
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
            run_timeout: agent
                .run_timeout_seconds
                .map(std::time::Duration::from_secs),
            guardrails: vec![],
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
        });
    }

    // Shared memory
    if let Some(ref memory_config) = config.memory {
        let memory: Arc<dyn Memory> = match memory_config {
            MemoryConfig::InMemory => Arc::new(InMemoryStore::new()),
            MemoryConfig::Postgres { database_url, .. } => {
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

    let blackboard: Arc<dyn Blackboard> = Arc::new(InMemoryBlackboard::new());
    {
        let mut bb_guard = shared.blackboard.lock().expect("blackboard lock poisoned");
        *bb_guard = Some(Arc::clone(&blackboard));
    }
    builder = builder.blackboard(blackboard);

    if let Some(ref knowledge_config) = config.knowledge {
        let kb = load_knowledge_base(knowledge_config).await?;
        builder = builder.knowledge(kb);
    }

    // Wire hierarchical instruction files (HEARTBIT.md)
    if let Ok(cwd) = std::env::current_dir() {
        let paths = heartbit::discover_instruction_files(&cwd);
        if !paths.is_empty() {
            if let Ok(text) = heartbit::load_instructions(&paths) {
                if !text.is_empty() {
                    builder = builder.instruction_text(text);
                }
            }
        }
    }

    // Wire LSP integration if configured
    if let Some(ref lsp_config) = config.lsp {
        if lsp_config.enabled {
            let workspace_root = std::env::current_dir().unwrap_or_default();
            builder = builder.lsp_manager(Arc::new(heartbit::LspManager::new(workspace_root)));
        }
    }

    let mut orchestrator = builder.build()?;
    let mut output = orchestrator.run(task).await?;

    let has_overrides = config.agents.iter().any(|a| a.provider.is_some());
    if !has_overrides {
        output.estimated_cost_usd =
            heartbit::estimate_cost(&config.provider.model, &output.tokens_used);
    }
    Ok(output)
}

/// Run with environment variables — single agent.
async fn run_from_env(
    task: &str,
    ui_handle: Weak<MainWindow>,
    shared: Arc<SharedState>,
    approve: bool,
) -> Result<AgentOutput> {
    let provider = build_provider_from_env()?;
    let model = resolve_model_from_env();

    let on_event = build_on_event(Arc::clone(&shared), ui_handle.clone());
    let on_text = build_on_text(Arc::clone(&shared), ui_handle.clone());
    let on_question = build_on_question(Arc::clone(&shared), ui_handle.clone());

    let mut tools = create_builtin_tools(Some(on_question));
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

    let on_input = build_on_input(Arc::clone(&shared), ui_handle.clone());

    let mut agent_builder = AgentRunner::builder(provider)
        .name("heartbit")
        .system_prompt(
            "You are a skilled software engineer. Use the available tools to accomplish \
             the task. Work methodically: read relevant files before making changes, \
             verify your work by running tests, and explain your reasoning. \
             When modifying code, always read the file first.",
        )
        .tools(tools)
        .max_turns(max_turns)
        .summarize_threshold(summarize_threshold)
        .max_tool_output_bytes(max_tool_output_bytes)
        .tool_timeout(std::time::Duration::from_secs(tool_timeout_secs))
        .on_text(on_text)
        .on_event(on_event)
        .on_input(on_input);

    if approve {
        let on_approval = build_on_approval(Arc::clone(&shared));
        agent_builder = agent_builder.on_approval(on_approval);
    }
    // Wire learned permissions from disk
    if let Some(learned) = load_learned_permissions() {
        if let Ok(guard) = learned.lock() {
            if !guard.rules().is_empty() {
                let mut ruleset = heartbit::PermissionRuleset::default();
                ruleset.append_rules(guard.rules());
                agent_builder = agent_builder.permission_rules(ruleset);
            }
        }
        agent_builder = agent_builder.learned_permissions(learned);
    }
    if let Ok(effort_str) = std::env::var("HEARTBIT_REASONING_EFFORT") {
        agent_builder =
            agent_builder.reasoning_effort(heartbit::config::parse_reasoning_effort(&effort_str)?);
    }
    if std::env::var("HEARTBIT_ENABLE_REFLECTION")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        agent_builder = agent_builder.enable_reflection(true);
    }
    if let Ok(threshold) = std::env::var("HEARTBIT_COMPRESSION_THRESHOLD") {
        if let Ok(n) = threshold.parse::<usize>() {
            agent_builder = agent_builder.tool_output_compression_threshold(n);
        }
    }
    if let Ok(max) = std::env::var("HEARTBIT_MAX_TOOLS_PER_TURN") {
        if let Ok(n) = max.parse::<usize>() {
            agent_builder = agent_builder.max_tools_per_turn(n);
        }
    }
    if let Ok(max) = std::env::var("HEARTBIT_MAX_IDENTICAL_TOOL_CALLS") {
        if let Ok(n) = max.parse::<u32>() {
            agent_builder = agent_builder.max_identical_tool_calls(n);
        }
    }
    // Wire hierarchical instruction files (HEARTBIT.md)
    if let Ok(cwd) = std::env::current_dir() {
        let paths = heartbit::discover_instruction_files(&cwd);
        if !paths.is_empty() {
            if let Ok(text) = heartbit::load_instructions(&paths) {
                if !text.is_empty() {
                    agent_builder = agent_builder.instruction_text(text);
                }
            }
        }
    }
    // Wire LSP integration if enabled via env
    if std::env::var("HEARTBIT_LSP_ENABLED")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(false)
    {
        let workspace_root = std::env::current_dir().unwrap_or_default();
        agent_builder =
            agent_builder.lsp_manager(Arc::new(heartbit::LspManager::new(workspace_root)));
    }

    let runner = agent_builder.build()?;

    let mut output = runner.execute(task).await?;
    output.estimated_cost_usd = heartbit::estimate_cost(&model, &output.tokens_used);
    Ok(output)
}

// --- Helper functions ported from CLI ---

fn retry_config_from(config: &HeartbitConfig) -> Option<RetryConfig> {
    config.provider.retry.as_ref().map(RetryConfig::from)
}

fn build_provider_from_config(config: &HeartbitConfig) -> Result<Arc<BoxedProvider>> {
    let retry = retry_config_from(config);
    match config.provider.name.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY env var required for anthropic provider")?;
            let base = if config.provider.prompt_caching {
                AnthropicProvider::with_prompt_caching(&api_key, &config.provider.model)
            } else {
                AnthropicProvider::new(&api_key, &config.provider.model)
            };
            match retry {
                Some(rc) => Ok(Arc::new(BoxedProvider::new(RetryingProvider::new(
                    base, rc,
                )))),
                None => Ok(Arc::new(BoxedProvider::new(base))),
            }
        }
        "openrouter" => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY env var required for openrouter provider")?;
            let base = OpenRouterProvider::new(&api_key, &config.provider.model);
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

fn build_agent_provider(
    config: &heartbit::AgentProviderConfig,
    retry: Option<RetryConfig>,
) -> Result<Arc<BoxedProvider>> {
    match config.name.as_str() {
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY env var required for anthropic provider")?;
            let base = if config.prompt_caching {
                AnthropicProvider::with_prompt_caching(&api_key, &config.model)
            } else {
                AnthropicProvider::new(&api_key, &config.model)
            };
            match retry {
                Some(rc) => Ok(Arc::new(BoxedProvider::new(RetryingProvider::new(
                    base, rc,
                )))),
                None => Ok(Arc::new(BoxedProvider::new(base))),
            }
        }
        "openrouter" => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY env var required for openrouter provider")?;
            let base = OpenRouterProvider::new(&api_key, &config.model);
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

fn build_provider_from_env() -> Result<Arc<BoxedProvider>> {
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
            let prompt_caching = std::env::var("HEARTBIT_PROMPT_CACHING")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let base = if prompt_caching {
                AnthropicProvider::with_prompt_caching(&api_key, &model)
            } else {
                AnthropicProvider::new(&api_key, &model)
            };
            Ok(Arc::new(BoxedProvider::new(
                RetryingProvider::with_defaults(base),
            )))
        }
        "openrouter" => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY env var required for openrouter provider")?;
            let model = std::env::var("HEARTBIT_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-sonnet-4".into());
            Ok(Arc::new(BoxedProvider::new(
                RetryingProvider::with_defaults(OpenRouterProvider::new(&api_key, &model)),
            )))
        }
        other => bail!("Unknown provider: {other}. Use 'anthropic' or 'openrouter'."),
    }
}

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

async fn load_mcp_tools(agent_name: &str, mcp_servers: &[McpServerEntry]) -> Vec<Arc<dyn Tool>> {
    let mut tools = Vec::new();
    for entry in mcp_servers {
        let url = entry.url();
        tracing::info!(agent = %agent_name, url = %url, "connecting to MCP server");
        let result = match entry.auth_header() {
            Some(auth) => McpClient::connect_with_auth(url, auth).await,
            None => McpClient::connect(url).await,
        };
        match result {
            Ok(client) => {
                for tool in client.into_tools() {
                    tracing::info!(tool = %tool.definition().name, "registered MCP tool");
                    tools.push(tool);
                }
            }
            Err(e) => {
                tracing::warn!(
                    agent = %agent_name, url = %url, error = %e,
                    "failed to connect to MCP server, skipping"
                );
            }
        }
    }
    tools
}

async fn load_a2a_tools(agent_name: &str, a2a_agents: &[McpServerEntry]) -> Vec<Arc<dyn Tool>> {
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
                    tracing::info!(tool = %tool.definition().name, "registered A2A agent tool");
                    tools.push(tool);
                }
            }
            Err(e) => {
                tracing::warn!(
                    agent = %agent_name, url = %url, error = %e,
                    "failed to discover A2A agent, skipping"
                );
            }
        }
    }
    tools
}

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
                let path = Path::new(path);
                match loader::load_file(&*kb, path, &chunk_config).await {
                    Ok(count) => {
                        tracing::info!(path = %path.display(), chunks = count, "indexed knowledge file");
                    }
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "failed to load knowledge file");
                    }
                }
            }
            KnowledgeSourceConfig::Glob { pattern } => {
                match loader::load_glob(&*kb, pattern, &chunk_config).await {
                    Ok(count) => {
                        tracing::info!(pattern = %pattern, chunks = count, "indexed knowledge glob");
                    }
                    Err(e) => {
                        tracing::warn!(pattern = %pattern, error = %e, "failed to load knowledge glob");
                    }
                }
            }
            KnowledgeSourceConfig::Url { url } => {
                match loader::load_url(&*kb, url, &chunk_config).await {
                    Ok(count) => {
                        tracing::info!(url = %url, chunks = count, "indexed knowledge URL");
                    }
                    Err(e) => {
                        tracing::warn!(url = %url, error = %e, "failed to load knowledge URL");
                    }
                }
            }
        }
    }

    Ok(kb)
}

fn create_builtin_tools(on_question: Option<Arc<heartbit::OnQuestion>>) -> Vec<Arc<dyn Tool>> {
    builtin_tools(BuiltinToolsConfig {
        on_question,
        ..Default::default()
    })
}

/// Load learned permissions from the default path (`~/.config/heartbit/permissions.toml`).
fn load_learned_permissions() -> Option<Arc<std::sync::Mutex<heartbit::LearnedPermissions>>> {
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

fn parse_env<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn parse_csv_env(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}
