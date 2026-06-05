//! `heartbit-tui` — a terminal UI to chat with the Heartbit framework, in the
//! spirit of Claude Code / codex / opencode.
//!
//! Architecture (see `tasks/tui-2026-06-02.md`):
//! - **Main thread**: the async UI loop (ratatui draw + crossterm `EventStream`),
//!   owning the pure [`App`] state. All terminal/channel I/O lives here.
//! - **Dedicated agent thread** (built lazily on the first message): a
//!   `multi_thread` tokio runtime (≥2 workers, so a blocking tool can never
//!   starve the cancellation poll) that `block_on`s `AgentRunner::execute()`.
//!   The runner's synchronous callbacks bridge to the UI over channels; `on_input`
//!   (async) feeds turns back so conversation history is preserved.
//! - `on_approval` is synchronous → it round-trips over a blocking
//!   `std::sync::mpsc` (the agent thread idles, waiting on the human). Zero core
//!   changes.
//! - The OpenRouter token can come from `OPENROUTER_API_KEY`, the config file
//!   (`~/.config/heartbit/tui.toml`), or be entered/updated from inside the TUI
//!   (a masked modal at startup if unset, or `/key`). The agent's `bash` gets a
//!   no-secrets env allowlist, so the token never enters the tool environment.

mod app;
mod cells;
mod composer;
mod config;
mod diff;
mod markdown;
mod models;
mod msg;
mod session;
mod ui;

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{Event, EventStream, KeyEventKind};
use futures::StreamExt;
use heartbit_core::tool::builtins::{BuiltinToolsConfig, builtin_tools};
use heartbit_core::{
    AgentEvent, AgentRunner, ApprovalDecision, BoxedProvider, InterruptHandle, OnApproval, OnEvent,
    OnInput, OnText, OpenRouterProvider, Orchestrator, PermissionAction, PermissionRule,
    PermissionRuleset, RetryingProvider, SubAgentConfig,
};
use tokio::sync::Mutex;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::app::{App, Effect};
use crate::cells::Cell;
use crate::msg::{Msg, PendingTool};

const SYSTEM_PROMPT: &str = "You are Heartbit, an expert software engineering assistant running in a \
terminal UI, with tools to read, search, edit, and run code in the user's workspace. Be concise and \
direct. When you change code, make the smallest correct change, then verify it. Prefer showing your \
work through tool use over describing it. When a task is done, say so briefly.";

const DEFAULT_MODEL: &str = "qwen/qwen3-235b-a22b-2507";

/// Opt-in file logging for diagnosing the interrupt chain. The TUI owns the
/// terminal (alt-screen) so stderr is unusable; route `heartbit::interrupt`
/// checkpoints to a file instead. Set `HEARTBIT_TUI_DEBUG=1` (→
/// `/tmp/heartbit-tui-debug.log`) or `HEARTBIT_TUI_DEBUG=/path/to/log`.
fn init_debug_logging() {
    let Ok(val) = std::env::var("HEARTBIT_TUI_DEBUG") else {
        return;
    };
    if val.is_empty() {
        return;
    }
    let path = if val == "1" || val == "true" {
        "/tmp/heartbit-tui-debug.log".to_string()
    } else {
        val
    };
    let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(std::sync::Mutex::new(file))
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            "heartbit::interrupt=info",
        ))
        .try_init();
    tracing::info!(target: "heartbit::interrupt", "--- heartbit-tui debug logging started ---");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_debug_logging();
    let cfg = config::TuiConfig::load();
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .or(cfg.openrouter_api_key.clone());
    let model = std::env::var("HEARTBIT_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
        .or(cfg.model.clone())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let has_anthropic = std::env::var("ANTHROPIC_API_KEY")
        .map(|s| !s.is_empty())
        .unwrap_or(false);

    // agent -> UI (sync sends from callbacks); UI -> agent (async on_input feed).
    let (ui_tx, ui_rx) = tokio::sync::mpsc::unbounded_channel::<Msg>();
    let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let input_rx = Arc::new(Mutex::new(input_rx));
    let cwd = std::env::current_dir()?;

    let mut app = App::new(model);
    app.api_key = api_key;
    app.has_fallback_provider = has_anthropic;
    app.mcp_servers = cfg.mcp_servers.clone();
    app.multi_agent = cfg.multi_agent;
    app.context_recall = cfg.context_recall;
    // Seed the roster's available squad so the full pool is visible in the TUI
    // (and the user can see when only part of it is actually dispatched).
    if cfg.multi_agent {
        app.squad = app::DEFAULT_SQUAD.iter().map(|s| s.to_string()).collect();
    }
    // Fetch the OpenRouter catalog at startup (public endpoint) so the status-line
    // context bar knows the model's window and the /model picker is pre-warmed.
    app.models_loading = true;
    app.effects.push(Effect::FetchModels);
    // No provider configured at all → open the key prompt immediately.
    if app.api_key.is_none() && !has_anthropic {
        app.modal = Some(app::Modal::KeyEntry(app::KeyEntryModal::default()));
    }

    let interrupt = InterruptHandle::new();
    // Shared permission posture (0=default,1=accept-edits,2=plan,3=auto), cycled
    // by the UI (Shift+Tab) and read live by the agent thread's `on_approval`.
    let perm_mode = Arc::new(std::sync::atomic::AtomicU8::new(0));
    // A per-launch session id (time + pid) for transcript persistence.
    let session_id = format!(
        "{:x}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        std::process::id()
    );

    let mut terminal = ratatui::init();
    // Capture the mouse so the wheel arrives as scroll events we route to the
    // transcript — without it, terminals translate the wheel into ↑/↓ arrows
    // (which would scroll the composer's command history instead).
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture);
    let result = run_ui(
        &mut terminal,
        &mut app,
        ui_rx,
        input_tx,
        ui_tx,
        input_rx,
        cwd,
        interrupt,
        perm_mode,
        session_id,
    )
    .await;
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
    ratatui::restore();
    result
}

/// Resolve a provider. OpenRouter is preferred (the project's qwen setup); an
/// `ANTHROPIC_API_KEY` env var is a fallback when no OpenRouter key is configured.
fn build_provider(
    openrouter_key: Option<String>,
    model: &str,
) -> anyhow::Result<Arc<BoxedProvider>> {
    if let Some(key) = openrouter_key {
        let base = OpenRouterProvider::new(key, model);
        return Ok(Arc::new(BoxedProvider::new(
            RetryingProvider::with_defaults(base),
        )));
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY")
        && !key.is_empty()
    {
        // The default model is an OpenRouter id; pick a sane Claude id for Anthropic.
        let anthropic_model = if model.contains('/') {
            "claude-sonnet-4-6"
        } else {
            model
        };
        let base = heartbit_core::AnthropicProvider::new(&key, anthropic_model);
        return Ok(Arc::new(BoxedProvider::new(
            RetryingProvider::with_defaults(base),
        )));
    }
    anyhow::bail!("no OpenRouter API key configured (set one with /key or OPENROUTER_API_KEY)")
}

/// The agent engine the TUI drives: a single [`AgentRunner`], or a multi-agent
/// [`Orchestrator`] (dynamic delegation + squads). Both expose the same
/// interactive loop (one call drives a whole `on_input` multi-turn session).
enum Engine {
    Single(Box<AgentRunner<BoxedProvider>>),
    Multi(Box<Orchestrator<BoxedProvider>>),
}

impl Engine {
    /// Run the (multi-turn) session, starting with `first`.
    async fn run(&mut self, first: &str) -> anyhow::Result<()> {
        match self {
            Engine::Single(r) => {
                r.execute(first).await?;
            }
            Engine::Multi(o) => {
                o.run(first).await?;
            }
        }
        Ok(())
    }
}

/// A fresh set of workspace-rooted builtin tools (each call gets its own
/// `FileTracker` etc.). The API key never enters the tool env (safe allowlist).
fn fresh_builtins(
    cwd: &std::path::Path,
    context_recall_store: Option<&Arc<heartbit_core::ContextRecallStore>>,
) -> Vec<Arc<dyn heartbit_core::tool::Tool>> {
    let mut tool_cfg = BuiltinToolsConfig::default();
    tool_cfg.workspace = Some(cwd.to_path_buf());
    tool_cfg.dangerous_tools = true;
    tool_cfg.env_policy = heartbit_core::workspace::EnvPolicy::default();
    // When present, registers fetch_full_output / recall_context for restore-on-demand.
    tool_cfg.context_recall_store = context_recall_store.cloned();
    builtin_tools(tool_cfg)
}

/// A gentle session-prune config paired with context-recall: keep the last few
/// turns at full fidelity, truncate older tool results to ~1KB (each carrying a
/// `fetch_full_output(<id>)` marker so the full content is recoverable).
fn gentle_prune_config() -> heartbit_core::SessionPruneConfig {
    heartbit_core::SessionPruneConfig {
        keep_recent_n: 3,
        pruned_tool_result_max_bytes: 1024,
        preserve_task: true,
    }
}

/// Two capable generalist sub-agents (≥2 → dynamic squads auto-enable). Each gets
/// fresh builtins + a clone of the connected MCP tools, so delegated work can
/// actually read/edit/run, not just talk. Distinct descriptions guide routing.
fn default_sub_agents(
    cwd: &std::path::Path,
    mcp_tools: &[Arc<dyn heartbit_core::tool::Tool>],
) -> Vec<SubAgentConfig> {
    let make = |name: &str, description: &str, prompt: &str| {
        let mut tools = fresh_builtins(cwd, None);
        tools.extend(mcp_tools.iter().cloned());
        SubAgentConfig {
            name: name.into(),
            description: description.into(),
            system_prompt: prompt.into(),
            tools,
            max_turns: Some(60),
            max_tokens: Some(8192),
            ..Default::default()
        }
    };
    vec![
        make(
            app::DEFAULT_SQUAD[0],
            "General implementation agent: reads, searches, edits, and runs code in the workspace. Use for concrete file changes, builds, tests, and command execution.",
            "You are a focused implementation engineer. Do the delegated task end-to-end with the tools, make the smallest correct change, verify it, and report a concise result.",
        ),
        make(
            app::DEFAULT_SQUAD[1],
            "Investigation agent: explores the codebase and gathers facts (search, read files, run read-only commands). Use to understand, locate, or analyze before changes.",
            "You are a careful researcher. Investigate the delegated question using the tools, then report concrete findings (file paths, line numbers, facts) — do not make changes unless asked.",
        ),
    ]
}

/// Build the agent engine (single or multi-agent), wiring the synchronous
/// callbacks to the UI channels. The OpenRouter token is passed only to the
/// provider, never into the tool environment (bash gets a no-secrets allowlist).
#[allow(clippy::too_many_arguments)]
async fn build_engine(
    api_key: Option<String>,
    model: &str,
    ui_tx: UnboundedSender<Msg>,
    input_rx: Arc<Mutex<UnboundedReceiver<String>>>,
    cwd: PathBuf,
    interrupt: InterruptHandle,
    mcp_servers: Vec<config::McpServerSpec>,
    multi_agent: bool,
    context_recall: bool,
    perm_mode: Arc<std::sync::atomic::AtomicU8>,
) -> anyhow::Result<Engine> {
    let provider = build_provider(api_key, model)?;

    // Connect MCP once (on this thread's runtime — the stdio transport binds to
    // its spawn runtime). The tools are Arc, shared across agents.
    let mut mcp_tools: Vec<Arc<dyn heartbit_core::tool::Tool>> = Vec::new();
    for spec in &mcp_servers {
        let label = spec.label();
        let _ = ui_tx.send(Msg::Notice(format!("connecting MCP {label}…")));
        match connect_mcp(spec).await {
            Ok(t) => {
                let _ = ui_tx.send(Msg::Notice(format!(
                    "MCP {label}: connected ({} tools)",
                    t.len()
                )));
                mcp_tools.extend(t);
            }
            Err(e) => {
                let _ = ui_tx.send(Msg::Notice(format!("MCP {label}: failed — {e}")));
            }
        }
    }
    // Context restore-on-demand (single-agent path only): a per-run store that
    // indexes every tool output so the gentle pruner's truncation is reversible
    // (the model restores via fetch_full_output / recall_context).
    let recall_store = (context_recall && !multi_agent)
        .then(|| Arc::new(heartbit_core::ContextRecallStore::new()));
    // The single agent's tools: builtins FIRST so MCP can't shadow a trusted one.
    let mut tools = fresh_builtins(&cwd, recall_store.as_ref());
    tools.extend(mcp_tools.iter().cloned());

    let on_text: Arc<OnText> = {
        let tx = ui_tx.clone();
        Arc::new(move |s: &str| {
            let _ = tx.send(Msg::StreamDelta(s.to_string()));
        })
    };
    // Reasoning models stream their chain-of-thought separately — surface it live
    // (dimmed) ahead of the answer.
    let on_reasoning: Arc<heartbit_core::OnReasoning> = {
        let tx = ui_tx.clone();
        Arc::new(move |s: &str| {
            let _ = tx.send(Msg::ReasoningDelta(s.to_string()));
        })
    };
    let on_event: Arc<OnEvent> = {
        let tx = ui_tx.clone();
        Arc::new(move |e: AgentEvent| {
            // Opt-in diagnostics (HEARTBIT_TUI_DEBUG): surface tool dispatch +
            // multi-agent delegation, which the TUI owns the terminal so can't show.
            if let AgentEvent::ToolCallStarted {
                tool_name, agent, ..
            } = &e
            {
                tracing::info!(target: "heartbit::interrupt", tool_started = %tool_name, agent = %agent, "tool dispatched");
            }
            if let AgentEvent::SubAgentsDispatched { agents, .. } = &e {
                tracing::info!(target: "heartbit::interrupt", ?agents, "sub-agents dispatched");
            }
            if let Some(m) = Msg::from_event(e) {
                let _ = tx.send(m);
            }
        })
    };
    let on_approval: Arc<OnApproval> = {
        let tx = ui_tx.clone();
        let perm_mode = perm_mode.clone();
        Arc::new(move |calls: &[heartbit_core::llm::types::ToolCall]| {
            for c in calls {
                tracing::info!(target: "heartbit::interrupt", approval_for = %c.name, "on_approval");
            }
            // Permission posture gates the prompt (read live, cross-thread):
            //   auto → allow all; accept-edits → allow if the whole batch is edits;
            //   plan → deny any mutating tool (read-only); default → ask (modal).
            let is_edit = |n: &str| matches!(n, "edit" | "write" | "patch");
            let is_mutating = |n: &str| matches!(n, "edit" | "write" | "patch" | "bash");
            match perm_mode.load(std::sync::atomic::Ordering::Relaxed) {
                3 => return ApprovalDecision::Allow,
                1 if calls.iter().all(|c| is_edit(&c.name)) => return ApprovalDecision::Allow,
                2 if calls.iter().any(|c| is_mutating(&c.name)) => return ApprovalDecision::Deny,
                _ => {}
            }
            let tools = calls
                .iter()
                .map(|c| PendingTool {
                    name: c.name.clone(),
                    input: serde_json::to_string(&c.input).unwrap_or_default(),
                })
                .collect();
            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
            if tx
                .send(Msg::Approval {
                    tools,
                    reply: reply_tx,
                })
                .is_err()
            {
                return ApprovalDecision::Deny;
            }
            reply_rx.recv().unwrap_or(ApprovalDecision::Deny)
        })
    };
    let on_input: Arc<OnInput> = {
        let rx = input_rx;
        Arc::new(move || {
            let rx = rx.clone();
            Box::pin(async move { rx.lock().await.recv().await })
                as Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
        })
    };

    // Load project context (AGENTS.md / CLAUDE.md / HEARTBIT.md) by walking up to
    // the git root + the global config — like Claude Code's CLAUDE.md, but the
    // cross-tool AGENTS.md standard takes priority. Injected into the system prompt.
    let context_paths = heartbit_core::discover_instruction_files(&cwd);
    let project_context = heartbit_core::load_instructions(&context_paths).unwrap_or_default();
    if !project_context.is_empty() {
        let names: Vec<String> = context_paths
            .iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .collect();
        let _ = ui_tx.send(Msg::Notice(format!(
            "loaded project context: {}",
            names.join(", ")
        )));
    }

    if multi_agent {
        // Multi-agent: an Orchestrator that dynamically delegates to a squad of
        // capable sub-agents (delegate_task / form_squad). The orchestrator agent
        // is a router; the sub-agents hold the tools. Same callbacks + interrupt
        // as the single agent (one run() drives the whole multi-turn session).
        let _ = ui_tx.send(Msg::Notice(
            "multi-agent workflow ON — orchestrator + worker/researcher squad".into(),
        ));
        let mut builder = Orchestrator::builder(provider)
            .max_turns(300)
            .workspace(cwd.clone())
            .instruction_text(project_context)
            .permission_rules(default_permissions())
            .on_text(on_text)
            .on_event(on_event)
            .on_approval(on_approval)
            .on_input(on_input)
            .interrupt(interrupt);
        for cfg in default_sub_agents(&cwd, &mcp_tools) {
            builder = builder.sub_agent_full(cfg);
        }
        let orch = builder.build()?;
        Ok(Engine::Multi(Box::new(orch)))
    } else {
        let mut rb = AgentRunner::builder(provider)
            .name("heartbit")
            .system_prompt(SYSTEM_PROMPT)
            .instruction_text(project_context)
            .tools(tools)
            .max_turns(300)
            .workspace(cwd)
            .permission_rules(default_permissions())
            .on_text(on_text)
            .on_reasoning(on_reasoning)
            .on_event(on_event)
            .on_approval(on_approval)
            .on_input(on_input)
            .interrupt(interrupt);
        // Restore-on-demand: share the SAME store for indexing, and pair it with a
        // gentle pruner so old tool outputs truncate to a restorable marker.
        if let Some(store) = &recall_store {
            rb = rb
                .context_recall_store(store.clone())
                .session_prune_config(gentle_prune_config());
            let _ = ui_tx.send(Msg::Notice(
                "context restore-on-demand ON — old tool outputs are recoverable via fetch_full_output / recall_context".into(),
            ));
        }
        let runner = rb.build()?;
        Ok(Engine::Single(Box::new(runner)))
    }
}

/// Connect one configured MCP server and return its tools. A preset spins up its
/// bundled server (e.g. chrome-devtools); a command is spawned over stdio. The
/// returned tools own the server subprocess (killed when the runner is dropped).
async fn connect_mcp(
    spec: &config::McpServerSpec,
) -> anyhow::Result<Vec<Arc<dyn heartbit_core::tool::Tool>>> {
    if let Some(preset) = &spec.preset {
        Ok(heartbit_core::connect_preset(preset).await?)
    } else if let Some(command) = &spec.command {
        let env = std::collections::HashMap::new();
        let client = heartbit_core::McpClient::connect_stdio(command, &spec.args, &env).await?;
        Ok(client.into_tools())
    } else {
        anyhow::bail!("invalid MCP server (no preset or command)")
    }
}

/// A Claude-Code-like default policy: read-only tools run silently; everything
/// that can mutate the workspace (write/edit/patch/bash/…) asks the human via the
/// approval modal. `[a]`lways-allow persists as a learned rule for that tool.
fn default_permissions() -> PermissionRuleset {
    let allow = |tool: &str| PermissionRule {
        tool: tool.into(),
        pattern: "*".into(),
        action: PermissionAction::Allow,
    };
    PermissionRuleset::new(vec![
        allow("read"),
        allow("grep"),
        allow("glob"),
        allow("list"),
        allow("todoread"),
        allow("todowrite"),
        PermissionRule {
            tool: "*".into(),
            pattern: "*".into(),
            action: PermissionAction::Ask,
        },
    ])
}

/// Walk the project for `@`-mention autocomplete: relative file paths, skipping
/// dot-dirs and common build/vendor dirs, capped so huge repos stay responsive.
fn walk_project_files(root: &std::path::Path) -> Vec<String> {
    const SKIP: [&str; 6] = [".git", "target", "node_modules", ".venv", "dist", "build"];
    const CAP: usize = 8000;
    let mut out = Vec::new();
    let walker = walkdir::WalkDir::new(root)
        .max_depth(12)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            // skip hidden dirs and known build/vendor dirs (but allow files)
            !(e.file_type().is_dir() && (name.starts_with('.') || SKIP.contains(&name.as_ref())))
        });
    for entry in walker.flatten() {
        if entry.file_type().is_file()
            && let Ok(rel) = entry.path().strip_prefix(root)
        {
            out.push(rel.to_string_lossy().into_owned());
            if out.len() >= CAP {
                break;
            }
        }
    }
    out.sort();
    out
}

/// Translate a raw crossterm event into a [`Msg`] (or ignore it).
fn translate(event: Event) -> Option<Msg> {
    use crossterm::event::MouseEventKind;
    match event {
        Event::Key(k) if k.kind == KeyEventKind::Press => Some(Msg::Key(k)),
        Event::Paste(s) => Some(Msg::Paste(s)),
        Event::Resize(..) => Some(Msg::Resize),
        // Mouse capture is on, so the wheel arrives as scroll events (not arrow
        // keys) — route it to the transcript, leaving ↑/↓ for command history.
        Event::Mouse(m) => match m.kind {
            MouseEventKind::ScrollUp => Some(Msg::WheelUp),
            MouseEventKind::ScrollDown => Some(Msg::WheelDown),
            _ => None,
        },
        _ => None,
    }
}

/// Spawn the agent thread if a provider is configured; returns whether it was
/// spawned. Building the runner here (on the agent thread) connects MCP servers
/// **eagerly** — at startup, or the moment a key is set — so they're ready
/// before the first message (Claude-Code-style). The thread then waits for the
/// first user message on the same channel `on_input` uses, runs `execute`, and
/// `on_input` feeds subsequent turns. A multi_thread runtime keeps a blocking
/// tool from starving the interrupt's cancellation poll.
#[allow(clippy::too_many_arguments)]
fn spawn_agent(
    app: &App,
    ui_tx: &UnboundedSender<Msg>,
    input_rx: &Arc<Mutex<UnboundedReceiver<String>>>,
    cwd: &std::path::Path,
    interrupt: &InterruptHandle,
    epoch: u64,
    perm_mode: &Arc<std::sync::atomic::AtomicU8>,
) -> bool {
    if app.api_key.is_none() && !app.has_fallback_provider {
        return false; // no provider yet — wait until a key is set
    }
    let api_key = app.api_key.clone();
    let model = app.model.clone();
    let mcp_servers = app.mcp_servers.clone();
    let multi_agent = app.multi_agent;
    let context_recall = app.context_recall;
    let runner_tx = ui_tx.clone();
    let done_tx = ui_tx.clone();
    let input_rx = input_rx.clone();
    let cwd = cwd.to_path_buf();
    let interrupt = interrupt.clone();
    let perm_mode = perm_mode.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("agent runtime");
        rt.block_on(async move {
            match build_engine(
                api_key,
                &model,
                runner_tx,
                input_rx.clone(),
                cwd,
                interrupt,
                mcp_servers,
                multi_agent,
                context_recall,
                perm_mode,
            )
            .await
            {
                Ok(mut engine) => {
                    // MCP is now connected (eagerly). Wait for the first user message,
                    // then run; `on_input` feeds the rest from the same channel.
                    let first = input_rx.lock().await.recv().await;
                    if let Some(first) = first {
                        let _ = engine.run(&first).await;
                    }
                }
                Err(e) => {
                    let _ = done_tx.send(Msg::Notice(format!("cannot start agent: {e}")));
                }
            }
            // Signal this thread's exit (with its epoch) so the UI can respawn —
            // ignoring a stale exit if the engine was already replaced.
            let _ = done_tx.send(Msg::AgentExited(epoch));
        });
    });
    true
}

#[allow(clippy::too_many_arguments)]
async fn run_ui(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    mut ui_rx: UnboundedReceiver<Msg>,
    mut input_tx: UnboundedSender<String>,
    ui_tx: UnboundedSender<Msg>,
    mut input_rx: Arc<Mutex<UnboundedReceiver<String>>>,
    cwd: PathBuf,
    interrupt: InterruptHandle,
    perm_mode: Arc<std::sync::atomic::AtomicU8>,
    session_id: String,
) -> anyhow::Result<()> {
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(120));
    // Auto-save bookkeeping: persist when the transcript grows while idle.
    let mut last_saved_len = 0usize;
    // Monotonic spawn epoch: each agent thread carries one, so a stale exit from a
    // thread we already replaced (e.g. on an `/agents` restart) can be ignored.
    let mut agent_epoch: u64 = 0;
    // Eager start: if a provider is already configured (env/config/fallback),
    // spawn the agent now so MCP servers connect at STARTUP, before any message.
    agent_epoch += 1;
    let mut agent_started = spawn_agent(
        app,
        &ui_tx,
        &input_rx,
        &cwd,
        &interrupt,
        agent_epoch,
        &perm_mode,
    );

    loop {
        terminal.draw(|f| ui::view(f, app))?;

        tokio::select! {
            maybe_event = events.next() => {
                if let Some(Ok(event)) = maybe_event
                    && let Some(m) = translate(event)
                {
                    app.update(m);
                }
            }
            maybe_msg = ui_rx.recv() => {
                if let Some(m) = maybe_msg {
                    // The agent thread exited (build failure or session end) → allow
                    // the next message to rebuild the runner. Ignore a stale exit
                    // whose epoch we already superseded (an `/agents` restart).
                    if let Msg::AgentExited(e) = m
                        && e == agent_epoch
                    {
                        agent_started = false;
                    }
                    app.update(m);
                    while let Ok(m2) = ui_rx.try_recv() {
                        if let Msg::AgentExited(e) = m2
                            && e == agent_epoch
                        {
                            agent_started = false;
                        }
                        app.update(m2);
                    }
                }
            }
            _ = tick.tick() => {
                if app.running {
                    app.update(Msg::Tick);
                }
            }
        }

        for effect in std::mem::take(&mut app.effects) {
            match effect {
                Effect::SendInput(text) => {
                    // Spawn-if-not-started (covers the no-key-at-startup path and a
                    // re-spawn after a session ended via RunCompleted). The first
                    // message reaches the agent over the same channel `on_input` uses
                    // — the thread consumes it before `execute`.
                    if !agent_started {
                        agent_epoch += 1;
                        agent_started = spawn_agent(
                            app,
                            &ui_tx,
                            &input_rx,
                            &cwd,
                            &interrupt,
                            agent_epoch,
                            &perm_mode,
                        );
                    }
                    let _ = input_tx.send(text);
                }
                Effect::SaveKey(key) => {
                    let mut cfg = config::TuiConfig::load();
                    cfg.openrouter_api_key = Some(key);
                    if let Err(e) = cfg.save() {
                        app.history
                            .push(Cell::Notice(format!("could not save config: {e}")));
                    }
                    // A key was just set → connect the agent (and MCP) eagerly now.
                    if !agent_started {
                        agent_epoch += 1;
                        agent_started = spawn_agent(
                            app,
                            &ui_tx,
                            &input_rx,
                            &cwd,
                            &interrupt,
                            agent_epoch,
                            &perm_mode,
                        );
                    }
                }
                Effect::SaveModel(model) => {
                    let mut cfg = config::TuiConfig::load();
                    cfg.model = Some(model);
                    if let Err(e) = cfg.save() {
                        app.history
                            .push(Cell::Notice(format!("could not save config: {e}")));
                    }
                }
                Effect::SaveMcp(servers) => {
                    let mut cfg = config::TuiConfig::load();
                    cfg.mcp_servers = servers;
                    if let Err(e) = cfg.save() {
                        app.history
                            .push(Cell::Notice(format!("could not save config: {e}")));
                    }
                }
                Effect::SetPermissionMode(m) => {
                    // Apply live to the shared gate the agent's on_approval reads.
                    perm_mode.store(m, std::sync::atomic::Ordering::Relaxed);
                }
                Effect::ExportSession => {
                    let md = session::to_markdown(&app.history);
                    let path = cwd.join(format!("heartbit-session-{session_id}.md"));
                    match std::fs::write(&path, md) {
                        Ok(()) => app
                            .history
                            .push(Cell::Notice(format!("exported to {}", path.display()))),
                        Err(e) => app
                            .history
                            .push(Cell::Notice(format!("export failed: {e}"))),
                    }
                }
                Effect::ListSessions => {
                    let metas = session::list(&session::sessions_dir());
                    let _ = ui_tx.send(Msg::SessionsListed(metas));
                }
                Effect::ResumeSession(id) => match session::load(&session::sessions_dir(), &id) {
                    Ok(s) => {
                        let _ = ui_tx.send(Msg::SessionLoaded(s.history));
                    }
                    Err(e) => app
                        .history
                        .push(Cell::Notice(format!("could not resume: {e}"))),
                },
                Effect::SaveContextRecall(on) => {
                    // Persist; applies on the next agent start (it changes the tool
                    // set + pruner, so we don't hot-swap a running engine).
                    let mut cfg = config::TuiConfig::load();
                    cfg.context_recall = on;
                    if let Err(e) = cfg.save() {
                        app.history
                            .push(Cell::Notice(format!("could not save config: {e}")));
                    }
                }
                Effect::SaveMultiAgent(on) => {
                    let mut cfg = config::TuiConfig::load();
                    cfg.multi_agent = on;
                    if let Err(e) = cfg.save() {
                        app.history
                            .push(Cell::Notice(format!("could not save config: {e}")));
                    }
                    // Activate the new mode NOW if the agent is idle: end the parked
                    // idle thread by replacing the input channel (its `on_input`
                    // recv() returns None → it exits), bump the epoch so its stale
                    // exit is ignored, and clear `agent_started` so the next message
                    // respawns in the new mode. (A running turn applies on next start.)
                    if agent_started && !app.running {
                        agent_epoch += 1; // supersede the idle thread
                        let (ntx, nrx) = tokio::sync::mpsc::unbounded_channel::<String>();
                        input_tx = ntx; // drop the old sender → idle on_input → None → exit
                        input_rx = Arc::new(Mutex::new(nrx));
                        agent_started = false;
                        app.history.push(Cell::Notice(
                            "restarting agent to apply the new mode — send a message".into(),
                        ));
                    } else if app.running {
                        app.history.push(Cell::Notice(
                            "a task is running — toggle again once it finishes to apply".into(),
                        ));
                    }
                }
                Effect::FetchModels => {
                    // Fetch the OpenRouter catalog off the UI thread; the result
                    // comes back as Msg::ModelsLoaded / ModelsFailed.
                    let tx = ui_tx.clone();
                    tokio::spawn(async move {
                        let msg = match models::fetch_openrouter_models().await {
                            Ok(m) => Msg::ModelsLoaded(m),
                            Err(e) => Msg::ModelsFailed(e.to_string()),
                        };
                        let _ = tx.send(msg);
                    });
                }
                Effect::WalkFiles => {
                    // Build the @-mention file index off the UI thread.
                    let tx = ui_tx.clone();
                    let root = cwd.clone();
                    tokio::spawn(async move {
                        let files = tokio::task::spawn_blocking(move || walk_project_files(&root))
                            .await
                            .unwrap_or_default();
                        let _ = tx.send(Msg::FilesLoaded(files));
                    });
                }
                Effect::Interrupt => {
                    tracing::info!(
                        target: "heartbit::interrupt",
                        checkpoint = "CP1_tui_effect_interrupt",
                        running = app.running,
                        "Effect::Interrupt dequeued in UI loop"
                    );
                    interrupt.interrupt();
                    tracing::info!(
                        target: "heartbit::interrupt",
                        checkpoint = "CP2_handle_interrupt_called",
                        is_cancelled = interrupt.is_interrupted(),
                        "interrupt.interrupt() returned"
                    );
                }
                Effect::Quit => app.should_quit = true,
            }
        }

        // Auto-save the transcript at turn boundaries (idle + changed) so the
        // session is always resumable; and once more on quit.
        if !app.running && app.history.len() != last_saved_len {
            save_session(&session_id, &app.history);
            last_saved_len = app.history.len();
        }

        if app.should_quit {
            save_session(&session_id, &app.history);
            break;
        }
    }
    Ok(())
}

/// Persist the current transcript under the session id (best-effort, silent).
fn save_session(id: &str, history: &[crate::cells::Cell]) {
    let s = session::Session {
        id: id.to_string(),
        created: id.to_string(),
        history: history.to_vec(),
    };
    let _ = session::save(&session::sessions_dir(), &s);
}
