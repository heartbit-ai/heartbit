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
mod markdown;
mod msg;
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
    OnInput, OnText, OpenRouterProvider, PermissionAction, PermissionRule, PermissionRuleset,
    RetryingProvider,
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
    // No provider configured at all → open the key prompt immediately.
    if app.api_key.is_none() && !has_anthropic {
        app.modal = Some(app::Modal::KeyEntry(app::KeyEntryModal::default()));
    }

    let interrupt = InterruptHandle::new();

    let mut terminal = ratatui::init();
    let result = run_ui(
        &mut terminal,
        &mut app,
        ui_rx,
        input_tx,
        ui_tx,
        input_rx,
        cwd,
        interrupt,
    )
    .await;
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

/// Build the agent runner, wiring the synchronous callbacks to the UI channels.
/// The OpenRouter token is passed only to the provider, never into the tool
/// environment (bash gets a no-secrets env allowlist).
async fn build_runner(
    api_key: Option<String>,
    model: &str,
    ui_tx: UnboundedSender<Msg>,
    input_rx: Arc<Mutex<UnboundedReceiver<String>>>,
    cwd: PathBuf,
    interrupt: InterruptHandle,
    mcp_servers: Vec<config::McpServerSpec>,
) -> anyhow::Result<AgentRunner<BoxedProvider>> {
    let provider = build_provider(api_key, model)?;

    let mut tool_cfg = BuiltinToolsConfig::default();
    tool_cfg.workspace = Some(cwd.clone());
    tool_cfg.dangerous_tools = true;
    // Do NOT inherit the host env into the agent's bash: the safe allowlist
    // (PATH/HOME/… , no secrets) keeps the API key out of the tool environment.
    tool_cfg.env_policy = heartbit_core::workspace::EnvPolicy::default();
    // Builtins FIRST so a connected MCP server cannot shadow a trusted builtin.
    // MCP connection happens here (on the agent thread's runtime) because the
    // stdio transport binds to its spawning runtime.
    let mut tools = builtin_tools(tool_cfg);
    for spec in &mcp_servers {
        let label = spec.label();
        let _ = ui_tx.send(Msg::Notice(format!("connecting MCP {label}…")));
        match connect_mcp(spec).await {
            Ok(mcp_tools) => {
                let n = mcp_tools.len();
                tools.extend(mcp_tools);
                let _ = ui_tx.send(Msg::Notice(format!("MCP {label}: connected ({n} tools)")));
            }
            Err(e) => {
                let _ = ui_tx.send(Msg::Notice(format!("MCP {label}: failed — {e}")));
            }
        }
    }

    let on_text: Arc<OnText> = {
        let tx = ui_tx.clone();
        Arc::new(move |s: &str| {
            let _ = tx.send(Msg::StreamDelta(s.to_string()));
        })
    };
    let on_event: Arc<OnEvent> = {
        let tx = ui_tx.clone();
        Arc::new(move |e: AgentEvent| {
            if let Some(m) = Msg::from_event(e) {
                let _ = tx.send(m);
            }
        })
    };
    let on_approval: Arc<OnApproval> = {
        let tx = ui_tx.clone();
        Arc::new(move |calls: &[heartbit_core::llm::types::ToolCall]| {
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

    let runner = AgentRunner::builder(provider)
        .name("heartbit")
        .system_prompt(SYSTEM_PROMPT)
        .tools(tools)
        .max_turns(300)
        .workspace(cwd)
        .permission_rules(default_permissions())
        .on_text(on_text)
        .on_event(on_event)
        .on_approval(on_approval)
        .on_input(on_input)
        .interrupt(interrupt)
        .build()?;
    Ok(runner)
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

/// Translate a raw crossterm event into a [`Msg`] (or ignore it).
fn translate(event: Event) -> Option<Msg> {
    match event {
        Event::Key(k) if k.kind == KeyEventKind::Press => Some(Msg::Key(k)),
        Event::Paste(s) => Some(Msg::Paste(s)),
        Event::Resize(..) => Some(Msg::Resize),
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
fn spawn_agent(
    app: &App,
    ui_tx: &UnboundedSender<Msg>,
    input_rx: &Arc<Mutex<UnboundedReceiver<String>>>,
    cwd: &std::path::Path,
    interrupt: &InterruptHandle,
) -> bool {
    if app.api_key.is_none() && !app.has_fallback_provider {
        return false; // no provider yet — wait until a key is set
    }
    let api_key = app.api_key.clone();
    let model = app.model.clone();
    let mcp_servers = app.mcp_servers.clone();
    let runner_tx = ui_tx.clone();
    let done_tx = ui_tx.clone();
    let input_rx = input_rx.clone();
    let cwd = cwd.to_path_buf();
    let interrupt = interrupt.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("agent runtime");
        rt.block_on(async move {
            match build_runner(
                api_key,
                &model,
                runner_tx,
                input_rx.clone(),
                cwd,
                interrupt,
                mcp_servers,
            )
            .await
            {
                Ok(runner) => {
                    // MCP is now connected (eagerly). Wait for the first user message,
                    // then run; `on_input` feeds the rest from the same channel.
                    let first = input_rx.lock().await.recv().await;
                    if let Some(first) = first {
                        let _ = runner.execute(&first).await;
                    }
                }
                Err(e) => {
                    let _ = done_tx.send(Msg::Notice(format!("cannot start agent: {e}")));
                }
            }
            // Reset `agent_started` in the UI loop (covers build failure / session end).
            let _ = done_tx.send(Msg::RunCompleted);
        });
    });
    true
}

#[allow(clippy::too_many_arguments)]
async fn run_ui(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    mut ui_rx: UnboundedReceiver<Msg>,
    input_tx: UnboundedSender<String>,
    ui_tx: UnboundedSender<Msg>,
    input_rx: Arc<Mutex<UnboundedReceiver<String>>>,
    cwd: PathBuf,
    interrupt: InterruptHandle,
) -> anyhow::Result<()> {
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(120));
    // Eager start: if a provider is already configured (env/config/fallback),
    // spawn the agent now so MCP servers connect at STARTUP, before any message.
    let mut agent_started = spawn_agent(app, &ui_tx, &input_rx, &cwd, &interrupt);

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
                    // the next message to rebuild the runner (and reconnect MCP).
                    if matches!(m, Msg::RunCompleted) {
                        agent_started = false;
                    }
                    app.update(m);
                    while let Ok(m2) = ui_rx.try_recv() {
                        if matches!(m2, Msg::RunCompleted) {
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
                        agent_started = spawn_agent(app, &ui_tx, &input_rx, &cwd, &interrupt);
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
                        agent_started = spawn_agent(app, &ui_tx, &input_rx, &cwd, &interrupt);
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

        if app.should_quit {
            break;
        }
    }
    Ok(())
}
