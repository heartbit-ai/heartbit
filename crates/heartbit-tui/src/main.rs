//! `heartbit-tui` — a terminal UI to chat with the Heartbit framework, in the
//! spirit of Claude Code / codex / opencode.
//!
//! Architecture (see `tasks/tui-2026-06-02.md`):
//! - **Main thread**: the async UI loop (ratatui draw + crossterm `EventStream`),
//!   owning the pure [`App`] state. All terminal/channel I/O lives here.
//! - **Dedicated agent thread**: a `current_thread` tokio runtime that
//!   `block_on`s `AgentRunner::execute()`. The runner's synchronous callbacks
//!   bridge to the UI over channels; `on_input` (async) feeds turns back so the
//!   conversation history is preserved inside a single `execute()` call.
//! - `on_approval` is synchronous, so it round-trips over a blocking
//!   `std::sync::mpsc` (the agent thread idles, waiting on the human) — never
//!   touching the agent runtime. Zero core changes.

mod app;
mod cells;
mod composer;
mod msg;
mod ui;

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{Event, EventStream, KeyEventKind};
use futures::StreamExt;
use heartbit_core::tool::builtins::{BuiltinToolsConfig, builtin_tools};
use heartbit_core::{
    AgentEvent, AgentRunner, ApprovalDecision, BoxedProvider, OnApproval, OnEvent, OnInput, OnText,
    OpenRouterProvider, PermissionAction, PermissionRule, PermissionRuleset, RetryingProvider,
};

use crate::app::{App, Effect};
use crate::msg::{Msg, PendingTool};

const SYSTEM_PROMPT: &str = "You are Heartbit, an expert software engineering assistant running in a \
terminal UI, with tools to read, search, edit, and run code in the user's workspace. Be concise and \
direct. When you change code, make the smallest correct change, then verify it. Prefer showing your \
work through tool use over describing it. When a task is done, say so briefly.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Provider + tools are built up-front; the agent run starts on the first message.
    let (provider, model_name) = build_provider()?;
    let cwd = std::env::current_dir()?;
    let mut tool_cfg = BuiltinToolsConfig::default();
    tool_cfg.workspace = Some(cwd.clone());
    tool_cfg.dangerous_tools = true;
    let tools = builtin_tools(tool_cfg);

    // agent -> UI (sync sends from callbacks); UI -> agent (async on_input feed).
    let (ui_tx, ui_rx) = tokio::sync::mpsc::unbounded_channel::<Msg>();
    let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let input_rx = Arc::new(tokio::sync::Mutex::new(input_rx));

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
            // Synchronous round-trip: block THIS (agent) thread on a std channel
            // while the human answers in the UI thread. No tokio involvement.
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
        let rx = input_rx.clone();
        Arc::new(move || {
            let rx = rx.clone();
            Box::pin(async move {
                // Returns None when all input senders are dropped (UI quit) -> the
                // agent's internal REPL loop ends and execute() returns.
                rx.lock().await.recv().await
            }) as Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
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
        .build()?;

    // Terminal: ratatui::init enables raw mode + alt screen + a panic hook that
    // restores the terminal. We do NOT enable mouse capture, so native
    // selection/copy keep working.
    let mut terminal = ratatui::init();
    let mut app = App::new(model_name);
    let result = run_ui(
        &mut terminal,
        &mut app,
        ui_rx,
        input_tx,
        Some(runner),
        ui_tx,
    )
    .await;
    ratatui::restore();
    result
}

/// Resolve a provider from the environment (OpenRouter preferred, matching the
/// project's qwen setup). Returns the provider and a display model name.
fn build_provider() -> anyhow::Result<(Arc<BoxedProvider>, String)> {
    if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
        let model =
            std::env::var("HEARTBIT_MODEL").unwrap_or_else(|_| "qwen/qwen3-235b-a22b-2507".into());
        let base = OpenRouterProvider::new(key, &model);
        let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(base)));
        return Ok((provider, model));
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let model = std::env::var("HEARTBIT_MODEL").unwrap_or_else(|_| "claude-sonnet-4-6".into());
        let base = heartbit_core::AnthropicProvider::new(&key, &model);
        let provider = Arc::new(BoxedProvider::new(RetryingProvider::with_defaults(base)));
        return Ok((provider, model));
    }
    anyhow::bail!(
        "no API key found — set OPENROUTER_API_KEY (recommended) or ANTHROPIC_API_KEY, and \
         optionally HEARTBIT_MODEL"
    )
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
        // Catch-all: anything else (write/edit/patch/bash/skill/…) prompts.
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

async fn run_ui(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    mut ui_rx: tokio::sync::mpsc::UnboundedReceiver<Msg>,
    input_tx: tokio::sync::mpsc::UnboundedSender<String>,
    mut runner: Option<AgentRunner<BoxedProvider>>,
    ui_tx_done: tokio::sync::mpsc::UnboundedSender<Msg>,
) -> anyhow::Result<()> {
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(120));

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
                    app.update(m);
                    // Coalesce a burst of streaming deltas into one redraw.
                    while let Ok(m2) = ui_rx.try_recv() {
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
                    if let Some(r) = runner.take() {
                        // First message: start the agent on its own OS thread.
                        let done_tx = ui_tx_done.clone();
                        std::thread::spawn(move || {
                            let rt = tokio::runtime::Builder::new_current_thread()
                                .enable_all()
                                .build()
                                .expect("agent runtime");
                            rt.block_on(async move {
                                let _ = r.execute(&text).await;
                                let _ = done_tx.send(Msg::RunCompleted);
                            });
                        });
                    } else {
                        // Subsequent messages feed the on_input REPL loop.
                        let _ = input_tx.send(text);
                    }
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
