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
mod lessons;
mod markdown;
mod models;
mod msg;
mod session;
mod trace;
mod trace_stats;
mod ui;

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{Event, EventStream, KeyEventKind};
use futures::StreamExt;
use heartbit_core::tool::builtins::{BuiltinToolsConfig, builtin_tools};
use heartbit_core::{
    AgentEvent, ApprovalDecision, BoxedProvider, InterruptHandle, OnApproval, OnEvent, OnInput,
    OnText, OpenRouterProvider, Orchestrator, PermissionAction, PermissionRule, PermissionRuleset,
    RetryingProvider, SubAgentConfig,
};
use tokio::sync::Mutex;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::app::{App, Effect};
use crate::cells::Cell;
use crate::msg::{Msg, PendingTool};

const DEFAULT_MODEL: &str = "qwen/qwen3-235b-a22b-2507";

/// Initialize tracing: the always-on trace bridge (`heartbit::interrupt` →
/// `core_trace` records in the session trace) plus the legacy opt-in debug
/// file (`HEARTBIT_TUI_DEBUG=1` → `/tmp/heartbit-tui-debug.log` — unchanged,
/// additive). Both layers carry a `Targets` filter scoped to the interrupt
/// target, so global span/event interest stays as narrow as before.
fn init_tracing(trace_handle: trace::TraceHandle) {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let target_filter = || {
        tracing_subscriber::filter::Targets::new().with_target(
            trace::INTERRUPT_TARGET,
            tracing::level_filters::LevelFilter::INFO,
        )
    };
    let bridge = trace::TracingBridge::new(trace_handle).with_filter(target_filter());
    let legacy = legacy_debug_file().map(|file| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(file))
            .with_filter(target_filter())
    });
    let _ = tracing_subscriber::registry()
        .with(bridge)
        .with(legacy)
        .try_init();
    tracing::info!(target: "heartbit::interrupt", "--- heartbit-tui debug logging started ---");
}

/// The legacy `HEARTBIT_TUI_DEBUG` file, if requested. The TUI owns the
/// terminal (alt-screen) so stderr is unusable; `HEARTBIT_TUI_DEBUG=1` routes
/// `heartbit::interrupt` checkpoints to `/tmp/heartbit-tui-debug.log` (or a
/// custom path). Same semantics as before the trace existed.
fn legacy_debug_file() -> Option<std::fs::File> {
    let val = std::env::var("HEARTBIT_TUI_DEBUG").ok()?;
    if val.is_empty() {
        return None;
    }
    let path = if val == "1" || val == "true" {
        "/tmp/heartbit-tui-debug.log".to_string()
    } else {
        val
    };
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // A per-launch session id (time + pid) for transcript persistence — it
    // also keys the execution trace.
    let session_id = format!(
        "{:x}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        std::process::id()
    );
    // agent -> UI (sync sends from callbacks); UI -> agent (async on_input feed).
    let (ui_tx, ui_rx) = tokio::sync::mpsc::unbounded_channel::<Msg>();
    // Always-on execution trace (one JSONL per launch, next to the session
    // JSON). A write error self-disables tracing and surfaces ONE notice —
    // the trace must never take down or block the session.
    let trace_handle = trace::spawn_writer(
        trace::trace_path(&session::sessions_dir(), &session_id),
        Box::new({
            let tx = ui_tx.clone();
            move |e| {
                let _ = tx.send(Msg::Notice(e));
            }
        }),
    );
    // NOTE: init_tracing is deliberately called AFTER the session_started
    // record below — its banner line is captured by the bridge and would
    // otherwise claim seq 0 (the trace must START with session_started).
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

    let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let input_rx = Arc::new(Mutex::new(input_rx));
    let cwd = std::env::current_dir()?;

    let mut app = App::new(model);
    app.api_key = api_key;
    app.has_fallback_provider = has_anthropic;
    app.mcp_servers = cfg.mcp_servers.clone();
    app.multi_agent = cfg.multi_agent;
    app.context_recall = cfg.context_recall;
    app.verify_command = cfg.verify_command.clone();
    // The unified entry agent can ALWAYS delegate (the squad is always available),
    // so seed the roster's available squad unconditionally — it shows when the
    // agent actually dispatches sub-agents.
    app.squad = app::DEFAULT_SQUAD.iter().map(|s| s.to_string()).collect();
    // Fetch the OpenRouter catalog at startup (public endpoint) so the status-line
    // context bar knows the model's window and the /model picker is pre-warmed.
    app.models_loading = true;
    app.effects.push(Effect::FetchModels);
    // No provider configured at all → open the key prompt immediately.
    if app.api_key.is_none() && !has_anthropic {
        app.modal = Some(app::Modal::KeyEntry(app::KeyEntryModal::default()));
    }

    let interrupt = InterruptHandle::new();
    // Shared permission posture (0=normal, 1=plan, 2=yolo), cycled by the UI
    // (Shift+Tab) and read live by the agent thread's `on_approval`.
    let perm_mode = Arc::new(std::sync::atomic::AtomicU8::new(0));

    // Config snapshot at launch — the trace's first record (init_tracing runs
    // after this so its captured banner can't claim seq 0).
    trace_handle.record_ui(&trace::UiEvent::SessionStarted {
        version: env!("CARGO_PKG_VERSION").into(),
        session_id: session_id.clone(),
        model: app.model.clone(),
        permission_mode: app.permission_mode.label().to_lowercase(),
        mcp_servers: cfg.mcp_servers.iter().map(|s| s.label()).collect(),
        context_recall: app.context_recall,
        verify_command: app.verify_command.clone(),
    });
    init_tracing(trace_handle.clone());

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
        trace_handle,
    )
    .await;
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
    ratatui::restore();
    result
}

/// Resolve a provider. OpenRouter is preferred (the project's qwen setup); an
/// `ANTHROPIC_API_KEY` env var is a fallback when no OpenRouter key is configured.
/// `on_retry` surfaces retry attempts as `AgentEvent::RetryAttempt` (mirrors
/// heartbit-cli's `build_on_retry`) — retries are diagnostic gold and were
/// previously invisible in the TUI.
fn build_provider(
    openrouter_key: Option<String>,
    model: &str,
    on_retry: Arc<heartbit_core::OnRetry>,
) -> anyhow::Result<Arc<BoxedProvider>> {
    if let Some(key) = openrouter_key {
        let base = OpenRouterProvider::new(key, model);
        return Ok(Arc::new(BoxedProvider::new(
            RetryingProvider::with_defaults(base).with_on_retry(on_retry),
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
            RetryingProvider::with_defaults(base).with_on_retry(on_retry),
        )));
    }
    anyhow::bail!("no OpenRouter API key configured (set one with /key or OPENROUTER_API_KEY)")
}

/// The unified entry agent the TUI drives (option C): the [`Orchestrator`]
/// evolved into ONE capable agent that decides per request — answer directly,
/// do simple work, delegate to the squad, or run a workflow. A thin newtype so
/// the run loop has a stable handle; one call drives a whole `on_input`
/// multi-turn session.
struct Engine(Box<Orchestrator<BoxedProvider>>);

impl Engine {
    /// Run the (multi-turn) session, starting with `first`.
    async fn run(&mut self, first: &str) -> anyhow::Result<()> {
        self.0.run(first).await?;
        Ok(())
    }
}

/// A fresh set of workspace-rooted builtin tools (each call gets its own
/// `FileTracker` etc.). The API key never enters the tool env (safe allowlist).
fn fresh_builtins(
    cwd: &std::path::Path,
    context_recall_store: Option<&Arc<heartbit_core::ContextRecallStore>>,
    todo_store: Option<&Arc<heartbit_core::tool::builtins::TodoStore>>,
) -> Vec<Arc<dyn heartbit_core::tool::Tool>> {
    let mut tool_cfg = BuiltinToolsConfig::default();
    tool_cfg.workspace = Some(cwd.to_path_buf());
    tool_cfg.dangerous_tools = true;
    tool_cfg.env_policy = heartbit_core::workspace::EnvPolicy::default();
    // When present, registers fetch_full_output / recall_context for restore-on-demand.
    tool_cfg.context_recall_store = context_recall_store.cloned();
    // Share the SAME todo store the runner recites from (long-horizon planning),
    // so todowrite/todoread and the per-turn recitation see one list.
    if let Some(store) = todo_store {
        tool_cfg.todo_store = store.clone();
    }
    builtin_tools(tool_cfg)
}

/// Test mode: `HEARTBIT_CONTEXT_DEBUG=1` lowers the pruning + compaction
/// thresholds so both fire within a few turns (instead of needing a full long
/// session), making the context-management behaviour observable for live testing.
fn context_debug_mode() -> bool {
    std::env::var("HEARTBIT_CONTEXT_DEBUG")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// A gentle session-prune config paired with context-recall: keep the last few
/// turns at full fidelity, truncate older tool results to ~1KB (each carrying a
/// `fetch_full_output(<id>)` marker so the full content is recoverable). In
/// `HEARTBIT_CONTEXT_DEBUG` mode it's aggressive (keep 1 turn, 200B) so pruning
/// is visible immediately.
fn gentle_prune_config() -> heartbit_core::SessionPruneConfig {
    if context_debug_mode() {
        heartbit_core::SessionPruneConfig {
            keep_recent_n: 1,
            pruned_tool_result_max_bytes: 200,
            preserve_task: true,
        }
    } else {
        heartbit_core::SessionPruneConfig {
            keep_recent_n: 3,
            pruned_tool_result_max_bytes: 1024,
            preserve_task: true,
        }
    }
}

/// Two capable generalist sub-agents (≥2 → dynamic squads auto-enable). Each gets
/// fresh builtins + a clone of the connected MCP tools, so delegated work can
/// actually read/edit/run, not just talk. Distinct descriptions guide routing.
fn default_sub_agents(
    cwd: &std::path::Path,
    mcp_tools: &[Arc<dyn heartbit_core::tool::Tool>],
    context_recall: bool,
    context_window: Option<u32>,
    replan_on_verify_fail: bool,
) -> Vec<SubAgentConfig> {
    let make = |name: &str, description: &str, prompt: &str| {
        // Multi-agent context enablement: each sub-agent gets its OWN context
        // stores (per-agent isolation) — it recites/restores only its own plan
        // and tool outputs. The recall store is paired with the gentle pruner so
        // restore markers are produced; recitation is always on (self-gates to
        // nothing until the agent writes todos).
        let recall = context_recall.then(|| Arc::new(heartbit_core::ContextRecallStore::new()));
        let todo = Arc::new(heartbit_core::tool::builtins::TodoStore::new());
        let mut tools = fresh_builtins(cwd, recall.as_ref(), Some(&todo));
        tools.extend(mcp_tools.iter().cloned());
        SubAgentConfig {
            name: name.into(),
            description: description.into(),
            system_prompt: prompt.into(),
            tools,
            max_turns: Some(60),
            max_tokens: Some(8192),
            session_prune_config: recall.as_ref().map(|_| gentle_prune_config()),
            context: heartbit_core::SubAgentContextConfig {
                todo_store: Some(todo),
                context_recall_store: recall,
                context_window_tokens: context_window,
                replan_on_verify_fail,
            },
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
    context_recall: bool,
    context_window: Option<u32>,
    verify_command: Option<String>,
    perm_mode: Arc<std::sync::atomic::AtomicU8>,
    trace: trace::TraceHandle,
) -> anyhow::Result<Engine> {
    // on_event is defined BEFORE the provider so retry attempts can flow
    // through the same path (event → trace tap + UI message).
    let on_event: Arc<OnEvent> = {
        let tx = ui_tx.clone();
        let trace_events = trace.clone();
        Arc::new(move |e: AgentEvent| {
            // Lossless trace tap — BEFORE Msg::from_event (which drops a subset).
            trace_events.record_agent(&e);
            // Legacy opt-in diagnostics (HEARTBIT_TUI_DEBUG) — unchanged. These
            // also land as `core_trace` records via the bridge; the canonical
            // typed stream is `agent`/`ui`, so the redundancy is accepted.
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
    // Wire RetryAttempt emission (mirrors heartbit-cli's build_on_retry).
    let on_retry: Arc<heartbit_core::OnRetry> = {
        let cb = on_event.clone();
        Arc::new(
            move |attempt: u32, max_retries: u32, delay_ms: u64, error_class: &str| {
                cb(AgentEvent::RetryAttempt {
                    agent: "(provider)".into(), // provider-level: agent name unavailable
                    attempt,
                    max_retries,
                    delay_ms,
                    error_class: error_class.to_string(),
                });
            },
        )
    };
    let provider = build_provider(api_key, model, on_retry)?;

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
    // Unified entry agent (option C): ONE capable agent that decides per request
    // — answer directly, do simple work, delegate, or run a workflow. It ALWAYS
    // gets its own context stack (recitation + restore-on-demand) and the
    // run_workflow tool; the squad stays available for delegation.
    let recall_store = context_recall.then(|| Arc::new(heartbit_core::ContextRecallStore::new()));
    let todo_store = Arc::new(heartbit_core::tool::builtins::TodoStore::new());
    // The entry agent's direct tools: builtins FIRST so MCP can't shadow a trusted one.
    let mut tools = fresh_builtins(&cwd, recall_store.as_ref(), Some(&todo_store));
    tools.extend(mcp_tools.iter().cloned());
    // Self-verification (opt-in via /verify): a deterministic `verify` tool that
    // runs the project's build/test command (VERIFY_RESULT: PASS/FAIL).
    if let Some(cmd) = verify_command.as_deref().filter(|c| !c.is_empty()) {
        tools.push(Arc::new(heartbit_core::VerifyCommandTool::new(
            cwd.clone(),
            vec![cmd.to_string()],
        )));
    }
    // run_workflow: named recipes (parallel_review, …) reachable by the agent.
    let registry = heartbit_core::default_registry();
    let recipe_meta = registry.meta();
    tools.push(Arc::new(heartbit_core::RunWorkflowTool::new(
        registry,
        provider.clone(),
    )));

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
    let on_approval: Arc<OnApproval> = {
        let tx = ui_tx.clone();
        let perm_mode = perm_mode.clone();
        let trace_approvals = trace.clone();
        Arc::new(move |calls: &[heartbit_core::llm::types::ToolCall]| {
            let started = std::time::Instant::now();
            let names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
            for c in calls {
                tracing::info!(target: "heartbit::interrupt", approval_for = %c.name, "on_approval");
            }
            // Load the mode ONCE: the gate and every trace record below must
            // agree (a Plan-mode read-only batch falls through to the modal —
            // its record must still say "plan", not "normal").
            let mode = perm_mode.load(std::sync::atomic::Ordering::Relaxed);
            let mode_str = trace::mode_label(mode);
            // Every gate resolution is traced with the human think-time and the
            // mode in effect (rung-2 learning gold).
            let record = |decision: &ApprovalDecision, latency_ms: u64| {
                trace_approvals.record_ui(&trace::UiEvent::Approval {
                    tools: names.clone(),
                    decision: trace::decision_label(decision).into(),
                    latency_ms,
                    mode: mode_str.into(),
                });
            };
            // Execution mode gates the prompt (u8 from PermissionMode::as_u8):
            // 2=YOLO → allow all; 1=Plan → deny any mutating tool (read-only
            // batches still ask via the modal); 0=Normal → ask (modal).
            let is_mutating = |n: &str| matches!(n, "edit" | "write" | "patch" | "bash");
            match mode {
                2 => {
                    record(&ApprovalDecision::Allow, 0);
                    return ApprovalDecision::Allow;
                }
                1 if calls.iter().any(|c| is_mutating(&c.name)) => {
                    record(&ApprovalDecision::Deny, 0);
                    return ApprovalDecision::Deny;
                }
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
                record(&ApprovalDecision::Deny, 0);
                return ApprovalDecision::Deny;
            }
            let decision = reply_rx.recv().unwrap_or(ApprovalDecision::Deny);
            record(
                &decision,
                u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            );
            decision
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

    // Unified entry agent (option C): ALWAYS built, no static mode flag. The
    // orchestrator evolved into ONE capable agent — it holds direct tools +
    // delegation tools (delegate_task / form_squad) + run_workflow, and decides
    // per request via tool choice (answer directly / do simple work / delegate /
    // run a workflow). One run() drives the whole multi-turn session.
    let _ = ui_tx.send(Msg::Notice(
        "unified agent — answers directly, delegates, or runs a workflow as the task warrants"
            .into(),
    ));
    let replan = verify_command.as_deref().is_some_and(|c| !c.is_empty());
    if recall_store.is_some() {
        let _ = ui_tx.send(Msg::Notice(
            "context restore-on-demand ON — old tool outputs recoverable via fetch_full_output / recall_context".into(),
        ));
    }
    if context_window.is_some() && context_debug_mode() {
        let _ = ui_tx.send(Msg::Notice(
            "HEARTBIT_CONTEXT_DEBUG: proactive compaction at the model window".into(),
        ));
    }
    // Verify nudge: when /verify is active, instruct the agent (via the appended
    // instruction text) to run `verify` after code changes and treat
    // VERIFY_RESULT as truth — paired with the bounded replan gate below.
    let instructions = match verify_command.as_deref().filter(|c| !c.is_empty()) {
        Some(cmd) => format!(
            "{project_context}\n\nWhen you change code, run the `verify` tool (it runs `{cmd}`) and \
             FIX any failures before saying you're done. VERIFY_RESULT: PASS/FAIL is the source of \
             truth — never claim success without a PASS."
        ),
        None => project_context,
    };
    let mut builder = Orchestrator::builder(provider)
        .entry_agent(tools)
        .entry_workflow_recipes(recipe_meta)
        .entry_context(heartbit_core::SubAgentContextConfig {
            todo_store: Some(todo_store),
            context_recall_store: recall_store,
            context_window_tokens: context_window,
            replan_on_verify_fail: replan,
        })
        .max_turns(300)
        .workspace(cwd.clone())
        .instruction_text(instructions)
        .permission_rules(default_permissions())
        .on_text(on_text)
        .on_reasoning(on_reasoning)
        .on_event(on_event)
        .on_approval(on_approval)
        .on_input(on_input)
        .interrupt(interrupt);
    // The squad available for delegation: each sub-agent gets its own context
    // stack (recitation / restore-on-demand / compaction / replan).
    for cfg in default_sub_agents(&cwd, &mcp_tools, context_recall, context_window, replan) {
        builder = builder.sub_agent_full(cfg);
    }
    let orch = builder.build()?;
    Ok(Engine(Box::new(orch)))
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
    trace: &trace::TraceHandle,
    reason: &'static str,
) -> bool {
    if app.api_key.is_none() && !app.has_fallback_provider {
        return false; // no provider yet — wait until a key is set
    }
    // Snapshot the config that actually shapes the engine — the eager-spawn
    // stale-config bug class is diagnosed from these records.
    trace.record_ui(&trace::UiEvent::AgentSpawned {
        epoch,
        model: app.model.clone(),
        reason: reason.into(),
        context_recall: app.context_recall,
        verify_command: app.verify_command.clone(),
    });
    let api_key = app.api_key.clone();
    let model = app.model.clone();
    let mcp_servers = app.mcp_servers.clone();
    let context_recall = app.context_recall;
    let context_window = app.context_limit().map(|w| w.min(u32::MAX as u64) as u32);
    let verify_command = app.verify_command.clone();
    let runner_tx = ui_tx.clone();
    let done_tx = ui_tx.clone();
    let input_rx = input_rx.clone();
    let cwd = cwd.to_path_buf();
    let interrupt = interrupt.clone();
    let perm_mode = perm_mode.clone();
    let trace = trace.clone();
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
                context_recall,
                context_window,
                verify_command,
                perm_mode,
                trace,
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
    input_tx: UnboundedSender<String>,
    ui_tx: UnboundedSender<Msg>,
    input_rx: Arc<Mutex<UnboundedReceiver<String>>>,
    cwd: PathBuf,
    interrupt: InterruptHandle,
    perm_mode: Arc<std::sync::atomic::AtomicU8>,
    session_id: String,
    trace: trace::TraceHandle,
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
        &trace,
        "startup",
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
            let effect_name = effect.name();
            let effect_started = std::time::Instant::now();
            match effect {
                Effect::SendInput(text) => {
                    trace.record_ui(&trace::UiEvent::UserInput { text: text.clone() });
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
                            &trace,
                            "respawn",
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
                            &trace,
                            "key_set",
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
                    let old = perm_mode.swap(m, std::sync::atomic::Ordering::Relaxed);
                    if old != m {
                        trace.record_ui(&trace::UiEvent::ModeChanged {
                            from: trace::mode_label(old).into(),
                            to: trace::mode_label(m).into(),
                        });
                    }
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
                        trace.record_ui(&trace::UiEvent::SessionResumed {
                            from_id: id.clone(),
                        });
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
                Effect::SaveVerifyCommand(cmd) => {
                    let mut cfg = config::TuiConfig::load();
                    cfg.verify_command = cmd;
                    if let Err(e) = cfg.save() {
                        app.history
                            .push(Cell::Notice(format!("could not save config: {e}")));
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
                    // Typed ui records mirror CP1/CP2 (stats counts cp1); the
                    // tracing lines stay for the legacy debug file + core_trace.
                    trace.record_ui(&trace::UiEvent::InterruptRequested {
                        checkpoint: "cp1_effect_dequeued".into(),
                        running: app.running,
                    });
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
                    trace.record_ui(&trace::UiEvent::InterruptRequested {
                        checkpoint: "cp2_handle_interrupted".into(),
                        running: app.running,
                    });
                }
                Effect::ComputeStats(target) => {
                    let tx = ui_tx.clone();
                    let sid = session_id.clone();
                    tokio::spawn(async move {
                        let result = tokio::task::spawn_blocking(move || {
                            let dir = session::sessions_dir();
                            let path = trace::resolve_trace_target(&dir, &sid, target.as_deref())?;
                            let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
                            Ok::<String, String>(trace_stats::compute(file).render())
                        })
                        .await
                        .unwrap_or_else(|e| Err(e.to_string()));
                        let _ = tx.send(Msg::StatsReady(result));
                    });
                }
                Effect::Analyze(target) => {
                    let tx = ui_tx.clone();
                    let sid = session_id.clone();
                    let workdir = cwd.clone();
                    tokio::spawn(async move {
                        let prepared = tokio::task::spawn_blocking(move || {
                            let dir = session::sessions_dir();
                            let path = trace::resolve_trace_target(&dir, &sid, target.as_deref())?;
                            let id = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .and_then(|n| n.strip_suffix(".trace.jsonl"))
                                .unwrap_or("session")
                                .to_string();
                            // Stage a snapshot into the workspace FIRST, then
                            // compute stats FROM THE COPY: the agent's builtins
                            // (read/grep/write) REJECT absolute paths when
                            // workspace-rooted, and snapshotting before reading
                            // freezes a still-growing trace — stats and the file
                            // the agent greps are the same instant (one open).
                            let staged = format!("heartbit-trace-{id}.jsonl");
                            let staged_path = workdir.join(&staged);
                            std::fs::copy(&path, &staged_path).map_err(|e| e.to_string())?;
                            let file =
                                std::fs::File::open(&staged_path).map_err(|e| e.to_string())?;
                            let stats = trace_stats::compute(file);
                            let stats_json =
                                serde_json::to_string_pretty(&stats).map_err(|e| e.to_string())?;
                            Ok::<(String, String), String>((
                                format!("analyzing session {id} (staged: {staged})"),
                                trace::build_analyze_prompt(&staged, &id, &stats_json),
                            ))
                        })
                        .await
                        .unwrap_or_else(|e| Err(e.to_string()));
                        let _ = tx.send(match prepared {
                            Ok((display, task)) => Msg::AnalyzeReady { display, task },
                            Err(e) => Msg::AnalyzeFailed(e),
                        });
                    });
                }
                Effect::Learn => {
                    let tx = ui_tx.clone();
                    let workdir = cwd.clone();
                    tokio::spawn(async move {
                        let prepared = tokio::task::spawn_blocking(move || {
                            // Stage the global lessons (or the template) into cwd —
                            // the agent's builtins reject absolute paths.
                            let staged = workdir.join(lessons::STAGED_LESSONS);
                            let current = lessons::load_lessons()
                                .unwrap_or_else(|| lessons::LESSONS_TEMPLATE.to_string());
                            std::fs::write(&staged, &current).map_err(|e| e.to_string())?;
                            let digest = lessons::file_digest(&staged)
                                .ok_or_else(|| "staged lessons unreadable".to_string())?;
                            // Newest ≤ 3 diagnosis reports (by mtime).
                            let mut diags: Vec<(std::time::SystemTime, String)> = Vec::new();
                            let entries = std::fs::read_dir(&workdir).map_err(|e| e.to_string())?;
                            for e in entries.flatten() {
                                let name = e.file_name().to_string_lossy().into_owned();
                                if name.starts_with("heartbit-diagnosis-") && name.ends_with(".md")
                                {
                                    let mtime = e
                                        .metadata()
                                        .and_then(|m| m.modified())
                                        .unwrap_or(std::time::UNIX_EPOCH);
                                    diags.push((mtime, name));
                                }
                            }
                            if diags.is_empty() {
                                return Err("no diagnosis found — run /analyze first".to_string());
                            }
                            diags.sort_by_key(|(t, _)| std::cmp::Reverse(*t));
                            let names: Vec<String> =
                                diags.into_iter().take(3).map(|(_, n)| n).collect();
                            Ok::<(String, String, u64), String>((
                                format!("learning from {} diagnosis report(s)", names.len()),
                                lessons::build_learn_prompt(lessons::STAGED_LESSONS, &names),
                                digest,
                            ))
                        })
                        .await
                        .unwrap_or_else(|e| Err(e.to_string()));
                        let _ = tx.send(match prepared {
                            Ok((display, task, staged_digest)) => Msg::LearnReady {
                                display,
                                task,
                                staged_digest,
                            },
                            Err(e) => Msg::LearnFailed(e),
                        });
                    });
                }
                Effect::CommitLessons(staged_digest) => {
                    // Cheap + sync: re-hash, skip if the agent never rewrote it,
                    // validate, then atomically promote to the global file.
                    let staged = cwd.join(lessons::STAGED_LESSONS);
                    match lessons::file_digest(&staged) {
                        Some(d) if d == staged_digest => {
                            app.history
                                .push(Cell::Notice("lessons unchanged — nothing to commit".into()));
                        }
                        Some(_) => match lessons::validate_staged(&staged) {
                            Ok(n) => match lessons::commit_lessons(&staged) {
                                Ok(()) => app.history.push(Cell::Notice(format!(
                                    "lessons updated ({n} lessons) — apply on next start"
                                ))),
                                Err(e) => app
                                    .history
                                    .push(Cell::Notice(format!("lessons NOT committed: {e}"))),
                            },
                            Err(e) => app
                                .history
                                .push(Cell::Notice(format!("lessons NOT committed: {e}"))),
                        },
                        None => app.history.push(Cell::Notice(
                            "lessons NOT committed: staged file missing".into(),
                        )),
                    }
                }
                Effect::Quit => app.should_quit = true,
            }
            trace.record_ui(&trace::UiEvent::Effect {
                name: effect_name.into(),
                duration_ms: u64::try_from(effect_started.elapsed().as_millis())
                    .unwrap_or(u64::MAX),
            });
        }

        // Auto-save the transcript at turn boundaries (idle + changed) so the
        // session is always resumable; and once more on quit. Failures used to
        // be silent — they now surface as `error` trace records.
        if !app.running && app.history.len() != last_saved_len {
            if let Err(e) = save_session(&session_id, &app.history) {
                trace.record_ui(&trace::UiEvent::Error {
                    context: "session_save".into(),
                    message: e.to_string(),
                });
            }
            last_saved_len = app.history.len();
        }

        if app.should_quit {
            if let Err(e) = save_session(&session_id, &app.history) {
                trace.record_ui(&trace::UiEvent::Error {
                    context: "session_save".into(),
                    message: e.to_string(),
                });
            }
            break;
        }
    }
    Ok(())
}

/// Persist the current transcript under the session id (best-effort; errors
/// surface as trace records at the call sites).
fn save_session(id: &str, history: &[crate::cells::Cell]) -> std::io::Result<()> {
    let s = session::Session {
        id: id.to_string(),
        created: id.to_string(),
        history: history.to_vec(),
    };
    session::save(&session::sessions_dir(), &s)
}
