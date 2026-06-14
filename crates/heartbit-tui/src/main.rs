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
mod splash;
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
    AgentEvent, ApprovalDecision, AuthStyle, BoxedProvider, InterruptHandle, OnApproval, OnEvent,
    OnInput, OnText, OpenAiCompatProvider, OpenRouterProvider, Orchestrator, PermissionAction,
    PermissionRule, PermissionRuleset, RetryingProvider, SubAgentConfig,
};
use tokio::sync::Mutex;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::app::{App, Effect};
use crate::cells::Cell;
use crate::msg::{Msg, PendingTool};

const DEFAULT_MODEL: &str = "qwen/qwen3-235b-a22b-2507";

/// Per-turn budget for a delegated sub-agent. A real audit/research task fans
/// out into dozens of read/grep/bash turns; the old cap of 60 made the
/// investigation sub-agent die mid-task with "Max turns exceeded" (live trace
/// 6a2e92e3: the researcher failed at 60 after 41 reads + 17 bash, so the audit
/// returned partial). Kept well below the entry agent's 300 so a runaway
/// sub-agent still terminates.
const SUB_AGENT_MAX_TURNS: usize = 200;

/// Process-global experience store (new 2026-frontier core): records each
/// completed conversation as a trajectory and recalls a learned procedure to
/// prime a later similar request — the self-improvement flywheel, without
/// threading state through the agent-spawn machinery. Single-process TUI, so a
/// `OnceLock` is the natural home; lives for the session.
static TRAJECTORY_STORE: std::sync::OnceLock<Arc<heartbit_core::agent::TrajectoryStore>> =
    std::sync::OnceLock::new();

fn trajectory_store() -> &'static Arc<heartbit_core::agent::TrajectoryStore> {
    TRAJECTORY_STORE.get_or_init(|| Arc::new(heartbit_core::agent::TrajectoryStore::new(200)))
}

/// Warning notice when `HEARTBIT_MODEL` is set: the env var has higher
/// precedence than the config, so `/model` changes are silently ignored until
/// it is unset (live-finding footgun). Pure for testing; the caller passes the
/// env value. `None` when the env is unset/empty.
fn heartbit_model_override_notice(env_model: Option<&str>) -> Option<String> {
    let m = env_model.map(str::trim).filter(|s| !s.is_empty())?;
    Some(format!(
        "\u{26a0} HEARTBIT_MODEL={m} overrides the model — /model changes won't apply until you unset it"
    ))
}

#[cfg(test)]
mod model_override_tests {
    use super::heartbit_model_override_notice as f;

    #[test]
    fn env_set_warns() {
        let n = f(Some("mistralai/mistral-medium-3-5")).expect("warns");
        assert!(n.contains("HEARTBIT_MODEL=mistralai/mistral-medium-3-5"));
        assert!(n.contains("/model"));
    }

    #[test]
    fn env_unset_or_empty_is_silent() {
        assert!(f(None).is_none());
        assert!(f(Some("")).is_none());
        assert!(f(Some("   ")).is_none());
    }
}

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

fn main() -> anyhow::Result<()> {
    // Config loads FIRST, on the still-single-threaded main: search-provider
    // keys from tui.toml are exported to the process env here, BEFORE any
    // thread exists (`std::env::set_var` is unsafe once threads run — the
    // tokio runtime and the trace writer both spawn them). Live finding: a
    // key parked in .env never reached the process and search silently fell
    // back to scraped DuckDuckGo.
    let cfg = config::TuiConfig::load();
    for (var, val) in [
        ("EXA_API_KEY", &cfg.exa_api_key),
        ("TAVILY_API_KEY", &cfg.tavily_api_key),
        ("BRAVE_API_KEY", &cfg.brave_api_key),
    ] {
        if std::env::var_os(var).is_none()
            && let Some(v) = val.as_deref().filter(|v| !v.is_empty())
        {
            // SAFETY: no threads have been spawned yet (no tokio runtime, no
            // trace writer) — mutating the environment is race-free here.
            unsafe { std::env::set_var(var, v) };
        }
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(cfg))
}

async fn run(cfg: config::TuiConfig) -> anyhow::Result<()> {
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
    // A custom OpenAI-compatible endpoint (e.g. a ChatGPT-subscription Codex
    // proxy, a local model, or the real OpenAI API) is a complete provider on its
    // own — no OpenRouter key needed, so the agent may spawn.
    let custom_endpoint = std::env::var("HEARTBIT_OPENAI_BASE_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let has_custom_endpoint = custom_endpoint.is_some();
    let has_fallback = has_anthropic || has_custom_endpoint;

    let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let input_rx = Arc::new(Mutex::new(input_rx));
    let cwd = std::env::current_dir()?;

    let mut app = App::new(model);
    app.api_key = api_key;
    app.has_fallback_provider = has_fallback;
    app.custom_endpoint = custom_endpoint;
    app.mcp_servers = cfg.mcp_servers.clone();
    app.multi_agent = cfg.multi_agent;
    app.context_recall = cfg.context_recall;
    app.verify_command = cfg.verify_command.clone();
    app.prompt_caching = cfg.prompt_caching;
    app.splash = cfg.splash.then_some(0);
    // Same deterministic registry the engine builds — /workflows lists it.
    app.workflow_recipes = heartbit_core::default_registry().meta();
    // Per-directory persistent prompt history: ↑ recalls prompts from
    // previous sessions in THIS directory.
    app.composer
        .seed_history(session::load_prompt_history(&session::prompt_history_file(
            &cwd,
        )));
    app.fast_model = cfg.fast_model.clone();
    app.frontier_model = cfg.frontier_model.clone();
    app.workflow_journal_dir = session::sessions_dir().join("journals").join(&session_id);
    // The unified entry agent can ALWAYS delegate (the squad is always available),
    // so seed the roster's available squad unconditionally — it shows when the
    // agent actually dispatches sub-agents.
    app.squad = app::DEFAULT_SQUAD.iter().map(|s| s.to_string()).collect();
    // Fetch the OpenRouter catalog at startup (public endpoint) so the status-line
    // context bar knows the model's window and the /model picker is pre-warmed.
    app.models_loading = true;
    app.effects.push(Effect::FetchModels);
    // Surface the HEARTBIT_MODEL override footgun: if it's set, /model is a
    // no-op and the user would otherwise be baffled.
    if let Some(notice) =
        heartbit_model_override_notice(std::env::var("HEARTBIT_MODEL").ok().as_deref())
    {
        app.history.push(Cell::Notice(notice));
    }
    // Custom OpenAI-compatible endpoint in use: tell the user (and remind them to
    // set the matching model with /model — the default is an OpenRouter id).
    if let Some(url) = app.custom_endpoint.clone() {
        app.history.push(Cell::Notice(format!(
            "custom OpenAI-compatible endpoint: {url} — set the model with /model \
             (e.g. gpt-5.5 for a ChatGPT-subscription Codex proxy; check its \
             /v1/models). This takes priority over OpenRouter."
        )));
    }
    // No provider configured at all → open the key prompt immediately.
    if app.api_key.is_none() && !has_fallback {
        app.modal = Some(app::Modal::KeyEntry(app::KeyEntryModal::default()));
    }

    let interrupt = InterruptHandle::new();
    // True while the engine idles at the on_input boundary (set by the agent
    // thread, cleared by the UI before each send): an Esc that lands in that
    // window must NOT latch the interrupt token — the runner only rearms it
    // inside a turn, so the next message would instantly self-interrupt.
    let agent_parked = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Shared permission posture (0=normal, 1=plan, 2=yolo), cycled by the UI
    // (Shift+Tab) and read live by the agent thread's `on_approval`.
    let perm_mode = Arc::new(std::sync::atomic::AtomicU8::new(
        app::PermissionMode::Yolo.as_u8(),
    ));
    // Request-intent pin (0 = auto): shared with the core router so
    // `/mode study|clarify|execute|answer` takes effect live.
    let request_mode_pin = Arc::new(std::sync::atomic::AtomicU8::new(0));

    // Config snapshot at launch — the trace's first record (init_tracing runs
    // after this so its captured banner can't claim seq 0).
    let lessons_loaded = lessons::load_lessons()
        .map(|c| lessons::lesson_count(&c))
        .unwrap_or(0);
    trace_handle.record_ui(&trace::UiEvent::SessionStarted {
        version: env!("CARGO_PKG_VERSION").into(),
        session_id: session_id.clone(),
        model: app.model.clone(),
        permission_mode: app.permission_mode.label().to_lowercase(),
        mcp_servers: cfg.mcp_servers.iter().map(|s| s.label()).collect(),
        context_recall: app.context_recall,
        verify_command: app.verify_command.clone(),
        lessons_loaded,
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
        agent_parked,
        perm_mode,
        request_mode_pin,
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
    custom_endpoint: Option<&str>,
    openrouter_key: Option<String>,
    model: &str,
    on_retry: Arc<heartbit_core::OnRetry>,
    prompt_caching: bool,
) -> anyhow::Result<Arc<BoxedProvider>> {
    // Custom OpenAI-compatible endpoint (`/codex`, or `HEARTBIT_OPENAI_BASE_URL` at
    // startup). Takes PRIORITY over OpenRouter so the same config can target: a
    // local model (Ollama / vLLM / LM Studio), the real OpenAI API, OR a localhost
    // proxy that bridges a ChatGPT-subscription Codex token to OpenAI-compatible
    // requests (so the agent runs on the subscription's quota — see
    // docs/chatgpt-subscription.md). A key present ⇒ Bearer over HTTPS (real API);
    // no key ⇒ AuthStyle::None, which is what permits a non-HTTPS localhost
    // base_url (the Codex proxy).
    if let Some(base_url) = custom_endpoint.filter(|u| !u.trim().is_empty()) {
        let base_url = base_url.to_string();
        let is_tls = base_url.trim_start().starts_with("https://");
        let key = std::env::var("HEARTBIT_OPENAI_API_KEY").unwrap_or_default();
        // Bearer requires HTTPS (the provider rejects a key over plain http). A
        // localhost proxy is `http://`, so a stray `HEARTBIT_OPENAI_API_KEY` in the
        // environment must NOT force Bearer there — fall back to `AuthStyle::None`,
        // which is what permits the non-TLS URL. Bearer only when both a key is
        // present AND the endpoint is TLS (a real HTTPS API).
        let auth = if !key.trim().is_empty() && is_tls {
            AuthStyle::Bearer
        } else {
            AuthStyle::None
        };
        let base = OpenAiCompatProvider::new(key, model, base_url, auth);
        return Ok(Arc::new(BoxedProvider::new(
            RetryingProvider::with_defaults(base).with_on_retry(on_retry),
        )));
    }
    if let Some(key) = openrouter_key {
        // Prompt caching (default ON): cache_control breakpoints land on the
        // system prompt + conversation prefix. Qwen/Anthropic/Gemini routes
        // honour them via OpenRouter (live: 99.7% of the prompt cached on
        // qwen3.7-max); non-supporting routes strip them (verified live on
        // qwen3-235b). Escape hatch: `prompt_caching = false` in tui.toml.
        let mut base = OpenRouterProvider::new(key, model);
        if prompt_caching {
            base = base.with_prompt_caching();
        }
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
    /// Run the (multi-turn) session, starting with `first`. Returns the final
    /// [`AgentOutput`](heartbit_core::AgentOutput) so the caller can record it
    /// (e.g. into the experience store).
    async fn run(&mut self, first: &str) -> anyhow::Result<heartbit_core::AgentOutput> {
        Ok(self.0.run(first).await?)
    }
}

/// A fresh set of workspace-rooted builtin tools (each call gets its own
/// `FileTracker` etc.). The API key never enters the tool env (safe allowlist).
fn fresh_builtins(
    cwd: &std::path::Path,
    context_recall_store: Option<&Arc<heartbit_core::ContextRecallStore>>,
    todo_store: Option<&Arc<heartbit_core::tool::builtins::TodoStore>>,
    on_question: Option<Arc<heartbit_core::tool::builtins::OnQuestion>>,
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
    // When present, registers the `question` tool (agent-to-user structured
    // questions). Entry agent ONLY — clarification owns the user channel and
    // does not propagate to sub-agents.
    tool_cfg.on_question = on_question;
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
        let mut tools = fresh_builtins(cwd, recall.as_ref(), Some(&todo), None);
        tools.extend(mcp_tools.iter().cloned());
        SubAgentConfig {
            name: name.into(),
            description: description.into(),
            system_prompt: prompt.into(),
            tools,
            max_turns: Some(SUB_AGENT_MAX_TURNS),
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
            "You are a focused implementation engineer. Do the delegated task end-to-end with the tools, make the smallest correct change, verify it, and report a concise result. Return your findings in your final message — do NOT write scratch/coordination files in the repo; if you genuinely need a working file, put it under ./scratch (gitignored), never the repo root.",
        ),
        make(
            app::DEFAULT_SQUAD[1],
            "Investigation agent: explores the codebase and gathers facts (search, read files, run read-only commands). Use to understand, locate, or analyze before changes.",
            "You are a careful researcher. Investigate the delegated question using the tools, then report concrete findings (file paths, line numbers, facts) — do not make changes unless asked. For a read-only task, do not write ANY files (no scratchpad/blackboard in the repo): return everything in your final message; if a working file is truly needed, put it under ./scratch (gitignored), never the repo root.",
        ),
    ]
}

/// Build the agent engine (single or multi-agent), wiring the synchronous
/// callbacks to the UI channels. The OpenRouter token is passed only to the
/// provider, never into the tool environment (bash gets a no-secrets allowlist).
#[allow(clippy::too_many_arguments)]
async fn build_engine(
    custom_endpoint: Option<String>,
    api_key: Option<String>,
    model: &str,
    ui_tx: UnboundedSender<Msg>,
    input_rx: Arc<Mutex<UnboundedReceiver<String>>>,
    cwd: PathBuf,
    interrupt: InterruptHandle,
    agent_parked: Arc<std::sync::atomic::AtomicBool>,
    mcp_servers: Vec<config::McpServerSpec>,
    context_recall: bool,
    context_window: Option<u32>,
    verify_command: Option<String>,
    perm_mode: Arc<std::sync::atomic::AtomicU8>,
    trace: trace::TraceHandle,
    prompt_caching: bool,
    fast_model: Option<String>,
    frontier_model: Option<String>,
    request_mode_pin: Arc<std::sync::atomic::AtomicU8>,
    workflow_journal_dir: PathBuf,
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
    let provider = build_provider(
        custom_endpoint.as_deref(),
        api_key.clone(),
        model,
        on_retry.clone(),
        prompt_caching,
    )?;

    // Connect MCP once (on this thread's runtime — the stdio transport binds to
    // its spawn runtime). The tools are Arc, shared across agents. Successes
    // fold into the single startup summary line; failures stay loud.
    let mut mcp_tools: Vec<Arc<dyn heartbit_core::tool::Tool>> = Vec::new();
    let mut summary_parts: Vec<String> = Vec::new();
    for spec in &mcp_servers {
        let label = spec.label();
        let _ = ui_tx.send(Msg::Notice(format!("connecting MCP {label}…")));
        match connect_mcp(spec).await {
            Ok(t) => {
                summary_parts.push(format!("MCP {label} ({} tools)", t.len()));
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
    // Agent-to-user structured questions (completion-loop harness P5): the
    // `question` tool sends the request to the UI thread (options modal) and
    // awaits the user's selections through a oneshot channel. Entry agent only.
    let on_question: Arc<heartbit_core::tool::builtins::OnQuestion> = {
        let tx = ui_tx.clone();
        Arc::new(move |req: heartbit_core::tool::builtins::QuestionRequest| {
            let tx = tx.clone();
            Box::pin(async move {
                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                tx.send(Msg::Question {
                    request: req,
                    reply: reply_tx,
                })
                .map_err(|_| heartbit_core::Error::Agent("UI channel closed".into()))?;
                reply_rx.await.map_err(|_| {
                    heartbit_core::Error::Agent(
                        "the user dismissed the question — proceed with your best \
                         judgment and state the assumption"
                            .into(),
                    )
                })
            })
        })
    };
    // The entry agent's direct tools: builtins FIRST so MCP can't shadow a trusted one.
    let mut tools = fresh_builtins(
        &cwd,
        recall_store.as_ref(),
        Some(&todo_store),
        Some(on_question),
    );
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
    // Model-role resolver shared by workflow stages ("fast") and the advisor
    // ("frontier"): roles map to configured models, anything else is a raw
    // model id; resolving to the main model reuses the session provider.
    let provider_factory: Arc<heartbit_core::ProviderFactory> = {
        let api_key = api_key.clone();
        let on_retry = on_retry.clone();
        let main_provider = provider.clone();
        let main_model = model.to_string();
        let fast = fast_model.clone();
        let frontier = frontier_model.clone();
        // A custom endpoint (e.g. the Codex proxy) is the ONLY provider this
        // session has — sub-role models must target it too, not OpenRouter.
        let custom_endpoint = custom_endpoint.clone();
        Arc::new(move |role: &str| {
            let resolved = match role {
                "main" | "" => return Ok(main_provider.clone()),
                "fast" => fast.clone().unwrap_or_else(|| main_model.clone()),
                "frontier" => frontier.clone().unwrap_or_else(|| main_model.clone()),
                other => other.to_string(),
            };
            if resolved == main_model {
                return Ok(main_provider.clone());
            }
            build_provider(
                custom_endpoint.as_deref(),
                api_key.clone(),
                &resolved,
                on_retry.clone(),
                true,
            )
            .map_err(|e| heartbit_core::Error::Config(format!("provider for '{role}': {e}")))
        })
    };
    // Recipe-internal agents stream their events to the TRACE ONLY — not to
    // the UI Msg plane: an inner runner's text-only LlmDone / RunCompleted
    // would flip `running=false` (and commit staged lessons) mid-recipe. The
    // trace tap is what /stats and /analyze read, which is where a recipe run
    // was previously a single opaque tool call.
    let recipe_trace = trace.clone();
    tools.push(Arc::new(
        heartbit_core::RunWorkflowTool::new(registry, provider.clone())
            .with_agent_events(Arc::new(move |e: AgentEvent| recipe_trace.record_agent(&e)))
            .with_provider_factory(provider_factory.clone())
            // Session-scoped resume: re-asking the SAME workflow in this
            // session replays completed agents; new session = fresh.
            .with_journal_dir(workflow_journal_dir)
            // Repo root for Isolation::Worktree recipes.
            .with_workspace(cwd.clone()),
    ));
    // The advisor: a frontier-model reviewer over the FULL transcript (the
    // runner snapshots it into ExecutionContext at every tool dispatch).
    match provider_factory("frontier") {
        Ok(frontier) => {
            tools.push(Arc::new(heartbit_core::AdvisorTool::new(frontier)));
            summary_parts.push(match &frontier_model {
                Some(m) => format!("advisor: {m}"),
                // A same-model self-review is a weaker advisor — say so.
                None => {
                    "advisor: main model (set frontier_model for a stronger reviewer)".to_string()
                }
            });
        }
        Err(e) => {
            let _ = ui_tx.send(Msg::Notice(format!("advisor disabled: {e}")));
        }
    }
    // Session handoff (completion-loop harness P10/P11): purpose-tailored
    // brief over the same transcript seam — distilled by the "fast" role
    // (falls back to the main model), written under <config>/handoffs/.
    match provider_factory("fast") {
        Ok(fast) => {
            tools.push(Arc::new(heartbit_core::SessionHandoffTool::new(
                fast,
                session::handoffs_dir(),
            )));
        }
        Err(e) => {
            let _ = ui_tx.send(Msg::Notice(format!("handoff disabled: {e}")));
        }
    }
    // Judge-gated completion, runtime path: the entry agent installs its own
    // acceptance criteria via `set_goal` (or the user via /goal); the "fast"
    // role judges every natural stop until they're met.
    let entry_goal_judge = match provider_factory("fast") {
        Ok(judge) => Some(judge),
        Err(e) => {
            let _ = ui_tx.send(Msg::Notice(format!("goal judge disabled: {e}")));
            None
        }
    };
    // Scope guard (anti-drift): the agent declares its blast radius with
    // `set_scope`; edits outside it are denied. Unseeded = no restriction.
    let scope_guard = Arc::new(heartbit_core::ScopeGuard::new(vec![]));
    tools.push(Arc::new(
        heartbit_core::SetScopeTool::new(scope_guard.clone())
            // Outside→inside protection: a task scoped outside this repo
            // (e.g. "un répertoire temporaire") can't silently relocate into
            // it (live finding 6a25947c: it re-scoped into the repo twice and
            // even added itself to the workspace Cargo.toml).
            .with_workspace(cwd.clone()),
    ));

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
        let parked = agent_parked;
        Arc::new(move || {
            let rx = rx.clone();
            let parked = parked.clone();
            Box::pin(async move {
                // The engine idles here between turns. While parked, an Esc has
                // nothing to interrupt — the UI skips the cancel, because the
                // runner only rearms the token INSIDE a turn, so a cancel that
                // lands now would poison the NEXT turn (instant self-interrupt
                // of the user's next message). The UI clears the flag BEFORE
                // sending, so a message queued right at the boundary stays
                // interruptible.
                parked.store(true, std::sync::atomic::Ordering::SeqCst);
                let msg = rx.lock().await.recv().await;
                parked.store(false, std::sync::atomic::Ordering::SeqCst);
                msg
            }) as Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
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
        summary_parts.push(format!("context: {}", names.join(", ")));
    }

    // Unified entry agent (option C): ALWAYS built, no static mode flag. The
    // orchestrator evolved into ONE capable agent — it holds direct tools +
    // delegation tools (delegate_task / form_squad) + run_workflow, and decides
    // per request via tool choice (answer directly / do simple work / delegate /
    // run a workflow). One run() drives the whole multi-turn session. (The
    // blurb lives in /help now — startup stays one compact summary line.)
    let replan = verify_command.as_deref().is_some_and(|c| !c.is_empty());
    // Which search backend Auto mode will actually use — surfaced so a
    // missing API key is visible BEFORE a session degrades to the scraped
    // DuckDuckGo fallback (which bot-walls under repeated queries).
    summary_parts.push(format!(
        "search: {}",
        heartbit_core::tool::builtins::search_provider_label()
    ));
    if recall_store.is_some() {
        summary_parts.push("recall ON".into());
    }
    if verify_command.as_deref().is_some_and(|c| !c.is_empty()) {
        summary_parts.push("verify ON".into());
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
    // Learned lessons (self-improvement rung 2): inject the distilled lessons
    // as standing guidance, after project context + verify nudge.
    let instructions = match lessons::load_lessons() {
        Some(lessons) => {
            let n = lessons::lesson_count(&lessons);
            summary_parts.push(format!("{n} lessons"));
            format!("{instructions}\n\n## Learned lessons (self-improvement — /learn)\n{lessons}")
        }
        None => instructions,
    };
    // Orchestration-selection guidance (Cemri et al. 2503.13657). The entry agent
    // decides whether to fan out; this is exactly the "decompose a broad task into
    // INDEPENDENT parallel sub-agent tasks, don't hand one giant task to one
    // sub-agent" advice. It lived in core as a public const but was wired into NO
    // prompt (shelfware) — a live audit run gave the whole investigation to a
    // single researcher, which then died at its turn cap. Inject it here so the
    // entry agent actually sees it.
    let instructions = format!(
        "{instructions}\n\n## When to delegate vs. fan out\n{}\n\
         For a broad audit/survey/migration, split the work into several focused, \
         INDEPENDENT sub-agent tasks (e.g. one per area or risk class) and delegate \
         them together, rather than one large task to a single sub-agent.",
        heartbit_core::MULTI_AGENT_SELECTION_GUIDANCE,
    );
    // ONE compact startup line instead of the old five-notice wall (campaign
    // round-1 frame evidence). Failures above stay as their own loud notices.
    let _ = ui_tx.send(Msg::Notice(format!(
        "ready — {}",
        if summary_parts.is_empty() {
            "builtins only".to_string()
        } else {
            summary_parts.join(" · ")
        }
    )));
    // SECURITY: surface heartbit-core's lethal-trifecta check (Willison, Jun
    // 2025) as a startup notice. When the agent's tools can simultaneously read
    // private data, ingest untrusted content, AND communicate externally, an
    // indirect prompt injection can exfiltrate the private data. (New capability
    // from the 2026-frontier core.) Analyse BEFORE `tools` is moved below.
    if let Some(warning) = heartbit_core::tool::analyze_tools(&tools).warning() {
        let _ = ui_tx.send(Msg::Notice(format!("⚠ security — {warning}")));
    }
    let mut builder = Orchestrator::builder(provider)
        .entry_agent(tools)
        .guardrail(scope_guard)
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
        .interrupt(interrupt)
        // Doom-loop detection (live /analyze finding: a 7-error retry spiral
        // ran unchecked — the infra existed in core but was never enabled
        // here). Identical batches abort fast; near-duplicates get more rope.
        .max_identical_tool_calls(3)
        .max_fuzzy_identical_tool_calls(5);
    if let Some(judge) = entry_goal_judge {
        builder = builder.entry_goal_judge(judge);
    }
    // Request-intent router: marker layer + "fast" classifier + safe default,
    // with the /mode pin shared from the UI thread.
    let router_fast = provider_factory("fast").ok();
    builder = builder.entry_request_router(Arc::new(
        heartbit_core::agent::router::RequestRouter::new(router_fast)
            .with_pin(request_mode_pin.clone()),
    ));
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
        // Harness-dialogue tools: no workspace effect — asking the user to
        // APPROVE the agent's request to ask them a question is a double
        // modal (live finding, session 6a254624). set_goal/set_scope mutate
        // in-process harness state only; handoff writes its brief OUTSIDE the
        // workspace (<config>/handoffs).
        allow("question"),
        allow("set_goal"),
        allow("set_scope"),
        allow("handoff"),
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

/// Gate an agent-thread message before it reaches the reducer: a stale
/// [`Msg::AgentExited`] (an epoch we already replaced) must be DROPPED — the
/// reducer can't see epochs and would flip `running=false` / clear `learning`
/// in the middle of the NEW engine's run. The current epoch's exit clears
/// `agent_started` so the next message respawns.
fn admit_agent_msg(msg: &Msg, agent_epoch: u64, agent_started: &mut bool) -> bool {
    match msg {
        Msg::AgentExited(e) if *e == agent_epoch => {
            *agent_started = false;
            true
        }
        Msg::AgentExited(_) => false,
        _ => true,
    }
}

/// Sends [`Msg::AgentExited`] when dropped — including on a PANIC anywhere in
/// the agent thread (runtime build, engine build, the runner). Without it the
/// exit signal lived at the end of the happy path only: a panic left
/// `agent_started` stuck true, so every subsequent message was silently sent
/// into a dead channel (spinner forever, no diagnosis — stderr is the
/// alt-screen).
struct AgentExitGuard {
    tx: UnboundedSender<Msg>,
    epoch: u64,
}

impl Drop for AgentExitGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let _ = self.tx.send(Msg::Notice(
                "agent thread panicked — the engine will restart on your next message".into(),
            ));
        }
        let _ = self.tx.send(Msg::AgentExited(self.epoch));
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
    agent_parked: &Arc<std::sync::atomic::AtomicBool>,
    epoch: u64,
    perm_mode: &Arc<std::sync::atomic::AtomicU8>,
    request_mode_pin: &Arc<std::sync::atomic::AtomicU8>,
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
    let custom_endpoint = app.custom_endpoint.clone();
    let api_key = app.api_key.clone();
    let model = app.model.clone();
    let mcp_servers = app.mcp_servers.clone();
    let context_recall = app.context_recall;
    let context_window = app.context_limit().map(|w| w.min(u32::MAX as u64) as u32);
    let verify_command = app.verify_command.clone();
    let prompt_caching = app.prompt_caching;
    let fast_model = app.fast_model.clone();
    let frontier_model = app.frontier_model.clone();
    let request_mode_pin = request_mode_pin.clone();
    let workflow_journal_dir = app.workflow_journal_dir.clone();
    let runner_tx = ui_tx.clone();
    let done_tx = ui_tx.clone();
    let input_rx = input_rx.clone();
    let cwd = cwd.to_path_buf();
    let interrupt = interrupt.clone();
    let agent_parked = agent_parked.clone();
    let perm_mode = perm_mode.clone();
    let trace = trace.clone();
    std::thread::spawn(move || {
        // Declared BEFORE anything that can panic: its Drop signals this
        // thread's exit (with its epoch) so the UI can respawn — even when the
        // runtime build, the engine build, or the runner panics. The UI
        // ignores a stale exit if the engine was already replaced.
        let _exit_guard = AgentExitGuard {
            tx: done_tx.clone(),
            epoch,
        };
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("agent runtime");
        rt.block_on(async move {
            match build_engine(
                custom_endpoint,
                api_key,
                &model,
                runner_tx,
                input_rx.clone(),
                cwd,
                interrupt,
                agent_parked.clone(),
                mcp_servers,
                context_recall,
                context_window,
                verify_command,
                perm_mode,
                trace,
                prompt_caching,
                fast_model,
                frontier_model,
                request_mode_pin,
                workflow_journal_dir,
            )
            .await
            {
                Ok(mut engine) => {
                    // MCP is now connected (eagerly). Wait for the first user message,
                    // then run; `on_input` feeds the rest from the same channel. The
                    // parked flag mirrors the on_input contract: while waiting, an Esc
                    // has nothing to interrupt (see Effect::Interrupt).
                    agent_parked.store(true, std::sync::atomic::Ordering::SeqCst);
                    let first = input_rx.lock().await.recv().await;
                    agent_parked.store(false, std::sync::atomic::Ordering::SeqCst);
                    if let Some(first) = first {
                        // Experience flywheel (new core): prime the request with a
                        // learned procedure from a similar SUCCESSFUL conversation
                        // earlier this session, then record this run. Empty store
                        // (fresh session) → no priming → identical behaviour, so
                        // this is purely additive. The prime is invisible to the
                        // UI (which already shows the user's typed message); only
                        // the agent sees it.
                        let store = trajectory_store();
                        let primed = match store.skill_hint(&first) {
                            Some(hint) => {
                                format!("{hint}\n\n---\nNow handle this request:\n{first}")
                            }
                            None => first.clone(),
                        };
                        let result = engine.run(&primed).await;
                        if let Ok(output) = &result {
                            store.record(heartbit_core::agent::Trajectory {
                                task: first.clone(),
                                actions: output
                                    .tool_call_results
                                    .iter()
                                    .map(|r| r.tool_name.clone())
                                    .collect(),
                                success: output.goal_met != Some(false),
                                result: output.result.clone(),
                            });
                        }
                    }
                }
                Err(e) => {
                    let _ = done_tx.send(Msg::Notice(format!("cannot start agent: {e}")));
                }
            }
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
    agent_parked: Arc<std::sync::atomic::AtomicBool>,
    perm_mode: Arc<std::sync::atomic::AtomicU8>,
    request_mode_pin: Arc<std::sync::atomic::AtomicU8>,
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
        &agent_parked,
        agent_epoch,
        &perm_mode,
        &request_mode_pin,
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
                    // the next message to rebuild the runner. A stale exit whose
                    // epoch we already superseded is dropped BEFORE the reducer
                    // (it would flip running=false mid-new-run).
                    if admit_agent_msg(&m, agent_epoch, &mut agent_started) {
                        app.update(m);
                    }
                    while let Ok(m2) = ui_rx.try_recv() {
                        if admit_agent_msg(&m2, agent_epoch, &mut agent_started) {
                            app.update(m2);
                        }
                    }
                }
            }
            _ = tick.tick() => {
                // Ticks animate the spinner (while running) and advance the
                // startup splash (while idle) — without the splash arm the
                // overlay would never auto-dismiss on a quiet startup (live
                // pty finding: key-skip worked, the timer never fired).
                if app.running || app.splash.is_some() {
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
                            &agent_parked,
                            agent_epoch,
                            &perm_mode,
                            &request_mode_pin,
                            &trace,
                            "respawn",
                        );
                    }
                    // Cleared BEFORE the send: the engine is about to (or already
                    // does) work on this message — an Esc from here on must reach
                    // the interrupt token, even if the engine hasn't woken yet.
                    agent_parked.store(false, std::sync::atomic::Ordering::SeqCst);
                    let _ = input_tx.send(text);
                }
                Effect::RespawnAgent => {
                    // A model/advisor change: the live agent captured the old
                    // model at spawn. Recreate the input channel — dropping the
                    // old sender closes it, so the current agent (idle, blocked
                    // on `on_input`) reaches end-of-input and exits — then mark
                    // not-started so the NEXT message spawns a fresh agent that
                    // reads the new `app.model`. `on_input` holds a clone of
                    // `input_rx`; nothing else clones `input_tx`, so the close
                    // is clean. The reducer defers this effect to turn-idle;
                    // this guard is the backstop — swapping mid-turn would let
                    // the next message spawn a SECOND engine while the old one
                    // still runs (audit 2026-06-09).
                    if app.running {
                        app.pending_respawn = true;
                    } else {
                        let (ntx, nrx) = tokio::sync::mpsc::unbounded_channel::<String>();
                        input_tx = ntx;
                        input_rx = Arc::new(Mutex::new(nrx));
                        agent_started = false;
                    }
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
                            &agent_parked,
                            agent_epoch,
                            &perm_mode,
                            &request_mode_pin,
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
                Effect::SaveFrontierModel(model) => {
                    let mut cfg = config::TuiConfig::load();
                    cfg.frontier_model = model;
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
                Effect::SetRequestModePin(v) => {
                    request_mode_pin.store(v, std::sync::atomic::Ordering::Relaxed);
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
                Effect::PersistPrompt(prompt) => {
                    if let Err(e) =
                        session::append_prompt_history(&session::prompt_history_file(&cwd), &prompt)
                    {
                        tracing::warn!(error = %e, "could not persist prompt history");
                    }
                }
                Effect::ListHandoffs => {
                    let briefs = session::list_handoffs(&session::handoffs_dir());
                    let _ = ui_tx.send(Msg::HandoffsListed(briefs));
                }
                Effect::SeedFromHandoff(path) => match std::fs::read_to_string(&path) {
                    Ok(brief) => {
                        if agent_started {
                            // Honest note: an already-running engine keeps its
                            // context — for a PURE fresh session, pick the brief
                            // right after starting the TUI.
                            app.history.push(Cell::Notice(
                                "seeding into the CURRENT session (restart the TUI and pick \
                                 the brief at startup for a pure session)"
                                    .into(),
                            ));
                        }
                        // Inline (NOT a queued Effect::SendInput): an effect
                        // pushed mid-drain waits for the next key/tick — at
                        // idle that strands the seed indefinitely.
                        let text = format!(
                            "Continue from this handoff brief. Follow its Next steps; use its \
                             pointers instead of re-deriving context.\n\n{brief}"
                        );
                        trace.record_ui(&trace::UiEvent::UserInput { text: text.clone() });
                        if !agent_started {
                            agent_epoch += 1;
                            agent_started = spawn_agent(
                                app,
                                &ui_tx,
                                &input_rx,
                                &cwd,
                                &interrupt,
                                &agent_parked,
                                agent_epoch,
                                &perm_mode,
                                &request_mode_pin,
                                &trace,
                                "handoff_seed",
                            );
                        }
                        agent_parked.store(false, std::sync::atomic::Ordering::SeqCst);
                        let _ = input_tx.send(text);
                    }
                    Err(e) => app
                        .history
                        .push(Cell::Notice(format!("could not read brief: {e}"))),
                },
                Effect::EmergencyHandoff(error) => {
                    match session::write_emergency_brief(
                        &session::handoffs_dir(),
                        &session_id,
                        &error,
                        &app.history,
                    ) {
                        Ok(path) => app.history.push(Cell::Notice(format!(
                            "emergency handoff brief written: {} (pick it up with /handoff)",
                            path.display()
                        ))),
                        Err(e) => app
                            .history
                            .push(Cell::Notice(format!("could not write brief: {e}"))),
                    }
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
                    // Fetch the model catalog off the UI thread; the result comes
                    // back as Msg::ModelsLoaded / ModelsFailed. When a custom
                    // endpoint is active (e.g. the Codex proxy via /codex), list
                    // ITS models (`/v1/models`) so the picker offers the
                    // subscription's models, not the OpenRouter catalogue.
                    let tx = ui_tx.clone();
                    let endpoint = app.custom_endpoint.clone();
                    tokio::spawn(async move {
                        let result = match &endpoint {
                            Some(base) => models::fetch_openai_models(base).await,
                            None => models::fetch_openrouter_models().await,
                        };
                        let msg = match result {
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
                    // Turn-boundary race: when the engine is already parked at
                    // on_input there is nothing to interrupt — and the runner
                    // only rearms the token INSIDE a turn, so a cancel latched
                    // now would instantly self-interrupt the NEXT message.
                    if agent_parked.load(std::sync::atomic::Ordering::SeqCst) {
                        trace.record_ui(&trace::UiEvent::InterruptRequested {
                            checkpoint: "cp2_skipped_engine_idle".into(),
                            running: app.running,
                        });
                    } else {
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
                }
                Effect::ComputeStats(target) => {
                    let tx = ui_tx.clone();
                    let sid = session_id.clone();
                    tokio::spawn(async move {
                        let result = tokio::task::spawn_blocking(move || {
                            let dir = session::sessions_dir();
                            let path = trace::resolve_trace_target(&dir, &sid, target.as_deref())?;
                            let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
                            // "abc-1.trace.jsonl" → "abc-1" — the card's label.
                            let label = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("?")
                                .trim_end_matches(".jsonl")
                                .trim_end_matches(".trace")
                                .to_string();
                            Ok::<_, String>((label, Box::new(trace_stats::compute(file))))
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
            // Drain the trace tail before exiting — the writer thread is
            // detached, so without this the final records could vanish.
            let _ = trace.flush_blocking(std::time::Duration::from_millis(750));
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

#[cfg(test)]
mod permission_tests {
    use super::*;

    #[test]
    fn harness_dialogue_tools_are_auto_allowed() {
        // Live finding (session 6a254624): in normal mode the `question` tool
        // hit the wildcard Ask rule — the user had to APPROVE the agent's
        // request to ask them a question (double modal). Harness-dialogue
        // tools (question, set_goal, set_scope, handoff) touch no workspace
        // file and must never sit behind an approval prompt.
        let rules = default_permissions();
        for tool in ["question", "set_goal", "set_scope", "handoff"] {
            assert_eq!(
                rules.evaluate(tool, &serde_json::json!({})),
                Some(PermissionAction::Allow),
                "{tool} must be auto-allowed"
            );
        }
        // Mutating tools still go through approval.
        for tool in ["write", "edit", "bash", "patch"] {
            assert_eq!(
                rules.evaluate(tool, &serde_json::json!({})),
                Some(PermissionAction::Ask),
                "{tool} must still ask"
            );
        }
    }
}

#[cfg(test)]
mod sub_agent_config_tests {
    use super::*;

    #[test]
    fn delegated_sub_agents_get_a_high_turn_budget() {
        // Regression: the old 60-turn cap killed the audit's investigation
        // sub-agent mid-task (live trace 6a2e92e3 → "Max turns exceeded").
        // A delegated audit/research run needs a much larger budget.
        let cwd = std::path::PathBuf::from("/tmp");
        let agents = default_sub_agents(&cwd, &[], false, None, false);
        assert!(!agents.is_empty());
        for a in &agents {
            assert_eq!(
                a.max_turns,
                Some(SUB_AGENT_MAX_TURNS),
                "{} must carry the high sub-agent turn budget",
                a.name
            );
            // Floor check via the runtime value (not the const) so it stays well
            // above the old 60-turn cap that killed the audit's researcher.
            assert!(
                a.max_turns.unwrap() >= 200,
                "{} sub-agent budget must stay >= 200 (old cap 60 failed)",
                a.name
            );
        }
    }
}

#[cfg(test)]
mod agent_lifecycle_tests {
    use super::*;

    #[test]
    fn stale_agent_exit_is_not_admitted_to_the_reducer() {
        // Audit 2026-06-09: the epoch guard only protected `agent_started`;
        // the reducer still saw the stale exit and flipped running=false /
        // cleared `learning` in the middle of the NEW engine's run.
        let mut started = true;
        assert!(
            !admit_agent_msg(&Msg::AgentExited(1), 2, &mut started),
            "a stale exit (epoch 1 vs current 2) must be dropped"
        );
        assert!(started, "a stale exit must not reset agent_started");
        assert!(
            admit_agent_msg(&Msg::AgentExited(2), 2, &mut started),
            "the current epoch's exit reaches the reducer"
        );
        assert!(!started, "the current epoch's exit resets agent_started");
        let mut started = true;
        assert!(
            admit_agent_msg(&Msg::Notice("x".into()), 2, &mut started),
            "non-exit messages always pass"
        );
        assert!(started);
    }

    #[test]
    fn agent_exit_guard_sends_exited_even_on_panic() {
        // Audit 2026-06-09: the exit signal lived at the end of the happy path
        // only — a panic anywhere in the agent thread left `agent_started`
        // stuck true and every subsequent message was silently dropped.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Msg>();
        let handle = std::thread::spawn(move || {
            let _guard = AgentExitGuard { tx, epoch: 7 };
            panic!("boom");
        });
        assert!(handle.join().is_err(), "the thread did panic");
        let mut saw_exit = false;
        let mut saw_notice = false;
        while let Ok(m) = rx.try_recv() {
            match m {
                Msg::AgentExited(7) => saw_exit = true,
                Msg::Notice(n) if n.contains("panicked") => saw_notice = true,
                _ => {}
            }
        }
        assert!(saw_exit, "AgentExited(epoch) must be sent on panic");
        assert!(saw_notice, "the panic must surface as a visible notice");
    }
}
