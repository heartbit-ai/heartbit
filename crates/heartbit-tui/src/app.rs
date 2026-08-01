//! Application state + the pure `update(Msg)` reducer. No terminal, no channels:
//! state mutations only, with side-effects pushed onto `effects` for the edge
//! (main loop) to perform. This is what makes the whole interaction unit-testable.

use std::collections::HashMap;
use std::sync::mpsc::SyncSender;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use heartbit_core::{ApprovalDecision, TokenUsage};

use crate::cells::{Cell, ToolStatus};
use crate::composer::Composer;
use crate::config::McpServerSpec;
use crate::models::ModelEntry;
use crate::msg::{Msg, PendingTool};

/// How many transcript lines a PageUp/PageDown moves.
const SCROLL_STEP: u16 = 8;
/// How many transcript lines one mouse-wheel notch moves.
const WHEEL_STEP: u16 = 3;

/// Names of the default sub-agent pool — single source of truth shared by
/// `main::default_sub_agents` (what's actually built) and the TUI roster (the
/// available squad shown as Idle), so the two can't drift apart.
pub const DEFAULT_SQUAD: [&str; 2] = ["worker", "researcher"];

/// Live state of one agent in the multi-agent roster panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentState {
    /// Available in the squad but not (yet) dispatched this turn.
    Idle,
    /// Actively working (a tool is running, or synthesizing).
    Working,
    /// Finished successfully.
    Done,
    /// Finished with an error.
    Failed,
}

/// One row in the multi-agent roster: an agent, its live state, what it is doing
/// right now (deterministic — its latest tool/event, no extra LLM call), and the
/// token cost once it completes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRow {
    pub name: String,
    pub state: AgentState,
    pub activity: String,
    pub tokens: u32,
}

/// A task in the live todo panel (mirrors the agent's `todowrite` tool, which
/// always sends the COMPLETE list — so the panel just reflects the latest call).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoRow {
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// Parse a `todowrite` tool input's `todos[]` into rows (empty if malformed).
pub fn parse_todos(input_json: &str) -> Vec<TodoRow> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(input_json) else {
        return Vec::new();
    };
    let Some(arr) = v.get("todos").and_then(|t| t.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|t| {
            let content = t.get("content")?.as_str()?.to_string();
            let status = match t.get("status").and_then(|s| s.as_str()) {
                Some("completed") => TodoStatus::Completed,
                Some("in_progress") => TodoStatus::InProgress,
                _ => TodoStatus::Pending,
            };
            Some(TodoRow { content, status })
        })
        .collect()
}

/// Workspace-safe slug for `/research` artifacts: lowercase alphanumerics
/// joined by single dashes, capped at 40 chars; degenerate input → "research".
fn research_slug(question: &str) -> String {
    let mut slug = String::new();
    let mut dash = false;
    for c in question.chars().take(80).flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
            dash = false;
        } else if !dash && !slug.is_empty() {
            slug.push('-');
            dash = true;
        }
        if slug.len() >= 40 {
            break;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "research".into()
    } else {
        slug
    }
}

/// Global execution mode, cycled with Shift+Tab (Claude Code style) or set with
/// `/mode`. Gates the `on_approval` bridge so the user controls how autonomously
/// the agent acts this session. Three modes: Normal · Plan · YOLO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    /// **Normal** — ask (the approval modal) before consequential tool calls.
    Normal,
    /// **Plan** — read-only: the agent investigates and proposes, but cannot
    /// modify (edit/write/patch/bash are auto-denied). Switch to Normal/YOLO to
    /// execute the plan.
    Plan,
    /// **YOLO** — auto-allow everything; no interruptions. You only live once.
    Yolo,
}

/// Picker order — matches the Shift+Tab cycle.
pub const MODES: [PermissionMode; 3] = [
    PermissionMode::Normal,
    PermissionMode::Plan,
    PermissionMode::Yolo,
];

impl PermissionMode {
    /// Next mode in the Shift+Tab cycle: Normal → Plan → YOLO → Normal.
    pub fn next(self) -> Self {
        match self {
            PermissionMode::Normal => PermissionMode::Plan,
            PermissionMode::Plan => PermissionMode::Yolo,
            PermissionMode::Yolo => PermissionMode::Normal,
        }
    }
    /// Compact wire value shared with the (cross-thread) approval gate.
    pub fn as_u8(self) -> u8 {
        match self {
            PermissionMode::Normal => 0,
            PermissionMode::Plan => 1,
            PermissionMode::Yolo => 2,
        }
    }
    /// Short status-line label.
    pub fn label(self) -> &'static str {
        match self {
            PermissionMode::Normal => "normal",
            PermissionMode::Plan => "plan",
            PermissionMode::Yolo => "YOLO",
        }
    }
    /// One-line description for `/mode` feedback.
    pub fn describe(self) -> &'static str {
        match self {
            PermissionMode::Normal => "asks before consequential actions",
            PermissionMode::Plan => "read-only — investigates and proposes, never modifies",
            PermissionMode::Yolo => "auto-allows everything, no interruptions",
        }
    }
    /// Parse a `/mode` argument (accepts the old names too).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "normal" | "default" => Some(PermissionMode::Normal),
            "plan" => Some(PermissionMode::Plan),
            // "auto" now belongs to the request-mode pin release (/mode auto);
            // yolo keeps its own name.
            "yolo" => Some(PermissionMode::Yolo),
            _ => None,
        }
    }
}

/// Reasoning-effort level the user selected. `Off` (the default) omits the field
/// entirely, reproducing today's requests bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EffortLevel {
    #[default]
    Off,
    Low,
    Medium,
    High,
}

impl EffortLevel {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// The four levels in picker order.
    pub const ALL: [Self; 4] = [Self::Off, Self::Low, Self::Medium, Self::High];
}

/// Slash commands offered by the `/` autocomplete menu: (name, description).
pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "list commands"),
    ("/mode", "set execution mode: normal | plan | yolo"),
    ("/model", "set the model (`/model advisor` for the advisor)"),
    ("/effort", "set reasoning effort (off|low|medium|high)"),
    (
        "/handoff",
        "brief for another session (`/handoff <purpose>`)",
    ),
    (
        "/goal",
        "judge-gated objective (`/goal <objective>` | clear)",
    ),
    ("/mcp", "list / add / clear MCP servers"),
    ("/agents", "toggle multi-agent workflow mode"),
    ("/context-recall", "toggle context restore-on-demand"),
    (
        "/verify",
        "set the project verify command (self-verify + repair)",
    ),
    ("/clear", "clear the transcript"),
    ("/resume", "reopen a saved session"),
    ("/export", "export the transcript to Markdown"),
    ("/stats", "trace stats — this session, `last`, or <id>"),
    (
        "/analyze",
        "agent diagnosis of a trace — this session, `last`, or <id>",
    ),
    (
        "/learn",
        "distill /analyze findings into persistent lessons",
    ),
    ("/research", "deep research — fan-out, verify, cited report"),
    (
        "/workflows",
        "list the registered workflow recipes (run_workflow)",
    ),
    ("/key", "set the OpenRouter API key"),
    (
        "/codex",
        "run on a ChatGPT-subscription Codex proxy (`/codex [url|off]`)",
    ),
    ("/diff", "show the working-tree diff"),
    ("/quit", "exit the TUI"),
];

/// A side-effect for the edge (main loop) to perform after an update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// Submit a user message to the agent (start the run, or feed `on_input`).
    SendInput(String),
    /// Persist a new OpenRouter API key to the config file.
    SaveKey(String),
    /// Persist a new model id to the config file.
    SaveModel(String),
    /// Persist the reasoning-effort level to the config file (`None` = off,
    /// omitting the field entirely rather than storing `"off"`).
    SaveReasoningEffort(Option<String>),
    /// Persist the advisor's frontier model to the config file (None = clear,
    /// falling back to the main model).
    SaveFrontierModel(Option<String>),
    /// Persist the MCP server list to the config file.
    SaveMcp(Vec<McpServerSpec>),
    /// Fetch the OpenRouter model catalog (async, for the model picker).
    FetchModels,
    /// Walk the project tree (async) to build the `@`-mention file index.
    WalkFiles,
    /// Persist the multi-agent (orchestrator) mode flag to the config file.
    SaveContextRecall(bool),
    SaveVerifyCommand(Option<String>),
    /// Apply the permission mode to the shared (cross-thread) approval gate.
    SetPermissionMode(u8),
    /// Pin the request-intent mode (0 = auto; RequestMode::as_pin_u8 codes).
    SetRequestModePin(u8),
    /// Rebuild the agent thread so a model/advisor change takes effect — the
    /// running agent captured the OLD model at spawn (live finding: /model
    /// persisted but the long-lived agent kept the old model). Applied on the
    /// user's next message.
    RespawnAgent,
    /// Export the current transcript (read from `App.history`) to a Markdown file.
    ExportSession,
    /// List saved sessions (for the `/resume` picker).
    ListSessions,
    /// Load a saved session by id.
    ResumeSession(String),
    /// Abandon the in-flight turn (abort generation), keeping the session.
    Interrupt,
    /// Append a submitted prompt to the per-directory persistent history.
    PersistPrompt(String),
    /// List handoff briefs (for the `/handoff` picker).
    ListHandoffs,
    /// Seed the session from a chosen handoff brief (path of the .md file).
    SeedFromHandoff(std::path::PathBuf),
    /// Write a deterministic emergency handoff brief after a terminal run
    /// failure (no LLM — the provider may be the thing that failed).
    EmergencyHandoff(String),
    /// Compute trace stats (None = current session; "last" | <id> otherwise).
    ComputeStats(Option<String>),
    /// Prepare an `/analyze` run (resolve trace, compute stats, build prompt).
    Analyze(Option<String>),
    /// Prepare a `/learn` run (stage lessons, gather diagnoses, build prompt).
    Learn,
    /// Validate + commit the staged lessons (digest = value at stage time).
    CommitLessons(u64),
    /// Gather the cumulative working-tree diff (`/diff`): `git diff HEAD`
    /// plus untracked files, run off the UI thread — git is I/O, never
    /// invoked from the reducer.
    GitDiff,
    /// Fire a desktop notification (focus-gated turn-completion / approval —
    /// Task 7). `title`/`body` may carry agent-controlled text (tool names,
    /// provider error strings) UNsanitized: `notify::emit` sanitizes at the
    /// I/O boundary, immediately before writing bytes, not here.
    Notify {
        title: String,
        body: String,
    },
    /// Tear down and exit.
    Quit,
}

impl Effect {
    /// Stable name for the trace (`ui` `effect` records).
    pub fn name(&self) -> &'static str {
        match self {
            Effect::SendInput(_) => "send_input",
            Effect::SaveKey(_) => "save_key",
            Effect::SaveModel(_) => "save_model",
            Effect::SaveReasoningEffort(_) => "save_reasoning_effort",
            Effect::SaveFrontierModel(_) => "save_frontier_model",
            Effect::SaveMcp(_) => "save_mcp",
            Effect::FetchModels => "fetch_models",
            Effect::WalkFiles => "walk_files",
            Effect::SaveContextRecall(_) => "save_context_recall",
            Effect::SaveVerifyCommand(_) => "save_verify_command",
            Effect::SetPermissionMode(_) => "set_permission_mode",
            Effect::SetRequestModePin(_) => "set_request_mode_pin",
            Effect::RespawnAgent => "respawn_agent",
            Effect::ExportSession => "export_session",
            Effect::ListSessions => "list_sessions",
            Effect::ResumeSession(_) => "resume_session",
            Effect::Interrupt => "interrupt",
            Effect::PersistPrompt(_) => "persist_prompt",
            Effect::ListHandoffs => "list_handoffs",
            Effect::SeedFromHandoff(_) => "seed_from_handoff",
            Effect::EmergencyHandoff(_) => "emergency_handoff",
            Effect::ComputeStats(_) => "compute_stats",
            Effect::Analyze(_) => "analyze",
            Effect::Learn => "learn",
            Effect::CommitLessons(_) => "commit_lessons",
            Effect::GitDiff => "git_diff",
            Effect::Notify { .. } => "notify",
            Effect::Quit => "quit",
        }
    }
}

/// A pending tool-approval prompt.
pub struct ApprovalModal {
    pub tools: Vec<PendingTool>,
    pub reply: SyncSender<ApprovalDecision>,
}

/// A masked prompt to enter/update the OpenRouter API key.
#[derive(Default)]
pub struct KeyEntryModal {
    pub input: String,
}

/// The OpenRouter model picker overlay: a search query + the highlighted row
/// (an index into the FILTERED list, which is derived from `App.models`).
#[derive(Default)]
pub struct ModelPicker {
    pub query: String,
    pub selected: usize,
    /// Which model the selection applies to: the main agent model or the
    /// advisor's frontier model (`/model advisor`).
    pub target: ModelTarget,
}

/// What a model-picker selection sets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelTarget {
    /// The main agent model (`/model`).
    #[default]
    Main,
    /// The advisor's frontier model (`/model advisor`).
    Advisor,
}

/// Reverse history search (Ctrl+R): a query + the highlighted match index.
#[derive(Default)]
pub struct HistorySearch {
    pub query: String,
    pub sel: usize,
}

/// The `/resume` session picker: the saved-session list + highlighted row.
#[derive(Default)]
pub struct SessionPicker {
    pub sessions: Vec<crate::session::SessionMeta>,
    pub sel: usize,
}

/// Agent-to-user structured question modal (the `question` builtin): one or
/// more questions answered in sequence; selections are sent back through the
/// oneshot channel when the last question is confirmed.
pub struct QuestionModal {
    pub request: heartbit_core::tool::builtins::QuestionRequest,
    /// Consumed when the final answer is sent. Dropped on Esc (dismissal).
    pub reply:
        Option<tokio::sync::oneshot::Sender<heartbit_core::tool::builtins::QuestionResponse>>,
    /// Index of the question currently displayed.
    pub current: usize,
    /// Highlighted option for the current question.
    pub selected: usize,
    /// Multi-select toggles for the CURRENT question (reset on advance).
    pub picked: Vec<bool>,
    /// Confirmed answers (labels) for already-answered questions.
    pub answers: Vec<Vec<String>>,
}

/// A modal overlay.
pub enum Modal {
    Approval(ApprovalModal),
    KeyEntry(KeyEntryModal),
    ModelPicker(ModelPicker),
    Question(QuestionModal),
    /// `/handoff` brief picker: saved briefs, newest first.
    HandoffPicker {
        briefs: Vec<crate::session::HandoffMeta>,
        sel: usize,
    },
    /// `/mode` picker: choose the execution mode (`sel` indexes [`MODES`]).
    ModePicker {
        sel: usize,
    },
    /// `/effort` picker: choose the reasoning-effort level (`sel` indexes
    /// [`EffortLevel::ALL`]).
    EffortPicker {
        sel: usize,
    },
    HistorySearch(HistorySearch),
    SessionPicker(SessionPicker),
}

/// A message held in the visible queue: the user-facing text (shown in the
/// queue box, drained into the transcript as a `Cell::User`, and reloaded by
/// Up for editing) kept SEPARATE from the wire payload actually sent to the
/// agent (`Effect::SendInput`), which may carry an invisible directive the
/// display must never show — e.g. Plan mode's read-only prefix. A single
/// shared string here would leak that directive into the transcript and back
/// into the composer the moment a Plan-mode submit landed mid-turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QueuedInput {
    pub(crate) display: String,
    pub(crate) wire: String,
}

#[cfg(test)]
impl From<&str> for QueuedInput {
    /// Test-only convenience: most fixtures don't care about the display/wire
    /// split, so both sides get the same text.
    fn from(s: &str) -> Self {
        Self {
            display: s.to_string(),
            wire: s.to_string(),
        }
    }
}

/// The full UI state.
pub struct App {
    pub history: Vec<Cell>,
    /// Assistant text being streamed for the current turn (not yet finalized).
    pub active: Option<String>,
    /// Chain-of-thought being streamed for the current LLM call (reasoning
    /// models). Rendered live, dimmed, then flushed to a `Cell::Reasoning` ahead
    /// of the answer when the answer/tool starts (`finalize_active`).
    pub active_reasoning: Option<String>,
    pub composer: Composer,
    pub modal: Option<Modal>,
    pub model: String,
    /// Reasoning-effort level (`/effort`), gated to OpenRouter/custom-endpoint
    /// providers only — `main.rs::effort_for_provider` never lets it reach the
    /// `ANTHROPIC_API_KEY` fallback. Applies on next agent start.
    pub effort: EffortLevel,
    /// The OpenRouter API key in effect (from env, config, or set in-TUI).
    pub api_key: Option<String>,
    /// True when a provider can start without an OpenRouter key (e.g. an
    /// `ANTHROPIC_API_KEY` env fallback, or a custom endpoint via `/codex`) — so a
    /// no-key submit need not prompt.
    pub has_fallback_provider: bool,
    /// A custom OpenAI-compatible base URL (`http://127.0.0.1:.../v1`), set by
    /// `/codex` or the `HEARTBIT_OPENAI_BASE_URL` env var at startup. When `Some`,
    /// the engine builds an `OpenAiCompatProvider` against it (priority over
    /// OpenRouter), so the TUI can run on a ChatGPT-subscription Codex proxy or any
    /// OpenAI-compatible endpoint. Applies on the next agent start.
    pub custom_endpoint: Option<String>,
    /// While `/codex` is active, the `(model, has_fallback_provider)` that were in
    /// effect BEFORE activation — restored by `/codex off`. `/codex` is a SESSION
    /// override (never persisted): the model swap must not leak a Codex model id
    /// into the saved config, which would brick the next cold start when the proxy
    /// is gone. `None` ⇒ Codex not active.
    pub codex_saved: Option<(String, bool)>,
    pub tokens: TokenUsage,
    /// MCP servers to connect when the agent starts (mirrors the config file).
    pub mcp_servers: Vec<McpServerSpec>,
    /// The OpenRouter model catalog (lazily fetched for the picker).
    pub models: Vec<ModelEntry>,
    /// True while the catalog fetch is in flight.
    pub models_loading: bool,
    /// Multi-agent orchestrator mode (applies on next agent start).
    pub multi_agent: bool,
    /// Set once a delegation happens this session (the option-C entry agent
    /// delegates dynamically without the manual `/agents` toggle). Activates the
    /// roster + per-agent tool-cell badges so parallel sub-agents are attributed
    /// — live finding 6a25eb4d: 4 parallel workers ran with zero attribution.
    pub saw_delegation: bool,
    /// Context restore-on-demand (single-agent path): index tool outputs + gentle
    /// session pruner so old tool results truncate to a restorable marker. ON by
    /// default; toggled via `/context-recall`, applies on next start.
    pub context_recall: bool,
    /// Optional project verification command (`/verify <cmd>`). When set, the
    /// agent gets a `verify` tool + a self-verify prompt nudge. Applies next start.
    pub verify_command: Option<String>,
    /// OpenRouter prompt-caching breakpoints (escape hatch in tui.toml; ON by
    /// default — non-supporting routes strip the markers harmlessly).
    pub prompt_caching: bool,
    /// The available sub-agent pool (multi-agent mode), seeded into the roster as
    /// Idle at the start of each turn so the user always sees the whole squad —
    /// and can tell when only some of it actually gets dispatched.
    pub squad: Vec<String>,
    /// Live roster of agents for the current turn (multi-agent mode): who was
    /// instantiated and what each is doing right now. Ordered by first-seen.
    pub agents: Vec<AgentRow>,
    /// Live task list mirrored from the agent's latest `todowrite` call.
    pub todos: Vec<TodoRow>,
    /// Global permission posture (Shift+Tab cycles it).
    pub permission_mode: PermissionMode,
    /// Project file paths for `@`-mention autocomplete (walked lazily on first `@`).
    pub file_index: Vec<String>,
    /// True once a file-index walk has been requested (so we don't re-walk).
    pub files_requested: bool,
    /// Current context fill (latest request's input tokens) — for the status bar.
    pub context_tokens: u32,
    /// Time-to-first-token of the latest turn (ms) — status-line throughput.
    pub last_ttft_ms: u64,
    pub running: bool,
    /// Whether the terminal window currently has focus (`EnableFocusChange`).
    /// Defaults `true` so a terminal that never reports focus reads as
    /// focused. Task 7 (notifications) reads this to decide whether to notify.
    pub focused: bool,
    /// Desktop notifications on turn-completion/approval while unfocused
    /// (`tui.toml`'s `notify`, default on). Gates alongside `focused` and
    /// `splash`; the bytes are written by `notify::emit` from the main loop's
    /// effect pass — never here, never in `view()`, never on the agent thread.
    pub notify: bool,
    /// A model/advisor change landed MID-RUN: the engine rebuild is deferred
    /// to turn-idle (an immediate channel swap would let the next message
    /// spawn a second engine while the old one still runs — audit 2026-06-09).
    pub pending_respawn: bool,
    /// In-flight /learn: staged-lessons digest at stage time (commit guard).
    pub learning: Option<u64>,
    /// When `true`, the transcript stays pinned to the newest content (the
    /// default). The user un-pins by scrolling up; scrolling back to the bottom
    /// re-pins. While un-pinned, the view is anchored from the TOP (see
    /// `scroll_top`) so streaming output growing at the bottom never drifts the
    /// read position.
    pub follow: bool,
    /// Absolute rows hidden ABOVE the viewport when un-pinned (`!follow`).
    /// Top-anchored so bottom growth doesn't move it.
    pub scroll_top: u16,
    /// Last `max_off` (total rows − viewport height) the renderer computed, fed
    /// back so the wheel handler can convert a follow→scrolled transition into a
    /// top-anchored offset. Interior-mutable: written from `view()` (`&App`).
    last_max_off: std::cell::Cell<u16>,
    /// Highlighted row in the `/` command-autocomplete menu.
    pub menu_selected: usize,
    pub spinner: usize,
    /// Startup splash tick counter — `Some(t)` while the splash overlay is up
    /// (armed by main from config); `None` once dismissed (timer or any key).
    pub splash: Option<u8>,
    /// `(name, description)` of the registered workflow recipes — set by main
    /// from the same registry the agent gets, listed by `/workflows`.
    pub workflow_recipes: Vec<(String, String)>,
    /// Model for the "fast" role (workflow stages); falls back to the main model.
    pub fast_model: Option<String>,
    /// Model for the "frontier" role (the advisor); falls back to the main model.
    pub frontier_model: Option<String>,
    /// Session-scoped dir for run_workflow resume journals.
    pub workflow_journal_dir: std::path::PathBuf,
    pub should_quit: bool,
    pub effects: Vec<Effect>,
    /// Maps an in-flight tool_call_id to its index in `history`.
    tool_index: HashMap<String, usize>,
    /// Messages submitted while a turn was in flight, held HERE rather than
    /// pushed into the invisible unbounded input channel so the user can see,
    /// edit and cancel them. Invariant: non-empty ⇒ `running`.
    pub(crate) queued: std::collections::VecDeque<QueuedInput>,
    /// View-side memoization of highlighted Markdown (interior-mutable, same
    /// precedent as `last_max_off`): `terminal.draw()` re-renders every agent
    /// cell on every keystroke and every 120ms tick, so re-running syntect
    /// uncached would burn a syntax parser dozens of times a second. The
    /// reducer never reads or writes this field — see `Msg::Resize` below.
    pub(crate) md: crate::markdown::MarkdownCache,
}

impl App {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            history: Vec::new(),
            active: None,
            active_reasoning: None,
            composer: Composer::new(),
            modal: None,
            model: model.into(),
            effort: EffortLevel::default(),
            api_key: None,
            has_fallback_provider: false,
            custom_endpoint: None,
            codex_saved: None,
            tokens: TokenUsage::default(),
            mcp_servers: Vec::new(),
            models: Vec::new(),
            models_loading: false,
            multi_agent: false,
            saw_delegation: false,
            context_recall: true,
            verify_command: None,
            prompt_caching: true,
            squad: Vec::new(),
            agents: Vec::new(),
            todos: Vec::new(),
            // YOLO by default (user decision 2026-06-07): approvals
            // auto-allow; Shift+Tab / /mode switch live when more caution
            // is wanted. The status line always shows non-Normal modes.
            permission_mode: PermissionMode::Yolo,
            file_index: Vec::new(),
            files_requested: false,
            context_tokens: 0,
            last_ttft_ms: 0,
            running: false,
            focused: true,
            notify: true,
            pending_respawn: false,
            learning: None,
            follow: true,
            scroll_top: 0,
            last_max_off: std::cell::Cell::new(0),
            menu_selected: 0,
            spinner: 0,
            splash: None,
            workflow_recipes: Vec::new(),
            fast_model: None,
            frontier_model: None,
            workflow_journal_dir: std::path::PathBuf::new(),
            should_quit: false,
            effects: Vec::new(),
            tool_index: HashMap::new(),
            queued: std::collections::VecDeque::new(),
            md: crate::markdown::MarkdownCache::default(),
        }
    }

    /// The current model's context window (tokens), from the OpenRouter catalog
    /// if loaded — used to draw the context-fill bar. `None` falls back to a
    /// raw token count in the status line.
    pub fn context_limit(&self) -> Option<u64> {
        self.models
            .iter()
            .find(|m| m.id == self.model)
            .and_then(|m| m.context)
    }

    /// Finalize the streamed assistant text into a transcript cell.
    fn finalize_active(&mut self) {
        // Flush any streamed reasoning FIRST, so a `Cell::Reasoning` lands above
        // the answer/tool it preceded (covers the tool-loop: each LLM call's
        // thinking settles above its own output).
        if let Some(reasoning) = self.active_reasoning.take() {
            let trimmed = reasoning.trim();
            if !trimmed.is_empty() {
                self.history.push(Cell::Reasoning(trimmed.to_string()));
            }
        }
        if let Some(text) = self.active.take() {
            let trimmed = text.trim_end();
            if !trimmed.is_empty() {
                self.history.push(Cell::Agent(trimmed.to_string()));
            }
        }
    }

    /// Compute the renderer's vertical scroll offset (rows hidden above the
    /// viewport) for a given `max_off` (= total wrapped rows − viewport height),
    /// and cache `max_off` so the wheel handlers can anchor a fresh scroll. When
    /// `follow` (the default), the view tracks the bottom; once the user scrolls
    /// up it's TOP-anchored at `scroll_top`, so content growing at the bottom
    /// never drifts the read position. Called once per frame by `view()`.
    pub fn scroll_offset(&self, max_off: u16) -> u16 {
        self.last_max_off.set(max_off);
        if self.follow {
            max_off
        } else {
            self.scroll_top.min(max_off)
        }
    }

    /// Scroll the transcript up by `step` rows, un-pinning from the bottom. The
    /// first step off the bottom converts the current bottom offset into a
    /// top-anchored position (using the last rendered `max_off`).
    fn scroll_up(&mut self, step: u16) {
        if self.follow {
            self.follow = false;
            self.scroll_top = self.last_max_off.get().saturating_sub(step);
        } else {
            self.scroll_top = self.scroll_top.saturating_sub(step);
        }
    }

    /// Scroll the transcript down by `step` rows. Reaching the bottom re-pins to
    /// `follow` (so new output auto-scrolls again).
    fn scroll_down(&mut self, step: u16) {
        if self.follow {
            return;
        }
        let next = self.scroll_top.saturating_add(step);
        if next >= self.last_max_off.get() {
            self.follow = true;
        } else {
            self.scroll_top = next;
        }
    }

    /// Whether the multi-agent roster + tool-cell badges are live: the user
    /// opted in via `/agents`, OR a delegation happened this session (the
    /// option-C entry agent delegates dynamically). Single-agent chat that never
    /// delegates stays clean (no roster, no badges).
    fn roster_active(&self) -> bool {
        self.multi_agent || self.saw_delegation
    }

    /// Find or create an agent row (first-seen order), and set it Working with
    /// the given activity. Only tracked once the roster is active.
    fn agent_set_working(&mut self, name: &str, activity: impl Into<String>) {
        if !self.roster_active() {
            return;
        }
        let activity = activity.into();
        if let Some(row) = self.agents.iter_mut().find(|r| r.name == name) {
            // An available (Idle) agent becomes Working when it's dispatched or
            // emits a tool event. Don't resurrect a finished (Done/Failed) agent
            // from a late event.
            if matches!(row.state, AgentState::Idle | AgentState::Working) {
                row.state = AgentState::Working;
                row.activity = activity;
            }
        } else {
            self.agents.push(AgentRow {
                name: name.to_string(),
                state: AgentState::Working,
                activity,
                tokens: 0,
            });
        }
    }

    /// Mark an agent finished (Done/Failed) with its token cost.
    fn agent_finish(&mut self, name: &str, success: bool, tokens: u32) {
        if !self.roster_active() {
            return;
        }
        if let Some(row) = self.agents.iter_mut().find(|r| r.name == name) {
            row.state = if success {
                AgentState::Done
            } else {
                AgentState::Failed
            };
            row.activity = if success { "done" } else { "failed" }.into();
            row.tokens = row.tokens.max(tokens);
        }
    }

    /// The agent badge to stamp on a tool cell: the agent name once the roster is
    /// active (so the transcript shows who ran each tool), `None` for clean
    /// single-agent chat that hasn't delegated.
    fn agent_badge(&self, agent: &str) -> Option<String> {
        if self.roster_active() {
            Some(agent.to_string())
        } else {
            None
        }
    }

    /// Reset the roster for a new turn: seed the available squad as Idle so the
    /// whole pool is visible from the start (dispatched agents then flip to
    /// Working). In single-agent mode the squad is empty → roster stays empty.
    fn seed_idle_squad(&mut self) {
        self.agents = self
            .squad
            .iter()
            .map(|name| AgentRow {
                name: name.clone(),
                state: AgentState::Idle,
                activity: "available".into(),
                tokens: 0,
            })
            .collect();
    }

    /// Mark every still-Working agent as Done (the turn produced a final answer).
    fn agents_settle(&mut self) {
        for row in &mut self.agents {
            if row.state == AgentState::Working {
                row.state = AgentState::Done;
                row.activity = "done".into();
            }
        }
    }

    /// Apply a message, mutating state and queuing effects.
    pub fn update(&mut self, msg: Msg) {
        match msg {
            Msg::Tick => {
                self.spinner = self.spinner.wrapping_add(1);
                if let Some(t) = self.splash {
                    let t = t.saturating_add(1);
                    self.splash = (t < crate::splash::SPLASH_TICKS).then_some(t);
                }
            }
            // Deliberately NOT touching `self.md` here (or anywhere in
            // `update`/its helpers — the reducer stays pure, no cache reads
            // or writes). A resize cannot serve stale content from the
            // Markdown cache: entries hold LOGICAL lines, and wrapping to the
            // terminal width happens at draw time in `ui::view` via
            // `Paragraph::wrap` against the live `transcript_area.width` —
            // nothing width-derived is ever stored under the cache's key
            // (see `MarkdownCache`'s doc comment).
            Msg::Resize => {}
            // Mouse wheel scrolls the transcript (output history). Over-scrolling
            // is harmless — the renderer clamps the offset to the top.
            Msg::WheelUp => self.scroll_up(WHEEL_STEP),
            Msg::WheelDown => self.scroll_down(WHEEL_STEP),
            Msg::Paste(s) => {
                // A paste during the splash dismisses the overlay too — unlike a
                // key (which the splash consumes outright), the paste already
                // carries content the user wants landed, so fall through to the
                // existing insert below instead of dropping it.
                if self.splash.is_some() {
                    self.splash = None;
                }
                match &mut self.modal {
                    // Pasting into a prompt must land in that field, not the
                    // composer hidden behind the modal.
                    Some(Modal::KeyEntry(m)) => m.input.push_str(&s.replace(['\n', '\r'], "")),
                    Some(Modal::ModelPicker(p)) => {
                        p.query.push_str(&s.replace(['\n', '\r'], ""));
                        p.selected = 0;
                    }
                    Some(Modal::HistorySearch(h)) => {
                        h.query.push_str(&s.replace(['\n', '\r'], ""));
                        h.sel = 0;
                    }
                    Some(Modal::Approval(_))
                    | Some(Modal::Question(_))
                    | Some(Modal::SessionPicker(_))
                    | Some(Modal::HandoffPicker { .. })
                    | Some(Modal::ModePicker { .. })
                    | Some(Modal::EffortPicker { .. }) => {}
                    None => self.composer.insert_str(&s),
                }
            }
            Msg::FocusChanged(focused) => self.focused = focused,
            Msg::Key(key) => {
                // Any key dismisses the splash and is CONSUMED — an impatient
                // first keypress must not leak a stray char into the composer
                // (nor reach a modal hidden beneath the overlay).
                if self.splash.is_some() {
                    self.splash = None;
                    let _ = key;
                    return;
                }
                if self.modal.is_some() {
                    self.handle_modal_key(key);
                } else {
                    self.handle_key(key);
                }
            }

            Msg::TurnStarted => self.running = true,
            Msg::StreamDelta(s) => {
                self.running = true;
                // No manual scroll reset: when `follow` is set the view is already
                // pinned to the bottom; when the user scrolled up we must NOT yank.
                self.active.get_or_insert_with(String::new).push_str(&s);
            }
            // Live chain-of-thought (reasoning models): append to the in-progress
            // reasoning buffer, rendered dimmed above the answer as it streams.
            Msg::ReasoningDelta(s) => {
                self.running = true;
                self.active_reasoning
                    .get_or_insert_with(String::new)
                    .push_str(&s);
            }
            Msg::LlmDone {
                usage,
                had_tool_calls,
                ttft_ms,
            } => {
                let was_running = self.running;
                self.finalize_active();
                self.tokens.input_tokens =
                    self.tokens.input_tokens.saturating_add(usage.input_tokens);
                self.tokens.output_tokens = self
                    .tokens
                    .output_tokens
                    .saturating_add(usage.output_tokens);
                // Cache-hit metric (context observability): accumulate prompt-cache
                // reads so the status line shows how much of the prompt was served
                // from cache rather than re-billed as fresh input.
                self.tokens.cache_read_input_tokens = self
                    .tokens
                    .cache_read_input_tokens
                    .saturating_add(usage.cache_read_input_tokens);
                // The latest request's input tokens ≈ current context fill (the
                // whole conversation is the prompt), not summed — for the bar.
                if usage.input_tokens > 0 {
                    self.context_tokens = usage.input_tokens;
                }
                if ttft_ms > 0 {
                    self.last_ttft_ms = ttft_ms;
                }
                // A text-only turn means the agent now idles awaiting input — the
                // orchestrator finished synthesizing, so settle the roster.
                if !had_tool_calls {
                    self.running = false;
                    self.agents_settle();
                    // /learn turn finished → commit the staged lessons (digest-guarded).
                    if let Some(digest) = self.learning.take() {
                        self.effects.push(Effect::CommitLessons(digest));
                    }
                    self.flush_pending_respawn();
                    // Turn boundary: release at most one queued message. If
                    // one was waiting, `running` goes back to `true`.
                    self.drain_one_queued();
                    self.notify_turn_idle(was_running, "Heartbit", "turn complete");
                }
            }
            // A sub-agent's LLM call: its cost is real (session totals) but its
            // lifecycle is NOT the run's — running/roster/learning/context bar
            // stay untouched (the unattributed leak flipped running=false
            // mid-run and settled still-working siblings — audit 2026-06-09).
            Msg::SubAgentLlmDone { usage } => {
                self.tokens.input_tokens =
                    self.tokens.input_tokens.saturating_add(usage.input_tokens);
                self.tokens.output_tokens = self
                    .tokens
                    .output_tokens
                    .saturating_add(usage.output_tokens);
                self.tokens.cache_read_input_tokens = self
                    .tokens
                    .cache_read_input_tokens
                    .saturating_add(usage.cache_read_input_tokens);
            }
            Msg::ToolStarted {
                id,
                name,
                input,
                agent,
            } => {
                self.finalize_active(); // the assistant preamble (if any) is done
                self.agent_set_working(&agent, &name);
                // Mirror the agent's task list into the live todo panel.
                if name == "todowrite" {
                    self.todos = parse_todos(&input);
                }
                let idx = self.history.len();
                self.tool_index.insert(id, idx);
                self.history.push(Cell::Tool {
                    name,
                    input,
                    status: ToolStatus::Running,
                    output: None,
                    duration_ms: None,
                    agent: self.agent_badge(&agent),
                });
            }
            Msg::AgentsDispatched(names) => {
                // A delegation activates the roster + badges for the rest of the
                // session, even without the manual /agents toggle.
                if !names.is_empty() {
                    self.saw_delegation = true;
                }
                for n in &names {
                    self.agent_set_working(n, "dispatched");
                }
                if !names.is_empty() {
                    self.history.push(Cell::Notice(format!(
                        "→ delegating to {}",
                        format_dispatch_names(&names)
                    )));
                }
            }
            Msg::SubAgentDone {
                agent,
                success,
                tokens,
                error,
            } => {
                self.agent_finish(&agent, success, tokens);
                // A sub-agent failure stays non-fatal (the orchestrator gets the
                // error as a tool result and continues), but it must be VISIBLE —
                // a silent "Max turns exceeded" let a half-done audit pass for a
                // finished one (live trace 6a2e92e3).
                if !success {
                    let reason = error.unwrap_or_else(|| "failed".into());
                    self.history.push(Cell::Notice(format!(
                        "⚠ sub-agent {agent} failed: {reason} — its result is partial; \
                         the orchestrator continues with what it returned."
                    )));
                }
            }
            Msg::AgentSpawned { name, task } => {
                self.agent_set_working(&name, "spawned");
                self.history.push(Cell::Notice(format!(
                    "✦ spawned {name}: {}",
                    first_words(&task, 60)
                )));
            }
            Msg::ToolCompleted {
                id,
                is_error,
                output,
                duration_ms,
            } => {
                if let Some(&idx) = self.tool_index.get(&id)
                    && let Some(Cell::Tool {
                        status,
                        output: out,
                        duration_ms: dur,
                        ..
                    }) = self.history.get_mut(idx)
                {
                    *status = if is_error {
                        ToolStatus::Failed
                    } else {
                        ToolStatus::Ok
                    };
                    *out = Some(output);
                    *dur = Some(duration_ms);
                }
                self.tool_index.remove(&id);
            }
            Msg::Notice(text) => self.history.push(Cell::Notice(text)),
            Msg::RunCompleted => {
                let was_running = self.running;
                self.finalize_active();
                self.running = false;
                // Backstop: a session ending right at the answer still commits the
                // staged lessons (digest-guarded at the edge).
                if let Some(digest) = self.learning.take() {
                    self.effects.push(Effect::CommitLessons(digest));
                }
                self.flush_pending_respawn();
                // Turn boundary: release at most one queued message. If one
                // was waiting, `running` goes back to `true`.
                self.drain_one_queued();
                self.notify_turn_idle(was_running, "Heartbit", "turn complete");
            }
            Msg::AgentExited(_) => {
                let was_running = self.running;
                self.finalize_active();
                self.running = false;
                // An abnormal/early thread exit must NOT commit a half-rewritten
                // file — and clearing here prevents a stale flag leaking into the
                // next run's first text-only LlmDone.
                self.learning = None;
                self.flush_pending_respawn();
                // Abnormal end: there is no live turn to release into — drop
                // the backlog rather than stranding it silently.
                self.drop_queued();
                self.notify_turn_idle(was_running, "Heartbit", "session ended");
            }
            Msg::RunFailed(error) => {
                let was_running = self.running;
                self.finalize_active();
                self.running = false;
                self.learning = None;
                self.flush_pending_respawn();
                // Abnormal end: same as AgentExited — drop, don't drain.
                self.drop_queued();
                self.history
                    .push(Cell::Notice(format!("run failed: {error}")));
                // Captured before `error` moves into EmergencyHandoff below;
                // sanitized later at the `notify::emit` I/O boundary, not here
                // (a provider error string is agent/provider-controlled text).
                self.notify_turn_idle(
                    was_running,
                    "Heartbit — run failed",
                    first_words(&error, 80),
                );
                // P12: a run failure is terminal for this engine — leave a
                // deterministic emergency brief so the next session can
                // continue deliberately (the 402 incident left an amnesiac
                // respawn with no bridge).
                self.effects.push(Effect::EmergencyHandoff(error));
                // Failed sessions are exactly what /learn distills best.
                self.history.push(Cell::Notice(
                    "tip: /learn can distill lessons from this failure".into(),
                ));
            }
            Msg::Approval { tools, reply } => {
                // Not a `running` transition (the turn stays in flight while the
                // user decides) — gate directly on `running` instead of a
                // was_running→!running edge. Tool names are agent-controlled
                // text; sanitized later at the `notify::emit` I/O boundary.
                if self.running
                    && self.notify
                    && !self.focused
                    && self.splash.is_none()
                    && !tools.is_empty()
                {
                    let names = tools
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.effects.push(Effect::Notify {
                        title: "Heartbit — approval needed".into(),
                        body: names,
                    });
                }
                self.modal = Some(Modal::Approval(ApprovalModal { tools, reply }));
            }
            Msg::Question { request, reply } => {
                let n_opts = request
                    .questions
                    .first()
                    .map(|q| q.options.len())
                    .unwrap_or(0);
                self.modal = Some(Modal::Question(QuestionModal {
                    request,
                    reply: Some(reply),
                    current: 0,
                    selected: 0,
                    picked: vec![false; n_opts],
                    answers: Vec::new(),
                }));
            }
            Msg::ModelsLoaded(models) => {
                self.models = models;
                self.models_loading = false;
            }
            Msg::FilesLoaded(files) => self.file_index = files,
            Msg::HandoffsListed(briefs) => {
                if briefs.is_empty() {
                    self.history.push(Cell::Notice(
                        "no handoff briefs yet — `/handoff <purpose>` creates one".into(),
                    ));
                } else {
                    self.modal = Some(Modal::HandoffPicker { briefs, sel: 0 });
                }
            }
            Msg::SessionsListed(sessions) => {
                if sessions.is_empty() {
                    self.history
                        .push(Cell::Notice("no saved sessions yet".into()));
                } else {
                    self.modal = Some(Modal::SessionPicker(SessionPicker { sessions, sel: 0 }));
                }
            }
            Msg::SessionLoaded(history) => {
                self.history = history;
                // The transcript was replaced wholesale: drop stale in-flight
                // tool ids (a late completion would overwrite a resumed cell)
                // and any half-streamed buffers from the previous transcript.
                self.tool_index.clear();
                self.active = None;
                self.active_reasoning = None;
                self.follow = true;
                self.history
                    .push(Cell::Notice("— session resumed —".into()));
            }
            Msg::StatsReady(Ok((label, stats))) => {
                self.history.push(Cell::Stats { label, stats });
            }
            Msg::StatsReady(Err(e)) => {
                self.history.push(Cell::Notice(format!("stats: {e}")));
            }
            Msg::GitDiffReady(Ok(text)) if text.trim().is_empty() => {
                self.history
                    .push(Cell::Notice("no changes in the working tree".into()));
            }
            Msg::GitDiffReady(Ok(text)) => {
                self.history.push(Cell::Diff {
                    lines: crate::gitdiff::parse(&text),
                });
            }
            Msg::GitDiffReady(Err(e)) => {
                self.history
                    .push(Cell::Notice(format!("git diff failed: {e}")));
            }
            Msg::AnalyzeReady { display, task } => {
                // The async prep (trace fetch + prompt build) may resolve
                // while a DIFFERENT turn is still in flight — queue rather
                // than bypass straight into the invisible channel. Queuing
                // `display` alongside `task` also means a mid-turn /analyze
                // shows the friendly label, not the raw tool instruction,
                // once it's later drained.
                let was_idle = !self.running;
                self.send_or_queue(display.clone(), task);
                if was_idle {
                    self.history.push(Cell::User(display));
                    self.running = true;
                    self.follow = true;
                    self.seed_idle_squad();
                }
            }
            Msg::AnalyzeFailed(e) => {
                self.history.push(Cell::Notice(format!("analyze: {e}")));
            }
            Msg::LearnReady {
                display,
                task,
                staged_digest,
            } => {
                // Stage the digest now so the next CommitLessons can skip the
                // commit if the staged-lessons file is still exactly what it
                // was at stage time (digest match → no-op, nothing rewrote
                // it). This is a no-change guard, not a turn-affinity fix:
                // `self.learning` is a single slot, so if a turn is already
                // in flight when /learn runs, THAT turn's idle (not /learn's)
                // drains this slot first — /learn's own commit can then fire
                // at the wrong boundary or be skipped entirely. Known
                // limitation, tracked as a follow-up; not fixed here.
                self.learning = Some(staged_digest);
                let was_idle = !self.running;
                self.send_or_queue(display.clone(), task);
                if was_idle {
                    self.history.push(Cell::User(display));
                    self.running = true;
                    self.follow = true;
                    self.seed_idle_squad();
                }
            }
            Msg::LearnFailed(e) => {
                self.history.push(Cell::Notice(format!("learn: {e}")));
            }
            Msg::ModelsFailed(err) => {
                self.models_loading = false;
                // Only notify when the fetch was USER-initiated (the picker is
                // open). The eager startup fetch fails SILENTLY — the user never
                // asked for it, and the context bar already falls back to a raw
                // token count without the catalog.
                if matches!(self.modal, Some(Modal::ModelPicker(_))) {
                    self.modal = None;
                    self.history.push(Cell::Notice(format!(
                        "could not load models: {err} — use /model <name>"
                    )));
                }
            }
        }
    }

    /// Candidates for the `/` autocomplete menu given the current composer text,
    /// or empty when the menu should not show (not typing a bare `/command`).
    pub fn command_candidates(&self) -> Vec<(&'static str, &'static str)> {
        if self.modal.is_some() {
            return Vec::new();
        }
        let text = self.composer.text();
        if !text.starts_with('/') || text.contains(char::is_whitespace) {
            return Vec::new();
        }
        SLASH_COMMANDS
            .iter()
            .filter(|(name, _)| name.starts_with(text.as_str()))
            .copied()
            .collect()
    }

    /// File-path candidates for `@`-mention autocomplete given the `@token` at
    /// the cursor, filtered from the project file index (empty if no `@` or the
    /// index isn't loaded yet).
    pub fn mention_candidates(&self) -> Vec<String> {
        let Some(prefix) = self.composer.mention_prefix() else {
            return Vec::new();
        };
        if self.modal.is_some() {
            return Vec::new();
        }
        let p = prefix.to_lowercase();
        self.file_index
            .iter()
            .filter(|f| f.to_lowercase().contains(&p))
            .take(20)
            .cloned()
            .collect()
    }

    /// Whether a completion menu (commands OR `@`-mentions) is currently showing.
    pub fn menu_open(&self) -> bool {
        !self.command_candidates().is_empty() || !self.mention_candidates().is_empty()
    }

    /// True when the active completion menu is `@`-mentions (vs slash commands).
    fn menu_is_mentions(&self) -> bool {
        self.command_candidates().is_empty() && !self.mention_candidates().is_empty()
    }

    /// Number of items in the active completion menu.
    fn menu_len(&self) -> usize {
        let c = self.command_candidates().len();
        if c > 0 {
            c
        } else {
            self.mention_candidates().len()
        }
    }

    fn menu_selected_command(&self) -> Option<&'static str> {
        let cands = self.command_candidates();
        if cands.is_empty() {
            return None;
        }
        Some(cands[self.menu_selected.min(cands.len() - 1)].0)
    }

    /// Move the menu highlight (wrapping).
    fn menu_move(&mut self, delta: isize) {
        let n = self.menu_len();
        if n == 0 {
            return;
        }
        let cur = self.menu_selected.min(n - 1) as isize;
        self.menu_selected = (cur + delta).rem_euclid(n as isize) as usize;
    }

    /// Tab: complete the selected item. Commands → `/cmd ` (ready for args);
    /// `@`-mentions → the file path + a space.
    fn menu_complete(&mut self) {
        if self.menu_is_mentions() {
            let cands = self.mention_candidates();
            if let Some(path) = cands.get(self.menu_selected.min(cands.len().saturating_sub(1))) {
                self.composer.complete_mention(path);
                self.menu_selected = 0;
            }
        } else if let Some(name) = self.menu_selected_command() {
            self.composer.set_text(&format!("{name} "));
            self.menu_selected = 0;
        }
    }

    /// Enter on the menu: complete an `@`-mention, or run the selected command.
    fn menu_run(&mut self) {
        if self.menu_is_mentions() {
            self.menu_complete();
        } else if let Some(name) = self.menu_selected_command() {
            self.composer.set_text(name);
            self.submit();
            self.menu_selected = 0;
        }
    }

    fn submit(&mut self) {
        let text = self.composer.text();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            self.composer.clear();
            return;
        }
        // Slash commands are handled locally and never recorded in input history
        // (so a `/key <token>` secret can't be recalled with the Up arrow).
        if let Some(cmd) = trimmed.strip_prefix('/') {
            self.handle_slash(cmd.to_string());
            self.composer.clear();
            return;
        }
        // No provider configured yet → ask for the key, keeping the message.
        if self.api_key.is_none() && !self.has_fallback_provider {
            self.open_key_modal();
            return;
        }
        let text = self.composer.take();
        // Persist the raw prompt to the per-directory history (slash commands
        // never reach here — they returned above, so secrets stay unrecallable).
        // This is independent of whether the turn is sent now or queued — the
        // recall history cares about what was TYPED, not when it was SENT.
        self.effects.push(Effect::PersistPrompt(text.clone()));
        // Plan mode: prefix a read-only directive so the agent PRODUCES A PLAN
        // instead of attempting edits and getting silently denied by the gate.
        // Per-turn (reflects the mode at send time); the display stays clean.
        let sent = if self.permission_mode == PermissionMode::Plan {
            format!(
                "[PLAN MODE — you are READ-ONLY this turn. Investigate with read/search tools, \
                 then present a concise, numbered PLAN of what you WOULD do. Do NOT edit/write/\
                 patch files or run mutating commands — they will be blocked. Ask me to switch to \
                 normal/YOLO mode to execute.]\n\n{text}"
            )
        } else {
            text.clone()
        };
        // A turn already in flight: `send_or_queue` below holds this one
        // visibly instead of pushing straight into the invisible input
        // channel. The start-of-turn bookkeeping (display cell, roster reset)
        // only applies to a genuinely fresh turn — repeating it mid-turn would
        // reset the LIVE roster back to Idle and double-display the message
        // once it's eventually drained.
        let was_idle = !self.running;
        self.send_or_queue(text.clone(), sent);
        if was_idle {
            self.history.push(Cell::User(text)); // display the user's text verbatim
            self.running = true;
            self.follow = true; // jump back to the newest when the user sends
            self.seed_idle_squad(); // fresh roster: the whole squad, available
        }
    }

    /// The single choke point for every user-visible send: `display` is what
    /// the human sees (queue box, transcript, Up-recall); `wire` is the exact
    /// payload that becomes `Effect::SendInput` and may carry an invisible
    /// directive `display` must not (e.g. Plan mode's read-only prefix). Six
    /// call sites reach this in addition to `submit` above (AnalyzeReady,
    /// LearnReady, `/goal` clear, `/goal` set, `/handoff`, `/research`) —
    /// routing them all through here is what keeps the queue honest instead
    /// of leaving mid-turn bypasses straight into the invisible input channel.
    fn send_or_queue(&mut self, display: String, wire: String) {
        if self.running {
            self.queued.push_back(QueuedInput { display, wire });
        } else {
            self.effects.push(Effect::SendInput(wire));
        }
    }

    /// Release at most ONE queued message at a turn boundary. Releasing
    /// several would push the rest back into the invisible channel — the
    /// very defect this queue exists to fix.
    fn drain_one_queued(&mut self) {
        if let Some(q) = self.queued.pop_front() {
            self.history.push(Cell::User(q.display));
            self.running = true; // the drained message starts a fresh turn
            // Dropping `follow` on a mid-turn SUBMIT is fine (the user was
            // scrolled up reading); but a DRAIN starts a brand-new turn whose
            // reply should be visible — re-arm it here, or the new answer
            // streams off-screen while the view stays wherever it was left.
            self.follow = true;
            self.effects.push(Effect::SendInput(q.wire));
        }
    }

    /// Drop the entire backlog with a recoverable notice — used whenever a
    /// turn ends WITHOUT a next turn to release into (failure, unexpected
    /// exit, user interrupt), or when the user explicitly cancels the
    /// backlog with Esc. The text is gone, but the notice makes the drop
    /// visible instead of silently stranding the messages.
    fn drop_queued(&mut self) {
        let n = self.queued.len();
        if n == 0 {
            return;
        }
        self.queued.clear();
        self.history.push(Cell::Notice(format!(
            "{n} queued message{} dropped — retype if still needed",
            if n == 1 { "" } else { "s" }
        )));
    }

    /// Push `Effect::Notify` for a turn-idle site, gated on: notifications
    /// enabled, terminal unfocused, no splash overlay, AND the turn genuinely
    /// just ended — `was_running` (captured before this message's mutations)
    /// was `true` and `running` is now `false`. That transition is the
    /// dedupe: if an earlier message in the same turn boundary (e.g.
    /// `LlmDone{false}`) already flipped `running` to `false`, a later one
    /// (e.g. a stale `RunCompleted`) sees `was_running == false` here and
    /// stays silent — at most one notification fires per turn. It is also
    /// what suppresses notification when a queued message was drained right
    /// back into a fresh turn (`running` returns to `true` before this call).
    fn notify_turn_idle(&mut self, was_running: bool, title: &str, body: impl Into<String>) {
        if was_running && !self.running && self.notify && !self.focused && self.splash.is_none() {
            self.effects.push(Effect::Notify {
                title: title.to_string(),
                body: body.into(),
            });
        }
    }

    fn handle_slash(&mut self, cmd: String) {
        let mut parts = cmd.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("");
        let arg = parts
            .next()
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        match name {
            "key" | "login" => {
                if arg.is_empty() {
                    self.open_key_modal();
                } else {
                    self.set_api_key(arg);
                }
            }
            "model" | "models" => {
                if arg.is_empty() {
                    self.open_model_picker(ModelTarget::Main);
                } else if arg == "advisor" {
                    self.open_model_picker(ModelTarget::Advisor);
                } else if let Some(rest) = arg.strip_prefix("advisor ") {
                    let rest = rest.trim();
                    if rest.eq_ignore_ascii_case("clear") || rest.eq_ignore_ascii_case("off") {
                        self.clear_frontier_model();
                    } else {
                        self.set_frontier_model(rest.to_string());
                    }
                } else {
                    self.set_model(arg);
                }
            }
            "effort" => {
                if arg.is_empty() {
                    self.open_effort_picker();
                } else if let Some(level) = EffortLevel::parse(&arg) {
                    self.set_effort(level);
                } else {
                    self.history
                        .push(Cell::Notice("usage: /effort off|low|medium|high".into()));
                }
            }
            "codex" => self.activate_codex(arg),
            "mcp" => self.handle_mcp(arg),
            "mode" => self.set_mode(arg),
            "agents" | "agent" | "workflow" => self.toggle_multi_agent(arg),
            "context-recall" | "recall" => self.toggle_context_recall(arg),
            "verify" => self.set_verify_command(arg),
            "clear" | "new" => {
                self.history.clear();
                // Stale in-flight ids would point into the REPLACED transcript —
                // a late ToolCompleted must not overwrite a regrown cell.
                self.tool_index.clear();
                // `/clear` is reachable mid-stream (submit() dispatches slashes
                // unconditionally, the engine keeps running). A surviving
                // half-streamed buffer would keep growing via StreamDelta and
                // get flushed as a ghost Cell into the just-cleared transcript —
                // drop it, exactly as the SessionLoaded path does.
                self.active = None;
                self.active_reasoning = None;
                self.todos.clear();
                self.agents.clear();
                self.follow = true;
                self.history
                    .push(Cell::Notice("— transcript cleared —".into()));
            }
            "export" => self.effects.push(Effect::ExportSession),
            "resume" => self.effects.push(Effect::ListSessions),
            "diff" => self.effects.push(Effect::GitDiff),
            "goal" => {
                if arg.is_empty() {
                    self.history.push(Cell::Notice(
                        "usage: /goal <objective> (the judge gates completion until met) \
                         · /goal clear"
                            .into(),
                    ));
                } else if arg.eq_ignore_ascii_case("clear") {
                    // Display is the slash command itself — these instructions
                    // were never shown verbatim even on an immediate send, so
                    // a mid-turn queue+drain gets a friendly label instead of
                    // reusing the raw tool-call text.
                    self.send_or_queue(
                        "/goal clear".to_string(),
                        "Call the `set_goal` tool with clear=true (remove the completion goal)."
                            .to_string(),
                    );
                } else {
                    self.send_or_queue(
                        format!("/goal {arg}"),
                        format!(
                            "Call the `set_goal` tool now with this objective, then keep working \
                             toward it: \"{arg}\""
                        ),
                    );
                }
            }
            "handoff" => {
                if arg.is_empty() {
                    // Bare: browse saved briefs (a purposeless handoff is
                    // invalid by design — the purpose tailors the brief).
                    self.effects.push(Effect::ListHandoffs);
                } else {
                    self.send_or_queue(
                        format!("/handoff {arg}"),
                        format!(
                            "Call the `handoff` tool now with purpose: \"{arg}\". Then tell me \
                             the brief's path in one line."
                        ),
                    );
                }
            }
            "stats" => {
                let target = if arg.is_empty() { None } else { Some(arg) };
                self.effects.push(Effect::ComputeStats(target));
            }
            "analyze" => {
                // Mirror submit()'s provider gate: `/analyze` starts a real run,
                // so without a key (and no fallback) prompt for one instead of
                // setting running=true against a no-op agent (spinner forever).
                if self.api_key.is_none() && !self.has_fallback_provider {
                    self.open_key_modal();
                    return;
                }
                let target = if arg.is_empty() { None } else { Some(arg) };
                self.effects.push(Effect::Analyze(target));
            }
            "learn" => {
                if self.api_key.is_none() && !self.has_fallback_provider {
                    self.open_key_modal();
                    return;
                }
                // Refuse mid-run / re-entrant learns: the in-flight turn's
                // text-only LlmDone would consume the digest EARLY and the
                // real learn result would be silently dropped.
                if self.running || self.learning.is_some() {
                    self.history
                        .push(Cell::Notice("finish the current turn before /learn".into()));
                    return;
                }
                self.effects.push(Effect::Learn);
            }
            "workflows" => {
                if self.workflow_recipes.is_empty() {
                    self.history
                        .push(Cell::Notice("no workflow recipes registered".into()));
                    return;
                }
                self.history.push(Cell::Notice(format!(
                    "workflow recipes ({}) — the agent runs them via the run_workflow tool \
                     when the task fits (or ask for one by name):",
                    self.workflow_recipes.len()
                )));
                for (name, desc) in &self.workflow_recipes {
                    self.history
                        .push(Cell::Notice(format!("  • {name} — {desc}")));
                }
            }
            "research" => {
                if arg.is_empty() {
                    self.history.push(Cell::Notice(
                        "usage: /research <question> — fan-out research, cross-verify, cited report"
                            .into(),
                    ));
                    return;
                }
                if self.api_key.is_none() && !self.has_fallback_provider {
                    self.open_key_modal();
                    return;
                }
                let slug = research_slug(&arg);
                let task = format!(
                    "Call the run_workflow tool now with recipe=\"deep_research\" and \
                     args={{\"question\": {q}}}. Do NOT search, browse, or implement \
                     anything yourself before the workflow returns. When it returns, \
                     write the report verbatim to research-{slug}.md (workspace-relative \
                     path) with the write tool, then give a 5-10 line summary of the key \
                     findings and sources. If the workflow returns an error, report it — \
                     do not improvise your own research.",
                    q = serde_json::to_string(&arg).unwrap_or_else(|_| format!("\"{arg}\"")),
                );
                // /research isn't guarded against mid-run reentry either — a
                // second /research while one is already going through must
                // queue, not bypass into the invisible channel.
                let display = format!("researching: {arg}");
                let was_idle = !self.running;
                self.send_or_queue(display.clone(), task);
                if was_idle {
                    self.history.push(Cell::User(display));
                    self.running = true;
                    self.follow = true;
                    self.seed_idle_squad();
                }
            }
            "help" => {
                self.history.push(Cell::Notice(
                    "commands: /mode [normal|plan|yolo] · /model [name] · /effort [off|low|medium|high] · \
                     /mcp [list|add …|clear] · /stats · /analyze · /learn · /research <question> · \
                     /verify <cmd> · /diff · /clear · /resume · /export · /key · /quit"
                        .into(),
                ));
                self.history.push(Cell::Notice(
                    "unified agent — answers directly, delegates to the worker/researcher squad, or \
                     runs a workflow as the task warrants · old tool outputs stay recoverable via \
                     fetch_full_output / recall_context"
                        .into(),
                ));
                self.history.push(Cell::Notice(
                    "the API key is stored 0600 at ~/.config/heartbit/tui.toml — \
                     readable by shell tools you approve"
                        .into(),
                ));
            }
            "quit" | "exit" => self.quit(),
            other => self
                .history
                .push(Cell::Notice(format!("unknown command: /{other}"))),
        }
    }

    /// `/mcp [list]` · `/mcp add <preset|command [args…]>` · `/mcp clear`.
    /// Changes are persisted and take effect when the agent (re)starts — the
    /// runner is built (with MCP tools) lazily on the first message.
    fn handle_mcp(&mut self, arg: String) {
        let arg = arg.trim();
        let mut parts = arg.splitn(2, char::is_whitespace);
        let sub = parts.next().unwrap_or("");
        let rest = parts.next().map(str::trim).unwrap_or("");
        match sub {
            "" | "list" => {
                if self.mcp_servers.is_empty() {
                    self.history.push(Cell::Notice(
                        "no MCP servers. Add one: /mcp add chrome-devtools  (or /mcp add <command> [args…])"
                            .into(),
                    ));
                } else {
                    self.history.push(Cell::Notice(format!(
                        "MCP servers ({}) — connect on next agent start:",
                        self.mcp_servers.len()
                    )));
                    for s in &self.mcp_servers {
                        self.history
                            .push(Cell::Notice(format!("  • {}", s.label())));
                    }
                }
            }
            "add" => {
                if rest.is_empty() {
                    self.history.push(Cell::Notice(
                        "usage: /mcp add <preset> | /mcp add <command> [args…]".into(),
                    ));
                    return;
                }
                let spec = parse_mcp_add(rest);
                let label = spec.label();
                self.mcp_servers.push(spec);
                self.effects.push(Effect::SaveMcp(self.mcp_servers.clone()));
                self.history.push(Cell::Notice(format!(
                    "MCP server added: {label} — connects on next agent start (send a message)"
                )));
            }
            "clear" | "remove" => {
                self.mcp_servers.clear();
                self.effects.push(Effect::SaveMcp(Vec::new()));
                self.history.push(Cell::Notice(
                    "MCP servers cleared (relaunch to apply)".into(),
                ));
            }
            other => {
                self.history.push(Cell::Notice(format!(
                    "unknown /mcp subcommand: {other} (try /mcp list | add | clear)"
                )));
            }
        }
    }

    fn open_key_modal(&mut self) {
        self.modal = Some(Modal::KeyEntry(KeyEntryModal::default()));
    }

    fn set_api_key(&mut self, key: String) {
        self.api_key = Some(key.clone());
        self.effects.push(Effect::SaveKey(key));
        self.history
            .push(Cell::Notice("OpenRouter API key saved.".into()));
    }

    /// Request the engine rebuild a model/advisor change needs: immediate when
    /// idle; DEFERRED to turn-idle when a turn is in flight (swapping the input
    /// channel under a live engine lets the next message spawn a second engine
    /// alongside it — audit 2026-06-09). Returns the notice suffix.
    fn queue_respawn(&mut self) -> &'static str {
        if self.running {
            self.pending_respawn = true;
            "applies when the current turn ends"
        } else {
            self.effects.push(Effect::RespawnAgent);
            "active on your next message"
        }
    }

    /// The turn just went idle: flush a deferred model/advisor respawn.
    fn flush_pending_respawn(&mut self) {
        if self.pending_respawn {
            self.pending_respawn = false;
            self.effects.push(Effect::RespawnAgent);
        }
    }

    /// Set the model (same semantics as `/model <name>`): update, persist, notice.
    /// Takes effect on the next agent start.
    fn set_model(&mut self, model: String) {
        self.model = model.clone();
        self.effects.push(Effect::SaveModel(model.clone()));
        // Rebuild the agent so the new model is actually used (the live agent
        // captured the old model at spawn — a persisted change alone never
        // reached it).
        let when = self.queue_respawn();
        self.history
            .push(Cell::Notice(format!("model set to {model} — {when}")));
    }

    /// Set the reasoning-effort level (same semantics as `/effort <level>`):
    /// update, persist, notice. `Off` clears the config key entirely (rather
    /// than persisting `"off"`) so a stale value never lingers. Takes effect
    /// on the next agent start (gated to OpenRouter/custom-endpoint providers
    /// in `main.rs::effort_for_provider` — never the Anthropic fallback).
    fn set_effort(&mut self, level: EffortLevel) {
        self.effort = level;
        let saved = (level != EffortLevel::Off).then(|| level.label().to_string());
        self.effects.push(Effect::SaveReasoningEffort(saved));
        let when = self.queue_respawn();
        self.history.push(Cell::Notice(format!(
            "reasoning effort set to {} — {when}",
            level.label()
        )));
    }

    /// Bare `/effort`: open the picker, preselected on the current level.
    fn open_effort_picker(&mut self) {
        let sel = EffortLevel::ALL
            .iter()
            .position(|l| *l == self.effort)
            .unwrap_or(0);
        self.modal = Some(Modal::EffortPicker { sel });
    }

    /// `/codex [url|off]` — one command to run the TUI on a ChatGPT-subscription
    /// Codex quota. It points the engine at a local Codex→OpenAI-compatible proxy
    /// (default `http://127.0.0.1:10531/v1`), switches the model to a Codex id, and
    /// respawns the agent. `off` reverts to the normal provider on the next start.
    ///
    /// ⚠ Using a ChatGPT-subscription token outside Codex is a likely Terms-of-
    /// Service violation (ban risk) and is fragile — see docs/chatgpt-subscription.md.
    /// The TUI only points at the proxy; it never touches the Codex token itself.
    fn activate_codex(&mut self, arg: String) {
        let arg = arg.trim();
        if matches!(arg, "off" | "clear" | "stop") {
            self.custom_endpoint = None;
            // Restore the pre-codex model + fallback (the override is session-only).
            if let Some((model, fallback)) = self.codex_saved.take() {
                self.model = model;
                self.has_fallback_provider = fallback;
            }
            // The model source changed back to OpenRouter — reload that catalog.
            self.refresh_model_catalog();
            let when = self.queue_respawn();
            self.history.push(Cell::Notice(format!(
                "codex endpoint cleared — reverting to your normal provider ({}), {when}",
                self.model
            )));
            return;
        }
        // The proxy's OpenAI-compatible base URL (arg overrides the default port).
        let url = if arg.is_empty() {
            "http://127.0.0.1:10531/v1".to_string()
        } else {
            arg.to_string()
        };
        // Stash the model + fallback to restore on `/codex off` — but only on the
        // FIRST activation, so re-running `/codex <url>` keeps the original values.
        if self.codex_saved.is_none() {
            self.codex_saved = Some((self.model.clone(), self.has_fallback_provider));
        }
        self.custom_endpoint = Some(url.clone());
        // A custom endpoint IS a provider — let a no-OpenRouter-key session start.
        self.has_fallback_provider = true;
        // Load the proxy's model list into the /model picker (replaces the stale
        // OpenRouter catalogue) so the user can pick any model the subscription
        // exposes through Codex.
        self.refresh_model_catalog();
        // The Codex backend exposes its own model set (discovered live, account/
        // version-dependent — e.g. gpt-5.5, gpt-5.4, gpt-5.4-mini). `gpt-5.5` is the
        // current flagship default; switch with `/model <id>` (check the proxy's
        // `/v1/models`). SESSION-ONLY: deliberately NOT persisted (no `SaveModel`) —
        // a model id written to config would brick the next launch when the proxy
        // is gone.
        self.model = "gpt-5.5".to_string();
        // Best-effort: warn (don't block) if the Codex login token is absent — the
        // proxy needs `~/.codex/auth.json` (run `codex login`). Advisory only, so
        // the reducer stays testable (it never depends on this file existing).
        let auth_missing = std::env::var("HOME")
            .ok()
            .map(|h| !std::path::Path::new(&h).join(".codex/auth.json").exists())
            .unwrap_or(false);
        let when = self.queue_respawn();
        let mut notice = format!(
            "codex endpoint → {url} · model gpt-5.5 — {when}. \
             (switch with /model <id> — see the proxy's /v1/models.) \
             ⚠ subscription-token use outside Codex risks a ToS ban (see \
             docs/chatgpt-subscription.md); start the local proxy first."
        );
        if auth_missing {
            notice.push_str(" Note: ~/.codex/auth.json not found — run `codex login`.");
        }
        self.history.push(Cell::Notice(notice));
    }

    /// Set the advisor's frontier model (`/model advisor <name>`): persist,
    /// notice. Takes effect on the next agent start (the advisor provider is
    /// built once at engine spawn).
    fn set_frontier_model(&mut self, model: String) {
        self.frontier_model = Some(model.clone());
        self.effects
            .push(Effect::SaveFrontierModel(Some(model.clone())));
        let when = self.queue_respawn();
        self.history.push(Cell::Notice(format!(
            "advisor model set to {model} — {when}"
        )));
    }

    /// Clear the advisor's frontier model (`/model advisor clear`): the
    /// advisor falls back to the main model on the next start.
    fn clear_frontier_model(&mut self) {
        self.frontier_model = None;
        self.effects.push(Effect::SaveFrontierModel(None));
        let when = self.queue_respawn();
        self.history.push(Cell::Notice(format!(
            "advisor model cleared — falls back to the main model ({when})"
        )));
    }

    /// `/mode [normal|plan|yolo]` — set the execution mode (same as Shift+Tab).
    /// Bare `/mode` reports the current one. Applied live to the approval gate.
    fn set_mode(&mut self, arg: String) {
        if arg.trim().is_empty() {
            // Bare `/mode`: open the picker, preselected on the current mode.
            let sel = MODES
                .iter()
                .position(|m| *m == self.permission_mode)
                .unwrap_or(0);
            self.modal = Some(Modal::ModePicker { sel });
            return;
        }
        match PermissionMode::parse(&arg) {
            Some(mode) => {
                self.permission_mode = mode;
                self.effects.push(Effect::SetPermissionMode(mode.as_u8()));
                self.history.push(Cell::Notice(format!(
                    "{} mode — {}",
                    mode.label(),
                    mode.describe()
                )));
            }
            None => {
                // Request-intent pins: `/mode study|clarify|execute|answer`
                // forces the response mode for every following request;
                // `/mode auto` resumes routing. Orthogonal to the approval
                // modes above (live effect, no respawn).
                let trimmed = arg.trim().to_lowercase();
                if trimmed == "auto" {
                    self.effects.push(Effect::SetRequestModePin(0));
                    self.history
                        .push(Cell::Notice("request mode: auto (router decides)".into()));
                } else if let Some(mode) =
                    heartbit_core::agent::router::RequestMode::parse(&trimmed)
                {
                    self.effects
                        .push(Effect::SetRequestModePin(mode.as_pin_u8()));
                    self.history.push(Cell::Notice(format!(
                        "request mode PINNED to {} — /mode auto to release",
                        mode.label()
                    )));
                } else {
                    self.history.push(Cell::Notice(format!(
                        "usage: /mode [normal|plan|yolo] · request mode: /mode \
                         [answer|study|clarify|execute|auto] (currently {})",
                        self.permission_mode.label()
                    )));
                }
            }
        }
    }

    /// `/agents` — informational. The static multi-agent mode was removed: the
    /// entry agent is now ALWAYS unified — it answers directly, does simple work
    /// itself, delegates to the squad, or runs a workflow, deciding per request.
    fn toggle_multi_agent(&mut self, _arg: String) {
        self.history.push(Cell::Notice(
            "the agent is always unified now — it answers directly, delegates to \
             the worker/researcher squad, or runs a workflow as the task warrants. \
             No mode to toggle."
                .into(),
        ));
    }

    /// Toggle context restore-on-demand (single-agent path). Persisted; applies on
    /// the next agent start.
    fn toggle_context_recall(&mut self, arg: String) {
        let new = match arg.trim().to_lowercase().as_str() {
            "on" | "true" | "1" => true,
            "off" | "false" | "0" => false,
            "" => !self.context_recall,
            other => {
                self.history.push(Cell::Notice(format!(
                    "usage: /context-recall [on|off] (currently {})",
                    if self.context_recall { "on" } else { "off" }
                )));
                let _ = other;
                return;
            }
        };
        self.context_recall = new;
        self.effects.push(Effect::SaveContextRecall(new));
        self.history.push(Cell::Notice(format!(
            "context restore-on-demand {} (applies on next start){}",
            if new { "ON" } else { "OFF" },
            if new {
                " — old tool outputs prune to a restorable marker (fetch_full_output / recall_context)"
            } else {
                ""
            }
        )));
    }

    /// Set (or clear with `off`/empty) the project verification command. When set,
    /// the agent gets a `verify` tool running it + a self-verify prompt nudge.
    /// Persisted; applies on the next agent start.
    fn set_verify_command(&mut self, arg: String) {
        let arg = arg.trim();
        let new = match arg {
            "" => {
                let cur = self.verify_command.as_deref().unwrap_or("(unset)");
                self.history.push(Cell::Notice(format!(
                    "usage: /verify <command>  ·  /verify off  (currently: {cur})"
                )));
                return;
            }
            "off" | "none" | "clear" => None,
            cmd => Some(cmd.to_string()),
        };
        self.verify_command = new.clone();
        self.effects.push(Effect::SaveVerifyCommand(new.clone()));
        self.history.push(Cell::Notice(match &new {
            Some(c) => format!(
                "verify command set to `{c}` (applies on next start) — the agent will self-verify after code changes"
            ),
            None => "verify command cleared (applies on next start)".into(),
        }));
    }

    /// Open the OpenRouter model picker for `target` (main model or advisor),
    /// fetching the catalog on first use.
    /// Force a reload of the `/model` picker catalogue from the CURRENT source —
    /// the custom endpoint's `/v1/models` when one is set (e.g. the Codex proxy),
    /// else the OpenRouter catalogue. Called when the source changes (`/codex`
    /// on/off) so the picker never shows the wrong provider's models.
    fn refresh_model_catalog(&mut self) {
        self.models.clear();
        self.models_loading = true;
        self.effects.push(Effect::FetchModels);
    }

    fn open_model_picker(&mut self, target: ModelTarget) {
        self.modal = Some(Modal::ModelPicker(ModelPicker {
            target,
            ..ModelPicker::default()
        }));
        if self.models.is_empty() && !self.models_loading {
            self.models_loading = true;
            self.effects.push(Effect::FetchModels);
        }
    }

    /// Keys for the model-picker modal: type to filter, ↑/↓ select, Enter set,
    /// Esc cancel. Enter may arrive as a raw CR/LF char on some terminals.
    fn handle_model_picker_key(&mut self, key: KeyEvent) {
        let query = match &self.modal {
            Some(Modal::ModelPicker(p)) => p.query.clone(),
            _ => return,
        };
        let filtered = crate::models::filter_models(&self.models, &query);
        let n = filtered.len();
        match key.code {
            KeyCode::Esc => self.modal = None,
            KeyCode::Up if n > 0 => {
                if let Some(Modal::ModelPicker(p)) = &mut self.modal {
                    p.selected = (p.selected.min(n - 1) + n - 1) % n;
                }
            }
            KeyCode::Down if n > 0 => {
                if let Some(Modal::ModelPicker(p)) = &mut self.modal {
                    p.selected = (p.selected.min(n - 1) + 1) % n;
                }
            }
            KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n') => {
                let (sel, target) = match &self.modal {
                    Some(Modal::ModelPicker(p)) => (p.selected, p.target),
                    _ => (0, ModelTarget::Main),
                };
                if let Some(&idx) = filtered.get(sel.min(n.saturating_sub(1))) {
                    let id = self.models[idx].id.clone();
                    self.modal = None;
                    match target {
                        ModelTarget::Main => self.set_model(id),
                        ModelTarget::Advisor => self.set_frontier_model(id),
                    }
                }
            }
            KeyCode::Backspace => {
                if let Some(Modal::ModelPicker(p)) = &mut self.modal {
                    p.query.pop();
                    p.selected = 0;
                }
            }
            KeyCode::Char(c) => {
                if let Some(Modal::ModelPicker(p)) = &mut self.modal {
                    p.query.push(c);
                    p.selected = 0;
                }
            }
            _ => {}
        }
    }

    fn quit(&mut self) {
        self.should_quit = true;
        self.effects.push(Effect::Quit);
    }

    fn handle_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        // Slash-command autocomplete: while the `/` menu is open, arrows/Tab/Enter/
        // Esc drive the menu instead of the composer/history.
        if self.menu_open() {
            match key.code {
                KeyCode::Up => return self.menu_move(-1),
                KeyCode::Down => return self.menu_move(1),
                KeyCode::Tab => return self.menu_complete(),
                // Run the highlighted command. Some terminals deliver Enter as a
                // raw CR/LF char rather than KeyCode::Enter — accept both so a
                // selection ALWAYS runs (else the char would just close the menu).
                KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n') => {
                    return self.menu_run();
                }
                KeyCode::Esc => return self.composer.clear(),
                _ => {}
            }
        }
        match key.code {
            KeyCode::Enter => {
                if shift || alt {
                    self.composer.newline();
                } else {
                    self.submit();
                }
            }
            KeyCode::Char('c') | KeyCode::Char('d') if ctrl => self.quit(),
            // Shift+Tab cycles the execution mode (Normal → Plan → YOLO),
            // applied live to the approval gate.
            KeyCode::BackTab => {
                self.permission_mode = self.permission_mode.next();
                self.effects
                    .push(Effect::SetPermissionMode(self.permission_mode.as_u8()));
                self.history.push(Cell::Notice(format!(
                    "{} mode — {}",
                    self.permission_mode.label(),
                    self.permission_mode.describe()
                )));
            }
            // Ctrl+U clears the DRAFT only — the recall history (seeded from
            // previous sessions in this directory) must survive.
            KeyCode::Char('u') if ctrl => self.composer.clear(),
            // Ctrl+R: reverse-search the submit history.
            KeyCode::Char('r') if ctrl => {
                self.modal = Some(Modal::HistorySearch(HistorySearch::default()));
            }
            KeyCode::Char(c) if !ctrl && !alt => {
                self.composer.insert_char(c);
                self.menu_selected = 0; // re-filter from the top
                // First `@`: lazily kick off the project file-index walk.
                if self.composer.mention_prefix().is_some() && !self.files_requested {
                    self.files_requested = true;
                    self.effects.push(Effect::WalkFiles);
                }
            }
            KeyCode::Backspace => {
                self.composer.backspace();
                self.menu_selected = 0;
            }
            KeyCode::Left => self.composer.move_left(),
            KeyCode::Right => self.composer.move_right(),
            // A non-empty queue changes what Up means: pop the newest queued
            // entry back for editing instead of recalling prompt history —
            // editing what's about to be sent takes priority over recall.
            KeyCode::Up => {
                if let Some(q) = self.queued.pop_back() {
                    // The CLEAN display text, never the wire payload — a
                    // Plan-mode queued message must not bring its invisible
                    // directive back into the composer.
                    self.composer.set_text(&q.display);
                } else {
                    self.composer.history_prev();
                }
            }
            KeyCode::Down => self.composer.history_next(),
            KeyCode::PageUp => self.scroll_up(SCROLL_STEP),
            KeyCode::PageDown => self.scroll_down(SCROLL_STEP),
            // A non-empty queue changes what Esc means too: drop the backlog
            // (visibly, via a notice) rather than interrupting the turn that
            // is actually still in flight. Only once the queue is empty does
            // Esc fall through to its running/idle meaning — so idle-Esc
            // (queue always empty then, by the invariant) is unchanged.
            KeyCode::Esc => {
                if !self.queued.is_empty() {
                    self.drop_queued();
                } else if self.running {
                    self.interrupt();
                } else {
                    self.composer.clear();
                }
            }
            _ => {}
        }
    }

    /// Abandon the in-flight turn: ask the agent to stop, finalize whatever was
    /// streamed, and return to idle. (Gated on `running` by the caller so a stray
    /// Esc while idle can't cancel the next, not-yet-started turn.)
    fn interrupt(&mut self) {
        self.learning = None; // Esc aborts a /learn — never commit a half-rewritten file
        self.effects.push(Effect::Interrupt);
        self.finalize_active();
        self.history.push(Cell::Notice("interrupted".into()));
        // An interrupt after a string of tool errors usually means the user
        // watched the agent struggle — suggest distilling the lesson.
        let errors = self
            .history
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    Cell::Tool {
                        status: crate::cells::ToolStatus::Failed,
                        ..
                    }
                )
            })
            .count();
        if errors >= 3 {
            self.history.push(Cell::Notice(format!(
                "{errors} tool errors this session — /learn can distill lessons from them"
            )));
        }
        self.running = false;
        // The interrupt ends the turn — a deferred model change applies now.
        self.flush_pending_respawn();
        // No live turn to release into — drop the backlog (defensive: the
        // Esc handler above already routes a non-empty queue away from here,
        // but keep the invariant locally true too).
        self.drop_queued();
    }

    fn handle_modal_key(&mut self, key: KeyEvent) {
        match self.modal {
            Some(Modal::Approval(_)) => self.handle_approval_key(key),
            Some(Modal::Question(_)) => self.handle_question_key(key),
            Some(Modal::KeyEntry(_)) => self.handle_key_entry(key),
            Some(Modal::ModelPicker(_)) => self.handle_model_picker_key(key),
            Some(Modal::HistorySearch(_)) => self.handle_history_search_key(key),
            Some(Modal::SessionPicker(_)) => self.handle_session_picker_key(key),
            Some(Modal::HandoffPicker { .. }) => self.handle_handoff_picker_key(key),
            Some(Modal::ModePicker { .. }) => self.handle_mode_picker_key(key),
            Some(Modal::EffortPicker { .. }) => self.handle_effort_picker_key(key),
            None => {}
        }
    }

    /// Question-modal keys: ↑/↓ highlight, Space toggles (multi-select),
    /// Enter confirms the current question (advances or sends the answers),
    /// Esc dismisses (drops the sender → the tool reports the dismissal).
    fn handle_question_key(&mut self, key: KeyEvent) {
        let Some(Modal::Question(m)) = &mut self.modal else {
            return;
        };
        let Some(q) = m.request.questions.get(m.current) else {
            self.modal = None;
            return;
        };
        let n = q.options.len();
        match key.code {
            KeyCode::Esc => {
                self.modal = None; // sender dropped → dismissal
                self.history
                    .push(Cell::Notice("question dismissed".to_string()));
            }
            KeyCode::Up if n > 0 => m.selected = (m.selected + n - 1) % n,
            KeyCode::Down if n > 0 => m.selected = (m.selected + 1) % n,
            KeyCode::Char(' ') if q.multiple => {
                if let Some(p) = m.picked.get_mut(m.selected) {
                    *p = !*p;
                }
            }
            KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n') => {
                // Confirm the current question: multi-select takes the toggled
                // set (falling back to the highlighted option when none was
                // toggled); single-select takes the highlighted option.
                let labels: Vec<String> = if q.multiple {
                    let toggled: Vec<String> = q
                        .options
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| m.picked.get(*i).copied().unwrap_or(false))
                        .map(|(_, o)| o.label.clone())
                        .collect();
                    if toggled.is_empty() {
                        q.options
                            .get(m.selected)
                            .map(|o| vec![o.label.clone()])
                            .unwrap_or_default()
                    } else {
                        toggled
                    }
                } else {
                    q.options
                        .get(m.selected)
                        .map(|o| vec![o.label.clone()])
                        .unwrap_or_default()
                };
                m.answers.push(labels);
                m.current += 1;
                m.selected = 0;
                let next_opts = m
                    .request
                    .questions
                    .get(m.current)
                    .map(|q| q.options.len())
                    .unwrap_or(0);
                m.picked = vec![false; next_opts];
                if m.current >= m.request.questions.len() {
                    // All answered: deliver and close.
                    let answers = std::mem::take(&mut m.answers);
                    let summary = answers
                        .iter()
                        .map(|a| a.join(", "))
                        .collect::<Vec<_>>()
                        .join(" · ");
                    if let Some(reply) = m.reply.take() {
                        let _ =
                            reply.send(heartbit_core::tool::builtins::QuestionResponse { answers });
                    }
                    self.modal = None;
                    self.history
                        .push(Cell::Notice(format!("answered: {summary}")));
                }
            }
            _ => {}
        }
    }

    /// `/mode` picker keys: ↑/↓ select (wrap), Enter apply, Esc cancel.
    fn handle_mode_picker_key(&mut self, key: KeyEvent) {
        let n = MODES.len();
        match key.code {
            KeyCode::Esc => self.modal = None,
            KeyCode::Up => {
                if let Some(Modal::ModePicker { sel }) = &mut self.modal {
                    *sel = (*sel + n - 1) % n;
                }
            }
            KeyCode::Down => {
                if let Some(Modal::ModePicker { sel }) = &mut self.modal {
                    *sel = (*sel + 1) % n;
                }
            }
            KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n') => {
                let mode = match &self.modal {
                    Some(Modal::ModePicker { sel }) => MODES.get(*sel).copied(),
                    _ => None,
                };
                self.modal = None;
                // Same application path as `/mode <arg>`.
                if let Some(mode) = mode {
                    self.permission_mode = mode;
                    self.effects.push(Effect::SetPermissionMode(mode.as_u8()));
                    self.history.push(Cell::Notice(format!(
                        "{} mode — {}",
                        mode.label(),
                        mode.describe()
                    )));
                }
            }
            _ => {}
        }
    }

    /// `/effort` picker keys: ↑/↓ select (wrap), Enter apply, Esc cancel.
    fn handle_effort_picker_key(&mut self, key: KeyEvent) {
        let n = EffortLevel::ALL.len();
        match key.code {
            KeyCode::Esc => self.modal = None,
            KeyCode::Up => {
                if let Some(Modal::EffortPicker { sel }) = &mut self.modal {
                    *sel = (*sel + n - 1) % n;
                }
            }
            KeyCode::Down => {
                if let Some(Modal::EffortPicker { sel }) = &mut self.modal {
                    *sel = (*sel + 1) % n;
                }
            }
            KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n') => {
                let level = match &self.modal {
                    Some(Modal::EffortPicker { sel }) => EffortLevel::ALL.get(*sel).copied(),
                    _ => None,
                };
                self.modal = None;
                // Same application path as `/effort <arg>`.
                if let Some(level) = level {
                    self.set_effort(level);
                }
            }
            _ => {}
        }
    }

    /// `/resume` picker keys: ↑/↓ select, Enter load, Esc cancel.
    fn handle_session_picker_key(&mut self, key: KeyEvent) {
        let n = match &self.modal {
            Some(Modal::SessionPicker(p)) => p.sessions.len(),
            _ => return,
        };
        match key.code {
            KeyCode::Esc => self.modal = None,
            KeyCode::Up if n > 0 => {
                if let Some(Modal::SessionPicker(p)) = &mut self.modal {
                    p.sel = (p.sel.min(n - 1) + n - 1) % n;
                }
            }
            KeyCode::Down if n > 0 => {
                if let Some(Modal::SessionPicker(p)) = &mut self.modal {
                    p.sel = (p.sel.min(n - 1) + 1) % n;
                }
            }
            KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n') => {
                let id = match &self.modal {
                    Some(Modal::SessionPicker(p)) => p
                        .sessions
                        .get(p.sel.min(n.saturating_sub(1)))
                        .map(|s| s.id.clone()),
                    _ => None,
                };
                self.modal = None;
                if let Some(id) = id {
                    self.effects.push(Effect::ResumeSession(id));
                }
            }
            _ => {}
        }
    }

    /// Handoff-brief picker keys: ↑/↓ select (wrap), Enter seed, Esc cancel.
    fn handle_handoff_picker_key(&mut self, key: KeyEvent) {
        let n = match &self.modal {
            Some(Modal::HandoffPicker { briefs, .. }) => briefs.len(),
            _ => return,
        };
        match key.code {
            KeyCode::Esc => self.modal = None,
            KeyCode::Up if n > 0 => {
                if let Some(Modal::HandoffPicker { sel, .. }) = &mut self.modal {
                    *sel = ((*sel).min(n - 1) + n - 1) % n;
                }
            }
            KeyCode::Down if n > 0 => {
                if let Some(Modal::HandoffPicker { sel, .. }) = &mut self.modal {
                    *sel = ((*sel).min(n - 1) + 1) % n;
                }
            }
            KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n') => {
                let path = match &self.modal {
                    Some(Modal::HandoffPicker { briefs, sel }) => briefs
                        .get((*sel).min(n.saturating_sub(1)))
                        .map(|b| b.path.clone()),
                    _ => None,
                };
                self.modal = None;
                if let Some(path) = path {
                    self.effects.push(Effect::SeedFromHandoff(path));
                }
            }
            _ => {}
        }
    }

    /// Submit-history entries containing `query` (case-insensitive), newest-first
    /// and de-duplicated — the reverse-search match list.
    pub fn history_matches(&self, query: &str) -> Vec<String> {
        let q = query.to_lowercase();
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for e in self.composer.history().iter().rev() {
            if (q.is_empty() || e.to_lowercase().contains(&q)) && seen.insert(e.clone()) {
                out.push(e.clone());
            }
        }
        out
    }

    /// Ctrl+R modal keys: type to filter, Ctrl+R cycles matches, Enter loads the
    /// selected prompt into the composer, Esc cancels.
    fn handle_history_search_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let query = match &self.modal {
            Some(Modal::HistorySearch(h)) => h.query.clone(),
            _ => return,
        };
        let matches = self.history_matches(&query);
        let n = matches.len();
        match key.code {
            KeyCode::Esc => self.modal = None,
            KeyCode::Char('r') if ctrl => {
                if let Some(Modal::HistorySearch(h)) = &mut self.modal
                    && n > 0
                {
                    h.sel = (h.sel + 1) % n;
                }
            }
            KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n') => {
                let sel = match &self.modal {
                    Some(Modal::HistorySearch(h)) => h.sel,
                    _ => 0,
                };
                if let Some(m) = matches.get(sel.min(n.saturating_sub(1))) {
                    self.composer.set_text(m);
                }
                self.modal = None;
            }
            KeyCode::Backspace => {
                if let Some(Modal::HistorySearch(h)) = &mut self.modal {
                    h.query.pop();
                    h.sel = 0;
                }
            }
            KeyCode::Char(c) if !ctrl => {
                if let Some(Modal::HistorySearch(h)) = &mut self.modal {
                    h.query.push(c);
                    h.sel = 0;
                }
            }
            _ => {}
        }
    }

    fn handle_approval_key(&mut self, key: KeyEvent) {
        let decision = match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                Some(ApprovalDecision::Allow)
            }
            KeyCode::Char('a') | KeyCode::Char('A') => Some(ApprovalDecision::AlwaysAllow),
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => Some(ApprovalDecision::Deny),
            KeyCode::Char('d') | KeyCode::Char('D') => Some(ApprovalDecision::AlwaysDeny),
            _ => None,
        };
        if let Some(decision) = decision
            && let Some(Modal::Approval(modal)) = self.modal.take()
        {
            // Best-effort: if the agent thread is gone, the decision is moot.
            let _ = modal.reply.send(decision);
        }
    }

    fn handle_key_entry(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Char(c) if !ctrl && !alt => {
                if let Some(Modal::KeyEntry(m)) = self.modal.as_mut() {
                    m.input.push(c);
                }
            }
            KeyCode::Backspace => {
                if let Some(Modal::KeyEntry(m)) = self.modal.as_mut() {
                    m.input.pop();
                }
            }
            KeyCode::Enter => {
                if let Some(Modal::KeyEntry(m)) = self.modal.take() {
                    let key = m.input.trim().to_string();
                    if !key.is_empty() {
                        self.set_api_key(key);
                    }
                }
            }
            KeyCode::Esc => self.modal = None,
            _ => {}
        }
    }
}

/// First `max` chars of the first line of `s`, ellipsized — for compact summaries.
fn first_words(s: &str, max: usize) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    if line.chars().count() > max {
        let t: String = line.chars().take(max).collect();
        format!("{t}…")
    } else {
        line.to_string()
    }
}

/// Format a dispatch roster for the "→ delegating to …" notice, collapsing
/// duplicate agent names with a count (first-seen order). A `delegate_task` with
/// 4 tasks all named "worker" reads "worker ×4", not "worker, worker, worker,
/// worker" (which looks like a bug) — live finding 6a25eb4d.
fn format_dispatch_names(names: &[String]) -> String {
    let mut order: Vec<&str> = Vec::new();
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for n in names {
        if !counts.contains_key(n.as_str()) {
            order.push(n.as_str());
        }
        *counts.entry(n.as_str()).or_insert(0) += 1;
    }
    order
        .iter()
        .map(|name| {
            let c = counts[name];
            if c > 1 {
                format!("{name} ×{c}")
            } else {
                (*name).to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parse a `/mcp add` argument: a lone known-preset name becomes a preset
/// server, otherwise the first token is the command and the rest are its args.
fn parse_mcp_add(rest: &str) -> McpServerSpec {
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let first = tokens.first().copied().unwrap_or("");
    if tokens.len() == 1 && heartbit_core::known_presets().contains(&first) {
        McpServerSpec::preset(first)
    } else {
        let args = tokens
            .get(1..)
            .unwrap_or(&[])
            .iter()
            .map(|s| s.to_string())
            .collect();
        McpServerSpec::stdio(first, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    fn key(code: KeyCode) -> Msg {
        Msg::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }
    fn ctrl(c: char) -> Msg {
        Msg::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL))
    }
    fn typed(app: &mut App, s: &str) {
        for c in s.chars() {
            app.update(key(KeyCode::Char(c)));
        }
    }
    /// An app that already has an API key (so submits go straight to the agent).
    fn keyed() -> App {
        let mut app = App::new("m");
        app.api_key = Some("sk-or-test".into());
        app
    }

    #[test]
    fn effect_names_are_stable_snake_case() {
        assert_eq!(Effect::FetchModels.name(), "fetch_models");
        assert_eq!(Effect::SendInput("x".into()).name(), "send_input");
        assert_eq!(Effect::Interrupt.name(), "interrupt");
    }

    #[test]
    fn submit_without_key_opens_key_modal_and_keeps_message() {
        let mut app = App::new("m"); // no key, no fallback
        typed(&mut app, "hi");
        app.update(key(KeyCode::Enter));
        assert!(
            matches!(app.modal, Some(Modal::KeyEntry(_))),
            "no-key submit must open the key prompt"
        );
        assert_eq!(app.composer.text(), "hi", "the message must be preserved");
        assert!(
            app.effects.is_empty(),
            "must not send to the agent without a key"
        );
    }

    #[test]
    fn submit_with_fallback_provider_sends_without_key() {
        let mut app = App::new("m");
        app.has_fallback_provider = true; // e.g. ANTHROPIC_API_KEY present
        typed(&mut app, "hi");
        app.update(key(KeyCode::Enter));
        assert_eq!(
            app.effects,
            vec![
                Effect::PersistPrompt("hi".into()),
                Effect::SendInput("hi".into())
            ]
        );
        assert!(app.modal.is_none());
    }

    #[test]
    fn key_entry_modal_sets_key_and_emits_save() {
        let mut app = App::new("m"); // opens nothing yet
        app.open_key_modal();
        typed(&mut app, "sk-or-secret");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.api_key.as_deref(), Some("sk-or-secret"));
        assert_eq!(app.effects, vec![Effect::SaveKey("sk-or-secret".into())]);
        assert!(app.modal.is_none());
    }

    #[test]
    fn slash_key_sets_key_without_leaking_into_history() {
        let mut app = App::new("m");
        typed(&mut app, "/key sk-or-topsecret");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.api_key.as_deref(), Some("sk-or-topsecret"));
        assert!(
            app.effects
                .contains(&Effect::SaveKey("sk-or-topsecret".into()))
        );
        // The secret must NOT be recallable via input history.
        app.composer.history_prev();
        assert_ne!(app.composer.text(), "/key sk-or-topsecret");
        assert_eq!(app.composer.text(), "");
    }

    #[test]
    fn paste_into_key_modal_appends_to_key_field_not_composer() {
        let mut app = App::new("m");
        app.open_key_modal();
        app.update(Msg::Paste("sk-or-pasted-key".into()));
        app.update(key(KeyCode::Enter));
        assert_eq!(app.api_key.as_deref(), Some("sk-or-pasted-key"));
        assert!(
            app.composer.is_empty(),
            "paste must not leak into the composer"
        );
    }

    #[test]
    fn slash_model_sets_model_and_saves() {
        let mut app = keyed();
        typed(&mut app, "/model openai/gpt-x");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.model, "openai/gpt-x");
        assert!(
            app.effects
                .contains(&Effect::SaveModel("openai/gpt-x".into()))
        );
    }

    #[test]
    fn slash_codex_sets_endpoint_model_and_respawns() {
        let mut app = keyed(); // idle → respawn is immediate
        typed(&mut app, "/codex");
        app.update(key(KeyCode::Enter));
        assert_eq!(
            app.custom_endpoint.as_deref(),
            Some("http://127.0.0.1:10531/v1"),
            "bare /codex uses the default proxy URL"
        );
        assert_eq!(app.model, "gpt-5.5");
        assert!(app.has_fallback_provider, "a custom endpoint IS a provider");
        // SESSION-ONLY: the Codex model id must NOT be persisted (it would brick
        // the next cold start once the proxy is gone).
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::SaveModel(_))),
            "/codex must not persist the model to config"
        );
        assert!(
            app.effects.contains(&Effect::RespawnAgent),
            "the model/endpoint switch must rebuild the idle agent"
        );
    }

    #[test]
    fn slash_codex_refreshes_the_model_catalog_from_the_proxy() {
        let mut app = keyed();
        // Seed a stale OpenRouter catalogue so we can prove it gets cleared.
        app.models = vec![crate::models::ModelEntry {
            id: "openrouter/old".into(),
            name: "old".into(),
            context: None,
        }];
        typed(&mut app, "/codex");
        app.update(key(KeyCode::Enter));
        assert!(
            app.models.is_empty(),
            "the stale OpenRouter catalogue must be cleared so the proxy's loads"
        );
        assert!(app.models_loading, "a refetch is in flight");
        assert!(
            app.effects.contains(&Effect::FetchModels),
            "/codex must refetch the catalogue (now from the proxy's /v1/models)"
        );
    }

    #[test]
    fn slash_codex_off_restores_the_prior_model_and_fallback() {
        let mut app = keyed(); // keyed() ⇒ model "m", no fallback
        let prior_model = app.model.clone();
        let prior_fallback = app.has_fallback_provider;
        typed(&mut app, "/codex http://127.0.0.1:8080/v1");
        app.update(key(KeyCode::Enter));
        assert_eq!(
            app.custom_endpoint.as_deref(),
            Some("http://127.0.0.1:8080/v1")
        );
        assert_eq!(app.model, "gpt-5.5");

        typed(&mut app, "/codex off");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.custom_endpoint, None, "/codex off clears the endpoint");
        assert_eq!(app.model, prior_model, "the pre-codex model is restored");
        assert_eq!(
            app.has_fallback_provider, prior_fallback,
            "the pre-codex fallback flag is restored"
        );
        assert!(app.codex_saved.is_none(), "the stash is consumed");
    }

    #[test]
    fn slash_mcp_add_preset_records_and_saves() {
        let mut app = keyed();
        typed(&mut app, "/mcp add chrome-devtools");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.mcp_servers.len(), 1);
        assert_eq!(
            app.mcp_servers[0].preset.as_deref(),
            Some("chrome-devtools"),
            "a known preset name must become a preset server"
        );
        assert!(
            app.effects
                .contains(&Effect::SaveMcp(app.mcp_servers.clone()))
        );
    }

    #[test]
    fn slash_mcp_add_command_becomes_stdio() {
        let mut app = keyed();
        typed(&mut app, "/mcp add npx -y some-mcp");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.mcp_servers.len(), 1);
        let s = &app.mcp_servers[0];
        assert_eq!(s.command.as_deref(), Some("npx"));
        assert_eq!(s.args, vec!["-y", "some-mcp"]);
        assert!(s.preset.is_none());
    }

    #[test]
    fn slash_mcp_clear_empties_and_saves() {
        let mut app = keyed();
        app.mcp_servers = vec![McpServerSpec::preset("chrome-devtools")];
        typed(&mut app, "/mcp clear");
        app.update(key(KeyCode::Enter));
        assert!(app.mcp_servers.is_empty());
        assert!(app.effects.contains(&Effect::SaveMcp(Vec::new())));
    }

    #[test]
    fn slash_mcp_list_when_empty_hints_how_to_add() {
        let mut app = keyed();
        typed(&mut app, "/mcp");
        app.update(key(KeyCode::Enter));
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("no MCP servers"))),
            "empty /mcp must hint how to add a server"
        );
        // A pure config command must never start a run.
        assert!(
            app.effects
                .iter()
                .all(|e| !matches!(e, Effect::SendInput(_)))
        );
    }

    #[test]
    fn slash_opens_command_menu_with_all_commands() {
        let mut app = keyed();
        typed(&mut app, "/");
        assert!(app.menu_open());
        assert_eq!(app.command_candidates().len(), SLASH_COMMANDS.len());
    }

    #[test]
    fn command_menu_filters_by_prefix() {
        let mut app = keyed();
        typed(&mut app, "/m");
        let names: Vec<&str> = app.command_candidates().iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"/mode") && names.contains(&"/model") && names.contains(&"/mcp"));
        assert!(!names.contains(&"/help"));
        typed(&mut app, "o"); // "/mo" → /mode and /model (shared prefix)
        let names: Vec<&str> = app.command_candidates().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["/mode", "/model"]);
        typed(&mut app, "del"); // "/model" → only /model
        let names: Vec<&str> = app.command_candidates().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["/model"]);
    }

    #[test]
    fn command_menu_closes_after_a_space() {
        let mut app = keyed();
        typed(&mut app, "/model ");
        assert!(!app.menu_open(), "a space ends command-name typing");
    }

    #[test]
    fn menu_arrows_navigate_and_wrap() {
        let mut app = keyed();
        typed(&mut app, "/");
        let n = app.command_candidates().len();
        app.update(key(KeyCode::Down));
        assert_eq!(app.menu_selected, 1);
        app.update(key(KeyCode::Up));
        assert_eq!(app.menu_selected, 0);
        app.update(key(KeyCode::Up)); // wrap to the bottom
        assert_eq!(app.menu_selected, n - 1);
    }

    #[test]
    fn menu_tab_completes_with_trailing_space() {
        let mut app = keyed();
        typed(&mut app, "/mod"); // → [/mode, /model]; selected=0 → /mode
        app.update(Msg::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)));
        assert_eq!(app.composer.text(), "/mode ");
        assert!(!app.menu_open(), "completion closes the menu");
    }

    #[test]
    fn menu_enter_runs_navigated_command() {
        let mut app = keyed();
        typed(&mut app, "/"); // all commands, selected = 0 (/help)
        app.update(key(KeyCode::Down)); // selected = 1 (/mode)
        app.update(key(KeyCode::Down)); // selected = 2 (/model)
        app.update(key(KeyCode::Enter));
        // /model opens the picker — proving the NAVIGATED command ran, not /help.
        assert!(
            matches!(app.modal, Some(Modal::ModelPicker(_))),
            "navigated /model must run (open the picker), not /help"
        );
    }

    #[test]
    fn menu_runs_when_enter_arrives_as_cr_char() {
        // Some terminals deliver Enter as a raw CR/LF character.
        let mut app = keyed();
        typed(&mut app, "/he");
        app.update(Msg::Key(KeyEvent::new(
            KeyCode::Char('\r'),
            KeyModifiers::NONE,
        )));
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("commands"))),
            "CR-as-char Enter must still run the command"
        );
    }

    #[test]
    fn menu_enter_runs_the_selected_command() {
        let mut app = keyed();
        typed(&mut app, "/he"); // → /help
        app.update(key(KeyCode::Enter));
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("commands"))),
            "Enter must run /help (not the partial /he)"
        );
        assert!(app.composer.is_empty());
    }

    #[test]
    fn slash_model_no_arg_opens_picker_and_fetches() {
        let mut app = keyed();
        typed(&mut app, "/model");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.modal, Some(Modal::ModelPicker(_))));
        assert!(app.models_loading);
        assert!(app.effects.contains(&Effect::FetchModels));
    }

    #[test]
    fn slash_model_with_arg_sets_directly_no_picker() {
        let mut app = keyed();
        typed(&mut app, "/model openai/gpt-x");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.model, "openai/gpt-x");
        assert!(
            app.effects
                .contains(&Effect::SaveModel("openai/gpt-x".into()))
        );
        assert!(app.modal.is_none(), "a direct set must not open the picker");
    }

    #[test]
    fn model_picker_loads_filters_and_selects() {
        let mut app = keyed();
        typed(&mut app, "/model");
        app.update(key(KeyCode::Enter)); // open picker + FetchModels
        app.update(Msg::ModelsLoaded(vec![
            ModelEntry {
                id: "qwen/q".into(),
                name: "Qwen".into(),
                context: None,
            },
            ModelEntry {
                id: "anthropic/claude".into(),
                name: "Claude".into(),
                context: None,
            },
            ModelEntry {
                id: "openai/gpt".into(),
                name: "GPT".into(),
                context: None,
            },
        ]));
        assert!(!app.models_loading);
        for c in "claude".chars() {
            app.update(key(KeyCode::Char(c)));
        }
        app.update(key(KeyCode::Enter)); // select the only match
        assert_eq!(app.model, "anthropic/claude");
        assert!(app.modal.is_none(), "selecting closes the picker");
        assert!(
            app.effects
                .contains(&Effect::SaveModel("anthropic/claude".into()))
        );
    }

    #[test]
    fn model_picker_esc_cancels_without_change() {
        let mut app = keyed();
        typed(&mut app, "/model");
        app.update(key(KeyCode::Enter));
        app.update(Msg::ModelsLoaded(vec![ModelEntry {
            id: "x/y".into(),
            name: "X".into(),
            context: None,
        }]));
        app.update(key(KeyCode::Esc));
        assert!(app.modal.is_none());
        assert_eq!(app.model, "m", "Esc must not change the model");
    }

    #[test]
    fn submit_persists_the_raw_prompt() {
        let mut app = keyed();
        typed(&mut app, "corrige le bug du parseur");
        app.update(key(KeyCode::Enter));
        assert!(
            app.effects
                .contains(&Effect::PersistPrompt("corrige le bug du parseur".into())),
            "user prompts must reach the per-directory history"
        );
    }

    #[test]
    fn slash_commands_are_never_persisted() {
        let mut app = keyed();
        typed(&mut app, "/key sk-secret-123");
        app.update(key(KeyCode::Enter));
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::PersistPrompt(_))),
            "slash commands (possibly secrets) must never be persisted"
        );
    }

    #[test]
    fn slash_mode_pins_request_mode_and_releases() {
        let mut app = keyed();
        typed(&mut app, "/mode study");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::SetRequestModePin(3)));
        typed(&mut app, "/mode auto");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::SetRequestModePin(0)));
        // permission modes still work
        typed(&mut app, "/mode yolo");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.permission_mode, PermissionMode::Yolo);
    }

    #[test]
    fn run_failed_suggests_learn() {
        let mut app = keyed();
        app.update(Msg::RunFailed("boom".into()));
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("/learn"))),
            "a failed run must point at /learn"
        );
    }

    #[test]
    fn interrupt_after_many_tool_errors_suggests_learn() {
        let mut app = keyed();
        for _ in 0..3 {
            app.history.push(Cell::Tool {
                name: "bash".into(),
                input: "{}".into(),
                status: crate::cells::ToolStatus::Failed,
                output: Some("error[E0308]".into()),
                duration_ms: Some(5),
                agent: None,
            });
        }
        app.running = true;
        app.update(key(KeyCode::Esc)); // interrupt
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("/learn"))),
            "an error-heavy interrupted session must point at /learn: {:?}",
            app.history.len()
        );
    }

    #[test]
    fn slash_model_requests_respawn_so_the_change_takes_effect() {
        // Live finding (session 6a25ca5e): /model persisted but the long-lived
        // agent kept the old model. Changing the model must request a respawn.
        let mut app = keyed();
        typed(&mut app, "/model qwen/qwen3-vl-235b");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.model, "qwen/qwen3-vl-235b");
        assert!(
            app.effects
                .contains(&Effect::SaveModel("qwen/qwen3-vl-235b".into()))
        );
        assert!(
            app.effects.contains(&Effect::RespawnAgent),
            "a model change must request an agent respawn"
        );
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("next message"))),
            "the notice must say the change applies on the next message"
        );
    }

    #[test]
    fn slash_model_advisor_requests_respawn() {
        let mut app = keyed();
        typed(&mut app, "/model advisor anthropic/claude-opus-4");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::RespawnAgent));
        // and clearing too
        let mut app2 = keyed();
        typed(&mut app2, "/model advisor clear");
        app2.update(key(KeyCode::Enter));
        assert!(app2.effects.contains(&Effect::RespawnAgent));
    }

    #[test]
    fn slash_model_mid_run_defers_respawn_to_turn_idle() {
        // Audit 2026-06-09: an immediate RespawnAgent mid-run swaps the input
        // channel out from under the live engine — the next message then spawns
        // a SECOND engine while the old one still runs (shared workspace/UI).
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "/model openai/gpt-x");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.model, "openai/gpt-x", "the choice is not dropped");
        assert!(
            app.effects
                .contains(&Effect::SaveModel("openai/gpt-x".into())),
            "persistence still happens immediately"
        );
        assert!(
            !app.effects.contains(&Effect::RespawnAgent),
            "no respawn while a turn is in flight"
        );
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("turn ends"))),
            "the notice must say the change applies when the turn ends"
        );
        // Turn-idle (final text-only LlmDone) flushes the deferred respawn.
        app.effects.clear();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(app.effects.contains(&Effect::RespawnAgent));
        // Once flushed, it must not fire again on the next idle turn.
        app.effects.clear();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(!app.effects.contains(&Effect::RespawnAgent));
    }

    #[test]
    fn deferred_respawn_flushes_on_interrupt_too() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "/model advisor anthropic/claude-opus-4");
        app.update(key(KeyCode::Enter));
        assert!(!app.effects.contains(&Effect::RespawnAgent));
        app.update(key(KeyCode::Esc)); // interrupt ends the turn
        assert!(
            app.effects.contains(&Effect::RespawnAgent),
            "Esc ends the turn — the deferred respawn must flush"
        );
    }

    #[test]
    fn effort_level_parse_and_label_roundtrip() {
        for (s, lvl) in [
            ("off", EffortLevel::Off),
            ("low", EffortLevel::Low),
            ("medium", EffortLevel::Medium),
            ("high", EffortLevel::High),
        ] {
            assert_eq!(EffortLevel::parse(s), Some(lvl));
            assert_eq!(lvl.label(), s);
        }
        assert_eq!(EffortLevel::parse("HIGH"), Some(EffortLevel::High));
        assert_eq!(EffortLevel::parse("turbo"), None);
        assert_eq!(EffortLevel::default(), EffortLevel::Off);
    }

    #[test]
    fn slash_effort_sets_level_persists_and_requests_respawn() {
        let mut app = keyed();
        typed(&mut app, "/effort high");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.effort, EffortLevel::High);
        assert!(
            app.effects
                .contains(&Effect::SaveReasoningEffort(Some("high".into())))
        );
        assert!(app.effects.contains(&Effect::RespawnAgent));
    }

    #[test]
    fn slash_effort_off_clears_and_drops_the_config_key() {
        let mut app = keyed();
        typed(&mut app, "/effort high");
        app.update(key(KeyCode::Enter));
        app.effects.clear();
        typed(&mut app, "/effort off");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.effort, EffortLevel::Off);
        assert!(app.effects.contains(&Effect::SaveReasoningEffort(None)));
    }

    #[test]
    fn slash_effort_mid_run_defers_respawn_to_turn_idle() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "/effort low");
        app.update(key(KeyCode::Enter));
        assert!(app.pending_respawn);
        assert!(!app.effects.contains(&Effect::RespawnAgent));
    }

    #[test]
    fn slash_effort_unknown_arg_reports_usage_and_changes_nothing() {
        let mut app = keyed();
        typed(&mut app, "/effort turbo");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.effort, EffortLevel::Off);
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::SaveReasoningEffort(_)))
        );
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("usage")))
        );
    }

    #[test]
    fn slash_effort_bare_opens_picker_preselected_on_current_level() {
        let mut app = keyed();
        app.effort = EffortLevel::Medium;
        typed(&mut app, "/effort");
        app.update(key(KeyCode::Enter));
        assert!(matches!(
            app.modal,
            Some(Modal::EffortPicker { sel: 2 }) // Medium is EffortLevel::ALL[2]
        ));
    }

    #[test]
    fn effort_picker_enter_applies_the_highlighted_level() {
        let mut app = keyed();
        app.modal = Some(Modal::EffortPicker { sel: 3 }); // High
        app.update(key(KeyCode::Enter));
        assert_eq!(app.effort, EffortLevel::High);
        assert!(app.modal.is_none());
        assert!(
            app.effects
                .contains(&Effect::SaveReasoningEffort(Some("high".into())))
        );
    }

    #[test]
    fn effort_picker_esc_cancels_without_changing_anything() {
        let mut app = keyed();
        app.effort = EffortLevel::Off;
        app.modal = Some(Modal::EffortPicker { sel: 3 });
        app.update(key(KeyCode::Esc));
        assert!(app.modal.is_none());
        assert_eq!(app.effort, EffortLevel::Off);
        assert!(app.effects.is_empty());
    }

    #[test]
    fn paste_into_effort_picker_is_a_no_op() {
        // Mirrors the ModePicker contract in Msg::Paste's match — a picker
        // has no text field, so a paste must not leak anywhere.
        let mut app = keyed();
        app.modal = Some(Modal::EffortPicker { sel: 0 });
        app.update(Msg::Paste("ignored".into()));
        assert!(matches!(app.modal, Some(Modal::EffortPicker { sel: 0 })));
        assert!(app.composer.is_empty());
    }

    #[test]
    fn sub_agent_llm_done_accumulates_cost_without_lifecycle() {
        // Audit 2026-06-09: sub-agent LlmResponse leaked into Msg::LlmDone —
        // flipping running=false mid-run, settling the roster after the FIRST
        // sibling finished, committing /learn early, and overwriting the
        // context-fill bar with the sub-agent's window.
        let mut app = multi();
        app.running = true;
        app.learning = Some(42);
        app.context_tokens = 123;
        app.update(Msg::AgentsDispatched(vec![
            "worker".into(),
            "researcher".into(),
        ]));
        app.update(Msg::SubAgentLlmDone {
            usage: TokenUsage {
                input_tokens: 1000,
                output_tokens: 50,
                ..Default::default()
            },
        });
        assert_eq!(app.tokens.input_tokens, 1000, "cost still counts");
        assert_eq!(app.tokens.output_tokens, 50);
        assert!(app.running, "a sub-agent turn must not idle the run");
        assert_eq!(app.learning, Some(42), "no early /learn commit");
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::CommitLessons(_)))
        );
        assert_eq!(
            app.context_tokens, 123,
            "the context bar shows the ENTRY agent's fill, not the sub-agent's"
        );
        assert!(
            app.agents.iter().all(|r| r.state == AgentState::Working),
            "still-working siblings must not be settled Done"
        );
    }

    #[test]
    fn clear_drops_stale_tool_index() {
        // Audit 2026-06-09: /clear left tool_index pointing into the OLD
        // transcript — an in-flight ToolCompleted then overwrote whatever Tool
        // cell occupied that index in the regrown history.
        let mut app = keyed();
        app.history.push(Cell::User("hi".into()));
        app.update(tool_started("heartbit", "bash")); // id "heartbit-bash" at idx 1
        typed(&mut app, "/clear");
        app.update(key(KeyCode::Enter));
        // Regrow: a NEW tool lands at the same index the old id pointed to.
        app.update(Msg::ToolStarted {
            id: "t2".into(),
            name: "read".into(),
            input: "{}".into(),
            agent: "heartbit".into(),
        });
        app.update(Msg::ToolCompleted {
            id: "heartbit-bash".into(),
            is_error: true,
            output: "BOOM".into(),
            duration_ms: 9,
        });
        let t2 = app
            .history
            .iter()
            .find(|c| matches!(c, Cell::Tool { name, .. } if name == "read"))
            .expect("the new tool cell exists");
        assert!(
            matches!(
                t2,
                Cell::Tool {
                    status: ToolStatus::Running,
                    output: None,
                    ..
                }
            ),
            "a stale completion must not corrupt the regrown transcript"
        );
    }

    #[test]
    fn clear_mid_stream_drops_the_half_streamed_buffer() {
        // Audit-review 2026-06-09: /clear is reachable while the agent streams
        // (the engine keeps running). A surviving `active`/`active_reasoning`
        // buffer would keep growing and be flushed as a ghost Cell into the
        // just-cleared transcript. It must be dropped, like SessionLoaded does.
        let mut app = keyed();
        app.history.push(Cell::User("hi".into()));
        app.update(Msg::ReasoningDelta("thinking…".into()));
        app.update(Msg::StreamDelta("half an answer".into()));
        assert!(app.active.is_some(), "precondition: a live buffer exists");
        typed(&mut app, "/clear");
        app.update(key(KeyCode::Enter));
        assert!(app.active.is_none(), "/clear must drop the streamed buffer");
        assert!(
            app.active_reasoning.is_none(),
            "/clear must drop the streamed reasoning buffer"
        );
        // The in-flight turn continues; the next finalize must not resurrect
        // pre-clear content into the cleared transcript.
        app.update(Msg::StreamDelta("post-clear text".into()));
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        let ghost = app.history.iter().any(|c| {
            matches!(c, Cell::Agent(t) if t.contains("half an answer"))
                || matches!(c, Cell::Reasoning(t) if t.contains("thinking"))
        });
        assert!(!ghost, "no pre-clear content may reappear after /clear");
    }

    #[test]
    fn session_loaded_drops_stale_tool_index_and_stream() {
        let mut app = keyed();
        app.update(tool_started("heartbit", "bash")); // id "heartbit-bash" at idx 0
        app.update(Msg::StreamDelta("half-streamed".into()));
        app.update(Msg::SessionLoaded(vec![Cell::Tool {
            name: "write".into(),
            input: "{}".into(),
            status: ToolStatus::Ok,
            output: Some("done".into()),
            duration_ms: Some(1),
            agent: None,
        }]));
        assert!(
            app.active.is_none(),
            "no stale stream over a resumed session"
        );
        app.update(Msg::ToolCompleted {
            id: "heartbit-bash".into(),
            is_error: true,
            output: "BOOM".into(),
            duration_ms: 9,
        });
        assert!(
            matches!(
                &app.history[0],
                Cell::Tool { name, output: Some(o), .. } if name == "write" && o == "done"
            ),
            "a stale completion must not overwrite the resumed transcript"
        );
    }

    #[test]
    fn slash_goal_with_objective_sends_set_goal_instruction() {
        let mut app = keyed();
        typed(&mut app, "/goal tous les tests passent");
        app.update(key(KeyCode::Enter));
        let sent = app.effects.iter().find_map(|e| match e {
            Effect::SendInput(s) => Some(s.clone()),
            _ => None,
        });
        let sent = sent.expect("must submit a set_goal instruction");
        assert!(sent.contains("set_goal"), "names the tool: {sent}");
        assert!(
            sent.contains("tous les tests passent"),
            "carries the objective verbatim: {sent}"
        );
    }

    #[test]
    fn slash_goal_clear_sends_clear_instruction() {
        let mut app = keyed();
        typed(&mut app, "/goal clear");
        app.update(key(KeyCode::Enter));
        let sent = app.effects.iter().find_map(|e| match e {
            Effect::SendInput(s) => Some(s.clone()),
            _ => None,
        });
        let sent = sent.expect("must submit");
        assert!(sent.contains("set_goal") && sent.contains("clear"));
    }

    #[test]
    fn slash_goal_bare_notices_usage() {
        let mut app = keyed();
        typed(&mut app, "/goal");
        app.update(key(KeyCode::Enter));
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("/goal <"))),
            "bare /goal explains usage"
        );
    }

    #[test]
    fn slash_handoff_with_purpose_sends_tool_instruction() {
        let mut app = keyed();
        typed(&mut app, "/handoff prototype the picker UI");
        app.update(key(KeyCode::Enter));
        let sent = app.effects.iter().find_map(|e| match e {
            Effect::SendInput(s) => Some(s.clone()),
            _ => None,
        });
        let sent = sent.expect("must submit an instruction to the agent");
        assert!(sent.contains("handoff"), "names the tool: {sent}");
        assert!(
            sent.contains("prototype the picker UI"),
            "carries the purpose verbatim: {sent}"
        );
    }

    #[test]
    fn slash_handoff_bare_lists_briefs() {
        let mut app = keyed();
        typed(&mut app, "/handoff");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::ListHandoffs));
    }

    #[test]
    fn handoff_picker_enter_seeds_selected_brief() {
        let mut app = keyed();
        app.update(Msg::HandoffsListed(vec![
            crate::session::HandoffMeta {
                file_name: "2026-06-07-prototype.md".into(),
                path: std::path::PathBuf::from("/tmp/h/2026-06-07-prototype.md"),
                preview: "Purpose: prototype the picker".into(),
            },
            crate::session::HandoffMeta {
                file_name: "2026-06-06-refactor.md".into(),
                path: std::path::PathBuf::from("/tmp/h/2026-06-06-refactor.md"),
                preview: "Purpose: refactor".into(),
            },
        ]));
        assert!(matches!(app.modal, Some(Modal::HandoffPicker { .. })));
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Enter));
        assert!(app.modal.is_none());
        assert!(
            app.effects
                .contains(&Effect::SeedFromHandoff(std::path::PathBuf::from(
                    "/tmp/h/2026-06-06-refactor.md"
                )))
        );
    }

    #[test]
    fn handoffs_listed_empty_notices_instead_of_modal() {
        let mut app = keyed();
        app.update(Msg::HandoffsListed(vec![]));
        assert!(app.modal.is_none());
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("no handoff"))),
        );
    }

    #[test]
    fn run_failed_pushes_emergency_handoff_effect() {
        let mut app = keyed();
        typed(&mut app, "do something big");
        app.update(key(KeyCode::Enter));
        app.update(Msg::RunFailed("API error (402): out of credits".into()));
        assert!(
            app.effects
                .iter()
                .any(|e| matches!(e, Effect::EmergencyHandoff(err) if err.contains("402"))),
            "a terminal failure must leave a brief behind: {:?}",
            app.effects
        );
    }

    fn question_msg(
        multiple: bool,
        n_questions: usize,
    ) -> (
        Msg,
        tokio::sync::oneshot::Receiver<heartbit_core::tool::builtins::QuestionResponse>,
    ) {
        use heartbit_core::tool::builtins::{Question, QuestionOption, QuestionRequest};
        let q = |i: usize| Question {
            question: format!("Which approach for part {i}?"),
            header: format!("Q{i}"),
            options: vec![
                QuestionOption {
                    label: "option-a".into(),
                    description: "the first way".into(),
                },
                QuestionOption {
                    label: "option-b".into(),
                    description: "the second way".into(),
                },
            ],
            multiple,
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        (
            Msg::Question {
                request: QuestionRequest {
                    questions: (0..n_questions).map(q).collect(),
                },
                reply: tx,
            },
            rx,
        )
    }

    #[test]
    fn question_msg_opens_modal_and_enter_returns_single_label() {
        let mut app = keyed();
        let (msg, mut rx) = question_msg(false, 1);
        app.update(msg);
        assert!(matches!(app.modal, Some(Modal::Question(_))));
        app.update(key(KeyCode::Down)); // highlight option-b
        app.update(key(KeyCode::Enter));
        assert!(app.modal.is_none(), "answering closes the modal");
        let resp = rx.try_recv().expect("answer delivered");
        assert_eq!(resp.answers, vec![vec!["option-b".to_string()]]);
    }

    #[test]
    fn question_modal_multi_select_space_toggles_and_enter_returns_all() {
        let mut app = keyed();
        let (msg, mut rx) = question_msg(true, 1);
        app.update(msg);
        app.update(key(KeyCode::Char(' '))); // toggle option-a
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Char(' '))); // toggle option-b
        app.update(key(KeyCode::Enter));
        let resp = rx.try_recv().expect("answer delivered");
        assert_eq!(
            resp.answers,
            vec![vec!["option-a".to_string(), "option-b".to_string()]]
        );
    }

    #[test]
    fn question_modal_advances_through_questions() {
        let mut app = keyed();
        let (msg, mut rx) = question_msg(false, 2);
        app.update(msg);
        app.update(key(KeyCode::Enter)); // Q0 → option-a (default highlight)
        assert!(
            matches!(app.modal, Some(Modal::Question(_))),
            "modal stays open for the next question"
        );
        app.update(key(KeyCode::Down));
        app.update(key(KeyCode::Enter)); // Q1 → option-b
        let resp = rx.try_recv().expect("answer delivered");
        assert_eq!(
            resp.answers,
            vec![vec!["option-a".to_string()], vec!["option-b".to_string()]]
        );
    }

    #[test]
    fn question_modal_esc_drops_reply_channel() {
        let mut app = keyed();
        let (msg, mut rx) = question_msg(false, 1);
        app.update(msg);
        app.update(key(KeyCode::Esc));
        assert!(app.modal.is_none());
        assert!(
            rx.try_recv().is_err(),
            "dismissing must drop the sender so the tool reports failure"
        );
    }

    #[test]
    fn slash_model_advisor_no_arg_opens_picker_for_advisor() {
        let mut app = keyed();
        typed(&mut app, "/model advisor");
        app.update(key(KeyCode::Enter));
        assert!(
            matches!(
                app.modal,
                Some(Modal::ModelPicker(ModelPicker {
                    target: ModelTarget::Advisor,
                    ..
                }))
            ),
            "bare `/model advisor` opens the picker targeting the advisor"
        );
        assert!(app.models_loading);
        assert!(app.effects.contains(&Effect::FetchModels));
    }

    #[test]
    fn slash_model_advisor_with_arg_sets_frontier_and_saves() {
        let mut app = keyed();
        typed(&mut app, "/model advisor anthropic/claude-opus-4");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.model, "m", "the MAIN model must not change");
        assert!(app.modal.is_none(), "a direct set must not open the picker");
        assert!(app.effects.contains(&Effect::SaveFrontierModel(Some(
            "anthropic/claude-opus-4".into()
        ))));
        assert_eq!(
            app.frontier_model.as_deref(),
            Some("anthropic/claude-opus-4"),
            "the in-memory advisor model drives the next respawn"
        );
        assert!(
            app.history.iter().any(
                |c| matches!(c, Cell::Notice(n) if n.contains("advisor model set to anthropic/claude-opus-4"))
            ),
            "must notice the advisor model change"
        );
    }

    #[test]
    fn slash_model_advisor_clear_unsets_frontier() {
        let mut app = keyed();
        app.frontier_model = Some("a/opus".into());
        typed(&mut app, "/model advisor clear");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::SaveFrontierModel(None)));
        assert!(app.frontier_model.is_none(), "in-memory value cleared too");
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("advisor model cleared"))),
            "must notice the fallback to the main model"
        );
    }

    #[test]
    fn model_picker_advisor_enter_sets_frontier_not_main() {
        let mut app = keyed();
        typed(&mut app, "/model advisor");
        app.update(key(KeyCode::Enter)); // open advisor picker + FetchModels
        app.update(Msg::ModelsLoaded(vec![
            ModelEntry {
                id: "qwen/q".into(),
                name: "Qwen".into(),
                context: None,
            },
            ModelEntry {
                id: "anthropic/claude".into(),
                name: "Claude".into(),
                context: None,
            },
        ]));
        for c in "claude".chars() {
            app.update(key(KeyCode::Char(c)));
        }
        app.update(key(KeyCode::Enter)); // select the only match
        assert_eq!(app.model, "m", "the MAIN model must not change");
        assert!(app.modal.is_none(), "selecting closes the picker");
        assert!(
            app.effects
                .contains(&Effect::SaveFrontierModel(Some("anthropic/claude".into())))
        );
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::SaveModel(_))),
            "advisor selection must not save the main model"
        );
    }

    #[test]
    fn models_failed_closes_picker_and_notices() {
        let mut app = keyed();
        typed(&mut app, "/model");
        app.update(key(KeyCode::Enter));
        app.update(Msg::ModelsFailed("offline".into()));
        assert!(app.modal.is_none(), "failure closes the picker");
        assert!(!app.models_loading);
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("could not load models"))),
            "must fall back with a notice"
        );
    }

    #[test]
    fn eager_models_failure_is_silent_no_notice() {
        // The startup catalog fetch (no picker open) must NOT push a notice — the
        // user never asked for it; the context bar falls back to a raw count.
        let mut app = keyed();
        app.update(Msg::ModelsFailed("offline".into()));
        assert!(!app.models_loading);
        assert!(
            app.history.is_empty(),
            "an unrequested startup fetch failure must be silent"
        );
    }

    #[test]
    fn slash_agents_is_informational() {
        let mut app = keyed();
        typed(&mut app, "/agents");
        app.update(key(KeyCode::Enter));
        // No mode to toggle anymore — it just prints an informational notice.
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("always unified"))),
            "/agents should print the unified-agent notice"
        );
    }

    #[test]
    fn slash_context_recall_toggles_and_saves() {
        let mut app = keyed();
        assert!(app.context_recall, "context-recall is ON by default");
        typed(&mut app, "/context-recall off");
        app.update(key(KeyCode::Enter));
        assert!(!app.context_recall, "explicit off disables it");
        assert!(app.effects.contains(&Effect::SaveContextRecall(false)));
        // bare toggle flips it back on
        typed(&mut app, "/context-recall");
        app.update(key(KeyCode::Enter));
        assert!(app.context_recall);
        assert!(app.effects.contains(&Effect::SaveContextRecall(true)));
    }

    #[test]
    fn slash_verify_sets_and_clears_the_command() {
        let mut app = keyed();
        assert!(app.verify_command.is_none());
        typed(&mut app, "/verify cargo test");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.verify_command.as_deref(), Some("cargo test"));
        assert!(
            app.effects
                .contains(&Effect::SaveVerifyCommand(Some("cargo test".into())))
        );
        typed(&mut app, "/verify off");
        app.update(key(KeyCode::Enter));
        assert!(app.verify_command.is_none());
        assert!(app.effects.contains(&Effect::SaveVerifyCommand(None)));
    }

    #[test]
    fn slash_agents_never_starts_a_run() {
        let mut app = keyed();
        typed(&mut app, "/agents on");
        app.update(key(KeyCode::Enter));
        // a pure informational command must never start a run
        assert!(
            app.effects
                .iter()
                .all(|e| !matches!(e, Effect::SendInput(_)))
        );
    }

    /// An app in multi-agent mode (roster tracking active).
    fn multi() -> App {
        let mut app = keyed();
        app.multi_agent = true;
        app.squad = DEFAULT_SQUAD.iter().map(|s| s.to_string()).collect();
        app
    }
    fn tool_started(agent: &str, name: &str) -> Msg {
        Msg::ToolStarted {
            id: format!("{agent}-{name}"),
            name: name.into(),
            input: "{}".into(),
            agent: agent.into(),
        }
    }

    #[test]
    fn roster_tracks_dispatch_work_and_done() {
        let mut app = multi();
        app.update(Msg::AgentsDispatched(vec![
            "worker".into(),
            "researcher".into(),
        ]));
        assert_eq!(app.agents.len(), 2);
        assert!(app.agents.iter().all(|r| r.state == AgentState::Working));
        // the delegation line is shown in the transcript
        assert!(app.history.iter().any(
            |c| matches!(c, Cell::Notice(n) if n.contains("delegating to worker, researcher"))
        ),);
        // worker starts a tool → activity reflects it
        app.update(tool_started("worker", "write"));
        let w = app.agents.iter().find(|r| r.name == "worker").unwrap();
        assert_eq!(w.activity, "write");
        assert_eq!(w.state, AgentState::Working);
        // worker completes with a token cost
        app.update(Msg::SubAgentDone {
            agent: "worker".into(),
            success: true,
            tokens: 1234,
            error: None,
        });
        let w = app.agents.iter().find(|r| r.name == "worker").unwrap();
        assert_eq!(w.state, AgentState::Done);
        assert_eq!(w.tokens, 1234);
    }

    #[test]
    fn failed_sub_agent_pushes_a_visible_notice() {
        // Live trace 6a2e92e3: the researcher hit "Max turns (60) exceeded" and
        // the failure was swallowed (roster row only). It must be surfaced.
        let mut app = multi();
        app.update(Msg::AgentsDispatched(vec!["researcher".into()]));
        app.update(Msg::SubAgentDone {
            agent: "researcher".into(),
            success: false,
            tokens: 10,
            error: Some("Max turns (60) exceeded".into()),
        });
        assert!(
            app.history.iter().any(|c| matches!(
                c,
                Cell::Notice(n) if n.contains("researcher failed") && n.contains("Max turns")
            )),
            "a sub-agent failure must surface a visible notice naming the reason"
        );
        // …and stay non-fatal (the run is not marked failed).
        assert!(
            !app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("run failed")))
        );
    }

    #[test]
    fn roster_orchestrator_appears_and_settles_on_text_turn() {
        let mut app = multi();
        app.update(tool_started("orchestrator", "delegate_task"));
        assert_eq!(app.agents[0].name, "orchestrator");
        assert_eq!(app.agents[0].state, AgentState::Working);
        // a final text-only turn settles all still-working agents to Done
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert_eq!(app.agents[0].state, AgentState::Done);
    }

    #[test]
    fn solo_tool_call_inert_but_delegation_activates_roster() {
        // Live finding 6a25eb4d: the option-C entry agent delegates dynamically,
        // but the roster was gated behind the manual /agents flag — so 4 parallel
        // workers ran with ZERO attribution. New contract: a solo tool call stays
        // inert (no roster noise for pure chat), but a delegation activates the
        // roster even without the manual toggle.
        let mut app = keyed(); // multi_agent = false
        app.update(tool_started("heartbit", "bash"));
        assert!(
            app.agents.is_empty(),
            "a solo tool call before any delegation must not open the roster"
        );
        app.update(Msg::AgentsDispatched(vec!["x".into()]));
        assert_eq!(app.agents.len(), 1, "delegation opens the roster");
        assert_eq!(app.agents[0].name, "x");
    }

    #[test]
    fn delegation_enables_sub_agent_badges_without_manual_toggle() {
        let mut app = keyed(); // multi_agent = false
        // Before delegation: a solo cell is unbadged (clean for pure chat).
        app.update(tool_started("orchestrator", "bash"));
        assert!(matches!(
            app.history.last(),
            Some(Cell::Tool { agent: None, .. })
        ));
        // After delegation: sub-agent tool cells carry their agent badge so the
        // transcript shows who ran each tool.
        app.update(Msg::AgentsDispatched(vec!["worker".into()]));
        app.update(tool_started("worker", "write"));
        assert!(matches!(
            app.history.last(),
            Some(Cell::Tool { agent: Some(a), .. }) if a == "worker"
        ));
    }

    #[test]
    fn dispatch_notice_counts_duplicate_agents() {
        // "→ delegating to worker, worker, worker, worker" reads like a bug;
        // count-format it so 4 parallel workers are legible.
        assert_eq!(
            format_dispatch_names(&[
                "worker".into(),
                "worker".into(),
                "worker".into(),
                "worker".into()
            ]),
            "worker ×4"
        );
        assert_eq!(
            format_dispatch_names(&["worker".into(), "researcher".into()]),
            "worker, researcher"
        );
        assert_eq!(
            format_dispatch_names(&["a".into(), "a".into(), "b".into()]),
            "a ×2, b"
        );
    }

    #[test]
    fn new_user_turn_reseeds_the_idle_squad() {
        let mut app = multi();
        app.update(Msg::AgentsDispatched(vec!["worker".into()]));
        // worker working + researcher idle-seeded? (no submit yet → just worker)
        typed(&mut app, "next task");
        app.update(key(KeyCode::Enter));
        // A fresh turn shows the WHOLE squad again, all Idle (not a stale roster).
        assert_eq!(app.agents.len(), 2);
        assert!(
            app.agents.iter().all(|r| r.state == AgentState::Idle),
            "each turn restarts with the full squad available (idle)"
        );
    }

    #[test]
    fn multi_agent_turn_seeds_available_squad_then_dispatch_flips_one() {
        let mut app = multi();
        typed(&mut app, "do something");
        app.update(key(KeyCode::Enter));
        // The whole squad is visible up-front, available.
        assert_eq!(app.agents.len(), 2);
        assert!(app.agents.iter().all(|r| r.state == AgentState::Idle));
        // The orchestrator dispatches to worker only…
        app.update(Msg::AgentsDispatched(vec!["worker".into()]));
        let worker = app.agents.iter().find(|r| r.name == "worker").unwrap();
        let researcher = app.agents.iter().find(|r| r.name == "researcher").unwrap();
        assert_eq!(worker.state, AgentState::Working);
        assert_eq!(
            researcher.state,
            AgentState::Idle,
            "an undispatched squad member stays visible as idle (so the user sees what DIDN'T run)"
        );
    }

    #[test]
    fn tool_cell_carries_agent_badge_only_in_multi_mode() {
        let mut app = multi();
        app.update(tool_started("worker", "write"));
        assert!(matches!(
            app.history.last(),
            Some(Cell::Tool { agent: Some(a), .. }) if a == "worker"
        ));
        let mut single = keyed();
        single.update(tool_started("heartbit", "write"));
        assert!(matches!(
            single.history.last(),
            Some(Cell::Tool { agent: None, .. })
        ));
    }

    #[test]
    fn slash_help_and_unknown_emit_notices_not_sends() {
        let mut app = keyed();
        typed(&mut app, "/help");
        app.update(key(KeyCode::Enter));
        typed(&mut app, "/wat");
        app.update(key(KeyCode::Enter));
        assert!(
            app.effects
                .iter()
                .all(|e| !matches!(e, Effect::SendInput(_)))
        );
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("commands")))
        );
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("unknown command")))
        );
    }

    #[test]
    fn submit_creates_user_cell_and_send_effect() {
        let mut app = keyed();
        typed(&mut app, "hello");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.history.last(), Some(Cell::User(t)) if t == "hello"));
        assert_eq!(
            app.effects,
            vec![
                Effect::PersistPrompt("hello".into()),
                Effect::SendInput("hello".into())
            ]
        );
        assert!(app.running);
        assert!(app.composer.is_empty());
    }

    #[test]
    fn blank_submit_is_ignored() {
        let mut app = App::new("m");
        app.update(key(KeyCode::Enter));
        assert!(app.history.is_empty());
        assert!(app.effects.is_empty());
    }

    #[test]
    fn shift_enter_inserts_newline_not_submit() {
        let mut app = App::new("m");
        typed(&mut app, "a");
        app.update(Msg::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)));
        typed(&mut app, "b");
        assert_eq!(app.composer.text(), "a\nb");
        assert!(app.effects.is_empty(), "shift+enter must not submit");
    }

    #[test]
    fn ctrl_u_clears_the_draft_but_keeps_recall_history() {
        let mut app = keyed();
        app.composer.seed_history(vec!["earlier prompt".into()]);
        typed(&mut app, "a draft");
        app.update(Msg::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)));
        typed(&mut app, "with two lines");
        app.update(ctrl('u'));
        // The draft is gone and the cursor is genuinely reset (row too).
        assert!(app.composer.text().is_empty());
        assert_eq!(app.composer.cursor(), (0, 0));
        // …but the seeded history survives: Up recalls it.
        app.update(key(KeyCode::Up));
        assert_eq!(app.composer.text(), "earlier prompt");
    }

    #[test]
    fn reasoning_streams_live_then_flushes_above_the_answer() {
        let mut app = App::new("m");
        // Reasoning streams first into its own buffer (rendered live, dimmed)…
        app.update(Msg::ReasoningDelta("let me ".into()));
        app.update(Msg::ReasoningDelta("think".into()));
        assert_eq!(app.active_reasoning.as_deref(), Some("let me think"));
        assert!(
            app.history.is_empty(),
            "still streaming — nothing finalized yet"
        );
        // …then the answer streams; on finalize, reasoning lands ABOVE the answer.
        app.update(Msg::StreamDelta("the answer".into()));
        app.update(Msg::LlmDone {
            had_tool_calls: false,
            usage: TokenUsage::default(),
            ttft_ms: 0,
        });
        assert!(
            matches!(&app.history[0], Cell::Reasoning(t) if t == "let me think"),
            "reasoning cell must come first"
        );
        assert!(
            matches!(&app.history[1], Cell::Agent(t) if t == "the answer"),
            "the answer cell follows the reasoning"
        );
        assert_eq!(app.active_reasoning, None, "buffer cleared after flush");
    }

    #[test]
    fn streaming_then_lldone_finalizes_agent_cell() {
        let mut app = App::new("m");
        app.update(Msg::StreamDelta("Hel".into()));
        app.update(Msg::StreamDelta("lo".into()));
        assert_eq!(app.active.as_deref(), Some("Hello"));
        app.update(Msg::LlmDone {
            had_tool_calls: false,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
            ttft_ms: 0,
        });
        assert!(app.active.is_none());
        assert!(matches!(app.history.last(), Some(Cell::Agent(t)) if t == "Hello"));
        assert_eq!(app.tokens.input_tokens, 10);
        assert_eq!(app.tokens.output_tokens, 5);
    }

    #[test]
    fn text_turn_goes_idle_but_tool_turn_stays_running() {
        let mut app = App::new("m");
        app.running = true;
        // A turn that calls tools keeps the agent working.
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: true,
            ttft_ms: 0,
        });
        assert!(app.running, "tool turn should stay running");
        // A text-only turn means the agent now idles awaiting input.
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(!app.running, "text-only turn should go idle");
    }

    #[test]
    fn tool_lifecycle_running_then_completed() {
        let mut app = App::new("m");
        app.update(Msg::ToolStarted {
            id: "t1".into(),
            name: "bash".into(),
            input: "{}".into(),
            agent: "heartbit".into(),
        });
        assert!(matches!(
            app.history.last(),
            Some(Cell::Tool {
                status: ToolStatus::Running,
                ..
            })
        ));
        app.update(Msg::ToolCompleted {
            id: "t1".into(),
            is_error: false,
            output: "done".into(),
            duration_ms: 12,
        });
        match app.history.last() {
            Some(Cell::Tool {
                status,
                output,
                duration_ms,
                ..
            }) => {
                assert_eq!(*status, ToolStatus::Ok);
                assert_eq!(output.as_deref(), Some("done"));
                assert_eq!(*duration_ms, Some(12));
            }
            _ => panic!("expected finalized tool cell"),
        }
    }

    #[test]
    fn tool_preamble_is_finalized_before_tool_cell() {
        let mut app = App::new("m");
        app.update(Msg::StreamDelta("let me check".into()));
        app.update(Msg::ToolStarted {
            id: "t1".into(),
            name: "read".into(),
            input: "{}".into(),
            agent: "heartbit".into(),
        });
        // The streamed preamble became an Agent cell, then the tool cell.
        assert!(matches!(app.history.first(), Some(Cell::Agent(t)) if t == "let me check"));
        assert!(matches!(app.history.last(), Some(Cell::Tool { .. })));
        assert!(app.active.is_none());
    }

    #[test]
    fn approval_modal_opens_and_allows() {
        let mut app = App::new("m");
        let (tx, rx) = sync_channel(1);
        app.update(Msg::Approval {
            tools: vec![PendingTool {
                name: "bash".into(),
                input: "rm -rf".into(),
            }],
            reply: tx,
        });
        assert!(app.modal.is_some());
        app.update(key(KeyCode::Char('y')));
        assert!(app.modal.is_none());
        assert_eq!(rx.recv().unwrap(), ApprovalDecision::Allow);
    }

    #[test]
    fn approval_modal_denies_on_n() {
        let mut app = App::new("m");
        let (tx, rx) = sync_channel(1);
        app.update(Msg::Approval {
            tools: vec![PendingTool {
                name: "bash".into(),
                input: "x".into(),
            }],
            reply: tx,
        });
        app.update(key(KeyCode::Char('n')));
        assert_eq!(rx.recv().unwrap(), ApprovalDecision::Deny);
        assert!(app.modal.is_none());
    }

    #[test]
    fn keys_while_modal_open_do_not_reach_composer() {
        let mut app = App::new("m");
        let (tx, _rx) = sync_channel(1);
        app.update(Msg::Approval {
            tools: vec![],
            reply: tx,
        });
        typed(&mut app, "zzz"); // 'z' is not an answer key → ignored, not composed
        assert!(app.composer.is_empty());
    }

    #[test]
    fn esc_while_running_interrupts_and_finalizes() {
        let mut app = keyed();
        app.running = true;
        app.active = Some("partial reply".into());
        app.update(key(KeyCode::Esc));
        assert!(app.effects.contains(&Effect::Interrupt));
        assert!(!app.running, "interrupt returns to idle");
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Agent(t) if t == "partial reply")),
            "the streamed partial is finalized"
        );
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("interrupted"))),
        );
    }

    #[test]
    fn esc_while_idle_clears_composer_does_not_interrupt() {
        let mut app = keyed(); // not running
        typed(&mut app, "draft");
        app.update(key(KeyCode::Esc));
        assert!(app.composer.is_empty());
        assert!(
            !app.effects.contains(&Effect::Interrupt),
            "a stray Esc while idle must NOT cancel the next turn"
        );
    }

    #[test]
    fn ctrl_c_quits() {
        let mut app = App::new("m");
        app.update(ctrl('c'));
        assert!(app.should_quit);
        assert_eq!(app.effects, vec![Effect::Quit]);
    }

    #[test]
    fn run_failed_sets_idle_and_notice() {
        let mut app = App::new("m");
        app.running = true;
        app.update(Msg::RunFailed("boom".into()));
        assert!(!app.running);
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("boom"))),
            "failure notice present (the /learn tip may follow it)"
        );
    }

    #[test]
    fn todowrite_populates_the_todo_panel_model() {
        let mut app = keyed();
        app.update(Msg::ToolStarted {
            id: "t".into(),
            name: "todowrite".into(),
            input: serde_json::json!({"todos":[
                {"content":"build the thing","status":"in_progress","priority":"high"},
                {"content":"write tests","status":"pending","priority":"medium"},
                {"content":"done already","status":"completed","priority":"low"},
            ]})
            .to_string(),
            agent: "heartbit".into(),
        });
        assert_eq!(app.todos.len(), 3);
        assert_eq!(app.todos[0].content, "build the thing");
        assert_eq!(app.todos[0].status, TodoStatus::InProgress);
        assert_eq!(app.todos[2].status, TodoStatus::Completed);
        // a later todowrite replaces the list wholesale
        app.update(Msg::ToolStarted {
            id: "t2".into(),
            name: "todowrite".into(),
            input: serde_json::json!({"todos":[{"content":"only one","status":"pending","priority":"low"}]}).to_string(),
            agent: "heartbit".into(),
        });
        assert_eq!(app.todos.len(), 1);
        assert_eq!(app.todos[0].content, "only one");
    }

    #[test]
    fn parse_todos_is_robust_to_malformed_input() {
        assert!(parse_todos("not json").is_empty());
        assert!(parse_todos(r#"{"x":1}"#).is_empty());
        assert!(parse_todos(r#"{"todos":"nope"}"#).is_empty());
    }

    #[test]
    fn at_sign_triggers_file_walk_and_shows_mentions() {
        let mut app = keyed();
        app.file_index = vec![
            "src/main.rs".into(),
            "src/app.rs".into(),
            "README.md".into(),
        ];
        // typing '@' requests the walk once
        typed(&mut app, "look at @");
        assert!(app.files_requested);
        assert!(app.effects.contains(&Effect::WalkFiles));
        // a partial filters the index
        typed(&mut app, "app");
        let c = app.mention_candidates();
        assert_eq!(c, vec!["src/app.rs"]);
        assert!(app.menu_open(), "mention menu should be open");
    }

    #[test]
    fn mention_tab_completes_the_path_in_place() {
        let mut app = keyed();
        app.file_index = vec!["src/app.rs".into()];
        typed(&mut app, "edit @ap");
        // Tab completes the @token to the file path + a space
        app.update(Msg::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)));
        assert_eq!(app.composer.text(), "edit src/app.rs ");
        assert!(!app.menu_open(), "completion closes the menu");
    }

    #[test]
    fn ctrl_r_reverse_search_finds_and_loads_a_prior_prompt() {
        let mut app = keyed();
        // build some history
        for cmd in ["fix the parser", "add a test", "fix the renderer"] {
            typed(&mut app, cmd);
            app.update(key(KeyCode::Enter));
        }
        // open reverse search, type "fix"
        app.update(ctrl('r'));
        assert!(matches!(app.modal, Some(Modal::HistorySearch(_))));
        for c in "fix".chars() {
            app.update(key(KeyCode::Char(c)));
        }
        // newest-first: "fix the renderer" is the first match
        let m = app.history_matches("fix");
        assert_eq!(m, vec!["fix the renderer", "fix the parser"]);
        // Ctrl+R cycles to the 2nd match, Enter loads it
        app.update(ctrl('r'));
        app.update(key(KeyCode::Enter));
        assert_eq!(app.composer.text(), "fix the parser");
        assert!(app.modal.is_none(), "Enter closes the search");
    }

    #[test]
    fn slash_clear_empties_transcript_and_panels() {
        let mut app = keyed();
        app.history.push(Cell::User("hi".into()));
        app.todos = vec![TodoRow {
            content: "x".into(),
            status: TodoStatus::Pending,
        }];
        typed(&mut app, "/clear");
        app.update(key(KeyCode::Enter));
        assert!(app.todos.is_empty());
        // only the "cleared" notice remains
        assert_eq!(app.history.len(), 1);
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("cleared")));
    }

    #[test]
    fn slash_export_and_resume_emit_effects() {
        let mut app = keyed();
        typed(&mut app, "/export");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::ExportSession));
        typed(&mut app, "/resume");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::ListSessions));
    }

    #[test]
    fn slash_stats_pushes_compute_effect() {
        let mut app = keyed();
        typed(&mut app, "/stats");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::ComputeStats(None)));
        typed(&mut app, "/stats last");
        app.update(key(KeyCode::Enter));
        assert!(
            app.effects
                .contains(&Effect::ComputeStats(Some("last".into())))
        );
    }

    #[test]
    fn stats_ready_renders_a_stats_card() {
        let mut app = keyed();
        app.update(Msg::StatsReady(Ok((
            "t1".into(),
            Box::new(crate::trace_stats::TraceStats::default()),
        ))));
        assert!(matches!(app.history.last(), Some(Cell::Stats { label, .. }) if label == "t1"));
        app.update(Msg::StatsReady(Err("no trace".into())));
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("no trace")));
    }

    #[test]
    fn slash_diff_requests_the_working_tree_diff() {
        let mut app = keyed();
        typed(&mut app, "/diff");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::GitDiff));
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::SendInput(_))),
            "/diff is local — it must not consume an LLM turn"
        );
    }

    #[test]
    fn git_diff_ready_renders_a_diff_cell_and_empty_is_a_notice() {
        let mut app = keyed();
        app.update(Msg::GitDiffReady(Ok("@@ -1,1 +1,1 @@\n-a\n+b\n".into())));
        assert!(app.history.iter().any(|c| matches!(c, Cell::Diff { .. })));

        let mut app2 = keyed();
        app2.update(Msg::GitDiffReady(Ok(String::new())));
        assert!(
            app2.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("no changes")))
        );

        let mut app3 = keyed();
        app3.update(Msg::GitDiffReady(Err("not a git repository".into())));
        assert!(
            app3.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("git")))
        );
    }

    #[test]
    fn slash_analyze_pushes_analyze_effect() {
        let mut app = keyed();
        typed(&mut app, "/analyze");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::Analyze(None)));
        typed(&mut app, "/analyze last");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::Analyze(Some("last".into()))));
    }

    #[test]
    fn analyze_ready_starts_a_run_with_the_task() {
        let mut app = keyed();
        app.update(Msg::AnalyzeReady {
            display: "analyzing session s1".into(),
            task: "the big prompt".into(),
        });
        assert!(matches!(app.history.last(), Some(Cell::User(t)) if t.contains("s1")));
        assert!(app.running);
        assert!(
            app.effects
                .contains(&Effect::SendInput("the big prompt".into()))
        );
    }

    #[test]
    fn analyze_failed_is_a_notice_not_a_run() {
        let mut app = keyed();
        app.update(Msg::AnalyzeFailed("no trace".into()));
        assert!(!app.running);
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("no trace")));
    }

    #[test]
    fn slash_analyze_without_key_opens_key_modal_not_a_run() {
        let mut app = App::new("m"); // no key, no fallback
        typed(&mut app, "/analyze");
        app.update(key(KeyCode::Enter));
        assert!(
            matches!(app.modal, Some(Modal::KeyEntry(_))),
            "no-key /analyze must open the key prompt, not start a run"
        );
        assert!(!app.running);
        assert!(
            !app.effects.iter().any(|e| matches!(e, Effect::Analyze(_))),
            "must not prepare an analyze run without a provider"
        );
    }

    #[test]
    fn session_picker_loads_selected_session() {
        use crate::session::SessionMeta;
        let mut app = keyed();
        app.update(Msg::SessionsListed(vec![
            SessionMeta {
                id: "a".into(),
                preview: "first".into(),
                turns: 1,
            },
            SessionMeta {
                id: "b".into(),
                preview: "second".into(),
                turns: 2,
            },
        ]));
        assert!(matches!(app.modal, Some(Modal::SessionPicker(_))));
        app.update(key(KeyCode::Down)); // select "b"
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::ResumeSession("b".into())));
        assert!(app.modal.is_none());
        // and loading replaces history
        app.update(Msg::SessionLoaded(vec![Cell::User("restored".into())]));
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::User(t) if t == "restored"))
        );
    }

    #[test]
    fn shift_tab_cycles_permission_mode_and_emits_effect() {
        let mut app = keyed();
        assert_eq!(
            app.permission_mode,
            PermissionMode::Yolo,
            "YOLO is the default"
        );
        app.update(key(KeyCode::BackTab)); // wraps to Normal
        assert_eq!(app.permission_mode, PermissionMode::Normal);
        assert!(app.effects.contains(&Effect::SetPermissionMode(0)));
        app.update(key(KeyCode::BackTab)); // → Plan
        assert_eq!(app.permission_mode, PermissionMode::Plan);
        assert!(app.effects.contains(&Effect::SetPermissionMode(1)));
        app.update(key(KeyCode::BackTab)); // → YOLO
        assert_eq!(app.permission_mode, PermissionMode::Yolo);
        assert!(app.effects.contains(&Effect::SetPermissionMode(2)));
        // a notice records the change
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("mode")))
        );
    }

    #[test]
    fn plan_mode_prefixes_a_read_only_directive_but_keeps_display_clean() {
        let mut app = keyed();
        app.permission_mode = PermissionMode::Plan;
        typed(&mut app, "refactor the parser");
        app.update(key(KeyCode::Enter));
        // The SENT text carries the plan directive...
        let sent = app
            .effects
            .iter()
            .find_map(|e| match e {
                Effect::SendInput(t) => Some(t.clone()),
                _ => None,
            })
            .expect("a message was sent");
        assert!(
            sent.contains("PLAN MODE"),
            "sent text gets the plan directive"
        );
        assert!(sent.contains("refactor the parser"));
        // ...but the transcript shows the user's text verbatim (no directive).
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::User(t) if t == "refactor the parser")),
            "display stays clean"
        );
    }

    #[test]
    fn normal_mode_sends_text_unmodified() {
        let mut app = keyed();
        typed(&mut app, "hello");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::SendInput("hello".into())));
    }

    #[test]
    fn slash_mode_sets_named_mode() {
        let mut app = keyed();
        typed(&mut app, "/mode yolo");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.permission_mode, PermissionMode::Yolo);
        assert!(app.effects.contains(&Effect::SetPermissionMode(2)));
        typed(&mut app, "/mode plan");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.permission_mode, PermissionMode::Plan);
        typed(&mut app, "/mode normal");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.permission_mode, PermissionMode::Normal);
        // bad arg → a usage notice, mode unchanged
        typed(&mut app, "/mode bogus");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.permission_mode, PermissionMode::Normal);
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("usage: /mode")))
        );
    }

    #[test]
    fn mouse_wheel_scrolls_transcript_not_command_history() {
        let mut app = keyed();
        // type a command so there'd be composer history to (wrongly) move
        typed(&mut app, "earlier");
        app.update(key(KeyCode::Enter));
        let composed_before = app.composer.text();
        // A frame establishes the viewport height (max_off = 100 hidden rows).
        app.scroll_offset(100);
        // Wheel up scrolls the transcript output, leaving the composer untouched.
        app.update(Msg::WheelUp);
        app.update(Msg::WheelUp);
        assert!(!app.follow, "scrolling up unpins from the bottom");
        assert_eq!(
            app.composer.text(),
            composed_before,
            "the wheel must NOT touch the composer (command history stays on ↑/↓)"
        );
    }

    #[test]
    fn pageup_scrolls_back_pagedown_returns() {
        let mut app = App::new("m");
        app.scroll_offset(100);
        app.update(key(KeyCode::PageUp));
        assert!(!app.follow, "PageUp unpins");
        // Page back down to the bottom re-pins to follow.
        for _ in 0..20 {
            app.scroll_offset(100);
            app.update(key(KeyCode::PageDown));
        }
        assert!(app.follow, "reaching the bottom re-pins to follow");
    }

    #[test]
    fn streaming_does_not_yank_view_when_user_scrolled_up() {
        let mut app = App::new("m");
        app.scroll_offset(100);
        app.update(Msg::WheelUp);
        assert!(!app.follow);
        // Answer deltas stream in — the view must STAY where the user parked it
        // (no snap to bottom).
        app.update(Msg::StreamDelta("answer".into()));
        app.update(Msg::StreamDelta(" more".into()));
        assert!(
            !app.follow,
            "streaming must not re-pin the view while the user is reading history"
        );
    }

    #[test]
    fn scrolled_up_stays_put_while_reasoning_streams_verbosely() {
        // The exact scenario the user reported: scroll up mid-turn while a
        // reasoning model streams a long chain-of-thought at the bottom. Drives
        // both new features together — streaming growth + the scroll model.
        let mut app = App::new("m");
        app.scroll_offset(100); // a frame: following, max_off = 100
        app.update(Msg::WheelUp); // park the view up
        assert!(!app.follow);
        let parked = app.scroll_offset(100);
        // 30 lines of reasoning stream in, growing the transcript at the bottom…
        for _ in 0..30 {
            app.update(Msg::ReasoningDelta("a thinking line\n".into()));
        }
        // …a later frame sees far more content (max_off jumps to 200).
        let after = app.scroll_offset(200);
        assert!(!app.follow, "must remain unpinned while reasoning streams");
        assert_eq!(
            parked, after,
            "top-anchored: the read position must not move as reasoning streams below"
        );
    }

    #[test]
    fn failed_turn_flushes_reasoning_and_does_not_leak_into_next_turn() {
        let mut app = App::new("m");
        app.update(Msg::ReasoningDelta("turn-one thought".into()));
        // The turn fails before the answer — the buffer must be flushed, not kept.
        app.update(Msg::RunFailed("boom".into()));
        assert_eq!(
            app.active_reasoning, None,
            "a failed turn must not leave reasoning buffered for the next turn"
        );
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Reasoning(t) if t == "turn-one thought")),
            "the partial thinking settles into history rather than vanishing"
        );
        // The next turn's reasoning starts clean (no concatenation onto stale text).
        app.update(Msg::ReasoningDelta("turn-two thought".into()));
        assert_eq!(app.active_reasoning.as_deref(), Some("turn-two thought"));
    }

    #[test]
    fn scrolled_up_view_does_not_drift_as_content_grows() {
        let mut app = App::new("m");
        app.scroll_offset(100);
        app.update(Msg::WheelUp);
        let off_before = app.scroll_offset(100);
        // A long reasoning stream adds 50 rows at the bottom.
        let off_after = app.scroll_offset(150);
        assert_eq!(
            off_before, off_after,
            "top-anchored while scrolled up: the read position must not drift down"
        );
    }

    #[test]
    fn following_stays_pinned_to_bottom_as_content_grows() {
        let app = App::new("m");
        // Following (default) → offset tracks the bottom regardless of growth.
        assert_eq!(app.scroll_offset(100), 100);
        assert_eq!(app.scroll_offset(150), 150);
    }

    #[test]
    fn sending_a_message_re_pins_to_bottom() {
        let mut app = keyed();
        app.scroll_offset(100);
        app.update(Msg::WheelUp);
        assert!(!app.follow);
        typed(&mut app, "hello");
        app.update(key(KeyCode::Enter));
        assert!(app.follow, "submitting a message jumps back to the newest");
    }

    #[test]
    fn slash_learn_pushes_learn_effect_with_key() {
        let mut app = keyed();
        typed(&mut app, "/learn");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::Learn));
    }

    #[test]
    fn slash_learn_mid_run_or_reentrant_is_refused() {
        // While a turn is in flight, the in-flight LlmDone would consume the
        // learn digest EARLY and the real learn result would be silently
        // dropped — /learn must refuse instead.
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "/learn");
        app.update(key(KeyCode::Enter));
        assert!(!app.effects.contains(&Effect::Learn));
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("current turn")));
        // Re-entry while a learn is already armed is refused too.
        let mut app = keyed();
        app.learning = Some(1);
        typed(&mut app, "/learn");
        app.update(key(KeyCode::Enter));
        assert!(!app.effects.contains(&Effect::Learn));
    }

    #[test]
    fn slash_learn_without_key_opens_key_modal_not_a_run() {
        let mut app = App::new("m");
        typed(&mut app, "/learn");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.modal, Some(Modal::KeyEntry(_))));
        assert!(!app.running);
        assert!(!app.effects.contains(&Effect::Learn));
    }

    #[test]
    fn slash_workflows_lists_recipes() {
        let mut app = keyed();
        app.workflow_recipes = vec![
            ("parallel_review".into(), "Review a target.".into()),
            ("deep_research".into(), "Research-first deep dive.".into()),
        ];
        typed(&mut app, "/workflows");
        app.update(key(KeyCode::Enter));
        let notices: Vec<String> = app
            .history
            .iter()
            .filter_map(|c| match c {
                Cell::Notice(n) => Some(n.clone()),
                _ => None,
            })
            .collect();
        assert!(
            notices.iter().any(|n| n.contains("parallel_review")),
            "{notices:?}"
        );
        assert!(
            notices.iter().any(|n| n.contains("deep_research")),
            "{notices:?}"
        );
        assert!(
            notices.iter().any(|n| n.contains("run_workflow")),
            "must say HOW they are reached: {notices:?}"
        );
        assert!(!app.running, "informational — no run starts");
    }

    #[test]
    fn bare_mode_opens_picker_preselected_on_current() {
        let mut app = keyed();
        app.permission_mode = PermissionMode::Plan;
        typed(&mut app, "/mode");
        app.update(key(KeyCode::Enter));
        assert!(
            matches!(app.modal, Some(Modal::ModePicker { sel: 1 })),
            "picker must open on the CURRENT mode (Plan = index 1)"
        );
    }

    #[test]
    fn mode_picker_enter_applies_esc_cancels_and_wraps() {
        let mut app = keyed();
        app.modal = Some(Modal::ModePicker { sel: 0 });
        app.update(key(KeyCode::Up)); // wrap 0 → 2 (YOLO)
        assert!(matches!(app.modal, Some(Modal::ModePicker { sel: 2 })));
        app.update(key(KeyCode::Enter));
        assert_eq!(app.permission_mode, PermissionMode::Yolo);
        assert!(app.modal.is_none());
        assert!(app.effects.contains(&Effect::SetPermissionMode(2)));
        // Esc path: no change, no effect.
        let mut app = keyed();
        app.modal = Some(Modal::ModePicker { sel: 2 });
        app.update(key(KeyCode::Esc));
        assert!(app.modal.is_none());
        assert_eq!(app.permission_mode, PermissionMode::Yolo); // untouched default
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::SetPermissionMode(_)))
        );
    }

    #[test]
    fn mode_with_arg_still_sets_directly() {
        let mut app = keyed();
        typed(&mut app, "/mode yolo");
        app.update(key(KeyCode::Enter));
        assert!(app.modal.is_none(), "arg path must NOT open the picker");
        assert_eq!(app.permission_mode, PermissionMode::Yolo);
    }

    #[test]
    fn splash_auto_dismisses_after_its_ticks() {
        let mut app = keyed();
        app.splash = Some(0);
        for _ in 0..(crate::splash::SPLASH_TICKS - 1) {
            app.update(Msg::Tick);
        }
        assert!(app.splash.is_some(), "still up one tick before the end");
        app.update(Msg::Tick);
        assert_eq!(app.splash, None, "gone at SPLASH_TICKS");
    }

    #[test]
    fn splash_key_dismisses_and_is_consumed() {
        let mut app = keyed();
        app.splash = Some(2);
        app.update(key(KeyCode::Char('h')));
        assert_eq!(app.splash, None, "any key dismisses");
        assert_eq!(app.composer.text(), "", "the dismissing key must NOT type");
        app.update(key(KeyCode::Char('h')));
        assert_eq!(app.composer.text(), "h", "subsequent keys flow normally");
    }

    #[test]
    fn multiline_paste_lands_as_one_draft_and_does_not_submit() {
        let mut app = keyed();
        app.update(Msg::Paste("line one\nline two\nline three".into()));
        assert_eq!(app.composer.text(), "line one\nline two\nline three");
        // A paste NEVER submits.
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::SendInput(_)))
        );
        assert!(app.history.is_empty());
    }

    #[test]
    fn paste_mid_draft_preserves_the_tail_and_leaves_cursor_after_insert() {
        let mut app = keyed();
        typed(&mut app, "ab");
        app.update(key(KeyCode::Left)); // cursor between a and b
        app.update(Msg::Paste("X\nY".into()));
        assert_eq!(app.composer.text(), "aX\nYb");
    }

    #[test]
    fn crlf_paste_yields_single_newlines() {
        let mut app = keyed();
        app.update(Msg::Paste("a\r\nb".into()));
        assert_eq!(app.composer.text(), "a\nb");
    }

    #[test]
    fn paste_during_splash_dismisses_the_overlay_and_keeps_the_text() {
        let mut app = keyed();
        app.splash = Some(0);
        app.update(Msg::Paste("hello".into()));
        assert!(app.splash.is_none(), "the paste must dismiss the splash");
        assert_eq!(app.composer.text(), "hello");
    }

    #[test]
    fn focus_defaults_to_focused_and_tracks_both_directions() {
        let mut app = App::new("m");
        assert!(
            app.focused,
            "a terminal that never reports focus must read as focused"
        );
        app.update(Msg::FocusChanged(false));
        assert!(!app.focused);
        app.update(Msg::FocusChanged(true));
        assert!(app.focused);
    }

    fn unfocused_running() -> App {
        let mut app = keyed();
        app.notify = true;
        app.focused = false;
        app.running = true;
        app
    }

    fn notified(app: &App) -> bool {
        app.effects
            .iter()
            .any(|e| matches!(e, Effect::Notify { .. }))
    }

    #[test]
    fn notify_on_turn_idle_when_unfocused() {
        let mut app = unfocused_running();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(notified(&app));
    }

    #[test]
    fn focused_terminal_suppresses_notify() {
        let mut app = unfocused_running();
        app.focused = true;
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(!notified(&app));
    }

    #[test]
    fn notify_disabled_by_config_suppresses() {
        let mut app = unfocused_running();
        app.notify = false;
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(!notified(&app));
    }

    #[test]
    fn tool_turn_does_not_notify() {
        let mut app = unfocused_running();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: true,
            ttft_ms: 0,
        });
        assert!(!notified(&app));
    }

    #[test]
    fn at_most_one_notify_per_turn() {
        let mut app = unfocused_running();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        app.effects.clear();
        app.update(Msg::RunCompleted);
        assert!(
            !notified(&app),
            "RunCompleted must not re-notify after LlmDone"
        );
    }

    #[test]
    fn notify_suppressed_during_splash() {
        let mut app = unfocused_running();
        app.splash = Some(0);
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(!notified(&app));
    }

    // --- Task 7 follow-up: the sites the six brief tests above don't drive
    // (approval, RunFailed's error content, drained-queue suppression) ---

    #[test]
    fn approval_notifies_when_unfocused() {
        let mut app = unfocused_running();
        let (tx, _rx) = sync_channel(1);
        app.update(Msg::Approval {
            tools: vec![PendingTool {
                name: "bash".into(),
                input: "rm -rf".into(),
            }],
            reply: tx,
        });
        assert!(notified(&app));
    }

    #[test]
    fn approval_while_focused_does_not_notify() {
        let mut app = unfocused_running();
        app.focused = true;
        let (tx, _rx) = sync_channel(1);
        app.update(Msg::Approval {
            tools: vec![PendingTool {
                name: "bash".into(),
                input: "rm -rf".into(),
            }],
            reply: tx,
        });
        assert!(!notified(&app));
    }

    #[test]
    fn run_failed_notifies_with_the_error() {
        let mut app = unfocused_running();
        app.update(Msg::RunFailed("boom: provider timeout".into()));
        let body = app.effects.iter().find_map(|e| match e {
            Effect::Notify { body, .. } => Some(body.clone()),
            _ => None,
        });
        assert_eq!(
            body.as_deref(),
            Some("boom: provider timeout"),
            "the error must reach the notify body, read out before `error` moves into EmergencyHandoff"
        );
    }

    #[test]
    fn drained_queue_suppresses_the_turn_end_notify() {
        let mut app = unfocused_running();
        // `unfocused_running()` already sets `running = true`, so this submit
        // queues (via the real send_or_queue choke point) instead of sending.
        typed(&mut app, "next thing");
        app.update(key(KeyCode::Enter));
        app.effects.clear();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(
            !notified(&app),
            "a drained queue starts a fresh turn — no notify"
        );
        assert!(
            app.running,
            "the drained message should have restarted a turn"
        );
    }

    #[test]
    fn research_slug_is_safe_and_bounded() {
        assert_eq!(
            research_slug("How does Plate Solving work?"),
            "how-does-plate-solving-work"
        );
        assert_eq!(research_slug("éàç!!"), "research");
        assert!(research_slug(&"x".repeat(200)).len() <= 40);
    }

    #[test]
    fn slash_research_builds_the_imperative_task() {
        let mut app = keyed();
        typed(&mut app, "/research plate solving algorithms");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.history.last(), Some(Cell::User(t)) if t.contains("researching")));
        assert!(app.running);
        let task = app
            .effects
            .iter()
            .find_map(|e| match e {
                Effect::SendInput(t) => Some(t.clone()),
                _ => None,
            })
            .expect("task sent");
        assert!(task.contains("run_workflow"), "{task}");
        // The tool's schema param is `recipe` (required) — the template must
        // name it exactly, not paraphrase it (a weaker model won't bridge it).
        assert!(task.contains("recipe=\"deep_research\""), "{task}");
        assert!(
            task.contains("research-plate-solving-algorithms.md"),
            "{task}"
        );
        assert!(task.to_lowercase().contains("do not improvise"), "{task}");
    }

    #[test]
    fn slash_research_empty_arg_is_usage_no_key_is_modal() {
        let mut app = keyed();
        typed(&mut app, "/research");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("usage")));
        assert!(!app.running);
        let mut app = App::new("m");
        typed(&mut app, "/research topic");
        app.update(key(KeyCode::Enter));
        assert!(matches!(app.modal, Some(Modal::KeyEntry(_))));
        assert!(!app.running);
    }

    #[test]
    fn learn_ready_starts_run_and_arms_the_commit() {
        let mut app = keyed();
        app.update(Msg::LearnReady {
            display: "learning from 2 diagnoses".into(),
            task: "the prompt".into(),
            staged_digest: 42,
        });
        assert!(matches!(app.history.last(), Some(Cell::User(t)) if t.contains("diagnoses")));
        assert!(app.running);
        assert_eq!(app.learning, Some(42));
        assert!(
            app.effects
                .contains(&Effect::SendInput("the prompt".into()))
        );
    }

    #[test]
    fn turn_idle_llmdone_commits_once_and_disarms() {
        let mut app = keyed();
        app.learning = Some(42);
        app.running = true;
        // tool-use turn must NOT commit
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: true,
            ttft_ms: 0,
        });
        assert_eq!(app.learning, Some(42));
        assert!(!app.effects.contains(&Effect::CommitLessons(42)));
        // text-only turn (turn-idle) commits and disarms
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert_eq!(app.learning, None);
        assert!(app.effects.contains(&Effect::CommitLessons(42)));
        // a second idle turn must not commit again
        app.effects.clear();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::CommitLessons(_)))
        );
    }

    #[test]
    fn run_completed_is_a_commit_backstop() {
        let mut app = keyed();
        app.learning = Some(7);
        app.update(Msg::RunCompleted);
        assert_eq!(app.learning, None);
        assert!(app.effects.contains(&Effect::CommitLessons(7)));
    }

    #[test]
    fn run_failed_and_interrupt_disarm_without_commit() {
        let mut app = keyed();
        app.learning = Some(7);
        app.update(Msg::RunFailed("boom".into()));
        assert_eq!(app.learning, None);
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::CommitLessons(_)))
        );
        // Esc-interrupt: the synthetic LlmDone that follows must find the flag cleared
        app.learning = Some(8);
        app.running = true;
        app.update(key(KeyCode::Esc));
        assert_eq!(
            app.learning, None,
            "Esc must disarm before the synthetic LlmDone"
        );
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::CommitLessons(_)))
        );
    }

    #[test]
    fn learn_failed_is_a_notice_not_a_run() {
        let mut app = keyed();
        app.update(Msg::LearnFailed(
            "no diagnosis found — run /analyze first".into(),
        ));
        assert!(!app.running);
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("/analyze")));
    }

    // --- Task 6: visible input queue ---

    #[test]
    fn submit_while_running_queues_instead_of_sending() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "second thing");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.queued.len(), 1);
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::SendInput(_)))
        );
    }

    #[test]
    fn queued_message_drains_at_turn_idle_as_a_user_cell() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "queued one");
        app.update(key(KeyCode::Enter));
        app.effects.clear();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(app.queued.is_empty());
        assert!(
            app.effects
                .contains(&Effect::SendInput("queued one".into()))
        );
    }

    #[test]
    fn turn_idle_drains_only_one_queued_message() {
        let mut app = keyed();
        app.running = true;
        for t in ["a", "b"] {
            typed(&mut app, t);
            app.update(key(KeyCode::Enter));
        }
        app.effects.clear();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert_eq!(
            app.queued.len(),
            1,
            "releasing several would re-hide the rest"
        );
    }

    #[test]
    fn tool_calling_llm_done_does_not_drain_the_queue() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "later");
        app.update(key(KeyCode::Enter));
        app.effects.clear();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: true,
            ttft_ms: 0,
        });
        assert_eq!(app.queued.len(), 1, "the turn is not over — a tool is next");
    }

    #[test]
    fn run_failed_and_agent_exit_drop_the_queue() {
        for msg in [Msg::RunFailed("boom".into()), Msg::AgentExited(1)] {
            let mut app = keyed();
            app.running = true;
            typed(&mut app, "stranded");
            app.update(key(KeyCode::Enter));
            app.update(msg);
            assert!(
                app.queued.is_empty(),
                "a failed turn must not strand the queue"
            );
        }
    }

    #[test]
    fn queue_is_empty_whenever_the_turn_is_idle() {
        let mut app = keyed();
        assert!(app.queued.is_empty() && !app.running);
        typed(&mut app, "immediate");
        app.update(key(KeyCode::Enter));
        assert!(app.queued.is_empty(), "an idle submit sends, never queues");
        assert!(app.effects.contains(&Effect::SendInput("immediate".into())));
    }

    #[test]
    fn up_arrow_pops_the_newest_queued_message_for_editing() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "edit me");
        app.update(key(KeyCode::Enter));
        app.update(key(KeyCode::Up));
        assert!(app.queued.is_empty());
        assert_eq!(app.composer.text(), "edit me");
    }

    #[test]
    fn up_arrow_with_an_empty_queue_still_recalls_prompt_history() {
        // Idle-Up must keep its pre-existing meaning (recall) when there is
        // nothing queued — only a non-empty queue changes what Up does.
        let mut app = keyed();
        typed(&mut app, "first prompt");
        app.update(key(KeyCode::Enter)); // idle submit — sends immediately, never queues
        assert!(app.queued.is_empty());
        app.composer.seed_history(vec!["first prompt".into()]);
        app.update(key(KeyCode::Up));
        assert_eq!(app.composer.text(), "first prompt");
    }

    #[test]
    fn esc_drops_the_queue_without_interrupting_the_running_turn() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "queued while busy");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.queued.len(), 1);
        app.effects.clear();
        app.update(key(KeyCode::Esc));
        assert!(app.queued.is_empty(), "Esc drops the backlog");
        assert!(app.running, "the in-flight turn itself is untouched");
        assert!(
            !app.effects.contains(&Effect::Interrupt),
            "a queue-drop is not an interrupt"
        );
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("dropped"))),
            "the drop must be visible, not silent"
        );
    }

    #[test]
    fn esc_twice_drops_then_interrupts() {
        // Review rename (2026-07-30): `interrupt()`'s own internal
        // `drop_queued()` call is unreachable — its single caller (the Esc
        // key handler) already routes a non-empty queue to `drop_queued()`
        // directly and only calls `interrupt()` once the queue is empty. This
        // test proves that two-press Esc behavior at the KEY-HANDLER level,
        // not that `interrupt()`'s internal call fires — it never does.
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "stranded by interrupt");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.queued.len(), 1);
        app.update(key(KeyCode::Esc)); // queue non-empty → drops the queue, not an interrupt
        assert!(app.queued.is_empty());
        assert!(app.running);
        // A second Esc now finds an empty queue — falls through to interrupt.
        app.update(key(KeyCode::Esc));
        assert!(!app.running);
    }

    #[test]
    fn run_completed_drains_one_queued_message() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "queued for completion");
        app.update(key(KeyCode::Enter));
        app.effects.clear();
        app.update(Msg::RunCompleted);
        assert!(app.queued.is_empty());
        assert!(
            app.effects
                .contains(&Effect::SendInput("queued for completion".into()))
        );
        assert!(app.running, "the drained message starts a new turn");
    }

    #[test]
    fn drain_re_arms_follow_so_the_new_reply_is_visible() {
        // Review F2 (2026-07-30): dropping `follow` on a mid-turn SUBMIT is
        // fine (the user was scrolled up reading); the cost lands at DRAIN,
        // where a brand-new turn begins and its reply must not stream
        // off-screen while the view stays wherever the user left it.
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "queued while scrolled up");
        app.update(key(KeyCode::Enter));
        // Scroll away from the bottom — mirrors `sending_a_message_re_pins_to_bottom`.
        app.scroll_offset(100);
        app.update(Msg::WheelUp);
        assert!(!app.follow, "scrolling up unpins from the bottom");
        app.update(Msg::RunCompleted);
        assert!(
            app.follow,
            "a drained message starts a new turn — its reply must be visible"
        );
    }

    #[test]
    fn mid_turn_analyze_ready_queues_instead_of_bypassing() {
        // Review F3: only `submit()` had mid-turn coverage — the other six
        // choke-point senders could regress to a direct `effects.push(
        // Effect::SendInput(..))` and every existing idle-only test (e.g.
        // `analyze_ready_starts_a_run_with_the_task`) would stay green. This
        // guards the pattern on a second, representative sender.
        let mut app = keyed();
        app.running = true;
        app.update(Msg::AnalyzeReady {
            display: "analyzing session s1".into(),
            task: "the big prompt".into(),
        });
        assert_eq!(
            app.queued.len(),
            1,
            "AnalyzeReady must queue, not bypass, mid-turn"
        );
        assert!(
            !app.effects
                .iter()
                .any(|e| matches!(e, Effect::SendInput(_))),
            "no direct send while a turn is already running"
        );
    }

    #[test]
    fn plan_mode_queued_message_keeps_display_clean_when_drained() {
        // A single queued String can't carry "clean display" and "plan-
        // prefixed wire payload" separately — a mid-turn Plan-mode submit
        // must NOT leak the internal directive into the transcript once
        // drained (advisor finding: the idle path already gets this right
        // via `plan_mode_prefixes_a_read_only_directive_but_keeps_display_clean`,
        // but that test never exercises `was_idle == false`).
        let mut app = keyed();
        app.permission_mode = PermissionMode::Plan;
        app.running = true;
        typed(&mut app, "check the parser");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.queued.len(), 1);
        app.effects.clear();
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
            ttft_ms: 0,
        });
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::User(t) if t == "check the parser")),
            "the drained display must be the user's clean text, not the internal \
             directive: {:?}",
            app.history
        );
        let sent = app
            .effects
            .iter()
            .find_map(|e| match e {
                Effect::SendInput(t) => Some(t.clone()),
                _ => None,
            })
            .expect("a message was sent");
        assert!(
            sent.contains("PLAN MODE"),
            "the wire payload keeps the directive"
        );
        assert!(sent.contains("check the parser"));
    }

    #[test]
    fn up_arrow_pops_the_clean_display_even_in_plan_mode() {
        let mut app = keyed();
        app.permission_mode = PermissionMode::Plan;
        app.running = true;
        typed(&mut app, "check the parser");
        app.update(key(KeyCode::Enter));
        app.update(key(KeyCode::Up));
        assert_eq!(
            app.composer.text(),
            "check the parser",
            "no PLAN MODE directive leaking into the composer on edit"
        );
    }
}
