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

/// Live state of one agent in the multi-agent roster panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentState {
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

/// Global permission posture, cycled with Shift+Tab (Claude Code style). Gates
/// the `on_approval` bridge so the user can pre-set how consequential tool calls
/// are handled for the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    /// Ask (the approval modal) for tools that aren't silently allowed by rules.
    Default,
    /// Auto-allow file edits (edit/write/patch); still ask for bash & others.
    AcceptEdits,
    /// Read-only posture: auto-deny mutating tools (edit/write/patch/bash).
    Plan,
    /// Auto-allow everything (bypass).
    Auto,
}

impl PermissionMode {
    /// Next mode in the Shift+Tab cycle.
    pub fn next(self) -> Self {
        match self {
            PermissionMode::Default => PermissionMode::AcceptEdits,
            PermissionMode::AcceptEdits => PermissionMode::Plan,
            PermissionMode::Plan => PermissionMode::Auto,
            PermissionMode::Auto => PermissionMode::Default,
        }
    }
    /// Compact wire value shared with the (cross-thread) approval gate.
    pub fn as_u8(self) -> u8 {
        match self {
            PermissionMode::Default => 0,
            PermissionMode::AcceptEdits => 1,
            PermissionMode::Plan => 2,
            PermissionMode::Auto => 3,
        }
    }
    /// Short status-line label.
    pub fn label(self) -> &'static str {
        match self {
            PermissionMode::Default => "default",
            PermissionMode::AcceptEdits => "accept-edits",
            PermissionMode::Plan => "plan",
            PermissionMode::Auto => "auto",
        }
    }
}

/// Slash commands offered by the `/` autocomplete menu: (name, description).
pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "list commands"),
    ("/model", "show or set the model"),
    ("/mcp", "list / add / clear MCP servers"),
    ("/agents", "toggle multi-agent workflow mode"),
    ("/key", "set the OpenRouter API key"),
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
    /// Persist the MCP server list to the config file.
    SaveMcp(Vec<McpServerSpec>),
    /// Fetch the OpenRouter model catalog (async, for the model picker).
    FetchModels,
    /// Persist the multi-agent (orchestrator) mode flag to the config file.
    SaveMultiAgent(bool),
    /// Apply the permission mode to the shared (cross-thread) approval gate.
    SetPermissionMode(u8),
    /// Abandon the in-flight turn (abort generation), keeping the session.
    Interrupt,
    /// Tear down and exit.
    Quit,
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
}

/// A modal overlay.
pub enum Modal {
    Approval(ApprovalModal),
    KeyEntry(KeyEntryModal),
    ModelPicker(ModelPicker),
}

/// The full UI state.
pub struct App {
    pub history: Vec<Cell>,
    /// Assistant text being streamed for the current turn (not yet finalized).
    pub active: Option<String>,
    pub composer: Composer,
    pub modal: Option<Modal>,
    pub model: String,
    /// The OpenRouter API key in effect (from env, config, or set in-TUI).
    pub api_key: Option<String>,
    /// True when a provider can start without an OpenRouter key (e.g. an
    /// `ANTHROPIC_API_KEY` env fallback) — so a no-key submit need not prompt.
    pub has_fallback_provider: bool,
    pub tokens: TokenUsage,
    /// MCP servers to connect when the agent starts (mirrors the config file).
    pub mcp_servers: Vec<McpServerSpec>,
    /// The OpenRouter model catalog (lazily fetched for the picker).
    pub models: Vec<ModelEntry>,
    /// True while the catalog fetch is in flight.
    pub models_loading: bool,
    /// Multi-agent orchestrator mode (applies on next agent start).
    pub multi_agent: bool,
    /// Live roster of agents for the current turn (multi-agent mode): who was
    /// instantiated and what each is doing right now. Ordered by first-seen.
    pub agents: Vec<AgentRow>,
    /// Live task list mirrored from the agent's latest `todowrite` call.
    pub todos: Vec<TodoRow>,
    /// Global permission posture (Shift+Tab cycles it).
    pub permission_mode: PermissionMode,
    /// Current context fill (latest request's input tokens) — for the status bar.
    pub context_tokens: u32,
    /// Time-to-first-token of the latest turn (ms) — status-line throughput.
    pub last_ttft_ms: u64,
    pub running: bool,
    /// Lines scrolled up from the bottom (0 = pinned to newest).
    pub scroll: u16,
    /// Highlighted row in the `/` command-autocomplete menu.
    pub menu_selected: usize,
    pub spinner: usize,
    pub should_quit: bool,
    pub effects: Vec<Effect>,
    /// Maps an in-flight tool_call_id to its index in `history`.
    tool_index: HashMap<String, usize>,
}

impl App {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            history: Vec::new(),
            active: None,
            composer: Composer::new(),
            modal: None,
            model: model.into(),
            api_key: None,
            has_fallback_provider: false,
            tokens: TokenUsage::default(),
            mcp_servers: Vec::new(),
            models: Vec::new(),
            models_loading: false,
            multi_agent: false,
            agents: Vec::new(),
            todos: Vec::new(),
            permission_mode: PermissionMode::Default,
            context_tokens: 0,
            last_ttft_ms: 0,
            running: false,
            scroll: 0,
            menu_selected: 0,
            spinner: 0,
            should_quit: false,
            effects: Vec::new(),
            tool_index: HashMap::new(),
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
        if let Some(text) = self.active.take() {
            let trimmed = text.trim_end();
            if !trimmed.is_empty() {
                self.history.push(Cell::Agent(trimmed.to_string()));
            }
        }
    }

    /// Find or create an agent row (first-seen order), and set it Working with
    /// the given activity. Only tracked in multi-agent mode.
    fn agent_set_working(&mut self, name: &str, activity: impl Into<String>) {
        if !self.multi_agent {
            return;
        }
        let activity = activity.into();
        if let Some(row) = self.agents.iter_mut().find(|r| r.name == name) {
            // Don't resurrect a finished agent from a late event.
            if row.state == AgentState::Working {
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
        if !self.multi_agent {
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

    /// The agent badge to stamp on a tool cell: the agent name in multi-agent
    /// mode (so the transcript shows who ran each tool), `None` in single mode.
    fn agent_badge(&self, agent: &str) -> Option<String> {
        if self.multi_agent {
            Some(agent.to_string())
        } else {
            None
        }
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
            Msg::Tick => self.spinner = self.spinner.wrapping_add(1),
            Msg::Resize => {}
            // Mouse wheel scrolls the transcript (output history). Over-scrolling
            // is harmless — the renderer clamps the offset to the top.
            Msg::WheelUp => self.scroll = self.scroll.saturating_add(WHEEL_STEP),
            Msg::WheelDown => self.scroll = self.scroll.saturating_sub(WHEEL_STEP),
            Msg::Paste(s) => match &mut self.modal {
                // Pasting into a prompt must land in that field, not the composer
                // hidden behind the modal.
                Some(Modal::KeyEntry(m)) => m.input.push_str(&s.replace(['\n', '\r'], "")),
                Some(Modal::ModelPicker(p)) => {
                    p.query.push_str(&s.replace(['\n', '\r'], ""));
                    p.selected = 0;
                }
                Some(Modal::Approval(_)) => {}
                None => self.composer.insert_str(&s),
            },
            Msg::Key(key) => {
                if self.modal.is_some() {
                    self.handle_modal_key(key);
                } else {
                    self.handle_key(key);
                }
            }

            Msg::TurnStarted => self.running = true,
            Msg::StreamDelta(s) => {
                self.running = true;
                self.scroll = 0; // autoscroll to newest while streaming
                self.active.get_or_insert_with(String::new).push_str(&s);
            }
            Msg::LlmDone {
                usage,
                had_tool_calls,
                ttft_ms,
            } => {
                self.finalize_active();
                self.tokens.input_tokens =
                    self.tokens.input_tokens.saturating_add(usage.input_tokens);
                self.tokens.output_tokens = self
                    .tokens
                    .output_tokens
                    .saturating_add(usage.output_tokens);
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
                }
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
                self.scroll = 0;
            }
            Msg::AgentsDispatched(names) => {
                for n in &names {
                    self.agent_set_working(n, "dispatched");
                }
                if !names.is_empty() {
                    self.history.push(Cell::Notice(format!(
                        "→ delegating to {}",
                        names.join(", ")
                    )));
                }
            }
            Msg::SubAgentDone {
                agent,
                success,
                tokens,
            } => self.agent_finish(&agent, success, tokens),
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
            Msg::RunCompleted | Msg::AgentExited(_) => {
                self.finalize_active();
                self.running = false;
            }
            Msg::RunFailed(error) => {
                self.finalize_active();
                self.running = false;
                self.history
                    .push(Cell::Notice(format!("run failed: {error}")));
            }
            Msg::Approval { tools, reply } => {
                self.modal = Some(Modal::Approval(ApprovalModal { tools, reply }));
            }
            Msg::ModelsLoaded(models) => {
                self.models = models;
                self.models_loading = false;
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

    /// Whether the `/` command menu is currently showing.
    pub fn menu_open(&self) -> bool {
        !self.command_candidates().is_empty()
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
        let n = self.command_candidates().len();
        if n == 0 {
            return;
        }
        let cur = self.menu_selected.min(n - 1) as isize;
        self.menu_selected = (cur + delta).rem_euclid(n as isize) as usize;
    }

    /// Tab: complete to the selected command + a trailing space (ready for args).
    fn menu_complete(&mut self) {
        if let Some(name) = self.menu_selected_command() {
            self.composer.set_text(&format!("{name} "));
            self.menu_selected = 0;
        }
    }

    /// Enter on the menu: complete to the selected command and run it now.
    fn menu_run(&mut self) {
        if let Some(name) = self.menu_selected_command() {
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
        self.history.push(Cell::User(text.clone()));
        self.running = true;
        self.scroll = 0;
        self.agents.clear(); // fresh roster for this turn
        self.effects.push(Effect::SendInput(text));
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
                    self.open_model_picker();
                } else {
                    self.set_model(arg);
                }
            }
            "mcp" => self.handle_mcp(arg),
            "agents" | "agent" | "workflow" => self.toggle_multi_agent(arg),
            "help" => {
                self.history.push(Cell::Notice(
                    "commands: /key [token] · /model [name] · /mcp [list|add …|clear] · /help · /quit"
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

    /// Set the model (same semantics as `/model <name>`): update, persist, notice.
    /// Takes effect on the next agent start.
    fn set_model(&mut self, model: String) {
        self.model = model.clone();
        self.effects.push(Effect::SaveModel(model.clone()));
        self.history.push(Cell::Notice(format!(
            "model set to {model} — active on next start"
        )));
    }

    /// `/agents [on|off]` — toggle multi-agent orchestrator mode (dynamic
    /// delegation + squads). Persisted; applies on the next agent start.
    fn toggle_multi_agent(&mut self, arg: String) {
        let new = match arg.trim().to_lowercase().as_str() {
            "on" | "true" | "1" => true,
            "off" | "false" | "0" => false,
            "" => !self.multi_agent,
            other => {
                self.history.push(Cell::Notice(format!(
                    "usage: /agents [on|off] (currently {})",
                    if self.multi_agent { "on" } else { "off" }
                )));
                let _ = other;
                return;
            }
        };
        self.multi_agent = new;
        self.effects.push(Effect::SaveMultiAgent(new));
        self.history.push(Cell::Notice(format!(
            "multi-agent workflow {}{}",
            if new { "ON" } else { "OFF" },
            if new {
                " — orchestrator delegates to a worker/researcher squad"
            } else {
                ""
            }
        )));
    }

    /// Open the OpenRouter model picker, fetching the catalog on first use.
    fn open_model_picker(&mut self) {
        self.modal = Some(Modal::ModelPicker(ModelPicker::default()));
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
                let sel = match &self.modal {
                    Some(Modal::ModelPicker(p)) => p.selected,
                    _ => 0,
                };
                if let Some(&idx) = filtered.get(sel.min(n.saturating_sub(1))) {
                    let id = self.models[idx].id.clone();
                    self.modal = None;
                    self.set_model(id);
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
            // Shift+Tab cycles the permission posture (default → accept-edits →
            // plan → auto), applied live to the approval gate.
            KeyCode::BackTab => {
                self.permission_mode = self.permission_mode.next();
                self.effects
                    .push(Effect::SetPermissionMode(self.permission_mode.as_u8()));
                self.history.push(Cell::Notice(format!(
                    "permission mode: {}",
                    self.permission_mode.label()
                )));
            }
            KeyCode::Char('u') if ctrl => self.composer = Composer::new(),
            KeyCode::Char(c) if !ctrl && !alt => {
                self.composer.insert_char(c);
                self.menu_selected = 0; // re-filter from the top
            }
            KeyCode::Backspace => {
                self.composer.backspace();
                self.menu_selected = 0;
            }
            KeyCode::Left => self.composer.move_left(),
            KeyCode::Right => self.composer.move_right(),
            KeyCode::Up => self.composer.history_prev(),
            KeyCode::Down => self.composer.history_next(),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_add(SCROLL_STEP),
            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(SCROLL_STEP),
            // Esc interrupts a running turn; when idle it just clears the composer.
            KeyCode::Esc => {
                if self.running {
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
        self.effects.push(Effect::Interrupt);
        self.finalize_active();
        self.history.push(Cell::Notice("interrupted".into()));
        self.running = false;
    }

    fn handle_modal_key(&mut self, key: KeyEvent) {
        match self.modal {
            Some(Modal::Approval(_)) => self.handle_approval_key(key),
            Some(Modal::KeyEntry(_)) => self.handle_key_entry(key),
            Some(Modal::ModelPicker(_)) => self.handle_model_picker_key(key),
            None => {}
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
        assert_eq!(app.effects, vec![Effect::SendInput("hi".into())]);
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
        assert!(names.contains(&"/model") && names.contains(&"/mcp"));
        assert!(!names.contains(&"/help"));
        typed(&mut app, "o"); // "/mo" → only /model
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
        typed(&mut app, "/mo"); // → /model
        app.update(Msg::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)));
        assert_eq!(app.composer.text(), "/model ");
        assert!(!app.menu_open(), "completion closes the menu");
    }

    #[test]
    fn menu_enter_runs_navigated_command() {
        let mut app = keyed();
        typed(&mut app, "/"); // all commands, selected = 0 (/help)
        app.update(key(KeyCode::Down)); // selected = 1 (/model)
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
    fn slash_agents_toggles_and_saves() {
        let mut app = keyed();
        assert!(!app.multi_agent);
        typed(&mut app, "/agents");
        app.update(key(KeyCode::Enter));
        assert!(app.multi_agent, "bare /agents toggles on");
        assert!(app.effects.contains(&Effect::SaveMultiAgent(true)));
        // explicit off
        typed(&mut app, "/agents off");
        app.update(key(KeyCode::Enter));
        assert!(!app.multi_agent);
        assert!(app.effects.contains(&Effect::SaveMultiAgent(false)));
    }

    #[test]
    fn slash_agents_on_is_idempotent_and_notices() {
        let mut app = keyed();
        typed(&mut app, "/agents on");
        app.update(key(KeyCode::Enter));
        assert!(app.multi_agent);
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("multi-agent workflow ON"))),
        );
        // a pure config command must never start a run
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
        });
        let w = app.agents.iter().find(|r| r.name == "worker").unwrap();
        assert_eq!(w.state, AgentState::Done);
        assert_eq!(w.tokens, 1234);
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
    fn roster_is_inert_in_single_agent_mode() {
        let mut app = keyed(); // multi_agent = false
        app.update(tool_started("heartbit", "bash"));
        app.update(Msg::AgentsDispatched(vec!["x".into()]));
        assert!(app.agents.is_empty(), "no roster tracking in single mode");
    }

    #[test]
    fn roster_clears_on_new_user_turn() {
        let mut app = multi();
        app.update(Msg::AgentsDispatched(vec!["worker".into()]));
        assert_eq!(app.agents.len(), 1);
        typed(&mut app, "next task");
        app.update(key(KeyCode::Enter));
        assert!(
            app.agents.is_empty(),
            "each turn starts with a fresh roster"
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
        assert_eq!(app.effects, vec![Effect::SendInput("hello".into())]);
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
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("boom")));
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
    fn shift_tab_cycles_permission_mode_and_emits_effect() {
        let mut app = keyed();
        assert_eq!(app.permission_mode, PermissionMode::Default);
        app.update(key(KeyCode::BackTab));
        assert_eq!(app.permission_mode, PermissionMode::AcceptEdits);
        assert!(app.effects.contains(&Effect::SetPermissionMode(1)));
        app.update(key(KeyCode::BackTab)); // Plan
        app.update(key(KeyCode::BackTab)); // Auto
        assert_eq!(app.permission_mode, PermissionMode::Auto);
        assert!(app.effects.contains(&Effect::SetPermissionMode(3)));
        app.update(key(KeyCode::BackTab)); // wraps to Default
        assert_eq!(app.permission_mode, PermissionMode::Default);
        assert!(app.effects.contains(&Effect::SetPermissionMode(0)));
        // a notice records the change
        assert!(
            app.history
                .iter()
                .any(|c| matches!(c, Cell::Notice(n) if n.contains("permission mode")))
        );
    }

    #[test]
    fn mouse_wheel_scrolls_transcript_not_command_history() {
        let mut app = keyed();
        // type a command so there'd be composer history to (wrongly) move
        typed(&mut app, "earlier");
        app.update(key(KeyCode::Enter));
        let composed_before = app.composer.text();
        // wheel up scrolls the transcript output, leaving the composer untouched
        app.update(Msg::WheelUp);
        app.update(Msg::WheelUp);
        assert_eq!(app.scroll, 2 * WHEEL_STEP);
        assert_eq!(
            app.composer.text(),
            composed_before,
            "the wheel must NOT touch the composer (command history stays on ↑/↓)"
        );
        app.update(Msg::WheelDown);
        assert_eq!(app.scroll, WHEEL_STEP);
    }

    #[test]
    fn pageup_scrolls_back_pagedown_returns() {
        let mut app = App::new("m");
        app.update(key(KeyCode::PageUp));
        assert_eq!(app.scroll, SCROLL_STEP);
        app.update(key(KeyCode::PageDown));
        assert_eq!(app.scroll, 0);
    }
}
