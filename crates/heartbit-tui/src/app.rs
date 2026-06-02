//! Application state + the pure `update(Msg)` reducer. No terminal, no channels:
//! state mutations only, with side-effects pushed onto `effects` for the edge
//! (main loop) to perform. This is what makes the whole interaction unit-testable.

use std::collections::HashMap;
use std::sync::mpsc::SyncSender;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use heartbit_core::{ApprovalDecision, TokenUsage};

use crate::cells::{Cell, ToolStatus};
use crate::composer::Composer;
use crate::msg::{Msg, PendingTool};

/// How many transcript lines a PageUp/PageDown moves.
const SCROLL_STEP: u16 = 8;

/// A side-effect for the edge (main loop) to perform after an update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// Submit a user message to the agent (start the run, or feed `on_input`).
    SendInput(String),
    /// Tear down and exit.
    Quit,
}

/// A pending tool-approval prompt.
pub struct ApprovalModal {
    pub tools: Vec<PendingTool>,
    pub reply: SyncSender<ApprovalDecision>,
}

/// The full UI state.
pub struct App {
    pub history: Vec<Cell>,
    /// Assistant text being streamed for the current turn (not yet finalized).
    pub active: Option<String>,
    pub composer: Composer,
    pub modal: Option<ApprovalModal>,
    pub model: String,
    pub tokens: TokenUsage,
    pub running: bool,
    /// Lines scrolled up from the bottom (0 = pinned to newest).
    pub scroll: u16,
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
            tokens: TokenUsage::default(),
            running: false,
            scroll: 0,
            spinner: 0,
            should_quit: false,
            effects: Vec::new(),
            tool_index: HashMap::new(),
        }
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

    /// Apply a message, mutating state and queuing effects.
    pub fn update(&mut self, msg: Msg) {
        match msg {
            Msg::Tick => self.spinner = self.spinner.wrapping_add(1),
            Msg::Resize => {}
            Msg::Paste(s) => self.composer.insert_str(&s),
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
            } => {
                self.finalize_active();
                self.tokens.input_tokens =
                    self.tokens.input_tokens.saturating_add(usage.input_tokens);
                self.tokens.output_tokens = self
                    .tokens
                    .output_tokens
                    .saturating_add(usage.output_tokens);
                // A text-only turn means the agent now idles awaiting input.
                if !had_tool_calls {
                    self.running = false;
                }
            }
            Msg::ToolStarted { id, name, input } => {
                self.finalize_active(); // the assistant preamble (if any) is done
                let idx = self.history.len();
                self.tool_index.insert(id, idx);
                self.history.push(Cell::Tool {
                    name,
                    input,
                    status: ToolStatus::Running,
                    output: None,
                    duration_ms: None,
                });
                self.scroll = 0;
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
                self.modal = Some(ApprovalModal { tools, reply });
            }
        }
    }

    fn submit(&mut self) {
        let text = self.composer.take();
        if text.trim().is_empty() {
            return;
        }
        self.history.push(Cell::User(text.clone()));
        self.running = true;
        self.scroll = 0;
        self.effects.push(Effect::SendInput(text));
    }

    fn quit(&mut self) {
        self.should_quit = true;
        self.effects.push(Effect::Quit);
    }

    fn handle_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Enter => {
                if shift || alt {
                    self.composer.newline();
                } else {
                    self.submit();
                }
            }
            KeyCode::Char('c') | KeyCode::Char('d') if ctrl => self.quit(),
            KeyCode::Char('u') if ctrl => self.composer = Composer::new(),
            KeyCode::Char(c) if !ctrl && !alt => self.composer.insert_char(c),
            KeyCode::Backspace => self.composer.backspace(),
            KeyCode::Left => self.composer.move_left(),
            KeyCode::Right => self.composer.move_right(),
            KeyCode::Up => self.composer.history_prev(),
            KeyCode::Down => self.composer.history_next(),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_add(SCROLL_STEP),
            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(SCROLL_STEP),
            KeyCode::Esc => self.composer = Composer::new(),
            _ => {}
        }
    }

    fn handle_modal_key(&mut self, key: KeyEvent) {
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
            && let Some(modal) = self.modal.take()
        {
            // Best-effort: if the agent thread is gone, the decision is moot.
            let _ = modal.reply.send(decision);
        }
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

    #[test]
    fn submit_creates_user_cell_and_send_effect() {
        let mut app = App::new("m");
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
        });
        assert!(app.running, "tool turn should stay running");
        // A text-only turn means the agent now idles awaiting input.
        app.update(Msg::LlmDone {
            usage: TokenUsage::default(),
            had_tool_calls: false,
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
    fn pageup_scrolls_back_pagedown_returns() {
        let mut app = App::new("m");
        app.update(key(KeyCode::PageUp));
        assert_eq!(app.scroll, SCROLL_STEP);
        app.update(key(KeyCode::PageDown));
        assert_eq!(app.scroll, 0);
    }
}
