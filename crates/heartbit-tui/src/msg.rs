//! The `Msg` type: everything that can change the app state, from either the
//! terminal (keys/paste/resize/tick) or the agent thread (mapped [`AgentEvent`]s
//! and the synchronous approval round-trip). [`Msg::from_event`] is a pure
//! translation of the framework's events into UI messages.

use std::sync::mpsc::SyncSender;

use crossterm::event::KeyEvent;
use heartbit_core::{AgentEvent, ApprovalDecision, TokenUsage};

/// One tool call awaiting approval (name + pretty-printed input).
#[derive(Debug, Clone)]
pub struct PendingTool {
    pub name: String,
    pub input: String,
}

/// Everything that can drive an [`crate::app::App`] state transition.
pub enum Msg {
    // ---- from the terminal ----
    Key(KeyEvent),
    Paste(String),
    Resize,
    /// Periodic tick (spinner animation / coalesced redraw).
    Tick,

    // ---- from the agent thread (mapped AgentEvent) ----
    TurnStarted,
    /// The assistant finished a message for this turn; finalize the streamed
    /// text and update the token status from `usage`. `had_tool_calls` is false
    /// for a final text answer (the agent will then idle awaiting input).
    LlmDone {
        usage: TokenUsage,
        had_tool_calls: bool,
    },
    StreamDelta(String),
    ToolStarted {
        id: String,
        name: String,
        input: String,
    },
    ToolCompleted {
        id: String,
        is_error: bool,
        output: String,
        duration_ms: u64,
    },
    Notice(String),
    RunCompleted,
    RunFailed(String),

    // ---- the synchronous approval round-trip ----
    Approval {
        tools: Vec<PendingTool>,
        reply: SyncSender<ApprovalDecision>,
    },
}

impl Msg {
    /// Translate a framework [`AgentEvent`] into a UI [`Msg`], or `None` if it has
    /// no visible effect. Streaming text arrives separately via the `on_text`
    /// callback (as [`Msg::StreamDelta`]), not from events.
    pub fn from_event(event: AgentEvent) -> Option<Msg> {
        match event {
            AgentEvent::TurnStarted { .. } => Some(Msg::TurnStarted),
            AgentEvent::LlmResponse {
                usage,
                tool_call_count,
                ..
            } => Some(Msg::LlmDone {
                usage,
                had_tool_calls: tool_call_count > 0,
            }),
            AgentEvent::ToolCallStarted {
                tool_name,
                tool_call_id,
                input,
                ..
            } => Some(Msg::ToolStarted {
                id: tool_call_id,
                name: tool_name,
                input,
            }),
            AgentEvent::ToolCallCompleted {
                tool_call_id,
                is_error,
                output,
                duration_ms,
                ..
            } => Some(Msg::ToolCompleted {
                id: tool_call_id,
                is_error,
                output,
                duration_ms,
            }),
            AgentEvent::RunCompleted { .. } => Some(Msg::RunCompleted),
            AgentEvent::RunFailed { error, .. } => Some(Msg::RunFailed(error)),
            AgentEvent::GuardrailDenied {
                reason, tool_name, ..
            } => Some(Msg::Notice(format!(
                "guardrail denied {}: {reason}",
                tool_name.as_deref().unwrap_or("tool")
            ))),
            AgentEvent::GuardrailWarned {
                reason, tool_name, ..
            } => Some(Msg::Notice(format!(
                "guardrail warning {}: {reason}",
                tool_name.as_deref().unwrap_or("tool")
            ))),
            AgentEvent::AutoCompactionTriggered { .. } => {
                Some(Msg::Notice("context auto-compacted".into()))
            }
            AgentEvent::DoomLoopDetected { .. } => {
                Some(Msg::Notice("doom-loop detected — intervening".into()))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_started_maps_id_name_input() {
        let ev = AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "bash".into(),
            tool_call_id: "tc1".into(),
            input: "{\"command\":\"ls\"}".into(),
        };
        match Msg::from_event(ev) {
            Some(Msg::ToolStarted { id, name, input }) => {
                assert_eq!(id, "tc1");
                assert_eq!(name, "bash");
                assert!(input.contains("ls"));
            }
            _ => panic!("expected ToolStarted"),
        }
    }

    #[test]
    fn tool_completed_maps_error_and_duration() {
        let ev = AgentEvent::ToolCallCompleted {
            agent: "a".into(),
            tool_name: "verify".into(),
            tool_call_id: "tc2".into(),
            is_error: true,
            duration_ms: 99,
            output: "boom".into(),
        };
        match Msg::from_event(ev) {
            Some(Msg::ToolCompleted {
                id,
                is_error,
                output,
                duration_ms,
            }) => {
                assert_eq!(id, "tc2");
                assert!(is_error);
                assert_eq!(output, "boom");
                assert_eq!(duration_ms, 99);
            }
            _ => panic!("expected ToolCompleted"),
        }
    }

    #[test]
    fn run_failed_carries_error() {
        let ev = AgentEvent::RunFailed {
            agent: "a".into(),
            error: "kaboom".into(),
            partial_usage: TokenUsage::default(),
        };
        assert!(matches!(Msg::from_event(ev), Some(Msg::RunFailed(e)) if e == "kaboom"));
    }

    #[test]
    fn guardrail_denied_becomes_notice() {
        let ev = AgentEvent::GuardrailDenied {
            agent: "a".into(),
            hook: "pre_tool".into(),
            reason: "blocked".into(),
            tool_name: Some("bash".into()),
        };
        match Msg::from_event(ev) {
            Some(Msg::Notice(n)) => assert!(n.contains("blocked") && n.contains("bash")),
            _ => panic!("expected Notice"),
        }
    }

    #[test]
    fn unmapped_event_is_none() {
        let ev = AgentEvent::RunStarted {
            agent: "a".into(),
            task: "t".into(),
        };
        assert!(Msg::from_event(ev).is_none());
    }
}
