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
    /// Mouse wheel — scroll the transcript (output history), NOT the composer's
    /// command history (which stays on ↑/↓).
    WheelUp,
    WheelDown,
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
        /// Time-to-first-token for this turn (ms) — shown in the status line.
        ttft_ms: u64,
    },
    StreamDelta(String),
    /// A live chain-of-thought delta (reasoning models) — streamed into the
    /// in-progress reasoning buffer as it arrives, ahead of the answer.
    ReasoningDelta(String),
    ToolStarted {
        id: String,
        name: String,
        input: String,
        /// Which agent ran the tool (the orchestrator or a named sub-agent).
        agent: String,
    },
    ToolCompleted {
        id: String,
        is_error: bool,
        output: String,
        duration_ms: u64,
    },
    /// The orchestrator delegated a task to these sub-agents (parallel fan-out).
    AgentsDispatched(Vec<String>),
    /// A sub-agent finished (with its accumulated token cost).
    SubAgentDone {
        agent: String,
        success: bool,
        tokens: u32,
    },
    /// The orchestrator dynamically spawned a sub-agent with a scoped task.
    AgentSpawned {
        name: String,
        task: String,
    },
    Notice(String),
    RunCompleted,
    /// The agent thread (of a given spawn epoch) has exited — its `execute`/`run`
    /// returned (session ended or build failed). The epoch lets the UI ignore a
    /// stale exit from a thread it already replaced (e.g. on an `/agents` restart).
    AgentExited(u64),
    RunFailed(String),

    // ---- OpenRouter model catalog (fetched async for the model picker) ----
    ModelsLoaded(Vec<crate::models::ModelEntry>),
    ModelsFailed(String),
    /// The project file index (walked async for `@`-mention autocomplete).
    FilesLoaded(Vec<String>),
    /// Saved sessions listed for the `/resume` picker.
    SessionsListed(Vec<crate::session::SessionMeta>),
    /// Trace stats computed (rendered table) — or why they couldn't be.
    StatsReady(Result<(String, Box<crate::trace_stats::TraceStats>), String>),
    /// `/analyze` prepared: show `display` as the user cell, send `task` to
    /// the agent (the Plan-mode `sent ≠ displayed` precedent).
    AnalyzeReady {
        display: String,
        task: String,
    },
    /// `/analyze` could not prepare (no trace, stats error…).
    AnalyzeFailed(String),
    /// `/learn` prepared: show `display`, send `task`; `staged_digest` is the
    /// staged lessons file's content hash at stage time (the commit guard).
    LearnReady {
        display: String,
        task: String,
        staged_digest: u64,
    },
    /// `/learn` could not prepare (no diagnosis, stage error…).
    LearnFailed(String),
    /// A resumed session's transcript (replaces the current history).
    SessionLoaded(Vec<crate::cells::Cell>),

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
                time_to_first_token_ms,
                ..
            } => Some(Msg::LlmDone {
                usage,
                had_tool_calls: tool_call_count > 0,
                ttft_ms: time_to_first_token_ms,
            }),
            AgentEvent::ToolCallStarted {
                tool_name,
                tool_call_id,
                input,
                agent,
            } => Some(Msg::ToolStarted {
                id: tool_call_id,
                name: tool_name,
                input,
                agent,
            }),
            // Reasoning is streamed live via the `on_reasoning` callback (→
            // ReasoningDelta), so the post-hoc event is ignored here to avoid a
            // duplicate cell. (The event remains for non-streaming consumers.)
            AgentEvent::Reasoning { .. } => None,
            AgentEvent::SubAgentsDispatched { agents, .. } => Some(Msg::AgentsDispatched(agents)),
            AgentEvent::SubAgentCompleted {
                agent,
                success,
                usage,
            } => Some(Msg::SubAgentDone {
                agent,
                success,
                tokens: usage.input_tokens + usage.output_tokens,
            }),
            AgentEvent::AgentSpawned {
                spawned_name, task, ..
            } => Some(Msg::AgentSpawned {
                name: spawned_name,
                task,
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
                Some(Msg::Notice("context auto-compacted (overflow)".into()))
            }
            // Proactive compaction backstop (#2/#3) fired — surface it so the user
            // can see the working set was summarized (dropped content stays
            // restorable via fetch_full_output / recall_context).
            AgentEvent::ContextSummarized { .. } => Some(Msg::Notice(
                "🗜 context compacted — older turns summarized (restorable)".into(),
            )),
            AgentEvent::DoomLoopDetected { .. } => {
                Some(Msg::Notice("doom-loop detected — intervening".into()))
            }
            AgentEvent::RetryAttempt {
                attempt,
                max_retries,
                delay_ms,
                error_class,
                ..
            } => Some(Msg::Notice(format!(
                "LLM retry {attempt}/{max_retries} in {delay_ms}ms ({error_class})"
            ))),
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
            Some(Msg::ToolStarted {
                id,
                name,
                input,
                agent,
            }) => {
                assert_eq!(id, "tc1");
                assert_eq!(name, "bash");
                assert!(input.contains("ls"));
                assert_eq!(agent, "a", "the agent identity must be threaded through");
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
    fn retry_attempt_becomes_a_notice() {
        let ev = AgentEvent::RetryAttempt {
            agent: "(provider)".into(),
            attempt: 2,
            max_retries: 3,
            delay_ms: 1500,
            error_class: "rate_limited".into(),
        };
        match Msg::from_event(ev) {
            Some(Msg::Notice(n)) => {
                assert!(n.contains("2/3"), "got: {n}");
                assert!(n.contains("rate_limited"), "got: {n}");
            }
            _ => panic!("expected Notice"),
        }
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
    fn post_hoc_reasoning_event_is_not_mapped_streaming_is_the_source() {
        // The TUI streams reasoning live via the on_reasoning callback, so the
        // post-hoc event must NOT also map to a Msg (it would double-render).
        let ev = AgentEvent::Reasoning {
            agent: "a".into(),
            turn: 1,
            text: "thinking hard".into(),
        };
        assert!(Msg::from_event(ev).is_none());
    }

    #[test]
    fn context_summarized_event_surfaces_a_notice() {
        let ev = AgentEvent::ContextSummarized {
            agent: "a".into(),
            turn: 7,
            usage: TokenUsage::default(),
        };
        assert!(
            matches!(Msg::from_event(ev), Some(Msg::Notice(n)) if n.contains("compacted")),
            "ContextSummarized must surface a compaction notice"
        );
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
