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

/// The entry agent's event name: core's `Orchestrator` names its entry runner
/// `"orchestrator"`. Run lifecycle (the `running` flag, roster settle, /learn
/// commit, terminal failure) must key off THIS agent only — the delegation
/// plane forwards `on_event` to every sub-agent, so their lifecycle events
/// arrive on the same channel under their own names (audit 2026-06-09).
pub const ENTRY_AGENT: &str = "orchestrator";

/// Everything that can drive an [`crate::app::App`] state transition.
pub enum Msg {
    // ---- from the terminal ----
    Key(KeyEvent),
    Paste(String),
    /// The terminal window gained (`true`) or lost (`false`) focus. Requires
    /// `EnableFocusChange`; terminals that do not report it never send this, so
    /// `App::focused` stays `true` and every consumer sees today's behaviour.
    FocusChanged(bool),
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
        /// The failure reason when `!success` (e.g. "Max turns (60) exceeded"),
        /// so the UI can surface it instead of swallowing it. `None` on success.
        error: Option<String>,
    },
    /// A sub-agent's LLM call finished: accumulate its cost into the session
    /// totals WITHOUT touching the run lifecycle (`running`, roster settle,
    /// /learn commit, context-fill bar) — those belong to the entry agent.
    SubAgentLlmDone {
        usage: TokenUsage,
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
    /// Saved handoff briefs (for the `/handoff` picker).
    HandoffsListed(Vec<crate::session::HandoffMeta>),

    // ---- the synchronous approval round-trip ----
    Approval {
        tools: Vec<PendingTool>,
        reply: SyncSender<ApprovalDecision>,
    },
    /// Agent-to-user structured question (the `question` builtin): render the
    /// options modal and send the selections back through the oneshot channel.
    /// Dropping the sender (Esc) makes the tool report a dismissal.
    Question {
        request: heartbit_core::tool::builtins::QuestionRequest,
        reply: tokio::sync::oneshot::Sender<heartbit_core::tool::builtins::QuestionResponse>,
    },
}

impl Msg {
    /// Translate a framework [`AgentEvent`] into a UI [`Msg`], or `None` if it has
    /// no visible effect. Streaming text arrives separately via the `on_text`
    /// callback (as [`Msg::StreamDelta`]), not from events. Lifecycle events
    /// (turn/LLM/run) are gated on [`ENTRY_AGENT`]: forwarded sub-agent copies
    /// map to roster/cost messages instead, never to run-state transitions.
    pub fn from_event(event: AgentEvent) -> Option<Msg> {
        match event {
            AgentEvent::TurnStarted { agent, .. } => {
                (agent == ENTRY_AGENT).then_some(Msg::TurnStarted)
            }
            AgentEvent::LlmResponse {
                agent,
                usage,
                tool_call_count,
                time_to_first_token_ms,
                ..
            } => Some(if agent == ENTRY_AGENT {
                Msg::LlmDone {
                    usage,
                    had_tool_calls: tool_call_count > 0,
                    ttft_ms: time_to_first_token_ms,
                }
            } else {
                Msg::SubAgentLlmDone { usage }
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
                error: None,
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
            AgentEvent::RunCompleted { agent, .. } => {
                // A sub-agent's completion is already a roster row finish (the
                // SubAgentCompleted event above) — only the ENTRY agent's
                // completion ends the run.
                (agent == ENTRY_AGENT).then_some(Msg::RunCompleted)
            }
            AgentEvent::RunFailed {
                agent,
                error,
                partial_usage,
            } => Some(if agent == ENTRY_AGENT {
                Msg::RunFailed(error)
            } else {
                // A sub-agent failure is routine (delegate_task returns the
                // error as a tool result; the orchestrator continues) — mark
                // the roster row failed, never the whole run — but surface the
                // reason (handler pushes a Notice) so a silent 60-turn death
                // doesn't pass for a finished audit.
                Msg::SubAgentDone {
                    agent,
                    success: false,
                    tokens: partial_usage.input_tokens + partial_usage.output_tokens,
                    error: Some(error),
                }
            }),
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
            agent: ENTRY_AGENT.into(),
            error: "kaboom".into(),
            partial_usage: TokenUsage::default(),
        };
        assert!(matches!(Msg::from_event(ev), Some(Msg::RunFailed(e)) if e == "kaboom"));
    }

    // ---- delegation-plane attribution (audit 2026-06-09): sub-agent lifecycle
    // events must NOT masquerade as the entry agent's — they flipped `running`
    // mid-run, settled the roster early, committed /learn lessons before the
    // rewrite, and declared the whole run failed on a sub-agent failure.

    fn llm_response(agent: &str, tool_call_count: usize) -> AgentEvent {
        AgentEvent::LlmResponse {
            agent: agent.into(),
            turn: 1,
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 10,
                ..Default::default()
            },
            stop_reason: heartbit_core::StopReason::EndTurn,
            tool_call_count,
            text: String::new(),
            latency_ms: 5,
            model: None,
            time_to_first_token_ms: 2,
        }
    }

    #[test]
    fn entry_llm_response_maps_to_llm_done() {
        match Msg::from_event(llm_response(ENTRY_AGENT, 0)) {
            Some(Msg::LlmDone {
                usage,
                had_tool_calls,
                ttft_ms,
            }) => {
                assert_eq!(usage.input_tokens, 100);
                assert!(!had_tool_calls);
                assert_eq!(ttft_ms, 2);
            }
            _ => panic!("entry-agent LlmResponse must map to LlmDone"),
        }
    }

    #[test]
    fn sub_agent_llm_response_maps_to_usage_only_not_llm_done() {
        match Msg::from_event(llm_response("worker", 0)) {
            Some(Msg::SubAgentLlmDone { usage }) => {
                assert_eq!(usage.input_tokens, 100, "the cost is still accumulated");
            }
            Some(Msg::LlmDone { .. }) => {
                panic!("a sub-agent LlmResponse must not flip the run lifecycle")
            }
            _ => panic!("expected SubAgentLlmDone"),
        }
    }

    #[test]
    fn sub_agent_run_completed_is_dropped_entry_maps() {
        let sub = AgentEvent::RunCompleted {
            agent: "researcher".into(),
            total_usage: TokenUsage::default(),
            tool_calls_made: 3,
        };
        assert!(
            Msg::from_event(sub).is_none(),
            "SubAgentCompleted already finishes the roster row"
        );
        let entry = AgentEvent::RunCompleted {
            agent: ENTRY_AGENT.into(),
            total_usage: TokenUsage::default(),
            tool_calls_made: 3,
        };
        assert!(matches!(Msg::from_event(entry), Some(Msg::RunCompleted)));
    }

    #[test]
    fn sub_agent_run_failed_maps_to_roster_failure_not_run_failed() {
        let ev = AgentEvent::RunFailed {
            agent: "worker".into(),
            error: "Max turns (60) exceeded".into(),
            partial_usage: TokenUsage {
                input_tokens: 7,
                output_tokens: 3,
                ..Default::default()
            },
        };
        match Msg::from_event(ev) {
            Some(Msg::SubAgentDone {
                agent,
                success,
                tokens,
                error,
            }) => {
                assert_eq!(agent, "worker");
                assert!(!success);
                assert_eq!(tokens, 10);
                assert_eq!(
                    error.as_deref(),
                    Some("Max turns (60) exceeded"),
                    "the failure reason must be carried through for the UI to show"
                );
            }
            Some(Msg::RunFailed(_)) => {
                panic!("a sub-agent failure must not declare the whole run failed")
            }
            _ => panic!("expected SubAgentDone"),
        }
    }

    #[test]
    fn sub_agent_turn_started_is_dropped_entry_maps() {
        let sub = AgentEvent::TurnStarted {
            agent: "worker".into(),
            turn: 1,
            max_turns: 60,
        };
        assert!(Msg::from_event(sub).is_none());
        let entry = AgentEvent::TurnStarted {
            agent: ENTRY_AGENT.into(),
            turn: 1,
            max_turns: 300,
        };
        assert!(matches!(Msg::from_event(entry), Some(Msg::TurnStarted)));
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
