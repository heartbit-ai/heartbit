//! Workflow-level events and the `phase()` / `log()` observability helpers.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::llm::types::TokenUsage;

use super::ctx::WorkflowCtx;

/// Events emitted by the workflow combinators, mirroring the per-agent
/// [`AgentEvent`](crate::AgentEvent) plane at the workflow level.
///
/// This is a **separate** enum from `AgentEvent` (rather than new variants on
/// it) because `AgentEvent` is not `#[non_exhaustive]` and has an exhaustive
/// `type_name()` match plus external matchers — extending it would be a
/// breaking change. `WorkflowEvent` *is* `#[non_exhaustive]` so later phases
/// can add variants without breaking downstream matchers.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkflowEvent {
    /// A new phase began; subsequently-issued agents group under `title`.
    PhaseStarted {
        /// Phase title.
        title: String,
    },
    /// An agent leaf began executing.
    AgentStarted {
        /// Agent label (defaults to a generated name when unset).
        label: String,
        /// The phase this agent was issued under, if any.
        phase: Option<String>,
    },
    /// An agent leaf completed successfully.
    AgentFinished {
        /// Agent label.
        label: String,
        /// Token usage reported by the underlying runner.
        usage: TokenUsage,
    },
    /// An agent leaf's result was replayed from the resume journal (no model
    /// call). `usage` is the originally-recorded usage, for progress totals.
    AgentReplayed {
        /// Agent label.
        label: String,
        /// Token usage recorded when this output was first produced.
        usage: TokenUsage,
    },
    /// An agent leaf was skipped (e.g. cancelled mid-run); the combinator
    /// surfaces this as `Ok(None)`.
    AgentSkipped {
        /// Agent label.
        label: String,
    },
    /// An agent-domain failure inside a combinator; the slot collapses to
    /// `None` (the combinator never rejects on an agent-domain error).
    AgentFailed {
        /// Agent label.
        label: String,
        /// Display string of the underlying error.
        error: String,
    },
    /// A free-form progress line emitted via [`log`].
    LogLine {
        /// Message text.
        msg: String,
    },
}

/// Callback type for receiving [`WorkflowEvent`]s. Kept object-safe and
/// `Send + Sync` so it can be shared across concurrently-running agents.
pub type OnWorkflowEvent = dyn Fn(WorkflowEvent) + Send + Sync;

/// Begin a new phase. Emits [`WorkflowEvent::PhaseStarted`] and sets the
/// default phase for subsequently-issued agents. The returned [`PhaseGuard`]
/// restores the prior default phase when dropped, so phases nest correctly.
///
/// Note: the phase is the default for agents *issued after* this call; each
/// [`AgentCall`](super::agent::AgentCall) snapshots the default at construction,
/// so concurrently-running agents never observe a torn phase.
#[must_use = "the phase ends when the returned guard is dropped"]
pub fn phase(ctx: &WorkflowCtx, title: impl Into<String>) -> PhaseGuard {
    let title = title.into();
    ctx.emit(WorkflowEvent::PhaseStarted {
        title: title.clone(),
    });
    let prior = ctx.swap_default_phase(Some(Arc::from(title.as_str())));
    PhaseGuard {
        ctx: ctx.clone(),
        prior,
    }
}

/// Emit a free-form progress line ([`WorkflowEvent::LogLine`]).
pub fn log(ctx: &WorkflowCtx, msg: impl Into<String>) {
    ctx.emit(WorkflowEvent::LogLine { msg: msg.into() });
}

/// RAII guard returned by [`phase`]; restores the prior default phase on drop.
pub struct PhaseGuard {
    ctx: WorkflowCtx,
    prior: Option<Arc<str>>,
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        self.ctx.swap_default_phase(self.prior.take());
    }
}

#[cfg(test)]
mod tests {
    use super::WorkflowEvent;

    #[test]
    fn workflow_event_serializes_with_snake_case_type_tag() {
        let event = WorkflowEvent::PhaseStarted {
            title: "review".into(),
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains(r#""type":"phase_started""#), "json: {json}");
        assert!(json.contains(r#""title":"review""#), "json: {json}");
    }

    #[test]
    fn log_line_event_roundtrips_through_json() {
        let event = WorkflowEvent::LogLine {
            msg: "3/10 found".into(),
        };
        let json = serde_json::to_string(&event).expect("serialize");
        let back: WorkflowEvent = serde_json::from_str(&json).expect("deserialize");
        match back {
            WorkflowEvent::LogLine { msg } => assert_eq!(msg, "3/10 found"),
            other => panic!("expected LogLine, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // phase() / log() observability helpers
    // -----------------------------------------------------------------------

    use std::sync::{Arc, Mutex};

    use super::super::ctx::WorkflowCtx;
    use super::{OnWorkflowEvent, log, phase};
    use crate::agent::test_helpers::MockProvider;
    use crate::llm::BoxedProvider;

    /// Build a ctx whose events are captured into the returned `Vec`.
    fn ctx_capturing() -> (WorkflowCtx, Arc<Mutex<Vec<WorkflowEvent>>>) {
        let sink: Arc<Mutex<Vec<WorkflowEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let sink2 = Arc::clone(&sink);
        let cb: Arc<OnWorkflowEvent> = Arc::new(move |ev: WorkflowEvent| {
            sink2.lock().expect("sink lock").push(ev);
        });
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::new(MockProvider::new(vec![]))))
            .on_event(cb)
            .build()
            .expect("build ctx");
        (ctx, sink)
    }

    #[test]
    fn phase_emits_started_and_sets_default_phase() {
        let (ctx, sink) = ctx_capturing();
        assert!(ctx.current_phase().is_none());

        let _guard = phase(&ctx, "review");
        assert_eq!(ctx.current_phase().as_deref(), Some("review"));

        let events = sink.lock().expect("lock");
        assert!(
            matches!(events.first(), Some(WorkflowEvent::PhaseStarted { title }) if title == "review"),
            "events: {events:?}"
        );
    }

    #[test]
    fn phase_guard_restores_prior_default_on_drop() {
        let (ctx, _sink) = ctx_capturing();
        {
            let _outer = phase(&ctx, "outer");
            assert_eq!(ctx.current_phase().as_deref(), Some("outer"));
            {
                let _inner = phase(&ctx, "inner");
                assert_eq!(ctx.current_phase().as_deref(), Some("inner"));
            }
            // inner dropped -> restored to outer
            assert_eq!(ctx.current_phase().as_deref(), Some("outer"));
        }
        // outer dropped -> restored to None
        assert!(ctx.current_phase().is_none());
    }

    #[test]
    fn log_emits_log_line() {
        let (ctx, sink) = ctx_capturing();
        log(&ctx, "halfway there");
        let events = sink.lock().expect("lock");
        assert!(
            matches!(events.last(), Some(WorkflowEvent::LogLine { msg }) if msg == "halfway there"),
            "events: {events:?}"
        );
    }
}
