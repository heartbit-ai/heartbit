//! [`ProgressTracker`] — fold [`WorkflowEvent`]s into a live [`RunProgress`]
//! snapshot, the Rust analog of Claude Code's `/workflows` progress view.
//!
//! A tracker installs itself as the ctx's event sink
//! ([`WorkflowCtxBuilder::on_event`](super::ctx::WorkflowCtxBuilder::on_event))
//! and accumulates per-phase agent counts, lifecycle tallies, and token totals.
//! [`snapshot`](ProgressTracker::snapshot) returns a cheap clone for rendering.

use std::sync::{Arc, Mutex};

use crate::llm::types::TokenUsage;

use super::event::{OnWorkflowEvent, WorkflowEvent};

/// Per-phase progress: how many agents were issued under a given phase title.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhaseProgress {
    /// The phase title (from [`WorkflowEvent::PhaseStarted`]).
    pub title: String,
    /// Agents issued under this phase so far.
    pub agents_started: u64,
}

/// A point-in-time snapshot of a workflow run's progress.
#[derive(Debug, Clone, Default)]
pub struct RunProgress {
    /// The most recently started phase, if any.
    pub current_phase: Option<String>,
    /// One entry per phase, in start order.
    pub phases: Vec<PhaseProgress>,
    /// Total agents issued (across all phases and the no-phase default).
    pub agents_started: u64,
    /// Agents that completed live.
    pub agents_finished: u64,
    /// Agents whose output was replayed from the resume journal.
    pub agents_replayed: u64,
    /// Agents skipped (e.g. cancelled mid-run).
    pub agents_skipped: u64,
    /// Agents that failed with an agent-domain error.
    pub agents_failed: u64,
    /// Count of `log()` lines emitted.
    pub log_lines: u64,
    /// Accumulated token usage across finished + replayed agents.
    pub total_tokens: TokenUsage,
}

impl RunProgress {
    /// Fold one [`WorkflowEvent`] into this snapshot.
    ///
    /// The match is intentionally exhaustive (no `_` arm): because this lives in
    /// the same crate as [`WorkflowEvent`], adding a variant there makes this
    /// fail to compile until a handling decision is made here.
    pub fn apply(&mut self, event: &WorkflowEvent) {
        match event {
            WorkflowEvent::PhaseStarted { title } => {
                self.current_phase = Some(title.clone());
                self.phases.push(PhaseProgress {
                    title: title.clone(),
                    agents_started: 0,
                });
            }
            WorkflowEvent::AgentStarted { phase, .. } => {
                self.agents_started += 1;
                if let Some(phase) = phase
                    && let Some(p) = self.phases.iter_mut().rev().find(|p| &p.title == phase)
                {
                    p.agents_started += 1;
                }
            }
            WorkflowEvent::AgentFinished { usage, .. } => {
                self.agents_finished += 1;
                self.total_tokens += *usage;
            }
            WorkflowEvent::AgentReplayed { usage, .. } => {
                self.agents_replayed += 1;
                self.total_tokens += *usage;
            }
            WorkflowEvent::AgentSkipped { .. } => self.agents_skipped += 1,
            WorkflowEvent::AgentFailed { .. } => self.agents_failed += 1,
            WorkflowEvent::LogLine { .. } => self.log_lines += 1,
        }
    }
}

/// Accumulates [`WorkflowEvent`]s into a shared [`RunProgress`]. Cheap to clone
/// (an `Arc` inside); install via [`callback`](Self::callback).
#[derive(Clone, Default)]
pub struct ProgressTracker(Arc<Mutex<RunProgress>>);

impl ProgressTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// A point-in-time clone of the accumulated progress.
    pub fn snapshot(&self) -> RunProgress {
        self.0.lock().expect("progress lock poisoned").clone()
    }

    /// An event sink that folds each [`WorkflowEvent`] into this tracker. Pass to
    /// [`WorkflowCtxBuilder::on_event`](super::ctx::WorkflowCtxBuilder::on_event).
    pub fn callback(&self) -> Arc<OnWorkflowEvent> {
        let inner = Arc::clone(&self.0);
        Arc::new(move |event: WorkflowEvent| {
            if let Ok(mut progress) = inner.lock() {
                progress.apply(&event);
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            ..Default::default()
        }
    }

    #[test]
    fn phase_started_tracks_current_and_appends() {
        let mut p = RunProgress::default();
        p.apply(&WorkflowEvent::PhaseStarted {
            title: "research".into(),
        });
        p.apply(&WorkflowEvent::PhaseStarted {
            title: "verify".into(),
        });
        assert_eq!(p.current_phase.as_deref(), Some("verify"));
        assert_eq!(p.phases.len(), 2);
        assert_eq!(p.phases[0].title, "research");
        assert_eq!(p.phases[1].title, "verify");
    }

    #[test]
    fn agent_started_bumps_global_and_matching_phase() {
        let mut p = RunProgress::default();
        p.apply(&WorkflowEvent::PhaseStarted {
            title: "research".into(),
        });
        p.apply(&WorkflowEvent::AgentStarted {
            label: "a".into(),
            phase: Some("research".into()),
        });
        p.apply(&WorkflowEvent::AgentStarted {
            label: "b".into(),
            phase: Some("research".into()),
        });
        // An agent with no phase bumps only the global counter.
        p.apply(&WorkflowEvent::AgentStarted {
            label: "c".into(),
            phase: None,
        });
        assert_eq!(p.agents_started, 3);
        assert_eq!(p.phases[0].agents_started, 2);
    }

    #[test]
    fn finished_and_replayed_both_accumulate_tokens_but_count_separately() {
        let mut p = RunProgress::default();
        p.apply(&WorkflowEvent::AgentFinished {
            label: "a".into(),
            usage: usage(100, 50),
        });
        p.apply(&WorkflowEvent::AgentReplayed {
            label: "b".into(),
            usage: usage(10, 5),
        });
        assert_eq!(p.agents_finished, 1);
        assert_eq!(p.agents_replayed, 1);
        // Tokens accumulate across both finished and replayed.
        assert_eq!(p.total_tokens.input_tokens, 110);
        assert_eq!(p.total_tokens.output_tokens, 55);
    }

    #[test]
    fn skipped_failed_and_log_counters() {
        let mut p = RunProgress::default();
        p.apply(&WorkflowEvent::AgentSkipped { label: "a".into() });
        p.apply(&WorkflowEvent::AgentFailed {
            label: "b".into(),
            error: "boom".into(),
        });
        p.apply(&WorkflowEvent::AgentFailed {
            label: "c".into(),
            error: "bang".into(),
        });
        p.apply(&WorkflowEvent::LogLine { msg: "tick".into() });
        assert_eq!(p.agents_skipped, 1);
        assert_eq!(p.agents_failed, 2);
        assert_eq!(p.log_lines, 1);
    }

    #[test]
    fn callback_folds_events_into_snapshot() {
        let tracker = ProgressTracker::new();
        let cb = tracker.callback();
        cb(WorkflowEvent::PhaseStarted { title: "p1".into() });
        cb(WorkflowEvent::AgentStarted {
            label: "a".into(),
            phase: Some("p1".into()),
        });
        cb(WorkflowEvent::AgentFinished {
            label: "a".into(),
            usage: usage(7, 3),
        });
        let snap = tracker.snapshot();
        assert_eq!(snap.current_phase.as_deref(), Some("p1"));
        assert_eq!(snap.agents_started, 1);
        assert_eq!(snap.agents_finished, 1);
        assert_eq!(snap.total_tokens.input_tokens, 7);
    }
}
