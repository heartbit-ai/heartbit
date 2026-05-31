//! Plan artifact + replan decision logic (capability 8 of the browser-bot spec).
//!
//! Deep-research convergence (Manus, Plan-and-Act, Online-Mind2Web/WebJudge) is
//! unanimous: for an a11y-grounded web agent, *grounding is not the gap* —
//! long-horizon **planning and recovery** is. The single biggest lever over a
//! bare ReAct loop is an explicit plan with **replanning on failure**: decompose
//! the task into ordered subgoals, execute one at a time, verify each, and when a
//! step stalls, revise the plan instead of blindly retrying the same action.
//!
//! Two hard-won design rules from the research are baked in here:
//!
//! 1. **The plan is a harness-owned artifact, not something the executor
//!    rewrites every turn.** Manus found an agent that rewrites its whole
//!    `todo.md` each step burned ~1/3 of all actions on bookkeeping. So [`Plan`]
//!    is rendered deterministically by the harness ([`Plan::render`]) and the
//!    model only *flips a step's status*; it never re-emits the list to make
//!    progress.
//! 2. **The replan trigger is a pure decision over the verified step outcome.**
//!    [`replan_decision`] is a clock-free, LLM-free state machine mapping
//!    `(StepOutcome, attempts, budget)` to a [`ReplanAction`] — so every branch
//!    (advance / retry / replan / give up / done) is exhaustively unit-testable,
//!    mirroring [`super::settle::step`] and [`super::verify::diff`]. The LLM
//!    planner/executor is a thin async shell layered on top (built later, like
//!    settle's driver), keeping the load-bearing logic deterministic.

use std::fmt::Write as _;

/// Lifecycle of a single plan step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    /// Not started.
    Pending,
    /// Currently executing.
    Active,
    /// Verified complete.
    Done,
    /// Abandoned after exhausting retries/replans (recorded, not silently
    /// dropped — keeping failures visible is itself a research-backed tactic).
    Failed,
}

impl StepStatus {
    /// The checkbox-style glyph used in [`Plan::render`].
    fn glyph(self) -> char {
        match self {
            StepStatus::Pending => ' ',
            StepStatus::Active => '>',
            StepStatus::Done => 'x',
            StepStatus::Failed => '!',
        }
    }
}

/// One subgoal in a [`Plan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanStep {
    /// Human/LLM-readable subgoal description.
    pub description: String,
    /// Current lifecycle status.
    pub status: StepStatus,
}

impl PlanStep {
    /// A fresh `Pending` step.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            status: StepStatus::Pending,
        }
    }
}

/// An ordered, harness-owned task plan: a top-level goal decomposed into steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The overall task this plan accomplishes (re-surfaced each turn to fight
    /// goal-drift / "lost in the middle" on long runs).
    pub goal: String,
    /// Ordered subgoals.
    pub steps: Vec<PlanStep>,
}

impl Plan {
    /// Build a plan from a goal and an ordered list of step descriptions.
    pub fn new(
        goal: impl Into<String>,
        steps: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            goal: goal.into(),
            steps: steps.into_iter().map(PlanStep::new).collect(),
        }
    }

    /// Index of the first non-terminal (`Pending`/`Active`) step — the one the
    /// executor should be working on. `None` when every step is `Done`/`Failed`.
    pub fn current(&self) -> Option<usize> {
        self.steps
            .iter()
            .position(|s| matches!(s.status, StepStatus::Pending | StepStatus::Active))
    }

    /// Whether every step reached a terminal status (`Done` or `Failed`).
    pub fn is_complete(&self) -> bool {
        self.current().is_none()
    }

    /// Whether all steps are specifically `Done` (the success condition).
    pub fn all_done(&self) -> bool {
        !self.steps.is_empty() && self.steps.iter().all(|s| s.status == StepStatus::Done)
    }

    /// Mark the step at `idx`, returning `false` if `idx` is out of range (no
    /// panic — library code must not index blindly).
    pub fn set_status(&mut self, idx: usize, status: StepStatus) -> bool {
        match self.steps.get_mut(idx) {
            Some(step) => {
                step.status = status;
                true
            }
            None => false,
        }
    }

    /// Deterministic, cache-stable rendering for re-injection into the prompt
    /// near the context tail (goal recitation). The harness owns this; the model
    /// reads it and only requests status flips — it never reproduces the list to
    /// make progress.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(self.goal.len() + self.steps.len() * 32);
        let _ = writeln!(out, "Goal: {}", self.goal);
        for (i, step) in self.steps.iter().enumerate() {
            let _ = writeln!(
                out,
                "{}. [{}] {}",
                i + 1,
                step.status.glyph(),
                step.description
            );
        }
        // Trim the final newline so the artifact concatenates predictably.
        if out.ends_with('\n') {
            out.pop();
        }
        out
    }
}

/// The verified outcome of executing the current step (the input signal to
/// [`replan_decision`]). Derived by the harness from the executor's
/// [`AgentOutput`](crate::AgentOutput) plus a [`super::verify`] state-diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// The step's subgoal was verified achieved.
    Succeeded,
    /// The action ran but produced no progress (e.g. a verified
    /// [`super::verify::ActionEffect::NoOp`]) or a recoverable error — worth a
    /// bounded retry of the *same* step.
    Stalled,
    /// A hard failure that retrying the same step won't fix (the page diverged
    /// from the plan's assumptions) — the plan itself needs revising.
    Diverged,
}

/// What the loop should do next, decided purely from the step outcome and the
/// retry/replan budget consumed so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplanAction {
    /// Step succeeded: mark it `Done` and move to the next step.
    Advance,
    /// Retry the *same* step (a bounded number of times) before escalating.
    Retry,
    /// Ask the planner to revise the remaining steps (the page diverged or the
    /// step exhausted its retries).
    Replan,
    /// Stop: the whole task succeeded (no steps remain).
    Done,
    /// Stop: out of retry/replan budget — give up with the plan as-is.
    GiveUp,
}

/// Bounds on recovery effort, so a stuck agent can't loop forever.
#[derive(Debug, Clone, Copy)]
pub struct ReplanBudget {
    /// Max retries of a single step before it escalates to a replan.
    pub max_step_retries: u32,
    /// Max whole-plan revisions before giving up.
    pub max_replans: u32,
}

impl Default for ReplanBudget {
    fn default() -> Self {
        Self {
            max_step_retries: 2,
            max_replans: 3,
        }
    }
}

/// Pure replan decision: given the verified `outcome` of the current step, how
/// many times that step has already been retried (`step_attempts`, not counting
/// the first try), how many whole-plan replans have happened (`replans_used`),
/// and whether any non-terminal step remains (`has_current`), decide the next
/// [`ReplanAction`].
///
/// Branch order is significant and exhaustively tested:
/// - `Succeeded` + no step remaining → [`ReplanAction::Done`].
/// - `Succeeded` + a step remaining → [`ReplanAction::Advance`].
/// - `Stalled` + retries left → [`ReplanAction::Retry`]; else fall through to
///   the replan path (a step that won't progress is a planning problem).
/// - `Diverged`, or a stall that exhausted step retries → [`ReplanAction::Replan`]
///   while replan budget remains, else [`ReplanAction::GiveUp`].
pub fn replan_decision(
    budget: &ReplanBudget,
    outcome: StepOutcome,
    step_attempts: u32,
    replans_used: u32,
    has_current: bool,
) -> ReplanAction {
    match outcome {
        StepOutcome::Succeeded => {
            if has_current {
                ReplanAction::Advance
            } else {
                ReplanAction::Done
            }
        }
        StepOutcome::Stalled if step_attempts < budget.max_step_retries => ReplanAction::Retry,
        // Diverged, or stalled past the per-step retry ceiling: revise the plan
        // if we still can, otherwise give up rather than spin.
        StepOutcome::Stalled | StepOutcome::Diverged => {
            if replans_used < budget.max_replans {
                ReplanAction::Replan
            } else {
                ReplanAction::GiveUp
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Plan artifact ---

    #[test]
    fn current_points_at_first_unfinished_step() {
        let mut p = Plan::new("book a flight", ["search", "select", "pay"]);
        assert_eq!(p.current(), Some(0));
        assert!(p.set_status(0, StepStatus::Done));
        assert_eq!(p.current(), Some(1), "advances past the Done step");
        assert!(p.set_status(1, StepStatus::Active));
        assert_eq!(p.current(), Some(1), "an Active step is still 'current'");
    }

    #[test]
    fn is_complete_and_all_done_distinguish_failure() {
        let mut p = Plan::new("task", ["a", "b"]);
        assert!(!p.is_complete());
        assert!(!p.all_done());
        p.set_status(0, StepStatus::Done);
        p.set_status(1, StepStatus::Failed);
        assert!(p.is_complete(), "all steps terminal → complete");
        assert!(!p.all_done(), "a Failed step means not all_done");
    }

    #[test]
    fn all_done_is_false_for_empty_plan() {
        let p = Plan::new("nothing", Vec::<String>::new());
        assert!(!p.all_done(), "an empty plan has not accomplished anything");
        assert!(p.is_complete(), "but it has no pending work");
    }

    #[test]
    fn set_status_out_of_range_is_false_not_panic() {
        let mut p = Plan::new("task", ["only"]);
        assert!(!p.set_status(9, StepStatus::Done));
        assert_eq!(p.steps[0].status, StepStatus::Pending, "unchanged");
    }

    #[test]
    fn render_is_deterministic_and_shows_status_glyphs() {
        let mut p = Plan::new("buy milk", ["open store", "add to cart", "checkout"]);
        p.set_status(0, StepStatus::Done);
        p.set_status(1, StepStatus::Active);
        let r = p.render();
        assert_eq!(
            r,
            "Goal: buy milk\n1. [x] open store\n2. [>] add to cart\n3. [ ] checkout"
        );
        // Cache-stability: rendering twice yields byte-identical output.
        assert_eq!(p.render(), r);
    }

    // --- pure replan_decision state machine ---

    fn bud() -> ReplanBudget {
        ReplanBudget {
            max_step_retries: 2,
            max_replans: 3,
        }
    }

    #[test]
    fn success_with_steps_remaining_advances() {
        assert_eq!(
            replan_decision(&bud(), StepOutcome::Succeeded, 0, 0, true),
            ReplanAction::Advance
        );
    }

    #[test]
    fn success_with_no_steps_remaining_is_done() {
        assert_eq!(
            replan_decision(&bud(), StepOutcome::Succeeded, 0, 0, false),
            ReplanAction::Done
        );
    }

    #[test]
    fn stall_retries_until_ceiling_then_replans() {
        // attempts 0 and 1 (< max_step_retries=2) → Retry.
        assert_eq!(
            replan_decision(&bud(), StepOutcome::Stalled, 0, 0, true),
            ReplanAction::Retry
        );
        assert_eq!(
            replan_decision(&bud(), StepOutcome::Stalled, 1, 0, true),
            ReplanAction::Retry
        );
        // attempt 2 (== ceiling) → escalate to Replan (retrying won't help).
        assert_eq!(
            replan_decision(&bud(), StepOutcome::Stalled, 2, 0, true),
            ReplanAction::Replan
        );
    }

    #[test]
    fn diverged_replans_immediately_without_retrying() {
        // A hard divergence skips per-step retries entirely.
        assert_eq!(
            replan_decision(&bud(), StepOutcome::Diverged, 0, 0, true),
            ReplanAction::Replan
        );
    }

    #[test]
    fn replan_budget_exhaustion_gives_up() {
        // replans_used == max_replans=3 → no revisions left → GiveUp.
        assert_eq!(
            replan_decision(&bud(), StepOutcome::Diverged, 0, 3, true),
            ReplanAction::GiveUp
        );
        assert_eq!(
            replan_decision(&bud(), StepOutcome::Stalled, 5, 3, true),
            ReplanAction::GiveUp
        );
    }

    #[test]
    fn zero_retry_budget_escalates_on_first_stall() {
        let b = ReplanBudget {
            max_step_retries: 0,
            max_replans: 1,
        };
        // No retries allowed → first stall goes straight to Replan.
        assert_eq!(
            replan_decision(&b, StepOutcome::Stalled, 0, 0, true),
            ReplanAction::Replan
        );
    }
}
