//! [`WorkflowCtx`] — the shared run context threaded into every combinator.
//!
//! Holds the run-wide concurrency cap (a single [`Semaphore`] shared across the
//! whole run), the runaway agent backstop, a cancellation token, an optional
//! event sink, and the "default phase" applied to subsequently-issued agents.
//! It is cheap to [`Clone`] (an `Arc` inside) so combinator thunks can capture
//! their own handle.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::error::Error;
use crate::llm::BoxedProvider;

use super::budget::Budget;
use super::event::{OnWorkflowEvent, WorkflowEvent};
use super::journal::{ResumeMode, RunJournal};

/// Hard upper bound on concurrently-running agents, matching Claude Code's
/// `min(16, cores - 2)`.
const MAX_CONCURRENCY_CAP: usize = 16;

/// Default total-agent runaway backstop (Claude Code uses 1000 per run).
const DEFAULT_MAX_AGENTS: u64 = 1000;

/// Default concurrency cap: `min(16, available_parallelism - 2)`, at least 1.
///
/// Mirrors the idiom in [`crate::agent::batch`] but clamps to the workflow cap.
pub(crate) fn default_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .saturating_sub(2)
        .clamp(1, MAX_CONCURRENCY_CAP)
}

/// A run-level limit breach that must halt the whole run, distinct from an
/// agent-domain failure. When one occurs inside a combinator the breach is
/// recorded (set-once) and the run-wide cancellation token is fired, so the
/// breach survives the combinator's `Err -> None` collapse and the caller can
/// detect it via [`WorkflowCtx::control_breach`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ControlBreach {
    /// The shared token budget was exhausted.
    Budget {
        /// Weighted cost recorded when admission was refused.
        used: u64,
        /// The configured ceiling.
        limit: u64,
    },
    /// The runaway agent backstop was hit.
    AgentBackstop {
        /// The configured maximum agents per run.
        limit: u64,
    },
}

/// Shared, cheaply-cloneable workflow run context.
#[derive(Clone)]
pub struct WorkflowCtx {
    inner: Arc<CtxInner>,
}

struct CtxInner {
    /// Type-erased provider shared by every agent leaf.
    provider: Arc<BoxedProvider>,
    /// Optional default tool set wired into every agent leaf that does not
    /// supply its own per-call tools. Shared with nested workflows.
    base_tools: Option<Arc<Vec<Arc<dyn crate::tool::Tool>>>>,
    /// Global concurrency limiter; permits acquired only at the agent() leaf.
    sem: Arc<Semaphore>,
    /// Monotonic count of agents ever issued (runaway backstop; never
    /// decremented). `Arc` so a nested workflow shares the same counter.
    spawned: Arc<AtomicU64>,
    /// Maximum total agents per run.
    max_agents: u64,
    /// Shared hard-ceiling token-equivalent spend pool.
    budget: Budget,
    /// First run-level limit breach, if any (set-once). `Arc` so a nested
    /// workflow's breach is visible on the parent ctx.
    control: Arc<Mutex<Option<ControlBreach>>>,
    /// Optional resume journal: memoizes agent outputs for deterministic replay.
    journal: Option<Arc<RunJournal>>,
    /// Optional workflow event sink.
    events: Option<Arc<OnWorkflowEvent>>,
    /// Cooperative cancellation (pause/stop); P1 only races it to `Ok(None)`.
    cancel: CancellationToken,
    /// Default phase for subsequently-issued agents. `std` lock: never held
    /// across `.await` (snapshotted at `AgentCall` construction). NOT shared
    /// with a nested workflow — each level groups its own agents.
    default_phase: RwLock<Option<Arc<str>>>,
    /// Nesting depth: 0 for a top-level run, 1 for a `workflow()` child. A
    /// second level is rejected by [`WorkflowCtx::nested`].
    depth: u8,
}

impl std::fmt::Debug for WorkflowCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkflowCtx")
            .field("max_agents", &self.inner.max_agents)
            .field("permits", &self.inner.sem.available_permits())
            .field("spawned", &self.inner.spawned.load(Ordering::Relaxed))
            .field("budget_spent", &self.inner.budget.spent())
            .field("budget_total", &self.inner.budget.total())
            .finish()
    }
}

impl WorkflowCtx {
    /// Start building a context from a type-erased provider.
    pub fn builder(provider: Arc<BoxedProvider>) -> WorkflowCtxBuilder {
        WorkflowCtxBuilder {
            provider,
            base_tools: None,
            max_concurrency: None,
            max_agents: None,
            budget: None,
            journal: None,
            events: None,
            cancel: None,
        }
    }

    /// Whether the run has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancel.is_cancelled()
    }

    /// Stop the run: fires the shared cancellation token so in-flight agents
    /// wind down at their next leaf cancel-check (and combinator tasks abort).
    /// Idempotent. Restarting a stopped run is done by re-running with a resume
    /// [`journal`](WorkflowCtxBuilder::journal): completed agents replay, the
    /// stopped ones run live.
    pub fn stop(&self) {
        self.inner.cancel.cancel();
    }

    /// A clone of the run's cancellation token (for external pause/stop wiring).
    pub fn cancellation_token(&self) -> CancellationToken {
        self.inner.cancel.clone()
    }

    /// Total agents issued so far (monotonic; includes rejected admissions).
    pub fn spawned(&self) -> u64 {
        self.inner.spawned.load(Ordering::Relaxed)
    }

    /// The configured runaway agent backstop.
    pub fn max_agents(&self) -> u64 {
        self.inner.max_agents
    }

    /// The shared token-equivalent budget pool.
    pub fn budget(&self) -> &Budget {
        &self.inner.budget
    }

    /// Remaining budget headroom, or `u64::MAX` if unbounded. Drives
    /// `loop-until-budget`: `while ctx.remaining() > threshold { … }`.
    pub fn remaining(&self) -> u64 {
        self.inner.budget.remaining()
    }

    /// The first run-level limit breach, if one has occurred.
    pub fn control_breach(&self) -> Option<ControlBreach> {
        self.inner.control.lock().ok().and_then(|g| *g)
    }

    // ----- pub(crate) seams consumed by the agent() leaf (slice 5) -----

    /// A clone of the shared, type-erased provider for building an agent leaf.
    pub(crate) fn provider(&self) -> Arc<BoxedProvider> {
        Arc::clone(&self.inner.provider)
    }

    /// The ctx's default tool set (cloned `Vec`), if any. Used by an agent leaf
    /// that supplies no per-call tools.
    pub(crate) fn base_tools(&self) -> Option<Vec<Arc<dyn crate::tool::Tool>>> {
        self.inner.base_tools.as_ref().map(|t| t.as_ref().clone())
    }

    /// A clone of the global concurrency limiter; acquired only at the leaf.
    pub(crate) fn semaphore(&self) -> Arc<Semaphore> {
        Arc::clone(&self.inner.sem)
    }

    /// Borrow the run's cancellation token (for the leaf's `select!` race).
    pub(crate) fn cancel_token(&self) -> &CancellationToken {
        &self.inner.cancel
    }

    /// Reserve one slot against the runaway backstop. Monotonic: a rejected
    /// admission still counts (it is a backstop, not a live gauge). On breach,
    /// records the control breach and fires run-wide cancellation.
    pub(crate) fn register_agent(&self) -> Result<(), Error> {
        let prior = self.inner.spawned.fetch_add(1, Ordering::Relaxed);
        if prior >= self.inner.max_agents {
            let limit = self.inner.max_agents;
            self.record_breach(ControlBreach::AgentBackstop { limit });
            return Err(Error::AgentBudgetExceeded { limit });
        }
        Ok(())
    }

    /// Admit one agent against the shared budget. On exhaustion, records the
    /// control breach, fires run-wide cancellation, and returns the error.
    pub(crate) fn admit_budget(&self) -> Result<(), Error> {
        match self.inner.budget.check_admit() {
            Ok(()) => Ok(()),
            Err(err) => {
                if let Error::BudgetExceeded { used, limit } = err {
                    self.record_breach(ControlBreach::Budget { used, limit });
                }
                Err(err)
            }
        }
    }

    /// Record one agent's completed cost against the shared budget.
    pub(crate) fn record_spend(&self, usage: &crate::llm::types::TokenUsage) {
        self.inner.budget.record(usage);
    }

    /// A clone of the resume journal handle, if journaling is enabled.
    pub(crate) fn journal_arc(&self) -> Option<Arc<RunJournal>> {
        self.inner.journal.clone()
    }

    /// Create a one-level-deeper child context that **shares** this run's
    /// provider, concurrency limiter, runaway backstop counter, budget pool,
    /// breach state, journal, event sink, and cancellation token — but gets a
    /// fresh default-phase. Rejects a second level with [`Error::Config`].
    pub(crate) fn nested(&self) -> Result<WorkflowCtx, Error> {
        if self.inner.depth >= 1 {
            return Err(Error::Config(
                "workflow() may nest only one level deep".into(),
            ));
        }
        Ok(WorkflowCtx {
            inner: Arc::new(CtxInner {
                provider: Arc::clone(&self.inner.provider),
                base_tools: self.inner.base_tools.clone(),
                sem: Arc::clone(&self.inner.sem),
                spawned: Arc::clone(&self.inner.spawned),
                max_agents: self.inner.max_agents,
                budget: self.inner.budget.clone(),
                control: Arc::clone(&self.inner.control),
                journal: self.inner.journal.clone(),
                events: self.inner.events.clone(),
                cancel: self.inner.cancel.clone(),
                default_phase: RwLock::new(None),
                depth: self.inner.depth + 1,
            }),
        })
    }

    /// Record a run-level breach (set-once) and fire run-wide cancellation so
    /// in-flight combinator tasks wind down. Idempotent on the stored breach.
    pub(crate) fn record_breach(&self, breach: ControlBreach) {
        if let Ok(mut guard) = self.inner.control.lock()
            && guard.is_none()
        {
            *guard = Some(breach);
        }
        self.inner.cancel.cancel();
    }

    /// Emit a workflow event if a sink is installed.
    pub(crate) fn emit(&self, event: WorkflowEvent) {
        if let Some(cb) = &self.inner.events {
            cb(event);
        }
    }

    /// Snapshot the current default phase (the phase newly-issued agents adopt).
    pub(crate) fn current_phase(&self) -> Option<Arc<str>> {
        self.inner.default_phase.read().ok().and_then(|g| g.clone())
    }

    /// Replace the default phase, returning the prior value (for RAII restore).
    pub(crate) fn swap_default_phase(&self, next: Option<Arc<str>>) -> Option<Arc<str>> {
        match self.inner.default_phase.write() {
            Ok(mut g) => std::mem::replace(&mut *g, next),
            Err(_) => None,
        }
    }
}

/// Builder for [`WorkflowCtx`].
pub struct WorkflowCtxBuilder {
    provider: Arc<BoxedProvider>,
    base_tools: Option<Arc<Vec<Arc<dyn crate::tool::Tool>>>>,
    max_concurrency: Option<usize>,
    max_agents: Option<u64>,
    budget: Option<Budget>,
    journal: Option<Arc<RunJournal>>,
    events: Option<Arc<OnWorkflowEvent>>,
    cancel: Option<CancellationToken>,
}

impl WorkflowCtxBuilder {
    /// Set the default tool set wired into every agent leaf that does not supply
    /// its own per-call tools (see [`AgentCall::tools`](super::agent::AgentCall::tools)).
    pub fn base_tools(mut self, tools: Vec<Arc<dyn crate::tool::Tool>>) -> Self {
        self.base_tools = Some(Arc::new(tools));
        self
    }

    /// Override the concurrency cap (must be >= 1).
    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = Some(n);
        self
    }

    /// Override the runaway agent backstop (must be >= 1).
    pub fn max_agents(mut self, n: u64) -> Self {
        self.max_agents = Some(n);
        self
    }

    /// Set a hard token-equivalent budget ceiling for the run (`0` ⇒ unbounded).
    /// Default is unbounded.
    pub fn budget(mut self, total: u64) -> Self {
        self.budget = Some(Budget::with_total(total));
        self
    }

    /// Share an existing [`Budget`] pool (e.g. a parent run's, for nested
    /// workflows). This and [`budget`](Self::budget) write the same slot, so
    /// whichever is called *last* wins (last-write-wins, not unconditional
    /// precedence).
    pub fn budget_pool(mut self, budget: Budget) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Enable deterministic resume by journaling agent outputs to `path` (a
    /// `.jsonl` file). [`ResumeMode::Fresh`] starts a new journal;
    /// [`ResumeMode::Resume`] replays matching calls from an existing one.
    /// Returns an error if the journal file cannot be opened.
    ///
    /// See [`RunJournal`] for the soundness scope (return values, not side
    /// effects; single process; fixed provider).
    pub fn journal(
        mut self,
        path: impl AsRef<std::path::Path>,
        mode: ResumeMode,
    ) -> Result<Self, Error> {
        self.journal = Some(Arc::new(RunJournal::open(path.as_ref(), mode)?));
        Ok(self)
    }

    /// Share an existing [`RunJournal`] handle (e.g. a parent run's).
    pub fn journal_handle(mut self, journal: Arc<RunJournal>) -> Self {
        self.journal = Some(journal);
        self
    }

    /// Install a workflow event sink.
    pub fn on_event(mut self, callback: Arc<OnWorkflowEvent>) -> Self {
        self.events = Some(callback);
        self
    }

    /// Provide an external cancellation token (defaults to a fresh one).
    pub fn cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Build the context.
    ///
    /// Rejects a zero concurrency cap or zero agent backstop (consistent with
    /// the zero-rejection in [`crate::agent::batch`] and the workflow agents).
    pub fn build(self) -> Result<WorkflowCtx, Error> {
        let max_concurrency = self.max_concurrency.unwrap_or_else(default_concurrency);
        if max_concurrency == 0 {
            return Err(Error::Config(
                "WorkflowCtx max_concurrency must be at least 1".into(),
            ));
        }
        let max_agents = self.max_agents.unwrap_or(DEFAULT_MAX_AGENTS);
        if max_agents == 0 {
            return Err(Error::Config(
                "WorkflowCtx max_agents must be at least 1".into(),
            ));
        }
        Ok(WorkflowCtx {
            inner: Arc::new(CtxInner {
                provider: self.provider,
                base_tools: self.base_tools,
                sem: Arc::new(Semaphore::new(max_concurrency)),
                spawned: Arc::new(AtomicU64::new(0)),
                max_agents,
                budget: self.budget.unwrap_or_default(),
                control: Arc::new(Mutex::new(None)),
                journal: self.journal,
                events: self.events,
                cancel: self.cancel.unwrap_or_default(),
                default_phase: RwLock::new(None),
                depth: 0,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_helpers::MockProvider;

    fn provider() -> Arc<BoxedProvider> {
        Arc::new(BoxedProvider::new(MockProvider::new(vec![])))
    }

    #[test]
    fn default_concurrency_in_range() {
        let c = default_concurrency();
        assert!((1..=MAX_CONCURRENCY_CAP).contains(&c), "got {c}");
    }

    #[test]
    fn builder_rejects_zero_concurrency() {
        let result = WorkflowCtx::builder(provider()).max_concurrency(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_zero_max_agents() {
        let result = WorkflowCtx::builder(provider()).max_agents(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_accepts_defaults() {
        let ctx = WorkflowCtx::builder(provider()).build().expect("build");
        assert_eq!(ctx.max_agents(), DEFAULT_MAX_AGENTS);
        assert!(!ctx.is_cancelled());
    }

    // ----- P6: nesting + stop() -----

    fn budgeted(total: u64) -> WorkflowCtx {
        WorkflowCtx::builder(provider())
            .budget(total)
            .build()
            .expect("build")
    }

    #[test]
    fn nested_shares_budget_pool() {
        let parent = budgeted(1000);
        let child = parent.nested().expect("nest");
        // A spend recorded through the child is visible on the parent's budget.
        child.record_spend(&crate::llm::types::TokenUsage {
            input_tokens: 30,
            ..Default::default()
        });
        assert_eq!(parent.budget().spent(), 30);
        assert_eq!(child.budget().spent(), 30);
    }

    #[test]
    fn nested_shares_runaway_backstop() {
        let parent = WorkflowCtx::builder(provider())
            .max_agents(2)
            .build()
            .expect("build");
        let child = parent.nested().expect("nest");
        // Two registrations across parent+child exhaust the shared backstop.
        assert!(parent.register_agent().is_ok()); // prior 0
        assert!(child.register_agent().is_ok()); // prior 1
        assert!(child.register_agent().is_err()); // prior 2 >= 2 -> breach
        // The monotonic counter is shared, not per-ctx.
        assert_eq!(parent.spawned(), 3);
        assert_eq!(child.spawned(), 3);
    }

    #[test]
    fn child_breach_cancels_parent() {
        let parent = budgeted(1000);
        let child = parent.nested().expect("nest");
        child.record_breach(ControlBreach::AgentBackstop { limit: 5 });
        // Breach + cancellation propagate to the shared parent ctx.
        assert!(parent.is_cancelled());
        assert!(matches!(
            parent.control_breach(),
            Some(ControlBreach::AgentBackstop { limit: 5 })
        ));
    }

    #[test]
    fn nesting_is_one_level_only() {
        let parent = budgeted(1000);
        let child = parent.nested().expect("first level ok");
        let grandchild = child.nested();
        assert!(
            grandchild.is_err(),
            "a second nesting level must be rejected"
        );
    }

    #[test]
    fn child_has_independent_default_phase() {
        let parent = budgeted(1000);
        parent.swap_default_phase(Some(std::sync::Arc::from("parent-phase")));
        let child = parent.nested().expect("nest");
        // The child starts with no default phase of its own.
        assert!(child.current_phase().is_none());
        // And setting the child's phase does not leak back to the parent.
        child.swap_default_phase(Some(std::sync::Arc::from("child-phase")));
        assert_eq!(parent.current_phase().as_deref(), Some("parent-phase"));
        assert_eq!(child.current_phase().as_deref(), Some("child-phase"));
    }

    #[test]
    fn stop_sets_cancelled() {
        let ctx = budgeted(1000);
        assert!(!ctx.is_cancelled());
        ctx.stop();
        assert!(ctx.is_cancelled());
    }
}
