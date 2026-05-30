//! [`WorkflowCtx`] — the shared run context threaded into every combinator.
//!
//! Holds the run-wide concurrency cap (a single [`Semaphore`] shared across the
//! whole run), the runaway agent backstop, a cancellation token, an optional
//! event sink, and the "default phase" applied to subsequently-issued agents.
//! It is cheap to [`Clone`] (an `Arc` inside) so combinator thunks can capture
//! their own handle.

use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::error::Error;
use crate::llm::BoxedProvider;

use super::event::{OnWorkflowEvent, WorkflowEvent};

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

/// Shared, cheaply-cloneable workflow run context.
#[derive(Clone)]
pub struct WorkflowCtx {
    inner: Arc<CtxInner>,
}

struct CtxInner {
    /// Type-erased provider shared by every agent leaf.
    provider: Arc<BoxedProvider>,
    /// Global concurrency limiter; permits acquired only at the agent() leaf.
    sem: Arc<Semaphore>,
    /// Monotonic count of agents ever issued (runaway backstop; never decremented).
    spawned: AtomicU64,
    /// Maximum total agents per run.
    max_agents: u64,
    /// Optional workflow event sink.
    events: Option<Arc<OnWorkflowEvent>>,
    /// Cooperative cancellation (pause/stop); P1 only races it to `Ok(None)`.
    cancel: CancellationToken,
    /// Default phase for subsequently-issued agents. `std` lock: never held
    /// across `.await` (snapshotted at `AgentCall` construction).
    default_phase: RwLock<Option<Arc<str>>>,
}

impl std::fmt::Debug for WorkflowCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkflowCtx")
            .field("max_agents", &self.inner.max_agents)
            .field("permits", &self.inner.sem.available_permits())
            .field("spawned", &self.inner.spawned.load(Ordering::Relaxed))
            .finish()
    }
}

impl WorkflowCtx {
    /// Start building a context from a type-erased provider.
    pub fn builder(provider: Arc<BoxedProvider>) -> WorkflowCtxBuilder {
        WorkflowCtxBuilder {
            provider,
            max_concurrency: None,
            max_agents: None,
            events: None,
            cancel: None,
        }
    }

    /// Whether the run has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancel.is_cancelled()
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

    // ----- pub(crate) seams consumed by the agent() leaf (slice 5) -----

    /// A clone of the shared, type-erased provider for building an agent leaf.
    pub(crate) fn provider(&self) -> Arc<BoxedProvider> {
        Arc::clone(&self.inner.provider)
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
    /// admission still counts (it is a backstop, not a live gauge).
    pub(crate) fn register_agent(&self) -> Result<(), Error> {
        let prior = self.inner.spawned.fetch_add(1, Ordering::Relaxed);
        if prior >= self.inner.max_agents {
            return Err(Error::AgentBudgetExceeded {
                limit: self.inner.max_agents,
            });
        }
        Ok(())
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
    max_concurrency: Option<usize>,
    max_agents: Option<u64>,
    events: Option<Arc<OnWorkflowEvent>>,
    cancel: Option<CancellationToken>,
}

impl WorkflowCtxBuilder {
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
                sem: Arc::new(Semaphore::new(max_concurrency)),
                spawned: AtomicU64::new(0),
                max_agents,
                events: self.events,
                cancel: self.cancel.unwrap_or_default(),
                default_phase: RwLock::new(None),
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
}
