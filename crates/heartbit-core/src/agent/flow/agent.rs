//! [`agent`] — the atomic unit of a workflow: spawn one sub-agent.
//!
//! `agent(&ctx, prompt)` returns a fluent [`AgentCall`] that owns a clone of the
//! [`WorkflowCtx`] (so the returned `run()` future is `'static` and can be moved
//! into a [`parallel`](super::parallel)/[`pipeline`](super::pipeline) task). The
//! leaf is where the run-wide concurrency cap, the runaway backstop, and (in
//! later phases) the token budget and journal are enforced.
//!
//! P1 ships the text terminal only: `run() -> Result<Option<String>, Error>`,
//! where `Ok(None)` means the call was skipped (e.g. the run was cancelled),
//! mirroring Claude Code's `null`. The `Err -> None` collapse is the job of the
//! *combinators*, never of `agent()` — control errors (backstop, cancellation,
//! and later the budget) must propagate.

use std::sync::Arc;

use crate::agent::{AgentOutput, AgentRunner};
use crate::error::Error;

use super::ctx::WorkflowCtx;
use super::event::WorkflowEvent;

/// Per-call options for an [`agent`] leaf. `#[non_exhaustive]` so later phases
/// can add fields (schema, model, isolation, …) without breaking callers.
#[derive(Clone, Default, Debug)]
#[non_exhaustive]
pub struct AgentOpts {
    /// Display label for events/observability. Defaults to `"agent"`.
    pub label: Option<String>,
    /// Explicit phase override. When unset, the leaf adopts the context's
    /// default phase snapshotted at construction time.
    pub phase: Option<String>,
}

/// A fluent, owned builder for one [`agent`] leaf. Created by [`agent`].
pub struct AgentCall {
    ctx: WorkflowCtx,
    prompt: String,
    opts: AgentOpts,
    /// Default phase captured when this call was *constructed*, so concurrently
    /// running agents never observe a torn phase if another phase begins later.
    phase_snapshot: Option<Arc<str>>,
}

/// Begin an [`AgentCall`] against `ctx`. Snapshots the current default phase.
pub fn agent(ctx: &WorkflowCtx, prompt: impl Into<String>) -> AgentCall {
    let phase_snapshot = ctx.current_phase();
    AgentCall {
        ctx: ctx.clone(),
        prompt: prompt.into(),
        opts: AgentOpts::default(),
        phase_snapshot,
    }
}

impl AgentCall {
    /// Set the display label (used in events and as the runner name).
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.opts.label = Some(label.into());
        self
    }

    /// Override the phase this call is grouped under.
    pub fn phase(mut self, phase: impl Into<String>) -> Self {
        self.opts.phase = Some(phase.into());
        self
    }

    /// The effective phase: explicit override, else the construction snapshot.
    fn effective_phase(&self) -> Option<String> {
        self.opts
            .phase
            .clone()
            .or_else(|| self.phase_snapshot.as_deref().map(str::to_owned))
    }

    /// Run the agent leaf to completion.
    ///
    /// Leaf sequence: acquire a concurrency permit -> register against the
    /// runaway backstop -> admit against the shared budget -> race the run
    /// against cancellation -> on success record the spend and emit
    /// `AgentFinished`. `Ok(None)` is returned only when the run was cancelled
    /// (skipped); agent-domain failures return `Err` so the combinators can
    /// decide whether to collapse them to `None`. The backstop and budget are
    /// *control* errors: they record a breach and fire run-wide cancellation,
    /// so they survive a combinator's `Err -> None` collapse.
    pub async fn run(self) -> Result<Option<String>, Error> {
        let label = self
            .opts
            .label
            .clone()
            .unwrap_or_else(|| "agent".to_string());
        let phase = self.effective_phase();

        // 1. Concurrency permit (the run-wide cap). Held until this leaf finishes.
        let _permit = self
            .ctx
            .semaphore()
            .acquire_owned()
            .await
            .map_err(|_| Error::Agent("flow concurrency limiter closed".to_string()))?;

        // 2. Runaway backstop (monotonic; a rejected admission still counts).
        self.ctx.register_agent()?;

        // 3. Budget admission (hard ceiling). On exhaustion this records a
        //    control breach and fires run-wide cancellation before returning.
        self.ctx.admit_budget()?;

        self.ctx.emit(WorkflowEvent::AgentStarted {
            label: label.clone(),
            phase,
        });

        // 4. Race the agent run against cooperative cancellation.
        let output = tokio::select! {
            biased;
            _ = self.ctx.cancel_token().cancelled() => {
                self.ctx.emit(WorkflowEvent::AgentSkipped { label });
                return Ok(None);
            }
            result = run_one(&self.ctx, &self.prompt, &self.opts) => result?,
        };

        // 5. Record the completed cost against the shared budget, then finish.
        self.ctx.record_spend(&output.tokens_used);
        self.ctx.emit(WorkflowEvent::AgentFinished {
            label,
            usage: output.tokens_used,
        });
        Ok(Some(output.result))
    }
}

/// Build and execute a fresh per-call [`AgentRunner`] over the shared provider.
///
/// Seam: later phases attach the schema, per-call model, budget recording, and
/// journal hooks here without changing the leaf sequence.
async fn run_one(ctx: &WorkflowCtx, prompt: &str, opts: &AgentOpts) -> Result<AgentOutput, Error> {
    let mut builder = AgentRunner::builder(ctx.provider());
    if let Some(label) = &opts.label {
        builder = builder.name(label.clone());
    }
    let runner = builder.build()?;
    runner.execute(prompt).await
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::agent::test_helpers::MockProvider;
    use crate::error::Error;
    use crate::llm::types::TokenUsage;
    use crate::llm::types::{CompletionRequest, CompletionResponse, ContentBlock, StopReason};
    use crate::llm::{BoxedProvider, LlmProvider};

    use super::super::parallel::{BoxThunk, parallel, thunk};

    /// Provider that tracks peak concurrency, sleeping to force overlap.
    struct ConcurrencyTrackingProvider {
        current: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }

    impl LlmProvider for ConcurrencyTrackingProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            let now = self.current.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(50)).await;
            self.current.fetch_sub(1, Ordering::SeqCst);
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    ..Default::default()
                },
                model: None,
            })
        }
        fn model_name(&self) -> Option<&str> {
            Some("concurrency-mock")
        }
    }

    #[tokio::test]
    async fn text_terminal_returns_agent_output() {
        let mock = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "mock text",
            10,
            5,
        )]));
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::from_arc(Arc::clone(&mock))))
            .build()
            .expect("build ctx");

        let out = agent(&ctx, "do the thing").run().await.expect("run ok");
        assert_eq!(out.as_deref(), Some("mock text"));
        // The provider was actually invoked exactly once.
        assert_eq!(mock.captured_requests.lock().expect("lock").len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrency_cap_is_respected_and_used() {
        let current = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let provider = ConcurrencyTrackingProvider {
            current: Arc::clone(&current),
            peak: Arc::clone(&peak),
        };
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::new(provider)))
            .max_concurrency(2)
            .max_agents(1000)
            .build()
            .expect("build ctx");

        // 10 agents through parallel(); the leaf permit caps in-flight at 2.
        let thunks: Vec<BoxThunk<String>> = (0..10)
            .map(|i| {
                let ctx = ctx.clone();
                thunk(move || async move {
                    Ok(agent(&ctx, format!("task {i}"))
                        .run()
                        .await?
                        .unwrap_or_default())
                })
            })
            .collect();
        let out = parallel(&ctx, thunks).await;

        assert_eq!(out.iter().filter(|o| o.is_some()).count(), 10);
        let observed = peak.load(Ordering::SeqCst);
        assert!(observed <= 2, "peak {observed} exceeded cap 2");
        assert_eq!(
            observed, 2,
            "cap should actually be reached, peak was {observed}"
        );
    }

    #[tokio::test]
    async fn backstop_rejects_excess_agents() {
        let mock = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("a", 1, 1),
            MockProvider::text_response("b", 1, 1),
        ]));
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::from_arc(Arc::clone(&mock))))
            .max_agents(2)
            .build()
            .expect("build ctx");

        assert!(agent(&ctx, "1").run().await.is_ok());
        assert!(agent(&ctx, "2").run().await.is_ok());
        let third = agent(&ctx, "3").run().await;
        match third {
            Err(Error::AgentBudgetExceeded { limit }) => assert_eq!(limit, 2),
            other => panic!("expected AgentBudgetExceeded {{ limit: 2 }}, got {other:?}"),
        }
        // The rejected third agent never reached the provider.
        assert_eq!(mock.captured_requests.lock().expect("lock").len(), 2);
    }

    #[tokio::test]
    async fn cancelled_run_skips_without_invoking_provider() {
        let mock = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "should not run",
            1,
            1,
        )]));
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::from_arc(Arc::clone(&mock))))
            .build()
            .expect("build ctx");
        ctx.cancellation_token().cancel(); // pre-cancel

        let out = agent(&ctx, "task").run().await.expect("run ok");
        assert!(out.is_none(), "cancelled run must yield Ok(None)");
        assert!(
            mock.captured_requests.lock().expect("lock").is_empty(),
            "provider must not be invoked when cancelled"
        );
    }

    // -----------------------------------------------------------------------
    // P2: shared budget wired through the leaf
    // -----------------------------------------------------------------------

    use super::super::ctx::ControlBreach;
    // `BoxThunk` is already imported at the top of this `mod tests`; re-import
    // only the aliased fns to avoid an E0252 duplicate.
    use super::super::parallel::{parallel as flow_parallel, thunk as flow_thunk};

    #[tokio::test]
    async fn leaf_records_weighted_cost_into_budget() {
        // input 10 + output 5 -> weighted cost 15.
        let mock = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "x", 10, 5,
        )]));
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::from_arc(mock)))
            .build()
            .expect("build ctx");

        assert_eq!(ctx.budget().spent(), 0);
        agent(&ctx, "task").run().await.expect("run ok");
        assert_eq!(ctx.budget().spent(), 15);
    }

    #[tokio::test]
    async fn sequential_budget_ceiling_rejects_after_exhaustion() {
        // Each agent costs 10 (input 10, output 0); ceiling 25.
        let mock = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("a", 10, 0),
            MockProvider::text_response("b", 10, 0),
            MockProvider::text_response("c", 10, 0),
        ]));
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::from_arc(Arc::clone(&mock))))
            .budget(25)
            .build()
            .expect("build ctx");

        assert!(agent(&ctx, "1").run().await.is_ok()); // spent 0->10
        assert!(agent(&ctx, "2").run().await.is_ok()); // spent 10->20
        assert!(agent(&ctx, "3").run().await.is_ok()); // spent 20->30 (admitted at 20<25)
        let fourth = agent(&ctx, "4").run().await; // admit sees 30>=25
        match fourth {
            Err(Error::BudgetExceeded { used, limit }) => {
                assert_eq!(used, 30);
                assert_eq!(limit, 25);
            }
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
        // The rejected fourth agent never reached the provider.
        assert_eq!(mock.captured_requests.lock().expect("lock").len(), 3);
    }

    #[tokio::test]
    async fn unbounded_budget_leaves_runs_unthrottled() {
        let mock = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 1_000, 1_000,
        )]));
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::from_arc(mock)))
            .build() // default: unbounded
            .expect("build ctx");
        assert!(agent(&ctx, "task").run().await.is_ok());
        assert_eq!(ctx.budget().spent(), 2_000);
        assert_eq!(ctx.remaining(), u64::MAX);
        assert!(!ctx.is_cancelled());
        assert!(ctx.control_breach().is_none());
    }

    // -----------------------------------------------------------------------
    // P2: sticky control-error semantics through a combinator
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn budget_breach_in_parallel_is_sticky_and_trips_cancellation() {
        // Pre-exhaust the budget so every agent's admission fails.
        let mock = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("x", 1, 1),
            MockProvider::text_response("y", 1, 1),
        ]));
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::from_arc(Arc::clone(&mock))))
            .budget(1)
            .build()
            .expect("build ctx");
        ctx.budget().record(&crate::llm::types::TokenUsage {
            input_tokens: 5,
            ..Default::default()
        }); // spent 5 >= 1: pool exhausted

        let thunks: Vec<BoxThunk<String>> = (0..2)
            .map(|i| {
                let ctx = ctx.clone();
                flow_thunk(move || async move {
                    Ok(agent(&ctx, format!("p{i}"))
                        .run()
                        .await?
                        .unwrap_or_default())
                })
            })
            .collect();
        let out = flow_parallel(&ctx, thunks).await;

        // Combinator collapsed the control errors to None ...
        assert!(out.iter().all(Option::is_none));
        // ... but the breach is sticky: cancellation fired and is recorded.
        assert!(
            ctx.is_cancelled(),
            "a control breach must fire run-wide cancel"
        );
        assert!(matches!(
            ctx.control_breach(),
            Some(ControlBreach::Budget { limit: 1, .. })
        ));
        // No agent ran (admission failed before the provider).
        assert!(mock.captured_requests.lock().expect("lock").is_empty());
    }

    #[tokio::test]
    async fn agent_domain_error_in_parallel_is_not_sticky() {
        // Empty provider -> each agent errors with an agent-domain error.
        let mock = Arc::new(MockProvider::new(vec![]));
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::from_arc(mock)))
            .build()
            .expect("build ctx");

        let thunks: Vec<BoxThunk<String>> = (0..2)
            .map(|i| {
                let ctx = ctx.clone();
                flow_thunk(move || async move {
                    Ok(agent(&ctx, format!("p{i}"))
                        .run()
                        .await?
                        .unwrap_or_default())
                })
            })
            .collect();
        let out = flow_parallel(&ctx, thunks).await;

        assert!(out.iter().all(Option::is_none));
        // Agent-domain errors collapse to None WITHOUT tripping the run.
        assert!(!ctx.is_cancelled(), "a domain error must NOT fire cancel");
        assert!(ctx.control_breach().is_none());
    }

    // -----------------------------------------------------------------------
    // P2: accounting under concurrency + loop-until-budget
    // -----------------------------------------------------------------------

    /// Provider that returns a fixed usage on *every* call (never exhausts), so
    /// many concurrent agents don't run out of canned responses.
    struct FixedCostProvider {
        input: u32,
        output: u32,
    }

    impl LlmProvider for FixedCostProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text { text: "ok".into() }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: self.input,
                    output_tokens: self.output,
                    ..Default::default()
                },
                model: None,
            })
        }
        fn model_name(&self) -> Option<&str> {
            Some("fixed-cost-mock")
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_spend_accounting_loses_nothing() {
        // 64 agents, each costing 2 (input 1 + output 1), run concurrently under
        // the cap. The atomic accumulator must total exactly 64 × 2 = 128.
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::new(FixedCostProvider {
            input: 1,
            output: 1,
        })))
        .max_concurrency(8)
        .build()
        .expect("build ctx");

        let thunks: Vec<BoxThunk<String>> = (0..64)
            .map(|i| {
                let ctx = ctx.clone();
                thunk(move || async move {
                    Ok(agent(&ctx, format!("t{i}"))
                        .run()
                        .await?
                        .unwrap_or_default())
                })
            })
            .collect();
        let out = parallel(&ctx, thunks).await;

        assert_eq!(out.iter().filter(|o| o.is_some()).count(), 64);
        assert_eq!(ctx.budget().spent(), 128, "atomic accounting lost a record");
    }

    #[tokio::test]
    async fn loop_until_budget_terminates() {
        // Budget 100; each agent costs 10 (input 10). The loop guard
        // `remaining() >= 10` admits exactly 10 agents, then stops at remaining 0.
        let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::new(FixedCostProvider {
            input: 10,
            output: 0,
        })))
        .budget(100)
        .build()
        .expect("build ctx");

        let mut ran = 0u32;
        while ctx.remaining() >= 10 {
            agent(&ctx, "work").run().await.expect("run ok");
            ran += 1;
            assert!(ran <= 100, "loop-until-budget failed to terminate");
        }
        assert_eq!(ran, 10);
        assert_eq!(ctx.budget().spent(), 100);
        assert_eq!(ctx.remaining(), 0);
    }
}
