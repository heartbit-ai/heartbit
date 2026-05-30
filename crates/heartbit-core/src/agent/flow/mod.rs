//! Dynamic-workflow combinator core (P1).
//!
//! A pure-Rust async combinator layer that reproduces the *semantics* of
//! Claude Code "dynamic workflows" — [`parallel`] (barrier, fail-soft),
//! [`pipeline`] (no-barrier per-item streaming), [`phase`]/[`log`]
//! observability, and a run-wide concurrency cap — on top of the existing
//! [`AgentRunner`](crate::AgentRunner). A "workflow" is ordinary developer-written
//! async Rust against these free functions; we do **not** embed a script
//! engine, IR, or macro DSL.
//!
//! These primitives are the deliberate *duals* of the compile-time workflow
//! agents in [`crate::agent::workflow`]: [`parallel`] is fail-SOFT where
//! [`ParallelAgent`](crate::ParallelAgent) is fail-FAST, and [`pipeline`] has
//! NO inter-item barrier where [`SequentialAgent`](crate::SequentialAgent)
//! advances stage-by-stage. Both layers coexist; nothing existing is changed.
//!
//! # Load-bearing invariant
//!
//! The `Err -> None` collapse lives **only** in [`parallel`]/[`pipeline`] at the
//! join/stage boundary, **never** inside [`agent`]. If `agent()` swallowed
//! errors, the concurrency backstop (and, in later phases, the token budget)
//! would stop propagating and silently break. Control errors must reach the
//! caller; only agent-domain failures collapse to `None`.

pub mod agent;
pub mod budget;
pub mod ctx;
pub mod event;
pub mod journal;
pub mod parallel;
pub mod pipeline;

// Re-export the free functions at the `flow` level so callers write
// `flow::agent(..)`, `flow::parallel(..)`, etc. A module and a same-named
// function coexist (type vs value namespace), so `flow::agent` is the module in
// path position and the function in call position.
pub use agent::agent;
pub use ctx::WorkflowCtx;
pub use event::{log, phase};
pub use parallel::{parallel, thunk};
pub use pipeline::pipeline;

#[cfg(test)]
mod smoke_tests {
    //! End-to-end smoke test exercising the whole P1 surface through the
    //! **public crate-root re-exports** (`crate::WorkflowCtx`, `crate::agent`,
    //! `crate::parallel`, `crate::pipeline`, `crate::phase`), proving the wiring
    //! in `lib.rs` resolves and the combinators compose.

    use std::sync::Arc;

    use serde_json::Value;

    use crate::agent::test_helpers::MockProvider;
    use crate::llm::BoxedProvider;

    #[tokio::test]
    async fn p1_surface_composes_through_public_paths() {
        // Mock provider with plenty of responses (single + 2 parallel + 2 pipeline).
        let responses: Vec<_> = (0..6)
            .map(|i| MockProvider::text_response(&format!("r{i}"), 5, 3))
            .collect();
        let provider = Arc::new(BoxedProvider::new(MockProvider::new(responses)));
        // Root-level type re-export.
        let ctx = crate::WorkflowCtx::builder(provider)
            .max_concurrency(4)
            .build()
            .expect("build ctx");

        // Free functions via the `crate::flow::` namespace.
        // phase() returns an RAII guard; agents issued under it adopt the phase.
        let _phase = crate::flow::phase(&ctx, "smoke");

        // Single agent leaf.
        let single = crate::flow::agent(&ctx, "solo")
            .label("solo")
            .run()
            .await
            .expect("single ok");
        assert!(single.is_some());

        // Fail-soft barrier fan-out over a homogeneous work-list (no boxing).
        let thunks: Vec<_> = (0..2)
            .map(|i| {
                let ctx = ctx.clone();
                move || async move { crate::flow::agent(&ctx, format!("p{i}")).run().await }
            })
            .collect();
        let fanned = crate::flow::parallel(&ctx, thunks).await;
        assert_eq!(fanned.iter().filter(|o| o.is_some()).count(), 2);

        // No-barrier pipeline whose stage calls an agent leaf.
        let piped = crate::flow::pipeline(&ctx, vec![10usize, 20usize])
            .stage(move |_prev, item, _idx| {
                let ctx = ctx.clone();
                async move {
                    let text = crate::flow::agent(&ctx, format!("item {item}"))
                        .run()
                        .await?
                        .unwrap_or_default();
                    Ok(Value::String(text))
                }
            })
            .run()
            .await;
        assert_eq!(piped.iter().filter(|o| o.is_some()).count(), 2);
    }
}
