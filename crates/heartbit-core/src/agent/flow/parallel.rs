//! [`parallel`] — a BARRIER fan-out that is fail-SOFT and submission-ordered.
//!
//! This is the deliberate dual of [`ParallelAgent`](crate::ParallelAgent):
//! - `ParallelAgent` is fail-FAST (first `Err` aborts via `?`) and merges
//!   results sorted by agent name.
//! - `parallel` is fail-SOFT: a thunk that returns `Err` *or panics* yields
//!   `None` in its slot, every other thunk still runs, and the call itself
//!   never rejects. Results stay in submission order.
//!
//! Filter the `None`s (`results.into_iter().flatten()`) before use.
//!
//! # Homogeneous vs heterogeneous thunks
//!
//! The common case — fanning out over a work-list — produces one closure type
//! via `.map()` and needs no boxing:
//!
//! ```ignore
//! let outs = parallel(&ctx, (0..n).map(|i| move || async move { work(i).await }).collect()).await;
//! ```
//!
//! Distinct thunks have distinct types and cannot share a `Vec`. Wrap each in
//! [`thunk`] to erase the type:
//!
//! ```ignore
//! let outs = parallel(&ctx, vec![
//!     thunk({ let c = ctx.clone(); move || async move { agent(&c, "review auth").run().await } }),
//!     thunk({ let c = ctx.clone(); move || async move { agent(&c, "review perf").run().await } }),
//! ]).await;
//! ```

use std::future::Future;
use std::pin::Pin;

use tokio::task::JoinSet;

use crate::error::Error;

use super::ctx::WorkflowCtx;

/// A type-erased `parallel`/scatter thunk: a boxed nullary closure returning a
/// boxed future. Use [`thunk`] to build one. Lets heterogeneous thunks share a
/// single `Vec` (bare closures each have a distinct, unnameable type).
pub type BoxThunk<R> = Box<
    dyn FnOnce() -> Pin<Box<dyn Future<Output = Result<R, Error>> + Send + 'static>>
        + Send
        + 'static,
>;

/// Box a thunk into a [`BoxThunk`] so heterogeneous thunks can share a `Vec`.
pub fn thunk<R, F, Fut>(f: F) -> BoxThunk<R>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<R, Error>> + Send + 'static,
    R: Send + 'static,
{
    Box::new(move || {
        Box::pin(f()) as Pin<Box<dyn Future<Output = Result<R, Error>> + Send + 'static>>
    })
}

/// Run `thunks` concurrently and await them all (a barrier), returning one slot
/// per thunk in submission order. A thunk that errors or panics yields `None`.
///
/// `ctx` is threaded for API symmetry with the other combinators and so later
/// phases can add cancellation / event emission without a signature change; the
/// concurrency cap and budget are enforced at the [`agent`](super::agent::agent)
/// leaf inside each thunk, not here.
pub async fn parallel<R, F, Fut>(_ctx: &WorkflowCtx, thunks: Vec<F>) -> Vec<Option<R>>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<R, Error>> + Send + 'static,
    R: Send + 'static,
{
    let n = thunks.len();
    let mut set = JoinSet::new();
    for (i, thunk) in thunks.into_iter().enumerate() {
        set.spawn(async move { (i, thunk().await) });
    }

    let mut out: Vec<Option<R>> = (0..n).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        match joined {
            // Thunk completed and returned Ok: place at its submission index.
            Ok((i, Ok(value))) => out[i] = Some(value),
            // Thunk returned Err: leave the slot None (fail-soft).
            Ok((_i, Err(_e))) => {}
            // Thunk panicked or was aborted: leave the slot None (isolation).
            Err(_join_err) => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::agent::test_helpers::MockProvider;
    use crate::llm::BoxedProvider;

    fn ctx() -> WorkflowCtx {
        WorkflowCtx::builder(Arc::new(BoxedProvider::new(MockProvider::new(vec![]))))
            .build()
            .expect("build ctx")
    }

    #[tokio::test]
    async fn results_in_submission_order_despite_completion_order() {
        let ctx = ctx();
        // Thunk 0 sleeps longest, thunk 2 returns immediately: completion order
        // is the reverse of submission order, but results must match submission.
        let thunks: Vec<BoxThunk<i32>> = vec![
            thunk(|| async {
                tokio::time::sleep(Duration::from_millis(40)).await;
                Ok(10)
            }),
            thunk(|| async {
                tokio::time::sleep(Duration::from_millis(20)).await;
                Ok(20)
            }),
            thunk(|| async { Ok(30) }),
        ];
        let out = parallel(&ctx, thunks).await;
        assert_eq!(out, vec![Some(10), Some(20), Some(30)]);
    }

    #[tokio::test]
    async fn errored_thunk_yields_none_others_unaffected() {
        let ctx = ctx();
        let thunks: Vec<BoxThunk<i32>> = vec![
            thunk(|| async { Ok(1) }),
            thunk(|| async { Err(Error::Agent("boom".into())) }),
            thunk(|| async { Ok(3) }),
        ];
        let out = parallel(&ctx, thunks).await;
        // Never rejects; the failed slot is None, siblings are Some.
        assert_eq!(out, vec![Some(1), None, Some(3)]);
    }

    // Multi-thread runtime so the panicking task runs on a tokio worker, not the
    // libtest thread; `JoinSet` catches it into `Err(JoinError)` (our
    // `Err(_join_err)` arm -> `None`). The deliberate panic still prints one line
    // to stderr — that is expected, not a failure.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn panicking_thunk_is_isolated() {
        let ctx = ctx();
        let thunks: Vec<BoxThunk<i32>> = vec![
            thunk(|| async { Ok(1) }),
            thunk(|| async { panic!("intentional test panic") }),
            thunk(|| async { Ok(3) }),
        ];
        let out = parallel(&ctx, thunks).await;
        // JoinSet isolates the panic: that slot is None, siblings complete.
        assert_eq!(out, vec![Some(1), None, Some(3)]);
    }

    #[tokio::test]
    async fn homogeneous_map_fanout_needs_no_boxing() {
        let ctx = ctx();
        // The common case: map over a work-list, one closure type, no thunk().
        let work = vec![1, 2, 3, 4];
        let thunks: Vec<_> = work
            .into_iter()
            .map(|x| move || async move { Ok::<i32, Error>(x * 100) })
            .collect();
        let out = parallel(&ctx, thunks).await;
        assert_eq!(out, vec![Some(100), Some(200), Some(300), Some(400)]);
    }

    #[tokio::test]
    async fn empty_thunks_returns_empty() {
        let ctx = ctx();
        let thunks: Vec<BoxThunk<i32>> = Vec::new();
        let out = parallel(&ctx, thunks).await;
        assert!(out.is_empty());
    }
}
