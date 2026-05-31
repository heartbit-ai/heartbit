//! [`pipeline`] — NO-BARRIER per-item streaming through a chain of stages.
//!
//! This is the deliberate dual of [`SequentialAgent`](crate::SequentialAgent),
//! which advances *all* work stage-by-stage (a barrier: wall-clock =
//! sum-of-slowest-per-stage). `pipeline` runs one future per item that folds
//! the whole stage chain **sequentially inside itself**, with every item's
//! chain driven **concurrently**. So item A can be in stage 3 while item B is
//! still in stage 1, and wall-clock = the slowest single-item chain — not the
//! sum of per-stage maxima.
//!
//! A stage that returns `Err` drops that item to `None` and skips the item's
//! remaining stages; sibling items are unaffected. Each stage receives
//! `(prev_result, original_item, index)`. The flow value is `serde_json::Value`
//! (typed flow is a later phase).

use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;
use tokio::task::JoinSet;

use crate::error::Error;

use super::ctx::WorkflowCtx;

/// A single pipeline stage: `(prev_result, original_item, index) -> Result<Value>`.
type Stage<I> = Arc<
    dyn Fn(
            Value,
            Arc<I>,
            usize,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<Value, Error>> + Send>>
        + Send
        + Sync,
>;

/// Builder for a no-barrier [`pipeline`]. Add stages with [`stage`](Self::stage),
/// then drive every item's chain concurrently with [`run`](Self::run).
pub struct PipelineBuilder<I> {
    #[allow(dead_code)] // ctx is threaded for symmetry / future cancellation+events
    ctx: WorkflowCtx,
    items: Vec<I>,
    stages: Vec<Stage<I>>,
}

/// Begin a no-barrier pipeline over `items`.
pub fn pipeline<I: Send + Sync + 'static>(ctx: &WorkflowCtx, items: Vec<I>) -> PipelineBuilder<I> {
    PipelineBuilder {
        ctx: ctx.clone(),
        items,
        stages: Vec::new(),
    }
}

impl<I: Send + Sync + 'static> PipelineBuilder<I> {
    /// Append a stage. The stage closure gets `(prev_result, item, index)` and
    /// returns the next flow value (or `Err` to drop this item).
    pub fn stage<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Value, Arc<I>, usize) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Value, Error>> + Send + 'static,
    {
        self.stages
            .push(Arc::new(move |v, item, idx| Box::pin(f(v, item, idx))));
        self
    }

    /// Run every item's stage chain concurrently (NO inter-item barrier).
    /// Returns one slot per item in original order; a dropped item is `None`.
    ///
    /// Each item gets exactly one task that folds the *entire* stage chain
    /// sequentially inside itself; all item-tasks run on a single [`JoinSet`].
    /// So item A can be in stage 3 while item B is still in stage 1 — there is
    /// no synchronization point between stages across items. Wall-clock = the
    /// slowest single-item chain, not the sum of per-stage maxima.
    pub async fn run(self) -> Vec<Option<Value>> {
        let n = self.items.len();
        let stages = Arc::new(self.stages);
        let mut set = JoinSet::new();
        for (idx, item) in self.items.into_iter().enumerate() {
            let stages = Arc::clone(&stages);
            let item = Arc::new(item);
            set.spawn(async move {
                let mut acc = Value::Null;
                for st in stages.iter() {
                    match st(acc, Arc::clone(&item), idx).await {
                        Ok(next) => acc = next,
                        // Drop this item; skip its remaining stages (siblings unaffected).
                        Err(_e) => return (idx, None),
                    }
                }
                (idx, Some(acc))
            });
        }

        let mut out: Vec<Option<Value>> = (0..n).map(|_| None).collect();
        while let Some(joined) = set.join_next().await {
            // A panicking item-task leaves its slot None (isolation).
            if let Ok((idx, res)) = joined {
                out[idx] = res;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use tokio::sync::Notify;

    use super::*;
    use crate::agent::test_helpers::MockProvider;
    use crate::llm::BoxedProvider;

    fn ctx() -> WorkflowCtx {
        WorkflowCtx::builder(Arc::new(BoxedProvider::new(MockProvider::new(vec![]))))
            .build()
            .expect("build ctx")
    }

    /// THE headline property. Two items, two stages, with a deterministic
    /// cross-dependency: item 0's stage 0 blocks until item 1 reaches stage 1.
    /// A no-barrier pipeline completes (item 1 streams ahead into stage 1 while
    /// item 0 is still in stage 0). A barrier pipeline deadlocks — item 1 can't
    /// enter stage 1 until every item finishes stage 0, but item 0's stage 0 is
    /// waiting on item 1 — so it times out. Notify-based, not sleep-flaky.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_inter_item_barrier_allows_item_to_stream_ahead() {
        let ctx = ctx();
        let order = Arc::new(Mutex::new(Vec::<String>::new()));
        let gate = Arc::new(Notify::new()); // signalled when item 1 reaches stage 1

        let o0 = Arc::clone(&order);
        let g0 = Arc::clone(&gate);
        let stage0 = move |_acc: Value, _item: Arc<usize>, idx: usize| {
            let o = Arc::clone(&o0);
            let g = Arc::clone(&g0);
            async move {
                o.lock().expect("order").push(format!("s0-start-{idx}"));
                if idx == 0 {
                    // item 0 cannot finish stage 0 until item 1 reaches stage 1
                    g.notified().await;
                }
                o.lock().expect("order").push(format!("s0-end-{idx}"));
                Ok(Value::Null)
            }
        };

        let o1 = Arc::clone(&order);
        let g1 = Arc::clone(&gate);
        let stage1 = move |_acc: Value, _item: Arc<usize>, idx: usize| {
            let o = Arc::clone(&o1);
            let g = Arc::clone(&g1);
            async move {
                o.lock().expect("order").push(format!("s1-start-{idx}"));
                if idx == 1 {
                    g.notify_one(); // unblock item 0's stage 0
                }
                Ok(Value::Null)
            }
        };

        let fut = pipeline(&ctx, vec![0usize, 1usize])
            .stage(stage0)
            .stage(stage1)
            .run();
        let result = tokio::time::timeout(Duration::from_secs(5), fut).await;
        assert!(
            result.is_ok(),
            "pipeline deadlocked — an inter-item barrier was detected"
        );

        let log = order.lock().expect("order");
        let s1_1 = log
            .iter()
            .position(|e| e == "s1-start-1")
            .expect("item 1 should reach stage 1");
        let s0_end_0 = log
            .iter()
            .position(|e| e == "s0-end-0")
            .expect("item 0 should finish stage 0");
        assert!(
            s1_1 < s0_end_0,
            "item 1 must reach stage 1 before item 0 finishes stage 0; order={log:?}"
        );
    }

    #[tokio::test]
    async fn stage_error_drops_item_and_skips_its_remaining_stages() {
        let ctx = ctx();
        let stage1_runs = Arc::new(AtomicUsize::new(0));

        let stage0 = move |_acc: Value, _item: Arc<usize>, idx: usize| async move {
            if idx == 1 {
                Err(Error::Agent("item 1 fails stage 0".into()))
            } else {
                Ok(Value::from(idx as i64))
            }
        };
        let counter = Arc::clone(&stage1_runs);
        let stage1 = move |acc: Value, _item: Arc<usize>, _idx: usize| {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(acc)
            }
        };

        let out = pipeline(&ctx, vec![0usize, 1usize, 2usize])
            .stage(stage0)
            .stage(stage1)
            .run()
            .await;

        assert_eq!(out[0], Some(Value::from(0)));
        assert_eq!(out[1], None, "failed item must be None");
        assert_eq!(out[2], Some(Value::from(2)));
        // stage 1 ran only for items 0 and 2, never for the dropped item 1.
        assert_eq!(stage1_runs.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn results_placed_at_original_index() {
        let ctx = ctx();
        let stage0 = move |_acc: Value, _item: Arc<usize>, idx: usize| async move {
            Ok(Value::from((idx * 10) as i64))
        };
        let out = pipeline(&ctx, vec![0usize, 1usize, 2usize])
            .stage(stage0)
            .run()
            .await;
        assert_eq!(
            out,
            vec![
                Some(Value::from(0)),
                Some(Value::from(10)),
                Some(Value::from(20))
            ]
        );
    }

    #[tokio::test]
    async fn empty_items_returns_empty() {
        let ctx = ctx();
        let out = pipeline(&ctx, Vec::<usize>::new())
            .stage(move |_a: Value, _i: Arc<usize>, _idx: usize| async move { Ok(Value::Null) })
            .run()
            .await;
        assert!(out.is_empty());
    }

    /// Regression: `pipeline` must NOT acquire a per-item/per-stage concurrency
    /// permit. If it did, an `agent()` leaf inside a stage would deadlock at
    /// `max_concurrency == 1` — the item-task would hold the only permit while
    /// its own stage's leaf waited for one. Permits live ONLY at the leaf, so an
    /// item runs its chain (acquiring + releasing the leaf permit) and N>cap
    /// items still complete. Times out (fails) if a barrier/permit is introduced.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_permit_pipeline_with_agent_stages_does_not_deadlock() {
        let provider = Arc::new(BoxedProvider::new(MockProvider::new(
            (0..3)
                .map(|i| MockProvider::text_response(&format!("r{i}"), 1, 1))
                .collect(),
        )));
        let ctx = WorkflowCtx::builder(provider)
            .max_concurrency(1)
            .build()
            .expect("build ctx");

        let stage_ctx = ctx.clone();
        let fut = pipeline(&ctx, vec![0usize, 1usize, 2usize])
            .stage(move |_prev, item, _idx| {
                let ctx = stage_ctx.clone();
                async move {
                    let text = crate::agent::flow::agent(&ctx, format!("item {}", *item))
                        .run()
                        .await?
                        .unwrap_or_default();
                    Ok(Value::String(text))
                }
            })
            .run();

        let out = tokio::time::timeout(Duration::from_secs(5), fut)
            .await
            .expect("pipeline with agent() stages must not deadlock at concurrency=1");
        assert_eq!(
            out.iter().filter(|o| o.is_some()).count(),
            3,
            "all three items complete: {out:?}"
        );
    }
}
