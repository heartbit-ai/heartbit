//! [`Budget`] — a shared, lock-free, hard-ceiling spend pool for a workflow run.
//!
//! One `Budget` is shared (by `Arc`) across every agent leaf in a run (and, in
//! later phases, across nested workflows). It is **record-only**: an agent is
//! admitted if the pool is not yet exhausted ([`check_admit`](Budget::check_admit)),
//! runs, then its weighted cost is recorded ([`record`](Budget::record)). Because
//! admission is checked *before* the cost is known, concurrent agents can
//! overshoot the ceiling by at most `concurrency × per-agent-cost` — the same
//! bound the Claude Code target accepts. The ceiling itself is hard: once spent
//! reaches the total, further admissions fail with [`Error::BudgetExceeded`].
//!
//! # Unit: cost-weighted token-equivalents (not micro-USD)
//!
//! `spent`/`total`/`remaining` are in **token-equivalents**: a single, model-
//! independent unit so the pool stays coherent across known and unknown models
//! (a mock model and a priced model accumulate the same unit). Cache traffic is
//! weighted to reflect real cost — cache *writes* ×1.25, cache *reads* ×0.1,
//! input/output/reasoning ×1 — matching Anthropic's cache pricing ratios. This
//! matches the existing per-runner `max_total_tokens` convention. (Budgeting in
//! true micro-USD would require pricing every model, including mocks; that is a
//! deliberate later mode, swappable by changing only [`weighted_cost`].)

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::Error;
use crate::llm::types::TokenUsage;

/// Shared hard-ceiling spend pool. Cheap to [`Clone`] (an `Arc` inside).
#[derive(Clone, Debug)]
pub struct Budget(Arc<BudgetInner>);

#[derive(Debug)]
struct BudgetInner {
    /// Ceiling in token-equivalents. `None` ⇒ unbounded.
    total: Option<NonZeroU64>,
    /// Cumulative weighted cost recorded so far.
    spent: AtomicU64,
}

impl Default for Budget {
    fn default() -> Self {
        Self::unbounded()
    }
}

impl Budget {
    /// An unbounded budget: every admission succeeds; `remaining()` is `u64::MAX`.
    pub fn unbounded() -> Self {
        Self(Arc::new(BudgetInner {
            total: None,
            spent: AtomicU64::new(0),
        }))
    }

    /// A budget with a hard ceiling of `total` token-equivalents. `0` ⇒ unbounded.
    pub fn with_total(total: u64) -> Self {
        Self(Arc::new(BudgetInner {
            total: NonZeroU64::new(total),
            spent: AtomicU64::new(0),
        }))
    }

    /// The configured ceiling, or `None` if unbounded.
    pub fn total(&self) -> Option<u64> {
        self.0.total.map(NonZeroU64::get)
    }

    /// Total weighted cost recorded so far.
    pub fn spent(&self) -> u64 {
        self.0.spent.load(Ordering::Relaxed)
    }

    /// Remaining headroom, or `u64::MAX` if unbounded.
    pub fn remaining(&self) -> u64 {
        match self.0.total {
            None => u64::MAX,
            Some(t) => t.get().saturating_sub(self.spent()),
        }
    }

    /// Admission check: `Ok` while there is headroom, `Err` once spent has
    /// reached (or passed) the ceiling. Boundary matches `runner.rs`
    /// (`used > max`) and the Restate budget object (`new_total > limit`):
    /// spending exactly to the limit is allowed; the *next* admission fails.
    pub(crate) fn check_admit(&self) -> Result<(), Error> {
        if let Some(total) = self.0.total {
            let spent = self.spent();
            if spent >= total.get() {
                return Err(Error::BudgetExceeded {
                    used: spent,
                    limit: total.get(),
                });
            }
        }
        Ok(())
    }

    /// Record one agent's weighted cost after it completes.
    pub(crate) fn record(&self, usage: &TokenUsage) {
        self.0
            .spent
            .fetch_add(weighted_cost(usage), Ordering::AcqRel);
    }
}

/// Cost-weighted token-equivalents for one usage record. Model-independent.
///
/// Each `u32` field is widened to `u64` *before* arithmetic so a high-volume
/// turn cannot overflow. Cache writes are weighted ×1.25 and cache reads ×0.1
/// to mirror Anthropic's cache pricing; input/output/reasoning are ×1.
fn weighted_cost(usage: &TokenUsage) -> u64 {
    let input = usage.input_tokens as u64;
    let output = usage.output_tokens as u64;
    let reasoning = usage.reasoning_tokens as u64;
    let cache_write = usage.cache_creation_input_tokens as u64;
    let cache_read = usage.cache_read_input_tokens as u64;
    input + output + reasoning + (cache_write * 5 / 4) + (cache_read / 10)
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
    fn unbounded_admits_and_reports_max_remaining() {
        let b = Budget::unbounded();
        assert_eq!(b.total(), None);
        assert_eq!(b.remaining(), u64::MAX);
        assert!(b.check_admit().is_ok());
        b.record(&usage(1_000, 1_000));
        // Still unbounded after recording.
        assert!(b.check_admit().is_ok());
        assert_eq!(b.remaining(), u64::MAX);
    }

    #[test]
    fn with_total_reports_total_and_remaining() {
        let b = Budget::with_total(100);
        assert_eq!(b.total(), Some(100));
        assert_eq!(b.spent(), 0);
        assert_eq!(b.remaining(), 100);
    }

    #[test]
    fn admit_ok_below_limit_err_at_or_above() {
        let b = Budget::with_total(100);
        assert!(b.check_admit().is_ok());
        b.record(&usage(60, 0)); // spent 60 < 100
        assert!(b.check_admit().is_ok());
        b.record(&usage(40, 0)); // spent 100 >= 100
        match b.check_admit() {
            Err(Error::BudgetExceeded { used, limit }) => {
                assert_eq!(used, 100);
                assert_eq!(limit, 100);
            }
            other => panic!("expected BudgetExceeded {{used:100, limit:100}}, got {other:?}"),
        }
    }

    #[test]
    fn record_accumulates_exactly() {
        let b = Budget::with_total(1_000_000);
        b.record(&usage(100, 50));
        b.record(&usage(200, 80));
        assert_eq!(b.spent(), 100 + 50 + 200 + 80);
        assert_eq!(b.remaining(), 1_000_000 - 430);
    }

    #[test]
    fn record_widens_u32_fields_without_overflow() {
        let b = Budget::unbounded();
        // Each field is u32::MAX; the sum exceeds u32::MAX and must not wrap.
        b.record(&usage(u32::MAX, u32::MAX));
        assert_eq!(b.spent(), u32::MAX as u64 + u32::MAX as u64);
    }

    #[test]
    fn weighted_cost_applies_cache_ratios() {
        // input/output/reasoning ×1
        assert_eq!(weighted_cost(&usage(100, 50)), 150);
        // cache read ×0.1
        let cache_read = TokenUsage {
            cache_read_input_tokens: 1_000,
            ..Default::default()
        };
        assert_eq!(weighted_cost(&cache_read), 100);
        // cache write ×1.25
        let cache_write = TokenUsage {
            cache_creation_input_tokens: 400,
            ..Default::default()
        };
        assert_eq!(weighted_cost(&cache_write), 500);
        // reasoning ×1
        let reasoning = TokenUsage {
            reasoning_tokens: 42,
            ..Default::default()
        };
        assert_eq!(weighted_cost(&reasoning), 42);
    }

    #[test]
    fn cache_read_heavy_usage_is_cheap() {
        // 10_000 cache-read tokens should weigh the same as 1_000 plain tokens.
        let heavy = TokenUsage {
            cache_read_input_tokens: 10_000,
            ..Default::default()
        };
        assert_eq!(weighted_cost(&heavy), 1_000);
    }
}
