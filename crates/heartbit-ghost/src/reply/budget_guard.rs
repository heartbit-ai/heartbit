//! Daily-budget guard — short-circuits mention drafting when the
//! persona's daily LLM token usage has reached the configured cap.
//! Default budget is `None` (unlimited); operator must opt in.
//!
//! See P1.7 spec §7.

use super::budget::{BudgetError, DailyTokenBudget};
use super::spam_guard::SkipReason;

/// Daily-budget guard.
pub struct DailyBudgetGuard {
    /// `None` means unlimited (always returns `Ok(None)`).
    budget: Option<u64>,
}

impl DailyBudgetGuard {
    /// Construct from a configured budget. `None` disables the guard.
    pub fn new(budget: Option<u64>) -> Self {
        Self { budget }
    }

    /// Returns `Some(SkipReason::DailyBudgetExhausted)` when the
    /// persona's usage today is at or above the configured budget.
    /// `None` budget always proceeds (unlimited).
    pub async fn should_skip(
        &self,
        persona: &str,
        tracker: &dyn DailyTokenBudget,
    ) -> Result<Option<SkipReason>, BudgetError> {
        let Some(budget) = self.budget else {
            return Ok(None);
        };
        let used = tracker.usage_today(persona).await?;
        if used >= budget {
            Ok(Some(SkipReason::DailyBudgetExhausted { used, budget }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reply::InMemoryDailyBudget;

    #[tokio::test]
    async fn proceeds_when_below_budget() {
        let tracker = InMemoryDailyBudget::new();
        tracker.record_usage("p", 100).await.unwrap();
        let guard = DailyBudgetGuard::new(Some(500));
        assert!(guard.should_skip("p", &tracker).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn skips_when_at_or_above_budget() {
        let tracker = InMemoryDailyBudget::new();
        tracker.record_usage("p", 500).await.unwrap();
        let guard = DailyBudgetGuard::new(Some(500));
        let result = guard.should_skip("p", &tracker).await.unwrap();
        match result {
            Some(SkipReason::DailyBudgetExhausted { used, budget }) => {
                assert_eq!(used, 500);
                assert_eq!(budget, 500);
            }
            other => panic!("expected DailyBudgetExhausted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn none_budget_always_proceeds() {
        let tracker = InMemoryDailyBudget::new();
        // Record way more than any budget could allow:
        for _ in 0..100 {
            tracker.record_usage("p", 100_000).await.unwrap();
        }
        let guard = DailyBudgetGuard::new(None);
        assert!(guard.should_skip("p", &tracker).await.unwrap().is_none());
    }
}
