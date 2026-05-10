//! Daily token budget tracker for the mentions pipeline.
//!
//! Tracks total LLM tokens spent per persona per UTC day. The
//! [`crate::reply::DailyBudgetGuard`] (P1.7 task 11) consults this
//! to short-circuit mention drafts when the daily cap is hit.
//!
//! See P1.7 spec §7.

use std::future::Future;
use std::pin::Pin;

use chrono::{NaiveDate, Utc};

/// Errors raised by [`DailyTokenBudget`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    /// I/O failure (file not readable, write failed, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSONL parse failure on replay.
    #[error("parse: {0}")]
    Parse(String),
}

/// Persistent storage for daily-budget accounting.
pub trait DailyTokenBudget: Send + Sync {
    /// Total tokens recorded for `persona` on the current UTC day.
    fn usage_today<'a>(
        &'a self,
        persona: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<u64, BudgetError>> + Send + 'a>>;

    /// Append `tokens` to today's accumulator for `persona`.
    fn record_usage<'a>(
        &'a self,
        persona: &'a str,
        tokens: u64,
    ) -> Pin<Box<dyn Future<Output = Result<(), BudgetError>> + Send + 'a>>;
}

/// Volatile in-memory budget tracker. For tests and dev runs.
pub struct InMemoryDailyBudget {
    pub(crate) inner: tokio::sync::RwLock<InMemoryInner>,
}

/// Storage backing [`InMemoryDailyBudget`].
#[derive(Default)]
pub(crate) struct InMemoryInner {
    /// (date, persona) → tokens used.
    pub(crate) usage: std::collections::HashMap<(NaiveDate, String), u64>,
}

impl Default for InMemoryDailyBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryDailyBudget {
    /// Construct an empty budget tracker.
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::RwLock::new(InMemoryInner::default()),
        }
    }
}

impl DailyTokenBudget for InMemoryDailyBudget {
    fn usage_today<'a>(
        &'a self,
        persona: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<u64, BudgetError>> + Send + 'a>> {
        Box::pin(async move {
            let today = Utc::now().date_naive();
            Ok(self
                .inner
                .read()
                .await
                .usage
                .get(&(today, persona.to_string()))
                .copied()
                .unwrap_or(0))
        })
    }

    fn record_usage<'a>(
        &'a self,
        persona: &'a str,
        tokens: u64,
    ) -> Pin<Box<dyn Future<Output = Result<(), BudgetError>> + Send + 'a>> {
        Box::pin(async move {
            let today = Utc::now().date_naive();
            *self
                .inner
                .write()
                .await
                .usage
                .entry((today, persona.to_string()))
                .or_insert(0) += tokens;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_record_then_read_round_trip() {
        let b = InMemoryDailyBudget::new();
        assert_eq!(b.usage_today("p").await.unwrap(), 0);
        b.record_usage("p", 100).await.unwrap();
        b.record_usage("p", 200).await.unwrap();
        assert_eq!(b.usage_today("p").await.unwrap(), 300);
    }

    #[tokio::test]
    async fn in_memory_per_persona_isolation() {
        let b = InMemoryDailyBudget::new();
        b.record_usage("a", 50).await.unwrap();
        b.record_usage("b", 100).await.unwrap();
        assert_eq!(b.usage_today("a").await.unwrap(), 50);
        assert_eq!(b.usage_today("b").await.unwrap(), 100);
        assert_eq!(b.usage_today("nonexistent").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn in_memory_zero_tokens_recorded_is_idempotent() {
        let b = InMemoryDailyBudget::new();
        b.record_usage("p", 0).await.unwrap();
        b.record_usage("p", 0).await.unwrap();
        assert_eq!(b.usage_today("p").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn in_memory_usage_today_isolates_by_date_in_inner() {
        // We can't easily mock Utc::now(), but we can verify that
        // entries keyed by yesterday's date don't show up in today's
        // count.
        let b = InMemoryDailyBudget::new();
        let yesterday = Utc::now().date_naive() - chrono::Duration::days(1);
        b.inner
            .write()
            .await
            .usage
            .insert((yesterday, "p".to_string()), 999);
        assert_eq!(
            b.usage_today("p").await.unwrap(),
            0,
            "yesterday must not leak"
        );
    }
}
