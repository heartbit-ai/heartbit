use restate_sdk::prelude::*;
use serde::{Deserialize, Serialize};

/// Restate virtual object for tracking token budgets.
///
/// The object key is the task or agent ID. Tracks cumulative token usage
/// and enforces limits. Returns a `TerminalError` when budget is exceeded
/// (non-retryable — the workflow should abort).
#[restate_sdk::object]
pub trait TokenBudgetObject {
    /// Record token usage. Returns error if budget exceeded.
    async fn record_usage(usage: Json<TokenUsageRecord>) -> Result<(), HandlerError>;

    /// Get current usage for this budget.
    #[shared]
    async fn get_usage() -> Result<Json<TokenUsageRecord>, HandlerError>;

    /// Set the budget limit.
    async fn set_limit(limit: u64) -> Result<(), HandlerError>;
}

/// Token usage record.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsageRecord {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl TokenUsageRecord {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

pub struct TokenBudgetObjectImpl;

impl TokenBudgetObject for TokenBudgetObjectImpl {
    async fn record_usage(
        &self,
        ctx: ObjectContext<'_>,
        Json(usage): Json<TokenUsageRecord>,
    ) -> Result<(), HandlerError> {
        let current_in = ctx.get::<u64>("input_tokens").await?.unwrap_or(0);
        let current_out = ctx.get::<u64>("output_tokens").await?.unwrap_or(0);
        let limit = ctx.get::<u64>("limit").await?.unwrap_or(u64::MAX);

        let new_in = current_in.saturating_add(usage.input_tokens);
        let new_out = current_out.saturating_add(usage.output_tokens);
        let new_total = new_in.saturating_add(new_out);

        if new_total > limit {
            return Err(TerminalError::new(format!(
                "Token budget exceeded: {new_total} > {limit} for {}",
                ctx.key()
            ))
            .into());
        }

        ctx.set("input_tokens", new_in);
        ctx.set("output_tokens", new_out);
        Ok(())
    }

    async fn get_usage(
        &self,
        ctx: SharedObjectContext<'_>,
    ) -> Result<Json<TokenUsageRecord>, HandlerError> {
        let input_tokens = ctx.get::<u64>("input_tokens").await?.unwrap_or(0);
        let output_tokens = ctx.get::<u64>("output_tokens").await?.unwrap_or(0);
        Ok(Json(TokenUsageRecord {
            input_tokens,
            output_tokens,
        }))
    }

    async fn set_limit(&self, ctx: ObjectContext<'_>, limit: u64) -> Result<(), HandlerError> {
        ctx.set("limit", limit);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_record_total() {
        let record = TokenUsageRecord {
            input_tokens: 100,
            output_tokens: 50,
        };
        assert_eq!(record.total(), 150);
    }

    #[test]
    fn token_usage_record_roundtrips() {
        let record = TokenUsageRecord {
            input_tokens: 1000,
            output_tokens: 500,
        };
        let json = serde_json::to_string(&record).unwrap();
        let parsed: TokenUsageRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.input_tokens, 1000);
        assert_eq!(parsed.output_tokens, 500);
    }
}
