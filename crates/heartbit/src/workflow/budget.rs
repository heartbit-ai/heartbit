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
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

impl TokenUsageRecord {
    /// Total tokens processed (all categories).
    ///
    /// This counts raw token throughput, not cost-equivalent tokens.
    /// Cache reads are counted at face value even though they cost less
    /// (0.1x input price on Anthropic). For cost-based budgeting, use
    /// the individual fields with appropriate weights.
    pub fn total(&self) -> u64 {
        self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
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
        let current_cache_create = ctx
            .get::<u64>("cache_creation_input_tokens")
            .await?
            .unwrap_or(0);
        let current_cache_read = ctx
            .get::<u64>("cache_read_input_tokens")
            .await?
            .unwrap_or(0);
        let limit = ctx.get::<u64>("limit").await?.unwrap_or(u64::MAX);

        let new_in = current_in.saturating_add(usage.input_tokens);
        let new_out = current_out.saturating_add(usage.output_tokens);
        let new_cache_create =
            current_cache_create.saturating_add(usage.cache_creation_input_tokens);
        let new_cache_read = current_cache_read.saturating_add(usage.cache_read_input_tokens);
        let new_total = new_in
            .saturating_add(new_out)
            .saturating_add(new_cache_create)
            .saturating_add(new_cache_read);

        // Always persist the updated totals so get_usage() reports
        // accurate numbers even when the budget is exceeded.
        ctx.set("input_tokens", new_in);
        ctx.set("output_tokens", new_out);
        ctx.set("cache_creation_input_tokens", new_cache_create);
        ctx.set("cache_read_input_tokens", new_cache_read);

        if new_total > limit {
            return Err(TerminalError::new(format!(
                "Token budget exceeded: {new_total} > {limit} for {}",
                ctx.key()
            ))
            .into());
        }
        Ok(())
    }

    async fn get_usage(
        &self,
        ctx: SharedObjectContext<'_>,
    ) -> Result<Json<TokenUsageRecord>, HandlerError> {
        let input_tokens = ctx.get::<u64>("input_tokens").await?.unwrap_or(0);
        let output_tokens = ctx.get::<u64>("output_tokens").await?.unwrap_or(0);
        let cache_creation_input_tokens = ctx
            .get::<u64>("cache_creation_input_tokens")
            .await?
            .unwrap_or(0);
        let cache_read_input_tokens = ctx
            .get::<u64>("cache_read_input_tokens")
            .await?
            .unwrap_or(0);
        Ok(Json(TokenUsageRecord {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
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
            ..Default::default()
        };
        assert_eq!(record.total(), 150);
    }

    #[test]
    fn token_usage_record_total_includes_cache() {
        let record = TokenUsageRecord {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 80,
            cache_read_input_tokens: 60,
        };
        assert_eq!(record.total(), 290);
    }

    #[test]
    fn token_usage_record_roundtrips() {
        let record = TokenUsageRecord {
            input_tokens: 1000,
            output_tokens: 500,
            cache_creation_input_tokens: 800,
            cache_read_input_tokens: 600,
        };
        let json = serde_json::to_string(&record).unwrap();
        let parsed: TokenUsageRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.input_tokens, 1000);
        assert_eq!(parsed.output_tokens, 500);
        assert_eq!(parsed.cache_creation_input_tokens, 800);
        assert_eq!(parsed.cache_read_input_tokens, 600);
    }

    #[test]
    fn token_usage_record_backward_compat() {
        // Old JSON without cache fields should deserialize with defaults
        let json = r#"{"input_tokens":100,"output_tokens":50}"#;
        let parsed: TokenUsageRecord = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.input_tokens, 100);
        assert_eq!(parsed.output_tokens, 50);
        assert_eq!(parsed.cache_creation_input_tokens, 0);
        assert_eq!(parsed.cache_read_input_tokens, 0);
    }
}
