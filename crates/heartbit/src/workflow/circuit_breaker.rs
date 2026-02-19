use restate_sdk::prelude::*;
use serde::{Deserialize, Serialize};

/// Circuit breaker states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Restate virtual object for per-provider circuit breaking.
///
/// The object key is the provider name (e.g., "anthropic", "openrouter").
/// State (open/closed/half-open) is persisted by Restate and shared across
/// all workers.
///
/// **Recovery**: Once the circuit opens (5 consecutive failures), it stays open
/// until `try_half_open()` or `reset()` is called externally. There is no
/// automatic timeout-based recovery — operators must probe readiness via the
/// Restate HTTP API (e.g., `POST /CircuitBreakerObject/<provider>/try_half_open`).
#[restate_sdk::object]
pub trait CircuitBreakerObject {
    /// Record a successful call. May transition half-open → closed.
    async fn record_success() -> Result<(), HandlerError>;

    /// Record a failed call. May transition closed → open.
    async fn record_failure() -> Result<(), HandlerError>;

    /// Check if the circuit is open (calls should be rejected).
    #[shared]
    async fn is_open() -> Result<bool, HandlerError>;

    /// Get the current circuit state.
    #[shared]
    async fn get_state() -> Result<Json<CircuitState>, HandlerError>;

    /// Attempt to transition from open to half-open (for retry probing).
    async fn try_half_open() -> Result<(), HandlerError>;

    /// Reset circuit to closed (manual recovery).
    async fn reset() -> Result<(), HandlerError>;
}

pub struct CircuitBreakerObjectImpl;

impl CircuitBreakerObject for CircuitBreakerObjectImpl {
    async fn record_success(&self, ctx: ObjectContext<'_>) -> Result<(), HandlerError> {
        let state = read_state(&ctx).await?;

        match state {
            CircuitState::HalfOpen => {
                let successes = ctx.get::<u32>("successes").await?.unwrap_or(0) + 1;
                let threshold = ctx.get::<u32>("success_threshold").await?.unwrap_or(2);

                if successes >= threshold {
                    // Transition to closed
                    ctx.set("state", "closed".to_string());
                    ctx.set("failures", 0u32);
                    ctx.set("successes", 0u32);
                } else {
                    ctx.set("successes", successes);
                }
            }
            CircuitState::Closed => {
                // Counter-reset semantics: any success resets the consecutive
                // failure count. This means intermittent failures (e.g., 4 fail,
                // 1 success, 4 fail) never trip the breaker. This is intentional
                // for LLM providers where transient errors are common.
                ctx.set("failures", 0u32);
            }
            CircuitState::Open => {
                // Shouldn't happen — calls should be blocked when open
            }
        }

        Ok(())
    }

    async fn record_failure(&self, ctx: ObjectContext<'_>) -> Result<(), HandlerError> {
        let state = read_state(&ctx).await?;

        match state {
            CircuitState::Closed => {
                let failures = ctx.get::<u32>("failures").await?.unwrap_or(0) + 1;
                let threshold = ctx.get::<u32>("failure_threshold").await?.unwrap_or(5);

                if failures >= threshold {
                    ctx.set("state", "open".to_string());
                    ctx.set("failures", failures);
                } else {
                    ctx.set("failures", failures);
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open goes back to open
                let threshold = ctx.get::<u32>("failure_threshold").await?.unwrap_or(5);
                ctx.set("state", "open".to_string());
                ctx.set("failures", threshold);
                ctx.set("successes", 0u32);
            }
            CircuitState::Open => {
                // Already open, nothing to do
            }
        }

        Ok(())
    }

    async fn is_open(&self, ctx: SharedObjectContext<'_>) -> Result<bool, HandlerError> {
        let state = read_shared_state(&ctx).await?;
        Ok(state == CircuitState::Open)
    }

    async fn get_state(
        &self,
        ctx: SharedObjectContext<'_>,
    ) -> Result<Json<CircuitState>, HandlerError> {
        let state = read_shared_state(&ctx).await?;
        Ok(Json(state))
    }

    async fn try_half_open(&self, ctx: ObjectContext<'_>) -> Result<(), HandlerError> {
        let state = read_state(&ctx).await?;
        if state == CircuitState::Open {
            ctx.set("state", "half_open".to_string());
            ctx.set("successes", 0u32);
        }
        Ok(())
    }

    async fn reset(&self, ctx: ObjectContext<'_>) -> Result<(), HandlerError> {
        ctx.set("state", "closed".to_string());
        ctx.set("failures", 0u32);
        ctx.set("successes", 0u32);
        Ok(())
    }
}

async fn read_state(ctx: &ObjectContext<'_>) -> Result<CircuitState, HandlerError> {
    let state_str = ctx
        .get::<String>("state")
        .await?
        .unwrap_or_else(|| "closed".into());
    Ok(parse_state(&state_str))
}

async fn read_shared_state(ctx: &SharedObjectContext<'_>) -> Result<CircuitState, HandlerError> {
    let state_str = ctx
        .get::<String>("state")
        .await?
        .unwrap_or_else(|| "closed".into());
    Ok(parse_state(&state_str))
}

fn parse_state(s: &str) -> CircuitState {
    match s {
        "open" => CircuitState::Open,
        "half_open" => CircuitState::HalfOpen,
        _ => CircuitState::Closed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_state_roundtrips() {
        let state = CircuitState::Open;
        let json = serde_json::to_string(&state).unwrap();
        assert_eq!(json, "\"open\"");
        let parsed: CircuitState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, CircuitState::Open);
    }

    #[test]
    fn parse_state_variants() {
        assert_eq!(parse_state("closed"), CircuitState::Closed);
        assert_eq!(parse_state("open"), CircuitState::Open);
        assert_eq!(parse_state("half_open"), CircuitState::HalfOpen);
        assert_eq!(parse_state("unknown"), CircuitState::Closed);
    }
}
