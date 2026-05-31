//! Error type for all heartbit-core fallible operations.

use std::time::Duration;

use crate::types::TokenUsage;
use thiserror::Error;

/// Top-level error type for the heartbit-core crate.
///
/// All fallible public APIs return `Result<T, Error>`. Callers should match on
/// specific variants rather than converting to strings so that retry logic and
/// error reporting remain precise.
///
/// ## Retryable variants
///
/// The following variants indicate transient conditions that callers *may* retry:
/// - [`Error::Http`] — network-level failures (connection reset, timeout, …)
/// - [`Error::Api`] with `status >= 500` or `status == 429`
/// - [`Error::TenantOverloaded`] — back off and retry when capacity is available
/// - [`Error::CircuitOpen`] — retry after the `until` instant
///
/// ## Token accounting
///
/// [`Error::WithPartialUsage`] wraps any other variant and carries the token
/// usage accumulated before the failure. Inspect it with [`Error::partial_usage`]
/// to charge tokens even on error.
#[derive(Error, Debug)]
pub enum Error {
    /// An HTTP-level error from the `reqwest` client (network failure, TLS error, etc.).
    ///
    /// Potentially retryable depending on the underlying cause.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization or deserialization failed.
    ///
    /// Indicates a protocol mismatch or a malformed API response. Not retryable.
    #[error("JSON serialization/deserialization failed: {0}")]
    Json(#[from] serde_json::Error),

    /// The LLM API returned a non-2xx HTTP status code.
    ///
    /// `status == 429` is rate-limited (retryable). `status >= 500` is a
    /// server error (retryable). `status == 400` / `401` / `403` are not
    /// retryable without changing the request.
    #[error("API error ({status}): {message}")]
    Api {
        /// HTTP status code returned by the API.
        status: u16,
        /// Human-readable error message from the response body.
        message: String,
    },

    /// A general agent-level error not covered by a more specific variant.
    ///
    /// Produced by tool execution failures, orchestrator logic errors, and
    /// other agent-layer problems.
    #[error("Agent error: {0}")]
    Agent(String),

    /// Authentication or authorization failure.
    ///
    /// Typically indicates a missing or invalid API key. Not retryable without
    /// supplying valid credentials.
    #[error("Authentication error: {0}")]
    Auth(String),

    /// The agent loop reached its configured maximum turn count without finishing.
    ///
    /// Not retryable — callers should increase `max_turns` or redesign the task.
    #[error("Max turns ({0}) exceeded")]
    MaxTurnsExceeded(usize),

    /// The LLM response was cut off because `max_tokens` was reached.
    ///
    /// The agent loop surfaces this as an error when truncation is fatal. Callers
    /// can increase `max_tokens` or compress context and retry.
    #[error("Response truncated (max_tokens reached)")]
    Truncated,

    /// The agent run exceeded the configured wall-clock timeout.
    ///
    /// Potentially retryable with a longer timeout or a simpler task.
    #[error("Run timed out after {0:?}")]
    RunTimeout(Duration),

    /// An error originating from the Model Context Protocol (MCP) client or server.
    ///
    /// Covers handshake failures, protocol violations, and tool call errors
    /// returned by remote MCP servers.
    #[error("MCP error: {0}")]
    Mcp(String),

    /// An error from the Agent-to-Agent (A2A) protocol layer.
    ///
    /// Returned when communicating with remote A2A agents fails.
    #[error("A2A error: {0}")]
    A2a(String),

    /// An error in configuration parsing or validation.
    ///
    /// Produced by `HeartbitConfig` deserialization and by builder `build()` calls
    /// that detect invalid combinations of options.
    #[error("Configuration error: {0}")]
    Config(String),

    /// A persistence-layer error (e.g., PostgreSQL task-store failure).
    ///
    /// Potentially retryable on transient connection errors.
    #[error("Store error: {0}")]
    Store(String),

    /// An error in the agent memory subsystem (recall, store, prune, etc.).
    #[error("Memory error: {0}")]
    Memory(String),

    /// An error in the knowledge-base subsystem (indexing, chunking, search).
    #[error("Knowledge error: {0}")]
    Knowledge(String),

    /// A guardrail denied or errored during a request.
    ///
    /// Produced when a [`crate::Guardrail`] hook returns `Deny` or when the
    /// guardrail itself fails. The message contains the denial reason.
    #[error("Guardrail error: {0}")]
    Guardrail(String),

    /// An error in the daemon execution path (Kafka consumer, dispatcher, etc.).
    #[error("Daemon error: {0}")]
    Daemon(String),

    /// An error in the sensor pipeline (RSS, webhook, schedule triggers).
    #[error("Sensor error: {0}")]
    Sensor(String),

    /// The agent exceeded its token budget before completing.
    ///
    /// `used` is the total tokens consumed; `limit` is the configured cap.
    /// Not retryable without either increasing the budget or reducing the task.
    #[error("Token budget exceeded: used {used}, limit {limit}")]
    BudgetExceeded {
        /// Total tokens consumed before the budget was exhausted.
        used: u64,
        /// The configured token budget that was exceeded.
        limit: u64,
    },

    /// A workflow run hit its runaway agent backstop (total agents per run).
    ///
    /// Produced by the dynamic-workflow combinator core when more than `limit`
    /// agents are issued in a single run. Distinct from [`Self::BudgetExceeded`]
    /// (which is about tokens); this is a structural guard against runaway loops.
    #[error("Agent backstop exceeded: limit {limit} agents per run")]
    AgentBudgetExceeded {
        /// The configured maximum number of agents per run.
        limit: u64,
    },

    /// An error in the WebSocket/session channel layer.
    #[error("Channel error: {0}")]
    Channel(String),

    /// An error originating from the Telegram bot adapter.
    #[error("Telegram error: {0}")]
    Telegram(String),

    /// A kill switch was activated, terminating the agent run immediately.
    ///
    /// Produced by the kill-switch guardrail when a prohibited pattern is detected.
    #[error("Kill switch activated: {0}")]
    KillSwitch(String),

    /// The agent attempted a filesystem operation that violates the sandbox policy.
    ///
    /// Produced by `CorePathPolicy::check_path` or the Landlock sandbox.
    #[error("Sandbox violation: {0}")]
    Sandbox(String),

    /// The tenant has reached its maximum concurrent-request capacity.
    ///
    /// Retryable: callers should back off and retry after a delay.
    #[error("tenant {tenant_id} overloaded: in_flight={in_flight}, cap={cap}")]
    TenantOverloaded {
        /// The tenant identifier that is overloaded.
        tenant_id: String,
        /// Number of requests currently in flight for this tenant.
        in_flight: usize,
        /// Maximum allowed concurrent requests for this tenant.
        cap: usize,
    },

    /// The LLM provider's circuit breaker is open; requests are being shed.
    ///
    /// Retryable: callers should retry after the `until` instant has passed.
    #[error("circuit breaker open: retry after {until:?} (prev open duration: {prev_duration:?})")]
    CircuitOpen {
        /// The instant after which requests should be retried.
        until: std::time::Instant,
        /// How long the circuit was open in the previous open window.
        prev_duration: std::time::Duration,
    },

    /// Wraps another error with partial token usage accumulated before failure.
    ///
    /// Used by `AgentRunner::execute` to surface tokens consumed before an error.
    /// Inspect partial usage with [`Error::partial_usage`]. Re-wrapping an existing
    /// `WithPartialUsage` replaces the usage rather than nesting.
    #[error("{source}")]
    WithPartialUsage {
        /// The underlying error that caused the agent run to abort.
        #[source]
        source: Box<Error>,
        /// Token usage accumulated before the error occurred.
        usage: TokenUsage,
    },
}

impl Error {
    /// Wrap this error with partial token usage data.
    ///
    /// If `self` is already `WithPartialUsage`, the inner error is unwrapped
    /// first to prevent nesting. The new `usage` replaces the old one.
    pub fn with_partial_usage(self, usage: TokenUsage) -> Self {
        let inner = match self {
            Error::WithPartialUsage { source, .. } => *source,
            other => other,
        };
        Error::WithPartialUsage {
            source: Box::new(inner),
            usage,
        }
    }

    /// Wrap this error with the sum of `prior` usage and the error's own partial usage.
    ///
    /// Shorthand for `e.with_partial_usage(prior + e.partial_usage())`.
    pub fn accumulate_usage(self, prior: TokenUsage) -> Self {
        let mut usage = prior;
        usage += self.partial_usage();
        self.with_partial_usage(usage)
    }

    /// Extract partial token usage from this error.
    /// Returns `TokenUsage::default()` for errors that don't carry usage data.
    pub fn partial_usage(&self) -> TokenUsage {
        match self {
            Error::WithPartialUsage { usage, .. } => *usage,
            _ => TokenUsage::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let err = Error::Api {
            status: 429,
            message: "rate limited".into(),
        };
        assert_eq!(err.to_string(), "API error (429): rate limited");

        let err = Error::MaxTurnsExceeded(10);
        assert_eq!(err.to_string(), "Max turns (10) exceeded");

        let err = Error::Truncated;
        assert_eq!(err.to_string(), "Response truncated (max_tokens reached)");
    }

    #[test]
    fn error_auth_display_message() {
        let err = Error::Auth("invalid token".into());
        assert_eq!(err.to_string(), "Authentication error: invalid token");
    }

    #[test]
    fn error_mcp_display_message() {
        let err = Error::Mcp("connection refused".into());
        assert_eq!(err.to_string(), "MCP error: connection refused");
    }

    #[test]
    fn error_a2a_display_message() {
        let err = Error::A2a("agent not found".into());
        assert_eq!(err.to_string(), "A2A error: agent not found");
    }

    #[test]
    fn error_store_display_message() {
        let err = Error::Store("connection refused".into());
        assert_eq!(err.to_string(), "Store error: connection refused");
    }

    #[test]
    fn error_memory_display_message() {
        let err = Error::Memory("not found".into());
        assert_eq!(err.to_string(), "Memory error: not found");
    }

    #[test]
    fn error_knowledge_display_message() {
        let err = Error::Knowledge("file not found".into());
        assert_eq!(err.to_string(), "Knowledge error: file not found");
    }

    #[test]
    fn error_guardrail_display_message() {
        let err = Error::Guardrail("PII detected in output".into());
        assert_eq!(err.to_string(), "Guardrail error: PII detected in output");
    }

    #[test]
    fn error_daemon_display_message() {
        let err = Error::Daemon("broker connection refused".into());
        assert_eq!(err.to_string(), "Daemon error: broker connection refused");
    }

    #[test]
    fn error_sensor_display_message() {
        let err = Error::Sensor("RSS feed unreachable".into());
        assert_eq!(err.to_string(), "Sensor error: RSS feed unreachable");
    }

    #[test]
    fn error_channel_display_message() {
        let err = Error::Channel("connection closed".into());
        assert_eq!(err.to_string(), "Channel error: connection closed");
    }

    #[test]
    fn error_telegram_display_message() {
        let err = Error::Telegram("bot token invalid".into());
        assert_eq!(err.to_string(), "Telegram error: bot token invalid");
    }

    #[test]
    fn error_run_timeout_display_message() {
        let err = Error::RunTimeout(Duration::from_secs(30));
        assert_eq!(err.to_string(), "Run timed out after 30s");
    }

    #[test]
    fn run_timeout_with_partial_usage() {
        let usage = TokenUsage {
            input_tokens: 200,
            output_tokens: 100,
            ..Default::default()
        };
        let err = Error::RunTimeout(Duration::from_secs(60)).with_partial_usage(usage);
        assert_eq!(err.to_string(), "Run timed out after 60s");
        let partial = err.partial_usage();
        assert_eq!(partial.input_tokens, 200);
        assert_eq!(partial.output_tokens, 100);
    }

    #[test]
    fn with_partial_usage_wraps_error() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        let err = Error::MaxTurnsExceeded(5).with_partial_usage(usage);
        assert_eq!(err.to_string(), "Max turns (5) exceeded");
        let partial = err.partial_usage();
        assert_eq!(partial.input_tokens, 100);
        assert_eq!(partial.output_tokens, 50);
    }

    #[test]
    fn with_partial_usage_unwraps_existing() {
        let inner_usage = TokenUsage {
            input_tokens: 50,
            output_tokens: 25,
            ..Default::default()
        };
        let outer_usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        // First wrap
        let err = Error::MaxTurnsExceeded(5).with_partial_usage(inner_usage);
        // Second wrap should unwrap the first, not nest
        let err = err.with_partial_usage(outer_usage);

        // Should be exactly one layer of WithPartialUsage
        match &err {
            Error::WithPartialUsage { source, usage } => {
                assert!(
                    matches!(**source, Error::MaxTurnsExceeded(5)),
                    "inner error should be MaxTurnsExceeded, got: {source}"
                );
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.output_tokens, 50);
            }
            other => panic!("expected WithPartialUsage, got: {other}"),
        }
    }

    #[test]
    fn error_budget_exceeded_display_message() {
        let err = Error::BudgetExceeded {
            used: 150000,
            limit: 100000,
        };
        assert_eq!(
            err.to_string(),
            "Token budget exceeded: used 150000, limit 100000"
        );
    }

    #[test]
    fn budget_exceeded_with_partial_usage() {
        let usage = TokenUsage {
            input_tokens: 100000,
            output_tokens: 50000,
            ..Default::default()
        };
        let err = Error::BudgetExceeded {
            used: 150000,
            limit: 100000,
        }
        .with_partial_usage(usage);
        assert_eq!(
            err.to_string(),
            "Token budget exceeded: used 150000, limit 100000"
        );
        let partial = err.partial_usage();
        assert_eq!(partial.input_tokens, 100000);
        assert_eq!(partial.output_tokens, 50000);
    }

    #[test]
    fn partial_usage_returns_default_for_plain_errors() {
        let err = Error::Truncated;
        let partial = err.partial_usage();
        assert_eq!(partial, TokenUsage::default());
    }
}
