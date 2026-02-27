use std::time::Duration;

use crate::llm::types::TokenUsage;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON serialization/deserialization failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Max turns ({0}) exceeded")]
    MaxTurnsExceeded(usize),

    #[error("Response truncated (max_tokens reached)")]
    Truncated,

    #[error("Run timed out after {0:?}")]
    RunTimeout(Duration),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("A2A error: {0}")]
    A2a(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Store error: {0}")]
    Store(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Knowledge error: {0}")]
    Knowledge(String),

    #[error("Guardrail error: {0}")]
    Guardrail(String),

    #[error("Daemon error: {0}")]
    Daemon(String),

    #[error("Sensor error: {0}")]
    Sensor(String),

    #[error("Token budget exceeded: used {used}, limit {limit}")]
    BudgetExceeded { used: u64, limit: u64 },

    #[error("Channel error: {0}")]
    Channel(String),

    #[error("Telegram error: {0}")]
    Telegram(String),

    /// Wraps another error with partial token usage accumulated before failure.
    /// Used by `AgentRunner::execute` to surface tokens consumed before an error.
    #[error("{source}")]
    WithPartialUsage {
        #[source]
        source: Box<Error>,
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
