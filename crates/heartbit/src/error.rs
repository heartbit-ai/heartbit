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

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Store error: {0}")]
    Store(String),

    #[error("Memory error: {0}")]
    Memory(String),
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
    fn error_store_display_message() {
        let err = Error::Store("connection refused".into());
        assert_eq!(err.to_string(), "Store error: connection refused");
    }

    #[test]
    fn error_memory_display_message() {
        let err = Error::Memory("not found".into());
        assert_eq!(err.to_string(), "Memory error: not found");
    }
}
