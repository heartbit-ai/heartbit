use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON serialization/deserialization failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("SSE parse error: {0}")]
    SseParse(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Max turns ({0}) exceeded")]
    MaxTurnsExceeded(usize),

    #[error("Response truncated (max_tokens reached)")]
    Truncated,
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

        let err = Error::ToolNotFound("unknown_tool".into());
        assert_eq!(err.to_string(), "Tool not found: unknown_tool");

        let err = Error::MaxTurnsExceeded(10);
        assert_eq!(err.to_string(), "Max turns (10) exceeded");

        let err = Error::Truncated;
        assert_eq!(err.to_string(), "Response truncated (max_tokens reached)");
    }
}
