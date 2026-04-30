use crate::error::Error;

/// Actionable classification of LLM provider errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// The conversation context exceeds the model's context window.
    ContextOverflow,
    /// Rate limited (HTTP 429). Already retried by `RetryingProvider`.
    RateLimited,
    /// Authentication failure (HTTP 401/403).
    AuthError,
    /// Server-side failure (HTTP 500/502/503/529).
    ServerError,
    /// Client error that is not overflow (other HTTP 400).
    InvalidRequest,
    /// Unrecognized error — no actionable recovery.
    Unknown,
}

/// Classify an [`Error`] into an actionable [`ErrorClass`].
///
/// Primarily useful for `Error::Api` errors where the HTTP status code and
/// message body determine recovery strategy.
pub fn classify(error: &Error) -> ErrorClass {
    // Unwrap WithPartialUsage to classify the inner error.
    let inner = match error {
        Error::WithPartialUsage { source, .. } => source.as_ref(),
        other => other,
    };

    match inner {
        Error::Api { status, message } => classify_api(*status, message),
        Error::Http(_) => ErrorClass::Unknown,
        _ => ErrorClass::Unknown,
    }
}

fn classify_api(status: u16, message: &str) -> ErrorClass {
    match status {
        401 | 403 => ErrorClass::AuthError,
        429 => ErrorClass::RateLimited,
        500 | 502 | 503 | 529 => ErrorClass::ServerError,
        400 => {
            if is_context_overflow(message) {
                ErrorClass::ContextOverflow
            } else {
                ErrorClass::InvalidRequest
            }
        }
        _ => ErrorClass::Unknown,
    }
}

/// Check if an error message indicates context overflow.
///
/// Uses case-insensitive substring matching (no regex dependency).
fn is_context_overflow(message: &str) -> bool {
    const PATTERNS: &[&str] = &[
        "prompt is too long",
        "maximum context length",
        "context_length_exceeded",
        "context window",
        "too many tokens",
        "input is too long",
        "exceeds the model's maximum context",
        "request too large",
        "content too large",
    ];

    let lower = message.to_lowercase();
    PATTERNS.iter().any(|p| lower.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Auth errors ---

    #[test]
    fn classify_401_as_auth_error() {
        let err = Error::Api {
            status: 401,
            message: "Unauthorized".into(),
        };
        assert_eq!(classify(&err), ErrorClass::AuthError);
    }

    #[test]
    fn classify_403_as_auth_error() {
        let err = Error::Api {
            status: 403,
            message: "Forbidden".into(),
        };
        assert_eq!(classify(&err), ErrorClass::AuthError);
    }

    // --- Rate limited ---

    #[test]
    fn classify_429_as_rate_limited() {
        let err = Error::Api {
            status: 429,
            message: "Too Many Requests".into(),
        };
        assert_eq!(classify(&err), ErrorClass::RateLimited);
    }

    // --- Server errors ---

    #[test]
    fn classify_500_as_server_error() {
        let err = Error::Api {
            status: 500,
            message: "Internal Server Error".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ServerError);
    }

    #[test]
    fn classify_502_as_server_error() {
        let err = Error::Api {
            status: 502,
            message: "Bad Gateway".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ServerError);
    }

    #[test]
    fn classify_503_as_server_error() {
        let err = Error::Api {
            status: 503,
            message: "Service Unavailable".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ServerError);
    }

    #[test]
    fn classify_529_as_server_error() {
        let err = Error::Api {
            status: 529,
            message: "Overloaded".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ServerError);
    }

    // --- Context overflow (400 with overflow patterns) ---

    #[test]
    fn classify_400_prompt_too_long() {
        let err = Error::Api {
            status: 400,
            message: "prompt is too long".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_400_maximum_context_length() {
        let err = Error::Api {
            status: 400,
            message: "This request exceeds the maximum context length".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_400_context_length_exceeded() {
        let err = Error::Api {
            status: 400,
            message: "context_length_exceeded".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_400_request_too_large() {
        let err = Error::Api {
            status: 400,
            message: "request too large for this model".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_400_content_too_large() {
        let err = Error::Api {
            status: 400,
            message: "content too large".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    /// `max_tokens` in a 400 message can mean parameter validation (e.g.,
    /// "max_tokens: 4096 must be less than ..."), not context overflow.
    /// We should NOT classify it as ContextOverflow.
    #[test]
    fn classify_400_max_tokens_parameter_is_not_overflow() {
        let err = Error::Api {
            status: 400,
            message: "max_tokens: 4096 must be less than 2048".into(),
        };
        assert_eq!(classify(&err), ErrorClass::InvalidRequest);
    }

    #[test]
    fn classify_400_context_window() {
        let err = Error::Api {
            status: 400,
            message: "exceeds the context window".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_400_too_many_tokens() {
        let err = Error::Api {
            status: 400,
            message: "too many tokens in the request".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_400_input_too_long() {
        let err = Error::Api {
            status: 400,
            message: "input is too long for model".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_400_exceeds_model_maximum_context() {
        let err = Error::Api {
            status: 400,
            message: "exceeds the model's maximum context length".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    #[test]
    fn classify_400_case_insensitive() {
        let err = Error::Api {
            status: 400,
            message: "PROMPT IS TOO LONG".into(),
        };
        assert_eq!(classify(&err), ErrorClass::ContextOverflow);
    }

    // --- Invalid request (400 without overflow pattern) ---

    #[test]
    fn classify_400_generic_as_invalid_request() {
        let err = Error::Api {
            status: 400,
            message: "invalid parameter: temperature must be between 0 and 1".into(),
        };
        assert_eq!(classify(&err), ErrorClass::InvalidRequest);
    }

    // --- HTTP / network errors ---

    #[test]
    fn classify_http_error_as_unknown() {
        // Build a reqwest error by making a request to an invalid URL.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let http_err = rt
            .block_on(reqwest::get("http://[::0]:1"))
            .expect_err("should fail");
        let err = Error::Http(http_err);
        assert_eq!(classify(&err), ErrorClass::Unknown);
    }

    // --- Other error variants ---

    #[test]
    fn classify_agent_error_as_unknown() {
        let err = Error::Agent("something went wrong".into());
        assert_eq!(classify(&err), ErrorClass::Unknown);
    }

    #[test]
    fn classify_max_turns_exceeded_as_unknown() {
        let err = Error::MaxTurnsExceeded(10);
        assert_eq!(classify(&err), ErrorClass::Unknown);
    }

    #[test]
    fn classify_truncated_as_unknown() {
        let err = Error::Truncated;
        assert_eq!(classify(&err), ErrorClass::Unknown);
    }

    #[test]
    fn classify_config_error_as_unknown() {
        let err = Error::Config("bad config".into());
        assert_eq!(classify(&err), ErrorClass::Unknown);
    }

    #[test]
    fn classify_mcp_error_as_unknown() {
        let err = Error::Mcp("connection refused".into());
        assert_eq!(classify(&err), ErrorClass::Unknown);
    }

    // --- WithPartialUsage unwrapping ---

    #[test]
    fn classify_unwraps_with_partial_usage() {
        use crate::llm::types::TokenUsage;

        let inner = Error::Api {
            status: 429,
            message: "rate limited".into(),
        };
        let wrapped = inner.with_partial_usage(TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        });
        assert_eq!(classify(&wrapped), ErrorClass::RateLimited);
    }

    #[test]
    fn classify_unwraps_partial_usage_context_overflow() {
        use crate::llm::types::TokenUsage;

        let inner = Error::Api {
            status: 400,
            message: "prompt is too long".into(),
        };
        let wrapped = inner.with_partial_usage(TokenUsage::default());
        assert_eq!(classify(&wrapped), ErrorClass::ContextOverflow);
    }

    // --- Unknown status codes ---

    #[test]
    fn classify_unknown_status_as_unknown() {
        let err = Error::Api {
            status: 418,
            message: "I'm a teapot".into(),
        };
        assert_eq!(classify(&err), ErrorClass::Unknown);
    }
}
