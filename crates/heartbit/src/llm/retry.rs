use std::time::Duration;

use crate::error::Error;
use crate::llm::types::{CompletionRequest, CompletionResponse};

use super::LlmProvider;

/// Configuration for retry behavior on transient LLM API failures.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries, just the initial call).
    pub max_retries: u32,
    /// Base delay for exponential backoff (doubled on each retry).
    pub base_delay: Duration,
    /// Maximum delay cap.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
        }
    }
}

impl From<&crate::config::RetryProviderConfig> for RetryConfig {
    fn from(r: &crate::config::RetryProviderConfig) -> Self {
        Self {
            max_retries: r.max_retries,
            base_delay: Duration::from_millis(r.base_delay_ms),
            max_delay: Duration::from_millis(r.max_delay_ms),
        }
    }
}

/// Wraps any `LlmProvider` with automatic retry + exponential backoff.
///
/// Retries on:
/// - HTTP 429 (rate limit)
/// - HTTP 500, 502, 503, 529 (server errors)
/// - Network errors (`Error::Http`)
///
/// Does NOT retry on:
/// - HTTP 400, 401, 403, 404 (client errors — retrying won't help)
/// - JSON/SSE parse errors (deterministic failures)
/// - Agent/Config/Memory/Store errors (not LLM-related)
pub struct RetryingProvider<P> {
    inner: P,
    config: RetryConfig,
}

impl<P> RetryingProvider<P> {
    pub fn new(inner: P, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    /// Wrap a provider with default retry config (3 retries, 500ms base delay).
    pub fn with_defaults(inner: P) -> Self {
        Self::new(inner, RetryConfig::default())
    }
}

/// Determine whether an error is transient and worth retrying.
fn is_retryable(err: &Error) -> bool {
    match err {
        Error::Api { status, .. } => matches!(*status, 429 | 500 | 502 | 503 | 529),
        Error::Http(_) => true,
        _ => false,
    }
}

/// Compute the delay for a given attempt using exponential backoff.
/// Attempt 0 = base_delay, attempt 1 = 2*base_delay, etc.
fn compute_delay(config: &RetryConfig, attempt: u32) -> Duration {
    let delay = config
        .base_delay
        .saturating_mul(1u32.checked_shl(attempt).unwrap_or(u32::MAX));
    delay.min(config.max_delay)
}

impl<P: LlmProvider> LlmProvider for RetryingProvider<P> {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let mut last_err: Option<Error> = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = compute_delay(&self.config, attempt - 1);
                tracing::warn!(
                    attempt = attempt,
                    max_retries = self.config.max_retries,
                    delay_ms = delay.as_millis() as u64,
                    error = %last_err.as_ref().expect("last_err set before retry"),
                    "retrying LLM call after transient failure"
                );
                tokio::time::sleep(delay).await;
            }

            match self.inner.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) if is_retryable(&e) => {
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }

        // All retries exhausted — return the last error
        Err(last_err.expect("at least one attempt must have been made"))
    }

    // NOTE: On retry, `on_text` is called again from the start of the new
    // response. The final `CompletionResponse` is correct, but the callback
    // may receive duplicate text deltas if a previous attempt partially streamed.
    async fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &super::OnText,
    ) -> Result<CompletionResponse, Error> {
        let mut last_err: Option<Error> = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = compute_delay(&self.config, attempt - 1);
                tracing::warn!(
                    attempt = attempt,
                    max_retries = self.config.max_retries,
                    delay_ms = delay.as_millis() as u64,
                    error = %last_err.as_ref().expect("last_err set before retry"),
                    "retrying streaming LLM call after transient failure"
                );
                tokio::time::sleep(delay).await;
            }

            match self.inner.stream_complete(request.clone(), on_text).await {
                Ok(response) => return Ok(response),
                Err(e) if is_retryable(&e) => {
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_err.expect("at least one attempt must have been made"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{Message, StopReason, TokenUsage};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A mock provider that fails the first N calls with a specified error,
    /// then succeeds.
    struct FailNTimes {
        remaining_failures: AtomicU32,
        error_factory: Box<dyn Fn() -> Error + Send + Sync>,
        call_count: Arc<AtomicU32>,
    }

    impl FailNTimes {
        fn new(
            failures: u32,
            error_factory: impl Fn() -> Error + Send + Sync + 'static,
        ) -> (Self, Arc<AtomicU32>) {
            let count = Arc::new(AtomicU32::new(0));
            (
                Self {
                    remaining_failures: AtomicU32::new(failures),
                    error_factory: Box::new(error_factory),
                    call_count: count.clone(),
                },
                count,
            )
        }
    }

    fn success_response() -> CompletionResponse {
        CompletionResponse {
            content: vec![crate::llm::types::ContentBlock::Text { text: "ok".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        }
    }

    impl LlmProvider for FailNTimes {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            if self.remaining_failures.load(Ordering::SeqCst) > 0 {
                self.remaining_failures.fetch_sub(1, Ordering::SeqCst);
                return Err((self.error_factory)());
            }
            Ok(success_response())
        }
    }

    fn test_request() -> CompletionRequest {
        CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("test")],
            tools: vec![],
            max_tokens: 100,
        }
    }

    fn fast_config(max_retries: u32) -> RetryConfig {
        RetryConfig {
            max_retries,
            base_delay: Duration::from_millis(1), // Fast for tests
            max_delay: Duration::from_millis(10),
        }
    }

    #[tokio::test]
    async fn succeeds_on_first_attempt() {
        let (mock, count) = FailNTimes::new(0, || Error::Api {
            status: 429,
            message: "rate limited".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let result = provider.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retries_on_429_and_succeeds() {
        let (mock, count) = FailNTimes::new(2, || Error::Api {
            status: 429,
            message: "rate limited".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let result = provider.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn retries_on_500_and_succeeds() {
        let (mock, count) = FailNTimes::new(1, || Error::Api {
            status: 500,
            message: "internal server error".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let result = provider.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retries_on_502_and_succeeds() {
        let (mock, count) = FailNTimes::new(1, || Error::Api {
            status: 502,
            message: "bad gateway".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let result = provider.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retries_on_503_and_succeeds() {
        let (mock, count) = FailNTimes::new(1, || Error::Api {
            status: 503,
            message: "service unavailable".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let result = provider.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retries_on_529_and_succeeds() {
        let (mock, count) = FailNTimes::new(1, || Error::Api {
            status: 529,
            message: "overloaded".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let result = provider.complete(test_request()).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn exhausts_retries_and_returns_last_error() {
        let (mock, count) = FailNTimes::new(10, || Error::Api {
            status: 429,
            message: "rate limited".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(2));

        let result = provider.complete(test_request()).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::Api { status: 429, .. }));
        assert_eq!(count.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn does_not_retry_400() {
        let (mock, count) = FailNTimes::new(5, || Error::Api {
            status: 400,
            message: "bad request".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let result = provider.complete(test_request()).await;
        assert!(result.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 1); // No retries
    }

    #[tokio::test]
    async fn does_not_retry_401() {
        let (mock, count) = FailNTimes::new(5, || Error::Api {
            status: 401,
            message: "unauthorized".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let result = provider.complete(test_request()).await;
        assert!(result.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn does_not_retry_json_parse_error() {
        let (mock, count) = FailNTimes::new(5, || {
            Error::Json(serde_json::from_str::<()>("invalid").unwrap_err())
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let result = provider.complete(test_request()).await;
        assert!(result.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn zero_retries_means_single_attempt() {
        let (mock, count) = FailNTimes::new(1, || Error::Api {
            status: 429,
            message: "rate limited".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(0));

        let result = provider.complete(test_request()).await;
        assert!(result.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stream_complete_retries_on_transient_failure() {
        // FailNTimes only implements complete; the default stream_complete
        // delegates to complete. RetryingProvider::stream_complete retries
        // through that chain.
        let (mock, count) = FailNTimes::new(2, || Error::Api {
            status: 429,
            message: "rate limited".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let on_text: &crate::llm::OnText = &|_| {};
        let result = provider.stream_complete(test_request(), on_text).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn stream_complete_does_not_retry_non_retryable() {
        let (mock, count) = FailNTimes::new(5, || Error::Api {
            status: 400,
            message: "bad request".into(),
        });
        let provider = RetryingProvider::new(mock, fast_config(3));

        let on_text: &crate::llm::OnText = &|_| {};
        let result = provider.stream_complete(test_request(), on_text).await;
        assert!(result.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 1); // No retries
    }

    #[test]
    fn default_config_values() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(30));
    }

    #[test]
    fn is_retryable_checks() {
        // Retryable
        assert!(is_retryable(&Error::Api {
            status: 429,
            message: "".into()
        }));
        assert!(is_retryable(&Error::Api {
            status: 500,
            message: "".into()
        }));
        assert!(is_retryable(&Error::Api {
            status: 502,
            message: "".into()
        }));
        assert!(is_retryable(&Error::Api {
            status: 503,
            message: "".into()
        }));
        assert!(is_retryable(&Error::Api {
            status: 529,
            message: "".into()
        }));

        // Not retryable
        assert!(!is_retryable(&Error::Api {
            status: 400,
            message: "".into()
        }));
        assert!(!is_retryable(&Error::Api {
            status: 401,
            message: "".into()
        }));
        assert!(!is_retryable(&Error::Api {
            status: 403,
            message: "".into()
        }));
        assert!(!is_retryable(&Error::Api {
            status: 404,
            message: "".into()
        }));
        assert!(!is_retryable(&Error::Agent("test".into())));
        assert!(!is_retryable(&Error::Config("test".into())));
        assert!(!is_retryable(&Error::Memory("test".into())));
    }

    #[test]
    fn compute_delay_exponential_backoff() {
        let config = RetryConfig {
            max_retries: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
        };

        assert_eq!(compute_delay(&config, 0), Duration::from_millis(100));
        assert_eq!(compute_delay(&config, 1), Duration::from_millis(200));
        assert_eq!(compute_delay(&config, 2), Duration::from_millis(400));
        assert_eq!(compute_delay(&config, 3), Duration::from_millis(800));
    }

    #[test]
    fn compute_delay_caps_at_max() {
        let config = RetryConfig {
            max_retries: 10,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(5),
        };

        // attempt 3 = 8000ms, capped to 5000ms
        assert_eq!(compute_delay(&config, 3), Duration::from_secs(5));
        assert_eq!(compute_delay(&config, 10), Duration::from_secs(5));
    }

    #[test]
    fn compute_delay_handles_overflow() {
        let config = RetryConfig {
            max_retries: 100,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
        };

        // Very large attempt number should not panic
        let delay = compute_delay(&config, 50);
        assert_eq!(delay, Duration::from_secs(60)); // capped at max
    }

    #[test]
    fn retry_config_from_provider_config() {
        let provider_config = crate::config::RetryProviderConfig {
            max_retries: 5,
            base_delay_ms: 1000,
            max_delay_ms: 60000,
        };
        let config = RetryConfig::from(&provider_config);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.base_delay, Duration::from_millis(1000));
        assert_eq!(config.max_delay, Duration::from_millis(60000));
    }
}
