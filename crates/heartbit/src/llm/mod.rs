pub mod anthropic;
pub mod error_class;
pub mod openrouter;
pub mod pricing;
pub mod retry;
pub mod types;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Error;
use crate::llm::types::{CompletionRequest, CompletionResponse};

/// Callback invoked with each text delta during streaming.
pub type OnText = dyn Fn(&str) + Send + Sync;

/// Decision returned by the `OnApproval` callback.
///
/// `Allow` and `Deny` behave like the previous `true`/`false` return.
/// `AlwaysAllow` and `AlwaysDeny` additionally persist the decision as a
/// learned permission rule so it survives across sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    /// Allow this time.
    Allow,
    /// Deny this time.
    Deny,
    /// Allow and persist as a permission rule.
    AlwaysAllow,
    /// Deny and persist as a permission rule.
    AlwaysDeny,
}

impl ApprovalDecision {
    /// Returns `true` when the decision allows execution.
    pub fn is_allowed(self) -> bool {
        matches!(self, Self::Allow | Self::AlwaysAllow)
    }

    /// Returns `true` when the decision should be persisted.
    pub fn is_persistent(self) -> bool {
        matches!(self, Self::AlwaysAllow | Self::AlwaysDeny)
    }
}

impl From<bool> for ApprovalDecision {
    fn from(allowed: bool) -> Self {
        if allowed { Self::Allow } else { Self::Deny }
    }
}

/// Callback invoked before tool execution for human-in-the-loop approval.
///
/// Receives the list of tool calls the LLM wants to execute.
/// Returns an [`ApprovalDecision`] indicating whether to proceed.
/// `AlwaysAllow`/`AlwaysDeny` additionally persist the decision as a
/// learned permission rule.
pub type OnApproval = dyn Fn(&[crate::llm::types::ToolCall]) -> ApprovalDecision + Send + Sync;

/// Trait for LLM providers.
///
/// Uses RPITIT (`impl Future`) which means this trait is NOT dyn-compatible.
/// All consumers are generic over `P: LlmProvider`. This is intentional:
/// one provider per process, no need for trait objects.
///
/// For dynamic dispatch, use [`BoxedProvider`] which wraps any `LlmProvider`
/// behind [`DynLlmProvider`].
pub trait LlmProvider: Send + Sync {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, Error>> + Send;

    /// Stream a completion, calling `on_text` for each text delta as it arrives.
    ///
    /// The returned `CompletionResponse` contains the full accumulated response
    /// (same as `complete()`), but text was emitted incrementally via the callback.
    ///
    /// Default: falls back to `complete()` (no incremental streaming).
    fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &OnText,
    ) -> impl Future<Output = Result<CompletionResponse, Error>> + Send {
        let _ = on_text;
        self.complete(request)
    }

    /// Return the model identifier, if known.
    ///
    /// Used for audit trail events. Default returns `None`.
    fn model_name(&self) -> Option<&str> {
        None
    }
}

// ---------------------------------------------------------------------------
// DynLlmProvider — object-safe adapter for LlmProvider (RPITIT → dyn)
// ---------------------------------------------------------------------------

/// Object-safe version of [`LlmProvider`] for dynamic dispatch.
///
/// `LlmProvider` uses RPITIT (not dyn-compatible). This trait wraps it via
/// `Pin<Box<dyn Future>>` so providers can be stored as `Arc<dyn DynLlmProvider>`.
///
/// A blanket impl covers all `LlmProvider` types automatically.
///
/// Used by the Restate service layer (`AgentServiceImpl`) and by
/// [`BoxedProvider`] for type-erased standalone use.
pub trait DynLlmProvider: Send + Sync {
    fn complete<'a>(
        &'a self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>>;

    fn stream_complete<'a>(
        &'a self,
        request: CompletionRequest,
        on_text: &'a OnText,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>>;

    fn model_name(&self) -> Option<&str>;
}

impl<P: LlmProvider> DynLlmProvider for P {
    fn complete<'a>(
        &'a self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
        Box::pin(LlmProvider::complete(self, request))
    }

    fn stream_complete<'a>(
        &'a self,
        request: CompletionRequest,
        on_text: &'a OnText,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
        Box::pin(LlmProvider::stream_complete(self, request, on_text))
    }

    fn model_name(&self) -> Option<&str> {
        LlmProvider::model_name(self)
    }
}

// ---------------------------------------------------------------------------
// BoxedProvider — type-erased LlmProvider via DynLlmProvider
// ---------------------------------------------------------------------------

/// Type-erased LLM provider for use when dynamic dispatch is needed.
///
/// Wraps any [`LlmProvider`] behind `Box<dyn DynLlmProvider>`. Implements
/// `LlmProvider` itself, so it can be used with `AgentRunner<BoxedProvider>`
/// and `Orchestrator<BoxedProvider>`, eliminating the need for generic code
/// at the call site.
///
/// # Example
///
/// ```ignore
/// let provider = BoxedProvider::new(AnthropicProvider::new(key, model));
/// let runner = AgentRunner::builder(Arc::new(provider))
///     .name("agent")
///     .build()?;
/// ```
pub struct BoxedProvider(Box<dyn DynLlmProvider>);

impl BoxedProvider {
    /// Create a type-erased provider from any concrete `LlmProvider`.
    pub fn new<P: LlmProvider + 'static>(provider: P) -> Self {
        Self(Box::new(provider))
    }

    /// Create a type-erased provider from an `Arc<P>`.
    ///
    /// Useful when the provider is already behind an `Arc` (e.g., shared between
    /// the orchestrator and sub-agents) and needs to be converted to `BoxedProvider`
    /// for type erasure without consuming the original.
    pub fn from_arc<P: LlmProvider + 'static>(provider: Arc<P>) -> Self {
        /// Internal adapter: delegates to the `Arc<P>` inner provider.
        struct ArcAdapter<P>(Arc<P>);

        impl<P: LlmProvider> LlmProvider for ArcAdapter<P> {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                self.0.complete(request).await
            }

            async fn stream_complete(
                &self,
                request: CompletionRequest,
                on_text: &OnText,
            ) -> Result<CompletionResponse, Error> {
                self.0.stream_complete(request, on_text).await
            }

            fn model_name(&self) -> Option<&str> {
                self.0.model_name()
            }
        }

        Self(Box::new(ArcAdapter(provider)))
    }
}

impl LlmProvider for BoxedProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        self.0.complete(request).await
    }

    async fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &OnText,
    ) -> Result<CompletionResponse, Error> {
        self.0.stream_complete(request, on_text).await
    }

    fn model_name(&self) -> Option<&str> {
        self.0.model_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ContentBlock, Message, StopReason, TokenUsage};
    use std::sync::{Arc, Mutex};

    struct FakeProvider;

    impl LlmProvider for FakeProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "fake".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            })
        }
    }

    struct StreamingFakeProvider;

    impl LlmProvider for StreamingFakeProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            panic!("should call stream_complete, not complete");
        }

        async fn stream_complete(
            &self,
            _request: CompletionRequest,
            on_text: &OnText,
        ) -> Result<CompletionResponse, Error> {
            on_text("hello");
            on_text(" world");
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "hello world".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            })
        }
    }

    fn test_request() -> CompletionRequest {
        CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("test")],
            tools: vec![],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        }
    }

    #[test]
    fn dyn_llm_provider_wraps_provider() {
        let provider = FakeProvider;
        let dyn_provider: &dyn DynLlmProvider = &provider;
        let _ = dyn_provider;
    }

    #[tokio::test]
    async fn boxed_provider_delegates_complete() {
        let provider = BoxedProvider::new(FakeProvider);
        // Disambiguate: BoxedProvider implements both LlmProvider and DynLlmProvider
        let response = LlmProvider::complete(&provider, test_request())
            .await
            .unwrap();
        assert_eq!(response.text(), "fake");
    }

    #[tokio::test]
    async fn boxed_provider_delegates_stream_complete() {
        let provider = BoxedProvider::new(StreamingFakeProvider);
        let received = Arc::new(Mutex::new(Vec::<String>::new()));
        let received_clone = received.clone();
        let on_text: &OnText = &move |text: &str| {
            received_clone
                .lock()
                .expect("test lock")
                .push(text.to_string());
        };

        let response = LlmProvider::stream_complete(&provider, test_request(), on_text)
            .await
            .unwrap();
        assert_eq!(response.text(), "hello world");

        let texts = received.lock().expect("test lock");
        assert_eq!(*texts, vec!["hello", " world"]);
    }

    #[test]
    fn boxed_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BoxedProvider>();
    }

    #[tokio::test]
    async fn boxed_provider_default_stream_falls_back_to_complete() {
        // FakeProvider only implements complete; stream_complete should fall back
        let provider = BoxedProvider::new(FakeProvider);
        let on_text: &OnText = &|_| {};
        let response = LlmProvider::stream_complete(&provider, test_request(), on_text)
            .await
            .unwrap();
        assert_eq!(response.text(), "fake");
    }

    #[tokio::test]
    async fn boxed_provider_from_arc_delegates_complete() {
        let provider = Arc::new(FakeProvider);
        let boxed = BoxedProvider::from_arc(provider);
        let response = LlmProvider::complete(&boxed, test_request()).await.unwrap();
        assert_eq!(response.text(), "fake");
    }

    #[tokio::test]
    async fn boxed_provider_from_arc_delegates_stream_complete() {
        let provider = Arc::new(StreamingFakeProvider);
        let boxed = BoxedProvider::from_arc(provider);
        let received = Arc::new(Mutex::new(Vec::<String>::new()));
        let received_clone = received.clone();
        let on_text: &OnText = &move |text: &str| {
            received_clone
                .lock()
                .expect("test lock")
                .push(text.to_string());
        };
        let response = LlmProvider::stream_complete(&boxed, test_request(), on_text)
            .await
            .unwrap();
        assert_eq!(response.text(), "hello world");
        let texts = received.lock().expect("test lock");
        assert_eq!(*texts, vec!["hello", " world"]);
    }

    #[test]
    fn model_name_default_is_none() {
        let provider = FakeProvider;
        assert!(LlmProvider::model_name(&provider).is_none());
    }

    #[test]
    fn boxed_provider_preserves_model_name() {
        struct NamedProvider;
        impl LlmProvider for NamedProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                unimplemented!()
            }
            fn model_name(&self) -> Option<&str> {
                Some("test-model")
            }
        }
        let boxed = BoxedProvider::new(NamedProvider);
        assert_eq!(LlmProvider::model_name(&boxed), Some("test-model"));
    }

    #[test]
    fn boxed_provider_from_arc_preserves_model_name() {
        struct NamedProvider;
        impl LlmProvider for NamedProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                unimplemented!()
            }
            fn model_name(&self) -> Option<&str> {
                Some("arc-model")
            }
        }
        let boxed = BoxedProvider::from_arc(Arc::new(NamedProvider));
        assert_eq!(LlmProvider::model_name(&boxed), Some("arc-model"));
    }

    #[tokio::test]
    async fn boxed_provider_from_arc_shares_underlying_provider() {
        // Verify from_arc shares the underlying provider via Arc (not a copy)
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        struct CountingProvider(Arc<std::sync::atomic::AtomicUsize>);
        impl LlmProvider for CountingProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, crate::error::Error> {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "counted".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                })
            }
        }

        let inner = Arc::new(CountingProvider(call_count.clone()));
        let boxed1 = BoxedProvider::from_arc(inner.clone());
        let boxed2 = BoxedProvider::from_arc(inner);

        LlmProvider::complete(&boxed1, test_request())
            .await
            .unwrap();
        LlmProvider::complete(&boxed2, test_request())
            .await
            .unwrap();

        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "both boxed providers should share the same underlying provider"
        );
    }

    // --- ApprovalDecision ---

    #[test]
    fn approval_decision_from_true() {
        let decision = ApprovalDecision::from(true);
        assert_eq!(decision, ApprovalDecision::Allow);
        assert!(decision.is_allowed());
        assert!(!decision.is_persistent());
    }

    #[test]
    fn approval_decision_from_false() {
        let decision = ApprovalDecision::from(false);
        assert_eq!(decision, ApprovalDecision::Deny);
        assert!(!decision.is_allowed());
        assert!(!decision.is_persistent());
    }

    #[test]
    fn approval_decision_always_allow() {
        let decision = ApprovalDecision::AlwaysAllow;
        assert!(decision.is_allowed());
        assert!(decision.is_persistent());
    }

    #[test]
    fn approval_decision_always_deny() {
        let decision = ApprovalDecision::AlwaysDeny;
        assert!(!decision.is_allowed());
        assert!(decision.is_persistent());
    }
}
