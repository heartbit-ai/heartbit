use crate::error::Error;
use crate::llm::types::{CompletionRequest, CompletionResponse, ContentBlock, StopReason};
use crate::llm::{DynLlmProvider, LlmProvider, OnText};

/// Evaluates whether a cheaper model's response is "good enough"
/// to avoid escalating to a more expensive tier.
pub trait ConfidenceGate: Send + Sync {
    fn accept(&self, request: &CompletionRequest, response: &CompletionResponse) -> bool;
}

/// Zero-cost heuristic gate (no extra LLM calls).
pub struct HeuristicGate {
    /// Minimum output tokens for acceptance (default: 5).
    pub min_output_tokens: u32,
    /// Refusal phrases that trigger escalation.
    pub refusal_patterns: Vec<String>,
    /// Accept responses that include tool calls (default: true).
    pub accept_tool_calls: bool,
    /// Escalate on MaxTokens stop reason (default: true).
    pub escalate_on_max_tokens: bool,
}

impl Default for HeuristicGate {
    fn default() -> Self {
        Self {
            min_output_tokens: 5,
            refusal_patterns: default_refusal_patterns(),
            accept_tool_calls: true,
            escalate_on_max_tokens: true,
        }
    }
}

fn default_refusal_patterns() -> Vec<String> {
    [
        "I don't know",
        "I'm not sure",
        "I cannot",
        "I can't",
        "I'm unable",
        "beyond my capabilities",
        "I apologize, but",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

impl ConfidenceGate for HeuristicGate {
    fn accept(&self, _request: &CompletionRequest, response: &CompletionResponse) -> bool {
        // 1. Accept tool calls immediately
        if self.accept_tool_calls
            && response
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
        {
            return true;
        }

        // 2. Reject on MaxTokens
        if self.escalate_on_max_tokens && response.stop_reason == StopReason::MaxTokens {
            return false;
        }

        // 3. Reject short responses
        if response.usage.output_tokens < self.min_output_tokens {
            return false;
        }

        // 4. Reject refusal patterns (case-insensitive)
        let text = response.text().to_lowercase();
        for pattern in &self.refusal_patterns {
            if text.contains(&pattern.to_lowercase()) {
                return false;
            }
        }

        // 5. Accept
        true
    }
}

/// A tier in the cascade: a provider with a human-readable label.
pub struct CascadeTier {
    provider: Box<dyn DynLlmProvider>,
    label: String,
}

/// Tries cheaper models first, escalating to more expensive tiers
/// when the confidence gate rejects a response or a tier errors.
///
/// The final tier always accepts (no gate check).
/// Non-final tiers use `complete()` even for `stream_complete()` calls
/// to avoid streaming tokens that might be discarded.
pub struct CascadingProvider {
    tiers: Vec<CascadeTier>,
    gate: Box<dyn ConfidenceGate>,
}

impl CascadingProvider {
    pub fn builder() -> CascadingProviderBuilder {
        CascadingProviderBuilder {
            tiers: Vec::new(),
            gate: None,
        }
    }
}

impl LlmProvider for CascadingProvider {
    fn model_name(&self) -> Option<&str> {
        self.tiers.first().map(|t| t.label.as_str())
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        for (i, tier) in self.tiers.iter().enumerate() {
            let is_last = i == self.tiers.len() - 1;
            match tier.provider.complete(request.clone()).await {
                Ok(mut response) => {
                    if is_last || self.gate.accept(&request, &response) {
                        response.model = Some(tier.label.clone());
                        tracing::info!(
                            tier = %tier.label,
                            is_last,
                            output_tokens = response.usage.output_tokens,
                            "cascade: accepted response"
                        );
                        return Ok(response);
                    }
                    tracing::info!(
                        from = %tier.label,
                        to = %self.tiers[i + 1].label,
                        "cascade: gate rejected, escalating"
                    );
                }
                Err(e) if is_last => return Err(e),
                Err(e) => {
                    tracing::warn!(
                        tier = %tier.label,
                        error = %e,
                        "cascade: tier failed, escalating"
                    );
                }
            }
        }
        unreachable!("cascade must have at least one tier")
    }

    async fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &OnText,
    ) -> Result<CompletionResponse, Error> {
        // Single tier: stream directly
        if self.tiers.len() == 1 {
            let mut resp = self.tiers[0]
                .provider
                .stream_complete(request, on_text)
                .await?;
            resp.model = Some(self.tiers[0].label.clone());
            return Ok(resp);
        }

        // Non-final tiers: use complete() to avoid streaming tokens we might discard
        for (i, tier) in self.tiers.iter().enumerate() {
            let is_last = i == self.tiers.len() - 1;
            if is_last {
                let mut resp = tier.provider.stream_complete(request, on_text).await?;
                resp.model = Some(tier.label.clone());
                return Ok(resp);
            }
            match tier.provider.complete(request.clone()).await {
                Ok(mut response) if self.gate.accept(&request, &response) => {
                    response.model = Some(tier.label.clone());
                    tracing::info!(
                        tier = %tier.label,
                        output_tokens = response.usage.output_tokens,
                        "cascade: cheap tier accepted (stream path)"
                    );
                    // Emit text as a single chunk for streaming callers
                    let text = response.text();
                    if !text.is_empty() {
                        on_text(&text);
                    }
                    return Ok(response);
                }
                Ok(_) => {
                    tracing::info!(
                        from = %tier.label,
                        to = %self.tiers[i + 1].label,
                        "cascade: gate rejected, escalating"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        tier = %tier.label,
                        error = %e,
                        "cascade: tier failed, escalating"
                    );
                }
            }
        }
        unreachable!("cascade stream_complete exhausted all tiers without returning")
    }
}

/// Builder for [`CascadingProvider`].
pub struct CascadingProviderBuilder {
    tiers: Vec<CascadeTier>,
    gate: Option<Box<dyn ConfidenceGate>>,
}

impl CascadingProviderBuilder {
    /// Add a tier (cheapest first, most expensive last).
    pub fn add_tier(
        mut self,
        label: impl Into<String>,
        provider: impl LlmProvider + 'static,
    ) -> Self {
        self.tiers.push(CascadeTier {
            provider: Box::new(provider),
            label: label.into(),
        });
        self
    }

    /// Set the confidence gate. Defaults to [`HeuristicGate`] with default settings.
    pub fn gate(mut self, gate: impl ConfidenceGate + 'static) -> Self {
        self.gate = Some(Box::new(gate));
        self
    }

    /// Build the cascading provider.
    pub fn build(self) -> Result<CascadingProvider, Error> {
        if self.tiers.is_empty() {
            return Err(Error::Config(
                "CascadingProvider requires at least one tier".into(),
            ));
        }
        Ok(CascadingProvider {
            tiers: self.tiers,
            gate: self
                .gate
                .unwrap_or_else(|| Box::new(HeuristicGate::default())),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ContentBlock, Message, StopReason, TokenUsage};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    fn text_response(text: &str, output_tokens: u32) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                output_tokens,
                ..Default::default()
            },
            model: None,
        }
    }

    fn tool_response() -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "search".into(),
                input: json!({"q": "rust"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage {
                output_tokens: 20,
                ..Default::default()
            },
            model: None,
        }
    }

    fn max_tokens_response() -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "truncated...".into(),
            }],
            stop_reason: StopReason::MaxTokens,
            usage: TokenUsage {
                output_tokens: 100,
                ..Default::default()
            },
            model: None,
        }
    }

    fn test_request() -> CompletionRequest {
        CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("hello")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        }
    }

    // -- HeuristicGate tests --

    #[test]
    fn heuristic_gate_accepts_normal_response() {
        let gate = HeuristicGate::default();
        let req = test_request();
        let resp = text_response("Salut Pascal! Comment vas-tu?", 10);
        assert!(gate.accept(&req, &resp));
    }

    #[test]
    fn heuristic_gate_rejects_short_response() {
        let gate = HeuristicGate::default();
        let req = test_request();
        let resp = text_response("Hi", 2);
        assert!(!gate.accept(&req, &resp));
    }

    #[test]
    fn heuristic_gate_rejects_refusal_patterns() {
        let gate = HeuristicGate::default();
        let req = test_request();

        let patterns = [
            "I don't know the answer to that.",
            "I'm not sure about this topic.",
            "I cannot help with that request.",
            "I can't do that.",
            "I'm unable to assist with this.",
            "That is beyond my capabilities.",
            "I apologize, but I need more context.",
        ];
        for text in patterns {
            let resp = text_response(text, 20);
            assert!(!gate.accept(&req, &resp), "should reject: {text}");
        }
    }

    #[test]
    fn heuristic_gate_accepts_tool_calls() {
        let gate = HeuristicGate::default();
        let req = test_request();
        let resp = tool_response();
        assert!(gate.accept(&req, &resp));
    }

    #[test]
    fn heuristic_gate_rejects_max_tokens() {
        let gate = HeuristicGate::default();
        let req = test_request();
        let resp = max_tokens_response();
        assert!(!gate.accept(&req, &resp));
    }

    #[test]
    fn heuristic_gate_default_patterns() {
        let gate = HeuristicGate::default();
        assert_eq!(gate.min_output_tokens, 5);
        assert!(gate.accept_tool_calls);
        assert!(gate.escalate_on_max_tokens);
        assert!(!gate.refusal_patterns.is_empty());
        assert!(gate.refusal_patterns.len() >= 7);
    }

    #[test]
    fn heuristic_gate_case_insensitive_refusal() {
        let gate = HeuristicGate::default();
        let req = test_request();
        // "I DON'T KNOW" should match "I don't know"
        let resp = text_response("I DON'T KNOW about that", 10);
        assert!(!gate.accept(&req, &resp));
    }

    // -- Mock providers for CascadingProvider tests --

    struct FixedProvider {
        label: &'static str,
        response: Result<CompletionResponse, Error>,
        call_count: AtomicUsize,
    }

    impl FixedProvider {
        fn ok(label: &'static str, response: CompletionResponse) -> Self {
            Self {
                label,
                response: Ok(response),
                call_count: AtomicUsize::new(0),
            }
        }

        fn err(label: &'static str) -> Self {
            Self {
                label,
                response: Err(Error::Api {
                    status: 500,
                    message: "tier error".into(),
                }),
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl LlmProvider for FixedProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            match &self.response {
                Ok(r) => Ok(r.clone()),
                Err(e) => Err(Error::Api {
                    status: match e {
                        Error::Api { status, .. } => *status,
                        _ => 500,
                    },
                    message: format!("{} error", self.label),
                }),
            }
        }

        async fn stream_complete(
            &self,
            _request: CompletionRequest,
            on_text: &OnText,
        ) -> Result<CompletionResponse, Error> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            match &self.response {
                Ok(r) => {
                    let text = r.text();
                    if !text.is_empty() {
                        on_text(&text);
                    }
                    Ok(r.clone())
                }
                Err(_) => Err(Error::Api {
                    status: 500,
                    message: format!("{} error", self.label),
                }),
            }
        }

        fn model_name(&self) -> Option<&str> {
            Some(self.label)
        }
    }

    /// Gate that always accepts.
    struct AlwaysAccept;
    impl ConfidenceGate for AlwaysAccept {
        fn accept(&self, _req: &CompletionRequest, _resp: &CompletionResponse) -> bool {
            true
        }
    }

    /// Gate that always rejects.
    struct AlwaysReject;
    impl ConfidenceGate for AlwaysReject {
        fn accept(&self, _req: &CompletionRequest, _resp: &CompletionResponse) -> bool {
            false
        }
    }

    // -- CascadingProvider tests --

    #[tokio::test]
    async fn single_tier_delegates_directly() {
        let provider = CascadingProvider::builder()
            .add_tier(
                "haiku",
                FixedProvider::ok("haiku", text_response("hello", 10)),
            )
            .gate(AlwaysAccept)
            .build()
            .unwrap();

        let resp = LlmProvider::complete(&provider, test_request())
            .await
            .unwrap();
        assert_eq!(resp.text(), "hello");
        assert_eq!(resp.model.as_deref(), Some("haiku"));
    }

    #[tokio::test]
    async fn two_tier_accepts_cheap_when_gate_passes() {
        let provider = CascadingProvider::builder()
            .add_tier(
                "haiku",
                FixedProvider::ok("haiku", text_response("Salut!", 10)),
            )
            .add_tier(
                "sonnet",
                FixedProvider::ok("sonnet", text_response("expensive", 50)),
            )
            .gate(AlwaysAccept)
            .build()
            .unwrap();

        let resp = LlmProvider::complete(&provider, test_request())
            .await
            .unwrap();
        assert_eq!(resp.text(), "Salut!");
        assert_eq!(resp.model.as_deref(), Some("haiku"));
        // expensive provider was never called (we can't check this with the current
        // setup since we moved providers into tiers, but the response proves haiku was used)
    }

    #[tokio::test]
    async fn two_tier_escalates_when_gate_rejects() {
        let provider = CascadingProvider::builder()
            .add_tier(
                "haiku",
                FixedProvider::ok("haiku", text_response("dunno", 10)),
            )
            .add_tier(
                "sonnet",
                FixedProvider::ok("sonnet", text_response("great answer", 50)),
            )
            .gate(AlwaysReject)
            .build()
            .unwrap();

        let resp = LlmProvider::complete(&provider, test_request())
            .await
            .unwrap();
        // Gate rejected haiku, so sonnet should be used.
        // Final tier always accepts regardless of gate.
        assert_eq!(resp.text(), "great answer");
        assert_eq!(resp.model.as_deref(), Some("sonnet"));
    }

    #[tokio::test]
    async fn three_tier_skips_erroring_tier() {
        let provider = CascadingProvider::builder()
            .add_tier("haiku", FixedProvider::err("haiku"))
            .add_tier(
                "sonnet",
                FixedProvider::ok("sonnet", text_response("mid", 10)),
            )
            .add_tier(
                "opus",
                FixedProvider::ok("opus", text_response("expensive", 50)),
            )
            .gate(AlwaysAccept)
            .build()
            .unwrap();

        let resp = LlmProvider::complete(&provider, test_request())
            .await
            .unwrap();
        assert_eq!(resp.text(), "mid");
        assert_eq!(resp.model.as_deref(), Some("sonnet"));
    }

    #[tokio::test]
    async fn final_tier_always_accepts() {
        // AlwaysReject gate, but final tier should still be returned
        let provider = CascadingProvider::builder()
            .add_tier(
                "haiku",
                FixedProvider::ok("haiku", text_response("cheap", 10)),
            )
            .add_tier(
                "sonnet",
                FixedProvider::ok("sonnet", text_response("final", 50)),
            )
            .gate(AlwaysReject)
            .build()
            .unwrap();

        let resp = LlmProvider::complete(&provider, test_request())
            .await
            .unwrap();
        assert_eq!(resp.text(), "final");
        assert_eq!(resp.model.as_deref(), Some("sonnet"));
    }

    #[tokio::test]
    async fn stream_uses_complete_for_non_final_tiers() {
        // Track which method was called. We use a special provider that panics on
        // stream_complete for the cheap tier (non-final tiers should use complete()).
        struct CompleteOnlyProvider;
        impl LlmProvider for CompleteOnlyProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                Ok(text_response("cheap answer", 10))
            }
            async fn stream_complete(
                &self,
                _request: CompletionRequest,
                _on_text: &OnText,
            ) -> Result<CompletionResponse, Error> {
                panic!("non-final tier should not call stream_complete");
            }
        }

        let provider = CascadingProvider::builder()
            .add_tier("cheap", CompleteOnlyProvider)
            .add_tier(
                "expensive",
                FixedProvider::ok("expensive", text_response("expensive", 50)),
            )
            .gate(AlwaysAccept)
            .build()
            .unwrap();

        let on_text: &OnText = &|_| {};
        let resp = LlmProvider::stream_complete(&provider, test_request(), on_text)
            .await
            .unwrap();
        assert_eq!(resp.text(), "cheap answer");
    }

    #[tokio::test]
    async fn stream_emits_text_when_cheap_accepted() {
        let collected = Arc::new(Mutex::new(Vec::<String>::new()));
        let collected_clone = collected.clone();
        let on_text: &OnText = &move |text: &str| {
            collected_clone.lock().expect("lock").push(text.to_string());
        };

        let provider = CascadingProvider::builder()
            .add_tier(
                "cheap",
                FixedProvider::ok("cheap", text_response("hello world", 10)),
            )
            .add_tier(
                "expensive",
                FixedProvider::ok("expensive", text_response("expensive", 50)),
            )
            .gate(AlwaysAccept)
            .build()
            .unwrap();

        let resp = LlmProvider::stream_complete(&provider, test_request(), on_text)
            .await
            .unwrap();
        assert_eq!(resp.text(), "hello world");

        let texts = collected.lock().expect("lock");
        assert_eq!(*texts, vec!["hello world"]);
    }

    #[tokio::test]
    async fn stream_streams_final_tier() {
        // When gate rejects cheap tier, final tier should use stream_complete
        let streamed = Arc::new(Mutex::new(Vec::<String>::new()));
        let streamed_clone = streamed.clone();
        let on_text: &OnText = &move |text: &str| {
            streamed_clone.lock().expect("lock").push(text.to_string());
        };

        struct StreamingProvider;
        impl LlmProvider for StreamingProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                panic!("final tier with streaming should use stream_complete");
            }
            async fn stream_complete(
                &self,
                _request: CompletionRequest,
                on_text: &OnText,
            ) -> Result<CompletionResponse, Error> {
                on_text("streamed ");
                on_text("response");
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "streamed response".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage {
                        output_tokens: 20,
                        ..Default::default()
                    },
                    model: None,
                })
            }
        }

        let provider = CascadingProvider::builder()
            .add_tier(
                "cheap",
                FixedProvider::ok("cheap", text_response("dunno", 10)),
            )
            .add_tier("expensive", StreamingProvider)
            .gate(AlwaysReject)
            .build()
            .unwrap();

        let resp = LlmProvider::stream_complete(&provider, test_request(), on_text)
            .await
            .unwrap();
        assert_eq!(resp.text(), "streamed response");
        assert_eq!(resp.model.as_deref(), Some("expensive"));

        let texts = streamed.lock().expect("lock");
        assert_eq!(*texts, vec!["streamed ", "response"]);
    }

    #[tokio::test]
    async fn response_model_set_to_accepting_tier() {
        let provider = CascadingProvider::builder()
            .add_tier("haiku", FixedProvider::err("haiku"))
            .add_tier(
                "sonnet",
                FixedProvider::ok("sonnet", text_response("answer", 10)),
            )
            .gate(AlwaysAccept)
            .build()
            .unwrap();

        let resp = LlmProvider::complete(&provider, test_request())
            .await
            .unwrap();
        assert_eq!(resp.model.as_deref(), Some("sonnet"));
    }

    #[test]
    fn builder_rejects_zero_tiers() {
        let result = CascadingProvider::builder().gate(AlwaysAccept).build();
        assert!(result.is_err());
    }

    #[test]
    fn cascading_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CascadingProvider>();
    }

    #[test]
    fn builder_defaults_to_heuristic_gate() {
        let provider = CascadingProvider::builder()
            .add_tier("haiku", FixedProvider::ok("haiku", text_response("hi", 10)))
            .build()
            .unwrap();
        // Should build without explicit gate
        assert_eq!(LlmProvider::model_name(&provider), Some("haiku"));
    }

    #[tokio::test]
    async fn single_tier_streams_directly() {
        // Single tier should use stream_complete, not complete
        struct StreamOnlyProvider;
        impl LlmProvider for StreamOnlyProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                panic!("single tier should stream directly");
            }
            async fn stream_complete(
                &self,
                _request: CompletionRequest,
                on_text: &OnText,
            ) -> Result<CompletionResponse, Error> {
                on_text("streamed");
                Ok(text_response("streamed", 10))
            }
        }

        let provider = CascadingProvider::builder()
            .add_tier("only", StreamOnlyProvider)
            .gate(AlwaysAccept)
            .build()
            .unwrap();

        let on_text: &OnText = &|_| {};
        let resp = LlmProvider::stream_complete(&provider, test_request(), on_text)
            .await
            .unwrap();
        assert_eq!(resp.text(), "streamed");
        assert_eq!(resp.model.as_deref(), Some("only"));
    }

    #[tokio::test]
    async fn all_tiers_error_returns_last_error() {
        let provider = CascadingProvider::builder()
            .add_tier("tier1", FixedProvider::err("tier1"))
            .add_tier("tier2", FixedProvider::err("tier2"))
            .gate(AlwaysAccept)
            .build()
            .unwrap();

        let err = LlmProvider::complete(&provider, test_request())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("tier2"), "error: {err}");
    }

    #[tokio::test]
    async fn heuristic_gate_integration_with_cascade() {
        // Cheap gives short answer → gate rejects → escalates to expensive
        let provider = CascadingProvider::builder()
            .add_tier("haiku", FixedProvider::ok("haiku", text_response("Hi", 2)))
            .add_tier(
                "sonnet",
                FixedProvider::ok("sonnet", text_response("detailed answer here", 30)),
            )
            // Default HeuristicGate with min_output_tokens=5
            .build()
            .unwrap();

        let resp = LlmProvider::complete(&provider, test_request())
            .await
            .unwrap();
        assert_eq!(resp.text(), "detailed answer here");
        assert_eq!(resp.model.as_deref(), Some("sonnet"));
    }
}
