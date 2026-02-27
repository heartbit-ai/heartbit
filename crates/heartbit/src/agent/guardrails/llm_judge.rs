//! LLM-as-Judge guardrail.
//!
//! Sends LLM responses and (optionally) tool call inputs to a cheap judge
//! model for safety evaluation. Closes the gap vs Google ADK's
//! Gemini-as-Judge pattern.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::agent::guardrail::{GuardAction, Guardrail, GuardrailMeta};
use crate::error::Error;
use crate::llm::types::{CompletionRequest, CompletionResponse, ContentBlock, Message, ToolCall};
use crate::llm::{BoxedProvider, LlmProvider};

/// Verdict returned by the judge LLM.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict {
    Safe,
    Unsafe { reason: String },
    Warn { reason: String },
}

/// Parse a judge LLM response for a verdict line.
///
/// Looks for `VERDICT: SAFE`, `VERDICT: UNSAFE: <reason>`, or
/// `VERDICT: WARN: <reason>`. If no verdict line is found, returns `None`
/// (caller should fail-open).
fn parse_verdict(text: &str) -> Option<Verdict> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("VERDICT:") {
            let rest = rest.trim();
            if rest.eq_ignore_ascii_case("SAFE") {
                return Some(Verdict::Safe);
            }
            if let Some(reason) = rest
                .strip_prefix("UNSAFE:")
                .or_else(|| rest.strip_prefix("unsafe:"))
            {
                let reason = reason.trim();
                return Some(Verdict::Unsafe {
                    reason: if reason.is_empty() {
                        "unspecified".to_string()
                    } else {
                        reason.to_string()
                    },
                });
            }
            if let Some(reason) = rest
                .strip_prefix("WARN:")
                .or_else(|| rest.strip_prefix("warn:"))
            {
                let reason = reason.trim();
                return Some(Verdict::Warn {
                    reason: if reason.is_empty() {
                        "unspecified".to_string()
                    } else {
                        reason.to_string()
                    },
                });
            }
            // VERDICT: line found but not recognized — continue scanning
        }
    }
    None
}

/// Convert a verdict to a `GuardAction`.
fn verdict_to_action(verdict: Option<Verdict>) -> GuardAction {
    match verdict {
        Some(Verdict::Safe) | None => GuardAction::Allow,
        Some(Verdict::Unsafe { reason }) => GuardAction::deny(reason),
        Some(Verdict::Warn { reason }) => GuardAction::warn(reason),
    }
}

/// Default system prompt template for the judge LLM.
fn default_system_prompt(criteria: &[String]) -> String {
    let criteria_list = criteria
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {}", i + 1, c))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are a safety judge. Evaluate the following content against these criteria:\n\
         \n\
         {criteria_list}\n\
         \n\
         Respond with EXACTLY one verdict line in this format:\n\
         - VERDICT: SAFE (if content passes all criteria)\n\
         - VERDICT: UNSAFE: <brief reason> (if content clearly violates criteria)\n\
         - VERDICT: WARN: <brief reason> (if content is borderline or suspicious)\n\
         \n\
         Be concise. Output only the verdict line."
    )
}

/// LLM-as-Judge guardrail.
///
/// Sends content to a cheap LLM (e.g., Haiku, Gemini Flash) for safety
/// evaluation. The judge provider is separate from the main agent's LLM.
///
/// **Hooks implemented:**
/// - `post_llm`: Evaluates the LLM's response text for safety.
/// - `pre_tool`: Optionally evaluates tool call inputs (when `evaluate_tool_inputs` is true).
///
/// **Fail-open:** On judge timeout or error, the guardrail defaults to `Allow`
/// with a `tracing::warn!` log.
pub struct LlmJudgeGuardrail {
    judge_provider: Arc<BoxedProvider>,
    system_prompt: String,
    timeout: Duration,
    evaluate_tool_inputs: bool,
    max_judge_tokens: u32,
}

impl std::fmt::Debug for LlmJudgeGuardrail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmJudgeGuardrail")
            .field("system_prompt", &self.system_prompt)
            .field("timeout", &self.timeout)
            .field("evaluate_tool_inputs", &self.evaluate_tool_inputs)
            .field("max_judge_tokens", &self.max_judge_tokens)
            .finish_non_exhaustive()
    }
}

impl LlmJudgeGuardrail {
    /// Create a builder for `LlmJudgeGuardrail`.
    pub fn builder(judge_provider: Arc<BoxedProvider>) -> LlmJudgeGuardrailBuilder {
        LlmJudgeGuardrailBuilder {
            judge_provider,
            criteria: Vec::new(),
            timeout: Duration::from_secs(10),
            evaluate_tool_inputs: false,
            max_judge_tokens: 256,
            custom_system_prompt: None,
        }
    }

    /// Call the judge LLM with the given content and return a `GuardAction`.
    ///
    /// On timeout or error, returns `Allow` (fail-open).
    async fn judge(&self, content: &str) -> GuardAction {
        let request = CompletionRequest {
            system: self.system_prompt.clone(),
            messages: vec![Message::user(content)],
            tools: vec![],
            max_tokens: self.max_judge_tokens,
            tool_choice: None,
            reasoning_effort: None,
        };

        let result = tokio::time::timeout(
            self.timeout,
            LlmProvider::complete(self.judge_provider.as_ref(), request),
        )
        .await;

        match result {
            Ok(Ok(response)) => {
                let text: String = response
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                verdict_to_action(parse_verdict(&text))
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "LLM judge call failed, allowing (fail-open)");
                GuardAction::Allow
            }
            Err(_elapsed) => {
                tracing::warn!("LLM judge timed out, allowing (fail-open)");
                GuardAction::Allow
            }
        }
    }
}

impl GuardrailMeta for LlmJudgeGuardrail {
    fn name(&self) -> &str {
        "llm_judge"
    }
}

impl Guardrail for LlmJudgeGuardrail {
    fn post_llm(
        &self,
        response: &CompletionResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        let text: String = response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();

        Box::pin(async move {
            if text.is_empty() {
                return Ok(GuardAction::Allow);
            }
            Ok(self.judge(&text).await)
        })
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        if !self.evaluate_tool_inputs {
            return Box::pin(async { Ok(GuardAction::Allow) });
        }

        let content = format!(
            "Tool: {}\nInput: {}",
            call.name,
            serde_json::to_string(&call.input).unwrap_or_else(|_| call.input.to_string())
        );

        Box::pin(async move { Ok(self.judge(&content).await) })
    }
}

/// Builder for [`LlmJudgeGuardrail`].
pub struct LlmJudgeGuardrailBuilder {
    judge_provider: Arc<BoxedProvider>,
    criteria: Vec<String>,
    timeout: Duration,
    evaluate_tool_inputs: bool,
    max_judge_tokens: u32,
    custom_system_prompt: Option<String>,
}

impl LlmJudgeGuardrailBuilder {
    /// Add a safety criterion to evaluate against.
    pub fn criterion(mut self, criterion: impl Into<String>) -> Self {
        self.criteria.push(criterion.into());
        self
    }

    /// Add multiple criteria at once.
    pub fn criteria(mut self, criteria: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.criteria.extend(criteria.into_iter().map(Into::into));
        self
    }

    /// Set the timeout for judge LLM calls (default: 10 seconds).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable evaluation of tool call inputs via `pre_tool` (default: false).
    pub fn evaluate_tool_inputs(mut self, evaluate: bool) -> Self {
        self.evaluate_tool_inputs = evaluate;
        self
    }

    /// Set the max tokens for judge responses (default: 256).
    pub fn max_judge_tokens(mut self, max_tokens: u32) -> Self {
        self.max_judge_tokens = max_tokens;
        self
    }

    /// Set a custom system prompt (overrides the default criteria-based prompt).
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.custom_system_prompt = Some(prompt.into());
        self
    }

    /// Build the guardrail.
    ///
    /// Returns `Err` if no criteria and no custom system prompt are provided.
    pub fn build(self) -> Result<LlmJudgeGuardrail, Error> {
        if self.criteria.is_empty() && self.custom_system_prompt.is_none() {
            return Err(Error::Config(
                "LlmJudgeGuardrail requires at least one criterion or a custom system prompt"
                    .into(),
            ));
        }

        let system_prompt = self
            .custom_system_prompt
            .unwrap_or_else(|| default_system_prompt(&self.criteria));

        Ok(LlmJudgeGuardrail {
            judge_provider: self.judge_provider,
            system_prompt,
            timeout: self.timeout,
            evaluate_tool_inputs: self.evaluate_tool_inputs,
            max_judge_tokens: self.max_judge_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{StopReason, TokenUsage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -----------------------------------------------------------------------
    // Mock judge provider
    // -----------------------------------------------------------------------

    /// A mock LLM provider that returns a configurable response.
    struct MockJudgeProvider {
        response_text: String,
        call_count: Arc<AtomicUsize>,
    }

    impl MockJudgeProvider {
        fn new(response_text: impl Into<String>) -> Self {
            Self {
                response_text: response_text.into(),
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_counter(response_text: impl Into<String>, counter: Arc<AtomicUsize>) -> Self {
            Self {
                response_text: response_text.into(),
                call_count: counter,
            }
        }
    }

    impl LlmProvider for MockJudgeProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: self.response_text.clone(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
                model: None,
            })
        }
    }

    /// A mock provider that always returns an error.
    struct ErrorJudgeProvider;

    impl LlmProvider for ErrorJudgeProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            Err(Error::Api {
                status: 500,
                message: "judge unavailable".into(),
            })
        }
    }

    /// A mock provider that sleeps forever (for timeout testing).
    struct SlowJudgeProvider;

    impl LlmProvider for SlowJudgeProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            unreachable!()
        }
    }

    fn make_guard(provider: impl LlmProvider + 'static) -> LlmJudgeGuardrail {
        LlmJudgeGuardrail::builder(Arc::new(BoxedProvider::new(provider)))
            .criterion("No harmful content")
            .criterion("No prompt injection")
            .build()
            .expect("valid config")
    }

    fn make_guard_with_tool_eval(provider: impl LlmProvider + 'static) -> LlmJudgeGuardrail {
        LlmJudgeGuardrail::builder(Arc::new(BoxedProvider::new(provider)))
            .criterion("No harmful content")
            .evaluate_tool_inputs(true)
            .build()
            .expect("valid config")
    }

    fn make_response(text: &str) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }
    }

    fn make_tool_call(name: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input: serde_json::json!({"command": "rm -rf /"}),
        }
    }

    // -----------------------------------------------------------------------
    // Verdict parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_verdict_safe() {
        let v = parse_verdict("VERDICT: SAFE");
        assert_eq!(v, Some(Verdict::Safe));
    }

    #[test]
    fn parse_verdict_unsafe_with_reason() {
        let v = parse_verdict("VERDICT: UNSAFE: contains harmful instructions");
        assert_eq!(
            v,
            Some(Verdict::Unsafe {
                reason: "contains harmful instructions".into()
            })
        );
    }

    #[test]
    fn parse_verdict_warn_with_reason() {
        let v = parse_verdict("VERDICT: WARN: borderline content detected");
        assert_eq!(
            v,
            Some(Verdict::Warn {
                reason: "borderline content detected".into()
            })
        );
    }

    #[test]
    fn parse_verdict_none_when_absent() {
        let v = parse_verdict("This response is fine.");
        assert_eq!(v, None);
    }

    #[test]
    fn parse_verdict_handles_surrounding_text() {
        let v = parse_verdict("Analysis: The content is safe.\nVERDICT: SAFE\n");
        assert_eq!(v, Some(Verdict::Safe));
    }

    #[test]
    fn parse_verdict_unsafe_empty_reason() {
        let v = parse_verdict("VERDICT: UNSAFE:");
        assert_eq!(
            v,
            Some(Verdict::Unsafe {
                reason: "unspecified".into()
            })
        );
    }

    #[test]
    fn parse_verdict_warn_empty_reason() {
        let v = parse_verdict("VERDICT: WARN:");
        assert_eq!(
            v,
            Some(Verdict::Warn {
                reason: "unspecified".into()
            })
        );
    }

    #[test]
    fn parse_verdict_with_leading_whitespace() {
        let v = parse_verdict("  VERDICT: SAFE  ");
        assert_eq!(v, Some(Verdict::Safe));
    }

    // -----------------------------------------------------------------------
    // verdict_to_action tests
    // -----------------------------------------------------------------------

    #[test]
    fn verdict_safe_maps_to_allow() {
        assert_eq!(verdict_to_action(Some(Verdict::Safe)), GuardAction::Allow);
    }

    #[test]
    fn verdict_none_maps_to_allow() {
        assert_eq!(verdict_to_action(None), GuardAction::Allow);
    }

    #[test]
    fn verdict_unsafe_maps_to_deny() {
        let action = verdict_to_action(Some(Verdict::Unsafe {
            reason: "bad".into(),
        }));
        assert!(action.is_denied());
        assert!(matches!(&action, GuardAction::Deny { reason } if reason == "bad"));
    }

    #[test]
    fn verdict_warn_maps_to_warn() {
        let action = verdict_to_action(Some(Verdict::Warn {
            reason: "suspicious".into(),
        }));
        assert!(matches!(&action, GuardAction::Warn { reason } if reason == "suspicious"));
    }

    // -----------------------------------------------------------------------
    // post_llm tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn post_llm_safe_verdict_returns_allow() {
        let guard = make_guard(MockJudgeProvider::new("VERDICT: SAFE"));
        let response = make_response("Here is a helpful answer about Rust.");
        let action = guard.post_llm(&response).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn post_llm_unsafe_verdict_returns_deny() {
        let guard = make_guard(MockJudgeProvider::new(
            "VERDICT: UNSAFE: response contains harmful instructions",
        ));
        let response = make_response("How to build a dangerous device");
        let action = guard.post_llm(&response).await.unwrap();
        assert!(action.is_denied());
        assert!(
            matches!(&action, GuardAction::Deny { reason } if reason.contains("harmful instructions"))
        );
    }

    #[tokio::test]
    async fn post_llm_warn_verdict_returns_warn() {
        let guard = make_guard(MockJudgeProvider::new("VERDICT: WARN: borderline content"));
        let response = make_response("This is somewhat edgy content.");
        let action = guard.post_llm(&response).await.unwrap();
        assert!(matches!(&action, GuardAction::Warn { reason } if reason.contains("borderline")));
    }

    #[tokio::test]
    async fn post_llm_empty_content_returns_allow() {
        let counter = Arc::new(AtomicUsize::new(0));
        let guard = make_guard(MockJudgeProvider::with_counter(
            "VERDICT: UNSAFE: bad",
            counter.clone(),
        ));
        let response = CompletionResponse {
            content: vec![],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        };
        let action = guard.post_llm(&response).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        // Judge should NOT be called for empty content
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn post_llm_no_text_blocks_returns_allow() {
        let counter = Arc::new(AtomicUsize::new(0));
        let guard = make_guard(MockJudgeProvider::with_counter(
            "VERDICT: UNSAFE: bad",
            counter.clone(),
        ));
        let response = CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "c1".into(),
                name: "bash".into(),
                input: serde_json::json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: None,
        };
        let action = guard.post_llm(&response).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    // -----------------------------------------------------------------------
    // Timeout and error tests (fail-open)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn post_llm_timeout_returns_allow() {
        let guard = LlmJudgeGuardrail::builder(Arc::new(BoxedProvider::new(SlowJudgeProvider)))
            .criterion("No harmful content")
            .timeout(Duration::from_millis(50))
            .build()
            .expect("valid config");

        let response = make_response("Some content to evaluate.");
        let action = guard.post_llm(&response).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn post_llm_judge_error_returns_allow() {
        let guard = make_guard(ErrorJudgeProvider);
        let response = make_response("Some content to evaluate.");
        let action = guard.post_llm(&response).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    // -----------------------------------------------------------------------
    // pre_tool tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn pre_tool_disabled_returns_allow() {
        let counter = Arc::new(AtomicUsize::new(0));
        let guard = make_guard(MockJudgeProvider::with_counter(
            "VERDICT: UNSAFE: dangerous",
            counter.clone(),
        ));
        let call = make_tool_call("bash");
        let action = guard.pre_tool(&call).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        // Judge should NOT be called
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn pre_tool_enabled_evaluates_tool_input() {
        let guard = make_guard_with_tool_eval(MockJudgeProvider::new(
            "VERDICT: UNSAFE: destructive command",
        ));
        let call = make_tool_call("bash");
        let action = guard.pre_tool(&call).await.unwrap();
        assert!(action.is_denied());
        assert!(matches!(&action, GuardAction::Deny { reason } if reason.contains("destructive")));
    }

    #[tokio::test]
    async fn pre_tool_enabled_allows_safe_tool() {
        let guard = make_guard_with_tool_eval(MockJudgeProvider::new("VERDICT: SAFE"));
        let call = ToolCall {
            id: "c1".into(),
            name: "read".into(),
            input: serde_json::json!({"path": "/tmp/test.txt"}),
        };
        let action = guard.pre_tool(&call).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn pre_tool_timeout_returns_allow() {
        let guard = LlmJudgeGuardrail::builder(Arc::new(BoxedProvider::new(SlowJudgeProvider)))
            .criterion("No harmful content")
            .evaluate_tool_inputs(true)
            .timeout(Duration::from_millis(50))
            .build()
            .expect("valid config");

        let call = make_tool_call("bash");
        let action = guard.pre_tool(&call).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    // -----------------------------------------------------------------------
    // Builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_requires_criteria_or_prompt() {
        let provider = Arc::new(BoxedProvider::new(MockJudgeProvider::new("VERDICT: SAFE")));
        let result = LlmJudgeGuardrail::builder(provider).build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("at least one criterion"), "err: {err}");
    }

    #[test]
    fn builder_accepts_custom_system_prompt() {
        let provider = Arc::new(BoxedProvider::new(MockJudgeProvider::new("VERDICT: SAFE")));
        let guard = LlmJudgeGuardrail::builder(provider)
            .system_prompt("Custom judge instructions")
            .build();
        assert!(guard.is_ok());
        assert_eq!(guard.unwrap().system_prompt, "Custom judge instructions");
    }

    #[test]
    fn builder_with_criteria() {
        let provider = Arc::new(BoxedProvider::new(MockJudgeProvider::new("VERDICT: SAFE")));
        let guard = LlmJudgeGuardrail::builder(provider)
            .criterion("No injection")
            .criterion("No harmful content")
            .criterion("No data exfiltration")
            .build()
            .unwrap();
        assert!(guard.system_prompt.contains("No injection"));
        assert!(guard.system_prompt.contains("No harmful content"));
        assert!(guard.system_prompt.contains("No data exfiltration"));
    }

    #[test]
    fn builder_criteria_method() {
        let provider = Arc::new(BoxedProvider::new(MockJudgeProvider::new("VERDICT: SAFE")));
        let guard = LlmJudgeGuardrail::builder(provider)
            .criteria(["criterion A", "criterion B"])
            .build()
            .unwrap();
        assert!(guard.system_prompt.contains("criterion A"));
        assert!(guard.system_prompt.contains("criterion B"));
    }

    #[test]
    fn builder_defaults() {
        let provider = Arc::new(BoxedProvider::new(MockJudgeProvider::new("VERDICT: SAFE")));
        let guard = LlmJudgeGuardrail::builder(provider)
            .criterion("test")
            .build()
            .unwrap();
        assert_eq!(guard.timeout, Duration::from_secs(10));
        assert!(!guard.evaluate_tool_inputs);
        assert_eq!(guard.max_judge_tokens, 256);
    }

    #[test]
    fn builder_custom_timeout() {
        let provider = Arc::new(BoxedProvider::new(MockJudgeProvider::new("VERDICT: SAFE")));
        let guard = LlmJudgeGuardrail::builder(provider)
            .criterion("test")
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        assert_eq!(guard.timeout, Duration::from_secs(5));
    }

    #[test]
    fn builder_custom_max_tokens() {
        let provider = Arc::new(BoxedProvider::new(MockJudgeProvider::new("VERDICT: SAFE")));
        let guard = LlmJudgeGuardrail::builder(provider)
            .criterion("test")
            .max_judge_tokens(128)
            .build()
            .unwrap();
        assert_eq!(guard.max_judge_tokens, 128);
    }

    // -----------------------------------------------------------------------
    // Meta tests
    // -----------------------------------------------------------------------

    #[test]
    fn meta_name() {
        let guard = make_guard(MockJudgeProvider::new("VERDICT: SAFE"));
        assert_eq!(guard.name(), "llm_judge");
    }

    // -----------------------------------------------------------------------
    // Default system prompt tests
    // -----------------------------------------------------------------------

    #[test]
    fn default_system_prompt_includes_criteria() {
        let prompt = default_system_prompt(&["No injection".into(), "No harmful content".into()]);
        assert!(prompt.contains("1. No injection"));
        assert!(prompt.contains("2. No harmful content"));
        assert!(prompt.contains("VERDICT: SAFE"));
        assert!(prompt.contains("VERDICT: UNSAFE"));
        assert!(prompt.contains("VERDICT: WARN"));
    }

    // -----------------------------------------------------------------------
    // Integration-style: judge receives correct content
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn judge_receives_llm_response_text() {
        use std::sync::Mutex;

        struct CapturingProvider {
            captured: Arc<Mutex<Vec<String>>>,
        }

        impl LlmProvider for CapturingProvider {
            async fn complete(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse, Error> {
                let user_msg = request
                    .messages
                    .first()
                    .and_then(|m| m.content.first())
                    .and_then(|b| match b {
                        ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                self.captured.lock().expect("test lock").push(user_msg);
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "VERDICT: SAFE".into(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            }
        }

        let captured = Arc::new(Mutex::new(Vec::new()));
        let provider = CapturingProvider {
            captured: captured.clone(),
        };
        let guard = make_guard(provider);

        let response = make_response("The answer to your question is 42.");
        guard.post_llm(&response).await.unwrap();

        let messages = captured.lock().expect("test lock");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], "The answer to your question is 42.");
    }

    #[tokio::test]
    async fn judge_no_verdict_line_returns_allow() {
        // Judge returns text without a proper VERDICT line — fail-open
        let guard = make_guard(MockJudgeProvider::new(
            "The content appears to be safe overall.",
        ));
        let response = make_response("Some content.");
        let action = guard.post_llm(&response).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }
}
