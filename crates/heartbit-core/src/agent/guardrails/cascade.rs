//! Cheap → expensive cascading guardrail.
//!
//! The 2026 production guardrail pattern (Anthropic Constitutional
//! Classifiers++, arXiv 2601.04603, ~40× cost reduction; Qwen3Guard-Stream,
//! arXiv 2510.14276) is a two-stage cascade: a **cheap screen** runs on every
//! response and **short-circuits to Allow** when nothing looks suspicious, only
//! **escalating to an expensive** check (an LLM judge) on the small fraction of
//! traffic the screen flags.
//!
//! [`CascadingGuardrail`] wraps any inner [`Guardrail`] (e.g.
//! [`LlmJudgeGuardrail`](super::LlmJudgeGuardrail)) behind a cheap text predicate:
//! the inner guard's `post_llm` / `pre_tool` run ONLY when the screen fires,
//! turning a per-response LLM-judge cost into a per-*flagged*-response cost.
//!
//! This differs from [`ConditionalGuardrail`](super::ConditionalGuardrail),
//! which gates the *tool* hooks by tool NAME and always runs the inner `post_llm`
//! — here the gate is a content screen over the response text / tool input.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::guardrail::{GuardAction, Guardrail};
use crate::error::Error;
use crate::llm::types::{CompletionResponse, ToolCall};
use crate::tool::ToolOutput;

/// A cheap content screen: returns `true` when the text is suspicious enough to
/// escalate to the expensive inner guardrail.
pub type CheapScreen = dyn Fn(&str) -> bool + Send + Sync;

/// Runs `inner` only when a cheap `screen` flags the content (cheap → expensive
/// cascade). See the module docs.
pub struct CascadingGuardrail {
    inner: Arc<dyn Guardrail>,
    screen: Arc<CheapScreen>,
}

impl CascadingGuardrail {
    /// Wrap `inner`, escalating to it only when `screen(text)` is `true`.
    pub fn new(inner: Arc<dyn Guardrail>, screen: Arc<CheapScreen>) -> Self {
        Self { inner, screen }
    }

    /// Convenience: escalate when the (lowercased) text contains any of `needles`.
    pub fn with_keywords(inner: Arc<dyn Guardrail>, needles: Vec<String>) -> Self {
        let screen: Arc<CheapScreen> = Arc::new(move |text: &str| {
            let lower = text.to_lowercase();
            needles.iter().any(|n| lower.contains(&n.to_lowercase()))
        });
        Self::new(inner, screen)
    }
}

impl Guardrail for CascadingGuardrail {
    fn name(&self) -> &str {
        "cascading"
    }

    fn post_llm(
        &self,
        response: &mut CompletionResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        // Cheap screen on the response text. Clean → skip the expensive inner.
        if !(self.screen)(&response.text()) {
            return Box::pin(async { Ok(GuardAction::Allow) });
        }
        // Flagged → delegate to the expensive inner guard (its own post_llm
        // applies any mutations synchronously per the trait contract).
        self.inner.post_llm(response)
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        let screened = serde_json::to_string(&call.input).unwrap_or_default();
        if !(self.screen)(&screened) {
            return Box::pin(async { Ok(GuardAction::Allow) });
        }
        self.inner.pre_tool(call)
    }

    fn post_tool(
        &self,
        call: &ToolCall,
        output: &mut ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        if !(self.screen)(&output.content) {
            return Box::pin(async { Ok(()) });
        }
        self.inner.post_tool(call, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ContentBlock, StopReason, TokenUsage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Inner guard that Denies and counts how many times it was consulted.
    struct CountingDeny {
        calls: Arc<AtomicUsize>,
    }
    impl Guardrail for CountingDeny {
        fn post_llm(
            &self,
            _response: &mut CompletionResponse,
        ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Ok(GuardAction::Deny {
                    reason: "expensive judge denied".into(),
                })
            })
        }
        fn pre_tool(
            &self,
            _call: &ToolCall,
        ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Ok(GuardAction::Deny {
                    reason: "expensive tool judge denied".into(),
                })
            })
        }
        fn post_tool(
            &self,
            _call: &ToolCall,
            output: &mut ToolOutput,
        ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            output.content.push_str(" [judged]");
            Box::pin(async { Ok(()) })
        }
    }

    fn tool_call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: "bash".into(),
            input,
        }
    }

    fn text_response(text: &str) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            stop_reason: StopReason::EndTurn,
            reasoning: None,
            usage: TokenUsage::default(),
            model: None,
        }
    }

    #[tokio::test]
    async fn clean_text_skips_the_expensive_inner() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(CountingDeny {
            calls: Arc::clone(&calls),
        });
        let guard = CascadingGuardrail::with_keywords(inner, vec!["password".into()]);
        let mut resp = text_response("Here is a normal helpful answer.");
        let action = guard.post_llm(&mut resp).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "the expensive judge must NOT run on clean text"
        );
    }

    #[tokio::test]
    async fn flagged_text_escalates_to_the_inner() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(CountingDeny {
            calls: Arc::clone(&calls),
        });
        let guard = CascadingGuardrail::with_keywords(inner, vec!["password".into()]);
        let mut resp = text_response("Sure, the admin PASSWORD is hunter2.");
        let action = guard.post_llm(&mut resp).await.unwrap();
        assert!(matches!(action, GuardAction::Deny { .. }));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "flagged text must escalate to the expensive judge exactly once"
        );
    }

    #[tokio::test]
    async fn clean_tool_input_skips_inner_pre_tool() {
        let calls = Arc::new(AtomicUsize::new(0));
        let guard = CascadingGuardrail::with_keywords(
            Arc::new(CountingDeny {
                calls: Arc::clone(&calls),
            }),
            vec!["rm -rf".into()],
        );
        let action = guard
            .pre_tool(&tool_call(serde_json::json!({"command": "ls -la"})))
            .await
            .unwrap();
        assert_eq!(action, GuardAction::Allow);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "clean input must not escalate"
        );
    }

    #[tokio::test]
    async fn flagged_tool_input_escalates_pre_tool() {
        let calls = Arc::new(AtomicUsize::new(0));
        let guard = CascadingGuardrail::with_keywords(
            Arc::new(CountingDeny {
                calls: Arc::clone(&calls),
            }),
            vec!["rm -rf".into()],
        );
        let action = guard
            .pre_tool(&tool_call(serde_json::json!({"command": "rm -rf /"})))
            .await
            .unwrap();
        assert!(matches!(action, GuardAction::Deny { .. }));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "flagged input escalates once"
        );
    }

    #[tokio::test]
    async fn clean_tool_output_skips_inner_post_tool() {
        let calls = Arc::new(AtomicUsize::new(0));
        let guard = CascadingGuardrail::with_keywords(
            Arc::new(CountingDeny {
                calls: Arc::clone(&calls),
            }),
            vec!["secret".into()],
        );
        let mut out = ToolOutput::success("ordinary output");
        guard
            .post_tool(&tool_call(serde_json::json!({})), &mut out)
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "clean output must not escalate"
        );
        assert_eq!(out.content, "ordinary output", "clean output is untouched");
    }

    #[tokio::test]
    async fn flagged_tool_output_escalates_post_tool() {
        let calls = Arc::new(AtomicUsize::new(0));
        let guard = CascadingGuardrail::with_keywords(
            Arc::new(CountingDeny {
                calls: Arc::clone(&calls),
            }),
            vec!["secret".into()],
        );
        let mut out = ToolOutput::success("the secret is 42");
        guard
            .post_tool(&tool_call(serde_json::json!({})), &mut out)
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "flagged output escalates once"
        );
        assert!(
            out.content.contains("[judged]"),
            "the inner post_tool ran and mutated the output: {}",
            out.content
        );
    }

    #[test]
    fn name_is_cascading() {
        let guard = CascadingGuardrail::with_keywords(
            Arc::new(CountingDeny {
                calls: Arc::new(AtomicUsize::new(0)),
            }),
            vec!["x".into()],
        );
        assert_eq!(guard.name(), "cascading");
    }
}
