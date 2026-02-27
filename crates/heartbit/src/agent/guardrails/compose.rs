//! Composition operators for guardrails.
//!
//! - [`GuardrailChain`]: Ordered pipeline — first `Deny` wins.
//! - [`WarnToDeny`]: Graduated containment — N consecutive `Warn` → `Deny`.
//! - [`ConditionalGuardrail`]: Predicate-gated — only runs when condition is true.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::agent::guardrail::{GuardAction, Guardrail, GuardrailMeta};
use crate::error::Error;
use crate::llm::types::{CompletionRequest, CompletionResponse, ToolCall};
use crate::tool::ToolOutput;

// ---------------------------------------------------------------------------
// GuardrailChain
// ---------------------------------------------------------------------------

/// Ordered pipeline of guardrails — first `Deny` wins.
///
/// Equivalent to the default `Vec<Arc<dyn Guardrail>>` behavior in the agent
/// loop, but wrapped as a single `Guardrail` for nested composition.
///
/// **Implementation note**: The `Guardrail` trait's lifetime elision ties the
/// returned future to `&self` only (not to reference parameters like `call`
/// or `response`). This means inner guardrails perform their work (including
/// mutations) synchronously during the call, and the returned futures only
/// carry no-op cleanup. We eagerly evaluate all inner guardrails and collect
/// their futures for awaiting.
pub struct GuardrailChain {
    guardrails: Vec<Arc<dyn Guardrail>>,
}

impl GuardrailChain {
    pub fn new(guardrails: Vec<Arc<dyn Guardrail>>) -> Self {
        Self { guardrails }
    }
}

impl GuardrailMeta for GuardrailChain {
    fn name(&self) -> &str {
        "chain"
    }
}

impl Guardrail for GuardrailChain {
    fn pre_llm(
        &self,
        request: &mut CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        // Eagerly call each guardrail's pre_llm (synchronous mutations run now).
        let futs: Vec<_> = self.guardrails.iter().map(|g| g.pre_llm(request)).collect();
        Box::pin(async move {
            for fut in futs {
                fut.await?;
            }
            Ok(())
        })
    }

    fn post_llm(
        &self,
        response: &CompletionResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        // Eagerly call each guardrail's post_llm.
        let futs: Vec<_> = self
            .guardrails
            .iter()
            .map(|g| g.post_llm(response))
            .collect();
        Box::pin(async move {
            let mut worst = GuardAction::Allow;
            for fut in futs {
                let action = fut.await?;
                if action.is_denied() {
                    return Ok(action);
                }
                // Escalate Allow → Warn (keep first Warn reason)
                if matches!(action, GuardAction::Warn { .. }) && matches!(worst, GuardAction::Allow)
                {
                    worst = action;
                }
            }
            Ok(worst)
        })
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        // Eagerly call each guardrail's pre_tool.
        let futs: Vec<_> = self.guardrails.iter().map(|g| g.pre_tool(call)).collect();
        Box::pin(async move {
            let mut worst = GuardAction::Allow;
            for fut in futs {
                let action = fut.await?;
                if action.is_denied() {
                    return Ok(action);
                }
                if matches!(action, GuardAction::Warn { .. }) && matches!(worst, GuardAction::Allow)
                {
                    worst = action;
                }
            }
            Ok(worst)
        })
    }

    fn post_tool(
        &self,
        call: &ToolCall,
        output: &mut ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        // Eagerly call each guardrail's post_tool (mutations run synchronously).
        let futs: Vec<_> = self
            .guardrails
            .iter()
            .map(|g| g.post_tool(call, output))
            .collect();
        Box::pin(async move {
            for fut in futs {
                fut.await?;
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// WarnToDeny
// ---------------------------------------------------------------------------

/// Graduated containment: converts N consecutive `Warn` actions to `Deny`.
///
/// Wraps an inner guardrail. Tracks consecutive `Warn` actions across calls.
/// When the count reaches `threshold`, the `Warn` is escalated to `Deny`.
/// Any `Allow` resets the counter.
pub struct WarnToDeny {
    inner: Arc<dyn Guardrail>,
    threshold: u32,
    consecutive_warns: AtomicU32,
}

impl WarnToDeny {
    pub fn new(inner: Arc<dyn Guardrail>, threshold: u32) -> Self {
        Self {
            inner,
            threshold,
            consecutive_warns: AtomicU32::new(0),
        }
    }

    fn escalate_if_needed(&self, action: GuardAction) -> GuardAction {
        match &action {
            GuardAction::Warn { reason } => {
                let prev = self.consecutive_warns.fetch_add(1, Ordering::Relaxed);
                if prev + 1 >= self.threshold {
                    self.consecutive_warns.store(0, Ordering::Relaxed);
                    GuardAction::deny(format!(
                        "Escalated after {} consecutive warnings: {reason}",
                        self.threshold
                    ))
                } else {
                    action
                }
            }
            GuardAction::Allow => {
                self.consecutive_warns.store(0, Ordering::Relaxed);
                action
            }
            GuardAction::Deny { .. } => {
                self.consecutive_warns.store(0, Ordering::Relaxed);
                action
            }
        }
    }
}

impl GuardrailMeta for WarnToDeny {
    fn name(&self) -> &str {
        "warn_to_deny"
    }
}

impl Guardrail for WarnToDeny {
    fn pre_llm(
        &self,
        request: &mut CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        self.inner.pre_llm(request)
    }

    fn post_llm(
        &self,
        response: &CompletionResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        // Eagerly call inner (doesn't capture `response` per trait elision).
        let fut = self.inner.post_llm(response);
        Box::pin(async move {
            let action = fut.await?;
            Ok(self.escalate_if_needed(action))
        })
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        // Eagerly call inner (doesn't capture `call` per trait elision).
        let fut = self.inner.pre_tool(call);
        Box::pin(async move {
            let action = fut.await?;
            Ok(self.escalate_if_needed(action))
        })
    }

    fn post_tool(
        &self,
        call: &ToolCall,
        output: &mut ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        self.inner.post_tool(call, output)
    }
}

// ---------------------------------------------------------------------------
// ConditionalGuardrail
// ---------------------------------------------------------------------------

/// Predicate-gated guardrail — only runs the inner guardrail when the
/// predicate returns `true` for the tool name.
///
/// Use for patterns like "apply this guardrail only to MCP tools" or
/// "only check bash tool calls".
///
/// The predicate receives the tool name for `pre_tool`/`post_tool` hooks.
/// For `pre_llm`/`post_llm` hooks (no tool name), the inner guardrail
/// always runs.
pub struct ConditionalGuardrail {
    inner: Arc<dyn Guardrail>,
    predicate: Arc<dyn Fn(&str) -> bool + Send + Sync>,
}

impl ConditionalGuardrail {
    pub fn new(
        inner: Arc<dyn Guardrail>,
        predicate: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    ) -> Self {
        Self { inner, predicate }
    }
}

impl GuardrailMeta for ConditionalGuardrail {
    fn name(&self) -> &str {
        "conditional"
    }
}

impl Guardrail for ConditionalGuardrail {
    fn pre_llm(
        &self,
        request: &mut CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        self.inner.pre_llm(request)
    }

    fn post_llm(
        &self,
        response: &CompletionResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        self.inner.post_llm(response)
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        if (self.predicate)(&call.name) {
            self.inner.pre_tool(call)
        } else {
            Box::pin(async { Ok(GuardAction::Allow) })
        }
    }

    fn post_tool(
        &self,
        call: &ToolCall,
        output: &mut ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        if (self.predicate)(&call.name) {
            self.inner.post_tool(call, output)
        } else {
            Box::pin(async { Ok(()) })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{StopReason, TokenUsage};

    /// A guardrail that always denies pre_tool calls.
    struct AlwaysDenyGuardrail;
    impl Guardrail for AlwaysDenyGuardrail {
        fn pre_tool(
            &self,
            _call: &ToolCall,
        ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
            Box::pin(async { Ok(GuardAction::deny("blocked")) })
        }
        fn post_llm(
            &self,
            _response: &CompletionResponse,
        ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
            Box::pin(async { Ok(GuardAction::deny("blocked")) })
        }
    }

    /// A guardrail that always allows.
    struct AlwaysAllowGuardrail;
    impl Guardrail for AlwaysAllowGuardrail {}

    /// A guardrail that always warns.
    struct AlwaysWarnGuardrail;
    impl Guardrail for AlwaysWarnGuardrail {
        fn pre_tool(
            &self,
            _call: &ToolCall,
        ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
            Box::pin(async { Ok(GuardAction::warn("suspicious")) })
        }
        fn post_llm(
            &self,
            _response: &CompletionResponse,
        ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
            Box::pin(async { Ok(GuardAction::warn("suspicious")) })
        }
    }

    fn test_call(name: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input: serde_json::json!({}),
        }
    }

    fn test_response() -> CompletionResponse {
        CompletionResponse {
            content: vec![],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }
    }

    // --- GuardrailChain tests ---

    #[tokio::test]
    async fn chain_first_deny_wins() {
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysAllowGuardrail) as Arc<dyn Guardrail>,
            Arc::new(AlwaysDenyGuardrail),
            Arc::new(AlwaysAllowGuardrail),
        ]);
        let action = chain.pre_tool(&test_call("bash")).await.unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn chain_all_allow() {
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysAllowGuardrail) as Arc<dyn Guardrail>,
            Arc::new(AlwaysAllowGuardrail),
        ]);
        let action = chain.pre_tool(&test_call("read")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn chain_post_llm_first_deny_wins() {
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysAllowGuardrail) as Arc<dyn Guardrail>,
            Arc::new(AlwaysDenyGuardrail),
        ]);
        let action = chain.post_llm(&test_response()).await.unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn chain_empty_allows() {
        let chain = GuardrailChain::new(vec![]);
        let action = chain.pre_tool(&test_call("bash")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn chain_propagates_warn() {
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysAllowGuardrail) as Arc<dyn Guardrail>,
            Arc::new(AlwaysWarnGuardrail),
            Arc::new(AlwaysAllowGuardrail),
        ]);
        let action = chain.pre_tool(&test_call("bash")).await.unwrap();
        assert!(
            matches!(action, GuardAction::Warn { .. }),
            "expected Warn, got: {action:?}"
        );
    }

    #[tokio::test]
    async fn chain_deny_trumps_warn() {
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysWarnGuardrail) as Arc<dyn Guardrail>,
            Arc::new(AlwaysDenyGuardrail),
        ]);
        let action = chain.pre_tool(&test_call("bash")).await.unwrap();
        assert!(action.is_denied(), "Deny should win over Warn");
    }

    #[tokio::test]
    async fn chain_post_llm_propagates_warn() {
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysWarnGuardrail) as Arc<dyn Guardrail>,
            Arc::new(AlwaysAllowGuardrail),
        ]);
        let action = chain.post_llm(&test_response()).await.unwrap();
        assert!(matches!(action, GuardAction::Warn { .. }));
    }

    // --- WarnToDeny tests ---

    #[tokio::test]
    async fn warn_to_deny_escalates_after_threshold() {
        let inner = Arc::new(AlwaysWarnGuardrail) as Arc<dyn Guardrail>;
        let g = WarnToDeny::new(inner, 3);
        let call = test_call("bash");

        // First two: still Warn
        let a1 = g.pre_tool(&call).await.unwrap();
        assert!(matches!(a1, GuardAction::Warn { .. }));
        let a2 = g.pre_tool(&call).await.unwrap();
        assert!(matches!(a2, GuardAction::Warn { .. }));

        // Third: escalated to Deny
        let a3 = g.pre_tool(&call).await.unwrap();
        assert!(a3.is_denied());
        if let GuardAction::Deny { reason } = &a3 {
            assert!(reason.contains("3 consecutive warnings"));
        }
    }

    #[tokio::test]
    async fn warn_to_deny_resets_on_allow() {
        let inner = Arc::new(AlwaysWarnGuardrail) as Arc<dyn Guardrail>;
        let g = WarnToDeny::new(inner, 3);
        let call = test_call("bash");

        // Two warns
        g.pre_tool(&call).await.unwrap();
        g.pre_tool(&call).await.unwrap();

        // Reset (simulating an Allow from inner)
        g.consecutive_warns.store(0, Ordering::Relaxed);

        // Two more warns — should not escalate yet
        let a1 = g.pre_tool(&call).await.unwrap();
        assert!(matches!(a1, GuardAction::Warn { .. }));
        let a2 = g.pre_tool(&call).await.unwrap();
        assert!(matches!(a2, GuardAction::Warn { .. }));

        // Third → escalate
        let a3 = g.pre_tool(&call).await.unwrap();
        assert!(a3.is_denied());
    }

    #[tokio::test]
    async fn warn_to_deny_allow_resets_counter() {
        let g = WarnToDeny::new(Arc::new(AlwaysAllowGuardrail) as Arc<dyn Guardrail>, 1);
        let call = test_call("bash");
        // Set counter artificially
        g.consecutive_warns.store(5, Ordering::Relaxed);
        let action = g.pre_tool(&call).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        assert_eq!(g.consecutive_warns.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn warn_to_deny_post_llm_escalates() {
        let inner = Arc::new(AlwaysWarnGuardrail) as Arc<dyn Guardrail>;
        let g = WarnToDeny::new(inner, 2);
        let resp = test_response();

        let a1 = g.post_llm(&resp).await.unwrap();
        assert!(matches!(a1, GuardAction::Warn { .. }));

        let a2 = g.post_llm(&resp).await.unwrap();
        assert!(a2.is_denied());
    }

    // --- ConditionalGuardrail tests ---

    #[tokio::test]
    async fn conditional_runs_when_predicate_true() {
        let g = ConditionalGuardrail::new(
            Arc::new(AlwaysDenyGuardrail) as Arc<dyn Guardrail>,
            Arc::new(|name: &str| name == "bash"),
        );
        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn conditional_skips_when_false() {
        let g = ConditionalGuardrail::new(
            Arc::new(AlwaysDenyGuardrail) as Arc<dyn Guardrail>,
            Arc::new(|name: &str| name == "bash"),
        );
        let action = g.pre_tool(&test_call("read")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn conditional_post_tool_skips_when_false() {
        let g = ConditionalGuardrail::new(
            Arc::new(AlwaysDenyGuardrail) as Arc<dyn Guardrail>,
            Arc::new(|name: &str| name == "bash"),
        );
        let call = test_call("read");
        let mut output = ToolOutput::success("data".to_string());
        g.post_tool(&call, &mut output).await.unwrap();
        assert_eq!(output.content, "data");
    }

    #[tokio::test]
    async fn conditional_llm_hooks_always_run() {
        let g = ConditionalGuardrail::new(
            Arc::new(AlwaysDenyGuardrail) as Arc<dyn Guardrail>,
            Arc::new(|_name: &str| false),
        );
        let action = g.post_llm(&test_response()).await.unwrap();
        assert!(action.is_denied());
    }

    // --- Meta tests ---

    #[test]
    fn chain_meta_name() {
        let chain = GuardrailChain::new(vec![]);
        assert_eq!(chain.name(), "chain");
    }

    #[test]
    fn warn_to_deny_meta_name() {
        let g = WarnToDeny::new(Arc::new(AlwaysAllowGuardrail) as Arc<dyn Guardrail>, 3);
        assert_eq!(g.name(), "warn_to_deny");
    }

    #[test]
    fn conditional_meta_name() {
        let g = ConditionalGuardrail::new(
            Arc::new(AlwaysAllowGuardrail) as Arc<dyn Guardrail>,
            Arc::new(|_: &str| true),
        );
        assert_eq!(g.name(), "conditional");
    }
}
