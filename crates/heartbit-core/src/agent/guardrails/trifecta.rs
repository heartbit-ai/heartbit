//! Runtime lethal-trifecta guardrail — blocks the exfiltration step.
//!
//! [`analyze_tools`](crate::tool::analyze_tools) answers a *configuration*
//! question ("could this agent form the trifecta?") and warns at build time.
//! This guardrail answers the *runtime* question ("did it actually happen?") and
//! **denies the action**: it observes what has flowed through the agent so far
//! and blocks an outbound tool call once all three legs have genuinely occurred:
//!
//! 1. private data was **read** into the context, AND
//! 2. untrusted content was **ingested** into the context, AND
//! 3. the call being attempted can **exfiltrate**.
//!
//! Any single leg on its own is harmless and is never blocked — a run that only
//! read local files, or only fetched a web page, keeps full use of its outbound
//! tools. That asymmetry is the point: capability analysis over-approximates
//! (every agent with `bash` trips it), while runtime observation blocks exactly
//! the step where an indirect prompt injection would become an exfiltration.
//!
//! **Opt-in.** Install it with
//! [`AgentRunnerBuilder::guardrail`](crate::agent::AgentRunnerBuilder::guardrail);
//! nothing changes for callers that do not. Note that an unrestricted shell is
//! all three legs by itself (see
//! [`classify_tool_name`](crate::tool::security::classify_tool_name)), so with
//! `bash` in the tool set this guard is maximally strict by design — it is meant
//! for tool sets where the legs are separable.
//!
//! Leg attribution uses the name-based heuristic
//! [`classify_tool_name`](crate::tool::security::classify_tool_name), because a
//! guardrail sees a [`ToolCall`] (a name plus input), not the `Tool` object — so
//! a per-tool [`Tool::security_exposure`](crate::tool::Tool::security_exposure)
//! override is not visible here. The heuristic is deliberately conservative.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::agent::guardrail::{GuardAction, Guardrail};
use crate::error::Error;
use crate::llm::types::ToolCall;
use crate::tool::ToolOutput;
use crate::tool::security::classify_tool_name;

/// Denies an outbound tool call once private data AND untrusted content have
/// both entered the context. See the module docs.
#[derive(Debug, Default)]
pub struct TrifectaGuard {
    /// A tool that reads private data has completed in this run.
    read_private: AtomicBool,
    /// A tool that ingests untrusted content has completed in this run.
    ingested_untrusted: AtomicBool,
}

impl TrifectaGuard {
    /// A guard with no legs observed yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// `(private_data_read, untrusted_content_ingested)` observed so far — for
    /// diagnostics and tests.
    pub fn legs_seen(&self) -> (bool, bool) {
        (
            self.read_private.load(Ordering::SeqCst),
            self.ingested_untrusted.load(Ordering::SeqCst),
        )
    }

    /// Whether `call` must be denied given what has already flowed.
    ///
    /// Split out of `pre_tool` so the decision is synchronous and unit-testable.
    fn would_deny(&self, call: &ToolCall) -> bool {
        let (private, untrusted) = self.legs_seen();
        classify_tool_name(&call.name).can_exfiltrate && private && untrusted
    }
}

impl Guardrail for TrifectaGuard {
    fn name(&self) -> &str {
        "trifecta"
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        if !self.would_deny(call) {
            return Box::pin(async { Ok(GuardAction::Allow) });
        }
        let reason = format!(
            "lethal trifecta: '{}' can send data externally, and this run has ALREADY \
             read private data AND ingested untrusted content — an injection in that \
             content could be exfiltrating it now. Blocked. Summarize what you need \
             without the untrusted text, use a quarantined reader for it, or ask the \
             user to approve this send explicitly.",
            call.name
        );
        Box::pin(async move { Ok(GuardAction::deny(reason)) })
    }

    fn post_tool(
        &self,
        call: &ToolCall,
        _output: &mut ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        // Observed AFTER execution: the data is now in the context either way, so
        // record the legs even for an errored output (conservative).
        let exposure = classify_tool_name(&call.name);
        if exposure.reads_private_data {
            self.read_private.store(true, Ordering::SeqCst);
        }
        if exposure.ingests_untrusted_content {
            self.ingested_untrusted.store(true, Ordering::SeqCst);
        }
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::ToolDefinition;
    use crate::tool::Tool;
    use std::sync::Arc;

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: format!("c-{name}"),
            name: name.into(),
            input: serde_json::json!({}),
        }
    }

    async fn observe(guard: &TrifectaGuard, name: &str) {
        let mut out = ToolOutput::success("ok");
        guard.post_tool(&call(name), &mut out).await.unwrap();
    }

    // ── Frontier invariant #1 (trifecta) ──
    // All three legs co-occurring at runtime BLOCKS the outbound action.
    #[tokio::test]
    async fn blocks_exfiltration_once_all_three_legs_have_occurred() {
        let guard = TrifectaGuard::new();
        observe(&guard, "read").await; // leg 1: private data
        observe(&guard, "take_snapshot").await; // leg 2: untrusted content
        assert_eq!(guard.legs_seen(), (true, true));

        // leg 3: the outbound call itself → denied.
        match guard.pre_tool(&call("send_email")).await.unwrap() {
            GuardAction::Deny { reason } => {
                assert!(reason.contains("lethal trifecta"), "actionable: {reason}");
                assert!(reason.contains("send_email"), "names the tool: {reason}");
            }
            other => panic!("expected Deny once all three legs occurred, got {other:?}"),
        }
    }

    // No single leg in isolation may produce a false-positive block.
    #[tokio::test]
    async fn does_not_block_when_only_private_data_was_read() {
        let guard = TrifectaGuard::new();
        observe(&guard, "read").await;
        assert_eq!(guard.legs_seen(), (true, false));
        assert!(matches!(
            guard.pre_tool(&call("send_email")).await.unwrap(),
            GuardAction::Allow
        ));
    }

    #[tokio::test]
    async fn does_not_block_when_only_untrusted_content_was_ingested() {
        let guard = TrifectaGuard::new();
        observe(&guard, "take_snapshot").await;
        assert_eq!(guard.legs_seen(), (false, true));
        assert!(matches!(
            guard.pre_tool(&call("send_email")).await.unwrap(),
            GuardAction::Allow
        ));
    }

    #[tokio::test]
    async fn does_not_block_a_tool_that_cannot_exfiltrate() {
        let guard = TrifectaGuard::new();
        observe(&guard, "read").await;
        observe(&guard, "take_snapshot").await;
        // Both legs present, but `grep` has no outbound channel → allowed.
        assert!(matches!(
            guard.pre_tool(&call("grep")).await.unwrap(),
            GuardAction::Allow
        ));
    }

    #[tokio::test]
    async fn a_fresh_run_blocks_nothing() {
        let guard = TrifectaGuard::new();
        for tool in ["send_email", "webfetch", "read", "grep"] {
            assert!(
                matches!(
                    guard.pre_tool(&call(tool)).await.unwrap(),
                    GuardAction::Allow
                ),
                "{tool} must be allowed before any leg has occurred"
            );
        }
    }

    #[test]
    fn name_is_trifecta() {
        assert_eq!(TrifectaGuard::new().name(), "trifecta");
    }

    /// End-to-end: with the guard installed on a real `AgentRunner`, the blocked
    /// exfiltration tool's `execute()` is NEVER reached, while the two harmless
    /// legs run normally.
    #[tokio::test(flavor = "multi_thread")]
    async fn denied_exfiltration_never_reaches_the_tool_in_a_real_run() {
        use crate::agent::AgentRunner;
        use crate::agent::test_helpers::MockProvider;
        use crate::llm::types::{CompletionResponse, ContentBlock, StopReason, TokenUsage};
        use std::sync::atomic::AtomicUsize;

        /// Records every execution so we can prove the send never happened.
        struct Probe {
            name: &'static str,
            runs: Arc<AtomicUsize>,
        }
        impl Tool for Probe {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: self.name.into(),
                    description: format!("probe {}", self.name),
                    input_schema: serde_json::json!({"type":"object","properties":{}}),
                }
            }
            fn execute(
                &self,
                _ctx: &crate::ExecutionContext,
                _input: serde_json::Value,
            ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
                self.runs.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Ok(ToolOutput::success("done")) })
            }
        }
        fn tool_use(id: &str, name: &str) -> CompletionResponse {
            CompletionResponse {
                content: vec![ContentBlock::ToolUse {
                    id: id.into(),
                    name: name.into(),
                    input: serde_json::json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            }
        }

        let read_runs = Arc::new(AtomicUsize::new(0));
        let fetch_runs = Arc::new(AtomicUsize::new(0));
        let send_runs = Arc::new(AtomicUsize::new(0));

        // The model reads a file, fetches a page, then tries to send it out.
        let provider = Arc::new(MockProvider::new(vec![
            tool_use("c1", "read"),
            tool_use("c2", "take_snapshot"),
            tool_use("c3", "send_email"),
            MockProvider::text_response("done", 5, 1),
        ]));
        let runner = AgentRunner::builder(provider.clone())
            .name("test")
            .system_prompt("sys")
            .tool(Arc::new(Probe {
                name: "read",
                runs: read_runs.clone(),
            }))
            .tool(Arc::new(Probe {
                name: "take_snapshot",
                runs: fetch_runs.clone(),
            }))
            .tool(Arc::new(Probe {
                name: "send_email",
                runs: send_runs.clone(),
            }))
            .guardrail(Arc::new(TrifectaGuard::new()))
            .max_turns(6)
            .build()
            .unwrap();
        runner.execute("go").await.unwrap();

        // The two harmless legs executed…
        assert_eq!(read_runs.load(Ordering::SeqCst), 1, "read should run");
        assert_eq!(fetch_runs.load(Ordering::SeqCst), 1, "fetch should run");
        // …and the exfiltration never reached the tool.
        assert_eq!(
            send_runs.load(Ordering::SeqCst),
            0,
            "the trifecta-denied send must never reach the tool"
        );

        // The agent saw the denial as an error tool result and kept going.
        let reqs = provider.captured_requests.lock().unwrap();
        let denied = reqs
            .iter()
            .flat_map(|r| r.messages.iter())
            .flat_map(|m| m.content.iter())
            .any(|b| {
                matches!(b, ContentBlock::ToolResult { is_error, content, .. }
                if *is_error && content.contains("lethal trifecta"))
            });
        assert!(denied, "the denial must surface as an error tool result");
    }
}
