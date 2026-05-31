//! Browser reliability decorators over raw MCP tools (spec capability 4).
//!
//! `chrome-devtools-mcp`'s interaction tools (`click`, `fill`, …) have two
//! hazards the research flagged as the #1 operational failure modes:
//!
//! 1. They default `includeSnapshot:false`, so after acting the model is blind
//!    to the resulting page unless it remembers to ask. We inject
//!    `includeSnapshot:true` so every mutating action returns a fresh
//!    observation.
//! 2. `uid` handles are snapshot-scoped. Acting on a `uid` from a stale
//!    snapshot fails with a "no element with uid" error. We retry once after
//!    forcing a re-observation (the wrapper surfaces a structured hint so the
//!    loop re-grounds), then escalate.
//!
//! [`ReliableInteractionTool`] is an `Arc<dyn Tool>` decorator: it wraps any
//! inner tool, so it composes with the MCP-stamped tools from
//! [`connect_preset`](crate::connect_preset) without the MCP layer knowing.
//! This mirrors the existing `find_closest_tool` tool-name-repair pattern.

use std::sync::Arc;

use serde_json::Value;

use crate::error::Error;
use crate::tool::{Tool, ToolOutput};
use crate::{ExecutionContext, llm::types::ToolDefinition};

/// chrome-devtools-mcp interaction tools that MUTATE the page and therefore both
/// (a) benefit from a forced fresh snapshot and (b) are subject to stale-uid
/// failures. Navigation/read tools are intentionally excluded.
pub(crate) const MUTATING_TOOLS: &[&str] = &[
    "click",
    "fill",
    "fill_form",
    "hover",
    "drag",
    "type_text",
    "press_key",
    "upload_file",
];

/// Heuristic: does this tool error indicate the target `uid` was not found in
/// the current snapshot (i.e. the agent acted on a stale handle)?
pub(crate) fn is_stale_uid_error(output: &ToolOutput) -> bool {
    if !output.is_error {
        return false;
    }
    let c = output.content.to_lowercase();
    // chrome-devtools-mcp phrasings + defensive variants.
    (c.contains("uid") && (c.contains("no element") || c.contains("not found")))
        || c.contains("no element with uid")
        || c.contains("stale")
        || (c.contains("element") && c.contains("no longer"))
}

/// Wraps an interaction tool to (1) force `includeSnapshot:true` and (2) retry
/// once on a stale-uid error.
pub struct ReliableInteractionTool {
    inner: Arc<dyn Tool>,
    /// Whether the wrapped tool is in [`MUTATING_TOOLS`] (decided at construction).
    mutating: bool,
}

impl ReliableInteractionTool {
    /// Wrap `inner`. Non-mutating tools pass through unchanged (no snapshot
    /// injection, no retry) so this is safe to apply uniformly across a tool set.
    pub fn wrap(inner: Arc<dyn Tool>) -> Self {
        let mutating = MUTATING_TOOLS.contains(&inner.definition().name.as_str());
        Self { inner, mutating }
    }

    /// Wrap every tool in a set; the result is a drop-in replacement for an
    /// agent's `base_tools`/`tools`.
    pub fn wrap_all(tools: Vec<Arc<dyn Tool>>) -> Vec<Arc<dyn Tool>> {
        tools
            .into_iter()
            .map(|t| Arc::new(Self::wrap(t)) as Arc<dyn Tool>)
            .collect()
    }

    /// Add `includeSnapshot:true` to a mutating tool's input object.
    fn with_snapshot(&self, mut input: Value) -> Value {
        if self.mutating
            && let Value::Object(ref mut map) = input
        {
            map.insert("includeSnapshot".to_string(), Value::Bool(true));
        }
        input
    }
}

impl Tool for ReliableInteractionTool {
    fn definition(&self) -> ToolDefinition {
        self.inner.definition()
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
    {
        let input = self.with_snapshot(input);
        // Own the ctx inside the future so it borrows only `&self` (the trait's
        // elided `'_`), not the separate `ctx` parameter borrow — otherwise the
        // borrowed `ctx` escapes the method (E0521). ExecutionContext is Clone.
        let ctx = ctx.clone();
        Box::pin(async move {
            let first = self.inner.execute(&ctx, input.clone()).await?;
            if self.mutating && is_stale_uid_error(&first) {
                // Stale handle: the snapshot the agent grounded on is gone.
                // Retry once; the forced includeSnapshot on this retry returns a
                // fresh tree so the loop can re-ground. Append a structured hint.
                let retried = self.inner.execute(&ctx, input).await?;
                if retried.is_error {
                    return Ok(ToolOutput::error(format!(
                        "{}\n\n[browser] stale uid: re-snapshot and re-resolve the \
                         target element from the latest snapshot, then retry.",
                        retried.content
                    )));
                }
                return Ok(retried);
            }
            Ok(first)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A mock tool that records each input it receives and returns scripted
    /// outputs in order (cycling on the last).
    struct ScriptedTool {
        name: String,
        calls: Arc<AtomicUsize>,
        inputs: Arc<Mutex<Vec<Value>>>,
        outputs: Vec<ToolOutput>,
    }

    impl ScriptedTool {
        fn new(
            name: &str,
            outputs: Vec<ToolOutput>,
        ) -> (Arc<Self>, Arc<AtomicUsize>, Arc<Mutex<Vec<Value>>>) {
            let calls = Arc::new(AtomicUsize::new(0));
            let inputs = Arc::new(Mutex::new(Vec::new()));
            let t = Arc::new(Self {
                name: name.to_string(),
                calls: Arc::clone(&calls),
                inputs: Arc::clone(&inputs),
                outputs,
            });
            (t, calls, inputs)
        }
    }

    impl Tool for ScriptedTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: "scripted".into(),
                input_schema: serde_json::json!({ "type": "object" }),
            }
        }
        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>,
        > {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            self.inputs.lock().expect("inputs lock").push(input);
            let idx = n.min(self.outputs.len() - 1);
            let out = self.outputs[idx].clone();
            Box::pin(async move { Ok(out) })
        }
    }

    #[tokio::test]
    async fn injects_include_snapshot_for_mutating_tool() {
        let (inner, calls, inputs) = ScriptedTool::new("click", vec![ToolOutput::success("ok")]);
        let tool = ReliableInteractionTool::wrap(inner);
        tool.execute(
            &ExecutionContext::default(),
            serde_json::json!({ "uid": "1_2" }),
        )
        .await
        .expect("ok");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let recorded = &inputs.lock().expect("lock")[0];
        assert_eq!(
            recorded["includeSnapshot"],
            Value::Bool(true),
            "mutating tool must get includeSnapshot:true, got {recorded}"
        );
        assert_eq!(recorded["uid"], "1_2", "original args preserved");
    }

    #[tokio::test]
    async fn does_not_inject_for_non_mutating_tool() {
        let (inner, _calls, inputs) =
            ScriptedTool::new("take_snapshot", vec![ToolOutput::success("tree")]);
        let tool = ReliableInteractionTool::wrap(inner);
        tool.execute(&ExecutionContext::default(), serde_json::json!({}))
            .await
            .expect("ok");
        let recorded = &inputs.lock().expect("lock")[0];
        assert!(
            recorded.get("includeSnapshot").is_none(),
            "non-mutating tool must not be modified, got {recorded}"
        );
    }

    #[tokio::test]
    async fn retries_once_on_stale_uid_then_succeeds() {
        let (inner, calls, _inputs) = ScriptedTool::new(
            "click",
            vec![
                ToolOutput::error("Error: no element with uid 1_2"),
                ToolOutput::success("clicked"),
            ],
        );
        let tool = ReliableInteractionTool::wrap(inner);
        let out = tool
            .execute(
                &ExecutionContext::default(),
                serde_json::json!({ "uid": "1_2" }),
            )
            .await
            .expect("ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "must retry exactly once");
        assert!(!out.is_error, "second attempt succeeded");
        assert_eq!(out.content, "clicked");
    }

    #[tokio::test]
    async fn stale_uid_twice_escalates_with_hint() {
        let (inner, calls, _inputs) =
            ScriptedTool::new("click", vec![ToolOutput::error("no element with uid 1_2")]);
        let tool = ReliableInteractionTool::wrap(inner);
        let out = tool
            .execute(
                &ExecutionContext::default(),
                serde_json::json!({ "uid": "1_2" }),
            )
            .await
            .expect("ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "one retry then give up");
        assert!(out.is_error);
        assert!(
            out.content.contains("re-snapshot"),
            "escalation must hint re-grounding, got {}",
            out.content
        );
    }

    #[tokio::test]
    async fn non_stale_error_is_not_retried() {
        let (inner, calls, _inputs) =
            ScriptedTool::new("click", vec![ToolOutput::error("Error: network timeout")]);
        let tool = ReliableInteractionTool::wrap(inner);
        let out = tool
            .execute(
                &ExecutionContext::default(),
                serde_json::json!({ "uid": "1_2" }),
            )
            .await
            .expect("ok");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "non-stale error must not retry"
        );
        assert!(out.is_error);
    }

    #[test]
    fn is_stale_uid_error_matches_phrasings() {
        assert!(is_stale_uid_error(&ToolOutput::error(
            "no element with uid 1_2"
        )));
        assert!(is_stale_uid_error(&ToolOutput::error(
            "Error: uid 3_7 not found in snapshot"
        )));
        assert!(!is_stale_uid_error(&ToolOutput::error("network timeout")));
        assert!(!is_stale_uid_error(&ToolOutput::success(
            "no element with uid (but success?)"
        )));
    }

    #[test]
    fn wrap_all_preserves_count_and_names() {
        let (a, _, _) = ScriptedTool::new("click", vec![ToolOutput::success("x")]);
        let (b, _, _) = ScriptedTool::new("take_snapshot", vec![ToolOutput::success("y")]);
        let wrapped = ReliableInteractionTool::wrap_all(vec![a, b]);
        assert_eq!(wrapped.len(), 2);
        assert_eq!(wrapped[0].definition().name, "click");
        assert_eq!(wrapped[1].definition().name, "take_snapshot");
    }
}
