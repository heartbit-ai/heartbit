//! `DistillingTool` — an `Arc<dyn Tool>` decorator that distills snapshot output
//! before it re-enters the agent's context (token-control wiring).
//!
//! [`super::distill::distill_snapshot`] is a pure transform, but nothing applied
//! it to live tool output: every `take_snapshot` (and every mutating tool with
//! `includeSnapshot:true`) returned the FULL accessibility tree, which the agent
//! loop then re-sends on every subsequent turn. On a busy page that is the single
//! largest input-token cost. This decorator runs the distiller on any output that
//! looks like a snapshot (contains a `uid=` line), dropping the redundant echoed
//! `StaticText` while preserving every interactive `uid` — so the agent sees a
//! smaller, clearer tree and the conversation grows far more slowly.
//!
//! It composes with [`super::harness::ReliableInteractionTool`]: distillation
//! wraps the OUTSIDE (`DistillingTool::wrap(reliable)`), so a mutating tool's
//! forced fresh snapshot is also distilled. Distillation is idempotent and only
//! touches `uid`-bearing content, so applying it uniformly is safe — non-snapshot
//! outputs (network lists, console logs, plain text) pass through untouched.

use std::sync::Arc;

use serde_json::Value;

use crate::error::Error;
use crate::execution_context::ExecutionContext;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

use super::distill::{DistillConfig, distill_snapshot};

/// Decorator that distills snapshot-shaped output of the wrapped tool.
pub struct DistillingTool {
    inner: Arc<dyn Tool>,
    cfg: DistillConfig,
}

impl DistillingTool {
    /// Wrap `inner`, distilling its snapshot output with `cfg`.
    pub fn wrap(inner: Arc<dyn Tool>, cfg: DistillConfig) -> Self {
        Self { inner, cfg }
    }

    /// Wrap every tool in a set; a drop-in replacement for an agent's tools.
    pub fn wrap_all(tools: Vec<Arc<dyn Tool>>, cfg: DistillConfig) -> Vec<Arc<dyn Tool>> {
        tools
            .into_iter()
            .map(|t| Arc::new(Self::wrap(t, cfg.clone())) as Arc<dyn Tool>)
            .collect()
    }
}

impl Tool for DistillingTool {
    fn definition(&self) -> ToolDefinition {
        self.inner.definition()
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>>
    {
        // Own everything the future needs so it borrows only `&self`'s elided
        // lifetime (mirrors ReliableInteractionTool's ctx-clone fix).
        let inner = Arc::clone(&self.inner);
        let cfg = self.cfg.clone();
        let ctx = ctx.clone();
        Box::pin(async move {
            let out = inner.execute(&ctx, input).await?;
            // Only distill snapshot-shaped output; pass everything else through.
            if out.content.contains("uid=") {
                Ok(ToolOutput {
                    content: distill_snapshot(&out.content, &cfg),
                    is_error: out.is_error,
                })
            } else {
                Ok(out)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool returning a fixed output, for decorator tests.
    struct FixedTool {
        name: String,
        output: ToolOutput,
    }
    impl Tool for FixedTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: "fixed".into(),
                input_schema: serde_json::json!({ "type": "object" }),
            }
        }
        fn execute(
            &self,
            _ctx: &ExecutionContext,
            _input: Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + '_>,
        > {
            let out = self.output.clone();
            Box::pin(async move { Ok(out) })
        }
    }

    fn fixed(name: &str, content: &str, is_error: bool) -> Arc<dyn Tool> {
        Arc::new(FixedTool {
            name: name.to_string(),
            output: ToolOutput {
                content: content.to_string(),
                is_error,
            },
        })
    }

    // A snapshot with the canonical redundancy: each link followed by a child
    // StaticText echoing its name (which the distiller drops).
    const SNAP: &str = r#"## Latest page snapshot
uid=1_0 RootWebArea "Hacker News" url="https://news.ycombinator.com/"
  uid=1_2 link "Hacker News" url="https://news.ycombinator.com/news"
    uid=1_3 StaticText "Hacker News"
  uid=1_4 link "new" url="https://news.ycombinator.com/newest"
    uid=1_5 StaticText "new""#;

    #[tokio::test]
    async fn distills_snapshot_output_and_shrinks_it() {
        let tool = DistillingTool::wrap(
            fixed("take_snapshot", SNAP, false),
            DistillConfig::default(),
        );
        let out = tool
            .execute(&ExecutionContext::default(), serde_json::json!({}))
            .await
            .expect("ok");
        assert!(
            out.content.len() < SNAP.len(),
            "distilled output must be smaller"
        );
        // Interactive uids survive...
        assert!(out.content.contains("uid=1_2 link \"Hacker News\""));
        assert!(out.content.contains("uid=1_4 link \"new\""));
        // ...redundant echoed StaticText is gone.
        assert!(!out.content.contains("uid=1_3 StaticText \"Hacker News\""));
        // The header (no uid) is preserved.
        assert!(out.content.contains("## Latest page snapshot"));
    }

    #[tokio::test]
    async fn preserves_every_interactive_uid() {
        let tool = DistillingTool::wrap(
            fixed("take_snapshot", SNAP, false),
            DistillConfig::default(),
        );
        let out = tool
            .execute(&ExecutionContext::default(), serde_json::json!({}))
            .await
            .expect("ok");
        for uid in ["uid=1_2", "uid=1_4"] {
            assert!(out.content.contains(uid), "interactive {uid} must survive");
        }
    }

    #[tokio::test]
    async fn non_snapshot_output_passes_through_untouched() {
        // A network-list / console output with no uid lines must be unchanged.
        let raw = "reqid=3 GET https://example.com/ [304]\nreqid=4 GET /style.css [200]";
        let tool = DistillingTool::wrap(
            fixed("list_network_requests", raw, false),
            DistillConfig::default(),
        );
        let out = tool
            .execute(&ExecutionContext::default(), serde_json::json!({}))
            .await
            .expect("ok");
        assert_eq!(out.content, raw, "non-snapshot output must be untouched");
    }

    #[tokio::test]
    async fn preserves_is_error_flag() {
        let tool = DistillingTool::wrap(
            fixed("click", "Error: no element with uid=9_9", true),
            DistillConfig::default(),
        );
        let out = tool
            .execute(&ExecutionContext::default(), serde_json::json!({}))
            .await
            .expect("ok");
        assert!(
            out.is_error,
            "error flag must be preserved through distillation"
        );
    }

    #[tokio::test]
    async fn definition_is_passthrough() {
        let tool = DistillingTool::wrap(
            fixed("take_snapshot", SNAP, false),
            DistillConfig::default(),
        );
        assert_eq!(tool.definition().name, "take_snapshot");
    }

    #[test]
    fn wrap_all_preserves_count_and_names() {
        let tools = vec![
            fixed("take_snapshot", SNAP, false),
            fixed("click", "ok", false),
        ];
        let wrapped = DistillingTool::wrap_all(tools, DistillConfig::default());
        assert_eq!(wrapped.len(), 2);
        assert_eq!(wrapped[0].definition().name, "take_snapshot");
        assert_eq!(wrapped[1].definition().name, "click");
    }
}
