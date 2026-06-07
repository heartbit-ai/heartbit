//! `set_scope` — the entry agent declares its working scope after planning.
//! Shares the [`ScopeGuard`] with the runner's guardrails: mutating tool calls
//! (edit/write) outside the declared roots are then denied with actionable
//! feedback. Re-declaring (or widening) the scope is itself a visible,
//! auditable tool call — out-of-scope work can't drift in silently.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::agent::guardrails::ScopeGuard;
use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

/// Declare/replace the working scope enforced by the shared [`ScopeGuard`].
pub struct SetScopeTool {
    guard: Arc<ScopeGuard>,
}

impl SetScopeTool {
    /// `guard` must be the SAME guard registered in the runner's guardrails.
    pub fn new(guard: Arc<ScopeGuard>) -> Self {
        Self { guard }
    }
}

impl Tool for SetScopeTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "set_scope".into(),
            description: "Declare the files/directories your current task is allowed to \
                          MODIFY (edit/write outside them will be denied). Call it after \
                          planning substantive work; call it again to widen the scope — \
                          widening is deliberate and visible, drift is not. An empty list \
                          removes the restriction."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "In-scope roots (files or directories)."
                    },
                    "reason": {
                        "type": "string",
                        "description": "One line on why this is the scope (or why it widened)."
                    }
                },
                "required": ["paths"]
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let Some(paths) = input.get("paths").and_then(|v| v.as_array()) else {
                return Ok(ToolOutput::error("paths is required (an array of strings)"));
            };
            let roots: Vec<PathBuf> = paths
                .iter()
                .filter_map(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from)
                .collect();
            let n = roots.len();
            self.guard.set(roots);
            if n == 0 {
                return Ok(ToolOutput::success(
                    "scope cleared — mutations are unrestricted",
                ));
            }
            let listing = self
                .guard
                .roots()
                .iter()
                .map(|p| format!("- {}", p.display()))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(ToolOutput::success(format!(
                "scope set ({n} roots) — edit/write outside these will be denied:\n{listing}"
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::guardrail::{GuardAction, Guardrail};
    use crate::llm::types::ToolCall;

    fn call(name: &str, path: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input: json!({"file_path": path}),
        }
    }

    #[tokio::test]
    async fn set_scope_seeds_the_shared_guard() {
        let guard = Arc::new(ScopeGuard::new(vec![]));
        let tool = SetScopeTool::new(guard.clone());
        let out = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"paths": ["/tmp/x/src"], "reason": "feature work"}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("/tmp/x/src"));
        // The SAME guard now denies out-of-scope mutations…
        let denied = guard
            .pre_tool(&call("edit", "/tmp/elsewhere/a.rs"))
            .await
            .unwrap();
        assert!(matches!(denied, GuardAction::Deny { .. }));
        // …and allows in-scope ones.
        let allowed = guard
            .pre_tool(&call("write", "/tmp/x/src/new.rs"))
            .await
            .unwrap();
        assert!(matches!(allowed, GuardAction::Allow));
    }

    #[tokio::test]
    async fn set_scope_replaces_not_extends() {
        let guard = Arc::new(ScopeGuard::new(vec![PathBuf::from("/tmp/old")]));
        let tool = SetScopeTool::new(guard.clone());
        tool.execute(
            &crate::ExecutionContext::default(),
            json!({"paths": ["/tmp/new"]}),
        )
        .await
        .unwrap();
        let denied = guard
            .pre_tool(&call("edit", "/tmp/old/a.rs"))
            .await
            .unwrap();
        assert!(
            matches!(denied, GuardAction::Deny { .. }),
            "the old root must be gone (replace semantics)"
        );
    }

    #[tokio::test]
    async fn empty_paths_clears_the_restriction() {
        let guard = Arc::new(ScopeGuard::new(vec![PathBuf::from("/tmp/x")]));
        let tool = SetScopeTool::new(guard.clone());
        tool.execute(&crate::ExecutionContext::default(), json!({"paths": []}))
            .await
            .unwrap();
        let action = guard
            .pre_tool(&call("edit", "/anywhere/a.rs"))
            .await
            .unwrap();
        assert!(matches!(action, GuardAction::Allow));
    }

    #[tokio::test]
    async fn missing_paths_is_an_error() {
        let guard = Arc::new(ScopeGuard::new(vec![]));
        let tool = SetScopeTool::new(guard);
        let out = tool
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[test]
    fn definition_name() {
        let tool = SetScopeTool::new(Arc::new(ScopeGuard::new(vec![])));
        assert_eq!(tool.definition().name, "set_scope");
    }
}
