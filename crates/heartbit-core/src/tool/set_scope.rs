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
    /// The host workspace root: a scope that was entirely OUTSIDE it must not
    /// silently move back INSIDE (live finding 6a25947c: scoped /tmp as
    /// requested, then re-scoped into the host repo twice, leaving junk).
    workspace: Option<std::path::PathBuf>,
    /// One-shot: the first outside→inside transition is refused with
    /// guidance; an explicit retry passes (bounded friction).
    inside_warned: std::sync::atomic::AtomicBool,
}

impl SetScopeTool {
    /// `guard` must be the SAME guard registered in the runner's guardrails.
    pub fn new(guard: Arc<ScopeGuard>) -> Self {
        Self {
            guard,
            workspace: None,
            inside_warned: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Set the host workspace root enabling the outside→inside transition
    /// check.
    pub fn with_workspace(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.workspace = Some(root.into());
        self
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
            // Outside→inside transition check: a task deliberately scoped
            // OUTSIDE the host workspace (e.g. "a temporary directory") must
            // not silently relocate INTO it. First attempt is refused with
            // guidance; an explicit retry passes (deliberate, visible).
            if let Some(ws) = &self.workspace {
                let current = self.guard.roots();
                let current_all_outside =
                    !current.is_empty() && current.iter().all(|r| !r.starts_with(ws));
                let new_some_inside = roots.iter().any(|r| r.starts_with(ws));
                if current_all_outside
                    && new_some_inside
                    && !self
                        .inside_warned
                        .swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    return Ok(ToolOutput::error(format!(
                        "scope was deliberately OUTSIDE this workspace ({}) — moving it \
                         INSIDE ({}) contradicts that (a 'temporary directory' stays \
                         outside the repository). Keep building in the current scope, or \
                         if relocating is genuinely needed, ask the user with the \
                         `question` tool first, then call set_scope again.",
                        current
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                        ws.display()
                    )));
                }
            }
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

    #[tokio::test]
    async fn outside_to_inside_workspace_rescope_is_refused_once() {
        // Live finding 6a25947c: scoped /tmp (as requested), then re-scoped
        // INTO the host repo ("temporaire à l'intérieur du workspace") and
        // littered it. The first such transition is refused with guidance;
        // an explicit retry passes (deliberate).
        let guard = Arc::new(ScopeGuard::new(vec![]));
        let tool = SetScopeTool::new(guard.clone()).with_workspace("/repo");
        tool.execute(
            &crate::ExecutionContext::default(),
            json!({"paths": ["/tmp/proj"]}),
        )
        .await
        .unwrap();
        let out = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"paths": ["/repo/crm_temp"]}),
            )
            .await
            .unwrap();
        assert!(out.is_error, "first outside→inside move must be refused");
        assert!(out.content.contains("OUTSIDE this workspace"));
        // The guard still holds the OUTSIDE scope.
        let roots = guard.roots();
        assert_eq!(roots.len(), 1);
        assert!(roots[0].to_string_lossy().contains("/tmp/proj"));
        // A deliberate retry passes (one-shot friction).
        let out2 = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"paths": ["/repo/crm_temp"]}),
            )
            .await
            .unwrap();
        assert!(!out2.is_error, "the explicit retry is allowed");
    }

    #[tokio::test]
    async fn inside_workspace_rescope_is_normal() {
        // Normal repo work (scope inside the workspace from the start) is
        // never bothered by the transition check.
        let guard = Arc::new(ScopeGuard::new(vec![]));
        let tool = SetScopeTool::new(guard).with_workspace("/repo");
        tool.execute(
            &crate::ExecutionContext::default(),
            json!({"paths": ["/repo/src"]}),
        )
        .await
        .unwrap();
        let out = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"paths": ["/repo/src", "/repo/tests"]}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
    }
}
