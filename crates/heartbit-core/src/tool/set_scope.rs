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
    /// The host workspace root. A scope root OUTSIDE it is futile: the file
    /// tools reject any write outside the workspace regardless of scope (live
    /// finding 6a265efd — models scoped /tmp for "a temporary directory", every
    /// write was rejected, then they thrashed into the repo via a symlink and
    /// stray dirs). The first outside scope is refused with a redirect to an
    /// in-workspace scratch subdir; relocating INTO the workspace is the correct
    /// course-correction, never refused.
    workspace: Option<std::path::PathBuf>,
    /// One-shot: the first scope declaring an OUTSIDE root is refused with the
    /// scratch redirect; an explicit retry passes (e.g. read-only / bash use).
    outside_warned: std::sync::atomic::AtomicBool,
}

impl SetScopeTool {
    /// `guard` must be the SAME guard registered in the runner's guardrails.
    pub fn new(guard: Arc<ScopeGuard>) -> Self {
        Self {
            guard,
            workspace: None,
            outside_warned: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Set the host workspace root enabling the outside-scope redirect.
    /// Also shared with the guard so it anchors relative file paths against
    /// the same root (set_scope declares ABSOLUTE roots; the file tools pass
    /// workspace-RELATIVE paths — without a common anchor the guard falsely
    /// denied in-scope writes).
    pub fn with_workspace(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        let root = root.into();
        self.guard.set_workspace(&root);
        self.workspace = Some(root);
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
                          widening is deliberate and visible, drift is not. These are \
                          WRITABLE roots, not a read-only filter: scoping to \".\" makes \
                          the whole repo writable, and an EMPTY list REMOVES the \
                          restriction (allows ALL mutations) — it does NOT make the task \
                          read-only. To stay read-only, simply do not modify files; to \
                          limit edits, scope to the specific files you will change."
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
                .map(|s| {
                    // Resolve RELATIVE roots against the workspace so the scope
                    // is ONE canonical absolute location. A relative `./scratch`
                    // is ambiguous (the write tool reads it workspace-relative,
                    // bash relative to its drifting cwd) — live finding
                    // 6a25d21b. Absolute paths are kept as-is.
                    let p = PathBuf::from(s);
                    match (&self.workspace, p.is_relative()) {
                        (Some(ws), true) => ws.join(p),
                        _ => p,
                    }
                })
                .collect();
            let n = roots.len();
            // Outside-scope redirect: declaring a scope root OUTSIDE the host
            // workspace is futile — the file tools reject any write outside the
            // workspace regardless of scope. The first such scope is refused with
            // a redirect to an in-workspace scratch subdir; an explicit retry
            // passes (e.g. read-only or bash-only use). Relocating INTO the
            // workspace is the correct course-correction and is never refused
            // (live finding 6a265efd: the old outside→inside refusal told the
            // agent to "keep building in /tmp" — impossible — so it symlinked and
            // littered the repo).
            if let Some(ws) = &self.workspace {
                let outside: Vec<&PathBuf> = roots.iter().filter(|r| !r.starts_with(ws)).collect();
                if !outside.is_empty()
                    && !self
                        .outside_warned
                        .swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    return Ok(ToolOutput::error(format!(
                        "scope root(s) {} are OUTSIDE this workspace ({}) — the file tools can \
                         only write INSIDE it, so an outside scope cannot be acted on. For \
                         temporary or scratch work, scope a SUBDIRECTORY inside the workspace \
                         instead (e.g. ./scratch-<name>), kept gitignored so it stays \
                         disposable. Re-issue unchanged to override (e.g. read-only or \
                         bash-only use).",
                        outside
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

    // End-to-end (audit 2026-06-09): with_workspace shares the anchor with
    // the guard, so an absolute scope declared via set_scope and a
    // workspace-RELATIVE write target resolve identically — the in-scope
    // write is allowed, the out-of-scope one denied.
    #[tokio::test]
    async fn with_workspace_anchors_relative_writes_against_absolute_scope() {
        let guard = Arc::new(ScopeGuard::new(vec![]));
        let tool = SetScopeTool::new(guard.clone()).with_workspace("/repo");
        // The agent declares a relative scope; set_scope resolves it to
        // /repo/scratch-x (absolute).
        tool.execute(
            &crate::ExecutionContext::default(),
            json!({"paths": ["scratch-x"], "reason": "scratch work"}),
        )
        .await
        .unwrap();
        // A workspace-relative write inside the scope is allowed…
        let allowed = guard
            .pre_tool(&call("write", "scratch-x/new.rs"))
            .await
            .unwrap();
        assert!(
            matches!(allowed, GuardAction::Allow),
            "in-scope relative write must be allowed: {allowed:?}"
        );
        // …one outside it is denied.
        let denied = guard.pre_tool(&call("edit", "src/main.rs")).await.unwrap();
        assert!(
            matches!(denied, GuardAction::Deny { .. }),
            "out-of-scope relative write must be denied: {denied:?}"
        );
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

    #[tokio::test]
    async fn relative_scope_is_resolved_against_the_workspace() {
        // Live finding 6a25d21b: a relative scope "./scratch-crm" was ambiguous
        // — write resolved it workspace-relative, bash via its drifting cwd.
        // set_scope now anchors a relative root to the workspace (one canonical
        // absolute location).
        let guard = Arc::new(ScopeGuard::new(vec![]));
        let tool = SetScopeTool::new(guard.clone()).with_workspace("/repo");
        let out = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"paths": ["./scratch-crm"]}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        let roots = guard.roots();
        assert_eq!(roots, vec![PathBuf::from("/repo/scratch-crm")]);
        // An absolute root INSIDE the workspace is kept verbatim.
        tool.execute(
            &crate::ExecutionContext::default(),
            json!({"paths": ["/repo/elsewhere"]}),
        )
        .await
        .unwrap();
        assert_eq!(guard.roots(), vec![PathBuf::from("/repo/elsewhere")]);
    }

    #[test]
    fn definition_name() {
        let tool = SetScopeTool::new(Arc::new(ScopeGuard::new(vec![])));
        assert_eq!(tool.definition().name, "set_scope");
    }

    #[tokio::test]
    async fn outside_scope_is_refused_once_with_scratch_redirect() {
        // Live finding 6a265efd: models scoped /tmp for "a temporary directory",
        // but the file tools reject every write outside the workspace, so the
        // outside scope is futile. It is refused with a redirect to an
        // in-workspace scratch subdir; relocating INTO the workspace is the
        // correct course-correction and is NEVER refused (the old guard refused
        // it, telling the agent to "keep building in /tmp" — impossible — so it
        // symlinked and littered the repo).
        let guard = Arc::new(ScopeGuard::new(vec![]));
        let tool = SetScopeTool::new(guard.clone()).with_workspace("/repo");
        // An OUTSIDE scope is refused with the scratch redirect, and NOT applied.
        let out = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"paths": ["/tmp/proj"]}),
            )
            .await
            .unwrap();
        assert!(out.is_error, "an outside scope must be refused");
        assert!(out.content.contains("OUTSIDE this workspace"));
        assert!(
            out.content.contains("scratch"),
            "must redirect to an in-workspace scratch subdir: {}",
            out.content
        );
        assert!(
            guard.roots().is_empty(),
            "the futile outside scope must NOT be applied"
        );
        // Relocating INTO the workspace is the correct move — never refused.
        let inside = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"paths": ["/repo/scratch-crm"]}),
            )
            .await
            .unwrap();
        assert!(!inside.is_error, "an in-workspace scope is always allowed");
        assert_eq!(guard.roots(), vec![PathBuf::from("/repo/scratch-crm")]);
        // The one-shot is consumed: an explicit retry of an outside scope passes
        // (e.g. read-only / bash-only use).
        let retry = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"paths": ["/tmp/proj"]}),
            )
            .await
            .unwrap();
        assert!(!retry.is_error, "the explicit retry overrides the redirect");
    }

    #[tokio::test]
    async fn inside_workspace_rescope_is_normal() {
        // Normal repo work (scope inside the workspace from the start) is
        // never bothered by the outside-scope redirect.
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
