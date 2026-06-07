//! Deterministic scope guard.
//!
//! A `pre_tool` guardrail that bounds the *direction* of work: mutating tool
//! calls (`edit` / `write` / `patch`) whose target file path falls OUTSIDE an
//! allowlist of roots are denied with actionable feedback, prompting a
//! scope-expansion question instead of silent drift.
//!
//! No LLM cost — the check is a lexical path comparison (no filesystem access).
//! It mirrors the path-allowlist `PreToolUse` hook pattern from Claude Code and
//! the agent-guardrails literature: cheap, deterministic, and seeded from the
//! task's known files (e.g. `FileTracker` paths or the plan) by the harness.
//!
//! Semantics (see the completion-loop-harness design, P7):
//! - Empty allowlist → guard is unseeded → **allow everything** (the harness
//!   seeds it later; an unseeded guard must never block).
//! - Non-mutating tools (`read`, `grep`, `bash`, …) → allow.
//! - Mutating call with no parseable file path (e.g. `patch`, whose path lives
//!   inside the diff text) → allow.
//! - Mutating call on a path under ANY allowlisted root → allow.
//! - Mutating call on a path OUTSIDE every allowlisted root → **deny** with a
//!   scope-expansion message naming the path and the roots.
//!
//! Path comparison is lexical (`normalize_path` resolves `.`/`..` without
//! touching the filesystem) and component-wise (`Path::starts_with`), so a
//! sibling whose name merely shares a string prefix with a root (e.g.
//! `/x/src-other` vs root `/x/src`) is correctly treated as out-of-scope.
//! Relative roots only match relative paths and vice versa.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{PoisonError, RwLock};

use crate::agent::guardrail::{GuardAction, Guardrail};
use crate::error::Error;
use crate::llm::types::ToolCall;
use crate::workspace::normalize_path;

/// Tools that mutate files. Mirrors the TUI plan-mode `is_mutating` gate
/// (`heartbit-tui` `main.rs`), minus `bash` (which has no parseable path).
/// `patch` is included so its omission of a top-level path field is an
/// *intentional* "no parseable path → allow", not an accidental gap.
const MUTATING_TOOLS: &[&str] = &["edit", "write", "patch"];

/// Extract the target file path from a mutating tool's input, if one is present
/// as a top-level field. `edit` and `write` use `"file_path"`; `patch` embeds
/// its path inside the diff text and so has no parseable top-level path.
fn extract_path(input: &serde_json::Value) -> Option<PathBuf> {
    input
        .get("file_path")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

/// Deterministic path-allowlist scope guard.
///
/// Construct with the initial in-scope roots ([`ScopeGuard::new`]) and extend
/// at runtime via [`ScopeGuard::allow`] when the user approves a
/// scope-expansion. Roots may be individual files or directories; a file path
/// is in-scope when it equals, or is a descendant of, any allowlisted root.
pub struct ScopeGuard {
    /// Normalized in-scope roots. An empty Vec means "unseeded → allow all".
    allowed: RwLock<Vec<PathBuf>>,
}

impl ScopeGuard {
    /// Create a scope guard seeded with the given allowlist of roots.
    ///
    /// An empty `allowed` leaves the guard unseeded — it will allow every
    /// mutation until [`ScopeGuard::allow`] adds at least one root.
    pub fn new(allowed: Vec<PathBuf>) -> Self {
        let normalized = allowed.iter().map(|p| normalize_path(p)).collect();
        Self {
            allowed: RwLock::new(normalized),
        }
    }

    /// Extend the allowlist at runtime with an additional in-scope root.
    ///
    /// Called by the runner when the user approves a scope expansion. Takes a
    /// write lock briefly (never held across `.await`).
    pub fn allow(&self, path: impl AsRef<Path>) {
        let normalized = normalize_path(path.as_ref());
        let mut roots = self.allowed.write().unwrap_or_else(PoisonError::into_inner);
        if !roots.contains(&normalized) {
            roots.push(normalized);
        }
    }

    /// Decide the guard action for a tool call, synchronously.
    ///
    /// Pulled out of `pre_tool` so the `RwLock` is never held across an
    /// `.await` (the returned future must be `Send`). Returns `Allow` for
    /// non-mutating tools, mutating tools with no parseable path, an empty
    /// allowlist, or in-scope paths; `Deny` otherwise.
    fn decide(&self, call: &ToolCall) -> GuardAction {
        if !MUTATING_TOOLS.contains(&call.name.as_str()) {
            return GuardAction::Allow;
        }
        let Some(path) = extract_path(&call.input) else {
            // Mutating call without a parseable path (e.g. `patch`) → allow.
            return GuardAction::Allow;
        };
        let target = normalize_path(&path);

        let roots = self.allowed.read().unwrap_or_else(PoisonError::into_inner);
        // Unseeded guard → allow everything.
        if roots.is_empty() {
            return GuardAction::Allow;
        }
        // Component-wise containment: equals-root OR descendant-of-root.
        let in_scope = roots.iter().any(|root| target.starts_with(root));
        if in_scope {
            return GuardAction::Allow;
        }

        let roots_display = roots
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        GuardAction::deny(format!(
            "scope guard: '{}' is outside the current task scope ([{}]). \
             If this change is genuinely needed, ask the user to expand the scope \
             (question tool) or add it to the plan first.",
            target.display(),
            roots_display,
        ))
    }
}

impl Guardrail for ScopeGuard {
    fn name(&self) -> &str {
        "scope_guard"
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        // Decide synchronously (lock not held across the await).
        let action = self.decide(call);
        Box::pin(async move { Ok(action) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mutating_call(tool: &str, file_path: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: tool.into(),
            input: json!({ "file_path": file_path }),
        }
    }

    #[tokio::test]
    async fn out_of_scope_mutation_is_denied() {
        let guard = ScopeGuard::new(vec![PathBuf::from("/tmp/x/src/a.rs")]);
        let call = mutating_call("edit", "/tmp/x/src/b.rs");
        let action = guard.pre_tool(&call).await.unwrap();
        match action {
            GuardAction::Deny { reason } => {
                assert!(reason.contains("/tmp/x/src/b.rs"), "path named: {reason}");
                assert!(reason.contains("scope guard"), "actionable: {reason}");
            }
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn in_scope_mutation_allowed() {
        let guard = ScopeGuard::new(vec![PathBuf::from("/tmp/x/src/a.rs")]);
        let call = mutating_call("edit", "/tmp/x/src/a.rs");
        let action = guard.pre_tool(&call).await.unwrap();
        assert!(matches!(action, GuardAction::Allow));
    }

    #[tokio::test]
    async fn directory_root_allows_descendants() {
        let guard = ScopeGuard::new(vec![PathBuf::from("/tmp/x/src")]);
        let call = mutating_call("write", "/tmp/x/src/new.rs");
        let action = guard.pre_tool(&call).await.unwrap();
        assert!(matches!(action, GuardAction::Allow));
    }

    #[tokio::test]
    async fn sibling_prefix_is_not_in_scope() {
        // String-prefix would falsely allow `/tmp/x/src-other`; component-wise
        // `starts_with` correctly denies it.
        let guard = ScopeGuard::new(vec![PathBuf::from("/tmp/x/src")]);
        let call = mutating_call("write", "/tmp/x/src-other/b.rs");
        let action = guard.pre_tool(&call).await.unwrap();
        match action {
            GuardAction::Deny { reason } => {
                assert!(reason.contains("/tmp/x/src-other/b.rs"), "{reason}");
            }
            other => panic!("expected Deny for sibling-prefix path, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_mutating_tools_pass() {
        let guard = ScopeGuard::new(vec![PathBuf::from("/tmp/x/src/a.rs")]);
        for tool in ["read", "grep", "bash"] {
            let call = ToolCall {
                id: "c1".into(),
                name: tool.into(),
                // read/grep anywhere, bash with a path-shaped arg — all allowed.
                input: json!({ "file_path": "/etc/passwd", "path": "/anywhere" }),
            };
            let action = guard.pre_tool(&call).await.unwrap();
            assert!(
                matches!(action, GuardAction::Allow),
                "non-mutating tool {tool} should pass"
            );
        }
    }

    #[tokio::test]
    async fn patch_has_no_parseable_path_allowed() {
        // `patch` is mutating but its path lives inside `patch_text`, not a
        // top-level field → no parseable path → allow (even out-of-tree).
        let guard = ScopeGuard::new(vec![PathBuf::from("/tmp/x/src")]);
        let call = ToolCall {
            id: "c1".into(),
            name: "patch".into(),
            input: json!({ "patch_text": "--- a/etc/passwd\n+++ b/etc/passwd\n" }),
        };
        let action = guard.pre_tool(&call).await.unwrap();
        assert!(matches!(action, GuardAction::Allow));
    }

    #[tokio::test]
    async fn empty_allowlist_allows_all() {
        let guard = ScopeGuard::new(vec![]);
        let call = mutating_call("edit", "/anywhere/at/all.rs");
        let action = guard.pre_tool(&call).await.unwrap();
        assert!(matches!(action, GuardAction::Allow));
    }

    #[tokio::test]
    async fn allow_extends_at_runtime() {
        let guard = ScopeGuard::new(vec![PathBuf::from("/tmp/x/src/a.rs")]);
        let call = mutating_call("edit", "/tmp/x/other/c.rs");

        // Initially out of scope → denied.
        assert!(guard.pre_tool(&call).await.unwrap().is_denied());

        // User approves a scope expansion.
        guard.allow("/tmp/x/other");

        // Now in scope → allowed.
        let action = guard.pre_tool(&call).await.unwrap();
        assert!(matches!(action, GuardAction::Allow));
    }

    #[tokio::test]
    async fn dotdot_in_path_is_normalized_before_check() {
        // `/tmp/x/src/../b.rs` lexically resolves to `/tmp/x/b.rs`, outside
        // the `/tmp/x/src` root → denied. Pins the lexical normalization.
        let guard = ScopeGuard::new(vec![PathBuf::from("/tmp/x/src")]);
        let call = mutating_call("edit", "/tmp/x/src/../b.rs");
        let action = guard.pre_tool(&call).await.unwrap();
        match action {
            GuardAction::Deny { reason } => {
                assert!(reason.contains("/tmp/x/b.rs"), "normalized path: {reason}");
            }
            other => panic!("expected Deny after normalization, got {other:?}"),
        }
    }

    #[test]
    fn guard_name() {
        let guard = ScopeGuard::new(vec![]);
        assert_eq!(guard.name(), "scope_guard");
    }
}
