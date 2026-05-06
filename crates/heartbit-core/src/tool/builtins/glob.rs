use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::sandbox::CorePathPolicy;
use crate::tool::{Tool, ToolOutput};

const MAX_RESULTS: usize = 100;

/// Builtin tool that finds files matching a glob pattern.
///
/// Walks the workspace (or the current working directory when no workspace is
/// set) and returns up to `MAX_RESULTS = 100` paths that match the given pattern
/// (e.g., `**/*.rs`). Results are sorted lexicographically. Protected paths are
/// filtered from the output so the agent cannot enumerate sensitive locations.
///
/// SECURITY (F-FS-4): when a `CorePathPolicy` is set, every result path is
/// checked against it before being included; results outside the allowed
/// directories (or matching deny-globs) are silently dropped.
pub struct GlobTool {
    workspace: Option<PathBuf>,
    protected_paths: Arc<Vec<PathBuf>>,
    path_policy: Option<Arc<CorePathPolicy>>,
}

impl GlobTool {
    pub fn new(workspace: Option<PathBuf>, protected_paths: Arc<Vec<PathBuf>>) -> Self {
        Self {
            workspace,
            protected_paths,
            path_policy: None,
        }
    }

    /// Set a `CorePathPolicy` that filters glob results.
    pub fn with_path_policy(mut self, policy: Arc<CorePathPolicy>) -> Self {
        self.path_policy = Some(policy);
        self
    }
}

impl Tool for GlobTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob".into(),
            description: "Find files matching a glob pattern. Returns file paths sorted by \
                          path length (shortest first). Skips hidden files."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match (e.g. \"**/*.rs\", \"src/**/*.ts\")"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search in (default: current directory)"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let pattern = input
                .get("pattern")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("pattern is required".into()))?;

            let base_path_str = input.get("path").and_then(|v| v.as_str());

            let base = match base_path_str {
                Some(p) => {
                    match super::resolve_path(p, self.workspace.as_deref(), &self.protected_paths) {
                        Ok(p) => p,
                        Err(msg) => return Ok(ToolOutput::error(msg)),
                    }
                }
                None => self.workspace.clone().unwrap_or_else(|| PathBuf::from(".")),
            };
            let base_path = base.display().to_string();
            if !base.exists() {
                return Ok(ToolOutput::error(format!("Path not found: {base_path}")));
            }

            // Build the full pattern (escape base_path to prevent metacharacter interpretation)
            let full_pattern = if base.is_dir() {
                let base_str = glob::Pattern::escape(base_path.trim_end_matches('/'));
                format!("{base_str}/{pattern}")
            } else {
                pattern.to_string()
            };

            let entries = glob::glob(&full_pattern)
                .map_err(|e| Error::Agent(format!("Invalid glob pattern: {e}")))?;

            let mut paths: Vec<String> = Vec::new();

            // Workspace is already canonical (from builtin_tools()).
            let ws_ref = self.workspace.as_deref();

            for entry in entries {
                match entry {
                    Ok(path) => {
                        // Skip hidden files/directories (only check relative to base)
                        let relative = path.strip_prefix(&base).unwrap_or(&path);
                        let has_hidden = relative
                            .components()
                            .any(|c| c.as_os_str().to_str().is_some_and(|s| s.starts_with('.')));
                        if has_hidden {
                            continue;
                        }

                        // Symlink post-filter: only canonicalize actual symlinks
                        // to avoid O(results) syscalls for regular files.
                        if let Some(ws) = ws_ref
                            && path
                                .symlink_metadata()
                                .is_ok_and(|m| m.file_type().is_symlink())
                            && let Ok(canonical) = path.canonicalize()
                            && !canonical.starts_with(ws)
                        {
                            continue;
                        }

                        // SECURITY (F-FS-4): apply the path policy to every
                        // result. Without this, glob can enumerate arbitrary
                        // directories outside the workspace when the LLM
                        // provides an absolute `path` and no workspace is set.
                        if let Some(ref policy) = self.path_policy
                            && policy.check_path(&path).is_err()
                        {
                            continue;
                        }

                        // Reject any symlink whose target lies outside the
                        // path policy, even when no workspace is configured.
                        if path
                            .symlink_metadata()
                            .is_ok_and(|m| m.file_type().is_symlink())
                            && let Ok(canonical) = path.canonicalize()
                            && let Some(ref policy) = self.path_policy
                            && policy.check_path(&canonical).is_err()
                        {
                            continue;
                        }

                        // Convert to relative path
                        let display = relative.display().to_string();
                        paths.push(display);

                        if paths.len() >= MAX_RESULTS {
                            break;
                        }
                    }
                    Err(_) => continue, // Skip unreadable entries
                }
            }

            // Sort by path length (shortest first)
            paths.sort_by_key(|p| p.len());

            if paths.is_empty() {
                Ok(ToolOutput::success("No files matched the pattern."))
            } else {
                let count = paths.len();
                let truncated = if count >= MAX_RESULTS {
                    format!("\n\n(Results limited to {MAX_RESULTS} files)")
                } else {
                    String::new()
                };
                Ok(ToolOutput::success(format!(
                    "{}{truncated}",
                    paths.join("\n")
                )))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = GlobTool::new(None, Arc::new(Vec::new()));
        assert_eq!(tool.definition().name, "glob");
    }

    #[tokio::test]
    async fn glob_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();

        let tool = GlobTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(json!({
                "pattern": "*.rs",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("a.rs"));
        assert!(result.content.contains("b.rs"));
        assert!(!result.content.contains("c.txt"));
    }

    #[tokio::test]
    async fn glob_recursive_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(dir.path().join("top.rs"), "").unwrap();
        std::fs::write(sub.join("nested.rs"), "").unwrap();

        let tool = GlobTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(json!({
                "pattern": "**/*.rs",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("top.rs"));
        assert!(result.content.contains("nested.rs"));
    }

    #[tokio::test]
    async fn glob_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();

        let tool = GlobTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(json!({
                "pattern": "*.xyz",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("No files matched"));
    }

    #[tokio::test]
    async fn glob_skips_hidden() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("visible.rs"), "").unwrap();
        std::fs::write(dir.path().join(".hidden.rs"), "").unwrap();

        let tool = GlobTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(json!({
                "pattern": "*.rs",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("visible.rs"));
        assert!(!result.content.contains(".hidden.rs"));
    }

    #[tokio::test]
    async fn glob_nonexistent_path() {
        let tool = GlobTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(json!({
                "pattern": "*.rs",
                "path": "/tmp/nonexistent_heartbit_test_dir_12345"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }
}
