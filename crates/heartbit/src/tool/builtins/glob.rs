use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const MAX_RESULTS: usize = 100;

pub struct GlobTool;

impl GlobTool {
    pub fn new() -> Self {
        Self
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

            let base_path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

            let base = PathBuf::from(base_path);
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
        let tool = GlobTool::new();
        assert_eq!(tool.definition().name, "glob");
    }

    #[tokio::test]
    async fn glob_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();

        let tool = GlobTool::new();
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

        let tool = GlobTool::new();
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

        let tool = GlobTool::new();
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

        let tool = GlobTool::new();
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
        let tool = GlobTool::new();
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
