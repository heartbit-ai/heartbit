use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const MAX_ENTRIES: usize = 1000;
const MAX_DEPTH: usize = 20;

/// Default directories/files to always skip.
const DEFAULT_IGNORES: &[&str] = &[
    "node_modules",
    "dist",
    "build",
    ".git",
    "target",
    "__pycache__",
    ".DS_Store",
    "*.pyc",
    "*.o",
    "*.so",
    "*.dylib",
];

pub struct ListTool;

impl ListTool {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for ListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list".into(),
            description: "List directory contents as an indented tree. Skips hidden files, \
                          node_modules, .git, target, and other common build artifacts. \
                          Maximum 1000 entries."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: current directory)"
                    },
                    "ignore": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional glob patterns to ignore"
                    }
                }
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

            let user_ignores: Vec<String> = input
                .get("ignore")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let root = PathBuf::from(path);
            if !root.exists() {
                return Ok(ToolOutput::error(format!("Path not found: {path}")));
            }
            if !root.is_dir() {
                return Ok(ToolOutput::error(format!("{path} is not a directory")));
            }

            // Compile ignore patterns
            let mut ignore_patterns: Vec<glob::Pattern> = Vec::new();
            for pat in DEFAULT_IGNORES
                .iter()
                .copied()
                .chain(user_ignores.iter().map(|s| s.as_str()))
            {
                if let Ok(p) = glob::Pattern::new(pat) {
                    ignore_patterns.push(p);
                }
            }

            let output = tokio::task::spawn_blocking(move || {
                let mut buf = String::new();
                let mut count = 0;
                build_tree(&root, &ignore_patterns, &mut buf, &mut count, 0);
                if count >= MAX_ENTRIES {
                    buf.push_str(&format!("\n(Listing truncated at {MAX_ENTRIES} entries)"));
                }
                buf
            })
            .await
            .map_err(|e| Error::Agent(format!("List task failed: {e}")))?;

            Ok(ToolOutput::success(output))
        })
    }
}

fn build_tree(
    dir: &Path,
    ignore_patterns: &[glob::Pattern],
    output: &mut String,
    count: &mut usize,
    depth: usize,
) {
    if *count >= MAX_ENTRIES || depth >= MAX_DEPTH {
        return;
    }

    let mut entries: Vec<std::fs::DirEntry> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
        Err(_) => return,
    };

    // Sort: directories first, then alphabetical
    entries.sort_by(|a, b| {
        let a_dir = a.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let b_dir = b.file_type().map(|t| t.is_dir()).unwrap_or(false);
        b_dir
            .cmp(&a_dir)
            .then_with(|| a.file_name().cmp(&b.file_name()))
    });

    let indent = "  ".repeat(depth);

    for entry in entries {
        if *count >= MAX_ENTRIES {
            return;
        }

        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };

        // Skip hidden files
        if name_str.starts_with('.') {
            continue;
        }

        // Check ignore patterns
        if ignore_patterns.iter().any(|p| p.matches(name_str)) {
            continue;
        }

        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        *count += 1;

        if is_dir {
            output.push_str(&format!("{indent}- {name_str}/\n"));
            build_tree(&entry.path(), ignore_patterns, output, count, depth + 1);
        } else {
            output.push_str(&format!("{indent}- {name_str}\n"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = ListTool::new();
        assert_eq!(tool.definition().name, "list");
    }

    #[tokio::test]
    async fn list_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("c.rs"), "").unwrap();

        let tool = ListTool::new();
        let result = tool
            .execute(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("sub/"));
        assert!(result.content.contains("a.rs"));
        assert!(result.content.contains("b.txt"));
        assert!(result.content.contains("c.rs"));
    }

    #[tokio::test]
    async fn list_skips_hidden_and_defaults() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("visible.rs"), "").unwrap();
        std::fs::write(dir.path().join(".hidden"), "").unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir(dir.path().join("node_modules")).unwrap();

        let tool = ListTool::new();
        let result = tool
            .execute(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("visible.rs"));
        assert!(!result.content.contains(".hidden"));
        assert!(!result.content.contains(".git"));
        assert!(!result.content.contains("node_modules"));
    }

    #[tokio::test]
    async fn list_with_custom_ignore() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("keep.rs"), "").unwrap();
        std::fs::write(dir.path().join("skip.log"), "").unwrap();

        let tool = ListTool::new();
        let result = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap(),
                "ignore": ["*.log"]
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("keep.rs"));
        assert!(!result.content.contains("skip.log"));
    }

    #[tokio::test]
    async fn list_nonexistent_path() {
        let tool = ListTool::new();
        let result = tool
            .execute(json!({"path": "/tmp/nonexistent_heartbit_test_dir_12345"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn list_file_not_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "content").unwrap();

        let tool = ListTool::new();
        let result = tool
            .execute(json!({"path": path.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not a directory"));
    }

    #[tokio::test]
    async fn list_directories_first() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("aaa_file.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("zzz_dir")).unwrap();

        let tool = ListTool::new();
        let result = tool
            .execute(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        // Directory should come before file despite alphabetical ordering
        let dir_pos = result.content.find("zzz_dir/").unwrap();
        let file_pos = result.content.find("aaa_file.txt").unwrap();
        assert!(
            dir_pos < file_pos,
            "Directories should be listed before files"
        );
    }
}
