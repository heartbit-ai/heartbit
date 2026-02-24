use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const MAX_MATCHES: usize = 100;

pub struct GrepTool {
    workspace: Option<PathBuf>,
}

impl GrepTool {
    pub fn new(workspace: Option<PathBuf>) -> Self {
        Self { workspace }
    }
}

impl Tool for GrepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".into(),
            description: "Search file contents using regex patterns. Uses ripgrep (rg) when \
                          available, falls back to built-in regex search. Returns matching lines \
                          with file paths and line numbers."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: current directory)"
                    },
                    "include": {
                        "type": "string",
                        "description": "File glob pattern to filter (e.g. \"*.rs\", \"*.py\")"
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "Treat pattern as literal string (default: false)"
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

            let path_str = input.get("path").and_then(|v| v.as_str());

            let include = input.get("include").and_then(|v| v.as_str());
            let literal = input
                .get("literal")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let search_path = match path_str {
                Some(p) => super::resolve_path(p, self.workspace.as_deref()),
                None => self.workspace.clone().unwrap_or_else(|| PathBuf::from(".")),
            };
            let path = search_path.display().to_string();
            if !search_path.exists() {
                return Ok(ToolOutput::error(format!("Path not found: {path}")));
            }

            // Try ripgrep first
            match try_ripgrep(pattern, &path, include, literal).await {
                Ok(output) => Ok(output),
                Err(_) => {
                    // Fallback to built-in regex search (sync IO, run on blocking thread)
                    let pattern = pattern.to_string();
                    let include = include.map(String::from);
                    tokio::task::spawn_blocking(move || {
                        fallback_grep(&pattern, &search_path, include.as_deref(), literal)
                    })
                    .await
                    .map_err(|e| Error::Agent(format!("Grep task failed: {e}")))?
                }
            }
        })
    }
}

async fn try_ripgrep(
    pattern: &str,
    path: &str,
    include: Option<&str>,
    literal: bool,
) -> Result<ToolOutput, Error> {
    let mut cmd = tokio::process::Command::new("rg");
    cmd.arg("-H")
        .arg("-n")
        .arg("--color")
        .arg("never")
        // Cap output at source to avoid buffering unbounded results
        .arg("--max-count")
        .arg((MAX_MATCHES + 1).to_string());

    if literal {
        cmd.arg("-F");
    }

    if let Some(glob_pattern) = include {
        cmd.arg("--glob").arg(glob_pattern);
    }

    cmd.arg(pattern).arg(path);

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let output = cmd
        .output()
        .await
        .map_err(|e| Error::Agent(format!("rg not available: {e}")))?;

    // rg exit code 0 = matches found, 1 = no matches, 2 = error
    match output.status.code() {
        Some(0) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Single-pass: collect first MAX_MATCHES lines and count total
            let mut lines = Vec::with_capacity(MAX_MATCHES);
            let mut total = 0;
            for line in stdout.lines() {
                total += 1;
                if lines.len() < MAX_MATCHES {
                    lines.push(line);
                }
            }
            let truncated = if total > MAX_MATCHES {
                format!("\n\n(showing first {MAX_MATCHES} of {total} matches)")
            } else {
                String::new()
            };
            Ok(ToolOutput::success(format!(
                "Found {} matches\n\n{}{}",
                lines.len(),
                lines.join("\n"),
                truncated,
            )))
        }
        Some(1) => Ok(ToolOutput::success("No matches found.")),
        _ => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(Error::Agent(format!("rg error: {stderr}")))
        }
    }
}

fn fallback_grep(
    pattern: &str,
    path: &Path,
    include: Option<&str>,
    literal: bool,
) -> Result<ToolOutput, Error> {
    let re_pattern = if literal {
        regex::escape(pattern)
    } else {
        pattern.to_string()
    };

    let re = regex::Regex::new(&re_pattern)
        .map_err(|e| Error::Agent(format!("Invalid regex pattern: {e}")))?;

    let include_pattern = include
        .map(glob::Pattern::new)
        .transpose()
        .map_err(|e| Error::Agent(format!("Invalid include pattern: {e}")))?;

    let mut matches = Vec::new();

    let walker: Box<dyn Iterator<Item = walkdir::DirEntry>> = if path.is_file() {
        Box::new(
            walkdir::WalkDir::new(path)
                .into_iter()
                .filter_map(|e| e.ok()),
        )
    } else {
        Box::new(
            walkdir::WalkDir::new(path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| !is_hidden(e))
                .filter(|e| e.file_type().is_file()),
        )
    };

    for entry in walker {
        if !entry.file_type().is_file() {
            continue;
        }

        if let Some(ref ip) = include_pattern {
            let name = entry.file_name().to_str().unwrap_or("");
            // Match against filename first, then relative path (consistent with rg --glob)
            if !ip.matches(name) {
                let rel = entry
                    .path()
                    .strip_prefix(path)
                    .unwrap_or(entry.path())
                    .to_str()
                    .unwrap_or("");
                if !ip.matches(rel) {
                    continue;
                }
            }
        }

        let file_path = entry.path();
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue, // Skip binary/unreadable files
        };

        for (line_num, line) in content.lines().enumerate() {
            if re.is_match(line) {
                matches.push(format!(
                    "{}:{}: {}",
                    file_path.display(),
                    line_num + 1,
                    line
                ));
                if matches.len() >= MAX_MATCHES {
                    break;
                }
            }
        }

        if matches.len() >= MAX_MATCHES {
            break;
        }
    }

    if matches.is_empty() {
        Ok(ToolOutput::success("No matches found."))
    } else {
        let count = matches.len();
        let truncated = if count >= MAX_MATCHES {
            format!("\n\n(showing first {MAX_MATCHES} matches, there may be more)")
        } else {
            String::new()
        };
        Ok(ToolOutput::success(format!(
            "Found {count} matches\n\n{}{}",
            matches.join("\n"),
            truncated,
        )))
    }
}

fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|s| s.starts_with('.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = GrepTool::new(None);
        assert_eq!(tool.definition().name, "grep");
    }

    #[tokio::test]
    async fn grep_finds_pattern_in_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world\nfoo bar\nhello again\n").unwrap();

        let tool = GrepTool::new(None);
        let result = tool
            .execute(json!({
                "pattern": "hello",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
        assert!(result.content.contains("Found"));
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world\n").unwrap();

        let tool = GrepTool::new(None);
        let result = tool
            .execute(json!({
                "pattern": "xyz_not_here",
                "path": path.to_str().unwrap()
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("No matches"));
    }

    #[tokio::test]
    async fn grep_literal_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "price is $5.00\nnot a regex\n").unwrap();

        let tool = GrepTool::new(None);
        let result = tool
            .execute(json!({
                "pattern": "$5.00",
                "path": path.to_str().unwrap(),
                "literal": true
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("$5.00"));
    }

    #[tokio::test]
    async fn grep_nonexistent_path() {
        let tool = GrepTool::new(None);
        let result = tool
            .execute(json!({
                "pattern": "test",
                "path": "/tmp/nonexistent_heartbit_test_dir_12345"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn grep_include_filter() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("match.rs"), "fn hello() {}\n").unwrap();
        std::fs::write(dir.path().join("skip.txt"), "fn hello() {}\n").unwrap();

        let tool = GrepTool::new(None);
        let result = tool
            .execute(json!({
                "pattern": "hello",
                "path": dir.path().to_str().unwrap(),
                "include": "*.rs"
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("match.rs"));
    }

    #[tokio::test]
    async fn grep_include_path_pattern() {
        // Include pattern with a directory component should match relative paths.
        // This uses the fallback grep (not rg) by creating a specific structure.
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("src");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("main.rs"), "fn match_me() {}\n").unwrap();
        std::fs::write(dir.path().join("root.rs"), "fn match_me() {}\n").unwrap();

        // Test via fallback_grep directly
        let result = super::fallback_grep("match_me", dir.path(), Some("src/*.rs"), false).unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("main.rs"), "{}", result.content);
        assert!(
            !result.content.contains("root.rs"),
            "root.rs should not match src/*.rs: {}",
            result.content
        );
    }
}
