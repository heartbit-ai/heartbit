//! `repo_inspect` builtin — reads files and greps within a constrained
//! subset of the heartbit repo. Backs the `repo_researcher` agent for
//! the heartbit-rs:x persona.

use heartbit_core::Error;
use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::Deserialize;
use serde_json::Value;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

#[derive(Debug, Deserialize)]
#[serde(tag = "op")]
enum Op {
    #[serde(rename = "read_file")]
    ReadFile {
        path: String,
        range: Option<(usize, usize)>,
    },
    #[serde(rename = "grep_repo")]
    GrepRepo {
        pattern: String,
        glob: Option<String>,
    },
}

/// Tool that lets the `repo_researcher` agent read files and `git grep`
/// inside the heartbit repo, restricted to `crates/heartbit-core` and
/// `crates/heartbit-cli`.
pub struct RepoInspectTool {
    repo_root: PathBuf,
    allowed_prefixes: Vec<PathBuf>,
    max_file_lines: usize,
    max_grep_hits: usize,
}

impl RepoInspectTool {
    /// Build the tool, anchoring it at the canonicalized `repo_root` and
    /// scoping reads/greps to `crates/heartbit-core` and `crates/heartbit-cli`.
    ///
    /// Returns an error if `repo_root` cannot be canonicalized.
    pub fn new(repo_root: impl Into<PathBuf>) -> Result<Self, Error> {
        let repo_root = repo_root
            .into()
            .canonicalize()
            .map_err(|e| Error::Agent(format!("repo_root canonicalize: {e}")))?;
        Ok(Self {
            allowed_prefixes: vec![
                repo_root.join("crates/heartbit-core"),
                repo_root.join("crates/heartbit-cli"),
            ],
            repo_root,
            max_file_lines: 1000,
            max_grep_hits: 100,
        })
    }

    fn resolve_within_allowed(&self, path: &str) -> Result<PathBuf, String> {
        if path.starts_with('/') {
            return Err(format!("absolute paths are not allowed: {path}"));
        }
        let candidate = self.repo_root.join(path);
        let canonical = candidate
            .canonicalize()
            .map_err(|e| format!("path resolve: {path}: {e}"))?;
        if !self
            .allowed_prefixes
            .iter()
            .any(|p| canonical.starts_with(p))
        {
            return Err(format!(
                "path {path} resolves outside the allowed prefixes (heartbit-core / heartbit-cli)"
            ));
        }
        Ok(canonical)
    }
}

impl Tool for RepoInspectTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "repo_inspect".into(),
            description: "Read or grep files inside the heartbit repo, restricted to \
                          crates/heartbit-core and crates/heartbit-cli."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["read_file", "grep_repo"]},
                    "path": {"type": "string", "description": "relative path from repo root (read_file)"},
                    "range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[start_line, end_line] 1-based inclusive (read_file, optional)"
                    },
                    "pattern": {"type": "string", "description": "regex pattern (grep_repo)"},
                    "glob": {"type": "string", "description": "optional file glob (grep_repo)"}
                },
                "required": ["op"]
            }),
        }
    }

    fn execute<'a>(
        &'a self,
        _ctx: &ExecutionContext,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>> {
        Box::pin(async move {
            let op: Op = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("repo_inspect input: {e}"))),
            };
            match op {
                Op::ReadFile { path, range } => self.do_read_file(&path, range).await,
                Op::GrepRepo { pattern, glob } => {
                    self.do_grep_repo(&pattern, glob.as_deref()).await
                }
            }
        })
    }
}

impl RepoInspectTool {
    async fn do_read_file(
        &self,
        path: &str,
        range: Option<(usize, usize)>,
    ) -> Result<ToolOutput, Error> {
        let resolved = match self.resolve_within_allowed(path) {
            Ok(p) => p,
            Err(msg) => return Ok(ToolOutput::error(msg)),
        };
        let text = tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|e| Error::Agent(format!("read_file({path}): {e}")))?;
        let all_lines: Vec<&str> = text.lines().collect();
        let (start, end) = match range {
            Some((s, e)) => (s.max(1), e.min(all_lines.len())),
            None => (1, all_lines.len()),
        };
        if end < start {
            return Ok(ToolOutput::error(format!(
                "range start ({start}) > end ({end}) for {path}"
            )));
        }
        let span = end - start + 1;
        if span > self.max_file_lines {
            return Ok(ToolOutput::error(format!(
                "requested {span} lines from {path}; max is {} — pass an explicit range",
                self.max_file_lines
            )));
        }
        let mut out = String::new();
        for (i, line) in all_lines.iter().enumerate().take(end).skip(start - 1) {
            out.push_str(&format!("{}: {}\n", i + 1, line));
        }
        Ok(ToolOutput::success(out))
    }

    async fn do_grep_repo(&self, pattern: &str, glob: Option<&str>) -> Result<ToolOutput, Error> {
        // Use git grep for .gitignore-respecting search restricted to
        // allowed prefixes. The :(top) prefix scopes pathspecs to repo
        // root (independent of current cwd inside).
        let mut cmd = tokio::process::Command::new("git");
        cmd.current_dir(&self.repo_root);
        cmd.arg("grep").arg("-n").arg("-e").arg(pattern);
        if let Some(g) = glob {
            cmd.arg("--")
                .arg(g)
                .arg(":(top)crates/heartbit-core")
                .arg(":(top)crates/heartbit-cli");
        } else {
            cmd.arg("--")
                .arg(":(top)crates/heartbit-core")
                .arg(":(top)crates/heartbit-cli");
        }
        let output = cmd
            .output()
            .await
            .map_err(|e| Error::Agent(format!("git grep: {e}")))?;
        // git grep returns 1 on no-match — treat as empty result, not error.
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        if stdout.is_empty() {
            return Ok(ToolOutput::success(format!("(no matches for {pattern})")));
        }
        let total_lines = stdout.lines().count();
        let lines: Vec<&str> = stdout.lines().take(self.max_grep_hits).collect();
        let truncated = total_lines > self.max_grep_hits;
        let mut out = lines.join("\n");
        if truncated {
            out.push_str(&format!(
                "\n... ({} more hits truncated; cap is {})",
                total_lines - self.max_grep_hits,
                self.max_grep_hits
            ));
        }
        Ok(ToolOutput::success(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_repo() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        // Mimic the real repo layout
        let core_dir = tmp.path().join("crates/heartbit-core/src");
        let cli_dir = tmp.path().join("crates/heartbit-cli/src");
        let other_dir = tmp.path().join("crates/heartbit-other/src");
        std::fs::create_dir_all(&core_dir).unwrap();
        std::fs::create_dir_all(&cli_dir).unwrap();
        std::fs::create_dir_all(&other_dir).unwrap();
        std::fs::write(
            core_dir.join("lib.rs"),
            "pub trait Tool {}\npub fn hello() {}\n",
        )
        .unwrap();
        std::fs::write(cli_dir.join("main.rs"), "fn main() { println!(\"hi\"); }\n").unwrap();
        std::fs::write(other_dir.join("lib.rs"), "pub fn out_of_scope() {}\n").unwrap();
        // Init as a git repo so git grep works.
        let _ = std::process::Command::new("git")
            .args(["init"])
            .current_dir(tmp.path())
            .output();
        let _ = std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(tmp.path())
            .output();
        let _ = std::process::Command::new("git")
            .args([
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-m",
                "init",
            ])
            .current_dir(tmp.path())
            .output();
        tmp
    }

    #[tokio::test]
    async fn read_file_returns_lines_with_numbers() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"op": "read_file", "path": "crates/heartbit-core/src/lib.rs"}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("1: pub trait Tool"));
        assert!(out.content.contains("2: pub fn hello"));
    }

    #[tokio::test]
    async fn read_file_respects_range() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"op": "read_file", "path": "crates/heartbit-core/src/lib.rs", "range": [2, 2]}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("2: pub fn hello"));
        assert!(!out.content.contains("1: "));
    }

    #[tokio::test]
    async fn read_file_rejects_path_outside_allowed_prefixes() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"op": "read_file", "path": "crates/heartbit-other/src/lib.rs"}),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("outside the allowed prefixes"));
    }

    #[tokio::test]
    async fn read_file_rejects_absolute_path() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"op": "read_file", "path": "/etc/passwd"}),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("absolute paths are not allowed"));
    }

    #[tokio::test]
    async fn grep_repo_finds_matches_in_core_and_cli_only() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"op": "grep_repo", "pattern": "pub fn"}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("crates/heartbit-core/src/lib.rs"));
        assert!(!out.content.contains("crates/heartbit-other"));
    }

    #[tokio::test]
    async fn grep_repo_returns_no_match_message_for_empty_result() {
        let tmp = fixture_repo();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"op": "grep_repo", "pattern": "nonexistent_xyzzy"}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("no matches"));
    }
}
