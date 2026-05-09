//! `repo_inspect` builtin — reads files and greps within a constrained
//! subset of the heartbit repo. Backs the `repo_researcher` agent for
//! the heartbit-rs:x persona.

use heartbit_core::Error;
use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::Deserialize;
use serde_json::Value;
use std::future::Future;
use std::path::{Path, PathBuf};
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
    #[serde(rename = "list_features")]
    ListFeatures,
    #[serde(rename = "feature_demo")]
    FeatureDemo { name: String },
}

#[derive(Debug, Deserialize, Clone)]
struct FeatureMenu {
    #[serde(default = "default_menu_version")]
    #[allow(dead_code)]
    pub version: u32,
    pub feature: Vec<FeatureEntry>,
}

fn default_menu_version() -> u32 {
    1
}

#[derive(Debug, Deserialize, Clone)]
struct FeatureEntry {
    pub name: String,
    pub description: String,
    pub canonical_file: String,
    pub key_types: Vec<String>,
    pub payoff: String,
}

impl FeatureMenu {
    fn load(repo_root: &Path) -> Option<Self> {
        let path = repo_root.join("crates/heartbit-ghost/data/heartbit-rs-features.toml");
        let text = std::fs::read_to_string(&path).ok()?;
        toml::from_str(&text).ok()
    }
}

/// Tool that lets the `repo_researcher` agent read files and `git grep`
/// inside the heartbit repo, restricted to `crates/heartbit-core` and
/// `crates/heartbit-cli`.
pub struct RepoInspectTool {
    repo_root: PathBuf,
    allowed_prefixes: Vec<PathBuf>,
    max_file_lines: usize,
    max_grep_hits: usize,
    menu: Option<FeatureMenu>,
}

impl RepoInspectTool {
    /// Build the tool, anchoring it at the canonicalized `repo_root` and
    /// scoping reads/greps to `crates/heartbit-core` and `crates/heartbit-cli`.
    ///
    /// Returns an error if `repo_root` or either allowed prefix cannot be
    /// canonicalized (e.g. the directory does not exist). Both sides are
    /// canonicalized so `starts_with` comparisons remain correct in the
    /// presence of symlinks anywhere in the path.
    pub fn new(repo_root: impl Into<PathBuf>) -> Result<Self, Error> {
        let repo_root = repo_root
            .into()
            .canonicalize()
            .map_err(|e| Error::Agent(format!("repo_root canonicalize: {e}")))?;
        let allowed_prefixes = vec![
            repo_root.join("crates/heartbit-core"),
            repo_root.join("crates/heartbit-cli"),
        ]
        .into_iter()
        .map(|p| {
            p.canonicalize()
                .map_err(|e| Error::Agent(format!("allowed prefix canonicalize: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
        let menu = FeatureMenu::load(&repo_root);
        Ok(Self {
            repo_root,
            allowed_prefixes,
            max_file_lines: 1000,
            max_grep_hits: 100,
            menu,
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
                    "op": {"type": "string", "enum": ["read_file", "grep_repo", "list_features", "feature_demo"]},
                    "path": {"type": "string", "description": "relative path from repo root (read_file)"},
                    "range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[start_line, end_line] 1-based inclusive (read_file, optional)"
                    },
                    "pattern": {"type": "string", "description": "regex pattern (grep_repo)"},
                    "glob": {"type": "string", "description": "optional file glob (grep_repo)"},
                    "name": {"type": "string", "description": "feature name (feature_demo)"}
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
                Op::ListFeatures => self.do_list_features().await,
                Op::FeatureDemo { name } => self.do_feature_demo(&name).await,
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
        // git grep exit codes: 0 = matches found, 1 = no matches, anything
        // else (typically 128) = hard error such as bad regex, not a git
        // repo, or invalid pathspec — those also produce empty stdout, so
        // distinguishing on exit code is required to avoid silently
        // masking real failures as "no matches".
        match output.status.code() {
            Some(0) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
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
            Some(1) => Ok(ToolOutput::success(format!("(no matches for {pattern})"))),
            other => {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let code = other
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".into());
                Ok(ToolOutput::error(format!(
                    "git grep failed (exit {code}): {stderr}"
                )))
            }
        }
    }

    async fn do_list_features(&self) -> Result<ToolOutput, Error> {
        let menu = match self.menu.as_ref() {
            Some(m) => m,
            None => {
                return Ok(ToolOutput::error(
                    "feature menu not loaded — \
                     crates/heartbit-ghost/data/heartbit-rs-features.toml is missing"
                        .to_string(),
                ));
            }
        };
        let mut out = String::new();
        for f in &menu.feature {
            out.push_str(&format!("- {}: {} ({})\n", f.name, f.description, f.payoff));
        }
        Ok(ToolOutput::success(out))
    }

    async fn do_feature_demo(&self, name: &str) -> Result<ToolOutput, Error> {
        let menu = match self.menu.as_ref() {
            Some(m) => m,
            None => {
                return Ok(ToolOutput::error(
                    "feature menu not loaded — heartbit-rs-features.toml is missing".to_string(),
                ));
            }
        };
        match menu.feature.iter().find(|f| f.name == name) {
            Some(f) => Ok(ToolOutput::success(format!(
                "name: {}\ndescription: {}\ncanonical_file: {}\nkey_types: {}\npayoff: {}",
                f.name,
                f.description,
                f.canonical_file,
                f.key_types.join(", "),
                f.payoff,
            ))),
            None => Ok(ToolOutput::error(format!("unknown feature: {name}"))),
        }
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
    async fn read_file_rejects_request_exceeding_max_file_lines() {
        let tmp = fixture_repo();
        // Write a large (>1000-line) source file inside an allowed prefix
        // so the requested span actually exceeds max_file_lines (the read
        // path clamps `end` to the file length, so a tiny file would just
        // be returned in full).
        let big_path = tmp.path().join("crates/heartbit-core/src/big.rs");
        let big_contents: String = (0..1500).map(|i| format!("// line {i}\n")).collect();
        std::fs::write(&big_path, &big_contents).unwrap();

        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({
                    "op": "read_file",
                    "path": "crates/heartbit-core/src/big.rs",
                    "range": [1, 1001]
                }),
            )
            .await
            .unwrap();
        assert!(out.is_error, "expected is_error=true, got: {}", out.content);
        assert!(
            out.content.contains("max is 1000") || out.content.contains("explicit range"),
            "unexpected error message: {}",
            out.content
        );
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

    #[tokio::test]
    async fn list_features_returns_menu_entries() {
        let tmp = fixture_repo();
        // Copy the real menu file into the fixture's data/ dir so the
        // tool can find it relative to repo_root.
        let data_dir = tmp.path().join("crates/heartbit-ghost/data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let menu_src = std::env::current_dir()
            .unwrap()
            .join("crates/heartbit-ghost/data/heartbit-rs-features.toml");
        // If running from the workspace root or the crate dir, find the
        // menu wherever it actually is (resilient to test-runner cwd).
        let menu_path = if menu_src.exists() {
            menu_src
        } else {
            std::path::PathBuf::from("data/heartbit-rs-features.toml")
        };
        std::fs::copy(&menu_path, data_dir.join("heartbit-rs-features.toml")).unwrap();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(&ExecutionContext::default(), json!({"op": "list_features"}))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("tool_trait"));
        assert!(out.content.contains("agent_runner"));
        assert!(out.content.contains("memory_trait"));
    }

    #[tokio::test]
    async fn feature_demo_returns_full_record_for_existing_name() {
        let tmp = fixture_repo();
        let data_dir = tmp.path().join("crates/heartbit-ghost/data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let menu_src = std::env::current_dir()
            .unwrap()
            .join("crates/heartbit-ghost/data/heartbit-rs-features.toml");
        let menu_path = if menu_src.exists() {
            menu_src
        } else {
            std::path::PathBuf::from("data/heartbit-rs-features.toml")
        };
        std::fs::copy(&menu_path, data_dir.join("heartbit-rs-features.toml")).unwrap();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"op": "feature_demo", "name": "tool_trait"}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("crates/heartbit-core/src/tool/mod.rs"));
        assert!(out.content.contains("ToolDefinition"));
    }

    #[tokio::test]
    async fn feature_demo_returns_error_for_unknown_name() {
        let tmp = fixture_repo();
        let data_dir = tmp.path().join("crates/heartbit-ghost/data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let menu_src = std::env::current_dir()
            .unwrap()
            .join("crates/heartbit-ghost/data/heartbit-rs-features.toml");
        let menu_path = if menu_src.exists() {
            menu_src
        } else {
            std::path::PathBuf::from("data/heartbit-rs-features.toml")
        };
        std::fs::copy(&menu_path, data_dir.join("heartbit-rs-features.toml")).unwrap();
        let tool = RepoInspectTool::new(tmp.path()).unwrap();
        let out = tool
            .execute(
                &ExecutionContext::default(),
                json!({"op": "feature_demo", "name": "no_such_feature"}),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("unknown feature"));
    }
}
