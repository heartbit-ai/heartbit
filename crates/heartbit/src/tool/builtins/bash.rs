use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Mutex;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const MAX_TIMEOUT_MS: u64 = 600_000;
const MAX_OUTPUT_CHARS: usize = 30_000;
const HEAD_TAIL_SIZE: usize = 14_000;

pub struct BashTool {
    /// Tracked working directory that persists across calls.
    cwd: Mutex<PathBuf>,
}

impl BashTool {
    pub fn new() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        Self {
            cwd: Mutex::new(cwd),
        }
    }
}

impl Tool for BashTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".into(),
            description: "Execute a bash command. Working directory persists between calls. \
                          Captures stdout and stderr. Default timeout: 120s, max: 600s."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in milliseconds (default 120000, max 600000)"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let command = input
                .get("command")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("command is required".into()))?;

            let timeout_ms = input
                .get("timeout")
                .and_then(|v| v.as_u64())
                .unwrap_or(DEFAULT_TIMEOUT_MS)
                .min(MAX_TIMEOUT_MS);

            let cwd = {
                let guard = self.cwd.lock().expect("bash cwd lock poisoned");
                guard.clone()
            };

            // Build the command: cd to tracked cwd, run user command, then print pwd.
            // Detect trailing `&` to avoid `{ cmd &; }` syntax error
            // (`&` is a command terminator — `&;` is invalid bash).
            let wrapped = if command.trim_end().ends_with('&') {
                format!("{{ {} }}", command)
            } else {
                format!("{{ {}; }}", command)
            };
            let full_command = format!(
                "cd {} && {}; __exit_code=$?; echo; echo \"__HEARTBIT_CWD__=$(pwd)\"; exit $__exit_code",
                shell_escape(&cwd.display().to_string()),
                wrapped
            );

            let child = tokio::process::Command::new("bash")
                .arg("-c")
                .arg(&full_command)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .map_err(|e| Error::Agent(format!("Failed to spawn bash: {e}")))?;

            let timeout_duration = std::time::Duration::from_millis(timeout_ms);

            let output =
                match tokio::time::timeout(timeout_duration, child.wait_with_output()).await {
                    Ok(Ok(output)) => output,
                    Ok(Err(e)) => return Ok(ToolOutput::error(format!("Command failed: {e}"))),
                    Err(_) => {
                        // kill_on_drop ensures cleanup when `child` is dropped here
                        return Ok(ToolOutput::error(format!(
                            "Command timed out after {timeout_ms}ms"
                        )));
                    }
                };

            let exit_code = output.status.code().unwrap_or(-1);
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            // Extract new cwd from the output
            let (user_stdout, new_cwd) = extract_cwd(&stdout);
            if let Some(new_dir) = new_cwd {
                let mut guard = self.cwd.lock().expect("bash cwd lock poisoned");
                *guard = PathBuf::from(new_dir);
            }

            // Combine output
            let mut combined = String::new();
            if !user_stdout.is_empty() {
                combined.push_str(&user_stdout);
            }
            if !stderr.is_empty() {
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(&stderr);
            }

            // Truncate if needed
            let combined = truncate_middle(&combined, MAX_OUTPUT_CHARS);

            let exit_info = format!("\n\n(exit code: {exit_code})");
            let output_text = format!("{combined}{exit_info}");

            if exit_code == 0 {
                Ok(ToolOutput::success(output_text))
            } else {
                Ok(ToolOutput::error(output_text))
            }
        })
    }
}

/// Extract the cwd marker from stdout, returning (user output, optional new cwd).
fn extract_cwd(stdout: &str) -> (String, Option<String>) {
    if let Some(marker_pos) = stdout.rfind("__HEARTBIT_CWD__=") {
        let user_output = stdout[..marker_pos].trim_end().to_string();
        let cwd_line = &stdout[marker_pos + "__HEARTBIT_CWD__=".len()..];
        let cwd = cwd_line.trim().to_string();
        if cwd.is_empty() {
            (user_output, None)
        } else {
            (user_output, Some(cwd))
        }
    } else {
        (stdout.to_string(), None)
    }
}

/// Truncate output with middle omission if it exceeds max bytes.
fn truncate_middle(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }

    let total_lines = text.lines().count();

    // Find char boundaries for head/tail slices
    let head_end = super::floor_char_boundary(text, HEAD_TAIL_SIZE.min(text.len()));
    let tail_start = ceil_char_boundary(text, text.len().saturating_sub(HEAD_TAIL_SIZE));
    let head = &text[..head_end];
    let tail = &text[tail_start..];

    // Count omitted lines (approximate)
    let head_lines = head.lines().count();
    let tail_lines = tail.lines().count();
    let omitted = total_lines.saturating_sub(head_lines + tail_lines);

    format!("{head}\n\n... [{omitted} lines truncated] ...\n\n{tail}")
}

/// Find the smallest char boundary >= target.
fn ceil_char_boundary(text: &str, target: usize) -> usize {
    let mut pos = target.min(text.len());
    while pos < text.len() && !text.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = BashTool::new();
        assert_eq!(tool.definition().name, "bash");
    }

    #[tokio::test]
    async fn bash_echo() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("hello"));
        assert!(result.content.contains("exit code: 0"));
    }

    #[tokio::test]
    async fn bash_failing_command() {
        let tool = BashTool::new();
        let result = tool.execute(json!({"command": "exit 42"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("exit code: 42"));
    }

    #[tokio::test]
    async fn bash_preserves_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let tool = BashTool::new();

        // Change directory
        tool.execute(json!({"command": format!("cd {}", dir.path().display())}))
            .await
            .unwrap();

        // Verify pwd shows the new directory
        let result = tool.execute(json!({"command": "pwd"})).await.unwrap();
        assert!(
            result.content.contains(&dir.path().display().to_string()),
            "expected cwd to be {}, got: {}",
            dir.path().display(),
            result.content
        );
    }

    #[tokio::test]
    async fn bash_timeout() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({"command": "sleep 10", "timeout": 500}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({"command": "echo err >&2"}))
            .await
            .unwrap();
        assert!(result.content.contains("err"));
    }

    #[test]
    fn extract_cwd_parses_marker() {
        let stdout = "some output\n__HEARTBIT_CWD__=/home/user\n";
        let (user, cwd) = extract_cwd(stdout);
        assert_eq!(user, "some output");
        assert_eq!(cwd, Some("/home/user".into()));
    }

    #[test]
    fn extract_cwd_no_marker() {
        let stdout = "just output\n";
        let (user, cwd) = extract_cwd(stdout);
        assert_eq!(user, "just output\n");
        assert!(cwd.is_none());
    }

    #[test]
    fn truncate_middle_short_text() {
        let text = "hello world";
        assert_eq!(truncate_middle(text, 100), text);
    }

    #[test]
    fn truncate_middle_long_text() {
        let text = "a\n".repeat(20_000);
        let result = truncate_middle(&text, MAX_OUTPUT_CHARS);
        assert!(result.contains("truncated"));
        assert!(result.len() < text.len());
    }

    #[test]
    fn truncate_middle_multibyte_utf8() {
        // Each '🦀' is 4 bytes. Build a string that must be truncated.
        let text = "🦀".repeat(10_000); // 40,000 bytes
        let result = truncate_middle(&text, MAX_OUTPUT_CHARS);
        // Should not panic and should be valid UTF-8
        assert!(result.contains("truncated"));
        // Verify it's valid UTF-8 (String is UTF-8 by construction, but let's check boundaries)
        assert!(result.len() < text.len());
    }

    #[test]
    fn shell_escape_simple_path() {
        assert_eq!(shell_escape("/tmp/foo"), "'/tmp/foo'");
    }

    #[test]
    fn shell_escape_path_with_single_quote() {
        // dir's should become 'dir'\''s'
        assert_eq!(shell_escape("dir's"), "'dir'\\''s'");
    }

    #[test]
    fn shell_escape_path_with_spaces() {
        assert_eq!(shell_escape("/tmp/my dir"), "'/tmp/my dir'");
    }

    #[test]
    fn shell_escape_empty_string() {
        assert_eq!(shell_escape(""), "''");
    }

    #[tokio::test]
    async fn bash_trailing_ampersand_no_syntax_error() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({"command": "sleep 0.01 &"}))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "trailing & should not cause syntax error: {}",
            result.content
        );
        assert!(result.content.contains("exit code: 0"));
    }

    #[tokio::test]
    async fn bash_background_with_foreground() {
        let tool = BashTool::new();
        let result = tool
            .execute(json!({"command": "echo before & echo after"}))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "background & foreground should work: {}",
            result.content
        );
        assert!(result.content.contains("after"));
    }
}
