use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

use super::file_tracker::FileTracker;

pub struct EditTool {
    file_tracker: Arc<FileTracker>,
    workspace: Option<PathBuf>,
}

impl EditTool {
    pub fn new(file_tracker: Arc<FileTracker>, workspace: Option<PathBuf>) -> Self {
        Self {
            file_tracker,
            workspace,
        }
    }
}

impl Tool for EditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit".into(),
            description: "Edit a file by replacing an exact string. The old_string must appear \
                          exactly once in the file. The file must have been read first."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path, or relative to workspace"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace (must appear exactly once)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let file_path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("file_path is required".into()))?;

            let old_string = input
                .get("old_string")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("old_string is required".into()))?;

            let new_string = input
                .get("new_string")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("new_string is required".into()))?;

            let path = super::resolve_path(file_path, self.workspace.as_deref());

            if !path.exists() {
                return Ok(ToolOutput::error(format!("File not found: {file_path}")));
            }

            // No-op guard
            if old_string == new_string {
                return Ok(ToolOutput::error(
                    "old_string and new_string are identical. No change needed.",
                ));
            }

            // Read-before-write guard
            if let Err(msg) = self.file_tracker.check_unmodified(&path) {
                return Ok(ToolOutput::error(msg));
            }

            // Read current content
            let content = tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| Error::Agent(format!("Cannot read file: {e}")))?;

            // Check occurrence count
            let count = content.matches(old_string).count();
            if count == 0 {
                return Ok(ToolOutput::error(
                    "String not found in file. Make sure the old_string matches exactly, \
                     including whitespace and indentation.",
                ));
            }
            if count > 1 {
                return Ok(ToolOutput::error(format!(
                    "String appears {count} times, must be unique. Add more surrounding context \
                     to make the match unique."
                )));
            }

            // Splice — count == 1 was verified above, so this cannot fail
            let Some(idx) = content.find(old_string) else {
                return Ok(ToolOutput::error(
                    "Internal error: string vanished after count check",
                ));
            };
            let new_content =
                String::from(&content[..idx]) + new_string + &content[idx + old_string.len()..];

            // Write
            tokio::fs::write(&path, &new_content)
                .await
                .map_err(|e| Error::Agent(format!("Cannot write file: {e}")))?;

            // Update tracker
            let _ = self.file_tracker.record_read(&path);

            // Build output: show changed lines with context
            let output = format_edit_snippet(&new_content, idx, new_string.len());

            Ok(ToolOutput::success(output))
        })
    }
}

/// Format a snippet of the edited file showing lines around the change.
fn format_edit_snippet(content: &str, change_offset: usize, change_len: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();

    // Find which lines the change spans
    let mut offset = 0;
    let mut start_line = 0;
    let mut end_line = lines.len().saturating_sub(1);
    for (i, line) in lines.iter().enumerate() {
        let line_end = offset + line.len() + 1; // +1 for newline
        if offset <= change_offset && change_offset < line_end {
            start_line = i;
        }
        if offset <= change_offset + change_len && change_offset + change_len <= line_end {
            end_line = i;
            break;
        }
        offset = line_end;
    }

    // Show 2 lines of context before/after
    let ctx_start = start_line.saturating_sub(2);
    let ctx_end = (end_line + 3).min(lines.len());

    let mut output = String::new();
    for (i, line) in lines.iter().enumerate().take(ctx_end).skip(ctx_start) {
        output.push_str(&format!("{:>6}\t{}\n", i + 1, line));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tracker = Arc::new(FileTracker::new());
        let tool = EditTool::new(tracker, None);
        assert_eq!(tool.definition().name, "edit");
    }

    #[tokio::test]
    async fn edit_replaces_exact_match() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world\ngoodbye world\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let tool = EditTool::new(tracker, None);
        let result = tool
            .execute(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "hello world",
                "new_string": "hi universe"
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hi universe\ngoodbye world\n");
    }

    #[tokio::test]
    async fn edit_fails_when_not_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tracker = Arc::new(FileTracker::new());
        let tool = EditTool::new(tracker, None);

        let result = tool
            .execute(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "bye"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("has not been read yet"));
    }

    #[tokio::test]
    async fn edit_fails_when_string_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let tool = EditTool::new(tracker, None);
        let result = tool
            .execute(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "xyz",
                "new_string": "abc"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn edit_fails_when_string_appears_multiple_times() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello hello hello").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let tool = EditTool::new(tracker, None);
        let result = tool
            .execute(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "bye"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("3 times"));
    }

    #[tokio::test]
    async fn edit_fails_on_noop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let tool = EditTool::new(tracker, None);
        let result = tool
            .execute(json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "hello"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("identical"));
    }

    #[tokio::test]
    async fn edit_nonexistent_file() {
        let tracker = Arc::new(FileTracker::new());
        let tool = EditTool::new(tracker, None);
        let result = tool
            .execute(json!({
                "file_path": "/tmp/nonexistent_heartbit_test_12345.txt",
                "old_string": "a",
                "new_string": "b"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("File not found"));
    }

    #[test]
    fn format_edit_snippet_change_at_eof() {
        // When the change is at the very end, the snippet should show the last lines
        let content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nchanged\n";
        let change_offset = content.rfind("changed").unwrap();
        let snippet = format_edit_snippet(content, change_offset, "changed".len());
        // The snippet should show the last lines including "changed", NOT the top of the file
        assert!(
            snippet.contains("changed"),
            "snippet should contain the changed text: {snippet}"
        );
        assert!(
            snippet.contains("line 5") || snippet.contains("line 6"),
            "snippet should show context near EOF: {snippet}"
        );
    }
}
