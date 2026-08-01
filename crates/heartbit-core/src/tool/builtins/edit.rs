use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::sandbox::CorePathPolicy;
use crate::tool::{Tool, ToolOutput};

use super::file_tracker::FileTracker;

/// Builtin tool that performs exact-string in-place replacements within a file.
///
/// Locates a unique `old_string` in the file and replaces it with `new_string`,
/// writing the result back atomically. Because the match must be unique, `EditTool`
/// is safer than line-number-based edits for files that may shift between turns.
/// Like `WriteTool`, it requires a prior `ReadTool` call to guard against editing
/// files the agent has not seen. For patch-format multi-hunk edits use `PatchTool`.
pub struct EditTool {
    file_tracker: Arc<FileTracker>,
    workspace: Option<PathBuf>,
    protected_paths: Arc<Vec<PathBuf>>,
    path_policy: Option<Arc<CorePathPolicy>>,
    formatters: Option<Arc<crate::tool::builtins::format::FormatterConfig>>,
}

impl EditTool {
    pub fn new(
        file_tracker: Arc<FileTracker>,
        workspace: Option<PathBuf>,
        protected_paths: Arc<Vec<PathBuf>>,
    ) -> Self {
        Self {
            file_tracker,
            workspace,
            protected_paths,
            path_policy: None,
            formatters: None,
        }
    }

    /// Set a `CorePathPolicy` that restricts file paths beyond what the
    /// workspace + protected_paths combination already enforces. The policy's
    /// `check_path` is called before any I/O.
    pub fn with_path_policy(mut self, policy: Arc<CorePathPolicy>) -> Self {
        self.path_policy = Some(policy);
        self
    }

    /// Format the content with these formatters before writing.
    #[must_use]
    pub fn with_formatters(
        mut self,
        formatters: Arc<crate::tool::builtins::format::FormatterConfig>,
    ) -> Self {
        self.formatters = Some(formatters);
        self
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
                        "description": super::path_param_description(self.workspace.as_deref())
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
        _ctx: &crate::ExecutionContext,
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

            let path = match super::resolve_path(
                file_path,
                self.workspace.as_deref(),
                &self.protected_paths,
            ) {
                Ok(p) => p,
                Err(msg) => return Ok(ToolOutput::error(msg)),
            };

            if !path.exists() {
                return Ok(ToolOutput::error(format!("File not found: {file_path}")));
            }

            // SECURITY (F-FS-1): mirror write.rs/patch.rs. With a policy active,
            // recompose the canonical target and write it via the symlink-safe
            // component walk (`write_beneath_root`) so an intermediate- or
            // final-component symlink swap (parallel tool-call race) cannot
            // redirect the write outside the policy root. EditTool was previously
            // left out of this hardening and wrote via plain `tokio::fs::write`,
            // which follows symlinks. Without a policy, fall back to the
            // O_NOFOLLOW writer like the sibling tools.
            let (target, write_root) = match &self.path_policy {
                Some(policy) => match policy.check_path_for_create(&path) {
                    Ok(canonical) => {
                        let root = policy
                            .allowed_root_for(&canonical)
                            .map(std::path::Path::to_path_buf);
                        (canonical, root)
                    }
                    Err(e) => return Ok(ToolOutput::error(format!("path policy: {e}"))),
                },
                None => (path.clone(), None),
            };

            // No-op guard
            if old_string == new_string {
                return Ok(ToolOutput::error(
                    "old_string and new_string are identical. No change needed.",
                ));
            }

            // Read-before-write guard
            if let Err(msg) = self.file_tracker.check_unmodified(&target) {
                return Ok(ToolOutput::error(msg));
            }

            // Read current content
            let content = tokio::fs::read_to_string(&target)
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

            // Format in memory BEFORE the single write: keeps the post-write
            // record_read mtime matching the final bytes, keeps the returned
            // snippet consistent with disk, and never hands the subprocess a
            // path (F-FS-1 symlink hardening stays in force). Runs before
            // `format_edit_snippet` so the snippet the model sees matches what
            // lands on disk.
            let new_content = match &self.formatters {
                Some(fc) => match super::format::format_content(fc, &target, &new_content).await {
                    Some(formatted) => formatted,
                    None => new_content,
                },
                None => new_content,
            };

            // Write (symlink-safe; see F-FS-1 note above)
            match write_root {
                Some(root) => {
                    super::write_beneath_root(&root, &target, new_content.as_bytes())
                        .await
                        .map_err(|e| Error::Agent(format!("Cannot write file: {e}")))?;
                }
                None => {
                    super::write_no_follow(&target, new_content.as_bytes())
                        .await
                        .map_err(|e| Error::Agent(format!("Cannot write file: {e}")))?;
                }
            }

            // Update tracker
            let _ = self.file_tracker.record_read(&target);

            // Build output: show changed lines with context. `idx`/
            // `new_string.len()` were computed against the PRE-format buffer;
            // if formatting shrank the content past that point, an unclamped
            // offset makes `format_edit_snippet` fall through to dumping the
            // WHOLE (already-formatted) buffer instead of a bounded window.
            // Clamping degrades that case to "tail of file" instead.
            let snippet_offset = idx.min(new_content.len());
            let snippet_len = new_string
                .len()
                .min(new_content.len().saturating_sub(snippet_offset));
            let output = format_edit_snippet(&new_content, snippet_offset, snippet_len);

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
        let tool = EditTool::new(tracker, None, Arc::new(Vec::new()));
        assert_eq!(tool.definition().name, "edit");
    }

    // Description must match the policy (see write.rs counterpart): with a
    // workspace set, absolute paths are rejected — say so.
    #[test]
    fn path_description_matches_workspace_policy() {
        let tracker = Arc::new(FileTracker::new());
        let sandboxed = EditTool::new(
            tracker.clone(),
            Some(std::path::PathBuf::from("/ws")),
            Arc::new(Vec::new()),
        );
        let desc = sandboxed.definition().input_schema["properties"]["file_path"]["description"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            desc.contains("relative to the workspace") && desc.contains("rejected"),
            "workspace-set description must warn absolute is rejected: {desc}"
        );
        let open = EditTool::new(tracker, None, Arc::new(Vec::new()));
        let desc = open.definition().input_schema["properties"]["file_path"]["description"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(desc.contains("Absolute"), "{desc}");
    }

    #[tokio::test]
    async fn edit_replaces_exact_match() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world\ngoodbye world\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let tool = EditTool::new(tracker, None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "hello world",
                    "new_string": "hi universe"
                }),
            )
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
        let tool = EditTool::new(tracker, None, Arc::new(Vec::new()));

        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "bye"
                }),
            )
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

        let tool = EditTool::new(tracker, None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "xyz",
                    "new_string": "abc"
                }),
            )
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

        let tool = EditTool::new(tracker, None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "bye"
                }),
            )
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

        let tool = EditTool::new(tracker, None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "hello"
                }),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("identical"));
    }

    #[tokio::test]
    async fn edit_nonexistent_file() {
        let tracker = Arc::new(FileTracker::new());
        let tool = EditTool::new(tracker, None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": "/tmp/nonexistent_heartbit_test_12345.txt",
                    "old_string": "a",
                    "new_string": "b"
                }),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("File not found"));
    }

    #[tokio::test]
    async fn edit_tool_rejects_path_outside_policy() {
        use crate::sandbox::CorePathPolicy;

        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let policy = Arc::new(
            CorePathPolicy::builder()
                .allow_dir(allowed.path())
                .build()
                .unwrap(),
        );

        // Create a file outside the allowed dir
        let target = outside.path().join("evil.txt");
        std::fs::write(&target, "hello").unwrap();

        // No workspace — absolute paths are accepted by resolve_path
        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&target).unwrap();

        let tool = EditTool::new(tracker, None, Arc::new(Vec::new())).with_path_policy(policy);

        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": target.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "bye"
                }),
            )
            .await
            .unwrap();
        assert!(
            result.is_error,
            "expected sandbox violation, got: {:?}",
            result.content
        );
        assert!(
            result.content.contains("path policy"),
            "expected path policy error, got: {:?}",
            result.content
        );
    }

    #[tokio::test]
    async fn edit_tool_allows_path_inside_policy() {
        use crate::sandbox::CorePathPolicy;

        let allowed = tempfile::tempdir().unwrap();
        let policy = Arc::new(
            CorePathPolicy::builder()
                .allow_dir(allowed.path())
                .build()
                .unwrap(),
        );

        let target = allowed.path().join("ok.txt");
        std::fs::write(&target, "hello world").unwrap();

        // No workspace — absolute paths are accepted by resolve_path
        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&target).unwrap();

        let tool = EditTool::new(tracker, None, Arc::new(Vec::new())).with_path_policy(policy);

        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": target.to_str().unwrap(),
                    "old_string": "hello world",
                    "new_string": "goodbye world"
                }),
            )
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "expected success, got: {:?}",
            result.content
        );
    }

    #[tokio::test]
    async fn edit_formats_content_before_the_single_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rs");
        std::fs::write(&path, "hello world\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let mut fc = crate::tool::builtins::format::FormatterConfig::default();
        fc.set("rs", vec!["tr".into(), "a-z".into(), "A-Z".into()]);

        let tool = EditTool::new(tracker.clone(), None, Arc::new(Vec::new()))
            .with_formatters(Arc::new(fc));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "hi"
                }),
            )
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        // On disk: the post-replacement buffer went through the formatter.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "HI WORLD\n");
        // The mtime guard matches the FINAL bytes — a follow-up edit with no
        // intervening read must pass.
        assert!(tracker.check_unmodified(&path).is_ok());
    }

    #[tokio::test]
    async fn edit_survives_a_broken_formatter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rs");
        std::fs::write(&path, "hello world\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let mut fc = crate::tool::builtins::format::FormatterConfig::default();
        fc.set("rs", vec!["heartbit-no-such-formatter-binary".into()]);

        let tool = EditTool::new(tracker, None, Arc::new(Vec::new())).with_formatters(Arc::new(fc));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "hello",
                    "new_string": "hi"
                }),
            )
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hi world\n");
    }

    #[tokio::test]
    async fn edit_snippet_clamps_to_the_formatted_buffer_not_the_whole_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rs");
        let mut original = String::new();
        for n in 1..=30 {
            original.push_str(&format!("line{n:02}\n"));
        }
        std::fs::write(&path, &original).unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // `head -c 65` truncates the POST-replacement buffer down to 65 bytes
        // — the first 9 full lines plus 2 chars of the 10th — far short of
        // where `idx`/`new_string.len()` (computed against the PRE-format,
        // ~200+ byte buffer) point.
        let mut fc = crate::tool::builtins::format::FormatterConfig::default();
        fc.set("rs", vec!["head".into(), "-c".into(), "65".into()]);

        let tool = EditTool::new(tracker.clone(), None, Arc::new(Vec::new()))
            .with_formatters(Arc::new(fc));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "file_path": path.to_str().unwrap(),
                    "old_string": "line30",
                    "new_string": "line30_EXTENDED_TAIL_TEXT"
                }),
            )
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        // On disk: the truncated buffer.
        assert_eq!(std::fs::read_to_string(&path).unwrap().len(), 65);
        // The snippet must be bounded by the (short) formatted content — a
        // stale, pre-format offset must NOT fall through to dumping the
        // whole buffer from line 1.
        assert!(
            !result.content.contains("line01"),
            "snippet should not dump from the start of the file: {}",
            result.content
        );
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
