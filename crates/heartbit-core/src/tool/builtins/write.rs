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

/// Builtin tool that writes full file contents (create or overwrite).
///
/// Requires the target file to have been read first via `ReadTool` in the same
/// session (mtime-based guard in `FileTracker`), preventing the agent from
/// overwriting files it has never seen. Protected paths and an optional
/// `CorePathPolicy` act as additional guardrails. Prefer `EditTool` for
/// targeted in-place modifications and `PatchTool` for unified-diff patches;
/// use `WriteTool` only when creating a new file or rewriting the entire content.
pub struct WriteTool {
    file_tracker: Arc<FileTracker>,
    workspace: Option<PathBuf>,
    protected_paths: Arc<Vec<PathBuf>>,
    path_policy: Option<Arc<CorePathPolicy>>,
}

impl WriteTool {
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
        }
    }

    /// Set a `CorePathPolicy` that restricts file paths beyond what the
    /// workspace + protected_paths combination already enforces. The policy's
    /// `check_path` is called before any I/O.
    pub fn with_path_policy(mut self, policy: Arc<CorePathPolicy>) -> Self {
        self.path_policy = Some(policy);
        self
    }
}

impl Tool for WriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write".into(),
            description: "Write content to a file. Creates parent directories if needed. \
                          If the file already exists, it must have been read first (read-before-write guard)."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path, or relative to workspace"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
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

            let content = input
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("content is required".into()))?;

            let path = match super::resolve_path(
                file_path,
                self.workspace.as_deref(),
                &self.protected_paths,
            ) {
                Ok(p) => p,
                Err(msg) => return Ok(ToolOutput::error(msg)),
            };

            if let Some(policy) = &self.path_policy {
                // SECURITY (F-FS-1): canonicalize the parent and recompose
                // `parent.canonical + filename`. The previous "walk up to first
                // existing ancestor, check, then write the original path"
                // pattern left a TOCTOU window: a parallel tool call (bash
                // dispatched alongside write via tokio::JoinSet) could replace
                // an intermediate component with a symlink pointing outside the
                // workspace between the check and the open. The new method
                // binds the path to the *real* parent and `O_NOFOLLOW` on the
                // open closes the race entirely.
                if let Err(e) = policy.check_path_for_create(&path) {
                    return Ok(ToolOutput::error(format!("path policy: {e}")));
                }
            }

            // If file exists, enforce read-before-write guard
            if path.exists() {
                if let Err(msg) = self.file_tracker.check_unmodified(&path) {
                    return Ok(ToolOutput::error(msg));
                }

                // Skip write if content identical
                if let Ok(existing) = tokio::fs::read_to_string(&path).await
                    && existing == content
                {
                    return Ok(ToolOutput::success(format!(
                        "File unchanged: {file_path} (content identical)"
                    )));
                }
            }

            // Create parent directories
            if let Some(parent) = path.parent()
                && !parent.exists()
            {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| Error::Agent(format!("Cannot create directories: {e}")))?;
            }

            // Write the file with O_NOFOLLOW (Unix) so the open syscall fails
            // if any component of `path` is a symlink. This neutralises the
            // residual TOCTOU window where a parallel tool call could have
            // replaced the path between policy check and open.
            let bytes = content.len();
            super::write_no_follow(&path, content.as_bytes())
                .await
                .map_err(|e| Error::Agent(format!("Cannot write file: {e}")))?;

            // Update tracker (so subsequent edits pass the guard)
            let _ = self.file_tracker.record_read(&path);

            Ok(ToolOutput::success(format!(
                "File written: {file_path} ({bytes} bytes)"
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tracker = Arc::new(FileTracker::new());
        let tool = WriteTool::new(tracker, None, Arc::new(Vec::new()));
        assert_eq!(tool.definition().name, "write");
    }

    #[tokio::test]
    async fn write_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.txt");

        let tracker = Arc::new(FileTracker::new());
        let tool = WriteTool::new(tracker.clone(), None, Arc::new(Vec::new()));

        let result = tool
            .execute(json!({"file_path": path.to_str().unwrap(), "content": "hello world"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("File written"));
        assert!(result.content.contains("11 bytes"));

        // Verify content
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "hello world");

        // File should be tracked
        assert!(tracker.was_read(&path));
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("deep").join("file.txt");

        let tracker = Arc::new(FileTracker::new());
        let tool = WriteTool::new(tracker, None, Arc::new(Vec::new()));

        let result = tool
            .execute(json!({"file_path": path.to_str().unwrap(), "content": "nested"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "nested");
    }

    #[tokio::test]
    async fn write_existing_file_requires_read_first() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("existing.txt");
        std::fs::write(&path, "original").unwrap();

        let tracker = Arc::new(FileTracker::new());
        let tool = WriteTool::new(tracker, None, Arc::new(Vec::new()));

        // Try to write without reading first
        let result = tool
            .execute(json!({"file_path": path.to_str().unwrap(), "content": "new content"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("has not been read yet"));
    }

    #[tokio::test]
    async fn write_skips_when_content_identical() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("same.txt");
        std::fs::write(&path, "same content").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let tool = WriteTool::new(tracker, None, Arc::new(Vec::new()));
        let result = tool
            .execute(json!({"file_path": path.to_str().unwrap(), "content": "same content"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("unchanged"));
    }

    #[tokio::test]
    async fn write_existing_file_after_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("existing.txt");
        std::fs::write(&path, "original").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let tool = WriteTool::new(tracker, None, Arc::new(Vec::new()));
        let result = tool
            .execute(json!({"file_path": path.to_str().unwrap(), "content": "updated"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "updated");
    }

    #[tokio::test]
    async fn write_tool_rejects_path_outside_policy() {
        use crate::sandbox::CorePathPolicy;

        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let policy = Arc::new(
            CorePathPolicy::builder()
                .allow_dir(allowed.path())
                .build()
                .unwrap(),
        );

        // No workspace — absolute paths are accepted by resolve_path
        let tool = WriteTool::new(Arc::new(FileTracker::new()), None, Arc::new(Vec::new()))
            .with_path_policy(policy);

        let target = outside.path().join("evil.txt");
        let result = tool
            .execute(serde_json::json!({
                "file_path": target.to_string_lossy(),
                "content": "x"
            }))
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
    async fn write_tool_allows_path_inside_policy() {
        use crate::sandbox::CorePathPolicy;

        let allowed = tempfile::tempdir().unwrap();
        let policy = Arc::new(
            CorePathPolicy::builder()
                .allow_dir(allowed.path())
                .build()
                .unwrap(),
        );

        // No workspace — absolute paths are accepted by resolve_path
        let tool = WriteTool::new(Arc::new(FileTracker::new()), None, Arc::new(Vec::new()))
            .with_path_policy(policy);

        let target = allowed.path().join("ok.txt");
        let result = tool
            .execute(serde_json::json!({
                "file_path": target.to_string_lossy(),
                "content": "x"
            }))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "expected success, got: {:?}",
            result.content
        );
    }

    /// SECURITY (F-FS-1): if the target path is a symlink pointing outside the
    /// allowed directory, the open syscall must fail with `O_NOFOLLOW` rather
    /// than follow the link and write through it. This protects against TOCTOU
    /// where a parallel tool call swaps in a symlink between policy check and
    /// open.
    #[cfg(unix)]
    #[tokio::test]
    async fn write_refuses_to_follow_symlink_pointing_outside_workspace() {
        use crate::sandbox::CorePathPolicy;
        use std::os::unix::fs::symlink;

        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();

        // Create a "victim" file that the LLM should NOT be able to overwrite.
        let victim = outside.path().join("victim.txt");
        std::fs::write(&victim, "ORIGINAL CONTENT").unwrap();

        // Inside the allowed dir, create a symlink pointing to the victim file.
        let link = allowed.path().join("link.txt");
        symlink(&victim, &link).unwrap();

        let policy = Arc::new(
            CorePathPolicy::builder()
                .allow_dir(allowed.path())
                .build()
                .unwrap(),
        );
        let tracker = Arc::new(FileTracker::new());
        // Must record the symlink target as read so the read-before-write
        // guard doesn't preempt the security check we are testing.
        let _ = tracker.record_read(&link);
        let tool = WriteTool::new(tracker, None, Arc::new(Vec::new())).with_path_policy(policy);

        // Attempt to overwrite via the symlink path — must fail. The error
        // can surface as either `Err(Error::Agent("Cannot write file: ..."))`
        // (from the failed open syscall propagating up through `?`) or as
        // `Ok(ToolOutput::error)` from the policy reject. Both prove the
        // security invariant; the victim file must remain untouched.
        let outcome = tool
            .execute(serde_json::json!({
                "file_path": link.to_string_lossy(),
                "content": "PWNED"
            }))
            .await;
        match outcome {
            Ok(r) => assert!(
                r.is_error,
                "expected error tool output; got success: {:?}",
                r.content
            ),
            Err(e) => {
                let s = e.to_string().to_lowercase();
                assert!(
                    s.contains("symbolic")
                        || s.contains("symlink")
                        || s.contains("nofollow")
                        || s.contains("loop"),
                    "expected symlink-related error; got: {e}"
                );
            }
        }

        // Victim must remain untouched — that's the security invariant.
        let after = std::fs::read_to_string(&victim).unwrap();
        assert_eq!(
            after, "ORIGINAL CONTENT",
            "victim file was modified despite symlink rejection"
        );
    }
}
