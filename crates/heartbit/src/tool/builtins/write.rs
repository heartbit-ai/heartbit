use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

use super::file_tracker::FileTracker;

pub struct WriteTool {
    file_tracker: Arc<FileTracker>,
}

impl WriteTool {
    pub fn new(file_tracker: Arc<FileTracker>) -> Self {
        Self { file_tracker }
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
                        "description": "Absolute path to the file to write"
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

            let path = PathBuf::from(file_path);

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

            // Write the file
            let bytes = content.len();
            tokio::fs::write(&path, content)
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
        let tool = WriteTool::new(tracker);
        assert_eq!(tool.definition().name, "write");
    }

    #[tokio::test]
    async fn write_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.txt");

        let tracker = Arc::new(FileTracker::new());
        let tool = WriteTool::new(tracker.clone());

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
        let tool = WriteTool::new(tracker);

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
        let tool = WriteTool::new(tracker);

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

        let tool = WriteTool::new(tracker);
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

        let tool = WriteTool::new(tracker);
        let result = tool
            .execute(json!({"file_path": path.to_str().unwrap(), "content": "updated"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "updated");
    }
}
