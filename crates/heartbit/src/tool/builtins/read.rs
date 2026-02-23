use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

use super::file_tracker::FileTracker;

const MAX_FILE_SIZE: u64 = 256 * 1024; // 256 KB
const MAX_LINE_LENGTH: usize = 2000;
const DEFAULT_LIMIT: usize = 2000;

pub struct ReadTool {
    file_tracker: Arc<FileTracker>,
}

impl ReadTool {
    pub fn new(file_tracker: Arc<FileTracker>) -> Self {
        Self { file_tracker }
    }
}

impl Tool for ReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read".into(),
            description: "Read a file from the filesystem. Returns content with line numbers. \
                          Detects binary files and rejects them. Max file size: 256 KB."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start reading from"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of lines to read (default 2000)"
                    }
                },
                "required": ["file_path"]
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

            let offset = input
                .get("offset")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(1);

            let limit = input
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_LIMIT);

            let path = PathBuf::from(file_path);

            // Check if it's a directory
            if path.is_dir() {
                return Ok(ToolOutput::error(format!(
                    "{file_path} is a directory. Use the 'list' tool to list directory contents."
                )));
            }

            // Check existence
            if !path.exists() {
                let suggestion = suggest_similar_file(&path);
                let msg = match suggestion {
                    Some(s) => format!("File not found: {file_path}. Did you mean: {s}?"),
                    None => format!("File not found: {file_path}"),
                };
                return Ok(ToolOutput::error(msg));
            }

            // Check file size
            let metadata = std::fs::metadata(&path)
                .map_err(|e| Error::Agent(format!("Cannot read metadata: {e}")))?;
            if metadata.len() > MAX_FILE_SIZE {
                return Ok(ToolOutput::error(format!(
                    "File too large ({} bytes). Maximum supported size is {} bytes.",
                    metadata.len(),
                    MAX_FILE_SIZE
                )));
            }

            // Read file
            let content = tokio::fs::read(&path)
                .await
                .map_err(|e| Error::Agent(format!("Cannot read file: {e}")))?;

            // Binary detection: check first 4096 bytes for non-printable chars
            let sample_size = content.len().min(4096);
            let non_printable = content[..sample_size]
                .iter()
                .filter(|&&b| b != b'\n' && b != b'\r' && b != b'\t' && (b < 0x20 || b == 0x7f))
                .count();
            if non_printable > sample_size * 30 / 100 {
                return Ok(ToolOutput::error(format!(
                    "File appears to be binary ({non_printable} non-printable bytes in first {sample_size} bytes). Cannot display."
                )));
            }

            let text = String::from_utf8_lossy(&content);
            let lines: Vec<&str> = text.lines().collect();
            let total_lines = lines.len();

            // Apply offset (1-based) and limit
            let start = if offset > 0 { offset - 1 } else { 0 };
            let end = (start + limit).min(total_lines);

            if start >= total_lines {
                return Ok(ToolOutput::error(format!(
                    "Offset {offset} is beyond the end of the file ({total_lines} lines)."
                )));
            }

            let mut output = String::new();
            for (idx, line) in lines[start..end].iter().enumerate() {
                let line_num = start + idx + 1;
                let truncated_line = if line.len() > MAX_LINE_LENGTH {
                    let end = super::floor_char_boundary(line, MAX_LINE_LENGTH);
                    format!("{}...", &line[..end])
                } else {
                    line.to_string()
                };
                output.push_str(&format!("{line_num:>6}\t{truncated_line}\n"));
            }

            if end < total_lines {
                output.push_str(&format!(
                    "\n({} more lines not shown. Use offset/limit to read more.)",
                    total_lines - end
                ));
            }

            // Record the read
            let _ = self.file_tracker.record_read(&path);

            Ok(ToolOutput::success(output))
        })
    }
}

/// Try to find a similarly-named file in the same directory.
fn suggest_similar_file(path: &Path) -> Option<String> {
    let parent = path.parent()?;
    let target_name = path.file_name()?.to_str()?;

    let entries = std::fs::read_dir(parent).ok()?;
    let mut best: Option<(usize, String)> = None;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = match name.to_str() {
            Some(s) => s,
            None => continue, // Skip non-UTF-8 filenames
        };
        let distance = levenshtein(target_name, name_str);
        if distance <= 3 {
            match &best {
                None => best = Some((distance, entry.path().display().to_string())),
                Some((best_dist, _)) if distance < *best_dist => {
                    best = Some((distance, entry.path().display().to_string()));
                }
                _ => {}
            }
        }
    }

    best.map(|(_, name)| name)
}

use crate::util::levenshtein;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tracker = Arc::new(FileTracker::new());
        let tool = ReadTool::new(tracker);
        assert_eq!(tool.definition().name, "read");
    }

    #[tokio::test]
    async fn read_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line one\nline two\nline three\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        let tool = ReadTool::new(tracker.clone());

        let result = tool
            .execute(json!({"file_path": path.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("line one"));
        assert!(result.content.contains("line two"));
        assert!(result.content.contains("line three"));
        // Line numbers should be present
        assert!(result.content.contains("     1\t"));
        assert!(result.content.contains("     2\t"));

        // File should be tracked
        assert!(tracker.was_read(&path));
    }

    #[tokio::test]
    async fn read_with_offset_and_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let content = (1..=10)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, &content).unwrap();

        let tracker = Arc::new(FileTracker::new());
        let tool = ReadTool::new(tracker);

        let result = tool
            .execute(json!({"file_path": path.to_str().unwrap(), "offset": 3, "limit": 2}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("line 3"));
        assert!(result.content.contains("line 4"));
        assert!(!result.content.contains("line 2"));
        assert!(!result.content.contains("line 5"));
    }

    #[tokio::test]
    async fn read_nonexistent_file() {
        let tracker = Arc::new(FileTracker::new());
        let tool = ReadTool::new(tracker);

        let result = tool
            .execute(json!({"file_path": "/tmp/nonexistent_heartbit_test_12345.txt"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("File not found"));
    }

    #[tokio::test]
    async fn read_directory_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let tracker = Arc::new(FileTracker::new());
        let tool = ReadTool::new(tracker);

        let result = tool
            .execute(json!({"file_path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("directory"));
    }

    #[tokio::test]
    async fn read_binary_file_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("binary.bin");
        // Write mostly non-printable bytes (null bytes exceed 30% threshold)
        let data: Vec<u8> = vec![0u8; 1000];
        std::fs::write(&path, &data).unwrap();

        let tracker = Arc::new(FileTracker::new());
        let tool = ReadTool::new(tracker);

        let result = tool
            .execute(json!({"file_path": path.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("binary"));
    }

    #[test]
    fn levenshtein_distance() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn levenshtein_unicode() {
        // Multi-byte chars must not panic (was using byte-length matrix with char iteration)
        assert_eq!(levenshtein("café", "cafe"), 1);
        assert_eq!(levenshtein("日本語", "日本語"), 0);
        assert_eq!(levenshtein("日本語", "日本人"), 1);
    }
}
