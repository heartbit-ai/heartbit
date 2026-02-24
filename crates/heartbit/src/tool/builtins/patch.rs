use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

use super::file_tracker::FileTracker;

pub struct PatchTool {
    file_tracker: Arc<FileTracker>,
    workspace: Option<PathBuf>,
}

impl PatchTool {
    pub fn new(file_tracker: Arc<FileTracker>, workspace: Option<PathBuf>) -> Self {
        Self {
            file_tracker,
            workspace,
        }
    }
}

impl Tool for PatchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "patch".into(),
            description: "Apply a unified diff patch to one or more files. Each modified file \
                          must have been read first (read-before-write guard). Supports standard \
                          unified diff format."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "patch_text": {
                        "type": "string",
                        "description": "The unified diff text to apply"
                    }
                },
                "required": ["patch_text"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let patch_text = input
                .get("patch_text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("patch_text is required".into()))?;

            let file_patches = parse_unified_diff(patch_text)?;

            if file_patches.is_empty() {
                return Ok(ToolOutput::error(
                    "No valid hunks found in patch text. Ensure it's in unified diff format.",
                ));
            }

            // Pre-check: all modified files must have been read
            for fp in &file_patches {
                if fp.is_new {
                    if super::resolve_path(&fp.path, self.workspace.as_deref()).exists() {
                        return Ok(ToolOutput::error(format!(
                            "File {} already exists (patch says it's new)",
                            fp.path
                        )));
                    }
                } else if let Err(msg) = self
                    .file_tracker
                    .check_unmodified(&super::resolve_path(&fp.path, self.workspace.as_deref()))
                {
                    return Ok(ToolOutput::error(msg));
                }
            }

            let mut files_changed = 0;
            let mut additions = 0;
            let mut removals = 0;

            for fp in &file_patches {
                let path = super::resolve_path(&fp.path, self.workspace.as_deref());

                if fp.is_delete {
                    if path.exists() {
                        tokio::fs::remove_file(&path)
                            .await
                            .map_err(|e| Error::Agent(format!("Cannot delete {}: {e}", fp.path)))?;
                    }
                    files_changed += 1;
                    continue;
                }

                let content = if fp.is_new {
                    String::new()
                } else {
                    tokio::fs::read_to_string(&path)
                        .await
                        .map_err(|e| Error::Agent(format!("Cannot read {}: {e}", fp.path)))?
                };

                let mut lines: Vec<String> = content.lines().map(String::from).collect();

                // Apply hunks in forward order using a single-pass approach.
                // We build a new line vector by copying unchanged regions between hunks
                // and applying each hunk's changes inline.
                let mut sorted_hunks = fp.hunks.clone();
                sorted_hunks.sort_by_key(|h| h.old_start);

                let mut new_lines: Vec<String> = Vec::with_capacity(lines.len());
                let mut cursor = 0; // current position in original lines

                for hunk in &sorted_hunks {
                    let start = if hunk.old_start == 0 {
                        0
                    } else {
                        hunk.old_start - 1
                    };

                    // Detect overlapping hunks
                    if start < cursor {
                        return Ok(ToolOutput::error(format!(
                            "Overlapping hunks in {}: hunk at line {} overlaps with previous hunk (cursor at line {})",
                            fp.path,
                            start + 1,
                            cursor + 1,
                        )));
                    }

                    // Copy unchanged lines before this hunk
                    while cursor < start && cursor < lines.len() {
                        new_lines.push(lines[cursor].clone());
                        cursor += 1;
                    }

                    // Apply changes in a single pass, verifying context/removed lines
                    for change in &hunk.changes {
                        match change {
                            Change::Context(expected) => {
                                if cursor >= lines.len() {
                                    return Ok(ToolOutput::error(format!(
                                        "Context mismatch in {} at line {}: expected {:?}, but file has only {} lines",
                                        fp.path,
                                        cursor + 1,
                                        expected,
                                        lines.len(),
                                    )));
                                }
                                if !fuzzy_lines_match(&lines[cursor], expected) {
                                    return Ok(ToolOutput::error(format!(
                                        "Context mismatch in {} at line {}: expected {:?}, got {:?}",
                                        fp.path,
                                        cursor + 1,
                                        expected,
                                        lines[cursor]
                                    )));
                                }
                                new_lines.push(lines[cursor].clone());
                                cursor += 1;
                            }
                            Change::Remove(expected) => {
                                if cursor >= lines.len() {
                                    return Ok(ToolOutput::error(format!(
                                        "Remove mismatch in {} at line {}: expected {:?}, but file has only {} lines",
                                        fp.path,
                                        cursor + 1,
                                        expected,
                                        lines.len(),
                                    )));
                                }
                                if !fuzzy_lines_match(&lines[cursor], expected) {
                                    return Ok(ToolOutput::error(format!(
                                        "Remove mismatch in {} at line {}: expected {:?}, got {:?}",
                                        fp.path,
                                        cursor + 1,
                                        expected,
                                        lines[cursor]
                                    )));
                                }
                                cursor += 1; // skip removed line
                                removals += 1;
                            }
                            Change::Add(line) => {
                                new_lines.push(line.clone());
                                additions += 1;
                            }
                        }
                    }
                }

                // Copy any remaining lines after the last hunk
                while cursor < lines.len() {
                    new_lines.push(lines[cursor].clone());
                    cursor += 1;
                }

                lines = new_lines;

                // Write the modified file
                let new_content = if lines.is_empty() {
                    String::new()
                } else {
                    let mut result = lines.join("\n");
                    if content.ends_with('\n') || fp.is_new {
                        result.push('\n');
                    }
                    result
                };

                // Create parent dirs for new files
                if fp.is_new
                    && let Some(parent) = path.parent()
                    && !parent.exists()
                {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|e| Error::Agent(format!("Cannot create directories: {e}")))?;
                }

                tokio::fs::write(&path, &new_content)
                    .await
                    .map_err(|e| Error::Agent(format!("Cannot write {}: {e}", fp.path)))?;

                let _ = self.file_tracker.record_read(&path);
                files_changed += 1;
            }

            Ok(ToolOutput::success(format!(
                "Patch applied: {files_changed} file(s) changed, {additions} addition(s), {removals} removal(s)"
            )))
        })
    }
}

// --- Multi-pass fuzzy matching ---

/// Match pass for progressive fuzzy line matching.
#[derive(Debug, Clone, Copy)]
enum MatchPass {
    /// Exact byte-for-byte match.
    Exact,
    /// Match after stripping trailing whitespace.
    TrimEnd,
    /// Match after stripping leading and trailing whitespace.
    TrimBoth,
    /// Match after normalizing unicode characters (smart quotes → ASCII, etc.).
    UnicodeNormalize,
}

const MATCH_PASSES: &[MatchPass] = &[
    MatchPass::Exact,
    MatchPass::TrimEnd,
    MatchPass::TrimBoth,
    MatchPass::UnicodeNormalize,
];

/// Check if two lines match under the given pass strategy.
fn lines_match(actual: &str, expected: &str, pass: MatchPass) -> bool {
    match pass {
        MatchPass::Exact => actual == expected,
        MatchPass::TrimEnd => actual.trim_end() == expected.trim_end(),
        MatchPass::TrimBoth => actual.trim() == expected.trim(),
        MatchPass::UnicodeNormalize => normalize_unicode(actual) == normalize_unicode(expected),
    }
}

/// Try all passes in order, return true if any pass matches.
fn fuzzy_lines_match(actual: &str, expected: &str) -> bool {
    MATCH_PASSES
        .iter()
        .any(|pass| lines_match(actual, expected, *pass))
}

/// Normalize unicode characters that LLMs commonly substitute:
/// - Smart/curly quotes → straight quotes
/// - En/em dashes → hyphens
/// - Non-breaking space → regular space
/// - Other common unicode whitespace → ASCII space
fn normalize_unicode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            '\u{2013}' | '\u{2014}' => '-',
            '\u{00A0}' | '\u{2007}' | '\u{202F}' => ' ',
            _ => c,
        })
        .collect::<String>()
        .trim()
        .to_string()
}

// --- Unified diff parser ---

#[derive(Debug, Clone)]
struct FilePatch {
    path: String,
    is_new: bool,
    is_delete: bool,
    hunks: Vec<Hunk>,
}

#[derive(Debug, Clone)]
struct Hunk {
    old_start: usize,
    changes: Vec<Change>,
}

#[derive(Debug, Clone)]
enum Change {
    Context(String),
    Add(String),
    Remove(String),
}

fn parse_unified_diff(text: &str) -> Result<Vec<FilePatch>, Error> {
    let lines: Vec<&str> = text.lines().collect();
    let mut patches = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        // Look for --- / +++ headers
        if i + 1 < lines.len() && lines[i].starts_with("--- ") && lines[i + 1].starts_with("+++ ") {
            let old_path = extract_path(lines[i]);
            let new_path = extract_path(lines[i + 1]);

            let is_new = old_path == "/dev/null";
            let is_delete = new_path == "/dev/null";

            let path = if is_new {
                new_path.clone()
            } else {
                old_path.clone()
            };

            // Security: reject path traversal (.. components)
            if path != "/dev/null"
                && std::path::Path::new(&path)
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                return Err(Error::Agent(format!("Path traversal rejected: '{path}'")));
            }

            i += 2;

            let mut hunks = Vec::new();
            while i < lines.len() && lines[i].starts_with("@@ ") {
                let (hunk, next_i) = parse_hunk(&lines, i)?;
                hunks.push(hunk);
                i = next_i;
            }

            patches.push(FilePatch {
                path,
                is_new,
                is_delete,
                hunks,
            });
        } else {
            i += 1;
        }
    }

    Ok(patches)
}

fn extract_path(line: &str) -> String {
    let path = line
        .strip_prefix("--- ")
        .or_else(|| line.strip_prefix("+++ "))
        .unwrap_or(line);

    // Remove a/ or b/ prefix
    let path = path
        .strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path);

    // Remove timestamp suffix if present (e.g., "\t2024-01-01 00:00:00")
    path.split('\t').next().unwrap_or(path).to_string()
}

fn parse_hunk(lines: &[&str], start: usize) -> Result<(Hunk, usize), Error> {
    let header = lines[start];

    // Parse @@ -old_start,old_count +new_start,new_count @@
    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(Error::Agent(format!("Invalid hunk header: {header}")));
    }

    let old_range = parts[1].strip_prefix('-').unwrap_or(parts[1]);
    let old_start: usize = old_range
        .split(',')
        .next()
        .unwrap_or("1")
        .parse()
        .map_err(|_| Error::Agent(format!("Cannot parse hunk start in: {header}")))?;

    let mut changes = Vec::new();
    let mut i = start + 1;

    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("@@ ") || line.starts_with("--- ") || line.starts_with("+++ ") {
            break;
        }

        if let Some(content) = line.strip_prefix('+') {
            changes.push(Change::Add(content.to_string()));
        } else if let Some(content) = line.strip_prefix('-') {
            changes.push(Change::Remove(content.to_string()));
        } else if let Some(content) = line.strip_prefix(' ') {
            changes.push(Change::Context(content.to_string()));
        } else if line == "\\ No newline at end of file" {
            // Skip this marker
        } else {
            // Treat as context line (the line itself is the content)
            changes.push(Change::Context(line.to_string()));
        }

        i += 1;
    }

    Ok((Hunk { old_start, changes }, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tracker = Arc::new(FileTracker::new());
        let tool = PatchTool::new(tracker, None);
        assert_eq!(tool.definition().name, "patch");
    }

    #[tokio::test]
    async fn patch_applies_simple_diff() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line 1\nline 2\nline 3\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1,3 +1,3 @@\n line 1\n-line 2\n+line TWO\n line 3\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("1 file(s) changed"));
        assert!(result.content.contains("1 addition"));
        assert!(result.content.contains("1 removal"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("line TWO"));
        assert!(!content.contains("line 2"));
    }

    #[tokio::test]
    async fn patch_rejects_unread_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "content\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1 +1 @@\n-content\n+changed\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("has not been read yet"));
    }

    #[tokio::test]
    async fn patch_empty_diff() {
        let tracker = Arc::new(FileTracker::new());
        let tool = PatchTool::new(tracker, None);
        let result = tool
            .execute(json!({"patch_text": "no diff here\n"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("No valid hunks"));
    }

    #[test]
    fn parse_unified_diff_basic() {
        let diff =
            "--- a/file.txt\n+++ b/file.txt\n@@ -1,3 +1,3 @@\n line 1\n-old\n+new\n line 3\n";
        let patches = parse_unified_diff(diff).unwrap();
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].path, "file.txt");
        assert_eq!(patches[0].hunks.len(), 1);
        assert_eq!(patches[0].hunks[0].old_start, 1);
    }

    #[test]
    fn extract_path_strips_prefix() {
        assert_eq!(extract_path("--- a/src/main.rs"), "src/main.rs");
        assert_eq!(extract_path("+++ b/src/main.rs"), "src/main.rs");
        assert_eq!(extract_path("--- /dev/null"), "/dev/null");
    }

    #[tokio::test]
    async fn patch_creates_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new_file.txt");

        let tracker = Arc::new(FileTracker::new());

        let patch = format!(
            "--- /dev/null\n+++ b/{}\n@@ -0,0 +1,2 @@\n+hello\n+world\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("hello"));
        assert!(content.contains("world"));
    }

    #[tokio::test]
    async fn patch_interleaved_add_remove() {
        // This test catches the two-pass bug: remove then add with context
        // lines in between must produce correct output.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("interleaved.txt");
        std::fs::write(&path, "line1\nline2\nline3\nline4\nline5\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Remove line2, add replacement, keep context around it
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1,5 +1,5 @@\n line1\n-line2\n+replaced2\n line3\n-line4\n+replaced4\n line5\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            content, "line1\nreplaced2\nline3\nreplaced4\nline5\n",
            "interleaved add/remove produced wrong output: {content}"
        );
    }

    #[tokio::test]
    async fn patch_rejects_context_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mismatch.txt");
        std::fs::write(&path, "line 1\nline 2\nline 3\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Patch has wrong context line (says "wrong context" but file has "line 1")
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1,3 +1,3 @@\n wrong context\n-line 2\n+replaced\n line 3\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(
            result.is_error,
            "expected error but got: {}",
            result.content
        );
        assert!(
            result.content.contains("Context mismatch"),
            "got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn patch_rejects_remove_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mismatch2.txt");
        std::fs::write(&path, "line 1\nline 2\nline 3\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Patch tries to remove "wrong line" but actual line is "line 2"
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1,3 +1,3 @@\n line 1\n-wrong line\n+replaced\n line 3\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(
            result.is_error,
            "expected error but got: {}",
            result.content
        );
        assert!(
            result.content.contains("Remove mismatch"),
            "got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn patch_deletes_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("to_delete.txt");
        std::fs::write(&path, "content\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        let patch = format!(
            "--- a/{0}\n+++ /dev/null\n@@ -1 +0,0 @@\n-content\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn rejects_path_traversal_in_new_file() {
        let patch = "\
--- /dev/null
+++ b/../../etc/evil.sh
@@ -0,0 +1 @@
+malicious content
";
        let result = parse_unified_diff(patch);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Path traversal rejected"),
            "expected path traversal error, got: {err}"
        );
    }

    #[tokio::test]
    async fn rejects_path_traversal_in_existing_file() {
        let patch = "\
--- a/../../../etc/passwd
+++ b/../../../etc/passwd
@@ -1,3 +1,3 @@
 context
-old
+new
 context
";
        let result = parse_unified_diff(patch);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Path traversal rejected"),
            "expected path traversal error, got: {err}"
        );
    }

    #[tokio::test]
    async fn patch_multi_hunk_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.txt");
        std::fs::write(
            &path,
            "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n",
        )
        .unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Two hunks: replace line2 and line8
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n\
             @@ -1,4 +1,4 @@\n line1\n-line2\n+LINE_TWO\n line3\n line4\n\
             @@ -7,4 +7,4 @@\n line7\n-line8\n+LINE_EIGHT\n line9\n line10\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("2 addition"));
        assert!(result.content.contains("2 removal"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            content,
            "line1\nLINE_TWO\nline3\nline4\nline5\nline6\nline7\nLINE_EIGHT\nline9\nline10\n"
        );
    }

    #[tokio::test]
    async fn patch_multi_hunk_out_of_order() {
        // Hunks provided in reverse order — parser should still apply correctly
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reverse.txt");
        std::fs::write(&path, "a\nb\nc\nd\ne\nf\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Hunk for line 5 before hunk for line 2
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n\
             @@ -4,3 +4,3 @@\n d\n-e\n+E\n f\n\
             @@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "a\nB\nc\nd\nE\nf\n");
    }

    #[tokio::test]
    async fn patch_multi_file() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("file1.txt");
        let p2 = dir.path().join("file2.txt");
        std::fs::write(&p1, "hello\n").unwrap();
        std::fs::write(&p2, "world\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&p1).unwrap();
        tracker.record_read(&p2).unwrap();

        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1 +1 @@\n-hello\n+HELLO\n\
             --- a/{1}\n+++ b/{1}\n@@ -1 +1 @@\n-world\n+WORLD\n",
            p1.display(),
            p2.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("2 file(s) changed"));

        assert_eq!(std::fs::read_to_string(&p1).unwrap(), "HELLO\n");
        assert_eq!(std::fs::read_to_string(&p2).unwrap(), "WORLD\n");
    }

    #[test]
    fn parse_multi_hunk_diff() {
        let diff = "--- a/f.txt\n+++ b/f.txt\n\
                    @@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n\
                    @@ -8,3 +8,3 @@\n x\n-y\n+Y\n z\n";
        let patches = parse_unified_diff(diff).unwrap();
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].hunks.len(), 2);
        assert_eq!(patches[0].hunks[0].old_start, 1);
        assert_eq!(patches[0].hunks[1].old_start, 8);
    }

    #[tokio::test]
    async fn patch_rejects_context_past_eof() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short.txt");
        std::fs::write(&path, "only one line\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Hunk expects 3 context lines but file only has 1
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1,3 +1,3 @@\n only one line\n-second line\n+replaced\n third line\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(
            result.is_error,
            "expected error but got: {}",
            result.content
        );
        assert!(
            result.content.contains("mismatch") || result.content.contains("has only"),
            "got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn patch_rejects_remove_past_eof() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short2.txt");
        std::fs::write(&path, "line1\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Hunk tries to remove a line that doesn't exist
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1,2 +1,1 @@\n line1\n-nonexistent\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(
            result.is_error,
            "expected error but got: {}",
            result.content
        );
        assert!(
            result.content.contains("mismatch") || result.content.contains("has only"),
            "got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn patch_rejects_overlapping_hunks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlap.txt");
        std::fs::write(&path, "a\nb\nc\nd\ne\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Two hunks that overlap: first covers lines 1-3, second starts at line 2
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n\
             @@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n\
             @@ -2,3 +2,3 @@\n b\n-c\n+C\n d\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(
            result.is_error,
            "expected error but got: {}",
            result.content
        );
        assert!(
            result.content.contains("Overlapping"),
            "got: {}",
            result.content
        );
    }

    #[test]
    fn extract_path_strips_timestamp() {
        let line = "--- a/file.txt\t2024-01-01 00:00:00.000000000 +0000";
        assert_eq!(extract_path(line), "file.txt");
    }

    #[test]
    fn parse_rejects_invalid_hunk_start() {
        let patch = "--- a/file.txt\n+++ b/file.txt\n@@ -abc,3 +1,3 @@\n line 1\n";
        let err = parse_unified_diff(patch).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Cannot parse hunk start"),
            "expected parse error, got: {msg}"
        );
    }

    // --- Multi-pass fuzzy matching tests ---

    #[test]
    fn fuzzy_match_exact() {
        assert!(fuzzy_lines_match("hello world", "hello world"));
    }

    #[test]
    fn fuzzy_match_trailing_whitespace() {
        assert!(fuzzy_lines_match("hello   ", "hello"));
        assert!(fuzzy_lines_match("hello", "hello   "));
        assert!(fuzzy_lines_match("hello  \t", "hello"));
    }

    #[test]
    fn fuzzy_match_leading_whitespace() {
        assert!(fuzzy_lines_match("  hello", "hello"));
        assert!(fuzzy_lines_match("hello", "  hello"));
        assert!(fuzzy_lines_match("\thello", "hello"));
    }

    #[test]
    fn fuzzy_match_smart_quotes() {
        // Curly double quotes vs straight
        assert!(fuzzy_lines_match("\u{201C}hello\u{201D}", "\"hello\""));
        // Curly single quotes vs straight
        assert!(fuzzy_lines_match("\u{2018}hello\u{2019}", "'hello'"));
    }

    #[test]
    fn fuzzy_match_em_dash() {
        assert!(fuzzy_lines_match("foo\u{2014}bar", "foo-bar"));
        // En dash too
        assert!(fuzzy_lines_match("foo\u{2013}bar", "foo-bar"));
    }

    #[test]
    fn fuzzy_match_non_breaking_space() {
        assert!(fuzzy_lines_match("foo\u{00A0}bar", "foo bar"));
        // Figure space
        assert!(fuzzy_lines_match("foo\u{2007}bar", "foo bar"));
        // Narrow no-break space
        assert!(fuzzy_lines_match("foo\u{202F}bar", "foo bar"));
    }

    #[test]
    fn fuzzy_match_rejects_different() {
        assert!(!fuzzy_lines_match("hello", "world"));
        assert!(!fuzzy_lines_match("abc", "def"));
    }

    #[tokio::test]
    async fn patch_applies_with_trailing_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trailing.txt");
        // File has trailing spaces on lines
        std::fs::write(&path, "line 1   \nline 2  \nline 3\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Patch context/remove lines have NO trailing spaces
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1,3 +1,3 @@\n line 1\n-line 2\n+line TWO\n line 3\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("1 file(s) changed"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("line TWO"));
        assert!(!content.contains("line 2"));
    }

    #[tokio::test]
    async fn patch_applies_with_smart_quotes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quotes.txt");
        // File has smart/curly quotes
        std::fs::write(&path, "say \u{201C}hello\u{201D}\nother line\n").unwrap();

        let tracker = Arc::new(FileTracker::new());
        tracker.record_read(&path).unwrap();

        // Patch uses straight quotes
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1,2 +1,2 @@\n-say \"hello\"\n+say \"goodbye\"\n other line\n",
            path.display()
        );

        let tool = PatchTool::new(tracker, None);
        let result = tool.execute(json!({"patch_text": patch})).await.unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("1 file(s) changed"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("goodbye"));
    }
}
