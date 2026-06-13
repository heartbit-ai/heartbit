use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::sandbox::CorePathPolicy;
use crate::tool::{Tool, ToolOutput};

use super::SKIP_DIRS;

const MAX_MATCHES: usize = 100;
/// Max bytes for a single emitted match line (path + line number + content).
/// A single `.d` dep file line under `target/` can be ~100KB; without this an
/// individual line could blow the model context even at MAX_MATCHES = 100.
const MAX_LINE_BYTES: usize = 2_000;
/// Max total bytes of all match lines combined.
const MAX_TOTAL_BYTES: usize = 100_000;
/// Hard cap on raw bytes ingested from rg's stdout. `--max-count` limits
/// matches PER FILE, not globally, so a permissive pattern over a large tree
/// can emit far more than the rendered caps — the reader stops (and rg is
/// killed) once this many bytes are collected, bounding process memory at the
/// source. 10x `MAX_TOTAL_BYTES` leaves room for per-line truncation to still
/// fill the rendered output.
const MAX_RG_STDOUT_BYTES: usize = MAX_TOTAL_BYTES * 10;
/// Cap on rg's stderr (only ever a short error message on exit code 2).
const MAX_RG_STDERR_BYTES: usize = 8_192;

/// Builtin tool that searches file contents for a regex pattern.
///
/// Delegates to the system `grep` binary (with `-r -n --include` flags) and
/// returns up to `MAX_MATCHES = 100` matching lines with file name and line number.
/// The search is scoped to the configured workspace root when set, and respects
/// `protected_paths` to prevent the agent from searching sensitive directories.
///
/// SECURITY (F-FS-4): when a `CorePathPolicy` is set, `path` is checked
/// against it before grep runs. Without this, the LLM can pass an absolute
/// `path = "/home"` and read every file under it.
pub struct GrepTool {
    workspace: Option<PathBuf>,
    protected_paths: Arc<Vec<PathBuf>>,
    path_policy: Option<Arc<CorePathPolicy>>,
}

impl GrepTool {
    pub fn new(workspace: Option<PathBuf>, protected_paths: Arc<Vec<PathBuf>>) -> Self {
        Self {
            workspace,
            protected_paths,
            path_policy: None,
        }
    }

    /// Set a `CorePathPolicy` that restricts the directory grep can search.
    pub fn with_path_policy(mut self, policy: Arc<CorePathPolicy>) -> Self {
        self.path_policy = Some(policy);
        self
    }
}

impl Tool for GrepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".into(),
            description: "Search file contents using regex patterns. Uses ripgrep (rg) when \
                          available, falls back to built-in regex search. Returns matching lines \
                          with file paths and line numbers. Skips build/dependency directories \
                          (target, node_modules, dist, build, .git, __pycache__) and hidden \
                          files; long lines and total output are truncated. Pass a direct file \
                          path to search inside a skipped directory."
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
        _ctx: &crate::ExecutionContext,
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
                Some(p) => {
                    match super::resolve_path(p, self.workspace.as_deref(), &self.protected_paths) {
                        Ok(p) => p,
                        Err(msg) => return Ok(ToolOutput::error(msg)),
                    }
                }
                None => self.workspace.clone().unwrap_or_else(|| PathBuf::from(".")),
            };
            // SECURITY (F-FS-4): apply the path policy to the search root
            // before invoking ripgrep / fallback search. Without this, an LLM
            // can grep `/etc` or `/home` for secrets when no workspace is set.
            if let Some(ref policy) = self.path_policy
                && let Err(e) = policy.check_path(&search_path)
            {
                return Ok(ToolOutput::error(format!("path policy: {e}")));
            }
            let path = search_path.display().to_string();
            if !search_path.exists() {
                return Ok(ToolOutput::error(format!("Path not found: {path}")));
            }

            // When the path is a single file, the skip-dir filter must NOT
            // apply (mirrors the fallback's single-file branch) — the user
            // explicitly targeted that file even if it lives under target/.
            let is_file = search_path.is_file();

            // Try ripgrep first
            match try_ripgrep(pattern, &path, include, literal, is_file).await {
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
    is_file: bool,
) -> Result<ToolOutput, Error> {
    let mut cmd = tokio::process::Command::new("rg");
    cmd.arg("-H")
        .arg("-n")
        .arg("--color")
        .arg("never")
        // Per-FILE match cap (rg has no global one) — the global byte cap is
        // enforced by the capped stdout reader below (MAX_RG_STDOUT_BYTES).
        .arg("--max-count")
        .arg((MAX_MATCHES + 1).to_string());

    if literal {
        cmd.arg("-F");
    }

    // Skip build/dependency dirs for parity with the fallback walker. rg honors
    // .gitignore inside a git repo, but not in non-git dirs — these globs make
    // the behavior consistent everywhere. NOT applied to a single explicit file:
    // rg's `--glob` exclusions outrank the "explicit path overrides ignore" rule,
    // so excluding `target` here would make `grep <file under target/>` return
    // nothing — matching the fallback's single-file branch that skips this.
    if !is_file {
        for dir in SKIP_DIRS {
            cmd.arg("--glob").arg(format!("!{dir}"));
        }
    }

    if let Some(glob_pattern) = include {
        cmd.arg("--glob").arg(glob_pattern);
    }

    cmd.arg(pattern).arg(path);

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    // If this future is dropped mid-flight (interrupt), don't leave rg running.
    cmd.kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .map_err(|e| Error::Agent(format!("rg not available: {e}")))?;

    let mut stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| Error::Agent("rg stdout unavailable".into()))?;
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| Error::Agent("rg stderr unavailable".into()))?;
    // Drain stderr concurrently (retaining only a short head) so rg can never
    // block on a full stderr pipe while we read stdout.
    let stderr_task = tokio::spawn(async move {
        let mut pipe = stderr_pipe;
        read_capped(&mut pipe, MAX_RG_STDERR_BYTES, true).await
    });

    // R2: bound the read + wait with a timeout. Without it, `rg` reading a
    // blocking special file (FIFO/named pipe) or sitting on a hung mount
    // (hard NFS / stalled FUSE) never returns, so this `.await` never completes
    // and the grep tool call hangs the agent forever with no error surfaced.
    // `BashTool` and `codegen/verify.rs` wrap their waits the same way.
    const RG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

    // Global source-side cap: stop ingesting once MAX_RG_STDOUT_BYTES are
    // collected — `--max-count` is only per-file, so a permissive pattern over
    // a large tree could otherwise buffer hundreds of MB before the rendered
    // caps apply.
    let (raw, source_capped) = match tokio::time::timeout(
        RG_TIMEOUT,
        read_capped(&mut stdout_pipe, MAX_RG_STDOUT_BYTES, false),
    )
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(Error::Agent(format!("rg read failed: {e}"))),
        Err(_elapsed) => {
            let _ = child.start_kill();
            stderr_task.abort();
            return Ok(ToolOutput::error(format!(
                "grep timed out after {}s (a blocking special file or stalled filesystem?)",
                RG_TIMEOUT.as_secs()
            )));
        }
    };
    if source_capped {
        // We already hold more than the rendered caps can use; stop rg instead
        // of letting it stream the rest of the tree.
        let _ = child.start_kill();
    }
    let status = match tokio::time::timeout(RG_TIMEOUT, child.wait()).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return Err(Error::Agent(format!("rg failed: {e}"))),
        Err(_elapsed) => {
            let _ = child.start_kill();
            stderr_task.abort();
            return Ok(ToolOutput::error(format!(
                "grep timed out after {}s waiting for rg to exit",
                RG_TIMEOUT.as_secs()
            )));
        }
    };
    let stderr_raw = match stderr_task.await {
        Ok(Ok((bytes, _))) => bytes,
        _ => Vec::new(),
    };

    // rg exit code 0 = matches found, 1 = no matches, 2 = error. A kill after
    // the source cap means matches were found: treat it as success.
    let code = if source_capped {
        Some(0)
    } else {
        status.code()
    };
    match code {
        Some(0) => {
            let stdout = String::from_utf8_lossy(&raw);
            // Apply the same per-line and total-byte caps as the fallback path
            // so a giant single line (e.g. a minified/dep file) can't blow the
            // model context even when rg is the one producing output.
            let mut lines = Vec::with_capacity(MAX_MATCHES);
            let mut total_bytes = 0usize;
            let mut byte_capped = false;
            for line in stdout.lines() {
                if !push_capped(&mut lines, &mut total_bytes, line) {
                    byte_capped = true;
                    break;
                }
                if lines.len() >= MAX_MATCHES {
                    break;
                }
            }
            let count_capped = !byte_capped && lines.len() >= MAX_MATCHES;
            Ok(ToolOutput::success(render_matches(
                &lines,
                byte_capped,
                count_capped,
            )))
        }
        Some(1) => Ok(ToolOutput::success("No matches found.")),
        _ => {
            let stderr = String::from_utf8_lossy(&stderr_raw);
            Err(Error::Agent(format!("rg error: {stderr}")))
        }
    }
}

/// Read from `reader`, retaining at most `cap` bytes. With `drain` set, keeps
/// reading (and discarding) to EOF after the cap so the writer never blocks on
/// a full pipe; otherwise returns as soon as the cap is reached. Returns the
/// collected bytes and whether the cap was hit.
async fn read_capped<R>(reader: &mut R, cap: usize, drain: bool) -> std::io::Result<(Vec<u8>, bool)>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt as _;
    let mut out = Vec::new();
    let mut buf = [0u8; 8_192];
    let mut capped = false;
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        if out.len() < cap {
            let take = n.min(cap - out.len());
            out.extend_from_slice(&buf[..take]);
        }
        if out.len() >= cap {
            capped = true;
            if !drain {
                break;
            }
        }
    }
    Ok((out, capped))
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
    let mut total = 0usize;

    let walker: Box<dyn Iterator<Item = walkdir::DirEntry>> = if path.is_file() {
        // Single-file branch: grep the file as given, even inside a build dir.
        Box::new(
            walkdir::WalkDir::new(path)
                .into_iter()
                .filter_map(|e| e.ok()),
        )
    } else {
        // Directory branch: PRUNE skipped build dirs and hidden dirs via
        // filter_entry (stops descent — fixes the multi-second sweep of
        // target/). The root (depth 0) is always kept so explicitly targeting
        // a dir named e.g. `target`, or tempfile's dot-prefixed temp roots,
        // still works.
        Box::new(
            walkdir::WalkDir::new(path)
                .into_iter()
                .filter_entry(|e| e.depth() == 0 || (!is_skipped_dir(e) && !is_hidden(e)))
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file()),
        )
    };

    'outer: for entry in walker {
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

        let mut byte_capped = false;
        for (line_num, line) in content.lines().enumerate() {
            if re.is_match(line) {
                let entry = format!("{}:{}: {}", file_path.display(), line_num + 1, line);
                if !push_capped(&mut matches, &mut total, &entry) {
                    byte_capped = true;
                    break;
                }
                if matches.len() >= MAX_MATCHES {
                    break;
                }
            }
        }

        if byte_capped {
            return Ok(ToolOutput::success(render_matches(&matches, true, false)));
        }
        if matches.len() >= MAX_MATCHES {
            break 'outer;
        }
    }

    if matches.is_empty() {
        Ok(ToolOutput::success("No matches found."))
    } else {
        Ok(ToolOutput::success(render_matches(
            &matches,
            false,
            matches.len() >= MAX_MATCHES,
        )))
    }
}

/// Render collected match lines with the appropriate truncation footer.
/// The byte-cap footer takes precedence over the count-cap footer.
fn render_matches(matches: &[String], byte_capped: bool, count_capped: bool) -> String {
    let count = matches.len();
    let footer = if byte_capped {
        format!("\n\n(output truncated at {MAX_TOTAL_BYTES} bytes)")
    } else if count_capped {
        format!("\n\n(showing first {MAX_MATCHES} matches, there may be more)")
    } else {
        String::new()
    };
    format!("Found {count} matches\n\n{}{footer}", matches.join("\n"))
}

fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|s| s.starts_with('.'))
}

/// True when `entry` is a directory whose name is in [`SKIP_DIRS`]. Used by
/// `filter_entry` to PRUNE the subtree (stop descent) rather than just filter
/// the dir entry itself — this is what keeps grep from sweeping `target/`.
fn is_skipped_dir(entry: &walkdir::DirEntry) -> bool {
    entry.file_type().is_dir()
        && entry
            .file_name()
            .to_str()
            .is_some_and(|name| SKIP_DIRS.contains(&name))
}

/// Append one `entry` match line to `matches`, applying the per-line and total
/// byte caps shared by the ripgrep and fallback paths.
///
/// Returns `false` when the total cap is hit (caller should stop). On per-line
/// overflow the line is truncated at a UTF-8 boundary with an inline marker.
fn push_capped(matches: &mut Vec<String>, total: &mut usize, entry: &str) -> bool {
    let line = if entry.len() > MAX_LINE_BYTES {
        let cut = super::floor_char_boundary(entry, MAX_LINE_BYTES);
        format!(
            "{} …[line truncated, {} bytes total]",
            &entry[..cut],
            entry.len()
        )
    } else {
        entry.to_string()
    };

    // +1 for the newline that joins lines in the rendered output.
    if *total + line.len() + 1 > MAX_TOTAL_BYTES {
        return false;
    }
    *total += line.len() + 1;
    matches.push(line);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = GrepTool::new(None, Arc::new(Vec::new()));
        assert_eq!(tool.definition().name, "grep");
    }

    #[tokio::test]
    async fn grep_finds_pattern_in_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world\nfoo bar\nhello again\n").unwrap();

        let tool = GrepTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "pattern": "hello",
                    "path": dir.path().to_str().unwrap()
                }),
            )
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

        let tool = GrepTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "pattern": "xyz_not_here",
                    "path": path.to_str().unwrap()
                }),
            )
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

        let tool = GrepTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "pattern": "$5.00",
                    "path": path.to_str().unwrap(),
                    "literal": true
                }),
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("$5.00"));
    }

    #[tokio::test]
    async fn grep_nonexistent_path() {
        let tool = GrepTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "pattern": "test",
                    "path": "/tmp/nonexistent_heartbit_test_dir_12345"
                }),
            )
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

        let tool = GrepTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "pattern": "hello",
                    "path": dir.path().to_str().unwrap(),
                    "include": "*.rs"
                }),
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("match.rs"));
    }

    #[test]
    fn fallback_grep_skips_build_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let deps = dir.path().join("target").join("debug").join("deps");
        std::fs::create_dir_all(&deps).unwrap();
        std::fs::write(deps.join("x.d"), "needle here\n").unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("main.rs"), "needle here\n").unwrap();

        let result = super::fallback_grep("needle", dir.path(), None, false).unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("main.rs"),
            "should find src match: {}",
            result.content
        );
        assert!(
            !result.content.contains("target"),
            "must not descend into target/: {}",
            result.content
        );
    }

    #[test]
    fn fallback_grep_caps_giant_line() {
        let dir = tempfile::tempdir().unwrap();
        // One matching line of ~200KB on a single line.
        let mut huge = String::with_capacity(200_000);
        huge.push_str("needle ");
        huge.push_str(&"a".repeat(200_000));
        huge.push('\n');
        std::fs::write(dir.path().join("big.txt"), &huge).unwrap();

        let result = super::fallback_grep("needle", dir.path(), None, false).unwrap();
        assert!(!result.is_error);
        // Each emitted match line must be capped near MAX_LINE_BYTES (+ marker slack).
        let longest = result.content.lines().map(str::len).max().unwrap_or(0);
        assert!(
            longest <= super::MAX_LINE_BYTES + 80,
            "match line not capped: longest = {longest}"
        );
        assert!(
            result.content.contains("line truncated"),
            "missing truncation indicator: {}",
            &result.content[..result.content.len().min(300)]
        );
        // Total output far below the raw 200KB.
        assert!(
            result.content.len() < 10_000,
            "total output too large: {}",
            result.content.len()
        );
    }

    #[test]
    fn fallback_grep_caps_total_bytes() {
        let dir = tempfile::tempdir().unwrap();
        // ~60 files each with a ~3KB matching line => ~180KB > 100KB cap.
        for i in 0..60 {
            let line = format!("needle {}\n", "x".repeat(3_000));
            std::fs::write(dir.path().join(format!("f{i}.txt")), line).unwrap();
        }

        let result = super::fallback_grep("needle", dir.path(), None, false).unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.len() <= super::MAX_TOTAL_BYTES + 2_000,
            "total output not capped: {}",
            result.content.len()
        );
        assert!(
            result.content.contains("output truncated"),
            "missing total-truncation note: ...{}",
            &result.content[result.content.len().saturating_sub(120)..]
        );
    }

    #[test]
    fn fallback_grep_explicit_file_in_build_dir_still_works() {
        let dir = tempfile::tempdir().unwrap();
        let deps = dir.path().join("target").join("debug").join("deps");
        std::fs::create_dir_all(&deps).unwrap();
        let file = deps.join("x.d");
        std::fs::write(&file, "needle here\n").unwrap();

        // Path points DIRECTLY at the file inside target/ -> single-file branch,
        // skip-dirs must NOT apply.
        let result = super::fallback_grep("needle", &file, None, false).unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("needle"),
            "explicit file in build dir should still grep: {}",
            result.content
        );
    }

    // End-to-end through execute() — exercises the rg path when rg is installed,
    // the fallback otherwise. Both must skip build dirs on a directory search.
    #[tokio::test]
    async fn grep_execute_skips_build_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let deps = dir.path().join("target").join("debug").join("deps");
        std::fs::create_dir_all(&deps).unwrap();
        std::fs::write(deps.join("x.d"), "needle here\n").unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("main.rs"), "needle here\n").unwrap();

        let tool = GrepTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }),
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("main.rs"),
            "should find src match: {}",
            result.content
        );
        assert!(
            !result.content.contains("target"),
            "must not include target/ paths: {}",
            result.content
        );
    }

    // The skip-dir exclusion must NOT block an explicitly targeted file inside a
    // build dir — on the rg path this means NOT passing `--glob '!target'` for a
    // single file (rg's glob exclusions outrank the explicit-path-overrides rule).
    #[tokio::test]
    async fn grep_execute_explicit_file_in_build_dir_still_works() {
        let dir = tempfile::tempdir().unwrap();
        let deps = dir.path().join("target").join("debug").join("deps");
        std::fs::create_dir_all(&deps).unwrap();
        let file = deps.join("x.d");
        std::fs::write(&file, "needle here\n").unwrap();

        let tool = GrepTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({ "pattern": "needle", "path": file.to_str().unwrap() }),
            )
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("needle"),
            "explicit file in build dir should still grep (rg path): {}",
            result.content
        );
    }

    // ---- source-side rg stdout cap (audit 2026-06-09) ----

    #[tokio::test]
    async fn read_capped_stops_at_cap() {
        let data = vec![b'a'; 3 * MAX_RG_STDOUT_BYTES];
        let mut reader = &data[..];
        let (out, capped) = super::read_capped(&mut reader, MAX_RG_STDOUT_BYTES, false)
            .await
            .unwrap();
        assert!(capped, "3x the cap must report capped");
        assert_eq!(
            out.len(),
            MAX_RG_STDOUT_BYTES,
            "must retain exactly the cap"
        );
        assert!(
            !reader.is_empty(),
            "without drain, reading must stop at the cap (memory/latency bound)"
        );
    }

    #[tokio::test]
    async fn read_capped_small_input_uncapped() {
        let data = b"hello".to_vec();
        let mut reader = &data[..];
        let (out, capped) = super::read_capped(&mut reader, 100, false).await.unwrap();
        assert!(!capped);
        assert_eq!(out, b"hello");
    }

    #[tokio::test]
    async fn read_capped_drain_consumes_to_eof_but_retains_cap() {
        let data = vec![b'x'; 100_000];
        let mut reader = &data[..];
        let (out, capped) = super::read_capped(&mut reader, 1_000, true).await.unwrap();
        assert!(capped);
        assert_eq!(out.len(), 1_000);
        assert!(reader.is_empty(), "drain mode must consume to EOF");
    }

    // End-to-end: a search whose raw match volume exceeds MAX_RG_STDOUT_BYTES
    // must still return a correctly capped result (rg path: source-capped read
    // + kill; fallback path: walker caps). Exercises whichever path is live.
    #[tokio::test]
    async fn grep_execute_source_volume_beyond_rg_cap_still_capped() {
        let dir = tempfile::tempdir().unwrap();
        // 15 files x 101 matching lines x ~800 bytes ≈ 1.2MB of raw matches,
        // beyond MAX_RG_STDOUT_BYTES (1MB) even with --max-count 101 per file.
        let line = format!("needle {}\n", "y".repeat(800));
        for i in 0..15 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), line.repeat(101)).unwrap();
        }

        let tool = GrepTool::new(None, Arc::new(Vec::new()));
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }),
            )
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.content);
        assert!(
            result.content.len() <= MAX_TOTAL_BYTES + 2_000,
            "rendered output not capped: {}",
            result.content.len()
        );
        assert!(
            result.content.contains("truncated") || result.content.contains("showing first"),
            "missing truncation footer: ...{}",
            &result.content[result.content.len().saturating_sub(150)..]
        );
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
