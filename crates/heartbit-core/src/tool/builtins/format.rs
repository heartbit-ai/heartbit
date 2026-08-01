//! Post-edit formatters for the file-writing builtins.
//!
//! A formatter runs as **stdin → stdout**: the content is piped in and the
//! formatted content read back. The subprocess never receives a path, which is
//! load-bearing for three reasons:
//!
//! 1. The write path is symlink-hardened (F-FS-1, `write_beneath_root` /
//!    `write_no_follow`). Shelling out to `rustfmt <path>` would write the file
//!    outside that hardening.
//! 2. The writers capture the post-write mtime with `FileTracker::record_read`
//!    immediately after their single write. Formatting the buffer BEFORE that
//!    write means the recorded mtime already matches the final bytes — there is
//!    no window in which the guard is stale.
//! 3. Blast radius: a formatter that cannot see a path cannot modify any other
//!    file (`rustfmt lib.rs` would otherwise reformat the whole crate).
//!
//! Every failure mode is **fail-open**: a missing binary, non-zero exit, timeout,
//! empty or non-UTF-8 output all leave the content untouched and the write
//! succeeds. Formatting is a convenience, never a gate.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// Default wall-clock budget for one formatter invocation.
pub const DEFAULT_FORMAT_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum content size handed to a formatter. Larger content is written
/// unformatted rather than paying an unbounded subprocess round-trip.
pub const MAX_FORMAT_BYTES: usize = 2 * 1024 * 1024;

/// Extension → formatter command. An empty config formats nothing.
#[derive(Debug, Clone)]
pub struct FormatterConfig {
    /// Lowercased extension (no dot) → argv. argv\[0\] is the binary.
    by_ext: HashMap<String, Vec<String>>,
    /// Per-invocation wall-clock budget.
    pub timeout: Duration,
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            by_ext: HashMap::new(),
            timeout: DEFAULT_FORMAT_TIMEOUT,
        }
    }
}

impl FormatterConfig {
    /// Register `argv` for files with extension `ext` (case-insensitive, no dot).
    pub fn set(&mut self, ext: &str, argv: Vec<String>) {
        self.by_ext.insert(ext.to_lowercase(), argv);
    }

    /// True when no formatter is configured (the default) — no subprocess ever runs.
    pub fn is_empty(&self) -> bool {
        self.by_ext.is_empty()
    }

    /// The argv for `path`'s extension, if one is configured.
    pub fn command_for(&self, path: &Path) -> Option<&[String]> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        self.by_ext.get(&ext).map(|v| v.as_slice())
    }
}

/// Format `content` for `path`, or return `None` to write it unchanged.
///
/// `None` on every failure path (fail-open). See the module docs.
pub async fn format_content(cfg: &FormatterConfig, path: &Path, content: &str) -> Option<String> {
    if cfg.is_empty() || content.len() > MAX_FORMAT_BYTES {
        return None;
    }
    let argv = cfg.command_for(path)?;
    let (bin, args) = argv.split_first()?;

    let mut child = tokio::process::Command::new(bin)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .ok()?; // missing binary → skip

    let mut stdin = child.stdin.take()?;
    let bytes = content.as_bytes().to_vec();
    // Write stdin CONCURRENTLY with reading stdout: a formatter whose output
    // exceeds the pipe buffer would otherwise deadlock on large files.
    let writer = tokio::spawn(async move {
        use tokio::io::AsyncWriteExt;
        stdin.write_all(&bytes).await?;
        stdin.shutdown().await
    });

    let output = match tokio::time::timeout(cfg.timeout, child.wait_with_output()).await {
        Ok(Ok(o)) => o,
        // Timeout or spawn-level error → skip. `kill_on_drop` reaps the child.
        _ => {
            writer.abort();
            return None;
        }
    };

    // CORRECTNESS: Rust ignores SIGPIPE, so a formatter that exits 0 without
    // reading all of stdin (a `head`-style wrapper, an internal size limit)
    // makes our write fail with a plain EPIPE `io::Error` instead of killing
    // the process — but `output.status` can still read as success with
    // non-empty (partial) stdout. A formatter that never consumed the whole
    // buffer cannot have faithfully transformed it, so only trust the output
    // when the writer finished writing (and closing) stdin without error.
    match writer.await {
        Ok(Ok(())) => {}
        _ => return None,
    }

    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }
    let formatted = String::from_utf8(output.stdout).ok()?; // non-UTF-8 → skip
    Some(formatted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn cfg(ext: &str, cmd: &[&str]) -> FormatterConfig {
        let mut c = FormatterConfig::default();
        c.set(ext, cmd.iter().map(|s| s.to_string()).collect());
        c
    }

    #[test]
    fn default_config_is_empty_and_matches_nothing() {
        let c = FormatterConfig::default();
        assert!(c.is_empty());
        assert!(c.command_for(Path::new("a.rs")).is_none());
    }

    #[test]
    fn extension_lookup_is_case_insensitive() {
        let c = cfg("rs", &["cat"]);
        assert!(c.command_for(Path::new("a.RS")).is_some());
        assert!(c.command_for(Path::new("a.rs")).is_some());
        assert!(c.command_for(Path::new("a.py")).is_none());
        assert!(c.command_for(Path::new("noext")).is_none());
    }

    #[tokio::test]
    async fn formats_through_stdin_stdout() {
        // `tr a-z A-Z` is a formatter with no path argument: proof the contract
        // is content-in / content-out and the subprocess never sees a path.
        let c = cfg("rs", &["tr", "a-z", "A-Z"]);
        let out = format_content(&c, Path::new("a.rs"), "hello").await;
        assert_eq!(out.as_deref(), Some("HELLO"));
    }

    #[tokio::test]
    async fn unconfigured_extension_is_skipped() {
        let c = cfg("rs", &["tr", "a-z", "A-Z"]);
        assert!(
            format_content(&c, Path::new("a.py"), "hello")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn missing_binary_is_silently_skipped() {
        let c = cfg("rs", &["heartbit-no-such-formatter-binary"]);
        assert!(format_content(&c, Path::new("a.rs"), "x").await.is_none());
    }

    #[tokio::test]
    async fn nonzero_exit_is_skipped() {
        let c = cfg("rs", &["false"]);
        assert!(format_content(&c, Path::new("a.rs"), "x").await.is_none());
    }

    #[tokio::test]
    async fn empty_stdout_is_skipped() {
        let c = cfg("rs", &["true"]);
        assert!(format_content(&c, Path::new("a.rs"), "x").await.is_none());
    }

    #[tokio::test]
    async fn non_utf8_stdout_is_skipped() {
        let c = cfg("rs", &["printf", "\\xff\\xfe"]);
        assert!(format_content(&c, Path::new("a.rs"), "x").await.is_none());
    }

    #[tokio::test]
    async fn timeout_is_skipped_and_does_not_hang() {
        let mut c = cfg("rs", &["sleep", "30"]);
        c.timeout = std::time::Duration::from_millis(150);
        let started = std::time::Instant::now();
        assert!(format_content(&c, Path::new("a.rs"), "x").await.is_none());
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[tokio::test]
    async fn partial_stdin_consumption_is_skipped_not_truncated() {
        // `head -c 10` reads only its first chunk of stdin then exits 0,
        // closing the read end of the pipe long before the writer can finish
        // sending 1MB. Rust ignores SIGPIPE, so the writer's write_all fails
        // with a plain io::Error (EPIPE) instead of killing the process —
        // but the child's exit status and (truncated) stdout can still look
        // like a success. A formatter that never consumed the whole buffer
        // cannot have faithfully transformed it: this must fail open, never
        // hand back the truncated bytes it happened to echo.
        let c = cfg("rs", &["head", "-c", "10"]);
        let big = "x".repeat(1_000_000);
        let out = format_content(&c, Path::new("a.rs"), &big).await;
        assert!(out.is_none(), "expected fail-open, got: {out:?}");
    }

    #[tokio::test]
    async fn large_content_does_not_deadlock() {
        // Writing stdin and reading stdout must be concurrent, or a formatter
        // whose output exceeds the pipe buffer deadlocks.
        let c = cfg("rs", &["cat"]);
        let big = "x".repeat(1_000_000);
        let out = format_content(&c, Path::new("a.rs"), &big).await;
        assert_eq!(out.as_deref().map(str::len), Some(1_000_000));
    }
}
