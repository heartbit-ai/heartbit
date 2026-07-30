# TUI Wave 1 — Table Stakes + Shelfware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap between heartbit's TUI and its own engine (persistent approvals, reasoning-effort control), fix the input plane (bracketed paste, focus, Shift+Enter), make the reading surface legible (syntax highlighting, word-level diffs, `/diff`), and stop wasting LLM turns on formatting.

**Architecture:** heartbit-tui is an Elm-style reducer — `Msg → App → Effect` in `app.rs`; the main loop in `main.rs` performs every effect. The reducer must stay I/O-free (that purity is what makes its 301 tests possible). One heartbit-core change (C3, post-edit formatters); everything else is TUI-only. Every core change is additive with a default that reproduces today's behaviour bit-for-bit.

**Tech Stack:** Rust (edition 2024), ratatui 0.29 (`unstable-rendered-line-info`), crossterm 0.28 (`event-stream`, `bracketed-paste`), tokio, syntect 5 (new dep, Task 8).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-26-tui-wave1-table-stakes-design.md`; framework `2026-06-26-tui-sota-upgrade-design.md`. Read the spec's §2 decisions (D-1..D-5) before starting.
- **Gate, workspace-wide, after every task** (lesson 2026-06-13 — `-p heartbit-tui` misses Task 4's core test and Task 9's `DiffLine` reshape):
  `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
- **The reducer performs no I/O.** No `std::fs`, `std::process`, `std::env`, `reqwest` or `crossterm::execute!` inside `App::update` or any `handle_*` helper. Terminal writes and subprocesses belong to the main loop's effect pass.
- **Additive core only.** After Task 1, `git diff --stat -- crates/heartbit-core` must be empty for Tasks 2–9 except the one new test in Task 4.
- **Exactly one existing test changes in this whole plan:** `crates/heartbit-tui/src/main.rs:1876-1883` in Task 4. Any other pre-existing test that needs editing is a signal you broke something — stop and re-read the spec.
- **New `tui.toml` fields are plain `bool` or `Option<String>`, never a typed enum.** `TuiConfig::load_from` (`config.rs:160-165`) swallows any parse error and returns `Default`, so one typo in a typed field would silently wipe the whole config including `openrouter_api_key`.
- **Commit after every task.** Message prefix per the repo's convention (`feat(tui):`, `fix(tui):`, `feat(core):`). End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Terminal-dependent behaviour is never claimed as proven by `cargo test`** (framework §7.5). Paste, Kitty flags and notifications are verified by the manual script in the spec's §7.

---

## File Structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| `crates/heartbit-core/src/tool/builtins/format.rs` | **new** — `FormatterConfig`, `format_content()`: run a formatter as stdin→stdout, fail-open | 1 |
| `crates/heartbit-core/src/tool/builtins/{mod,write,edit,patch}.rs` | wire `formatters` into the three writers, before the single write | 1 |
| `crates/heartbit-tui/src/app.rs` | the reducer: Ctrl+U, focus, queue, `/effort`, notify gating | 2,3,5,6,7 |
| `crates/heartbit-tui/src/ui.rs` | approval hint, queue rendering, highlight-cache frame boundary | 2,6,8 |
| `crates/heartbit-tui/src/main.rs` | terminal modes, panic hook, `translate()`, engine wiring, effect pass | 3,4,5,7,9 |
| `crates/heartbit-tui/src/msg.rs` | `Msg::FocusChanged` | 3 |
| `crates/heartbit-tui/src/config.rs` | `keyboard_enhancement`, `notify`, `reasoning_effort`, `syntax_theme` | 3,5,7,8 |
| `crates/heartbit-tui/src/notify.rs` | **new** — pure escape-sequence formatting + sanitizing; one thin `emit()` | 7 |
| `crates/heartbit-tui/src/markdown.rs` | syntect highlighting of fenced blocks | 8 |
| `crates/heartbit-tui/src/diff.rs` + `cells.rs` | `DiffLine.emph`, word-level pairing, shared render | 9 |
| `crates/heartbit-tui/src/gitdiff.rs` | **new** — pure unified-diff parse for `/diff` (git I/O lives in main.rs) | 9 |

---

## Task 1: Post-edit formatters (heartbit-core, C3)

Placed first: it is the only core change, it touches the security-hardened write path, and it shares **no file** with Tasks 2–9 (the spec calls for it "in parallel from the start").

**Files:**
- Create: `crates/heartbit-core/src/tool/builtins/format.rs`
- Modify: `crates/heartbit-core/src/tool/builtins/mod.rs` (add `pub mod format;` + the `BuiltinToolsConfig.formatters` field)
- Modify: `crates/heartbit-core/src/tool/builtins/write.rs:147` (format `content` before `let bytes = content.len();`)
- Modify: `crates/heartbit-core/src/tool/builtins/edit.rs:~177` and `patch.rs:~277` (same, before their single write)
- Test: in `format.rs` (`#[cfg(test)] mod tests`) + one integration test per writer in its own file

**Interfaces:**
- Consumes: `super::{write_no_follow, write_beneath_root}` (`mod.rs:84`, `:134`), `FileTracker::record_read` (`file_tracker.rs:41`)
- Produces: `pub struct FormatterConfig`, `pub const DEFAULT_FORMAT_TIMEOUT: Duration`, `pub async fn format_content(cfg: &FormatterConfig, path: &Path, content: &str) -> Option<String>`, and `BuiltinToolsConfig.formatters: Option<FormatterConfig>`

- [ ] **Step 1: Write the failing tests** — create `crates/heartbit-core/src/tool/builtins/format.rs` with the module doc, the types, and this test module:

```rust
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
        assert!(format_content(&c, Path::new("a.py"), "hello").await.is_none());
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
    async fn large_content_does_not_deadlock() {
        // Writing stdin and reading stdout must be concurrent, or a formatter
        // whose output exceeds the pipe buffer deadlocks.
        let c = cfg("rs", &["cat"]);
        let big = "x".repeat(1_000_000);
        let out = format_content(&c, Path::new("a.rs"), &big).await;
        assert_eq!(out.as_deref().map(str::len), Some(1_000_000));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p heartbit-core --lib builtins::format`
Expected: FAIL to compile — `FormatterConfig`, `format_content`, `DEFAULT_FORMAT_TIMEOUT` are not defined.

- [ ] **Step 3: Implement `format.rs`**

```rust
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
        let _ = stdin.write_all(&bytes).await;
        let _ = stdin.shutdown().await;
    });

    let output = match tokio::time::timeout(cfg.timeout, child.wait_with_output()).await {
        Ok(Ok(o)) => o,
        // Timeout or spawn-level error → skip. `kill_on_drop` reaps the child.
        _ => {
            writer.abort();
            return None;
        }
    };
    let _ = writer.await;

    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }
    let formatted = String::from_utf8(output.stdout).ok()?; // non-UTF-8 → skip
    if formatted == content {
        return None; // no change: keep the original buffer
    }
    Some(formatted)
}
```

Register the module in `crates/heartbit-core/src/tool/builtins/mod.rs` next to the other `pub mod` lines:

```rust
pub mod format;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p heartbit-core --lib builtins::format`
Expected: PASS, 10 tests.

- [ ] **Step 5: Add the config field**

In `crates/heartbit-core/src/tool/builtins/mod.rs`, add to `BuiltinToolsConfig` (after `path_policy`):

```rust
    /// Optional post-edit formatters. `None` (default) = no subprocess ever
    /// runs and every write is byte-identical to today's.
    pub formatters: Option<Arc<format::FormatterConfig>>,
```

`BuiltinToolsConfig` derives/implements `Default`; confirm `formatters` defaults to `None` (add `formatters: None` if `Default` is hand-written).

- [ ] **Step 6: Write the failing writer-integration test** (in `write.rs`'s test module)

```rust
    #[tokio::test]
    async fn write_formats_content_before_the_single_write() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("a.rs");
        let tracker = std::sync::Arc::new(FileTracker::new());
        let mut fc = crate::tool::builtins::format::FormatterConfig::default();
        fc.set("rs", vec!["tr".into(), "a-z".into(), "A-Z".into()]);

        let tool = WriteTool::new(tracker.clone(), None, std::sync::Arc::new(Vec::new()))
            .with_formatters(std::sync::Arc::new(fc));
        tool.execute(
            &crate::ExecutionContext::default(),
            serde_json::json!({"file_path": target.to_str().unwrap(), "content": "hello"}),
        )
        .await
        .unwrap();

        // On disk: formatted.
        assert_eq!(tokio::fs::read_to_string(&target).await.unwrap(), "HELLO");
        // The mtime guard matches the FINAL bytes — a follow-up edit with no
        // intervening read must pass, which is the whole point of formatting
        // before the single write.
        assert!(tracker.check_unmodified(&target).is_ok());
    }

    #[tokio::test]
    async fn write_without_formatters_is_byte_identical() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("a.rs");
        let tracker = std::sync::Arc::new(FileTracker::new());
        let tool = WriteTool::new(tracker, None, std::sync::Arc::new(Vec::new()));
        tool.execute(
            &crate::ExecutionContext::default(),
            serde_json::json!({"file_path": target.to_str().unwrap(), "content": "hello"}),
        )
        .await
        .unwrap();
        assert_eq!(tokio::fs::read_to_string(&target).await.unwrap(), "hello");
    }
```

Run: `cargo test -p heartbit-core --lib builtins::write` → FAIL: no method `with_formatters`.

- [ ] **Step 7: Wire the formatter into the three writers**

For each of `write.rs`, `edit.rs`, `patch.rs`: add the field + builder method mirroring the existing `with_path_policy` (`edit.rs:46`, `patch.rs:47`):

```rust
    /// Optional post-edit formatters (applied to the buffer before the write).
    formatters: Option<Arc<crate::tool::builtins::format::FormatterConfig>>,
```

```rust
    /// Format the content with these formatters before writing.
    #[must_use]
    pub fn with_formatters(
        mut self,
        formatters: Arc<crate::tool::builtins::format::FormatterConfig>,
    ) -> Self {
        self.formatters = Some(formatters);
        self
    }
```

In `write.rs`, immediately **before** `let bytes = content.len();` (`write.rs:147`):

```rust
            // Format in memory BEFORE the single write: keeps the post-write
            // record_read mtime matching the final bytes, keeps the returned
            // snippet consistent with disk, and never hands the subprocess a
            // path (F-FS-1 symlink hardening stays in force).
            let content = match &self.formatters {
                Some(fc) => match super::format::format_content(fc, &target, &content).await {
                    Some(formatted) => formatted,
                    None => content,
                },
                None => content,
            };
```

Apply the identical block in `edit.rs` before its write at `:177` (so `format_edit_snippet(&new_content, …)` at `:194` sees the formatted text) and in `patch.rs` before `:277`. Finally, in `builtin_tools(...)`, thread `config.formatters` into all three constructors:

```rust
    let mut write_tool = WriteTool::new(/* … existing args … */);
    if let Some(fc) = config.formatters.clone() {
        write_tool = write_tool.with_formatters(fc);
    }
```

- [ ] **Step 8: Run the writer tests**

Run: `cargo test -p heartbit-core --lib builtins::` — Expected: PASS, including the two new tests and every pre-existing write/edit/patch test unchanged.

- [ ] **Step 9: Run the full gate and commit**

```bash
cargo fmt --all && cargo fmt --all -- --check && \
cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add crates/heartbit-core/src/tool/builtins/
git commit -m "$(cat <<'EOF'
feat(core): post-edit formatters for write/edit/patch (C3)

Formatters run as stdin->stdout on the content in memory, BEFORE the single
write. The subprocess never receives a path, which (a) keeps the post-write
FileTracker::record_read mtime matching the final bytes, (b) keeps the snippet
returned to the model consistent with disk, (c) preserves the F-FS-1
write_beneath_root/write_no_follow symlink hardening, and (d) bounds blast
radius (a formatter cannot touch another file).

Fail-open on every path: missing binary, non-zero exit, timeout, empty or
non-UTF-8 output leave the content untouched and the write succeeds. stdin is
written concurrently with reading stdout so large files cannot deadlock.

Additive: BuiltinToolsConfig.formatters defaults to None, so no subprocess runs
and every write stays byte-identical.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Two micro-defects (0.6)

**Files:**
- Modify: `crates/heartbit-tui/src/app.rs:1837` (Ctrl+U)
- Modify: `crates/heartbit-tui/src/ui.rs:416` (approval hint)
- Test: `crates/heartbit-tui/src/app.rs` (reducer tests), `crates/heartbit-tui/src/ui.rs` (render test)

**Interfaces:**
- Consumes: `Composer::clear()` (`composer.rs:224` — preserves `history`, unlike `Composer::new()`)
- Produces: nothing new

- [ ] **Step 1: Write the failing tests** (in `app.rs`'s `mod tests`)

```rust
    #[test]
    fn ctrl_u_clears_the_draft_but_keeps_recall_history() {
        let mut app = keyed();
        app.composer.seed_history(vec!["earlier prompt".into()]);
        typed(&mut app, "a draft\nwith two lines");
        app.update(key_mod(KeyCode::Char('u'), KeyModifiers::CONTROL));
        // The draft is gone and the cursor is genuinely reset (row too).
        assert!(app.composer.text().is_empty());
        assert_eq!(app.composer.cursor(), (0, 0));
        // …but the seeded history survives: Up recalls it.
        app.update(key(KeyCode::Up));
        assert_eq!(app.composer.text(), "earlier prompt");
    }
```

and in `ui.rs`'s `mod tests`:

```rust
    #[test]
    fn approval_modal_hint_lists_every_answer_key() {
        let mut app = App::new("m");
        app.modal = Some(Modal::Approval(ApprovalModal {
            calls: vec![("bash".into(), "{}".into())],
            ..Default::default()
        }));
        let frame = render_to_string(&app, 100, 30);
        // Every key the reducer actually handles must be advertised.
        for k in ["y", "n", "a", "d"] {
            assert!(
                frame.contains(&format!("[{k}]")),
                "approval hint must advertise [{k}]:\n{frame}"
            );
        }
    }
```

Adapt `keyed()`, `typed()`, `key()`, `key_mod()`, `render_to_string()` and the `ApprovalModal` construction to the helpers already present in those test modules — read them first; do not invent new helpers.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p heartbit-tui ctrl_u_clears_the_draft_but_keeps_recall_history approval_modal_hint_lists_every_answer_key`
Expected: FAIL — `Up` yields an empty composer (history wiped), and the hint has no `[d]`.

- [ ] **Step 3: Fix both defects**

`app.rs:1837`:

```rust
            // Ctrl+U clears the DRAFT only — the recall history (seeded from
            // previous sessions in this directory) must survive.
            KeyCode::Char('u') if ctrl => self.composer.clear(),
```

`ui.rs:416`: add `[d]` to the hint literal, matching the keys `handle_approval_key` (`app.rs:2166-2182`) actually handles, in the same visual style as the existing entries.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p heartbit-tui` — Expected: PASS, 303 tests (301 + 2).

- [ ] **Step 5: Gate and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add crates/heartbit-tui/src/app.rs crates/heartbit-tui/src/ui.rs
git commit -m "$(cat <<'EOF'
fix(tui): Ctrl+U keeps recall history; approval hint advertises [d]

Ctrl+U replaced the whole Composer, discarding the per-directory prompt history
seeded at startup — so Up-arrow recall silently died the first time a user
cleared a draft. Clear the draft only (Composer::clear preserves history).

The approval modal handles [d] = AlwaysDeny but never advertised it.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Terminal modes — bracketed paste, focus events, Kitty flags (0.5 + 0.4)

Both items write `main.rs:320-324`, `:341-342` and the panic hook, so they ship as one pass (spec §4).

**Files:**
- Modify: `crates/heartbit-tui/src/main.rs:320-324` (enable), `:341-342` (teardown), the panic hook, `translate()` `:1096-1111`
- Modify: `crates/heartbit-tui/src/msg.rs` (`Msg::FocusChanged(bool)`)
- Modify: `crates/heartbit-tui/src/app.rs` (`App::focused`, the `Msg::FocusChanged` arm, the splash check in `Msg::Paste`)
- Modify: `crates/heartbit-tui/src/config.rs` (`keyboard_enhancement: bool`, default `true`)
- Test: `app.rs` (reducer), `main.rs` (`translate` + the escape-sequence bytes)

**Interfaces:**
- Consumes: `Composer::insert_str` (`composer.rs:180-188` — already converts `\n`→`newline()`, drops `\r`)
- Produces: `Msg::FocusChanged(bool)`, `App::focused: bool` (default `true`) — Task 7 reads `App::focused`

- [ ] **Step 1: Write the failing reducer + unit tests**

```rust
    // app.rs — mod tests
    #[test]
    fn multiline_paste_lands_as_one_draft_and_does_not_submit() {
        let mut app = keyed();
        app.update(Msg::Paste("line one\nline two\nline three".into()));
        assert_eq!(app.composer.text(), "line one\nline two\nline three");
        // A paste NEVER submits.
        assert!(!app.effects.iter().any(|e| matches!(e, Effect::SendInput(_))));
        assert!(app.history.is_empty());
    }

    #[test]
    fn paste_mid_draft_preserves_the_tail_and_leaves_cursor_after_insert() {
        let mut app = keyed();
        typed(&mut app, "ab");
        app.update(key(KeyCode::Left)); // cursor between a and b
        app.update(Msg::Paste("X\nY".into()));
        assert_eq!(app.composer.text(), "aX\nYb");
    }

    #[test]
    fn crlf_paste_yields_single_newlines() {
        let mut app = keyed();
        app.update(Msg::Paste("a\r\nb".into()));
        assert_eq!(app.composer.text(), "a\nb");
    }

    #[test]
    fn paste_during_splash_dismisses_the_overlay_and_keeps_the_text() {
        let mut app = keyed();
        app.splash = Some(0);
        app.update(Msg::Paste("hello".into()));
        assert!(app.splash.is_none(), "the paste must dismiss the splash");
        assert_eq!(app.composer.text(), "hello");
    }

    #[test]
    fn focus_defaults_to_focused_and_tracks_both_directions() {
        let mut app = App::new("m");
        assert!(app.focused, "a terminal that never reports focus must read as focused");
        app.update(Msg::FocusChanged(false));
        assert!(!app.focused);
        app.update(Msg::FocusChanged(true));
        assert!(app.focused);
    }
```

```rust
    // main.rs — mod tests (new or existing)
    #[test]
    fn translate_maps_focus_events_and_paste() {
        use crossterm::event::Event;
        assert!(matches!(translate(Event::FocusGained), Some(Msg::FocusChanged(true))));
        assert!(matches!(translate(Event::FocusLost), Some(Msg::FocusChanged(false))));
        assert!(matches!(translate(Event::Paste("x".into())), Some(Msg::Paste(s)) if s == "x"));
    }

    #[test]
    fn kitty_push_and_pop_emit_the_minimal_sequences() {
        // Exactly DISAMBIGUATE_ESCAPE_CODES: any future flag addition must be a
        // deliberate, test-updating change (spec D-5).
        assert_eq!(kitty_push_sequence(), "\x1b[>1u");
        assert_eq!(kitty_pop_sequence(), "\x1b[<1u");
    }

    #[test]
    fn kitty_pop_is_emitted_at_most_once() {
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert!(take_pushed(&flag));
        assert!(!take_pushed(&flag), "a second teardown must not re-emit the pop");
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p heartbit-tui multiline_paste paste_mid_draft crlf_paste paste_during_splash focus_defaults translate_maps kitty_`
Expected: FAIL to compile — `Msg::FocusChanged`, `App::focused`, `kitty_push_sequence`, `take_pushed` do not exist; `paste_during_splash` fails at runtime (the splash is not dismissed).

- [ ] **Step 3: Implement**

`msg.rs` — add to `enum Msg`:

```rust
    /// The terminal window gained (`true`) or lost (`false`) focus. Requires
    /// `EnableFocusChange`; terminals that do not report it never send this, so
    /// `App::focused` stays `true` and every consumer sees today's behaviour.
    FocusChanged(bool),
```

`app.rs` — add the field near `pub running: bool`, initialise `focused: true` in `App::new`, and add the arm:

```rust
            Msg::FocusChanged(focused) => self.focused = focused,
```

Extend the `Msg::Paste` arm (`app.rs:738`) with the same splash guard the key path uses at `:761` — dismiss the overlay, then fall through to the existing insert.

`main.rs` — `translate()` (`:1096-1111`), before the `_ => None` arm:

```rust
        Event::FocusGained => Some(Msg::FocusChanged(true)),
        Event::FocusLost => Some(Msg::FocusChanged(false)),
```

`main.rs` — the terminal-mode helpers and the wrapped panic hook:

```rust
/// Exactly `DISAMBIGUATE_ESCAPE_CODES` (spec D-5): it is sufficient for
/// Shift+Enter (`CSI 13;2u` → `'\r'` → `KeyCode::Enter` + SHIFT) and it leaves
/// Shift+Tab as `BackTab`, so the permission-mode cycle survives. Do NOT add
/// `REPORT_EVENT_TYPES`: `KeyEvent::kind` is only populated under it on Unix and
/// `translate` admits only `KeyEventKind::Press`, so held keys would stop
/// auto-repeating.
fn kitty_push_sequence() -> &'static str {
    "\x1b[>1u"
}

fn kitty_pop_sequence() -> &'static str {
    "\x1b[<1u"
}

/// True exactly once, for the first caller. Guarantees the pop is emitted at
/// most once even when both the panic hook and the normal exit path run.
fn take_pushed(flag: &std::sync::atomic::AtomicBool) -> bool {
    flag.swap(false, std::sync::atomic::Ordering::SeqCst)
}

static KITTY_PUSHED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Disable every terminal mode this process enabled, newest first. Safe to call
/// from the panic hook and from the normal exit path — the Kitty pop is guarded
/// so it is emitted at most once.
fn restore_terminal_modes() {
    use std::io::stdout;
    if take_pushed(&KITTY_PUSHED) {
        // Its own execute!: queue! short-circuits on the first error, and the
        // kitty pop must not be lost because a later command failed.
        let _ = crossterm::execute!(stdout(), crossterm::event::PopKeyboardEnhancementFlags);
    }
    let _ = crossterm::execute!(
        stdout(),
        crossterm::event::DisableBracketedPaste,
        crossterm::event::DisableFocusChange,
        crossterm::event::DisableMouseCapture
    );
}
```

At the enable site (`main.rs:324`), after `ratatui::init()`:

```rust
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::event::EnableMouseCapture,
        crossterm::event::EnableBracketedPaste,
        crossterm::event::EnableFocusChange
    );
    // Pushed UNCONDITIONALLY (spec D-3): no capability probe.
    // `supports_keyboard_enhancement()` blocks up to 2000 ms and errors on
    // terminals that do not answer — including our own pty harness — and it
    // would buy nothing there, because Alt+Enter already works everywhere. A
    // private-mode CSI is ignored by terminals that do not implement it.
    if cfg.keyboard_enhancement {
        if crossterm::execute!(
            std::io::stdout(),
            crossterm::event::PushKeyboardEnhancementFlags(
                crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        )
        .is_ok()
        {
            KITTY_PUSHED.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    // WRAP ratatui's panic hook, never replace it: replacing loses its own
    // restore() and would leave raw mode on after a panic.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal_modes();
        prev_hook(info);
    }));
```

At the normal teardown (`main.rs:341-342`), call `restore_terminal_modes();` **before** `ratatui::restore()`.

`config.rs` — add `pub keyboard_enhancement: bool` to `TuiConfig` and `keyboard_enhancement: true` to its hand-written `Default` (a plain `bool`, per the Global Constraints).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p heartbit-tui` — Expected: PASS (308 tests).

- [ ] **Step 5: Verify manually in a real terminal** (this half cannot be proven by `cargo test`)

In Kitty/Ghostty/WezTerm/foot: paste a 5-line block → **one** draft; Shift+Enter → newline; Alt+Enter → newline (unchanged). In xterm: paste still lands as one draft, and **no stray characters** appear from the unconditional push. Force a panic → the shell is usable with no `stty sane`, and no `[I`/`[O` appear when the window gains/loses focus.

- [ ] **Step 6: Gate and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add crates/heartbit-tui/src/{main.rs,app.rs,msg.rs,config.rs}
git commit -m "$(cat <<'EOF'
feat(tui): bracketed paste, focus tracking and Kitty keyboard flags

Bracketed paste was never enabled although the crossterm feature was compiled
in and the Event::Paste arm existed, so a pasted newline arrived as Enter and
submitted the prompt early. The reducer already handled it correctly
(insert_str converts \n to newline) — only the terminal-mode enable was missing.

Also: EnableFocusChange + Msg::FocusChanged (App::focused defaults true, so
terminals that never report focus behave exactly as before) — Task 7 needs it;
and PushKeyboardEnhancementFlags(DISAMBIGUATE_ESCAPE_CODES) so Shift+Enter
receives its modifier. Pushed unconditionally: the capability probe blocks up to
2000ms on terminals that gain nothing from it, and Alt+Enter already worked.

Panic-safe restore WRAPS ratatui's hook instead of replacing it, and the Kitty
pop is guarded to fire at most once. Msg::Paste now honours the splash overlay.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Persistent approval rules (0.1)

Per spec D-2 this is a **bug fix**: today's `[a]` does not even hold for the rest of the session.

**Files:**
- Modify: `crates/heartbit-tui/src/main.rs:1038-1066` (`default_permissions` → `merged_permissions`, drop the terminal catch-all), `:987` (builder), the `on_approval` tail `:853`, and load before `:958`
- Modify (existing test): `crates/heartbit-tui/src/main.rs:1876-1883`
- Create test: in `crates/heartbit-core/src/agent/runner.rs` (`ask_and_none_both_route_to_approval`)
- Test: `main.rs` (`merged_permissions` unit tests)

**Interfaces:**
- Consumes: `OrchestratorBuilder::learned_permissions(Arc<Mutex<LearnedPermissions>>)` (`orchestrator.rs:2606` — verified to forward to the entry runner `:3153-3155` **and** all three sub-agent spawn paths `:610`, `:1150`, `:1593`, so one call suffices); `LearnedPermissions::{load, save, rules, default_path}` (`permission.rs:159-292`)
- Produces: `fn merged_permissions(learned: &[PermissionRule]) -> PermissionRuleset`

- [ ] **Step 1: Write the failing tests**

```rust
    // main.rs — mod tests
    #[test]
    fn merged_permissions_with_no_learned_rules_matches_today() {
        let rules = merged_permissions(&[]);
        // The 10 explicit allows are unchanged…
        for tool in ["read", "grep", "glob", "list", "todoread", "todowrite"] {
            assert_eq!(
                rules.evaluate(tool, &serde_json::json!({})),
                Some(PermissionAction::Allow),
                "{tool} must still be auto-allowed"
            );
        }
        // …and the terminal catch-all is GONE: unmatched tools fall through to
        // None, which runner.rs:2097 routes to approval exactly like Some(Ask).
        for tool in ["write", "edit", "bash", "patch"] {
            assert_eq!(rules.evaluate(tool, &serde_json::json!({})), None, "{tool}");
        }
    }

    #[test]
    fn merged_permissions_orders_learned_rules_first() {
        let learned = vec![PermissionRule {
            tool: "bash".into(),
            pattern: "*".into(),
            action: PermissionAction::Allow,
        }];
        // evaluate is first-match-wins, so a learned rule must precede the
        // defaults or it can never be reached.
        assert_eq!(
            merged_permissions(&learned).evaluate("bash", &serde_json::json!({})),
            Some(PermissionAction::Allow)
        );
    }
```

```rust
    // crates/heartbit-core/src/agent/runner.rs — mod tests
    /// Pins the arm the TUI's permission model depends on: dropping the terminal
    /// `*/*→Ask` rule is only safe because `None` and `Some(Ask)` both route to
    /// human approval. If this ever diverges, the TUI silently becomes Yolo.
    #[test]
    fn ask_and_none_both_route_to_approval() {
        use crate::agent::permission::{PermissionAction, PermissionRule, PermissionRuleset};
        let asked = PermissionRuleset::new(vec![PermissionRule {
            tool: "bash".into(),
            pattern: "*".into(),
            action: PermissionAction::Ask,
        }]);
        assert_eq!(
            asked.evaluate("bash", &serde_json::json!({})),
            Some(PermissionAction::Ask)
        );
        // An unmatched tool yields None — and the runner treats the two
        // identically at runner.rs:2097 (`Some(Ask) | None => needs_approval`).
        assert_eq!(asked.evaluate("write", &serde_json::json!({})), None);
    }
```

Also change the existing assertion at `main.rs:1876-1883` from `Some(PermissionAction::Ask)` to `None` for write/edit/bash/patch, adding a comment naming `runner.rs:2097` as the reason it is equivalent.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p heartbit-tui merged_permissions && cargo test -p heartbit-core --lib ask_and_none`
Expected: FAIL — `merged_permissions` does not exist.

- [ ] **Step 3: Implement**

Rename `default_permissions()` to `merged_permissions(learned: &[PermissionRule]) -> PermissionRuleset`; build the vec as **learned rules first**, then the 10 existing allows, and **delete** the terminal `{tool:"*", pattern:"*", action:Ask}` block (`main.rs:1060-1064`). Update the doc comment at `:1035`, which today already (falsely) claims `[a]` persists.

Load before the "ready — …" send (`main.rs:958`), so the count reaches that line:

```rust
    // Learned approval rules, persisted 0600 next to tui.toml. Loaded BEFORE the
    // "ready — …" summary is sent at :958 so the count can appear in it.
    let learned_path = config::learned_permissions_path();
    let learned = std::sync::Arc::new(std::sync::Mutex::new(
        heartbit_core::agent::permission::LearnedPermissions::load(&learned_path)
            .unwrap_or_else(|_| {
                heartbit_core::agent::permission::LearnedPermissions::new(learned_path.clone())
            }),
    ));
    let learned_rules = learned.lock().expect("learned permissions").rules().to_vec();
    if !learned_rules.is_empty() {
        summary_parts.push(format!("{} learned rules", learned_rules.len()));
    }
```

Replace `main.rs:987` with `.permission_rules(merged_permissions(&learned_rules))` and add `.learned_permissions(learned.clone())` to the same builder chain.

Add `learned_permissions_path()` to `config.rs` beside the existing `tui.toml` path helper, honouring `HEARTBIT_TUI_CONFIG` so the acceptance script can isolate it.

In the `on_approval` closure tail (`main.rs:853`), when the decision is persistent, push a notice naming the tool **and** the file — core's failure `tracing::warn!` (`runner.rs:680-683`) is silently dropped because `init_tracing` filters to `trace::INTERRUPT_TARGET` only (`main.rs:116-121`), so a notice is the only signal:

```rust
        if decision.is_persistent() {
            let _ = ui_tx.send(Msg::Notice(format!(
                "remembered: {} — persisted to {}",
                names.join(", "),
                learned_path_for_notice.display()
            )));
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --workspace` — Expected: PASS. The only pre-existing test that changed is `main.rs:1876-1883`.

- [ ] **Step 5: Gate and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add crates/heartbit-tui/src/{main.rs,config.rs} crates/heartbit-core/src/agent/runner.rs
git commit -m "$(cat <<'EOF'
fix(tui): approval rules actually persist — drop the dead terminal catch-all

Pressing [a] on an approval did not hold, not even for the rest of the session.
default_permissions() ended with a terminal {"*","*",Ask}; PermissionRuleset::
evaluate is first-match-wins and append_rules extends at the TAIL, so both the
live in-session rule (runner.rs:673) and any loaded rule sat behind a rule that
always matches.

Drop the catch-all and order learned rules FIRST. Behaviour-identical, because
the single production consumer maps Some(Ask) and None to the same arm
(runner.rs:2097) and has_permission_rules() stays true on the 10 remaining
allows — pinned by a new core test so this cannot silently regress into Yolo.

Wire LearnedPermissions through OrchestratorBuilder::learned_permissions (one
call: it forwards to the entry runner and all three sub-agent spawn paths).
Persisting a rule now emits a notice naming the tool and the file, because
core's warn! is filtered out of the TUI's tracing subscriber.

Note: the feature is Normal-mode-only — the TUI defaults to Yolo, where
on_approval short-circuits before the modal.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `/effort` (0.3)

**Files:**
- Modify: `crates/heartbit-tui/src/app.rs` (`EffortLevel`, `App::effort`, `/effort` in `SLASH_COMMANDS` `:181` + `handle_slash` `:1218`, `Modal::EffortPicker`, `Effect::SaveReasoningEffort`)
- Modify: `crates/heartbit-tui/src/config.rs` (`reasoning_effort: Option<String>`)
- Modify: `crates/heartbit-tui/src/main.rs` (`build_engine` param + the gating helper + the effect handler)
- Modify: `crates/heartbit-tui/src/ui.rs` (status line + the picker modal)

**Interfaces:**
- Consumes: `heartbit_core::ReasoningEffort {High, Medium, Low, None}` (`llm/types.rs:151-162`), `OrchestratorBuilder::reasoning_effort` (`orchestrator.rs:2668`), `SubAgentConfig.reasoning_effort` (`orchestrator.rs:2186`), `App::queue_respawn()` (`app.rs:1437`)
- Produces: `EffortLevel {Off, Low, Medium, High}` with `parse(&str) -> Option<Self>` and `label(&self) -> &'static str`; `fn effort_for_provider(level, custom_endpoint, openrouter_key) -> Option<ReasoningEffort>`

- [ ] **Step 1: Write the failing tests**

```rust
    // app.rs — mod tests
    #[test]
    fn effort_level_parse_and_label_roundtrip() {
        for (s, lvl) in [
            ("off", EffortLevel::Off),
            ("low", EffortLevel::Low),
            ("medium", EffortLevel::Medium),
            ("high", EffortLevel::High),
        ] {
            assert_eq!(EffortLevel::parse(s), Some(lvl));
            assert_eq!(lvl.label(), s);
        }
        assert_eq!(EffortLevel::parse("HIGH"), Some(EffortLevel::High));
        assert_eq!(EffortLevel::parse("turbo"), None);
        assert_eq!(EffortLevel::default(), EffortLevel::Off);
    }

    #[test]
    fn slash_effort_sets_level_persists_and_requests_respawn() {
        let mut app = keyed();
        typed(&mut app, "/effort high");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.effort, EffortLevel::High);
        assert!(app.effects.contains(&Effect::SaveReasoningEffort(Some("high".into()))));
        assert!(app.effects.contains(&Effect::RespawnAgent));
    }

    #[test]
    fn slash_effort_off_clears_and_drops_the_config_key() {
        let mut app = keyed();
        typed(&mut app, "/effort high");
        app.update(key(KeyCode::Enter));
        app.effects.clear();
        typed(&mut app, "/effort off");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.effort, EffortLevel::Off);
        assert!(app.effects.contains(&Effect::SaveReasoningEffort(None)));
    }

    #[test]
    fn slash_effort_mid_run_defers_respawn_to_turn_idle() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "/effort low");
        app.update(key(KeyCode::Enter));
        assert!(app.pending_respawn);
        assert!(!app.effects.contains(&Effect::RespawnAgent));
    }

    #[test]
    fn slash_effort_unknown_arg_reports_usage_and_changes_nothing() {
        let mut app = keyed();
        typed(&mut app, "/effort turbo");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.effort, EffortLevel::Off);
        assert!(!app.effects.iter().any(|e| matches!(e, Effect::SaveReasoningEffort(_))));
        assert!(app.history.iter().any(|c| matches!(c, Cell::Notice(n) if n.contains("usage"))));
    }
```

```rust
    // main.rs — mod tests
    #[test]
    fn effort_never_reaches_the_anthropic_fallback_provider() {
        use heartbit_core::ReasoningEffort;
        // OpenRouter or a custom endpoint: the effort is threaded.
        assert_eq!(
            effort_for_provider(EffortLevel::High, None, Some("sk-or-x")),
            Some(ReasoningEffort::High)
        );
        assert_eq!(
            effort_for_provider(EffortLevel::Low, Some("http://127.0.0.1:1/v1"), None),
            Some(ReasoningEffort::Low)
        );
        // Anthropic fallback (no custom endpoint, no OpenRouter key): NEVER.
        // Anthropic's non-streaming ApiContentBlock is Text|ToolUse with
        // #[serde(tag="type")] and no #[serde(other)] (anthropic.rs:778-789), so
        // a returned `thinking` block fails deserialization on the sub-agent path.
        assert_eq!(effort_for_provider(EffortLevel::High, None, None), None);
        // Off always omits the field — never ReasoningEffort::None, which would
        // send reasoning:{"effort":"none"}, a request the TUI never sent before.
        assert_eq!(effort_for_provider(EffortLevel::Off, None, Some("sk-or-x")), None);
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p heartbit-tui effort`
Expected: FAIL to compile — `EffortLevel`, `App::effort`, `Effect::SaveReasoningEffort`, `effort_for_provider` do not exist.

- [ ] **Step 3: Implement**

`app.rs`:

```rust
/// Reasoning-effort level the user selected. `Off` (the default) omits the field
/// entirely, reproducing today's requests bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EffortLevel {
    #[default]
    Off,
    Low,
    Medium,
    High,
}

impl EffortLevel {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// The four levels in picker order.
    pub const ALL: [Self; 4] = [Self::Off, Self::Low, Self::Medium, Self::High];
}
```

Add `pub effort: EffortLevel` to `App` (init `EffortLevel::default()`), `Effect::SaveReasoningEffort(Option<String>)` **plus its `Effect::name()` arm** (`"save_reasoning_effort"` — the match has no wildcard, so omitting it is a compile error), a `("/effort", "set reasoning effort (off|low|medium|high)")` entry in `SLASH_COMMANDS`, and the `"effort"` arm in `handle_slash`:

```rust
            "effort" => {
                if arg.is_empty() {
                    self.open_effort_picker();
                } else if let Some(level) = EffortLevel::parse(&arg) {
                    self.set_effort(level);
                } else {
                    self.history.push(Cell::Notice(
                        "usage: /effort off|low|medium|high".into(),
                    ));
                }
            }
```

`set_effort` mirrors `set_model` (`app.rs:1457`): assign, push `Effect::SaveReasoningEffort`, call `queue_respawn()`, push a notice with the returned suffix. `open_effort_picker`/`Modal::EffortPicker` copy `ModePicker` (`app.rs:383-399` shows the shape) — and remember the `Msg::Paste` and `handle_modal_key` matches over `self.modal` have **no wildcard**, so a new variant is a compile error until handled.

`config.rs`: `pub reasoning_effort: Option<String>` (plain `Option<String>`, per Global Constraints) + `None` in `Default`.

`main.rs`:

```rust
/// Gate the effort by provider. Only OpenRouter and custom OpenAI-compatible
/// endpoints get it; the ANTHROPIC_API_KEY fallback must never receive it (see
/// the test for why). `Off` always omits the field.
fn effort_for_provider(
    level: EffortLevel,
    custom_endpoint: Option<&str>,
    openrouter_key: Option<&str>,
) -> Option<heartbit_core::ReasoningEffort> {
    use heartbit_core::ReasoningEffort;
    if custom_endpoint.is_none() && openrouter_key.is_none() {
        return None;
    }
    match level {
        EffortLevel::Off => None,
        EffortLevel::Low => Some(ReasoningEffort::Low),
        EffortLevel::Medium => Some(ReasoningEffort::Medium),
        EffortLevel::High => Some(ReasoningEffort::High),
    }
}
```

Thread one `Option<ReasoningEffort>`, computed once, into both `OrchestratorBuilder::reasoning_effort` and every `SubAgentConfig.reasoning_effort` in `default_sub_agents`. Add the `build_engine` parameter and `spawn_agent`'s snapshot line — the signature is already 19 positional params, so add at the **end** and double-check the single call site (`main.rs:1220-1240`) for transposition. Handle `Effect::SaveReasoningEffort` in the effect pass beside `Effect::SaveModel`. Show `effort:<label>` in the status line when not `Off`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --workspace` — Expected: PASS.

- [ ] **Step 5: Gate and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add crates/heartbit-tui/src/{app.rs,ui.rs,main.rs,config.rs}
git commit -m "$(cat <<'EOF'
feat(tui): /effort — expose core's reasoning-effort control

ReasoningEffort shipped in heartbit-core with provider mappings and builder
seams, and the TUI never referenced it. /effort off|low|medium|high (bare
/effort opens a picker), persisted in tui.toml, shown in the status line, applied
to the entry agent AND every sub-agent, deferring the respawn to turn-idle when a
turn is in flight.

Provider gating is load-bearing, not cosmetic: the effort must never reach the
ANTHROPIC_API_KEY fallback, because Anthropic's non-streaming ApiContentBlock is
Text|ToolUse with no #[serde(other)], so a returned thinking block would fail
deserialization on the sub-agent path. `off` omits the field entirely rather than
sending ReasoningEffort::None, so the default is bit-for-bit today's request.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Visible input queue (T1.3)

**Files:**
- Modify: `crates/heartbit-tui/src/app.rs` (`App::queued`, `send_or_queue`, drain at the four turn-idle sites, `Up`/Esc handling)
- Modify: `crates/heartbit-tui/src/ui.rs` (render the queue above the composer)
- Test: `app.rs`

**Interfaces:**
- Consumes: `Effect::SendInput` (unchanged), `App::running`
- Produces: `App::queued: VecDeque<String>` and `fn send_or_queue(&mut self, text: String)` — the single choke point for all seven senders (spec D-4)

- [ ] **Step 1: Write the failing tests**

```rust
    #[test]
    fn submit_while_running_queues_instead_of_sending() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "second thing");
        app.update(key(KeyCode::Enter));
        assert_eq!(app.queued.len(), 1);
        assert!(!app.effects.iter().any(|e| matches!(e, Effect::SendInput(_))));
    }

    #[test]
    fn queued_message_drains_at_turn_idle_as_a_user_cell() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "queued one");
        app.update(key(KeyCode::Enter));
        app.effects.clear();
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        assert!(app.queued.is_empty());
        assert!(app.effects.contains(&Effect::SendInput("queued one".into())));
    }

    #[test]
    fn turn_idle_drains_only_one_queued_message() {
        let mut app = keyed();
        app.running = true;
        for t in ["a", "b"] {
            typed(&mut app, t);
            app.update(key(KeyCode::Enter));
        }
        app.effects.clear();
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        assert_eq!(app.queued.len(), 1, "releasing several would re-hide the rest");
    }

    #[test]
    fn tool_calling_llm_done_does_not_drain_the_queue() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "later");
        app.update(key(KeyCode::Enter));
        app.effects.clear();
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: true, ttft_ms: 0 });
        assert_eq!(app.queued.len(), 1, "the turn is not over — a tool is next");
    }

    #[test]
    fn run_failed_and_agent_exit_drop_the_queue() {
        for msg in [Msg::RunFailed("boom".into()), Msg::AgentExited(1)] {
            let mut app = keyed();
            app.running = true;
            typed(&mut app, "stranded");
            app.update(key(KeyCode::Enter));
            app.update(msg);
            assert!(app.queued.is_empty(), "a failed turn must not strand the queue");
        }
    }

    #[test]
    fn queue_is_empty_whenever_the_turn_is_idle() {
        let mut app = keyed();
        assert!(app.queued.is_empty() && !app.running);
        typed(&mut app, "immediate");
        app.update(key(KeyCode::Enter));
        assert!(app.queued.is_empty(), "an idle submit sends, never queues");
        assert!(app.effects.contains(&Effect::SendInput("immediate".into())));
    }

    #[test]
    fn up_arrow_pops_the_newest_queued_message_for_editing() {
        let mut app = keyed();
        app.running = true;
        typed(&mut app, "edit me");
        app.update(key(KeyCode::Enter));
        app.update(key(KeyCode::Up));
        assert!(app.queued.is_empty());
        assert_eq!(app.composer.text(), "edit me");
    }
```

Adapt `Msg::AgentExited`'s real shape to the enum in `msg.rs` before writing that test.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p heartbit-tui queue queued submit_while_running up_arrow_pops`
Expected: FAIL to compile — `App::queued` does not exist.

- [ ] **Step 3: Implement**

```rust
    /// Messages submitted while a turn was in flight, held HERE rather than
    /// pushed into the invisible unbounded input channel so the user can see,
    /// edit and cancel them. Invariant: non-empty ⇒ `running`.
    pub queued: std::collections::VecDeque<String>,
```

```rust
    /// The single choke point for every user-visible send. Seven call sites push
    /// `Effect::SendInput` (submit, AnalyzeReady, LearnReady, /goal, /handoff ×2,
    /// /research); routing them all through here is what keeps the queue honest
    /// instead of leaving six mid-turn bypasses.
    fn send_or_queue(&mut self, text: String) {
        if self.running {
            self.queued.push_back(text);
        } else {
            self.effects.push(Effect::SendInput(text));
        }
    }

    /// Release at most ONE queued message at a turn boundary. Releasing several
    /// would push the rest back into the invisible channel.
    fn drain_one_queued(&mut self) {
        if let Some(text) = self.queued.pop_front() {
            self.history.push(Cell::User(text.clone()));
            self.effects.push(Effect::SendInput(text));
        }
    }
```

Replace the `Effect::SendInput(...)` push at all seven sites with `self.send_or_queue(...)`. Call `drain_one_queued()` at the **four** turn-idle sites (`app.rs:818` `LlmDone{had_tool_calls:false}`, `:935` `RunCompleted`) and clear the queue with a recoverable notice at the two failure sites (`:945` `AgentExited`, `:954` `RunFailed`) and on interrupt. Render in `ui.rs` above the composer with `Constraint::Length(queue_height)` where `queue_height` returns `0` for an empty queue, so an idle frame is identical to today's. Handle `Up` (pop newest into the composer) and Esc (drop with a notice) only while `queued` is non-empty, so idle-Esc keeps its current meaning.

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test -p heartbit-tui` — Expected: PASS. `effect_names_are_stable_snake_case` (`app.rs:2298`) must pass **untouched** — this task adds no `Effect` variant.

- [ ] **Step 5: Gate and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add crates/heartbit-tui/src/{app.rs,ui.rs}
git commit -m "$(cat <<'EOF'
feat(tui): visible input queue for messages typed mid-turn

Submitting while the agent ran pushed straight into the unbounded input channel:
accepted, delivered at the next on_input boundary, and completely invisible — the
user could not tell whether the message was queued or lost. Hold them in
App::queued instead, render them above the composer, Up to edit, Esc to drop.

All SEVEN Effect::SendInput senders route through one send_or_queue choke point
(submit, AnalyzeReady, LearnReady, /goal, /handoff x2, /research), so there is no
mid-turn bypass, and the queue drains at all FOUR turn-idle sites — hooking only
LlmDone would strand messages on a run failure.

At most one message is released per boundary; releasing several would push the
rest back into the invisible channel. No new Msg or Effect variant.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Focus-gated notifications (T1.9)

**Files:**
- Create: `crates/heartbit-tui/src/notify.rs`
- Modify: `crates/heartbit-tui/src/app.rs` (`Effect::Notify`, fire at turn-idle + approval), `main.rs` (module decl + effect handler), `config.rs` (`notify: bool`)
- Test: `notify.rs` (unit), `app.rs` (reducer gating)

**Interfaces:**
- Consumes: `App::focused` (Task 3), the four turn-idle sites (Task 6 touched the same lines — read them first)
- Produces: `notify::{sanitize_field, sequence, Terminal}`; `Effect::Notify { title: String, body: String }`

- [ ] **Step 1: Write the failing tests**

```rust
    // notify.rs — mod tests
    #[test]
    fn sanitize_strips_osc_terminators_and_caps_length() {
        // C0, DEL, C1 (incl. U+009C ST) and ';' must not survive: agent-controlled
        // text (tool names, provider errors) reaches the terminal through here.
        assert_eq!(sanitize_field("a\x07b\x1bc\x9cd;e\x7f"), "abcde");
        assert_eq!(sanitize_field(&"x".repeat(500)).len(), MAX_FIELD);
    }

    #[test]
    fn sequence_is_exactly_one_per_terminal() {
        assert_eq!(sequence(Terminal::Osc777, "T", "B"), "\x1b]777;notify;T;B\x07");
        assert_eq!(sequence(Terminal::Osc9, "T", "B"), "\x1b]9;T: B\x07");
        assert_eq!(sequence(Terminal::Bell, "T", "B"), "\x07");
    }

    #[test]
    fn terminal_from_env_maps_known_ids_and_never_double_notifies() {
        // kitty/WezTerm/Ghostty implement BOTH OSC 777 and OSC 9 — pick 777 only.
        assert_eq!(Terminal::from_ids(Some("xterm-kitty"), None), Terminal::Osc777);
        assert_eq!(Terminal::from_ids(None, Some("WezTerm")), Terminal::Osc777);
        assert_eq!(Terminal::from_ids(None, Some("ghostty")), Terminal::Osc777);
        assert_eq!(Terminal::from_ids(Some("xterm-256color"), None), Terminal::Bell);
    }
```

```rust
    // app.rs — mod tests
    fn unfocused_running() -> App {
        let mut app = keyed();
        app.notify = true;
        app.focused = false;
        app.running = true;
        app
    }

    fn notified(app: &App) -> bool {
        app.effects.iter().any(|e| matches!(e, Effect::Notify { .. }))
    }

    #[test]
    fn notify_on_turn_idle_when_unfocused() {
        let mut app = unfocused_running();
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        assert!(notified(&app));
    }

    #[test]
    fn focused_terminal_suppresses_notify() {
        let mut app = unfocused_running();
        app.focused = true;
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        assert!(!notified(&app));
    }

    #[test]
    fn notify_disabled_by_config_suppresses() {
        let mut app = unfocused_running();
        app.notify = false;
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        assert!(!notified(&app));
    }

    #[test]
    fn tool_turn_does_not_notify() {
        let mut app = unfocused_running();
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: true, ttft_ms: 0 });
        assert!(!notified(&app));
    }

    #[test]
    fn at_most_one_notify_per_turn() {
        let mut app = unfocused_running();
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        app.effects.clear();
        app.update(Msg::RunCompleted);
        assert!(!notified(&app), "RunCompleted must not re-notify after LlmDone");
    }

    #[test]
    fn notify_suppressed_during_splash() {
        let mut app = unfocused_running();
        app.splash = Some(0);
        app.update(Msg::LlmDone { usage: TokenUsage::default(), had_tool_calls: false, ttft_ms: 0 });
        assert!(!notified(&app));
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p heartbit-tui notify sanitize sequence terminal_from_env`
Expected: FAIL to compile — the `notify` module and `Effect::Notify` do not exist.

- [ ] **Step 3: Implement `notify.rs`**

```rust
//! Desktop notifications via terminal escape sequences.
//!
//! Pure formatting (`sequence`) and sanitizing (`sanitize_field`) plus one thin
//! I/O wrapper (`emit`) called ONLY from the main loop's effect pass — never from
//! the reducer and never from the agent thread.
//!
//! Exactly ONE sequence is sent per terminal: kitty, WezTerm and Ghostty
//! implement both OSC 777 and OSC 9, so sending both notifies twice.

/// Max chars kept per field after sanitizing.
pub(crate) const MAX_FIELD: usize = 120;

/// Which notification sequence this terminal understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Terminal {
    /// `OSC 777 ; notify ; title ; body BEL` — kitty, WezTerm, Ghostty.
    Osc777,
    /// `OSC 9 ; text BEL` — terminals that take 9 but not 777.
    Osc9,
    /// No OSC notification support: fall back to the bell.
    Bell,
}

impl Terminal {
    /// Resolve from `TERM` / `TERM_PROGRAM`.
    pub(crate) fn from_ids(term: Option<&str>, term_program: Option<&str>) -> Self {
        let hay = format!(
            "{} {}",
            term.unwrap_or_default().to_ascii_lowercase(),
            term_program.unwrap_or_default().to_ascii_lowercase()
        );
        if ["kitty", "wezterm", "ghostty", "foot"].iter().any(|t| hay.contains(t)) {
            Self::Osc777
        } else if ["iterm", "alacritty"].iter().any(|t| hay.contains(t)) {
            Self::Osc9
        } else {
            Self::Bell
        }
    }

    pub(crate) fn from_env() -> Self {
        Self::from_ids(
            std::env::var("TERM").ok().as_deref(),
            std::env::var("TERM_PROGRAM").ok().as_deref(),
        )
    }
}

/// Strip everything that could terminate or re-open an OSC string, plus the `;`
/// field separator, then cap the length. Removes C0 (`< 0x20`), DEL (`0x7f`) and
/// C1 (`U+0080..=U+009F`, which includes ST).
pub(crate) fn sanitize_field(s: &str) -> String {
    s.chars()
        .filter(|c| {
            let n = *c as u32;
            n >= 0x20 && n != 0x7f && !(0x80..=0x9f).contains(&n) && *c != ';'
        })
        .take(MAX_FIELD)
        .collect()
}

/// The exact bytes to write. Fields must already be sanitized.
pub(crate) fn sequence(term: Terminal, title: &str, body: &str) -> String {
    match term {
        Terminal::Osc777 => format!("\x1b]777;notify;{title};{body}\x07"),
        Terminal::Osc9 => format!("\x1b]9;{title}: {body}\x07"),
        Terminal::Bell => "\x07".to_string(),
    }
}

/// Write the notification. Called ONLY from the main loop's effect pass, after
/// `terminal.draw()` returned. Emits OSC + BEL only: nothing here moves the
/// cursor, alters the screen buffer or writes a newline, so the alt-screen frame
/// is byte-identical.
pub(crate) fn emit(title: &str, body: &str) {
    let seq = sequence(
        Terminal::from_env(),
        &sanitize_field(title),
        &sanitize_field(body),
    );
    use std::io::Write;
    let mut out = std::io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}
```

`app.rs`: add `pub notify: bool` (from config) and `Effect::Notify { title: String, body: String }` **plus its `Effect::name()` arm** (`"notify"`). Fire from the turn-idle path only when `self.notify && !self.focused && self.splash.is_none()`, guarded by the existing `was_running` transition so `LlmDone` → `RunCompleted` notifies once; and from the approval-request path while running. Never for a sub-agent (`Msg::SubAgentDone`/`SubAgentLlmDone`). `main.rs`: `mod notify;` and the effect arm calling `notify::emit`. `config.rs`: `pub notify: bool` (plain `bool`).

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test -p heartbit-tui` — Expected: PASS.

- [ ] **Step 5: Gate and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add crates/heartbit-tui/src/{notify.rs,app.rs,main.rs,config.rs}
git commit -m "$(cat <<'EOF'
feat(tui): focus-gated turn-completion and approval notifications

Emit a terminal notification when a turn ends or an approval is waiting, only
while the window is unfocused (Task 3's focus tracking) and never during the
splash. Exactly ONE sequence per terminal: kitty/WezTerm/Ghostty implement both
OSC 777 and OSC 9, so sending both would notify twice.

Agent-controlled text (tool names, provider errors) is sanitized before it
reaches the terminal: C0, DEL, C1 (incl. ST) and ';' are stripped and fields
capped, so a tool name cannot terminate or re-open the OSC string. Bytes are
written from the main loop's effect pass only — never the reducer, never the
agent thread — and are OSC + BEL only, so the alt-screen frame is unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Syntax highlighting (T1.7a)

**Files:**
- Modify: root `Cargo.toml` (`[workspace.dependencies]`), `crates/heartbit-tui/Cargo.toml`
- Modify: `crates/heartbit-tui/src/markdown.rs` (highlight fenced blocks), `app.rs` (`App::md` cache + invalidate on `Msg::Resize`), `ui.rs:16` (`begin_frame()`), `config.rs` (`syntax_theme: Option<String>`)
- Test: `markdown.rs`

**Interfaces:**
- Consumes: `SyntaxSet::load_defaults_newlines()`, `SyntaxSet::find_syntax_by_token`, `ThemeSet::load_defaults()` (all infallible embedded dumps — no I/O)
- Produces: `MarkdownCache` with `begin_frame()`, exactly one caller (the first statement of `ui::transcript_lines`)

- [ ] **Step 1: Write the failing tests** (in `markdown.rs`'s test module)

```rust
    #[test]
    fn fenced_rust_block_is_syntax_highlighted() {
        let lines = render("```rust\nfn main() {}\n```");
        let colours: std::collections::HashSet<_> = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .filter_map(|s| s.style.fg)
            .collect();
        assert!(colours.len() > 1, "a highlighted block uses more than one colour");
    }

    #[test]
    fn unknown_language_falls_back_to_flat_code_colour() {
        let lines = render("```notalanguage\nfn main() {}\n```");
        let colours: std::collections::HashSet<_> = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .filter_map(|s| s.style.fg)
            .collect();
        assert_eq!(colours.len(), 1, "unknown languages keep the flat code colour");
    }

    #[test]
    fn highlighting_preserves_code_characters_and_line_count() {
        // THE invariant: highlighting must not change one character or one line,
        // or every existing markdown assertion silently becomes wrong.
        let src = "```rust\nfn main() {\n    let x = 1;\n}\n```";
        let hl = render(src);
        let flat = render(&src.replace("```rust", "```"));
        assert_eq!(all_text(&hl), all_text(&flat));
        assert_eq!(hl.len(), flat.len());
    }

    #[test]
    fn fence_info_string_takes_the_first_token() {
        // "```rust,ignore" and "```rust title=x" both mean rust.
        for info in ["rust", "rust,ignore", "rust title=x"] {
            assert_eq!(language_token(info), "rust");
        }
    }

    #[test]
    fn markdown_cache_hit_matches_the_uncached_render() {
        let cache = MarkdownCache::default();
        let src = "```rust\nfn main() {}\n```";
        cache.begin_frame();
        let first = cache.render(src);
        let second = cache.render(src); // served from the cache
        assert_eq!(first, second);
        assert_eq!(first, render(src));
    }

    #[test]
    fn markdown_cache_sweeps_entries_not_reused_next_frame() {
        let cache = MarkdownCache::default();
        cache.begin_frame();
        let _ = cache.render("```rust\nfn a() {}\n```");
        assert_eq!(cache.len(), 1);
        cache.begin_frame(); // new frame, entry not touched
        cache.begin_frame(); // …and swept
        assert_eq!(cache.len(), 0, "unused entries must not grow forever");
    }
```

Read `markdown.rs`'s existing test helpers first and reuse them (`render`, `all_text` or equivalents); do not invent new ones.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p heartbit-tui markdown`
Expected: FAIL to compile — `MarkdownCache`, `language_token` do not exist; the highlighting tests fail on a single colour.

- [ ] **Step 3: Add the dependency and implement**

Root `Cargo.toml` `[workspace.dependencies]`:

```toml
syntect = { version = "5", default-features = false, features = ["default-fancy"] }
```

`crates/heartbit-tui/Cargo.toml`: `syntect = { workspace = true }`.

In `markdown.rs`: a `OnceLock<SyntaxSet>` + `OnceLock<Theme>`; `language_token(info: &str) -> &str` (first whitespace/comma-delimited token); route a fenced block through syntect when `find_syntax_by_token` resolves, mapping syntect styles to ratatui `Style` — otherwise keep today's flat colour. Strip exactly one trailing `\n` from the buffered block and skip ranges whose text is only `"\n"`, so the emitted line count is unchanged.

`MarkdownCache`: interior-mutable (`RefCell<HashMap<String, Vec<Line<'static>>>>` + a `RefCell<HashSet<String>>` of keys used this frame), keyed by the **source text** — never a bare hash, because a collision would render another cell's content with no test able to catch it. `begin_frame()` sweeps keys unused in the previous frame. Add `pub(crate) md: MarkdownCache` to `App`, call `app.md.begin_frame()` as the **first statement** of `ui::transcript_lines` (`ui.rs:16`), and clear the cache from the `Msg::Resize` arm (`app.rs:733`, a no-op today) so a width change cannot serve stale wrapping. The reducer must never *read* the cache — grep `App::update` for `.md` and find nothing.

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test -p heartbit-tui` — Expected: PASS, with every pre-existing `markdown::tests` case unchanged.

- [ ] **Step 5: Gate and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add Cargo.toml crates/heartbit-tui/Cargo.toml crates/heartbit-tui/src/{markdown.rs,app.rs,ui.rs,config.rs}
git commit -m "$(cat <<'EOF'
feat(tui): syntax-highlight fenced code blocks (syntect)

Code rendered in a single colour. Route fenced blocks through syntect (embedded
dumps, no I/O), falling back to the flat colour for unknown languages.

Caching is mandatory, not an optimisation: terminal.draw runs at the top of the
loop, so transcript_lines re-renders every agent cell on every keystroke and
every 120ms tick. Styled lines are memoized in an interior-mutable cache keyed
by the SOURCE TEXT (never a bare hash — a collision would render another cell's
content), swept per frame by a begin_frame() with exactly one caller, and cleared
on Msg::Resize (previously a no-op) so a width change cannot serve stale wrapping.

Invariant: highlighting changes no character and no line count, which is what
keeps every existing markdown assertion valid.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Word-level diffs and `/diff` (T1.7b + T1.7c)

Last, and the largest (L). It reshapes `DiffLine`, so it touches `diff.rs`, `cells.rs` and five existing tests.

**Files:**
- Modify: `crates/heartbit-tui/src/diff.rs:12-16` (`DiffLine.emph`, pairing, `parse_unified`), `cells.rs:95-107` (render emphasis, shared `render_diff_lines`)
- Create: `crates/heartbit-tui/src/gitdiff.rs` (pure parse for `/diff`)
- Modify: `crates/heartbit-tui/src/app.rs` (`/diff` command + `Effect::GitDiff` + `Msg::GitDiffReady`), `main.rs` (the effect handler running git via `spawn_blocking`)
- Test: `diff.rs`, `gitdiff.rs`, `app.rs`

**Interfaces:**
- Consumes: `diff_preview(tool_name, input, max)` (`cells.rs:94`) — signature **unchanged**, so `ui.rs:402` inherits emphasis with no edit
- Produces: `DiffLine { kind, text, emph: Vec<std::ops::Range<usize>> }`; `fn word_emphasis(del: &str, add: &str) -> (Vec<Range<usize>>, Vec<Range<usize>>)`

- [ ] **Step 1: Write the failing tests**

```rust
    // diff.rs — mod tests
    #[test]
    fn single_token_change_emphasises_only_that_token() {
        let (del, add) = word_emphasis("let x = 1;", "let x = 2;");
        assert_eq!(del.len(), 1);
        assert_eq!(add.len(), 1);
        assert_eq!(&"let x = 1;"[del[0].clone()], "1");
        assert_eq!(&"let x = 2;"[add[0].clone()], "2");
    }

    #[test]
    fn identical_lines_yield_no_emphasis() {
        let (del, add) = word_emphasis("same", "same");
        assert!(del.is_empty() && add.is_empty());
    }

    #[test]
    fn two_disjoint_changes_are_both_emphasised() {
        let (del, _) = word_emphasis("a b c", "x b y");
        assert_eq!(del.len(), 2, "both ends changed: {del:?}");
        // Ranges are sorted, non-overlapping and on char boundaries.
        assert!(del.windows(2).all(|w| w[0].end <= w[1].start));
    }

    #[test]
    fn emphasis_ranges_are_char_boundaries_for_multibyte_text() {
        let (del, add) = word_emphasis("héllo wörld", "héllo tërre");
        for (s, rs) in [("héllo wörld", &del), ("héllo tërre", &add)] {
            for r in rs.iter() {
                assert!(s.is_char_boundary(r.start) && s.is_char_boundary(r.end));
            }
        }
    }

    #[test]
    fn del_line_starting_with_a_comment_dash_is_not_dropped_as_a_file_header() {
        // "--- a/x" is a header, but a deleted line whose content starts with
        // "--" is real content and must survive.
        let d = parse_unified("@@ -1,1 +1,1 @@\n---- a comment\n+ok\n");
        assert!(d.iter().any(|l| l.text.contains("a comment")));
    }

    #[test]
    fn hunkless_patch_text_takes_the_legacy_path_unchanged() {
        // Core's patch parser requires "@@ " (patch.rs:425), so hunkless text
        // must behave exactly as before this task.
        let d = parse_unified("+added\n-removed\n");
        assert_eq!(d.len(), 2);
        assert!(d.iter().all(|l| l.emph.is_empty()));
    }
```

```rust
    // app.rs — mod tests
    #[test]
    fn slash_diff_requests_the_working_tree_diff() {
        let mut app = keyed();
        typed(&mut app, "/diff");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::GitDiff));
        assert!(!app.effects.iter().any(|e| matches!(e, Effect::SendInput(_))),
            "/diff is local — it must not consume an LLM turn");
    }

    #[test]
    fn git_diff_ready_renders_a_diff_cell_and_empty_is_a_notice() {
        let mut app = keyed();
        app.update(Msg::GitDiffReady(Ok("@@ -1,1 +1,1 @@\n-a\n+b\n".into())));
        assert!(app.history.iter().any(|c| matches!(c, Cell::Diff { .. })));

        let mut app2 = keyed();
        app2.update(Msg::GitDiffReady(Ok(String::new())));
        assert!(app2.history.iter().any(|c| matches!(c, Cell::Notice(n) if n.contains("no changes"))));

        let mut app3 = keyed();
        app3.update(Msg::GitDiffReady(Err("not a git repository".into())));
        assert!(app3.history.iter().any(|c| matches!(c, Cell::Notice(n) if n.contains("git"))));
    }
```

Match `Cell::Diff`'s real shape in `cells.rs` before writing that assertion.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p heartbit-tui diff`
Expected: FAIL to compile — `word_emphasis`, `DiffLine.emph`, `Effect::GitDiff`, `Msg::GitDiffReady` do not exist.

- [ ] **Step 3: Implement**

Add `pub emph: Vec<std::ops::Range<usize>>` to `DiffLine` with the invariant documented (sorted, non-overlapping, non-empty, char-boundary bounds). Implement `word_emphasis` by hand — **do not add the `similar` crate**: it is absent from `Cargo.lock` and the project's standing precedent is hand-rolled (SSE parser maison; `levenshtein()` duplicated by hand). Tokenize both sides on word boundaries, trim the common prefix and suffix at token granularity, and emit byte ranges for the middle. In `parse_unified`, pair adjacent Del/Add runs of equal length and fill both sides' `emph`.

In `cells.rs`, add a shared `render_diff_lines` that splits a line into spans at the `emph` ranges (brighter fg/bg for changed spans) and reuses the existing `"  … ({} more diff lines)"` cap wording. A `DiffLine` with `emph.is_empty()` must render to exactly one span with today's content and style — assert that in a test.

`/diff`: a `("/diff", "show the working-tree diff")` entry, the `"diff"` arm pushing `Effect::GitDiff`, `Msg::GitDiffReady(Result<String, String>)`, and the main-loop handler running `git diff HEAD` plus untracked files under `tokio::task::spawn_blocking` (git is I/O — never in the reducer). A non-git cwd yields `Err` → a notice, not an error. Reuse `gitdiff::parse` (which may simply delegate to `diff::parse_unified`) and cap the rendered length.

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --workspace` — Expected: PASS, and specifically `patch_parses_unified_diff_by_leading_char` (`diff.rs:129-141`) and `long_diff_is_capped_with_more_note` (`cells.rs:696`) pass **unchanged**.

- [ ] **Step 5: Gate and commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add crates/heartbit-tui/src/{diff.rs,cells.rs,gitdiff.rs,app.rs,main.rs}
git commit -m "$(cat <<'EOF'
feat(tui): word-level diff emphasis and /diff

Diffs were line-level only, so a one-character change looked like a whole
rewritten line. Pair adjacent Del/Add runs and emphasise just the changed spans
(hand-rolled, ~120 pure lines — no new dependency, matching the project's
existing hand-rolled precedents). diff_preview keeps its exact signature, so the
approval modal inherits emphasis with zero edits.

/diff renders the cumulative working-tree diff through the same DiffLine
renderer. Git is I/O, so it runs in the main loop under spawn_blocking, never in
the reducer; a non-git cwd is a notice, not an error.

A DiffLine with no emphasis renders exactly as before, and hunkless patch text
takes the legacy path bit-for-bit, so the patch tool cannot regress.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Step 1: Full gate**

```bash
cargo fmt --all -- --check && \
cargo clippy --workspace --all-targets -- -D warnings && \
cargo test --workspace
```
Expected: exit 0, 0 failed.

- [ ] **Step 2: Confirm the blast radius**

```bash
git diff --stat 37bb8a5..HEAD -- crates/heartbit-core
```
Expected: only `tool/builtins/{format,mod,write,edit,patch}.rs` (Task 1) and one new test in `agent/runner.rs` (Task 4).

- [ ] **Step 3: Run the manual acceptance signal** — the script in the spec's §7, steps 1–6, in Kitty/Ghostty/WezTerm/foot (not tmux). Per framework §7.5, this half must not be claimed as proven by `cargo test`.

- [ ] **Step 4: Record the outcome** in `tasks/` and note anything the manual pass revealed that the tests did not.
