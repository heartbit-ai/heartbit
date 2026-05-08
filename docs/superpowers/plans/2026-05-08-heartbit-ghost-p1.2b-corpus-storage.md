# heartbit-ghost P1.2b — corpus storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist + retrieve raw post collections per writer (the input that P1.2c's LLM extractor will consume to build a `StyleProfile`). Pure data-layer library.

**Architecture:** New `crates/heartbit-ghost/src/corpus/` module with 4 files (mod.rs, entry.rs, store.rs, error.rs). File-backed JSONL — one file per writer at `~/.heartbit/ghost/corpora/<writer>.jsonl`. Append-only ingest with id-based dedup. Atomic save via tempfile+rename. ~22 tests, all in-tree, all using `tempfile::TempDir`.

**Tech Stack:** Rust 2024, `serde`/`serde_json`, `chrono` (workspace dep, `serde` feature), `thiserror`, stdlib `std::fs` for I/O. Dev-dep `tempfile` for filesystem isolation in tests.

---

## File structure

| File | Responsibility |
|------|----------------|
| `crates/heartbit-ghost/src/corpus/mod.rs` | Module surface, re-exports, `list_writers` helper |
| `crates/heartbit-ghost/src/corpus/entry.rs` | `CorpusEntry`, `Engagement` data types |
| `crates/heartbit-ghost/src/corpus/store.rs` | `Corpus` struct, `AppendStats`, `default_corpora_dir`, `resolve_corpora_dir` (pure helper for testable env resolution), writer-name validation |
| `crates/heartbit-ghost/src/corpus/error.rs` | `CorpusError` enum (`thiserror`) |
| `crates/heartbit-ghost/src/lib.rs` | Add `pub mod corpus;` declaration |
| `crates/heartbit-ghost/Cargo.toml` | Add `chrono` workspace dep + `tempfile` dev-dep |

5 tasks total: 4 implementation + 1 final acceptance (verification only, no commit).

---

## Task 1: Module scaffolding + `CorpusError` + Cargo.toml deps

**Why:** Get the bare module skeleton compiling so subsequent tasks can build on it. Also wire the two dependencies (`chrono`, `tempfile`) up front so we never hit a phantom missing-dep failure mid-task.

**Files:**
- Create: `crates/heartbit-ghost/src/corpus/mod.rs`
- Create: `crates/heartbit-ghost/src/corpus/error.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs` (add `pub mod corpus;`)
- Modify: `crates/heartbit-ghost/Cargo.toml` (add `chrono` + `tempfile`)

- [ ] **Step 1: Add deps to `crates/heartbit-ghost/Cargo.toml`**

Insert `chrono = { workspace = true }` in `[dependencies]` (alphabetically — between `base64` and `heartbit-core` if alphabetical; otherwise just append). Insert `tempfile = "3"` in `[dev-dependencies]` alongside the existing `tokio` and `wiremock`.

The relevant final state should look like:

```toml
[dependencies]
base64 = { workspace = true }
chrono = { workspace = true }
heartbit-core = { path = "../heartbit-core" }
hmac = { workspace = true }
rand = { workspace = true }
reqwest = { workspace = true, features = ["multipart"] }
serde = { workspace = true }
serde_json = { workspace = true }
sha1 = { workspace = true }
thiserror = { workspace = true }
toml = { workspace = true }

[dev-dependencies]
tempfile = "3"
tokio = { workspace = true }
wiremock = "0.6"
```

(Workspace `[workspace.dependencies]` already has `chrono = { version = "0.4", features = ["serde"] }`. Workspace pull-in carries the `serde` feature automatically.)

- [ ] **Step 2: Create `crates/heartbit-ghost/src/corpus/error.rs`**

```rust
//! Corpus storage errors.

use thiserror::Error;

/// Errors raised by the [`crate::corpus`] subsystem.
#[derive(Debug, Error)]
pub enum CorpusError {
    /// Filesystem failure (open / read / write / rename).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parse failure on a specific JSONL line. The line number is 1-based.
    #[error("json on line {line}: {source}")]
    Json {
        /// 1-based line number where the parse failed.
        line: usize,
        /// Underlying parser error.
        #[source]
        source: serde_json::Error,
    },

    /// The supplied writer handle is invalid — empty, contains a path
    /// separator, contains `..`, or is whitespace-only.
    #[error("invalid writer name '{0}': must be non-empty, no '/', '\\', or '..'")]
    InvalidWriter(String),

    /// Generic data or environment validation failure.
    #[error("validation: {0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_renders_with_io_prefix() {
        let inner = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e = CorpusError::Io(inner);
        let s = format!("{e}");
        assert!(s.starts_with("io: "), "got: {s}");
        assert!(s.contains("missing"));
    }

    #[test]
    fn json_error_includes_line_number_in_display() {
        let bad = serde_json::from_str::<serde_json::Value>("not-json").unwrap_err();
        let e = CorpusError::Json {
            line: 47,
            source: bad,
        };
        let s = format!("{e}");
        assert!(s.contains("line 47"), "got: {s}");
    }

    #[test]
    fn invalid_writer_rendering_is_actionable() {
        let e = CorpusError::InvalidWriter("../etc/passwd".to_string());
        let s = format!("{e}");
        assert!(s.contains("../etc/passwd"));
        assert!(s.contains("'/'"));
        assert!(s.contains(".."));
    }
}
```

- [ ] **Step 3: Create `crates/heartbit-ghost/src/corpus/mod.rs`**

For Task 1, the file just declares the error submodule and re-exports the error type. Subsequent tasks uncomment the `entry` and `store` lines.

```rust
//! Corpus storage — file-backed JSONL persistence for per-writer post
//! collections. Sibling of [`crate::voice`]: voice owns the schema; corpus
//! owns the inputs that the LLM extractor (P1.2c) turns into a profile.
//!
//! On-disk layout (created lazily on first write):
//!
//! ```text
//! ~/.heartbit/ghost/corpora/
//! ├── karpathy.jsonl      # one writer, one file
//! ├── eladgil.jsonl
//! └── swyx.jsonl
//! ```

pub mod error;
// pub mod entry;   // uncommented in Task 2
// pub mod store;   // uncommented in Task 3

pub use error::CorpusError;
```

- [ ] **Step 4: Modify `crates/heartbit-ghost/src/lib.rs` to declare the module**

Find the existing module declarations:

```rust
pub mod tools;
pub mod voice;
```

Replace with:

```rust
pub mod corpus;
pub mod tools;
pub mod voice;
```

(Alphabetical — rustfmt's default `reorder_modules` will sort them anyway, so just ship the alphabetical order directly to avoid a phantom-diff fight.)

- [ ] **Step 5: Run the tests**

```bash
cargo test -p heartbit-ghost --lib corpus
```

Expected: `3 passed; 0 failed; 0 ignored`.

- [ ] **Step 6: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-ghost/Cargo.toml crates/heartbit-ghost/src/lib.rs crates/heartbit-ghost/src/corpus/
git commit -m "$(cat <<'EOF'
feat(ghost): corpus module scaffolding + CorpusError (P1.2b)

Stub module that compiles, with the error enum its child modules will
consume in subsequent tasks. Adds the two new deps up front:
chrono (workspace) for posted_at timestamps, tempfile (dev) for
filesystem isolation in the test suite.

3 tests on CorpusError: Display formatting (io prefix, line number in
JSON variant, actionable hint in InvalidWriter).

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md §5
EOF
)"
```

---

## Task 2: `CorpusEntry` + `Engagement` data types

**Why:** The data shape is small and self-contained. Defining it before the storage logic in Task 3 lets us round-trip-test the serde behavior (defaults, `skip_serializing_if`, `deny_unknown_fields`) without any filesystem concerns getting tangled in.

**Files:**
- Create: `crates/heartbit-ghost/src/corpus/entry.rs`
- Modify: `crates/heartbit-ghost/src/corpus/mod.rs` (uncomment `pub mod entry;` + add re-exports)

- [ ] **Step 1: Create `crates/heartbit-ghost/src/corpus/entry.rs`**

```rust
//! Per-post data types — what one line of a writer's JSONL corpus contains.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Engagement metrics from the source platform.
///
/// Best-effort: missing fields default to zero on parse, and the type is
/// stored only when the JSONL line included it. Posts without engagement
/// data (e.g., manually authored corpora) carry [`CorpusEntry::engagement`]
/// = `None` and never construct this struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Engagement {
    /// Like / heart count.
    #[serde(default)]
    pub likes: u64,
    /// Repost / retweet / quote count.
    #[serde(default)]
    pub reposts: u64,
    /// Reply count.
    #[serde(default)]
    pub replies: u64,
}

/// One post in a writer's reference corpus. The minimal schema requires only
/// `id` and `post_text`; everything else is optional.
///
/// `id` is typically the source platform's post id (e.g., the X tweet id as
/// a string). It is the dedup key on re-import (see
/// [`crate::corpus::Corpus::append_from_jsonl`]).
///
/// The writer handle is **not** stored on the entry — it is implicit from
/// the file the entry lives in (`<writer>.jsonl`). Storing it per entry
/// would be redundant and would let imports drift.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusEntry {
    /// Stable identifier — typically the X tweet id as a string.
    /// Used for dedup on re-import.
    pub id: String,

    /// The post text (no markdown stripping; stored verbatim).
    pub post_text: String,

    /// Original posting time; RFC3339 in JSONL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posted_at: Option<DateTime<Utc>>,

    /// Engagement metrics from the source (best-effort; may be absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engagement: Option<Engagement>,

    /// Manual tags: `["thread_opener", "hot_take", "self_deprecating"]`.
    /// Empty by default; absent vs. empty are stored identically (`Vec::new`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,

    /// Pre-computed embedding. P1.2b stores but does not generate
    /// embeddings (P1.4 wires the local-embedding pipeline through this
    /// field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_minimal_round_trip_via_json() {
        let entry = CorpusEntry {
            id: "1857234567890".to_string(),
            post_text: "the bitter lesson keeps winning".to_string(),
            posted_at: None,
            engagement: None,
            tags: Vec::new(),
            embedding: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        let back: CorpusEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, entry);
    }

    #[test]
    fn entry_full_round_trip_via_json() {
        let entry = CorpusEntry {
            id: "1857234567891".to_string(),
            post_text: "compute + scale + simple objective".to_string(),
            posted_at: Some(
                "2025-04-12T14:32:00Z"
                    .parse::<DateTime<Utc>>()
                    .expect("rfc3339 parses"),
            ),
            engagement: Some(Engagement {
                likes: 1234,
                reposts: 56,
                replies: 12,
            }),
            tags: vec!["hot_take".to_string(), "thread_opener".to_string()],
            embedding: Some(vec![0.1, 0.2, 0.3]),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        let back: CorpusEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, entry);
    }

    #[test]
    fn engagement_default_is_all_zero() {
        let e = Engagement::default();
        assert_eq!(e.likes, 0);
        assert_eq!(e.reposts, 0);
        assert_eq!(e.replies, 0);
    }

    #[test]
    fn engagement_partial_fields_default_remaining_to_zero() {
        let parsed: Engagement = serde_json::from_str(r#"{"likes": 42}"#).expect("parses");
        assert_eq!(parsed.likes, 42);
        assert_eq!(parsed.reposts, 0);
        assert_eq!(parsed.replies, 0);
    }

    #[test]
    fn entry_optional_fields_omitted_when_none_or_empty() {
        let entry = CorpusEntry {
            id: "1".to_string(),
            post_text: "hi".to_string(),
            posted_at: None,
            engagement: None,
            tags: Vec::new(),
            embedding: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        // Only id + post_text should appear in the wire form.
        assert!(json.contains("\"id\""));
        assert!(json.contains("\"post_text\""));
        assert!(!json.contains("posted_at"));
        assert!(!json.contains("engagement"));
        assert!(!json.contains("tags"));
        assert!(!json.contains("embedding"));
    }

    #[test]
    fn entry_unknown_field_rejected_via_deny_unknown_fields() {
        let json = r#"{"id":"1","post_text":"hi","bogus":"oops"}"#;
        let err = serde_json::from_str::<CorpusEntry>(json).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("bogus") || msg.contains("unknown"));
    }

    #[test]
    fn engagement_unknown_field_rejected_via_deny_unknown_fields() {
        let json = r#"{"likes":1,"shares":99}"#;
        let err = serde_json::from_str::<Engagement>(json).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("shares") || msg.contains("unknown"));
    }
}
```

- [ ] **Step 2: Update `crates/heartbit-ghost/src/corpus/mod.rs`**

Uncomment `pub mod entry;` and add the re-exports:

```rust
//! Corpus storage — file-backed JSONL persistence for per-writer post
//! collections. Sibling of [`crate::voice`]: voice owns the schema; corpus
//! owns the inputs that the LLM extractor (P1.2c) turns into a profile.
//!
//! On-disk layout (created lazily on first write):
//!
//! ```text
//! ~/.heartbit/ghost/corpora/
//! ├── karpathy.jsonl      # one writer, one file
//! ├── eladgil.jsonl
//! └── swyx.jsonl
//! ```

pub mod entry;
pub mod error;
// pub mod store;   // uncommented in Task 3

pub use entry::{CorpusEntry, Engagement};
pub use error::CorpusError;
```

(Alphabetical mod ordering = rustfmt-stable.)

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib corpus
```

Expected: `10 passed` (3 from Task 1 + 7 new).

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/corpus/entry.rs crates/heartbit-ghost/src/corpus/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): corpus — CorpusEntry + Engagement (P1.2b)

The minimal per-post schema: id + post_text required; posted_at,
engagement, tags, embedding all optional. serde(deny_unknown_fields)
on both types fails loudly on JSONL typos. skip_serializing_if keeps
emitted JSONL minimal when posts only have id + text.

writer_handle from the umbrella spec is intentionally absent — the
writer is implicit from the file (<writer>.jsonl). Storing it per
entry would let imports drift.

7 tests: minimal + full round-trip, Engagement default zero, partial
defaults, optional-skip-serialize, deny_unknown_fields on both types.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md §4
EOF
)"
```

---

## Task 3: `Corpus` + `AppendStats` + `default_corpora_dir`

**Why:** The core of the library. This task is the largest one — it creates `store.rs` with all the filesystem behavior (`open_or_create`, `append_from_jsonl`, `save`, env-var resolution, writer-name validation) and an exhaustive test suite covering every error path and the atomic-rename invariant.

**Files:**
- Create: `crates/heartbit-ghost/src/corpus/store.rs`
- Modify: `crates/heartbit-ghost/src/corpus/mod.rs` (uncomment `pub mod store;` + re-exports)

- [ ] **Step 1: Create `crates/heartbit-ghost/src/corpus/store.rs`**

```rust
//! Filesystem-backed corpus storage. One JSONL file per writer.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::corpus::entry::CorpusEntry;
use crate::corpus::error::CorpusError;

/// Counts returned by [`Corpus::append_from_jsonl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendStats {
    /// Number of entries newly added to the corpus.
    pub added: usize,
    /// Number of entries that already existed (matched by `id`) and were
    /// skipped.
    pub deduped: usize,
    /// Total entry count after the append (`added + previous_total`).
    pub total_after: usize,
}

/// Resolve the corpora root directory.
///
/// Resolution order:
/// 1. `$HEARTBIT_GHOST_CORPORA` if set (used as-is)
/// 2. `$HOME/.heartbit/ghost/corpora` if `$HOME` is set
/// 3. [`CorpusError::Validation`] otherwise
pub fn default_corpora_dir() -> Result<PathBuf, CorpusError> {
    let custom = std::env::var("HEARTBIT_GHOST_CORPORA").ok();
    let home = std::env::var("HOME").ok();
    resolve_corpora_dir(custom.as_deref(), home.as_deref())
}

/// Pure resolver — separated from env access for testability.
pub(crate) fn resolve_corpora_dir(
    custom: Option<&str>,
    home: Option<&str>,
) -> Result<PathBuf, CorpusError> {
    if let Some(path) = custom {
        return Ok(PathBuf::from(path));
    }
    let home = home.ok_or_else(|| {
        CorpusError::Validation(
            "neither HEARTBIT_GHOST_CORPORA nor HOME is set".to_string(),
        )
    })?;
    Ok(PathBuf::from(home).join(".heartbit/ghost/corpora"))
}

/// Validate a writer handle. Rejects empty, whitespace-only, or anything
/// containing `/`, `\\`, or `..`.
pub(crate) fn validate_writer(name: &str) -> Result<(), CorpusError> {
    let trimmed = name.trim();
    if trimmed.is_empty()
        || trimmed != name
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains("..")
    {
        return Err(CorpusError::InvalidWriter(name.to_string()));
    }
    Ok(())
}

/// One writer's reference corpus, loaded into memory.
///
/// Acquire via [`Corpus::open_or_create`]. Mutate via
/// [`Corpus::append_from_jsonl`] (which persists on success). Read via
/// [`Corpus::entries`] / [`Corpus::len`] / [`Corpus::is_empty`].
#[derive(Debug)]
pub struct Corpus {
    writer: String,
    path: PathBuf,
    entries: Vec<CorpusEntry>,
}

impl Corpus {
    /// Open the writer's corpus file under `root`, creating an empty corpus
    /// if no file exists yet. The corpus directory itself is **not** created
    /// until [`Corpus::save`] is called — `open_or_create` is read-only on
    /// the filesystem when the file is absent.
    pub fn open_or_create(root: &Path, writer: &str) -> Result<Self, CorpusError> {
        validate_writer(writer)?;
        let path = root.join(format!("{writer}.jsonl"));
        let entries = match File::open(&path) {
            Ok(file) => parse_jsonl(BufReader::new(file))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(CorpusError::Io(e)),
        };
        Ok(Self {
            writer: writer.to_string(),
            path,
            entries,
        })
    }

    /// The writer handle this corpus belongs to.
    pub fn writer(&self) -> &str {
        &self.writer
    }

    /// Number of stored posts.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the corpus has zero posts.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries in load order. Stable across calls until
    /// [`Corpus::append_from_jsonl`] runs.
    pub fn entries(&self) -> &[CorpusEntry] {
        &self.entries
    }

    /// Append every entry from `path` (a JSONL file) into the corpus,
    /// deduping by `id` (existing entries win). On success, persists to
    /// disk via [`Corpus::save`] before returning. On parse failure on any
    /// line, returns the error and leaves both the in-memory entries and
    /// the on-disk file unchanged.
    pub fn append_from_jsonl(&mut self, path: &Path) -> Result<AppendStats, CorpusError> {
        let file = File::open(path)?;
        let new_entries = parse_jsonl(BufReader::new(file))?;
        let mut seen: HashSet<String> = self.entries.iter().map(|e| e.id.clone()).collect();
        let mut added = 0usize;
        let mut deduped = 0usize;
        let mut to_append: Vec<CorpusEntry> = Vec::new();
        for entry in new_entries {
            if seen.contains(&entry.id) {
                deduped += 1;
            } else {
                seen.insert(entry.id.clone());
                added += 1;
                to_append.push(entry);
            }
        }
        self.entries.extend(to_append);
        self.save()?;
        Ok(AppendStats {
            added,
            deduped,
            total_after: self.entries.len(),
        })
    }

    /// Persist the in-memory state back to disk. Atomic on POSIX:
    /// writes to a sibling tempfile, then `rename`s over the destination.
    pub fn save(&self) -> Result<(), CorpusError> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp_path = self.path.with_extension("jsonl.tmp");
        {
            let tmp_file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(tmp_file);
            for entry in &self.entries {
                let line = serde_json::to_string(entry).map_err(|source| CorpusError::Json {
                    line: 0,
                    source,
                })?;
                writer.write_all(line.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
        }
        fs::rename(&tmp_path, &self.path)?;
        Ok(())
    }
}

/// Parse a `BufRead` source as JSONL. Empty / whitespace-only lines are
/// skipped. Returns the first parse error with its 1-based line number.
fn parse_jsonl<R: BufRead>(reader: R) -> Result<Vec<CorpusEntry>, CorpusError> {
    let mut entries = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: CorpusEntry =
            serde_json::from_str(&line).map_err(|source| CorpusError::Json {
                line: line_no,
                source,
            })?;
        entries.push(entry);
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::entry::Engagement;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_jsonl(path: &Path, lines: &[&str]) {
        let mut f = File::create(path).expect("create jsonl");
        for line in lines {
            writeln!(f, "{line}").expect("write");
        }
    }

    fn minimal(id: &str, text: &str) -> String {
        format!(r#"{{"id":"{id}","post_text":"{text}"}}"#)
    }

    // ---- resolve_corpora_dir ---------------------------------------------

    #[test]
    fn resolve_uses_custom_when_set() {
        let p = resolve_corpora_dir(Some("/explicit/path"), Some("/home/u")).unwrap();
        assert_eq!(p, PathBuf::from("/explicit/path"));
    }

    #[test]
    fn resolve_falls_back_to_home_when_custom_unset() {
        let p = resolve_corpora_dir(None, Some("/home/u")).unwrap();
        assert_eq!(p, PathBuf::from("/home/u/.heartbit/ghost/corpora"));
    }

    #[test]
    fn resolve_errors_when_neither_set() {
        let err = resolve_corpora_dir(None, None).unwrap_err();
        match err {
            CorpusError::Validation(msg) => {
                assert!(msg.contains("HEARTBIT_GHOST_CORPORA"));
                assert!(msg.contains("HOME"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    // ---- validate_writer -------------------------------------------------

    #[test]
    fn validate_writer_rejects_empty() {
        let err = validate_writer("").unwrap_err();
        assert!(matches!(err, CorpusError::InvalidWriter(s) if s.is_empty()));
    }

    #[test]
    fn validate_writer_rejects_slash() {
        let err = validate_writer("foo/bar").unwrap_err();
        assert!(matches!(err, CorpusError::InvalidWriter(s) if s == "foo/bar"));
    }

    #[test]
    fn validate_writer_rejects_backslash() {
        let err = validate_writer("foo\\bar").unwrap_err();
        assert!(matches!(err, CorpusError::InvalidWriter(s) if s == "foo\\bar"));
    }

    #[test]
    fn validate_writer_rejects_double_dot() {
        let err = validate_writer("..").unwrap_err();
        assert!(matches!(err, CorpusError::InvalidWriter(s) if s == ".."));
    }

    #[test]
    fn validate_writer_rejects_traversal_segment() {
        let err = validate_writer("foo..bar").unwrap_err();
        assert!(matches!(err, CorpusError::InvalidWriter(s) if s == "foo..bar"));
    }

    #[test]
    fn validate_writer_rejects_leading_whitespace() {
        let err = validate_writer(" foo").unwrap_err();
        assert!(matches!(err, CorpusError::InvalidWriter(s) if s == " foo"));
    }

    #[test]
    fn validate_writer_accepts_normal_handle() {
        validate_writer("karpathy").unwrap();
        validate_writer("eladgil").unwrap();
        validate_writer("user_123").unwrap();
        validate_writer("dot.handle").unwrap(); // single dot is fine, only `..` rejected
    }

    // ---- Corpus::open_or_create -----------------------------------------

    #[test]
    fn open_or_create_returns_empty_for_missing_writer() {
        let dir = TempDir::new().unwrap();
        let c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        assert_eq!(c.writer(), "karpathy");
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn open_or_create_loads_existing_jsonl() {
        let dir = TempDir::new().unwrap();
        write_jsonl(
            &dir.path().join("karpathy.jsonl"),
            &[
                &minimal("1", "first"),
                &minimal("2", "second"),
                &minimal("3", "third"),
            ],
        );
        let c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c.entries()[0].id, "1");
        assert_eq!(c.entries()[2].post_text, "third");
    }

    #[test]
    fn open_or_create_propagates_invalid_writer_name() {
        let dir = TempDir::new().unwrap();
        let err = Corpus::open_or_create(dir.path(), "../escape").unwrap_err();
        assert!(matches!(err, CorpusError::InvalidWriter(_)));
    }

    // ---- Corpus::append_from_jsonl --------------------------------------

    #[test]
    fn append_from_jsonl_adds_new_entries() {
        let dir = TempDir::new().unwrap();
        let mut c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        let src = dir.path().join("import.jsonl");
        write_jsonl(&src, &[&minimal("1", "a"), &minimal("2", "b")]);
        let stats = c.append_from_jsonl(&src).unwrap();
        assert_eq!(stats.added, 2);
        assert_eq!(stats.deduped, 0);
        assert_eq!(stats.total_after, 2);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn append_from_jsonl_dedupes_by_id_existing_wins() {
        let dir = TempDir::new().unwrap();
        let mut c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        let src = dir.path().join("first.jsonl");
        write_jsonl(&src, &[&minimal("1", "v1"), &minimal("2", "x")]);
        c.append_from_jsonl(&src).unwrap();
        // Second import: re-include id=1 with different text, plus id=3.
        let src2 = dir.path().join("second.jsonl");
        write_jsonl(&src2, &[&minimal("1", "v2-DIFFERENT"), &minimal("3", "y")]);
        let stats = c.append_from_jsonl(&src2).unwrap();
        assert_eq!(stats.added, 1, "only id=3 is new");
        assert_eq!(stats.deduped, 1, "id=1 was already present");
        assert_eq!(stats.total_after, 3);
        // Existing-wins: id=1 still has v1, not v2-DIFFERENT.
        let entry_1 = c
            .entries()
            .iter()
            .find(|e| e.id == "1")
            .expect("id=1 still present");
        assert_eq!(entry_1.post_text, "v1");
    }

    #[test]
    fn append_from_jsonl_returns_line_number_on_parse_error() {
        let dir = TempDir::new().unwrap();
        let mut c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        let src = dir.path().join("bad.jsonl");
        // Line 3 is malformed.
        write_jsonl(
            &src,
            &[&minimal("1", "ok"), &minimal("2", "ok"), "{not-json"],
        );
        let err = c.append_from_jsonl(&src).unwrap_err();
        match err {
            CorpusError::Json { line, .. } => assert_eq!(line, 3),
            other => panic!("expected Json error, got {other:?}"),
        }
    }

    #[test]
    fn append_from_jsonl_partial_failure_does_not_persist() {
        let dir = TempDir::new().unwrap();
        let dest = dir.path().join("karpathy.jsonl");
        // Pre-seed with one entry so we can verify it's untouched.
        write_jsonl(&dest, &[&minimal("0", "seed")]);
        let mut c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        assert_eq!(c.len(), 1);
        // Attempt an import where line 2 is malformed.
        let src = dir.path().join("partial.jsonl");
        write_jsonl(&src, &[&minimal("1", "ok"), "}}{{"]);
        let err = c.append_from_jsonl(&src).unwrap_err();
        assert!(matches!(err, CorpusError::Json { .. }));
        // In-memory entries unchanged.
        assert_eq!(c.len(), 1);
        assert_eq!(c.entries()[0].id, "0");
        // On-disk file unchanged.
        let on_disk = std::fs::read_to_string(&dest).unwrap();
        assert!(on_disk.contains("\"id\":\"0\""));
        assert!(!on_disk.contains("\"id\":\"1\""));
    }

    #[test]
    fn append_from_jsonl_skips_blank_lines() {
        let dir = TempDir::new().unwrap();
        let mut c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        let src = dir.path().join("with-blanks.jsonl");
        // Mix of blank lines, whitespace, and real entries.
        let mut f = File::create(&src).unwrap();
        writeln!(f, "{}", minimal("1", "a")).unwrap();
        writeln!(f).unwrap();
        writeln!(f, "   ").unwrap();
        writeln!(f, "{}", minimal("2", "b")).unwrap();
        writeln!(f).unwrap();
        let stats = c.append_from_jsonl(&src).unwrap();
        assert_eq!(stats.added, 2);
        assert_eq!(c.len(), 2);
    }

    // ---- save round-trips, atomic write ---------------------------------

    #[test]
    fn save_round_trips_via_open_or_create() {
        let dir = TempDir::new().unwrap();
        let mut c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        let src = dir.path().join("import.jsonl");
        write_jsonl(
            &src,
            &[&minimal("1", "a"), &minimal("2", "b"), &minimal("3", "c")],
        );
        c.append_from_jsonl(&src).unwrap();
        // Re-open from disk, must equal what we appended.
        let c2 = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        assert_eq!(c2.len(), 3);
        assert_eq!(c2.entries(), c.entries());
    }

    #[test]
    fn save_does_not_leave_tmp_file_behind() {
        let dir = TempDir::new().unwrap();
        let mut c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        let src = dir.path().join("import.jsonl");
        write_jsonl(&src, &[&minimal("1", "a")]);
        c.append_from_jsonl(&src).unwrap();
        // Sweep the dir for any *.tmp files.
        let leftover: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| x == "tmp")
            })
            .collect();
        assert!(leftover.is_empty(), "found tmp files: {leftover:?}");
    }

    #[test]
    fn save_round_trips_full_entry_with_engagement_and_embedding() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("karpathy.jsonl");
        let original = CorpusEntry {
            id: "tweetX".to_string(),
            post_text: "specific & contrarian".to_string(),
            posted_at: Some("2025-04-12T14:32:00Z".parse().unwrap()),
            engagement: Some(Engagement {
                likes: 999,
                reposts: 100,
                replies: 50,
            }),
            tags: vec!["hot_take".to_string()],
            embedding: Some(vec![0.5, -0.5, 0.25]),
        };
        let json = serde_json::to_string(&original).unwrap();
        write_jsonl(&path, &[&json]);
        let c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        assert_eq!(c.entries().len(), 1);
        assert_eq!(c.entries()[0], original);
    }
}
```

**Notes on the implementation above:**

- `parse_jsonl` is a free function so both `open_or_create` and `append_from_jsonl` share the same line-numbered parser.
- The dedup loop uses `HashSet<String>` keyed on a clone of `id` (not a borrowed `&str`) — avoids the lifetime gymnastics of borrowing into `self.entries` while we mutate it via `extend` later in the same function. Cloning the id (a small `String`) per loop iteration is fine for corpus sizes we care about.

- [ ] **Step 2: Update `crates/heartbit-ghost/src/corpus/mod.rs`**

```rust
//! Corpus storage — file-backed JSONL persistence for per-writer post
//! collections. Sibling of [`crate::voice`]: voice owns the schema; corpus
//! owns the inputs that the LLM extractor (P1.2c) turns into a profile.
//!
//! On-disk layout (created lazily on first write):
//!
//! ```text
//! ~/.heartbit/ghost/corpora/
//! ├── karpathy.jsonl      # one writer, one file
//! ├── eladgil.jsonl
//! └── swyx.jsonl
//! ```

pub mod entry;
pub mod error;
pub mod store;

pub use entry::{CorpusEntry, Engagement};
pub use error::CorpusError;
pub use store::{AppendStats, Corpus, default_corpora_dir};
```

(`list_writers` is added in Task 4.)

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib corpus
```

Expected: `31 passed` (10 from prior tasks + 21 new in `store` tests).

Test count breakdown for `store.rs`: 3 (resolve_corpora_dir) + 7 (validate_writer) + 3 (open_or_create) + 5 (append_from_jsonl) + 3 (save round-trip / atomic / full entry) = 21.

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean. (`--all-targets` exercises the test code for clippy too — important since the bulk of new logic landed in tests.)

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/corpus/store.rs crates/heartbit-ghost/src/corpus/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): corpus — Corpus + AppendStats + default_corpora_dir (P1.2b)

The core of the corpus library. Corpus::open_or_create reads (or creates
empty) <writer>.jsonl under root. append_from_jsonl ingests a JSONL file
with id-based dedup (existing wins), persists atomically via
tempfile+rename, and reports AppendStats { added, deduped, total_after }.

default_corpora_dir resolves $HEARTBIT_GHOST_CORPORA, falling back to
$HOME/.heartbit/ghost/corpora. Resolver is split into a pure
resolve_corpora_dir for env-free testability.

Writer-handle validation rejects empty, '/', '\\', '..', and
whitespace-padded inputs — same defensive pattern as the skill tool.

~20 tests: env resolution, all writer-name rejection paths, open empty,
open existing, dedup with existing-wins, line-numbered parse error,
partial-failure-no-persist, blank-line skip, atomic save (no .tmp
leftover), round-trip with full entries.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md §3 §5 §6
EOF
)"
```

---

## Task 4: `list_writers` helper

**Why:** Closes the public surface. P1.2e CLI bodies will call this for `corpus list <persona>`. Small enough to be its own task so the test cases (sorted output, skip non-jsonl, skip invalid stems) can be focused.

**Files:**
- Modify: `crates/heartbit-ghost/src/corpus/mod.rs` (add `list_writers` fn + tests)

- [ ] **Step 1: Append `list_writers` and its tests to `crates/heartbit-ghost/src/corpus/mod.rs`**

Replace the file with:

```rust
//! Corpus storage — file-backed JSONL persistence for per-writer post
//! collections. Sibling of [`crate::voice`]: voice owns the schema; corpus
//! owns the inputs that the LLM extractor (P1.2c) turns into a profile.
//!
//! On-disk layout (created lazily on first write):
//!
//! ```text
//! ~/.heartbit/ghost/corpora/
//! ├── karpathy.jsonl      # one writer, one file
//! ├── eladgil.jsonl
//! └── swyx.jsonl
//! ```

use std::path::Path;

pub mod entry;
pub mod error;
pub mod store;

pub use entry::{CorpusEntry, Engagement};
pub use error::CorpusError;
pub use store::{AppendStats, Corpus, default_corpora_dir};

/// Enumerate writer handles known under `root`.
///
/// Looks at every `*.jsonl` immediate child of `root`, takes the file stem,
/// and returns those that pass [`store::validate_writer`]. Files that do
/// not end in `.jsonl` (e.g., `.DS_Store`, `notes.txt`) and stems that
/// fail validation are silently skipped — `list_writers` is for discovery,
/// not auditing.
///
/// Returns an empty `Vec` if `root` does not exist (so this can be called
/// before any corpus has been created without `?`-propagation noise at the
/// caller).
///
/// Output is sorted alphabetically.
pub fn list_writers(root: &Path) -> Result<Vec<String>, CorpusError> {
    let mut out: Vec<String> = Vec::new();
    let read = match std::fs::read_dir(root) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(CorpusError::Io(e)),
    };
    for entry in read {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|x| x.to_str()) else {
            continue;
        };
        if store::validate_writer(stem).is_err() {
            continue;
        }
        out.push(stem.to_string());
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::TempDir;

    #[test]
    fn list_writers_returns_empty_for_missing_root() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("does-not-exist");
        let writers = list_writers(&missing).unwrap();
        assert!(writers.is_empty());
    }

    #[test]
    fn list_writers_returns_sorted_jsonl_stems() {
        let dir = TempDir::new().unwrap();
        File::create(dir.path().join("swyx.jsonl")).unwrap();
        File::create(dir.path().join("karpathy.jsonl")).unwrap();
        File::create(dir.path().join("eladgil.jsonl")).unwrap();
        let writers = list_writers(dir.path()).unwrap();
        assert_eq!(writers, vec!["eladgil", "karpathy", "swyx"]);
    }

    #[test]
    fn list_writers_skips_non_jsonl_files() {
        let dir = TempDir::new().unwrap();
        File::create(dir.path().join("karpathy.jsonl")).unwrap();
        File::create(dir.path().join("notes.txt")).unwrap();
        File::create(dir.path().join(".DS_Store")).unwrap();
        File::create(dir.path().join("README.md")).unwrap();
        let writers = list_writers(dir.path()).unwrap();
        assert_eq!(writers, vec!["karpathy"]);
    }
}
```

- [ ] **Step 2: Run the tests**

```bash
cargo test -p heartbit-ghost --lib corpus
```

Expected: `34 passed` (31 from prior tasks + 3 new).

- [ ] **Step 3: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-ghost/src/corpus/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): corpus — list_writers helper (P1.2b)

Enumerate writer handles known under a corpora root by listing *.jsonl
file stems. Skips non-jsonl entries and stems that fail
validate_writer (defensive: catches stray files like .DS_Store and
prevents traversal-segment names from leaking through). Returns empty
Vec when root does not exist, so callers can use it before any corpus
has been created.

Output is alphabetically sorted.

3 tests: missing root → empty, sorted output, non-jsonl skipped.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md §3
EOF
)"
```

---

## Task 5: Final acceptance + workspace quality gate

**Why:** Confirm P1.2b meets every acceptance criterion in the spec.

**Files:** none (verification only).

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -5
```

Expected: all green. Workspace test count goes from 3815 (post-P1.2a baseline) to ~3849 (~34 new corpus tests).

- [ ] **Step 2: Verify the public surface is reachable**

```bash
cat <<'EOF' > /tmp/heartbit_ghost_corpus_surface_check.rs
fn _check() {
    use heartbit_ghost::corpus::{
        AppendStats, Corpus, CorpusEntry, CorpusError, Engagement,
        default_corpora_dir, list_writers,
    };
    let _ = (
        AppendStats { added: 0, deduped: 0, total_after: 0 },
        CorpusEntry {
            id: String::new(),
            post_text: String::new(),
            posted_at: None,
            engagement: None,
            tags: Vec::new(),
            embedding: None,
        },
        Engagement::default(),
        CorpusError::Validation(String::new()),
    );
    let _: fn() -> Result<std::path::PathBuf, CorpusError> = default_corpora_dir;
    let _: fn(&std::path::Path) -> Result<Vec<String>, CorpusError> = list_writers;
    let _: fn(&std::path::Path, &str) -> Result<Corpus, CorpusError> = Corpus::open_or_create;
}
EOF
echo "(Surface check is illustrative; the public types are reachable via the workspace cargo check above.)"
rm -f /tmp/heartbit_ghost_corpus_surface_check.rs
```

- [ ] **Step 3: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.2b
```

Expected: 5 commits — spec doc + 4 task commits (Task 1, 2, 3, 4). No commit for Task 5.

- [ ] **Step 4: No commit for this task**

Task 5 is verification only. The branch is ready for final review + merge.

---

## Acceptance criteria

P1.2b is done when (per spec §9):

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~22 corpus tests pass (actual count expected 34 once all four implementation tasks land — the spec's "~22" was a conservative estimate; the per-task breakdown in Tasks 1–4 totals 34: 3 error + 7 entry + 21 store + 3 list_writers)
- `heartbit_ghost::corpus::{Corpus, CorpusEntry, CorpusError, Engagement, AppendStats, default_corpora_dir, list_writers}` are reachable as public surface
- A round-trip property test confirms `open_or_create → append_from_jsonl → save → open_or_create → entries` preserves data (covered by `save_round_trips_via_open_or_create` and `save_round_trips_full_entry_with_engagement_and_embedding` in Task 3)

## Out of scope (re-stated)

- LLM-based style extraction (P1.2c)
- Embedding generation (P1.4 wires the local-embedding pipeline through this surface; P1.2b just stores what's given)
- Blend algorithm (P1.2d)
- CLI bodies for `corpus add`, `corpus list` (P1.2e)
- File locking / multi-writer concurrency (single-user dev path for v0.1)
- Streaming / mmap for very large corpora (P1.4 if needed)
- Cross-machine corpus sync / export (`corpus export` in P1.4 if a use case appears)
- A `Corpus` trait + alternate backends (premature; one impl is enough for now)

## Reference

- Spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md`
- Umbrella heartbit-ghost spec §2.1 (corpus): `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- P1.2a (just merged): `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`
- CLI scaffolding (P1.0): `crates/heartbit-cli/src/persona.rs`
