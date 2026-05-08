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
        CorpusError::Validation("neither HEARTBIT_GHOST_CORPORA nor HOME is set".to_string())
    })?;
    Ok(PathBuf::from(home).join(".heartbit/ghost/corpora"))
}

/// Validate a writer handle. Rejects empty, whitespace-only, or anything
/// containing `/`, `\`, or `..`.
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
        let prev_len = self.entries.len();
        self.entries.extend(to_append);
        if let Err(e) = self.save() {
            self.entries.truncate(prev_len);
            return Err(e);
        }
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
                let line = serde_json::to_string(entry)
                    .map_err(|source| CorpusError::Json { line: 0, source })?;
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
    fn append_from_jsonl_rolls_back_in_memory_on_save_failure() {
        let dir = TempDir::new().unwrap();
        // Open the (empty) corpus first — open_or_create reads the file if it
        // exists, so we must do this BEFORE planting the blocking directory
        // (File::open on a directory succeeds on Linux, but reading lines
        // then fails with IsADirectory before we'd ever reach append).
        let mut c = Corpus::open_or_create(dir.path(), "karpathy").unwrap();
        assert!(c.is_empty());

        // Now pre-create a *directory* at the corpus path so the final
        // fs::rename in save() fails (cross-platform: rename(file,
        // non-empty-dir) errors on Linux, macOS, and Windows).
        let corpus_path = dir.path().join("karpathy.jsonl");
        std::fs::create_dir(&corpus_path).unwrap();
        // Drop a sentinel file inside the directory so it's non-empty —
        // some platforms allow rename over an empty directory.
        std::fs::write(corpus_path.join("sentinel"), b"keep me").unwrap();

        let src = dir.path().join("import.jsonl");
        write_jsonl(&src, &[&minimal("1", "a"), &minimal("2", "b")]);

        let err = c.append_from_jsonl(&src).unwrap_err();
        assert!(
            matches!(err, CorpusError::Io(_)),
            "expected Io, got {err:?}"
        );

        // Critical assertion: in-memory entries rolled back to pre-call state.
        assert!(
            c.is_empty(),
            "rollback failed — in-memory has {} entries after save error",
            c.len()
        );
    }

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
