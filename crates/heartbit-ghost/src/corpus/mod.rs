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
