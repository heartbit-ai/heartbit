//! [`RunJournal`] — deterministic resume for workflow runs.
//!
//! A run journal memoizes each [`agent`](super::agent::agent) leaf's output to a
//! JSONL file so a re-run can replay unchanged work instead of re-calling the
//! model. It is **content-addressed**: each entry is keyed by
//! `(content_hash, occurrence)` where `content_hash` covers the call's *inputs*
//! (prompt, requested model, schema) and `occurrence` is how many times that
//! exact hash has been issued so far in the run.
//!
//! # Why content-addressed, not positional
//!
//! Keying on a global call index would make inserting or removing one early
//! `agent()` call shift every later index, so the whole tail would miss on
//! resume even though its prompts are byte-identical. Content addressing keeps a
//! call's key stable regardless of position, so an unchanged call replays
//! wherever it sits. The `occurrence` tiebreaker keeps byte-identical calls in a
//! loop distinct. A call whose inputs *did* change re-runs, and so do any later
//! calls whose prompts are derived from its (now different) output — data
//! dependencies propagate invalidation for free.
//!
//! # Soundness and scope (read before relying on this)
//!
//! - **Return values, not side effects.** The journal restores an agent's
//!   returned [`AgentOutput`]; it does **not** re-execute the agent, so any
//!   filesystem/network/message side effects are *not* replayed. Only journal
//!   agents that are pure with respect to their return value. (P5 worktree
//!   isolation will gate side-effecting agents out of the journal.)
//! - **Single process, single journal, fixed provider.** One [`RunJournal`] per
//!   path per process; no advisory file lock is taken in this phase, so two
//!   journals on one path corrupt the log. The journal is also scoped to the
//!   ctx's provider — sharing a path across different providers/models would
//!   replay the wrong model's answer (the requested model is hashed, but the
//!   provider identity is not).
//! - **Concurrency.** Call-for-call replay is guaranteed only for
//!   sequentially-issued calls. Inside `parallel()`/`pipeline()` the occurrence
//!   counter races, so those calls may re-run on resume — safe, since a raced
//!   occurrence is a false *miss*, never a false hit.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::agent::AgentOutput;
use crate::error::Error;

/// Whether to start a fresh journal or resume from an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeMode {
    /// Truncate any existing journal and start empty (no replay).
    Fresh,
    /// Load an existing journal (if any) and replay matching calls.
    Resume,
}

/// Identity of one journaled agent call: its input hash plus how many times that
/// hash had already been issued in the run (to disambiguate identical calls).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct CallKey {
    /// Hex SHA-256 over the call's inputs (prompt, requested model, schema).
    pub content_hash: String,
    /// 0-based count of prior calls with this exact `content_hash` this run.
    pub occurrence: u64,
}

/// One line of the JSONL journal.
#[derive(Debug, Serialize, Deserialize)]
struct JournalRecord {
    /// Format version, for forward compatibility.
    v: u16,
    key: CallKey,
    output: AgentOutput,
}

/// A content-addressed, JSONL-backed memo of agent outputs for one run.
#[derive(Debug)]
pub struct RunJournal {
    /// Loaded + appended entries, keyed by [`CallKey`].
    entries: Mutex<HashMap<CallKey, AgentOutput>>,
    /// Per-`content_hash` issue counter, for assigning [`CallKey::occurrence`].
    seen: Mutex<HashMap<String, u64>>,
    /// Append handle for new records.
    file: Mutex<std::fs::File>,
}

impl RunJournal {
    /// Open a journal at `path` (a `.jsonl` file). [`ResumeMode::Fresh`]
    /// truncates any existing file; [`ResumeMode::Resume`] loads existing
    /// records (skipping a torn final line from a prior crash) and appends.
    pub fn open(path: &Path, mode: ResumeMode) -> Result<Self, Error> {
        use std::fs::OpenOptions;
        use std::io::{BufRead, BufReader};

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| Error::Store(e.to_string()))?;
        }

        let mut entries: HashMap<CallKey, AgentOutput> = HashMap::new();
        if mode == ResumeMode::Resume && path.exists() {
            let file = std::fs::File::open(path).map_err(|e| Error::Store(e.to_string()))?;
            for line in BufReader::new(file).lines() {
                let line = line.map_err(|e| Error::Store(e.to_string()))?;
                if line.trim().is_empty() {
                    continue;
                }
                // A torn final line (crash mid-write) or a forward-incompatible
                // record fails to parse — skip it rather than abort the resume.
                if let Ok(rec) = serde_json::from_str::<JournalRecord>(&line) {
                    entries.insert(rec.key, rec.output);
                }
            }
        }

        // Fresh truncates; Resume appends (creating the file if absent).
        let file = match mode {
            ResumeMode::Fresh => OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path),
            ResumeMode::Resume => OpenOptions::new().append(true).create(true).open(path),
        }
        .map_err(|e| Error::Store(e.to_string()))?;

        Ok(Self {
            entries: Mutex::new(entries),
            seen: Mutex::new(HashMap::new()),
            file: Mutex::new(file),
        })
    }

    /// Assign the next 0-based occurrence index for `content_hash` (mutating).
    pub(crate) fn next_occurrence(&self, content_hash: &str) -> u64 {
        let mut seen = self.seen.lock().expect("journal seen lock poisoned");
        let counter = seen.entry(content_hash.to_string()).or_insert(0);
        let occurrence = *counter;
        *counter += 1;
        occurrence
    }

    /// Look up a cached output by key (a clone, so callers don't hold the lock).
    pub(crate) fn lookup(&self, key: &CallKey) -> Option<AgentOutput> {
        self.entries
            .lock()
            .expect("journal entries lock poisoned")
            .get(key)
            .cloned()
    }

    /// Append one record to the JSONL file and the in-memory map. The record and
    /// its trailing newline are written in a single `write_all` so a crash
    /// leaves at most a torn *final* line (skipped on the next resume).
    pub(crate) fn append(&self, key: &CallKey, output: &AgentOutput) -> Result<(), Error> {
        use std::io::Write;

        let record = JournalRecord {
            v: 1,
            key: key.clone(),
            output: output.clone(),
        };
        let mut line = serde_json::to_string(&record)?;
        line.push('\n');
        {
            let mut file = self.file.lock().expect("journal file lock poisoned");
            file.write_all(line.as_bytes())
                .map_err(|e| Error::Store(e.to_string()))?;
            file.flush().map_err(|e| Error::Store(e.to_string()))?;
        }
        self.entries
            .lock()
            .expect("journal entries lock poisoned")
            .insert(key.clone(), output.clone());
        Ok(())
    }
}

/// Recursively sort object keys so logically-equal JSON hashes identically,
/// independent of the `serde_json` `preserve_order` feature (which is off in
/// this workspace but could be turned on transitively by Cargo feature
/// unification). Arrays keep their order; object *values* are canonicalized too.
pub(crate) fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<String, Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), canonical_json(v)))
                .collect();
            // serde_json::Map collects from a BTreeMap in sorted-key order.
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_json).collect()),
        other => other.clone(),
    }
}

/// Lowercase-hex a byte slice (avoids adding a `hex` dependency to the crate).
fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Hex SHA-256 over a call's output-determining inputs. `model` is `None` today
/// (no per-call model override yet); it is part of the hashed format now so the
/// on-disk journal stays valid when a future `.model()` starts passing `Some`.
pub(crate) fn content_hash(prompt: &str, model: Option<&str>, schema: Option<&Value>) -> String {
    let mut hasher = Sha256::new();
    // Domain-separate fields with a 0x00 byte so concatenations can't collide
    // (prompt "a" + model "b" must not equal prompt "ab" + model "").
    hasher.update(prompt.as_bytes());
    hasher.update([0u8]);
    hasher.update(model.unwrap_or("").as_bytes());
    hasher.update([0u8]);
    if let Some(schema) = schema {
        let canonical = serde_json::to_string(&canonical_json(schema)).unwrap_or_default();
        hasher.update(canonical.as_bytes());
    }
    to_hex(&hasher.finalize())
}

/// Derive a stable run id from a seed (e.g. a workflow name + serialized args)
/// so "same inputs ⇒ same journal file". Hex SHA-256 of the seed.
pub fn derive_run_id(seed: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    to_hex(&hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(text: &str, input: u32, output: u32) -> AgentOutput {
        AgentOutput {
            result: text.to_string(),
            tokens_used: crate::llm::types::TokenUsage {
                input_tokens: input,
                output_tokens: output,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn key(hash: &str, occurrence: u64) -> CallKey {
        CallKey {
            content_hash: hash.to_string(),
            occurrence,
        }
    }

    // ----- canonicalization + hashing -----

    #[test]
    fn canonical_json_sorts_nested_keys() {
        let a = serde_json::json!({ "b": 1, "a": { "y": 2, "x": 3 } });
        let b = serde_json::json!({ "a": { "x": 3, "y": 2 }, "b": 1 });
        // Same logical content, different key order -> identical canonical form.
        assert_eq!(canonical_json(&a), canonical_json(&b));
        // Canonical serialization has sorted keys.
        let s = serde_json::to_string(&canonical_json(&a)).unwrap();
        assert!(s.starts_with(r#"{"a":"#), "not sorted: {s}");
    }

    #[test]
    fn content_hash_stable_under_schema_key_reorder() {
        let s1 = serde_json::json!({ "type": "object", "required": ["x"] });
        let s2 = serde_json::json!({ "required": ["x"], "type": "object" });
        assert_eq!(
            content_hash("p", None, Some(&s1)),
            content_hash("p", None, Some(&s2)),
            "reordered-but-equal schemas must hash the same"
        );
    }

    #[test]
    fn content_hash_distinguishes_inputs() {
        let base = content_hash("prompt", None, None);
        assert_ne!(base, content_hash("other", None, None), "prompt matters");
        assert_ne!(
            base,
            content_hash("prompt", Some("haiku"), None),
            "model matters"
        );
        let schema = serde_json::json!({ "type": "string" });
        assert_ne!(
            base,
            content_hash("prompt", None, Some(&schema)),
            "schema matters"
        );
    }

    #[test]
    fn derive_run_id_is_deterministic_and_seed_sensitive() {
        assert_eq!(derive_run_id("a|b"), derive_run_id("a|b"));
        assert_ne!(derive_run_id("a|b"), derive_run_id("a|c"));
    }

    // ----- journal open / append / lookup -----

    #[test]
    fn fresh_journal_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let j = RunJournal::open(&dir.path().join("j.jsonl"), ResumeMode::Fresh).unwrap();
        assert!(j.lookup(&key("h", 0)).is_none());
    }

    #[test]
    fn append_then_lookup_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let j = RunJournal::open(&dir.path().join("j.jsonl"), ResumeMode::Fresh).unwrap();
        j.append(&key("h", 0), &out("hello", 10, 5)).unwrap();
        let got = j.lookup(&key("h", 0)).expect("hit");
        assert_eq!(got.result, "hello");
        assert_eq!(got.tokens_used.input_tokens, 10);
    }

    #[test]
    fn occurrence_disambiguates_identical_content() {
        let dir = tempfile::tempdir().unwrap();
        let j = RunJournal::open(&dir.path().join("j.jsonl"), ResumeMode::Fresh).unwrap();
        // Same hash issued twice -> occurrences 0 then 1.
        assert_eq!(j.next_occurrence("h"), 0);
        assert_eq!(j.next_occurrence("h"), 1);
        assert_eq!(j.next_occurrence("other"), 0);
        j.append(&key("h", 0), &out("first", 1, 1)).unwrap();
        j.append(&key("h", 1), &out("second", 1, 1)).unwrap();
        assert_eq!(j.lookup(&key("h", 0)).unwrap().result, "first");
        assert_eq!(j.lookup(&key("h", 1)).unwrap().result, "second");
    }

    #[test]
    fn resume_reads_existing_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        {
            let j = RunJournal::open(&path, ResumeMode::Fresh).unwrap();
            j.append(&key("h", 0), &out("persisted", 7, 3)).unwrap();
        }
        // Re-open in Resume mode: the entry is loaded from disk.
        let j2 = RunJournal::open(&path, ResumeMode::Resume).unwrap();
        assert_eq!(j2.lookup(&key("h", 0)).unwrap().result, "persisted");
    }

    #[test]
    fn fresh_truncates_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        {
            let j = RunJournal::open(&path, ResumeMode::Fresh).unwrap();
            j.append(&key("h", 0), &out("old", 1, 1)).unwrap();
        }
        // Fresh again -> previous entries gone.
        let j2 = RunJournal::open(&path, ResumeMode::Fresh).unwrap();
        assert!(j2.lookup(&key("h", 0)).is_none());
    }

    #[test]
    fn torn_final_line_is_skipped_on_resume() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        {
            let j = RunJournal::open(&path, ResumeMode::Fresh).unwrap();
            j.append(&key("h", 0), &out("good", 1, 1)).unwrap();
        }
        // Simulate a crash mid-write: append a partial, unterminated JSON line.
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(br#"{"v":1,"key":{"content_hash":"h","occ"#)
                .unwrap();
        }
        // Resume must load the good record and ignore the torn one (no panic).
        let j2 = RunJournal::open(&path, ResumeMode::Resume).unwrap();
        assert_eq!(j2.lookup(&key("h", 0)).unwrap().result, "good");
    }

    #[test]
    fn resume_on_missing_file_is_empty_not_error() {
        let dir = tempfile::tempdir().unwrap();
        let j = RunJournal::open(&dir.path().join("absent.jsonl"), ResumeMode::Resume).unwrap();
        assert!(j.lookup(&key("h", 0)).is_none());
    }
}
