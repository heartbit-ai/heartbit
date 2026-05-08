# heartbit-ghost P1.2b — corpus storage design

**Status:** approved 2026-05-08
**Branch:** `feat/heartbit-ghost-p1.2b`
**Predecessor:** P1.2a (style profile schema, merged to main 2026-05-07)
**Successors:** P1.2c (LLM style extractor), P1.2d (blend algorithm), P1.2e (CLI bodies)

## 1. Goal

Persist and retrieve raw post collections per writer — the input that P1.2c's LLM extractor will consume to build a `StyleProfile`. The library surface needs to be enough that:

- P1.2c can call `Corpus::open_or_create(...)` + `corpus.entries()` to feed the extractor
- P1.2e CLI bodies can wire `corpus add <writer> <path>` and `corpus list <persona>` against this without rework

Out of scope for this phase: any LLM call, any embedding generation, any blending, any CLI body. Those are P1.2c / P1.2d / P1.2e on follow-up branches.

## 2. Architecture

A new `corpus/` module inside `heartbit-ghost`, sibling to `voice/`:

```
crates/heartbit-ghost/src/corpus/
├── mod.rs       # public surface, re-exports, list_writers helper
├── entry.rs     # CorpusEntry, Engagement
├── store.rs     # Corpus, AppendStats, default_corpora_dir
└── error.rs     # CorpusError
```

On-disk layout (created lazily on first write):

```
~/.heartbit/ghost/corpora/
├── karpathy.jsonl     # one writer, one file
├── eladgil.jsonl
└── swyx.jsonl
```

Each `<writer>.jsonl` is line-delimited JSON. Each line is a `CorpusEntry`. `corpus add` is append-only with id-based dedup so re-running on the same file is idempotent.

**Path resolution** — single helper:

```rust
pub fn default_corpora_dir() -> Result<PathBuf, CorpusError>;
```

Checks `$HEARTBIT_GHOST_CORPORA` first, falls back to `$HOME/.heartbit/ghost/corpora` via `dirs::home_dir()`. Lets tests use a `TempDir` without monkey-patching `$HOME`.

**No new dependencies** — `serde`, `serde_json`, `thiserror`, `chrono`, `dirs` are already in the workspace.

## 3. Public API

```rust
// in heartbit-ghost::corpus

pub fn default_corpora_dir() -> Result<PathBuf, CorpusError>;
pub fn list_writers(root: &Path) -> Result<Vec<String>, CorpusError>;

pub struct Corpus { /* writer, path, entries */ }

impl Corpus {
    pub fn open_or_create(root: &Path, writer: &str) -> Result<Self, CorpusError>;
    pub fn writer(&self) -> &str;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn entries(&self) -> &[CorpusEntry];
    pub fn append_from_jsonl(&mut self, path: &Path) -> Result<AppendStats, CorpusError>;
    pub fn save(&self) -> Result<(), CorpusError>;
}

pub struct AppendStats {
    pub added: usize,
    pub deduped: usize,
    pub total_after: usize,
}
```

`append_from_jsonl` calls `save()` internally on success — the common-case flow (`open_or_create` → `append_from_jsonl`) needs no manual save call.

`Remove` is **not** in P1.2b. CLI scaffolding has only `Add` and `List`; the umbrella roadmap mentioned a `Remove` that was never wired. P1.2e can decide whether `Remove` means single-entry or whole-writer when it has a real CLI command demanding the operation.

## 4. Data types

```rust
// corpus/entry.rs

/// One post in a writer's reference corpus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusEntry {
    /// Stable identifier — typically the X tweet id as a string.
    /// Used for dedup on re-import.
    pub id: String,

    /// The post text (no markdown stripping; stored verbatim).
    pub post_text: String,

    /// Original posting time; RFC3339.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posted_at: Option<DateTime<Utc>>,

    /// Engagement metrics from the source (best-effort).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engagement: Option<Engagement>,

    /// Manual tags: ["thread_opener", "hot_take", "self_deprecating"].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,

    /// Pre-computed embedding (P1.2b doesn't generate; only stores).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Engagement {
    #[serde(default)] pub likes: u64,
    #[serde(default)] pub reposts: u64,
    #[serde(default)] pub replies: u64,
}
```

**Field decisions:**

- `posted_at: DateTime<Utc>` (chrono) — workspace dep already present. RFC3339 string in JSONL ↔ `chrono::DateTime` in Rust.
- `engagement` — flat struct, matches the umbrella spec's nested-object shape.
- `embedding: Vec<f32>` — `f32` (not `f64`) matches the project's local-embedding pipeline and halves disk footprint when present.
- `skip_serializing_if` on optional fields keeps emitted JSONL minimal when posts have only `id` + `post_text`.
- `deny_unknown_fields` — fail-loud-on-typos discipline carried over from P1.2a.

`writer_handle` from the umbrella spec is **not** stored per entry — it's implicit from the file the entry lives in. Storing it per entry would be redundant and would let imports drift (a `karpathy.jsonl` file with `writer_handle: "eladgil"` lines is a footgun).

## 5. Error handling

```rust
// corpus/error.rs
#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json on line {line}: {source}")]
    Json {
        line: usize,
        #[source] source: serde_json::Error,
    },

    #[error("invalid writer name '{0}': must be non-empty, no '/', '\\', or '..'")]
    InvalidWriter(String),

    #[error("validation: {0}")]
    Validation(String),
}
```

**Semantics:**

- **Per-line JSONL parse errors are fatal.** If line 47 fails to parse, `append_from_jsonl` returns `CorpusError::Json { line: 47, source }` and **does not persist** any of the file's entries. Partial ingest leaves the corpus in an ambiguous state; caller fixes the JSONL and re-runs.
- **Empty / blank lines in JSONL: skipped** (not errored). Many JSONL exporters emit a trailing newline.
- **Dedup: existing entry wins.** Tags / engagement / embeddings the user has already curated are not silently overwritten on re-import. The new entry is counted as `deduped`.
- **Atomic save**: `save()` writes to `<path>.tmp`, then `fs::rename`s over the original. Prevents half-written corpora on crash mid-write.
- **Writer-handle validation** rejects empty string, `/`, `\`, `..`, and leading/trailing whitespace. `list_writers` skips files whose stems fail validation rather than erroring (defensive against random files in the dir).

The `Validation` variant is reserved for future invariants (e.g., `len(post_text) ≤ 280`); P1.2b doesn't construct it but having it on the enum lets P1.2c add invariants without a breaking change to `CorpusError`.

## 6. Concurrency, scale

- **No file locking.** P1.2b is single-process, library-only, single-user dev path. Two concurrent `append_from_jsonl` calls on the same writer file can race — fine for v0.1.
- **All entries in memory on `open_or_create`.** A writer with 10k posts × 4 KB embedding ≈ 40 MB — fine for in-memory representation. If a writer's corpus grows to millions, P1.4 can introduce streaming / mmap; not now.

## 7. Testing

~22 tests, all in-tree, all using `tempfile::TempDir` for filesystem isolation. No mocks — the surface is small enough that real I/O is the simplest correct test.

**`corpus/entry.rs` (~6 tests):** serde round-trip (minimal + full), Engagement default zero, optional-field skip serialize, `deny_unknown_fields` reject, RFC3339 round-trip.

**`corpus/store.rs` (~14 tests):** env var resolution, home-dir fallback, open empty / open existing, all four writer-name validation paths (empty, slash, backslash, double-dot), append fresh / append-with-dedup, line-numbered parse error, partial-failure-no-persist, blank-line-skip, save→reopen round-trip, atomic-tempfile-no-lingering.

**`corpus/mod.rs` (~2 tests):** `list_writers` sorted output, `list_writers` skips invalid filenames (`.DS_Store`, `*.txt`).

**1 round-trip property test:** generic write→save→reopen→equal across the data types.

Quality gate (mirrors P1.2a): `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features`. Workspace test count expected to go from 3815 → ~3837.

## 8. Architecture decisions (ADs)

**AD-1 — File-backed JSONL, not the memory system.** The umbrella spec §2.1 said "Stored under the persona's namespace in the memory system (Episodic memory)" without rationale. A writer's reference corpus is read-mostly static reference data; the memory system's mechanism (decay, reflection, BM25 indexing of prose, consolidation) doesn't help and would tie corpus persistence to whichever memory backend the rest of the daemon uses. File-backed JSONL is the simplest thing that works, is trivially upgradable to a `Corpus` trait if a second backend ever appears, and matches the JSONL ingest format the CLI already accepts. Replaces umbrella spec §2.1 storage choice.

**AD-2 — User-home, not repo-relative.** Corpora are personal data, potentially copyrighted, potentially large. Storing under `~/.heartbit/ghost/corpora/` keeps source clean and matches OS conventions. Cross-team sharing is a future feature (`corpus export` in P1.4 if needed).

**AD-3 — Minimal entry schema.** `id` + `post_text` required, everything else optional. Lets manually authored corpora work without engagement metrics, lets corpora without embeddings ingest before P1.4 wires the local-embedding pipeline through this code path, and aligns with the umbrella extractor's "missing-OK" prompt design.

**AD-4 — `id` from source (X tweet id), not auto-generated.** Stable across imports; allows dedup; keeps a back-reference to the source. Manually-authored corpora that don't have a natural id provide one in the JSONL (any string works — `manual-001`, a UUID, a hash; the library doesn't care).

**AD-5 — Existing-wins dedup.** Re-running `corpus add` on the same JSONL is a no-op rather than a destructive overwrite. User-curated tags / engagement / embeddings on the existing entry are preserved.

**AD-6 — No `Remove` in P1.2b.** CLI scaffolding has only Add + List. The umbrella roadmap mentioned `Remove` but never wired it. YAGNI: P1.2e decides what Remove means (single-entry by id? clear whole writer?) when it has a real command demanding the operation.

**AD-7 — Writer-handle validation rejects path traversal.** Same defensive pattern as P1.1 tools and the skill tool. `corpus add ../../etc/passwd posts.jsonl` cannot escape `~/.heartbit/ghost/corpora/`.

## 9. Acceptance criteria

P1.2b is done when:

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~22 corpus tests pass; coverage spans serde round-trip, all error paths, atomic-save behavior
- `heartbit_ghost::corpus::{Corpus, CorpusEntry, CorpusError, Engagement, AppendStats, default_corpora_dir, list_writers}` are reachable as public surface
- A round-trip property test confirms `open_or_create → append_from_jsonl → save → open_or_create → entries` preserves data

## 10. Out of scope (re-stated)

- LLM-based style extraction (P1.2c)
- Embedding generation (P1.4 wires the local-embedding pipeline through this surface; P1.2b just stores what's given)
- Blend algorithm (P1.2d)
- CLI bodies for `corpus add`, `corpus list` (P1.2e)
- File locking / multi-writer concurrency (single-user dev path for v0.1)
- Streaming / mmap for very large corpora (P1.4 if needed)
- Cross-machine corpus sync / export (`corpus export` in P1.4 if a use case appears)
- A `Corpus` trait + alternate backends (premature; one impl is enough for now)

## 11. Reference

- Umbrella heartbit-ghost spec §2.1 (corpus): `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- P1.2a spec (just merged): `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`
- CLI scaffolding (P1.0): `crates/heartbit-cli/src/persona.rs`
- Foundation: `docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md`
