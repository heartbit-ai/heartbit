# heartbit-ghost P1.2e — CLI bodies design

**Status:** approved 2026-05-08
**Branch:** `feat/heartbit-ghost-p1.2e`
**Predecessors:** P1.2a (style profile schema), P1.2b (corpus storage), P1.2c (LLM extractor), P1.2d (blend algorithm). All merged to `main`.
**Successor:** P1.3 (generation pipeline + Telegram review). P1.2e closes out P1.2.

## 1. Goal

Wire the four prior P1.2 phases together via the existing `heartbit-cli` persona surface. Replace the four stub error-returns in `crates/heartbit-cli/src/persona.rs` (`corpus add`, `corpus list`, `profile rebuild`, `profile diff`) with working bodies, and ship the supporting infrastructure for persona config + profile snapshot persistence + profile diffing.

After P1.2e ships, the user can:

```bash
heartbit persona corpus add karpathy ~/Downloads/karpathy.jsonl
heartbit persona corpus list x
heartbit persona profile rebuild x      # produces v1, v2, v3 ... snapshots
heartbit persona profile diff x v3 v4
```

Out of scope: generation pipeline (P1.3), runtime conditioning of writer agents (P1.3 / P1.4), Telegram integration (P1.3), per-writer extraction cache, dry-run / from-cache flags on `profile rebuild`, profile auto-deletion on no-change, autonomy-phase config, audit log integration.

## 2. Architecture

P1.2e adds **3 new library modules** in `heartbit-ghost::voice` and **wires 4 CLI bodies** in `heartbit-cli/src/persona.rs`. Library-vs-CLI split: pure logic (load config, persist snapshot, compute diff) lives in heartbit-ghost where it can be unit-tested in isolation; CLI dispatch is just orchestration + user-facing error formatting.

```
crates/heartbit-ghost/src/voice/         (new files added to the existing module)
├── persona_config.rs    NEW — PersonaConfig + PersonaConfigError
├── snapshot.rs          NEW — Snapshot + SnapshotMeta + SnapshotStore + SnapshotError
└── diff.rs              NEW — ProfileDiff + FieldChange + ChangeKind + render_profile_diff

crates/heartbit-cli/src/
└── persona.rs           MODIFY — replace 4 stub error-returns with bodies
```

**No new workspace dependencies.** `chrono` and `tempfile` from P1.2b. `sha2` is already pulled in transitively. The diff renderer is hand-rolled (no `similar` crate — structured per-field, not text-diff).

**Composition for `profile rebuild`** (the heaviest body):

```
PersonaConfig::load(persona_name)?                         // P1.2e new
  → BlendRecipe + version
build_provider_from_env(None)?                             // existing CLI helper
let extractor = StyleExtractor::builder(provider).build(); // P1.2c
for entry in recipe.blend {
    Corpus::open_or_create(corpora_dir, &entry.writer)?    // P1.2b
    extractor.extract(&corpus).await?                       // P1.2c
    profiles.insert(...);
}
let merged = blend_profiles(&recipe, &profiles)?;           // P1.2d
SnapshotStore::open(profiles_dir, persona)?
    .save_new(merged, &recipe)?                             // P1.2e new
```

## 3. Persona config

**Path resolution** (`voice/persona_config.rs`):

1. `$HEARTBIT_GHOST_PERSONAS/<persona>.toml` if env var set
2. `$HOME/.heartbit/ghost/personas/<persona>.toml` otherwise
3. `PersonaConfigError::NotFound(path)` if absent

**On-disk shape**:

```toml
# ~/.heartbit/ghost/personas/x.toml
version = 1

[recipe]
version = 1

[[recipe.blend]]
writer = "karpathy"
weight = 0.30

[[recipe.blend]]
writer = "eladgil"
weight = 0.20

# ... up to 10 writers per BlendRecipe::validate()

[recipe.overrides]
thread_max_length = 7
# (any subset of PartialStyleProfile fields)
```

**Rust types**:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersonaConfig {
    #[serde(default = "default_version")]
    pub version: u32,
    pub recipe: BlendRecipe,
}

impl PersonaConfig {
    pub fn load(persona_name: &str) -> Result<Self, PersonaConfigError>;
    pub fn load_from_path(path: &Path) -> Result<Self, PersonaConfigError>;
}

#[derive(Debug, thiserror::Error)]
pub enum PersonaConfigError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("recipe: {0}")]
    Recipe(VoiceError),
    #[error("persona config not found at {}", _0.display())]
    NotFound(PathBuf),
    #[error("validation: {0}")]
    Validation(String),
}
```

`PersonaConfig::version: u32` is forward-compat (in case the file format gains non-recipe fields like autonomy phase, audit policy — those are P1.4). Currently must equal 1; otherwise `Validation`.

`load_from_path` is the pure helper used by tests (no env access). `load(persona_name)` resolves the path then delegates.

## 4. Snapshot file format + storage

**Path layout** (`voice/snapshot.rs`):

```
~/.heartbit/ghost/profiles/<persona>/
├── v1.toml
├── v2.toml
├── v3.toml
└── latest.toml      ← copy of the highest-numbered snapshot
```

Env override: `HEARTBIT_GHOST_PROFILES`. Falls back to `$HOME/.heartbit/ghost/profiles`.

**File format**:

```toml
# v3.toml (or latest.toml — same content)
[meta]
version = 3                          # snapshot version, NOT StyleProfile version
hash = "f3a8c1...abc12345"           # sha256 of the [profile] body for integrity
recipe_hash = "9b2d44...def67890"    # sha256 of the recipe TOML for reproducibility
generated_at = "2026-05-08T18:42:00Z"

[profile]
version = 1                          # StyleProfile schema version (see P1.2a)
sentence_length_target = "short"
sentence_length_distribution = [40, 30, 20, 10]
fragment_frequency = "common"
opening_patterns = ["claim_first", "number_first"]
opening_pattern_weights = [0.6, 0.4]

[profile.formatting]
lowercase = true
periods = "optional"
em_dashes = "forbidden"
quotation_marks = "double"
line_breaks = "single"

# ... rest of StyleProfile fields under [profile] / [profile.formatting] ...
```

The profile is nested under `[profile]` (NOT `#[serde(flatten)]`) to avoid a key collision: both `SnapshotMeta` and `StyleProfile` declare a `version` field, so flattening would emit two `version =` keys at the same level — invalid TOML, ambiguous parse. Nesting under `[profile]` makes the two version fields cleanly distinguishable as `meta.version` (snapshot) and `profile.version` (StyleProfile schema).

**Rust types**:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub meta: SnapshotMeta,
    pub profile: StyleProfile,    // nested as [profile] in TOML (see §4 note)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotMeta {
    pub version: u32,           // snapshot version (1, 2, 3, ...)
    pub hash: String,           // sha256 hex of body
    pub recipe_hash: String,    // sha256 hex of source recipe TOML
    pub generated_at: DateTime<Utc>,
}

pub struct SnapshotStore {
    persona_dir: PathBuf,
}

impl SnapshotStore {
    pub fn open(profiles_root: &Path, persona: &str) -> Result<Self, SnapshotError>;
    pub fn save_new(&self, profile: StyleProfile, recipe: &BlendRecipe)
        -> Result<u32, SnapshotError>;
    pub fn load(&self, version: u32) -> Result<Snapshot, SnapshotError>;
    pub fn load_latest(&self) -> Result<Option<Snapshot>, SnapshotError>;
    pub fn next_version(&self) -> Result<u32, SnapshotError>;
}

pub fn default_profiles_dir() -> Result<PathBuf, SnapshotError>;
```

**No `#[serde(flatten)]`** on `Snapshot.profile` — see §4 note. The profile is nested under `[profile]` to avoid the `version`-field collision with `SnapshotMeta`.

**Atomic save (`save_new`)**:

1. Compute `version = self.next_version()?` (1 if no prior snapshots)
2. Compute `recipe_hash = sha256(toml::to_string(recipe)?)`
3. Build `Snapshot { meta: SnapshotMeta { version, hash: PLACEHOLDER, recipe_hash, generated_at: now() }, profile }`
4. Serialize body separately (StyleProfile only) to compute `hash = sha256(body)`
5. Replace `PLACEHOLDER` with the actual hash; serialize the full snapshot TOML
6. Write to `<persona_dir>/v<N>.toml.tmp`, then `fs::rename` to `v<N>.toml`
7. Write the same content to `latest.toml.tmp`, then `fs::rename` to `latest.toml`
8. Return `version`

If step 7 fails after step 6 succeeds, `v<N>.toml` exists but `latest.toml` is stale. `load_latest` is defensive: if `latest.toml` is missing or its `[meta].version` doesn't match the highest `v*.toml` on disk, fall back to scanning `v*.toml` files and returning the highest.

**`next_version`**: scans `<persona_dir>/v*.toml`, parses the version from the filename (stripping `v` prefix and `.toml` suffix), returns `max + 1`. If no files exist, returns 1.

**Hash computation** uses sha2 (already in workspace via heartbit-core's transitive deps). 64-char lowercase hex.

## 5. CLI bodies

All four bodies live in `crates/heartbit-cli/src/persona.rs`'s `dispatch` function, replacing the existing stub error-returns. The existing "registry not empty" guard is preserved (sanity check that heartbit-ghost is linked in).

### 5.1 `corpus add <writer> <path>`

Persona-agnostic. Thin wrapper around P1.2b.

```rust
PersonaCommand::Corpus { sub: CorpusCommand::Add { writer, path } } => {
    if registry.is_empty() {
        return Err(anyhow!("{}", NO_PERSONAS_REGISTERED));
    }
    let root = heartbit_ghost::corpus::default_corpora_dir()
        .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
    let mut corpus = heartbit_ghost::corpus::Corpus::open_or_create(&root, &writer)
        .map_err(|e| anyhow!("open corpus for '{writer}': {e}"))?;
    let stats = corpus.append_from_jsonl(&path)
        .map_err(|e| anyhow!("import {} into corpus '{writer}': {e}", path.display()))?;
    println!(
        "ok: added {} new ({} deduped); total {} for writer '{}'",
        stats.added, stats.deduped, stats.total_after, writer
    );
    Ok(())
}
```

### 5.2 `corpus list <persona>`

Loads the persona config, enumerates writers from the recipe with corpus presence + post count.

```rust
PersonaCommand::Corpus { sub: CorpusCommand::List { name: persona_name } } => {
    if registry.get(&persona_name).is_none() {
        return Err(anyhow!(
            "persona '{persona_name}' not found. {}", registry_suffix(registry)
        ));
    }
    let config = heartbit_ghost::voice::PersonaConfig::load(&persona_name)
        .map_err(|e| anyhow!("load persona config for '{persona_name}': {e}"))?;
    let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
        .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
    println!("Persona '{}': {} writer(s)", persona_name, config.recipe.blend.len());
    for entry in &config.recipe.blend {
        match heartbit_ghost::corpus::Corpus::open_or_create(&corpora_root, &entry.writer) {
            Ok(c) if c.is_empty() => {
                println!("  {} (weight {:.2}) — MISSING (no corpus on disk)",
                    entry.writer, entry.weight);
            }
            Ok(c) => {
                println!("  {} (weight {:.2}) — {} posts",
                    entry.writer, entry.weight, c.len());
            }
            Err(e) => {
                println!("  {} (weight {:.2}) — ERROR: {e}", entry.writer, entry.weight);
            }
        }
    }
    Ok(())
}
```

### 5.3 `profile rebuild <persona>`

The heaviest body. Wires Corpus → Extractor → Blender → SnapshotStore.

```rust
PersonaCommand::Profile { sub: ProfileCommand::Rebuild { name: persona_name } } => {
    if registry.get(&persona_name).is_none() {
        return Err(anyhow!(
            "persona '{persona_name}' not found. {}", registry_suffix(registry)
        ));
    }
    let config = heartbit_ghost::voice::PersonaConfig::load(&persona_name)
        .map_err(|e| anyhow!("load persona config: {e}"))?;
    config.recipe.validate()
        .map_err(|e| anyhow!("invalid recipe in persona config: {e}"))?;

    let provider = build_provider_from_env(None)
        .map_err(|e| anyhow!("build llm provider: {e}"))?;
    let extractor = heartbit_ghost::voice::StyleExtractor::builder(provider).build();
    let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
        .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;

    let mut profiles: HashMap<String, heartbit_ghost::voice::StyleProfile> = HashMap::new();
    for entry in &config.recipe.blend {
        println!("extracting profile for '{}' (weight {:.2})...",
            entry.writer, entry.weight);
        let corpus = heartbit_ghost::corpus::Corpus::open_or_create(
            &corpora_root, &entry.writer
        ).map_err(|e| anyhow!("open corpus for '{}': {e}", entry.writer))?;
        let profile = extractor.extract(&corpus).await
            .map_err(|e| anyhow!("extract profile for '{}': {e}", entry.writer))?;
        profiles.insert(entry.writer.clone(), profile);
    }

    let merged = heartbit_ghost::voice::blend_profiles(&config.recipe, &profiles)
        .map_err(|e| anyhow!("blend profiles: {e}"))?;

    let profiles_root = heartbit_ghost::voice::default_profiles_dir()
        .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;
    let store = heartbit_ghost::voice::SnapshotStore::open(&profiles_root, &persona_name)
        .map_err(|e| anyhow!("open snapshot store: {e}"))?;
    let new_version = store.save_new(merged, &config.recipe)
        .map_err(|e| anyhow!("save snapshot: {e}"))?;

    println!("ok: persona '{}' rebuilt as v{}", persona_name, new_version);
    Ok(())
}
```

**Behavior:**
- **Sequential extraction** — one writer at a time. Per-writer extraction is one LLM call (cheap) but rate-limited; concurrent calls hit limits faster.
- **Abort on first writer-extraction failure** — no partial snapshot is written.
- **No per-writer cache between rebuilds** — every invocation re-extracts every writer. Cache is a P1.4 concern.
- **Identical merged profile produces a NEW v<N>.toml** — no content-addressing dedup. User can see "v3 == v4" via `profile diff` and decide.

### 5.4 `profile diff <persona> <v1> <v2>`

Loads two snapshots, computes a structured diff, renders.

```rust
PersonaCommand::Profile { sub: ProfileCommand::Diff { name: persona_name, v1, v2 } } => {
    if registry.get(&persona_name).is_none() {
        return Err(anyhow!(
            "persona '{persona_name}' not found. {}", registry_suffix(registry)
        ));
    }
    let v1_n = parse_version(&v1)?;
    let v2_n = parse_version(&v2)?;

    let profiles_root = heartbit_ghost::voice::default_profiles_dir()
        .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;
    let store = heartbit_ghost::voice::SnapshotStore::open(&profiles_root, &persona_name)
        .map_err(|e| anyhow!("open snapshot store: {e}"))?;
    let s1 = store.load(v1_n).map_err(|e| anyhow!("load v{v1_n}: {e}"))?;
    let s2 = store.load(v2_n).map_err(|e| anyhow!("load v{v2_n}: {e}"))?;

    let diff = heartbit_ghost::voice::ProfileDiff::compute(&s1.profile, &s2.profile);
    println!("{}", heartbit_ghost::voice::render_profile_diff(&diff, &s1.meta, &s2.meta));
    Ok(())
}

fn parse_version(arg: &str) -> Result<u32> {
    arg.strip_prefix('v').unwrap_or(arg).parse::<u32>()
        .map_err(|_| anyhow!("expected version like 'v3' or '3', got '{arg}'"))
}
```

## 6. Diff renderer

Structured per-field diff, not text diff.

```rust
// voice/diff.rs

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProfileDiff {
    pub changes: Vec<FieldChange>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldChange {
    pub field: String,
    pub kind: ChangeKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChangeKind {
    /// Categorical / numeric / bool — old → new, both as snake_case strings.
    Scalar { old: String, new: String },

    /// List-of-strings — added + removed (relative to old).
    StringList { added: Vec<String>, removed: Vec<String> },

    /// opening_patterns + opening_pattern_weights — per-pattern delta.
    WeightedList { entries: Vec<WeightedEntry> },

    /// sentence_length_distribution — 4 buckets with old → new each.
    Distribution { old: [u8; 4], new: [u8; 4] },
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedEntry {
    pub item: String,
    pub old_weight: Option<f64>,
    pub new_weight: Option<f64>,
}

impl ProfileDiff {
    pub fn compute(old: &StyleProfile, new: &StyleProfile) -> Self;
    pub fn is_empty(&self) -> bool;
}

pub fn render_profile_diff(
    diff: &ProfileDiff,
    old_meta: &SnapshotMeta,
    new_meta: &SnapshotMeta,
) -> String;
```

**Field iteration order**: declaration order of `StyleProfile` fields. Predictable.

**Header behavior** (`render_profile_diff`):
- `recipe_hash` identical between v1 and v2 → "(same recipe)" suffix; diff is purely from extractor non-determinism or corpus changes.
- `recipe_hash` differs → "(recipe changed)" suffix.

**No-change output**:

```
Profile diff: x v3 → v4 — no changes
```

**Sample non-empty output**:

```
Profile diff: x v3 → v4 (recipe-hash: 9b2d44... → 9b2d44...; same recipe)

emoji_policy: rare_punchline_only → never
sentence_length_distribution: [40, 30, 20, 10] → [35, 35, 22, 8]
voice_traits:
  + technical
  + humble
  - aphoristic
opening_patterns:
  claim_first: 0.40 → 0.52
  number_first: 0.20 → 0.36
  + scene_first: 0.12
  - aphoristic_first: 0.20

(11 other fields unchanged)
```

## 7. Error handling

**At the CLI boundary**: every call site uses `anyhow!("<context>: {e}")` to add a user-readable prefix. Library errors carry their structured messages; CLI prepends *which step failed*.

**Missing persona config** (`profile rebuild`, `profile diff`, `corpus list`): `PersonaConfigError::NotFound(path)` carries the resolved path. CLI surfaces with template guidance.

**Missing corpus** (`profile rebuild`): `Corpus::open_or_create` returns empty corpus → `extractor.extract` returns `ExtractError::EmptyCorpus(writer)`. CLI surfaces with the writer handle.

**LLM rate limit / network failure**: `ExtractError::Llm(...)` propagates. User retries.

**Snapshot version conflict** (rare): two concurrent `profile rebuild` calls might race on `next_version() == 4`. Atomic `fs::rename` makes the second write either overwrite or fail. Single-user dev path; file locking deferred to P1.4.

**`profile diff` for missing version**: `store.load(v_n)` returns `SnapshotError::NotFound { version, persona_dir }`. CLI surfaces clearly.

**`latest.toml` recovery**: if missing or its `[meta].version` doesn't match the highest `v*.toml` on disk, `load_latest` falls back to scanning `v*.toml` files. Defensive.

## 8. Testing

**~26 new tests, all in-tree, all using `tempfile::TempDir` for filesystem isolation.**

| Module | Coverage | Tests |
|--------|----------|-------|
| `persona_config.rs` | load_from_path happy, NotFound, bad TOML, recipe validation failure, version != 1 | 5 |
| `snapshot.rs` | save+load round-trip, atomic rename (no leftover .tmp), version increments, latest.toml updates, hash stability, load_latest fallback when latest.toml is missing, deny_unknown_fields on Snapshot | 8 |
| `diff.rs` | identical profiles → empty, scalar change, list add/remove, weighted-list change, distribution change, render output for each ChangeKind, render no-change | 7 |
| `heartbit-cli/persona.rs` | dispatch error paths: registry-empty (corpus add), persona-not-found (corpus list / profile rebuild / profile diff), parse_version on bad input. Body-level behavior is exercised via the library helpers below; no test mutates env. | 6 |

**Env-mutation strategy**: NO test mutates process env. Mirrors the P1.2b corpora pattern (pure `resolve_corpora_dir(custom, home)` helper that takes env values as arguments).

- `voice/persona_config.rs` exposes `pub(crate) fn resolve_persona_config_path(persona: &str, custom: Option<&str>, home: Option<&str>) -> Result<PathBuf, PersonaConfigError>` — tested directly with explicit args.
- `voice/snapshot.rs` exposes `pub(crate) fn resolve_profiles_dir(custom: Option<&str>, home: Option<&str>) -> Result<PathBuf, SnapshotError>` — tested directly with explicit args.
- The public `PersonaConfig::load(persona_name)` and `default_profiles_dir()` are thin wrappers that read env then call the pure helpers. Their tests assert wiring (the wrapper passes the env values through) by mocking via the pure helper directly — no `std::env::set_var` anywhere.
- CLI dispatch tests use `SnapshotStore::open(root, name)` and `PersonaConfig::load_from_path(path)` directly with `TempDir`-derived paths, bypassing the env layer entirely.

**`profile rebuild` end-to-end coverage** comes from the library helpers (each layer — Corpus, StyleExtractor, blend_profiles, SnapshotStore — is independently tested) plus the spec's "smoke test" acceptance criterion. The CLI dispatch tests cover only error paths since the body is composed of already-tested library calls. If a future feature warrants CLI-level orchestration tests (e.g., loop control flow), that is the time to factor out a `pub(crate) fn rebuild_inner(corpora_root, persona_dir, profiles_root, provider, persona_name)` helper and test it directly — premature for P1.2e.

**Quality gate** (mirrors prior phases):

```bash
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
```

Workspace test count: 3899 → ~3925.

## 9. Architecture decisions (ADs)

**AD-1 — Persona config in user-home**, not repo-relative. Matches the P1.2b corpora convention; per-machine, per-user; not committed to source. Path: `~/.heartbit/ghost/personas/<persona>.toml` with `HEARTBIT_GHOST_PERSONAS` env override.

**AD-2 — Numbered snapshot versions** (`v1`, `v2`, `v3`, ... + `latest.toml`). Auto-incrementing version names map directly to CLI args (`profile diff x v3 v4`); no hash-typing burden. `latest.toml` is a copy (not symlink, for cross-OS portability) of the highest version. Content-addressing was rejected because the CLI ergonomics are terrible for content hashes.

**AD-3 — Snapshot file format with `[meta]` + `[profile]` tables**. Single TOML file per snapshot; readable; parseable as one `Snapshot` struct. Profile is nested under `[profile]` (not `#[serde(flatten)]`) to avoid the `version`-key collision between `SnapshotMeta` and `StyleProfile`. Hash + recipe_hash + generated_at in `[meta]` enables integrity checks and reproducibility tracking.

**AD-4 — `corpus list <persona>` lists writers from the recipe with corpus stats**, not all writers on disk. The most actionable interpretation: shows which writers the persona references AND which have corpus data, signaling "rebuild will fail until you add this writer" without running rebuild.

**AD-5 — Structured per-field diff renderer**, not text diff. The diff understands the schema (knows `opening_patterns` is a weighted set, knows `voice_traits` is order-insensitive) and renders semantically. Avoids noise from key reordering and weight-sort shuffling. Hand-rolled (~80 LOC of renderer); no `similar` crate.

**AD-6 — Abort on first writer-extraction failure** in `profile rebuild`; no partial snapshot is written. No per-writer extraction cache between rebuilds in v0.1. Simpler; per-writer caching with corpus-hash invalidation is a P1.4 optimization.

**AD-7 — Library-vs-CLI split**: `PersonaConfig`, `SnapshotStore`, `ProfileDiff` live in `heartbit-ghost::voice` (testable in isolation); CLI dispatch in `heartbit-cli/src/persona.rs` is just orchestration. Sets up cleaner reuse for P1.3's daemon (which will also need to load persona configs and produce profile snapshots).

**AD-8 — Pure resolver helpers; zero env-mutation in tests**. Mirrors P1.2b's `resolve_corpora_dir(custom, home)` pattern. Both `voice/persona_config.rs` and `voice/snapshot.rs` expose `pub(crate)` resolver functions that take `Option<&str>` for the env values; the env-reading public wrappers (`PersonaConfig::load`, `default_profiles_dir`) call these pure helpers. No test sets `std::env::set_var`, avoiding the parallelism hazard without adding `serial_test` as a new workspace dep.

## 10. Acceptance criteria

P1.2e is done when:

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features` and `cargo check -p heartbit-cli --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~26 new tests pass; coverage spans every CLI body, every diff `ChangeKind`, snapshot round-trip + atomic rename, persona config load + error paths
- `heartbit_ghost::voice::{PersonaConfig, PersonaConfigError, Snapshot, SnapshotMeta, SnapshotStore, SnapshotError, ProfileDiff, FieldChange, ChangeKind, WeightedEntry, render_profile_diff, default_profiles_dir}` are reachable as public surface
- All 4 CLI bodies in `heartbit-cli/src/persona.rs` work end-to-end via direct construction of `SnapshotStore` / `Corpus` / `PersonaConfig` with `TempDir`-backed paths (no env-var mutation in tests)
- A smoke test demonstrates the full pipeline: `corpus add` → `profile rebuild` (with mock LLM) → `profile diff v1 v2`

## 11. Out of scope (re-stated)

- Generation pipeline (P1.3)
- Telegram review delivery (P1.3)
- Runtime conditioning of writer agents (P1.4)
- Per-writer extraction cache between rebuilds (P1.4)
- `--dry-run`, `--from-cache`, `--force` flags on `profile rebuild`
- Auto-skip rebuild when output is identical to previous version
- Profile auto-deletion / GC of old snapshots
- Autonomy-phase config (P1.4)
- Audit log integration (P1.4)
- Multi-tenant persona configs (the daemon's per-tenant persona override layer — P1.4)
- File locking on `profile rebuild` (acceptable race in single-user dev path)
- A migration tool when the snapshot format changes (defer until first breaking change)

## 12. Reference

- Umbrella heartbit-ghost spec §2.3 (blend computation references "personas/x.toml snapshot"): `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- P1.2a spec (StyleProfile + BlendRecipe + PartialStyleProfile): `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`
- P1.2b spec (corpus storage): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md`
- P1.2c spec (LLM extractor): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md`
- P1.2d spec (blend algorithm): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2d-blend-algorithm-design.md`
- Existing CLI scaffolding: `crates/heartbit-cli/src/persona.rs`
- Existing `build_provider_from_env`: `crates/heartbit-cli/src/main.rs:2017`
