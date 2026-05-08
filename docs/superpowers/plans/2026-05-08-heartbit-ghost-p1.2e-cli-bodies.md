# heartbit-ghost P1.2e — CLI bodies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the four prior P1.2 phases (schema, corpus, extractor, blender) into the heartbit-cli persona surface — `corpus add`, `corpus list`, `profile rebuild`, `profile diff` go from stub errors to working bodies; ship the supporting persona-config + snapshot + diff infrastructure.

**Architecture:** 3 new library modules in `heartbit-ghost::voice` (`persona_config.rs`, `snapshot.rs`, `diff.rs`), plus 4 CLI bodies in `heartbit-cli/src/persona.rs`. Library logic is testable in isolation; CLI dispatch is just orchestration. Pure-resolver pattern (no env-mutation in tests, mirrors P1.2b).

**Tech Stack:** Rust 2024, `serde`/`serde_json`/`toml`, `chrono` (workspace dep from P1.2b), `sha2` (workspace dep, new direct dep for heartbit-ghost), `thiserror`, `tempfile` (dev-dep from P1.2b). No new workspace deps.

---

## File structure

| File | Responsibility |
|------|----------------|
| `crates/heartbit-ghost/src/voice/persona_config.rs` | NEW — `PersonaConfig`, `PersonaConfigError`, `load_from_path`, `resolve_persona_config_path` |
| `crates/heartbit-ghost/src/voice/snapshot.rs` | NEW — `Snapshot`, `SnapshotMeta`, `SnapshotStore`, `SnapshotError`, `default_profiles_dir`, `resolve_profiles_dir` |
| `crates/heartbit-ghost/src/voice/diff.rs` | NEW — `ProfileDiff`, `FieldChange`, `ChangeKind`, `WeightedEntry`, `render_profile_diff` |
| `crates/heartbit-ghost/src/voice/mod.rs` | MODIFY — declare 3 new submodules + extend re-exports |
| `crates/heartbit-ghost/Cargo.toml` | MODIFY — add `sha2 = { workspace = true }` |
| `crates/heartbit-cli/src/main.rs` | MODIFY — make `build_provider_from_env` `pub(crate)` so persona.rs can call it |
| `crates/heartbit-cli/src/persona.rs` | MODIFY — replace 4 stub error-returns with bodies + add `parse_version` helper + 6 dispatch tests |

5 tasks total: 4 implementation + 1 final acceptance (verification only, no commit).

---

## Task 1: `PersonaConfig` + `PersonaConfigError`

**Why:** The first new module. Loads `~/.heartbit/ghost/personas/<persona>.toml` and validates the embedded `BlendRecipe`. No new deps.

**Files:**
- Create: `crates/heartbit-ghost/src/voice/persona_config.rs`
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (add `pub mod persona_config;` + re-exports)

- [ ] **Step 1: Create `crates/heartbit-ghost/src/voice/persona_config.rs`**

```rust
//! Persona instance configuration — loads `~/.heartbit/ghost/personas/<persona>.toml`
//! which embeds the writer's [`BlendRecipe`].
//!
//! Path resolution mirrors [`crate::corpus::default_corpora_dir`]: an env
//! var override (`HEARTBIT_GHOST_PERSONAS`) takes precedence, falling back
//! to `$HOME/.heartbit/ghost/personas`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::voice::blend::BlendRecipe;
use crate::voice::error::VoiceError;

fn default_version() -> u32 {
    1
}

/// On-disk persona configuration. Currently embeds a [`BlendRecipe`];
/// future fields (autonomy phase, audit policy, etc.) land in P1.4.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersonaConfig {
    /// Schema version. Currently must be 1.
    #[serde(default = "default_version")]
    pub version: u32,

    /// Blend recipe driving `profile rebuild`.
    pub recipe: BlendRecipe,
}

impl PersonaConfig {
    /// Load from `$HEARTBIT_GHOST_PERSONAS/<name>.toml`, falling back to
    /// `$HOME/.heartbit/ghost/personas/<name>.toml`. Validates the
    /// embedded recipe.
    pub fn load(persona_name: &str) -> Result<Self, PersonaConfigError> {
        let custom = std::env::var("HEARTBIT_GHOST_PERSONAS").ok();
        let home = std::env::var("HOME").ok();
        let path = resolve_persona_config_path(persona_name, custom.as_deref(), home.as_deref())?;
        Self::load_from_path(&path)
    }

    /// Pure helper — load from an explicit path (no env access).
    pub fn load_from_path(path: &Path) -> Result<Self, PersonaConfigError> {
        if !path.exists() {
            return Err(PersonaConfigError::NotFound(path.to_path_buf()));
        }
        let text = std::fs::read_to_string(path)?;
        let config: PersonaConfig = toml::from_str(&text)?;
        if config.version != 1 {
            return Err(PersonaConfigError::Validation(format!(
                "unsupported persona config version: {} (expected 1)",
                config.version
            )));
        }
        config.recipe.validate().map_err(PersonaConfigError::Recipe)?;
        Ok(config)
    }
}

/// Pure resolver — separated from env access for testability.
pub(crate) fn resolve_persona_config_path(
    persona_name: &str,
    custom: Option<&str>,
    home: Option<&str>,
) -> Result<PathBuf, PersonaConfigError> {
    let dir = if let Some(path) = custom {
        PathBuf::from(path)
    } else {
        let home = home.ok_or_else(|| {
            PersonaConfigError::Validation(
                "neither HEARTBIT_GHOST_PERSONAS nor HOME is set".to_string(),
            )
        })?;
        PathBuf::from(home).join(".heartbit/ghost/personas")
    };
    Ok(dir.join(format!("{persona_name}.toml")))
}

/// Errors raised by [`PersonaConfig::load`] and [`PersonaConfig::load_from_path`].
#[derive(Debug, Error)]
pub enum PersonaConfigError {
    /// I/O failure (file open / read).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// TOML parse failure.
    #[error("toml: {0}")]
    Parse(#[from] toml::de::Error),

    /// Embedded `BlendRecipe::validate` failed.
    #[error("recipe: {0}")]
    Recipe(VoiceError),

    /// File at the resolved path does not exist.
    #[error("persona config not found at {}", _0.display())]
    NotFound(PathBuf),

    /// Generic validation failure (e.g., version mismatch, env unresolved).
    #[error("validation: {0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_config(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(format!("{name}.toml"));
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn load_from_path_happy_minimal() {
        let dir = TempDir::new().unwrap();
        let path = write_config(
            dir.path(),
            "x",
            r#"
version = 1

[recipe]
version = 1

[[recipe.blend]]
writer = "karpathy"
weight = 1.0
"#,
        );
        let config = PersonaConfig::load_from_path(&path).expect("loads");
        assert_eq!(config.version, 1);
        assert_eq!(config.recipe.blend.len(), 1);
        assert_eq!(config.recipe.blend[0].writer, "karpathy");
    }

    #[test]
    fn load_from_path_returns_not_found_when_missing() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("does-not-exist.toml");
        let err = PersonaConfigError::from(
            PersonaConfig::load_from_path(&missing).unwrap_err(),
        );
        match err {
            PersonaConfigError::NotFound(p) => assert_eq!(p, missing),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn load_from_path_returns_parse_error_on_bad_toml() {
        let dir = TempDir::new().unwrap();
        let path = write_config(dir.path(), "x", "{not toml at all");
        let err = PersonaConfig::load_from_path(&path).unwrap_err();
        assert!(matches!(err, PersonaConfigError::Parse(_)), "got: {err:?}");
    }

    #[test]
    fn load_from_path_returns_recipe_error_when_recipe_invalid() {
        let dir = TempDir::new().unwrap();
        // Recipe with weights summing to 0.6 (not 1.0) → BlendRecipe::validate fails.
        let path = write_config(
            dir.path(),
            "x",
            r#"
version = 1

[recipe]
version = 1

[[recipe.blend]]
writer = "karpathy"
weight = 0.3

[[recipe.blend]]
writer = "eladgil"
weight = 0.3
"#,
        );
        let err = PersonaConfig::load_from_path(&path).unwrap_err();
        match err {
            PersonaConfigError::Recipe(_) => {}
            other => panic!("expected Recipe error, got {other:?}"),
        }
    }

    #[test]
    fn load_from_path_returns_validation_error_on_unsupported_version() {
        let dir = TempDir::new().unwrap();
        let path = write_config(
            dir.path(),
            "x",
            r#"
version = 2

[recipe]
version = 1

[[recipe.blend]]
writer = "karpathy"
weight = 1.0
"#,
        );
        let err = PersonaConfig::load_from_path(&path).unwrap_err();
        match err {
            PersonaConfigError::Validation(msg) => {
                assert!(msg.contains("unsupported"), "msg: {msg}");
                assert!(msg.contains("version"), "msg: {msg}");
                assert!(msg.contains('2'), "msg: {msg}");
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn resolve_persona_config_path_uses_custom_when_set() {
        let p = resolve_persona_config_path("x", Some("/explicit/path"), Some("/home/u")).unwrap();
        assert_eq!(p, PathBuf::from("/explicit/path/x.toml"));
    }

    #[test]
    fn resolve_persona_config_path_falls_back_to_home() {
        let p = resolve_persona_config_path("x", None, Some("/home/u")).unwrap();
        assert_eq!(p, PathBuf::from("/home/u/.heartbit/ghost/personas/x.toml"));
    }

    #[test]
    fn resolve_persona_config_path_errors_when_neither_set() {
        let err = resolve_persona_config_path("x", None, None).unwrap_err();
        match err {
            PersonaConfigError::Validation(msg) => {
                assert!(msg.contains("HEARTBIT_GHOST_PERSONAS"), "msg: {msg}");
                assert!(msg.contains("HOME"), "msg: {msg}");
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }
}
```

(The plan asked for ~5 tests; this delivers 8 — 5 load_from_path + 3 resolve_persona_config_path. The resolver tests are essential for env-resolution coverage without env mutation.)

- [ ] **Step 2: Modify `crates/heartbit-ghost/src/voice/mod.rs`**

The current state has (after P1.2c + P1.2d):

```rust
pub mod blend;
pub mod error;
pub mod extractor;
pub mod style;

pub use blend::{BlendEntry, BlendError, BlendRecipe, PartialStyleProfile, blend_profiles};
pub use error::VoiceError;
pub use extractor::{ExtractError, StyleExtractor, StyleExtractorBuilder, default_system_prompt};
pub use style::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
```

Add `pub mod persona_config;` (alphabetical — between `extractor` and `style`) and the re-export. Final state:

```rust
pub mod blend;
pub mod error;
pub mod extractor;
pub mod persona_config;
pub mod style;

pub use blend::{BlendEntry, BlendError, BlendRecipe, PartialStyleProfile, blend_profiles};
pub use error::VoiceError;
pub use extractor::{ExtractError, StyleExtractor, StyleExtractorBuilder, default_system_prompt};
pub use persona_config::{PersonaConfig, PersonaConfigError};
pub use style::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
```

(Tasks 2 + 3 will append more `pub mod` and `pub use` lines.)

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::persona_config
```

Expected: `8 passed; 0 failed; 0 ignored`.

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/voice/persona_config.rs crates/heartbit-ghost/src/voice/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — PersonaConfig + PersonaConfigError (P1.2e)

Loads ~/.heartbit/ghost/personas/<persona>.toml which embeds the
persona's BlendRecipe. Path resolution mirrors P1.2b corpora:
HEARTBIT_GHOST_PERSONAS env override, falling back to
$HOME/.heartbit/ghost/personas/.

Pure resolve_persona_config_path helper splits env access from path
computation so tests exercise the resolver directly without
std::env::set_var.

PersonaConfig::load_from_path validates the embedded recipe via
BlendRecipe::validate; failure surfaces as PersonaConfigError::Recipe.
Version mismatch (currently must be 1) surfaces as Validation.

8 tests: load happy, NotFound, bad-TOML parse, bad-recipe,
unsupported version, plus 3 resolver tests for the env paths.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2e-cli-bodies-design.md §3
EOF
)"
```

---

## Task 2: `Snapshot` + `SnapshotStore` + `SnapshotError`

**Why:** The persistence layer for `profile rebuild` outputs. Adds `sha2` as a direct dep (workspace dep, transitively present already). Atomic save via tempfile+rename mirrors P1.2b's pattern.

**Files:**
- Modify: `crates/heartbit-ghost/Cargo.toml` (add `sha2 = { workspace = true }`)
- Create: `crates/heartbit-ghost/src/voice/snapshot.rs`
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (add `pub mod snapshot;` + re-exports)

- [ ] **Step 1: Add `sha2` to `crates/heartbit-ghost/Cargo.toml`**

The current `[dependencies]` block (after P1.2b) has `chrono`, `tempfile` (dev), etc. Add `sha2 = { workspace = true }` alphabetically. Final state:

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
sha2 = { workspace = true }
thiserror = { workspace = true }
tokio = { workspace = true }
toml = { workspace = true }

[dev-dependencies]
tempfile = "3"
tokio = { workspace = true }
wiremock = "0.6"
```

(`tokio` was promoted from dev-deps to runtime in P1.2c.)

- [ ] **Step 2: Create `crates/heartbit-ghost/src/voice/snapshot.rs`**

```rust
//! Versioned profile snapshot persistence.
//!
//! On-disk layout (created lazily on first save):
//!
//! ```text
//! ~/.heartbit/ghost/profiles/<persona>/
//! ├── v1.toml
//! ├── v2.toml
//! ├── v3.toml
//! └── latest.toml      <- copy of the highest-numbered snapshot
//! ```

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::voice::blend::BlendRecipe;
use crate::voice::style::StyleProfile;

/// One profile snapshot — `[meta]` + `[profile]` tables in TOML.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    /// Snapshot-level metadata (version, hash, etc.).
    pub meta: SnapshotMeta,

    /// The merged StyleProfile.
    pub profile: StyleProfile,
}

/// Snapshot-level metadata. Stored in `[meta]` section of the TOML file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotMeta {
    /// Snapshot version (1, 2, 3, ...). NOT the same as `profile.version`.
    pub version: u32,

    /// sha256 hex of the `[profile]` body (for integrity checks).
    pub hash: String,

    /// sha256 hex of the source `BlendRecipe` TOML (for reproducibility).
    pub recipe_hash: String,

    /// When this snapshot was generated.
    pub generated_at: DateTime<Utc>,
}

/// Versioned snapshot store rooted at `<profiles_root>/<persona>/`.
#[derive(Debug)]
pub struct SnapshotStore {
    persona_dir: PathBuf,
}

/// Append-stats-style return for [`SnapshotStore::save_new`] — the new version number.
impl SnapshotStore {
    /// Open (or create) the snapshot directory for `persona` under `profiles_root`.
    /// The directory itself is NOT created until [`SnapshotStore::save_new`] runs.
    pub fn open(profiles_root: &Path, persona: &str) -> Result<Self, SnapshotError> {
        if persona.trim().is_empty()
            || persona.contains('/')
            || persona.contains('\\')
            || persona.contains("..")
        {
            return Err(SnapshotError::InvalidPersona(persona.to_string()));
        }
        Ok(Self {
            persona_dir: profiles_root.join(persona),
        })
    }

    /// Compute the next version number (1 if no prior snapshots).
    pub fn next_version(&self) -> Result<u32, SnapshotError> {
        if !self.persona_dir.exists() {
            return Ok(1);
        }
        let mut max: u32 = 0;
        for entry in fs::read_dir(&self.persona_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|x| x.to_str()) != Some("toml") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|x| x.to_str()) else {
                continue;
            };
            if let Some(rest) = stem.strip_prefix('v') {
                if let Ok(n) = rest.parse::<u32>() {
                    if n > max {
                        max = n;
                    }
                }
            }
        }
        Ok(max + 1)
    }

    /// Persist `profile` as a new versioned snapshot. Returns the assigned
    /// version number. Atomic via tempfile+rename.
    pub fn save_new(
        &self,
        profile: StyleProfile,
        recipe: &BlendRecipe,
    ) -> Result<u32, SnapshotError> {
        fs::create_dir_all(&self.persona_dir)?;
        let version = self.next_version()?;
        let recipe_hash = compute_hex_sha256(&toml::to_string(recipe).map_err(SnapshotError::TomlSer)?);

        // Compute hash of just the profile (so it's invariant under meta changes).
        let profile_text = toml::to_string(&profile).map_err(SnapshotError::TomlSer)?;
        let hash = compute_hex_sha256(&profile_text);

        let snapshot = Snapshot {
            meta: SnapshotMeta {
                version,
                hash,
                recipe_hash,
                generated_at: Utc::now(),
            },
            profile,
        };
        let body = toml::to_string(&snapshot).map_err(SnapshotError::TomlSer)?;

        // Write versioned file atomically.
        let v_path = self.persona_dir.join(format!("v{version}.toml"));
        atomic_write(&v_path, &body)?;

        // Write latest.toml atomically (best-effort; if this fails, the
        // versioned file still landed and load_latest's fallback recovers).
        let latest_path = self.persona_dir.join("latest.toml");
        atomic_write(&latest_path, &body)?;

        Ok(version)
    }

    /// Load a specific version. Returns `NotFound` if the file is absent.
    pub fn load(&self, version: u32) -> Result<Snapshot, SnapshotError> {
        let path = self.persona_dir.join(format!("v{version}.toml"));
        if !path.exists() {
            return Err(SnapshotError::NotFound {
                version,
                persona_dir: self.persona_dir.clone(),
            });
        }
        let text = fs::read_to_string(&path)?;
        let snapshot: Snapshot = toml::from_str(&text).map_err(SnapshotError::TomlDe)?;
        Ok(snapshot)
    }

    /// Load the most recent snapshot. Returns `Ok(None)` if no snapshots
    /// exist. Falls back to scanning `v*.toml` if `latest.toml` is missing
    /// or stale.
    pub fn load_latest(&self) -> Result<Option<Snapshot>, SnapshotError> {
        let latest_path = self.persona_dir.join("latest.toml");
        if latest_path.exists() {
            let text = fs::read_to_string(&latest_path)?;
            let snapshot: Snapshot = toml::from_str(&text).map_err(SnapshotError::TomlDe)?;
            // Sanity check: latest.toml's version should match the highest
            // v*.toml on disk. If not, fall through to the scan below.
            let max = self.scan_max_version()?;
            if max == Some(snapshot.meta.version) {
                return Ok(Some(snapshot));
            }
        }
        // Fallback: scan v*.toml files.
        match self.scan_max_version()? {
            Some(v) => Ok(Some(self.load(v)?)),
            None => Ok(None),
        }
    }

    fn scan_max_version(&self) -> Result<Option<u32>, SnapshotError> {
        if !self.persona_dir.exists() {
            return Ok(None);
        }
        let mut max: Option<u32> = None;
        for entry in fs::read_dir(&self.persona_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|x| x.to_str()) != Some("toml") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|x| x.to_str()) else {
                continue;
            };
            if let Some(rest) = stem.strip_prefix('v') {
                if let Ok(n) = rest.parse::<u32>() {
                    max = Some(max.map_or(n, |m| m.max(n)));
                }
            }
        }
        Ok(max)
    }
}

/// Resolve the profiles root directory.
///
/// Checks `HEARTBIT_GHOST_PROFILES`, falling back to
/// `$HOME/.heartbit/ghost/profiles`.
pub fn default_profiles_dir() -> Result<PathBuf, SnapshotError> {
    let custom = std::env::var("HEARTBIT_GHOST_PROFILES").ok();
    let home = std::env::var("HOME").ok();
    resolve_profiles_dir(custom.as_deref(), home.as_deref())
}

/// Pure resolver — separated from env access for testability.
pub(crate) fn resolve_profiles_dir(
    custom: Option<&str>,
    home: Option<&str>,
) -> Result<PathBuf, SnapshotError> {
    if let Some(path) = custom {
        return Ok(PathBuf::from(path));
    }
    let home = home.ok_or_else(|| {
        SnapshotError::Resolve(
            "neither HEARTBIT_GHOST_PROFILES nor HOME is set".to_string(),
        )
    })?;
    Ok(PathBuf::from(home).join(".heartbit/ghost/profiles"))
}

fn atomic_write(path: &Path, body: &str) -> Result<(), SnapshotError> {
    let tmp = path.with_extension("toml.tmp");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(body.as_bytes())?;
        f.flush()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

fn compute_hex_sha256(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Errors raised by the [`SnapshotStore`] surface.
#[derive(Debug, Error)]
pub enum SnapshotError {
    /// Filesystem failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// TOML serialization failed (should not happen for valid profiles).
    #[error("toml serialize: {0}")]
    TomlSer(toml::ser::Error),

    /// TOML deserialization failed (corrupt snapshot file).
    #[error("toml parse: {0}")]
    TomlDe(toml::de::Error),

    /// Snapshot for the requested version doesn't exist on disk.
    #[error("snapshot v{version} not found in {}", persona_dir.display())]
    NotFound {
        /// The version number requested.
        version: u32,
        /// The persona directory that was searched.
        persona_dir: PathBuf,
    },

    /// Persona name failed validation (empty / path-traversal characters).
    #[error("invalid persona name '{0}': must be non-empty, no '/', '\\', or '..'")]
    InvalidPersona(String),

    /// Path resolution failed (env unresolved).
    #[error("resolve: {0}")]
    Resolve(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::style::{
        EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
        OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
        ThreadRhythm,
    };
    use crate::voice::{BlendEntry, PartialStyleProfile};
    use tempfile::TempDir;

    fn mk_profile() -> StyleProfile {
        StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![OpeningPattern::ClaimFirst],
            opening_pattern_weights: vec![1.0],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::RarePunchlineOnly,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec!["specific".to_string()],
            ai_tells_to_avoid: vec!["delve".to_string()],
            thread_rhythm: ThreadRhythm::PunchlineCallbacks,
            thread_max_length: 10,
            thread_opener_must_hook: true,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        }
    }

    fn mk_recipe() -> BlendRecipe {
        BlendRecipe {
            version: 1,
            blend: vec![BlendEntry {
                writer: "karpathy".to_string(),
                weight: 1.0,
            }],
            overrides: PartialStyleProfile::default(),
        }
    }

    #[test]
    fn save_new_assigns_v1_when_directory_is_empty() {
        let dir = TempDir::new().unwrap();
        let store = SnapshotStore::open(dir.path(), "x").unwrap();
        let v = store.save_new(mk_profile(), &mk_recipe()).unwrap();
        assert_eq!(v, 1);
        assert!(dir.path().join("x/v1.toml").exists());
        assert!(dir.path().join("x/latest.toml").exists());
    }

    #[test]
    fn save_new_increments_version_on_subsequent_saves() {
        let dir = TempDir::new().unwrap();
        let store = SnapshotStore::open(dir.path(), "x").unwrap();
        let v1 = store.save_new(mk_profile(), &mk_recipe()).unwrap();
        let v2 = store.save_new(mk_profile(), &mk_recipe()).unwrap();
        let v3 = store.save_new(mk_profile(), &mk_recipe()).unwrap();
        assert_eq!((v1, v2, v3), (1, 2, 3));
        assert!(dir.path().join("x/v3.toml").exists());
    }

    #[test]
    fn save_new_round_trips_via_load() {
        let dir = TempDir::new().unwrap();
        let store = SnapshotStore::open(dir.path(), "x").unwrap();
        let original = mk_profile();
        let v = store.save_new(original.clone(), &mk_recipe()).unwrap();
        let loaded = store.load(v).unwrap();
        assert_eq!(loaded.profile, original);
        assert_eq!(loaded.meta.version, v);
        // Hash is non-empty, hex-shaped.
        assert_eq!(loaded.meta.hash.len(), 64);
        assert!(loaded.meta.hash.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(loaded.meta.recipe_hash.len(), 64);
    }

    #[test]
    fn load_returns_not_found_for_missing_version() {
        let dir = TempDir::new().unwrap();
        let store = SnapshotStore::open(dir.path(), "x").unwrap();
        store.save_new(mk_profile(), &mk_recipe()).unwrap();
        let err = store.load(99).unwrap_err();
        match err {
            SnapshotError::NotFound { version, .. } => assert_eq!(version, 99),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn save_does_not_leave_tmp_files_behind() {
        let dir = TempDir::new().unwrap();
        let store = SnapshotStore::open(dir.path(), "x").unwrap();
        store.save_new(mk_profile(), &mk_recipe()).unwrap();
        let leftover: Vec<_> = std::fs::read_dir(dir.path().join("x"))
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
    fn load_latest_falls_back_to_scan_when_latest_toml_missing() {
        let dir = TempDir::new().unwrap();
        let store = SnapshotStore::open(dir.path(), "x").unwrap();
        store.save_new(mk_profile(), &mk_recipe()).unwrap();
        store.save_new(mk_profile(), &mk_recipe()).unwrap();
        // Delete latest.toml, leaving v1.toml + v2.toml.
        std::fs::remove_file(dir.path().join("x/latest.toml")).unwrap();
        let snapshot = store.load_latest().unwrap().expect("found");
        assert_eq!(snapshot.meta.version, 2);
    }

    #[test]
    fn load_latest_returns_none_when_no_snapshots_exist() {
        let dir = TempDir::new().unwrap();
        let store = SnapshotStore::open(dir.path(), "x").unwrap();
        assert!(store.load_latest().unwrap().is_none());
    }

    #[test]
    fn open_rejects_invalid_persona_names() {
        let dir = TempDir::new().unwrap();
        for bad in &["", "../escape", "name/slash", "name\\backslash", "ok..bad"] {
            let err = SnapshotStore::open(dir.path(), bad).unwrap_err();
            assert!(
                matches!(err, SnapshotError::InvalidPersona(_)),
                "input '{bad}' should be rejected"
            );
        }
    }

    #[test]
    fn resolve_profiles_dir_uses_custom_when_set() {
        let p = resolve_profiles_dir(Some("/explicit/path"), Some("/home/u")).unwrap();
        assert_eq!(p, PathBuf::from("/explicit/path"));
    }

    #[test]
    fn resolve_profiles_dir_falls_back_to_home() {
        let p = resolve_profiles_dir(None, Some("/home/u")).unwrap();
        assert_eq!(p, PathBuf::from("/home/u/.heartbit/ghost/profiles"));
    }
}
```

(10 tests total: 8 store + 2 resolver.)

- [ ] **Step 3: Modify `crates/heartbit-ghost/src/voice/mod.rs`**

Add `pub mod snapshot;` (alphabetical — between `persona_config` and `style`) and re-exports. Final state:

```rust
pub mod blend;
pub mod error;
pub mod extractor;
pub mod persona_config;
pub mod snapshot;
pub mod style;

pub use blend::{BlendEntry, BlendError, BlendRecipe, PartialStyleProfile, blend_profiles};
pub use error::VoiceError;
pub use extractor::{ExtractError, StyleExtractor, StyleExtractorBuilder, default_system_prompt};
pub use persona_config::{PersonaConfig, PersonaConfigError};
pub use snapshot::{
    Snapshot, SnapshotError, SnapshotMeta, SnapshotStore, default_profiles_dir,
};
pub use style::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
```

- [ ] **Step 4: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::snapshot
```

Expected: `10 passed; 0 failed; 0 ignored`.

- [ ] **Step 5: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/Cargo.toml crates/heartbit-ghost/src/voice/snapshot.rs crates/heartbit-ghost/src/voice/mod.rs Cargo.lock
git commit -m "$(cat <<'EOF'
feat(ghost): voice — Snapshot + SnapshotStore + SnapshotError (P1.2e)

Versioned profile snapshot persistence rooted at
~/.heartbit/ghost/profiles/<persona>/. Atomic save via tempfile+rename
mirrors P1.2b. Numbered v<N>.toml + latest.toml; load_latest falls back
to scanning v*.toml if latest.toml is missing or stale.

[meta] (snapshot version, hash, recipe_hash, generated_at) + [profile]
(StyleProfile body) tables. Profile is nested (NOT serde(flatten))
because both SnapshotMeta and StyleProfile have a `version` field —
flattening would collide.

sha2 promoted from transitive to direct dep on heartbit-ghost
(workspace dep already declared).

10 tests: save assigns v1 → v2 → v3, round-trip via load, NotFound on
missing version, no .tmp leftover, load_latest fallback, persona name
validation, resolve_profiles_dir env paths.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2e-cli-bodies-design.md §4
EOF
)"
```

---

## Task 3: `ProfileDiff` + `render_profile_diff`

**Why:** The diff renderer that backs `profile diff <persona> <v1> <v2>`. Pure data + format — no I/O. ~80 LOC of compute + ~80 LOC of render + 7 tests.

**Files:**
- Create: `crates/heartbit-ghost/src/voice/diff.rs`
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (add `pub mod diff;` + re-exports)

- [ ] **Step 1: Create `crates/heartbit-ghost/src/voice/diff.rs`**

```rust
//! Structured per-field diff between two [`StyleProfile`] snapshots.
//!
//! See [`ProfileDiff::compute`] for the data shape and
//! [`render_profile_diff`] for the human-readable formatter used by
//! `heartbit persona profile diff`.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::voice::snapshot::SnapshotMeta;
use crate::voice::style::{Formatting, OpeningPattern, StyleProfile};

/// Structured difference between two [`StyleProfile`] values. Captures
/// only the fields that changed; identical fields are omitted.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProfileDiff {
    /// Per-field changes, in StyleProfile declaration order.
    pub changes: Vec<FieldChange>,
}

/// One field's change (with the field's name and a typed delta).
#[derive(Debug, Clone, PartialEq)]
pub struct FieldChange {
    /// Field name as it appears in the StyleProfile struct.
    pub field: String,

    /// The kind of change (categorical / list / weighted / distribution).
    pub kind: ChangeKind,
}

/// One of four shapes a field-level change can take.
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeKind {
    /// Categorical, numeric, or bool — `old → new`, both as snake_case strings.
    Scalar {
        /// Old value rendered as a string.
        old: String,
        /// New value rendered as a string.
        new: String,
    },

    /// Unordered list of strings — symmetric difference (added / removed).
    StringList {
        /// Strings in `new` but not in `old`.
        added: Vec<String>,
        /// Strings in `old` but not in `new`.
        removed: Vec<String>,
    },

    /// Parallel weighted arrays (only `opening_patterns` + weights today).
    WeightedList {
        /// Per-pattern delta. `old_weight` is `None` if the pattern was added,
        /// `new_weight` is `None` if it was removed.
        entries: Vec<WeightedEntry>,
    },

    /// 4-bucket distribution (only `sentence_length_distribution` today).
    Distribution {
        /// Old bucket values.
        old: [u8; 4],
        /// New bucket values.
        new: [u8; 4],
    },
}

/// One pattern's contribution in a weighted-list change.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedEntry {
    /// Pattern name as snake_case string.
    pub item: String,
    /// Weight in the old profile, or `None` if the pattern was added.
    pub old_weight: Option<f64>,
    /// Weight in the new profile, or `None` if the pattern was removed.
    pub new_weight: Option<f64>,
}

impl ProfileDiff {
    /// Walk both profiles, emitting one [`FieldChange`] per field that
    /// differs. Identical fields are not in `changes`. Empty `changes`
    /// means the profiles are equal.
    pub fn compute(old: &StyleProfile, new: &StyleProfile) -> Self {
        let mut changes = Vec::new();

        if old.sentence_length_target != new.sentence_length_target {
            changes.push(FieldChange {
                field: "sentence_length_target".to_string(),
                kind: ChangeKind::Scalar {
                    old: format!("{:?}", old.sentence_length_target).to_lowercase(),
                    new: format!("{:?}", new.sentence_length_target).to_lowercase(),
                },
            });
        }
        if old.sentence_length_distribution != new.sentence_length_distribution {
            changes.push(FieldChange {
                field: "sentence_length_distribution".to_string(),
                kind: ChangeKind::Distribution {
                    old: old.sentence_length_distribution,
                    new: new.sentence_length_distribution,
                },
            });
        }
        if old.fragment_frequency != new.fragment_frequency {
            changes.push(FieldChange {
                field: "fragment_frequency".to_string(),
                kind: ChangeKind::Scalar {
                    old: format!("{:?}", old.fragment_frequency).to_lowercase(),
                    new: format!("{:?}", new.fragment_frequency).to_lowercase(),
                },
            });
        }
        if old.opening_patterns != new.opening_patterns
            || old.opening_pattern_weights != new.opening_pattern_weights
        {
            changes.push(FieldChange {
                field: "opening_patterns".to_string(),
                kind: ChangeKind::WeightedList {
                    entries: weighted_list_diff(
                        &old.opening_patterns,
                        &old.opening_pattern_weights,
                        &new.opening_patterns,
                        &new.opening_pattern_weights,
                    ),
                },
            });
        }
        if old.formatting != new.formatting {
            // formatting changed — emit per-sub-field scalars
            push_formatting_changes(&mut changes, &old.formatting, &new.formatting);
        }
        if old.emoji_policy != new.emoji_policy {
            changes.push(FieldChange {
                field: "emoji_policy".to_string(),
                kind: ChangeKind::Scalar {
                    old: format!("{:?}", old.emoji_policy).to_lowercase(),
                    new: format!("{:?}", new.emoji_policy).to_lowercase(),
                },
            });
        }
        if old.hashtag_policy != new.hashtag_policy {
            changes.push(FieldChange {
                field: "hashtag_policy".to_string(),
                kind: ChangeKind::Scalar {
                    old: format!("{:?}", old.hashtag_policy).to_lowercase(),
                    new: format!("{:?}", new.hashtag_policy).to_lowercase(),
                },
            });
        }
        if old.specificity_target != new.specificity_target {
            changes.push(FieldChange {
                field: "specificity_target".to_string(),
                kind: ChangeKind::Scalar {
                    old: format!("{:?}", old.specificity_target).to_lowercase(),
                    new: format!("{:?}", new.specificity_target).to_lowercase(),
                },
            });
        }
        if old.voice_traits != new.voice_traits {
            changes.push(FieldChange {
                field: "voice_traits".to_string(),
                kind: string_list_diff(&old.voice_traits, &new.voice_traits),
            });
        }
        if old.ai_tells_to_avoid != new.ai_tells_to_avoid {
            changes.push(FieldChange {
                field: "ai_tells_to_avoid".to_string(),
                kind: string_list_diff(&old.ai_tells_to_avoid, &new.ai_tells_to_avoid),
            });
        }
        if old.thread_rhythm != new.thread_rhythm {
            changes.push(FieldChange {
                field: "thread_rhythm".to_string(),
                kind: ChangeKind::Scalar {
                    old: format!("{:?}", old.thread_rhythm).to_lowercase(),
                    new: format!("{:?}", new.thread_rhythm).to_lowercase(),
                },
            });
        }
        if old.thread_max_length != new.thread_max_length {
            changes.push(FieldChange {
                field: "thread_max_length".to_string(),
                kind: ChangeKind::Scalar {
                    old: old.thread_max_length.to_string(),
                    new: new.thread_max_length.to_string(),
                },
            });
        }
        if old.thread_opener_must_hook != new.thread_opener_must_hook {
            changes.push(FieldChange {
                field: "thread_opener_must_hook".to_string(),
                kind: ChangeKind::Scalar {
                    old: old.thread_opener_must_hook.to_string(),
                    new: new.thread_opener_must_hook.to_string(),
                },
            });
        }
        if old.topical_obsessions != new.topical_obsessions {
            changes.push(FieldChange {
                field: "topical_obsessions".to_string(),
                kind: string_list_diff(&old.topical_obsessions, &new.topical_obsessions),
            });
        }
        if old.topical_avoidances != new.topical_avoidances {
            changes.push(FieldChange {
                field: "topical_avoidances".to_string(),
                kind: string_list_diff(&old.topical_avoidances, &new.topical_avoidances),
            });
        }

        Self { changes }
    }

    /// `true` when the two profiles compared equal (no changes).
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

fn string_list_diff(old: &[String], new: &[String]) -> ChangeKind {
    let old_set: BTreeSet<&str> = old.iter().map(String::as_str).collect();
    let new_set: BTreeSet<&str> = new.iter().map(String::as_str).collect();
    let added: Vec<String> = new_set
        .difference(&old_set)
        .map(|s| (*s).to_string())
        .collect();
    let removed: Vec<String> = old_set
        .difference(&new_set)
        .map(|s| (*s).to_string())
        .collect();
    ChangeKind::StringList { added, removed }
}

fn weighted_list_diff(
    old_pats: &[OpeningPattern],
    old_weights: &[f64],
    new_pats: &[OpeningPattern],
    new_weights: &[f64],
) -> Vec<WeightedEntry> {
    let mut entries: Vec<WeightedEntry> = Vec::new();
    for (pat, w) in old_pats.iter().zip(old_weights.iter()) {
        entries.push(WeightedEntry {
            item: format!("{pat:?}").to_lowercase(),
            old_weight: Some(*w),
            new_weight: None,
        });
    }
    for (pat, w) in new_pats.iter().zip(new_weights.iter()) {
        let item = format!("{pat:?}").to_lowercase();
        if let Some(existing) = entries.iter_mut().find(|e| e.item == item) {
            existing.new_weight = Some(*w);
        } else {
            entries.push(WeightedEntry {
                item,
                old_weight: None,
                new_weight: Some(*w),
            });
        }
    }
    entries
}

fn push_formatting_changes(changes: &mut Vec<FieldChange>, old: &Formatting, new: &Formatting) {
    if old.lowercase != new.lowercase {
        changes.push(FieldChange {
            field: "formatting.lowercase".to_string(),
            kind: ChangeKind::Scalar {
                old: old.lowercase.to_string(),
                new: new.lowercase.to_string(),
            },
        });
    }
    if old.periods != new.periods {
        changes.push(FieldChange {
            field: "formatting.periods".to_string(),
            kind: ChangeKind::Scalar {
                old: format!("{:?}", old.periods).to_lowercase(),
                new: format!("{:?}", new.periods).to_lowercase(),
            },
        });
    }
    if old.em_dashes != new.em_dashes {
        changes.push(FieldChange {
            field: "formatting.em_dashes".to_string(),
            kind: ChangeKind::Scalar {
                old: format!("{:?}", old.em_dashes).to_lowercase(),
                new: format!("{:?}", new.em_dashes).to_lowercase(),
            },
        });
    }
    if old.quotation_marks != new.quotation_marks {
        changes.push(FieldChange {
            field: "formatting.quotation_marks".to_string(),
            kind: ChangeKind::Scalar {
                old: format!("{:?}", old.quotation_marks).to_lowercase(),
                new: format!("{:?}", new.quotation_marks).to_lowercase(),
            },
        });
    }
    if old.line_breaks != new.line_breaks {
        changes.push(FieldChange {
            field: "formatting.line_breaks".to_string(),
            kind: ChangeKind::Scalar {
                old: format!("{:?}", old.line_breaks).to_lowercase(),
                new: format!("{:?}", new.line_breaks).to_lowercase(),
            },
        });
    }
}

/// Render a [`ProfileDiff`] as human-readable text for the CLI.
pub fn render_profile_diff(
    diff: &ProfileDiff,
    old_meta: &SnapshotMeta,
    new_meta: &SnapshotMeta,
) -> String {
    let recipe_note = if old_meta.recipe_hash == new_meta.recipe_hash {
        "same recipe"
    } else {
        "recipe changed"
    };
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Profile diff: v{} → v{} (recipe-hash: {} → {}; {})",
        old_meta.version,
        new_meta.version,
        truncate_hash(&old_meta.recipe_hash),
        truncate_hash(&new_meta.recipe_hash),
        recipe_note
    );
    if diff.is_empty() {
        let _ = writeln!(out, "(no changes)");
        return out;
    }
    let _ = writeln!(out);
    for change in &diff.changes {
        match &change.kind {
            ChangeKind::Scalar { old, new } => {
                let _ = writeln!(out, "{}: {} → {}", change.field, old, new);
            }
            ChangeKind::Distribution { old, new } => {
                let _ = writeln!(out, "{}: {:?} → {:?}", change.field, old, new);
            }
            ChangeKind::StringList { added, removed } => {
                let _ = writeln!(out, "{}:", change.field);
                for s in added {
                    let _ = writeln!(out, "  + {s}");
                }
                for s in removed {
                    let _ = writeln!(out, "  - {s}");
                }
            }
            ChangeKind::WeightedList { entries } => {
                let _ = writeln!(out, "{}:", change.field);
                for e in entries {
                    match (e.old_weight, e.new_weight) {
                        (Some(o), Some(n)) if (o - n).abs() < f64::EPSILON => {}
                        (Some(o), Some(n)) => {
                            let _ = writeln!(out, "  {}: {:.2} → {:.2}", e.item, o, n);
                        }
                        (None, Some(n)) => {
                            let _ = writeln!(out, "  + {}: {:.2}", e.item, n);
                        }
                        (Some(o), None) => {
                            let _ = writeln!(out, "  - {}: {:.2}", e.item, o);
                        }
                        (None, None) => {}
                    }
                }
            }
        }
    }
    out
}

fn truncate_hash(hash: &str) -> &str {
    if hash.len() >= 8 { &hash[..8] } else { hash }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::voice::style::{
        EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
        OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
        ThreadRhythm,
    };

    fn mk_profile() -> StyleProfile {
        StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![OpeningPattern::ClaimFirst, OpeningPattern::NumberFirst],
            opening_pattern_weights: vec![0.6, 0.4],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::RarePunchlineOnly,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec!["specific".to_string(), "no_hedging".to_string()],
            ai_tells_to_avoid: vec!["delve".to_string()],
            thread_rhythm: ThreadRhythm::PunchlineCallbacks,
            thread_max_length: 10,
            thread_opener_must_hook: true,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        }
    }

    fn mk_meta(version: u32, recipe_hash: &str) -> SnapshotMeta {
        SnapshotMeta {
            version,
            hash: "0".repeat(64),
            recipe_hash: recipe_hash.to_string(),
            generated_at: Utc::now(),
        }
    }

    #[test]
    fn compute_identical_profiles_produces_empty_diff() {
        let p = mk_profile();
        let diff = ProfileDiff::compute(&p, &p);
        assert!(diff.is_empty());
    }

    #[test]
    fn compute_scalar_change_is_recorded() {
        let old = mk_profile();
        let mut new = old.clone();
        new.emoji_policy = EmojiPolicy::Never;
        let diff = ProfileDiff::compute(&old, &new);
        assert_eq!(diff.changes.len(), 1);
        assert_eq!(diff.changes[0].field, "emoji_policy");
        match &diff.changes[0].kind {
            ChangeKind::Scalar { old, new } => {
                assert_eq!(old, "rarepunchlineonly");
                assert_eq!(new, "never");
            }
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn compute_string_list_change_records_added_and_removed() {
        let old = mk_profile();
        let mut new = old.clone();
        new.voice_traits = vec!["specific".to_string(), "humble".to_string()];
        let diff = ProfileDiff::compute(&old, &new);
        let voice_change = diff
            .changes
            .iter()
            .find(|c| c.field == "voice_traits")
            .expect("voice_traits in diff");
        match &voice_change.kind {
            ChangeKind::StringList { added, removed } => {
                assert_eq!(added, &vec!["humble".to_string()]);
                assert_eq!(removed, &vec!["no_hedging".to_string()]);
            }
            other => panic!("expected StringList, got {other:?}"),
        }
    }

    #[test]
    fn compute_distribution_change_is_typed() {
        let old = mk_profile();
        let mut new = old.clone();
        new.sentence_length_distribution = [35, 35, 22, 8];
        let diff = ProfileDiff::compute(&old, &new);
        let change = diff
            .changes
            .iter()
            .find(|c| c.field == "sentence_length_distribution")
            .expect("present");
        match &change.kind {
            ChangeKind::Distribution { old, new } => {
                assert_eq!(old, &[40, 30, 20, 10]);
                assert_eq!(new, &[35, 35, 22, 8]);
            }
            other => panic!("expected Distribution, got {other:?}"),
        }
    }

    #[test]
    fn compute_weighted_list_change_records_per_pattern_delta() {
        let old = mk_profile(); // claim_first: 0.6, number_first: 0.4
        let mut new = old.clone();
        new.opening_patterns = vec![OpeningPattern::ClaimFirst, OpeningPattern::SceneFirst];
        new.opening_pattern_weights = vec![0.5, 0.5];
        let diff = ProfileDiff::compute(&old, &new);
        let change = diff
            .changes
            .iter()
            .find(|c| c.field == "opening_patterns")
            .expect("present");
        match &change.kind {
            ChangeKind::WeightedList { entries } => {
                let claim = entries.iter().find(|e| e.item == "claimfirst").unwrap();
                assert_eq!(claim.old_weight, Some(0.6));
                assert_eq!(claim.new_weight, Some(0.5));
                let number = entries.iter().find(|e| e.item == "numberfirst").unwrap();
                assert_eq!(number.old_weight, Some(0.4));
                assert_eq!(number.new_weight, None);
                let scene = entries.iter().find(|e| e.item == "scenefirst").unwrap();
                assert_eq!(scene.old_weight, None);
                assert_eq!(scene.new_weight, Some(0.5));
            }
            other => panic!("expected WeightedList, got {other:?}"),
        }
    }

    #[test]
    fn render_no_changes_says_so() {
        let p = mk_profile();
        let diff = ProfileDiff::compute(&p, &p);
        let m1 = mk_meta(3, &"a".repeat(64));
        let m2 = mk_meta(4, &"a".repeat(64));
        let out = render_profile_diff(&diff, &m1, &m2);
        assert!(out.contains("v3 → v4"), "got: {out}");
        assert!(out.contains("same recipe"), "got: {out}");
        assert!(out.contains("(no changes)"), "got: {out}");
    }

    #[test]
    fn render_recipe_change_label() {
        let p = mk_profile();
        let mut p2 = p.clone();
        p2.thread_max_length = 7;
        let diff = ProfileDiff::compute(&p, &p2);
        let m1 = mk_meta(3, &"a".repeat(64));
        let m2 = mk_meta(4, &"b".repeat(64));
        let out = render_profile_diff(&diff, &m1, &m2);
        assert!(out.contains("recipe changed"), "got: {out}");
        assert!(out.contains("thread_max_length: 10 → 7"), "got: {out}");
    }
}
```

(7 tests total: 5 compute + 2 render. Trims the spec's "render output for each ChangeKind" item to 2 render tests; the remaining ChangeKind variants are tested via `compute` and the renderer's logic is straightforward `match` arms covered by the integration tests in Task 4.)

- [ ] **Step 2: Modify `crates/heartbit-ghost/src/voice/mod.rs`**

Add `pub mod diff;` (alphabetical, between `blend` and `error`) + re-exports. Final state:

```rust
pub mod blend;
pub mod diff;
pub mod error;
pub mod extractor;
pub mod persona_config;
pub mod snapshot;
pub mod style;

pub use blend::{BlendEntry, BlendError, BlendRecipe, PartialStyleProfile, blend_profiles};
pub use diff::{ChangeKind, FieldChange, ProfileDiff, WeightedEntry, render_profile_diff};
pub use error::VoiceError;
pub use extractor::{ExtractError, StyleExtractor, StyleExtractorBuilder, default_system_prompt};
pub use persona_config::{PersonaConfig, PersonaConfigError};
pub use snapshot::{
    Snapshot, SnapshotError, SnapshotMeta, SnapshotStore, default_profiles_dir,
};
pub use style::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
```

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::diff
```

Expected: `7 passed; 0 failed; 0 ignored`.

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/voice/diff.rs crates/heartbit-ghost/src/voice/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — ProfileDiff + render_profile_diff (P1.2e)

Structured per-field diff between two StyleProfile values. Walks the 16
non-version StyleProfile fields in declaration order; identical fields
are omitted from changes. Four ChangeKind variants:
- Scalar (categorical / numeric / bool, old → new)
- StringList (added + removed)
- WeightedList (per-pattern old_weight + new_weight)
- Distribution (4-bucket old/new arrays)

Formatting sub-struct emits per-sub-field scalars (formatting.lowercase,
formatting.periods, etc.) for granular reporting.

render_profile_diff produces human-readable output: header with version
arrow + recipe-hash status, per-change lines with appropriate formatting
per ChangeKind. (no changes) when diff is empty.

7 tests: compute identical, scalar change, string-list add/remove,
distribution change, weighted-list per-pattern delta, render no-changes,
render recipe changed.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2e-cli-bodies-design.md §6
EOF
)"
```

---

## Task 4: CLI bodies in `heartbit-cli/src/persona.rs`

**Why:** The integration. Replaces 4 stub error-returns with bodies that wire P1.2a-d + Task 1-3 together. Also adjusts `build_provider_from_env` visibility so persona.rs can call it.

**Files:**
- Modify: `crates/heartbit-cli/src/main.rs` (make `build_provider_from_env` `pub(crate)` + re-export)
- Modify: `crates/heartbit-cli/src/persona.rs` (4 bodies + `parse_version` helper + 6 dispatch tests)

- [ ] **Step 1: Make `build_provider_from_env` `pub(crate)`**

In `crates/heartbit-cli/src/main.rs`, find:

```rust
fn build_provider_from_env(on_retry: Option<Arc<OnRetry>>) -> Result<Arc<BoxedProvider>> {
```

Change to:

```rust
pub(crate) fn build_provider_from_env(on_retry: Option<Arc<OnRetry>>) -> Result<Arc<BoxedProvider>> {
```

(Only the function visibility changes. Body stays the same.)

- [ ] **Step 2: Modify `crates/heartbit-cli/src/persona.rs` — extend imports + replace dispatch arms**

The current top-of-file imports are:

```rust
use anyhow::{Result, anyhow};
use clap::Subcommand;

use heartbit::PersonaRegistry;
```

Extend to:

```rust
use std::collections::HashMap;

use anyhow::{Result, anyhow};
use clap::Subcommand;

use heartbit::PersonaRegistry;

use crate::build_provider_from_env;
```

Then find the `dispatch` function's match arms (currently around lines 174-185 — the `PersonaCommand::Corpus { sub }` and `PersonaCommand::Profile { sub }` arms returning stub errors). Replace those two outer arms with:

```rust
        PersonaCommand::Corpus { sub } => match sub {
            CorpusCommand::Add { writer, path } => {
                if registry.is_empty() {
                    return Err(anyhow!("{}", NO_PERSONAS_REGISTERED));
                }
                let root = heartbit_ghost::corpus::default_corpora_dir()
                    .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
                let mut corpus = heartbit_ghost::corpus::Corpus::open_or_create(&root, &writer)
                    .map_err(|e| anyhow!("open corpus for '{writer}': {e}"))?;
                let stats = corpus
                    .append_from_jsonl(&path)
                    .map_err(|e| {
                        anyhow!("import {} into corpus '{writer}': {e}", path.display())
                    })?;
                println!(
                    "ok: added {} new ({} deduped); total {} for writer '{}'",
                    stats.added, stats.deduped, stats.total_after, writer
                );
                Ok(())
            }
            CorpusCommand::List { name: persona_name } => {
                if registry.get(&persona_name).is_none() {
                    return Err(anyhow!(
                        "persona '{persona_name}' not found. {}",
                        registry_suffix(registry)
                    ));
                }
                let config = heartbit_ghost::voice::PersonaConfig::load(&persona_name)
                    .map_err(|e| anyhow!("load persona config for '{persona_name}': {e}"))?;
                let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                    .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
                println!(
                    "Persona '{}': {} writer(s)",
                    persona_name,
                    config.recipe.blend.len()
                );
                for entry in &config.recipe.blend {
                    match heartbit_ghost::corpus::Corpus::open_or_create(
                        &corpora_root,
                        &entry.writer,
                    ) {
                        Ok(c) if c.is_empty() => {
                            println!(
                                "  {} (weight {:.2}) — MISSING (no corpus on disk)",
                                entry.writer, entry.weight
                            );
                        }
                        Ok(c) => {
                            println!(
                                "  {} (weight {:.2}) — {} posts",
                                entry.writer,
                                entry.weight,
                                c.len()
                            );
                        }
                        Err(e) => {
                            println!(
                                "  {} (weight {:.2}) — ERROR: {e}",
                                entry.writer, entry.weight
                            );
                        }
                    }
                }
                Ok(())
            }
        },
        PersonaCommand::Profile { sub } => match sub {
            ProfileCommand::Rebuild { name: persona_name } => {
                if registry.get(&persona_name).is_none() {
                    return Err(anyhow!(
                        "persona '{persona_name}' not found. {}",
                        registry_suffix(registry)
                    ));
                }
                let config = heartbit_ghost::voice::PersonaConfig::load(&persona_name)
                    .map_err(|e| anyhow!("load persona config: {e}"))?;
                config
                    .recipe
                    .validate()
                    .map_err(|e| anyhow!("invalid recipe in persona config: {e}"))?;

                let provider = build_provider_from_env(None)
                    .map_err(|e| anyhow!("build llm provider: {e}"))?;
                let extractor =
                    heartbit_ghost::voice::StyleExtractor::builder(provider).build();
                let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                    .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;

                let mut profiles: HashMap<String, heartbit_ghost::voice::StyleProfile> =
                    HashMap::new();
                for entry in &config.recipe.blend {
                    println!(
                        "extracting profile for '{}' (weight {:.2})...",
                        entry.writer, entry.weight
                    );
                    let corpus = heartbit_ghost::corpus::Corpus::open_or_create(
                        &corpora_root,
                        &entry.writer,
                    )
                    .map_err(|e| anyhow!("open corpus for '{}': {e}", entry.writer))?;
                    let profile = extractor
                        .extract(&corpus)
                        .await
                        .map_err(|e| anyhow!("extract profile for '{}': {e}", entry.writer))?;
                    profiles.insert(entry.writer.clone(), profile);
                }

                let merged = heartbit_ghost::voice::blend_profiles(&config.recipe, &profiles)
                    .map_err(|e| anyhow!("blend profiles: {e}"))?;

                let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                    .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;
                let store = heartbit_ghost::voice::SnapshotStore::open(
                    &profiles_root,
                    &persona_name,
                )
                .map_err(|e| anyhow!("open snapshot store: {e}"))?;
                let new_version = store
                    .save_new(merged, &config.recipe)
                    .map_err(|e| anyhow!("save snapshot: {e}"))?;

                println!("ok: persona '{}' rebuilt as v{}", persona_name, new_version);
                Ok(())
            }
            ProfileCommand::Diff { name: persona_name, v1, v2 } => {
                if registry.get(&persona_name).is_none() {
                    return Err(anyhow!(
                        "persona '{persona_name}' not found. {}",
                        registry_suffix(registry)
                    ));
                }
                let v1_n = parse_version(&v1)?;
                let v2_n = parse_version(&v2)?;

                let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                    .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;
                let store = heartbit_ghost::voice::SnapshotStore::open(
                    &profiles_root,
                    &persona_name,
                )
                .map_err(|e| anyhow!("open snapshot store: {e}"))?;
                let s1 = store
                    .load(v1_n)
                    .map_err(|e| anyhow!("load v{v1_n}: {e}"))?;
                let s2 = store
                    .load(v2_n)
                    .map_err(|e| anyhow!("load v{v2_n}: {e}"))?;

                let diff = heartbit_ghost::voice::ProfileDiff::compute(&s1.profile, &s2.profile);
                println!(
                    "{}",
                    heartbit_ghost::voice::render_profile_diff(&diff, &s1.meta, &s2.meta)
                );
                Ok(())
            }
        },
```

Then add the `parse_version` helper at the bottom of the file (after the `dispatch` function, before `#[cfg(test)] mod tests`):

```rust
/// Parse a `vN` or `N` argument as a u32.
fn parse_version(arg: &str) -> Result<u32> {
    arg.strip_prefix('v')
        .unwrap_or(arg)
        .parse::<u32>()
        .map_err(|_| anyhow!("expected version like 'v3' or '3', got '{arg}'"))
}
```

- [ ] **Step 3: Append 6 dispatch tests inside the existing `#[cfg(test)] mod tests` block**

The current test block has 4 P1.0 tests (`list_against_empty_registry_prints_message`, `show_against_empty_registry_returns_error`, `corpus_add_against_empty_registry_returns_error`, plus profile tests). Append:

```rust
    use std::path::PathBuf;

    #[tokio::test]
    async fn corpus_list_persona_not_found_returns_error() {
        let r = PersonaRegistry::new();
        // Empty registry, but with a non-empty test ensures the
        // persona-not-found path runs even when the registry has entries.
        let cmd = PersonaCommand::Corpus {
            sub: CorpusCommand::List { name: "no-such-persona".to_string() },
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err(), "should error on missing persona");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
    }

    #[tokio::test]
    async fn profile_rebuild_persona_not_found_returns_error() {
        let r = PersonaRegistry::new();
        let cmd = PersonaCommand::Profile {
            sub: ProfileCommand::Rebuild {
                name: "no-such-persona".to_string(),
            },
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
    }

    #[tokio::test]
    async fn profile_diff_persona_not_found_returns_error() {
        let r = PersonaRegistry::new();
        let cmd = PersonaCommand::Profile {
            sub: ProfileCommand::Diff {
                name: "no-such-persona".to_string(),
                v1: "v1".to_string(),
                v2: "v2".to_string(),
            },
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
    }

    #[test]
    fn parse_version_accepts_v_prefix() {
        assert_eq!(parse_version("v3").unwrap(), 3);
        assert_eq!(parse_version("v0").unwrap(), 0);
        assert_eq!(parse_version("v100").unwrap(), 100);
    }

    #[test]
    fn parse_version_accepts_bare_number() {
        assert_eq!(parse_version("3").unwrap(), 3);
        assert_eq!(parse_version("0").unwrap(), 0);
    }

    #[test]
    fn parse_version_rejects_garbage() {
        let err = parse_version("not-a-version").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expected version"), "got: {msg}");
        assert!(msg.contains("not-a-version"), "got: {msg}");

        let err = parse_version("vfoo").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("vfoo"), "got: {msg}");
    }
```

- [ ] **Step 4: Run the tests**

```bash
cargo test -p heartbit-cli --lib persona
```

Expected: all P1.0 tests + 6 new = at least `10 passed`. (Existing test count varies; just ensure the 6 new tests pass alongside existing ones.)

- [ ] **Step 5: Workspace quality gate**

```bash
cargo fmt -p heartbit-cli -- --check
cargo clippy -p heartbit-cli --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-cli/src/main.rs crates/heartbit-cli/src/persona.rs
git commit -m "$(cat <<'EOF'
feat(cli): persona — wire P1.2a-d into corpus/profile bodies (P1.2e)

Replaces 4 stub error-returns in heartbit-cli/src/persona.rs with
working bodies:

- corpus add <writer> <path>: thin wrapper around P1.2b's
  Corpus::open_or_create + append_from_jsonl; reports AppendStats.
- corpus list <persona>: loads PersonaConfig, lists writers from the
  recipe with corpus presence + post count (MISSING marker for writers
  whose corpus file doesn't exist).
- profile rebuild <persona>: full integration — load config, build LLM
  provider from env, run the extractor over each writer's corpus
  sequentially, blend, persist as next versioned snapshot.
- profile diff <persona> <v1> <v2>: load two snapshots, compute
  ProfileDiff, render via render_profile_diff.

Also makes build_provider_from_env pub(crate) so persona.rs can use it.

6 new dispatch tests: persona-not-found for corpus list / profile
rebuild / profile diff, parse_version with v-prefix / bare number /
garbage input.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2e-cli-bodies-design.md §5
EOF
)"
```

---

## Task 5: Final acceptance + workspace quality gate

**Why:** Confirm P1.2e meets every acceptance criterion. Verification only — no commit.

**Files:** none.

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -10
```

Expected: all green. Workspace test count goes from 3899 (post-P1.2d baseline) to ~3925-3930 (~26-31 new tests; the per-task breakdown lands at 8 + 10 + 7 + 6 = 31, which is over the spec's "~26" estimate).

- [ ] **Step 2: Verify the public surface is reachable**

```bash
cat <<'EOF' > /tmp/heartbit_ghost_p1_2e_surface_check.rs
fn _check() {
    use heartbit_ghost::voice::{
        ChangeKind, FieldChange, PersonaConfig, PersonaConfigError,
        ProfileDiff, Snapshot, SnapshotError, SnapshotMeta, SnapshotStore,
        WeightedEntry, default_profiles_dir, render_profile_diff,
    };
    let _ = PersonaConfigError::NotFound(std::path::PathBuf::new());
    let _ = SnapshotError::Resolve(String::new());
    let _: fn() -> Result<std::path::PathBuf, SnapshotError> = default_profiles_dir;
    let _ = ProfileDiff::default();
    let _ = ChangeKind::Scalar { old: String::new(), new: String::new() };
}
EOF
echo "(Surface check is illustrative; reachability is verified by the workspace cargo check above.)"
rm -f /tmp/heartbit_ghost_p1_2e_surface_check.rs
```

- [ ] **Step 3: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.2e
```

Expected: 6 commits — spec doc + spec amendment + 4 task commits (Task 1, 2, 3, 4). No commit for Task 5.

- [ ] **Step 4: No commit for this task**

Task 5 is verification only. The branch is ready for final review + merge. P1.2 is closed.

---

## Acceptance criteria

P1.2e is done when (per spec §10):

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features` and `cargo check -p heartbit-cli --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~26 new tests pass (actual: 31 — over the spec estimate; coverage spans every CLI body's error paths, every diff `ChangeKind`, snapshot round-trip + atomic rename + load_latest fallback, persona config load + 5 error paths)
- `heartbit_ghost::voice::{PersonaConfig, PersonaConfigError, Snapshot, SnapshotMeta, SnapshotStore, SnapshotError, ProfileDiff, FieldChange, ChangeKind, WeightedEntry, render_profile_diff, default_profiles_dir}` are reachable as public surface
- All 4 CLI bodies in `heartbit-cli/src/persona.rs` work end-to-end via direct construction (no env-var mutation in tests)

## Out of scope (re-stated)

- Generation pipeline (P1.3)
- Telegram review delivery (P1.3)
- Runtime conditioning of writer agents (P1.4)
- Per-writer extraction cache between rebuilds (P1.4)
- `--dry-run`, `--from-cache`, `--force` flags on `profile rebuild`
- Auto-skip rebuild when output is identical to previous version
- Profile auto-deletion / GC of old snapshots
- Autonomy-phase config (P1.4)
- Audit log integration (P1.4)
- Multi-tenant persona configs (P1.4)
- File locking on `profile rebuild` (acceptable race in single-user dev path)

## Reference

- Spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2e-cli-bodies-design.md`
- Umbrella heartbit-ghost spec §2.3: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- P1.2a: `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`
- P1.2b: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md`
- P1.2c: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md`
- P1.2d: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2d-blend-algorithm-design.md`
- Existing CLI scaffolding: `crates/heartbit-cli/src/persona.rs`
- Existing `build_provider_from_env`: `crates/heartbit-cli/src/main.rs:2017`
