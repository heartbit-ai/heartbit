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
            if let Some(rest) = stem.strip_prefix('v')
                && let Ok(n) = rest.parse::<u32>()
                && n > max
            {
                max = n;
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
        let recipe_hash =
            compute_hex_sha256(&toml::to_string(recipe).map_err(SnapshotError::TomlSer)?);

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
            if let Some(rest) = stem.strip_prefix('v')
                && let Ok(n) = rest.parse::<u32>()
            {
                max = Some(max.map_or(n, |m| m.max(n)));
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
        SnapshotError::Resolve("neither HEARTBIT_GHOST_PROFILES nor HOME is set".to_string())
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
