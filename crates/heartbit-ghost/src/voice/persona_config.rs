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
        config
            .recipe
            .validate()
            .map_err(PersonaConfigError::Recipe)?;
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
        let err = PersonaConfig::load_from_path(&missing).unwrap_err();
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
