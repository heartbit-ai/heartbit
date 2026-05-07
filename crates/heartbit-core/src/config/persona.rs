//! `[[persona]]` config block for declaring persona instances in
//! `heartbit.toml` / `daemon.toml`.
//!
//! Phase 0 scope: Validation is **lexical only** — the `recipe` field is
//! checked syntactically (must be `<crate>:<name>`), but no [`PersonaRegistry`]
//! lookup happens here. The registry is empty in this phase; the resolution
//! step runs at daemon startup once persona crates are loaded.
//!
//! [`PersonaRegistry`]: crate::persona::PersonaRegistry

use serde::Deserialize;

use crate::error::Error;
use crate::persona::AuthorshipMode;

/// Autonomy phase progression for a persona instance.
///
/// Phases drive the routing fraction of candidate posts that go to human
/// review versus auto-publish. A persona typically progresses from
/// `Calibration` (100% review) toward `Autonomous` (10% sampled review)
/// as confidence in its outputs grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonaPhase {
    /// 100% candidates routed to review.
    #[default]
    Calibration,
    /// 80% review / 20% auto-publish (high-confidence).
    Supervised,
    /// 10% review (sampled) / 90% auto-publish.
    Autonomous,
    /// Only flagged candidates routed to review.
    Sentinel,
}

/// One `[[persona]]` block.
///
/// `recipe` is a key of the form `<crate_short>:<recipe>` (e.g.
/// `"heartbit-ghost:x"`) that names a persona recipe registered in the
/// [`PersonaRegistry`]. The lookup is deferred to daemon startup; this
/// struct only enforces the lexical shape.
///
/// [`PersonaRegistry`]: crate::persona::PersonaRegistry
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaConfig {
    /// Local instance name (must be unique within the config file).
    pub name: String,
    /// Recipe key in the form `<crate_short>:<recipe>`, e.g. `"heartbit-ghost:x"`.
    pub recipe: String,
    /// Glob for env-var credential lookup, e.g. `"X_*"`.
    #[serde(default)]
    pub credentials_env: Option<String>,
    /// Authorship mode (default `human_assisted`).
    #[serde(default)]
    pub authorship_mode: AuthorshipMode,
    /// Initial autonomy phase.
    #[serde(default)]
    pub phase: PersonaPhase,
    /// Persona-specific overrides (free-form; interpreted by the recipe's `expand()`).
    #[serde(default = "default_overrides")]
    pub overrides: toml::Value,
}

/// Default value for [`PersonaConfig::overrides`] — an empty TOML table.
///
/// `toml::Value` does not implement [`Default`], so `#[serde(default)]` and
/// the `..Default::default()` struct-literal pattern need an explicit helper.
fn default_overrides() -> toml::Value {
    toml::Value::Table(toml::map::Map::new())
}

impl PersonaConfig {
    /// Lexical validation: recipe key parses, name is non-empty.
    /// Does not consult the registry.
    pub fn validate(&self) -> Result<(), Error> {
        if self.name.trim().is_empty() {
            return Err(Error::Config("persona name must be non-empty".into()));
        }
        if !self.recipe.contains(':') {
            return Err(Error::Config(format!(
                "persona '{}' recipe '{}' must be of the form '<crate>:<name>'",
                self.name, self.recipe
            )));
        }
        let (lhs, rhs) = self.recipe.split_once(':').unwrap();
        if lhs.trim().is_empty() || rhs.trim().is_empty() {
            return Err(Error::Config(format!(
                "persona '{}' recipe '{}' has empty crate or name component",
                self.name, self.recipe
            )));
        }
        Ok(())
    }
}

impl Default for PersonaConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            recipe: String::new(),
            credentials_env: None,
            authorship_mode: AuthorshipMode::default(),
            phase: PersonaPhase::default(),
            overrides: default_overrides(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml_text: &str) -> Result<PersonaConfig, toml::de::Error> {
        toml::from_str::<PersonaConfig>(toml_text)
    }

    #[test]
    fn parses_minimal_persona() {
        let c: PersonaConfig = parse(
            r#"
            name = "x"
            recipe = "heartbit-ghost:x"
            "#,
        )
        .expect("parses");
        assert_eq!(c.name, "x");
        assert_eq!(c.recipe, "heartbit-ghost:x");
        assert_eq!(c.authorship_mode, AuthorshipMode::HumanAssisted);
        assert_eq!(c.phase, PersonaPhase::Calibration);
    }

    #[test]
    fn parses_full_persona() {
        let c: PersonaConfig = parse(
            r#"
            name = "x"
            recipe = "heartbit-ghost:x"
            credentials_env = "X_*"
            authorship_mode = "autonomous_undisclosed"
            phase = "supervised"
            "#,
        )
        .expect("parses");
        assert_eq!(c.credentials_env.as_deref(), Some("X_*"));
        assert_eq!(c.authorship_mode, AuthorshipMode::AutonomousUndisclosed);
        assert_eq!(c.phase, PersonaPhase::Supervised);
    }

    #[test]
    fn validate_rejects_empty_name() {
        let c = PersonaConfig {
            name: "".into(),
            recipe: "heartbit-ghost:x".into(),
            ..Default::default()
        };
        let err = c.validate().unwrap_err();
        assert!(matches!(err, Error::Config(s) if s.contains("non-empty")));
    }

    #[test]
    fn validate_rejects_recipe_without_colon() {
        let c = PersonaConfig {
            name: "x".into(),
            recipe: "heartbit-ghost-x".into(),
            ..Default::default()
        };
        let err = c.validate().unwrap_err();
        assert!(matches!(err, Error::Config(s) if s.contains("'<crate>:<name>'")));
    }

    #[test]
    fn validate_rejects_empty_lhs_or_rhs() {
        for bad in [":x", "heartbit-ghost:", ":"] {
            let c = PersonaConfig {
                name: "x".into(),
                recipe: bad.into(),
                ..Default::default()
            };
            let err = c.validate().unwrap_err();
            assert!(
                matches!(err, Error::Config(_)),
                "expected Config error for recipe {:?}",
                bad
            );
        }
    }

    #[test]
    fn rejects_unknown_phase() {
        let result = parse(
            r#"
            name = "x"
            recipe = "heartbit-ghost:x"
            phase = "ludicrous"
            "#,
        );
        assert!(result.is_err());
    }
}
