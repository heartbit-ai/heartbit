//! Persona-related value types. Most variants are intentionally empty in
//! Phase 0: concrete persona crates extend them as they land.

use crate::config::AgentConfig;
use crate::config::OrchestratorConfig;
use crate::tool::Tool;
use std::sync::Arc;

/// Per-instance parameters supplied at expansion time. Constructed from the
/// `[[persona]]` block in `HeartbitConfig`.
#[derive(Debug, Clone)]
pub struct PersonaParams {
    /// Persona-specific overrides (free-form TOML; interpreted by `expand()`).
    pub overrides: toml::Value,
    /// Glob for environment-variable credential lookup, e.g. `"X_*"`.
    pub credentials_env: Option<String>,
    /// Authorship mode for posts/communications produced by the persona.
    pub authorship_mode: AuthorshipMode,
}

impl Default for PersonaParams {
    fn default() -> Self {
        Self {
            // `toml::Value` doesn't implement `Default`; an empty table is
            // the natural neutral element for "no overrides".
            overrides: toml::Value::Table(toml::map::Map::new()),
            credentials_env: None,
            authorship_mode: AuthorshipMode::default(),
        }
    }
}

/// What a persona expands into at startup.
#[derive(Default)]
pub struct PersonaExpansion {
    /// Sub-agents the persona requires.
    pub agents: Vec<AgentConfig>,
    /// Orchestrator config for the persona's pipeline.
    pub orchestrator: OrchestratorConfig,
    /// Tool instances contributed by the persona.
    pub tools: Vec<Arc<dyn Tool>>,
    /// Trigger specs (cron / sensors / mention polling / manual). Empty in Phase 0.
    pub triggers: Vec<TriggerSpec>,
    /// Optional review channel spec. None in Phase 0.
    pub review: Option<ReviewSpec>,
}

impl std::fmt::Debug for PersonaExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaExpansion")
            .field("agents", &self.agents.len())
            .field("tools", &self.tools.len())
            .field("triggers", &self.triggers.len())
            .field("review", &self.review.is_some())
            .finish()
    }
}

/// Authorship mode declared per persona instance.
///
/// Operators are responsible for ensuring the chosen mode is permitted under
/// the target platform's terms of service and applicable regulation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorshipMode {
    /// A human approves every action before it goes out.
    #[default]
    HumanAssisted,
    /// Fully autonomous; the agent's bot status is publicly disclosed
    /// (account labelled, ToS-compliant).
    AutonomousDisclosed,
    /// Fully autonomous; the agent's bot status is not disclosed. Operator
    /// owns regulatory compliance.
    AutonomousUndisclosed,
}

/// Trigger spec — concrete variants land with their consumers (Phase 1).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TriggerSpec {}

/// Review channel spec — concrete variants land with their consumers (Phase 1).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ReviewSpec {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorship_mode_default_is_human_assisted() {
        assert_eq!(AuthorshipMode::default(), AuthorshipMode::HumanAssisted);
    }

    #[test]
    fn authorship_mode_serde_round_trip() {
        let mode = AuthorshipMode::AutonomousUndisclosed;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"autonomous_undisclosed\"");
        let parsed: AuthorshipMode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, mode);
    }

    #[test]
    fn persona_params_default_authorship_is_human_assisted() {
        let p = PersonaParams::default();
        assert_eq!(p.authorship_mode, AuthorshipMode::HumanAssisted);
    }

    #[test]
    fn persona_expansion_default_is_empty() {
        let e = PersonaExpansion::default();
        assert!(e.agents.is_empty());
        assert!(e.tools.is_empty());
        assert!(e.triggers.is_empty());
        assert!(e.review.is_none());
    }
}
