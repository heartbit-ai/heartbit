//! Persona-related value types. Most variants are intentionally empty in
//! Phase 0: concrete persona crates extend them as they land.

use crate::config::AgentConfig;
use crate::config::OrchestratorConfig;
use crate::tool::Tool;
use std::sync::Arc;

/// Per-instance parameters supplied at expansion time. Constructed from the
/// `[[persona]]` block in `HeartbitConfig`.
#[derive(Debug, Clone, Default)]
pub struct PersonaParams {
    /// Persona-specific overrides (free-form TOML table; interpreted by
    /// `expand()`). An empty [`toml::Table`] is the neutral element.
    pub overrides: toml::Table,
    /// Glob for environment-variable credential lookup, e.g. `"X_*"`.
    pub credentials_env: Option<String>,
    /// Authorship mode for posts/communications produced by the persona.
    pub authorship_mode: AuthorshipMode,
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
    /// Optional addendum appended to the persona's system prompt at
    /// expansion time. Implementations use this to scope a single
    /// persona to multiple sub-modes (e.g. one persona that posts
    /// generally vs. one that focuses on a specific topic cluster).
    /// `None` for personas without per-mode variation.
    pub mode_addendum: Option<&'static str>,
    /// Optional persona-specific topic context provider for proactive
    /// posting. When present, `handle_persona_post` calls
    /// [`crate::persona::TopicContextProvider::build_context`] before
    /// invoking the topic generator. When absent, the handler injects
    /// only the post history + topic_brief from config. See heartbit-ghost
    /// P1.6 spec §5 for the rationale.
    pub topic_context_provider: Option<std::sync::Arc<dyn crate::persona::TopicContextProvider>>,
}

impl std::fmt::Debug for PersonaExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaExpansion")
            .field("agents", &self.agents.len())
            .field("tools", &self.tools.len())
            .field("triggers", &self.triggers.len())
            .field("review", &self.review.is_some())
            .field("mode_addendum", &self.mode_addendum.is_some())
            .field(
                "topic_context_provider",
                &self.topic_context_provider.is_some(),
            )
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

/// Placeholder for future persona trigger specifications (cron, sensor,
/// mention-polling, manual). Currently has no variants — implementors
/// should not pattern-match on this type. Reserved as a stable
/// type-level seam for trigger taxonomy (Phase 1+).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TriggerSpec {}

/// Placeholder for future persona review-channel specifications.
/// Currently has no variants — implementors should not pattern-match
/// on this type. Reserved as a stable type-level seam for review-flow
/// taxonomy (Phase 1+).
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

    #[test]
    fn persona_expansion_default_mode_addendum_is_none() {
        let e = PersonaExpansion::default();
        assert!(e.mode_addendum.is_none());
    }

    #[test]
    fn persona_expansion_carries_static_mode_addendum() {
        const ADDENDUM: &str = "topic-cluster:rust — test fixture";
        let e = PersonaExpansion {
            mode_addendum: Some(ADDENDUM),
            ..PersonaExpansion::default()
        };
        assert_eq!(e.mode_addendum, Some(ADDENDUM));
    }
}
