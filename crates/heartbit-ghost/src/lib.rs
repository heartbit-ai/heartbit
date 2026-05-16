//! `heartbit-ghost` — best-in-class autonomous X (Twitter) agent persona.
//!
//! P1.3a wires 7 sub-agent recipes (`researcher`, `writer`, `style_critic`,
//! `judge`, `fact_check`, `image_generator`, `publisher`) plus 5 tool
//! instances (`websearch`, `webfetch`, `image_generate`, `twitter_thread`,
//! `twitter_reply`) into `XGhostPersona::expand()`. The pipeline
//! orchestrator that chains these agents lands in P1.3b; the Telegram
//! review channel in P1.3d; trigger specs (cron / mention polling) and
//! audit log integration in P1.4.

#![deny(missing_docs)]

use std::sync::Arc;

use heartbit_core::{Persona, PersonaExpansion, PersonaParams, PersonaRegistry};

pub mod agents;
pub mod corpus;
pub mod heartbit_rs;
pub mod pipeline;
pub mod posts;
pub mod quote;
pub mod reply;
pub mod review;
pub mod tools;
pub mod voice;

/// Stable persona identifier — used as the registry key and as the
/// `recipe = "..."` value in `[[persona]]` config blocks.
pub const PERSONA_NAME: &str = "heartbit-ghost:x";

/// The X (Twitter) ghost persona.
///
/// As of P1.3a, [`XGhostPersona::expand`] returns a [`PersonaExpansion`]
/// with 7 sub-agent recipes and 5 tool instances (see the module-level
/// docs). Pipeline orchestration, Telegram review, triggers, and audit
/// log integration land in P1.3b/d and P1.4.
pub struct XGhostPersona {
    /// Persona version string, derived at compile time from the workspace
    /// `Cargo.toml`.
    version: &'static str,
}

impl XGhostPersona {
    /// Construct a new instance of the persona.
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

impl Default for XGhostPersona {
    fn default() -> Self {
        Self::new()
    }
}

impl Persona for XGhostPersona {
    fn name(&self) -> &str {
        PERSONA_NAME
    }

    fn description(&self) -> &str {
        "Best-in-class autonomous X (Twitter) agent. P1.3a: 7 sub-agents wired; pipeline orchestration lands in P1.3b."
    }

    fn version(&self) -> &str {
        self.version
    }

    fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, heartbit_core::Error> {
        let agents = vec![
            agents::researcher_recipe(),
            agents::writer_recipe(),
            agents::style_critic_recipe(),
            agents::judge_recipe(),
            agents::fact_check_recipe(),
            agents::image_generator_recipe(),
            agents::publisher_recipe(),
        ];

        let tools = agents::tools_for_persona();

        Ok(PersonaExpansion {
            agents,
            tools,
            topic_context_provider: Some(std::sync::Arc::new(
                crate::posts::XGhostTopicContext::new(),
            )),
            // P1.3b populates orchestrator.
            // P1.3d populates review.
            // P1.4 populates triggers.
            ..PersonaExpansion::default()
        })
    }
}

/// Register the X ghost persona into the supplied registry.
///
/// Callers (e.g. `heartbit-cli` at startup, the daemon at boot) build a
/// `PersonaRegistry`, call this function (and any other persona crates'
/// equivalent functions), then pass the populated registry to the CLI
/// dispatch / daemon dispatch / etc.
pub fn register(registry: &mut PersonaRegistry) {
    registry.register(Arc::new(XGhostPersona::new()));
    heartbit_rs::register(registry);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_name_is_stable() {
        let p = XGhostPersona::new();
        assert_eq!(p.name(), "heartbit-ghost:x");
        assert_eq!(p.name(), PERSONA_NAME);
    }

    #[test]
    fn description_is_non_empty_and_marks_current_phase() {
        let p = XGhostPersona::new();
        let desc = p.description();
        assert!(!desc.is_empty());
        assert!(
            desc.contains("P1.3") || desc.contains("sub-agent"),
            "description should reflect the current phase; got: {desc}"
        );
    }

    #[test]
    fn stub_version_matches_cargo_pkg_version() {
        let p = XGhostPersona::new();
        assert_eq!(p.version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn expand_returns_seven_agents_and_five_tools_in_declared_order() {
        let p = XGhostPersona::new();
        let params = PersonaParams::default();
        let exp = p.expand(&params).expect("expand returns Ok");
        assert_eq!(exp.agents.len(), 7);
        assert_eq!(exp.tools.len(), 5);

        let agent_names: Vec<&str> = exp.agents.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            agent_names,
            vec![
                "researcher",
                "writer",
                "style_critic",
                "judge",
                "fact_check",
                "image_generator",
                "publisher",
            ]
        );

        let tool_names: Vec<String> = exp.tools.iter().map(|t| t.definition().name).collect();
        assert_eq!(
            tool_names,
            vec![
                "websearch".to_string(),
                "webfetch".to_string(),
                "image_generate".to_string(),
                "twitter_thread".to_string(),
                "twitter_reply".to_string(),
            ]
        );

        // Triggers and review remain default (P1.3d / P1.4).
        assert!(exp.triggers.is_empty());
        assert!(exp.review.is_none());

        // topic_context_provider should be populated.
        assert!(
            exp.topic_context_provider.is_some(),
            "XGhost should populate topic_context_provider"
        );
    }

    #[test]
    fn register_adds_personas_to_empty_registry() {
        let mut r = PersonaRegistry::new();
        assert!(r.is_empty());
        register(&mut r);
        assert_eq!(r.len(), 2);
        assert!(r.get(PERSONA_NAME).is_some());
        assert!(r.get(crate::heartbit_rs::PERSONA_NAME).is_some());
        let mut names = r.list();
        names.sort();
        assert_eq!(names, vec!["heartbit-ghost:x", "heartbit-rs:x"]);
    }

    #[test]
    fn register_twice_is_idempotent() {
        // PersonaRegistry::register is last-write-wins, so calling register()
        // twice should leave exactly one entry per key (two personas total).
        let mut r = PersonaRegistry::new();
        register(&mut r);
        register(&mut r);
        assert_eq!(r.len(), 2);
        assert!(r.get(PERSONA_NAME).is_some());
        assert!(r.get(crate::heartbit_rs::PERSONA_NAME).is_some());
    }
}

#[cfg(test)]
mod blog_deps_smoke {
    #[test]
    fn blog_deps_compile() {
        let _md = pulldown_cmark::Parser::new("# hello");
        let env = minijinja::Environment::new();
        let _ = env.render_str("{{ x }}", minijinja::context! { x => "y" });
        let _yaml: serde_yaml::Value = serde_yaml::from_str("a: 1").unwrap();
        let _slug = slug::slugify("Hello World!");
    }
}
