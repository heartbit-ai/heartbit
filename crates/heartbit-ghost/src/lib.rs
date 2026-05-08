//! `heartbit-ghost` — best-in-class autonomous X (Twitter) agent persona.
//!
//! P1.0 (this release) ships a scaffolding stub: the persona registers itself
//! into `heartbit_core::PersonaRegistry` so the CLI surface lights up, but
//! `expand()` returns an empty `PersonaExpansion` (no agents, no tools, no
//! triggers, no review channel). Real bodies land in P1.1 (X tool family),
//! P1.2 (voice modeling), P1.3 (generation pipeline + Telegram review), and
//! P1.4 (autonomy phases + audit + dataset export).

#![deny(missing_docs)]

use std::sync::Arc;

use heartbit_core::{Persona, PersonaExpansion, PersonaParams, PersonaRegistry};

pub mod corpus;
pub mod tools;
pub mod voice;

/// Stable persona identifier — used as the registry key and as the
/// `recipe = "..."` value in `[[persona]]` config blocks.
pub const PERSONA_NAME: &str = "heartbit-ghost:x";

/// Scaffolding stub for the X (Twitter) ghost persona.
///
/// In P1.0 this expands to an empty `PersonaExpansion`. Real expansion
/// (sub-agents, tools, triggers, review spec) lands in P1.1+.
pub struct XGhostPersona {
    /// Persona version string, derived at compile time from the workspace
    /// `Cargo.toml`.
    version: &'static str,
}

impl XGhostPersona {
    /// Create a new stub instance.
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
        "Best-in-class autonomous X (Twitter) agent. Scaffolding stub — Phase 1 P1.0."
    }

    fn version(&self) -> &str {
        self.version
    }

    fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, heartbit_core::Error> {
        // P1.0 stub: empty expansion. P1.1+ fills this with the real persona
        // (sub-agent recipes, X tool family, triggers, Telegram review).
        Ok(PersonaExpansion::default())
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
    fn stub_description_is_non_empty_and_marks_p1_0() {
        let p = XGhostPersona::new();
        let desc = p.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("P1.0") || desc.contains("Scaffolding") || desc.contains("stub"));
    }

    #[test]
    fn stub_version_matches_cargo_pkg_version() {
        let p = XGhostPersona::new();
        assert_eq!(p.version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn stub_expand_returns_empty_expansion() {
        let p = XGhostPersona::new();
        let params = PersonaParams::default();
        let exp = p.expand(&params).expect("expand returns Ok");
        assert!(exp.agents.is_empty());
        assert!(exp.tools.is_empty());
        assert!(exp.triggers.is_empty());
        assert!(exp.review.is_none());
    }

    #[test]
    fn register_adds_persona_to_empty_registry() {
        let mut r = PersonaRegistry::new();
        assert!(r.is_empty());
        register(&mut r);
        assert_eq!(r.len(), 1);
        assert!(r.get(PERSONA_NAME).is_some());
        assert_eq!(r.list(), vec!["heartbit-ghost:x"]);
    }

    #[test]
    fn register_twice_is_idempotent() {
        // PersonaRegistry::register is last-write-wins, so calling register()
        // twice should leave exactly one entry under the same key.
        let mut r = PersonaRegistry::new();
        register(&mut r);
        register(&mut r);
        assert_eq!(r.len(), 1);
        assert!(r.get(PERSONA_NAME).is_some());
    }
}
