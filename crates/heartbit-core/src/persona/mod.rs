//! Persona registry: a small abstraction so concrete persona crates
//! (e.g. `heartbit-ghost`) plug in identically. Empty in Phase 0;
//! concrete personas land in Phase 1.

pub mod types;

use std::collections::HashMap;
use std::sync::Arc;

pub use types::{AuthorshipMode, PersonaExpansion, PersonaParams, ReviewSpec, TriggerSpec};

use crate::error::Error;

/// A persona is a recipe that expands into agent configurations, tools,
/// triggers, and a review spec. Implementations live in dedicated crates
/// (e.g. `heartbit-ghost`) and register themselves into a `PersonaRegistry`
/// at startup.
pub trait Persona: Send + Sync {
    /// Stable persona identifier. Convention: `<crate_short_name>:<recipe>`,
    /// e.g. `"heartbit-ghost:x"`.
    fn name(&self) -> &str;

    /// One-line human-readable description.
    fn description(&self) -> &str;

    /// Persona version (semver-ish). Useful for audit logs.
    fn version(&self) -> &str;

    /// Expand the persona into runtime artifacts using the per-instance params.
    fn expand(&self, params: &PersonaParams) -> Result<PersonaExpansion, Error>;
}

/// In-memory registry of personas by name. Constructed at startup; concrete
/// personas (from dependent crates) call `register()` during initialization.
pub struct PersonaRegistry {
    personas: HashMap<String, Arc<dyn Persona>>,
}

impl Default for PersonaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PersonaRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            personas: HashMap::new(),
        }
    }

    /// Register a persona. The persona's `name()` is used as the key.
    /// Registering the same name twice replaces the previous entry (last-write-wins).
    pub fn register(&mut self, persona: Arc<dyn Persona>) {
        self.personas.insert(persona.name().to_string(), persona);
    }

    /// Look up a persona by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Persona>> {
        self.personas.get(name).cloned()
    }

    /// List the names of all registered personas.
    pub fn list(&self) -> Vec<&str> {
        self.personas.keys().map(|k| k.as_str()).collect()
    }

    /// Number of registered personas.
    pub fn len(&self) -> usize {
        self.personas.len()
    }

    /// True if no personas are registered.
    pub fn is_empty(&self) -> bool {
        self.personas.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPersona;

    impl Persona for DummyPersona {
        fn name(&self) -> &str {
            "dummy:p"
        }
        fn description(&self) -> &str {
            "test persona"
        }
        fn version(&self) -> &str {
            "0.1.0"
        }
        fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, Error> {
            Ok(PersonaExpansion::default())
        }
    }

    #[test]
    fn registry_starts_empty() {
        let r = PersonaRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.list().is_empty());
        assert!(r.get("anything").is_none());
    }

    #[test]
    fn register_and_get_round_trip() {
        let mut r = PersonaRegistry::new();
        r.register(Arc::new(DummyPersona));
        assert_eq!(r.len(), 1);
        assert!(r.get("dummy:p").is_some());
        assert_eq!(r.list(), vec!["dummy:p"]);
    }

    #[test]
    fn register_same_name_replaces() {
        let mut r = PersonaRegistry::new();
        r.register(Arc::new(DummyPersona));
        r.register(Arc::new(DummyPersona));
        assert_eq!(r.len(), 1);
    }
}
