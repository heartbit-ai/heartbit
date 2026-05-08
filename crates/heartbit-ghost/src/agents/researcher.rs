//! Researcher sub-agent — websearch + webfetch. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// Construct the researcher [`AgentConfig`].
pub fn researcher_recipe() -> AgentConfig {
    super::stub_recipe("researcher")
}
