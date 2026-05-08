//! Fact check sub-agent — claim verification. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// Construct the fact check [`AgentConfig`].
pub fn fact_check_recipe() -> AgentConfig {
    super::stub_recipe("fact_check")
}
