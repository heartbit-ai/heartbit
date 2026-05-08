//! Judge sub-agent — multi-candidate ranking. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// Construct the judge [`AgentConfig`].
pub fn judge_recipe() -> AgentConfig {
    super::stub_recipe("judge")
}
