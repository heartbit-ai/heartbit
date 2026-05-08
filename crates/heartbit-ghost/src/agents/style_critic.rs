//! Style critic sub-agent — voice match + AI-tell detection. Partially reusable.

use heartbit_core::config::AgentConfig;

/// Construct the style critic [`AgentConfig`].
pub fn style_critic_recipe() -> AgentConfig {
    super::stub_recipe("style_critic")
}
