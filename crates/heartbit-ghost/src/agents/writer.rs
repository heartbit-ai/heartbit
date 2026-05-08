//! Writer sub-agent — style-conditioned generation. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// Construct the writer [`AgentConfig`].
pub fn writer_recipe() -> AgentConfig {
    super::stub_recipe("writer")
}
