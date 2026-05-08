//! Publisher sub-agent — Twitter-specific final API call.

use heartbit_core::config::AgentConfig;

/// Construct the publisher [`AgentConfig`].
pub fn publisher_recipe() -> AgentConfig {
    super::stub_recipe("publisher")
}
