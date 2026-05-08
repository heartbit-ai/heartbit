//! Image generator sub-agent — optional accompanying image. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// Construct the image generator [`AgentConfig`].
pub fn image_generator_recipe() -> AgentConfig {
    super::stub_recipe("image_generator")
}
