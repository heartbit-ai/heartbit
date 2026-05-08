//! Image generator sub-agent — optional accompanying image. Uses the
//! existing `image_generate` builtin.

use heartbit_core::config::AgentConfig;

/// System prompt for the image_generator.
pub const IMAGE_GENERATOR_SYSTEM_PROMPT: &str = r#"You produce an image to accompany a social media post when one would meaningfully add to it.

INPUT
The user message contains: the approved draft, the persona's voice guidelines.

DECISION
First decide: does this post benefit from an image at all? If the post is purely textual / aphoristic / a quoted reply, output the literal string "no_image" (lowercase, no quotes, no punctuation) and stop.

If yes, call `image_generate` with a concise visual prompt (one or two sentences). Avoid:
- Real people's likenesses (no "a photo of @karpathy"); use abstracted compositions instead.
- Brand logos.
- Text overlays that duplicate the post text.

Return the image_generate tool's output (URL + alt text) as your final answer."#;

/// Construct the image_generator [`AgentConfig`].
pub fn image_generator_recipe() -> AgentConfig {
    AgentConfig {
        name: "image_generator".to_string(),
        description: "Optionally produce an accompanying image for a draft.".to_string(),
        system_prompt: IMAGE_GENERATOR_SYSTEM_PROMPT.to_string(),
        max_turns: Some(2),
        max_tokens: Some(1024),
        reasoning_effort: Some("low".to_string()),
        ..super::stub_recipe("image_generator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_generator_recipe_has_expected_shape() {
        let cfg = image_generator_recipe();
        assert_eq!(cfg.name, "image_generator");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(2));
        assert_eq!(cfg.max_tokens, Some(1024));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(
            cfg.response_schema.is_none(),
            "image_generator output is the tool result, not structured"
        );
    }
}
