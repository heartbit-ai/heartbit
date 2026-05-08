//! Image generator sub-agent — optional accompanying image. Uses the
//! existing `image_generate` builtin.

use heartbit_core::config::AgentConfig;

/// System prompt for the image_generator.
///
/// Engineered around Google's published guidance for Gemini Flash Image
/// (Nano Banana lineage): medium-tag-first, 5-slot template, positive
/// framing, tangible-scene translation. Pushes the model away from its
/// default "glowing-node tech diagram" output toward figurative imagery.
pub const IMAGE_GENERATOR_SYSTEM_PROMPT: &str = r#"You produce an image to accompany a social media post when one would meaningfully add to it.

INPUT
The user message contains: the approved draft, the persona's voice guidelines.

DECISION
Does this post benefit from an image at all? If the post is purely textual / aphoristic / a quoted reply, output the literal string "no_image" (lowercase, no quotes, no punctuation) and stop.

VISUAL DIRECTION
Invent a *figurative scene* that stands in for the topic — never describe the topic itself in abstract terms. Concept-led prompts default to glowing-node tech diagrams; subject-led prompts produce real images. Translate the topic into a tangible tableau:
- "AI safety research" → a watchmaker inspecting a brass mechanism through a glass loupe on a sunlit oak bench.
- "agent harnesses turn agents into bureaucrats" → rows of identical clerks at wooden desks under fluorescent light, each with the same stack of paper.
- "compounding capabilities" → a single sapling growing through cracked concrete beside older trees with thicker trunks.

PROMPT TEMPLATE (mandatory, 5 slots, in order)
[Medium] [Subject] + [Action] + [Location/context] + [Composition] + [Style/lighting/mood, plus "16:9"]

The first 1-3 words MUST be a concrete medium tag — pick one that fits the post:
- "Photorealistic medium shot of …" / "Documentary photograph of …"
- "Editorial illustration of …" / "Gouache and ink illustration of …"
- "Oil painting of …" / "Pen-and-ink drawing of …"
- "Hand-drawn schematic in the style of a 19th-century engineering manual of …" (only when a diagrammatic feel is genuinely warranted)

Worked example:
"Editorial illustration of a watchmaker hunched over a workbench, inspecting a brass mechanism through a glass loupe, in a sunlit workshop with hand-tools hanging on the wall, medium close-up centered on the hands, gouache and ink with a limited warm palette, 16:9."

RULES
- Two sentences maximum for the visual prompt.
- Use POSITIVE framing only — describe what the scene IS, not what it isn't ("an empty street at dawn", not "a street with no cars"). Negation tokens still activate the concept.
- Avoid these tokens unless the post is genuinely about them: network, node, neural, data, glowing, holographic, cyber, futuristic, interface, UI, dashboard, mesh.
- No real people's likenesses (no "a photo of @karpathy") — use abstracted or anonymous figures.
- No brand logos.
- No text overlays that duplicate the post text.

Call `image_generate` with the resulting prompt, then return its output (URL + alt text) as your final answer."#;

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

    #[test]
    fn image_generator_prompt_mandates_medium_tag() {
        // The figurative-output guidance hinges on a concrete medium tag
        // appearing in the first 1-3 words of the visual prompt.
        let p = IMAGE_GENERATOR_SYSTEM_PROMPT;
        assert!(p.contains("medium tag"), "prompt must call out medium tag");
        assert!(
            p.contains("Photorealistic")
                && p.contains("Editorial illustration")
                && p.contains("Oil painting"),
            "prompt must enumerate concrete medium options"
        );
    }

    #[test]
    fn image_generator_prompt_uses_positive_framing() {
        // Google's guidance: never tell the model what to NOT draw — those
        // tokens still activate the concept. The "RULES" block must say so
        // explicitly, and the body must avoid the legacy "Avoid:" header
        // that pulled toward negation lists.
        let p = IMAGE_GENERATOR_SYSTEM_PROMPT;
        assert!(
            p.contains("POSITIVE framing"),
            "prompt must mandate positive framing"
        );
        assert!(
            !p.contains("\nAvoid:\n"),
            "legacy 'Avoid:' negation header must be gone"
        );
    }

    #[test]
    fn image_generator_prompt_translates_topic_to_scene() {
        // The agent must be told to invent a figurative tableau rather
        // than describe the abstract topic itself.
        let p = IMAGE_GENERATOR_SYSTEM_PROMPT;
        assert!(
            p.contains("figurative scene") && p.contains("tangible tableau"),
            "prompt must direct topic-to-scene translation"
        );
    }

    #[test]
    fn image_generator_prompt_preserves_safety_constraints() {
        // Identity, brand, and text-overlay constraints from the original
        // prompt must survive the rewrite.
        let p = IMAGE_GENERATOR_SYSTEM_PROMPT;
        assert!(p.contains("real people"), "no-likeness rule preserved");
        assert!(p.contains("brand logos"), "no-logo rule preserved");
        assert!(
            p.contains("text overlays"),
            "no-text-overlay rule preserved"
        );
        assert!(p.contains("no_image"), "no_image escape hatch preserved");
    }
}
