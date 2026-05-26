//! Image generator sub-agent — optional accompanying image. Uses the
//! existing `image_generate` builtin.

use heartbit_core::config::AgentConfig;

/// System prompt for the image_generator.
///
/// Tuned for **abstract black-and-white engraving / etching / lithograph**
/// aesthetics. The image may be conceptual and need not literally depict
/// the post's subject — visual mood and composition matter more than
/// narrative fit. Built on Google's Gemini Flash Image guidance
/// (medium-tag-first, 5-slot template, positive framing).
pub const IMAGE_GENERATOR_SYSTEM_PROMPT: &str = r#"You produce an image to accompany a social media post when one would meaningfully add to it.

INPUT
The user message contains: the approved draft, the persona's voice guidelines.

DECISION
Does this post benefit from an image at all? If the post is purely textual / aphoristic / a quoted reply, output the literal string "no_image" (lowercase, no quotes, no punctuation) and stop.

VISUAL DIRECTION
The image is a **black-and-white engraving / etching / wood-engraving / lithograph** — abstract or conceptual, not a literal illustration of the post. Think 19th-century scientific engravings, Gustave Doré's hatched lines, M. C. Escher's geometric impossibilities, Franz Masereel's silent novels, Albrecht Dürer's fine cross-hatching.

The image does NOT need to depict the post's subject. It can be a tangentially-related metaphor (a single mechanical part, a celestial map, an architectural fragment, a hand, a fold of cloth, a knot, an empty interior, a horizon line) or a purely abstract composition (geometric impossibility, tessellation, radiating lines, layered isobars). Mood and texture beat literal narrative fit.

PROMPT TEMPLATE (mandatory, 5 slots, in order)
[Medium tag] [Subject] + [Action or arrangement] + [Location/context, OPTIONAL for purely abstract] + [Composition] + [Style/light/mood, always including "black and white" and "16:9"]

The first 2-4 words MUST be one of these medium tags (engraving family ONLY):
- "Black-and-white wood engraving of …"
- "Black-and-white copper-plate etching of …"
- "Black-and-white pen-and-ink engraving in the style of a 19th-century scientific atlas, depicting …"
- "Black-and-white lithograph of …"
- "Black-and-white scratchboard illustration of …"

Worked examples:
- "Black-and-white wood engraving of a single brass key floating in front of an empty doorway, the door's wood-grain rendered in tight parallel hatching, sharp white edges on a deep ink-black background, 16:9."
- "Black-and-white copper-plate etching in the style of a 19th-century scientific atlas, depicting a tessellation of overlapping hexagonal mechanical parts, fine cross-hatched shading, centered composition, museum-engraving aesthetic, 16:9."
- "Black-and-white scratchboard illustration of a knotted rope hanging in empty space, white lines on solid black, dramatic chiaroscuro, minimalist composition, 16:9."

RULES
- Two sentences maximum for the visual prompt.
- Black and white ONLY. No color tokens (sepia, ochre, blue, etc.). If a token of color slips in, the image will return color.
- Always include the literal phrase "black and white" in the prompt — Gemini honors it best when explicit.
- Use POSITIVE framing — describe what the image IS, not what it isn't. Negation tokens still activate the concept.
- Avoid these tokens unless the post is genuinely about them: network, node, neural, data, glowing, holographic, cyber, futuristic, interface, UI, dashboard, mesh, photorealistic, photograph, watercolor, pastel, gouache.
- No real people's likenesses (no "a portrait of @karpathy") — anonymous figures only, faces partially obscured if at all.
- No brand logos.
- No text overlays. Captions and labels in scientific-atlas-style images are fine if generic ("Fig. 1", "Plate IV") and never reproduce the post text.

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

/// System prompt for the image_search agent (Openverse online search).
///
/// Same decision shape as the AI generator — decide whether an image
/// fits, then either output `no_image` or call the search tool — but
/// the action is `openverse_image_search` with concise keywords rather
/// than `image_generate` with a visual prompt.
pub const IMAGE_SEARCH_SYSTEM_PROMPT: &str = r#"You find an existing CC0/public-domain image to accompany a social media post when one would meaningfully add to it.

INPUT
The user message contains: the approved draft, the persona's voice guidelines.

DECISION
Does this post benefit from an image at all? If the post is purely textual / aphoristic / a quoted reply, output the literal string "no_image" (lowercase, no quotes, no punctuation) and stop.

SEARCH
If an image fits, call `openverse_image_search` with 2-4 concise SEARCH KEYWORDS describing the subject — not a sentence. Examples: "rust programming code", "data center servers", "ocean waves storm". Prefer concrete nouns over abstract phrasing. Then return the tool's output as your final answer.

RULES
- Keywords only — no full sentences, no punctuation-heavy queries.
- No real people's names or likeness searches.
- No brand-logo searches."#;

/// Construct the image_search [`AgentConfig`] for the Openverse online
/// path. Mirrors [`image_generator_recipe`]'s shape but instructs the
/// agent to call `openverse_image_search` with keywords.
pub fn image_search_recipe() -> AgentConfig {
    AgentConfig {
        name: "image_search".to_string(),
        description: "Optionally find an accompanying CC0/public-domain image for a draft."
            .to_string(),
        system_prompt: IMAGE_SEARCH_SYSTEM_PROMPT.to_string(),
        max_turns: Some(2),
        max_tokens: Some(1024),
        reasoning_effort: Some("low".to_string()),
        ..super::stub_recipe("image_search")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_search_recipe_has_expected_shape() {
        let cfg = image_search_recipe();
        assert_eq!(cfg.name, "image_search");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(2));
        assert_eq!(cfg.max_tokens, Some(1024));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
    }

    #[test]
    fn image_search_prompt_names_openverse_tool_and_keywords() {
        let p = IMAGE_SEARCH_SYSTEM_PROMPT;
        assert!(
            p.contains("openverse_image_search"),
            "prompt must name the openverse_image_search tool"
        );
        assert!(
            p.contains("no_image"),
            "no_image escape hatch must be present"
        );
        assert!(
            p.contains("KEYWORDS") || p.contains("keywords"),
            "prompt must steer toward concise keywords"
        );
    }

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
    fn image_generator_prompt_mandates_engraving_medium_tag() {
        // The visual style must be locked to the engraving family — the
        // prompt forces a `Black-and-white <engraving variant>` opener.
        let p = IMAGE_GENERATOR_SYSTEM_PROMPT;
        assert!(p.contains("medium tag"), "prompt must call out medium tag");
        assert!(
            p.contains("wood engraving")
                && p.contains("copper-plate etching")
                && p.contains("lithograph"),
            "prompt must enumerate engraving-family medium options"
        );
        assert!(
            p.contains("black and white"),
            "prompt must mandate the literal 'black and white' phrase"
        );
        // Forbidden color / non-engraving mediums must NOT appear as
        // recommendations — they were the old figurative-prompt set.
        for forbidden in [
            "Photorealistic medium shot",
            "Editorial illustration of",
            "Oil painting of",
            "Gouache and ink illustration",
        ] {
            assert!(
                !p.contains(forbidden),
                "engraving prompt must drop the figurative medium tag {forbidden:?}"
            );
        }
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
    fn image_generator_prompt_allows_abstract_or_tangential_subject() {
        // The new direction explicitly RELAXES the literal-illustration
        // mandate. Abstract / conceptual / tangentially-related images
        // are allowed — mood and texture beat narrative fit.
        let p = IMAGE_GENERATOR_SYSTEM_PROMPT;
        assert!(
            p.contains("abstract") || p.contains("conceptual"),
            "prompt must allow abstract / conceptual imagery"
        );
        assert!(
            p.contains("does NOT need to depict") || p.contains("not a literal illustration"),
            "prompt must say the image need not literally depict the post"
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
