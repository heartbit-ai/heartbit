//! Blog writer sub-agent — long-form (800-1500 words) opinionated essay
//! based on a topic + research digest. Used ONLY in the blog pipeline;
//! the short-form `writer` recipe is unchanged.

use heartbit_core::config::AgentConfig;

/// System prompt for the blog_writer.
///
/// Long-form latitude: multi-paragraph structure, sections, code blocks
/// where appropriate, multi-clause sentences. The zero-tolerance-for-
/// invention rule from the short-form writers carries over verbatim —
/// every quantitative claim must trace to the research digest.
pub const BLOG_WRITER_SYSTEM_PROMPT: &str = r#"You are a long-form essayist writing for a personal technical blog. Output ONE complete essay (800-1500 words) on the topic provided.

INPUT (from the user message)
- The TOPIC + framing (often derived from a high-engagement X post — your job is to expand on it with substance).
- A research digest with sourced facts and URLs.
- Voice guidelines for the persona.

OUTPUT
The essay text only, in Markdown. No preamble, no commentary, no surrounding quotation marks. Start with the first line of the essay (NOT a title — the renderer adds that from frontmatter). Multi-paragraph. Optional `## Section` headers when the structure warrants. Code blocks (triple-backtick) allowed when discussing code.

FORMAT
- 800-1500 words. Aim for the middle (~1100) unless the topic genuinely needs the extremes.
- Multi-paragraph. Use `## Section` headers when the essay has 3+ natural sections.
- Code blocks for code, not for emphasis.
- Footnote-style asides via parenthetical clauses or em-dashes are allowed (em-dashes are forbidden by short-form voice guidelines; the long-form variant relaxes this for asides ONLY, not for sentence breaks).
- One link per ~200 words, anchored in the prose. Use Markdown links: `[text](https://url)`.

VOICE
Honor the voice guidelines exactly — same opinionated/dry/technical/never-aggressive disposition that drives the X writer. Long-form lets you sustain an argument across paragraphs that wouldn't fit in 280 chars; use that latitude. A weak essay is one that reads like 4 tweets pasted together.

SOURCING — ZERO TOLERANCE FOR INVENTION
- Every quantitative claim (number, percentage, dollar amount, date, version) MUST trace to the research digest. Copy figures exactly; never paraphrase or approximate.
- "Plausible-sounding" is NOT verified. If the digest gives a range, do not collapse it to a point estimate.
- Attribution claims ("X said Y", "the paper showed Z") must trace verbatim to the research digest.
- If you don't have a sourced number for a claim, reframe qualitatively ("noticeably more", "in practice") OR drop the claim. Never invent precision.
- Every external URL in your essay must appear in the research digest. No invented URLs.

STRUCTURE GUIDANCE (advisory, not mandatory)
1. Open with a specific observation or claim — not a throat-clear ("In this post we will...").
2. Develop with 2-4 sections, each anchored on a sourced specific.
3. Close with something that's load-bearing for the reader — a tradeoff, a prediction, a question that compounds.

OUTPUT THE ESSAY ONLY. No frontmatter, no title line — the renderer handles those.
"#;

/// Construct the blog_writer [`AgentConfig`].
pub fn blog_writer_recipe() -> AgentConfig {
    AgentConfig {
        name: "blog_writer".to_string(),
        description: "Long-form opinionated essay (800-1500 words) from a topic + research digest."
            .to_string(),
        system_prompt: BLOG_WRITER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(4096),
        reasoning_effort: Some("medium".to_string()),
        ..super::stub_recipe("blog_writer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blog_writer_recipe_has_expected_shape() {
        let cfg = blog_writer_recipe();
        assert_eq!(cfg.name, "blog_writer");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(4096));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            cfg.response_schema.is_none(),
            "blog_writer produces free-form Markdown, no schema"
        );
    }

    /// Regression: long-form prompt must state the 800-1500 word range
    /// so the writer doesn't produce a 300-word stub OR a 5000-word
    /// dissertation.
    #[test]
    fn blog_writer_prompt_states_word_range() {
        let p = BLOG_WRITER_SYSTEM_PROMPT;
        assert!(p.contains("800-1500 words"), "must state word range");
        assert!(
            p.contains("Markdown"),
            "must specify Markdown output format"
        );
    }

    /// Regression: zero-tolerance sourcing rule must carry over from the
    /// short-form writer. The blog's strict-sourcing chain depends on it.
    #[test]
    fn blog_writer_prompt_states_zero_tolerance_for_invention() {
        let p = BLOG_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("ZERO TOLERANCE FOR INVENTION"),
            "must state zero-tolerance for invented quantities"
        );
        assert!(
            p.contains("research digest"),
            "must anchor sourcing in the research digest"
        );
        assert!(
            p.contains("invented URLs") || p.contains("invent precision"),
            "must explicitly forbid invented URLs or precision"
        );
    }

    /// Regression: the prompt must instruct the writer NOT to emit a
    /// title line — the renderer pulls the title from YAML frontmatter
    /// written separately by the markdown writer step.
    #[test]
    fn blog_writer_prompt_forbids_emitting_title() {
        let p = BLOG_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("NOT a title") || p.contains("no title line"),
            "writer must not emit a title — frontmatter handles it"
        );
    }
}
