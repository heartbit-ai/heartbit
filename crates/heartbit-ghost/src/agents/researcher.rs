//! Researcher sub-agent — websearch + webfetch. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// System prompt for the researcher. Platform-agnostic — does not mention
/// X / Twitter; reusable for future personas (LinkedIn, blog, newsletter).
pub const RESEARCHER_SYSTEM_PROMPT: &str = r#"You are a research analyst. Given a topic or question from the user, find substance: facts, sources, recent developments, and quotable specifics.

PROCESS
1. Use `websearch` to discover relevant articles, posts, papers, and announcements.
2. Use `webfetch` to read the most promising 2-4 sources in full.
3. Synthesize a structured digest. No editorializing — your job is to surface signal, not opinion.

OUTPUT FORMAT (free-form text, no JSON):
- 1-2 sentence framing of the topic.
- 5-8 bullet points of concrete facts, each with a source link inline.
- A short list of "open questions" the topic raises (3-5 items).
- Notable quotes (1-3) attributed by author and link.

SOURCING — ZERO TOLERANCE FOR INVENTION
- Every quantitative claim (number, percentage, dollar amount, date, count, version) MUST have an inline source link from a URL you actually fetched. Never paraphrase numbers — copy the exact figure from the source.
- "Many companies report X", "studies show Y", "researchers found Z" without a specific URL = drop the claim entirely. Vague aggregates are the most common hallucination vector.
- A bullet without a URL is not a fact. Either find the source or omit the bullet.
- Attribution claims ("X said Y", "the paper showed Z") must trace to a fetched source. No invented quotes.
- If you cannot find any sourced facts, return a SHORT digest stating that and listing the open questions only. Better to ship less than to ship fabrication — the writer downstream is required to refuse unsourced numbers, so unsupported bullets just get dropped anyway.

Do NOT write the post itself. The writer agent will compose. Do NOT speculate beyond what the sources support."#;

/// Construct the researcher [`AgentConfig`].
pub fn researcher_recipe() -> AgentConfig {
    AgentConfig {
        name: "researcher".to_string(),
        description: "Find substance: facts, sources, quotable specifics on a topic.".to_string(),
        system_prompt: RESEARCHER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(20),
        max_tokens: Some(4096),
        reasoning_effort: Some("medium".to_string()),
        ..super::stub_recipe("researcher")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn researcher_recipe_has_expected_shape() {
        let cfg = researcher_recipe();
        assert_eq!(cfg.name, "researcher");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(20));
        assert_eq!(cfg.max_tokens, Some(4096));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            cfg.response_schema.is_none(),
            "researcher produces free-form digest, no schema"
        );
    }

    #[test]
    fn researcher_prompt_is_platform_agnostic() {
        let s = RESEARCHER_SYSTEM_PROMPT.to_lowercase();
        assert!(
            !s.contains("twitter"),
            "researcher prompt must not mention twitter; got snippet: {s}"
        );
        assert!(
            !s.contains("(twitter)") && !s.contains("on x ") && !s.contains(" x ("),
            "researcher prompt must not mention X as a platform; got snippet: {s}"
        );
    }

    /// Regression: zero-tolerance for invented stats in the digest. The
    /// researcher is the first line of defense; if it fabricates a
    /// quantitative claim, downstream writer + fact_check both pass it
    /// through because they only check against the digest, not the world.
    #[test]
    fn researcher_prompt_states_zero_tolerance_for_invention() {
        let p = RESEARCHER_SYSTEM_PROMPT;
        assert!(
            p.contains("ZERO TOLERANCE") || p.contains("zero tolerance"),
            "researcher prompt must state zero-tolerance for invented numbers; got: {p}"
        );
        assert!(
            p.contains("source link") || p.contains("URL"),
            "researcher prompt must require explicit source links for quant claims; got: {p}"
        );
        assert!(
            p.contains("drop the claim") || p.contains("omit") || p.contains("not a fact"),
            "researcher prompt must instruct dropping unsourced claims; got: {p}"
        );
    }
}
