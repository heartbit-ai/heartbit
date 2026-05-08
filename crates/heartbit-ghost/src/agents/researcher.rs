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
}
