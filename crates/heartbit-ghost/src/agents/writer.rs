//! Writer sub-agent — style-conditioned post generation. Reusable across
//! personas (renamed `social_writer` in the umbrella spec; file kept as
//! `writer.rs` for terseness).

use heartbit_core::config::AgentConfig;

/// System prompt for the writer. Style-profile-free — the orchestrator
/// (P1.3b) supplies voice guidelines + topic context + few-shot
/// exemplars in the user message at runtime.
pub const WRITER_SYSTEM_PROMPT: &str = r#"You are a social media writer. Produce one short, engaging post draft per call.

INPUT
The user message contains: a topic or research digest, voice guidelines for the persona, and optionally a few exemplar posts to mirror.

OUTPUT
The post text only. No preamble. No markdown fences. No commentary. No quote characters around the text. If the topic warrants a thread, separate tweets with a blank line.

Honor the voice guidelines exactly. If they say "no em-dashes", use no em-dashes. If they say "lowercase", lowercase everything. If they say "no hedging", make claims, not suggestions."#;

/// Construct the writer [`AgentConfig`].
pub fn writer_recipe() -> AgentConfig {
    AgentConfig {
        name: "writer".to_string(),
        description: "Style-conditioned post generation. One draft per call.".to_string(),
        system_prompt: WRITER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(1024),
        reasoning_effort: Some("low".to_string()),
        ..super::stub_recipe("writer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_recipe_has_expected_shape() {
        let cfg = writer_recipe();
        assert_eq!(cfg.name, "writer");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(1024));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(
            cfg.response_schema.is_none(),
            "writer produces free-form draft, no schema"
        );
    }

    #[test]
    fn writer_prompt_is_platform_agnostic() {
        let s = WRITER_SYSTEM_PROMPT.to_lowercase();
        assert!(
            !s.contains("twitter"),
            "writer prompt must not mention twitter; got snippet: {s}"
        );
        assert!(
            !s.contains("(twitter)") && !s.contains("on x ") && !s.contains(" x ("),
            "writer prompt must not mention X as a platform; got snippet: {s}"
        );
    }
}
