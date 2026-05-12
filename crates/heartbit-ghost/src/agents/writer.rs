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
The post text only. No preamble. No markdown fences. No commentary. No quote characters around the text.

FORMAT — HARD CONSTRAINTS
- Each tweet ≤280 characters HARD CAP. Counts include spaces and emoji.
- If your content would exceed 280 chars, split it into a thread. Separate tweets with a BLANK LINE (i.e. a double newline). A single newline is a line break WITHIN one tweet, NOT a tweet separator. Drafts that put each sentence on its own single-newline line are read as ONE long tweet and will be rejected.
- Respect the persona's thread_max_length from the voice guidelines — never emit more tweets than that. Prefer fewer, denser tweets to a long thread.

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

    /// Regression: the publish gate rejects any tweet that exceeds 280
    /// chars (see `pipeline::publish_gate::check_publish_gate`). The writer
    /// must know this hard cap so it splits long content into a thread
    /// instead of producing a single 1000+ char block with internal `\n`
    /// line breaks.
    #[test]
    fn writer_prompt_states_280_per_tweet_cap() {
        let p = WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("280"),
            "prompt must state the 280-char per-tweet cap; got: {p}"
        );
        assert!(
            p.contains("blank line") || p.contains("double newline"),
            "prompt must explain thread separator (blank line / double newline); got: {p}"
        );
    }
}
