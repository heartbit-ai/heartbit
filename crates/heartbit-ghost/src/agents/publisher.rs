//! Publisher sub-agent — Twitter-specific final API call.
//! NOT reusable across personas.

use heartbit_core::config::AgentConfig;

/// System prompt for the publisher. The ONLY recipe in heartbit-ghost
/// that mentions X / Twitter — the others are platform-agnostic.
pub const PUBLISHER_SYSTEM_PROMPT: &str = r#"You publish a finalized social post to X (Twitter).

INPUT
The user message contains: the approved post text and an indication of post shape (single, thread, or reply).

TOOL CHOICE
- For any post that's NOT a reply (single tweet OR a chained thread): call `twitter_thread`. Pass a single-element array for one tweet; pass the full sequence for a thread.
- For a reply to an existing tweet: call `twitter_reply` with the target tweet id.

The post text is approved — do not modify, do not paraphrase, do not "improve" it. Pass it through verbatim.

Return the tool's output (the tweet id) without commentary."#;

/// Construct the publisher [`AgentConfig`].
pub fn publisher_recipe() -> AgentConfig {
    AgentConfig {
        name: "publisher".to_string(),
        description: "Twitter-specific publisher. Calls twitter_thread or twitter_reply."
            .to_string(),
        system_prompt: PUBLISHER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(2),
        max_tokens: Some(512),
        reasoning_effort: Some("low".to_string()),
        ..super::stub_recipe("publisher")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publisher_recipe_has_expected_shape() {
        let cfg = publisher_recipe();
        assert_eq!(cfg.name, "publisher");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(2));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(
            cfg.response_schema.is_none(),
            "publisher output is the tool result (tweet id)"
        );
    }

    #[test]
    fn publisher_prompt_mentions_x_or_twitter() {
        let s = PUBLISHER_SYSTEM_PROMPT.to_lowercase();
        assert!(
            s.contains("twitter") || s.contains("x (twitter)") || s.contains(" x "),
            "publisher prompt must reference X/Twitter; got: {s}"
        );
    }
}
