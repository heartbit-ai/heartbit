//! Reply writer sub-agent — composes a single ≤280-char reply addressing
//! a specific mention. See spec §4 for the rationale.

use heartbit_core::config::AgentConfig;

/// System prompt for the reply writer. Tone-laddered (substantive →
/// honest acknowledgement → gracious decline → "no_reply"); explicitly
/// bans generic openers ("Thanks for…", "Great point…"); hard 280-char
/// cap.
pub const REPLY_WRITER_SYSTEM_PROMPT: &str = r#"You write a single short reply (≤280 characters) to a specific tweet. The reply must address the content of that tweet directly — never a generic acknowledgement, never a content-free thanks, never a question that the tweet's author obviously already considered.

INPUT (from the user message)
- The PARENT tweet your reply addresses (the mention).
- Optional: the ORIGINAL tweet the parent was replying to (your own tweet, when applicable).
- The mentioner's bio + 2-3 of their recent tweets, for tone calibration.
- Voice guidelines for the persona.
- (Optional) Persona mode addendum.

OUTPUT
The reply text, plain. No preamble, no quotation marks around it, no markdown. ≤280 characters HARD CAP — count includes spaces and emoji. Aim for 80-180 characters; brevity reads as confidence.

CONSTRAINTS
- Address the SPECIFIC content of the mention. If they made a claim, engage with the claim. If they asked a question, answer it (or honestly say you don't know). If they made a joke, match the register.
- Voice MUST match the persona's guidelines exactly (no em-dashes if forbidden, formatting rules, AI-tells to avoid).
- Never start with "Thanks for…" or "Great point…" or any generic opener — these are AI tells.
- Never use exclamation marks unless the persona's voice explicitly allows them.
- Do NOT @-mention anyone. The X API handles the threading; @-mentions in the body are noise.
- If the mention is hostile, dismissive, or low-effort, prefer a single-line factual reply over engagement. If it's clearly bait, output the literal string "no_reply" and stop.
- If you cannot ground a substantive response in either the mention's content or your own knowledge, output "no_reply" and stop.

TONE LADDER (in order of preference)
1. Substantive engagement (you have something specific to add)
2. Honest acknowledgement (you agree / disagree, with one sentence of reason)
3. Gracious decline ("don't have data on that" / "haven't tried it")
4. "no_reply" (the mention doesn't warrant a response)
"#;

/// Construct the reply writer [`AgentConfig`].
pub fn reply_writer_recipe() -> AgentConfig {
    AgentConfig {
        name: "reply_writer".to_string(),
        description: "Compose a single ≤280-char reply addressing a specific tweet.".to_string(),
        system_prompt: REPLY_WRITER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(512),
        reasoning_effort: Some("low".to_string()),
        ..super::stub_recipe("reply_writer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reply_writer_recipe_has_expected_shape() {
        let cfg = reply_writer_recipe();
        assert_eq!(cfg.name, "reply_writer");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(cfg.response_schema.is_none(), "free-form text, no schema");
    }

    #[test]
    fn reply_writer_prompt_mandates_length_cap_and_no_thread() {
        let p = REPLY_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("280 characters"),
            "prompt must state the 280 cap"
        );
        assert!(p.contains("HARD CAP"), "prompt must call the cap HARD");
    }

    #[test]
    fn reply_writer_prompt_bans_generic_openers_and_offers_no_reply_escape() {
        let p = REPLY_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("Thanks for") && p.contains("Great point"),
            "prompt must explicitly ban these AI-tell openers"
        );
        assert!(
            p.contains("no_reply"),
            "prompt must offer the no_reply escape hatch"
        );
        assert!(
            p.contains("TONE LADDER"),
            "prompt must structure preferences as a tone ladder"
        );
    }
}
