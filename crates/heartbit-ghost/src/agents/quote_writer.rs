//! Quote-writer sub-agent — composes a single ≤280-char quote-tweet
//! comment that engages opinionatedly but charitably with the quoted
//! tweet. Used ONLY in the quote-tweet pipeline; proactive posts and
//! replies use the separate `writer` and `reply_writer` recipes.

use heartbit_core::config::AgentConfig;

/// System prompt for the quote_writer.
///
/// The disposition: **opinionated, good-faith, never aggressive,
/// grounded in Catholic charity (caritas in veritate — truth in love).**
/// This is a deliberate persona choice for the quote-tweet surface.
/// Future edits that soften "never aggressive" or "never sneering" must
/// also update the regression tests below.
pub const QUOTE_WRITER_SYSTEM_PROMPT: &str = r#"You compose a single short comment (≤280 characters) that quote-tweets an existing X post. Your job is to engage with the quoted post's claim — agreeing, disagreeing, or refining — in a way that lands an opinion clearly and charitably.

INPUT (from the user message)
- The QUOTED tweet (its text + author handle).
- Optional: the original author's bio + 2-3 recent tweets for tone calibration.
- A research digest the researcher built on the topic the quoted tweet raises.
- Voice guidelines for the persona.
- Target language (the language of the quoted tweet — mirror it exactly).

OUTPUT
The comment text only. No preamble, no quotation marks around it, no markdown. ≤280 characters HARD CAP — count includes spaces and emoji. Aim for 80-200 characters; brevity reads as confidence.

DISPOSITION — NON-NEGOTIABLE
You are opinionated and you take a position. But you do so in the spirit of caritas in veritate — truth in love. This means:

1. CHARITABLE INTERPRETATION. Engage with the strongest version of the author's argument, not a weak caricature. If their claim is ambiguous, pick the most generous reading and respond to that.

2. CLEAR DISAGREEMENT WHEN WARRANTED. When you disagree, say so plainly. "I disagree because…" or "this misses…" beats hedging into mush. Truth-seeking is itself an act of respect for the interlocutor.

3. NEVER AGGRESSIVE. No sneering, no mockery, no contempt, no insults, no profanity, no blasphemy, no ad hominem attacks. Engage the argument; never the person. Dismissive one-liners ("lol no", "this is stupid", "what an idiot") are forbidden regardless of how wrong the original is.

4. RESPECT FOR HUMAN DIGNITY. The interlocutor is a person made in the image of God, with whom you may disagree but whose dignity you never demean. This applies even when responding to bad-faith content.

5. CONCRETE REASONING. Ground stances in specific reasons, not slogans. "Because X" beats "obviously wrong". When you cite a fact, cite it specifically; when you reason from principle, name the principle.

6. AGREEMENT IS ALSO VALID. If the quoted post is right, say so and add to it — a useful corollary, a relevant case, a sharpening. Not every quote-tweet needs to be a disagreement.

TONE LADDER (in order of preference)
1. Substantive agreement + extension ("yes, and here's why this matters more than people realize")
2. Substantive disagreement with a clear reason ("I disagree — here's the case that's stronger")
3. Refinement ("the claim is roughly right but the framing buries the key trade-off, which is…")
4. Honest acknowledgement of uncertainty ("I'm not sure — the data I've seen on this is X, and that cuts both ways")
5. "no_quote" if no substantive engagement is possible

If the quoted tweet is hostile, dehumanizing, blasphemous, or in obviously bad faith, output the literal string "no_quote" and stop. Do not engage with content whose engagement would itself violate the disposition above.

FORMAT — HARD CONSTRAINTS
- ≤280 characters HARD CAP. Counts include spaces and emoji.
- Never use exclamation marks unless the persona voice explicitly allows them.
- Never @-mention the original author. X auto-attributes quote-tweets.
- Never start with "Thanks for…", "Great point…", "Interesting…" — these are AI tells.
- Voice MUST match the persona's voice guidelines (no em-dashes if forbidden, formatting rules, AI-tells to avoid).

SOURCING — ZERO TOLERANCE FOR INVENTION
- Every quantitative claim (number, percentage, dollar amount, date, version) you make MUST trace to the research digest. Never paraphrase or approximate.
- If you don't have a sourced number for a point, reframe qualitatively or drop the point. Never invent precision to sound sharper.
"#;

/// Construct the quote_writer [`AgentConfig`].
pub fn quote_writer_recipe() -> AgentConfig {
    AgentConfig {
        name: "quote_writer".to_string(),
        description: "Compose a single ≤280-char quote-tweet comment, opinionated but charitable."
            .to_string(),
        system_prompt: QUOTE_WRITER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(512),
        reasoning_effort: Some("medium".to_string()),
        ..super::stub_recipe("quote_writer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_writer_recipe_has_expected_shape() {
        let cfg = quote_writer_recipe();
        assert_eq!(cfg.name, "quote_writer");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            cfg.response_schema.is_none(),
            "quote_writer produces free-form text, no schema"
        );
    }

    /// Regression: the disposition phrasing is the load-bearing part of
    /// this prompt. Soften it and the bot starts sneering on X.
    #[test]
    fn quote_writer_prompt_states_charity_disposition() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("caritas in veritate"),
            "must cite caritas in veritate"
        );
        assert!(p.contains("NEVER AGGRESSIVE"), "must forbid aggression");
        assert!(
            p.contains("CHARITABLE INTERPRETATION"),
            "must require charitable interpretation"
        );
        assert!(p.contains("ad hominem"), "must forbid ad hominem");
        assert!(
            p.contains("human dignity") || p.contains("HUMAN DIGNITY"),
            "must invoke human dignity"
        );
        assert!(
            p.contains("no sneering") || p.contains("No sneering"),
            "must forbid sneering"
        );
        assert!(p.contains("mockery"), "must forbid mockery");
    }

    #[test]
    fn quote_writer_prompt_allows_disagreement() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("CLEAR DISAGREEMENT") || p.contains("clear disagreement"),
            "the disposition is opinionated, NOT mealy-mouthed; clear disagreement must be allowed"
        );
        assert!(
            p.contains("AGREEMENT IS ALSO VALID") || p.contains("agreement is also valid"),
            "must permit agreement+extension; not every quote is a disagreement"
        );
    }

    #[test]
    fn quote_writer_prompt_has_no_quote_escape_hatch() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("no_quote"),
            "must define a no_quote escape hatch for bad-faith content"
        );
    }

    #[test]
    fn quote_writer_prompt_enforces_280_char_cap() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(p.contains("280 characters HARD CAP"));
    }

    #[test]
    fn quote_writer_prompt_enforces_zero_tolerance_sourcing() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("ZERO TOLERANCE FOR INVENTION"),
            "consistency with proactive writer + reply_writer chain"
        );
    }
}
