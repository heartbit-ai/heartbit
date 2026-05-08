//! Style critic sub-agent — voice match + AI-tell detection. Partially
//! reusable across personas (the schema is generic; the criterion list
//! may be tuned per platform later).

use heartbit_core::config::AgentConfig;

/// System prompt for the style critic.
pub const STYLE_CRITIC_SYSTEM_PROMPT: &str = r#"You score how well a draft post matches a target voice and flag AI-tells.

INPUT
The user message contains: the draft, the voice guidelines for the persona, and optionally specific phrases to avoid.

DECISION
Return a JSON object exactly matching the response schema:
- verdict: "pass" | "revise" | "reject"
- reason: short string explaining the verdict (required for "revise" and "reject")
- style_match_score: float in [0.0, 1.0] — higher means better voice match

VERDICTS
- "pass": draft matches the voice; no AI-tells; ready to ship.
- "revise": draft is recoverable but has issues. Common causes: AI-tell phrase, hedging, off-voice phrasing, wrong sentence-length distribution.
- "reject": draft is fundamentally off (wrong topic, defamatory, off-voice in a way revision won't fix).

Be strict on AI-tells. Phrases like "delve into", "tapestry", "navigate", "it's important to note", "balanced both-sides", "while it's true that", and "as an AI" are immediate revise triggers if voice guidelines list them. Score 0.0-0.4 = severe mismatch; 0.4-0.7 = mismatch; 0.7-0.9 = acceptable; 0.9-1.0 = excellent."#;

/// Construct the style_critic [`AgentConfig`].
pub fn style_critic_recipe() -> AgentConfig {
    AgentConfig {
        name: "style_critic".to_string(),
        description: "Score voice match and flag AI-tells. Returns pass/revise/reject + score."
            .to_string(),
        system_prompt: STYLE_CRITIC_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(512),
        reasoning_effort: Some("medium".to_string()),
        response_schema: Some(serde_json::json!({
            "type": "object",
            "required": ["verdict", "style_match_score"],
            "properties": {
                "verdict": { "type": "string", "enum": ["pass", "revise", "reject"] },
                "reason": { "type": "string" },
                "style_match_score": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
            }
        })),
        ..super::stub_recipe("style_critic")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_critic_recipe_has_expected_shape() {
        let cfg = style_critic_recipe();
        assert_eq!(cfg.name, "style_critic");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            cfg.response_schema.is_some(),
            "style_critic produces structured verdict"
        );
        let schema = cfg.response_schema.as_ref().unwrap();
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        let required_names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(required_names.contains(&"verdict"));
        assert!(required_names.contains(&"style_match_score"));
    }
}
