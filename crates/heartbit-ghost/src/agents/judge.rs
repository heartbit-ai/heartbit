//! Judge sub-agent — multi-candidate ranking. Reusable across personas.

use heartbit_core::config::AgentConfig;

/// System prompt for the judge.
pub const JUDGE_SYSTEM_PROMPT: &str = r#"You rank N candidate drafts and pick the best one.

INPUT
The user message contains: the topic context, the voice guidelines, and N candidate drafts numbered from 0.

DECISION
Return a JSON object exactly matching the response schema:
- chosen_index: integer in [0, N-1] — the index of the best candidate
- reasoning: short string (1-2 sentences) explaining why the chosen candidate beats the others

CRITERIA (in priority order)
1. Voice match — does it sound like the persona's writer would write it?
2. Substance — does it say something concrete and worth reading?
3. AI-tells — fewer is better.
4. Specificity — concrete numbers, names, examples beat vague claims.
5. Engagement — would a thoughtful human reader stop scrolling?

Pick decisively. If two candidates are genuinely tied, prefer the lower index (deterministic tiebreak)."#;

/// Construct the judge [`AgentConfig`].
pub fn judge_recipe() -> AgentConfig {
    AgentConfig {
        name: "judge".to_string(),
        description: "Multi-candidate ranking. Picks the best draft from N options.".to_string(),
        system_prompt: JUDGE_SYSTEM_PROMPT.to_string(),
        // 2 turns so the agent's schema-validation retry can fire once
        // when the LLM returns malformed JSON. Same rationale as
        // style_critic / fact_check.
        max_turns: Some(2),
        max_tokens: Some(512),
        reasoning_effort: Some("medium".to_string()),
        response_schema: Some(serde_json::json!({
            "type": "object",
            "required": ["chosen_index", "reasoning"],
            "properties": {
                "chosen_index": { "type": "integer", "minimum": 0 },
                "reasoning": { "type": "string" }
            }
        })),
        ..super::stub_recipe("judge")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judge_recipe_has_expected_shape() {
        let cfg = judge_recipe();
        assert_eq!(cfg.name, "judge");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(
            cfg.max_turns,
            Some(2),
            "schema-validation retry needs 2 turns; see comment on the field"
        );
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            cfg.response_schema.is_some(),
            "judge produces structured pick"
        );
    }

    #[test]
    fn judge_prompt_is_platform_agnostic() {
        let s = JUDGE_SYSTEM_PROMPT.to_lowercase();
        assert!(
            !s.contains("twitter"),
            "judge prompt must not mention twitter; got: {s}"
        );
        assert!(
            !s.contains("(twitter)") && !s.contains("on x ") && !s.contains(" x ("),
            "judge prompt must not mention X as a platform; got: {s}"
        );
    }
}
