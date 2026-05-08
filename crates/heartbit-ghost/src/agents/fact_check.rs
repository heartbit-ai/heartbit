//! Fact check sub-agent — claim verification against research output.
//! Reusable across personas.

use heartbit_core::config::AgentConfig;

/// System prompt for the fact_check agent.
pub const FACT_CHECK_SYSTEM_PROMPT: &str = r#"You verify the factual claims in a draft post against the research digest produced earlier in the pipeline.

INPUT
The user message contains: the draft, the research digest with sources.

DECISION
Return a JSON object exactly matching the response schema:
- verdict: "verified" | "unverifiable"
- reason: required when verdict is "unverifiable" — short string explaining what claim couldn't be verified

PROCESS
1. Extract every factual claim in the draft (numbers, attributions, dates, "X said Y" assertions, "the paper showed Z" references).
2. For each claim, check whether the research digest supports it.
3. If any claim is contradicted by the digest or absent from it, return "unverifiable" with the specific claim called out.
4. If every claim is supported, return "verified".

Do NOT verify against your training data. Only the supplied research digest counts. Aesthetic / stylistic / opinion content is not subject to fact-check (skip it)."#;

/// Construct the fact_check [`AgentConfig`].
pub fn fact_check_recipe() -> AgentConfig {
    AgentConfig {
        name: "fact_check".to_string(),
        description: "Verify factual claims in a draft against the research digest.".to_string(),
        system_prompt: FACT_CHECK_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(1024),
        reasoning_effort: Some("medium".to_string()),
        response_schema: Some(serde_json::json!({
            "type": "object",
            "required": ["verdict"],
            "properties": {
                "verdict": { "type": "string", "enum": ["verified", "unverifiable"] },
                "reason": { "type": "string" }
            }
        })),
        ..super::stub_recipe("fact_check")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_check_recipe_has_expected_shape() {
        let cfg = fact_check_recipe();
        assert_eq!(cfg.name, "fact_check");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(1024));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            cfg.response_schema.is_some(),
            "fact_check produces structured verdict"
        );
    }
}
