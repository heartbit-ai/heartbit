//! Topic generator sub-agent — proposes ONE specific thread topic
//! (or "no_topic") from pre-fetched static context. See spec §4.

use heartbit_core::config::AgentConfig;

/// System prompt for the topic generator. No tools — pure text-in /
/// text-out. The handler pre-fetches all context and injects it into
/// the user message. Single line of plain text, ≤120 chars; or the
/// literal string "no_topic" if nothing fresh.
pub const TOPIC_GENERATOR_SYSTEM_PROMPT: &str = r#"You propose ONE specific topic worth a thread (or "no_topic" if nothing fresh to say). Your inputs vary by persona — see the user message.

OUTPUT
Either a single line of plain text (the topic) — terse, ≤120 chars, no preamble, no quotation marks — OR the literal string "no_topic" if:
- you've already covered every input
- nothing in the inputs warrants a thread
- the inputs are too thin to ground a substantive post

CONSTRAINTS
- The topic must be ground-able: the writer should be able to draft a thread without inventing facts. If you can't say what specific point to make, output "no_topic".
- Avoid duplicating recent posts. Recent posts are in your inputs.
- Avoid generic topics ("AI is changing everything"). Be specific ("calibrated abstention vs forced answers in tool-use loops").
- One topic only. The thread structure is the writer's job, not yours.
"#;

/// Construct the topic generator [`AgentConfig`].
pub fn topic_generator_recipe() -> AgentConfig {
    AgentConfig {
        name: "topic_generator".to_string(),
        description:
            "Propose one specific thread topic (or 'no_topic') from pre-fetched static context."
                .to_string(),
        system_prompt: TOPIC_GENERATOR_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(512),
        reasoning_effort: Some("low".to_string()),
        ..super::stub_recipe("topic_generator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_generator_recipe_has_expected_shape() {
        let cfg = topic_generator_recipe();
        assert_eq!(cfg.name, "topic_generator");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(cfg.response_schema.is_none(), "free-form text, no schema");
    }

    #[test]
    fn topic_generator_prompt_mandates_no_topic_escape() {
        let p = TOPIC_GENERATOR_SYSTEM_PROMPT;
        assert!(
            p.contains("no_topic"),
            "prompt must offer no_topic escape hatch"
        );
        assert!(p.contains("OUTPUT"), "prompt must specify OUTPUT format");
    }

    #[test]
    fn topic_generator_prompt_bans_generic_and_demands_specificity() {
        let p = TOPIC_GENERATOR_SYSTEM_PROMPT;
        assert!(p.contains("Be specific"), "prompt must demand specificity");
        assert!(
            p.contains("ground-able") || p.contains("ground"),
            "prompt must require groundable topics"
        );
    }
}
