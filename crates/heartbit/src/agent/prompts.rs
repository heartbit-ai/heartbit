/// Multi-agent collaboration prompt appended to sub-agent system prompts.
///
/// Contains `{name}` and `{description}` placeholders that must be replaced
/// before injection using `.replace()`.
pub const MULTI_AGENT_COLLAB_PROMPT: &str = "\n\n\
--- MULTI-AGENT COLLABORATION PROTOCOL ---

You are agent `{name}` with role: {description}.

## Blackboard Protocol
- **Before starting**: read the blackboard to check what other agents have already produced.
- **During work**: write intermediate results to the blackboard so other agents can see your progress.
- **After completion**: write your final results to the blackboard under the key `agent:{name}`.

## Deduplication
- Before executing your task, verify it has not already been completed by another agent.
- If the blackboard already contains a satisfactory answer for your task, report that instead of redoing the work.

## Cross-Verification
- After producing your results, compare them against any related outputs from other agents on the blackboard.
- Flag contradictions or inconsistencies in your final output.

## Execution Loop: Perceive -> Plan -> Act -> Reflect
1. **Perceive**: read the blackboard and understand the current state of the shared workspace.
2. **Plan**: decide what steps are needed, considering what other agents have done.
3. **Act**: execute your plan using available tools.
4. **Reflect**: review your output for correctness and consistency with other agents' work.

## Result Sharing
- Write your final output to the blackboard with a clear, descriptive key.
- Include a brief summary so other agents can quickly understand your contribution.
";

/// Replace `{name}` and `{description}` placeholders in the collaboration prompt.
pub(crate) fn render_collab_prompt(name: &str, description: &str) -> String {
    MULTI_AGENT_COLLAB_PROMPT
        .replace("{name}", name)
        .replace("{description}", description)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_non_empty() {
        assert!(!MULTI_AGENT_COLLAB_PROMPT.is_empty());
    }

    #[test]
    fn prompt_under_reasonable_size() {
        // Should be under 4 KB — it's supplemental guidance, not a novel
        assert!(MULTI_AGENT_COLLAB_PROMPT.len() < 4096);
    }

    #[test]
    fn prompt_contains_expected_sections() {
        let prompt = MULTI_AGENT_COLLAB_PROMPT;
        assert!(prompt.contains("Blackboard Protocol"));
        assert!(prompt.contains("Deduplication"));
        assert!(prompt.contains("Cross-Verification"));
        assert!(prompt.contains("Perceive"));
        assert!(prompt.contains("Plan"));
        assert!(prompt.contains("Act"));
        assert!(prompt.contains("Reflect"));
        assert!(prompt.contains("Result Sharing"));
    }

    #[test]
    fn prompt_has_placeholders() {
        assert!(MULTI_AGENT_COLLAB_PROMPT.contains("{name}"));
        assert!(MULTI_AGENT_COLLAB_PROMPT.contains("{description}"));
    }

    #[test]
    fn render_replaces_placeholders() {
        let rendered = render_collab_prompt("researcher", "Finds relevant papers");
        assert!(!rendered.contains("{name}"));
        assert!(!rendered.contains("{description}"));
        assert!(rendered.contains("`researcher`"));
        assert!(rendered.contains("Finds relevant papers"));
        assert!(rendered.contains("agent:researcher"));
    }

    #[test]
    fn render_handles_special_characters_in_name() {
        let rendered = render_collab_prompt("code-analyzer", "Analyzes {code} structures");
        assert!(rendered.contains("`code-analyzer`"));
        assert!(rendered.contains("Analyzes {code} structures"));
    }
}
