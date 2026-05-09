//! Repo-grounded researcher sub-agent — backs the heartbit-rs:x persona.
//! Uses the `repo_inspect` builtin as primary substance; `websearch` /
//! `webfetch` are available for external context only.

use heartbit_core::config::AgentConfig;

/// System prompt for the repo-grounded researcher.
pub const REPO_RESEARCHER_SYSTEM_PROMPT: &str = r#"You are a research analyst for a Rust agent framework called heartbit-rs. Given a feature name or topic, find the substance: the canonical file where it lives, the key types, a representative code excerpt, and a one-sentence payoff for someone reading about it.

PROCESS
1. If the user named a feature in the menu (e.g., "tool_trait", "memory_bm25"), call `repo_inspect` with `op: "feature_demo"` and read the canonical_file via `op: "read_file"`.
2. If the user gave a free-form topic, call `repo_inspect` with `op: "list_features"` first to see what's available, then either pick the closest one or use `op: "grep_repo"` to locate definitions yourself.
3. Read at most 2-3 files; pick the smallest excerpt that demonstrates the feature (typically a trait definition, a struct + 1-2 methods, or a single public function). Aim for ≤30 lines per excerpt.
4. `websearch` / `webfetch` are available ONLY for OPTIONAL external context (e.g. "how this compares to LangGraph", "the original paper"). They are NEVER the primary substance. The substance always comes from the repo.

OUTPUT FORMAT (free-form text, no JSON):
- Feature name + 1-sentence framing.
- Canonical file path (e.g., `crates/heartbit-core/src/tool/mod.rs`).
- Key types: comma-separated list.
- Code excerpt: ≤30 lines, fenced ```rust block, with the line numbers if from a range.
- Payoff: 1-2 sentences on what this enables for someone using the framework.
- Optional: 1-2 external context bullets with sources.

Do NOT write the post. The writer composes. Do NOT speculate beyond what the files show."#;

/// Construct the repo-grounded researcher [`AgentConfig`].
pub fn repo_researcher_recipe() -> AgentConfig {
    AgentConfig {
        name: "repo_researcher".to_string(),
        description:
            "Find substance about a heartbit-rs feature: canonical file, code excerpt, payoff."
                .to_string(),
        system_prompt: REPO_RESEARCHER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(25),
        max_tokens: Some(4096),
        reasoning_effort: Some("medium".to_string()),
        ..super::stub_recipe("repo_researcher")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_researcher_recipe_has_expected_shape() {
        let cfg = repo_researcher_recipe();
        assert_eq!(cfg.name, "repo_researcher");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(25));
        assert_eq!(cfg.max_tokens, Some(4096));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(cfg.response_schema.is_none(), "free-form digest, no schema");
    }

    #[test]
    fn repo_researcher_prompt_routes_users_to_repo_inspect_first() {
        let p = REPO_RESEARCHER_SYSTEM_PROMPT;
        assert!(
            p.contains("repo_inspect"),
            "prompt mentions the primary tool"
        );
        assert!(
            p.contains("feature_demo") && p.contains("list_features"),
            "prompt names the menu ops"
        );
    }

    #[test]
    fn repo_researcher_prompt_explicitly_demotes_websearch() {
        let p = REPO_RESEARCHER_SYSTEM_PROMPT;
        assert!(
            p.contains("OPTIONAL") || p.contains("optional"),
            "prompt must mark websearch as optional"
        );
        assert!(
            p.contains("never the primary substance") || p.contains("NEVER the primary substance"),
            "prompt must explicitly demote websearch from primary substance"
        );
    }
}
