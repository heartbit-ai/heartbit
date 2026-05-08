//! Sub-agent recipes for the heartbit-ghost X (Twitter) persona.
//!
//! 7 recipes (per umbrella spec §5):
//!
//! - [`researcher`] — websearch + webfetch (reusable beyond Twitter)
//! - [`writer`] — style-conditioned generation, no tools (reusable)
//! - [`style_critic`] — voice match + AI-tell detection, no tools (partial)
//! - [`judge`] — multi-candidate ranking, no tools (reusable)
//! - [`fact_check`] — claim verification, no tools (reusable)
//! - [`image_generator`] — image_generate (reusable)
//! - [`publisher`] — twitter_thread + twitter_reply (Twitter-specific)
//!
//! Each recipe is a `pub fn <name>_recipe() -> AgentConfig` constructed
//! inline from a `&'static str` system prompt constant in the same file.
//!
//! [`tools_for_persona`] returns the 5 tool instances those recipes
//! reference, ready for inclusion in [`heartbit_core::PersonaExpansion::tools`].

use std::sync::Arc;

use heartbit_core::config::AgentConfig;
use heartbit_core::tool::Tool;

pub mod fact_check;
pub mod image_generator;
pub mod judge;
pub mod publisher;
pub mod researcher;
pub mod style_critic;
pub mod writer;

pub use fact_check::fact_check_recipe;
pub use image_generator::image_generator_recipe;
pub use judge::judge_recipe;
pub use publisher::publisher_recipe;
pub use researcher::researcher_recipe;
pub use style_critic::style_critic_recipe;
pub use writer::writer_recipe;

/// Build a stub [`AgentConfig`] with only `name` set.
///
/// `AgentConfig` does not implement `Default` (the upstream struct
/// intentionally omits the derive — see `clone_config` in
/// `heartbit_core::config::agent`), so we enumerate every field
/// explicitly. Task 2 will replace each recipe body with prompts
/// and per-recipe knobs; this helper is a temporary scaffold.
pub(crate) fn stub_recipe(name: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        description: String::new(),
        system_prompt: String::new(),
        template: None,
        skills: Vec::new(),
        mcp_servers: Vec::new(),
        a2a_agents: Vec::new(),
        context_strategy: None,
        summarize_threshold: None,
        tool_timeout_seconds: None,
        max_tool_output_bytes: None,
        max_turns: None,
        max_tokens: None,
        response_schema: None,
        run_timeout_seconds: None,
        provider: None,
        reasoning_effort: None,
        enable_reflection: None,
        tool_output_compression_threshold: None,
        max_tools_per_turn: None,
        tool_profile: None,
        max_identical_tool_calls: None,
        max_fuzzy_identical_tool_calls: None,
        max_tool_calls_per_turn: None,
        session_prune: None,
        recursive_summarization: None,
        reflection_threshold: None,
        consolidate_on_exit: None,
        max_total_tokens: None,
        guardrails: None,
        response_cache_size: None,
        mcp_resources: Default::default(),
        dangerous_tools: false,
        audit_mode: None,
        builtin_tools: None,
    }
}

/// Construct the 5 tool instances the persona's 7 recipes reference.
///
/// Returned in declared order: `websearch`, `webfetch`, `image_generate`,
/// `twitter_thread`, `twitter_reply`. (See spec §2 for the `twitter_post`
/// scope cut.)
pub fn tools_for_persona() -> Vec<Arc<dyn Tool>> {
    use crate::tools::{TwitterReplyTool, TwitterThreadTool};
    use heartbit_core::tool::builtins::{ImageGenerateTool, WebFetchTool, WebSearchTool};

    vec![
        Arc::new(WebSearchTool::new()),
        Arc::new(WebFetchTool::new()),
        Arc::new(ImageGenerateTool::new()),
        Arc::new(TwitterThreadTool::new()),
        Arc::new(TwitterReplyTool::new()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_for_persona_returns_five_distinct_tools_in_declared_order() {
        let tools = tools_for_persona();
        assert_eq!(tools.len(), 5);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert_eq!(
            names,
            vec![
                "websearch".to_string(),
                "webfetch".to_string(),
                "image_generate".to_string(),
                "twitter_thread".to_string(),
                "twitter_reply".to_string(),
            ]
        );
    }
}
