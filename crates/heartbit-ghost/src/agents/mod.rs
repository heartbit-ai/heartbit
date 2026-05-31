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

pub mod blog_writer;
pub mod fact_check;
pub mod image_generator;
pub mod judge;
pub mod publisher;
pub mod quote_writer;
pub mod reply_writer;
pub mod repo_researcher;
pub mod researcher;
pub mod style_critic;
pub mod topic_generator;
pub mod writer;

pub use blog_writer::{BLOG_WRITER_SYSTEM_PROMPT, blog_writer_recipe};
pub use fact_check::fact_check_recipe;
pub use image_generator::{image_generator_recipe, image_search_recipe};
pub use judge::judge_recipe;
pub use publisher::publisher_recipe;
pub use quote_writer::{QUOTE_WRITER_SYSTEM_PROMPT, quote_writer_recipe};
pub use reply_writer::reply_writer_recipe;
pub use repo_researcher::repo_researcher_recipe;
pub use researcher::researcher_recipe;
pub use style_critic::style_critic_recipe;
pub use topic_generator::topic_generator_recipe;
pub use writer::writer_recipe;

/// Build a minimal `AgentConfig` with only the `name` field set; all other
/// fields are at their type defaults (`None` / empty / `false` / `Default::default()`).
///
/// Used as the `..super::stub_recipe("<name>")` fallback inside each
/// `<name>_recipe()` constructor — every recipe sets 5-7 fields explicitly
/// (name, description, system_prompt, max_turns, max_tokens, etc.) and lets
/// the remaining ~28 fields fall through to this helper's defaults.
///
/// Permanent internal helper. `AgentConfig` does not derive `Default` (the
/// omission is intentional — see `clone_config` in `heartbit_core::config::agent`),
/// so this helper is the canonical way to construct a baseline `AgentConfig`
/// inside this crate.
pub(crate) fn stub_recipe(name: &str) -> AgentConfig {
    AgentConfig {
        name: name.to_string(),
        description: String::new(),
        system_prompt: String::new(),
        template: None,
        skills: Vec::new(),
        skill_dirs: Vec::new(),
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

/// Tool set for the heartbit-rs:x persona — the existing five plus
/// `RepoInspectTool` rooted at the supplied path. Crate-private; the
/// public [`tools_for_heartbit_rs`] resolves the root from environment
/// then delegates here.
pub(crate) fn tools_for_heartbit_rs_with_root(repo_root: std::path::PathBuf) -> Vec<Arc<dyn Tool>> {
    use crate::tools::{RepoInspectTool, TwitterReplyTool, TwitterThreadTool};
    use heartbit_core::tool::builtins::ImageGenerateTool;

    let repo_inspect: Arc<dyn Tool> = match RepoInspectTool::new(&repo_root) {
        Ok(t) => Arc::new(t),
        Err(e) => {
            // If we can't construct repo_inspect at startup, the persona
            // is unusable. Fail loudly rather than silently shipping a
            // crippled tool set.
            panic!(
                "failed to construct RepoInspectTool from {repo_root:?}: {e} \
                 (set HEARTBIT_REPO_ROOT to the workspace root)"
            );
        }
    };

    // websearch / webfetch are deliberately omitted: the persona's
    // contract is "every claim grounded in the local repo", and giving
    // the researcher access to web tools causes it to default to
    // websearch (finding unrelated public Rust crates) instead of
    // calling repo_inspect on the local source. If external context is
    // ever needed for adjacent topics, add a separate per-agent tool
    // whitelist mechanism upstream rather than re-adding the temptation.
    vec![
        repo_inspect,
        Arc::new(ImageGenerateTool::new()),
        Arc::new(TwitterThreadTool::new()),
        Arc::new(TwitterReplyTool::new()),
    ]
}

/// Tool set for the heartbit-rs:x persona — same as
/// [`tools_for_heartbit_rs_with_root`] but resolves the repo root from
/// the `HEARTBIT_REPO_ROOT` env var (or `cwd()`).
pub fn tools_for_heartbit_rs() -> Vec<Arc<dyn Tool>> {
    let repo_root = std::env::var("HEARTBIT_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .expect("current_dir() failed; set HEARTBIT_REPO_ROOT explicitly")
        });
    tools_for_heartbit_rs_with_root(repo_root)
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

    #[test]
    fn tools_for_heartbit_rs_returns_four_tools_including_repo_inspect() {
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tools = tools_for_heartbit_rs_with_root(repo_root);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert!(
            names.iter().any(|n| n == "repo_inspect"),
            "repo_inspect must be in the tool list; got: {names:?}"
        );
        // websearch + webfetch are deliberately excluded so the
        // researcher can't default to web lookups and skip repo_inspect.
        assert!(
            !names.iter().any(|n| n == "websearch" || n == "webfetch"),
            "websearch / webfetch must not be present; got: {names:?}"
        );
        assert_eq!(
            tools.len(),
            4,
            "expected repo_inspect + image_generate + twitter_thread + twitter_reply"
        );
    }
}
