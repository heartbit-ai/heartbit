//! Agent template system for reusable, composable agent configurations.
//!
//! Three layers, each independently useful:
//!
//! - **Templates**: Bundled agent presets (`template = "coder"`)
//! - **Skills**: Auto-injected domain expertise (`skills = ["rust-expert"]`)
//! - **Variables**: Prompt variable substitution (`{agent_name}`, custom vars)
//!
//! Resolution happens at the config boundary (before agent construction).
//! All downstream code sees a fully materialized `AgentConfig`.

mod merge;
pub mod registry;
pub mod skills;
pub mod variables;

use std::collections::HashMap;

use serde::Deserialize;

use crate::config::AgentConfig;
use crate::error::Error;

/// Template metadata.
#[allow(missing_docs)]
#[derive(Debug, Clone, Deserialize)]
pub struct TemplateMeta {
    /// Human-readable description of this template.
    pub description: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Parent template name for inheritance (recursive, max depth 5).
    #[serde(default)]
    pub extends: Option<String>,
}

fn default_version() -> String {
    "1.0".into()
}

/// A partial agent config where all fields are optional.
/// Used for template defaults that can be overridden by user config.
#[allow(missing_docs)]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialAgentConfig {
    pub system_prompt: Option<String>,
    pub max_tokens: Option<u32>,
    pub max_turns: Option<usize>,
    pub tool_profile: Option<String>,
    pub dangerous_tools: Option<bool>,
    pub max_identical_tool_calls: Option<u32>,
    pub max_fuzzy_identical_tool_calls: Option<u32>,
    pub max_tool_calls_per_turn: Option<u32>,
    pub reasoning_effort: Option<String>,
    pub enable_reflection: Option<bool>,
    pub tool_timeout_seconds: Option<u64>,
    pub max_tool_output_bytes: Option<usize>,
    pub run_timeout_seconds: Option<u64>,
    pub tool_output_compression_threshold: Option<usize>,
    pub max_tools_per_turn: Option<usize>,
    pub response_cache_size: Option<usize>,
    pub max_total_tokens: Option<u64>,
}

/// A bundled or user-defined agent template.
#[derive(Debug, Clone, Deserialize)]
pub struct AgentTemplate {
    /// Template metadata (name, description).
    pub meta: TemplateMeta,
    /// Default agent settings supplied by the template.
    #[serde(default)]
    pub agent: PartialAgentConfig,
}

/// Resolve an `AgentConfig` that may reference a template and/or skills
/// into a fully materialized config with no template references.
///
/// Called at the config boundary before agent construction.
/// If no template or skills are specified, returns a clone of the input unchanged.
pub fn resolve_agent_config(
    config: &AgentConfig,
    variables: &HashMap<String, String>,
) -> Result<AgentConfig, Error> {
    let mut resolved = if let Some(ref template_name) = config.template {
        // Step 1: Resolve template chain
        let template = merge::resolve_template_chain(template_name)?;
        // Step 2: Apply template defaults, merge with user config
        merge::apply_template(config, &template)
    } else {
        config.clone_config()
    };

    // Step 3: Inject skills into system_prompt
    if !config.skills.is_empty() {
        let skills_section = skills::load_skills(&config.skills)?;
        resolved.system_prompt = format!("{}{skills_section}", resolved.system_prompt);
    }

    // Step 3b: Inject the Level-1 skill catalog (progressive disclosure) for any
    // configured `skill_dirs`. Unlike `skills` (eager full-content injection),
    // this lists only `name: description` per discovered skill so the model knows
    // which skills exist and loads each one's body on demand via the `skill` tool.
    if !config.skill_dirs.is_empty() {
        let registry = crate::skill::SkillRegistry::discover(&config.skill_dirs);
        if let Some(section) = registry.system_prompt_section() {
            resolved.system_prompt = if resolved.system_prompt.is_empty() {
                section
            } else {
                format!("{}\n\n{section}", resolved.system_prompt)
            };
        }
    }

    // Step 4: Substitute variables
    let workspace = variables.get("workspace").map(|s| s.as_str());
    let all_vars =
        variables::build_variables(&resolved.name, &resolved.description, workspace, variables);
    resolved.system_prompt = variables::substitute(&resolved.system_prompt, &all_vars);

    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(name: &str) -> AgentConfig {
        AgentConfig {
            name: name.into(),
            description: "test agent".into(),
            system_prompt: String::new(),
            mcp_servers: vec![],
            a2a_agents: vec![],
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
            template: None,
            skills: vec![],
            skill_dirs: vec![],
        }
    }

    #[test]
    fn resolve_no_template_passthrough() {
        let config = test_config("plain");
        let resolved = resolve_agent_config(&config, &HashMap::new()).unwrap();
        assert_eq!(resolved.name, "plain");
        assert!(resolved.system_prompt.is_empty());
    }

    #[test]
    fn resolve_with_template() {
        let mut config = test_config("my-coder");
        config.template = Some("coder".into());
        let resolved = resolve_agent_config(&config, &HashMap::new()).unwrap();

        assert!(!resolved.system_prompt.is_empty());
        assert!(resolved.max_tokens.is_some());
        assert!(resolved.template.is_none()); // Cleared after resolution
    }

    #[test]
    fn resolve_with_skills() {
        let mut config = test_config("skilled");
        config.skills = vec!["rust-expert".into()];
        let resolved = resolve_agent_config(&config, &HashMap::new()).unwrap();

        assert!(resolved.system_prompt.contains("Loaded Skills"));
        assert!(resolved.system_prompt.contains("rust-expert"));
    }

    #[test]
    fn resolve_with_template_and_skills() {
        let mut config = test_config("full");
        config.template = Some("coder".into());
        config.skills = vec!["rust-expert".into()];
        let resolved = resolve_agent_config(&config, &HashMap::new()).unwrap();

        // Has both template prompt and skill injection
        assert!(resolved.system_prompt.contains("Loaded Skills"));
        assert!(resolved.system_prompt.len() > 100);
    }

    #[test]
    fn resolve_with_variables() {
        let mut config = test_config("var-test");
        config.system_prompt = "Hello {agent_name}, project: {project}".into();

        let mut vars = HashMap::new();
        vars.insert("project".into(), "heartbit".into());
        let resolved = resolve_agent_config(&config, &vars).unwrap();

        assert_eq!(resolved.system_prompt, "Hello var-test, project: heartbit");
    }

    #[test]
    fn resolve_variables_in_template_prompt() {
        let mut config = test_config("tmpl-var");
        config.template = Some("coder".into());
        let resolved = resolve_agent_config(&config, &HashMap::new()).unwrap();

        // Variables like {agent_name} should be substituted
        // The template prompt may or may not contain {agent_name}, but
        // the resolution should not error
        assert!(!resolved.system_prompt.is_empty());
    }

    #[test]
    fn resolve_unknown_template_error() {
        let mut config = test_config("bad");
        config.template = Some("nonexistent-template".into());
        let err = resolve_agent_config(&config, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("unknown template"));
    }

    #[test]
    fn resolve_unknown_skill_error() {
        let mut config = test_config("bad");
        config.skills = vec!["nonexistent-skill".into()];
        let err = resolve_agent_config(&config, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("unknown skill"));
    }

    #[test]
    fn resolve_user_override_wins() {
        let mut config = test_config("override");
        config.template = Some("coder".into());
        config.max_turns = Some(5);
        let resolved = resolve_agent_config(&config, &HashMap::new()).unwrap();

        assert_eq!(resolved.max_turns, Some(5));
    }

    #[test]
    fn backward_compat_no_template_no_skills() {
        let mut config = test_config("legacy");
        config.system_prompt = "I am a legacy agent.".into();
        config.max_tokens = Some(2048);
        let resolved = resolve_agent_config(&config, &HashMap::new()).unwrap();

        assert_eq!(resolved.system_prompt, "I am a legacy agent.");
        assert_eq!(resolved.max_tokens, Some(2048));
    }

    #[test]
    fn resolve_injects_skill_dir_catalog() {
        // A skill_dirs entry holding `<name>/SKILL.md` yields a Level-1 catalog
        // (name: description) in the resolved system prompt — NOT the full body.
        let tmp = tempfile::tempdir().unwrap();
        let skill_dir = tmp.path().join("pdf-tool");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: pdf-tool\ndescription: Extract text from PDFs. Use for PDF tasks.\n---\n# PDF Tool\n\nSECRET_BODY_MARKER step one.\n",
        )
        .unwrap();

        let mut config = test_config("disclosed");
        config.skill_dirs = vec![tmp.path().to_string_lossy().into_owned()];
        let resolved = resolve_agent_config(&config, &HashMap::new()).unwrap();

        assert!(
            resolved
                .system_prompt
                .contains("pdf-tool: Extract text from PDFs"),
            "catalog line missing: {}",
            resolved.system_prompt
        );
        assert!(
            resolved.system_prompt.contains("`skill` tool"),
            "should instruct the model to load skills via the skill tool"
        );
        // Progressive disclosure: the BODY must NOT be eagerly injected.
        assert!(
            !resolved.system_prompt.contains("SECRET_BODY_MARKER"),
            "skill body must NOT be injected at level 1: {}",
            resolved.system_prompt
        );
    }

    #[test]
    fn resolve_without_skill_dirs_injects_no_catalog() {
        let config = test_config("plain");
        let resolved = resolve_agent_config(&config, &HashMap::new()).unwrap();
        assert!(!resolved.system_prompt.contains("Available skills"));
    }
}
