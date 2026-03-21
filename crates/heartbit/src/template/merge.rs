use std::collections::HashSet;

use crate::config::AgentConfig;
use crate::error::Error;

use super::registry::resolve_template;
use super::{AgentTemplate, PartialAgentConfig};

/// Maximum depth for template `extends` chains (prevents infinite recursion).
const MAX_EXTENDS_DEPTH: usize = 5;

/// Resolve the full template chain for a given template name.
///
/// Walks the `extends` chain recursively (max depth 5), detects circular
/// references, and returns the fully merged `PartialAgentConfig`.
pub(super) fn resolve_template_chain(name: &str) -> Result<AgentTemplate, Error> {
    let mut visited = HashSet::new();
    resolve_recursive(name, &mut visited, 0)
}

fn resolve_recursive(
    name: &str,
    visited: &mut HashSet<String>,
    depth: usize,
) -> Result<AgentTemplate, Error> {
    if depth > MAX_EXTENDS_DEPTH {
        return Err(Error::Config(format!(
            "template extends chain exceeds maximum depth of {MAX_EXTENDS_DEPTH}"
        )));
    }

    if !visited.insert(name.to_string()) {
        return Err(Error::Config(format!(
            "circular template reference detected: '{name}'"
        )));
    }

    let template = resolve_template(name)?;

    if let Some(ref parent_name) = template.meta.extends {
        let parent = resolve_recursive(parent_name, visited, depth + 1)?;
        // Merge: parent → child (child wins for each field)
        let merged_agent = merge_partial(&parent.agent, &template.agent);
        Ok(AgentTemplate {
            meta: template.meta,
            agent: merged_agent,
        })
    } else {
        Ok(template)
    }
}

/// Merge two `PartialAgentConfig`s: parent → child (child wins).
fn merge_partial(parent: &PartialAgentConfig, child: &PartialAgentConfig) -> PartialAgentConfig {
    PartialAgentConfig {
        system_prompt: child
            .system_prompt
            .clone()
            .or_else(|| parent.system_prompt.clone()),
        max_tokens: child.max_tokens.or(parent.max_tokens),
        max_turns: child.max_turns.or(parent.max_turns),
        tool_profile: child
            .tool_profile
            .clone()
            .or_else(|| parent.tool_profile.clone()),
        dangerous_tools: child.dangerous_tools.or(parent.dangerous_tools),
        max_identical_tool_calls: child
            .max_identical_tool_calls
            .or(parent.max_identical_tool_calls),
        max_fuzzy_identical_tool_calls: child
            .max_fuzzy_identical_tool_calls
            .or(parent.max_fuzzy_identical_tool_calls),
        reasoning_effort: child
            .reasoning_effort
            .clone()
            .or_else(|| parent.reasoning_effort.clone()),
        enable_reflection: child.enable_reflection.or(parent.enable_reflection),
        tool_timeout_seconds: child.tool_timeout_seconds.or(parent.tool_timeout_seconds),
        max_tool_output_bytes: child.max_tool_output_bytes.or(parent.max_tool_output_bytes),
        run_timeout_seconds: child.run_timeout_seconds.or(parent.run_timeout_seconds),
        tool_output_compression_threshold: child
            .tool_output_compression_threshold
            .or(parent.tool_output_compression_threshold),
        max_tools_per_turn: child.max_tools_per_turn.or(parent.max_tools_per_turn),
        response_cache_size: child.response_cache_size.or(parent.response_cache_size),
        max_total_tokens: child.max_total_tokens.or(parent.max_total_tokens),
    }
}

/// Apply template defaults to an `AgentConfig`.
///
/// Template fields are used as defaults — explicit `AgentConfig` values take
/// precedence (except `system_prompt` which uses append/replace semantics).
pub(super) fn apply_template(config: &AgentConfig, template: &AgentTemplate) -> AgentConfig {
    let template_prompt = template.agent.system_prompt.as_deref().unwrap_or("");

    // System prompt semantics:
    // - User prompt starting with `!` → replaces template prompt entirely
    // - Empty user prompt → use template prompt as-is
    // - Non-empty user prompt → append to template prompt
    let system_prompt = if config.system_prompt.starts_with('!') {
        config.system_prompt[1..].to_string()
    } else if config.system_prompt.is_empty() {
        template_prompt.to_string()
    } else {
        format!("{template_prompt}\n\n{}", config.system_prompt)
    };

    AgentConfig {
        name: config.name.clone(),
        description: config.description.clone(),
        system_prompt,
        mcp_servers: config.mcp_servers.clone(),
        a2a_agents: config.a2a_agents.clone(),
        context_strategy: config.context_strategy.clone(),
        summarize_threshold: config.summarize_threshold,
        tool_timeout_seconds: config
            .tool_timeout_seconds
            .or(template.agent.tool_timeout_seconds),
        max_tool_output_bytes: config
            .max_tool_output_bytes
            .or(template.agent.max_tool_output_bytes),
        max_turns: config.max_turns.or(template.agent.max_turns),
        max_tokens: config.max_tokens.or(template.agent.max_tokens),
        response_schema: config.response_schema.clone(),
        run_timeout_seconds: config
            .run_timeout_seconds
            .or(template.agent.run_timeout_seconds),
        provider: config.provider.clone(),
        reasoning_effort: config
            .reasoning_effort
            .clone()
            .or_else(|| template.agent.reasoning_effort.clone()),
        enable_reflection: config
            .enable_reflection
            .or(template.agent.enable_reflection),
        tool_output_compression_threshold: config
            .tool_output_compression_threshold
            .or(template.agent.tool_output_compression_threshold),
        max_tools_per_turn: config
            .max_tools_per_turn
            .or(template.agent.max_tools_per_turn),
        tool_profile: config
            .tool_profile
            .clone()
            .or_else(|| template.agent.tool_profile.clone()),
        max_identical_tool_calls: config
            .max_identical_tool_calls
            .or(template.agent.max_identical_tool_calls),
        max_fuzzy_identical_tool_calls: config
            .max_fuzzy_identical_tool_calls
            .or(template.agent.max_fuzzy_identical_tool_calls),
        session_prune: config.session_prune.clone(),
        recursive_summarization: config.recursive_summarization,
        reflection_threshold: config.reflection_threshold,
        consolidate_on_exit: config.consolidate_on_exit,
        max_total_tokens: config.max_total_tokens.or(template.agent.max_total_tokens),
        guardrails: config.guardrails.clone(),
        response_cache_size: config
            .response_cache_size
            .or(template.agent.response_cache_size),
        mcp_resources: config.mcp_resources,
        dangerous_tools: if config.dangerous_tools {
            true
        } else {
            template.agent.dangerous_tools.unwrap_or(false)
        },
        audit_mode: config.audit_mode.clone(),
        builtin_tools: config.builtin_tools.clone(),
        // Template-only fields: pass through
        template: None,     // Already resolved
        skills: Vec::new(), // Already injected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config(name: &str) -> AgentConfig {
        AgentConfig {
            name: name.into(),
            description: "test".into(),
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
        }
    }

    #[test]
    fn resolve_coder_chain() {
        let template = resolve_template_chain("coder").unwrap();
        assert!(
            !template
                .agent
                .system_prompt
                .as_deref()
                .unwrap_or("")
                .is_empty()
        );
    }

    #[test]
    fn apply_template_uses_template_defaults() {
        let template = resolve_template_chain("coder").unwrap();
        let config = minimal_config("test");
        let resolved = apply_template(&config, &template);

        // System prompt comes from template
        assert!(!resolved.system_prompt.is_empty());
        // max_tokens comes from template
        assert!(resolved.max_tokens.is_some());
    }

    #[test]
    fn apply_template_user_overrides_win() {
        let template = resolve_template_chain("coder").unwrap();
        let mut config = minimal_config("test");
        config.max_turns = Some(5);
        let resolved = apply_template(&config, &template);

        assert_eq!(resolved.max_turns, Some(5));
    }

    #[test]
    fn apply_template_system_prompt_append() {
        let template = resolve_template_chain("coder").unwrap();
        let mut config = minimal_config("test");
        config.system_prompt = "Focus on Rust.".into();
        let resolved = apply_template(&config, &template);

        // Template prompt + user prompt
        assert!(resolved.system_prompt.contains("Focus on Rust."));
        assert!(resolved.system_prompt.len() > 20); // Has template content too
    }

    #[test]
    fn apply_template_system_prompt_replace() {
        let template = resolve_template_chain("coder").unwrap();
        let mut config = minimal_config("test");
        config.system_prompt = "!Custom prompt only.".into();
        let resolved = apply_template(&config, &template);

        assert_eq!(resolved.system_prompt, "Custom prompt only.");
    }

    #[test]
    fn apply_template_empty_prompt_uses_template() {
        let template = resolve_template_chain("coder").unwrap();
        let config = minimal_config("test");
        let resolved = apply_template(&config, &template);

        // Uses template prompt directly
        assert!(!resolved.system_prompt.is_empty());
    }

    #[test]
    fn merge_partial_child_wins() {
        let parent = PartialAgentConfig {
            max_tokens: Some(4096),
            max_turns: Some(10),
            dangerous_tools: Some(false),
            ..Default::default()
        };
        let child = PartialAgentConfig {
            max_tokens: Some(8192),
            ..Default::default()
        };
        let merged = merge_partial(&parent, &child);
        assert_eq!(merged.max_tokens, Some(8192)); // child wins
        assert_eq!(merged.max_turns, Some(10)); // parent fallback
        assert_eq!(merged.dangerous_tools, Some(false)); // parent fallback
    }

    #[test]
    fn no_template_passthrough() {
        let config = minimal_config("test");
        assert!(config.template.is_none());
        // Without a template, config should work as-is (backward compat)
    }
}
