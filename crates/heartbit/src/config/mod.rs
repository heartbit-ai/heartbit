mod agent;
mod daemon;
mod guardrails;
mod memory;
mod provider;
mod sensor;

// Re-export everything so the public API doesn't change.
pub use agent::{
    AgentConfig, AgentProviderConfig, ContextStrategyConfig, DispatchMode, McpResourceMode,
    McpServerEntry, OrchestratorConfig, SessionPruneConfigToml, SpawnConfig,
};
pub use daemon::{
    ActiveHoursConfig, AuthConfig, DaemonConfig, DaemonMcpServerConfig, DaemonMemoryConfig,
    HeartbitPulseConfig, KafkaConfig, MetricsConfig, ScheduleEntry, TokenExchangeConfig, WsConfig,
};
pub use guardrails::{
    ActionBudgetConfig, ActionBudgetRuleConfig, BehavioralConfig, BehavioralRuleConfig,
    GuardrailsConfig, InjectionConfig, InputConstraintConfig, LlmJudgeConfig, PiiConfig,
    SecretPatternConfig, SecretScanConfig, ToolPolicyConfig, ToolPolicyRuleConfig,
};
pub use memory::{
    EmbeddingConfig, KnowledgeConfig, KnowledgeSourceConfig, LspConfig, MemoryConfig,
    RestateConfig, TelemetryConfig, WorkspaceConfig,
};
pub use provider::{
    CascadeConfig, CascadeGateConfig, CascadeTierConfig, ProviderConfig, RetryProviderConfig,
};
pub use sensor::{
    SalienceConfig, SensorConfig, SensorRoutingConfig, SensorSourceConfig, StoryCorrelationConfig,
    TokenBudgetConfig,
};

pub use crate::agent::routing::RoutingMode;

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::agent::permission::PermissionRule;
use crate::agent::tool_filter::ToolProfile;
use crate::llm::types::ReasoningEffort;

/// Known builtin tool names for validation of `builtin_tools` allowlists.
pub const KNOWN_BUILTINS: &[&str] = &[
    "bash",
    "read",
    "write",
    "edit",
    "grep",
    "glob",
    "list",
    "patch",
    "webfetch",
    "websearch",
    "image_generate",
    "tts",
    "skill",
    "todoread",
    "todowrite",
    "question",
    "twitter_post",
    "todo_manage",
];

/// Sensory modality — the type of information a sensor captures.
///
/// Defined in config so it's always available for TOML deserialization
/// even when the `sensor` feature is disabled. Re-exported from `sensor` module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorModality {
    /// Email body, RSS content, chat messages.
    Text,
    /// Photos, screenshots, documents-as-images.
    Image,
    /// Voice notes, calls, podcasts.
    Audio,
    /// Weather JSON, GPS coordinates, API responses.
    Structured,
}

impl std::fmt::Display for SensorModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensorModality::Text => write!(f, "text"),
            SensorModality::Image => write!(f, "image"),
            SensorModality::Audio => write!(f, "audio"),
            SensorModality::Structured => write!(f, "structured"),
        }
    }
}

pub use heartbit_core::types::TrustLevel;

/// Parse a workflow type string into the enum.
pub fn parse_workflow_type(s: &str) -> Result<crate::agent::workflow::WorkflowType, Error> {
    use crate::agent::workflow::WorkflowType;
    match s.to_lowercase().as_str() {
        "dag" => Ok(WorkflowType::Dag),
        "sequential" => Ok(WorkflowType::Sequential),
        "parallel" => Ok(WorkflowType::Parallel),
        "loop" => Ok(WorkflowType::Loop),
        "debate" => Ok(WorkflowType::Debate),
        "voting" => Ok(WorkflowType::Voting),
        "mixture" => Ok(WorkflowType::Mixture),
        _ => Err(Error::Config(format!(
            "invalid workflow_type '{}': must be dag, sequential, parallel, loop, debate, voting, or mixture",
            s
        ))),
    }
}

/// Parse a tool profile string into the enum.
pub fn parse_tool_profile(s: &str) -> Result<ToolProfile, Error> {
    match s.to_lowercase().as_str() {
        "conversational" => Ok(ToolProfile::Conversational),
        "standard" => Ok(ToolProfile::Standard),
        "full" => Ok(ToolProfile::Full),
        _ => Err(Error::Config(format!(
            "invalid tool_profile '{}': must be conversational, standard, or full",
            s
        ))),
    }
}

/// Parse a reasoning effort string into the enum.
pub fn parse_reasoning_effort(s: &str) -> Result<ReasoningEffort, Error> {
    match s.to_lowercase().as_str() {
        "high" => Ok(ReasoningEffort::High),
        "medium" => Ok(ReasoningEffort::Medium),
        "low" => Ok(ReasoningEffort::Low),
        "none" => Ok(ReasoningEffort::None),
        _ => Err(Error::Config(format!(
            "invalid reasoning_effort '{}': must be high, medium, low, or none",
            s
        ))),
    }
}

/// Shared default helper used by multiple submodules.
fn default_true() -> bool {
    true
}

/// Top-level configuration loaded from `heartbit.toml`.
#[derive(Debug, Deserialize)]
pub struct HeartbitConfig {
    #[serde(default)]
    pub provider: ProviderConfig,
    #[serde(default)]
    pub orchestrator: OrchestratorConfig,
    #[serde(default)]
    pub agents: Vec<AgentConfig>,
    /// Custom variables for template prompt substitution.
    /// Available as `{var_name}` in system_prompt, template prompts, and skill content.
    #[serde(default)]
    pub variables: std::collections::HashMap<String, String>,
    pub restate: Option<RestateConfig>,
    pub telemetry: Option<TelemetryConfig>,
    pub memory: Option<MemoryConfig>,
    pub knowledge: Option<KnowledgeConfig>,
    /// Declarative permission rules applied to all agents.
    /// Rules are evaluated in order — first match wins.
    #[serde(default)]
    pub permissions: Vec<PermissionRule>,
    /// Optional LSP integration for diagnostics after file-modifying tools.
    pub lsp: Option<LspConfig>,
    /// Daemon mode configuration for Kafka-backed long-running execution.
    pub daemon: Option<DaemonConfig>,
    /// Optional workspace configuration for agent home directories.
    pub workspace: Option<WorkspaceConfig>,
    /// Guardrails configuration for injection detection, PII, and tool policies.
    #[serde(default)]
    pub guardrails: Option<GuardrailsConfig>,
}

impl HeartbitConfig {
    /// Parse a TOML string into a `HeartbitConfig`.
    pub fn from_toml(content: &str) -> Result<Self, Error> {
        let config: Self = toml::from_str(content).map_err(|e| Error::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Read and parse a TOML config file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, Error> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("failed to read {}: {e}", path.display())))?;
        Self::from_toml(&content)
    }

    fn validate(&self) -> Result<(), Error> {
        // Validate top-level provider (skip when running as cloud-delegated runtime
        // where per-request provider keys are used instead of a global config).
        let daemon_only = self.daemon.is_some() && self.agents.is_empty();
        if !daemon_only {
            if self.provider.name.is_empty() {
                return Err(Error::Config("provider.name must not be empty".into()));
            }
            if self.provider.model.is_empty() {
                return Err(Error::Config("provider.model must not be empty".into()));
            }
        }
        if self.orchestrator.max_turns == 0 {
            return Err(Error::Config(
                "orchestrator.max_turns must be at least 1".into(),
            ));
        }
        if self.orchestrator.max_tokens == 0 {
            return Err(Error::Config(
                "orchestrator.max_tokens must be at least 1".into(),
            ));
        }
        // Validate orchestrator context strategy
        match &self.orchestrator.context_strategy {
            Some(ContextStrategyConfig::SlidingWindow { max_tokens }) if *max_tokens == 0 => {
                return Err(Error::Config(
                    "orchestrator.context_strategy.max_tokens must be at least 1".into(),
                ));
            }
            Some(ContextStrategyConfig::Summarize { threshold }) if *threshold == 0 => {
                return Err(Error::Config(
                    "orchestrator.context_strategy.threshold must be at least 1".into(),
                ));
            }
            _ => {}
        }
        if self.orchestrator.summarize_threshold == Some(0) {
            return Err(Error::Config(
                "orchestrator.summarize_threshold must be at least 1".into(),
            ));
        }
        if matches!(
            self.orchestrator.context_strategy,
            Some(ContextStrategyConfig::Summarize { .. })
                | Some(ContextStrategyConfig::SlidingWindow { .. })
        ) && self.orchestrator.summarize_threshold.is_some()
        {
            return Err(Error::Config(
                "orchestrator: cannot set both context_strategy \
                 and summarize_threshold; use one or the other"
                    .into(),
            ));
        }
        if self.orchestrator.tool_timeout_seconds == Some(0) {
            return Err(Error::Config(
                "orchestrator.tool_timeout_seconds must be at least 1".into(),
            ));
        }
        if self.orchestrator.max_tool_output_bytes == Some(0) {
            return Err(Error::Config(
                "orchestrator.max_tool_output_bytes must be at least 1".into(),
            ));
        }
        if self.orchestrator.run_timeout_seconds == Some(0) {
            return Err(Error::Config(
                "orchestrator.run_timeout_seconds must be at least 1".into(),
            ));
        }
        if let Some(ref effort) = self.orchestrator.reasoning_effort {
            parse_reasoning_effort(effort)?;
        }
        if self.orchestrator.tool_output_compression_threshold == Some(0) {
            return Err(Error::Config(
                "orchestrator.tool_output_compression_threshold must be at least 1".into(),
            ));
        }
        if self.orchestrator.max_tools_per_turn == Some(0) {
            return Err(Error::Config(
                "orchestrator.max_tools_per_turn must be at least 1".into(),
            ));
        }
        if let Some(ref profile) = self.orchestrator.tool_profile {
            parse_tool_profile(profile)?;
        }
        if self.orchestrator.max_identical_tool_calls == Some(0) {
            return Err(Error::Config(
                "orchestrator.max_identical_tool_calls must be at least 1".into(),
            ));
        }
        if self.orchestrator.max_fuzzy_identical_tool_calls == Some(0) {
            return Err(Error::Config(
                "orchestrator.max_fuzzy_identical_tool_calls must be at least 1".into(),
            ));
        }

        // Validate spawn config
        if let Some(ref spawn) = self.orchestrator.spawn {
            if spawn.max_spawned_agents == 0 {
                return Err(Error::Config(
                    "orchestrator.spawn.max_spawned_agents must be at least 1".into(),
                ));
            }
            if spawn.max_turns == 0 {
                return Err(Error::Config(
                    "orchestrator.spawn.max_turns must be at least 1".into(),
                ));
            }
            if spawn.max_tokens == 0 {
                return Err(Error::Config(
                    "orchestrator.spawn.max_tokens must be at least 1".into(),
                ));
            }
            if spawn.max_total_tokens == 0 {
                return Err(Error::Config(
                    "orchestrator.spawn.max_total_tokens must be at least 1".into(),
                ));
            }
            if spawn.tool_allowlist.is_empty() {
                tracing::warn!(
                    "orchestrator.spawn.tool_allowlist is empty — spawned agents will be reasoning-only"
                );
            }
        }

        // Validate cascade config: enabled requires at least one tier
        if let Some(ref cascade) = self.provider.cascade
            && cascade.enabled
            && cascade.tiers.is_empty()
        {
            return Err(Error::Config(
                "provider.cascade.enabled is true but no tiers are configured; \
                 add at least one [[provider.cascade.tiers]] entry"
                    .into(),
            ));
        }
        // Validate cascade tier models are non-empty
        if let Some(ref cascade) = self.provider.cascade {
            for (i, tier) in cascade.tiers.iter().enumerate() {
                if tier.model.is_empty() {
                    return Err(Error::Config(format!(
                        "provider.cascade.tiers[{i}].model must not be empty"
                    )));
                }
            }
        }

        // Validate retry config: base_delay_ms <= max_delay_ms
        if let Some(ref retry) = self.provider.retry
            && retry.base_delay_ms > retry.max_delay_ms
        {
            return Err(Error::Config(format!(
                "provider.retry.base_delay_ms ({}) must not exceed max_delay_ms ({})",
                retry.base_delay_ms, retry.max_delay_ms
            )));
        }

        // Ensure agent names are unique
        let mut seen = std::collections::HashSet::new();
        for agent in &self.agents {
            if agent.name.is_empty() {
                return Err(Error::Config("agent name must not be empty".into()));
            }
            if !seen.insert(&agent.name) {
                return Err(Error::Config(format!(
                    "duplicate agent name: '{}'",
                    agent.name
                )));
            }
            // Validate context strategy max_tokens > 0
            match &agent.context_strategy {
                Some(ContextStrategyConfig::SlidingWindow { max_tokens }) if *max_tokens == 0 => {
                    return Err(Error::Config(format!(
                        "agent '{}': context_strategy.max_tokens must be at least 1",
                        agent.name
                    )));
                }
                Some(ContextStrategyConfig::Summarize { threshold }) if *threshold == 0 => {
                    return Err(Error::Config(format!(
                        "agent '{}': context_strategy.threshold must be at least 1",
                        agent.name
                    )));
                }
                _ => {}
            }
            if agent.max_turns == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': max_turns must be at least 1",
                    agent.name
                )));
            }
            if agent.max_tokens == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': max_tokens must be at least 1",
                    agent.name
                )));
            }
            if agent.tool_timeout_seconds == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': tool_timeout_seconds must be at least 1",
                    agent.name
                )));
            }
            if agent.max_tool_output_bytes == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': max_tool_output_bytes must be at least 1",
                    agent.name
                )));
            }
            if agent.run_timeout_seconds == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': run_timeout_seconds must be at least 1",
                    agent.name
                )));
            }
            // Validate per-agent provider config
            if let Some(ref p) = agent.provider {
                if p.name.is_empty() {
                    return Err(Error::Config(format!(
                        "agent '{}': provider.name must not be empty",
                        agent.name
                    )));
                }
                if p.model.is_empty() {
                    return Err(Error::Config(format!(
                        "agent '{}': provider.model must not be empty",
                        agent.name
                    )));
                }
            }
            // Validate MCP server entries
            for (i, entry) in agent.mcp_servers.iter().enumerate() {
                if entry.url().is_empty() {
                    return Err(Error::Config(format!(
                        "agent '{}': mcp_servers[{i}].url must not be empty",
                        agent.name
                    )));
                }
            }
            // Validate A2A agent entries
            for (i, entry) in agent.a2a_agents.iter().enumerate() {
                if entry.url().is_empty() {
                    return Err(Error::Config(format!(
                        "agent '{}': a2a_agents[{i}].url must not be empty",
                        agent.name
                    )));
                }
            }
            if agent.summarize_threshold == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': summarize_threshold must be at least 1",
                    agent.name
                )));
            }
            if matches!(
                agent.context_strategy,
                Some(ContextStrategyConfig::Summarize { .. })
                    | Some(ContextStrategyConfig::SlidingWindow { .. })
            ) && agent.summarize_threshold.is_some()
            {
                return Err(Error::Config(format!(
                    "agent '{}': cannot set both context_strategy and summarize_threshold; \
                     use one or the other",
                    agent.name
                )));
            }
            if let Some(ref effort) = agent.reasoning_effort {
                parse_reasoning_effort(effort).map_err(|_| {
                    Error::Config(format!(
                        "agent '{}': invalid reasoning_effort '{}': must be high, medium, low, or none",
                        agent.name, effort
                    ))
                })?;
            }
            if agent.tool_output_compression_threshold == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': tool_output_compression_threshold must be at least 1",
                    agent.name
                )));
            }
            if agent.max_tools_per_turn == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': max_tools_per_turn must be at least 1",
                    agent.name
                )));
            }
            if agent.max_identical_tool_calls == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': max_identical_tool_calls must be at least 1",
                    agent.name
                )));
            }
            if agent.max_fuzzy_identical_tool_calls == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': max_fuzzy_identical_tool_calls must be at least 1",
                    agent.name
                )));
            }
            if agent.max_total_tokens == Some(0) {
                return Err(Error::Config(format!(
                    "agent '{}': max_total_tokens must be at least 1",
                    agent.name
                )));
            }
            if let Some(ref profile) = agent.tool_profile {
                parse_tool_profile(profile).map_err(|_| {
                    Error::Config(format!(
                        "agent '{}': invalid tool_profile '{}': must be conversational, standard, or full",
                        agent.name, profile
                    ))
                })?;
            }
            if let Some(ref bt) = agent.builtin_tools {
                for name in bt {
                    if !KNOWN_BUILTINS.contains(&name.as_str()) {
                        return Err(Error::Config(format!(
                            "agent '{}': unknown builtin tool '{}'; known builtins: {}",
                            agent.name,
                            name,
                            KNOWN_BUILTINS.join(", ")
                        )));
                    }
                }
            }
        }

        // Validate knowledge config
        if let Some(ref knowledge) = self.knowledge {
            if knowledge.chunk_size == 0 {
                return Err(Error::Config(
                    "knowledge.chunk_size must be at least 1".into(),
                ));
            }
            if knowledge.chunk_overlap >= knowledge.chunk_size {
                return Err(Error::Config(format!(
                    "knowledge.chunk_overlap ({}) must be less than chunk_size ({})",
                    knowledge.chunk_overlap, knowledge.chunk_size
                )));
            }
        }

        // Validate daemon config
        if let Some(ref daemon) = self.daemon {
            if daemon.max_concurrent_tasks == 0 {
                return Err(Error::Config(
                    "daemon.max_concurrent_tasks must be at least 1".into(),
                ));
            }
            if let Some(ref kafka) = daemon.kafka {
                if kafka.brokers.is_empty() {
                    return Err(Error::Config(
                        "daemon.kafka.brokers must not be empty".into(),
                    ));
                }
                if kafka.consumer_group.is_empty() {
                    return Err(Error::Config(
                        "daemon.kafka.consumer_group must not be empty".into(),
                    ));
                }
                if kafka.commands_topic.is_empty() {
                    return Err(Error::Config(
                        "daemon.kafka.commands_topic must not be empty".into(),
                    ));
                }
                if kafka.events_topic.is_empty() {
                    return Err(Error::Config(
                        "daemon.kafka.events_topic must not be empty".into(),
                    ));
                }
            }
            // Validate auth config
            if let Some(ref auth) = daemon.auth {
                if auth.bearer_tokens.is_empty() && auth.jwks_url.is_none() {
                    return Err(Error::Config(
                        "daemon.auth requires at least bearer_tokens or jwks_url".into(),
                    ));
                }
                for (i, token) in auth.bearer_tokens.iter().enumerate() {
                    if token.is_empty() {
                        return Err(Error::Config(format!(
                            "daemon.auth.bearer_tokens[{i}] must not be empty"
                        )));
                    }
                }
                if let Some(ref url) = auth.jwks_url
                    && url.is_empty()
                {
                    return Err(Error::Config(
                        "daemon.auth.jwks_url must not be empty".into(),
                    ));
                }
                if let Some(ref te) = auth.token_exchange {
                    if te.exchange_url.is_empty() {
                        return Err(Error::Config(
                            "daemon.auth.token_exchange.exchange_url must not be empty".into(),
                        ));
                    }
                    if te.client_id.is_empty() {
                        return Err(Error::Config(
                            "daemon.auth.token_exchange.client_id must not be empty".into(),
                        ));
                    }
                    if te.client_secret.is_empty() {
                        return Err(Error::Config(
                            "daemon.auth.token_exchange.client_secret must not be empty".into(),
                        ));
                    }
                    if te.tenant_id.is_none() && te.agent_token.is_empty() {
                        return Err(Error::Config(
                            "daemon.auth.token_exchange: set tenant_id for auto-fetch, or provide a static agent_token".into(),
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::guardrails::default_pii_detectors;
    use super::*;

    #[test]
    fn parse_full_config() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_turns = 15
max_tokens = 8192

[[agents]]
name = "researcher"
description = "Research specialist"
system_prompt = "You are a research specialist."

[[agents]]
name = "coder"
description = "Coding expert"
system_prompt = "You are a coding expert."
mcp_servers = ["http://localhost:8000/mcp"]

[restate]
endpoint = "http://localhost:9070"
"#;

        let config = HeartbitConfig::from_toml(toml).unwrap();

        assert_eq!(config.provider.name, "anthropic");
        assert_eq!(config.provider.model, "claude-sonnet-4-20250514");
        assert_eq!(config.orchestrator.max_turns, 15);
        assert_eq!(config.orchestrator.max_tokens, 8192);
        assert_eq!(config.agents.len(), 2);
        assert_eq!(config.agents[0].name, "researcher");
        assert_eq!(config.agents[0].mcp_servers.len(), 0);
        assert_eq!(config.agents[1].name, "coder");
        assert_eq!(
            config.agents[1].mcp_servers,
            vec![McpServerEntry::Simple("http://localhost:8000/mcp".into())]
        );

        let restate = config.restate.unwrap();
        assert_eq!(restate.endpoint, "http://localhost:9070");
    }

    #[test]
    fn parse_minimal_config() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;

        let config = HeartbitConfig::from_toml(toml).unwrap();

        assert_eq!(config.provider.name, "anthropic");
        assert_eq!(config.orchestrator.max_turns, 10);
        assert_eq!(config.orchestrator.max_tokens, 4096);
        assert!(config.agents.is_empty());
        assert!(config.restate.is_none());
    }

    #[test]
    fn missing_required_provider_field() {
        let toml = r#"
[provider]
name = "anthropic"
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("model"),
            "error should mention missing field: {msg}"
        );
    }

    #[test]
    fn missing_provider_section() {
        let toml = r#"
[[agents]]
name = "researcher"
description = "Research"
system_prompt = "You research."
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("provider"),
            "error should mention missing section: {msg}"
        );
    }

    #[test]
    fn invalid_toml_syntax() {
        let toml = "this is not valid toml {{{";
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn from_file_nonexistent_path() {
        let err = HeartbitConfig::from_file(std::path::Path::new("/nonexistent/heartbit.toml"))
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("failed to read"), "error: {msg}");
    }

    #[test]
    fn orchestrator_defaults_applied() {
        let toml = r#"
[provider]
name = "openrouter"
model = "anthropic/claude-sonnet-4"

[orchestrator]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.max_turns, 10);
        assert_eq!(config.orchestrator.max_tokens, 4096);
        assert!(config.orchestrator.context_strategy.is_none());
        assert!(config.orchestrator.summarize_threshold.is_none());
        assert!(config.orchestrator.tool_timeout_seconds.is_none());
        assert!(config.orchestrator.max_tool_output_bytes.is_none());
    }

    #[test]
    fn orchestrator_context_strategy_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator.context_strategy]
type = "sliding_window"
max_tokens = 16000
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(
            config.orchestrator.context_strategy,
            Some(ContextStrategyConfig::SlidingWindow { max_tokens: 16000 })
        );
        assert!(config.orchestrator.summarize_threshold.is_none());
    }

    #[test]
    fn agent_config_mcp_servers_default_empty() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "basic"
description = "Basic agent"
system_prompt = "You are basic."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].mcp_servers.is_empty());
    }

    #[test]
    fn agent_max_total_tokens_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "quoter"
description = "Quoter agent"
system_prompt = "You quote."
max_total_tokens = 100000
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].max_total_tokens, Some(100000));
    }

    #[test]
    fn agent_max_total_tokens_defaults_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "basic"
description = "Basic agent"
system_prompt = "You are basic."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].max_total_tokens.is_none());
    }

    #[test]
    fn config_rejects_zero_agent_max_total_tokens() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "quoter"
description = "Quoter"
system_prompt = "Quote."
max_total_tokens = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string()
                .contains("max_total_tokens must be at least 1"),
            "error: {err}"
        );
    }

    #[test]
    fn parse_context_strategy_unlimited() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
context_strategy = { type = "unlimited" }
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(
            config.agents[0].context_strategy,
            Some(ContextStrategyConfig::Unlimited)
        );
    }

    #[test]
    fn parse_context_strategy_sliding_window() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
context_strategy = { type = "sliding_window", max_tokens = 100000 }
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(
            config.agents[0].context_strategy,
            Some(ContextStrategyConfig::SlidingWindow { max_tokens: 100000 })
        );
    }

    #[test]
    fn parse_context_strategy_summarize() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
context_strategy = { type = "summarize", threshold = 80000 }
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(
            config.agents[0].context_strategy,
            Some(ContextStrategyConfig::Summarize { threshold: 80000 })
        );
    }

    #[test]
    fn context_strategy_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].context_strategy.is_none());
    }

    #[test]
    fn parse_memory_config_in_memory() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[memory]
type = "in_memory"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(matches!(config.memory, Some(MemoryConfig::InMemory)));
    }

    #[test]
    fn parse_memory_config_postgres() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[memory]
type = "postgres"
database_url = "postgresql://localhost/heartbit"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        match &config.memory {
            Some(MemoryConfig::Postgres {
                database_url,
                embedding,
            }) => {
                assert_eq!(database_url, "postgresql://localhost/heartbit");
                assert!(embedding.is_none(), "embedding should default to None");
            }
            other => panic!("expected Postgres config, got: {other:?}"),
        }
    }

    #[test]
    fn parse_memory_config_postgres_with_embedding() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[memory]
type = "postgres"
database_url = "postgresql://localhost/heartbit"

[memory.embedding]
provider = "openai"
model = "text-embedding-3-large"
api_key_env = "MY_OPENAI_KEY"
base_url = "https://custom-api.example.com"
dimension = 3072
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        match &config.memory {
            Some(MemoryConfig::Postgres {
                database_url,
                embedding,
            }) => {
                assert_eq!(database_url, "postgresql://localhost/heartbit");
                let emb = embedding.as_ref().expect("embedding config should be set");
                assert_eq!(emb.provider, "openai");
                assert_eq!(emb.model, "text-embedding-3-large");
                assert_eq!(emb.api_key_env, "MY_OPENAI_KEY");
                assert_eq!(
                    emb.base_url.as_deref(),
                    Some("https://custom-api.example.com")
                );
                assert_eq!(emb.dimension, Some(3072));
            }
            other => panic!("expected Postgres config, got: {other:?}"),
        }
    }

    #[test]
    fn parse_memory_config_embedding_defaults() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[memory]
type = "postgres"
database_url = "postgresql://localhost/heartbit"

[memory.embedding]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        match &config.memory {
            Some(MemoryConfig::Postgres { embedding, .. }) => {
                let emb = embedding.as_ref().expect("embedding config should be set");
                assert_eq!(emb.provider, "none");
                assert_eq!(emb.model, "text-embedding-3-small");
                assert_eq!(emb.api_key_env, "OPENAI_API_KEY");
                assert!(emb.base_url.is_none());
                assert!(emb.dimension.is_none());
            }
            other => panic!("expected Postgres config, got: {other:?}"),
        }
    }

    #[test]
    fn memory_config_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.memory.is_none());
    }

    #[test]
    fn zero_max_turns_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_turns = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("max_turns must be at least 1"), "error: {msg}");
    }

    #[test]
    fn zero_max_tokens_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_tokens = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("max_tokens must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn parse_tool_timeout_seconds() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
tool_timeout_seconds = 60
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].tool_timeout_seconds, Some(60));
    }

    #[test]
    fn tool_timeout_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].tool_timeout_seconds.is_none());
    }

    #[test]
    fn parse_max_tool_output_bytes() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
max_tool_output_bytes = 16384
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].max_tool_output_bytes, Some(16384));
    }

    #[test]
    fn max_tool_output_bytes_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].max_tool_output_bytes.is_none());
    }

    #[test]
    fn parse_per_agent_max_turns() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "browser"
description = "Browser"
system_prompt = "Browse."
max_turns = 20
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].max_turns, Some(20));
    }

    #[test]
    fn parse_per_agent_max_tokens() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "writer"
description = "Writer"
system_prompt = "Write."
max_tokens = 16384
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].max_tokens, Some(16384));
    }

    #[test]
    fn per_agent_limits_default_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].max_turns.is_none());
        assert!(config.agents[0].max_tokens.is_none());
    }

    #[test]
    fn per_agent_zero_max_turns_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
max_turns = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("max_turns must be at least 1"), "error: {msg}");
    }

    #[test]
    fn per_agent_zero_max_tokens_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
max_tokens = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("max_tokens must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn parse_response_schema() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "analyst"
description = "Analyst"
system_prompt = "Analyze."

[agents.response_schema]
type = "object"

[agents.response_schema.properties.score]
type = "number"

[agents.response_schema.properties.summary]
type = "string"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let schema = config.agents[0].response_schema.as_ref().unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["score"]["type"], "number");
        assert_eq!(schema["properties"]["summary"]["type"], "string");
    }

    #[test]
    fn response_schema_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].response_schema.is_none());
    }

    #[test]
    fn duplicate_agent_names_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "researcher"
description = "First"
system_prompt = "First."

[[agents]]
name = "researcher"
description = "Second"
system_prompt = "Second."
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("duplicate agent name"), "error: {msg}");
    }

    #[test]
    fn per_agent_zero_summarize_threshold_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
summarize_threshold = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("summarize_threshold must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn per_agent_summarize_threshold_with_context_strategy_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
summarize_threshold = 8000

[agents.context_strategy]
type = "sliding_window"
max_tokens = 50000
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("cannot set both context_strategy and summarize_threshold"),
            "error: {msg}"
        );
    }

    #[test]
    fn per_agent_summarize_threshold_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
summarize_threshold = 8000
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].summarize_threshold, Some(8000));
    }

    #[test]
    fn parse_retry_config() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[provider.retry]
max_retries = 5
base_delay_ms = 1000
max_delay_ms = 60000
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let retry = config.provider.retry.unwrap();
        assert_eq!(retry.max_retries, 5);
        assert_eq!(retry.base_delay_ms, 1000);
        assert_eq!(retry.max_delay_ms, 60000);
    }

    #[test]
    fn retry_config_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.provider.retry.is_none());
    }

    #[test]
    fn retry_config_uses_defaults_for_missing_fields() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[provider.retry]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let retry = config.provider.retry.unwrap();
        assert_eq!(retry.max_retries, 3);
        assert_eq!(retry.base_delay_ms, 500);
        assert_eq!(retry.max_delay_ms, 30000);
    }

    #[test]
    fn zero_context_strategy_max_tokens_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
context_strategy = { type = "sliding_window", max_tokens = 0 }
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("context_strategy.max_tokens must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn zero_summarize_threshold_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
context_strategy = { type = "summarize", threshold = 0 }
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("context_strategy.threshold must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn retry_base_delay_exceeds_max_delay_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[provider.retry]
base_delay_ms = 60000
max_delay_ms = 1000
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("base_delay_ms") && msg.contains("max_delay_ms"),
            "error: {msg}"
        );
    }

    #[test]
    fn retry_base_delay_equals_max_delay_accepted() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[provider.retry]
base_delay_ms = 5000
max_delay_ms = 5000
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let retry = config.provider.retry.unwrap();
        assert_eq!(retry.base_delay_ms, 5000);
        assert_eq!(retry.max_delay_ms, 5000);
    }

    #[test]
    fn zero_tool_timeout_seconds_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
tool_timeout_seconds = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("tool_timeout_seconds must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn zero_max_tool_output_bytes_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."
max_tool_output_bytes = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("max_tool_output_bytes must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn empty_agent_name_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = ""
description = "Test"
system_prompt = "You test."
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("agent name must not be empty"), "error: {msg}");
    }

    #[test]
    fn parse_knowledge_config_with_all_source_types() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[knowledge]
chunk_size = 2000
chunk_overlap = 400

[[knowledge.sources]]
type = "file"
path = "README.md"

[[knowledge.sources]]
type = "glob"
pattern = "docs/**/*.md"

[[knowledge.sources]]
type = "url"
url = "https://docs.example.com/api"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let knowledge = config.knowledge.unwrap();
        assert_eq!(knowledge.chunk_size, 2000);
        assert_eq!(knowledge.chunk_overlap, 400);
        assert_eq!(knowledge.sources.len(), 3);
        assert!(matches!(
            knowledge.sources[0],
            KnowledgeSourceConfig::File { .. }
        ));
        assert!(matches!(
            knowledge.sources[1],
            KnowledgeSourceConfig::Glob { .. }
        ));
        assert!(matches!(
            knowledge.sources[2],
            KnowledgeSourceConfig::Url { .. }
        ));
    }

    #[test]
    fn knowledge_config_defaults() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[knowledge]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let knowledge = config.knowledge.unwrap();
        assert_eq!(knowledge.chunk_size, 1000);
        assert_eq!(knowledge.chunk_overlap, 200);
        assert!(knowledge.sources.is_empty());
    }

    #[test]
    fn knowledge_config_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.knowledge.is_none());
    }

    #[test]
    fn knowledge_zero_chunk_size_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[knowledge]
chunk_size = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("chunk_size must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn knowledge_overlap_exceeds_chunk_size_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[knowledge]
chunk_size = 100
chunk_overlap = 100
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("chunk_overlap") && msg.contains("less than chunk_size"),
            "error: {msg}"
        );
    }

    #[test]
    fn prompt_caching_defaults_false() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(!config.provider.prompt_caching);
    }

    #[test]
    fn prompt_caching_parses_true() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
prompt_caching = true
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.provider.prompt_caching);
    }

    #[test]
    fn prompt_caching_backward_compat() {
        // Old config without prompt_caching should parse fine
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[provider.retry]
max_retries = 3
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(!config.provider.prompt_caching);
        assert!(config.provider.retry.is_some());
    }

    #[test]
    fn orchestrator_zero_context_strategy_max_tokens_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator.context_strategy]
type = "sliding_window"
max_tokens = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("orchestrator.context_strategy.max_tokens must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn orchestrator_zero_context_strategy_threshold_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator.context_strategy]
type = "summarize"
threshold = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("orchestrator.context_strategy.threshold must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn orchestrator_zero_summarize_threshold_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
summarize_threshold = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("orchestrator.summarize_threshold must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn orchestrator_summarize_conflict_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
summarize_threshold = 8000

[orchestrator.context_strategy]
type = "summarize"
threshold = 16000
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cannot set both"), "error: {msg}");
    }

    #[test]
    fn orchestrator_sliding_window_plus_summarize_threshold_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
summarize_threshold = 8000

[orchestrator.context_strategy]
type = "sliding_window"
max_tokens = 16000
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cannot set both"), "error: {msg}");
    }

    #[test]
    fn orchestrator_unlimited_plus_summarize_threshold_allowed() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
summarize_threshold = 8000

[orchestrator.context_strategy]
type = "unlimited"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.summarize_threshold, Some(8000));
    }

    #[test]
    fn orchestrator_zero_tool_timeout_seconds_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
tool_timeout_seconds = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("orchestrator.tool_timeout_seconds must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn orchestrator_zero_max_tool_output_bytes_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_tool_output_bytes = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("orchestrator.max_tool_output_bytes must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn orchestrator_tool_timeout_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
tool_timeout_seconds = 120
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.tool_timeout_seconds, Some(120));
    }

    #[test]
    fn orchestrator_max_tool_output_bytes_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_tool_output_bytes = 32768
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.max_tool_output_bytes, Some(32768));
    }

    #[test]
    fn knowledge_overlap_less_than_chunk_size_accepted() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[knowledge]
chunk_size = 100
chunk_overlap = 50
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let knowledge = config.knowledge.unwrap();
        assert_eq!(knowledge.chunk_size, 100);
        assert_eq!(knowledge.chunk_overlap, 50);
    }

    #[test]
    fn mcp_server_entry_simple_string() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
mcp_servers = ["http://localhost:8000/mcp"]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].mcp_servers.len(), 1);
        assert_eq!(
            config.agents[0].mcp_servers[0],
            McpServerEntry::Simple("http://localhost:8000/mcp".into())
        );
        assert_eq!(
            config.agents[0].mcp_servers[0].url(),
            "http://localhost:8000/mcp"
        );
        assert!(config.agents[0].mcp_servers[0].auth_header().is_none());
    }

    #[test]
    fn mcp_server_entry_full_with_auth() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
mcp_servers = [{ url = "http://gateway:8080/mcp", auth_header = "Bearer tok_xxx" }]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].mcp_servers.len(), 1);
        assert_eq!(
            config.agents[0].mcp_servers[0].url(),
            "http://gateway:8080/mcp"
        );
        assert_eq!(
            config.agents[0].mcp_servers[0].auth_header(),
            Some("Bearer tok_xxx")
        );
    }

    #[test]
    fn mcp_server_entry_full_without_auth() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
mcp_servers = [{ url = "http://localhost:8000/mcp" }]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(
            config.agents[0].mcp_servers[0].url(),
            "http://localhost:8000/mcp"
        );
        assert!(config.agents[0].mcp_servers[0].auth_header().is_none());
    }

    #[test]
    fn mcp_server_entry_mixed_simple_and_full() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
mcp_servers = [
    "http://localhost:8000/mcp",
    { url = "http://gateway:8080/mcp", auth_header = "Bearer tok_xxx" }
]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].mcp_servers.len(), 2);
        assert!(config.agents[0].mcp_servers[0].auth_header().is_none());
        assert_eq!(
            config.agents[0].mcp_servers[1].auth_header(),
            Some("Bearer tok_xxx")
        );
    }

    #[test]
    fn mcp_server_entry_full_empty_url_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
mcp_servers = [{ url = "" }]
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("url must not be empty"), "error: {msg}");
    }

    #[test]
    fn mcp_server_entry_roundtrip() {
        let simple = McpServerEntry::Simple("http://localhost/mcp".into());
        let json = serde_json::to_string(&simple).unwrap();
        let parsed: McpServerEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(simple, parsed);

        let full = McpServerEntry::Full {
            url: "http://gateway/mcp".into(),
            auth_header: Some("Bearer tok".into()),
            resource: None,
            scopes: None,
        };
        let json = serde_json::to_string(&full).unwrap();
        let parsed: McpServerEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(full, parsed);
    }

    #[test]
    fn mcp_server_entry_scopes_resource_roundtrip() {
        let full = McpServerEntry::Full {
            url: "http://gmail-mcp.example.com/mcp".into(),
            auth_header: None,
            resource: Some("https://gmail.googleapis.com".into()),
            scopes: Some(vec!["gmail.readonly".into(), "gmail.send".into()]),
        };
        let json = serde_json::to_string(&full).unwrap();
        let parsed: McpServerEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(full, parsed);

        // resource() falls back to url when resource is None
        let no_resource = McpServerEntry::Full {
            url: "http://mcp.example.com".into(),
            auth_header: None,
            resource: None,
            scopes: None,
        };
        assert_eq!(no_resource.resource(), Some("http://mcp.example.com"));

        // resource() returns explicit resource when set
        assert_eq!(full.resource(), Some("https://gmail.googleapis.com"));

        // scopes() returns the configured scopes
        assert_eq!(
            full.scopes(),
            Some(["gmail.readonly".to_string(), "gmail.send".to_string()].as_slice())
        );

        // Simple variant has resource = url, no scopes
        let simple = McpServerEntry::Simple("http://localhost/mcp".into());
        assert_eq!(simple.resource(), Some("http://localhost/mcp"));
        assert_eq!(simple.scopes(), None);

        // Stdio variant has no resource, no scopes
        let stdio = McpServerEntry::Stdio {
            command: "npx".into(),
            args: vec![],
            env: Default::default(),
        };
        assert_eq!(stdio.resource(), None);
        assert_eq!(stdio.scopes(), None);
    }

    #[test]
    fn orchestrator_run_timeout_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
run_timeout_seconds = 300
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.run_timeout_seconds, Some(300));
    }

    #[test]
    fn orchestrator_run_timeout_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.orchestrator.run_timeout_seconds.is_none());
    }

    #[test]
    fn orchestrator_zero_run_timeout_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
run_timeout_seconds = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("run_timeout_seconds must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn agent_run_timeout_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
run_timeout_seconds = 120
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].run_timeout_seconds, Some(120));
    }

    #[test]
    fn agent_run_timeout_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].run_timeout_seconds.is_none());
    }

    #[test]
    fn agent_zero_run_timeout_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
run_timeout_seconds = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("run_timeout_seconds must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn mcp_server_backward_compat_bare_strings() {
        // Existing configs with bare string arrays must keep working
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "coder"
description = "Coding expert"
system_prompt = "You code."
mcp_servers = ["http://localhost:8000/mcp", "http://localhost:9000/mcp"]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].mcp_servers.len(), 2);
        assert_eq!(
            config.agents[0].mcp_servers[0].url(),
            "http://localhost:8000/mcp"
        );
        assert_eq!(
            config.agents[0].mcp_servers[1].url(),
            "http://localhost:9000/mcp"
        );
    }

    #[test]
    fn per_agent_provider_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-opus-4-20250514"

[[agents]]
name = "researcher"
description = "Research"
system_prompt = "Research."

[agents.provider]
name = "anthropic"
model = "claude-haiku-4-5-20251001"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let agent_provider = config.agents[0].provider.as_ref().unwrap();
        assert_eq!(agent_provider.name, "anthropic");
        assert_eq!(agent_provider.model, "claude-haiku-4-5-20251001");
        assert!(!agent_provider.prompt_caching);
    }

    #[test]
    fn per_agent_provider_with_prompt_caching() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-opus-4-20250514"

[[agents]]
name = "researcher"
description = "Research"
system_prompt = "Research."

[agents.provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
prompt_caching = true
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let agent_provider = config.agents[0].provider.as_ref().unwrap();
        assert!(agent_provider.prompt_caching);
    }

    #[test]
    fn per_agent_provider_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].provider.is_none());
    }

    #[test]
    fn per_agent_provider_empty_model_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."

[agents.provider]
name = "anthropic"
model = ""
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("provider.model must not be empty"),
            "error: {msg}"
        );
    }

    #[test]
    fn per_agent_provider_openrouter() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-opus-4-20250514"

[[agents]]
name = "cheap"
description = "Cheap agent"
system_prompt = "Be frugal."

[agents.provider]
name = "openrouter"
model = "anthropic/claude-haiku-4-5"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let p = config.agents[0].provider.as_ref().unwrap();
        assert_eq!(p.name, "openrouter");
        assert_eq!(p.model, "anthropic/claude-haiku-4-5");
    }

    #[test]
    fn mixed_agents_with_and_without_provider() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-opus-4-20250514"

[[agents]]
name = "researcher"
description = "Research"
system_prompt = "Research."

[agents.provider]
name = "anthropic"
model = "claude-haiku-4-5-20251001"

[[agents]]
name = "coder"
description = "Coding"
system_prompt = "Code."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].provider.is_some());
        assert!(config.agents[1].provider.is_none());
    }

    #[test]
    fn per_agent_provider_empty_name_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."

[agents.provider]
name = ""
model = "claude-haiku-4-5-20251001"
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("provider.name must not be empty"),
            "error: {msg}"
        );
    }

    #[test]
    fn enable_squads_config_parsed() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
enable_squads = false
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.enable_squads, Some(false));
    }

    #[test]
    fn enable_squads_default_auto() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(
            config.orchestrator.enable_squads.is_none(),
            "enable_squads should default to None (auto)"
        );
    }

    #[test]
    fn enable_squads_true_parsed() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
enable_squads = true
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.enable_squads, Some(true));
    }

    #[test]
    fn a2a_agents_defaults_empty() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.agents[0].a2a_agents.is_empty());
    }

    #[test]
    fn a2a_agents_parses_simple() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
a2a_agents = ["http://localhost:9000"]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].a2a_agents.len(), 1);
        assert_eq!(
            config.agents[0].a2a_agents[0].url(),
            "http://localhost:9000"
        );
        assert!(config.agents[0].a2a_agents[0].auth_header().is_none());
    }

    #[test]
    fn a2a_agents_parses_full_with_auth() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
a2a_agents = [{ url = "http://gateway:8080", auth_header = "Bearer tok_a2a" }]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].a2a_agents.len(), 1);
        assert_eq!(config.agents[0].a2a_agents[0].url(), "http://gateway:8080");
        assert_eq!(
            config.agents[0].a2a_agents[0].auth_header(),
            Some("Bearer tok_a2a")
        );
    }

    #[test]
    fn a2a_agents_empty_url_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
a2a_agents = [""]
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("a2a_agents") && msg.contains("url must not be empty"),
            "error: {msg}"
        );
    }

    #[test]
    fn a2a_agents_mixed_with_mcp_servers() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "hybrid"
description = "Hybrid agent"
system_prompt = "You are hybrid."
mcp_servers = ["http://localhost:8000/mcp"]
a2a_agents = ["http://localhost:9000"]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].mcp_servers.len(), 1);
        assert_eq!(config.agents[0].a2a_agents.len(), 1);
    }

    #[test]
    fn mcp_server_entry_simple_empty_url_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "Test."
mcp_servers = [""]
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("url must not be empty"), "error: {msg}");
    }

    #[test]
    fn config_enable_reflection_orchestrator() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
enable_reflection = true

[[agents]]
name = "a"
description = "A"
system_prompt = "s"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.enable_reflection, Some(true));
    }

    #[test]
    fn config_enable_reflection_per_agent() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "reflective"
description = "R"
system_prompt = "s"
enable_reflection = true
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].enable_reflection, Some(true));
    }

    #[test]
    fn config_rejects_zero_compression_threshold() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
tool_output_compression_threshold = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string()
                .contains("tool_output_compression_threshold")
        );
    }

    #[test]
    fn config_rejects_zero_max_tools_per_turn() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_tools_per_turn = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(err.to_string().contains("max_tools_per_turn"));
    }

    #[test]
    fn config_rejects_zero_agent_compression_threshold() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "a"
description = "d"
system_prompt = "s"
tool_output_compression_threshold = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string()
                .contains("tool_output_compression_threshold"),
            "error: {err}"
        );
    }

    #[test]
    fn config_rejects_zero_agent_max_tools_per_turn() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "a"
description = "d"
system_prompt = "s"
max_tools_per_turn = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string().contains("max_tools_per_turn"),
            "error: {err}"
        );
    }

    #[test]
    fn config_rejects_zero_orchestrator_max_identical_tool_calls() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_identical_tool_calls = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string().contains("max_identical_tool_calls"),
            "error: {err}"
        );
    }

    #[test]
    fn config_rejects_zero_agent_max_identical_tool_calls() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "a"
description = "d"
system_prompt = "s"
max_identical_tool_calls = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string().contains("max_identical_tool_calls"),
            "error: {err}"
        );
    }

    #[test]
    fn config_parses_max_identical_tool_calls() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_identical_tool_calls = 5

[[agents]]
name = "a"
description = "d"
system_prompt = "s"
max_identical_tool_calls = 3
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.orchestrator.max_identical_tool_calls, Some(5));
        assert_eq!(config.agents[0].max_identical_tool_calls, Some(3));
    }

    #[test]
    fn config_max_identical_tool_calls_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "a"
description = "d"
system_prompt = "s"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.orchestrator.max_identical_tool_calls.is_none());
        assert!(config.agents[0].max_identical_tool_calls.is_none());
    }

    // --- Permission Rules Config Tests ---

    #[test]
    fn config_parses_permission_rules() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "a"
description = "d"
system_prompt = "s"

[[permissions]]
tool = "read_file"
action = "allow"

[[permissions]]
tool = "bash"
pattern = "rm *"
action = "deny"

[[permissions]]
tool = "*"
pattern = "*.env*"
action = "deny"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.permissions.len(), 3);
        assert_eq!(config.permissions[0].tool, "read_file");
        assert_eq!(config.permissions[0].pattern, "*"); // default
        assert_eq!(
            config.permissions[0].action,
            crate::agent::permission::PermissionAction::Allow
        );
        assert_eq!(config.permissions[1].tool, "bash");
        assert_eq!(config.permissions[1].pattern, "rm *");
        assert_eq!(
            config.permissions[1].action,
            crate::agent::permission::PermissionAction::Deny
        );
        assert_eq!(config.permissions[2].tool, "*");
        assert_eq!(config.permissions[2].pattern, "*.env*");
    }

    #[test]
    fn config_defaults_to_empty_permissions() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "a"
description = "d"
system_prompt = "s"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.permissions.is_empty());
    }

    #[test]
    fn lsp_config_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.lsp.is_none());
    }

    #[test]
    fn lsp_config_enabled_defaults_true() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[lsp]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let lsp = config.lsp.unwrap();
        assert!(lsp.enabled);
    }

    #[test]
    fn lsp_config_disabled() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[lsp]
enabled = false
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let lsp = config.lsp.unwrap();
        assert!(!lsp.enabled);
    }

    #[test]
    fn parse_session_prune_with_preserve_task() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."

[agents.session_prune]
keep_recent_n = 3
pruned_tool_result_max_bytes = 100
preserve_task = false
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let sp = config.agents[0].session_prune.as_ref().unwrap();
        assert_eq!(sp.keep_recent_n, 3);
        assert_eq!(sp.pruned_tool_result_max_bytes, 100);
        assert!(!sp.preserve_task);
    }

    #[test]
    fn config_telemetry_parses_observability_mode() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[telemetry]
otlp_endpoint = "http://localhost:4317"
observability_mode = "analysis"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let telemetry = config.telemetry.unwrap();
        assert_eq!(telemetry.observability_mode.as_deref(), Some("analysis"));
    }

    #[test]
    fn config_telemetry_observability_mode_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[telemetry]
otlp_endpoint = "http://localhost:4317"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let telemetry = config.telemetry.unwrap();
        assert!(telemetry.observability_mode.is_none());
    }

    #[test]
    fn dispatch_mode_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.orchestrator.dispatch_mode.is_none());
    }

    #[test]
    fn dispatch_mode_sequential_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
dispatch_mode = "sequential"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(
            config.orchestrator.dispatch_mode,
            Some(DispatchMode::Sequential)
        );
    }

    #[test]
    fn dispatch_mode_parallel_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
dispatch_mode = "parallel"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(
            config.orchestrator.dispatch_mode,
            Some(DispatchMode::Parallel)
        );
    }

    #[test]
    fn dispatch_mode_invalid_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
dispatch_mode = "bananas"
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn session_prune_preserve_task_defaults_to_true() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test"
system_prompt = "You test."

[agents.session_prune]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let sp = config.agents[0].session_prune.as_ref().unwrap();
        assert_eq!(sp.keep_recent_n, 2); // default
        assert_eq!(sp.pruned_tool_result_max_bytes, 200); // default
        assert!(sp.preserve_task); // default true
    }

    #[test]
    fn daemon_config_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon]
bind = "0.0.0.0:8080"
max_concurrent_tasks = 8

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        assert_eq!(daemon.bind, "0.0.0.0:8080");
        assert_eq!(daemon.max_concurrent_tasks, 8);
        let kafka = daemon.kafka.unwrap();
        assert_eq!(kafka.brokers, "localhost:9092");
        assert_eq!(kafka.consumer_group, "heartbit-daemon");
        assert_eq!(kafka.commands_topic, "heartbit.commands");
        assert_eq!(kafka.events_topic, "heartbit.events");
        assert_eq!(kafka.dead_letter_topic, "heartbit.dead-letter");
    }

    #[test]
    fn daemon_config_defaults() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        assert_eq!(daemon.bind, "127.0.0.1:3000");
        assert_eq!(daemon.max_concurrent_tasks, 4);
    }

    #[test]
    fn daemon_config_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.daemon.is_none());
    }

    #[test]
    fn daemon_zero_max_concurrent_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon]
max_concurrent_tasks = 0

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("max_concurrent_tasks must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn daemon_empty_brokers_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = ""
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("brokers must not be empty"), "error: {msg}");
    }

    #[test]
    fn daemon_config_metrics_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config: HeartbitConfig = toml::from_str(toml).unwrap();
        let daemon = config.daemon.unwrap();
        assert!(daemon.metrics.is_none());
    }

    #[test]
    fn daemon_config_metrics_enabled_explicit() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.metrics]
enabled = true
"#;
        let config: HeartbitConfig = toml::from_str(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let metrics = daemon.metrics.unwrap();
        assert!(metrics.enabled);
    }

    #[test]
    fn daemon_config_metrics_disabled() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.metrics]
enabled = false
"#;
        let config: HeartbitConfig = toml::from_str(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let metrics = daemon.metrics.unwrap();
        assert!(!metrics.enabled);
    }

    #[test]
    fn daemon_config_metrics_section_present_defaults_enabled() {
        // When [daemon.metrics] is present but `enabled` is omitted, defaults to true
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.metrics]
"#;
        let config: HeartbitConfig = toml::from_str(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let metrics = daemon.metrics.unwrap();
        assert!(metrics.enabled);
    }

    #[test]
    fn sensor_source_name() {
        let rss = SensorSourceConfig::Rss {
            name: "tech_rss".into(),
            feeds: vec!["https://example.com/feed".into()],
            interest_keywords: vec![],
            poll_interval_seconds: 900,
        };
        assert_eq!(rss.name(), "tech_rss");

        let webhook = SensorSourceConfig::Webhook {
            name: "github_events".into(),
            path: "/webhooks/github".into(),
            secret_env: None,
        };
        assert_eq!(webhook.name(), "github_events");
    }

    // Sensor validation tests removed — sensors field moved to gateway crate.

    // (sensor_duplicate_source_names removed — field moved to gateway)

    #[test]
    fn kafka_dead_letter_topic_custom() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"
dead_letter_topic = "my.custom.dead-letter"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        assert_eq!(
            daemon.kafka.unwrap().dead_letter_topic,
            "my.custom.dead-letter"
        );
    }

    // ws_config_* tests removed — ws field moved to gateway crate.

    #[test]
    fn daemon_database_url_default_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.daemon.unwrap().database_url.is_none());
    }

    #[test]
    fn daemon_database_url_present() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon]
database_url = "postgresql://localhost/heartbit_tasks"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(
            config.daemon.unwrap().database_url.as_deref(),
            Some("postgresql://localhost/heartbit_tasks")
        );
    }

    #[test]
    fn workspace_config_explicit_root() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_turns = 5
max_tokens = 4096

[workspace]
root = "/custom/workspaces"

[[agents]]
name = "test"
description = "test"
system_prompt = "test"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let ws = config.workspace.unwrap();
        assert_eq!(ws.root, "/custom/workspaces");
    }

    #[test]
    fn workspace_config_default_root() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_turns = 5
max_tokens = 4096

[workspace]

[[agents]]
name = "test"
description = "test"
system_prompt = "test"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let ws = config.workspace.unwrap();
        // Should use the default path
        assert!(ws.root.contains(".heartbit/workspaces"));
    }

    #[test]
    fn workspace_config_absent() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_turns = 5
max_tokens = 4096

[[agents]]
name = "test"
description = "test"
system_prompt = "test"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.workspace.is_none());
    }

    // heartbit_pulse config tests removed — field moved to gateway crate.

    #[test]
    fn active_hours_parse_valid() {
        let ah = ActiveHoursConfig {
            start: "08:30".into(),
            end: "22:00".into(),
        };
        assert_eq!(ah.parse_start().unwrap(), (8, 30));
        assert_eq!(ah.parse_end().unwrap(), (22, 0));
    }

    #[test]
    fn active_hours_parse_midnight() {
        let ah = ActiveHoursConfig {
            start: "00:00".into(),
            end: "23:59".into(),
        };
        assert_eq!(ah.parse_start().unwrap(), (0, 0));
        assert_eq!(ah.parse_end().unwrap(), (23, 59));
    }

    #[test]
    fn active_hours_parse_invalid_format() {
        let ah = ActiveHoursConfig {
            start: "8am".into(),
            end: "22:00".into(),
        };
        assert!(ah.parse_start().is_err());
    }

    #[test]
    fn active_hours_parse_out_of_range() {
        let ah = ActiveHoursConfig {
            start: "25:00".into(),
            end: "22:00".into(),
        };
        assert!(ah.parse_start().is_err());
    }

    // validate_rejects_pulse tests removed — heartbit_pulse field moved to gateway crate.

    #[test]
    fn routing_defaults_to_auto_when_missing() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[[agents]]
name = "worker"
description = "worker agent"
system_prompt = "you are a worker"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        assert_eq!(config.orchestrator.routing, RoutingMode::Auto);
        assert!(config.orchestrator.escalation);
    }

    #[test]
    fn routing_parses_always_orchestrate() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[orchestrator]
routing = "always_orchestrate"

[[agents]]
name = "worker"
description = "worker agent"
system_prompt = "you are a worker"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        assert_eq!(config.orchestrator.routing, RoutingMode::AlwaysOrchestrate);
    }

    #[test]
    fn routing_parses_single_agent() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[orchestrator]
routing = "single_agent"

[[agents]]
name = "worker"
description = "worker agent"
system_prompt = "you are a worker"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        assert_eq!(config.orchestrator.routing, RoutingMode::SingleAgent);
    }

    #[test]
    fn escalation_defaults_to_true() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[[agents]]
name = "worker"
description = "worker agent"
system_prompt = "you are a worker"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        assert!(config.orchestrator.escalation);
    }

    #[test]
    fn escalation_can_be_disabled() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[orchestrator]
escalation = false

[[agents]]
name = "worker"
description = "worker agent"
system_prompt = "you are a worker"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        assert!(!config.orchestrator.escalation);
    }

    #[test]
    fn auth_config_valid() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
bearer_tokens = ["my-secret-key", "rotation-key-2"]
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let auth = config.daemon.unwrap().auth.unwrap();
        assert_eq!(auth.bearer_tokens.len(), 2);
        assert_eq!(auth.bearer_tokens[0], "my-secret-key");
    }

    #[test]
    fn auth_config_empty_tokens_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
bearer_tokens = []
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("daemon.auth requires at least bearer_tokens or jwks_url"),
            "got: {err}"
        );
    }

    #[test]
    fn auth_config_empty_token_string_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
bearer_tokens = [""]
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("daemon.auth.bearer_tokens[0] must not be empty"),
            "got: {err}"
        );
    }

    #[test]
    fn auth_config_none_is_valid() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        assert!(config.daemon.unwrap().auth.is_none());
    }

    #[test]
    fn auth_config_jwks_only_is_valid() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
jwks_url = "https://idp.example.com/.well-known/jwks.json"
issuer = "https://idp.example.com"
audience = "heartbit-api"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let auth = config.daemon.unwrap().auth.unwrap();
        assert!(auth.bearer_tokens.is_empty());
        assert_eq!(
            auth.jwks_url.as_deref(),
            Some("https://idp.example.com/.well-known/jwks.json")
        );
        assert_eq!(auth.issuer.as_deref(), Some("https://idp.example.com"));
        assert_eq!(auth.audience.as_deref(), Some("heartbit-api"));
    }

    #[test]
    fn auth_config_empty_jwks_url_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
bearer_tokens = ["valid-token"]
jwks_url = ""
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("daemon.auth.jwks_url must not be empty"),
            "got: {err}"
        );
    }

    #[test]
    fn auth_config_no_tokens_no_jwks_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
issuer = "https://idp.example.com"
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("daemon.auth requires at least bearer_tokens or jwks_url"),
            "got: {err}"
        );
    }

    // --- Token exchange config ---

    #[test]
    fn token_exchange_config_valid() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
jwks_url = "https://idp.example.com/.well-known/jwks.json"

[daemon.auth.token_exchange]
exchange_url = "https://idp.example.com/oauth/token"
client_id = "heartbit-agent"
client_secret = "secret123"
agent_token = "agent-cred-token"
scopes = ["crm:read", "crm:write"]
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let te = config.daemon.unwrap().auth.unwrap().token_exchange.unwrap();
        assert_eq!(te.exchange_url, "https://idp.example.com/oauth/token");
        assert_eq!(te.client_id, "heartbit-agent");
        assert_eq!(te.client_secret, "secret123");
        assert_eq!(te.agent_token, "agent-cred-token");
        assert_eq!(te.scopes, vec!["crm:read", "crm:write"]);
    }

    #[test]
    fn token_exchange_empty_exchange_url_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
jwks_url = "https://idp.example.com/.well-known/jwks.json"

[daemon.auth.token_exchange]
exchange_url = ""
client_id = "heartbit-agent"
client_secret = "secret123"
agent_token = "agent-cred-token"
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("daemon.auth.token_exchange.exchange_url must not be empty"),
            "got: {err}"
        );
    }

    #[test]
    fn token_exchange_empty_client_id_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
jwks_url = "https://idp.example.com/.well-known/jwks.json"

[daemon.auth.token_exchange]
exchange_url = "https://idp.example.com/oauth/token"
client_id = ""
client_secret = "secret123"
agent_token = "agent-cred-token"
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("daemon.auth.token_exchange.client_id must not be empty"),
            "got: {err}"
        );
    }

    #[test]
    fn token_exchange_empty_agent_token_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
jwks_url = "https://idp.example.com/.well-known/jwks.json"

[daemon.auth.token_exchange]
exchange_url = "https://idp.example.com/oauth/token"
client_id = "heartbit-agent"
client_secret = "secret123"
agent_token = ""
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("daemon.auth.token_exchange: set tenant_id for auto-fetch"),
            "got: {err}"
        );
    }

    #[test]
    fn token_exchange_none_is_valid() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.auth]
jwks_url = "https://idp.example.com/.well-known/jwks.json"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        assert!(
            config
                .daemon
                .unwrap()
                .auth
                .unwrap()
                .token_exchange
                .is_none()
        );
    }

    // --- Cascade config ---

    #[test]
    fn cascade_config_parses_full() {
        let toml_str = r#"
[provider]
name = "openrouter"
model = "anthropic/claude-sonnet-4"

[provider.cascade]
enabled = true

[[provider.cascade.tiers]]
model = "anthropic/claude-3.5-haiku"

[provider.cascade.gate]
type = "heuristic"
min_output_tokens = 10
accept_tool_calls = false
escalate_on_max_tokens = false
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let cascade = config.provider.cascade.unwrap();
        assert!(cascade.enabled);
        assert_eq!(cascade.tiers.len(), 1);
        assert_eq!(cascade.tiers[0].model, "anthropic/claude-3.5-haiku");
        match &cascade.gate {
            CascadeGateConfig::Heuristic {
                min_output_tokens,
                accept_tool_calls,
                escalate_on_max_tokens,
            } => {
                assert_eq!(*min_output_tokens, 10);
                assert!(!accept_tool_calls);
                assert!(!escalate_on_max_tokens);
            }
        }
    }

    #[test]
    fn cascade_config_defaults_when_absent() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        assert!(config.provider.cascade.is_none());
    }

    #[test]
    fn cascade_config_gate_defaults() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[provider.cascade]
enabled = true

[[provider.cascade.tiers]]
model = "claude-3.5-haiku"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let cascade = config.provider.cascade.unwrap();
        match &cascade.gate {
            CascadeGateConfig::Heuristic {
                min_output_tokens,
                accept_tool_calls,
                escalate_on_max_tokens,
            } => {
                assert_eq!(*min_output_tokens, 5);
                assert!(accept_tool_calls);
                assert!(escalate_on_max_tokens);
            }
        }
    }

    #[test]
    fn validate_rejects_cascade_enabled_without_tiers() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[provider.cascade]
enabled = true
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string().contains("no tiers are configured"),
            "error: {err}"
        );
    }

    #[test]
    fn cascade_disabled_with_tiers_is_valid() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[provider.cascade]
enabled = false

[[provider.cascade.tiers]]
model = "claude-3.5-haiku"
"#;
        // enabled=false means the tiers are ignored; should not error
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let cascade = config.provider.cascade.unwrap();
        assert!(!cascade.enabled);
    }

    #[test]
    fn agent_provider_cascade_config_parses() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "researcher"
description = "Research agent"
system_prompt = "You are a researcher."

[agents.provider]
name = "openrouter"
model = "anthropic/claude-sonnet-4"

[agents.provider.cascade]
enabled = true

[[agents.provider.cascade.tiers]]
model = "anthropic/claude-3.5-haiku"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let agent_cascade = config.agents[0]
            .provider
            .as_ref()
            .unwrap()
            .cascade
            .as_ref()
            .unwrap();
        assert!(agent_cascade.enabled);
        assert_eq!(agent_cascade.tiers.len(), 1);
    }

    #[test]
    fn validate_rejects_empty_provider_name() {
        let toml_str = r#"
[provider]
name = ""
model = "claude-sonnet-4-20250514"
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string().contains("provider.name must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_rejects_empty_provider_model() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = ""
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string().contains("provider.model must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_rejects_empty_cascade_tier_model() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[provider.cascade]
enabled = true

[[provider.cascade.tiers]]
model = ""
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("provider.cascade.tiers[0].model must not be empty"),
            "error: {err}"
        );
    }

    // MCP sensor config tests removed — sensors field moved to gateway crate.
    // sensor_source_mcp_serde, sensor_source_mcp_defaults, sensor_source_mcp_with_enrichment,
    // sensor_source_mcp_with_auth_header, sensor_source_mcp_simple_server,
    // sensor_source_mcp_empty_tool_name_rejected, sensor_source_mcp_empty_kafka_topic_rejected,
    // sensor_source_mcp_stdio_server, sensor_source_mcp_stdio_empty_command_rejected,
    // sensor_source_mcp_stdio_defaults

    #[test]
    fn mcp_server_entry_stdio_roundtrip() {
        let stdio = McpServerEntry::Stdio {
            command: "npx".into(),
            args: vec!["-y".into(), "my-mcp-server".into()],
            env: std::collections::HashMap::from([("KEY".into(), "val".into())]),
        };
        let json = serde_json::to_string(&stdio).unwrap();
        let parsed: McpServerEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(stdio, parsed);
    }

    #[test]
    fn mcp_server_entry_display_name() {
        let simple = McpServerEntry::Simple("http://localhost/mcp".into());
        assert_eq!(simple.display_name(), "http://localhost/mcp");

        let full = McpServerEntry::Full {
            url: "http://gateway/mcp".into(),
            auth_header: Some("Bearer tok".into()),
            resource: None,
            scopes: None,
        };
        assert_eq!(full.display_name(), "http://gateway/mcp");

        let stdio = McpServerEntry::Stdio {
            command: "npx".into(),
            args: vec!["-y".into(), "server".into()],
            env: Default::default(),
        };
        assert_eq!(stdio.display_name(), "npx -y server");

        let stdio_no_args = McpServerEntry::Stdio {
            command: "my-server".into(),
            args: vec![],
            env: Default::default(),
        };
        assert_eq!(stdio_no_args.display_name(), "my-server");
    }

    // --- Daemon validation tests ---

    #[test]
    fn validate_daemon_empty_consumer_group() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"
consumer_group = ""
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string().contains("consumer_group must not be empty"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_daemon_empty_commands_topic() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"
commands_topic = ""
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string().contains("commands_topic must not be empty"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_daemon_empty_events_topic() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"
events_topic = ""
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string().contains("events_topic must not be empty"),
            "got: {err}"
        );
    }

    // --- SensorModality tests (always available, not gated on `sensor` feature) ---

    #[test]
    fn sensor_modality_serde_roundtrip() {
        for modality in [
            SensorModality::Text,
            SensorModality::Image,
            SensorModality::Audio,
            SensorModality::Structured,
        ] {
            let json = serde_json::to_string(&modality).unwrap();
            let back: SensorModality = serde_json::from_str(&json).unwrap();
            assert_eq!(back, modality);
        }
    }

    #[test]
    fn sensor_modality_snake_case() {
        assert_eq!(
            serde_json::to_string(&SensorModality::Text).unwrap(),
            r#""text""#
        );
        assert_eq!(
            serde_json::to_string(&SensorModality::Image).unwrap(),
            r#""image""#
        );
        assert_eq!(
            serde_json::to_string(&SensorModality::Audio).unwrap(),
            r#""audio""#
        );
        assert_eq!(
            serde_json::to_string(&SensorModality::Structured).unwrap(),
            r#""structured""#
        );
    }

    #[test]
    fn sensor_modality_display() {
        assert_eq!(SensorModality::Text.to_string(), "text");
        assert_eq!(SensorModality::Image.to_string(), "image");
        assert_eq!(SensorModality::Audio.to_string(), "audio");
        assert_eq!(SensorModality::Structured.to_string(), "structured");
    }

    // --- TrustLevel tests (always available, not gated on `sensor` feature) ---

    #[test]
    fn trust_level_default_is_unknown() {
        assert_eq!(TrustLevel::default(), TrustLevel::Unknown);
    }

    #[test]
    fn trust_level_ordering() {
        assert!(TrustLevel::Quarantined < TrustLevel::Unknown);
        assert!(TrustLevel::Unknown < TrustLevel::Known);
        assert!(TrustLevel::Known < TrustLevel::Verified);
        assert!(TrustLevel::Verified < TrustLevel::Owner);
    }

    #[test]
    fn trust_level_serde_roundtrip() {
        for t in [
            TrustLevel::Quarantined,
            TrustLevel::Unknown,
            TrustLevel::Known,
            TrustLevel::Verified,
            TrustLevel::Owner,
        ] {
            let json = serde_json::to_string(&t).unwrap();
            let parsed: TrustLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, t);
        }
    }

    #[test]
    fn trust_level_display() {
        assert_eq!(TrustLevel::Quarantined.to_string(), "quarantined");
        assert_eq!(TrustLevel::Unknown.to_string(), "unknown");
        assert_eq!(TrustLevel::Known.to_string(), "known");
        assert_eq!(TrustLevel::Verified.to_string(), "verified");
        assert_eq!(TrustLevel::Owner.to_string(), "owner");
    }

    #[test]
    fn trust_level_resolve_owner() {
        let trust = TrustLevel::resolve(
            Some("owner@example.com"),
            &["owner@example.com".into()],
            &[],
            &[],
        );
        assert_eq!(trust, TrustLevel::Owner);
    }

    #[test]
    fn trust_level_resolve_verified() {
        let trust = TrustLevel::resolve(
            Some("alice@example.com"),
            &[],
            &["alice@example.com".into()],
            &[],
        );
        assert_eq!(trust, TrustLevel::Verified);
    }

    #[test]
    fn trust_level_resolve_blocked() {
        let trust = TrustLevel::resolve(
            Some("spammer@evil.com"),
            &[],
            &[],
            &["spammer@evil.com".into()],
        );
        assert_eq!(trust, TrustLevel::Quarantined);
    }

    #[test]
    fn trust_level_resolve_unknown() {
        let trust = TrustLevel::resolve(Some("stranger@example.com"), &[], &[], &[]);
        assert_eq!(trust, TrustLevel::Unknown);
    }

    #[test]
    fn trust_level_resolve_none_sender() {
        let trust = TrustLevel::resolve(None, &[], &[], &[]);
        assert_eq!(trust, TrustLevel::Unknown);
    }

    #[test]
    fn trust_level_owner_trumps_blocked() {
        let trust = TrustLevel::resolve(
            Some("owner@example.com"),
            &["owner@example.com".into()],
            &[],
            &["owner@example.com".into()],
        );
        assert_eq!(trust, TrustLevel::Owner);
    }

    #[test]
    fn trust_level_resolve_case_insensitive() {
        let trust = TrustLevel::resolve(
            Some("Owner@Example.COM"),
            &["owner@example.com".into()],
            &[],
            &[],
        );
        assert_eq!(trust, TrustLevel::Owner);
    }

    // --- Guardrails config tests ---

    #[test]
    fn guardrails_config_default_empty() {
        let config: GuardrailsConfig = toml::from_str("").unwrap();
        assert!(config.injection.is_none());
        assert!(config.pii.is_none());
        assert!(config.tool_policy.is_none());
    }

    #[test]
    fn guardrails_config_roundtrip() {
        let toml_str = r#"
[injection]
threshold = 0.3
mode = "warn"

[pii]
action = "redact"
detectors = ["email", "ssn"]

[tool_policy]
default_action = "allow"

[[tool_policy.rules]]
tool = "bash"
action = "deny"
input_constraints = []

[[tool_policy.rules]]
tool = "gmail_send_*"
action = "warn"
input_constraints = []
"#;
        let config: GuardrailsConfig = toml::from_str(toml_str).unwrap();

        // Injection
        let inj = config.injection.as_ref().unwrap();
        assert!((inj.threshold - 0.3).abs() < 0.01);
        assert_eq!(inj.mode, "warn");

        // PII
        let pii = config.pii.as_ref().unwrap();
        assert_eq!(pii.action, "redact");
        assert_eq!(pii.detectors, vec!["email", "ssn"]);

        // Tool policy
        let tp = config.tool_policy.as_ref().unwrap();
        assert_eq!(tp.default_action, "allow");
        assert_eq!(tp.rules.len(), 2);
        assert_eq!(tp.rules[0].tool, "bash");
        assert_eq!(tp.rules[0].action, "deny");
        assert_eq!(tp.rules[1].tool, "gmail_send_*");
        assert_eq!(tp.rules[1].action, "warn");

        // Verify serialization roundtrip
        let serialized = toml::to_string(&config).unwrap();
        let _back: GuardrailsConfig = toml::from_str(&serialized).unwrap();
    }

    #[test]
    fn guardrails_config_with_input_constraints() {
        let toml_str = r#"
[tool_policy]
default_action = "deny"

[[tool_policy.rules]]
tool = "read"
action = "allow"

[[tool_policy.rules.input_constraints]]
path = "path"
deny_pattern = "^/etc/"
"#;
        let config: GuardrailsConfig = toml::from_str(toml_str).unwrap();
        let tp = config.tool_policy.unwrap();
        assert_eq!(tp.default_action, "deny");
        assert_eq!(tp.rules.len(), 1);
        assert_eq!(tp.rules[0].input_constraints.len(), 1);
        assert_eq!(tp.rules[0].input_constraints[0].path, "path");
        assert_eq!(
            tp.rules[0].input_constraints[0].deny_pattern.as_deref(),
            Some("^/etc/")
        );
    }

    #[test]
    fn heartbit_config_with_guardrails() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[guardrails.injection]
threshold = 0.5
mode = "deny"

[guardrails.pii]
action = "redact"
"#;
        let config: HeartbitConfig = toml::from_str(toml_str).unwrap();
        let guardrails = config.guardrails.unwrap();
        assert!(guardrails.injection.is_some());
        assert!(guardrails.pii.is_some());
        assert!(guardrails.tool_policy.is_none());
    }

    #[test]
    fn guardrails_config_build_empty() {
        let config = GuardrailsConfig::default();
        assert!(config.is_empty());
        let guardrails = config.build().unwrap();
        assert!(guardrails.is_empty());
    }

    #[test]
    fn guardrails_config_build_injection() {
        let config = GuardrailsConfig {
            injection: Some(InjectionConfig {
                threshold: 0.3,
                mode: "warn".into(),
            }),
            ..Default::default()
        };
        let guardrails = config.build().unwrap();
        assert_eq!(guardrails.len(), 1);
    }

    #[test]
    fn guardrails_config_build_pii() {
        let config = GuardrailsConfig {
            pii: Some(PiiConfig {
                action: "redact".into(),
                detectors: vec!["email".into(), "phone".into()],
            }),
            ..Default::default()
        };
        let guardrails = config.build().unwrap();
        assert_eq!(guardrails.len(), 1);
    }

    #[test]
    fn guardrails_config_build_tool_policy() {
        let config = GuardrailsConfig {
            tool_policy: Some(ToolPolicyConfig {
                default_action: "allow".into(),
                rules: vec![ToolPolicyRuleConfig {
                    tool: "bash".into(),
                    action: "deny".into(),
                    input_constraints: vec![],
                }],
            }),
            ..Default::default()
        };
        let guardrails = config.build().unwrap();
        assert_eq!(guardrails.len(), 1);
    }

    #[test]
    fn guardrails_config_build_all_three() {
        let config = GuardrailsConfig {
            injection: Some(InjectionConfig {
                threshold: 0.5,
                mode: "deny".into(),
            }),
            pii: Some(PiiConfig {
                action: "warn".into(),
                detectors: default_pii_detectors(),
            }),
            tool_policy: Some(ToolPolicyConfig {
                default_action: "allow".into(),
                rules: vec![],
            }),
            llm_judge: None,
            secret_scan: None,
            behavioral: None,
            action_budget: None,
        };
        assert!(!config.is_empty());
        let guardrails = config.build().unwrap();
        assert_eq!(guardrails.len(), 3);
    }

    #[test]
    fn guardrails_config_build_invalid_mode_errors() {
        let config = GuardrailsConfig {
            injection: Some(InjectionConfig {
                threshold: 0.5,
                mode: "invalid".into(),
            }),
            ..Default::default()
        };
        let err = config.build().err().expect("should fail");
        assert!(
            err.to_string().contains("invalid injection mode"),
            "error: {err}"
        );
    }

    #[test]
    fn guardrails_config_build_invalid_pii_action_errors() {
        let config = GuardrailsConfig {
            pii: Some(PiiConfig {
                action: "destroy".into(),
                detectors: vec!["email".into()],
            }),
            ..Default::default()
        };
        let err = config.build().err().expect("should fail");
        assert!(
            err.to_string().contains("invalid PII action"),
            "error: {err}"
        );
    }

    #[test]
    fn guardrails_config_build_invalid_detector_errors() {
        let config = GuardrailsConfig {
            pii: Some(PiiConfig {
                action: "redact".into(),
                detectors: vec!["dna_sequence".into()],
            }),
            ..Default::default()
        };
        let err = config.build().err().expect("should fail");
        assert!(
            err.to_string().contains("unknown PII detector"),
            "error: {err}"
        );
    }

    #[test]
    fn guardrails_config_build_invalid_regex_errors() {
        let config = GuardrailsConfig {
            tool_policy: Some(ToolPolicyConfig {
                default_action: "allow".into(),
                rules: vec![ToolPolicyRuleConfig {
                    tool: "bash".into(),
                    action: "allow".into(),
                    input_constraints: vec![InputConstraintConfig {
                        path: "command".into(),
                        deny_pattern: Some("[invalid".into()),
                        max_length: None,
                    }],
                }],
            }),
            ..Default::default()
        };
        let err = config.build().err().expect("should fail");
        assert!(
            err.to_string().contains("invalid deny_pattern"),
            "error: {err}"
        );
    }

    #[test]
    fn guardrails_config_build_with_input_constraints() {
        let config = GuardrailsConfig {
            tool_policy: Some(ToolPolicyConfig {
                default_action: "deny".into(),
                rules: vec![ToolPolicyRuleConfig {
                    tool: "bash".into(),
                    action: "allow".into(),
                    input_constraints: vec![
                        InputConstraintConfig {
                            path: "command".into(),
                            deny_pattern: Some(r"rm\s+-rf".into()),
                            max_length: None,
                        },
                        InputConstraintConfig {
                            path: "command".into(),
                            deny_pattern: None,
                            max_length: Some(1024),
                        },
                    ],
                }],
            }),
            ..Default::default()
        };
        let guardrails = config.build().unwrap();
        assert_eq!(guardrails.len(), 1);
    }

    #[test]
    fn guardrails_config_build_from_toml() {
        let toml_str = r#"
[injection]
threshold = 0.4
mode = "warn"

[pii]
action = "deny"
detectors = ["email", "ssn"]

[tool_policy]
default_action = "allow"

[[tool_policy.rules]]
tool = "bash"
action = "deny"

[[tool_policy.rules]]
tool = "gmail_send_*"
action = "warn"
"#;
        let config: GuardrailsConfig = toml::from_str(toml_str).unwrap();
        let guardrails = config.build().unwrap();
        assert_eq!(guardrails.len(), 3);
    }

    #[test]
    fn guardrails_config_llm_judge_from_toml() {
        let toml_str = r#"
[llm_judge]
criteria = ["no harmful content", "no personal attacks"]
evaluate_tool_inputs = true
timeout_seconds = 15
max_judge_tokens = 512
"#;
        let config: GuardrailsConfig = toml::from_str(toml_str).unwrap();
        let judge_cfg = config.llm_judge.as_ref().expect("llm_judge should be set");
        assert_eq!(judge_cfg.criteria.len(), 2);
        assert!(judge_cfg.evaluate_tool_inputs);
        assert_eq!(judge_cfg.timeout_seconds, 15);
        assert_eq!(judge_cfg.max_judge_tokens, 512);
    }

    #[test]
    fn guardrails_config_llm_judge_defaults() {
        let toml_str = r#"
[llm_judge]
criteria = ["safety"]
"#;
        let config: GuardrailsConfig = toml::from_str(toml_str).unwrap();
        let judge_cfg = config.llm_judge.as_ref().expect("llm_judge should be set");
        assert!(!judge_cfg.evaluate_tool_inputs);
        assert_eq!(judge_cfg.timeout_seconds, 10);
        assert_eq!(judge_cfg.max_judge_tokens, 256);
    }

    #[test]
    fn guardrails_config_build_skips_judge_without_provider() {
        let config = GuardrailsConfig {
            llm_judge: Some(LlmJudgeConfig {
                criteria: vec!["safety".into()],
                evaluate_tool_inputs: false,
                timeout_seconds: 10,
                max_judge_tokens: 256,
            }),
            ..Default::default()
        };
        // build() delegates to build_with_judge(None) — skips LLM judge
        let guardrails = config.build().unwrap();
        assert_eq!(guardrails.len(), 0);
    }

    #[test]
    fn guardrails_config_is_empty_with_only_llm_judge() {
        let config = GuardrailsConfig {
            llm_judge: Some(LlmJudgeConfig {
                criteria: vec!["safety".into()],
                evaluate_tool_inputs: false,
                timeout_seconds: 10,
                max_judge_tokens: 256,
            }),
            ..Default::default()
        };
        assert!(!config.is_empty());
    }

    #[test]
    fn parse_local_embedding_config() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test agent"
system_prompt = "You are a test agent."

[memory]
type = "postgres"
database_url = "postgresql://localhost/heartbit"

[memory.embedding]
provider = "local"
model = "all-MiniLM-L6-v2"
cache_dir = "/tmp/fastembed"
"#;

        let config = HeartbitConfig::from_toml(toml).unwrap();
        let memory = config.memory.expect("memory should be present");
        match memory {
            MemoryConfig::Postgres { embedding, .. } => {
                let emb = embedding.expect("embedding config should be present");
                assert_eq!(emb.provider, "local");
                assert_eq!(emb.model, "all-MiniLM-L6-v2");
                assert_eq!(emb.cache_dir.as_deref(), Some("/tmp/fastembed"));
            }
            _ => panic!("expected Postgres memory config"),
        }
    }

    #[test]
    fn parse_local_embedding_config_defaults() {
        // provider = "local" without model or cache_dir — should use defaults
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "test"
description = "Test agent"
system_prompt = "You are a test agent."

[memory]
type = "postgres"
database_url = "postgresql://localhost/heartbit"

[memory.embedding]
provider = "local"
"#;

        let config = HeartbitConfig::from_toml(toml).unwrap();
        let memory = config.memory.expect("memory should be present");
        match memory {
            MemoryConfig::Postgres { embedding, .. } => {
                let emb = embedding.expect("embedding config should be present");
                assert_eq!(emb.provider, "local");
                // model defaults to "text-embedding-3-small" (OpenAI default)
                // CLI handles this by treating it as "unset" → uses AllMiniLML6V2
                assert_eq!(emb.model, "text-embedding-3-small");
                assert!(emb.cache_dir.is_none());
                assert!(emb.base_url.is_none());
                assert!(emb.dimension.is_none());
            }
            _ => panic!("expected Postgres memory config"),
        }
    }

    #[test]
    fn auth_config_backward_compat() {
        let toml_str = r#"
            bearer_tokens = ["tok-abc", "tok-xyz"]
        "#;
        let auth: AuthConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(auth.bearer_tokens, vec!["tok-abc", "tok-xyz"]);
        assert!(auth.jwks_url.is_none());
        assert!(auth.issuer.is_none());
        assert!(auth.audience.is_none());
        assert!(auth.user_id_claim.is_none());
        assert!(auth.tenant_id_claim.is_none());
        assert!(auth.roles_claim.is_none());
        assert!(auth.token_exchange.is_none());
    }

    #[test]
    fn auth_config_with_jwks() {
        let toml_str = r#"
            bearer_tokens = ["tok-1"]
            jwks_url = "https://idp.example.com/.well-known/jwks.json"
            issuer = "https://idp.example.com"
            audience = "heartbit-api"
            user_id_claim = "sub"
            tenant_id_claim = "org_id"
            roles_claim = "permissions"
        "#;
        let auth: AuthConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(auth.bearer_tokens, vec!["tok-1"]);
        assert_eq!(
            auth.jwks_url.as_deref(),
            Some("https://idp.example.com/.well-known/jwks.json")
        );
        assert_eq!(auth.issuer.as_deref(), Some("https://idp.example.com"));
        assert_eq!(auth.audience.as_deref(), Some("heartbit-api"));
        assert_eq!(auth.user_id_claim.as_deref(), Some("sub"));
        assert_eq!(auth.tenant_id_claim.as_deref(), Some("org_id"));
        assert_eq!(auth.roles_claim.as_deref(), Some("permissions"));
    }

    #[test]
    fn auth_config_empty_is_valid() {
        let toml_str = "";
        let auth: AuthConfig = toml::from_str(toml_str).unwrap();
        assert!(auth.bearer_tokens.is_empty());
        assert!(auth.jwks_url.is_none());
        assert!(auth.issuer.is_none());
        assert!(auth.audience.is_none());
        assert!(auth.user_id_claim.is_none());
        assert!(auth.tenant_id_claim.is_none());
        assert!(auth.roles_claim.is_none());
    }

    #[test]
    fn auth_config_mixed() {
        let toml_str = r#"
            bearer_tokens = ["static-key"]
            jwks_url = "https://auth.corp.io/.well-known/jwks.json"
            audience = "heartbit"
        "#;
        let auth: AuthConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(auth.bearer_tokens, vec!["static-key"]);
        assert_eq!(
            auth.jwks_url.as_deref(),
            Some("https://auth.corp.io/.well-known/jwks.json")
        );
        assert!(auth.issuer.is_none());
        assert_eq!(auth.audience.as_deref(), Some("heartbit"));
        assert!(auth.user_id_claim.is_none());
        assert!(auth.tenant_id_claim.is_none());
        assert!(auth.roles_claim.is_none());
    }

    #[test]
    fn mcp_resource_mode_default_is_tools() {
        let mode: McpResourceMode = Default::default();
        assert_eq!(mode, McpResourceMode::Tools);
    }

    #[test]
    fn mcp_resource_mode_deserialize() {
        #[derive(Deserialize)]
        struct Wrapper {
            mode: McpResourceMode,
        }
        let w: Wrapper = toml::from_str(r#"mode = "tools""#).unwrap();
        assert_eq!(w.mode, McpResourceMode::Tools);
        let w: Wrapper = toml::from_str(r#"mode = "context""#).unwrap();
        assert_eq!(w.mode, McpResourceMode::Context);
        let w: Wrapper = toml::from_str(r#"mode = "none""#).unwrap();
        assert_eq!(w.mode, McpResourceMode::None);
    }

    #[test]
    fn agent_config_mcp_resources_default() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_turns = 10

[[agents]]
name = "test"
description = "A test agent"
system_prompt = "You are a test."
"#;
        let config: HeartbitConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.agents[0].mcp_resources, McpResourceMode::Tools);
    }

    #[test]
    fn agent_config_mcp_resources_explicit() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]
max_turns = 10

[[agents]]
name = "test"
description = "A test agent"
system_prompt = "You are a test."
mcp_resources = "none"
"#;
        let config: HeartbitConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.agents[0].mcp_resources, McpResourceMode::None);
    }

    // daemon_mcp_server tests removed — mcp_server field moved to gateway crate.

    #[test]
    fn parse_workflow_type_valid() {
        use crate::agent::workflow::WorkflowType;
        assert_eq!(parse_workflow_type("dag").unwrap(), WorkflowType::Dag);
        assert_eq!(parse_workflow_type("DAG").unwrap(), WorkflowType::Dag);
        assert_eq!(
            parse_workflow_type("sequential").unwrap(),
            WorkflowType::Sequential
        );
        assert_eq!(
            parse_workflow_type("parallel").unwrap(),
            WorkflowType::Parallel
        );
        assert_eq!(parse_workflow_type("loop").unwrap(), WorkflowType::Loop);
        assert_eq!(parse_workflow_type("debate").unwrap(), WorkflowType::Debate);
        assert_eq!(parse_workflow_type("voting").unwrap(), WorkflowType::Voting);
        assert_eq!(
            parse_workflow_type("mixture").unwrap(),
            WorkflowType::Mixture
        );
    }

    #[test]
    fn parse_workflow_type_invalid() {
        assert!(parse_workflow_type("").is_err());
        assert!(parse_workflow_type("unknown").is_err());
        assert!(parse_workflow_type("rm -rf /").is_err());
    }

    #[test]
    fn validate_rejects_unknown_builtin() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "researcher"
description = "Research specialist"
builtin_tools = ["websearch", "nonexistent"]
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown builtin tool 'nonexistent'"),
            "expected unknown builtin error, got: {msg}"
        );
    }

    #[test]
    fn agent_config_builtin_tools_deserialization() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "researcher"
description = "Research specialist"
builtin_tools = ["websearch", "webfetch"]

[[agents]]
name = "publisher"
description = "Publisher"
builtin_tools = []
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(
            config.agents[0].builtin_tools,
            Some(vec!["websearch".into(), "webfetch".into()])
        );
        assert_eq!(config.agents[1].builtin_tools, Some(vec![]));
    }

    #[test]
    fn agent_config_builtin_tools_absent_is_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[[agents]]
name = "researcher"
description = "Research specialist"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert_eq!(config.agents[0].builtin_tools, None);
    }
}
