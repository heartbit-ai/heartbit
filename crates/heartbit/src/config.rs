use serde::Deserialize;

use crate::Error;

/// Top-level configuration loaded from `heartbit.toml`.
#[derive(Debug, Deserialize)]
pub struct HeartbitConfig {
    pub provider: ProviderConfig,
    #[serde(default)]
    pub orchestrator: OrchestratorConfig,
    #[serde(default)]
    pub agents: Vec<AgentConfig>,
    pub restate: Option<RestateConfig>,
    pub telemetry: Option<TelemetryConfig>,
    pub memory: Option<MemoryConfig>,
    pub knowledge: Option<KnowledgeConfig>,
}

/// LLM provider configuration.
#[derive(Debug, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub model: String,
    /// Retry configuration for transient LLM API failures.
    pub retry: Option<RetryProviderConfig>,
    /// Enable Anthropic prompt caching (system prompt + tool definitions).
    /// Only effective for the `anthropic` provider. Defaults to `false`.
    #[serde(default)]
    pub prompt_caching: bool,
}

/// Retry configuration for transient LLM API failures (429, 500, 502, 503, 529).
#[derive(Debug, Deserialize)]
pub struct RetryProviderConfig {
    /// Maximum retry attempts (default: 3).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Base delay in milliseconds for exponential backoff (default: 500).
    #[serde(default = "default_base_delay_ms")]
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds (default: 30000).
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,
}

fn default_max_retries() -> u32 {
    3
}

fn default_base_delay_ms() -> u64 {
    500
}

fn default_max_delay_ms() -> u64 {
    30_000
}

/// Orchestrator-level settings with sensible defaults.
#[derive(Debug, Deserialize)]
pub struct OrchestratorConfig {
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

fn default_max_turns() -> usize {
    10
}

fn default_max_tokens() -> u32 {
    4096
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_turns: default_max_turns(),
            max_tokens: default_max_tokens(),
        }
    }
}

/// Context window management strategy.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextStrategyConfig {
    /// No trimming (default).
    Unlimited,
    /// Sliding window: trim old messages to stay within `max_tokens`.
    SlidingWindow { max_tokens: u32 },
    /// Summarize: compress old messages when context exceeds `threshold` tokens.
    Summarize { threshold: u32 },
}

/// A sub-agent defined in the configuration file.
#[derive(Debug, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    #[serde(default)]
    pub mcp_servers: Vec<String>,
    /// Context window management strategy for this agent.
    pub context_strategy: Option<ContextStrategyConfig>,
    /// Timeout in seconds for individual tool executions.
    pub tool_timeout_seconds: Option<u64>,
    /// Maximum byte size for individual tool output. Results exceeding this
    /// limit are truncated with a `[truncated]` suffix.
    pub max_tool_output_bytes: Option<usize>,
    /// Per-agent turn limit. Overrides the orchestrator default when set.
    pub max_turns: Option<usize>,
    /// Per-agent token limit. Overrides the orchestrator default when set.
    pub max_tokens: Option<u32>,
    /// Optional JSON Schema for structured output. Expressed as an inline
    /// TOML table that maps to the JSON Schema object. When set, the agent
    /// receives a synthetic `__respond__` tool and returns structured JSON.
    pub response_schema: Option<serde_json::Value>,
}

/// Memory configuration for the orchestrator.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemoryConfig {
    /// In-memory store (for development/testing).
    InMemory,
    /// PostgreSQL-backed store.
    Postgres { database_url: String },
}

/// Knowledge base configuration for document retrieval.
#[derive(Debug, Deserialize)]
pub struct KnowledgeConfig {
    /// Maximum byte length per chunk.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Number of overlapping bytes between consecutive chunks.
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    /// Document sources to index.
    #[serde(default)]
    pub sources: Vec<KnowledgeSourceConfig>,
}

fn default_chunk_size() -> usize {
    1000
}

fn default_chunk_overlap() -> usize {
    200
}

/// A single knowledge source to index.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KnowledgeSourceConfig {
    /// A single file path.
    File { path: String },
    /// A glob pattern matching multiple files.
    Glob { pattern: String },
    /// A URL to fetch and index.
    Url { url: String },
}

/// Restate server connection settings.
#[derive(Debug, Deserialize)]
pub struct RestateConfig {
    pub endpoint: String,
}

/// OpenTelemetry configuration.
#[derive(Debug, Deserialize)]
pub struct TelemetryConfig {
    /// OTLP endpoint (e.g., "http://localhost:4317")
    pub otlp_endpoint: String,
    /// Service name reported to the collector
    #[serde(default = "default_service_name")]
    pub service_name: String,
}

fn default_service_name() -> String {
    "heartbit".into()
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

        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
            vec!["http://localhost:8000/mcp"]
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
        assert!(matches!(config.memory, Some(MemoryConfig::Postgres { .. })));
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
}
