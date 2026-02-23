use serde::{Deserialize, Serialize};

use crate::Error;
use crate::agent::permission::PermissionRule;
use crate::llm::types::ReasoningEffort;

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
    /// Declarative permission rules applied to all agents.
    /// Rules are evaluated in order — first match wins.
    #[serde(default)]
    pub permissions: Vec<PermissionRule>,
    /// Optional LSP integration for diagnostics after file-modifying tools.
    pub lsp: Option<LspConfig>,
    /// Daemon mode configuration for Kafka-backed long-running execution.
    pub daemon: Option<DaemonConfig>,
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

/// Dispatch mode for orchestrator delegation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchMode {
    /// All delegated tasks run in parallel via JoinSet (default).
    Parallel,
    /// One task at a time. Schema constrains `maxItems: 1` on delegate_task.
    Sequential,
}

/// Orchestrator-level settings with sensible defaults.
#[derive(Debug, Deserialize)]
pub struct OrchestratorConfig {
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Context window management strategy for the orchestrator's own conversation.
    pub context_strategy: Option<ContextStrategyConfig>,
    /// Token threshold for summarization of the orchestrator's own context.
    pub summarize_threshold: Option<u32>,
    /// Timeout in seconds for the orchestrator's own tool calls.
    pub tool_timeout_seconds: Option<u64>,
    /// Maximum byte size for tool output on the orchestrator's own tools.
    pub max_tool_output_bytes: Option<usize>,
    /// Wall-clock deadline in seconds for the entire orchestrator run.
    pub run_timeout_seconds: Option<u64>,
    /// Enable the `form_squad` tool for dynamic agent squad formation.
    /// When `None` (default), auto-enabled when there are >= 2 agents.
    /// Set to `false` to disable for a simpler prompt with fewer tokens.
    pub enable_squads: Option<bool>,
    /// Reasoning/thinking effort level. Enables extended thinking on models
    /// that support it (e.g., Qwen3 via OpenRouter, Claude with extended thinking).
    /// Valid values: "high", "medium", "low", "none".
    pub reasoning_effort: Option<String>,
    /// Enable reflection prompts after tool results. When true, the agent pauses
    /// to assess tool outputs before deciding the next action (Reflexion/CRITIC pattern).
    pub enable_reflection: Option<bool>,
    /// Tool output compression threshold in bytes. Outputs exceeding this size
    /// are compressed via an LLM call that preserves factual content.
    pub tool_output_compression_threshold: Option<usize>,
    /// Maximum number of tool definitions sent per LLM turn. When agents have
    /// many tools, filtering to the most relevant reduces context usage and cost.
    pub max_tools_per_turn: Option<usize>,
    /// Maximum consecutive identical tool-call turns before doom loop detection
    /// triggers. When reached, tool calls get error results instead of executing.
    pub max_identical_tool_calls: Option<u32>,
    /// Dispatch mode for orchestrator delegation. When `Sequential`, the
    /// delegate_task schema constrains `maxItems: 1` so the LLM dispatches
    /// one agent at a time. Defaults to `Parallel` when absent.
    pub dispatch_mode: Option<DispatchMode>,
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
            context_strategy: None,
            summarize_threshold: None,
            tool_timeout_seconds: None,
            max_tool_output_bytes: None,
            run_timeout_seconds: None,
            enable_squads: None,
            reasoning_effort: None,
            enable_reflection: None,
            tool_output_compression_threshold: None,
            max_tools_per_turn: None,
            max_identical_tool_calls: None,
            dispatch_mode: None,
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

/// An MCP server entry: either a bare URL string or a full config with auth.
///
/// Supports backward-compatible TOML: bare strings (`"http://..."`) deserialize
/// as `Simple`, while inline tables (`{ url = "...", auth_header = "..." }`)
/// deserialize as `Full`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum McpServerEntry {
    /// Bare URL string (backward-compatible).
    Simple(String),
    /// Full entry with optional auth header.
    Full {
        url: String,
        #[serde(default)]
        auth_header: Option<String>,
    },
}

impl McpServerEntry {
    /// Get the server URL.
    pub fn url(&self) -> &str {
        match self {
            McpServerEntry::Simple(url) => url,
            McpServerEntry::Full { url, .. } => url,
        }
    }

    /// Get the optional auth header value.
    pub fn auth_header(&self) -> Option<&str> {
        match self {
            McpServerEntry::Simple(_) => None,
            McpServerEntry::Full { auth_header, .. } => auth_header.as_deref(),
        }
    }
}

/// Per-agent provider override. When set on an agent, overrides the
/// orchestrator's default provider for that agent only.
#[derive(Debug, Deserialize)]
pub struct AgentProviderConfig {
    pub name: String,
    pub model: String,
    /// Enable Anthropic prompt caching for this agent.
    #[serde(default)]
    pub prompt_caching: bool,
}

/// A sub-agent defined in the configuration file.
#[derive(Debug, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerEntry>,
    /// A2A agent endpoints to discover and register as tools.
    #[serde(default)]
    pub a2a_agents: Vec<McpServerEntry>,
    /// Context window management strategy for this agent.
    pub context_strategy: Option<ContextStrategyConfig>,
    /// Token threshold at which to trigger automatic summarization.
    /// Only valid when `context_strategy` is not `SlidingWindow`.
    pub summarize_threshold: Option<u32>,
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
    /// Wall-clock deadline in seconds for this agent's run.
    pub run_timeout_seconds: Option<u64>,
    /// Optional per-agent LLM provider override. When set, this agent uses
    /// a different model/provider instead of the orchestrator's default.
    pub provider: Option<AgentProviderConfig>,
    /// Reasoning/thinking effort level. Overrides the orchestrator default.
    /// Valid values: "high", "medium", "low", "none".
    pub reasoning_effort: Option<String>,
    /// Enable reflection prompts after tool results. Overrides the orchestrator default.
    pub enable_reflection: Option<bool>,
    /// Tool output compression threshold in bytes. Overrides the orchestrator default.
    pub tool_output_compression_threshold: Option<usize>,
    /// Maximum tools per turn for this agent. Overrides the orchestrator default.
    pub max_tools_per_turn: Option<usize>,
    /// Maximum consecutive identical tool-call turns before doom loop detection.
    /// Overrides the orchestrator default.
    pub max_identical_tool_calls: Option<u32>,
    /// Session pruning: truncate old tool results to save tokens.
    /// When set, enables session-level pruning before each LLM call.
    pub session_prune: Option<SessionPruneConfigToml>,
    /// Enable recursive (cluster-then-summarize) summarization for long conversations.
    pub recursive_summarization: Option<bool>,
    /// Cumulative importance threshold for memory reflection triggers.
    /// When the sum of stored memory importance values exceeds this threshold,
    /// the store tool appends a reflection hint to guide the agent.
    pub reflection_threshold: Option<u32>,
    /// When true, run memory consolidation at session end (clusters related
    /// episodic memories into semantic summaries). Requires memory and adds
    /// LLM calls at session end.
    pub consolidate_on_exit: Option<bool>,
}

/// TOML representation of session pruning configuration.
#[derive(Debug, Deserialize)]
pub struct SessionPruneConfigToml {
    /// Number of recent message pairs to keep at full fidelity. Default: 2.
    #[serde(default = "default_keep_recent_n")]
    pub keep_recent_n: usize,
    /// Maximum bytes for a pruned tool result. Default: 200.
    #[serde(default = "default_pruned_max_bytes")]
    pub pruned_tool_result_max_bytes: usize,
    /// Whether to preserve the first user message (task). Default: true.
    #[serde(default = "default_preserve_task")]
    pub preserve_task: bool,
}

fn default_keep_recent_n() -> usize {
    2
}

fn default_pruned_max_bytes() -> usize {
    200
}

fn default_preserve_task() -> bool {
    true
}

/// Memory configuration for the orchestrator.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemoryConfig {
    /// In-memory store (for development/testing).
    InMemory,
    /// PostgreSQL-backed store.
    Postgres {
        database_url: String,
        /// Optional embedding configuration for hybrid retrieval.
        #[serde(default)]
        embedding: Option<EmbeddingConfig>,
    },
}

/// Configuration for embedding generation (optional).
///
/// When configured under `[memory.embedding]`, enables hybrid retrieval
/// (BM25 + vector cosine) for improved recall quality.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    /// Provider name: "openai" or "none" (default).
    #[serde(default = "default_embedding_provider")]
    pub provider: String,
    /// Model name for the embedding API.
    #[serde(default = "default_embedding_model")]
    pub model: String,
    /// Environment variable name containing the API key.
    #[serde(default = "default_embedding_api_key_env")]
    pub api_key_env: String,
    /// Base URL for the embedding API (optional, defaults to OpenAI).
    pub base_url: Option<String>,
    /// Embedding vector dimension (auto-detected from model if omitted).
    pub dimension: Option<usize>,
}

fn default_embedding_provider() -> String {
    "none".into()
}

fn default_embedding_model() -> String {
    "text-embedding-3-small".into()
}

fn default_embedding_api_key_env() -> String {
    "OPENAI_API_KEY".into()
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
    /// Observability verbosity mode: "production", "analysis", or "debug".
    /// Overridden by `HEARTBIT_OBSERVABILITY` env var.
    #[serde(default)]
    pub observability_mode: Option<String>,
}

fn default_service_name() -> String {
    "heartbit".into()
}

/// LSP integration configuration.
///
/// When present, language servers are spawned lazily after file-modifying tools
/// and diagnostics are appended to tool output.
#[derive(Debug, Deserialize)]
pub struct LspConfig {
    /// Whether LSP integration is enabled. Default: `true` when the section is present.
    #[serde(default = "default_lsp_enabled")]
    pub enabled: bool,
}

fn default_lsp_enabled() -> bool {
    true
}

fn default_daemon_bind() -> String {
    "127.0.0.1:3000".into()
}

fn default_max_concurrent() -> usize {
    4
}

fn default_consumer_group() -> String {
    "heartbit-daemon".into()
}

fn default_commands_topic() -> String {
    "heartbit.commands".into()
}

fn default_events_topic() -> String {
    "heartbit.events".into()
}

fn default_dead_letter_topic() -> String {
    "heartbit.dead-letter".into()
}

fn default_true() -> bool {
    true
}

/// Prometheus metrics configuration for daemon mode.
#[derive(Debug, Clone, Deserialize)]
pub struct MetricsConfig {
    /// Whether Prometheus metrics are enabled. Defaults to `true`.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

/// Daemon mode configuration for long-running Kafka-backed execution.
#[derive(Debug, Clone, Deserialize)]
pub struct DaemonConfig {
    pub kafka: KafkaConfig,
    #[serde(default = "default_daemon_bind")]
    pub bind: String,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_tasks: usize,
    #[serde(default)]
    pub schedules: Vec<ScheduleEntry>,
    /// Prometheus metrics configuration. Metrics are enabled by default.
    #[serde(default)]
    pub metrics: Option<MetricsConfig>,
    /// Sensor layer configuration for continuous perception.
    pub sensors: Option<SensorConfig>,
    /// WebSocket configuration for bidirectional user↔agent communication.
    pub ws: Option<WsConfig>,
    /// Telegram bot configuration for interactive chat via Telegram DMs.
    #[cfg(feature = "telegram")]
    pub telegram: Option<crate::channel::telegram::TelegramConfig>,
    /// PostgreSQL URL for durable task persistence. When absent, tasks are
    /// stored in-memory (lost on restart).
    #[serde(default)]
    pub database_url: Option<String>,
}

/// WebSocket configuration for bidirectional user↔agent communication.
#[derive(Debug, Clone, Deserialize)]
pub struct WsConfig {
    /// Whether WebSocket endpoint is enabled. Defaults to `true`.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Timeout in seconds for blocking interactions (approval, input, question).
    /// Defaults to 120 seconds.
    #[serde(default = "default_interaction_timeout")]
    pub interaction_timeout_seconds: u64,
    /// Maximum concurrent WebSocket connections. Defaults to 100.
    #[serde(default = "default_max_ws_connections")]
    pub max_connections: usize,
    /// PostgreSQL URL for durable session persistence. When absent, sessions
    /// are stored in-memory (lost on restart).
    #[serde(default)]
    pub database_url: Option<String>,
}

fn default_interaction_timeout() -> u64 {
    120
}

fn default_max_ws_connections() -> usize {
    100
}

/// Sensor layer configuration for continuous perception.
#[derive(Debug, Clone, Deserialize)]
pub struct SensorConfig {
    /// Master switch for the sensor layer. Defaults to `true`.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Model routing configuration for triage decisions.
    #[serde(default)]
    pub routing: Option<SensorRoutingConfig>,
    /// Salience scoring weights for triage promotion.
    #[serde(default)]
    pub salience: Option<SalienceConfig>,
    /// Token budget limits for the sensor pipeline.
    #[serde(default)]
    pub token_budget: Option<TokenBudgetConfig>,
    /// Story correlation settings.
    #[serde(default)]
    pub stories: Option<StoryCorrelationConfig>,
    /// Sensor source definitions.
    #[serde(default)]
    pub sources: Vec<SensorSourceConfig>,
}

/// Model routing configuration for sensor triage.
#[derive(Debug, Clone, Deserialize)]
pub struct SensorRoutingConfig {
    /// Which model tier to use for triage: "local", "cloud_light", "cloud_frontier".
    #[serde(default = "default_triage_model")]
    pub triage_model: String,
    /// Path to local GGUF model file (for local SLM inference).
    pub local_model_path: Option<String>,
    /// Confidence threshold below which to escalate to a higher model tier.
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
}

fn default_triage_model() -> String {
    "cloud_light".into()
}

fn default_confidence_threshold() -> f64 {
    0.85
}

/// Salience scoring weights for triage promotion decisions.
#[derive(Debug, Clone, Deserialize)]
pub struct SalienceConfig {
    /// Weight for urgency signals (0.0-1.0).
    #[serde(default = "default_urgency_weight")]
    pub urgency_weight: f64,
    /// Weight for novelty signals (0.0-1.0).
    #[serde(default = "default_novelty_weight")]
    pub novelty_weight: f64,
    /// Weight for relevance signals (0.0-1.0).
    #[serde(default = "default_relevance_weight")]
    pub relevance_weight: f64,
    /// Minimum salience score for promotion (0.0-1.0).
    #[serde(default = "default_salience_threshold")]
    pub threshold: f64,
}

fn default_urgency_weight() -> f64 {
    0.3
}

fn default_novelty_weight() -> f64 {
    0.3
}

fn default_relevance_weight() -> f64 {
    0.4
}

fn default_salience_threshold() -> f64 {
    0.3
}

/// Token budget limits for the sensor pipeline.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenBudgetConfig {
    /// Maximum tokens per hour across all sensor processing.
    #[serde(default = "default_hourly_limit")]
    pub hourly_limit: usize,
    /// Maximum queued events before back-pressure.
    #[serde(default = "default_queue_size")]
    pub queue_size: usize,
}

fn default_hourly_limit() -> usize {
    100_000
}

fn default_queue_size() -> usize {
    200
}

/// Story correlation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct StoryCorrelationConfig {
    /// Time window in hours for correlating events into stories.
    #[serde(default = "default_correlation_window_hours")]
    pub correlation_window_hours: u64,
    /// Maximum events tracked per story before archival.
    #[serde(default = "default_max_events_per_story")]
    pub max_events_per_story: usize,
    /// Hours of inactivity after which a story is marked stale.
    #[serde(default = "default_stale_after_hours")]
    pub stale_after_hours: u64,
}

fn default_correlation_window_hours() -> u64 {
    4
}

fn default_max_events_per_story() -> usize {
    50
}

fn default_stale_after_hours() -> u64 {
    24
}

/// A sensor source definition.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SensorSourceConfig {
    /// JMAP email sensor (push/poll).
    JmapEmail {
        name: String,
        server: String,
        username: String,
        /// Environment variable containing the password.
        password_env: String,
        /// Senders that get automatic `Priority::High`.
        #[serde(default)]
        priority_senders: Vec<String>,
        /// Senders whose emails are silently dropped during triage.
        #[serde(default)]
        blocked_senders: Vec<String>,
        #[serde(default = "default_email_poll_interval")]
        poll_interval_seconds: u64,
    },
    /// RSS/Atom feed sensor.
    Rss {
        name: String,
        feeds: Vec<String>,
        #[serde(default)]
        interest_keywords: Vec<String>,
        #[serde(default = "default_rss_poll_interval")]
        poll_interval_seconds: u64,
    },
    /// Directory watcher for images.
    Image {
        name: String,
        watch_directory: String,
        #[serde(default = "default_file_poll_interval")]
        poll_interval_seconds: u64,
    },
    /// Directory watcher for audio files.
    Audio {
        name: String,
        watch_directory: String,
        /// Whisper model size: "tiny", "base", "small", "medium", "large".
        #[serde(default = "default_whisper_model")]
        whisper_model: String,
        /// Known contacts whose voice recordings get priority triage.
        #[serde(default)]
        known_contacts: Vec<String>,
        #[serde(default = "default_file_poll_interval")]
        poll_interval_seconds: u64,
    },
    /// Weather API sensor.
    Weather {
        name: String,
        /// Environment variable containing the API key.
        api_key_env: String,
        locations: Vec<String>,
        #[serde(default = "default_weather_poll_interval")]
        poll_interval_seconds: u64,
        /// When true, only promote weather alerts (not regular readings).
        #[serde(default)]
        alert_only: bool,
    },
    /// Generic webhook receiver.
    Webhook {
        name: String,
        /// URL path for the webhook endpoint (e.g., "/webhooks/github").
        path: String,
        /// Environment variable containing the webhook secret.
        secret_env: Option<String>,
    },
}

impl SensorSourceConfig {
    /// Get the name of this sensor source.
    pub fn name(&self) -> &str {
        match self {
            SensorSourceConfig::JmapEmail { name, .. }
            | SensorSourceConfig::Rss { name, .. }
            | SensorSourceConfig::Image { name, .. }
            | SensorSourceConfig::Audio { name, .. }
            | SensorSourceConfig::Weather { name, .. }
            | SensorSourceConfig::Webhook { name, .. } => name,
        }
    }
}

fn default_email_poll_interval() -> u64 {
    60
}

fn default_rss_poll_interval() -> u64 {
    900
}

fn default_file_poll_interval() -> u64 {
    30
}

fn default_whisper_model() -> String {
    "base".into()
}

fn default_weather_poll_interval() -> u64 {
    1800
}

/// Kafka broker connection settings.
#[derive(Debug, Clone, Deserialize)]
pub struct KafkaConfig {
    pub brokers: String,
    #[serde(default = "default_consumer_group")]
    pub consumer_group: String,
    #[serde(default = "default_commands_topic")]
    pub commands_topic: String,
    #[serde(default = "default_events_topic")]
    pub events_topic: String,
    /// Topic for events that failed triage processing.
    #[serde(default = "default_dead_letter_topic")]
    pub dead_letter_topic: String,
}

/// A scheduled task entry for the cron scheduler.
#[derive(Debug, Clone, Deserialize)]
pub struct ScheduleEntry {
    pub name: String,
    pub cron: String,
    pub task: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
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
        if self.orchestrator.max_identical_tool_calls == Some(0) {
            return Err(Error::Config(
                "orchestrator.max_identical_tool_calls must be at least 1".into(),
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

        // Validate sensor config
        if let Some(ref daemon) = self.daemon
            && let Some(ref sensors) = daemon.sensors
        {
            if let Some(ref routing) = sensors.routing {
                let valid = ["local", "cloud_light", "cloud_frontier"];
                if !valid.contains(&routing.triage_model.as_str()) {
                    return Err(Error::Config(format!(
                        "daemon.sensors.routing.triage_model '{}' is invalid; \
                         must be one of: {}",
                        routing.triage_model,
                        valid.join(", ")
                    )));
                }
                if !(0.0..=1.0).contains(&routing.confidence_threshold) {
                    return Err(Error::Config(
                        "daemon.sensors.routing.confidence_threshold must be between 0.0 and 1.0"
                            .into(),
                    ));
                }
            }
            if let Some(ref salience) = sensors.salience {
                let sum =
                    salience.urgency_weight + salience.novelty_weight + salience.relevance_weight;
                if (sum - 1.0).abs() > 0.01 {
                    return Err(Error::Config(format!(
                        "daemon.sensors.salience weights must sum to ~1.0, got {sum:.3}"
                    )));
                }
                if !(0.0..=1.0).contains(&salience.threshold) {
                    return Err(Error::Config(
                        "daemon.sensors.salience.threshold must be between 0.0 and 1.0".into(),
                    ));
                }
            }
            if let Some(ref budget) = sensors.token_budget {
                if budget.hourly_limit == 0 {
                    return Err(Error::Config(
                        "daemon.sensors.token_budget.hourly_limit must be at least 1".into(),
                    ));
                }
                if budget.queue_size == 0 {
                    return Err(Error::Config(
                        "daemon.sensors.token_budget.queue_size must be at least 1".into(),
                    ));
                }
            }
            if let Some(ref stories) = sensors.stories {
                if stories.correlation_window_hours == 0 {
                    return Err(Error::Config(
                        "daemon.sensors.stories.correlation_window_hours must be at least 1".into(),
                    ));
                }
                if stories.stale_after_hours == 0 {
                    return Err(Error::Config(
                        "daemon.sensors.stories.stale_after_hours must be at least 1".into(),
                    ));
                }
                if stories.max_events_per_story == 0 {
                    return Err(Error::Config(
                        "daemon.sensors.stories.max_events_per_story must be at least 1".into(),
                    ));
                }
            }
            // Validate source names are unique
            let mut source_names = std::collections::HashSet::new();
            for source in &sensors.sources {
                let name = source.name();
                if name.is_empty() {
                    return Err(Error::Config(
                        "daemon.sensors.sources[].name must not be empty".into(),
                    ));
                }
                if !source_names.insert(name.to_string()) {
                    return Err(Error::Config(format!(
                        "duplicate sensor source name: '{name}'"
                    )));
                }
            }
        }

        // Validate daemon config
        if let Some(ref daemon) = self.daemon {
            if daemon.max_concurrent_tasks == 0 {
                return Err(Error::Config(
                    "daemon.max_concurrent_tasks must be at least 1".into(),
                ));
            }
            if daemon.kafka.brokers.is_empty() {
                return Err(Error::Config(
                    "daemon.kafka.brokers must not be empty".into(),
                ));
            }
            let mut schedule_names = std::collections::HashSet::new();
            for (i, schedule) in daemon.schedules.iter().enumerate() {
                if schedule.name.is_empty() {
                    return Err(Error::Config(format!(
                        "daemon.schedules[{i}].name must not be empty"
                    )));
                }
                if !schedule_names.insert(&schedule.name) {
                    return Err(Error::Config(format!(
                        "duplicate daemon schedule name: '{}'",
                        schedule.name
                    )));
                }
                if schedule.cron.parse::<cron::Schedule>().is_err() {
                    return Err(Error::Config(format!(
                        "daemon.schedules[{i}] '{}': invalid cron expression '{}'",
                        schedule.name, schedule.cron
                    )));
                }
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
        };
        let json = serde_json::to_string(&full).unwrap();
        let parsed: McpServerEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(full, parsed);
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

[[daemon.schedules]]
name = "daily-review"
cron = "0 0 9 * * *"
task = "Review yesterday's work"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        assert_eq!(daemon.bind, "0.0.0.0:8080");
        assert_eq!(daemon.max_concurrent_tasks, 8);
        assert_eq!(daemon.kafka.brokers, "localhost:9092");
        assert_eq!(daemon.kafka.consumer_group, "heartbit-daemon");
        assert_eq!(daemon.kafka.commands_topic, "heartbit.commands");
        assert_eq!(daemon.kafka.events_topic, "heartbit.events");
        assert_eq!(daemon.kafka.dead_letter_topic, "heartbit.dead-letter");
        assert_eq!(daemon.schedules.len(), 1);
        assert_eq!(daemon.schedules[0].name, "daily-review");
        assert!(daemon.schedules[0].enabled);
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
        assert!(daemon.schedules.is_empty());
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
    fn daemon_duplicate_schedule_names_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.schedules]]
name = "daily"
cron = "0 0 9 * * *"
task = "First"

[[daemon.schedules]]
name = "daily"
cron = "0 0 18 * * *"
task = "Second"
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("duplicate daemon schedule name"),
            "error: {msg}"
        );
    }

    #[test]
    fn daemon_empty_schedule_name_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.schedules]]
name = ""
cron = "0 0 9 * * *"
task = "Something"
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("name must not be empty"), "error: {msg}");
    }

    #[test]
    fn daemon_invalid_cron_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.schedules]]
name = "bad"
cron = "not a cron"
task = "Something"
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid cron expression"), "error: {msg}");
    }

    #[test]
    fn daemon_schedule_enabled_defaults_true() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.schedules]]
name = "test"
cron = "0 0 9 * * *"
task = "Something"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.daemon.unwrap().schedules[0].enabled);
    }

    #[test]
    fn daemon_schedule_disabled() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.schedules]]
name = "test"
cron = "0 0 9 * * *"
task = "Something"
enabled = false
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(!config.daemon.unwrap().schedules[0].enabled);
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

    #[cfg(feature = "telegram")]
    #[test]
    fn daemon_config_telegram_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.telegram]
token = "123:ABC"
dm_policy = "open"
allowed_users = [111, 222]
inactivity_timeout_seconds = 600
max_concurrent = 10
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let tg = daemon.telegram.unwrap();
        assert!(tg.enabled);
        assert_eq!(tg.token.as_deref(), Some("123:ABC"));
        assert_eq!(tg.dm_policy, crate::channel::telegram::DmPolicy::Open);
        assert_eq!(tg.allowed_users, vec![111, 222]);
        assert_eq!(tg.inactivity_timeout_seconds, 600);
        assert_eq!(tg.max_concurrent, 10);
        // Defaults
        assert_eq!(tg.session_expiry_seconds, 86400);
        assert_eq!(tg.interaction_timeout_seconds, 120);
        assert_eq!(tg.stream_debounce_ms, 500);
    }

    #[cfg(feature = "telegram")]
    #[test]
    fn daemon_config_telegram_defaults_to_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        assert!(daemon.telegram.is_none());
    }

    #[test]
    fn parse_sensor_config() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.sensors]
enabled = true

[daemon.sensors.routing]
triage_model = "cloud_light"
confidence_threshold = 0.85

[daemon.sensors.salience]
urgency_weight = 0.3
novelty_weight = 0.3
relevance_weight = 0.4
threshold = 0.3

[daemon.sensors.token_budget]
hourly_limit = 100000
queue_size = 200

[daemon.sensors.stories]
correlation_window_hours = 4
max_events_per_story = 50
stale_after_hours = 24

[[daemon.sensors.sources]]
type = "rss"
name = "tech_news"
feeds = ["https://hnrss.org/frontpage"]
interest_keywords = ["rust", "ai"]
poll_interval_seconds = 900

[[daemon.sensors.sources]]
type = "weather"
name = "local_weather"
api_key_env = "OPENWEATHERMAP_API_KEY"
locations = ["Paris,FR"]
alert_only = true
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let sensors = daemon.sensors.unwrap();
        assert!(sensors.enabled);
        assert_eq!(sensors.sources.len(), 2);

        let routing = sensors.routing.unwrap();
        assert_eq!(routing.triage_model, "cloud_light");

        let salience = sensors.salience.unwrap();
        assert!((salience.urgency_weight - 0.3).abs() < f64::EPSILON);

        let stories = sensors.stories.unwrap();
        assert_eq!(stories.correlation_window_hours, 4);
        assert_eq!(stories.max_events_per_story, 50);
    }

    #[test]
    fn sensor_config_absent_is_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        assert!(daemon.sensors.is_none());
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

    #[test]
    fn sensor_invalid_routing_model() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.sensors.routing]
triage_model = "invalid_tier"
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("triage_model"), "error: {msg}");
    }

    #[test]
    fn sensor_invalid_salience_weights() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.sensors.salience]
urgency_weight = 0.5
novelty_weight = 0.5
relevance_weight = 0.5
threshold = 0.3
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("sum"), "error: {msg}");
    }

    #[test]
    fn sensor_duplicate_source_names() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.sensors.sources]]
type = "rss"
name = "feed1"
feeds = ["https://example.com/feed"]

[[daemon.sensors.sources]]
type = "rss"
name = "feed1"
feeds = ["https://example.com/feed2"]
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("duplicate"), "error: {msg}");
    }

    #[test]
    fn sensor_zero_token_budget() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.sensors.token_budget]
hourly_limit = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("hourly_limit"), "error: {msg}");
    }

    #[test]
    fn sensor_zero_correlation_window_hours_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.sensors.stories]
correlation_window_hours = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("correlation_window_hours must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn sensor_zero_stale_after_hours_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.sensors.stories]
stale_after_hours = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("stale_after_hours must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn sensor_zero_max_events_per_story_rejected() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.sensors.stories]
max_events_per_story = 0
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("max_events_per_story must be at least 1"),
            "error: {msg}"
        );
    }

    #[test]
    fn parse_jmap_email_sensor() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.sensors.sources]]
type = "jmap_email"
name = "work_email"
server = "https://jmap.example.com"
username = "user@example.com"
password_env = "HEARTBIT_JMAP_PASSWORD"
priority_senders = ["boss@company.com"]
poll_interval_seconds = 60
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let sensors = daemon.sensors.unwrap();
        assert_eq!(sensors.sources.len(), 1);
        match &sensors.sources[0] {
            SensorSourceConfig::JmapEmail {
                name,
                server,
                priority_senders,
                ..
            } => {
                assert_eq!(name, "work_email");
                assert_eq!(server, "https://jmap.example.com");
                assert_eq!(priority_senders, &["boss@company.com"]);
            }
            other => panic!("expected JmapEmail, got: {other:?}"),
        }
    }

    #[test]
    fn parse_jmap_email_with_blocked_senders() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.sensors.sources]]
type = "jmap_email"
name = "work_email"
server = "https://jmap.example.com"
username = "user@example.com"
password_env = "HEARTBIT_JMAP_PASSWORD"
priority_senders = ["boss@company.com"]
blocked_senders = ["spam@example.com", "noreply@marketing.com"]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let sensors = daemon.sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::JmapEmail {
                blocked_senders, ..
            } => {
                assert_eq!(
                    blocked_senders,
                    &["spam@example.com", "noreply@marketing.com"]
                );
            }
            other => panic!("expected JmapEmail, got: {other:?}"),
        }
    }

    #[test]
    fn parse_jmap_email_blocked_senders_defaults_empty() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.sensors.sources]]
type = "jmap_email"
name = "work_email"
server = "https://jmap.example.com"
username = "user@example.com"
password_env = "HEARTBIT_JMAP_PASSWORD"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let sensors = daemon.sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::JmapEmail {
                blocked_senders, ..
            } => {
                assert!(blocked_senders.is_empty());
            }
            other => panic!("expected JmapEmail, got: {other:?}"),
        }
    }

    #[test]
    fn parse_audio_with_known_contacts() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.sensors.sources]]
type = "audio"
name = "voice_notes"
watch_directory = "/tmp/audio"
known_contacts = ["Alice", "Bob"]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let sensors = daemon.sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::Audio { known_contacts, .. } => {
                assert_eq!(known_contacts, &["Alice", "Bob"]);
            }
            other => panic!("expected Audio, got: {other:?}"),
        }
    }

    #[test]
    fn parse_audio_known_contacts_defaults_empty() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.sensors.sources]]
type = "audio"
name = "voice_notes"
watch_directory = "/tmp/audio"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let sensors = daemon.sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::Audio { known_contacts, .. } => {
                assert!(known_contacts.is_empty());
            }
            other => panic!("expected Audio, got: {other:?}"),
        }
    }

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
        assert_eq!(daemon.kafka.dead_letter_topic, "my.custom.dead-letter");
    }

    #[test]
    fn ws_config_defaults() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.ws]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let ws = config.daemon.unwrap().ws.unwrap();
        assert!(ws.enabled);
        assert_eq!(ws.interaction_timeout_seconds, 120);
        assert_eq!(ws.max_connections, 100);
    }

    #[test]
    fn ws_config_custom_values() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.ws]
enabled = false
interaction_timeout_seconds = 60
max_connections = 50
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let ws = config.daemon.unwrap().ws.unwrap();
        assert!(!ws.enabled);
        assert_eq!(ws.interaction_timeout_seconds, 60);
        assert_eq!(ws.max_connections, 50);
    }

    #[test]
    fn ws_config_absent_is_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.daemon.unwrap().ws.is_none());
    }

    #[test]
    fn ws_config_database_url_default_none() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.ws]
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let ws = config.daemon.unwrap().ws.unwrap();
        assert!(ws.database_url.is_none());
    }

    #[test]
    fn ws_config_database_url_present() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-3-5-sonnet"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.ws]
database_url = "postgresql://localhost/heartbit_sessions"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let ws = config.daemon.unwrap().ws.unwrap();
        assert_eq!(
            ws.database_url.as_deref(),
            Some("postgresql://localhost/heartbit_sessions")
        );
    }

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
}
