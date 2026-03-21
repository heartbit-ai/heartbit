use serde::{Deserialize, Serialize};

use crate::agent::routing::RoutingMode;

use super::guardrails::GuardrailsConfig;

/// Dispatch mode for orchestrator delegation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchMode {
    /// All delegated tasks run in parallel via JoinSet (default).
    Parallel,
    /// One task at a time. Schema constrains `maxItems: 1` on delegate_task.
    Sequential,
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

/// Per-agent provider override. When set on an agent, overrides the
/// orchestrator's default provider for that agent only.
#[derive(Debug, Clone, Deserialize)]
pub struct AgentProviderConfig {
    pub name: String,
    pub model: String,
    /// Custom API endpoint URL (overrides the default for the provider).
    /// Useful for self-hosted models, Azure, or proxies.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Direct API key (alternative to environment variable).
    /// Prefer env vars in production; this is for testing/local dev.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Enable Anthropic prompt caching for this agent.
    #[serde(default)]
    pub prompt_caching: bool,
    /// Per-agent model cascading override.
    pub cascade: Option<super::provider::CascadeConfig>,
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
    /// Tool profile for pre-filtering tool definitions. Valid values:
    /// "conversational", "standard", "full". Defaults to no filtering.
    pub tool_profile: Option<String>,
    /// Maximum consecutive identical tool-call turns before doom loop detection
    /// triggers. When reached, tool calls get error results instead of executing.
    pub max_identical_tool_calls: Option<u32>,
    /// Maximum consecutive fuzzy-identical tool-call turns before doom loop detection.
    /// Fuzzy matching compares sorted tool names (ignoring inputs).
    pub max_fuzzy_identical_tool_calls: Option<u32>,
    /// Dispatch mode for orchestrator delegation. When `Sequential`, the
    /// delegate_task schema constrains `maxItems: 1` so the LLM dispatches
    /// one agent at a time. Defaults to `Parallel` when absent.
    pub dispatch_mode: Option<DispatchMode>,
    /// Task routing strategy: `auto` (default), `always_orchestrate`, `single_agent`.
    /// `auto` uses heuristic scoring + capability matching to route simple tasks
    /// to a single agent and complex tasks to the orchestrator.
    #[serde(default)]
    pub routing: RoutingMode,
    /// Escalate from single-agent to orchestrator on failure. Default: true.
    /// When a single-agent run fails with MaxTurnsExceeded, doom loop, or
    /// excessive compaction, the task is re-run through the orchestrator.
    #[serde(default = "super::default_true")]
    pub escalation: bool,
    /// Append the multi-agent collaboration prompt to sub-agent system prompts.
    /// Teaches sub-agents blackboard protocol, dedup, cross-verification, and
    /// structured execution. Default: true.
    #[serde(default)]
    pub multi_agent_prompt: Option<bool>,
    /// Dynamic agent spawning configuration. When present, enables the `spawn_agent`
    /// tool on the orchestrator, allowing the LLM to create specialist agents at runtime.
    pub spawn: Option<SpawnConfig>,
}

/// Configuration for dynamic agent spawning via `spawn_agent`.
///
/// Controls security boundaries: which tools spawned agents may use,
/// how many can be created, and their token budgets.
#[derive(Debug, Clone, Deserialize)]
pub struct SpawnConfig {
    /// Maximum number of agents that can be spawned per orchestrator run.
    #[serde(default = "default_max_spawned_agents")]
    pub max_spawned_agents: u32,
    /// Allowlist of tool names that spawned agents may use.
    /// Only builtin tools from this list are available; unknown names
    /// are rejected at build time.
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    /// Maximum turns per spawned agent.
    #[serde(default = "default_spawn_max_turns")]
    pub max_turns: usize,
    /// Maximum tokens per LLM call for spawned agents.
    #[serde(default = "default_spawn_max_tokens")]
    pub max_tokens: u32,
    /// Cumulative token budget across ALL spawned agents in a single run.
    #[serde(default = "default_max_total_tokens")]
    pub max_total_tokens: u64,
}

fn default_max_spawned_agents() -> u32 {
    3
}

fn default_spawn_max_turns() -> usize {
    15
}

fn default_spawn_max_tokens() -> u32 {
    4096
}

fn default_max_total_tokens() -> u64 {
    50_000
}

pub(super) fn default_max_turns() -> usize {
    10
}

pub(super) fn default_max_tokens() -> u32 {
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
            tool_profile: None,
            max_identical_tool_calls: None,
            max_fuzzy_identical_tool_calls: None,
            dispatch_mode: None,
            routing: RoutingMode::default(),
            escalation: true,
            multi_agent_prompt: None,
            spawn: None,
        }
    }
}

/// An MCP server entry: a bare URL string, a full HTTP config with auth, or a
/// stdio command to spawn as a child process.
///
/// Supports backward-compatible TOML: bare strings (`"http://..."`) deserialize
/// as `Simple`, inline tables with `url` (`{ url = "...", auth_header = "..." }`)
/// as `Full`, and inline tables with `command` (`{ command = "npx", args = [...] }`)
/// as `Stdio`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum McpServerEntry {
    /// Bare URL string (backward-compatible).
    Simple(String),
    /// Full HTTP entry with optional auth header.
    Full {
        url: String,
        #[serde(default)]
        auth_header: Option<String>,
        /// RFC 8707 resource indicator — audience for exchanged tokens.
        /// Defaults to the `url` value when absent.
        #[serde(default)]
        resource: Option<String>,
        /// OAuth scopes required by this MCP server (e.g., `["gmail.readonly"]`).
        #[serde(default)]
        scopes: Option<Vec<String>>,
    },
    /// Stdio transport — spawn a child process communicating via stdin/stdout.
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: std::collections::HashMap<String, String>,
    },
}

impl McpServerEntry {
    /// Get the server URL (empty string for stdio entries).
    pub fn url(&self) -> &str {
        match self {
            McpServerEntry::Simple(url) => url,
            McpServerEntry::Full { url, .. } => url,
            McpServerEntry::Stdio { .. } => "",
        }
    }

    /// Get the optional auth header value.
    pub fn auth_header(&self) -> Option<&str> {
        match self {
            McpServerEntry::Simple(_) => None,
            McpServerEntry::Full { auth_header, .. } => auth_header.as_deref(),
            McpServerEntry::Stdio { .. } => None,
        }
    }

    /// Whether this entry uses stdio transport.
    pub fn is_stdio(&self) -> bool {
        matches!(self, McpServerEntry::Stdio { .. })
    }

    /// Get the RFC 8707 resource indicator (audience for token exchange).
    /// Returns the explicit `resource` if set, otherwise falls back to the URL.
    pub fn resource(&self) -> Option<&str> {
        match self {
            McpServerEntry::Simple(url) => Some(url.as_str()),
            McpServerEntry::Full { resource, url, .. } => {
                Some(resource.as_deref().unwrap_or(url.as_str()))
            }
            McpServerEntry::Stdio { .. } => None,
        }
    }

    /// Get the OAuth scopes configured for this MCP server.
    pub fn scopes(&self) -> Option<&[String]> {
        match self {
            McpServerEntry::Full { scopes, .. } => scopes.as_deref(),
            _ => None,
        }
    }

    /// Human-readable description for logging.
    pub fn display_name(&self) -> String {
        match self {
            McpServerEntry::Simple(url) => url.clone(),
            McpServerEntry::Full { url, .. } => url.clone(),
            McpServerEntry::Stdio { command, args, .. } => {
                if args.is_empty() {
                    command.clone()
                } else {
                    format!("{} {}", command, args.join(" "))
                }
            }
        }
    }
}

/// How MCP resources are surfaced to agents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum McpResourceMode {
    /// Resources become callable tools (agent decides when to read).
    #[default]
    Tools,
    /// Pre-fetch resource content and inject into system prompt.
    Context,
    /// Skip resource discovery entirely.
    None,
}

/// A sub-agent defined in the configuration file.
#[derive(Debug, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub system_prompt: String,
    /// Agent template to use as a base. The template provides default values
    /// for system_prompt, max_tokens, max_turns, and other settings.
    /// User-specified values override template defaults.
    #[serde(default)]
    pub template: Option<String>,
    /// Skills to auto-inject into the system prompt at config resolution time.
    /// Each skill name maps to a bundled or filesystem SKILL.md file.
    #[serde(default)]
    pub skills: Vec<String>,
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
    /// Tool profile for pre-filtering tool definitions. Valid values:
    /// "conversational" (memory + question only), "standard" (builtins only),
    /// "full" (all tools). When absent, no pre-filtering is applied.
    pub tool_profile: Option<String>,
    /// Maximum consecutive identical tool-call turns before doom loop detection.
    /// Overrides the orchestrator default.
    pub max_identical_tool_calls: Option<u32>,
    /// Maximum consecutive fuzzy-identical tool-call turns before doom loop detection.
    /// Fuzzy matching compares sorted tool names (ignoring inputs). Overrides orchestrator default.
    pub max_fuzzy_identical_tool_calls: Option<u32>,
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
    /// Hard limit on cumulative tokens (input + output) across all turns.
    /// When exceeded, the agent returns an error with partial usage data.
    pub max_total_tokens: Option<u64>,
    /// Per-agent guardrails override. When set, overrides the top-level
    /// `[guardrails]` section for this agent.
    pub guardrails: Option<GuardrailsConfig>,
    /// LRU response cache capacity (number of entries). When set, identical
    /// LLM requests (same system prompt, messages, tool names) return cached
    /// responses without calling the LLM. Only non-streaming calls are cached.
    #[serde(default)]
    pub response_cache_size: Option<usize>,
    /// How MCP resources are surfaced to the agent.
    /// `"tools"` (default) — resources become callable tools.
    /// `"context"` — pre-fetch and inject into system prompt.
    /// `"none"` — skip resource discovery.
    #[serde(default)]
    pub mcp_resources: McpResourceMode,
    /// Enable dangerous tools (bash) for this agent. Default: false in daemon mode.
    #[serde(default)]
    pub dangerous_tools: bool,
    /// Audit mode: "full" (default) or "metadata_only".
    /// MetadataOnly strips user content from audit records.
    #[serde(default)]
    pub audit_mode: Option<String>,
}

/// TOML representation of session pruning configuration.
#[derive(Debug, Clone, Deserialize)]
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

impl AgentConfig {
    /// Clone all fields of this config into a new `AgentConfig`.
    ///
    /// `AgentConfig` intentionally does not derive `Clone` (to keep the derive
    /// list short and avoid accidental copies in hot paths). Use this method
    /// when an explicit copy is needed (e.g., template resolution).
    pub fn clone_config(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            system_prompt: self.system_prompt.clone(),
            template: self.template.clone(),
            skills: self.skills.clone(),
            mcp_servers: self.mcp_servers.clone(),
            a2a_agents: self.a2a_agents.clone(),
            context_strategy: self.context_strategy.clone(),
            summarize_threshold: self.summarize_threshold,
            tool_timeout_seconds: self.tool_timeout_seconds,
            max_tool_output_bytes: self.max_tool_output_bytes,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            response_schema: self.response_schema.clone(),
            run_timeout_seconds: self.run_timeout_seconds,
            provider: self.provider.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            enable_reflection: self.enable_reflection,
            tool_output_compression_threshold: self.tool_output_compression_threshold,
            max_tools_per_turn: self.max_tools_per_turn,
            tool_profile: self.tool_profile.clone(),
            max_identical_tool_calls: self.max_identical_tool_calls,
            max_fuzzy_identical_tool_calls: self.max_fuzzy_identical_tool_calls,
            session_prune: self.session_prune.clone(),
            recursive_summarization: self.recursive_summarization,
            reflection_threshold: self.reflection_threshold,
            consolidate_on_exit: self.consolidate_on_exit,
            max_total_tokens: self.max_total_tokens,
            guardrails: self.guardrails.clone(),
            response_cache_size: self.response_cache_size,
            mcp_resources: self.mcp_resources,
            dangerous_tools: self.dangerous_tools,
            audit_mode: self.audit_mode.clone(),
        }
    }
}

impl AgentProviderConfig {
    /// Clone via Option::as_ref → clone pattern for non-Clone containers.
    pub fn take_ref(opt: &Option<Self>) -> Option<Self> {
        opt.clone()
    }
}

impl SessionPruneConfigToml {
    /// Clone via Option::as_ref → clone pattern for non-Clone containers.
    pub fn take_ref(opt: &Option<Self>) -> Option<Self> {
        opt.clone()
    }
}
