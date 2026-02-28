use serde::{Deserialize, Serialize};

use crate::Error;
use crate::agent::permission::PermissionRule;
use crate::agent::routing::RoutingMode;
use crate::agent::tool_filter::ToolProfile;
use crate::llm::types::ReasoningEffort;

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

/// Trust classification for the sender of an external message.
///
/// Resolved deterministically from config lists — never LLM-based.
/// Ordered from least to most trusted; `PartialOrd`/`Ord` follow declaration order.
///
/// Defined in config so it's always available for TOML/JSON deserialization
/// even when the `sensor` feature is disabled. Re-exported from `sensor::triage::context`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    /// Explicitly blocked sender. Zero action permitted.
    Quarantined,
    /// No prior relationship. Read-only access.
    #[default]
    Unknown,
    /// Recognized but not privileged.
    Known,
    /// In the priority senders list. May trigger replies (with approval).
    Verified,
    /// The system owner. Full access.
    Owner,
}

impl TrustLevel {
    /// Resolve trust level from sender email against config lists.
    ///
    /// Priority: Owner > Blocked(Quarantined) > Priority(Verified) > Unknown.
    /// Matching is case-insensitive.
    pub fn resolve(
        sender: Option<&str>,
        owner_emails: &[String],
        priority_senders: &[String],
        blocked_senders: &[String],
    ) -> Self {
        let sender = match sender {
            Some(s) if !s.trim().is_empty() => s.trim(),
            _ => return TrustLevel::Unknown,
        };
        let lower = sender.to_lowercase();

        if owner_emails
            .iter()
            .any(|e| e.trim().to_lowercase() == lower)
        {
            return TrustLevel::Owner;
        }
        if blocked_senders
            .iter()
            .any(|e| e.trim().to_lowercase() == lower)
        {
            return TrustLevel::Quarantined;
        }
        if priority_senders
            .iter()
            .any(|e| e.trim().to_lowercase() == lower)
        {
            return TrustLevel::Verified;
        }
        TrustLevel::Unknown
    }
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustLevel::Quarantined => write!(f, "quarantined"),
            TrustLevel::Unknown => write!(f, "unknown"),
            TrustLevel::Known => write!(f, "known"),
            TrustLevel::Verified => write!(f, "verified"),
            TrustLevel::Owner => write!(f, "owner"),
        }
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
    /// Optional workspace configuration for agent home directories.
    pub workspace: Option<WorkspaceConfig>,
    /// Guardrails configuration for injection detection, PII, and tool policies.
    #[serde(default)]
    pub guardrails: Option<GuardrailsConfig>,
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
    /// Model cascading configuration. When enabled, tries cheaper models first
    /// and escalates to the main model only when the confidence gate rejects.
    pub cascade: Option<CascadeConfig>,
}

/// Model cascading configuration for cost-efficient LLM selection.
///
/// When enabled, the provider tries cheaper model tiers first and only
/// escalates to the main (most expensive) model when the confidence gate
/// rejects the cheaper response or the tier errors.
#[derive(Debug, Deserialize)]
pub struct CascadeConfig {
    /// Enable model cascading. Default: false.
    #[serde(default)]
    pub enabled: bool,
    /// Model tiers from cheapest to most expensive.
    /// The main `[provider].model` is always the implicit final tier.
    #[serde(default)]
    pub tiers: Vec<CascadeTierConfig>,
    /// Confidence gate configuration. Default: heuristic with sensible defaults.
    #[serde(default)]
    pub gate: CascadeGateConfig,
}

/// A single tier in the model cascade.
#[derive(Debug, Deserialize)]
pub struct CascadeTierConfig {
    pub model: String,
}

/// Confidence gate configuration for model cascading.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CascadeGateConfig {
    /// Heuristic gate: zero-cost checks on response length, refusal patterns, etc.
    Heuristic {
        /// Minimum output tokens for acceptance (default: 5).
        #[serde(default = "default_min_output_tokens")]
        min_output_tokens: u32,
        /// Accept responses that include tool calls (default: true).
        #[serde(default = "default_true")]
        accept_tool_calls: bool,
        /// Escalate on MaxTokens stop reason (default: true).
        #[serde(default = "default_true")]
        escalate_on_max_tokens: bool,
    },
}

impl Default for CascadeGateConfig {
    fn default() -> Self {
        Self::Heuristic {
            min_output_tokens: default_min_output_tokens(),
            accept_tool_calls: true,
            escalate_on_max_tokens: true,
        }
    }
}

fn default_min_output_tokens() -> u32 {
    5
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
    /// Tool profile for pre-filtering tool definitions. Valid values:
    /// "conversational", "standard", "full". Defaults to no filtering.
    pub tool_profile: Option<String>,
    /// Maximum consecutive identical tool-call turns before doom loop detection
    /// triggers. When reached, tool calls get error results instead of executing.
    pub max_identical_tool_calls: Option<u32>,
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
    #[serde(default = "default_true")]
    pub escalation: bool,
}

fn default_true() -> bool {
    true
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
            tool_profile: None,
            max_identical_tool_calls: None,
            dispatch_mode: None,
            routing: RoutingMode::default(),
            escalation: true,
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

/// Per-agent provider override. When set on an agent, overrides the
/// orchestrator's default provider for that agent only.
#[derive(Debug, Deserialize)]
pub struct AgentProviderConfig {
    pub name: String,
    pub model: String,
    /// Enable Anthropic prompt caching for this agent.
    #[serde(default)]
    pub prompt_caching: bool,
    /// Per-agent model cascading override.
    pub cascade: Option<CascadeConfig>,
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
    /// Tool profile for pre-filtering tool definitions. Valid values:
    /// "conversational" (memory + question only), "standard" (builtins only),
    /// "full" (all tools). When absent, no pre-filtering is applied.
    pub tool_profile: Option<String>,
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
    /// Hard limit on cumulative tokens (input + output) across all turns.
    /// When exceeded, the agent returns an error with partial usage data.
    pub max_total_tokens: Option<u64>,
    /// Per-agent guardrails override. When set, overrides the top-level
    /// `[guardrails]` section for this agent.
    pub guardrails: Option<GuardrailsConfig>,
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

/// Workspace configuration for agent home directories.
///
/// When configured, each agent gets a persistent home directory where it
/// can freely create and organize files. File tools resolve relative paths
/// against this directory.
#[derive(Debug, Deserialize)]
pub struct WorkspaceConfig {
    /// Workspace root directory. All agents share this single directory.
    /// Defaults to `~/.heartbit/workspaces` if not specified.
    #[serde(default = "default_workspace_root")]
    pub root: String,
}

fn default_workspace_root() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.heartbit/workspaces")
}

// ---------------------------------------------------------------------------
// Guardrails configuration
// ---------------------------------------------------------------------------

/// Top-level guardrails configuration.
///
/// Enables declarative guardrail setup via TOML. Each sub-section creates
/// the corresponding guardrail and adds it to the agent's guardrail chain.
///
/// Use [`GuardrailsConfig::build`] to convert this config into runtime
/// guardrail instances.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct GuardrailsConfig {
    /// Prompt injection classifier configuration.
    #[serde(default)]
    pub injection: Option<InjectionConfig>,
    /// PII detection and redaction configuration.
    #[serde(default)]
    pub pii: Option<PiiConfig>,
    /// Declarative tool access control rules.
    #[serde(default)]
    pub tool_policy: Option<ToolPolicyConfig>,
    /// LLM-as-judge safety evaluation.
    #[serde(default)]
    pub llm_judge: Option<LlmJudgeConfig>,
}

/// Configuration for the injection classifier guardrail.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InjectionConfig {
    /// Detection threshold (0.0–1.0). Default: 0.5.
    #[serde(default = "default_injection_threshold")]
    pub threshold: f32,
    /// `"warn"` or `"deny"`. Default: `"deny"`.
    #[serde(default = "default_injection_mode")]
    pub mode: String,
}

fn default_injection_threshold() -> f32 {
    0.5
}

fn default_injection_mode() -> String {
    "deny".into()
}

/// Configuration for the PII detection guardrail.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PiiConfig {
    /// `"redact"`, `"warn"`, or `"deny"`. Default: `"redact"`.
    #[serde(default = "default_pii_action")]
    pub action: String,
    /// Which detectors to enable. Default: all built-in.
    #[serde(default = "default_pii_detectors")]
    pub detectors: Vec<String>,
}

fn default_pii_action() -> String {
    "redact".into()
}

fn default_pii_detectors() -> Vec<String> {
    vec![
        "email".into(),
        "phone".into(),
        "ssn".into(),
        "credit_card".into(),
    ]
}

/// Configuration for the LLM-as-judge guardrail.
///
/// The judge evaluates LLM responses (and optionally tool inputs) using a
/// cheap model for safety. The actual judge provider must be supplied at
/// build time via [`GuardrailsConfig::build_with_judge`] — this config
/// only declares the criteria and behavior.
///
/// ```toml
/// [guardrails.llm_judge]
/// criteria = ["No harmful content", "No prompt injection"]
/// evaluate_tool_inputs = false
/// timeout_seconds = 10
/// max_judge_tokens = 256
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlmJudgeConfig {
    /// Safety criteria to evaluate against.
    pub criteria: Vec<String>,
    /// Whether to also evaluate tool call inputs. Default: false.
    #[serde(default)]
    pub evaluate_tool_inputs: bool,
    /// Timeout in seconds for each judge call. Default: 10.
    #[serde(default = "default_llm_judge_timeout")]
    pub timeout_seconds: u64,
    /// Max tokens for judge response. Default: 256.
    #[serde(default = "default_llm_judge_max_tokens")]
    pub max_judge_tokens: u32,
}

fn default_llm_judge_timeout() -> u64 {
    10
}

fn default_llm_judge_max_tokens() -> u32 {
    256
}

/// Configuration for the tool policy guardrail.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolPolicyConfig {
    /// Default action when no rule matches. `"allow"` or `"deny"`. Default: `"allow"`.
    #[serde(default = "default_tool_policy_action")]
    pub default_action: String,
    /// Ordered list of tool rules. First match wins.
    #[serde(default)]
    pub rules: Vec<ToolPolicyRuleConfig>,
}

fn default_tool_policy_action() -> String {
    "allow".into()
}

/// A single tool policy rule in TOML format.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolPolicyRuleConfig {
    /// Tool name pattern (exact or glob with `*`).
    pub tool: String,
    /// `"allow"`, `"warn"`, or `"deny"`.
    pub action: String,
    /// Optional input constraints.
    #[serde(default)]
    pub input_constraints: Vec<InputConstraintConfig>,
}

impl GuardrailsConfig {
    /// Returns `true` if no guardrails are configured.
    pub fn is_empty(&self) -> bool {
        self.injection.is_none()
            && self.pii.is_none()
            && self.tool_policy.is_none()
            && self.llm_judge.is_none()
    }

    /// Build runtime guardrail instances from this configuration.
    ///
    /// Returns a `Vec<Arc<dyn Guardrail>>` ready to be passed to
    /// `AgentRunnerBuilder::guardrails()` or `OrchestratorBuilder::guardrails()`.
    ///
    /// Order: injection → PII → tool policy → LLM judge. Each section that
    /// is `Some` creates the corresponding guardrail instance.
    ///
    /// **Note:** If `[guardrails.llm_judge]` is configured, you must use
    /// [`build_with_judge`](Self::build_with_judge) instead, passing the
    /// judge provider. This method ignores the `llm_judge` section.
    pub fn build(
        &self,
    ) -> Result<Vec<std::sync::Arc<dyn crate::agent::guardrail::Guardrail>>, Error> {
        self.build_with_judge(None)
    }

    /// Build runtime guardrail instances, optionally including the LLM judge.
    ///
    /// Pass `Some(provider)` to enable the LLM-as-judge guardrail when
    /// `[guardrails.llm_judge]` is configured. The provider should be a
    /// cheap model (e.g., Haiku, Gemini Flash) separate from the main agent.
    pub fn build_with_judge(
        &self,
        judge_provider: Option<std::sync::Arc<crate::llm::BoxedProvider>>,
    ) -> Result<Vec<std::sync::Arc<dyn crate::agent::guardrail::Guardrail>>, Error> {
        use std::sync::Arc;

        use crate::agent::guardrail::Guardrail;
        use crate::agent::guardrails::injection::{GuardrailMode, InjectionClassifierGuardrail};
        use crate::agent::guardrails::pii::{PiiAction, PiiDetector, PiiGuardrail};
        use crate::agent::guardrails::tool_policy::{
            InputConstraint, ToolPolicyGuardrail, ToolRule,
        };

        let mut guardrails: Vec<Arc<dyn Guardrail>> = Vec::new();

        // 1. Injection classifier
        if let Some(cfg) = &self.injection {
            let mode = match cfg.mode.as_str() {
                "warn" => GuardrailMode::Warn,
                "deny" => GuardrailMode::Deny,
                other => {
                    return Err(Error::Config(format!(
                        "invalid injection mode: `{other}` (expected \"warn\" or \"deny\")"
                    )));
                }
            };
            guardrails.push(Arc::new(InjectionClassifierGuardrail::new(
                cfg.threshold,
                mode,
            )));
        }

        // 2. PII detection
        if let Some(cfg) = &self.pii {
            let action = match cfg.action.as_str() {
                "redact" => PiiAction::Redact,
                "warn" => PiiAction::Warn,
                "deny" => PiiAction::Deny,
                other => {
                    return Err(Error::Config(format!(
                        "invalid PII action: `{other}` (expected \"redact\", \"warn\", or \"deny\")"
                    )));
                }
            };
            let detectors: Vec<PiiDetector> = cfg
                .detectors
                .iter()
                .map(|name| match name.as_str() {
                    "email" => Ok(PiiDetector::Email),
                    "phone" => Ok(PiiDetector::Phone),
                    "ssn" => Ok(PiiDetector::Ssn),
                    "credit_card" => Ok(PiiDetector::CreditCard),
                    other => Err(Error::Config(format!(
                        "unknown PII detector: `{other}` (expected email, phone, ssn, or credit_card)"
                    ))),
                })
                .collect::<Result<_, _>>()?;
            guardrails.push(Arc::new(PiiGuardrail::new(detectors, action)));
        }

        // 3. Tool policy
        if let Some(cfg) = &self.tool_policy {
            let default_action = parse_guard_action(&cfg.default_action)?;
            let mut rules = Vec::with_capacity(cfg.rules.len());
            for rule_cfg in &cfg.rules {
                let action = parse_guard_action(&rule_cfg.action)?;
                let mut constraints = Vec::new();
                for ic in &rule_cfg.input_constraints {
                    if let Some(pattern_str) = &ic.deny_pattern {
                        let pattern = regex::Regex::new(pattern_str).map_err(|e| {
                            Error::Config(format!("invalid deny_pattern `{pattern_str}`: {e}"))
                        })?;
                        constraints.push(InputConstraint::FieldDenied {
                            path: ic.path.clone(),
                            pattern,
                        });
                    }
                    if let Some(max) = ic.max_length {
                        constraints.push(InputConstraint::MaxFieldLength {
                            path: ic.path.clone(),
                            max_bytes: max,
                        });
                    }
                }
                rules.push(ToolRule {
                    tool_pattern: rule_cfg.tool.clone(),
                    action,
                    input_constraints: constraints,
                });
            }
            guardrails.push(Arc::new(ToolPolicyGuardrail::new(rules, default_action)));
        }

        // 4. LLM-as-judge
        if let Some(cfg) = &self.llm_judge
            && let Some(provider) = judge_provider
        {
            let mut builder =
                crate::agent::guardrails::llm_judge::LlmJudgeGuardrail::builder(provider)
                    .criteria(cfg.criteria.clone())
                    .timeout(std::time::Duration::from_secs(cfg.timeout_seconds))
                    .max_judge_tokens(cfg.max_judge_tokens);
            if cfg.evaluate_tool_inputs {
                builder = builder.evaluate_tool_inputs(true);
            }
            let judge = builder
                .build()
                .map_err(|e| Error::Config(format!("llm_judge guardrail build failed: {e}")))?;
            guardrails.push(Arc::new(judge));
        }
        // If no judge_provider supplied, silently skip — caller should use
        // `build_with_judge(Some(provider))` when `[guardrails.llm_judge]` is set.

        Ok(guardrails)
    }
}

/// Parse a guard action string from config (`"allow"`, `"warn"`, `"deny"`).
fn parse_guard_action(s: &str) -> Result<crate::agent::guardrail::GuardAction, Error> {
    match s {
        "allow" => Ok(crate::agent::guardrail::GuardAction::Allow),
        "warn" => Ok(crate::agent::guardrail::GuardAction::warn(String::new())),
        "deny" => Ok(crate::agent::guardrail::GuardAction::deny(String::new())),
        other => Err(Error::Config(format!(
            "invalid action: `{other}` (expected \"allow\", \"warn\", or \"deny\")"
        ))),
    }
}

/// Input constraint configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputConstraintConfig {
    /// JSON path to the field (e.g., `"command"`, `"path"`).
    pub path: String,
    /// Regex pattern — if the field matches, the constraint is violated.
    #[serde(default)]
    pub deny_pattern: Option<String>,
    /// Maximum byte length for the field's string value.
    #[serde(default)]
    pub max_length: Option<usize>,
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
    /// Provider name: "openai", "local", or "none" (default).
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
    /// Model cache directory for local embedding provider (optional).
    pub cache_dir: Option<String>,
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
    /// Heartbit pulse configuration for autonomous periodic awareness.
    pub heartbit_pulse: Option<HeartbitPulseConfig>,
    /// HTTP API authentication configuration.
    pub auth: Option<AuthConfig>,
    /// Email addresses of the system owner (for trust resolution).
    #[serde(default)]
    pub owner_emails: Vec<String>,
}

/// HTTP API authentication configuration for the daemon.
#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    /// Bearer tokens that grant API access. Multiple tokens support key rotation.
    #[serde(default)]
    pub bearer_tokens: Vec<String>,
    /// JWKS endpoint URL for JWT signature verification
    /// (e.g. `"https://idp.example.com/.well-known/jwks.json"`).
    pub jwks_url: Option<String>,
    /// Expected JWT issuer (`iss` claim). Validated when present.
    pub issuer: Option<String>,
    /// Expected JWT audience (`aud` claim). Validated when present.
    pub audience: Option<String>,
    /// JWT claim to extract user ID from. Defaults to `"sub"`.
    pub user_id_claim: Option<String>,
    /// JWT claim to extract tenant ID from. Defaults to `"tid"`.
    pub tenant_id_claim: Option<String>,
    /// JWT claim to extract roles from. Defaults to `"roles"`.
    pub roles_claim: Option<String>,
    /// RFC 8693 Token Exchange configuration for per-user MCP auth delegation.
    /// When configured, the daemon exchanges user JWTs for MCP-scoped delegated tokens.
    pub token_exchange: Option<TokenExchangeConfig>,
}

/// RFC 8693 Token Exchange configuration for per-user MCP auth delegation.
///
/// When configured, each task submitted with a JWT gets a user-scoped delegated
/// token injected into MCP requests. The daemon acts as the agent (actor) and
/// exchanges the user's subject token for a scoped access token.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenExchangeConfig {
    /// Token exchange endpoint URL (e.g. `"https://idp.example.com/oauth/token"`).
    pub exchange_url: String,
    /// OAuth client ID for the daemon/agent.
    pub client_id: String,
    /// OAuth client secret for the daemon/agent.
    pub client_secret: String,
    /// The agent's own credential token (used as `actor_token` in RFC 8693).
    pub agent_token: String,
    /// OAuth scopes to request for the delegated token. Defaults to empty.
    #[serde(default)]
    pub scopes: Vec<String>,
}

/// Heartbit pulse configuration for autonomous periodic awareness.
///
/// When enabled, the daemon periodically reviews its persistent todo list
/// and decides what to work on next — a cognitive pulse loop.
#[derive(Debug, Clone, Deserialize)]
pub struct HeartbitPulseConfig {
    /// Enable the heartbit pulse. Defaults to `false`.
    #[serde(default)]
    pub enabled: bool,
    /// Interval in seconds between heartbit pulse ticks. Defaults to 1800 (30 min).
    #[serde(default = "default_pulse_interval")]
    pub interval_seconds: u64,
    /// Active hours window. When set, the pulse only fires within this window.
    pub active_hours: Option<ActiveHoursConfig>,
    /// Custom prompt override for the heartbit pulse. When absent, the
    /// default built-in prompt is used.
    pub prompt: Option<String>,
    /// Number of consecutive HEARTBIT_OK responses before doubling the
    /// interval (idle backoff). Defaults to 6 (3h at 30min interval).
    #[serde(default = "default_idle_backoff_threshold")]
    pub idle_backoff_threshold: u32,
}

fn default_pulse_interval() -> u64 {
    1800
}

fn default_idle_backoff_threshold() -> u32 {
    6
}

/// Active hours window for the heartbit pulse.
#[derive(Debug, Clone, Deserialize)]
pub struct ActiveHoursConfig {
    /// Start time in "HH:MM" format (24-hour).
    pub start: String,
    /// End time in "HH:MM" format (24-hour).
    pub end: String,
}

impl ActiveHoursConfig {
    /// Parse the start hour and minute. Returns `(hour, minute)`.
    pub fn parse_start(&self) -> Result<(u32, u32), Error> {
        parse_hhmm(&self.start)
    }

    /// Parse the end hour and minute. Returns `(hour, minute)`.
    pub fn parse_end(&self) -> Result<(u32, u32), Error> {
        parse_hhmm(&self.end)
    }
}

fn parse_hhmm(s: &str) -> Result<(u32, u32), Error> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(Error::Config(format!(
            "invalid time format '{}': expected HH:MM",
            s
        )));
    }
    let hour: u32 = parts[0]
        .parse()
        .map_err(|_| Error::Config(format!("invalid hour in '{s}'")))?;
    let minute: u32 = parts[1]
        .parse()
        .map_err(|_| Error::Config(format!("invalid minute in '{s}'")))?;
    if hour > 23 || minute > 59 {
        return Err(Error::Config(format!(
            "time '{}' out of range (00:00-23:59)",
            s
        )));
    }
    Ok((hour, minute))
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
    /// Generic MCP sensor — polls a tool on any MCP server.
    Mcp {
        name: String,
        /// MCP server endpoint (string URL, `{url, auth_header}`, or `{command, args, env}`).
        server: Box<McpServerEntry>,
        /// MCP tool to call each poll cycle.
        tool_name: String,
        /// Arguments passed to the tool (default: `{}`).
        #[serde(default = "default_empty_object")]
        tool_args: serde_json::Value,
        /// Kafka topic to produce events to.
        kafka_topic: String,
        /// Sensory modality of produced events (default: `"text"`).
        #[serde(default = "default_mcp_modality")]
        modality: SensorModality,
        /// Poll interval in seconds (default: 60).
        #[serde(default = "default_mcp_poll_interval")]
        poll_interval_seconds: u64,
        /// JSON field path for item ID (default: `"id"`).
        #[serde(default = "default_id_field")]
        id_field: String,
        /// JSON field for event content (default: entire item as JSON).
        #[serde(default)]
        content_field: Option<String>,
        /// JSON field containing items array in tool result (default: root is array).
        #[serde(default)]
        items_field: Option<String>,
        /// Priority senders for email triage (only when `kafka_topic = "hb.sensor.email"`).
        #[serde(default)]
        priority_senders: Vec<String>,
        /// Blocked senders for email triage.
        #[serde(default)]
        blocked_senders: Vec<String>,
        /// Optional enrichment tool to call for each new item (e.g., `gmail_get_message`).
        /// When set, the sensor calls this tool with the item's ID to fetch detailed
        /// metadata (headers, body, labels) before producing to Kafka.
        #[serde(default)]
        enrich_tool: Option<String>,
        /// Parameter name for the item ID when calling the enrichment tool (default: `"id"`).
        #[serde(default)]
        enrich_id_param: Option<String>,
        /// Dedup TTL in seconds. Seen IDs older than this are evicted. Default: 7 days.
        #[serde(default = "default_dedup_ttl_seconds")]
        dedup_ttl_seconds: u64,
    },
}

fn default_dedup_ttl_seconds() -> u64 {
    7 * 24 * 3600 // 7 days
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
            | SensorSourceConfig::Webhook { name, .. }
            | SensorSourceConfig::Mcp { name, .. } => name,
        }
    }

    /// Get priority and blocked sender lists for trust resolution.
    ///
    /// Returns `(priority_senders, blocked_senders)`. Only email-type sources
    /// have these lists; other source types return empty slices.
    pub fn sender_lists(&self) -> (&[String], &[String]) {
        match self {
            SensorSourceConfig::JmapEmail {
                priority_senders,
                blocked_senders,
                ..
            }
            | SensorSourceConfig::Mcp {
                priority_senders,
                blocked_senders,
                ..
            } => (priority_senders, blocked_senders),
            _ => (&[], &[]),
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

fn default_mcp_poll_interval() -> u64 {
    60
}

fn default_mcp_modality() -> SensorModality {
    SensorModality::Text
}

fn default_id_field() -> String {
    "id".into()
}

fn default_empty_object() -> serde_json::Value {
    serde_json::json!({})
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
        // Validate top-level provider
        if self.provider.name.is_empty() {
            return Err(Error::Config("provider.name must not be empty".into()));
        }
        if self.provider.model.is_empty() {
            return Err(Error::Config("provider.model must not be empty".into()));
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
                // MCP-specific field validation.
                if let SensorSourceConfig::Mcp {
                    tool_name,
                    kafka_topic,
                    server,
                    ..
                } = source
                {
                    if tool_name.is_empty() {
                        return Err(Error::Config(format!(
                            "sensor '{name}': tool_name must not be empty"
                        )));
                    }
                    if kafka_topic.is_empty() {
                        return Err(Error::Config(format!(
                            "sensor '{name}': kafka_topic must not be empty"
                        )));
                    }
                    match &**server {
                        McpServerEntry::Stdio { command, .. } => {
                            if command.is_empty() {
                                return Err(Error::Config(format!(
                                    "sensor '{name}': server command must not be empty"
                                )));
                            }
                        }
                        _ => {
                            if server.url().is_empty() {
                                return Err(Error::Config(format!(
                                    "sensor '{name}': server URL must not be empty"
                                )));
                            }
                        }
                    }
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
            if daemon.kafka.consumer_group.is_empty() {
                return Err(Error::Config(
                    "daemon.kafka.consumer_group must not be empty".into(),
                ));
            }
            if daemon.kafka.commands_topic.is_empty() {
                return Err(Error::Config(
                    "daemon.kafka.commands_topic must not be empty".into(),
                ));
            }
            if daemon.kafka.events_topic.is_empty() {
                return Err(Error::Config(
                    "daemon.kafka.events_topic must not be empty".into(),
                ));
            }
            // Validate heartbit pulse config
            if let Some(ref pulse) = daemon.heartbit_pulse {
                if pulse.interval_seconds == 0 {
                    return Err(Error::Config(
                        "daemon.heartbit_pulse.interval_seconds must be at least 1".into(),
                    ));
                }
                if let Some(ref hours) = pulse.active_hours {
                    hours.parse_start().map_err(|_| {
                        Error::Config(format!(
                            "daemon.heartbit_pulse.active_hours.start '{}' is invalid; expected HH:MM",
                            hours.start
                        ))
                    })?;
                    hours.parse_end().map_err(|_| {
                        Error::Config(format!(
                            "daemon.heartbit_pulse.active_hours.end '{}' is invalid; expected HH:MM",
                            hours.end
                        ))
                    })?;
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
                    if te.agent_token.is_empty() {
                        return Err(Error::Config(
                            "daemon.auth.token_exchange.agent_token must not be empty".into(),
                        ));
                    }
                }
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
                if schedule.task.is_empty() {
                    return Err(Error::Config(format!(
                        "daemon.schedules[{i}] '{}': task must not be empty",
                        schedule.name
                    )));
                }
                #[cfg(feature = "daemon")]
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
    #[cfg(feature = "daemon")]
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

    #[test]
    fn heartbit_pulse_config_parses() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.heartbit_pulse]
enabled = true
interval_seconds = 900

[daemon.heartbit_pulse.active_hours]
start = "08:00"
end = "22:00"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let daemon = config.daemon.unwrap();
        let pulse = daemon.heartbit_pulse.unwrap();
        assert!(pulse.enabled);
        assert_eq!(pulse.interval_seconds, 900);
        assert_eq!(pulse.idle_backoff_threshold, 6); // default
        assert!(pulse.prompt.is_none());
        let hours = pulse.active_hours.unwrap();
        assert_eq!(hours.start, "08:00");
        assert_eq!(hours.end, "22:00");
    }

    #[test]
    fn heartbit_pulse_config_defaults() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.heartbit_pulse]
enabled = true
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        let pulse = config.daemon.unwrap().heartbit_pulse.unwrap();
        assert!(pulse.enabled);
        assert_eq!(pulse.interval_seconds, 1800); // default
        assert_eq!(pulse.idle_backoff_threshold, 6); // default
        assert!(pulse.active_hours.is_none());
    }

    #[test]
    fn heartbit_pulse_config_absent() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml).unwrap();
        assert!(config.daemon.unwrap().heartbit_pulse.is_none());
    }

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

    #[test]
    fn validate_rejects_pulse_zero_interval() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.heartbit_pulse]
enabled = true
interval_seconds = 0
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("daemon.heartbit_pulse.interval_seconds"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_rejects_pulse_invalid_active_hours() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[daemon.heartbit_pulse]
enabled = true
interval_seconds = 1800

[daemon.heartbit_pulse.active_hours]
start = "bad"
end = "22:00"
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string()
                .contains("daemon.heartbit_pulse.active_hours.start"),
            "got: {err}"
        );
    }

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
                .contains("daemon.auth.token_exchange.agent_token must not be empty"),
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

    // --- MCP sensor config tests ---

    #[test]
    fn sensor_source_mcp_serde() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "gmail_inbox"
server = "http://localhost:3000/mcp"
tool_name = "search_emails"
tool_args = { query = "is:unread", max_results = 50 }
kafka_topic = "hb.sensor.email"
modality = "text"
poll_interval_seconds = 60
id_field = "messageId"
content_field = "snippet"
priority_senders = ["boss@company.com"]
blocked_senders = ["spam@marketing.com"]

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let sensors = config.daemon.unwrap().sensors.unwrap();
        assert_eq!(sensors.sources.len(), 1);
        match &sensors.sources[0] {
            SensorSourceConfig::Mcp {
                name,
                server,
                tool_name,
                tool_args,
                kafka_topic,
                modality,
                poll_interval_seconds,
                id_field,
                content_field,
                items_field,
                priority_senders,
                blocked_senders,
                enrich_tool,
                enrich_id_param,
                dedup_ttl_seconds,
            } => {
                assert_eq!(name, "gmail_inbox");
                assert_eq!(server.url(), "http://localhost:3000/mcp");
                assert_eq!(tool_name, "search_emails");
                assert_eq!(tool_args["query"], "is:unread");
                assert_eq!(tool_args["max_results"], 50);
                assert_eq!(kafka_topic, "hb.sensor.email");
                assert_eq!(*modality, SensorModality::Text);
                assert_eq!(*poll_interval_seconds, 60);
                assert_eq!(id_field, "messageId");
                assert_eq!(content_field.as_deref(), Some("snippet"));
                assert!(items_field.is_none());
                assert_eq!(priority_senders, &["boss@company.com"]);
                assert_eq!(blocked_senders, &["spam@marketing.com"]);
                assert!(enrich_tool.is_none());
                assert_eq!(*dedup_ttl_seconds, 7 * 24 * 3600, "default dedup TTL");
                assert!(enrich_id_param.is_none());
            }
            other => panic!("expected Mcp variant, got {other:?}"),
        }
    }

    #[test]
    fn sensor_source_mcp_defaults() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "test_mcp"
server = "http://localhost:3000/mcp"
tool_name = "get_data"
kafka_topic = "hb.sensor.test"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let sensors = config.daemon.unwrap().sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::Mcp {
                poll_interval_seconds,
                modality,
                id_field,
                tool_args,
                content_field,
                items_field,
                priority_senders,
                blocked_senders,
                enrich_tool,
                enrich_id_param,
                ..
            } => {
                assert_eq!(*poll_interval_seconds, 60, "default poll interval");
                assert_eq!(*modality, SensorModality::Text, "default modality");
                assert_eq!(id_field, "id", "default id_field");
                assert_eq!(*tool_args, serde_json::json!({}), "default tool_args");
                assert!(content_field.is_none(), "default content_field");
                assert!(items_field.is_none(), "default items_field");
                assert!(priority_senders.is_empty(), "default priority_senders");
                assert!(blocked_senders.is_empty(), "default blocked_senders");
                assert!(enrich_tool.is_none(), "default enrich_tool");
                assert!(enrich_id_param.is_none(), "default enrich_id_param");
            }
            other => panic!("expected Mcp variant, got {other:?}"),
        }
    }

    #[test]
    fn sensor_source_mcp_with_enrichment() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "gmail_inbox"
server = "http://localhost:4000/mcp"
tool_name = "gmail_list_messages"
tool_args = { q = "is:unread", maxResults = 20 }
kafka_topic = "hb.sensor.email"
id_field = "id"
content_field = "snippet"
items_field = "messages"
enrich_tool = "gmail_get_message"
enrich_id_param = "messageId"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let sensors = config.daemon.unwrap().sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::Mcp {
                enrich_tool,
                enrich_id_param,
                ..
            } => {
                assert_eq!(enrich_tool.as_deref(), Some("gmail_get_message"));
                assert_eq!(enrich_id_param.as_deref(), Some("messageId"));
            }
            other => panic!("expected Mcp variant, got {other:?}"),
        }
    }

    #[test]
    fn sensor_source_mcp_with_auth_header() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "auth_mcp"
tool_name = "list_items"
kafka_topic = "hb.sensor.items"

[daemon.sensors.sources.server]
url = "http://gateway:3000/mcp"
auth_header = "Bearer secret-token"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let sensors = config.daemon.unwrap().sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::Mcp { server, .. } => {
                assert_eq!(server.url(), "http://gateway:3000/mcp");
                assert_eq!(server.auth_header(), Some("Bearer secret-token"));
            }
            other => panic!("expected Mcp variant, got {other:?}"),
        }
    }

    #[test]
    fn sensor_source_mcp_simple_server() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "simple_mcp"
server = "http://localhost:8080/mcp"
tool_name = "poll"
kafka_topic = "hb.sensor.data"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let sensors = config.daemon.unwrap().sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::Mcp { server, .. } => {
                assert_eq!(server.url(), "http://localhost:8080/mcp");
                assert!(server.auth_header().is_none());
            }
            other => panic!("expected Mcp variant, got {other:?}"),
        }
    }

    #[test]
    fn sensor_source_mcp_empty_tool_name_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "bad_mcp"
server = "http://localhost:3000/mcp"
tool_name = ""
kafka_topic = "hb.sensor.test"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string().contains("tool_name must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn sensor_source_mcp_empty_kafka_topic_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "bad_mcp"
server = "http://localhost:3000/mcp"
tool_name = "get_data"
kafka_topic = ""

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string().contains("kafka_topic must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn sensor_source_mcp_stdio_server() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "gmail_stdio"
tool_name = "gmail_search"
kafka_topic = "hb.sensor.email"

[daemon.sensors.sources.server]
command = "npx"
args = ["-y", "@anthropic/google-workspace-mcp"]

[daemon.sensors.sources.server.env]
GOOGLE_OAUTH_CREDENTIALS = "/path/to/creds.json"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let sensors = config.daemon.unwrap().sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::Mcp { server, .. } => {
                assert!(server.is_stdio());
                assert_eq!(server.url(), ""); // no URL for stdio
                assert!(server.auth_header().is_none());
                assert_eq!(
                    server.display_name(),
                    "npx -y @anthropic/google-workspace-mcp"
                );
                if let McpServerEntry::Stdio { command, args, env } = &**server {
                    assert_eq!(command, "npx");
                    assert_eq!(args, &["-y", "@anthropic/google-workspace-mcp"]);
                    assert_eq!(
                        env.get("GOOGLE_OAUTH_CREDENTIALS").unwrap(),
                        "/path/to/creds.json"
                    );
                } else {
                    panic!("expected Stdio variant");
                }
            }
            other => panic!("expected Mcp, got {other:?}"),
        }
    }

    #[test]
    fn sensor_source_mcp_stdio_empty_command_rejected() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "bad_stdio"
tool_name = "search"
kafka_topic = "hb.sensor.test"

[daemon.sensors.sources.server]
command = ""

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let err = HeartbitConfig::from_toml(toml_str).unwrap_err();
        assert!(
            err.to_string().contains("server command must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn sensor_source_mcp_stdio_defaults() {
        let toml_str = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[orchestrator]

[[daemon.sensors.sources]]
type = "mcp"
name = "stdio_defaults"
tool_name = "list_events"
kafka_topic = "hb.sensor.calendar"

[daemon.sensors.sources.server]
command = "my-mcp-server"

[daemon.kafka]
brokers = "localhost:9092"
"#;
        let config = HeartbitConfig::from_toml(toml_str).unwrap();
        let sensors = config.daemon.unwrap().sensors.unwrap();
        match &sensors.sources[0] {
            SensorSourceConfig::Mcp { server, .. } => {
                if let McpServerEntry::Stdio { command, args, env } = &**server {
                    assert_eq!(command, "my-mcp-server");
                    assert!(args.is_empty(), "args should default to empty");
                    assert!(env.is_empty(), "env should default to empty");
                } else {
                    panic!("expected Stdio variant");
                }
            }
            other => panic!("expected Mcp, got {other:?}"),
        }
    }

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

    #[test]
    fn validate_daemon_schedule_empty_task() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"

[daemon.kafka]
brokers = "localhost:9092"

[[daemon.schedules]]
name = "noop"
cron = "0 0 * * * *"
task = ""
"#;
        let err = HeartbitConfig::from_toml(toml).unwrap_err();
        assert!(
            err.to_string().contains("task must not be empty"),
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
}
