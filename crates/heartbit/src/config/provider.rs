use serde::Deserialize;

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
        #[serde(default = "super::default_true")]
        accept_tool_calls: bool,
        /// Escalate on MaxTokens stop reason (default: true).
        #[serde(default = "super::default_true")]
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
