#![allow(missing_docs)]
use serde::Deserialize;
use std::time::Duration;

/// LLM provider configuration.
///
/// When running as a cloud-delegated runtime (daemon mode with no agents),
/// the provider section can be omitted — per-request provider keys are used instead.
#[derive(Debug, Default, Deserialize)]
pub struct ProviderConfig {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub model: String,
    /// Custom API endpoint URL (overrides the default for the provider).
    /// Useful for self-hosted models, Azure, or proxies.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Direct API key (alternative to environment variable).
    /// Prefer env vars in production; this is for testing/local dev.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Retry configuration for transient LLM API failures.
    pub retry: Option<RetryProviderConfig>,
    /// Enable Anthropic prompt caching (system prompt + tool definitions).
    /// Only effective for the `anthropic` provider. Defaults to `false`.
    #[serde(default)]
    pub prompt_caching: bool,
    /// Model cascading configuration. When enabled, tries cheaper models first
    /// and escalates to the main model only when the confidence gate rejects.
    pub cascade: Option<CascadeConfig>,
    /// Circuit breaker configuration for this provider.
    /// When absent, sensible defaults are used (5 failures → 30 s open, max 300 s).
    #[serde(default)]
    pub circuit: ProviderCircuitConfig,
}

/// Model cascading configuration for cost-efficient LLM selection.
///
/// When enabled, the provider tries cheaper model tiers first and only
/// escalates to the main (most expensive) model when the confidence gate
/// rejects the cheaper response or the tier errors.
#[derive(Debug, Clone, Deserialize)]
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
#[derive(Debug, Clone, Deserialize)]
pub struct CascadeTierConfig {
    pub model: String,
}

/// Confidence gate configuration for model cascading.
#[derive(Debug, Clone, Deserialize)]
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

/// Circuit breaker configuration for the LLM provider.
///
/// Controls how quickly the circuit opens on consecutive failures and how long
/// it stays open before allowing a probe request through. All fields are optional;
/// absent fields fall back to [`crate::llm::circuit::CircuitConfig`] defaults.
#[derive(Debug, Clone, Default, serde::Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCircuitConfig {
    /// Number of consecutive failures before the circuit opens. Must be > 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_threshold: Option<u32>,
    /// Initial duration in seconds the circuit stays open after tripping. Must be > 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_open_duration_seconds: Option<u32>,
    /// Maximum backoff duration in seconds before a half-open probe. Must be > 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_open_duration_seconds: Option<u32>,
    /// Backoff multiplier applied after each re-trip (exponential backoff).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_multiplier: Option<f64>,
}

impl From<&ProviderCircuitConfig> for crate::llm::circuit::CircuitConfig {
    fn from(c: &ProviderCircuitConfig) -> Self {
        let default = crate::llm::circuit::CircuitConfig::default();
        Self {
            failure_threshold: c.failure_threshold.unwrap_or(default.failure_threshold),
            initial_open_duration: c
                .initial_open_duration_seconds
                .map(|s| std::time::Duration::from_secs(u64::from(s)))
                .unwrap_or(default.initial_open_duration),
            max_open_duration: c
                .max_open_duration_seconds
                .map(|s| std::time::Duration::from_secs(u64::from(s)))
                .unwrap_or(default.max_open_duration),
            backoff_multiplier: c.backoff_multiplier.unwrap_or(default.backoff_multiplier),
        }
    }
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

impl From<&RetryProviderConfig> for crate::llm::retry::RetryConfig {
    fn from(r: &RetryProviderConfig) -> Self {
        Self {
            max_retries: r.max_retries,
            base_delay: Duration::from_millis(r.base_delay_ms),
            max_delay: Duration::from_millis(r.max_delay_ms),
        }
    }
}
