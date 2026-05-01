/// Controls the verbosity of agent execution observability.
///
/// Configurable via:
/// 1. `HEARTBIT_OBSERVABILITY` env var (highest priority)
/// 2. `[telemetry] observability_mode` in config TOML
/// 3. `AgentRunnerBuilder::observability_mode()` / `OrchestratorBuilder::observability_mode()`
/// 4. Default: `Production`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObservabilityMode {
    /// Near-zero overhead: span names + durations only.
    #[default]
    Production,
    /// Metrics: tokens, latencies, costs, stop reasons.
    Analysis,
    /// Full payloads (truncated to 4KB) for debugging.
    Debug,
}

impl ObservabilityMode {
    /// Parse from a case-insensitive string. Returns `None` for unknown values.
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "production" | "prod" => Some(Self::Production),
            "analysis" | "analyze" => Some(Self::Analysis),
            "debug" | "dbg" => Some(Self::Debug),
            _ => None,
        }
    }

    /// Resolve the effective mode from all configuration sources.
    ///
    /// Priority: env var > config string > builder value > default.
    pub fn resolve(
        env_key: &str,
        config_str: Option<&str>,
        builder_value: Option<ObservabilityMode>,
    ) -> Self {
        // 1. Environment variable (highest priority)
        if let Ok(val) = std::env::var(env_key) {
            if let Some(mode) = Self::from_str_loose(&val) {
                return mode;
            }
            tracing::warn!(
                env = env_key,
                value = %val,
                "unknown observability mode, falling back"
            );
        }
        // 2. Config file string
        if let Some(s) = config_str {
            if let Some(mode) = Self::from_str_loose(s) {
                return mode;
            }
            tracing::warn!(
                value = %s,
                "unknown observability mode in config, falling back"
            );
        }
        // 3. Builder value
        if let Some(mode) = builder_value {
            return mode;
        }
        // 4. Default
        Self::default()
    }

    /// Whether this mode includes metrics (token counts, latencies, costs).
    pub fn includes_metrics(self) -> bool {
        matches!(self, Self::Analysis | Self::Debug)
    }

    /// Whether this mode includes full payloads (request/response text, tool I/O).
    pub fn includes_payloads(self) -> bool {
        matches!(self, Self::Debug)
    }
}

impl std::fmt::Display for ObservabilityMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Production => write!(f, "production"),
            Self::Analysis => write!(f, "analysis"),
            Self::Debug => write!(f, "debug"),
        }
    }
}

/// Environment variable key for observability mode.
pub const OBSERVABILITY_ENV_KEY: &str = "HEARTBIT_OBSERVABILITY";

// --- OpenTelemetry GenAI Semantic Convention constants (v1.38.0) ---
// Used as span attribute names so OTel-compatible backends (Jaeger, Grafana
// Tempo, Arize Phoenix) recognize the spans as GenAI operations.

/// The GenAI system producing the response (e.g., "anthropic", "openrouter").
pub const GEN_AI_SYSTEM: &str = "gen_ai.system";
/// Model requested by the caller.
pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
/// Model that actually generated the response (may differ from request).
pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";
/// Reason the model stopped generating (e.g., "end_turn", "tool_use").
pub const GEN_AI_RESPONSE_FINISH_REASON: &str = "gen_ai.response.finish_reasons";
/// Number of input tokens consumed.
pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
/// Number of output tokens generated.
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_default_is_production() {
        assert_eq!(ObservabilityMode::default(), ObservabilityMode::Production);
    }

    #[test]
    fn from_str_loose_all_variants() {
        assert_eq!(
            ObservabilityMode::from_str_loose("production"),
            Some(ObservabilityMode::Production)
        );
        assert_eq!(
            ObservabilityMode::from_str_loose("prod"),
            Some(ObservabilityMode::Production)
        );
        assert_eq!(
            ObservabilityMode::from_str_loose("PRODUCTION"),
            Some(ObservabilityMode::Production)
        );
        assert_eq!(
            ObservabilityMode::from_str_loose("analysis"),
            Some(ObservabilityMode::Analysis)
        );
        assert_eq!(
            ObservabilityMode::from_str_loose("analyze"),
            Some(ObservabilityMode::Analysis)
        );
        assert_eq!(
            ObservabilityMode::from_str_loose("ANALYSIS"),
            Some(ObservabilityMode::Analysis)
        );
        assert_eq!(
            ObservabilityMode::from_str_loose("debug"),
            Some(ObservabilityMode::Debug)
        );
        assert_eq!(
            ObservabilityMode::from_str_loose("dbg"),
            Some(ObservabilityMode::Debug)
        );
        assert_eq!(
            ObservabilityMode::from_str_loose("DEBUG"),
            Some(ObservabilityMode::Debug)
        );
    }

    #[test]
    fn from_str_loose_unknown_returns_none() {
        assert_eq!(ObservabilityMode::from_str_loose("banana"), None);
        assert_eq!(ObservabilityMode::from_str_loose(""), None);
    }

    #[test]
    fn resolve_env_overrides_config() {
        // Use a unique env var to avoid test interference
        let key = "HEARTBIT_OBSERVABILITY_TEST_1";
        unsafe {
            std::env::set_var(key, "debug");
        }
        let mode =
            ObservabilityMode::resolve(key, Some("production"), Some(ObservabilityMode::Analysis));
        assert_eq!(mode, ObservabilityMode::Debug);
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn resolve_config_overrides_builder() {
        let key = "HEARTBIT_OBSERVABILITY_TEST_2";
        unsafe {
            std::env::remove_var(key);
        }
        let mode =
            ObservabilityMode::resolve(key, Some("analysis"), Some(ObservabilityMode::Production));
        assert_eq!(mode, ObservabilityMode::Analysis);
    }

    #[test]
    fn resolve_default_fallback() {
        let key = "HEARTBIT_OBSERVABILITY_TEST_3";
        unsafe {
            std::env::remove_var(key);
        }
        let mode = ObservabilityMode::resolve(key, None, None);
        assert_eq!(mode, ObservabilityMode::Production);
    }

    #[test]
    fn includes_metrics_analysis_and_debug() {
        assert!(!ObservabilityMode::Production.includes_metrics());
        assert!(ObservabilityMode::Analysis.includes_metrics());
        assert!(ObservabilityMode::Debug.includes_metrics());
    }

    #[test]
    fn includes_payloads_debug_only() {
        assert!(!ObservabilityMode::Production.includes_payloads());
        assert!(!ObservabilityMode::Analysis.includes_payloads());
        assert!(ObservabilityMode::Debug.includes_payloads());
    }

    #[test]
    fn display_impl() {
        assert_eq!(ObservabilityMode::Production.to_string(), "production");
        assert_eq!(ObservabilityMode::Analysis.to_string(), "analysis");
        assert_eq!(ObservabilityMode::Debug.to_string(), "debug");
    }

    // --- GenAI Semantic Convention constants tests ---

    #[test]
    fn gen_ai_constants_match_spec() {
        // Verify string values match OTel GenAI Semantic Conventions v1.38.0
        assert_eq!(GEN_AI_SYSTEM, "gen_ai.system");
        assert_eq!(GEN_AI_REQUEST_MODEL, "gen_ai.request.model");
        assert_eq!(GEN_AI_RESPONSE_MODEL, "gen_ai.response.model");
        assert_eq!(
            GEN_AI_RESPONSE_FINISH_REASON,
            "gen_ai.response.finish_reasons"
        );
        assert_eq!(GEN_AI_USAGE_INPUT_TOKENS, "gen_ai.usage.input_tokens");
        assert_eq!(GEN_AI_USAGE_OUTPUT_TOKENS, "gen_ai.usage.output_tokens");
    }
}
