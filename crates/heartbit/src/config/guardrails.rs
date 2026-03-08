use serde::{Deserialize, Serialize};

use crate::Error;

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
    /// Secret scanning configuration.
    #[serde(default)]
    pub secret_scan: Option<SecretScanConfig>,
    /// Behavioral monitoring configuration.
    #[serde(default)]
    pub behavioral: Option<BehavioralConfig>,
    /// Action budget guardrail configuration.
    #[serde(default)]
    pub action_budget: Option<ActionBudgetConfig>,
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

pub(super) fn default_pii_detectors() -> Vec<String> {
    vec![
        "email".into(),
        "phone".into(),
        "ssn".into(),
        "credit_card".into(),
    ]
}

/// Configuration for the secret scanning guardrail.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SecretScanConfig {
    /// Action: `"redact"` or `"deny"`. Default: `"redact"`.
    #[serde(default = "default_secret_action")]
    pub action: String,
    /// Additional custom patterns as label+regex pairs.
    #[serde(default)]
    pub custom_patterns: Vec<SecretPatternConfig>,
}

/// A custom secret pattern (label + regex).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SecretPatternConfig {
    /// Human-readable label for the secret type.
    pub label: String,
    /// Regex pattern to match.
    pub pattern: String,
}

fn default_secret_action() -> String {
    "redact".into()
}

/// A single behavioral rule in TOML format.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct BehavioralRuleConfig {
    /// Rule type: `"frequency_limit"`, `"suspicious_sequence"`, or `"denial_spike"`.
    #[serde(rename = "type")]
    pub rule_type: String,
    /// Tool name pattern (for `frequency_limit`).
    #[serde(default)]
    pub tool_pattern: Option<String>,
    /// Maximum count threshold (for `frequency_limit`).
    #[serde(default)]
    pub max_count: Option<usize>,
    /// Time window in seconds (for `frequency_limit` and `denial_spike`).
    #[serde(default)]
    pub window_seconds: Option<u64>,
    /// First tool pattern in a suspicious sequence.
    #[serde(default)]
    pub first: Option<String>,
    /// Second tool pattern in a suspicious sequence.
    #[serde(default)]
    pub then: Option<String>,
    /// Turn window for suspicious sequences.
    #[serde(default)]
    pub within_turns: Option<usize>,
    /// Maximum denied calls before spike triggers (for `denial_spike`).
    #[serde(default)]
    pub max_denied: Option<usize>,
}

/// Behavioral monitoring guardrail configuration.
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct BehavioralConfig {
    /// Maximum entries in the sliding window. Default: 200.
    #[serde(default = "default_behavioral_window_size")]
    pub window_size: usize,
    /// Time-to-live for window entries in seconds. Default: 1800 (30 min).
    #[serde(default = "default_behavioral_window_ttl")]
    pub window_ttl_seconds: u64,
    /// Behavioral rules to enforce.
    #[serde(default)]
    pub rules: Vec<BehavioralRuleConfig>,
}

fn default_behavioral_window_size() -> usize {
    200
}

fn default_behavioral_window_ttl() -> u64 {
    1800
}

/// A single action budget rule in TOML format.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ActionBudgetRuleConfig {
    /// Tool name pattern (exact or glob with `*`).
    pub tool_pattern: String,
    /// Maximum number of calls allowed.
    pub max_calls: usize,
}

/// Action budget guardrail configuration.
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct ActionBudgetConfig {
    /// Default budget for tools not matching any rule.
    #[serde(default)]
    pub default_budget: Option<usize>,
    /// Per-tool budget rules.
    #[serde(default)]
    pub rules: Vec<ActionBudgetRuleConfig>,
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

impl GuardrailsConfig {
    /// Returns `true` if no guardrails are configured.
    pub fn is_empty(&self) -> bool {
        self.injection.is_none()
            && self.pii.is_none()
            && self.tool_policy.is_none()
            && self.llm_judge.is_none()
            && self.secret_scan.is_none()
            && self.behavioral.is_none()
            && self.action_budget.is_none()
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
        if let Some(cfg) = &self.llm_judge {
            if let Some(provider) = judge_provider {
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
            } else {
                tracing::warn!(
                    "[guardrails.llm_judge] is configured but no judge provider was supplied — \
                     LLM judge guardrail will NOT be active. Use build_with_judge(Some(provider))."
                );
            }
        }

        // 5. Secret scanner
        if let Some(cfg) = &self.secret_scan {
            use crate::agent::guardrails::secret_scanner::{SecretAction, SecretScannerGuardrail};

            let action = match cfg.action.as_str() {
                "redact" => SecretAction::Redact,
                "deny" => SecretAction::Deny,
                other => {
                    return Err(Error::Config(format!(
                        "invalid secret_scan action: `{other}` (expected \"redact\" or \"deny\")"
                    )));
                }
            };
            let mut builder = SecretScannerGuardrail::builder().action(action);
            for cp in &cfg.custom_patterns {
                let re = regex::Regex::new(&cp.pattern).map_err(|e| {
                    Error::Config(format!(
                        "invalid secret_scan custom pattern `{}`: {e}",
                        cp.label
                    ))
                })?;
                builder = builder.custom_pattern(&cp.label, re);
            }
            guardrails.push(Arc::new(builder.build()));
        }

        // 6. Behavioral monitor
        if let Some(cfg) = &self.behavioral {
            use crate::agent::guardrails::behavioral::{BehaviorRule, BehavioralMonitorGuardrail};

            let mut builder = BehavioralMonitorGuardrail::builder()
                .window_size(cfg.window_size)
                .window_ttl(std::time::Duration::from_secs(cfg.window_ttl_seconds));

            for rule_cfg in &cfg.rules {
                let rule = match rule_cfg.rule_type.as_str() {
                    "frequency_limit" => BehaviorRule::FrequencyLimit {
                        tool_pattern: rule_cfg.tool_pattern.clone().unwrap_or_else(|| "*".into()),
                        max_count: rule_cfg.max_count.unwrap_or(10),
                        window: std::time::Duration::from_secs(
                            rule_cfg.window_seconds.unwrap_or(60),
                        ),
                    },
                    "suspicious_sequence" => BehaviorRule::SuspiciousSequence {
                        first: rule_cfg.first.clone().unwrap_or_default(),
                        then: rule_cfg.then.clone().unwrap_or_default(),
                        within_turns: rule_cfg.within_turns.unwrap_or(3),
                    },
                    "denial_spike" => BehaviorRule::DenialSpike {
                        max_denied: rule_cfg.max_denied.unwrap_or(5),
                        window: std::time::Duration::from_secs(
                            rule_cfg.window_seconds.unwrap_or(60),
                        ),
                    },
                    other => {
                        return Err(Error::Config(format!(
                            "unknown behavioral rule type: `{other}` \
                             (expected \"frequency_limit\", \"suspicious_sequence\", or \"denial_spike\")"
                        )));
                    }
                };
                builder = builder.rule(rule);
            }
            guardrails.push(Arc::new(builder.build()));
        }

        // 7. Action budget
        if let Some(cfg) = &self.action_budget {
            use crate::agent::guardrails::action_budget::ActionBudgetGuardrail;

            let mut builder = ActionBudgetGuardrail::builder();
            if let Some(default) = cfg.default_budget {
                builder = builder.default_budget(default);
            }
            for rule in &cfg.rules {
                builder = builder.rule(&rule.tool_pattern, rule.max_calls);
            }
            guardrails.push(Arc::new(builder.build()));
        }

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
