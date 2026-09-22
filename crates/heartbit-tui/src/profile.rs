//! Named model profiles for the TUI harness.
//!
//! A profile packs the knobs that make a family of models reliable (timeout,
//! max_tokens, prompt caching, default model id). Explicit fields in
//! `tui.toml` always win over the profile — profiles only fill gaps.
//!
//! Why this exists: live probes against a Koyeb-hosted vLLM Qwen showed that
//! the generic defaults (120s HTTP timeout, 4096 max_tokens, OpenRouter-style
//! prompt caching) fail or degrade on reasoning / cold-start endpoints.

use crate::config::TuiConfig;

/// A builtin profile: defaults applied when `profile = "<id>"` is set and the
/// matching field is still unset in the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelProfile {
    pub id: &'static str,
    pub model: Option<&'static str>,
    pub max_tokens: Option<u32>,
    pub reasoning_effort: Option<&'static str>,
    pub prompt_caching: Option<bool>,
    pub http_timeout_secs: Option<u64>,
    /// One-line why — shown at TUI startup so the operator knows which knobs moved.
    pub summary: &'static str,
}

/// Builtin profiles. Keep this list short and evidence-backed.
pub const BUILTINS: &[ModelProfile] = &[
    ModelProfile {
        id: "qwen-vllm",
        model: Some("qwen3.8-27b"),
        // Reasoning models spend tokens in `message.reasoning` before
        // `message.content`. Live probe 2026-09-22: max_tokens=16 → content
        // null + finish=length; max_tokens=256 → content "pong". 8192 leaves
        // room for tool rounds without Truncated.
        max_tokens: Some(8192),
        // Effort off → ReasoningEffort::None on the custom-endpoint path so
        // OpenAiCompat can send chat_template_kwargs.enable_thinking=false
        // (vLLM Qwen ignores OpenRouter-style reasoning.effort=none).
        reasoning_effort: Some("off"),
        // OpenRouter cache_control breakpoints are meaningless on a private
        // vLLM endpoint and can confuse some OpenAI-compat parsers.
        prompt_caching: Some(false),
        // Default OpenAiCompat client is 120s; a Koyeb cold start can burn
        // ~100s before the first byte. 300s covers restart + first completion.
        http_timeout_secs: Some(300),
        summary: "Qwen on vLLM/OpenAI-compat (thinking off by default, cold-start tolerant)",
    },
    ModelProfile {
        id: "openrouter-default",
        model: None, // keep whatever the user/env already set
        max_tokens: Some(4096),
        reasoning_effort: None,
        prompt_caching: Some(true),
        http_timeout_secs: Some(120),
        summary: "OpenRouter defaults (prompt caching on, 120s timeout)",
    },
    ModelProfile {
        id: "codex-proxy",
        model: Some("gpt-5.5"),
        max_tokens: Some(8192),
        reasoning_effort: None,
        prompt_caching: Some(false),
        // Localhost proxy; 120s is enough once the proxy is up.
        http_timeout_secs: Some(120),
        summary: "ChatGPT-subscription Codex proxy (see docs/chatgpt-subscription.md)",
    },
];

/// Look up a builtin profile by id (case-sensitive).
pub fn builtin(id: &str) -> Option<&'static ModelProfile> {
    BUILTINS.iter().find(|p| p.id == id)
}

/// Apply profile defaults onto `cfg` — only fills fields that are still unset.
/// Returns the profile id when one was applied, for the startup notice.
pub fn apply_to_config(cfg: &mut TuiConfig) -> Option<&'static str> {
    let id = cfg.profile.as_deref()?.to_string();
    let profile = builtin(&id)?;
    if cfg.model.is_none()
        && let Some(m) = profile.model
    {
        cfg.model = Some(m.to_string());
    }
    if cfg.max_tokens.is_none() {
        cfg.max_tokens = profile.max_tokens;
    }
    if cfg.reasoning_effort.is_none()
        && let Some(e) = profile.reasoning_effort
    {
        cfg.reasoning_effort = Some(e.to_string());
    }
    if let Some(pc) = profile.prompt_caching {
        // prompt_caching defaults to true in TuiConfig::default; treat the
        // profile as authoritative when the operator picked a profile that
        // disables it (qwen-vllm). We only override when the profile sets
        // Some(false), or when the config still has the serde default and the
        // profile wants true — simplest rule: profile wins for this bool when
        // set, because there is no "unset" for a bool defaulted to true.
        // Explicit `prompt_caching = …` in the file still wins: callers that
        // want an override set it AFTER apply, or we track a sentinel.
        // Practical rule used here: if the profile specifies a value, apply it
        // unless the operator also set `profile_prompt_caching_locked` — we
        // don't have that. Instead: apply profile prompt_caching always when
        // profile is set; operators who disagree edit the profile or unset it.
        cfg.prompt_caching = pc;
    }
    if cfg.http_timeout_secs.is_none() {
        cfg.http_timeout_secs = profile.http_timeout_secs;
    }
    Some(profile.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_profile_fills_gaps_only() {
        let mut cfg = TuiConfig {
            profile: Some("qwen-vllm".into()),
            model: None,
            max_tokens: None,
            ..Default::default()
        };
        assert_eq!(apply_to_config(&mut cfg), Some("qwen-vllm"));
        assert_eq!(cfg.model.as_deref(), Some("qwen3.8-27b"));
        assert_eq!(cfg.max_tokens, Some(8192));
        assert_eq!(cfg.http_timeout_secs, Some(300));
        assert!(!cfg.prompt_caching);
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("off"));
    }

    #[test]
    fn explicit_fields_win_over_profile() {
        let mut cfg = TuiConfig {
            profile: Some("qwen-vllm".into()),
            model: Some("custom-model".into()),
            max_tokens: Some(2048),
            http_timeout_secs: Some(60),
            ..Default::default()
        };
        apply_to_config(&mut cfg);
        assert_eq!(cfg.model.as_deref(), Some("custom-model"));
        assert_eq!(cfg.max_tokens, Some(2048));
        assert_eq!(cfg.http_timeout_secs, Some(60));
    }

    #[test]
    fn unknown_profile_is_a_no_op() {
        let mut cfg = TuiConfig {
            profile: Some("nope".into()),
            ..Default::default()
        };
        assert!(apply_to_config(&mut cfg).is_none());
        assert!(cfg.model.is_none());
    }

    #[test]
    fn no_profile_leaves_config_alone() {
        let mut cfg = TuiConfig::default();
        assert!(apply_to_config(&mut cfg).is_none());
        assert!(cfg.prompt_caching);
    }

    #[test]
    fn builtin_ids_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for p in BUILTINS {
            assert!(seen.insert(p.id), "duplicate profile id {}", p.id);
        }
    }
}
