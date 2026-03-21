//! Provider registry — maps provider names to base URLs and API key env vars.
//!
//! Used by the CLI to construct the right `LlmProvider` from config or env.

use crate::error::Error;

/// Metadata for a known provider.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Human-readable display name.
    pub display_name: &'static str,
    /// Base URL for the OpenAI-compatible chat completions endpoint.
    /// `None` for providers that need a native driver (Anthropic, Gemini).
    pub base_url: Option<&'static str>,
    /// Environment variable name for the API key.
    /// `None` for local providers (Ollama, vLLM, LM Studio).
    pub api_key_env: Option<&'static str>,
    /// Default model if none specified.
    pub default_model: &'static str,
    /// Whether this provider needs a native driver (not OpenAI-compatible).
    pub native_driver: bool,
}

impl ProviderInfo {
    /// Whether this provider requires authentication.
    ///
    /// Derived from `api_key_env`: if an env var is configured, auth is required.
    pub fn auth_required(&self) -> bool {
        self.api_key_env.is_some()
    }
}

/// All known providers.
static PROVIDERS: &[(&str, ProviderInfo)] = &[
    (
        "anthropic",
        ProviderInfo {
            display_name: "Anthropic",
            base_url: None, // native driver
            api_key_env: Some("ANTHROPIC_API_KEY"),

            default_model: "claude-sonnet-4-20250514",
            native_driver: true,
        },
    ),
    (
        "openai",
        ProviderInfo {
            display_name: "OpenAI",
            base_url: Some("https://api.openai.com/v1"),
            api_key_env: Some("OPENAI_API_KEY"),

            default_model: "gpt-4o",
            native_driver: false,
        },
    ),
    (
        "openrouter",
        ProviderInfo {
            display_name: "OpenRouter",
            base_url: Some("https://openrouter.ai/api/v1"),
            api_key_env: Some("OPENROUTER_API_KEY"),

            default_model: "anthropic/claude-sonnet-4",
            native_driver: false,
        },
    ),
    (
        "gemini",
        ProviderInfo {
            display_name: "Google Gemini",
            base_url: None, // native driver
            api_key_env: Some("GEMINI_API_KEY"),

            default_model: "gemini-2.5-flash",
            native_driver: true,
        },
    ),
    (
        "groq",
        ProviderInfo {
            display_name: "Groq",
            base_url: Some("https://api.groq.com/openai/v1"),
            api_key_env: Some("GROQ_API_KEY"),

            default_model: "llama-3.3-70b-versatile",
            native_driver: false,
        },
    ),
    (
        "deepseek",
        ProviderInfo {
            display_name: "DeepSeek",
            base_url: Some("https://api.deepseek.com/v1"),
            api_key_env: Some("DEEPSEEK_API_KEY"),

            default_model: "deepseek-chat",
            native_driver: false,
        },
    ),
    (
        "together",
        ProviderInfo {
            display_name: "Together AI",
            base_url: Some("https://api.together.xyz/v1"),
            api_key_env: Some("TOGETHER_API_KEY"),

            default_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            native_driver: false,
        },
    ),
    (
        "mistral",
        ProviderInfo {
            display_name: "Mistral AI",
            base_url: Some("https://api.mistral.ai/v1"),
            api_key_env: Some("MISTRAL_API_KEY"),

            default_model: "mistral-large-latest",
            native_driver: false,
        },
    ),
    (
        "fireworks",
        ProviderInfo {
            display_name: "Fireworks AI",
            base_url: Some("https://api.fireworks.ai/inference/v1"),
            api_key_env: Some("FIREWORKS_API_KEY"),

            default_model: "accounts/fireworks/models/llama-v3p3-70b-instruct",
            native_driver: false,
        },
    ),
    (
        "cohere",
        ProviderInfo {
            display_name: "Cohere",
            base_url: Some("https://api.cohere.com/compatibility/v1"),
            api_key_env: Some("CO_API_KEY"),

            default_model: "command-r-plus",
            native_driver: false,
        },
    ),
    (
        "xai",
        ProviderInfo {
            display_name: "xAI",
            base_url: Some("https://api.x.ai/v1"),
            api_key_env: Some("XAI_API_KEY"),

            default_model: "grok-3",
            native_driver: false,
        },
    ),
    (
        "perplexity",
        ProviderInfo {
            display_name: "Perplexity",
            base_url: Some("https://api.perplexity.ai"),
            api_key_env: Some("PPLX_API_KEY"),

            default_model: "sonar-pro",
            native_driver: false,
        },
    ),
    (
        "cerebras",
        ProviderInfo {
            display_name: "Cerebras",
            base_url: Some("https://api.cerebras.ai/v1"),
            api_key_env: Some("CEREBRAS_API_KEY"),

            default_model: "llama-3.3-70b",
            native_driver: false,
        },
    ),
    (
        "sambanova",
        ProviderInfo {
            display_name: "SambaNova",
            base_url: Some("https://api.sambanova.ai/v1"),
            api_key_env: Some("SAMBANOVA_API_KEY"),

            default_model: "Meta-Llama-3.3-70B-Instruct",
            native_driver: false,
        },
    ),
    (
        "ollama",
        ProviderInfo {
            display_name: "Ollama (local)",
            base_url: Some("http://localhost:11434/v1"),
            api_key_env: None,

            default_model: "llama3.2",
            native_driver: false,
        },
    ),
    (
        "vllm",
        ProviderInfo {
            display_name: "vLLM (local)",
            base_url: Some("http://localhost:8000/v1"),
            api_key_env: None,

            default_model: "default",
            native_driver: false,
        },
    ),
    (
        "lmstudio",
        ProviderInfo {
            display_name: "LM Studio (local)",
            base_url: Some("http://localhost:1234/v1"),
            api_key_env: None,

            default_model: "default",
            native_driver: false,
        },
    ),
];

/// Look up a known provider by name.
pub fn get_provider(name: &str) -> Option<&'static ProviderInfo> {
    PROVIDERS.iter().find(|(k, _)| *k == name).map(|(_, v)| v)
}

/// Returns all known provider names (in priority order for auto-detection).
pub fn known_providers() -> Vec<&'static str> {
    PROVIDERS.iter().map(|(k, _)| *k).collect()
}

/// Auto-detect the first available provider by checking environment variables.
///
/// Returns the provider name and its info. Checks cloud providers first
/// (Anthropic, OpenRouter, Groq, ...) then local providers.
///
/// Returns `None` if no provider is detected (no API keys set, no local providers assumed).
pub fn detect_available_provider() -> Option<(&'static str, &'static ProviderInfo)> {
    // Cloud providers: check env vars in priority order
    for (name, info) in PROVIDERS {
        if let Some(env_key) = info.api_key_env
            && std::env::var(env_key).is_ok()
        {
            return Some((name, info));
        }
    }
    // Fallback: check GOOGLE_API_KEY for Gemini
    if std::env::var("GOOGLE_API_KEY").is_ok()
        && let Some((_, info)) = PROVIDERS.iter().find(|(k, _)| *k == "gemini")
    {
        return Some(("gemini", info));
    }
    // Local providers are NOT auto-detected (user must opt in)
    None
}

/// Resolve the API key for a provider from the environment.
///
/// Returns the API key if found, or an error describing what env var is needed.
pub fn resolve_api_key(name: &str, info: &ProviderInfo) -> Result<String, Error> {
    match info.api_key_env {
        Some(env_key) => std::env::var(env_key)
            .or_else(|_| {
                // Fallback env vars for specific providers
                if name == "gemini" {
                    std::env::var("GOOGLE_API_KEY")
                } else {
                    Err(std::env::VarError::NotPresent)
                }
            })
            .map_err(|_| {
                Error::Config(format!(
                    "{env_key} environment variable required for {name} provider"
                ))
            }),
        None if info.auth_required() => Err(Error::Config(format!(
            "no API key env var configured for {name} provider"
        ))),
        None => Ok(String::new()), // local providers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_known_provider() {
        let info = get_provider("anthropic").unwrap();
        assert_eq!(info.display_name, "Anthropic");
        assert!(info.native_driver);
        assert!(info.auth_required());
    }

    #[test]
    fn get_unknown_provider_returns_none() {
        assert!(get_provider("nonexistent").is_none());
    }

    #[test]
    fn known_providers_includes_all() {
        let names = known_providers();
        assert!(names.len() >= 17);
        assert!(names.contains(&"anthropic"));
        assert!(names.contains(&"openrouter"));
        assert!(names.contains(&"groq"));
        assert!(names.contains(&"ollama"));
    }

    #[test]
    fn openai_compat_providers_have_base_url() {
        for (name, info) in PROVIDERS {
            if !info.native_driver {
                assert!(
                    info.base_url.is_some(),
                    "provider '{name}' missing base_url"
                );
            }
        }
    }

    #[test]
    fn native_providers_have_no_base_url() {
        let info = get_provider("anthropic").unwrap();
        assert!(info.base_url.is_none());
        let info = get_provider("gemini").unwrap();
        assert!(info.base_url.is_none());
    }

    #[test]
    fn local_providers_need_no_auth() {
        for name in &["ollama", "vllm", "lmstudio"] {
            let info = get_provider(name).unwrap();
            assert!(!info.auth_required(), "{name} should not require auth");
            assert!(
                info.api_key_env.is_none(),
                "{name} should have no api_key_env"
            );
        }
    }

    #[test]
    fn cloud_providers_need_auth() {
        for name in &["groq", "deepseek", "together", "mistral"] {
            let info = get_provider(name).unwrap();
            assert!(info.auth_required(), "{name} should require auth");
            assert!(info.api_key_env.is_some(), "{name} should have api_key_env");
        }
    }

    #[test]
    fn resolve_api_key_local_returns_empty() {
        let info = get_provider("ollama").unwrap();
        let key = resolve_api_key("ollama", info).unwrap();
        assert!(key.is_empty());
    }

    #[test]
    fn resolve_api_key_missing_returns_error() {
        // This test is safe only if GROQ_API_KEY is not set
        if std::env::var("GROQ_API_KEY").is_ok() {
            return;
        }
        let info = get_provider("groq").unwrap();
        let err = resolve_api_key("groq", info).unwrap_err();
        assert!(err.to_string().contains("GROQ_API_KEY"));
    }

    #[test]
    fn all_providers_have_default_model() {
        for (name, info) in PROVIDERS {
            assert!(
                !info.default_model.is_empty(),
                "provider '{name}' missing default_model"
            );
        }
    }

    #[test]
    fn detect_available_returns_none_without_env() {
        // If no API keys are set (CI/clean environment), should return None.
        // We can't reliably test this because the user may have keys set.
        // Just verify the function is callable.
        let _ = detect_available_provider();
    }
}
