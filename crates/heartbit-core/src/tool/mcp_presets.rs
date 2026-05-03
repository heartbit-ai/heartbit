//! Bundled MCP server presets for common integrations.
//!
//! Presets are embedded at compile time from `configs/mcp-presets/*.json`.
//! Users can reference them by name in config files:
//!
//! ```toml
//! [[agents.mcp_servers]]
//! preset = "github"
//! ```

#![allow(missing_docs)]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Error;

/// A bundled MCP server preset.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpPreset {
    pub name: String,
    pub description: String,
    pub command: String,
    pub args: Vec<String>,
    pub env_keys: Vec<String>,
}

/// Embedded preset data (compile-time).
static PRESETS: &[(&str, &str)] = &[
    ("github", include_str!("../../mcp-presets/github.json")),
    ("gitlab", include_str!("../../mcp-presets/gitlab.json")),
    ("slack", include_str!("../../mcp-presets/slack.json")),
    ("notion", include_str!("../../mcp-presets/notion.json")),
    (
        "postgresql",
        include_str!("../../mcp-presets/postgresql.json"),
    ),
    (
        "brave-search",
        include_str!("../../mcp-presets/brave-search.json"),
    ),
    ("sentry", include_str!("../../mcp-presets/sentry.json")),
    ("linear", include_str!("../../mcp-presets/linear.json")),
    (
        "google-calendar",
        include_str!("../../mcp-presets/google-calendar.json"),
    ),
    ("jira", include_str!("../../mcp-presets/jira.json")),
];

/// Resolve a preset name to its MCP server configuration.
///
/// Returns `Error::Config` if the preset name is unknown.
pub fn resolve_preset(name: &str) -> Result<McpPreset, Error> {
    for (key, json) in PRESETS {
        if *key == name {
            let preset: McpPreset = serde_json::from_str(json)
                .map_err(|e| Error::Config(format!("failed to parse preset '{name}': {e}")))?;
            return Ok(preset);
        }
    }
    Err(Error::Config(format!(
        "unknown MCP preset '{name}'. Available presets: {}",
        known_presets().join(", ")
    )))
}

/// Returns the list of all known preset names.
pub fn known_presets() -> Vec<&'static str> {
    PRESETS.iter().map(|(k, _)| *k).collect()
}

/// Check which environment variables for a preset are currently set.
///
/// Returns a map of env var name to whether it is set.
pub fn check_preset_env(preset: &McpPreset) -> HashMap<String, bool> {
    preset
        .env_keys
        .iter()
        .map(|key| (key.clone(), std::env::var(key).is_ok()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_github_preset() {
        let preset = resolve_preset("github").expect("should resolve github");
        assert_eq!(preset.name, "GitHub");
        assert_eq!(preset.command, "npx");
        assert!(!preset.env_keys.is_empty());
    }

    #[test]
    fn resolve_all_presets() {
        for name in known_presets() {
            let preset = resolve_preset(name).unwrap_or_else(|e| {
                panic!("failed to resolve preset '{name}': {e}");
            });
            assert!(!preset.name.is_empty(), "preset '{name}' has empty name");
            assert!(
                !preset.command.is_empty(),
                "preset '{name}' has empty command"
            );
        }
    }

    #[test]
    fn resolve_unknown_preset_returns_error() {
        let err = resolve_preset("nonexistent").unwrap_err();
        assert!(err.to_string().contains("unknown MCP preset"));
        assert!(err.to_string().contains("github")); // lists available presets
    }

    #[test]
    fn known_presets_returns_all() {
        let presets = known_presets();
        assert_eq!(presets.len(), 10);
        assert!(presets.contains(&"github"));
        assert!(presets.contains(&"jira"));
    }

    #[test]
    fn check_preset_env_returns_status() {
        let preset = resolve_preset("github").expect("should resolve github");
        let status = check_preset_env(&preset);
        assert!(status.contains_key("GITHUB_PERSONAL_ACCESS_TOKEN"));
    }

    #[test]
    fn preset_json_round_trip() {
        // Verify all presets can be deserialized and re-serialized
        for name in known_presets() {
            let preset = resolve_preset(name).expect("should resolve");
            let json = serde_json::to_string(&preset);
            assert!(json.is_ok(), "preset '{name}' failed to serialize");
        }
    }
}
