//! Bundled MCP server presets for common integrations.
//!
//! Presets are embedded at compile time from `configs/mcp-presets/*.json`.
//! Users can reference them by name in config files:
//!
//! ```toml
//! [[agents.mcp_servers]]
//! preset = "github"
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Error;
use crate::tool::Tool;
use crate::tool::mcp::McpClient;

/// A bundled MCP server preset.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpPreset {
    /// Preset identifier (e.g. `github`, `slack`).
    pub name: String,
    /// Human-readable preset description.
    pub description: String,
    /// Command to spawn the MCP server (typically `npx`).
    pub command: String,
    /// Command-line arguments.
    pub args: Vec<String>,
    /// Required environment-variable keys (looked up at runtime).
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
    (
        "chrome-devtools",
        include_str!("../../mcp-presets/chrome-devtools.json"),
    ),
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

/// Connect to a bundled MCP preset over stdio and return its tools, ready for
/// [`AgentRunnerBuilder::tools`](crate::AgentRunnerBuilder::tools) or
/// [`WorkflowCtxBuilder::base_tools`](crate::WorkflowCtxBuilder::base_tools).
///
/// This is the one-call bridge from a preset name to a usable tool set: it
/// resolves the preset, forwards any of its `env_keys` that are present in the
/// process environment to the spawned server, performs the MCP handshake, and
/// stamps the discovered tools as `Arc<dyn Tool>`. Missing `env_keys` are simply
/// omitted — the server decides whether they are required.
///
/// The returned tools own the server subprocess (via a shared transport), so the
/// child process lives as long as any tool is held and is killed on drop.
///
/// # Security
///
/// When combining these with trusted builtins, register the builtins **first**
/// (`builtins ++ mcp_tools`) so a hostile MCP server cannot shadow a trusted
/// builtin name (first-wins de-duplication in the runner).
pub async fn connect_preset(name: &str) -> Result<Vec<Arc<dyn Tool>>, Error> {
    let preset = resolve_preset(name)?;
    let env: HashMap<String, String> = preset
        .env_keys
        .iter()
        .filter_map(|key| std::env::var(key).ok().map(|val| (key.clone(), val)))
        .collect();
    let client = McpClient::connect_stdio(&preset.command, &preset.args, &env).await?;
    Ok(client.into_tools())
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
        assert_eq!(presets.len(), 11);
        assert!(presets.contains(&"github"));
        assert!(presets.contains(&"jira"));
        assert!(presets.contains(&"chrome-devtools"));
    }

    #[test]
    fn chrome_devtools_preset_is_bundled() {
        let p = resolve_preset("chrome-devtools").expect("chrome-devtools preset exists");
        assert_eq!(p.command, "npx");
        // Launches the official browser-automation server package.
        assert!(
            p.args.iter().any(|a| a.contains("chrome-devtools-mcp")),
            "args must launch chrome-devtools-mcp: {:?}",
            p.args
        );
        // CI/bot-safe defaults: no UI, throwaway profile that auto-cleans.
        assert!(
            p.args.iter().any(|a| a == "--headless"),
            "expected --headless"
        );
        assert!(
            p.args.iter().any(|a| a == "--isolated"),
            "expected --isolated"
        );
        // Driving a browser needs no secrets.
        assert!(p.env_keys.is_empty());
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
