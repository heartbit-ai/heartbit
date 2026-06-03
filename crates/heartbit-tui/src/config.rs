//! Persistent TUI configuration (`~/.config/heartbit/tui.toml`) — primarily the
//! OpenRouter API token, so it can be set once (or from inside the TUI) instead
//! of exporting it into the environment every time.
//!
//! Security: the file holds a secret, so it is written **0600** atomically (a
//! temp file opened 0600 in the same directory, then `rename`d into place — no
//! create-then-chmod window, no partial writes). `0600` protects against *other
//! local users*; it does not protect against the agent's own `bash` (same uid,
//! no Landlock in the default build). Crucially, storing the key here is safer
//! than exporting `OPENROUTER_API_KEY`, because the key never enters the agent's
//! tool environment (the TUI passes a no-secrets env allowlist to bash).

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One configured MCP server: a bundled preset (e.g. `chrome-devtools`) or a
/// stdio command to spawn. Serialized as a TOML array-of-tables entry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerSpec {
    /// A bundled preset name (takes precedence over `command`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    /// A command to spawn an MCP server over stdio (when `preset` is None).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Arguments for the stdio `command`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
}

impl McpServerSpec {
    /// A bundled-preset server.
    pub fn preset(name: impl Into<String>) -> Self {
        Self {
            preset: Some(name.into()),
            ..Default::default()
        }
    }

    /// A stdio command server.
    pub fn stdio(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: Some(command.into()),
            args,
            ..Default::default()
        }
    }

    /// A short human label for notices and `/mcp list`.
    pub fn label(&self) -> String {
        if let Some(p) = &self.preset {
            format!("preset:{p}")
        } else if let Some(c) = &self.command {
            if self.args.is_empty() {
                c.clone()
            } else {
                format!("{c} {}", self.args.join(" "))
            }
        } else {
            "<invalid>".into()
        }
    }
}

/// Persisted TUI settings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuiConfig {
    /// The OpenRouter API token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openrouter_api_key: Option<String>,
    /// The model id (e.g. `qwen/qwen3-235b-a22b-2507`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// MCP servers to connect when the agent starts (builtins still take priority).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServerSpec>,
    /// Run as a multi-agent orchestrator (dynamic delegation + squads) instead of
    /// a single agent. Toggled in-TUI via `/agents`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub multi_agent: bool,
}

impl TuiConfig {
    /// Load from `path`, returning defaults if the file is missing or malformed
    /// (a corrupt config must never prevent the TUI from starting).
    pub fn load_from(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Atomically persist to `path` with 0600 permissions, creating parent dirs.
    pub fn save_to(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                // Best-effort: tighten the config dir to owner-only.
                let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
            }
        }
        let body = toml::to_string_pretty(self).map_err(io::Error::other)?;
        let tmp = path.with_extension("tmp");
        write_secret(&tmp, body.as_bytes())?;
        // rename preserves the temp file's 0600 mode → final file is 0600.
        std::fs::rename(&tmp, path)
    }

    /// Load from the default location ([`config_path`]).
    pub fn load() -> Self {
        Self::load_from(&config_path())
    }

    /// Persist to the default location ([`config_path`]).
    pub fn save(&self) -> io::Result<()> {
        self.save_to(&config_path())
    }
}

/// Resolve the config file path: `HEARTBIT_TUI_CONFIG` env override, else
/// `$XDG_CONFIG_HOME/heartbit/tui.toml`, else `$HOME/.config/heartbit/tui.toml`.
pub fn config_path() -> PathBuf {
    if let Ok(p) = std::env::var("HEARTBIT_TUI_CONFIG")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    let base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".config")
        });
    base.join("heartbit").join("tui.toml")
}

#[cfg(unix)]
fn write_secret(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(bytes)
}

#[cfg(not(unix))]
fn write_secret(path: &Path, bytes: &[u8]) -> io::Result<()> {
    std::fs::write(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_key_and_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("tui.toml");
        let cfg = TuiConfig {
            openrouter_api_key: Some("sk-or-xyz".into()),
            model: Some("qwen/q".into()),
            ..Default::default()
        };
        cfg.save_to(&path).unwrap();
        assert!(
            path.exists(),
            "save should create the file (and parent dirs)"
        );
        let loaded = TuiConfig::load_from(&path);
        assert_eq!(loaded, cfg);
    }

    #[test]
    fn roundtrip_mcp_servers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tui.toml");
        let cfg = TuiConfig {
            mcp_servers: vec![
                McpServerSpec::preset("chrome-devtools"),
                McpServerSpec::stdio("npx", vec!["-y".into(), "some-mcp".into()]),
            ],
            ..Default::default()
        };
        cfg.save_to(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("chrome-devtools"),
            "preset missing in toml: {text}"
        );
        let loaded = TuiConfig::load_from(&path);
        assert_eq!(loaded, cfg, "mcp_servers must round-trip");
        assert_eq!(
            loaded.mcp_servers[0].preset.as_deref(),
            Some("chrome-devtools")
        );
        assert_eq!(loaded.mcp_servers[1].command.as_deref(), Some("npx"));
        assert_eq!(loaded.mcp_servers[1].args, vec!["-y", "some-mcp"]);
        assert_eq!(loaded.mcp_servers[0].label(), "preset:chrome-devtools");
    }

    #[test]
    fn roundtrip_multi_agent_flag() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tui.toml");
        TuiConfig {
            multi_agent: true,
            ..Default::default()
        }
        .save_to(&path)
        .unwrap();
        assert!(TuiConfig::load_from(&path).multi_agent);
        // default-false is omitted from the file
        let path2 = dir.path().join("t2.toml");
        TuiConfig::default().save_to(&path2).unwrap();
        assert!(
            !std::fs::read_to_string(&path2)
                .unwrap()
                .contains("multi_agent")
        );
    }

    #[test]
    fn missing_file_loads_default() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = TuiConfig::load_from(&dir.path().join("nope.toml"));
        assert_eq!(loaded, TuiConfig::default());
        assert!(loaded.openrouter_api_key.is_none());
    }

    #[test]
    fn malformed_file_loads_default_not_panics() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "this is not = valid = toml [[[").unwrap();
        assert_eq!(TuiConfig::load_from(&path), TuiConfig::default());
    }

    #[test]
    fn none_fields_are_omitted_from_serialized_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tui.toml");
        TuiConfig {
            openrouter_api_key: Some("k".into()),
            model: None,
            ..Default::default()
        }
        .save_to(&path)
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("openrouter_api_key"));
        assert!(
            !text.contains("model"),
            "None model must be omitted: {text}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn saved_file_is_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tui.toml");
        TuiConfig {
            openrouter_api_key: Some("secret".into()),
            model: None,
            ..Default::default()
        }
        .save_to(&path)
        .unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "config holding a secret must be owner-only");
        // No leftover temp file.
        assert!(!path.with_extension("tmp").exists());
    }

    #[test]
    fn config_path_honors_env_override() {
        // Safe to mutate env in this isolated check.
        unsafe { std::env::set_var("HEARTBIT_TUI_CONFIG", "/tmp/explicit/tui.toml") };
        assert_eq!(config_path(), PathBuf::from("/tmp/explicit/tui.toml"));
        unsafe { std::env::remove_var("HEARTBIT_TUI_CONFIG") };
    }
}
