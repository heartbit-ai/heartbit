use serde::{Deserialize, Serialize};

/// DM access policy for the Telegram bot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DmPolicy {
    /// Only users in `allowed_users` may interact (default).
    #[default]
    Allowlist,
    /// Any user may send DMs.
    Open,
    /// Telegram integration is disabled (bot won't respond to DMs).
    Disabled,
}

fn default_true() -> bool {
    true
}

fn default_inactivity_timeout() -> u64 {
    1800 // 30 minutes
}

fn default_session_expiry() -> u64 {
    86400 // 24 hours
}

fn default_interaction_timeout() -> u64 {
    120 // 2 minutes
}

fn default_max_concurrent() -> usize {
    50
}

fn default_stream_debounce_ms() -> u64 {
    500
}

fn default_memory_recall_limit() -> usize {
    5
}

fn default_institutional_recall_limit() -> usize {
    3
}

/// Configuration for the Telegram bot channel.
#[derive(Debug, Clone, Deserialize)]
pub struct TelegramConfig {
    /// Whether the Telegram bot is enabled. Defaults to `true`.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Bot token. Falls back to `HEARTBIT_TELEGRAM_TOKEN` env var when absent.
    #[serde(default)]
    pub token: Option<String>,

    /// DM access policy. Defaults to `Allowlist`.
    #[serde(default)]
    pub dm_policy: DmPolicy,

    /// User IDs permitted to interact when `dm_policy` is `Allowlist`.
    #[serde(default)]
    pub allowed_users: Vec<i64>,

    /// Seconds of inactivity before a session is considered idle.
    /// Triggers memory consolidation. Defaults to 1800 (30 min).
    #[serde(default = "default_inactivity_timeout")]
    pub inactivity_timeout_seconds: u64,

    /// Seconds before an idle session expires and is deleted.
    /// Memory persists independently. Defaults to 86400 (24h).
    #[serde(default = "default_session_expiry")]
    pub session_expiry_seconds: u64,

    /// Timeout in seconds for blocking interactions (approval, input, question).
    /// Defaults to 120 seconds.
    #[serde(default = "default_interaction_timeout")]
    pub interaction_timeout_seconds: u64,

    /// Maximum number of concurrent Telegram chat sessions. Defaults to 50.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    /// Debounce interval in milliseconds for streaming text edits. Defaults to 500.
    #[serde(default = "default_stream_debounce_ms")]
    pub stream_debounce_ms: u64,

    /// PostgreSQL URL for session persistence. Falls back to `daemon.database_url`.
    #[serde(default)]
    pub database_url: Option<String>,

    /// Maximum number of memory entries to recall per message. Defaults to 5.
    #[serde(default = "default_memory_recall_limit")]
    pub memory_recall_limit: usize,

    /// Maximum number of institutional memory entries to recall per message.
    /// Institutional memories are shared knowledge from daemon task results.
    /// Defaults to 3.
    #[serde(default = "default_institutional_recall_limit")]
    pub institutional_recall_limit: usize,

    /// When true, skip memory recall for very short trivial messages
    /// (< 20 chars, no question mark). Saves tokens on greetings. Defaults to false.
    #[serde(default)]
    pub memory_skip_trivial: bool,

    /// Chat IDs to receive proactive notifications for background tasks
    /// (sensor, cron, heartbeat pulse). Empty = no notifications.
    #[serde(default)]
    pub notify_chat_ids: Vec<i64>,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            token: None,
            dm_policy: DmPolicy::default(),
            allowed_users: Vec::new(),
            inactivity_timeout_seconds: default_inactivity_timeout(),
            session_expiry_seconds: default_session_expiry(),
            interaction_timeout_seconds: default_interaction_timeout(),
            max_concurrent: default_max_concurrent(),
            stream_debounce_ms: default_stream_debounce_ms(),
            database_url: None,
            memory_recall_limit: default_memory_recall_limit(),
            institutional_recall_limit: default_institutional_recall_limit(),
            memory_skip_trivial: false,
            notify_chat_ids: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dm_policy_default_is_allowlist() {
        assert_eq!(DmPolicy::default(), DmPolicy::Allowlist);
    }

    #[test]
    fn dm_policy_serde_roundtrip() {
        for (variant, expected_json) in [
            (DmPolicy::Allowlist, "\"allowlist\""),
            (DmPolicy::Open, "\"open\""),
            (DmPolicy::Disabled, "\"disabled\""),
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, expected_json);
            let deserialized: DmPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, variant);
        }
    }

    #[test]
    fn telegram_config_defaults() {
        let config = TelegramConfig::default();
        assert!(config.enabled);
        assert!(config.token.is_none());
        assert_eq!(config.dm_policy, DmPolicy::Allowlist);
        assert!(config.allowed_users.is_empty());
        assert_eq!(config.inactivity_timeout_seconds, 1800);
        assert_eq!(config.session_expiry_seconds, 86400);
        assert_eq!(config.interaction_timeout_seconds, 120);
        assert_eq!(config.max_concurrent, 50);
        assert_eq!(config.stream_debounce_ms, 500);
        assert!(config.database_url.is_none());
        assert_eq!(config.memory_recall_limit, 5);
        assert_eq!(config.institutional_recall_limit, 3);
        assert!(!config.memory_skip_trivial);
        assert!(config.notify_chat_ids.is_empty());
    }

    #[test]
    fn telegram_config_toml_roundtrip() {
        let toml_str = r#"
enabled = true
token = "123:ABC"
dm_policy = "open"
allowed_users = [111, 222]
inactivity_timeout_seconds = 600
session_expiry_seconds = 3600
interaction_timeout_seconds = 60
max_concurrent = 10
stream_debounce_ms = 250
database_url = "postgres://localhost/heartbit"
memory_recall_limit = 3
institutional_recall_limit = 5
memory_skip_trivial = true
notify_chat_ids = [111, 222]
"#;
        let config: TelegramConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert_eq!(config.token.as_deref(), Some("123:ABC"));
        assert_eq!(config.dm_policy, DmPolicy::Open);
        assert_eq!(config.allowed_users, vec![111, 222]);
        assert_eq!(config.inactivity_timeout_seconds, 600);
        assert_eq!(config.session_expiry_seconds, 3600);
        assert_eq!(config.interaction_timeout_seconds, 60);
        assert_eq!(config.max_concurrent, 10);
        assert_eq!(config.stream_debounce_ms, 250);
        assert_eq!(
            config.database_url.as_deref(),
            Some("postgres://localhost/heartbit")
        );
        assert_eq!(config.memory_recall_limit, 3);
        assert_eq!(config.institutional_recall_limit, 5);
        assert!(config.memory_skip_trivial);
        assert_eq!(config.notify_chat_ids, vec![111, 222]);
    }

    #[test]
    fn telegram_config_minimal_toml() {
        let toml_str = "";
        let config: TelegramConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert!(config.token.is_none());
        assert_eq!(config.dm_policy, DmPolicy::Allowlist);
        assert!(config.allowed_users.is_empty());
        assert_eq!(config.inactivity_timeout_seconds, 1800);
    }

    #[test]
    fn telegram_config_disabled_policy() {
        let toml_str = r#"
dm_policy = "disabled"
"#;
        let config: TelegramConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.dm_policy, DmPolicy::Disabled);
    }

    #[test]
    fn dm_policy_invalid_value() {
        let result: Result<DmPolicy, _> = serde_json::from_str("\"invalid\"");
        assert!(result.is_err());
    }

    #[test]
    fn notify_chat_ids_default_empty() {
        let config = TelegramConfig::default();
        assert!(config.notify_chat_ids.is_empty());
    }

    #[test]
    fn notify_chat_ids_toml_roundtrip() {
        let toml_str = r#"
notify_chat_ids = [111, 222, 333]
"#;
        let config: TelegramConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.notify_chat_ids, vec![111, 222, 333]);
    }

    #[test]
    fn notify_chat_ids_backward_compat() {
        // Old TOML without notify_chat_ids should deserialize with empty vec
        let toml_str = r#"
enabled = true
dm_policy = "open"
"#;
        let config: TelegramConfig = toml::from_str(toml_str).unwrap();
        assert!(config.notify_chat_ids.is_empty());
    }

    #[test]
    fn institutional_recall_limit_default() {
        let config = TelegramConfig::default();
        assert_eq!(config.institutional_recall_limit, 3);
    }

    #[test]
    fn institutional_recall_limit_from_toml() {
        let toml_str = r#"
institutional_recall_limit = 5
"#;
        let config: TelegramConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.institutional_recall_limit, 5);
    }

    #[test]
    fn institutional_recall_limit_backward_compat() {
        // Old TOML without institutional_recall_limit should default to 3
        let toml_str = r#"
enabled = true
memory_recall_limit = 10
"#;
        let config: TelegramConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.institutional_recall_limit, 3);
    }
}
