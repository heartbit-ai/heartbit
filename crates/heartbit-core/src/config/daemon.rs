#![allow(missing_docs)]
use serde::Deserialize;

use crate::Error;

/// Daemon mode configuration for runtime execution.
///
/// When `kafka` is absent, the daemon runs in HTTP-only mode: it serves
/// `/v1/tasks/execute` for cloud-delegated execution but does not consume
/// Kafka commands or produce events. This is the recommended mode for the
/// 3-tier architecture where Kafka lives in the gateway, not the runtime.
///
/// Input-source fields (schedules, sensors, ws, telegram, heartbit_pulse,
/// owner_emails, mcp_server) have been moved to the gateway crate as part
/// of the 3-tier architecture refactoring. The type definitions are kept
/// here so they can be re-used by the gateway.
#[derive(Debug, Clone, Deserialize)]
pub struct DaemonConfig {
    #[serde(default)]
    pub kafka: Option<KafkaConfig>,
    #[serde(default = "default_daemon_bind")]
    pub bind: String,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_tasks: usize,
    /// Prometheus metrics configuration. Metrics are enabled by default.
    #[serde(default)]
    pub metrics: Option<MetricsConfig>,
    /// PostgreSQL URL for durable task persistence. When absent, tasks are
    /// stored in-memory (lost on restart).
    #[serde(default)]
    pub database_url: Option<String>,
    /// HTTP API authentication configuration.
    pub auth: Option<AuthConfig>,
    /// Memory access control configuration.
    #[serde(default)]
    pub memory: DaemonMemoryConfig,
    /// Audit log retention configuration.
    #[serde(default)]
    pub audit: DaemonAuditConfig,
    /// Idempotency-key TTL sweep configuration.
    #[serde(default)]
    pub idempotency: IdempotencyConfig,
    /// Per-persona X/Twitter mention-poll configurations.
    /// Each entry launches a `MentionPollScheduler` that fires
    /// `DaemonCommand::MentionPoll` on its configured interval.
    #[serde(default)]
    pub persona_mentions: Vec<PersonaMentionsConfig>,
    /// Per-persona proactive-posting configuration. One entry per
    /// persona that has proactive posting enabled.
    #[serde(default)]
    pub persona_posts: Vec<PersonaPostsConfig>,
}

/// MCP server configuration for the daemon.
///
/// When present, the daemon exposes an MCP-compatible endpoint at `/mcp`
/// so external MCP clients can discover and call heartbit tools/resources.
#[derive(Debug, Clone, Deserialize)]
pub struct DaemonMcpServerConfig {
    /// Server name reported in the `initialize` response. Defaults to `"heartbit"`.
    #[serde(default = "default_mcp_server_name")]
    pub name: String,
    /// Whether to expose heartbit tools via MCP. Default: `true`.
    #[serde(default = "super::default_true")]
    pub expose_tools: bool,
    /// Whether to expose resources (tasks, memory, knowledge) via MCP. Default: `true`.
    #[serde(default = "super::default_true")]
    pub expose_resources: bool,
    /// Whether to expose prompts via MCP. Default: `false`.
    #[serde(default)]
    pub expose_prompts: bool,
}

fn default_mcp_server_name() -> String {
    "heartbit".into()
}

/// Audit log retention configuration for the daemon.
///
/// Controls automatic pruning of old audit log entries.
/// When `retain_days` is set and a Postgres store is attached, the daemon
/// spawns a background task that deletes entries older than `retain_days`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DaemonAuditConfig {
    /// Number of days to retain audit log entries. Entries older than this
    /// are deleted by the background prune task. `None` disables pruning.
    #[serde(default)]
    pub retain_days: Option<u32>,
    /// Interval in minutes between prune runs. Defaults to 60 (hourly).
    #[serde(default)]
    pub prune_interval_minutes: Option<u64>,
}

/// Idempotency-key sweep settings.
///
/// When `ttl_hours` is `Some`, the daemon runs a background task that nulls
/// out idempotency keys older than the TTL. The row itself is retained so
/// existing primary-key lookups still work; only the dedup contract expires.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct IdempotencyConfig {
    /// Hours to retain idempotency keys before the sweep nulls them out.
    /// Default `None` disables the sweep.
    #[serde(default)]
    pub ttl_hours: Option<u32>,
    /// How often the sweep runs, in minutes. Default 60.
    #[serde(default)]
    pub sweep_interval_minutes: Option<u32>,
}

/// Per-persona X/Twitter mention-poll configuration.
///
/// Each entry in `[[daemon.persona_mentions]]` configures one persona's
/// periodic mention polling loop. When `enabled = true`, the daemon spawns
/// a `MentionPollScheduler` for each entry that fires a
/// `DaemonCommand::MentionPoll` on the configured interval.
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaMentionsConfig {
    /// Persona slug (must match a loaded persona file, e.g. `"heartbit-ghost"`).
    pub persona: String,
    /// Enable mention polling for this persona. Defaults to `true`.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
    /// Seconds between mention polls. Defaults to 300 (5 minutes, X API rate-limit-safe).
    #[serde(default = "default_poll_interval_seconds")]
    pub poll_interval_seconds: u64,
    /// X/Twitter user-id to poll mentions for (e.g. `"100"`).
    pub user_id: String,
    /// Maximum candidates to generate replies for per poll cycle. Defaults to 5.
    #[serde(default = "default_candidates_per_reply")]
    pub candidates_per_reply: usize,
    /// Which mention store backend to use: `"in_memory"` or `"jsonl"`.
    /// Defaults to `"in_memory"`.
    #[serde(default = "default_mention_store")]
    pub mention_store: String,
    /// File path for the JSONL mention store (only used when
    /// `mention_store = "jsonl"`). When absent, a per-persona default path is
    /// derived from the persona slug.
    #[serde(default)]
    pub mention_store_path: Option<String>,

    // ── P1.7 loop-protection guards ─────────────────────────────────────────
    /// Enable the thread-depth guard (skips mentions whose parent tweet was
    /// already replied to by this persona). Defaults to `true`.
    #[serde(default = "super::default_true")]
    pub enable_thread_depth_guard: bool,
    /// Enable the bot-heuristic guard. Defaults to `true`.
    #[serde(default = "super::default_true")]
    pub enable_bot_heuristic_guard: bool,
    /// Extra substrings that flag a handle as suspicious (supplements
    /// built-in defaults when non-empty). Defaults to `[]` (use built-ins).
    #[serde(default)]
    pub suspicious_handle_patterns: Vec<String>,
    /// Minimum follower/following ratio; below this is one bot signal.
    /// Defaults to 0.05.
    #[serde(default = "default_min_follower_following_ratio")]
    pub min_follower_following_ratio: f32,
    /// Minimum account age in days; younger accounts trigger one bot signal.
    /// Defaults to 7.
    #[serde(default = "default_min_account_age_days")]
    pub min_account_age_days: i64,
    /// Number of bot signals required to skip a mention. Defaults to 2.
    #[serde(default = "default_bot_heuristic_threshold")]
    pub bot_heuristic_threshold: usize,
    /// Maximum replies per unique conversation. Defaults to 2.
    #[serde(default = "default_per_conversation_max_replies")]
    pub per_conversation_max_replies: usize,
    /// Daily LLM token budget per persona. `null` (default) disables the cap.
    #[serde(default)]
    pub daily_token_budget: Option<u64>,
    /// Budget tracker backend: `"in_memory"` or `"jsonl"`. Defaults to
    /// `"in_memory"`.
    #[serde(default = "default_p1_7_budget_store")]
    pub budget_store: String,
    /// File path for the JSONL budget store (only used when
    /// `budget_store = "jsonl"`). Required when `budget_store = "jsonl"`.
    #[serde(default)]
    pub budget_path: Option<String>,
}

fn default_poll_interval_seconds() -> u64 {
    300
}

fn default_candidates_per_reply() -> usize {
    2
}

fn default_mention_store() -> String {
    "in_memory".into()
}

fn default_min_follower_following_ratio() -> f32 {
    0.05
}

fn default_min_account_age_days() -> i64 {
    7
}

fn default_bot_heuristic_threshold() -> usize {
    2
}

fn default_per_conversation_max_replies() -> usize {
    2
}

fn default_p1_7_budget_store() -> String {
    "in_memory".into()
}

/// Memory access control configuration for the daemon.
///
/// Controls which users can write to shared institutional memory.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DaemonMemoryConfig {
    /// Roles that are allowed to write to shared institutional memory.
    ///
    /// When empty (the default), all users can write — backward compatible.
    /// When non-empty, only users with at least one of these roles can write.
    ///
    /// Example: `["admin", "knowledge_manager"]`
    #[serde(default)]
    pub shared_write_roles: Vec<String>,
}

/// HTTP API authentication configuration for the daemon.
#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    /// Bearer tokens that grant API access. Multiple tokens support key rotation.
    #[serde(default)]
    pub bearer_tokens: Vec<String>,
    /// JWKS endpoint URL for JWT signature verification
    /// (e.g. `"https://idp.example.com/.well-known/jwks.json"`).
    pub jwks_url: Option<String>,
    /// Expected JWT issuer (`iss` claim). Validated when present.
    pub issuer: Option<String>,
    /// Expected JWT audience (`aud` claim). Validated when present.
    pub audience: Option<String>,
    /// JWT claim to extract user ID from. Defaults to `"sub"`.
    pub user_id_claim: Option<String>,
    /// JWT claim to extract tenant ID from. Defaults to `"tid"`.
    pub tenant_id_claim: Option<String>,
    /// JWT claim to extract roles from. Defaults to `"roles"`.
    pub roles_claim: Option<String>,
    /// RFC 8693 Token Exchange configuration for per-user MCP auth delegation.
    /// When configured, the daemon exchanges user JWTs for MCP-scoped delegated tokens.
    pub token_exchange: Option<TokenExchangeConfig>,
}

/// RFC 8693 Token Exchange configuration for per-user MCP auth delegation.
///
/// When configured, each task submitted with a JWT gets a user-scoped delegated
/// token injected into MCP requests. The daemon acts as the agent (actor) and
/// exchanges the user's subject token for a scoped access token.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenExchangeConfig {
    /// Token exchange endpoint URL (e.g. `"https://idp.example.com/oauth/token"`).
    pub exchange_url: String,
    /// OAuth client ID for the daemon/agent.
    pub client_id: String,
    /// OAuth client secret for the daemon/agent.
    pub client_secret: String,
    /// NHI tenant ID — used for `X-Tenant-ID` header in `client_credentials` grant.
    /// When set, `agent_token` is fetched and cached automatically; no static token needed.
    pub tenant_id: Option<String>,
    /// Static fallback agent token (`actor_token` in RFC 8693).
    /// Used only when `tenant_id` is absent (backward-compat).
    #[serde(default)]
    pub agent_token: String,
    /// OAuth scopes to request for the delegated token. Defaults to empty.
    #[serde(default)]
    pub scopes: Vec<String>,
}

/// Heartbit pulse configuration for autonomous periodic awareness.
///
/// When enabled, the daemon periodically reviews its persistent todo list
/// and decides what to work on next — a cognitive pulse loop.
#[derive(Debug, Clone, Deserialize)]
pub struct HeartbitPulseConfig {
    /// Enable the heartbit pulse. Defaults to `false`.
    #[serde(default)]
    pub enabled: bool,
    /// Interval in seconds between heartbit pulse ticks. Defaults to 1800 (30 min).
    #[serde(default = "default_pulse_interval")]
    pub interval_seconds: u64,
    /// Active hours window. When set, the pulse only fires within this window.
    pub active_hours: Option<ActiveHoursConfig>,
    /// Custom prompt override for the heartbit pulse. When absent, the
    /// default built-in prompt is used.
    pub prompt: Option<String>,
    /// Number of consecutive HEARTBIT_OK responses before doubling the
    /// interval (idle backoff). Defaults to 6 (3h at 30min interval).
    #[serde(default = "default_idle_backoff_threshold")]
    pub idle_backoff_threshold: u32,
}

fn default_pulse_interval() -> u64 {
    1800
}

fn default_idle_backoff_threshold() -> u32 {
    6
}

/// Active hours window for the heartbit pulse.
#[derive(Debug, Clone, Deserialize)]
pub struct ActiveHoursConfig {
    /// Start time in "HH:MM" format (24-hour).
    pub start: String,
    /// End time in "HH:MM" format (24-hour).
    pub end: String,
}

impl ActiveHoursConfig {
    /// Parse the start hour and minute. Returns `(hour, minute)`.
    pub fn parse_start(&self) -> Result<(u32, u32), Error> {
        parse_hhmm(&self.start)
    }

    /// Parse the end hour and minute. Returns `(hour, minute)`.
    pub fn parse_end(&self) -> Result<(u32, u32), Error> {
        parse_hhmm(&self.end)
    }
}

fn parse_hhmm(s: &str) -> Result<(u32, u32), Error> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(Error::Config(format!(
            "invalid time format '{}': expected HH:MM",
            s
        )));
    }
    let hour: u32 = parts[0]
        .parse()
        .map_err(|_| Error::Config(format!("invalid hour in '{s}'")))?;
    let minute: u32 = parts[1]
        .parse()
        .map_err(|_| Error::Config(format!("invalid minute in '{s}'")))?;
    if hour > 23 || minute > 59 {
        return Err(Error::Config(format!(
            "time '{}' out of range (00:00-23:59)",
            s
        )));
    }
    Ok((hour, minute))
}

/// Per-persona proactive-posting configuration.
///
/// When present, the daemon registers a `PersonaPostScheduler` that
/// fires `DaemonCommand::PersonaPost` on the configured cadence
/// (gated by `active_hours`). The handler runs the topic generator,
/// drafts candidate threads via the existing review pipeline, sends
/// them to Telegram for review, posts the chosen draft.
///
/// Configured under `[[daemon.persona_posts]]` blocks.
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaPostsConfig {
    /// Persona registry name (e.g. `"heartbit-ghost:x"`).
    pub persona: String,
    /// Whether this poster is enabled.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
    /// Posting interval, in seconds. Default 14400 (4 hours).
    /// Validation: must be ≥60 (rejected at config load otherwise).
    #[serde(default = "default_post_interval_seconds")]
    pub post_interval_seconds: u64,
    /// Randomness applied to each `post_interval_seconds` tick, as a
    /// percentage (`0`–`50`). Default `25` = ±25%, so a base 4h interval
    /// fires somewhere in [3h, 5h]. Prevents the daemon from posting
    /// on a perfectly predictable cadence — that's a textbook bot
    /// signature on X. `0` disables jitter (use only for tests).
    #[serde(default = "default_post_interval_jitter_pct")]
    pub interval_jitter_pct: u32,
    /// Optional `"HH:MM-HH:MM"` window during which posts are allowed.
    /// Outside this window, the scheduler tick is a no-op. When absent,
    /// posting is allowed 24/7.
    #[serde(default)]
    pub active_hours: Option<ActiveHoursConfig>,
    /// Number of candidate threads to draft per tick (1..=10).
    /// Default 3.
    #[serde(default = "default_post_candidates")]
    pub candidates_per_draft: usize,
    /// Backend for the post history store: `"in_memory"` or `"jsonl"`.
    #[serde(default = "default_post_history_store")]
    pub post_history_store: String,
    /// Path to the JSONL store file (only used when
    /// `post_history_store == "jsonl"`). Tilde expansion happens at
    /// store-construction time.
    #[serde(default)]
    pub post_history_path: Option<String>,
    /// How far back to check for topic duplicates. Default 30 days.
    #[serde(default = "default_post_history_lookback_days")]
    pub post_history_lookback_days: i64,
    /// Optional fallback brief used when the persona declares no
    /// topic-context provider, or appended to the provider's output.
    #[serde(default)]
    pub topic_brief: Option<String>,

    // --- Engagement feedback (P2.0) ---
    /// Engagement-collector tick interval, in seconds. Default 21600 = 6h.
    /// Each tick batch-refreshes every eligible Posted tweet's metrics
    /// from the X API and writes a fresh [`EngagementSnapshot`].
    #[serde(default = "default_engagement_refresh_seconds")]
    pub engagement_refresh_seconds: u64,
    /// How many top-engaged posts to inject as writer few-shot exemplars.
    /// Default 5. Set to `0` to disable injection — the writer then runs
    /// without exemplars, matching pre-P2.0 behavior.
    #[serde(default = "default_engagement_top_n")]
    pub engagement_top_n: usize,
    /// Ignore tweets older than this many days when refreshing engagement.
    /// Default 30 — older tweets rarely accrue new engagement.
    #[serde(default = "default_engagement_max_age_days")]
    pub engagement_max_age_days: i64,
    /// Ignore tweets younger than this many hours when refreshing
    /// engagement. Default 24 — gives the algorithm time to fan out.
    #[serde(default = "default_engagement_min_age_hours")]
    pub engagement_min_age_hours: i64,
}

fn default_post_interval_seconds() -> u64 {
    14400
}

fn default_post_interval_jitter_pct() -> u32 {
    25
}

fn default_post_candidates() -> usize {
    3
}

fn default_post_history_store() -> String {
    "in_memory".into()
}

fn default_post_history_lookback_days() -> i64 {
    30
}

fn default_engagement_refresh_seconds() -> u64 {
    21600
}

fn default_engagement_top_n() -> usize {
    5
}

fn default_engagement_max_age_days() -> i64 {
    30
}

fn default_engagement_min_age_hours() -> i64 {
    24
}

/// WebSocket configuration for bidirectional user↔agent communication.
#[derive(Debug, Clone, Deserialize)]
pub struct WsConfig {
    /// Whether WebSocket endpoint is enabled. Defaults to `true`.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
    /// Timeout in seconds for blocking interactions (approval, input, question).
    /// Defaults to 120 seconds.
    #[serde(default = "default_interaction_timeout")]
    pub interaction_timeout_seconds: u64,
    /// Maximum concurrent WebSocket connections. Defaults to 100.
    #[serde(default = "default_max_ws_connections")]
    pub max_connections: usize,
    /// PostgreSQL URL for durable session persistence. When absent, sessions
    /// are stored in-memory (lost on restart).
    #[serde(default)]
    pub database_url: Option<String>,
}

fn default_interaction_timeout() -> u64 {
    120
}

fn default_max_ws_connections() -> usize {
    100
}

/// Kafka broker connection settings.
#[derive(Debug, Clone, Deserialize)]
pub struct KafkaConfig {
    pub brokers: String,
    #[serde(default = "default_consumer_group")]
    pub consumer_group: String,
    #[serde(default = "default_commands_topic")]
    pub commands_topic: String,
    #[serde(default = "default_events_topic")]
    pub events_topic: String,
    /// Topic for events that failed triage processing.
    #[serde(default = "default_dead_letter_topic")]
    pub dead_letter_topic: String,
}

/// A scheduled task entry for the cron scheduler.
#[derive(Debug, Clone, Deserialize)]
pub struct ScheduleEntry {
    pub name: String,
    pub cron: String,
    pub task: String,
    #[serde(default = "super::default_true")]
    pub enabled: bool,
}

/// Prometheus metrics configuration for daemon mode.
#[derive(Debug, Clone, Deserialize)]
pub struct MetricsConfig {
    /// Whether Prometheus metrics are enabled. Defaults to `true`.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
}

fn default_daemon_bind() -> String {
    "127.0.0.1:3000".into()
}

fn default_max_concurrent() -> usize {
    4
}

fn default_consumer_group() -> String {
    "heartbit-daemon".into()
}

fn default_commands_topic() -> String {
    "heartbit.commands".into()
}

fn default_events_topic() -> String {
    "heartbit.events".into()
}

fn default_dead_letter_topic() -> String {
    "heartbit.dead-letter".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persona_posts_config_parses_with_defaults() {
        let toml = r#"
[[persona_posts]]
persona = "heartbit-ghost:x"
"#;
        #[derive(Deserialize)]
        struct Shim {
            persona_posts: Vec<PersonaPostsConfig>,
        }
        let cfg: Shim = toml::from_str(toml).unwrap();
        assert_eq!(cfg.persona_posts.len(), 1);
        let p = &cfg.persona_posts[0];
        assert_eq!(p.persona, "heartbit-ghost:x");
        assert!(p.enabled);
        assert_eq!(p.post_interval_seconds, 14400);
        assert_eq!(p.candidates_per_draft, 3);
        assert_eq!(p.post_history_store, "in_memory");
        assert_eq!(p.post_history_lookback_days, 30);
        assert!(p.active_hours.is_none());
        assert!(p.topic_brief.is_none());
        assert!(p.post_history_path.is_none());
        // Engagement defaults (P2.0).
        assert_eq!(p.engagement_refresh_seconds, 21600);
        assert_eq!(p.engagement_top_n, 5);
        assert_eq!(p.engagement_max_age_days, 30);
        assert_eq!(p.engagement_min_age_hours, 24);
    }

    #[test]
    fn persona_posts_config_parses_engagement_overrides() {
        let toml = r#"
[[persona_posts]]
persona = "heartbit-ghost:x"
engagement_refresh_seconds = 3600
engagement_top_n = 0
engagement_max_age_days = 7
engagement_min_age_hours = 6
"#;
        #[derive(Deserialize)]
        struct Shim {
            persona_posts: Vec<PersonaPostsConfig>,
        }
        let cfg: Shim = toml::from_str(toml).unwrap();
        let p = &cfg.persona_posts[0];
        assert_eq!(p.engagement_refresh_seconds, 3600);
        assert_eq!(p.engagement_top_n, 0);
        assert_eq!(p.engagement_max_age_days, 7);
        assert_eq!(p.engagement_min_age_hours, 6);
    }

    #[test]
    fn persona_posts_config_parses_full() {
        let toml = r#"
[[persona_posts]]
persona = "heartbit-ghost:x"
enabled = true
post_interval_seconds = 7200
candidates_per_draft = 5
post_history_store = "jsonl"
post_history_path = "~/.heartbit/posts.jsonl"
post_history_lookback_days = 14
topic_brief = "agents, Rust, LLMs"

[persona_posts.active_hours]
start = "09:00"
end = "22:00"
"#;
        #[derive(Deserialize)]
        struct Shim {
            persona_posts: Vec<PersonaPostsConfig>,
        }
        let cfg: Shim = toml::from_str(toml).unwrap();
        let p = &cfg.persona_posts[0];
        assert_eq!(p.persona, "heartbit-ghost:x");
        assert!(p.enabled);
        assert_eq!(p.post_interval_seconds, 7200);
        assert_eq!(p.candidates_per_draft, 5);
        assert_eq!(p.post_history_store, "jsonl");
        assert_eq!(
            p.post_history_path.as_deref(),
            Some("~/.heartbit/posts.jsonl")
        );
        assert_eq!(p.post_history_lookback_days, 14);
        assert_eq!(p.topic_brief.as_deref(), Some("agents, Rust, LLMs"));
        assert!(p.active_hours.is_some());
        let h = p.active_hours.as_ref().unwrap();
        assert_eq!(h.start, "09:00");
        assert_eq!(h.end, "22:00");
    }
}
