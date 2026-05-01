use serde::Deserialize;

use super::SensorModality;
use super::agent::McpServerEntry;

/// Sensor layer configuration for continuous perception.
#[derive(Debug, Clone, Deserialize)]
pub struct SensorConfig {
    /// Master switch for the sensor layer. Defaults to `true`.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
    /// Model routing configuration for triage decisions.
    #[serde(default)]
    pub routing: Option<SensorRoutingConfig>,
    /// Salience scoring weights for triage promotion.
    #[serde(default)]
    pub salience: Option<SalienceConfig>,
    /// Token budget limits for the sensor pipeline.
    #[serde(default)]
    pub token_budget: Option<TokenBudgetConfig>,
    /// Story correlation settings.
    #[serde(default)]
    pub stories: Option<StoryCorrelationConfig>,
    /// Sensor source definitions.
    #[serde(default)]
    pub sources: Vec<SensorSourceConfig>,
}

/// Model routing configuration for sensor triage.
#[derive(Debug, Clone, Deserialize)]
pub struct SensorRoutingConfig {
    /// Which model tier to use for triage: "local", "cloud_light", "cloud_frontier".
    #[serde(default = "default_triage_model")]
    pub triage_model: String,
    /// Path to local GGUF model file (for local SLM inference).
    pub local_model_path: Option<String>,
    /// Confidence threshold below which to escalate to a higher model tier.
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
}

fn default_triage_model() -> String {
    "cloud_light".into()
}

fn default_confidence_threshold() -> f64 {
    0.85
}

/// Salience scoring weights for triage promotion decisions.
#[derive(Debug, Clone, Deserialize)]
pub struct SalienceConfig {
    /// Weight for urgency signals (0.0-1.0).
    #[serde(default = "default_urgency_weight")]
    pub urgency_weight: f64,
    /// Weight for novelty signals (0.0-1.0).
    #[serde(default = "default_novelty_weight")]
    pub novelty_weight: f64,
    /// Weight for relevance signals (0.0-1.0).
    #[serde(default = "default_relevance_weight")]
    pub relevance_weight: f64,
    /// Minimum salience score for promotion (0.0-1.0).
    #[serde(default = "default_salience_threshold")]
    pub threshold: f64,
}

fn default_urgency_weight() -> f64 {
    0.3
}

fn default_novelty_weight() -> f64 {
    0.3
}

fn default_relevance_weight() -> f64 {
    0.4
}

fn default_salience_threshold() -> f64 {
    0.3
}

/// Token budget limits for the sensor pipeline.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenBudgetConfig {
    /// Maximum tokens per hour across all sensor processing.
    #[serde(default = "default_hourly_limit")]
    pub hourly_limit: usize,
    /// Maximum queued events before back-pressure.
    #[serde(default = "default_queue_size")]
    pub queue_size: usize,
}

fn default_hourly_limit() -> usize {
    100_000
}

fn default_queue_size() -> usize {
    200
}

/// Story correlation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct StoryCorrelationConfig {
    /// Time window in hours for correlating events into stories.
    #[serde(default = "default_correlation_window_hours")]
    pub correlation_window_hours: u64,
    /// Maximum events tracked per story before archival.
    #[serde(default = "default_max_events_per_story")]
    pub max_events_per_story: usize,
    /// Hours of inactivity after which a story is marked stale.
    #[serde(default = "default_stale_after_hours")]
    pub stale_after_hours: u64,
}

fn default_correlation_window_hours() -> u64 {
    4
}

fn default_max_events_per_story() -> usize {
    50
}

fn default_stale_after_hours() -> u64 {
    24
}

/// A sensor source definition.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SensorSourceConfig {
    /// JMAP email sensor (push/poll).
    JmapEmail {
        name: String,
        server: String,
        username: String,
        /// Environment variable containing the password.
        password_env: String,
        /// Senders that get automatic `Priority::High`.
        #[serde(default)]
        priority_senders: Vec<String>,
        /// Senders whose emails are silently dropped during triage.
        #[serde(default)]
        blocked_senders: Vec<String>,
        #[serde(default = "default_email_poll_interval")]
        poll_interval_seconds: u64,
    },
    /// RSS/Atom feed sensor.
    Rss {
        name: String,
        feeds: Vec<String>,
        #[serde(default)]
        interest_keywords: Vec<String>,
        #[serde(default = "default_rss_poll_interval")]
        poll_interval_seconds: u64,
    },
    /// Directory watcher for images.
    Image {
        name: String,
        watch_directory: String,
        #[serde(default = "default_file_poll_interval")]
        poll_interval_seconds: u64,
    },
    /// Directory watcher for audio files.
    Audio {
        name: String,
        watch_directory: String,
        /// Whisper model size: "tiny", "base", "small", "medium", "large".
        #[serde(default = "default_whisper_model")]
        whisper_model: String,
        /// Known contacts whose voice recordings get priority triage.
        #[serde(default)]
        known_contacts: Vec<String>,
        #[serde(default = "default_file_poll_interval")]
        poll_interval_seconds: u64,
    },
    /// Weather API sensor.
    Weather {
        name: String,
        /// Environment variable containing the API key.
        api_key_env: String,
        locations: Vec<String>,
        #[serde(default = "default_weather_poll_interval")]
        poll_interval_seconds: u64,
        /// When true, only promote weather alerts (not regular readings).
        #[serde(default)]
        alert_only: bool,
    },
    /// Generic webhook receiver.
    Webhook {
        name: String,
        /// URL path for the webhook endpoint (e.g., "/webhooks/github").
        path: String,
        /// Environment variable containing the webhook secret.
        secret_env: Option<String>,
    },
    /// Generic MCP sensor — polls a tool on any MCP server.
    Mcp {
        name: String,
        /// MCP server endpoint (string URL, `{url, auth_header}`, or `{command, args, env}`).
        server: Box<McpServerEntry>,
        /// MCP tool to call each poll cycle.
        tool_name: String,
        /// Arguments passed to the tool (default: `{}`).
        #[serde(default = "default_empty_object")]
        tool_args: serde_json::Value,
        /// Kafka topic to produce events to.
        kafka_topic: String,
        /// Sensory modality of produced events (default: `"text"`).
        #[serde(default = "default_mcp_modality")]
        modality: SensorModality,
        /// Poll interval in seconds (default: 60).
        #[serde(default = "default_mcp_poll_interval")]
        poll_interval_seconds: u64,
        /// JSON field path for item ID (default: `"id"`).
        #[serde(default = "default_id_field")]
        id_field: String,
        /// JSON field for event content (default: entire item as JSON).
        #[serde(default)]
        content_field: Option<String>,
        /// JSON field containing items array in tool result (default: root is array).
        #[serde(default)]
        items_field: Option<String>,
        /// Priority senders for email triage (only when `kafka_topic = "hb.sensor.email"`).
        #[serde(default)]
        priority_senders: Vec<String>,
        /// Blocked senders for email triage.
        #[serde(default)]
        blocked_senders: Vec<String>,
        /// Optional enrichment tool to call for each new item (e.g., `gmail_get_message`).
        /// When set, the sensor calls this tool with the item's ID to fetch detailed
        /// metadata (headers, body, labels) before producing to Kafka.
        #[serde(default)]
        enrich_tool: Option<String>,
        /// Parameter name for the item ID when calling the enrichment tool (default: `"id"`).
        #[serde(default)]
        enrich_id_param: Option<String>,
        /// Dedup TTL in seconds. Seen IDs older than this are evicted. Default: 7 days.
        #[serde(default = "default_dedup_ttl_seconds")]
        dedup_ttl_seconds: u64,
    },
}

fn default_dedup_ttl_seconds() -> u64 {
    7 * 24 * 3600 // 7 days
}

impl SensorSourceConfig {
    /// Get the name of this sensor source.
    pub fn name(&self) -> &str {
        match self {
            SensorSourceConfig::JmapEmail { name, .. }
            | SensorSourceConfig::Rss { name, .. }
            | SensorSourceConfig::Image { name, .. }
            | SensorSourceConfig::Audio { name, .. }
            | SensorSourceConfig::Weather { name, .. }
            | SensorSourceConfig::Webhook { name, .. }
            | SensorSourceConfig::Mcp { name, .. } => name,
        }
    }

    /// Get priority and blocked sender lists for trust resolution.
    ///
    /// Returns `(priority_senders, blocked_senders)`. Only email-type sources
    /// have these lists; other source types return empty slices.
    pub fn sender_lists(&self) -> (&[String], &[String]) {
        match self {
            SensorSourceConfig::JmapEmail {
                priority_senders,
                blocked_senders,
                ..
            }
            | SensorSourceConfig::Mcp {
                priority_senders,
                blocked_senders,
                ..
            } => (priority_senders, blocked_senders),
            _ => (&[], &[]),
        }
    }
}

fn default_email_poll_interval() -> u64 {
    60
}

fn default_rss_poll_interval() -> u64 {
    900
}

fn default_file_poll_interval() -> u64 {
    30
}

fn default_whisper_model() -> String {
    "base".into()
}

fn default_weather_poll_interval() -> u64 {
    1800
}

fn default_mcp_poll_interval() -> u64 {
    60
}

fn default_mcp_modality() -> SensorModality {
    SensorModality::Text
}

fn default_id_field() -> String {
    "id".into()
}

fn default_empty_object() -> serde_json::Value {
    serde_json::json!({})
}
