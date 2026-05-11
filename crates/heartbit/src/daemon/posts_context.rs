//! Bundle of per-persona dependencies for the proactive-posts pipeline.
//! Constructed by the CLI at daemon startup, injected into [`DaemonCore`]
//! via [`DaemonCore::with_posts_context`], and consumed by the dispatcher
//! arm for [`crate::daemon::types::DaemonCommand::PersonaPost`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::config::ActiveHoursConfig;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::PersonaRegistry;

use heartbit_ghost::posts::{EngagementStore, PostHistoryStore};
use heartbit_ghost::review::ReviewDelivery;

/// Per-persona state for one `[[daemon.persona_posts]]` entry.
pub struct PersonaPostEntry {
    /// Post history store (de-dup + outcome recording).
    pub history: Arc<dyn PostHistoryStore>,
    /// Polling interval (used by the scheduler at startup).
    pub interval: Duration,
    /// `±jitter_pct%` randomization applied to each scheduler tick.
    /// 0 = deterministic clock (use only for tests). 25 = ±25% (default
    /// when loaded from config).
    pub interval_jitter_pct: u32,
    /// Optional active-hours window.
    pub active_hours: Option<ActiveHoursConfig>,
    /// Number of candidate threads per tick.
    pub candidates_per_draft: usize,
    /// Lookback for the duplicate check.
    pub history_lookback: chrono::Duration,
    /// Optional fallback brief.
    pub topic_brief: Option<String>,
    /// Operator's X user_id.
    pub operator_user_id: String,

    // --- Engagement-feedback wiring (Task 3 of the engagement loop) ---
    /// Engagement collector interval (default 6h).
    pub engagement_refresh: Duration,
    /// `±jitter_pct%` applied to the engagement collector cadence.
    /// Clamped to `0..=50` by the scheduler.
    pub engagement_jitter_pct: u32,
    /// Engagement snapshot store (JSONL or in-memory).
    pub engagement_store: Arc<dyn EngagementStore>,
    /// Skip tweets younger than this when refreshing. Default 24h.
    pub engagement_min_age_hours: i64,
    /// Skip tweets older than this when refreshing. Default 30d.
    pub engagement_max_age_days: i64,
    /// How many top-engaged posts to inject as few-shot exemplars when
    /// the writer agent runs. Default 5. `0` disables injection.
    /// Wired into the post pipeline in Task 5.
    pub engagement_top_n: usize,
}

impl std::fmt::Debug for PersonaPostEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaPostEntry")
            .field("interval", &self.interval)
            .field("active_hours_set", &self.active_hours.is_some())
            .field("candidates_per_draft", &self.candidates_per_draft)
            .field("history_lookback", &self.history_lookback)
            .field("topic_brief_set", &self.topic_brief.is_some())
            .field("operator_user_id", &self.operator_user_id)
            .field("engagement_refresh", &self.engagement_refresh)
            .field("engagement_jitter_pct", &self.engagement_jitter_pct)
            // `engagement_store` is always present (non-Option), so a *_set
            // boolean would always be true and carry no signal. Skip it.
            .field("engagement_min_age_hours", &self.engagement_min_age_hours)
            .field("engagement_max_age_days", &self.engagement_max_age_days)
            .field("engagement_top_n", &self.engagement_top_n)
            .finish()
    }
}

/// Daemon-wide context for the proactive-posts pipeline. Constructed
/// once at startup and shared via `Arc` across handler invocations.
pub struct PostsContext {
    /// Persona registry (for `expand()` calls).
    pub registry: Arc<PersonaRegistry>,
    /// LLM provider for sub-agents.
    pub provider: Arc<BoxedProvider>,
    /// Telegram (or mock) review delivery.
    pub delivery: Arc<dyn ReviewDelivery>,
    /// `twitter_thread` tool — used by `run_review_pipeline` to post.
    pub twitter_thread: Arc<dyn Tool>,
    /// Credentials resolver shared across all post handlers.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Root directory containing per-persona corpora.
    pub corpora_root: PathBuf,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: PathBuf,
    /// Per-persona configuration. Keyed by persona name.
    pub entries: HashMap<String, PersonaPostEntry>,
}

impl std::fmt::Debug for PostsContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostsContext")
            .field("personas", &self.entries.keys().collect::<Vec<_>>())
            .field("corpora_root", &self.corpora_root)
            .field("profiles_root", &self.profiles_root)
            .finish()
    }
}
