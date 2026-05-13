//! Daemon-wide shared state for the quote-tweet pipeline. Constructed
//! once at startup by `heartbit-cli` from `[[daemon.persona_quotes]]`
//! and shared via `Arc` across handler invocations.
//!
//! Mirrors [`crate::daemon::posts_context::PostsContext`] closely.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::config::ActiveHoursConfig;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::PersonaRegistry;

use heartbit_ghost::quote::sources::{QuoteSeenStore, QuoteSource};

/// One persona's quote-tweet runtime config.
pub struct PersonaQuoteEntry {
    /// Source-tweet fetcher (`XUserTimelineSource` in prod; mock in tests).
    pub source: Arc<dyn QuoteSource>,
    /// Already-quoted dedup store (in-memory or JSONL).
    pub seen_store: Arc<dyn QuoteSeenStore>,
    /// Polling interval (used by the scheduler at startup).
    pub interval: Duration,
    /// `±jitter_pct%` randomization applied to each scheduler tick.
    /// 0 = deterministic clock (use only for tests). 25 = ±25% (default
    /// when loaded from config).
    pub interval_jitter_pct: u32,
    /// Optional active-hours window.
    pub active_hours: Option<ActiveHoursConfig>,
    /// Curated source X user IDs (numeric strings) to poll.
    pub source_user_ids: Vec<String>,
    /// Number of candidate quote-tweets to draft per chosen source tweet.
    pub candidates_per_draft: usize,
    /// Maximum age (hours) of a source tweet for it to be quote-able.
    /// `0` disables the age filter.
    pub max_age_hours: i64,
    /// Maximum number of source tweets to draft+review per scheduler tick.
    pub max_candidates_per_tick: usize,
    /// Optional override LLM provider for `quote_writer` + style critic.
    /// `None` falls back to `QuotesContext.provider` for those stages.
    pub writer_provider: Option<Arc<BoxedProvider>>,
}

impl std::fmt::Debug for PersonaQuoteEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaQuoteEntry")
            .field("interval", &self.interval)
            .field("interval_jitter_pct", &self.interval_jitter_pct)
            .field("active_hours_set", &self.active_hours.is_some())
            .field("source_user_ids", &self.source_user_ids)
            .field("candidates_per_draft", &self.candidates_per_draft)
            .field("max_age_hours", &self.max_age_hours)
            .field("max_candidates_per_tick", &self.max_candidates_per_tick)
            .field("writer_provider_set", &self.writer_provider.is_some())
            .finish()
    }
}

/// Daemon-wide context for the quote-tweet pipeline. Constructed once
/// at startup and shared via `Arc` across handler invocations.
pub struct QuotesContext {
    /// Persona registry (for `expand()` calls).
    pub registry: Arc<PersonaRegistry>,
    /// LLM provider for sub-agents (researcher + fact_check by default;
    /// writer + critic when `PersonaQuoteEntry.writer_provider` is None).
    pub provider: Arc<BoxedProvider>,
    /// Telegram (or mock) quote-review delivery.
    pub delivery: Arc<dyn heartbit_ghost::quote::QuoteReviewDelivery>,
    /// `twitter_quote` tool — used by `run_quote_pipeline` to post.
    pub twitter_quote_tool: Arc<dyn Tool>,
    /// Credential resolver shared across all quote handlers.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Root directory containing per-persona corpora.
    pub corpora_root: PathBuf,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: PathBuf,
    /// Per-persona configuration. Keyed by persona name.
    pub entries: HashMap<String, PersonaQuoteEntry>,
}

impl std::fmt::Debug for QuotesContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotesContext")
            .field("personas", &self.entries.keys().collect::<Vec<_>>())
            .field("corpora_root", &self.corpora_root)
            .field("profiles_root", &self.profiles_root)
            .finish()
    }
}
