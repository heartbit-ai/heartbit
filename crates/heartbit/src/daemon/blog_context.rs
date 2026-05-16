//! Daemon-wide shared state for the blog pipeline. Constructed once
//! at startup by `heartbit-cli` from `[daemon.persona_blog]` and shared
//! via `Arc` across handler invocations.
//!
//! Mirrors [`crate::daemon::quotes_context::QuotesContext`] in shape,
//! with a single entry rather than a per-persona map — the blog
//! pipeline is intentionally scoped to one persona per daemon (one
//! site per operator).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use heartbit_core::CredentialResolver;
use heartbit_core::config::ActiveHoursConfig;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::PersonaRegistry;
use heartbit_ghost::blog::BlogReviewDelivery;
use heartbit_ghost::posts::TopPostsProvider;

/// Per-persona blog scheduler/runtime configuration.
pub struct PersonaBlogEntry {
    /// Provider used by the scheduler to pick this week's seed post.
    pub top_posts_provider: Arc<dyn TopPostsProvider>,
    /// Time between scheduled blog ticks (weekly by default).
    pub interval: Duration,
    /// Random jitter applied to the interval (percent of interval).
    pub interval_jitter_pct: u32,
    /// Optional posting-window restriction.
    pub active_hours: Option<ActiveHoursConfig>,
    /// How far back to look for the seed post.
    pub seed_lookback_days: i64,
    /// Number of candidate essays per pipeline tick.
    pub candidates_per_draft: usize,
    /// Where to write the source-of-truth `*.md` files.
    pub posts_dir: PathBuf,
    /// Output directory for rendered HTML.
    pub out_dir: PathBuf,
    /// `style.css` location.
    pub style_css: PathBuf,
    /// Public site URL (used for canonical/sitemap/RSS).
    pub site_url: String,
    /// Site title (`<title>` and index header).
    pub site_title: String,
    /// Optional override provider for the writer stage.
    pub writer_provider: Option<Arc<BoxedProvider>>,
}

impl std::fmt::Debug for PersonaBlogEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaBlogEntry")
            .field("interval", &self.interval)
            .field("interval_jitter_pct", &self.interval_jitter_pct)
            .field("active_hours_set", &self.active_hours.is_some())
            .field("seed_lookback_days", &self.seed_lookback_days)
            .field("candidates_per_draft", &self.candidates_per_draft)
            .field("posts_dir", &self.posts_dir)
            .field("out_dir", &self.out_dir)
            .field("style_css", &self.style_css)
            .field("site_url", &self.site_url)
            .field("site_title", &self.site_title)
            .field("writer_provider_set", &self.writer_provider.is_some())
            .finish()
    }
}

/// Daemon-wide shared state for the blog pipeline. Constructed once
/// at startup and shared via `Arc` across handler invocations.
pub struct BlogContext {
    /// Persona registry (looks up persona/expand by name).
    pub registry: Arc<PersonaRegistry>,
    /// Default LLM provider (used by researcher/critic/fact_check; also
    /// the writer fallback).
    pub provider: Arc<BoxedProvider>,
    /// Telegram delivery for operator review.
    pub delivery: Arc<dyn BlogReviewDelivery>,
    /// X API credentials (currently unused by blog but mirrors quote
    /// context shape).
    pub credentials: Arc<dyn CredentialResolver>,
    /// Root containing corpora subdirs per persona.
    pub corpora_root: PathBuf,
    /// Root containing voice profiles per persona.
    pub profiles_root: PathBuf,
    /// Single entry — keyed by persona name for consistency with other
    /// contexts even though there's only one blog per daemon.
    pub entry: PersonaBlogEntry,
    /// The persona name this blog context is scoped to.
    pub persona_name: String,
}

impl std::fmt::Debug for BlogContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlogContext")
            .field("persona_name", &self.persona_name)
            .field("corpora_root", &self.corpora_root)
            .field("profiles_root", &self.profiles_root)
            .field("entry", &self.entry)
            .finish()
    }
}
