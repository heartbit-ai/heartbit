//! Handler for `DaemonCommand::PersonaBlog`. Selects the X-derived
//! seed, runs `run_blog_pipeline`, handles outcomes.
//!
//! Mirrors [`crate::daemon::persona_quote_handler::handle_persona_quote`]
//! in shape (deps struct + free function), with the trigger being the
//! highest-engagement post in the configured lookback window rather
//! than a curated source tweet.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use heartbit_core::CredentialResolver;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::{PersonaParams, PersonaRegistry};
use heartbit_ghost::blog::{
    BlogConfig, BlogOutcome, BlogReviewDelivery, DEFAULT_TOP_N, run_blog_pipeline, select_blog_seed,
};
use heartbit_ghost::posts::TopPostsProvider;

/// Inputs to one blog handler tick. Borrows everything the daemon
/// already owns; the handler doesn't hold state across calls.
pub struct PersonaBlogDeps<'a> {
    /// Persona name to load from the registry.
    pub persona_name: &'a str,
    /// Persona registry (loads persona by name).
    pub registry: &'a PersonaRegistry,
    /// Top-posts provider (engagement-ranked X posts).
    pub top_posts_provider: &'a dyn TopPostsProvider,
    /// How far back to look for the seed (days).
    pub seed_lookback_days: i64,
    /// Default LLM provider.
    pub provider: Arc<BoxedProvider>,
    /// Optional writer-stage provider override.
    pub writer_provider: Option<Arc<BoxedProvider>>,
    /// Telegram delivery.
    pub delivery: Arc<dyn BlogReviewDelivery>,
    /// X credentials (currently unused for blog).
    pub credentials: Arc<dyn CredentialResolver>,
    /// Number of candidate essays per tick.
    pub candidates_per_draft: usize,
    /// Corpora root.
    pub corpora_root: &'a Path,
    /// Voice profiles root.
    pub profiles_root: &'a Path,
    /// Where to write `*.md` posts.
    pub posts_dir: &'a Path,
    /// Where to render HTML output.
    pub out_dir: &'a Path,
    /// Path to `style.css`.
    pub style_css: &'a Path,
    /// Public site URL.
    pub site_url: &'a str,
    /// Site title.
    pub site_title: &'a str,
}

/// Run one blog pipeline tick. Selects the seed via
/// [`select_blog_seed`]; on `NoEligibleSeed` returns `Ok(())` after
/// logging (no error — that's a normal "nothing to write this week"
/// outcome). On any other path, delegates to [`run_blog_pipeline`].
/// Never panics.
pub async fn handle_persona_blog(deps: PersonaBlogDeps<'_>) -> Result<()> {
    let persona = deps
        .registry
        .get(deps.persona_name)
        .ok_or_else(|| anyhow::anyhow!("persona '{}' not registered", deps.persona_name))?;
    let _expansion = persona
        .expand(&PersonaParams::default())
        .map_err(|e| anyhow::anyhow!("expand persona '{}': {e}", deps.persona_name))?;

    let seed = match select_blog_seed(
        deps.top_posts_provider,
        DEFAULT_TOP_N,
        deps.seed_lookback_days,
        Utc::now(),
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            tracing::info!(
                persona = %deps.persona_name,
                error = %e,
                "blog: no eligible seed this week — skipping tick"
            );
            return Ok(());
        }
    };

    tracing::info!(
        persona = %deps.persona_name,
        seed_tweet_id = %seed.source_tweet_id,
        seed_score = seed.engagement_score,
        "blog: selected seed"
    );

    let cfg = BlogConfig {
        persona_name: deps.persona_name,
        provider: deps.provider.clone(),
        writer_provider: deps.writer_provider.clone(),
        corpora_root: deps.corpora_root,
        profiles_root: deps.profiles_root,
        on_progress: Some(Arc::new(|s: &str| tracing::info!("blog: {s}"))),
        seed,
        candidates_per_draft: deps.candidates_per_draft,
        delivery: deps.delivery.clone(),
        credentials: deps.credentials.clone(),
        posts_dir: deps.posts_dir,
        out_dir: deps.out_dir,
        style_css: deps.style_css,
        site_url: deps.site_url,
        site_title: deps.site_title,
    };

    match run_blog_pipeline(cfg).await {
        Ok(out) => {
            tracing::info!(
                persona = %deps.persona_name,
                outcome = ?out.outcome,
                "blog pipeline complete"
            );
            if let BlogOutcome::Posted {
                post_path,
                post_url,
                ..
            } = &out.outcome
            {
                tracing::info!(
                    persona = %deps.persona_name,
                    %post_url,
                    path = %post_path.display(),
                    "blog: post published"
                );
            }
        }
        Err(e) => {
            tracing::error!(
                persona = %deps.persona_name,
                error = %e,
                "blog pipeline failed"
            );
        }
    }
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::{CredentialResolver as CredentialResolverTrait, Secret};
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use heartbit_core::persona::{Persona, PersonaExpansion, PersonaParams, PersonaRegistry};

    use heartbit_ghost::blog::{BlogOutcome, BlogReviewDelivery, BlogReviewMessage};
    use heartbit_ghost::posts::{TopPost, TopPostsFut, TopPostsProvider};
    use heartbit_ghost::review::{
        DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReviewDeliveryError,
    };

    use tempfile::TempDir;

    // ─── StubTestPersona ─────────────────────────────────────────────────────

    struct StubTestPersona {
        name: String,
    }

    impl Persona for StubTestPersona {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "stub persona for tests"
        }
        fn version(&self) -> &str {
            "0.0.1"
        }
        fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, CoreError> {
            Ok(PersonaExpansion::default())
        }
    }

    // ─── MockTopPostsProvider ────────────────────────────────────────────────

    struct MockTopPostsProvider {
        posts: Vec<TopPost>,
        calls: AtomicUsize,
    }

    impl MockTopPostsProvider {
        fn arc(posts: Vec<TopPost>) -> Arc<MockTopPostsProvider> {
            Arc::new(Self {
                posts,
                calls: AtomicUsize::new(0),
            })
        }
    }

    impl TopPostsProvider for MockTopPostsProvider {
        fn top_n<'a>(&'a self, n: usize) -> TopPostsFut<'a> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let posts: Vec<TopPost> = self.posts.iter().take(n).cloned().collect();
            Box::pin(async move { Ok(posts) })
        }
    }

    // ─── MockBlogReviewDelivery ──────────────────────────────────────────────
    //
    // Counts deliver_and_await calls so we can assert pipeline invocation
    // status from the handler tests without exercising the full pipeline.

    struct MockBlogReviewDelivery {
        deliver_calls: Mutex<usize>,
    }

    impl MockBlogReviewDelivery {
        fn arc() -> Arc<MockBlogReviewDelivery> {
            Arc::new(Self {
                deliver_calls: Mutex::new(0),
            })
        }

        fn deliver_calls(&self) -> usize {
            *self.deliver_calls.lock().unwrap()
        }
    }

    impl BlogReviewDelivery for MockBlogReviewDelivery {
        fn deliver_and_await<'a>(
            &'a self,
            _msg: &'a BlogReviewMessage,
        ) -> Pin<Box<dyn Future<Output = Result<DeliveredReview, ReviewDeliveryError>> + Send + 'a>>
        {
            *self.deliver_calls.lock().unwrap() += 1;
            Box::pin(async move {
                Ok(DeliveredReview {
                    outcome: DeliveryOutcome::Skip,
                    receipt: DeliveryReceipt {
                        data: serde_json::Value::Null,
                    },
                })
            })
        }

        fn report<'a>(
            &'a self,
            _receipt: DeliveryReceipt,
            _outcome: BlogOutcome,
        ) -> Pin<Box<dyn Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    // ─── MockProvider ────────────────────────────────────────────────────────
    //
    // Required for happy-path-ish coverage if we ever add one; here we
    // construct a no-op provider just to give the deps a value when the
    // pipeline is *not* expected to run.

    struct MockProvider;

    impl MockProvider {
        fn arc() -> Arc<BoxedProvider> {
            Arc::new(BoxedProvider::new(MockProvider))
        }
    }

    impl LlmProvider for MockProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, CoreError> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "noop".to_string(),
                }],
                usage: TokenUsage::default(),
                stop_reason: StopReason::EndTurn,
                model: None,
            })
        }
    }

    // ─── StubCredentialResolver ──────────────────────────────────────────────

    struct StubCredentialResolver;

    impl CredentialResolverTrait for StubCredentialResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(Secret::new("stub")) })
        }
    }

    // ─── Test harness fixtures ───────────────────────────────────────────────

    fn tmp_paths() -> (
        TempDir,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
    ) {
        let dir = TempDir::new().unwrap();
        let posts_dir = dir.path().join("posts");
        let out_dir = dir.path().join("public");
        let style_css = dir.path().join("style.css");
        std::fs::write(&style_css, "body{}").unwrap();
        (dir, posts_dir, out_dir, style_css)
    }

    // ─── Test: unknown persona returns Err ───────────────────────────────────

    #[tokio::test]
    async fn handle_persona_blog_unknown_persona_errors() {
        let registry = PersonaRegistry::new();
        let top = MockTopPostsProvider::arc(vec![]);
        let delivery = MockBlogReviewDelivery::arc();
        let provider = MockProvider::arc();
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);
        let (_dir, posts_dir, out_dir, style_css) = tmp_paths();
        let profiles_root = _dir.path().to_path_buf();

        let deps = PersonaBlogDeps {
            persona_name: "missing-persona",
            registry: &registry,
            top_posts_provider: top.as_ref(),
            seed_lookback_days: 7,
            provider,
            writer_provider: None,
            delivery,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            posts_dir: &posts_dir,
            out_dir: &out_dir,
            style_css: &style_css,
            site_url: "https://pascal.heartbit.ai",
            site_title: "pascal.heartbit.ai",
        };

        let err = handle_persona_blog(deps)
            .await
            .expect_err("expected error for unknown persona");
        assert!(err.to_string().contains("not registered"), "got: {err}");
        // Pipeline-side delivery must never be invoked.
        // (Note: we can't downcast Arc<dyn _>, so assert calls via the
        // concrete Arc held above.)
    }

    // ─── Test: no eligible seed returns Ok and skips pipeline ────────────────

    #[tokio::test]
    async fn handle_persona_blog_no_seed_returns_ok_and_skips_pipeline() {
        let mut registry = PersonaRegistry::new();
        registry.register(Arc::new(StubTestPersona { name: "x".into() }));
        // Empty top-posts list → SeedError::NoEligibleSeed → handler
        // returns Ok and never invokes the pipeline (so delivery stays
        // at zero calls).
        let top = MockTopPostsProvider::arc(vec![]);
        let delivery = MockBlogReviewDelivery::arc();
        let delivery_for_assertion = delivery.clone();
        let provider = MockProvider::arc();
        let credentials: Arc<dyn CredentialResolver> = Arc::new(StubCredentialResolver);
        let (_dir, posts_dir, out_dir, style_css) = tmp_paths();
        let profiles_root = _dir.path().to_path_buf();

        let deps = PersonaBlogDeps {
            persona_name: "x",
            registry: &registry,
            top_posts_provider: top.as_ref(),
            seed_lookback_days: 7,
            provider,
            writer_provider: None,
            delivery: delivery as Arc<dyn BlogReviewDelivery>,
            credentials,
            candidates_per_draft: 1,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            posts_dir: &posts_dir,
            out_dir: &out_dir,
            style_css: &style_css,
            site_url: "https://pascal.heartbit.ai",
            site_title: "pascal.heartbit.ai",
        };

        handle_persona_blog(deps)
            .await
            .expect("no-seed path must be Ok");
        // top_n was queried exactly once during select_blog_seed.
        assert_eq!(
            top.calls.load(Ordering::SeqCst),
            1,
            "top_posts_provider should be called once for seed selection"
        );
        // Pipeline never ran → delivery.deliver_and_await never invoked.
        assert_eq!(
            delivery_for_assertion.deliver_calls(),
            0,
            "delivery must NOT be invoked when seed selection fails"
        );
    }
}
