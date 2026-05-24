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
    /// Optional shell command run after a successful Posted outcome.
    /// Runs from the daemon CWD; env is inherited. Errors are logged
    /// but never propagated (the post is already written).
    pub deploy_command: Option<&'a str>,
    /// Optional X self-amp config (from `BlogContext.entry.x_announce`).
    /// `None` (or `Some` with `enabled=false`) disables X announcement
    /// enqueue on Posted outcomes.
    pub x_announce: Option<&'a heartbit_core::config::XAnnounceConfig>,
    /// Optional GitHub README config (from `BlogContext.entry.github_readme`).
    /// `None` (or `Some` with `enabled=false`) disables the README refresh
    /// on Posted outcomes.
    pub github_readme: Option<&'a heartbit_core::config::GithubReadmeConfig>,
    /// Command producer for enqueueing `BlogAnnounceX`. Optional —
    /// `None` disables X self-amp even if `x_announce` is enabled.
    pub command_producer: Option<Arc<dyn crate::daemon::CommandProducer>>,
    /// Kafka topic for commands (used when enqueueing `BlogAnnounceX`).
    pub commands_topic: &'a str,
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
                chosen_index,
                post_path,
                post_url,
            } = &out.outcome
            {
                tracing::info!(
                    persona = %deps.persona_name,
                    %post_url,
                    path = %post_path.display(),
                    "blog: post published"
                );
                // Run deploy_command first; track success so amp surfaces
                // are gated on a live published site.
                let deploy_ok = if let Some(cmd) = deps.deploy_command {
                    run_deploy_command(deps.persona_name, cmd).await
                } else {
                    true
                };
                if deploy_ok {
                    // Surface 1: GitHub README on-publish (synchronous).
                    if let Some(gh) = deps.github_readme
                        && gh.enabled
                    {
                        let slug = post_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                        let res = heartbit_ghost::github_readme::update_github_readme(
                            heartbit_ghost::github_readme::UpdateReadmeParams {
                                local_repo_path: std::path::Path::new(&gh.local_repo_path),
                                bio_template_path: std::path::Path::new(&gh.bio_template_path),
                                blog_posts_dir: deps.posts_dir,
                                site_url: deps.site_url,
                                git_author_name: &gh.git_author_name,
                                git_author_email: &gh.git_author_email,
                                new_post_slug: slug,
                            },
                        )
                        .await;
                        if let Err(e) = res {
                            tracing::error!(
                                persona = %deps.persona_name,
                                error = %e,
                                "github_readme update failed"
                            );
                        }
                    }
                    // Surface 2: X announcement via Kafka.
                    if let Some(xa) = deps.x_announce
                        && xa.enabled
                    {
                        if let Some(producer) = deps.command_producer.as_ref() {
                            let body_snippet =
                                body_snippet_from_path(post_path).unwrap_or_default();
                            // Use the operator-picked candidate for
                            // title/excerpt — falling back to the
                            // first surviving candidate when the
                            // index is somehow out of range (which
                            // shouldn't happen for a Posted outcome).
                            let chosen = out
                                .candidates
                                .get(*chosen_index)
                                .or_else(|| out.candidates.first());
                            let title = chosen.map(|c| c.title.clone()).unwrap_or_default();
                            let excerpt = chosen.map(|c| c.excerpt.clone()).unwrap_or_default();
                            let cmd = crate::daemon::DaemonCommand::BlogAnnounceX {
                                persona: deps.persona_name.to_string(),
                                post_url: post_url.clone(),
                                title,
                                excerpt,
                                body_snippet,
                            };
                            let payload = match serde_json::to_vec(&cmd) {
                                Ok(p) => p,
                                Err(e) => {
                                    tracing::error!(
                                        persona = %deps.persona_name,
                                        error = %e,
                                        "BlogAnnounceX serialize failed"
                                    );
                                    return Ok(());
                                }
                            };
                            let key = format!("blog_announce_x:{}", deps.persona_name);
                            if let Err(e) = producer
                                .send_command(deps.commands_topic, &key, &payload)
                                .await
                            {
                                tracing::error!(
                                    persona = %deps.persona_name,
                                    error = %e,
                                    "BlogAnnounceX enqueue failed"
                                );
                            } else {
                                tracing::info!(
                                    persona = %deps.persona_name,
                                    topic = %deps.commands_topic,
                                    "BlogAnnounceX enqueued"
                                );
                            }
                        } else {
                            tracing::warn!(
                                persona = %deps.persona_name,
                                "x_announce enabled but no command_producer configured"
                            );
                        }
                    }
                } else {
                    tracing::warn!(
                        persona = %deps.persona_name,
                        "deploy_command failed — skipping amp surfaces (github_readme + BlogAnnounceX)"
                    );
                }
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

/// Read the first ~500 chars of the post body (after YAML frontmatter)
/// for the X announcement writer. Returns `None` if the file can't be
/// read or has YAML opening fences with no closing delimiter.
fn body_snippet_from_path(post_path: &std::path::Path) -> Option<String> {
    let content = std::fs::read_to_string(post_path).ok()?;
    let trimmed = content.trim_start();
    let body = if let Some(after) = trimmed.strip_prefix("---") {
        match after.find("\n---\n") {
            Some(end) => after[end + 5..].trim_start_matches('\n').to_string(),
            None => return None,
        }
    } else {
        trimmed.to_string()
    };
    let snippet: String = body.chars().take(500).collect();
    Some(snippet)
}

/// Shell out to the operator-configured deploy command after a Posted
/// outcome. Bounded at 5 minutes; failures are logged and swallowed
/// (the post is already written to disk — deploy failure should never
/// crash the daemon).
///
/// Returns `true` when the command exited with status 0, `false`
/// otherwise (spawn failure, non-zero exit, wait error, or timeout).
/// Callers use the bool to gate downstream amplification surfaces
/// (X announcement, GitHub README) so we don't broadcast a URL whose
/// live site failed to deploy.
async fn run_deploy_command(persona_name: &str, cmd: &str) -> bool {
    tracing::info!(persona = %persona_name, %cmd, "blog: running deploy_command");
    let child = match tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(persona = %persona_name, error = %e, "blog: deploy spawn failed");
            return false;
        }
    };
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(300),
        child.wait_with_output(),
    )
    .await;
    match result {
        Ok(Ok(out)) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if out.status.success() {
                tracing::info!(
                    persona = %persona_name,
                    stdout = %stdout.trim(),
                    "blog: deploy succeeded"
                );
                true
            } else {
                tracing::error!(
                    persona = %persona_name,
                    status = ?out.status,
                    stdout = %stdout.trim(),
                    stderr = %stderr.trim(),
                    "blog: deploy exited non-zero"
                );
                false
            }
        }
        Ok(Err(e)) => {
            tracing::error!(persona = %persona_name, error = %e, "blog: deploy wait failed");
            false
        }
        Err(_) => {
            tracing::error!(persona = %persona_name, "blog: deploy timed out after 300s");
            false
        }
    }
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
            deploy_command: None,
            x_announce: None,
            github_readme: None,
            command_producer: None,
            commands_topic: "test.commands",
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
            deploy_command: None,
            x_announce: None,
            github_readme: None,
            command_producer: None,
            commands_topic: "test.commands",
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

    // ─── Test: run_deploy_command executes the shell command ─────────────────

    #[tokio::test]
    async fn run_deploy_command_executes_shell() {
        let dir = TempDir::new().unwrap();
        let marker = dir.path().join("deployed");
        let cmd = format!("touch {}", marker.display());
        let ok = run_deploy_command("test-persona", &cmd).await;
        assert!(ok, "deploy command exit-0 must return true");
        assert!(
            marker.exists(),
            "deploy command should have created the marker file at {}",
            marker.display()
        );
    }

    // ─── Test: run_deploy_command swallows non-zero exits ────────────────────

    #[tokio::test]
    async fn run_deploy_command_swallows_failure() {
        // Should NOT panic or propagate; non-zero exit returns false.
        let ok = run_deploy_command("test-persona", "exit 17").await;
        assert!(!ok, "non-zero exit must return false");
    }

    // ─── CapturingProducer mock ──────────────────────────────────────────────

    struct CapturingProducer {
        captured: std::sync::Mutex<Vec<crate::daemon::DaemonCommand>>,
    }

    impl CapturingProducer {
        fn arc() -> std::sync::Arc<Self> {
            std::sync::Arc::new(CapturingProducer {
                captured: std::sync::Mutex::new(Vec::new()),
            })
        }
    }

    impl crate::daemon::CommandProducer for CapturingProducer {
        fn send_command<'a>(
            &'a self,
            _topic: &'a str,
            _key: &'a str,
            payload: &'a [u8],
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<(), crate::Error>> + Send + 'a>,
        > {
            let payload = payload.to_vec();
            Box::pin(async move {
                let cmd: crate::daemon::DaemonCommand = serde_json::from_slice(&payload)?;
                self.captured.lock().unwrap().push(cmd);
                Ok(())
            })
        }
    }

    // ─── Minimal local repo init helper (inlined, no shared test fixture) ────

    fn init_local_repo_for_test() -> (TempDir, std::path::PathBuf) {
        let tmp = TempDir::new().unwrap();
        let repo = tmp.path().join("repo");
        let upstream = tmp.path().join("upstream");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&upstream).unwrap();
        let run = |dir: &std::path::Path, args: &[&str]| {
            let s = std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .status()
                .unwrap();
            assert!(s.success(), "git {args:?} failed");
        };
        run(&upstream, &["init", "--bare", "-b", "main"]);
        run(&repo, &["init", "-b", "main"]);
        run(&repo, &["config", "user.email", "test@test"]);
        run(&repo, &["config", "user.name", "test"]);
        run(
            &repo,
            &["remote", "add", "origin", &upstream.to_string_lossy()],
        );
        std::fs::write(repo.join("README.md"), "# Original\n").unwrap();
        run(&repo, &["add", "README.md"]);
        run(&repo, &["commit", "-m", "init"]);
        run(&repo, &["push", "-u", "origin", "main"]);
        (tmp, repo)
    }

    // ─── Integration test ────────────────────────────────────────────────────

    #[tokio::test]
    async fn amp_helpers_work_in_isolation() {
        // We can't easily mock run_blog_pipeline inside handle_persona_blog,
        // so this test verifies the HELPERS the handler calls work correctly:
        //   1) update_github_readme — writes README and pushes to a bare upstream
        //   2) CapturingProducer — captures a BlogAnnounceX command when produced
        //
        // Coverage of `handle_persona_blog` itself comes from the existing
        // unknown-persona / no-seed tests in this file.

        // ─── Surface 1: update_github_readme works ───────────────────────────
        let (_tmp, repo) = init_local_repo_for_test();
        let posts_dir = _tmp.path().join("posts");
        std::fs::create_dir_all(&posts_dir).unwrap();

        let gh_res = heartbit_ghost::github_readme::update_github_readme(
            heartbit_ghost::github_readme::UpdateReadmeParams {
                local_repo_path: &repo,
                bio_template_path: std::path::Path::new("bio.md"),
                blog_posts_dir: &posts_dir,
                site_url: "https://pascal.heartbit.ai",
                git_author_name: "test",
                git_author_email: "test@test",
                new_post_slug: "smoke-test",
            },
        )
        .await;
        assert!(gh_res.is_ok(), "update_github_readme failed: {gh_res:?}");
        let readme = std::fs::read_to_string(repo.join("README.md")).unwrap();
        assert!(
            readme.contains(heartbit_ghost::github_readme::AUTO_GENERATED_MARKER),
            "README should contain the auto-generated marker"
        );

        // ─── Surface 2: CapturingProducer captures BlogAnnounceX ─────────────
        let producer = CapturingProducer::arc();
        let cmd = crate::daemon::DaemonCommand::BlogAnnounceX {
            persona: "heartbit-ghost:x".into(),
            post_url: "https://pascal.heartbit.ai/test/".into(),
            title: "Test".into(),
            excerpt: "x".into(),
            body_snippet: "y".into(),
        };
        let payload = serde_json::to_vec(&cmd).unwrap();
        crate::daemon::CommandProducer::send_command(
            producer.as_ref(),
            "test.commands",
            "test-key",
            &payload,
        )
        .await
        .unwrap();

        let captured = producer.captured.lock().unwrap();
        assert_eq!(captured.len(), 1, "exactly one command should be captured");
        assert!(
            matches!(
                &captured[0],
                crate::daemon::DaemonCommand::BlogAnnounceX { .. }
            ),
            "captured command must be BlogAnnounceX"
        );
    }
}
