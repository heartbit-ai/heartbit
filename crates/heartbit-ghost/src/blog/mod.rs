//! Personal-blog pipeline — picks an X-derived topic seed, drafts a
//! long-form essay via `blog_writer`, routes through Telegram review,
//! commits Markdown to disk + renders the static site.

pub mod announce;
pub mod markdown;
pub mod prompts;
pub mod render;
pub mod seed;
pub mod templates;

pub use markdown::{BlogPostFrontmatter, WriteMarkdownError, write_post_markdown};
pub use render::{RenderError, RenderedPostMeta, render_site};
pub use seed::{BlogSeed, DEFAULT_TOP_N, SeedError, select_blog_seed};

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::llm::types::TokenUsage;
use thiserror::Error;

use crate::pipeline::{FactVerdict, PipelineError, ProgressCallback};

/// Configuration for one blog-pipeline run.
pub struct BlogConfig<'a> {
    /// Persona instance name (used to load the StyleProfile snapshot).
    pub persona_name: &'a str,
    /// LLM provider shared across researcher / critic / fact_check by
    /// default. Also used by the writer when `writer_provider` is `None`.
    pub provider: Arc<BoxedProvider>,
    /// Optional override for the writer stage. When `Some`, the writer
    /// routes through this provider instead of `provider`. Researcher,
    /// critic, and fact_check always use `provider`.
    pub writer_provider: Option<Arc<BoxedProvider>>,
    /// Root directory containing per-persona corpora.
    pub corpora_root: &'a Path,
    /// Root directory containing per-persona style profiles.
    pub profiles_root: &'a Path,
    /// Optional progress callback invoked with a short status string at
    /// each pipeline stage start.
    pub on_progress: Option<ProgressCallback>,
    /// The X-derived topic seed.
    pub seed: BlogSeed,
    /// Number of distinct candidate essays to generate (1..=3).
    pub candidates_per_draft: usize,
    /// Telegram-or-mock delivery layer for the blog review.
    pub delivery: Arc<dyn BlogReviewDelivery>,
    /// Credential resolver — threaded through for parity with the quote
    /// and review pipelines (currently unused since the blog pipeline
    /// writes to disk rather than posting to X).
    pub credentials: Arc<dyn CredentialResolver>,
    /// Directory where the post's Markdown file will be written.
    pub posts_dir: &'a Path,
    /// Output directory the static-site renderer writes to.
    pub out_dir: &'a Path,
    /// Path to the `style.css` source file (copied into `out_dir`).
    pub style_css: &'a Path,
    /// Canonical site URL (e.g. `https://pascal.heartbit.ai`).
    pub site_url: &'a str,
    /// Human-readable site name (rendered into `<title>` and the RSS
    /// channel title).
    pub site_title: &'a str,
}

/// Result of one blog-pipeline tick.
#[derive(Debug, Clone)]
pub struct BlogOutput {
    /// The seed that drove this run.
    pub seed: BlogSeed,
    /// All candidate essays generated this run (surviving the pre-filter
    /// sweep). Empty when the outcome is `NoSeed` or
    /// `AllCandidatesGateRejected`.
    pub candidates: Vec<BlogCandidateRecord>,
    /// Aggregated token usage across researcher + writer + critic +
    /// fact_check, summed over all candidates.
    pub usage_summary: TokenUsage,
    /// Terminal state of this run.
    pub outcome: BlogOutcome,
}

/// One candidate essay record. The title / slug / excerpt are extracted
/// from the draft body up-front so the Telegram review can render a
/// header and the publish step doesn't have to re-parse.
#[derive(Debug, Clone)]
pub struct BlogCandidateRecord {
    /// Full Markdown draft (no frontmatter — the renderer adds it).
    pub draft: String,
    /// Voice-match score from the style critic, 0.0..=1.0.
    pub style_match_score: f32,
    /// Parsed fact-check verdict.
    pub fact_check_verdict: FactVerdict,
    /// Title extracted from the draft (first non-blank line, stripped of
    /// leading `#`).
    pub title: String,
    /// URL slug derived from the title via `slug::slugify`.
    pub slug: String,
    /// First-paragraph excerpt (trimmed to ~160 chars).
    pub excerpt: String,
}

/// Terminal state of a blog-pipeline run.
#[derive(Debug, Clone)]
pub enum BlogOutcome {
    /// Operator picked candidate `chosen_index` — the post was written
    /// to disk and the static site was re-rendered.
    Posted {
        /// 0-based index into the surviving candidates list.
        chosen_index: usize,
        /// Absolute path of the written Markdown file.
        post_path: PathBuf,
        /// Public URL of the rendered post.
        post_url: String,
    },
    /// Operator pressed Skip.
    Skipped,
    /// Telegram review timed out without a pick.
    TimedOut,
    /// Every candidate was dropped by the pre-filter sweep (Unverifiable
    /// fact verdict, or empty title/slug). No Telegram review was sent.
    AllCandidatesGateRejected {
        /// Per-candidate drop reasons.
        reasons: Vec<String>,
    },
    /// No eligible X-derived seed was found — the pipeline did not run.
    /// (Surfaced only when callers gate `run_blog_pipeline` behind a
    /// `select_blog_seed` call that returned `SeedError::NoEligibleSeed`;
    /// `run_blog_pipeline` itself never produces this from a valid `seed`.)
    NoSeed,
}

/// Errors raised by [`run_blog_pipeline`].
#[derive(Debug, Error)]
pub enum BlogError {
    /// Underlying pipeline error (snapshot load, builder, agent execution).
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Telegram delivery error.
    #[error("delivery: {0}")]
    Delivery(#[from] crate::review::ReviewDeliveryError),
    /// Markdown writer failure.
    #[error("markdown: {0}")]
    Markdown(#[from] markdown::WriteMarkdownError),
    /// Static-site renderer failure.
    #[error("render: {0}")]
    Render(#[from] render::RenderError),
    /// `BlogConfig` validation failed at run start.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Telegram-or-mock review delivery for blog-essay messages. Mirrors
/// [`crate::review::ReviewDelivery`] but carries the blog-shaped message
/// type ([`BlogReviewMessage`]) and outcome ([`BlogOutcome`]).
///
/// Methods use the project's `Pin<Box<dyn Future>>` desugaring to stay
/// object-safe without the `async-trait` crate.
pub trait BlogReviewDelivery: Send + Sync {
    /// Deliver a blog review to the user (Telegram bot or mock) and
    /// await their pick (or timeout).
    fn deliver_and_await<'a>(
        &'a self,
        message: &'a BlogReviewMessage,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        crate::review::DeliveredReview,
                        crate::review::ReviewDeliveryError,
                    >,
                > + Send
                + 'a,
        >,
    >;

    /// Report final outcome back to the delivery layer (edits the
    /// original message, etc.). Non-fatal — the runtime logs and
    /// continues on error.
    fn report<'a>(
        &'a self,
        receipt: crate::review::DeliveryReceipt,
        outcome: BlogOutcome,
    ) -> Pin<Box<dyn Future<Output = Result<(), crate::review::ReviewDeliveryError>> + Send + 'a>>;
}

/// Message body for a blog review.
#[derive(Debug, Clone)]
pub struct BlogReviewMessage {
    /// Persona instance name (rendered in the header).
    pub persona_name: String,
    /// Seed text (the X-derived topic).
    pub seed_text: String,
    /// Seed source URL.
    pub seed_url: String,
    /// Full essay Markdown for each surviving candidate. The operator
    /// reads the full thing in Telegram and picks one (or skips).
    pub candidates: Vec<String>,
    /// UUID for keyboard callback correlation.
    pub interaction_id: uuid::Uuid,
}

/// Execute one blog-pipeline run.
///
/// Flow: snapshot load → research → N parallel writer→critic→fact_check
/// → extract title/slug/excerpt → pre-filter (drop Unverifiable + empty
/// title) → BlogReviewDelivery::deliver_and_await → on Pick:
/// `write_post_markdown` + `render_site` → report → return.
pub async fn run_blog_pipeline(cfg: BlogConfig<'_>) -> Result<BlogOutput, BlogError> {
    use crate::agents::{
        blog_writer_recipe, fact_check_recipe, researcher_recipe, style_critic_recipe,
    };
    use heartbit_core::Tool;
    use heartbit_core::tool::builtins::{WebFetchTool, WebSearchTool};

    // 1. Validate.
    if !(1..=3).contains(&cfg.candidates_per_draft) {
        return Err(BlogError::InvalidConfig(format!(
            "candidates_per_draft must be in 1..=3 (got {})",
            cfg.candidates_per_draft,
        )));
    }

    let progress = |s: &str| {
        if let Some(cb) = cfg.on_progress.as_ref() {
            cb(s);
        }
    };

    // 2. Load profile snapshot.
    progress("Loading profile snapshot...");
    let store = crate::voice::SnapshotStore::open(cfg.profiles_root, cfg.persona_name)
        .map_err(PipelineError::from)?;
    let snapshot = store
        .load_latest()
        .map_err(PipelineError::from)?
        .ok_or_else(|| PipelineError::NoProfileSnapshot {
            persona: cfg.persona_name.to_string(),
            profiles_dir: cfg.profiles_root.join(cfg.persona_name),
        })?;
    let profile = snapshot.profile;

    // 3. Build the 4 sub-agent runners.
    //    - researcher + critic + fact_check use `cfg.provider` (canonical).
    //    - writer optionally uses `cfg.writer_provider` (override).
    let writer_provider = cfg
        .writer_provider
        .clone()
        .unwrap_or_else(|| cfg.provider.clone());

    // Researcher always uses the public web — the blog pipeline does NOT
    // currently expose a researcher_override knob (Task 9 scope).
    let researcher_tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(WebSearchTool::new()),
        Arc::new(WebFetchTool::new()),
    ];
    let researcher = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        researcher_recipe(),
        researcher_tools,
    )
    .map_err(|source| PipelineError::Builder {
        stage: "researcher".to_string(),
        source,
    })?;
    let writer = crate::pipeline::runner_from_recipe(
        writer_provider.clone(),
        blog_writer_recipe(),
        Vec::new(),
    )
    .map_err(|source| PipelineError::Builder {
        stage: "blog_writer".to_string(),
        source,
    })?;
    let critic = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        style_critic_recipe(),
        Vec::new(),
    )
    .map_err(|source| PipelineError::Builder {
        stage: "style_critic".to_string(),
        source,
    })?;
    let fact =
        crate::pipeline::runner_from_recipe(cfg.provider.clone(), fact_check_recipe(), Vec::new())
            .map_err(|source| PipelineError::Builder {
                stage: "fact_check".to_string(),
                source,
            })?;

    let mut total_usage = TokenUsage::default();
    let voice_guidelines = crate::pipeline::render_style_profile_as_english(&profile);

    // 4. Run researcher.
    progress("Researching seed topic...");
    let seed_input: prompts::BlogSeedInput<'_> = (&cfg.seed).into();
    let research_msg = prompts::build_blog_research_user_message(&seed_input);
    let researcher_out =
        researcher
            .execute(&research_msg)
            .await
            .map_err(|source| PipelineError::Agent {
                stage: "researcher".to_string(),
                source,
            })?;
    let digest = researcher_out.result.clone();
    total_usage += researcher_out.tokens_used;

    // 5. Generate N candidate essays in parallel via tokio::JoinSet.
    progress(&format!(
        "Generating {} candidate(s) in parallel...",
        cfg.candidates_per_draft
    ));
    let writer = Arc::new(writer);
    let critic = Arc::new(critic);
    let fact = Arc::new(fact);
    let voice_owned: Arc<str> = voice_guidelines.clone().into();
    let digest_owned: Arc<str> = digest.clone().into();
    let seed_owned = Arc::new(cfg.seed.clone());

    type CandidateResult = (String, f32, FactVerdict, TokenUsage);
    let mut joinset: tokio::task::JoinSet<Result<CandidateResult, PipelineError>> =
        tokio::task::JoinSet::new();
    for _ in 0..cfg.candidates_per_draft {
        let writer = writer.clone();
        let critic = critic.clone();
        let fact = fact.clone();
        let voice = voice_owned.clone();
        let digest = digest_owned.clone();
        let seed = seed_owned.clone();
        joinset.spawn(async move {
            let seed_input: prompts::BlogSeedInput<'_> = (&*seed).into();
            let writer_msg = prompts::build_blog_writer_user_message(&digest, &seed_input, &voice);
            let writer_out =
                writer
                    .execute(&writer_msg)
                    .await
                    .map_err(|source| PipelineError::Agent {
                        stage: "blog_writer".to_string(),
                        source,
                    })?;
            // Long-form output preserved verbatim — no length normalize.
            let draft = writer_out.result.trim().to_string();
            // Style critic.
            let critic_msg = prompts::build_blog_critic_user_message(&draft, &voice);
            let critic_out =
                critic
                    .execute(&critic_msg)
                    .await
                    .map_err(|source| PipelineError::Agent {
                        stage: "style_critic".to_string(),
                        source,
                    })?;
            let style_score = parse_style_match_score(&critic_out.result).unwrap_or(0.5);
            // Fact check.
            let fact_msg = prompts::build_blog_fact_user_message(&draft, &digest);
            let fact_out =
                fact.execute(&fact_msg)
                    .await
                    .map_err(|source| PipelineError::Agent {
                        stage: "fact_check".to_string(),
                        source,
                    })?;
            let fact_verdict = crate::pipeline::parse_fact_verdict(&fact_out.result)
                .map_err(|source| PipelineError::FactCheckParseFailed { source })?;
            let mut usage = writer_out.tokens_used;
            usage += critic_out.tokens_used;
            usage += fact_out.tokens_used;
            Ok((draft, style_score, fact_verdict, usage))
        });
    }

    let mut survivors: Vec<BlogCandidateRecord> = Vec::new();
    while let Some(handle) = joinset.join_next().await {
        let (draft, style_score, fact_verdict, usage) =
            handle.map_err(|e| PipelineError::Agent {
                stage: "candidate".to_string(),
                source: heartbit_core::error::Error::Agent(format!("join: {e}")),
            })??;
        total_usage += usage;
        let (title, slug, excerpt) = extract_title_slug_excerpt(&draft);
        survivors.push(BlogCandidateRecord {
            draft,
            style_match_score: style_score,
            fact_check_verdict: fact_verdict,
            title,
            slug,
            excerpt,
        });
    }

    // 6. Pre-filter — drop Unverifiable + empty-title/slug candidates
    //    BEFORE delivery so the operator never sees a draft we can't ship.
    let mut filtered: Vec<BlogCandidateRecord> = Vec::with_capacity(survivors.len());
    let mut drop_reasons: Vec<String> = Vec::new();
    for c in survivors {
        if let FactVerdict::Unverifiable { reason } = &c.fact_check_verdict {
            drop_reasons.push(format!("unverifiable: {reason}"));
            continue;
        }
        if c.title.is_empty() || c.slug.is_empty() {
            drop_reasons.push("empty title/slug".to_string());
            continue;
        }
        filtered.push(c);
    }

    if filtered.is_empty() {
        return Ok(BlogOutput {
            seed: cfg.seed,
            candidates: Vec::new(),
            usage_summary: total_usage,
            outcome: BlogOutcome::AllCandidatesGateRejected {
                reasons: drop_reasons,
            },
        });
    }

    let survivors = filtered;

    // 7. Telegram review delivery.
    progress("Sending review to user...");
    let drafts_for_review: Vec<String> = survivors.iter().map(|c| c.draft.clone()).collect();
    let msg = BlogReviewMessage {
        persona_name: cfg.persona_name.to_string(),
        seed_text: cfg.seed.text.clone(),
        seed_url: cfg.seed.source_url.clone(),
        candidates: drafts_for_review,
        interaction_id: uuid::Uuid::new_v4(),
    };
    let delivered = cfg.delivery.deliver_and_await(&msg).await?;

    // 8. Branch on outcome. TimedOut has a receipt too, but the quote
    //    and review pipelines also call `report()` for skip/timeout —
    //    mirror that pattern so the delivery layer can edit the original
    //    message uniformly.
    let outcome = match delivered.outcome {
        crate::review::DeliveryOutcome::Pick(idx) if idx < survivors.len() => {
            // 9. Write + render.
            let chosen = &survivors[idx];
            progress(&format!("Writing post for candidate {idx}..."));
            let front = BlogPostFrontmatter {
                title: chosen.title.clone(),
                date: chrono::Utc::now(),
                slug: chosen.slug.clone(),
                excerpt: chosen.excerpt.clone(),
                tags: Vec::new(),
            };
            let post_path = write_post_markdown(cfg.posts_dir, &front, &chosen.draft)?;
            progress("Rendering static site...");
            render_site(
                cfg.posts_dir,
                cfg.out_dir,
                &render::RenderConfig {
                    site_url: cfg.site_url,
                    site_title: cfg.site_title,
                    style_css: cfg.style_css,
                },
            )?;
            let post_url = format!("{}/{}/", cfg.site_url.trim_end_matches('/'), chosen.slug);
            BlogOutcome::Posted {
                chosen_index: idx,
                post_path,
                post_url,
            }
        }
        crate::review::DeliveryOutcome::Pick(_) => BlogOutcome::Skipped, // unreachable
        crate::review::DeliveryOutcome::Skip => BlogOutcome::Skipped,
        crate::review::DeliveryOutcome::TimedOut => BlogOutcome::TimedOut,
    };

    // 10. Optional report-back to delivery (non-fatal).
    let _ = cfg
        .delivery
        .report(delivered.receipt, outcome.clone())
        .await;

    Ok(BlogOutput {
        seed: cfg.seed,
        candidates: survivors,
        usage_summary: total_usage,
        outcome,
    })
}

// Helpers — pure functions ----------------------------------------------

/// Parse the `style_critic` JSON output and pull `style_match_score`.
/// Returns `None` when the JSON is malformed or the field is missing.
fn parse_style_match_score(raw: &str) -> Option<f32> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    v.get("style_match_score")?.as_f64().map(|x| x as f32)
}

/// Extract `(title, slug, excerpt)` from a Markdown draft.
///
/// - **title**: first non-blank line, stripped of leading `#` and whitespace.
/// - **slug**: `slug::slugify(title)`.
/// - **excerpt**: first paragraph after the title line, trimmed to ~160 chars.
fn extract_title_slug_excerpt(draft: &str) -> (String, String, String) {
    let title = draft
        .lines()
        .map(|l| l.trim())
        .find(|l| !l.is_empty())
        .map(|l| l.trim_start_matches('#').trim().to_string())
        .unwrap_or_default();
    let slug = slug::slugify(&title);
    let body = if let Some(idx) = draft.find('\n') {
        &draft[idx + 1..]
    } else {
        ""
    };
    let first_para = body
        .split("\n\n")
        .map(|p| p.trim())
        .find(|p| !p.is_empty())
        .unwrap_or("");
    let excerpt: String = first_para.chars().take(160).collect();
    (title, slug, excerpt)
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::Utc;
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::CredentialResolver as CredentialResolverTrait;
    use heartbit_core::execution_context::Secret;
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// MockBlogReviewDelivery — returns a pre-canned outcome. An optional
    /// error mode simulates delivery failure (used to assert delivery is
    /// never called in the all-rejected tests).
    struct MockBlogReviewDelivery {
        outcome: Option<crate::review::DeliveryOutcome>,
        error_msg: Option<String>,
        reports: Mutex<Vec<BlogOutcome>>,
        deliver_calls: Mutex<usize>,
    }

    impl MockBlogReviewDelivery {
        fn arc(outcome: crate::review::DeliveryOutcome) -> Arc<MockBlogReviewDelivery> {
            Arc::new(MockBlogReviewDelivery {
                outcome: Some(outcome),
                error_msg: None,
                reports: Mutex::new(Vec::new()),
                deliver_calls: Mutex::new(0),
            })
        }

        fn errored(reason: &str) -> Arc<MockBlogReviewDelivery> {
            Arc::new(MockBlogReviewDelivery {
                outcome: None,
                error_msg: Some(reason.to_string()),
                reports: Mutex::new(Vec::new()),
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
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::review::DeliveredReview,
                            crate::review::ReviewDeliveryError,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            *self.deliver_calls.lock().unwrap() += 1;
            let outcome = self.outcome.clone();
            let error_msg = self.error_msg.clone();
            Box::pin(async move {
                if let Some(msg) = error_msg {
                    return Err(crate::review::ReviewDeliveryError::Transport(msg));
                }
                Ok(crate::review::DeliveredReview {
                    outcome: outcome.expect("either outcome or error_msg must be set"),
                    receipt: crate::review::DeliveryReceipt {
                        data: serde_json::Value::Null,
                    },
                })
            })
        }

        fn report<'a>(
            &'a self,
            _receipt: crate::review::DeliveryReceipt,
            outcome: BlogOutcome,
        ) -> Pin<Box<dyn Future<Output = Result<(), crate::review::ReviewDeliveryError>> + Send + 'a>>
        {
            self.reports.lock().unwrap().push(outcome);
            Box::pin(async move { Ok(()) })
        }
    }

    /// MockProvider — same shape as `quote/mod.rs::tests::MockProvider`.
    struct MockProvider {
        responses: Mutex<VecDeque<String>>,
    }

    impl MockProvider {
        fn arc(responses: Vec<&str>) -> Arc<BoxedProvider> {
            let p = MockProvider {
                responses: Mutex::new(responses.into_iter().map(String::from).collect()),
            };
            Arc::new(BoxedProvider::new(p))
        }
    }

    impl LlmProvider for MockProvider {
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CoreError>> + Send {
            let response = self.responses.lock().unwrap().pop_front();
            let has_respond = request
                .tools
                .iter()
                .any(|t| t.name == heartbit_core::llm::types::RESPOND_TOOL_NAME);
            async move {
                let text =
                    response.ok_or_else(|| CoreError::Agent("mock exhausted".to_string()))?;
                let content = if has_respond {
                    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                        CoreError::Agent(format!("mock: canned response is not valid JSON: {e}"))
                    })?;
                    vec![ContentBlock::ToolUse {
                        id: "respond_1".to_string(),
                        name: "__respond__".to_string(),
                        input: value,
                    }]
                } else {
                    vec![ContentBlock::Text { text }]
                };
                Ok(CompletionResponse {
                    content,
                    usage: TokenUsage::default(),
                    stop_reason: if has_respond {
                        StopReason::ToolUse
                    } else {
                        StopReason::EndTurn
                    },
                    model: None,
                })
            }
        }
    }

    /// Stub credential resolver — never invoked in blog mock tests.
    struct StubCredentialResolver;

    impl CredentialResolverTrait for StubCredentialResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(Secret::new("stub")) })
        }
    }

    /// Snapshot fixture — same shape as `quote/mod.rs::tests::seed_snapshot`.
    fn seed_snapshot(persona: &str) -> (TempDir, std::path::PathBuf) {
        use crate::voice::{
            BlendEntry, BlendRecipe, EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency,
            HashtagPolicy, LineBreaks, OpeningPattern, PartialStyleProfile, PeriodsPolicy,
            QuotationMarks, SentenceLengthTarget, SnapshotStore, SpecificityTarget, StyleProfile,
            ThreadRhythm,
        };
        let dir = TempDir::new().unwrap();
        let profile = StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![OpeningPattern::ClaimFirst],
            opening_pattern_weights: vec![1.0],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::Never,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec!["specific".to_string()],
            ai_tells_to_avoid: vec!["delve".to_string()],
            thread_rhythm: ThreadRhythm::Linear,
            thread_max_length: 5,
            thread_opener_must_hook: false,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        };
        let recipe = BlendRecipe {
            version: 1,
            blend: vec![BlendEntry {
                writer: "k".to_string(),
                weight: 1.0,
            }],
            overrides: PartialStyleProfile::default(),
        };
        let store = SnapshotStore::open(dir.path(), persona).unwrap();
        store.save_new(profile, &recipe).unwrap();
        let root = dir.path().to_path_buf();
        (dir, root)
    }

    /// Fixture seed (high-engagement X post analog).
    fn fixture_seed() -> BlogSeed {
        BlogSeed {
            text: "Every tool response your LLM agent consumes is a potential attack vector."
                .into(),
            source_url: "https://twitter.com/i/web/status/9001".into(),
            source_tweet_id: "9001".into(),
            source_posted_at: Utc::now(),
            engagement_score: 5.0,
            rationale: "Top-engagement X post in the last 7 days (score 5.00, posted 2026-05-15)."
                .into(),
        }
    }

    /// Fixture draft — title on first line, then paragraph.
    const FIXTURE_DRAFT: &str = "# Agent Loops Cost Money

Opening paragraph that explains the topic concretely.

## Section

More detail about the topic with sourced specifics from the digest.
";

    /// Builder helper for `BlogConfig` (single candidate by default).
    #[allow(clippy::too_many_arguments)]
    fn mk_blog_cfg<'a>(
        profiles_root: &'a std::path::Path,
        posts_dir: &'a std::path::Path,
        out_dir: &'a std::path::Path,
        style_css: &'a std::path::Path,
        provider: Arc<BoxedProvider>,
        delivery: Arc<dyn BlogReviewDelivery>,
        candidates_per_draft: usize,
        seed: BlogSeed,
    ) -> BlogConfig<'a> {
        BlogConfig {
            persona_name: "x",
            provider,
            writer_provider: None,
            corpora_root: profiles_root,
            profiles_root,
            on_progress: None,
            seed,
            candidates_per_draft,
            delivery,
            credentials: Arc::new(StubCredentialResolver),
            posts_dir,
            out_dir,
            style_css,
            site_url: "https://pascal.heartbit.ai",
            site_title: "pascal.heartbit.ai",
        }
    }

    /// Write a dummy `style.css` and return its absolute path. Used by
    /// the success-path tests so `render_site` finds it.
    fn write_style(dir: &std::path::Path) -> std::path::PathBuf {
        let p = dir.join("style.css");
        std::fs::write(&p, "body{}").unwrap();
        p
    }

    // --- Test 1: happy path — pick(0), write markdown, render site -------

    #[tokio::test]
    async fn run_blog_pipeline_pick_index_0_writes_markdown_and_renders_site() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let tmp = TempDir::new().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style_css = write_style(tmp.path());

        let provider = MockProvider::arc(vec![
            "research digest about agent loops",
            FIXTURE_DRAFT,
            r#"{"verdict":"pass","style_match_score":0.92}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery_concrete =
            MockBlogReviewDelivery::arc(crate::review::DeliveryOutcome::Pick(0));
        let delivery_trait: Arc<dyn BlogReviewDelivery> = delivery_concrete.clone();
        let cfg = mk_blog_cfg(
            &profiles_root,
            &posts_dir,
            &out_dir,
            &style_css,
            provider,
            delivery_trait,
            1,
            fixture_seed(),
        );
        let out = run_blog_pipeline(cfg).await.expect("happy path");
        match &out.outcome {
            BlogOutcome::Posted {
                chosen_index,
                post_path,
                post_url,
            } => {
                assert_eq!(*chosen_index, 0);
                assert!(
                    post_path.exists(),
                    "post_path should exist; got: {}",
                    post_path.display()
                );
                assert!(
                    post_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap()
                        .ends_with("-agent-loops-cost-money.md"),
                    "post filename should end with the slug; got: {}",
                    post_path.display()
                );
                assert!(
                    post_url.ends_with("/agent-loops-cost-money/"),
                    "post_url should be slugged; got: {post_url}"
                );
            }
            other => panic!("expected Posted, got {other:?}"),
        }
        // Rendered HTML at out_dir/<slug>/index.html must exist.
        let rendered_html = out_dir.join("agent-loops-cost-money").join("index.html");
        assert!(
            rendered_html.exists(),
            "rendered HTML should exist; got: {}",
            rendered_html.display()
        );
        // report() received the Posted outcome.
        let reports = delivery_concrete.reports.lock().unwrap();
        assert_eq!(reports.len(), 1, "report() should be called exactly once");
        match &reports[0] {
            BlogOutcome::Posted { chosen_index, .. } => assert_eq!(*chosen_index, 0),
            other => panic!("expected report() to receive Posted, got {other:?}"),
        }
    }

    // --- Test 2: skip — no file written -----------------------------------

    #[tokio::test]
    async fn run_blog_pipeline_skip_returns_skipped_no_write() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let tmp = TempDir::new().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style_css = write_style(tmp.path());

        let provider = MockProvider::arc(vec![
            "research digest",
            FIXTURE_DRAFT,
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockBlogReviewDelivery::arc(crate::review::DeliveryOutcome::Skip);
        let cfg = mk_blog_cfg(
            &profiles_root,
            &posts_dir,
            &out_dir,
            &style_css,
            provider,
            delivery as Arc<dyn BlogReviewDelivery>,
            1,
            fixture_seed(),
        );
        let out = run_blog_pipeline(cfg).await.expect("skip is success");
        assert!(
            matches!(out.outcome, BlogOutcome::Skipped),
            "expected Skipped, got {:?}",
            out.outcome
        );
        // posts_dir must be empty (no markdown written).
        let entries: Vec<_> = std::fs::read_dir(&posts_dir)
            .map(|d| d.filter_map(|e| e.ok()).collect())
            .unwrap_or_default();
        assert!(
            entries.is_empty(),
            "posts_dir must be empty on skip; got {} entries",
            entries.len()
        );
    }

    // --- Test 3: timed out — no file written ------------------------------

    #[tokio::test]
    async fn run_blog_pipeline_timed_out_returns_timed_out_no_write() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let tmp = TempDir::new().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style_css = write_style(tmp.path());

        let provider = MockProvider::arc(vec![
            "research digest",
            FIXTURE_DRAFT,
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockBlogReviewDelivery::arc(crate::review::DeliveryOutcome::TimedOut);
        let cfg = mk_blog_cfg(
            &profiles_root,
            &posts_dir,
            &out_dir,
            &style_css,
            provider,
            delivery as Arc<dyn BlogReviewDelivery>,
            1,
            fixture_seed(),
        );
        let out = run_blog_pipeline(cfg).await.expect("timeout is success");
        assert!(
            matches!(out.outcome, BlogOutcome::TimedOut),
            "expected TimedOut, got {:?}",
            out.outcome
        );
        let entries: Vec<_> = std::fs::read_dir(&posts_dir)
            .map(|d| d.filter_map(|e| e.ok()).collect())
            .unwrap_or_default();
        assert!(
            entries.is_empty(),
            "posts_dir must be empty on timeout; got {} entries",
            entries.len()
        );
    }

    // --- Test 4: all candidates have empty title — gate-rejected, no deliver

    #[tokio::test]
    async fn run_blog_pipeline_all_candidates_gate_rejected_skips_delivery() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let tmp = TempDir::new().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style_css = write_style(tmp.path());

        // Writer returns whitespace-only draft → empty title/slug → drop.
        let provider = MockProvider::arc(vec![
            "research digest",
            "   \n\n   \n",
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery_concrete = MockBlogReviewDelivery::errored("delivery must not be called");
        let delivery_trait: Arc<dyn BlogReviewDelivery> = delivery_concrete.clone();
        let cfg = mk_blog_cfg(
            &profiles_root,
            &posts_dir,
            &out_dir,
            &style_css,
            provider,
            delivery_trait,
            1,
            fixture_seed(),
        );
        let out = run_blog_pipeline(cfg)
            .await
            .expect("all-gate-rejected is success");
        match out.outcome {
            BlogOutcome::AllCandidatesGateRejected { reasons } => {
                assert!(!reasons.is_empty(), "expected drop reasons");
                assert!(
                    reasons[0].contains("empty title"),
                    "first reason should reference empty title; got: {reasons:?}"
                );
            }
            other => panic!("expected AllCandidatesGateRejected, got {other:?}"),
        }
        assert_eq!(
            delivery_concrete.deliver_calls(),
            0,
            "delivery must not be called when all candidates are pre-filtered"
        );
    }

    // --- Test 5: all candidates Unverifiable — gate-rejected, no deliver --

    #[tokio::test]
    async fn run_blog_pipeline_all_unverifiable_returns_all_candidates_gate_rejected() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let tmp = TempDir::new().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style_css = write_style(tmp.path());

        let provider = MockProvider::arc(vec![
            "research digest",
            FIXTURE_DRAFT,
            r#"{"verdict":"pass","style_match_score":0.85}"#,
            r#"{"verdict":"unverifiable","reason":"no data"}"#,
        ]);
        let delivery_concrete = MockBlogReviewDelivery::errored("delivery must not be called");
        let delivery_trait: Arc<dyn BlogReviewDelivery> = delivery_concrete.clone();
        let cfg = mk_blog_cfg(
            &profiles_root,
            &posts_dir,
            &out_dir,
            &style_css,
            provider,
            delivery_trait,
            1,
            fixture_seed(),
        );
        let out = run_blog_pipeline(cfg)
            .await
            .expect("all-unverifiable is success");
        match out.outcome {
            BlogOutcome::AllCandidatesGateRejected { reasons } => {
                assert!(!reasons.is_empty(), "expected drop reasons");
                assert!(
                    reasons[0].starts_with("unverifiable:"),
                    "first reason should start with 'unverifiable:'; got: {reasons:?}"
                );
            }
            other => panic!("expected AllCandidatesGateRejected, got {other:?}"),
        }
        assert_eq!(
            delivery_concrete.deliver_calls(),
            0,
            "delivery must not be called when all candidates are pre-filtered"
        );
    }

    // --- Test 6: post is written to slugged subdir -----------------------

    #[tokio::test]
    async fn run_blog_pipeline_writes_post_to_slugged_subdir() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let tmp = TempDir::new().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style_css = write_style(tmp.path());

        let provider = MockProvider::arc(vec![
            "research digest",
            FIXTURE_DRAFT,
            r#"{"verdict":"pass","style_match_score":0.92}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockBlogReviewDelivery::arc(crate::review::DeliveryOutcome::Pick(0));
        let cfg = mk_blog_cfg(
            &profiles_root,
            &posts_dir,
            &out_dir,
            &style_css,
            provider,
            delivery as Arc<dyn BlogReviewDelivery>,
            1,
            fixture_seed(),
        );
        let out = run_blog_pipeline(cfg).await.expect("happy path");
        assert!(matches!(out.outcome, BlogOutcome::Posted { .. }));

        // Markdown file at posts_dir/<YYYY-MM-DD>-agent-loops-cost-money.md
        let entries: Vec<_> = std::fs::read_dir(&posts_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1, "exactly one post markdown should exist");
        let fname = entries[0]
            .path()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap()
            .to_string();
        assert!(
            fname.ends_with("-agent-loops-cost-money.md"),
            "filename should be <date>-agent-loops-cost-money.md; got: {fname}"
        );
        // Rendered HTML at out_dir/agent-loops-cost-money/index.html
        let html = out_dir.join("agent-loops-cost-money").join("index.html");
        assert!(
            html.exists(),
            "rendered HTML should exist at slugged subdir; got: {}",
            html.display()
        );
    }

    // --- Test 7: render failure is propagated ----------------------------

    #[tokio::test]
    async fn run_blog_pipeline_render_failure_is_reported() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let tmp = TempDir::new().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        // Deliberately point at a style.css that does NOT exist — render_site
        // returns RenderError::StyleNotFound, which run_blog_pipeline wraps
        // in BlogError::Render.
        let missing_style = tmp.path().join("definitely-not-here.css");

        let provider = MockProvider::arc(vec![
            "research digest",
            FIXTURE_DRAFT,
            r#"{"verdict":"pass","style_match_score":0.92}"#,
            r#"{"verdict":"verified"}"#,
        ]);
        let delivery = MockBlogReviewDelivery::arc(crate::review::DeliveryOutcome::Pick(0));
        let cfg = mk_blog_cfg(
            &profiles_root,
            &posts_dir,
            &out_dir,
            &missing_style,
            provider,
            delivery as Arc<dyn BlogReviewDelivery>,
            1,
            fixture_seed(),
        );
        let err = run_blog_pipeline(cfg).await.unwrap_err();
        match err {
            BlogError::Render(render::RenderError::StyleNotFound(_)) => {}
            other => panic!("expected BlogError::Render(StyleNotFound), got {other:?}"),
        }
    }
}
