mod auth;
mod eval;
mod execute;
mod handlers;
mod memory;
pub(crate) mod operator_id;
mod types;
mod validate;

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::Router;
use axum::middleware;
use axum::routing::{get, post};
use tokio_util::sync::CancellationToken;

use heartbit::daemon::kafka;
use heartbit::{
    AgentEvent, AgentOutput, DaemonCore, DaemonMetrics, Error as HeartbitError, HeartbitConfig,
    InMemoryTaskStore, JwtValidator, Memory, MentionContext, PersonaMentionEntry, PostgresStore,
    ReplySharedContext,
};

use crate::{build_on_retry, build_provider_from_config, init_tracing_from_config};

use self::auth::{JwtMiddlewareState, auth_middleware, jwt_auth_middleware, resolve_auth_tokens};
use self::handlers::{
    HttpMetrics, cors_middleware, handle_approval, handle_cancel, handle_get, handle_healthz,
    handle_list, handle_metrics, handle_readyz, handle_stats, handle_stream, handle_submit,
    handle_usage, http_metrics_middleware, mcp_tools_for_user, validate_path_component,
};
use self::memory::build_institutional_entry;
use self::types::{AppState, PendingApprovals};

// --- Daemon startup ---

pub async fn run_daemon(
    config_path: &std::path::Path,
    bind_override: Option<&str>,
    verbose: bool,
    observability_flag: Option<&str>,
) -> Result<()> {
    let mut config = HeartbitConfig::from_file(config_path)
        .with_context(|| format!("failed to load config from {}", config_path.display()))?;

    // Resolve agent templates, skills, and variables.
    let variables = config.variables.clone();
    for i in 0..config.agents.len() {
        if config.agents[i].template.is_some()
            || !config.agents[i].skills.is_empty()
            || !config.agents[i].skill_dirs.is_empty()
        {
            let resolved = heartbit::resolve_agent_config(&config.agents[i], &variables)
                .with_context(|| {
                    format!(
                        "failed to resolve template for agent '{}'",
                        config.agents[i].name
                    )
                })?;
            config.agents[i] = resolved;
        }
    }

    init_tracing_from_config(&config)?;

    let daemon_config = config
        .daemon
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("[daemon] section required in config for daemon mode"))?
        .clone();

    let bind = bind_override
        .map(String::from)
        .unwrap_or_else(|| daemon_config.bind.clone());

    // Create metrics if enabled (default: enabled when no config or when enabled=true)
    let metrics_enabled = daemon_config.metrics.as_ref().is_none_or(|m| m.enabled);
    let metrics = if metrics_enabled {
        let m = DaemonMetrics::new().context("failed to create Prometheus metrics")?;
        tracing::info!("Prometheus metrics enabled");
        Some(Arc::new(m))
    } else {
        None
    };

    // Create a shared PgPool when a database URL is configured.
    // Reused by both the task store and the execute handler (for persistent memory).
    let db_pool: Option<sqlx::PgPool> = if let Some(ref db_url) = daemon_config.database_url {
        let pool = sqlx::PgPool::connect(db_url)
            .await
            .context("failed to connect to database")?;
        Some(pool)
    } else {
        None
    };

    // Build task store: PostgreSQL if configured, in-memory otherwise
    let store: Arc<dyn heartbit::TaskStore> = if let Some(ref pool) = db_pool {
        let pg_store = heartbit::PostgresTaskStore::new(pool.clone());
        pg_store
            .run_migration()
            .await
            .context("failed to run task migration")?;
        tracing::info!("task store: PostgreSQL");
        Arc::new(pg_store)
    } else {
        tracing::info!("task store: in-memory (tasks lost on restart)");
        Arc::new(InMemoryTaskStore::new())
    };

    // Run memory table migration when database is available.
    // This is done once at startup so the execute handler can create
    // PostgresMemoryStore instances without per-request migration overhead.
    if let Some(ref pool) = db_pool {
        let mut pg_mem = heartbit::PostgresMemoryStore::new(pool.clone());
        pg_mem
            .run_migration()
            .await
            .context("failed to run memory migration")?;
        tracing::info!("memory store: PostgreSQL migration complete");
    }

    // Create cancellation token
    let cancel = CancellationToken::new();

    // Keep a reference for background tasks that may need the store after it's moved.
    let store_for_tasks = store.clone();

    // B5b: per-tenant in-flight token tracker (optional; None = unbounded)
    let tenant_tracker: Option<Arc<heartbit::TenantTokenTracker>> = config
        .orchestrator
        .max_tokens_in_flight_per_tenant
        .map(|cap| Arc::new(heartbit::TenantTokenTracker::new(cap)));

    // B5b: circuit breaker tracker (always created; defaults are sensible when unconfigured)
    let circuit_tracker: Arc<heartbit::CircuitTracker> = Arc::new(heartbit::CircuitTracker::new(
        heartbit::CircuitConfig::from(&config.provider.circuit),
    ));

    // Create DaemonCore + DaemonHandle — Kafka is optional
    let (core, handle) = if let Some(ref kafka_config) = daemon_config.kafka {
        tracing::info!("ensuring Kafka topics exist");
        kafka::ensure_topics(kafka_config)
            .await
            .context("failed to ensure Kafka topics")?;

        let producer =
            kafka::create_producer(kafka_config).context("failed to create Kafka producer")?;
        let consumer = kafka::create_commands_consumer(kafka_config)
            .context("failed to create Kafka consumer")?;

        let (core, handle) =
            DaemonCore::new(&daemon_config, consumer, producer, store, cancel.clone());
        // Attach Postgres store for audit retention (used by HEARTBIT_AUDIT_RETAIN_DAYS).
        let core = if let Some(ref pool) = db_pool {
            core.with_postgres_store(std::sync::Arc::new(PostgresStore::new(pool.clone())))
        } else {
            core
        };
        // B5b: thread tenant tracker into DaemonCore for in-flight token enforcement
        let core = if let Some(ref tracker) = tenant_tracker {
            core.with_tenant_tracker(tracker.clone())
        } else {
            core
        };

        // P1.5 Task 13: wire MentionPollScheduler + real handler dispatch.
        // Build MentionContext from enabled persona_mentions entries when present.
        let core = match build_mention_context(&config, &daemon_config).await {
            Ok(Some(mc)) => core.with_mention_context(Arc::new(mc)),
            Ok(None) => core,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "failed to build MentionContext, continuing without mention polling"
                );
                core
            }
        };

        tracing::info!("Kafka consumer/producer initialized");
        (Some(core), handle)
    } else {
        tracing::info!("no [daemon.kafka] configured — running in HTTP-only mode");
        if daemon_config.persona_mentions.iter().any(|e| e.enabled) {
            tracing::warn!(
                "[[daemon.persona_mentions]] entries are configured but [daemon.kafka] is absent \
                 — mention polling requires Kafka and will NOT run in HTTP-only mode"
            );
        }
        let handle = heartbit::DaemonHandle::http_only(store);
        (None, handle)
    };

    // B5b: thread tenant tracker into DaemonHandle
    let handle = if let Some(ref tracker) = tenant_tracker {
        handle.with_tenant_tracker(tracker.clone())
    } else {
        handle
    };

    // Lifted out so the BlogContext build block can reference the same
    // `PostsContext` (the blog reuses the posts pipeline's history +
    // engagement stores for top-engagement seed selection).
    let mut posts_ctx_for_blog: Option<std::sync::Arc<heartbit::PostsContext>> = None;

    // --- Build PostsContext from daemon_config.persona_posts ---
    let core = if !daemon_config.persona_posts.is_empty() && daemon_config.kafka.is_some() {
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        heartbit_ghost::register(&mut registry);

        let provider = crate::build_provider_from_config(&config, None)
            .map_err(|e| anyhow::anyhow!("build llm provider for posts context: {e}"))?;

        let delivery: std::sync::Arc<dyn heartbit_ghost::review::ReviewDelivery> =
            std::sync::Arc::new(
                crate::persona_review::TelegramReviewDelivery::from_env()
                    .context("construct TelegramReviewDelivery for posts context")?,
            );

        let twitter_thread: std::sync::Arc<dyn heartbit_core::Tool> =
            std::sync::Arc::new(heartbit_ghost::tools::TwitterThreadTool::new());

        let credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver> =
            std::sync::Arc::new(crate::persona_review::EnvCredentialResolver);

        let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
            .map_err(|e| anyhow::anyhow!("resolve corpora dir: {e}"))?;
        let profiles_root = heartbit_ghost::voice::default_profiles_dir()
            .map_err(|e| anyhow::anyhow!("resolve profiles dir: {e}"))?;

        let mut entries: std::collections::HashMap<String, heartbit::PersonaPostEntry> =
            std::collections::HashMap::new();
        for cfg in &daemon_config.persona_posts {
            if !cfg.enabled {
                continue;
            }
            if cfg.post_interval_seconds < 60 {
                anyhow::bail!(
                    "[[daemon.persona_posts]] persona='{}': post_interval_seconds must be \u{2265}60",
                    cfg.persona
                );
            }
            let history: std::sync::Arc<dyn heartbit_ghost::posts::PostHistoryStore> =
                match cfg.post_history_store.as_str() {
                    "in_memory" => {
                        std::sync::Arc::new(heartbit_ghost::posts::InMemoryPostHistoryStore::new())
                    }
                    "jsonl" => {
                        let raw_path = cfg.post_history_path.as_deref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "persona_posts persona='{}' uses jsonl but no post_history_path set",
                            cfg.persona
                        )
                    })?;
                        let path = expand_tilde(raw_path)?;
                        std::sync::Arc::new(
                            heartbit_ghost::posts::JsonlPostHistoryStore::open(&path)
                                .await
                                .with_context(|| {
                                    format!("open jsonl post history at {}", path.display())
                                })?,
                        )
                    }
                    other => anyhow::bail!(
                        "unknown post_history_store backend '{other}' for persona '{}' \
                         (expected 'in_memory' or 'jsonl')",
                        cfg.persona
                    ),
                };
            let operator_user_id = match operator_id::resolve_operator_user_id(
                &cfg.persona,
                &daemon_config.persona_mentions,
                |k| std::env::var(k).ok(),
            ) {
                Ok((id, source)) => {
                    tracing::info!(
                        persona = %cfg.persona,
                        source = ?source,
                        "resolved operator_user_id for persona_posts entry"
                    );
                    id
                }
                Err(e) => {
                    // Loud banner: this is strictly worse for ops than a
                    // crash-loop if it goes unnoticed, so log at error level
                    // and bump a metric. Daemon stays up; other personas run.
                    tracing::error!(
                        persona = %cfg.persona,
                        "SKIPPING [[daemon.persona_posts]] entry: {e}"
                    );
                    if let Some(ref m) = metrics {
                        m.inc_persona_posts_skipped(&cfg.persona, "missing_operator_user_id");
                    }
                    continue;
                }
            };
            if entries.contains_key(&cfg.persona) {
                anyhow::bail!(
                    "duplicate [[daemon.persona_posts]] entry for persona '{}'",
                    cfg.persona
                );
            }
            // Engagement-feedback wiring (P2.0 final). The engagement store
            // is co-located with the post history: when `post_history_store`
            // is "jsonl", we open a JSONL engagement store under
            // `.heartbit/engagement/{persona}.jsonl`; otherwise in-memory.
            // `JoinedTopPostsProvider` joins the two stores at writer-time
            // to surface top-engaged posts as few-shot exemplars.
            let engagement_store: std::sync::Arc<dyn heartbit_ghost::posts::EngagementStore> =
                match cfg.post_history_store.as_str() {
                    "jsonl" => {
                        let raw_path = format!(
                            ".heartbit/engagement/{}.jsonl",
                            cfg.persona.replace(':', "-")
                        );
                        let path = expand_tilde(&raw_path)?;
                        std::sync::Arc::new(
                            heartbit_ghost::posts::JsonlEngagementStore::open(&path)
                                .await
                                .with_context(|| {
                                    format!("open engagement jsonl at {}", path.display())
                                })?,
                        )
                    }
                    _ => std::sync::Arc::new(heartbit_ghost::posts::InMemoryEngagementStore::new()),
                };
            let top_posts_provider: std::sync::Arc<dyn heartbit_ghost::posts::TopPostsProvider> =
                std::sync::Arc::new(heartbit_ghost::posts::JoinedTopPostsProvider::new(
                    history.clone(),
                    engagement_store.clone(),
                    cfg.persona.clone(),
                ));
            entries.insert(
                cfg.persona.clone(),
                heartbit::PersonaPostEntry {
                    history,
                    interval: std::time::Duration::from_secs(cfg.post_interval_seconds),
                    interval_jitter_pct: cfg.interval_jitter_pct,
                    active_hours: cfg.active_hours.clone(),
                    candidates_per_draft: cfg.candidates_per_draft,
                    history_lookback: chrono::Duration::days(cfg.post_history_lookback_days),
                    topic_brief: cfg.topic_brief.clone(),
                    operator_user_id,
                    engagement_refresh: std::time::Duration::from_secs(
                        cfg.engagement_refresh_seconds,
                    ),
                    // Reuse the post-jitter pct — engagement cadence has the
                    // same "don't look like a bot" concern as posting.
                    engagement_jitter_pct: cfg.interval_jitter_pct,
                    engagement_store,
                    engagement_min_age_hours: cfg.engagement_min_age_hours,
                    engagement_max_age_days: cfg.engagement_max_age_days,
                    engagement_top_n: cfg.engagement_top_n,
                    top_posts_provider: Some(top_posts_provider),
                    // How to produce the optional head-tweet image, from
                    // `[[daemon.persona_posts]].image_source` (default online).
                    image_source: cfg.image_source,
                    // Build the per-persona writer+critic provider override
                    // from `[daemon.persona_posts.writer_provider]` when set.
                    // Retry/on_retry are intentionally None — the override
                    // provider runs inside the same agent loop and reuses
                    // the parent's circuit breaker layer above; the
                    // RetryingProvider wrapping is global to all providers
                    // built from `build_provider_from_config`, not part of
                    // per-agent overrides today. Operators who want retry
                    // on the writer provider specifically should set it
                    // via `[daemon.persona_posts.writer_provider.retry]`
                    // (not wired in v1 — falls back to provider defaults).
                    writer_provider: cfg
                        .writer_provider
                        .as_ref()
                        .map(|wp_cfg| crate::build_agent_provider(wp_cfg, None, None))
                        .transpose()
                        .with_context(|| {
                            format!(
                                "build writer_provider override for persona '{}'",
                                cfg.persona
                            )
                        })?,
                },
            );
        }

        if entries.is_empty() {
            // All entries were disabled.
            core
        } else {
            let posts_ctx = std::sync::Arc::new(heartbit::PostsContext {
                registry: std::sync::Arc::new(registry),
                provider,
                delivery,
                twitter_thread,
                credentials,
                corpora_root,
                profiles_root,
                entries,
            });
            tracing::info!(
                personas = ?posts_ctx.entries.keys().collect::<Vec<_>>(),
                "posts context configured"
            );
            // Capture for the BlogContext build block below — the blog
            // pipeline reuses these post-history / engagement stores for
            // its seed selection. Arc::clone is cheap.
            posts_ctx_for_blog = Some(posts_ctx.clone());
            core.map(|c| c.with_posts_context(posts_ctx))
        }
    } else if !daemon_config.persona_posts.is_empty() {
        tracing::warn!(
            "persona_posts entries configured but [daemon.kafka] is absent; \
             posts pipeline requires Kafka mode and will be skipped"
        );
        core
    } else {
        core
    };

    // --- Build QuotesContext from daemon_config.persona_quotes ---
    let core = if !daemon_config.persona_quotes.is_empty() && daemon_config.kafka.is_some() {
        let mut registry = heartbit_core::persona::PersonaRegistry::new();
        heartbit_ghost::register(&mut registry);

        let provider = crate::build_provider_from_config(&config, None)
            .map_err(|e| anyhow::anyhow!("build llm provider for quotes context: {e}"))?;

        let delivery: std::sync::Arc<dyn heartbit_ghost::quote::QuoteReviewDelivery> =
            std::sync::Arc::new(
                crate::persona_review::TelegramQuoteReviewDelivery::from_env()
                    .context("construct TelegramQuoteReviewDelivery for quotes context")?,
            );

        // Quote-tweet pipeline publishes as a REPLY rather than a quote
        // because X soft-restricts quote-tweets from automated / new /
        // low-engagement accounts (observed 2026-05-13: every quote
        // attempt 403'd with "you have not been mentioned or are not part
        // of the conversation thread"). The X UI test confirmed the
        // restriction is at the bot-account level, not per-source-tweet.
        // Replies use a different endpoint shape and aren't subject to
        // the same restriction. The pipeline keeps its "quote" naming
        // (config block, type names, file names) for continuity — the
        // publish action just routes to `twitter_reply` instead.
        let twitter_quote_tool: std::sync::Arc<dyn heartbit_core::Tool> =
            std::sync::Arc::new(heartbit_ghost::tools::TwitterReplyTool::new());

        let credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver> =
            std::sync::Arc::new(crate::persona_review::EnvCredentialResolver);

        let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
            .map_err(|e| anyhow::anyhow!("resolve corpora dir: {e}"))?;
        let profiles_root = heartbit_ghost::voice::default_profiles_dir()
            .map_err(|e| anyhow::anyhow!("resolve profiles dir: {e}"))?;

        // Build a shared XClient for the timeline source. Failure here is
        // FATAL — the source poller is load-bearing for the quote loop, so
        // we bail rather than degrade silently (matches the fail-fast
        // discipline of the rest of the quotes block).
        let synthetic_ctx = heartbit_core::ExecutionContext {
            credentials: Some(credentials.clone()),
            ..heartbit_core::ExecutionContext::default()
        };
        let xclient = std::sync::Arc::new(
            heartbit_ghost::tools::client::XClient::from_context(&synthetic_ctx)
                .await
                .with_context(|| {
                    "build XClient for quote source (check X_CONSUMER_KEY/SECRET + \
                     X_ACCESS_TOKEN/SECRET)"
                })?,
        );

        let mut entries: std::collections::HashMap<String, heartbit::PersonaQuoteEntry> =
            std::collections::HashMap::new();
        for cfg in &daemon_config.persona_quotes {
            if !cfg.enabled {
                continue;
            }
            if cfg.poll_interval_seconds < 60 {
                anyhow::bail!(
                    "[[daemon.persona_quotes]] persona='{}': poll_interval_seconds must be \u{2265}60",
                    cfg.persona
                );
            }
            let seen_store: std::sync::Arc<dyn heartbit_ghost::quote::sources::QuoteSeenStore> =
                match cfg.seen_store.as_str() {
                    "in_memory" => std::sync::Arc::new(
                        heartbit_ghost::quote::sources::InMemoryQuoteSeenStore::new(),
                    ),
                    "jsonl" => {
                        let raw_path = cfg.seen_store_path.as_deref().ok_or_else(|| {
                            anyhow::anyhow!(
                                "persona_quotes persona='{}' uses jsonl but no seen_store_path set",
                                cfg.persona
                            )
                        })?;
                        let path = expand_tilde(raw_path)?;
                        std::sync::Arc::new(
                            heartbit_ghost::quote::sources::JsonlQuoteSeenStore::open(&path)
                                .await
                                .with_context(|| {
                                    format!("open jsonl quote seen store at {}", path.display())
                                })?,
                        )
                    }
                    other => anyhow::bail!(
                        "unknown seen_store backend '{other}' for persona '{}' \
                         (expected 'in_memory' or 'jsonl')",
                        cfg.persona
                    ),
                };
            let source: std::sync::Arc<dyn heartbit_ghost::quote::sources::QuoteSource> =
                std::sync::Arc::new(heartbit_ghost::quote::sources::XUserTimelineSource::new(
                    xclient.clone(),
                ));
            if entries.contains_key(&cfg.persona) {
                anyhow::bail!(
                    "duplicate [[daemon.persona_quotes]] entry for persona '{}'",
                    cfg.persona
                );
            }
            entries.insert(
                cfg.persona.clone(),
                heartbit::PersonaQuoteEntry {
                    source,
                    seen_store,
                    interval: std::time::Duration::from_secs(cfg.poll_interval_seconds),
                    interval_jitter_pct: cfg.interval_jitter_pct,
                    active_hours: cfg.active_hours.clone(),
                    source_user_ids: cfg.source_user_ids.clone(),
                    candidates_per_draft: cfg.candidates_per_draft,
                    max_age_hours: cfg.max_age_hours,
                    max_candidates_per_tick: cfg.max_candidates_per_tick,
                    // Same retry/on_retry rationale as posts: the override
                    // provider is wrapped only by the global retry layer
                    // already applied to `build_provider_from_config`.
                    writer_provider: cfg
                        .writer_provider
                        .as_ref()
                        .map(|wp_cfg| crate::build_agent_provider(wp_cfg, None, None))
                        .transpose()
                        .with_context(|| {
                            format!(
                                "build writer_provider override for persona '{}'",
                                cfg.persona
                            )
                        })?,
                },
            );
        }

        if entries.is_empty() {
            // All entries were disabled.
            core
        } else {
            let quotes_ctx = std::sync::Arc::new(heartbit::QuotesContext {
                registry: std::sync::Arc::new(registry),
                provider,
                delivery,
                twitter_quote_tool,
                credentials,
                corpora_root,
                profiles_root,
                entries,
            });
            tracing::info!(
                personas = ?quotes_ctx.entries.keys().collect::<Vec<_>>(),
                "quotes context configured"
            );
            core.map(|c| c.with_quotes_context(quotes_ctx))
        }
    } else if !daemon_config.persona_quotes.is_empty() {
        tracing::warn!(
            "persona_quotes entries configured but [daemon.kafka] is absent; \
             quotes pipeline requires Kafka mode and will be skipped"
        );
        core
    } else {
        core
    };

    // --- Build BlogContext from daemon_config.persona_blog ---
    //
    // The blog pipeline reuses the matching `[[daemon.persona_posts]]`
    // entry's post history + engagement stores for seed selection, so
    // posts_ctx_for_blog must be present (which transitively requires
    // Kafka mode — the blog scheduler also publishes via the Kafka
    // producer).
    let core = if let Some(ref blog_cfg) = daemon_config.persona_blog {
        if !blog_cfg.enabled {
            tracing::info!("persona_blog config present but disabled — skipping");
            core
        } else {
            let posts_ctx_inner = posts_ctx_for_blog.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "persona_blog requires posts context (no enabled \
                     [[daemon.persona_posts]] entries configured, or [daemon.kafka] \
                     is missing — the blog reuses the posts pipeline's history + \
                     engagement stores for seed selection)"
                )
            })?;
            let posts_entry = posts_ctx_inner
                .entries
                .get(&blog_cfg.persona)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "persona_blog references persona '{}' but no matching \
                     [[daemon.persona_posts]] entry exists — the blog reuses \
                     the post history + engagement stores for seed selection",
                        blog_cfg.persona
                    )
                })?;

            let posts_dir = expand_tilde(&blog_cfg.posts_dir)?;
            let out_dir = expand_tilde(&blog_cfg.out_dir)?;
            let style_css = std::path::PathBuf::from("blog-site/style.css");

            std::fs::create_dir_all(&posts_dir)
                .with_context(|| format!("create posts_dir at {}", posts_dir.display()))?;
            std::fs::create_dir_all(&out_dir)
                .with_context(|| format!("create out_dir at {}", out_dir.display()))?;

            let blog_delivery: std::sync::Arc<dyn heartbit_ghost::blog::BlogReviewDelivery> =
                std::sync::Arc::new(
                    crate::persona_review::TelegramBlogReviewDelivery::from_env()
                        .context("construct TelegramBlogReviewDelivery for blog context")?,
                );

            // Same retry/on_retry rationale as posts and quotes: the
            // per-agent override provider is wrapped only by the global
            // retry layer already applied to `build_provider_from_config`.
            let writer_provider: Option<std::sync::Arc<heartbit::BoxedProvider>> = blog_cfg
                .writer_provider
                .as_ref()
                .map(|wp_cfg| crate::build_agent_provider(wp_cfg, None, None))
                .transpose()
                .with_context(|| {
                    format!(
                        "build writer_provider override for persona_blog '{}'",
                        blog_cfg.persona
                    )
                })?;

            let top_posts_provider = posts_entry.top_posts_provider.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "persona_blog requires the matching [[daemon.persona_posts]] \
                     entry to have a top_posts_provider configured (auto-built \
                     when post_history_store='jsonl' or 'in_memory' — verify the \
                     posts entry isn't disabled)"
                )
            })?;

            // The corpora + profiles roots are re-resolved here. The
            // existing posts/quotes blocks pay the same cost — keep this
            // block self-contained rather than threading the prior values
            // through extra locals.
            let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                .map_err(|e| anyhow::anyhow!("resolve corpora dir: {e}"))?;
            let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                .map_err(|e| anyhow::anyhow!("resolve profiles dir: {e}"))?;

            let entry = heartbit::PersonaBlogEntry {
                top_posts_provider,
                interval: std::time::Duration::from_secs(blog_cfg.poll_interval_seconds),
                interval_jitter_pct: blog_cfg.interval_jitter_pct,
                active_hours: blog_cfg.active_hours.clone(),
                seed_lookback_days: blog_cfg.seed_lookback_days,
                candidates_per_draft: blog_cfg.candidates_per_draft,
                posts_dir,
                out_dir,
                style_css,
                site_url: blog_cfg.site_url.clone(),
                site_title: blog_cfg.site_title.clone(),
                writer_provider,
                deploy_command: blog_cfg.deploy_command.clone(),
                x_announce: blog_cfg.x_announce.clone(),
                github_readme: blog_cfg.github_readme.clone(),
            };

            let blog_ctx = std::sync::Arc::new(heartbit::BlogContext {
                registry: posts_ctx_inner.registry.clone(),
                provider: posts_ctx_inner.provider.clone(),
                delivery: blog_delivery,
                credentials: posts_ctx_inner.credentials.clone(),
                corpora_root,
                profiles_root,
                entry,
                persona_name: blog_cfg.persona.clone(),
            });
            tracing::info!(
                persona = %blog_ctx.persona_name,
                site_url = %blog_ctx.entry.site_url,
                "blog context configured"
            );
            core.map(|c| c.with_blog_context(blog_ctx))
        }
    } else {
        core
    };

    // Create shared memory store (one store for all execution paths)
    let shared_memory: Option<Arc<dyn Memory>> = if let Some(ref mem_config) = config.memory {
        match crate::create_memory_store(mem_config).await {
            Ok(store) => {
                tracing::info!("daemon shared memory: enabled");
                Some(store)
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to create shared memory, continuing without");
                None
            }
        }
    } else {
        None
    };

    // Provision workspace once (idempotent, reused by all execution paths)
    let daemon_workspace_dir =
        crate::provision_workspace(&crate::workspace_root_from_config(&config));

    // Pre-load MCP + A2A tools once for all agents (daemon reuses across tasks).
    // Also populate the transport pool for per-user auth stamping.
    let transport_pool = Arc::new(heartbit::McpTransportPool::new());
    let tool_cache: Arc<HashMap<String, Vec<Arc<dyn heartbit::tool::Tool>>>> = {
        let mut cache = HashMap::new();
        for agent in &config.agents {
            let mut agent_tools = crate::load_mcp_tools(&agent.name, &agent.mcp_servers).await;
            agent_tools.extend(crate::load_a2a_tools(&agent.name, &agent.a2a_agents).await);
            // Warm the transport pool for HTTP MCP servers (enables per-user auth stamping)
            for entry in &agent.mcp_servers {
                if !entry.is_stdio()
                    && !transport_pool.contains(entry.url())
                    && let Err(e) = transport_pool
                        .get_or_connect(entry.url(), entry.auth_header().map(String::from))
                        .await
                {
                    tracing::warn!(
                        agent = %agent.name,
                        server = %entry.display_name(),
                        error = %e,
                        "failed to warm transport pool (tools still loaded via McpClient)"
                    );
                }
            }
            if !agent_tools.is_empty() {
                tracing::info!(
                    agent = %agent.name,
                    tools = agent_tools.len(),
                    "cached MCP/A2A tools"
                );
            }
            cache.insert(agent.name.clone(), agent_tools);
        }
        Arc::new(cache)
    };

    // Signal handler
    heartbit::signal::spawn_shutdown_handler(cancel.clone());

    // Resolve observability mode
    let config_obs = config
        .telemetry
        .as_ref()
        .and_then(|t| t.observability_mode.as_deref());
    let mode = crate::resolve_observability(observability_flag, config_obs, verbose);

    // Shared map for subject tokens: "{tenant_id}:{user_id}" -> raw JWT.
    // Populated by HTTP submit handler, consumed by TokenExchangeAuthProvider.
    let user_tokens: Arc<std::sync::RwLock<HashMap<String, String>>> =
        Arc::new(std::sync::RwLock::new(HashMap::new()));

    // Create per-user auth provider from token_exchange config (dynamic MCP auth)
    let auth_provider: Option<Arc<dyn heartbit::AuthProvider>> = config
        .daemon
        .as_ref()
        .and_then(|d| d.auth.as_ref())
        .and_then(|a| a.token_exchange.as_ref())
        .map(|te| {
            tracing::info!(
                exchange_url = %te.exchange_url,
                client_id = %te.client_id,
                "token exchange auth provider configured for per-user MCP delegation"
            );
            Arc::new(
                heartbit::TokenExchangeAuthProvider::new(
                    &te.exchange_url,
                    &te.client_id,
                    &te.client_secret,
                    &te.agent_token,
                )
                .with_tenant_id(te.tenant_id.clone())
                .with_scopes(te.scopes.clone())
                .with_user_tokens(user_tokens.clone()),
            ) as Arc<dyn heartbit::AuthProvider>
        });

    // Build the runner closure that creates an Orchestrator per task
    let config_arc = Arc::new(config);
    let config_for_state = config_arc.clone();
    let runner_metrics = metrics.clone();
    let runner_memory = shared_memory.clone();
    let runner_workspace = daemon_workspace_dir.clone();
    let runner_tools = tool_cache.clone();
    let runner_auth_provider = auth_provider;
    let state_auth_provider = runner_auth_provider.clone();
    let runner_transport_pool = transport_pool.clone();
    let runner_user_tokens = user_tokens.clone();
    // B5b: capture trackers for per-task provider wrapping
    let runner_circuit_tracker = circuit_tracker.clone();
    let runner_tenant_tracker = tenant_tracker.clone();
    // Shared pending approvals map: REST endpoint writes decisions, on_approval callback reads them.
    let pending_approvals: PendingApprovals = Arc::new(std::sync::Mutex::new(HashMap::new()));
    let runner_pending_approvals = pending_approvals.clone();
    let build_runner = move |task_id: uuid::Uuid,
                             task_text: String,
                             source: String,
                             story_id: Option<String>,
                             trust_level: Option<heartbit::TrustLevel>,
                             on_event_fn: Arc<dyn Fn(AgentEvent) + Send + Sync>,
                             user_id: Option<String>,
                             tenant_id: Option<String>,
                             user_roles: Vec<String>,
                             mcp_auth_tokens: Option<HashMap<String, String>>|
          -> Pin<
        Box<dyn std::future::Future<Output = Result<AgentOutput, HeartbitError>> + Send>,
    > {
        let config = config_arc.clone();
        let task_metrics = runner_metrics.clone();
        let memory = runner_memory.clone();
        let workspace_dir = runner_workspace.clone();
        let tools = runner_tools.clone();
        let task_auth_provider = runner_auth_provider.clone();
        let task_transport_pool = runner_transport_pool.clone();
        let task_user_tokens = runner_user_tokens.clone();
        let task_pending_approvals = runner_pending_approvals.clone();
        // B5b: capture trackers inside async block
        let task_circuit_tracker = runner_circuit_tracker.clone();
        let task_tenant_tracker = runner_tenant_tracker.clone();
        Box::pin(async move {
            // Wrap on_event to also record metrics
            let on_event: Arc<heartbit::OnEvent> = if let Some(ref m) = task_metrics {
                let inner = on_event_fn;
                let metrics = m.clone();
                let tenant_label = tenant_id.clone();
                Arc::new(move |event: AgentEvent| {
                    metrics.record_event(&event, tenant_label.as_deref());
                    inner(event);
                })
            } else {
                on_event_fn
            };

            let on_retry = build_on_retry(&on_event);
            let base_provider = build_provider_from_config(&config, Some(on_retry.clone()))
                .map_err(|e| HeartbitError::Daemon(e.to_string()))?;
            // B5b: wrap with circuit breaker using per-task tenant scope
            let scope =
                heartbit::TenantScope::from_audit_fields(tenant_id.as_deref(), user_id.as_deref());
            let provider = crate::wrap_with_circuit(
                base_provider,
                Arc::clone(&task_circuit_tracker),
                &config.provider.name,
                scope,
            );

            // Record submission and track active tasks (after provider creation to avoid gauge leak on error)
            if let Some(ref m) = task_metrics {
                m.record_task_submitted(tenant_id.as_deref(), &source);
                m.tasks_active().inc();
            }
            let start = Instant::now();

            let on_text: Arc<heartbit::OnText> = Arc::new(|_: &str| {});

            // Build on_approval callback for REST/SSE tasks: blocks on mpsc channel,
            // resolved by POST /v1/tasks/{id}/approve endpoint.
            let approval_task_id = task_id;
            let approval_map = task_pending_approvals;
            let on_approval: Arc<heartbit::OnApproval> = Arc::new(move |_tool_calls| {
                let (tx, rx) = std::sync::mpsc::channel();
                {
                    let mut pending = approval_map.lock().expect("pending_approvals lock");
                    pending.insert(approval_task_id, tx);
                }
                // block_in_place tells tokio this thread is about to block, so it can
                // compensate by spawning additional worker threads.
                tokio::task::block_in_place(|| match rx.recv_timeout(Duration::from_secs(300)) {
                    Ok(decision) => decision,
                    Err(_) => {
                        let mut pending = approval_map.lock().expect("pending_approvals lock");
                        pending.remove(&approval_task_id);
                        heartbit::ApprovalDecision::Deny
                    }
                })
            });

            // Wire SensorSecurityGuardrail for sensor-sourced tasks
            let (mut guardrails, memory_confidentiality_cap): (
                Vec<Arc<dyn heartbit::Guardrail>>,
                Option<heartbit::Confidentiality>,
            ) = if source.starts_with("sensor:") {
                let trust = trust_level.unwrap_or(heartbit::TrustLevel::Unknown);
                let guardrail =
                    heartbit::SensorSecurityGuardrail::new(source.clone(), trust, vec![]);
                // Memory confidentiality cap based on trust level
                let cap = match trust {
                    heartbit::TrustLevel::Owner | heartbit::TrustLevel::Verified => None,
                    _ => Some(heartbit::Confidentiality::Public),
                };
                (vec![Arc::new(guardrail)], cap)
            } else {
                (vec![], None)
            };

            // Append config-based guardrails (injection, PII, tool policy, LLM judge)
            if let Some(ref gc) = config.guardrails {
                match gc.build() {
                    Ok(config_guardrails) => guardrails.extend(config_guardrails),
                    Err(e) => {
                        tracing::error!(error = %e, "failed to build config guardrails for daemon task, skipping");
                    }
                }
            }

            // Per-user memory namespace: wrap shared memory with user-scoped prefix
            let task_memory: Option<Arc<dyn heartbit::Memory>> = if let Some(ref uid) = user_id
                && let Some(ref tid) = tenant_id
                && let Some(ref mem) = memory
            {
                let ns_prefix = format!("tenant:{tid}:user:{uid}");
                let ns = heartbit::NamespacedMemory::new(mem.clone(), ns_prefix.clone());
                tracing::debug!(namespace = %ns_prefix, "memory namespaced to tenant/user");
                Some(Arc::new(ns))
            } else {
                memory
            };

            // Per-user workspace isolation: scope to {workspace}/{tenant_id}/{user_id}/
            // Sanitize IDs to prevent path traversal (reject /, \, .., absolute paths)
            let task_workspace = if let (Some(base), Some(tid), Some(uid)) =
                (&workspace_dir, &tenant_id, &user_id)
            {
                if let Err(e) =
                    validate_path_component(tid).and_then(|_| validate_path_component(uid))
                {
                    tracing::error!(
                        tenant_id = %tid,
                        user_id = %uid,
                        error = %e,
                        "rejected unsafe tenant/user ID in workspace path"
                    );
                    return Err(HeartbitError::Daemon(format!(
                        "invalid tenant/user ID for workspace: {e}"
                    )));
                }
                let scoped = base.join(tid).join(uid);
                if let Err(e) = tokio::fs::create_dir_all(&scoped).await {
                    tracing::warn!(
                        path = %scoped.display(),
                        error = %e,
                        "failed to create per-user workspace"
                    );
                }
                tracing::debug!(
                    path = %scoped.display(),
                    "workspace scoped to tenant/user"
                );
                Some(scoped)
            } else {
                workspace_dir
            };

            // Dynamic MCP auth -- resolve per-user tools via transport pool.
            // When mcp_auth_tokens are provided (from cloud/gateway), use DirectAuthProvider.
            // Otherwise fall back to TokenExchangeAuthProvider.
            let effective_auth: Option<Arc<dyn heartbit::AuthProvider>> =
                if let Some(tokens) = mcp_auth_tokens {
                    Some(Arc::new(heartbit::DirectAuthProvider::new(tokens)))
                } else {
                    task_auth_provider
                };

            let user_tools: Option<HashMap<String, Vec<Arc<dyn heartbit::tool::Tool>>>> =
                if let (Some(ap), Some(uid), Some(tid)) = (&effective_auth, &user_id, &tenant_id) {
                    mcp_tools_for_user(&config, ap, uid, tid, &task_transport_pool).await
                } else {
                    None
                };

            // Clean up subject token after exchange -- prevent unbounded growth
            // of the shared user_tokens map across long-lived daemon sessions.
            if let (Some(uid), Some(tid)) = (&user_id, &tenant_id) {
                let key = format!("{tid}:{uid}");
                if let Ok(mut tokens) = task_user_tokens.write() {
                    tokens.remove(&key);
                }
            }

            let effective_tools = user_tools.as_ref().unwrap_or(&tools);

            // Compute role-gated shared memory write permission.
            let allow_shared_write = config
                .daemon
                .as_ref()
                .map(|d| &d.memory.shared_write_roles)
                .is_none_or(|roles| {
                    roles.is_empty() || user_roles.iter().any(|r| roles.contains(r))
                });

            // Check if any agent has dangerous_tools enabled.
            let dangerous_tools = config.agents.iter().any(|a| a.dangerous_tools);

            let result = crate::RuntimeBuilder::new(provider, &config, &task_text, on_text)
                .on_approval(Some(on_approval))
                .on_event(Some(on_event))
                .observability_mode(mode)
                .story_id(story_id.as_deref())
                .external_memory(task_memory)
                .workspace_dir(task_workspace)
                .pre_loaded_tools(Some(effective_tools))
                .guardrails(guardrails)
                .memory_confidentiality_cap(memory_confidentiality_cap)
                .audit_user_id(user_id.as_deref())
                .audit_tenant_id(tenant_id.as_deref())
                .allow_shared_write(allow_shared_write)
                .dangerous_tools(dangerous_tools)
                // B5b: thread tenant and circuit trackers into orchestrator/agent runner
                .tenant_tracker(task_tenant_tracker)
                .circuit_tracker(Some(task_circuit_tracker))
                .run()
                .await
                .map_err(|e| HeartbitError::Daemon(e.to_string()));

            let duration_secs = start.elapsed().as_secs_f64();
            if let Some(ref m) = task_metrics {
                m.tasks_active().dec();
                m.record_task_by_source(&source);
                match &result {
                    Ok(_) => m.record_task_completed(duration_secs, tenant_id.as_deref()),
                    Err(_) => m.record_task_failed(duration_secs, tenant_id.as_deref()),
                }
            }

            result
        })
    };

    // Build Kafka brokers string for readiness check
    let kafka_brokers = daemon_config.kafka.as_ref().map(|k| k.brokers.clone());

    // Resolve auth tokens from config + env
    let auth_tokens = resolve_auth_tokens(
        daemon_config
            .auth
            .as_ref()
            .map(|a| a.bearer_tokens.as_slice())
            .unwrap_or_default(),
        std::env::var("HEARTBIT_API_KEY").ok(),
    );

    if auth_tokens.is_some() {
        tracing::info!("HTTP API authentication enabled (bearer token)");
    } else {
        tracing::warn!("HTTP API authentication disabled — all routes are public");
    }

    // Build JWT validator from config (for multi-tenant auth)
    let jwt_validator = daemon_config
        .auth
        .as_ref()
        .and_then(|auth| auth.jwks_url.as_ref())
        .map(|jwks_url| {
            let auth = daemon_config.auth.as_ref().unwrap();
            let mut validator =
                JwtValidator::new(jwks_url.clone(), auth.issuer.clone(), auth.audience.clone());
            if let Some(ref claim) = auth.user_id_claim {
                validator = validator.with_user_id_claim(claim.clone());
            }
            if let Some(ref claim) = auth.tenant_id_claim {
                validator = validator.with_tenant_id_claim(claim.clone());
            }
            if let Some(ref claim) = auth.roles_claim {
                validator = validator.with_roles_claim(claim.clone());
            }
            tracing::info!("JWT/JWKS authentication enabled ({})", jwks_url);
            Arc::new(validator)
        });

    // In HTTP-only mode, spawn the audit retention prune task here.
    // (DaemonCore::run handles the Kafka-mode case.)
    if core.is_none()
        && let Some(ref pool) = db_pool
    {
        let retain_days = daemon_config
            .audit
            .retain_days
            .or_else(|| {
                std::env::var("HEARTBIT_AUDIT_RETAIN_DAYS")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .filter(|&d| d > 0);
        let interval_minutes = daemon_config
            .audit
            .prune_interval_minutes
            .filter(|&m| m > 0)
            .unwrap_or(60);

        if let Some(days) = retain_days {
            let store = std::sync::Arc::new(PostgresStore::new(pool.clone()));
            let cancel_prune = cancel.clone();
            let retain = chrono::Duration::days(days as i64);
            let interval = std::time::Duration::from_secs(interval_minutes * 60);
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(interval);
                tick.tick().await;
                loop {
                    tokio::select! {
                        _ = cancel_prune.cancelled() => {
                            tracing::info!("audit prune task: cancellation received, exiting");
                            break;
                        }
                        _ = tick.tick() => {
                            match store.prune_audit(retain).await {
                                Ok(n) => tracing::info!(removed = n, "audit retention prune"),
                                Err(e) => tracing::warn!(error = %e, "audit prune failed"),
                            }
                        }
                    }
                }
            });
            tracing::info!(
                retain_days = days,
                interval_minutes = interval_minutes,
                "audit retention prune task started (HTTP-only mode)"
            );
        }
    }

    // B5b: idempotency-key TTL sweep task (HTTP-only mode).
    // In Kafka mode, DaemonCore::run handles this.
    if core.is_none()
        && let Some(ttl_hours) = daemon_config.idempotency.ttl_hours
    {
        let sweep_store = store_for_tasks.clone();
        let cancel_sweep = cancel.clone();
        let interval_min = daemon_config
            .idempotency
            .sweep_interval_minutes
            .unwrap_or(60);
        let interval = std::time::Duration::from_secs(u64::from(interval_min) * 60);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.tick().await; // skip immediate fire
            loop {
                tokio::select! {
                    _ = cancel_sweep.cancelled() => {
                        tracing::info!("idempotency sweep: cancellation received, exiting");
                        break;
                    }
                    _ = tick.tick() => {
                        let cutoff = chrono::Utc::now()
                            - chrono::Duration::hours(i64::from(ttl_hours));
                        match sweep_store.sweep_expired_idempotency_keys(cutoff) {
                            Ok(n) if n > 0 => {
                                tracing::info!(swept = n, "idempotency keys expired")
                            }
                            Ok(_) => {}
                            Err(e) => {
                                tracing::warn!(error = %e, "idempotency sweep failed")
                            }
                        }
                    }
                }
            }
        });
        tracing::info!(
            ttl_hours,
            interval_minutes = interval_min,
            "idempotency key TTL sweep task started (HTTP-only mode)"
        );
    }

    // Start HTTP server
    let app_state = AppState {
        handle,
        start_time: Instant::now(),
        metrics: metrics.clone(),
        cancel: cancel.clone(),
        kafka_brokers,
        config: config_for_state,
        observability_mode: mode,
        shared_memory: shared_memory.clone(),
        workspace_dir: daemon_workspace_dir,
        tool_cache: tool_cache.clone(),
        jwt_validator: jwt_validator.clone(),
        user_tokens: user_tokens.clone(),
        auth_provider: state_auth_provider,
        transport_pool: transport_pool.clone(),
        pending_approvals,
        db_pool,
    };

    // Public routes -- health, readiness, metrics -- never require auth
    let public_routes = Router::new()
        .route("/v1/health", get(handle_healthz))
        .route("/v1/ready", get(handle_readyz))
        .route("/v1/metrics", get(handle_metrics));

    // Protected routes -- tasks, SSE, stats
    let protected_routes = Router::new()
        .route("/v1/tasks", post(handle_submit))
        .route("/v1/tasks", get(handle_list))
        .route("/v1/tasks/{id}", get(handle_get))
        .route("/v1/tasks/{id}/cancel", post(handle_cancel))
        .route("/v1/tasks/{id}/stream", get(handle_stream))
        .route("/v1/tasks/{id}/approve", post(handle_approval))
        .route("/v1/tasks/execute", post(execute::handle_execute))
        .route("/v1/tasks/eval", post(eval::handle_eval))
        .route("/v1/stats", get(handle_stats))
        .route("/v1/usage", get(handle_usage));

    // Apply auth middleware layers.
    // JWT middleware (innermost): validates JWT, injects UserContext into extensions.
    // Bearer token middleware (outermost): gates access by static token.
    // Layers execute outer-to-inner, so bearer auth runs first, then JWT enrichment.
    let jwt_is_sole_auth = jwt_validator.is_some() && auth_tokens.is_none();
    let mut protected_routes = protected_routes;
    if let Some(ref validator) = jwt_validator {
        let jwt_state = JwtMiddlewareState {
            validator: validator.clone(),
            // When JWT is the only auth, it must reject unauthenticated requests.
            // When bearer tokens are also configured, JWT only enriches.
            required: jwt_is_sole_auth,
        };
        protected_routes = protected_routes.layer(middleware::from_fn_with_state(
            jwt_state,
            jwt_auth_middleware,
        ));
    }

    let routes = if let Some(ref tokens) = auth_tokens {
        let authed = protected_routes.layer(middleware::from_fn_with_state(
            tokens.clone(),
            auth_middleware,
        ));
        public_routes.merge(authed)
    } else {
        // Either JWT-only auth (required=true rejects unauthenticated) or no auth at all
        public_routes.merge(protected_routes)
    };

    let mut app = routes.with_state(app_state);

    // Permissive CORS for local dashboard access (file:// or localhost origins).
    app = app.layer(middleware::from_fn(cors_middleware));

    // Add HTTP metrics middleware when metrics are enabled
    if let Some(ref m) = metrics {
        let http_metrics =
            HttpMetrics::register(m.registry()).context("failed to register HTTP metrics")?;
        app = app.layer(middleware::from_fn_with_state(
            http_metrics,
            http_metrics_middleware,
        ));
    }

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("failed to bind to {bind}"))?;
    tracing::info!(bind = %bind, "runtime HTTP server started");

    let http_cancel = cancel.clone();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                http_cancel.cancelled().await;
            })
            .await
            .ok();
    });

    // Build on_complete callback for institutional memory persistence
    let on_complete: Option<Arc<heartbit::OnTaskComplete>> = if shared_memory.is_some() {
        let persist_memory = shared_memory.clone();
        Some(Arc::new(move |outcome: heartbit::TaskOutcome| {
            // Auto-persist completed task results to institutional memory
            if outcome.state == heartbit::TaskState::Completed
                && let Some(ref result) = outcome.result_summary
                && !result.is_empty()
                && let Some(ref memory) = persist_memory
            {
                let memory = memory.clone();
                let entry = build_institutional_entry(
                    &outcome.id,
                    &outcome.source,
                    result,
                    outcome.story_id.as_deref(),
                    outcome.user_id.as_deref(),
                    outcome.tenant_id.as_deref(),
                );
                let scope =
                    heartbit::TenantScope::new(outcome.tenant_id.clone().unwrap_or_default());
                tokio::spawn(async move {
                    if let Err(e) = memory.store(&scope, entry).await {
                        tracing::warn!(
                            error = %e,
                            "failed to persist institutional memory"
                        );
                    }
                });
            }
        }))
    } else {
        None
    };

    // Run the DaemonCore consumer loop (blocks until cancellation), or
    // wait for shutdown in HTTP-only mode.
    if let Some(core) = core {
        tracing::info!("runtime core started, consuming from Kafka");
        core.run(build_runner, on_complete)
            .await
            .context("daemon core error")?;
    } else {
        tracing::info!("runtime started in HTTP-only mode (no Kafka consumer)");
        cancel.cancelled().await;
    }

    tracing::info!("runtime shut down gracefully");
    Ok(())
}

/// Static-validate the daemon config and print findings to stderr.
/// Returns Ok if no issues found, Err with a count summary otherwise.
///
/// Performs no network calls and no Kafka/DB initialization — safe to run
/// against a production config file from anywhere.
pub async fn validate_config_only(config_path: &std::path::Path) -> Result<()> {
    let config = HeartbitConfig::from_file(config_path)
        .with_context(|| format!("failed to load config from {}", config_path.display()))?;

    let issues =
        validate::validate_daemon_config(&config, |k| std::env::var(k).ok(), |p| p.exists());

    if issues.is_empty() {
        eprintln!("✓ {} validates clean", config_path.display());
        return Ok(());
    }

    eprintln!("✗ {} has {} issue(s):", config_path.display(), issues.len());
    for issue in &issues {
        eprintln!("  - {issue}");
    }
    anyhow::bail!("config validation found {} issue(s)", issues.len());
}

/// Expand a leading `~/` to `$HOME/`. Returns an error if `$HOME` is
/// unset. Paths without a tilde prefix are returned unchanged.
fn expand_tilde(s: &str) -> anyhow::Result<std::path::PathBuf> {
    use anyhow::Context as _;
    if let Some(rest) = s.strip_prefix("~/") {
        let home = std::env::var("HOME").context("$HOME not set")?;
        Ok(std::path::PathBuf::from(home).join(rest))
    } else {
        Ok(std::path::PathBuf::from(s))
    }
}

// ---------------------------------------------------------------------------
// P1.5 Task 13: Build MentionContext from [[daemon.persona_mentions]] config.
// Returns `Ok(None)` when there are no enabled entries.
// Returns `Ok(Some(mc))` when at least one entry is wired up successfully.
// Returns `Err(...)` only on a hard configuration failure (missing env vars,
// bad paths) that should abort the mention-polling path gracefully.
// ---------------------------------------------------------------------------
async fn build_mention_context(
    config: &HeartbitConfig,
    daemon_config: &heartbit::DaemonConfig,
) -> Result<Option<MentionContext>> {
    let enabled_entries: Vec<&heartbit::PersonaMentionsConfig> = daemon_config
        .persona_mentions
        .iter()
        .filter(|e| e.enabled)
        .collect();

    if enabled_entries.is_empty() {
        return Ok(None);
    }

    // Resolve corpora/profiles root dirs.
    let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
        .context("failed to resolve corpora root for mention context")?;
    let profiles_root = heartbit_ghost::voice::default_profiles_dir()
        .context("failed to resolve profiles root for mention context")?;

    // Shared credential resolver (env-based).
    let credentials: Arc<dyn heartbit_core::CredentialResolver> =
        Arc::new(crate::persona_review::EnvCredentialResolver);

    // Build LLM provider for the reply pipeline (no retry event callback needed here).
    let provider = build_provider_from_config(config, None)
        .context("failed to build provider for mention context")?;

    // Build persona registry.
    let mut registry = heartbit::PersonaRegistry::new();
    heartbit_ghost::register(&mut registry);

    // Build twitter_reply tool.
    let twitter_reply_tool: Arc<dyn heartbit::tool::Tool> =
        Arc::new(heartbit_ghost::tools::TwitterReplyTool);

    // Build twitter_mentions tool (shared across entries).
    let mentions_tool: Arc<dyn heartbit::tool::Tool> =
        Arc::new(heartbit_ghost::tools::TwitterMentionsTool);

    // Build ReplyReviewDelivery — Telegram if env vars are present, else log-only stub.
    let delivery: Arc<dyn heartbit_ghost::reply::ReplyReviewDelivery> =
        match crate::persona_review::TelegramReplyReviewDelivery::from_env() {
            Ok(d) => {
                tracing::info!("mention context: Telegram review delivery configured");
                Arc::new(d)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Telegram review delivery not configured (HEARTBIT_TELEGRAM_TOKEN / HEARTBIT_TELEGRAM_REVIEW_CHAT_ID unset), \
                     falling back to log-only delivery"
                );
                Arc::new(LogOnlyReplyDelivery)
            }
        };

    // Build a content-aware scam judge using the same provider. Runs ONE
    // cheap LLM call per surviving mention BEFORE the multi-agent pipeline,
    // catching crypto pumps / spam / ads that the structural guards miss.
    // Fail-open: judge errors degrade to OK so legit traffic is never
    // silently suppressed.
    let scam_judge: Option<Arc<heartbit_ghost::reply::ScamJudge>> = Some(Arc::new(
        heartbit_ghost::reply::ScamJudge::new(provider.clone()),
    ));
    tracing::info!("mention context: scam-judge pre-pipeline filter enabled");

    let reply_ctx = ReplySharedContext {
        registry: Arc::new(registry),
        provider,
        delivery,
        twitter_tool: twitter_reply_tool,
        credentials: credentials.clone(),
        corpora_root,
        profiles_root,
        scam_judge,
    };

    // Build one PersonaMentionEntry per enabled config entry.
    let mut entries = Vec::with_capacity(enabled_entries.len());
    for cfg in enabled_entries {
        // Resolve mention store backend.
        let store: Arc<dyn heartbit_ghost::reply::MentionStore> = match cfg.mention_store.as_str() {
            "jsonl" => {
                let path = cfg
                    .mention_store_path
                    .clone()
                    .unwrap_or_else(|| format!(".heartbit/mentions/{}.jsonl", cfg.persona));
                match heartbit_ghost::reply::JsonlMentionStore::open(&path).await {
                    Ok(s) => Arc::new(s),
                    Err(e) => {
                        tracing::warn!(
                            persona = %cfg.persona,
                            path,
                            error = %e,
                            "failed to open JSONL mention store, falling back to in-memory"
                        );
                        Arc::new(heartbit_ghost::reply::InMemoryMentionStore::new())
                    }
                }
            }
            _ => Arc::new(heartbit_ghost::reply::InMemoryMentionStore::new()),
        };

        // Resolve budget tracker backend.
        let budget_tracker: Arc<dyn heartbit_ghost::reply::DailyTokenBudget> =
            match cfg.budget_store.as_str() {
                "jsonl" => {
                    let path = cfg.budget_path.as_deref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "persona '{}': budget_store = 'jsonl' requires budget_path",
                            cfg.persona
                        )
                    })?;
                    let path = expand_tilde(path)?;
                    match heartbit_ghost::reply::JsonlDailyBudget::open(&path).await {
                        Ok(b) => Arc::new(b),
                        Err(e) => {
                            tracing::warn!(
                                persona = %cfg.persona,
                                path = %path.display(),
                                error = %e,
                                "failed to open JSONL budget store, falling back to in-memory"
                            );
                            Arc::new(heartbit_ghost::reply::InMemoryDailyBudget::new())
                        }
                    }
                }
                _ => Arc::new(heartbit_ghost::reply::InMemoryDailyBudget::new()),
            };

        // Build optional bot-heuristic config.
        let bot_heuristic = if cfg.enable_bot_heuristic_guard && cfg.bot_heuristic_threshold > 0 {
            let patterns = if cfg.suspicious_handle_patterns.is_empty() {
                heartbit_ghost::reply::BotHeuristicConfig::defaults().suspicious_handle_patterns
            } else {
                cfg.suspicious_handle_patterns.clone()
            };
            Some(heartbit_ghost::reply::BotHeuristicConfig {
                suspicious_handle_patterns: patterns,
                min_follower_following_ratio: cfg.min_follower_following_ratio,
                min_account_age_days: cfg.min_account_age_days,
                threshold: cfg.bot_heuristic_threshold,
            })
        } else {
            None
        };

        let mut entry = PersonaMentionEntry::new(
            &cfg.persona,
            &cfg.user_id,
            cfg.poll_interval_seconds,
            cfg.candidates_per_reply,
            store,
            credentials.clone(),
            10, // max_results default
        );
        entry.enable_thread_depth_guard = cfg.enable_thread_depth_guard;
        entry.bot_heuristic = bot_heuristic;
        entry.per_conversation_max_replies = cfg.per_conversation_max_replies;
        entry.budget_tracker = budget_tracker;
        entry.daily_token_budget = cfg.daily_token_budget;
        entries.push(entry);
    }

    tracing::info!(
        entries = entries.len(),
        "MentionContext built with persona mention entries"
    );

    // Build a shared XClient for per-mention enrichment (author handle +
    // parent tweet). Failure here is non-fatal — degrade to V1 (bot guard
    // inert, no parent context) and warn-log so the operator can fix the
    // credentials. The credentials come from the same EnvCredentialResolver
    // already used by other tools.
    let synthetic_ctx = heartbit_core::ExecutionContext {
        credentials: Some(credentials.clone()),
        ..heartbit_core::ExecutionContext::default()
    };
    let mc = MentionContext::new(entries, reply_ctx, mentions_tool);
    let mc = match heartbit_ghost::tools::client::XClient::from_context(&synthetic_ctx).await {
        Ok(client) => {
            tracing::info!(
                "mention context: X enrichment enabled (bot guard live, parent fetch on)"
            );
            // Attach a shared in-memory enrichment cache: dedups repeated
            // /2/users/:id (24h TTL) and /2/tweets/:id (indefinite, immutable)
            // calls across the daemon's lifetime. Cuts X API spend on
            // recurring authors and shared parent tweets.
            let cache = Arc::new(heartbit_ghost::reply::EnrichmentCache::new());
            tracing::info!("mention context: enrichment cache attached (24h user TTL)");
            mc.with_enricher(Arc::new(client))
                .with_enrichment_cache(cache)
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "X enrichment disabled (bot guard inert, no parent context) — check X_CONSUMER_KEY/SECRET + X_ACCESS_TOKEN/SECRET"
            );
            mc
        }
    };
    Ok(Some(mc))
}

// ---------------------------------------------------------------------------
// Log-only ReplyReviewDelivery — used when Telegram is not configured.
// Logs the review message and auto-approves the first candidate.
// ---------------------------------------------------------------------------
struct LogOnlyReplyDelivery;

impl heartbit_ghost::reply::ReplyReviewDelivery for LogOnlyReplyDelivery {
    fn deliver<'a>(
        &'a self,
        msg: heartbit_ghost::reply::ReplyReviewMessage,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        heartbit_ghost::review::DeliveredReview,
                        heartbit_ghost::review::ReviewDeliveryError,
                    >,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            tracing::info!(
                mention_id = %msg.mention.id,
                candidates = msg.candidates.len(),
                "LogOnlyReplyDelivery: auto-approving first candidate (no Telegram configured)"
            );
            Ok(heartbit_ghost::review::DeliveredReview {
                outcome: heartbit_ghost::review::DeliveryOutcome::Pick(0),
                receipt: heartbit_ghost::review::DeliveryReceipt {
                    data: serde_json::Value::Null,
                },
            })
        })
    }

    fn report<'a>(
        &'a self,
        _receipt: heartbit_ghost::review::DeliveryReceipt,
        _outcome: heartbit_ghost::reply::ReplyOutcome,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<(), heartbit_ghost::review::ReviewDeliveryError>,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move { Ok(()) })
    }
}
