//! `heartbit persona <sub>` subcommand surface.
//!
//! Functional shells against the `PersonaRegistry` populated at startup by
//! linked persona crates (e.g. `heartbit-ghost`). Once persona crates
//! register their recipes, these subcommands light up without any CLI
//! changes. P1.0 ships the registration shell; subcommand bodies land in
//! later sub-phases (P1.1–P1.4).

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use clap::Subcommand;

use heartbit::{PersonaParams, PersonaRegistry};
use heartbit_ghost::posts::PostHistoryStore as _;

use crate::build_provider_from_env;

#[derive(Debug, Subcommand)]
pub enum PersonaCommand {
    /// List registered personas.
    List,

    /// Show the configured persona instance (expanded TOML).
    Show {
        /// Persona instance name as declared in `[[persona]]`.
        name: String,
    },

    /// Run the persona once with a one-off prompt; print result to stdout.
    Run {
        /// Persona instance name.
        name: String,
        /// Run a single dry-run prompt without posting.
        #[arg(long, value_name = "PROMPT")]
        once: String,
        /// Send candidates to Telegram for review and post on user pick.
        /// Without this flag: runs P1.3c direct mode (judge picks; stdout only).
        #[arg(long, default_value = "false")]
        review: bool,
    },

    /// Post a proactive thread on demand (no daemon needed).
    ///
    /// Without `--topic`: invokes the full topic-generator pipeline and
    /// requires `HEARTBIT_GHOST_OPERATOR_USER_ID` to be set.
    ///
    /// With `--topic`: skips the topic generator and calls the review
    /// pipeline directly (equivalent to `persona run --review`).
    Post {
        /// Persona instance name.
        name: String,
        /// Override the topic; skips the topic generator.
        #[arg(long)]
        topic: Option<String>,
        /// Override the candidate count (defaults to 3).
        #[arg(long)]
        candidates: Option<usize>,
    },

    /// List recent post history for a persona (reads the JSONL store).
    Posts {
        /// Persona instance name.
        name: String,
        /// Maximum number of entries to return.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Path to the JSONL post history file.
        /// Defaults to `~/.heartbit/ghost/posts/<persona>.jsonl`.
        #[arg(long)]
        history_path: Option<String>,
    },

    /// Manage the persona's reference corpus.
    Corpus {
        #[command(subcommand)]
        sub: CorpusCommand,
    },

    /// Manage the persona's blended style profile.
    Profile {
        #[command(subcommand)]
        sub: ProfileCommand,
    },

    /// Set the persona's autonomy phase.
    Phase {
        /// Persona instance name.
        name: String,
        /// Phase: calibration | supervised | autonomous | sentinel.
        #[arg(long)]
        set: String,
    },

    /// Reply to a single mention on demand (no daemon needed).
    /// Fetches the mention via the X API, runs the reply pipeline,
    /// sends candidate drafts to Telegram for review, posts the chosen
    /// draft. Useful for testing the reply flow without the cron.
    Reply {
        /// Persona instance name.
        name: String,
        /// X tweet ID of the mention to reply to.
        #[arg(long)]
        mention_id: String,
        /// Number of distinct candidate replies to generate (1..=3).
        #[arg(long, default_value = "2")]
        candidates: usize,
    },

    /// List recent mentions of the operator's X account.
    /// Use the printed mention id with `persona reply --mention-id <id>`.
    Mentions {
        /// Persona instance name (currently unused; reserved for per-persona
        /// X account scoping in the future).
        name: String,
        /// Maximum number of mentions to return (5..=100).
        #[arg(long, default_value = "10")]
        limit: u32,
        /// Only return mentions newer than this id.
        #[arg(long)]
        since_id: Option<String>,
        /// Operator X user_id. Defaults to whoever the OAuth1 creds resolve
        /// via `GET /2/users/me`.
        #[arg(long)]
        user_id: Option<String>,
    },

    /// Halt this persona on a running daemon.
    Pause {
        /// Persona instance name.
        name: String,
    },

    /// Resume this persona on a running daemon.
    Resume {
        /// Persona instance name.
        name: String,
    },

    /// Export the user-preference dataset for external training (L3).
    ExportPreferences {
        /// Persona instance name.
        name: String,
        /// Output format. Default: jsonl.
        #[arg(long, default_value = "jsonl")]
        format: String,
    },

    /// Show recent posts with their full audit trail.
    Audit {
        /// Persona instance name.
        name: String,
        /// Time window, e.g. `24h`, `7d`.
        #[arg(long, default_value = "24h")]
        since: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum CorpusCommand {
    /// Add an exemplar corpus from a JSONL file.
    Add {
        /// Writer handle (without `@`), e.g. `karpathy`.
        writer: String,
        /// Path to a JSONL file of posts.
        path: std::path::PathBuf,
    },
    /// List the corpus sources for a persona.
    List {
        /// Persona instance name.
        name: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum ProfileCommand {
    /// Recompute the blended style profile from the current corpus.
    Rebuild {
        /// Persona instance name.
        name: String,
    },
    /// Diff two profile versions.
    Diff {
        /// Persona instance name.
        name: String,
        /// First version, e.g. `v3`.
        v1: String,
        /// Second version, e.g. `v4`.
        v2: String,
    },
}

const NO_PERSONAS_REGISTERED: &str = "No personas registered. (heartbit-ghost or another persona crate must be linked into this build.)";

/// Build a human-readable suffix listing available personas (or the
/// empty-registry hint if none are registered). Used by every error
/// site so `persona show`, `persona corpus`, and `persona profile`
/// surface the same set of names that `persona list` prints.
fn registry_suffix(registry: &PersonaRegistry) -> String {
    let available = registry.list();
    if available.is_empty() {
        NO_PERSONAS_REGISTERED.to_string()
    } else {
        format!("Available personas: {}.", available.join(", "))
    }
}

/// Dispatch a `persona` subcommand against the registry populated by
/// linked persona crates (e.g. `heartbit-ghost`).
pub async fn run(cmd: PersonaCommand) -> Result<()> {
    let mut registry = PersonaRegistry::new();
    heartbit_ghost::register(&mut registry);
    dispatch(cmd, &registry).await
}

async fn dispatch(cmd: PersonaCommand, registry: &PersonaRegistry) -> Result<()> {
    match cmd {
        PersonaCommand::List => {
            let names = registry.list();
            if names.is_empty() {
                println!("No personas registered.");
            } else {
                for name in names {
                    println!("{name}");
                }
            }
            Ok(())
        }
        PersonaCommand::Run { name, once, review } => {
            let persona = registry.get(&name).ok_or_else(|| {
                anyhow!("persona '{name}' not found. {}", registry_suffix(registry))
            })?;
            let expansion = persona
                .expand(&PersonaParams::default())
                .map_err(|e| anyhow!("expand persona '{name}': {e}"))?;

            // If the persona declares a non-default researcher (currently
            // only `repo_researcher` for heartbit-rs:x), pull it out of
            // the expansion's agent vec along with the matching tools
            // (filtered to research-relevant ones) so the pipeline uses
            // it instead of the legacy `researcher_recipe()`.
            let researcher_override = expansion
                .agents
                .iter()
                .find(|a| a.name == "repo_researcher")
                .map(|recipe| {
                    let recipe = std::sync::Arc::new(recipe.clone_config());
                    let tools: Vec<std::sync::Arc<dyn heartbit_core::Tool>> = expansion
                        .tools
                        .iter()
                        .filter(|t| t.definition().name == "repo_inspect")
                        .cloned()
                        .collect();
                    (recipe, tools)
                });

            let provider =
                build_provider_from_env(None).map_err(|e| anyhow!("build llm provider: {e}"))?;
            let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
            let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;

            let on_progress: std::sync::Arc<dyn Fn(&str) + Send + Sync> =
                std::sync::Arc::new(|s: &str| eprintln!("> {s}"));

            if review {
                // P1.3d review mode: Telegram + post.
                let cfg = crate::persona_review::review_config_from_env(
                    &name,
                    &once,
                    3,
                    provider,
                    &corpora_root,
                    &profiles_root,
                    Some(on_progress),
                    expansion.mode_addendum,
                    researcher_override.clone(),
                )
                .await
                .map_err(|e| anyhow!("review config: {e}"))?;

                let n_requested = cfg.candidates_per_draft;
                let output = heartbit_ghost::review::run_review_pipeline(cfg)
                    .await
                    .map_err(|e| anyhow!("review pipeline: {e}"))?;

                eprintln!(
                    "> ok: candidates={}/{}, outcome={:?}",
                    output.candidates.len(),
                    n_requested,
                    output.outcome,
                );
                Ok(())
            } else {
                // P1.3c direct mode (unchanged).
                let cfg = heartbit_ghost::pipeline::PipelineConfig {
                    persona_name: &name,
                    topic: &once,
                    provider,
                    corpora_root: &corpora_root,
                    profiles_root: &profiles_root,
                    on_progress: Some(on_progress),
                    candidates_per_draft: 3,
                    mode_addendum: expansion.mode_addendum,
                    researcher_override,
                };

                let n_requested = cfg.candidates_per_draft;
                let output = heartbit_ghost::pipeline::run_pipeline(cfg)
                    .await
                    .map_err(|e| anyhow!("pipeline: {e}"))?;

                // run_pipeline already printed final_draft to stdout.
                eprintln!(
                    "> ok: candidates={}/{}, chosen={}, revise iterations={}, style match={:.2}, fact check={:?}, image={}",
                    output.candidates.len(),
                    n_requested,
                    output.chosen_index,
                    output.revise_iterations,
                    output.style_match_score,
                    output.fact_check_verdict,
                    output
                        .image
                        .as_ref()
                        .map(|i| i.url.as_str())
                        .unwrap_or("none"),
                );
                Ok(())
            }
        }
        PersonaCommand::Post {
            name,
            topic,
            candidates,
        } => {
            let persona = registry.get(&name).ok_or_else(|| {
                anyhow!("persona '{name}' not found. {}", registry_suffix(registry))
            })?;
            let expansion = persona
                .expand(&PersonaParams::default())
                .map_err(|e| anyhow!("expand persona '{name}': {e}"))?;

            // Researcher override (heartbit-rs:x uses repo_researcher).
            let researcher_override = expansion
                .agents
                .iter()
                .find(|a| a.name == "repo_researcher")
                .map(|recipe| {
                    let recipe = std::sync::Arc::new(recipe.clone_config());
                    let tools: Vec<std::sync::Arc<dyn heartbit_core::Tool>> = expansion
                        .tools
                        .iter()
                        .filter(|t| t.definition().name == "repo_inspect")
                        .cloned()
                        .collect();
                    (recipe, tools)
                });

            let provider =
                build_provider_from_env(None).map_err(|e| anyhow!("build llm provider: {e}"))?;
            let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
            let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;

            let on_progress: std::sync::Arc<dyn Fn(&str) + Send + Sync> =
                std::sync::Arc::new(|s: &str| eprintln!("> {s}"));

            // With --topic: skip the topic generator and call the review
            // pipeline directly (mirrors `persona run --review`).
            if let Some(t) = topic {
                let cfg = crate::persona_review::review_config_from_env(
                    &name,
                    &t,
                    candidates.unwrap_or(3),
                    provider,
                    &corpora_root,
                    &profiles_root,
                    Some(on_progress),
                    expansion.mode_addendum,
                    researcher_override,
                )
                .await
                .map_err(|e| anyhow!("review config: {e}"))?;
                let n_requested = cfg.candidates_per_draft;
                let output = heartbit_ghost::review::run_review_pipeline(cfg)
                    .await
                    .map_err(|e| anyhow!("review pipeline: {e}"))?;
                eprintln!(
                    "> ok: candidates={}/{}, outcome={:?}",
                    output.candidates.len(),
                    n_requested,
                    output.outcome,
                );
                return Ok(());
            }

            // Without --topic: invoke handle_persona_post with an ephemeral
            // in-memory post history store (daemon uses a persistent JSONL
            // store; CLI is for one-off testing).
            let operator_user_id = std::env::var("HEARTBIT_GHOST_OPERATOR_USER_ID")
                .map_err(|_| anyhow!(
                    "HEARTBIT_GHOST_OPERATOR_USER_ID must be set for `persona post` without --topic"
                ))?;
            let history: std::sync::Arc<dyn heartbit_ghost::posts::PostHistoryStore> =
                std::sync::Arc::new(heartbit_ghost::posts::InMemoryPostHistoryStore::new());
            let delivery: std::sync::Arc<dyn heartbit_ghost::review::ReviewDelivery> =
                std::sync::Arc::new(
                    crate::persona_review::TelegramReviewDelivery::from_env()
                        .map_err(|e| anyhow!("construct TelegramReviewDelivery: {e}"))?,
                );
            let twitter_tool: std::sync::Arc<dyn heartbit_core::Tool> =
                std::sync::Arc::new(heartbit_ghost::tools::TwitterThreadTool::new());
            let credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver> =
                std::sync::Arc::new(crate::persona_review::EnvCredentialResolver);
            let deps = heartbit::PersonaPostDeps {
                persona_name: &name,
                registry,
                history: &*history,
                history_lookback: chrono::Duration::days(30),
                topic_brief: None,
                operator_user_id: &operator_user_id,
                provider,
                delivery,
                twitter_tool,
                credentials,
                candidates_per_draft: candidates.unwrap_or(3),
                corpora_root: &corpora_root,
                profiles_root: &profiles_root,
                // One-shot CLI invocation: no engagement provider wired in
                // here (no history backing store to join against). Cold-start
                // semantics — exactly the same as a fresh daemon run.
                top_posts_provider: None,
                top_n: 0,
                // One-shot CLI uses the global provider for all stages.
                writer_provider: None,
            };
            let outcome = heartbit::handle_persona_post(deps)
                .await
                .map_err(|e| anyhow!("persona post: {e}"))?;
            eprintln!("> ok: outcome={outcome:?}");
            Ok(())
        }
        PersonaCommand::Posts {
            name,
            limit,
            history_path,
        } => {
            let path = match history_path {
                Some(p) => crate::persona_review::expand_tilde_str(&p)?,
                None => {
                    let home = std::env::var("HOME").map_err(|_| anyhow!("$HOME not set"))?;
                    std::path::PathBuf::from(home)
                        .join(".heartbit/ghost/posts")
                        .join(format!("{name}.jsonl"))
                }
            };
            if !path.exists() {
                println!("(no history at {})", path.display());
                return Ok(());
            }
            let store = heartbit_ghost::posts::JsonlPostHistoryStore::open(&path)
                .await
                .map_err(|e| anyhow!("open {}: {e}", path.display()))?;
            let recent = store
                .recent(&name, limit)
                .await
                .map_err(|e| anyhow!("recent: {e}"))?;
            if recent.is_empty() {
                println!("(no entries for persona '{name}')");
                return Ok(());
            }
            println!("Recent posts for {name} ({}):", recent.len());
            for (i, e) in recent.iter().enumerate() {
                let when = e.posted_at.format("%Y-%m-%d %H:%M");
                let tweet = e.tweet_id.as_deref().unwrap_or("-");
                let topic_display = if e.topic.is_empty() {
                    "(none)".to_string()
                } else {
                    e.topic.clone()
                };
                println!(
                    "  [{i}] {when} tweet={tweet}\n      topic: {topic_display}\n      outcome: {:?}",
                    e.outcome,
                );
            }
            Ok(())
        }
        PersonaCommand::Reply {
            name,
            mention_id,
            candidates,
        } => {
            let persona = registry.get(&name).ok_or_else(|| {
                anyhow!("persona '{name}' not found. {}", registry_suffix(registry))
            })?;
            let expansion = persona
                .expand(&PersonaParams::default())
                .map_err(|e| anyhow!("expand persona '{name}': {e}"))?;

            // Researcher override (heartbit-rs:x uses repo_researcher).
            let researcher_override = expansion
                .agents
                .iter()
                .find(|a| a.name == "repo_researcher")
                .map(|recipe| {
                    let recipe = std::sync::Arc::new(recipe.clone_config());
                    let tools: Vec<std::sync::Arc<dyn heartbit_core::Tool>> = expansion
                        .tools
                        .iter()
                        .filter(|t| t.definition().name == "repo_inspect")
                        .cloned()
                        .collect();
                    (recipe, tools)
                });

            let provider =
                build_provider_from_env(None).map_err(|e| anyhow!("build llm provider: {e}"))?;
            let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
            let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;

            eprintln!("> Fetching mention {mention_id}...");
            let mention = crate::persona_review::fetch_mention_one_off(&mention_id)
                .await
                .map_err(|e| anyhow!("fetch mention: {e}"))?;
            eprintln!(
                "> Mention from @{} (author_id={}): {}",
                mention.author_handle, mention.author_id, mention.text,
            );

            let on_progress: std::sync::Arc<dyn Fn(&str) + Send + Sync> =
                std::sync::Arc::new(|s: &str| eprintln!("> {s}"));

            let cfg = crate::persona_review::reply_config_from_env(
                &name,
                provider,
                &corpora_root,
                &profiles_root,
                Some(on_progress),
                mention,
                None, // parent — V1: not fetched (would need a second X API call)
                None, // mentioner_context — V1: not fetched
                candidates,
                expansion.mode_addendum,
                researcher_override,
            )
            .await
            .map_err(|e| anyhow!("reply config: {e}"))?;

            let output = heartbit_ghost::reply::run_reply_pipeline(cfg)
                .await
                .map_err(|e| anyhow!("reply pipeline: {e}"))?;
            eprintln!(
                "> ok: candidates={}, outcome={:?}",
                output.candidates.len(),
                output.outcome,
            );
            Ok(())
        }
        PersonaCommand::Mentions {
            name: _,
            limit,
            since_id,
            user_id,
        } => {
            let resolved_user_id = match user_id {
                Some(id) => id,
                None => {
                    let me = crate::persona_review::fetch_authenticated_user()
                        .await
                        .map_err(|e| anyhow!("resolve operator user_id: {e}"))?;
                    eprintln!(
                        "> Authenticated as @{} ({}) — user_id={}",
                        me.username, me.name, me.id,
                    );
                    me.id
                }
            };
            let mentions = crate::persona_review::list_recent_mentions(
                &resolved_user_id,
                limit,
                since_id.as_deref(),
            )
            .await
            .map_err(|e| anyhow!("list mentions: {e}"))?;
            if mentions.is_empty() {
                println!("(no mentions)");
                return Ok(());
            }
            println!("Recent mentions ({}):", mentions.len());
            for (i, m) in mentions.iter().enumerate() {
                let author = m.author_id.as_deref().unwrap_or("?");
                let when = m.created_at.as_deref().unwrap_or("?");
                let preview: String = m.text.chars().take(140).collect();
                println!("  [{i}] id={} author_id={author} at={when}", m.id);
                println!("      {preview}");
            }
            Ok(())
        }
        PersonaCommand::Show { name }
        | PersonaCommand::Phase { name, .. }
        | PersonaCommand::Pause { name }
        | PersonaCommand::Resume { name }
        | PersonaCommand::ExportPreferences { name, .. }
        | PersonaCommand::Audit { name, .. } => {
            if registry.get(&name).is_none() {
                return Err(anyhow!(
                    "persona '{name}' not found. {}",
                    registry_suffix(registry)
                ));
            }
            // Other subcommands land in P1.4.
            Err(anyhow!(
                "persona '{name}': subcommand body is not yet implemented. \
                 Use `heartbit persona run` for one-off generation; other \
                 subcommands land in P1.4."
            ))
        }
        PersonaCommand::Corpus { sub } => match sub {
            CorpusCommand::Add { writer, path } => {
                if registry.is_empty() {
                    return Err(anyhow!("{}", NO_PERSONAS_REGISTERED));
                }
                let root = heartbit_ghost::corpus::default_corpora_dir()
                    .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
                let mut corpus = heartbit_ghost::corpus::Corpus::open_or_create(&root, &writer)
                    .map_err(|e| anyhow!("open corpus for '{writer}': {e}"))?;
                let stats = corpus.append_from_jsonl(&path).map_err(|e| {
                    anyhow!("import {} into corpus '{writer}': {e}", path.display())
                })?;
                println!(
                    "ok: added {} new ({} deduped); total {} for writer '{}'",
                    stats.added, stats.deduped, stats.total_after, writer
                );
                Ok(())
            }
            CorpusCommand::List { name: persona_name } => {
                if registry.get(&persona_name).is_none() {
                    return Err(anyhow!(
                        "persona '{persona_name}' not found. {}",
                        registry_suffix(registry)
                    ));
                }
                let config = heartbit_ghost::voice::PersonaConfig::load(&persona_name)
                    .map_err(|e| anyhow!("load persona config for '{persona_name}': {e}"))?;
                let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                    .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
                println!(
                    "Persona '{}': {} writer(s)",
                    persona_name,
                    config.recipe.blend.len()
                );
                for entry in &config.recipe.blend {
                    match heartbit_ghost::corpus::Corpus::open_or_create(
                        &corpora_root,
                        &entry.writer,
                    ) {
                        Ok(c) if c.is_empty() => {
                            println!(
                                "  {} (weight {:.2}) — MISSING (no corpus on disk)",
                                entry.writer, entry.weight
                            );
                        }
                        Ok(c) => {
                            println!(
                                "  {} (weight {:.2}) — {} posts",
                                entry.writer,
                                entry.weight,
                                c.len()
                            );
                        }
                        Err(e) => {
                            println!(
                                "  {} (weight {:.2}) — ERROR: {e}",
                                entry.writer, entry.weight
                            );
                        }
                    }
                }
                Ok(())
            }
        },
        PersonaCommand::Profile { sub } => match sub {
            ProfileCommand::Rebuild { name: persona_name } => {
                if registry.get(&persona_name).is_none() {
                    return Err(anyhow!(
                        "persona '{persona_name}' not found. {}",
                        registry_suffix(registry)
                    ));
                }
                let config = heartbit_ghost::voice::PersonaConfig::load(&persona_name)
                    .map_err(|e| anyhow!("load persona config: {e}"))?;
                config
                    .recipe
                    .validate()
                    .map_err(|e| anyhow!("invalid recipe in persona config: {e}"))?;

                let provider = build_provider_from_env(None)
                    .map_err(|e| anyhow!("build llm provider: {e}"))?;
                let extractor = heartbit_ghost::voice::StyleExtractor::builder(provider).build();
                let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                    .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;

                let mut profiles: HashMap<String, heartbit_ghost::voice::StyleProfile> =
                    HashMap::new();
                for entry in &config.recipe.blend {
                    println!(
                        "extracting profile for '{}' (weight {:.2})...",
                        entry.writer, entry.weight
                    );
                    let corpus = heartbit_ghost::corpus::Corpus::open_or_create(
                        &corpora_root,
                        &entry.writer,
                    )
                    .map_err(|e| anyhow!("open corpus for '{}': {e}", entry.writer))?;
                    let profile = extractor
                        .extract(&corpus)
                        .await
                        .map_err(|e| anyhow!("extract profile for '{}': {e}", entry.writer))?;
                    profiles.insert(entry.writer.clone(), profile);
                }

                let merged = heartbit_ghost::voice::blend_profiles(&config.recipe, &profiles)
                    .map_err(|e| anyhow!("blend profiles: {e}"))?;

                let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                    .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;
                let store =
                    heartbit_ghost::voice::SnapshotStore::open(&profiles_root, &persona_name)
                        .map_err(|e| anyhow!("open snapshot store: {e}"))?;
                let new_version = store
                    .save_new(merged, &config.recipe)
                    .map_err(|e| anyhow!("save snapshot: {e}"))?;

                println!("ok: persona '{}' rebuilt as v{}", persona_name, new_version);
                Ok(())
            }
            ProfileCommand::Diff {
                name: persona_name,
                v1,
                v2,
            } => {
                if registry.get(&persona_name).is_none() {
                    return Err(anyhow!(
                        "persona '{persona_name}' not found. {}",
                        registry_suffix(registry)
                    ));
                }
                let v1_n = parse_version(&v1)?;
                let v2_n = parse_version(&v2)?;

                let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                    .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;
                let store =
                    heartbit_ghost::voice::SnapshotStore::open(&profiles_root, &persona_name)
                        .map_err(|e| anyhow!("open snapshot store: {e}"))?;
                let s1 = store.load(v1_n).map_err(|e| anyhow!("load v{v1_n}: {e}"))?;
                let s2 = store.load(v2_n).map_err(|e| anyhow!("load v{v2_n}: {e}"))?;

                let diff = heartbit_ghost::voice::ProfileDiff::compute(&s1.profile, &s2.profile);
                println!(
                    "{}",
                    heartbit_ghost::voice::render_profile_diff(&diff, &s1.meta, &s2.meta)
                );
                Ok(())
            }
        },
    }
}

/// Parse a `vN` or `N` argument as a u32.
fn parse_version(arg: &str) -> Result<u32> {
    arg.strip_prefix('v')
        .unwrap_or(arg)
        .parse::<u32>()
        .map_err(|_| anyhow!("expected version like 'v3' or '3', got '{arg}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn list_against_empty_registry_prints_message() {
        let r = PersonaRegistry::new();
        let result = dispatch(PersonaCommand::List, &r).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn show_against_empty_registry_returns_error() {
        let r = PersonaRegistry::new();
        let result = dispatch(PersonaCommand::Show { name: "x".into() }, &r).await;
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("persona 'x' not found"));
        assert!(msg.contains("No personas registered"));
    }

    #[tokio::test]
    async fn corpus_add_against_empty_registry_returns_error() {
        let r = PersonaRegistry::new();
        let result = dispatch(
            PersonaCommand::Corpus {
                sub: CorpusCommand::Add {
                    writer: "karpathy".into(),
                    path: std::path::PathBuf::from("/tmp/x.jsonl"),
                },
            },
            &r,
        )
        .await;
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("No personas registered"));
    }

    #[tokio::test]
    async fn profile_rebuild_against_empty_registry_returns_error() {
        let r = PersonaRegistry::new();
        let result = dispatch(
            PersonaCommand::Profile {
                sub: ProfileCommand::Rebuild { name: "x".into() },
            },
            &r,
        )
        .await;
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("No personas registered"));
    }

    #[tokio::test]
    async fn show_unknown_persona_with_registered_persona_lists_available() {
        // Manually populate the registry with the heartbit-ghost stub, then
        // ask for a name that isn't there; the error must surface the
        // available persona name(s).
        let mut r = PersonaRegistry::new();
        heartbit_ghost::register(&mut r);
        let result = dispatch(
            PersonaCommand::Show {
                name: "doesnotexist".into(),
            },
            &r,
        )
        .await;
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("persona 'doesnotexist' not found"));
        assert!(msg.contains("Available personas"), "got: {msg}");
        assert!(msg.contains("heartbit-ghost:x"), "got: {msg}");
        // Must NOT regress to the empty-registry hint when one IS registered.
        assert!(!msg.contains("No personas registered"));
    }

    #[tokio::test]
    async fn corpus_list_persona_not_found_returns_error() {
        let r = PersonaRegistry::new();
        let cmd = PersonaCommand::Corpus {
            sub: CorpusCommand::List {
                name: "no-such-persona".to_string(),
            },
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err(), "should error on missing persona");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
    }

    #[tokio::test]
    async fn profile_rebuild_persona_not_found_returns_error() {
        let r = PersonaRegistry::new();
        let cmd = PersonaCommand::Profile {
            sub: ProfileCommand::Rebuild {
                name: "no-such-persona".to_string(),
            },
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
    }

    #[tokio::test]
    async fn profile_diff_persona_not_found_returns_error() {
        let r = PersonaRegistry::new();
        let cmd = PersonaCommand::Profile {
            sub: ProfileCommand::Diff {
                name: "no-such-persona".to_string(),
                v1: "v1".to_string(),
                v2: "v2".to_string(),
            },
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
    }

    #[test]
    fn parse_version_accepts_v_prefix() {
        assert_eq!(parse_version("v3").unwrap(), 3);
        assert_eq!(parse_version("v0").unwrap(), 0);
        assert_eq!(parse_version("v100").unwrap(), 100);
    }

    #[test]
    fn parse_version_accepts_bare_number() {
        assert_eq!(parse_version("3").unwrap(), 3);
        assert_eq!(parse_version("0").unwrap(), 0);
    }

    #[test]
    fn parse_version_rejects_garbage() {
        let err = parse_version("not-a-version").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expected version"), "got: {msg}");
        assert!(msg.contains("not-a-version"), "got: {msg}");

        let err = parse_version("vfoo").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("vfoo"), "got: {msg}");
    }

    #[tokio::test]
    async fn corpus_list_unknown_persona_with_registered_persona_lists_available() {
        let mut r = PersonaRegistry::new();
        heartbit_ghost::register(&mut r);
        let cmd = PersonaCommand::Corpus {
            sub: CorpusCommand::List {
                name: "no-such-persona".to_string(),
            },
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
        assert!(msg.contains("Available personas"), "got: {msg}");
        assert!(msg.contains("heartbit-ghost:x"), "got: {msg}");
        assert!(!msg.contains("No personas registered"), "got: {msg}");
    }

    #[tokio::test]
    async fn profile_rebuild_unknown_persona_with_registered_persona_lists_available() {
        let mut r = PersonaRegistry::new();
        heartbit_ghost::register(&mut r);
        let cmd = PersonaCommand::Profile {
            sub: ProfileCommand::Rebuild {
                name: "no-such-persona".to_string(),
            },
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
        assert!(msg.contains("Available personas"), "got: {msg}");
        assert!(msg.contains("heartbit-ghost:x"), "got: {msg}");
        assert!(!msg.contains("No personas registered"), "got: {msg}");
    }

    #[tokio::test]
    async fn profile_diff_unknown_persona_with_registered_persona_lists_available() {
        let mut r = PersonaRegistry::new();
        heartbit_ghost::register(&mut r);
        let cmd = PersonaCommand::Profile {
            sub: ProfileCommand::Diff {
                name: "no-such-persona".to_string(),
                v1: "v1".to_string(),
                v2: "v2".to_string(),
            },
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
        assert!(msg.contains("Available personas"), "got: {msg}");
        assert!(msg.contains("heartbit-ghost:x"), "got: {msg}");
        assert!(!msg.contains("No personas registered"), "got: {msg}");
    }

    #[tokio::test]
    async fn run_persona_not_found_returns_error() {
        let r = PersonaRegistry::new();
        let cmd = PersonaCommand::Run {
            name: "no-such-persona".to_string(),
            once: "topic".to_string(),
            review: false,
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
    }

    #[tokio::test]
    async fn run_unknown_persona_with_registered_persona_lists_available() {
        let mut r = PersonaRegistry::new();
        heartbit_ghost::register(&mut r);
        let cmd = PersonaCommand::Run {
            name: "no-such-persona".to_string(),
            once: "topic".to_string(),
            review: false,
        };
        let result = dispatch(cmd, &r).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no-such-persona"), "got: {msg}");
        assert!(msg.contains("Available personas"), "got: {msg}");
        assert!(msg.contains("heartbit-ghost:x"), "got: {msg}");
        assert!(!msg.contains("No personas registered"), "got: {msg}");
    }

    /// The one-off `persona post` CLI must hard-error when
    /// HEARTBIT_GHOST_OPERATOR_USER_ID is missing — different contract from
    /// the supervised daemon. If a future refactor wires the fallback helper
    /// in here too, this test will need to be updated *and* the change
    /// reviewed against `docs/operating-heartbit.md`.
    #[test]
    fn persona_post_uses_strict_env_var_check_not_fallback_helper() {
        // Grep the source file for the canonical strict pattern. We assert
        // on a stable substring rather than the literal error message so
        // wording tweaks don't break the test.
        let src = include_str!("persona.rs");
        assert!(
            src.contains(r#"std::env::var("HEARTBIT_GHOST_OPERATOR_USER_ID")"#),
            "strict env-var check removed from persona.rs"
        );
        assert!(
            src.contains("must be set for `persona post` without --topic"),
            "strict error message changed; if intentional, update the doc"
        );
        // Negative: the fallback helper must NOT be used from this file —
        // it would silently substitute a persona_mentions value, masking
        // a config typo from the operator running the one-off command.
        // Split the identifier so this assertion doesn't match itself.
        let fallback_fn = ["resolve", "_operator_user_id"].concat();
        assert!(
            !src.contains(&fallback_fn),
            "persona post must keep strict env-var contract — see Task 4 plan"
        );
    }
}
