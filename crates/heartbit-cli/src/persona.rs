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

use heartbit::PersonaRegistry;

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
        PersonaCommand::Show { name }
        | PersonaCommand::Run { name, .. }
        | PersonaCommand::Phase { name, .. }
        | PersonaCommand::Pause { name }
        | PersonaCommand::Resume { name }
        | PersonaCommand::ExportPreferences { name, .. }
        | PersonaCommand::Audit { name, .. } => {
            if registry.get(&name).is_none() {
                let suffix = registry_suffix(registry);
                return Err(anyhow!("persona '{name}' not found. {suffix}"));
            }
            // P1.0 ships the registration shell; subcommand bodies land in
            // later sub-phases (P1.1–P1.4) alongside the persona's tools,
            // voice modeling, and pipeline.
            Err(anyhow!(
                "persona '{name}': subcommand body is not yet implemented (P1.0 scaffolding stub). The persona is registered; its tools, voice modeling, and pipeline land in later sub-phases."
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
        assert!(msg.contains("Available personas: heartbit-ghost:x"));
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
}
