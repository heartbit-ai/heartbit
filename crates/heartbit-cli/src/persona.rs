//! `heartbit persona <sub>` subcommand surface.
//!
//! Functional shells against the `PersonaRegistry` populated at startup by
//! linked persona crates (e.g. `heartbit-ghost`). Once persona crates
//! register their recipes, these subcommands light up without any CLI
//! changes. P1.0 ships the registration shell; subcommand bodies land in
//! later sub-phases (P1.1–P1.4).

use anyhow::{Result, anyhow};
use clap::Subcommand;

use heartbit::PersonaRegistry;

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
                let available = registry.list();
                let suffix = if available.is_empty() {
                    NO_PERSONAS_REGISTERED.to_string()
                } else {
                    format!("Available personas: {}.", available.join(", "))
                };
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
            CorpusCommand::Add { .. } | CorpusCommand::List { .. } => Err(anyhow!(
                "corpus management requires a registered persona. {NO_PERSONAS_REGISTERED}"
            )),
        },
        PersonaCommand::Profile { sub } => match sub {
            ProfileCommand::Rebuild { .. } | ProfileCommand::Diff { .. } => Err(anyhow!(
                "profile management requires a registered persona. {NO_PERSONAS_REGISTERED}"
            )),
        },
    }
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
}
