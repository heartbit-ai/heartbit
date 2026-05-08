//! Generation pipeline — wires the P1.3a sub-agent recipes into a working
//! single-candidate path.
//!
//! Public entry: [`run_pipeline`] (added in Task 3).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use heartbit_core::agent::AgentRunner;
use heartbit_core::config::AgentConfig;
use heartbit_core::error::Error as CoreError;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::llm::types::{ReasoningEffort, TokenUsage};
use heartbit_core::tool::Tool;
use thiserror::Error;

use crate::voice::SnapshotError;

pub mod prompts;
pub mod publish_gate;
pub mod style_render;
pub mod verdicts;

pub use publish_gate::{PublishGateError, check_publish_gate};
pub use style_render::render_style_profile_as_english;
pub use verdicts::{FactVerdict, StyleVerdict, parse_critic_verdict, parse_fact_verdict};

/// Progress callback type — invoked with a short status string at each
/// pipeline stage start. Used by `PipelineConfig::on_progress`.
pub type ProgressCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// Configuration for one pipeline run.
pub struct PipelineConfig<'a> {
    /// Persona instance name (used to load the StyleProfile snapshot).
    pub persona_name: &'a str,
    /// Topic / prompt for this run.
    pub topic: &'a str,
    /// LLM provider (shared across all 4 sub-agents).
    pub provider: Arc<BoxedProvider>,
    /// Corpora root (currently unused; reserved for P1.3e few-shot retrieval).
    pub corpora_root: &'a Path,
    /// Profiles root (passed to SnapshotStore::open).
    pub profiles_root: &'a Path,
    /// Optional progress callback. Called with a short status string at each
    /// pipeline stage start.
    pub on_progress: Option<ProgressCallback>,
}

/// Output of a successful pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// The final post draft (single tweet or `\n\n`-separated thread).
    pub final_draft: String,
    /// Researcher's digest text.
    pub research_digest: String,
    /// `style_match_score` from the critic on the accepted draft.
    pub style_match_score: f64,
    /// Number of writer iterations until pass (1..=3).
    pub revise_iterations: usize,
    /// Fact-check verdict on the final draft.
    pub fact_check_verdict: FactVerdict,
    /// Accumulated token usage across all 4 agent calls.
    pub usage_summary: TokenUsage,
}

/// Errors raised by [`run_pipeline`].
#[derive(Debug, Error)]
pub enum PipelineError {
    /// No StyleProfile snapshot exists for this persona.
    #[error(
        "no profile snapshot for persona '{persona}' at {}; run `heartbit persona profile rebuild {persona}` first",
        profiles_dir.display()
    )]
    NoProfileSnapshot {
        /// Persona name passed in.
        persona: String,
        /// Resolved profiles directory path.
        profiles_dir: PathBuf,
    },

    /// SnapshotStore I/O / parse failure.
    #[error("snapshot: {0}")]
    Snapshot(#[from] SnapshotError),

    /// AgentRunner construction failed.
    #[error("agent builder for stage '{stage}': {source}")]
    Builder {
        /// Which stage's builder failed.
        stage: String,
        /// Underlying core error.
        #[source]
        source: CoreError,
    },

    /// Agent execution error (network, LLM error, etc.).
    #[error("agent execution at stage '{stage}': {source}")]
    Agent {
        /// Which stage's agent was running.
        stage: String,
        /// Underlying core error.
        #[source]
        source: CoreError,
    },

    /// style_critic returned a malformed verdict.
    #[error("style_critic verdict parse: {source}")]
    CriticParseFailed {
        /// The underlying serde_json error.
        #[source]
        source: serde_json::Error,
        /// The raw critic output that failed to parse.
        raw: String,
    },

    /// fact_check returned a malformed verdict.
    #[error("fact_check verdict parse: {source}")]
    FactCheckParseFailed {
        /// The underlying serde_json error.
        #[source]
        source: serde_json::Error,
        /// The raw fact_check output that failed to parse.
        raw: String,
    },

    /// style_critic returned `Reject` — draft is fundamentally off.
    #[error("style_critic rejected the draft: {reason}")]
    Rejected {
        /// Reason from the critic.
        reason: String,
        /// 0.0..=1.0 score.
        score: f64,
    },

    /// 3 revise iterations exhausted without `Pass`.
    #[error("revise loop exhausted after {iterations} iterations; last reason: {last_reason}")]
    MaxRevisionsExceeded {
        /// Number of iterations attempted.
        iterations: usize,
        /// The final draft produced (failed).
        last_draft: String,
        /// The last critic feedback.
        last_reason: String,
        /// 0.0..=1.0 score from the last iteration.
        last_score: f64,
    },

    /// publish_gate rejected the final draft.
    #[error("publish_gate: {0}")]
    PublishGate(#[from] PublishGateError),
}

/// Build an [`AgentRunner`] from a P1.3a [`AgentConfig`] recipe and a
/// (possibly empty) tool subset.
///
/// Maps `AgentConfig.{name, system_prompt, max_turns, max_tokens,
/// reasoning_effort, response_schema}` onto the corresponding builder
/// methods. The `description` field is metadata-only (not used at
/// runtime). Reasoning effort strings map: "high" → High, "medium" →
/// Medium, "low" → Low; absent or unknown → no `.reasoning_effort()` call.
#[allow(dead_code)] // Task 3 wires the caller (run_pipeline).
pub(crate) fn runner_from_recipe(
    provider: Arc<BoxedProvider>,
    recipe: AgentConfig,
    tools: Vec<Arc<dyn Tool>>,
) -> Result<AgentRunner<BoxedProvider>, CoreError> {
    let mut builder = AgentRunner::builder(provider)
        .name(recipe.name)
        .system_prompt(recipe.system_prompt)
        .tools(tools);
    if let Some(n) = recipe.max_turns {
        builder = builder.max_turns(n);
    }
    if let Some(n) = recipe.max_tokens {
        builder = builder.max_tokens(n);
    }
    if let Some(effort) = recipe.reasoning_effort.as_deref() {
        match effort {
            "high" => builder = builder.reasoning_effort(ReasoningEffort::High),
            "medium" => builder = builder.reasoning_effort(ReasoningEffort::Medium),
            "low" => builder = builder.reasoning_effort(ReasoningEffort::Low),
            _ => { /* unknown — leave default */ }
        }
    }
    if let Some(schema) = recipe.response_schema {
        builder = builder.structured_schema(schema);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_profile_snapshot_error_renders_with_persona_and_path() {
        let e = PipelineError::NoProfileSnapshot {
            persona: "x".to_string(),
            profiles_dir: PathBuf::from("/tmp/profiles"),
        };
        let s = format!("{e}");
        assert!(s.contains("'x'"), "got: {s}");
        assert!(s.contains("/tmp/profiles"), "got: {s}");
        assert!(s.contains("profile rebuild x"), "got: {s}");
    }

    #[test]
    fn rejected_error_renders_with_reason() {
        let e = PipelineError::Rejected {
            reason: "off-topic".to_string(),
            score: 0.2,
        };
        let s = format!("{e}");
        assert!(s.contains("off-topic"), "got: {s}");
        assert!(s.contains("rejected"), "got: {s}");
    }

    #[test]
    fn max_revisions_error_renders_with_iterations_and_reason() {
        let e = PipelineError::MaxRevisionsExceeded {
            iterations: 3,
            last_draft: "draft".to_string(),
            last_reason: "still off-voice".to_string(),
            last_score: 0.6,
        };
        let s = format!("{e}");
        assert!(s.contains("3 iterations"), "got: {s}");
        assert!(s.contains("still off-voice"), "got: {s}");
    }
}
