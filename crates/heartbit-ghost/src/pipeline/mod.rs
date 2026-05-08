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
// NOTE: P1.3b only forwards the 6 fields above. Other AgentConfig fields
// (summarize_threshold, tool_profile, max_identical_tool_calls, etc.) are
// intentionally ignored for the single-candidate pipeline. P1.3c+ may
// extend this if needed.
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

/// Execute one generation pipeline run: research → write → critic
/// (with revise loop, max 3) → fact_check → publish_gate → stdout.
pub async fn run_pipeline(cfg: PipelineConfig<'_>) -> Result<PipelineOutput, PipelineError> {
    let progress = |msg: &str| {
        if let Some(cb) = cfg.on_progress.as_ref() {
            cb(msg);
        }
    };

    // 1. Load StyleProfile snapshot.
    progress("Loading profile snapshot...");
    let store = crate::voice::SnapshotStore::open(cfg.profiles_root, cfg.persona_name)?;
    let snapshot = store
        .load_latest()?
        .ok_or_else(|| PipelineError::NoProfileSnapshot {
            persona: cfg.persona_name.to_string(),
            profiles_dir: cfg.profiles_root.join(cfg.persona_name),
        })?;
    let profile = snapshot.profile;

    // 2. Build the 4 AgentRunner instances from P1.3a recipes.
    use crate::agents::{fact_check_recipe, researcher_recipe, style_critic_recipe, writer_recipe};
    use heartbit_core::tool::builtins::{WebFetchTool, WebSearchTool};

    let researcher_tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(WebSearchTool::new()),
        Arc::new(WebFetchTool::new()),
    ];
    let researcher =
        runner_from_recipe(cfg.provider.clone(), researcher_recipe(), researcher_tools).map_err(
            |e| PipelineError::Builder {
                stage: "researcher".to_string(),
                source: e,
            },
        )?;
    let writer =
        runner_from_recipe(cfg.provider.clone(), writer_recipe(), Vec::new()).map_err(|e| {
            PipelineError::Builder {
                stage: "writer".to_string(),
                source: e,
            }
        })?;
    let critic = runner_from_recipe(cfg.provider.clone(), style_critic_recipe(), Vec::new())
        .map_err(|e| PipelineError::Builder {
            stage: "style_critic".to_string(),
            source: e,
        })?;
    let fact =
        runner_from_recipe(cfg.provider.clone(), fact_check_recipe(), Vec::new()).map_err(|e| {
            PipelineError::Builder {
                stage: "fact_check".to_string(),
                source: e,
            }
        })?;

    let mut total_usage = TokenUsage::default();

    // 3. Researcher.
    progress("Researching topic...");
    let researcher_out = researcher
        .execute(cfg.topic)
        .await
        .map_err(|e| PipelineError::Agent {
            stage: "researcher".to_string(),
            source: e,
        })?;
    let research_digest = researcher_out.result.clone();
    total_usage += researcher_out.tokens_used;

    // 4. Render voice guidelines.
    let voice_guidelines = render_style_profile_as_english(&profile);

    // 5. Revise loop.
    let mut prev_revision: Option<(String, String)> = None;
    let mut final_state: Option<(String, f64, usize)> = None;
    let mut last_score: f64 = 0.0;

    for iter in 1..=3usize {
        progress(&format!("Drafting (iter {iter})..."));
        let writer_msg = prompts::build_writer_user_message(
            cfg.topic,
            &research_digest,
            &voice_guidelines,
            prev_revision.as_ref(),
        );
        let writer_out = writer
            .execute(&writer_msg)
            .await
            .map_err(|e| PipelineError::Agent {
                stage: format!("writer (iter {iter})"),
                source: e,
            })?;
        let draft = writer_out.result.clone();
        total_usage += writer_out.tokens_used;

        progress(&format!("Style-checking (iter {iter})..."));
        let critic_msg = prompts::build_critic_user_message(&draft, &voice_guidelines);
        let critic_out = critic
            .execute(&critic_msg)
            .await
            .map_err(|e| PipelineError::Agent {
                stage: format!("style_critic (iter {iter})"),
                source: e,
            })?;
        total_usage += critic_out.tokens_used;
        let verdict = parse_critic_verdict(&critic_out.result).map_err(|e| {
            PipelineError::CriticParseFailed {
                source: e,
                raw: critic_out.result.clone(),
            }
        })?;
        last_score = verdict.score();

        match verdict {
            StyleVerdict::Pass { score } => {
                final_state = Some((draft, score, iter));
                break;
            }
            StyleVerdict::Reject { reason, score } => {
                return Err(PipelineError::Rejected { reason, score });
            }
            StyleVerdict::Revise { reason, .. } => {
                prev_revision = Some((draft, reason));
                continue;
            }
        }
    }

    let (final_draft, score, iterations) = match final_state {
        Some(v) => v,
        None => {
            let (last_draft, last_reason) = prev_revision.unwrap_or_default();
            return Err(PipelineError::MaxRevisionsExceeded {
                iterations: 3,
                last_draft,
                last_reason,
                last_score,
            });
        }
    };

    // 6. fact_check (non-blocking on Unverifiable).
    progress("Fact-checking...");
    let fact_msg = prompts::build_fact_user_message(&final_draft, &research_digest);
    let fact_out = fact
        .execute(&fact_msg)
        .await
        .map_err(|e| PipelineError::Agent {
            stage: "fact_check".to_string(),
            source: e,
        })?;
    total_usage += fact_out.tokens_used;
    let fact_verdict =
        parse_fact_verdict(&fact_out.result).map_err(|e| PipelineError::FactCheckParseFailed {
            source: e,
            raw: fact_out.result.clone(),
        })?;
    if let FactVerdict::Unverifiable { ref reason } = fact_verdict {
        progress(&format!("fact_check unverifiable: {reason}"));
    }

    // 7. publish_gate.
    progress("Running publish_gate...");
    check_publish_gate(&final_draft, &profile)?;

    // 8. Print + return.
    println!("{final_draft}");
    progress("Done.");
    Ok(PipelineOutput {
        final_draft,
        research_digest,
        style_match_score: score,
        revise_iterations: iterations,
        fact_check_verdict: fact_verdict,
        usage_summary: total_usage,
    })
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

    use heartbit_core::llm::LlmProvider;
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason,
    };
    use std::future::Future;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// MockProvider returns a queue of canned responses, one per `complete()` call.
    /// Calls past the queue's end return `Error::Agent("mock exhausted")`.
    struct MockProvider {
        responses: Mutex<std::collections::VecDeque<String>>,
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
            // If the runner injected the synthetic `__respond__` tool (i.e. the
            // recipe set `response_schema`), the structured-output guard
            // requires a `ToolUse` block; otherwise plain `Text` is fine.
            let has_respond = request.tools.iter().any(|t| t.name == "__respond__");
            async move {
                let text =
                    response.ok_or_else(|| CoreError::Agent("mock exhausted".to_string()))?;
                let (content, stop_reason) = if has_respond {
                    let input: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                        CoreError::Agent(format!("mock: canned response is not valid JSON: {e}"))
                    })?;
                    (
                        vec![ContentBlock::ToolUse {
                            id: "respond_1".to_string(),
                            name: "__respond__".to_string(),
                            input,
                        }],
                        StopReason::ToolUse,
                    )
                } else {
                    (vec![ContentBlock::Text { text }], StopReason::EndTurn)
                };
                Ok(CompletionResponse {
                    content,
                    usage: TokenUsage::default(),
                    stop_reason,
                    model: None,
                })
            }
        }
    }

    /// Snapshot fixture — minimal valid StyleProfile + recipe, saved to TempDir.
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

    #[test]
    fn runner_from_recipe_maps_reasoning_effort_string_to_enum() {
        // The function should accept "high"/"medium"/"low" and silently
        // ignore "none", None, or any unknown string. We can't introspect
        // the built runner, but we can confirm none of these inputs panic
        // or return Err for the same baseline recipe.
        use crate::agents::writer_recipe;
        use heartbit_core::config::AgentConfig;

        let provider = MockProvider::arc(vec![]);

        for effort in [
            Some("high"),
            Some("medium"),
            Some("low"),
            Some("none"),
            Some("unknown"),
            None,
        ] {
            let mut recipe: AgentConfig = writer_recipe();
            recipe.reasoning_effort = effort.map(String::from);
            let result = runner_from_recipe(provider.clone(), recipe, Vec::new());
            assert!(
                result.is_ok(),
                "expected Ok for reasoning_effort={effort:?}; got: {:?}",
                result.err()
            );
        }
    }

    #[tokio::test]
    async fn run_pipeline_happy_path_single_iteration() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- AI is moving fast", // researcher
            "concrete short post",                   // writer iter 1
            r#"{"verdict": "pass", "style_match_score": 0.92}"#, // critic iter 1
            r#"{"verdict": "verified"}"#,            // fact_check
        ]);
        let corpora = profiles_root.clone();
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "AI capabilities",
            provider,
            corpora_root: &corpora,
            profiles_root: &profiles_root,
            on_progress: None,
        };
        let out = run_pipeline(cfg).await.expect("happy path");
        assert_eq!(out.final_draft, "concrete short post");
        assert_eq!(out.revise_iterations, 1);
        assert!((out.style_match_score - 0.92).abs() < 1e-9);
        assert_eq!(out.fact_check_verdict, FactVerdict::Verified);
    }

    #[tokio::test]
    async fn run_pipeline_revise_once_then_pass() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- topic notes",        // researcher
            "first draft with em-dashes — like this", // writer iter 1
            r#"{"verdict": "revise", "reason": "uses em-dashes", "style_match_score": 0.6}"#, // critic iter 1
            "second draft, no em-dashes", // writer iter 2
            r#"{"verdict": "pass", "style_match_score": 0.91}"#, // critic iter 2
            r#"{"verdict": "verified"}"#, // fact_check
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "Rust async patterns",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
        };
        let out = run_pipeline(cfg).await.expect("revise then pass");
        assert_eq!(out.final_draft, "second draft, no em-dashes");
        assert_eq!(out.revise_iterations, 2);
    }

    #[tokio::test]
    async fn run_pipeline_max_revisions_exceeded() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes", // researcher
            "draft 1",                   // writer iter 1
            r#"{"verdict": "revise", "reason": "off-voice (1)", "style_match_score": 0.5}"#, // critic iter 1
            "draft 2", // writer iter 2
            r#"{"verdict": "revise", "reason": "off-voice (2)", "style_match_score": 0.5}"#, // critic iter 2
            "draft 3", // writer iter 3
            r#"{"verdict": "revise", "reason": "off-voice (3)", "style_match_score": 0.5}"#, // critic iter 3
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "topic",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
        };
        let err = run_pipeline(cfg).await.unwrap_err();
        match err {
            PipelineError::MaxRevisionsExceeded {
                iterations,
                last_reason,
                ..
            } => {
                assert_eq!(iterations, 3);
                assert!(
                    last_reason.contains("(3)"),
                    "last reason should be from iter 3; got: {last_reason}"
                );
            }
            other => panic!("expected MaxRevisionsExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_pipeline_critic_reject_aborts() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes", // researcher
            "off-topic draft",           // writer iter 1
            r#"{"verdict": "reject", "reason": "off-topic", "style_match_score": 0.1}"#, // critic iter 1
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "topic",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
        };
        let err = run_pipeline(cfg).await.unwrap_err();
        match err {
            PipelineError::Rejected { reason, score } => {
                assert_eq!(reason, "off-topic");
                assert!((score - 0.1).abs() < 1e-9);
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_pipeline_no_profile_snapshot_returns_error() {
        // Don't seed a snapshot — the persona dir doesn't exist on disk.
        let dir = TempDir::new().unwrap();
        let provider = MockProvider::arc(vec![]); // never called
        let root = dir.path().to_path_buf();
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "topic",
            provider,
            corpora_root: &root,
            profiles_root: &root,
            on_progress: None,
        };
        let err = run_pipeline(cfg).await.unwrap_err();
        match err {
            PipelineError::NoProfileSnapshot { persona, .. } => {
                assert_eq!(persona, "x");
            }
            other => panic!("expected NoProfileSnapshot, got {other:?}"),
        }
    }
}
