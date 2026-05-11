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

pub mod dedup;
pub mod prompts;
pub mod publish_gate;
pub mod style_render;
pub mod verdicts;

pub use publish_gate::{PublishGateError, check_publish_gate};
pub use style_render::render_style_profile_as_english;
pub use verdicts::{
    FactVerdict, JudgeVerdict, StyleVerdict, VerdictParseError, parse_critic_verdict,
    parse_fact_verdict, parse_judge_verdict,
};

/// One candidate draft produced by the writer→critic→fact_check chain.
#[derive(Debug, Clone)]
pub struct CandidateRecord {
    /// 0-based generation slot. Preserved across parallel scheduling so the
    /// caller can correlate with the order they were requested in. After
    /// dedup, indices in `PipelineOutput.candidates` are NOT contiguous —
    /// the `variant_index` field tells you the original generation slot.
    pub variant_index: usize,
    /// The draft text.
    pub draft: String,
    /// Style critic's score on the accepted draft (post-revise loop).
    pub style_match_score: f64,
    /// Number of revise iterations until pass (1..=3).
    pub revise_iterations: usize,
    /// Fact-check verdict on the draft.
    pub fact_check_verdict: FactVerdict,
    /// Tokens used by this candidate's writer + critic + fact_check chain.
    /// `run_pipeline` sums these across all candidates into
    /// `PipelineOutput.usage_summary`.
    pub usage: TokenUsage,
}

/// Optional image attached to the chosen candidate by `image_generator`.
#[derive(Debug, Clone)]
pub struct ImageAttachment {
    /// Image URL returned by the `image_generate` tool.
    pub url: String,
    /// Optional alt text for accessibility.
    pub alt_text: Option<String>,
}

/// Progress callback type — invoked with a short status string at each
/// pipeline stage start. Used by `PipelineConfig::on_progress`.
pub type ProgressCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// Researcher override — `(recipe, tools)` pair that, when supplied,
/// replaces the pipeline's default `researcher_recipe()` +
/// `[WebSearchTool, WebFetchTool]`. The `Arc<AgentConfig>` is required
/// because `AgentConfig` does not derive `Clone` while
/// `PipelineConfig` does.
pub type ResearcherOverride = (Arc<AgentConfig>, Vec<Arc<dyn Tool>>);

/// Configuration for one pipeline run.
#[derive(Clone)]
pub struct PipelineConfig<'a> {
    /// Persona instance name (used to load the StyleProfile snapshot).
    pub persona_name: &'a str,
    /// Topic / prompt for this run.
    pub topic: &'a str,
    /// LLM provider (shared across all sub-agents).
    pub provider: Arc<BoxedProvider>,
    /// Corpora root (currently unused; reserved for P1.3e few-shot retrieval).
    pub corpora_root: &'a Path,
    /// Profiles root (passed to SnapshotStore::open).
    pub profiles_root: &'a Path,
    /// Optional progress callback. Called with a short status string at each
    /// pipeline stage start.
    pub on_progress: Option<ProgressCallback>,
    /// Number of distinct candidate drafts to generate. Default: 3.
    /// Validated `1..=10` at the start of `run_pipeline`. Set to 1 to
    /// recover the P1.3b single-candidate behavior (judge skipped).
    pub candidates_per_draft: usize,
    /// Persona-specific mode addendum surfaced in the writer's user
    /// message after voice_guidelines. None for personas that don't
    /// have one (heartbit-ghost:x).
    pub mode_addendum: Option<&'a str>,
    /// Override the default researcher (recipe + tools). When `Some`,
    /// the pipeline uses these instead of the legacy
    /// `researcher_recipe()` plus `WebSearchTool`+`WebFetchTool`.
    /// heartbit-rs:x supplies its `repo_researcher_recipe()` plus
    /// `RepoInspectTool` here so the agent is forced to read the local
    /// repo instead of the public web.
    pub researcher_override: Option<ResearcherOverride>,
}

/// Output of a successful pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// The final post draft (single tweet or `\n\n`-separated thread).
    /// Equals `candidates[chosen_index].draft`.
    pub final_draft: String,
    /// Researcher's digest text.
    pub research_digest: String,
    /// `style_match_score` from the critic on the chosen draft.
    /// Equals `candidates[chosen_index].style_match_score`.
    pub style_match_score: f64,
    /// Number of writer iterations until pass on the chosen draft (1..=3).
    /// Equals `candidates[chosen_index].revise_iterations`.
    pub revise_iterations: usize,
    /// Fact-check verdict on the chosen draft. Equals
    /// `candidates[chosen_index].fact_check_verdict`.
    pub fact_check_verdict: FactVerdict,
    /// Accumulated token usage across all sub-agent calls.
    pub usage_summary: TokenUsage,
    /// All distinct candidate drafts (1..=`candidates_per_draft` after dedup).
    pub candidates: Vec<CandidateRecord>,
    /// Index into `candidates` of the chosen draft. Validated `0..len`.
    pub chosen_index: usize,
    /// Judge's reasoning string. The literal sentinel
    /// `"single candidate, no ranking needed"` when only one candidate was
    /// ranked (judge skipped — see AD-8). Otherwise the judge's free-form
    /// explanation of why the chosen candidate beats the others.
    pub judge_reasoning: String,
    /// Image attached to the chosen draft, if `image_generator` decided
    /// to generate one. `None` when the recipe returned `"no_image"` or
    /// when the call failed (failures are non-blocking).
    pub image: Option<ImageAttachment>,
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

    /// `style_critic` returned a malformed verdict.
    #[error("style_critic verdict parse: {source}")]
    CriticParseFailed {
        /// The wrapped verdict-parse error (carries `raw` internally).
        #[source]
        source: VerdictParseError,
    },

    /// `fact_check` returned a malformed verdict.
    #[error("fact_check verdict parse: {source}")]
    FactCheckParseFailed {
        /// The wrapped verdict-parse error (carries `raw` internally).
        #[source]
        source: VerdictParseError,
    },

    /// `judge` returned a malformed verdict or out-of-range `chosen_index`.
    #[error("judge verdict parse: {source}")]
    JudgeParseFailed {
        /// The wrapped verdict-parse error (carries `raw` internally).
        #[source]
        source: VerdictParseError,
    },

    /// `style_critic` returned `Reject` — draft is fundamentally off.
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

    /// `publish_gate` rejected the chosen draft.
    #[error("publish_gate: {0}")]
    PublishGate(#[from] PublishGateError),

    /// Two or more candidate generation tasks collected errors and zero
    /// candidates survived. Single-error collapses (typical of
    /// `candidates_per_draft: 1`) surface the underlying error directly
    /// instead — see the `errors.swap_remove(0)` shortcut at the top of
    /// the JoinSet collection path in `run_pipeline`.
    #[error("all {n} candidates failed: {errors:?}")]
    AllCandidatesFailed {
        /// Per-candidate errors collected from the JoinSet.
        errors: Vec<PipelineError>,
        /// Number of candidates attempted.
        n: usize,
    },

    /// `PipelineConfig` validation failed at run start.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
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
pub fn runner_from_recipe(
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

/// Parse the image_generator's output. Returns `None` for `"no_image"` or
/// when no URL is recoverable. Tries JSON first, falls back to extracting
/// the first http(s):// URL substring.
pub(crate) fn parse_image_generator_output(raw: &str) -> Option<ImageAttachment> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.eq_ignore_ascii_case("no_image") {
        return None;
    }
    // Try JSON shape first: {"url": "...", "alt_text": "..."}.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        let url = value.get("url").and_then(|v| v.as_str()).map(String::from);
        let alt_text = value
            .get("alt_text")
            .and_then(|v| v.as_str())
            .map(String::from);
        if let Some(url) = url {
            return Some(ImageAttachment { url, alt_text });
        }
    }
    // Fallback: extract first http(s):// URL.
    let lower = trimmed.to_lowercase();
    let http_idx = lower.find("http://");
    let https_idx = lower.find("https://");
    let start = match (http_idx, https_idx) {
        (Some(h), Some(s)) => Some(h.min(s)),
        (Some(h), None) => Some(h),
        (None, Some(s)) => Some(s),
        (None, None) => None,
    }?;
    let rest = &trimmed[start..];
    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    let url = rest[..end].trim_end_matches(['.', ',', ';']).to_string();
    let surrounding = trimmed.replacen(&url, "", 1).trim().to_string();
    Some(ImageAttachment {
        url,
        alt_text: if surrounding.is_empty() {
            None
        } else {
            Some(surrounding)
        },
    })
}

/// Drop near-duplicate candidates per `LEVENSHTEIN_DUPLICATE_THRESHOLD`.
/// Lower variant_index wins on collision (declaration-order tiebreak).
pub(crate) fn dedup_candidates(candidates: Vec<CandidateRecord>) -> Vec<CandidateRecord> {
    if candidates.len() <= 1 {
        return candidates;
    }
    let drafts: Vec<&str> = candidates.iter().map(|c| c.draft.as_str()).collect();
    let kept = dedup::distinct_indices(&drafts, dedup::LEVENSHTEIN_DUPLICATE_THRESHOLD);
    kept.into_iter().map(|i| candidates[i].clone()).collect()
}

/// Run one writer→style_critic (revise loop, max 3 iters)→fact_check
/// chain, producing one [`CandidateRecord`]. Each call is independent;
/// `run_pipeline` spawns N of these in parallel via `tokio::JoinSet`
/// (Task 3). For Task 2's intermediate state, `run_pipeline` calls this
/// once with `(0, 1)` to preserve P1.3b single-candidate behavior.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn generate_candidate(
    variant_idx: usize,
    total_variants: usize,
    topic: &str,
    research_digest: &str,
    voice_guidelines: &str,
    writer: &AgentRunner<BoxedProvider>,
    critic: &AgentRunner<BoxedProvider>,
    fact: &AgentRunner<BoxedProvider>,
    mode_addendum: Option<&str>,
    exemplar_block: Option<&str>,
) -> Result<CandidateRecord, PipelineError> {
    // Revise loop.
    let mut prev_revision: Option<(String, String)> = None;
    let mut final_state: Option<(String, f64, usize)> = None;
    let mut last_score: f64 = 0.0;
    let mut total_usage = TokenUsage::default();

    for iter in 1..=3usize {
        let writer_msg = prompts::build_writer_user_message(
            topic,
            research_digest,
            voice_guidelines,
            prev_revision.as_ref(),
            variant_idx,
            total_variants,
            mode_addendum,
            exemplar_block,
        );
        let writer_out = writer
            .execute(&writer_msg)
            .await
            .map_err(|e| PipelineError::Agent {
                stage: format!("writer (variant {variant_idx}, iter {iter})"),
                source: e,
            })?;
        let draft = writer_out.result.clone();
        total_usage += writer_out.tokens_used;

        let critic_msg = prompts::build_critic_user_message(&draft, voice_guidelines);
        let critic_out = critic
            .execute(&critic_msg)
            .await
            .map_err(|e| PipelineError::Agent {
                stage: format!("style_critic (variant {variant_idx}, iter {iter})"),
                source: e,
            })?;
        total_usage += critic_out.tokens_used;
        let verdict = parse_critic_verdict(&critic_out.result)
            .map_err(|source| PipelineError::CriticParseFailed { source })?;
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

    let (draft, style_match_score, revise_iterations) = match final_state {
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

    // Fact-check (non-blocking on Unverifiable).
    let fact_msg = prompts::build_fact_user_message(&draft, research_digest);
    let fact_out = fact
        .execute(&fact_msg)
        .await
        .map_err(|e| PipelineError::Agent {
            stage: format!("fact_check (variant {variant_idx})"),
            source: e,
        })?;
    total_usage += fact_out.tokens_used;
    let fact_check_verdict = parse_fact_verdict(&fact_out.result)
        .map_err(|source| PipelineError::FactCheckParseFailed { source })?;

    Ok(CandidateRecord {
        variant_index: variant_idx,
        draft,
        style_match_score,
        revise_iterations,
        fact_check_verdict,
        usage: total_usage,
    })
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

    // Validate config (after snapshot load so missing snapshot reports
    // `NoProfileSnapshot` first, not `InvalidConfig`).
    if !(1..=10).contains(&cfg.candidates_per_draft) {
        return Err(PipelineError::InvalidConfig(format!(
            "candidates_per_draft must be in 1..=10 (got {})",
            cfg.candidates_per_draft,
        )));
    }

    // 2. Build the 6 AgentRunner instances from P1.3a recipes.
    use crate::agents::{
        fact_check_recipe, image_generator_recipe, judge_recipe, researcher_recipe,
        style_critic_recipe, writer_recipe,
    };
    use heartbit_core::tool::builtins::{ImageGenerateTool, WebFetchTool, WebSearchTool};

    // The researcher is the only agent that varies by persona today.
    // heartbit-ghost:x → default researcher_recipe() + [websearch, webfetch].
    // heartbit-rs:x   → repo_researcher_recipe() + [repo_inspect] (via override).
    let (researcher_recipe_used, researcher_tools): (AgentConfig, Vec<Arc<dyn Tool>>) =
        match cfg.researcher_override.as_ref() {
            Some((recipe, tools)) => ((**recipe).clone_config(), tools.clone()),
            None => (
                researcher_recipe(),
                vec![
                    Arc::new(WebSearchTool::new()),
                    Arc::new(WebFetchTool::new()),
                ],
            ),
        };
    let researcher = runner_from_recipe(
        cfg.provider.clone(),
        researcher_recipe_used,
        researcher_tools,
    )
    .map_err(|e| PipelineError::Builder {
        stage: "researcher".to_string(),
        source: e,
    })?;
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
    let judge =
        runner_from_recipe(cfg.provider.clone(), judge_recipe(), Vec::new()).map_err(|e| {
            PipelineError::Builder {
                stage: "judge".to_string(),
                source: e,
            }
        })?;
    let image_gen_tools: Vec<Arc<dyn Tool>> = vec![Arc::new(ImageGenerateTool::new())];
    let image_generator = runner_from_recipe(
        cfg.provider.clone(),
        image_generator_recipe(),
        image_gen_tools,
    )
    .map_err(|e| PipelineError::Builder {
        stage: "image_generator".to_string(),
        source: e,
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

    // 5. Parallel candidate generation (N tasks via tokio::JoinSet).
    let n = cfg.candidates_per_draft;
    progress(&format!("Generating {n} candidate(s) in parallel..."));

    // Wrap shared state for spawn closures (Send + 'static requirements).
    let writer = std::sync::Arc::new(writer);
    let critic = std::sync::Arc::new(critic);
    let fact = std::sync::Arc::new(fact);
    let topic_owned: String = cfg.topic.to_string();
    let digest_owned = std::sync::Arc::new(research_digest.clone());
    let voice_owned = std::sync::Arc::new(voice_guidelines.clone());
    // Convert to owned Arc<str> so the spawn closures are 'static.
    let mode_addendum_owned: Option<std::sync::Arc<str>> =
        cfg.mode_addendum.map(std::sync::Arc::from);

    let mut joinset: tokio::task::JoinSet<Result<CandidateRecord, PipelineError>> =
        tokio::task::JoinSet::new();
    for i in 0..n {
        let writer = writer.clone();
        let critic = critic.clone();
        let fact = fact.clone();
        let topic = topic_owned.clone();
        let digest = digest_owned.clone();
        let voice = voice_owned.clone();
        let mode_addendum = mode_addendum_owned.clone();
        joinset.spawn(async move {
            generate_candidate(
                i,
                n,
                &topic,
                &digest,
                &voice,
                &writer,
                &critic,
                &fact,
                mode_addendum.as_deref(),
                None, // standalone path: no engagement-driven exemplar block
            )
            .await
        });
    }

    let mut candidates: Vec<CandidateRecord> = Vec::with_capacity(n);
    let mut errors: Vec<PipelineError> = Vec::new();
    while let Some(res) = joinset.join_next().await {
        match res {
            Ok(Ok(rec)) => candidates.push(rec),
            Ok(Err(e)) => {
                progress(&format!("candidate failed: {e}"));
                errors.push(e);
            }
            Err(joinerr) => progress(&format!("candidate task panicked: {joinerr}")),
        }
    }
    candidates.sort_by_key(|c| c.variant_index);

    if candidates.is_empty() {
        // For N=1 (P1.3b back-compat) or any case that collapses to a
        // single error, surface the underlying PipelineError directly.
        // AllCandidatesFailed is reserved for the genuine N>1 collapse.
        if errors.len() == 1 {
            return Err(errors.swap_remove(0));
        }
        return Err(PipelineError::AllCandidatesFailed { errors, n });
    }

    // 6. Dedup + retry-once on collapse.
    let mut survivors = dedup_candidates(candidates);
    if survivors.len() < n {
        let missing = n - survivors.len();
        progress(&format!(
            "candidates collapsed ({missing} duplicates) — refilling once"
        ));
        let next_idx = survivors.iter().map(|c| c.variant_index).max().unwrap_or(0) + 1;

        let mut joinset2: tokio::task::JoinSet<Result<CandidateRecord, PipelineError>> =
            tokio::task::JoinSet::new();
        for offset in 0..missing {
            let i = next_idx + offset;
            let writer = writer.clone();
            let critic = critic.clone();
            let fact = fact.clone();
            let topic = topic_owned.clone();
            let digest = digest_owned.clone();
            let voice = voice_owned.clone();
            let mode_addendum = mode_addendum_owned.clone();
            joinset2.spawn(async move {
                generate_candidate(
                    i,
                    n,
                    &topic,
                    &digest,
                    &voice,
                    &writer,
                    &critic,
                    &fact,
                    mode_addendum.as_deref(),
                    None, // standalone path: no engagement-driven exemplar block
                )
                .await
            });
        }
        while let Some(res) = joinset2.join_next().await {
            if let Ok(Ok(rec)) = res {
                survivors.push(rec);
            }
            // Retry failures are silent — we already have `survivors.len() >= 1`,
            // so ship-with-fewer is acceptable.
        }
        survivors.sort_by_key(|c| c.variant_index);
        survivors = dedup_candidates(survivors);

        if survivors.len() < n {
            progress(&format!(
                "ship-with-fewer: {} of {} distinct candidates after retry",
                survivors.len(),
                n,
            ));
        }
    }

    // Sum per-candidate usage into total.
    for c in &survivors {
        total_usage += c.usage;
    }

    // 7. Judge (skipped when N=1).
    let chosen_index: usize;
    let judge_reasoning: String;
    if survivors.len() == 1 {
        chosen_index = 0;
        judge_reasoning = "single candidate, no ranking needed".to_string();
        progress("Single candidate — judge skipped.");
    } else {
        progress("Judging candidates...");
        let judge_msg = prompts::build_judge_user_message(cfg.topic, &voice_guidelines, &survivors);
        let judge_out = judge
            .execute(&judge_msg)
            .await
            .map_err(|e| PipelineError::Agent {
                stage: "judge".to_string(),
                source: e,
            })?;
        total_usage += judge_out.tokens_used;
        let verdict = parse_judge_verdict(&judge_out.result, survivors.len())
            .map_err(|source| PipelineError::JudgeParseFailed { source })?;
        chosen_index = verdict.chosen_index;
        judge_reasoning = verdict.reasoning;
    }

    let chosen = &survivors[chosen_index];
    let final_draft = chosen.draft.clone();
    let style_match_score = chosen.style_match_score;
    let revise_iterations = chosen.revise_iterations;
    let fact_check_verdict = chosen.fact_check_verdict.clone();

    if let FactVerdict::Unverifiable { ref reason } = fact_check_verdict {
        progress(&format!("fact_check unverifiable: {reason}"));
    }

    // 8. image_generator on chosen draft (always runs; recipe decides "no_image").
    progress("Generating optional image...");
    let image_msg = prompts::build_image_generator_user_message(&final_draft, &voice_guidelines);
    let image: Option<ImageAttachment> = match image_generator.execute(&image_msg).await {
        Ok(out) => {
            total_usage += out.tokens_used;
            // P1.3g: prefer the raw `image_generate` tool output (full,
            // untruncated marker) over the model's text response. The
            // tool result that re-entered the conversation was redacted
            // to a placeholder, so falling back to `out.result` would
            // lose the base64 payload. The fallback is only useful when
            // the agent declined to call the tool (e.g. "no_image").
            let raw_tool_output: Option<String> = out
                .tool_call_results
                .iter()
                .find(|r| r.tool_name == "image_generate" && !r.is_error)
                .map(|r| r.output.clone());
            match raw_tool_output {
                Some(raw) => parse_image_generator_output(&raw),
                None => parse_image_generator_output(&out.result),
            }
        }
        Err(e) => {
            progress(&format!("image_generator failed (non-blocking): {e}"));
            None
        }
    };

    // 9. publish_gate on chosen draft.
    progress("Running publish_gate...");
    check_publish_gate(&final_draft, &profile)?;

    // 10. Print + return.
    println!("{final_draft}");
    progress("Done.");
    Ok(PipelineOutput {
        final_draft,
        research_digest,
        style_match_score,
        revise_iterations,
        fact_check_verdict,
        usage_summary: total_usage,
        candidates: survivors,
        chosen_index,
        judge_reasoning,
        image,
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
        CompletionRequest, CompletionResponse, ContentBlock, Role, StopReason,
    };
    use std::future::Future;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// MockProvider with per-agent routing. Responses are keyed by a
    /// substring that uniquely identifies the requester (matched against
    /// the system prompt). First non-empty key with a substring match
    /// wins; the empty `""` key is the wildcard fallback for back-compat
    /// with single-queue callers.
    ///
    /// Per-route queues let parallel sub-agents (e.g. 3 writers spawned
    /// concurrently via `tokio::JoinSet`) all draw from the same
    /// "writer" queue without interleaving with the critic / fact_check
    /// / judge / image_generator queues.
    ///
    /// The optional `recorded_user_messages` field captures the first
    /// user-message text from every `complete()` call. Used by tests that
    /// need to verify what the LLM actually received.
    struct MockProvider {
        routes: Mutex<Vec<(String, std::collections::VecDeque<String>)>>,
        recorded_user_messages: Arc<Mutex<Vec<String>>>,
    }

    impl MockProvider {
        /// Construct with explicit per-route queues. Each route is a
        /// `(substring_key, responses)` pair. The substring is matched
        /// against the request's system prompt.
        fn route(routes: Vec<(&str, Vec<&str>)>) -> Arc<BoxedProvider> {
            let (provider, _) = Self::route_with_recorder(routes);
            provider
        }

        /// Like `route`, but also returns the recorder so the caller can
        /// inspect what user messages were received by the mock.
        fn route_with_recorder(
            routes: Vec<(&str, Vec<&str>)>,
        ) -> (Arc<BoxedProvider>, Arc<Mutex<Vec<String>>>) {
            let mapped: Vec<(String, std::collections::VecDeque<String>)> = routes
                .into_iter()
                .map(|(key, responses)| {
                    (
                        key.to_string(),
                        responses.into_iter().map(String::from).collect(),
                    )
                })
                .collect();
            let recorder: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let p = MockProvider {
                routes: Mutex::new(mapped),
                recorded_user_messages: Arc::clone(&recorder),
            };
            (Arc::new(BoxedProvider::new(p)), recorder)
        }

        /// Backward-compat helper: single-queue version. Internally
        /// routes everything to the wildcard `""` key.
        fn arc(responses: Vec<&str>) -> Arc<BoxedProvider> {
            Self::route(vec![("", responses)])
        }
    }

    impl LlmProvider for MockProvider {
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CoreError>> + Send {
            // Record the first user-message text for assertion in tests that
            // use `route_with_recorder`.
            let user_text = request
                .messages
                .iter()
                .find(|m| m.role == Role::User)
                .and_then(|m| m.content.first())
                .and_then(|c| match c {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            self.recorded_user_messages.lock().unwrap().push(user_text);

            // Find which route this request belongs to by matching the
            // system prompt against route keys. First non-empty key with
            // a substring match wins; "" key is the wildcard fallback.
            let system = request.system.as_str();
            let mut routes = self.routes.lock().unwrap();
            let chosen_idx = routes
                .iter()
                .position(|(key, _)| !key.is_empty() && system.contains(key.as_str()))
                .or_else(|| routes.iter().position(|(key, _)| key.is_empty()));

            let response = chosen_idx.and_then(|i| routes[i].1.pop_front());
            // Drop the lock before constructing the future.
            drop(routes);

            // If the runner injected the synthetic `__respond__` tool (i.e. the
            // recipe set `response_schema`), the structured-output guard
            // requires a `ToolUse` block; otherwise plain `Text` is fine.
            let has_respond = request
                .tools
                .iter()
                .any(|t| t.name == heartbit_core::llm::types::RESPOND_TOOL_NAME);
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

    /// Drive `run_pipeline` with `mode_addendum = Some("FRAMEWORK_DEMO_FIXTURE")` and
    /// assert the writer's LLM call actually receives that string in its user message.
    ///
    /// This exercises the full threading path:
    ///   PipelineConfig.mode_addendum
    ///   → spawn-loop Arc<str> conversion (mode_addendum_owned.as_deref())
    ///   → generate_candidate(mode_addendum)
    ///   → build_writer_user_message(mode_addendum)
    ///   → CompletionRequest.messages[0] received by the mock.
    ///
    /// A regression at any link (field dropped, Arc deref wrong, param ignored,
    /// builder omitted) causes the assertion to fail.
    #[tokio::test]
    async fn mode_addendum_some_value_appears_in_writer_user_message() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let (provider, recorder) = MockProvider::route_with_recorder(vec![
            ("research analyst", vec!["Research digest:\n- AI notes"]),
            ("social media writer", vec!["concrete short post"]),
            (
                "score how well a draft post",
                vec![r#"{"verdict": "pass", "style_match_score": 0.92}"#],
            ),
            (
                "verify the factual claims",
                vec![r#"{"verdict": "verified"}"#],
            ),
            ("produce an image to accompany", vec!["no_image"]),
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "addendum test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 1,
            mode_addendum: Some("FRAMEWORK_DEMO_FIXTURE"),
            researcher_override: None,
        };
        run_pipeline(cfg).await.expect("pipeline should succeed");
        let received = recorder.lock().unwrap();
        assert!(
            received
                .iter()
                .any(|msg| msg.contains("FRAMEWORK_DEMO_FIXTURE")),
            "expected at least one LLM call to contain 'FRAMEWORK_DEMO_FIXTURE'; \
             recorded messages: {received:#?}",
        );
    }

    /// Complementary sanity check: when `mode_addendum` is `None` the fixture
    /// string must NOT appear anywhere in the received messages.
    #[tokio::test]
    async fn mode_addendum_none_omits_addendum_from_writer_user_message() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let (provider, recorder) = MockProvider::route_with_recorder(vec![
            ("research analyst", vec!["Research digest:\n- AI notes"]),
            ("social media writer", vec!["concrete short post"]),
            (
                "score how well a draft post",
                vec![r#"{"verdict": "pass", "style_match_score": 0.92}"#],
            ),
            (
                "verify the factual claims",
                vec![r#"{"verdict": "verified"}"#],
            ),
            ("produce an image to accompany", vec!["no_image"]),
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "addendum test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 1,
            mode_addendum: None,
            researcher_override: None,
        };
        run_pipeline(cfg).await.expect("pipeline should succeed");
        let received = recorder.lock().unwrap();
        assert!(
            received
                .iter()
                .all(|msg| !msg.contains("FRAMEWORK_DEMO_FIXTURE")),
            "expected no LLM call to contain 'FRAMEWORK_DEMO_FIXTURE' when addendum is None; \
             recorded messages: {received:#?}",
        );
    }

    #[tokio::test]
    async fn run_pipeline_happy_path_single_iteration() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- AI is moving fast", // researcher
            "concrete short post",                   // writer iter 1
            r#"{"verdict": "pass", "style_match_score": 0.92}"#, // critic iter 1
            r#"{"verdict": "verified"}"#,            // fact_check
            "no_image", // image_generator (single-candidate path still calls it)
        ]);
        let corpora = profiles_root.clone();
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "AI capabilities",
            provider,
            corpora_root: &corpora,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 1,
            mode_addendum: None,
            researcher_override: None,
        };
        let out = run_pipeline(cfg).await.expect("happy path");
        assert_eq!(out.final_draft, "concrete short post");
        assert_eq!(out.revise_iterations, 1);
        assert!((out.style_match_score - 0.92).abs() < 1e-9);
        assert_eq!(out.fact_check_verdict, FactVerdict::Verified);
        assert_eq!(out.candidates.len(), 1, "single-candidate path");
        assert_eq!(out.chosen_index, 0);
        assert_eq!(out.judge_reasoning, "single candidate, no ranking needed");
        assert!(out.image.is_none(), "image_generator returned no_image");
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
            "no_image",                   // image_generator
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "Rust async patterns",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 1,
            mode_addendum: None,
            researcher_override: None,
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
            candidates_per_draft: 1,
            mode_addendum: None,
            researcher_override: None,
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
            candidates_per_draft: 1,
            mode_addendum: None,
            researcher_override: None,
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
            candidates_per_draft: 1,
            mode_addendum: None,
            researcher_override: None,
        };
        let err = run_pipeline(cfg).await.unwrap_err();
        match err {
            PipelineError::NoProfileSnapshot { persona, .. } => {
                assert_eq!(persona, "x");
            }
            other => panic!("expected NoProfileSnapshot, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_pipeline_three_candidates_judge_picks_index_1() {
        let (_dir, profiles_root) = seed_snapshot("x");
        // Per-agent routing avoids cross-agent FIFO interleaving across the
        // 3 parallel writer/critic/fact pipelines. Note: writer responses
        // are still consumed in queue order regardless of which variant
        // pops them — completion order across the 3 spawned tasks is
        // non-deterministic, so the variant→draft binding is non-deterministic.
        // We assert chosen_index (deterministic from the judge's canned
        // response) and `image.is_some()`, but accept ANY of the 3 drafts
        // as final_draft.
        let provider = MockProvider::route(vec![
            ("research analyst", vec!["Research digest:\n- topic notes"]),
            (
                "social media writer",
                vec![
                    "draft alpha distinct content",
                    "draft bravo with totally different framing",
                    "draft charlie via yet another angle",
                ],
            ),
            (
                "score how well a draft post",
                vec![
                    r#"{"verdict": "pass", "style_match_score": 0.80}"#,
                    r#"{"verdict": "pass", "style_match_score": 0.92}"#,
                    r#"{"verdict": "pass", "style_match_score": 0.85}"#,
                ],
            ),
            (
                "verify the factual claims",
                vec![
                    r#"{"verdict": "verified"}"#,
                    r#"{"verdict": "verified"}"#,
                    r#"{"verdict": "verified"}"#,
                ],
            ),
            (
                "rank N candidate drafts",
                vec![r#"{"chosen_index": 1, "reasoning": "bravo has more specific examples"}"#],
            ),
            (
                "produce an image to accompany",
                vec![r#"{"url": "https://example.com/img.png", "alt_text": "abstract"}"#],
            ),
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "diversity test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 3,
            mode_addendum: None,
            researcher_override: None,
        };
        let out = run_pipeline(cfg).await.expect("3-candidate happy path");
        assert_eq!(out.candidates.len(), 3);
        assert_eq!(out.chosen_index, 1);
        assert!(
            [
                "draft alpha distinct content",
                "draft bravo with totally different framing",
                "draft charlie via yet another angle",
            ]
            .contains(&out.final_draft.as_str()),
            "final_draft must be one of the canned writer responses, got: {}",
            out.final_draft,
        );
        assert!(out.judge_reasoning.contains("bravo"));
        assert!(out.image.is_some());
        let image = out.image.as_ref().unwrap();
        assert_eq!(image.url, "https://example.com/img.png");
    }

    #[tokio::test]
    async fn run_pipeline_collapse_then_refill_succeeds() {
        let (_dir, profiles_root) = seed_snapshot("x");
        // Variants 0 and 1 produce IDENTICAL drafts (collapse). Variant 2
        // is distinct. After dedup we have 2 distinct candidates. Refill
        // spawns 1 task for the missing slot; that task produces a 3rd
        // distinct draft. We use per-agent routing because the 3 parallel
        // critic / fact_check calls would otherwise pop responses in
        // non-deterministic order from a shared FIFO queue.
        let provider = MockProvider::route(vec![
            ("research analyst", vec!["Research digest:\n- topic notes"]),
            (
                "social media writer",
                vec![
                    // Variants 0 and 1 produce the same draft (collapse).
                    "duplicate draft text",
                    "duplicate draft text",
                    // Variant 2 distinct.
                    "completely different distinct draft",
                    // Refill produces a 3rd distinct draft.
                    "third distinct draft from refill",
                ],
            ),
            (
                "score how well a draft post",
                vec![
                    r#"{"verdict": "pass", "style_match_score": 0.80}"#,
                    r#"{"verdict": "pass", "style_match_score": 0.85}"#,
                    r#"{"verdict": "pass", "style_match_score": 0.90}"#,
                    r#"{"verdict": "pass", "style_match_score": 0.88}"#,
                ],
            ),
            (
                "verify the factual claims",
                vec![
                    r#"{"verdict": "verified"}"#,
                    r#"{"verdict": "verified"}"#,
                    r#"{"verdict": "verified"}"#,
                    r#"{"verdict": "verified"}"#,
                ],
            ),
            (
                "rank N candidate drafts",
                vec![r#"{"chosen_index": 0, "reasoning": "first one"}"#],
            ),
            ("produce an image to accompany", vec!["no_image"]),
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "collapse test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 3,
            mode_addendum: None,
            researcher_override: None,
        };
        let out = run_pipeline(cfg).await.expect("collapse+refill succeeds");
        assert_eq!(
            out.candidates.len(),
            3,
            "after refill we should have 3 distinct candidates"
        );
        assert!(out.image.is_none(), "no_image should produce None");
    }

    #[tokio::test]
    async fn run_pipeline_all_candidates_fail_returns_error() {
        let (_dir, profiles_root) = seed_snapshot("x");
        // All 3 candidates fail at the writer stage (mock exhausts after
        // researcher). The pipeline must return AllCandidatesFailed with
        // n=3 and at least one collected error.
        let provider = MockProvider::arc(vec![
            "Research digest:\n- topic notes",
            // Mock exhausts here. All 3 candidate writer.execute() calls fail
            // with Error::Agent("mock exhausted").
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "fail test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 3,
            mode_addendum: None,
            researcher_override: None,
        };
        let err = run_pipeline(cfg).await.unwrap_err();
        match err {
            PipelineError::AllCandidatesFailed { n, errors } => {
                assert_eq!(n, 3);
                assert!(!errors.is_empty(), "expected at least one collected error");
            }
            other => panic!("expected AllCandidatesFailed, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_pipeline_invalid_candidates_per_draft_rejected() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![]); // never reached
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "config test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 0, // invalid
            mode_addendum: None,
            researcher_override: None,
        };
        let err = run_pipeline(cfg).await.unwrap_err();
        match err {
            PipelineError::InvalidConfig(msg) => {
                assert!(msg.contains("candidates_per_draft"), "msg: {msg}");
                assert!(msg.contains("1..=10"), "msg: {msg}");
            }
            other => panic!("expected InvalidConfig, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_pipeline_image_generator_no_image_yields_none() {
        // Single candidate path so the test sequence is short. image_generator
        // returns "no_image"; PipelineOutput.image must be None and judge is
        // skipped (N=1).
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            "concrete post",
            r#"{"verdict": "pass", "style_match_score": 0.92}"#,
            r#"{"verdict": "verified"}"#,
            "no_image",
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "image-skip test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 1,
            mode_addendum: None,
            researcher_override: None,
        };
        let out = run_pipeline(cfg).await.expect("happy path with no_image");
        assert_eq!(out.candidates.len(), 1);
        assert!(out.image.is_none());
        // Sanity: judge was skipped because N=1.
        assert_eq!(out.judge_reasoning, "single candidate, no ranking needed");
    }
}
