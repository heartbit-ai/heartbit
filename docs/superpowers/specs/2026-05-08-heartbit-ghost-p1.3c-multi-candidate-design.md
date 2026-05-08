# heartbit-ghost P1.3c — Multi-candidate generation + judge ranking + image_generator + verdict-error refactor

**Status:** approved 2026-05-08
**Predecessor:** P1.3b (single-candidate pipeline) merged to `main` at `1f0ea2d`
**Branch:** `feat/heartbit-ghost-p1.3c` (created off `main`)
**Successor:** P1.3d — Telegram review delivery + publisher + autonomy phase 0

---

## 1. Goal

Extend `heartbit_ghost::pipeline::run_pipeline` to produce **N distinct candidate drafts** (default 3), rank them via the existing `judge` recipe, attach an optional image to the chosen draft via `image_generator`, and surface the full candidate set + ranking metadata in `PipelineOutput`. P1.3b's single-candidate behavior remains accessible by setting `candidates_per_draft: 1`.

Folded follow-up: wrap the `serde_json::Error` leak in `verdicts.rs::parse_*_verdict` behind a dedicated `VerdictParseError`, and add a third parser for the judge's structured output.

## 2. Architecture

```
research → ┐
           ├─ parallel × N (tokio::JoinSet) ─→ writer i  → style_critic i (revise loop, max 3) → fact_check i
           ┘                                            │
                                                        ▼
                                  Levenshtein dedup (>0.85 ratio = collapsed)
                                                        │
                                                        ▼
                                      refill missing slots (cap = 1 retry pass)
                                                        │
                                                        ▼
                                                  judge (skipped when N=1)
                                                        │
                                                        ▼
                                            image_generator on chosen draft
                                                        │
                                                        ▼
                                            publish_gate on chosen draft
                                                        │
                                                        ▼
                                            println! chosen draft to stdout
```

**Researcher** runs once (shared output across all candidates). **Writer→critic→fact_check** chains run in parallel (one per candidate slot) via `tokio::JoinSet`. **Judge / image_generator / publish_gate** are sequential downstream of all candidates. Per-candidate failures (LLM error, parse failure, fact-check `Unverifiable`) are non-blocking unless **all N candidates fail**, in which case `PipelineError::AllCandidatesFailed` aborts the run.

`fact_check`'s `Unverifiable` verdict remains non-blocking per AD-6 in P1.3b: candidates carrying `FactVerdict::Unverifiable` still enter the dedup/judge stages, but their reason is logged via `on_progress`.

## 3. Public API extensions

### 3.1 `PipelineConfig`

```rust
pub struct PipelineConfig<'a> {
    // existing fields unchanged from P1.3b...
    pub persona_name: &'a str,
    pub topic: &'a str,
    pub provider: Arc<BoxedProvider>,
    pub corpora_root: &'a Path,
    pub profiles_root: &'a Path,
    pub on_progress: Option<ProgressCallback>,

    // NEW in P1.3c:
    /// Number of distinct candidate drafts to generate. Default: 3.
    /// Set to 1 to recover P1.3b's single-candidate behavior (skips judge,
    /// image_generator dedup, etc., where they would be no-ops).
    /// Validated 1..=10 at the start of run_pipeline.
    pub candidates_per_draft: usize,
}
```

A builder/`Default` constructor sets `candidates_per_draft = 3`. Existing P1.3b call sites (CLI `persona run`, integration tests) update to either set the field explicitly or rely on `Default`.

### 3.2 `PipelineOutput`

```rust
pub struct PipelineOutput {
    // existing fields keep their meaning — they mirror the chosen candidate's values:
    pub final_draft: String,                  // = candidates[chosen_index].draft
    pub style_match_score: f64,               // = candidates[chosen_index].style_match_score
    pub revise_iterations: usize,             // = candidates[chosen_index].revise_iterations
    pub fact_check_verdict: FactVerdict,      // = candidates[chosen_index].fact_check_verdict
    pub research_digest: String,
    pub usage_summary: TokenUsage,

    // NEW in P1.3c:
    pub candidates: Vec<CandidateRecord>,     // 1..=candidates_per_draft after dedup
    pub chosen_index: usize,                  // index into `candidates`, validated 0..len
    pub judge_reasoning: String,              // empty when N=1 (judge skipped)
    pub image: Option<ImageAttachment>,       // image attached to the chosen draft
}

#[derive(Debug, Clone)]
pub struct CandidateRecord {
    /// 0-based slot index — preserved across parallel scheduling so
    /// `candidates[i].variant_index == i` is NOT guaranteed (post-dedup
    /// indices change). The `variant_index` field tells you the original
    /// generation slot.
    pub variant_index: usize,
    pub draft: String,
    pub style_match_score: f64,
    pub revise_iterations: usize,
    pub fact_check_verdict: FactVerdict,
}

#[derive(Debug, Clone)]
pub struct ImageAttachment {
    pub url: String,
    pub alt_text: Option<String>,
}
```

`image` lives on `PipelineOutput` (not `CandidateRecord`) because P1.3c only generates an image on the chosen draft. P1.3d may move it to `CandidateRecord` if Telegram review wants per-candidate images, but that's a future refactor.

`research_digest` and `usage_summary` carry over verbatim from P1.3b.

## 4. Sub-pipeline orchestration

### 4.1 New private helper

```rust
async fn generate_candidate(
    variant_idx: usize,
    total: usize,
    cfg: &PipelineConfig<'_>,
    research_digest: &str,
    voice_guidelines: &str,
    writer: &AgentRunner<BoxedProvider>,
    critic: &AgentRunner<BoxedProvider>,
    fact: &AgentRunner<BoxedProvider>,
) -> Result<CandidateRecord, PipelineError>;
```

Body = current P1.3b revise loop, with two changes:

1. **Variant-aware writer prompt:** when `total > 1`, the writer's user message gains:
   ```
   You are generating variant {variant_idx + 1} of {total}. Pursue a
   distinct angle from the other variants — emphasize different
   aspects, examples, or framing.
   ```
   (Skipped when `total == 1` to keep P1.3b's prompt byte-identical.)
2. **Higher temperature on follow-up variants:** P1.3a's writer recipe has no temperature override; this stays. The variant prompt + LLM stochasticity provide the diversity. (No code path change for temperature in P1.3c — LLMs already produce divergent outputs across calls.)

Returns `CandidateRecord` with `variant_index = variant_idx`. Errors propagate up through `?` and are caught at the JoinSet collection layer (next section).

### 4.2 Parallel collection

```rust
let mut joinset = tokio::task::JoinSet::new();
let n = cfg.candidates_per_draft;
for i in 0..n {
    let cfg = cfg.clone();           // PipelineConfig: derive(Clone)
    let digest = research_digest.clone();
    let guidelines = voice_guidelines.clone();
    let writer = writer.clone();     // AgentRunner: cheap to clone (Arc<...>)
    let critic = critic.clone();
    let fact = fact.clone();
    joinset.spawn(async move {
        generate_candidate(i, n, &cfg, &digest, &guidelines, &writer, &critic, &fact).await
    });
}

let mut candidates: Vec<CandidateRecord> = Vec::with_capacity(n);
let mut errors: Vec<PipelineError> = Vec::new();
while let Some(res) = joinset.join_next().await {
    match res {
        Ok(Ok(rec))    => candidates.push(rec),
        Ok(Err(e))     => { progress(&format!("candidate failed: {e}")); errors.push(e); }
        Err(joinerr)   => progress(&format!("candidate task panicked: {joinerr}")),
    }
}
candidates.sort_by_key(|c| c.variant_index);   // restore declared order

if candidates.is_empty() {
    return Err(PipelineError::AllCandidatesFailed { errors });
}
```

`PipelineConfig`, `AgentRunner`, and `ProgressCallback` (`Arc<dyn Fn>`) are all cheap to clone — `derive(Clone)` on `PipelineConfig` is a single attribute change.

`PipelineError::AllCandidatesFailed { errors: Vec<PipelineError> }` is a new variant; its display message lists the per-candidate errors (concatenated, capped to first 3).

### 4.3 Cloning `Arc<BoxedProvider>` and `Arc<dyn Fn>` across `tokio::spawn`

Both are `Send + Sync + 'static` (verified — `BoxedProvider` is `Send + Sync`; `Arc<dyn Fn(&str) + Send + Sync>` is the type alias `ProgressCallback`). The `cfg.clone()` is shallow (Arc bump). `&'a Path` fields don't survive `'static` — so the clone has to convert them to `PathBuf` for the spawned task, OR the task takes `&'_` refs by tightly scoping the lifetime. Simpler: `PipelineConfig::clone()` produces a snapshot suitable for spawning (already true if `&Path` fields are owned `PathBuf` in the cloned struct).

Path-handling tweak: `PipelineConfig` keeps `&'a Path` for callers (zero-allocation public API), but internally `run_pipeline` materializes a `(PathBuf, PathBuf)` pair to pass into the spawned tasks. No public-API change.

## 5. Levenshtein dedup + collapse handling

### 5.1 New helpers in `pipeline/dedup.rs`

```rust
/// 0.0..=1.0. 1.0 when strings are identical (character-for-character).
/// Defined as `1 - levenshtein(a, b) / max(a.len(), b.len())` (char counts).
pub(crate) fn levenshtein_ratio(a: &str, b: &str) -> f64;

/// Greedily compute the set of "distinct" indices: walk in declaration
/// order; an index survives if its draft has Levenshtein ratio ≤
/// `threshold` against every already-surviving index. The lower-indexed
/// of any colliding pair wins (variant_index 0 takes precedence over 1).
pub(crate) fn distinct_indices(drafts: &[&str], threshold: f64) -> Vec<usize>;
```

Threshold constant: `LEVENSHTEIN_DUPLICATE_THRESHOLD: f64 = 0.85` per umbrella spec §6.1.

Levenshtein implementation: standard O(m·n) DP via `Vec<Vec<usize>>`. ~30 LOC. Two existing copies live in `heartbit-core/src/agent/mod.rs` and `heartbit-core/src/tool/builtins/read.rs` per CLAUDE.md's "no premature abstraction" guidance — adding a third here keeps that policy consistent (we'd consolidate at four). If a `pub fn levenshtein` from `heartbit-core` becomes available, switch to it; until then, duplicate.

### 5.2 Collapse + retry loop

```rust
fn dedup_and_retry(
    candidates: &mut Vec<CandidateRecord>,
    /* ...generation context for retries... */
) -> Result<(), PipelineError> {
    let drafts: Vec<&str> = candidates.iter().map(|c| c.draft.as_str()).collect();
    let distinct: Vec<usize> = distinct_indices(&drafts, 0.85);
    let collapsed = candidates.len() - distinct.len();

    if collapsed > 0 {
        progress(&format!(
            "candidates collapsed ({collapsed} near-duplicates) — refilling once"
        ));
        // Refill: generate `collapsed` more candidates in parallel,
        // assigning variant_indices that don't conflict with surviving ones.
        // Then re-run distinct_indices on the merged set.
        let next_idx = candidates.iter().map(|c| c.variant_index).max().unwrap_or(0) + 1;
        // ... spawn JoinSet with `collapsed` tasks, variant_idx = next_idx + offset ...
        // ... merge results, re-run distinct_indices once ...
    }

    // Drop duplicates: keep only distinct indices.
    let final_distinct = distinct_indices(
        &candidates.iter().map(|c| c.draft.as_str()).collect::<Vec<_>>(),
        0.85,
    );
    *candidates = final_distinct.into_iter().map(|i| candidates[i].clone()).collect();

    if candidates.len() < target_count {
        progress(&format!(
            "ship-with-fewer: {} of {} distinct candidates after retry",
            candidates.len(), target_count
        ));
    }
    Ok(())
}
```

**Cap = 1 retry pass.** If after the refill we still have collapse, ship whatever distinct count we have (1, 2, or `target_count`). The judge handles N=1 trivially (skip).

### 5.3 Judge skip optimization

```rust
let chosen_index: usize;
let judge_reasoning: String;
if candidates.len() == 1 {
    chosen_index = 0;
    judge_reasoning = "single candidate, no ranking needed".to_string();
} else {
    // Build judge user message with N candidates, parse, validate.
    let verdict = parse_judge_verdict(&judge_out.result, candidates.len())?;
    chosen_index = verdict.chosen_index;
    judge_reasoning = verdict.reasoning;
}
```

This makes `candidates_per_draft: 1` (P1.3b back-compat) skip the judge LLM call entirely. It also handles the dedup-collapsed-to-1 edge case correctly.

## 6. Judge integration

### 6.1 User message format

```
Topic: {topic}

Voice guidelines:
{voice_guidelines}

You have {N} candidate drafts to choose from. Pick the best one.

CANDIDATES

[0]
{candidates[0].draft}

[1]
{candidates[1].draft}

[2]
{candidates[2].draft}

Return your verdict as JSON per the schema. The chosen_index must be in [0, {N-1}].
```

The judge recipe (P1.3a `agents/judge.rs`) already has `response_schema: { chosen_index, reasoning }` — `parse_judge_verdict` deserializes it and validates `chosen_index < N`.

### 6.2 New `parse_judge_verdict`

In `pipeline/verdicts.rs`:

```rust
pub fn parse_judge_verdict(raw: &str, n: usize) -> Result<JudgeVerdict, VerdictParseError> {
    let unfenced = strip_fence(raw.trim());
    let parsed: JudgeRaw = serde_json::from_str(unfenced)
        .map_err(|source| VerdictParseError::Judge { source, raw: raw.to_string() })?;
    if parsed.chosen_index >= n {
        return Err(VerdictParseError::JudgeChoiceOutOfRange {
            chosen_index: parsed.chosen_index,
            n,
            raw: raw.to_string(),
        });
    }
    Ok(JudgeVerdict {
        chosen_index: parsed.chosen_index,
        reasoning: parsed.reasoning,
    })
}

#[derive(Debug, Deserialize)]
struct JudgeRaw {
    chosen_index: usize,
    reasoning: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JudgeVerdict {
    pub chosen_index: usize,
    pub reasoning: String,
}
```

## 7. Image generator integration

### 7.1 When it runs

After judge returns `chosen_index`, run the `image_generator` recipe on `candidates[chosen_index].draft`. The recipe's own system prompt (P1.3a) decides whether an image adds value — it returns the literal string `"no_image"` if not, otherwise it calls the `image_generate` builtin tool and returns its output (URL + alt text).

### 7.2 Output parsing

```rust
fn parse_image_generator_output(raw: &str) -> Option<ImageAttachment>;
```

- Trim, lowercase: if equals `"no_image"`, return `None`.
- Otherwise, the recipe's contract is "Return the image_generate tool's output (URL + alt text) as your final answer." So `raw` is text containing the URL + optional alt text. Parse via:
  - Try to parse as JSON `{"url": "...", "alt_text": "..."}` (preferred output shape).
  - Fall back: heuristic — first URL-shaped substring in the text becomes `url`; everything after the URL becomes `alt_text` (best-effort).
  - On parse failure (no URL found): emit `progress("image_generator output not parseable: {raw[..80]}")` and return `None` (non-blocking).

Image-generator failures are non-blocking by design: the chosen draft ships to stdout regardless. This is appropriate for P1.3c where the image is purely a candidate enhancement; P1.3d may tighten this.

### 7.3 Tool wiring

The `image_generator` recipe needs the `image_generate` tool. In P1.3a `agents/mod.rs::tools_for_persona()`, this tool is already in the persona's tool set. `runner_from_recipe` (P1.3b) passes the persona's tool subset to the builder; the image_generator recipe gets `[image_generate]` as its tool subset.

Update to `run_pipeline`'s `build_runner` calls: image_generator gets `vec![image_generate_tool]` instead of `vec![]`.

## 8. `verdicts.rs` refactor — folded P1.3b code-review follow-up

Replaces the current `Result<_, serde_json::Error>` in `parse_critic_verdict` and `parse_fact_verdict`.

```rust
#[derive(Debug, thiserror::Error)]
pub enum VerdictParseError {
    #[error("critic verdict parse: {source}")]
    Critic {
        #[source]
        source: serde_json::Error,
        raw: String,
    },
    #[error("fact_check verdict parse: {source}")]
    Fact {
        #[source]
        source: serde_json::Error,
        raw: String,
    },
    #[error("judge verdict parse: {source}")]
    Judge {
        #[source]
        source: serde_json::Error,
        raw: String,
    },
    #[error("judge chose index {chosen_index} out of range [0, {n})")]
    JudgeChoiceOutOfRange {
        chosen_index: usize,
        n: usize,
        raw: String,
    },
}

pub fn parse_critic_verdict(raw: &str) -> Result<StyleVerdict, VerdictParseError>;
pub fn parse_fact_verdict(raw: &str) -> Result<FactVerdict, VerdictParseError>;
pub fn parse_judge_verdict(raw: &str, n: usize) -> Result<JudgeVerdict, VerdictParseError>;
```

`PipelineError` updates:

```rust
pub enum PipelineError {
    // ...existing variants...

    // CHANGED: source is now VerdictParseError (was serde_json::Error).
    #[error("style_critic verdict parse: {source}")]
    CriticParseFailed {
        #[source]
        source: VerdictParseError,
    },
    #[error("fact_check verdict parse: {source}")]
    FactCheckParseFailed {
        #[source]
        source: VerdictParseError,
    },

    // NEW:
    #[error("judge verdict parse: {source}")]
    JudgeParseFailed {
        #[source]
        source: VerdictParseError,
    },
    #[error("all {n} candidates failed: {errors:?}")]
    AllCandidatesFailed {
        errors: Vec<PipelineError>,  // boxed indirectly via Vec, no recursive size
        n: usize,
    },
}
```

The `raw` field moves from `PipelineError::CriticParseFailed { source, raw }` (P1.3b) into `VerdictParseError::Critic { source, raw }` — no information loss, and the parse error now carries everything it needs in one place.

Public-API impact: callers of `parse_critic_verdict` / `parse_fact_verdict` from external code (no known consumers today) must adapt to `Result<_, VerdictParseError>` instead of `Result<_, serde_json::Error>`. Internally, `run_pipeline`'s `?` flows still work via `#[from] VerdictParseError` on the wrapping `PipelineError` variants.

## 9. Error handling

`PipelineError` gains 2 new variants:

| Variant | Cause | Display |
|---|---|---|
| `JudgeParseFailed { source: VerdictParseError }` | Judge LLM returned malformed JSON or out-of-range `chosen_index` | `judge verdict parse: {source}` |
| `AllCandidatesFailed { errors, n }` | All N candidate generation tasks failed (per-task errors are collected, capped to first 3 in display) | `all {n} candidates failed: {errors:?}` |

Existing variants (`Builder`, `Agent`, etc.) keep their meanings. Image-generator failures do **not** raise a new variant — they're non-blocking and merely log via `on_progress`.

Validation at the start of `run_pipeline`:

```rust
if !(1..=10).contains(&cfg.candidates_per_draft) {
    return Err(PipelineError::InvalidConfig(format!(
        "candidates_per_draft must be in 1..=10 (got {})",
        cfg.candidates_per_draft,
    )));
}
```

`PipelineError::InvalidConfig(String)` is also new — small, generic, scoped to config-validation failures at pipeline start.

## 10. Testing

| File | Tests | What they cover |
|---|---|---|
| `pipeline/dedup.rs` | 6 unit | `levenshtein_ratio` (identical=1.0, single-char-diff, empty strings, unicode); `distinct_indices` (all distinct, 2 collapsed pairs, 3 identical → 1) |
| `pipeline/verdicts.rs` | +4 unit | `parse_judge_verdict` (happy path, fence stripped, out-of-range index error, malformed JSON); existing 8 tests update to new `VerdictParseError` type |
| `pipeline/mod.rs` (integration) | +5 `#[tokio::test]` | (1) 3-candidate happy path: 3 distinct drafts, judge picks index 1, image returned. (2) candidates_per_draft=1: skips judge, image, dedup. (3) collapse: 2 of 3 candidates near-duplicate (Lev > 0.85), refill succeeds, ships 3 distinct. (4) all candidates fail → `AllCandidatesFailed`. (5) image_generator returns `"no_image"`: `PipelineOutput.image == None`, `final_draft` still ships. |
| `pipeline/mod.rs` (existing) | 5 P1.3b tests | All 5 pass with `candidates_per_draft: 1` (no behavioral change for single-candidate path). |
| `heartbit-cli/src/persona.rs` | 0 new | `Run` arm passes `candidates_per_draft: 3` (default) — no new test needed; the 2 existing dispatch tests don't exercise the pipeline body. |

**MockProvider extensions:** the existing `MockProvider` (P1.3b test helper) handles plain text and `__respond__` ToolUse. P1.3c needs:
- The `judge` recipe has `response_schema` → `__respond__` path (already handled).
- The `image_generator` recipe has no `response_schema` → returns plain text → already handled.
- The `image_generator` calling the `image_generate` tool: the mock receives a tool-using turn and must return either `"no_image"` (plain text) or a synthetic `image_generate` tool result. **Decision:** integration tests stub `image_generator`'s output as plain text (`"no_image"` or a JSON-shaped image result) — the mock doesn't simulate full tool-use round trips. `image_generate` tool results in tests are pre-baked into the canned responses.

Total test delta: ~15 new tests. Workspace count: 3973 → ~3988.

## 11. ADs (architecture decisions)

| AD | Decision | Reason |
|---|---|---|
| AD-1 | Diversity = variant prompt + LLM stochasticity (no exemplar rotation) | Spec §6.1 calls for "different few-shot rotation"; P1.3a doesn't include exemplar selection yet, and P1.3e plans the proper pick-storage + exemplar retrieval. Building it now duplicates P1.3e work. The variant prompt + LLM call independence produce sufficient diversity for the user-visible feature (3 distinct candidates). |
| AD-2 | Bounded retry on collapse (cap = 1 pass) | Strict regen-until-distinct (spec literal) risks unbounded LLM cost on adversarial cases. One retry passes the spirit of the spec ("regenerate the missing slot") while keeping cost predictable. Ship-with-fewer is surfaced via `on_progress`. |
| AD-3 | image_generator runs always; recipe decides "no_image" | Per spec §3 sub-agent table, image_generator is part of the pipeline. Its own prompt has internal "no_image" gating, so always-on doesn't generate unwanted images. Cost is 1 cheap LLM call per pipeline run (low reasoning effort). |
| AD-4 | Parallel candidate generation via `tokio::JoinSet` | Wall-clock matters for the eventual daemon path. JoinSet is a familiar pattern in heartbit-core. Per-task failure handling: drop the failing slot, abort only when all fail. |
| AD-5 | `candidates_per_draft: usize` config knob (default 3, range 1..=10) | Single change makes P1.3b's behavior reachable as a special case (single candidate, judge skipped). Preserves P1.3b's 5 integration tests verbatim modulo new fields. Range cap prevents accidental cost blowups. |
| AD-6 | `image: Option<ImageAttachment>` lives on `PipelineOutput`, not `CandidateRecord` | P1.3c only generates image for the chosen candidate. P1.3d may move to per-candidate if Telegram review wants previews; that's a contained refactor. |
| AD-7 | Levenshtein helper duplicated, not extracted | Per CLAUDE.md "no premature abstraction" — three existing copies in `heartbit-core` is the bar; adding a fourth in `heartbit-ghost::pipeline::dedup` keeps that policy consistent. Extract on the next consumer. |
| AD-8 | Judge skipped when N=1 (post-dedup or by config) | Saves an LLM call. Determinism: `chosen_index = 0`, `judge_reasoning = "single candidate, no ranking needed"`. Same path handles the dedup-collapse-to-1 and the `candidates_per_draft: 1` cases. |
| AD-9 | `VerdictParseError` wraps `serde_json::Error` (P1.3b follow-up) | Stops `serde_json` types from leaking into `heartbit-ghost`'s public API. Adds a `JudgeChoiceOutOfRange` variant (out-of-range judge index = parse error, not a runtime variant). |
| AD-10 | image_generator output parsing: try JSON, fall back to URL-shaped substring extraction | The recipe's prompt says "Return the image_generate tool's output (URL + alt text) as your final answer" — output shape is loose. Best-effort parsing is appropriate; failures are non-blocking (`None` image, draft still ships). |

## 12. Acceptance criteria

P1.3c is done when:

1. `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green.
2. ~15 new tests pass: 6 dedup + 4 verdicts + 5 integration. Existing 5 P1.3b integration tests pass with `candidates_per_draft: 1`.
3. New public surface reachable from `heartbit_ghost::pipeline`: `CandidateRecord`, `ImageAttachment`, `JudgeVerdict`, `VerdictParseError`, `parse_judge_verdict`. The Levenshtein helpers (`levenshtein_ratio`, `distinct_indices`) stay `pub(crate)` per AD-7.
4. `heartbit persona run heartbit-ghost:x --once "<topic>"` produces 3 candidate drafts, judge picks one, image_generator runs (returns `no_image` or attaches an image URL), final chosen draft prints to stdout, full candidate set + ranking metadata on stderr via `on_progress`.
5. Live LLM run against the existing P1.3b corpora (`karpathy` + `bcherny`) produces a coherent ranked output. (User-driven verification, same path as P1.3b's acceptance §5.)
6. `verdicts.rs` parse functions return `Result<_, VerdictParseError>` (no `serde_json::Error` in public signatures).

## 13. Out of scope (deferred)

- Telegram review delivery → P1.3d
- `publisher` recipe usage / actual posting → P1.3d
- Pick storage (which candidate the user picks) → P1.3e
- Exemplar pool / few-shot rotation from corpora → P1.3e (replaces AD-1's variant-prompt-only diversity)
- Autonomy phase logic (Phase 0 calibration, auto-publish gates) → P1.3d / P1.4
- Audit log integration → P1.4
- LLM-based content guardrails (PII, brand safety, harassment) → P1.4
- Trigger specs (cron / mention polling) → P1.4
- X "weighted character" counting in `publish_gate` → P1.4

## 14. Reference

- Umbrella spec: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md` §3 (pipeline), §6 (A/B feedback)
- P1.3a recipes: `crates/heartbit-ghost/src/agents/{judge,image_generator}.rs`
- P1.3b spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3b-pipeline-orchestrator-design.md`
- P1.3b code-review follow-up (verdict-error refactor): inline final-review notes on the merge commit `1f0ea2d`
- Levenshtein implementations: `heartbit-core/src/agent/mod.rs`, `heartbit-core/src/tool/builtins/read.rs`
- `tokio::JoinSet` usage in heartbit-core: `crates/heartbit-core/src/agent/orchestrator.rs` (for reference)
