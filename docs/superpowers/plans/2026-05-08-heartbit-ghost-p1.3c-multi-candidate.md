# heartbit-ghost P1.3c — Multi-candidate generation + judge ranking + image_generator + verdict-error refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `heartbit_ghost::pipeline::run_pipeline` to produce N distinct candidate drafts (default 3), rank them via the existing P1.3a `judge` recipe, optionally attach an image to the chosen draft via the P1.3a `image_generator` recipe, and surface the full candidate set + ranking metadata in `PipelineOutput`. Fold in the P1.3b code-review follow-up by wrapping `serde_json::Error` behind a new `VerdictParseError`.

**Architecture:** Researcher runs once. Per-candidate writer→style_critic (revise loop)→fact_check chains run in parallel via `tokio::JoinSet` (spec AD-4). After collection we run Levenshtein dedup with a single bounded retry pass (AD-2). Judge picks the winner (skipped trivially when N=1, AD-8). image_generator runs on the chosen draft, deciding internally whether to emit `"no_image"` (AD-3). publish_gate guards the chosen draft and we `println!` it to stdout. P1.3b's single-candidate behavior remains accessible via `PipelineConfig.candidates_per_draft: 1`.

**Tech Stack:** Rust 2024, `tokio::task::JoinSet` for parallel candidate generation, `thiserror` for error types, `serde_json` for verdict parsing (now wrapped in `VerdictParseError`), Levenshtein DP for dedup. No new workspace deps.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `crates/heartbit-ghost/src/pipeline/verdicts.rs` | MODIFY | Add `VerdictParseError` enum (Critic / Fact / Judge / JudgeChoiceOutOfRange variants). Convert `parse_critic_verdict` / `parse_fact_verdict` return type to `Result<_, VerdictParseError>`. Add `JudgeVerdict` struct + `parse_judge_verdict(raw, n)`. Existing 8 tests adjust assertions to new error type. 4 new tests for `parse_judge_verdict`. |
| `crates/heartbit-ghost/src/pipeline/dedup.rs` | NEW | `LEVENSHTEIN_DUPLICATE_THRESHOLD: f64 = 0.85` + `pub(crate) fn levenshtein` + `pub(crate) fn levenshtein_ratio` + `pub(crate) fn distinct_indices`. 6 unit tests. |
| `crates/heartbit-ghost/src/pipeline/prompts.rs` | MODIFY | Add `pub(crate) fn build_judge_user_message`. Add `pub(crate) fn build_image_generator_user_message`. Add the variant-aware writer prompt (mutates `build_writer_user_message` signature). |
| `crates/heartbit-ghost/src/pipeline/mod.rs` | MODIFY | Add `pub mod dedup;`. Add types: `CandidateRecord`, `ImageAttachment`, `parse_image_generator_output`. Update `PipelineConfig` (new `candidates_per_draft` + `derive(Clone)`). Update `PipelineOutput` (new `candidates` / `chosen_index` / `judge_reasoning` / `image` fields). Update `PipelineError` (CriticParseFailed / FactCheckParseFailed source-type change; new `JudgeParseFailed` / `AllCandidatesFailed` / `InvalidConfig` variants). Extract revise loop into `generate_candidate`. Replace single-candidate body with parallel JoinSet, dedup, retry-once, judge, image_generator. 5 new integration tests. |
| `crates/heartbit-cli/src/persona.rs` | MODIFY | `Run` arm: pass `candidates_per_draft: 3` (default already; explicit for clarity). Update the post-pipeline summary log to print candidate count + chosen index. |

4 implementation tasks + 1 final acceptance.

---

## Task 1: Foundation — verdicts refactor + dedup module + new types

**Why:** The verdict-error refactor is a breaking change to public signatures, so it must land first; everything downstream uses the new types. Levenshtein helpers are pure functions that Task 3 needs. The new `CandidateRecord` / `ImageAttachment` types let Task 2/3 build on top without churn.

**Files:**
- Modify: `crates/heartbit-ghost/src/pipeline/verdicts.rs`
- Create: `crates/heartbit-ghost/src/pipeline/dedup.rs`
- Modify: `crates/heartbit-ghost/src/pipeline/mod.rs`

- [ ] **Step 1: Replace `crates/heartbit-ghost/src/pipeline/verdicts.rs` body**

The file currently has `parse_critic_verdict` / `parse_fact_verdict` returning `Result<_, serde_json::Error>` and a `strip_fence` helper. Replace the whole file with:

```rust
//! Structured verdict parsing for `style_critic`, `fact_check`, and `judge`.

use serde::Deserialize;
use thiserror::Error;

/// Errors raised by the three verdict parsers. Wraps `serde_json::Error`
/// so it doesn't leak into `heartbit-ghost`'s public API.
#[derive(Debug, Error)]
pub enum VerdictParseError {
    /// `style_critic` returned malformed JSON or an unknown verdict variant.
    #[error("critic verdict parse: {source}")]
    Critic {
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
        /// The raw critic output that failed to parse, kept for diagnostics.
        raw: String,
    },

    /// `fact_check` returned malformed JSON or an unknown verdict variant.
    #[error("fact_check verdict parse: {source}")]
    Fact {
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
        /// The raw fact_check output that failed to parse, kept for diagnostics.
        raw: String,
    },

    /// `judge` returned malformed JSON.
    #[error("judge verdict parse: {source}")]
    Judge {
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
        /// The raw judge output that failed to parse, kept for diagnostics.
        raw: String,
    },

    /// `judge` returned a `chosen_index` outside the valid range `[0, n)`.
    #[error("judge chose index {chosen_index} out of range [0, {n})")]
    JudgeChoiceOutOfRange {
        /// The out-of-range index the judge returned.
        chosen_index: usize,
        /// The number of candidates the judge was given.
        n: usize,
        /// The raw judge output, kept for diagnostics.
        raw: String,
    },
}

/// Critic verdict — three branches that drive the revise loop.
#[derive(Debug, Clone, PartialEq)]
pub enum StyleVerdict {
    /// Draft is acceptable; ship it.
    Pass {
        /// 0.0..=1.0 voice match score.
        score: f64,
    },
    /// Draft is recoverable; loop back to the writer with this reason.
    Revise {
        /// Short feedback string fed into the writer's next user message.
        reason: String,
        /// 0.0..=1.0 voice match score.
        score: f64,
    },
    /// Draft is fundamentally off; abort the pipeline.
    Reject {
        /// Short reason explaining the rejection.
        reason: String,
        /// 0.0..=1.0 voice match score.
        score: f64,
    },
}

impl StyleVerdict {
    /// Returns the 0.0..=1.0 voice match score for any verdict variant.
    pub fn score(&self) -> f64 {
        match self {
            StyleVerdict::Pass { score }
            | StyleVerdict::Revise { score, .. }
            | StyleVerdict::Reject { score, .. } => *score,
        }
    }
}

/// Fact-check verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum FactVerdict {
    /// Every factual claim is supported by the research digest.
    Verified,
    /// At least one claim is contradicted by or absent from the digest.
    Unverifiable {
        /// Short reason naming the offending claim.
        reason: String,
    },
}

/// Judge verdict — picks one of N candidate drafts.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeVerdict {
    /// Index into the input candidate slice, validated `0..n` by
    /// [`parse_judge_verdict`].
    pub chosen_index: usize,
    /// Short reasoning string from the judge.
    pub reasoning: String,
}

#[derive(Debug, Deserialize)]
struct CriticRaw {
    verdict: String,
    #[serde(default)]
    reason: Option<String>,
    style_match_score: f64,
}

#[derive(Debug, Deserialize)]
struct FactRaw {
    verdict: String,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JudgeRaw {
    chosen_index: usize,
    reasoning: String,
}

/// Parse the critic's raw output as JSON.
pub fn parse_critic_verdict(raw: &str) -> Result<StyleVerdict, VerdictParseError> {
    let unfenced = strip_fence(raw.trim());
    let parsed: CriticRaw = serde_json::from_str(unfenced).map_err(|source| {
        VerdictParseError::Critic {
            source,
            raw: raw.to_string(),
        }
    })?;
    let verdict = match parsed.verdict.as_str() {
        "pass" => StyleVerdict::Pass {
            score: parsed.style_match_score,
        },
        "revise" => StyleVerdict::Revise {
            reason: parsed.reason.unwrap_or_else(|| "unspecified".to_string()),
            score: parsed.style_match_score,
        },
        "reject" => StyleVerdict::Reject {
            reason: parsed.reason.unwrap_or_else(|| "unspecified".to_string()),
            score: parsed.style_match_score,
        },
        other => {
            let source = serde::de::Error::unknown_variant(other, &["pass", "revise", "reject"]);
            return Err(VerdictParseError::Critic {
                source,
                raw: raw.to_string(),
            });
        }
    };
    Ok(verdict)
}

/// Parse the fact_check raw output as JSON.
pub fn parse_fact_verdict(raw: &str) -> Result<FactVerdict, VerdictParseError> {
    let unfenced = strip_fence(raw.trim());
    let parsed: FactRaw = serde_json::from_str(unfenced).map_err(|source| {
        VerdictParseError::Fact {
            source,
            raw: raw.to_string(),
        }
    })?;
    let verdict = match parsed.verdict.as_str() {
        "verified" => FactVerdict::Verified,
        "unverifiable" => FactVerdict::Unverifiable {
            reason: parsed.reason.unwrap_or_else(|| "unspecified".to_string()),
        },
        other => {
            let source = serde::de::Error::unknown_variant(other, &["verified", "unverifiable"]);
            return Err(VerdictParseError::Fact {
                source,
                raw: raw.to_string(),
            });
        }
    };
    Ok(verdict)
}

/// Parse the judge's raw output as JSON. Validates `chosen_index` against
/// `n` (the number of candidates the judge was given).
pub fn parse_judge_verdict(raw: &str, n: usize) -> Result<JudgeVerdict, VerdictParseError> {
    let unfenced = strip_fence(raw.trim());
    let parsed: JudgeRaw = serde_json::from_str(unfenced).map_err(|source| {
        VerdictParseError::Judge {
            source,
            raw: raw.to_string(),
        }
    })?;
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

/// Strip a single ```json … ``` (or ``` … ```) fence pair if present.
fn strip_fence(s: &str) -> &str {
    let body = s
        .strip_prefix("```json\n")
        .or_else(|| s.strip_prefix("```json"))
        .or_else(|| s.strip_prefix("```\n"))
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    body.strip_suffix("```")
        .map(str::trim)
        .unwrap_or(body)
        .trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- StyleVerdict (existing tests, updated for VerdictParseError) ----

    #[test]
    fn parse_critic_pass() {
        let raw = r#"{"verdict": "pass", "style_match_score": 0.92}"#;
        let v = parse_critic_verdict(raw).unwrap();
        assert_eq!(v, StyleVerdict::Pass { score: 0.92 });
    }

    #[test]
    fn parse_critic_revise_with_reason() {
        let raw = r#"{"verdict": "revise", "reason": "uses em-dashes", "style_match_score": 0.65}"#;
        let v = parse_critic_verdict(raw).unwrap();
        assert_eq!(
            v,
            StyleVerdict::Revise {
                reason: "uses em-dashes".to_string(),
                score: 0.65,
            }
        );
    }

    #[test]
    fn parse_critic_reject_with_reason() {
        let raw = r#"{"verdict": "reject", "reason": "off-topic", "style_match_score": 0.2}"#;
        let v = parse_critic_verdict(raw).unwrap();
        assert_eq!(
            v,
            StyleVerdict::Reject {
                reason: "off-topic".to_string(),
                score: 0.2,
            }
        );
    }

    #[test]
    fn parse_critic_strips_markdown_fence() {
        let raw = "```json\n{\"verdict\": \"pass\", \"style_match_score\": 0.9}\n```";
        let v = parse_critic_verdict(raw).unwrap();
        assert_eq!(v, StyleVerdict::Pass { score: 0.9 });
    }

    #[test]
    fn parse_critic_malformed_returns_critic_variant() {
        let raw = "definitely not json";
        let err = parse_critic_verdict(raw).unwrap_err();
        match err {
            VerdictParseError::Critic { raw: r, .. } => assert_eq!(r, "definitely not json"),
            other => panic!("expected Critic variant, got: {other:?}"),
        }
    }

    // ---- FactVerdict (existing tests, updated for VerdictParseError) ----

    #[test]
    fn parse_fact_verified() {
        let raw = r#"{"verdict": "verified"}"#;
        let v = parse_fact_verdict(raw).unwrap();
        assert_eq!(v, FactVerdict::Verified);
    }

    #[test]
    fn parse_fact_unverifiable_with_reason() {
        let raw = r#"{"verdict": "unverifiable", "reason": "no source for the 47% figure"}"#;
        let v = parse_fact_verdict(raw).unwrap();
        assert_eq!(
            v,
            FactVerdict::Unverifiable {
                reason: "no source for the 47% figure".to_string()
            }
        );
    }

    #[test]
    fn parse_fact_unknown_verdict_returns_fact_variant() {
        let raw = r#"{"verdict": "maybe"}"#;
        let err = parse_fact_verdict(raw).unwrap_err();
        match err {
            VerdictParseError::Fact { raw: r, .. } => assert_eq!(r, r#"{"verdict": "maybe"}"#),
            other => panic!("expected Fact variant, got: {other:?}"),
        }
    }

    // ---- JudgeVerdict (NEW) ----

    #[test]
    fn parse_judge_happy_path() {
        let raw = r#"{"chosen_index": 1, "reasoning": "candidate 1 has more specific examples"}"#;
        let v = parse_judge_verdict(raw, 3).unwrap();
        assert_eq!(
            v,
            JudgeVerdict {
                chosen_index: 1,
                reasoning: "candidate 1 has more specific examples".to_string(),
            }
        );
    }

    #[test]
    fn parse_judge_strips_markdown_fence() {
        let raw = "```json\n{\"chosen_index\": 0, \"reasoning\": \"first one\"}\n```";
        let v = parse_judge_verdict(raw, 2).unwrap();
        assert_eq!(v.chosen_index, 0);
    }

    #[test]
    fn parse_judge_out_of_range_returns_specific_variant() {
        let raw = r#"{"chosen_index": 5, "reasoning": "anything"}"#;
        let err = parse_judge_verdict(raw, 3).unwrap_err();
        match err {
            VerdictParseError::JudgeChoiceOutOfRange { chosen_index, n, .. } => {
                assert_eq!(chosen_index, 5);
                assert_eq!(n, 3);
            }
            other => panic!("expected JudgeChoiceOutOfRange, got: {other:?}"),
        }
    }

    #[test]
    fn parse_judge_malformed_returns_judge_variant() {
        let raw = "not json";
        let err = parse_judge_verdict(raw, 3).unwrap_err();
        match err {
            VerdictParseError::Judge { raw: r, .. } => assert_eq!(r, "not json"),
            other => panic!("expected Judge variant, got: {other:?}"),
        }
    }
}
```

- [ ] **Step 2: Run verdicts tests, expect all 12 to pass**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-ghost --lib pipeline::verdicts 2>&1 | tail -8
```

Expected: `12 passed` (5 critic + 3 fact + 4 judge).

The pipeline module's call sites are now broken because `parse_*_verdict` return type changed. We fix that in Step 4.

- [ ] **Step 3: Create `crates/heartbit-ghost/src/pipeline/dedup.rs`**

```rust
//! Levenshtein-based candidate dedup. Pure functions; no I/O.

/// Drafts with Levenshtein ratio above this threshold are considered
/// near-duplicates and one of the pair is dropped per umbrella spec §6.1.
pub(crate) const LEVENSHTEIN_DUPLICATE_THRESHOLD: f64 = 0.85;

/// Levenshtein distance via standard O(m·n) DP. Distance is in characters
/// (not bytes), so unicode multi-byte sequences count as one each.
pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let m = av.len();
    let n = bv.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if av[i - 1] == bv[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Levenshtein ratio in [0.0, 1.0]. 1.0 = identical; 0.0 = completely different.
/// Defined as `1.0 - distance / max(len_a, len_b)`. Empty-vs-empty = 1.0.
pub(crate) fn levenshtein_ratio(a: &str, b: &str) -> f64 {
    let len_a = a.chars().count();
    let len_b = b.chars().count();
    let max_len = len_a.max(len_b);
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

/// Greedy distinct-set computation. Walks `drafts` in declaration order;
/// each index survives if its Levenshtein ratio is `<= threshold` against
/// every already-surviving index. The lower-indexed of any colliding pair
/// wins (variant 0 takes precedence over variant 1).
pub(crate) fn distinct_indices(drafts: &[&str], threshold: f64) -> Vec<usize> {
    let mut survivors: Vec<usize> = Vec::with_capacity(drafts.len());
    for (i, draft) in drafts.iter().enumerate() {
        let collides = survivors
            .iter()
            .any(|&j| levenshtein_ratio(draft, drafts[j]) > threshold);
        if !collides {
            survivors.push(i);
        }
    }
    survivors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_identical_strings_zero_distance() {
        assert_eq!(levenshtein("hello", "hello"), 0);
        assert_eq!(levenshtein_ratio("hello", "hello"), 1.0);
    }

    #[test]
    fn levenshtein_single_char_diff_one_distance() {
        assert_eq!(levenshtein("hello", "hallo"), 1);
        assert!((levenshtein_ratio("hello", "hallo") - 0.8).abs() < 1e-9);
    }

    #[test]
    fn levenshtein_empty_strings_ratio_is_one() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein_ratio("", ""), 1.0);
    }

    #[test]
    fn levenshtein_handles_unicode_as_chars_not_bytes() {
        // "é" is 2 bytes UTF-8 but 1 char.
        assert_eq!(levenshtein("café", "cafe"), 1);
    }

    #[test]
    fn distinct_indices_all_distinct_keeps_all() {
        let drafts = vec!["alpha is one", "beta is two", "gamma is three"];
        let out = distinct_indices(&drafts, 0.85);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn distinct_indices_two_near_duplicates_keeps_lower_index() {
        // 1 and 2 are identical (ratio = 1.0 > 0.85). 0 is distinct.
        let drafts = vec!["the first draft is long", "alpha", "alpha"];
        let out = distinct_indices(&drafts, 0.85);
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn distinct_indices_three_identical_collapse_to_one() {
        let drafts = vec!["same", "same", "same"];
        let out = distinct_indices(&drafts, 0.85);
        assert_eq!(out, vec![0]);
    }
}
```

7 tests instead of 6 — extra one (`levenshtein_handles_unicode_as_chars_not_bytes`) is cheap and proves the char-not-byte semantics that the umbrella spec implicitly requires.

- [ ] **Step 4: Update `crates/heartbit-ghost/src/pipeline/mod.rs` — types + error variants + call site fixups**

This is the big mechanical change. Open `crates/heartbit-ghost/src/pipeline/mod.rs` and apply these edits:

**4a)** Add `pub mod dedup;` to the module declarations near the top of the file (after `pub mod prompts;`):

```rust
pub mod dedup;
pub mod prompts;
pub mod publish_gate;
pub mod style_render;
pub mod verdicts;
```

**4b)** Re-exports — add the new types after the existing `pub use` lines:

```rust
pub use publish_gate::{PublishGateError, check_publish_gate};
pub use style_render::render_style_profile_as_english;
pub use verdicts::{
    FactVerdict, JudgeVerdict, StyleVerdict, VerdictParseError, parse_critic_verdict,
    parse_fact_verdict, parse_judge_verdict,
};
```

**4c)** Add the new `CandidateRecord` and `ImageAttachment` structs immediately after `pub use verdicts::...`:

```rust
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
```

**4d)** Update `PipelineConfig` to add `candidates_per_draft` + `derive(Clone)`:

```rust
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
}

impl<'a> Default for PipelineConfig<'a> {
    /// Note: required fields (persona_name, topic, provider, corpora_root,
    /// profiles_root) have no sensible defaults — this `Default` is only
    /// intended for tests that override every field. Callers should
    /// construct `PipelineConfig` explicitly.
    fn default() -> Self {
        // Default never makes sense for borrowed paths; this exists so
        // tests can derive partial defaults via struct-update syntax. We
        // rely on the borrow checker to force callers to override paths.
        unimplemented!("PipelineConfig has no sensible default; construct explicitly")
    }
}
```

Actually scratch the `Default` impl — `PipelineConfig` has no sensible default because of borrowed `&'a Path` fields. Test code constructs it explicitly. **Skip the `Default` impl.** Just `#[derive(Clone)]`.

So the final `PipelineConfig` is just:

```rust
/// Configuration for one pipeline run.
#[derive(Clone)]
pub struct PipelineConfig<'a> {
    pub persona_name: &'a str,
    pub topic: &'a str,
    pub provider: Arc<BoxedProvider>,
    pub corpora_root: &'a Path,
    pub profiles_root: &'a Path,
    pub on_progress: Option<ProgressCallback>,
    /// Number of distinct candidate drafts to generate. Default: 3.
    /// Validated `1..=10` at the start of `run_pipeline`. Set to 1 to
    /// recover the P1.3b single-candidate behavior (judge skipped).
    pub candidates_per_draft: usize,
}
```

**4e)** Update `PipelineOutput` to add the multi-candidate fields:

```rust
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
    /// Judge's reasoning string. Empty when only one candidate was ranked
    /// (judge skipped — see AD-8).
    pub judge_reasoning: String,
    /// Image attached to the chosen draft, if `image_generator` decided
    /// to generate one. `None` when the recipe returned `"no_image"` or
    /// when the call failed (failures are non-blocking).
    pub image: Option<ImageAttachment>,
}
```

**4f)** Update `PipelineError` — change source type on `CriticParseFailed` / `FactCheckParseFailed`, add `JudgeParseFailed`, `AllCandidatesFailed`, `InvalidConfig`. Replace the existing variants entirely with this block:

```rust
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

    /// All N candidate generation tasks failed.
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
```

**4g)** Update `run_pipeline`'s call sites to use the new error types. The current calls look like:

```rust
let verdict = parse_critic_verdict(&critic_out.result).map_err(|e| {
    PipelineError::CriticParseFailed { source: e, raw: critic_out.result.clone() }
})?;
```

Change to:

```rust
let verdict = parse_critic_verdict(&critic_out.result)
    .map_err(|source| PipelineError::CriticParseFailed { source })?;
```

(Drop the `raw` argument — it now lives inside `VerdictParseError::Critic.raw`.)

Same change for `parse_fact_verdict` → `PipelineError::FactCheckParseFailed { source }`. The `raw` field on these `PipelineError` variants no longer exists.

Search-and-replace in `run_pipeline` body:

| Old | New |
|---|---|
| `PipelineError::CriticParseFailed { source: e, raw: critic_out.result.clone() }` | `PipelineError::CriticParseFailed { source: e }` |
| `PipelineError::FactCheckParseFailed { source: e, raw: fact_out.result.clone() }` | `PipelineError::FactCheckParseFailed { source: e }` |

(Two call sites — one for critic, one for fact_check.)

**4h)** Update the existing 3 `PipelineError` display tests in the same file's `#[cfg(test)] mod tests` block. The tests previously asserted on `CriticParseFailed { source, raw }` shape; now they don't apply (we have new tests on `parse_*_verdict` itself). The `MaxRevisionsExceeded`, `NoProfileSnapshot`, and `Rejected` display tests don't change. Verify visually that no test references `CriticParseFailed.raw` or `FactCheckParseFailed.raw` — the search-and-replace covered the production code, not the tests.

If a test does reference `.raw` on those variants, delete that assertion (the field is gone). If a test asserts on `CriticParseFailed` display, it should still work because the message format `"style_critic verdict parse: {source}"` is unchanged.

- [ ] **Step 5: Run pipeline tests; verify nothing broke**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-ghost --lib pipeline 2>&1 | tail -10
```

Expected: `33 passed` — `27 (P1.3b)` + `7 (dedup)` − `0` (existing tests survive) + `1 (extra unicode test in dedup)` = `34`. Actually let me recount:

| File | P1.3b count | P1.3c delta |
|---|---|---|
| style_render | 4 | 0 |
| verdicts | 8 | +4 (judge) |
| publish_gate | 6 | 0 |
| dedup | 0 | +7 (new file, including unicode) |
| mod (error display + integration) | 9 (3 + 1 reasoning + 5 integration) | 0 (Task 1 doesn't add) |
| **total** | **27** | **+11** = **38** |

Expected: `38 passed`.

If a verdicts assertion fails because the test references `serde_json::Error` directly, fix it inline (`match err { VerdictParseError::Critic { .. } => ... }` instead of `assert!(format!("{err}").contains("expected"))`).

- [ ] **Step 6: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
cargo test -p heartbit-ghost --lib pipeline 2>&1 | tail -3
```

All three clean.

- [ ] **Step 7: Commit**

```bash
cd /home/pleclech/projects/heartbit
git add crates/heartbit-ghost/src/pipeline/verdicts.rs \
        crates/heartbit-ghost/src/pipeline/dedup.rs \
        crates/heartbit-ghost/src/pipeline/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): pipeline — verdicts refactor + dedup module + new types (P1.3c)

Foundation layer for P1.3c. Three pure modules / additions, no
orchestration changes yet:

- verdicts.rs: introduce `VerdictParseError` enum (Critic / Fact /
  Judge / JudgeChoiceOutOfRange variants) wrapping `serde_json::Error`
  so it stops leaking into heartbit-ghost's public API. This is the
  P1.3b code-review follow-up; spec §8 details. Add `JudgeVerdict`
  struct + `parse_judge_verdict(raw, n)` validating chosen_index.
  4 new tests (happy / fence-stripped / out-of-range / malformed).

- dedup.rs (NEW): standard O(m·n) Levenshtein DP +
  `levenshtein_ratio` (chars not bytes, [0.0, 1.0]) +
  `distinct_indices` (greedy collision-free walk, declaration-order
  tiebreak). `LEVENSHTEIN_DUPLICATE_THRESHOLD = 0.85` per umbrella
  spec §6.1. 7 tests (identical / single-char-diff / empty / unicode /
  all-distinct / two-collide / three-identical).

- mod.rs: new types `CandidateRecord`, `ImageAttachment`. Update
  `PipelineConfig` (add `candidates_per_draft`, `#[derive(Clone)]`).
  Update `PipelineOutput` (add `candidates`, `chosen_index`,
  `judge_reasoning`, `image`). Update `PipelineError` source type on
  CriticParseFailed / FactCheckParseFailed; add `JudgeParseFailed`,
  `AllCandidatesFailed`, `InvalidConfig` variants. Mechanical fix-up
  to existing `run_pipeline` call sites (drop the now-redundant `raw`
  field — it lives in `VerdictParseError` now).

11 net new tests: 4 verdicts + 7 dedup. No orchestration changes;
all 5 P1.3b integration tests still pass.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3c-multi-candidate-design.md §3, §5.1, §6.2, §8

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `generate_candidate` extraction + `candidates_per_draft` validation + variant-aware writer prompt

**Why:** Pull the writer→critic→fact_check revise loop body out of `run_pipeline` into a standalone `async fn generate_candidate(...)`. This is the unit Task 3 will spawn N times in parallel. Variant-aware writer prompt (per AD-1: variant prompt + LLM stochasticity = diversity) lands now since it's part of the generate_candidate signature. `candidates_per_draft` validation lands at the top of `run_pipeline`.

**Files:**
- Modify: `crates/heartbit-ghost/src/pipeline/prompts.rs`
- Modify: `crates/heartbit-ghost/src/pipeline/mod.rs`

- [ ] **Step 1: Update `crates/heartbit-ghost/src/pipeline/prompts.rs`**

Replace `build_writer_user_message` with a variant-aware version. The current signature is:

```rust
pub(crate) fn build_writer_user_message(
    topic: &str,
    research_digest: &str,
    voice_guidelines: &str,
    prev_revision: Option<&(String, String)>,
) -> String;
```

Update to:

```rust
/// Construct the writer's user message.
///
/// On the first iteration (no `prev_revision`), only includes topic +
/// research digest + voice guidelines. On revision, also includes the
/// previous draft and the critic's feedback.
///
/// When `total_variants > 1`, appends a "you are generating variant X
/// of N" line to encourage diversity across parallel candidate slots.
pub(crate) fn build_writer_user_message(
    topic: &str,
    research_digest: &str,
    voice_guidelines: &str,
    prev_revision: Option<&(String, String)>,
    variant_index: usize,
    total_variants: usize,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("Topic: {topic}\n\n"));
    out.push_str("Research digest:\n");
    out.push_str(research_digest);
    out.push_str("\n\n");
    out.push_str(voice_guidelines);
    out.push('\n');

    if let Some((prev_draft, critic_reason)) = prev_revision {
        out.push_str("\nPREVIOUS DRAFT:\n");
        out.push_str(prev_draft);
        out.push_str("\n\nSTYLE CRITIC FEEDBACK:\n");
        out.push_str(critic_reason);
        out.push_str(
            "\n\nPlease produce a revised draft addressing the feedback. \
             Output the post text only.\n",
        );
    } else {
        out.push_str("\nProduce one draft. Output the post text only.\n");
    }

    if total_variants > 1 {
        out.push_str(&format!(
            "\nYou are generating variant {} of {}. Pursue a distinct angle \
             from the other variants — emphasize different aspects, examples, \
             or framing.\n",
            variant_index + 1,
            total_variants,
        ));
    }

    out
}
```

The two new parameters (`variant_index`, `total_variants`) at the END of the signature so the existing single-candidate call site only needs to add `0, 1` (or use the variant-aware default).

- [ ] **Step 2: Update the single existing call site of `build_writer_user_message` in `mod.rs`**

In the `run_pipeline` body (P1.3b's revise loop), change:

```rust
let writer_msg = prompts::build_writer_user_message(
    cfg.topic,
    &research_digest,
    &voice_guidelines,
    prev_revision.as_ref(),
);
```

to:

```rust
let writer_msg = prompts::build_writer_user_message(
    cfg.topic,
    &research_digest,
    &voice_guidelines,
    prev_revision.as_ref(),
    0,  // variant_index — single candidate path
    1,  // total_variants — single candidate path (the new line is omitted)
);
```

(Two call sites total in the current revise loop — one initial draft, one revision. Both get `0, 1`. Wait, actually it's the SAME call inside a loop, just one site to update.)

- [ ] **Step 3: Add `candidates_per_draft` validation at the top of `run_pipeline`**

After loading the snapshot but before building the agents, add:

```rust
/// Validation at the start of run_pipeline.
if !(1..=10).contains(&cfg.candidates_per_draft) {
    return Err(PipelineError::InvalidConfig(format!(
        "candidates_per_draft must be in 1..=10 (got {})",
        cfg.candidates_per_draft,
    )));
}
```

Place after the snapshot load (so a missing snapshot reports `NoProfileSnapshot` first, not `InvalidConfig`).

- [ ] **Step 4: Update test fixtures + integration tests to set `candidates_per_draft: 1`**

Find all `PipelineConfig { ... }` literal constructions in the test mod. Each needs `candidates_per_draft: 1` added. Currently 5 integration tests construct `PipelineConfig`. Add the field to each:

```rust
let cfg = PipelineConfig {
    persona_name: "x",
    topic: "AI capabilities",
    provider,
    corpora_root: &corpora,
    profiles_root: &profiles_root,
    on_progress: None,
    candidates_per_draft: 1,  // <-- NEW
};
```

Also update the `runner_from_recipe_maps_reasoning_effort_string_to_enum` test if it constructs `PipelineConfig` (it doesn't — it only uses `MockProvider::arc(vec![])` directly, so no change needed).

- [ ] **Step 5: Extract the revise loop body into `generate_candidate`**

Add this function to `mod.rs`, placed right above `run_pipeline` (and below `runner_from_recipe`):

```rust
/// Run one writer→style_critic (revise loop, max 3 iters)→fact_check
/// chain, producing one [`CandidateRecord`]. Each call is independent;
/// `run_pipeline` spawns N of these in parallel via `tokio::JoinSet`.
async fn generate_candidate(
    variant_idx: usize,
    total_variants: usize,
    topic: &str,
    research_digest: &str,
    voice_guidelines: &str,
    writer: &AgentRunner<BoxedProvider>,
    critic: &AgentRunner<BoxedProvider>,
    fact: &AgentRunner<BoxedProvider>,
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
        );
        let writer_out = writer.execute(&writer_msg).await.map_err(|e| {
            PipelineError::Agent {
                stage: format!("writer (variant {variant_idx}, iter {iter})"),
                source: e,
            }
        })?;
        let draft = writer_out.result.clone();
        total_usage += writer_out.tokens_used;

        let critic_msg = prompts::build_critic_user_message(&draft, voice_guidelines);
        let critic_out = critic.execute(&critic_msg).await.map_err(|e| {
            PipelineError::Agent {
                stage: format!("style_critic (variant {variant_idx}, iter {iter})"),
                source: e,
            }
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
    let fact_out = fact.execute(&fact_msg).await.map_err(|e| {
        PipelineError::Agent {
            stage: format!("fact_check (variant {variant_idx})"),
            source: e,
        }
    })?;
    total_usage += fact_out.tokens_used;
    let fact_check_verdict = parse_fact_verdict(&fact_out.result)
        .map_err(|source| PipelineError::FactCheckParseFailed { source })?;

    // Note: we drop `total_usage` here — usage accumulation will move to
    // run_pipeline in Task 3 (per-candidate usage gets summed at the
    // collection layer). For Task 2's intermediate single-candidate path,
    // the existing run_pipeline body adds its own usage tracking.

    Ok(CandidateRecord {
        variant_index: variant_idx,
        draft,
        style_match_score,
        revise_iterations,
        fact_check_verdict,
    })
}
```

Note: `generate_candidate` records its token usage onto the returned `CandidateRecord.usage` field (already declared in Task 1 Step 4c). `run_pipeline` then sums per-candidate usage at the JoinSet collection layer in Task 3. The function's last expression returns:

```rust
    Ok(CandidateRecord {
        variant_index: variant_idx,
        draft,
        style_match_score,
        revise_iterations,
        fact_check_verdict,
        usage: total_usage,
    })
```

- [ ] **Step 6: Update `run_pipeline` body to call `generate_candidate`**

Replace the existing revise-loop section + fact-check section (everything from "Revise loop" through "fact_verdict = parse_fact_verdict(...)?;") with a single call:

```rust
// Generate one candidate (variant 0 of 1 — single-candidate compatibility).
progress("Generating candidate...");
let candidate = generate_candidate(
    0,
    cfg.candidates_per_draft,  // = 1 in this Task 2 intermediate state when caller sets it
    cfg.topic,
    &research_digest,
    &voice_guidelines,
    &writer,
    &critic,
    &fact,
)
.await?;

total_usage += candidate.usage;

let final_draft = candidate.draft.clone();
let style_match_score = candidate.style_match_score;
let revise_iterations = candidate.revise_iterations;
let fact_check_verdict = candidate.fact_check_verdict.clone();

if let FactVerdict::Unverifiable { ref reason } = fact_check_verdict {
    progress(&format!("fact_check unverifiable: {reason}"));
}
```

The existing `progress("Researching topic...")`, `progress("Running publish_gate...")`, etc. calls stay where they are. Only the writer→critic→fact section gets replaced.

- [ ] **Step 7: Update the `PipelineOutput` construction in `run_pipeline`**

The current `Ok(PipelineOutput { ... })` needs the new `candidates`, `chosen_index`, `judge_reasoning`, `image` fields:

```rust
Ok(PipelineOutput {
    final_draft,
    research_digest,
    style_match_score,
    revise_iterations,
    fact_check_verdict,
    usage_summary: total_usage,
    candidates: vec![candidate],
    chosen_index: 0,
    judge_reasoning: String::new(),
    image: None,
})
```

- [ ] **Step 8: Run pipeline tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-ghost --lib pipeline 2>&1 | tail -10
```

Expected: `38 passed` (same as Task 1, no behavioral change). The 5 integration tests now exercise the `generate_candidate` extraction; the 3 PipelineError display tests still pass.

If any integration test fails, the most likely cause is the new `usage` field on `CandidateRecord` (test fixtures may need updating). Add `usage: TokenUsage::default()` to any `CandidateRecord { ... }` literal constructed in tests.

- [ ] **Step 9: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 10: Commit**

```bash
git add crates/heartbit-ghost/src/pipeline/prompts.rs \
        crates/heartbit-ghost/src/pipeline/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): pipeline — extract generate_candidate + variant prompt + config validation (P1.3c)

Pull P1.3b's writer→critic (revise loop)→fact_check body out of
`run_pipeline` into a standalone `async fn generate_candidate`. This is
the unit Task 3 will spawn N times in parallel via tokio::JoinSet.

- prompts.rs: `build_writer_user_message` gains `variant_index` +
  `total_variants` parameters. When `total_variants > 1`, appends a
  "variant X of N — pursue a distinct angle" line. P1.3a writer
  prompt is unchanged when total_variants == 1.

- mod.rs: new `generate_candidate` private fn returning
  `CandidateRecord` (with new `usage: TokenUsage` field). `run_pipeline`
  body shrinks: replaces ~80 LOC of revise-loop + fact-check with one
  call to generate_candidate. Validates `candidates_per_draft` in
  `1..=10` at the start of run_pipeline. Output construction populates
  the new multi-candidate fields trivially (vec of 1, chosen_index 0).

No behavior change yet — all 5 P1.3b integration tests pass with
candidates_per_draft: 1. 38 pipeline tests still passing.

Task 3 will replace the single generate_candidate call with parallel
JoinSet machinery + dedup + judge + image_generator.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3c-multi-candidate-design.md §4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Parallel candidate generation + dedup-with-retry + judge + image_generator + integration tests

**Why:** The core orchestration change. Wrap `generate_candidate` in `tokio::JoinSet` for parallel execution; collect results into a `Vec<CandidateRecord>`; dedup via Levenshtein with one bounded retry pass; call the judge (skip if N=1); call the image_generator on the chosen draft; return everything in `PipelineOutput`. 5 new integration tests validate the multi-candidate flow.

**Files:**
- Modify: `crates/heartbit-ghost/src/pipeline/prompts.rs` (add 2 builders)
- Modify: `crates/heartbit-ghost/src/pipeline/mod.rs` (parallel orchestration + image_generator parser + 5 new integration tests)

- [ ] **Step 1: Add judge + image_generator user-message builders to `prompts.rs`**

Append to `prompts.rs`:

```rust
/// Construct the judge's user message. Numbered candidate list with
/// voice guidelines and topic context.
pub(crate) fn build_judge_user_message(
    topic: &str,
    voice_guidelines: &str,
    candidates: &[crate::pipeline::CandidateRecord],
) -> String {
    let mut msg = format!("Topic: {topic}\n\n");
    msg.push_str(voice_guidelines);
    msg.push_str("\n\n");
    msg.push_str(&format!(
        "You have {} candidate drafts to choose from. Pick the best one.\n\n",
        candidates.len(),
    ));
    msg.push_str("CANDIDATES\n\n");
    for (i, c) in candidates.iter().enumerate() {
        msg.push_str(&format!("[{i}]\n{}\n\n", c.draft));
    }
    msg.push_str(&format!(
        "Return your verdict as JSON per the schema. The chosen_index must be in [0, {}].\n",
        candidates.len() - 1,
    ));
    msg
}

/// Construct the image_generator's user message.
pub(crate) fn build_image_generator_user_message(
    chosen_draft: &str,
    voice_guidelines: &str,
) -> String {
    format!(
        "Approved draft:\n{chosen_draft}\n\n{voice_guidelines}\n\n\
         Decide whether to attach an image. If no, output the literal \
         string \"no_image\". If yes, call image_generate with a concise \
         visual prompt and return its output.\n"
    )
}
```

- [ ] **Step 2: Add `parse_image_generator_output` to `mod.rs`**

Add right after the `runner_from_recipe` function:

```rust
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
        let alt_text = value.get("alt_text").and_then(|v| v.as_str()).map(String::from);
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
    let end = rest
        .find(|c: char| c.is_whitespace())
        .unwrap_or(rest.len());
    let url = rest[..end].trim_end_matches(|c: char| c == '.' || c == ',' || c == ';').to_string();
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
```

- [ ] **Step 3: Add a private `dedup_candidates` helper to `mod.rs`**

Right after `parse_image_generator_output`:

```rust
/// Drop near-duplicate candidates per `LEVENSHTEIN_DUPLICATE_THRESHOLD`.
/// Lower variant_index wins on collision (declaration-order tiebreak).
fn dedup_candidates(candidates: Vec<CandidateRecord>) -> Vec<CandidateRecord> {
    if candidates.len() <= 1 {
        return candidates;
    }
    let drafts: Vec<&str> = candidates.iter().map(|c| c.draft.as_str()).collect();
    let kept = dedup::distinct_indices(&drafts, dedup::LEVENSHTEIN_DUPLICATE_THRESHOLD);
    kept.into_iter().map(|i| candidates[i].clone()).collect()
}
```

- [ ] **Step 4: Replace `run_pipeline` body with parallel orchestration**

The existing single-candidate body (post-Task 2) calls `generate_candidate` once. Replace that section + everything below it (judge / image / output construction) with the full multi-candidate orchestration.

**Find** (in `run_pipeline`, after the snapshot load + agent construction):

```rust
// Generate one candidate (variant 0 of 1 — single-candidate compatibility).
progress("Generating candidate...");
let candidate = generate_candidate(
    0,
    cfg.candidates_per_draft,
    cfg.topic,
    &research_digest,
    &voice_guidelines,
    &writer,
    &critic,
    &fact,
)
.await?;

total_usage += candidate.usage;

let final_draft = candidate.draft.clone();
// ...rest of existing single-candidate body through the Ok(PipelineOutput) construction...
```

**Replace with:**

```rust
// 5. Parallel candidate generation (N tasks via tokio::JoinSet).
let n = cfg.candidates_per_draft;
progress(&format!("Generating {} candidate(s) in parallel...", n));

// Wrap shared state for spawn closures (Send + Sync requirements).
let writer = std::sync::Arc::new(writer);
let critic = std::sync::Arc::new(critic);
let fact = std::sync::Arc::new(fact);
let topic_owned: String = cfg.topic.to_string();
let digest_owned = std::sync::Arc::new(research_digest.clone());
let voice_owned = std::sync::Arc::new(voice_guidelines.clone());

let mut joinset: tokio::task::JoinSet<Result<CandidateRecord, PipelineError>> =
    tokio::task::JoinSet::new();
for i in 0..n {
    let writer = writer.clone();
    let critic = critic.clone();
    let fact = fact.clone();
    let topic = topic_owned.clone();
    let digest = digest_owned.clone();
    let voice = voice_owned.clone();
    joinset.spawn(async move {
        generate_candidate(i, n, &topic, &digest, &voice, &writer, &critic, &fact).await
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
        joinset2.spawn(async move {
            generate_candidate(i, n, &topic, &digest, &voice, &writer, &critic, &fact).await
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
    let judge_out = judge.execute(&judge_msg).await.map_err(|e| {
        PipelineError::Agent {
            stage: "judge".to_string(),
            source: e,
        }
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
        parse_image_generator_output(&out.result)
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
```

- [ ] **Step 5: Build the image_generator and judge `AgentRunner`s in `run_pipeline`**

In the agent-construction section (where `researcher`, `writer`, `critic`, `fact` are built), add `judge` and `image_generator`:

**Find:**

```rust
let researcher_tools: Vec<Arc<dyn Tool>> =
    vec![Arc::new(WebSearchTool::new()), Arc::new(WebFetchTool::new())];
let researcher = runner_from_recipe(cfg.provider.clone(), researcher_recipe(), researcher_tools)
    .map_err(|e| PipelineError::Builder { stage: "researcher".to_string(), source: e })?;
let writer = runner_from_recipe(cfg.provider.clone(), writer_recipe(), Vec::new())
    .map_err(|e| PipelineError::Builder { stage: "writer".to_string(), source: e })?;
let critic = runner_from_recipe(cfg.provider.clone(), style_critic_recipe(), Vec::new())
    .map_err(|e| PipelineError::Builder { stage: "style_critic".to_string(), source: e })?;
let fact = runner_from_recipe(cfg.provider.clone(), fact_check_recipe(), Vec::new())
    .map_err(|e| PipelineError::Builder { stage: "fact_check".to_string(), source: e })?;
```

**Append:**

```rust
use crate::agents::{image_generator_recipe, judge_recipe};
use heartbit_core::tool::builtins::ImageGenerateTool;

let judge = runner_from_recipe(cfg.provider.clone(), judge_recipe(), Vec::new())
    .map_err(|e| PipelineError::Builder { stage: "judge".to_string(), source: e })?;
let image_gen_tools: Vec<Arc<dyn Tool>> = vec![Arc::new(ImageGenerateTool::new())];
let image_generator = runner_from_recipe(cfg.provider.clone(), image_generator_recipe(), image_gen_tools)
    .map_err(|e| PipelineError::Builder { stage: "image_generator".to_string(), source: e })?;
```

- [ ] **Step 6: Update existing 5 P1.3b integration tests to assert on new fields**

Each integration test asserts on the existing `final_draft`, `style_match_score`, etc. They need NO logic changes (`candidates_per_draft: 1` produces equivalent behavior modulo new fields). Add ONE assertion to each:

For `run_pipeline_happy_path_single_iteration`, after the existing assertions, add:

```rust
        assert_eq!(out.candidates.len(), 1, "single-candidate path");
        assert_eq!(out.chosen_index, 0);
        assert_eq!(out.judge_reasoning, "single candidate, no ranking needed");
        assert!(out.image.is_none(), "image_generator returned no_image or failed");
```

Wait — with `candidates_per_draft: 1`, the image_generator IS still called per AD-3. The MockProvider needs to return something for it. Update each test's MockProvider response queue:

For `run_pipeline_happy_path_single_iteration`:

```rust
        let provider = MockProvider::arc(vec![
            "Research digest:\n- AI is moving fast",                      // researcher
            "concrete short post",                                         // writer iter 1
            r#"{"verdict": "pass", "style_match_score": 0.92}"#,          // critic iter 1
            r#"{"verdict": "verified"}"#,                                  // fact_check
            "no_image",                                                    // image_generator (single-candidate path still calls it)
        ]);
```

(Add the `"no_image"` line at the end. The judge is skipped because `candidates_per_draft: 1`.)

Same pattern for the other 4 tests:

| Test | Add to provider queue |
|---|---|
| `run_pipeline_happy_path_single_iteration` | `"no_image"` (image_generator skips) |
| `run_pipeline_revise_once_then_pass` | `"no_image"` (image_generator skips) |
| `run_pipeline_max_revisions_exceeded` | (no addition — pipeline aborts before image) |
| `run_pipeline_critic_reject_aborts` | (no addition — pipeline aborts before image) |
| `run_pipeline_no_profile_snapshot_returns_error` | (no addition — pipeline aborts before any LLM call) |

- [ ] **Step 7: Add 5 new integration tests**

Inside the existing `#[cfg(test)] mod tests` block, append after the 5 P1.3b tests:

```rust
    #[tokio::test]
    async fn run_pipeline_three_candidates_judge_picks_index_1() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            // Researcher.
            "Research digest:\n- topic notes",
            // Variant 0: writer + critic + fact.
            "draft alpha distinct content",
            r#"{"verdict": "pass", "style_match_score": 0.80}"#,
            r#"{"verdict": "verified"}"#,
            // Variant 1: writer + critic + fact.
            "draft bravo with totally different framing",
            r#"{"verdict": "pass", "style_match_score": 0.92}"#,
            r#"{"verdict": "verified"}"#,
            // Variant 2: writer + critic + fact.
            "draft charlie via yet another angle",
            r#"{"verdict": "pass", "style_match_score": 0.85}"#,
            r#"{"verdict": "verified"}"#,
            // Judge.
            r#"{"chosen_index": 1, "reasoning": "bravo has more specific examples"}"#,
            // image_generator.
            r#"{"url": "https://example.com/img.png", "alt_text": "abstract"}"#,
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "diversity test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 3,
        };
        let out = run_pipeline(cfg).await.expect("3-candidate happy path");
        assert_eq!(out.candidates.len(), 3);
        assert_eq!(out.chosen_index, 1);
        assert_eq!(out.final_draft, "draft bravo with totally different framing");
        assert!(out.judge_reasoning.contains("bravo"));
        assert!(out.image.is_some());
        let image = out.image.as_ref().unwrap();
        assert_eq!(image.url, "https://example.com/img.png");
    }

    #[tokio::test]
    async fn run_pipeline_collapse_then_refill_succeeds() {
        let (_dir, profiles_root) = seed_snapshot("x");
        // Variants 0 and 1 produce IDENTICAL drafts (collapse). Variant 2 is distinct.
        // After dedup we have 2 distinct (variant 0 wins over 1). Refill spawns 1 task
        // for the missing slot; that task produces a 3rd distinct draft.
        let provider = MockProvider::arc(vec![
            "Research digest:\n- topic notes",
            // Variant 0 (kept).
            "duplicate draft text",
            r#"{"verdict": "pass", "style_match_score": 0.80}"#,
            r#"{"verdict": "verified"}"#,
            // Variant 1 (collapsed - same as variant 0).
            "duplicate draft text",
            r#"{"verdict": "pass", "style_match_score": 0.85}"#,
            r#"{"verdict": "verified"}"#,
            // Variant 2 (kept - distinct).
            "completely different distinct draft",
            r#"{"verdict": "pass", "style_match_score": 0.90}"#,
            r#"{"verdict": "verified"}"#,
            // Refill - variant 3 (distinct).
            "third distinct draft from refill",
            r#"{"verdict": "pass", "style_match_score": 0.88}"#,
            r#"{"verdict": "verified"}"#,
            // Judge over the 3 distinct ones.
            r#"{"chosen_index": 0, "reasoning": "first one"}"#,
            // image_generator.
            "no_image",
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "collapse test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 3,
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
        // All 3 candidates fail at the writer stage (mock exhausts after researcher).
        let provider = MockProvider::arc(vec![
            "Research digest:\n- topic notes",
            // Mock exhausts here. All 3 candidate writer.execute() calls fail with
            // Error::Agent("mock exhausted").
        ]);
        let cfg = PipelineConfig {
            persona_name: "x",
            topic: "fail test",
            provider,
            corpora_root: &profiles_root,
            profiles_root: &profiles_root,
            on_progress: None,
            candidates_per_draft: 3,
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
        // returns "no_image"; PipelineOutput.image must be None.
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
        };
        let out = run_pipeline(cfg).await.expect("happy path with no_image");
        assert_eq!(out.candidates.len(), 1);
        assert!(out.image.is_none());
        // Sanity: judge was skipped because N=1.
        assert_eq!(out.judge_reasoning, "single candidate, no ranking needed");
    }
```

Note: the 4th new test (`run_pipeline_invalid_candidates_per_draft_rejected`) passes `candidates_per_draft: 0` — assert it returns `InvalidConfig` BEFORE any LLM call.

Note 2: `MockProvider`'s structured-output detection sees `__respond__` in `request.tools` for the `judge`, `style_critic`, and `fact_check` agents (all have `response_schema`). The `image_generator` has NO `response_schema`, so it returns plain text. The `"no_image"` and `"https://example.com/img.png"` responses for image_generator go through the plain-text path — verify the MockProvider does the right thing.

If the MockProvider currently can't distinguish (i.e., always emits ToolUse when `__respond__` is in tools), the issue is that image_generator has the `image_generate` tool in its tool set (not `__respond__`), so the mock should fall through to plain text. **This works as-is** assuming the mock checks for `__respond__` specifically by name. Verify.

For the JSON image case (`{"url": ..., "alt_text": ...}`), the mock returns it as plain text. `parse_image_generator_output` handles JSON parsing.

- [ ] **Step 8: Run all pipeline tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-ghost --lib pipeline 2>&1 | tail -15
```

Expected: `43 passed` (38 from Tasks 1+2 + 5 new integration tests). If the 4 new judge-using integration tests fail with "mock exhausted" or similar, double-check the response queue order.

The `run_pipeline_three_candidates_judge_picks_index_1` test response queue order is critical — researcher first, then per-candidate writer/critic/fact (3 each), then judge, then image_generator. The MockProvider pops responses in FIFO order across all calls; per-candidate parallelism means responses can be consumed in non-deterministic order across the 3 writer/critic/fact triples.

**This is a real correctness concern.** With 3 parallel candidates each making 3 LLM calls, the MockProvider's single FIFO queue may interleave consumers — variant 1's writer might pop variant 0's critic response. To make tests deterministic, the test infrastructure needs per-variant queues OR the test setup must ensure all 9 (or 12) per-candidate responses are interchangeable.

**Quick fix:** make the canned responses for variants identical by structure (each is a writer-text + critic-pass + fact-verified triple) but distinguish them only by content where the test checks for distinctness. The test asserting `chosen_index == 1` and `final_draft == "draft bravo..."` only works if the test can guarantee the judge sees variant 1 = "draft bravo".

**Robust fix:** extend `MockProvider` to track call counts per agent name, so each agent gets a deterministic sequence. The agent name is in `CompletionRequest.system` or similar. Adding per-agent queues is a ~30-LOC test-helper change.

**Decision:** for Task 3, use the simpler per-agent-name routing. The mock checks the system prompt prefix (e.g., "You are a writer ...") to dispatch to the right queue. This is a one-time test-helper improvement that future P1.3 tasks will benefit from.

Update `MockProvider` (in the test mod) to have separate queues per agent name. The agent name is identifiable by matching against the system prompt's first line or against a distinguishing substring:

```rust
struct MockProvider {
    /// Responses keyed by a substring that uniquely identifies the
    /// requester. Match is FIRST-substring-match wins.
    routes: std::sync::Mutex<Vec<(String, std::collections::VecDeque<String>)>>,
}

impl MockProvider {
    fn route(routes: Vec<(&str, Vec<&str>)>) -> std::sync::Arc<BoxedProvider> {
        let mapped: Vec<(String, std::collections::VecDeque<String>)> = routes
            .into_iter()
            .map(|(key, responses)| {
                (
                    key.to_string(),
                    responses.into_iter().map(String::from).collect(),
                )
            })
            .collect();
        let p = MockProvider {
            routes: std::sync::Mutex::new(mapped),
        };
        std::sync::Arc::new(BoxedProvider::new(p))
    }

    /// Backward-compat helper: legacy single-queue version. Internally
    /// routes everything to a single "*" key matched as a wildcard.
    fn arc(responses: Vec<&str>) -> std::sync::Arc<BoxedProvider> {
        Self::route(vec![("", responses)])
    }
}

impl LlmProvider for MockProvider {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, CoreError>> + Send {
        // Find which route this request belongs to by matching the system
        // prompt against route keys. First non-empty key with a substring
        // match wins; "" key is the wildcard fallback.
        let system = request.system.as_deref().unwrap_or("");
        let mut routes = self.routes.lock().unwrap();
        let chosen_idx = routes
            .iter()
            .position(|(key, _)| !key.is_empty() && system.contains(key.as_str()))
            .or_else(|| routes.iter().position(|(key, _)| key.is_empty()));

        let response = match chosen_idx.and_then(|i| {
            let q = &mut routes[i].1;
            q.pop_front()
        }) {
            Some(s) => Some(s),
            None => None,
        };

        // Determine if this is a structured (`__respond__`) call.
        let has_respond = request
            .tools
            .iter()
            .any(|t| t.name == heartbit_core::llm::types::RESPOND_TOOL_NAME);
        async move {
            let text = response.ok_or_else(|| CoreError::Agent("mock exhausted".to_string()))?;
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
```

Existing tests using `MockProvider::arc(vec![...])` keep working (wildcard route). New 3-candidate test uses `MockProvider::route([("writer", [...]), ("style_critic", [...]), ("fact_check", [...]), ("judge", [...]), ("image", [...]), ("researcher", [...])])` — the keys match against the system prompt's distinguishing name.

The 3-candidate test re-written:

```rust
    #[tokio::test]
    async fn run_pipeline_three_candidates_judge_picks_index_1() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::route(vec![
            ("research substantive material", vec!["Research digest:\n- topic notes"]),
            (
                "social media writer",
                vec![
                    "draft alpha distinct content",
                    "draft bravo with totally different framing",
                    "draft charlie via yet another angle",
                ],
            ),
            (
                "evaluate one social media draft against",
                vec![
                    r#"{"verdict": "pass", "style_match_score": 0.80}"#,
                    r#"{"verdict": "pass", "style_match_score": 0.92}"#,
                    r#"{"verdict": "pass", "style_match_score": 0.85}"#,
                ],
            ),
            (
                "fact-check a social media draft",
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
        // ... rest unchanged
    }
```

The route keys are substrings unique to each P1.3a recipe's system prompt. They're stable as long as the recipes' system prompts don't drift.

Verify each substring against the actual recipe system prompts (Task 1 already exposes them via the `*_recipe()` factory functions → `agents/*.rs`).

**Note:** writer responses are still consumed in queue order regardless of which variant is which (parallel writers all hit the same "social media writer" key). The variant→draft binding in the test depends on the order the writers complete. With `tokio::JoinSet` and identical sub-tasks, completion order is non-deterministic. **The test asserts `chosen_index == 1` and `final_draft == "draft bravo..."` only if the responses are consumed in order 0→1→2.**

The test's correctness depends on: the writers all kick off near-simultaneously, the responses are consumed FIFO, and `candidates.sort_by_key(|c| c.variant_index)` post-collection re-orders by the original variant_idx — which is set by the `for i in 0..n` spawn loop, NOT by completion order. So variant 0's writer pops the first writer response; variant 1's writer pops the second.

But variants don't strictly pop in 0→1→2 order from a shared queue. If variant 1's writer task gets scheduled first (mock is FIFO), it pops "draft alpha" — and now `candidate.draft = "draft alpha"` while `candidate.variant_index = 1`. The test's `final_draft` assertion of "draft bravo" no longer holds.

**This is a real test-isolation issue.** The simplest robust fix: have the mock routes index by call count AND by route key, but assign canned responses based on the request body's content (not call order). Specifically: for the writer route, match on the request's user message content for "variant 1 of 3" (etc.) substring.

**Alternative simpler fix:** make all 3 writer responses identical in structure, and the test asserts `chosen_index` is whatever the judge picks, and `final_draft` is `candidates[chosen_index].draft`. We don't bind specific variants to specific drafts. The test still validates the integration end-to-end.

```rust
        // Use distinct content per variant; assertion checks the chain works,
        // not which specific draft each variant got.
        // The judge just picks one; we assert chosen_index is reachable and
        // image generation worked.
        // Note: with 3 parallel writers consuming a shared FIFO queue, draft
        // ↔ variant_index binding is non-deterministic — that's why we don't
        // assert a specific binding.
```

Update assertions:

```rust
        let out = run_pipeline(cfg).await.expect("3-candidate happy path");
        assert_eq!(out.candidates.len(), 3);
        assert_eq!(out.chosen_index, 1);  // judge picked index 1 deterministically
        // final_draft equals candidates[1].draft — content depends on which
        // variant got "draft bravo", but it's always SOME draft from the queue.
        assert!(["draft alpha distinct content", "draft bravo with totally different framing", "draft charlie via yet another angle"]
            .contains(&out.final_draft.as_str()));
        assert!(out.judge_reasoning.contains("bravo"));
        assert!(out.image.is_some());
```

This makes the test deterministic on `chosen_index` and `image` while accepting any of the 3 drafts as the final_draft (since the FIFO order isn't enforced across parallel tasks).

- [ ] **Step 9: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 10: Commit**

```bash
cd /home/pleclech/projects/heartbit
git add crates/heartbit-ghost/src/pipeline/prompts.rs \
        crates/heartbit-ghost/src/pipeline/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): pipeline — parallel multi-candidate + dedup + judge + image_generator (P1.3c)

Replace P1.3b's single-candidate body of `run_pipeline` with the full
multi-candidate orchestration:

- Parallel candidate generation via `tokio::JoinSet` — N writers run
  concurrently. AgentRunner wrapped in Arc for spawn closures
  (Send + 'static requirement). Per-candidate failures collected;
  AllCandidatesFailed only when ALL fail. Sort by variant_index after
  collection for stable ordering.

- Levenshtein dedup with bounded retry: drafts with ratio > 0.85 are
  treated as duplicates; lower variant_index wins. On collapse, spawn
  one refill pass for the missing slots (variant_index continues from
  max+1). Ship-with-fewer if still collapsed after refill.

- Judge integration: build_judge_user_message renders numbered
  candidate list. parse_judge_verdict validates chosen_index against
  candidates.len(). Judge skipped (chosen_index = 0) when survivors
  collapse to 1, with informative judge_reasoning. Saves an LLM call
  for the candidates_per_draft: 1 path.

- image_generator integration: always runs on chosen draft;
  parse_image_generator_output handles "no_image" → None, JSON
  {url, alt_text} → Some, fallback first-URL extraction → Some,
  failure → None (non-blocking).

- prompts.rs: new build_judge_user_message,
  build_image_generator_user_message.

- Test-helper improvement: MockProvider gains per-agent routing via
  system-prompt substring matching. Existing wildcard-route helper
  preserved for back-compat. Necessary for the 3-candidate integration
  test where the judge / image_generator / writer / critic / fact
  routes must be answered with distinct response queues.

5 new integration tests: 3-candidate happy path, collapse+refill,
all-fail, invalid-config, no_image-yields-None. All 5 P1.3b
integration tests pass with `candidates_per_draft: 1` after adding a
"no_image" canned response for image_generator.

48 pipeline tests pass total. Workspace count: 3973 → 3992 (the +6
new tests + ~13 from Tasks 1+2 = 19 net gain).

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3c-multi-candidate-design.md §2, §4, §5, §6, §7

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CLI summary tweak + final acceptance

**Why:** Update the `persona run` post-pipeline summary log to surface candidate count and chosen index. Verify workspace-wide quality gate. No new code other than a small log line — most of P1.3c is in pipeline/.

**Files:**
- Modify: `crates/heartbit-cli/src/persona.rs`

- [ ] **Step 1: Update CLI post-pipeline summary**

In `crates/heartbit-cli/src/persona.rs`, find the `Run` arm's summary line (looks like `eprintln!("> ok: revise iterations={}, style match={:.2}, fact check={:?}", ...)`) and replace with:

```rust
            eprintln!(
                "> ok: candidates={}/{}, chosen={}, revise iterations={}, style match={:.2}, fact check={:?}, image={}",
                output.candidates.len(),
                cfg.candidates_per_draft,  // wait — cfg is moved into run_pipeline
                output.chosen_index,
                output.revise_iterations,
                output.style_match_score,
                output.fact_check_verdict,
                output.image.as_ref().map(|i| i.url.as_str()).unwrap_or("none"),
            );
```

Wait — `cfg` is moved into `run_pipeline(cfg).await`. We need to remember `candidates_per_draft` before the move:

```rust
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
                output.image.as_ref().map(|i| i.url.as_str()).unwrap_or("none"),
            );
            Ok(())
```

Also: `cfg` is currently constructed without `candidates_per_draft` (P1.3b code). Add it:

```rust
            let cfg = heartbit_ghost::pipeline::PipelineConfig {
                persona_name: &name,
                topic: &once,
                provider,
                corpora_root: &corpora_root,
                profiles_root: &profiles_root,
                on_progress: Some(on_progress),
                candidates_per_draft: 3,  // P1.3c default
            };
```

- [ ] **Step 2: Run the existing CLI tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-cli --bin heartbit persona 2>&1 | tail -10
```

Expected: 16 tests pass (existing 14 + 2 P1.3b dispatch tests). No new tests in P1.3c — the dispatch tests don't exercise the pipeline body, and the new `candidates_per_draft` field is a literal constant in the CLI body.

- [ ] **Step 3: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -p heartbit-cli -- --check
cargo clippy -p heartbit-cli --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 4: Commit**

```bash
cd /home/pleclech/projects/heartbit
git add crates/heartbit-cli/src/persona.rs
git commit -m "$(cat <<'EOF'
feat(cli): persona run — wire P1.3c multi-candidate pipeline (P1.3c)

CLI body adds `candidates_per_draft: 3` to PipelineConfig (P1.3c
default). Post-pipeline summary on stderr now surfaces candidate
count, chosen index, and image URL (or "none"). Capture
`cfg.candidates_per_draft` into a local before the cfg move so the
summary can compare requested vs. delivered count (which can differ
when dedup collapses + refill leaves fewer than N).

run_pipeline already prints the chosen final_draft to stdout — same
contract as P1.3b. Pipe-friendly: the chosen draft is the only stdout
output, all progress + summary goes to stderr.

No new tests; the 2 existing dispatch tests short-circuit at the
registry check before pipeline body runs.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3c-multi-candidate-design.md §12

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final acceptance + workspace quality gate + final review

**Why:** Confirm P1.3c meets every acceptance criterion. Verification only — no commit.

**Files:** none.

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -10
```

Expected: all green. Workspace test count: 3973 (post-P1.3b baseline) → **~3992** (+19 net new tests):
- Task 1: 11 (4 verdicts + 7 dedup)
- Task 2: 0 (extraction, no test changes)
- Task 3: 5 integration tests
- Task 4: 0 (no new tests)
- Plus a few cascading test count tweaks.

(Spec said "~15 new tests" — we're slightly over due to the unicode dedup test, the explicit invalid-config test, and the no_image test. All justified.)

- [ ] **Step 2: Verify the public surface is reachable**

```bash
cd /home/pleclech/projects/heartbit
cat > /tmp/p1_3c_surface_check.rs <<'EOF'
fn _check() {
    use heartbit_ghost::pipeline::{
        PipelineConfig, PipelineOutput, PipelineError,
        FactVerdict, StyleVerdict, JudgeVerdict, VerdictParseError,
        PublishGateError, CandidateRecord, ImageAttachment,
        check_publish_gate, parse_critic_verdict, parse_fact_verdict, parse_judge_verdict,
        render_style_profile_as_english, run_pipeline,
    };
    let _ = JudgeVerdict { chosen_index: 0, reasoning: String::new() };
    let _ = PipelineError::AllCandidatesFailed { errors: vec![], n: 3 };
    let _ = ImageAttachment { url: String::new(), alt_text: None };
    let _ = CandidateRecord {
        variant_index: 0,
        draft: String::new(),
        style_match_score: 0.0,
        revise_iterations: 1,
        fact_check_verdict: FactVerdict::Verified,
        usage: heartbit_core::llm::types::TokenUsage::default(),
    };
}
EOF
echo "(surface check is illustrative; cargo build covers it)"
rm -f /tmp/p1_3c_surface_check.rs
```

- [ ] **Step 3: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.3c
```

Expected: 5 commits — spec doc + 4 task commits.

- [ ] **Step 4: No commit for this task**

Task 5 is verification only. The branch is ready for final review + merge. P1.3c complete; P1.3d (Telegram review delivery + publisher + autonomy phase 0) is the next sub-phase.

---

## Acceptance criteria

P1.3c is done when (per spec §12):

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features` and `cargo check -p heartbit-cli --all-features`.
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green.
- ~19 net new tests pass (4 verdicts + 7 dedup + 5 integration + 3 edge tests; all 5 P1.3b integration tests still pass with `candidates_per_draft: 1`).
- New public surface from `heartbit_ghost::pipeline`: `CandidateRecord`, `ImageAttachment`, `JudgeVerdict`, `VerdictParseError`, `parse_judge_verdict` (all reachable as documented in §12.3 of the spec).
- `heartbit persona run heartbit-ghost:x --once "<topic>"` produces 3 candidate drafts, judge picks one, image_generator runs (returns `no_image` or attaches an image URL), final chosen draft prints to stdout, candidate count + chosen index summary on stderr.
- `parse_critic_verdict` / `parse_fact_verdict` return `Result<_, VerdictParseError>` (no `serde_json::Error` in public signatures).

## Out of scope (re-stated from spec §13)

- Telegram review delivery → P1.3d
- `publisher` recipe usage / actual posting → P1.3d
- Pick storage → P1.3e
- Exemplar pool / few-shot rotation from corpora → P1.3e
- Autonomy phase logic → P1.3d / P1.4
- Audit log integration → P1.4
- LLM-based content guardrails → P1.4
- Trigger specs → P1.4

## Reference

- Spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3c-multi-candidate-design.md`
- P1.3a recipes (consumed by Task 3): `crates/heartbit-ghost/src/agents/{judge,image_generator}.rs`
- P1.3b plan (predecessor): `docs/superpowers/plans/2026-05-08-heartbit-ghost-p1.3b-pipeline-orchestrator.md`
- `tokio::JoinSet` reference usage: `crates/heartbit-core/src/agent/orchestrator.rs`
- `RESPOND_TOOL_NAME` constant: `crates/heartbit-core/src/llm/types.rs:11`
