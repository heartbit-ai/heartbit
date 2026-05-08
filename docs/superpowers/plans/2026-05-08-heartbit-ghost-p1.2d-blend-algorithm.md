# heartbit-ghost P1.2d — blend algorithm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pure deterministic transformation that takes a `BlendRecipe` (P1.2a) plus a map of writer-handle → `StyleProfile` and produces one merged, validated `StyleProfile` per the rules in umbrella spec §2.3.

**Architecture:** Extends the existing `crates/heartbit-ghost/src/voice/blend.rs` (P1.2a). Adds `blend_profiles()` free function + `BlendError` enum + 6 private helpers (`distribute_largest_remainder`, `weighted_vote_categorical`, `opening_pattern_decl_index`, `merge_opening_patterns`, `union_dedup_strings`, `apply_overrides`). All in one file, all in-tree tests, no async, no I/O.

**Tech Stack:** Rust 2024, stdlib `HashMap`, `thiserror`, plus the existing `voice::style` types (`StyleProfile` + the 11 closed-vocab enums + `Formatting`).

---

## File structure

| File | Responsibility |
|------|----------------|
| `crates/heartbit-ghost/src/voice/blend.rs` | All P1.2d additions: `BlendError`, `blend_profiles`, 5 private helpers, all 26 P1.2d tests, the `mk_profile` / `mk_recipe` / `mk_profiles` test helpers |
| `crates/heartbit-ghost/src/voice/mod.rs` | Add `blend_profiles` + `BlendError` to the existing extractor re-exports |

4 tasks total: 3 implementation + 1 final acceptance (verification only, no commit).

---

## Task 1: Scaffolding + `BlendError`

**Why:** Get `BlendError` into the module surface so subsequent tasks can return it. Mirrors P1.2c Task 1's scaffolding pattern.

**Files:**
- Modify: `crates/heartbit-ghost/src/voice/blend.rs` (append `BlendError` + 3 tests)
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (add `BlendError` re-export)

- [ ] **Step 1: Append `BlendError` to `crates/heartbit-ghost/src/voice/blend.rs`**

Find the existing `impl BlendRecipe { ... }` block (currently around lines 103–160 with `from_toml` and `validate`) and append the following AFTER it (before the existing `#[cfg(test)] mod tests` block):

```rust
// ─────────────────────────────────────────────────────────────────────────
// P1.2d — blend algorithm
// ─────────────────────────────────────────────────────────────────────────

/// Errors raised by [`blend_profiles`].
#[derive(Debug, thiserror::Error)]
pub enum BlendError {
    /// The recipe references a writer handle that is not in the profiles map.
    #[error("missing profile for writer '{0}'")]
    MissingProfile(String),

    /// Result of merge + override application failed
    /// [`crate::voice::StyleProfile::validate`]. `inner` is the underlying
    /// validation error. Most commonly this surfaces when a user-supplied
    /// `recipe.overrides` violates a parallel-array length, sum-to-100 /
    /// sum-to-1.0, or range invariant.
    #[error("merged profile failed validation: {inner}")]
    PostMergeValidation {
        /// The underlying validation error from `StyleProfile::validate`.
        #[source]
        inner: VoiceError,
    },
}
```

(`VoiceError` is already in scope via the existing `use crate::voice::error::VoiceError;` import at the top of the file. `thiserror` is a workspace dep already pulled in by P1.2a's `BlendRecipe`.)

- [ ] **Step 2: Append 3 BlendError tests to the existing `#[cfg(test)] mod tests` block**

The current tests block contains 12 P1.2a tests. Append the new tests at the END of the block, BEFORE the closing `}`:

```rust
    // ─────────────────────────────────────────────────────────────────────
    // P1.2d — BlendError display
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn missing_profile_error_includes_writer_handle() {
        let e = BlendError::MissingProfile("karpathy".to_string());
        let s = format!("{e}");
        assert!(s.contains("karpathy"), "got: {s}");
        assert!(s.starts_with("missing profile"), "got: {s}");
    }

    #[test]
    fn post_merge_validation_error_renders_inner_message() {
        let inner = VoiceError::Validation("thread_max_length must be 1..=25 (got 30)".to_string());
        let e = BlendError::PostMergeValidation { inner };
        let s = format!("{e}");
        assert!(s.starts_with("merged profile failed validation:"), "got: {s}");
        assert!(s.contains("thread_max_length"), "got: {s}");
        assert!(s.contains("got 30"), "got: {s}");
    }

    #[test]
    fn post_merge_validation_inner_is_reachable() {
        let inner = VoiceError::Validation("test message".to_string());
        let e = BlendError::PostMergeValidation { inner };
        if let BlendError::PostMergeValidation { inner: i } = &e {
            assert!(matches!(i, VoiceError::Validation(_)));
        } else {
            panic!("not PostMergeValidation");
        }
    }
```

- [ ] **Step 3: Modify `crates/heartbit-ghost/src/voice/mod.rs` to re-export `BlendError`**

The current state of the `pub use blend::...` line (after P1.2a + P1.2c) is:

```rust
pub use blend::{BlendEntry, BlendRecipe, PartialStyleProfile};
```

Replace with:

```rust
pub use blend::{BlendEntry, BlendError, BlendRecipe, PartialStyleProfile};
```

(Alphabetical, rustfmt-stable. Task 3 will append `blend_profiles` to the same set.)

- [ ] **Step 4: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::blend
```

Expected: `15 passed; 0 failed; 0 ignored` (12 from P1.2a + 3 new).

- [ ] **Step 5: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/voice/blend.rs crates/heartbit-ghost/src/voice/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — blend module BlendError scaffolding (P1.2d)

The error enum that blend_profiles (Task 3) will return. Two variants
cover the two distinct failure modes:
- MissingProfile(writer): recipe references a writer not in the
  profiles map. Carries the writer handle for debuggability.
- PostMergeValidation { inner: VoiceError }: merged profile failed
  StyleProfile::validate. inner wraps the underlying schema error.

3 tests on BlendError display + reachability.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2d-blend-algorithm-design.md §3
EOF
)"
```

---

## Task 2: `distribute_largest_remainder` Hare-quota helper

**Why:** The most algorithmically rich helper. Unit-testing it in isolation is valuable — it pins down the "declaration-order tiebreak in the residual distribution" behavior independently of `blend_profiles` plumbing. The other 4 helpers in Task 3 are tested indirectly via the integration tests.

**Files:**
- Modify: `crates/heartbit-ghost/src/voice/blend.rs` (append `distribute_largest_remainder` + 2 tests)

- [ ] **Step 1: Append `distribute_largest_remainder` to `crates/heartbit-ghost/src/voice/blend.rs` (after `BlendError`, before the `#[cfg(test)] mod tests` block)**

```rust
/// Hare quota (largest-remainder method): distribute `total` units across
/// 4 buckets according to f64 weights, producing 4 integer values that
/// sum to exactly `total`.
///
/// Algorithm:
/// 1. Floor each weighted f64 to u32, accumulate `assigned`.
/// 2. residual = total - assigned (in 0..4).
/// 3. Sort buckets by (frac DESC, declaration index ASC).
/// 4. Add 1 to each of the first `residual` buckets in that order.
#[allow(dead_code)] // wired in by Task 3's blend_profiles
fn distribute_largest_remainder(weighted: [f64; 4], total: u32) -> [u8; 4] {
    let floors: [u32; 4] = [
        weighted[0] as u32,
        weighted[1] as u32,
        weighted[2] as u32,
        weighted[3] as u32,
    ];
    let assigned: u32 = floors[0] + floors[1] + floors[2] + floors[3];
    let residual = total.saturating_sub(assigned);

    let mut fracs: [(usize, f64); 4] = [(0, 0.0); 4];
    for i in 0..4 {
        fracs[i] = (i, weighted[i] - floors[i] as f64);
    }
    // Sort: largest fractional remainder first; declaration-order tiebreak.
    fracs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let mut out: [u8; 4] = [0; 4];
    for i in 0..4 {
        out[i] = floors[i].min(255) as u8;
    }
    let extras = (residual as usize).min(4);
    for r in 0..extras {
        let idx = fracs[r].0;
        out[idx] = out[idx].saturating_add(1);
    }
    out
}
```

Notes:
- The `#[allow(dead_code)]` is required because the function has no caller until Task 3 wires it into `blend_profiles`. **Task 3 must remove this attribute.**
- The `.min(255)` and `.saturating_add(1)` guards are defensive against malformed input; in practice, `weighted` values should always be in `0.0..=100.0` and the residual in `0..=4`, so the saturation never fires. They prevent any chance of u32→u8 truncation or u8 overflow.

- [ ] **Step 2: Append the 2 unit tests inside the existing `#[cfg(test)] mod tests` block**

Append after the 3 BlendError tests added in Task 1:

```rust
    // ─────────────────────────────────────────────────────────────────────
    // P1.2d — distribute_largest_remainder (Hare quota)
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn distribute_largest_remainder_already_summing_to_total_is_identity() {
        // 0.6 × [40, 30, 20, 10] + 0.4 × [20, 30, 30, 20] = [32.0, 30.0, 24.0, 14.0]
        // Floors sum to 100, residual is 0, no redistribution needed.
        let result = distribute_largest_remainder([32.0, 30.0, 24.0, 14.0], 100);
        assert_eq!(result, [32, 30, 24, 14]);
        assert_eq!(result.iter().map(|&x| x as u32).sum::<u32>(), 100);
    }

    #[test]
    fn distribute_largest_remainder_residual_uses_declaration_order_tiebreak() {
        // Weighted: [33.5, 33.0, 33.0, 0.5] — floors [33, 33, 33, 0] sum 99,
        // residual 1. Fractional parts: 0.5, 0.0, 0.0, 0.5 — buckets 0 and 3
        // tie at 0.5. Declaration-order tiebreak: bucket 0 wins, gets the +1.
        let result = distribute_largest_remainder([33.5, 33.0, 33.0, 0.5], 100);
        assert_eq!(result, [34, 33, 33, 0]);
        assert_eq!(result.iter().map(|&x| x as u32).sum::<u32>(), 100);
    }
```

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::blend
```

Expected: `17 passed; 0 failed; 0 ignored` (15 from prior tasks + 2 new).

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean. (`--all-targets` exercises the test code; tests are now non-trivial.)

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/voice/blend.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — blend Hare-quota distribute_largest_remainder helper (P1.2d)

The most algorithmically rich helper of the blender. Distributes a total
across 4 buckets according to f64 weights, producing 4 integers that sum
to exactly the total. Floor each, then assign the residual one unit at
a time to the buckets with the largest fractional parts; declaration-
order tiebreak when fractional parts are equal.

Carries #[allow(dead_code)] until Task 3's blend_profiles wires it in.

2 direct unit tests: identity (no residual), residual + declaration-
order tiebreak (the [33.5, 33.0, 33.0, 0.5] adversarial case).

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2d-blend-algorithm-design.md §5.1
EOF
)"
```

---

## Task 3: `blend_profiles` body + remaining helpers + integration tests

**Why:** The orchestration. Pulls everything together: 4 more private helpers, the main 16-field merge body, override application, post-merge validation. Adds 21 integration tests covering every merge rule, every error path, the 5-writer canonical example, and the determinism guarantee. The largest task.

**Files:**
- Modify: `crates/heartbit-ghost/src/voice/blend.rs` (append 4 helpers + `blend_profiles` + test helpers + 21 tests; remove the `#[allow(dead_code)]` from `distribute_largest_remainder`)
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (extend `pub use blend::...` with `blend_profiles`)

### Step 1: Remove `#[allow(dead_code)]` from `distribute_largest_remainder`

In `voice/blend.rs`, find the line `#[allow(dead_code)] // wired in by Task 3's blend_profiles` immediately above `fn distribute_largest_remainder` and **delete that attribute line**.

### Step 2: Update top-of-file imports

The current top-of-file imports are:

```rust
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::voice::error::VoiceError;
use crate::voice::style::{
    EmDashPolicy, EmojiPolicy, FragmentFrequency, Formatting, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    ThreadRhythm,
};
```

Add `use std::collections::HashMap;` and `use crate::voice::style::StyleProfile;`. Final state:

```rust
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::voice::error::VoiceError;
use crate::voice::style::{
    EmDashPolicy, EmojiPolicy, FragmentFrequency, Formatting, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
```

(rustfmt may sort `StyleProfile` alphabetically with the other imports — let it.)

### Step 3: Append the 4 remaining helpers (above the `#[cfg(test)] mod tests` block, after `distribute_largest_remainder`)

```rust
/// Vote across enum variants in `order` (declaration order). Returns the
/// variant with the highest summed weight; ties resolved by `order`.
///
/// Each profile contributes its `pick(profile)` choice with weight
/// `recipe.blend[i].weight`. Earlier-declared variants win ties.
fn weighted_vote_categorical<E: Copy + PartialEq>(
    blend: &[BlendEntry],
    profiles: &HashMap<String, StyleProfile>,
    pick: impl Fn(&StyleProfile) -> E,
    order: &[E],
) -> Result<E, BlendError> {
    let mut weights: Vec<f64> = vec![0.0; order.len()];
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        let chosen = pick(profile);
        if let Some(idx) = order.iter().position(|v| *v == chosen) {
            weights[idx] += entry.weight;
        }
    }
    // Seed with order[0]; strict-> for declaration-order tiebreak.
    let mut best_idx = 0usize;
    let mut best_w = weights[0];
    for (i, &w) in weights.iter().enumerate().skip(1) {
        if w > best_w {
            best_w = w;
            best_idx = i;
        }
    }
    Ok(order[best_idx])
}

/// Declaration-order index of an `OpeningPattern` variant (0..7).
/// Used as a sort tiebreak in [`merge_opening_patterns`].
fn opening_pattern_decl_index(p: &OpeningPattern) -> usize {
    match p {
        OpeningPattern::ClaimFirst => 0,
        OpeningPattern::NumberFirst => 1,
        OpeningPattern::SceneFirst => 2,
        OpeningPattern::QuestionFirst => 3,
        OpeningPattern::AphoristicFirst => 4,
        OpeningPattern::AnecdoteFirst => 5,
        OpeningPattern::ContrarianFirst => 6,
        // `#[non_exhaustive]` requires this fallback; future variants
        // sort to the bottom until added explicitly.
        _ => usize::MAX,
    }
}

/// Merge per-profile (opening_patterns, opening_pattern_weights) parallel
/// arrays into a single normalized parallel pair. Output is sorted by
/// (weight DESC, declaration-order ASC).
fn merge_opening_patterns(
    blend: &[BlendEntry],
    profiles: &HashMap<String, StyleProfile>,
) -> Result<(Vec<OpeningPattern>, Vec<f64>), BlendError> {
    let mut acc: Vec<(OpeningPattern, f64)> = Vec::new();
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        for (pat, w) in profile
            .opening_patterns
            .iter()
            .zip(profile.opening_pattern_weights.iter())
        {
            match acc.iter_mut().find(|(p, _)| p == pat) {
                Some(slot) => slot.1 += entry.weight * w,
                None => acc.push((*pat, entry.weight * w)),
            }
        }
    }
    // Defensive normalization against f64 drift.
    let total: f64 = acc.iter().map(|(_, w)| *w).sum();
    if total > 0.0 && (total - 1.0).abs() > 1e-9 {
        for (_, w) in acc.iter_mut() {
            *w /= total;
        }
    }
    // Sort by (weight DESC, declaration-order ASC).
    acc.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(opening_pattern_decl_index(&a.0).cmp(&opening_pattern_decl_index(&b.0)))
    });
    let (patterns, weights) = acc.into_iter().unzip();
    Ok((patterns, weights))
}

/// Union + dedup over `recipe.blend` iteration order. First occurrence
/// (across the recipe.blend iteration) wins position. Case-sensitive.
fn union_dedup_strings(
    blend: &[BlendEntry],
    profiles: &HashMap<String, StyleProfile>,
    pick: impl Fn(&StyleProfile) -> &[String],
) -> Result<Vec<String>, BlendError> {
    let mut out: Vec<String> = Vec::new();
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        for s in pick(profile) {
            if !out.iter().any(|existing| existing == s) {
                out.push(s.clone());
            }
        }
    }
    Ok(out)
}

/// Apply [`PartialStyleProfile`] overrides to a merged `StyleProfile`.
/// For each `Some(value)` field, replace the merged value unconditionally.
fn apply_overrides(merged: &mut StyleProfile, overrides: &PartialStyleProfile) {
    if let Some(v) = overrides.sentence_length_target {
        merged.sentence_length_target = v;
    }
    if let Some(v) = overrides.sentence_length_distribution {
        merged.sentence_length_distribution = v;
    }
    if let Some(v) = overrides.fragment_frequency {
        merged.fragment_frequency = v;
    }
    if let Some(v) = overrides.opening_patterns.as_ref() {
        merged.opening_patterns = v.clone();
    }
    if let Some(v) = overrides.opening_pattern_weights.as_ref() {
        merged.opening_pattern_weights = v.clone();
    }
    if let Some(v) = overrides.formatting {
        merged.formatting = v;
    }
    if let Some(v) = overrides.emoji_policy {
        merged.emoji_policy = v;
    }
    if let Some(v) = overrides.hashtag_policy {
        merged.hashtag_policy = v;
    }
    if let Some(v) = overrides.specificity_target {
        merged.specificity_target = v;
    }
    if let Some(v) = overrides.voice_traits.as_ref() {
        merged.voice_traits = v.clone();
    }
    if let Some(v) = overrides.ai_tells_to_avoid.as_ref() {
        merged.ai_tells_to_avoid = v.clone();
    }
    if let Some(v) = overrides.thread_rhythm {
        merged.thread_rhythm = v;
    }
    if let Some(v) = overrides.thread_max_length {
        merged.thread_max_length = v;
    }
    if let Some(v) = overrides.thread_opener_must_hook {
        merged.thread_opener_must_hook = v;
    }
    if let Some(v) = overrides.topical_obsessions.as_ref() {
        merged.topical_obsessions = v.clone();
    }
    if let Some(v) = overrides.topical_avoidances.as_ref() {
        merged.topical_avoidances = v.clone();
    }
}
```

Notes:
- `weighted_vote_categorical` is generic over `E: Copy + PartialEq`. The closed-vocab enums (`SentenceLengthTarget`, `EmojiPolicy`, etc.) all implement these traits. `bool` also satisfies them, so the same helper works for `formatting.lowercase` and `thread_opener_must_hook` with the order `&[false, true]` — earlier `false` wins ties (the boolean tiebreak is `false` per AD-5).
- `opening_pattern_decl_index` has a `_ => usize::MAX` fallback because `OpeningPattern` is `#[non_exhaustive]`. Future variants would sort to the bottom until the match arm is updated.
- `apply_overrides` clones strings/vectors for the list overrides because `PartialStyleProfile` owns those fields. The blender takes a `&PartialStyleProfile` so we can't move out.

### Step 4: Append `blend_profiles` (after the helpers, before tests)

```rust
/// Apply a [`BlendRecipe`] to a map of writer-handle → [`StyleProfile`],
/// producing one merged, validated [`StyleProfile`].
///
/// Per umbrella spec §2.3:
/// - Numeric fields → weighted average (Hare quota for the
///   `sentence_length_distribution` array)
/// - Categorical (enum + bool) fields → weighted vote, declaration-order
///   tiebreak (or `false` for booleans)
/// - List-of-strings fields → union, deduped, stable insertion order
/// - `opening_patterns` + `opening_pattern_weights` → weighted accumulator
///   per pattern, normalized to sum 1.0
/// - `recipe.overrides` (a [`PartialStyleProfile`]) → replaces blended
///   values where `Some(_)`, applied last
/// - Result is validated via [`StyleProfile::validate`] before returning
///
/// `recipe.validate()` is **NOT** called by this function; callers should
/// validate the recipe before passing it in (or rely on `from_toml`'s
/// internal validation).
pub fn blend_profiles(
    recipe: &BlendRecipe,
    profiles: &HashMap<String, StyleProfile>,
) -> Result<StyleProfile, BlendError> {
    let blend = &recipe.blend;

    // sentence_length_distribution: weighted f64 average + Hare quota.
    let mut weighted_dist: [f64; 4] = [0.0; 4];
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        for i in 0..4 {
            weighted_dist[i] += entry.weight * profile.sentence_length_distribution[i] as f64;
        }
    }
    let sentence_length_distribution = distribute_largest_remainder(weighted_dist, 100);

    // thread_max_length: weighted f64 average → round → clamp.
    let mut weighted_thread_max: f64 = 0.0;
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        weighted_thread_max += entry.weight * profile.thread_max_length as f64;
    }
    let thread_max_length: u32 = (weighted_thread_max.round() as u32).clamp(1, 25);

    // opening_patterns + opening_pattern_weights (parallel arrays).
    let (opening_patterns, opening_pattern_weights) = merge_opening_patterns(blend, profiles)?;

    // Categorical votes — explicit declaration-order slices for each enum.
    let sentence_length_target = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.sentence_length_target,
        &[
            SentenceLengthTarget::Short,
            SentenceLengthTarget::Mixed,
            SentenceLengthTarget::Long,
        ],
    )?;
    let fragment_frequency = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.fragment_frequency,
        &[
            FragmentFrequency::Rare,
            FragmentFrequency::Occasional,
            FragmentFrequency::Common,
        ],
    )?;
    let emoji_policy = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.emoji_policy,
        &[
            EmojiPolicy::Never,
            EmojiPolicy::RarePunchlineOnly,
            EmojiPolicy::Occasional,
            EmojiPolicy::Frequent,
        ],
    )?;
    let hashtag_policy = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.hashtag_policy,
        &[
            HashtagPolicy::Never,
            HashtagPolicy::Rare,
            HashtagPolicy::TopicRelevant,
            HashtagPolicy::Always,
        ],
    )?;
    let specificity_target = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.specificity_target,
        &[
            SpecificityTarget::Low,
            SpecificityTarget::Medium,
            SpecificityTarget::High,
        ],
    )?;
    let thread_rhythm = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.thread_rhythm,
        &[
            ThreadRhythm::Linear,
            ThreadRhythm::ListThenPayoff,
            ThreadRhythm::PunchlineCallbacks,
        ],
    )?;

    // Formatting sub-struct: per-field independent votes.
    let formatting = Formatting {
        lowercase: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.lowercase,
            &[false, true], // tiebreak: false wins
        )?,
        periods: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.periods,
            &[
                PeriodsPolicy::Always,
                PeriodsPolicy::Optional,
                PeriodsPolicy::Rare,
            ],
        )?,
        em_dashes: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.em_dashes,
            &[
                EmDashPolicy::Preferred,
                EmDashPolicy::Ok,
                EmDashPolicy::Forbidden,
            ],
        )?,
        quotation_marks: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.quotation_marks,
            &[
                QuotationMarks::Double,
                QuotationMarks::Single,
                QuotationMarks::Smart,
            ],
        )?,
        line_breaks: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.line_breaks,
            &[
                LineBreaks::Single,
                LineBreaks::Double,
                LineBreaks::Rhythmic,
            ],
        )?,
    };

    let thread_opener_must_hook = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.thread_opener_must_hook,
        &[false, true], // tiebreak: false wins
    )?;

    // List-of-strings: union + dedup with stable insertion order.
    let voice_traits = union_dedup_strings(blend, profiles, |p| &p.voice_traits)?;
    let ai_tells_to_avoid = union_dedup_strings(blend, profiles, |p| &p.ai_tells_to_avoid)?;
    let topical_obsessions = union_dedup_strings(blend, profiles, |p| &p.topical_obsessions)?;
    let topical_avoidances = union_dedup_strings(blend, profiles, |p| &p.topical_avoidances)?;

    // Construct merged profile.
    let mut merged = StyleProfile {
        version: 1,
        sentence_length_target,
        sentence_length_distribution,
        fragment_frequency,
        opening_patterns,
        opening_pattern_weights,
        formatting,
        emoji_policy,
        hashtag_policy,
        specificity_target,
        voice_traits,
        ai_tells_to_avoid,
        thread_rhythm,
        thread_max_length,
        thread_opener_must_hook,
        topical_obsessions,
        topical_avoidances,
    };

    // Apply user overrides last.
    apply_overrides(&mut merged, &recipe.overrides);

    // Post-merge validation.
    merged
        .validate()
        .map_err(|inner| BlendError::PostMergeValidation { inner })?;

    Ok(merged)
}
```

### Step 5: Modify `voice/mod.rs` to re-export `blend_profiles`

The current `pub use blend::...` line (after Task 1) is:

```rust
pub use blend::{BlendEntry, BlendError, BlendRecipe, PartialStyleProfile};
```

Replace with:

```rust
pub use blend::{BlendEntry, BlendError, BlendRecipe, PartialStyleProfile, blend_profiles};
```

(rustfmt may sort case-insensitively. Let it.)

### Step 6: Append the test helpers + 21 integration tests

Inside the existing `#[cfg(test)] mod tests` block, append AFTER the 5 P1.2d tests already there (3 BlendError + 2 distribute_largest_remainder):

```rust
    use std::collections::HashMap;

    // ─────────────────────────────────────────────────────────────────────
    // P1.2d — test helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Returns a baseline valid StyleProfile (mutate fields per test).
    /// Schema matches the §2.2 spec example.
    fn mk_profile() -> StyleProfile {
        StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![
                OpeningPattern::ClaimFirst,
                OpeningPattern::NumberFirst,
                OpeningPattern::SceneFirst,
                OpeningPattern::QuestionFirst,
            ],
            opening_pattern_weights: vec![0.4, 0.2, 0.2, 0.2],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::RarePunchlineOnly,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec!["specific".to_string(), "no_hedging".to_string()],
            ai_tells_to_avoid: vec!["delve".to_string()],
            thread_rhythm: ThreadRhythm::PunchlineCallbacks,
            thread_max_length: 10,
            thread_opener_must_hook: true,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        }
    }

    fn mk_recipe(entries: Vec<(&str, f64)>, overrides: PartialStyleProfile) -> BlendRecipe {
        BlendRecipe {
            version: 1,
            blend: entries
                .into_iter()
                .map(|(w, weight)| BlendEntry {
                    writer: w.to_string(),
                    weight,
                })
                .collect(),
            overrides,
        }
    }

    fn mk_profiles(entries: Vec<(&str, StyleProfile)>) -> HashMap<String, StyleProfile> {
        entries
            .into_iter()
            .map(|(w, p)| (w.to_string(), p))
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────
    // P1.2d — blend_profiles integration tests
    // ─────────────────────────────────────────────────────────────────────

    // ---- Single-writer identity -----------------------------------------

    #[test]
    fn single_writer_recipe_returns_input_profile_modulo_overrides() {
        let p = mk_profile();
        let recipe = mk_recipe(vec![("k", 1.0)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("k", p.clone())]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out, p);
    }

    // ---- Numeric merging ------------------------------------------------

    #[test]
    fn sentence_length_distribution_weighted_average_simple() {
        let mut a = mk_profile();
        a.sentence_length_distribution = [40, 30, 20, 10];
        let mut b = mk_profile();
        b.sentence_length_distribution = [20, 30, 30, 20];
        // 0.5 × [40,30,20,10] + 0.5 × [20,30,30,20] = [30, 30, 25, 15]
        let recipe = mk_recipe(
            vec![("a", 0.5), ("b", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.sentence_length_distribution, [30, 30, 25, 15]);
    }

    #[test]
    fn sentence_length_distribution_hare_quota_residual_distribution() {
        // 0.5 × [33, 33, 33, 1] + 0.5 × [34, 33, 33, 0] = [33.5, 33.0, 33.0, 0.5]
        // Floors [33, 33, 33, 0] sum 99 → residual 1 → bucket 0 wins
        // (declaration-order tiebreak vs bucket 3, both have frac=0.5).
        let mut a = mk_profile();
        a.sentence_length_distribution = [33, 33, 33, 1];
        let mut b = mk_profile();
        b.sentence_length_distribution = [34, 33, 33, 0];
        let recipe = mk_recipe(
            vec![("a", 0.5), ("b", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.sentence_length_distribution, [34, 33, 33, 0]);
    }

    #[test]
    fn thread_max_length_weighted_average_rounded() {
        let mut a = mk_profile();
        a.thread_max_length = 10;
        let mut b = mk_profile();
        b.thread_max_length = 20;
        // 0.6 × 10 + 0.4 × 20 = 14
        let recipe = mk_recipe(
            vec![("a", 0.6), ("b", 0.4)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.thread_max_length, 14);
    }

    #[test]
    fn thread_max_length_clamps_to_valid_range() {
        // Two profiles at max 25 each; weighted avg is 25, but f64 rounding
        // could theoretically produce 25.0000001 → 26. Clamp catches that.
        let mut a = mk_profile();
        a.thread_max_length = 25;
        let mut b = mk_profile();
        b.thread_max_length = 25;
        let recipe = mk_recipe(
            vec![("a", 0.5), ("b", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.thread_max_length, 25);
    }

    // ---- Categorical voting ---------------------------------------------

    #[test]
    fn enum_vote_picks_highest_weighted_variant() {
        let mut a = mk_profile();
        a.emoji_policy = EmojiPolicy::Never;
        let mut b = mk_profile();
        b.emoji_policy = EmojiPolicy::Frequent;
        // a wins 0.7 vs b's 0.3 → Never.
        let recipe = mk_recipe(
            vec![("a", 0.7), ("b", 0.3)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.emoji_policy, EmojiPolicy::Never);
    }

    #[test]
    fn enum_vote_tiebreaks_by_declaration_order() {
        let mut a = mk_profile();
        a.emoji_policy = EmojiPolicy::Frequent;
        let mut b = mk_profile();
        b.emoji_policy = EmojiPolicy::Never;
        // 0.5/0.5 tie. Never declared first → wins.
        let recipe = mk_recipe(
            vec![("a", 0.5), ("b", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.emoji_policy, EmojiPolicy::Never);
    }

    #[test]
    fn bool_vote_tiebreaks_to_false() {
        let mut a = mk_profile();
        a.formatting.lowercase = true;
        let mut b = mk_profile();
        b.formatting.lowercase = false;
        // 0.5/0.5 tie. false declared first in &[false, true] → wins.
        let recipe = mk_recipe(
            vec![("a", 0.5), ("b", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert!(!out.formatting.lowercase);
    }

    // ---- Opening-pattern merge ------------------------------------------

    #[test]
    fn opening_patterns_merge_weighted_accumulator_normalizes_to_1() {
        let mut a = mk_profile();
        a.opening_patterns = vec![OpeningPattern::ClaimFirst, OpeningPattern::NumberFirst];
        a.opening_pattern_weights = vec![0.4, 0.6];
        let mut b = mk_profile();
        b.opening_patterns = vec![OpeningPattern::ClaimFirst, OpeningPattern::SceneFirst];
        b.opening_pattern_weights = vec![0.7, 0.3];
        // 0.6 × {claim_first: 0.4, number_first: 0.6}
        // 0.4 × {claim_first: 0.7, scene_first: 0.3}
        // = claim_first: 0.52, number_first: 0.36, scene_first: 0.12 (sum 1.0)
        let recipe = mk_recipe(
            vec![("a", 0.6), ("b", 0.4)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        let total: f64 = out.opening_pattern_weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "got total: {total}");
        // Sorted by weight DESC.
        assert_eq!(out.opening_patterns[0], OpeningPattern::ClaimFirst);
        assert!((out.opening_pattern_weights[0] - 0.52).abs() < 1e-9);
        assert_eq!(out.opening_patterns[1], OpeningPattern::NumberFirst);
        assert_eq!(out.opening_patterns[2], OpeningPattern::SceneFirst);
    }

    #[test]
    fn opening_patterns_output_sorted_by_weight_desc_with_declaration_tiebreak() {
        let mut a = mk_profile();
        a.opening_patterns = vec![OpeningPattern::SceneFirst, OpeningPattern::ClaimFirst];
        a.opening_pattern_weights = vec![0.5, 0.5];
        let mut b = mk_profile();
        b.opening_patterns = vec![OpeningPattern::ClaimFirst, OpeningPattern::SceneFirst];
        b.opening_pattern_weights = vec![0.5, 0.5];
        // Each pattern gets 0.5 × 0.5 + 0.5 × 0.5 = 0.5. Tied weights.
        // Declaration-order tiebreak: ClaimFirst (idx 0) before SceneFirst (idx 2).
        let recipe = mk_recipe(
            vec![("a", 0.5), ("b", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.opening_patterns, vec![OpeningPattern::ClaimFirst, OpeningPattern::SceneFirst]);
    }

    // ---- List union+dedup -----------------------------------------------

    #[test]
    fn list_of_strings_union_preserves_first_occurrence_order() {
        let mut a = mk_profile();
        a.voice_traits = vec!["alpha".to_string(), "bravo".to_string()];
        let mut b = mk_profile();
        b.voice_traits = vec!["charlie".to_string(), "alpha".to_string()];
        // Recipe order: a then b.
        // Expected: ["alpha", "bravo", "charlie"] — alpha from a wins position 0,
        // bravo from a is position 1, charlie from b is position 2 (alpha is dedup'd).
        let recipe = mk_recipe(
            vec![("a", 0.5), ("b", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(
            out.voice_traits,
            vec!["alpha".to_string(), "bravo".to_string(), "charlie".to_string()]
        );
    }

    #[test]
    fn list_of_strings_dedup_is_case_sensitive() {
        let mut a = mk_profile();
        a.voice_traits = vec!["specific".to_string()];
        let mut b = mk_profile();
        b.voice_traits = vec!["Specific".to_string()];
        let recipe = mk_recipe(
            vec![("a", 0.5), ("b", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        // Both kept: "specific" and "Specific" are distinct.
        assert_eq!(
            out.voice_traits,
            vec!["specific".to_string(), "Specific".to_string()]
        );
    }

    // ---- Override application -------------------------------------------

    #[test]
    fn overrides_replace_blended_value_unconditionally() {
        let p = mk_profile(); // thread_max_length = 10
        let recipe = mk_recipe(
            vec![("k", 1.0)],
            PartialStyleProfile {
                thread_max_length: Some(7),
                ..Default::default()
            },
        );
        let profiles = mk_profiles(vec![("k", p)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.thread_max_length, 7);
    }

    #[test]
    fn override_violation_surfaces_as_post_merge_validation() {
        let p = mk_profile();
        let recipe = mk_recipe(
            vec![("k", 1.0)],
            PartialStyleProfile {
                // 30 violates the 1..=25 invariant.
                thread_max_length: Some(30),
                ..Default::default()
            },
        );
        let profiles = mk_profiles(vec![("k", p)]);
        let err = blend_profiles(&recipe, &profiles).unwrap_err();
        match err {
            BlendError::PostMergeValidation { inner } => {
                let msg = format!("{inner}");
                assert!(msg.contains("thread_max_length"), "msg: {msg}");
                assert!(msg.contains("1..=25"), "msg: {msg}");
            }
            other => panic!("expected PostMergeValidation, got {other:?}"),
        }
    }

    #[test]
    fn partial_parallel_array_override_caught_by_validate() {
        let p = mk_profile();
        // Override patterns but not weights → length mismatch after apply.
        let recipe = mk_recipe(
            vec![("k", 1.0)],
            PartialStyleProfile {
                opening_patterns: Some(vec![
                    OpeningPattern::ClaimFirst,
                    OpeningPattern::NumberFirst,
                    OpeningPattern::SceneFirst,
                ]),
                ..Default::default()
            },
        );
        let profiles = mk_profiles(vec![("k", p)]);
        let err = blend_profiles(&recipe, &profiles).unwrap_err();
        match err {
            BlendError::PostMergeValidation { inner } => {
                let msg = format!("{inner}");
                assert!(msg.contains("opening_patterns"), "msg: {msg}");
                assert!(msg.contains("same length"), "msg: {msg}");
            }
            other => panic!("expected PostMergeValidation, got {other:?}"),
        }
    }

    // ---- Error paths ----------------------------------------------------

    #[test]
    fn missing_profile_for_writer_returns_missing_profile_error() {
        let p = mk_profile();
        // Recipe references "k" and "ghost" but profiles only has "k".
        let recipe = mk_recipe(
            vec![("k", 0.5), ("ghost", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("k", p)]);
        let err = blend_profiles(&recipe, &profiles).unwrap_err();
        match err {
            BlendError::MissingProfile(w) => assert_eq!(w, "ghost"),
            other => panic!("expected MissingProfile, got {other:?}"),
        }
    }

    #[test]
    fn recipe_with_writer_listed_twice_uses_first_lookup() {
        // BlendRecipe::validate would reject duplicate writers, but if a
        // hand-constructed unvalidated recipe is passed, document the
        // behavior: each weighted contribution sums independently.
        let mut p = mk_profile();
        p.thread_max_length = 10;
        // Manually construct unvalidated recipe with duplicate "k".
        let recipe = BlendRecipe {
            version: 1,
            blend: vec![
                BlendEntry { writer: "k".to_string(), weight: 0.4 },
                BlendEntry { writer: "k".to_string(), weight: 0.6 },
            ],
            overrides: PartialStyleProfile::default(),
        };
        let profiles = mk_profiles(vec![("k", p)]);
        // 0.4 × 10 + 0.6 × 10 = 10 → still 10 (k contributes twice).
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.thread_max_length, 10);
    }

    // ---- Determinism + happy path ---------------------------------------

    #[test]
    fn blend_is_deterministic_across_runs() {
        let p_a = {
            let mut x = mk_profile();
            x.thread_max_length = 12;
            x.emoji_policy = EmojiPolicy::Never;
            x
        };
        let p_b = {
            let mut x = mk_profile();
            x.thread_max_length = 18;
            x.emoji_policy = EmojiPolicy::Occasional;
            x
        };
        let recipe = mk_recipe(
            vec![("a", 0.6), ("b", 0.4)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", p_a), ("b", p_b)]);
        let baseline = blend_profiles(&recipe, &profiles).unwrap();
        for _ in 0..10 {
            let again = blend_profiles(&recipe, &profiles).unwrap();
            assert_eq!(again, baseline);
        }
    }

    #[test]
    fn blend_output_passes_style_profile_validate() {
        let p = mk_profile();
        let recipe = mk_recipe(vec![("k", 1.0)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("k", p)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        out.validate().expect("merged profile is valid");
    }

    // ---- Five-writer canonical example ----------------------------------

    #[test]
    fn five_writer_canonical_blend_matches_spec_example() {
        // The umbrella spec §2.3 AI/tech blend.
        let p_karpathy = {
            let mut x = mk_profile();
            x.thread_max_length = 12;
            x.voice_traits = vec!["technical".to_string(), "humble".to_string()];
            x
        };
        let p_eladgil = {
            let mut x = mk_profile();
            x.thread_max_length = 8;
            x.voice_traits = vec!["signal_dense".to_string()];
            x
        };
        let p_swyx = {
            let mut x = mk_profile();
            x.thread_max_length = 10;
            x.voice_traits = vec!["builder".to_string()];
            x
        };
        let p_naval = {
            let mut x = mk_profile();
            x.thread_max_length = 5;
            x.voice_traits = vec!["aphoristic".to_string()];
            x
        };
        let p_sama = {
            let mut x = mk_profile();
            x.thread_max_length = 6;
            x.voice_traits = vec!["sparse".to_string()];
            x
        };
        let recipe = mk_recipe(
            vec![
                ("karpathy", 0.30),
                ("eladgil", 0.20),
                ("swyx", 0.20),
                ("naval", 0.15),
                ("sama", 0.15),
            ],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![
            ("karpathy", p_karpathy),
            ("eladgil", p_eladgil),
            ("swyx", p_swyx),
            ("naval", p_naval),
            ("sama", p_sama),
        ]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        // Smoke: the result is a valid StyleProfile with all 5 writers'
        // voice_traits unioned (5 distinct strings).
        out.validate().expect("merged profile is valid");
        assert_eq!(out.voice_traits.len(), 5);
        assert!(out.voice_traits.contains(&"technical".to_string()));
        assert!(out.voice_traits.contains(&"sparse".to_string()));
        // 0.30×12 + 0.20×8 + 0.20×10 + 0.15×5 + 0.15×6 = 3.6 + 1.6 + 2.0 + 0.75 + 0.9 = 8.85 → 9
        assert_eq!(out.thread_max_length, 9);
    }

    // ---- Floating-point drift edge case ---------------------------------

    #[test]
    fn opening_pattern_weights_normalized_under_f64_drift() {
        // Construct profiles where the cross-product produces weights that
        // sum to a value off by >1e-9 due to f64 multiplication drift.
        // We use 3 profiles with carefully chosen weights to amplify drift.
        let mut a = mk_profile();
        a.opening_patterns = vec![OpeningPattern::ClaimFirst];
        a.opening_pattern_weights = vec![1.0];
        let mut b = mk_profile();
        b.opening_patterns = vec![OpeningPattern::NumberFirst];
        b.opening_pattern_weights = vec![1.0];
        let mut c = mk_profile();
        c.opening_patterns = vec![OpeningPattern::SceneFirst];
        c.opening_pattern_weights = vec![1.0];
        // Weights chosen to hit f64 representation edges: 0.1 + 0.2 + 0.7
        // = 0.9999999999999999 in f64.
        let recipe = mk_recipe(
            vec![("a", 0.1), ("b", 0.2), ("c", 0.7)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b), ("c", c)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        // After defensive normalization, sum is within 1e-6 → validate passes.
        out.validate()
            .expect("merged profile passes validate after f64 normalization");
        let total: f64 = out.opening_pattern_weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "weights sum should normalize to 1.0, got {total}"
        );
    }
```

### Step 7: Run the tests

```bash
cargo test -p heartbit-ghost --lib voice::blend
```

Expected: `38 passed` (17 from prior tasks + 21 new in Task 3). Test count breakdown: 12 P1.2a + 3 BlendError display (Task 1) + 2 distribute_largest_remainder (Task 2) + 21 integration (Task 3) = 38.

### Step 8: Workspace quality gate

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

**Both must be clean.** Specifically: clippy must NOT flag `distribute_largest_remainder`, `weighted_vote_categorical`, `merge_opening_patterns`, `union_dedup_strings`, `apply_overrides`, or `opening_pattern_decl_index` as `dead_code`. If any are flagged, `blend_profiles` is not actually using them.

### Step 9: Commit

```bash
git add crates/heartbit-ghost/src/voice/blend.rs crates/heartbit-ghost/src/voice/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — blend_profiles + remaining helpers (P1.2d)

The orchestration layer. blend_profiles applies a BlendRecipe to a map
of writer-handle → StyleProfile and produces a merged, validated
StyleProfile per umbrella spec §2.3:
- 16 fields merged via per-rule helpers (Hare quota, weighted vote,
  weighted accumulator, union+dedup)
- recipe.overrides applied last (PartialStyleProfile fields with
  Some(_) replace blended values)
- StyleProfile::validate() runs on the result; failure surfaces as
  BlendError::PostMergeValidation

5 private helpers: weighted_vote_categorical (generic over E: Copy + Eq;
also handles bool with &[false, true]), opening_pattern_decl_index
(declaration-order index for sort tiebreak), merge_opening_patterns
(weighted accumulator + defensive normalization), union_dedup_strings
(O(N×M) stable insertion order), apply_overrides (16-field replace).

Removes the temporary #[allow(dead_code)] from Task 2 now that
distribute_largest_remainder has a non-test caller.

21 new integration tests covering: single-writer identity, numeric
merging (4), categorical voting (3), opening-pattern merge (2), list
union+dedup (2), override application (3), error paths (2), determinism
+ happy path (2), 5-writer canonical example, f64 drift edge case.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2d-blend-algorithm-design.md §3 §4 §5
EOF
)"
```

---

## Task 4: Final acceptance + workspace quality gate

**Why:** Confirm P1.2d meets every acceptance criterion in the spec. Verification only — no commit.

**Files:** none.

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -10
```

Expected: all green. Workspace test count goes from 3873 (post-P1.2c baseline) to ~3899 (~26 new blend tests: 3 BlendError display + 2 distribute_largest_remainder + 21 integration).

- [ ] **Step 2: Verify the public surface is reachable**

```bash
cat <<'EOF' > /tmp/heartbit_ghost_blend_surface_check.rs
fn _check() {
    use heartbit_ghost::voice::{
        BlendEntry, BlendError, BlendRecipe, PartialStyleProfile, blend_profiles,
    };
    use std::collections::HashMap;
    let _: fn(&BlendRecipe, &HashMap<String, _>) -> Result<_, BlendError> = blend_profiles;
    let _ = BlendError::MissingProfile(String::new());
    let _ = BlendEntry { writer: String::new(), weight: 0.0 };
    let _ = PartialStyleProfile::default();
    let _ = BlendRecipe { version: 1, blend: vec![], overrides: PartialStyleProfile::default() };
}
EOF
echo "(Surface check is illustrative; the public types are reachable via the workspace cargo check above.)"
rm -f /tmp/heartbit_ghost_blend_surface_check.rs
```

- [ ] **Step 3: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.2d
```

Expected: 4 commits — spec doc + 3 task commits (Task 1, 2, 3). No commit for Task 4.

- [ ] **Step 4: No commit for this task**

Task 4 is verification only. The branch is ready for final review + merge.

---

## Acceptance criteria

P1.2d is done when (per spec §9):

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~21 blend-algorithm integration tests pass (actual count once all three implementation tasks land: 26 = 3 BlendError + 2 distribute_largest_remainder + 21 integration; the spec's "~21" was the integration-test count; the per-task breakdown adds 5 supporting tests)
- `heartbit_ghost::voice::{blend_profiles, BlendError}` are reachable as public surface
- The five-writer canonical example test (§2.3 AI/tech blend) produces a valid merged `StyleProfile`

## Out of scope (re-stated)

- Persistence to disk / `personas/x.toml` snapshot (P1.2e)
- Profile diff format (P1.2e)
- Recipe-level validation (already in `BlendRecipe::validate`)
- Profile-level validation on input (assumed valid; trust the caller)
- Versioned snapshot / hash / commit-on-change (P1.2e)
- Runtime conditioning of the writer agent (P1.4)
- Parallel processing (single-threaded blender; if needed, P1.4 wraps in `tokio::spawn_blocking`)
- LLM-based corpus extraction (P1.2c — already merged)

## Reference

- Spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2d-blend-algorithm-design.md`
- Umbrella heartbit-ghost spec §2.3: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- P1.2a (BlendRecipe + PartialStyleProfile + StyleProfile): `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`
- P1.2b (corpus storage): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md`
- P1.2c (LLM extractor): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md`
- Existing `voice/blend.rs` (P1.2a code): `crates/heartbit-ghost/src/voice/blend.rs`
