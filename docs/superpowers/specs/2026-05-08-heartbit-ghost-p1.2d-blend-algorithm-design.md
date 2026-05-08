# heartbit-ghost P1.2d — blend algorithm design

**Status:** approved 2026-05-08
**Branch:** `feat/heartbit-ghost-p1.2d`
**Predecessors:** P1.2a (style profile schema), P1.2b (corpus storage), P1.2c (LLM style extractor). All merged to `main`.
**Successors:** P1.2e (CLI bodies), P1.4 (runtime conditioning).

## 1. Goal

Apply a `BlendRecipe` (P1.2a) to a map of writer-handle → `StyleProfile` and produce one merged, validated `StyleProfile`. Pure deterministic data transformation — no LLM, no async, no I/O.

The library surface needs to be enough that:

- P1.2e's `heartbit persona profile rebuild` CLI body can call `blend_profiles(&recipe, &profiles_map)` after running the LLM extractor over each writer's corpus
- P1.4's runtime conditioning can blend per-tenant overrides into a base profile

Out of scope for this phase: persistence (no disk write — caller serializes the result via `toml::to_string` if needed), recipe validation (already in `BlendRecipe::validate`), profile validation on input (assumed valid; trust the caller), versioned snapshot / hash / `personas/x.toml` produce-and-commit, profile diff format.

## 2. Architecture

Extends the existing `crates/heartbit-ghost/src/voice/blend.rs` (P1.2a). Single file holds all blend-related code: data types (BlendRecipe / BlendEntry / PartialStyleProfile), the algorithm (`blend_profiles`), the error type (`BlendError`), and tests. Cohesion argument wins over file-size argument; comparable to `voice/style.rs` after P1.2a.

**No new dependencies** — pure stdlib (`HashMap`), no async, no third-party crates beyond what `voice/blend.rs` already pulls in.

**Re-exports added to `voice/mod.rs`**: `blend_profiles`, `BlendError`.

## 3. Public API

```rust
// in heartbit-ghost::voice::blend

pub fn blend_profiles(
    recipe: &BlendRecipe,
    profiles: &HashMap<String, StyleProfile>,
) -> Result<StyleProfile, BlendError>;

#[derive(Debug, thiserror::Error)]
pub enum BlendError {
    /// The recipe references a writer handle that is not in the profiles map.
    #[error("missing profile for writer '{0}'")]
    MissingProfile(String),

    /// Result of merge + override application failed StyleProfile::validate.
    #[error("merged profile failed validation: {inner}")]
    PostMergeValidation {
        #[source]
        inner: VoiceError,
    },
}
```

**Design decisions:**

- **Free function, not method on `BlendRecipe`** — both inputs are equally important; matches the established free-function pattern (`default_corpora_dir`, `list_writers`, `default_system_prompt`); avoids method-name conflict with `BlendRecipe::blend` field.
- **`recipe.validate()` is NOT called by `blend_profiles`** — the spec assumes a validated recipe (caller responsibility). Avoids redundant work when the caller has already validated (typical case: immediately after `BlendRecipe::from_toml`, which validates internally). If the caller passes garbage, the math runs but the output likely fails post-merge validation.
- **`BlendError` is separate from `VoiceError` and `ExtractError`** — distinguishes "you forgot a writer" from "schema is broken" from "LLM produced bad JSON". `MissingProfile` carries the writer handle for debuggability.
- **No `InvalidOverride` variant** — partial parallel-array overrides surface as `PostMergeValidation` containing the length-mismatch message from `StyleProfile::validate`. Single source of truth for structural invariants.

## 4. Field-by-field merge rules

The 16 non-`version` fields of `StyleProfile`, with the merge rule applied to each:

| Field | Type | Merge rule |
|-------|------|-----------|
| `sentence_length_target` | `SentenceLengthTarget` (enum) | Weighted vote, declaration-order tiebreak (`Short` → `Mixed` → `Long`) |
| `sentence_length_distribution` | `[u8; 4]` | Weighted f64 average + Hare-quota rounding to keep sum=100 |
| `fragment_frequency` | `FragmentFrequency` (enum) | Weighted vote, declaration-order tiebreak (`Rare` → `Occasional` → `Common`) |
| `opening_patterns` + `opening_pattern_weights` | parallel `Vec<...>` | Weighted accumulator per pattern, normalize to sum 1.0, sort by weight DESC with declaration-order tiebreak |
| `formatting.lowercase` | `bool` | Weighted vote, tiebreak: `false` (more conservative — sentence case is the X norm) |
| `formatting.periods` | `PeriodsPolicy` (enum) | Weighted vote, declaration-order tiebreak (`Always` → `Optional` → `Rare`) |
| `formatting.em_dashes` | `EmDashPolicy` (enum) | Weighted vote, declaration-order tiebreak (`Preferred` → `Ok` → `Forbidden`) |
| `formatting.quotation_marks` | `QuotationMarks` (enum) | Weighted vote, declaration-order tiebreak (`Double` → `Single` → `Smart`) |
| `formatting.line_breaks` | `LineBreaks` (enum) | Weighted vote, declaration-order tiebreak (`Single` → `Double` → `Rhythmic`) |
| `emoji_policy` | `EmojiPolicy` (enum) | Weighted vote, declaration-order tiebreak |
| `hashtag_policy` | `HashtagPolicy` (enum) | Weighted vote, declaration-order tiebreak |
| `specificity_target` | `SpecificityTarget` (enum) | Weighted vote, declaration-order tiebreak (`Low` → `Medium` → `High`) |
| `voice_traits` | `Vec<String>` | Union + dedup, stable insertion order, case-sensitive |
| `ai_tells_to_avoid` | `Vec<String>` | Union + dedup, stable insertion order, case-sensitive |
| `thread_rhythm` | `ThreadRhythm` (enum) | Weighted vote, declaration-order tiebreak |
| `thread_max_length` | `u32` | Weighted f64 average → `.round()` → `.clamp(1, 25)` |
| `thread_opener_must_hook` | `bool` | Weighted vote, tiebreak: `false` (more lenient — "no need to earn the read") |
| `topical_obsessions` | `Vec<String>` | Union + dedup, stable insertion order, case-sensitive |
| `topical_avoidances` | `Vec<String>` | Union + dedup, stable insertion order, case-sensitive |

**`Formatting` sub-struct merging is per-field, not whole-struct.** The 5 `formatting.*` rows above each get an independent vote; the merger constructs a fresh `Formatting { lowercase: vote(...), periods: vote(...), em_dashes: vote(...), quotation_marks: vote(...), line_breaks: vote(...) }`.

**Apply order** (from umbrella spec §2.3 step 3):

1. Start with empty merged profile fields
2. For every field in the table above, run its merge rule across all profiles weighted by `recipe.blend[i].weight`
3. Apply `recipe.overrides`: for each field where `recipe.overrides.<field>` is `Some(value)`, replace the merged value with `value`
4. Set `version = 1`
5. Run `merged.validate()` — return `Ok(merged)` or `Err(BlendError::PostMergeValidation)`

## 5. Algorithm internals

### 5.1 Hare quota (largest-remainder) for `[u8; 4]`

```rust
/// Distribute `total` units across N buckets according to f64 weights,
/// producing N integer values that sum to exactly `total`.
fn distribute_largest_remainder(weighted: [f64; 4], total: u32) -> [u8; 4] {
    // 1. floor each weighted value, accumulate `assigned`
    let floors: [u32; 4] = weighted.map(|x| x as u32);
    let assigned: u32 = floors.iter().sum();
    let residual = total.saturating_sub(assigned);
    // 2. fractional parts paired with original index
    let mut fracs: [(usize, f64); 4] = [(0, 0.0); 4];
    for i in 0..4 {
        fracs[i] = (i, weighted[i] - floors[i] as f64);
    }
    // 3. sort by (frac DESC, declaration index ASC)
    fracs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    // 4. add 1 to the first `residual` buckets
    let mut out = [0u8; 4];
    for i in 0..4 {
        out[i] = floors[i].min(255) as u8;
    }
    for r in 0..(residual as usize).min(4) {
        let idx = fracs[r].0;
        out[idx] = out[idx].saturating_add(1);
    }
    out
}
```

Worked example: 0.5 × `[33, 33, 33, 1]` + 0.5 × `[34, 33, 33, 0]` → weighted `[33.5, 33.0, 33.0, 0.5]`. Floors `[33, 33, 33, 0]` sum 99 → residual 1 → goes to bucket 0 (frac=0.5; bucket 3 also has frac=0.5 but loses by declaration-order tiebreak). Result: `[34, 33, 33, 0]`.

### 5.2 Weighted vote across enum variants

```rust
/// Vote across enum variants in `order` (declaration order). Returns the
/// variant with the highest summed weight; ties resolved by `order`.
fn weighted_vote_categorical<E: Copy + PartialEq>(
    blend: &[BlendEntry],
    profiles: &HashMap<String, StyleProfile>,
    pick: impl Fn(&StyleProfile) -> E,
    order: &[E],
) -> Result<E, BlendError> {
    // accumulate weights per variant (parallel Vec keyed by order position)
    let mut weights = vec![0.0f64; order.len()];
    for entry in blend {
        let profile = profiles.get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        let chosen = pick(profile);
        if let Some(idx) = order.iter().position(|v| *v == chosen) {
            weights[idx] += entry.weight;
        }
    }
    // seed with order[0], then strict-> for declaration-order tiebreak
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
```

The `order` argument is the explicit declaration sequence: `&[Short, Mixed, Long]`, `&[Always, Optional, Rare]`, etc. For each enum, the caller passes the declaration-order slice.

**Why no `HashMap<E, f64>` accumulator**: closed-vocab enums in P1.2a don't derive `Hash` (`#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]`). Linear-scan over `&[E]` is trivial at 7-variant max and avoids backfilling derives.

### 5.3 Opening-pattern merge

```rust
/// Merge per-profile (opening_patterns, opening_pattern_weights) parallel
/// arrays into a single normalized parallel pair.
fn merge_opening_patterns(
    blend: &[BlendEntry],
    profiles: &HashMap<String, StyleProfile>,
) -> Result<(Vec<OpeningPattern>, Vec<f64>), BlendError> {
    let mut acc: Vec<(OpeningPattern, f64)> = Vec::new();
    for entry in blend {
        let profile = profiles.get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        for (pat, w) in profile.opening_patterns.iter().zip(profile.opening_pattern_weights.iter()) {
            match acc.iter_mut().find(|(p, _)| p == pat) {
                Some(slot) => slot.1 += entry.weight * w,
                None => acc.push((*pat, entry.weight * w)),
            }
        }
    }
    // normalize defensively against f64 drift
    let total: f64 = acc.iter().map(|(_, w)| *w).sum();
    if total > 0.0 && (total - 1.0).abs() > 1e-9 {
        for (_, w) in acc.iter_mut() {
            *w /= total;
        }
    }
    // sort by (weight DESC, declaration-order ASC)
    acc.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            .then(opening_pattern_decl_index(&a.0).cmp(&opening_pattern_decl_index(&b.0)))
    });
    Ok(acc.into_iter().unzip())
}
```

`opening_pattern_decl_index(p)` returns 0..7 according to declaration order in `OpeningPattern`. Hardcoded match (7 variants — no premature abstraction).

Worked example: karpathy(0.6) `claim_first=0.4, number_first=0.6`; naval(0.4) `claim_first=0.7, scene_first=0.3`. Accumulator: `claim_first = 0.6×0.4 + 0.4×0.7 = 0.52`, `number_first = 0.36`, `scene_first = 0.12`. Sum = 1.0 (already normalized). Sort by weight DESC: `[ClaimFirst, NumberFirst, SceneFirst]` with weights `[0.52, 0.36, 0.12]`.

### 5.4 List union+dedup with stable order

```rust
/// Union-then-dedup over `recipe.blend` iteration order.
/// First occurrence wins position. Case-sensitive comparison.
fn union_dedup_strings(
    blend: &[BlendEntry],
    profiles: &HashMap<String, StyleProfile>,
    pick: impl Fn(&StyleProfile) -> &[String],
) -> Result<Vec<String>, BlendError> {
    let mut out: Vec<String> = Vec::new();
    for entry in blend {
        let profile = profiles.get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        for s in pick(profile) {
            if !out.iter().any(|existing| existing == s) {
                out.push(s.clone());
            }
        }
    }
    Ok(out)
}
```

Single pass, O(N×M) where N=writers and M=avg list length per profile. At 5 writers × ~10 strings per list, contention is negligible.

### 5.5 `thread_max_length`

```rust
let weighted: f64 = recipe.blend.iter()
    .map(|e| {
        let profile = profiles.get(&e.writer).ok_or(...)?;
        Ok(e.weight * profile.thread_max_length as f64)
    })
    .sum::<Result<f64, _>>()?;
let merged: u32 = (weighted.round() as u32).clamp(1, 25);
```

`.clamp(1, 25)` is defensive against f64 precision drift (e.g., `25.0000001` rounding to 26). The mathematical result of weighted-averaging values in `1..=25` is in `1..=25`, but `.clamp` makes the precision guarantee explicit.

## 6. Edge cases

**Empty profiles map**: `BlendRecipe::validate` already requires `1..=10` blend entries, so the recipe always references at least one writer. If `profiles` is empty, the first per-field iteration's `profiles.get(...)` returns `None` → `BlendError::MissingProfile(first_writer)`.

**Recipe with one writer**: merge collapses to identity (every weighted average is `1.0 × single_value`, every vote returns the single profile's choice). Output equals input modulo Hare-quota rounding (which has zero residual since the single-input weighted distribution already sums to 100). Useful sanity test.

**`recipe.overrides` violates schema invariants**: blender applies the override unconditionally; `merged.validate()` catches it; user gets `BlendError::PostMergeValidation` containing the schema validation message.

**Partial parallel-array overrides** (e.g., `overrides.opening_patterns = Some(...)` without `opening_pattern_weights`): blender applies the partial override → length mismatch → `validate()` returns "must have the same length" → `PostMergeValidation`. No special-case logic needed.

**Floating-point drift on `opening_pattern_weights`**: defensively normalized when `(sum - 1.0).abs() > 1e-9`. `validate()`'s `1e-6` tolerance is the safety net.

**Determinism**: every step is deterministic. `HashMap` is only used for input lookup, not iteration order. All iteration is over `recipe.blend` (a `Vec`).

## 7. Testing

~21 tests, all in-tree (`#[cfg(test)] mod tests` in `voice/blend.rs`, appended to the existing P1.2a tests block). Pure unit tests; no async, no I/O.

**Single-writer identity (1 test)**: `single_writer_recipe_returns_input_profile_modulo_overrides`

**Numeric merging (4 tests)**: `sentence_length_distribution_weighted_average_simple`, `sentence_length_distribution_hare_quota_residual_distribution`, `thread_max_length_weighted_average_rounded`, `thread_max_length_clamps_to_valid_range`

**Categorical voting (3 tests)**: `enum_vote_picks_highest_weighted_variant`, `enum_vote_tiebreaks_by_declaration_order`, `bool_vote_tiebreaks_to_false`

**Opening-pattern merge (2 tests)**: `opening_patterns_merge_weighted_accumulator_normalizes_to_1`, `opening_patterns_output_sorted_by_weight_desc_with_declaration_tiebreak`

**List union+dedup (2 tests)**: `list_of_strings_union_preserves_first_occurrence_order`, `list_of_strings_dedup_is_case_sensitive`

**Override application (3 tests)**: `overrides_replace_blended_value_unconditionally`, `override_violation_surfaces_as_post_merge_validation`, `partial_parallel_array_override_caught_by_validate`

**Error paths (2 tests)**: `missing_profile_for_writer_returns_missing_profile_error`, `recipe_with_writer_listed_twice_uses_first_lookup` (documents behavior under unvalidated recipe)

**Determinism + happy path (2 tests)**: `blend_is_deterministic_across_runs` (10× same inputs → all outputs equal), `blend_output_passes_style_profile_validate`

**Five-writer canonical example (1 test)**: `five_writer_canonical_blend_matches_spec_example` — uses umbrella §2.3 AI/tech blend (`karpathy 0.30, eladgil 0.20, swyx 0.20, naval 0.15, sama 0.15`)

**Floating-point drift edge case (1 test)**: `opening_pattern_weights_normalized_under_f64_drift`

**Test helpers** (~30 LOC inside `mod tests`): `mk_profile()`, `mk_recipe(entries, overrides)`, `mk_profiles(entries)` — fluent builders for valid `StyleProfile` / `BlendRecipe` / `HashMap`.

**Quality gate** (mirrors prior phases):

```bash
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
```

Workspace test count: 3873 → ~3894.

## 8. Architecture decisions (ADs)

**AD-1 — Free function `blend_profiles`, not a method on `BlendRecipe`.** Both inputs are equally important; matches the established free-function pattern (`default_corpora_dir`, `list_writers`, `default_system_prompt`); avoids method-name conflict with `BlendRecipe::blend` field; better discoverability via the module surface (`heartbit_ghost::voice::blend_profiles`).

**AD-2 — Enum declaration order for tiebreak.** Closed-vocab enums in P1.2a are declared in semantic order (e.g., `EmojiPolicy::Never < RarePunchlineOnly < Occasional < Frequent`) — earlier variants tend to be the more conservative choice. Picking the first declared variant on a tied vote means: when uncertain, lean conservative. Stable across schema changes that don't reorder variants.

**AD-3 — Hare quota (largest-remainder) for `[u8; 4]` rounding.** Textbook integer-allocation algorithm. Deterministic, fair across buckets, no bias toward any one bucket. ~15 LOC. Alternative (naive round-and-fix-up) biases all rounding error onto the largest bucket and produces measurably skewed distributions on adversarial inputs.

**AD-4 — `Vec<(K, V)>` linear scan, not `HashMap<K, V>`, for the opening-pattern accumulator.** P1.2a's `OpeningPattern` enum (and the other 10 closed-vocab enums) doesn't derive `Hash`. Adding `Hash` to all 11 enums is a P1.2a backfit — small but spreads the change. Linear scan over a 7-variant max enum is trivially fast. Keeps the change scoped to `voice/blend.rs`.

**AD-5 — Boolean tiebreak to `false`.** For both bools (`lowercase`, `thread_opener_must_hook`), `false` is the more conservative default — sentence case is the X norm; "no need to earn the read" is the more lenient thread policy. Picking the false-leaning option on tie is a deliberate "when uncertain, don't impose an unusual choice" heuristic.

**AD-6 — `recipe.validate()` is NOT called inside `blend_profiles`.** Avoids redundant work in the common case (caller has already validated, e.g., immediately after `BlendRecipe::from_toml`). Documented as "assumed validated" in the function doc-comment. If unvalidated input is passed, the math runs but the result likely fails post-merge validation.

**AD-7 — No `InvalidOverride` variant on `BlendError`.** Partial parallel-array overrides surface as `PostMergeValidation` containing the length-mismatch message from `StyleProfile::validate`. Single source of truth for structural invariants; no special-case logic in the blender.

## 9. Acceptance criteria

P1.2d is done when:

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~21 blend-algorithm tests pass; coverage spans every merge rule in §4 (numeric, categorical, list, parallel-array, override application), every error variant, and the determinism guarantee
- `heartbit_ghost::voice::{blend_profiles, BlendError}` are reachable as public surface
- The five-writer canonical example test (§2.3 AI/tech blend) produces a valid merged `StyleProfile`

## 10. Out of scope (re-stated)

- Persistence to disk / `personas/x.toml` snapshot (P1.2e CLI body)
- Profile diff format (P1.2e — `heartbit persona profile diff x v3 v4`)
- Recipe-level validation (already in `BlendRecipe::validate`)
- Profile-level validation on input (assumed valid; trust the caller)
- Versioned snapshot / hash / commit-on-change (P1.2e)
- Runtime conditioning of the writer agent (P1.4)
- LLM-based corpus extraction (P1.2c — already merged)
- Parallel processing (the algorithm is fast enough single-threaded; if needed, P1.4 can wrap calls in `tokio::spawn_blocking`)

## 11. Reference

- Umbrella heartbit-ghost spec §2.3 (blend computation): `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- P1.2a (style profile schema + BlendRecipe + PartialStyleProfile): `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`
- P1.2b (corpus storage): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2b-corpus-storage-design.md`
- P1.2c (LLM style extractor): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md`
- Existing `voice/blend.rs` (P1.2a code): `crates/heartbit-ghost/src/voice/blend.rs`
