# `heartbit-ghost` P1.2a — Style Profile Schema Design

**Date:** 2026-05-07
**Status:** design (awaiting review)
**Depends on:** P1.0 (heartbit-ghost crate scaffolding) + P1.1 (X tool family) — both merged to `main`.
**Implements:** [`2026-05-07-heartbit-ghost-x-agent-design.md`](2026-05-07-heartbit-ghost-x-agent-design.md) §2.2 (style profile schema) — the **first** of five P1.2 sub-phases.

## Mission

Ship the structured-data foundation of voice modeling: concrete Rust types (`StyleProfile` + `BlendRecipe`) with TOML serde and parse-time validation. The schema is the load-bearing data structure for the entire voice subsystem — writer agents will be prompted from it, the style-critic will evaluate against it, and the blender (P1.2d) will operate on it.

P1.2a ships **types only**. No LLM, no corpus storage, no blender, no CLI bodies. Each of those is a follow-up sub-phase on its own branch:

| Sub-phase | Scope |
|---|---|
| **P1.2a (this)** | Style profile schema |
| P1.2b | Reference corpus storage (memory namespace, JSONL load) |
| P1.2c | Style extractor sub-agent (LLM reads corpus → emits profile) |
| P1.2d | Blend algorithm (merge weighted profiles) |
| P1.2e | CLI bodies (`corpus add/list/remove`, `profile rebuild/diff`) |

## Architecture decisions

### AD-1 · Crate placement: `heartbit-ghost::voice`

New submodule at `crates/heartbit-ghost/src/voice/`. Promotes to `heartbit-core` only when a second persona crate (heartbit-coder, heartbit-researcher, etc.) needs voice modeling.

**Rationale.** P1.2c (extractor — needs LLM) and P1.2d (blender) live in the same crate; keeping the schema next to its consumers is cleanest. A separate `heartbit-ghost-voice` crate would isolate dependencies but pays for crate boundaries on a feature only heartbit-ghost uses today.

### AD-2 · Scope: `StyleProfile` + `BlendRecipe` ship together

Both pure-data types belong in P1.2a. `StyleProfile` describes a single voice (or the *output* of blending). `BlendRecipe` describes how to combine N profiles (entries: `{writer, weight}` + optional `overrides: PartialStyleProfile`). P1.2d (blender) takes a `BlendRecipe` + N `StyleProfile`s → produces a single `StyleProfile`.

**Rationale.** Including both now makes the "how voice is described in TOML" story complete in P1.2a. Splitting risks designing `BlendRecipe.overrides` in isolation from the `StyleProfile` shape it must mirror.

### AD-3 · Field typing: strong enums for closed vocabulary; `Vec<String>` for free-form

Closed-vocabulary fields become enums with `#[serde(rename_all = "snake_case")]` and `#[non_exhaustive]`. Genuinely free-form fields stay `Vec<String>`. Concrete table in §1.

**Rationale.** Parse-time errors catch typos like `sentence_length_target = "shortish"` immediately. `#[non_exhaustive]` allows future variant additions without breaking external consumers' pattern matches. `Vec<String>` for genuinely open fields like `voice_traits` and `ai_tells_to_avoid` (the writer prompt vocabulary is open-ended; enumerating it would be high-maintenance and lossy).

### AD-4 · Versioning: `version: u32` + `#[serde(default)]`

Explicit `version: u32` field on every top-level type, default 1. `#[serde(default)]` on every optional field for forward-compat. Breaking deserialization on unsupported versions returns `VoiceError::UnsupportedVersion(N)`.

**Rationale.** A `version` field lets us catch incompatible profiles loudly when we make breaking changes later. The `serde(default)` story handles non-breaking additions transparently. We don't need a migration system in P1.2a — the version field reserves the option without paying for unused machinery.

## 1. Closed-vocabulary enums

All declared with `#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)] + #[serde(rename_all = "snake_case")] + #[non_exhaustive]`. Each lives in `style.rs` (or its own file if it grows logic).

| Enum | Variants | TOML tokens |
|---|---|---|
| `SentenceLengthTarget` | `Short`, `Mixed`, `Long` | `"short"`, `"mixed"`, `"long"` |
| `FragmentFrequency` | `Rare`, `Occasional`, `Common` | `"rare"`, `"occasional"`, `"common"` |
| `OpeningPattern` | `ClaimFirst`, `NumberFirst`, `SceneFirst`, `QuestionFirst`, `AphoristicFirst`, `AnecdoteFirst`, `ContrarianFirst` | `"claim_first"`, `"number_first"`, `"scene_first"`, `"question_first"`, `"aphoristic_first"`, `"anecdote_first"`, `"contrarian_first"` |
| `PeriodsPolicy` | `Always`, `Optional`, `Rare` | `"always"`, `"optional"`, `"rare"` |
| `EmDashPolicy` | `Preferred`, `Ok`, `Forbidden` | `"preferred"`, `"ok"`, `"forbidden"` |
| `QuotationMarks` | `Double`, `Single`, `Smart` | `"double"`, `"single"`, `"smart"` |
| `LineBreaks` | `Single`, `Double`, `Rhythmic` | `"single"`, `"double"`, `"rhythmic"` |
| `EmojiPolicy` | `Never`, `RarePunchlineOnly`, `Occasional`, `Frequent` | `"never"`, `"rare_punchline_only"`, `"occasional"`, `"frequent"` |
| `HashtagPolicy` | `Never`, `Rare`, `TopicRelevant`, `Always` | `"never"`, `"rare"`, `"topic_relevant"`, `"always"` |
| `SpecificityTarget` | `Low`, `Medium`, `High` | `"low"`, `"medium"`, `"high"` |
| `ThreadRhythm` | `Linear`, `ListThenPayoff`, `PunchlineCallbacks` | `"linear"`, `"list_then_payoff"`, `"punchline_callbacks"` |

Each enum has a sensible `Default` impl so `PartialStyleProfile`'s `Option<...>` fields can use `#[serde(default)]`.

## 2. `Formatting` sub-struct

```rust
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Formatting {
    pub lowercase: bool,
    pub periods: PeriodsPolicy,
    pub em_dashes: EmDashPolicy,
    pub quotation_marks: QuotationMarks,
    pub line_breaks: LineBreaks,
}

impl Default for Formatting {
    fn default() -> Self {
        Self {
            lowercase: false,
            periods: PeriodsPolicy::Always,
            em_dashes: EmDashPolicy::Ok,
            quotation_marks: QuotationMarks::Double,
            line_breaks: LineBreaks::Single,
        }
    }
}
```

Nested under `StyleProfile.formatting` to match the TOML's `formatting.lowercase = ...` syntax.

## 3. `StyleProfile`

The structured fingerprint of a single voice. Parses the §2.2 TOML example directly.

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StyleProfile {
    #[serde(default = "default_version")]
    pub version: u32,

    // Sentence-level
    pub sentence_length_target: SentenceLengthTarget,
    pub sentence_length_distribution: [u8; 4],   // [< 10, 10-20, 20-40, > 40] words
    pub fragment_frequency: FragmentFrequency,

    // Opening patterns (parallel arrays — invariant: same length, weights sum to 1.0)
    pub opening_patterns: Vec<OpeningPattern>,
    pub opening_pattern_weights: Vec<f64>,

    // Formatting
    pub formatting: Formatting,

    // Vocabulary
    pub emoji_policy: EmojiPolicy,
    pub hashtag_policy: HashtagPolicy,
    pub specificity_target: SpecificityTarget,

    // Free-form descriptors (consumed by writer/critic LLM prompts)
    pub voice_traits: Vec<String>,
    pub ai_tells_to_avoid: Vec<String>,

    // Thread structure
    pub thread_rhythm: ThreadRhythm,
    pub thread_max_length: u32,                   // 1..=25 (matches TwitterThreadTool::MAX_THREAD_LENGTH)
    pub thread_opener_must_hook: bool,

    // Topical
    pub topical_obsessions: Vec<String>,
    pub topical_avoidances: Vec<String>,
}

fn default_version() -> u32 { 1 }

impl StyleProfile {
    /// Parse + validate from TOML. Returns Err on schema/version/validation failures.
    pub fn from_toml(s: &str) -> Result<Self, VoiceError> {
        let parsed: Self = toml::from_str(s)?;
        parsed.validate()?;
        Ok(parsed)
    }

    /// Internal validation. Called by from_toml; safe to call directly.
    pub fn validate(&self) -> Result<(), VoiceError>;
}
```

`#[serde(deny_unknown_fields)]` ensures typos in TOML keys (e.g., `sentance_length_target = ...`) fail loudly at parse time rather than silently using defaults.

## 4. `BlendRecipe` + `PartialStyleProfile`

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendRecipe {
    #[serde(default = "default_version")]
    pub version: u32,
    pub blend: Vec<BlendEntry>,             // 1..=10 entries, weights sum to 1.0
    #[serde(default)]
    pub overrides: PartialStyleProfile,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendEntry {
    pub writer: String,                     // corpus key (resolved by P1.2b)
    pub weight: f64,                        // 0.0..=1.0
}

/// Every StyleProfile field as Option<_>. Used in BlendRecipe.overrides.
/// User wins over blend in P1.2d's merge.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialStyleProfile {
    #[serde(default)] pub sentence_length_target: Option<SentenceLengthTarget>,
    #[serde(default)] pub sentence_length_distribution: Option<[u8; 4]>,
    #[serde(default)] pub fragment_frequency: Option<FragmentFrequency>,
    #[serde(default)] pub opening_patterns: Option<Vec<OpeningPattern>>,
    #[serde(default)] pub opening_pattern_weights: Option<Vec<f64>>,
    #[serde(default)] pub formatting: Option<Formatting>,
    #[serde(default)] pub emoji_policy: Option<EmojiPolicy>,
    #[serde(default)] pub hashtag_policy: Option<HashtagPolicy>,
    #[serde(default)] pub specificity_target: Option<SpecificityTarget>,
    #[serde(default)] pub voice_traits: Option<Vec<String>>,
    #[serde(default)] pub ai_tells_to_avoid: Option<Vec<String>>,
    #[serde(default)] pub thread_rhythm: Option<ThreadRhythm>,
    #[serde(default)] pub thread_max_length: Option<u32>,
    #[serde(default)] pub thread_opener_must_hook: Option<bool>,
    #[serde(default)] pub topical_obsessions: Option<Vec<String>>,
    #[serde(default)] pub topical_avoidances: Option<Vec<String>>,
}
```

`PartialStyleProfile` is hand-written (not macro-derived from `StyleProfile`) — the field count is bounded (~16) and writing it by hand keeps the diff easy to review. If the field set ever grows substantially, a derive macro becomes worth considering.

`BlendRecipe::from_toml(s) -> Result<Self, VoiceError>` and `validate(&self)` follow the same pattern as `StyleProfile`.

## 5. Validation rules

### `StyleProfile::validate(&self) -> Result<(), VoiceError>`

| Rule | Error message |
|---|---|
| `version == 1` | `UnsupportedVersion(N)` |
| `sentence_length_distribution.iter().sum::<u32>() == 100` (use u32 to avoid u8 overflow) | `Validation("sentence_length_distribution must sum to 100")` |
| `opening_patterns.len() == opening_pattern_weights.len()` | `Validation("opening_patterns and opening_pattern_weights must have the same length")` |
| `(opening_pattern_weights.iter().sum::<f64>() - 1.0).abs() < 1e-6` (only when non-empty) | `Validation("opening_pattern_weights must sum to 1.0")` |
| Each weight in 0.0..=1.0 | `Validation("opening_pattern_weights must each be in [0, 1]")` |
| `(1..=25).contains(&thread_max_length)` | `Validation("thread_max_length must be 1..=25")` |

### `BlendRecipe::validate(&self) -> Result<(), VoiceError>`

| Rule | Error message |
|---|---|
| `version == 1` | `UnsupportedVersion(N)` |
| `(1..=10).contains(&blend.len())` | `Validation("blend must contain 1..=10 entries")` |
| `(blend.iter().map(|e| e.weight).sum::<f64>() - 1.0).abs() < 1e-6` | `Validation("blend weights must sum to 1.0")` |
| Each `weight` in `0.0..=1.0` | `Validation("each blend weight must be in [0, 1]")` |
| Writer names non-empty after trim | `Validation("blend.writer must not be empty")` |
| Writer names unique within the recipe | `Validation("duplicate writer in blend: '{name}'")` |

`PartialStyleProfile` does not validate (it's a partial; constraints apply only after merging in P1.2d).

## 6. Error type

```rust
#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    #[error("toml parse: {0}")]
    Parse(#[from] toml::de::Error),

    #[error("unsupported profile version: {0} (expected 1)")]
    UnsupportedVersion(u32),

    #[error("validation: {0}")]
    Validation(String),
}
```

Lives in `voice/error.rs`. Re-exported via `voice::mod.rs`.

## 7. File structure

```
crates/heartbit-ghost/src/
  lib.rs                 # add `pub mod voice;`
  voice/
    mod.rs               # re-exports of types and VoiceError
    error.rs             # VoiceError
    style.rs             # StyleProfile + Formatting + 11 enums + validate()
    blend.rs             # BlendRecipe + BlendEntry + PartialStyleProfile + validate()
```

`Cargo.toml` adds nothing — `serde`, `serde_json`, `thiserror`, `toml` are already deps.

## 8. Test plan (~25-30 tests)

### Round-trip tests
- Serialize the §2.2 example → struct → TOML → struct, deep equality (verifies serde works in both directions)

### Spec verbatim
- Parse the §2.2 example exactly as written → succeeds + validates

### Closed-vocabulary rejection
- `sentence_length_target = "shortish"` (typo) → parse error mentions valid variants
- `formatting.em_dashes = "loved"` → parse error
- One test per closed-vocabulary enum (samples — not all 11)

### Validation negative tests
- `sentence_length_distribution = [40, 30, 20, 5]` (sums to 95) → `Validation("...sum to 100")`
- `opening_patterns.len() != opening_pattern_weights.len()` → length-mismatch error
- `opening_pattern_weights = [0.3, 0.3, 0.3, 0.05]` (sums to 0.95) → `Validation("weights must sum to 1.0")`
- `opening_pattern_weights = [-0.1, 0.5, 0.3, 0.3]` (negative) → out-of-range
- `thread_max_length = 30` → `Validation("must be 1..=25")`
- `thread_max_length = 0` → same

### Versioning
- `version = 1` (explicit) → parses + validates
- `version` absent → defaults to 1, parses + validates
- `version = 999` → `UnsupportedVersion(999)`

### Forward compat
- Profile written today with all fields set → parses
- Same TOML with a future-added field stripped → still parses (uses default)

### `BlendRecipe` tests (mirror StyleProfile structure)
- Happy path: 5 writers, weights `[0.3, 0.2, 0.2, 0.15, 0.15]` → parses + validates
- Empty blend `blend = []` → `Validation("1..=10 entries")`
- 11 entries → same error
- Weights sum to 0.99 → `Validation("...sum to 1.0")`
- Negative weight → out-of-range
- Duplicate writer → `Validation("duplicate writer: 'karpathy'")`
- Empty writer name (`writer = ""`) → `Validation("must not be empty")`
- Whitespace-only writer name (`writer = "   "`) → same

### `PartialStyleProfile` tests
- Empty TOML `[overrides]` → all fields `None`
- Partial population (`overrides = { lowercase = true, em_dashes = "forbidden" }`) → only those fields `Some`
- All fields populated → all fields `Some`, equivalent to full StyleProfile

### Defaults via `#[serde(default)]`
- Each `Default` impl on the enums returns a sensible variant (e.g., `EmojiPolicy::default() == EmojiPolicy::RarePunchlineOnly`)

Total: ~28 named tests across `style.rs`, `blend.rs`, `error.rs`.

## 9. Acceptance criteria

P1.2a is done when:

- All types compile cleanly under `cargo check -p heartbit-ghost`
- `cargo fmt -- --check && cargo clippy -p heartbit-ghost -- -D warnings && cargo test -p heartbit-ghost --lib voice` all green
- ~28 tests pass; coverage spans round-trip, validation negative paths, and version handling
- The §2.2 example parses verbatim
- `heartbit_ghost::voice::{StyleProfile, BlendRecipe, BlendEntry, PartialStyleProfile, Formatting, VoiceError, /* 11 enums */}` are reachable as a public surface

## 10. Out of scope (explicit)

- Actual blend algorithm (P1.2d — `merge_profiles(BlendRecipe, &[StyleProfile]) -> StyleProfile`)
- Corpus storage / loading (P1.2b — JSONL exemplar pool in `heartbit_core::Memory`)
- LLM-based style extractor (P1.2c — sub-agent that reads corpus → emits `StyleProfile`)
- CLI bodies (P1.2e — `corpus add/list/remove`, `profile rebuild/diff`)
- Translation of `StyleProfile` to writer-prompt strings (P1.2c — runtime conditioning)
- Profile diff format (P1.2e)
- Migrations between profile versions (deferred until first breaking change)
- Schema-driven validation of `PartialStyleProfile` (constraints only apply after merge)
- Profile snapshot hashing for audit trails (deferred to P1.4 audit work)

## 11. Open questions (deferred to implementation)

These are minor decisions the implementer can make during the plan:

- Whether `OpeningPattern` ships exactly the 7 variants listed here or starts with the 4 from §2.2 (`ClaimFirst`, `NumberFirst`, `SceneFirst`, `QuestionFirst`) and adds the others as needed. `#[non_exhaustive]` makes either path safe.
- Whether `PartialStyleProfile` is hand-written (one struct, 16 fields) or generated via a derive macro. Hand-written is fine; a macro becomes attractive when the field count grows or duplication causes drift.
- Exact error message wording — the table above is illustrative; the implementer can sharpen during TDD.
- Whether `validate()` collects all errors and reports them together, or fails fast on the first violation. Fail-fast is simpler and matches the "first-Deny-wins" pattern used elsewhere in the codebase. Adopt fail-fast.

## 12. Reference

- Umbrella spec: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md` §2.2 (the TOML example)
- Foundation Phase 0 spec: `docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md`
- P1.0 spec / plan: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md` §9 P1.0; plan at `docs/superpowers/plans/2026-05-07-heartbit-ghost-p1.0-scaffolding.md`
- P1.1 spec / plan: `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.1-x-tools-design.md`; plan at `docs/superpowers/plans/2026-05-07-heartbit-ghost-p1.1-x-tools.md`
