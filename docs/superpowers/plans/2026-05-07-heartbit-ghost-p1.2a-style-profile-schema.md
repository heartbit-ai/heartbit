# heartbit-ghost P1.2a — Style Profile Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `crates/heartbit-ghost/src/voice/` with `StyleProfile` + `BlendRecipe` + `PartialStyleProfile` + 11 closed-vocabulary enums + `Formatting` + `VoiceError` — pure data types with TOML serde and parse-time validation. ~28 tests.

**Architecture:** New submodule `voice/` under heartbit-ghost. Types only — no LLM, no corpus, no blender, no CLI (those are P1.2b/c/d/e). Closed-vocabulary fields use `#[non_exhaustive]` enums with snake_case serde rename; free-form fields stay `Vec<String>`. Validation runs at parse time via `Result`-returning factories.

**Tech Stack:** Rust 2024, `serde`, `serde_json`, `toml`, `thiserror` — all already in heartbit-ghost's deps.

**Spec:** `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`

**Branch:** `feat/heartbit-ghost-p1.2a`

---

## File Structure

### New files
- `crates/heartbit-ghost/src/voice/mod.rs` — module declarations + re-exports
- `crates/heartbit-ghost/src/voice/error.rs` — `VoiceError`
- `crates/heartbit-ghost/src/voice/style.rs` — 11 enums + `Formatting` + `StyleProfile`
- `crates/heartbit-ghost/src/voice/blend.rs` — `BlendRecipe` + `BlendEntry` + `PartialStyleProfile`

### Modified files
- `crates/heartbit-ghost/src/lib.rs` — add `pub mod voice;`

### No Cargo.toml changes
`serde`, `serde_json`, `toml`, `thiserror` are already deps of heartbit-ghost (added in P1.1 for the X tools). No new dependencies needed for P1.2a.

---

## Task 1: Module scaffolding + `VoiceError`

**Why:** Get the new module wired up so subsequent tasks have a place to land. Smallest possible self-contained commit.

**Files:**
- Create: `crates/heartbit-ghost/src/voice/error.rs`
- Create: `crates/heartbit-ghost/src/voice/mod.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs`

- [ ] **Step 1: Create `error.rs`**

```rust
//! Error type for voice modeling (style profile + blend recipe).

use thiserror::Error;

/// Errors produced when parsing or validating voice modeling types.
#[derive(Debug, Error)]
pub enum VoiceError {
    /// TOML deserialization failed (syntax, missing required field, unknown enum variant, etc.).
    #[error("toml parse: {0}")]
    Parse(#[from] toml::de::Error),

    /// Profile or recipe declares a `version` we don't know how to deserialize.
    #[error("unsupported profile version: {0} (expected 1)")]
    UnsupportedVersion(u32),

    /// A semantic invariant failed (sums don't match, ranges out of bounds, duplicates, etc.).
    #[error("validation: {0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_version_message() {
        let err = VoiceError::UnsupportedVersion(2);
        let s = format!("{err}");
        assert!(s.contains("unsupported profile version"));
        assert!(s.contains("2"));
        assert!(s.contains("expected 1"));
    }

    #[test]
    fn validation_message_propagates_payload() {
        let err = VoiceError::Validation("weights must sum to 1.0".into());
        let s = format!("{err}");
        assert!(s.starts_with("validation:"));
        assert!(s.contains("weights must sum to 1.0"));
    }

    #[test]
    fn parse_error_wraps_toml_de() {
        // Force a toml::de::Error by deserializing invalid TOML.
        let toml_err: toml::de::Error = toml::from_str::<i32>("not a number").unwrap_err();
        let err: VoiceError = toml_err.into();
        let s = format!("{err}");
        assert!(s.starts_with("toml parse:"));
    }
}
```

- [ ] **Step 2: Create `mod.rs`**

```rust
//! Voice modeling — style profiles, blend recipes, partial overrides.
//!
//! P1.2a ships pure data types only. Future sub-phases:
//! - P1.2b: corpus storage (memory namespace, JSONL load)
//! - P1.2c: LLM-based style extractor (corpus → StyleProfile)
//! - P1.2d: blend algorithm (BlendRecipe + N profiles → 1 profile)
//! - P1.2e: CLI bodies for `corpus add/list/remove`, `profile rebuild/diff`

pub mod error;
pub mod style;
pub mod blend;

pub use error::VoiceError;
pub use style::{
    EmDashPolicy, EmojiPolicy, FragmentFrequency, Formatting, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
pub use blend::{BlendEntry, BlendRecipe, PartialStyleProfile};
```

(`style` and `blend` modules don't exist yet — tasks 2-4 create them. The `pub mod` declarations here will fail to compile until those land. That's intentional: each subsequent task's "compile passes" milestone gates its own commit.)

- [ ] **Step 3: Add `pub mod voice;` to `lib.rs`**

Open `crates/heartbit-ghost/src/lib.rs`. Find the existing module declarations (after `XGhostPersona` definition or alongside `pub mod tools;`). Add:

```rust
pub mod voice;
```

Place it next to `pub mod tools;` (alphabetical — `tools` < `voice`).

- [ ] **Step 4: Verify the error module compiles in isolation**

Because `mod.rs` declares `pub mod style;` and `pub mod blend;` (which don't exist yet), the crate as a whole won't compile until Task 2 lands. To verify Task 1's actual deliverable (the error type), comment out the `style`/`blend` lines and the `pub use style::*;` / `pub use blend::*;` lines in `mod.rs` for now. They'll be uncommented in Tasks 2 and 4.

```rust
// In mod.rs — temporarily for Task 1 only:
pub mod error;
// pub mod style;   // uncommented in Task 2
// pub mod blend;   // uncommented in Task 4

pub use error::VoiceError;
// re-exports from style and blend uncommented in Tasks 2 and 4
```

Run:
```bash
cargo test -p heartbit-ghost --lib voice::error
```

Expected: 3 passed.

- [ ] **Step 5: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/voice/error.rs \
        crates/heartbit-ghost/src/voice/mod.rs \
        crates/heartbit-ghost/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice module scaffolding + VoiceError (P1.2a)

Lays the voice/ submodule down with the error type. Subsequent tasks
fill in style.rs (StyleProfile + 11 enums + Formatting) and blend.rs
(BlendRecipe + BlendEntry + PartialStyleProfile).

3 tests on VoiceError display + From<toml::de::Error> wrapping.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md
EOF
)"
```

---

## Task 2: 11 closed-vocabulary enums + `Formatting`

**Why:** Build the schema vocabulary used by `StyleProfile` (next task) and `PartialStyleProfile` (Task 4).

**Files:**
- Create: `crates/heartbit-ghost/src/voice/style.rs` (start of file — enums + Formatting only; StyleProfile lands in Task 3)
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (uncomment `pub mod style;` + the style re-exports)

- [ ] **Step 1: Create `style.rs` with all 11 enums + `Formatting`**

```rust
//! Style profile schema — closed-vocabulary enums, formatting struct, and
//! the top-level `StyleProfile` type (added in Task 3).

use serde::{Deserialize, Serialize};

/// Sentence-length distribution preference for a voice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SentenceLengthTarget {
    /// Target sub-10-word sentences. Punchy, paratactic.
    Short,
    /// Mix of short and medium sentences. Balanced.
    #[default]
    Mixed,
    /// Target 30+ word sentences. Subordinate clauses, considered.
    Long,
}

/// How often the writer uses sentence fragments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FragmentFrequency {
    /// Rare. Almost always full sentences.
    Rare,
    /// Mid. Occasional fragment for rhythm.
    #[default]
    Occasional,
    /// Frequent. Fragments are part of the cadence.
    Common,
}

/// Common patterns for how a post starts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OpeningPattern {
    /// "X is wrong about Y."
    #[default]
    ClaimFirst,
    /// "5 lessons from..."
    NumberFirst,
    /// "I was at..."
    SceneFirst,
    /// "What if..."
    QuestionFirst,
    /// Aphorism / one-liner truth.
    AphoristicFirst,
    /// "Last night I saw..." — mini-narrative hook.
    AnecdoteFirst,
    /// Direct contrarian framing — "Everyone's wrong that..."
    ContrarianFirst,
}

/// Sentence-final period policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PeriodsPolicy {
    /// Always end sentences with `.`
    #[default]
    Always,
    /// Sometimes drop the final period (especially on punchlines).
    Optional,
    /// Most sentences end without a period.
    Rare,
}

/// Em-dash usage policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EmDashPolicy {
    /// Em-dashes are a signature device.
    Preferred,
    /// Em-dashes are fine when appropriate.
    #[default]
    Ok,
    /// Em-dashes are avoided (often an AI tell).
    Forbidden,
}

/// Quotation mark style.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum QuotationMarks {
    /// `"` — straight double quotes (US standard).
    #[default]
    Double,
    /// `'` — straight single quotes.
    Single,
    /// `“ ”` — typographic curly quotes.
    Smart,
}

/// Line-break density.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum LineBreaks {
    /// One line break between paragraphs.
    #[default]
    Single,
    /// Double-spaced — visual breathing room.
    Double,
    /// Rhythmic, intentional spacing for emphasis.
    Rhythmic,
}

/// Emoji usage policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EmojiPolicy {
    /// No emoji ever.
    Never,
    /// Emoji only when it IS the punchline.
    #[default]
    RarePunchlineOnly,
    /// Occasional emoji for tone.
    Occasional,
    /// Emoji used liberally.
    Frequent,
}

/// Hashtag usage policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum HashtagPolicy {
    /// Never use hashtags (they reduce engagement on X).
    #[default]
    Never,
    /// Rarely — only when essential.
    Rare,
    /// Use hashtags only when topic-relevant.
    TopicRelevant,
    /// Use hashtags routinely.
    Always,
}

/// How specific the writer's claims tend to be.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpecificityTarget {
    /// Vague generalities.
    Low,
    /// Mix of general and specific.
    #[default]
    Medium,
    /// Real names, real numbers, real anecdotes — high specificity.
    High,
}

/// How threads are structured.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ThreadRhythm {
    /// Tweet-after-tweet, no special structure.
    #[default]
    Linear,
    /// List of items, then a closing payoff.
    ListThenPayoff,
    /// Sets up a punchline, calls back to it later in the thread.
    PunchlineCallbacks,
}

/// Formatting habits — how the writer renders prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Formatting {
    /// All-lowercase.
    pub lowercase: bool,
    /// Period policy.
    pub periods: PeriodsPolicy,
    /// Em-dash policy.
    pub em_dashes: EmDashPolicy,
    /// Quotation mark style.
    pub quotation_marks: QuotationMarks,
    /// Line-break density.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentence_length_target_serde_round_trip() {
        for variant in [
            SentenceLengthTarget::Short,
            SentenceLengthTarget::Mixed,
            SentenceLengthTarget::Long,
        ] {
            let s = serde_json::to_string(&variant).unwrap();
            let back: SentenceLengthTarget = serde_json::from_str(&s).unwrap();
            assert_eq!(back, variant);
        }
        // Confirm the wire shape is snake_case strings.
        assert_eq!(
            serde_json::to_string(&SentenceLengthTarget::Long).unwrap(),
            "\"long\""
        );
    }

    #[test]
    fn em_dash_policy_serde_round_trip() {
        let s = serde_json::to_string(&EmDashPolicy::Forbidden).unwrap();
        assert_eq!(s, "\"forbidden\"");
        let back: EmDashPolicy = serde_json::from_str(&s).unwrap();
        assert_eq!(back, EmDashPolicy::Forbidden);
    }

    #[test]
    fn emoji_policy_rare_punchline_only_uses_snake_case() {
        let s = serde_json::to_string(&EmojiPolicy::RarePunchlineOnly).unwrap();
        assert_eq!(s, "\"rare_punchline_only\"");
        let back: EmojiPolicy = serde_json::from_str("\"rare_punchline_only\"").unwrap();
        assert_eq!(back, EmojiPolicy::RarePunchlineOnly);
    }

    #[test]
    fn unknown_variant_rejected_at_parse_time() {
        let err = serde_json::from_str::<SentenceLengthTarget>("\"shortish\"").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("unknown variant") || s.contains("shortish"));
    }

    #[test]
    fn formatting_defaults_are_sensible() {
        let f = Formatting::default();
        assert!(!f.lowercase);
        assert_eq!(f.periods, PeriodsPolicy::Always);
        assert_eq!(f.em_dashes, EmDashPolicy::Ok);
        assert_eq!(f.quotation_marks, QuotationMarks::Double);
        assert_eq!(f.line_breaks, LineBreaks::Single);
    }

    #[test]
    fn formatting_serde_round_trip_via_toml() {
        let f = Formatting {
            lowercase: true,
            periods: PeriodsPolicy::Optional,
            em_dashes: EmDashPolicy::Forbidden,
            quotation_marks: QuotationMarks::Smart,
            line_breaks: LineBreaks::Rhythmic,
        };
        let s = toml::to_string(&f).unwrap();
        let back: Formatting = toml::from_str(&s).unwrap();
        assert_eq!(back, f);
    }

    #[test]
    fn enum_defaults_match_spec_intent() {
        assert_eq!(SentenceLengthTarget::default(), SentenceLengthTarget::Mixed);
        assert_eq!(FragmentFrequency::default(), FragmentFrequency::Occasional);
        assert_eq!(OpeningPattern::default(), OpeningPattern::ClaimFirst);
        assert_eq!(PeriodsPolicy::default(), PeriodsPolicy::Always);
        assert_eq!(EmDashPolicy::default(), EmDashPolicy::Ok);
        assert_eq!(QuotationMarks::default(), QuotationMarks::Double);
        assert_eq!(LineBreaks::default(), LineBreaks::Single);
        assert_eq!(EmojiPolicy::default(), EmojiPolicy::RarePunchlineOnly);
        assert_eq!(HashtagPolicy::default(), HashtagPolicy::Never);
        assert_eq!(SpecificityTarget::default(), SpecificityTarget::Medium);
        assert_eq!(ThreadRhythm::default(), ThreadRhythm::Linear);
    }
}
```

- [ ] **Step 2: Uncomment `pub mod style;` and the style re-export in `voice/mod.rs`**

Edit `crates/heartbit-ghost/src/voice/mod.rs` — uncomment the `style` lines:

```rust
pub mod error;
pub mod style;
// pub mod blend;   // uncommented in Task 4

pub use error::VoiceError;
pub use style::{
    EmDashPolicy, EmojiPolicy, FragmentFrequency, Formatting, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    ThreadRhythm,
};
// (StyleProfile re-export added by Task 3 below — for now, leave it out so step 3 compiles)
```

(`StyleProfile` doesn't exist yet — it lands in Task 3. Keep it out of the re-export until then.)

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::style
```

Expected: 7 passed.

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/voice/style.rs crates/heartbit-ghost/src/voice/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — 11 closed-vocabulary enums + Formatting (P1.2a)

The schema vocabulary used by StyleProfile (Task 3) and
PartialStyleProfile (Task 4). Each enum is non_exhaustive with
serde(rename_all = "snake_case") so unknown variants in TOML fail
loudly at parse time and future variants don't break consumers'
pattern matches.

Each enum has a sensible Default impl so serde(default) on the
PartialStyleProfile fields works cleanly.

7 tests: per-enum serde round-trip, snake_case wire format,
unknown-variant rejection, Formatting::default sanity check,
TOML round-trip, every enum default cross-checked.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md §1
EOF
)"
```

---

## Task 3: `StyleProfile` + `from_toml` + `validate`

**Why:** The top-level type that parses §2.2 of the umbrella spec verbatim. Validation enforces the invariants (sums, ranges, parallel array lengths).

**Files:**
- Modify: `crates/heartbit-ghost/src/voice/style.rs` (append `StyleProfile` to the existing file)
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (add `StyleProfile` to the re-export list)

- [ ] **Step 1: Append `StyleProfile` to `style.rs`**

After the `Formatting` block (and before the `#[cfg(test)] mod tests` block — or move tests below `StyleProfile`):

```rust
use crate::voice::error::VoiceError;

/// Returns the current style profile schema version (1).
fn default_version() -> u32 {
    1
}

/// Structured fingerprint of a voice. Parses directly from the §2.2 TOML
/// example in the umbrella heartbit-ghost spec.
///
/// # Construction
///
/// Use [`StyleProfile::from_toml`] for the full parse + validate flow:
///
/// ```rust,no_run
/// use heartbit_ghost::voice::StyleProfile;
///
/// let toml_text = std::fs::read_to_string("personas/x.toml").unwrap();
/// let profile = StyleProfile::from_toml(&toml_text).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StyleProfile {
    /// Schema version. Currently must be 1.
    #[serde(default = "default_version")]
    pub version: u32,

    // ---- Sentence-level ----
    /// Preferred sentence length category.
    pub sentence_length_target: SentenceLengthTarget,
    /// Distribution across length buckets `[<10, 10-20, 20-40, >40]` words. Sums to 100.
    pub sentence_length_distribution: [u8; 4],
    /// Frequency of sentence fragments.
    pub fragment_frequency: FragmentFrequency,

    // ---- Opening patterns (parallel arrays) ----
    /// Patterns the writer uses to open posts.
    pub opening_patterns: Vec<OpeningPattern>,
    /// Probability weights parallel to `opening_patterns`. Sums to 1.0.
    pub opening_pattern_weights: Vec<f64>,

    /// Formatting habits.
    pub formatting: Formatting,

    /// Emoji usage policy.
    pub emoji_policy: EmojiPolicy,
    /// Hashtag usage policy.
    pub hashtag_policy: HashtagPolicy,
    /// Specificity target — vague vs. real-numbers-and-names.
    pub specificity_target: SpecificityTarget,

    /// Free-form descriptors used by writer/critic LLM prompts.
    pub voice_traits: Vec<String>,
    /// Free-form phrases the critic flags as AI-tells to avoid.
    pub ai_tells_to_avoid: Vec<String>,

    // ---- Thread structure ----
    /// How threads are structured.
    pub thread_rhythm: ThreadRhythm,
    /// Maximum thread length. Bounded 1..=25 (matches `TwitterThreadTool::MAX_THREAD_LENGTH`).
    pub thread_max_length: u32,
    /// Whether thread openers must hook (i.e. earn the read).
    pub thread_opener_must_hook: bool,

    // ---- Topical ----
    /// Topics the persona will gravitate toward.
    pub topical_obsessions: Vec<String>,
    /// Topics the persona will avoid unless explicitly nudged.
    pub topical_avoidances: Vec<String>,
}

impl StyleProfile {
    /// Parse a `StyleProfile` from a TOML string and validate it.
    ///
    /// Returns `Err` on TOML syntax errors, unknown enum variants, missing
    /// required fields, unsupported schema versions, or any failed validation
    /// invariant (see [`StyleProfile::validate`]).
    pub fn from_toml(s: &str) -> Result<Self, VoiceError> {
        let parsed: Self = toml::from_str(s)?;
        parsed.validate()?;
        Ok(parsed)
    }

    /// Run the validation rules. Called by [`StyleProfile::from_toml`]; safe
    /// to call directly on a profile constructed by hand.
    pub fn validate(&self) -> Result<(), VoiceError> {
        if self.version != 1 {
            return Err(VoiceError::UnsupportedVersion(self.version));
        }

        // sentence_length_distribution must sum to 100. Use u32 to avoid
        // overflow if all four entries were near u8::MAX.
        let dist_sum: u32 = self
            .sentence_length_distribution
            .iter()
            .map(|&v| u32::from(v))
            .sum();
        if dist_sum != 100 {
            return Err(VoiceError::Validation(format!(
                "sentence_length_distribution must sum to 100 (got {dist_sum})"
            )));
        }

        // opening_patterns and opening_pattern_weights are parallel arrays.
        if self.opening_patterns.len() != self.opening_pattern_weights.len() {
            return Err(VoiceError::Validation(format!(
                "opening_patterns and opening_pattern_weights must have the same length \
                 (patterns={}, weights={})",
                self.opening_patterns.len(),
                self.opening_pattern_weights.len()
            )));
        }

        if !self.opening_pattern_weights.is_empty() {
            let weights_sum: f64 = self.opening_pattern_weights.iter().sum();
            if (weights_sum - 1.0).abs() > 1e-6 {
                return Err(VoiceError::Validation(format!(
                    "opening_pattern_weights must sum to 1.0 (got {weights_sum})"
                )));
            }
            for &w in &self.opening_pattern_weights {
                if !(0.0..=1.0).contains(&w) {
                    return Err(VoiceError::Validation(format!(
                        "opening_pattern_weights must each be in [0, 1] (got {w})"
                    )));
                }
            }
        }

        if !(1..=25).contains(&self.thread_max_length) {
            return Err(VoiceError::Validation(format!(
                "thread_max_length must be 1..=25 (got {})",
                self.thread_max_length
            )));
        }

        Ok(())
    }
}
```

- [ ] **Step 2: Add `StyleProfile` to `voice/mod.rs` re-exports**

```rust
pub use style::{
    EmDashPolicy, EmojiPolicy, FragmentFrequency, Formatting, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
```

(Add `StyleProfile,` between `SpecificityTarget` and `ThreadRhythm` alphabetically.)

- [ ] **Step 3: Add `StyleProfile` tests to the existing `tests` mod in `style.rs`**

Inside the existing `#[cfg(test)] mod tests`:

```rust
    /// The §2.2 example from the umbrella heartbit-ghost spec, verbatim.
    /// This is the load-bearing fixture: if it stops parsing, the spec drifted.
    const SPEC_EXAMPLE: &str = r#"
version = 1

sentence_length_target = "short"
sentence_length_distribution = [40, 30, 20, 10]
fragment_frequency = "common"

opening_patterns = ["claim_first", "number_first", "scene_first", "question_first"]
opening_pattern_weights = [0.4, 0.2, 0.2, 0.2]

[formatting]
lowercase = true
periods = "optional"
em_dashes = "forbidden"
quotation_marks = "double"
line_breaks = "single"

emoji_policy = "rare_punchline_only"
hashtag_policy = "never"
specificity_target = "high"

voice_traits = [
    "specific",
    "contrarian_when_defensible",
    "self_deprecating_occasional",
    "no_hedging",
    "no_balanced_both_sides",
]

ai_tells_to_avoid = [
    "delve",
    "tapestry",
    "navigate",
    "it's important to note",
    "in conclusion",
    "balanced both-sides",
    "as an AI",
    "I cannot",
    "while it's true that",
]

thread_rhythm = "punchline_callbacks"
thread_max_length = 10
thread_opener_must_hook = true

topical_obsessions = ["AI capabilities", "engineering craftsmanship", "research taste"]
topical_avoidances = ["politics", "stock_picks", "celebrity_gossip"]
"#;

    fn valid_profile() -> StyleProfile {
        StyleProfile::from_toml(SPEC_EXAMPLE).expect("spec example must parse")
    }

    #[test]
    fn spec_example_parses_and_validates() {
        let profile = StyleProfile::from_toml(SPEC_EXAMPLE).expect("parses");
        assert_eq!(profile.version, 1);
        assert_eq!(profile.sentence_length_target, SentenceLengthTarget::Short);
        assert_eq!(profile.sentence_length_distribution, [40, 30, 20, 10]);
        assert_eq!(profile.formatting.lowercase, true);
        assert_eq!(profile.formatting.em_dashes, EmDashPolicy::Forbidden);
        assert_eq!(profile.thread_max_length, 10);
        assert!(profile.ai_tells_to_avoid.iter().any(|s| s == "delve"));
    }

    #[test]
    fn round_trip_serialization_preserves_data() {
        let original = valid_profile();
        let serialized = toml::to_string(&original).unwrap();
        let reparsed = StyleProfile::from_toml(&serialized).unwrap();
        assert_eq!(reparsed, original);
    }

    #[test]
    fn version_absent_defaults_to_1() {
        let toml = SPEC_EXAMPLE.replace("version = 1\n", "");
        let profile = StyleProfile::from_toml(&toml).expect("parses");
        assert_eq!(profile.version, 1);
    }

    #[test]
    fn version_999_rejected() {
        let toml = SPEC_EXAMPLE.replace("version = 1", "version = 999");
        let err = StyleProfile::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::UnsupportedVersion(v) => assert_eq!(v, 999),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn sentence_length_distribution_sum_99_rejected() {
        let toml = SPEC_EXAMPLE.replace(
            "sentence_length_distribution = [40, 30, 20, 10]",
            "sentence_length_distribution = [40, 30, 20, 9]",
        );
        let err = StyleProfile::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("sentence_length_distribution"));
                assert!(msg.contains("100"));
                assert!(msg.contains("99"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn opening_patterns_weights_length_mismatch_rejected() {
        let toml = SPEC_EXAMPLE.replace(
            "opening_pattern_weights = [0.4, 0.2, 0.2, 0.2]",
            "opening_pattern_weights = [0.5, 0.5]",
        );
        let err = StyleProfile::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("opening_patterns"));
                assert!(msg.contains("opening_pattern_weights"));
                assert!(msg.contains("same length"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn opening_pattern_weights_sum_0_95_rejected() {
        let toml = SPEC_EXAMPLE.replace(
            "opening_pattern_weights = [0.4, 0.2, 0.2, 0.2]",
            "opening_pattern_weights = [0.35, 0.2, 0.2, 0.2]",
        );
        let err = StyleProfile::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("opening_pattern_weights"));
                assert!(msg.contains("1.0"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn opening_pattern_weights_negative_rejected() {
        let toml = SPEC_EXAMPLE.replace(
            "opening_pattern_weights = [0.4, 0.2, 0.2, 0.2]",
            "opening_pattern_weights = [-0.1, 0.5, 0.3, 0.3]",
        );
        let err = StyleProfile::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("opening_pattern_weights"));
                assert!(msg.contains("[0, 1]"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn thread_max_length_30_rejected() {
        let toml = SPEC_EXAMPLE.replace("thread_max_length = 10", "thread_max_length = 30");
        let err = StyleProfile::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("thread_max_length"));
                assert!(msg.contains("1..=25"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn thread_max_length_0_rejected() {
        let toml = SPEC_EXAMPLE.replace("thread_max_length = 10", "thread_max_length = 0");
        let err = StyleProfile::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => assert!(msg.contains("thread_max_length")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn thread_max_length_25_boundary_accepted() {
        let toml = SPEC_EXAMPLE.replace("thread_max_length = 10", "thread_max_length = 25");
        let profile = StyleProfile::from_toml(&toml).expect("boundary value 25 accepted");
        assert_eq!(profile.thread_max_length, 25);
    }

    #[test]
    fn typo_in_enum_variant_rejected() {
        let toml = SPEC_EXAMPLE.replace(
            "sentence_length_target = \"short\"",
            "sentence_length_target = \"shortish\"",
        );
        let err = StyleProfile::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Parse(_) => { /* correct path */ }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn unknown_field_rejected_via_deny_unknown_fields() {
        let toml = format!("{SPEC_EXAMPLE}\nbogus_field = \"oops\"");
        let err = StyleProfile::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Parse(_) => { /* correct path */ }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }
```

- [ ] **Step 4: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::style
```

Expected: 19 passed (7 from Task 2 + 12 new).

- [ ] **Step 5: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/voice/style.rs crates/heartbit-ghost/src/voice/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — StyleProfile with from_toml + validate (P1.2a)

The top-level fingerprint type. Parses §2.2 of the umbrella heartbit-ghost
spec verbatim. serde(deny_unknown_fields) rejects typos at parse time.
Validation enforces:
- version == 1 (UnsupportedVersion otherwise)
- sentence_length_distribution sums to 100
- opening_patterns / opening_pattern_weights parallel + weights sum 1.0
- weights individually in [0, 1]
- thread_max_length in 1..=25 (matches TwitterThreadTool cap from P1.1)

12 new tests (19 total in voice::style): spec verbatim parse, round-trip,
version handling, every validation negative path, boundary cases,
deny_unknown_fields, enum typo rejection.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md §3 §5
EOF
)"
```

---

## Task 4: `BlendRecipe` + `BlendEntry` + `PartialStyleProfile`

**Why:** Closes the P1.2a scope. `BlendRecipe` drives P1.2d's blender; `PartialStyleProfile` is the override shape.

**Files:**
- Create: `crates/heartbit-ghost/src/voice/blend.rs`
- Modify: `crates/heartbit-ghost/src/voice/mod.rs` (uncomment `pub mod blend;` and the blend re-export)

- [ ] **Step 1: Create `blend.rs`**

```rust
//! Blend recipe — declares how N writer profiles combine into one, with an
//! optional `PartialStyleProfile` of user-provided overrides applied last.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::voice::error::VoiceError;
use crate::voice::style::{
    EmDashPolicy, EmojiPolicy, FragmentFrequency, Formatting, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    ThreadRhythm,
};

fn default_version() -> u32 {
    1
}

/// Recipe for blending N writer profiles into a single `StyleProfile`.
///
/// The blend algorithm itself lives in P1.2d; this type just declares the
/// inputs (writer references with weights) and any user-provided overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendRecipe {
    /// Schema version. Currently must be 1.
    #[serde(default = "default_version")]
    pub version: u32,
    /// 1..=10 entries; weights sum to 1.0.
    pub blend: Vec<BlendEntry>,
    /// Optional user overrides applied after the blend.
    #[serde(default)]
    pub overrides: PartialStyleProfile,
}

/// One writer's contribution to a blend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendEntry {
    /// Corpus key for this writer. Resolved against the corpus store in P1.2b.
    pub writer: String,
    /// Weight in the blend; in [0, 1]. All weights in a recipe sum to 1.0.
    pub weight: f64,
}

/// Every `StyleProfile` field as `Option<_>`. Used for blend overrides where
/// only a subset of fields is provided.
///
/// Constraints (sums, ranges) only apply after merge — this partial doesn't
/// validate on its own.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialStyleProfile {
    /// Override for `sentence_length_target`.
    #[serde(default)]
    pub sentence_length_target: Option<SentenceLengthTarget>,
    /// Override for `sentence_length_distribution`.
    #[serde(default)]
    pub sentence_length_distribution: Option<[u8; 4]>,
    /// Override for `fragment_frequency`.
    #[serde(default)]
    pub fragment_frequency: Option<FragmentFrequency>,
    /// Override for `opening_patterns`.
    #[serde(default)]
    pub opening_patterns: Option<Vec<OpeningPattern>>,
    /// Override for `opening_pattern_weights`.
    #[serde(default)]
    pub opening_pattern_weights: Option<Vec<f64>>,
    /// Override for `formatting`.
    #[serde(default)]
    pub formatting: Option<Formatting>,
    /// Override for `emoji_policy`.
    #[serde(default)]
    pub emoji_policy: Option<EmojiPolicy>,
    /// Override for `hashtag_policy`.
    #[serde(default)]
    pub hashtag_policy: Option<HashtagPolicy>,
    /// Override for `specificity_target`.
    #[serde(default)]
    pub specificity_target: Option<SpecificityTarget>,
    /// Override for `voice_traits`.
    #[serde(default)]
    pub voice_traits: Option<Vec<String>>,
    /// Override for `ai_tells_to_avoid`.
    #[serde(default)]
    pub ai_tells_to_avoid: Option<Vec<String>>,
    /// Override for `thread_rhythm`.
    #[serde(default)]
    pub thread_rhythm: Option<ThreadRhythm>,
    /// Override for `thread_max_length`.
    #[serde(default)]
    pub thread_max_length: Option<u32>,
    /// Override for `thread_opener_must_hook`.
    #[serde(default)]
    pub thread_opener_must_hook: Option<bool>,
    /// Override for `topical_obsessions`.
    #[serde(default)]
    pub topical_obsessions: Option<Vec<String>>,
    /// Override for `topical_avoidances`.
    #[serde(default)]
    pub topical_avoidances: Option<Vec<String>>,
}

impl BlendRecipe {
    /// Parse a `BlendRecipe` from a TOML string and validate it.
    pub fn from_toml(s: &str) -> Result<Self, VoiceError> {
        let parsed: Self = toml::from_str(s)?;
        parsed.validate()?;
        Ok(parsed)
    }

    /// Validation rules:
    /// - `version` must equal 1
    /// - `blend.len()` in 1..=10
    /// - `blend` weights each in [0, 1] and sum to 1.0 (epsilon 1e-6)
    /// - each `writer` non-empty after trim, unique within the recipe
    pub fn validate(&self) -> Result<(), VoiceError> {
        if self.version != 1 {
            return Err(VoiceError::UnsupportedVersion(self.version));
        }

        if !(1..=10).contains(&self.blend.len()) {
            return Err(VoiceError::Validation(format!(
                "blend must contain 1..=10 entries (got {})",
                self.blend.len()
            )));
        }

        let weights_sum: f64 = self.blend.iter().map(|e| e.weight).sum();
        if (weights_sum - 1.0).abs() > 1e-6 {
            return Err(VoiceError::Validation(format!(
                "blend weights must sum to 1.0 (got {weights_sum})"
            )));
        }

        for entry in &self.blend {
            if !(0.0..=1.0).contains(&entry.weight) {
                return Err(VoiceError::Validation(format!(
                    "each blend weight must be in [0, 1] (got {} for '{}')",
                    entry.weight, entry.writer
                )));
            }
            if entry.writer.trim().is_empty() {
                return Err(VoiceError::Validation(
                    "blend.writer must not be empty".into(),
                ));
            }
        }

        let mut seen: HashSet<&str> = HashSet::new();
        for entry in &self.blend {
            if !seen.insert(entry.writer.as_str()) {
                return Err(VoiceError::Validation(format!(
                    "duplicate writer in blend: '{}'",
                    entry.writer
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AI/tech blend (matches the umbrella heartbit-ghost spec §2.3 example).
    const VALID_BLEND: &str = r#"
version = 1

[[blend]]
writer = "karpathy"
weight = 0.30

[[blend]]
writer = "eladgil"
weight = 0.20

[[blend]]
writer = "swyx"
weight = 0.20

[[blend]]
writer = "naval"
weight = 0.15

[[blend]]
writer = "sama"
weight = 0.15
"#;

    #[test]
    fn five_writer_happy_path_parses_and_validates() {
        let recipe = BlendRecipe::from_toml(VALID_BLEND).expect("parses");
        assert_eq!(recipe.version, 1);
        assert_eq!(recipe.blend.len(), 5);
        assert_eq!(recipe.blend[0].writer, "karpathy");
        assert!((recipe.blend.iter().map(|e| e.weight).sum::<f64>() - 1.0).abs() < 1e-6);
        assert_eq!(recipe.overrides, PartialStyleProfile::default());
    }

    #[test]
    fn empty_blend_rejected() {
        let toml = "version = 1\nblend = []\n";
        let err = BlendRecipe::from_toml(toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("1..=10"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn eleven_entries_rejected() {
        let mut entries = String::from("version = 1\n");
        for i in 0..11 {
            entries.push_str(&format!(
                "[[blend]]\nwriter = \"w{i}\"\nweight = {}\n\n",
                1.0 / 11.0
            ));
        }
        let err = BlendRecipe::from_toml(&entries).unwrap_err();
        match err {
            VoiceError::Validation(msg) => assert!(msg.contains("1..=10")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn weights_sum_0_99_rejected() {
        let toml = VALID_BLEND.replace("weight = 0.30", "weight = 0.29");
        let err = BlendRecipe::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("blend weights"));
                assert!(msg.contains("1.0"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn negative_weight_rejected() {
        let toml = VALID_BLEND.replace("weight = 0.15", "weight = -0.15"); // matches the first 0.15 occurrence
        let err = BlendRecipe::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("[0, 1]") || msg.contains("blend weight"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_writer_rejected() {
        let toml = r#"
version = 1

[[blend]]
writer = "karpathy"
weight = 0.5

[[blend]]
writer = "karpathy"
weight = 0.5
"#;
        let err = BlendRecipe::from_toml(toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("duplicate"));
                assert!(msg.contains("karpathy"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn empty_writer_name_rejected() {
        let toml = r#"
version = 1

[[blend]]
writer = ""
weight = 1.0
"#;
        let err = BlendRecipe::from_toml(toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => assert!(msg.contains("writer must not be empty")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn whitespace_only_writer_name_rejected() {
        let toml = r#"
version = 1

[[blend]]
writer = "   "
weight = 1.0
"#;
        let err = BlendRecipe::from_toml(toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => assert!(msg.contains("writer must not be empty")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn version_999_rejected() {
        let toml = VALID_BLEND.replace("version = 1", "version = 999");
        let err = BlendRecipe::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::UnsupportedVersion(v) => assert_eq!(v, 999),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn version_absent_defaults_to_1() {
        let toml = VALID_BLEND.replace("version = 1\n", "");
        let recipe = BlendRecipe::from_toml(&toml).expect("parses");
        assert_eq!(recipe.version, 1);
    }

    #[test]
    fn partial_style_profile_default_is_all_none() {
        let p = PartialStyleProfile::default();
        assert!(p.sentence_length_target.is_none());
        assert!(p.formatting.is_none());
        assert!(p.thread_max_length.is_none());
        assert!(p.voice_traits.is_none());
    }

    #[test]
    fn partial_overrides_accept_subset_of_fields() {
        let toml = r#"
version = 1

[[blend]]
writer = "karpathy"
weight = 1.0

[overrides]
thread_max_length = 7

[overrides.formatting]
lowercase = true
periods = "always"
em_dashes = "forbidden"
quotation_marks = "double"
line_breaks = "single"
"#;
        let recipe = BlendRecipe::from_toml(toml).expect("parses");
        assert_eq!(recipe.overrides.thread_max_length, Some(7));
        let formatting = recipe.overrides.formatting.expect("set");
        assert!(formatting.lowercase);
        assert_eq!(formatting.em_dashes, EmDashPolicy::Forbidden);
        // Other fields remain None.
        assert!(recipe.overrides.voice_traits.is_none());
        assert!(recipe.overrides.thread_rhythm.is_none());
    }
}
```

- [ ] **Step 2: Uncomment `pub mod blend;` + the blend re-export in `voice/mod.rs`**

```rust
pub mod error;
pub mod style;
pub mod blend;

pub use error::VoiceError;
pub use style::{
    EmDashPolicy, EmojiPolicy, FragmentFrequency, Formatting, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
pub use blend::{BlendEntry, BlendRecipe, PartialStyleProfile};
```

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib voice::blend
```

Expected: 11 passed.

- [ ] **Step 4: Workspace quality gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/voice/blend.rs crates/heartbit-ghost/src/voice/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): voice — BlendRecipe + PartialStyleProfile (P1.2a)

Closes P1.2a's data-types scope. BlendRecipe declares the inputs to
P1.2d's blend algorithm (writer + weight + optional overrides);
PartialStyleProfile is every StyleProfile field as Option<_>, used for
the user-provided overrides that win over the blended values.

Validation: version == 1, blend.len() in 1..=10, weights individually
in [0, 1] and sum to 1.0 (epsilon 1e-6), writer names non-empty after
trim and unique within the recipe.

11 new tests: 5-writer happy path, empty/oversize blend rejection,
weights sum mismatch, negative weight, duplicate writer, empty/
whitespace-only writer name, version handling, PartialStyleProfile
default + partial population.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md §4 §5
EOF
)"
```

---

## Task 5: Final acceptance + workspace quality gate

**Why:** Confirm P1.2a meets every acceptance criterion in the spec.

**Files:** none (verification only).

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -10
```

Expected: all green. Test count goes from 3779 (post-P1.1 baseline) to ~3809 (~30 new voice tests).

- [ ] **Step 2: Verify the spec verbatim parses against the actual binary**

A 1-line confirmation that the umbrella spec's §2.2 TOML example is parseable through the public API:

```bash
cargo test -p heartbit-ghost --lib spec_example_parses_and_validates 2>&1 | tail -3
```

Expected: `1 passed`.

- [ ] **Step 3: Verify the public surface is reachable**

```bash
cat <<'EOF' > /tmp/heartbit_ghost_voice_surface_check.rs
fn _check() {
    use heartbit_ghost::voice::{
        BlendEntry, BlendRecipe, EmDashPolicy, EmojiPolicy, FragmentFrequency, Formatting,
        HashtagPolicy, LineBreaks, OpeningPattern, PartialStyleProfile, PeriodsPolicy,
        QuotationMarks, SentenceLengthTarget, SpecificityTarget, StyleProfile, ThreadRhythm,
        VoiceError,
    };
    let _ = (
        BlendEntry { writer: String::new(), weight: 0.0 },
        BlendRecipe { version: 1, blend: vec![], overrides: PartialStyleProfile::default() },
        EmDashPolicy::Ok,
        EmojiPolicy::Never,
        FragmentFrequency::Rare,
        Formatting::default(),
        HashtagPolicy::Never,
        LineBreaks::Single,
        OpeningPattern::ClaimFirst,
        PartialStyleProfile::default(),
        PeriodsPolicy::Always,
        QuotationMarks::Double,
        SentenceLengthTarget::Short,
        SpecificityTarget::High,
        ThreadRhythm::Linear,
        VoiceError::UnsupportedVersion(0),
    );
    let _: fn(&str) -> Result<StyleProfile, VoiceError> = StyleProfile::from_toml;
    let _: fn(&str) -> Result<BlendRecipe, VoiceError> = BlendRecipe::from_toml;
}
EOF
echo "(Surface check is a compile-time assertion; verified by the workspace cargo check above.)"
rm -f /tmp/heartbit_ghost_voice_surface_check.rs
```

The above is illustrative — every public type is reachable because `voice/mod.rs` re-exports them and the workspace `cargo check` already covers it. No actual file changes here; this step is just confirmation.

- [ ] **Step 4: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.2a
```

Expected: 5 commits — spec doc + 4 task commits (Task 1, 2, 3, 4). No commit for Task 5 (verification only).

- [ ] **Step 5: No commit for this task**

Task 5 is verification. The branch is now ready for final review + merge.

---

## Acceptance criteria

P1.2a is done when (per spec §9):

- All public types compile cleanly under `cargo check -p heartbit-ghost`
- `cargo fmt -- --check && cargo clippy -p heartbit-ghost -- -D warnings && cargo test -p heartbit-ghost --lib voice` all green
- ~30 tests pass; coverage spans round-trip, validation negative paths, version handling
- The §2.2 example parses verbatim through `StyleProfile::from_toml`
- `heartbit_ghost::voice::{StyleProfile, BlendRecipe, BlendEntry, PartialStyleProfile, Formatting, VoiceError, /* 11 enums */}` are reachable as public surface

## Out of scope (re-stated)

- Actual blend algorithm (P1.2d)
- Corpus storage / loading (P1.2b)
- LLM-based style extractor (P1.2c)
- CLI bodies (P1.2e — `corpus add/list/remove`, `profile rebuild/diff`)
- Translation of `StyleProfile` to writer-prompt strings (P1.2c)
- Profile diff format (P1.2e)
- Migrations between profile versions (deferred until first breaking change)

## Reference

- Spec: `docs/superpowers/specs/2026-05-07-heartbit-ghost-p1.2a-style-profile-schema-design.md`
- Umbrella heartbit-ghost spec §2.2: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- Foundation: `docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md` (released as `2026.507.3` on crates.io)
- P1.0 plan: `docs/superpowers/plans/2026-05-07-heartbit-ghost-p1.0-scaffolding.md`
- P1.1 plan: `docs/superpowers/plans/2026-05-07-heartbit-ghost-p1.1-x-tools.md`
