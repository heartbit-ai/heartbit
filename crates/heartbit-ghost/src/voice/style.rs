//! Style profile schema — closed-vocabulary enums, formatting struct, and
//! the top-level `StyleProfile` type (added in Task 3).

use serde::{Deserialize, Serialize};

use crate::voice::error::VoiceError;

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

    /// The §2.2 example from the umbrella heartbit-ghost spec.
    /// This is the load-bearing fixture: if it stops parsing, the spec drifted.
    ///
    /// Note: TOML grammar requires that all top-level keys appear before any
    /// `[table]` header — once a header opens, every subsequent bare key is
    /// scoped inside that table. The plan's example placed `[formatting]` in
    /// the middle with more top-level keys after it; we move `[formatting]`
    /// to the end of the document. All keys/values are preserved verbatim.
    const SPEC_EXAMPLE: &str = r#"
version = 1

sentence_length_target = "short"
sentence_length_distribution = [40, 30, 20, 10]
fragment_frequency = "common"

opening_patterns = ["claim_first", "number_first", "scene_first", "question_first"]
opening_pattern_weights = [0.4, 0.2, 0.2, 0.2]

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

[formatting]
lowercase = true
periods = "optional"
em_dashes = "forbidden"
quotation_marks = "double"
line_breaks = "single"
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
        assert!(profile.formatting.lowercase);
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
}
