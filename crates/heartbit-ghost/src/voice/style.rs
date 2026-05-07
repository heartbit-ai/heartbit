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
