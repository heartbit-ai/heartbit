//! Voice modeling — style profiles, blend recipes, partial overrides.
//!
//! P1.2a ships pure data types only. Future sub-phases:
//! - P1.2b: corpus storage (memory namespace, JSONL load)
//! - P1.2c: LLM-based style extractor (corpus → StyleProfile)
//! - P1.2d: blend algorithm (BlendRecipe + N profiles → 1 profile)
//! - P1.2e: CLI bodies for `corpus add/list/remove`, `profile rebuild/diff`

pub mod error;
pub mod style;
// pub mod blend;   // uncommented in Task 4

pub use error::VoiceError;
pub use style::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
