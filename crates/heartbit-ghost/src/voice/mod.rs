//! Voice modeling — style profiles, blend recipes, partial overrides.
//!
//! P1.2a ships pure data types only. Future sub-phases:
//! - P1.2b: corpus storage (memory namespace, JSONL load)
//! - P1.2c: LLM-based style extractor (corpus → StyleProfile)
//! - P1.2d: blend algorithm (BlendRecipe + N profiles → 1 profile)
//! - P1.2e: CLI bodies for `corpus add/list/remove`, `profile rebuild/diff`

pub mod blend;
pub mod error;
pub mod extractor;
pub mod style;

pub use blend::{BlendEntry, BlendRecipe, PartialStyleProfile};
pub use error::VoiceError;
pub use extractor::{ExtractError, StyleExtractor, StyleExtractorBuilder, default_system_prompt};
pub use style::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};
