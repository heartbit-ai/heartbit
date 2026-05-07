//! Voice modeling — style profiles, blend recipes, partial overrides.
//!
//! P1.2a ships pure data types only. Future sub-phases:
//! - P1.2b: corpus storage (memory namespace, JSONL load)
//! - P1.2c: LLM-based style extractor (corpus → StyleProfile)
//! - P1.2d: blend algorithm (BlendRecipe + N profiles → 1 profile)
//! - P1.2e: CLI bodies for `corpus add/list/remove`, `profile rebuild/diff`

pub mod error;
// pub mod style;   // uncommented in Task 2
// pub mod blend;   // uncommented in Task 4

pub use error::VoiceError;
// re-exports from style and blend uncommented in Tasks 2 and 4
