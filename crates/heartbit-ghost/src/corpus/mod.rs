//! Corpus storage — file-backed JSONL persistence for per-writer post
//! collections. Sibling of [`crate::voice`]: voice owns the schema; corpus
//! owns the inputs that the LLM extractor (P1.2c) turns into a profile.
//!
//! On-disk layout (created lazily on first write):
//!
//! ```text
//! ~/.heartbit/ghost/corpora/
//! ├── karpathy.jsonl      # one writer, one file
//! ├── eladgil.jsonl
//! └── swyx.jsonl
//! ```

pub mod error;
// pub mod entry;   // uncommented in Task 2
// pub mod store;   // uncommented in Task 3

pub use error::CorpusError;
