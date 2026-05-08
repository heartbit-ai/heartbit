//! LLM-based style extractor — turns a [`Corpus`] into a validated
//! [`StyleProfile`] via a single analyst-persona LLM call (umbrella spec
//! §2.3). Sibling of [`crate::voice::style`] (the schema) and
//! [`crate::corpus::Corpus`] (the input).
//!
//! Bodies for [`StyleExtractor`], [`StyleExtractorBuilder`],
//! [`default_system_prompt`], and the pure helpers land in subsequent
//! tasks.

use thiserror::Error;

use crate::voice::error::VoiceError;

/// Errors raised by [`StyleExtractor::extract`] (added in Task 3).
#[derive(Debug, Error)]
pub enum ExtractError {
    /// The corpus had zero entries; nothing to analyze.
    #[error("corpus is empty for writer '{0}'")]
    EmptyCorpus(String),

    /// The underlying LLM call failed (network, auth, rate limit, etc.).
    #[error("llm: {0}")]
    Llm(#[source] heartbit_core::Error),

    /// The LLM call exceeded the configured timeout.
    #[error("llm call timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// The LLM produced no text content (e.g., refusal, empty response).
    #[error("llm produced no text response")]
    EmptyResponse,

    /// JSON parse failure. `raw` carries the offending output for debugging.
    #[error("json parse: {source}")]
    JsonParse {
        /// The underlying serde_json error.
        #[source]
        source: serde_json::Error,
        /// The raw LLM output that failed to parse.
        raw: String,
    },

    /// Parsed cleanly but failed [`crate::voice::StyleProfile::validate`]
    /// (sums, ranges, etc.). `raw` carries the offending output; `inner`
    /// is the validation error.
    #[error("validation: {inner}")]
    Validation {
        /// The underlying validation error from `StyleProfile::validate`.
        #[source]
        inner: VoiceError,
        /// The raw LLM output that produced an invalid profile.
        raw: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_corpus_error_includes_writer_handle() {
        let e = ExtractError::EmptyCorpus("karpathy".to_string());
        let s = format!("{e}");
        assert!(s.contains("karpathy"), "got: {s}");
        assert!(s.starts_with("corpus is empty"), "got: {s}");
    }

    #[test]
    fn jsonparse_error_renders_with_source_message() {
        let bad = serde_json::from_str::<serde_json::Value>("not-json").unwrap_err();
        let e = ExtractError::JsonParse {
            source: bad,
            raw: "not-json".to_string(),
        };
        let s = format!("{e}");
        assert!(s.starts_with("json parse: "), "got: {s}");
    }

    #[test]
    fn validation_error_carries_raw_and_inner() {
        let inner = VoiceError::Validation("weights must sum to 1.0".to_string());
        let raw = r#"{"opening_pattern_weights":[0.5,0.4]}"#.to_string();
        let e = ExtractError::Validation {
            inner,
            raw: raw.clone(),
        };
        let s = format!("{e}");
        assert!(s.contains("validation"), "got: {s}");
        assert!(s.contains("weights must sum to 1.0"), "got: {s}");
        // raw is reachable for debugging
        if let ExtractError::Validation { raw: r, .. } = &e {
            assert_eq!(r, &raw);
        } else {
            panic!("not a Validation variant");
        }
    }
}
