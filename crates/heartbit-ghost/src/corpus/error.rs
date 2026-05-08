//! Corpus storage errors.

use thiserror::Error;

/// Errors raised by the [`crate::corpus`] subsystem.
#[derive(Debug, Error)]
pub enum CorpusError {
    /// Filesystem failure (open / read / write / rename).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parse failure on a specific JSONL line. The line number is 1-based.
    #[error("json on line {line}: {source}")]
    Json {
        /// 1-based line number where the parse failed.
        line: usize,
        /// Underlying parser error.
        #[source]
        source: serde_json::Error,
    },

    /// The supplied writer handle is invalid — empty, contains a path
    /// separator, contains `..`, or is whitespace-only.
    #[error("invalid writer name '{0}': must be non-empty, no '/', '\\', or '..'")]
    InvalidWriter(String),

    /// Generic data or environment validation failure.
    #[error("validation: {0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_renders_with_io_prefix() {
        let inner = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e = CorpusError::Io(inner);
        let s = format!("{e}");
        assert!(s.starts_with("io: "), "got: {s}");
        assert!(s.contains("missing"));
    }

    #[test]
    fn json_error_includes_line_number_in_display() {
        let bad = serde_json::from_str::<serde_json::Value>("not-json").unwrap_err();
        let e = CorpusError::Json {
            line: 47,
            source: bad,
        };
        let s = format!("{e}");
        assert!(s.contains("line 47"), "got: {s}");
    }

    #[test]
    fn invalid_writer_rendering_is_actionable() {
        let e = CorpusError::InvalidWriter("../etc/passwd".to_string());
        let s = format!("{e}");
        assert!(s.contains("../etc/passwd"));
        assert!(s.contains("'/'"));
        assert!(s.contains(".."));
    }
}
