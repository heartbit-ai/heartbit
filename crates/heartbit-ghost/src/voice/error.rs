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
