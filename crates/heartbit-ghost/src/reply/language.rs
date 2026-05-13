//! Language detection for reply / quote-tweet inputs.
//!
//! When the operator's account is mentioned in French (or any other
//! language), the daemon's reply should be in the same language. This
//! module wraps `whatlang` to provide a stable detection API with
//! sensible defaults for short / ambiguous inputs.
//!
//! The voice profile stays English-described — voice_traits like
//! "casual_authority" are concepts, not strings. Modern LLMs handle the
//! register-transfer to other languages without losing the persona's
//! identity. Only the *target language* is explicitly instructed.
//!
//! Reused by the future quote-tweet path with no changes.

use whatlang::{Detector, Lang};

/// Detected language for a reply target.
///
/// `code` is the ISO 639-3 three-letter code that `whatlang` produces
/// natively ("eng", "fra", "deu", "spa", ...). The `english_name` field
/// is the language name in English, used in the LLM prompt because LLMs
/// respond more reliably to "Respond in French." than to "Respond in fra."
/// (the latter sometimes triggers code-mode output).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplyLanguage {
    /// ISO 639-3 three-letter code. Always lowercase.
    pub code: String,
    /// Human-readable name in English (e.g. "French", "Japanese").
    /// What goes into the LLM prompt.
    pub english_name: String,
}

impl ReplyLanguage {
    /// English fallback used for short / ambiguous / unreliable inputs.
    /// We default to English (vs. failing) because mirroring the wrong
    /// language to an English speaker reads as broken; English back to
    /// a French speaker reads as a translation choice. The asymmetric
    /// risk pushes us toward an English default.
    pub fn english() -> Self {
        Self {
            code: "eng".to_string(),
            english_name: "English".to_string(),
        }
    }

    /// Build from a `whatlang::Lang` enum value.
    fn from_whatlang(lang: Lang) -> Self {
        Self {
            code: lang.code().to_string(),
            english_name: lang.eng_name().to_string(),
        }
    }
}

/// Minimum number of characters before we trust the detector. Whatlang's
/// own docs note that under ~20 chars the confidence is unreliable for
/// most languages. Short tweets ("yes", "lol", "+1") have no signal —
/// defaulting to English avoids cargo-culting a wrong language.
const MIN_TEXT_LEN_FOR_DETECTION: usize = 20;

/// Minimum confidence to accept a detection. Whatlang returns a 0.0–1.0
/// score; values < 0.6 are typically code-switched or short. We default
/// to English on low confidence rather than risking a confident-but-wrong
/// language tag.
const MIN_CONFIDENCE: f64 = 0.6;

/// Detect the language of `text` for reply purposes.
///
/// Returns `ReplyLanguage::english()` when the text is too short, the
/// detector returns nothing, or the confidence is below `MIN_CONFIDENCE`.
pub fn detect_reply_language(text: &str) -> ReplyLanguage {
    let trimmed = text.trim();
    if trimmed.chars().count() < MIN_TEXT_LEN_FOR_DETECTION {
        return ReplyLanguage::english();
    }

    let detector = Detector::new();
    let info = match detector.detect(trimmed) {
        Some(info) => info,
        None => return ReplyLanguage::english(),
    };
    if info.confidence() < MIN_CONFIDENCE {
        return ReplyLanguage::english();
    }
    ReplyLanguage::from_whatlang(info.lang())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_french_from_substantial_text() {
        let text = "Je trouve que les agents LLM sont plus efficaces \
                    quand on limite leur boucle de récursion à trois itérations.";
        let lang = detect_reply_language(text);
        assert_eq!(lang.code, "fra", "expected French, got {lang:?}");
        assert_eq!(lang.english_name, "French");
    }

    #[test]
    fn detects_german_from_substantial_text() {
        let text = "Das ist ein gutes Beispiel für die Verwendung \
                    von Schaltkreis-Brechern in Agenten-Schleifen.";
        let lang = detect_reply_language(text);
        assert_eq!(lang.code, "deu");
        assert_eq!(lang.english_name, "German");
    }

    #[test]
    fn detects_spanish_from_substantial_text() {
        let text = "Los agentes de inteligencia artificial necesitan límites \
                    estrictos para evitar bucles infinitos en producción.";
        let lang = detect_reply_language(text);
        assert_eq!(lang.code, "spa");
    }

    #[test]
    fn detects_english_from_substantial_text() {
        let text = "Agent loops without guardrails are credit cards on a while loop.";
        let lang = detect_reply_language(text);
        assert_eq!(lang.code, "eng");
    }

    #[test]
    fn short_text_defaults_to_english() {
        // Below MIN_TEXT_LEN_FOR_DETECTION (20 chars) — even if French,
        // we can't trust the detector and default to English.
        let text = "merci bcp";
        let lang = detect_reply_language(text);
        assert_eq!(
            lang,
            ReplyLanguage::english(),
            "short text must default to English regardless of content"
        );
    }

    #[test]
    fn empty_text_defaults_to_english() {
        assert_eq!(detect_reply_language(""), ReplyLanguage::english());
        assert_eq!(detect_reply_language("   "), ReplyLanguage::english());
    }

    #[test]
    fn english_helper_returns_expected_shape() {
        let en = ReplyLanguage::english();
        assert_eq!(en.code, "eng");
        assert_eq!(en.english_name, "English");
    }

    #[test]
    fn long_substantive_text_round_trip() {
        // Confirm the detector picks up the dominant language even when
        // there's some English brand noise mixed in (e.g. "LLM", "API").
        // This is the realistic mention shape — non-English speakers
        // often code-mix English technical terms.
        let text = "Vraiment intéressant ton article sur les LLMs. \
                    J'ai du mal avec l'API quand je fais des appels parallèles. \
                    Une recommandation sur la gestion des erreurs ?";
        let lang = detect_reply_language(text);
        assert_eq!(
            lang.code, "fra",
            "dominant French should survive English tech terms"
        );
    }
}
