//! User-message builders for each quote-pipeline stage. Pure string
//! composition — same shape as `reply/prompts.rs`.

use super::sources::QuoteCandidate;
use crate::reply::language::ReplyLanguage;

/// Build the mini-researcher's user message for a quote-tweet target.
pub(crate) fn build_quote_research_user_message(source: &QuoteCandidate) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "SOURCE TWEET (from @{}, posted {}):\n> {}\n\n",
        source.author_handle,
        source.posted_at.to_rfc3339(),
        source.text,
    ));
    out.push_str(
        "Identify the SPECIFIC claim or framing to engage with in 1-3 sentences. \
         Surface any quantitative claims (numbers, percentages, dates, citations) and \
         whether they are supported by reputable sources. Do NOT compose the quote-tweet — \
         the quote_writer composes it next.\n",
    );
    out
}

/// Build the quote_writer's user message: digest, source tweet, voice
/// guidelines, target language.
pub(crate) fn build_quote_writer_user_message(
    digest: &str,
    source: &QuoteCandidate,
    voice_guidelines: &str,
    language: &ReplyLanguage,
) -> String {
    let mut out = String::new();
    out.push_str("Research digest (claims to verify + framing to engage with):\n");
    out.push_str(digest);
    out.push_str("\n\n");
    out.push_str(&format!(
        "QUOTED TWEET (the post you are quoting; from @{}):\n> {}\n\n",
        source.author_handle, source.text,
    ));
    out.push_str(voice_guidelines);
    out.push('\n');
    out.push_str(&format!(
        "\nRESPOND IN {}. Mirror the quoted tweet's language exactly — do not switch to English just because the voice guidelines are English-described.\n",
        language.english_name
    ));
    out.push_str("\nCompose ONE quote-tweet comment (≤280 chars). Output the comment text only.\n");
    out
}

/// Build the style critic's user message for a quote-tweet candidate.
pub(crate) fn build_quote_critic_user_message(draft: &str, voice_guidelines: &str) -> String {
    format!(
        "Quote-tweet comment draft to evaluate:\n{draft}\n\n{voice_guidelines}\n\
         Score the draft and return your verdict as JSON per the schema.\n"
    )
}

/// Build the fact-check's user message for a quote-tweet draft.
pub(crate) fn build_quote_fact_user_message(draft: &str, digest: &str) -> String {
    format!(
        "Quote-tweet comment draft to verify:\n{draft}\n\nResearch digest (only source of truth):\n{digest}\n\
         Verify and return your verdict as JSON per the schema.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn fixture_source() -> QuoteCandidate {
        QuoteCandidate {
            id: "1".into(),
            text: "Microservices solve every problem".into(),
            author_id: "42".into(),
            author_handle: "shipit".into(),
            posted_at: Utc.with_ymd_and_hms(2026, 5, 13, 9, 0, 0).unwrap(),
        }
    }

    #[test]
    fn writer_message_injects_language_directive() {
        let french = ReplyLanguage {
            code: "fra".to_string(),
            english_name: "French".to_string(),
        };
        let s = build_quote_writer_user_message("digest", &fixture_source(), "VOICE", &french);
        assert!(s.contains("RESPOND IN French."));
        assert!(s.contains("QUOTED TWEET"));
        assert!(s.contains("@shipit"));
    }

    #[test]
    fn writer_message_includes_source_tweet_text() {
        let s = build_quote_writer_user_message(
            "digest",
            &fixture_source(),
            "VOICE",
            &ReplyLanguage::english(),
        );
        assert!(s.contains("Microservices solve every problem"));
    }

    #[test]
    fn research_message_quotes_source() {
        let s = build_quote_research_user_message(&fixture_source());
        assert!(s.contains("@shipit"));
        assert!(s.contains("Microservices solve every problem"));
        assert!(s.contains("SPECIFIC claim"));
    }
}
