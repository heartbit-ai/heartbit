//! User-message builders for each reply-pipeline stage. Pure string
//! composition — same shape as `pipeline/prompts.rs`.

use super::language::ReplyLanguage;
use super::{Mention, MentionerContext, TweetSnapshot};

/// Build the mini-researcher's user message: the parent tweet (if any),
/// the mention itself, and abridged context about the mentioner.
pub(crate) fn build_reply_research_user_message(
    mention: &Mention,
    parent: Option<&TweetSnapshot>,
    mentioner: Option<&MentionerContext>,
) -> String {
    let mut out = String::new();
    if let Some(p) = parent {
        out.push_str("PARENT TWEET (yours, posted ");
        out.push_str(&p.posted_at.to_rfc3339());
        out.push_str("):\n> ");
        out.push_str(&p.text);
        out.push_str("\n\n");
    }
    out.push_str(&format!(
        "THEIR REPLY (from @{}, posted {}):\n> {}\n\n",
        mention.author_handle,
        mention.posted_at.to_rfc3339(),
        mention.text,
    ));
    if let Some(m) = mentioner {
        out.push_str("MENTIONER CONTEXT\n");
        if let Some(bio) = &m.bio {
            out.push_str(&format!("- bio: {bio}\n"));
        }
        if let Some(fc) = m.follower_count {
            out.push_str(&format!("- followers: {fc}\n"));
        }
        if !m.recent_tweets.is_empty() {
            out.push_str("- recent tweets:\n");
            for t in m.recent_tweets.iter().take(3) {
                let abridged: String = t.text.chars().take(100).collect();
                out.push_str(&format!("    > {abridged}\n"));
            }
        }
        out.push('\n');
    }
    out.push_str(
        "Identify the SPECIFIC point to engage with in 1-3 sentences. \
         Do NOT compose the reply — the writer composes it next.\n",
    );
    out
}

/// Build the writer's user message — the digest from the researcher,
/// then voice guidelines, optional mode_addendum, the target language,
/// and a clear final instruction.
///
/// `language` is the detected language of the mention text (see
/// [`super::language::detect_reply_language`]). When the language is
/// English the line is still emitted — it's cheap insurance against the
/// LLM "translating" an English mention into another language.
pub(crate) fn build_reply_writer_user_message(
    digest: &str,
    voice_guidelines: &str,
    mode_addendum: Option<&str>,
    language: &ReplyLanguage,
) -> String {
    let mut out = String::new();
    out.push_str("Research digest (the specific point to engage with):\n");
    out.push_str(digest);
    out.push_str("\n\n");
    out.push_str(voice_guidelines);
    out.push('\n');
    if let Some(addendum) = mode_addendum {
        out.push('\n');
        out.push_str(addendum);
        out.push('\n');
    }
    // Language instruction sits AFTER voice guidelines + addendum so the
    // model treats it as the most-recent / most-specific constraint. The
    // voice profile is English-described (voice_traits, ai_tells_to_avoid)
    // but the LLM cross-translates the register to the target language —
    // this is well within modern LLM capabilities and avoids per-language
    // voice profiles for now.
    out.push_str(&format!(
        "\nRESPOND IN {}. Mirror the mention's language exactly — do not switch to English just because the voice guidelines are English-described.\n",
        language.english_name
    ));
    out.push_str("\nCompose ONE reply (≤280 chars). Output the reply text only.\n");
    out
}

/// Build the style critic's user message for a reply candidate.
pub(crate) fn build_reply_critic_user_message(draft: &str, voice_guidelines: &str) -> String {
    format!(
        "Reply draft to evaluate:\n{draft}\n\n{voice_guidelines}\n\
         Score the draft and return your verdict as JSON per the schema.\n"
    )
}

/// Build the fact-check's user message for a reply.
pub(crate) fn build_reply_fact_user_message(draft: &str, digest: &str) -> String {
    format!(
        "Reply draft to verify:\n{draft}\n\nResearch digest (only source of truth):\n{digest}\n\
         Verify and return your verdict as JSON per the schema.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn fixture_mention() -> Mention {
        Mention {
            id: "abc".into(),
            text: "how does this compare to rig-rs?".into(),
            author_id: "777".into(),
            author_handle: "grumpy_dev".into(),
            posted_at: Utc.with_ymd_and_hms(2026, 5, 8, 11, 2, 0).unwrap(),
            in_reply_to_tweet_id: Some("parent_id".into()),
            conversation_id: None,
        }
    }

    #[test]
    fn research_message_quotes_mention_and_parent() {
        let m = fixture_mention();
        let p = TweetSnapshot {
            id: "parent_id".into(),
            text: "Implement two methods, get a fully wired tool.".into(),
            posted_at: Utc.with_ymd_and_hms(2026, 5, 8, 10, 14, 0).unwrap(),
        };
        let s = build_reply_research_user_message(&m, Some(&p), None);
        assert!(s.contains("Implement two methods"));
        assert!(s.contains("how does this compare to rig-rs"));
        assert!(s.contains("@grumpy_dev"));
        assert!(s.contains("Identify the SPECIFIC point"));
    }

    #[test]
    fn writer_message_appends_addendum_after_voice_guidelines() {
        let s = build_reply_writer_user_message(
            "engage with: rig-rs comparison",
            "VOICE GUIDELINES",
            Some("EVANGELISM MODE — fixture"),
            &ReplyLanguage::english(),
        );
        let voice_pos = s.find("VOICE GUIDELINES").expect("voice present");
        let add_pos = s
            .find("EVANGELISM MODE — fixture")
            .expect("addendum present");
        assert!(voice_pos < add_pos, "addendum must follow voice");
        assert!(s.contains("≤280 chars"));
    }

    #[test]
    fn writer_message_omits_addendum_block_when_none() {
        let s = build_reply_writer_user_message("digest", "VOICE", None, &ReplyLanguage::english());
        assert!(!s.contains("EVANGELISM"));
        assert!(s.contains("Compose ONE reply"));
    }

    #[test]
    fn writer_message_injects_target_language() {
        let french = ReplyLanguage {
            code: "fra".to_string(),
            english_name: "French".to_string(),
        };
        let s = build_reply_writer_user_message("digest", "VOICE", None, &french);
        assert!(
            s.contains("RESPOND IN French."),
            "must include explicit language directive; got: {s}"
        );
        assert!(
            s.contains("do not switch to English"),
            "must remind the model not to fall back to English; got: {s}"
        );
    }

    #[test]
    fn writer_message_language_directive_follows_addendum() {
        // Layering matters: voice → addendum → language → final compose
        // instruction. The most-specific constraint (language) must be
        // the last thing the model sees before the action verb.
        let french = ReplyLanguage {
            code: "fra".to_string(),
            english_name: "French".to_string(),
        };
        let s =
            build_reply_writer_user_message("digest", "VOICE", Some("ADDENDUM_MARKER"), &french);
        let addendum_pos = s.find("ADDENDUM_MARKER").expect("addendum present");
        let lang_pos = s
            .find("RESPOND IN French.")
            .expect("language directive present");
        let compose_pos = s.find("Compose ONE reply").expect("compose present");
        assert!(addendum_pos < lang_pos, "language must follow addendum");
        assert!(lang_pos < compose_pos, "compose must follow language");
    }
}
