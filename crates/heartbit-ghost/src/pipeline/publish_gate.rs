//! Deterministic pre-publish guard. Char count + thread length only.
//! LLM-based content guardrails (PII / brand safety / etc.) are P1.4.

use thiserror::Error;

use crate::voice::StyleProfile;

/// Errors raised by [`check_publish_gate`].
#[derive(Debug, Error)]
pub enum PublishGateError {
    /// One of the tweets exceeds 280 characters.
    #[error("tweet {index} exceeds 280 chars (got {len}); offending text: {text:?}")]
    TweetTooLong {
        /// 0-based tweet index in the thread.
        index: usize,
        /// Character count.
        len: usize,
        /// The offending tweet text.
        text: String,
    },

    /// The thread has more tweets than `profile.thread_max_length`.
    #[error("thread length {actual} exceeds profile.thread_max_length {max}")]
    ThreadTooLong {
        /// Actual tweet count.
        actual: u32,
        /// Profile-imposed maximum.
        max: u32,
    },

    /// The draft is empty or contains only whitespace.
    #[error("draft is empty")]
    EmptyDraft,
}

/// Validate `draft` against the persona's `profile`. Splits the draft on
/// `\n\n` boundaries to identify thread tweets.
pub fn check_publish_gate(draft: &str, profile: &StyleProfile) -> Result<(), PublishGateError> {
    let tweets: Vec<&str> = draft
        .split("\n\n")
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();

    if tweets.is_empty() {
        return Err(PublishGateError::EmptyDraft);
    }

    let max = profile.thread_max_length;
    let actual = tweets.len() as u32;
    if actual > max {
        return Err(PublishGateError::ThreadTooLong { actual, max });
    }

    for (i, tweet) in tweets.iter().enumerate() {
        let len = tweet.chars().count();
        if len > 280 {
            return Err(PublishGateError::TweetTooLong {
                index: i,
                len,
                text: (*tweet).to_string(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::{
        EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
        OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
        ThreadRhythm,
    };

    fn profile_with_max(thread_max_length: u32) -> StyleProfile {
        StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![OpeningPattern::ClaimFirst],
            opening_pattern_weights: vec![1.0],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::RarePunchlineOnly,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec![],
            ai_tells_to_avoid: vec![],
            thread_rhythm: ThreadRhythm::Linear,
            thread_max_length,
            thread_opener_must_hook: false,
            topical_obsessions: vec![],
            topical_avoidances: vec![],
        }
    }

    #[test]
    fn single_tweet_under_280_passes() {
        let p = profile_with_max(10);
        check_publish_gate("a short post", &p).unwrap();
    }

    #[test]
    fn single_tweet_over_280_rejected() {
        let p = profile_with_max(10);
        let long = "a".repeat(281);
        let err = check_publish_gate(&long, &p).unwrap_err();
        match err {
            PublishGateError::TweetTooLong { index, len, .. } => {
                assert_eq!(index, 0);
                assert_eq!(len, 281);
            }
            other => panic!("expected TweetTooLong, got {other:?}"),
        }
    }

    #[test]
    fn thread_within_limit_passes() {
        let p = profile_with_max(3);
        let thread = "first tweet\n\nsecond tweet\n\nthird tweet";
        check_publish_gate(thread, &p).unwrap();
    }

    #[test]
    fn thread_exceeding_limit_rejected() {
        let p = profile_with_max(2);
        let thread = "one\n\ntwo\n\nthree";
        let err = check_publish_gate(thread, &p).unwrap_err();
        match err {
            PublishGateError::ThreadTooLong { actual, max } => {
                assert_eq!(actual, 3);
                assert_eq!(max, 2);
            }
            other => panic!("expected ThreadTooLong, got {other:?}"),
        }
    }

    #[test]
    fn thread_with_individual_tweet_too_long_rejected() {
        let p = profile_with_max(5);
        let big = "x".repeat(290);
        let thread = format!("ok first\n\n{big}\n\nthird");
        let err = check_publish_gate(&thread, &p).unwrap_err();
        match err {
            PublishGateError::TweetTooLong { index, len, .. } => {
                assert_eq!(index, 1);
                assert_eq!(len, 290);
            }
            other => panic!("expected TweetTooLong, got {other:?}"),
        }
    }

    #[test]
    fn empty_or_whitespace_draft_rejected() {
        let p = profile_with_max(10);
        assert!(matches!(
            check_publish_gate("", &p).unwrap_err(),
            PublishGateError::EmptyDraft
        ));
        assert!(matches!(
            check_publish_gate("   \n\n   \n\n", &p).unwrap_err(),
            PublishGateError::EmptyDraft
        ));
    }
}
