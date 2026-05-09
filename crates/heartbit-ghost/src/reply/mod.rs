//! Reply pipeline — drafts a single short reply to a specific mention,
//! routes to Telegram for review, posts via `twitter_reply` on user pick.
//!
//! See spec §2/§5 for the architecture; this file holds the value types
//! and the public surface. The runtime lives in [`run_reply_pipeline`]
//! once Task 5 lands it.

use chrono::{DateTime, Utc};

pub mod prompts;

/// A mention of the operator's account fetched from `twitter_mentions`.
#[derive(Debug, Clone)]
pub struct Mention {
    /// X tweet ID of the mention itself.
    pub id: String,
    /// Plain text of the mention.
    pub text: String,
    /// X user ID of the mentioner.
    pub author_id: String,
    /// Public handle of the mentioner (sans `@`).
    pub author_handle: String,
    /// When the mention was posted.
    pub posted_at: DateTime<Utc>,
    /// Tweet ID this mention is replying to (None when it's a top-level
    /// `@operator …` mention rather than a reply on an operator's tweet).
    pub in_reply_to_tweet_id: Option<String>,
}

/// A small snapshot of a tweet (text + timing). Used as a parent-tweet
/// context for the reply researcher.
#[derive(Debug, Clone)]
pub struct TweetSnapshot {
    /// X tweet ID.
    pub id: String,
    /// Plain text of the tweet.
    pub text: String,
    /// When the tweet was posted.
    pub posted_at: DateTime<Utc>,
}

/// Tone-calibration context about the mentioner. None of these are
/// strictly required; the writer degrades gracefully if missing.
#[derive(Debug, Clone, Default)]
pub struct MentionerContext {
    /// Public handle of the mentioner (sans `@`).
    pub handle: String,
    /// Bio text from the mentioner's profile, if available.
    pub bio: Option<String>,
    /// Up to 3 recent tweets, abridged.
    pub recent_tweets: Vec<TweetSnapshot>,
    /// Follower count of the mentioner, if available.
    pub follower_count: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mention_round_trips_through_clone() {
        let m = Mention {
            id: "1".into(),
            text: "hi".into(),
            author_id: "12".into(),
            author_handle: "alice".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: Some("99".into()),
        };
        let copy = m.clone();
        assert_eq!(copy.id, m.id);
        assert_eq!(copy.text, m.text);
        assert_eq!(copy.in_reply_to_tweet_id, m.in_reply_to_tweet_id);
    }

    #[test]
    fn mentioner_context_default_has_empty_handle_and_no_recent_tweets() {
        let m = MentionerContext::default();
        assert!(m.handle.is_empty());
        assert!(m.bio.is_none());
        assert!(m.recent_tweets.is_empty());
        assert!(m.follower_count.is_none());
    }
}
