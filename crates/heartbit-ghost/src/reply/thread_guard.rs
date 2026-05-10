//! Thread-depth guard — skips a mention when its parent tweet is in
//! our replied set (i.e., this is a continuation of a thread we already
//! engaged with). Catches the dominant AI-to-AI loop shape: another
//! bot replies to our reply, threading on our original tweet.
//!
//! See P1.7 spec §4.

use super::storage::{MentionStore, StoreError};
use super::{Mention, spam_guard::SkipReason};

/// Thread-depth guard. Async because it consults the [`MentionStore`].
pub struct ThreadDepthGuard {
    enabled: bool,
}

impl ThreadDepthGuard {
    /// Construct an enabled guard.
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Construct a guard with an explicit enable flag (use `false`
    /// to disable; the guard then always returns `Ok(None)`).
    pub fn with_enabled(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Returns `Some(SkipReason::OwnThreadContinuation)` when the
    /// mention's parent is in our replied set; `None` to proceed.
    /// Errors propagate from the store.
    pub async fn should_skip(
        &self,
        mention: &Mention,
        store: &dyn MentionStore,
    ) -> Result<Option<SkipReason>, StoreError> {
        if !self.enabled {
            return Ok(None);
        }
        let Some(parent_id) = mention.in_reply_to_tweet_id.as_deref() else {
            return Ok(None);
        };
        if store.was_replied(parent_id).await? {
            Ok(Some(SkipReason::OwnThreadContinuation))
        } else {
            Ok(None)
        }
    }
}

impl Default for ThreadDepthGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reply::InMemoryMentionStore;
    use chrono::Utc;

    fn fixture_mention(in_reply_to: Option<&str>) -> Mention {
        Mention {
            id: "m1".into(),
            text: "hi".into(),
            author_id: "1".into(),
            author_handle: "x".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: in_reply_to.map(String::from),
            conversation_id: None,
        }
    }

    #[tokio::test]
    async fn skips_when_parent_in_replied_set() {
        let store = InMemoryMentionStore::new();
        store.mark_replied("parent_id").await.unwrap();
        let guard = ThreadDepthGuard::new();
        let m = fixture_mention(Some("parent_id"));
        assert_eq!(
            guard.should_skip(&m, &store).await.unwrap(),
            Some(SkipReason::OwnThreadContinuation)
        );
    }

    #[tokio::test]
    async fn proceeds_when_parent_not_in_replied_set() {
        let store = InMemoryMentionStore::new();
        let guard = ThreadDepthGuard::new();
        let m = fixture_mention(Some("unknown_parent"));
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn proceeds_when_no_parent_id() {
        let store = InMemoryMentionStore::new();
        store.mark_replied("anything").await.unwrap();
        let guard = ThreadDepthGuard::new();
        let m = fixture_mention(None);
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn disabled_guard_always_returns_none() {
        let store = InMemoryMentionStore::new();
        store.mark_replied("parent_id").await.unwrap();
        let guard = ThreadDepthGuard::with_enabled(false);
        let m = fixture_mention(Some("parent_id"));
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }
}
