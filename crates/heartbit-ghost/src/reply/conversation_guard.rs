//! Conversation-depth guard — caps reply count per X conversation.
//! Catches third-party-joined threads where each new participant
//! could otherwise drag heartbit into 5+ message exchanges.
//!
//! See P1.7 spec §6.

use super::storage::{MentionStore, StoreError};
use super::{Mention, spam_guard::SkipReason};

/// Conversation-depth guard.
pub struct ConversationDepthGuard {
    cap: usize,
}

impl ConversationDepthGuard {
    /// Construct with `cap`. `cap = 0` disables the guard.
    pub fn new(cap: usize) -> Self {
        Self { cap }
    }

    /// Returns `Some(SkipReason::ConversationDepthExceeded)` when the
    /// conversation already has ≥ `cap` replies from us. `None` when
    /// the cap is 0, the mention has no `conversation_id`, or the
    /// count is below cap.
    pub async fn should_skip(
        &self,
        mention: &Mention,
        store: &dyn MentionStore,
    ) -> Result<Option<SkipReason>, StoreError> {
        if self.cap == 0 {
            return Ok(None);
        }
        let Some(conversation_id) = mention.conversation_id.as_deref() else {
            return Ok(None);
        };
        let count = store.replies_in_conversation(conversation_id).await?;
        if count >= self.cap {
            Ok(Some(SkipReason::ConversationDepthExceeded {
                conversation_id: conversation_id.to_string(),
                count,
                cap: self.cap,
            }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reply::InMemoryMentionStore;
    use chrono::Utc;

    fn fixture_mention(conversation_id: Option<&str>) -> Mention {
        Mention {
            id: "m1".into(),
            text: "hi".into(),
            author_id: "1".into(),
            author_handle: "x".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: None,
            conversation_id: conversation_id.map(String::from),
        }
    }

    #[tokio::test]
    async fn skips_at_cap() {
        let store = InMemoryMentionStore::new();
        store.record_reply_in_conversation("c1").await.unwrap();
        store.record_reply_in_conversation("c1").await.unwrap();
        let guard = ConversationDepthGuard::new(2);
        let m = fixture_mention(Some("c1"));
        let result = guard.should_skip(&m, &store).await.unwrap();
        match result {
            Some(SkipReason::ConversationDepthExceeded {
                conversation_id,
                count,
                cap,
            }) => {
                assert_eq!(conversation_id, "c1");
                assert_eq!(count, 2);
                assert_eq!(cap, 2);
            }
            other => panic!("expected ConversationDepthExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn proceeds_below_cap() {
        let store = InMemoryMentionStore::new();
        store.record_reply_in_conversation("c1").await.unwrap();
        let guard = ConversationDepthGuard::new(2);
        let m = fixture_mention(Some("c1"));
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn proceeds_when_conversation_id_absent() {
        let store = InMemoryMentionStore::new();
        let guard = ConversationDepthGuard::new(2);
        let m = fixture_mention(None);
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn cap_zero_disables() {
        let store = InMemoryMentionStore::new();
        for _ in 0..10 {
            store.record_reply_in_conversation("c1").await.unwrap();
        }
        let guard = ConversationDepthGuard::new(0);
        let m = fixture_mention(Some("c1"));
        assert!(guard.should_skip(&m, &store).await.unwrap().is_none());
    }
}
