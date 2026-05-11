//! Mention storage for the reply pipeline. Tracks (per persona, per
//! operator user id) the highest mention id we've seen (`since_id`)
//! plus the set of mention ids we've already replied to.

use chrono::{DateTime, Utc};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

/// Errors raised by [`MentionStore`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// I/O failure (file not readable, write failed, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSONL parse failure on replay.
    #[error("parse: {0}")]
    Parse(String),
}

/// Persistent storage for mention-poll state and reply audit trail.
///
/// Trait uses `Pin<Box<dyn Future>>` desugaring (matches
/// [`crate::reply::ReplyReviewDelivery`]) — no `async-trait` dependency.
pub trait MentionStore: Send + Sync {
    /// Read the current `since_id` for `(persona, user_id)`.
    fn since_id_for<'a>(
        &'a self,
        persona: &'a str,
        user_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, StoreError>> + Send + 'a>>;

    /// Bump `since_id` to `new_id` if it's strictly larger (lexicographically).
    fn bump_since_id<'a>(
        &'a self,
        persona: &'a str,
        user_id: &'a str,
        new_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>>;

    /// Mark `mention_id` as having been processed (so the poller doesn't retry).
    fn mark_replied<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>>;

    /// Whether `mention_id` is in the replied set.
    fn was_replied<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>>;

    /// Mark `mention_id` as having reached a `Posted` outcome (a real
    /// X reply was published). Used by the reply handler to dedupe
    /// Kafka redeliveries (default `enable.auto.commit=true` has a ~5s
    /// window where a crashed daemon will re-consume the same
    /// `ReplyDraft` and re-run the pipeline). Distinct from
    /// [`Self::mark_replied`], which is also called on Skip / TimedOut /
    /// GateRejected outcomes purely to prevent re-dispatch on the next
    /// poll.
    fn mark_posted<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>>;

    /// Whether `mention_id` was previously marked as Posted. See
    /// [`Self::mark_posted`] for the redelivery-dedup rationale.
    fn was_posted<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>>;

    /// Number of replies sent to `author_id` since `since`. Used by the
    /// per-author rate limit (default: max 3 / 24h).
    fn replies_to_author_since<'a>(
        &'a self,
        author_id: &'a str,
        since: DateTime<Utc>,
    ) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>>;

    /// Record that we just replied to a mention authored by `author_id`.
    fn record_reply_to_author<'a>(
        &'a self,
        author_id: &'a str,
        ts: DateTime<Utc>,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>>;

    /// Number of replies we've already sent in `conversation_id`.
    /// Used by the conversation-depth guard (P1.7).
    fn replies_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>>;

    /// Record that we just sent a reply in `conversation_id`.
    /// Called from the daemon's ReplyDraft handler after a successful
    /// `Posted` outcome.
    fn record_reply_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>>;
}

// In-memory impl —————————————————————————————————————

/// Volatile in-memory store. Useful for tests and dev runs.
pub struct InMemoryMentionStore {
    inner: tokio::sync::RwLock<InMemoryInner>,
}

#[derive(Default)]
struct InMemoryInner {
    /// (persona, user_id) → since_id
    since: std::collections::HashMap<(String, String), String>,
    /// mention_id → () — set on any pipeline outcome (defensive re-dispatch guard).
    replied: std::collections::HashSet<String>,
    /// mention_id → () — set ONLY when the pipeline returned Posted (real X
    /// reply published). Used by the reply handler to short-circuit Kafka
    /// redeliveries.
    posted: std::collections::HashSet<String>,
    /// (author_id, ts) — append log for rate-limit queries.
    author_replies: Vec<(String, DateTime<Utc>)>,
    /// conversation_id → reply count (P1.7).
    convo_replies: std::collections::HashMap<String, usize>,
}

impl Default for InMemoryMentionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryMentionStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::RwLock::new(InMemoryInner::default()),
        }
    }
}

impl MentionStore for InMemoryMentionStore {
    fn since_id_for<'a>(
        &'a self,
        persona: &'a str,
        user_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            Ok(g.since
                .get(&(persona.to_string(), user_id.to_string()))
                .cloned())
        })
    }

    fn bump_since_id<'a>(
        &'a self,
        persona: &'a str,
        user_id: &'a str,
        new_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let mut g = self.inner.write().await;
            let key = (persona.to_string(), user_id.to_string());
            match g.since.get(&key) {
                Some(prev) if prev.as_str() >= new_id => {} // monotonic — do not regress
                _ => {
                    g.since.insert(key, new_id.to_string());
                }
            }
            Ok(())
        })
    }

    fn mark_replied<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.inner
                .write()
                .await
                .replied
                .insert(mention_id.to_string());
            Ok(())
        })
    }

    fn was_replied<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>> {
        Box::pin(async move { Ok(self.inner.read().await.replied.contains(mention_id)) })
    }

    fn mark_posted<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.inner
                .write()
                .await
                .posted
                .insert(mention_id.to_string());
            Ok(())
        })
    }

    fn was_posted<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>> {
        Box::pin(async move { Ok(self.inner.read().await.posted.contains(mention_id)) })
    }

    fn replies_to_author_since<'a>(
        &'a self,
        author_id: &'a str,
        since: DateTime<Utc>,
    ) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            Ok(g.author_replies
                .iter()
                .filter(|(a, ts)| a == author_id && *ts >= since)
                .count())
        })
    }

    fn record_reply_to_author<'a>(
        &'a self,
        author_id: &'a str,
        ts: DateTime<Utc>,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.inner
                .write()
                .await
                .author_replies
                .push((author_id.to_string(), ts));
            Ok(())
        })
    }

    fn replies_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            Ok(self
                .inner
                .read()
                .await
                .convo_replies
                .get(conversation_id)
                .copied()
                .unwrap_or(0))
        })
    }

    fn record_reply_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            *self
                .inner
                .write()
                .await
                .convo_replies
                .entry(conversation_id.to_string())
                .or_insert(0) += 1;
            Ok(())
        })
    }
}

// JSONL-backed impl ——————————————————————————————————

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum StoreEvent {
    SinceId {
        persona: String,
        user_id: String,
        id: String,
    },
    Replied {
        mention_id: String,
    },
    /// A `Posted` outcome — used by the reply handler to dedupe Kafka
    /// redeliveries. Distinct from `Replied`, which is written on any
    /// terminal outcome.
    Posted {
        mention_id: String,
    },
    AuthorReply {
        author_id: String,
        ts: DateTime<Utc>,
    },
    ConversationReply {
        conversation_id: String,
    },
}

/// JSONL append-only store. Replays the file at [`JsonlMentionStore::open`]
/// time into an in-memory mirror; subsequent writes both append to the file
/// and update the mirror.
pub struct JsonlMentionStore {
    path: PathBuf,
    inner: tokio::sync::RwLock<InMemoryInner>,
}

impl JsonlMentionStore {
    /// Open or create the store at `path`. Replays existing JSONL events
    /// into the in-memory mirror.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let store = Self {
            path,
            inner: tokio::sync::RwLock::new(InMemoryInner::default()),
        };
        store.replay().await?;
        Ok(store)
    }

    async fn replay(&self) -> Result<(), StoreError> {
        let text = match tokio::fs::read_to_string(&self.path).await {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let mut g = self.inner.write().await;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let evt: StoreEvent = serde_json::from_str(line)
                .map_err(|e| StoreError::Parse(format!("line {line:?}: {e}")))?;
            match evt {
                StoreEvent::SinceId {
                    persona,
                    user_id,
                    id,
                } => {
                    g.since.insert((persona, user_id), id);
                }
                StoreEvent::Replied { mention_id } => {
                    g.replied.insert(mention_id);
                }
                StoreEvent::Posted { mention_id } => {
                    g.posted.insert(mention_id);
                }
                StoreEvent::AuthorReply { author_id, ts } => {
                    g.author_replies.push((author_id, ts));
                }
                StoreEvent::ConversationReply { conversation_id } => {
                    *g.convo_replies.entry(conversation_id).or_insert(0) += 1;
                }
            }
        }
        Ok(())
    }

    async fn append(&self, evt: &StoreEvent) -> Result<(), StoreError> {
        use tokio::io::AsyncWriteExt;
        let serialized =
            serde_json::to_string(evt).map_err(|e| StoreError::Parse(format!("{e}")))?;
        let mut f = tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)
            .await?;
        f.write_all(serialized.as_bytes()).await?;
        f.write_all(b"\n").await?;
        f.sync_data().await?;
        Ok(())
    }
}

impl MentionStore for JsonlMentionStore {
    fn since_id_for<'a>(
        &'a self,
        persona: &'a str,
        user_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<String>, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            Ok(self
                .inner
                .read()
                .await
                .since
                .get(&(persona.to_string(), user_id.to_string()))
                .cloned())
        })
    }

    fn bump_since_id<'a>(
        &'a self,
        persona: &'a str,
        user_id: &'a str,
        new_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            // Monotonic check first (read lock, then upgrade to write only if needed).
            {
                let g = self.inner.read().await;
                if let Some(prev) = g.since.get(&(persona.to_string(), user_id.to_string()))
                    && prev.as_str() >= new_id
                {
                    return Ok(());
                }
            }
            self.append(&StoreEvent::SinceId {
                persona: persona.to_string(),
                user_id: user_id.to_string(),
                id: new_id.to_string(),
            })
            .await?;
            self.inner.write().await.since.insert(
                (persona.to_string(), user_id.to_string()),
                new_id.to_string(),
            );
            Ok(())
        })
    }

    fn mark_replied<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.append(&StoreEvent::Replied {
                mention_id: mention_id.to_string(),
            })
            .await?;
            self.inner
                .write()
                .await
                .replied
                .insert(mention_id.to_string());
            Ok(())
        })
    }

    fn was_replied<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>> {
        Box::pin(async move { Ok(self.inner.read().await.replied.contains(mention_id)) })
    }

    fn mark_posted<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.append(&StoreEvent::Posted {
                mention_id: mention_id.to_string(),
            })
            .await?;
            self.inner
                .write()
                .await
                .posted
                .insert(mention_id.to_string());
            Ok(())
        })
    }

    fn was_posted<'a>(
        &'a self,
        mention_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>> {
        Box::pin(async move { Ok(self.inner.read().await.posted.contains(mention_id)) })
    }

    fn replies_to_author_since<'a>(
        &'a self,
        author_id: &'a str,
        since: DateTime<Utc>,
    ) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            Ok(g.author_replies
                .iter()
                .filter(|(a, ts)| a == author_id && *ts >= since)
                .count())
        })
    }

    fn record_reply_to_author<'a>(
        &'a self,
        author_id: &'a str,
        ts: DateTime<Utc>,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.append(&StoreEvent::AuthorReply {
                author_id: author_id.to_string(),
                ts,
            })
            .await?;
            self.inner
                .write()
                .await
                .author_replies
                .push((author_id.to_string(), ts));
            Ok(())
        })
    }

    fn replies_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            Ok(self
                .inner
                .read()
                .await
                .convo_replies
                .get(conversation_id)
                .copied()
                .unwrap_or(0))
        })
    }

    fn record_reply_in_conversation<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.append(&StoreEvent::ConversationReply {
                conversation_id: conversation_id.to_string(),
            })
            .await?;
            *self
                .inner
                .write()
                .await
                .convo_replies
                .entry(conversation_id.to_string())
                .or_insert(0) += 1;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[tokio::test]
    async fn in_memory_since_id_is_monotonic() {
        let store = InMemoryMentionStore::new();
        assert_eq!(store.since_id_for("p", "u").await.unwrap(), None);
        // Use same-length snowflake-style IDs so lexicographic order == numeric order.
        store
            .bump_since_id("p", "u", "1000000000000000100")
            .await
            .unwrap();
        store
            .bump_since_id("p", "u", "1000000000000000050")
            .await
            .unwrap(); // older — should NOT regress
        store
            .bump_since_id("p", "u", "1000000000000000200")
            .await
            .unwrap();
        assert_eq!(
            store.since_id_for("p", "u").await.unwrap().unwrap(),
            "1000000000000000200"
        );
    }

    #[tokio::test]
    async fn in_memory_replied_set_round_trips() {
        let store = InMemoryMentionStore::new();
        assert!(!store.was_replied("m1").await.unwrap());
        store.mark_replied("m1").await.unwrap();
        assert!(store.was_replied("m1").await.unwrap());
    }

    #[tokio::test]
    async fn in_memory_per_author_rate_count_filters_by_since() {
        let store = InMemoryMentionStore::new();
        let now = Utc::now();
        store
            .record_reply_to_author("a1", now - Duration::hours(48))
            .await
            .unwrap();
        store
            .record_reply_to_author("a1", now - Duration::hours(12))
            .await
            .unwrap();
        store
            .record_reply_to_author("a1", now - Duration::hours(1))
            .await
            .unwrap();
        let recent = store
            .replies_to_author_since("a1", now - Duration::hours(24))
            .await
            .unwrap();
        assert_eq!(recent, 2, "should count only the within-24h entries");
    }

    #[tokio::test]
    async fn jsonl_store_round_trips_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mentions.jsonl");
        {
            let s1 = JsonlMentionStore::open(&path).await.unwrap();
            s1.bump_since_id("p", "u", "100").await.unwrap();
            s1.mark_replied("m1").await.unwrap();
            s1.record_reply_to_author("a1", Utc::now()).await.unwrap();
        }
        // Reload from disk.
        let s2 = JsonlMentionStore::open(&path).await.unwrap();
        assert_eq!(s2.since_id_for("p", "u").await.unwrap().unwrap(), "100");
        assert!(s2.was_replied("m1").await.unwrap());
    }

    #[tokio::test]
    async fn jsonl_store_handles_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.jsonl");
        let s = JsonlMentionStore::open(&path).await.unwrap();
        assert_eq!(s.since_id_for("p", "u").await.unwrap(), None);
        assert!(!s.was_replied("m1").await.unwrap());
    }

    #[tokio::test]
    async fn in_memory_record_then_count_in_conversation_round_trip() {
        let store = InMemoryMentionStore::new();
        assert_eq!(store.replies_in_conversation("c1").await.unwrap(), 0);
        store.record_reply_in_conversation("c1").await.unwrap();
        store.record_reply_in_conversation("c1").await.unwrap();
        store.record_reply_in_conversation("c2").await.unwrap();
        assert_eq!(store.replies_in_conversation("c1").await.unwrap(), 2);
        assert_eq!(store.replies_in_conversation("c2").await.unwrap(), 1);
        assert_eq!(
            store.replies_in_conversation("nonexistent").await.unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn jsonl_conversation_replies_persist_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conv.jsonl");
        {
            let s1 = JsonlMentionStore::open(&path).await.unwrap();
            s1.record_reply_in_conversation("c1").await.unwrap();
            s1.record_reply_in_conversation("c1").await.unwrap();
        }
        let s2 = JsonlMentionStore::open(&path).await.unwrap();
        assert_eq!(s2.replies_in_conversation("c1").await.unwrap(), 2);
    }

    #[tokio::test]
    async fn in_memory_posted_flag_is_distinct_from_replied() {
        let store = InMemoryMentionStore::new();
        // Marking replied should NOT mark posted (replied is set on Skip / Timeout / Posted alike).
        store.mark_replied("m1").await.unwrap();
        assert!(store.was_replied("m1").await.unwrap());
        assert!(
            !store.was_posted("m1").await.unwrap(),
            "mark_replied must not implicitly mark posted"
        );

        // Marking posted should NOT mark replied either — orthogonal flags.
        store.mark_posted("m2").await.unwrap();
        assert!(store.was_posted("m2").await.unwrap());
        assert!(
            !store.was_replied("m2").await.unwrap(),
            "mark_posted must not implicitly mark replied"
        );

        // Setting both works.
        store.mark_replied("m3").await.unwrap();
        store.mark_posted("m3").await.unwrap();
        assert!(store.was_replied("m3").await.unwrap());
        assert!(store.was_posted("m3").await.unwrap());
    }

    #[tokio::test]
    async fn jsonl_posted_flag_persists_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("posted.jsonl");
        {
            let s1 = JsonlMentionStore::open(&path).await.unwrap();
            s1.mark_posted("m1").await.unwrap();
            s1.mark_replied("m2").await.unwrap();
        }
        // Reload from disk — `posted` is its own flag, separate from `replied`.
        let s2 = JsonlMentionStore::open(&path).await.unwrap();
        assert!(s2.was_posted("m1").await.unwrap());
        assert!(!s2.was_replied("m1").await.unwrap()); // posted only, not replied
        assert!(s2.was_replied("m2").await.unwrap());
        assert!(!s2.was_posted("m2").await.unwrap()); // replied only, not posted
    }
}
