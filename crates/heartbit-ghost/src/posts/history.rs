//! Post history store for proactive posting. Tracks per-persona
//! `PostHistoryEntry` records (posted_at + topic + outcome + tweet_id).
//! Used by the topic generator's input (recent history block) and by
//! the duplicate check before the pipeline runs.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use chrono::{Duration, Utc};

use super::PostHistoryEntry;

/// Errors raised by [`PostHistoryStore`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// I/O failure (file not readable, write failed, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSONL parse failure on replay.
    #[error("parse: {0}")]
    Parse(String),
}

/// Persistent storage for proactive post outcomes.
///
/// Implementations use `Pin<Box<dyn Future>>` desugaring (matches
/// the rest of P1.5/P1.6 — no async-trait dep).
pub trait PostHistoryStore: Send + Sync {
    /// Append one entry for `persona`.
    fn record<'a>(
        &'a self,
        persona: &'a str,
        entry: PostHistoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>>;

    /// Most-recent-first up to `limit` entries for `persona`.
    fn recent<'a>(
        &'a self,
        persona: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PostHistoryEntry>, StoreError>> + Send + 'a>>;

    /// Whether a topic case-insensitive-equal to `topic` was already
    /// recorded for `persona` in the lookback `within`.
    fn was_posted_recently<'a>(
        &'a self,
        persona: &'a str,
        topic: &'a str,
        within: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>>;
}

// -- in-memory impl ----------------------------------------------------

/// Volatile in-memory store. Useful for tests and dev runs.
pub struct InMemoryPostHistoryStore {
    inner: tokio::sync::RwLock<InMemoryInner>,
}

#[derive(Default)]
struct InMemoryInner {
    /// persona → vec of entries (append order; most recent at the end).
    entries: std::collections::HashMap<String, Vec<PostHistoryEntry>>,
}

impl Default for InMemoryPostHistoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryPostHistoryStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::RwLock::new(InMemoryInner::default()),
        }
    }
}

impl PostHistoryStore for InMemoryPostHistoryStore {
    fn record<'a>(
        &'a self,
        persona: &'a str,
        entry: PostHistoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.inner
                .write()
                .await
                .entries
                .entry(persona.to_string())
                .or_default()
                .push(entry);
            Ok(())
        })
    }

    fn recent<'a>(
        &'a self,
        persona: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PostHistoryEntry>, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            let v = g.entries.get(persona).cloned().unwrap_or_default();
            let mut out: Vec<PostHistoryEntry> = v.into_iter().rev().take(limit).collect();
            out.shrink_to_fit();
            Ok(out)
        })
    }

    fn was_posted_recently<'a>(
        &'a self,
        persona: &'a str,
        topic: &'a str,
        within: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            let cutoff = Utc::now() - within;
            let topic_lower = topic.to_lowercase();
            Ok(g.entries
                .get(persona)
                .map(|v| {
                    v.iter()
                        .any(|e| e.posted_at >= cutoff && e.topic.to_lowercase() == topic_lower)
                })
                .unwrap_or(false))
        })
    }
}

// -- JSONL impl --------------------------------------------------------

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StoreLine {
    persona: String,
    entry: PostHistoryEntry,
}

/// JSONL append-only post-history store. Replays the file at [`open`]
/// time into an in-memory mirror; subsequent records both append and
/// update the mirror.
pub struct JsonlPostHistoryStore {
    path: PathBuf,
    inner: tokio::sync::RwLock<InMemoryInner>,
}

impl JsonlPostHistoryStore {
    /// Open or create the store at `path`. Replays existing JSONL events.
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
            let parsed: StoreLine = serde_json::from_str(line)
                .map_err(|e| StoreError::Parse(format!("line {line:?}: {e}")))?;
            g.entries
                .entry(parsed.persona)
                .or_default()
                .push(parsed.entry);
        }
        Ok(())
    }

    async fn append(&self, persona: &str, entry: &PostHistoryEntry) -> Result<(), StoreError> {
        use tokio::io::AsyncWriteExt;
        let line = StoreLine {
            persona: persona.to_string(),
            entry: entry.clone(),
        };
        let serialized =
            serde_json::to_string(&line).map_err(|e| StoreError::Parse(format!("{e}")))?;
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

impl PostHistoryStore for JsonlPostHistoryStore {
    fn record<'a>(
        &'a self,
        persona: &'a str,
        entry: PostHistoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.append(persona, &entry).await?;
            self.inner
                .write()
                .await
                .entries
                .entry(persona.to_string())
                .or_default()
                .push(entry);
            Ok(())
        })
    }

    fn recent<'a>(
        &'a self,
        persona: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PostHistoryEntry>, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            let v = g.entries.get(persona).cloned().unwrap_or_default();
            Ok(v.into_iter().rev().take(limit).collect())
        })
    }

    fn was_posted_recently<'a>(
        &'a self,
        persona: &'a str,
        topic: &'a str,
        within: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            let cutoff = Utc::now() - within;
            let topic_lower = topic.to_lowercase();
            Ok(g.entries
                .get(persona)
                .map(|v| {
                    v.iter()
                        .any(|e| e.posted_at >= cutoff && e.topic.to_lowercase() == topic_lower)
                })
                .unwrap_or(false))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posts::PostOutcome;

    fn fixture_entry(topic: &str, ago: chrono::Duration) -> PostHistoryEntry {
        PostHistoryEntry {
            posted_at: Utc::now() - ago,
            topic: topic.into(),
            outcome: PostOutcome::Posted {
                chosen_index: 0,
                url: "https://x.com/i/web/status/123".into(),
            },
            tweet_id: Some("123".into()),
            text: None,
        }
    }

    #[tokio::test]
    async fn in_memory_record_then_recent_returns_most_recent_first() {
        let store = InMemoryPostHistoryStore::new();
        store
            .record("p", fixture_entry("topic A", chrono::Duration::hours(2)))
            .await
            .unwrap();
        store
            .record("p", fixture_entry("topic B", chrono::Duration::hours(1)))
            .await
            .unwrap();
        let recent = store.recent("p", 5).await.unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].topic, "topic B");
        assert_eq!(recent[1].topic, "topic A");
    }

    #[tokio::test]
    async fn in_memory_was_posted_recently_is_case_insensitive_within_window() {
        let store = InMemoryPostHistoryStore::new();
        store
            .record(
                "p",
                fixture_entry("Calibrated Abstention", chrono::Duration::hours(12)),
            )
            .await
            .unwrap();
        // Same topic, different case, within 24h → true
        let hit = store
            .was_posted_recently("p", "calibrated abstention", chrono::Duration::hours(24))
            .await
            .unwrap();
        assert!(hit);
        // Outside the window → false
        let miss = store
            .was_posted_recently("p", "calibrated abstention", chrono::Duration::hours(6))
            .await
            .unwrap();
        assert!(!miss);
    }

    #[tokio::test]
    async fn in_memory_per_persona_isolation() {
        let store = InMemoryPostHistoryStore::new();
        store
            .record("a", fixture_entry("foo", chrono::Duration::hours(1)))
            .await
            .unwrap();
        let recent_a = store.recent("a", 5).await.unwrap();
        let recent_b = store.recent("b", 5).await.unwrap();
        assert_eq!(recent_a.len(), 1);
        assert!(recent_b.is_empty());
    }

    #[tokio::test]
    async fn jsonl_round_trips_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("posts.jsonl");
        {
            let s1 = JsonlPostHistoryStore::open(&path).await.unwrap();
            s1.record("p", fixture_entry("alpha", chrono::Duration::hours(1)))
                .await
                .unwrap();
            s1.record("p", fixture_entry("beta", chrono::Duration::minutes(30)))
                .await
                .unwrap();
        }
        let s2 = JsonlPostHistoryStore::open(&path).await.unwrap();
        let recent = s2.recent("p", 5).await.unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].topic, "beta");
        assert_eq!(recent[1].topic, "alpha");
    }

    #[tokio::test]
    async fn jsonl_handles_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.jsonl");
        let s = JsonlPostHistoryStore::open(&path).await.unwrap();
        assert!(s.recent("p", 5).await.unwrap().is_empty());
    }
}
