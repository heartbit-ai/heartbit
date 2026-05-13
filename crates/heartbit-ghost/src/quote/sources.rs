//! Source-tweet polling + already-quoted dedup store.
//!
//! `XUserTimelineSource` fetches recent tweets from a curated X user's
//! timeline via `GET /2/users/{id}/tweets` (the same endpoint
//! `posts::topic_context::fetch_own_tweets` uses). `QuoteSeenStore`
//! tracks which source tweet IDs the persona has already quoted so we
//! don't double-quote.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use crate::tools::client::{XApiError, XClient};

/// A candidate source tweet returned from `XUserTimelineSource::recent`.
///
/// `id`, `text`, and `author_handle` are the load-bearing fields for
/// quote drafting. `posted_at` is used by the age filter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuoteCandidate {
    /// Tweet ID.
    pub id: String,
    /// Raw tweet text.
    pub text: String,
    /// X user ID of the tweet author.
    pub author_id: String,
    /// X @handle of the tweet author (resolved from expansions).
    pub author_handle: String,
    /// UTC timestamp the tweet was posted.
    pub posted_at: DateTime<Utc>,
}

/// Object-safe async trait for fetching recent tweets from a source.
/// Production wires `XUserTimelineSource`; tests wire a mock.
pub trait QuoteSource: Send + Sync {
    /// Fetch up to 10 recent non-reply, non-retweet tweets from the given X user ID.
    fn recent<'a>(
        &'a self,
        user_id: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<QuoteCandidate>, XApiError>> + Send + 'a>,
    >;
}

/// Production source: X v2 `/2/users/{id}/tweets`.
pub struct XUserTimelineSource {
    client: Arc<XClient>,
}

impl XUserTimelineSource {
    /// Construct a new source backed by the given `XClient`.
    pub fn new(client: Arc<XClient>) -> Self {
        Self { client }
    }
}

#[derive(Debug, Deserialize)]
struct TimelineResp {
    #[serde(default)]
    data: Vec<TimelineItem>,
    #[serde(default)]
    includes: Option<TimelineIncludes>,
}

#[derive(Debug, Deserialize)]
struct TimelineItem {
    id: String,
    text: String,
    author_id: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
struct TimelineIncludes {
    #[serde(default)]
    users: Vec<TimelineUser>,
}

#[derive(Debug, Deserialize)]
struct TimelineUser {
    id: String,
    username: String,
}

impl QuoteSource for XUserTimelineSource {
    fn recent<'a>(
        &'a self,
        user_id: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<QuoteCandidate>, XApiError>> + Send + 'a>,
    > {
        let client = self.client.clone();
        Box::pin(async move {
            let path = format!("/2/users/{user_id}/tweets");
            let query: Vec<(&str, &str)> = vec![
                ("max_results", "10"),
                ("tweet.fields", "author_id,created_at"),
                ("expansions", "author_id"),
                ("user.fields", "username"),
                ("exclude", "replies,retweets"),
            ];
            let resp: TimelineResp = client.get_json(&path, &query).await?;
            let users = resp.includes.map(|i| i.users).unwrap_or_default();
            let candidates: Vec<QuoteCandidate> = resp
                .data
                .into_iter()
                .map(|t| {
                    let author_handle = users
                        .iter()
                        .find(|u| u.id == t.author_id)
                        .map(|u| u.username.clone())
                        .unwrap_or_else(|| t.author_id.clone());
                    QuoteCandidate {
                        id: t.id,
                        text: t.text,
                        author_id: t.author_id,
                        author_handle,
                        posted_at: t.created_at,
                    }
                })
                .collect();
            Ok(candidates)
        })
    }
}

/// Object-safe async trait for the already-quoted dedup store.
pub trait QuoteSeenStore: Send + Sync {
    /// Record that we've drafted/quoted the given source tweet ID.
    fn record<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), QuoteStoreError>> + Send + 'a>>;

    /// Return true if we've already quoted this source tweet.
    fn was_seen<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<bool, QuoteStoreError>> + Send + 'a>,
    >;
}

/// Errors from the [`QuoteSeenStore`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum QuoteStoreError {
    /// I/O error (e.g. from file operations in [`JsonlQuoteSeenStore`]).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialisation/deserialisation error.
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

/// In-memory store (tests + ephemeral runs).
pub struct InMemoryQuoteSeenStore {
    seen: Mutex<std::collections::HashSet<String>>,
}

impl InMemoryQuoteSeenStore {
    /// Create a new, empty in-memory store.
    pub fn new() -> Self {
        Self {
            seen: Mutex::new(std::collections::HashSet::new()),
        }
    }
}

impl Default for InMemoryQuoteSeenStore {
    fn default() -> Self {
        Self::new()
    }
}

impl QuoteSeenStore for InMemoryQuoteSeenStore {
    fn record<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), QuoteStoreError>> + Send + 'a>>
    {
        Box::pin(async move {
            let key = format!("{persona}\0{tweet_id}");
            self.seen.lock().await.insert(key);
            Ok(())
        })
    }

    fn was_seen<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<bool, QuoteStoreError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let key = format!("{persona}\0{tweet_id}");
            Ok(self.seen.lock().await.contains(&key))
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SeenEntry {
    persona: String,
    tweet_id: String,
    seen_at: DateTime<Utc>,
}

/// JSONL-backed store for restart durability.
pub struct JsonlQuoteSeenStore {
    path: PathBuf,
    cache: Mutex<std::collections::HashSet<String>>,
}

impl JsonlQuoteSeenStore {
    /// Open (creating if absent) the JSONL store and warm-load the cache.
    pub async fn open(path: &std::path::Path) -> Result<Self, QuoteStoreError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut cache = std::collections::HashSet::new();
        if path.exists() {
            let content = tokio::fs::read_to_string(path).await?;
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let entry: SeenEntry = serde_json::from_str(line)?;
                cache.insert(format!("{}\0{}", entry.persona, entry.tweet_id));
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            cache: Mutex::new(cache),
        })
    }
}

impl QuoteSeenStore for JsonlQuoteSeenStore {
    fn record<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), QuoteStoreError>> + Send + 'a>>
    {
        Box::pin(async move {
            let key = format!("{persona}\0{tweet_id}");
            let mut cache = self.cache.lock().await;
            if cache.contains(&key) {
                return Ok(());
            }
            let entry = SeenEntry {
                persona: persona.to_string(),
                tweet_id: tweet_id.to_string(),
                seen_at: Utc::now(),
            };
            let line = format!("{}\n", serde_json::to_string(&entry)?);
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .await?;
            file.write_all(line.as_bytes()).await?;
            cache.insert(key);
            Ok(())
        })
    }

    fn was_seen<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<bool, QuoteStoreError>> + Send + 'a>,
    > {
        Box::pin(async move {
            let key = format!("{persona}\0{tweet_id}");
            Ok(self.cache.lock().await.contains(&key))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_store_records_and_recalls() {
        let store = InMemoryQuoteSeenStore::new();
        assert!(!store.was_seen("p", "123").await.unwrap());
        store.record("p", "123").await.unwrap();
        assert!(store.was_seen("p", "123").await.unwrap());
        // Different persona = different key.
        assert!(!store.was_seen("other", "123").await.unwrap());
        // Different tweet = different key.
        assert!(!store.was_seen("p", "999").await.unwrap());
    }

    #[tokio::test]
    async fn jsonl_store_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seen.jsonl");
        {
            let store = JsonlQuoteSeenStore::open(&path).await.unwrap();
            store.record("p", "111").await.unwrap();
            store.record("p", "222").await.unwrap();
        }
        // Reopen and check the cache warm-loaded.
        let store = JsonlQuoteSeenStore::open(&path).await.unwrap();
        assert!(store.was_seen("p", "111").await.unwrap());
        assert!(store.was_seen("p", "222").await.unwrap());
        assert!(!store.was_seen("p", "333").await.unwrap());
    }

    #[tokio::test]
    async fn jsonl_record_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seen.jsonl");
        let store = JsonlQuoteSeenStore::open(&path).await.unwrap();
        store.record("p", "111").await.unwrap();
        store.record("p", "111").await.unwrap();
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(
            content.lines().count(),
            1,
            "duplicate record() calls must not write twice; got:\n{content}"
        );
    }

    /// Persona slugs in this codebase contain `:` (e.g. "heartbit-ghost:x").
    /// The key separator must NOT be `:` or persona="heartbit-ghost" +
    /// tweet="x:111" would collide with persona="heartbit-ghost:x" +
    /// tweet="111". Pinning the collision-safety here so a future
    /// "just use colon" simplification trips this test.
    #[tokio::test]
    async fn key_separator_is_collision_safe_for_colon_in_persona_name() {
        let store = InMemoryQuoteSeenStore::new();
        // Record under persona "heartbit-ghost:x" with tweet "111".
        store.record("heartbit-ghost:x", "111").await.unwrap();
        // Different persona ("heartbit-ghost") with a tweet that LOOKS
        // like it concatenates to the same key MUST be was_seen=false.
        // If the separator were `:`, both would produce "heartbit-ghost:x:111".
        assert!(
            !store.was_seen("heartbit-ghost", "x:111").await.unwrap(),
            "persona/tweet key collision detected — separator must not appear in slugs"
        );
    }
}
