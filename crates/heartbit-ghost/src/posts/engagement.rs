//! Per-tweet engagement metrics — captured periodically by the
//! `engagement_collector`, joined against post history by
//! `TopPostsProvider` to feed the writer agent's few-shot exemplars.
//!
//! See `docs/superpowers/specs/2026-05-11-heartbit-ghost-engagement-feedback-design.md`.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::posts::{PostHistoryStore, PostOutcome};
use crate::tools::client::{XApiError, XClient};

/// One snapshot of a tweet's engagement metrics, captured at
/// `captured_at`. Stored append-only — the most recent snapshot
/// per `tweet_id` is the "current" value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementSnapshot {
    /// X tweet ID.
    pub tweet_id: String,
    /// When this snapshot was captured (the daemon's local UTC clock).
    pub captured_at: DateTime<Utc>,
    /// Number of likes.
    pub likes: u64,
    /// Number of replies.
    pub replies: u64,
    /// Number of retweets.
    pub retweets: u64,
    /// Number of quote tweets.
    pub quotes: u64,
    /// Number of bookmarks.
    pub bookmarks: u64,
    /// Number of impressions. Optional because (1) X API may omit it
    /// for very fresh tweets and (2) older snapshots written before
    /// the field was added must still parse.
    #[serde(default)]
    pub impressions: Option<u64>,
}

/// Errors raised by [`EngagementStore`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum EngagementStoreError {
    /// I/O failure (file not readable, write failed, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSONL parse failure on replay (or serialization failure on append).
    #[error("parse: {0}")]
    Parse(String),
}

/// Boxed future returned by [`EngagementStore::latest_per_tweet`].
/// Aliased to keep the signature readable (and clippy quiet on
/// `type_complexity`).
pub type LatestPerTweetFut<'a> = Pin<
    Box<
        dyn Future<Output = Result<HashMap<String, EngagementSnapshot>, EngagementStoreError>>
            + Send
            + 'a,
    >,
>;

/// Append-only store for engagement snapshots.
///
/// `Pin<Box<dyn Future>>` desugaring (matches `MentionStore`).
pub trait EngagementStore: Send + Sync {
    /// Append a snapshot.
    fn record<'a>(
        &'a self,
        snap: EngagementSnapshot,
    ) -> Pin<Box<dyn Future<Output = Result<(), EngagementStoreError>> + Send + 'a>>;

    /// Return the latest snapshot per tweet_id.
    fn latest_per_tweet<'a>(&'a self) -> LatestPerTweetFut<'a>;
}

// ─── In-memory impl ────────────────────────────────────────────────────────

/// Volatile in-memory store. Useful for tests and dev runs.
pub struct InMemoryEngagementStore {
    inner: tokio::sync::RwLock<HashMap<String, EngagementSnapshot>>,
}

impl Default for InMemoryEngagementStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryEngagementStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::RwLock::new(HashMap::new()),
        }
    }
}

impl EngagementStore for InMemoryEngagementStore {
    fn record<'a>(
        &'a self,
        snap: EngagementSnapshot,
    ) -> Pin<Box<dyn Future<Output = Result<(), EngagementStoreError>> + Send + 'a>> {
        Box::pin(async move {
            let mut g = self.inner.write().await;
            // Always overwrite — newer snapshots replace older for the same
            // tweet_id. This matches "latest snapshot is current".
            g.insert(snap.tweet_id.clone(), snap);
            Ok(())
        })
    }

    fn latest_per_tweet<'a>(&'a self) -> LatestPerTweetFut<'a> {
        Box::pin(async move { Ok(self.inner.read().await.clone()) })
    }
}

// ─── JSONL-backed impl ─────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum StoreEvent {
    Snapshot {
        tweet_id: String,
        captured_at: DateTime<Utc>,
        likes: u64,
        replies: u64,
        retweets: u64,
        quotes: u64,
        bookmarks: u64,
        #[serde(default)]
        impressions: Option<u64>,
    },
}

/// JSONL append-only store. Replays the file at [`JsonlEngagementStore::open`]
/// time into an in-memory mirror; subsequent writes both append to the file
/// and update the mirror.
pub struct JsonlEngagementStore {
    path: PathBuf,
    inner: tokio::sync::RwLock<HashMap<String, EngagementSnapshot>>,
}

impl JsonlEngagementStore {
    /// Open or create the store at `path`. Replays existing JSONL events
    /// into the in-memory mirror.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self, EngagementStoreError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let store = Self {
            path,
            inner: tokio::sync::RwLock::new(HashMap::new()),
        };
        store.replay().await?;
        Ok(store)
    }

    async fn replay(&self) -> Result<(), EngagementStoreError> {
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
                .map_err(|e| EngagementStoreError::Parse(format!("line {line:?}: {e}")))?;
            let StoreEvent::Snapshot {
                tweet_id,
                captured_at,
                likes,
                replies,
                retweets,
                quotes,
                bookmarks,
                impressions,
            } = evt;
            // Newer entries (later in file) overwrite — replay order = arrival order.
            g.insert(
                tweet_id.clone(),
                EngagementSnapshot {
                    tweet_id,
                    captured_at,
                    likes,
                    replies,
                    retweets,
                    quotes,
                    bookmarks,
                    impressions,
                },
            );
        }
        Ok(())
    }

    async fn append(&self, evt: &StoreEvent) -> Result<(), EngagementStoreError> {
        use tokio::io::AsyncWriteExt;
        let serialized =
            serde_json::to_string(evt).map_err(|e| EngagementStoreError::Parse(format!("{e}")))?;
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

impl EngagementStore for JsonlEngagementStore {
    fn record<'a>(
        &'a self,
        snap: EngagementSnapshot,
    ) -> Pin<Box<dyn Future<Output = Result<(), EngagementStoreError>> + Send + 'a>> {
        Box::pin(async move {
            let evt = StoreEvent::Snapshot {
                tweet_id: snap.tweet_id.clone(),
                captured_at: snap.captured_at,
                likes: snap.likes,
                replies: snap.replies,
                retweets: snap.retweets,
                quotes: snap.quotes,
                bookmarks: snap.bookmarks,
                impressions: snap.impressions,
            };
            self.append(&evt).await?;
            self.inner.write().await.insert(snap.tweet_id.clone(), snap);
            Ok(())
        })
    }

    fn latest_per_tweet<'a>(&'a self) -> LatestPerTweetFut<'a> {
        Box::pin(async move { Ok(self.inner.read().await.clone()) })
    }
}

// ─── refresh_engagement: X API batch fetch ─────────────────────────────────

/// Outcome of one refresh cycle. Used by the scheduler/handler for INFO logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefreshOutcome {
    /// Tweets that passed the age filters and were sent to the API.
    pub queried: usize,
    /// Tweets we successfully snapshotted.
    pub refreshed: usize,
    /// Tweets younger than `min_age_hours` — skipped.
    pub skipped_too_young: usize,
    /// Tweets older than `max_age_days` — skipped.
    pub skipped_too_old: usize,
    /// Tweets the API omitted from the response (deleted, private, etc.).
    pub failed: usize,
}

/// Errors raised by [`refresh_engagement`].
#[derive(Debug, thiserror::Error)]
pub enum RefreshError {
    /// Reading post history failed.
    #[error("history: {0}")]
    History(String),
    /// The X API call (or response parse) failed.
    #[error("api: {0}")]
    Api(#[from] XApiError),
    /// Writing the snapshot to the engagement store failed.
    #[error("store: {0}")]
    Store(#[from] EngagementStoreError),
}

/// Tweet API response shape (subset we care about).
#[derive(Debug, Deserialize)]
struct TweetApiResponse {
    #[serde(default)]
    data: Vec<TweetApiItem>,
}

#[derive(Debug, Deserialize)]
struct TweetApiItem {
    id: String,
    #[serde(default)]
    public_metrics: Option<TweetPublicMetrics>,
}

#[derive(Debug, Deserialize)]
struct TweetPublicMetrics {
    #[serde(default)]
    like_count: u64,
    #[serde(default)]
    reply_count: u64,
    #[serde(default)]
    retweet_count: u64,
    #[serde(default)]
    quote_count: u64,
    #[serde(default)]
    bookmark_count: u64,
    #[serde(default)]
    impression_count: Option<u64>,
}

/// Batch-refresh engagement metrics for every `Posted` entry in the
/// persona's history within `[now - max_age_days, now - min_age_hours]`.
/// Up to 100 tweet ids per X API call (the documented limit).
pub async fn refresh_engagement(
    client: &XClient,
    history: &dyn PostHistoryStore,
    store: &dyn EngagementStore,
    persona: &str,
    now: DateTime<Utc>,
    max_age_days: i64,
    min_age_hours: i64,
) -> Result<RefreshOutcome, RefreshError> {
    let entries = history
        .recent(persona, 1_000)
        .await
        .map_err(|e| RefreshError::History(e.to_string()))?;

    let max_age = chrono::Duration::days(max_age_days);
    let min_age = chrono::Duration::hours(min_age_hours);

    let mut skipped_too_young = 0usize;
    let mut skipped_too_old = 0usize;
    let mut eligible_ids: Vec<String> = Vec::new();
    for entry in &entries {
        let Some(ref id) = entry.tweet_id else {
            continue;
        };
        if !matches!(entry.outcome, PostOutcome::Posted { .. }) {
            continue;
        }
        let age = now - entry.posted_at;
        if age < min_age {
            skipped_too_young += 1;
            continue;
        }
        if age > max_age {
            skipped_too_old += 1;
            continue;
        }
        eligible_ids.push(id.clone());
    }

    let mut refreshed = 0usize;
    let mut failed_in_response = 0usize;

    for chunk in eligible_ids.chunks(100) {
        let ids = chunk.join(",");
        let response: TweetApiResponse = client
            .get_json(
                "/2/tweets",
                &[("ids", &ids), ("tweet.fields", "public_metrics")],
            )
            .await?;
        let returned: std::collections::HashSet<&str> =
            response.data.iter().map(|t| t.id.as_str()).collect();
        // Whatever the API omitted = "failed" (deleted/private/etc.).
        failed_in_response += chunk
            .iter()
            .filter(|id| !returned.contains(id.as_str()))
            .count();

        let captured_at = Utc::now();
        for item in response.data {
            let metrics = item.public_metrics.unwrap_or(TweetPublicMetrics {
                like_count: 0,
                reply_count: 0,
                retweet_count: 0,
                quote_count: 0,
                bookmark_count: 0,
                impression_count: None,
            });
            store
                .record(EngagementSnapshot {
                    tweet_id: item.id,
                    captured_at,
                    likes: metrics.like_count,
                    replies: metrics.reply_count,
                    retweets: metrics.retweet_count,
                    quotes: metrics.quote_count,
                    bookmarks: metrics.bookmark_count,
                    impressions: metrics.impression_count,
                })
                .await?;
            refreshed += 1;
        }
    }

    Ok(RefreshOutcome {
        queried: eligible_ids.len(),
        refreshed,
        skipped_too_young,
        skipped_too_old,
        failed: failed_in_response,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::tempdir;

    fn snap(tweet_id: &str, likes: u64, impressions: Option<u64>) -> EngagementSnapshot {
        EngagementSnapshot {
            tweet_id: tweet_id.into(),
            captured_at: Utc::now(),
            likes,
            replies: 0,
            retweets: 0,
            quotes: 0,
            bookmarks: 0,
            impressions,
        }
    }

    #[tokio::test]
    async fn in_memory_record_and_latest_round_trip() {
        let store = InMemoryEngagementStore::new();
        store.record(snap("t1", 10, Some(100))).await.unwrap();
        store.record(snap("t2", 50, None)).await.unwrap();
        let latest = store.latest_per_tweet().await.unwrap();
        assert_eq!(latest.len(), 2);
        assert_eq!(latest["t1"].likes, 10);
        assert_eq!(latest["t1"].impressions, Some(100));
        assert_eq!(latest["t2"].likes, 50);
    }

    #[tokio::test]
    async fn in_memory_latest_returns_most_recent_for_same_tweet() {
        let store = InMemoryEngagementStore::new();
        store.record(snap("t1", 10, Some(100))).await.unwrap();
        // A second snapshot for the same tweet — likes have grown.
        store.record(snap("t1", 25, Some(250))).await.unwrap();
        let latest = store.latest_per_tweet().await.unwrap();
        assert_eq!(latest.len(), 1);
        assert_eq!(latest["t1"].likes, 25);
        assert_eq!(latest["t1"].impressions, Some(250));
    }

    #[tokio::test]
    async fn jsonl_round_trips_across_reload() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("engagement.jsonl");
        {
            let s = JsonlEngagementStore::open(&path).await.unwrap();
            s.record(snap("t1", 10, Some(100))).await.unwrap();
            s.record(snap("t2", 50, None)).await.unwrap();
            s.record(snap("t1", 30, Some(200))).await.unwrap(); // newer
        }
        let s = JsonlEngagementStore::open(&path).await.unwrap();
        let latest = s.latest_per_tweet().await.unwrap();
        assert_eq!(latest["t1"].likes, 30);
        assert_eq!(latest["t1"].impressions, Some(200));
        assert_eq!(latest["t2"].impressions, None);
    }

    #[tokio::test]
    async fn jsonl_handles_missing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("does_not_exist.jsonl");
        let s = JsonlEngagementStore::open(&path).await.unwrap();
        assert!(s.latest_per_tweet().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn jsonl_parses_legacy_snapshot_without_impressions() {
        // Backward-compat: a snapshot written before `impressions` was added
        // must still parse (missing field → None).
        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy.jsonl");
        let line = r#"{"event":"snapshot","tweet_id":"t_legacy","captured_at":"2026-05-01T00:00:00Z","likes":7,"replies":0,"retweets":0,"quotes":0,"bookmarks":0}"#;
        tokio::fs::write(&path, format!("{line}\n")).await.unwrap();

        let s = JsonlEngagementStore::open(&path).await.unwrap();
        let latest = s.latest_per_tweet().await.unwrap();
        assert_eq!(latest["t_legacy"].likes, 7);
        assert_eq!(latest["t_legacy"].impressions, None);
    }

    // ─── refresh_engagement ────────────────────────────────────────────

    use crate::posts::{InMemoryPostHistoryStore, PostHistoryEntry, PostHistoryStore, PostOutcome};
    use chrono::Duration;
    use heartbit_core::Secret;
    use wiremock::matchers::{method, path as wm_path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn posted_entry(tweet_id: &str, hours_ago: i64) -> PostHistoryEntry {
        PostHistoryEntry {
            posted_at: Utc::now() - Duration::hours(hours_ago),
            topic: format!("topic_{tweet_id}"),
            outcome: PostOutcome::Posted {
                chosen_index: 0,
                url: format!("https://x.com/u/status/{tweet_id}"),
            },
            tweet_id: Some(tweet_id.into()),
            text: Some(format!("text for {tweet_id}")),
        }
    }

    fn test_client(server_uri: &str) -> XClient {
        XClient::new(
            server_uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("test client builds")
    }

    fn client_from(server: &MockServer) -> XClient {
        test_client(&server.uri())
    }

    #[tokio::test]
    async fn refresh_happy_path_snapshots_eligible_tweets() {
        let history = InMemoryPostHistoryStore::new();
        // Insert tB first, tA second — `recent()` returns most-recent
        // (last-inserted) first, so the resulting `ids` chunk is "tA,tB".
        history.record("p", posted_entry("tB", 72)).await.unwrap();
        history.record("p", posted_entry("tA", 48)).await.unwrap();
        let store = InMemoryEngagementStore::new();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets"))
            .and(query_param("ids", "tA,tB"))
            .and(query_param("tweet.fields", "public_metrics"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"id": "tA", "public_metrics": {"like_count": 10, "reply_count": 1, "retweet_count": 2, "quote_count": 0, "bookmark_count": 3, "impression_count": 200}},
                    {"id": "tB", "public_metrics": {"like_count": 50, "reply_count": 5, "retweet_count": 8, "quote_count": 1, "bookmark_count": 9, "impression_count": 1500}}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;
        let client = test_client(&server.uri());

        let outcome = refresh_engagement(&client, &history, &store, "p", Utc::now(), 30, 24)
            .await
            .unwrap();
        assert_eq!(outcome.queried, 2);
        assert_eq!(outcome.refreshed, 2);
        assert_eq!(outcome.failed, 0);

        let latest = store.latest_per_tweet().await.unwrap();
        assert_eq!(latest["tA"].likes, 10);
        assert_eq!(latest["tA"].impressions, Some(200));
        assert_eq!(latest["tB"].likes, 50);
    }

    #[tokio::test]
    async fn refresh_skips_too_young() {
        let history = InMemoryPostHistoryStore::new();
        history
            .record("p", posted_entry("recent", 1)) // 1h old, below min_age
            .await
            .unwrap();
        let store = InMemoryEngagementStore::new();
        let server = MockServer::start().await;
        // No mock — if refresh queries the API, the test will fail with 404.

        let outcome = refresh_engagement(
            &client_from(&server),
            &history,
            &store,
            "p",
            Utc::now(),
            30,
            24,
        )
        .await
        .unwrap();
        assert_eq!(outcome.queried, 0);
        assert_eq!(outcome.skipped_too_young, 1);
    }

    #[tokio::test]
    async fn refresh_skips_too_old() {
        let history = InMemoryPostHistoryStore::new();
        history
            .record("p", posted_entry("old", 24 * 45)) // 45 days
            .await
            .unwrap();
        let store = InMemoryEngagementStore::new();
        let server = MockServer::start().await;

        let outcome = refresh_engagement(
            &client_from(&server),
            &history,
            &store,
            "p",
            Utc::now(),
            30,
            24,
        )
        .await
        .unwrap();
        assert_eq!(outcome.queried, 0);
        assert_eq!(outcome.skipped_too_old, 1);
    }

    #[tokio::test]
    async fn refresh_handles_partial_failure() {
        // 3 tweets queried; X returns only 2 in the data array (one was deleted).
        let history = InMemoryPostHistoryStore::new();
        for id in ["tX", "tY", "tZ"] {
            history.record("p", posted_entry(id, 48)).await.unwrap();
        }
        let store = InMemoryEngagementStore::new();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"id": "tX", "public_metrics": {"like_count": 1, "reply_count": 0, "retweet_count": 0, "quote_count": 0, "bookmark_count": 0, "impression_count": 10}},
                    {"id": "tZ", "public_metrics": {"like_count": 9, "reply_count": 0, "retweet_count": 0, "quote_count": 0, "bookmark_count": 0, "impression_count": 90}}
                ]
            })))
            .mount(&server)
            .await;
        let client = test_client(&server.uri());

        let outcome = refresh_engagement(&client, &history, &store, "p", Utc::now(), 30, 24)
            .await
            .unwrap();
        assert_eq!(outcome.queried, 3);
        assert_eq!(outcome.refreshed, 2);
        assert_eq!(outcome.failed, 1);
    }

    #[tokio::test]
    async fn refresh_batches_at_100_ids() {
        // 150 eligible tweets → must produce TWO API calls (100 + 50).
        let history = InMemoryPostHistoryStore::new();
        for i in 0..150 {
            let id = format!("t{i:03}");
            history.record("p", posted_entry(&id, 48)).await.unwrap();
        }
        let store = InMemoryEngagementStore::new();

        let server = MockServer::start().await;
        // Match any /2/tweets call and respond with an empty data array.
        // Simplest — partial failure counted; we assert the call count.
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": []
            })))
            .expect(2)
            .mount(&server)
            .await;
        let client = test_client(&server.uri());

        let _ = refresh_engagement(&client, &history, &store, "p", Utc::now(), 30, 24)
            .await
            .unwrap();
        // wiremock .expect(2) verifies on Drop.
    }

    #[tokio::test]
    async fn refresh_empty_history_no_api_call() {
        let history = InMemoryPostHistoryStore::new();
        let store = InMemoryEngagementStore::new();
        let server = MockServer::start().await;
        // No mock — any API call would fail the test.
        let client = test_client(&server.uri());

        let outcome = refresh_engagement(&client, &history, &store, "p", Utc::now(), 30, 24)
            .await
            .unwrap();
        assert_eq!(outcome.queried, 0);
        assert_eq!(outcome.refreshed, 0);
    }
}
