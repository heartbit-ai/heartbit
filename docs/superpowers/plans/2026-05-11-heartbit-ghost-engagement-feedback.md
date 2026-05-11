# Engagement Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a closed-loop engagement signal so the proactive-post `writer` agent learns from the operator's audience — top-engaged recent posts become few-shot exemplars in every new draft.

**Architecture:** Four modular layers. `EngagementStore` (JSONL + in-memory) holds per-tweet `EngagementSnapshot`s. `refresh_engagement` batch-fetches `GET /2/tweets?ids=...&tweet.fields=public_metrics` on a scheduled tick. `TopPostsProvider` joins engagement snapshots against post history + composite score → top-N. `persona_post_handler` prepends an exemplar block to the writer's user message just before invocation. Proactive-post-only; reply pipeline unchanged.

**Tech Stack:** Rust 2024 edition. Existing patterns: `MentionStore`/`JsonlMentionStore` (JSONL append-only with replay), `XClient::get_json` (OAuth1 user-context), `PersonaPostScheduler` (jittered interval). New: `chrono::Utc` for timestamps; no new external deps.

---

## File Structure

**New files:**
- `crates/heartbit-ghost/src/posts/engagement.rs` — `EngagementSnapshot`, `EngagementStore` trait, `InMemoryEngagementStore`, `JsonlEngagementStore`, `refresh_engagement` free fn, `TopPost`, `TopPostsProvider` trait, `JoinedTopPostsProvider`
- `crates/heartbit/src/daemon/engagement_collector.rs` — `EngagementCollectorScheduler` (scheduler that dispatches `DaemonCommand::EngagementRefresh` on jittered cadence)
- `crates/heartbit/src/daemon/engagement_refresh_handler.rs` — `handle_engagement_refresh` (free-function handler invoked when the consumer reads the command)

**Modified files:**
- `crates/heartbit-ghost/src/posts/mod.rs` — re-export new types; extend `PostHistoryEntry` with `text: Option<String>`
- `crates/heartbit-core/src/config/daemon.rs` — extend `PersonaPostsConfig` with 4 engagement fields + defaults
- `crates/heartbit/src/daemon/types.rs` — new `DaemonCommand::EngagementRefresh { persona }` variant
- `crates/heartbit/src/daemon/posts_context.rs` — extend `PersonaPostEntry` with `engagement_store`, `top_posts_provider`, refresh cadence fields
- `crates/heartbit/src/daemon/core.rs` — spawn `EngagementCollectorScheduler` next to `PersonaPostScheduler`; route `DaemonCommand::EngagementRefresh` to the handler
- `crates/heartbit/src/daemon/persona_post_handler.rs` — record `text` on Posted; call `TopPostsProvider::top_n` + prepend exemplar block to writer user message
- `crates/heartbit/src/daemon/mod.rs` — module declarations + re-exports
- `crates/heartbit-cli/src/daemon/mod.rs` — build `JsonlEngagementStore` + `JoinedTopPostsProvider` at startup; thread into `PersonaPostEntry`

---

## Task 1 — `EngagementSnapshot` + `EngagementStore` (storage layer)

**Files:**
- Create: `crates/heartbit-ghost/src/posts/engagement.rs`
- Modify: `crates/heartbit-ghost/src/posts/mod.rs:1-30` (module decl + re-exports)

### Step 1: Write the failing tests

Add to a new module `crates/heartbit-ghost/src/posts/engagement.rs` (tests at the bottom):

```rust
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
}
```

### Step 2: Run tests — verify they fail to compile

Run: `cargo test -p heartbit-ghost --lib posts::engagement`
Expected: compilation errors (`EngagementSnapshot`, `InMemoryEngagementStore`, `JsonlEngagementStore` not defined).

### Step 3: Implement `EngagementSnapshot` + `EngagementStoreError` + `EngagementStore` trait

Top of `crates/heartbit-ghost/src/posts/engagement.rs`:

```rust
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
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(String),
}

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
    fn latest_per_tweet<'a>(
        &'a self,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<HashMap<String, EngagementSnapshot>, EngagementStoreError>>
                + Send
                + 'a,
        >,
    >;
}
```

### Step 4: Implement `InMemoryEngagementStore`

Append to `engagement.rs`:

```rust
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

    fn latest_per_tweet<'a>(
        &'a self,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<HashMap<String, EngagementSnapshot>, EngagementStoreError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move { Ok(self.inner.read().await.clone()) })
    }
}
```

### Step 5: Implement `JsonlEngagementStore`

Append to `engagement.rs`:

```rust
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

/// JSONL append-only store. Replays the file at `open()` time into an
/// in-memory mirror; subsequent writes append + update mirror.
pub struct JsonlEngagementStore {
    path: PathBuf,
    inner: tokio::sync::RwLock<HashMap<String, EngagementSnapshot>>,
}

impl JsonlEngagementStore {
    /// Open or create the store at `path`. Replays existing JSONL into memory.
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
        let serialized = serde_json::to_string(evt)
            .map_err(|e| EngagementStoreError::Parse(format!("{e}")))?;
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

    fn latest_per_tweet<'a>(
        &'a self,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<HashMap<String, EngagementSnapshot>, EngagementStoreError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move { Ok(self.inner.read().await.clone()) })
    }
}
```

### Step 6: Wire module declaration + re-exports

Modify `crates/heartbit-ghost/src/posts/mod.rs`:

```rust
// at the top of the existing module declarations
pub mod engagement;

// at the bottom of the existing re-exports
pub use engagement::{
    EngagementSnapshot, EngagementStore, EngagementStoreError, InMemoryEngagementStore,
    JsonlEngagementStore,
};
```

### Step 7: Run tests — verify they pass

Run: `cargo test -p heartbit-ghost --lib posts::engagement`
Expected: all 5 tests pass.

### Step 8: Quality gate

Run: `cargo fmt -- --check && cargo clippy -p heartbit-ghost --all-targets -- -D warnings && cargo test -p heartbit-ghost`
Expected: clean.

### Step 9: Commit

```bash
git add crates/heartbit-ghost/src/posts/engagement.rs crates/heartbit-ghost/src/posts/mod.rs
git commit -m "feat(ghost): EngagementSnapshot + EngagementStore (in-mem + JSONL)

Storage layer for the engagement feedback loop. Mirrors the pattern of
MentionStore: trait with InMemory + Jsonl impls, latest_per_tweet returns
the most recent snapshot. Backward-compat parsing for legacy snapshots
without impressions.

Refs spec docs/superpowers/specs/2026-05-11-heartbit-ghost-engagement-feedback-design.md"
```

---

## Task 2 — `refresh_engagement` (X API batch fetch)

**Files:**
- Modify: `crates/heartbit-ghost/src/posts/engagement.rs` (add `refresh_engagement` + tests at the bottom of the impl section)
- (Existing: `crates/heartbit-ghost/src/tools/client.rs` `XClient::get_json` — already supports the call shape)

### Step 1: Add the new module-level types + tests first

Append to `engagement.rs`:

```rust
// Public alongside the store types
use crate::posts::{PostHistoryEntry, PostHistoryStore, PostOutcome};
use crate::tools::client::{XApiError, XClient};

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
    #[error("history: {0}")]
    History(String),
    #[error("api: {0}")]
    Api(#[from] XApiError),
    #[error("store: {0}")]
    Store(#[from] EngagementStoreError),
}
```

Tests for `refresh_engagement` (append to existing `#[cfg(test)] mod tests`):

```rust
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

    #[tokio::test]
    async fn refresh_happy_path_snapshots_eligible_tweets() {
        let history = InMemoryPostHistoryStore::new();
        history
            .record("p", posted_entry("tA", 48))
            .await
            .unwrap();
        history
            .record("p", posted_entry("tB", 72))
            .await
            .unwrap();
        let store = InMemoryEngagementStore::new();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets"))
            .and(query_param("ids", "tA,tB"))
            .and(query_param(
                "tweet.fields",
                "public_metrics",
            ))
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
        // No mock — if refresh queries the API, the test will fail with connection refused.

        let outcome = refresh_engagement(&client_from(&server), &history, &store, "p", Utc::now(), 30, 24)
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

        let outcome = refresh_engagement(&client_from(&server), &history, &store, "p", Utc::now(), 30, 24)
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
        let mut ids = Vec::new();
        for i in 0..150 {
            let id = format!("t{i:03}");
            history.record("p", posted_entry(&id, 48)).await.unwrap();
            ids.push(id);
        }
        let store = InMemoryEngagementStore::new();

        let server = MockServer::start().await;
        // Match any /2/tweets call and respond with the requested ids. Use
        // a permissive matcher that just returns whatever was in `ids`.
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": []  // simplest — partial failure counted; we assert call count
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
```

Helper used by tests:
```rust
    fn client_from(server: &MockServer) -> XClient {
        test_client(&server.uri())
    }
```

### Step 2: Run — verify failures

Run: `cargo test -p heartbit-ghost --lib posts::engagement`
Expected: compile errors (`refresh_engagement` not defined; `PostHistoryEntry.text` not yet a field).

### Step 3: Patch `PostHistoryEntry` to add `text: Option<String>`

Modify `crates/heartbit-ghost/src/posts/mod.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostHistoryEntry {
    pub posted_at: DateTime<Utc>,
    pub topic: String,
    pub outcome: PostOutcome,
    pub tweet_id: Option<String>,
    /// First tweet of the thread (or single tweet) text. Captured at
    /// post time so `TopPostsProvider` doesn't need to round-trip the
    /// X API to render exemplars. `#[serde(default)]` for backward
    /// compatibility with entries written before P2.0.
    #[serde(default)]
    pub text: Option<String>,
}
```

Update any test/fixture call sites that construct `PostHistoryEntry` directly so they compile (add `text: None` or `text: Some(...)` — `git grep "PostHistoryEntry {"` to find them).

### Step 4: Implement `refresh_engagement`

Append to `engagement.rs`:

```rust
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
            .get_json("/2/tweets", &[("ids", &ids), ("tweet.fields", "public_metrics")])
            .await?;
        let returned: std::collections::HashSet<&str> =
            response.data.iter().map(|t| t.id.as_str()).collect();
        // Whatever the API omitted = "failed" (deleted/private/etc.).
        failed_in_response += chunk.iter().filter(|id| !returned.contains(id.as_str())).count();

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
```

### Step 5: Run tests — verify pass

Run: `cargo test -p heartbit-ghost --lib posts::engagement`
Expected: 11 tests pass (5 from Task 1 + 6 new).

### Step 6: Quality gate

Run: `cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: clean (any `PostHistoryEntry { ... }` literal not updated must be fixed).

### Step 7: Commit

```bash
git add -A
git commit -m "feat(ghost): refresh_engagement helper + PostHistoryEntry.text

Batch GET /2/tweets?ids=...&tweet.fields=public_metrics, 100 ids per call.
Skip too-young (default <24h) and too-old (default >30d) entries.
PostHistoryEntry gains text: Option<String> (#[serde(default)] for backcompat).

Refs spec."
```

---

## Task 3 — `EngagementCollectorScheduler` + Kafka command + handler

**Files:**
- Create: `crates/heartbit/src/daemon/engagement_collector.rs`
- Create: `crates/heartbit/src/daemon/engagement_refresh_handler.rs`
- Modify: `crates/heartbit/src/daemon/types.rs` (new `DaemonCommand::EngagementRefresh` variant)
- Modify: `crates/heartbit/src/daemon/mod.rs` (re-exports)
- Modify: `crates/heartbit/src/daemon/core.rs` (spawn scheduler + route command)
- Modify: `crates/heartbit/src/daemon/posts_context.rs` (extend `PersonaPostEntry` with engagement fields)

### Step 1: Add `DaemonCommand::EngagementRefresh` variant

Modify `crates/heartbit/src/daemon/types.rs`:

```rust
    /// Refresh engagement metrics for every Posted tweet in the persona's
    /// history. Dispatched by `EngagementCollectorScheduler` on the
    /// configured cadence.
    EngagementRefresh {
        /// Persona name (e.g. `"heartbit-ghost:x"`).
        persona: String,
    },
```

Add a serde round-trip test next to the existing `daemon_command_persona_post_serde_round_trips`:

```rust
    #[test]
    fn daemon_command_engagement_refresh_serde_round_trips() {
        let cmd = DaemonCommand::EngagementRefresh {
            persona: "heartbit-ghost:x".into(),
        };
        let s = serde_json::to_string(&cmd).unwrap();
        let parsed: DaemonCommand = serde_json::from_str(&s).unwrap();
        match parsed {
            DaemonCommand::EngagementRefresh { persona } => {
                assert_eq!(persona, "heartbit-ghost:x");
            }
            other => panic!("expected EngagementRefresh, got {other:?}"),
        }
    }
```

### Step 2: Implement `EngagementCollectorScheduler`

Create `crates/heartbit/src/daemon/engagement_collector.rs`:

```rust
//! Periodic engagement-refresh scheduler. Fires
//! `DaemonCommand::EngagementRefresh` per configured persona on a
//! jittered cadence (default 6h ±25%).
//!
//! No `active_hours` gate — engagement collection is cheap and runs
//! around the clock. The handler will skip too-young / too-old tweets
//! at refresh time.

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::CommandProducer;
use super::types::DaemonCommand;

/// Per-persona engagement collector. Pattern mirrors PersonaPostScheduler.
pub struct EngagementCollectorScheduler {
    persona: String,
    interval: Duration,
    jitter_pct: u32,
    producer: Arc<dyn CommandProducer>,
    commands_topic: String,
}

impl EngagementCollectorScheduler {
    pub fn new(
        persona: impl Into<String>,
        interval: Duration,
        jitter_pct: u32,
        producer: Arc<dyn CommandProducer>,
        commands_topic: impl Into<String>,
    ) -> Self {
        Self {
            persona: persona.into(),
            interval,
            jitter_pct: jitter_pct.min(50),
            producer,
            commands_topic: commands_topic.into(),
        }
    }

    fn jittered_interval(&self) -> Duration {
        if self.jitter_pct == 0 {
            return self.interval;
        }
        let base = self.interval.as_secs_f64();
        let pct = self.jitter_pct as f64 / 100.0;
        let factor = 1.0 + (rand::random::<f64>() * 2.0 - 1.0) * pct;
        let next = (base * factor).max(60.0);
        Duration::from_secs_f64(next)
    }

    pub async fn run(self, cancel: CancellationToken) {
        loop {
            let next = self.jittered_interval();
            tracing::debug!(
                persona = %self.persona,
                next_sleep_secs = next.as_secs(),
                "engagement collector: sleeping until next refresh"
            );
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::info!(persona = %self.persona, "engagement collector shutting down");
                    break;
                }
                _ = tokio::time::sleep(next) => {
                    let cmd = DaemonCommand::EngagementRefresh { persona: self.persona.clone() };
                    let payload = match serde_json::to_vec(&cmd) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!(error = %e, "failed to serialize EngagementRefresh");
                            continue;
                        }
                    };
                    let key = format!("engagement:{}", self.persona);
                    if let Err(e) = self
                        .producer
                        .send_command(&self.commands_topic, &key, &payload)
                        .await
                    {
                        tracing::error!(persona = %self.persona, error = %e, "failed to dispatch EngagementRefresh");
                    } else {
                        tracing::debug!(persona = %self.persona, "engagement collector dispatched");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ChannelCommandProducer;
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn fires_engagement_refresh_after_interval() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler = EngagementCollectorScheduler::new(
            "heartbit-ghost:x",
            Duration::from_secs(60),
            0, // deterministic test
            producer,
            "test.commands",
        );
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(scheduler.run(cancel_for_task));

        tokio::time::advance(Duration::from_secs(61)).await;
        let (key, payload) = rx.recv().await.expect("scheduler should have fired");
        assert_eq!(key, "engagement:heartbit-ghost:x");
        let cmd: DaemonCommand = serde_json::from_slice(&payload).unwrap();
        assert!(matches!(&cmd, DaemonCommand::EngagementRefresh { persona } if persona == "heartbit-ghost:x"));
        cancel.cancel();
        let _ = handle.await;
    }

    #[test]
    fn jitter_clamps_at_50() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let s = EngagementCollectorScheduler::new(
            "p",
            Duration::from_secs(3600),
            500, // clamped to 50
            producer,
            "test",
        );
        // 50% jitter at 3600s base → [1800, 5400].
        for _ in 0..20 {
            let n = s.jittered_interval().as_secs();
            assert!((1_800..=5_400).contains(&n));
        }
    }
}
```

### Step 3: Implement `handle_engagement_refresh`

Create `crates/heartbit/src/daemon/engagement_refresh_handler.rs`:

```rust
//! Free-function handler for `DaemonCommand::EngagementRefresh`. Wires
//! the `refresh_engagement` helper into the daemon's consumer loop.

use std::sync::Arc;

use heartbit_ghost::posts::{EngagementStore, PostHistoryStore, RefreshError, refresh_engagement};
use heartbit_ghost::tools::client::XClient;

/// Dependencies for one `handle_engagement_refresh` invocation.
pub struct EngagementRefreshDeps<'a> {
    pub persona: &'a str,
    pub client: &'a XClient,
    pub history: &'a dyn PostHistoryStore,
    pub store: &'a dyn EngagementStore,
    pub max_age_days: i64,
    pub min_age_hours: i64,
}

pub async fn handle_engagement_refresh(
    deps: EngagementRefreshDeps<'_>,
) -> Result<(), RefreshError> {
    let outcome = refresh_engagement(
        deps.client,
        deps.history,
        deps.store,
        deps.persona,
        chrono::Utc::now(),
        deps.max_age_days,
        deps.min_age_hours,
    )
    .await?;
    tracing::info!(
        persona = %deps.persona,
        queried = outcome.queried,
        refreshed = outcome.refreshed,
        skipped_too_young = outcome.skipped_too_young,
        skipped_too_old = outcome.skipped_too_old,
        failed = outcome.failed,
        "engagement refresh complete"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_ghost::posts::{
        InMemoryEngagementStore, InMemoryPostHistoryStore, PostHistoryEntry, PostOutcome,
    };
    use heartbit_core::Secret;
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_client(uri: &str) -> XClient {
        XClient::new(uri, Secret::new("ck"), Secret::new("cs"), Secret::new("at"), Secret::new("ats")).unwrap()
    }

    #[tokio::test]
    async fn handler_invokes_refresh_and_returns_ok_on_empty_history() {
        let history = InMemoryPostHistoryStore::new();
        let store = InMemoryEngagementStore::new();
        let server = MockServer::start().await;
        let client = test_client(&server.uri());

        let deps = EngagementRefreshDeps {
            persona: "p",
            client: &client,
            history: &history,
            store: &store,
            max_age_days: 30,
            min_age_hours: 24,
        };
        handle_engagement_refresh(deps).await.expect("ok on empty");
    }

    #[tokio::test]
    async fn handler_records_snapshot_for_eligible_tweet() {
        let history = InMemoryPostHistoryStore::new();
        history
            .record(
                "p",
                PostHistoryEntry {
                    posted_at: chrono::Utc::now() - chrono::Duration::hours(48),
                    topic: "topic_a".into(),
                    outcome: PostOutcome::Posted {
                        chosen_index: 0,
                        url: "https://x.com/u/status/tA".into(),
                    },
                    tweet_id: Some("tA".into()),
                    text: Some("hello".into()),
                },
            )
            .await
            .unwrap();
        let store = InMemoryEngagementStore::new();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{
                    "id": "tA",
                    "public_metrics": {"like_count": 5, "reply_count": 1, "retweet_count": 0, "quote_count": 0, "bookmark_count": 1, "impression_count": 50}
                }]
            })))
            .mount(&server)
            .await;
        let client = test_client(&server.uri());

        let deps = EngagementRefreshDeps {
            persona: "p",
            client: &client,
            history: &history,
            store: &store,
            max_age_days: 30,
            min_age_hours: 24,
        };
        handle_engagement_refresh(deps).await.unwrap();
        let latest = store.latest_per_tweet().await.unwrap();
        assert_eq!(latest["tA"].likes, 5);
    }
}
```

### Step 4: Wire into mod.rs

Modify `crates/heartbit/src/daemon/mod.rs`:

```rust
pub mod engagement_collector;
pub mod engagement_refresh_handler;

pub use engagement_collector::EngagementCollectorScheduler;
pub use engagement_refresh_handler::{EngagementRefreshDeps, handle_engagement_refresh};
```

### Step 5: Wire into `posts_context.rs` (extend `PersonaPostEntry`)

Modify `crates/heartbit/src/daemon/posts_context.rs`:

```rust
use std::sync::Arc;
use std::time::Duration;
use heartbit_ghost::posts::EngagementStore;

pub struct PersonaPostEntry {
    // ... existing fields ...
    /// Engagement collector interval (default 6h).
    pub engagement_refresh: Duration,
    /// Jitter percentage for the engagement collector. Default 25.
    pub engagement_jitter_pct: u32,
    /// Engagement store (JSONL or in-memory).
    pub engagement_store: Arc<dyn EngagementStore>,
    /// Skip tweets younger than this. Default 24h.
    pub engagement_min_age_hours: i64,
    /// Skip tweets older than this. Default 30d.
    pub engagement_max_age_days: i64,
    /// How many top-engaged posts to inject. Default 5. `0` disables injection.
    pub engagement_top_n: usize,
}
```

(The Debug impl below it needs corresponding `.field(...)` lines.)

### Step 6: Wire into `core.rs` — spawn scheduler + route command

In `core.rs`, in the same `if let Some(ctx) = self.posts_context.as_ref()` block where the `PersonaPostScheduler` is spawned, also spawn:

```rust
                use crate::daemon::EngagementCollectorScheduler;

                let scheduler = EngagementCollectorScheduler::new(
                    persona.clone(),
                    entry.engagement_refresh,
                    entry.engagement_jitter_pct,
                    Arc::new(crate::daemon::kafka::KafkaCommandProducer::new(self.producer.clone()))
                        as Arc<dyn super::CommandProducer>,
                    self.commands_topic.clone(),
                );
                let cancel_for_task = self.cancel.child_token();
                tokio::spawn(scheduler.run(cancel_for_task));
                tracing::info!(persona = %persona, "engagement collector spawned");
```

In the consumer-loop `match cmd { ... }` block (parallel to `PersonaPost`), add:

```rust
                        DaemonCommand::EngagementRefresh { persona } => {
                            let Some(ctx) = self.posts_context.clone() else {
                                tracing::warn!(persona = %persona, "EngagementRefresh: no posts_context");
                                continue;
                            };
                            let Some(entry) = ctx.entries.get(&persona) else {
                                tracing::warn!(persona = %persona, "EngagementRefresh for unknown persona");
                                continue;
                            };
                            // We need an XClient — reuse the one from MentionContext if both are
                            // configured (single shared OAuth1 client per daemon).
                            let Some(ref mc) = self.mention_context else {
                                tracing::warn!(persona = %persona, "EngagementRefresh: no mention_context (no XClient)");
                                continue;
                            };
                            let Some(client_arc) = mc.enricher.clone() else {
                                tracing::warn!(persona = %persona, "EngagementRefresh: no XClient enricher attached");
                                continue;
                            };
                            let history = entry.history.clone();
                            let engagement = entry.engagement_store.clone();
                            let max_age = entry.engagement_max_age_days;
                            let min_age = entry.engagement_min_age_hours;
                            let persona_owned = persona.clone();
                            tokio::spawn(async move {
                                let deps = crate::daemon::EngagementRefreshDeps {
                                    persona: &persona_owned,
                                    client: client_arc.as_ref(),
                                    history: history.as_ref(),
                                    store: engagement.as_ref(),
                                    max_age_days: max_age,
                                    min_age_hours: min_age,
                                };
                                if let Err(e) = crate::daemon::handle_engagement_refresh(deps).await {
                                    tracing::error!(persona = %persona_owned, error = %e, "engagement refresh failed");
                                }
                            });
                        }
```

### Step 7: Run tests — verify pass

Run: `cargo test -p heartbit --lib --features daemon engagement_collector`
Run: `cargo test -p heartbit --lib --features daemon engagement_refresh_handler`
Expected: 2 + 2 tests pass.

### Step 8: Quality gate

Run: `cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: clean.

### Step 9: Commit

```bash
git add -A
git commit -m "feat(daemon): EngagementCollectorScheduler + EngagementRefresh handler

New scheduler fires DaemonCommand::EngagementRefresh on a jittered cadence
(default 6h ±25%). Handler invokes refresh_engagement and logs the outcome.
Shares the XClient from MentionContext (single OAuth1 client per daemon)."
```

---

## Task 4 — Record `text` in `PostHistoryEntry` at post time

**Files:**
- Modify: `crates/heartbit/src/daemon/persona_post_handler.rs` — on `Posted` outcome, set `text: Some(first_tweet_text)`

### Step 1: Find where the `Posted` outcome is recorded

In `persona_post_handler.rs`, near the bottom of the success path, locate the `history.record(persona, PostHistoryEntry { ... })` call for the `Posted` case. The first tweet's text is available in scope as part of the chosen candidate.

### Step 2: Write the failing test

In `crates/heartbit/src/daemon/persona_post_handler.rs::tests`, add:

```rust
    #[tokio::test]
    async fn happy_path_records_text_on_posted_entry() {
        // Reuse the existing happy_path_runs_pipeline_and_records test scaffold.
        // After the pipeline returns Posted, assert that the entry in the
        // history store has text = Some("..."), matching the first tweet
        // of the chosen candidate.
        let history = InMemoryPostHistoryStore::new();
        // ... wire deps as in happy_path_runs_pipeline_and_records ...
        // ... run handler ...
        let recent = history.recent("stub:x", 5).await.unwrap();
        let posted = recent
            .iter()
            .find(|e| matches!(e.outcome, PostOutcome::Posted { .. }))
            .expect("one Posted entry");
        assert!(posted.text.is_some(), "Posted entry must carry text");
        assert!(!posted.text.as_deref().unwrap().is_empty());
    }
```

### Step 3: Run — verify failure

Run: `cargo test -p heartbit --lib --features daemon happy_path_records_text_on_posted_entry`
Expected: assertion fails — current code records `text: None`.

### Step 4: Implement: pass the first tweet text through

Two changes:

(a) The pipeline produces a `Vec<String>` of candidate threads; the chosen index gives us the candidate. Each candidate is a thread; the **first** tweet of the chosen thread is the exemplar text. Make sure that's in scope when recording.

(b) When constructing the `PostHistoryEntry { ... }` literal for the `Posted` case, add `text: Some(first_tweet_text.clone()),`.

### Step 5: Run tests — verify pass

Run: `cargo test -p heartbit --lib --features daemon persona_post_handler`
Expected: all tests pass including the new one.

### Step 6: Quality gate + commit

```bash
git add -A
git commit -m "feat(daemon): record first-tweet text on Posted history entries

Lets TopPostsProvider render exemplars without round-tripping the X API."
```

---

## Task 5 — `TopPostsProvider` + writer injection + CLI plumbing

**Files:**
- Modify: `crates/heartbit-ghost/src/posts/engagement.rs` — `TopPost`, `TopPostsProvider` trait, `JoinedTopPostsProvider`
- Modify: `crates/heartbit-ghost/src/posts/mod.rs` — re-exports
- Modify: `crates/heartbit-core/src/config/daemon.rs` — 4 new config fields with defaults
- Modify: `crates/heartbit/src/daemon/persona_post_handler.rs` — prepend exemplar block to writer user message
- Modify: `crates/heartbit/src/daemon/posts_context.rs` — `PersonaPostEntry.top_posts_provider: Option<Arc<dyn TopPostsProvider>>`
- Modify: `crates/heartbit-cli/src/daemon/mod.rs` — construct `JsonlEngagementStore` + `JoinedTopPostsProvider`

### Step 1: Write the failing tests for `JoinedTopPostsProvider`

Add to `engagement.rs::tests`:

```rust
    fn snap_with(tweet_id: &str, likes: u64, replies: u64, retweets: u64, quotes: u64, impressions: Option<u64>) -> EngagementSnapshot {
        EngagementSnapshot {
            tweet_id: tweet_id.into(), captured_at: Utc::now(),
            likes, replies, retweets, quotes, bookmarks: 0, impressions,
        }
    }

    #[tokio::test]
    async fn top_n_orders_by_composite_score_descending() {
        let history = InMemoryPostHistoryStore::new();
        for id in ["a", "b", "c"] {
            history.record("p", PostHistoryEntry {
                posted_at: Utc::now() - Duration::hours(48),
                topic: id.into(),
                outcome: PostOutcome::Posted { chosen_index: 0, url: format!("u/{id}") },
                tweet_id: Some(id.into()),
                text: Some(format!("text_{id}")),
            }).await.unwrap();
        }
        let store = InMemoryEngagementStore::new();
        // a: 10 + 0 + 0 + 0 = 10
        // b: 5 + 3*5 + 0 + 0 = 20  (replies-heavy)
        // c: 100 + 0 + 0 + 0 = 100 (likes-heavy)
        store.record(snap_with("a", 10, 0, 0, 0, None)).await.unwrap();
        store.record(snap_with("b", 5, 5, 0, 0, None)).await.unwrap();
        store.record(snap_with("c", 100, 0, 0, 0, None)).await.unwrap();

        let provider = JoinedTopPostsProvider::new(
            Arc::new(history),
            Arc::new(store),
            "p".to_string(),
        );
        let top = provider.top_n(3).await.unwrap();
        let ids: Vec<&str> = top.iter().map(|p| p.tweet_id.as_str()).collect();
        assert_eq!(ids, vec!["c", "b", "a"]);
    }

    #[tokio::test]
    async fn top_n_skips_entries_without_text() {
        let history = InMemoryPostHistoryStore::new();
        // Entry "no_text" has tweet_id but no text — must be skipped.
        history.record("p", PostHistoryEntry {
            posted_at: Utc::now() - Duration::hours(48),
            topic: "x".into(),
            outcome: PostOutcome::Posted { chosen_index: 0, url: "u/x".into() },
            tweet_id: Some("no_text".into()),
            text: None,
        }).await.unwrap();
        history.record("p", PostHistoryEntry {
            posted_at: Utc::now() - Duration::hours(48),
            topic: "y".into(),
            outcome: PostOutcome::Posted { chosen_index: 0, url: "u/y".into() },
            tweet_id: Some("with_text".into()),
            text: Some("hello".into()),
        }).await.unwrap();
        let store = InMemoryEngagementStore::new();
        store.record(snap_with("no_text", 100, 0, 0, 0, None)).await.unwrap();
        store.record(snap_with("with_text", 5, 0, 0, 0, None)).await.unwrap();

        let provider = JoinedTopPostsProvider::new(Arc::new(history), Arc::new(store), "p".to_string());
        let top = provider.top_n(5).await.unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].tweet_id, "with_text");
    }

    #[tokio::test]
    async fn top_n_returns_empty_on_no_engagement_data() {
        let history = InMemoryPostHistoryStore::new();
        let store = InMemoryEngagementStore::new();
        let provider = JoinedTopPostsProvider::new(Arc::new(history), Arc::new(store), "p".to_string());
        assert!(provider.top_n(5).await.unwrap().is_empty());
    }
```

### Step 2: Run — verify compile failure

Expected: `TopPost`, `TopPostsProvider`, `JoinedTopPostsProvider` not defined.

### Step 3: Implement `TopPost` + `TopPostsProvider` + `JoinedTopPostsProvider`

Append to `engagement.rs`:

```rust
use crate::posts::PostHistoryStore;

/// One ranked post returned by `TopPostsProvider`. Sufficient to render
/// as a writer-prompt exemplar without further lookups.
#[derive(Debug, Clone)]
pub struct TopPost {
    pub tweet_id: String,
    pub text: String,
    pub posted_at: DateTime<Utc>,
    pub engagement_score: f64,
}

/// Rank a persona's recent Posted history by composite engagement score.
pub trait TopPostsProvider: Send + Sync {
    fn top_n<'a>(
        &'a self,
        n: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<TopPost>, EngagementStoreError>> + Send + 'a>>;
}

/// Standard impl: joins `PostHistoryStore::recent` (for text + posted_at)
/// against `EngagementStore::latest_per_tweet` (for metrics).
pub struct JoinedTopPostsProvider {
    history: std::sync::Arc<dyn PostHistoryStore>,
    engagement: std::sync::Arc<dyn EngagementStore>,
    persona: String,
}

impl JoinedTopPostsProvider {
    pub fn new(
        history: std::sync::Arc<dyn PostHistoryStore>,
        engagement: std::sync::Arc<dyn EngagementStore>,
        persona: String,
    ) -> Self {
        Self { history, engagement, persona }
    }
}

fn composite_score(snap: &EngagementSnapshot) -> f64 {
    let base = snap.likes as f64
        + 3.0 * (snap.replies as f64)
        + 2.0 * (snap.retweets as f64)
        + 2.0 * (snap.quotes as f64);
    base + snap.impressions.map(|i| 0.0001 * (i as f64)).unwrap_or(0.0)
}

impl TopPostsProvider for JoinedTopPostsProvider {
    fn top_n<'a>(
        &'a self,
        n: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<TopPost>, EngagementStoreError>> + Send + 'a>> {
        Box::pin(async move {
            if n == 0 {
                return Ok(Vec::new());
            }
            // Pull a generous slice of history; we filter to Posted-with-text below.
            let history_entries = self
                .history
                .recent(&self.persona, 1_000)
                .await
                .map_err(|e| EngagementStoreError::Parse(format!("history: {e}")))?;
            let mut text_by_id: std::collections::HashMap<String, (String, DateTime<Utc>)> =
                std::collections::HashMap::new();
            for e in history_entries {
                if !matches!(e.outcome, PostOutcome::Posted { .. }) {
                    continue;
                }
                let (Some(id), Some(text)) = (e.tweet_id, e.text) else {
                    continue;
                };
                if text.is_empty() {
                    continue;
                }
                text_by_id.insert(id, (text, e.posted_at));
            }

            let snapshots = self.engagement.latest_per_tweet().await?;
            let mut ranked: Vec<TopPost> = snapshots
                .into_iter()
                .filter_map(|(id, snap)| {
                    let (text, posted_at) = text_by_id.remove(&id)?;
                    Some(TopPost {
                        tweet_id: id,
                        text,
                        posted_at,
                        engagement_score: composite_score(&snap),
                    })
                })
                .collect();
            ranked.sort_by(|a, b| {
                b.engagement_score
                    .partial_cmp(&a.engagement_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.posted_at.cmp(&a.posted_at)) // tie: recency wins
            });
            ranked.truncate(n);
            Ok(ranked)
        })
    }
}
```

Re-export from `crates/heartbit-ghost/src/posts/mod.rs`:

```rust
pub use engagement::{
    JoinedTopPostsProvider, RefreshError, RefreshOutcome, TopPost, TopPostsProvider,
    refresh_engagement,
};
```

### Step 4: Run tests — verify pass

Run: `cargo test -p heartbit-ghost --lib posts::engagement`
Expected: all 3 new tests pass + 11 from prior tasks.

### Step 5: Add config fields

Modify `crates/heartbit-core/src/config/daemon.rs`. Inside `PersonaPostsConfig`:

```rust
    /// Engagement-collector tick interval (seconds). Default 21600 = 6h.
    #[serde(default = "default_engagement_refresh_seconds")]
    pub engagement_refresh_seconds: u64,

    /// How many top-engaged posts to inject as writer exemplars. Default 5.
    /// Set to 0 to disable the few-shot injection.
    #[serde(default = "default_engagement_top_n")]
    pub engagement_top_n: usize,

    /// Ignore tweets older than this many days. Default 30.
    #[serde(default = "default_engagement_max_age_days")]
    pub engagement_max_age_days: i64,

    /// Ignore tweets younger than this many hours. Default 24.
    #[serde(default = "default_engagement_min_age_hours")]
    pub engagement_min_age_hours: i64,
```

Defaults:

```rust
fn default_engagement_refresh_seconds() -> u64 { 21600 }
fn default_engagement_top_n() -> usize { 5 }
fn default_engagement_max_age_days() -> i64 { 30 }
fn default_engagement_min_age_hours() -> i64 { 24 }
```

### Step 6: Update `PersonaPostEntry` to carry the provider + config

In `crates/heartbit/src/daemon/posts_context.rs`:

```rust
use heartbit_ghost::posts::TopPostsProvider;

pub struct PersonaPostEntry {
    // ... existing ...
    pub top_posts_provider: Option<Arc<dyn TopPostsProvider>>,
    // (engagement_store + cadence fields already added in Task 3 step 5)
}
```

### Step 7: Wire writer injection in `persona_post_handler`

Just before the writer agent invocation, build the exemplar block and prepend to user_message:

```rust
let exemplar_block: String = match deps.top_posts_provider.as_deref() {
    Some(provider) if deps.top_n > 0 => match provider.top_n(deps.top_n).await {
        Ok(top) if top.len() >= 3 => {
            let now = chrono::Utc::now();
            let mut s = String::from(
                "EXEMPLARS — your highest-engaged posts from the last 30 days.\n\
                 Study the voice, structure, and angle. Do NOT copy literally.\n\n",
            );
            for p in &top {
                let age = (now - p.posted_at).num_days();
                s.push_str(&format!(
                    "[{} days ago, engagement score {:.0}]\n{}\n\n",
                    age, p.engagement_score, p.text
                ));
            }
            s.push_str("---\n\n");
            s
        }
        _ => String::new(),
    },
    _ => String::new(),
};
let user_message = format!("{exemplar_block}{user_message}");
```

`deps.top_posts_provider` and `deps.top_n` are new fields on `PersonaPostDeps`. Add them, defaulting to `None` and `0` in tests that don't exercise the feature.

### Step 8: Write the integration test (writer sees the exemplar block)

```rust
    #[tokio::test]
    async fn writer_receives_exemplar_block_when_top_posts_present() {
        // Seed an InMemoryEngagementStore + InMemoryPostHistoryStore with 3
        // Posted entries (each with text + tweet_id) and 3 EngagementSnapshots
        // such that JoinedTopPostsProvider returns 3 TopPosts.
        // Wire the provider into PersonaPostDeps with top_n = 5.
        // Use a MockProvider that captures the writer's last user message.
        // Assert the captured message starts with "EXEMPLARS —".
    }

    #[tokio::test]
    async fn writer_unaffected_when_fewer_than_three_exemplars() {
        // Seed only 1 entry. Run handler. Assert writer's last user message
        // does NOT contain "EXEMPLARS".
    }
```

### Step 9: CLI plumbing

In `crates/heartbit-cli/src/daemon/mod.rs`, in `build_mention_context` (or wherever `PersonaPostEntry` is constructed in the posts loop near line 280):

```rust
// Build the engagement store (JSONL when configured, in-memory otherwise).
let engagement_store: Arc<dyn heartbit_ghost::posts::EngagementStore> = match cfg.post_history_store.as_str() {
    "jsonl" => {
        // Co-locate with post history in .heartbit/engagement/
        let raw_path = format!(".heartbit/engagement/{}.jsonl", cfg.persona.replace(':', "-"));
        let path = expand_tilde(&raw_path)?;
        Arc::new(
            heartbit_ghost::posts::JsonlEngagementStore::open(&path)
                .await
                .with_context(|| format!("open engagement jsonl at {}", path.display()))?,
        )
    }
    _ => Arc::new(heartbit_ghost::posts::InMemoryEngagementStore::new()),
};

let top_posts_provider: Arc<dyn heartbit_ghost::posts::TopPostsProvider> = Arc::new(
    heartbit_ghost::posts::JoinedTopPostsProvider::new(
        history.clone(),
        engagement_store.clone(),
        cfg.persona.clone(),
    ),
);
```

Then in `PersonaPostEntry { ... }`:

```rust
engagement_refresh: Duration::from_secs(cfg.engagement_refresh_seconds),
engagement_jitter_pct: cfg.interval_jitter_pct, // reuse the post-jitter pct
engagement_store,
engagement_min_age_hours: cfg.engagement_min_age_hours,
engagement_max_age_days: cfg.engagement_max_age_days,
engagement_top_n: cfg.engagement_top_n,
top_posts_provider: Some(top_posts_provider),
```

### Step 10: Quality gate

Run: `cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: clean.

### Step 11: Commit

```bash
git add -A
git commit -m "feat(ghost): writer few-shot injection from top-engaged history

JoinedTopPostsProvider joins PostHistoryStore + EngagementStore, ranks by
composite score (3*replies + 2*retweets + 2*quotes + likes + 0.0001*impressions),
returns top N (default 5). persona_post_handler prepends an exemplar block
to the writer's user_message when >= 3 exemplars are available. Cold-start
(< 3 exemplars) preserves existing behavior. CLI wires JsonlEngagementStore
co-located with post history."
```

---

## Final Steps

After all 5 tasks complete:

1. **End-to-end smoke test against the running daemon**:
   - Restart `heartbit-daemon` so it picks up the new build with engagement collector spawned
   - Verify journal shows `engagement collector spawned persona=heartbit-ghost:x`
   - Optionally trigger an immediate refresh by publishing `{"type":"engagement_refresh","persona":"heartbit-ghost:x"}` to the `heartbit.commands` Kafka topic; watch for the `engagement refresh complete queried=N refreshed=M` INFO line
   - After the next proactive post, verify the daemon log shows the exemplar count (add a `tracing::info!("post: %d exemplars injected", count)` if useful)

2. **Push** to `origin/main`.

3. **Operational note**: legacy `PostHistoryEntry`s recorded before this PR have `text: None` and won't appear as exemplars until they age out of the 30-day window. New posts written after the upgrade will accumulate correctly. This is acceptable cold-start behavior.
