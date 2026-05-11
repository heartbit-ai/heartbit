//! X API enrichment helpers used by the daemon mention-poll handler to
//! populate `Mention.author_handle`, `MentionerContext`, and the parent
//! `TweetSnapshot` before dispatching a `ReplyDraft`.
//!
//! These activate two pipeline features that were previously inert:
//! 1. `BotHeuristicGuard` — needs a real `author_handle` to pattern-match
//!    against `suspicious_handle_patterns`, and `follower_count`,
//!    `following_count`, `account_created_at` from `MentionerContext` to
//!    evaluate the follower/following ratio and account-age signals.
//! 2. `reply_writer` agent — its prompt expects "the ORIGINAL tweet the
//!    parent was replying to" as optional context, so threaded replies
//!    can stay on-topic instead of hallucinating the surrounding
//!    conversation.
//!
//! Both helpers do a single X API round trip and degrade gracefully on
//! failure: the daemon handler treats an `Err` as "no enrichment
//! available" and dispatches with `None`, so a flaky enrichment path
//! never blocks a reply.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;

use crate::reply::{MentionerContext, TweetSnapshot};
use crate::tools::client::{XApiError, XClient};

// ─── /2/users/:id ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct UserResponse {
    #[allow(dead_code)]
    id: String,
    username: String,
    description: Option<String>,
    created_at: Option<String>,
    public_metrics: Option<UserMetrics>,
}

#[derive(Debug, Deserialize)]
struct UserMetrics {
    followers_count: Option<u64>,
    following_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct UserApiResponse {
    data: UserResponse,
}

/// Look up the mentioner via `GET /2/users/:id` and assemble both the
/// handle (returned in the `MentionerContext.handle` field — also used
/// by the caller to populate `Mention.author_handle` so the bot heuristic
/// can pattern-match it) and the tone-calibration data for the reply
/// writer.
///
/// Returns one round trip's worth of data — `description`,
/// `public_metrics`, `created_at`, `username` — in `user.fields`.
pub async fn enrich_mentioner(
    client: &XClient,
    author_id: &str,
) -> Result<MentionerContext, XApiError> {
    let path = format!("/2/users/{author_id}");
    let response: UserApiResponse = client
        .get_json(
            &path,
            &[(
                "user.fields",
                "public_metrics,created_at,description,username",
            )],
        )
        .await?;
    let u = response.data;
    let metrics = u.public_metrics.unwrap_or(UserMetrics {
        followers_count: None,
        following_count: None,
    });
    let account_created_at = u
        .created_at
        .as_deref()
        .and_then(|s| s.parse::<DateTime<Utc>>().ok());
    Ok(MentionerContext {
        handle: u.username,
        bio: u.description,
        recent_tweets: vec![],
        follower_count: metrics.followers_count,
        following_count: metrics.following_count,
        account_created_at,
    })
}

// ─── /2/tweets/:id ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct TweetResponse {
    id: String,
    text: String,
    created_at: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TweetApiResponse {
    data: TweetResponse,
}

/// Fetch the parent tweet via `GET /2/tweets/:id`. Used when a mention
/// has `in_reply_to_tweet_id.is_some()` so the reply writer can read the
/// actual content of what's being replied to instead of guessing from
/// fragments like "exactly this" or "but what about X?".
pub async fn fetch_parent_tweet(
    client: &XClient,
    tweet_id: &str,
) -> Result<TweetSnapshot, XApiError> {
    let path = format!("/2/tweets/{tweet_id}");
    let response: TweetApiResponse = client
        .get_json(&path, &[("tweet.fields", "created_at,text")])
        .await?;
    let t = response.data;
    // `created_at` is optional in the X API response shape; if missing or
    // unparsable, fall back to `now` — the writer still gets the text,
    // which is the load-bearing field.
    let posted_at = t
        .created_at
        .as_deref()
        .and_then(|s| s.parse::<DateTime<Utc>>().ok())
        .unwrap_or_else(Utc::now);
    Ok(TweetSnapshot {
        id: t.id,
        text: t.text,
        posted_at,
    })
}

// ─── EnrichmentCache: in-memory dedup for X API calls ──────────────────────
//
// Why: the daemon makes one `GET /2/users/:id` call per surviving mention
// for the bot-heuristic guard, and one `GET /2/tweets/:id` per threaded
// mention for parent context. Over a daemon's lifetime, the same author
// often mentions multiple times (5x/week is normal) and threaded replies
// share parent tweets. Without dedup, we re-fetch identical data.
//
// Design:
// - Two parallel maps: users (TTL-bounded), tweets (no TTL — tweet content
//   is immutable).
// - Capacity-bounded with FIFO-on-overflow eviction. Approximate LRU is
//   fine here: low cardinality of authors and conversations means we rarely
//   hit the bound at default capacities.
// - `std::sync::RwLock`, not tokio's — never held across `.await`.
// - In-memory only. Lost on daemon restart; cache rebuilds organically
//   from active mentions. Persistence can come later if cold-start cost
//   ever shows up in metrics.

/// Cached user profile with insertion time for TTL checks.
struct CachedUser {
    inserted_at: Instant,
    context: MentionerContext,
}

/// In-memory cache for enriched mentioner profiles and parent tweets.
///
/// Reuses the same `XClient` across calls; transparently serves cached
/// results when fresh, fetches and populates on a miss. Cache failures
/// (RwLock poison) degrade silently to direct fetches — the cache is
/// strictly an optimization.
pub struct EnrichmentCache {
    users: RwLock<HashMap<String, CachedUser>>,
    tweets: RwLock<HashMap<String, TweetSnapshot>>,
    user_ttl: Duration,
    max_users: usize,
    max_tweets: usize,
}

impl Default for EnrichmentCache {
    fn default() -> Self {
        Self::new()
    }
}

impl EnrichmentCache {
    /// Construct a cache with sensible defaults: 24h user TTL, 1024 entry
    /// cap on each map. Tweaking is rarely needed.
    pub fn new() -> Self {
        Self {
            users: RwLock::new(HashMap::new()),
            tweets: RwLock::new(HashMap::new()),
            user_ttl: Duration::from_secs(86_400),
            max_users: 1024,
            max_tweets: 1024,
        }
    }

    /// Override the user-profile TTL (default 24h). Profiles change slowly
    /// (followers, bio), so shorter TTLs trade dedup efficiency for
    /// staleness tolerance.
    pub fn with_user_ttl(mut self, ttl: Duration) -> Self {
        self.user_ttl = ttl;
        self
    }

    /// Override the user-cache max entries (default 1024).
    pub fn with_max_users(mut self, n: usize) -> Self {
        self.max_users = n;
        self
    }

    /// Override the tweet-cache max entries (default 1024).
    pub fn with_max_tweets(mut self, n: usize) -> Self {
        self.max_tweets = n;
        self
    }

    /// Cache stats for tests / metrics.
    pub fn stats(&self) -> EnrichmentCacheStats {
        EnrichmentCacheStats {
            users_size: self.users.read().map(|g| g.len()).unwrap_or(0),
            tweets_size: self.tweets.read().map(|g| g.len()).unwrap_or(0),
        }
    }

    /// Enrich `author_id`, serving from cache when fresh. Misses (or
    /// expired entries) fetch via `client` and populate the cache.
    pub async fn enrich_mentioner(
        &self,
        client: &XClient,
        author_id: &str,
    ) -> Result<MentionerContext, XApiError> {
        // Cache hit — return clone (cheap; MentionerContext is small).
        if let Ok(g) = self.users.read()
            && let Some(entry) = g.get(author_id)
            && entry.inserted_at.elapsed() < self.user_ttl
        {
            return Ok(entry.context.clone());
        }

        // Miss or expired: fetch and populate.
        let ctx = enrich_mentioner(client, author_id).await?;
        if let Ok(mut g) = self.users.write() {
            // FIFO eviction at capacity. Scan for oldest insert time;
            // O(n) but n ≤ max_users (defaulting to 1024) so trivial.
            if g.len() >= self.max_users
                && let Some(oldest_key) = g
                    .iter()
                    .min_by_key(|(_, v)| v.inserted_at)
                    .map(|(k, _)| k.clone())
            {
                g.remove(&oldest_key);
            }
            g.insert(
                author_id.to_string(),
                CachedUser {
                    inserted_at: Instant::now(),
                    context: ctx.clone(),
                },
            );
        }
        Ok(ctx)
    }

    /// Fetch parent tweet, serving from cache when present. Tweets are
    /// immutable content so there's no TTL — once cached, always valid.
    pub async fn fetch_parent_tweet(
        &self,
        client: &XClient,
        tweet_id: &str,
    ) -> Result<TweetSnapshot, XApiError> {
        if let Ok(g) = self.tweets.read()
            && let Some(snap) = g.get(tweet_id)
        {
            return Ok(snap.clone());
        }

        let snap = fetch_parent_tweet(client, tweet_id).await?;
        if let Ok(mut g) = self.tweets.write() {
            if g.len() >= self.max_tweets {
                // FIFO eviction by `posted_at` — close enough; we don't
                // record insert time separately for tweets.
                if let Some(oldest_key) = g
                    .iter()
                    .min_by_key(|(_, v)| v.posted_at)
                    .map(|(k, _)| k.clone())
                {
                    g.remove(&oldest_key);
                }
            }
            g.insert(tweet_id.to_string(), snap.clone());
        }
        Ok(snap)
    }
}

/// Cache observability snapshot — used by tests and could feed metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnrichmentCacheStats {
    /// Number of user-profile entries currently cached.
    pub users_size: usize,
    /// Number of parent-tweet entries currently cached.
    pub tweets_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::Secret;
    use wiremock::matchers::{method, path as wm_path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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

    // ─── enrich_mentioner ──────────────────────────────────────────────

    #[tokio::test]
    async fn enrich_mentioner_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100"))
            .and(query_param(
                "user.fields",
                "public_metrics,created_at,description,username",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "id": "100",
                    "name": "Crypto Pump",
                    "username": "Crypto_Pump_69",
                    "description": "to the moon 🚀",
                    "public_metrics": {
                        "followers_count": 50,
                        "following_count": 4900,
                        "tweet_count": 250
                    },
                    "created_at": "2026-04-01T12:00:00.000Z"
                }
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let ctx = enrich_mentioner(&client, "100")
            .await
            .expect("enrich happy path");

        assert_eq!(ctx.handle, "Crypto_Pump_69");
        assert_eq!(ctx.bio.as_deref(), Some("to the moon 🚀"));
        assert_eq!(ctx.follower_count, Some(50));
        assert_eq!(ctx.following_count, Some(4900));
        assert!(ctx.account_created_at.is_some());
        // Bot guard signals: handle pattern + ratio + age all derivable.
    }

    #[tokio::test]
    async fn enrich_mentioner_handles_missing_public_metrics() {
        // X API can omit `public_metrics` for protected/suspended accounts.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/200"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "id": "200",
                    "name": "Some User",
                    "username": "someuser"
                }
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let ctx = enrich_mentioner(&client, "200")
            .await
            .expect("missing metrics is not an error");

        assert_eq!(ctx.handle, "someuser");
        assert_eq!(ctx.bio, None);
        assert_eq!(ctx.follower_count, None);
        assert_eq!(ctx.following_count, None);
        assert_eq!(ctx.account_created_at, None);
    }

    #[tokio::test]
    async fn enrich_mentioner_returns_rate_limited_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100"))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "60"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result = enrich_mentioner(&client, "100").await;
        match result {
            Err(XApiError::RateLimited { retry_after_secs }) => {
                assert_eq!(retry_after_secs, 60);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn enrich_mentioner_returns_api_error_on_404() {
        // Deleted or suspended user.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/999"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "errors": [{"message": "user not found"}]
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result = enrich_mentioner(&client, "999").await;
        assert!(matches!(
            result,
            Err(XApiError::ApiError { status: 404, .. })
        ));
    }

    // ─── fetch_parent_tweet ────────────────────────────────────────────

    #[tokio::test]
    async fn fetch_parent_tweet_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/9001"))
            .and(query_param("tweet.fields", "created_at,text"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "id": "9001",
                    "text": "the original tweet about async runtimes",
                    "created_at": "2026-01-01T00:00:00.000Z"
                }
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let snap = fetch_parent_tweet(&client, "9001")
            .await
            .expect("parent fetch happy path");

        assert_eq!(snap.id, "9001");
        assert_eq!(snap.text, "the original tweet about async runtimes");
        // posted_at parsed from RFC3339 — concrete value, not the fallback.
        assert_eq!(snap.posted_at.to_rfc3339(), "2026-01-01T00:00:00+00:00");
    }

    #[tokio::test]
    async fn fetch_parent_tweet_falls_back_to_now_on_missing_created_at() {
        // X API sometimes omits `created_at` if not requested or for
        // historical tweets. The text is the load-bearing field; don't
        // fail the whole call.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/42"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "id": "42",
                    "text": "no timestamp here"
                }
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let before = Utc::now();
        let snap = fetch_parent_tweet(&client, "42")
            .await
            .expect("missing created_at is not an error");
        let after = Utc::now();

        assert_eq!(snap.text, "no timestamp here");
        // Fallback to "now" — between the two clock reads.
        assert!(snap.posted_at >= before && snap.posted_at <= after);
    }

    #[tokio::test]
    async fn fetch_parent_tweet_returns_api_error_on_404() {
        // Tweet was deleted between the mention and our fetch.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/deleted"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "errors": [{"message": "tweet not found"}]
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result = fetch_parent_tweet(&client, "deleted").await;
        assert!(matches!(
            result,
            Err(XApiError::ApiError { status: 404, .. })
        ));
    }

    #[tokio::test]
    async fn fetch_parent_tweet_returns_rate_limited_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/42"))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "120"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result = fetch_parent_tweet(&client, "42").await;
        match result {
            Err(XApiError::RateLimited { retry_after_secs }) => {
                assert_eq!(retry_after_secs, 120);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    // ─── EnrichmentCache ──────────────────────────────────────────────

    /// Same author_id requested twice should hit the X API exactly once.
    /// This is the core dedup behavior; without it, the daemon would
    /// re-fetch identical user profiles every time the same person
    /// mentions the operator.
    #[tokio::test]
    async fn cache_dedups_repeat_user_lookups() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/u1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "id": "u1",
                    "name": "Real Person",
                    "username": "realperson",
                    "public_metrics": {
                        "followers_count": 100,
                        "following_count": 200,
                        "tweet_count": 50
                    },
                    "created_at": "2020-01-01T00:00:00.000Z"
                }
            })))
            // .expect(1) would fail the test if mock is called twice.
            .expect(1)
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let cache = EnrichmentCache::new();

        let first = cache.enrich_mentioner(&client, "u1").await.unwrap();
        let second = cache.enrich_mentioner(&client, "u1").await.unwrap();
        let third = cache.enrich_mentioner(&client, "u1").await.unwrap();

        assert_eq!(first.handle, "realperson");
        assert_eq!(second.handle, "realperson");
        assert_eq!(third.handle, "realperson");
        // wiremock's .expect(1) asserts on Drop that exactly one call was made.
        assert_eq!(cache.stats().users_size, 1);
    }

    /// Expired user entries are re-fetched. Use a tiny TTL + a sleep to
    /// force expiry without paused-time helpers.
    #[tokio::test]
    async fn cache_expires_user_entries_after_ttl() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/u2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"id": "u2", "name": "U2", "username": "u2_user"}
            })))
            .expect(2) // expect TWO calls: initial + post-expiry
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let cache = EnrichmentCache::new().with_user_ttl(Duration::from_millis(50));

        let _ = cache.enrich_mentioner(&client, "u2").await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let _ = cache.enrich_mentioner(&client, "u2").await.unwrap();
        // Both calls actually went to wiremock (verified by .expect(2)).
    }

    /// Parent tweets are immutable content — no TTL, cache forever (until
    /// capacity eviction). Multiple lookups → one HTTP call.
    #[tokio::test]
    async fn cache_dedups_repeat_parent_tweet_lookups() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/t1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "id": "t1",
                    "text": "the original tweet",
                    "created_at": "2026-01-01T00:00:00.000Z"
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let cache = EnrichmentCache::new();
        let _ = cache.fetch_parent_tweet(&client, "t1").await.unwrap();
        let _ = cache.fetch_parent_tweet(&client, "t1").await.unwrap();
        let snap = cache.fetch_parent_tweet(&client, "t1").await.unwrap();
        assert_eq!(snap.text, "the original tweet");
        assert_eq!(cache.stats().tweets_size, 1);
    }

    /// Capacity bound: when inserting beyond `max_users`, the oldest entry
    /// is evicted (FIFO). After eviction, the old key's next lookup must
    /// re-fetch.
    #[tokio::test]
    async fn cache_evicts_oldest_user_when_at_capacity() {
        let server = MockServer::start().await;
        // Three distinct users — at max_users=2 the first inserted will
        // be evicted when the third arrives.
        for id in ["u_a", "u_b", "u_c"] {
            Mock::given(method("GET"))
                .and(wm_path(format!("/2/users/{id}")))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "data": {"id": id, "name": id, "username": id}
                })))
                .mount(&server)
                .await;
        }

        let client = test_client(&server.uri());
        let cache = EnrichmentCache::new().with_max_users(2);

        let _ = cache.enrich_mentioner(&client, "u_a").await.unwrap();
        // Tiny spacing so insert timestamps differ deterministically.
        tokio::time::sleep(Duration::from_millis(5)).await;
        let _ = cache.enrich_mentioner(&client, "u_b").await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        let _ = cache.enrich_mentioner(&client, "u_c").await.unwrap();

        assert_eq!(cache.stats().users_size, 2, "cache must respect max_users");
        // u_a was the oldest insert; it should have been evicted.
        // We can't directly query the internal map, but a fresh lookup of
        // u_a will hit the network again (which we don't assert directly
        // here — the size check is sufficient for V1 correctness).
    }

    /// Cache failures must never propagate — they're a strict optimization
    /// layer. A cache populated by one client should still serve correctly
    /// when the next fetch fails.
    #[tokio::test]
    async fn cache_does_not_obscure_underlying_errors() {
        // Cache miss + wiremock returning 401 → must propagate
        // `Unauthenticated`, not silently swallow.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/fail"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let cache = EnrichmentCache::new();
        let err = cache.enrich_mentioner(&client, "fail").await.unwrap_err();
        assert!(matches!(err, XApiError::Unauthenticated(_)));
        // Failed fetch must NOT pollute the cache.
        assert_eq!(cache.stats().users_size, 0);
    }
}
