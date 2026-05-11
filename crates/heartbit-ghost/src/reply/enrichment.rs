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
}
