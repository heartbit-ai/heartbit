//! Free-function handler for `DaemonCommand::EngagementRefresh`. Wires
//! the `refresh_engagement` helper into the daemon's consumer loop.
//!
//! Pattern matches `reply_draft_handler`/`mention_poll_handler`: a
//! dependency struct + a free async function that the dispatcher arm
//! calls inside a `tokio::spawn`.

use heartbit_ghost::posts::{EngagementStore, PostHistoryStore, RefreshError, refresh_engagement};
use heartbit_ghost::tools::client::XClient;

/// Dependencies for one `handle_engagement_refresh` invocation.
///
/// Borrowed references — the handler is short-lived and the caller
/// owns the underlying `Arc`s. The lifetime parameter ties everything
/// to one dispatch.
pub struct EngagementRefreshDeps<'a> {
    /// Persona name (e.g. `"heartbit-ghost:x"`).
    pub persona: &'a str,
    /// Shared X API client (reused from `MentionContext::enricher`).
    pub client: &'a XClient,
    /// Persona's post history (read-only — refresh never writes here).
    pub history: &'a dyn PostHistoryStore,
    /// Engagement snapshot store (JSONL or in-memory).
    pub store: &'a dyn EngagementStore,
    /// Skip tweets older than this. Default 30.
    pub max_age_days: i64,
    /// Skip tweets younger than this. Default 24.
    pub min_age_hours: i64,
}

/// Drive one refresh cycle. The X API call(s) are batched inside
/// `refresh_engagement`; this handler exists only to provide the
/// dependency-glue + INFO-level outcome logging.
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
    use heartbit_core::Secret;
    use heartbit_ghost::posts::{
        InMemoryEngagementStore, InMemoryPostHistoryStore, PostHistoryEntry, PostOutcome,
    };
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_client(uri: &str) -> XClient {
        XClient::new(
            uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("test XClient must construct")
    }

    /// Empty history → handler succeeds without making any API call.
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
        handle_engagement_refresh(deps)
            .await
            .expect("ok on empty history");
    }

    /// One Posted tweet, 48h old, eligible → handler records a snapshot.
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
                    "public_metrics": {
                        "like_count": 5,
                        "reply_count": 1,
                        "retweet_count": 0,
                        "quote_count": 0,
                        "bookmark_count": 1,
                        "impression_count": 50
                    }
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
        let snap = latest.get("tA").expect("tA must have a snapshot");
        assert_eq!(snap.likes, 5);
        assert_eq!(snap.replies, 1);
        assert_eq!(snap.bookmarks, 1);
    }
}
