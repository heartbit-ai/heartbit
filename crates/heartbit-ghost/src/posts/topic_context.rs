//! `TopicContextProvider` — persona-specific pre-fetch strategy that
//! assembles the topic generator's input context. The agent itself is
//! a singleton (no tools); each persona declares HOW to build its
//! context block via this trait.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::CredentialResolver;

pub use heartbit_core::persona::TopicContextProvider;

use super::PostHistoryEntry;

/// Dependencies passed to [`XGhostTopicContext::build_context_inner`]
/// during pre-fetch.
pub struct TopicContextDeps<'a> {
    /// Credentials for any X API calls the provider needs (own tweets,
    /// mentions). The provider is responsible for building its own
    /// `XClient` from these.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Operator's X user_id (resolved at config load).
    pub operator_user_id: &'a str,
    /// Recent post history (most-recent-first), passed verbatim into
    /// the rendered context so the generator avoids duplicates.
    pub recent_history: Vec<PostHistoryEntry>,
}

/// X-grounded topic context for `heartbit-ghost:x`. Pre-fetches the
/// operator's own recent tweets + recent mentions and renders them
/// alongside the post history.
pub struct XGhostTopicContext {
    base_url: String,
}

impl Default for XGhostTopicContext {
    fn default() -> Self {
        Self::new()
    }
}

impl XGhostTopicContext {
    /// Production constructor. Uses the X API base URL.
    pub fn new() -> Self {
        Self {
            base_url: crate::tools::client::X_API_BASE_URL.to_string(),
        }
    }

    /// Test constructor. Lets a mock server URI override the base URL.
    #[cfg(test)]
    pub(crate) fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }

    /// Inner implementation retaining the rich `TopicContextDeps` shape.
    /// Called directly by tests; the core-trait impl decodes JSON and
    /// delegates here.
    pub(crate) async fn build_context_inner<'a>(
        &'a self,
        deps: &'a TopicContextDeps<'a>,
    ) -> Result<String, anyhow::Error> {
        // Build XClient (dedicated method so tests can override base_url).
        let client = match build_client(&self.base_url, deps.credentials.clone()).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "topic context: client build failed; returning history only");
                return Ok(render_history_only(&deps.recent_history));
            }
        };

        let own_tweets = fetch_own_tweets(&client, deps.operator_user_id).await;
        let mentions = fetch_recent_mentions(&client, deps.operator_user_id).await;

        let mut out = String::new();
        // RECENT POSTS block
        out.push_str("RECENT POSTS (yours, last 10):\n");
        match own_tweets {
            Ok(tweets) => {
                if tweets.is_empty() {
                    out.push_str("(none)\n");
                } else {
                    for t in tweets.iter().take(10) {
                        let when = t.created_at.as_deref().unwrap_or("?");
                        let preview: String = t.text.chars().take(140).collect();
                        out.push_str(&format!("- [{when}] \"{preview}\"\n"));
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "own tweets fetch failed");
                out.push_str("(unavailable: api error)\n");
            }
        }
        out.push('\n');

        // RECENT MENTIONS block
        out.push_str("RECENT MENTIONS (last 10):\n");
        match mentions {
            Ok(ms) => {
                if ms.is_empty() {
                    out.push_str("(none)\n");
                } else {
                    for m in ms.iter().take(10) {
                        let preview: String = m.text.chars().take(140).collect();
                        out.push_str(&format!("- {preview}\n"));
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "mentions fetch failed");
                out.push_str("(unavailable: api error)\n");
            }
        }
        out.push('\n');

        // RECENT POST HISTORY block
        out.push_str(&render_history_only(&deps.recent_history));
        Ok(out)
    }
}

impl TopicContextProvider for XGhostTopicContext {
    fn build_context<'a>(
        &'a self,
        operator_user_id: &'a str,
        recent_history_json: &'a str,
        credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver>,
    ) -> Pin<Box<dyn Future<Output = Result<String, anyhow::Error>> + Send + 'a>> {
        Box::pin(async move {
            let recent_history: Vec<PostHistoryEntry> =
                serde_json::from_str(recent_history_json).unwrap_or_default();
            let deps = TopicContextDeps {
                credentials,
                operator_user_id,
                recent_history,
            };
            self.build_context_inner(&deps).await
        })
    }
}

/// Repo-grounded topic context for `heartbit-rs:x`. Inspects the
/// local repo (commits, recently-modified modules) to surface fresh
/// material for the topic generator.
pub struct HeartbitRsXTopicContext {
    repo_root: std::path::PathBuf,
}

impl HeartbitRsXTopicContext {
    /// Construct from a repo root path (the same path the persona's
    /// `RepoInspectTool` uses).
    pub fn new(repo_root: std::path::PathBuf) -> Self {
        Self { repo_root }
    }
}

impl TopicContextProvider for HeartbitRsXTopicContext {
    fn build_context<'a>(
        &'a self,
        _operator_user_id: &'a str,
        recent_history_json: &'a str,
        _credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver>,
    ) -> Pin<Box<dyn Future<Output = Result<String, anyhow::Error>> + Send + 'a>> {
        Box::pin(async move {
            let recent_history: Vec<PostHistoryEntry> =
                serde_json::from_str(recent_history_json).unwrap_or_default();
            let mut out = String::new();

            // Recent commits via `git log` (out-of-process).
            out.push_str("RECENT COMMITS (last 10):\n");
            match fetch_recent_commits(&self.repo_root, 10).await {
                Ok(lines) if !lines.is_empty() => {
                    for line in lines.iter().take(10) {
                        out.push_str(&format!("- {line}\n"));
                    }
                }
                Ok(_) => out.push_str("(none)\n"),
                Err(e) => {
                    tracing::warn!(error = %e, "git log failed");
                    out.push_str("(unavailable)\n");
                }
            }
            out.push('\n');

            // Recently-modified module names (top-level under crates/).
            out.push_str("RECENTLY-MODIFIED MODULES (last 24h):\n");
            match fetch_recently_modified_modules(&self.repo_root).await {
                Ok(mods) if !mods.is_empty() => {
                    for m in mods.iter().take(10) {
                        out.push_str(&format!("- {m}\n"));
                    }
                }
                Ok(_) => out.push_str("(none)\n"),
                Err(e) => {
                    tracing::warn!(error = %e, "module scan failed");
                    out.push_str("(unavailable)\n");
                }
            }
            out.push('\n');

            // Post history block.
            out.push_str(&render_history_only(&recent_history));
            Ok(out)
        })
    }
}

// --- private helpers ----------------------------------------------------

async fn fetch_recent_commits(
    repo_root: &std::path::Path,
    n: usize,
) -> Result<Vec<String>, anyhow::Error> {
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("log")
        .arg("--oneline")
        .arg(format!("-{n}"))
        .output()
        .await?;
    if !output.status.success() {
        anyhow::bail!(
            "git log failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(String::from)
        .collect())
}

async fn fetch_recently_modified_modules(
    repo_root: &std::path::Path,
) -> Result<Vec<String>, anyhow::Error> {
    use chrono::Duration;
    use chrono::Utc;

    let cutoff = Utc::now() - Duration::hours(24);
    let cutoff_unix = cutoff.timestamp();
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("log")
        .arg(format!("--since={cutoff_unix}"))
        .arg("--name-only")
        .arg("--pretty=format:")
        .output()
        .await?;
    if !output.status.success() {
        anyhow::bail!(
            "git log --name-only failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut modules = std::collections::BTreeSet::new();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("crates/")
            && let Some(slash) = rest.find('/')
        {
            modules.insert(rest[..slash].to_string());
        }
    }
    Ok(modules.into_iter().collect())
}

async fn build_client(
    base_url: &str,
    creds: Arc<dyn CredentialResolver>,
) -> Result<crate::tools::client::XClient, crate::tools::client::XApiError> {
    use crate::tools::client::{XApiError, XClient};

    let consumer_key = creds.resolve("X_CONSUMER_KEY").await.map_err(|e| {
        XApiError::CredentialResolutionFailed {
            name: "X_CONSUMER_KEY".into(),
            source: e,
        }
    })?;
    let consumer_secret = creds.resolve("X_CONSUMER_SECRET").await.map_err(|e| {
        XApiError::CredentialResolutionFailed {
            name: "X_CONSUMER_SECRET".into(),
            source: e,
        }
    })?;
    let access_token = creds.resolve("X_ACCESS_TOKEN").await.map_err(|e| {
        XApiError::CredentialResolutionFailed {
            name: "X_ACCESS_TOKEN".into(),
            source: e,
        }
    })?;
    let access_token_secret = creds.resolve("X_ACCESS_TOKEN_SECRET").await.map_err(|e| {
        XApiError::CredentialResolutionFailed {
            name: "X_ACCESS_TOKEN_SECRET".into(),
            source: e,
        }
    })?;
    XClient::new(
        base_url,
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret,
    )
}

#[derive(Debug, serde::Deserialize)]
struct OwnTweetItem {
    #[allow(dead_code)]
    id: String,
    text: String,
    #[serde(default)]
    created_at: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct OwnTweetsResp {
    #[serde(default)]
    data: Vec<OwnTweetItem>,
}

async fn fetch_own_tweets(
    client: &crate::tools::client::XClient,
    user_id: &str,
) -> Result<Vec<OwnTweetItem>, crate::tools::client::XApiError> {
    let path = format!("/2/users/{user_id}/tweets");
    let query: Vec<(&str, &str)> = vec![
        ("max_results", "10"),
        ("tweet.fields", "created_at"),
        ("exclude", "replies,retweets"),
    ];
    let resp: OwnTweetsResp = client.get_json(&path, &query).await?;
    Ok(resp.data)
}

#[derive(Debug, serde::Deserialize)]
struct MentionItem {
    #[allow(dead_code)]
    id: String,
    text: String,
}

#[derive(Debug, serde::Deserialize)]
struct MentionsResp {
    #[serde(default)]
    data: Vec<MentionItem>,
}

async fn fetch_recent_mentions(
    client: &crate::tools::client::XClient,
    user_id: &str,
) -> Result<Vec<MentionItem>, crate::tools::client::XApiError> {
    let path = format!("/2/users/{user_id}/mentions");
    let query: Vec<(&str, &str)> = vec![("max_results", "10"), ("tweet.fields", "author_id")];
    let resp: MentionsResp = client.get_json(&path, &query).await?;
    Ok(resp.data)
}

fn render_history_only(history: &[PostHistoryEntry]) -> String {
    use super::PostOutcome;
    let mut out = String::new();
    out.push_str("RECENT POST HISTORY (last 5 from store):\n");
    if history.is_empty() {
        out.push_str("(none)\n");
    } else {
        for entry in history.iter().take(5) {
            let when = entry.posted_at.format("%Y-%m-%d");
            let outcome = match &entry.outcome {
                PostOutcome::Posted { .. } => "Posted",
                PostOutcome::Skipped => "Skipped",
                PostOutcome::TimedOut => "TimedOut",
                PostOutcome::NoTopic => "NoTopic",
                PostOutcome::SkippedDuplicate => "SkippedDuplicate",
                PostOutcome::GateRejected { .. } => "GateRejected",
                PostOutcome::PublishFailed { .. } => "PublishFailed",
            };
            let topic = if entry.topic.is_empty() {
                "(no topic)"
            } else {
                entry.topic.as_str()
            };
            out.push_str(&format!("- [{when}] {outcome}: {topic}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::{PostHistoryEntry, PostOutcome};
    use chrono::Utc;

    #[test]
    fn topic_context_deps_can_be_constructed_with_zero_history() {
        struct StubCreds;
        impl CredentialResolver for StubCreds {
            fn resolve(
                &self,
                _name: &str,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>>
                        + Send
                        + '_,
                >,
            > {
                Box::pin(async { Ok(heartbit_core::Secret::new("x")) })
            }
        }
        let creds: Arc<dyn CredentialResolver> = Arc::new(StubCreds);
        let deps = TopicContextDeps {
            credentials: creds,
            operator_user_id: "12345",
            recent_history: vec![],
        };
        assert!(deps.recent_history.is_empty());
        assert_eq!(deps.operator_user_id, "12345");
    }

    /// Mock CredentialResolver that returns canned secrets per name.
    struct CannedCreds;
    impl CredentialResolver for CannedCreds {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(heartbit_core::Secret::new("x")) })
        }
    }

    #[tokio::test]
    async fn xghost_context_assembles_recent_posts_mentions_and_history() {
        use wiremock::matchers::{method, path as wm_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/12345/tweets"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"id": "1", "text": "first own post", "created_at": "2026-05-08T00:00:00.000Z"}
                ],
                "meta": {}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/12345/mentions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {
                        "id": "9001",
                        "text": "asking about agent loops",
                        "author_id": "777",
                        "created_at": "2026-05-09T00:00:00.000Z"
                    }
                ],
                "meta": {}
            })))
            .mount(&server)
            .await;

        let creds: Arc<dyn CredentialResolver> = Arc::new(CannedCreds);
        let history = vec![PostHistoryEntry {
            posted_at: Utc::now(),
            topic: "calibrated abstention".into(),
            outcome: PostOutcome::Posted {
                chosen_index: 0,
                url: "https://x.com/i/web/status/100".into(),
            },
            tweet_id: Some("100".into()),
        }];
        let deps = TopicContextDeps {
            credentials: creds,
            operator_user_id: "12345",
            recent_history: history,
        };

        let provider = XGhostTopicContext::with_base_url(&server.uri());
        let ctx = provider
            .build_context_inner(&deps)
            .await
            .expect("happy path");
        assert!(ctx.contains("RECENT POSTS"), "context: {ctx}");
        assert!(ctx.contains("first own post"), "context: {ctx}");
        assert!(ctx.contains("RECENT MENTIONS"), "context: {ctx}");
        assert!(ctx.contains("asking about agent loops"), "context: {ctx}");
        assert!(ctx.contains("RECENT POST HISTORY"), "context: {ctx}");
        assert!(ctx.contains("calibrated abstention"), "context: {ctx}");
    }

    #[tokio::test]
    async fn xghost_context_degrades_gracefully_on_api_error() {
        use wiremock::matchers::{method, path as wm_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/12345/tweets"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/12345/mentions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data": [], "meta": {}})),
            )
            .mount(&server)
            .await;

        let creds: Arc<dyn CredentialResolver> = Arc::new(CannedCreds);
        let deps = TopicContextDeps {
            credentials: creds,
            operator_user_id: "12345",
            recent_history: vec![],
        };
        let provider = XGhostTopicContext::with_base_url(&server.uri());
        let ctx = provider.build_context_inner(&deps).await.expect("graceful");
        assert!(
            ctx.contains("RECENT POSTS")
                || ctx.contains("RECENT MENTIONS")
                || ctx.contains("RECENT POST HISTORY"),
            "context: {ctx}"
        );
        assert!(
            ctx.contains("(unavailable: api error)") || ctx.contains("(none)"),
            "context: {ctx}"
        );
    }

    #[tokio::test]
    async fn heartbit_rs_context_handles_missing_repo_gracefully() {
        let provider = HeartbitRsXTopicContext::new(std::path::PathBuf::from("/nonexistent/path"));
        struct StubCreds;
        impl CredentialResolver for StubCreds {
            fn resolve(
                &self,
                _name: &str,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>>
                        + Send
                        + '_,
                >,
            > {
                Box::pin(async { Ok(heartbit_core::Secret::new("x")) })
            }
        }
        let creds: Arc<dyn CredentialResolver> = Arc::new(StubCreds);
        let history_json = "[]";
        let ctx = provider
            .build_context("anything", history_json, creds)
            .await
            .expect("graceful");
        assert!(
            ctx.contains("RECENT COMMITS"),
            "should still render headers: {ctx}"
        );
        assert!(
            ctx.contains("(unavailable)"),
            "should degrade gracefully: {ctx}"
        );
    }
}
