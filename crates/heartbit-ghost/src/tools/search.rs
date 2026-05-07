//! `twitter_search` — search X recent tweets matching a query.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

#[derive(Debug, Deserialize)]
struct SearchInput {
    query: String,
    #[serde(default = "default_max_results")]
    max_results: u32,
    #[serde(default)]
    since_id: Option<String>,
}

fn default_max_results() -> u32 {
    10
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Tweet {
    id: String,
    text: String,
    author_id: Option<String>,
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchOutput {
    tweets: Vec<Tweet>,
    next_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchApiResponse {
    #[serde(default)]
    data: Vec<Tweet>,
    #[serde(default)]
    meta: SearchMeta,
}

#[derive(Debug, Default, Deserialize)]
struct SearchMeta {
    #[serde(default)]
    next_token: Option<String>,
}

/// Search recent X tweets matching a query.
pub struct TwitterSearchTool;

impl Default for TwitterSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterSearchTool {
    /// Create the tool. Credentials are resolved at execute-time via
    /// `ExecutionContext::credentials`.
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_search".into(),
            description: "Search recent X (Twitter) posts. Returns up to `max_results` tweets matching the query. Use `since_id` to paginate forward (newer than the given tweet id).".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "X search query (supports operators like from:user, lang:en, hashtags, etc.). Max 512 chars.",
                        "maxLength": 512
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 100,
                        "default": 10,
                        "description": "Number of tweets to return (10-100)."
                    },
                    "since_id": {
                        "type": "string",
                        "description": "Optional. Return only tweets newer than this id."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let parsed: SearchInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json =
                        serde_json::to_string(&out).expect("SearchOutput fields are infallible");
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &SearchInput) -> Result<SearchOutput, XApiError> {
    let max_str = input.max_results.to_string();
    let mut query: Vec<(&str, &str)> = vec![
        ("query", &input.query),
        ("max_results", &max_str),
        ("tweet.fields", "author_id,created_at"),
    ];
    if let Some(since) = input.since_id.as_deref() {
        query.push(("since_id", since));
    }
    let response: SearchApiResponse = client.get_json("/2/tweets/search/recent", &query).await?;
    Ok(SearchOutput {
        tweets: response.data,
        next_token: response.meta.next_token,
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

    #[tokio::test]
    async fn search_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/search/recent"))
            .and(query_param("query", "rust"))
            .and(query_param("max_results", "10"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"id": "1", "text": "rust is great", "author_id": "100", "created_at": "2026-01-01T00:00:00.000Z"}
                ],
                "meta": {"next_token": "next-page-1"}
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = SearchInput {
            query: "rust".into(),
            max_results: 10,
            since_id: None,
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.tweets.len(), 1);
        assert_eq!(result.tweets[0].text, "rust is great");
        assert_eq!(result.next_token.as_deref(), Some("next-page-1"));
    }

    #[tokio::test]
    async fn search_with_since_id_passes_param() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/search/recent"))
            .and(query_param("since_id", "12345"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data": [], "meta": {}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = SearchInput {
            query: "rust".into(),
            max_results: 10,
            since_id: Some("12345".into()),
        };
        let result = call_x(&client, &input).await.expect("with since_id");
        assert!(result.tweets.is_empty());
        assert!(result.next_token.is_none());
    }

    #[tokio::test]
    async fn search_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/search/recent"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = SearchInput {
            query: "x".into(),
            max_results: 10,
            since_id: None,
        };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn search_returns_rate_limited_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/tweets/search/recent"))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "15"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = SearchInput {
            query: "x".into(),
            max_results: 10,
            since_id: None,
        };
        let result = call_x(&client, &input).await;
        match result {
            Err(XApiError::RateLimited { retry_after_secs }) => assert_eq!(retry_after_secs, 15),
            Err(other) => panic!("expected RateLimited, got {:?}", other),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[tokio::test]
    async fn execute_no_credentials_returns_clear_error() {
        let tool = TwitterSearchTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"query": "anything"}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("no credential resolver"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterSearchTool::new();
        assert_eq!(tool.definition().name, "twitter_search");
    }
}
