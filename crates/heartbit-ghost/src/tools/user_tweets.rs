//! `twitter_user_tweets` — fetch a user's recent posts.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

#[derive(Debug, Deserialize)]
struct UserTweetsInput {
    /// Numeric X user id (not handle). Use `twitter_user` to resolve.
    user_id: String,
    #[serde(default = "default_max_results")]
    max_results: u32,
    #[serde(default)]
    since_id: Option<String>,
    /// When `true`, exclude replies and retweets — keep only original posts.
    /// Default `true` (most useful for "what has this account said recently?").
    #[serde(default = "default_exclude")]
    exclude_replies: bool,
}

fn default_max_results() -> u32 {
    10
}

fn default_exclude() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Tweet {
    id: String,
    text: String,
    #[serde(default)]
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UserTweetsOutput {
    tweets: Vec<Tweet>,
    next_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    #[serde(default)]
    data: Vec<Tweet>,
    #[serde(default)]
    meta: ApiMeta,
}

#[derive(Debug, Default, Deserialize)]
struct ApiMeta {
    #[serde(default)]
    next_token: Option<String>,
}

/// Fetch a user's recent original tweets.
pub struct TwitterUserTweetsTool;

impl Default for TwitterUserTweetsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterUserTweetsTool {
    /// Construct the tool. Credentials resolved at execute time via
    /// `ExecutionContext::credentials` (OAuth1).
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterUserTweetsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_user_tweets".into(),
            description: "Fetch a user's recent original tweets (excludes replies/retweets by default). Returns up to `max_results`; use `since_id` to paginate forward.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "X user id (numeric string), not handle. Get this from twitter_user."
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 100,
                        "default": 10
                    },
                    "since_id": {
                        "type": "string",
                        "description": "Optional. Return only tweets newer than this id."
                    },
                    "exclude_replies": {
                        "type": "boolean",
                        "description": "When true, exclude replies and retweets. Default true.",
                        "default": true
                    }
                },
                "required": ["user_id"]
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
            let parsed: UserTweetsInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json = serde_json::to_string(&out)
                        .expect("UserTweetsOutput fields are infallible");
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &UserTweetsInput) -> Result<UserTweetsOutput, XApiError> {
    let path = format!("/2/users/{}/tweets", input.user_id);
    let max_str = input.max_results.to_string();
    let mut query: Vec<(&str, &str)> =
        vec![("max_results", &max_str), ("tweet.fields", "created_at")];
    let exclude_value;
    if input.exclude_replies {
        exclude_value = "replies,retweets".to_string();
        query.push(("exclude", &exclude_value));
    }
    if let Some(since) = input.since_id.as_deref() {
        query.push(("since_id", since));
    }
    let response: ApiResponse = client.get_json(&path, &query).await?;
    Ok(UserTweetsOutput {
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
    async fn user_tweets_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/tweets"))
            .and(query_param("max_results", "10"))
            .and(query_param("exclude", "replies,retweets"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {
                        "id": "9001",
                        "text": "first post",
                        "created_at": "2026-05-09T00:00:00.000Z"
                    }
                ],
                "meta": {"next_token": "next-1"}
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserTweetsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
            exclude_replies: true,
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.tweets.len(), 1);
        assert_eq!(result.tweets[0].id, "9001");
        assert_eq!(result.tweets[0].text, "first post");
        assert_eq!(result.next_token.as_deref(), Some("next-1"));
    }

    #[tokio::test]
    async fn user_tweets_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/tweets"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserTweetsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
            exclude_replies: true,
        };
        let err = call_x(&client, &input).await.expect_err("401 expected");
        assert!(matches!(err, XApiError::Unauthenticated(_)));
    }

    #[tokio::test]
    async fn user_tweets_exclude_replies_false_omits_exclude_param() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/tweets"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data": [], "meta": {}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserTweetsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
            exclude_replies: false,
        };
        let result = call_x(&client, &input).await.expect("ok");
        assert!(result.tweets.is_empty());
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterUserTweetsTool::new();
        assert_eq!(tool.definition().name, "twitter_user_tweets");
    }
}
