//! `twitter_user` — look up an X user by handle.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

#[derive(Debug, Deserialize)]
struct UserInput {
    /// Handle without leading `@`.
    handle: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct UserResponse {
    id: String,
    name: String,
    username: String,
    description: Option<String>,
    public_metrics: Option<UserMetrics>,
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct UserMetrics {
    followers_count: Option<u64>,
    following_count: Option<u64>,
    tweet_count: Option<u64>,
}

/// X v2 GET /2/users/by/username/:handle response wrapper.
#[derive(Debug, Deserialize)]
struct UserApiResponse {
    data: UserResponse,
}

/// Look up an X user by handle.
pub struct TwitterUserTool;

impl Default for TwitterUserTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterUserTool {
    /// Create the tool. Credentials are resolved at execute-time via
    /// `ExecutionContext::credentials`.
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterUserTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_user".into(),
            description: "Look up an X (Twitter) user by handle. Returns id, name, description, and follower/following/tweet counts.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "User handle without the leading @, e.g. \"karpathy\""
                    }
                },
                "required": ["handle"]
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
            let parsed: UserInput = match serde_json::from_value(input) {
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
                        serde_json::to_string(&out).expect("UserResponse fields are infallible");
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &UserInput) -> Result<UserResponse, XApiError> {
    let path = format!("/2/users/by/username/{}", input.handle);
    let response: UserApiResponse = client
        .get_json(
            &path,
            &[("user.fields", "description,public_metrics,created_at")],
        )
        .await?;
    Ok(response.data)
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
    async fn user_lookup_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/by/username/karpathy"))
            .and(query_param(
                "user.fields",
                "description,public_metrics,created_at",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "id": "33836629",
                    "name": "Andrej Karpathy",
                    "username": "karpathy",
                    "description": "AI/ML.",
                    "public_metrics": {
                        "followers_count": 1000000,
                        "following_count": 100,
                        "tweet_count": 5000
                    },
                    "created_at": "2009-04-21T22:00:00.000Z"
                }
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserInput {
            handle: "karpathy".into(),
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.id, "33836629");
        assert_eq!(result.username, "karpathy");
        assert_eq!(
            result.public_metrics.unwrap().followers_count,
            Some(1000000)
        );
    }

    #[tokio::test]
    async fn user_lookup_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/by/username/anyhandle"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserInput {
            handle: "anyhandle".into(),
        };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn user_lookup_returns_rate_limited_on_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/by/username/anyhandle"))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "60"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserInput {
            handle: "anyhandle".into(),
        };
        let result = call_x(&client, &input).await;
        match result {
            Err(XApiError::RateLimited { retry_after_secs }) => assert_eq!(retry_after_secs, 60),
            Err(other) => panic!("expected RateLimited, got {:?}", other),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[tokio::test]
    async fn user_lookup_surfaces_x_error_on_4xx() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/by/username/nosuchuser"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "errors": [{"message": "user not found"}]
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserInput {
            handle: "nosuchuser".into(),
        };
        let result = call_x(&client, &input).await;
        match result {
            Err(XApiError::ApiError { status, message }) => {
                assert_eq!(status, 404);
                assert_eq!(message, "user not found");
            }
            Err(other) => panic!("expected ApiError, got {:?}", other),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[tokio::test]
    async fn execute_returns_error_when_credentials_missing() {
        let tool = TwitterUserTool::new();
        let ctx = ExecutionContext::default(); // no resolver
        let result = tool
            .execute(&ctx, serde_json::json!({"handle": "anything"}))
            .await
            .expect("Tool::execute returns Ok with ToolOutput::error on credential failure");
        assert!(result.is_error);
        assert!(result.content.contains("no credential resolver"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterUserTool::new();
        let def = tool.definition();
        assert_eq!(def.name, "twitter_user");
    }
}
