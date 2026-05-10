//! `twitter_mentions` — fetch mentions of a specific X user.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

#[derive(Debug, Deserialize)]
struct MentionsInput {
    user_id: String,
    #[serde(default = "default_max_results")]
    max_results: u32,
    #[serde(default)]
    since_id: Option<String>,
}

fn default_max_results() -> u32 {
    10
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Mention {
    id: String,
    text: String,
    author_id: Option<String>,
    created_at: Option<String>,
    in_reply_to_user_id: Option<String>,
    #[serde(default)]
    conversation_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MentionsOutput {
    mentions: Vec<Mention>,
    next_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MentionsApiResponse {
    #[serde(default)]
    data: Vec<Mention>,
    #[serde(default)]
    meta: MentionsMeta,
}

#[derive(Debug, Default, Deserialize)]
struct MentionsMeta {
    #[serde(default)]
    next_token: Option<String>,
}

/// Fetch recent mentions of a specific X user.
pub struct TwitterMentionsTool;

impl Default for TwitterMentionsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterMentionsTool {
    /// Create the tool. Credentials are resolved at execute-time via
    /// `ExecutionContext::credentials`.
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterMentionsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_mentions".into(),
            description: "Fetch recent mentions of a specific X user. Returns up to `max_results` mentions; use `since_id` to paginate forward.".into(),
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
                        "description": "Optional. Return only mentions newer than this id."
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
            let parsed: MentionsInput = match serde_json::from_value(input) {
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
                        serde_json::to_string(&out).expect("MentionsOutput fields are infallible");
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &MentionsInput) -> Result<MentionsOutput, XApiError> {
    let path = format!("/2/users/{}/mentions", input.user_id);
    let max_str = input.max_results.to_string();
    let mut query: Vec<(&str, &str)> = vec![
        ("max_results", &max_str),
        (
            "tweet.fields",
            "author_id,created_at,in_reply_to_user_id,conversation_id",
        ),
    ];
    if let Some(since) = input.since_id.as_deref() {
        query.push(("since_id", since));
    }
    let response: MentionsApiResponse = client.get_json(&path, &query).await?;
    Ok(MentionsOutput {
        mentions: response.data,
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
    async fn mentions_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/mentions"))
            .and(query_param("max_results", "10"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {
                        "id": "9001",
                        "text": "hey @user",
                        "author_id": "200",
                        "created_at": "2026-01-01T00:00:00.000Z",
                        "in_reply_to_user_id": "100",
                        "conversation_id": "conv-root-1"
                    }
                ],
                "meta": {"next_token": "next-mentions-1"}
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = MentionsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.mentions.len(), 1);
        assert_eq!(result.mentions[0].id, "9001");
        assert_eq!(
            result.mentions[0].in_reply_to_user_id.as_deref(),
            Some("100")
        );
        assert_eq!(
            result.mentions[0].conversation_id.as_deref(),
            Some("conv-root-1")
        );
    }

    #[tokio::test]
    async fn mentions_with_since_id_passes_param() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/mentions"))
            .and(query_param("since_id", "8000"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data": [], "meta": {}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = MentionsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: Some("8000".into()),
        };
        let result = call_x(&client, &input).await.expect("with since_id");
        assert!(result.mentions.is_empty());
    }

    #[tokio::test]
    async fn mentions_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/mentions"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = MentionsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
        };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn execute_no_credentials_returns_clear_error() {
        let tool = TwitterMentionsTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"user_id": "100"}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("no credential resolver"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterMentionsTool::new();
        assert_eq!(tool.definition().name, "twitter_mentions");
    }

    #[tokio::test]
    async fn mentions_includes_conversation_id_in_response() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/mentions"))
            .and(query_param(
                "tweet.fields",
                "author_id,created_at,in_reply_to_user_id,conversation_id",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {
                        "id": "9001",
                        "text": "hey @user",
                        "author_id": "200",
                        "created_at": "2026-01-01T00:00:00.000Z",
                        "in_reply_to_user_id": "100",
                        "conversation_id": "conv-root-1"
                    }
                ],
                "meta": {}
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = MentionsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.mentions.len(), 1);
        assert_eq!(
            result.mentions[0].conversation_id.as_deref(),
            Some("conv-root-1")
        );
    }
}
