//! `twitter_reply` — reply to an existing tweet.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

const MAX_TWEET_LENGTH: usize = 280;

#[derive(Debug, Deserialize)]
struct ReplyInput {
    text: String,
    in_reply_to: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct ReplyOutput {
    tweet_id: String,
    url: String,
}

/// X v2 POST /2/tweets request body for a reply.
#[derive(Debug, Serialize)]
struct ReplyRequest<'a> {
    text: &'a str,
    reply: ReplyTo<'a>,
}

#[derive(Debug, Serialize)]
struct ReplyTo<'a> {
    in_reply_to_tweet_id: &'a str,
}

#[derive(Debug, Deserialize)]
struct ReplyApiResponse {
    data: ReplyApiData,
}

#[derive(Debug, Deserialize)]
struct ReplyApiData {
    id: String,
}

/// Reply to an X tweet by posting a new tweet linked via `in_reply_to_tweet_id`.
pub struct TwitterReplyTool;

impl Default for TwitterReplyTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterReplyTool {
    /// Create the tool. Credentials are resolved at execute-time via
    /// `ExecutionContext::credentials`.
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterReplyTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_reply".into(),
            description: "Reply to an X (Twitter) tweet. Posts a new tweet linked to in_reply_to. Max 280 chars.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Reply text. Max 280 characters.",
                        "maxLength": 280
                    },
                    "in_reply_to": {
                        "type": "string",
                        "description": "Tweet id (numeric string) being replied to."
                    }
                },
                "required": ["text", "in_reply_to"]
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
            let parsed: ReplyInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let char_count = parsed.text.chars().count();
            if parsed.text.is_empty() {
                return Ok(ToolOutput::error("text must not be empty"));
            }
            if char_count > MAX_TWEET_LENGTH {
                return Ok(ToolOutput::error(format!(
                    "Reply exceeds {MAX_TWEET_LENGTH} characters (got {char_count})."
                )));
            }
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json =
                        serde_json::to_string(&out).expect("ReplyOutput fields are infallible");
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &ReplyInput) -> Result<ReplyOutput, XApiError> {
    let body = ReplyRequest {
        text: &input.text,
        reply: ReplyTo {
            in_reply_to_tweet_id: &input.in_reply_to,
        },
    };
    let response: ReplyApiResponse = client.post_json("/2/tweets", &body).await?;
    let tweet_id = response.data.id;
    let url = format!("https://twitter.com/i/web/status/{}", tweet_id);
    Ok(ReplyOutput { tweet_id, url })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::Secret;
    use wiremock::matchers::{body_json, method, path as wm_path};
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
    async fn reply_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .and(body_json(serde_json::json!({
                "text": "thanks!",
                "reply": {"in_reply_to_tweet_id": "9001"}
            })))
            .respond_with(
                ResponseTemplate::new(201)
                    .set_body_json(serde_json::json!({"data": {"id": "9999"}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = ReplyInput {
            text: "thanks!".into(),
            in_reply_to: "9001".into(),
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.tweet_id, "9999");
        assert!(result.url.contains("9999"));
    }

    #[tokio::test]
    async fn reply_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = ReplyInput {
            text: "x".into(),
            in_reply_to: "1".into(),
        };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn execute_rejects_text_over_280_chars() {
        let tool = TwitterReplyTool::new();
        let ctx = ExecutionContext::default();
        let too_long = "a".repeat(281);
        let result = tool
            .execute(
                &ctx,
                serde_json::json!({"text": too_long, "in_reply_to": "1"}),
            )
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("exceeds 280"));
    }

    #[tokio::test]
    async fn execute_rejects_empty_text() {
        let tool = TwitterReplyTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"text": "", "in_reply_to": "1"}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("must not be empty"));
    }

    #[tokio::test]
    async fn execute_no_credentials_returns_clear_error() {
        let tool = TwitterReplyTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"text": "hi", "in_reply_to": "1"}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("no credential resolver"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterReplyTool::new();
        assert_eq!(tool.definition().name, "twitter_reply");
    }
}
