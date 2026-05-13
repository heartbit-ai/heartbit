//! `twitter_quote` — post a quote-tweet (a new tweet that references
//! an existing tweet via the X v2 `quote_tweet_id` field).

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

const MAX_TWEET_LENGTH: usize = 280;

#[derive(Debug, Deserialize)]
struct QuoteInput {
    text: String,
    quote_tweet_id: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct QuoteOutput {
    tweet_id: String,
    url: String,
}

/// X v2 POST /2/tweets request body for a quote-tweet.
#[derive(Debug, Serialize)]
struct QuoteRequest<'a> {
    text: &'a str,
    quote_tweet_id: &'a str,
}

#[derive(Debug, Deserialize)]
struct QuoteApiResponse {
    data: QuoteApiData,
}

#[derive(Debug, Deserialize)]
struct QuoteApiData {
    id: String,
}

/// Post a quote-tweet by `text` + `quote_tweet_id`.
pub struct TwitterQuoteTool;

impl Default for TwitterQuoteTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterQuoteTool {
    /// Create the tool. Credentials are resolved at execute-time via
    /// `ExecutionContext::credentials`.
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterQuoteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_quote".into(),
            description: "Post a quote-tweet on X. Wraps POST /2/tweets with quote_tweet_id. Max 280 chars of comment text.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Quote-tweet comment text. Max 280 characters.",
                        "maxLength": 280
                    },
                    "quote_tweet_id": {
                        "type": "string",
                        "description": "Numeric tweet id of the tweet being quoted."
                    }
                },
                "required": ["text", "quote_tweet_id"]
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
            let parsed: QuoteInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let char_count = parsed.text.chars().count();
            if parsed.text.is_empty() {
                return Ok(ToolOutput::error("text must not be empty"));
            }
            if char_count > MAX_TWEET_LENGTH {
                return Ok(ToolOutput::error(format!(
                    "Quote text exceeds {MAX_TWEET_LENGTH} characters (got {char_count})."
                )));
            }
            if parsed.quote_tweet_id.is_empty()
                || !parsed.quote_tweet_id.chars().all(|c| c.is_ascii_digit())
            {
                return Ok(ToolOutput::error(
                    "quote_tweet_id must be a non-empty numeric string",
                ));
            }
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json =
                        serde_json::to_string(&out).expect("QuoteOutput fields are infallible");
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &QuoteInput) -> Result<QuoteOutput, XApiError> {
    let body = QuoteRequest {
        text: &input.text,
        quote_tweet_id: &input.quote_tweet_id,
    };
    let response: QuoteApiResponse = client.post_json("/2/tweets", &body).await?;
    let tweet_id = response.data.id;
    let url = format!("https://twitter.com/i/web/status/{}", tweet_id);
    Ok(QuoteOutput { tweet_id, url })
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
    async fn quote_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .and(body_json(serde_json::json!({
                "text": "thoughtful agreement",
                "quote_tweet_id": "9001"
            })))
            .respond_with(
                ResponseTemplate::new(201)
                    .set_body_json(serde_json::json!({"data": {"id": "9999"}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = QuoteInput {
            text: "thoughtful agreement".into(),
            quote_tweet_id: "9001".into(),
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.tweet_id, "9999");
        assert!(result.url.contains("9999"));
    }

    #[tokio::test]
    async fn quote_rejects_text_over_280_chars() {
        // Validation happens BEFORE the HTTP call — no server needed.
        let tool = TwitterQuoteTool::new();
        let input = serde_json::json!({
            "text": "x".repeat(281),
            "quote_tweet_id": "9001"
        });
        let ctx = ExecutionContext::default();
        let out = tool
            .execute(&ctx, input)
            .await
            .expect("validation returns Ok(ToolOutput::error)");
        assert!(out.is_error);
        assert!(
            out.content.contains("exceeds 280"),
            "expected length-rejection; got: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn quote_rejects_non_numeric_id() {
        let tool = TwitterQuoteTool::new();
        let input = serde_json::json!({
            "text": "ok",
            "quote_tweet_id": "not-a-number"
        });
        let ctx = ExecutionContext::default();
        let out = tool.execute(&ctx, input).await.expect("Ok");
        assert!(out.is_error);
        assert!(
            out.content.contains("non-empty numeric string"),
            "got: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn quote_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = QuoteInput {
            text: "x".into(),
            quote_tweet_id: "1".into(),
        };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterQuoteTool::new();
        assert_eq!(tool.definition().name, "twitter_quote");
    }

    #[tokio::test]
    async fn execute_rejects_empty_text() {
        let tool = TwitterQuoteTool::new();
        let input = serde_json::json!({
            "text": "",
            "quote_tweet_id": "9001"
        });
        let ctx = ExecutionContext::default();
        let out = tool.execute(&ctx, input).await.expect("Ok");
        assert!(out.is_error);
        assert!(
            out.content.contains("must not be empty"),
            "got: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn execute_no_credentials_returns_clear_error() {
        let tool = TwitterQuoteTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(
                &ctx,
                serde_json::json!({"text": "hi", "quote_tweet_id": "1"}),
            )
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("no credential resolver"));
    }
}
