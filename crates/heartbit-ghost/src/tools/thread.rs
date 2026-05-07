//! `twitter_thread` — post a thread (sequence of linked tweets).

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

const MAX_TWEET_LENGTH: usize = 280;
const MAX_THREAD_LENGTH: usize = 25;

#[derive(Debug, Deserialize)]
struct ThreadInput {
    tweets: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct ThreadOutput {
    thread_root_id: String,
    tweet_ids: Vec<String>,
    urls: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PostRequest<'a> {
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reply: Option<ReplyTo<'a>>,
}

#[derive(Debug, Serialize)]
struct ReplyTo<'a> {
    in_reply_to_tweet_id: &'a str,
}

#[derive(Debug, Deserialize)]
struct PostApiResponse {
    data: PostApiData,
}

#[derive(Debug, Deserialize)]
struct PostApiData {
    id: String,
}

/// Post a thread of tweets (1..=25 entries, each ≤280 chars).
///
/// Each tweet is posted in sequence; tweets after the first set
/// `reply.in_reply_to_tweet_id` to the previous tweet's id, forming
/// a linked thread on X. Fails fast on the first error; tweets posted
/// before the failure stay live (X has no rollback API).
pub struct TwitterThreadTool;

impl Default for TwitterThreadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterThreadTool {
    /// Construct a new TwitterThreadTool. Credentials are resolved at
    /// execute-time via `ExecutionContext::credentials`.
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterThreadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_thread".into(),
            description: "Post a thread of tweets (1..=25 entries, each ≤280 chars). Each tweet is posted in sequence and linked via reply.in_reply_to_tweet_id to the previous one. Fails fast on the first X error; tweets posted before the failure stay live (X has no rollback API).".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "tweets": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 280},
                        "minItems": 1,
                        "maxItems": 25,
                        "description": "Ordered list of tweet texts. The first becomes the thread root; each subsequent tweet replies to the previous."
                    }
                },
                "required": ["tweets"]
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
            let parsed: ThreadInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            if parsed.tweets.is_empty() {
                return Ok(ToolOutput::error(
                    "tweets array must contain at least 1 entry",
                ));
            }
            if parsed.tweets.len() > MAX_THREAD_LENGTH {
                return Ok(ToolOutput::error(format!(
                    "thread length {} exceeds maximum {}",
                    parsed.tweets.len(),
                    MAX_THREAD_LENGTH
                )));
            }
            for (i, t) in parsed.tweets.iter().enumerate() {
                if t.is_empty() {
                    return Ok(ToolOutput::error(format!("tweet[{i}] must not be empty")));
                }
                let n = t.chars().count();
                if n > MAX_TWEET_LENGTH {
                    return Ok(ToolOutput::error(format!(
                        "tweet[{i}] exceeds {MAX_TWEET_LENGTH} chars (got {n})"
                    )));
                }
            }
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json =
                        serde_json::to_string(&out).expect("ThreadOutput fields are infallible");
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &ThreadInput) -> Result<ThreadOutput, XApiError> {
    let mut tweet_ids: Vec<String> = Vec::with_capacity(input.tweets.len());
    let mut prev_id: Option<String> = None;
    for text in &input.tweets {
        let body = PostRequest {
            text,
            reply: prev_id.as_ref().map(|id| ReplyTo {
                in_reply_to_tweet_id: id,
            }),
        };
        let resp: PostApiResponse = client.post_json("/2/tweets", &body).await?;
        prev_id = Some(resp.data.id.clone());
        tweet_ids.push(resp.data.id);
    }
    let urls: Vec<String> = tweet_ids
        .iter()
        .map(|id| format!("https://twitter.com/i/web/status/{id}"))
        .collect();
    let thread_root_id = tweet_ids[0].clone();
    Ok(ThreadOutput {
        thread_root_id,
        tweet_ids,
        urls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::Secret;
    use wiremock::matchers::{method, path as wm_path};
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
    async fn thread_three_tweets_chain_correctly() {
        let server = MockServer::start().await;
        // First post: no reply field
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(
                ResponseTemplate::new(201)
                    .set_body_json(serde_json::json!({"data": {"id": "1001"}})),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Second post: reply to 1001 → returns 1002
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(
                ResponseTemplate::new(201)
                    .set_body_json(serde_json::json!({"data": {"id": "1002"}})),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Third post: reply to 1002 → returns 1003
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(
                ResponseTemplate::new(201)
                    .set_body_json(serde_json::json!({"data": {"id": "1003"}})),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = ThreadInput {
            tweets: vec!["one".into(), "two".into(), "three".into()],
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.thread_root_id, "1001");
        assert_eq!(result.tweet_ids, vec!["1001", "1002", "1003"]);
        assert_eq!(result.urls.len(), 3);
    }

    #[tokio::test]
    async fn thread_fails_fast_on_mid_thread_error() {
        let server = MockServer::start().await;
        // First post succeeds
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(
                ResponseTemplate::new(201)
                    .set_body_json(serde_json::json!({"data": {"id": "1001"}})),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Second post fails 401
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .respond_with(ResponseTemplate::new(401).set_body_string("token expired"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = ThreadInput {
            tweets: vec!["one".into(), "two".into()],
        };
        let result = call_x(&client, &input).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn execute_rejects_empty_thread() {
        let tool = TwitterThreadTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"tweets": []}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("at least 1 entry"));
    }

    #[tokio::test]
    async fn execute_rejects_thread_over_25() {
        let tool = TwitterThreadTool::new();
        let ctx = ExecutionContext::default();
        let many: Vec<&str> = (0..26).map(|_| "x").collect();
        let result = tool
            .execute(&ctx, serde_json::json!({"tweets": many}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("exceeds maximum 25"));
    }

    #[tokio::test]
    async fn execute_rejects_empty_individual_tweet() {
        let tool = TwitterThreadTool::new();
        let ctx = ExecutionContext::default();
        let result = tool
            .execute(&ctx, serde_json::json!({"tweets": ["ok", ""]}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("tweet[1]"));
    }

    #[tokio::test]
    async fn execute_rejects_individual_tweet_over_280() {
        let tool = TwitterThreadTool::new();
        let ctx = ExecutionContext::default();
        let too_long = "a".repeat(281);
        let result = tool
            .execute(&ctx, serde_json::json!({"tweets": ["ok", too_long]}))
            .await
            .expect("Tool::execute returns Ok");
        assert!(result.is_error);
        assert!(result.content.contains("tweet[1]"));
        assert!(result.content.contains("280"));
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterThreadTool::new();
        assert_eq!(tool.definition().name, "twitter_thread");
    }
}
