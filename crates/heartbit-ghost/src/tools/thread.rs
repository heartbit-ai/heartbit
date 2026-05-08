//! `twitter_thread` — post a thread (sequence of linked tweets).

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

const MAX_TWEET_LENGTH: usize = 280;
const MAX_THREAD_LENGTH: usize = 25;
const MAX_HEAD_IMAGE_BYTES: usize = 5 * 1024 * 1024; // 5 MiB

/// Sniff MIME type from the first bytes of an image. Returns one of
/// `"image/png"`, `"image/jpeg"`, `"image/gif"`, `"image/webp"`. Falls
/// back to `"image/png"` when no magic-bytes match (Gemini's image
/// preview model returns PNG by default, and X infers from media_type).
fn sniff_mime(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n" {
        return "image/png";
    }
    if bytes.len() >= 3 && &bytes[..3] == b"\xFF\xD8\xFF" {
        return "image/jpeg";
    }
    if bytes.len() >= 6 && (&bytes[..6] == b"GIF87a" || &bytes[..6] == b"GIF89a") {
        return "image/gif";
    }
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "image/webp";
    }
    "image/png"
}

/// Decode the caller's base64 string and validate size.
/// Returns `(bytes, sniffed_mime)` on success, or an error message
/// suitable for `ToolOutput::error`.
fn decode_and_validate_head_image(b64: &str) -> Result<(Vec<u8>, &'static str), String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("invalid head_image_b64: {e}"))?;
    if bytes.len() > MAX_HEAD_IMAGE_BYTES {
        return Err(format!(
            "head image exceeds 5 MiB (got {} bytes)",
            bytes.len()
        ));
    }
    if bytes.is_empty() {
        return Err("head_image_b64 decoded to zero bytes".to_string());
    }
    let mime = sniff_mime(&bytes);
    Ok((bytes, mime))
}

#[derive(Debug, Deserialize)]
struct ThreadInput {
    tweets: Vec<String>,
    /// Optional base64-encoded image bytes to attach to the FIRST tweet.
    /// MIME type is sniffed from the bytes (PNG/JPEG/WebP/GIF).
    /// When `None`, posts text-only (existing behavior preserved).
    #[serde(default)]
    head_image_b64: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    media: Option<Media>,
}

#[derive(Debug, Serialize)]
struct Media {
    media_ids: Vec<String>,
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
                    },
                    "head_image_b64": {
                        "type": "string",
                        "description": "Optional base64-encoded image to attach to the first tweet (PNG/JPEG/WebP/GIF, ≤5 MiB). When omitted, posts text-only."
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
    // 1. If head_image_b64 is set, decode → validate → upload → media_id.
    let mut head_media_ids: Option<Vec<String>> = None;
    if let Some(b64) = input.head_image_b64.as_ref() {
        let (bytes, mime) = decode_and_validate_head_image(b64).map_err(XApiError::Validation)?;
        let media_id = client.upload_image_chunked(&bytes, mime).await?;
        head_media_ids = Some(vec![media_id]);
    }

    // 2. Post tweets sequentially. First tweet gets the optional media;
    // subsequent tweets are text-only and reply to the previous.
    let mut tweet_ids: Vec<String> = Vec::with_capacity(input.tweets.len());
    let mut prev_id: Option<String> = None;
    for (idx, text) in input.tweets.iter().enumerate() {
        let media = if idx == 0 {
            head_media_ids.take().map(|media_ids| Media { media_ids })
        } else {
            None
        };
        let body = PostRequest {
            text,
            reply: prev_id.as_ref().map(|id| ReplyTo {
                in_reply_to_tweet_id: id,
            }),
            media,
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
        use wiremock::matchers::body_partial_json;

        let server = MockServer::start().await;
        // First post: text "one", NO reply field (impl omits via skip_serializing_if).
        // body_partial_json with {"text": "one"} verifies the text content; the impl's
        // skip_serializing_if guarantees no reply key is sent for the root tweet.
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .and(body_partial_json(serde_json::json!({"text": "one"})))
            .respond_with(
                ResponseTemplate::new(201)
                    .set_body_json(serde_json::json!({"data": {"id": "1001"}})),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Second post: text "two", reply.in_reply_to_tweet_id = "1001"
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .and(body_partial_json(serde_json::json!({
                "text": "two",
                "reply": {"in_reply_to_tweet_id": "1001"}
            })))
            .respond_with(
                ResponseTemplate::new(201)
                    .set_body_json(serde_json::json!({"data": {"id": "1002"}})),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Third post: text "three", reply.in_reply_to_tweet_id = "1002"
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .and(body_partial_json(serde_json::json!({
                "text": "three",
                "reply": {"in_reply_to_tweet_id": "1002"}
            })))
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
            head_image_b64: None,
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
            head_image_b64: None,
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

    use base64::Engine;

    #[test]
    fn thread_input_deserializes_with_head_image_b64() {
        let json = r#"{"tweets": ["hello"], "head_image_b64": "iVBORw0KGgo="}"#;
        let parsed: ThreadInput = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.tweets, vec!["hello".to_string()]);
        assert_eq!(parsed.head_image_b64.as_deref(), Some("iVBORw0KGgo="));
    }

    #[test]
    fn thread_input_deserializes_without_head_image_b64() {
        // Existing P1.3d callers don't pass the field; verify back-compat.
        let json = r#"{"tweets": ["hello"]}"#;
        let parsed: ThreadInput = serde_json::from_str(json).unwrap();
        assert!(parsed.head_image_b64.is_none());
    }

    #[test]
    fn sniff_mime_recognizes_known_magic_bytes() {
        assert_eq!(sniff_mime(b"\x89PNG\r\n\x1a\nfoo"), "image/png");
        assert_eq!(sniff_mime(b"\xFF\xD8\xFFfoo"), "image/jpeg");
        assert_eq!(sniff_mime(b"GIF87afoo"), "image/gif");
        assert_eq!(sniff_mime(b"GIF89afoo"), "image/gif");
        // WebP: bytes 0..4 = RIFF, bytes 8..12 = WEBP.
        let mut webp = b"RIFF\x00\x00\x00\x00WEBP".to_vec();
        webp.push(0xAA);
        assert_eq!(sniff_mime(&webp), "image/webp");
        // Unknown → png fallback.
        assert_eq!(sniff_mime(b"definitely not an image"), "image/png");
        // Too short → png fallback (won't match any magic).
        assert_eq!(sniff_mime(b"abc"), "image/png");
    }

    #[test]
    fn decode_and_validate_head_image_rejects_oversize() {
        // Build a 6 MiB raw byte vec, encode to base64.
        let big = vec![0u8; 6 * 1024 * 1024];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&big);
        let err = decode_and_validate_head_image(&b64).unwrap_err();
        assert!(err.contains("exceeds 5 MiB"), "got: {err}");
    }
}
