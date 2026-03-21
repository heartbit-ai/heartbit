use std::fmt::Write as _;
use std::future::Future;
use std::pin::Pin;
use std::time::SystemTime;

use base64::Engine;
use hmac::{Hmac, Mac};
use serde_json::json;
use sha1::Sha1;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const X_API_URL: &str = "https://api.twitter.com/2/tweets";
const MAX_TWEET_LENGTH: usize = 280;

type HmacSha1 = Hmac<Sha1>;

/// Per-tenant X/Twitter credentials for OAuth 1.0a signing.
#[derive(Clone)]
pub struct TwitterCredentials {
    pub consumer_key: String,
    pub consumer_secret: String,
    pub access_token: String,
    pub access_token_secret: String,
}

impl std::fmt::Debug for TwitterCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwitterCredentials")
            .field("consumer_key", &"[REDACTED]")
            .field("consumer_secret", &"[REDACTED]")
            .field("access_token", &"[REDACTED]")
            .field("access_token_secret", &"[REDACTED]")
            .finish()
    }
}

/// Builtin tool for posting tweets to X/Twitter via API v2.
///
/// Uses OAuth 1.0a signing with per-tenant credentials injected at runtime.
/// Only instantiated when `TwitterCredentials` are provided (multi-tenant).
pub struct TwitterPostTool {
    credentials: TwitterCredentials,
    client: reqwest::Client,
}

impl TwitterPostTool {
    pub fn new(credentials: TwitterCredentials) -> Self {
        Self {
            credentials,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build reqwest client"),
        }
    }
}

/// Percent-encode a string per RFC 5849 (OAuth 1.0a).
fn percent_encode(s: &str) -> String {
    let mut encoded = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                // write! to String is infallible — avoids temporary allocation from format!
                let _ = write!(encoded, "%{byte:02X}");
            }
        }
    }
    encoded
}

/// Build the OAuth 1.0a Authorization header for a POST request.
fn build_oauth_header(
    url: &str,
    consumer_key: &str,
    consumer_secret: &str,
    access_token: &str,
    access_token_secret: &str,
    nonce: &str,
    timestamp: u64,
) -> Result<String, Error> {
    let oauth_params = [
        ("oauth_consumer_key", consumer_key),
        ("oauth_nonce", nonce),
        ("oauth_signature_method", "HMAC-SHA1"),
        ("oauth_timestamp", &timestamp.to_string()),
        ("oauth_token", access_token),
        ("oauth_version", "1.0"),
    ];

    // Build parameter string (sorted by key)
    let param_string: String = oauth_params
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
        .collect::<Vec<_>>()
        .join("&");

    // Build signature base string: METHOD&url&params
    let base_string = format!(
        "POST&{}&{}",
        percent_encode(url),
        percent_encode(&param_string),
    );

    // Sign with HMAC-SHA1
    let signing_key = format!(
        "{}&{}",
        percent_encode(consumer_secret),
        percent_encode(access_token_secret),
    );

    let mut mac = HmacSha1::new_from_slice(signing_key.as_bytes())
        .map_err(|e| Error::Agent(format!("HMAC key error: {e}")))?;
    mac.update(base_string.as_bytes());
    let signature = base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());

    // Build Authorization header
    Ok(format!(
        "OAuth oauth_consumer_key=\"{}\", \
         oauth_nonce=\"{}\", \
         oauth_signature=\"{}\", \
         oauth_signature_method=\"HMAC-SHA1\", \
         oauth_timestamp=\"{}\", \
         oauth_token=\"{}\", \
         oauth_version=\"1.0\"",
        percent_encode(consumer_key),
        percent_encode(nonce),
        percent_encode(&signature),
        timestamp,
        percent_encode(access_token),
    ))
}

impl Tool for TwitterPostTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_post".into(),
            description: "Post a tweet to X/Twitter. Maximum 280 characters.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The tweet text to post (max 280 characters)"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let text = input
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("text is required".into()))?;

            if text.is_empty() {
                return Ok(ToolOutput::error("text must not be empty"));
            }

            let char_count = text.chars().count();
            if char_count > MAX_TWEET_LENGTH {
                return Ok(ToolOutput::error(format!(
                    "Tweet exceeds {MAX_TWEET_LENGTH} characters (got {char_count}). \
                     Please shorten your tweet."
                )));
            }

            // Generate OAuth nonce and timestamp
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|e| Error::Agent(format!("system time error: {e}")))?
                .as_secs();

            // UUID v4 provides cryptographically random nonce (required by RFC 5849)
            let nonce = uuid::Uuid::new_v4().to_string().replace('-', "");

            let auth_header = build_oauth_header(
                X_API_URL,
                &self.credentials.consumer_key,
                &self.credentials.consumer_secret,
                &self.credentials.access_token,
                &self.credentials.access_token_secret,
                &nonce,
                timestamp,
            )?;

            let body = json!({ "text": text });

            let response = self
                .client
                .post(X_API_URL)
                .header("Authorization", &auth_header)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| Error::Agent(format!("X API request failed: {e}")))?;

            let status = response.status();
            let response_body: serde_json::Value = response
                .json()
                .await
                .map_err(|e| Error::Agent(format!("Failed to parse X API response: {e}")))?;

            if !status.is_success() {
                let detail = response_body
                    .get("detail")
                    .and_then(|v| v.as_str())
                    .or_else(|| response_body.get("title").and_then(|v| v.as_str()))
                    .unwrap_or("Unknown error");
                return Ok(ToolOutput::error(format!(
                    "X API error (HTTP {}): {detail}",
                    status.as_u16()
                )));
            }

            // Extract tweet ID from response
            let tweet_id = response_body
                .get("data")
                .and_then(|d| d.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            Ok(ToolOutput::success(format!(
                "Tweet posted successfully!\n\
                 Tweet ID: {tweet_id}\n\
                 URL: https://x.com/i/status/{tweet_id}\n\
                 Text: {text}"
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_credentials() -> TwitterCredentials {
        TwitterCredentials {
            consumer_key: "test_consumer_key".into(),
            consumer_secret: "test_consumer_secret".into(),
            access_token: "test_access_token".into(),
            access_token_secret: "test_access_token_secret".into(),
        }
    }

    #[test]
    fn definition_has_correct_name() {
        let tool = TwitterPostTool::new(test_credentials());
        assert_eq!(tool.definition().name, "twitter_post");
    }

    #[test]
    fn definition_requires_text() {
        let tool = TwitterPostTool::new(test_credentials());
        let schema = &tool.definition().input_schema;
        let required = schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "text");
    }

    #[test]
    fn percent_encode_unreserved() {
        assert_eq!(percent_encode("abc123"), "abc123");
        assert_eq!(
            percent_encode("hello-world_test.v2~"),
            "hello-world_test.v2~"
        );
    }

    #[test]
    fn percent_encode_reserved() {
        assert_eq!(percent_encode("hello world"), "hello%20world");
        assert_eq!(percent_encode("a&b=c"), "a%26b%3Dc");
        assert_eq!(percent_encode("100%"), "100%25");
    }

    #[test]
    fn percent_encode_special_chars() {
        assert_eq!(percent_encode("/"), "%2F");
        assert_eq!(percent_encode(":"), "%3A");
        assert_eq!(percent_encode("@"), "%40");
    }

    #[test]
    fn build_oauth_header_produces_valid_format() {
        let header = build_oauth_header(
            "https://api.twitter.com/2/tweets",
            "consumer_key",
            "consumer_secret",
            "access_token",
            "access_token_secret",
            "testnonce123",
            1234567890,
        )
        .unwrap();

        assert!(header.starts_with("OAuth "));
        assert!(header.contains("oauth_consumer_key=\"consumer_key\""));
        assert!(header.contains("oauth_nonce=\"testnonce123\""));
        assert!(header.contains("oauth_signature_method=\"HMAC-SHA1\""));
        assert!(header.contains("oauth_timestamp=\"1234567890\""));
        assert!(header.contains("oauth_token=\"access_token\""));
        assert!(header.contains("oauth_version=\"1.0\""));
        assert!(header.contains("oauth_signature=\""));
    }

    #[test]
    fn build_oauth_header_signature_is_deterministic() {
        let h1 = build_oauth_header(X_API_URL, "ck", "cs", "at", "ats", "nonce", 1000).unwrap();
        let h2 = build_oauth_header(X_API_URL, "ck", "cs", "at", "ats", "nonce", 1000).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn build_oauth_header_different_nonce_produces_different_signature() {
        let h1 = build_oauth_header(X_API_URL, "ck", "cs", "at", "ats", "nonce1", 1000).unwrap();
        let h2 = build_oauth_header(X_API_URL, "ck", "cs", "at", "ats", "nonce2", 1000).unwrap();
        assert_ne!(h1, h2);
    }

    #[tokio::test]
    async fn rejects_empty_text() {
        let tool = TwitterPostTool::new(test_credentials());
        let result = tool.execute(json!({"text": ""})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("must not be empty"));
    }

    #[tokio::test]
    async fn rejects_text_too_long() {
        let tool = TwitterPostTool::new(test_credentials());
        let long = "a".repeat(281);
        let result = tool.execute(json!({"text": long})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("exceeds 280 characters"));
    }

    #[tokio::test]
    async fn rejects_missing_text() {
        let tool = TwitterPostTool::new(test_credentials());
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("text is required"), "got: {err}");
    }

    #[test]
    fn credentials_debug_redacts_secrets() {
        let creds = test_credentials();
        let debug = format!("{creds:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("test_consumer_key"));
        assert!(!debug.contains("test_consumer_secret"));
    }

    #[tokio::test]
    async fn accepts_280_chars() {
        // 280 chars should pass validation (will fail at HTTP level, but that's expected)
        let tool = TwitterPostTool::new(test_credentials());
        let text = "a".repeat(280);
        let result = tool.execute(json!({"text": text})).await;
        // Should not be a validation error — will fail at HTTP level
        match result {
            Ok(output) => {
                // Network error is fine, but should NOT be a validation error
                if output.is_error {
                    assert!(
                        !output.content.contains("exceeds"),
                        "280 chars should not be rejected: {}",
                        output.content
                    );
                }
            }
            Err(_) => {
                // Network error is expected with fake credentials
            }
        }
    }
}
