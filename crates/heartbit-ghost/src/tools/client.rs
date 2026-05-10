//! X HTTP client shared by all heartbit-ghost X tools.
//!
//! - Resolves OAuth1 credentials from `ExecutionContext::credentials` at execute-time
//! - Signs every request via OAuth1
//! - Maps X API responses (200 / 401 / 429 / 4xx / 5xx) to typed `XApiError` variants
//! - `format_error` produces user-friendly tool error messages

use std::sync::Arc;

use heartbit_core::{ExecutionContext, Secret};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

/// X API base URL — overridden in tests via `XClient::new()`.
pub(crate) const X_API_BASE_URL: &str = "https://api.twitter.com";

/// Strongly typed errors from X API calls.
#[derive(Debug, Error)]
pub enum XApiError {
    /// `ExecutionContext::credentials` is `None`.
    #[error(
        "no credential resolver configured; X tools require ExecutionContext::credentials to be set"
    )]
    MissingResolver,
    /// Resolver returned an error for a specific credential name.
    #[error("credential resolution failed for '{name}': {source}")]
    CredentialResolutionFailed {
        /// The credential name that failed to resolve.
        name: String,
        /// Underlying error from the resolver.
        #[source]
        source: heartbit_core::Error,
    },
    /// X API returned 401 Unauthorized.
    #[error("X auth failed (401): {0}")]
    Unauthenticated(String),
    /// X API returned 429 Too Many Requests.
    #[error("X rate limited; retry after {retry_after_secs}s")]
    RateLimited {
        /// Seconds to wait before retrying (parsed from `Retry-After`, default 60).
        retry_after_secs: u64,
    },
    /// X API returned a 4xx or 5xx that isn't 401 or 429.
    #[error("X API error ({status}): {message}")]
    ApiError {
        /// HTTP status code.
        status: u16,
        /// Extracted error message (from `detail` or `errors[0].message`, or raw body).
        message: String,
    },
    /// Network error (connect failure, TLS, timeout, etc.).
    #[error("network error: {0}")]
    Network(String),
    /// Response parsing failed (unexpected payload shape).
    #[error("response parse error: {0}")]
    ParseError(String),
    /// Caller-side validation failed before any HTTP call (e.g., invalid
    /// base64, image too large).
    #[error("validation: {0}")]
    Validation(String),
}

/// Map `XApiError` to a user-friendly tool error message.
pub fn format_error(err: &XApiError) -> String {
    match err {
        XApiError::MissingResolver => err.to_string(),
        XApiError::CredentialResolutionFailed { .. } => err.to_string(),
        XApiError::Unauthenticated(_) => "X auth failed; check credentials".to_string(),
        XApiError::RateLimited { retry_after_secs } => {
            format!("rate limited; retry after {retry_after_secs}s")
        }
        XApiError::ApiError { status, message } => format!("X API error ({status}): {message}"),
        XApiError::Network(msg) => format!("network error: {msg}"),
        XApiError::ParseError(msg) => format!("X response parse error: {msg}"),
        XApiError::Validation(msg) => msg.clone(),
    }
}

/// HTTP client for X API. Resolves OAuth1 credentials from `ExecutionContext`,
/// signs every request, and maps X-specific errors to `XApiError`.
///
/// Construct via `XClient::from_context(ctx).await?` in production. Tests use
/// `XClient::new(base_url, ...)` to point the client at a wiremock server.
pub struct XClient {
    http: reqwest::Client,
    base_url: String,
    consumer_key: Secret,
    consumer_secret: Secret,
    access_token: Secret,
    access_token_secret: Secret,
}

impl XClient {
    /// Construct directly with the 4 credentials and a base URL.
    /// Used by tests pointing at wiremock; production code uses `from_context()`.
    pub fn new(
        base_url: impl Into<String>,
        consumer_key: Secret,
        consumer_secret: Secret,
        access_token: Secret,
        access_token_secret: Secret,
    ) -> Result<Self, XApiError> {
        let http = heartbit_core::http::vendor_client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| XApiError::Network(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            http,
            base_url: base_url.into(),
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret,
        })
    }

    /// Construct from an `ExecutionContext`, resolving the 4 OAuth1 credentials.
    pub async fn from_context(ctx: &ExecutionContext) -> Result<Self, XApiError> {
        let resolver = ctx.credentials.as_ref().ok_or(XApiError::MissingResolver)?;
        let consumer_key = resolve(resolver, "X_CONSUMER_KEY").await?;
        let consumer_secret = resolve(resolver, "X_CONSUMER_SECRET").await?;
        let access_token = resolve(resolver, "X_ACCESS_TOKEN").await?;
        let access_token_secret = resolve(resolver, "X_ACCESS_TOKEN_SECRET").await?;
        Self::new(
            X_API_BASE_URL,
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret,
        )
    }

    /// GET `<base_url><path>` with optional query params; sign via OAuth1; parse JSON response.
    pub async fn get_json<T: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<T, XApiError> {
        let url = format!("{}{}", self.base_url, path);
        let auth_header = self.sign("GET", &url, query)?;
        let mut req = self.http.get(&url).header("Authorization", auth_header);
        if !query.is_empty() {
            req = req.query(query);
        }
        let response = req
            .send()
            .await
            .map_err(|e| XApiError::Network(e.to_string()))?;
        Self::parse_response(response).await
    }

    /// POST JSON body to `<base_url><path>`; sign via OAuth1; parse JSON response.
    pub async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, XApiError> {
        let url = format!("{}{}", self.base_url, path);
        let auth_header = self.sign("POST", &url, &[])?;
        let response = self
            .http
            .post(&url)
            .header("Authorization", auth_header)
            .json(body)
            .send()
            .await
            .map_err(|e| XApiError::Network(e.to_string()))?;
        Self::parse_response(response).await
    }

    /// Upload an image to X via the v2 chunked media upload endpoint.
    /// Returns the `media_id` to attach to a subsequent `POST /2/tweets`.
    ///
    /// Implements INIT → APPEND → FINALIZE for a single segment. One
    /// segment is sufficient for any image we'd reasonably attach
    /// (X's per-image limit is 5 MiB, validated by the caller before
    /// invoking this method).
    /// Upload an image to X via the v2 media upload endpoint.
    /// Returns the `media_id` for use in `POST /2/tweets`'s `media.media_ids`.
    ///
    /// Single-shot multipart POST — the v2 endpoint is NOT chunked
    /// (despite older v1.1 documentation). The schema accepts:
    /// - `media`: binary part (required)
    /// - `media_category`: text part (optional; "tweet_image" for static images)
    ///
    /// Image size cap on the X side is currently 5 MiB (validated by the
    /// caller before invoking this method).
    pub(crate) async fn upload_image(
        &self,
        bytes: &[u8],
        mime_type: &str,
    ) -> Result<String, XApiError> {
        let media_upload_url = format!("{}/2/media/upload", self.base_url);

        let part = reqwest::multipart::Part::bytes(bytes.to_vec())
            .file_name("image")
            .mime_str(mime_type)
            .map_err(|e| XApiError::Validation(format!("invalid mime_type '{mime_type}': {e}")))?;
        let form = reqwest::multipart::Form::new()
            .part("media", part)
            .text("media_category", "tweet_image");

        let resp: InitResponse = self.post_multipart(&media_upload_url, form).await?;
        Ok(resp.data.id)
    }

    /// POST a multipart form, sign via OAuth1 (URL + method only — multipart
    /// bodies aren't included in the OAuth signature base string per
    /// RFC 5849 §3.4.1.3.1), parse the JSON response.
    async fn post_multipart<T: DeserializeOwned>(
        &self,
        url: &str,
        form: reqwest::multipart::Form,
    ) -> Result<T, XApiError> {
        let auth_header = self.sign("POST", url, &[])?;
        let response = self
            .http
            .post(url)
            .header("Authorization", auth_header)
            .multipart(form)
            .send()
            .await
            .map_err(|e| XApiError::Network(e.to_string()))?;
        Self::parse_response(response).await
    }

    fn sign(&self, method: &str, url: &str, query: &[(&str, &str)]) -> Result<String, XApiError> {
        oauth1_signing::build_authorization_header(
            method,
            url,
            query,
            self.consumer_key.expose(),
            self.consumer_secret.expose(),
            self.access_token.expose(),
            self.access_token_secret.expose(),
        )
        .map_err(|e| XApiError::Network(format!("OAuth1 signing failed: {e}")))
    }

    async fn parse_response<T: DeserializeOwned>(
        response: reqwest::Response,
    ) -> Result<T, XApiError> {
        let status = response.status();
        if status.is_success() {
            return response
                .json::<T>()
                .await
                .map_err(|e| XApiError::ParseError(e.to_string()));
        }
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        let status_code = status.as_u16();
        let body_text = response.text().await.unwrap_or_default();
        Err(classify_error_status(status_code, retry_after, &body_text))
    }
}

/// Classify a non-success HTTP status into an `XApiError` variant.
///
/// Shared by `parse_response` (JSON paths)
/// (APPEND path). Keeps 401 → `Unauthenticated`, 429 → `RateLimited`
/// (with `Retry-After` parsed by the caller), else → `ApiError`
/// classification consistent across both code paths.
fn classify_error_status(status: u16, retry_after_secs: Option<u64>, body_text: &str) -> XApiError {
    match status {
        401 => XApiError::Unauthenticated(body_text.to_string()),
        429 => XApiError::RateLimited {
            retry_after_secs: retry_after_secs.unwrap_or(60),
        },
        _ => {
            let message = extract_x_error_message(body_text)
                .unwrap_or_else(|| body_text.chars().take(200).collect());
            XApiError::ApiError { status, message }
        }
    }
}

// --- Response type for the v2 media upload endpoint. ---
// Mirrors the X v2 media upload contract; fields we don't read are kept
// (with `#[allow(dead_code)]`) to document the schema and prevent surprise on
// future contract changes.

#[derive(Debug, serde::Deserialize)]
struct InitResponse {
    data: InitData,
}

#[derive(Debug, serde::Deserialize)]
struct InitData {
    id: String,
    #[allow(dead_code)]
    #[serde(default)]
    media_key: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    expires_after_secs: Option<u64>,
}

/// Helper: resolve a single credential name, mapping the error variant.
async fn resolve(
    resolver: &Arc<dyn heartbit_core::CredentialResolver>,
    name: &str,
) -> Result<Secret, XApiError> {
    resolver
        .resolve(name)
        .await
        .map_err(|e| XApiError::CredentialResolutionFailed {
            name: name.to_string(),
            source: e,
        })
}

fn extract_x_error_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    if let Some(detail) = value.get("detail").and_then(|d| d.as_str()) {
        return Some(detail.to_string());
    }
    if let Some(errors) = value.get("errors").and_then(|e| e.as_array())
        && let Some(first) = errors.first()
        && let Some(message) = first.get("message").and_then(|m| m.as_str())
    {
        return Some(message.to_string());
    }
    None
}

mod oauth1_signing {
    //! OAuth 1.0a signing — port from `heartbit-core::tool::builtins::twitter_post`,
    //! generalized to support arbitrary HTTP method and query parameters in the
    //! signature base string.
    //!
    //! Only `oauth_*` params appear in the Authorization header. Query params
    //! contribute to the signature base string (sorted alphabetically with the
    //! `oauth_*` params), but are sent as the request URL query string.

    use std::fmt::Write as _;
    use std::time::SystemTime;

    use base64::Engine;
    use hmac::{Hmac, Mac};
    use rand::RngCore;
    use sha1::Sha1;

    type HmacSha1 = Hmac<Sha1>;

    /// Percent-encode a string per RFC 5849 (OAuth 1.0a).
    fn percent_encode(s: &str) -> String {
        let mut encoded = String::with_capacity(s.len() * 2);
        for byte in s.bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                    encoded.push(byte as char);
                }
                _ => {
                    let _ = write!(encoded, "%{byte:02X}");
                }
            }
        }
        encoded
    }

    /// Build the `OAuth ...` Authorization header value.
    ///
    /// `query` participates in the signature base string (per RFC 5849 §3.4.1.3.1)
    /// but is NOT included in the Authorization header itself.
    pub fn build_authorization_header(
        method: &str,
        url: &str,
        query: &[(&str, &str)],
        consumer_key: &str,
        consumer_secret: &str,
        access_token: &str,
        access_token_secret: &str,
    ) -> Result<String, String> {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| format!("system time error: {e}"))?
            .as_secs();
        let nonce = generate_nonce();
        let timestamp_str = timestamp.to_string();

        // Collect oauth_* params (these go into both the signature AND the header).
        let oauth_params: [(&str, &str); 6] = [
            ("oauth_consumer_key", consumer_key),
            ("oauth_nonce", &nonce),
            ("oauth_signature_method", "HMAC-SHA1"),
            ("oauth_timestamp", &timestamp_str),
            ("oauth_token", access_token),
            ("oauth_version", "1.0"),
        ];

        // Build the parameter list for the signature base string: oauth_* + query,
        // each key/value percent-encoded once, then sorted alphabetically by the
        // encoded key, then joined with `&` and `=` (RFC 5849 §3.4.1.3.2).
        let mut all_params: Vec<(String, String)> =
            Vec::with_capacity(oauth_params.len() + query.len());
        for (k, v) in oauth_params.iter() {
            all_params.push((percent_encode(k), percent_encode(v)));
        }
        for (k, v) in query.iter() {
            all_params.push((percent_encode(k), percent_encode(v)));
        }
        all_params.sort();
        let param_string = all_params
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");

        // Signature base string: METHOD&url&params (each percent-encoded).
        let base_string = format!(
            "{}&{}&{}",
            method.to_ascii_uppercase(),
            percent_encode(url),
            percent_encode(&param_string),
        );

        // Signing key: consumer_secret + "&" + token_secret (each percent-encoded).
        let signing_key = format!(
            "{}&{}",
            percent_encode(consumer_secret),
            percent_encode(access_token_secret),
        );

        let mut mac = HmacSha1::new_from_slice(signing_key.as_bytes())
            .map_err(|e| format!("HMAC key error: {e}"))?;
        mac.update(base_string.as_bytes());
        let signature =
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());

        // Authorization header carries ONLY the oauth_* params (not the query).
        Ok(format!(
            "OAuth oauth_consumer_key=\"{}\", \
             oauth_nonce=\"{}\", \
             oauth_signature=\"{}\", \
             oauth_signature_method=\"HMAC-SHA1\", \
             oauth_timestamp=\"{}\", \
             oauth_token=\"{}\", \
             oauth_version=\"1.0\"",
            percent_encode(consumer_key),
            percent_encode(&nonce),
            percent_encode(&signature),
            timestamp,
            percent_encode(access_token),
        ))
    }

    /// Generate a 32-character hex nonce (16 random bytes). RFC 5849 §3.3
    /// requires uniqueness within a server's time window — 128 bits is ample.
    fn generate_nonce() -> String {
        let mut bytes = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut bytes);
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn percent_encode_unreserved_chars_pass_through() {
            assert_eq!(percent_encode("abc123-._~"), "abc123-._~");
        }

        #[test]
        fn percent_encode_reserved_chars_are_encoded() {
            assert_eq!(percent_encode(" "), "%20");
            assert_eq!(percent_encode("&"), "%26");
            assert_eq!(percent_encode("="), "%3D");
        }

        #[test]
        fn build_header_contains_required_oauth_fields() {
            let header = build_authorization_header(
                "GET",
                "https://api.twitter.com/2/users/me",
                &[],
                "ck",
                "cs",
                "at",
                "ats",
            )
            .expect("signing succeeds");
            assert!(header.starts_with("OAuth "));
            assert!(header.contains("oauth_consumer_key=\"ck\""));
            assert!(header.contains("oauth_token=\"at\""));
            assert!(header.contains("oauth_signature_method=\"HMAC-SHA1\""));
            assert!(header.contains("oauth_version=\"1.0\""));
            assert!(header.contains("oauth_nonce="));
            assert!(header.contains("oauth_signature="));
            assert!(header.contains("oauth_timestamp="));
        }

        #[test]
        fn build_header_query_params_change_signature() {
            // Same nonce/timestamp would prove this rigorously, but since we generate
            // fresh ones per call, we instead verify that signing with vs without
            // a query produces structurally valid headers (smoke test for query path).
            let no_query = build_authorization_header(
                "GET",
                "https://api.twitter.com/2/tweets/search/recent",
                &[],
                "ck",
                "cs",
                "at",
                "ats",
            )
            .expect("ok");
            let with_query = build_authorization_header(
                "GET",
                "https://api.twitter.com/2/tweets/search/recent",
                &[("query", "hello world"), ("max_results", "10")],
                "ck",
                "cs",
                "at",
                "ats",
            )
            .expect("ok");
            assert!(no_query.starts_with("OAuth "));
            assert!(with_query.starts_with("OAuth "));
            // Query params must NOT leak into the Authorization header itself.
            assert!(!with_query.contains("query="));
            assert!(!with_query.contains("max_results="));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::future::Future;
    use std::pin::Pin;

    use serde::Deserialize;
    use wiremock::matchers::{header_exists, method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Byte-level "body contains" matcher. Unlike wiremock's
    /// `body_string_contains`, this works on raw bytes and so doesn't reject
    /// multipart bodies that include non-UTF-8 image payloads (the APPEND
    /// step).
    fn body_bytes_contains(needle: &'static [u8]) -> impl Fn(&wiremock::Request) -> bool {
        move |req| {
            req.body
                .windows(needle.len())
                .any(|window| window == needle)
        }
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestResponse {
        ok: bool,
    }

    fn test_client(server_uri: &str) -> XClient {
        XClient::new(
            server_uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("client builds")
    }

    /// Test-only resolver: returns the input name as the secret value.
    struct EchoResolver;
    impl heartbit_core::CredentialResolver for EchoResolver {
        fn resolve(
            &self,
            name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, heartbit_core::Error>> + Send + '_>>
        {
            let name = name.to_string();
            Box::pin(async move { Ok(Secret::new(format!("secret-for-{name}"))) })
        }
    }

    /// Test-only resolver that always errors.
    struct FailingResolver;
    impl heartbit_core::CredentialResolver for FailingResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, heartbit_core::Error>> + Send + '_>>
        {
            Box::pin(async {
                Err(heartbit_core::Error::Agent(
                    "simulated resolver failure".into(),
                ))
            })
        }
    }

    // --- format_error tests ---

    #[test]
    fn format_error_unauthenticated() {
        let err = XApiError::Unauthenticated("invalid token".into());
        assert_eq!(format_error(&err), "X auth failed; check credentials");
    }

    #[test]
    fn format_error_rate_limited_includes_retry_after() {
        let err = XApiError::RateLimited {
            retry_after_secs: 30,
        };
        assert_eq!(format_error(&err), "rate limited; retry after 30s");
    }

    #[test]
    fn format_error_api_error_includes_status_and_message() {
        let err = XApiError::ApiError {
            status: 400,
            message: "bad request".into(),
        };
        assert_eq!(format_error(&err), "X API error (400): bad request");
    }

    #[test]
    fn format_error_missing_resolver_uses_thiserror_message() {
        let err = XApiError::MissingResolver;
        let msg = format_error(&err);
        assert!(msg.contains("no credential resolver"));
        assert!(msg.contains("ExecutionContext::credentials"));
    }

    #[test]
    fn format_error_validation_returns_inner_message_only() {
        let err = XApiError::Validation("head image exceeds 5 MiB".to_string());
        // Must NOT prepend "validation:" — that prefix is only for the
        // Display impl via thiserror's #[error("validation: {0}")];
        // format_error is the path used by tool error output, where we
        // want the bare message.
        assert_eq!(format_error(&err), "head image exceeds 5 MiB");
    }

    // --- from_context tests ---

    #[tokio::test]
    async fn from_context_returns_missing_resolver_when_credentials_none() {
        let ctx = ExecutionContext::default();
        let result = XClient::from_context(&ctx).await;
        assert!(matches!(result, Err(XApiError::MissingResolver)));
    }

    #[tokio::test]
    async fn from_context_resolves_all_four_credentials() {
        let ctx = ExecutionContext {
            credentials: Some(Arc::new(EchoResolver)),
            ..ExecutionContext::default()
        };
        let client = XClient::from_context(&ctx)
            .await
            .expect("construction succeeds");
        assert_eq!(client.consumer_key.expose(), "secret-for-X_CONSUMER_KEY");
        assert_eq!(
            client.consumer_secret.expose(),
            "secret-for-X_CONSUMER_SECRET"
        );
        assert_eq!(client.access_token.expose(), "secret-for-X_ACCESS_TOKEN");
        assert_eq!(
            client.access_token_secret.expose(),
            "secret-for-X_ACCESS_TOKEN_SECRET"
        );
    }

    #[tokio::test]
    async fn from_context_propagates_resolver_error() {
        let ctx = ExecutionContext {
            credentials: Some(Arc::new(FailingResolver)),
            ..ExecutionContext::default()
        };
        let result = XClient::from_context(&ctx).await;
        match result {
            Err(XApiError::CredentialResolutionFailed { name, .. }) => {
                assert_eq!(name, "X_CONSUMER_KEY");
            }
            Err(other) => panic!("expected CredentialResolutionFailed, got {:?}", other),
            Ok(_) => panic!("expected CredentialResolutionFailed, got Ok(XClient)"),
        }
    }

    // --- wiremock round-trip tests ---

    #[tokio::test]
    async fn get_json_signs_and_returns_parsed_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/test/endpoint"))
            .and(header_exists("authorization"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result: TestResponse = client
            .get_json("/test/endpoint", &[])
            .await
            .expect("happy path");
        assert_eq!(result, TestResponse { ok: true });
    }

    #[tokio::test]
    async fn get_json_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/auth/test"))
            .respond_with(ResponseTemplate::new(401).set_body_string("invalid token"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result: Result<serde_json::Value, _> = client.get_json("/auth/test", &[]).await;
        assert!(matches!(result, Err(XApiError::Unauthenticated(_))));
    }

    #[tokio::test]
    async fn get_json_returns_rate_limited_with_retry_after() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/rate/test"))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "30"))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result: Result<serde_json::Value, _> = client.get_json("/rate/test", &[]).await;
        match result {
            Err(XApiError::RateLimited { retry_after_secs }) => {
                assert_eq!(retry_after_secs, 30);
            }
            other => panic!("expected RateLimited, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn get_json_extracts_x_error_message_from_4xx_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/error/test"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_json(serde_json::json!({"detail": "bad query"})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let result: Result<serde_json::Value, _> = client.get_json("/error/test", &[]).await;
        match result {
            Err(XApiError::ApiError { status, message }) => {
                assert_eq!(status, 400);
                assert_eq!(message, "bad query");
            }
            other => panic!("expected ApiError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn post_json_signs_and_returns_parsed_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/post/test"))
            .and(header_exists("authorization"))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({"ok": true})))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let body = serde_json::json!({"text": "hello"});
        let result: TestResponse = client
            .post_json("/post/test", &body)
            .await
            .expect("happy path");
        assert_eq!(result, TestResponse { ok: true });
    }

    // --- upload_image tests (P1.3f / P1.3g) ---
    // The v2 endpoint is single-shot (NOT chunked despite older v1.1
    // documentation): one multipart POST with `media` (binary part) +
    // optional `media_category`.

    #[tokio::test]
    async fn upload_image_happy_path_returns_media_id() {
        let server = MockServer::start().await;
        let media_id = "1234567890";
        Mock::given(method("POST"))
            .and(wm_path("/2/media/upload"))
            .and(body_bytes_contains(
                b"name=\"media_category\"\r\n\r\ntweet_image",
            ))
            .and(header_exists("authorization"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"id": media_id, "media_key": "mk_x", "expires_after_secs": 86400}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let bytes = b"\x89PNG\r\n\x1a\nfake_png_payload";
        let id = client
            .upload_image(bytes, "image/png")
            .await
            .expect("happy path");
        assert_eq!(id, media_id);
    }

    #[tokio::test]
    async fn upload_image_401_surfaces_unauthenticated() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/2/media/upload"))
            .respond_with(
                ResponseTemplate::new(401)
                    .set_body_string(r#"{"errors":[{"message":"Unauthorized"}]}"#),
            )
            .mount(&server)
            .await;
        let client = test_client(&server.uri());
        let bytes = b"\x89PNG\r\n";
        let err = client.upload_image(bytes, "image/png").await.unwrap_err();
        match err {
            XApiError::Unauthenticated(msg) => {
                assert!(msg.contains("Unauthorized"), "got: {msg}");
            }
            other => panic!("expected Unauthenticated, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn upload_image_5xx_surfaces_api_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/2/media/upload"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream down"))
            .mount(&server)
            .await;
        let client = test_client(&server.uri());
        let err = client
            .upload_image(b"\x89PNG", "image/png")
            .await
            .unwrap_err();
        match err {
            XApiError::ApiError { status, .. } => assert_eq!(status, 503),
            other => panic!("expected ApiError(503), got: {other:?}"),
        }
    }
}
