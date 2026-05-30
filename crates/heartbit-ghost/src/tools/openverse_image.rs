use std::future::Future;
use std::pin::Pin;

use base64::Engine;
use serde_json::json;

use heartbit_core::error::Error;
use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::tool::{Tool, ToolOutput};

/// Default Openverse API host. Tests override this via
/// [`OpenverseImageSearchTool::with_base_url`].
const DEFAULT_BASE_URL: &str = "https://api.openverse.org";

/// Marker prefix shared with [`heartbit_core::tool::builtins::ImageGenerateTool`]; downstream
/// handlers (e.g. `extract_image_marker` in the X-post pipeline) parse base64 image data
/// out of `[IMAGE:base64:<mime>;base64,<data>]`.
const IMAGE_MARKER_PREFIX: &str = "[IMAGE:base64:";

/// Literal returned when no suitable image is found. The pipeline treats this
/// (case-insensitively) as "no image" and skips attachment.
const NO_IMAGE: &str = "no_image";

/// Cap for the downloaded image body (~5 MiB). X media for jpg/png comfortably
/// fits below this; larger payloads are rejected and the next result is tried.
const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

/// Builtin tool that searches Openverse for a CC0 / public-domain image.
///
/// Queries the Openverse image search API filtered to `license=cc0,pdm`
/// (Creative Commons Zero + Public Domain Mark — neither requires attribution,
/// the safest filter for an automated brand account), downloads the first
/// usable result, and returns it as a base64 image marker in the same
/// `[IMAGE:base64:<mime>;base64,<data>]` shape emitted by
/// `ImageGenerateTool`. Returns the literal `no_image` when nothing
/// suitable is found. No API key is required for anonymous use.
pub struct OpenverseImageSearchTool {
    client: reqwest::Client,
    base_url: String,
}

impl OpenverseImageSearchTool {
    /// Create an `OpenverseImageSearchTool` pointed at the public Openverse API.
    ///
    /// Panics if the HTTP client cannot be built. Use
    /// [`OpenverseImageSearchTool::try_new`] if you need to handle the error.
    pub fn new() -> Self {
        Self::try_new().expect("failed to build reqwest client")
    }

    /// Create an `OpenverseImageSearchTool`, returning `Err` on failure.
    ///
    /// Returns `Err` if the underlying HTTP client cannot be constructed
    /// (e.g., TLS initialisation failure).
    pub fn try_new() -> Result<Self, heartbit_core::error::Error> {
        let client = heartbit_core::http::vendor_client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| {
                heartbit_core::error::Error::Agent(format!("failed to build reqwest client: {e}"))
            })?;
        Ok(Self {
            client,
            base_url: DEFAULT_BASE_URL.to_string(),
        })
    }

    /// Create an `OpenverseImageSearchTool` whose search endpoint points at the
    /// given base URL (used by tests to target a wiremock server). Image
    /// download URLs come from the search response body, so they are already
    /// rooted at the desired host.
    ///
    /// Panics if the HTTP client cannot be built.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        let client = heartbit_core::http::vendor_client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("failed to build reqwest client");
        Self {
            client,
            base_url: base_url.into(),
        }
    }

    /// Download `url`, returning `(mime, base64)` on success or `None` if the
    /// request failed, returned non-2xx, or exceeded the size cap. Errors are
    /// swallowed so the caller can fall through to the next result.
    async fn download_encoded(&self, url: &str) -> Option<(String, String)> {
        let response = self
            .client
            .get(url)
            .header("User-Agent", "heartbit-ghost/1.0")
            .send()
            .await
            .ok()?;

        if !response.status().is_success() {
            return None;
        }

        let mime = mime_from_response(&response, url);

        let (bytes, truncated) = heartbit_core::http::read_body_capped(response, MAX_IMAGE_BYTES)
            .await
            .ok()?;
        if truncated || bytes.is_empty() {
            return None;
        }

        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        Some((mime, b64))
    }
}

impl Default for OpenverseImageSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Infer the image MIME type from the response `Content-Type` header, falling
/// back to the URL extension, then to `image/jpeg`.
fn mime_from_response(response: &reqwest::Response, url: &str) -> String {
    if let Some(ct) = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        && ct.starts_with("image/")
    {
        // Drop any `; charset=...` suffix.
        let mime = ct.split(';').next().unwrap_or(ct).trim();
        return mime.to_string();
    }
    mime_from_url(url)
}

/// Infer a MIME type from a URL's file extension. Defaults to `image/jpeg`.
fn mime_from_url(url: &str) -> String {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let ext = path
        .rsplit('.')
        .next()
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "jpg" | "jpeg" => "image/jpeg",
        _ => "image/jpeg",
    }
    .to_string()
}

impl Tool for OpenverseImageSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "openverse_image_search".into(),
            description:
                "Search Openverse for a CC0/public-domain image matching the query and return it \
                 as a base64 image marker, or 'no_image' if nothing suitable. No attribution \
                 required (license=cc0,pdm)."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords for the image"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn redact_for_history(&self, output: &str) -> String {
        // Replace the [IMAGE:base64:<huge_data>] marker body with a tiny
        // placeholder so the ~MB-scale base64 payload does not re-enter the
        // conversation on the next LLM turn (which would trip Anthropic's
        // 200k-token context cap). The full marker is preserved on
        // `AgentOutput.tool_call_results` for the caller. Mirrors
        // `ImageGenerateTool::redact_for_history`.
        if let Some(start) = output.find(IMAGE_MARKER_PREFIX) {
            let after_prefix = &output[start + IMAGE_MARKER_PREFIX.len()..];
            if let Some(end) = after_prefix.find(']') {
                let body = &after_prefix[..end];
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(body.as_bytes());
                let hex = format!("{:x}", hasher.finalize());
                let short = &hex[..12];
                let placeholder = format!("[IMAGE:redacted:{short}]");
                let mut redacted =
                    String::with_capacity(output.len() - body.len() + placeholder.len());
                redacted.push_str(&output[..start]);
                redacted.push_str(&placeholder);
                redacted.push_str(&after_prefix[end + 1..]);
                return redacted;
            }
        }
        output.to_string()
    }

    fn execute(
        &self,
        _ctx: &heartbit_core::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let query = input
                .get("query")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|q| !q.is_empty());
            let query = match query {
                Some(q) => q,
                None => return Ok(ToolOutput::error("query required")),
            };

            let endpoint = format!("{}/v1/images/", self.base_url.trim_end_matches('/'));
            let response = self
                .client
                .get(&endpoint)
                .query(&[
                    ("q", query),
                    ("license", "cc0,pdm"),
                    ("page_size", "8"),
                    ("extension", "jpg,png"),
                ])
                .header("User-Agent", "heartbit-ghost/1.0")
                .send()
                .await
                .map_err(|e| Error::Agent(format!("openverse search failed: {e}")))?;

            let status = response.status();
            if !status.is_success() {
                let body = heartbit_core::http::read_text_capped(response, 4 * 1024)
                    .await
                    .unwrap_or_default();
                return Ok(ToolOutput::error(format!(
                    "openverse search failed: HTTP {} {body}",
                    status.as_u16()
                )));
            }

            let (bytes, _) = heartbit_core::http::read_body_capped(
                response,
                heartbit_core::http::DEFAULT_VENDOR_BODY_CAP,
            )
            .await
            .map_err(|e| Error::Agent(format!("openverse search failed: {e}")))?;
            let data: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Agent(format!("openverse search failed: {e}")))?;

            let results = data.get("results").and_then(|v| v.as_array());
            let results = match results {
                Some(r) if !r.is_empty() => r,
                _ => return Ok(ToolOutput::success(NO_IMAGE)),
            };

            for result in results {
                let url = match result.get("url").and_then(|v| v.as_str()) {
                    Some(u) if !u.is_empty() => u,
                    _ => continue,
                };
                if let Some((mime, b64)) = self.download_encoded(url).await {
                    return Ok(ToolOutput::success(format!(
                        "{IMAGE_MARKER_PREFIX}{mime};base64,{b64}]\n\n\
                         Image from Openverse for: {query}"
                    )));
                }
            }

            // Every candidate failed to download.
            Ok(ToolOutput::success(NO_IMAGE))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn definition_has_correct_name() {
        let tool = OpenverseImageSearchTool::new();
        assert_eq!(tool.definition().name, "openverse_image_search");
    }

    #[test]
    fn definition_requires_query() {
        let tool = OpenverseImageSearchTool::new();
        let def = tool.definition();
        let required = def.input_schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "query");
    }

    #[test]
    fn mime_from_url_extensions() {
        assert_eq!(mime_from_url("https://x/img.png"), "image/png");
        assert_eq!(mime_from_url("https://x/img.jpg"), "image/jpeg");
        assert_eq!(mime_from_url("https://x/img.jpeg?w=1"), "image/jpeg");
        assert_eq!(mime_from_url("https://x/noext"), "image/jpeg");
    }

    #[test]
    fn redact_for_history_replaces_marker_body() {
        let tool = OpenverseImageSearchTool::new();
        let raw = "[IMAGE:base64:image/jpeg;base64,iVBORw0KGgo=]\n\nImage from Openverse for: cats";
        let redacted = tool.redact_for_history(raw);
        assert!(redacted.starts_with("[IMAGE:redacted:"));
        assert!(redacted.contains("Image from Openverse for: cats"));
        assert!(!redacted.contains("iVBORw0KGgo"));
        // Idempotent: same input → same hash.
        assert_eq!(tool.redact_for_history(raw), redacted);
    }

    #[tokio::test]
    async fn returns_base64_marker_on_hit() {
        let server = MockServer::start().await;

        let img_url = format!("{}/img.jpg", server.uri());
        Mock::given(method("GET"))
            .and(wm_path("/v1/images/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "result_count": 1,
                "results": [ { "id": "1", "title": "t", "url": img_url, "license": "cc0" } ]
            })))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(wm_path("/img.jpg"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(vec![1u8, 2, 3, 4, 5])
                    .insert_header("Content-Type", "image/jpeg"),
            )
            .mount(&server)
            .await;

        let tool = OpenverseImageSearchTool::with_base_url(server.uri());
        let result = tool
            .execute(
                &heartbit_core::ExecutionContext::default(),
                json!({"query": "cats"}),
            )
            .await
            .expect("ok");
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(
            result
                .content
                .starts_with("[IMAGE:base64:image/jpeg;base64,"),
            "marker mismatch: {}",
            result.content
        );
        // Verify the base64 payload round-trips and downstream parsing strips
        // the mime prefix to the canonical base64.
        let expected_b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3, 4, 5]);
        assert!(result.content.contains(&expected_b64));
    }

    #[tokio::test]
    async fn returns_no_image_when_no_results() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/v1/images/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "result_count": 0,
                "results": []
            })))
            .mount(&server)
            .await;

        let tool = OpenverseImageSearchTool::with_base_url(server.uri());
        let result = tool
            .execute(
                &heartbit_core::ExecutionContext::default(),
                json!({"query": "nothing"}),
            )
            .await
            .expect("ok");
        assert!(!result.is_error);
        assert_eq!(result.content, "no_image");
    }

    #[tokio::test]
    async fn errors_on_empty_query() {
        let tool = OpenverseImageSearchTool::new();

        let result = tool
            .execute(&heartbit_core::ExecutionContext::default(), json!({}))
            .await
            .expect("ok");
        assert!(result.is_error);
        assert!(result.content.contains("query required"));

        let result = tool
            .execute(
                &heartbit_core::ExecutionContext::default(),
                json!({"query": "   "}),
            )
            .await
            .expect("ok");
        assert!(result.is_error);
        assert!(result.content.contains("query required"));
    }

    #[tokio::test]
    async fn skips_failing_download_tries_next() {
        let server = MockServer::start().await;

        let bad_url = format!("{}/missing.jpg", server.uri());
        let good_url = format!("{}/good.png", server.uri());
        Mock::given(method("GET"))
            .and(wm_path("/v1/images/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "result_count": 2,
                "results": [
                    { "id": "1", "url": bad_url, "license": "cc0" },
                    { "id": "2", "url": good_url, "license": "pdm" }
                ]
            })))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(wm_path("/missing.jpg"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(wm_path("/good.png"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(vec![9u8; 16])
                    .insert_header("Content-Type", "image/png"),
            )
            .mount(&server)
            .await;

        let tool = OpenverseImageSearchTool::with_base_url(server.uri());
        let result = tool
            .execute(
                &heartbit_core::ExecutionContext::default(),
                json!({"query": "dogs"}),
            )
            .await
            .expect("ok");
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(
            result
                .content
                .starts_with("[IMAGE:base64:image/png;base64,"),
            "should use the second result's png: {}",
            result.content
        );
    }
}
