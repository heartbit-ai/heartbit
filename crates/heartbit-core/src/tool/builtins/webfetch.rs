use std::future::Future;
use std::pin::Pin;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const MAX_RESPONSE_BYTES: usize = 5 * 1024 * 1024; // 5 MB
const MAX_OUTPUT_CHARS: usize = 50_000;
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const MAX_TIMEOUT_SECS: u64 = 120;

/// Builtin tool that fetches a URL and returns the response body as text.
///
/// Designed for agents that need to retrieve documentation, APIs, or web pages.
/// Responses are capped at 5 MB and then character-truncated to 50 000 chars;
/// HTML is not stripped — the agent receives the raw body. By default the tool
/// blocks requests to private/link-local IP ranges (`IpPolicy::Strict`) to
/// prevent SSRF; set `HEARTBIT_ALLOW_PRIVATE_IPS=1` to relax this for local
/// development.
pub struct WebFetchTool {
    client: reqwest::Client,
    ip_policy: crate::http::IpPolicy,
}

impl WebFetchTool {
    /// Construct with `IpPolicy::default()` — `Strict` unless
    /// `HEARTBIT_ALLOW_PRIVATE_IPS=1` is set in the environment.
    ///
    /// Panics if the HTTP client cannot be built. Use [`WebFetchTool::try_new`]
    /// if you need to handle the error.
    pub fn new() -> Self {
        Self::try_with_ip_policy(crate::http::IpPolicy::default())
            .expect("failed to build reqwest client")
    }

    /// Construct with `IpPolicy::default()`, returning `Err` on failure.
    ///
    /// Returns `Err` if the underlying HTTP client cannot be constructed
    /// (e.g., TLS initialisation failure).
    #[allow(dead_code)]
    pub fn try_new() -> Result<Self, crate::error::Error> {
        Self::try_with_ip_policy(crate::http::IpPolicy::default())
    }

    /// Construct with an explicit IP policy.
    ///
    /// Use `IpPolicy::AllowPrivate` only for single-tenant / dev
    /// deployments where the agent legitimately needs to access internal
    /// services.
    ///
    /// Panics if the HTTP client cannot be built. Use [`WebFetchTool::try_with_ip_policy`]
    /// if you need to handle the error.
    #[allow(dead_code)]
    pub fn with_ip_policy(ip_policy: crate::http::IpPolicy) -> Self {
        Self::try_with_ip_policy(ip_policy).expect("failed to build reqwest client")
    }

    /// Construct with an explicit IP policy, returning `Err` on failure.
    ///
    /// Returns `Err` if the underlying HTTP client cannot be constructed.
    pub fn try_with_ip_policy(
        ip_policy: crate::http::IpPolicy,
    ) -> Result<Self, crate::error::Error> {
        let client = crate::http::safe_client_builder()
            .user_agent("heartbit/0.1")
            .build()
            .map_err(|e| {
                crate::error::Error::Agent(format!("failed to build reqwest client: {e}"))
            })?;
        Ok(Self { client, ip_policy })
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WebFetchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "webfetch".into(),
            description: "Fetch content from a URL via HTTP GET. Supports text, markdown, \
                          and HTML output formats. Max response: 5 MB."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["text", "markdown", "html"],
                        "description": "Output format (default: markdown)"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default 30, max 120)"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let url = input
                .get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("url is required".into()))?;

            let format = input
                .get("format")
                .and_then(|v| v.as_str())
                .unwrap_or("markdown");

            let timeout_secs = input
                .get("timeout")
                .and_then(|v| v.as_u64())
                .unwrap_or(DEFAULT_TIMEOUT_SECS)
                .min(MAX_TIMEOUT_SECS);

            // Validate scheme + private-IP blocklist via crate::http::SafeUrl.
            let safe_url = match crate::http::SafeUrl::parse(url, self.ip_policy).await {
                Ok(u) => u,
                Err(e) => return Ok(ToolOutput::error(e.to_string())),
            };

            let response = self
                .client
                .get(safe_url.as_str())
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .send()
                .await
                .map_err(|e| Error::Agent(format!("HTTP request failed: {e}")))?;

            let status = response.status();
            if !status.is_success() {
                return Ok(ToolOutput::error(format!(
                    "HTTP {}: {}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or("Unknown")
                )));
            }

            // Pre-check Content-Length if available
            if let Some(len) = response.content_length()
                && len > MAX_RESPONSE_BYTES as u64
            {
                return Ok(ToolOutput::error(format!(
                    "Response too large ({len} bytes). Maximum: {MAX_RESPONSE_BYTES} bytes."
                )));
            }

            // Stream body with size limit (Content-Length can be absent or wrong)
            let mut bytes = Vec::new();
            let mut stream = response.bytes_stream();
            use futures::StreamExt;
            while let Some(chunk) = stream.next().await {
                let chunk =
                    chunk.map_err(|e| Error::Agent(format!("Failed to read response: {e}")))?;
                bytes.extend_from_slice(&chunk);
                if bytes.len() > MAX_RESPONSE_BYTES {
                    return Ok(ToolOutput::error(format!(
                        "Response too large (>{MAX_RESPONSE_BYTES} bytes). Download aborted."
                    )));
                }
            }

            let body = String::from_utf8_lossy(&bytes).to_string();

            let output = match format {
                "html" => body,
                "text" => crate::util::strip_html_tags(&body),
                _ => html_to_markdown(&body),
            };

            // Truncate if needed
            let output = if output.len() > MAX_OUTPUT_CHARS {
                let cut = super::floor_char_boundary(&output, MAX_OUTPUT_CHARS);
                let omitted = output.len() - cut;
                format!("{}\n\n[truncated: {omitted} chars omitted]", &output[..cut])
            } else {
                output
            };

            Ok(ToolOutput::success(format!(
                "Fetched {url} (HTTP {}):\n\n{output}",
                status.as_u16()
            )))
        })
    }
}

/// Simple HTML to markdown conversion.
///
/// Preserves headers, links, paragraphs, and lists. Strips other tags.
/// Skips content inside `<script>` and `<style>` tags.
fn html_to_markdown(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut tag_name = String::new();
    let mut collecting_tag = false;
    let mut last_was_space = false;
    let mut skip_content = false; // true inside <script> or <style>

    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
            tag_name.clear();
            collecting_tag = true;
        } else if ch == '>' && in_tag {
            in_tag = false;
            collecting_tag = false;

            let tag_lower = tag_name.to_lowercase();

            // Check for script/style end tags before anything else
            match tag_lower.as_str() {
                "/script" | "/style" => {
                    skip_content = false;
                    continue;
                }
                "script" | "style" => {
                    skip_content = true;
                    continue;
                }
                _ => {}
            }

            if skip_content {
                continue;
            }

            // Map HTML tags to markdown
            match tag_lower.as_str() {
                "h1" => result.push_str("\n# "),
                "h2" => result.push_str("\n## "),
                "h3" => result.push_str("\n### "),
                "h4" => result.push_str("\n#### "),
                "h5" => result.push_str("\n##### "),
                "h6" => result.push_str("\n###### "),
                "/h1" | "/h2" | "/h3" | "/h4" | "/h5" | "/h6" => result.push('\n'),
                "p" | "/p" | "br" | "br/" => {
                    if !result.ends_with('\n') {
                        result.push('\n');
                    }
                }
                "li" => result.push_str("\n- "),
                "/li" => {}
                "strong" | "b" => result.push_str("**"),
                "/strong" | "/b" => result.push_str("**"),
                "em" | "i" => result.push('*'),
                "/em" | "/i" => result.push('*'),
                "code" => result.push('`'),
                "/code" => result.push('`'),
                "pre" => result.push_str("\n```\n"),
                "/pre" => result.push_str("\n```\n"),
                _ => {
                    // For other tags, add a space to separate content
                    if !last_was_space && !result.is_empty() {
                        result.push(' ');
                        last_was_space = true;
                    }
                }
            }
        } else if in_tag && collecting_tag {
            if ch.is_whitespace() {
                collecting_tag = false; // Stop collecting after tag name (attributes follow)
            } else {
                tag_name.push(ch);
            }
        } else if !in_tag && !skip_content {
            if ch.is_whitespace() {
                if !last_was_space {
                    result.push(if ch == '\n' { '\n' } else { ' ' });
                    last_was_space = true;
                }
            } else {
                result.push(ch);
                last_was_space = false;
            }
        }
    }

    // Clean up excessive newlines
    while result.contains("\n\n\n") {
        result = result.replace("\n\n\n", "\n\n");
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = WebFetchTool::new();
        assert_eq!(tool.definition().name, "webfetch");
    }

    #[test]
    fn html_to_markdown_headers() {
        let html = "<h1>Title</h1><h2>Subtitle</h2>";
        let md = html_to_markdown(html);
        assert!(md.contains("# Title"));
        assert!(md.contains("## Subtitle"));
    }

    #[test]
    fn html_to_markdown_paragraphs() {
        let html = "<p>First paragraph</p><p>Second paragraph</p>";
        let md = html_to_markdown(html);
        assert!(md.contains("First paragraph"));
        assert!(md.contains("Second paragraph"));
    }

    #[test]
    fn html_to_markdown_links_stripped() {
        // Simple version: links are stripped to just text
        let html = "<a href=\"https://example.com\">link text</a>";
        let md = html_to_markdown(html);
        assert!(md.contains("link text"));
    }

    #[test]
    fn html_to_markdown_code() {
        let html = "<code>foo</code>";
        let md = html_to_markdown(html);
        assert!(md.contains("`foo`"));
    }

    #[test]
    fn html_to_markdown_skips_script_content() {
        let html = "<p>Hello</p><script>var x = 1; alert('xss');</script><p>World</p>";
        let md = html_to_markdown(html);
        assert!(md.contains("Hello"));
        assert!(md.contains("World"));
        assert!(!md.contains("alert"));
        assert!(!md.contains("var x"));
    }

    #[test]
    fn html_to_markdown_skips_style_content() {
        let html = "<p>Hello</p><style>body { color: red; }</style><p>World</p>";
        let md = html_to_markdown(html);
        assert!(md.contains("Hello"));
        assert!(md.contains("World"));
        assert!(!md.contains("color"));
    }

    #[tokio::test]
    async fn webfetch_rejects_file_scheme() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "file:///etc/passwd"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("scheme") || result.content.contains("invalid URL"),
            "got: {}",
            result.content,
        );
    }

    #[tokio::test]
    async fn webfetch_rejects_ftp_scheme() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "ftp://example.com/file"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("scheme") || result.content.contains("invalid URL"),
            "got: {}",
            result.content,
        );
    }

    #[test]
    fn html_to_markdown_h5_h6() {
        let html = "<h5>Heading 5</h5><h6>Heading 6</h6>";
        let md = html_to_markdown(html);
        assert!(md.contains("##### Heading 5"));
        assert!(md.contains("###### Heading 6"));
    }

    #[tokio::test]
    async fn rejects_uppercase_ftp_scheme() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "FTP://example.com/file"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("scheme") || result.content.contains("invalid URL"),
            "got: {}",
            result.content,
        );
    }

    #[tokio::test]
    async fn webfetch_rejects_loopback() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "http://127.0.0.1/"}))
            .await
            .unwrap();
        assert!(result.is_error, "loopback must be rejected by default");
        assert!(
            result.content.contains("private/loopback"),
            "rejection message should explain why; got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn webfetch_rejects_imds() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "http://169.254.169.254/latest/meta-data/"}))
            .await
            .unwrap();
        assert!(result.is_error, "AWS/GCE IMDS must be rejected");
    }

    #[tokio::test]
    async fn webfetch_rejects_rfc1918() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "http://10.0.0.1/"}))
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn webfetch_rejects_localhost_dns() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(json!({"url": "http://localhost/"}))
            .await
            .unwrap();
        assert!(
            result.is_error,
            "localhost (resolves to 127.0.0.1/::1) must be rejected"
        );
    }

    #[tokio::test]
    async fn webfetch_with_allow_private_ips_does_not_reject_loopback() {
        // Use with_ip_policy directly; do NOT mutate global env in tests.
        let tool = WebFetchTool::with_ip_policy(crate::http::IpPolicy::AllowPrivate);
        // The address won't resolve to anything reachable, so the request
        // itself fails — but it should NOT fail with the SSRF rejection.
        // The request-level failure may surface as either
        // `Ok(ToolOutput::error(..))` or `Err(Error::Agent(..))`; either way
        // the message must NOT contain the private-IP rejection text.
        let outcome = tool.execute(json!({"url": "http://127.0.0.1:1/"})).await;
        let message = match outcome {
            Ok(out) => {
                assert!(out.is_error, "request to closed port should error");
                out.content
            }
            Err(e) => e.to_string(),
        };
        assert!(
            !message.contains("private/loopback"),
            "AllowPrivate should bypass the SSRF rejection; got: {message}",
        );
    }
}
