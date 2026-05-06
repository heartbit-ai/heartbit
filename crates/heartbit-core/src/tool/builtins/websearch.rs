use std::future::Future;
use std::pin::Pin;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const DEFAULT_NUM_RESULTS: u64 = 8;
const MAX_NUM_RESULTS: u64 = 50;

/// Search provider selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchProvider {
    /// Auto-detect: tries providers in priority order based on available API keys.
    /// Priority: Exa -> Tavily -> Brave -> DuckDuckGo (zero-config fallback).
    #[default]
    Auto,
    Exa,
    Tavily,
    Brave,
    DuckDuckGo,
}

/// Builtin tool that performs web searches and returns ranked result snippets.
///
/// Delegates to one of four backends selected via `SearchProvider`; the default
/// `Auto` mode tries Exa → Tavily → Brave → DuckDuckGo in priority order based
/// on which API keys are present in the environment. DuckDuckGo requires no key
/// and is always available as a zero-config fallback. Results are returned as
/// title + URL + snippet text, capped at `MAX_NUM_RESULTS = 50`.
pub struct WebSearchTool {
    client: reqwest::Client,
    provider: SearchProvider,
}

impl WebSearchTool {
    /// Create a `WebSearchTool` with automatic provider selection.
    ///
    /// Panics if the HTTP client cannot be built (TLS initialisation failure).
    /// Use [`WebSearchTool::try_new`] if you need to handle the error.
    pub fn new() -> Self {
        Self::try_new().expect("failed to build reqwest client")
    }

    /// Create a `WebSearchTool` with automatic provider selection.
    ///
    /// Returns `Err` if the underlying HTTP client cannot be constructed
    /// (e.g., TLS initialisation failure).
    pub fn try_new() -> Result<Self, crate::error::Error> {
        let client = crate::http::vendor_client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| {
                crate::error::Error::Agent(format!("failed to build reqwest client: {e}"))
            })?;
        Ok(Self {
            client,
            provider: SearchProvider::Auto,
        })
    }

    /// Create a `WebSearchTool` that uses a specific search provider
    /// instead of auto-detecting from environment variables.
    ///
    /// Panics if the HTTP client cannot be built. Use [`WebSearchTool::try_with_provider`]
    /// if you need to handle the error.
    #[allow(dead_code)] // Public API — not yet re-exported from lib.rs
    pub fn with_provider(provider: SearchProvider) -> Self {
        Self::try_with_provider(provider).expect("failed to build reqwest client")
    }

    /// Create a `WebSearchTool` that uses a specific search provider.
    ///
    /// Returns `Err` if the underlying HTTP client cannot be constructed.
    pub fn try_with_provider(provider: SearchProvider) -> Result<Self, crate::error::Error> {
        let client = crate::http::vendor_client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| {
                crate::error::Error::Agent(format!("failed to build reqwest client: {e}"))
            })?;
        Ok(Self { client, provider })
    }

    async fn search_with(
        &self,
        provider: &SearchProvider,
        query: &str,
        num_results: u64,
    ) -> Result<Vec<SearchResult>, Error> {
        match provider {
            SearchProvider::Auto => Err(Error::Agent(
                "Auto should not be passed to search_with".into(),
            )),
            SearchProvider::Exa => self.search_exa(query, num_results).await,
            SearchProvider::Tavily => self.search_tavily(query, num_results).await,
            SearchProvider::Brave => self.search_brave(query, num_results).await,
            SearchProvider::DuckDuckGo => self.search_duckduckgo(query, num_results).await,
        }
    }

    async fn search_exa(&self, query: &str, num_results: u64) -> Result<Vec<SearchResult>, Error> {
        let api_key = std::env::var("EXA_API_KEY")
            .map_err(|_| Error::Agent("EXA_API_KEY environment variable not set".into()))?;

        let body = json!({
            "query": query,
            "numResults": num_results,
            "contents": {
                "text": true
            }
        });

        let response = self
            .client
            .post("https://api.exa.ai/search")
            .header("x-api-key", &api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Agent(format!("Exa API request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            // SECURITY (F-NET-1): cap error body.
            let error_body = crate::http::read_text_capped(response, 4 * 1024)
                .await
                .unwrap_or_default();
            return Err(Error::Agent(format!(
                "Exa API error (HTTP {}): {error_body}",
                status.as_u16()
            )));
        }

        // SECURITY (F-NET-1): cap successful body. Search results can be a
        // few hundred KB; 5 MiB is a comfortable upper bound.
        let (bytes, _) =
            crate::http::read_body_capped(response, crate::http::DEFAULT_VENDOR_BODY_CAP)
                .await
                .map_err(|e| Error::Agent(format!("Failed to read Exa response: {e}")))?;
        let data: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Agent(format!("Failed to parse Exa response: {e}")))?;

        let results = data
            .get("results")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|r| SearchResult {
                        title: r
                            .get("title")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        url: r
                            .get("url")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        text: r
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(results)
    }

    async fn search_tavily(
        &self,
        query: &str,
        num_results: u64,
    ) -> Result<Vec<SearchResult>, Error> {
        let api_key = std::env::var("TAVILY_API_KEY")
            .map_err(|_| Error::Agent("TAVILY_API_KEY environment variable not set".into()))?;

        let body = json!({
            "api_key": api_key,
            "query": query,
            "max_results": num_results,
            "include_answer": false
        });

        let response = self
            .client
            .post("https://api.tavily.com/search")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Agent(format!("Tavily API request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let error_body = crate::http::read_text_capped(response, 4 * 1024)
                .await
                .unwrap_or_default();
            return Err(Error::Agent(format!(
                "Tavily API error (HTTP {}): {error_body}",
                status.as_u16()
            )));
        }

        let (bytes, _) =
            crate::http::read_body_capped(response, crate::http::DEFAULT_VENDOR_BODY_CAP)
                .await
                .map_err(|e| Error::Agent(format!("Failed to read Tavily response: {e}")))?;
        let data: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Agent(format!("Failed to parse Tavily response: {e}")))?;

        Ok(parse_tavily_results(&data))
    }

    async fn search_brave(
        &self,
        query: &str,
        num_results: u64,
    ) -> Result<Vec<SearchResult>, Error> {
        let api_key = std::env::var("BRAVE_API_KEY")
            .map_err(|_| Error::Agent("BRAVE_API_KEY environment variable not set".into()))?;

        let response = self
            .client
            .get("https://api.search.brave.com/res/v1/web/search")
            .query(&[("q", query), ("count", &num_results.to_string())])
            .header("X-Subscription-Token", &api_key)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| Error::Agent(format!("Brave API request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let error_body = crate::http::read_text_capped(response, 4 * 1024)
                .await
                .unwrap_or_default();
            return Err(Error::Agent(format!(
                "Brave API error (HTTP {}): {error_body}",
                status.as_u16()
            )));
        }

        let (bytes, _) =
            crate::http::read_body_capped(response, crate::http::DEFAULT_VENDOR_BODY_CAP)
                .await
                .map_err(|e| Error::Agent(format!("Failed to read Brave response: {e}")))?;
        let data: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Agent(format!("Failed to parse Brave response: {e}")))?;

        Ok(parse_brave_results(&data))
    }

    async fn search_duckduckgo(
        &self,
        query: &str,
        num_results: u64,
    ) -> Result<Vec<SearchResult>, Error> {
        let response = self
            .client
            .get("https://html.duckduckgo.com/html/")
            .query(&[("q", query)])
            // SECURITY (F-NET-5): generic User-Agent. The previous string
            // explicitly identified the framework, enabling search-engine
            // fingerprinting and tailored prompt injection in HTML.
            .header("User-Agent", "Mozilla/5.0 (compatible)")
            .send()
            .await
            .map_err(|e| Error::Agent(format!("DuckDuckGo request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Err(Error::Agent(format!(
                "DuckDuckGo error (HTTP {})",
                status.as_u16()
            )));
        }

        // SECURITY (F-NET-1): cap DuckDuckGo HTML response. Search HTML is
        // typically <500 KiB; 5 MiB is a generous cap.
        let html = crate::http::read_text_capped(response, crate::http::DEFAULT_VENDOR_BODY_CAP)
            .await
            .map_err(|e| Error::Agent(format!("Failed to read DuckDuckGo response: {e}")))?;

        Ok(parse_duckduckgo_html(&html, num_results))
    }
}

struct SearchResult {
    title: String,
    url: String,
    text: String,
}

fn detect_providers() -> Vec<SearchProvider> {
    let mut providers = Vec::new();
    if std::env::var("EXA_API_KEY").is_ok() {
        providers.push(SearchProvider::Exa);
    }
    if std::env::var("TAVILY_API_KEY").is_ok() {
        providers.push(SearchProvider::Tavily);
    }
    if std::env::var("BRAVE_API_KEY").is_ok() {
        providers.push(SearchProvider::Brave);
    }
    // DuckDuckGo is always available (no API key needed)
    providers.push(SearchProvider::DuckDuckGo);
    providers
}

fn parse_tavily_results(data: &serde_json::Value) -> Vec<SearchResult> {
    data.get("results")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|r| SearchResult {
                    title: r
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    url: r
                        .get("url")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    text: r
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_brave_results(data: &serde_json::Value) -> Vec<SearchResult> {
    data.get("web")
        .and_then(|v| v.get("results"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|r| SearchResult {
                    title: r
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    url: r
                        .get("url")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    text: r
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_duckduckgo_html(html: &str, max_results: u64) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let max = max_results as usize;

    // Parse result__a links for title + URL
    let mut search_start = 0;
    while results.len() < max {
        let marker = "class=\"result__a\"";
        let Some(marker_pos) = html[search_start..].find(marker) else {
            break;
        };
        let abs_marker = search_start + marker_pos;

        // Find the href attribute (look backwards for <a, then find href)
        let tag_start_region = abs_marker.saturating_sub(200);
        let tag_region = &html[tag_start_region..abs_marker];
        let href = tag_region
            .rfind("href=\"")
            .and_then(|pos| {
                let start = tag_start_region + pos + 6;
                html[start..].find('"').map(|end| &html[start..start + end])
            })
            .unwrap_or("");

        // Extract inner text: from after the '>' that closes the <a> tag to the next </a>
        let after_marker = abs_marker + marker.len();
        let title = html[after_marker..]
            .find('>')
            .and_then(|gt| {
                let text_start = after_marker + gt + 1;
                html[text_start..].find("</a>").map(|end| {
                    // Strip any inner HTML tags
                    strip_html_tags(&html[text_start..text_start + end])
                })
            })
            .unwrap_or_default();

        // Find corresponding snippet
        let snippet_marker = "class=\"result__snippet\"";
        let snippet = html[after_marker..]
            .find(snippet_marker)
            .and_then(|pos| {
                let snippet_start = after_marker + pos + snippet_marker.len();
                html[snippet_start..].find('>').and_then(|gt| {
                    let text_start = snippet_start + gt + 1;
                    // Find closing tag
                    html[text_start..]
                        .find("</")
                        .map(|end| strip_html_tags(&html[text_start..text_start + end]))
                })
            })
            .unwrap_or_default();

        // Decode the URL - DuckDuckGo wraps URLs in a redirect
        let url = decode_ddg_url(href);

        if !title.is_empty() || !url.is_empty() {
            results.push(SearchResult {
                title: title.trim().to_string(),
                url,
                text: snippet.trim().to_string(),
            });
        }

        search_start = after_marker;
    }

    results
}

/// Strip HTML tags from a string and decode common HTML entities (simple best-effort).
fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            out.push(ch);
        }
    }
    // Decode common HTML entities
    decode_html_entities(&out)
}

fn decode_html_entities(input: &str) -> String {
    input
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
        .replace("&apos;", "'")
}

/// Decode DuckDuckGo redirect URLs. They look like:
/// `//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=...`
/// We extract and decode the `uddg` parameter.
fn decode_ddg_url(href: &str) -> String {
    if let Some(uddg_start) = href.find("uddg=") {
        let value_start = uddg_start + 5;
        let value_end = href[value_start..]
            .find('&')
            .map(|pos| value_start + pos)
            .unwrap_or(href.len());
        let encoded = &href[value_start..value_end];
        url_decode(encoded)
    } else {
        href.to_string()
    }
}

/// Minimal percent-decoding for URLs (multi-byte UTF-8 safe).
fn url_decode(input: &str) -> String {
    let mut bytes = Vec::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if hex.len() == 2 {
                if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                    bytes.push(byte);
                } else {
                    bytes.push(b'%');
                    bytes.extend(hex.bytes());
                }
            } else {
                bytes.push(b'%');
                bytes.extend(hex.bytes());
            }
        } else if c == '+' {
            bytes.push(b' ');
        } else {
            let mut buf = [0u8; 4];
            let encoded = c.encode_utf8(&mut buf);
            bytes.extend(encoded.bytes());
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn format_results(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No search results found.".into();
    }

    let mut output = format!("Search results for \"{query}\":\n\n");

    for (i, result) in results.iter().enumerate() {
        let title = if result.title.is_empty() {
            "Untitled"
        } else {
            &result.title
        };

        // Truncate text snippet (char-boundary safe)
        let snippet = if result.text.len() > 500 {
            let end = super::floor_char_boundary(&result.text, 500);
            format!("{}...", &result.text[..end])
        } else {
            result.text.clone()
        };

        output.push_str(&format!(
            "{}. **{}**\n   {}\n   {}\n\n",
            i + 1,
            title,
            result.url,
            snippet.trim()
        ));
    }

    output
}

impl Tool for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "websearch".into(),
            description:
                "Search the web using multiple providers (Exa, Tavily, Brave, DuckDuckGo). \
                          Auto-detects available API keys and cascades on failure. \
                          DuckDuckGo requires no API key."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 8)"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let query = input
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("query is required".into()))?;

            let num_results = input
                .get("num_results")
                .and_then(|v| v.as_u64())
                .unwrap_or(DEFAULT_NUM_RESULTS)
                .min(MAX_NUM_RESULTS);

            let providers = match self.provider {
                SearchProvider::Auto => detect_providers(),
                specific => vec![specific],
            };

            let mut last_error = String::new();
            for provider in &providers {
                match self.search_with(provider, query, num_results).await {
                    Ok(results) => return Ok(ToolOutput::success(format_results(query, &results))),
                    Err(e) => {
                        tracing::warn!(provider = ?provider, error = %e, "search provider failed, trying next");
                        last_error = e.to_string();
                    }
                }
            }

            Ok(ToolOutput::error(format!(
                "All search providers failed. Last error: {last_error}"
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = WebSearchTool::new();
        assert_eq!(tool.definition().name, "websearch");
    }

    #[test]
    fn search_provider_default_is_auto() {
        let provider = SearchProvider::default();
        assert_eq!(provider, SearchProvider::Auto);
    }

    #[test]
    fn with_provider_constructor() {
        let tool = WebSearchTool::with_provider(SearchProvider::Brave);
        assert_eq!(tool.provider, SearchProvider::Brave);
    }

    #[test]
    fn detect_providers_with_no_keys() {
        // This test may see env vars from the environment, so we just verify
        // DuckDuckGo is always the last provider.
        let providers = detect_providers();
        assert!(!providers.is_empty());
        assert_eq!(*providers.last().unwrap(), SearchProvider::DuckDuckGo);
    }

    #[test]
    fn format_results_empty() {
        let output = format_results("test", &[]);
        assert_eq!(output, "No search results found.");
    }

    #[test]
    fn format_results_single_result() {
        let results = vec![SearchResult {
            title: "Rust Programming".into(),
            url: "https://rust-lang.org".into(),
            text: "A systems programming language.".into(),
        }];
        let output = format_results("rust", &results);
        assert!(output.contains("Search results for \"rust\""));
        assert!(output.contains("1. **Rust Programming**"));
        assert!(output.contains("https://rust-lang.org"));
        assert!(output.contains("A systems programming language."));
    }

    #[test]
    fn format_results_truncates_long_text() {
        let long_text = "x".repeat(600);
        let results = vec![SearchResult {
            title: "Long".into(),
            url: "https://example.com".into(),
            text: long_text,
        }];
        let output = format_results("q", &results);
        assert!(output.contains("..."), "long text should be truncated");
        let snippet_line = output.lines().find(|l| l.contains("xxx")).unwrap();
        assert!(
            snippet_line.len() < 520,
            "snippet too long: {}",
            snippet_line.len()
        );
    }

    #[test]
    fn format_results_missing_fields() {
        let results = vec![SearchResult {
            title: String::new(),
            url: String::new(),
            text: String::new(),
        }];
        let output = format_results("q", &results);
        assert!(
            output.contains("Untitled"),
            "missing title should default to Untitled"
        );
    }

    #[test]
    fn format_results_multiple_results() {
        let results = vec![
            SearchResult {
                title: "A".into(),
                url: "https://a.com".into(),
                text: "First".into(),
            },
            SearchResult {
                title: "B".into(),
                url: "https://b.com".into(),
                text: "Second".into(),
            },
        ];
        let output = format_results("q", &results);
        assert!(output.contains("1. **A**"));
        assert!(output.contains("2. **B**"));
    }

    #[test]
    fn format_results_with_search_result_struct() {
        let results = vec![
            SearchResult {
                title: "Test Title".into(),
                url: "https://example.com".into(),
                text: "Some description text".into(),
            },
            SearchResult {
                title: "Another".into(),
                url: "https://other.com".into(),
                text: "More text here".into(),
            },
        ];
        let output = format_results("my query", &results);
        assert!(output.contains("Search results for \"my query\""));
        assert!(output.contains("1. **Test Title**"));
        assert!(output.contains("https://example.com"));
        assert!(output.contains("Some description text"));
        assert!(output.contains("2. **Another**"));
        assert!(output.contains("https://other.com"));
    }

    #[test]
    fn parse_tavily_response() {
        let data = json!({
            "results": [
                {
                    "title": "Rust Lang",
                    "url": "https://rust-lang.org",
                    "content": "A systems programming language focused on safety."
                },
                {
                    "title": "Crates.io",
                    "url": "https://crates.io",
                    "content": "The Rust community's crate registry."
                }
            ]
        });
        let results = parse_tavily_results(&data);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust Lang");
        assert_eq!(results[0].url, "https://rust-lang.org");
        assert_eq!(
            results[0].text,
            "A systems programming language focused on safety."
        );
        assert_eq!(results[1].title, "Crates.io");
    }

    #[test]
    fn parse_tavily_response_empty() {
        let data = json!({"results": []});
        let results = parse_tavily_results(&data);
        assert!(results.is_empty());
    }

    #[test]
    fn parse_brave_response() {
        let data = json!({
            "web": {
                "results": [
                    {
                        "title": "Brave Search",
                        "url": "https://brave.com",
                        "description": "A privacy-focused search engine."
                    },
                    {
                        "title": "Wikipedia",
                        "url": "https://en.wikipedia.org",
                        "description": "The free encyclopedia."
                    }
                ]
            }
        });
        let results = parse_brave_results(&data);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Brave Search");
        assert_eq!(results[0].url, "https://brave.com");
        assert_eq!(results[0].text, "A privacy-focused search engine.");
        assert_eq!(results[1].title, "Wikipedia");
    }

    #[test]
    fn parse_brave_response_empty() {
        let data = json!({"web": {"results": []}});
        let results = parse_brave_results(&data);
        assert!(results.is_empty());
    }

    #[test]
    fn parse_brave_response_missing_web() {
        let data = json!({});
        let results = parse_brave_results(&data);
        assert!(results.is_empty());
    }

    #[test]
    fn parse_duckduckgo_html_results() {
        let html = r#"
        <div class="result">
            <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frust-lang.org&rut=abc" class="result__a">
                <b>Rust</b> Programming Language
            </a>
            <a class="result__snippet">A language empowering everyone to build reliable software.</a>
        </div>
        <div class="result">
            <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fcrates.io" class="result__a">
                Crates.io
            </a>
            <a class="result__snippet">The Rust package registry.</a>
        </div>
        "#;
        let results = parse_duckduckgo_html(html, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust Programming Language");
        assert_eq!(results[0].url, "https://rust-lang.org");
        assert_eq!(
            results[0].text,
            "A language empowering everyone to build reliable software."
        );
        assert_eq!(results[1].title, "Crates.io");
        assert_eq!(results[1].url, "https://crates.io");
        assert_eq!(results[1].text, "The Rust package registry.");
    }

    #[test]
    fn parse_duckduckgo_html_respects_limit() {
        let html = r#"
        <a href="https://a.com" class="result__a">A</a>
        <a class="result__snippet">Desc A</a>
        <a href="https://b.com" class="result__a">B</a>
        <a class="result__snippet">Desc B</a>
        <a href="https://c.com" class="result__a">C</a>
        <a class="result__snippet">Desc C</a>
        "#;
        let results = parse_duckduckgo_html(html, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn parse_duckduckgo_html_empty() {
        let results = parse_duckduckgo_html("<html><body>No results</body></html>", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn strip_html_tags_basic() {
        assert_eq!(strip_html_tags("<b>bold</b> text"), "bold text");
        assert_eq!(strip_html_tags("no tags"), "no tags");
        assert_eq!(strip_html_tags("<a href=\"x\">link</a>"), "link");
    }

    #[test]
    fn strip_html_tags_decodes_entities() {
        assert_eq!(strip_html_tags("foo &amp; bar"), "foo & bar");
        assert_eq!(strip_html_tags("&lt;not a tag&gt;"), "<not a tag>");
        assert_eq!(strip_html_tags("it&#39;s"), "it's");
        assert_eq!(strip_html_tags("a&nbsp;b"), "a b");
    }

    #[test]
    fn url_decode_basic() {
        assert_eq!(
            url_decode("https%3A%2F%2Fexample.com"),
            "https://example.com"
        );
        assert_eq!(url_decode("hello%20world"), "hello world");
        assert_eq!(url_decode("noencode"), "noencode");
    }

    #[test]
    fn decode_ddg_url_with_uddg() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=abc";
        assert_eq!(decode_ddg_url(href), "https://example.com");
    }

    #[test]
    fn decode_ddg_url_direct() {
        assert_eq!(decode_ddg_url("https://example.com"), "https://example.com");
    }

    #[test]
    fn url_decode_multibyte() {
        assert_eq!(url_decode("%C3%A9"), "é");
        assert_eq!(url_decode("%E4%B8%AD%E6%96%87"), "中文");
    }

    #[tokio::test]
    async fn websearch_auto_cascade_no_keys() {
        // With no API keys set, Auto should still include DuckDuckGo.
        // We can't actually hit the network in unit tests, but we verify the
        // provider list logic.
        let providers = detect_providers();
        assert!(providers.contains(&SearchProvider::DuckDuckGo));
    }
}
