use std::future::Future;
use std::pin::Pin;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const DEFAULT_NUM_RESULTS: u64 = 8;
const MAX_NUM_RESULTS: u64 = 50;

pub struct WebSearchTool {
    client: reqwest::Client,
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build reqwest client"),
        }
    }
}

impl Tool for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "websearch".into(),
            description: "Search the web using Exa AI. Requires EXA_API_KEY environment variable. \
                          Returns titles, URLs, and text snippets."
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

            let api_key = std::env::var("EXA_API_KEY").map_err(|_| {
                Error::Agent(
                    "EXA_API_KEY environment variable not set. Web search requires an Exa AI API key."
                        .into(),
                )
            })?;

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
                let error_body = response.text().await.unwrap_or_default();
                return Ok(ToolOutput::error(format!(
                    "Exa API error (HTTP {}): {error_body}",
                    status.as_u16()
                )));
            }

            let data: serde_json::Value = response
                .json()
                .await
                .map_err(|e| Error::Agent(format!("Failed to parse Exa response: {e}")))?;

            let results = data
                .get("results")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            Ok(ToolOutput::success(format_results(query, &results)))
        })
    }
}

fn format_results(query: &str, results: &[serde_json::Value]) -> String {
    if results.is_empty() {
        return "No search results found.".into();
    }

    let mut output = format!("Search results for \"{query}\":\n\n");

    for (i, result) in results.iter().enumerate() {
        let title = result
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Untitled");
        let url = result.get("url").and_then(|v| v.as_str()).unwrap_or("");
        let text = result.get("text").and_then(|v| v.as_str()).unwrap_or("");

        // Truncate text snippet (char-boundary safe)
        let snippet = if text.len() > 500 {
            let end = super::floor_char_boundary(text, 500);
            format!("{}...", &text[..end])
        } else {
            text.to_string()
        };

        output.push_str(&format!(
            "{}. **{}**\n   {}\n   {}\n\n",
            i + 1,
            title,
            url,
            snippet.trim()
        ));
    }

    output
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
    fn format_results_empty() {
        let output = format_results("test", &[]);
        assert_eq!(output, "No search results found.");
    }

    #[test]
    fn format_results_single_result() {
        let results = vec![json!({
            "title": "Rust Programming",
            "url": "https://rust-lang.org",
            "text": "A systems programming language."
        })];
        let output = format_results("rust", &results);
        assert!(output.contains("Search results for \"rust\""));
        assert!(output.contains("1. **Rust Programming**"));
        assert!(output.contains("https://rust-lang.org"));
        assert!(output.contains("A systems programming language."));
    }

    #[test]
    fn format_results_truncates_long_text() {
        let long_text = "x".repeat(600);
        let results = vec![json!({
            "title": "Long",
            "url": "https://example.com",
            "text": long_text
        })];
        let output = format_results("q", &results);
        assert!(output.contains("..."), "long text should be truncated");
        // The snippet should be <=503 chars (500 + "...")
        let snippet_line = output.lines().find(|l| l.contains("xxx")).unwrap();
        assert!(
            snippet_line.len() < 520,
            "snippet too long: {}",
            snippet_line.len()
        );
    }

    #[test]
    fn format_results_missing_fields() {
        let results = vec![json!({})];
        let output = format_results("q", &results);
        assert!(
            output.contains("Untitled"),
            "missing title should default to Untitled"
        );
    }

    #[test]
    fn format_results_multiple_results() {
        let results = vec![
            json!({"title": "A", "url": "https://a.com", "text": "First"}),
            json!({"title": "B", "url": "https://b.com", "text": "Second"}),
        ];
        let output = format_results("q", &results);
        assert!(output.contains("1. **A**"));
        assert!(output.contains("2. **B**"));
    }

    #[tokio::test]
    async fn websearch_requires_api_key() {
        if std::env::var("EXA_API_KEY").is_ok() {
            // Can't safely test missing-key path when the key is set.
            // env::remove_var is unsound under parallel test execution.
            return;
        }

        let tool = WebSearchTool::new();
        let result = tool.execute(json!({"query": "test"})).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("EXA_API_KEY"), "got: {err}");
    }
}
