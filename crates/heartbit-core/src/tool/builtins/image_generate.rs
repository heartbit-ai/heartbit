use std::future::Future;
use std::pin::Pin;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const DEFAULT_MODEL: &str = "google/gemini-3.1-flash-image-preview";
const IMAGE_MARKER_PREFIX: &str = "[IMAGE:base64:";

/// Builtin tool that generates images from a text prompt via the OpenRouter image API.
///
/// Sends a request to the configured model (default: `google/gemini-3.1-flash-image-preview`)
/// and returns the result as a base64-encoded inline image block marked with the
/// `[IMAGE:base64:...]` prefix so downstream handlers can decode it. Requires an
/// `OPENROUTER_API_KEY` environment variable; construction fails gracefully via
/// `try_new` if the HTTP client cannot be initialised.
pub struct ImageGenerateTool {
    client: reqwest::Client,
}

impl ImageGenerateTool {
    /// Create an `ImageGenerateTool`.
    ///
    /// Panics if the HTTP client cannot be built. Use [`ImageGenerateTool::try_new`]
    /// if you need to handle the error.
    pub fn new() -> Self {
        Self::try_new().expect("failed to build reqwest client")
    }

    /// Create an `ImageGenerateTool`, returning `Err` on failure.
    ///
    /// Returns `Err` if the underlying HTTP client cannot be constructed
    /// (e.g., TLS initialisation failure).
    pub fn try_new() -> Result<Self, crate::error::Error> {
        let client = crate::http::vendor_client_builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| {
                crate::error::Error::Agent(format!("failed to build reqwest client: {e}"))
            })?;
        Ok(Self { client })
    }
}

impl Default for ImageGenerateTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ImageGenerateTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "image_generate".into(),
            description: "Generate an image from a text prompt using Gemini via OpenRouter. \
                          Requires OPENROUTER_API_KEY environment variable. \
                          Returns base64-encoded image data."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    },
                    "style": {
                        "type": "string",
                        "description": "Optional style modifier (e.g., 'photorealistic', 'illustration', 'minimal')"
                    },
                    "model": {
                        "type": "string",
                        "description": "OpenRouter model ID (default: google/gemini-3.1-flash-image-preview)"
                    }
                },
                "required": ["prompt"]
            }),
        }
    }

    fn redact_for_history(&self, output: &str) -> String {
        // Replace the [IMAGE:base64:<huge_data>] marker body with a tiny
        // placeholder. Keep the bracket structure so any naïve parser
        // doesn't break, and append a SHA-256 short hash so different
        // images produce different placeholder strings (debugging).
        //
        // Without this, the ~600 KB base64 payload re-enters the
        // conversation on the next LLM turn and trips Anthropic's
        // 200k-token context cap. The full marker is preserved on
        // `AgentOutput.tool_call_results` for the caller (e.g. a
        // pipeline that needs to attach the image to a tweet).
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
        // No marker found — return verbatim.
        output.to_string()
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let prompt = input
                .get("prompt")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("prompt is required".into()))?;

            let style = input.get("style").and_then(|v| v.as_str());
            let model = input
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or(DEFAULT_MODEL);

            let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
                Error::Agent(
                    "OPENROUTER_API_KEY environment variable not set. \
                     Image generation requires an OpenRouter API key."
                        .into(),
                )
            })?;

            // Build the generation prompt
            let full_prompt = match style {
                Some(s) => format!("Generate an image in {s} style: {prompt}"),
                None => format!("Generate an image: {prompt}"),
            };

            let body = json!({
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "modalities": ["image", "text"],
                "image_config": {
                    "aspect_ratio": "16:9"
                }
            });

            let response = self
                .client
                .post("https://openrouter.ai/api/v1/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| Error::Agent(format!("OpenRouter API request failed: {e}")))?;

            let status = response.status();
            if !status.is_success() {
                // SECURITY (F-NET-1): cap error body.
                let error_body = crate::http::read_text_capped(response, 4 * 1024)
                    .await
                    .unwrap_or_default();
                return Ok(ToolOutput::error(format!(
                    "OpenRouter API error (HTTP {}): {error_body}",
                    status.as_u16()
                )));
            }

            // SECURITY (F-NET-1): cap successful body before parsing JSON.
            // Image responses include base64 — generous 15 MiB cap.
            let (bytes, was_truncated) = crate::http::read_body_capped(response, 15 * 1024 * 1024)
                .await
                .map_err(|e| Error::Agent(format!("Failed to read OpenRouter response: {e}")))?;
            if was_truncated {
                return Ok(ToolOutput::error(
                    "OpenRouter image response exceeded 15 MiB cap",
                ));
            }
            let data: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Agent(format!("Failed to parse OpenRouter response: {e}")))?;

            // Extract image data from response
            // Gemini returns inline_data with base64 in the content parts
            extract_image_from_response(&data, prompt)
        })
    }
}

/// Extract base64 image data from an OpenRouter image generation response.
///
/// OpenRouter returns images in `choices[0].message.images[].image_url.url`
/// as data URLs (`data:image/png;base64,...`). Also handles fallback formats
/// like `content` array with `inline_data` or `image_url` parts.
fn extract_image_from_response(
    data: &serde_json::Value,
    prompt: &str,
) -> Result<ToolOutput, Error> {
    let choices = data.get("choices").and_then(|v| v.as_array());

    if let Some(choices) = choices {
        for choice in choices {
            let msg = match choice.get("message") {
                Some(m) => m,
                None => continue,
            };

            // Primary: OpenRouter documented format — message.images[].image_url.url
            if let Some(images) = msg.get("images").and_then(|i| i.as_array()) {
                for image in images {
                    if let Some(url) = image
                        .get("image_url")
                        .and_then(|iu| iu.get("url"))
                        .and_then(|u| u.as_str())
                        && let Some(b64_data) = url.strip_prefix("data:")
                    {
                        return Ok(ToolOutput::success(format!(
                            "{IMAGE_MARKER_PREFIX}{b64_data}]\n\n\
                             Generated image for: {prompt}"
                        )));
                    }
                }
            }

            // Fallback: content array with inline_data or image_url parts
            if let Some(content_parts) = msg.get("content").and_then(|c| c.as_array()) {
                for part in content_parts {
                    if let Some(inline) = part.get("inline_data")
                        && let Some(b64) = inline.get("data").and_then(|d| d.as_str())
                    {
                        let mime = inline
                            .get("mime_type")
                            .and_then(|m| m.as_str())
                            .unwrap_or("image/png");
                        return Ok(ToolOutput::success(format!(
                            "{IMAGE_MARKER_PREFIX}{mime};{b64}]\n\n\
                             Generated image for: {prompt}"
                        )));
                    }
                    if let Some(image_url) = part.get("image_url")
                        && let Some(url) = image_url.get("url").and_then(|u| u.as_str())
                        && let Some(b64_data) = url.strip_prefix("data:")
                    {
                        return Ok(ToolOutput::success(format!(
                            "{IMAGE_MARKER_PREFIX}{b64_data}]\n\n\
                             Generated image for: {prompt}"
                        )));
                    }
                }
            }

            // Fallback: content as string — model returned text instead of image
            if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                return Ok(ToolOutput::error(format!(
                    "Model returned text instead of image: {text}"
                )));
            }
        }
    }

    Ok(ToolOutput::error(format!(
        "No image data found in response. Raw response: {}",
        serde_json::to_string(data).unwrap_or_default()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = ImageGenerateTool::new();
        let def = tool.definition();
        assert_eq!(def.name, "image_generate");
    }

    #[test]
    fn definition_has_required_prompt() {
        let tool = ImageGenerateTool::new();
        let def = tool.definition();
        let required = def.input_schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "prompt");
    }

    #[test]
    fn definition_has_optional_style_and_model() {
        let tool = ImageGenerateTool::new();
        let def = tool.definition();
        let props = def.input_schema["properties"].as_object().unwrap();
        assert!(props.contains_key("style"));
        assert!(props.contains_key("model"));
        assert!(props.contains_key("prompt"));
    }

    #[tokio::test]
    async fn image_generate_requires_api_key() {
        if std::env::var("OPENROUTER_API_KEY").is_ok() {
            return;
        }

        let tool = ImageGenerateTool::new();
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"prompt": "a cat"}),
            )
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("OPENROUTER_API_KEY"), "got: {err}");
    }

    #[tokio::test]
    async fn image_generate_requires_prompt() {
        if std::env::var("OPENROUTER_API_KEY").is_ok() {
            return;
        }

        let tool = ImageGenerateTool::new();
        let result = tool
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("prompt is required"), "got: {err}");
    }

    #[test]
    fn extract_openrouter_images_array() {
        // Primary documented format: message.images[].image_url.url
        let response = json!({
            "choices": [{
                "message": {
                    "images": [{
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
                        }
                    }]
                }
            }]
        });

        let result = extract_image_from_response(&response, "mountains").unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains(IMAGE_MARKER_PREFIX));
        assert!(
            result
                .content
                .contains("image/png;base64,iVBORw0KGgoAAAANSUhEUg==")
        );
        assert!(result.content.contains("Generated image for: mountains"));
    }

    #[test]
    fn extract_gemini_inline_data() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": [{
                        "type": "image",
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUg=="
                        }
                    }]
                }
            }]
        });

        let result = extract_image_from_response(&response, "test prompt").unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains(IMAGE_MARKER_PREFIX),
            "should contain image marker"
        );
        assert!(
            result.content.contains("image/png"),
            "should contain mime type"
        );
        assert!(
            result.content.contains("iVBORw0KGgoAAAANSUhEUg=="),
            "should contain base64 data"
        );
        assert!(
            result.content.contains("Generated image for: test prompt"),
            "should contain prompt reference"
        );
    }

    #[test]
    fn extract_openrouter_image_url() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
                        }
                    }]
                }
            }]
        });

        let result = extract_image_from_response(&response, "sunset").unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains(IMAGE_MARKER_PREFIX));
        assert!(result.content.contains("image/jpeg"));
        assert!(result.content.contains("/9j/4AAQSkZJRg=="));
    }

    #[test]
    fn extract_no_image_returns_error() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "I cannot generate images."
                }
            }]
        });

        let result = extract_image_from_response(&response, "cat").unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("text instead of image"));
    }

    #[test]
    fn extract_empty_response_returns_error() {
        let response = json!({});
        let result = extract_image_from_response(&response, "cat").unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("No image data found"));
    }

    #[test]
    fn redact_for_history_replaces_marker_body_with_short_hash() {
        let tool = ImageGenerateTool::new();
        let raw =
            "[IMAGE:base64:image/png;base64,iVBORw0KGgo=AAAAAAAAA]\n\nGenerated image for: test";
        let redacted = tool.redact_for_history(raw);
        assert!(redacted.starts_with("[IMAGE:redacted:"));
        assert!(redacted.contains("]\n\nGenerated image for: test"));
        assert!(!redacted.contains("iVBORw0KGgo"));
        // Same input → same hash → idempotent.
        assert_eq!(tool.redact_for_history(raw), redacted);
    }

    #[test]
    fn redact_for_history_passes_through_when_no_marker() {
        let tool = ImageGenerateTool::new();
        assert_eq!(
            tool.redact_for_history("plain text without marker"),
            "plain text without marker"
        );
    }

    #[test]
    fn image_marker_format() {
        // Verify downstream agents can parse the marker
        let output = format!("{IMAGE_MARKER_PREFIX}image/png;abc123]");
        assert!(output.starts_with("[IMAGE:base64:"));
        assert!(output.ends_with(']'));
        // Extract mime and data
        let inner = output
            .strip_prefix("[IMAGE:base64:")
            .unwrap()
            .strip_suffix(']')
            .unwrap();
        let (mime, data) = inner.split_once(';').unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(data, "abc123");
    }
}
