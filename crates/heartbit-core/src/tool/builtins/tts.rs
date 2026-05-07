use std::future::Future;
use std::pin::Pin;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

/// Valid voices for OpenAI TTS.
const OPENAI_VOICES: &[&str] = &["alloy", "echo", "fable", "onyx", "nova", "shimmer"];

/// Valid output formats for OpenAI TTS.
const OPENAI_FORMATS: &[&str] = &["mp3", "opus", "aac", "flac", "wav", "pcm"];

/// Maximum text length to send (characters).
const MAX_TEXT_LENGTH: usize = 4096;

/// Text-to-Speech tool using the OpenAI TTS API.
///
/// Converts text to speech audio. Requires `OPENAI_API_KEY` environment variable.
/// Returns base64-encoded audio data.
pub struct TtsTool {
    client: reqwest::Client,
}

impl TtsTool {
    /// Create a `TtsTool`.
    ///
    /// Panics if the HTTP client cannot be built. Use [`TtsTool::try_new`]
    /// if you need to handle the error.
    pub fn new() -> Self {
        Self::try_new().expect("failed to build reqwest client")
    }

    /// Create a `TtsTool`, returning `Err` on failure.
    ///
    /// Returns `Err` if the underlying HTTP client cannot be constructed
    /// (e.g., TLS initialisation failure).
    pub fn try_new() -> Result<Self, crate::error::Error> {
        let client = crate::http::vendor_client_builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| {
                crate::error::Error::Agent(format!("failed to build reqwest client: {e}"))
            })?;
        Ok(Self { client })
    }
}

impl Tool for TtsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "tts".into(),
            description: "Convert text to speech audio using OpenAI TTS API. \
                          Requires OPENAI_API_KEY environment variable. \
                          Returns base64-encoded audio data."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to convert to speech (max 4096 characters)"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice to use: alloy, echo, fable, onyx, nova, shimmer (default: alloy)",
                        "enum": OPENAI_VOICES
                    },
                    "model": {
                        "type": "string",
                        "description": "TTS model: tts-1 or tts-1-hd (default: tts-1)",
                        "enum": ["tts-1", "tts-1-hd"]
                    },
                    "format": {
                        "type": "string",
                        "description": "Output audio format (default: mp3)",
                        "enum": OPENAI_FORMATS
                    },
                    "speed": {
                        "type": "number",
                        "description": "Speed multiplier 0.25 to 4.0 (default: 1.0)"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
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
            if char_count > MAX_TEXT_LENGTH {
                return Ok(ToolOutput::error(format!(
                    "text exceeds maximum length of {MAX_TEXT_LENGTH} characters (got {char_count})",
                )));
            }

            let voice = input
                .get("voice")
                .and_then(|v| v.as_str())
                .unwrap_or("alloy");

            if !OPENAI_VOICES.contains(&voice) {
                return Ok(ToolOutput::error(format!(
                    "invalid voice '{voice}'. Valid voices: {}",
                    OPENAI_VOICES.join(", ")
                )));
            }

            let model = input
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("tts-1");

            if model != "tts-1" && model != "tts-1-hd" {
                return Ok(ToolOutput::error(format!(
                    "invalid model '{model}'. Use 'tts-1' or 'tts-1-hd'"
                )));
            }

            let format = input
                .get("format")
                .and_then(|v| v.as_str())
                .unwrap_or("mp3");

            if !OPENAI_FORMATS.contains(&format) {
                return Ok(ToolOutput::error(format!(
                    "invalid format '{format}'. Valid formats: {}",
                    OPENAI_FORMATS.join(", ")
                )));
            }

            let speed = input.get("speed").and_then(|v| v.as_f64()).unwrap_or(1.0);

            if !(0.25..=4.0).contains(&speed) {
                return Ok(ToolOutput::error(
                    "speed must be between 0.25 and 4.0".to_string(),
                ));
            }

            let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                Error::Agent(
                    "OPENAI_API_KEY environment variable not set. TTS requires an OpenAI API key."
                        .into(),
                )
            })?;

            let body = json!({
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": format,
                "speed": speed,
            });

            let response = self
                .client
                .post("https://api.openai.com/v1/audio/speech")
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| Error::Agent(format!("TTS API request failed: {e}")))?;

            let status = response.status();
            if !status.is_success() {
                // SECURITY (F-NET-1): cap error body to 4 KiB.
                let error_body = crate::http::read_text_capped(response, 4 * 1024)
                    .await
                    .unwrap_or_default();
                let truncated = if error_body.len() > 500 {
                    let end = super::floor_char_boundary(&error_body, 500);
                    format!("{}...", &error_body[..end])
                } else {
                    error_body
                };
                return Ok(ToolOutput::error(format!(
                    "TTS API error (HTTP {}): {truncated}",
                    status.as_u16()
                )));
            }

            // SECURITY (F-NET-1): TTS audio can legitimately be up to a few MB
            // (especially for long-form text). Use a generous cap (10 MiB) to
            // accommodate that while protecting against a hostile vendor.
            let (audio_bytes, was_truncated) =
                crate::http::read_body_capped(response, 10 * 1024 * 1024)
                    .await
                    .map_err(|e| Error::Agent(format!("Failed to read TTS response: {e}")))?;
            if was_truncated {
                return Ok(ToolOutput::error(
                    "TTS response exceeded 10 MiB cap; refusing to truncate audio",
                ));
            }

            use base64::Engine;
            let encoded = base64::engine::general_purpose::STANDARD.encode(&audio_bytes);

            // Estimate duration: ~150 words per minute, ~5 chars per word
            let word_count = text.split_whitespace().count();
            let duration_estimate_secs = (word_count as f64 / 150.0 * 60.0 / speed).max(1.0);

            Ok(ToolOutput::success(format!(
                "Audio generated successfully.\n\
                 Format: {format}\n\
                 Voice: {voice}\n\
                 Duration estimate: {duration_estimate_secs:.1}s\n\
                 Size: {} bytes\n\
                 Base64 audio data:\n{encoded}",
                audio_bytes.len()
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = TtsTool::new();
        assert_eq!(tool.definition().name, "tts");
    }

    #[test]
    fn definition_requires_text() {
        let tool = TtsTool::new();
        let schema = &tool.definition().input_schema;
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("text")));
    }

    #[test]
    fn definition_lists_voices() {
        let tool = TtsTool::new();
        let schema = &tool.definition().input_schema;
        let voices = schema["properties"]["voice"]["enum"].as_array().unwrap();
        assert_eq!(voices.len(), 6);
        assert!(voices.contains(&json!("alloy")));
        assert!(voices.contains(&json!("shimmer")));
    }

    #[tokio::test]
    async fn rejects_empty_text() {
        let tool = TtsTool::new();
        let result = tool
            .execute(&crate::ExecutionContext::default(), json!({"text": ""}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("must not be empty"));
    }

    #[tokio::test]
    async fn rejects_text_too_long() {
        let tool = TtsTool::new();
        let long = "a".repeat(MAX_TEXT_LENGTH + 1);
        let result = tool
            .execute(&crate::ExecutionContext::default(), json!({"text": long}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("exceeds maximum length"));
    }

    #[tokio::test]
    async fn rejects_invalid_voice() {
        let tool = TtsTool::new();
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"text": "hello", "voice": "invalid"}),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("invalid voice"));
    }

    #[tokio::test]
    async fn rejects_invalid_model() {
        let tool = TtsTool::new();
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"text": "hello", "model": "gpt-4"}),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("invalid model"));
    }

    #[tokio::test]
    async fn rejects_invalid_format() {
        let tool = TtsTool::new();
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"text": "hello", "format": "wma"}),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("invalid format"));
    }

    #[tokio::test]
    async fn rejects_speed_too_low() {
        let tool = TtsTool::new();
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"text": "hello", "speed": 0.1}),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("speed must be"));
    }

    #[tokio::test]
    async fn rejects_speed_too_high() {
        let tool = TtsTool::new();
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"text": "hello", "speed": 5.0}),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("speed must be"));
    }

    #[tokio::test]
    async fn requires_api_key() {
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return; // Can't safely test missing key when it's set
        }
        let tool = TtsTool::new();
        let result = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"text": "hello"}),
            )
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("OPENAI_API_KEY"), "got: {err}");
    }

    #[tokio::test]
    async fn rejects_missing_text() {
        let tool = TtsTool::new();
        let result = tool
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await;
        assert!(result.is_err());
    }
}
