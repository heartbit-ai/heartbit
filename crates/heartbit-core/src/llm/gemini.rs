//! Google Gemini LLM provider via the Gemini REST API.

use bytes::Bytes;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::anthropic::SseParser;
use crate::llm::types::{
    CompletionRequest, CompletionResponse, ContentBlock, Role, StopReason, TokenUsage, ToolChoice,
};

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Google Gemini LLM provider (native API).
///
/// Uses the Gemini generateContent / streamGenerateContent endpoints.
/// Supports function calling and streaming.
pub struct GeminiProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl GeminiProvider {
    /// Create a new Gemini provider with the given API key and model identifier.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: BASE_URL.into(),
        }
    }

    /// Create with a custom base URL (for testing or Vertex AI).
    pub fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
        }
    }

    fn generate_url(&self) -> String {
        format!(
            "{}/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            self.model
        )
    }

    fn stream_url(&self) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url.trim_end_matches('/'),
            self.model
        )
    }
}

// ---------------------------------------------------------------------------
// Request types (private, for serialization)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiToolConfig>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<GeminiToolBehavior>,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    #[serde(default)]
    args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiInlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiToolConfig {
    function_declarations: Vec<GeminiFunctionDecl>,
}

#[derive(Serialize)]
struct GeminiFunctionDecl {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiToolBehavior {
    function_calling_config: FunctionCallingConfig,
}

#[derive(Serialize)]
struct FunctionCallingConfig {
    mode: String,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "allowedFunctionNames"
    )]
    allowed_function_names: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Response types (private, for deserialization)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    cached_content_token_count: u32,
}

// ---------------------------------------------------------------------------
// Build request
// ---------------------------------------------------------------------------

/// Build a mapping from tool_use_id → function name from the message history.
///
/// Gemini's `functionResponse` requires the function name, but our internal
/// `ToolResult` only carries `tool_use_id`. We scan prior messages for
/// `ToolUse` blocks to resolve the mapping.
fn build_tool_id_to_name(request: &CompletionRequest) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for msg in &request.messages {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, name, .. } = block {
                map.insert(id.clone(), name.clone());
            }
        }
    }
    map
}

/// Normalize a JSON Schema for Gemini compatibility.
///
/// Gemini rejects schemas containing `$schema`, `additionalProperties`,
/// `anyOf`/`oneOf`, `$ref`, `$defs`, `default`, `title`, `const`, `format`,
/// and type arrays. This function strips or transforms these fields.
fn normalize_gemini_schema(schema: &serde_json::Value) -> serde_json::Value {
    match schema {
        serde_json::Value::Object(map) => {
            let mut result = serde_json::Map::new();
            for (key, value) in map {
                match key.as_str() {
                    // Skip unsupported fields
                    "$schema"
                    | "$ref"
                    | "$defs"
                    | "additionalProperties"
                    | "default"
                    | "title"
                    | "const"
                    | "format" => continue,
                    // Handle anyOf/oneOf with null — flatten to single type + nullable
                    "anyOf" | "oneOf" => {
                        if let Some(variants) = value.as_array() {
                            let non_null: Vec<&serde_json::Value> = variants
                                .iter()
                                .filter(|v| v.get("type").and_then(|t| t.as_str()) != Some("null"))
                                .collect();
                            if non_null.len() == 1 {
                                // Single non-null type — flatten and mark nullable
                                if let serde_json::Value::Object(inner) = non_null[0] {
                                    for (k, v) in inner {
                                        result.insert(k.clone(), normalize_gemini_schema(v));
                                    }
                                }
                                result.insert("nullable".into(), serde_json::Value::Bool(true));
                                continue;
                            }
                        }
                        // Can't simplify — skip entirely
                        continue;
                    }
                    // Handle type arrays: ["string", "null"] → "string" + nullable
                    "type" => {
                        if let Some(arr) = value.as_array() {
                            let types: Vec<&str> = arr
                                .iter()
                                .filter_map(|v| v.as_str())
                                .filter(|t| *t != "null")
                                .collect();
                            let has_null = arr.iter().any(|v| v.as_str() == Some("null"));
                            if types.len() == 1 {
                                result.insert("type".into(), serde_json::json!(types[0]));
                                if has_null {
                                    result.insert("nullable".into(), serde_json::Value::Bool(true));
                                }
                            } else {
                                result.insert(key.clone(), value.clone());
                            }
                            continue;
                        }
                        result.insert(key.clone(), value.clone());
                        continue;
                    }
                    _ => {
                        result.insert(key.clone(), normalize_gemini_schema(value));
                    }
                }
            }
            serde_json::Value::Object(result)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(normalize_gemini_schema).collect())
        }
        other => other.clone(),
    }
}

fn build_gemini_request(request: &CompletionRequest) -> GeminiRequest {
    let tool_id_to_name = build_tool_id_to_name(request);
    let mut contents = Vec::new();

    for msg in &request.messages {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "model",
        };

        let mut parts = Vec::new();
        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => {
                    parts.push(GeminiPart::Text { text: text.clone() });
                }
                ContentBlock::Image { media_type, data } => {
                    parts.push(GeminiPart::InlineData {
                        inline_data: GeminiInlineData {
                            mime_type: media_type.clone(),
                            data: data.clone(),
                        },
                    });
                }
                ContentBlock::ToolUse {
                    id: _, name, input, ..
                } => {
                    parts.push(GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            name: name.clone(),
                            args: input.clone(),
                        },
                    });
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let response = if *is_error {
                        serde_json::json!({"error": content})
                    } else {
                        serde_json::json!({"result": content})
                    };
                    let name = tool_id_to_name
                        .get(tool_use_id)
                        .cloned()
                        .unwrap_or_else(|| "tool_response".into());
                    parts.push(GeminiPart::FunctionResponse {
                        function_response: GeminiFunctionResponse { name, response },
                    });
                }
                ContentBlock::Audio { .. } => {} // Not yet supported
            }
        }

        if !parts.is_empty() {
            contents.push(GeminiContent {
                role: Some(role.into()),
                parts,
            });
        }
    }

    // System instruction
    let system_instruction = if request.system.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: None, // Gemini system_instruction must omit the role field
            parts: vec![GeminiPart::Text {
                text: request.system.clone(),
            }],
        })
    };

    // Tools
    let tools = if request.tools.is_empty() {
        None
    } else {
        let decls: Vec<GeminiFunctionDecl> = request
            .tools
            .iter()
            .map(|t| GeminiFunctionDecl {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: normalize_gemini_schema(&t.input_schema),
            })
            .collect();
        Some(vec![GeminiToolConfig {
            function_declarations: decls,
        }])
    };

    // Tool choice
    let tool_config = request.tool_choice.as_ref().map(|tc| {
        let (mode, names) = match tc {
            ToolChoice::Auto => ("AUTO".into(), None),
            ToolChoice::Any => ("ANY".into(), None),
            ToolChoice::Tool { name } => ("ANY".into(), Some(vec![name.clone()])),
        };
        GeminiToolBehavior {
            function_calling_config: FunctionCallingConfig {
                mode,
                allowed_function_names: names,
            },
        }
    });

    GeminiRequest {
        contents,
        system_instruction,
        tools,
        generation_config: Some(GenerationConfig {
            max_output_tokens: Some(request.max_tokens),
        }),
        tool_config,
    }
}

// ---------------------------------------------------------------------------
// Parse response
// ---------------------------------------------------------------------------

fn parse_gemini_response(resp: GeminiResponse) -> Result<CompletionResponse, Error> {
    let candidate = resp
        .candidates
        .into_iter()
        .next()
        .ok_or_else(|| Error::Api {
            status: 502,
            message: "empty candidates in Gemini response".into(),
        })?;

    let mut content = Vec::new();
    if let Some(gemini_content) = candidate.content {
        for part in gemini_content.parts {
            match part {
                GeminiPart::Text { text } if !text.is_empty() => {
                    content.push(ContentBlock::Text { text });
                }
                GeminiPart::FunctionCall { function_call } => {
                    content.push(ContentBlock::ToolUse {
                        id: uuid::Uuid::new_v4().to_string(),
                        name: function_call.name,
                        input: function_call.args,
                    });
                }
                _ => {}
            }
        }
    }

    let has_tool_calls = content
        .iter()
        .any(|c| matches!(c, ContentBlock::ToolUse { .. }));

    let stop_reason = match candidate.finish_reason.as_deref() {
        Some("STOP") => {
            if has_tool_calls {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }
        }
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        Some("SAFETY") => StopReason::EndTurn,
        Some("RECITATION") => StopReason::EndTurn,
        Some(other) => {
            warn!(
                finish_reason = other,
                "unknown Gemini finish_reason, treating as EndTurn"
            );
            StopReason::EndTurn
        }
        None => {
            if has_tool_calls {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }
        }
    };

    let usage = resp
        .usage_metadata
        .map_or(TokenUsage::default(), |u| TokenUsage {
            input_tokens: u.prompt_token_count,
            output_tokens: u.candidates_token_count,
            cache_read_input_tokens: u.cached_content_token_count,
            ..Default::default()
        });

    Ok(CompletionResponse {
        content,
        stop_reason,
        usage,
        model: None,
    })
}

// ---------------------------------------------------------------------------
// LlmProvider implementation
// ---------------------------------------------------------------------------

impl LlmProvider for GeminiProvider {
    fn model_name(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let body = build_gemini_request(&request);

        let response = self
            .client
            .post(self.generate_url())
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error_from_response(response).await);
        }

        let api_response: GeminiResponse = response.json().await?;
        parse_gemini_response(api_response)
    }

    async fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &crate::llm::OnText,
    ) -> Result<CompletionResponse, Error> {
        let body = build_gemini_request(&request);

        let response = self
            .client
            .post(self.stream_url())
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error_from_response(response).await);
        }

        parse_gemini_stream(response.bytes_stream(), on_text).await
    }
}

// ---------------------------------------------------------------------------
// Streaming parser
// ---------------------------------------------------------------------------

async fn parse_gemini_stream<S>(
    stream: S,
    on_text: &crate::llm::OnText,
) -> Result<CompletionResponse, Error>
where
    S: futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    let mut parser = SseParser::new();
    let mut utf8_buf: Vec<u8> = Vec::new();
    let mut all_text = String::new();
    let mut tool_calls: Vec<(String, serde_json::Value)> = Vec::new();
    let mut finish_reason: Option<String> = None;
    let mut usage = TokenUsage::default();

    tokio::pin!(stream);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(Error::Http)?;
        utf8_buf.extend_from_slice(&chunk);

        let valid_len = match std::str::from_utf8(&utf8_buf) {
            Ok(_) => utf8_buf.len(),
            Err(e) => e.valid_up_to(),
        };

        if valid_len > 0 {
            let s = std::str::from_utf8(&utf8_buf[..valid_len])
                .expect("valid_up_to guarantees valid UTF-8");
            for event in parser.feed(s) {
                process_gemini_event(
                    &event.data,
                    on_text,
                    &mut all_text,
                    &mut tool_calls,
                    &mut finish_reason,
                    &mut usage,
                );
            }
        }
        utf8_buf.drain(..valid_len);
    }

    // Flush remaining
    if !utf8_buf.is_empty()
        && let Ok(s) = std::str::from_utf8(&utf8_buf)
    {
        for event in parser.feed(s) {
            process_gemini_event(
                &event.data,
                on_text,
                &mut all_text,
                &mut tool_calls,
                &mut finish_reason,
                &mut usage,
            );
        }
    }
    for event in parser.flush() {
        process_gemini_event(
            &event.data,
            on_text,
            &mut all_text,
            &mut tool_calls,
            &mut finish_reason,
            &mut usage,
        );
    }

    let mut content = Vec::new();
    if !all_text.is_empty() {
        content.push(ContentBlock::Text { text: all_text });
    }
    for (name, args) in tool_calls {
        content.push(ContentBlock::ToolUse {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            input: args,
        });
    }

    let has_tool_calls = content
        .iter()
        .any(|c| matches!(c, ContentBlock::ToolUse { .. }));
    let stop_reason = match finish_reason.as_deref() {
        Some("STOP") if has_tool_calls => StopReason::ToolUse,
        Some("STOP") => StopReason::EndTurn,
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        _ => {
            if has_tool_calls {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }
        }
    };

    Ok(CompletionResponse {
        content,
        stop_reason,
        usage,
        model: None,
    })
}

fn process_gemini_event(
    data: &str,
    on_text: &crate::llm::OnText,
    all_text: &mut String,
    tool_calls: &mut Vec<(String, serde_json::Value)>,
    finish_reason: &mut Option<String>,
    usage: &mut TokenUsage,
) {
    let chunk: GeminiResponse = match serde_json::from_str(data) {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "failed to parse Gemini streaming chunk, skipping");
            return;
        }
    };

    if let Some(candidate) = chunk.candidates.first() {
        if let Some(ref content) = candidate.content {
            for part in &content.parts {
                match part {
                    GeminiPart::Text { text } if !text.is_empty() => {
                        all_text.push_str(text);
                        on_text(text);
                    }
                    GeminiPart::FunctionCall { function_call } => {
                        tool_calls.push((function_call.name.clone(), function_call.args.clone()));
                    }
                    _ => {}
                }
            }
        }
        if candidate.finish_reason.is_some() {
            *finish_reason = candidate.finish_reason.clone();
        }
    }

    if let Some(ref u) = chunk.usage_metadata {
        *usage = TokenUsage {
            input_tokens: u.prompt_token_count,
            output_tokens: u.candidates_token_count,
            cache_read_input_tokens: u.cached_content_token_count,
            ..Default::default()
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{Message, ToolDefinition};
    use serde_json::json;

    #[test]
    fn generate_url_correct() {
        let p = GeminiProvider::new("key", "gemini-2.5-flash");
        assert_eq!(
            p.generate_url(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn stream_url_correct() {
        let p = GeminiProvider::new("key", "gemini-2.5-flash");
        assert!(p.stream_url().contains("streamGenerateContent?alt=sse"));
    }

    #[test]
    fn custom_base_url() {
        let p = GeminiProvider::with_base_url("key", "model", "https://custom.api/v1");
        assert_eq!(
            p.generate_url(),
            "https://custom.api/v1/models/model:generateContent"
        );
    }

    #[test]
    fn build_request_with_system() {
        let req = CompletionRequest {
            system: "You are helpful".into(),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        assert!(gemini_req.system_instruction.is_some());
        let si = gemini_req.system_instruction.unwrap();
        assert_eq!(si.parts.len(), 1);
    }

    #[test]
    fn build_request_without_system() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        assert!(gemini_req.system_instruction.is_none());
    }

    #[test]
    fn build_request_with_tools() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("Search for rust")],
            tools: vec![ToolDefinition {
                name: "websearch".into(),
                description: "Search the web".into(),
                input_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            }],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        assert!(gemini_req.tools.is_some());
        let tools = gemini_req.tools.unwrap();
        assert_eq!(tools[0].function_declarations.len(), 1);
        assert_eq!(tools[0].function_declarations[0].name, "websearch");
    }

    #[test]
    fn build_request_tool_choice_auto() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("test")],
            tools: vec![ToolDefinition {
                name: "tool1".into(),
                description: "d".into(),
                input_schema: json!({}),
            }],
            max_tokens: 100,
            tool_choice: Some(ToolChoice::Auto),
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        let tc = gemini_req.tool_config.unwrap();
        assert_eq!(tc.function_calling_config.mode, "AUTO");
    }

    #[test]
    fn build_request_tool_choice_specific() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("test")],
            tools: vec![ToolDefinition {
                name: "mytool".into(),
                description: "d".into(),
                input_schema: json!({}),
            }],
            max_tokens: 100,
            tool_choice: Some(ToolChoice::Tool {
                name: "mytool".into(),
            }),
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        let tc = gemini_req.tool_config.unwrap();
        assert_eq!(
            tc.function_calling_config
                .allowed_function_names
                .as_ref()
                .unwrap(),
            &["mytool"]
        );
    }

    #[test]
    fn parse_text_response() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::Text {
                        text: "Hello!".into(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 5,
                cached_content_token_count: 0,
            }),
        };
        let result = parse_gemini_response(resp).unwrap();
        assert_eq!(result.text(), "Hello!");
        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }

    #[test]
    fn parse_tool_call_response() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            name: "websearch".into(),
                            args: json!({"q": "rust"}),
                        },
                    }],
                }),
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: None,
        };
        let result = parse_gemini_response(resp).unwrap();
        assert_eq!(result.stop_reason, StopReason::ToolUse);
        assert!(
            matches!(&result.content[0], ContentBlock::ToolUse { name, .. } if name == "websearch")
        );
    }

    #[test]
    fn parse_empty_candidates_returns_error() {
        let resp = GeminiResponse {
            candidates: vec![],
            usage_metadata: None,
        };
        assert!(parse_gemini_response(resp).is_err());
    }

    #[test]
    fn parse_max_tokens_stop_reason() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::Text {
                        text: "partial".into(),
                    }],
                }),
                finish_reason: Some("MAX_TOKENS".into()),
            }],
            usage_metadata: None,
        };
        let result = parse_gemini_response(resp).unwrap();
        assert_eq!(result.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn model_name_returns_model() {
        let p = GeminiProvider::new("key", "gemini-2.5-flash");
        assert_eq!(p.model_name(), Some("gemini-2.5-flash"));
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GeminiProvider>();
    }

    #[test]
    fn request_serializes_correctly() {
        let req = CompletionRequest {
            system: "Be helpful".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 50,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        let json = serde_json::to_value(&gemini_req).unwrap();
        assert!(json["systemInstruction"].is_object());
        assert!(json["contents"].is_array());
        assert_eq!(json["generationConfig"]["maxOutputTokens"], 50);
    }

    #[test]
    fn build_request_with_image() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "What's in this image?".into(),
                    },
                    ContentBlock::Image {
                        media_type: "image/png".into(),
                        data: "base64data".into(),
                    },
                ],
            }],
            tools: vec![],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        assert_eq!(gemini_req.contents[0].parts.len(), 2);
    }

    #[test]
    fn parse_mixed_text_and_tool_call() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".into()),
                    parts: vec![
                        GeminiPart::Text {
                            text: "Let me search.".into(),
                        },
                        GeminiPart::FunctionCall {
                            function_call: GeminiFunctionCall {
                                name: "search".into(),
                                args: json!({"q": "test"}),
                            },
                        },
                    ],
                }),
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: None,
        };
        let result = parse_gemini_response(resp).unwrap();
        assert_eq!(result.content.len(), 2);
        assert_eq!(result.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn build_request_tool_result_resolves_function_name() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "websearch".into(),
                        input: json!({"q": "rust"}),
                    }],
                },
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "call-1".into(),
                        content: "Rust is a programming language".into(),
                        is_error: false,
                    }],
                },
            ],
            tools: vec![],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        // The user message should contain a functionResponse with the resolved name
        let user_content = &gemini_req.contents[1];
        assert_eq!(user_content.role.as_deref(), Some("user"));
        match &user_content.parts[0] {
            GeminiPart::FunctionResponse {
                function_response, ..
            } => {
                assert_eq!(function_response.name, "websearch");
            }
            other => panic!("expected FunctionResponse, got: {other:?}"),
        }
    }

    #[test]
    fn build_request_tool_result_fallback_name() {
        // When tool_use_id has no matching ToolUse, falls back to "tool_response"
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "unknown-id".into(),
                    content: "result".into(),
                    is_error: false,
                }],
            }],
            tools: vec![],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        match &gemini_req.contents[0].parts[0] {
            GeminiPart::FunctionResponse {
                function_response, ..
            } => {
                assert_eq!(function_response.name, "tool_response");
            }
            other => panic!("expected FunctionResponse, got: {other:?}"),
        }
    }

    #[test]
    fn parse_safety_finish_reason() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::Text {
                        text: "blocked".into(),
                    }],
                }),
                finish_reason: Some("SAFETY".into()),
            }],
            usage_metadata: None,
        };
        let result = parse_gemini_response(resp).unwrap();
        assert_eq!(result.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn parse_no_finish_reason_with_tool_calls() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            name: "search".into(),
                            args: json!({}),
                        },
                    }],
                }),
                finish_reason: None,
            }],
            usage_metadata: None,
        };
        let result = parse_gemini_response(resp).unwrap();
        assert_eq!(result.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn build_request_tool_choice_any() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("test")],
            tools: vec![ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                input_schema: json!({}),
            }],
            max_tokens: 100,
            tool_choice: Some(ToolChoice::Any),
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        let tc = gemini_req.tool_config.unwrap();
        assert_eq!(tc.function_calling_config.mode, "ANY");
        assert!(tc.function_calling_config.allowed_function_names.is_none());
    }

    #[test]
    fn trailing_slash_in_base_url_handled() {
        let p = GeminiProvider::with_base_url("key", "model", "https://example.com/v1/");
        assert_eq!(
            p.generate_url(),
            "https://example.com/v1/models/model:generateContent"
        );
    }

    #[test]
    fn parse_cached_content_token_count() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: Some("model".into()),
                    parts: vec![GeminiPart::Text {
                        text: "cached".into(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: 50,
                candidates_token_count: 10,
                cached_content_token_count: 30,
            }),
        };
        let result = parse_gemini_response(resp).unwrap();
        assert_eq!(result.usage.cache_read_input_tokens, 30);
    }

    #[test]
    fn build_request_no_tool_config_when_no_choice() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("test")],
            tools: vec![],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        assert!(gemini_req.tool_config.is_none());
    }

    #[test]
    fn build_request_error_tool_result() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call-1".into(),
                    content: "something went wrong".into(),
                    is_error: true,
                }],
            }],
            tools: vec![],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        match &gemini_req.contents[0].parts[0] {
            GeminiPart::FunctionResponse {
                function_response, ..
            } => {
                assert_eq!(
                    function_response.response,
                    json!({"error": "something went wrong"})
                );
            }
            other => panic!("expected FunctionResponse, got: {other:?}"),
        }
    }

    #[test]
    fn system_instruction_omits_role() {
        let req = CompletionRequest {
            system: "Be helpful".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 50,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        let si = gemini_req.system_instruction.unwrap();
        assert!(si.role.is_none());
        // Verify it serializes without role
        let json = serde_json::to_value(&si).unwrap();
        assert!(json.get("role").is_none());
    }

    #[test]
    fn normalize_strips_unsupported_fields() {
        let schema = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "MySchema",
            "additionalProperties": false,
            "properties": {
                "name": {
                    "type": "string",
                    "default": "foo",
                    "format": "uri"
                }
            }
        });
        let normalized = normalize_gemini_schema(&schema);
        assert!(normalized.get("$schema").is_none());
        assert!(normalized.get("title").is_none());
        assert!(normalized.get("additionalProperties").is_none());
        let props = normalized.get("properties").unwrap();
        let name = props.get("name").unwrap();
        assert!(name.get("default").is_none());
        assert!(name.get("format").is_none());
        assert_eq!(name.get("type").unwrap(), "string");
    }

    #[test]
    fn normalize_flattens_any_of_with_null() {
        let schema = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        });
        let normalized = normalize_gemini_schema(&schema);
        assert_eq!(normalized.get("type").unwrap(), "string");
        assert_eq!(normalized.get("nullable").unwrap(), true);
        assert!(normalized.get("anyOf").is_none());
    }

    #[test]
    fn normalize_flattens_type_array_with_null() {
        let schema = json!({
            "type": ["string", "null"]
        });
        let normalized = normalize_gemini_schema(&schema);
        assert_eq!(normalized.get("type").unwrap(), "string");
        assert_eq!(normalized.get("nullable").unwrap(), true);
    }

    #[test]
    fn normalize_preserves_valid_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"}
            },
            "required": ["query"]
        });
        let normalized = normalize_gemini_schema(&schema);
        assert_eq!(normalized, schema);
    }

    #[test]
    fn build_request_normalizes_tool_schemas() {
        let req = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("test")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search".into(),
                input_schema: json!({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "q": {"type": "string", "title": "Query"}
                    }
                }),
            }],
            max_tokens: 100,
            tool_choice: None,
            reasoning_effort: None,
        };
        let gemini_req = build_gemini_request(&req);
        let tools = gemini_req.tools.unwrap();
        let params = &tools[0].function_declarations[0].parameters;
        assert!(params.get("$schema").is_none());
        assert!(params.get("additionalProperties").is_none());
        let q = params.get("properties").unwrap().get("q").unwrap();
        assert!(q.get("title").is_none());
    }
}
