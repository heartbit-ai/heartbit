use bytes::Bytes;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use tracing::warn;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::anthropic::SseParser;
use crate::llm::types::{
    CompletionRequest, CompletionResponse, ContentBlock, Role, StopReason, TokenUsage, ToolChoice,
    ToolDefinition,
};

const API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// OpenRouter LLM provider (OpenAI-compatible API).
///
/// Supports both non-streaming (`complete`) and streaming (`stream_complete`)
/// modes. Streaming uses OpenAI's SSE format with `choices[].delta` chunks.
pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

impl LlmProvider for OpenRouterProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let body = build_openai_request(&self.model, &request)?;

        let response = self
            .client
            .post(API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            // Sanitize body for auth failures to avoid leaking API key fragments in logs
            let message = if status.as_u16() == 401 || status.as_u16() == 403 {
                format!("authentication failed (HTTP {})", status.as_u16())
            } else {
                response
                    .text()
                    .await
                    .unwrap_or_else(|e| format!("<body read error: {e}>"))
            };
            return Err(Error::Api {
                status: status.as_u16(),
                message,
            });
        }

        let api_response: OpenAiResponse = response.json().await?;
        into_completion_response(api_response)
    }

    async fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &crate::llm::OnText,
    ) -> Result<CompletionResponse, Error> {
        let mut body = build_openai_request(&self.model, &request)?;
        body["stream"] = serde_json::json!(true);
        body["stream_options"] = serde_json::json!({"include_usage": true});

        let response = self
            .client
            .post(API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let message = if status.as_u16() == 401 || status.as_u16() == 403 {
                format!("authentication failed (HTTP {})", status.as_u16())
            } else {
                response
                    .text()
                    .await
                    .unwrap_or_else(|e| format!("<body read error: {e}>"))
            };
            return Err(Error::Api {
                status: status.as_u16(),
                message,
            });
        }

        parse_openai_stream(response.bytes_stream(), on_text).await
    }
}

// --- Request building: our types → OpenAI format ---

fn build_openai_request(
    model: &str,
    request: &CompletionRequest,
) -> Result<serde_json::Value, Error> {
    let mut messages = Vec::new();

    // System message
    if !request.system.is_empty() {
        messages.push(serde_json::json!({
            "role": "system",
            "content": request.system,
        }));
    }

    // Convert our messages to OpenAI format
    for msg in &request.messages {
        match msg.role {
            Role::User => {
                // Collect text blocks into a single message to avoid consecutive user messages
                let mut text_parts = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            text_parts.push(text.as_str());
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            // OpenAI format has no is_error field; prefix content
                            // so the LLM sees the error context.
                            let content = if *is_error {
                                format!("[ERROR] {content}")
                            } else {
                                content.clone()
                            };
                            messages.push(serde_json::json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": content,
                            }));
                        }
                        _ => {}
                    }
                }
                if !text_parts.is_empty() {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": text_parts.join("\n\n"),
                    }));
                } else if !msg
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
                {
                    // No text blocks and no tool results — add empty user message
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": "",
                    }));
                }
            }
            Role::Assistant => {
                let text: String = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");

                let tool_calls: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": serde_json::to_string(input)
                                    .expect("serde_json::Value serialization is infallible"),
                            }
                        })),
                        _ => None,
                    })
                    .collect();

                let mut msg_json = serde_json::json!({
                    "role": "assistant",
                });

                if !text.is_empty() {
                    msg_json["content"] = serde_json::Value::String(text);
                } else {
                    msg_json["content"] = serde_json::Value::Null;
                }

                if !tool_calls.is_empty() {
                    msg_json["tool_calls"] = serde_json::Value::Array(tool_calls);
                }

                messages.push(msg_json);
            }
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": request.max_tokens,
    });

    // Convert tools to OpenAI function format
    if !request.tools.is_empty() {
        let tools: Vec<serde_json::Value> = request.tools.iter().map(tool_to_openai).collect();
        body["tools"] = serde_json::Value::Array(tools);
    }

    // Convert tool_choice to OpenAI format
    if let Some(ref tc) = request.tool_choice {
        body["tool_choice"] = tool_choice_to_openai(tc);
    }

    Ok(body)
}

/// Convert our `ToolChoice` to OpenAI's format.
///
/// - `Auto` → `"auto"` (string)
/// - `Any` → `"required"` (OpenAI's equivalent of "must call a tool")
/// - `Tool { name }` → `{"type": "function", "function": {"name": "..."}}`
fn tool_choice_to_openai(tc: &ToolChoice) -> serde_json::Value {
    match tc {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::Any => serde_json::json!("required"),
        ToolChoice::Tool { name } => serde_json::json!({
            "type": "function",
            "function": {"name": name}
        }),
    }
}

fn tool_to_openai(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
    })
}

// --- Response parsing: OpenAI format → our types ---

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Deserialize)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiFunction,
}

#[derive(Deserialize)]
struct OpenAiFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Default)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

fn into_completion_response(api: OpenAiResponse) -> Result<CompletionResponse, Error> {
    let choice = api.choices.into_iter().next().ok_or_else(|| Error::Api {
        status: 0,
        message: "empty choices array in response".into(),
    })?;

    let mut content = Vec::new();

    // Text content
    if let Some(text) = choice.message.content
        && !text.is_empty()
    {
        content.push(ContentBlock::Text { text });
    }

    // Tool calls
    if let Some(tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value = if tc.function.arguments.is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(&tc.function.arguments).unwrap_or_else(|e| {
                    tracing::warn!(
                        tool = %tc.function.name,
                        error = %e,
                        "malformed tool arguments JSON, defaulting to empty object"
                    );
                    serde_json::json!({})
                })
            };
            content.push(ContentBlock::ToolUse {
                id: tc.id,
                name: tc.function.name,
                input,
            });
        }
    }

    let has_tool_calls = content
        .iter()
        .any(|c| matches!(c, ContentBlock::ToolUse { .. }));

    // Normalize: some providers send "stop" even when tool calls are present.
    let stop_reason = match choice.finish_reason.as_deref() {
        Some("tool_calls") => StopReason::ToolUse,
        Some("stop") if has_tool_calls => StopReason::ToolUse,
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some(other) => {
            warn!(
                finish_reason = other,
                "unknown finish_reason, treating as EndTurn"
            );
            StopReason::EndTurn
        }
        None => StopReason::EndTurn,
    };

    let usage = api.usage.map_or(TokenUsage::default(), |u| TokenUsage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        cache_creation_input_tokens: u.cache_creation_input_tokens,
        cache_read_input_tokens: u.cache_read_input_tokens,
    });

    Ok(CompletionResponse {
        content,
        stop_reason,
        usage,
    })
}

// --- Streaming: OpenAI SSE delta format ---

#[derive(Deserialize, Default)]
struct StreamingChunk {
    #[serde(default)]
    choices: Vec<StreamingChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct StreamingChoice {
    #[serde(default)]
    delta: StreamingDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<StreamingToolCallDelta>>,
}

#[derive(Deserialize)]
struct StreamingToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamingFunctionDelta>,
}

#[derive(Deserialize)]
struct StreamingFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Default)]
struct AccumulatedToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Parse an OpenAI-format SSE stream, emitting text deltas via `on_text`.
///
/// Reuses the `SseParser` from the Anthropic module for SSE framing.
/// The JSON payload format differs: OpenAI uses `choices[].delta` with
/// incremental content and tool call fragments.
pub(crate) async fn parse_openai_stream<S>(
    stream: S,
    on_text: &super::OnText,
) -> Result<CompletionResponse, Error>
where
    S: futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    let mut parser = SseParser::new();
    let mut utf8_buf: Vec<u8> = Vec::new();
    let mut text = String::new();
    let mut tool_calls: Vec<AccumulatedToolCall> = Vec::new();
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
                process_openai_event(
                    &event.data,
                    on_text,
                    &mut text,
                    &mut tool_calls,
                    &mut finish_reason,
                    &mut usage,
                );
            }
        }
        utf8_buf.drain(..valid_len);
    }

    // Remaining buffer
    if !utf8_buf.is_empty()
        && let Ok(s) = std::str::from_utf8(&utf8_buf)
    {
        for event in parser.feed(s) {
            process_openai_event(
                &event.data,
                on_text,
                &mut text,
                &mut tool_calls,
                &mut finish_reason,
                &mut usage,
            );
        }
    }

    for event in parser.flush() {
        process_openai_event(
            &event.data,
            on_text,
            &mut text,
            &mut tool_calls,
            &mut finish_reason,
            &mut usage,
        );
    }

    // Build content blocks
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(ContentBlock::Text { text });
    }
    for tc in tool_calls {
        let input = if tc.arguments.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&tc.arguments).unwrap_or_else(|e| {
                warn!(tool = %tc.name, error = %e, "malformed streaming tool arguments");
                serde_json::json!({})
            })
        };
        content.push(ContentBlock::ToolUse {
            id: tc.id,
            name: tc.name,
            input,
        });
    }

    let has_tool_calls = content
        .iter()
        .any(|c| matches!(c, ContentBlock::ToolUse { .. }));

    // Normalize: some providers send "stop" even when tool calls are present.
    // The agent loop checks content (not stop_reason) for tool detection, but
    // we normalize here so the CompletionResponse is semantically correct.
    let stop_reason = match finish_reason.as_deref() {
        Some("tool_calls") => StopReason::ToolUse,
        Some("stop") if has_tool_calls => StopReason::ToolUse,
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some(other) => {
            warn!(
                finish_reason = other,
                "unknown finish_reason in stream, treating as EndTurn"
            );
            StopReason::EndTurn
        }
        None => StopReason::EndTurn,
    };

    Ok(CompletionResponse {
        content,
        stop_reason,
        usage,
    })
}

fn process_openai_event(
    data: &str,
    on_text: &super::OnText,
    text: &mut String,
    tool_calls: &mut Vec<AccumulatedToolCall>,
    finish_reason: &mut Option<String>,
    usage: &mut TokenUsage,
) {
    if data == "[DONE]" {
        return;
    }

    let chunk: StreamingChunk = match serde_json::from_str(data) {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "failed to parse streaming chunk, skipping");
            return;
        }
    };

    if let Some(choice) = chunk.choices.first() {
        if let Some(ref content) = choice.delta.content {
            text.push_str(content);
            on_text(content);
        }

        if let Some(ref tcs) = choice.delta.tool_calls {
            for tc_delta in tcs {
                while tool_calls.len() <= tc_delta.index {
                    tool_calls.push(AccumulatedToolCall::default());
                }
                let tc = &mut tool_calls[tc_delta.index];
                if let Some(ref id) = tc_delta.id {
                    tc.id.clone_from(id);
                }
                if let Some(ref func) = tc_delta.function {
                    if let Some(ref name) = func.name {
                        tc.name.clone_from(name);
                    }
                    if let Some(ref args) = func.arguments {
                        tc.arguments.push_str(args);
                    }
                }
            }
        }

        if choice.finish_reason.is_some() {
            *finish_reason = choice.finish_reason.clone();
        }
    }

    if let Some(ref u) = chunk.usage {
        *usage = TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cache_creation_input_tokens: u.cache_creation_input_tokens,
            cache_read_input_tokens: u.cache_read_input_tokens,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::Message;
    use serde_json::json;

    // --- Request building tests ---

    #[test]
    fn build_request_minimal() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("hello")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body = build_openai_request("anthropic/claude-sonnet-4", &request).unwrap();
        assert_eq!(body["model"], "anthropic/claude-sonnet-4");
        assert_eq!(body["max_tokens"], 1024);

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello");
    }

    #[test]
    fn build_request_with_system() {
        let request = CompletionRequest {
            system: "You are helpful.".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body = build_openai_request("model", &request).unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn build_request_with_tools() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("search")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search the web".into(),
                input_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            }],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body = build_openai_request("model", &request).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "search");
    }

    #[test]
    fn build_request_assistant_with_tool_calls() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![
                Message::user("search for rust"),
                Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentBlock::Text {
                            text: "Let me search.".into(),
                        },
                        ContentBlock::ToolUse {
                            id: "call-1".into(),
                            name: "search".into(),
                            input: json!({"q": "rust"}),
                        },
                    ],
                },
            ],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body = build_openai_request("model", &request).unwrap();
        let messages = body["messages"].as_array().unwrap();
        let assistant_msg = &messages[1];
        assert_eq!(assistant_msg["role"], "assistant");
        assert_eq!(assistant_msg["content"], "Let me search.");
        assert_eq!(assistant_msg["tool_calls"][0]["id"], "call-1");
        assert_eq!(assistant_msg["tool_calls"][0]["function"]["name"], "search");
    }

    #[test]
    fn build_request_tool_results() {
        use crate::llm::types::ToolResult;

        let request = CompletionRequest {
            system: String::new(),
            messages: vec![Message::tool_results(vec![
                ToolResult::success("call-1", "found it"),
                ToolResult::error("call-2", "not found"),
            ])],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body = build_openai_request("model", &request).unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2); // Two separate tool messages
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "call-1");
        assert_eq!(messages[0]["content"], "found it");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call-2");
        // Error tool results get [ERROR] prefix since OpenAI format has no is_error field
        assert_eq!(messages[1]["content"], "[ERROR] not found");
    }

    // --- Response parsing tests ---

    #[test]
    fn parse_text_response() {
        let api = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiMessage {
                    content: Some("Hello!".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(OpenAiUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                ..Default::default()
            }),
        };

        let response = into_completion_response(api).unwrap();
        assert_eq!(response.text(), "Hello!");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[test]
    fn parse_tool_call_response() {
        let api = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiMessage {
                    content: Some("Let me search.".into()),
                    tool_calls: Some(vec![OpenAiToolCall {
                        id: "call_abc".into(),
                        function: OpenAiFunction {
                            name: "search".into(),
                            arguments: r#"{"q":"rust"}"#.into(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: Some(OpenAiUsage {
                prompt_tokens: 20,
                completion_tokens: 10,
                ..Default::default()
            }),
        };

        let response = into_completion_response(api).unwrap();
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.text(), "Let me search.");

        let calls = response.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].input["q"], "rust");
    }

    #[test]
    fn parse_max_tokens_response() {
        let api = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiMessage {
                    content: Some("truncated...".into()),
                    tool_calls: None,
                },
                finish_reason: Some("length".into()),
            }],
            usage: None,
        };

        let response = into_completion_response(api).unwrap();
        assert_eq!(response.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn parse_empty_choices_errors() {
        let api = OpenAiResponse {
            choices: vec![],
            usage: None,
        };

        let err = into_completion_response(api).unwrap_err();
        assert!(err.to_string().contains("empty choices"));
    }

    #[test]
    fn parse_parallel_tool_calls() {
        let api = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiMessage {
                    content: None,
                    tool_calls: Some(vec![
                        OpenAiToolCall {
                            id: "call_1".into(),
                            function: OpenAiFunction {
                                name: "search".into(),
                                arguments: r#"{"q":"a"}"#.into(),
                            },
                        },
                        OpenAiToolCall {
                            id: "call_2".into(),
                            function: OpenAiFunction {
                                name: "read".into(),
                                arguments: r#"{"path":"/tmp"}"#.into(),
                            },
                        },
                    ]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        };

        let response = into_completion_response(api).unwrap();
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].name, "read");
    }

    #[test]
    fn parse_stop_with_tool_calls_normalizes_to_tool_use() {
        // Some providers send finish_reason "stop" even with tool_calls present.
        // We normalize to ToolUse for semantic correctness.
        let api = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiMessage {
                    content: None,
                    tool_calls: Some(vec![OpenAiToolCall {
                        id: "call_1".into(),
                        function: OpenAiFunction {
                            name: "search".into(),
                            arguments: "{}".into(),
                        },
                    }]),
                },
                finish_reason: Some("stop".into()), // wrong but real
            }],
            usage: None,
        };

        let response = into_completion_response(api).unwrap();
        assert_eq!(response.stop_reason, StopReason::ToolUse); // normalized
        assert_eq!(response.tool_calls().len(), 1);
    }

    #[test]
    fn build_request_multi_text_blocks_concatenated() {
        // Multiple text blocks in a user message should be concatenated into
        // a single message to avoid consecutive user messages (OpenAI constraint).
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "First paragraph.".into(),
                    },
                    ContentBlock::Text {
                        text: "Second paragraph.".into(),
                    },
                ],
            }],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body = build_openai_request("model", &request).unwrap();
        let messages = body["messages"].as_array().unwrap();
        // Should produce a single user message, not two
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(
            messages[0]["content"],
            "First paragraph.\n\nSecond paragraph."
        );
    }

    #[test]
    fn build_request_mixed_user_message_tool_results_before_text() {
        // When a User message has both Text and ToolResult blocks, OpenAI format
        // requires tool messages immediately after the assistant's tool_calls.
        // The current implementation correctly emits tool messages first, then
        // the user text message, regardless of block order in the source message.
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![
                Message::user("search for rust"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "call-1".into(),
                        name: "search".into(),
                        input: json!({"q": "rust"}),
                    }],
                },
                // Mixed message: text + tool result
                Message {
                    role: Role::User,
                    content: vec![
                        ContentBlock::Text {
                            text: "Here are the results:".into(),
                        },
                        ContentBlock::ToolResult {
                            tool_use_id: "call-1".into(),
                            content: "found it".into(),
                            is_error: false,
                        },
                    ],
                },
            ],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body = build_openai_request("model", &request).unwrap();
        let messages = body["messages"].as_array().unwrap();
        // user + assistant + tool + user(text)
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        // Tool result comes before user text (correct for OpenAI format)
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call-1");
        assert_eq!(messages[3]["role"], "user");
        assert_eq!(messages[3]["content"], "Here are the results:");
    }

    // --- tool_choice tests ---

    #[test]
    fn build_request_no_tool_choice_omits_field() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };
        let body = build_openai_request("model", &request).unwrap();
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn build_request_tool_choice_auto() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: Some(ToolChoice::Auto),
        };
        let body = build_openai_request("model", &request).unwrap();
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn build_request_tool_choice_any() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: Some(ToolChoice::Any),
        };
        let body = build_openai_request("model", &request).unwrap();
        // OpenAI uses "required" for "must call a tool"
        assert_eq!(body["tool_choice"], "required");
    }

    #[test]
    fn build_request_tool_choice_specific_tool() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: Some(ToolChoice::Tool {
                name: "search".into(),
            }),
        };
        let body = build_openai_request("model", &request).unwrap();
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "search");
    }

    // --- Roundtrip test: request → response → request ---

    #[test]
    fn full_conversation_roundtrip() {
        use crate::llm::types::ToolResult;

        // Build initial request
        let request1 = CompletionRequest {
            system: "You are helpful.".into(),
            messages: vec![Message::user("search for rust")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search".into(),
                input_schema: json!({"type": "object"}),
            }],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body1 = build_openai_request("model", &request1).unwrap();
        assert!(body1["messages"].as_array().unwrap().len() == 2); // system + user

        // Simulate tool call response
        let response1 = into_completion_response(OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiMessage {
                    content: Some("Searching...".into()),
                    tool_calls: Some(vec![OpenAiToolCall {
                        id: "call_1".into(),
                        function: OpenAiFunction {
                            name: "search".into(),
                            arguments: r#"{"q":"rust"}"#.into(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        })
        .unwrap();

        // Build follow-up with tool results
        let request2 = CompletionRequest {
            system: "You are helpful.".into(),
            messages: vec![
                Message::user("search for rust"),
                Message {
                    role: Role::Assistant,
                    content: response1.content,
                },
                Message::tool_results(vec![ToolResult::success("call_1", "Rust is great")]),
            ],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
        };

        let body2 = build_openai_request("model", &request2).unwrap();
        let msgs = body2["messages"].as_array().unwrap();
        // system + user + assistant (with tool_calls) + tool result
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[3]["role"], "tool");
    }

    // --- Streaming tests ---

    fn make_sse_data(chunks: &[&str]) -> String {
        chunks
            .iter()
            .map(|c| format!("data: {c}\n\n"))
            .collect::<Vec<_>>()
            .join("")
            + "data: [DONE]\n\n"
    }

    #[tokio::test]
    async fn stream_text_response() {
        let sse = make_sse_data(&[
            r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
        ]);

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse))]);
        let received = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let r = received.clone();
        let on_text: &crate::llm::OnText = &move |t: &str| {
            r.lock().expect("lock").push(t.to_string());
        };

        let response = parse_openai_stream(stream, on_text).await.unwrap();
        assert_eq!(response.text(), "Hello world");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);

        let texts = received.lock().expect("lock");
        assert_eq!(*texts, vec!["Hello", " world"]);
    }

    #[tokio::test]
    async fn stream_tool_call_response() {
        let sse = make_sse_data(&[
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"rust\"}"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":10}}"#,
        ]);

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse))]);
        let on_text: &crate::llm::OnText = &|_| {};

        let response = parse_openai_stream(stream, on_text).await.unwrap();
        assert_eq!(response.stop_reason, StopReason::ToolUse);

        let calls = response.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].input["q"], "rust");
    }

    #[tokio::test]
    async fn stream_parallel_tool_calls() {
        let sse = make_sse_data(&[
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"search","arguments":""}},{"index":1,"id":"c2","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}},{"index":1,"function":{"arguments":"{}"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ]);

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse))]);
        let on_text: &crate::llm::OnText = &|_| {};

        let response = parse_openai_stream(stream, on_text).await.unwrap();
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].name, "read");
    }

    #[tokio::test]
    async fn stream_text_with_tool_calls() {
        let sse = make_sse_data(&[
            r#"{"choices":[{"delta":{"content":"Let me search."},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"search","arguments":"{}"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ]);

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse))]);
        let on_text: &crate::llm::OnText = &|_| {};

        let response = parse_openai_stream(stream, on_text).await.unwrap();
        assert_eq!(response.text(), "Let me search.");
        assert_eq!(response.tool_calls().len(), 1);
    }

    #[tokio::test]
    async fn stream_max_tokens() {
        let sse = make_sse_data(&[
            r#"{"choices":[{"delta":{"content":"trunc"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#,
        ]);

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse))]);
        let on_text: &crate::llm::OnText = &|_| {};

        let response = parse_openai_stream(stream, on_text).await.unwrap();
        assert_eq!(response.stop_reason, StopReason::MaxTokens);
    }

    #[tokio::test]
    async fn stream_stop_with_tool_calls_normalizes() {
        // Provider sends "stop" with tool calls — should normalize to ToolUse
        let sse = make_sse_data(&[
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"search","arguments":"{}"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse))]);
        let on_text: &crate::llm::OnText = &|_| {};

        let response = parse_openai_stream(stream, on_text).await.unwrap();
        assert_eq!(response.stop_reason, StopReason::ToolUse); // normalized
        assert_eq!(response.tool_calls().len(), 1);
    }

    #[test]
    fn parse_response_with_cache_tokens() {
        let api = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiMessage {
                    content: Some("Hello!".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(OpenAiUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                cache_creation_input_tokens: 80,
                cache_read_input_tokens: 60,
            }),
        };

        let response = into_completion_response(api).unwrap();
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 20);
        assert_eq!(response.usage.cache_creation_input_tokens, 80);
        assert_eq!(response.usage.cache_read_input_tokens, 60);
    }

    #[test]
    fn parse_response_cache_tokens_default_when_missing() {
        let api = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiMessage {
                    content: Some("Hello!".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(OpenAiUsage {
                prompt_tokens: 50,
                completion_tokens: 10,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            }),
        };

        let response = into_completion_response(api).unwrap();
        assert_eq!(response.usage.cache_creation_input_tokens, 0);
        assert_eq!(response.usage.cache_read_input_tokens, 0);
    }

    #[tokio::test]
    async fn stream_cache_tokens_passthrough() {
        let sse = make_sse_data(&[
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"cache_creation_input_tokens":80,"cache_read_input_tokens":60}}"#,
        ]);

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse))]);
        let on_text: &crate::llm::OnText = &|_| {};

        let response = parse_openai_stream(stream, on_text).await.unwrap();
        assert_eq!(response.usage.cache_creation_input_tokens, 80);
        assert_eq!(response.usage.cache_read_input_tokens, 60);
    }

    #[tokio::test]
    async fn stream_chunked_delivery() {
        // SSE data split across multiple byte chunks
        let sse = make_sse_data(&[
            r#"{"choices":[{"delta":{"content":"he"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"content":"llo"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mid = sse.len() / 2;
        let chunk1 = Bytes::from(sse[..mid].to_string());
        let chunk2 = Bytes::from(sse[mid..].to_string());

        let stream = futures::stream::iter(vec![Ok(chunk1), Ok(chunk2)]);
        let on_text: &crate::llm::OnText = &|_| {};

        let response = parse_openai_stream(stream, on_text).await.unwrap();
        assert_eq!(response.text(), "hello");
    }
}
