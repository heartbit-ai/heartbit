use bytes::Bytes;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use tracing::warn;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{
    CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
    prompt_caching: bool,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            prompt_caching: false,
        }
    }

    /// Create an `AnthropicProvider` with prompt caching enabled.
    ///
    /// When enabled, the system prompt and tool definitions are annotated with
    /// `cache_control` breakpoints so that Anthropic caches the prefix
    /// server-side, reducing input tokens on subsequent turns.
    pub fn with_prompt_caching(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            prompt_caching: true,
        }
    }
}

impl LlmProvider for AnthropicProvider {
    fn model_name(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let body = build_request_body(&self.model, &request, self.prompt_caching)?;

        let response = self
            .client
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
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

        let api_response: ApiResponse = response.json().await?;
        Ok(into_completion_response(api_response))
    }

    async fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &super::OnText,
    ) -> Result<CompletionResponse, Error> {
        let mut body = build_request_body(&self.model, &request, self.prompt_caching)?;
        body["stream"] = serde_json::Value::Bool(true);

        let response = self
            .client
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
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

        parse_sse_stream_with_callback(response.bytes_stream(), on_text).await
    }
}

fn build_request_body(
    model: &str,
    request: &CompletionRequest,
    prompt_caching: bool,
) -> Result<serde_json::Value, Error> {
    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": request.max_tokens,
        "messages": request.messages,
    });

    if !request.system.is_empty() {
        if prompt_caching {
            // Array format with cache_control on the system block
            body["system"] = serde_json::json!([{
                "type": "text",
                "text": request.system,
                "cache_control": {"type": "ephemeral"}
            }]);
        } else {
            body["system"] = serde_json::Value::String(request.system.clone());
        }
    }

    if !request.tools.is_empty() {
        let mut tools = serde_json::to_value(&request.tools)?;
        if prompt_caching {
            // Add cache_control to the last tool definition
            if let Some(arr) = tools.as_array_mut()
                && let Some(last) = arr.last_mut()
            {
                last["cache_control"] = serde_json::json!({"type": "ephemeral"});
            }
        }
        body["tools"] = tools;
    }

    if prompt_caching {
        // Add cache_control to the last content block of the second-to-last
        // user message. This creates a cache breakpoint so the conversation
        // prefix up to that point is cached across turns.
        if let Some(messages) = body["messages"].as_array_mut() {
            let user_indices: Vec<usize> = messages
                .iter()
                .enumerate()
                .filter(|(_, m)| m["role"] == "user")
                .map(|(i, _)| i)
                .collect();
            // Place breakpoint on the second-to-last user message
            if user_indices.len() >= 2 {
                let idx = user_indices[user_indices.len() - 2];
                if let Some(content) = messages[idx]["content"].as_array_mut()
                    && let Some(last_block) = content.last_mut()
                {
                    last_block["cache_control"] = serde_json::json!({"type": "ephemeral"});
                }
            }
        }
    }

    if let Some(ref tc) = request.tool_choice {
        body["tool_choice"] = serde_json::to_value(tc)?;
    }

    // Add extended thinking for Anthropic models that support it.
    // Maps ReasoningEffort to budget_tokens (fraction of max_tokens).
    // Anthropic requires: budget_tokens >= 1024 and budget_tokens < max_tokens.
    if let Some(effort) = request.reasoning_effort {
        use crate::llm::types::ReasoningEffort;
        const MIN_THINKING_BUDGET: u32 = 1024;
        let raw_budget = match effort {
            ReasoningEffort::High => request.max_tokens.saturating_mul(4),
            ReasoningEffort::Medium => request.max_tokens.saturating_mul(2),
            ReasoningEffort::Low => request.max_tokens,
            ReasoningEffort::None => 0,
        };
        // Clamp: at least 1024 (Anthropic minimum), at most max_tokens - 1
        // (budget_tokens must be strictly less than max_tokens).
        if raw_budget > 0 && request.max_tokens > MIN_THINKING_BUDGET {
            let budget = raw_budget
                .max(MIN_THINKING_BUDGET)
                .min(request.max_tokens.saturating_sub(1));
            body["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget
            });
        }
    }

    Ok(body)
}

// --- SSE Parser ---

/// Incremental SSE parser that handles:
/// - `\n`, `\r\n`, and `\r` line endings
/// - Multiple `data:` fields (concatenated with `\n` per SSE spec)
/// - `data:value` (no space) and `data: value` (space stripped per spec)
/// - Arbitrary chunk boundaries (events split across chunks)
/// - Remaining data after stream ends (via `flush()`)
pub(crate) struct SseParser {
    buffer: String,
    event_type: String,
    data_lines: Vec<String>,
}

impl SseParser {
    pub(crate) fn new() -> Self {
        Self {
            buffer: String::new(),
            event_type: String::new(),
            data_lines: Vec::new(),
        }
    }

    /// Feed a chunk of data and return any complete events.
    pub(crate) fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        while let Some((line, consumed)) = self.next_line() {
            self.buffer.drain(..consumed);
            self.process_line(&line, &mut events);
        }

        events
    }

    /// Flush remaining data after the stream ends.
    pub(crate) fn flush(mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();

        // Process remaining buffer as a final line.
        // Strip trailing \r since next_line() defers lone \r at buffer end
        // (waiting for a possible \n in the next chunk that never arrives).
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            let line = line.trim_end_matches('\r');
            self.process_line(line, &mut events);
        }

        // Emit any pending event
        if let Some(event) = self.emit_event() {
            events.push(event);
        }

        events
    }

    fn next_line(&self) -> Option<(String, usize)> {
        let bytes = self.buffer.as_bytes();
        for i in 0..bytes.len() {
            match bytes[i] {
                b'\n' => {
                    return Some((self.buffer[..i].to_string(), i + 1));
                }
                b'\r' => {
                    if i + 1 >= bytes.len() {
                        // \r at end of buffer — might be \r\n split across chunks
                        return None;
                    }
                    let consumed = if bytes[i + 1] == b'\n' { i + 2 } else { i + 1 };
                    return Some((self.buffer[..i].to_string(), consumed));
                }
                _ => {}
            }
        }
        None
    }

    fn process_line(&mut self, line: &str, events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            // Blank line: dispatch event
            if let Some(event) = self.emit_event() {
                events.push(event);
            }
        } else if line.starts_with(':') {
            // Comment, ignore
        } else if let Some(rest) = line.strip_prefix("event:") {
            self.event_type = rest.strip_prefix(' ').unwrap_or(rest).to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            self.data_lines
                .push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
        }
        // Ignore other fields (id, retry, etc.)
    }

    fn emit_event(&mut self) -> Option<SseEvent> {
        if self.event_type.is_empty() && self.data_lines.is_empty() {
            return None;
        }

        Some(SseEvent {
            event_type: std::mem::take(&mut self.event_type),
            data: std::mem::take(&mut self.data_lines).join("\n"),
        })
    }
}

pub(crate) struct SseEvent {
    pub(crate) event_type: String,
    pub(crate) data: String,
}

/// Parse an SSE byte stream into a CompletionResponse, calling `on_text` for each text delta.
///
/// Uses a byte buffer to handle multi-byte UTF-8 sequences split across HTTP chunks.
/// Only complete UTF-8 sequences are decoded; incomplete trailing bytes are held
/// in the buffer until the next chunk arrives.
async fn parse_sse_stream_with_callback<S>(
    stream: S,
    on_text: &super::OnText,
) -> Result<CompletionResponse, Error>
where
    S: futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    let mut state = SseResponseState::default();
    let mut parser = SseParser::new();
    let mut utf8_buf: Vec<u8> = Vec::new();

    tokio::pin!(stream);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(Error::Http)?;
        utf8_buf.extend_from_slice(&chunk);

        // Decode the longest valid UTF-8 prefix, keeping incomplete trailing bytes
        let valid_len = match std::str::from_utf8(&utf8_buf) {
            Ok(_) => utf8_buf.len(),
            Err(e) => e.valid_up_to(),
        };

        if valid_len > 0 {
            // Safety: valid_len is guaranteed to be a valid UTF-8 boundary
            let text = std::str::from_utf8(&utf8_buf[..valid_len])
                .expect("valid_up_to guarantees valid UTF-8");
            let events = parser.feed(text);
            for event in events {
                process_sse_event(&mut state, &event, on_text)?;
            }
        }

        utf8_buf.drain(..valid_len);
    }

    // Process any remaining complete bytes (incomplete UTF-8 at end of stream is dropped)
    if !utf8_buf.is_empty()
        && let Ok(text) = std::str::from_utf8(&utf8_buf)
    {
        let events = parser.feed(text);
        for event in events {
            process_sse_event(&mut state, &event, on_text)?;
        }
    }

    for event in parser.flush() {
        process_sse_event(&mut state, &event, on_text)?;
    }

    Ok(state.into_response())
}

// --- SSE response accumulator ---

#[derive(Default)]
struct SseResponseState {
    content: Vec<ContentBlock>,
    current_text: Option<String>,
    current_tool_use: Option<PartialToolUse>,
    stop_reason: Option<StopReason>,
    usage: TokenUsage,
}

struct PartialToolUse {
    id: String,
    name: String,
    input_json: String,
}

impl SseResponseState {
    fn flush_current_block(&mut self) {
        if let Some(text) = self.current_text.take() {
            self.content.push(ContentBlock::Text { text });
        }
        if let Some(tool) = self.current_tool_use.take() {
            let input = serde_json::from_str(&tool.input_json).unwrap_or_else(|e| {
                tracing::warn!(
                    tool = %tool.name,
                    error = %e,
                    "malformed tool input JSON from SSE stream, defaulting to null"
                );
                serde_json::Value::Null
            });
            self.content.push(ContentBlock::ToolUse {
                id: tool.id,
                name: tool.name,
                input,
            });
        }
    }

    fn into_response(mut self) -> CompletionResponse {
        self.flush_current_block();
        CompletionResponse {
            content: self.content,
            stop_reason: self.stop_reason.unwrap_or(StopReason::EndTurn),
            usage: self.usage,
        }
    }
}

fn process_sse_event(
    state: &mut SseResponseState,
    event: &SseEvent,
    on_text: &super::OnText,
) -> Result<(), Error> {
    match event.event_type.as_str() {
        "message_start" => {
            if let Ok(parsed) = serde_json::from_str::<MessageStartEvent>(&event.data) {
                state.usage.input_tokens = parsed.message.usage.input_tokens;
                state.usage.cache_creation_input_tokens =
                    parsed.message.usage.cache_creation_input_tokens;
                state.usage.cache_read_input_tokens = parsed.message.usage.cache_read_input_tokens;
            }
        }
        "content_block_start" => {
            state.flush_current_block();
            if let Ok(parsed) = serde_json::from_str::<ContentBlockStartEvent>(&event.data) {
                match parsed.content_block.r#type.as_str() {
                    "text" => {
                        state.current_text = Some(String::new());
                    }
                    "tool_use" => {
                        state.current_tool_use = Some(PartialToolUse {
                            id: parsed.content_block.id.unwrap_or_default(),
                            name: parsed.content_block.name.unwrap_or_default(),
                            input_json: String::new(),
                        });
                    }
                    _ => {}
                }
            }
        }
        "content_block_delta" => {
            if let Ok(parsed) = serde_json::from_str::<ContentBlockDeltaEvent>(&event.data) {
                match parsed.delta.r#type.as_str() {
                    "text_delta" => {
                        if let Some(ref delta) = parsed.delta.text {
                            if !delta.is_empty() {
                                on_text(delta);
                            }
                            if let Some(ref mut text) = state.current_text {
                                text.push_str(delta);
                            }
                        }
                    }
                    "input_json_delta" => {
                        if let Some(ref mut tool) = state.current_tool_use {
                            tool.input_json
                                .push_str(&parsed.delta.partial_json.unwrap_or_default());
                        }
                    }
                    _ => {}
                }
            }
        }
        "content_block_stop" => {
            state.flush_current_block();
        }
        "message_delta" => {
            if let Ok(parsed) = serde_json::from_str::<MessageDeltaEvent>(&event.data) {
                state.stop_reason = parsed.delta.stop_reason.as_deref().map(parse_stop_reason);
                if let Some(usage) = parsed.usage {
                    state.usage.output_tokens = usage.output_tokens;
                }
            }
        }
        "ping" | "message_stop" => {}
        other => {
            warn!(event_type = other, "unknown SSE event type");
        }
    }

    Ok(())
}

fn parse_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        other => {
            warn!(
                stop_reason = other,
                "unknown stop_reason, treating as EndTurn"
            );
            StopReason::EndTurn
        }
    }
}

// --- API response types (non-streaming) ---

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ApiContentBlock>,
    stop_reason: String,
    usage: ApiUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

fn into_completion_response(api: ApiResponse) -> CompletionResponse {
    let content = api
        .content
        .into_iter()
        .map(|block| match block {
            ApiContentBlock::Text { text } => ContentBlock::Text { text },
            ApiContentBlock::ToolUse { id, name, input } => {
                ContentBlock::ToolUse { id, name, input }
            }
        })
        .collect();

    CompletionResponse {
        content,
        stop_reason: parse_stop_reason(&api.stop_reason),
        usage: TokenUsage {
            input_tokens: api.usage.input_tokens,
            output_tokens: api.usage.output_tokens,
            cache_creation_input_tokens: api.usage.cache_creation_input_tokens,
            cache_read_input_tokens: api.usage.cache_read_input_tokens,
            reasoning_tokens: 0,
        },
    }
}

// --- SSE event deserialization types ---

#[derive(Deserialize)]
struct MessageStartEvent {
    message: MessageStartMessage,
}

#[derive(Deserialize)]
struct MessageStartMessage {
    usage: ApiUsage,
}

#[derive(Deserialize)]
struct ContentBlockStartEvent {
    content_block: ContentBlockStart,
}

#[derive(Deserialize)]
struct ContentBlockStart {
    r#type: String,
    id: Option<String>,
    name: Option<String>,
}

#[derive(Deserialize)]
struct ContentBlockDeltaEvent {
    delta: ContentBlockDelta,
}

#[derive(Deserialize)]
struct ContentBlockDelta {
    r#type: String,
    text: Option<String>,
    partial_json: Option<String>,
}

#[derive(Deserialize)]
struct MessageDeltaEvent {
    delta: MessageDelta,
    usage: Option<MessageDeltaUsage>,
}

#[derive(Deserialize)]
struct MessageDelta {
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct MessageDeltaUsage {
    output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ContentBlock, Message, Role};
    use std::sync::Arc;

    // --- SseParser unit tests ---

    #[test]
    fn parser_basic_event() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: ping\ndata: {}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "ping");
        assert_eq!(events[0].data, "{}");
    }

    #[test]
    fn parser_crlf_line_endings() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: ping\r\ndata: {}\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "ping");
    }

    #[test]
    fn parser_cr_only_line_endings() {
        let mut parser = SseParser::new();
        // Trailing \r is ambiguous (could be start of \r\n), so feed + flush
        let mut events = parser.feed("event: ping\rdata: {}\r\r");
        events.extend(parser.flush());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "ping");
        assert_eq!(events[0].data, "{}");
    }

    #[test]
    fn parser_multi_data_lines_concatenated() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: test\ndata: line1\ndata: line2\ndata: line3\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn parser_data_no_space_after_colon() {
        let mut parser = SseParser::new();
        let events = parser.feed("event:test\ndata:value\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "test");
        assert_eq!(events[0].data, "value");
    }

    #[test]
    fn parser_comments_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed(": this is a comment\nevent: test\ndata: x\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "test");
    }

    #[test]
    fn parser_chunked_delivery() {
        let mut parser = SseParser::new();

        // Split an event across multiple chunks
        let events1 = parser.feed("event: te");
        assert!(events1.is_empty());

        let events2 = parser.feed("st\nda");
        assert!(events2.is_empty());

        let events3 = parser.feed("ta: hello\n\n");
        assert_eq!(events3.len(), 1);
        assert_eq!(events3[0].event_type, "test");
        assert_eq!(events3[0].data, "hello");
    }

    #[test]
    fn parser_crlf_split_across_chunks() {
        let mut parser = SseParser::new();

        // \r\n split: \r at end of first chunk
        let events1 = parser.feed("event: test\r");
        assert!(events1.is_empty()); // \r at end, wait for more

        let events2 = parser.feed("\ndata: x\r\n\r\n");
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].event_type, "test");
        assert_eq!(events2[0].data, "x");
    }

    #[test]
    fn parser_multiple_events_in_one_chunk() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: a\ndata: 1\n\nevent: b\ndata: 2\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "a");
        assert_eq!(events[1].event_type, "b");
    }

    #[test]
    fn parser_flush_remaining() {
        let mut parser = SseParser::new();

        // Incomplete event (no trailing blank line)
        let events = parser.feed("event: test\ndata: leftover");
        assert!(events.is_empty());

        let flushed = parser.flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].event_type, "test");
        assert_eq!(flushed[0].data, "leftover");
    }

    #[test]
    fn parser_empty_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: test\ndata: \n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn parser_flush_strips_trailing_cr() {
        let mut parser = SseParser::new();

        // Buffer ends with \r (next_line defers it, waiting for possible \n)
        let events = parser.feed("event: test\ndata: hello\r");
        assert!(events.is_empty());

        let flushed = parser.flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].data, "hello"); // no trailing \r
    }

    // --- build_request_body tests ---

    #[test]
    fn build_request_body_minimal() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };

        let body = build_request_body("claude-sonnet-4-20250514", &request, false).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["max_tokens"], 1024);
        assert!(body.get("system").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn build_request_body_with_system_and_tools() {
        use crate::llm::types::ToolDefinition;
        use serde_json::json;

        let request = CompletionRequest {
            system: "You are helpful.".into(),
            messages: vec![Message::user("hi")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search the web".into(),
                input_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            }],
            max_tokens: 2048,
            tool_choice: None,
            reasoning_effort: None,
        };

        let body = build_request_body("claude-sonnet-4-20250514", &request, false).unwrap();
        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["tools"][0]["name"], "search");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn build_request_body_no_tool_choice_omits_field() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn build_request_body_tool_choice_auto() {
        use crate::llm::types::ToolChoice;
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: Some(ToolChoice::Auto),
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert_eq!(body["tool_choice"]["type"], "auto");
    }

    #[test]
    fn build_request_body_tool_choice_any() {
        use crate::llm::types::ToolChoice;
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: Some(ToolChoice::Any),
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert_eq!(body["tool_choice"]["type"], "any");
    }

    #[test]
    fn build_request_body_tool_choice_specific_tool() {
        use crate::llm::types::ToolChoice;
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: Some(ToolChoice::Tool {
                name: "search".into(),
            }),
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "search");
    }

    #[test]
    fn build_request_body_reasoning_effort_medium() {
        use crate::llm::types::ReasoningEffort;
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 4096,
            tool_choice: None,
            reasoning_effort: Some(ReasoningEffort::Medium),
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        // 4096 * 2 = 8192, clamped to max_tokens - 1 = 4095
        assert_eq!(body["thinking"]["budget_tokens"], 4095);
    }

    #[test]
    fn build_request_body_reasoning_effort_high() {
        use crate::llm::types::ReasoningEffort;
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 4096,
            tool_choice: None,
            reasoning_effort: Some(ReasoningEffort::High),
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        // 4096 * 4 = 16384, clamped to max_tokens - 1 = 4095
        assert_eq!(body["thinking"]["budget_tokens"], 4095);
    }

    #[test]
    fn build_request_body_reasoning_effort_none_omits_thinking() {
        use crate::llm::types::ReasoningEffort;
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: Some(ReasoningEffort::None),
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn build_request_body_reasoning_effort_skipped_when_max_tokens_too_small() {
        use crate::llm::types::ReasoningEffort;
        // max_tokens <= MIN_THINKING_BUDGET (1024) → thinking skipped
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: Some(ReasoningEffort::Medium),
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn build_request_body_no_reasoning_omits_thinking() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert!(body.get("thinking").is_none());
    }

    // --- SSE stream integration tests ---

    #[tokio::test]
    async fn parse_sse_stream_text_response() {
        let sse_data = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello \"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world!\"}}\n\n\
            event: content_block_stop\n\
            data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n\
            event: message_stop\n\
            data: {\"type\":\"message_stop\"}\n\n";

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse_data))]);
        let response = parse_sse_stream_with_callback(stream, &|_| {})
            .await
            .unwrap();

        assert_eq!(response.text(), "Hello world!");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[tokio::test]
    async fn parse_sse_stream_chunked_delivery() {
        // Same SSE data but split into multiple chunks at arbitrary boundaries
        let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
            Ok(Bytes::from(
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_to",
            )),
            Ok(Bytes::from(
                "kens\":10,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",",
            )),
            Ok(Bytes::from(
                "\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            )),
            Ok(Bytes::from(
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            )),
        ];

        let stream = futures::stream::iter(chunks);
        let response = parse_sse_stream_with_callback(stream, &|_| {})
            .await
            .unwrap();

        assert_eq!(response.text(), "Hi");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.input_tokens, 10);
    }

    #[tokio::test]
    async fn parse_sse_stream_tool_use_response() {
        let sse_data = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"output_tokens\":0}}}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me search.\"}}\n\n\
            event: content_block_stop\n\
            data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_123\",\"name\":\"search\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\": \"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"rust\\\"}\"}}\n\n\
            event: content_block_stop\n\
            data: {\"type\":\"content_block_stop\",\"index\":1}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":15}}\n\n\
            event: message_stop\n\
            data: {\"type\":\"message_stop\"}\n\n";

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse_data))]);
        let response = parse_sse_stream_with_callback(stream, &|_| {})
            .await
            .unwrap();

        assert_eq!(response.stop_reason, StopReason::ToolUse);
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_123");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].input["q"], "rust");
        assert_eq!(response.text(), "Let me search.");
    }

    #[tokio::test]
    async fn parse_sse_stream_parallel_tool_calls() {
        let sse_data = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_a\",\"name\":\"search\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\": \\\"a\\\"}\"}}\n\n\
            event: content_block_stop\n\
            data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_b\",\"name\":\"read_file\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\": \\\"/tmp\\\"}\"}}\n\n\
            event: content_block_stop\n\
            data: {\"type\":\"content_block_stop\",\"index\":1}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":10}}\n\n\
            event: message_stop\n\
            data: {\"type\":\"message_stop\"}\n\n";

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse_data))]);
        let response = parse_sse_stream_with_callback(stream, &|_| {})
            .await
            .unwrap();

        let calls = response.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].name, "read_file");
    }

    #[tokio::test]
    async fn parse_sse_stream_crlf_format() {
        let sse_data = "event: message_start\r\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\r\n\r\n\
            event: content_block_start\r\n\
            data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\r\n\r\n\
            event: content_block_delta\r\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"OK\"}}\r\n\r\n\
            event: content_block_stop\r\n\
            data: {\"type\":\"content_block_stop\",\"index\":0}\r\n\r\n\
            event: message_delta\r\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\r\n\r\n\
            event: message_stop\r\n\
            data: {\"type\":\"message_stop\"}\r\n\r\n";

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse_data))]);
        let response = parse_sse_stream_with_callback(stream, &|_| {})
            .await
            .unwrap();

        assert_eq!(response.text(), "OK");
    }

    #[tokio::test]
    async fn parse_sse_stream_with_callback_invokes_on_text() {
        use std::sync::Mutex;

        let sse_data = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello \"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world!\"}}\n\n\
            event: content_block_stop\n\
            data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n\
            event: message_stop\n\
            data: {\"type\":\"message_stop\"}\n\n";

        let collected = Arc::new(Mutex::new(Vec::<String>::new()));
        let collected_clone = collected.clone();
        let on_text: &crate::llm::OnText = &move |text: &str| {
            collected_clone.lock().expect("lock").push(text.to_string());
        };

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse_data))]);
        let response = parse_sse_stream_with_callback(stream, on_text)
            .await
            .unwrap();

        assert_eq!(response.text(), "Hello world!");
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);

        let deltas = collected.lock().expect("lock");
        assert_eq!(*deltas, vec!["Hello ", "world!"]);
    }

    #[tokio::test]
    async fn parse_sse_stream_with_callback_tool_use_does_not_invoke_on_text() {
        use std::sync::Mutex;

        // Tool use blocks should NOT trigger the on_text callback
        let sse_data = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"search\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\": \\\"test\\\"}\"}}\n\n\
            event: content_block_stop\n\
            data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":3}}\n\n\
            event: message_stop\n\
            data: {\"type\":\"message_stop\"}\n\n";

        let collected = Arc::new(Mutex::new(Vec::<String>::new()));
        let collected_clone = collected.clone();
        let on_text: &crate::llm::OnText = &move |text: &str| {
            collected_clone.lock().expect("lock").push(text.to_string());
        };

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse_data))]);
        let response = parse_sse_stream_with_callback(stream, on_text)
            .await
            .unwrap();

        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.tool_calls().len(), 1);

        let deltas = collected.lock().expect("lock");
        assert!(
            deltas.is_empty(),
            "on_text should not be called for tool_use blocks"
        );
    }

    #[tokio::test]
    async fn parse_sse_stream_handles_utf8_split_across_chunks() {
        use std::sync::Mutex;

        // "Hello 🌍!" — the globe emoji is 4 bytes: F0 9F 8C 8D
        // We split the SSE data mid-emoji across two HTTP chunks.
        let full_sse = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello 🌍!\"}}\n\n\
            event: content_block_stop\n\
            data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n\
            event: message_stop\n\
            data: {\"type\":\"message_stop\"}\n\n";

        let bytes = full_sse.as_bytes();
        // Find the emoji position and split mid-emoji (after first 2 of 4 emoji bytes)
        let emoji_pos = bytes.windows(4).position(|w| w == "🌍".as_bytes()).unwrap();
        let split_point = emoji_pos + 2; // split in middle of 4-byte emoji

        let chunk1 = Bytes::copy_from_slice(&bytes[..split_point]);
        let chunk2 = Bytes::copy_from_slice(&bytes[split_point..]);

        let collected = Arc::new(Mutex::new(Vec::<String>::new()));
        let collected_clone = collected.clone();
        let on_text: &crate::llm::OnText = &move |text: &str| {
            collected_clone.lock().expect("lock").push(text.to_string());
        };

        let stream = futures::stream::iter(vec![Ok(chunk1), Ok(chunk2)]);
        let response = parse_sse_stream_with_callback(stream, on_text)
            .await
            .unwrap();

        assert_eq!(response.text(), "Hello 🌍!");

        let deltas = collected.lock().expect("lock");
        // The emoji should appear intact in the collected deltas
        let joined: String = deltas.iter().cloned().collect();
        assert_eq!(joined, "Hello 🌍!");
    }

    #[test]
    fn into_completion_response_maps_correctly() {
        let api = ApiResponse {
            content: vec![
                ApiContentBlock::Text {
                    text: "hello".into(),
                },
                ApiContentBlock::ToolUse {
                    id: "id1".into(),
                    name: "tool1".into(),
                    input: serde_json::json!({}),
                },
            ],
            stop_reason: "tool_use".into(),
            usage: ApiUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        };

        let response = into_completion_response(api);
        assert_eq!(response.content.len(), 2);
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 50);
    }

    // --- Prompt caching tests ---

    #[test]
    fn caching_disabled_system_is_string() {
        let request = CompletionRequest {
            system: "You are helpful.".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, false).unwrap();
        assert!(body["system"].is_string());
        assert_eq!(body["system"], "You are helpful.");
    }

    #[test]
    fn caching_enabled_system_is_array_with_cache_control() {
        let request = CompletionRequest {
            system: "You are helpful.".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, true).unwrap();
        assert!(body["system"].is_array());
        let arr = body["system"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "You are helpful.");
        assert_eq!(arr[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn caching_enabled_last_tool_gets_cache_control() {
        use crate::llm::types::ToolDefinition;
        use serde_json::json;

        let request = CompletionRequest {
            system: "sys".into(),
            messages: vec![Message::user("hi")],
            tools: vec![
                ToolDefinition {
                    name: "search".into(),
                    description: "Search".into(),
                    input_schema: json!({"type": "object"}),
                },
                ToolDefinition {
                    name: "read".into(),
                    description: "Read".into(),
                    input_schema: json!({"type": "object"}),
                },
            ],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, true).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        // Only the LAST tool should have cache_control
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn caching_enabled_single_tool_gets_cache_control() {
        use crate::llm::types::ToolDefinition;
        use serde_json::json;

        let request = CompletionRequest {
            system: "sys".into(),
            messages: vec![Message::user("hi")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search".into(),
                input_schema: json!({"type": "object"}),
            }],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, true).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn caching_enabled_no_tools_no_crash() {
        let request = CompletionRequest {
            system: "sys".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, true).unwrap();
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn caching_disabled_tools_have_no_cache_control() {
        use crate::llm::types::ToolDefinition;
        use serde_json::json;

        let request = CompletionRequest {
            system: "sys".into(),
            messages: vec![Message::user("hi")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search".into(),
                input_schema: json!({"type": "object"}),
            }],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, false).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert!(tools[0].get("cache_control").is_none());
    }

    #[test]
    fn api_usage_cache_fields_deserialize() {
        let json = r#"{"input_tokens":100,"output_tokens":50,"cache_creation_input_tokens":200,"cache_read_input_tokens":300}"#;
        let usage: ApiUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, 200);
        assert_eq!(usage.cache_read_input_tokens, 300);
    }

    #[test]
    fn api_usage_cache_fields_default_when_missing() {
        let json = r#"{"input_tokens":100,"output_tokens":50}"#;
        let usage: ApiUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn response_maps_cache_tokens() {
        let api = ApiResponse {
            content: vec![ApiContentBlock::Text { text: "hi".into() }],
            stop_reason: "end_turn".into(),
            usage: ApiUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: 200,
                cache_read_input_tokens: 300,
            },
        };
        let response = into_completion_response(api);
        assert_eq!(response.usage.cache_creation_input_tokens, 200);
        assert_eq!(response.usage.cache_read_input_tokens, 300);
    }

    #[tokio::test]
    async fn sse_stream_cache_tokens_from_message_start() {
        let sse_data = "event: message_start\n\
            data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0,\"cache_creation_input_tokens\":50,\"cache_read_input_tokens\":100}}}\n\n\
            event: content_block_start\n\
            data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
            event: content_block_delta\n\
            data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
            event: content_block_stop\n\
            data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
            event: message_delta\n\
            data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n\
            event: message_stop\n\
            data: {\"type\":\"message_stop\"}\n\n";

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse_data))]);
        let response = parse_sse_stream_with_callback(stream, &|_| {})
            .await
            .unwrap();

        assert_eq!(response.usage.cache_creation_input_tokens, 50);
        assert_eq!(response.usage.cache_read_input_tokens, 100);
    }

    #[test]
    fn constructor_sets_prompt_caching_false() {
        let provider = AnthropicProvider::new("key", "model");
        assert!(!provider.prompt_caching);
    }

    #[test]
    fn with_prompt_caching_sets_flag() {
        let provider = AnthropicProvider::with_prompt_caching("key", "model");
        assert!(provider.prompt_caching);
    }

    #[test]
    fn model_name_returns_configured_model() {
        let provider = AnthropicProvider::new("key", "claude-3-5-sonnet-20241022");
        assert_eq!(provider.model_name(), Some("claude-3-5-sonnet-20241022"));
    }

    #[test]
    fn caching_marks_second_to_last_user_message() {
        use serde_json::json;

        let request = CompletionRequest {
            system: "sys".into(),
            messages: vec![
                Message::user("first question"),
                Message::assistant("first answer"),
                Message::user("second question"),
                Message::assistant("second answer"),
                Message::user("third question"),
            ],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, true).unwrap();
        let messages = body["messages"].as_array().unwrap();

        // Second-to-last user message is index 2 ("second question")
        assert_eq!(
            messages[2]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        // First user message should NOT have cache_control
        assert!(messages[0]["content"][0].get("cache_control").is_none());
        // Last user message should NOT have cache_control
        assert!(messages[4]["content"][0].get("cache_control").is_none());
    }

    #[test]
    fn caching_no_message_breakpoint_with_single_user_message() {
        let request = CompletionRequest {
            system: "sys".into(),
            messages: vec![Message::user("only question")],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, true).unwrap();
        let messages = body["messages"].as_array().unwrap();

        // Only 1 user message — no breakpoint placed
        assert!(messages[0]["content"][0].get("cache_control").is_none());
    }

    #[test]
    fn caching_disabled_no_message_breakpoints() {
        let request = CompletionRequest {
            system: "sys".into(),
            messages: vec![
                Message::user("q1"),
                Message::assistant("a1"),
                Message::user("q2"),
            ],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, false).unwrap();
        let messages = body["messages"].as_array().unwrap();

        // No cache_control on any message when caching is disabled
        for msg in messages {
            if let Some(content) = msg["content"].as_array() {
                for block in content {
                    assert!(block.get("cache_control").is_none());
                }
            }
        }
    }

    #[test]
    fn caching_message_breakpoint_with_tool_results() {
        use serde_json::json;

        // Simulate a realistic agent conversation with tool calls
        let request = CompletionRequest {
            system: "sys".into(),
            messages: vec![
                Message::user("read file.txt"),
                Message::assistant("I'll read that file."),
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "call_1".into(),
                        content: "file contents here".into(),
                        is_error: false,
                    }],
                },
                Message::assistant("The file contains..."),
                Message::user("now edit it"),
            ],
            tools: vec![],
            max_tokens: 1024,
            tool_choice: None,
            reasoning_effort: None,
        };
        let body = build_request_body("model", &request, true).unwrap();
        let messages = body["messages"].as_array().unwrap();

        // Second-to-last user message is index 2 (the tool result)
        assert_eq!(
            messages[2]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        // Last user message should NOT have cache_control
        assert!(messages[4]["content"][0].get("cache_control").is_none());
    }
}
