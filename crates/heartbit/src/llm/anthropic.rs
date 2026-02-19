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
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

impl LlmProvider for AnthropicProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let body = build_request_body(&self.model, &request)?;

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
}

/// Streaming completion using SSE.
///
/// Currently not wired into the agent loop (which uses `complete()`).
/// Available for future streaming output support.
impl AnthropicProvider {
    pub async fn stream(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let mut body = build_request_body(&self.model, &request)?;
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

        parse_sse_stream(response.bytes_stream()).await
    }
}

fn build_request_body(
    model: &str,
    request: &CompletionRequest,
) -> Result<serde_json::Value, Error> {
    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": request.max_tokens,
        "messages": request.messages,
    });

    if !request.system.is_empty() {
        body["system"] = serde_json::Value::String(request.system.clone());
    }

    if !request.tools.is_empty() {
        body["tools"] = serde_json::to_value(&request.tools)?;
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

        // Process remaining buffer as a final line
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            self.process_line(&line, &mut events);
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

/// Parse an SSE byte stream into a CompletionResponse.
pub(crate) async fn parse_sse_stream<S>(stream: S) -> Result<CompletionResponse, Error>
where
    S: futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    let mut state = SseResponseState::default();
    let mut parser = SseParser::new();

    tokio::pin!(stream);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(Error::Http)?;
        let events = parser.feed(&String::from_utf8_lossy(&chunk));
        for event in events {
            process_sse_event(&mut state, &event)?;
        }
    }

    // Flush remaining data
    for event in parser.flush() {
        process_sse_event(&mut state, &event)?;
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

fn process_sse_event(state: &mut SseResponseState, event: &SseEvent) -> Result<(), Error> {
    match event.event_type.as_str() {
        "message_start" => {
            if let Ok(parsed) = serde_json::from_str::<MessageStartEvent>(&event.data) {
                state.usage.input_tokens = parsed.message.usage.input_tokens;
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
                        if let Some(ref mut text) = state.current_text {
                            text.push_str(&parsed.delta.text.unwrap_or_default());
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
    use crate::llm::types::Message;

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

    // --- build_request_body tests ---

    #[test]
    fn build_request_body_minimal() {
        let request = CompletionRequest {
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
        };

        let body = build_request_body("claude-sonnet-4-20250514", &request).unwrap();
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
        };

        let body = build_request_body("claude-sonnet-4-20250514", &request).unwrap();
        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["tools"][0]["name"], "search");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
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
        let response = parse_sse_stream(stream).await.unwrap();

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
        let response = parse_sse_stream(stream).await.unwrap();

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
        let response = parse_sse_stream(stream).await.unwrap();

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
        let response = parse_sse_stream(stream).await.unwrap();

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
        let response = parse_sse_stream(stream).await.unwrap();

        assert_eq!(response.text(), "OK");
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
            },
        };

        let response = into_completion_response(api);
        assert_eq!(response.content.len(), 2);
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 50);
    }
}
