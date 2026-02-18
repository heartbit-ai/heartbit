use bytes::Bytes;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use tracing::debug;

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
        let body = build_request_body(&self.model, &request);

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
            let message = response.text().await.unwrap_or_default();
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
impl AnthropicProvider {
    pub async fn stream(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let mut body = build_request_body(&self.model, &request);
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
            let message = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message,
            });
        }

        let byte_stream = response.bytes_stream();
        parse_sse_stream(byte_stream).await
    }
}

fn build_request_body(model: &str, request: &CompletionRequest) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": request.max_tokens,
        "messages": request.messages,
    });

    if !request.system.is_empty() {
        body["system"] = serde_json::Value::String(request.system.clone());
    }

    if !request.tools.is_empty() {
        body["tools"] = serde_json::to_value(&request.tools).unwrap_or_default();
    }

    body
}

// --- SSE Parser ---

/// Parse an SSE byte stream into a CompletionResponse.
///
/// Handles the Anthropic streaming format:
/// - `message_start`: contains usage info
/// - `content_block_start`: begins a content block
/// - `content_block_delta`: appends to current block
/// - `message_delta`: contains stop_reason and final usage
/// - `message_stop`: end of message
pub(crate) async fn parse_sse_stream<S>(stream: S) -> Result<CompletionResponse, Error>
where
    S: futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    let mut state = SseParseState::default();
    let mut buffer = String::new();

    tokio::pin!(stream);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(Error::Http)?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE events (separated by double newlines)
        while let Some(event_end) = buffer.find("\n\n") {
            let event_text = buffer[..event_end].to_string();
            buffer = buffer[event_end + 2..].to_string();

            if let Some(event) = parse_sse_event(&event_text) {
                process_sse_event(&mut state, &event)?;
            }
        }
    }

    Ok(state.into_response())
}

#[derive(Default)]
struct SseParseState {
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

impl SseParseState {
    fn flush_current_block(&mut self) {
        if let Some(text) = self.current_text.take() {
            self.content.push(ContentBlock::Text { text });
        }
        if let Some(tool) = self.current_tool_use.take() {
            let input = serde_json::from_str(&tool.input_json).unwrap_or_default();
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

struct SseEvent {
    event_type: String,
    data: String,
}

fn parse_sse_event(raw: &str) -> Option<SseEvent> {
    let mut event_type = String::new();
    let mut data = String::new();

    for line in raw.lines() {
        if let Some(value) = line.strip_prefix("event: ") {
            event_type = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("data: ") {
            data = value.to_string();
        }
    }

    if event_type.is_empty() {
        return None;
    }

    Some(SseEvent { event_type, data })
}

fn process_sse_event(state: &mut SseParseState, event: &SseEvent) -> Result<(), Error> {
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
                state.stop_reason = parsed.delta.stop_reason.as_deref().map(|s| match s {
                    "end_turn" => StopReason::EndTurn,
                    "tool_use" => StopReason::ToolUse,
                    "max_tokens" => StopReason::MaxTokens,
                    _ => StopReason::EndTurn,
                });
                if let Some(usage) = parsed.usage {
                    state.usage.output_tokens = usage.output_tokens;
                }
            }
        }
        "ping" | "message_stop" => {}
        other => {
            debug!(event_type = other, "unknown SSE event type");
        }
    }

    Ok(())
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

    let stop_reason = match api.stop_reason.as_str() {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    };

    CompletionResponse {
        content,
        stop_reason,
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

    #[test]
    fn build_request_body_minimal() {
        let request = CompletionRequest {
            model: "ignored".into(), // model comes from provider
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
        };

        let body = build_request_body("claude-sonnet-4-20250514", &request);
        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["max_tokens"], 1024);
        assert!(body.get("system").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn build_request_body_with_system_and_tools() {
        use crate::llm::types::{Message, ToolDefinition};
        use serde_json::json;

        let request = CompletionRequest {
            model: String::new(),
            system: "You are helpful.".into(),
            messages: vec![Message::user("hi")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search the web".into(),
                input_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            }],
            max_tokens: 2048,
        };

        let body = build_request_body("claude-sonnet-4-20250514", &request);
        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["tools"][0]["name"], "search");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn parse_sse_event_basic() {
        let raw = "event: message_start\ndata: {\"message\":{}}";
        let event = parse_sse_event(raw).unwrap();
        assert_eq!(event.event_type, "message_start");
        assert_eq!(event.data, "{\"message\":{}}");
    }

    #[test]
    fn parse_sse_event_ignores_empty() {
        let raw = "data: something";
        assert!(parse_sse_event(raw).is_none());
    }

    #[tokio::test]
    async fn parse_sse_stream_text_response() {
        let sse_data = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello \"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world!\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse_data))]);
        let response = parse_sse_stream(stream).await.unwrap();

        assert_eq!(response.text(), "Hello world!");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[tokio::test]
    async fn parse_sse_stream_tool_use_response() {
        let sse_data = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me search.\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_123\",\"name\":\"search\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\": \"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"rust\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":15}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

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
        let sse_data = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_a\",\"name\":\"search\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\": \\\"a\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_b\",\"name\":\"read_file\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\": \\\"/tmp\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":10}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let stream = futures::stream::iter(vec![Ok(Bytes::from(sse_data))]);
        let response = parse_sse_stream(stream).await.unwrap();

        let calls = response.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].name, "read_file");
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
