//! OpenAI-compatible `/v1/chat/completions` endpoint types and conversions.
//!
//! Allows any client speaking the OpenAI protocol (e.g., `openai` Python/Node
//! SDKs) to talk to heartbit agents. The `model` field maps to an agent name.

use serde::{Deserialize, Serialize};

/// Current Unix timestamp in seconds (0 on clock error).
fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// OpenAI-format chat completion request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Agent name (mapped from OpenAI's `model` field).
    pub model: String,
    /// Conversation messages in OpenAI format.
    pub messages: Vec<OaiMessage>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,
    /// Maximum tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Temperature (accepted but not forwarded — agents use their own config).
    #[serde(default)]
    pub temperature: Option<f64>,
}

/// An OpenAI-format message.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OaiMessage {
    pub role: String,
    #[serde(default)]
    pub content: OaiContent,
}

/// OpenAI message content — can be a string, array of parts, or null.
#[derive(Debug, Default, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum OaiContent {
    Text(String),
    Parts(Vec<OaiContentPart>),
    #[default]
    Null,
}

impl OaiContent {
    /// Extract the text content, joining parts if necessary.
    pub fn as_text(&self) -> String {
        match self {
            OaiContent::Text(s) => s.clone(),
            OaiContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    OaiContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            OaiContent::Null => String::new(),
        }
    }
}

/// A content part in the OpenAI multipart format.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum OaiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OaiImageUrl },
}

/// Image URL reference.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OaiImageUrl {
    pub url: String,
}

/// OpenAI-format chat completion response (non-streaming).
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OaiChoice>,
    pub usage: OaiUsage,
}

/// A single choice in the response.
#[derive(Debug, Serialize)]
pub struct OaiChoice {
    pub index: u32,
    pub message: OaiResponseMessage,
    pub finish_reason: String,
}

/// OpenAI-format tool call in a response.
#[derive(Debug, Serialize, Clone)]
pub struct OaiToolCall {
    pub index: u32,
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: &'static str,
    pub function: OaiFunction,
}

/// Function call details.
#[derive(Debug, Serialize, Clone)]
pub struct OaiFunction {
    pub name: String,
    pub arguments: String,
}

/// Response message.
#[derive(Debug, Serialize)]
pub struct OaiResponseMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
}

/// Token usage info.
#[derive(Debug, Serialize)]
pub struct OaiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// A single SSE chunk for streaming responses.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OaiStreamChoice>,
}

/// A streaming choice delta.
#[derive(Debug, Serialize)]
pub struct OaiStreamChoice {
    pub index: u32,
    pub delta: OaiDelta,
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Serialize)]
pub struct OaiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
}

/// Extract the last user message text from an OpenAI-format request.
///
/// This is the "task" text sent to the heartbit agent.
pub fn extract_task(req: &ChatCompletionRequest) -> String {
    // Find the last user message
    for msg in req.messages.iter().rev() {
        if msg.role == "user" {
            return msg.content.as_text();
        }
    }
    // Fallback: concatenate all messages
    req.messages
        .iter()
        .map(|m| m.content.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract a system prompt from the messages (first system message, if any).
pub fn extract_system_prompt(req: &ChatCompletionRequest) -> Option<String> {
    req.messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_text())
        .filter(|s| !s.is_empty())
}

/// Build a non-streaming response from agent output.
pub fn build_response(
    model: &str,
    text: &str,
    input_tokens: u32,
    output_tokens: u32,
) -> ChatCompletionResponse {
    let now = unix_timestamp();

    ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion",
        created: now,
        model: model.to_string(),
        choices: vec![OaiChoice {
            index: 0,
            message: OaiResponseMessage {
                role: "assistant".into(),
                content: Some(text.to_string()),
                tool_calls: None,
            },
            finish_reason: "stop".into(),
        }],
        usage: OaiUsage {
            prompt_tokens: input_tokens,
            completion_tokens: output_tokens,
            total_tokens: input_tokens + output_tokens,
        },
    }
}

/// Build a streaming text delta chunk.
pub fn build_text_chunk(model: &str, id: &str, text: &str) -> ChatCompletionChunk {
    let now = unix_timestamp();

    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created: now,
        model: model.to_string(),
        choices: vec![OaiStreamChoice {
            index: 0,
            delta: OaiDelta {
                role: None,
                content: Some(text.to_string()),
                tool_calls: None,
            },
            finish_reason: None,
        }],
    }
}

/// Build the initial streaming chunk with role.
pub fn build_role_chunk(model: &str, id: &str) -> ChatCompletionChunk {
    let now = unix_timestamp();

    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created: now,
        model: model.to_string(),
        choices: vec![OaiStreamChoice {
            index: 0,
            delta: OaiDelta {
                role: Some("assistant".into()),
                content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
    }
}

/// Build the final streaming chunk with finish_reason.
pub fn build_done_chunk(model: &str, id: &str) -> ChatCompletionChunk {
    let now = unix_timestamp();

    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created: now,
        model: model.to_string(),
        choices: vec![OaiStreamChoice {
            index: 0,
            delta: OaiDelta {
                role: None,
                content: None,
                tool_calls: None,
            },
            finish_reason: Some("stop".into()),
        }],
    }
}

/// Build a non-streaming response that includes tool calls.
pub fn build_response_with_tools(
    model: &str,
    text: Option<&str>,
    tool_calls: Vec<OaiToolCall>,
    input_tokens: u32,
    output_tokens: u32,
) -> ChatCompletionResponse {
    let has_tools = !tool_calls.is_empty();
    ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion",
        created: unix_timestamp(),
        model: model.to_string(),
        choices: vec![OaiChoice {
            index: 0,
            message: OaiResponseMessage {
                role: "assistant".into(),
                content: text.map(|s| s.to_string()),
                tool_calls: if has_tools { Some(tool_calls) } else { None },
            },
            finish_reason: if has_tools { "tool_calls" } else { "stop" }.into(),
        }],
        usage: OaiUsage {
            prompt_tokens: input_tokens,
            completion_tokens: output_tokens,
            total_tokens: input_tokens + output_tokens,
        },
    }
}

/// Model listing response for `GET /v1/models`.
#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

/// A single model entry.
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

/// Build a model list response from agent names.
pub fn build_model_list(agent_names: &[String]) -> ModelListResponse {
    let now = unix_timestamp();

    ModelListResponse {
        object: "list",
        data: agent_names
            .iter()
            .map(|name| ModelInfo {
                id: name.clone(),
                object: "model",
                created: now,
                owned_by: "heartbit".into(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserialize_simple_request() {
        let json = json!({
            "model": "assistant",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });
        let req: ChatCompletionRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.model, "assistant");
        assert_eq!(req.messages.len(), 1);
        assert!(!req.stream);
    }

    #[test]
    fn deserialize_streaming_request() {
        let json = json!({
            "model": "agent-1",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 500
        });
        let req: ChatCompletionRequest = serde_json::from_value(json).unwrap();
        assert!(req.stream);
        assert_eq!(req.max_tokens, Some(500));
    }

    #[test]
    fn deserialize_multipart_content() {
        let json = json!({
            "model": "assistant",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
                ]
            }]
        });
        let req: ChatCompletionRequest = serde_json::from_value(json).unwrap();
        let text = req.messages[0].content.as_text();
        assert_eq!(text, "What's this?");
    }

    #[test]
    fn deserialize_null_content() {
        let json = json!({
            "model": "assistant",
            "messages": [{"role": "assistant", "content": null}]
        });
        let req: ChatCompletionRequest = serde_json::from_value(json).unwrap();
        assert!(req.messages[0].content.as_text().is_empty());
    }

    #[test]
    fn extract_task_last_user_message() {
        let req = ChatCompletionRequest {
            model: "agent".into(),
            messages: vec![
                OaiMessage {
                    role: "system".into(),
                    content: OaiContent::Text("sys".into()),
                },
                OaiMessage {
                    role: "user".into(),
                    content: OaiContent::Text("first".into()),
                },
                OaiMessage {
                    role: "assistant".into(),
                    content: OaiContent::Text("reply".into()),
                },
                OaiMessage {
                    role: "user".into(),
                    content: OaiContent::Text("second".into()),
                },
            ],
            stream: false,
            max_tokens: None,
            temperature: None,
        };
        assert_eq!(extract_task(&req), "second");
    }

    #[test]
    fn extract_system_prompt_found() {
        let req = ChatCompletionRequest {
            model: "agent".into(),
            messages: vec![
                OaiMessage {
                    role: "system".into(),
                    content: OaiContent::Text("Be helpful".into()),
                },
                OaiMessage {
                    role: "user".into(),
                    content: OaiContent::Text("hi".into()),
                },
            ],
            stream: false,
            max_tokens: None,
            temperature: None,
        };
        assert_eq!(extract_system_prompt(&req), Some("Be helpful".into()));
    }

    #[test]
    fn extract_system_prompt_none() {
        let req = ChatCompletionRequest {
            model: "agent".into(),
            messages: vec![OaiMessage {
                role: "user".into(),
                content: OaiContent::Text("hi".into()),
            }],
            stream: false,
            max_tokens: None,
            temperature: None,
        };
        assert!(extract_system_prompt(&req).is_none());
    }

    #[test]
    fn build_response_correct_format() {
        let resp = build_response("agent-1", "Hello!", 10, 5);
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.model, "agent-1");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.choices[0].finish_reason, "stop");
        assert_eq!(resp.usage.total_tokens, 15);
    }

    #[test]
    fn build_text_chunk_format() {
        let chunk = build_text_chunk("agent", "id-1", "Hello");
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn build_role_chunk_format() {
        let chunk = build_role_chunk("agent", "id-1");
        assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn build_done_chunk_format() {
        let chunk = build_done_chunk("agent", "id-1");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn build_model_list_format() {
        let names = vec!["agent-1".into(), "agent-2".into()];
        let list = build_model_list(&names);
        assert_eq!(list.object, "list");
        assert_eq!(list.data.len(), 2);
        assert_eq!(list.data[0].id, "agent-1");
        assert_eq!(list.data[0].owned_by, "heartbit");
    }

    #[test]
    fn response_serializes_to_valid_json() {
        let resp = build_response("test", "Hi", 5, 3);
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert_eq!(json["object"], "chat.completion");
    }

    #[test]
    fn chunk_serializes_to_valid_json() {
        let chunk = build_text_chunk("test", "id-1", "chunk");
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["object"], "chat.completion.chunk");
    }

    #[test]
    fn response_with_tools_format() {
        let tools = vec![OaiToolCall {
            index: 0,
            id: "call_1".into(),
            call_type: "function",
            function: OaiFunction {
                name: "search".into(),
                arguments: r#"{"q":"rust"}"#.into(),
            },
        }];
        let resp = build_response_with_tools("agent", Some("Let me search"), tools, 10, 5);
        assert_eq!(resp.choices[0].finish_reason, "tool_calls");
        let msg = &resp.choices[0].message;
        assert!(msg.tool_calls.is_some());
        let tc = &msg.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.function.name, "search");
    }

    #[test]
    fn response_without_tools_omits_field() {
        let resp = build_response("agent", "Hello", 10, 5);
        let json = serde_json::to_value(&resp).unwrap();
        // tool_calls should not be present in JSON due to skip_serializing_if
        assert!(json["choices"][0]["message"].get("tool_calls").is_none());
    }

    #[test]
    fn streaming_delta_without_tools_omits_field() {
        let chunk = build_text_chunk("agent", "id", "hello");
        let json = serde_json::to_value(&chunk).unwrap();
        assert!(json["choices"][0]["delta"].get("tool_calls").is_none());
    }
}
