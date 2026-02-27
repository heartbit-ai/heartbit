use serde::{Deserialize, Serialize};

/// Name of the synthetic tool used for structured output.
pub(crate) const RESPOND_TOOL_NAME: &str = "__respond__";

/// Description for the synthetic `__respond__` tool.
pub(crate) const RESPOND_TOOL_DESCRIPTION: &str = "Produce your final structured response. Call this tool when you \
     have gathered all necessary information and are ready to return \
     your answer in the required format.";

/// Role in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A block of content within a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    Image {
        media_type: String,
        data: String,
    },
    Audio {
        format: String,
        data: String,
    },
}

/// A message in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn user_with_content(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }

    pub fn tool_results(results: Vec<ToolResult>) -> Self {
        Self {
            role: Role::User,
            content: results
                .into_iter()
                .map(|r| ContentBlock::ToolResult {
                    tool_use_id: r.tool_use_id,
                    content: r.content,
                    is_error: r.is_error,
                })
                .collect(),
        }
    }
}

/// Definition of a tool the LLM can call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Controls which tools the LLM is allowed or forced to call.
///
/// Maps to Anthropic's `tool_choice` parameter and OpenAI's equivalent.
/// When `None` is used in `CompletionRequest`, the provider's default behavior
/// applies (equivalent to `Auto`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the LLM decide whether to call tools. This is the default.
    Auto,
    /// Force the LLM to call at least one tool (any tool).
    Any,
    /// Force the LLM to call a specific tool by name.
    Tool { name: String },
}

/// Controls reasoning/thinking effort for models that support it.
/// Maps to OpenRouter's `reasoning` parameter and Anthropic's extended thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Maximum reasoning depth.
    High,
    /// Default reasoning depth.
    Medium,
    /// Minimal reasoning.
    Low,
    /// Disable reasoning entirely (fastest, cheapest).
    None,
}

/// A request to the LLM.
///
/// The model is not part of the request — it's a property of the provider.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    /// Optional tool choice constraint. `None` = provider default (auto).
    pub tool_choice: Option<ToolChoice>,
    /// Optional reasoning/thinking effort level. `None` = no reasoning.
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// Why the LLM stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

/// Token usage statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens used to create a new cache entry (Anthropic prompt caching).
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Tokens read from an existing cache entry (Anthropic prompt caching).
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    /// Tokens consumed by the model's internal reasoning/thinking.
    #[serde(default)]
    pub reasoning_tokens: u32,
}

impl TokenUsage {
    /// Total tokens consumed (input + output) as `u64`.
    pub fn total(&self) -> u64 {
        self.input_tokens as u64 + self.output_tokens as u64
    }
}

impl std::ops::AddAssign for TokenUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens += rhs.input_tokens;
        self.output_tokens += rhs.output_tokens;
        self.cache_creation_input_tokens += rhs.cache_creation_input_tokens;
        self.cache_read_input_tokens += rhs.cache_read_input_tokens;
        self.reasoning_tokens += rhs.reasoning_tokens;
    }
}

/// A response from the LLM.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
    /// Model that produced this response. Set by cascading/routing providers.
    /// `None` when the model is known statically from the provider.
    pub model: Option<String>,
}

impl CompletionResponse {
    /// Extract tool calls from the response content blocks.
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, name, input } => Some(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    /// Extract text from the response content blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
}

/// A tool call extracted from a response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn success(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn message_user_creates_text_content() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 1);
        assert_eq!(
            msg.content[0],
            ContentBlock::Text {
                text: "hello".into()
            }
        );
    }

    #[test]
    fn message_assistant_creates_text_content() {
        let msg = Message::assistant("response");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(
            msg.content[0],
            ContentBlock::Text {
                text: "response".into()
            }
        );
    }

    #[test]
    fn message_tool_results_creates_tool_result_blocks() {
        let msg = Message::tool_results(vec![
            ToolResult::success("call-1", "result 1"),
            ToolResult::error("call-2", "failed"),
        ]);
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 2);
        assert_eq!(
            msg.content[0],
            ContentBlock::ToolResult {
                tool_use_id: "call-1".into(),
                content: "result 1".into(),
                is_error: false,
            }
        );
        assert_eq!(
            msg.content[1],
            ContentBlock::ToolResult {
                tool_use_id: "call-2".into(),
                content: "failed".into(),
                is_error: true,
            }
        );
    }

    #[test]
    fn completion_response_extracts_tool_calls() {
        let response = CompletionResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Let me help.".into(),
                },
                ContentBlock::ToolUse {
                    id: "call-1".into(),
                    name: "search".into(),
                    input: json!({"query": "rust"}),
                },
                ContentBlock::ToolUse {
                    id: "call-2".into(),
                    name: "read_file".into(),
                    input: json!({"path": "/tmp/test"}),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
            model: None,
        };

        let calls = response.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].name, "read_file");
    }

    #[test]
    fn completion_response_extracts_text() {
        let response = CompletionResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Hello ".into(),
                },
                ContentBlock::ToolUse {
                    id: "x".into(),
                    name: "t".into(),
                    input: json!({}),
                },
                ContentBlock::Text {
                    text: "world".into(),
                },
            ],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        };

        assert_eq!(response.text(), "Hello world");
    }

    #[test]
    fn content_block_image_roundtrip() {
        let block = ContentBlock::Image {
            media_type: "image/jpeg".into(),
            data: "base64data".into(),
        };
        let json_str = serde_json::to_string(&block).unwrap();
        let roundtripped: ContentBlock = serde_json::from_str(&json_str).unwrap();
        assert_eq!(block, roundtripped);

        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "image");
        assert_eq!(json["media_type"], "image/jpeg");
        assert_eq!(json["data"], "base64data");
    }

    #[test]
    fn message_user_with_content_creates_mixed_blocks() {
        let msg = Message::user_with_content(vec![
            ContentBlock::Text {
                text: "describe this image".into(),
            },
            ContentBlock::Image {
                media_type: "image/png".into(),
                data: "base64data".into(),
            },
        ]);
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 2);
        assert!(
            matches!(&msg.content[0], ContentBlock::Text { text } if text == "describe this image")
        );
        assert!(
            matches!(&msg.content[1], ContentBlock::Image { media_type, .. } if media_type == "image/png")
        );
    }

    #[test]
    fn content_block_serializes_with_type_tag() {
        let block = ContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hello");
    }

    #[test]
    fn content_block_tool_use_roundtrips() {
        let block = ContentBlock::ToolUse {
            id: "id-1".into(),
            name: "search".into(),
            input: json!({"q": "test"}),
        };
        let json_str = serde_json::to_string(&block).unwrap();
        let roundtripped: ContentBlock = serde_json::from_str(&json_str).unwrap();
        assert_eq!(block, roundtripped);
    }

    #[test]
    fn role_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
    }

    #[test]
    fn stop_reason_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&StopReason::EndTurn).unwrap(),
            "\"end_turn\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::ToolUse).unwrap(),
            "\"tool_use\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::MaxTokens).unwrap(),
            "\"max_tokens\""
        );
    }

    #[test]
    fn tool_choice_auto_serializes() {
        let tc = ToolChoice::Auto;
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "auto");
    }

    #[test]
    fn tool_choice_any_serializes() {
        let tc = ToolChoice::Any;
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "any");
    }

    #[test]
    fn tool_choice_tool_serializes() {
        let tc = ToolChoice::Tool {
            name: "search".into(),
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "tool");
        assert_eq!(json["name"], "search");
    }

    #[test]
    fn tool_choice_roundtrips() {
        let choices = vec![
            ToolChoice::Auto,
            ToolChoice::Any,
            ToolChoice::Tool {
                name: "search".into(),
            },
        ];
        for tc in choices {
            let json = serde_json::to_string(&tc).unwrap();
            let parsed: ToolChoice = serde_json::from_str(&json).unwrap();
            assert_eq!(tc, parsed);
        }
    }

    #[test]
    fn tool_result_success_and_error() {
        let ok = ToolResult::success("id", "done");
        assert!(!ok.is_error);
        assert_eq!(ok.content, "done");

        let err = ToolResult::error("id", "failed");
        assert!(err.is_error);
        assert_eq!(err.content, "failed");
    }

    #[test]
    fn tool_call_roundtrips() {
        let tc = ToolCall {
            id: "call-1".into(),
            name: "search".into(),
            input: json!({"query": "rust"}),
        };
        let json_str = serde_json::to_string(&tc).unwrap();
        let parsed: ToolCall = serde_json::from_str(&json_str).unwrap();
        assert_eq!(tc, parsed);
    }

    #[test]
    fn token_usage_cache_fields_default_to_zero() {
        let usage = TokenUsage::default();
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
    }

    #[test]
    fn token_usage_cache_fields_roundtrip() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 200,
            cache_read_input_tokens: 300,
            reasoning_tokens: 400,
        };
        let json_str = serde_json::to_string(&usage).unwrap();
        let parsed: TokenUsage = serde_json::from_str(&json_str).unwrap();
        assert_eq!(usage, parsed);
    }

    #[test]
    fn token_usage_add_assign() {
        let mut a = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 10,
            cache_read_input_tokens: 20,
            reasoning_tokens: 30,
        };
        let b = TokenUsage {
            input_tokens: 200,
            output_tokens: 30,
            cache_creation_input_tokens: 5,
            cache_read_input_tokens: 15,
            reasoning_tokens: 25,
        };
        a += b;
        assert_eq!(a.input_tokens, 300);
        assert_eq!(a.output_tokens, 80);
        assert_eq!(a.cache_creation_input_tokens, 15);
        assert_eq!(a.cache_read_input_tokens, 35);
        assert_eq!(a.reasoning_tokens, 55);
    }

    #[test]
    fn token_usage_backward_compat_deserialization() {
        // Old format without cache/reasoning fields should still deserialize
        let json_str = r#"{"input_tokens":100,"output_tokens":50}"#;
        let parsed: TokenUsage = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.input_tokens, 100);
        assert_eq!(parsed.output_tokens, 50);
        assert_eq!(parsed.cache_creation_input_tokens, 0);
        assert_eq!(parsed.cache_read_input_tokens, 0);
        assert_eq!(parsed.reasoning_tokens, 0);
    }

    #[test]
    fn reasoning_effort_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::High).unwrap(),
            "\"high\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Medium).unwrap(),
            "\"medium\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Low).unwrap(),
            "\"low\""
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::None).unwrap(),
            "\"none\""
        );
    }

    #[test]
    fn reasoning_effort_roundtrips() {
        let efforts = vec![
            ReasoningEffort::High,
            ReasoningEffort::Medium,
            ReasoningEffort::Low,
            ReasoningEffort::None,
        ];
        for effort in efforts {
            let json = serde_json::to_string(&effort).unwrap();
            let parsed: ReasoningEffort = serde_json::from_str(&json).unwrap();
            assert_eq!(effort, parsed);
        }
    }

    #[test]
    fn content_block_audio_roundtrip() {
        let block = ContentBlock::Audio {
            format: "ogg".into(),
            data: "base64audiodata".into(),
        };
        let json_str = serde_json::to_string(&block).unwrap();
        let roundtripped: ContentBlock = serde_json::from_str(&json_str).unwrap();
        assert_eq!(block, roundtripped);

        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "audio");
        assert_eq!(json["format"], "ogg");
        assert_eq!(json["data"], "base64audiodata");
    }

    #[test]
    fn response_model_field_defaults_to_none() {
        let response = CompletionResponse {
            content: vec![ContentBlock::Text { text: "hi".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        };
        assert!(response.model.is_none());
    }

    #[test]
    fn response_model_roundtrip_with_value() {
        let response = CompletionResponse {
            content: vec![ContentBlock::Text { text: "hi".into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: Some("anthropic/claude-3.5-haiku".into()),
        };
        assert_eq!(
            response.model.as_deref(),
            Some("anthropic/claude-3.5-haiku")
        );
    }

    #[test]
    fn token_usage_total() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        assert_eq!(usage.total(), 150);
    }

    #[test]
    fn token_usage_total_no_overflow() {
        let usage = TokenUsage {
            input_tokens: u32::MAX,
            output_tokens: u32::MAX,
            ..Default::default()
        };
        assert_eq!(usage.total(), u32::MAX as u64 + u32::MAX as u64);
    }
}
