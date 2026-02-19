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
}

/// Why the LLM stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A response from the LLM.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
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
            .collect::<Vec<_>>()
            .join("")
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
        };

        assert_eq!(response.text(), "Hello world");
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
}
