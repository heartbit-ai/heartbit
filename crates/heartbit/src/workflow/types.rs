use serde::{Deserialize, Serialize};

use crate::llm::types::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, StopReason, TokenUsage,
    ToolChoice, ToolDefinition,
};

// Re-export DynLlmProvider from its canonical location in `llm` module.
pub use crate::llm::DynLlmProvider;

// ---------------------------------------------------------------------------
// Serializable request/response types for Restate services
// ---------------------------------------------------------------------------

/// Request for an LLM call activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallRequest {
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    /// Optional tool choice constraint for structured output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// Response from an LLM call activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
}

impl LlmCallResponse {
    /// Check if the response contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
    }

    /// Extract text from the response content.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

impl From<CompletionResponse> for LlmCallResponse {
    fn from(resp: CompletionResponse) -> Self {
        Self {
            content: resp.content,
            stop_reason: resp.stop_reason,
            usage: resp.usage,
        }
    }
}

impl LlmCallRequest {
    /// Convert to a `CompletionRequest` for the LLM provider.
    pub fn to_completion_request(&self) -> CompletionRequest {
        CompletionRequest {
            system: self.system.clone(),
            messages: self.messages.clone(),
            tools: self.tools.clone(),
            max_tokens: self.max_tokens,
            tool_choice: self.tool_choice.clone(),
        }
    }
}

/// Request for a tool call activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub tool_name: String,
    pub input: serde_json::Value,
    /// Optional timeout in seconds for the tool execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
    /// Maximum byte size for tool output. Oversized results are truncated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_bytes: Option<usize>,
}

/// Response from a tool call activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResponse {
    pub content: String,
    pub is_error: bool,
}

/// Input for an agent workflow run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub input: String,
    pub system_prompt: String,
    pub tool_defs: Vec<ToolDefinition>,
    pub max_turns: usize,
    pub max_tokens: u32,
    /// When true, tool execution requires human approval each round.
    #[serde(default)]
    pub approval_required: bool,
    /// Optional context window token limit. When set, old messages are trimmed
    /// (sliding window) to stay within this budget.
    #[serde(default)]
    pub context_window_tokens: Option<u32>,
    /// Token threshold for automatic summarization. When set, the agent will
    /// compress old messages via LLM summarization when context exceeds this limit.
    #[serde(default)]
    pub summarize_threshold: Option<u32>,
    /// Timeout in seconds for individual tool executions.
    #[serde(default)]
    pub tool_timeout_seconds: Option<u64>,
    /// Maximum byte size for individual tool output. Oversized results are
    /// truncated with a `[truncated]` suffix.
    #[serde(default)]
    pub max_tool_output_bytes: Option<usize>,
    /// Optional JSON Schema for structured output. When set, a synthetic
    /// `__respond__` tool is injected and the agent returns structured JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
}

/// Result from an agent workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub text: String,
    pub tokens: TokenUsage,
    pub tool_calls_made: usize,
    /// Structured output when the agent was configured with a `response_schema`.
    /// Contains the validated JSON conforming to the schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured: Option<serde_json::Value>,
}

/// Current status of an agent workflow (queryable via shared handler).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub current_turn: usize,
    pub max_turns: usize,
    pub state: String,
    /// Child agent workflow IDs (populated by orchestrator, empty for agents).
    #[serde(default)]
    pub child_workflows: Vec<String>,
}

/// Input for the orchestrator workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorTask {
    pub input: String,
    pub agents: Vec<AgentDef>,
    pub max_turns: usize,
    pub max_tokens: u32,
    /// When true, child agent workflows require human approval before tool execution.
    #[serde(default)]
    pub approval_required: bool,
}

/// Agent definition within an orchestrator task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDef {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub tool_defs: Vec<ToolDefinition>,
    /// Optional context window token limit for sliding window trimming.
    #[serde(default)]
    pub context_window_tokens: Option<u32>,
    /// Token threshold for automatic summarization.
    #[serde(default)]
    pub summarize_threshold: Option<u32>,
    /// Timeout in seconds for individual tool executions.
    #[serde(default)]
    pub tool_timeout_seconds: Option<u64>,
    /// Maximum byte size for individual tool output.
    #[serde(default)]
    pub max_tool_output_bytes: Option<usize>,
    /// Per-agent turn limit. Overrides orchestrator default when set.
    #[serde(default)]
    pub max_turns: Option<usize>,
    /// Per-agent token limit. Overrides orchestrator default when set.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Optional JSON Schema for structured output. When set, the agent
    /// returns structured JSON conforming to the schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
}

/// Result from the orchestrator workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorResult {
    pub text: String,
    pub tokens: TokenUsage,
}

/// Human decision for approval gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanDecision {
    pub approved: bool,
    pub reason: Option<String>,
    /// Explicit turn number to approve. When provided, the approval targets
    /// this specific turn's promise, avoiding a TOCTOU race from reading
    /// shared state. When `None`, falls back to reading `approval_turn` state.
    #[serde(default)]
    pub turn: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_call_request_roundtrips() {
        let req = LlmCallRequest {
            system: "You are helpful.".into(),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            max_tokens: 4096,
            tool_choice: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: LlmCallRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.system, "You are helpful.");
        assert_eq!(parsed.max_tokens, 4096);
    }

    #[test]
    fn llm_call_response_roundtrips() {
        let resp = LlmCallResponse {
            content: vec![ContentBlock::Text {
                text: "Hello!".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: LlmCallResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.text(), "Hello!");
        assert!(!parsed.has_tool_calls());
    }

    #[test]
    fn llm_call_response_detects_tool_calls() {
        let resp = LlmCallResponse {
            content: vec![ContentBlock::ToolUse {
                id: "tc_1".into(),
                name: "search".into(),
                input: serde_json::json!({"q": "rust"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::default(),
        };
        assert!(resp.has_tool_calls());
    }

    #[test]
    fn llm_call_request_to_completion_request() {
        let req = LlmCallRequest {
            system: "sys".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 2048,
            tool_choice: None,
        };
        let cr = req.to_completion_request();
        assert_eq!(cr.system, "sys");
        assert_eq!(cr.max_tokens, 2048);
    }

    #[test]
    fn llm_call_request_tool_choice_roundtrips() {
        let req = LlmCallRequest {
            system: "sys".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 2048,
            tool_choice: Some(ToolChoice::Tool {
                name: "search".into(),
            }),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: LlmCallRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.tool_choice,
            Some(ToolChoice::Tool {
                name: "search".into()
            })
        );
    }

    #[test]
    fn llm_call_request_tool_choice_defaults_to_none() {
        // Backward compatibility: old JSON without tool_choice still parses
        let json = r#"{"system":"sys","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[],"max_tokens":2048}"#;
        let parsed: LlmCallRequest = serde_json::from_str(json).unwrap();
        assert!(parsed.tool_choice.is_none());
    }

    #[test]
    fn llm_call_request_forwards_tool_choice_to_completion_request() {
        let req = LlmCallRequest {
            system: "sys".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 2048,
            tool_choice: Some(ToolChoice::Any),
        };
        let cr = req.to_completion_request();
        assert_eq!(cr.tool_choice, Some(ToolChoice::Any));
    }

    #[test]
    fn from_completion_response() {
        let cr = CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Done".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 5,
                output_tokens: 3,
            },
        };
        let resp: LlmCallResponse = cr.into();
        assert_eq!(resp.text(), "Done");
        assert_eq!(resp.usage.input_tokens, 5);
    }

    #[test]
    fn tool_call_request_roundtrips() {
        let req = ToolCallRequest {
            tool_name: "search".into(),
            input: serde_json::json!({"query": "rust"}),
            timeout_seconds: None,
            max_output_bytes: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: ToolCallRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tool_name, "search");
    }

    #[test]
    fn tool_call_request_max_output_bytes_roundtrips() {
        let req = ToolCallRequest {
            tool_name: "search".into(),
            input: serde_json::json!({}),
            timeout_seconds: Some(30),
            max_output_bytes: Some(4096),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: ToolCallRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.max_output_bytes, Some(4096));
    }

    #[test]
    fn tool_call_request_max_output_bytes_defaults_to_none() {
        // Backward compatibility: old JSON without max_output_bytes still parses
        let json = r#"{"tool_name":"search","input":{}}"#;
        let parsed: ToolCallRequest = serde_json::from_str(json).unwrap();
        assert!(parsed.max_output_bytes.is_none());
        assert!(parsed.timeout_seconds.is_none());
    }

    #[test]
    fn tool_call_response_roundtrips() {
        let resp = ToolCallResponse {
            content: "result text".into(),
            is_error: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: ToolCallResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content, "result text");
        assert!(!parsed.is_error);
    }

    #[test]
    fn agent_task_roundtrips() {
        let task = AgentTask {
            input: "Research Rust".into(),
            system_prompt: "You are a researcher.".into(),
            tool_defs: vec![],
            max_turns: 10,
            max_tokens: 4096,
            approval_required: false,
            context_window_tokens: None,
            summarize_threshold: None,
            tool_timeout_seconds: None,
            max_tool_output_bytes: None,
            response_schema: None,
        };
        let json = serde_json::to_string(&task).unwrap();
        let parsed: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.input, "Research Rust");
        assert_eq!(parsed.max_turns, 10);
        assert!(!parsed.approval_required);
    }

    #[test]
    fn agent_task_approval_required_defaults_false() {
        // Verify serde(default) works: omitting approval_required defaults to false
        let json =
            r#"{"input":"x","system_prompt":"y","tool_defs":[],"max_turns":5,"max_tokens":1024}"#;
        let parsed: AgentTask = serde_json::from_str(json).unwrap();
        assert!(!parsed.approval_required);
    }

    #[test]
    fn agent_result_roundtrips() {
        let result = AgentResult {
            text: "Rust is great".into(),
            tokens: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
            tool_calls_made: 3,
            structured: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.text, "Rust is great");
        assert_eq!(parsed.tool_calls_made, 3);
        assert!(parsed.structured.is_none());
    }

    #[test]
    fn orchestrator_task_roundtrips() {
        let task = OrchestratorTask {
            input: "Analyze data".into(),
            agents: vec![AgentDef {
                name: "researcher".into(),
                description: "Research specialist".into(),
                system_prompt: "You research.".into(),
                tool_defs: vec![],
                context_window_tokens: None,
                summarize_threshold: None,
                tool_timeout_seconds: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: None,
                response_schema: None,
            }],
            max_turns: 10,
            max_tokens: 8192,
            approval_required: true,
        };
        let json = serde_json::to_string(&task).unwrap();
        let parsed: OrchestratorTask = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.agents.len(), 1);
        assert_eq!(parsed.agents[0].name, "researcher");
        assert!(parsed.approval_required);
    }

    #[test]
    fn orchestrator_task_approval_required_defaults_false() {
        let json = r#"{"input":"x","agents":[],"max_turns":5,"max_tokens":1024}"#;
        let parsed: OrchestratorTask = serde_json::from_str(json).unwrap();
        assert!(!parsed.approval_required);
    }

    #[test]
    fn human_decision_roundtrips() {
        let decision = HumanDecision {
            approved: true,
            reason: Some("looks good".into()),
            turn: Some(3),
        };
        let json = serde_json::to_string(&decision).unwrap();
        let parsed: HumanDecision = serde_json::from_str(&json).unwrap();
        assert!(parsed.approved);
        assert_eq!(parsed.reason.unwrap(), "looks good");
        assert_eq!(parsed.turn, Some(3));
    }

    #[test]
    fn human_decision_turn_defaults_to_none() {
        let json = r#"{"approved":true,"reason":null}"#;
        let parsed: HumanDecision = serde_json::from_str(json).unwrap();
        assert!(parsed.approved);
        assert!(parsed.turn.is_none());
    }

    #[test]
    fn agent_result_structured_roundtrips() {
        let structured = serde_json::json!({"answer": "42"});
        let result = AgentResult {
            text: "{\"answer\": \"42\"}".into(),
            tokens: TokenUsage::default(),
            tool_calls_made: 1,
            structured: Some(structured.clone()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.structured, Some(structured));
    }

    #[test]
    fn agent_result_structured_defaults_to_none() {
        // Backward compatibility: old JSON without structured field still parses
        let json =
            r#"{"text":"ok","tokens":{"input_tokens":0,"output_tokens":0},"tool_calls_made":0}"#;
        let parsed: AgentResult = serde_json::from_str(json).unwrap();
        assert!(parsed.structured.is_none());
    }

    #[test]
    fn agent_task_response_schema_roundtrips() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        });
        let task = AgentTask {
            input: "test".into(),
            system_prompt: "sys".into(),
            tool_defs: vec![],
            max_turns: 5,
            max_tokens: 1024,
            approval_required: false,
            context_window_tokens: None,
            summarize_threshold: None,
            tool_timeout_seconds: None,
            max_tool_output_bytes: None,
            response_schema: Some(schema.clone()),
        };
        let json = serde_json::to_string(&task).unwrap();
        let parsed: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.response_schema, Some(schema));
    }

    #[test]
    fn agent_task_response_schema_defaults_to_none() {
        let json =
            r#"{"input":"x","system_prompt":"y","tool_defs":[],"max_turns":5,"max_tokens":1024}"#;
        let parsed: AgentTask = serde_json::from_str(json).unwrap();
        assert!(parsed.response_schema.is_none());
    }

    #[test]
    fn agent_def_response_schema_roundtrips() {
        let schema = serde_json::json!({"type": "object"});
        let def = AgentDef {
            name: "test".into(),
            description: "Test".into(),
            system_prompt: "sys".into(),
            tool_defs: vec![],
            context_window_tokens: None,
            summarize_threshold: None,
            tool_timeout_seconds: None,
            max_tool_output_bytes: None,
            max_turns: None,
            max_tokens: None,
            response_schema: Some(schema.clone()),
        };
        let json = serde_json::to_string(&def).unwrap();
        let parsed: AgentDef = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.response_schema, Some(schema));
    }

    // DynLlmProvider tests are in llm/mod.rs (canonical location)
}
