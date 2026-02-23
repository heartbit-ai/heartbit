use serde::{Deserialize, Serialize};

use crate::llm::types::{StopReason, TokenUsage};
use crate::tool::builtins::floor_char_boundary;

/// Maximum byte size for event payload strings (LLM text, tool I/O).
/// Payloads exceeding this are truncated with a `[truncated: N bytes omitted]` suffix.
pub(crate) const EVENT_MAX_PAYLOAD_BYTES: usize = 4096;

/// Truncate a string for event payloads. Short strings (≤ `max_bytes`) pass
/// through unchanged. Long strings are cut at a UTF-8 char boundary with a
/// `[truncated: N bytes omitted]` suffix appended (the suffix itself is not
/// counted against `max_bytes`).
pub(crate) fn truncate_for_event(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let cut = floor_char_boundary(text, max_bytes);
    let omitted = text.len() - cut;
    format!("{}[truncated: {omitted} bytes omitted]", &text[..cut])
}

/// Structured events emitted during agent and orchestrator execution.
///
/// All events carry the agent name for identification in multi-agent runs.
/// Events are emitted synchronously via the `OnEvent` callback — keep
/// handlers fast to avoid blocking the agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent loop started.
    RunStarted { agent: String, task: String },

    /// A new turn in the agent loop.
    TurnStarted {
        agent: String,
        turn: usize,
        max_turns: usize,
    },

    /// LLM call completed.
    LlmResponse {
        agent: String,
        turn: usize,
        usage: TokenUsage,
        stop_reason: StopReason,
        tool_call_count: usize,
        /// Truncated LLM response text.
        #[serde(default)]
        text: String,
        /// Wall-clock milliseconds for the LLM call.
        #[serde(default)]
        latency_ms: u64,
        /// Model name from the provider, if available.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Time-to-first-token in milliseconds (streaming only). 0 for non-streaming.
        #[serde(default)]
        time_to_first_token_ms: u64,
    },

    /// Tool execution started.
    ToolCallStarted {
        agent: String,
        tool_name: String,
        tool_call_id: String,
        /// Truncated JSON string of tool input.
        #[serde(default)]
        input: String,
    },

    /// Tool execution completed.
    ToolCallCompleted {
        agent: String,
        tool_name: String,
        tool_call_id: String,
        is_error: bool,
        duration_ms: u64,
        /// Truncated tool output content.
        #[serde(default)]
        output: String,
    },

    /// Human approval requested.
    ApprovalRequested {
        agent: String,
        turn: usize,
        tool_names: Vec<String>,
    },

    /// Human approval decision received.
    ApprovalDecision {
        agent: String,
        turn: usize,
        approved: bool,
    },

    /// Orchestrator dispatched sub-agents.
    SubAgentsDispatched { agent: String, agents: Vec<String> },

    /// A sub-agent completed.
    SubAgentCompleted {
        agent: String,
        success: bool,
        usage: TokenUsage,
    },

    /// Context was summarized due to threshold.
    ContextSummarized {
        agent: String,
        turn: usize,
        usage: TokenUsage,
    },

    /// Agent run completed successfully.
    RunCompleted {
        agent: String,
        total_usage: TokenUsage,
        tool_calls_made: usize,
    },

    /// A guardrail denied an LLM response or tool call.
    GuardrailDenied {
        agent: String,
        /// Which hook triggered the denial: `"post_llm"`, `"pre_tool"`, or `"post_tool"`.
        hook: String,
        reason: String,
        /// Set for `pre_tool` and `post_tool` denials, `None` for `post_llm`.
        tool_name: Option<String>,
    },

    /// Agent run failed.
    RunFailed {
        agent: String,
        error: String,
        partial_usage: TokenUsage,
    },

    /// An LLM retry attempt is about to happen (before the sleep).
    RetryAttempt {
        agent: String,
        /// Current attempt number (1-indexed).
        attempt: u32,
        /// Maximum retries configured.
        max_retries: u32,
        /// Delay in milliseconds before the retry.
        delay_ms: u64,
        /// Classified error that triggered the retry.
        #[serde(default)]
        error_class: String,
    },

    /// Doom loop detected: the agent repeated identical tool calls too many times.
    DoomLoopDetected {
        agent: String,
        turn: usize,
        /// Number of consecutive identical turns.
        consecutive_count: u32,
        /// Tool names in the repeated batch.
        #[serde(default)]
        tool_names: Vec<String>,
    },

    /// Session pruning truncated old tool results before an LLM call.
    SessionPruned {
        agent: String,
        turn: usize,
        /// Number of tool results that were truncated.
        tool_results_pruned: usize,
        /// Total bytes removed across all truncated tool results.
        bytes_saved: usize,
        /// Total tool results inspected (pruned + skipped).
        tool_results_total: usize,
    },

    /// Auto-compaction was triggered due to context overflow.
    AutoCompactionTriggered {
        agent: String,
        turn: usize,
        /// Whether compaction succeeded.
        success: bool,
        /// Token usage from the compaction LLM call.
        #[serde(default)]
        usage: TokenUsage,
    },

    /// A sensor event was processed through the triage pipeline.
    SensorEventProcessed {
        sensor_name: String,
        /// "promote", "drop", or "dead_letter"
        decision: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        priority: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        story_id: Option<String>,
    },

    /// A story was created or updated with new correlated events.
    StoryUpdated {
        story_id: String,
        subject: String,
        event_count: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        priority: Option<String>,
    },
}

/// Callback type for receiving structured agent events.
pub type OnEvent = dyn Fn(AgentEvent) + Send + Sync;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_serializes_to_tagged_json() {
        let event = AgentEvent::RunStarted {
            agent: "researcher".into(),
            task: "find info".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"run_started""#), "json: {json}");
        assert!(json.contains(r#""agent":"researcher""#), "json: {json}");
    }

    #[test]
    fn event_roundtrips_through_json() {
        let event = AgentEvent::LlmResponse {
            agent: "coder".into(),
            turn: 3,
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: 10,
                cache_read_input_tokens: 20,
                reasoning_tokens: 0,
            },
            stop_reason: StopReason::ToolUse,
            tool_call_count: 2,
            text: "hello world".into(),
            latency_ms: 42,
            model: Some("claude-3-5-sonnet".into()),
            time_to_first_token_ms: 0,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::LlmResponse {
                agent,
                turn,
                usage,
                tool_call_count,
                text,
                latency_ms,
                model,
                ..
            } => {
                assert_eq!(agent, "coder");
                assert_eq!(turn, 3);
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(tool_call_count, 2);
                assert_eq!(text, "hello world");
                assert_eq!(latency_ms, 42);
                assert_eq!(model.as_deref(), Some("claude-3-5-sonnet"));
            }
            other => panic!("expected LlmResponse, got: {other:?}"),
        }
    }

    #[test]
    fn tool_call_events_roundtrip() {
        let started = AgentEvent::ToolCallStarted {
            agent: "worker".into(),
            tool_name: "web_search".into(),
            tool_call_id: "call-1".into(),
            input: r#"{"query":"rust async"}"#.into(),
        };
        let json = serde_json::to_string(&started).unwrap();
        assert!(json.contains(r#""type":"tool_call_started""#));
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::ToolCallStarted { input, .. } => {
                assert_eq!(input, r#"{"query":"rust async"}"#);
            }
            other => panic!("expected ToolCallStarted, got: {other:?}"),
        }

        let completed = AgentEvent::ToolCallCompleted {
            agent: "worker".into(),
            tool_name: "web_search".into(),
            tool_call_id: "call-1".into(),
            is_error: false,
            duration_ms: 150,
            output: "search results here".into(),
        };
        let json = serde_json::to_string(&completed).unwrap();
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::ToolCallCompleted {
                duration_ms,
                is_error,
                output,
                ..
            } => {
                assert_eq!(duration_ms, 150);
                assert!(!is_error);
                assert_eq!(output, "search results here");
            }
            other => panic!("expected ToolCallCompleted, got: {other:?}"),
        }
    }

    #[test]
    fn all_variants_serialize() {
        // Ensure every variant can be serialized without error
        let events = vec![
            AgentEvent::RunStarted {
                agent: "a".into(),
                task: "t".into(),
            },
            AgentEvent::TurnStarted {
                agent: "a".into(),
                turn: 1,
                max_turns: 10,
            },
            AgentEvent::LlmResponse {
                agent: "a".into(),
                turn: 1,
                usage: TokenUsage::default(),
                stop_reason: StopReason::EndTurn,
                tool_call_count: 0,
                text: String::new(),
                latency_ms: 0,
                model: None,
                time_to_first_token_ms: 0,
            },
            AgentEvent::ToolCallStarted {
                agent: "a".into(),
                tool_name: "t".into(),
                tool_call_id: "c".into(),
                input: "{}".into(),
            },
            AgentEvent::ToolCallCompleted {
                agent: "a".into(),
                tool_name: "t".into(),
                tool_call_id: "c".into(),
                is_error: false,
                duration_ms: 0,
                output: String::new(),
            },
            AgentEvent::ApprovalRequested {
                agent: "a".into(),
                turn: 1,
                tool_names: vec!["t".into()],
            },
            AgentEvent::ApprovalDecision {
                agent: "a".into(),
                turn: 1,
                approved: true,
            },
            AgentEvent::SubAgentsDispatched {
                agent: "orchestrator".into(),
                agents: vec!["a".into()],
            },
            AgentEvent::SubAgentCompleted {
                agent: "a".into(),
                success: true,
                usage: TokenUsage::default(),
            },
            AgentEvent::ContextSummarized {
                agent: "a".into(),
                turn: 2,
                usage: TokenUsage::default(),
            },
            AgentEvent::RunCompleted {
                agent: "a".into(),
                total_usage: TokenUsage::default(),
                tool_calls_made: 0,
            },
            AgentEvent::GuardrailDenied {
                agent: "a".into(),
                hook: "post_llm".into(),
                reason: "unsafe".into(),
                tool_name: None,
            },
            AgentEvent::GuardrailDenied {
                agent: "a".into(),
                hook: "pre_tool".into(),
                reason: "blocked".into(),
                tool_name: Some("web_search".into()),
            },
            AgentEvent::GuardrailDenied {
                agent: "a".into(),
                hook: "post_tool".into(),
                reason: "output too long".into(),
                tool_name: Some("bash".into()),
            },
            AgentEvent::RunFailed {
                agent: "a".into(),
                error: "oops".into(),
                partial_usage: TokenUsage::default(),
            },
            AgentEvent::RetryAttempt {
                agent: "a".into(),
                attempt: 1,
                max_retries: 3,
                delay_ms: 500,
                error_class: "rate_limited".into(),
            },
            AgentEvent::DoomLoopDetected {
                agent: "a".into(),
                turn: 4,
                consecutive_count: 3,
                tool_names: vec!["web_search".into()],
            },
            AgentEvent::AutoCompactionTriggered {
                agent: "a".into(),
                turn: 2,
                success: true,
                usage: TokenUsage::default(),
            },
            AgentEvent::SensorEventProcessed {
                sensor_name: "tech_rss".into(),
                decision: "promote".into(),
                priority: Some("normal".into()),
                story_id: Some("story-123".into()),
            },
            AgentEvent::StoryUpdated {
                story_id: "story-123".into(),
                subject: "Rust ecosystem news".into(),
                event_count: 3,
                priority: Some("normal".into()),
            },
            AgentEvent::SessionPruned {
                agent: "a".into(),
                turn: 3,
                tool_results_pruned: 2,
                bytes_saved: 1500,
                tool_results_total: 4,
            },
        ];
        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let _back: AgentEvent = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn truncate_for_event_noop_when_short() {
        let short = "hello world";
        assert_eq!(truncate_for_event(short, 100), short);
    }

    #[test]
    fn truncate_for_event_zero_max_bytes() {
        let result = truncate_for_event("hello", 0);
        assert!(result.contains("[truncated: 5 bytes omitted]"));
        // Content portion should be empty
        assert!(result.starts_with("[truncated:"));
    }

    #[test]
    fn truncate_for_event_truncates_long_string() {
        let long = "a".repeat(5000);
        let result = truncate_for_event(&long, 100);
        assert!(result.len() < long.len());
        assert!(result.contains("[truncated:"));
        assert!(result.contains("bytes omitted]"));
    }

    #[test]
    fn truncate_for_event_preserves_utf8() {
        // "café" is 5 bytes: c(1) a(1) f(1) é(2)
        // Truncating at 4 bytes must not split the 'é'
        let text = format!("café{}", "x".repeat(5000));
        let result = truncate_for_event(&text, 4);
        // Should cut before 'é' (at byte 3) since byte 4 is mid-char
        assert!(result.starts_with("caf"));
        assert!(result.contains("[truncated:"));
    }

    #[test]
    fn llm_response_text_and_latency_roundtrip() {
        let event = AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: TokenUsage::default(),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: "some response text".into(),
            latency_ms: 123,
            model: Some("claude-3-opus".into()),
            time_to_first_token_ms: 0,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::LlmResponse {
                text,
                latency_ms,
                model,
                ..
            } => {
                assert_eq!(text, "some response text");
                assert_eq!(latency_ms, 123);
                assert_eq!(model.as_deref(), Some("claude-3-opus"));
            }
            other => panic!("expected LlmResponse, got: {other:?}"),
        }
    }

    #[test]
    fn llm_response_model_none_roundtrip() {
        let event = AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: TokenUsage::default(),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: String::new(),
            latency_ms: 0,
            model: None,
            time_to_first_token_ms: 0,
        };
        let json = serde_json::to_string(&event).unwrap();
        // model should not appear in JSON when None
        assert!(!json.contains("model"), "json: {json}");
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::LlmResponse { model, .. } => assert!(model.is_none()),
            other => panic!("expected LlmResponse, got: {other:?}"),
        }
    }

    #[test]
    fn tool_call_started_input_roundtrip() {
        let event = AgentEvent::ToolCallStarted {
            agent: "a".into(),
            tool_name: "read_file".into(),
            tool_call_id: "c1".into(),
            input: r#"{"path":"/tmp/f"}"#.into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::ToolCallStarted { input, .. } => {
                assert_eq!(input, r#"{"path":"/tmp/f"}"#);
            }
            other => panic!("expected ToolCallStarted, got: {other:?}"),
        }
    }

    #[test]
    fn tool_call_completed_output_roundtrip() {
        let event = AgentEvent::ToolCallCompleted {
            agent: "a".into(),
            tool_name: "bash".into(),
            tool_call_id: "c2".into(),
            is_error: false,
            duration_ms: 50,
            output: "command output here".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::ToolCallCompleted { output, .. } => {
                assert_eq!(output, "command output here");
            }
            other => panic!("expected ToolCallCompleted, got: {other:?}"),
        }
    }

    #[test]
    fn retry_attempt_roundtrip() {
        let event = AgentEvent::RetryAttempt {
            agent: "a".into(),
            attempt: 2,
            max_retries: 3,
            delay_ms: 1000,
            error_class: "rate_limited".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"retry_attempt""#));
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::RetryAttempt {
                agent,
                attempt,
                max_retries,
                delay_ms,
                error_class,
            } => {
                assert_eq!(agent, "a");
                assert_eq!(attempt, 2);
                assert_eq!(max_retries, 3);
                assert_eq!(delay_ms, 1000);
                assert_eq!(error_class, "rate_limited");
            }
            other => panic!("expected RetryAttempt, got: {other:?}"),
        }
    }

    #[test]
    fn doom_loop_detected_roundtrip() {
        let event = AgentEvent::DoomLoopDetected {
            agent: "b".into(),
            turn: 5,
            consecutive_count: 3,
            tool_names: vec!["web_search".into(), "read_file".into()],
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"doom_loop_detected""#));
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::DoomLoopDetected {
                agent,
                turn,
                consecutive_count,
                tool_names,
            } => {
                assert_eq!(agent, "b");
                assert_eq!(turn, 5);
                assert_eq!(consecutive_count, 3);
                assert_eq!(tool_names, vec!["web_search", "read_file"]);
            }
            other => panic!("expected DoomLoopDetected, got: {other:?}"),
        }
    }

    #[test]
    fn llm_response_ttft_roundtrip() {
        let event = AgentEvent::LlmResponse {
            agent: "a".into(),
            turn: 1,
            usage: TokenUsage::default(),
            stop_reason: StopReason::EndTurn,
            tool_call_count: 0,
            text: "hello".into(),
            latency_ms: 500,
            model: None,
            time_to_first_token_ms: 42,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(
            json.contains(r#""time_to_first_token_ms":42"#),
            "json: {json}"
        );
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::LlmResponse {
                time_to_first_token_ms,
                ..
            } => {
                assert_eq!(time_to_first_token_ms, 42);
            }
            other => panic!("expected LlmResponse, got: {other:?}"),
        }
    }

    #[test]
    fn backward_compat_llm_response_without_ttft() {
        // Old JSON without time_to_first_token_ms should deserialize with default 0
        let json = r#"{
            "type":"llm_response",
            "agent":"a",
            "turn":1,
            "usage":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0},
            "stop_reason":"end_turn",
            "tool_call_count":0,
            "text":"hello",
            "latency_ms":100
        }"#;
        let event: AgentEvent = serde_json::from_str(json).unwrap();
        match event {
            AgentEvent::LlmResponse {
                time_to_first_token_ms,
                ..
            } => {
                assert_eq!(time_to_first_token_ms, 0);
            }
            other => panic!("expected LlmResponse, got: {other:?}"),
        }
    }

    #[test]
    fn auto_compaction_triggered_roundtrip() {
        let usage = TokenUsage {
            input_tokens: 500,
            output_tokens: 200,
            ..Default::default()
        };
        let event = AgentEvent::AutoCompactionTriggered {
            agent: "c".into(),
            turn: 3,
            success: true,
            usage,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"auto_compaction_triggered""#));
        let back: AgentEvent = serde_json::from_str(&json).unwrap();
        match back {
            AgentEvent::AutoCompactionTriggered {
                agent,
                turn,
                success,
                usage,
            } => {
                assert_eq!(agent, "c");
                assert_eq!(turn, 3);
                assert!(success);
                assert_eq!(usage.input_tokens, 500);
                assert_eq!(usage.output_tokens, 200);
            }
            other => panic!("expected AutoCompactionTriggered, got: {other:?}"),
        }
    }

    #[test]
    fn backward_compatible_deserialization_without_new_fields() {
        // Old-format JSON without the new fields should still deserialize
        let json = r#"{
            "type":"llm_response",
            "agent":"a",
            "turn":1,
            "usage":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0},
            "stop_reason":"end_turn",
            "tool_call_count":0
        }"#;
        let event: AgentEvent = serde_json::from_str(json).unwrap();
        match event {
            AgentEvent::LlmResponse {
                text,
                latency_ms,
                model,
                ..
            } => {
                assert_eq!(text, "");
                assert_eq!(latency_ms, 0);
                assert!(model.is_none());
            }
            other => panic!("expected LlmResponse, got: {other:?}"),
        }

        let json = r#"{
            "type":"tool_call_started",
            "agent":"a",
            "tool_name":"t",
            "tool_call_id":"c"
        }"#;
        let event: AgentEvent = serde_json::from_str(json).unwrap();
        match event {
            AgentEvent::ToolCallStarted { input, .. } => assert_eq!(input, ""),
            other => panic!("expected ToolCallStarted, got: {other:?}"),
        }

        let json = r#"{
            "type":"tool_call_completed",
            "agent":"a",
            "tool_name":"t",
            "tool_call_id":"c",
            "is_error":false,
            "duration_ms":0
        }"#;
        let event: AgentEvent = serde_json::from_str(json).unwrap();
        match event {
            AgentEvent::ToolCallCompleted { output, .. } => assert_eq!(output, ""),
            other => panic!("expected ToolCallCompleted, got: {other:?}"),
        }
    }
}
