use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// --- Frame types ---

/// WebSocket frame discriminated by `type` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum WsFrame {
    /// Client -> Server request.
    Req {
        id: String,
        method: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    /// Server -> Client response.
    Res {
        id: String,
        ok: bool,
        #[serde(default)]
        payload: serde_json::Value,
    },
    /// Server -> Client push event.
    Event {
        event: String,
        payload: serde_json::Value,
        seq: u64,
    },
}

// --- Method constants ---

pub mod method {
    pub const CHAT_SEND: &str = "chat.send";
    pub const CHAT_ABORT: &str = "chat.abort";
    pub const CHAT_HISTORY: &str = "chat.history";
    pub const APPROVAL_RESOLVE: &str = "approval.resolve";
    pub const INPUT_RESOLVE: &str = "input.resolve";
    pub const QUESTION_RESOLVE: &str = "question.resolve";
    pub const SESSION_LIST: &str = "session.list";
    pub const SESSION_CREATE: &str = "session.create";
    pub const SESSION_DELETE: &str = "session.delete";
}

// --- Event constants ---

pub mod event {
    pub const CHAT_DELTA: &str = "chat.delta";
    pub const CHAT_FINAL: &str = "chat.final";
    pub const CHAT_ERROR: &str = "chat.error";
    pub const AGENT_EVENT: &str = "agent.event";
    pub const INPUT_NEEDED: &str = "input.needed";
    pub const APPROVAL_NEEDED: &str = "approval.needed";
    pub const QUESTION_NEEDED: &str = "question.needed";
}

// --- Payload structs ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSendParams {
    pub session_id: Uuid,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSendResult {
    pub task_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatAbortParams {
    pub session_id: Uuid,
    pub task_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatHistoryParams {
    pub session_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalResolveParams {
    pub interaction_id: Uuid,
    /// One of: "allow", "deny", "always_allow", "always_deny"
    pub decision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputResolveParams {
    pub interaction_id: Uuid,
    /// None or empty ends the session.
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResolveParams {
    pub interaction_id: Uuid,
    /// Per-question list of selected option labels.
    pub answers: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCreateParams {
    pub title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCreateResult {
    pub session_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDeleteParams {
    pub session_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionListResult {
    pub sessions: Vec<SessionSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: Uuid,
    pub title: Option<String>,
    pub created_at: DateTime<Utc>,
    pub message_count: usize,
}

// --- Push event payloads ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatDeltaPayload {
    pub session_id: Uuid,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFinalPayload {
    pub session_id: Uuid,
    pub result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatErrorPayload {
    pub session_id: Uuid,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEventPayload {
    pub session_id: Uuid,
    pub event: serde_json::Value,
}

/// Shared payload for `input.needed`, `approval.needed`, `question.needed`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionNeededPayload {
    pub session_id: Uuid,
    pub interaction_id: Uuid,
    /// Type-specific data (tool calls for approval, question for question, etc.)
    #[serde(default)]
    pub data: serde_json::Value,
}

// --- Helper methods ---

impl WsFrame {
    /// Create a success response frame.
    pub fn ok(id: impl Into<String>, payload: serde_json::Value) -> Self {
        WsFrame::Res {
            id: id.into(),
            ok: true,
            payload,
        }
    }

    /// Create an error response frame.
    pub fn err(id: impl Into<String>, message: impl Into<String>) -> Self {
        WsFrame::Res {
            id: id.into(),
            ok: false,
            payload: serde_json::json!({ "error": message.into() }),
        }
    }

    /// Create a push event frame.
    pub fn push(event: impl Into<String>, payload: serde_json::Value, seq: u64) -> Self {
        WsFrame::Event {
            event: event.into(),
            payload,
            seq,
        }
    }
}

impl ApprovalResolveParams {
    /// Parse the decision string into an `ApprovalDecision`.
    pub fn parse_decision(&self) -> Option<crate::llm::ApprovalDecision> {
        match self.decision.as_str() {
            "allow" => Some(crate::llm::ApprovalDecision::Allow),
            "deny" => Some(crate::llm::ApprovalDecision::Deny),
            "always_allow" => Some(crate::llm::ApprovalDecision::AlwaysAllow),
            "always_deny" => Some(crate::llm::ApprovalDecision::AlwaysDeny),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_frame_req_roundtrip() {
        let frame = WsFrame::Req {
            id: "abc".into(),
            method: "chat.send".into(),
            params: serde_json::json!({"key": "value"}),
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: WsFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            WsFrame::Req { id, method, params } => {
                assert_eq!(id, "abc");
                assert_eq!(method, "chat.send");
                assert_eq!(params["key"], "value");
            }
            other => panic!("expected Req, got {other:?}"),
        }
    }

    #[test]
    fn ws_frame_res_roundtrip() {
        let frame = WsFrame::Res {
            id: "123".into(),
            ok: true,
            payload: serde_json::json!({"data": 42}),
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: WsFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            WsFrame::Res { id, ok, payload } => {
                assert_eq!(id, "123");
                assert!(ok);
                assert_eq!(payload["data"], 42);
            }
            other => panic!("expected Res, got {other:?}"),
        }
    }

    #[test]
    fn ws_frame_event_roundtrip() {
        let frame = WsFrame::Event {
            event: "chat.delta".into(),
            payload: serde_json::json!({"text": "hello"}),
            seq: 7,
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: WsFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            WsFrame::Event {
                event,
                payload,
                seq,
            } => {
                assert_eq!(event, "chat.delta");
                assert_eq!(payload["text"], "hello");
                assert_eq!(seq, 7);
            }
            other => panic!("expected Event, got {other:?}"),
        }
    }

    #[test]
    fn ws_frame_req_from_json() {
        let raw = r#"{"type":"req","id":"r1","method":"chat.send","params":{"msg":"hi"}}"#;
        let frame: WsFrame = serde_json::from_str(raw).expect("parse");
        match frame {
            WsFrame::Req { id, method, params } => {
                assert_eq!(id, "r1");
                assert_eq!(method, "chat.send");
                assert_eq!(params["msg"], "hi");
            }
            other => panic!("expected Req, got {other:?}"),
        }
    }

    #[test]
    fn ws_frame_res_from_json() {
        let raw = r#"{"type":"res","id":"r2","ok":false,"payload":{"error":"bad"}}"#;
        let frame: WsFrame = serde_json::from_str(raw).expect("parse");
        match frame {
            WsFrame::Res { id, ok, payload } => {
                assert_eq!(id, "r2");
                assert!(!ok);
                assert_eq!(payload["error"], "bad");
            }
            other => panic!("expected Res, got {other:?}"),
        }
    }

    #[test]
    fn ws_frame_event_from_json() {
        let raw = r#"{"type":"event","event":"chat.delta","payload":{"text":"x"},"seq":1}"#;
        let frame: WsFrame = serde_json::from_str(raw).expect("parse");
        match frame {
            WsFrame::Event {
                event,
                payload,
                seq,
            } => {
                assert_eq!(event, "chat.delta");
                assert_eq!(payload["text"], "x");
                assert_eq!(seq, 1);
            }
            other => panic!("expected Event, got {other:?}"),
        }
    }

    #[test]
    fn ws_frame_unknown_type_fails() {
        let raw = r#"{"type":"unknown","data":{}}"#;
        let result = serde_json::from_str::<WsFrame>(raw);
        assert!(result.is_err());
    }

    #[test]
    fn chat_send_params_roundtrip() {
        let id = Uuid::new_v4();
        let params = ChatSendParams {
            session_id: id,
            message: "hello world".into(),
        };
        let json = serde_json::to_string(&params).expect("serialize");
        let parsed: ChatSendParams = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.session_id, id);
        assert_eq!(parsed.message, "hello world");
    }

    #[test]
    fn approval_resolve_parse_decision() {
        let make = |d: &str| ApprovalResolveParams {
            interaction_id: Uuid::new_v4(),
            decision: d.into(),
        };

        assert_eq!(
            make("allow").parse_decision(),
            Some(crate::llm::ApprovalDecision::Allow)
        );
        assert_eq!(
            make("deny").parse_decision(),
            Some(crate::llm::ApprovalDecision::Deny)
        );
        assert_eq!(
            make("always_allow").parse_decision(),
            Some(crate::llm::ApprovalDecision::AlwaysAllow)
        );
        assert_eq!(
            make("always_deny").parse_decision(),
            Some(crate::llm::ApprovalDecision::AlwaysDeny)
        );
        assert_eq!(make("bogus").parse_decision(), None);
    }

    #[test]
    fn ws_frame_ok_helper() {
        let frame = WsFrame::ok("id1", serde_json::json!({"result": true}));
        match frame {
            WsFrame::Res { id, ok, payload } => {
                assert_eq!(id, "id1");
                assert!(ok);
                assert_eq!(payload["result"], true);
            }
            other => panic!("expected Res, got {other:?}"),
        }
    }

    #[test]
    fn ws_frame_err_helper() {
        let frame = WsFrame::err("id2", "something failed");
        match frame {
            WsFrame::Res { id, ok, payload } => {
                assert_eq!(id, "id2");
                assert!(!ok);
                assert_eq!(payload["error"], "something failed");
            }
            other => panic!("expected Res, got {other:?}"),
        }
    }

    #[test]
    fn ws_frame_push_helper() {
        let frame = WsFrame::push("chat.delta", serde_json::json!({"text": "hi"}), 42);
        match frame {
            WsFrame::Event {
                event,
                payload,
                seq,
            } => {
                assert_eq!(event, "chat.delta");
                assert_eq!(payload["text"], "hi");
                assert_eq!(seq, 42);
            }
            other => panic!("expected Event, got {other:?}"),
        }
    }

    #[test]
    fn session_summary_roundtrip() {
        let summary = SessionSummary {
            id: Uuid::new_v4(),
            title: Some("Test session".into()),
            created_at: Utc::now(),
            message_count: 5,
        };
        let json = serde_json::to_string(&summary).expect("serialize");
        let parsed: SessionSummary = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.id, summary.id);
        assert_eq!(parsed.title, Some("Test session".into()));
        assert_eq!(parsed.message_count, 5);
    }

    #[test]
    fn interaction_needed_payload_roundtrip() {
        let sid = Uuid::new_v4();
        let iid = Uuid::new_v4();
        let payload = InteractionNeededPayload {
            session_id: sid,
            interaction_id: iid,
            data: serde_json::json!({"tools": ["bash"]}),
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        let parsed: InteractionNeededPayload = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.session_id, sid);
        assert_eq!(parsed.interaction_id, iid);
        assert_eq!(parsed.data["tools"][0], "bash");
    }

    #[test]
    fn ws_frame_req_default_params() {
        let raw = r#"{"type":"req","id":"x","method":"session.list"}"#;
        let frame: WsFrame = serde_json::from_str(raw).expect("parse");
        match frame {
            WsFrame::Req { params, .. } => {
                assert_eq!(params, serde_json::Value::Null);
            }
            other => panic!("expected Req, got {other:?}"),
        }
    }

    #[test]
    fn ws_frame_res_default_payload() {
        let raw = r#"{"type":"res","id":"x","ok":true}"#;
        let frame: WsFrame = serde_json::from_str(raw).expect("parse");
        match frame {
            WsFrame::Res { payload, .. } => {
                assert_eq!(payload, serde_json::Value::Null);
            }
            other => panic!("expected Res, got {other:?}"),
        }
    }

    #[test]
    fn interaction_needed_payload_default_data() {
        let raw = r#"{"session_id":"00000000-0000-0000-0000-000000000001","interaction_id":"00000000-0000-0000-0000-000000000002"}"#;
        let parsed: InteractionNeededPayload = serde_json::from_str(raw).expect("parse");
        assert_eq!(parsed.data, serde_json::Value::Null);
    }

    #[test]
    fn method_constants() {
        assert_eq!(method::CHAT_SEND, "chat.send");
        assert_eq!(method::CHAT_ABORT, "chat.abort");
        assert_eq!(method::CHAT_HISTORY, "chat.history");
        assert_eq!(method::APPROVAL_RESOLVE, "approval.resolve");
        assert_eq!(method::INPUT_RESOLVE, "input.resolve");
        assert_eq!(method::QUESTION_RESOLVE, "question.resolve");
        assert_eq!(method::SESSION_LIST, "session.list");
        assert_eq!(method::SESSION_CREATE, "session.create");
        assert_eq!(method::SESSION_DELETE, "session.delete");
    }

    #[test]
    fn event_constants() {
        assert_eq!(event::CHAT_DELTA, "chat.delta");
        assert_eq!(event::CHAT_FINAL, "chat.final");
        assert_eq!(event::CHAT_ERROR, "chat.error");
        assert_eq!(event::AGENT_EVENT, "agent.event");
        assert_eq!(event::INPUT_NEEDED, "input.needed");
        assert_eq!(event::APPROVAL_NEEDED, "approval.needed");
        assert_eq!(event::QUESTION_NEEDED, "question.needed");
    }
}
