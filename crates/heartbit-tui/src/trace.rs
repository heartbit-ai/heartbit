#![allow(dead_code)] // TODO(trace): remove once Task 5 wires the module

//! Always-on execution trace: one JSONL file per launch under
//! `<config-dir>/sessions/<id>.trace.jsonl`. Three sources feed one versioned
//! envelope: `agent` (raw [`heartbit_core::AgentEvent`]s, tapped before
//! `Msg::from_event`), `ui` (typed TUI-side happenings the framework can't
//! see), and `core_trace` (bridged `heartbit::interrupt` tracing checkpoints).
//! The writer is a dedicated thread behind a channel — tracing must NEVER
//! block or take down a session.
//! Spec: docs/superpowers/specs/2026-06-06-tui-debug-trace-design.md.

use serde::{Deserialize, Serialize};

/// Schema version of the trace envelope (the format contract for future
/// agent consumers — bump on breaking changes).
pub const TRACE_VERSION: u8 = 1;

/// Which tap produced a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceSrc {
    /// A raw framework [`heartbit_core::AgentEvent`] (serde-tagged by `type`).
    Agent,
    /// A typed TUI-side [`UiEvent`].
    Ui,
    /// A bridged `heartbit::interrupt` tracing event (CP3/CP4 etc.).
    CoreTrace,
}

/// One line of the trace file.
///
/// Evolution rule: any future field on the envelope or a UiEvent variant MUST be Option or #[serde(default)] so new readers tolerate old traces.
#[derive(Debug, Serialize, Deserialize)]
pub struct TraceRecord {
    pub v: u8,
    /// Monotonic per session — orders records when timestamps collide.
    pub seq: u64,
    /// RFC3339 UTC with millisecond precision.
    pub ts: String,
    pub src: TraceSrc,
    pub event: serde_json::Value,
}

/// TUI-side trace events — everything `AgentEvent` cannot see.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UiEvent {
    /// Config snapshot at launch.
    SessionStarted {
        version: String,
        session_id: String,
        model: String,
        permission_mode: String,
        mcp_servers: Vec<String>,
        context_recall: bool,
        verify_command: Option<String>,
    },
    /// A user message submitted to the agent (slash commands excluded).
    UserInput { text: String },
    /// An agent thread (re)spawn with the config that actually shapes the
    /// engine (the eager-spawn stale-config bug class).
    AgentSpawned {
        epoch: u64,
        model: String,
        /// "startup" | "respawn" | "key_set"
        reason: String,
        context_recall: bool,
        verify_command: Option<String>,
    },
    /// Permission mode transition (labels: normal | plan | yolo).
    ModeChanged { from: String, to: String },
    /// An [`crate::app::Effect`] executed at the edge.
    Effect { name: String, duration_ms: u64 },
    /// The approval gate resolved (human think-time is rung-2 gold).
    Approval {
        tools: Vec<String>,
        /// "allow" | "deny" | "always_allow" | "always_deny"
        decision: String,
        latency_ms: u64,
        /// Mode in effect: "normal" | "plan" | "yolo"
        mode: String,
    },
    /// TUI half of the interrupt chain (CP3/CP4 arrive via `core_trace`).
    InterruptRequested { checkpoint: String, running: bool },
    /// `/resume` loaded another session's transcript (trace lineage).
    SessionResumed { from_id: String },
    /// A TUI-side error that was previously silent (e.g. session save).
    Error { context: String, message: String },
}

/// RFC3339 UTC with milliseconds, e.g. `2026-06-06T12:34:56.789Z`.
pub fn now_rfc3339_millis() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Human label for the shared permission-mode u8 (0=normal, 1=plan, 2=yolo —
/// matches `PermissionMode::as_u8` and the `on_approval` gate).
pub fn mode_label(mode: u8) -> &'static str {
    match mode {
        1 => "plan",
        2 => "yolo",
        _ => "normal",
    }
}

/// Stable label for an [`heartbit_core::ApprovalDecision`].
pub fn decision_label(d: &heartbit_core::ApprovalDecision) -> &'static str {
    match d {
        heartbit_core::ApprovalDecision::Allow => "allow",
        heartbit_core::ApprovalDecision::Deny => "deny",
        heartbit_core::ApprovalDecision::AlwaysAllow => "always_allow",
        heartbit_core::ApprovalDecision::AlwaysDeny => "always_deny",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_roundtrips_with_versioned_fields() {
        let rec = TraceRecord {
            v: TRACE_VERSION,
            seq: 7,
            ts: now_rfc3339_millis(),
            src: TraceSrc::Ui,
            event: serde_json::json!({"type": "user_input", "text": "hi"}),
        };
        let line = serde_json::to_string(&rec).unwrap();
        assert!(line.contains("\"v\":1"));
        assert!(line.contains("\"src\":\"ui\""));
        let back: TraceRecord = serde_json::from_str(&line).unwrap();
        assert_eq!(back.seq, 7);
        assert_eq!(back.src, TraceSrc::Ui);
        assert_eq!(back.event["text"], "hi");
    }

    #[test]
    fn agent_event_serializes_lossless_under_the_envelope() {
        // The raw AgentEvent (serde tag = "type") must round-trip unchanged —
        // the trace is lossless w.r.t. what the framework emits.
        let ev = heartbit_core::AgentEvent::ToolCallStarted {
            agent: "entry".into(),
            tool_name: "bash".into(),
            tool_call_id: "tc1".into(),
            input: "{\"command\":\"ls\"}".into(),
        };
        let val = serde_json::to_value(&ev).unwrap();
        assert_eq!(val["type"], "tool_call_started");
        assert_eq!(val["tool_name"], "bash");
        let back: heartbit_core::AgentEvent = serde_json::from_value(val).unwrap();
        match back {
            heartbit_core::AgentEvent::ToolCallStarted { tool_call_id, .. } => {
                assert_eq!(tool_call_id, "tc1");
            }
            other => panic!("expected ToolCallStarted, got {other:?}"),
        }
    }

    #[test]
    fn ui_events_are_type_tagged_snake_case() {
        let ev = UiEvent::Approval {
            tools: vec!["bash".into()],
            decision: "allow".into(),
            latency_ms: 3400,
            mode: "normal".into(),
        };
        let val = serde_json::to_value(&ev).unwrap();
        assert_eq!(val["type"], "approval");
        assert_eq!(val["latency_ms"], 3400);
        let spawned = UiEvent::AgentSpawned {
            epoch: 2,
            model: "m".into(),
            reason: "respawn".into(),
            context_recall: true,
            verify_command: None,
        };
        assert_eq!(
            serde_json::to_value(&spawned).unwrap()["type"],
            "agent_spawned"
        );
    }

    #[test]
    fn timestamp_is_rfc3339_utc_millis() {
        let ts = now_rfc3339_millis();
        // e.g. 2026-06-06T12:34:56.789Z — ends in Z, has a millis dot.
        assert!(ts.ends_with('Z'), "got: {ts}");
        assert!(ts.contains('.'), "got: {ts}");
        assert!(chrono::DateTime::parse_from_rfc3339(&ts).is_ok());
    }

    #[test]
    fn mode_and_decision_labels() {
        assert_eq!(mode_label(0), "normal");
        assert_eq!(mode_label(1), "plan");
        assert_eq!(mode_label(2), "yolo");
        assert_eq!(mode_label(99), "normal");
        assert_eq!(
            decision_label(&heartbit_core::ApprovalDecision::AlwaysAllow),
            "always_allow"
        );
    }

    #[test]
    fn wire_shape_is_pinned_for_all_variants() {
        use serde_json::json;

        // TraceSrc wire strings — the format contract for the `src` discriminant.
        assert_eq!(
            serde_json::to_value(TraceSrc::Agent).unwrap(),
            json!("agent")
        );
        assert_eq!(serde_json::to_value(TraceSrc::Ui).unwrap(), json!("ui"));
        assert_eq!(
            serde_json::to_value(TraceSrc::CoreTrace).unwrap(),
            json!("core_trace")
        );

        // Each UiEvent variant: pin its `type` tag and every field name.
        let cases: Vec<(UiEvent, &str, &[&str])> = vec![
            (
                UiEvent::SessionStarted {
                    version: "v".into(),
                    session_id: "s".into(),
                    model: "m".into(),
                    permission_mode: "normal".into(),
                    mcp_servers: vec![],
                    context_recall: false,
                    verify_command: None,
                },
                "session_started",
                &[
                    "version",
                    "session_id",
                    "model",
                    "permission_mode",
                    "mcp_servers",
                    "context_recall",
                    "verify_command",
                ],
            ),
            (
                UiEvent::UserInput { text: "hi".into() },
                "user_input",
                &["text"],
            ),
            (
                UiEvent::AgentSpawned {
                    epoch: 1,
                    model: "m".into(),
                    reason: "startup".into(),
                    context_recall: false,
                    verify_command: None,
                },
                "agent_spawned",
                &[
                    "epoch",
                    "model",
                    "reason",
                    "context_recall",
                    "verify_command",
                ],
            ),
            (
                UiEvent::ModeChanged {
                    from: "normal".into(),
                    to: "plan".into(),
                },
                "mode_changed",
                &["from", "to"],
            ),
            (
                UiEvent::Effect {
                    name: "save".into(),
                    duration_ms: 1,
                },
                "effect",
                &["name", "duration_ms"],
            ),
            (
                UiEvent::Approval {
                    tools: vec!["bash".into()],
                    decision: "allow".into(),
                    latency_ms: 1,
                    mode: "normal".into(),
                },
                "approval",
                &["tools", "decision", "latency_ms", "mode"],
            ),
            (
                UiEvent::InterruptRequested {
                    checkpoint: "CP1".into(),
                    running: true,
                },
                "interrupt_requested",
                &["checkpoint", "running"],
            ),
            (
                UiEvent::SessionResumed {
                    from_id: "x".into(),
                },
                "session_resumed",
                &["from_id"],
            ),
            (
                UiEvent::Error {
                    context: "save".into(),
                    message: "boom".into(),
                },
                "error",
                &["context", "message"],
            ),
        ];
        assert_eq!(cases.len(), 9, "all UiEvent variants must be pinned");
        for (ev, tag, fields) in &cases {
            let val = serde_json::to_value(ev).unwrap();
            let obj = val.as_object().unwrap();
            assert_eq!(obj["type"], json!(tag), "wrong tag for {tag}");
            for field in *fields {
                assert!(obj.contains_key(*field), "{tag} missing field {field}");
            }
        }

        // TraceRecord top-level shape: exactly these five keys, nothing else.
        let rec = TraceRecord {
            v: TRACE_VERSION,
            seq: 0,
            ts: now_rfc3339_millis(),
            src: TraceSrc::Ui,
            event: json!({}),
        };
        let val = serde_json::to_value(&rec).unwrap();
        let mut keys: Vec<&str> = val
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["event", "seq", "src", "ts", "v"]);
    }
}
