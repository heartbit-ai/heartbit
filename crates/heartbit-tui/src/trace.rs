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
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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

/// A clonable producer handle. Producers serialize + send; a dedicated thread
/// owns the file. Cheap to clone across the UI loop, the agent thread's
/// callbacks, and the tracing bridge.
#[derive(Clone)]
pub struct TraceHandle {
    tx: std::sync::mpsc::Sender<String>,
    seq: Arc<AtomicU64>,
    disabled: Arc<AtomicBool>,
}

impl TraceHandle {
    /// Record one event under the versioned envelope. Never blocks, never
    /// panics; a no-op once the writer has self-disabled.
    pub fn record(&self, src: TraceSrc, event: serde_json::Value) {
        if self.disabled.load(Ordering::Relaxed) {
            return;
        }
        let rec = TraceRecord {
            v: TRACE_VERSION,
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            ts: now_rfc3339_millis(),
            src,
            event,
        };
        if let Ok(line) = serde_json::to_string(&rec) {
            let _ = self.tx.send(line);
        }
    }

    /// Record a typed TUI-side event.
    pub fn record_ui(&self, event: &UiEvent) {
        if let Ok(v) = serde_json::to_value(event) {
            self.record(TraceSrc::Ui, v);
        }
    }

    /// Record a raw framework event (lossless; serde tag = "type").
    pub fn record_agent(&self, event: &heartbit_core::AgentEvent) {
        if let Ok(v) = serde_json::to_value(event) {
            self.record(TraceSrc::Agent, v);
        }
    }

    /// True once a write error permanently disabled this session's trace.
    pub fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }
}

/// Spawn the writer thread for `path` (parent dirs created; file 0600,
/// append-only, flushed per line). `on_error` fires AT MOST ONCE if the file
/// can't be opened or written — the trace then self-disables for the session;
/// it must never take down or block a run.
pub fn spawn_writer(path: PathBuf, on_error: Box<dyn FnOnce(String) + Send>) -> TraceHandle {
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let disabled = Arc::new(AtomicBool::new(false));
    let handle = TraceHandle {
        tx,
        seq: Arc::new(AtomicU64::new(0)),
        disabled: disabled.clone(),
    };
    std::thread::spawn(move || {
        use std::io::Write;
        // Failure path: set the disable flag BEFORE firing the callback, then
        // return so the thread exits. `on_error` is consumed at most once.
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            disabled.store(true, Ordering::Relaxed);
            on_error(format!("trace disabled: {e}"));
            return;
        }
        let mut opts = std::fs::OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = match opts.open(&path) {
            Ok(f) => f,
            Err(e) => {
                disabled.store(true, Ordering::Relaxed);
                on_error(format!("trace disabled: {e}"));
                return;
            }
        };
        while let Ok(line) = rx.recv() {
            if writeln!(file, "{line}")
                .and_then(|()| file.flush())
                .is_err()
            {
                disabled.store(true, Ordering::Relaxed);
                on_error("trace disabled: write failed".into());
                return;
            }
        }
        // All senders dropped → session over; file already flushed per line.
    });
    handle
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

    fn read_lines(path: &std::path::Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(|l| l.to_string())
            .collect()
    }

    /// Poll until the writer thread has flushed `n` lines (or time out).
    fn wait_for_lines(path: &std::path::Path, n: usize) -> Vec<String> {
        for _ in 0..100 {
            let lines = read_lines(path);
            if lines.len() >= n {
                return lines;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        read_lines(path)
    }

    #[test]
    fn writer_appends_jsonl_with_monotonic_seq() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s1.trace.jsonl");
        let handle = spawn_writer(path.clone(), Box::new(|_| {}));
        handle.record_ui(&UiEvent::UserInput { text: "a".into() });
        handle.record_ui(&UiEvent::UserInput { text: "b".into() });
        handle.record_agent(&heartbit_core::AgentEvent::RunStarted {
            agent: "entry".into(),
            task: "t".into(),
        });
        let lines = wait_for_lines(&path, 3);
        assert_eq!(lines.len(), 3);
        let recs: Vec<TraceRecord> = lines
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(recs[0].seq, 0);
        assert_eq!(recs[1].seq, 1);
        assert_eq!(recs[2].seq, 2);
        assert_eq!(recs[0].src, TraceSrc::Ui);
        assert_eq!(recs[2].src, TraceSrc::Agent);
        assert_eq!(recs[2].event["type"], "run_started");
    }

    #[test]
    fn writer_creates_parent_dir_and_0600_perms() {
        let dir = tempfile::tempdir().unwrap();
        // Parent "sessions" dir does NOT exist yet — first-launch case.
        let path = dir.path().join("sessions").join("s2.trace.jsonl");
        let handle = spawn_writer(path.clone(), Box::new(|_| {}));
        handle.record_ui(&UiEvent::UserInput { text: "x".into() });
        let lines = wait_for_lines(&path, 1);
        assert_eq!(lines.len(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "trace file must be 0600");
        }
    }

    #[test]
    fn writer_self_disables_on_error_and_fires_callback_once() {
        let dir = tempfile::tempdir().unwrap();
        // The target path IS a directory → open fails → self-disable.
        let errors = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let errs = errors.clone();
        let handle = spawn_writer(
            dir.path().to_path_buf(),
            Box::new(move |e| errs.lock().unwrap().push(e)),
        );
        // Poll for the asserted event (the callback push) — the mutex acquire
        // on push establishes happens-before, so the prior Relaxed disable
        // store is visible to is_disabled below (provably race-free).
        for _ in 0..100 {
            if errors.lock().unwrap().len() == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(errors.lock().unwrap().len(), 1, "error callback fires once");
        assert!(
            handle.is_disabled(),
            "writer must self-disable on open error"
        );
        // Recording after disable is a silent no-op (must not panic).
        handle.record_ui(&UiEvent::UserInput {
            text: "ignored".into(),
        });
    }
}
