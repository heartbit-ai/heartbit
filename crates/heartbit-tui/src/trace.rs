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
        if self.is_disabled() {
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
/// append-only, written straight through per line — `std::fs::File` is
/// unbuffered, so each `writeln!` reaches the OS; no fsync by design).
/// `on_error` fires AT MOST ONCE if the file can't be opened or written —
/// the trace then self-disables for the session; it must never take down or
/// block a run.
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

/// The trace file for a session id, under the sessions dir.
pub fn trace_path(dir: &std::path::Path, session_id: &str) -> PathBuf {
    dir.join(format!("{session_id}.trace.jsonl"))
}

/// Resolve a `/stats` / `/analyze` target to a trace file path.
/// `None` → the current session; `"last"` → the most recently modified trace
/// EXCLUDING the current session; anything else → a session id.
pub fn resolve_trace_target(
    dir: &std::path::Path,
    current_id: &str,
    target: Option<&str>,
) -> Result<PathBuf, String> {
    let exists = |p: PathBuf| {
        if p.is_file() {
            Ok(p)
        } else {
            Err(format!("no trace at {}", p.display()))
        }
    };
    match target {
        None => exists(trace_path(dir, current_id)),
        Some("last") => {
            let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
            let entries = std::fs::read_dir(dir).map_err(|e| e.to_string())?;
            for e in entries.flatten() {
                let p = e.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !name.ends_with(".trace.jsonl") || name == format!("{current_id}.trace.jsonl") {
                    continue;
                }
                let mtime = e
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::UNIX_EPOCH);
                if newest.as_ref().is_none_or(|(t, _)| mtime > *t) {
                    newest = Some((mtime, p));
                }
            }
            newest
                .map(|(_, p)| p)
                .ok_or_else(|| "no previous session trace found".into())
        }
        Some(id) => exists(trace_path(dir, id)),
    }
}

/// The `/analyze` task template. Why a prompt const and not a SKILL.md:
/// skills are progressive disclosure for when the AGENT decides; here the
/// COMMAND knows the guidance is needed, every time. Promotable later.
///
/// `staged_trace` is a WORKSPACE-RELATIVE path (e.g. `heartbit-trace-s1.jsonl`)
/// — the edge stages a snapshot copy into cwd because the workspace-rooted
/// builtins reject absolute paths (resolve_path, F-FS containment).
pub fn build_analyze_prompt(staged_trace: &str, session_id: &str, stats_json: &str) -> String {
    format!(
        r#"Analyze this heartbit-tui execution trace and produce a diagnosis report.

## Inputs
- Trace file (JSONL, one record per line), in the current directory: {staged_trace}
  (a frozen snapshot of session {session_id} — safe to grep, it won't change)
- Deterministic stats (already computed — trust these numbers):
```json
{stats_json}
```

## Trace format reference (envelope v1)
Each line: {{"v":1,"seq":N,"ts":"<rfc3339>","src":"agent|ui|core_trace","event":{{...}}}}
- src="agent": raw framework events, tagged by event.type — turn_started,
  llm_response (usage/latency_ms/time_to_first_token_ms/stop_reason),
  tool_call_started/tool_call_completed (tool_name/is_error/duration_ms/output),
  retry_attempt, doom_loop_detected, guardrail_denied/warned, approval_requested,
  sub_agents_dispatched, run_completed/run_failed, context_summarized, …
- src="ui": session_started (config snapshot), user_input, agent_spawned
  (epoch/reason), mode_changed, approval (decision/latency_ms/mode), effect,
  interrupt_requested, session_resumed, error.
- src="core_trace": raw interrupt-chain log mirror — ignore unless diagnosing interrupts.

## How to investigate (IMPORTANT: do not read the whole file — it can be huge)
Prefer the `grep` and `read` tools (they run silently); use bash/jq only when
you need aggregation. Targeted spots, e.g.:
- errors:        grep "\"is_error\":true" in {staged_trace}
- one tool call: grep "\"tool_call_id\":\"<id>\"" in {staged_trace}
- retries:       grep "\"type\":\"retry_attempt\"" in {staged_trace}
- slowest LLM (bash): jq -r 'select(.event.type=="llm_response") | "\(.event.latency_ms) seq=\(.seq)"' {staged_trace} | sort -rn | head
- a moment in time: grep "\"seq\":42," in {staged_trace} (seq is monotonic — read neighbors for context)

## Diagnosis dimensions
1. Errors & root chains (failed tools → what the agent did next; did it recover?)
2. Loops & waste (doom loops, repeated similar calls, token-heavy turns)
3. Latency outliers (slow LLM calls / tools; TTFT anomalies)
4. Approval friction (denials, long human latencies, modal interruptions)
5. Interrupts (user Esc — what was the agent doing that prompted it?)
6. Config issues (spawn reasons/epochs, mode changes mid-session, MCP failures)

## Deliverable
1. Present the findings concisely in your answer (cite seq numbers as evidence).
2. Write the full report to heartbit-diagnosis-{session_id}.md (current
   directory) with sections: Summary, Findings (each: evidence seq refs +
   impact), Recommendations (ranked, concrete — config/prompt/code), Stats
   appendix.
"#
    )
}

/// The target core's interrupt-chain checkpoints (CP3/CP4) and the TUI's
/// legacy diagnostics log under. The bridge mirrors that target into the
/// trace as `core_trace` records — the legacy `HEARTBIT_TUI_DEBUG` file is
/// untouched (additive, per spec).
pub const INTERRUPT_TARGET: &str = "heartbit::interrupt";

/// A `tracing` layer that mirrors [`INTERRUPT_TARGET`] events into the trace.
pub struct TracingBridge {
    handle: TraceHandle,
}

impl TracingBridge {
    pub fn new(handle: TraceHandle) -> Self {
        Self { handle }
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for TracingBridge {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if event.metadata().target() != INTERRUPT_TARGET {
            return;
        }
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        self.handle.record(
            TraceSrc::CoreTrace,
            serde_json::json!({
                "type": "log",
                "target": INTERRUPT_TARGET,
                "level": event.metadata().level().to_string(),
                "fields": serde_json::Value::Object(visitor.fields),
            }),
        );
    }
}

/// Collects a tracing event's fields into a JSON map (the free-text message
/// arrives as the field named `message`).
#[derive(Default)]
struct FieldVisitor {
    fields: serde_json::Map<String, serde_json::Value>,
}

impl tracing::field::Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.fields.insert(
            field.name().into(),
            serde_json::Value::String(format!("{value:?}")),
        );
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.fields
            .insert(field.name().into(), serde_json::Value::String(value.into()));
    }
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.fields.insert(field.name().into(), value.into());
    }
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.fields.insert(field.name().into(), value.into());
    }
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.fields.insert(field.name().into(), value.into());
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

    #[test]
    fn concurrent_producers_get_unique_complete_seqs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrent.trace.jsonl");
        let handle = spawn_writer(path.clone(), Box::new(|_| {}));
        let producers: Vec<_> = (0..4)
            .map(|i| {
                let h = handle.clone();
                std::thread::spawn(move || {
                    for j in 0..25 {
                        h.record_ui(&UiEvent::UserInput {
                            text: format!("t{i}-{j}"),
                        });
                    }
                })
            })
            .collect();
        for p in producers {
            p.join().unwrap();
        }
        let lines = wait_for_lines(&path, 100);
        assert_eq!(lines.len(), 100);
        // file order is NOT authoritative under concurrent producers — the seq
        // field is. We assert the SET of seqs is exactly {0..100} (unique and
        // complete); readers sort by seq, never by file position.
        let mut seqs: Vec<u64> = lines
            .iter()
            .map(|l| serde_json::from_str::<TraceRecord>(l).unwrap().seq)
            .collect();
        seqs.sort_unstable();
        assert_eq!(seqs, (0..100).collect::<Vec<u64>>());
    }

    #[test]
    fn tracing_bridge_captures_interrupt_target_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bridge.trace.jsonl");
        let handle = spawn_writer(path.clone(), Box::new(|_| {}));
        let subscriber = tracing_subscriber::layer::SubscriberExt::with(
            tracing_subscriber::registry(),
            TracingBridge::new(handle),
        );
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(
                target: "heartbit::interrupt",
                checkpoint = "CP3_tool_cancel_arm_fired",
                turn = 4u64,
                "cancel armed"
            );
            tracing::info!(target: "some::other", "must NOT be captured");
        });
        let lines = wait_for_lines(&path, 1);
        assert_eq!(lines.len(), 1, "exactly the interrupt-target event");
        let rec: TraceRecord = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(rec.src, TraceSrc::CoreTrace);
        assert_eq!(rec.event["type"], "log");
        assert_eq!(
            rec.event["fields"]["checkpoint"],
            "CP3_tool_cancel_arm_fired"
        );
        assert_eq!(rec.event["fields"]["turn"], 4);
        assert_eq!(rec.event["fields"]["message"], "cancel armed");
    }

    #[test]
    fn trace_path_is_sessions_id_trace_jsonl() {
        let p = trace_path(std::path::Path::new("/tmp/sessions"), "abc-1");
        assert_eq!(
            p,
            std::path::PathBuf::from("/tmp/sessions/abc-1.trace.jsonl")
        );
    }

    #[test]
    fn resolve_target_current_last_and_id() {
        let dir = tempfile::tempdir().unwrap();
        let mk = |id: &str| {
            std::fs::write(trace_path(dir.path(), id), "{}\n").unwrap();
        };
        mk("aaa-1");
        std::thread::sleep(std::time::Duration::from_millis(20));
        mk("bbb-2"); // newer
        std::thread::sleep(std::time::Duration::from_millis(20));
        mk("cur-3"); // the current session
        // None → current session's own trace
        let p = resolve_trace_target(dir.path(), "cur-3", None).unwrap();
        assert!(p.ends_with("cur-3.trace.jsonl"));
        // "last" → most recent EXCLUDING current
        let p = resolve_trace_target(dir.path(), "cur-3", Some("last")).unwrap();
        assert!(p.ends_with("bbb-2.trace.jsonl"), "got {p:?}");
        // explicit id
        let p = resolve_trace_target(dir.path(), "cur-3", Some("aaa-1")).unwrap();
        assert!(p.ends_with("aaa-1.trace.jsonl"));
        // missing id → error mentions it
        let e = resolve_trace_target(dir.path(), "cur-3", Some("nope")).unwrap_err();
        assert!(e.contains("nope"));
    }

    #[test]
    fn analyze_prompt_embeds_staged_path_stats_and_deliverable() {
        let p = build_analyze_prompt("heartbit-trace-s1.jsonl", "s1", "{\"turns\":2}");
        assert!(p.contains("heartbit-trace-s1.jsonl"));
        assert!(p.contains("{\"turns\":2}"));
        assert!(p.contains("heartbit-diagnosis-s1.md"));
        assert!(p.contains("\"v\":"), "format reference present");
        assert!(p.to_lowercase().contains("do not read the whole file"));
        assert!(
            !p.contains("/.config/"),
            "must reference only workspace-relative paths — builtins reject absolute paths"
        );
    }
}
