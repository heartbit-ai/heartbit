# TUI Execution Trace + Diagnosis v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Always-on per-session JSONL execution trace for `heartbit-tui` (agent + UI + core-trace taps, versioned envelope), plus `/stats` (deterministic trace stats) and `/analyze` (agent-driven diagnosis report). Zero heartbit-core changes.

**Architecture:** A new `trace.rs` module owns a writer thread behind an mpsc channel; three producers feed it: the raw `AgentEvent` tap (in `build_engine`'s `on_event` closure, *before* `Msg::from_event`), typed `UiEvent`s recorded at the main-loop edge, and a `tracing::Layer` bridging core's `heartbit::interrupt` checkpoints. A new `trace_stats.rs` streams the JSONL into a `TraceStats` summary that `/stats` renders directly and `/analyze` embeds into a prompt for the regular agent.

**Tech Stack:** Rust edition 2024, serde/serde_json (already deps), chrono (workspace dep, to add to heartbit-tui), tracing-subscriber 0.3 (already dep, `registry` + `Targets` per-layer filters), tempfile (already dev-dep).

**Spec:** `docs/superpowers/specs/2026-06-06-tui-debug-trace-design.md`

---

## Verified ground truth (do not re-derive)

- `AgentEvent` is `#[serde(tag = "type", rename_all = "snake_case")]` — `crates/heartbit-core/src/agent/events.rs:32`. Serializing it yields `{"type":"tool_call_started",...}` directly.
- `heartbit_core` root exports: `OnRetry` (`pub use llm::retry::{OnRetry, RetryConfig, RetryingProvider}` at `lib.rs:150`), `AgentEvent`, `ApprovalDecision` (variants `Allow | Deny | AlwaysAllow | AlwaysDeny`), `OnEvent`, `TokenUsage`.
- `RetryingProvider::with_on_retry(Arc<OnRetry>) -> Self` is chainable (`retry.rs:74`). `OnRetry = dyn Fn(u32, u32, u64, &str) + Send + Sync` (attempt, max_retries, delay_ms, error_class).
- The CLI pattern to mirror: `build_on_retry` at `crates/heartbit-cli/src/main.rs:470-485` emits `AgentEvent::RetryAttempt { agent: "(provider)".into(), attempt, max_retries, delay_ms, error_class }`.
- `RetryAttempt` fields: `agent: String, attempt: u32, max_retries: u32, delay_ms: u64, error_class: String` (`events.rs:165`).
- TUI provider construction: `build_provider` at `crates/heartbit-tui/src/main.rs:168-193`, two `RetryingProvider::with_defaults(base)` sites (175, 189). It is called at the TOP of `build_engine` (main.rs:329) — **before** the `on_event` closure is defined (389). Task 6 reorders.
- `on_event` closure: main.rs:389-407. `on_approval` closure: main.rs:408-443 (mode gate: `2`=YOLO allow, `1`=Plan deny-mutating, else modal round-trip over `std::sync::mpsc::sync_channel(1)`).
- Permission u8 map (current, from `app.rs PermissionMode::as_u8` / on_approval): **0=normal, 1=plan, 2=yolo**. (A stale main.rs:130 comment says otherwise — trust the code.)
- `spawn_agent`: main.rs:623-687, called from 3 sites: eager startup (run_ui ~712), `Effect::SendInput` respawn (~770), `Effect::SaveKey` (~792). Epoch handling already exists.
- Effect dispatch loop: main.rs:761-905. CP1/CP2 tracing at 888-902. `save_session` (best-effort, swallows errors) at 923-930.
- Session id: `format!("{:x}-{}", unix_secs, pid)` at main.rs:134-141 — created **after** `init_debug_logging()`. Task 5 moves it earlier.
- `session::sessions_dir()` = `config_path().parent()/sessions` (`session.rs:28-33`).
- `app.rs`: `SLASH_COMMANDS` (147-163), `Effect` enum (167-195), `handle_slash` (840-893), `submit` (800-838; the Plan-mode `sent ≠ displayed` precedent lives here), `seed_idle_squad` exists.
- `msg.rs`: `Msg::from_event` (101-185); `_ => None` catch-all means `RetryAttempt` currently maps to nothing.
- `TokenUsage { input_tokens: u32, output_tokens: u32, cache_creation_input_tokens, cache_read_input_tokens, reasoning_tokens }`, `Copy` (`types.rs:91`).
- Workspace has `chrono = { version = "0.4", features = ["serde"] }`; `tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }` (registry is a default feature).
- heartbit-tui dev-deps already include `tempfile = "3"`.
- Quality gate: `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm`.
- **Spec divergence note:** the spec's `agent_spawned.multi_agent` field is replaced by `context_recall` + `verify` — the unified entry agent (option C) removed the static multi-agent flag from engine construction; `spawn_agent` doesn't read `app.multi_agent`. The spawn-config snapshot captures what actually varies the engine.

## File structure

- **Create** `crates/heartbit-tui/src/trace.rs` — envelope (`TraceRecord`, `TraceSrc`), `UiEvent`, `TraceHandle` + writer thread, `TracingBridge` layer, timestamp helper, path/target resolution, the `/analyze` prompt builder.
- **Create** `crates/heartbit-tui/src/trace_stats.rs` — `TraceStats` + streaming compute + percentile + text render.
- **Modify** `crates/heartbit-tui/src/main.rs` — wiring: writer creation, tracing init refactor, taps, retry wiring, new effects.
- **Modify** `crates/heartbit-tui/src/app.rs` — `/stats` + `/analyze` commands, new `Effect` variants, new `Msg` handling.
- **Modify** `crates/heartbit-tui/src/msg.rs` — `StatsReady`/`AnalyzeReady` variants + `RetryAttempt → Notice` mapping.
- **Modify** `crates/heartbit-tui/Cargo.toml` — add `chrono = { workspace = true }`.

Run all per-task tests from the workspace root with `cargo test -p heartbit-tui <filter>`.

---

### Task 1: Trace envelope + UiEvent types (`trace.rs`, pure)

**Files:**
- Modify: `crates/heartbit-tui/Cargo.toml`
- Create: `crates/heartbit-tui/src/trace.rs`
- Modify: `crates/heartbit-tui/src/main.rs` (one `mod trace;` line)

- [ ] **Step 1: Add the chrono dependency**

In `crates/heartbit-tui/Cargo.toml`, after the `walkdir` line in `[dependencies]`:

```toml
chrono = { workspace = true }
```

- [ ] **Step 2: Write the failing tests**

Create `crates/heartbit-tui/src/trace.rs`:

```rust
//! Always-on execution trace: one JSONL file per launch under
//! `<config-dir>/sessions/<id>.trace.jsonl`. Three sources feed one versioned
//! envelope: `agent` (raw [`AgentEvent`]s, tapped before `Msg::from_event`),
//! `ui` (typed TUI-side happenings the framework can't see), and `core_trace`
//! (bridged `heartbit::interrupt` tracing checkpoints). The writer is a
//! dedicated thread behind a channel — tracing must NEVER block or take down
//! a session. Spec: docs/superpowers/specs/2026-06-06-tui-debug-trace-design.md.

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
    chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
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
        assert_eq!(serde_json::to_value(&spawned).unwrap()["type"], "agent_spawned");
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
}
```

In `crates/heartbit-tui/src/main.rs`, add to the module list (after `mod session;`):

```rust
mod trace;
```

- [ ] **Step 3: Run tests to verify they fail, then pass**

Run: `cargo test -p heartbit-tui trace::`
Expected: compile error first (missing module) is NOT the goal — after creating the file it should compile and all 5 tests PASS. If `heartbit_core::ApprovalDecision` import fails, it is re-exported at the root (verified) — check for typos.

Note: `mod trace` will trip `dead_code` warnings under `-D warnings` until Task 5 wires it. Add `#![allow(dead_code)]` is NOT the project style — instead mark module-level: add `#[allow(dead_code)]` on items only if clippy complains at this stage, and REMOVE those allows in Task 5 when everything is used. (Tests count as uses for most items; `pub` items in a binary crate may still warn.)

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-tui/Cargo.toml crates/heartbit-tui/src/trace.rs crates/heartbit-tui/src/main.rs
git commit -m "feat(tui): trace envelope + UiEvent schema (v1 format contract)"
```

---

### Task 2: TraceWriter — channel + writer thread

**Files:**
- Modify: `crates/heartbit-tui/src/trace.rs`

- [ ] **Step 1: Write the failing tests**

Append to the `tests` module in `trace.rs`:

```rust
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
        // Give the writer thread time to fail the open.
        for _ in 0..100 {
            if handle.is_disabled() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(handle.is_disabled(), "writer must self-disable on open error");
        assert_eq!(errors.lock().unwrap().len(), 1, "error callback fires once");
        // Recording after disable is a silent no-op (must not panic).
        handle.record_ui(&UiEvent::UserInput { text: "ignored".into() });
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p heartbit-tui trace::`
Expected: FAIL — `spawn_writer`, `TraceHandle`, `record_ui`, `record_agent`, `is_disabled` not defined.

- [ ] **Step 3: Implement the writer**

Add to `trace.rs` (above the tests module):

```rust
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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
        let mut on_error = Some(on_error);
        let mut fail = |msg: String| {
            disabled.store(true, Ordering::Relaxed);
            if let Some(cb) = on_error.take() {
                cb(msg);
            }
        };
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            fail(format!("trace disabled: {e}"));
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
                fail(format!("trace disabled: {e}"));
                return;
            }
        };
        while let Ok(line) = rx.recv() {
            if writeln!(file, "{line}").and_then(|()| file.flush()).is_err() {
                fail("trace disabled: write failed".into());
                return;
            }
        }
        // All senders dropped → session over; file already flushed per line.
    });
    handle
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p heartbit-tui trace::`
Expected: all PASS (8 tests so far).

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-tui/src/trace.rs
git commit -m "feat(tui): trace writer thread — append-only 0600 JSONL, self-disabling"
```

---

### Task 3: TracingBridge — `heartbit::interrupt` → `core_trace` records

**Files:**
- Modify: `crates/heartbit-tui/src/trace.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module:

```rust
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
        assert_eq!(rec.event["fields"]["checkpoint"], "CP3_tool_cancel_arm_fired");
        assert_eq!(rec.event["fields"]["turn"], 4);
        assert_eq!(rec.event["fields"]["message"], "cancel armed");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p heartbit-tui trace::tests::tracing_bridge`
Expected: FAIL — `TracingBridge` not defined.

- [ ] **Step 3: Implement the bridge layer**

Add to `trace.rs`:

```rust
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
        self.fields
            .insert(field.name().into(), serde_json::Value::String(format!("{value:?}")));
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
```

Note: `record_str` gives clean strings for `checkpoint = "..."` style fields; `record_debug` catches `%`/`?` formatted ones (they arrive Debug-quoted, e.g. `"\"worker\""` for `?agents` — acceptable, this stream is a raw mirror that `trace_stats` never parses).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p heartbit-tui trace::`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-tui/src/trace.rs
git commit -m "feat(tui): tracing bridge — heartbit::interrupt checkpoints into the trace"
```

---

### Task 4: `RetryAttempt` → user-visible Notice (`msg.rs`)

**Files:**
- Modify: `crates/heartbit-tui/src/msg.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `msg.rs`:

```rust
    #[test]
    fn retry_attempt_becomes_a_notice() {
        let ev = AgentEvent::RetryAttempt {
            agent: "(provider)".into(),
            attempt: 2,
            max_retries: 3,
            delay_ms: 1500,
            error_class: "rate_limited".into(),
        };
        match Msg::from_event(ev) {
            Some(Msg::Notice(n)) => {
                assert!(n.contains("2/3"), "got: {n}");
                assert!(n.contains("rate_limited"), "got: {n}");
            }
            _ => panic!("expected Notice"),
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p heartbit-tui msg::tests::retry_attempt_becomes_a_notice`
Expected: FAIL — `RetryAttempt` falls into `_ => None`.

- [ ] **Step 3: Add the mapping**

In `msg.rs` `Msg::from_event`, before the `_ => None` arm:

```rust
            AgentEvent::RetryAttempt {
                attempt,
                max_retries,
                delay_ms,
                error_class,
                ..
            } => Some(Msg::Notice(format!(
                "LLM retry {attempt}/{max_retries} in {delay_ms}ms ({error_class})"
            ))),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p heartbit-tui msg::`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-tui/src/msg.rs
git commit -m "feat(tui): surface LLM retry attempts as transcript notices"
```

---

### Task 5: Wire the trace into `main.rs` (writer, taps, bridge, retry)

This is the I/O edge task — per project pattern (`I/O only in main.rs`), it is
validated by compilation, the existing test suite, the gate, and the live smoke
in Task 10. The pure helpers it relies on were tested in Tasks 1–3.

**Files:**
- Modify: `crates/heartbit-tui/src/main.rs`
- Modify: `crates/heartbit-tui/src/trace.rs` (one path helper + test)
- Modify: `crates/heartbit-tui/src/app.rs` (one pure helper + test)

- [ ] **Step 1: Add the trace-path helper + failing test (trace.rs)**

```rust
/// The trace file for a session id, under the sessions dir.
pub fn trace_path(dir: &std::path::Path, session_id: &str) -> PathBuf {
    dir.join(format!("{session_id}.trace.jsonl"))
}
```

Test (append to `trace.rs` tests):

```rust
    #[test]
    fn trace_path_is_sessions_id_trace_jsonl() {
        let p = trace_path(std::path::Path::new("/tmp/sessions"), "abc-1");
        assert_eq!(p, std::path::PathBuf::from("/tmp/sessions/abc-1.trace.jsonl"));
    }
```

Run: `cargo test -p heartbit-tui trace::tests::trace_path` → FAIL, implement, PASS.

- [ ] **Step 2: Add `Effect::name()` + failing test (app.rs)**

In `app.rs`, add an inherent impl right below the `Effect` enum:

```rust
impl Effect {
    /// Stable name for the trace (`ui` `effect` records).
    pub fn name(&self) -> &'static str {
        match self {
            Effect::SendInput(_) => "send_input",
            Effect::SaveKey(_) => "save_key",
            Effect::SaveModel(_) => "save_model",
            Effect::SaveMcp(_) => "save_mcp",
            Effect::FetchModels => "fetch_models",
            Effect::WalkFiles => "walk_files",
            Effect::SaveContextRecall(_) => "save_context_recall",
            Effect::SaveVerifyCommand(_) => "save_verify_command",
            Effect::SetPermissionMode(_) => "set_permission_mode",
            Effect::ExportSession => "export_session",
            Effect::ListSessions => "list_sessions",
            Effect::ResumeSession(_) => "resume_session",
            Effect::Interrupt => "interrupt",
            Effect::Quit => "quit",
        }
    }
}
```

Test (in `app.rs` tests module):

```rust
    #[test]
    fn effect_names_are_stable_snake_case() {
        assert_eq!(Effect::FetchModels.name(), "fetch_models");
        assert_eq!(Effect::SendInput("x".into()).name(), "send_input");
        assert_eq!(Effect::Interrupt.name(), "interrupt");
    }
```

Run: `cargo test -p heartbit-tui app::tests::effect_names` → FAIL, implement, PASS.
(Task 7/8 add two more variants; their arms join this match then.)

- [ ] **Step 3: Rework startup order + tracing init in `main()`**

Replace `init_debug_logging()` (main.rs:57-84) with a version that keeps the
legacy file layer AND registers the bridge, and reorder `main()` so the
session id + writer exist first. The new shape:

```rust
/// Initialize tracing: the always-on trace bridge (heartbit::interrupt →
/// `core_trace` records) plus the legacy opt-in debug file
/// (`HEARTBIT_TUI_DEBUG=1` → /tmp/heartbit-tui-debug.log — unchanged, additive).
fn init_tracing(trace_handle: trace::TraceHandle) {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let target_filter = || {
        tracing_subscriber::filter::Targets::new()
            .with_target(trace::INTERRUPT_TARGET, tracing::level_filters::LevelFilter::INFO)
    };
    let bridge = trace::TracingBridge::new(trace_handle).with_filter(target_filter());
    let legacy = legacy_debug_file().map(|file| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(file))
            .with_filter(target_filter())
    });
    let _ = tracing_subscriber::registry()
        .with(bridge)
        .with(legacy)
        .try_init();
    tracing::info!(target: "heartbit::interrupt", "--- heartbit-tui debug logging started ---");
}

/// The legacy `HEARTBIT_TUI_DEBUG` file, if requested (same semantics as before).
fn legacy_debug_file() -> Option<std::fs::File> {
    let val = std::env::var("HEARTBIT_TUI_DEBUG").ok()?;
    if val.is_empty() {
        return None;
    }
    let path = if val == "1" || val == "true" {
        "/tmp/heartbit-tui-debug.log".to_string()
    } else {
        val
    };
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok()
}
```

In `main()`: move the `session_id` block (currently main.rs:134-141) to the TOP
of `main()` (before `init_debug_logging()` is called today), then create the
writer and init tracing:

```rust
    // A per-launch session id (time + pid) — also keys the execution trace.
    let session_id = format!(
        "{:x}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        std::process::id()
    );
    // Always-on execution trace (one JSONL per launch). Write errors self-
    // disable tracing and surface ONE notice — never take down the session.
    let (ui_tx, ui_rx) = tokio::sync::mpsc::unbounded_channel::<Msg>();
    let trace_handle = trace::spawn_writer(
        trace::trace_path(&session::sessions_dir(), &session_id),
        Box::new({
            let tx = ui_tx.clone();
            move |e| {
                let _ = tx.send(Msg::Notice(e));
            }
        }),
    );
    init_tracing(trace_handle.clone());
```

(The `ui_tx`/`ui_rx` creation moves UP from its current spot at main.rs:104 so
the error callback can use it; delete the old line. Remove the old
`init_debug_logging()` definition and call.)

After `app` is fully configured (after the `app.modal` block, ~main.rs:127),
record the session snapshot:

```rust
    trace_handle.record_ui(&trace::UiEvent::SessionStarted {
        version: env!("CARGO_PKG_VERSION").into(),
        session_id: session_id.clone(),
        model: app.model.clone(),
        permission_mode: app.permission_mode.label().to_lowercase(),
        mcp_servers: cfg.mcp_servers.iter().map(|s| s.label()).collect(),
        context_recall: app.context_recall,
        verify_command: app.verify_command.clone(),
    });
```

(If `PermissionMode::label()` returns capitalized text, `to_lowercase()`
normalizes; verify against `app.rs` — labels are short like "Normal"/"YOLO".
If `McpServerSpec::label()` returns `String`, the `map` collect works as-is.)

Pass `trace_handle` into `run_ui` (new last parameter, `trace: trace::TraceHandle`).

- [ ] **Step 4: Thread the handle through `spawn_agent` and `build_engine`**

`spawn_agent` (main.rs:623): add params `trace: &trace::TraceHandle, reason: &'static str`; record the spawn before `std::thread::spawn`:

```rust
    trace.record_ui(&trace::UiEvent::AgentSpawned {
        epoch,
        model: app.model.clone(),
        reason: reason.into(),
        context_recall: app.context_recall,
        verify_command: app.verify_command.clone(),
    });
    let trace = trace.clone();
```

…and pass `trace` (the clone) into `build_engine` (new parameter). Update the
3 call sites with reasons: eager startup in `run_ui` → `"startup"`;
`Effect::SendInput` respawn → `"respawn"`; `Effect::SaveKey` → `"key_set"`.

`build_engine` (main.rs:316): add param `trace: trace::TraceHandle`. Two closure changes:

**(a) `on_event` (main.rs:389-407)** — tap the raw event FIRST (before
`Msg::from_event`, which drops/transforms; lossless requirement):

```rust
    let on_event: Arc<OnEvent> = {
        let tx = ui_tx.clone();
        let trace_events = trace.clone();
        Arc::new(move |e: AgentEvent| {
            // Lossless trace tap — BEFORE Msg::from_event (which drops a subset).
            trace_events.record_agent(&e);
            // Legacy opt-in diagnostics (HEARTBIT_TUI_DEBUG) — unchanged.
            if let AgentEvent::ToolCallStarted {
                tool_name, agent, ..
            } = &e
            {
                tracing::info!(target: "heartbit::interrupt", tool_started = %tool_name, agent = %agent, "tool dispatched");
            }
            if let AgentEvent::SubAgentsDispatched { agents, .. } = &e {
                tracing::info!(target: "heartbit::interrupt", ?agents, "sub-agents dispatched");
            }
            if let Some(m) = Msg::from_event(e) {
                let _ = tx.send(m);
            }
        })
    };
```

(Known, accepted redundancy: those two legacy `tracing::info!` lines ALSO land
as `core_trace` records via the bridge. The canonical typed stream is
`agent`/`ui`; `core_trace` is a raw mirror that `trace_stats` never parses.)

**(b) `on_approval` (main.rs:408-443)** — record every gate resolution with
human latency and the mode in effect:

```rust
    let on_approval: Arc<OnApproval> = {
        let tx = ui_tx.clone();
        let perm_mode = perm_mode.clone();
        let trace_approvals = trace.clone();
        Arc::new(move |calls: &[heartbit_core::llm::types::ToolCall]| {
            let started = std::time::Instant::now();
            let names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
            for c in calls {
                tracing::info!(target: "heartbit::interrupt", approval_for = %c.name, "on_approval");
            }
            let record = |decision: &ApprovalDecision, mode: &str, latency_ms: u64| {
                trace_approvals.record_ui(&trace::UiEvent::Approval {
                    tools: names.clone(),
                    decision: trace::decision_label(decision).into(),
                    latency_ms,
                    mode: mode.into(),
                });
            };
            let is_mutating = |n: &str| matches!(n, "edit" | "write" | "patch" | "bash");
            match perm_mode.load(std::sync::atomic::Ordering::Relaxed) {
                2 => {
                    record(&ApprovalDecision::Allow, "yolo", 0);
                    return ApprovalDecision::Allow;
                }
                1 if calls.iter().any(|c| is_mutating(&c.name)) => {
                    record(&ApprovalDecision::Deny, "plan", 0);
                    return ApprovalDecision::Deny;
                }
                _ => {}
            }
            let tools = calls
                .iter()
                .map(|c| PendingTool {
                    name: c.name.clone(),
                    input: serde_json::to_string(&c.input).unwrap_or_default(),
                })
                .collect();
            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
            if tx
                .send(Msg::Approval {
                    tools,
                    reply: reply_tx,
                })
                .is_err()
            {
                record(&ApprovalDecision::Deny, "normal", 0);
                return ApprovalDecision::Deny;
            }
            let decision = reply_rx.recv().unwrap_or(ApprovalDecision::Deny);
            record(
                &decision,
                "normal",
                started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            );
            decision
        })
    };
```

**(c) RetryAttempt wiring** — in `build_engine`, the provider is currently
built FIRST (line 329) but the retry callback needs `on_event`. Reorder: move
the `let on_event…` block ABOVE the provider construction, then:

```rust
    // Wire RetryAttempt emission (mirrors heartbit-cli's build_on_retry):
    // retries are diagnostic gold and were previously invisible in the TUI.
    let on_retry: Arc<heartbit_core::OnRetry> = {
        let cb = on_event.clone();
        Arc::new(
            move |attempt: u32, max_retries: u32, delay_ms: u64, error_class: &str| {
                cb(AgentEvent::RetryAttempt {
                    agent: "(provider)".into(),
                    attempt,
                    max_retries,
                    delay_ms,
                    error_class: error_class.to_string(),
                });
            },
        )
    };
    let provider = build_provider(api_key, model, on_retry)?;
```

…and change `build_provider` (main.rs:168) to accept and attach it at BOTH sites:

```rust
fn build_provider(
    openrouter_key: Option<String>,
    model: &str,
    on_retry: Arc<heartbit_core::OnRetry>,
) -> anyhow::Result<Arc<BoxedProvider>> {
    if let Some(key) = openrouter_key {
        let base = OpenRouterProvider::new(key, model);
        return Ok(Arc::new(BoxedProvider::new(
            RetryingProvider::with_defaults(base).with_on_retry(on_retry),
        )));
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY")
        && !key.is_empty()
    {
        let anthropic_model = if model.contains('/') {
            "claude-sonnet-4-6"
        } else {
            model
        };
        let base = heartbit_core::AnthropicProvider::new(&key, anthropic_model);
        return Ok(Arc::new(BoxedProvider::new(
            RetryingProvider::with_defaults(base).with_on_retry(on_retry),
        )));
    }
    anyhow::bail!("no OpenRouter API key configured (set one with /key or OPENROUTER_API_KEY)")
}
```

- [ ] **Step 5: Record UI events at the effect-dispatch edge (`run_ui`)**

In the `for effect in std::mem::take(&mut app.effects)` loop (main.rs:761),
wrap each effect with name + duration and add per-arm records:

```rust
        for effect in std::mem::take(&mut app.effects) {
            let effect_name = effect.name();
            let effect_started = std::time::Instant::now();
            match effect {
                // …existing arms, with these ADDITIONS:
```

- `Effect::SendInput(text)` arm — first line:
  ```rust
                    trace.record_ui(&trace::UiEvent::UserInput { text: text.clone() });
  ```
- `Effect::SetPermissionMode(m)` arm — replace body with:
  ```rust
                    let old = perm_mode.swap(m, std::sync::atomic::Ordering::Relaxed);
                    if old != m {
                        trace.record_ui(&trace::UiEvent::ModeChanged {
                            from: trace::mode_label(old).into(),
                            to: trace::mode_label(m).into(),
                        });
                    }
  ```
- `Effect::ResumeSession(id)` arm — in the `Ok(s)` branch, before sending:
  ```rust
                        trace.record_ui(&trace::UiEvent::SessionResumed {
                            from_id: id.clone(),
                        });
  ```
- `Effect::Interrupt` arm — mirror CP1/CP2 as typed records (the existing
  `tracing::info!` lines stay; they feed the legacy file + `core_trace`):
  ```rust
                    trace.record_ui(&trace::UiEvent::InterruptRequested {
                        checkpoint: "cp1_effect_dequeued".into(),
                        running: app.running,
                    });
                    // …existing tracing::info! CP1 + interrupt.interrupt() + CP2…
                    trace.record_ui(&trace::UiEvent::InterruptRequested {
                        checkpoint: "cp2_handle_interrupted".into(),
                        running: app.running,
                    });
  ```
- After the whole `match effect { … }` block (still inside the `for`):
  ```rust
            trace.record_ui(&trace::UiEvent::Effect {
                name: effect_name.into(),
                duration_ms: effect_started.elapsed().as_millis() as u64,
            });
  ```

Auto-save errors become visible — change `save_session` to return the result:

```rust
/// Persist the current transcript under the session id (errors surface in the trace).
fn save_session(id: &str, history: &[crate::cells::Cell]) -> std::io::Result<()> {
    let s = session::Session {
        id: id.to_string(),
        created: id.to_string(),
        history: history.to_vec(),
    };
    session::save(&session::sessions_dir(), &s)
}
```

…and at BOTH call sites (idle auto-save ~main.rs:909 and quit ~main.rs:915):

```rust
            if let Err(e) = save_session(&session_id, &app.history) {
                trace.record_ui(&trace::UiEvent::Error {
                    context: "session_save".into(),
                    message: e.to_string(),
                });
            }
```

- [ ] **Step 6: Compile + full crate tests**

Run: `cargo build -p heartbit-tui && cargo test -p heartbit-tui`
Expected: clean build, all tests PASS. Then `cargo clippy -p heartbit-tui --all-targets -- -D warnings` — fix `too_many_arguments` by extending the existing `#[allow(clippy::too_many_arguments)]` (already on `build_engine`, `spawn_agent`, `run_ui`). Remove any `#[allow(dead_code)]` left from Task 1.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-tui/src/main.rs crates/heartbit-tui/src/trace.rs crates/heartbit-tui/src/app.rs
git commit -m "feat(tui): always-on execution trace — agent/ui/core taps + RetryAttempt wiring"
```

---

### Task 6: `trace_stats.rs` — deterministic stats pre-pass

**Files:**
- Create: `crates/heartbit-tui/src/trace_stats.rs`
- Modify: `crates/heartbit-tui/src/main.rs` (one `mod trace_stats;` line)

- [ ] **Step 1: Write the failing tests**

Create `crates/heartbit-tui/src/trace_stats.rs` with the test module first:

```rust
//! Deterministic stats over a trace JSONL: streams line-by-line (never loads
//! the whole file), tolerant of torn/malformed lines and unknown event types.
//! This is the human `/stats` summary AND the measurement substrate the
//! self-improvement ladder builds on. Reads ONLY `agent` and `ui` records —
//! `core_trace` is a raw mirror, never parsed.

use std::collections::BTreeMap;
use std::io::BufRead;

use serde::Serialize;

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic-but-valid trace (the golden fixture).
    fn fixture() -> String {
        let lines = [
            r#"{"v":1,"seq":0,"ts":"2026-06-06T10:00:00.000Z","src":"ui","event":{"type":"session_started","version":"0.1.0","session_id":"s1","model":"m","permission_mode":"normal","mcp_servers":[],"context_recall":true,"verify_command":null}}"#,
            r#"{"v":1,"seq":1,"ts":"2026-06-06T10:00:01.000Z","src":"ui","event":{"type":"user_input","text":"do the thing"}}"#,
            r#"{"v":1,"seq":2,"ts":"2026-06-06T10:00:01.100Z","src":"agent","event":{"type":"turn_started","agent":"entry","turn":1,"max_turns":300}}"#,
            r#"{"v":1,"seq":3,"ts":"2026-06-06T10:00:03.000Z","src":"agent","event":{"type":"llm_response","agent":"entry","turn":1,"usage":{"input_tokens":1000,"output_tokens":50},"stop_reason":"tool_use","tool_call_count":1,"latency_ms":1900,"time_to_first_token_ms":400}}"#,
            r#"{"v":1,"seq":4,"ts":"2026-06-06T10:00:03.100Z","src":"ui","event":{"type":"approval","tools":["bash"],"decision":"allow","latency_ms":2500,"mode":"normal"}}"#,
            r#"{"v":1,"seq":5,"ts":"2026-06-06T10:00:06.000Z","src":"agent","event":{"type":"tool_call_completed","agent":"entry","tool_name":"bash","tool_call_id":"t1","is_error":false,"duration_ms":300,"output":"ok"}}"#,
            r#"{"v":1,"seq":6,"ts":"2026-06-06T10:00:06.500Z","src":"agent","event":{"type":"retry_attempt","agent":"(provider)","attempt":1,"max_retries":3,"delay_ms":1000,"error_class":"rate_limited"}}"#,
            r#"{"v":1,"seq":7,"ts":"2026-06-06T10:00:08.000Z","src":"agent","event":{"type":"turn_started","agent":"entry","turn":2,"max_turns":300}}"#,
            r#"{"v":1,"seq":8,"ts":"2026-06-06T10:00:09.000Z","src":"agent","event":{"type":"llm_response","agent":"entry","turn":2,"usage":{"input_tokens":1200,"output_tokens":80},"stop_reason":"end_turn","tool_call_count":0,"latency_ms":900,"time_to_first_token_ms":200}}"#,
            r#"{"v":1,"seq":9,"ts":"2026-06-06T10:00:09.100Z","src":"agent","event":{"type":"tool_call_completed","agent":"entry","tool_name":"bash","tool_call_id":"t2","is_error":true,"duration_ms":700,"output":"boom"}}"#,
            r#"{"v":1,"seq":10,"ts":"2026-06-06T10:00:09.200Z","src":"ui","event":{"type":"interrupt_requested","checkpoint":"cp1_effect_dequeued","running":true}}"#,
            r#"{"v":1,"seq":11,"ts":"2026-06-06T10:00:09.300Z","src":"agent","event":{"type":"run_completed","agent":"entry","total_usage":{"input_tokens":2200,"output_tokens":130},"tool_calls_made":2}}"#,
            "{ this line is torn garba",
        ];
        lines.join("\n")
    }

    #[test]
    fn golden_fixture_computes_exact_stats() {
        let s = compute(fixture().as_bytes());
        assert_eq!(s.records, 12);
        assert_eq!(s.skipped_lines, 1);
        assert_eq!(s.user_inputs, 1);
        assert_eq!(s.turns, 2);
        assert_eq!(s.llm_calls, 2);
        assert_eq!(s.total_input_tokens, 2200);
        assert_eq!(s.total_output_tokens, 130);
        // sorted latencies [900, 1900]: p50 = 900 (nearest-rank), p95 = 1900
        assert_eq!(s.llm_latency_p50_ms, 900);
        assert_eq!(s.llm_latency_p95_ms, 1900);
        assert_eq!(s.ttft_p50_ms, 200);
        let bash = s.tools.get("bash").expect("bash stats");
        assert_eq!(bash.count, 2);
        assert_eq!(bash.errors, 1);
        assert_eq!(bash.p50_ms, 300);
        assert_eq!(bash.p95_ms, 700);
        assert_eq!(s.retries, 1);
        assert_eq!(s.approvals, 1);
        assert_eq!(s.approval_denials, 0);
        assert_eq!(s.approval_mean_latency_ms, 2500);
        assert_eq!(s.interrupts, 1); // cp1 only — cp2 must not double-count
        assert_eq!(s.run_completed, 1);
        assert_eq!(s.run_failed, 0);
        assert_eq!(s.doom_loops, 0);
    }

    #[test]
    fn empty_input_yields_zeroed_stats() {
        let s = compute(&b""[..]);
        assert_eq!(s.records, 0);
        assert_eq!(s.turns, 0);
        assert_eq!(s.llm_latency_p50_ms, 0);
        assert!(s.tools.is_empty());
    }

    #[test]
    fn unknown_types_and_versions_are_counted_not_fatal() {
        let input = [
            r#"{"v":99,"seq":0,"ts":"t","src":"ui","event":{"type":"from_the_future"}}"#,
            r#"{"v":1,"seq":1,"ts":"t","src":"agent","event":{"type":"turn_started","agent":"a","turn":1,"max_turns":5}}"#,
        ]
        .join("\n");
        let s = compute(input.as_bytes());
        assert_eq!(s.records, 2); // parsed envelope = a record, even if unknown
        assert_eq!(s.turns, 1);
    }

    #[test]
    fn percentile_is_nearest_rank() {
        assert_eq!(pct(&[], 0.5), 0);
        assert_eq!(pct(&[10], 0.95), 10);
        assert_eq!(pct(&[10, 20, 30, 40], 0.5), 20);
        assert_eq!(pct(&[10, 20, 30, 40], 0.95), 40);
    }

    #[test]
    fn render_is_a_readable_table() {
        let s = compute(fixture().as_bytes());
        let out = s.render();
        assert!(out.contains("turns"), "got: {out}");
        assert!(out.contains("bash"), "got: {out}");
        assert!(out.contains("2200"), "tokens visible: {out}");
        assert!(out.contains("retries"), "got: {out}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Add `mod trace_stats;` to `main.rs`'s module list, then:
Run: `cargo test -p heartbit-tui trace_stats::`
Expected: FAIL — `compute`, `pct`, `TraceStats` not defined.

- [ ] **Step 3: Implement**

Add above the tests module:

```rust
/// Per-tool aggregate.
#[derive(Debug, Default, Serialize)]
pub struct ToolStat {
    pub count: usize,
    pub errors: usize,
    pub p50_ms: u64,
    pub p95_ms: u64,
    #[serde(skip)]
    durations: Vec<u64>,
}

/// The deterministic summary of one trace file.
#[derive(Debug, Default, Serialize)]
pub struct TraceStats {
    pub records: usize,
    pub skipped_lines: usize,
    pub user_inputs: usize,
    pub turns: usize,
    pub llm_calls: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub llm_latency_p50_ms: u64,
    pub llm_latency_p95_ms: u64,
    pub ttft_p50_ms: u64,
    pub ttft_p95_ms: u64,
    pub tools: BTreeMap<String, ToolStat>,
    pub retries: usize,
    pub doom_loops: usize,
    pub guardrail_denied: usize,
    pub guardrail_warned: usize,
    pub approvals: usize,
    pub approval_denials: usize,
    pub approval_mean_latency_ms: u64,
    pub interrupts: usize,
    pub compactions: usize,
    pub prunes: usize,
    pub run_completed: usize,
    pub run_failed: usize,
}

/// Nearest-rank percentile over a SORTED slice (0 for empty).
fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

/// Stream a trace and aggregate. Tolerant by design: torn/malformed lines and
/// unknown event types/versions are counted, never fatal.
pub fn compute(reader: impl std::io::Read) -> TraceStats {
    let mut s = TraceStats::default();
    let mut llm_latencies: Vec<u64> = Vec::new();
    let mut ttfts: Vec<u64> = Vec::new();
    let mut approval_latencies: Vec<u64> = Vec::new();
    for line in std::io::BufReader::new(reader).lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<serde_json::Value>(&line) else {
            s.skipped_lines += 1;
            continue;
        };
        let (Some(src), Some(ev)) = (rec["src"].as_str(), rec.get("event")) else {
            s.skipped_lines += 1;
            continue;
        };
        s.records += 1;
        let ty = ev["type"].as_str().unwrap_or("");
        match (src, ty) {
            ("ui", "user_input") => s.user_inputs += 1,
            ("ui", "approval") => {
                s.approvals += 1;
                if ev["decision"].as_str().unwrap_or("").contains("deny") {
                    s.approval_denials += 1;
                }
                approval_latencies.push(ev["latency_ms"].as_u64().unwrap_or(0));
            }
            ("ui", "interrupt_requested") => {
                // cp1 only — cp2 mirrors the same user action.
                if ev["checkpoint"].as_str().unwrap_or("").starts_with("cp1") {
                    s.interrupts += 1;
                }
            }
            ("agent", "turn_started") => s.turns += 1,
            ("agent", "llm_response") => {
                s.llm_calls += 1;
                s.total_input_tokens += ev["usage"]["input_tokens"].as_u64().unwrap_or(0);
                s.total_output_tokens += ev["usage"]["output_tokens"].as_u64().unwrap_or(0);
                llm_latencies.push(ev["latency_ms"].as_u64().unwrap_or(0));
                let ttft = ev["time_to_first_token_ms"].as_u64().unwrap_or(0);
                if ttft > 0 {
                    ttfts.push(ttft);
                }
            }
            ("agent", "tool_call_completed") => {
                let name = ev["tool_name"].as_str().unwrap_or("?").to_string();
                let stat = s.tools.entry(name).or_default();
                stat.count += 1;
                if ev["is_error"].as_bool().unwrap_or(false) {
                    stat.errors += 1;
                }
                stat.durations.push(ev["duration_ms"].as_u64().unwrap_or(0));
            }
            ("agent", "retry_attempt") => s.retries += 1,
            ("agent", "doom_loop_detected") | ("agent", "fuzzy_doom_loop_detected") => {
                s.doom_loops += 1;
            }
            ("agent", "guardrail_denied") => s.guardrail_denied += 1,
            ("agent", "guardrail_warned") => s.guardrail_warned += 1,
            ("agent", "auto_compaction_triggered") | ("agent", "context_summarized") => {
                s.compactions += 1;
            }
            ("agent", "session_pruned") => s.prunes += 1,
            ("agent", "run_completed") => s.run_completed += 1,
            ("agent", "run_failed") => s.run_failed += 1,
            _ => {} // unknown type/src: counted in records, otherwise ignored
        }
    }
    llm_latencies.sort_unstable();
    ttfts.sort_unstable();
    s.llm_latency_p50_ms = pct(&llm_latencies, 0.5);
    s.llm_latency_p95_ms = pct(&llm_latencies, 0.95);
    s.ttft_p50_ms = pct(&ttfts, 0.5);
    s.ttft_p95_ms = pct(&ttfts, 0.95);
    if !approval_latencies.is_empty() {
        s.approval_mean_latency_ms =
            approval_latencies.iter().sum::<u64>() / approval_latencies.len() as u64;
    }
    for stat in s.tools.values_mut() {
        stat.durations.sort_unstable();
        stat.p50_ms = pct(&stat.durations, 0.5);
        stat.p95_ms = pct(&stat.durations, 0.95);
    }
    s
}

impl TraceStats {
    /// A fixed-width text table (rendered into the transcript in a code fence).
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "records {:>6}   skipped {:>3}   user msgs {:>4}\n",
            self.records, self.skipped_lines, self.user_inputs
        ));
        out.push_str(&format!(
            "turns   {:>6}   llm calls {:>4}   completed {} / failed {}\n",
            self.turns, self.llm_calls, self.run_completed, self.run_failed
        ));
        out.push_str(&format!(
            "tokens  in {} / out {}\n",
            self.total_input_tokens, self.total_output_tokens
        ));
        out.push_str(&format!(
            "llm latency p50/p95  {}ms / {}ms   ttft p50/p95  {}ms / {}ms\n",
            self.llm_latency_p50_ms, self.llm_latency_p95_ms, self.ttft_p50_ms, self.ttft_p95_ms
        ));
        out.push_str(&format!(
            "friction: retries {}  doom-loops {}  guardrail deny/warn {}/{}  interrupts {}  compactions {}  prunes {}\n",
            self.retries,
            self.doom_loops,
            self.guardrail_denied,
            self.guardrail_warned,
            self.interrupts,
            self.compactions,
            self.prunes
        ));
        out.push_str(&format!(
            "approvals {} (denied {})  mean human latency {}ms\n",
            self.approvals, self.approval_denials, self.approval_mean_latency_ms
        ));
        if !self.tools.is_empty() {
            out.push_str("tools:\n");
            for (name, t) in &self.tools {
                out.push_str(&format!(
                    "  {:<14} ×{:<4} errors {:<3} p50/p95 {}ms/{}ms\n",
                    name, t.count, t.errors, t.p50_ms, t.p95_ms
                ));
            }
        }
        out
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p heartbit-tui trace_stats::`
Expected: all 5 PASS. (If the golden p50 assertion fails, check nearest-rank
math: for `[900,1900]`, rank = ceil(0.5·2)=1 → index 0 → 900. ✓)

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-tui/src/trace_stats.rs crates/heartbit-tui/src/main.rs
git commit -m "feat(tui): deterministic trace stats pre-pass (streaming, tolerant)"
```

---

### Task 7: `/stats` command

**Files:**
- Modify: `crates/heartbit-tui/src/app.rs`
- Modify: `crates/heartbit-tui/src/msg.rs`
- Modify: `crates/heartbit-tui/src/trace.rs` (target resolution helper)
- Modify: `crates/heartbit-tui/src/main.rs` (edge handling)

- [ ] **Step 1: Write the failing reducer tests (app.rs)**

In `app.rs` tests. House pattern (see `slash_verify_sets_and_clears_the_command`,
app.rs:1786): `typed(&mut app, "…")` only TYPES — follow it with
`app.update(key(KeyCode::Enter))` to submit; `keyed()` builds an app with an
API key:

```rust
    #[test]
    fn slash_stats_pushes_compute_effect() {
        let mut app = keyed();
        typed(&mut app, "/stats");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::ComputeStats(None)));
        typed(&mut app, "/stats last");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::ComputeStats(Some("last".into()))));
    }

    #[test]
    fn stats_ready_renders_into_transcript() {
        let mut app = keyed();
        app.update(Msg::StatsReady(Ok("turns 3\n".into())));
        assert!(matches!(app.history.last(), Some(Cell::Agent(t)) if t.contains("turns 3")));
        app.update(Msg::StatsReady(Err("no trace".into())));
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("no trace")));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p heartbit-tui app::tests::slash_stats`
Expected: FAIL — `Effect::ComputeStats` / `Msg::StatsReady` not defined.

- [ ] **Step 3: Implement reducer + msg sides**

`app.rs`:
- `SLASH_COMMANDS` — after the `/export` entry:
  ```rust
      ("/stats", "trace stats — this session, `last`, or <id>"),
  ```
- `Effect` enum — new variant:
  ```rust
      /// Compute trace stats (None = current session; "last" | <id> otherwise).
      ComputeStats(Option<String>),
  ```
  …and its `name()` arm: `Effect::ComputeStats(_) => "compute_stats",`
- `handle_slash` — before the `other =>` arm:
  ```rust
              "stats" => {
                  let target = if arg.is_empty() { None } else { Some(arg) };
                  self.effects.push(Effect::ComputeStats(target));
              }
  ```
- `update()` — handle the result (place near the `Msg::SessionsListed` arm):
  ```rust
              Msg::StatsReady(Ok(table)) => {
                  self.history
                      .push(Cell::Agent(format!("```\n{table}```")));
              }
              Msg::StatsReady(Err(e)) => {
                  self.history.push(Cell::Notice(format!("stats: {e}")));
              }
  ```

`msg.rs` — new variant (near `SessionsListed`):

```rust
    /// Trace stats computed (rendered table) — or why they couldn't be.
    StatsReady(Result<String, String>),
```

- [ ] **Step 4: Target resolution helper + failing test (trace.rs)**

```rust
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
```

Run (FAIL), then implement in `trace.rs`:

```rust
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
                if !name.ends_with(".trace.jsonl")
                    || name == format!("{current_id}.trace.jsonl")
                {
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
```

Run: `cargo test -p heartbit-tui trace::tests::resolve_target` → PASS.

- [ ] **Step 5: Edge handling (main.rs)**

In the effect loop, new arm (mirror the `Effect::WalkFiles` spawn_blocking pattern):

```rust
                Effect::ComputeStats(target) => {
                    let tx = ui_tx.clone();
                    let sid = session_id.clone();
                    tokio::spawn(async move {
                        let result = tokio::task::spawn_blocking(move || {
                            let dir = session::sessions_dir();
                            let path =
                                trace::resolve_trace_target(&dir, &sid, target.as_deref())?;
                            let file =
                                std::fs::File::open(&path).map_err(|e| e.to_string())?;
                            Ok::<String, String>(trace_stats::compute(file).render())
                        })
                        .await
                        .unwrap_or_else(|e| Err(e.to_string()));
                        let _ = tx.send(Msg::StatsReady(result));
                    });
                }
```

- [ ] **Step 6: Run the full crate tests**

Run: `cargo test -p heartbit-tui && cargo clippy -p heartbit-tui --all-targets -- -D warnings`
Expected: PASS, no warnings.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-tui/src/app.rs crates/heartbit-tui/src/msg.rs crates/heartbit-tui/src/trace.rs crates/heartbit-tui/src/main.rs
git commit -m "feat(tui): /stats — deterministic trace stats in the transcript"
```

---

### Task 8: `/analyze` command — agent-driven diagnosis

**Files:**
- Modify: `crates/heartbit-tui/src/trace.rs` (prompt builder)
- Modify: `crates/heartbit-tui/src/app.rs`
- Modify: `crates/heartbit-tui/src/msg.rs`
- Modify: `crates/heartbit-tui/src/main.rs`

**CRITICAL path constraint (verified `builtins/mod.rs:225-238`):** with
`workspace = Some(cwd)` — which the TUI sets — the `read`/`grep`/`write`
builtins REJECT absolute paths. The agent cannot touch
`~/.config/heartbit/sessions/` directly. Therefore the edge STAGES a snapshot
copy of the trace into cwd (`heartbit-trace-<id>.jsonl`) before sending the
prompt, and the diagnosis is written to cwd (`heartbit-diagnosis-<id>.md`,
following the `/export`-writes-to-cwd precedent, main.rs:825). Bonus: the
snapshot freezes the current session's trace, so the agent's greps don't race
the live writer. And since `read`/`grep` carry silent Allow permission rules,
investigation has ZERO approval friction (only the final `write` prompts once
in Normal mode).

- [ ] **Step 1: Write the failing prompt-builder test (trace.rs)**

```rust
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
```

Run: `cargo test -p heartbit-tui trace::tests::analyze_prompt` → FAIL.

- [ ] **Step 2: Implement the prompt builder (trace.rs)**

```rust
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
```

Run: `cargo test -p heartbit-tui trace::tests::analyze_prompt` → PASS.

- [ ] **Step 3: Write the failing reducer tests (app.rs)**

```rust
    #[test]
    fn slash_analyze_pushes_analyze_effect() {
        let mut app = keyed();
        typed(&mut app, "/analyze");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::Analyze(None)));
        typed(&mut app, "/analyze last");
        app.update(key(KeyCode::Enter));
        assert!(app.effects.contains(&Effect::Analyze(Some("last".into()))));
    }

    #[test]
    fn analyze_ready_starts_a_run_with_the_task() {
        let mut app = keyed();
        app.update(Msg::AnalyzeReady {
            display: "analyzing session s1".into(),
            task: "the big prompt".into(),
        });
        assert!(matches!(app.history.last(), Some(Cell::User(t)) if t.contains("s1")));
        assert!(app.running);
        assert!(app.effects.contains(&Effect::SendInput("the big prompt".into())));
    }

    #[test]
    fn analyze_failed_is_a_notice_not_a_run() {
        let mut app = keyed();
        app.update(Msg::AnalyzeFailed("no trace".into()));
        assert!(!app.running);
        assert!(matches!(app.history.last(), Some(Cell::Notice(n)) if n.contains("no trace")));
    }
```

Run → FAIL.

- [ ] **Step 4: Implement reducer + msg sides**

`msg.rs` — new variants:

```rust
    /// `/analyze` prepared: show `display` as the user cell, send `task` to
    /// the agent (the Plan-mode `sent ≠ displayed` precedent).
    AnalyzeReady { display: String, task: String },
    /// `/analyze` could not prepare (no trace, stats error…).
    AnalyzeFailed(String),
```

`app.rs`:
- `SLASH_COMMANDS` — after `/stats`:
  ```rust
      ("/analyze", "agent diagnosis of a trace — this session, `last`, or <id>"),
  ```
- `Effect` variant + `name()` arm:
  ```rust
      /// Prepare an `/analyze` run (resolve trace, compute stats, build prompt).
      Analyze(Option<String>),
  ```
  `Effect::Analyze(_) => "analyze",`
- `handle_slash`:
  ```rust
              "analyze" => {
                  let target = if arg.is_empty() { None } else { Some(arg) };
                  self.effects.push(Effect::Analyze(target));
              }
  ```
- `update()` arms (the AnalyzeReady body mirrors `submit()`'s send path —
  user cell, running, follow, squad seed, SendInput):
  ```rust
              Msg::AnalyzeReady { display, task } => {
                  self.history.push(Cell::User(display));
                  self.running = true;
                  self.follow = true;
                  self.seed_idle_squad();
                  self.effects.push(Effect::SendInput(task));
              }
              Msg::AnalyzeFailed(e) => {
                  self.history.push(Cell::Notice(format!("analyze: {e}")));
              }
  ```

- [ ] **Step 5: Edge handling (main.rs)**

New effect arm:

```rust
                Effect::Analyze(target) => {
                    let tx = ui_tx.clone();
                    let sid = session_id.clone();
                    let workdir = cwd.clone();
                    tokio::spawn(async move {
                        let prepared = tokio::task::spawn_blocking(move || {
                            let dir = session::sessions_dir();
                            let path =
                                trace::resolve_trace_target(&dir, &sid, target.as_deref())?;
                            let file =
                                std::fs::File::open(&path).map_err(|e| e.to_string())?;
                            let stats = trace_stats::compute(file);
                            let stats_json = serde_json::to_string_pretty(&stats)
                                .map_err(|e| e.to_string())?;
                            let id = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .and_then(|n| n.strip_suffix(".trace.jsonl"))
                                .unwrap_or("session")
                                .to_string();
                            // Stage a snapshot into the workspace: the agent's
                            // builtins (read/grep/write) REJECT absolute paths
                            // when workspace-rooted, and the copy freezes the
                            // current session's still-growing trace.
                            let staged = format!("heartbit-trace-{id}.jsonl");
                            std::fs::copy(&path, workdir.join(&staged))
                                .map_err(|e| e.to_string())?;
                            Ok::<(String, String), String>((
                                format!("analyzing session {id} (staged: {staged})"),
                                trace::build_analyze_prompt(&staged, &id, &stats_json),
                            ))
                        })
                        .await
                        .unwrap_or_else(|e| Err(e.to_string()));
                        let _ = tx.send(match prepared {
                            Ok((display, task)) => Msg::AnalyzeReady { display, task },
                            Err(e) => Msg::AnalyzeFailed(e),
                        });
                    });
                }
```

- [ ] **Step 6: Run the full crate tests**

Run: `cargo test -p heartbit-tui && cargo clippy -p heartbit-tui --all-targets -- -D warnings`
Expected: PASS, no warnings. (Note: `seed_idle_squad` is private — if the
reducer arm can't call it from `update`, both live in `impl App`, so it can.)

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-tui/src/trace.rs crates/heartbit-tui/src/app.rs crates/heartbit-tui/src/msg.rs crates/heartbit-tui/src/main.rs
git commit -m "feat(tui): /analyze — agent-driven trace diagnosis (stats + targeted greps)"
```

---

### Task 9: Workspace quality gate

- [ ] **Step 1: Run the full gate**

```bash
cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm
```

Expected: all three pass, zero warnings. If fmt fails: `cargo fmt --all` and re-run.

- [ ] **Step 2: Commit any gate fixes**

```bash
git add -u && git commit -m "chore(tui): gate fixes for trace feature" # only if needed
```

---

### Task 10: Live validation (pty harness)

Per the project's pty-harness lesson: assert on the SETTLED final frame,
space-insensitive; force a repaint via RESIZE to capture the current frame;
strip non-letters when matching streamed text.

- [ ] **Step 1: Trace file populates during a real session**

Run a real session (needs `OPENROUTER_API_KEY` from env or config):

```bash
cargo build -p heartbit-tui
# In a Python pty harness (pattern from earlier TUI validations):
#  1. launch target/debug/heartbit-tui
#  2. send: "run the command: seq 1 5" → approve the bash modal with 'y'
#  3. wait for ready, then Ctrl+C
```

Then verify the trace (newest trace in the sessions dir):

```bash
T=$(ls -t ~/.config/heartbit/sessions/*.trace.jsonl | head -1)
head -1 "$T" | python3 -c "import json,sys; r=json.load(sys.stdin); assert r['v']==1 and r['event']['type']=='session_started', r; print('session_started OK')"
grep -c '"type":"user_input"' "$T"        # expect ≥ 1
grep -c '"type":"agent_spawned"' "$T"     # expect ≥ 1 (reason startup)
grep -c '"type":"llm_response"' "$T"      # expect ≥ 1
grep -c '"type":"tool_call_completed"' "$T"  # expect ≥ 1 (the bash call)
grep -c '"type":"approval"' "$T"          # expect ≥ 1 (the modal decision)
stat -c %a "$T"                            # expect 600
```

- [ ] **Step 2: `/stats` renders a table**

In a second pty session: type `/stats` + Enter, force resize repaint, assert the
de-ANSI'd settled frame contains `turns` and `llm latency` (space-insensitive).

- [ ] **Step 3: `/analyze` produces a diagnosis citing real facts**

In the same session: `/analyze last` + Enter → the agent runs (read/grep are
silent; approve the final write modal). Wait for ready. Assert (in the cwd
the TUI was launched from — staging + diagnosis both land there):

```bash
test -s heartbit-trace-*.jsonl && echo "trace staged into workspace OK"
D=$(ls -t heartbit-diagnosis-*.md | head -1)
test -s "$D" && grep -qi "seq" "$D" && echo "diagnosis exists and cites seq evidence"
# The bar: a cited seq must actually exist in the staged trace — pick one
# number the report cites and verify it:
S=$(grep -o 'seq[ =:]*[0-9]\+' "$D" | grep -o '[0-9]\+' | head -1)
grep -q "\"seq\":$S," heartbit-trace-*.jsonl && echo "cited seq $S is real OK"
```

The bar (per the MCP lesson): the diagnosis must cite **specific seq numbers /
tool names that exist in the trace file**. A generic essay is a FAIL.

- [ ] **Step 4: Legacy debug log unaffected**

```bash
HEARTBIT_TUI_DEBUG=1 # … one short pty session …
grep -q "debug logging started" /tmp/heartbit-tui-debug.log && echo "legacy log OK"
```

- [ ] **Step 5: Final commit**

```bash
git add -A && git status   # review — no stray files
git commit -m "test(tui): live-validate execution trace + /stats + /analyze" # if validation artifacts/fixes
```

---

## Self-review (run after writing, fix inline)

1. **Spec coverage:** envelope v1 ✓ (T1) · writer thread/0600/self-disable ✓ (T2) · core_trace bridge subsumes CP3/CP4 ✓ (T3) · RetryAttempt wiring ✓ (T4+T5) · all UI event types from the spec table ✓ (T1/T5: session_started, user_input, agent_spawned, mode_changed, effect, approval+latency+mode, interrupt_requested, session_resumed, error) · lossless agent tap before `Msg::from_event` ✓ (T5) · no `on_text`/`on_reasoning` tracing ✓ (taps don't touch them) · stats dimensions ✓ (T6) · `/stats` ✓ (T7) · `/analyze` with stats-first + targeted greps + diagnosis.md ✓ (T8) · tolerant readers ✓ (T6) · legacy `HEARTBIT_TUI_DEBUG` untouched-but-additive ✓ (T5 `legacy_debug_file`) · live validation bar ✓ (T10).
2. **Known divergences (documented):** (a) `agent_spawned` carries `context_recall`/`verify_command` instead of the spec's `multi_agent` (the static flag no longer shapes the engine — unified entry agent). (b) The diagnosis is written to **cwd** (`heartbit-diagnosis-<id>.md`, `/export` precedent) over a **staged trace copy** (`heartbit-trace-<id>.jsonl`) — not "next to the trace" in the config dir as the spec first said: the workspace-rooted builtins reject absolute paths (`resolve_path`, verified `builtins/mod.rs:225-238`), and the snapshot also freezes a still-growing current-session trace. Spec updated to match.
3. **Type consistency check:** `TraceHandle.record_ui(&UiEvent)` / `record_agent(&AgentEvent)` used consistently in T5/T7/T8; `Effect::name()` (T5) gains arms in T7 (`compute_stats`) and T8 (`analyze`); `Msg::StatsReady(Result<String,String>)` matches the edge's `Ok(render())`/`Err(String)`; `resolve_trace_target(&dir, &sid, target.as_deref())` signature consistent between T7 and T8.
