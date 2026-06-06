# TUI Execution Trace + Diagnosis v1 — Design

**Date:** 2026-06-06
**Status:** Approved (brainstorming complete, advisor-reviewed)
**Scope:** `crates/heartbit-tui` only — **zero heartbit-core changes**

## Problem

The TUI has no persistent record of its internal behaviour. Debugging live issues
(interrupt chain, eager-spawn stale config, roster visibility) has required ad-hoc
instrumentation (`HEARTBIT_TUI_DEBUG=1` → `/tmp/heartbit-tui-debug.log`, interrupt
chain + 3 event types only). `--verbose` dumps events to stderr — unusable under the
TUI's alt-screen. Raw `AgentEvent`s are never persisted anywhere; the session JSON
(`Cell`s) is a lossy human-facing view, not a trace.

The long-term goal is a **self-improvement ladder**: the tool analyzes its own
execution traces and improves itself. That requires trace volume and a stable,
agent-consumable format from day one.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Trace consumer | Phased: human-readable first, agent-consumable format from day one |
| "Improves itself" end state | Staged ladder: ① diagnosis reports → ② learned rules/config tuning → ③ human-gated code patches. Each rung ships separately |
| Capture trigger | **Always-on** (like Claude Code transcripts). Debug mode later adds verbosity, it does not gate capture |
| This spec's scope | Rung 0 (trace) + Rung 1 (diagnosis v1: `/stats` + `/analyze`) |
| Approach | Event-spine, TUI-side tap (chosen over core-side `AuditTrail` backbone and tracing-subscriber JSON layer) |
| RetryAttempt | Wire it — TUI-side only (`with_on_retry`, mirroring the CLI's `build_on_retry`) |

**Why zero-core-change is load-bearing:** all 28 `AgentEvent` variants already reach
the TUI's `on_event` closure; `AgentEvent` is already `#[serde(tag = "type",
rename_all = "snake_case")]` (`events.rs:32`); the runner already emits
`ApprovalRequested`/`ApprovalDecision` (`runner.rs:1507-1557`); and
`RetryingProvider::with_on_retry` exists (`retry.rs:74`) with a proven CLI wiring
pattern (`heartbit-cli/src/main.rs:470-475`). Raw LLM *request* bodies are the only
thing a TUI tap cannot see — deferred by design (see Non-goals).

## Rung 0 — Always-on JSONL trace

### New module: `crates/heartbit-tui/src/trace.rs`

One trace file per launch: `~/.config/heartbit/sessions/<id>.trace.jsonl` — same id
as the session JSON, created `0600` at startup, append-only.

**Writer:** a dedicated writer thread behind an unbounded channel. Producers send
`TraceRecord`s; the thread serializes one JSON object per line, appends, flushes per
line. I/O never blocks the agent loop or the UI loop. On write error: tracing
self-disables for the session and emits one `Notice` — the trace must never take
down a session. Channel-send failures are ignored. Flush + close on drop/quit.

### Envelope (the format contract)

```json
{"v":1,"seq":42,"ts":"2026-06-06T12:34:56.789Z","src":"agent","event":{"type":"tool_call_started","tool_name":"bash","input":"…"}}
{"v":1,"seq":43,"ts":"…","src":"ui","event":{"type":"approval","tools":["bash"],"decision":"allow","latency_ms":3400,"mode":"normal"}}
{"v":1,"seq":44,"ts":"…","src":"core_trace","event":{"type":"log","target":"heartbit::interrupt","message":"CP3_tool_cancel_arm_fired","fields":{}}}
```

- `v` — schema version. The future agent-consumer depends on contract stability.
- `seq` — monotonic per session; orders records when timestamps collide.
- `src` — which tap produced the record.

### The three taps

**1. `src:"agent"`** — the raw `AgentEvent`, tapped inside the `on_event` closure
(`main.rs:389-407`) **before** `Msg::from_event` (which transforms and drops a
subset). Serde tagging is already in place; all 28 variants flow lossless.
`LlmResponse` is the text record — **never** trace `on_text`/`on_reasoning` deltas
(they fire per character; one file-write per character).

**2. `src:"ui"`** — what `AgentEvent` cannot see. Event types:

| type | fields | why |
|---|---|---|
| `session_started` | version, model, multi_agent, permission_mode, mcp_servers, session_id | config snapshot at launch |
| `user_input` | text | submitted messages aren't agent events |
| `agent_spawned` | epoch, model, multi_agent, reason: `startup\|respawn\|toggle` | the eager-spawn stale-config bug class |
| `mode_changed` | from, to | permission-mode context for approvals |
| `effect` | name, duration_ms | which effects run, how long they occupy the loop |
| `approval` | tools, decision, latency_ms, mode | human decision + think-time (rung-2 gold) |
| `interrupt_requested` | checkpoint: `cp1\|cp2` | TUI half of the interrupt chain |
| `session_resumed` | from_id | trace lineage across `/resume` |
| `error` | context, message | e.g. currently-silent session-save failures |

**3. `src:"core_trace"`** — a small `tracing::Layer` in `trace.rs` bridges core's
existing `heartbit::interrupt`-target log events (CP3/CP4) into the same channel.
The bridge is **always-on and additive**: in v1 the existing `HEARTBIT_TUI_DEBUG`
`/tmp` log mechanism is left untouched (it costs nothing); the trace file gets the
same events regardless of the env var. The debug-verbosity tier (next spec) will
give the env var its expanded meaning.

### RetryAttempt wiring

The TUI's two `RetryingProvider::with_defaults` sites (`main.rs:175,189`) gain
`.with_on_retry(...)` emitting `AgentEvent::RetryAttempt` into the same `on_event`
path — mirroring `heartbit-cli`'s `build_on_retry`. Retries become visible `agent`
records (they are diagnostic gold and currently invisible).

### Validation against real bugs (schema fitness test)

| Past bug | Visible in trace? |
|---|---|
| Eager-spawn stale config (`/agents on` mid-session no-op) | ✅ `agent_spawned{epoch, multi_agent, reason}` vs `mode_changed`/`effect` timing |
| Interrupt chain (mid-tool abort) | ✅ CP1/CP2 (`ui`) + CP3/CP4 (`core_trace`) in one ordered file |
| Roster "orchestrator+1" (same-name fan-out question) | ✅ `SubAgentsDispatched.agents` verbatim — finally answers `["worker"]` vs `["worker","worker"]` |
| Scroll drift | ❌ deliberately out — frame-level render state is a non-goal; TestBackend owns rendering |

## Rung 1 — Diagnosis v1

### `trace_stats.rs` — deterministic stats pre-pass

Pure Rust, **streams** the JSONL (never loads the whole file), skips
malformed/torn lines. Produces a serializable `TraceStats`:

- **Run shape:** turns, wall-clock duration, completed/failed/interrupted
- **Tokens:** in/out per turn + totals, compactions, session prunes
- **Latency:** LLM `latency_ms` p50/p95, TTFT p50/p95
- **Tools:** per-tool `{count, error_count, error_rate, duration p50/p95}`
- **Friction:** retries, doom-loop events, guardrail denials/warns, approvals
  `{count, denials, mean human latency_ms}`, interrupts

Rendered as a markdown table for humans; JSON for the agent. This is also the
measurement substrate rung 3 needs ("did the lesson reduce tool-error rate?").

### `/stats [last|<id>]`

Renders the stats table straight into the transcript. Zero LLM, instant.
Default target = current session so far; `last` = most recent previous session.

### `/analyze [last|<id>]`

Composes a task for the **regular agent** (no new engine):

1. Run the stats pre-pass.
2. Build a prompt from a **template const** containing: the stats JSON, the trace
   file path, the trace-format reference, jq/grep recipes, and the diagnosis
   dimensions (errors & root chains, doom loops, latency outliers, token waste,
   approval friction, interrupt causes).
3. Send through the normal `Effect::SendInput` path. The agent investigates
   *specific* spots with its existing read/grep/bash tools — it never slurps the
   whole file — then presents findings in the transcript **and** saves
   `<id>.diagnosis.md` next to the trace via its write tool (one approval prompt
   in Normal mode; none in YOLO).

**Why a prompt template, not a SKILL.md:** skills are progressive disclosure for
when the *agent* decides; here the *command* knows the guidance is needed, every
time. Trivially promotable to a skill later if other entry points appear.

## Error handling

- **Writer:** best-effort everywhere; self-disable on write error + one `Notice`.
- **Readers** (stats + agent): tolerant — skip unparseable lines; unknown
  `event.type` and unknown `v` are counted-and-skipped; torn final line expected
  (same stance as `RunJournal`).
- `/analyze` / `/stats` with no trace file → friendly notice. Empty trace → zeros.
- **Disk:** line size bounded by the existing 64KB event payload cap
  (`EVENT_MAX_PAYLOAD_BYTES`); no rotation in v1 (same policy as session JSONs).

## Testing (TDD)

- `trace.rs`: envelope round-trip (`AgentEvent` → record → JSON → parse →
  identical), seq monotonicity, writer-thread drain + flush on drop, 0600 perms,
  self-disable on write error, tracing-Layer bridge captures a
  `heartbit::interrupt` event.
- `trace_stats.rs`: golden test — synthetic JSONL fixture → exact expected
  `TraceStats` (p50/p95, error rates); malformed-line skip; empty file.
- `app.rs`: `/stats`, `/analyze` parsing → Effects; reducer tests for new notices.
- Retry wiring: closure emits `RetryAttempt` into the channel (mirrors CLI test).
- **Live validation bar** (pty harness, settled-frame assertions, space-insensitive):
  real session → trace file populated with `session_started` + agent events →
  `/stats` renders a table → `/analyze` produces a diagnosis citing real facts
  from the trace.

## Non-goals (this spec)

- **Raw LLM request bodies** (system prompt, full context). Next spec: a
  debug-verbosity tier via a file-backed `AuditTrail` in `Full` mode (which
  already holds untruncated requests) — *not* a new `AgentEvent` variant (would
  fight the 64KB cap and add enum churn for data only debug needs).
- Rungs 2–3: learned rules/config tuning, self-patching. Future specs; the
  `approval` records (human decision + latency) and `TraceStats` are designed as
  their inputs.
- Trace rotation/retention; CLI/daemon trace writers; frame-level UI state
  snapshots; OTLP changes.

## Ladder roadmap (context, not scope)

1. **Rung 0+1 (this spec):** trace + `/stats` + `/analyze`.
2. **Debug-verbosity tier:** file-backed `AuditTrail::Full` for raw requests.
3. **Rung 2:** diagnosis findings → institutional memory (`category:
   "improvement"`) → injected at startup (recall hook). Existing wired pieces:
   memory tools, shared memory, skill registry.
4. **Rung 3:** eval-measured improvement loop (`EvalComparison` deltas over
   `TraceStats`), then human-gated code patches.
