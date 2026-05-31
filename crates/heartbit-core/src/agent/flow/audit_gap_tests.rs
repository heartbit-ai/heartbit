//! Gap-closing regression tests for the dynamic-workflow combinator core
//! (flow audit, 2026-05-31).
//!
//! A multi-agent audit of `agent/flow/` found that several *correct* leaf
//! invariants had no test that would fail if they regressed. These tests pin
//! them end-to-end through the public `agent()` API, deliberately self-contained
//! (own provider mocks, public/`pub(crate)` API only) so they do not depend on
//! another module's private test helpers.
//!
//! Covered here:
//! - journal `occurrence` ordering wired through the leaf (two identical prompts
//!   replay their distinct outputs in order),
//! - a cancelled MISS does not poison the journal,
//! - the schema marker is part of the journal key (a no-schema entry never
//!   satisfies a same-prompt schema lookup),
//! - reordered-but-equal schema keys canonicalize to the same journal entry,
//! - the IN-FLIGHT cancel arm (cancel after the permit/backstop/budget, while
//!   the model call is awaiting) returns `Ok(None)`, records no spend, emits
//!   `AgentSkipped` not `AgentFinished`,
//! - a journal replay emits exactly one `AgentReplayed` and no
//!   `AgentStarted`/`AgentFinished`,
//! - the phase an agent is grouped under is snapshotted at construction and an
//!   explicit `.phase()` overrides it.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use crate::agent::flow::WorkflowCtx;
use crate::agent::flow::event::{self, OnWorkflowEvent, WorkflowEvent};
use crate::agent::flow::journal::ResumeMode;
use crate::agent::flow::{agent, pipeline, workflow};
use crate::agent::test_helpers::MockProvider;
use crate::llm::types::{
    CompletionRequest, CompletionResponse, ContentBlock, RESPOND_TOOL_NAME, StopReason, TokenUsage,
};
use crate::llm::{BoxedProvider, LlmProvider};

/// Build an event-capturing sink: a shared `Vec<WorkflowEvent>` plus the
/// `Arc<OnWorkflowEvent>` callback that appends to it.
fn event_sink() -> (Arc<Mutex<Vec<WorkflowEvent>>>, Arc<OnWorkflowEvent>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let cb: Arc<OnWorkflowEvent> = Arc::new(move |e: WorkflowEvent| {
        captured.lock().expect("events lock").push(e);
    });
    (events, cb)
}

/// Provider that signals when its `complete()` is entered, then blocks forever —
/// so a test can cancel the run WHILE the leaf is in-flight (after the
/// permit/backstop/budget were taken). The pending future is dropped on cancel.
struct NotifyProvider {
    entered: Arc<tokio::sync::Notify>,
}
impl LlmProvider for NotifyProvider {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, crate::error::Error> {
        self.entered.notify_one();
        std::future::pending::<()>().await;
        unreachable!("the run is cancelled out from under this future")
    }
    fn model_name(&self) -> Option<&str> {
        Some("notify-mock")
    }
}

/// Provider that counts how many times the model was actually called and returns
/// `live-{n}` text — lets a test prove a journal replay did NOT hit the model.
struct CountingProvider {
    calls: Arc<AtomicUsize>,
}
impl LlmProvider for CountingProvider {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, crate::error::Error> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CompletionResponse {
            content: vec![ContentBlock::Text {
                text: format!("live-{n}"),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
            model: None,
        })
    }
    fn model_name(&self) -> Option<&str> {
        Some("counting-mock")
    }
}

/// Provider that answers via the synthetic `__respond__` tool with a fixed
/// payload — mimics a model producing validated structured output.
struct RespondProvider {
    payload: serde_json::Value,
}
impl LlmProvider for RespondProvider {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, crate::error::Error> {
        Ok(CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "resp-1".into(),
                name: RESPOND_TOOL_NAME.into(),
                input: self.payload.clone(),
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage {
                input_tokens: 5,
                output_tokens: 5,
                ..Default::default()
            },
            model: None,
        })
    }
    fn model_name(&self) -> Option<&str> {
        Some("respond-mock")
    }
}

/// Provider that panics if the model is ever called — proves a code path took a
/// journal HIT (zero model work) rather than running live.
struct PanicProvider;
impl LlmProvider for PanicProvider {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, crate::error::Error> {
        panic!("provider must NOT be called on a journal hit");
    }
    fn model_name(&self) -> Option<&str> {
        Some("panic-mock")
    }
}

/// Build a journaled ctx over a `CountingProvider`.
fn counting_ctx(calls: &Arc<AtomicUsize>, path: &std::path::Path, mode: ResumeMode) -> WorkflowCtx {
    WorkflowCtx::builder(Arc::new(BoxedProvider::new(CountingProvider {
        calls: Arc::clone(calls),
    })))
    .journal(path, mode)
    .expect("open journal")
    .build()
    .expect("build ctx")
}

/// Two byte-identical `agent()` calls in one journaled run must take DISTINCT
/// occurrences (0 then 1) and, on resume, replay their distinct cached outputs
/// in the same order — proving the leaf wires `journal.next_occurrence` (not the
/// journal unit test in isolation). A bug pinning occurrence to 0 would replay
/// the first output twice and this test would catch it.
#[tokio::test]
async fn duplicate_prompts_replay_in_occurrence_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dup.jsonl");

    let calls1 = Arc::new(AtomicUsize::new(0));
    let ctx1 = counting_ctx(&calls1, &path, ResumeMode::Fresh);
    let a = agent(&ctx1, "dup").run().await.expect("run ok");
    let b = agent(&ctx1, "dup").run().await.expect("run ok");
    assert_eq!(a.as_deref(), Some("live-0"), "first call = occurrence 0");
    assert_eq!(b.as_deref(), Some("live-1"), "second call = occurrence 1");
    assert_eq!(calls1.load(Ordering::SeqCst), 2);

    let calls2 = Arc::new(AtomicUsize::new(0));
    let ctx2 = counting_ctx(&calls2, &path, ResumeMode::Resume);
    let a2 = agent(&ctx2, "dup").run().await.expect("run ok");
    let b2 = agent(&ctx2, "dup").run().await.expect("run ok");
    assert_eq!(a2.as_deref(), Some("live-0"), "first replay = occurrence 0");
    assert_eq!(
        b2.as_deref(),
        Some("live-1"),
        "second replay = occurrence 1"
    );
    assert_eq!(
        calls2.load(Ordering::SeqCst),
        0,
        "both replays must skip the provider"
    );
}

/// A cancelled MISS must not write anything to the journal — otherwise a later
/// resume would replay a bogus entry. The run is pre-cancelled (yields
/// `Ok(None)`); the resume of the same prompt must therefore be a live MISS, not
/// a replay.
#[tokio::test]
async fn cancelled_miss_does_not_append_to_journal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cancel.jsonl");

    let calls1 = Arc::new(AtomicUsize::new(0));
    let ctx1 = counting_ctx(&calls1, &path, ResumeMode::Fresh);
    ctx1.stop();
    let out = agent(&ctx1, "task").run().await.expect("run ok");
    assert!(out.is_none(), "cancelled run yields Ok(None)");
    assert_eq!(calls1.load(Ordering::SeqCst), 0, "provider not called");

    let calls2 = Arc::new(AtomicUsize::new(0));
    let ctx2 = counting_ctx(&calls2, &path, ResumeMode::Resume);
    let out2 = agent(&ctx2, "task").run().await.expect("run ok");
    assert_eq!(
        out2.as_deref(),
        Some("live-0"),
        "resume runs live (journal not poisoned)"
    );
    assert_eq!(
        calls2.load(Ordering::SeqCst),
        1,
        "the cancelled miss must not have been journaled"
    );
}

/// The schema is part of the journal content hash: a no-schema text entry must
/// NOT satisfy a same-prompt schema lookup (else a cached text answer would be
/// wrongly returned for a structured call). The schema call must MISS and run
/// live (the provider is invoked at least once).
#[tokio::test]
async fn schema_marker_changes_journal_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("schema.jsonl");

    let calls1 = Arc::new(AtomicUsize::new(0));
    let ctx1 = counting_ctx(&calls1, &path, ResumeMode::Fresh);
    agent(&ctx1, "p").run().await.expect("run ok");
    assert_eq!(calls1.load(Ordering::SeqCst), 1);

    let calls2 = Arc::new(AtomicUsize::new(0));
    let ctx2 = counting_ctx(&calls2, &path, ResumeMode::Resume);
    // Same prompt, but WITH a schema -> a different content hash -> a MISS, so
    // the provider IS called (regardless of how the live run then ends — a
    // CountingProvider returns text, not __respond__, which is fine: we only
    // assert the no-schema entry did not satisfy the schema lookup).
    let _res = agent(&ctx2, "p")
        .schema_value(json!({ "type": "object" }))
        .run()
        .await;
    assert!(
        calls2.load(Ordering::SeqCst) >= 1,
        "the schema call must MISS the no-schema journal entry and run live"
    );
}

/// A schema whose keys are byte-reordered-but-equal must canonicalize to the
/// SAME journal key as the original — proving the leaf hashes a *canonical*
/// schema, not its raw bytes. A resume with a `PanicProvider` proves the hit
/// takes zero model work. (The pre-existing `reordered_schema_keys_still_replay`
/// reused a byte-identical schema and so never exercised reordering.)
#[tokio::test]
async fn reordered_schema_keys_hit_same_journal_entry() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reorder.jsonl");
    let schema_a = json!({ "type": "object", "required": ["x"] });
    let schema_b = json!({ "required": ["x"], "type": "object" });

    let ctx1 = WorkflowCtx::builder(Arc::new(BoxedProvider::new(RespondProvider {
        payload: json!({ "x": 1 }),
    })))
    .journal(&path, ResumeMode::Fresh)
    .expect("open journal")
    .build()
    .expect("build ctx");
    let first = agent(&ctx1, "p")
        .schema_value(schema_a)
        .run()
        .await
        .expect("run ok");
    assert_eq!(first, Some(json!({ "x": 1 })));

    let ctx2 = WorkflowCtx::builder(Arc::new(BoxedProvider::new(PanicProvider)))
        .journal(&path, ResumeMode::Resume)
        .expect("open journal")
        .build()
        .expect("build ctx");
    let replayed = agent(&ctx2, "p")
        .schema_value(schema_b)
        .run()
        .await
        .expect("run ok");
    assert_eq!(
        replayed,
        Some(json!({ "x": 1 })),
        "reordered-but-equal schema keys must hit the same journal entry"
    );
}

/// The in-flight cancel arm (after permit + backstop + budget, while the model
/// call is awaiting) must return `Ok(None)`, record NO spend, emit
/// `AgentStarted` then `AgentSkipped` and NOT `AgentFinished` — yet the runaway
/// backstop counter has already counted the agent. This is the `biased`
/// `tokio::select!` arm; all other cancel tests pre-cancel and never reach it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inflight_cancel_returns_none_records_no_spend_and_emits_skipped() {
    let entered = Arc::new(tokio::sync::Notify::new());
    let provider = Arc::new(BoxedProvider::new(NotifyProvider {
        entered: Arc::clone(&entered),
    }));
    let (events, cb) = event_sink();
    let ctx = WorkflowCtx::builder(provider)
        .on_event(cb)
        .build()
        .expect("build ctx");

    let run_ctx = ctx.clone();
    let handle = tokio::spawn(async move { agent(&run_ctx, "task").label("late").run().await });

    // Wait until the leaf is genuinely in-flight: the permit/backstop/budget have
    // run, AgentStarted has been emitted, and the model call is now awaiting.
    entered.notified().await;
    ctx.stop();

    let out = handle.await.expect("join").expect("run ok");
    assert!(out.is_none(), "in-flight cancel must yield Ok(None)");
    assert_eq!(ctx.budget().spent(), 0, "a cancelled run records no spend");
    assert_eq!(
        ctx.spawned(),
        1,
        "permit + backstop ran before the cancel point"
    );

    let evs = events.lock().expect("events lock");
    assert!(
        evs.iter()
            .any(|e| matches!(e, WorkflowEvent::AgentStarted { .. })),
        "AgentStarted fires before the in-flight cancel: {evs:?}"
    );
    assert!(
        evs.iter()
            .any(|e| matches!(e, WorkflowEvent::AgentSkipped { .. })),
        "in-flight cancel emits AgentSkipped: {evs:?}"
    );
    assert!(
        !evs.iter()
            .any(|e| matches!(e, WorkflowEvent::AgentFinished { .. })),
        "a cancelled run must NOT emit AgentFinished: {evs:?}"
    );
}

/// A journal replay must emit exactly one `AgentReplayed` (carrying the cached
/// usage) and NEITHER `AgentStarted` NOR `AgentFinished` — it does zero work.
#[tokio::test]
async fn replay_emits_only_agent_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replay-ev.jsonl");

    // Seed the journal.
    let calls1 = Arc::new(AtomicUsize::new(0));
    let ctx1 = counting_ctx(&calls1, &path, ResumeMode::Fresh);
    agent(&ctx1, "p").run().await.expect("run ok");

    // Resume with an event sink.
    let (events, cb) = event_sink();
    let calls2 = Arc::new(AtomicUsize::new(0));
    let ctx2 = WorkflowCtx::builder(Arc::new(BoxedProvider::new(CountingProvider {
        calls: Arc::clone(&calls2),
    })))
    .journal(&path, ResumeMode::Resume)
    .expect("open journal")
    .on_event(cb)
    .build()
    .expect("build ctx");

    let out = agent(&ctx2, "p").run().await.expect("run ok");
    assert_eq!(out.as_deref(), Some("live-0"));
    assert_eq!(calls2.load(Ordering::SeqCst), 0, "replay calls no provider");

    let evs = events.lock().expect("events lock");
    let replayed = evs
        .iter()
        .filter(|e| matches!(e, WorkflowEvent::AgentReplayed { .. }))
        .count();
    let started = evs
        .iter()
        .filter(|e| matches!(e, WorkflowEvent::AgentStarted { .. }))
        .count();
    let finished = evs
        .iter()
        .filter(|e| matches!(e, WorkflowEvent::AgentFinished { .. }))
        .count();
    assert_eq!(replayed, 1, "exactly one AgentReplayed on replay: {evs:?}");
    assert_eq!(started, 0, "no AgentStarted on replay: {evs:?}");
    assert_eq!(finished, 0, "no AgentFinished on replay: {evs:?}");
}

/// The phase an agent is grouped under is snapshotted at CONSTRUCTION: a later
/// phase change (here, dropping the phase guard) must not retro-affect an
/// already-built call, and an explicit `.phase()` overrides the snapshot.
/// Verified via the `phase` field of the `AgentStarted` events.
#[tokio::test]
async fn phase_snapshot_captured_at_construction_and_overridable() {
    let (events, cb) = event_sink();
    let ctx = WorkflowCtx::builder(Arc::new(BoxedProvider::new(MockProvider::new(vec![
        MockProvider::text_response("a", 1, 1),
        MockProvider::text_response("b", 1, 1),
    ]))))
    .on_event(cb)
    .build()
    .expect("build ctx");

    // Build both calls UNDER phase "A"; call_c overrides to "C".
    let guard = event::phase(&ctx, "A");
    let call_a = agent(&ctx, "x");
    let call_c = agent(&ctx, "y").phase("C");
    drop(guard); // default phase restored to None — but the snapshots were taken.

    call_a.run().await.expect("run ok");
    call_c.run().await.expect("run ok");

    let evs = events.lock().expect("events lock");
    let phases: Vec<Option<String>> = evs
        .iter()
        .filter_map(|e| match e {
            WorkflowEvent::AgentStarted { phase, .. } => Some(phase.clone()),
            _ => None,
        })
        .collect();
    assert!(
        phases.contains(&Some("A".to_string())),
        "the construction-time snapshot ('A') must survive the guard drop: {phases:?}"
    );
    assert!(
        phases.contains(&Some("C".to_string())),
        "an explicit .phase('C') override must win: {phases:?}"
    );
}

/// Build a plain (no-journal) ctx over a `CountingProvider`.
fn plain_counting_ctx(calls: &Arc<AtomicUsize>) -> WorkflowCtx {
    WorkflowCtx::builder(Arc::new(BoxedProvider::new(CountingProvider {
        calls: Arc::clone(calls),
    })))
    .build()
    .expect("build ctx")
}

/// The `workflow()` nesting wrapper's happy path: it runs the body with a child
/// ctx that shares the parent's budget pool, and returns the body's value. No
/// test invoked the free fn before. Asserts the body ran (provider called) and
/// the child's spend is visible on the parent budget (shared pool).
#[tokio::test]
async fn workflow_runs_body_with_child_sharing_budget() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ctx = plain_counting_ctx(&calls);

    let result = workflow(&ctx, "research", |child| async move {
        // The child is a real, usable ctx: run an agent through it.
        agent(&child, "task").run().await
    })
    .await
    .expect("workflow ok");

    assert_eq!(result.as_deref(), Some("live-0"), "body value is returned");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "the body actually ran");
    // CountingProvider charges 10+5=15 weighted; the child's spend lands on the
    // shared parent budget pool.
    assert_eq!(
        ctx.budget().spent(),
        15,
        "the child's spend counts against the shared parent budget"
    );
}

/// `workflow()` allows exactly one level of nesting: calling it again from
/// inside a body must return `Error::Config` (not silently nest deeper).
#[tokio::test]
async fn workflow_nested_one_level_only_returns_config_error() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ctx = plain_counting_ctx(&calls);

    let result: Result<(), crate::error::Error> = workflow(&ctx, "outer", |child| async move {
        // A second level must be rejected.
        workflow(&child, "inner", |_grandchild| async move { Ok(()) }).await
    })
    .await;

    assert!(
        matches!(result, Err(crate::error::Error::Config(_))),
        "nesting a second workflow() level must return Error::Config, got {result:?}"
    );
}

/// A pipeline with NO stages returns one slot per item, each `Some(Value::Null)`
/// (the initial accumulator), in original order. Boundary case, previously
/// untested.
#[tokio::test]
async fn zero_stage_pipeline_returns_null_per_item() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ctx = plain_counting_ctx(&calls);
    let out = pipeline(&ctx, vec![10usize, 20usize, 30usize]).run().await;
    assert_eq!(
        out,
        vec![Some(Value::Null), Some(Value::Null), Some(Value::Null)],
        "a zero-stage pipeline yields the initial Null accumulator per item"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no stages -> no agent calls"
    );
}

/// Each stage receives the PREVIOUS stage's returned value as its `prev`
/// accumulator (the stage chain threads state). Previously, no test asserted the
/// `acc` hand-off between stages.
#[tokio::test]
async fn pipeline_threads_accumulator_between_stages() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ctx = plain_counting_ctx(&calls);
    let out = pipeline(&ctx, vec![7i64])
        // Stage 0 sees the initial Null accumulator and emits a number.
        .stage(|prev, _item, _idx| async move {
            assert_eq!(
                prev,
                Value::Null,
                "stage 0 starts from the Null accumulator"
            );
            Ok(Value::from(100i64))
        })
        // Stage 1 must receive stage 0's exact output.
        .stage(|prev, _item, _idx| async move {
            assert_eq!(
                prev,
                Value::from(100i64),
                "stage 1 must receive stage 0's returned value"
            );
            Ok(Value::from(prev.as_i64().unwrap() + 1))
        })
        .run()
        .await;
    assert_eq!(
        out,
        vec![Some(Value::from(101i64))],
        "the final accumulator is threaded through both stages"
    );
}
