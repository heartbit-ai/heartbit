# Heartbit Dynamic Workflows — Pure-Rust Async Combinator Core (Research + Design)

**Date:** 2026-05-30
*Produced by a 21-agent multi-agent research workflow (5 research facets + recommended-design judge + 8 adversarial verdicts + completeness critic, synthesized by a lead author).*

> **Verification note.** Every file:symbol below was checked against the live tree where the tool channel permitted. The tool channel degraded mid-session (a documented transport flakiness this run); items I could not personally re-confirm in this session are tagged **[verify-at-impl]** and must be re-greped before coding, per CLAUDE.md. The load-bearing dependency facts (git2/schemars/uuid-v5/jsonschema/blake3/sha2/tempfile, rustc 1.95.0, no `Agent` trait, `workspace.rs`/`sandbox.rs` at `crates/heartbit-core/src/`) were confirmed directly this session.

---

## 1. Executive Summary

**Goal.** Reproduce the *behaviour* of Claude Code "dynamic workflows" — pipeline / parallel / phase / shared-budget / per-stage structured-output / journal-resume / worktree-isolation — inside the Heartbit Rust harness, as an additive layer over the existing `AgentRunner`.

**Locked decision (non-negotiable).** Pure-Rust async **combinator core**. No JavaScript, no embedded scripting engine, no serializable workflow IR interpreter, no proc-macro DSL. A "workflow" is async Rust a developer writes against a combinator API:

```rust
let results = pipeline(&ctx, items)
    .stage(|x| agent(&ctx, summarize(x)).schema::<Summary>().run())
    .run()
    .await;
```

Runtime dynamism (loop-until-dry, loop-until-budget, dynamic fan-out over discovered work-lists) is plain Rust control flow over the combinators. Where a *model* must decide topology, we defer to Heartbit's existing LLM orchestrator (`DelegateTaskTool` / `FormSquadTool` / `SpawnAgentTool`). We match Claude Code **semantics**, not its LLM-authored-script authoring surface.

**Recommendation (5 bullets).**

- **Add one module tree `crates/heartbit-core/src/agent/flow/`** exposing free functions `agent / parallel / pipeline / phase / log / workflow` taking `&WorkflowCtx` first, plus a fluent `AgentCall<'_, T>` builder for `.schema::<T>()`. **Purely additive** — there is *no* `Agent` trait to refactor (the atomic unit is the concrete generic `AgentRunner<P>::execute(&self, &str) -> Result<AgentOutput, Error>`); `SequentialAgent` / `ParallelAgent` / `LoopAgent` / `Orchestrator` compile unchanged.
- **The new `parallel()` is fail-SOFT** (a throwing thunk → `None`, the call never rejects, submission order preserved). It is the deliberate *dual* of the existing fail-FAST `ParallelAgent`; they coexist. **`pipeline()` is NO-BARRIER** — one future per item folding the whole stage chain sequentially inside itself, all item-futures driven concurrently, so wall-clock = slowest single chain (not sum-of-slowest-per-stage). This is the single most important and most easily-mis-implemented semantic.
- **Shared hard-ceiling `Budget`** is lock-free (`Arc<AtomicU64>` + `Option<NonZeroU64>`), record-only (check-at-admission / record-on-completion), reusing `Error::BudgetExceeded { used, limit }` verbatim and accumulating in `u64` widened from the `u32` `TokenUsage` fields. A single global `Arc<tokio::sync::Semaphore>` (`min(16, cores-2)`) caps in-flight agents, acquired **at the agent() leaf only**; a separate `Arc<AtomicU64>` is the 1000-agent runaway backstop. **Charge by cost-weighted tokens, not a raw sum** (cache-read ≈ 0.1×, cache-write ≈ 1.25×).
- **Per-call typed structured output is a thin wrapper** over machinery that already exists: `AgentRunnerBuilder::structured_schema(Value)` injects `__respond__`, validates with the full `jsonschema` crate, and auto-retries on mismatch; the typed terminal just sets the schema (via `schemars`) and `serde_json::from_value::<T>(out.structured)`. **MVP needs zero runner change.**
- **Ship iteratively, TDD-first, each phase independently shippable.** Defer the two hard, dependency-adding phases (RunJournal-resume; git-worktree isolation) to last so their correctness burden cannot destabilize the spine. V1 is **standalone-path only** (same boundary as guardrails); the Restate durable path is a named non-goal for V1.

**Headline phased roadmap (one line each).**

- **P1** — Combinator core: `WorkflowCtx` + `parallel()` + `pipeline()` + `phase()`/`log()` + concurrency cap + backstop (zero new deps).
- **P2** — Shared `Budget` hard-ceiling pool (cost-weighted), threaded through the ctx and nested workflows.
- **P3** — Per-stage typed structured output `agent().schema::<T>()` (adds optional `schemars` feature).
- **P4** — `RunJournal` deterministic longest-unchanged-prefix resume (adds `uuid` v5; reuses `sha2`/adds `blake3`).
- **P5** — `isolation: Worktree` git-worktree-per-mutator (adds `git2`).
- **P6** — Observability `WorkflowEvent`/`/workflows` analog + one-level `workflow()` nesting + cancellation/abort controls.
- **P7** — Bridge to LLM orchestrator (model-decided dynamism) + Restate durable-path parity (later, out of V1 scope).

---

## 2. What We Are Matching

### 2.1 Claude Code dynamic-workflow semantics (the behavioural target)

- **`agent(prompt, opts) -> result`** — the atomic unit: spawns one subagent. Without a schema returns final text; with a schema the agent is *forced* to emit a validated object. `opts`: `label`, `phase`, `schema`, `model`, `isolation:"worktree"`, `agentType`. Returns `null` if skipped mid-run.
- **`parallel(thunks[]) -> results[]`** — **BARRIER**. Awaits all thunks. A thunk that throws resolves to `null` in the array (the call never rejects). Filter nulls before use.
- **`pipeline(items, ...stages) -> results[]`** — **NO BARRIER**. Each item flows through *all* stages independently; item A can be in stage 3 while item B is still in stage 1. Wall-clock = slowest single-item chain, **not** sum-of-slowest-per-stage. A stage that throws drops that item to `null` and skips its remaining stages. The default for multi-stage work. Each stage gets `(prevResult, originalItem, index)`.
- **`phase(title)`** — groups subsequent agents under a progress group. **`log(msg)`** — emit a progress line.
- **`workflow(nameOrRef, args) -> result`** — run another workflow inline; **one level of nesting only**. Child shares the run concurrency cap, agent counter, abort signal, and token budget.
- **`budget: { total, spent(), remaining() }`** — a **SHARED** token pool across the main loop and all workflows (not per-workflow). Hard ceiling: once `spent` reaches `total`, further `agent()` calls **throw**. `total=null ⇒ remaining()=Infinity`. Drives dynamic loops (`while budget.remaining() > 50000`) and static scaling (`FLEET = total ? floor(total/100000) : 5`).
- **Concurrency**: `min(16, cores-2)` concurrent agents; excess queues. 1000 agents total per run (runaway backstop).
- **`isolation:"worktree"`** — agent runs in a fresh git worktree (for parallel file mutators that would otherwise conflict); auto-removed if unchanged; expensive (~200-500 ms + disk). Parallel mutators only.
- **Resume** — a run has a `runId`; relaunch with `resumeFromRunId`. The longest *unchanged prefix* of `agent()` calls returns cached results instantly; the first edited/new call onward runs live. Same script + same args = 100% cache hit. Wall-clock and RNG are **forbidden** in scripts (would break deterministic replay).
- **Structured output** — schema validation at the tool-call layer; the model retries on mismatch.
- **Quality patterns** — adversarial verify (N skeptics, refute-by-default), perspective-diverse verify, judge panel (N attempts, score, synthesize), loop-until-dry, multi-modal sweep, completeness critic, loop-until-budget.
- **Execution** — runs in background; observed via a `/workflows` progress view (phases, agent counts, token totals, elapsed time; pause/resume/stop/restart-agent). Subagents run in `acceptEdits` and inherit the tool allowlist.

### 2.2 What we are explicitly NOT copying — and why

We do **not** reproduce Claude Code's *authoring model*: an LLM writes a JS script, a runtime sandbox interprets it, and the sandbox forbids `Date.now()`/`Math.random()` to keep replay deterministic. In a pure-Rust core that authoring surface collapses into **developer-written async Rust**:

- The "script" is a normal `async fn` against the combinator API. There is no IR, no interpreter, no second language.
- Runtime dynamism — variable fan-out, loop-until-X — is **plain Rust `while`/`for`/`match`** reading `ctx.budget().remaining()` and prior results. No new primitive is needed for any of Claude Code's quality patterns (§5.10 shows each in ~5 lines).
- **Model-decided topology** (the LLM picks *how many* / *which* sub-agents at runtime) is the one thing combinators cannot express, and it is *already solved* by Heartbit's `DelegateTaskTool` (`DispatchMode::Parallel|Sequential`), `FormSquadTool`, and `SpawnAgentTool`. The boundary is sharp: **code-decided fan-out → combinators; model-decided fan-out → orchestrator.**
- Determinism cannot be *enforced* by the language the way a JS sandbox enforces it (Rust cannot compile-forbid `Instant::now()`/`rand`). We make it a **documented discipline** plus an optional debug-mode divergence check, and we keep the journal honest about where replay is call-for-call vs coarse (§5.7, §7).

---

## 3. Heartbit Harness Today

All references verified against the live tree this session except where tagged **[verify-at-impl]**.

### 3.1 What already maps

| Capability | File : symbol | Status |
|---|---|---|
| Compile-time pipeline (BARRIER) | `crates/heartbit-core/src/agent/workflow.rs` : `SequentialAgent<P>` (`agents: Vec<AgentRunner<P>>`), pipes `prev.result → next input` | exists, wrong wall-clock for no-barrier |
| Compile-time fan-out (FAIL-FAST) | `workflow.rs` : `ParallelAgent<P>` (`Vec<Arc<AgentRunner<P>>>`, `tokio::JoinSet`, name-sorted merge, first-`Err` `?`) | exists, opposite contract to target |
| Loop with stop-closure | `workflow.rs` : `LoopAgent<P>` (`should_stop` + `max_iterations`) | exists |
| Other deterministic shapes | `workflow.rs` / `dag.rs` / `debate.rs` / `voting.rs` / `mixture.rs` + `WorkflowRouter` | exist |
| Atomic ReAct unit | `runner.rs` : `AgentRunner<P>::execute(&self, task: &str) -> Result<AgentOutput, Error>` | exists — **the** unit (no `Agent` trait) |
| Parallel **tool** execution | `runner.rs` : `execute_tools_parallel` — `JoinSet`, collect `(idx, output, …)`, re-order by index | exists — literal template for `parallel<T>` |
| Structured output | `runner.rs` : `structured_schema: Option<serde_json::Value>` + injected `__respond__` + retry-on-mismatch + `AgentOutput.structured` | exists — typed wrapper needs **zero** runner change |
| Schema validation | `crates/heartbit-core/src/tool/mod.rs` : `validate_tool_input` → `jsonschema::validator_for` + `iter_errors` (skips an *uncompilable* schema) | exists (full jsonschema), `jsonschema = "0.28"` |
| Per-runner hard token cap | `runner.rs` : `max_total_tokens: Option<u64>`, post-turn `if used > max → Error::BudgetExceeded.with_partial_usage(..)` | exists |
| Concurrency cap precedent | `crates/heartbit-core/src/agent/batch.rs` : `Arc<Semaphore::new(max_concurrency)>`, `std::thread::available_parallelism()`, `acquire()`, peak-gauge test | exists — template for global cap |
| In-memory response cache | `crates/heartbit-core/src/agent/cache.rs` : `ResponseCache` LRU `Mutex<Vec<(u64, CompletionResponse)>>`, FNV-1a key | exists — **not** a resumable journal |
| Durable journal (analog) | `crates/heartbit/src/workflow/agent_workflow.rs` (event-journal replay, skip completed activities) + `budget.rs` : `TokenBudgetObject` | exists — Restate-only durable analog |
| Filesystem isolation | `crates/heartbit-core/src/sandbox.rs` : `SandboxPolicy::workspace_only(..)`; `workspace.rs` : `normalize_path`/`resolve_within`; `bash.rs` : `BashTool::with_sandbox` + per-spawn `pre_exec(landlock_pre_exec)` | exists — Landlock, **no** git worktree |
| Per-agent workspace seam | `orchestrator.rs` : `SubAgentConfig.workspace: Option<PathBuf>` + `.sandbox_policy: Option<SandboxPolicy>`, consumed at dispatch | exists — the worktree injection point |
| Cancellation primitive | `tokio_util::sync::CancellationToken` (used in `signal.rs`, daemon `core.rs`) | exists — `tokio-util` already a dep |

### 3.2 What does not map

- **No no-barrier pipeline.** `SequentialAgent` advances *all* work stage-by-stage (barrier). Wall-clock = sum-of-slowest-per-stage, not slowest-single-chain.
- **No fail-soft parallel.** `ParallelAgent` is fail-fast (`?` on first error) and merges by name. Claude Code `parallel` is fail-soft, submission-ordered, never-rejects.
- **No shared budget pool.** `max_total_tokens` is a *per-runner* cap; there is no run-wide shared ceiling threaded through fan-out and nested workflows.
- **No run-wide concurrency cap on the fan-out paths.** `ParallelAgent`/`dag`/`voting`/`mixture`/orchestrator spawn into `JoinSet` *unbounded*; only `batch.rs` has a `Semaphore`.
- **No per-stage schema.** Structured output is configured per *runner*, not threaded as a combinator stage option (though the underlying machinery is reusable as-is).
- **No resumable journal.** `cache.rs` is in-memory, unordered, FNV-keyed — it is a within-run approximation, not an ordered content-addressed prefix log.
- **No git-worktree-per-agent.** Only Landlock + workspace-path jailing.
- **No `WorkflowEvent`/`/workflows` view.** `AgentEvent`/`OnEvent` exist for per-agent observability but are **not `#[non_exhaustive]`** **[verify-at-impl]**, so workflow events must live on a *separate* enum.
- **No `Error::Cancelled`, no `Error::AgentBudgetExceeded`.** `error.rs` has `RunTimeout` and the token `BudgetExceeded { used, limit }`; both new variants must be added (thiserror). **[verify-at-impl: confirm `Cancelled` absent]**

---

## 4. Gap Analysis (per target primitive)

| Target primitive | Gap to close |
|---|---|
| `agent(prompt, opts)` | Build/drive a per-call `AgentRunner<P>` behind a fluent `AgentCall`; thread permit + backstop + budget + journal + phase; return `Ok(None)` for skipped/cancelled (not `Err`). |
| `parallel(thunks)` | New fail-SOFT combinator: `JoinSet`, map `Err`/`JoinError`(panic/abort) → `None`, submission order, never reject. Distinct from `ParallelAgent`. |
| `pipeline(items, ...stages)` | New NO-BARRIER combinator: one future per item folding stages internally, all driven concurrently; stage `Err` → item `None` + skip remaining stages; flow type fixed to `serde_json::Value`. |
| `phase` / `log` | New `WorkflowEvent` plane + a per-call phase snapshot (not a single shared slot — see §7 risk). |
| `budget` (shared hard ceiling) | New lock-free `Budget(Arc<AtomicU64> + Option<NonZeroU64>)`; cost-weighted charge; reuse `Error::BudgetExceeded`; thread one `Arc` run-wide + into nested workflows. |
| concurrency cap + 1000 backstop | New `Arc<Semaphore>` (`min(16,cores-2)`) acquired at leaf + `Arc<AtomicU64>` backstop → new `Error::AgentBudgetExceeded`. |
| `schema::<T>()` | Thin typed wrapper: `schemars::schema_for!(T)` → `structured_schema(Value)` (existing) → `from_value::<T>(out.structured)`. Add optional `schemars` dep + feature. |
| resume / `runId` | New `RunJournal`: ordered JSONL keyed by `(call_index, content_hash)`, longest-unchanged-prefix replay; deterministic `runId` via `uuid::new_v5`. Add `uuid` v5; reuse `sha2` or add `blake3`. |
| `isolation:"worktree"` | New `WorktreeGuard` (git2 in `spawn_blocking`, dual cleanup) → feed `SubAgentConfig.workspace` + `SandboxPolicy::workspace_only`. Add `git2`. |
| `/workflows` view + controls | New `WorkflowEvent`/`OnWorkflowEvent`; `CancellationToken` (pause/stop) + per-task `AbortHandle` (restart-one). |
| `workflow()` nesting (1 level) | New free fn; `ctx.nested()` clones the shared handles, `depth` guard rejects a second level (`Error::Config`). |
| determinism (`no clock/RNG`) | Documented discipline + optional journaled `ctx.now()`/`ctx.rand_uuid()` + debug divergence check. |

---

## 5. Recommended Architecture — the pure-Rust async combinator core

**Placement.** New module tree `crates/heartbit-core/src/agent/flow/` — `mod.rs`, `ctx.rs`, `budget.rs`, `agent.rs`, `parallel.rs`, `pipeline.rs`, `event.rs`, `journal.rs` *(P4)*, `worktree.rs` *(P5)*. Declared `pub mod flow;` in `agent/mod.rs`, re-exported from `lib.rs` next to `pub use agent::workflow::{SequentialAgent, ParallelAgent, LoopAgent}`. The whole core (incl. `RunJournal` — pure std/tokio I/O) lives in **heartbit-core**; the Restate durable bridge lives later in the **heartbit** umbrella crate.

**API shape.** Free functions taking `&WorkflowCtx` first, with thin `impl WorkflowCtx` delegators for discoverability. Rationale: there is no trait to hang a method API on, and free fns compose cleanly with closures that both capture `ctx.clone()` and are passed into a combinator (a method-primary API hits borrow-checker friction there).

### 5.1 `WorkflowCtx` — threads budget + journal + concurrency + phase

```rust
// flow/ctx.rs
#[derive(Clone)]
pub struct WorkflowCtx {
    inner: std::sync::Arc<CtxInner>,
}

struct CtxInner {
    provider: std::sync::Arc<BoxedProvider>,          // existing object-safe provider keeps ctx CONCRETE despite AgentRunner<P>
    base_tools: std::sync::Arc<Vec<std::sync::Arc<dyn Tool>>>,
    budget: Budget,
    sem: std::sync::Arc<tokio::sync::Semaphore>,       // global min(16,cores-2) cap; shared run-wide + into nested workflows
    spawned: std::sync::atomic::AtomicU64,             // 1000-agent runaway backstop
    max_agents: u64,
    agent_seq: std::sync::atomic::AtomicU64,           // deterministic issue-order call_index (combinator INTRODUCES this)
    events: Option<std::sync::Arc<OnWorkflowEvent>>,
    cancel: tokio_util::sync::CancellationToken,       // tokio-util already a dep (signal.rs precedent)
    journal: Option<std::sync::Arc<RunJournal>>,       // P4
    run_id: String,
    depth: u8,                                          // one-level nesting guard
}

impl WorkflowCtx {
    pub fn builder(provider: std::sync::Arc<BoxedProvider>) -> WorkflowCtxBuilder { /* … */ }
    pub fn budget(&self) -> &Budget { &self.inner.budget }
    pub fn remaining(&self) -> u64 { self.inner.budget.remaining() }
    pub fn run_id(&self) -> &str { &self.inner.run_id }
    pub fn is_cancelled(&self) -> bool { self.inner.cancel.is_cancelled() }
    pub fn cancellation_token(&self) -> tokio_util::sync::CancellationToken { self.inner.cancel.clone() }

    pub(crate) fn next_agent_index(&self) -> u64 {
        self.inner.agent_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
    pub(crate) fn nested(&self) -> Result<WorkflowCtx, Error> {
        if self.inner.depth > 0 {
            return Err(Error::Config("workflow() may nest only one level".into()));
        }
        // clone CtxInner with depth+1, sharing budget/sem/spawned/cancel/journal by Arc
        // (the counters are AtomicU64 — share the SAME Arc, not a fresh atomic)
        // … construct nested ctx …
        # unimplemented!()
    }
    pub(crate) fn child_runner_builder(&self, opts: &AgentOpts) -> AgentRunnerBuilder<BoxedProvider> {
        // base_tools or opts.tools; provider or per-opts model override; max_total_tokens local cap; on_event forwarder
        # unimplemented!()
    }
}

pub struct WorkflowCtxBuilder { /* provider, total_budget, max_concurrency, max_agents=1000, events, base_tools, journal, cancel */ }

impl WorkflowCtxBuilder {
    pub fn budget(self, total: Option<u64>) -> Self { /* … */ self }
    pub fn max_concurrency(self, n: usize) -> Self { /* … */ self }
    pub fn max_agents(self, n: u64) -> Self { /* … */ self }
    pub fn tools(self, tools: Vec<std::sync::Arc<dyn Tool>>) -> Self { /* … */ self }
    pub fn on_event(self, cb: impl Fn(WorkflowEvent) + Send + Sync + 'static) -> Self { /* … */ self }
    pub fn resume(self, base_dir: std::path::PathBuf, run_id: impl Into<String>, mode: ResumeMode) -> Self { /* … */ self } // P4
    pub fn cancellation_token(self, t: tokio_util::sync::CancellationToken) -> Self { /* … */ self }
    pub fn build(self) -> Result<WorkflowCtx, Error> { /* reject max_agents==0 / max_concurrency==0 via Error::Config */ # unimplemented!() }
}

// default cap — copies the batch.rs idiom exactly:
fn default_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .saturating_sub(2)
        .clamp(1, 16)
}
```

`phase` deliberately is **not** a single shared mutable slot (see §7 — concurrent agents would stomp it). It is snapshotted per `AgentCall` at construction and carried into the nested ctx for `workflow()`.

### 5.2 `agent()` — the atomic unit (the fluent builder)

```rust
// flow/agent.rs
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct AgentOpts {
    pub label: Option<String>,
    pub phase: Option<String>,
    pub model: Option<String>,
    pub agent_type: Option<String>,
    pub isolation: Isolation,
    pub schema: Option<serde_json::Value>,
    pub tools: Option<Vec<std::sync::Arc<dyn Tool>>>,
    pub system_prompt: Option<String>,
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Isolation { #[default] None, Worktree }

pub struct NoSchema;
pub struct RawJson;

pub struct AgentCall<'c, T = NoSchema> {
    ctx: &'c WorkflowCtx,
    prompt: String,
    opts: AgentOpts,
    phase_snapshot: Option<std::sync::Arc<str>>,   // captured at construction — NOT read from a shared slot at exec time
    _t: std::marker::PhantomData<fn() -> T>,        // T lives ONLY here → object-safety preserved (runner stays Value-only)
}

impl<'c> AgentCall<'c, NoSchema> {
    pub fn label(mut self, s: impl Into<String>) -> Self { self.opts.label = Some(s.into()); self }
    pub fn phase(mut self, s: impl Into<String>) -> Self { self.opts.phase = Some(s.into()); self }
    pub fn model(mut self, s: impl Into<String>) -> Self { self.opts.model = Some(s.into()); self }
    pub fn isolation(mut self, i: Isolation) -> Self { self.opts.isolation = i; self }

    #[cfg(feature = "derive-schema")]
    pub fn schema<T>(self) -> AgentCall<'c, T>
    where T: schemars::JsonSchema + serde::de::DeserializeOwned + Send {
        let schema = serde_json::to_value(schemars::schema_for!(T))
            .expect("RootSchema is infallibly serializable");  // CLAUDE.md-sanctioned infallible expect
        AgentCall { ctx: self.ctx, prompt: self.prompt,
            opts: AgentOpts { schema: Some(schema), ..self.opts },
            phase_snapshot: self.phase_snapshot, _t: std::marker::PhantomData }
    }

    pub fn schema_value(self, schema: serde_json::Value) -> AgentCall<'c, RawJson> {
        AgentCall { ctx: self.ctx, prompt: self.prompt,
            opts: AgentOpts { schema: Some(schema), ..self.opts },
            phase_snapshot: self.phase_snapshot, _t: std::marker::PhantomData }
    }

    /// No-schema text terminal. Ok(None) == Claude Code null (skipped/cancelled mid-run), NOT Err.
    pub async fn run(self) -> Result<Option<String>, Error> { /* leaf sequence §5.2.1; map out.result */ # unimplemented!() }
}

impl<'c, T: serde::de::DeserializeOwned + Send> AgentCall<'c, T> {
    /// Typed terminal: forces a validated object via the existing __respond__ path, then from_value::<T>.
    pub async fn run(self) -> Result<Option<T>, Error> { /* leaf sequence; serde_json::from_value::<T>(out.structured?) */ # unimplemented!() }
}

impl<'c> AgentCall<'c, RawJson> {
    pub async fn run(self) -> Result<Option<serde_json::Value>, Error> { /* leaf sequence; out.structured */ # unimplemented!() }
}

pub fn agent(ctx: &WorkflowCtx, prompt: impl Into<String>) -> AgentCall<'_, NoSchema> { /* snapshot current phase */ # unimplemented!() }

impl WorkflowCtx { pub fn agent(&self, prompt: impl Into<String>) -> AgentCall<'_, NoSchema> { agent(self, prompt) } }
```

#### 5.2.1 The leaf sequence (load-bearing order)

Inside `run()`, in **exactly** this order (corrected per the completeness critic: journal HIT must precede backstop + budget so a 100%-cache-hit resume does no live work and is exempt from both ceilings):

```text
1. JOURNAL HIT CHECK (if ctx.journal is Some):
     let idx = ctx.next_agent_index();
     let key = (idx, content_hash(inputs));
     if let Some(cached) = journal.lookup(&key) { return Ok(Some(map(cached))); }   // exempt from steps 2/3, 0 tokens
2. PERMIT (block/queue): let _permit = ctx.inner.sem.clone().acquire_owned().await.map_err(|_| Error::Cancelled)?;
3. BACKSTOP: let n = ctx.inner.spawned.fetch_add(1, Relaxed); if n >= ctx.inner.max_agents { return Err(Error::AgentBudgetExceeded { limit: ctx.inner.max_agents }); }
4. BUDGET (fail-fast HARD): ctx.budget().check_admit()?;   // rejects once spent >= total (boundary matches runner.rs/Restate)
5. (P5) if opts.isolation == Worktree: allocate guard, set workspace + sandbox_policy
6. BUILD + RUN under cancel race:
     let out = tokio::select! {
         _ = ctx.inner.cancel.cancelled() => return Ok(None),        // OR Err(Cancelled); see §7 A8
         r = run_one(ctx, &prompt, &opts) => r?,
     };
7. RECORD (live MISS only): ctx.budget().record_weighted(out.model_name.as_deref(), &out.tokens_used);
8. JOURNAL APPEND (if Some): journal.append(key, &out)?;
9. return Ok(Some(map(out)));
```

`run_one` builds the per-call `AgentRunner` via `child_runner_builder`, setting `.structured_schema(opts.schema)`, `.on_event(forwarding_cb)` (folds `AgentEvent::LlmResponse.usage` into the workflow tally), `.tools(opts.tools or base_tools)`, `.max_total_tokens(per-call local cap)`, then calls `AgentRunner::execute(&prompt).await`.

> **Invariant (module doc, load-bearing):** the `Err → None` collapse lives **only** in `parallel()`/`pipeline()` at the join/stage boundary, **never** inside `agent()`. If `agent()` swallowed `Err`, `BudgetExceeded` and the backstop would stop propagating and both ceilings would silently break. Moreover, `BudgetExceeded` / `AgentBudgetExceeded` / `Cancelled` are **STICKY** — they propagate *past* the join to a run-level `Err`; only agent-domain errors collapse to `None` (see §7).

### 5.3 `parallel()` — BARRIER, fail-SOFT, submission-order, never rejects

```rust
// flow/parallel.rs
pub async fn parallel<R, F, Fut>(_ctx: &WorkflowCtx, thunks: Vec<F>) -> Vec<Option<R>>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<R, Error>> + Send + 'static,
    R: Send + 'static,
{
    let n = thunks.len();
    let mut set = tokio::task::JoinSet::new();
    for (i, th) in thunks.into_iter().enumerate() {
        set.spawn(async move { (i, th().await) });
    }
    let mut out: Vec<Option<R>> = (0..n).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((i, Ok(v)))  => out[i] = Some(v),
            Ok((i, Err(_e))) => { /* emit WorkflowEvent::AgentFailed; leave None */ }
            Err(_join)      => { /* panic/abort: slot stays None */ }
        }
    }
    out
}
```

This is the verified `runner.rs` `execute_tools_parallel` submission-order drain **minus** fail-fast and name-sort. It is the deliberate dual of `ParallelAgent` (same-input-to-all, first-`Err` `?`, name-sorted merge). `JoinSet` gives panic isolation and per-task `AbortHandle` (restart-one/stop-one).

> **Sticky-error handling.** When `Err(e)` is a *control* error (`BudgetExceeded`/`AgentBudgetExceeded`/`Cancelled`), record it into a shared `OnceCell<Error>` on the ctx and fire the run-wide `CancellationToken`; after the drain, if a sticky error was captured, the *caller's* higher-level combinator surfaces it as `Err`. A practical encoding is a `parallel_strict` variant returning `Result<Vec<Option<R>>, Error>` that returns the captured control error; the plain `parallel` keeps the never-rejects contract for the common case but still trips cancellation so the run halts.

### 5.4 `pipeline()` — NO BARRIER per-item streaming

```rust
// flow/pipeline.rs
pub type Stage<I> = std::sync::Arc<
    dyn Fn(serde_json::Value, std::sync::Arc<I>, usize)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value, Error>> + Send>>
        + Send + Sync,
>;

pub async fn pipeline<I: Send + Sync + 'static>(
    _ctx: &WorkflowCtx,
    items: Vec<I>,
    stages: Vec<Stage<I>>,
) -> Vec<Option<serde_json::Value>> {
    let n = items.len();
    let stages = std::sync::Arc::new(stages);
    let mut set = tokio::task::JoinSet::new();
    for (idx, item) in items.into_iter().enumerate() {
        let stages = stages.clone();
        let item = std::sync::Arc::new(item);
        set.spawn(async move {
            let mut acc = serde_json::Value::Null;
            for st in stages.iter() {
                match st(acc, item.clone(), idx).await {
                    Ok(v) => acc = v,
                    Err(_e) => { /* emit AgentFailed; drop this item, skip remaining stages */ return (idx, None); }
                }
            }
            (idx, Some(acc))
        });
    }
    let mut out: Vec<Option<serde_json::Value>> = (0..n).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        if let Ok((idx, res)) = joined { out[idx] = res; }   // panic → slot stays None
    }
    out
}
```

**Exact semantics (the headline):** the inner `for st in stages` loop is sequential **per item**; the outer `JoinSet` drives all items' chains **concurrently**. Therefore wall-clock = `max_i( Σ_k cost(i, k) )` = slowest single chain — structurally distinct from a fold-of-per-stage-`parallel()` (which is a barrier yielding `Σ_k max_i cost(i, k)`). A stage `Err` returns `(idx, None)` early, skipping the item's remaining stages = "drop to null, skip remaining stages". Each stage receives `(prevResult = acc, originalItem = item, index = idx)`.

> **Honest cap caveat (rustdoc):** the literal equality holds only when the global semaphore is not the binding constraint. Under saturation (`#items × simultaneously-active-stages > permits`) chains queue on permit acquisition and wall-clock degrades toward `~total-work / cap` — still strictly better than sum-of-slowest-per-stage, but no longer the exact slowest-single-chain. State the invariant as: *"no inter-item barrier; wall-clock = max(slowest single chain, ~total-work / concurrency-cap)."*

**Grafted variants** (opt-in, see §7):

- A typestate `PipelineBuilder<I, A>::stage::<B>(…) -> PipelineBuilder<I, B>` for compile-time-typed *static* 2–3-stage pipelines, advancing the flowing type `A → B` per stage. The fluent `pipeline(&ctx, items).stage(..).run()` surface is this builder.
- A `pipeline_streamed(ctx, items, stages) -> impl Stream<Item = (usize, Option<Value>)>` built on `stream::iter(items).map(per_item_chain).buffer_unordered(cap)` for very large / dynamically-discovered work-lists: memory `O(cap)` not `O(items)`. **Tradeoff (documented):** `buffer_unordered` is cooperative single-task polling — a panicking or CPU-bound stage panics/starves the whole driver. Keep the **`JoinSet` pipeline as the default** for panic isolation.

### 5.5 `phase()` / `log()` + the `WorkflowEvent` plane

```rust
// flow/event.rs
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkflowEvent {
    WorkflowStarted { name: String, total_budget: Option<u64> },
    PhaseStarted   { title: String },
    AgentStarted   { label: String, phase: Option<String>, agent_index: u64 },
    AgentFinished  { label: String, usage: TokenUsage, replayed: bool },
    AgentSkipped   { label: String },
    AgentFailed    { label: String, error: String },
    LogLine        { msg: String },
    WorkflowFinished { spent: u64, agents: u64 },
}

pub type OnWorkflowEvent = dyn Fn(WorkflowEvent) + Send + Sync;

/// RAII: sets the ctx's "default phase for subsequently-issued calls" + emits PhaseStarted;
/// restores the prior default on Drop. Per-AgentCall phase is SNAPSHOTTED at call construction,
/// so concurrent agents never stomp each other's labels.
pub fn phase(ctx: &WorkflowCtx, title: impl Into<String>) -> PhaseGuard { /* … */ # unimplemented!() }
pub fn log(ctx: &WorkflowCtx, msg: impl Into<String>) { /* emit LogLine */ }

pub struct PhaseGuard { /* prior: Option<Arc<str>>, ctx handle */ }
impl Drop for PhaseGuard { fn drop(&mut self) { /* restore prior default phase */ } }
```

`WorkflowEvent` is a **separate** `#[non_exhaustive]` enum because `AgentEvent` is *not* `#[non_exhaustive]` **[verify-at-impl]** — adding a variant to it is a breaking change (exhaustive `type_name()` match + external matchers). The ctx folds per-agent `AgentEvent` (read-only) into phase/agent-count/token/elapsed totals to power the `/workflows` view.

### 5.6 `Budget` — shared hard-ceiling pool, lock-free, cost-weighted

```rust
// flow/budget.rs
#[derive(Clone)]
pub struct Budget(std::sync::Arc<BudgetInner>);

struct BudgetInner {
    total: Option<std::num::NonZeroU64>,          // None ⇒ Infinity; 0 unrepresentable (matches Heartbit zero-rejection)
    spent: std::sync::atomic::AtomicU64,
}

impl Budget {
    pub fn unbounded() -> Self { Self(std::sync::Arc::new(BudgetInner { total: None, spent: 0.into() })) }
    pub fn with_total(total: u64) -> Self {
        Self(std::sync::Arc::new(BudgetInner { total: std::num::NonZeroU64::new(total), spent: 0.into() }))
    }
    pub fn total(&self) -> Option<u64> { self.0.total.map(std::num::NonZeroU64::get) }
    pub fn spent(&self) -> u64 { self.0.spent.load(std::sync::atomic::Ordering::Relaxed) }
    pub fn remaining(&self) -> u64 {
        match self.0.total { None => u64::MAX, Some(t) => t.get().saturating_sub(self.spent()) }
    }
    /// Hard admission. Boundary matches runner.rs (`used > max`) and Restate budget.rs (`new_total > limit`):
    /// allow spending exactly to the limit, reject once spent has REACHED/passed it.
    pub(crate) fn check_admit(&self) -> Result<(), Error> {
        if let Some(t) = self.0.total {
            let s = self.spent();
            if s >= t.get() { return Err(Error::BudgetExceeded { used: s, limit: t.get() }); }
        }
        Ok(())
    }
    /// Record AFTER completion. Widen u32 TokenUsage fields to u64; charge COST-WEIGHTED, not raw sum.
    pub(crate) fn record_weighted(&self, model: Option<&str>, usage: &TokenUsage) {
        let cost = weighted_cost(model, usage);
        self.0.spent.fetch_add(cost, std::sync::atomic::Ordering::AcqRel);
    }
}

/// COST-WEIGHTED so FLEET=floor(total/100000) and remaining()>50000 mean SPEND, not raw throughput.
/// Reuses the runner's own per-turn cost path (estimate_cost, re-exported from lib.rs).
fn weighted_cost(model: Option<&str>, usage: &TokenUsage) -> u64 {
    match model.and_then(|m| estimate_cost(m, usage)) {
        Some(usd) => (usd * 1_000_000.0).ceil() as u64,    // micro-USD
        None => {
            // unknown model: conservative fallback — widen each field to u64 (NEVER call a u32 total() first),
            // weight cache_read ~0.1x, cache_write ~1.25x, reasoning at output rate, OR use a top-tier price.
            (usage.input_tokens as u64)
                + (usage.output_tokens as u64)
                + (usage.cache_creation_input_tokens as u64) * 5 / 4
                + (usage.cache_read_input_tokens as u64) / 10
        }
    }
}
```

**Why record-only (not reserve-reconcile) by default.** Matches the Claude Code target ("spent reaches total → `agent()` throws") and maps **1:1 onto the existing Restate `TokenBudgetObject`** with *zero* new handlers. Its only weakness — over-admission bounded by `concurrency × cost` (because the permit caps in-flight agents at `C` and `check_admit` runs only while holding a permit) — is shared by the target itself. Offer reserve-then-reconcile (`try_reserve` CAS + `Reservation` `Drop`-refund + a *new* Restate `try_reserve` handler) as opt-in Phase-N hardening behind a flag; same `Budget` type.

**Determinism contract (rustdoc).** The budget reads no clock and no RNG (replay-safe by construction). The discipline falls on *loop authors*: `loop-until-budget` / fleet-sizing must derive iteration count and topology only from `remaining()` and prior results, never `Instant::now()`/`rand`. `let fleet = ctx.budget().total().map_or(5, |t| (t / 100_000) as usize).max(1);` is deterministic — keep it so.

### 5.7 Per-call typed structured output (`schema::<T>()`)

The MVP needs **zero runner change**. The runner already: (a) injects `__respond__` when `structured_schema` is set and *forces* it (errors if the model finishes without calling it), (b) validates each `__respond__` payload with the full `jsonschema` crate, (c) on mismatch pushes an `is_error` tool result "Please fix … call `__respond__` again" and *retries*, (d) returns `AgentOutput.structured: Some(Value)` on success.

So the typed terminal is:

```rust
// inside AgentCall<'c, T>::run() — after the leaf sequence yields `out: AgentOutput`
let v = out.structured.ok_or_else(|| Error::Agent("agent finished without structured output".into()))?;
serde_json::from_value::<T>(v).map_err(Error::Json)   // Error::Json is #[from] serde_json::Error
```

`serde` enforces what JSON Schema cannot express (unknown enum variants, exact Rust integer width, `#[serde(deny_unknown_fields)]`, custom `Deserialize`) **and** covers the runner's *uncompilable-schema-skip* gap. **Object-safety** is preserved because `T` lives only on `AgentCall<'_, T>` via `PhantomData<fn() -> T>`; the runner stays `Value`-only and `Tool` stays `Arc<dyn Tool>`.

> **Optional P3 hardening (additive runner change, distinct phase).** If you want a `from_value::<T>` failure to *also* become a model retry (not just a hard `Err` after `execute` returns), thread a type-erased `Box<dyn Fn(&serde_json::Value) -> Result<(), String> + Send + Sync>` validator into the `__respond__` `Ok`-branch (run after `validate_tool_input`, reuse the same `is_error` retry feedback). Object-safe, backward-compatible (`Option`, default `None`), bounded by `max_turns`/`max_total_tokens`. **[verify-at-impl: confirm whether a `structured_validator` builder setter already exists; the corpus could not confirm the symbol name — prefer this type-erased closure if absent.]**

**Retry/budget interaction (rustdoc + test).** Each schema-mismatch retry is another LLM turn that spends tokens and consumes the per-call `max_total_tokens` and `max_turns`. Map (a) jsonschema-mismatch-then-`max_turns` and (b) `BudgetExceeded` to **`Err`**; reserve `Ok(None)` strictly for cancellation/skip — never collapse a persistently-invalid item silently to `None`.

### 5.8 `RunJournal` / resume (P4)

Content-addressed memo with **longest-unchanged-PREFIX** semantics over a crash-tolerant JSONL run directory.

```rust
// flow/journal.rs
pub struct RunJournal {
    run_dir: std::path::PathBuf,
    entries: std::sync::RwLock<std::collections::HashMap<CallKey, AgentOutput>>,
    log: std::sync::Mutex<std::fs::File>,
    _lock: std::fs::File,             // File::try_lock — stable Rust 1.89 (toolchain is 1.95.0)
    mode: ResumeMode,
}

#[derive(Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
struct CallKey { call_index: u64, content_hash: [u8; 32] }

#[derive(serde::Serialize, serde::Deserialize)]
struct JournalRecord { v: u16, key: CallKey, model_name: Option<String>, output: AgentOutput }
// AgentOutput ALREADY derives Serialize+Deserialize+Default and is #[non_exhaustive] → ZERO new derives.

pub enum ResumeMode { Fresh, Resume }

impl RunJournal {
    pub fn open(base: &std::path::Path, run_id: &str, mode: ResumeMode) -> Result<Self, Error> {
        // try_lock <run_dir>/.lock → Error::Cancelled (mapped) on contention; replay JSONL skipping a torn last line
        # unimplemented!()
    }
    /// LAZY: a HIT clones the cached AgentOutput and DROPS the guard BEFORE any .await, returning
    /// WITHOUT polling run_live (0 tokens, NOT re-charged to the budget). A MISS runs live, appends
    /// one JSONL line via a SINGLE write_all(record + b"\n") to the O_APPEND fd, then inserts.
    pub(crate) async fn memoize<F, Fut>(&self, _call_index: u64, _content_hash: [u8; 32], _run_live: F)
        -> Result<AgentOutput, Error>
    where F: FnOnce() -> Fut, Fut: std::future::Future<Output = Result<AgentOutput, Error>> + Send {
        # unimplemented!()
    }
}

/// Deterministic — NO RNG, NO clock. Same workflow + same args ⇒ same run_id ⇒ same dir ⇒ 100% prefix cache hit.
pub fn run_id(workflow_name: &str, args: &serde_json::Value) -> String {
    // uuid::Uuid::new_v5(HB_NS, canonical(name + canonical_json(args))) — requires uuid "v5" feature
    # unimplemented!()
}

/// Hash INPUTS only — the REQUESTED model, NEVER AgentOutput.model_name (an OUTPUT set by the cascade).
/// MUST canonicalize first: the workspace enables serde_json preserve_order, so rebuild every object via
/// BTreeMap (provably deep sorted keys) before hashing, else logically-equal inputs hash differently (false MISS).
fn content_hash(/* prompt, requested_model, system_prompt, sorted_tool_names, schema, phase */) -> [u8; 32] {
    # unimplemented!()  // blake3 (add) or sha2 (present); 0x00 domain-separators between fields
}
```

**Why prefix, not pure content-addressing.** `content_hash` answers "did call N change"; `call_index` (issue-order) enforces "and everything after re-runs". A pure content-addressed memo would wrongly still hit on *reorder*; the `call_index` tie-breaker also correctly distinguishes byte-identical calls in a loop.

**Honest caveats (rustdoc + every call site):**

- Restores **return values**, not side effects. `AgentOutput.tool_call_results` restores the *record* of tool calls, not their effects (file writes, worktree mutations, messages are not re-executed). Resume assumes those persisted on disk/in the repo. This is the single most important limitation and the precise reason **worktree mutators are the risky case** — co-design P4 + P5 (see §7 risk: refuse to journal/replay any `isolation != None` call, or fail-closed when `Resume` is combined with worktree isolation).
- **Concurrency destabilizes the key.** `call_index = agent_seq.fetch_add(Relaxed)` races inside *any* `parallel()`/`pipeline()` region, so there the scheme degrades to content-addressed memoization. This is **safe** (a HIT needs both `call_index` *and* a collision-resistant `content_hash`, so a raced index is a false-MISS, never a false-HIT) but means **prefix-resume is call-for-call only for strictly-sequential regions.** Mitigation: use a structural/path-based key (region-id + within-region submission ordinal) instead of a global counter, or document the sequential-only soundness bound.
- Rust cannot compile-forbid `Instant::now()`/`rand` in a workflow body; this is documented discipline + an optional debug-mode divergence check.

### 5.9 Worktree isolation (P5)

Opt-in `isolation: Worktree` for parallel file mutators. The seam **already exists** — `WorktreeGuard` produces a `PathBuf` that slots into `SubAgentConfig.workspace` (consumed at dispatch), and `BuiltinToolsConfig.sandbox_policy = Some(SandboxPolicy::workspace_only(path))` jails the bash child via the existing per-spawn `pre_exec(landlock_pre_exec)`. File read/write/edit tools jail on the `workspace` `PathBuf` via `resolve_within`.

```rust
// flow/worktree.rs
pub struct WorktreeGuard {
    repo_path: std::path::PathBuf,    // stores ONLY Send plain data: git2::Worktree is !Send/!Sync
    name: String,                     // deterministic from (run_id, label, agent_index) — NEVER uuid/timestamp
    worktree_path: std::path::PathBuf,
    cleaned: bool,
}

#[non_exhaustive]
pub enum WorktreeOutcome { Pruned, Kept { path: std::path::PathBuf, branch: String } }

impl WorktreeGuard {
    /// git2::Repository::open + repo.worktree(..) INSIDE tokio::task::spawn_blocking (re-open repo locally).
    pub async fn create(repo_path: std::path::PathBuf, name: String, base: &std::path::Path)
        -> Result<Self, Error> { # unimplemented!() }
    pub fn workspace(&self) -> std::path::PathBuf { self.worktree_path.clone() }  // → SubAgentConfig.workspace
    /// Dirty-check INCLUDING untracked (StatusOptions::include_untracked(true) — else deletes new files);
    /// prune if clean, Kept{branch}(+optional unified-diff patch) if dirty.
    pub async fn cleanup(self) -> Result<WorktreeOutcome, Error> { # unimplemented!() }
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        if self.cleaned { return; }
        // BEST-EFFORT SYNC backstop ONLY: std::fs::remove_dir_all + sync git2 metadata prune + tracing::warn.
        // NEVER .await / spawn_blocking / block_on. (Synchronous git2 prune on the current thread is legal —
        // !Send only forbids moving the handle across threads/await, not a same-thread call.)
        # unimplemented!()
    }
}
```

**Async-RAII trap (must handle, see §7).** `cleanup()` is async and runs only on the happy path; `Drop` is sync best-effort. `parallel()`/`pipeline()` *abort* child tasks mid-run on outer-future drop, so async cleanup may never run; and a `spawn_blocking` create-closure can complete *after* its `await` was cancelled, orphaning a registration with **no Rust owner**. Mitigations: (1) close the create-window (allocate guards before entering the abortable scope, or construct an owning handle synchronously); (2) make deterministic names **collision-tolerant on resume** (adopt/prune a pre-existing same-named `.git/worktrees/<name>`, since `git2::Repository::worktree(name)` errors on a duplicate); (3) a startup sweep that prunes orphaned `.git/worktrees/*` matching the run namespace. **Test fixture gotcha:** a worktree cannot be created on a repo with unborn HEAD — the fixture must `init` + add + commit once first.

**Scope honesty.** This points `SandboxPolicy` at the worktree for the *bash child* only; reversible per-tokio-thread Landlock for the non-bash file tools is a pre-existing separate concern, and worktree isolation does **not** contain non-file side effects (network, messages). `git2` is a **genuinely new dependency** (verified absent) — add `git2 = "0.19"` to `[workspace.dependencies]` + heartbit-core.

### 5.10 Sub-workflow nesting (1 level) + dynamic patterns

```rust
// flow/mod.rs
pub async fn workflow<R, F, Fut>(ctx: &WorkflowCtx, _name: impl Into<String>, body: F) -> Result<R, Error>
where F: FnOnce(WorkflowCtx) -> Fut, Fut: std::future::Future<Output = Result<R, Error>> {
    let child = ctx.nested()?;   // depth+1; Err(Error::Config) on a second level. Shares budget/sem/counter/cancel/journal by Arc.
    body(child).await
}
```

Every Claude Code quality pattern is plain async Rust over the combinators — **no new primitive**:

```rust
// loop-until-budget + loop-until-dry
let mut empties = 0;
while ctx.budget().total().is_some() && ctx.remaining() > 50_000 {
    let round = parallel(&ctx, mk_thunks(&ctx)).await;
    if round.iter().all(Option::is_none) { empties += 1; if empties >= K { break; } } else { empties = 0; }
}

// static fleet scaling
let fleet = ctx.budget().total().map_or(5, |t| (t / 100_000) as usize).max(1);

// dynamic fan-out over a model-discovered work-list (schema forces a validated object)
let work: Vec<Item> = agent(&ctx, discover_prompt)
    .schema::<WorkList>().run().await?
    .map(|w| w.items).unwrap_or_default();
let results = pipeline(&ctx, work, stages).await;

// judge panel: N attempts → score → synthesize from the winner, grafting runners-up
let attempts = parallel(&ctx, (0..n).map(|i| {
    let c = ctx.clone();
    move || async move { agent(&c, attempt_prompt(i)).run().await }
}).collect()).await;

// adversarial verify: N skeptics, refute-by-default; pure-Rust majority kills the finding
let verdicts = parallel(&ctx, skeptic_thunks(&ctx)).await;
let killed = verdicts.iter().flatten().filter(|v| v.is_unsafe()).count() * 2 > verdicts.len();
```

### 5.11 Adapter for existing typed workflow agents (honest interop)

Because `SequentialAgent<P>`/`ParallelAgent<P>`/… are `AgentRunner<P>`-by-value generics with no `dyn` boundary, embedding one as a combinator stage needs an explicit adapter that also does the permit/budget/event bookkeeping `agent()` does:

```rust
pub fn run_subagent(
    ctx: &WorkflowCtx,
    exec: impl FnOnce(&str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<AgentOutput, Error>> + Send + '_>>,
    input: &str,
) -> impl std::future::Future<Output = Result<AgentOutput, Error>> { /* permit + backstop + budget + events */ # unimplemented!() }

pub fn stage_from_runner(ctx: &WorkflowCtx, runner: std::sync::Arc<AgentRunner<BoxedProvider>>) -> Stage<serde_json::Value> { # unimplemented!() }
```

> **Critical (from adversarial verdict A2):** `run_subagent` must **not** wrap a single permit around a whole multi-agent `.execute()` chain (e.g. `SequentialAgent::execute`) — that under-counts in-flight agents and is a latent deadlock if those inner agents ever acquire flow permits. Either document that embedded legacy agents do **not** participate in the flow cap, or re-route each inner agent through `agent()`. Pick one and forbid the permit-around-chain form in rustdoc + a debug-only re-entrancy guard (a task-local "permit held" flag that `debug_assert!`s on re-entry).

---

## 6. Claude Code → Heartbit Primitive Mapping

| CC primitive | Heartbit Rust equivalent | Status | Notes |
|---|---|---|---|
| `agent(prompt, opts) -> result` | `agent(&ctx, p).label/.phase/.model/.isolation.run()` over `AgentRunner<P>::execute` | **build-new** (wrapper) | `Ok(None)` = null; opts is `#[non_exhaustive]`; **zero** runner change for MVP |
| `agent(..).schema` (forced object) | `.schema::<T>()` → `structured_schema(Value)` + `__respond__` + retry + `from_value::<T>` | **extend** (thin) | machinery exists; add optional `schemars` feature; `jsonschema` already validates |
| `parallel(thunks) -> Option[]` (barrier, fail-soft) | `parallel(&ctx, thunks) -> Vec<Option<R>>` (`JoinSet`, `Err`/panic → `None`, submission order) | **build-new** | dual of fail-FAST `ParallelAgent`; reuses `runner.rs` drain idiom |
| `pipeline(items, ...stages)` (no barrier) | `pipeline(&ctx, items, stages) -> Vec<Option<Value>>` (one future/item folds chain; all concurrent) | **build-new** | NOT `SequentialAgent` (barrier); flow type = `Value`; typed/streamed variants opt-in |
| `phase(title)` / `log(msg)` | `phase(&ctx, t) -> PhaseGuard` (RAII) / `log(&ctx, m)`; `WorkflowEvent` plane | **build-new** | separate `#[non_exhaustive]` enum; per-call phase snapshot |
| `workflow(nameOrRef, args)` (1-level) | `workflow(&ctx, name, body)`; `ctx.nested()` shares handles; `depth` guard | **build-new** | second level → `Error::Config` |
| `budget {total, spent, remaining}` (shared hard ceiling) | `Budget(Arc<AtomicU64> + Option<NonZeroU64>)`; `check_admit`/`record_weighted` | **build-new** | reuse `Error::BudgetExceeded`; cost-weighted; record-only; maps 1:1 to Restate `TokenBudgetObject` |
| `min(16, cores-2)` concurrency | one shared `Arc<Semaphore>`, `acquire_owned` at the agent() leaf | **build-new** | copy `batch.rs`; thread same `Arc` into nested workflows |
| 1000-agent backstop | shared `Arc<AtomicU64>` + `Error::AgentBudgetExceeded { limit }` | **build-new** | distinct from token `BudgetExceeded` |
| `isolation:"worktree"` | `WorktreeGuard` (git2 + `spawn_blocking`) → `SubAgentConfig.workspace` + `SandboxPolicy::workspace_only` | **build-new** | add `git2`; seam already consumed at dispatch; dual cleanup |
| resume / `runId` / `resumeFromRunId` | `RunJournal` (ordered JSONL, `(call_index, content_hash)` prefix) + `uuid::new_v5(run_id)` | **build-new** | add `uuid` v5; reuse `sha2`/add `blake3`; canonicalize before hashing |
| `/workflows` view + pause/resume/stop/restart | `WorkflowEvent`/`OnWorkflowEvent` + `CancellationToken` (pause/stop) + per-task `AbortHandle` (restart-one) | **build-new** | needs a handle-exposing `parallel` variant for restart-one (see §7 A8) |
| structured retry-on-mismatch | already in `runner.rs` (`__respond__` validate + `is_error` retry) | **exists** | typed path bounds retries against `max_turns`/budget |
| `acceptEdits` + inherit allowlist | sub-agents inherit `base_tools` / `opts.tools`; existing permission model | **exists** | combinator threads tools; no new permission machinery |
| forbid wall-clock / RNG | documented discipline + optional debug divergence check + journaled `ctx.now()`/`ctx.rand_uuid()` | **extend** | Rust cannot compile-forbid; honesty in rustdoc |

---

## 7. Feasibility and Risks

### 7.1 Adversarial verdicts on the 8 load-bearing assumptions

| ID | Assumption (abbrev) | Verdict | Key caveat | Mitigation |
|---|---|---|---|---|
| **A1** | `pipeline()` = one-future-per-item folded chain, all concurrent ⇒ wall-clock = slowest single chain (not sum-of-slowest-per-stage) | **confirmed** (empirical probe: per-item 341 ms vs barrier 902 ms; "results correct" passes for BOTH impls — the silent trap) | literal equality false under semaphore saturation (→ `~total-work/cap`); `JoinSet` ≠ `FuturesUnordered` ≠ `buffer_unordered` for panic isolation | implement as `JoinSet` per-item folded chain; test the **interleaving** invariant via an ordering channel, never wall-clock asserts; document the cap-degradation |
| **A2** | one shared `Arc<Semaphore>` (`min16,cores-2`), `acquire_owned` at leaf only, shared into nested workflows, caps in-flight without deadlock/throttle | **uncertain** | leaf-only is sound, but `run_subagent` wrapping a permit around a whole legacy `.execute()` chain breaks it (under-count + latent deadlock); cap is on agent CALLS, not tool tasks | forbid permit-around-chain; debug-only task-local re-entrancy guard; nested-deadlock regression test (cap < N, parallel-of-parallel, assert completes + peak ≤ cap) |
| **A3** | record-only `Budget` over `AtomicU64`+`NonZeroU64`, check-at-admission/record-on-completion, overshoot ≤ `concurrency×cost`, reuse `BudgetExceeded` | **confirmed** (u32→u64 widening real & necessary; boundary `>` consistent across runner/Restate; `total_tokens` reliably populated; orderings sound) | over-admission bounded by `C×cost`; error/abort paths under-count (safe-side); journal-hit path must NOT record | keep leaf order `permit→backstop→check_admit→run→record`; 64-task stress test asserts `final spent == Σ recorded` AND a "passed-but-not-recorded" peak gauge ≤ `C`; record via `u.total() as u64` |
| **A4** | typed `schema::<T>()` over the existing `__respond__`/`validate_tool_input`/retry path with no runner change; `T` off all trait objects | **confirmed** (`structured_schema: Option<Value>` + forced `__respond__` + `jsonschema` validate + retry + `AgentOutput.structured` all exist; T in `PhantomData` only) | `.structured_validator::<T>()` does NOT exist (name); the additive closure is *unnecessary* for MVP (Ok-branch already validates) | MVP = set schema + `from_value::<T>`; before P3 grep for a builder setter; map max-turns-with-bad-output to `Err`, not `Ok(None)` |
| **A5** | `RunJournal` prefix-resume sound despite LLM nondeterminism: `(call_index, content_hash-over-canonicalized-inputs)`, honestly scoped | **uncertain** (no unsound false-HIT path given collision-resistant hash + compound key + mandatory canonicalization; preserve_order IS on → canonicalization mandatory) | "prefix" over-stated under concurrency (degrades to content-addressed memo — safe false-MISSes); worktree-pruned-then-replay is genuinely unsound if not guarded | reword prefix→"content-addressed memo with sequential-prefix ordering"; **runtime guard**: never journal/replay `isolation != None` calls (or fail-closed); property-test canonicalization; test no-double-charge + no-poll-on-hit |
| **A6** | purely additive: no `Agent` trait; `AgentRunner<P>::execute` is the unit; fail-soft `parallel` coexists with fail-fast `ParallelAgent`; separate `WorkflowEvent` because `AgentEvent` is NOT `#[non_exhaustive]` | **confirmed** (grep: zero `Agent` trait in core; `SequentialAgent`=`Vec<AgentRunner<P>>`, `ParallelAgent`=`Vec<Arc<AgentRunner<P>>>`, `LoopAgent`=`AgentRunner<P>`; `AgentEvent` has only `allow(missing_docs)`+derives+serde tag, no `non_exhaustive`) | "compiles byte-for-byte unchanged" holds for the spine; P3's optional validator hook may touch `builder.rs`/`runner.rs` (still additive API) | regression test constructing existing agents via public API + calling `.execute`; never extend `AgentEvent`; `cargo public-api`/doc-diff to assert no existing export changed |
| **A7** | `isolation:Worktree` composes with existing seams (no new runner plumbing): `WorktreeGuard.workspace()`→`SubAgentConfig.workspace`, `SandboxPolicy::workspace_only`→bash child; git2 in `spawn_blocking`; dual cleanup | **uncertain** (injection/composition half CONFIRMED: both fields exist, flow per-dispatch into a fresh `BuiltinToolsConfig`, bash applies Landlock per-spawn via `pre_exec`; cleanup/abort clause NOT established) | guaranteed orphan if aborted during `create().await` (no Rust owner → neither cleanup nor Drop fires); `remove_dir_all` misses `.git/worktrees/<name>` admin entry; deterministic names collide with leaked registrations on resume | close the create-window; collision-tolerant adoption on resume; startup sweep of orphaned worktrees; Drop prunes BOTH workdir and admin entry; explicit abort-path test |
| **A8** | three-layer cancellation (`CancellationToken` cooperative + `JoinSet abort_all` forceful + per-task `AbortHandle`) implements `/workflows` controls, partial usage attached on abort; `Cancelled`/`AgentBudgetExceeded` must be added | **uncertain** (error-variant facts CONFIRMED: `Cancelled` absent, `AgentBudgetExceeded` absent, `RunTimeout`/`BudgetExceeded` present, `accumulate_usage`/`with_partial_usage` present, `JoinSet::abort_all` already used, daemon already does cooperative-then-abort; the "layering correct" claim has two gaps) | **Gap 1:** leaf `select!`-drop discards the in-flight future → partial tokens lost on cancel (the runner must be threaded the token to return `Err(Cancelled).with_partial_usage`). **Gap 2:** `parallel()` returns `Vec<Option<R>>` and never exposes `AbortHandle`s → restart-one unreachable through the API | thread the token INTO `AgentRunner` (small additive builder field, default None) so cancel returns partial usage; OR document leaf-cancel forfeits partial tokens; add a `parallel_with_controls(..) -> (fut, Vec<AbortHandle>)`; add both Error variants; RED test asserting cancel → `Cancelled` + `partial_usage() > 0` |

**Net:** 4 confirmed (A1, A3, A4, A6), 4 uncertain-but-mitigable (A2, A5, A7, A8). None refuted. The uncertain four are all *characterization/enforcement* gaps with concrete mitigations, and three of them (A5/A7/A8) concentrate in the two late, optional phases (journal, worktree) and the observability/cancellation phase — exactly the work deferred to P4–P6 so it cannot destabilize the spine.

### 7.2 Completeness-critic missed risks (and resolutions)

1. **Cancellation through pipeline stages is unspecified.** Resolved by §5.2.1 (cancel race at the leaf), §7 A8 (thread token into runner for partial usage), and the sticky-error rule. *Open:* whether the runner gets the token (small additive change) or leaf-cancel forfeits partial tokens — decided in P6.
2. **Spawned-counter + budget-admission run before the journal HIT.** Resolved: §5.2.1 reorders the journal HIT to step 1; replayed calls are exempt from both ceilings.
3. **Shared phase slot incoherent under concurrency.** Resolved: §5.5 snapshots phase per `AgentCall`; the shared slot only holds the *default* for subsequently-issued calls.
4. **Nested-workflow + shared semaphore deadlock.** Resolved by the leaf-only acquire invariant + the `run_subagent` constraint (A2); add the nested-deadlock regression test.
5. **Cache-aware cost mis-metered.** Resolved: §5.6 charges `weighted_cost` (cache-read 0.1×, cache-write 1.25×), reusing `estimate_cost`; raw-sum rejected.
6. **Partial-failure→None defeats the hard ceiling.** Resolved by the **sticky-error rule** (§5.2.1, §5.3): control errors propagate past the join and trip cancellation; only agent-domain errors collapse to `None`.
7. **Global `call_index` race breaks the journal key under concurrency.** Resolved/scoped: §5.8 documents prefix-soundness for sequential regions only and proposes structural keys; the race is *safe* (false-MISS not false-HIT).
8. **Tokio semaphore fairness/starvation asserted not reasoned.** Acknowledged: acquire is FIFO; cancel-loses-position only on teardown; long worktree agents holding permits is a real head-of-line concern — documented, and the peak-gauge + "no-throttle-below-cap" tests cover the throughput claim.
9. **Restate durable contract punted.** Resolved as a **named non-goal** (§8 P7): process-local atomic + local JSONL have no meaning under Restate replay; on the durable path the combinator journal must be disabled or bridged (no double-journaling), since Restate's event-journal + `TokenBudgetObject` already do replay + budget.
10. **The load-bearing pipeline test is inherently flaky.** Resolved: §8 P1 testing strategy mandates ordering-channel / `Barrier`/`Notify` rendezvous + single-thread-or-time-paused runtime, never wall-clock asserts.
11. **Worktree cleanup leaks on failure/abort/panic.** Resolved by §5.9 + A7 mitigations (close create-window, collision-tolerant adoption, startup sweep, dual-prune Drop).
12. **Structured-output retry vs ceiling/max_turns.** Resolved: §5.7 maps exhausted retries to `Err`, bounds against `max_turns`/budget.

---

## 8. Iterative Roadmap (P1–P7)

Each phase is independently shippable and TDD-first. Gate for every merge: `cargo fmt -- --check && cargo clippy -- -D warnings && cargo test` (zero warnings). Tests live in-file in `#[cfg(test)] mod tests`, using the existing `crate::agent::test_helpers::{MockProvider, make_agent}` — **zero network**.

### P1 — Combinator core (WorkflowCtx + parallel + pipeline + phase)

- **Goal.** Close the run-wide concurrency-cap gap and land the no-barrier pipeline + fail-soft parallel on top of existing agents.
- **Deliverables.** `flow/ctx.rs` (`WorkflowCtx`, `WorkflowCtxBuilder`, default cap via `available_parallelism`), `flow/parallel.rs`, `flow/pipeline.rs` (default `JoinSet` + opt-in `pipeline_streamed`), `flow/event.rs` (`WorkflowEvent` `#[non_exhaustive]`, `OnWorkflowEvent`, `phase()`/`PhaseGuard`/`log()`), `flow/agent.rs` (`agent()` text terminal, leaf sequence sans budget/journal), the 1000-agent backstop (`Error::AgentBudgetExceeded`), `Error::Cancelled`. Re-exports in `lib.rs`.
- **Tests-first (RED).** (a) `parallel`: N successes → N `Some` in submission order; one `Err` → that slot `None`, others unaffected, call returns `Ok`-Vec; one **panic** → that slot `None` (assert isolation; contrast with `parallel_error_fails_fast`); staggered completion preserves positions. (b) `pipeline` **interleaving** via an ordering channel: item B reaches stage 2 before item A finishes stage 2 (and the contrast: a per-stage-barrier impl FAILS this); a stage `Err` drops only that item and skips its remaining stages (assert via a shared `AtomicUsize` that later stages never ran for the failed item) while siblings complete. (c) concurrency cap: > cap concurrent leaves, peak-in-flight via `fetch_max` `AtomicUsize` ≤ cap; a long pipeline does **not** throttle below cap (acquire-at-leaf, not around the chain). (d) backstop: the `(max_agents+1)`-th leaf → `AgentBudgetExceeded`.
- **Acceptance.** No-barrier proven by interleaving (not wall-clock); peak ≤ cap; backstop fires; `parallel` never rejects.
- **Dependencies.** **Zero new** (tokio, tokio-util, futures, parking_lot, std atomics all present).
- **Effort.** ~3–5 days.

### P2 — Shared Budget (hard-ceiling pool)

- **Goal.** Thread one shared cost-weighted ceiling through the ctx and nested workflows.
- **Deliverables.** `flow/budget.rs` (`Budget`, `check_admit`, `record_weighted`, `weighted_cost` delegating to re-exported `estimate_cost`), wire steps 2/4/7 into the leaf sequence, sticky-error propagation on `BudgetExceeded`/`AgentBudgetExceeded`/`Cancelled`.
- **Tests-first (RED).** (a) `check_admit` under limit `Ok`, at/over limit `Err` with `used`/`limit`; `total=None` always `Ok`, `remaining()==u64::MAX`. (b) `record` accumulates in `u64` widened from `u32` (large-value no-overflow). (c) `weighted_cost`: a cache-read-heavy usage costs ~0.1× via `estimate_cost`; unknown model falls back per policy. (d) **64-task stress**: `final spent == Σ recorded` AND a "passed-but-not-recorded" peak gauge ≤ `C` (the real overshoot bound). (e) a stage that throws `BudgetExceeded` aborts the run (sticky) and is NOT silently `None`.
- **Acceptance.** Bounded overshoot proven; `loop-until-budget` terminates; sticky control errors halt the run.
- **Dependencies.** Zero new (reuse `Error::BudgetExceeded`, `estimate_cost`).
- **Effort.** ~3–4 days.

### P3 — Per-stage typed structured output

- **Goal.** `agent(&ctx, p).schema::<T>().run() -> Result<Option<T>, Error>`.
- **Deliverables.** `flow/agent.rs` typed terminal + `schema::<T>()` / `schema_value(Value)` behind a `derive-schema` feature; optional in-loop deserialize-validator (type-erased closure into the `__respond__` Ok-branch) if a builder hook is absent.
- **Tests-first (RED).** typed round-trip; `__respond__`-skipped → `Error::Agent`; deserialize-mismatch → `Error::Json` (hard); the runner ALREADY auto-retries a jsonschema mismatch (assert existing behavior); nested struct + enum + `Option` round-trips; recursive-type guard (fall back to non-inlined / `Value`); (P3-hardening) bad-enum first `__respond__`, valid second → exactly one retry, typed result, bounded by `max_turns`/`max_total_tokens`.
- **Acceptance.** Typed output validated + retried; persistently-invalid item surfaces `Err`, never silent `None`; object-safety preserved.
- **Dependencies.** Add `schemars` (optional, `derive-schema` feature) to `[workspace.dependencies]` + heartbit-core. `jsonschema = "0.28"` already present. **[verify-at-impl: schemars version emits a schema the 0.28 validator accepts; confirm a public `structured_schema` builder setter.]**
- **Effort.** ~3–4 days (MVP), +2 days for the in-loop validator.

### P4 — RunJournal / resume

- **Goal.** Deterministic longest-unchanged-prefix replay; same script + args = 100% cache hit.
- **Deliverables.** `flow/journal.rs` (single-process `RunJournal` JSONL + `(call_index, content_hash)` + `canonical_json` + deterministic `run_id` via `uuid::new_v5`), wire the journal HIT as leaf step 1 (exempt from ceilings) + append on miss; the runtime guard refusing to journal/replay `isolation != None`. Later sub-steps: cross-process `try_lock`; optional compacted snapshot.
- **Tests-first (RED).** same inputs twice → second replays WITHOUT polling the closure (assert a flag inside the closure) AND `Budget::spent()` unchanged across the replayed prefix; changed input at k → `<k` replay, `k..` live; reordered logically-equal opts still HIT after canonicalization; a truncated final JSONL line is skipped on load and that call re-runs; `new_v5` run-id identical across two constructions; two `RunJournal`s on the same dir → second `try_lock` fails → `Error::Cancelled`(mapped)/`RunLocked`; cascade run twice (model escalates) still HITs (key on requested model, not `model_name`).
- **Acceptance.** Lazy hit costs 0 tokens + 0 budget; canonicalization defeats reorder false-misses; resume call-for-call for sequential regions, documented coarser for concurrent regions.
- **Dependencies.** Add `uuid` `v5` feature (currently `v4`,`serde` only); reuse `sha2` (present) OR add `blake3`; promote `tempfile` from dev to deps if the snapshot sub-step ships.
- **Effort.** ~5–7 days (hardest correctness burden).

### P5 — Worktree isolation

- **Goal.** Opt-in `isolation: Worktree` for parallel file mutators.
- **Deliverables.** `flow/worktree.rs` (`WorktreeGuard::create`/`cleanup`/`Drop`, `worktree_is_dirty` incl. untracked, deterministic `worktree_name`), gate on `opts.isolation`, set `SubAgentConfig.workspace` + `SandboxPolicy::workspace_only`, startup sweep + collision-tolerant adoption; co-design with P4's journal guard.
- **Tests-first (RED).** create→clean→cleanup prunes (`.git/worktrees/<name>` gone, `worktrees()` no longer lists it); create→write→cleanup KEEPS (dirty incl. untracked); `Drop` without cleanup removes the dir (panic backstop); deterministic name for same inputs; the produced `workspace()` `PathBuf` confines writes; **abort path**: abort the outer future mid-`create()` and mid-`execute()`, then assert no orphaned `.git/worktrees/<name>` survives the sweep AND an immediate retry with the same `(run_id,label,index)` does NOT error "name already exists". Fixture: `init` + add + commit before `worktree(..)`.
- **Acceptance.** Clean→prune, dirty→keep+patch; abort leaves no leak after sweep; bash child jailed to the worktree.
- **Dependencies.** Add `git2 = "0.19"` to `[workspace.dependencies]` + heartbit-core (genuinely new). `tempfile` (present, dev) suffices for fixtures.
- **Effort.** ~5–7 days.

### P6 — Observability + sub-workflow nesting + cancellation controls

- **Goal.** A `/workflows` analog (phases, agent counts, token totals, elapsed; pause/resume/stop/restart-one) and one-level `workflow()` nesting.
- **Deliverables.** `flow/mod.rs` `workflow()` + `ctx.nested()` (depth guard, shared handles); event folding into a progress aggregate; `CancellationToken` pause/stop; a `parallel_with_controls(..) -> (fut, Vec<AbortHandle>)` for restart-one; (decision) thread the token into `AgentRunner` for partial-usage-on-cancel.
- **Tests-first (RED).** nested `workflow()` shares budget/cap/counter/cancel (assert one stop fires everywhere; a second nesting level → `Error::Config`); cancel mid-run → `Cancelled` + `partial_usage() > 0` (this FAILS under leaf-only select!-drop → drives the runner-token decision); abort one task while siblings complete; the progress aggregate reports correct phase/agent-count/token totals during a barrier (proves per-call phase snapshot).
- **Acceptance.** Controls work; nesting shares the pool; partial usage preserved on cancel (or explicitly documented as forfeited).
- **Dependencies.** Zero new (`tokio-util` present); optional small additive `AgentRunnerBuilder::cancellation_token`.
- **Effort.** ~4–6 days.

### P7 — Bridge to LLM orchestrator + Restate durable parity

- **Goal.** Document and wire the boundary (code-decided fan-out → combinators; model-decided → orchestrator) and a durable-path bridge.
- **Deliverables.** rustdoc + examples mapping `DelegateTaskTool`/`FormSquadTool`/`SpawnAgentTool` to the combinator boundary; a `SharedBudget` trait with the std atomic impl (standalone) and a thin Restate `TokenBudgetObject` wrapper (durable, record-only, zero new handlers, one fixed object key = runId); a no-double-journaling rule disabling the combinator journal on the durable path; named non-goals for V1.
- **Tests-first (RED).** orchestrator dispatch still works (regression); `SharedBudget` trait: record→exceed→`TerminalError` on the durable impl; replay-determinism of journaled budget arithmetic; a combinator run on the durable path does NOT double-journal.
- **Acceptance.** Boundary documented + tested; durable budget parity via the trait; no journal conflict.
- **Dependencies.** Restate SDK 0.8 (present in heartbit umbrella); reserve-then-reconcile (opt-in) needs a new `try_reserve` handler — defer.
- **Effort.** ~5–8 days.

---

## 9. Open Questions and Decisions for the User

1. **Budget unit — COST or RAW tokens?** ✅ **RESOLVED in P2 (commit `f3f32db`) → cost-weighted token-equivalents** (not micro-USD). The original micro-USD recommendation was *incoherent*: its `weighted_cost` returned micro-USD for priced models (`estimate_cost` → `Some`) but a raw token count for unknown/mock models (the `None` fallback), so a single `spent` accumulator could not mean both. Shipped unit (`flow/budget.rs::weighted_cost`) is one model-independent **token-equivalent**: `input + output + reasoning + cache_write×1.25 + cache_read×0.1` — coherent across priced and mock models, matching the existing `max_total_tokens` in+out convention. `max_total_tokens` stays a separate per-runner cap. Switching to true micro-USD later is a one-function swap, once every model (incl. test mocks) is priced. *(Open only if budgets should be expressed in dollars on the CLI; the engine unit is settled.)*
2. **Budget strictness — record-only default (opt-in reserve-reconcile)?** Record-only over-admits by up to `concurrency×cost` (identical to the target, zero new Restate handlers). Do any fleets need a strict never-exceed ceiling from day one (adds `Reservation`/`Drop`/estimate + a new Restate `try_reserve` handler)?
3. **Default pipeline driver — `JoinSet` (panic isolation, `'static`+`Send`) vs the opt-in `buffer_unordered` stream (backpressure, but cooperative-poll panic semantics)?** Recommendation: `JoinSet` default, `pipeline_streamed` opt-in for very large dynamic work-lists. Confirm — it shapes the most-used combinator's failure semantics.
4. **Typed schema ergonomics — schemars feature vs hand-written?** ✅ **RESOLVED in P3 → a dep-free `StructuredSchema` trait, NO schemars.** Investigation found `schemars` already in the lockfile at three conflicting versions (0.8.22, 0.9.0, 1.2.1); pulling it into `heartbit-core` (zero optional deps, validates with `jsonschema 0.28`) would risk a schema-dialect mismatch. Shipped: `pub trait StructuredSchema: DeserializeOwned + Send { fn json_schema() -> Value }` (`flow/agent.rs`); `agent(&ctx,p).schema::<T>()` for any `T: StructuredSchema`, `.schema_value(Value)` for an ad-hoc schema. Author `json_schema()` by hand now, or via a future `#[derive(StructuredSchema)]` in `heartbit-macro` — still no schemars. Zero runner change: reuses the existing `__respond__` + `validate_tool_input` + retry path, then `from_value::<T>` (serde enforces what JSON Schema can't — enum variants, integer widths). `T` is phantom on `AgentCall<T>`, so the runner stays `Value`-only/object-safe.
5. **Resume determinism contract — documented discipline + debug divergence check, or ship journaled `ctx.now()`/`ctx.rand_uuid()` so the common clock/UUID cases are enforced?** Rust cannot compile-forbid `Instant::now()`/`rand`; budget-driven loops resume only at coarse granularity regardless.
6. **V1 scope — standalone-path only (same boundary as guardrails) acceptable?** The durable Restate path runs tools sequentially for journal ordering and is not replay-durable with these combinators yet; durable parity is a deliberate later phase (P7), not coupled into the join semantics now.
7. **Worktree priority — ship P1–P4 first and rely on the existing per-agent workspace `PathBuf` + Landlock until P5, or is git-worktree-per-mutator a hard requirement for an early milestone?** P5 is the highest-cost, lowest-priority phase (new `git2` dep, `!Send`, async-RAII trap, ~200–500 ms + disk per mutator) and only benefits *parallel* file mutators.
8. **API shape — free functions taking `&ctx` first (`agent(&ctx,p).schema::<T>().run()`, `pipeline(&ctx,items,stages)`) with thin `impl WorkflowCtx` delegators, or a method-primary API (`ctx.pipeline(..).stage(..).run()`) despite the borrow-checker friction when a stage closure both captures `ctx` and is passed into a combinator?** Recommendation: free-function primary.
9. **Cancellation partial-usage — accept a small additive `AgentRunnerBuilder::cancellation_token` so a cancelled agent returns `Err(Cancelled).with_partial_usage(..)`, or document that leaf-cancel forfeits the in-flight agent's partial tokens?** (A8 Gap 1.)
10. **`/workflows` restart-one — add a handle-exposing `parallel_with_controls(..) -> (fut, Vec<AbortHandle>)`, or scope V1 controls to stop-all only and route restart-one through re-running the journal with the failed leaf marked stale?** (A8 Gap 2.)

---

## 10. Sources

### Heartbit source references (key file:symbol)

- `crates/heartbit-core/src/agent/runner.rs` — `AgentRunner<P>::execute(&self, &str) -> Result<AgentOutput, Error>`; `AgentOutput` (Serialize/Deserialize/Default, `#[non_exhaustive]`, `.result/.tokens_used/.structured/.model_name/.tool_call_results`); `structured_schema: Option<serde_json::Value>` + `__respond__` injection + `validate_tool_input` + retry + "without calling `__respond__`" hard error; `max_total_tokens` post-turn cap → `Error::BudgetExceeded.with_partial_usage(..)`; `execute_tools_parallel` submission-order `JoinSet` drain.
- `crates/heartbit-core/src/agent/workflow.rs` — `SequentialAgent<P>` (`Vec<AgentRunner<P>>`, BARRIER, pipes `result`); `ParallelAgent<P>` (`Vec<Arc<AgentRunner<P>>>`, `JoinSet`, FAIL-FAST `?`, name-sorted merge); `LoopAgent<P>`; `parallel_error_fails_fast` / `parallel_output_sorted_by_name` tests.
- `crates/heartbit-core/src/agent/batch.rs` — `Arc<Semaphore::new(max_concurrency)>`, `std::thread::available_parallelism().map(|n| n.get()).unwrap_or(..)`, `acquire()`, `JoinSet`, current/peak `AtomicUsize` peak-gauge test.
- `crates/heartbit-core/src/agent/cache.rs` — `ResponseCache` LRU `Mutex<Vec<(u64, CompletionResponse)>>`, `fnv1a_hash` key (NOT a resumable journal).
- `crates/heartbit-core/src/agent/events.rs` — `AgentEvent` (`#[allow(missing_docs)]` + derives + `#[serde(tag="type")]`, **NOT** `#[non_exhaustive]`), `OnEvent` type alias.
- `crates/heartbit-core/src/agent/orchestrator.rs` — `SubAgentConfig.workspace: Option<PathBuf>` + `.sandbox_policy: Option<SandboxPolicy>` (consumed at dispatch); `DelegateTaskTool` (`DispatchMode::Parallel|Sequential`), `FormSquadTool`, `SpawnAgentTool`; `provider_override: Arc<BoxedProvider>`.
- `crates/heartbit-core/src/tool/mod.rs` — `validate_tool_input` → `jsonschema::validator_for` + `iter_errors` (skips uncompilable schema); `Tool` trait (`Arc<dyn Tool>`, `Pin<Box<dyn Future + Send>>`).
- `crates/heartbit-core/src/tool/builtins/mod.rs` / `bash.rs` — `BuiltinToolsConfig.workspace` + `.sandbox_policy`; `BashTool::with_sandbox` + per-spawn `pre_exec(landlock_pre_exec)`; `resolve_within`.
- `crates/heartbit-core/src/sandbox.rs` / `workspace.rs` — `SandboxPolicy::workspace_only(..)` / `from_path_policy`; `normalize_path` / `resolve_within`.
- `crates/heartbit-core/src/error.rs` — `Error::BudgetExceeded { used: u64, limit: u64 }`, `Error::WithPartialUsage`, `with_partial_usage()`/`accumulate_usage()`, `Error::RunTimeout`, `Error::Json (#[from])`, `Error::Config(String)`; **no** `Cancelled`/`AgentBudgetExceeded` (add).
- `crates/heartbit-core/src/llm/types.rs` — `TokenUsage` (u32 `input/output/cache_creation/cache_read/reasoning`, `Copy`, `AddAssign`, `total()`).
- `crates/heartbit-core/src/llm/pricing.rs` + `lib.rs` re-export — `estimate_cost(model, &TokenUsage) -> Option<f64>` (cache_read 0.1×, cache_write 1.25×, reasoning at output rate; `None` for unknown models).
- `crates/heartbit/src/workflow/agent_workflow.rs` — event-journal replay skipping completed activities (durable analog). `crates/heartbit/src/workflow/budget.rs` — `TokenBudgetObject` (`record_usage`, `set_limit`, `saturating_add`, `TerminalError` on `new_total > limit`).
- `crates/heartbit-core/src/agent/test_helpers.rs` — `MockProvider`, `make_agent` (TDD substrate, zero network).
- Manifests: root `Cargo.toml` (`jsonschema = "0.28"`, `sha2 = "0.10"`, `uuid features=["v4","serde"]` — **add v5**; **no git2/blake3**), `crates/heartbit-core/Cargo.toml` (`tempfile = "3"` dev; `tokio-util`/`futures`/`parking_lot`/`landlock` present; edition 2024). Toolchain: `rustc 1.95.0`.

### External documentation (aggregated, deduped)

- tokio `JoinSet` — https://docs.rs/tokio/latest/tokio/task/struct.JoinSet.html
- tokio `JoinError` (`is_panic`/`is_cancelled`) — https://docs.rs/tokio/latest/tokio/task/struct.JoinError.html
- tokio `AbortHandle` — https://docs.rs/tokio/latest/tokio/task/struct.AbortHandle.html
- tokio `Semaphore` (`acquire_owned`, `close`) — https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html
- tokio `OwnedSemaphorePermit` — https://docs.rs/tokio/latest/tokio/sync/struct.OwnedSemaphorePermit.html
- tokio `spawn_blocking` — https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html
- tokio `select!` — https://docs.rs/tokio/latest/tokio/macro.select.html
- tokio graceful shutdown — https://tokio.rs/tokio/topics/shutdown
- tokio_util `CancellationToken` — https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html
- futures `FuturesUnordered` — https://docs.rs/futures/latest/futures/stream/struct.FuturesUnordered.html
- futures `StreamExt::buffer_unordered` — https://docs.rs/futures/latest/futures/stream/trait.StreamExt.html#method.buffer_unordered
- futures `StreamExt::buffered` — https://docs.rs/futures/latest/futures/stream/trait.StreamExt.html#method.buffered
- futures `FuturesOrdered` — https://docs.rs/futures/latest/futures/stream/struct.FuturesOrdered.html
- futures `join_all` — https://docs.rs/futures/latest/futures/future/fn.join_all.html
- futures `try_join_all` — https://docs.rs/futures/latest/futures/future/fn.try_join_all.html
- futures `FutureExt::catch_unwind` — https://docs.rs/futures/latest/futures/future/trait.FutureExt.html#method.catch_unwind
- std `available_parallelism` — https://doc.rust-lang.org/std/thread/fn.available_parallelism.html
- std `AtomicU64` (`fetch_add`/`fetch_update`/`fetch_max`) — https://doc.rust-lang.org/std/sync/atomic/struct.AtomicU64.html
- std `AtomicU64::fetch_update` — https://doc.rust-lang.org/std/sync/atomic/struct.AtomicU64.html#method.fetch_update
- std `AtomicUsize` — https://doc.rust-lang.org/std/sync/atomic/struct.AtomicUsize.html
- std `NonZeroU64` — https://doc.rust-lang.org/std/num/struct.NonZeroU64.html
- std `Drop` — https://doc.rust-lang.org/std/ops/trait.Drop.html
- std `File` (`lock`/`try_lock`/`unlock`, stable 1.89) — https://doc.rust-lang.org/std/fs/struct.File.html
- blake3 — https://docs.rs/blake3/latest/blake3/
- tempfile `NamedTempFile` — https://docs.rs/tempfile/latest/tempfile/struct.NamedTempFile.html
- uuid `Uuid::new_v5` — https://docs.rs/uuid/latest/uuid/struct.Uuid.html
- serde_json `Map` (preserve_order / `sort_keys`/`sort_all_objects`) — https://docs.rs/serde_json/latest/serde_json/map/struct.Map.html
- serde_json `from_value` — https://docs.rs/serde_json/latest/serde_json/fn.from_value.html
- serde `de::DeserializeOwned` — https://docs.rs/serde/latest/serde/de/trait.DeserializeOwned.html
- schemars (`schema_for!`, `JsonSchema`, `SchemaSettings::inline_subschemas`) — https://docs.rs/schemars/latest/schemars/ ; macro: https://docs.rs/schemars/latest/schemars/macro.schema_for.html ; guide: https://graham.cool/schemars/ ; CHANGELOG (1.0 vs 0.8): https://github.com/GREsau/schemars/blob/master/CHANGELOG.md
- jsonschema (0.28 — `validator_for`/`iter_errors`) — https://docs.rs/jsonschema/latest/jsonschema/ ; pinned: https://docs.rs/crate/jsonschema/0.28.3
- git2 `Repository` — https://docs.rs/git2/latest/git2/struct.Repository.html
- git2 `Worktree` — https://docs.rs/git2/latest/git2/struct.Worktree.html
- git2 `WorktreeAddOptions` — https://docs.rs/git2/latest/git2/struct.WorktreeAddOptions.html
- git2 `WorktreePruneOptions` — https://docs.rs/git2/latest/git2/struct.WorktreePruneOptions.html
- git2 `StatusOptions` — https://docs.rs/git2/latest/git2/struct.StatusOptions.html
- git2 `Statuses` — https://docs.rs/git2/latest/git2/struct.Statuses.html
- git2-rs source (repo.rs) — https://github.com/rust-lang/git2-rs/blob/master/src/repo.rs
- libgit2 `git_worktree_add` — https://libgit2.org/docs/reference/main/worktree/git_worktree_add.html
- Git worktree manual — https://git-scm.com/docs/git-worktree
- fd-lock (MSRV < 1.89 fallback) — https://docs.rs/fd-lock
- loom (concurrency model checker, for the reserve-CAS variant) — https://docs.rs/loom/latest/loom/
- restate-sdk (0.8, virtual objects) — https://docs.rs/restate-sdk/latest/restate_sdk/
- Anthropic tool use (input_schema, strict, inline definitions) — https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Anthropic prompt caching (cache write ~1.25×, read ~0.1×) — https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Linux O_APPEND atomicity (≤ PIPE_BUF; ≥ 3.14 shared-fd fix; NFS no atomic append) — https://nullprogram.com/blog/2016/08/03/
- Evan Jones, "Durability: Linux File APIs" (fdatasync/fsync; fsync the containing directory) — https://www.evanjones.ca/durability-filesystem.html
