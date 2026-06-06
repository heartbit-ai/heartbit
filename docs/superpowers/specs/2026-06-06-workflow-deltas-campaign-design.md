# Workflow Deltas Campaign — Design

**Date:** 2026-06-06
**Status:** Goal-driven (user: "régler les deltas" + dynamic model-by-task + frontier advisor mode; Stop hook active)
**Sequence (advisor-validated):** Foundation (roles+factory) → A budget → C model-per-call → E advisor → B resume → D worktree P5b

## Foundation — model roles + provider factory (C and E share it)

One provider-construction seam, consumed by both new features:

- **Core** (`flow/ctx.rs`): `pub type ProviderFactory = dyn Fn(&str) -> Result<Arc<BoxedProvider>, Error> + Send + Sync;`
  `WorkflowCtxBuilder::provider_factory(Arc<ProviderFactory>)` + `CtxInner.provider_factory: Option<Arc<ProviderFactory>>` (shared into `nested()`).
  The `&str` argument is a ROLE NAME (`"fast"`, `"frontier"`) or a raw model id — the host's factory decides how to resolve it. Core stays string-keyed; no enum.
- **TUI** (`config.rs` + `main.rs`): `TuiConfig.fast_model: Option<String>`, `frontier_model: Option<String>`.
  Factory closure: `"main"`/unknown-empty → the session provider; `"fast"` → `fast_model` (fallback: main model); `"frontier"` → `frontier_model` (fallback: main model); any other string → treated as a model id. Built via the existing `build_provider` (same key, retry, caching).

## A — budget through `run_workflow`

- Input schema gains optional `"budget"` (integer ≥ 1, cost-weighted token-equivalents — the P2 unit).
- `execute()`: when present → `builder.budget(n)`.
- After the recipe returns, check `ctx.control_breach()`: a `Budget` breach makes the
  tool output an ERROR naming the limit (the entry agent reports honestly), even if
  the recipe collapsed agent failures to a degraded string.
- The tool description documents the arg so the entry agent can honor "use 10k tokens".

## C — per-call model (dynamic by task)

- `AgentOpts.model: Option<String>` + `AgentCall::model(impl Into<String>)`.
- `run_one`: `opts.model` + ctx factory present → leaf runs on `factory(model)?` provider;
  factory ABSENT → fall back to `ctx.provider()` and emit `WorkflowEvent::LogLine`
  ("model 'fast' requested but no provider factory — using the default") — degraded,
  never fatal (a recipe must not die because a host lacks the seam).
- Demonstrated use (capability + one consumer, NOT a learned router): `deep_research`
  runs its plan stage on `"fast"`; angle/verify/synthesize stay on the default.
- `run_workflow`'s ctx gets the factory threaded from the TUI.

## E — advisor mode (frontier reviewer, my own operating contract)

- **Transcript seam** (the crux): `ExecutionContext.transcript: Option<Arc<Vec<Message>>>`
  (`#[serde(skip)]`-free — EC isn't serialized; plain optional field, `None` default).
  The runner populates it at tool-dispatch time with a snapshot of `self.messages`.
- **`AdvisorTool`** (core, `tool/advisor.rs`): holds the frontier `Arc<BoxedProvider>`.
  - `definition()`: name `advisor`, NO parameters; description encodes the calling
    contract (before substantive work · when you believe the task is complete ·
    when stuck or changing approach; the full conversation is forwarded automatically).
  - `execute()`: render `ctx.transcript` to text (roles, text blocks, tool calls
    condensed with truncated outputs), send to the frontier provider under an
    advisor system prompt distilled from my own: independent skeptical reviewer,
    verify claims against evidence in the transcript, name what BLOCKS vs what
    doesn't, give the discriminating check rather than a verdict, no flattery.
  - No transcript in ctx → honest error ("advisor needs a runner that forwards the transcript").
- **TUI**: build `AdvisorTool` with `factory("frontier")`, add to the entry agent's tools.

## B — resume of an interrupted `run_workflow`

- Behavior (the tested contract, not just plumbing): re-invoking `run_workflow` with the
  SAME recipe+args replays completed agent calls at zero cost and continues the rest —
  the blog's "pick up where it left off", content-addressed (P4 semantics).
- `RunWorkflowTool::with_journal_dir(PathBuf)`: `execute()` derives
  `wf-{recipe}-{sha256(canonical args)[..12]}.jsonl` under the dir and opens it
  `ResumeMode::Resume` (replays when the file exists, fresh otherwise).
- **TUI**: journal dir = `<sessions>/journals/<session-id>/` — session-scoped: re-asking
  in the SAME session resumes; a new session starts fresh (no stale cross-session replay).
- End-to-end test: run a recipe against a counting mock → N provider calls; re-execute the
  same input → ZERO new provider calls, identical output.

## D — worktree isolation (P5b, un-parked by the user)

Per the parked design (memory `dynamic_workflows_design.md`):
- `git2` workspace dep. `Isolation { None, Worktree }` + `AgentOpts.isolation` +
  `AgentCall::isolation()`.
- `WorktreeGuard::create` in `tokio::task::spawn_blocking` (git2 handles are !Send —
  only PathBufs cross `.await`); deterministic collision-tolerant names
  (`run-{run_id}-{label}-{index}`); `cleanup`: prune if clean, keep + branch if dirty
  (`StatusOptions::include_untracked(true)`); sync best-effort `Drop` backstop;
  startup sweep utility for orphaned worktrees.
- Leaf: `Worktree` → create guard → `builder.workspace(guard.path())`.
- **Journal gate (B↔D interaction)**: the journal REFUSES to journal/replay calls with
  `isolation != None` (side effects are not restored by replay) — enforced in the leaf,
  tested.
- Fixture: tests create a temp git repo with an initial commit.

## Testing & validation bar

Per feature: TDD red→green, workspace gate, commit. Campaign close: live TUI pty —
(1) budget arg honored on a real run_workflow call; (2) deep_research plan stage on the
fast model (trace shows the model per llm_response); (3) advisor tool callable by the
entry agent and returning frontier advice; (4) interrupted-then-reasked workflow
replays (trace shows zero new angle LLM calls on the resumed part).

## Non-goals

Learned task→model routing; cross-session journal replay; advisor auto-invocation
hooks; Restate durable parity (P7's other half); model roles beyond the three names.
