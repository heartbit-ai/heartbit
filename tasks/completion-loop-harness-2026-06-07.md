# Completion-Loop Harness — Design Proposal

Date: 2026-06-07 · Crate: `heartbit-core` + `heartbit-tui` · Branch context: `feat/tui-streaming-markdown`

## User requirement (verbatim, FR)

> "un harness qui identifie les loops à effectuer pour compléter tout le travail demandé. il faut faire
> attention à la dérive. si l'utilisateur demande de créer une feature alors on doit créer les loops pour
> arriver à la finalité de ce qui a été demandé. Si le modèle identifie un manque dans la demande il peut
> faire une suggestion à l'utilisateur en utilisant l'équivalent du ask tool."

Plain English: a harness that (a) derives the loops needed to fully complete the requested work, (b) guards
against drift, (c) drives those loops to the finality of the request, and (d) when it finds a gap in the
request, surfaces a suggestion to the user via the equivalent of an ask tool.

---

## 1. Executive summary

The **execute → verify → judge-gated** *back half* of this harness already runs in `heartbit-core` today:
deterministic verify-replan (`runner.rs:1480`), the independent-judge goal-continuation loop
(`runner.rs:1508`, `goal.rs`), todo recitation at the context tail (`runner.rs:768`, `todo.rs:149`),
doom-loop detection (`doom_loop.rs`), and the hard budget backstops (`max_continuations=8`,
`MAX_VERIFY_REPLANS=8`, `max_turns`). The research corpus *validates* this design — independent judge over
evidence, deterministic stuck-detection, recitation — these are exactly the SOTA anti-drift levers.

What is genuinely missing is the **front half** and the **ask channel**:

1. **No intake → acceptance-criteria → gap-detection stage.** `GoalCondition` takes a single free-text
   objective; nothing turns the user's request into checkable criteria up front, and nothing elicits
   "what is missing."
2. **The ask tool is dead end-to-end.** The `question` builtin + `OnQuestion` callback are fully plumbed in
   core but **no TUI/CLI/daemon caller ever sets `.on_question(...)`** (only an internal forward at
   `orchestrator.rs:2809`). The TUI cannot pause and ask the user a structured question today.
3. **The entry agent has no goal wired.** `grep '\.goal(\|GoalCondition' heartbit-tui/src` = 0 hits;
   `OrchestratorBuilder` exposes no goal-forwarding to the entry agent. So judge-gated completion is **BUILD,
   not EXISTS** for the real (TUI) path.
4. **No clarify rule, no scope guard.** Nothing in `build_entry_agent_prompt` (`orchestrator.rs:1631`) or
   `instructions.rs` tells the agent to ask when underspecified; no guard bounds the *direction* of work.

The build is therefore small and additive — no new `CompletionLoopHarness` subsystem. We add a front-half
flow (criteria + gap-elicitation), wire the existing ask tool into the TUI, forward a goal to the entry
agent, add a per-loop `acceptance` field to `TodoItem`, and add a deterministic scope guard.

---

## 2. Research-backed mechanism table

| Mechanism | Who uses it | Maps to heartbit |
|---|---|---|
| **Done decided by the HARNESS, not the model** (enum state-machine plan, exit-code verify, separate judge) | Claude Code TodoWrite, Codex update_plan, Aider, Copilot CI, Jules | EXISTS: `GoalCondition` independent judge (`goal.rs`) + `VerifyCommandTool` exit-code sentinel (`verify.rs`) + verify-replan gate (`runner.rs:1480`). |
| **Acceptance = observable behavior, written UP FRONT** (failing-test-first; criteria before coding) | Codex ExecPlan, Factory test-writer, Devin test plan, CrUISE-AC | BUILD: a pre-run criteria-extraction leaf that produces the objective string for `GoalCondition`; ground the judge on verify evidence. |
| **External-signal-gated continuation** — intrinsic self-correction plateaus/degrades without a verifier (Huang et al. ICLR'24) | Reflexion, Self-Refine | EXISTS+refine: continuation loop re-injects judge reason (`goal.rs:116`); refine to carry the concrete `VERIFY_RESULT` when present. |
| **Structured multiple-choice clarification** (1–4 Q × 2–4 options, batched, blocking; ask only when guessing causes rework) | Claude Code AskUserQuestion, Cursor plan-mode, Factory | EXISTS (core) / DEAD (TUI): `question` tool + `OnQuestion` (`question.rs`) already has options+descriptions+single/multi-select. BUILD: wire `.on_question(...)` in the TUI + an intake rule. |
| **Completeness critic — "what is MISSING?" ≠ "is it DONE?"** (80–82% of generated criteria add relevant reqs) | CrUISE-AC (single-sourced, see §6.5) | BUILD: a distinct gap-elicitation prompt (one call) whose output becomes question payloads and/or new todos. |
| **Plan-pinning via tail recitation** — re-WRITE goal at context tail each turn (recency beats lost-in-the-middle) | Manus, Anthropic context-engineering | EXISTS: `recite_open_todos` appended to the last message every turn (`runner.rs:768`). Refine: also pin the verbatim criteria. |
| **Strong exclusive goal anchor** — drift is in-context pattern-matching, not token distance (AIES'25) | Manus, goal-drift study | BUILD: feed extracted criteria as the `GoalCondition` objective AND pin them in the system prompt for the whole run. |
| **Deterministic stuck / loop detection with hard thresholds** | OpenHands StuckDetector, SWE-agent ACI | EXISTS: `DoomLoopTracker` (exact + fuzzy) (`doom_loop.rs`); threshold transplant is optional refinement. |
| **Scope guard — pre-tool boundary check** (path allowlist; unrelated PR changes 35%→4%, single-sourced §6.5) | Claude Code PreToolUse hooks, agent-guardrails | BUILD: a `pre_tool` guardrail seeded from `FileTracker` paths; Deny out-of-scope mutation → scope-expansion question. Cheap, no LLM. |
| **Per-action plan-vs-action alignment audit** (LlamaFirewall AlignmentCheck) | Meta LlamaFirewall | OPTIONAL (cost): `pre_tool` LLM guardrail "does this serve an open todo?"; opt-in only (multiplies judge cost by tool-call count — see §6.1). |
| **Two distinct user gates: CLARIFY vs APPROVE, never conflated** | Claude Code (ExitPlanMode ≠ AskUserQuestion) | EXISTS: `OnApproval` (yes/no tool gate) and `OnQuestion` (choose-options) are already separate primitives. Keep them separate. |
| **Expected-outcome pre-commitment** ("annotate before acting → lies less") | Devin (single-vendor, §6.8) | DEFER: later P-step, not core. |
| **Resource/circuit-breaker backstop beneath semantic guards** | OpenHands, all coding agents | EXISTS: `max_turns`, `TokenUsage`, continuation/replan caps; refine to per-leaf budgets if needed. |

---

## 3. Proposed architecture

The ask gate and the goal **live at the ENTRY agent**, never at an isolated/worktree leaf — `OnQuestion`
owns the user channel and clarification does not propagate to sub-agents (research set 4; flat hierarchy).

**When the front half engages (self-gating, not always-on).** The user's requirement is conditional —
*"si l'utilisateur demande de créer une feature alors on doit créer les loops."* The front-half
(criteria-extraction + gap-detection + ask) engages on the **same routing branch that
`build_entry_agent_prompt` (`orchestrator.rs:1631`) already uses to decide "delegate / do-it-yourself" vs
"answer directly"**: substantial/feature requests opt into it (a `run_workflow` recipe the agent invokes,
exactly how `deep_research` is gated today); trivial/chat turns bypass it entirely and pay nothing — the goal
is simply not set and no todos are emitted, mirroring the existing self-gating of recitation. It is NOT
always-on middleware.

```
user request
   │
   ▼
[1] INTAKE + ACCEPTANCE-CRITERIA EXTRACTION        (new flow leaf, 1 LLM call; entry agent)
      request → bulleted, observable acceptance criteria ("a human/command can verify X")
      → becomes the GoalCondition objective string + pinned in system prompt
   │
   ▼
[2] GAP DETECTION (completeness critic)            (new flow leaf, 1 LLM call)
      inverse prompt: "what does the request imply but leave unspecified? (scope/risk/intent)"
      classify gaps: high-guess-rate (format/output) → proceed-with-stated-assumption;
                     low-guess-rate (conditional/edge/intent) → ask
   │
   ├── gaps that are load-bearing ──▶ [3] STRUCTURED ASK-USER GATE  (REUSE question builtin + OnQuestion)
   │                                       1–4 questions × 2–4 options, batched, blocking.
   │                                       answers fold back into criteria/assumptions.
   │   (no load-bearing gaps)
   ▼
[4] LOOP PLAN  (todos WITH done-conditions)        (BUILD: TodoItem.acceptance)
      one todo per milestone; each carries an observable acceptance condition; exactly-one in_progress.
   │
   ▼
[5] EXECUTE with VERIFY per loop                   (EXISTS: VerifyCommandTool + verify-replan gate)
      after edits, run verify; RED → deterministic replan (runner.rs:1480) before any completion.
   │
   ▼
[6] JUDGE-GATED COMPLETION                          (EXISTS core / BUILD wiring: GoalCondition on entry agent)
      independent judge over the 12K-char evidence tail; per-criterion verdict; not-met → continuation
      (bounded by max_continuations); advisor mode available as a stronger second reviewer at decision points.
   │
   ▼
[7] ANTI-DRIFT (continuous, every turn)
      recitation (todos + verbatim criteria at context tail) · criteria pinned in system prompt ·
      deterministic scope guard (path allowlist) · doom-loop detector · hard budget caps.
   │
   ▼
done = every acceptance criterion met (judge) AND no open todos AND verify green.
```

**Multi-turn policy (default, not just an open question).** The TUI is a multi-turn REPL, but every
mechanism in the corpus assumes one fixed objective. Default: the pinned criteria/todo-ledger are
authoritative **for the current request**; a *new user turn* may legitimately revise or replace them
(re-scoping is allowed and is NOT drift). Drift = the agent diverging from the current criteria **absent a
user turn**. This makes the scope guard definable and resolves the goal-anchor-vs-goal-switching tension
(set 5 caveat: strong anchoring hurts when goal-switching is legitimate — so anchor per-request, re-derive
on a new turn).

**The ask channel meets the user's "suggestion" need** via options-with-descriptions; the `question` tool
has no free-text answer field (labels only). An "Other → free text" escape is an OPTIONAL schema extension,
flagged open (§6), not P1 — the spec says REUSE.

---

## 3b. Handoff — native session-bridge ability (user-requested; research-verified 2026-06-07)

**What it is.** Distill the CURRENT session into a *purpose-tailored* Markdown brief that seeds a
DIFFERENT session — vs compaction (same lineage, generic summary) and resume (same lineage, full raw
transcript + stale environment). Canonical source: `mattpocock/skills` `/handoff`
(skills/productivity/handoff) + aihero.dev write-up. Notably, obra/superpowers (220k stars) has NO handoff
skill — it's an open feature request (#931): this is a real gap, not a commodity.

**The three distinctions (research-verified):**

| vs | Difference |
|---|---|
| Compaction | same thread, mechanical summary, "clobbers" progress → handoff: new lineage, purpose-curated, both sessions stay pure |
| Sub-agent delegation | programmatic, ephemeral, auto-returns in-run → handoff: human-carried, editable, deferred, cross-harness ("DIY sub-agent") |
| `/resume` restore | replays the FULL transcript (bloat + stale env inherited) → handoff: curated fresh seed, no stale environment |

**Why a short brief beats a long transcript (empirical):** NoLiMa (2502.05167): 10/12 models below 50% of
their short-context baseline at 32K tokens on latent-association retrieval; Chroma context-rot (18 models):
degradation appears early and persists — and semantically-similar-but-irrelevant content degrades MORE than
length alone (exactly what a stale transcript is).

**Design rules, ranked (from the canonical skill):** (1) the PURPOSE argument is mandatory and the doc is
tailored to it — without it a handoff is a worse compaction; (2) pointers to artifacts (paths/issues/diffs),
never duplication — brevity IS the mechanism; (3) redact secrets (a brief on disk is a leak surface);
(4) disposable location outside the workspace (no doc rot); (5) a "suggested recipes/skills" section primes
the next session.

**Canonical brief structure:** Purpose (the WHY) → Goal → State/Progress → What worked / what didn't →
Pointers to artifacts → Suggested recipes/skills → Next steps.

**Native fit in heartbit (not a skill — the seam exists):** `ExecutionContext.transcript` already feeds the
`advisor` tool the full conversation; a `handoff` builtin is the same seam with a different distillation:

- `handoff { purpose }` → one LLM call (handoff-doc system prompt embodying the rules above) over the
  transcript → writes `~/.config/heartbit/handoffs/<date>-<slug>.md` (outside the workspace; pruned by age;
  configurable) → returns the path + 3-line summary into the session.
- TUI: `/handoff <purpose>` (slash command → the tool), and a brief picker (alongside `/resume`) that seeds
  a NEW session with a chosen brief as the opening user message. Symmetric by construction: a child
  prototype session can `handoff` its learnings back, and the parent seeds from that brief (the
  bidirectional parent→child→parent loop).
- **Completion-harness integration (why this section is here):** the scope guard's Deny gains a third
  disposition — *expand scope / ask the user / **handoff*** — and the gap-elicitation leaf can propose
  handoff for out-of-scope discoveries. Explicit descoping sharpens the current loop plan (the out-of-scope
  todo is closed with a pointer to the brief).
- **Emergency handoff (incident-driven):** on a terminal `RunFailed` (the 2026-06-07 402 killed a session
  mid-work and the respawned agent was amnesiac), the TUI writes a best-effort brief from the saved session
  state so the next session can continue deliberately. Optional, last P-step.

**Failure modes to design against:** briefs that duplicate artifacts (pointer rule in the distillation
prompt); missing the WHY (purpose is a required arg); stale briefs (age-pruned dir); secrets (redaction
instruction + the existing protected-paths sensibility).

---

## 4. What EXISTS vs what to BUILD (smallest-change bias)

### EXISTS — reuse as-is
- Independent judge + continuation loop: `goal.rs` (`GoalCondition`, `evaluate`, `continuation_message`);
  loop at `runner.rs:1508-1534`. `AgentRunnerBuilder::goal()` at `builder.rs:522`.
- Deterministic verify-replan gate: `runner.rs:1480-1499`; `VerifyCommandTool` + `parse_latest_verify`
  (`codegen/verify.rs`).
- Todo store + tail recitation: `todo.rs` (`TodoStore`, `recite_open_todos:149`); injected at
  `runner.rs:768-774`. TUI wires `todo_store` via `entry_context` (`main.rs:733`).
- Ask tool primitive: `question.rs` (`QuestionTool`, `OnQuestion`, options + `multiple`); registered when
  `on_question` is Some (`builtins/mod.rs:523`, `builder.rs:725`).
- Anti-drift family: `doom_loop.rs`, `DelegationNudge` (`runner.rs:124`), `max_turns` / budgets.
- Flow combinators for the front-half pipeline: `flow/pipeline.rs`, `flow/agent.rs`, `flow/ctx.rs`;
  `run_workflow` registry (`workflow_tool.rs`); `advisor` reviewer.
- Instruction injection: `instructions.rs`; TUI passes assembled instructions via `instruction_text`
  (`main.rs:740`), already including the "VERIFY_RESULT is source of truth" nudge.

### BUILD — keep this list short (no premature abstraction)

1. **`acceptance: Option<String>` on `TodoItem`** — `todo.rs:16-23`. The per-loop done-condition the spec
   demands. `#[serde(default)]` for back-compat; surface it in `recite_open_todos` (`todo.rs:149`) and in the
   `todowrite` schema (`todo.rs:192`). *No* new status enum — proceed-with-the-existing 6 states.

2. **Forward a goal to the entry agent** — `OrchestratorBuilder` (around `orchestrator.rs:2327`, mirroring
   `on_question` at :2808) needs an `entry_goal(GoalCondition)` (or `entry_context.goal`) that reaches the
   entry `AgentRunnerBuilder::goal()`. Today there is no path; `builder.rs:522` exists but is unreachable
   from `Orchestrator::builder().entry_agent()`.

3. **TUI wires `.on_question(...)`** — `heartbit-tui/src/main.rs:729-752`. Bridge `OnQuestion` (async-returning)
   to a modal paralleling the existing `on_approval` modal (`main.rs:585`), reusing the sync-over-blocking-
   `std::mpsc` pattern; render options as a selectable list, return selected labels. This is the user's
   explicit "ask tool" requirement, dead today.

4. **Intake/clarify rule** — `build_entry_agent_prompt` (`orchestrator.rs:1631`) + an instructions snippet.
   Add the trigger verbatim: *"Ask the user via the `question` tool ONLY when the next step depends on user
   intent and guessing would cause rework (scope / risk / intent). Otherwise proceed and state your
   assumptions as todos."* No clarify rule exists anywhere today (§7 of inventory).

5. **Front-half flow leaves** — a small recipe (criteria-extraction → gap-elicitation) built on
   `flow/pipeline.rs`, registered like `deep_research` in `default_registry()` (`workflow_tool.rs:359`).
   Output: an acceptance-criteria string (→ entry goal + pinned) and a gap list (→ question payloads / todos).

6. **Deterministic scope guard** — a `pre_tool` `Guardrail` seeded with a path allowlist from `FileTracker`;
   Deny out-of-scope mutation with feedback, surfacing a scope-expansion question. No LLM. (The per-action
   *LLM* alignment audit is OPTIONAL and opt-in — §6.1.)

7. **`handoff` builtin tool** (§3b) — transcript via `ExecutionContext.transcript` (the advisor seam,
   `tool/advisor.rs`), one LLM call with the handoff-doc system prompt (purpose-tailored, pointers-not-
   duplication, redaction), write to `~/.config/heartbit/handoffs/`, return path + summary.

8. **TUI `/handoff` + brief seeding** — slash command (house pattern: bare → usage notice, arg = purpose);
   a brief picker (mirroring `/resume`'s `SessionPicker`) that opens a NEW session seeded with the chosen
   brief as the first user message.

---

## 5. P1..Pn TDD roadmap (failing-test-first; each step names its test)

**P1 — `TodoItem.acceptance` done-condition.**
Failing test `todo_item_carries_acceptance_condition` (`todo.rs` `#[cfg(test)]`): deserialize a `todowrite`
payload with `"acceptance": "tests pass: cargo test green"`, assert the field round-trips and that
`recite_open_todos` includes the acceptance line for an open item. Then add the field + recitation render.

**P2 — Entry-agent goal forwarding.**
Failing test `entry_agent_goal_is_wired` (`orchestrator.rs` tests): build an orchestrator via
`Orchestrator::builder(p).entry_agent(tools).entry_goal(GoalCondition::new("obj", judge))`, run against a
stub provider whose first reply is a bare "done" and assert the judge fired (continuation injected) — i.e.
the entry runner received a goal. Then add `entry_goal`/`entry_context.goal` plumbing to the entry builder.

**P3 — Criteria-extraction leaf.**
Failing test `extracts_observable_acceptance_criteria` (`flow`/recipe tests): given a request "add a
`/health` endpoint", a stubbed provider returns 3 bullets; assert the recipe yields a non-empty
criteria string phrased as observable behavior and that it is passed as the `GoalCondition` objective. Then
implement the leaf via `flow/pipeline.rs`.

**P4 — Gap-elicitation (completeness critic) leaf.**
Failing test `elicits_missing_low_guess_requirements`: stubbed provider returns one missing item
("auth required?"); assert it is emitted as a question payload (1–4 Q, ≥2 options) and high-guess items are
NOT asked (surfaced as assumption todos instead). Then implement the inverse-prompt leaf.

**P5 — TUI `on_question` modal (pty).**
Failing pty test `tui_renders_question_modal_and_returns_label` (`heartbit-tui` pty harness): drive a run
where the agent calls `question`; assert on the **settled final frame, space-insensitive** (de-ANSI
collapse) that the options render and a selection returns the chosen label to the agent. Then wire
`.on_question(...)` + the modal reducer. (Pty-harness lesson: assert the settled frame, never an in-loop poll.)

**P6 — Intake clarify rule.**
Failing test `entry_prompt_has_clarify_rule` (`orchestrator.rs` tests): assert `build_entry_agent_prompt(...)`
output contains the "ask only when guessing causes rework (scope/risk/intent)" trigger. Then add the snippet.

**P7 — Deterministic scope guard.**
Failing test `out_of_scope_mutation_is_denied` (guardrail tests): a guardrail seeded with allowlist
`["src/a.rs"]`; a `pre_tool` for an edit on `src/b.rs` returns `Deny` with a scope-expansion message; an edit
on `src/a.rs` returns `Allow`. Then implement the path-allowlist guard.

**P8 (optional, opt-in) — Per-criterion judge verdict.**
Failing test `judge_reports_per_criterion_status`: judge prompt fed a 3-criterion objective returns a
checklist; assert not-met lists the *specific* failing criterion in the continuation message. Opt-in; off by
default for cost.

**P9 (optional, defer) — Expected-outcome pre-commitment** (Devin, single-vendor §6.8). Failing test
`verify_records_predicted_outcome`. Defer until the front half is live and measured.

**P10 — `handoff` builtin.**
Failing test `handoff_writes_purpose_tailored_brief` (builtins tests): stub provider returns a brief; tool
called with `{purpose: "prototype the picker UI"}` against an `ExecutionContext` carrying a transcript →
asserts a file lands under the handoffs dir, contains the purpose, contains NO raw secret planted in the
transcript (redaction prompt is exercised by a content assertion), and the tool returns the path. Then
implement (mirror `advisor.rs`).

**P11 — TUI `/handoff` + brief picker seeding.**
Failing app-reducer test `slash_handoff_requires_purpose_and_dispatches` + pty test
`seeded_session_opens_with_brief` (settled-frame assert). Then wire the command + picker.

**P12 (optional) — Emergency handoff on terminal RunFailed.**
Failing test `terminal_failure_leaves_a_brief`: simulate a terminal run error in the TUI event loop →
assert a best-effort brief exists referencing the session id. Then implement (pure TUI-side, no core change).

**Gate after every P-step:** `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm
--all-targets -- -D warnings && cargo test --workspace --exclude mini-crm`.

---

## 6. Open questions for the user (critic gaps — addressed or marked open)

**6.1 Cost/latency of the layered guard stack (highest-value blind spot).** Each extra LLM guard
(per-action alignment audit, per-criterion judge, completeness critic, criteria-extraction, N-sample
ambiguity) adds cost; the per-action alignment audit multiplies the 12K-tail judge call by the tool-call
count. *Recommendation:* deterministic guards first (scope guard, verify-replan), make any per-tool LLM judge
**opt-in**, lean on existing caps (`max_continuations=8`, `MAX_VERIFY_REPLANS=8`, `max_turns`). **Open:** what
per-task token/latency budget is acceptable in interactive TUI vs daemon/batch?

**6.2 Multi-request / cross-turn session state.** Default proposed in §3 (pinned criteria authoritative
per-request; a new user turn may revise/replace; drift = divergence absent a turn). **Open:** carry-over policy
for "remaining" todos + partial criteria across turns — keep, re-derive, or merge? The journal is one-per-run
(`journal.rs`) while the TUI is multi-turn.

**6.3 Interaction with compaction/summarization.** The judge reads a 12K-char *tail* (`goal.rs:46`) while the
runner compacts at a threshold fraction; compaction can evict the very tool-output evidence the judge needs,
or the verbatim objective the anti-drift literature says must be re-asserted unparaphrased. *Recommendation:*
pin the verbatim criteria + latest verify evidence OUTSIDE the compactible window (recitation already
re-emits from the store, not the summary). **Open:** run judge before compaction, or mark post-compaction
summaries inadmissible as evidence?

**6.4 User abandons mid-loop.** Continuation loop, judge gate, and "drive to returned result" all assume the
user waits. `OnQuestion` can stay pending indefinitely. **Open:** disposition on abandonment —
checkpoint-and-suspend (Replit), auto-approve-and-proceed (Jules), or abort-and-discard? Interacts with
worktree/journal state so a resumed run isn't left half-applied. The seam exists (interrupt handle +
`on_input`) but no abandon semantics are defined.

**6.5 Thin / single-source load-bearing stats (verify before relying).** Treat as *deferred / single-sourced*:
"69.4% resolve / 30% fewer queries / 3.06 q/task" (arXiv 2603.26233); "80–82% criteria add relevant reqs"
(CrUISE-AC, arXiv 2501.15181); embedding-drift thresholds (arXiv 2601.12359); "unrelated PR changes 35%→4%"
(agent-guardrails repo + a blog). These motivate direction but are NOT hard design constraints.

**6.6 Non-implementable formulas.** Value-of-Information (`ask IF (1-confidence)×failure_cost >
interruption_cost`) hand-waves `confidence`; embedding-cosine drift `tau` is unspecified. Use VOI as the
*prompt framing* (reversibility → failure_cost; route irreversible to `OnApproval`), not a literal
computation. Drift detection (P-future): start with the cheap self-report variant (ask the agent to restate
its goal, judge vs pinned criteria) before any embedding `tau`.

**6.7 Self-reflection without a verifier (cross-angle contradiction).** Huang et al. (ICLR'24): intrinsic
self-correction without an external signal plateaus/degrades. *Resolution:* the no-verifier continuation
fallback should re-inject **localized, actionable** judge feedback (Self-Refine), not bare "keep working,"
and should not over-rely on reflection alone. When a verify command exists, it is the external signal.

**6.8 Expected-outcome pre-commitment.** Single vendor source (cognition.ai); no independent replication and
no spec for predicted-vs-actual scoring. **Deferred to P9** behind the verify tool.

---

## 7. Honest trade-offs

- **Judge cost.** Every natural-completion attempt fires a judge over a 12K-char tail; per-criterion and
  per-action variants multiply this. Mitigation: deterministic guards first, judge as the gate not the
  monitor, opt-in for anything per-tool-call.
- **Question fatigue.** Asking too much is a failure mode (ClarEval penalizes turns-to-clarify). Mitigation:
  batch 1–4 Q × 2–4 options in ONE `OnQuestion` round; trigger only on low-guess-rate intent gaps; proceed
  with stated assumptions otherwise.
- **Bounded budgets are dumb backstops.** `max_continuations`, `MAX_VERIFY_REPLANS`, `max_turns` cannot be
  reasoned around — a genuinely-hard task can hit the cap and report not-done rather than loop forever. That
  is the intended safe direction (report, don't spin), but it means "done" can be a bounded best-effort.
- **Scope guard false positives.** A path allowlist can block legitimate cross-file work; it Denies with a
  scope-expansion question rather than hard-failing, trading one interruption for prevented drift.

---

## 8. Sources (load-bearing)

Claude Code TodoWrite / ExitPlanMode / AskUserQuestion system prompts (Piebald-AI mirror; code.claude.com
agent-sdk user-input). Codex `update_plan` / ExecPlans (openai/codex `plan_tool.rs`,
developers.openai.com codex_exec_plans). Aider lint/test auto-fix. Cursor plan-mode. GitHub Copilot coding
agent. Manus context-engineering (recitation). OpenHands StuckDetector (docs.openhands.dev). SWE-agent ACI
(arXiv 2405.15793). Huang et al. LLMs-Cannot-Self-Correct (arXiv 2310.01798). Reflexion (2303.11366).
Self-Refine (2303.17651). NoLiMa (2502.05167). Zheng et al. LLM-as-Judge (2306.05685). CrUISE-AC
(2501.15181, single-sourced). ReWOO (2305.18323). Ask-or-Assume (2603.26233, single-sourced). ClarifyGPT
(2310.10996). VOI (2601.06407). Underspecification (2505.13360). ClarEval (2603.00187). Goal-drift AIES'25
(2505.02709). LlamaFirewall AlignmentCheck (2505.03574). Agentic Rubrics (2601.04171). Already cited in
`goal.rs`: An Illusion of Progress (2504.01382), Self-Preference Bias (2410.21819).

Handoff (§3b): mattpocock/skills `/handoff` (github.com/mattpocock/skills, skills/productivity/handoff) ·
aihero.dev/skills-handoff · ykdojo/claude-code-tips handoff variant (Goal/Progress/Worked/Didn't/Next
template) · obra/superpowers issue #931 (gap confirmation) · code.claude.com/docs/en/sessions (resume =
full-history restore) · Chroma context-rot (trychroma.com/research/context-rot, 18 models, no U-shape
claim) · NoLiMa (2502.05167).
