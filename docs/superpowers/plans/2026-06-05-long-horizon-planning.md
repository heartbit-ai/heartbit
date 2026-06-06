# Long-Horizon Planning (point 3) — Plan

> TDD, red→green→commit. From the planning deep-research (`wf_dde0e435`).

**Core gap:** heartbit's loop is pure ReAct — no persisted multi-step plan, and the `TodoStore` is agent-elected and never re-surfaced into context. Compaction re-derives "TODOS" textually from the conversation (lossy, decoupled from the actual store).

**SOTA design (do NOT add a new planner stage — make the existing loop plan-aware):**
- **Recitation** (Manus): re-surface the *actual* `TodoStore` open items at the **context tail every turn**, pushing the plan into recent attention (beats lost-in-the-middle). Zero extra LLM calls.
- **Compaction reads the store**: after a summary, the plan is reconstructed from the store, not the summary — one mechanism kills both drift *and* "plan must survive compaction" (composes with the #3 preservation schema just shipped).
- **Replan on out-of-plan** (AdaPlanner/Plan-and-Act): a *verify-fail* (or a repeated-identical-failure via `DoomLoopTracker`) is the out-of-plan signal → flip the relevant todo + force an off-cycle `GoalCondition` continuation. Replans are **gated on out-of-plan signals only**, never per-step (avoids the "always-plan degrades performance" thrash).
- **When NOT to plan:** self-gating — recite only when there ARE open todos (trivial/chat tasks never create todos → zero overhead). (The research's "use classify_query" step is moot: that classifier was deleted in point 4.)

**Reuses:** `Arc<TodoStore>` (tool/builtins/mod.rs:177), `GoalCondition` (goal.rs), the goal-continuation loop (runner.rs), the new compaction path.

**Risk avoided by construction:** judge-per-step thrash — recitation is a cheap string at the tail (no LLM); judging stays at boundaries; replans gated.

---

## Task 1: runner holds the shared `Arc<TodoStore>`

Mirror the `context_recall_store` plumbing. `AgentRunner` + `AgentRunnerBuilder` get `todo_store: Option<Arc<TodoStore>>`; setter `.todo_store(Arc<TodoStore>)`; constructor default `None`; build() transfer. (`TodoStore` re-exported; `TodoStatus` has Pending/InProgress to define "open".) Mechanical; compile-only.

## Task 2: recite open todos at the request tail each turn (self-gated)

In the per-turn request assembly (runner.rs ~612, where the pruned `request` is built), when `todo_store` is `Some` AND it has open items (Pending/InProgress), append a recitation block to the **tail** — to the content of the LAST message in `request.messages` (NOT the `system` prompt — that would bust the prompt cache, and not a new message — that would break role alternation). Format a compact `[plan: open items]\n- [ ] …` block.

- **Helper (pure, tested):** `fn recite_open_todos(rows: &[TodoRow-like]) -> Option<String>` → `None` when no open items; else a compact bullet block. Test: empty → None; mixed → only open items, in order.
- **Wiring:** after pruning, if the block is `Some`, append it as a trailing `ContentBlock::Text` on the last message. Test (runner-level, MockProvider capturing requests): after a `todowrite` with open items, the next request's last message contains the plan block at the tail; with no open todos, it does not.

## Task 3: compaction re-surfaces `TodoStore` state

When the proactive/reactive compaction fires (`inject_summary`), ensure the post-compaction context carries the *current* `TodoStore` open items (so the plan survives a summary from the store, not the lossy text). Simplest: the recitation in Task 2 already re-injects every turn from the store — so a post-compaction turn re-recites automatically. **Verify** this composes: a forced compaction followed by a turn still shows the plan block (sourced from the store). Test: trigger compaction, assert the next request still has the plan block. (If recitation already covers it, this task is a test + a doc note — no new code.)

> **STATUS 2026-06-05 — IMPLEMENTED.** Tasks 1-3: `TodoStore::open_items()` + pure `recite_open_todos()`; `AgentRunner`/`Builder.todo_store()`; per-turn recitation at the message tail; survives compaction by construction (store-sourced, decoupled from ctx); wired in TUI single-agent path. Commit fcda792. Task 4: `replan_on_verify_fail` bounded gate (below) — commit pending. All TDD-tested; workspace gate green.

## Task 4 (follow-on): replan trigger on out-of-plan — DONE

A `verify` tool result of `VERIFY_RESULT: FAIL` (or `DoomLoopTracker` firing) is the out-of-plan signal. On it: (a) leave a continuation guidance ("verification failed — revise the plan/todos and fix before finishing") so the agent doesn't natural-complete on red. The cleanest seam: when a `GoalCondition` is present, a verify-fail in the transcript already keeps it "not met"; for the no-goal interactive path, add a lightweight check that a `VERIFY_RESULT: FAIL` since the last `PASS` blocks "done" with a re-injected nudge. Scope/observe before building — this is the most behavioral piece; gate it behind opt-in and a test that a failed verify forces a continuation.

---

## Out of scope
DAGs / goal-stacks (fragment context — rejected by research); a separate planner model (premature, plan/executor drift); multi-agent planning.
