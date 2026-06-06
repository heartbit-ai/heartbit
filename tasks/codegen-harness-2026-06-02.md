# Code-Generation Harness (`codegen/`) — design + plan (2026-06-02)

Goal: a heartbit-core harness dedicated to **code generation**, inspired by `claw-code`,
**without duplicating** what we already have (and keeping our better parts).

## Evidence (5-agent investigation workflow wf_1fc8c778-8a7)

- **claw-code** is a thin Claude-Code ReAct clone. Its differentiated ideas
  (`green_contract.rs` tiered evidence gate with `TestCommandProvenance{command,exit_code,passed}`,
  `recovery_recipes.rs`, `lane_events.rs`) are **defined but provably unwired** (zero non-test
  consumers). Borrow the *idea* (exit-code evidence gate), not the integration.
- **heartbit already has, and better:** `GoalCondition` independent judge over transcript
  evidence (the WIRED green-gate), `EvaluatorOptimizerAgent`, `LlmJudgeGuardrail`,
  `DoomLoopTracker`, `ReflectionTracker`, `ErrorClass`, eval/scorers, debate/voting/mixture,
  flow combinators + `LoopAgent`, and **`BrowserAgentBuilder` as the domain-harness template**.
- **The one real gap (all agents converge):** no structured build/test/lint **verification
  tool** — agents get only raw `bash` and must parse fragile shell text; no machine-readable
  PASS/FAIL + exit-code provenance. Same deterministic-evidence primitive that made the
  tetris/browser judge reliable.

## Design (mirror `browser/`, reuse everything else)

New top-level module `crates/heartbit-core/src/codegen/`:

- `verify.rs` — PURE: `VerifyOutcome { passed, exit_code, command }`, `render_verify_result`,
  `parse_latest_verify(transcript) -> Option<VerifyOutcome>` (the deterministic-gate seam),
  plus `VerifyCommandTool` (builtin `Tool`).
- `builder.rs` — `CodeAgentBuilder<P>` (mirror `BrowserAgentBuilder`) + `CODE_SYSTEM_PROMPT`
  (loop invariants) + pure helpers `code_tools()`, `code_goal()`.
- `mod.rs` — re-exports + module doc.

### Verify output format (LOCKED — the one thing to decide up front)

Each command renders a block ending in a machine-greppable line; fail-fast on first non-zero;
the **last** line of the whole output is always the overall `VERIFY_RESULT:` (so it survives in
the goal judge's transcript tail):

```
$ cargo test --workspace
exit_code=0
--- output (tail, 3000 of 8123 bytes) ---
<tail of combined stdout+stderr>
VERIFY_RESULT: PASS exit_code=0 command=cargo test --workspace
```

- `passed = exit_code == 0` (claw's `TestCommandProvenance::passed`).
- A failing verify is **not** a tool error (`is_error=false`) — it is a successful verification
  reporting failure (the agent must read it and repair). `is_error=true` only if the command
  could not be spawned.
- `parse_latest_verify` finds the LAST `VERIFY_RESULT:` line → `{passed, exit_code, command}`.

### `VerifyCommandTool`

- `name = "verify"`, no agent-supplied command (runs the *configured* command(s) — its whole
  point vs `bash`). Spawns via `tokio::process` with `cwd = workspace`, configurable timeout,
  output tail cap. Sequential, fail-fast over the configured list.

### `CodeAgentBuilder<P>` (host on `AgentRunner`, NOT a flow leaf)

- `new(provider, workspace)`; `.verify_command(s)`, `.system_prompt`, `.name`, `.max_turns`
  (generous default e.g. 40), `.on_event`, `.max_identical_tool_calls`, `.guardrail`,
  `.tools_allow`, `.goal`, `.dangerous_tools` (default true), `.session_prune`.
- `build() -> Result<AgentRunner<P>, Error>` (sync; no MCP): code builtins
  (read/write/edit/patch/grep/glob/list[/bash]) rooted at `workspace` via `BuiltinToolsConfig`
  + the `verify` tool; `CODE_SYSTEM_PROMPT`; optional `.goal()`.
- **No `FilePathGuard`** — workspace jailing + `CorePathPolicy` + `protected_paths` already do
  this (avoid duplication). Builder just configures them.

### `code_goal(judge)` — repair driver via `agent.goal()`

`GoalCondition` objective keyed on the sentinel: *"complete only when the most recent `verify`
tool result in the transcript shows `VERIFY_RESULT: PASS` (exit_code 0); a done-claim without it
does NOT satisfy."* `with_max_continuations(~4)`. Goal re-injects the failure reason into the
SAME conversation → repair keeps context (stronger than `LoopAgent`).

### CODE_SYSTEM_PROMPT loop invariants (mirror BROWSER_SYSTEM_PROMPT)

UNDERSTAND (read before edit) → PLAN (ordered subgoals) → EDIT (minimal diffs, read-before-write)
→ VERIFY (call `verify`; never claim done without it) → REPAIR (read failure, fix cause,
re-verify; don't thrash) → FINISH (done only right after `VERIFY_RESULT: PASS`). + EFFICIENCY + SAFETY.

## Explicitly NOT building (claw's unwired layers / heartbit dupes)
lane_events/g004 contract (have events/observability) · recovery_recipes catalog (goal
continuation + reflection cover repair) · tiered GreenLevel (defer; claw never wired it) ·
worker screen-scraping (orthogonal) · mock-parity suite (have MockProvider) · per-framework
test-output parsers (exit code is universal; LLM reads tail for repair detail) ·
FilePathGuard (workspace jailing already exists).

## TDD slices (RED → GREEN)
1. `verify.rs` pure: `render_verify_result` + `parse_latest_verify` round-trip + sentinel format + tail cap.
2. `VerifyCommandTool`: run `true`/`false`/`sh -c 'exit N'` in a tempdir → PASS/FAIL sentinel, exit code, cwd=workspace, fail-fast multi-command, spawn-failure → is_error.
3. `CODE_SYSTEM_PROMPT` invariants present + `code_goal()` objective references the sentinel + `code_tools()` returns expected names rooted at workspace.
4. `CodeAgentBuilder::build()` wiring (MockProvider): tool list includes `verify`+code tools, system prompt set, goal wired, workspace rooted.
5. Wire `pub mod codegen;` + lib.rs re-exports.
6. Live `#[ignore]` qwen test (mirror `tetris_live`): build a tiny program in a tempdir, configured verify command, goal gates on PASS; run live.

## Gate: `cargo fmt -- --check && cargo clippy --all-targets -- -D warnings && cargo test`. Advisor sign-off + memory.

## STATUS: DONE (2026-06-02)
All 6 slices complete, TDD throughout (RED→GREEN per slice). 24 codegen unit tests + 1 live
`#[ignore]` qwen test green. **Workspace gate green**: `cargo fmt --all -- --check` +
`cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace` (0 failed).
Advisor signed off (3 points: workspace-gate cleared; VerifyCommandTool trust-model + timeout-orphan
documented). Live run loop: list→read→todowrite→verify[FAIL]→write→verify[PASS]; goal_met Some(true);
independent python3 oracle = exit 0. Post-live hardening: `parse_latest_verify` now requires the
canonical `exit_code=` sentinel (prose-echo spoof guard). Files:
`crates/heartbit-core/src/codegen/{mod,verify,builder,live}.rs`. NOT committed/PR'd (demo branch).
