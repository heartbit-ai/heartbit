# Multi-Aspect Audit Report — heartbit workspace

**Date:** 2026-06-09 · **Scope:** full workspace (11 crates) · **Method:** 10-aspect parallel audit with adversarial verification, then per-finding TDD fixes, then adversarial diff review.

## Outcome

- **Audit:** 10 aspect finders → 52 raw findings → **45 confirmed** after adversarial verification (3-vote refutation on critical/high), 7 refuted.
- **Fixes:** all 45 addressed (6 parallel fix agents + main-loop runner/router/scope cluster). Low-severity residuals documented below.
- **Gate:** fmt clean · clippy `-D warnings` 0 · **`cargo test --workspace --exclude mini-crm` = 5292 pass / 0 fail** (baseline 5225 → +67 regression tests).
- Plus a **flaky test root-caused and fixed** outside the audit (caught in baseline): `jsonl_record_is_idempotent` — missing `flush()` after a buffered `tokio::fs::File` write (durability bug, not just a test flake).

## Findings & fixes by aspect

### agent-loop / runner (high-impact subset)
- **[high] Plan-gate re-injected the dead "temporary directory means OUTSIDE this repository" guidance** that commit 5c7f319 was supposed to eliminate — the one message missed by that reconciliation. Now points to an in-workspace gitignored `./scratch-<name>`. (runner.rs)
- **[high] User interrupt was swallowed by the stop-gates** — the synthesized `[interrupted by user]` turn ran the goal judge / verify-replan / ask-act-study gates and auto-continued the run the user asked to stop. Added an `llm_interrupted` flag that short-circuits all six gates to `on_input`. (runner.rs)
- **[high] Goal continuation budget never re-armed on a new chat request** — a second `set_goal` could get ZERO continuations. Re-armed in the on_input block. (runner.rs)
- **[medium] Cache hits re-billed the original call's tokens** (totals, cost, budget, per-tenant) — zeroed `cached.usage`. (runner.rs)
- **[medium] Permission/approval-denied batches bypassed doom-loop detection** — a model hammering a denied tool spun to max_turns. Now records the pre-filter batch and hard-stops. (runner.rs)
- **[medium] Hard-escalation one-shot couldn't re-arm after an advisor consult** — reset `escalation_fired` on consult. (runner.rs)
- **[medium] 256KB ingest cap bypassed when a repair hint fired the same turn** (regression of 7de5df6 layer 2) — moved the cap before hint injection. (runner.rs)
- **[medium] Reactive overflow recovery was single-shot** — a second overflow after truncation dead-ended instead of escalating to summarization. Replaced the boolean with a staged ladder. (runner.rs)
- **[medium] Verify-replan budget was session-cumulative + scanned the whole transcript** — stale FAILs re-triggered it on unrelated requests; budget never reset. Now per-request, scoped to the current request's message suffix. (runner.rs)
- **[low] compress_tool_output sent unbounded input** — same overflow class as the 7de5df6 summary bug. Bounded with head+tail. (runner.rs)

### router-modes
- **[high] ReadOnly deny backstop covered only 4 tool names** — delegation/MCP/A2A executed with side effects in STUDY/ANSWER. Inverted to a whitelist mirror of the mask (`is_read_only_tool`); also suppressed the delegation nudge in read-only modes. (runner.rs/tool_filter.rs)
- **[medium] CLARIFY ask-first armed from CALLS not results** — a `question` batched with mutations let writes run before the user answered. Plan-gate now refuses the batch; flags commit only after the gates pass. (runner.rs)
- **[medium] bash not counted as a mutation** — bash-driven builds evaded ask-first/scope in CLARIFY. Counted now. (runner.rs)
- **[medium] Pinned STUDY/CLARIFY silently promoted to EXECUTE by a bare affirmation** — promotion now consults `router.pinned_mode()`. (runner.rs/router.rs)
- **[medium] GO_TOKENS matched as substrings** ("don't do it yet" promoted) — now whole-message `is_explicit_go`. (router.rs)
- **[medium] L0 STUDY markers matched inside words** ("analyseur") — word-boundary matching (accent-aware). (router.rs)
- **[medium] classify_l1 panicked when `}` preceded `{`** — added the `end<=start` guard. (router.rs)

### orchestrator
- **[high] Human-approval gate silently vanished for delegated work** — `on_approval` (and `learned_permissions`) were never forwarded to sub-agents; one approved delegate dropped the whole HITL gate. Now propagated across delegate_task / form_squad / spawn_agent, with a shared learned-permission store for the run. (orchestrator.rs)

### tools / channel / codegen
- **[medium] New-file write through a symlinked intermediate dir escaped the jail** — canonicalize the deepest existing ancestor and require it in-workspace. (tool/builtins/mod.rs)
- **[medium] chunk_message panicked on multibyte UTF-8** — char-boundary-safe slicing that always advances. (channel/mod.rs)
- **[medium] VerifyCommandTool orphaned its child** on interrupt/timeout — `kill_on_drop` + process-group + RAII killer. (codegen/verify.rs)
- **[low] VERIFY_RESULT sentinel forgeable** — anchored parse to line-start / tool-result wrapper (residual below). (codegen/verify.rs)
- **[low] try_ripgrep buffered unbounded stdout** — global byte cap + concurrent stderr drain + kill. (grep.rs)
- **[low] WebSearch sliced HTML mid-codepoint** — floor_char_boundary. (websearch.rs)

### TUI
- **[high] Sub-agent lifecycle events leaked into the UI plane** flipping the parent `running` flag — gated lifecycle on entry-agent attribution; sub-agent events map to cost-only / roster-only. (msg.rs/app.rs)
- **[high] /model mid-run spawned a second concurrent engine** on the same workspace — deferred to turn-idle (queue + flush at every idle transition), with a backstop guard in the effect handler. (app.rs/main.rs)
- **[low] /clear & /resume left tool_index stale** — cleared on clear/load. (app.rs)
- **[low] Agent-thread panic never sent AgentExited** (running stuck true) — RAII `AgentExitGuard` always signals. (main.rs)
- **[low] Esc race latched the interrupt token while idle** — shared `agent_parked` flag skips interrupt while parked. (app.rs/main.rs)
- **[low] Wrapped-row count truncated to u16** past 65,535 rows — saturating clamp. (ui.rs)

### config / release / hygiene
- **[medium] Tag push published heartbit-core with ZERO test gate** — added a test job; `publish-crate` now `needs: [build, test]`. (release.yml)
- **[medium] Example config set a nonexistent `[orchestrator].system_prompt`** — moved the pipeline directive to where it reaches the model. (configs/twitter-content-gen.toml)
- **[medium] Uncommitted Cargo.toml added two untracked stub crates as members** — removed (dirs left on disk). (Cargo.toml)
- **[medium] Daemon wildcard CORS + optional auth** exposed task execution to any browser origin — secure default OFF, configurable. (daemon/handlers.rs+mod.rs)
- **[low] CHANGELOG stale** (8 commits since the tag) — `[Unreleased]` added.
- **[low] Untracked scratch projects not gitignored** — added (incl. literal `wget-log`).
- **[low] heartbit-macro `#[tool(default)]` advertised a default it never applied** — generated code applies it; schema omits it from `required`. (macro/lib.rs)
- **[low] Gateway doc claimed haiku, code wired sonnet** — doc corrected. (gateway/main.rs)

## Residuals (low-severity, deliberate)
1. **VERIFY_RESULT** still forgeable by content reproducing the exact sentinel at a line start; robust fix needs runner-state keying on the tool's own VerifyOutcome.
2. **TUI Esc-race** narrowed to a sub-ms window; full closure needs core-side `rearm()` on park (currently `pub(crate)`).
3. **resolve_path `is_protected`** evaluates the lexical path; only matters for an in-workspace symlink to an in-workspace protected file — the outside case is now caught by containment.
4. `heartbit-cli/Cargo.toml`: `toml` promoted dev→deps to back the CORS serde-default config field.

## Not re-reported (accepted risk)
- The **bash tool bypasses the file-tool path jail** (option C, accepted 2026-06-07). The jail is enforced for file tools only.

## Adversarial review of the fixes (second pass)
A correctness review over all 26 changed files (6 reviewers + per-finding verification) found **5 real defects in the fixes themselves**, all then fixed with TDD:

1. **[medium] Verify-replan request-scoping regression (mine).** `request_start_msg` is an absolute message index; a mid-request compaction shrank the list below it → empty slice → the gate missed a RED verify kept in the tail and finished on red. **Fix:** re-anchor `request_start_msg` past the index-0 summary at every `inject_summary` site. Regression test `verify_replan_survives_midrequest_compaction` (proven red without the fix).
2. **[low] All-denied doom hard-stop ignored fuzzy repeats (mine).** Varying-input hammering of a denied tool still spun to max_turns. **Fix:** extracted `denied_batch_doom_abort` (dedupes the two identical blocks) handling exact AND fuzzy. Test `fuzzy_denied_tool_hammering_hits_doom_hard_stop`.
3. **[high] STUDY word-boundary regressed inflected French verbs.** "analyser/étudier/évaluer …" + a code anchor routed to EXECUTE (the safety-protected direction) because only bare stems were markers. **Fix:** added the inflected infinitive/imperative forms (not substrings of the noun forms analyseur/évaluateur; "explorer" deliberately excluded). Test `inflected_study_verbs_with_anchor_route_study_not_execute`.
4. **[medium] /clear left a half-streamed buffer.** `/clear` mid-stream didn't clear `active`/`active_reasoning` (SessionLoaded did) → pre-clear content resurrected as a ghost cell. **Fix:** mirror the SessionLoaded reset. Test `clear_mid_stream_drops_the_half_streamed_buffer`.
5. **[medium] release.yml binary upload not gated on tests.** The `build` job (GitHub Release upload) ran parallel to `test`; only crates.io was gated. **Fix:** `build` now `needs: test`.

## Verification trail
- Adversarial verification of every finding (3-vote refutation on critical/high) before fixing.
- TDD: each fix landed a failing test first; **71 new regression tests** pin the fixed behaviors (67 first pass + 4 review-fix); two of the review-fix tests proven red-without-fix.
- **Final gate: fmt clean · clippy `-D warnings` 0 · `cargo test --workspace --exclude mini-crm` = 5296 pass / 0 fail.**
