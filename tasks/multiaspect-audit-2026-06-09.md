# Multi-Aspect Audit — 2026-06-09

Goal: full multi-aspect audit of the heartbit workspace; test and fix everything needed.

## Plan
- [x] Baseline quality gate — fmt ✅, clippy ✅ (0 warnings), tests 5225/5225 on re-run; 1 FLAKY test caught in first run (fixed, see Fixes)
- [ ] Scout prior audits for open findings — DONE: all Critical+High closed (2026-06-05 remediation), release blockers cleared; F-FS-5 residual accepted (macOS sandbox)
- [ ] Audit workflow: 10 aspect finders (agent-loop, router-modes, context-mgmt, security-delta, panic-safety, concurrency, test-coverage, config-api-docs, tui, small-crates) → adversarial verify per finding
- [ ] Triage confirmed findings by severity
- [ ] Fix confirmed findings (TDD: test first, then fix)
- [ ] Re-run full quality gate — must be green
- [ ] Final audit report in tasks/, memory update

## Known context
- Accepted risk (do NOT re-report): bash escape hatch bypasses file-tool jail (option C, 2026-06-07)
- Workspace version 2026.607.1; CI auto-publish on tag (f3734f8)
- Uncommitted: Cargo.toml adds heartbit-crm-core/-cli (17-line stubs); scratch dirs crm-project/, plate_solving_*/, scratch-rpg-game/, wget-log untracked

## Findings
(populated after verification)

## Fixes
### Status (2026-06-09)
- Audit workflow: 45 confirmed findings (7 refuted) across 10 aspects.
- 5 parallel fix agents + my runner/router/scope cluster: ALL applied.
  - **runner/builder/scope_guard cluster (mine, 19 findings)**: plan-gate text reconciled; interrupt-skips-stop-gates; cache-hit zero-usage; denied-batch doom tracking; escalation re-arm; ReadOnly whitelist deny + nudge suppression; CLARIFY batched-question + bash mutation; pinned-mode promotion guard; ingest-cap reorder; overflow-recovery ladder; compress bound; verify-replan request-scoping + budget re-arm; goal continuation re-arm; ScopeGuard workspace anchoring. 16 new regression tests, all green.
  - **tui (7)**: lifecycle event attribution, deferred /model respawn, stale-exit guard, tool_index clear, panic AgentExited guard, Esc-race park flag, u16 row saturation. 294/294, clippy clean.
  - **tools/channel/codegen (6)**: symlink-intermediate jail escape, chunk_message UTF-8, VerifyCommandTool process-group, VERIFY_RESULT anchoring, rg stdout cap, websearch UTF-8. 383 green.
  - **router (3)**: classify_l1 brace panic, GO_TOKENS whole-message, STUDY word-boundary. 75 green.
  - **orchestrator (1, high)**: on_approval + learned_permissions propagate to sub-agents (delegate/squad/spawn). 120 green.
  - **hygiene (8)**: release.yml test gate, dead config key, Cargo.toml stub removal, .gitignore scratch, CHANGELOG, daemon CORS default-off, macro default param, gateway doc. (also promoted `toml` dep in heartbit-cli/Cargo.toml)
- [x] Full workspace gate (fmt+clippy+test) — **GREEN**: fmt clean, clippy 0 warnings, **5292 pass / 0 fail** (baseline 5225 → +67 regression tests).
- [x] Adversarial review of the FIXES (6 reviewers) — found 5 real defects in the fixes; ALL fixed with TDD (+4 regression tests, 2 proven red-without-fix). See report §"Adversarial review of the fixes".
- [x] Final gate after review-fixes — **GREEN: 5296 pass / 0 fail**, clippy 0, fmt clean.
- Report: `tasks/multiaspect-audit-report-2026-06-09.md`.

### Residuals (low-severity, documented — NOT blockers)
- VERIFY_RESULT sentinel: hardened to line/tool-result-wrapper anchoring in `parse_latest_verify`; a fully robust fix would key the gate on the tool's own VerifyOutcome in runner state (deferred — low).
- Esc-race (TUI): prevented via a shared `agent_parked` flag because `InterruptHandle::rearm()` is `pub(crate)`; the residual gap is the sub-ms window between the runner's last token check and its on_input park (only core-side rearm-on-park closes it fully).
- `is_protected` in resolve_path still evaluates the lexical path (kept minimal); only matters for an in-workspace symlink to an in-workspace protected file — the outside case is caught by the new deepest-existing-ancestor containment check.
- `crates/heartbit-cli/Cargo.toml`: `toml` promoted dev→deps to deliver the daemon CORS serde-default config field.

### Detail
- [x] FIX-1 (high, reliability+durability): `JsonlQuoteSeenStore::record` never flushed — `tokio::fs::File` buffers internally, so the appended line could be invisible to readers (flaky `jsonl_record_is_idempotent` under parallel load, `left: 0`) and lost on fast process exit despite the store's "restart durability" contract. Added `file.flush().await?` after `write_all` (crates/heartbit-ghost/src/quote/sources.rs:272). Targeted tests green (4/4).
