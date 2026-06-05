# Security Remediation — 2026-06-05 (re-triage of the 2026-05-06 audit)

Re-triage (workflow `wf_d5ade441`) of all 27 Critical+High findings against CURRENT code:
**15 FIXED, 12 open/partial.** This doc tracks closing the 12. Each fix = a reproducing
test (proves the vuln, then proves it's closed) + the full workspace gate.

## Open Criticals (the gate to "clean") — ALL CLOSED ✅
- [x] **F-MCP-1** — `connect_http` built a bare reqwest client (no `SafeDnsResolver`) → DNS-rebind to 169.254.169.254 leaked `auth_header`. **FIXED** (6ca6337): build via `crate::http::safe_client_builder()`.
- [x] **F-MCP-3** — MCP server `handle_request` had no auth when `auth_callback` was None. **FIXED**: fail-closed unless `allow_unauthenticated()` opted in; + session TTL/idle eviction + MAX_SESSIONS.
- [x] **F-FS-1** — intermediate-directory-symlink TOCTOU: canonical `composed` discarded; `O_NOFOLLOW` only guards the trailing component. **FIXED** (bc38a8d): new `write_beneath_root` walks from the trusted canonical root via `openat(O_NOFOLLOW)` per component — no intermediate OR final symlink survives; write re-validates at write time so swap-before/after-check both refused. `CorePathPolicy::allowed_root_for()`; write.rs + patch.rs routed through it. Primitive + e2e tests.
- [x] **F-AGENT-2** — `FormSquadTool` omitted orchestrator guardrails. **FIXED** (07782e0): added `guardrails` field, combine orchestrator+agent_def chains as the delegate path does. e2e propagation test.

## Open Highs — ALL CLOSED (except F-FS-5 accepted residual) ✅
- [x] **F-FS-2** — `BuiltinToolsConfig::default()` env_policy was `Inherit`. **FIXED** (batch 1): default to `EnvPolicy::default()` (safe Allowlist); daemon no-workspace branch fixed.
- [x] **F-MCP-2** — MCP tool names could shadow builtins via daemon ordering. **FIXED** (batch 1): builtins-first ordering (first-wins dedup).
- [x] **F-NET-1** — twitter media `.bytes().await` buffered the whole body before the 5 MB check → OOM. **FIXED** (this batch): `read_body_capped(.., 5MB+1)` + truncation check. SSRF already covered by `vendor_client_builder` (SafeDnsResolver + strict IP).
- [x] **F-LLM-5** — non-streaming success bodies (`response.json()`) uncapped on all 4 providers; anthropic error path used unbounded `.text()`. **FIXED** (this batch): new `read_json_capped` (16 MiB cap, `from_slice`) on all 4; anthropic errors routed via `api_error_from_response`.
- [x] **F-LLM-4** — gemini streaming accumulators uncapped; anthropic/gemini tool-call COUNT unbounded. **FIXED** (this batch): gemini text/tool-call caps mirroring openrouter; anthropic content-block count bounded in `flush_current_block`. (anthropic/openrouter text+args caps were already present.)
- [x] **F-AGENT-6** — base64/rot13-encoded injection overrides bypassed the classifier (flagged 0.30 but not decoded; rot13 0.00). **FIXED** (this batch): `decode_candidates` decodes base64 runs + rot13, re-scores via `score_core`, takes the MAX. (homoglyph + multilingual were already present.)
- [x] **F-AGENT-7** — `blackboard_write` emitted no audit record. **FIXED** (this batch): `BlackboardAudit` ctx threaded from the orchestrator; each write emits an `AuditRecord{agent,event_type:"blackboard_write",payload:{key,value_bytes},user_id,tenant_id}` (best-effort).
- [ ] **F-FS-5** — ACCEPTED RESIDUAL. No kernel sandbox on macOS / Linux `--no-default-features`; bash checks only cwd. The **shipped Linux binary is safe** (`default=["sandbox"]` → landlock). macOS `sandbox-exec` enforcement is a platform-specific follow-up requiring macOS test hardware; documented, not fixed here.

## Double-check (marked fixed, verify)
- [x] **F-NET-2** — DNS-rebind: audited reqwest client construction; F-MCP-1 (connect_http) was the surviving instance and is FIXED (batch 1). Providers use hardened builders.
- [x] **F-AGENT-4** — CONFIRMED INTENTIONAL. Judge fail-open is by design (availability over strictness) and is now *visible* via an audit event. A `fail_closed: bool` opt-in remains a future enhancement, not a vuln.

## Fix batches (sequential — many touch execute.rs / shared files)
1. **MCP:** F-MCP-1, F-MCP-3, F-MCP-2 (+ F-NET-2 reqwest-site audit).
2. **FS:** F-FS-2 (quick), F-FS-1 (deep, openat2), F-FS-5.
3. **AGENT:** F-AGENT-2, F-AGENT-6, F-AGENT-7 (+ F-AGENT-4 confirm).
4. **LLM/NET:** F-LLM-5, F-LLM-4, F-NET-1.
