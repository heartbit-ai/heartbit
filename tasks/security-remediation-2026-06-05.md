# Security Remediation — 2026-06-05 (re-triage of the 2026-05-06 audit)

Re-triage (workflow `wf_d5ade441`) of all 27 Critical+High findings against CURRENT code:
**15 FIXED, 12 open/partial.** This doc tracks closing the 12. Each fix = a reproducing
test (proves the vuln, then proves it's closed) + the full workspace gate.

## Open Criticals (the gate to "clean")
- [ ] **F-MCP-1** — `connect_http` builds a bare reqwest client (no `SafeDnsResolver`) → DNS-rebind to 169.254.169.254 leaks `auth_header`. **Fix:** build via `crate::http::safe_client_builder()` (~1 line, as A2A does).
- [ ] **F-MCP-3** — MCP server `handle_request` has no auth when `auth_callback` is None → network clients run `tools/call` bash/write unauthenticated. **Fix:** fail-closed unless `allow_unauthenticated()` opted in; + session TTL/idle eviction.
- [ ] **F-FS-1** — intermediate-directory-symlink TOCTOU: `check_path_for_create`'s canonical `composed` path is discarded; `O_NOFOLLOW` only guards the trailing component. write.rs:147 + patch.rs:272. **Fix:** write to the canonical `composed` target (or post-write re-canonicalize + `starts_with` check + unlink on mismatch); best = `openat2` RESOLVE_NO_SYMLINKS|RESOLVE_BENEATH. + intermediate-component swap test.
- [ ] **F-AGENT-2** — `FormSquadTool` omits orchestrator guardrails (the delegate path was fixed; squad path wasn't). **Fix:** add `guardrails` field, combine orchestrator+agent_def chains as the delegate path does.

## Open Highs
- [ ] **F-FS-2** — `BuiltinToolsConfig::default()` env_policy = `Inherit` (mod.rs:257) → CLI bash inherits ANTHROPIC_API_KEY/AWS_*/GITHUB_TOKEN. **Fix:** default to `EnvPolicy::default()` (safe Allowlist); fix daemon no-workspace branch (execute.rs:506); secret-name filter in the Inherit arm.
- [ ] **F-MCP-2** — MCP tool names forwarded raw → hostile `bash` shadows the builtin (daemon ordering); input_schema trusted. **Fix:** prefix `mcp_{server}_{tool}` (or hard-refuse builtin collisions); builtins-first ordering; validate input_schema.
- [ ] **F-NET-1** — twitter media upload `.bytes().await` unbounded → OOM. **Fix:** `read_body_capped(.., 5MB)`; route media_url through `SafeUrl::parse`.
- [ ] **F-LLM-5** — success bodies (`response.json()`) uncapped on all 4 providers; anthropic error path skips the capped helper. **Fix:** streamed MAX_BODY_BYTES + `from_slice`; route anthropic errors via `api_error_from_response`.
- [ ] **F-LLM-4** — gemini accumulators uncapped; anthropic/gemini tool-call COUNT unbounded. **Fix:** STREAM_MAX_TOOL_CALLS + text/args byte caps mirroring openrouter.
- [ ] **F-AGENT-6** — base64/rot13-encoded injection overrides bypass the classifier (score 0.30/0.00). **Fix:** decode flagged base64+rot13, re-score, take max.
- [ ] **F-FS-5** — no kernel sandbox on macOS / Linux `--no-default-features`; bash checks only cwd. **Fix:** `sandbox-exec` profile OR hard-refuse when policy required but no kernel enforcement. (Shipped Linux binary is safe — `default=["sandbox"]`.)
- [ ] **F-AGENT-7** — `blackboard_write` emits no audit record. **Fix:** emit one with caller/key/tenant.

## Double-check (marked fixed, verify)
- [ ] **F-NET-2** — DNS-rebind fixed for webfetch only; `SafeDnsResolver` is per-client-builder, not global. **Action:** audit EVERY `reqwest::Client` construction for the safe builder (F-MCP-1 is the same bug surviving).
- [ ] **F-AGENT-4** — judge fail-open made *visible* (audit event) but execution still proceeds. **Action:** confirm intent; consider the unimplemented `fail_closed: bool`.

## Fix batches (sequential — many touch execute.rs / shared files)
1. **MCP:** F-MCP-1, F-MCP-3, F-MCP-2 (+ F-NET-2 reqwest-site audit).
2. **FS:** F-FS-2 (quick), F-FS-1 (deep, openat2), F-FS-5.
3. **AGENT:** F-AGENT-2, F-AGENT-6, F-AGENT-7 (+ F-AGENT-4 confirm).
4. **LLM/NET:** F-LLM-5, F-LLM-4, F-NET-1.
