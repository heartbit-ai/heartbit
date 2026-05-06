# Performance Audit — `heartbit-core` (2026-05-06)

**Scope**: `crates/heartbit-core/` (~87 kLOC, runner.rs alone is 2355 lines).
**Method**: 7 parallel static audits, file:line evidence, severity = `frequency × per-call cost`. Audit-only — no fixes were committed during the audit phase.
**Goal**: best-in-class hot-path performance **without** weakening any closed security finding from `tasks/security-audit-heartbit-core-2026-05-06.md`.

---

## Executive Summary

**113 findings** across 7 sub-reports (linked below). All preserve security boundaries — every sub-audit explicitly enumerated REJECTED "obvious wins" that would re-open closed F-* findings.

| Severity     | Count | Cumulative est. impact (per 100-turn run) |
|--------------|-------|-------------------------------------------|
| **Critical** | 15    | 15–60 ms saved + 5–25 MB allocs avoided   |
| **High**     | 39    | 25–80 ms saved + ~5× lock contention reduction |
| **Medium**   | 38    | 5–20 ms saved                             |
| **Low**      | 21    | <5 ms (mostly micro-opts)                 |

**Headline aggregate**: optimistic 50–150 ms shaved per 100-turn agent run, plus a step-change in memory recall latency at 10k+ entries (BM25 inverted index alone saves 0.5–2 ms **per recall**).

**Sub-reports**:
- [Agent loop / runner.rs](perf-audit-runner.md) — 15 findings
- [LLM providers](perf-audit-llm.md) — 18 findings
- [MCP / A2A](perf-audit-mcp.md) — 12 findings
- [Memory subsystem](perf-audit-memory.md) — 18 findings (highest leverage)
- [Builtin tools](perf-audit-builtins.md) — 15 findings
- [Guardrails / observability / eval / channel](perf-audit-cross.md) — 17 findings
- [Cross-cutting allocations / locks](perf-audit-crosscut.md) — 18 patterns

---

## Cross-cutting themes (each appears in ≥3 sub-reports)

These are patterns where a single workspace-wide refactor wins everywhere at once. Prioritise these — atomic commits with high blast-radius, low risk.

### T1. Regex compiled per call → `LazyLock<Regex>` everywhere

Found in 5 sub-reports. The recent security cycle added several regex-using guardrails (`F-AGENT-15` multilingual phone, `F-MCP-16` `redact_idp_body`, `F-NET-7` HTML sanitization, `F-MCP-6` `sanitize_log_field`) without `LazyLock`, so each call recompiles.

**Sites** (23 occurrences):
- `crates/heartbit-core/src/agent/guardrails/secret_scanner.rs:36, 44, 278` — Slack/JWT/custom (warm)
- `crates/heartbit-core/src/agent/guardrails/injection.rs:66` — config-loaded pattern per guard exec (hot)
- `crates/heartbit-core/src/agent/guardrails/pii.rs:28, 344` — phone, token (warm/hot)
- `crates/heartbit-core/src/agent/guardrails/tool_policy.rs:296` — `rm\s+-rf` (warm)
- `crates/heartbit-core/src/agent/orchestrator.rs:1233` — agent-name validation per spawn
- `crates/heartbit-core/src/agent/evaluator.rs:156` — judge dynamic pattern
- `crates/heartbit-core/src/tool/mcp.rs:441` — `redact_idp_body` 3 patterns (hot during OAuth retry)
- `crates/heartbit-core/src/tool/builtins/webfetch.rs:232–249` — 3 HTML sanitize patterns (hot per fetch)
- `crates/heartbit-core/src/tool/builtins/grep.rs:216, 220–222` — fallback regex + glob include
- `crates/heartbit-core/src/tool/builtins/list.rs:135` — `DEFAULT_IGNORES`

**Hypothesised cost**: 50–500 µs per `Regex::new` × ~5 hot sites × 50 turns = **5–10 ms / run**.
**Fix**: workspace-wide LazyLock pass. Single PR.
**Security delta**: none — semantics identical.

### T2. `std::sync::RwLock` → `parking_lot::RwLock` on hot non-await locks

Found in 4 sub-reports. `parking_lot` is already a workspace dep (Cargo.toml:39).

**Hot/warm sites**:
- `crates/heartbit-core/src/agent/audit.rs:220` — every audit record
- `crates/heartbit-core/src/agent/runner.rs:158` — permission rules per tool call
- `crates/heartbit-core/src/agent/tenant_tracker.rs:34` — every token accumulation
- `crates/heartbit-core/src/llm/circuit.rs:220` — every LLM call
- `crates/heartbit-core/src/memory/in_memory.rs:32` — primary memory lock (P-MEM-1, **Critical**)
- `crates/heartbit-core/src/tool/mcp_server.rs:153` — sessions per message
- `crates/heartbit-core/src/channel/session.rs:107` — per routing
- `crates/heartbit-core/src/tool/builtins/file_tracker.rs:16` — per file op

Plus `Mutex` → `parking_lot::Mutex` for short critical sections (`memory/reflection.rs:12`, `agent/cache.rs`, `guardrails/action_budget.rs:31`).

**Hypothesised cost**: 50–100 ns saved per uncontended read × thousands of reads/run + dramatic reduction under contention. **5–10 ms / run** + concurrency unlock.
**Fix**: drop-in replacement (no API change). Heartbit doesn't rely on poisoning semantics.

### T3. `HashMap<String, _>` on hot lookup → `FxHashMap` (rustc-hash) on non-adversarial keys

Found in 3 sub-reports. SipHash is overkill for tool names, memory IDs, session UUIDs, model→price tables, `TokenCacheKey`.

**Sites**:
- `crates/heartbit-core/src/agent/runner.rs:95` — tool registry lookup per call
- `crates/heartbit-core/src/memory/in_memory.rs:32` — entries map
- `crates/heartbit-core/src/tool/mcp.rs:794` — token cache (P-MCP-4)
- `crates/heartbit-core/src/llm/pricing.rs:24–65` — match → `LazyLock<FxHashMap>` (P-LLM-4)
- `crates/heartbit-core/src/agent/tenant_tracker.rs:34`, `blackboard.rs:38`, `knowledge/in_memory.rs:21`

**Cost**: ~50 ns saved/lookup × thousands of lookups/run ≈ 0.3–1 ms / run, but compounds.
**Risk gating**: only swap on keys not user-controlled. Keep SipHash where attacker can pick the key.
**Cargo addition**: `rustc-hash = "2"` in workspace.

### T4. Cloning whole structs that should be shared via `Arc`

- `tool_defs: Vec<ToolDefinition>` cloned per `execute()` (`runner.rs:374, 397`) → `Arc<Vec<ToolDefinition>>` — **P-RUNNER-1, Critical**, 1–5 ms/session.
- `request: CompletionRequest` cloned per cascade tier (`cascade.rs:126, 180`) and per retry (`retry.rs:208`) → `Arc<CompletionRequest>` — **P-LLM-12, Critical**, 100–500 KB/escalation.
- `MemoryEntry` cloned per recall result (`memory/in_memory.rs:194`) → `Vec<Arc<MemoryEntry>>` or `Cow` — **P-MEM-5, Critical**, 1–5 ms/recall at N=10k.
- `CompletionResponse` cloned on cache put/get (`runner.rs:689`) → `Arc<CompletionResponse>` — **P-RUNNER-8, High**.
- `ToolDefinition.input_schema` reserialised per request in Anthropic (`anthropic.rs:257`) → `OnceLock<Value>` — **P-LLM-16, High**.
- `tool.definition().clone()` per `tools/list` (`mcp_server.rs:332`) → cache pre-serialised JSON — **P-MCP-10, Medium**.

**Theme**: convert "heap-allocated and copied" to "shared by `Arc`". No security delta — Arc semantics identical to clone.

### T5. SSE per-chunk allocations (streaming hot path)

- `SseParser::feed()` clones each line via `to_string()` (`anthropic.rs:438, 463, 479`) — **P-LLM-2, High** — 15–25 KB churn/response.
- `emit_event()` joins `data_lines` even when empty (`anthropic.rs:490–491`) — **P-LLM-14, Critical** — 1 alloc/SSE event.

**Fix**: `&str` slices into the buffer (lifetime adjustments) or `bytes::Bytes` zero-copy framing. Compounds at every chunk — biggest streaming-mode win.

### T6. Per-pattern regex scanning → `RegexSet` / `aho-corasick`

- **PII guardrail** runs 4 detectors as 4 separate `find_iter` passes (`pii.rs:79–101`) — **P-CROSS-2, High** — 4–8× speedup combining into a single `RegexSet`.
- **Secret scanner** runs 6+ patterns as 6 passes (`secret_scanner.rs:81–103`) — **P-CROSS-3, High** — `RegexSet` + bounded scanning on >100 KB.
- **Sensor security** loops INJECTION_PATTERNS substring (`sensor_security.rs:66–74`) → `aho-corasick::AhoCorasick` (P-CROSS-17).

These run every `post_llm` and `post_tool` — guardrail aggregate goes from ~50–200 ms / session down to ~10–20 ms (10–25× speedup on regex-heavy paths).

---

## Top 15 wins (ranked by expected impact)

| # | Finding | Impact | Effort | Sub-report | Security gate |
|---|---------|--------|--------|------------|---------------|
| 1 | **P-MEM-2**: BM25 inverted index | 0.5–2 ms / recall at N=10k (algorithmic step-change) | High | memory | F-MEM-5 tenant filter still applies |
| 2 | **P-MEM-1**: Memory `RwLock` → parking_lot + split read/write scope | 200–500 µs / recall + concurrent recalls | Medium | memory | none |
| 3 | **P-CROSS-2**: PII detectors → single `RegexSet` | 4–8× on `post_llm` | Low | guardrails | F-AGENT-15 patterns preserved |
| 4 | **P-CROSS-3**: Secret scanner → `RegexSet` + bounded | 50–70% on large outputs | Low | guardrails | none |
| 5 | **T1**: Workspace-wide regex `LazyLock` pass (5 hot sites) | 5–10 ms / run | Low | crosscut + others | none |
| 6 | **P-MEM-5**: `Vec<Arc<MemoryEntry>>` recall return | 1–5 ms / recall at scale | Medium | memory | none |
| 7 | **P-LLM-2 + P-LLM-14**: SSE zero-copy / delayed-join | per-chunk savings, compounds | High | LLM | none |
| 8 | **P-RUNNER-1**: `Arc<Vec<ToolDefinition>>` instead of clone | 1–5 ms / session start | Low | runner | none |
| 9 | **P-LLM-12 + P-LLM-6**: `Arc<CompletionRequest>` for cascade/retry | 100–500 KB / escalation | Low | LLM | none |
| 10| **P-MEM-3**: Lazy strength-decay caching | 50–100 µs / recall (×10k entries) | Low | memory | none |
| 11| **T2**: `parking_lot::RwLock` workspace swap | 5–10 ms / run | Low | crosscut + others | none |
| 12| **P-TOOL-5 + P-TOOL-14**: Patch fuzzy match short-circuit + pre-normalize | 50–200 µs / patch | Low | builtins | F-FS-12 defense-in-depth retained |
| 13| **P-CROSS-16**: Permission ruleset cached with mtime invalidation | 10–50 ms × 100 agents = 1–5 s / multi-agent startup | Low | guardrails | F-AGENT-13 caps preserved |
| 14| **P-CROSS-7**: Audit `strip_content` filter-in-place (no full clone) | 1 ms / record on 100 KB payloads | Medium | guardrails | F-AUTH-3 allow-list semantics identical |
| 15| **P-MEM-6**: Per-tenant secondary index | 100–300 µs / recall at high tenant count | Medium | memory | F-MEM-5 isolation reinforced |

---

## Per-area findings (counts and links)

| Area | File | Critical | High | Medium | Low | Total |
|------|------|----------|------|--------|-----|-------|
| Agent loop / runner | [perf-audit-runner.md](perf-audit-runner.md) | 1 | 7 | 4 | 3 | 15 |
| LLM providers | [perf-audit-llm.md](perf-audit-llm.md) | 4 | 7 | 5 | 2 | 18 |
| MCP / A2A | [perf-audit-mcp.md](perf-audit-mcp.md) | 1 | 3 | 6 | 2 | 12 |
| Memory subsystem | [perf-audit-memory.md](perf-audit-memory.md) | **5** | 8 | 4 | 1 | 18 |
| Builtin tools | [perf-audit-builtins.md](perf-audit-builtins.md) | 2 | 6 | 5 | 2 | 15 |
| Guardrails/observability | [perf-audit-cross.md](perf-audit-cross.md) | 1 | 4 | 8 | 4 | 17 |
| Cross-cutting patterns | [perf-audit-crosscut.md](perf-audit-crosscut.md) | 1 | 4 | 6 | 7 | 18 |
| **Total** | | **15** | **39** | **38** | **21** | **113** |

The memory subsystem is the highest-leverage area — 5 of 15 Critical findings are in `memory/`, and the BM25 + recall path dominates at scale.

---

## Security-regression matrix

Every sub-audit was constrained to flag any "performance win" that would weaken a closed F-* finding. **Total REJECTED candidate optimisations: 30+**, all documented in their respective sub-reports. The five most tempting that are **not** allowed:

| Tempting "win" | Re-opens | Status |
|---------------|----------|--------|
| Cache `MemoryEntry` results across tenants | F-MEM-5 (cross-tenant leak) | REJECTED |
| Share one `reqwest::Client` across `IpPolicy` contexts | F-NET-2 (DNS rebinding) | REJECTED |
| Lift `MAX_WALK_DEPTH=8` for skill discovery | F-FS-7 (symlink-fork DoS) | REJECTED |
| Lift `PERMISSIONS_FILE_MAX_BYTES=1MB` | F-AGENT-13 | REJECTED |
| Drop `redact_idp_body` regex pass to "save compile cost" | F-MCP-16 (token leak in IdP logs) | REJECTED — fix is `LazyLock`, not removal |
| Replace `subtle::ConstantTimeEq` with `==` | timing attack on tokens | REJECTED |
| Default `AuditMode::Full` again | F-AUTH-6 (privacy regression) | REJECTED |
| Skip nonce check in `PendingEntry` | F-AUTH-5 | REJECTED |
| Remove nonce-bearing cwd marker | F-FS-8 | REJECTED (counter+hash is acceptable replacement, P-TOOL-10) |

The fixes proposed in the top-15 wins **strengthen** the F-MEM-5 boundary (per-tenant secondary index in #15) and **preserve** every other closed finding.

---

## Validation strategy

| Category | Count | Approach |
|----------|-------|----------|
| `static-only` (code review proves equivalence) | ~45 | Land directly behind `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`. |
| `needs-bench` (correctness obvious, speedup needs measurement) | ~55 | Criterion benches landed in **task #65** (parallel deliverable). |
| `measured` (must profile to confirm hypothesis) | ~13 | flamegraph / `perf record` / `tokio-console` after benches. |

---

## Cargo.toml-level recommendations

Additive — none break existing builds.

```toml
# workspace.dependencies
parking_lot = { workspace = true }    # already present, adopt on hot non-await locks
rustc-hash  = "2"                     # NEW — for FxHashMap on non-adversarial keys
ahash       = "0.8"                   # OPTIONAL — for AHashMap or RandomState swap
smallvec    = "1"                     # NEW — for keywords, related_ids (typically <8 elements)
bytes       = { workspace = true }    # already present, adopt for SSE / MCP zero-copy
aho-corasick = "1"                    # OPTIONAL — for fixed-string secret patterns
```

**Out of scope for this crate** (binary-level decisions for `heartbit-cli`):
- `mimalloc` or `jemallocator` global allocator — typically 5–15% on tokio-heavy workloads. Future PR against `heartbit-cli`.
- `tokio-uring` — would require a full async-runtime decision, not a per-PR change.

---

## Notable findings deferred to follow-up audits

- **Cosine similarity SIMD** (`memory/hybrid.rs:39–62`, P-MEM-16) — 4–8× via `nalgebra` or x86 intrinsics. Low impact relative to algorithmic wins (P-MEM-2/3); revisit after BM25 index lands.
- **Box<dyn Future> / RPITIT migration** for trait methods (P-XCUT-11, P-LLM-5) — requires a major trait-level refactor. Defer until profiling shows it's >5% of total runtime; with I/O-bound LLM workloads, almost certainly is not.
- **Embedding batching** (P-MEM-10) — 10–100× HTTP-overhead reduction but requires a queue + flush window architecture decision. Worth its own design issue.
- **Daemon / Restate path** were explicitly out of scope for this `heartbit-core`-only audit.

---

## Phasing recommendation (3 PR cycle)

**Phase 1 — Drop-in workspace sweeps** (~1 PR, all-static, low-risk, big aggregate).
- Workspace `LazyLock<Regex>` adoption (T1)
- Workspace `parking_lot::RwLock` swap on hot sites (T2)
- `rustc-hash` adoption on identified non-adversarial maps (T3)
- `Arc<Vec<ToolDefinition>>`, `Arc<CompletionRequest>`, `Arc<MemoryEntry>` swaps (T4)
- Patch.rs short-circuit + pre-normalize (P-TOOL-5/14)
- `OnceLock` for cache_control JSON, pricing table, refusal patterns
- Permission ruleset cache with mtime invalidation (P-CROSS-16)
- Audit `strip_content` filter-in-place (P-CROSS-7)
- Bash UUID → atomic counter (P-TOOL-10)

Expected aggregate: **30–60 ms / run** + 100s of MB allocs avoided. All tested with existing 2330 unit tests + benches added in task #65.

**Phase 2 — Memory subsystem step-change** (~1 PR, algorithmic, highest leverage).
- BM25 inverted index (P-MEM-2)
- Per-tenant secondary index (P-MEM-6)
- Lazy strength-decay caching (P-MEM-3)
- Recall-result `Arc`/`Cow` return type (P-MEM-5)
- Split read/write lock scope (P-MEM-1 follow-on)
- `SmallVec` for related_ids/source_ids/keywords (P-MEM-13)

Expected: **0.5–2 ms / recall savings** at N=10k entries — the single biggest user-visible win for memory-heavy workloads.

**Phase 3 — SSE / streaming hot path** (~1 PR, lifetime-juggling, careful).
- `&str` slices into SSE buffer or `bytes::Bytes` zero-copy framing (P-LLM-2)
- Delayed `data_lines` join, skip on empty events (P-LLM-14)
- Reuse `Vec<u8>` buffer for JSON-RPC stdio serialisation (P-MCP-9)
- Tool definition serialisation cached at construction (P-LLM-16)

Expected: per-chunk savings that compound on long streamed responses (10s–100s of KB throughput).

---

## Audit cadence

This is the second-pass audit (security cycle was first). Recommended cadence:
1. Land Phase 1 + benches first (low-risk, validates the harness).
2. Re-run criterion before/after each phase to confirm hypothesised wins.
3. Profile a representative production trace (`tokio-console` or `perf record`) after Phase 3 to identify any remaining surprise hotspots.

---

**Audit date**: 2026-05-06
**Auditors**: 7 parallel static-analysis subagents + manual synthesis
**Sub-reports**: 7 (linked above)
**Total findings**: 113 (15C / 39H / 38M / 21L)
**REJECTED candidates that would weaken security**: 30+
**Recommended phasing**: 3 PRs (drop-in sweeps → memory step-change → streaming zero-copy)
