# Performance Audit v2 — heartbit workspace (2026-05-07)

**Scope**: full workspace (heartbit, heartbit-cli, heartbit-sensors, heartbit-telegram, heartbit-gateway) plus a second-order pass on cycle-1 changes in heartbit-core. v1 was heartbit-core-only.

**Predecessor**: v1 audit at `tasks/performance-audit-heartbit-core-2026-05-06.md` (113 findings). Cycle-1 fixes (released as v2026.507.1) delivered:
- `text_query_top10@10k` recall: **19.8 ms → 12.69 ms** (−36%)
- `agent_filter_top10@10k`: **3.20 ms → 2.02 ms** (−37%)
- `sse_parse/feed_16kb_one_shot`: **11.3 µs → 7.0 µs** (−38.5%)

**v2 result**: **65 items** total — 51 new perf findings + 14 bench coverage gaps. Pattern: daemon and edge crates carry the same hotspots v1 fixed in core, none of which were addressed by cycle 1.

---

## Executive summary

| Sub-report | Findings | Critical | High | Medium | Low |
|------------|----------|----------|------|--------|-----|
| [v2 — daemon](perf-audit-v2-daemon.md) | 22 | 2 | 11 | 6 | 3 |
| [v2 — sensors / telegram / gateway / cli](perf-audit-v2-edges.md) | 18 | 3 | 5 | 10 | 0 |
| [v2 — cycle-1 second-order](perf-audit-v2-cycle1-followup.md) | 11 | 0 | 1 | 4 | 6 |
| **Total perf findings** | **51** | **5** | **17** | **20** | **9** |
| [v2 — bench coverage gaps](perf-audit-v2-bench-gaps.md) | 14 | 1 | 4 | 6 | 3 |

**Cycle-1 regressions detected: 0.** All 3669 workspace tests still green; the lock-step invariant on the tokens cache holds end-to-end; SSE parser structure change is internal-only; `parking_lot` lock discipline is verified by the type system.

**Single dominant residual hotspot**: BM25 substring inner loop accounts for ~99% of the 12.69 ms `text@10k` recall (700k substring checks per recall). See "BM25 design decision" below.

---

## v2 themes (each spans ≥2 sub-reports)

### Theme A — Cycle 1 left the daemon and edges untouched

The same `T1`/`T2`/`T3`/`T4` cross-cutting patterns v1 found and cycle 1 fixed in `heartbit-core` are still present in **every other workspace crate**:

| Pattern | core (cycle 1) | daemon | sensors | telegram | gateway |
|---|---|---|---|---|---|
| `std::sync::RwLock` on hot non-await paths | ✅ swapped | ❌ unchanged | ❌ unchanged | ❌ unchanged | ❌ unchanged |
| `Regex::new()` at call site | ✅ LazyLock'd | partial | partial | n/a | partial |
| `format!()` in inner loop | partial | ❌ many sites | ❌ many sites | ❌ many sites | n/a |
| `Vec::new()` then push N times | partial | ❌ unchanged | ❌ unchanged | n/a | partial |
| Per-call JSON `to_vec`/`to_string` without pooling | n/a | ❌ Critical | n/a | n/a | n/a |

The v2 audit found these patterns at **40 sites** across the rest of the workspace (sum of P-V2-DAEMON-* and P-V2-EDGE-* findings that are direct echoes of v1 themes). Most are mechanical search-and-replace fixes already validated in `heartbit-core`.

### Theme B — JSON serialization without pooling on every daemon event

The daemon's per-task `on_event` closure (`crates/heartbit/src/daemon/core.rs:787-799`) calls `serde_json::to_vec(&event)` for every emitted agent event with no buffer reuse. At 50–500 events per task execution, that's **2.5–100 ms of avoidable allocation per task**, dominating the daemon command-dispatch hot path under realistic agent workloads.

This finding (`P-V2-DAEMON-1`, Critical) would have been one of v1's headline items had v1 covered the daemon. It's the largest single opportunity in v2.

### Theme C — Cycle-1 changes opened a measured residual hotspot

The cycle-1 second-order pass confirmed by reasoning (and cited the bench number) what the audit predicted: after Phase 2c eliminated per-entry tokenisation, the residual `12.69 ms text@10k` recall is essentially **all BM25 substring inner loop**. At 10k entries × 5 query tokens × 14 words avg = 700k substring checks per recall × ~18 ns SIMD-optimised `str::contains` ≈ **12.6 ms**, matching the observed bench within rounding.

This is unblocking by design, not engineering — see "BM25 design decision" below.

### Theme D — Bench coverage is thin for the "best in class" claim

14 bench coverage gaps. The most embarrassing: **zero benchmark exercises the agent ReAct turn**. All 6 v1 P-RUNNER findings (`Arc<tool_defs>`, doom-loop hashing, tool-name repair, `recently_used_tools` HashSet, etc.) are validated only by static analysis. A competitor reading the `2026.507.1` release notes would correctly observe that several headline perf claims have no measurement.

---

## Top 10 v2 findings (by expected impact)

| # | ID | Severity | Title | Sub-report | Validated by |
|---|---|---|---|---|---|
| 1 | P-V2-DAEMON-1 | Critical | Per-event JSON serialization without pooling | daemon | needs new bench |
| 2 | P-V2-2ND-8 | High | BM25 substring inner loop dominates 12.6/12.69 ms | cycle-1 | `memory_recall` (already green; needs flamegraph) |
| 3 | P-V2-EDGE-7 | Critical | Telegram bridge: 2 std::sync::RwLock on every message | edges | needs new bench |
| 4 | P-V2-DAEMON-3 | High | `event_channels` std::sync::RwLock contention | daemon | needs new bench |
| 5 | P-V2-EDGE-2 | Critical | Per-event entity Vec clone (sensors triage) | edges | needs new bench |
| 6 | P-V2-DAEMON-2 | Critical | Quadruple `source` String clone per task outcome | daemon | needs new bench |
| 7 | P-V2-DAEMON-10 | Medium | `stats()` / `usage_stats()` O(N) full-scan aggregation | daemon | needs new bench |
| 8 | P-V2-EDGE-4 | High | Sensor `StoryCorrelator` `std::sync::Mutex` on every event | edges | needs new bench |
| 9 | P-V2-DAEMON-4 | High | `tasks_by_state` String key clone per task in stats | daemon | needs new bench |
| 10 | Bench-NEW-1 | Critical | **Agent ReAct turn bench missing** (validates 6 P-RUNNER findings) | bench-gaps | n/a — adds the bench |

**Of 51 new findings, 47 require a bench that doesn't exist yet.** The bench-gaps audit identified the 8 missing harnesses + 6 holes in existing benches.

---

## BM25 design decision (was: deferred from v1)

The v1 audit's headline `P-MEM-2` finding (BM25 inverted index, predicted <2 ms text@10k) has been deferred twice in cycle 1 because both attempts regressed the bench. The cycle-1 second-order pass `P-V2-2ND-8` lays out the actual constraint:

- Current behaviour uses `word.contains(token)` — substring matching at the per-word level.
- Real queries depend on this: `"performance"` matches an entry containing `"performance-critical"`, `"performant"`, etc.
- A simple `HashMap<token, Vec<entry_id>>` inverted index requires **exact-word match**, breaking the existing semantics.

**The blocker is a product decision, not engineering.** Three viable paths:

1. **Hybrid index** (recommended) — exact-word inverted index as a fast path; fall back to full substring scan only for queries containing partial tokens (detected by failing the exact-word lookup). Estimated win: **12.69 → 8–10 ms** for the typical case where queries are full words; preserves substring semantics for partial tokens.
2. **Trigram suffix pre-filter** — index every 3-char substring of every word. Larger index (~3× storage) but rejects non-matching entries at O(N_matches × M) instead of O(N × M × L). Preserves substring semantics in full. Estimated win: **12.69 → 3–5 ms** with no semantic change.
3. **Opt-in exact-word query mode** — add a `MemoryQuery::exact_words: bool` flag; existing semantics unchanged by default; opt-in callers get the inverted-index fast path. Estimated win: **12.69 → 1–3 ms** when opted in.

**Recommendation**: ship (3) first (smallest blast radius, opt-in), then (2) as a follow-up if production queries warrant the storage overhead. Skip (1) — its complexity isn't justified once (3) exists.

This is a one-design-decision, ~200-LOC commit. It belongs in cycle 2 if it gets a green light.

---

## Recommended cycle 2 phasing (5 PRs)

### Phase 4 — daemon `T1+T2+T3` sweep (echoes cycle-1 Phase 1)

Mechanical translation of cycle-1 fixes to the daemon:
- `std::sync::RwLock` / `Mutex` → `parking_lot::*` on every hot non-await site (P-V2-DAEMON-3, …).
- `format!` in hot loops → write to reused buffers (P-V2-DAEMON-1's adjacent finding).
- `HashMap<String, _>` on hot lookup → `FxHashMap` (P-V2-DAEMON-4 uses this).

Estimated effort: ~1 PR, low risk. Bench validation: needs new daemon harness (Bench-NEW-3).

### Phase 5 — daemon JSON-pool + stats refactor

Address P-V2-DAEMON-1 (event serialization buffer pool), P-V2-DAEMON-2 (source clone reduction), and P-V2-DAEMON-10 (rolling stats aggregates instead of full-scan).

Estimated effort: ~1 PR. Bench: harness from Phase 4 plus a daemon stats bench.

### Phase 6 — sensors / telegram T1+T2+T3 sweep

Mechanical translation again — sensors and telegram are smaller scopes but follow the same pattern.

### Phase 7 — bench infrastructure expansion

Land the Critical bench gaps:
- Bench-NEW-1: **Agent ReAct turn** (mock provider). Highest priority — unblocks validation of 6 unvalidated v1 findings.
- Bench-NEW-2: MCP JSON-RPC roundtrip (stdio + HTTP).
- Gap-EX-1: `memory_recall` at N=100k + hybrid mode + graph expansion.

Estimated effort: ~1 PR for the agent ReAct bench (180–240 min per the bench-gaps report); separate PRs for the others.

### Phase 8 — BM25 inverted index (opt-in flag)

Implement the (3) variant from the BM25 design decision section: `MemoryQuery::exact_words: bool` flag, inverted index maintained in lock-step alongside the tokens cache.

Estimated effort: ~1 PR, ~200 LOC, validated by a new bench variant `text_query_top10/exact_words@10k`.

### Phase 9 (optional) — trigram suffix pre-filter

Only if production telemetry shows the substring fallback is hit often enough to justify the storage cost.

---

## Security-regression matrix (verbatim from v1)

The 30+ rejected "obvious wins" from v1 remain rejected. **Zero v2 findings require re-opening any closed F-* finding**. All four sub-audits independently verified this against the same discriminator list.

The five tempting "wins" most likely to surface in cycle 2 (carry forward):

| Tempting "win" | Re-opens | Status |
|---|---|---|
| Daemon caches `event_channels` across tenants | F-MEM-5 | REJECTED — must remain per-tenant |
| Telegram bridge shares `pending` HashMap across users | F-AUTH-5 | REJECTED — nonce binding required |
| Sensor pipeline shares one `reqwest::Client` across `IpPolicy` contexts | F-NET-2 | REJECTED — DNS rebinding |
| Lifting `MCP_STDIO_LINE_MAX_BYTES` for daemon throughput | F-MCP-4 | REJECTED — DoS bound |
| Defaulting `AuditMode::Full` for daemon debugging | F-AUTH-6 | REJECTED — privacy regression |

---

## Cycle-1 regression check: clean

The cycle-1 second-order audit (`tasks/perf-audit-v2-cycle1-followup.md`) verified all 8 cycle-1 commits against:
- correctness regressions (0 found)
- residual hotspots created (1 — BM25 substring loop, see above)
- code clarity / future-proofing (4 medium findings, mostly comments)
- security boundary preservation (verified — `parking_lot` lock discipline holds, audit allow-list semantics identical, SSE buffer cap accounting unchanged)

The two correctness-adjacent items worth tracking:
- **P-V2-2ND-2**: `add_link()` doesn't refresh the tokens cache. Currently safe because `add_link` only mutates `related_ids` (not part of `EntryTokens`), but adding a tokenised field in the future without updating `add_link` would silently corrupt the cache. Mitigate with a code comment + invariant doc.
- **P-V2-2ND-5**: Phase 2b's three explicit `drop()` calls in the recall function are brittle — a future refactor that misses the discipline would fail to compile in obvious cases but might subtly change behaviour in subtle cases. Mitigate by extracting graph-expansion into a helper function with a clean borrow boundary.

Neither is urgent; both are tracked as Low/Medium follow-ups.

---

## Notes on profile data

I attempted to capture flamegraph data via `samply`, but the kernel's `perf_event_paranoid` setting requires sudo to lower. Profile data is therefore **absent from this audit cycle**; all v2 findings are static-analysis with bench numbers as ground truth where available.

The cycle-1 second-order pass derives the BM25 substring-loop hotspot by **calculation**: 700k substring checks × 18 ns/check ≈ 12.6 ms ≈ the observed `12.69 ms` bench number. This is consistent with the prediction but unconfirmed by symbol-level profiling. Confirming by flamegraph is the first 5-min task in cycle 2.

---

## Sub-reports

- [`tasks/perf-audit-v2-daemon.md`](perf-audit-v2-daemon.md) — 22 findings (24 KB)
- [`tasks/perf-audit-v2-edges.md`](perf-audit-v2-edges.md) — 18 findings (26 KB)
- [`tasks/perf-audit-v2-cycle1-followup.md`](perf-audit-v2-cycle1-followup.md) — 11 findings (21 KB)
- [`tasks/perf-audit-v2-bench-gaps.md`](perf-audit-v2-bench-gaps.md) — 14 gaps (35 KB)

---

**Audit date**: 2026-05-07
**Auditors**: 4 parallel static-analysis subagents + manual synthesis
**Total v2 items**: 65 (51 perf findings + 14 bench gaps)
**Cycle-1 regressions detected**: 0
**Recommended phasing**: 5 PRs (daemon T1/T2/T3 sweep → daemon JSON-pool → sensors/telegram → bench infra → BM25 inverted index opt-in)
