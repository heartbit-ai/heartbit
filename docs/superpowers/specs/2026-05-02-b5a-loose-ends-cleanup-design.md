# B5a — Loose-Ends Cleanup: Wire Orphans, Delete Duplicates, Verify

**Date:** 2026-05-02
**Status:** Design — pending user approval before implementation plan
**Scope:** `crates/heartbit-core/src/{agent,tool}/`, `crates/heartbit/src/{lib.rs,sensor,channel/telegram,daemon}/`, `crates/heartbit/Cargo.toml`, `crates/heartbit-cli/src/`, repo-level git stash + verification matrix
**Estimated effort:** ~1 day, executed as 5 small independently-green commits + verification + push.
**Public API breakage:** Pre-release breaking changes only. Removes the umbrella's `sensor` and `telegram` features (both opt-in, never default). Adds `heartbit-sensors` and `heartbit-telegram` as the canonical entry points (already in workspace post-satellite-merge). heartbit-core is not yet on crates.io, so the `evaluator.rs`/`handoff.rs` relocations are zero-risk.

## Background

The B4 multi-tenant hardening round (merged via `54f6256`) and the satellite-crate integration round (merged via `df43145`) left a cluster of loose ends that are individually small but collectively block release confidence:

1. **4 unwired orphan files committed but inert.** Three of them (`crates/heartbit/src/agent/evaluator.rs`, `crates/heartbit/src/agent/handoff.rs`, `crates/heartbit/src/tool/handoff.rs`) use `crate::error::Error`, `crate::llm::*`, and `crate::tool::*` — paths that resolve only inside `heartbit-core` after the B3 extraction. The fourth (`crates/heartbit/src/daemon/runtime.rs`) is dead code: 1472 lines, no external imports, types duplicated in the live `daemon/runtime_types.rs`. They were preserved by `1e57828` so the work isn't lost; this round wires the live ones into core and discards the dead one.

2. **Umbrella vs satellite duplication.** The umbrella's `heartbit/src/sensor/*` and `heartbit/src/channel/telegram/*` modules (gated by `feature = "sensor"` and `feature = "telegram"`) duplicate the new `heartbit-sensors` and `heartbit-telegram` satellite crates. Long-running coexistence drifts. The earlier stalled extraction work in `stash@{0}` already mapped the migration; we apply it on top of the current state.

3. **Verification gaps.** No live Postgres integration test ran — the `#[ignore]`-gated tests against the new tenant columns + `prune_audit` DELETE were never exercised against a real DB. No daemon smoke test confirms the runtime actually boots and runs a task end-to-end. No cross-feature build matrix — the B4 daemon-without-postgres regression we already fixed proves feature combos matter.

4. **Stale state.** `stash@{0}` on main holds the now-superseded original extraction WIP (~25k lines deleted in the stash; superseded by what just merged). 24+ commits on local main are unpublished. `.claude/` is untracked but harmless.

These five clusters are the entire B5a scope. B5b (failure-mode hardening) and Release prep are tracked separately.

## Goals

1. **Live framework features land in `heartbit-core`.** `EvaluatorOptimizerAgent`, `HandoffRunner`, and `HandoffTool` join the workflow-agent family alongside `SequentialAgent` / `ParallelAgent` / `LoopAgent` already in core. Use paths resolve naturally with no rewrite.
2. **Umbrella becomes a thin wrapper for sensor/telegram.** The umbrella drops the `sensor` and `telegram` features entirely. Consumers wanting either capability `cargo add heartbit-sensors` or `cargo add heartbit-telegram` directly (`heartbit-cli` does this in this round; downstream users follow).
3. **Verification matrix establishes the post-B4/satellite baseline.** A reproducible cross-feature build script catches feature-gating regressions. Live Postgres test verifies the migration + scoped read end-to-end. Daemon smoke test confirms the runtime boots.
4. **Repository hygiene.** Stale stash dropped. Local main's commits pushed to `origin/main`.
5. **No regressions to merged work.** All tests still pass after every commit. Quality gates remain clean (fmt, clippy `-D warnings`, full lib test suite).

## Non-Goals

- **B5b failure-mode hardening.** Idempotency keys, per-tenant context-overflow accounting, structured retry policy with circuit breakers — separate round.
- **Release prep.** Version bump, CHANGELOG promotion, `cargo publish heartbit-core`, GitHub release tag, DNS for docs.heartbit.ai — gates on B5b being done. Separate round.
- **Refactoring inside `heartbit-sensors` or `heartbit-telegram`.** They compile against the current API; further cleanup (e.g., applying the project's clippy rules to the satellite test files) is its own round.
- **Touching `heartbit-cloud`.** That's a separate repo. Its consumption of the heartbit umbrella may need updates once the umbrella drops `sensor` / `telegram` features; that's the cloud repo's PR, not this round's.
- **Documentation reorganization.** The B4 multi-tenant recipe, the user docs, and CHANGELOG B4 entries stand. This round adds the verification commands as developer-facing, not user-facing, docs.

## Design

### Architecture

Five sequential commits, each independently green. Verification runs after the last commit; push happens last.

```
┌─────────────────────────────────────────────────────────────┐
│ Commit 1: Move evaluator + handoff into heartbit-core       │
│   crates/heartbit-core/src/agent/{evaluator,handoff}.rs     │
│   crates/heartbit-core/src/tool/handoff.rs                  │
│   crates/heartbit-core/src/{agent,tool}/mod.rs (mod decls)  │
│   git rm crates/heartbit/src/agent/{evaluator,handoff}.rs   │
│   git rm crates/heartbit/src/tool/handoff.rs                │
│                                                              │
│ Commit 2: Discard dead daemon/runtime.rs                    │
│   git rm crates/heartbit/src/daemon/runtime.rs              │
│                                                              │
│ Commit 3: Delete umbrella sensor/*; rewire CLI              │
│   git rm -r crates/heartbit/src/sensor/                     │
│   crates/heartbit/Cargo.toml: drop sensor feature + deps    │
│   crates/heartbit/src/lib.rs: drop pub mod sensor           │
│   crates/heartbit-cli/Cargo.toml: depend on heartbit-sensors│
│   heartbit-cli/src/daemon/*: heartbit::sensor::* →          │
│                              heartbit_sensors::*            │
│                                                              │
│ Commit 4: Delete umbrella channel/telegram/*; rewire CLI    │
│   Same pattern as Commit 3 for telegram.                    │
│                                                              │
│ Commit 5: Drop stale stash + push                           │
│   git stash drop stash@{0}                                  │
│   git push origin main                                      │
└─────────────────────────────────────────────────────────────┘

Verification (between Commit 4 and Commit 5):
  - Cross-feature build matrix script
  - Live Postgres integration test
  - Daemon smoke test
```

**No new types.** Five existing crates touched. Net deletions: ~25,000 lines (umbrella sensor/telegram modules, runtime.rs orphan). Net additions: ~1,200 lines (evaluator + handoff moved to core, with mod declarations and re-exports).

### Component 1: Move `evaluator.rs` + `agent/handoff.rs` + `tool/handoff.rs` to `heartbit-core`

The three files currently live in `heartbit/src/{agent,tool}/` (preserved by commit `1e57828`) but reference `crate::error::Error`, `crate::llm::LlmProvider`, `crate::tool::{Tool, ToolDefinition}`, `super::AgentRunner`, `super::AgentOutput`. After B3 those types live in `heartbit-core`. Moving the files into core resolves all paths without rewrites.

Files:

- Create: `crates/heartbit-core/src/agent/evaluator.rs` — `git mv` from umbrella
- Create: `crates/heartbit-core/src/agent/handoff.rs` — `git mv` from umbrella
- Create: `crates/heartbit-core/src/tool/handoff.rs` — `git mv` from umbrella
- Modify: `crates/heartbit-core/src/agent/mod.rs` — add `pub mod evaluator; pub mod handoff;`
- Modify: `crates/heartbit-core/src/tool/mod.rs` — add `pub mod handoff;`
- Modify: `crates/heartbit-core/src/lib.rs` — re-export public types: `pub use agent::evaluator::EvaluatorOptimizerAgent; pub use agent::handoff::HandoffRunner; pub use tool::handoff::{HandoffTool, HandoffTarget, HandoffContextMode, parse_handoff_sentinel};` (mirror the existing pattern for `SequentialAgent`, `ParallelAgent`, etc.).
- (No umbrella changes — the orphan files were not declared in the umbrella's `lib.rs`, so deleting them only removes inert files.)

Verification: `cargo build -p heartbit-core --offline && cargo test -p heartbit-core --lib --offline`. Tests count unchanged (no new tests added; existing tests in the moved files come along).

If the moved files have inline `#[cfg(test)] mod tests` (likely — they were authored as full features), those tests come with them. Verify by running `cargo test -p heartbit-core --lib evaluator handoff 2>&1 | tail`.

### Component 2: Discard dead `daemon/runtime.rs`

The orphan `crates/heartbit/src/daemon/runtime.rs` (1472 lines) defines `RuntimeMessageRole`, `RuntimeMessage`, `RuntimeRequest`, `RuntimeAgentConfig`, `RuntimeAdvancedConfig`, `RuntimeProviderType`, `RuntimeProviderConfig`, `RuntimeMcpServer`, `RuntimeGuardrailConfig`, `RuntimeMemoryConfig`, `RuntimeResponse`, `RuntimeSseEvent`, `RuntimeConfig`, `MemoryStoreType`, `RuntimeMemoryStoreConfig`, `McpConnectionCache`, and the function `runtime_response_from_output`.

The live `crates/heartbit/src/daemon/runtime_types.rs` (1299 lines) is the active source of truth. It overlaps on the basic types (RuntimeProviderType, RuntimeMcpServer, RuntimeProviderConfig, RuntimeAdvancedConfig, RuntimeAgentConfig, RuntimeGuardrailConfig, RuntimeMemoryConfig, RuntimeRequest, RuntimeResponse, RuntimeSseEvent) and has a richer surface (RuntimeSubAgentConfig, RuntimeOrchestratorConfig, RuntimeSpawnConfig, RuntimeWorkflow*, EdgeTransform, EdgeConditionPattern, RuntimeScorerConfig, RuntimeEvalRequest, RuntimeEvalResponse, RuntimeEvalSseEvent, RuntimeTwitterCredentials).

I verified `daemon::runtime` (singular, no `_types` suffix) is not referenced anywhere in the workspace via:

```bash
grep -rn "daemon::runtime[^_]\|use crate::daemon::runtime;" crates/ --include="*.rs"
```

Empty result. The unique types in `runtime.rs` (`RuntimeConfig`, `MemoryStoreType`, `RuntimeMemoryStoreConfig`, `McpConnectionCache`, `runtime_response_from_output`) are referenced only within `runtime.rs` itself. They look like prototype types from a partial implementation that was never integrated.

Files:

- Delete: `crates/heartbit/src/daemon/runtime.rs`

Verification: `cargo build --workspace --offline`. No symbol resolution failures expected.

If a reviewer flags any of the unique types as actually wanted, the preservation tarball at `~/heartbit-orphans-2026-05-02.tar.gz` and the `1e57828` commit both retain them.

### Component 3: Delete umbrella `heartbit/src/sensor/*`; rewire CLI

Currently:
- `crates/heartbit/src/sensor/{compression,manager,metrics,perception,routing,sources/{audio,image,jmap,mcp,rss,weather,webhook},stories,triage/{audio,context,email,image,rss,structured,webhook}}.rs` — gated by `#[cfg(feature = "sensor")]` in `heartbit/src/lib.rs:87-88`.
- `crates/heartbit/Cargo.toml` declares the `sensor` feature and the `quick-xml`/`hmac`/`sha2`/`hex`/`subtle` optional deps that gate-pull when `sensor` is enabled.
- `crates/heartbit-cli/src/daemon/*.rs` (and possibly other CLI files) reference `heartbit::sensor::*` types under `#[cfg(feature = "sensor")]`.

After this commit:
- `crates/heartbit/src/sensor/` is gone (`git rm -r`).
- `crates/heartbit/src/lib.rs` no longer declares `pub mod sensor;`.
- `crates/heartbit/Cargo.toml` drops the `sensor` feature definition + the 5 sensor-only deps. `full = ["daemon", "sensor", "restate", ...]` becomes `full = ["daemon", "restate", ...]`.
- `crates/heartbit-cli/Cargo.toml` adds `heartbit-sensors = { path = "../heartbit-sensors" }` (no `optional`; CLI always pulls the satellite when daemon is enabled).
- `crates/heartbit-cli/src/daemon/*.rs` rewrites `heartbit::sensor::*` → `heartbit_sensors::*` and drops the `cfg(feature = "sensor")` gates.

Files:

- Delete: `crates/heartbit/src/sensor/` (entire directory)
- Modify: `crates/heartbit/Cargo.toml` (drop feature + deps)
- Modify: `crates/heartbit/src/lib.rs` (drop `pub mod sensor;`)
- Modify: `crates/heartbit-cli/Cargo.toml` (add heartbit-sensors)
- Modify: `crates/heartbit-cli/src/daemon/*.rs` (rewrite imports)

The `stash@{0}` shows the historical attempt at this commit; we replicate its approach with a fresh diff against the post-B4 state.

### Component 4: Delete umbrella `channel/telegram/*`; rewire CLI

Mirror of Component 3 for telegram:

- Delete: `crates/heartbit/src/channel/telegram/`
- Modify: `crates/heartbit/Cargo.toml` (drop `telegram` feature + `teloxide` dep). `full` no longer lists `telegram`.
- Modify: `crates/heartbit/src/channel/mod.rs` (drop `pub mod telegram;`).
- Modify: `crates/heartbit-cli/Cargo.toml` (add heartbit-telegram).
- Modify: `crates/heartbit-cli/src/daemon/*.rs` (rewrite `heartbit::channel::telegram::*` → `heartbit_telegram::*`).

### Component 5: Drop stale `stash@{0}` + push to origin

`stash@{0}` on main holds the WIP from before B4 — the original satellite-crate extraction attempt. Its content is now fully superseded by the merged B4 round (`54f6256`) plus the satellite-integration round (`df43145`). Keeping it around invites accidental re-application.

Operations:

```bash
git stash drop stash@{0}
git push origin main
```

After this, `origin/main` matches local main (no commits pending publication).

### Verification matrix

Run after Commit 4, before Commit 5.

**Cross-feature build matrix** — script tests these combinations:

```bash
cargo build -p heartbit --no-default-features
cargo build -p heartbit --no-default-features --features daemon
cargo build -p heartbit --no-default-features --features daemon,postgres
cargo build -p heartbit --no-default-features --features daemon,postgres,restate
cargo build -p heartbit --no-default-features --features full
cargo build --workspace --all-features
```

Each must succeed with zero clippy warnings. Captures regressions like the B4-Task-7 daemon-without-postgres bug.

**Live Postgres integration test:**

```bash
# Start a local Postgres (one-shot, dev only)
docker run -d --name heartbit-pg-test -p 5433:5432 \
  -e POSTGRES_PASSWORD=test -e POSTGRES_DB=heartbit_test \
  pgvector/pgvector:pg17

# Wait for ready
until docker exec heartbit-pg-test pg_isready -U postgres; do sleep 1; done

# Run the #[ignore]-gated integration tests
DATABASE_URL=postgres://postgres:test@localhost:5433/heartbit_test \
  cargo test --workspace --all-features -- --ignored

# Cleanup
docker rm -f heartbit-pg-test
```

If Docker isn't available locally, document the gap and skip — CI on the next push will exercise these against the project's CI Postgres service. Documented as a known limitation, not a blocker.

**Daemon smoke test:**

```bash
ANTHROPIC_API_KEY=$YOUR_KEY \
  cargo run -p heartbit-cli --release -- daemon --config daemon-lite.toml &
DAEMON_PID=$!
sleep 5

# Verify it's up
curl -fsS http://localhost:8080/healthz
echo "healthz OK"

# Submit a tiny task via /v1/execute (or /v1/tasks/execute, depending on the route)
curl -fsS -X POST http://localhost:8080/v1/execute \
  -H "content-type: application/json" \
  -d '{"task": "Say hi in one word."}' || echo "submit failed"

# Cleanup — request graceful shutdown, NOT kill
curl -X POST http://localhost:8080/admin/shutdown 2>/dev/null || true
wait $DAEMON_PID
```

If the daemon doesn't expose a graceful shutdown endpoint, the smoke test ends with `kill -INT $DAEMON_PID` (SIGINT, which the cancellation token handles). Per CLAUDE.md, never `pkill` a running server: this kills only the specific process we started.

If `ANTHROPIC_API_KEY` isn't available locally, the smoke test is skipped with a documented gap. The submit step (which exercises the full task pipeline) is the only part that requires the key; the healthz part doesn't.

## Test plan

No new unit tests in this round. The verification matrix is the test plan:

- Per-commit: `cargo build --workspace`, `cargo test --workspace --lib`, `cargo clippy -- -D warnings`, `cargo fmt -- --check`. All four must pass before each commit lands.
- Post-Commit-4: cross-feature matrix + live Postgres + daemon smoke (above).
- Pre-Commit-5: confirm origin reachable (`git ls-remote origin HEAD`) and local is ahead in the expected direction (`git log origin/main..HEAD --oneline | wc -l`).

## Sequencing

| # | Commit / step | Notes |
|---|---|---|
| 1 | `refactor: move EvaluatorOptimizerAgent + handoff into heartbit-core` | Use `git mv` to preserve blame. |
| 2 | `chore: drop dead daemon/runtime.rs orphan` | Single `git rm`. |
| 3 | `refactor: delete umbrella sensor/* in favor of heartbit-sensors` | Largest diff in the round (~14k lines deleted). |
| 4 | `refactor: delete umbrella channel/telegram/* in favor of heartbit-telegram` | Mirror of #3 for telegram. |
| — | Verification matrix runs | No commit unless something fails. |
| 5 | `chore: drop stale stash + push to origin` | The `git stash drop` is local-only, not staged. Push lands the round + B4 + satellite work upstream. |

## Risks

1. **CLI rewrites in commits 3 + 4 are wider than they look.** `heartbit-cli/src/daemon/*.rs` may have many `heartbit::sensor::*` and `heartbit::channel::telegram::*` references. Mitigation: do the rewrite mechanically, run `cargo build` between batches, lean on the compiler's "did you mean" hints when a path is wrong.

2. **`Cargo.lock` will churn.** Dropping `quick-xml`, `hmac`, `sha2`, `hex`, `subtle`, and `teloxide` from the umbrella's optional deps shrinks the lockfile. The satellite crates already pull their own copies (no net change to actual transitive surface). Acceptable.

3. **`heartbit-cloud` may consume `heartbit::sensor::*` directly.** The cloud repo is separate; this round breaks any such consumption. Mitigation: out-of-scope per the non-goals; the cloud repo's PR follows. Document in the commit message of #3.

4. **Live Postgres test may not run if no DB is available.** Acceptable degraded path: the in-memory tests still cover the trait contract; the Postgres path is separately exercised in CI on push. Documented gap.

5. **Daemon smoke test may not run without `ANTHROPIC_API_KEY`.** Acceptable degraded path: the healthz check still validates that the daemon boots and binds. The submit-task part is skipped.

6. **Push to origin publishes 30+ commits.** Includes the entire B4 round and the satellite integration. If anyone is watching `origin/main` for a "tracking branch up-to-date" signal, they'll see a large jump. Not a correctness risk — just a notification.

## Out-of-Scope (deferred)

- **B5b — failure-mode hardening.** Idempotency keys, context-overflow accounting, retry circuit breakers. Separate round.
- **Release prep.** Version bump, CHANGELOG promotion, `cargo publish heartbit-core`, GitHub release tag, DNS for docs.heartbit.ai. Gates on B5b. Separate round.
- **Touching `heartbit-cloud`.** Downstream PR.
- **Documentation reorganization.** B4 docs stand.
