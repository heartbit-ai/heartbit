# B5a — Loose-Ends Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the four preserved orphan files (or discard the dead one), delete the umbrella's sensor/telegram modules now that the satellite crates are integrated, run a verification matrix that catches feature-gating regressions, and push the accumulated commits to origin.

**Architecture:** Five sequential commits + a verification block + push. Each commit independently green; bisect-friendly. No new types or features — pure cleanup and consolidation.

**Tech Stack:** Rust 2024 edition, `cargo`, `git`, optionally Docker (for live Postgres test) and `curl` (for daemon smoke test).

**Spec:** `docs/superpowers/specs/2026-05-02-b5a-loose-ends-cleanup-design.md`

**Pre-flight assumption:** Today's HEAD on `main` is `0ca24cf` (B5a spec) on top of `df43145` (satellite integration merge). 24+ commits ahead of `origin/main`.

---

## File structure

| Action | Path | Notes |
|---|---|---|
| Move | `crates/heartbit/src/agent/evaluator.rs` → `crates/heartbit-core/src/agent/evaluator.rs` | `git mv` preserves blame |
| Move | `crates/heartbit/src/agent/handoff.rs` → `crates/heartbit-core/src/agent/handoff.rs` | `git mv` |
| Move | `crates/heartbit/src/tool/handoff.rs` → `crates/heartbit-core/src/tool/handoff.rs` | `git mv` |
| Modify | `crates/heartbit-core/src/agent/mod.rs` | add `pub mod evaluator; pub mod handoff;` |
| Modify | `crates/heartbit-core/src/tool/mod.rs` | add `pub mod handoff;` |
| Modify | `crates/heartbit-core/src/lib.rs` | re-export public types |
| Delete | `crates/heartbit/src/daemon/runtime.rs` | dead orphan; `git rm` |
| Delete | `crates/heartbit/src/sensor/` (entire directory) | superseded by `heartbit-sensors` |
| Delete | `crates/heartbit/src/channel/telegram/` (entire directory) | superseded by `heartbit-telegram` |
| Modify | `crates/heartbit/src/lib.rs` | drop `pub mod sensor;` + `pub use sensor::*;` block, drop `pub mod telegram;` ref + telegram re-exports |
| Modify | `crates/heartbit/src/channel/mod.rs` | drop `pub mod telegram;` |
| Modify | `crates/heartbit/Cargo.toml` | drop `sensor` and `telegram` features + 6 optional deps (`quick-xml`, `sha2`, `hex`, `teloxide`, plus `hmac`/`subtle` if present) |

Heartbit-cli does NOT reference `heartbit::sensor::*` or `heartbit::channel::telegram::*` paths today (verified via grep). No CLI rewires required for commits 3/4.

---

## Pre-flight (Task 0)

- [ ] **Step 0.1: Confirm clean baseline**

```bash
cd /home/pleclech/projects/heartbit
git status   # working tree clean, on main, 0ca24cf at HEAD
git log --oneline -3
cargo fmt -- --check
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo test --workspace --lib --offline 2>&1 | grep "test result" | tail
```

Expected:
- `working tree clean`
- HEAD: `0ca24cf docs: B5a loose-ends cleanup design spec`
- fmt + clippy clean
- 4129 lib tests pass (1125 + 2270 + 621 + 113)

If any check fails, **stop** and investigate — the round assumes a clean baseline.

- [ ] **Step 0.2: Capture baseline counts for later comparison**

```bash
cargo test --workspace --lib --offline 2>&1 | grep -c "^test " || echo "save total for comparison"
```

Note the per-crate counts: heartbit 1125, heartbit-core 2270, heartbit-cli 621, heartbit-telegram 113. These should not regress.

---

## Task 1: Move EvaluatorOptimizerAgent + HandoffRunner + HandoffTool to heartbit-core

**Files:**
- Move: `crates/heartbit/src/agent/evaluator.rs` → `crates/heartbit-core/src/agent/evaluator.rs`
- Move: `crates/heartbit/src/agent/handoff.rs` → `crates/heartbit-core/src/agent/handoff.rs`
- Move: `crates/heartbit/src/tool/handoff.rs` → `crates/heartbit-core/src/tool/handoff.rs`
- Modify: `crates/heartbit-core/src/agent/mod.rs`
- Modify: `crates/heartbit-core/src/tool/mod.rs`
- Modify: `crates/heartbit-core/src/lib.rs`

- [ ] **Step 1.1: Inspect public surface of each file**

```bash
grep -nE "^pub (struct|enum|fn|trait|type)" crates/heartbit/src/agent/evaluator.rs
grep -nE "^pub (struct|enum|fn|trait|type)" crates/heartbit/src/agent/handoff.rs
grep -nE "^pub (struct|enum|fn|trait|type)" crates/heartbit/src/tool/handoff.rs
```

Note the public types — these are the items to re-export from `heartbit-core/src/lib.rs`.

- [ ] **Step 1.2: Move evaluator.rs**

```bash
git mv crates/heartbit/src/agent/evaluator.rs crates/heartbit-core/src/agent/evaluator.rs
```

- [ ] **Step 1.3: Move agent/handoff.rs**

```bash
git mv crates/heartbit/src/agent/handoff.rs crates/heartbit-core/src/agent/handoff.rs
```

- [ ] **Step 1.4: Move tool/handoff.rs**

```bash
git mv crates/heartbit/src/tool/handoff.rs crates/heartbit-core/src/tool/handoff.rs
```

- [ ] **Step 1.5: Add module declarations**

Edit `crates/heartbit-core/src/agent/mod.rs` — add (alphabetically, after `dag` before `events`):

```rust
pub mod evaluator;
pub mod handoff;
```

Edit `crates/heartbit-core/src/tool/mod.rs` — add (alphabetically, after `builtins` before `mcp`):

```rust
pub mod handoff;
```

- [ ] **Step 1.6: Add re-exports in heartbit-core/src/lib.rs**

Find the existing `pub use agent::workflow::{...};` block in `crates/heartbit-core/src/lib.rs`. Add the new re-exports nearby (alphabetical placement). Example:

```rust
pub use agent::evaluator::{EvaluatorOptimizerAgent, EvaluatorOptimizerAgentBuilder};
pub use agent::handoff::HandoffRunner;
pub use tool::handoff::{HandoffContextMode, HandoffTarget, HandoffTool, parse_handoff_sentinel};
```

If `EvaluatorOptimizerAgentBuilder` is not actually defined (verify in Step 1.1's output), drop the `Builder` half. Same for any names that turn out to differ from the spec — match what the file actually exports.

- [ ] **Step 1.7: Build + test heartbit-core**

```bash
cargo build -p heartbit-core --offline 2>&1 | tail -10
cargo test -p heartbit-core --lib --offline 2>&1 | grep "test result" | tail
```

Expected: clean build, test count `2270 + N` where N is the number of inline tests inside the moved files.

- [ ] **Step 1.8: Build + test workspace**

```bash
cargo build --workspace --offline 2>&1 | tail -5
cargo clippy --workspace --all-targets --offline -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib --offline 2>&1 | grep "test result" | tail
```

Expected: workspace builds, clippy clean, heartbit (umbrella) test count drops by 0 (the moved files weren't wired into umbrella's `lib.rs` — they were inert orphans), heartbit-core test count grows by N.

- [ ] **Step 1.9: Commit**

```bash
cargo fmt
git add crates/heartbit-core/src/agent/evaluator.rs \
        crates/heartbit-core/src/agent/handoff.rs \
        crates/heartbit-core/src/agent/mod.rs \
        crates/heartbit-core/src/tool/handoff.rs \
        crates/heartbit-core/src/tool/mod.rs \
        crates/heartbit-core/src/lib.rs \
        crates/heartbit/src/agent/evaluator.rs \
        crates/heartbit/src/agent/handoff.rs \
        crates/heartbit/src/tool/handoff.rs

git commit -m "refactor(core): move EvaluatorOptimizerAgent + HandoffRunner into heartbit-core

The three preserved orphan files (1e57828) used crate::error::Error,
crate::llm::LlmProvider, and crate::tool::ToolDefinition — paths that
resolve only inside heartbit-core after the B3 extraction. Move them
into core alongside SequentialAgent, ParallelAgent, LoopAgent which
already live there.

git mv preserves blame. Module declarations added to agent/mod.rs and
tool/mod.rs. Public types re-exported from heartbit_core::lib.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Discard dead daemon/runtime.rs orphan

**Files:**
- Delete: `crates/heartbit/src/daemon/runtime.rs`

- [ ] **Step 2.1: Confirm no references**

```bash
grep -rn "daemon::runtime[^_]\|use crate::daemon::runtime;" crates/ --include="*.rs"
```

Expected: empty result. The file is self-contained.

- [ ] **Step 2.2: Delete**

```bash
git rm crates/heartbit/src/daemon/runtime.rs
```

- [ ] **Step 2.3: Build + test**

```bash
cargo build --workspace --offline 2>&1 | tail -5
cargo test --workspace --lib --offline 2>&1 | grep "test result" | tail
```

Expected: clean build, test count unchanged from end of Task 1.

- [ ] **Step 2.4: Commit**

```bash
git commit -m "chore: drop dead daemon/runtime.rs orphan

1472 lines, no external imports, types duplicated in the live
daemon/runtime_types.rs. Preserved by 1e57828 and in
~/heartbit-orphans-2026-05-02.tar.gz if recovery is ever needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Delete umbrella sensor/* in favor of heartbit-sensors

**Files:**
- Delete: `crates/heartbit/src/sensor/` (entire directory)
- Modify: `crates/heartbit/Cargo.toml`
- Modify: `crates/heartbit/src/lib.rs`

- [ ] **Step 3.1: Confirm no external consumers**

```bash
grep -rn "heartbit::sensor::\|use crate::sensor::" crates/ --include="*.rs" | grep -v "crates/heartbit/src/sensor/"
```

Expected: empty result. The umbrella's `sensor` module is self-contained.

- [ ] **Step 3.2: Delete the sensor directory**

```bash
git rm -r crates/heartbit/src/sensor/
```

- [ ] **Step 3.3: Drop pub mod and re-exports from lib.rs**

Edit `crates/heartbit/src/lib.rs`. Remove these lines (line numbers approximate):

```rust
#[cfg(feature = "sensor")]
pub mod sensor;
```

And remove the entire `// --- Sensor re-exports (feature-gated) ---` block (lines ~180-194, ending with `pub use sensor::{Sensor, SensorEvent};`).

- [ ] **Step 3.4: Drop sensor feature + deps from Cargo.toml**

Edit `crates/heartbit/Cargo.toml`. From the `[features]` section, remove the line:

```toml
sensor = ["daemon", "heartbit-core/sensor", "dep:quick-xml", "dep:sha2", "dep:hex"]
```

From the `full` feature list, remove `"sensor"`. So:

```toml
full = ["daemon", "sensor", "restate", "postgres", "a2a", "telegram", "discord", "slack", "vault"]
```

becomes:

```toml
full = ["daemon", "restate", "postgres", "a2a", "telegram", "discord", "slack", "vault"]
```

(Don't remove `"telegram"` yet — that's Task 4.)

From the `[dependencies]` section, remove the optional deps that were sensor-only:

```toml
quick-xml = { workspace = true, optional = true }
sha2 = { workspace = true, optional = true }
hex = { workspace = true, optional = true }
```

If `hmac` and `subtle` are also present and unused, remove them too. Check by grepping `crates/heartbit/src/` for any `quick_xml`, `sha2`, `hex`, `hmac`, `subtle` references after deleting `sensor/`. If a dep is still referenced from a non-sensor module, leave it.

```bash
grep -rn "quick_xml\|sha2\|hmac::\|subtle::" crates/heartbit/src/ --include="*.rs"
```

- [ ] **Step 3.5: Build + test**

```bash
cargo build --workspace --offline 2>&1 | tail -10
cargo build -p heartbit --no-default-features --features daemon --offline 2>&1 | tail -5
cargo build -p heartbit --no-default-features --features full --offline 2>&1 | tail -5
cargo test --workspace --lib --offline 2>&1 | grep "test result" | tail
```

Expected: all builds clean. The umbrella test count drops by however many tests lived inline in `sensor/*.rs` modules (likely zero, since the satellite already owns them).

- [ ] **Step 3.6: Clippy**

```bash
cargo clippy --workspace --all-targets --offline -- -D warnings 2>&1 | tail -5
```

Expected: clean. If `dead_code` warnings surface for previously-sensor-only helpers (like `fnv1a_hash` in `crates/heartbit/src/util.rs` if it's still gated), leave them — they were already gated correctly during the satellite integration round.

- [ ] **Step 3.7: Commit**

```bash
cargo fmt
git add Cargo.lock crates/heartbit/Cargo.toml crates/heartbit/src/lib.rs
git commit -m "refactor: delete umbrella sensor/* in favor of heartbit-sensors

Removes ~14k lines of duplicated sensor pipeline code from the
umbrella. The heartbit-sensors satellite crate (workspace member,
integrated in df43145) is now the canonical source.

Drops the umbrella's 'sensor' feature and 5 sensor-only optional deps
(quick-xml, sha2, hex; hmac and subtle if also unused). The 'full'
feature no longer includes sensor.

Pre-release breaking change: callers using heartbit umbrella with
features = [\"sensor\"] migrate to depending on heartbit-sensors
directly. heartbit-cli already uses neither path (no rewires here).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Delete umbrella channel/telegram/* in favor of heartbit-telegram

**Files:**
- Delete: `crates/heartbit/src/channel/telegram/` (entire directory)
- Modify: `crates/heartbit/Cargo.toml`
- Modify: `crates/heartbit/src/lib.rs`
- Modify: `crates/heartbit/src/channel/mod.rs`

Mirror of Task 3.

- [ ] **Step 4.1: Confirm no external consumers**

```bash
grep -rn "heartbit::channel::telegram\|use crate::channel::telegram" crates/ --include="*.rs" | grep -v "crates/heartbit/src/channel/telegram/"
```

Expected: empty. (heartbit-telegram is the satellite; it doesn't import from the umbrella's old path.)

- [ ] **Step 4.2: Delete the telegram directory**

```bash
git rm -r crates/heartbit/src/channel/telegram/
```

- [ ] **Step 4.3: Drop pub mod and re-exports from lib.rs**

Edit `crates/heartbit/src/lib.rs`. Remove the entire `#[cfg(feature = "telegram")] pub use channel::telegram::{...};` block (around line 116).

If there's a `pub mod telegram;` declaration at the umbrella's `src/lib.rs` level, remove it too. (Most likely it's nested inside `src/channel/mod.rs`; see next step.)

- [ ] **Step 4.4: Drop pub mod telegram from channel/mod.rs**

Edit `crates/heartbit/src/channel/mod.rs`. Remove:

```rust
#[cfg(feature = "telegram")]
pub mod telegram;
```

If it has companion re-exports immediately after, remove those too.

- [ ] **Step 4.5: Drop telegram feature + dep from Cargo.toml**

Edit `crates/heartbit/Cargo.toml`. From `[features]`:

```toml
telegram = ["dep:teloxide"]
```

— remove that line.

From the `full` feature list (still in the form post-Task-3):

```toml
full = ["daemon", "restate", "postgres", "a2a", "telegram", "discord", "slack", "vault"]
```

becomes:

```toml
full = ["daemon", "restate", "postgres", "a2a", "discord", "slack", "vault"]
```

From `[dependencies]`, remove:

```toml
teloxide = { workspace = true, optional = true }
```

(only if no non-telegram code references it — check with `grep -rn "teloxide" crates/heartbit/src/`).

- [ ] **Step 4.6: Build + test**

```bash
cargo build --workspace --offline 2>&1 | tail -5
cargo build -p heartbit --no-default-features --features full --offline 2>&1 | tail -5
cargo test --workspace --lib --offline 2>&1 | grep "test result" | tail
```

Expected: all builds clean. Umbrella test count may drop further by however many tests lived inside `channel/telegram/*.rs` (the satellite reports 113, which originally lived in the umbrella).

- [ ] **Step 4.7: Clippy + fmt**

```bash
cargo clippy --workspace --all-targets --offline -- -D warnings 2>&1 | tail -5
cargo fmt -- --check
```

Expected: clean.

- [ ] **Step 4.8: Commit**

```bash
git add Cargo.lock crates/heartbit/Cargo.toml crates/heartbit/src/lib.rs \
        crates/heartbit/src/channel/mod.rs
git commit -m "refactor: delete umbrella channel/telegram/* in favor of heartbit-telegram

Mirror of the sensor extraction in the previous commit. heartbit-telegram
(workspace member, integrated in df43145) is now the canonical source.

Drops the umbrella's 'telegram' feature and the 'teloxide' optional
dep. The 'full' feature no longer includes telegram.

Pre-release breaking change: callers using heartbit umbrella with
features = [\"telegram\"] migrate to depending on heartbit-telegram
directly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Verification matrix

No commit. Each verification step either passes (continue) or fails (stop and investigate).

- [ ] **Step 5.1: Cross-feature build matrix**

```bash
cargo build -p heartbit --no-default-features --offline
cargo build -p heartbit --no-default-features --features daemon --offline
cargo build -p heartbit --no-default-features --features daemon,postgres --offline
cargo build -p heartbit --no-default-features --features daemon,postgres,restate --offline
cargo build -p heartbit --no-default-features --features full --offline
cargo build --workspace --all-features --offline
```

Each must finish with `Finished` and zero clippy warnings. If any combination fails, the failure is in scope — fix it before proceeding.

- [ ] **Step 5.2: Live Postgres integration test**

If Docker is available locally:

```bash
docker run -d --name heartbit-pg-test -p 5433:5432 \
  -e POSTGRES_PASSWORD=test -e POSTGRES_DB=heartbit_test \
  pgvector/pgvector:pg17

# Wait for ready (max 30s)
for _ in $(seq 1 30); do
  if docker exec heartbit-pg-test pg_isready -U postgres >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

DATABASE_URL=postgres://postgres:test@localhost:5433/heartbit_test \
  cargo test --workspace --all-features -- --ignored 2>&1 | tail -20

docker rm -f heartbit-pg-test
```

Expected: at minimum the audit + memory tenant integration tests pass:
- `audit_entries_by_scope_returns_only_matching_tenant`
- `prune_audit_deletes_old_rows`
- (any others marked `#[ignore = "requires DATABASE_URL"]`)

If Docker is not available:

```bash
echo "[ ] DEFERRED: live Postgres integration test (no local Docker; relies on CI)" \
  >> /tmp/b5a-deferred.log
```

Document the gap in the final commit message of Task 6 if any items got deferred.

- [ ] **Step 5.3: Daemon smoke test**

If `ANTHROPIC_API_KEY` is set in the environment:

```bash
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  cargo run -p heartbit-cli --release --offline -- daemon \
  --config daemon-lite.toml > /tmp/heartbit-daemon-smoke.log 2>&1 &
DAEMON_PID=$!

# Wait up to 30s for healthz
for _ in $(seq 1 30); do
  if curl -fsS http://localhost:8080/healthz >/dev/null 2>&1; then
    echo "healthz OK"
    break
  fi
  sleep 1
done

# Submit a tiny task
curl -fsS -X POST http://localhost:8080/v1/execute \
  -H "content-type: application/json" \
  -d '{"task": "Reply with the single word: hi"}' \
  --max-time 30 || echo "submit failed"

# Graceful shutdown — SIGINT, never SIGKILL/pkill
kill -INT $DAEMON_PID
wait $DAEMON_PID 2>/dev/null || true
tail -20 /tmp/heartbit-daemon-smoke.log
```

Expected: healthz returns 200, the submit returns a task id, the daemon log shows a successful task execution. The daemon exits cleanly on SIGINT.

If `ANTHROPIC_API_KEY` is not set:

```bash
echo "[ ] DEFERRED: daemon smoke test (no ANTHROPIC_API_KEY)" >> /tmp/b5a-deferred.log
```

- [ ] **Step 5.4: Final quality gate**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo test --workspace --offline 2>&1 | grep "test result" | tail
```

All three must pass cleanly. Test totals should match expectations: heartbit ~1125-N, heartbit-core ~2270+M, heartbit-cli 621, heartbit-telegram 113. Where N is the umbrella's tests lost when sensor/* and channel/telegram/* were deleted, and M is the tests gained in heartbit-core when the orphan files moved.

---

## Task 6: Drop stale stash + push to origin

- [ ] **Step 6.1: Verify stash is what we think it is**

```bash
git stash list
git stash show stash@{0} --stat | tail -5
```

Expected: `stash@{0}` description starts with `WIP on main: 1b52afd Release v2026.306.5` (from before B4 started). The diff stat shows ~25k lines deleted across `heartbit/src/sensor/*` and `heartbit/src/channel/telegram/*` — the original (incomplete) extraction WIP.

- [ ] **Step 6.2: Drop the stash**

```bash
git stash drop stash@{0}
git stash list
```

Expected: empty stash list.

- [ ] **Step 6.3: Confirm origin reachable + ahead-count**

```bash
git fetch origin main
git log origin/main..HEAD --oneline | wc -l
```

Expected: 30+ commits ahead (B4 + satellite + B5a). The exact number depends on how many commits B5a produced.

- [ ] **Step 6.4: Push**

```bash
git push origin main
```

Expected: clean push, branch updated. Watch the gitleaks pre-push hook output — it should report "no leaks found" (consistent with the per-commit hook output we've seen all round).

- [ ] **Step 6.5: Confirm remote in sync**

```bash
git log origin/main..HEAD --oneline | wc -l
```

Expected: `0`.

- [ ] **Step 6.6: No commit needed**

This task does git operations, not source changes. There's no "Step X: commit" because the stash drop is local-only and the push publishes existing commits.

---

## Final verification

- [ ] **Step F.1: One-shot quality gate replay**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo test --workspace --lib --offline 2>&1 | grep "test result" | tail
git status
git log origin/main..HEAD --oneline | wc -l
```

Expected: fmt clean, clippy clean, lib tests pass, working tree clean, `0` ahead of origin.

- [ ] **Step F.2: Spec coverage check**

Walk the spec's Goals (G1–G5):

- G1 (live framework features in heartbit-core) — Task 1 ✓
- G2 (umbrella drops sensor/telegram features) — Tasks 3 + 4 ✓
- G3 (verification matrix) — Task 5 ✓
- G4 (repo hygiene: stash + push) — Task 6 ✓
- G5 (no regressions) — covered by per-task quality gates + Step F.1

If any goal has no task, add it before declaring done.

---

## Self-review notes

**1. Spec coverage** — every Goal maps to at least one task; F.2 is the explicit cross-reference.

**2. No placeholders** — every step shows the actual command or actual code change. Commit messages are written verbatim. The only "if-then" branches are in the verification matrix where Docker / API key may not be available; those branches each have explicit fallback commands (write to a deferred-log file).

**3. Type consistency** — the only types touched are public re-exports in Step 1.6. The exact list (`EvaluatorOptimizerAgent`, `HandoffRunner`, `HandoffTool`, etc.) is verified against the actual public surface in Step 1.1 before being baked in. If reality differs from the spec's guess at the type list, Step 1.1's grep output is authoritative.

**4. Sequencing dependencies**
- Task 1 → Task 4: independent (Task 1 touches `agent/` and `tool/`; Tasks 3-4 touch `sensor/` and `channel/telegram/`).
- Task 3 → Task 4: independent (different feature flags, different module paths). Tasks 3 and 4 update the same `Cargo.toml` and `lib.rs` but in different sections; if the engineer batches them, fine — but separate commits are cleaner for bisect.
- Task 5 (verification) → must run after all source-modifying commits land.
- Task 6 (push) → must run last.
