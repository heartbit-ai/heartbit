# heartbit-core Release Audit — 2026-06-07

**Release manager decision. Branch:** `feat/tui-streaming-markdown` (60 commits ahead of `main`). **Subject:** heartbit-core crates.io release.

---

## 1. VERDICT: GO-WITH-CONDITIONS

The engineering is release-ready — the full quality gate is green (fmt clean, clippy `-D warnings` zero, **5206 tests pass / 0 fail**, heartbit-core green in isolation at 2888+8), the session's central security hypothesis (a malicious tool result forging a gate marker the model trusts) is **refuted with evidence**, and no confirmed code defect, panic-on-untrusted-input, or API leak survived adversarial verification. But the release is **physically blocked by one fact**: the workspace version still reads `2026.507.4`, which is **already git-tagged and already published to crates.io** — and crates.io versions are immutable — so `cargo publish` and `git tag` are both rejected until a one-line version bump. That is the single most important reason this is conditional rather than a clean GO: it is not a code problem, it is a mechanical precondition. Every surviving blocker/high is a process step satisfiable **now with zero code change**.

---

## 2. Confirmed blockers & highs (post-verification severity)

| Sev | ID | What | Where | Fix |
|-----|----|------|-------|-----|
| **BLOCKER** | `version-not-bumped` | Workspace version `2026.507.4` is already BOTH tagged (`v2026.507.4`) and published to crates.io (as **`heartbit-core`**); the 60-commit branch has zero version-bump (`git log main..feat -- Cargo.toml` empty). crates.io versions are immutable and the tag exists → publish + tag both rejected. | `Cargo.toml:11` (`[workspace.package]` header is line 9; the `version = "2026.507.4"` line is **11**), inherited via `version.workspace = true` by all crates. | Bump one line to a fresh calendar version (scheme `year.{M}{DD}.patch` → ~`2026.607.1`; **maintainer picks exact**). No code change. |
| **HIGH** | `changelog-stale` | CHANGELOG documents none of the release's headline features (deep_research, advisor mode, run_workflow resume, worktree isolation, intent-mode router, plan/act/ask gates, TUI splash/streaming markdown). `[Unreleased]` (CHANGELOG.md:120) is **misordered below** the released `## [2026.507.4]` heading (line 7) and describes already-shipped ghost X-tools. main is 162 commits past the last tag; branch 222. | `CHANGELOG.md` (no entry; `[Unreleased]` stale/misplaced). | Write a new dated entry **above** the 2026.507.4 heading covering the 60 branch commits; clear the stale `[Unreleased]`. No code change. Not a blocker: the enforced gate is fmt+clippy+test only; zero runtime impact. |

No other finding survives verification at blocker or high severity for a release decision.

---

## 3. Per-dimension status (one line each)

- **Quality Gate (fmt + clippy -D warnings + test):** RELEASE-READY — all 3 stages green; 5206 pass / 0 fail, deterministic across 2 runs; 107 ignored are all infra/live-gated.
- **Public API Surface Stability:** RELEASE-READY — `#![deny(missing_docs)]` compiles, `cargo doc` 0 errors, new types leak-free & constructible; only additive/non-breaking warts remain.
- **Security regressions:** RELEASE-READY — no blocker/high; gate markers not injectable, new shell-out (worktree) and network paths hardened, remediation fixes intact; 2 mediums are best-effort/bounded.
- **Robustness / panic-safety:** RELEASE-READY — no genuine panic-on-untrusted-input; new gate code uses `PoisonError::into_inner` consistently; tool panics caught by JoinSet → error output, never crash.
- **Tech-debt & coherence (gate family):** RELEASABLE — loop coherent, 52 tests pass; carries 3 mediums of latent behavioral debt (wish-detector overfire, mode-blind stop-gates, one-shot ask-gate) — none block.
- **Release Process Readiness:** NOT-READY-AS-IS but every blocker is a now-satisfiable process step — drives the verdict and the checklist below.

---

## 4. Release checklist (execute in order; only if proceeding)

There is **no publish automation** (no release-plz / cargo-release / publish CI), so the crates.io step is manual and the runbook is **dual-channel** (crates.io for the lib, GitHub Release for the CLI binary).

1. **Bump version** — edit `Cargo.toml:11` off `2026.507.4` to the new calendar version (maintainer picks; ~`2026.607.1`). [clears BLOCKER `version-not-bumped`]
2. **Write CHANGELOG** — add a dated entry above the `## [2026.507.4]` heading covering the 60 branch commits (ideally the ~222 since the last tag); clear the misordered `[Unreleased]` section. [clears HIGH `changelog-stale`]
3. **Fast-forward merge** — `git checkout main && git merge --ff-only feat/tui-streaming-markdown`. Verified clean: `merge-base == main HEAD (43d5cb1)`, branch is a strict descendant, `merge-tree` shows zero conflicts. [clears low `branch-unmerged`]
4. **Re-run the full gate on `main`** — `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm`. Must be green on the merged tip before tagging.
5. **Tag & push** — `git tag vX.Y.Z && git push origin main --tags`. The `v*` tag triggers `release.yml` → builds & uploads the **heartbit-cli binary** (GitHub Release only; CLI/TUI/umbrella are NOT on crates.io).
6. **Publish the crate manually** — `cargo publish -p heartbit-core` (heartbit-core has no intra-workspace path deps → independently publishable; the umbrella `heartbit` is intentionally not published, so no `version =` backfill on sibling path deps is needed for a core-only release).

---

## 5. What was checked and found CLEAN (scope you can trust)

- **Gate, all 3 stages:** fmt clean; clippy `-D warnings` **0 warnings** (forced 15.44s recompile to defeat stale cache); `--workspace --exclude mini-crm` **5206 pass / 0 fail**, reproduced byte-identical across 2 runs; heartbit-core green in isolation (2888 unit + 8 doctest).
- **Security (the central question):** gate markers (ask/act/plan/mode/study) are **runner-generated from the agent's own output, never parsed out of tool results** — a forged marker in a tool result unlocks nothing (read-only enforcement, plan gate, ScopeGuard all key off router-set state). Router classifier output bounded to a 4-variant enum (no path/cmd/file sink). New shell-out (`worktree.rs` git via `.args()`, name sanitized — no traversal/arg-injection). New websearch + deep_research reuse the hardened HTTP client (SafeDnsResolver Strict, capped bodies — no SSRF/OOM regression). Remediation fixes F-MCP-2 (tool-name shadowing), F-FS-2 (env policy) intact.
- **Panic-safety:** every non-test `.unwrap()` is a test misclassification or infallible (regex-literal compile, guarded split, validated-UTF8 slice); new gate RwLocks handle poisoning via `PoisonError::into_inner`; a panicking tool degrades to an error output, never aborts the session.
- **API surface:** new public types (GoalSlot, DelegationNudge, SetGoalTool/SetScopeTool/SessionHandoffTool, RequestRouter family) verified leak-free & constructible; `cargo doc` default profile 0 errors.
- **Gate-family coherence:** run-loop gate ordering coherent (no dead conditions, single re-arm point); `classify_query` fully deleted (not orphaned); fix-commits are hardening/TOCTOU/clippy, not redo-churn; the TOCTOU barrier test is a real wired unit test.
- **Release metadata:** license `MIT OR Apache-2.0`, README present, repository/keywords/categories set; CI gate runs fmt+clippy+test on push/PR.

**Out of scope by mandate:** `mini-crm` (WIP, excluded from the gate); live/infra-gated tests (Kafka, PostgreSQL, OpenRouter, live Chrome) — 107 intentionally ignored, not hiding logic bugs.

---

## 6. Refuted / downgraded by the adversarial pass

- **`router-family-not-reexported`** — audit said **high**, downgraded to **low**. Real ergonomic wart (router family reachable only via `agent::router::*`; `RoutedMode` return type un-nameable at root) but the fix is additive/non-breaking on a calendar-versioned crate. Does NOT block release; not in the table.
- **`branch-unmerged-but-clean-fast-forward`** — audit said **high**, downgraded to **low**. A single deterministic conflict-free `git merge --ff-only` with zero merge risk. Precondition for tagging (checklist step 3), not a substantive issue.
- **`changelog-stale`** — audit said **blocker**, downgraded to **high** (above). No enforced gate requires changelog currency; zero runtime impact. The "60 branch commits never touch CHANGELOG" sub-claim is the normal state of a feature branch (changelog lands at release-cut), and the "ghost X-tools already shipped" sub-claim was unverified — both dropped from the severity rationale.
- **`version-not-bumped`** — **confirmed blocker** (stands), with two evidence corrections folded in: the version is at `Cargo.toml:11` not 9, and the published crate is `heartbit-core` (the name `heartbit` 404s on crates.io). The collision against heartbit-core is real, so the blocker holds.

**Not promoted:** the 3 unverified mediums in the gate-family dimension (wish-detector over-fire re-introducing planning on a one-line edit; mode-blind ask/act gates misfiring in Answer/Study; one-shot ask-gate with no persistent backstop) and the 2 security mediums (handoff redaction LLM-instructed-only; barrier tools skip pre_tool guardrail) are real debt worth fixing soon but are **bounded, recoverable, and non-blocking** — none reaches the table.
