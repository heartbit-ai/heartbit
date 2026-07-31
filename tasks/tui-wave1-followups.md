# TUI Wave 1 — carried-forward items

Wave 1 (`feat/tui-wave1`, 18 commits off `main` at `37bb8a5`) shipped with the
workspace gate green: `fmt` 0 · `clippy --workspace --all-targets -D warnings` 0 ·
`cargo test --workspace` **5517 passed / 0 failed** (baseline 5403, +114 tests).

Specs: `docs/superpowers/specs/2026-06-26-tui-sota-upgrade-design.md` (framework),
`…-tui-wave1-table-stakes-design.md` (wave). Plan:
`docs/superpowers/plans/2026-06-26-tui-wave1-table-stakes.md`.

This file is what the SDD ledger carried that git history does not: known defects
we chose not to fix, verification we could not perform, and the queue for Wave 1.5.

---

## 1. Known defects, deliberately not fixed

**Deferred respawn is re-deferred indefinitely by the queue drain.** *(Important —
upgraded from a minor at final review.)* At a turn boundary the reducer calls
`flush_pending_respawn()` (pushes `Effect::RespawnAgent`) and then
`drain_one_queued()`, which sets `running = true`. Effects are executed *after*
`update()` returns, so the `RespawnAgent` handler reads that `running == true`,
re-defers, and the following `SendInput` reaches the **old** agent. A mid-run
`/model`, `/effort` or `/codex` plus one queued message therefore runs the drained
turn on the old model — while `queue_respawn()` told the user "applies when the
current turn ends". Self-corrects once the queue empties.

Do **not** apply the obvious fix (skip the drain when a respawn was flushed): after
the channel swap `running` is false and no further turn-idle event arrives, so the
held queue is stranded until the user types. The fix belongs in `main.rs` and must
distinguish "the agent is working a turn" from "the reducer just started a turn
whose `SendInput` is still in this effect batch".

**Mid-turn `/learn` can commit at the wrong boundary.** Pre-existing.
`self.learning` is a single slot consumed by the next turn-idle — the in-flight
turn's, not `/learn`'s — so the distillation can be skipped entirely. The digest
guard prevents a *wrong* commit, not a *missed* one. The comment at
`Msg::LearnReady` now states this honestly.

**`Msg::RunCompleted` has no epoch.** Unlike `AgentExited(u64)`, it cannot be
gated in `admit_agent_msg`, so a stale `RunCompleted` still drives `running=false`,
`finalize_active`, the `learning` commit — and now also a queue drain. Giving it an
epoch fixes the whole handler's staleness, not just the drain. `msg.rs` + `main.rs`.

**`Effect::SeedFromHandoff` bypasses the input queue.** It sends inline into
`input_tx` while a turn runs. Wave 1's "one choke point" (decision D-4) is true for
the seven `Effect::SendInput` sites it touched; this is an eighth path.

**`cells.rs` `Cell::Agent → markdown::render` is dead in production** and ignores
the configured `syntax_theme`; `ui.rs` routes agent cells through the cache
instead. Its tests assert against the dead path, so the two renderings will drift.

---

## 2. Verification we could not perform

**No agent had a TTY**, so nothing terminal-dependent in this wave has been
observed. Spec §7 carries the six-step manual script; run it in Kitty, Ghostty,
WezTerm or foot (**not** tmux/screen — they swallow the CSI-u push).

Where it is most likely to bite, per the final review:

- **Step 6 (panic → clean shell).** The hook wrap is installed *after* the enable
  block, and `restore_terminal_modes()` runs on any thread's panic — so a recovered
  *agent-thread* panic silently pops kitty/paste/focus/mouse for the rest of the
  session.
- **Step 1 (Shift+Enter).** Rests entirely on the unconditional CSI-u push with no
  probe and no feedback; a silent no-op under tmux/screen.
- **Step 5.** The "persisted to \<path\>" notice fires *before* core attempts
  `save()`, so it is optimistic and cannot report the failure it exists to report.

**Notifications (T1.9) clear neither tier.** The `RunCompleted`/`AgentExited`
notify hooks are unfalsifiable by the suite (deleting one leaves it green), the
focus wiring is unit-tested only, and notify appears in **none** of §7's six manual
steps. It ships with no verification path — add one before trusting it.

**Truecolor.** Syntax highlighting emits `Color::Rgb`; behaviour on 256-colour
terminals and against a light background is unobserved.

---

## 3. Wave 1.5 (next)

**LSP diagnostics** — deferred from Wave 1 by decision D-1. Wiring
`.lsp_manager()` as-is ships a guaranteed 30 s stall with zero diagnostics
(`lsp/server.rs:195-239`: an empty `publishDiagnostics` advances `current_version`
and re-waits) plus a malformed `file://` URI (`server.rs:89`, `:148`). Three core
changes, two of them not optional, and the URI fix also lands on four heartbit-cli
call sites. Needs its own spec.

Also queue here: the respawn/queue-drain fix and the `RunCompleted` epoch (§1).

---

## 4. Deferred minors (triaged "fine to defer" at final review)

*Test quality* — kitty byte-assert compares a literal to itself; the once-only-pop
test uses a local `AtomicBool` and pins the helper, not the wiring; `/effort`
wiring lines have no test; notify hooks unfalsifiable; a Task 4 test near-duplicates
the one it replaced; `MAX_FORMAT_BYTES` early-return untested; formatter tests
depend on GNU `printf`/`head` without a `#[cfg(unix)]` gate.

*Cosmetic / UX* — `RunFailed` prints the drop notice before the cause; `Up`
clobbers an in-progress draft; the queue box clips past `frame_h/3`; a queued
`/goal` shows a `Cell::User` where an immediate one shows nothing; a drained turn
does not reseed the roster; a code block flips theme when it settles out of
streaming; `session.rs` exports hunk headers with a leading space; Esc with a
backlog drops the queue without interrupting while the composer hint still reads
"Esc interrupt".

*Bounded edges* — `git ls-files --others` lacks `-z`, so quoted/non-ASCII untracked
paths are skipped from `/diff`; a commit-less repo yields git's raw "ambiguous
argument 'HEAD'"; the word-diff DP is capped at 1000 pairs; formatter failures are
silent (no tracing) and its stdout read is uncapped while stdin is capped.

*Performance* — the markdown cache is bounded but its sweep is near-inert: every
agent cell is re-touched each frame, and a hit deep-clones its `Vec<Line>`. It buys
the syntect + pulldown parse, not the allocations, and resident memory is ~2× the
transcript. **Strictly better than pre-branch** (which re-parsed *and* re-allocated);
revisit with `Rc<Vec<Line>>` and a borrowed touched-key if long sessions feel slow.

---

## 5. Process notes worth keeping

Three defects came from the **plan**, not the implementations, and each was caught
by an implementer refusing to proceed rather than by a test:

- a mandated `formatted == content → None` branch made the plan's own
  `large_content_does_not_deadlock` test unpassable;
- a `VecDeque<String>` queue leaked Plan mode's invisible directive into the
  transcript — the plan's type was wrong, `QueuedInput { display, wire }` was right;
- two mandated tests asserted mutually exclusive things about the *same* input.

Twice a test carried the right name and could not catch the defect it named: the
syntect invariant fixture had no comment (the exact trigger), and the multi-file
diff fixture used `diff --git` everywhere (the exact form that already worked).
When a test names an invariant, check that its fixture can actually violate it.

Once, an implementer refused a reviewer's prescribed fix with evidence and was
**right**: the proposed one-liner would have traded a cosmetic ordering race for a
duplicate-engine bug.
