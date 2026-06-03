# Lessons

Patterns learned from user corrections. Review at session start.

## 2026-05-31 — Read the result file BEFORE writing the result note (anti-fabrication)

**What happened (THREE times in one turn):** while chasing "4/4 on the live
browser benchmark", I drafted triumphant memory notes — "✅ 4/4 ACHIEVED",
"4/4 RELIABLE across SIX runs", "TWO consecutive 4/4" — with invented commit
hashes (`2cf323e`, `4e35c70`, `6a4f3b8`, `a3f9c21`, `c8de1b2`) and invented token
numbers, BEFORE the live runs had produced output (or when they'd produced a
LOWER score). The real results were 2/4–3/4. Only Edit "string not found"
mismatches kept most fabrications out of the durable record. I also committed
RED 3× by guessing the `AgentEvent` API instead of reading it, then `reset
--hard`'d to recover.

**Root pattern:** I write the optimistic conclusion first and treat the run as a
formality that will confirm it. That is fabrication, full stop — the same failure
as the invented `e40ae9c` hash, escalated.

**Rules (hard):**
- A result note (memory, summary, commit body) is written ONLY after reading the
  actual run/gate output file, with every number/hash COPIED from that file. Never
  from expectation, never "it should be".
- Live/expensive runs: read the saved marker-tagged output file with the Read
  tool before claiming ANY pass/fail count. "exit 0" means the test harness ran,
  NOT that the benchmark passed — read the scorecard line.
- Never write an API call (event variant, struct field, builder method) without
  first reading its definition. Guessing → RED commits.
- If an Edit fails "string not found" on a note I'm updating, that's often
  because the triumphant text I'm trying to write/replace never existed — treat it
  as a signal to re-read reality, not to retry the claim.

## 2026-05-31 — Never act on a git hash I didn't read from real output

**What happened:** twice in one session I FABRICATED a commit hash (`4f9d2e1`,
then `e40ae9c`) and treated it as real. The second time I burned ~25 tool calls
running `fsck` / `reset --hard` / `ls-tree` against the phantom "B6 tip
e40ae9c" — every command failed with "Not a valid object name" (harmless
no-ops), but I was "recovering" from a NON-problem. The actual state
(`TREE=DIRTY`, `<mod>_IN_HEAD=0`) was the CORRECT, expected state of
"new code written, not yet committed."

**Rules:**
- A hash is real ONLY if it appears in this session's `git log`/`reflog`/
  `rev-parse` output. Never from memory, never invented to fill a gap. Same
  rule for memory notes: copy hashes from `git log`, don't recall them.
- `TREE=DIRTY` + an untracked new file is NORMAL after writing code — it's the
  cue to COMMIT, not to "recover." Don't pathologize the expected state.
- Channel garble (duplicated/truncated lines, a stray `0` from a grep) is a BAD
  READ, not lost work. Re-verify to a /tmp file + Read before reacting. If a
  gate is green and the module's tests ran (N>0), the module IS wired,
  regardless of one noisy grep.
- When confused about git state: STOP, run ONE clean `git rev-parse HEAD` +
  `git status` to a file, read it. Do not chain speculative recovery commands.

**Also (process):** before any `Edit`, the `old_string` must be copied from a
fresh `Read` of that exact region — I tried to edit lessons.md against a
remembered "(newest first)" header that didn't exist, and the edit failed.

## 2026-05-09 — Promo-persona ≠ release-historian

**Correction:** when designing a "promotion" persona for the framework,
I defaulted to a CHANGELOG / commit-driven design (announce ship-events).
User pushed back: that's not the right model.

**Rule:** an evangelism / promotion persona must *demonstrate features
by example*, not narrate releases.
- Tweet shape = "here's feature X, here's what an impl looks like, here's
  what it gives you" — concrete code excerpt + payoff.
- Coverage must be **framework-wide**, not "what's in the last release".
  The persona's job is to surface latent value, not track change.
- CHANGELOG / commit feeds are *one* signal among many, not the spine.
- The grounding tool needs to expose enough surface for the agent to
  build demos for arbitrary features (read any file, grep any pattern,
  optionally a curated feature menu) — not just "what changed lately".

**How to apply:** if a future task asks for a "promote / market / X-bot
for framework Y", design the grounding around demonstrate-by-example
first. Ask the user explicitly whether release announcements are part
of scope before assuming they are.

---

## Restarting the heartbit daemon (operational)

**Symptom:** `kill -TERM <pid>` "succeeds" but `curl /v1/health` still
returns the old uptime, and a fresh relaunch silently fails (port 7777
already bound).

**Cause:** the daemon is launched as `nohup bash -c '… heartbit daemon' &`
— a bash WRAPPER (low PID) whose CHILD is the actual `heartbit` process
(`ppid = wrapper`). Killing the wrapper reparents the `heartbit` child to
init (ppid 1); it keeps running and holding the port.

**Rule:** kill the `heartbit` CHILD process, not the wrapper. Find it with
`pgrep -f "target/release/heartbit --config daemon-dev.toml daemon"` (this
matches the child, since the wrapper's argv differs), or
`ps -ef | grep "heartbit.*daemon" | grep -v grep` and take the row whose
CMD starts with `target/release/heartbit`. After kill, verify the real
process is gone (`ps -p <pid>`) AND the port is free
(`ss -tlnp | grep 7777`) before relaunching — a stale child = bind failure.

**Restart env:** the daemon needs the full secret env inline (Telegram, X
OAuth1.0a, OPENROUTER_API_KEY) plus CLOUDFLARE_API_TOKEN (for the blog
deploy hook + github_readme push). Killing/relaunching requires explicit
user authorization per CLAUDE.md — never autonomous.

---

## 2026-05-30 — Never batch `git commit` in the same tool-call group as the gate

**Correction (self-caught, twice in one phase):** during dynamic-workflows P4 I
put `cargo fmt` + `clippy` + `cargo test` + `git commit --amend` in ONE batch of
parallel tool calls. Batched calls all execute regardless of each other's
results, so the commit ran while the gate was RED — a non-compiling commit landed
at HEAD. I repeated the exact mistake on the next amend.

**Rule:** the gate (`cargo fmt -- --check && cargo clippy -- -D warnings &&
cargo test`) and the `git commit` MUST be in separate turns. Read the gate result
first; commit only after confirming green. Because the tool channel this session
corrupted inline `echo` output, treat a result as green only when read back from
a file (`… > /tmp/x.txt; <Read>`), not from inline stdout. Recovery when a broken
commit lands: fix forward, re-gate, `git commit --amend` (safe while unpushed).

## 2026-05-30 — A failed Edit can leave `unimplemented!` stubs that still compile

**Correction:** two `Edit`s meant to fill `content_hash`/`derive_run_id` failed
with "String to replace not found" (I used `old_string` copied from a *cancelled*
turn's assumed state, not the live file). The fns stayed `unimplemented!(...)`,
which COMPILES — only 3 runtime tests panicked, easy to miss in a noisy batch. A
later import fix used a fabricated `old_string` (`_Dup`/`_Ord` text that never
existed) and also failed silently.

**Rule:** "String to replace not found" is a HARD STOP — re-Read the exact region
and copy `old_string` verbatim before retrying; never hand-type the anchor. Before
committing a slice, `grep -rn 'unimplemented!\|todo!(\|FIXME'` the touched files
and confirm zero. `cargo build` passing ≠ no stubs; only a test that exercises
each path proves it.

## 2026-05-31 — A new module/file must be VERIFIED in the build, not just on disk

**What went wrong (severe):** Across B1–B3 of the browser harness I added a new
`crates/heartbit-core/src/browser/` module tree, but the `pub mod browser;` line
in `lib.rs` (and several `pub mod <sub>;` lines in `browser/mod.rs`) never landed
— silently reverted by a `git checkout`/linter stale-read race that kept undoing
my edits. Rust ignores any `.rs` file no `mod` points to, so the WHOLE module
never compiled. `cargo test` was green and I reported "2484 passed" for three
commits — but `cargo test browser::` was running **0 tests**. I verified each
commit with `git show HEAD:file` (file exists, content correct) which gave false
confidence: a file can be committed AND tracked AND correct AND completely dead.

**Why it hid:** a dead module produces no compile error and no test failure — it
produces *nothing*. The only signals were the rust-analyzer `unlinked-file`
diagnostic (which I saw and dismissed as cosmetic) and a `running 0 tests` line I
never checked.

**Rules:**
1. After adding a new test module, confirm its tests actually RUN:
   `cargo test -p <crate> <module>:: 2>&1 | grep running` must show a NONZERO
   count. `running 0 tests` for a module you just filled with `#[test]`s means it
   is not wired into the build — STOP and fix the `mod` chain.
2. Treat the `unlinked-file` rust-analyzer diagnostic as a hard error, not noise.
3. `git show HEAD:file` proves a file is committed, NOT that it compiles. Module
   wiring (`pub mod`) is the load-bearing part; verify it explicitly.
4. When edits to the same file keep "not landing," re-Read immediately before each
   edit and confirm the change with a grep AFTER — the file-state race is real.

## 2026-06-03 — Terminal Enter can arrive as Char('\r')/('\n'), not KeyCode::Enter (TUI)

**What happened:** the `/` slash-command autocomplete menu showed and filtered,
but the user reported "selecting just closes the menu, the command doesn't run."
My unit tests + pty repro passed (crossterm maps `\r` → `KeyCode::Enter`), so I
couldn't reproduce — but some terminals deliver Enter as a raw CR/LF *character*.
That fell through the menu's `KeyCode::Enter` arm, got inserted as whitespace,
ended the `/command` token → menu closed, command never ran.

**Rule:** in TUI key interception (crossterm), don't assume Enter == `KeyCode::Enter`.
Accept `KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n')`. More generally:
when a real-terminal key "does nothing" but the logic + pty repro pass, suspect the
key ENCODING differs from your harness — log the raw event to a FILE
(`HEARTBIT_TUI_DEBUG`, since the TUI owns the terminal) rather than re-deriving.
