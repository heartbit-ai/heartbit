# Lessons

Patterns learned from user corrections. Review at session start.

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
