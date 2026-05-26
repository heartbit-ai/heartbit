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
