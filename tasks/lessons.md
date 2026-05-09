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
