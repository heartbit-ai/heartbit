# TUI `/stats` Styled Card — Design

**Date:** 2026-06-06
**Status:** Approved (user chose "Carte stylée dans le transcript" with full preview)
**Scope:** `crates/heartbit-tui` (`trace_stats.rs`, `msg.rs`, `app.rs`, `cells.rs`, `main.rs`, `session.rs` export arm)

## Decision

`/stats` renders a **styled card in the transcript** instead of a raw
preformatted block: colored section labels, human units, a context-growth
sparkline, red error highlighting, cache hit-rate. Stays in the transcript →
scrollback and `/export` keep working.

Approved visual (colors: header magenta bold · errors red · friction green
when none / yellow otherwise · sparkline + labels dim):

```
▎ stats — session 6a2473a4 · 1m47s · 21 llm calls

  tokens     in 86.7k · out 6.9k · cache 41%
  context    4.4k → 7.1k /call (max 13.3k)  ▂▃▄▅▆█▆▅
  latency    llm p50 1.8s p95 19.8s · ttft 1.6s/16.4s
  runs       8 ok · 0 failed
  friction   aucune
  approvals  2 (0 refusées) · instantané

  tool          ×    err   p50     p95
  run_workflow  1    —     71.1s   71.1s
  webfetch      11   2 ⚠   0.4s    3.0s
  websearch     4    —     1.1s    1.1s
  write         1    —     2ms     2ms
```

(Display text stays English like the rest of the UI: "none", "denied",
"instant" — the preview's French words were illustrative.)

## Architecture

The edge stops pre-rendering: `Msg::StatsReady(Ok(...))` carries the
**struct**, and a new `Cell::Stats` renders it — testable in `cells.rs`,
exported via the existing plain renderer.

- `trace_stats.rs`:
  - `TraceStats` gains `pub total_cache_read_tokens: u64` (summed from
    `llm_response.usage.cache_read_input_tokens` — data already in the trace,
    never aggregated) and derives `Clone, Serialize, Deserialize` (session
    persistence of the cell) on top of its existing derives.
  - `render()` (plain string) is KEPT — used by the `/export` markdown arm.
- `msg.rs`: `Msg::StatsReady(Result<(String, Box<TraceStats>), String>)` — the
  `String` is the source label (session id or "last"-resolved id) the edge
  already knows; `Box` keeps the Msg small.
- `main.rs:1138`: send `(label, Box::new(trace_stats::compute(file)))` instead
  of `.render()`.
- `app.rs`: `Msg::StatsReady(Ok((label, stats)))` → push
  `Cell::Stats { label, stats }`.
- `cells.rs`:
  - `Cell::Stats { label: String, stats: trace_stats::TraceStats }`.
  - `to_lines()` arm building the card:
    - header `▎ stats — session {label} · {duration} · {llm_calls} llm calls`
      (magenta bold marker+title, dim meta);
    - `tokens` row: `fmt_tokens` units + `cache N%` =
      `total_cache_read_tokens * 100 / max(total_input_tokens, 1)`;
    - `context` row: first → last `/call` (max) + sparkline from
      `turn_input_tokens` downsampled to ≤ 24 buckets, glyphs `▁▂▃▄▅▆▇█`
      scaled to the max (skip the row when < 2 samples);
    - `latency` row: p50/p95 llm + ttft via `fmt_ms`;
    - `runs` row: `{run_completed} ok · {run_failed} failed` (failed red
      when > 0);
    - `friction` row: "none" (green) or the non-zero counters listed
      (yellow), e.g. `retries 2 · doom-loops 1 · interrupts 1`;
    - `approvals` row: `{approvals} ({denials} denied) · {mean}` — "instant"
      when mean < 100ms;
    - tools table: name / ×count / err (`—` dim, or `N ⚠` red) / p50 / p95,
      `{:<14}`-aligned, header row dim.
  - Pure helpers + tests: `fmt_tokens(u64)` (`982`, `4.4k`, `86.7k`, `1.2M`),
    `fmt_ms(u64)` (`2ms`, `0.4s`, `71.1s`, `1m47s`), `sparkline(&[u64], usize)`.
- `session.rs` export arm: `Cell::Stats { label, stats }` → heading
  `**stats — {label}**` + the plain `stats.render()` in a fenced block.

## Edge cases

- Empty/tiny trace: rows render with zeros; sparkline row omitted (< 2 calls).
- cache 0% renders as `cache 0%` (still shown — absence of caching is signal).
- Old saved sessions deserialize fine (no `Cell::Stats` in them); new sessions
  with `Cell::Stats` round-trip via serde.
- `/analyze` is untouched (it stages the raw trace, not the rendered table).

## Testing

- Helpers: `fmt_tokens`, `fmt_ms`, `sparkline` unit tests (boundaries: 999→
  `999`, 1000→`1.0k`, 59_999ms→`60.0s`, 60_000→`1m0s`; sparkline scales and
  downsamples).
- `Cell::Stats.to_lines()`: contains the section labels, the tool rows, `⚠`
  on a tool with errors, cache %, and the sparkline glyphs for a growth curve.
- Reducer: `StatsReady(Ok(...))` pushes a `Cell::Stats` (update the existing
  test that asserts a `Cell::Agent` table).
- Export: a `Cell::Stats` exports with the fenced plain table.
- Live (pty): `/stats <bridged-research-trace-id>` → settled frame shows
  `▎ stats`, tool rows, sparkline glyphs; `/export` written file contains the
  fenced table.

## Non-goals

Full-screen dashboard, per-agent breakdown rows, configurable sections,
French display text.
