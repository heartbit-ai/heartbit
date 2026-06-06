# TUI Splash Screen — Design

**Date:** 2026-06-06
**Status:** Approved (form/dismissal/opt-out chosen by user; design approved "oui go")
**Scope:** `crates/heartbit-tui` only (new `splash.rs` + small touches to `app.rs`, `ui.rs`, `config.rs`, `main.rs`)

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Form | **Full-screen splash with a beating heart** — block-art heart (2 pulses), `heartbit` block lettering, version + model line |
| Dismissal | **Auto after ~1.5s OR any key** — never blocks the impatient, never interrogates the contemplative |
| Opt-out | **`splash = true` in tui.toml** (default on; the `prompt_caching` config pattern) |

## Architecture

Splash is **App state reduced by existing messages** — no new threads, no timing
machinery beyond the 120ms `Msg::Tick` that already drives the spinner.

- `app.splash: Option<u8>` — tick counter. `Some(0)` at startup when enabled;
  `None` = no splash (disabled, dismissed, or finished).
- Rendered as a full-frame overlay in `view()`; while active, **modal rendering
  is skipped** (the no-key modal appears at dissolution, not over the heart).
- Pure-art module `splash.rs` exposes `splash_lines(tick, model) -> Vec<Line>`.

## Components

### `splash.rs` (new, ~100 lines)
- Two pre-drawn heart frames (half-block art): LARGE (7 rows) and SMALL (5 rows,
  padded to 7 so the layout never shifts).
- `heartbit` lettering (2 rows, half-block letters), version line
  (`v{CARGO_PKG_VERSION} · {model}`), dim hint line ("any key").
- `splash_lines(tick: u8, model: &str) -> Vec<Line<'static>>` — pure: picks the
  heart frame + color intensity from the tick. Beat rhythm over 13 ticks
  (~1.56s): ticks 0-3 LARGE bright (magenta bold), 4-5 SMALL dim (red),
  6-9 LARGE bright, 10-12 SMALL dim — two beats, then auto-dismiss.

### `app.rs`
- Field `pub splash: Option<u8>` (default `None` in `App::new`).
- `Msg::Tick`: if `Some(t)` → `t+1`; at `>= SPLASH_TICKS` (13) → `None`.
- `Msg::Key(_)` while `Some(_)`: set `None` and **consume the key** (return
  before composer/modal handling — composer stays empty).
- All other messages reduce normally underneath (startup notices, MCP results
  accumulate in history and appear at dissolution).

### `ui.rs`
- At the top of `view()`: if `app.splash == Some(t)`, render
  `splash_lines(t, &app.model)` centered both axes over the full frame area and
  **return** (no transcript, no status line, no composer, no modal).
- Small-terminal fallback: if `area.height < 16 || area.width < 44`, render a
  single centered line `♥ heartbit v{version}` instead of the art (state still
  auto-dismisses on schedule).

### `config.rs`
- `TuiConfig.splash: bool`, `#[serde(default = "default_true")]` — same pattern
  as `prompt_caching`. Persisted by `save()` like the other flags.

### `main.rs`
- After `App::new(...)`: `app.splash = cfg.splash.then_some(0);`

## Edge cases

- `splash = false` → `splash` stays `None`; rendering and input identical to today.
- Resize during splash → next tick re-renders centered for the new area.
- Mouse events during splash: ignored (only key events dismiss).
- No trace events — the splash is pure cosmetics; the trace still starts with
  `session_started`.

## Testing

- **Reducer** (`app.rs`): auto-dismiss after `SPLASH_TICKS` ticks; any key
  dismisses AND is consumed (composer empty); `None` config path never arms it.
- **Art** (`splash.rs`): `splash_lines` returns LARGE frame at tick 0, SMALL at
  tick 4, LARGE at tick 6 (frame identity via a marker row); every frame has the
  same line count (no layout shift).
- **Render** (`ui.rs`, `TestBackend`): heart lettering present while
  `splash=Some`, absent after; key modal NOT rendered while splash active;
  small-area fallback shows the one-liner.
- **Live (pty)**: startup frame contains the lettering; frame at ~2.5s no longer
  does (auto-dismiss); a second session with an immediate keypress skips it
  (composer unaffected); `splash = false` in config → lettering never appears.

## Non-goals

ECG sweep animation (the considered alternative), per-launch art variations,
configurable duration, splash on `/clear` or `/resume` (startup only).
