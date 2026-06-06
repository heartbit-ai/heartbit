# TUI Splash Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A full-screen startup splash for heartbit-tui — a beating block-art heart + `heartbit` lettering — that auto-dismisses after ~1.5s or on any key, opt-out via `splash = false` in tui.toml.

**Architecture:** Splash is App state (`splash: Option<u8>` tick counter) reduced by the existing 120ms `Msg::Tick` and by `Msg::Key` (dismiss + consume); `view()` renders a pure-art overlay from new `splash.rs` and skips everything else while active. No new threads or timers.

**Tech Stack:** Rust, ratatui 0.29 (`Paragraph` + `Alignment::Center`), the existing TUI Elm-style reducer.

Spec: `docs/superpowers/specs/2026-06-06-tui-splash-screen-design.md`.

Verified code anchors (2026-06-06):
- `app.rs` `Msg::Key(key)` arm checks `self.modal` then `handle_key` — splash consume goes BEFORE both.
- `app.rs:579` `Msg::Tick => self.spinner = self.spinner.wrapping_add(1),`
- `ui.rs:70` `pub fn view(frame: &mut Frame, app: &App) { let area = frame.area();`
- `config.rs` has `fn default_true() -> bool { true }` + `fn is_true(b: &bool) -> bool` and `prompt_caching` uses `#[serde(default = "default_true", skip_serializing_if = "is_true")]`.
- `main.rs:182-185` is the `app.multi_agent = cfg...` wiring block; mods are declared at `main.rs:20-32`.

---

### Task 1: `splash.rs` — pure art module

**Files:**
- Create: `crates/heartbit-tui/src/splash.rs`
- Modify: `crates/heartbit-tui/src/main.rs:20-32` (add `mod splash;` in the alphabetical mod list, between `mod session;` and `mod trace;`)
- Test: same file, `#[cfg(test)] mod tests`

- [x] **Step 1: Write the failing tests** — create `crates/heartbit-tui/src/splash.rs` containing ONLY the tests + a doc comment:

```rust
//! Startup splash: a beating block-art heart + `heartbit` lettering, rendered
//! by `ui::view` as a full-frame overlay while `App.splash` is `Some(tick)`.
//! Pure functions only — all timing lives in the reducer (`Msg::Tick`).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rhythm_is_two_beats() {
        // Bright/large on 0-3 and 6-9; small/dim on 4-5 and 10-12.
        for t in [0u8, 3, 6, 9] {
            assert!(is_large(t), "tick {t} must be the LARGE frame");
        }
        for t in [4u8, 5, 10, 12] {
            assert!(!is_large(t), "tick {t} must be the SMALL frame");
        }
    }

    #[test]
    fn frames_never_shift_layout() {
        let large = splash_lines(0, "m/x");
        let small = splash_lines(4, "m/x");
        assert_eq!(large.len(), small.len(), "line count must not change");
        assert!(large.len() >= 12, "heart + lettering + meta lines");
    }

    #[test]
    fn lines_carry_version_model_and_differ_by_frame() {
        let text = |lines: &[ratatui::text::Line]| -> String {
            lines
                .iter()
                .map(|l| {
                    l.spans
                        .iter()
                        .map(|s| s.content.as_ref())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let large = text(&splash_lines(0, "qwen/q3"));
        let small = text(&splash_lines(4, "qwen/q3"));
        assert!(large.contains(env!("CARGO_PKG_VERSION")));
        assert!(large.contains("qwen/q3"));
        assert_ne!(large, small, "beat frames must differ");
    }
}
```

- [x] **Step 2:** Add `mod splash;` to `main.rs` (between `mod session;` and `mod trace;`). Run: `cargo test -p heartbit-tui splash` — expected: FAIL to compile (`is_large`, `splash_lines` not found).

- [x] **Step 3: Implement** above the tests in `splash.rs`:

```rust
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Total splash duration in 120ms ticks (~1.56s — two heartbeats).
pub const SPLASH_TICKS: u8 = 13;

/// Large (systole) heart frame, 7 rows of half-block art.
const HEART_LARGE: [&str; 7] = [
    " ▄▄██▄▄   ▄▄██▄▄ ",
    "█████████████████",
    " ███████████████ ",
    "   ███████████   ",
    "     ███████     ",
    "       ███       ",
    "        ▀        ",
];

/// Small (diastole) frame — 5 rows padded to 7 so the layout never shifts.
const HEART_SMALL: [&str; 7] = [
    "",
    " ▄█▄ ▄█▄ ",
    "█████████",
    " ███████ ",
    "   ███   ",
    "    ▀    ",
    "",
];

/// `heartbit` in half-block letters (H E A R T B I T).
const LETTERING: [&str; 2] = [
    "█ █ █▀▀ ▄▀█ █▀█ ▀█▀ █▄▄ █ ▀█▀",
    "█▀█ ██▄ █▀█ █▀▄  █  █▄█ █  █ ",
];

/// Beat rhythm: LARGE on ticks 0-3 and 6-9, SMALL on 4-5 and 10-12.
pub(crate) fn is_large(tick: u8) -> bool {
    matches!(tick, 0..=3 | 6..=9)
}

/// The full splash as centered-ready lines: heart (beat frame by `tick`),
/// lettering, `v{version} · {model}`, and a dim dismissal hint.
pub fn splash_lines(tick: u8, model: &str) -> Vec<Line<'static>> {
    let (heart, heart_style) = if is_large(tick) {
        (
            &HEART_LARGE,
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        (&HEART_SMALL, Style::default().fg(Color::Red))
    };
    let mut lines: Vec<Line<'static>> = heart
        .iter()
        .map(|r| Line::from(Span::styled((*r).to_string(), heart_style)))
        .collect();
    lines.push(Line::raw(""));
    let letter_style = Style::default()
        .fg(Color::Magenta)
        .add_modifier(Modifier::BOLD);
    lines.extend(
        LETTERING
            .iter()
            .map(|r| Line::from(Span::styled((*r).to_string(), letter_style))),
    );
    let dim = Style::default().fg(Color::DarkGray);
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        format!("v{} · {model}", env!("CARGO_PKG_VERSION")),
        dim,
    )));
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "any key",
        dim.add_modifier(Modifier::ITALIC),
    )));
    lines
}
```

- [x] **Step 4:** Run: `cargo test -p heartbit-tui splash` — expected: 3 PASS. (`SPLASH_TICKS` is unused until Task 2 — if clippy complains at this point, ignore until Task 2 wires it; do not blanket-allow.)

- [x] **Step 5: Commit**

```bash
git add crates/heartbit-tui/src/splash.rs crates/heartbit-tui/src/main.rs
git commit -m "feat(tui): splash art module — beating heart frames + lettering (pure)"
```

---

### Task 2: reducer — arm, tick, dismiss, consume

**Files:**
- Modify: `crates/heartbit-tui/src/app.rs` (field near `pub spinner: usize` ~382, init near `spinner: 0` ~422, `Msg::Tick` arm ~579, `Msg::Key` arm ~600)
- Test: `app.rs` tests module

- [x] **Step 1: Write the failing tests** (app.rs tests module, near the other `Msg` reducer tests):

```rust
    #[test]
    fn splash_auto_dismisses_after_its_ticks() {
        let mut app = keyed();
        app.splash = Some(0);
        for _ in 0..(crate::splash::SPLASH_TICKS - 1) {
            app.update(Msg::Tick);
        }
        assert!(app.splash.is_some(), "still up one tick before the end");
        app.update(Msg::Tick);
        assert_eq!(app.splash, None, "gone at SPLASH_TICKS");
    }

    #[test]
    fn splash_key_dismisses_and_is_consumed() {
        let mut app = keyed();
        app.splash = Some(2);
        app.update(key(KeyCode::Char('h')));
        assert_eq!(app.splash, None, "any key dismisses");
        assert_eq!(app.composer.text(), "", "the dismissing key must NOT type");
        app.update(key(KeyCode::Char('h')));
        assert_eq!(app.composer.text(), "h", "subsequent keys flow normally");
    }
```

- [x] **Step 2:** Run: `cargo test -p heartbit-tui splash_` — expected: FAIL to compile (no field `splash` on `App`).

- [x] **Step 3: Implement** in `app.rs`:

Next to `pub spinner: usize,` (~382):

```rust
    /// Startup splash tick counter — `Some(t)` while the splash overlay is up
    /// (armed by main from config); `None` once dismissed (timer or any key).
    pub splash: Option<u8>,
```

Next to `spinner: 0,` in `App::new` (~422): `splash: None,`

Replace the `Msg::Tick` arm (~579):

```rust
            Msg::Tick => {
                self.spinner = self.spinner.wrapping_add(1);
                if let Some(t) = self.splash {
                    let t = t.saturating_add(1);
                    self.splash = (t < crate::splash::SPLASH_TICKS).then_some(t);
                }
            }
```

In the `Msg::Key(key)` arm, FIRST (before the `self.modal` check):

```rust
            Msg::Key(key) => {
                // Any key dismisses the splash and is CONSUMED — an impatient
                // first keypress must not leak a stray char into the composer
                // (nor reach a modal hidden beneath the overlay).
                if self.splash.is_some() {
                    self.splash = None;
                    let _ = key;
                    return;
                }
                if self.modal.is_some() {
```

- [x] **Step 4:** Run: `cargo test -p heartbit-tui` — expected: ALL PASS (the two new tests + no regressions).

- [x] **Step 5: Commit**

```bash
git add crates/heartbit-tui/src/app.rs
git commit -m "feat(tui): splash reducer — tick-driven auto-dismiss, key dismiss+consume"
```

---

### Task 3: `ui.rs` overlay rendering

**Files:**
- Modify: `crates/heartbit-tui/src/ui.rs:70-72` (top of `view()`); add `Alignment` to the `ratatui::layout` import at ui.rs:5
- Test: `ui.rs` tests module (TestBackend pattern — see `empty_transcript_shows_welcome_header` at ~765)

- [x] **Step 1: Write the failing tests** (ui.rs tests module; mirror the existing TestBackend helper usage):

```rust
    #[test]
    fn splash_overlay_replaces_everything() {
        let mut app = App::new("m");
        app.splash = Some(0);
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(80, 24)).unwrap();
        term.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(term.backend());
        assert!(text.contains("▄▄██▄▄"), "heart art visible:\n{text}");
        assert!(text.contains(env!("CARGO_PKG_VERSION")), "{text}");
        assert!(
            !text.contains("Type a message"),
            "composer hidden during splash:\n{text}"
        );
    }

    #[test]
    fn splash_hides_modals_and_clears_after() {
        let mut app = App::new("m");
        app.open_key_modal();
        app.splash = Some(0);
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(80, 24)).unwrap();
        term.draw(|f| view(f, &app)).unwrap();
        let during = buffer_text(term.backend());
        assert!(!during.contains("API key"), "modal hidden under splash:\n{during}");
        app.splash = None;
        term.draw(|f| view(f, &app)).unwrap();
        let after = buffer_text(term.backend());
        assert!(after.contains("API key"), "modal appears at dissolution:\n{after}");
        assert!(!after.contains("▄▄██▄▄"), "art gone:\n{after}");
    }

    #[test]
    fn splash_small_terminal_falls_back_to_one_liner() {
        let mut app = App::new("m");
        app.splash = Some(0);
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(40, 10)).unwrap();
        term.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(term.backend());
        assert!(text.contains("♥ heartbit"), "{text}");
        assert!(!text.contains("▄▄██▄▄"), "no art on tiny terminals:\n{text}");
    }
```

If `ui.rs` tests have no `buffer_text` helper, add one in the tests module:

```rust
    fn buffer_text(backend: &ratatui::backend::TestBackend) -> String {
        let buf = backend.buffer();
        let area = buf.area();
        (0..area.height)
            .map(|y| {
                (0..area.width)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
```

(Check first — the existing render tests likely already have an equivalent; reuse theirs and match its name.)

- [x] **Step 2:** Run: `cargo test -p heartbit-tui splash_overlay` — expected: FAIL (art not rendered; composer text present).

- [x] **Step 3: Implement** at the very top of `view()` (right after `let area = frame.area();`), plus add `Alignment` to the `ratatui::layout` import:

```rust
    // Startup splash: a full-frame overlay that pre-empts EVERYTHING — no
    // transcript, status, composer, or modal renders beneath it. Dismissal
    // (timer / any key) lives in the reducer; this is pure paint.
    if let Some(tick) = app.splash {
        let lines = if area.height < 16 || area.width < 44 {
            vec![Line::from(Span::styled(
                format!("♥ heartbit v{}", env!("CARGO_PKG_VERSION")),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ))]
        } else {
            crate::splash::splash_lines(tick, &app.model)
        };
        let h = (lines.len() as u16).min(area.height);
        let top = area.height.saturating_sub(h) / 2;
        let rect = Rect {
            x: area.x,
            y: area.y + top,
            width: area.width,
            height: h,
        };
        frame.render_widget(
            Paragraph::new(lines).alignment(Alignment::Center),
            rect,
        );
        return;
    }
```

- [x] **Step 4:** Run: `cargo test -p heartbit-tui` — expected: ALL PASS.

- [x] **Step 5: Commit**

```bash
git add crates/heartbit-tui/src/ui.rs
git commit -m "feat(tui): splash overlay rendering — centered art, modals deferred, tiny-term fallback"
```

---

### Task 4: config flag + arming

**Files:**
- Modify: `crates/heartbit-tui/src/config.rs` (field after `prompt_caching` ~111, `Default` impl ~137)
- Modify: `crates/heartbit-tui/src/main.rs` (config wiring block at ~182-185)
- Test: `config.rs` tests module

- [x] **Step 1: Write the failing tests** (config.rs tests module, beside the existing default/roundtrip tests):

```rust
    #[test]
    fn splash_defaults_on_and_parses_off() {
        assert!(TuiConfig::default().splash);
        let cfg: TuiConfig = toml::from_str("").unwrap();
        assert!(cfg.splash, "missing key means ON");
        let cfg: TuiConfig = toml::from_str("splash = false").unwrap();
        assert!(!cfg.splash);
    }
```

- [x] **Step 2:** Run: `cargo test -p heartbit-tui splash_defaults` — expected: FAIL to compile (no field `splash`).

- [x] **Step 3: Implement** — in `TuiConfig` after `prompt_caching`:

```rust
    /// Show the startup splash (the beating heart). Disable with
    /// `splash = false` in tui.toml.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub splash: bool,
```

In `impl Default for TuiConfig`: `splash: true,` (after `prompt_caching: true,`).

In `main.rs` after `app.prompt_caching = cfg.prompt_caching;` (~185):

```rust
    app.splash = cfg.splash.then_some(0);
```

- [x] **Step 4:** Run: `cargo test -p heartbit-tui` — expected: ALL PASS.

- [x] **Step 5: Commit**

```bash
git add crates/heartbit-tui/src/config.rs crates/heartbit-tui/src/main.rs
git commit -m "feat(tui): splash config flag (default on) + startup arming"
```

---

### Task 5: workspace gate

- [x] `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm` — all green. Commit only if fixes were needed.

---

### Task 6: live validation (pty)

Per the project bar (settled frame, space-insensitive assertions). `cargo build -p heartbit-tui` FIRST (tests do not rebuild the binary).

- [x] **Step 1:** Default path: fresh pty session (isolated `HEARTBIT_TUI_CONFIG`, temp cwd, key in env). Capture at ~0.5s: the de-ANSI'd frame contains the lettering rows (`█▀█` etc.) and the version. Capture at ~3s: lettering gone, normal welcome header + composer visible.
- [x] **Step 2:** Key-skip path: new session, send a key at ~0.3s, capture at ~1s: lettering gone AND the composer is empty (the key was consumed).
- [x] **Step 3:** Opt-out path: write `splash = false` into the isolated tui.toml, new session, capture at ~0.5s: NO lettering, normal startup immediately.
- [x] **Step 4:** Report results; mark plan checkboxes.

---

## Self-review

1. **Spec coverage:** art + rhythm (T1) · reducer arm/tick/dismiss/consume (T2) · overlay + modal deferral + tiny-term fallback (T3) · config opt-out + arming (T4) · gate (T5) · live incl. key-skip + opt-out (T6). Resize mid-splash needs no code (immediate-mode re-render each draw) — covered by design, nothing to implement. No trace events — nothing to do.
2. **Placeholders:** none — full code in every step; the one conditional instruction (reuse an existing `buffer_text`-style helper if present) names the exact fallback code.
3. **Type consistency:** `splash: Option<u8>` (T2) read by `view()` (T3) and armed by main (T4); `SPLASH_TICKS: u8 = 13` (T1) used by the reducer (T2); `splash_lines(tick: u8, model: &str)` signature consistent T1/T3.
