# TUI `/mode` Modal Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bare `/mode` opens a centered modal listing Normal · Plan · YOLO (current mode preselected) — ↑↓ wrap, Enter applies, Esc cancels; `/mode <arg>` unchanged.

**Architecture:** New `Modal::ModePicker { sel }` variant reduced exactly like the existing `SessionPicker` (key handler mirror) and rendered beside it in `ui.rs`; `set_mode("")` arms it instead of pushing a notice.

**Tech Stack:** Rust, ratatui 0.29, the TUI Elm-style reducer.

Spec: `docs/superpowers/specs/2026-06-06-tui-mode-picker-design.md`.

Verified anchors: `set_mode` at app.rs:1184 (bare-arg branch pushes a notice today); `handle_session_picker_key` at app.rs:1442 (the Esc/↑↓-wrap/Enter shape to mirror); modal dispatch at app.rs:1436; SessionPicker render arm at ui.rs:533; `PermissionMode::{label, describe, parse, as_u8}` all exist.

---

### Task 1: reducer — variant, open, navigate, apply

**Files:**
- Modify: `crates/heartbit-tui/src/app.rs`
- Test: `app.rs` tests module

- [x] **Step 1: failing tests** (beside the other slash tests):

```rust
    #[test]
    fn bare_mode_opens_picker_preselected_on_current() {
        let mut app = keyed();
        app.permission_mode = PermissionMode::Plan;
        typed(&mut app, "/mode");
        app.update(key(KeyCode::Enter));
        assert!(
            matches!(app.modal, Some(Modal::ModePicker { sel: 1 })),
            "picker must open on the CURRENT mode (Plan = index 1)"
        );
    }

    #[test]
    fn mode_picker_enter_applies_esc_cancels_and_wraps() {
        let mut app = keyed();
        app.modal = Some(Modal::ModePicker { sel: 0 });
        app.update(key(KeyCode::Up)); // wrap 0 → 2 (YOLO)
        assert!(matches!(app.modal, Some(Modal::ModePicker { sel: 2 })));
        app.update(key(KeyCode::Enter));
        assert_eq!(app.permission_mode, PermissionMode::Yolo);
        assert!(app.modal.is_none());
        assert!(app.effects.contains(&Effect::SetPermissionMode(2)));
        // Esc path: no change, no effect.
        let mut app = keyed();
        app.modal = Some(Modal::ModePicker { sel: 2 });
        app.update(key(KeyCode::Esc));
        assert!(app.modal.is_none());
        assert_eq!(app.permission_mode, PermissionMode::Normal);
        assert!(!app.effects.iter().any(|e| matches!(e, Effect::SetPermissionMode(_))));
    }

    #[test]
    fn mode_with_arg_still_sets_directly() {
        let mut app = keyed();
        typed(&mut app, "/mode yolo");
        app.update(key(KeyCode::Enter));
        assert!(app.modal.is_none(), "arg path must NOT open the picker");
        assert_eq!(app.permission_mode, PermissionMode::Yolo);
    }
```

- [x] **Step 2:** Run `cargo test -p heartbit-tui mode_picker bare_mode` — FAIL (no `ModePicker` variant).

- [x] **Step 3: implement** in `app.rs`:

Add the variant to `Modal`:

```rust
    /// `/mode` picker: choose the execution mode (sel indexes [`MODES`]).
    ModePicker { sel: usize },
```

Add beside `PermissionMode`:

```rust
/// Picker order — matches the Shift+Tab cycle.
pub const MODES: [PermissionMode; 3] =
    [PermissionMode::Normal, PermissionMode::Plan, PermissionMode::Yolo];
```

Replace `set_mode`'s bare-arg branch (the notice push) with:

```rust
        if arg.trim().is_empty() {
            let sel = MODES
                .iter()
                .position(|m| *m == self.permission_mode)
                .unwrap_or(0);
            self.modal = Some(Modal::ModePicker { sel });
            return;
        }
```

Add the dispatch arm in `handle_modal_key`:

```rust
            Some(Modal::ModePicker { .. }) => self.handle_mode_picker_key(key),
```

Add the handler (below `handle_session_picker_key`):

```rust
    /// `/mode` picker keys: ↑/↓ select (wrap), Enter apply, Esc cancel.
    fn handle_mode_picker_key(&mut self, key: KeyEvent) {
        let n = MODES.len();
        match key.code {
            KeyCode::Esc => self.modal = None,
            KeyCode::Up => {
                if let Some(Modal::ModePicker { sel }) = &mut self.modal {
                    *sel = (*sel + n - 1) % n;
                }
            }
            KeyCode::Down => {
                if let Some(Modal::ModePicker { sel }) = &mut self.modal {
                    *sel = (*sel + 1) % n;
                }
            }
            KeyCode::Enter | KeyCode::Char('\r') | KeyCode::Char('\n') => {
                let mode = match &self.modal {
                    Some(Modal::ModePicker { sel }) => MODES.get(*sel).copied(),
                    _ => None,
                };
                self.modal = None;
                if let Some(mode) = mode {
                    self.permission_mode = mode;
                    self.effects.push(Effect::SetPermissionMode(mode.as_u8()));
                    self.history.push(Cell::Notice(format!(
                        "{} mode — {}",
                        mode.label(),
                        mode.describe()
                    )));
                }
            }
            _ => {}
        }
    }
```

- [x] **Step 4:** `cargo test -p heartbit-tui` — ALL PASS. (One existing test may assert the OLD bare-`/mode` notice — if so, update it to assert the picker opens instead; that behavior change is the feature.)

- [x] **Step 5: Commit** `feat(tui): /mode opens a mode picker modal — reducer`

---

### Task 2: render arm

**Files:**
- Modify: `crates/heartbit-tui/src/ui.rs` (beside the SessionPicker arm at ~533)
- Test: `ui.rs` tests module

- [x] **Step 1: failing test**:

```rust
    #[test]
    fn mode_picker_modal_lists_all_modes() {
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new("m");
        app.modal = Some(Modal::ModePicker { sel: 1 });
        terminal.draw(|f| view(f, &app)).unwrap();
        let text = buffer_text(terminal.backend().buffer());
        for label in ["Normal", "Plan", "YOLO"] {
            assert!(text.contains(label), "missing {label}:\n{text}");
        }
        assert!(text.contains("read-only"), "Plan description shown:\n{text}");
    }
```

- [x] **Step 2:** Run — FAIL (non-exhaustive match or nothing rendered).

- [x] **Step 3: implement** the render arm beside SessionPicker (match its centering/Block conventions — reuse the same helper the picker uses for the centered rect). One `Line` per mode: `●`/space current-marker + `label` + ` — ` + `describe()`; the `sel` row styled `Modifier::REVERSED`; block title ` mode `; bottom hint line `↑↓ · Enter · Esc` dimmed.

```rust
        Some(Modal::ModePicker { sel }) => {
            let current = app.permission_mode;
            let lines: Vec<Line> = crate::app::MODES
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    let marker = if *m == current { "●" } else { " " };
                    let row = format!(" {marker} {:<6} — {}", m.label(), m.describe());
                    if i == *sel {
                        Line::from(Span::styled(
                            row,
                            Style::default().add_modifier(Modifier::REVERSED),
                        ))
                    } else {
                        Line::raw(row)
                    }
                })
                .chain(std::iter::once(Line::from(Span::styled(
                    " ↑↓ · Enter · Esc",
                    Style::default().fg(Color::DarkGray),
                ))))
                .collect();
            // …then render `lines` in the same centered Clear+Block+Paragraph
            // shell the SessionPicker arm uses (copy its rect math; height =
            // lines.len()+2 for the borders, width ~70 clamped to the frame).
        }
```

(Adapt to the SessionPicker arm's exact rendering shell when writing it — same Block style, same Clear.)

- [x] **Step 4:** `cargo test -p heartbit-tui` — ALL PASS.

- [x] **Step 5: Commit** `feat(tui): /mode picker modal rendering`

---

### Task 3: gate + live pty

- [x] `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm` — green.
- [x] `cargo build -p heartbit-tui`, then pty: type `/mode` + Enter → settled frame shows the three labels + `read-only`; send ↓ then Enter → frame's status line shows `Plan`; trace/no-crash sanity. Esc path: reopen, Esc → no mode change.
- [x] Mark plan checkboxes; report.

## Self-review

1. **Spec coverage:** open-preselected (T1) · ↑↓ wrap/Enter-apply/Esc (T1) · arg path unchanged (T1 test 3) · render with labels+descriptions+marker (T2) · Shift+Tab untouched (no change) · live validation (T3). ✓
2. **Placeholders:** the T2 shell instruction points at the concrete SessionPicker arm to copy — acceptable (exact code depends on its rect helper; the list-building code IS given).
3. **Type consistency:** `Modal::ModePicker { sel: usize }` everywhere; `MODES` pub const used by reducer + render; `Effect::SetPermissionMode(u8)` matches the existing effect.
