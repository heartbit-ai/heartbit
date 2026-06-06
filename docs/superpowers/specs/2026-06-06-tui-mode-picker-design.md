# TUI `/mode` Modal Picker — Design

**Date:** 2026-06-06
**Status:** Approved ("oui go")
**Scope:** `crates/heartbit-tui` only (`app.rs` + `ui.rs`)

## Behavior

| Input | Today | After |
|---|---|---|
| `/mode` (bare) | Notice with current mode + usage | **Opens a modal picker**, preselected on the current mode |
| `/mode plan` | Sets the mode directly | Unchanged |
| `/mode garbage` | Usage notice | Unchanged |
| Shift+Tab | Cycles modes, no modal | Unchanged |

## Components

### `app.rs`
- New modal variant: `Modal::ModePicker { sel: usize }` — no payload struct
  (nothing to load, unlike `SessionPicker`).
- `MODES: [PermissionMode; 3] = [Normal, Plan, Yolo]` — single source for
  picker order (matches the Shift+Tab cycle order).
- `set_mode("")`: open `Modal::ModePicker { sel: <index of current mode> }`
  instead of pushing the informational notice.
- `handle_mode_picker_key` (mirrors `handle_session_picker_key`):
  - `Esc` → close, no change.
  - `Up`/`Down` → move `sel` with wrap over the 3 entries.
  - `Enter` → apply `MODES[sel]` through the SAME path as `/mode <arg>`:
    set `permission_mode`, push `Effect::SetPermissionMode(mode.as_u8())`,
    push the "{label} mode — {describe}" notice; close the modal.
- Routing: `handle_modal_key` dispatches `Some(Modal::ModePicker { .. })` to
  the new handler (the existing modal gate already blocks composer input).

### `ui.rs`
- Render arm for `Modal::ModePicker { sel }` beside the SessionPicker arm:
  centered modal, title ` mode `, one row per mode —
  `● {label} — {describe()}` where `●` marks the CURRENT mode; the `sel` row
  is highlighted (reverse style, the SessionPicker convention); footer hint
  `↑↓ · Enter · Esc`.

## Edge cases

- Picker open while a run is active: allowed (mode changes mid-run already are,
  via Shift+Tab — the approval gate reads the atomic on next use).
- Splash active: modal rendering is already skipped under the splash; `/mode`
  cannot be typed during the splash anyway (keys dismiss it first).

## Testing

- Reducer: bare `/mode` opens the picker preselected on the current mode;
  `Enter` applies the selection (mode + effect + notice + modal closed);
  `Esc` closes without change or effect; ↑ from index 0 wraps to YOLO;
  `/mode plan` still sets directly without a modal.
- Render (TestBackend): the three labels and a description fragment visible
  while the picker is open.

## Non-goals

Quick-select keys (1/2/3 or n/p/y), mouse selection, mode-dependent styling
beyond the existing conventions.
