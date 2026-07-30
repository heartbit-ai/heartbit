# Wave 1 — Table Stakes + Shelfware (TUI sub-spec)

**Date:** 2026-06-26
**Status:** approved design, ready for an implementation plan
**Parent:** `2026-06-26-tui-sota-upgrade-design.md` (framework: principles A1–A5, core ledger C1–C10, decisions, wave boundaries)
**Basis:** commit `37bb8a5` · 11 per-item implementation contracts verified against the tree + one adversarial crosscheck pass (12 agents, read-only)

Wave 1 closes the gap between heartbit's TUI and its own engine, and fixes the
input plane. Its acceptance signal is in §7. **Core change: C3 only** — the
framework promise holds (see decision D-1).

---

## 1. What verification changed

Nine of eleven contracts came back **corrected**. Recording this because the
corrections are the spec:

| Claim in the research report | Verified reality |
|---|---|
| "`Msg::Paste` may only feed modals — enabling bracketed paste could make pastes vanish" | **Refuted.** `app.rs:755` `None => self.composer.insert_str(&s)` already routes to the composer, and `composer.rs:180-188` already converts `\n`→`newline()` and drops `\r` (test `composer.rs:438`). **Bracketed paste needs zero reducer work** — only the terminal-mode enable. |
| "Shift+Enter is the newline path and it is unreachable" | **Overstated.** `app.rs:1818` is `if shift \|\| alt` → **Alt+Enter already works on every terminal today.** Kitty flags add Shift+Enter; they do not unlock the *only* newline. |
| "Kitty flags are ~5 inert lines" | `supports_keyboard_enhancement()` (crossterm `terminal/sys/unix.rs:186-252`) writes `\x1b[?u\x1b[c` and **blocks up to 2000 ms**, erroring on any terminal that does not answer — including the project's own pty harness. See D-3. |
| "Wire `.learned_permissions()` — a ~5-line builder call" | **It is a live-bug fix.** `default_permissions()` ends with a terminal `{tool:"*", pattern:"*", action:Ask}` (`main.rs:1060-1064`); `evaluate` is first-match-wins (`permission.rs:137-142`); `append_rules` extends at the **tail** (`permission.rs:148-150`). So learned rules are unreachable — and today's in-session `[a]` **already** fails to hold: the next `bash` re-hits the catch-all and re-prompts. |
| "Post-edit formatters: run the formatter, then refresh the mtime" | **Mechanism defective in three ways** — see item C3/T1.8. The worst: shelling out to `rustfmt <path>` would write the file *outside* the F-FS-1 symlink hardening (`write_beneath_root`/`write_no_follow`). A security regression. |
| "Send OSC 777 + OSC 9 + BEL" | kitty, WezTerm and Ghostty implement **both** OSC 777 and OSC 9 → the user gets a **double** notification. Pick one per terminal. |
| "Wire `.lsp_manager()` — one builder call" | Would ship a guaranteed **30 s stall with zero diagnostics** on every clean file (`lsp/server.rs:195-239`: an empty `publishDiagnostics` advances `current_version` and re-waits) plus a malformed `file://` URI (`server.rs:89`,`:148`). See D-1. |

Two further defects surfaced that nobody had listed: `Msg::Paste` does not check
`self.splash` (`app.rs:738` vs the key path at `:761`), so with bracketed paste on
a paste during the ~3 s splash silently fills a composer the overlay hides; and
`Msg::Resize` is a **no-op** (`app.rs:733`), which the highlight cache must
account for.

---

## 2. Decisions

**D-1 — Item 0.2 (LSP diagnostics) is deferred out of Wave 1.**
It needs three *more* core changes and two are not optional: wiring
`.lsp_manager()` as-is makes every successful edit wait the full 30 s timeout and
return nothing. That contradicts the framework's "Wave 1 needs C3 only", and the
URI fix also lands on four heartbit-cli LSP call sites. Deferring keeps Wave 1
low-risk and gives the LSP work the spec and tests it deserves.
→ **Wave 1.5** (`tui-lsp-diagnostics-design.md`, dated when written), immediately
after Wave 1. Alternatives rejected: amending the ledger (dilutes the wave's purpose);
shipping default-off (makes "shelfware wired" a false claim and leaves a landmine).

**D-2 — Item 0.1 is scoped as a bug fix, not a wiring.**
It must drop the terminal `{"*","*",Ask}` rule from `default_permissions()`. This
is behaviour-identical because the single production consumer maps `Some(Ask)` and
`None` to the same arm (`runner.rs:2097` `Some(PermissionAction::Ask) | None =>
needs_approval.push(call)`), and `has_permission_rules()` (`runner.rs:619-621`)
stays true on the 10 remaining allow rules. It requires editing **one** existing
test (`main.rs:1876-1883`, `Some(Ask)` → `None`) — the only existing test any
Wave 1 item changes — plus a new **core** test pinning that arm so a future change
there cannot silently turn the TUI into Yolo.

**D-3 — Kitty flags are pushed unconditionally; no capability probe.**
Rejecting the contract's probe design: `supports_keyboard_enhancement()` costs up
to 2 s on exactly the terminals that gain nothing, and it breaks the pty harness.
A private-mode CSI (`\x1b[>1u`) is ignored by conforming terminals that do not
implement it, the pop (`\x1b[<1u`) likewise, and **Alt+Enter already works
everywhere**, so a terminal that ignores the push simply keeps today's behaviour
at zero startup cost. Escape hatch: `keyboard_enhancement = false` in `tui.toml`.
The acceptance script verifies no artifacts on a legacy terminal (§7 step 6).

**D-4 — The queue intercepts at one choke point.**
`Effect::SendInput` is pushed from **seven** sites (`app.rs:1032` AnalyzeReady,
`1047` LearnReady, `1276`/`1281`/`1293` `/goal` + `/handoff`, `1372` `/research`,
and `submit()` `1168-1209`). A queue implemented only in `submit()` leaves six
mid-turn bypasses, so all seven route through one private
`send_or_queue(&mut self, text: String)` helper.

**D-5 — The flag set is exactly `DISAMBIGUATE_ESCAPE_CODES`.**
Sufficient: `CSI 13;2u` → codepoint 13 → `'\r'` → `KeyCode::Enter` + SHIFT
(crossterm `parse.rs:547`), and Shift+Tab still yields `BackTab`
(`parse.rs:552-558`) so the mode cycle at `app.rs:1825` survives. Adding
`REPORT_EVENT_TYPES` is a **regression**: `KeyEvent::kind` is only populated under
that flag on Unix (`event.rs:744-748`) and `translate` admits only
`KeyEventKind::Press` (`main.rs:1100`) → `Repeat` events are dropped → **held
Backspace and arrows stop auto-repeating**.

---

## 3. Items

Ordering is §5. Effort per the verified contracts.

### 0.6 — Two micro-defects (S, no core) — *first, unblocks 0.1's discoverability*

- **(a)** `app.rs:1837`: `KeyCode::Char('u') if ctrl => self.composer = Composer::new()` discards the per-directory prompt history seeded by `composer.seed_history(...)`. Fix: `self.composer.clear()` (`composer.rs:224` keeps `history`).
- **(b)** `ui.rs:416`: the approval hint omits the working `d` = AlwaysDeny key (`ApprovalDecision::AlwaysDeny` exists, `llm/mod.rs:144`, honoured at `runner.rs:2121`).

**Invariants:** Ctrl+U still empties the draft and resets the cursor to `(0,0)` including after hard newlines; `lines == vec![Vec::new()]` never an empty `Vec` (`visual_cursor`/`mention_prefix` index `lines[row]` unguarded); `Up` still recalls the newest seeded entry (needs `hist_pos == None`); Ctrl+R search still non-empty; Ctrl+U still closes an open `/` or `@` menu.
**Tests:** `ctrl_u_clears_the_draft_but_keeps_recall_history`, `approval_modal_always_denies_on_d`, `approval_modal_hint_lists_every_answer_key`, `ctrl_u_then_retype_reopens_the_slash_menu`.

### 0.5 + 0.4 — Terminal modes: bracketed paste, focus events, Kitty flags (S, no core) — *one pass*

They write the same three places (`main.rs:320-324` setup, `:341-342` teardown, the panic hook), so they ship together.

- **Enable** `EnableBracketedPaste` + `EnableFocusChange` beside the existing `EnableMouseCapture`; push `\x1b[>1u` per D-3/D-5.
- **`translate()`** (`main.rs:1096-1111`) gains the `Event::FocusGained`/`FocusLost` arms (today swallowed by `_ => None`) → a new `Msg::FocusChanged(bool)`; `App::focused` defaults `true` in `App::new` and is written only by that Msg.
- **Splash fix:** `Msg::Paste` must check `self.splash` like the key path does (`app.rs:761`), dismissing the overlay and keeping the text.
- **Panic-safe restore must WRAP, not replace:** after `ratatui::init()`, `take_hook()` + `set_hook(move |i| { pop kitty; disable paste/focus/mouse; prev(i) })`. Replacing loses ratatui's own `restore()` and leaves raw mode on. Note ratatui's hook restore (`init.rs:225-231`) is only `disable_raw_mode` + `LeaveAlternateScreen` — it does not even disable mouse capture today (pre-existing leak this item closes).
- Push and pop each get their **own** `execute!` — `queue!` short-circuits on first error.
- The pop is emitted **at most once** per process (`AtomicBool::swap(false, SeqCst)`), and never if the push errored.

**Invariants:** the reducer performs no terminal I/O; all five existing modal branches of the `Msg::Paste` arm (`app.rs:741-754`) keep byte-identical behaviour; **a paste never submits** (no path from `Msg::Paste` to `submit()`/`SendInput`/`PersistPrompt`); a mid-draft paste preserves the tail and leaves the cursor after the insert; a terminal that never reports focus leaves `focused == true` forever, so every consumer reads exactly today's behaviour.
**Tests:** `multiline_paste_lands_as_one_draft_and_does_not_submit`, `paste_mid_draft_preserves_the_tail_and_leaves_cursor_after_insert`, `crlf_paste_yields_single_newlines`, `paste_during_splash_dismisses_the_overlay_and_keeps_the_text`, `paste_ending_in_a_mention_token_requests_the_file_index`, `focus_defaults_to_focused_and_tracks_both_directions`, `translate_maps_focus_events_and_paste`, `alt_enter_inserts_newline_not_submit`, `push_pop_emit_the_minimal_kitty_sequences` (asserts exactly `\x1b[>1u`/`\x1b[<1u`), `keyboard_flag_pop_is_exactly_once`.

### 0.1 — Persistent approval rules (M, no core code; one core *test*)

Per D-2. Wire `LearnedPermissions` (`permission.rs:159-292`: `load`, `save`,
`add_rule`, `rules`, `default_path`) through the single seam
`OrchestratorBuilder::learned_permissions(Arc<Mutex<LearnedPermissions>>)`
(`orchestrator.rs:2606`) — verified to forward to the entry runner
(`:3153-3155`) **and** all three sub-agent spawn paths (`:610`, `:1150`,
`:1593`), so one call suffices.

- Load **before** the "ready — …" line is sent (`main.rs:958`) so the count can appear in it; merge as `merged_permissions(learned.rules())` replacing `main.rs:987`, with learned rules ordered **ahead** of the defaults and the terminal catch-all removed.
- Storage under the TUI's config dir (`config.rs` helpers), `0600`, sibling of `tui.toml`; `HEARTBIT_TUI_CONFIG` must relocate it too (so the acceptance script can isolate it).
- Discoverability: a notice naming the tool **and** the file path when a rule is persisted. Core's failure path `tracing::warn!` (`runner.rs:680-683`) is **silently dropped** because `init_tracing` filters to `trace::INTERRUPT_TARGET` only (`main.rs:116-121`) — so the notice cannot rely on logs.
- **Say the scope honestly:** the TUI defaults to `PermissionMode::Yolo` (`app.rs:543`) and `on_approval` short-circuits `mode==2 → Allow` before the modal (`main.rs:824-828`). The whole feature is **Normal-mode-only**.

**Invariants:** with an absent/empty `permissions.toml`, `merged_permissions(&[])` resolves every tool exactly as `default_permissions()` did; `has_permission_rules()` stays true; the reducer is untouched (`app.rs`/`msg.rs` unchanged — `handle_approval_key` `app.rs:2166-2182` already emits AlwaysAllow/AlwaysDeny); `git diff --stat crates/heartbit-core` shows **only** the new test.
**Tests:** the edited `main.rs:1876-1883` (`Some(Ask)` → `None`); `merged_permissions_orders_learned_rules_first`; `merged_permissions_with_no_learned_rules_matches_today`; **core:** `ask_and_none_both_route_to_approval` pinning `runner.rs:2097`.
**Documented, not fixed:** sub-agents clone the ruleset at build time (`orchestrator.rs:491`,`:1037`), so a rule learned mid-session does not reach an already-built sub-agent. Pre-existing.

### 0.3 — `/effort` (M, no core)

Every seam exists: `ReasoningEffort {High, Medium, Low, None}` (`llm/types.rs:151-162`), `CompletionRequest.reasoning_effort` (`:180`), `OrchestratorBuilder::reasoning_effort` (`orchestrator.rs:2668`, applied `:3171-3173`), `SubAgentConfig.reasoning_effort` (`:2186`, applied `:542`,`:1083`).

- `/effort low|medium|high|off`, bare `/effort` opens a picker preselected on the current level (copy `ModePicker`); persisted in `tui.toml`; shown in the status line; mid-run change defers via `pending_respawn` (`flush_pending_respawn`, `app.rs:1478-1487`).
- **Provider gating is load-bearing, not cosmetic:** effort must reach only the OpenRouter / custom-endpoint providers. The `ANTHROPIC_API_KEY` fallback path must receive `None`, because on the non-streaming sub-agent path Anthropic's `ApiContentBlock` is `Text | ToolUse` with `#[serde(tag="type")]` and **no `#[serde(other)]`** (`anthropic.rs:778-789`) → a returned `thinking` block **fails deserialization**. Compute one gated `Option<ReasoningEffort>` in `build_engine` and give it to both the entry agent and every `SubAgentConfig`.
- `ReasoningEffort::None` is never constructed: `/effort off` maps to `Option::None` (omit the field), not to `Some(None)` — core's `None` emits `reasoning: {"effort":"none"}` (`openrouter.rs:378`), a request today's TUI never sends.

**Invariants:** default `off` ⇒ the builder call is never made ⇒ bit-for-bit today; new config field is a plain `Option<String>`, never a typed enum — `TuiConfig::load_from` (`config.rs:160-165`) **swallows any parse error and returns `Default`**, so one typo in a typed field would silently wipe the whole config including `openrouter_api_key`.
**Tests:** `effort_level_parse_and_label_roundtrip`, `effort_defaults_to_off_and_persists_nothing`, `slash_effort_sets_level_persists_and_requests_respawn`, `slash_effort_off_clears_and_drops_the_config_key`, `slash_effort_mid_run_defers_respawn_to_turn_idle`, `slash_effort_bare_opens_picker_preselected_on_current`, `slash_effort_unknown_arg_reports_usage_and_changes_nothing`.

### T1.3 — Visible input queue (M, no core)

Today a submit while running pushes straight into the unbounded channel
(`main.rs:234` → `:1398` → `on_input` `:861-881`) — accepted but invisible.

- New `App::queued: VecDeque<String>`; all seven senders go through `send_or_queue` (D-4); rendered above the composer; `Up` pops the newest for editing; Esc drops.
- **Turn-idle is four sites**, all must drain: `app.rs:818` (`LlmDone{had_tool_calls:false}`), `:935` (`RunCompleted`), `:945` (`AgentExited`), `:954` (`RunFailed`). Hooking only `:818` strands messages on a failure.
- Exactly **one** entry released per boundary — releasing several would push the rest back into the invisible channel and recreate the defect.
- Out of scope: injecting into the *current* turn (C6/Wave 3).

**Invariants:** no new `Msg`/`Effect` variant, so `effect_names_are_stable_snake_case` (`app.rs:2298`) passes untouched; `queued` non-empty ⇒ `running == true`; an empty queue renders a frame identical to today's (`Constraint::Length(0)`), so `line_count`/`max_off`/`scroll_offset` are unchanged.
**Tests:** `submit_while_running_queues_instead_of_sending`, `queued_message_drains_at_turn_idle_as_a_user_cell`, `turn_idle_drains_only_one_queued_message`, `tool_calling_llm_done_does_not_drain_the_queue`, `queue_is_empty_whenever_the_turn_is_idle`, `interrupt_drops_the_queue_with_a_recoverable_notice`, `run_failed_and_agent_exit_drop_the_queue`, `up_arrow_pops_the_newest_queued_message_for_editing`.

### T1.9 — Focus-gated notifications (S, no core) — *needs 0.5's focus state*

New `notify.rs`: pure `sequence()` + `sanitize_field()` and one thin `emit()`
called **only** from the main loop's effect pass (`main.rs:1368`, after
`terminal.draw()` returned — never from the agent thread).

- **Exactly one sequence per terminal, not three** (per §1). `notify::TerminalHints::from_env()` resolves `TERM`/`TERM_PROGRAM` to a single choice — OSC 777 where it is known-supported (kitty, WezTerm, Ghostty), OSC 9 for terminals that take that and not 777, BEL as the last resort — and the mapping is unit-tested per terminal id. Never OSC 777 *and* OSC 9: kitty/WezTerm/Ghostty implement both and would notify twice.
- `sanitize_field` strips C0 (`<0x20`), DEL, C1 (`U+0080..=U+009F`, incl. ST) and `;`, then caps at 120 chars — agent-controlled text (tool names, provider errors) reaches the terminal through this.
- Tri-state focus: `Unknown` **never** notifies (a terminal that does not report focus must behave exactly as today). Fire on turn-idle and on approval-request, only when `Unfocused`. Suppressed while `splash.is_some()`. Entry-agent only (`ENTRY_AGENT` gating already exists in `Msg::from_event`).
- At most one turn-end notification per turn (`was_running` guard dedupes `LlmDone`→`RunCompleted`).

**Invariants:** nothing written moves the cursor, alters the screen buffer, or emits a newline — OSC + BEL only, so the alt-screen frame is byte-identical; `notify` config field is a plain `bool`.
**Tests:** `notify_on_turn_idle_when_unfocused`, `focused_terminal_suppresses_notify`, `unknown_focus_suppresses_notify`, `notify_disabled_by_config_suppresses`, `at_most_one_notify_per_turn`, `tool_turn_does_not_notify`, `interrupt_does_not_notify`, `sub_agent_llm_done_never_notifies`, `approval_notifies_while_running_unfocused`.

### T1.7a — Syntax highlighting (M, no core)

`syntect = { version = "5", default-features = false, features = ["default-fancy"] }` (workspace dep); `SyntaxSet::load_defaults_newlines()` and `ThemeSet::load_defaults()` are infallible embedded dumps — **no I/O**.

- **Caching is mandatory, not an optimisation:** `terminal.draw` is at the top of the loop (`main.rs:1331`), so `transcript_lines` re-renders every `Cell::Agent` on **every keystroke and every 120 ms tick** (`ui.rs:151`, `:158-160`, `cells.rs:214`). Cache styled lines in an interior-mutable `App::md` (sanctioned precedent: `last_max_off: std::cell::Cell<u16>`, `app.rs:493`), swept per frame by a `begin_frame()` with **exactly one** caller (first statement of `ui::transcript_lines`).
- Key the cache by the **source text**, never by a bare hash — a collision would render another cell's content, a silent visible corruption no test would catch.
- Width-awareness: `Msg::Resize` is a no-op today (`app.rs:733`); either key on width or make Resize invalidate.

**Invariants:** highlighting never changes the characters of a code block (`all_text()` with and without a language tag must be equal) — this is what keeps the existing markdown assertions valid; emitted line count per block is identical to today's; `markdown.rs` performs no I/O; the reducer never reads or writes `App::md` (grep-checkable).
**Tests:** `fenced_rust_block_is_syntax_highlighted`, `fenced_block_without_language_keeps_flat_code_colour`, `unknown_language_falls_back_to_flat_code_colour`, `highlighting_preserves_code_characters_and_line_count`, `fence_info_string_takes_the_first_token`, `markdown_cache_hit_matches_the_uncached_render`, `markdown_cache_sweeps_entries_not_reused_next_frame`, `streaming_prefix_stays_cached_across_deltas`.

### T1.7b+c — Word-level diffs and `/diff` (L, no core) — *last; may slip to Wave 1.5*

- **Do not add the `similar` crate** — absent from `Cargo.lock`, and the project's standing precedent is hand-rolled (SSE parser maison; `levenshtein()` duplicated by hand). ~120 pure lines.
- `DiffLine` gains `emph: Vec<Range<usize>>` (sorted, non-overlapping, non-empty, char-boundary bounds). `diff_preview(tool_name, input, max)` keeps its **exact** signature (`cells.rs:94`) so the approval modal (`ui.rs:402`) inherits emphasis with zero edits.
- `/diff` = `git diff HEAD` + untracked, through the existing `DiffLine` renderer. Git is I/O → new `Effect` + `Msg` pair, executed in the main loop (`spawn_blocking`), never in the reducer. Bounded by the existing cap wording (`long_diff_is_capped_with_more_note`, `cells.rs:696`). Non-git cwd → a notice, not an error.

**Invariants:** a `DiffLine` with `emph.is_empty()` renders to exactly one span with today's content and style (unemphasised output bit-for-bit unchanged); `parse_unified` on text without any `@@` takes today's path bit-for-bit, so the `patch` tool cannot regress (core's parser requires `@@ `, `patch.rs:425`); `patch_parses_unified_diff_by_leading_char` and `long_diff_is_capped_with_more_note` pass **unchanged**.
**Tests:** `single_token_change_emphasises_only_that_token`, `two_disjoint_changes_are_both_emphasised`, `identical_lines_yield_no_emphasis`, `del_line_starting_with_a_comment_dash_is_not_dropped_as_a_file_header`, `hunkless_patch_text_takes_the_legacy_path_unchanged`, `binary_file_notice_survives_both_parser_branches`.

### C3 / T1.8 — Post-edit formatters (S, **the one core change**) — *parallel from the start*

New `crates/heartbit-core/src/tool/builtins/format.rs` (`FormatterConfig`,
`DEFAULT_FORMAT_TIMEOUT`, `PATH_PLACEHOLDER`) + one field
`BuiltinToolsConfig.formatters: Option<FormatterConfig>` defaulting to `None`.

The plan's mechanism was defective; the correct design is **stdin → stdout**, formatting the *content in memory before the single write*:

1. **The mtime law.** `record_read` is `FileTracker`'s only mutator (`file_tracker.rs:41`) and write/edit/patch call it immediately after writing (`write.rs:173`, `edit.rs:191`, `patch.rs:298`). Formatting the buffer *before* that single write means the recorded mtime already matches the final bytes — no refresh needed, no window.
2. **The snippet must match disk.** `format_edit_snippet(&new_content, …)` (`edit.rs:194`) is built from in-memory content; if the file were formatted after the write, the model would be shown text that no longer matches disk and its next `old_string` could miss.
3. **Symlink hardening must survive (F-FS-1).** `edit.rs:177-188` deliberately writes via `write_beneath_root`/`write_no_follow`. Shelling out to `rustfmt <path>` would reintroduce a follow-the-symlink write. The subprocess therefore gets **no path** — content in on stdin, formatted content out on stdout. This also bounds blast radius structurally: a formatter cannot touch any other file.

**Invariants:** exactly **one** write per tool call, still through `write_beneath_root`/`write_no_follow`; `check_unmodified` still evaluated **before** any subprocess spawns (a rejected edit costs nothing); **fail-open** — missing binary, non-zero exit, timeout, empty or non-UTF-8 stdout ⇒ the write succeeds unformatted; `BuiltinToolsConfig::default().formatters == None` ⇒ no subprocess, on-disk bytes and every tool-output string byte-identical.
**Tests:** `default_config_has_no_formatter_and_writes_bytes_verbatim`, `write_formats_content_before_the_single_write`, `write_then_edit_without_reread_passes_guard_after_formatting`, `formatter_nonzero_exit_never_fails_the_edit`, `missing_formatter_binary_is_silently_skipped`, `formatter_timeout_never_fails_the_write`, `large_content_through_formatter_does_not_deadlock`, `formatter_only_runs_for_configured_extension`, `formatter_extension_lookup_is_case_insensitive`.

---

## 4. Collisions to respect

- **`main.rs:320-324` / `:341-342` / the panic hook** — 0.4 and 0.5 both write them. One pass, one commit.
- **`translate()` `main.rs:1096-1111`** — 0.5 adds focus arms; 0.4 must not widen the `KeyEventKind` filter (D-5).
- **`build_engine`** (`main.rs:543-563`, already 19 positional params) and its single call site (`:1220-1240`) + `spawn_agent`'s snapshot (`:1186-1196`) — 0.3 appends a param. Transposition-prone: sequential only.
- **Builder chain `main.rs:974-1013`** — 0.1 replaces `:987`; 0.3 adds a conditional in `:999-1008`.
- **`summary_parts`** (`main.rs:616`) is **sent at `:958`, before the builder chain** — anything computed after it never reaches the "ready — …" line. 0.1 and 0.3 must load/compute before `:958`.
- **Six parallel lists**: `TuiConfig` + its hand-written `Default` (`config.rs:68-124`,`137-155`), `App` fields, `Effect`, `Effect::name()`, `SLASH_COMMANDS` (`app.rs:181`, count coupled at `app.rs:2517`), `handle_slash` (`app.rs:1218`).
- **Turn-idle × 4** (`app.rs:818/935/945/954`) — T1.3 drains and T1.9 notifies there: one edit, not two.
- **`DiffLine`** (`diff.rs:12-16`) — one consumer (`cells.rs:95-107`) + 5 tests reshape together.

---

## 5. Implementation order

1. **0.6** — disjoint; the `[d]` hint is 0.1's discoverability prerequisite.
2. **0.5 then 0.4, as one pass** — 0.5's panic-hook wrapper is the scaffold 0.4 extends; reversed it gets written twice.
3. **0.1** — depends only on 0.6.
4. **0.3** — adjacent to 0.1 so the builder chain / `App` / `Effect` / `TuiConfig` are each edited once.
5. **T1.3** after 0.5 — without bracketed paste a 5-line paste submits 5 messages, so testing the queue earlier is testing a bug.
6. **T1.9** immediately after T1.3 — same four turn-idle sites, needs 0.5's focus state.
7. **T1.7a then T1.7b+c** — last; b reshapes `DiffLine` + `cells.rs` + 5 tests.
8. **C3** in parallel from the start — heartbit-core only, independent of every TUI item.

---

## 6. Explicitly out of Wave 1

Item 0.2 LSP → **Wave 1.5** (D-1), together with its interrupt guard
(`runner.rs:2734`, no trait seam, manual-pty-only evidence). `/permissions
list|clear` and a `rules_for_decision` core extraction → later. Mid-session
sub-agent ruleset staleness → documented only. T1.3's core steer slot → C6/Wave 3.
Anything touching `initial_messages`/session reseed, checkpointing, `/rewind`,
`/branch`, Esc-Esc → Wave 2. `/context` breakdown → C8/Wave 2. Hooks,
skills-as-commands, background tasks → Wave 3. Ratatui 0.30 → research topic.
**Kept, because they are what make `/effort` non-cosmetic and non-crashing, not
creep:** `SubAgentConfig.reasoning_effort` and the Anthropic-fallback gate.

---

## 7. Verification

**Automated**, in this order — and per framework §7.5 the terminal-dependent half
below must **not** be claimed as proven by `cargo test`:

1. Reducer tests for every transition listed in §3.
2. Unit tests for every pure helper (escape-sequence bytes, focus gating, word-diff pairing, formatter table resolution, cache hit/sweep).
3. The core test pinning `runner.rs:2097` (0.1) and the nine C3 tests.
4. **The gate, workspace-wide** (lesson 2026-06-13 — `-p heartbit-tui` would miss 0.1's core test and T1.7b's `DiffLine` reshape):
   `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`

**Manual acceptance signal** — run in Kitty, Ghostty, WezTerm or foot (**not** tmux/screen: they eat the CSI-u push):

```bash
export HEARTBIT_TUI_CONFIG=/tmp/hb-w1/tui.toml   # isolates tui.toml AND permissions.toml
mkdir -p /tmp/hb-w1 && cd ~/projects/heartbit && cargo run -p heartbit-tui
```

1. Paste a 5-line block → **one draft** (one `Cell::User` on submit, not five submitted turns); Shift+Enter inserts a newline in place.
2. Shift+Tab until the status line reads `normal` (default is Yolo, so the modal is otherwise unreachable — D-2).
3. Ask for something that runs `bash`; the modal footer now shows `[d]`; press `a` → a notice names `bash` **and** the `permissions.toml` path.
4. Ask for a second `bash` action **in the same session** → **no modal** (this is the half that is broken today).
5. `/quit`, relaunch → the startup line reads `… · 1 learned rules`; the tool runs unprompted; `stat -c %a /tmp/hb-w1/permissions.toml` = `600`, one `[[rules]]` entry `tool = "bash"`, `action = "allow"`.
6. Force a panic → the shell is usable with **no `stty sane`**, no stray `[I`/`[O` on window focus, no paste-bracketing artifacts. Repeat step 1 in a legacy terminal (xterm) → no artifacts from the unconditional Kitty push (D-3).

Steps 1–6 cover 0.1, 0.4, 0.5 and the panic-restore invariant. 0.6, 0.3, T1.3,
T1.7, T1.9 and C3 rest on the automated tiers above.
