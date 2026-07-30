# TUI SOTA Upgrade — Framework Spec

**Date:** 2026-06-26
**Status:** approved (framework); sub-specs land per wave
**Basis:** commit `37bb8a5` (clean `main`) · research report `tasks/tui-sota-research-2026-06-26.md`
**Scope:** `crates/heartbit-tui` + the additive `heartbit-core` seams it requires

This is a *framework* spec. Its job is to fix the architecture, the core-change
ledger, the decisions, and the wave boundaries — nothing else. Each wave gets its
own sub-spec, written when that wave starts. Detail written now for work three
waves out would be stale by then.

---

## 1. Why

The research pass (13 agents: 3 internal audits, 6 external studies of opencode /
Claude Code / Codex CLI / Gemini CLI / Aider / Crush, 3 adversarial verification
lenses, synthesis) produced 54 candidates; 41 survived verification, 13 were
killed — most of them because **heartbit already had them**.

heartbit's TUI is *ahead* of every competitor on one axis: self-observability and
safety plumbing (always-on JSONL trace, `/stats`, `/analyze`, the `/learn`
lessons loop, doom-loop caps, epoch-guarded respawn, panic-safe exit guard). It is
behind on five, in order of severity:

1. **Sessions are amnesiac.** `/resume` restores only the display transcript
   (`main.rs:1569-1579`); `Effect::RespawnAgent` starts a fresh agent with no
   history (`main.rs:1400-1419`). Four separate features (rewind, branch,
   backtrack, true resume) are blocked on this one missing primitive.
2. **YOLO by default with no undo.** `PermissionMode::Yolo` is the default
   (`app.rs:543`) yet there is no checkpointing, no `/rewind`, no `/diff`, and no
   git awareness anywhere in the TUI.
3. **The input plane is below table stakes.** Bracketed paste is never enabled
   (only `EnableMouseCapture`, `main.rs:324`) although the crossterm feature is
   compiled in (`Cargo.toml:78`) and the `Event::Paste` arm exists
   (`main.rs:1101`); Kitty keyboard flags are never pushed, so the
   already-implemented-and-tested **Shift**+Enter newline (`app.rs:1817`, test
   `app.rs:3649`) never receives its modifier on most terminals. (Alt+Enter *does*
   work today — the branch is `if shift || alt` — so this is a missing second
   newline binding, not a total absence.)
4. **The reading surface is flat.** Code renders in one colour
   (`markdown.rs:10`), diffs are line-level only (`diff.rs`), no transcript
   search, no expandable tool output.
5. **The TUI trails its own engine.** Zero occurrences of `learned_permissions`,
   `lsp_manager` or `reasoning_effort` in `crates/heartbit-tui/src` — all three
   are shipped, tested, and already wired by heartbit-**cli**.

Point 5 is the cheapest upgrade in the project: capability that exists and is
tested, one builder call away from the user.

---

## 2. Section A — Architecture principles (non-negotiable)

**A1. The engine stays in-process.** The agent keeps running on its dedicated OS
thread with the synchronous `std::mpsc` approval bridge. opencode's
client/server split is attractive but the in-process path is load-bearing and
live-validated; an `attach` mode could only ever be an opt-in *second* path
(out of scope, §5).

**A2. Every core change is additive.** Each new API carries a default that
reproduces today's behaviour **bit-for-bit**. No breaking public API: the
workspace ships 5403 green tests and they stay green.

**A3. The reducer stays pure.** `Msg → App → Effect` is the only state-mutation
path; effects are performed by the main loop. No I/O in the reducer — that purity
is what makes the ~300 TUI reducer tests possible, and it is a hard constraint on
every design below. (The one sanctioned exception pattern is interior-mutable
view-side caching, precedent: `last_max_off: std::cell::Cell<u16>`.)

**A4. Nothing working is deleted.** Upgrades are additive, or they replace a path
explicitly identified as dead. No "cleanup" refactors ride along.

**A5. One definition surface per concept.** Where core already has a registry —
Agent Skills discovery, the `template/{registry,variables,merge}` stack — we
*extend* it. We do not add a second, parallel markdown format for agents or
commands. (This is the specific trap in porting opencode's design wholesale.)

---

## 3. Section B — heartbit-core work ledger

Every item is **additive**; none is breaking. The invariant column is the
acceptance condition, not advice.

| # | Item | Wave | Invariant it must preserve |
|---|------|------|----------------------------|
| **C1** | `initial_messages(Vec<Message>)` history-reseed seam on `AgentRunnerBuilder`, mirrored on the orchestrator entry agent | 2 | **Named acceptance criteria:** (a) a seeded history survives **one full compaction cycle** with absolute message indices correctly re-anchored (regression source: lessons 2026-06-09); (b) with `initial_messages` unset, behaviour is **byte-identical** to today; (c) alternating-role invariants hold for any seeded transcript, including one ending on a tool call |
| **C2** | Ranged compaction entry point ("summarize from / up to here") | 2 | Reuses the existing estimate-aware compaction path (`runner.rs:372-379`); automatic compaction behaviour unchanged |
| **C3** | Post-edit formatter hook in `write`/`edit`/`patch`, gated by `BuiltinToolsConfig` | 1 | Formatting happens **in memory, before the single write** (stdin→stdout, the subprocess never receives a path): that keeps `FileTracker`'s post-write `record_read` matching the final bytes, keeps the returned snippet consistent with disk, and preserves the F-FS-1 `write_beneath_root`/`write_no_follow` symlink hardening. A formatter failure never fails the edit; the default configuration is a no-op |
| **C4** | Skill invocation: `$ARGUMENTS` / `$N` / named substitution; opt-in `` !`cmd` `` pre-render; frontmatter `allowed-tools` as a one-turn grant via `PermissionRuleset::append_rules` | 3 | Skill-name path-traversal validation stays intact; the model-triggered skill path is unchanged; shell pre-render ships **default-off** |
| **C5** | `ShellHookGuardrail` (implements the existing `Guardrail` trait) + new firing points for SessionStart/SessionEnd/PreCompact | 3 | First-`Deny`-wins ordering; fail-open on timeout (precedent: `LlmJudgeGuardrail`); standalone-path scoping unchanged |
| **C6** | Steer slot checked at the tool boundary, sibling to `InterruptHandle` | 3 | Injected text becomes a correctly-ordered user message; never spliced mid-LLM-stream; interrupt semantics untouched |
| **C7** | Bash detach: mid-flight control seam, reaper task, `TaskRegistry`, `task_output` tool | 3 | Foreground teardown and process-group kill unchanged; interrupt still kills non-detached processes |
| **C8** | Context-breakdown API (per-section token estimates) | 2 | Uses the **same estimator as compaction**, so displayed numbers explain observed behaviour |
| **C9** | Content-bearing input seam (`Option<Vec<ContentBlock>>` beside `OnInput`) | 2–3 (opportunistic) | The existing `OnInput` signature is untouched |
| **C10** | Forced-delegation entry (`@agent`) beside `DelegateTaskTool` | 3 (opportunistic) | Flat hierarchy preserved: sub-agents still never spawn sub-agents |
| ~~C11~~ | Daemon parity for an attach mode | — | **Out of scope** (§5) |
| ~~C12~~ | Repo-map (tree-sitter symbol map) | — | **Out of scope** (§5) |

Wave 1 needs **C3 only**. That is deliberate: the cheapest wave carries the least
core risk.

---

## 4. Section C — Decisions

**C-1. Checkpointing is always on.** A shadow-git snapshot is taken before each
prompt, with no opt-in. Rationale: YOLO is the default posture, so the agent
already writes freely — the honest trade is to make everything undoable rather
than to ask for trust without offering a way back. A whole-tree shadow snapshot
(separate git-dir, work-tree = workspace) also catches edits made by `bash`,
which Claude Code's checkpointing explicitly does not. Cost is latency and disk
on large repos; the existing file-walker `SKIP_DIRS` list is the mitigation.
Rejected alternative: opt-in `/checkpoint` — the safety net would be missing
exactly when the user forgot to arm it.

**C-2. Hooks are framework-level.** `ShellHookGuardrail` lives in heartbit-core
and reads a project-level `.heartbit/hooks` configuration, so the TUI, the CLI
and the daemon all inherit it. Rationale: the core code is written either way
(the `Guardrail` trait is an exact fit for pre/post llm/tool), so scoping it to
`tui.toml` would duplicate the notion of project configuration and permanently
exclude the other execution paths. Rejected alternative: TUI-only hooks.

**C-3. Three things are out of scope, with reasons.** See §5.

**C-4. One definition surface per concept.** Restated from A5 because it is a
decision, not only a principle: user-invocable commands extend the **skill**
registry, and agent definitions extend the **template** registry. No parallel
markdown loaders.

---

## 5. Out of scope (and why)

- **Headless `attach` mode** (TUI as an HTTP/SSE client of the daemon). Strategic
  and genuinely attractive — heartbit already owns the hard half (Axum, SSE, WS,
  the A2H layer). But reaching parity plus replacing the synchronous `std::mpsc`
  approval bridge is a multi-week re-architecture (XL), and A1 says the
  in-process path stays the default. Revisit after Wave 3 with explicit appetite.
- **Repo-map** (Aider-style ranked symbol map). Requires a new core module with
  tree-sitter grammars in the build. We bet first on wiring the LSP that is
  already shipped (Wave 1, item 0.2), which covers most of the same grounding
  need at a fraction of the cost. Revisit only if LSP proves insufficient.
- **Bash sandboxing on by default.** Standing decision (2026-06-07): the bash
  escape hatch is **accepted** for local use. `SandboxPolicy` and the Landlock
  layer exist in core and stay available, but nothing is locked down without an
  explicit request from the maintainer.
- **`Viewport::Inline` / `insert_before` native-scrollback re-architecture.**
  Lines pushed into real terminal scrollback cannot be re-wrapped on resize; the
  current follow-bottom + visual-row scroll model works and is tested. Research
  topic, not roadmap.
- **Re-implementing anything on the shelfware list.** Granular permission rules,
  LSP diagnostics, reasoning-effort control, per-model context gauges and
  availability-based fallback chains (`CascadingProvider`) are all shipped and
  tested in core. A naive competitor-parity reading proposes rebuilding them;
  each is a builder call or a thin command away.

---

## 6. Section D — Wave boundaries

Each wave is a coherent shippable increment with exactly one acceptance signal —
a thing a human runs, not a test count.

### Wave 1 — Table stakes + shelfware
**Core:** C3 only. **Sub-spec:** `2026-06-26-tui-wave1-table-stakes-design.md`

Tier 0 (persistent approval rules, `/effort`, Kitty keyboard flags, bracketed
paste + focus events, two micro-defects) plus the visible input queue, syntax
highlighting, word-level diffs, `/diff`, post-edit formatters, and
turn-completion notifications.

> **LSP diagnostics moved to Wave 1.5** (sub-spec decision D-1): verification
> showed that wiring `.lsp_manager()` as-is ships a guaranteed 30 s stall with
> zero diagnostics (`lsp/server.rs:195-239`) plus a malformed `file://` URI
> (`server.rs:89`,`:148`), i.e. three more core changes of which two are not
> optional. Deferring is what keeps "Wave 1 needs C3 only" true.

> **Acceptance signal:** in a Kitty-protocol terminal, paste a 5-line block — it
> lands as a *single* draft and Shift+Enter inserts a newline; approve a tool
> with `a`, restart the TUI, and the same tool runs without prompting again.

### Wave 2 — Sessions become real
**Core:** C1, C2, C8 (+C9 opportunistically).

C1 history reseed → a `/resume` that actually restores context → always-on
shadow-git checkpointing with `/rewind` → `/branch`, Esc-Esc backtrack,
`/context`.

> **Acceptance signal:** kill the TUI mid-task, `/resume`, ask "where were we?" —
> the agent answers from restored context; `/rewind` to an earlier prompt
> restores both the files and the conversation.

### Wave 3 — Extensible harness
**Core:** C4, C5, C6, C7 (+C10 opportunistically).

Skills as user-invocable slash commands → framework-level lifecycle hooks →
mid-turn steering → background tasks.

> **Acceptance signal:** a user-authored `SKILL.md` appears in `/` autocomplete
> with its arguments; a `.heartbit/hooks` post-edit entry makes an agent edit land
> `cargo fmt --check`-clean with zero LLM fixup turn; a `cargo test` run
> backgrounded with Ctrl+B reports back through `/tasks`.

---

## 7. Verification strategy

Per wave, in this order:

1. **Reducer tests** for every state transition (the reducer is pure, so this is
   cheap and it is where regressions actually get caught).
2. **Unit tests** for every pure helper (diff pairing, focus gating, formatter
   table resolution, escape-sequence construction).
3. **Core integration tests** for each C-item, each pinning its invariant from
   §3 as a named test — in particular C1(a) compaction re-anchoring and C3's
   `FileTracker` mtime refresh.
4. **The workspace gate**, unchanged: `cargo fmt --all -- --check` ·
   `cargo clippy --workspace --all-targets -- -D warnings` ·
   `cargo test --workspace`.
5. **The wave's single acceptance signal**, run by a human in a real terminal.
   Terminal-dependent behaviour (paste, Kitty flags, notifications, images)
   cannot be proven by `cargo test` and must not be claimed as proven by it.

---

## 8. References

- Research report with the verification ledger (41 survivors, 13 killed):
  `tasks/tui-sota-research-2026-06-26.md`
- opencode: <https://opencode.ai/docs> · <https://github.com/sst/opencode>
- Claude Code: <https://docs.claude.com/en/docs/claude-code>
- Codex CLI (Rust/ratatui): <https://github.com/openai/codex>
- Prior TUI specs: `2026-06-05-unified-entry-agent-design.md`,
  `2026-06-06-tui-debug-trace-design.md`,
  `2026-06-05-context-restore-on-demand-design.md`
