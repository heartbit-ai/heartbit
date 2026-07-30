All key claims verified against the tree (respawn is fresh-context at main.rs:1400-1419, resume is display-only at main.rs:1569-1579, core seams exist at the cited lines). Composing the report.

# SOTA Upgrade Analysis — heartbit TUI + heartbit-core
Basis: commit `37bb8a5` (clean main), full TUI audit (15 files), 3-lens-verified candidate set, claims spot-verified against the tree 2026-07-30.

---

## 1. Verdict

The heartbit TUI is **ahead of opencode / Claude Code / Codex CLI in exactly one dimension and behind in four**. Ahead: self-observability and safety plumbing — always-on JSONL trace + `/stats` + `/analyze` + `/learn` lessons loop, deterministic emergency briefs, mode-gated approvals with trace records, doom-loop caps, epoch-guarded respawn, panic-safe exit guard. No competitor has that ladder. The real weaknesses:

1. **Sessions are amnesiac — the one structural gap.** `/resume` restores only the display transcript (`main.rs:1569-1579` sends `Msg::SessionLoaded` and nothing to the engine); `Effect::RespawnAgent` deliberately spawns a fresh agent with zero history replay (`main.rs:1400-1419`). Every session-tree feature the competition has — Claude Code checkpoint/rewind, `codex fork`, opencode server-side sessions — is blocked on this single missing primitive: an engine history-reseed path. Four verified candidates (rewind, branch, backtrack, true resume) all hit this wall.
2. **No trust/undo story under a default-YOLO mode.** YOLO is the default (`app.rs:543`), yet there is no checkpointing, no `/rewind`, no `/diff`, no git awareness anywhere in the TUI. Claude Code's checkpointing is the #1 enabler for letting agents run; heartbit asks for the trust without offering the undo.
3. **Input plane is below table stakes, including one live bug.** Bracketed paste is never enabled — only `EnableMouseCapture` at `main.rs:324` while the crossterm `bracketed-paste` feature sits compiled-in (`Cargo.toml:78`) and the `Event::Paste` arm is dead code (`main.rs:1101`): a pasted newline arrives as Enter and submits half the prompt. Kitty keyboard flags are never pushed, so the already-implemented-and-tested Shift+Enter newline (`app.rs:1797`, test `app.rs:3649`) is unreachable in most terminals. No Home/End/word-nav; mid-run input is silently parked (`main.rs:861-881`).
4. **The reading surface is flat.** Code renders in one color (`markdown.rs:10`), diffs are line-level only (`diff.rs`, 163 lines), no transcript search, no expandable tool output (full output exists only in the trace file), tables and link URLs are dropped (`markdown.rs:110-223`).
5. **Extensibility is compile-time — and core is ahead of its own TUI.** `SLASH_COMMANDS` is a const (`app.rs:181-223`) with no user-defined commands, hooks, or agent files, while heartbit-core already ships Agent Skills, a template registry (`template/{registry,variables,merge}.rs`), and the exact Guardrail seam hooks need. Worse, the TUI wires **none** of core's shipped B2/D1-era features: zero hits for `learned_permissions`, `lsp_manager`, `reasoning_effort` in `crates/heartbit-tui/src`. The TUI trails its own engine.

---

## 2. Tier 0 — free wins (core capability exists; pure wiring)

| # | Wire what | Where | User-visible change |
|---|---|---|---|
| 0.1 | **Persistent approval rules** — `LearnedPermissions` (load/save 0600 TOML, size caps) exists in core (`permission.rs:159-292`), builder seam `.learned_permissions()` exists (`builder.rs:445-451`), runner already persists AlwaysAllow/AlwaysDeny live (`runner.rs:644,2120,2171`), and heartbit-cli already wires it (`heartbit-cli/src/main.rs:321-323`) | `build_engine` in `heartbit-tui/src/main.rs` (~5 lines next to `.permission_rules(default_permissions())` at `main.rs:987`) | Pressing `a` on an approval survives restarts; the audit's top persistence gap closes for free |
| 0.2 | **LSP diagnostics after edits** — full `LspManager` (rust-analyzer/tsserver/pyright/gopls/clangd, debounce, broken-server memo, ~1500 lines in `lsp/`) is shipped and wired into the runner (`runner.rs:2737`, `append_lsp_diagnostics` at `:3463`) behind `.lsp_manager()` | Same builder call site | Compile errors appear in the edit tool result; agent self-corrects without burning a `cargo check` turn |
| 0.3 | **Reasoning-effort control** — `ReasoningEffort` on `CompletionRequest` (`llm/types.rs:150-181`), Anthropic budget mapping, OpenRouter param, `.reasoning_effort()` threading all shipped | New `/effort low\|medium\|high` command + tui.toml key | Thinking-budget control per session |
| 0.4 | **Kitty keyboard flags** — Shift+Enter handling and its test already exist (`app.rs:1797`, `app.rs:3649`); only the ~5-line `supports_keyboard_enhancement()` probe + `PushKeyboardEnhancementFlags` push/pop (restore AND panic hook) is missing (kitty protocol; claude-code and codex both push it) | `main.rs` terminal setup at `:324` | Shift+Enter inserts newline in kitty/WezTerm/Ghostty/foot/alacritty/iTerm2 |
| 0.5 | **Bracketed paste + focus events** — feature flag already on (`Cargo.toml:78`), `Event::Paste` arm already written (`main.rs:1101`) | `EnableBracketedPaste` + `EnableFocusChange` next to `EnableMouseCapture` (`main.rs:324`) | Fixes the multi-line-paste-submits bug; focus tracking unblocks Tier 1 notifications |
| 0.6 | **Two audit-found micro-defects** | `app.rs:1837` (Ctrl+U replaces the whole `Composer`, discarding seeded prompt history — clear the draft only); `ui.rs:415` (approval hint omits the working `d` = AlwaysDeny key) | Correctness + discoverability freebies |

Not Tier 0 despite being shipped in core: **bash sandboxing** (`with_sandbox_policy` at `bash.rs:148`, `SandboxPolicy`/Landlock in `sandbox.rs`) — blocked by the standing 2026-06-07 decision that the bash escape hatch is ACCEPTED; see Open Question 2.

---

## 3. Tier 1 — high impact

**T1.1 Engine history reseed (the enabler — internal, no external source).**
TUI: on resume/respawn-with-history, convert persisted session turns into engine messages. Core: **additive** `AgentRunnerBuilder::initial_messages(Vec<Message>)` (mirrored on the orchestrator entry agent) that seeds the conversation before the first `on_input`. Effort **M**. Risk: must respect alternating-role invariants and compaction re-anchoring (lesson 2026-06-09: re-anchor absolute message indices at compaction). Nothing else in this tier's session family ships without it.

**T1.2 Checkpointing + /rewind (claude-code).**
Whole-tree shadow-git snapshot (separate git-dir, work-tree = workspace) taken in `submit()` before each prompt — submit-time is the sound hook: `on_approval` never fires in default YOLO and `Msg::ToolCallStarted` races the write (verifier finding). Esc-Esc/`/rewind` overlay lists prompts; restore code / conversation / both; conversation-restore = truncate + respawn via T1.1. Core: **one additive API — ranged compaction** ("summarize from here") extending the existing compaction path (`runner.rs:372-379`); file snapshots need no core change. Bonus over Claude Code: shadow-git catches bash-made edits, which Claude Code explicitly does not. Effort **L**, impact **critical**. Risk: snapshot latency on large repos (mitigate: skip-dirs list already exists for the file walker).

**T1.3 Visible input queue + mid-turn steering (claude-code).**
Queue mechanics already function — submissions while running sit in the unbounded channel (`main.rs:234`) and deliver at the next `on_input` (`main.rs:861-881`); they're just invisible. TUI: `pending_inputs` VecDeque rendered above the composer, Up-to-edit, drain at turn-idle (the same LlmDone digest guard `/learn` uses) — **S, zero core**. Steering into the *current* turn: **additive** core steer slot checked at the tool boundary, sibling to `InterruptHandle` — **M**. Risk: injection ordering; never splice mid-LLM-stream.

**T1.4 Skills as user-invocable slash commands with `$ARGUMENTS` + `!`cmd`` (claude-code).**
TUI: `/` falls through to the skill registry; autocomplete shows names + argument-hints. Core (**additive**): extend `template/variables.rs` substitution with `$ARGUMENTS`/`$N`/named; **opt-in, config-gated** pre-render shell injection; frontmatter `allowed-tools` as a one-turn grant via the existing `PermissionRuleset::append_rules` (`permission.rs:148`). Effort **M**. Risk: `!`cmd`` is an injection surface — ship it default-off; keep skill-name path-traversal validation intact.

**T1.5 Config-driven lifecycle hooks (claude-code).**
`ShellHookGuardrail` implementing the existing `Guardrail` trait (pre/post llm/tool — exact fit) with the JSON stdin / exit-2-blocks / stdout-decision protocol + timeouts. Core (**additive**): the guardrail impl + new firing points for session-start/end and pre-compact (no such events today). Guardrails run on the standalone path only — which is the TUI's path, so fine. Effort **M**. Risk: hook latency on hot pre_tool path — enforce timeouts, fail-open like `LlmJudgeGuardrail`.

**T1.6 Background tasks: Ctrl+B + /tasks (claude-code).**
Core (**additive, genuinely L**): bash detach seam — today `bash.rs` spawns `kill_on_drop(true)` + process-group kill on teardown (`:255-317`), so detach needs a mid-flight control seam, a reaper task, a `TaskRegistry`, and a `task_output` tool. TUI: Ctrl+B on the active tool cell, `/tasks` panel, completion notices. Impact high — converts the interrupt plumbing from abort button into productivity feature; essential for long `VerifyCommandTool` suites. Risk: highest-complexity item in the tier; do last.

**T1.7 Reading surface: syntect highlighting + word-level diffs + /diff (bat/git-delta/codex-cli).**
(a) Route fenced code through syntect with a per-cell styled-line cache — **mandatory**, not an optimization: the transcript re-flattens into one Paragraph every frame (`ui.rs:153-166`). **M, TUI-only.** (b) git-delta-style intra-line emphasis via the `similar` crate on paired Del/Add runs in `diff.rs` — **S/M, pure, table-testable.** (c) `/diff` = `git diff HEAD` + untracked, parsed through the existing `DiffLine` renderer — **S** (codex-cli). Risk: none beyond dep weight (syntect).

**T1.8 Post-edit auto-formatters (opencode).**
Core (**additive**): formatter hook in write/edit/patch — extension→formatter table (start: rustfmt), availability-gated, config in `BuiltinToolsConfig`. **Invariant: refresh `FileTracker` mtime post-format** or the read-before-write guard trips. Effort **S**. Payoff: heartbit's own quality gate rejects on `cargo fmt --check`; this deletes a whole class of wasted LLM round-trips.

**T1.9 Turn-completion / approval-wait notifications (claude-code, codex, opencode).**
OSC 777 + OSC 9 + BEL written raw to stdout at turn-idle and on approval-request, gated on focus (needs 0.5's `EnableFocusChange`). `notify = true` toggle. **S, TUI-only.** Daily payoff for long orchestrator runs.

---

## 4. Tier 2 — polish / differentiators

- **/context breakdown** (claude-code) — per-section token accounting; core additive API on the existing estimator/compaction accounting. **S.** Serves heartbit's #1 harness lever (context management) — promote to Tier 1 if wave 2 has room.
- **Plan-approval handoff** (claude-code) — detect plan-turn completion in Plan mode (LlmDone without tool calls), 3-option modal, flip mode, auto-send execute turn. **S, TUI-only** (Shift+Tab cycling + badge already shipped, `app.rs:1827`).
- **/branch + Esc-Esc backtrack** (claude-code `/branch`, codex-cli backtrack) — after T1.1 both are mostly UI glue over session copy + reseed. **S each, post-reseed.**
- **/btw side questions** (claude-code) — one-shot `BoxedProvider` call off-thread, modal render, `f` forks via /branch. **M.** Honest caveat per verifier: no prompt-cache reuse — full-price context tokens per ask.
- **Image input** (codex-cli et al.) — v1 = @-mentioned `.png/.jpg` paths. `ContentBlock::Image` exists (`llm/types.rs:55`) but `OnInput` is `String`-only (`runner.rs:33`), so core needs an **additive** content-bearing input seam. **M**, not S.
- **Agent-file surface + @agent forced delegation** (opencode) — extend the existing template registry (`template/registry.rs`) rather than a second format; new core piece = deterministic forced-delegation entry beside `DelegateTaskTool`. **M.** Directly attacks the proven organic-delegation failure (mid-tier models never delegate; nudge `47007b0` unvalidated).
- **Repo-map** (aider) — new core module: tree-sitter symbol map, centrality ranking, token-budgeted render. **L**, heavy build deps. The one candidate where nothing in core helps (`knowledge/` is document RAG, not a symbol map).
- **Headless attach mode** (opencode) — TUI as HTTP/SSE client of the daemon. **XL, strategic.** heartbit owns the hard half (Axum + SSE + WS + A2H layer); parity endpoints + replacing the sync `std::mpsc` approval bridge is a multi-week rearchitecture. Only with explicit appetite (Open Question 3).

---

## 5. Explicitly NOT recommended

- **Re-implementing anything from the shelfware list.** Granular permission rules, LSP diagnostics, reasoning effort, per-model context gauge + derived compaction, availability-fallback chains (`CascadingProvider`, `cascade.rs:123`) are all shipped and tested in core. The naive competitor-parity reading proposes rebuilding them; every one is a builder call or a thin command away (Tier 0).
- **Client/server split as the default TUI path.** opencode's architecture is attractive, but the in-process dedicated-thread engine with a sync approval bridge is load-bearing and live-validated; attach must be an opt-in second path, never a replacement.
- **Default-on bash sandboxing.** Standing user decision (2026-06-07): the escape hatch is ACCEPTED for local use. Strictly opt-in `/sandbox` exposing the existing `SandboxPolicy`, and only after an explicit green light.
- **A second agent-definition format** (opencode-style parallel .md loader) and **a parallel .md-commands loader** distinct from skills. Core already has the template registry (inheritance, skills injection, variables) and the skill discovery/manifest/registry; one definition surface each, extended — not duplicated.
- **Per-prompt file pre-images inside write/edit/patch** as the checkpoint mechanism. Strictly weaker than whole-tree shadow-git (misses bash-made edits) and adds core surface the shadow-git approach avoids. Also `FileTracker` is an mtime guard, not a snapshot hook — don't bend it into one.
- **Viewport::Inline / insert_before native-scrollback rearchitecture.** Lines pushed into real terminal scrollback cannot be re-wrapped on resize; the current follow-bottom + visual-row scroll model works and is tested. Research topic, not roadmap.

---

## 6. Consolidated heartbit-core work items (all additive; none breaking)

| # | Item | For | Invariant it must preserve |
|---|---|---|---|
| C1 | `initial_messages()` history-reseed seam on builder + orchestrator entry | T1.1/T1.2, Tier 2 branch/backtrack | Alternating roles; compaction re-anchors absolute indices; zero behavior change when unset |
| C2 | Ranged compaction entry point ("summarize from/up-to here") | T1.2 | Reuses the existing estimate-aware path (`runner.rs:372-379`); auto-compaction untouched |
| C3 | Post-edit formatter hook in write/edit/patch, `BuiltinToolsConfig`-gated | T1.8 | `FileTracker` mtime refreshed post-format; failure of the formatter never fails the edit |
| C4 | Skill invocation: `$ARGUMENTS`/`$N`/named substitution; opt-in `!`cmd`` pre-render; `allowed-tools` one-turn grant via `append_rules` | T1.4 | Skill-name path-traversal validation; model-triggered skill path unchanged; shell pre-render default-off |
| C5 | `ShellHookGuardrail` + new firing points (SessionStart/End, PreCompact) | T1.5 | First-Deny-wins ordering; fail-open on timeout; standalone-path scoping unchanged |
| C6 | Steer slot checked at tool boundary (sibling of `InterruptHandle`) | T1.3 | Injected text becomes a properly-ordered user message; never spliced mid-stream; interrupt semantics untouched |
| C7 | Bash detach: control seam, reaper (drop `kill_on_drop` for detached), `TaskRegistry`, `task_output` tool | T1.6 | Foreground teardown/process-group kill unchanged; interrupt still kills non-detached |
| C8 | Context-breakdown API (per-section token estimates) | T2 /context | Same estimator as compaction so displayed numbers explain observed behavior |
| C9 | Content-bearing input seam (`Option<Vec<ContentBlock>>` alongside `OnInput`) | T2 images | Existing `OnInput = Fn() -> Future<Option<String>>` signature untouched |
| C10 | Forced-delegation entry (`@agent`) beside `DelegateTaskTool` | T2 agents | Flat hierarchy: sub-agents still never spawn |
| C11 | (Conditional) daemon parity: session fork/resume, pending-permission reply, model listing, OpenAPI spec | T2 attach | In-process path remains default and unchanged |
| C12 | (Conditional) repo-map module (tree-sitter, ranking, budgeted render) | T2 repo-map | Feature-gated; no dep weight on default build |

---

## 7. Recommended sequencing — 3 waves

**Wave 1 — "Table stakes + shelfware" (all S, one M; core: C3 only).**
Tier 0 entire (0.1–0.6) + T1.3 queue visibility (TUI half) + T1.7 (/diff, word-level diff, syntect) + T1.8 formatters + T1.9 notifications.
*Acceptance signal:* in WezTerm, paste a 5-line block — it lands as one draft with Shift+Enter newlines working; approve a tool with `a`, restart the TUI, the same tool runs without prompting.

**Wave 2 — "Sessions become real" (core: C1, C2).**
T1.1 reseed → true `/resume` → T1.2 shadow-git checkpointing + `/rewind` (+ ranged compaction) → Tier 2 `/branch`, Esc-Esc backtrack, `/context`.
*Acceptance signal:* kill the TUI mid-task, `/resume`, ask "where were we?" — the agent answers from restored context; `/rewind` to an earlier prompt restores both files and conversation.

**Wave 3 — "Extensible harness" (core: C4, C5, C6, C7).**
T1.4 skills-as-commands → T1.5 hooks → T1.3 steer seam → T1.6 background tasks.
*Acceptance signal:* a user-authored SKILL.md appears in `/` autocomplete with arguments, and a `[hooks]` post-edit entry makes an agent edit land `cargo fmt --check`-clean with zero LLM fixup turn, while a `cargo test` run backgrounded with Ctrl+B reports back via `/tasks`.

Tier 2 items not named above (plan handoff, /btw, images, agent files, repo-map, attach) slot in opportunistically or per Open Questions.

---

## 8. Open questions for the maintainer

1. **Checkpointing default:** shadow-git snapshot on *every* prompt (always-on trust story matching the YOLO default, at disk/latency cost on big repos) — or opt-in via a `/checkpoint` toggle?
2. **Sandbox:** leave bash as-is per your 2026-06-07 accept-the-escape-hatch decision — or green-light the strictly opt-in `/sandbox` (existing `SandboxPolicy` + deny-then-escalate approvals, codex-style)?
3. **Headless attach appetite:** commit to daemon-parity + TUI-as-client (XL, multi-week, unlocks IDE/web clients) this cycle — or defer until after Waves 1–3 ship?
4. **Hooks scope:** TUI-only (`tui.toml [hooks]`) — or framework-level (project `.heartbit/hooks` honored by CLI and daemon paths too, since `ShellHookGuardrail` lives in core either way)?
5. **Repo-map dep weight:** accept tree-sitter grammars in the default core build — or feature-gate/defer and bet on LSP wiring (Tier 0.2) covering most of the grounding gap first?

---

## Appendix — verification ledger

13 agents (3 internal audits + 6 external research + 3 adversarial verify lenses + synthesis). 54 raw candidates -> 41 survived, 13 killed by >=2 of 3 lenses (existence / already-present-in-heartbit / Rust-feasibility).

### Killed (do not re-propose)
- Granular allow/ask/deny permission rules with glob patterns and per-agent overrides
- @-file fuzzy autocomplete and ! shell passthrough in the composer
- Session /undo and /redo backed by git snapshots
- LSP integration: post-edit diagnostics fed back to the agent
- Custom slash commands from markdown files with template variables
- Attention system: desktop notifications and sound when the agent needs you
- Shadow-git checkpointing + /restore
- Mid-turn steering + queued follow-up prompts
- Native terminal scrollback: inline viewport + insert_before
- Unfocused-terminal notifications for approvals and turn completion
- Availability-based fallback model chains
- Reasoning-effort controls: /effort + model variants cycling
- Per-model context-window awareness: gauge + derived compaction threshold

### Survivors (title [source] impact/effort)
- Headless server + TUI-as-client attach mode [opencode] high/XL
- Post-edit auto-formatters [opencode] medium/S
- Markdown-defined agents with Tab cycling and @mention subagent invocation [opencode] medium/M
- Checkpointing + /rewind (Esc Esc) with summarize-from-point [claude-code] critical/L
- Queued messages (type-ahead steering while the agent runs) [claude-code] high/S
- Background tasks: Ctrl+B to background a running command + /tasks view [claude-code] high/L
- User-defined slash commands from skills ($ARGUMENTS + !`cmd` dynamic context injection) [claude-code] high/M
- Hooks: user-configurable lifecycle shell commands [claude-code] high/M
- Plan-approval handoff (approve plan -> flip mode -> execute) [claude-code] medium/M
- /context breakdown (what is eating the window) [claude-code] medium/S
- /branch session branching [claude-code] medium/S
- /btw side questions (ephemeral Q&A overlay) [claude-code] medium/M
- /diff — cumulative working-tree diff view [codex-cli] high/S
- Esc-Esc backtrack: edit a past message and fork [codex-cli] medium/M
- Repo-map: ranked codebase context injection [aider] high/L
- Image input: paste or attach screenshots [codex-cli] medium/S
- Opt-in sandboxed bash with escalation approvals [codex-cli] high/L
- Streaming syntax highlighting in code blocks (syntect) [syntect (bat / git-delta / codex-cli)] high/M
- Shift+Enter newline via Kitty keyboard protocol [Kitty keyboard protocol (claude-code, codex-cli)] high/S
- Delta-style word-level diff highlighting [git-delta] high/M
- Turn-completion desktop notifications (OSC 9/777 + BEL, focus-gated) [OSC 9/777 (claude-code, codex-cli, opencode)] high/S
- Copy that works: OSC 52 clipboard + mouse-capture toggle [OSC 52 (codex-cli, crossterm)] medium/S
- Light/dark terminal adaptation via OSC 10/11 (terminal-colorsaurus) [terminal-colorsaurus (delta, bat)] medium/M
- Native-scrollback mode: Viewport::Inline + insert_before [codex-cli / ratatui Viewport::Inline] high/L
- Ratatui 0.30 upgrade: emoji/VS16 correctness, run() API, ecosystem unblock [ratatui 0.30] medium/M
- Inline images via ratatui-image (browser-agent screenshots in the transcript) [ratatui-image (Kitty/iTerm2/Sixel protocols)] medium/M
- Checkpointing + /rewind via shadow git repo [claude-code] critical/L
- Pattern-scoped approvals persisted as allowlists [opencode] high/M
- Visible message queue + steer-vs-interrupt levels [amp] high/M
- Session changes review + git ship UX [aider] high/M
- Compaction warning + /context breakdown + cost meter [claude-code] medium/S
- Live worker status panel (per-sub-agent tool + attach/stop) [claude-code] medium/M
- Doom-loop and retry surfacing as interactive prompts [opencode] medium/S
- Multi-session worktree dashboard [claude-squad] high/XL
- Shareable session pages (thread URLs / HTML export) [amp] low/M
- Native subscription OAuth login (/login for Claude Pro/Max + ChatGPT) [opencode, codex-cli] critical/L
- models.dev-style model catalog: pricing + capabilities + limits in one registry [opencode] high/M
- Quota / rate-limit display with reset countdown (/usage) [claude-code, codex-cli] high/M
- Live session cost display + budget caps [litellm, claude-code, aider] medium/M
- Local model auto-discovery (Ollama / LM Studio / vLLM in the picker) [opencode] medium/S
- Mode-bound models: opusplan-style plan/apply switching + small_model role [claude-code, opencode, aider] medium/M
