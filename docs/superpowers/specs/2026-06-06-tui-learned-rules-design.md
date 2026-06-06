# TUI Learned Lessons (Self-Improvement Rung 2) — Design

**Date:** 2026-06-06
**Status:** Approved (brainstorming complete; user approved design + spec-and-implement)
**Scope:** `crates/heartbit-tui` only — **zero heartbit-core changes**
**Builds on:** rung 0+1 (`docs/superpowers/specs/2026-06-06-tui-debug-trace-design.md`, shipped)

## Problem

`/analyze` produces diagnosis reports with ranked, concrete recommendations — and they
die in a markdown file. Nothing feeds them back into the agent. Rung 2 closes the loop:
diagnosis findings become **persisted lessons** that are **injected into the agent's
system prompt at startup**, so the tool's behavior actually improves from its own traces.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Rule form | **Prose lessons** injected into the system prompt. Machine-actionable config (auto-allow permission rules, settings) = rung 2b, after the loop proves itself |
| Trigger | **Explicit `/learn`** — the human gate is the safety property; no analysis silently rewrites standing instructions |
| Scope | **Global**: `~/.config/heartbit/lessons.md` — the TOOL learns from its own operation; one file, every session, human-editable, 0600 |
| Distillation | An **agent task** (LLM merges/dedupes/rewrites — the part worth an LLM), not regex extraction; not a memory backend (no persistent store in the TUI; would force a core change) |

**The constraint that shapes the mechanics** (same as `/analyze`): workspace-rooted
builtins reject absolute paths — the agent cannot read or write
`~/.config/heartbit/lessons.md`. So the **edge stages the file into cwd** before the
run and **commits it back** after.

## Architecture

Two halves, one new concept:

- **The lessons file** `~/.config/heartbit/lessons.md` — flat markdown list under a
  `# heartbit lessons` heading, ≤ 25 lessons, one line each, ranked. **Rewritten**
  (never appended) by each `/learn`. Human-editable; deleting it resets the tool.
- **Learn pipeline**: `/learn` → stage → agent distills → commit-on-RunCompleted.
- **Injection point**: `build_engine` appends the lessons to the existing
  `instructions` composition (the AGENTS.md seam).

## Components

### `crates/heartbit-tui/src/lessons.rs` (new)

- `lessons_path() -> PathBuf` — `<config-dir>/lessons.md` (next to `tui.toml`).
- `load_lessons() -> Option<String>` — size-capped read (≤ `LESSONS_MAX_BYTES` =
  16 KB); `None` if absent, empty, or over cap (over-cap also surfaces a notice at
  the call site).
- `validate_staged(path) -> Result<usize, String>` — exists, non-empty, ≤ cap,
  starts with `# heartbit lessons`; returns the lesson count (list items).
- `commit_lessons(staged: &Path) -> std::io::Result<()>` — atomic write to the
  global path: temp file 0600 → rename (the `config.rs` save pattern).
- `STAGED_LESSONS: &str = "heartbit-lessons.md"` — the cwd staging name
  (gitignored alongside the other cwd artifacts).
- `LESSONS_TEMPLATE` — initial content when no global file exists yet
  (`# heartbit lessons\n` + a comment line explaining provenance).
- `build_learn_prompt(staged: &str, diagnoses: &[String]) -> String` — the task
  template: read the staged lessons + the named diagnosis file(s); distill their
  Recommendations into durable, GENERAL lessons (not session trivia); merge with
  existing — dedupe, drop stale/contradicted; keep ≤ 25 lessons, one line each,
  ranked by impact; REWRITE the staged file with the `write` tool; keep the
  `# heartbit lessons` heading. Workspace-relative paths only.

### `/learn` command (`app.rs`, `msg.rs`)

- `SLASH_COMMANDS`: `("/learn", "distill /analyze findings into persistent lessons")`.
- `handle_slash` `"learn"` arm: **no-key guard first** (the `/analyze` lesson —
  open the key modal, don't start a phantom run), then `Effect::Learn`.
- `Msg::LearnReady { display, task }` / `Msg::LearnFailed(String)` — mirror
  AnalyzeReady/Failed.
- `Msg::LearnReady` carries `staged_digest: u64` (hash of the staged file at
  stage time, computed by the edge with `std::hash::DefaultHasher`).
- Reducer `LearnReady`: `Cell::User(display)`, `running = true`, `follow = true`,
  `seed_idle_squad()`, `Effect::SendInput(task)`, **and
  `self.learning = Some(staged_digest)`** (new `App` field, `Option<u64>`).
- **Commit state machine** (load-bearing: in the TUI, `RunCompleted` only fires
  when the whole multi-turn session ends — the TURN-idle condition is
  `Msg::LlmDone { had_tool_calls: false, .. }`, the same condition that flips
  `running`):
  - `LlmDone { had_tool_calls: false }` while `learning.is_some()` → take the
    digest, clear the flag, push `Effect::CommitLessons(digest)`.
  - `Msg::RunCompleted` while `learning.is_some()` → same (backstop: session
    ends right at the answer).
  - `Msg::RunFailed` → clear the flag, no commit.
  - **Esc-interrupt** (the reducer arm that pushes `Effect::Interrupt`) → clear
    the flag FIRST — the interrupt synthesizes an EndTurn that arrives as a
    text-only `LlmDone`, which must NOT commit a half-rewritten file. Clearing
    at Esc-time reliably precedes the synthetic `LlmDone` (same-thread reducer).
  - `Effect::CommitLessons(digest)` at the edge re-hashes the staged file: if
    unchanged since staging (agent answered without rewriting), skip with
    `Notice("lessons unchanged — nothing to commit")` instead of a misleading
    "updated" notice.

### Edge handling (`main.rs`)

- `Effect::Learn` (async, mirrors Analyze): stage `load_lessons()` (or
  `LESSONS_TEMPLATE`) → cwd `heartbit-lessons.md`; glob `heartbit-diagnosis-*.md`
  in cwd, newest ≤ 3 by mtime — none → `LearnFailed("no diagnosis found — run
  /analyze first")`; build prompt → `LearnReady`.
- `Effect::CommitLessons(digest)` (sync, cheap): re-hash staged file — unchanged →
  skip notice; changed → `validate_staged(cwd/heartbit-lessons.md)` → ok:
  `commit_lessons` + `Notice("lessons updated (N lessons) — apply on next
  start")`; err: `Notice("lessons NOT committed: <reason>")`, global file untouched.
- Both get `Effect::name()` arms (`"learn"`, `"commit_lessons"`) → traced as `ui`
  effect records automatically.
- `.gitignore` gains `heartbit-lessons.md` (alongside the other cwd staging
  artifacts).

### Injection (`build_engine`)

After the project-context load: `lessons::load_lessons()` → if `Some`, append to
`instructions`:

```text
\n\n## Learned lessons (self-improvement — /learn)\n<content>
```

…and send `Notice("loaded N learned lessons")`. Next-start semantics, like every
other TUI config change.

### Trace integration (closing the observability loop)

- The `/learn` run itself is traced (normal agent path).
- `session_started` gains `lessons_loaded: usize` (`#[serde(default)]` — the
  envelope evolution rule; readers of old traces see 0). `/stats` and future
  rung-3 measurement can correlate behavior deltas with lessons going live.

## Error handling

- No diagnosis files in cwd → friendly `LearnFailed` notice.
- Agent never rewrites the staged file (or writes garbage) → `validate_staged`
  fails → clear notice, global untouched.
- Global lessons file over cap / unreadable at startup → skip injection + one notice.
- Interrupted learn run → flag cleared, nothing committed.
- The lessons file is recoverable by hand: it's plain markdown the user can edit
  or delete (deleting resets the tool's learned behavior).

## Testing (TDD)

- `lessons.rs`: path; template; `validate_staged` (missing/empty/over-cap/wrong
  heading/ok + count); `commit_lessons` atomic + 0600; `load_lessons` cap.
- Prompt builder: embeds staged name + diagnosis names + cap + "rewrite" + no
  absolute paths.
- Reducer: `/learn` no-key guard; `LearnReady` send-path + `learning` flag;
  `RunCompleted`+learning → `CommitLessons` + flag cleared; `RunFailed` clears
  without commit; `/learn` parsing.
- Injection: instructions composition includes the lessons section when the file
  exists (factor the composition into a testable pure helper if needed).
- **Live validation bar**: `/analyze` → `/learn` → approve the write → commit
  notice → relaunch → "loaded N learned lessons" → **behavioral proof**: plant a
  distinctive lesson (the Zorblax pattern) and verify the next session's answer
  reflects it; `session_started.lessons_loaded ≥ 1` in the new trace.

## Non-goals (this spec)

- Machine-actionable config changes (auto-allow permission rules, tool settings)
  — **rung 2b**, once the prose loop proves itself.
- Per-project lessons; auto-learn after `/analyze`; lesson provenance/rollback
  (the file is human-editable; git-less config dir); cross-session lesson
  EFFECTIVENESS measurement (**rung 3**: `/stats` deltas before/after a lesson).

## Ladder position

Rung 0+1 shipped (trace + `/stats` + `/analyze`). **This spec = rung 2.**
Rung 2b = machine-actionable config. Rung 3 = eval-measured improvement
(`TraceStats` deltas), then human-gated code patches.
