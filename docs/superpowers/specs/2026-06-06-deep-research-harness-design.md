# Deep-Research Harness — Design

**Date:** 2026-06-06
**Status:** Approved (brainstorming complete; user approved design + full cycle)
**Scope:** `crates/heartbit-core` (one recipe + registry entry) + `crates/heartbit-tui` (one command)
**Driving incident:** session `6a245538` — asked to "deep research" plate solving, the
agent's searches silently died on a bot-walled DuckDuckGo (now fixed: errors surface),
it fabricated URLs (8× 404) and drifted into implementation. There was no research
harness to route to: `default_registry()` only held `parallel_review`.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Trigger | **Recipe + `/research <question>`** — the recipe is registered (emergent routing possible), the command forces it deterministically with an imperative single-purpose order (the routing failure was the incident) |
| Output | **File + transcript** — report written to `research-<slug>.md` in cwd (gitignored, the `/analyze` artifact pattern) + a short summary in the conversation |
| Structure | **Staged flow recipe** (plan → parallel search/read → verify → synthesize), not a monolithic research agent (no structure = the failure mode) and not a TUI-native pipeline (wouldn't compose with CLI/daemon) |

## Architecture

Two deliverables, no new subsystems:

1. **Core**: `WorkflowRecipe` named `deep_research`, registered in `default_registry()`
   beside `parallel_review`. Every registry consumer (unified entry agent via
   `run_workflow`, CLI, daemon) gains it.
2. **TUI**: `/research <question>` slash command — reducer-only (no edge round-trip,
   simpler than `/analyze`: nothing to stage).

## The recipe (core)

**Args schema:** `{ question: string (required), angles: integer (optional, clamped
2..=6, default 4) }`.

**Stage 1 — Plan** (1 agent, talk-only): decompose the question into N complementary
research angles (definition/state of the art, algorithms/methods, existing
implementations, pitfalls/limits). Tolerant parsing (numbered lines or bullets,
trimmed); if parsing yields < 2 angles → deterministic fallback: `[question verbatim,
"state of the art: {question}"]`. This stage can never fail the workflow.

**Stage 2 — Search+Read** (`parallel` over angles): each angle agent gets ITS OWN
`WebSearchTool` + `WebFetchTool` instances via `AgentCall::tools` — the recipe
constructs them (`try_new()`), zero changes to `RunWorkflowTool` (whose ctx is
tool-less by design). Bounded per-angle prompt:
- 1-2 search queries for the angle, pick the 1-2 best sources, fetch them;
- extract findings with a `[URL]` citation per claim;
- output sections `FINDINGS:` / `SOURCES:`;
- **anti-fabrication rule (verbatim requirement):** "if search or fetch FAILS
  (blocked provider, 404), say so in FINDINGS — NEVER invent URLs or facts."
  (Made effective by the DDG bot-wall fix: failures now surface as errors.)

**Stage 3 — Verify** (1 agent, talk-only): cross-check the merged angle notes —
classify claims `CONFIRMED` (multi-source) / `SINGLE-SOURCE` / `CONTRADICTED`, list
gaps. No re-fetching in v1 (bounded).

**Stage 4 — Synthesize** (1 agent, talk-only): final markdown report — Summary,
Findings (each claim tagged with its confidence class + citations),
Contradictions & open questions, `## Sources` (deduplicated URL list). The recipe
returns the report **string** (recipe agents have no filesystem — by design).

**Failure semantics:** angle agents that error or return `None` are reported as
degraded coverage in the verify/synthesize stages; if ALL angles produce nothing,
the recipe returns `Err` → `run_workflow` tool error → the entry agent explains
honestly.

## `/research` (TUI)

- `SLASH_COMMANDS`: `("/research", "deep research — fan-out, verify, cited report")`.
- `handle_slash "research"`: empty arg → usage notice; no-key guard (open the key
  modal — the `/learn` precedent); then push `Cell::User("researching: <question>")`
  and the standard send path (`running`, `follow`, `seed_idle_squad`,
  `Effect::SendInput(task)`) with the imperative task:

  > Call the `run_workflow` tool now with name="deep_research" and
  > args={"question": "<question>"}. Do NOT search, browse, or implement anything
  > yourself before the workflow returns. When it returns, write the report
  > verbatim to `research-<slug>.md` (workspace-relative path) with the write
  > tool, then give a 5-10 line summary of the key findings and sources. If the
  > workflow returns an error, report it — do not improvise your own research.

- The slug: first 40 chars of the question, lowercased, `[^a-z0-9]+ → -`, trimmed
  of dashes; empty → `research`. Computed by the TUI (pure helper + test).
- `.gitignore` gains `research-*.md`.
- File write goes through the normal approval gate (one prompt in Normal, none in
  YOLO). If the user's question also asked for implementation, the entry agent
  proceeds AFTER the report, informed by it — the wanted research-first composition.

## Observability

`run_workflow` is an ordinary tool call: the whole research (dispatch, duration,
usage) lands in the session trace; `/stats` and `/analyze` see it. No new trace
events needed.

## Error handling summary

- Blocked search / missing API key → angle agents report it; degraded-coverage
  report instead of fabricated URLs; `ready — search: ddg-only` warns up front.
- Plan parse failure → deterministic fallback angles.
- All angles dead → recipe `Err` → honest tool error.
- Runaway bounds: angles clamped 2..=6, ≤ 2 fetches per angle, short-output prompts.

## Testing

- **Recipe** (scripted MockProvider, existing `workflow_tool` test pattern):
  - happy path: plan → 2 angles → verify → synthesize; report contains `## Sources`;
  - **tools wiring proof**: captured requests show websearch/webfetch tool defs
    present for angle calls and ABSENT for plan/verify/synthesize calls;
  - plan-parse fallback: garbage plan output → fallback angles used;
  - all-angles-dead → `Err`.
- **Angle-list parser**: numbered, bulleted, mixed, empty.
- **TUI reducer**: `/research` → display + imperative task (contains run_workflow,
  the slug filename, the no-improvisation rule); empty-arg usage; no-key guard.
- **Slug helper**: accents/spaces/length/empty.
- **Live validation bar**: real session with the Exa key configured (tui.toml seam
  shipped today) → `/research <small question>` → `research-*.md` exists with
  `## Sources` and real URLs, summary in transcript, full sequence in the trace.

## Non-goals (v1)

Search-result caching, contradiction-resolving re-fetch, recursive sub-questions,
PDF/HTML export, dedicated daemon endpoints (the registry already exposes the
recipe), report quality scoring (a rung-3 candidate: TraceStats on research runs).
