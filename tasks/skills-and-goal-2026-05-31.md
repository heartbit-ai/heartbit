# Skills v2 + Goal — design (2026-05-31)

Goal (user, 2026-05-31): bring heartbit-core **Skills** to SOTA per Claude Agent
Skills, and add a **`/goal`-equivalent** that works with dynamic workflows.
Research-first, TDD, advisor sign-off. Ship as **two separate PRs** (advisor
steer): Skills v2 first (well-specified, low-risk), then Goal (novel design).

## Authoritative specs

### Claude Agent Skills (from platform.claude.com docs, fetched 2026-05-31)
- A skill = a directory `skill-name/SKILL.md` + optional bundled scripts/resources.
- `SKILL.md` = YAML frontmatter + markdown body.
- Frontmatter fields:
  - `name` — **≤64 chars**, lowercase letters + digits + hyphens, **must equal the
    directory name**. Required.
  - `description` — **≤1024 chars**, says *what* the skill does AND *when* to use
    it (this is the trigger signal). Required.
  - `allowed-tools` — optional, comma-separated tool names; restricts tools while
    the skill is active.
  - `license`, `version`, `metadata` (key/value) — optional.
- SKILL.md body **< 500 lines**; split overflow into reference files.
- Three-level **progressive disclosure**:
  1. metadata (name+description) always in context;
  2. SKILL.md body loaded when the skill is triggered;
  3. bundled files loaded on demand (relative path; `reference/ scripts/ assets/`).
- Discovery: personal `~/.claude/skills`, project `.claude/skills`, plugins;
  project precedence over personal.

### Claude Code `/goal` (doc fetch failed — modeled from OBSERVED behavior in this session)
The Stop-hook condition operating on this very session is the reference impl:
- `/goal <text>` sets a **session-scoped** objective + a **Stop hook**.
- At would-be-stop time, the hook **judges** (LLM, against the transcript) whether
  the condition holds. If not, it **injects concrete "what remains" feedback and
  blocks the stop**, forcing continuation. If yes, it **auto-clears**.
- `/goal clear` clears early.
- Persists across turns and context compaction.

## Current state (mapped against live tree — CORRECTED)
TWO pre-existing skill systems, neither SOTA:
1. `tool/builtins/skill.rs::SkillTool` — ALREADY directory-based: reads
   `{dir}/{name}/SKILL.md`, walks up `.opencode/skills` + `.claude/skills` to the
   git root (depth-cap 8) + `~/.config/heartbit/skills`, lists sibling files,
   path-traversal-guarded, lists available skills on miss. ALWAYS in `builtin_tools`
   (no opt-in struct). GAPS: dumps SKILL.md raw (no frontmatter parse, no name-spec
   validation), and there is **no Level-1 catalog** — the model must already know
   the skill name (no discovery injected into the system prompt).
2. `template/skills.rs::load_skills(names)` — parses **TOML** frontmatter
   (`SkillFrontmatter{name,description,max_inject_tokens,tags}`), injects FULL skill
   content into the system prompt at config-resolution time for an explicit
   `AgentConfig.skills: Vec<String>`. 10 bundled skills. This is eager injection,
   the OPPOSITE of progressive disclosure, and TOML frontmatter ≠ Claude's YAML.

- `serde_yaml` IS a workspace dependency already (heartbit-ghost uses it) → adding
  it to heartbit-core adds no NEW workspace dep.
- Goal: none. Reusable seams: `evaluator.rs::EvaluatorOptimizerAgent` (generator +
  evaluator runners, regex `ACCEPT` acceptance, feedback-reinjection loop — the
  closest existing pattern); `LoopAgent.should_stop(&str)->bool` (sync, no LLM);
  the runner natural-completion branch (`runner.rs` ~1164: `tool_calls.is_empty()`
  → emit RunCompleted + return); `on_input` callback already re-enters the loop
  via `continue` — the exact seam to reuse; flow `WorkflowCtx` cancel/budget.
- `LlmProvider` (RPITIT `complete`), `BoxedProvider::{new,from_arc}`,
  `CompletionRequest{system,messages,tools,max_tokens,tool_choice,reasoning_effort}`,
  `Message{role,content:Vec<ContentBlock>}`, `ContentBlock::{Text,ToolUse,...}`,
  `StopReason::{EndTurn,ToolUse,MaxTokens}`, `AgentOutput{result,tool_calls_made,
  tokens_used,structured,estimated_cost_usd,model_name,tool_call_results}`.

## `/goal` spec (code.claude.com, fetched cleanly 2026-05-31)
- `/goal <condition>` = session-scoped **prompt-based Stop hook**. Setting it starts
  a turn immediately with the condition as the directive.
- After EACH turn: the condition + the conversation so far go to a **small fast model**
  (separate evaluator, Haiku by default — NOT the working model). It returns yes/no +
  a short reason. "no" → start another turn, reason injected as next-turn guidance;
  "yes" → clear the goal, record an achieved entry.
- The evaluator **does not call tools** — it judges only what the agent surfaced in
  the conversation. Condition ≤ 4000 chars; may include an "or stop after N turns"
  clause (the model self-reports progress against it).
- `/goal` (no arg) = status (condition, elapsed, turns, token spend, last reason).
  `/goal clear` (aliases stop/off/reset/none/cancel) clears early.
- Persists across `--resume`/`--continue` (condition carries; counters reset).
- CONFIRMS D3: independent fast-model judge over the transcript, re-inject the
  reason, bound with a turn cap, clear on satisfaction.

## Decisions

### D1 — Skills frontmatter: parse YAML with `serde_yaml` (already a workspace dep)
Claude's SKILL.md uses **YAML** frontmatter (`---` fenced), and `metadata` can be
an arbitrary nested map. `serde_yaml` is already a workspace dependency (used by
heartbit-ghost), so reusing it in heartbit-core adds NO new workspace dep and is
more robust than a hand-rolled `key: value` parser. Validate the parsed manifest
against the Claude name/description spec separately (name ≤64, `[a-z0-9-]`,
== dirname, no reserved words "anthropic"/"claude", no `<`/`>`; description
non-empty ≤1024, no `<`/`>`). Adversarial TDD covers malformed YAML + each rule.

### D2 — Skills model: SkillRegistry adds Level-1 disclosure over the existing SkillTool
The existing `SkillTool` already does Level-2 (load `{name}/SKILL.md` body) and
hints Level-3 (lists sibling files; the `read` builtin loads them). The SOTA gap is
Level-1 **discovery/disclosure** + frontmatter parsing/validation. So:
- New `skill` module (`tool/builtins/skill/` submodule: `manifest.rs`,
  `registry.rs`, keep `SkillTool` in place). `SkillManifest` = validated frontmatter
  (name, description, allowed_tools:Vec<String>, version, license, metadata:Map) +
  body + root dir.
- `SkillRegistry::discover(dirs)` reuses `SkillTool`'s search-dir logic, walks
  `dir/*/SKILL.md`, parses+validates each, dedups by name (project over personal),
  skips invalid with a warning (one bad skill must not break discovery).
- Level-1: `registry.catalog()` renders `name: description` lines injected into the
  system prompt (so the model knows which skills exist + when to use them — the
  whole point of progressive disclosure). Wire via the template/system-prompt path.
- Level-2: `SkillTool` GAINS frontmatter-aware output (strip the YAML, return the
  body + sibling list) and validates the name against the manifest spec. Behavior
  stays back-compatible for a SKILL.md without frontmatter (return raw, as today).
- Level-3: unchanged — `read` builtin on sibling files referenced by relative path.
- `allowed-tools`: parsed into the manifest; v2 surfaces it in the SkillTool output
  as guidance (a follow-up can gate tool dispatch while a skill is active).
- Reconcile system 2 (PROPOSED, pending advisor): keep `template/skills.rs`
  (TOML eager-injection + bundled skills) UNTOUCHED in PR1 — migrating its
  frontmatter TOML→YAML is a behavior change with its own blast radius. PR1 would
  be PURELY ADDITIVE: registry + Level-1 disclosure + YAML manifest parser + a
  frontmatter-aware SkillTool. A later PR reconciles the two formats.

### D3 — Goal lives in `AgentRunner`; flow composes it (THE fork, resolved)
- `GoalCondition { objective: String, judge: Arc<BoxedProvider>,
  max_continuations: u32 }` (+ builder). `max_continuations` is the goal's OWN cap,
  layered on top of `max_turns`, never conflated with it.
- `AgentRunnerBuilder::goal(GoalCondition)` (mirrors `max_identical_tool_calls`).
- Intercept ONLY the **natural-completion branch** in `runner.rs::execute()` (the
  `tool_calls.is_empty()` → emit RunCompleted → return path, ~runner.rs:1164):
  when the model returns a final answer AND a goal is set AND continuations remain
  → run an **independent** judge (separate `BoxedProvider`, impartial-evaluator
  system prompt, NO tools; sees the goal + the transcript/result). Satisfied →
  return `goal_met=true`. Not satisfied → inject a continuation `user` message with
  the judge's "what remains" reason, decrement the continuation cap, `continue`.
  Continuation cap exhausted → return `goal_met=false` (no infinite loop).
- **Other exits are NOT looped**: MaxTurns / Truncated / missing structured output
  return as today. The goal continuation cap layers on top of, and never overrides,
  `max_turns`. `AgentOutput` gains `goal_met: Option<bool>` (`None` when no goal).
- Goal recitation (optional): append the objective to the system prompt so it
  survives compaction (Manus-style).
- flow composition (a CLAIM TO PROVE, not assume): the flow `agent()` leaf builds a
  fresh `AgentRunner` in `run_one`; thread an optional goal from
  `WorkflowCtx`/`AgentCall` into `.goal()` there. MUST verify with a test that a
  goal-driven multi-turn leaf still (a) decrements the SHARED flow budget per
  continuation and (b) a budget breach mid-goal cancels the loop. If that test is
  hard to write, the composition is NOT free and the seam needs rework. **No second
  Goal implementation in flow.**
- Independence (anti over-report, per Illusion-of-Progress / WebJudge): the judge
  is a SEPARATE LLM call with an impartial-evaluator prompt — never the agent
  grading itself. **Test independence directly**: the judge provider receives a
  DIFFERENT request than the worker (distinct system prompt, no tools) AND a worker
  emitting "I am done" text does NOT satisfy the goal when the judge returns
  NOT-satisfied. Plus two mutation tests (always-satisfied → stops after 1 turn;
  always-unsatisfied → loops to the continuation cap, then returns goal_met=false —
  never infinite).

## Anti-vacuous-test rigor (advisor)
- **Goal judge mutation-test**: prove the test FAILS when the judge is wired
  always-satisfied (agent stops too early) AND always-unsatisfied (agent loops to
  the cap, not forever). Only then is the green meaningful.
- **Frontmatter parser adversarial RED first**: malformed YAML, missing
  frontmatter, name≠dirname, name with traversal/uppercase/overlong,
  description>1024, allowed-tools referencing unknown tools, body>500 lines (warn).
- Never assert a design fact from a partial/garbled read — full-file/subagent read
  first. (Session lesson: garbled chunked reads fabricated phantom bugs.)

## Plan
1. [research] bounded pass on goal-satisfaction judging (running) — fold into D3.
2. [skills] TDD: manifest parser → registry/discovery → progressive disclosure →
   config+CLI+builtins wiring → gate → PR → merge.
3. [goal] confirm D3 with advisor → TDD GoalCondition+judge in runner
   (mutation-verified) → flow composition → gate → PR → merge → final advisor.

## Verification
`cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D
warnings && cargo test --workspace` green at each PR. Mutation-verify the goal
judge and the occurrence-style invariants. Advisor sign-off before declaring done.

## Self-review hardening applied to Skills PR1 (#12) — 2026-05-31
NOTE: advisor() not yet successfully called this session (attempts were
interrupted); these were my OWN review findings, applied proactively. A real
advisor sign-off is still REQUIRED before merge per the goal.
1. **Unified discovery** (`skill/discovery.rs`): one canonical
   `search_dirs(root, extra)` + `catalog_for(root, extra)` used by BOTH the
   `SkillTool` load path AND the Level-1 catalog, so the catalog never advertises
   a skill the tool can't load. Proven by `catalog_and_search_dirs_agree`.
2. **Level-1 in the config path** (`resolve_agent_config` via `skill_dirs`) —
   DONE + tested. `BuiltinToolsConfig.skill_dirs` threads the same dirs into the
   `SkillTool` (`with_dirs`).
3. allowed-tools parsed + surfaced in tool output but NOT enforced — documented
   deferred non-goal for v1.

KNOWN GAP (be honest in the PR + to advisor): the **CLI/daemon run-path** Level-1
catalog injection did NOT land (channel dropped those edits; `catalog_for` is not
yet called from heartbit-cli). So a config-defined agent gets the catalog, but the
no-config single-agent CLI path does not yet. Options: (a) land CLI wiring on this
PR before merge, or (b) scope PR1 to the library mechanism + config path and do
CLI wiring in a fast follow-up. DECIDE WITH ADVISOR.

Goal (PR2) — my own design notes to raise with advisor:
- Independence test should be BEHAVIORAL (worker says "I am done", judge says NO →
  run continues) as primary; "different request" assertion secondary.
- FRONT-LOAD the flow-composition test before polishing the runner path.
- Continuation cap must NOT reset the turn counter; bound turns; always-unsatisfied
  loops to the cap, never forever.

Process: act → read ONE clean confirmation → state result. Don't end a turn on
"now I'll do X" narration. NEVER write "advisor said X" without an actual
advisor() call returning X.

## Skills PR1 (#12) — ADVISOR SIGN-OFF (2026-06-02, real advisor() call)
SIGNED OFF. The two real blockers from the adversarial review are fixed:
opt-in discovery (`catalog_from_dirs`, no ancestor/$HOME walk) closes the
prompt-injection vector; the `skill_dirs` resolve-gate is fixed at all three
sites. Manifest/registry/discovery are spec-faithful, mutation-verified
(precedence + body-absence), merged at `d7798c3`, full workspace gate green.
Reserved-word rule KEPT (the published overview doc mandates it; `contains`
matches "cannot contain"; divergence vs Anthropic's validate script documented).
allowed-tools-not-enforced + unknown-key leniency = accepted documented v1
non-goals. **Skills is done.**

## Goal — SOTA research synthesis (cited, 2026-06-02)
Primary source: Claude Code `/goal` doc — `/goal` is a session-scoped prompt-based
Stop hook; after each turn a small fast model judges the condition vs the
conversation (NOT calling tools), returns yes/no + reason; "no" → keep working
with the reason as guidance, "yes" → clear. Condition must be demonstrable by the
agent's own surfaced output.

arXiv corroboration (the "why independent judge" evidence):
- **"An Illusion of Progress? Assessing the Current State of Web Agents"**
  (arXiv:2504.01382, OSU-NLP 2025): web agents OVER-REPORT success; trusting the
  agent's own "done" claim inflates results. Their WebJudge (LLM-as-judge over the
  FINAL STATE, independent of the agent) reaches ~85% human agreement, 3.8% SR
  gap — far better than self-report. ⇒ judge the surfaced final state with an
  INDEPENDENT model, never the agent's self-assessment.
- **"Self-Preference Bias in LLM-as-a-Judge"** (arXiv:2410.21819, Wataoka 2024):
  LLMs systematically over-score their OWN outputs; bias persists even when
  authorship is hidden and scales with model size. ⇒ the goal judge must be a
  SEPARATE provider call with an impartial-evaluator prompt — a self-judge would
  over-report "done".
- **Mind2Web 2 / Agent-as-a-Judge** (arXiv:2506.21506): rubric/criteria-based
  independent judging of agentic task completion — supports criteria-conditioned
  judging over the transcript.
- Reflexion / Self-Refine (verbal self-critique loops): the re-injected judge
  reason is the "what remains" feedback that drives the next attempt; bounded by a
  continuation cap to avoid the known non-termination failure mode.

Design implications (all already in D3, now evidence-backed):
(a) INDEPENDENT judge, not self-judge [2410.21819]; (b) judge sees goal-as-criteria
+ transcript/final-state, NO tools [/goal doc, 2504.01382]; (c) bound both
premature-stop (judge gates the natural-completion exit) AND infinite-loop
(continuation cap + flow shared budget); (d) re-inject the judge's reason as a
user message [Reflexion]; (e) deterministic escape hatch = condition phrased as a
checkable output the agent surfaces [/goal doc].

## Goal PR2 — SHIPPED (2026-06-02, branch feat/goal-condition off main d7798c3)
`crates/heartbit-core/src/agent/goal.rs`:
- `GoalCondition { objective, judge: Arc<BoxedProvider>, max_continuations }` +
  `new()`/`with_max_continuations()`; `GoalVerdict { satisfied, reason }`.
- Independent judge: a SEPARATE provider call (`judge.complete`) with an
  impartial-evaluator system prompt and NO tools, over the agent's surfaced
  output. Verdict parser `GOAL_MET: YES | NO[: reason]`; unparseable/error →
  NOT-met (safe direction vs over-report). Cited rationale in the module doc
  (arXiv:2504.01382 WebJudge, arXiv:2410.21819 self-preference bias).
- Runner integration: `AgentRunnerBuilder::goal()`; `AgentOutput.goal_met:
  Option<bool>`. Intercepts ONLY the natural-completion branch (`tool_calls
  .is_empty()`): satisfied → return goal_met=Some(true); not-met & cap remaining
  → inject judge reason as a user message + `continue` (through the loop top, so
  it consumes a turn and respects max_turns — counter never reset); cap exhausted
  → goal_met=Some(false). MaxTurns/Truncated exits NOT looped.
- Flow composition (no second impl): `AgentCall::goal()` threads into `run_one`
  (bumps max_turns to fit continuations). The leaf's single `record_spend` carries
  the SUM across all continuations → shared-budget accrual; a breached budget
  aborts admission + fires run-wide cancel.

Tests (mutation-verified where it counts):
- goal_satisfied_stops_after_one_turn (always-YES → 1 turn).
- goal_unsatisfied_loops_to_cap_then_reports_false (always-NO, cap 2 → 3 turns,
  goal_met=false; NEVER infinite).
- worker_self_claim_does_not_satisfy_an_independent_no_judge (BEHAVIORAL
  independence: worker says "Done!", judge says NO → run continues).
- goal_continuations_respect_max_turns (low max_turns → MaxTurnsExceeded, bounded).
- no_goal_leaves_goal_met_none_and_one_turn.
- goal_driven_leaf_accrues_continuation_spend_into_shared_budget (flow: 3 turns
  × 15 = 45 on the shared budget).
- exhausted_budget_bounds_goal_leaf (flow: breach aborts the loop).
15 goal-module tests + 2 flow-composition tests. Full workspace gate green.

**DONE.** Both Skills (merged #12) and Goal are implemented, thoroughly tested,
and research-backed. Awaiting final advisor sign-off on Goal.

## Goal PR2 — adversarial review fixes (2026-06-02, DO-NOT-MERGE → addressed)
A 68k-token adversarial review rated termination PASS (solid) but DO-NOT-MERGE on
3 must-fixes — all addressed:
1. **Judge blindness (headline)**: the judge saw only `last_assistant_text()` (the
   agent's prose claim), not the tool-output EVIDENCE — undercutting the
   anti-over-report guarantee the doc claimed. FIXED: the judge now receives
   `ctx.conversation_text()` (renders `[Tool result: ...]` lines = evidence),
   tail-capped to 12k chars. Makes the WebJudge/"final state" framing true.
2. **Unaccounted judge tokens**: judge `response.usage` was dropped. FIXED:
   `evaluate()` returns `(GoalVerdict, TokenUsage)`; the runner folds judge usage
   into `total_usage`. Verified end-to-end by the flow test (worker 45 + judge 6
   = 51 on the shared budget).
3. **Solo-leaf mid-loop budget overshoot**: a solo goal leaf admitted once
   (pre-loop) and recorded once (post-loop), so it could overshoot the shared
   budget mid-loop (the test comment overclaimed the cancel-race covered it).
   FIXED: `run_one` caps the goal leaf's `max_total_tokens` at the budget
   remaining at leaf start; comment corrected.
Nice-to-haves also done: reject `goal + structured_schema` at build (was a silent
no-op); convergence test (NO→YES reaches the goal); `max_continuations=0` test;
`warn!` on judge-call failure. 19 goal-module + 2 flow-composition tests.
