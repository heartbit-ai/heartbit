# SOTA chrome-devtools-MCP Browser Bot in heartbit-core

Status: spec / implementation-ready
Audience: Heartbit core maintainer (expert Rust)
Date: 2026-05-31
Goal: build the best automated browser agent the framework can express, driven via the official `chrome-devtools` MCP server (stdio subprocess), reusing the agent runner, the dynamic-workflow combinator engine, structured typed output, memory, guardrails, and the MCP stdio client. The browser-control problem ("how to call Chrome") is already solved by the MCP. This spec is about the *architecture, capabilities, and reliability/safety layer* that make a browser agent actually work, and precisely which of those are already in heartbit-core versus must be built.

---

## 1. Executive Summary

A naive browser agent is a single ReAct loop that dumps every MCP tool plus a raw page dump at an LLM, asks it to click things, and trusts the model when it says "done." Every SOTA result in the corpus says that fails, and says *why*:

- **Grounding, not planning, is the bottleneck.** SeeAct (ICML'24) showed GPT-4V can describe the right action ~62% of the time under oracle grounding but the best *automatic* grounding (textual candidate selection over DOM/a11y handles) lands at ~40% step success — a ~20-27pt cliff caused purely by grounding. Picking an element by a **stable handle/id** from a finite candidate set beats emitting pixel coordinates (48.9% vs 15.1% element accuracy; arXiv:2401.01614). chrome-devtools-mcp's `take_snapshot` already hands the model exactly that: an accessibility tree where every interactable element carries a `uid`, and `click`/`fill`/`hover`/`drag` target by `uid`. **We get the SeeAct-winning grounding path for free.**
- **The I/O interface matters more than the scaffolding.** AgentOccam (ICLR'25) reached SOTA on WebArena (37.2% -> 43.1%, +161% over plain agents) by *pruning the observation* and *shrinking the action set* — no vision, no search (arXiv:2410.13825). This is the cheapest, highest-leverage thing to build.
- **The agent cannot tell whether its own action — or the whole task — succeeded.** "An Illusion of Progress" (COLM'25, arXiv:2504.01382) shows reported web-agent numbers are inflated; the #1 systemic weakness is false/premature "task done" and the #1 failure *class* is incomplete steps (filled a form but never clicked Submit). The fix is a separate verifier (WebJudge: key-points + key-screenshots, ~85.7% human agreement) — never trust the policy LLM's self-report.
- **Long-horizon execution is the binding constraint, not single-action grounding.** Success drops 31.6% easy->medium and another 15.4% medium->hard (arXiv:2504.01382). Plan/execute split + dynamic replanning is the answer: Agent-E (73.2% WebVoyager; arXiv:2407.13032) and Plan-and-Act (replanning adds +10.31%; arXiv:2503.09572).

The **5-7 highest-leverage things to build**, in priority order (cheap reliability wins first):

1. **Snapshot distiller** — prune/merge/Markdown-ify the a11y snapshot before it reaches the LLM (AgentOccam). Biggest reliability+token win per unit effort. *(NEW, small)*
2. **Forced fresh observation + stale-uid retry** — inject `includeSnapshot:true` on every mutating action; on "no element with uid" re-snapshot and retry once. *(NEW, small; mirrors existing tool-name-repair)*
3. **Post-action verification (state-diff)** — after each action, re-snapshot and confirm the page changed as intended; on no-op, replan. *(NEW, small)*
4. **Wait-for-stability / network-idle settle** — `wait_for` is text-only; synthesize readiness before snapshotting. *(EXTEND)*
5. **Plan/Execute split + stateful replan loop** — Planner (no browser tools, structured plan) -> Executor (chrome tools); conditional dynamic replanning. The split reuses existing combinators; the *stateful replan loop with a plan-progress channel* is the one real control-flow gap. *(split EXISTS; replan loop NEW)*
6. **WebJudge-style completion verifier** — key-point checklist + key-screenshot grading as the run stop-condition and eval gate. Depends on a small multimodal tool-result bridge. *(NEW)*
7. **Browser safety guardrails** — domain allowlist, label-aware destructive-action confirmation, page-prompt-injection / lethal-trifecta breaker. *(EXTEND existing guardrail hooks)*

**One-line verdict:** ~70% of what a SOTA browser agent needs is already in heartbit-core (uid grounding via MCP, ReAct runner, plan/execute combinators, structured output, memory tiers, guardrail hooks, doom-loop, budget, resume journal, and — code-verified — the LLM-side image content path); the genuinely NEW work is a thin **browser harness module** (snapshot distiller, settle, verify, stale-uid retry, multimodal tool-result bridge) plus a **stateful replan-loop combinator**, a **WebJudge completion verifier**, and **browser-specific guardrails** — all buildable without forking the MCP and without fine-tuning.

---

## 2. SOTA Landscape

### 2.1 Observation & grounding: a11y tree vs vision vs Set-of-Marks

The grounding question is settled and convergent:

- **Textual candidate-selection over DOM/a11y handles is the strongest real grounding.** SeeAct ranked three strategies on Mind2Web: Textual Choices 48.9% element accuracy >> Image Annotation / Set-of-Marks 15.1% >> free-form Element Attributes 4.7% (Oracle 72.9%) (arXiv:2401.01614). Set-of-Marks loses because GPT-4V hallucinates element references on cluttered full-page screenshots (~21% "visual illusion" errors) and labels obscure content.
- **Set-of-Mark (SoM)** converts grounding from coordinate-prediction into symbol-selection: segment/box interactable regions, overlay numbered marks, model emits the number (arXiv:2310.11441). It is a *fallback*, not a default: VisualWebArena measured SoM at only +1.3pts over a11y (18.9 vs 16.4; humans 88.7; arXiv:2401.13649). SoM is justified **only** for canvas/WebGL/iframe/image-map UIs the a11y tree cannot express. OmniParser (arXiv:2408.00203) is the blueprint when you truly need pure-vision interactable detection.
- **Native coordinate grounding** improved 2024->2025 (UI-TARS arXiv:2501.12326; Claude computer-use), but lives in the *model* + a coordinate API, not reachable through an a11y-tree MCP. For us the uid-textual path is the natural strength and correct default.
- **Verification channel is the inverse of the acting channel.** "Illusion of Progress" found a *screenshots+action-history* judge beats an a11y-tree judge and is more portable (not all sites expose a11y; a11y inflates latency/cost). So: **a11y (`take_snapshot` uids) for the acting policy; vision (`take_screenshot`) for the judge.**

**Implication for us:** `take_snapshot` is the observation of record. Never emit coordinates. Prune the snapshot (AgentOccam). Reserve screenshots for the completion judge and the narrow SoM fallback.

### 2.2 Control loops: ReAct vs plan-execute vs tree-search, + reflection

- **Plan/Execute hierarchy** is the highest-leverage architectural choice: a Planner that decomposes into abstract, element-id-agnostic subgoals + an Executor that binds uids from the live snapshot. Agent-E 73.2% WebVoyager (arXiv:2407.13032); Plan-and-Act 57.58% WebArena-Lite (arXiv:2503.09572); IBM CUGA ~61.7% WebArena frontier (arXiv:2506.03106).
- **Dynamic replanning** is the killer mechanism: re-invoke the planner with updated state after steps. Plan-and-Act: +10.31% over a static plan; the plan itself carries in-task state, substituting for hand-coded recovery. Its own stated improvement: make replanning *conditional* (executor signals "stuck/changed/failed") rather than every-step.
- **Reflexion** (arXiv:2303.11366) is the cheapest reliability multiplier: on failure, write a NL post-mortem to memory, recall on retry. Reused by LATS/WebPilot.
- **Tree-search / MCTS** (LATS arXiv:2310.04406; WebPilot AAAI'25 arXiv:2408.15978) gives the highest raw success but needs **backtracking / state-restore the real web cannot give** (no DOM+JS+cookie checkpoint via the MCP). **Do not build real-environment MCTS.**
- **Model-based simulation** (WebDreamer arXiv:2411.06559) is the safe substitute: use the LLM as a world model to *imagine* a candidate action's outcome, score it, execute only the best — most of tree-search's benefit without irreversible side effects. Gate to irreversible actions only.
- **Counter-current (AgentOccam):** representation beats search. Do the pruning + clean action set + scratchpad *first*, before any search/simulation.
- **Skill/workflow memory (AWM, arXiv:2409.07429):** induce reusable parameterized routines from successful runs (+24.6% rel Mind2Web / +51.1% rel WebArena, and *fewer* steps). Critical guardrail from its AWMAS ablation: workflows must be **in-context guidance the loop re-grounds each step, never rigid replay macros** (popups break fixed sequences; macro-use fell to 18.5%).

### 2.3 Leading OSS frameworks and their converged tool-set

- **browser-use:** DOM+a11y fused, filtered to visible+interactable, each element a numeric **index** the LLM references; optional SoM highlight overlay (`use_vision` toggle); a structured `current_state` reflection contract (`evaluation_previous_goal` / `memory` / `next_goal`) + an `action` array; `done(result)` terminal.
- **Skyvern:** specialist sub-agent **swarm** (interactable-element, navigation, data-extraction, password, 2FA/TOTP, autocomplete agents); re-derives a fresh handle every step (survives layout change). Maps onto our combinators + sub-agents.
- **Stagehand:** `observe()` returns structured candidate actions `{description, method, selector, arguments}` that `act()` replays with **zero LLM call** (self-healing on cache-miss); `extract({instruction, schema})` for typed data. DOM/a11y -> xpath, never pixels.
- **Agent-E:** planner + browser-navigation agent; DOM distillation into content-typed views (text_only / input_fields / all_fields) with a stable injected id (`mmid`); **"change observation"** (post-action DOM-diff + wait-for-stability); compound actions (`bulk_enter_text`).

**Converged minimal action vocabulary** (WebVoyager runs on 7; browser-use ~15; AgentOccam keeps 8): navigate, click[id], type/fill[id,text], hover[id], press_key, scroll (often dropped when the a11y tree is complete), select-option, tab switch/open/close, wait, extract, done/answer. **Every element-targeting action references a stable id, never coordinates.** chrome-devtools-mcp's ~26 tools are a superset; the agent should be shown a *curated subset*.

---

## 3. Capability Map

Status legend: **mcp-provides** = the chrome-devtools MCP tool covers it; **exists** = already in heartbit-core; **extend** = present primitive needs browser-specific wiring/config; **build-new** = net-new code in heartbit-core.

| # | Capability | SOTA rationale | Covered by | Status | Heartbit action |
|---|---|---|---|---|---|
| 1 | A11y-tree snapshot with stable per-element handles (PRIMARY grounding) | SeeAct: textual handle-selection 48.9% vs pixel SoM 15.1% el-acc. Every OSS framework grounds via a stable id, never coordinates. | `take_snapshot` (uid) + uid-addressed `click/fill/hover/drag`; **bundled `chrome-devtools` preset + `connect_preset` already turn these into `Vec<Arc<dyn Tool>>`** | mcp-provides | Call `tool::mcp_presets::connect_preset("chrome-devtools")` (already spawns npx + handshakes + stamps tools). System-prompt invariant: "a11y snapshot is the observation of record; never emit coordinates." No core code. |
| 2 | Observation pruning / interactable+viewport distillation | **PRIORITY-1.** AgentOccam: pruning the observation alone lifted WebArena 37.2->43.1 (+161% over plain), beating search machinery. Raw `take_snapshot` is close but not optimally pruned. | NEW (post-processor over snapshot text) | build-new | Add a snapshot-distiller: merge function-descriptive StaticText into the interactive element sharing its label, Markdown-ify table/list subtrees (drop columnheader/gridcell), drop non-interactable/redundant/off-viewport nodes, **PRESERVE uids**. Reuse the `tool_filter`/`ToolProfile` pattern. Do this first. |
| 3 | Multimodal tool-result bridge (carry screenshot bytes back into the loop) | Gates the WebJudge grader + SoM. Code-verified: LLM layer already supports images end-to-end (`ContentBlock::Image{media_type,data}` confirmed at types.rs:55; serialized in anthropic.rs + openai_compat.rs). The ONLY break is the tool-result path — `mcp_result_to_tool_output` (tool/mcp.rs:374) drops image blocks to a placeholder string at L395. | NEW (tool-result -> message bridge + MCP image passthrough) | build-new | Two narrow changes (NOT a new enum variant — `Image` exists): (a) let a tool result carry image bytes (add optional image payload to `ToolOutput`/`ToolResult`, or emit a sibling `ContentBlock::Image` alongside the text `ToolResult` in `runner.rs add_tool_results` — grep the exact site); (b) stop dropping image blocks in `mcp_result_to_tool_output` (`tool/mcp.rs:374`, placeholder at L395; the L2862 test that asserts the drop must be updated) — surface them instead of the placeholder. |
| 4 | Forced fresh observation + stale-uid re-snapshot discipline | Interaction tools default `includeSnapshot:false` (model flies blind). uids are snapshot-scoped; acting on a stale uid silently hits the wrong/no element — the #1 hazard of this server. | `includeSnapshot` param exists; discipline is runner-side | build-new | Inject `includeSnapshot:true` on every mutating interaction; on "no element with uid" error, re-snapshot + re-resolve target + retry once — mirror `find_closest_tool` tool-name-repair in `agent/mod.rs`. |
| 5 | Post-action verification via state-diff | **PRIORITY-2.** Agent-E "change observation"; Illusion-of-Progress "Incomplete Steps" is the #1 failure class. Even oracle grounding caps ~50% online -> verify-then-proceed mandatory. Code-verified gap: `post_tool` sees only `(&ToolCall, &str)` — cannot diff before/after. | NEW (agent-layer verify loop); primitives: `take_snapshot`+`evaluate_script`+console/network | build-new | Capture pre-action snapshot signature, execute, settle, re-snapshot, diff (URL? target gone/changed? expected element appeared? value set? new console error?). On no-op feed structured "observed change: no" to trigger retry/replan. Wire to `agent/events.rs` + episodic memory. |
| 6 | Wait-for-stability / network-idle settle | Playwright actionability is the reference contract. `wait_for` is TEXT-ARRAY only — no network-idle/DOM-stable/visible wait. Acting on a half-rendered SPA poisons the ephemeral-uid cadence. | `wait_for` (text) partially; readiness synthesized | extend | Composite "navigate/act -> settle -> snapshot": poll `list_network_requests` in-flight count and/or `evaluate_script(document.readyState + mutation-quiescence)`, bounded by timeout. Loop invariant: wait -> snapshot -> act -> verify. |
| 7 | Planner/Executor hierarchical split | Agent-E 73.2% WebVoyager; Plan-and-Act 57.58% WebArena-Lite. Planner emits abstract plan (no browser tools); Executor binds uids from live snapshot. | `SequentialAgent`/flow pipeline (`agent/workflow.rs`, `agent/flow/pipeline.rs`) + structured output | exists | Pipeline: Planner (structured ordered subgoals via `__respond__`, no chrome tools) -> Executor (chrome tools). No new core code for the split. |
| 8 | Stateful observe->act->verify->REPLAN loop + plan-progress channel | **PRIORITY-3 (main control-flow gap).** Plan-and-Act: replanning +10.31%; plan carries in-task state. NOT expressible today: `LoopAgent::should_stop` is `Fn(&str)->bool` (confirmed: builder workflow.rs:333, field L258, call site L302), runner consumes/returns text — no way to carry {step idx, remaining, action history, NEEDS_REPLAN} across turns. | NEW (extends combinator engine); Planner/Executor agents exist | build-new | Add a replan-loop primitive beside `LoopAgent`: snapshot -> (re)plan -> execute-ONE-step -> verify -> repeat, threading a typed plan-progress struct and a **conditional** replan trigger (executor emits NEEDS_REPLAN/stuck/step-failed). Budget/journal handle cost+resume but not plan state. |
| 9 | Error recovery via verbal self-reflection (Reflexion) | Cheapest reliability multiplier; reused by LATS/WebPilot. | Reflection memory (`memory/reflection.rs`) + runner retry | extend | On a failed attempt run a reflect step -> write Reflection MemoryEntry keyed to task; on next attempt recall + prepend. Pure wiring of existing memory-recall into per-attempt context. |
| 10 | Doom-loop / repeated-failed-action detection | Illusion-of-Progress: failed tasks ~2x steps, largely repeated actions + pop-ups. | `DoomLoopTracker`/`max_identical_tool_calls` (`agent/doom_loop.rs`) | exists | Enable + tune `max_identical_tool_calls` for the browser persona so repeated identical click/fill batches break into a replan. Config-only. |
| 11 | Long-context management (auto-compaction + history pruning) | Success drops 31.6%+15.4% easy->hard; Operator trajectories exceed 100 screenshots. AgentOccam dismisses earlier-plan observations. | Auto-compaction + session pruning (`agent/pruner.rs`) + episodic memory; tree-filtered replay is NEW | extend | Tune compaction/prune thresholds for browser trajectories; log completed subgoals to episodic memory. Optional: tree-path/pivotal-node filtered replay (AgentOccam). |
| 12 | Structured extraction (extract-with-schema) | Stagehand `extract`, browser-use `extract_structured_data`; most tasks terminate in a validated structured answer. | Structured typed output (`__respond__` + schema validation) | exists | Bind terminal answer/extraction to JSON schema. For data hidden in XHR, read via `list_network_requests`/`get_network_request`+`evaluate_script` rather than scraping the DOM. No core change. |
| 13 | observe()->act() resolution caching / self-healing replay | Stagehand: `observe()` -> `act()` replays with zero LLM call. CRITICAL: uids are ephemeral — cache a *resolution recipe*, not a uid. | Semantic memory (storage) + resume journal (`agent/flow/journal.rs`); resolve-recipe->uid shim is NEW | build-new | Cache keyed by (URL+instruction) -> recipe (`evaluate_script` CSS/XPath query OR element description to re-match against fresh `take_snapshot`), persisted in semantic memory. At replay: re-snapshot, re-resolve to current uid, execute; on miss fall back to LLM. |
| 14 | Agent Workflow Memory: induce reusable parameterized routines | AWM +24.6% rel Mind2Web / +51.1% rel WebArena (beats human-written-workflow SteP), fewer steps; margin GROWS under domain shift. Reflection tier summarizes lessons; it is NOT executable routines. | Memory exists but lacks a procedural tier; induction step is NEW | build-new | Add `MemoryType::Workflow` (procedural) holding (NL description + abstracted action sequence with `{placeholders}`); on an LLM-judge-confirmed success, run an induction step over the journal trajectory, write back. Retrieve top-k via BM25 recall (`memory/bm25.rs`) and inject as an "available routines" block for the Planner — **soft guidance, re-grounded each step (never macro-replay; AWMAS).** |
| 15 | WebJudge completion grading (key-point + key-screenshot outcome verifier) | **PRIORITY-4.** Illusion-of-Progress: #1 weakness is false "done." WebJudge ~85.7% human agreement, reliable to 80+ steps. Both the run stop-condition AND the eval gate. Code-verified gap: NO guardrail hook sees the trajectory OR screenshots. | `LlmJudgeGuardrail` (`guardrails/llm_judge.rs`) reusable as grader; trajectory+screenshot verifier is NEW; depends on row 3 | build-new | Add an on-completion/post-run verifier (NOT a per-call guardrail): (1) force task -> JSON key-requirement checklist via `__respond__`; (2) score trajectory screenshots, keep top-N (delta ~3); (3) reuse the judge model multimodally to mark each key point satisfied, gate success on ALL. Same verdict drives reflect/retry. |
| 16 | Programmatic success-predicate + uid-keyed trajectory recorder (replay/CI) | Eval forks: offline deterministic checkers (WebArena/Mind2Web; CI-able, exact) vs online LLM-judge. BrowserGym `bid` == chrome uid -> uid-space traces are replayable/machine-checkable. ST-WebAgentBench: score Completion-under-Policy (success AND zero violations). | Resume journal + observability spans (substrate); predicate + recorder NEW | build-new | Add (a) a programmatic success-predicate hook (closure over final snapshot/DOM/URL), preferred when a checker exists (LLM judge carries ~15% noise); (b) `TrajectoryRecorder` capturing (uid-space snapshot -> action+args) per step; (c) a policy-violation log scoring CuP, sharing the runtime guardrail policy objects. **REDACT credentials in the recorder.** |
| 17 | Set-of-Marks visual grounding (numbered-box overlay) — FALLBACK ONLY | DELIBERATELY LOW. SeeAct image-annotation 15.1% vs textual 48.9%; VWA SoM beats a11y by only +1.3; AgentOccam SOTA with pure text. uid<->geometry impedance: a JS overlay must also click via JS coords, bypassing the uid path. | NEW (overlay generator); depends on row 3; click via `click_at` (`--experimentalVision`) or JS coords | build-new | AFTER cheap wins: optional SoM gated to canvas/tree-invisible tasks — `take_screenshot`+`evaluate_script(getBoundingClientRect)` -> sparse numbered boxes (labels off content), return annotated image + number->box map, execute via JS coords. Strictly a fallback. |
| 18 | Multi-tab / page + navigation + dialog management | Real sites throw OAuth popups, new tabs, modal dialogs. browser-use/Skyvern handle as first-class. | `new_page`/`list_pages`/`select_page`/`close_page`, `navigate_page`, `handle_dialog` | mcp-provides | Wrap tools; loop convention detects new pages/pending dialogs after each action, surfaces as observations (auto-dismiss benign via `handle_dialog`, escalate consequential). |
| 19 | Safety: domain allowlist gating navigation/network/submission | Credentialed browser agent = lethal trifecta. WASP: frontier agents begin following injected page instructions ~17% of the time; gating off-allowlist navigation catches the dangerous first step. | `Guardrail::pre_tool` (sees `&ToolCall` incl. name + input Value, `agent/guardrail.rs` L155) | extend | Domain-allowlist guardrail: on `navigate_page`/submit/network tools, `Deny` when URL/host off-allowlist. Reuse first-Deny-wins stack + the domain-config already in heartbit-ghost. |
| 20 | Safety: prompt-injection-from-page + lethal-trifecta breaker | Every tool result is UNTRUSTED page content driving the next action. Principle: once untrusted input is ingested it must not trigger a consequential action. Code-verified: `injection.rs` exists, `post_tool` can `Modify(String)`, but the breaker needs trust-STATE carried ACROSS calls — current hooks are stateless per-call. | `injection.rs` + `post_tool` Modify (partial); cross-call trust-state NEW | extend | Stateful guardrail: (a) tag tool outputs carrying page content as untrusted; (b) once untrusted content + private data in scope, QUARANTINE the exfil leg — disable `evaluate_script`, restrict `navigate_page` to allowlist. Adopt a Dual-LLM quarantined-reader sub-agent (1-level nesting) returning only structured fields, never raw page text. Keep credentials OUT of context (drive a pre-authenticated session; never `fill_form` secrets). |
| 21 | Safety: destructive/irreversible-action confirmation (label-aware HITL) | Destructive-vs-safe is decided by ARGS + PAGE STATE, not tool name: `click(uid=X)` is identical whether X is "Delete account" or "Delete draft". Code-verified gap: the uid arg is OPAQUE — neither `OnApproval` nor `pre_tool` sees the resolved label. | `OnApproval` (`llm/mod.rs`) + `pre_tool` (arg-based) — both insufficient alone | extend | Thread last-snapshot element semantics (label/role/text for the target uid) into the approval/guardrail decision -> classify by reversibility/impact -> prompt 'About to click "Delete account" — confirm?'. Destructive-action classifier (pre_tool) auto-escalates `handle_dialog(accept)`, off-allowlist `navigate_page`, submits to delete/financial endpoints to `OnApproval`. |
| 22 | Safety: politeness / rate-limiting + robots-ToS | Production reliability needs per-step delay/backoff + robots/ToS respect; config of existing primitives. | `pre_tool` + `action_budget` guardrail (`guardrails/action_budget.rs`) | exists | Per-tool delay/backoff on nav/network; existing `action_budget` caps runaway browsing; robots/ToS = policy data the allowlist reads. |
| 23 | Bounded cost + deterministic resume across long sessions | Long trajectories + simulation are expensive; shared budget caps loops; content-addressed journal enables crash-resume without re-firing irreversible actions. | Flow budget (`agent/flow/budget.rs`) + resume journal (`agent/flow/journal.rs`) | exists | Wrap the browser workflow in a flow pipeline; shared budget bounds cost; journal records steps (also the raw input for trajectory recording + workflow induction). |
| 24 | Model-based simulation of irreversible actions (WebDreamer) — differentiating, optional | LLM as world model to imagine+score before committing; most of tree-search's benefit, avoids irreversible side effects; competitive with MCTS while safer. PREFER over MCTS (which needs state-restore the MCP can't give). | NEW (simulate-score-select); `LlmJudgeGuardrail` reusable as scorer; reversibility classifier from row 21 | build-new | Optional simulate-score-select for the small set of IRREVERSIBLE candidate actions: predict resulting a11y state, score imagined outcomes (typed `{score,rationale}` via judge), execute argmax, verify reality matched (reuse row 5). Reversible navigation skips simulation. AFTER cheap wins. **Do NOT build real-environment MCTS.** |

---

## 4. What chrome-devtools-mcp gives us for free

The MCP stdio client (`crates/heartbit-core/src/tool/mcp.rs`) already turns every tool below into `Arc<dyn Tool>`. **Raw browser control is free — no per-tool wrapping.** Tool surface (live server build), grouped by the capability it serves:

- **Observation (grounding):** `take_snapshot` -> text a11y tree, each interactable element carries a `uid` (capability 1). `take_screenshot` (png/jpeg/webp, fullPage, per-element via uid) -> for the judge + SoM fallback (capabilities 15/17). `list_console_messages`/`get_console_message`, `list_network_requests`/`get_network_request` -> readiness + failure signals + XHR/JSON extraction (capabilities 5/6/12).
- **Interaction (uid-addressed):** `click(uid,dblClick)`, `fill(uid,value)`, `fill_form([{uid,value}])` (checkbox/radio use 'true'/'false'), `hover`, `drag(from_uid,to_uid)`, `type_text(text,submitKey)` (keyboard into a *previously-focused* input — distinct from `fill`), `press_key('Enter','Control+A')`, `upload_file(uid,filePath)`. All target by `uid` only — no selectors (capability 1).
- **Navigation / pages / dialogs:** `navigate_page(url|back|forward|reload`, plus `initScript` run before page script, `ignoreCache)`, `new_page(background,isolatedContext)`, `list_pages`, `select_page`, `close_page`, `handle_dialog(accept|dismiss + promptText)` (capability 18).
- **Escape hatch:** `evaluate_script` — JS function string (may be async/Promise), returns JSON; **element handles are passed as PLAIN uid STRINGS in `args`** (`function:'(el)=>el.textContent', args:['<uid>']`), not `{uid}` objects. Used for settle/readiness, geometry (SoM), overlay detection, custom extraction, post-action verification (capabilities 5/6/13/17).
- **Sync:** `wait_for(text: NON-EMPTY ARRAY of strings, timeout?)` — resolves when *any* string appears. (capability 6, but see gaps.)
- **Emulation / perf / audit:** `emulate` (CPU+network throttle, geolocation, userAgent, colorScheme, viewport), `resize_page`, `performance_start_trace`/`stop`/`analyze_insight`, `take_heapsnapshot`, `lighthouse_audit` — relevant if the bot's job is site QA/measurement.
- **Behind `--experimentalVision`:** `click_at(x,y)` — the coordinate-click primitive needed for the SoM fallback (capability 17).

### What is MISSING (the build list) — none require an MCP change

1. **Per-element geometry / visibility flags.** `take_snapshot` returns no bounding boxes -> blocks viewport filtering and SoM overlays. Synthesize via `evaluate_script(getBoundingClientRect)`. Note the **uid<->geometry impedance mismatch**: geometry you extract is keyed to *your* selectors, not the opaque uid, so a JS overlay must also click via JS coords (capability 17).
2. **Opt-in observations.** Interaction tools default `includeSnapshot:false` -> the runner must inject `includeSnapshot:true` (capability 4).
3. **Network-idle / DOM-stable / element-visible wait.** `wait_for` is text-only -> synthesize readiness (capability 6).
4. **Snapshot pruning.** Raw tree is not optimally pruned -> distiller (capability 2).
5. **Post-action verification, stale-uid retry, completion grading, trajectory recording, safety policy.** All agent-layer logic (capabilities 4/5/15/16/19-21).
6. **Multimodal tool-result path.** Screenshots can't currently reach the model (capability 3).

---

## 5. What to build in heartbit-core

> **Verification status (read first).** All load-bearing symbols were **grep-confirmed against the live tree.** Confirmed: `ContentBlock::Image { media_type, data }` at `llm/types.rs:55`, and `ContentBlock::ToolResult { tool_use_id, content: String, is_error }` at `types.rs:45` — `content` is **String-only**, and `Message::tool_results` (`types.rs:105`) maps `ToolResult -> ContentBlock::ToolResult` dropping all non-string content; that is the exact image-bridge break. `ToolOutput` is `{ content: String, is_error: bool }` at `tool/mod.rs:19` (flat — capability 3 adds an optional image field cleanly). `mcp_result_to_tool_output` at `tool/mcp.rs:374`, drop placeholder at **L395**, drop-asserting test at L2862. `ctx.add_tool_results(...)` call sites at `runner.rs:1112,1321,1358,1403,1435,1581` (the synthetic-`ToolResult` build at L1112 is the image-bridge insertion point). `Guardrail::pre_tool` at `agent/guardrail.rs:155`, `post_tool` at L165, `post_tool_can_mutate_output` test at L286. `LoopAgent::should_stop` is `Fn(&str)->bool` — field `workflow.rs:258`, builder **L333**, call site L302. `find_closest_tool` exists (`agent/mod.rs`, tests at L3783+) — the retry pattern to mirror. **A `chrome-devtools` MCP preset is ALREADY bundled** (`mcp_presets.rs:56`, JSON at `mcp-presets/chrome-devtools.json`) with `npx` + `--headless --isolated` defaults and no env_keys (test L166), and **`connect_preset("chrome-devtools") -> Result<Vec<Arc<dyn Tool>>>` (`mcp_presets.rs:112`) already spawns + handshakes + stamps the tools.** The `browser/` module (§5.1-5.10) depends only on stable public primitives; the cross-cutting §5.0 edits are the only ones sensitive to line drift.

All new code lives in a single new module tree: **`crates/heartbit-core/src/browser/`**. It depends only on existing primitives (MCP client, runner, flow combinators, guardrails, memory, structured output). Public surface re-exported from `lib.rs`.

```
crates/heartbit-core/src/browser/
  mod.rs            // re-exports; BrowserAgent builder
  harness.rs        // BrowserHarness: connect preset -> Vec<Arc<dyn Tool>> + distill/settle/verify wrappers
  distill.rs        // snapshot distiller (capability 2)
  settle.rs         // wait-for-stability / network-idle (capability 6)
  verify.rs         // post-action state-diff (capability 5)
  observe_act.rs    // observe()/act()/extract() higher-level primitives + resolution-recipe cache (capabilities 12/13)
  replan.rs         // ReplanLoop combinator + PlanProgress state channel (capability 8)
  judge.rs          // WebJudge completion verifier (capability 15)
  trajectory.rs     // TrajectoryRecorder + success-predicate hook (capability 16)
  guard.rs          // browser guardrails: allowlist, destructive-classifier, injection breaker (capabilities 19-21)
  som.rs            // OPTIONAL Set-of-Marks fallback (capability 17) — phase B7
```

Plus three *narrow, cross-cutting* edits outside `browser/` (capability 3, image bridge):
- `tool/mod.rs` (where `ToolOutput` lives): add an optional image payload to `ToolOutput`.
- `tool/mcp.rs:374` (`mcp_result_to_tool_output`): surface MCP image blocks instead of the drop placeholder at L395; update the L2862 test that currently asserts the drop.
- `agent/runner.rs` (`add_tool_results` — grep the exact site): when a tool result carries image bytes, emit a sibling `ContentBlock::Image` (confirmed present and serializing — `types.rs:55`, plus anthropic.rs / openai_compat.rs request builders).
- For the chrome connection itself, reuse/extend the existing `tool/mcp_presets.rs` (already present) rather than hand-rolling the stdio spawn in `browser/harness.rs`.

### 5.1 Browser harness — connect the preset, expose tools

```rust
// browser/harness.rs
pub struct BrowserHarnessConfig {
    pub headless: bool,
    pub chrome_path: Option<PathBuf>,      // else MCP default
    pub mcp_command: McpStdioCommand,      // node + chrome-devtools-mcp entrypoint
    pub experimental_vision: bool,         // enables click_at for SoM
    pub settle: SettleConfig,
    pub distill: DistillConfig,
    pub allowlist: DomainAllowlist,
}

pub struct BrowserHarness {
    session: McpSession,                   // reuse existing MCP stdio client
    cfg: BrowserHarnessConfig,
}

impl BrowserHarness {
    /// Spawn the chrome-devtools-mcp subprocess, handshake, list tools.
    /// Reuse the EXISTING bundled preset: internally calls
    /// `tool::mcp_presets::connect_preset("chrome-devtools")` (already spawns
    /// npx chrome-devtools-mcp with --headless --isolated, handshakes, and
    /// returns Vec<Arc<dyn Tool>>). For non-default flags (--experimentalVision,
    /// custom chrome path) resolve the preset, mutate `McpPreset.args`, then
    /// `McpClient::connect_stdio`. Do NOT hand-roll the stdio spawn.
    pub async fn connect(cfg: BrowserHarnessConfig) -> Result<Self, Error>;

    /// Curated, browser-reliability-wrapped tools for an agent's base_tools.
    /// Each mutating interaction tool is wrapped to: inject includeSnapshot:true,
    /// settle-before-return, distill the returned snapshot, and record the step.
    /// take_snapshot is wrapped to distill. Returns the AgentOccam-style subset
    /// (navigate, click, fill, fill_form, type_text, press_key, hover, drag,
    ///  take_snapshot, take_screenshot, wait_for, handle_dialog, evaluate_script,
    ///  list/get network+console, page mgmt) — NOT the full ~26.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>>;
}
```

The wrappers are `Arc<dyn Tool>` decorators over the MCP tools — the harness owns the cross-cutting reliability behaviors (capabilities 2/4/5/6) so they apply uniformly without the LLM having to ask.

### 5.2 Snapshot distiller (capability 2)

```rust
// browser/distill.rs
pub struct DistillConfig { pub viewport_only: bool, pub markdown_tables: bool, pub drop_redundant: bool }

/// Transform raw take_snapshot text -> compact text, PRESERVING uids.
/// - merge function-descriptive StaticText into the interactive element sharing its label
/// - convert table/list subtrees to Markdown (drop columnheader/gridcell tokens)
/// - drop non-interactable / off-viewport / redundant nodes
pub fn distill_snapshot(raw: &str, cfg: &DistillConfig) -> String;
```

### 5.3 Settle / wait-for-stability (capability 6)

```rust
// browser/settle.rs
pub struct SettleConfig { pub timeout: Duration, pub idle_window: Duration, pub max_inflight: usize }

/// Poll list_network_requests in-flight count and evaluate_script(document.readyState
/// + a mutation-quiescence probe) until quiescent or timeout. Run before every snapshot.
pub async fn settle(session: &McpSession, cfg: &SettleConfig) -> Result<(), Error>;
```

### 5.4 Post-action verification (capability 5)

```rust
// browser/verify.rs
pub struct SnapshotSignature { url: String, interactable_uids: BTreeSet<String>, title: String, console_errors: usize }

pub enum ActionEffect { Changed(String /*summary*/), NoOp, Error(String) }

pub fn signature(snapshot: &str, url: &str, console: &[String]) -> SnapshotSignature;
pub fn diff(before: &SnapshotSignature, after: &SnapshotSignature) -> ActionEffect;
```
The harness wrapper calls `signature` before, executes, `settle`s, re-snapshots, `diff`s; on `NoOp`/`Error` it appends a structured `observed change: no` to the tool result so the loop replans. Emits `AgentEvent` (reuse `agent/events.rs`) and writes the step to episodic memory.

### 5.5 Stateful replan loop (capability 8) — the one real combinator gap

```rust
// browser/replan.rs
#[derive(Clone, Serialize, Deserialize)]
pub struct PlanProgress {
    pub steps: Vec<String>,        // abstract, element-id-agnostic subgoals
    pub current: usize,
    pub history: Vec<StepRecord>,  // (action, ActionEffect)
    pub signal: ExecutorSignal,    // Continue | NeedsReplan(reason) | Done(answer) | Stuck
}

pub enum ExecutorSignal { Continue, NeedsReplan(String), Done(String), Stuck(String) }

/// Like LoopAgent, but threads PlanProgress across turns instead of a bare &str,
/// and routes control back to the planner when the executor emits NeedsReplan/Stuck.
pub struct ReplanLoop<P: LlmProvider> {
    planner: Arc<AgentRunner<P>>,   // no browser tools; structured plan output
    executor: Arc<AgentRunner<P>>,  // chrome tools; emits one step + ExecutorSignal
    max_iterations: u32,
    replan: ReplanPolicy,           // Conditional (default) | EveryStep
}

impl<P: LlmProvider> ReplanLoop<P> {
    pub async fn run(&self, task: &str) -> Result<AgentOutput, Error>;
}
```
Why NEW: `LoopAgent::should_stop` is `Fn(&str)->bool` (confirmed: builder at workflow.rs:333, struct field L258, call site L302) and `AgentRunner` consumes/returns text — there is no way to carry `PlanProgress` or route control back to the planner. The replan loop sits *inside* a flow pipeline so it inherits the shared budget (`agent/flow/budget.rs`) and resume journal (`agent/flow/journal.rs`). The Planner/Executor agents themselves are plain `AgentRunner`s composed via the existing `SequentialAgent`/flow pipeline.

**Token accounting / mutability.** `Orchestrator::run` takes `&mut self` precisely to prevent unsound concurrent token accumulation (per project memory), so an executor that accumulates usage cannot be shared as a bare `Arc<AgentRunner<P>>` and mutated in place. Two clean options, pick one at implementation time: (a) have `ReplanLoop` *own* the planner/executor `AgentRunner`s (not `Arc`) and call them `&mut` sequentially per turn — the loop is single-threaded across turns anyway, so no `Arc` is needed; or (b) keep `Arc` but make each turn construct/borrow a fresh run-scoped accumulator and fold the returned `TokenUsage` (which is `Copy`, accumulate via `+=`) into `PlanProgress`/the shared flow budget after each step. Option (a) is simpler and matches the existing combinator ownership model; the API sketch above uses `Arc` only for illustration — prefer owned runners + `+=` folding of `AgentOutput.usage` into the flow budget.

### 5.6 Higher-level observe()/act()/extract() (capabilities 12/13)

```rust
// browser/observe_act.rs
pub struct CandidateAction { pub description: String, pub method: String, pub recipe: ResolutionRecipe, pub args: Value }
pub enum ResolutionRecipe { Uid(String) /*this-turn only*/, Script(String) /*CSS/XPath via evaluate_script*/, Describe(String) }

/// observe(): LLM proposes candidate actions over the current distilled snapshot.
pub async fn observe(h: &BrowserHarness, instruction: &str) -> Result<Vec<CandidateAction>, Error>;

/// act(): execute a candidate. If recipe is Script/Describe (cached), re-resolve to a
/// CURRENT uid against a fresh snapshot (uids are ephemeral!) then execute with ZERO LLM
/// call. On miss, fall back to observe()+LLM. Self-healing.
pub async fn act(h: &BrowserHarness, action: &CandidateAction) -> Result<ActionEffect, Error>;

/// extract(): structured typed output run against the distilled snapshot / network JSON.
pub async fn extract<T: DeserializeOwned + JsonSchema>(h: &BrowserHarness, instruction: &str) -> Result<T, Error>;
```
The recipe cache (keyed by URL+instruction) persists in semantic memory; this is the Stagehand pattern made ephemeral-uid-safe. AWM workflow induction (capability 14) writes `MemoryType::Workflow` entries from journal trajectories on judge-confirmed success and the Planner recalls them via existing BM25 (`memory/bm25.rs`).

### 5.7 WebJudge completion verifier (capability 15)

```rust
// browser/judge.rs
pub struct KeyPoint { pub requirement: String, pub satisfied: Option<bool> }

pub struct CompletionVerdict { pub success: bool, pub key_points: Vec<KeyPoint>, pub rationale: String }

/// 3 stages: (1) up front, force task -> Vec<KeyPoint> via __respond__ (structured output);
/// (2) score trajectory screenshots, keep top-N (delta ~3); (3) reuse LlmJudgeGuardrail's
/// judge model MULTIMODALLY to mark each key point; success iff ALL satisfied.
pub struct WebJudge { judge: Arc<BoxedProvider> }
impl WebJudge {
    pub async fn key_points(&self, task: &str) -> Result<Vec<KeyPoint>, Error>;
    pub async fn verify(&self, task: &str, kps: &[KeyPoint], shots: &[ImageRef], history: &[StepRecord]) -> Result<CompletionVerdict, Error>;
}
```
This is an on-completion/post-run verifier, NOT a per-call `Guardrail` (the hooks see only `&str`/`&ToolCall`, never the trajectory+screenshots). It depends on the multimodal tool-result bridge (5.0/capability 3) to feed screenshots in. The same verdict drives the reflect/retry loop (capability 9).

### 5.8 Trajectory recorder + success predicate (capability 16)

```rust
// browser/trajectory.rs
pub struct StepRecord { pub snapshot_sig: SnapshotSignature, pub action: Value, pub effect: ActionEffect } // REDACT secrets
pub struct TrajectoryRecorder { steps: Vec<StepRecord>, journal: JournalHandle }
pub type SuccessPredicate = Arc<dyn Fn(&SnapshotSignature, &str /*final url*/) -> bool + Send + Sync>;
pub struct PolicyViolationLog(pub Vec<PolicyViolation>); // scores Completion-under-Policy
```
Records uid-space (snapshot -> action) pairs into the journal for replay/CI/offline re-grading. Prefer the programmatic `SuccessPredicate` whenever a checker exists (the LLM judge carries ~15% noise).

### 5.9 Browser guardrails (capabilities 19-21)

Reuse the `Guardrail` trait (`agent/guardrail.rs` L155, `pre_tool` sees `&ToolCall`):

```rust
// browser/guard.rs
pub struct DomainAllowlistGuard { allow: HashSet<String> }       // Deny off-allowlist navigate/submit/network
pub struct DestructiveActionGuard { classifier: ... }            // escalate handle_dialog(accept), delete/financial submits to OnApproval; LABEL-AWARE
pub struct InjectionBreakerGuard { trust: Arc<RwLock<TrustState>> } // stateful: tag page output untrusted; quarantine evaluate_script + off-allowlist nav once untrusted+private in scope
```
`DestructiveActionGuard` requires the harness to thread the last-snapshot element label for the target uid into the decision (the uid is opaque). `InjectionBreakerGuard` is the only one needing cross-call state (current hooks are stateless). Optionally pair with a Dual-LLM quarantined-reader sub-agent (1-level nesting) that returns only structured fields, never raw page text.

### 5.10 BrowserAgent builder (the assembled product)

```rust
// browser/mod.rs
pub struct BrowserAgentBuilder<P: LlmProvider> { /* provider, harness cfg, judge, guards, memory, budget */ }
impl<P: LlmProvider> BrowserAgentBuilder<P> {
    pub fn provider(self, p: Arc<P>) -> Self;
    pub fn judge(self, j: Arc<BoxedProvider>) -> Self;
    pub fn allowlist(self, a: DomainAllowlist) -> Self;
    pub fn success_predicate(self, f: SuccessPredicate) -> Self;
    pub fn memory(self, m: Arc<dyn Memory>) -> Self;
    /// Assembles: BrowserHarness.connect -> flow pipeline { Planner -> ReplanLoop(executor) }
    /// wrapped in shared budget + journal, with WebJudge as the terminal verifier and
    /// browser guards installed. Returns a runnable agent.
    pub async fn build(self) -> Result<BrowserAgent<P>, Error>;
    pub async fn run(&self, task: &str) -> Result<CompletionVerdict, Error>;
}
```

---

## 6. Reliability Playbook (concrete harness behaviors)

- **Loop invariant: `settle -> snapshot(distill) -> act(includeSnapshot:true) -> settle -> snapshot -> verify(diff)`.** Never act on a freshly-rendered page; never reuse a uid across a mutation.
- **Waiting:** before every snapshot, run `settle` (5.3): poll `list_network_requests` in-flight + `evaluate_script(document.readyState==='complete' && mutation-quiescent)`, bounded by `SettleConfig.timeout`. `wait_for([text])` only when waiting for a *specific* expected string.
- **Verification:** after every mutating action, `diff` signatures (5.4). On `NoOp`, inject "observed change: no" and let the replan loop retry/replan instead of marching on (attacks the Incomplete-Steps failure class). Read `list_console_messages` for JS errors as a negative signal.
- **Stale-uid recovery:** on a "no element with uid" tool error, re-snapshot, re-resolve the intended target (by label/role from the last distilled snapshot or the recipe cache), retry **once**; then escalate to replan. Mirrors `find_closest_tool` tool-name-repair.
- **Error-recovery / replanning:** `ReplanLoop` (5.5) re-invokes the Planner when the Executor emits `NeedsReplan`/`Stuck` or when `diff` reports repeated `NoOp`. The replanned plan carries in-task state (no separate memory module needed for *within-task* progress; Plan-and-Act). Reflexion (capability 9) handles *cross-attempt* learning.
- **Dynamic pages / SPA:** the `settle` + distill + re-snapshot cadence is the SPA defense; never cache the DOM.
- **Dialogs:** after each action, check for a pending dialog; auto-`handle_dialog(dismiss)` benign cookie/consent prompts (detect by text), escalate `accept` on destructive confirms to `OnApproval` (capability 21).
- **Cookie/consent banners:** detect a consent banner in the distilled snapshot and dismiss it *before* the first task action (a common stuck-cause).
- **CAPTCHA / 2FA:** do not attempt to solve; route to a human-handoff via `OnQuestion`/`OnApproval`. A dedicated 2FA/TOTP sub-agent (Skyvern pattern) may inject a known TOTP without exposing the secret to the model.
- **Multi-tab:** after each action, `list_pages`; if a new page appeared (OAuth/popup), surface it as an observation and let the planner decide to `select_page`. Use `new_page(isolatedContext:true)` for isolated sessions.
- **Auth:** drive an already-logged-in session via the MCP layer; **never `fill_form` credentials** (keeps secrets out of context and out of the trajectory recorder).
- **Doom-loop:** enable `max_identical_tool_calls` so repeated identical click/fill batches break into a replan (capability 10).
- **Long-horizon:** tune auto-compaction/session-prune (capability 11); log completed subgoals to episodic memory; the shared budget caps runaway browsing (capability 23).

---

## 7. Safety

The trust boundary is the whole game: **every chrome-devtools tool result is untrusted page content that immediately drives the next action** — the canonical lethal trifecta (private data + untrusted content + exfil channel via `navigate_page`/`evaluate_script`/form-fill). Map to existing guardrail hooks:

- **Domain allowlist (capability 19, `pre_tool`):** `DomainAllowlistGuard` returns `Deny` for `navigate_page`/submission/network targets off the allowlist. Catches the dangerous first step even when full exploitation is rare (WASP ~17% begin-to-comply). Reuse the heartbit-ghost domain-config.
- **Destructive-action confirmation (capability 21, `pre_tool` + `OnApproval`):** destructive-vs-safe is decided by ARGS + PAGE STATE, not tool name. Thread the resolved element label for the target uid (the uid is opaque) into the decision so the runner can prompt 'About to click "Delete account" — confirm?'. Classify by reversibility/impact; auto-escalate `handle_dialog(accept)`, off-allowlist `navigate_page`, submits to delete/financial endpoints.
- **Prompt-injection-from-page + lethal-trifecta breaker (capability 20):** the a11y snapshot can contain adversarial instructions. `injection.rs` detects; `post_tool` can `Modify` output to neutralize. The breaker (`InjectionBreakerGuard`, stateful) tags page output untrusted and, once untrusted content + private data are in scope, **quarantines the exfil leg** — disables `evaluate_script` and restricts `navigate_page` to the allowlist. The strongest structural defense is a Dual-LLM quarantined-reader sub-agent (1-level nesting) that returns only structured fields, never raw page text, to the privileged actor.
- **Credential handling:** never put secrets in context; drive a pre-authenticated session; redact credentials in the trajectory recorder (capability 16). This also breaks the "private data" leg for the secret itself.
- **Politeness / robots-ToS (capability 22):** `pre_tool` delay/backoff on nav/network + the existing `action_budget` guardrail; robots/ToS as policy data the allowlist reads.
- **Eval scores Completion-under-Policy** (ST-WebAgentBench): success AND zero policy violations, via the `PolicyViolationLog` sharing the runtime guardrail policy objects.

---

## 8. Testing Strategy (TDD-first)

**Unit (CI, no browser) — mock MCP over tokio duplex.** Build a `MockMcpServer` that speaks the MCP framing over a `tokio::io::duplex` pair (the existing MCP client already takes an `AsyncRead+AsyncWrite`). Script canned responses: a `take_snapshot` text fixture with known uids, a `click` that returns a *mutated* snapshot (to test verify/diff), a `click` that returns the *same* snapshot (to test NoOp detection), a "no element with uid" error (to test stale-uid retry), and an image `take_screenshot` block (to test the multimodal bridge). Assert:
- `distill_snapshot` merges StaticText, Markdown-ifies tables, drops redundant nodes, preserves uids.
- the harness wrapper injects `includeSnapshot:true` and runs `settle` before returning.
- `diff` returns `NoOp` for identical signatures, `Changed` otherwise; the wrapper appends "observed change: no" on NoOp.
- stale-uid error triggers exactly one re-snapshot+retry, then escalates.
- `mcp_result_to_tool_output` surfaces image blocks (not the placeholder); `add_tool_results` emits a sibling `ContentBlock::Image`.
- `ReplanLoop` routes back to the planner on `NeedsReplan`/repeated NoOp and threads `PlanProgress`.
- `DomainAllowlistGuard` denies off-allowlist `navigate_page`; `DestructiveActionGuard` escalates a "Delete" click given the label; `InjectionBreakerGuard` quarantines `evaluate_script` after untrusted+private in scope.
- `WebJudge.verify` returns `success:false` when any key point is unsatisfied (mock judge provider).

**Live smoke (gated, behind a `cfg`/env flag, not default CI).** Confirmed installed: node v22, `chrome-devtools-mcp`, google-chrome. Spawn the real MCP subprocess with headless Chrome, then:
- `navigate_page("https://example.com")` -> `settle` -> `take_snapshot` -> assert the distilled snapshot contains "Example Domain" and "More information" with a clickable `uid`.
- end-to-end: `BrowserAgentBuilder.run("Find the 'More information' link and report its URL")` -> assert `CompletionVerdict.success == true` and the extracted URL is `iana.org`.

**Grading trajectories:** record every smoke run with `TrajectoryRecorder` (uid-space snapshot->action), grade offline with the `SuccessPredicate` when a checker exists, else `WebJudge`. Replay recorded trajectories in CI without a browser to catch regressions.

---

## 9. Iterative Roadmap (each phase independently shippable, TDD-first, ordered by leverage)

- **B1 — Vertical slice (smallest end-to-end).** `BrowserHarness.connect` (wrapping the already-bundled `connect_preset("chrome-devtools")`) + `tools()` exposing the curated MCP subset + `distill_snapshot` (capability 2) + `settle` (capability 6) + a plain `AgentRunner` over the tools. Tests: mock-MCP unit (distill, settle, includeSnapshot injection) + ONE live smoke (navigate example.com -> distilled snapshot contains "Example Domain"). *Note:* the chrome preset + transport are DONE (task #23) — B1 is genuinely small. *Ships: a real agent that navigates + snapshots reliably.*
- **B2 — Verify + stale-uid retry (capabilities 4/5).** Post-action `diff`, NoOp detection, one-shot re-snapshot retry. Tests: mutated vs identical snapshot, stale-uid error. *Ships: the agent stops believing no-op actions.*
- **B3 — Multimodal tool-result bridge (capability 3).** The two narrow edits + image passthrough. Tests: image block surfaces; sibling `ContentBlock::Image` emitted. *Ships: screenshots reach the model — unblocks B5.*
- **B4 — Plan/Execute + ReplanLoop (capabilities 7/8).** `PlanProgress`, `ReplanLoop`, conditional replan trigger, inside a flow pipeline (budget+journal). Tests: replan routing, plan-progress threading. *Ships: long-horizon tasks.*
- **B5 — WebJudge completion verifier (capability 15).** Key-points + key-screenshots + multimodal grading as the stop-condition; reflect/retry on fail (capability 9). Tests: per-key-point gating with a mock judge. *Ships: trustworthy "done."*
- **B6 — Safety guardrails (capabilities 19-21).** Allowlist, label-aware destructive confirmation, injection breaker + Dual-LLM reader. Tests: deny off-allowlist, escalate destructive, quarantine exfil. *Ships: safe on the live web.*
- **B7 — observe/act recipe cache + AWM workflow memory (capabilities 12/13/14).** Self-healing replay + `MemoryType::Workflow` induction + BM25 recall injection. Tests: cache-hit zero-LLM replay, re-resolution against fresh snapshot, induction-on-success. *Ships: cheaper/faster repeat runs that compound.*
- **B8 — Trajectory recorder + eval harness + programmatic predicate (capability 16).** `TrajectoryRecorder`, `SuccessPredicate`, `PolicyViolationLog`, CI replay. *Ships: regression-graded CI + CuP scoring.*
- **B9 (optional) — WebDreamer simulation for irreversible actions (capability 24).** Simulate-score-select gated to the destructive set; verify reality matched. *Ships: safer commits on irreversible actions. Do NOT build MCTS.*
- **B10 (optional) — Set-of-Marks fallback (capability 17).** `som.rs` gated to canvas/tree-invisible tasks. *Ships: visual-only widget support. Strictly a fallback.*

---

## 10. Sources

- SeeAct — GPT-4V is a Generalist Web Agent, if Grounded (ICML'24): https://arxiv.org/abs/2401.01614 ; HTML https://arxiv.org/html/2401.01614v3 ; project https://osu-nlp-group.github.io/SeeAct/
- Set-of-Mark Prompting: https://arxiv.org/abs/2310.11441 ; repo https://github.com/microsoft/SoM
- VisualWebArena (ACL'24): https://arxiv.org/abs/2401.13649
- WebVoyager (ACL'24): https://arxiv.org/abs/2401.13919 ; HTML https://arxiv.org/html/2401.13919v3 ; repo https://github.com/MinorJerry/WebVoyager
- AgentOccam (ICLR'25): https://arxiv.org/abs/2410.13825 ; HTML https://arxiv.org/html/2410.13825v1 ; OpenReview https://openreview.net/forum?id=oWdzUpOlkX
- OmniParser (Microsoft): https://arxiv.org/abs/2408.00203
- UI-TARS: https://arxiv.org/abs/2501.12326
- Agent-E: https://arxiv.org/abs/2407.13032 ; HTML https://arxiv.org/html/2407.13032v1
- Plan-and-Act: https://arxiv.org/abs/2503.09572 ; HTML https://arxiv.org/html/2503.09572v2
- WebDreamer (Is Your LLM Secretly a World Model of the Internet?): https://arxiv.org/abs/2411.06559 ; repo https://github.com/OSU-NLP-Group/WebDreamer
- WebPilot (AAAI'25): https://arxiv.org/abs/2408.15978 ; HTML https://arxiv.org/html/2408.15978v1 ; proceedings https://ojs.aaai.org/index.php/AAAI/article/view/35663
- LATS (Language Agent Tree Search): https://arxiv.org/abs/2310.04406 ; repo https://github.com/lapisrocks/LanguageAgentTreeSearch
- Reflexion (NeurIPS'23): https://arxiv.org/abs/2303.11366 ; repo https://github.com/noahshinn/reflexion
- Agent Workflow Memory (ACL'25): https://arxiv.org/abs/2409.07429 ; HTML https://arxiv.org/html/2409.07429v1 ; repo https://github.com/zorazrw/agent-workflow-memory
- An Illusion of Progress? (Online-Mind2Web + WebJudge, COLM'25): https://arxiv.org/abs/2504.01382 ; HTML https://arxiv.org/html/2504.01382v4 ; repo https://github.com/OSU-NLP-Group/Online-Mind2Web ; dataset https://huggingface.co/datasets/osunlp/Online-Mind2Web
- WebArena: https://arxiv.org/abs/2307.13854 ; leaderboard https://huggingface.co/spaces/ServiceNow/WebArena-leaderboard
- Mind2Web: https://arxiv.org/abs/2306.06070 ; (Multimodal-Mind2Web) https://arxiv.org/abs/2410.05243
- Tree Search for Language Model Agents: https://arxiv.org/html/2407.01476v1
- GAIA: https://arxiv.org/abs/2311.12983
- CUGA (IBM, WebArena SOTA): https://arxiv.org/abs/2506.03106 ; blog https://research.ibm.com/blog/cuga-enterprise-ai-agent
- SteP (Stacked LLM Policies): https://arxiv.org/abs/2310.03720
- The Impact of Element Ordering on LM Agent Performance: https://arxiv.org/html/2409.12089v2
- Survey on LLM-based Web Agents (2025): https://arxiv.org/abs/2503.23350
- LLM-Brained GUI Agents: A Survey: https://arxiv.org/abs/2411.18279
- Design Patterns for Securing LLM Agents against Prompt Injections: https://arxiv.org/abs/2506.08837 ; summary https://simonwillison.net/2025/Jun/13/prompt-injection-design-patterns/
- The lethal trifecta for AI agents (Simon Willison): https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
- ST-WebAgentBench (ServiceNow): https://arxiv.org/abs/2410.06703 ; HTML https://arxiv.org/html/2410.06703v1 ; PDF https://arxiv.org/pdf/2410.06703
- WASP (Web Agent Security vs Prompt Injection): https://arxiv.org/abs/2502.20383 ; HTML https://arxiv.org/html/2504.18575v1
- The BrowserGym Ecosystem for Web Agent Research: https://arxiv.org/html/2412.05467v1 ; repo https://github.com/ServiceNow/BrowserGym
- chrome-devtools-mcp tool reference: https://github.com/ChromeDevTools/chrome-devtools-mcp/blob/main/docs/tool-reference.md ; announcement https://developer.chrome.com/blog/chrome-devtools-mcp
- browser-use agent architecture: https://docs.browser-use.com/customize/agent/architecture ; actions https://docs.browser-use.com/customize/actions ; repo https://github.com/browser-use/browser-use ; controller source https://github.com/browser-use/browser-use/blob/main/browser_use/controller/service.py
- Stagehand: act https://docs.stagehand.dev/basics/act ; observe https://docs.stagehand.dev/basics/observe ; extract https://docs.stagehand.dev/basics/extract ; repo https://github.com/browserbase/stagehand
- Skyvern: mechanics https://docs.skyvern.com/getting-started/skyvern-mechanics ; how-it-works https://docs.skyvern.com/introduction/how-skyvern-works ; repo https://github.com/Skyvern-AI/skyvern
- Anthropic Computer use tool: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
- Anthropic — Building Effective Agents: https://www.anthropic.com/engineering/building-effective-agents
- microsoft/playwright-mcp: https://github.com/microsoft/playwright-mcp ; comparison https://www.bsmichael.io/playwright-mcp-vs-chrome-devtools-mcp
- Playwright Actionability / Auto-waiting: https://playwright.dev/docs/actionability
- Notte — Reliable Browser Automation Through Self-Validating, Reusable Workflows: https://notte.ai/blog/reliable-browser-automation-2025
- Building Browser Agents: 7 Hard-Won Lessons: https://www.aitidbits.ai/p/building-browser-agents
- Reliable Agents in Production (Braintrust): https://www.braintrust.dev/blog/reliable-agents
- IBM — Human-in-the-Loop: https://www.ibm.com/think/topics/human-in-the-loop
- IETF Web Bot Auth Architecture: https://datatracker.ietf.org/doc/draft-meunier-web-bot-auth-architecture/
- Robots Exclusion Protocol: https://en.wikipedia.org/wiki/Robots.txt
- The State of Computer and Browser Use AI Agents in 2025 (Masterman): https://medium.com/@tula.masterman/the-state-of-computer-and-browser-use-ai-agents-in-2025-frameworks-datasets-and-benchmarks-for-66dc23a9d6f6
- Skyvern 2.0 SOTA Web Navigation (HN): https://news.ycombinator.com/item?id=43570076
