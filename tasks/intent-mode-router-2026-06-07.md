# Request-Intent Router + Mode Contracts

**Date:** 2026-06-07
**Branch context:** `feat/tui-streaming-markdown`
**Status:** DESIGN PROPOSAL (pre-implementation; TDD roadmap inside)
**User requirement (verbatim, FR):** *"notre app doit répondre de manière adaptée en fonction de la demande. on doit vraiment coller à la demande. ici la phrase n'était pas une demande explicite de créer tout de suite un crm. […] le harnais doit cadrer tout cela pour compenser des prises de décision du modèle un peu hâtives."*

The live incident: `je souhaite créer un petit crm` (a hedged, underspecified **wish**) was read as a settled imperative — the model unilaterally picked "web app" and started writing files. The harness must **frame** the request into the right response mode and **enforce** it, compensating for the model's haste.

---

## 0. CORRECTION FIRST — the load-bearing claim the orchestrator must absorb

A prior analysis (research finding #4) asserted a "VERIFIED GAP": *"there is NO `PermissionMode::Plan` branch anywhere that denies a tool at the execution layer … heartbit's toggled plan mode IS the Claude Code #19874 prompt-only bug."*

**This is FALSE.** Its grep was scoped to `runner.rs` and missed the enforcement, which lives in the TUI `on_approval` callback:

- `crates/heartbit-tui/src/main.rs:691` — `1 if calls.iter().any(|c| is_mutating(&c.name)) => { record(&ApprovalDecision::Deny, 0); return ApprovalDecision::Deny; }`. This fires **before** the tool executes. heartbit has real layer-2 Plan enforcement; it is **not** the #19874 bug. (Verified this session: `main.rs:685–696`.)

**The real gap is FRAGMENTATION, and it is sharper:**

1. **Two enforcement sites disagree on bash.** The plan-gate (`runner.rs:178`, `PLAN_GATE_MUTATING = ["edit","write","patch"]`) **excludes** bash on purpose (it gates *content* mutation: a build-without-a-plan trigger; `mkdir` writes no content). The Plan-mode deny (`main.rs:685`, `is_mutating = matches!(n, "edit"|"write"|"patch"|"bash")`) **includes** bash (it gates *side effects* in a read-only turn). Both are individually principled; they are not unified.
2. **`PermissionMode` is TUI-only.** Verified: `grep -rn PermissionMode crates/heartbit-core crates/heartbit-cli` → **0 hits**; it appears only in `heartbit-tui` (`app.rs`, `main.rs`, `ui.rs`, `trace.rs`). The CLI, daemon, and Restate/workflow entrypoints have **no plan mode at all** — their only mode-fidelity mechanism is the wish-gate in `runner.rs`. The whole research corpus assumes "the harness" is one enforcement point; heartbit has four, and three carry less than the TUI.

**Design consequence:** request-mode fidelity belongs in **core** (`runner.rs`), the only locus all four entrypoints share. The TUI becomes one consumer of a core-side mode. This threads §4's "reuse plan-mode machinery" (reuse the *mechanism*) with the relocation needed so CLI/daemon/Restate inherit it.

---

## 1. Executive summary

**The thesis is correct and evidence-backed: charging ahead on a hedged/underspecified request is *expected* behaviour for mid-tier models, and a prompt cannot reliably fix it.**

- **Pragmatic understanding degrades with tier.** Ruis et al., *Goldilocks of Pragmatic Understanding* (arXiv 2210.14986): most LLMs are near-random on implicature and "default to surface-level comprehension, missing implied meanings"; in-context prompting does **not** close the gap — only scale or example-level tuning does. `je souhaite…` read as a settled spec is a textbook literal-reading failure.
- **The bias is measured, not anecdotal.** Korean indirect-speech-act study (arXiv 2502.10995): GPT-4 scored 84.72% on *direct* acts vs 58.06% on *indirect* (a 26.66-pt gap); "when the literal meaning is sufficiently interpretable, LLMs are inclined to interpret the utterance as a direct speech act." Humans show no such gap.
- **Instruction-following also degrades with tier.** IFEval (arXiv 2311.07911): GPT-4-class ~80% on simple instructions, smaller models **<50%**. A `[PLAN MODE — do not write files]` prompt is ignored ~half the time by exactly the models heartbit runs (mistral-medium, qwen).
- **Models almost never ask unprompted, and prompting doesn't fix it.** Ambig-SWE (arXiv 2502.13069): "models default to non-interactive behavior unless explicitly prompted, yet prompt engineering alone proves insufficient for reliable underspecificity detection." Interaction recovers 54–89% of the full-spec performance gap.

**Verdict:** the harness must carry the pragmatic + scoping load **deterministically**. The fix is a **request-intent router** that picks a response mode *before* the first LLM turn, plus **mode contracts** the harness *enforces* (not prompts). The router is hybrid — cheap deterministic markers first, a `fast`-role LLM classifier only for the ambiguous residue — and **defaults to the safer mode on uncertainty**, with a one-token user override (`vas-y` / `/mode`).

**What this is NOT:** model-cost routing (RouteLLM / cascade). heartbit already has cascading. This is *request-mode* routing — answer vs study vs clarify vs execute.

---

## 2. Request-mode taxonomy (chosen + justified)

Four modes, adopted from the failure analysis and justified by the **force × completeness** 2×2 (research #1):

| Mode | Illocutionary force (DAMSL) × completeness (ClarEval) | Example | Harness behaviour |
|---|---|---|---|
| **ANSWER** | assertive / info-request, **specified** | "what can you do", "explain X" | answer in prose; no tools, no gates |
| **EXECUTE** | directive, **specified** | "renomme foo en bar", "run the tests" | act directly; gates relaxed |
| **STUDY** | directive, **under-specified-for-design** | "regarde si on peut accélérer le build" | read-only; **must end in a written proposal** + explicit go/no-go |
| **CLARIFY** | desiderative **or** directive, **under-specified** | `je souhaite créer un petit crm` ← the incident | **ask first** (intake + question tool) before any mutation |

**Why this taxonomy and not another:**
- The two axes are *orthogonal and independently grounded* (DAMSL forward-function; ClarEval's three under-specification types). The current `is_wish_request` boolean **conflates** them — it treats "wish" as a proxy for "underspecified", so a *specified* wish (`j'aimerais que tu renommes foo en bar`) is wrongly gated and an *underspecified imperative* (`construis-moi un CRM`) slips through. The 2×2 separates them: specified-wish → EXECUTE, underspecified-imperative → CLARIFY.
- The incident collapsed **both** axes: desiderative force read as directive **and** underspecified scope read as specified → landed in EXECUTE when it belonged in CLARIFY.

**Considered alternatives (rejected as the spine, kept as prior art):**
- **Aider** `code/ask/architect/help` — near-exact precedent (mode determines write authority; ask/help read-only). We borrow the *enforcement idea* but its 4 classes are user-pinned modes, not an auto-router taxonomy.
- **Roo Code** tool-group modes (read/edit/command/mcp per mode) — borrowed as the **enforcement mechanism** (mode gates the *toolset*, §4), not as a taxonomy.
- We do **not** derive a 5th taxonomy. STUDY vs CLARIFY is the one subtle split — see §4 and open question O3.

---

## 3. The ROUTER design (hybrid, layered, safe-default)

A 3-layer defense-in-depth composition — the actual SOTA pattern (Aider + Cline + Roo + Claude Code collectively), every layer mapping to something heartbit already owns.

### Layer 0 — deterministic markers (fires on EVERY fresh request, ~free)
Expand the existing `is_wish_request` (`runner.rs:152`) into a small marker module that yields a *tentative* `(force, completeness)` signal. Costs nothing (string scan), resolves most traffic, fully multilingual by enumeration.

- **Force markers** — wish/desiderative (`je souhaite`, `j'aimerais`, `je voudrais`, **+ `il faudrait`, `ce serait bien`, `peux-tu`, `pourrais-tu`, `j'apprécierais`**, EN `i'd like`, `i would like`, `could you`, `it would be nice`); imperative (`crée`, `fais`, `ajoute`, `run`, `fix`, `rename`); interrogative (`?`, `comment`, `pourquoi`, `que sais-tu`).
- **Completeness markers** — design-heavy nouns with no spec (`crm`, `app`, `dashboard`, `feature`) → likely underspecified; concrete artifacts (a file path, a symbol, an exact command) → likely specified.
- **FR conditional morphology** (research #3): match `-rais`/`-rait` endings on volition lemmas (`voudrais`, `aimerais`, `faudrait`, `pourrais`) — the French grammatical marker of a *hedged, negotiable* request. A suffix check, still deterministic.

If Layer 0 is **confident** (a clear imperative+file-path → EXECUTE; a clear question → ANSWER; a wish+design-noun → CLARIFY), route and stop.

### Layer 1 — `fast`-role LLM classifier (fires only on the ambiguous residue)
When Layer 0 is *ambiguous*, call the `fast` provider role once with a tiny prompt: *"Classify this request into ANSWER / EXECUTE / STUDY / CLARIFY and give a 0–1 confidence. Output strict JSON."* This is the **decoupled Intent-Agent** pattern (research #1, *Ask or Assume?* — separate gatekeeper, not the executor deciding to behave) — and it satisfies the "deterministic, not prompt" requirement because **the harness enforces the returned label**; the executor model never gets to opt out.

- **Cost (verified, inventory §6):** `provider_factory("fast")` reuses the **live session provider** when no distinct `fast_model` is configured (the default) — so the call adds **only its own token cost, no new provider/connection setup**. Setting `fast_model` to a small model makes it cheaper still. The classifier prompt + reply is ~a few hundred tokens.
- **Latency honesty:** "router on every turn" ≠ "LLM call on every turn." Layer 0 resolves most turns with a string scan; **Layer 1 fires only on the ambiguous fresh-request residue.** (We deliberately do **not** cite morphllm's 430ms/$0.001 — vendor blog, unverified; see C2/O-sources.)
- **Multilingual:** the classifier covers what Layer 0's enumerated markers miss (conjugation, paraphrase) in any language the `fast` model handles. We do **not** add an embedding-similarity layer — new infra, violates smallest-change (deferred; O4).

### Layer 2 — confidence + safe default
- Above threshold → route to the predicted mode.
- **Below threshold → default to the SAFER mode** (CLARIFY for design-ish requests, else STUDY). A wrong guess toward "ask" costs one round-trip; a wrong guess toward "act" costs unwanted writes (the incident). Horvitz's cost-asymmetry (CHI '99, P7/P8): raising the cost of a wrong action *lowers* the action threshold → ask sooner.

### User override (mode is user-pinned; the model cannot self-promote)
- **Fast-path "do it":** if the message contains an explicit go-token (`vas-y`, `fais-le`, `do it`, `just build it`, `go`), force **EXECUTE** regardless of phrasing. The user's keypress always wins.
- **`/mode <answer|study|clarify|execute>`** pins the mode for the session (mirrors the existing `/mode` plan/normal/yolo command, `app.rs:170`). A pinned mode is never overridden by the router or the model.
- **Promote-to-act:** when the router defaults to a safe mode, surface a one-key promotion so safety costs one keystroke, not friction (Devin 30s-proceed / Cursor "jumping straight to Agent mode is fine").

---

## 4. MODE CONTRACTS — what the harness enforces, and HOW

**Enforcement principle (the discriminating axis):** prompt directives do **not** hold against a non-compliant model (research #4/#5; IFEval <50%). Two rungs that *do* hold, used together:
1. **Tool masking (primary)** — the mode hands the model a *restricted toolset*; it cannot call what it never received. heartbit substrate: `ToolProfile` + `filter_tools()` (`tool_filter.rs:71`).
2. **Execution-layer deny (backstop)** — if a masked-out or side-effecting call slips through, the harness refuses it before side effects. heartbit substrate: the `on_approval` deny pattern (`main.rs:691`), **relocated to core**.

The `app.rs:1099` prompt prefix stays only as *reinforcement*, never as the enforcement.

| Mode | Tool masking | Execution deny | Must-end-in | Reuses |
|---|---|---|---|---|
| **ANSWER** | read-only profile (no edit/write/patch; bash blocked) | n/a | prose answer | new read-only `ToolProfile` |
| **STUDY** | read-only profile (read/search/grep + `question`; **no** edit/write/patch; bash: read-only only) | deny mutating calls | a **written numbered proposal** + a `question`-tool **go/no-go** before any execute | Plan-mode deny pattern relocated to core + read-only profile + a "proposal-emitted" check |
| **CLARIFY** | full toolset minus first-mutation until intake runs | deny first mutation until a plan artifact exists | a `question`-tool batch (from intake gaps) **before** building | `intake` recipe (`intake.rs`) + `question` tool + existing plan-gate deny |
| **EXECUTE** | full toolset | normal gates only (Normal/YOLO as today) | the work | nothing new; gates relaxed |

**STUDY's "must end in a proposal" + go/no-go** is the contract that distinguishes it from a bare read-only turn: a STUDY turn that produces no proposal, or that tries to execute, is a contract violation the harness can detect (no `question`/proposal artifact emitted → re-inject a corrective, like the existing gates).

**The bash disagreement is resolved by intent, not by picking a winner:**
- Plan-gate keeps `PLAN_GATE_MUTATING` *excluding* bash (it gates content mutation; that semantics is correct for "build-without-a-plan").
- STUDY/ANSWER read-only contracts *include* bash (they gate side effects).
- **Unify long-term via reversibility/side-effect classification** (research #3/#5; C4): read-only bash (`ls`, `grep`, `cat`) allowed in STUDY; mutating bash (`rm`, `npm install`, `git commit`) denied. **Interim smallest-change:** STUDY blocks *all* bash; refine to read-only-bash-allowed in a later phase.

---

## 5. Integration with the existing gate triad — router PRE-EMPTS, gates STAY as backstops

heartbit already has three deterministic, model-agnostic, one-shot-per-request gates in `runner.rs` (all re-arm at `1694–1706`):
- **Ask-gate** (`1565`) — prose-question battery → re-ask via `question` tool.
- **Act-gate** (`1590`) — announced-then-stopped → execute or ask.
- **Plan-gate** (`1957`) — building-without-a-plan; tier-1 = `is_wish_request`, tier-2 = `PLAN_GATE_BACKSTOP_AT = 3` cumulative mutations.

**Decision (resolves B4 — replace vs backstop):** the router **pre-empts**, the gates **stay** (research §5: "router pre-empts, gates stay as backstops"). Concretely:
- The expanded marker set is **shared**: it feeds Layer 0 of the router **and** remains the plan-gate's tier-1 trigger. We do **not** replace `is_wish_request` — we *front* it. The deterministic `ToolResult::error` deny stays as the backstop the model can't bypass.
- **Plan-gate tier-1 (wish markers): KEPT, not subsumed.** Rationale: the router runs once per *fresh* request; the plan-gate fires *mid-turn* on the actual mutation batch. They catch different moments. A request routed to EXECUTE that the model then over-extends still hits the tier-2 backstop at the 3rd mutation. Defense-in-depth: router (front-half, per-request) + gates (in-flight, per-batch).
- **Net effect on the incident:** `je souhaite créer un petit crm` → Layer 0 (wish marker + design-noun, no spec) → CLARIFY → intake runs → `question` tool surfaces "CLI vs web vs TUI? datastore? scope?" *before* any write. Even if the router mis-routed to EXECUTE, the plan-gate tier-1 (wish) still blocks the first mutation. Two independent nets.

---

## 6. Follow-up-turn policy — "ok vas-y" must flip mode without re-routing friction

**The gap (B1, verified):** the runner re-runs `is_wish_request(&next_message)` per turn (`runner.rs:1695`) and resets all gate flags (`1697–1705`). A bare `ok vas-y` after a STUDY/CLARIFY proposal contains no wish marker → the gate silently disarms and there is **no model of "the prior turn proposed a plan; this short approval promotes it to EXECUTE under *that* plan."**

**Design:**
- At the per-turn reset block (`runner.rs:1694–1706`), **before** re-classifying, detect a **bare affirmation/continuation** deterministically (`vas-y`, `ok`, `oui`, `fais-le`, `go`, `yes`, `do it`, `continue` — short, no new substantive content).
- If the prior turn was STUDY or CLARIFY and emitted a proposal/criteria: **do NOT treat this as a fresh markerless request.** Instead:
  1. Carry the **prior turn's proposal/criteria forward as the spec** (the plan/scope already in context, or the intake brief).
  2. **Promote mode → EXECUTE** under that carried spec.
  3. Relax the gates for this turn (the front-half already happened — asking again would be the over-staging backlash).
- This is the consent-token pattern (research #2/#4) made *mode-aware*: the approval flips the mode, it is not re-routed as new input.

---

## 7. EXISTS vs BUILD (file:line, smallest-change bias)

| Need | EXISTS | DELTA to build |
|---|---|---|
| Mode enum, cycle, parse, labels | `app.rs:115` `PermissionMode` (TUI-only) | **Move/define a `RequestMode` (Answer/Study/Clarify/Execute) in core**; keep TUI `PermissionMode` as the manual override that maps onto it |
| Execution-layer deny before side effects | `main.rs:691` `on_approval` Plan deny | **Replicate the pattern in core** `runner.rs` keyed on `RequestMode`; TUI becomes a consumer |
| Tool masking by profile | `tool_filter.rs:71` `filter_tools` + `ToolProfile` (`Conversational`/`Standard`/`Full`) | **Add a `ReadOnly` profile** (read/search/grep + question; no edit/write/patch; bash gated) for STUDY/ANSWER |
| Deterministic wish/force markers | `runner.rs:152` `is_wish_request` (7 EN/FR substrings) | **Expand** to force×completeness markers + FR conditional morphology; share with the plan-gate tier-1 |
| Plan-gate / ask-gate / act-gate backstops | `runner.rs:1565/1590/1957` | **Keep unchanged**; router pre-empts |
| Front-half: criteria + classified gaps | `intake.rs` recipe (`intake`), `default_registry` | **Wire CLARIFY to run intake**, surface LOW-GUESS gaps via `question` before mutation (already the recipe's purpose) |
| `fast`-role classifier provider | `main.rs:558` `provider_factory`, role `"fast"` | **Add a `RequestRouter` stage** that calls `provider_factory("fast")` on the ambiguous residue |
| Per-turn reset seam (follow-up) | `runner.rs:1694–1706` | **Add bare-affirmation detection + mode carry-forward** here |
| `/mode` + go-token override | `app.rs:170` `PermissionMode::parse`, `/mode` command | **Extend** `/mode` vocabulary; add go-token fast-path in the router |
| Mode-decision audit | `trace.rs` (`mode_label`, Approval records) | **Add a mode-transition trace event** `{from, to, source, request_is_wish, confidence}` for `/analyze` + future per-user learning |

**Locus recommendation:** mode-state + enforcement in **core** (`runner.rs`) — the only locus CLI/daemon/Restate/TUI share. Phase it (P-roadmap): early phases unify core + fix the bash disagreement; later phases extend to CLI/daemon. **Trade-off, stated:** this is a larger code change than simply reusing the TUI deny — but reusing the TUI deny leaves three of four entrypoints with no mode fidelity (the A1 fragmentation gap).

---

## 8. TDD roadmap (P1..Pn) — tests FIRST (CLAUDE.md mandatory)

**P1 — Evaluation harness (SHIP BLOCKER; tests before any router code).**
A labeled `(request → expected_mode)` fixture set: native-FR (`je souhaite créer un petit crm` → CLARIFY; `renomme foo en bar` → EXECUTE; `que sais-tu faire` → ANSWER; `regarde si on peut accélérer le build` → STUDY), EN equivalents, **follow-up fixtures** (`[prior=STUDY] ok vas-y` → EXECUTE-under-prior-plan), **mixed/conditional** (`étudie et si c'est simple fais-le`). Metric: per-mode precision/recall + a **false-positive friction rate** (routed-to-ask when EXECUTE wanted). This is the regression gate for every later change. Fixtures are **native**, never machine-translated (research: translated benchmarks overstate robustness ~2–3 pts).

**P2 — Layer-0 deterministic markers.** Expand `is_wish_request` → force×completeness + FR morphology; unit-test each marker, shared with plan-gate tier-1. No LLM. (Test: a *specified* wish routes EXECUTE, an *underspecified* imperative routes CLARIFY — the conflation bug.)

**P3 — `RequestMode` in core + read-only `ToolProfile` + execution deny.** Define `RequestMode`; add `ToolProfile::ReadOnly`; relocate the on_approval-deny pattern to `runner.rs` keyed on mode. **Fix the bash disagreement** (STUDY blocks all bash interim). Tests: a STUDY turn cannot edit/write/patch/bash even if the model emits them.

**P4 — Layer-1 `fast`-role classifier + Layer-2 safe default.** `RequestRouter` stage; confidence threshold; default-to-safer on uncertainty; degraded-no-key path (P-fallback below). Tests with a mock `fast` provider returning each label + low-confidence → safe default.

**P5 — STUDY contract (must-end-in-proposal + go/no-go) & CLARIFY → intake wiring.** Tests: STUDY with no proposal artifact → corrective re-inject; CLARIFY fires intake gaps via `question` before first mutation.

**P6 — Follow-up inheritance.** Bare-affirmation detection at `runner.rs:1694`; carry prior plan forward; promote → EXECUTE without re-routing. Tests: `[prior=CLARIFY+proposal] vas-y` executes the proposed plan, does not re-clarify.

**P7 — User override surfaces.** `/mode` vocabulary extension; go-token fast-path; one-key promote-to-act in TUI. Tests: pinned mode survives a router disagreement; go-token forces EXECUTE.

**P8 — Audit + extend to CLI/daemon.** Mode-transition trace event; wire `RequestMode` enforcement into CLI/daemon entrypoints (inherits from core). `/analyze` cites mode transitions.

**P9 (deferred, flagged) — per-user adaptive default.** Single-sourced (Wijesekera 2017, Android) cross-domain analogy — **ship behind a flag, measure before trusting** (see O5).

Each phase ends green on: `cargo fmt --all -- --check && cargo clippy --workspace --exclude mini-crm --all-targets -- -D warnings && cargo test --workspace --exclude mini-crm`.

---

## 9. Open questions & honest trade-offs

- **O1 (degraded `fast` path, research C5):** when no `fast`/frontier model is configured, Layer 1 is unavailable. Decision: **Layer 0 + safe-default must hold the line alone** — deterministic markers route what they can, ambiguous residue defaults to CLARIFY/STUDY (the safe side). This keeps heartbit's honest no-key path working. Acceptable? (Recommend: yes — the safe default *is* the degraded behaviour.)
- **O2 (router latency on every turn):** Layer 0 is free; Layer 1 fires only on the ambiguous fresh-request residue and reuses the live provider (no connection cost). Trade-off: a small token cost + one round-trip on genuinely-ambiguous openers. Mitigated by Layer-0 coverage; measure the residue rate in P1.
- **O3 (STUDY vs CLARIFY, research C3):** which request feature picks STUDY (propose N options *without asking*) vs CLARIFY (*ask first*)? Working rule: a **design/scope** gap with mutually-exclusive options the user must choose → CLARIFY (ask, with options); an **investigation** request ("is X feasible / can we speed up Y") → STUDY (propose, no question needed). Validate on P1 fixtures.
- **O4 (multilingual mechanism, B5):** deterministic FR/EN markers + morphology (Layer 0) + `fast`-model coverage (Layer 1). Embedding-prototype similarity is **deferred** (new infra). Risk: marker gaps in other languages — the `fast` model is the safety net.
- **O5 (per-user default, B6):** the only source is Android-permissions (2017); the Defaulters/Contextuals split + 0.6 threshold is an **untested cross-domain transplant** in a coding-agent setting. Ship behind a flag, measure friction-rate deltas before trusting (P9).
- **O6 (mixed/conditional requests, B2):** `étudie et si simple fais-le` embeds an escalation condition. Does it decompose into STUDY → predicate("simple?") → EXECUTE, who evaluates the predicate (model proposes, harness gates the crossing via go/no-go), and does crossing need consent? **Open** — recommend STUDY-then-go/no-go as the v1 reduction (no auto-escalation without the user's `vas-y`).
- **O7 (over-staging backlash, research C-fatigue):** forcing CLARIFY on users who wanted EXECUTE is real friction (Cursor forums: "constantly asking permission"). Mitigations: the go-token fast-path, the one-key promote, and (later) per-user defaults. The false-positive friction rate (P1 metric) must be tracked, not assumed-zero.
- **O8 (source verification, C2):** the load-bearing *new-mechanism* stats (UA-Multi 69.40% vs 61.20%; ClarEval ~80% drop; cross-lingual 2.2–2.8 pt gaps) come from very recent arXiv-2603.x (2026) IDs not independently re-fetched this session, and the router-affordability numbers (430ms/$0.001/40–70%) are **vendor blogs** (morphllm, genta.dev) — **deliberately not used** as load-bearing here. The tier-degradation thesis rests on older, independently-checkable work (Ruis 2210.14986, IFEval 2311.07911, Korean-ISA 2502.10995, Ambig-SWE 2502.13069).

---

## 10. Sources

- Ruis et al., *The Goldilocks of Pragmatic Understanding* (arXiv 2210.14986) — mid-tier models default to literal readings; in-context prompting insufficient.
- Korean indirect speech acts in LLMs (arXiv 2502.10995) — GPT-4 84.72% direct vs 58.06% indirect.
- Zhou et al., *IFEval* (arXiv 2311.07911) — instruction-following <50% for smaller models, correlates with scale.
- Vijayvargiya et al., *Ambig-SWE* (arXiv 2502.13069) — models non-interactive unless prompted; prompting insufficient; interaction recovers 54–89% of the gap.
- *Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents* (arXiv 2603.26233, **unverified-recent**) — decoupled Intent-Agent gatekeeper.
- ClarEval (arXiv 2603.00187, **unverified-recent**) — three under-specification types; KQC metric.
- Horvitz, *Principles of Mixed-Initiative User Interfaces*, CHI '99 — EV-of-action threshold; three-way (no-op/dialog/act); cost-asymmetry P7/P8.
- DAMSL (Core & Allen 1997); Searle, *Indirect Speech Acts* (1975); French conditional-mood request-softening (univ-lyon2 thesis on *requêtes*).
- Aider chat modes (code/ask/architect/help; architect/editor split); Cline Plan/Act; Roo Code tool-group modes; Claude Code plan mode (#19874 — the bug heartbit does NOT have); Cursor Plan mode.
- Wijesekera et al., *Feasibility of Dynamically Granted Permissions* (arXiv 1703.02090) — per-user default, 0.6 threshold (cross-domain, flagged).
- **heartbit code verified this session:** `runner.rs` (`is_wish_request` 152–164, `PLAN_GATE_MUTATING` 178 excl. bash, plan-gate deny 1957–1998, per-turn reset 1694–1706); `main.rs` (Plan-mode `on_approval` deny 685–696, `is_mutating` incl. bash; `provider_factory` 558–578); `app.rs` (`PermissionMode` 114–173, prompt prefix 1099); `tool_filter.rs` (`ToolProfile`/`filter_tools` 71, no read-only profile); `intake.rs` (criteria+gaps recipe); `PermissionMode` absent from heartbit-core/heartbit-cli (grep, 0 hits).
