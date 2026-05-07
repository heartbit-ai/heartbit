# `heartbit-ghost` — Best-in-Class Autonomous X Agent Design

**Date:** 2026-05-07
**Status:** design (awaiting review)
**Depends on:** [`2026-05-07-heartbit-foundation-design.md`](2026-05-07-heartbit-foundation-design.md) — Phase 0 must land first.

## Mission

Ship a best-in-class autonomous X (Twitter) agent. The center of gravity is **voice**: posts must be indistinguishable from a thoughtful human writer to a casual reader, over a multi-week corpus. The supporting cast is the X API tool family, an A/B preference feedback loop delivered via Telegram, and an autonomy-progression model where the bot earns less human review by demonstrating it predicts the user's preferences.

Twitter content publishing is solved (a few hundred lines of HTTP). Voice quality is the moat.

## Scope summary

| Aspect | Decision |
|---|---|
| Crate name | `heartbit-ghost` |
| Reference writer niche (default blend) | AI/tech |
| A/B review channel | Telegram only (uses existing `heartbit-telegram`) |
| L3 (fine-tuning) | Out of scope; **export-only** dataset support |
| Authorship mode default | `autonomous_undisclosed` (deployer responsibility; safe operational defaults) |
| Pipeline executor | LLM-driven orchestrator (`SequentialAgent` deferred) |
| Persona shape | TOML-first, expanded into `[[agents]]` + orchestrator + tools at registration time |

## Architecture decisions

The 8 ADs locked during brainstorming. Decisions 1–4 spell the structural shape; 5–8 spell the operational and runtime model.

### AD-1 · Tool execution context (provided by foundation)
Every X tool uses `&ExecutionContext` to resolve per-tenant credentials at execute-time. Implementation lives in foundation (Phase 0); ghost is a consumer.

### AD-2 · Persona shape: TOML-first
The X persona ships as `crates/heartbit-ghost/personas/x.toml` inside this crate. It is a TOML fragment that the persona's `expand()` method converts into `[[agents]]` blocks, an orchestrator config, the X tool family, trigger specs, and a review spec. Users override by writing their own TOML that points at `recipe = "heartbit-ghost:x"` and supplies overrides under `[persona.<name>.style]`, `[persona.<name>.cadence]`, etc.

A thin code-side wrapper exists for tests and ad-hoc runs (`heartbit_ghost::x_persona()`), but TOML is the authoritative format.

### AD-3 · Pipeline executor: LLM-driven orchestrator
Sub-agent dispatch happens through the existing orchestrator path with `dispatch_mode = "sequential"` and `routing = "always_orchestrate"`, identical to the current `configs/twitter-content-gen.toml`. `SequentialAgent` (deterministic, zero-LLM-cost orchestration) is a future optimization, deliberately deferred.

### AD-4 · Voice modeling architecture: hybrid
Two-layer storage with different lifetimes:

- **Style profile** lives in the persona's TOML — versioned, immutable per release, auditable
- **Exemplar pool** lives in the existing memory system, namespaced per-persona — refreshable, evolves with use

See [§2 Voice modeling subsystem](#2-voice-modeling-subsystem) for the concrete profile schema and blend algorithm.

### AD-5 · Disclosure / authorship / ToS
`authorship_mode = "autonomous_undisclosed"` is available as a deployment toggle. The framework ships **safe by default** but allows full autonomy with deliberate opt-in. Always-on operational guards:

- **Audit log** of every post (persona id, blend snapshot hash, candidate set, final text, model+seeds, posted-at, X tweet id)
- **Kill switch** (`HEARTBIT_GHOST_HALT=1` env var; `heartbit daemon halt --persona <name>` runtime command)
- **Anti-coordination guard** (refuses >1 X account on the same tenant cross-engaging without `allow_cross_account = true` override)
- **Content guardrails always-on** (defamation, PII, harassment, electoral content) — pre-publish, cannot be disabled
- **Per-account daily post cap** (default 12, configurable but raises are audit-logged)

The deployer owns the disclosure decision per tenant; the framework owns making misuse take deliberate effort. See [§7 Audit, guardrails, kill-switch](#7-audit-guardrails-kill-switch).

### AD-6 · Preference-feedback architecture: three-layer roadmap

| Layer | Status | Mechanism |
|---|---|---|
| **L1 — In-context preference learning** | Ships v0.1 | `(candidates, pick, context, posted_engagement)` stored in memory. At write time, top-K most-similar past picks injected as few-shot exemplars in the writer agent's prompt. Zero training. |
| **L2 — Periodic profile refinement** | v0.2 roadmap | Every N picks (default 50), batch job runs a "style analyst" sub-agent. Computes profile delta, surfaces a proposed update, user reviews once. Profile is versioned; rollbacks supported. |
| **L3 — Fine-tuning / reward model** | Out of scope; **export-only** | `heartbit persona export-preferences --format jsonl` produces a clean dataset. User integrates with HF / Modal / RunPod / etc. externally. No training infrastructure in this tree. |

Telegram is the **only** review channel. The CLI exists for management; user picks happen via Telegram inline keyboards.

### AD-7 · Crate boundaries
`heartbit-ghost` depends on `heartbit-core` (engine + persona registry + ExecutionContext), `heartbit-telegram` (review UX channel). It registers itself with the `PersonaRegistry` at startup.

```
heartbit-core            (engine, registry trait, ExecutionContext)
   └── heartbit-ghost    (X tools, voice modeling, persona recipe, A/B loop)
heartbit-telegram         (existing — used as review channel)
heartbit-cli             (gains persona subcommand surface in foundation)
```

`heartbit-core` stays minimal. Heavy deps (X HTTP client, embedding model for style similarity, possibly an HTML extractor for source URLs) live in `heartbit-ghost`.

### AD-8 · Execution & trigger sources

The persona is daemon-hosted. Four trigger sources, all built on existing daemon infrastructure:

| Trigger | Source | Cadence | Purpose |
|---|---|---|---|
| **Cadence cron** | `CronScheduler` | Persona-configured, jittered (e.g., 3×/day with ±15 min jitter) | Routine posts, "daily digest" rhythm |
| **Reactive sensors** | `heartbit-sensors` | Event-driven on signal | Hot-take when news fires; RSS / feed / X-trending watchers |
| **Mention polling** | Light cron (default 5 min) | Fixed | Reply to mentions, DMs, quote-tweets |
| **Manual nudge** | Telegram DM to bot | On-demand | "Draft a post about X" |

Composition rules:
- **Per-account token bucket rate limiter** above all triggers — default 12/day, 1/hour burst — governs the absolute ceiling
- **De-duplication**: triggers within a 90-second window for the same persona/topic are coalesced into one draft
- **Backpressure**: if the Telegram review queue has ≥ 3 unanswered candidates, autonomous triggers pause

## 2. Voice modeling subsystem

The heart of the project. Three components: corpus, profile, runtime conditioning.

### 2.1 Reference corpus

A per-writer collection of high-engagement posts, scored and filtered. Stored under the persona's namespace in the memory system (Episodic memory). Each entry records:

- `writer_handle` (e.g., `karpathy`)
- `post_text`
- `posted_at`
- `engagement` (likes / reposts / replies — best-effort, may be missing)
- `tags` (manual: `["thread_opener", "hot_take", "self_deprecating"]`)
- `embedding` (for style-similarity retrieval)

Loaded via `heartbit persona corpus add <writer> <path-to-jsonl>` (CLI subcommand wired in foundation; body lights up in this phase).

### 2.2 Style profile schema

The structured fingerprint of a writer (single source) or blend (multi-writer). Lives in TOML, versioned alongside the persona.

```toml
[persona.x.style]
# Sentence-level
sentence_length_target = "short"           # "short" | "mixed" | "long"
sentence_length_distribution = [40, 30, 20, 10]  # % at lengths [<10, 10-20, 20-40, >40] words
fragment_frequency = "common"              # "rare" | "occasional" | "common"

# Opening patterns (how a post starts)
opening_patterns = [
    "claim_first",       # "X is wrong about Y."
    "number_first",      # "5 lessons from..."
    "scene_first",       # "I was at..."
    "question_first",    # "What if..."
]
opening_pattern_weights = [0.4, 0.2, 0.2, 0.2]

# Formatting
formatting.lowercase = true
formatting.periods = "optional"            # "always" | "optional" | "rare"
formatting.em_dashes = "forbidden"         # "preferred" | "ok" | "forbidden"
formatting.quotation_marks = "double"
formatting.line_breaks = "single"          # "single" | "double" | "rhythmic"

# Vocabulary policy
emoji_policy = "rare_punchline_only"       # "never" | "rare_punchline_only" | "occasional" | "frequent"
hashtag_policy = "never"                   # "never" | "rare" | "topic_relevant" | "always"
specificity_target = "high"                # high specificity = real numbers, real names

# Voice traits (free-text descriptors used by the writer prompt)
voice_traits = [
    "specific",
    "contrarian_when_defensible",
    "self_deprecating_occasional",
    "no_hedging",
    "no_balanced_both_sides",
]

# AI tells to actively avoid (the critic uses this list)
ai_tells_to_avoid = [
    "delve",
    "tapestry",
    "navigate",
    "it's important to note",
    "in conclusion",
    "balanced both-sides",
    "as an AI",
    "I cannot",
    "while it's true that",
]

# Thread structure
thread_rhythm = "punchline_callbacks"      # "linear" | "list_then_payoff" | "punchline_callbacks"
thread_max_length = 10
thread_opener_must_hook = true

# Topical obsessions (the persona will gravitate toward these)
topical_obsessions = ["AI capabilities", "engineering craftsmanship", "research taste"]

# Topical avoidances (will not post about these unless explicitly nudged)
topical_avoidances = ["politics", "stock_picks", "celebrity_gossip"]
```

This schema is the **load-bearing data structure** for the project. Writer agents are prompted with it. Critics evaluate against it. Profile-refinement (L2) updates fields in it.

### 2.3 Synthetic blend algorithm

The default ghost recipe ships with an AI/tech blend:

```toml
[persona.x.style]
blend = [
    { writer = "karpathy", weight = 0.30 },     # technical accessible humility
    { writer = "eladgil",  weight = 0.20 },     # signal-dense, no fluff
    { writer = "swyx",     weight = 0.20 },     # builder voice, frameworks
    { writer = "naval",    weight = 0.15 },     # aphoristic philosophical
    { writer = "sama",     weight = 0.15 },     # cryptic sparse hint-dropping
]

# Per-deployment overrides applied on top of the blended base
overrides = { lowercase = true, em_dashes = "forbidden", thread_max_length = 7 }
```

**Blend computation (offline, run by `heartbit persona profile rebuild x`):**

1. For each writer in `blend`, run a "style extractor" sub-agent on its corpus entries → produces a per-writer `style_profile.toml`
2. Merge the profiles using each writer's weight:
   - **Numeric fields** (e.g., `sentence_length_distribution`): weighted average
   - **Categorical fields** (e.g., `formatting.periods`): weighted vote, deterministic tiebreak
   - **List-of-strings** (e.g., `voice_traits`, `ai_tells_to_avoid`): union, deduped
   - **List-of-weighted** (e.g., `opening_patterns` + `opening_pattern_weights`): merge weights, normalize
3. Apply `overrides` last (user wins over blend)
4. Produce versioned `personas/x.toml` snapshot with a hash; commit if changed

The CLI exposes `heartbit persona profile diff x v3 v4` so users can see what a corpus refresh changed.

### 2.4 Runtime conditioning

The writer sub-agent's system prompt is constructed from:

1. The blended style profile (rendered as a structured English description, not raw TOML)
2. A rotating few-shot pool: K = 5 exemplars sampled from the corpus, weighted by writer share, freshness, and similarity to the current topic (L1 in-context preference learning — past user picks contribute heavily)
3. The current topic / context (research output, mention being replied to, etc.)
4. Constraints (char limit, thread length, posted-recently-don't-repeat list)

Rotation prevents the model from copying any single exemplar verbatim. Past user picks (when available) outweigh raw corpus exemplars in the few-shot pool.

## 3. Generation pipeline

Replaces the 3-writer-debate pattern in the current `configs/twitter-content-gen.toml`. A single style-conditioned writer with an internal critic loop produces stronger and more consistent voice than three independent writers blended by an external judge.

```
trigger
  ↓
researcher       (web search + fetch + summarize)
  ↓
writer           (style-conditioned; produces 1 candidate per call)
  ↓
style_critic     (scores voice match + AI-tell detection; blocking gate)
  ↓
revise loop      (writer regenerates; max 3 iterations or budget)
  ↓
fact_check       (cheap LLM verifies claims against research output)
  ↓
candidates       (3 candidates produced by re-running writer with rotated exemplars)
  ↓
publish_gate     (char count + brand safety + content guardrails — always-on)
  ↓
review_router    (Telegram for review, OR auto-publish based on phase + confidence)
  ↓
publisher        (calls X API tool; logs audit record)
```

Per-step contract:

| Step | Tools used | Output | Failure handling |
|---|---|---|---|
| `researcher` | `websearch`, `webfetch` | structured digest with sources | Fail loud; no post |
| `writer` | none (LLM only) | 1 tweet draft (or thread) | retry up to 2 |
| `style_critic` | none (LLM only) | verdict: `pass` \| `revise: <reason>` \| `reject` | `revise` loops back; `reject` aborts |
| `revise` | none | new draft | max 3 iterations |
| `fact_check` | none (uses research output) | verdict: `verified` \| `unverifiable: <reason>` | `unverifiable` may still pass with a flag |
| `candidates` | re-run `writer` x 2 with different few-shot rotations | 3 distinct candidates | min 2 required to proceed |
| `publish_gate` | none (deterministic checks) | `pass` \| `block: <reason>` | block aborts |
| `review_router` | `telegram_send_candidates` | sent for review OR auto-publish path | Telegram offline = queue locally |
| `publisher` | `twitter_post` / `twitter_thread` / `twitter_reply` | posted tweet id | retry on rate limit; abort on auth |

## 4. X tool family catalog

All tools take `&ExecutionContext`; credentials are resolved per-tenant from the context's `CredentialResolver`. Tool names are stable.

| Tool | Status | Purpose |
|---|---|---|
| `twitter_post` | **extend existing** to support `media` (image upload + alt text) | Post a single tweet, optionally with one image |
| `twitter_thread` | new | Post a sequence of N linked tweets |
| `twitter_reply` | new | Reply to a specific tweet by id |
| `twitter_search` | new | Search tweets / hashtags / users (read-only) |
| `twitter_mentions` | new | Read mentions / notifications (since-id pagination) |
| `twitter_user` | new | Look up a user; fetch recent posts |
| `twitter_dm` | new | Send / read DMs (requires elevated API access) |
| `twitter_schedule` | new | Queue a post for later (uses daemon's existing cron infrastructure) |
| `twitter_metrics` | new | Engagement analytics for a posted tweet |

**`twitter_dm` is gated** behind a per-deployment opt-in flag. Default off — DMs are a higher-risk surface (impersonation, harassment vectors) than public posting.

**`twitter_schedule`** is implemented as a `DaemonCommand::SubmitTask` with a delay; the daemon's existing cron path executes the deferred post. No new scheduling primitive.

**Existing `twitter_post`** is extended (not replaced) — backward compatible for users currently using it. The extension adds optional `media_url` / `media_alt_text` fields.

Each tool's input schema, output format, error modes, and rate-limit semantics are spelled out in the implementation plan (writing-plans output, not this design doc).

## 5. Sub-agent recipes

The persona expands into the following sub-agents, each shipped as a TOML fragment inside the crate. Three of them (`researcher`, `judge`, `social_writer`) are deliberately reusable beyond Twitter — they are the foundation for future personas (LinkedIn, blog, newsletter).

| Sub-agent | Reusable? | Tool access | Role |
|---|---|---|---|
| `researcher` | yes | `websearch`, `webfetch` | Find substance |
| `writer` | yes (renamed `social_writer`) | none | Style-conditioned generation |
| `style_critic` | partially | none | Voice match + AI-tell detection |
| `judge` | yes | none | Multi-candidate ranking (used in candidate-selection step) |
| `fact_check` | yes | none (uses research output context) | Claim verification against sources |
| `image_generator` | yes (existing) | `image_generate` | Optional accompanying image |
| `publisher` | no — Twitter-specific | `twitter_post`, `twitter_thread`, `twitter_reply` | Final API call |

Each ships with system prompts, model hints (e.g., `style_critic` can be a cheaper model; `writer` benefits from the strongest available), and `max_turns` defaults.

## 6. A/B feedback loop

### 6.1 Candidate generation

Default `candidates_per_draft = 3`. Each candidate is produced by running the writer with a different few-shot rotation (different sampled exemplars from the pool), to ensure they're meaningfully distinct. If 3 candidates collapse to ≤ 2 distinct outputs after dedup (Levenshtein ratio > 0.85), regenerate the missing slot.

### 6.2 Telegram delivery

A single Telegram message per draft. Format:

```
🪶 Draft for @<handle> — <topic shortname>

1️⃣  <candidate 1, full text>
2️⃣  <candidate 2, full text>
3️⃣  <candidate 3, full text>

Pick one, or /skip
```

Inline keyboard: `1` `2` `3` `Skip`. The `heartbit-telegram` crate already supports inline keyboards.

### 6.3 Pick storage

Every interaction (pick or skip) is stored in the persona's memory namespace as a `Reflection` memory entry:

```rust
PreferencePick {
    persona: String,
    candidates: Vec<String>,
    chosen_index: Option<usize>,    // None = skip
    context: Topic,                 // research summary, mention text, etc.
    selected_at: DateTime,
    posted_engagement: Option<EngagementSnapshot>,  // backfilled 24h after post
}
```

L1 in-context retrieval queries this store at write-time: top-K most-similar past *picks* (entries where `chosen_index = Some(_)`, by context-embedding similarity) become few-shot exemplars in the writer's prompt. Skipped entries are kept for analytics and audit but are **not** injected as exemplars (silence is not a positive signal).

### 6.4 Autonomy phases

| Phase | Behavior | Advance trigger |
|---|---|---|
| 0 — calibration | 100% candidates → Telegram | First 50 picks accumulated |
| 1 — supervised | 80% → Telegram, 20% auto-publish (high-confidence) | Persona-match score > 0.85 over rolling 100 |
| 2 — autonomous | 10% → Telegram (sampling), 90% auto-publish | Indefinite |
| 3 — sentinel | Only flagged posts → Telegram | Per-tenant policy |

**Persona-match score** is computed by `style_critic` on each generated candidate (regardless of phase). It's a 0.0–1.0 voice-match rating. Phase advancement uses the rolling average across phase-1 auto-published outputs *as evaluated against the previous phase's user picks* (i.e., does the auto-published 20% look like what the user has been picking?).

**Manual phase override.** `heartbit persona phase x --set <phase>` always works. Phase advancement is a default heuristic, not a hard gate.

**Skip handling.** A user `/skip` is **not** treated as a negative preference signal (silence ≠ rejection). It just means: don't post any of these candidates.

## 7. Audit, guardrails, kill-switch

### 7.1 Audit log

Every post produces an audit record stored both in the persona's memory namespace and (if configured) in the daemon's PostgreSQL audit table. Record fields:

- `tenant_id`, `user_id`
- `persona_name`, `persona_version` (the TOML hash at post time)
- `trigger_source` (cron / sensor / mention / manual)
- `candidates` (full text of all candidates)
- `pick_method` (`telegram_pick:<id>` | `auto_publish` | `cli_force`)
- `final_text`
- `model`, `seeds`, `cost`
- `style_match_score`
- `posted_at`, `tweet_id`
- `engagement_snapshot` (backfilled 24h, 7d)

`heartbit persona audit x --since 24h` queries this.

### 7.2 Kill switch

Three layers:

1. **Env var**: `HEARTBIT_GHOST_HALT=1` set at daemon startup → no posts, even from active triggers
2. **Runtime command**: `heartbit daemon halt --persona <name>` → halts a specific persona without restarting the daemon
3. **Anti-coordination guard**: framework refuses to operate >1 X account on the same tenant cross-engaging (replying / retweeting each other) without explicit `allow_cross_account = true` in the config. Cross-engagement detection: when about to reply/RT to an X account, check if that account is configured under the same `tenant_id` in any other `[[persona]]` instance.

### 7.3 Content guardrails (always-on)

Composed pre-`publisher`. Each is a non-bypassable check:

- **Char-count guard** — deterministic; rejects post if final text exceeds limits
- **PII guard** — regex + LlmJudgeGuardrail → catches phone numbers, emails, SSN-like strings
- **Defamation guard** — LlmJudgeGuardrail with a defamation-focused criterion ("does this make a verifiable factual claim about a named individual that could be defamatory if false?")
- **Harassment guard** — LlmJudgeGuardrail; flags slurs, targeted attacks
- **Electoral content guard** — LlmJudgeGuardrail; flags voting / candidate / election content. Default: warn-and-block; can be downgraded to warn-only with explicit deployment opt-in

These reuse the existing `LlmJudgeGuardrail` infrastructure (no new guardrail engine).

## 8. Eval plan

Voice quality is subjective. Without an eval loop, "best in class" is a wish.

### 8.1 Blind A/B (gold standard)

- 50 posts each from (a) the agent over a real week and (b) a real human writer in the same niche
- Anonymize handles
- Human raters guess which is which
- **Target metric**: detection rate ≤ 60% (random would be 50%)

### 8.2 Style-match scoring

For blend mode, hold out 10% of corpus exemplars at profile-rebuild time. Score generated posts against held-out exemplars via:
- Embedding similarity (top-K nearest)
- Critic-LLM "does this sound like the same writer?" verdict

Tracked over time as a leading indicator of voice drift.

### 8.3 Engagement rehearsal

A cheap LLM scorer rates each candidate 1–10 on predicted engagement before posting. Stored in the audit log; correlated with actual engagement after 24h to calibrate.

### 8.4 Detector sanity

Pass output through 2–3 free AI detectors (GPTZero, Originality.ai free tier, etc.) as a noise floor. Track pass rate over time. Not a target — these are weak signals — but useful for trend monitoring.

### 8.5 Eval cadence

- §8.1 (blind A/B): once per major persona-version release
- §8.2 (style-match): every post
- §8.3 (engagement rehearsal): every post
- §8.4 (detector sanity): nightly batch on the day's posts

## 9. Implementation phases

After Phase 0 (foundation) ships, this work breaks into 5 phases:

### P1.0 — Crate scaffolding (smallest, ships fast)
- Create `crates/heartbit-ghost/` workspace member
- Empty `Persona` impl that returns a stub `PersonaExpansion`
- Persona registers itself with `PersonaRegistry` at startup
- `recipe = "heartbit-ghost:x"` in TOML now resolves
- All foundation CLI subcommands now find the persona (still without functional bodies for most)

### P1.1 — X tool family (no persona use yet)
- Implement / extend tools: `twitter_post` (with media), `twitter_thread`, `twitter_reply`, `twitter_search`, `twitter_mentions`, `twitter_user`
- Defer `twitter_dm`, `twitter_schedule`, `twitter_metrics` to P1.4
- All tools use `&ExecutionContext` for per-tenant credentials
- Per-tool tests with HTTP mocking

### P1.2 — Voice modeling subsystem
- Reference corpus storage (memory namespace)
- Style profile schema (concrete struct + TOML serde)
- Style extractor sub-agent (produces per-writer profile from corpus)
- Blend algorithm
- `heartbit persona corpus add/list/remove`, `profile rebuild/diff` CLI bodies

### P1.3 — Generation pipeline + Telegram review
- All sub-agent recipes (researcher, writer, style_critic, judge, fact_check, image_generator, publisher)
- Pipeline orchestrator config
- Candidate generation (3-rotation)
- Telegram review delivery + pick storage
- Phase 0 calibration mode (always Telegram; no auto-publish yet)

### P1.4 — Autonomy phases + audit + remaining tools
- Phases 1, 2, 3 with the persona-match scoring + advancement triggers
- Full audit log (memory + Postgres)
- Kill switch (env var + daemon command)
- Anti-coordination guard
- Content guardrails (PII, defamation, harassment, electoral)
- Remaining tools: `twitter_dm` (gated), `twitter_schedule`, `twitter_metrics`
- `heartbit persona export-preferences` (L3 dataset export)

Each phase ships independently; the persona is usable (in calibration mode) after P1.3.

## 10. Out of scope (explicit)

- Harness mechanics refactor (`BuiltinToolsConfig` god struct, dead `ToolRisk` enum, schemars-driven tool schemas) — known debt, deferred indefinitely
- Per-tool rate-limiting middleware — replaced by the per-account token bucket above all triggers
- Streaming tool output, tool cancellation, tool lifecycle hooks (init/shutdown)
- L3 fine-tuning or reward-model training infrastructure — export-only; users integrate externally
- Workflow agent (`SequentialAgent` / `ParallelAgent`) executor for the persona pipeline — current LLM-driven orchestrator is sufficient
- Other personas (`heartbit-coder`, `heartbit-researcher`, etc.) — they will reuse this design's primitives; not built here
- Dashboard / web UI for review — Telegram is the only review channel
- A2A (agent-to-agent) integration with other agents reviewing posts — not needed for v0.1
- Multi-language voice modeling (the blend assumes English)
- Image generation quality engineering — relies on existing `image_generate` tool, no improvements scoped here

## 11. Open questions (deferred to implementation)

These are intentionally not blocking the spec; they're decisions that can be made during implementation with low cost of being wrong:

- Specific writer handles to seed the AI/tech default blend — the design says "AI/tech leaning"; the exact handles in the shipped `personas/x.toml` will be chosen during P1.2 (likely karpathy / eladgil / swyx / naval / sama, but subject to corpus availability and licensing review)
- Telegram chat_id management for multi-tenant deployments — single chat per tenant or single shared chat with tenant prefix; defer until first multi-tenant deployment is real
- Profile-version diff display format — text? structured table? side-by-side? defer to P1.2
- Mention-poll cadence under heavy mention volume — fixed 5 min for v0.1; dynamic backoff if rate-limited
- How long to keep candidate sets in audit (storage cost vs. forensic value) — default 90 days; configurable
- Whether `style_critic` should also evaluate the *winning* auto-published candidate retroactively (for ongoing calibration) — yes in P1.4, no for v0.1

## 12. Acceptance criteria

`heartbit-ghost` v0.1 is done when:

- A daemon runs with `[[persona]] recipe = "heartbit-ghost:x"` configured, in calibration phase, and successfully:
  - Receives a manual Telegram nudge → produces 3 candidates → user picks → tweet is posted
  - Fires on cadence cron → produces 3 candidates → user picks or skips
  - Polls mentions → drafts replies for each new mention → routes to review
- All 9 X tools have passing tests with HTTP mocking
- Voice modeling subsystem can ingest a 100-tweet corpus, extract a profile, blend across writers, and produce a versioned `personas/x.toml`
- Audit log records every post with the schema in §7.1
- Kill switch (env var) verifiably halts publishing
- `heartbit persona export-preferences x --format jsonl` produces a parsable JSONL file
- `cargo fmt -- --check && cargo clippy -- -D warnings && cargo test` green across the workspace, including the new crate

v0.2 (L2 + autonomy phase advancement automation) is a follow-up release.

## 13. Ethics, ToS, and operator responsibilities

Documented here so it lives alongside the design, not as an afterthought:

- **X / Twitter ToS.** Automation policies have shifted over time. As of 2026, X requires labeling for some classes of automation. Operators are responsible for compliance with the ToS in their jurisdiction. The framework provides the audit log to support this; it does not enforce labeling.
- **Right of publicity / writer imitation.** The default blend mode (synthetic voice from N writers) has lower legal exposure than single-writer cloning. The framework supports both; the deployer is responsible for ensuring use is permitted. Reference corpora the operator builds must be compliant with X's developer terms regarding data collection.
- **Astroturfing / coordinated inauthentic behavior.** The anti-coordination guard makes single-tenant multi-account abuse painful by default. Cross-tenant collusion is outside the framework's purview; operators must not collude.
- **Misinformation.** The fact-check sub-agent reduces but does not eliminate the risk of confidently wrong posts. Audit logs preserve traceability.
- **Disclosure.** `authorship_mode = "autonomous_undisclosed"` is offered. Whether to use it is the deployer's call; they own the decision and the regulatory exposure that comes with it.

This is a powerful product. Operators are expected to deploy it with judgment.

---

End of design.
