# heartbit-ghost P1.3b — pipeline orchestrator design

**Status:** approved 2026-05-08
**Branch:** `feat/heartbit-ghost-p1.3b`
**Predecessors:** P1.2a-e + P1.3a all merged. P1.3a shipped 7 sub-agent recipes + `tools_for_persona()`; `XGhostPersona::expand()` returns 7 agents + 5 tools.
**Successors:** P1.3c (multi-candidate generation + judge ranking + image_generator), P1.3d (Telegram review delivery + actual publisher call), P1.3e (pick storage + few-shot exemplar retrieval).

## 1. Goal

Wire the 7 sub-agent recipes from P1.3a into a working single-candidate generation pipeline:

```
researcher → writer → style_critic (with revise loop, max 3) → fact_check → publish_gate → stdout
```

Closes the runtime-style-injection gap deferred from P1.3a (the writer's user message receives the rendered style profile + topic at runtime), introduces the structured-verdict parsers for `style_critic` and `fact_check`, implements a deterministic `publish_gate` (char count + thread length), and wires the CLI body for `heartbit persona run <name> --once "<topic>"` to print the final draft to stdout.

Out of scope for this phase: multi-candidate generation (3-rotation + Levenshtein dedup) and `judge` ranking → P1.3c. Image generation → P1.3c+. Actual `publisher` call (posting to X) → P1.3d. Telegram delivery → P1.3d. Pick storage / few-shot exemplar retrieval → P1.3e. Autonomy phase logic → P1.3d/P1.4. Audit log integration → P1.4.

## 2. Architecture

New top-level module `crates/heartbit-ghost/src/pipeline/`. Five files:

```
crates/heartbit-ghost/src/pipeline/
├── mod.rs              # PipelineConfig, PipelineOutput, PipelineError, run_pipeline
├── style_render.rs     # render_style_profile_as_english(&StyleProfile) -> String
├── verdicts.rs         # StyleVerdict + FactVerdict + parsers
├── publish_gate.rs     # PublishGateError + check_publish_gate
└── prompts.rs          # build_writer_user_message + build_critic_user_message + build_fact_user_message
```

**No agent recipes are modified.** The 7 P1.3a recipes are consumed as-is. Revision is communicated via the writer's user message at runtime, not via a recipe change.

**Orchestration is manual**, not via `SequentialAgent` / `LoopAgent` from heartbit-core. The pipeline has custom per-stage input construction (style profile injection, structured verdict parsing, revision feedback) that doesn't fit the existing primitives' single-string-pipe contract. `LoopAgent` also can't take a `SequentialAgent` as its body, which would be needed to loop over `(writer + style_critic)`.

**Library/CLI split**: `pipeline::run_pipeline` is the library entry point. `heartbit-cli/src/persona.rs::Run { name, once }` constructs `PipelineConfig` and calls it.

**No new dependencies.** Reuses `Arc<BoxedProvider>` from P1.2c, `SnapshotStore::load_latest` from P1.2e, `AgentRunner` + `AgentRunnerBuilder` from heartbit-core.

## 3. Public API

```rust
// crates/heartbit-ghost/src/pipeline/mod.rs

pub async fn run_pipeline(cfg: PipelineConfig<'_>) -> Result<PipelineOutput, PipelineError>;

pub struct PipelineConfig<'a> {
    /// Persona instance name (used to load the StyleProfile snapshot).
    pub persona_name: &'a str,
    /// Topic / prompt for this run.
    pub topic: &'a str,
    /// LLM provider (shared across all 4 sub-agents in P1.3b).
    pub provider: Arc<BoxedProvider>,
    /// Corpora root (currently unused by the single-candidate path; reserved
    /// for future few-shot exemplar retrieval in P1.3e).
    pub corpora_root: &'a Path,
    /// Profiles root (passed to SnapshotStore::open).
    pub profiles_root: &'a Path,
    /// Optional progress callback. Called with a short status string at each
    /// pipeline stage start ("Researching topic...", "Drafting (iter 2)...").
    pub on_progress: Option<Arc<dyn Fn(&str) + Send + Sync>>,
}

pub struct PipelineOutput {
    pub final_draft: String,
    pub research_digest: String,
    pub style_match_score: f64,
    pub revise_iterations: usize,    // 1..=3
    pub fact_check_verdict: FactVerdict,
    pub usage_summary: TokenUsage,
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// No StyleProfile snapshot exists for this persona.
    #[error(
        "no profile snapshot for persona '{persona}' at {}; \
         run `heartbit persona profile rebuild {persona}` first",
        profiles_dir.display()
    )]
    NoProfileSnapshot { persona: String, profiles_dir: PathBuf },

    /// SnapshotStore I/O / parse failure.
    #[error("snapshot: {0}")]
    Snapshot(#[from] SnapshotError),

    /// AgentRunner construction failed (provider / config mismatch).
    #[error("agent builder: {0}")]
    Builder(#[source] heartbit_core::Error),

    /// Underlying agent execution error (network, LLM error, etc.).
    /// `stage` identifies which sub-agent was running.
    #[error("agent execution at stage '{stage}': {source}")]
    Agent {
        stage: String,
        #[source]
        source: heartbit_core::Error,
    },

    /// style_critic returned a malformed verdict (couldn't parse JSON or schema).
    #[error("style_critic verdict parse: {source}")]
    CriticParseFailed {
        #[source]
        source: serde_json::Error,
        raw: String,
    },

    /// fact_check returned a malformed verdict.
    #[error("fact_check verdict parse: {source}")]
    FactCheckParseFailed {
        #[source]
        source: serde_json::Error,
        raw: String,
    },

    /// style_critic returned `Reject` — draft is fundamentally off.
    #[error("style_critic rejected the draft: {reason}")]
    Rejected { reason: String, score: f64 },

    /// 3 revise iterations exhausted without `Pass`.
    #[error("revise loop exhausted after {iterations} iterations; last reason: {last_reason}")]
    MaxRevisionsExceeded {
        iterations: usize,
        last_draft: String,
        last_reason: String,
        last_score: f64,
    },

    /// publish_gate rejected the final draft.
    #[error("publish_gate: {0}")]
    PublishGate(#[from] PublishGateError),
}
```

**Why `&'a Path` for the roots**: caller owns the paths (typically from `default_corpora_dir()` / `default_profiles_dir()`); the pipeline borrows them for the duration of one run.

**Why `Arc<dyn Fn(&str) + Send + Sync>` for `on_progress`**: matches the existing `OnText` / `OnApproval` callback shapes in heartbit-core; supports both no-op (None) and CLI status-printing.

## 4. Data flow

```
1. Load StyleProfile:
   let store = SnapshotStore::open(cfg.profiles_root, cfg.persona_name)?;
   let snapshot = store.load_latest()?
       .ok_or(PipelineError::NoProfileSnapshot { persona, profiles_dir })?;
   let profile = snapshot.profile;

2. Build 4 AgentRunner instances from the P1.3a recipes:
   let runners = build_runners(&cfg.provider, /* tools subset */)?;
   // researcher gets websearch + webfetch
   // writer, style_critic, fact_check get no tools

3. Researcher: digest = runners.researcher.execute(cfg.topic).await
       .map_err(|e| PipelineError::Agent { stage: "researcher".into(), source: e })?
       .result;

4. Render voice guidelines: let voice = render_style_profile_as_english(&profile);

5. Revise loop (max 3 iterations):
   let mut prev_revision: Option<(String /*draft*/, String /*reason*/)> = None;
   let mut final_state: Option<(String /*draft*/, f64 /*score*/, usize /*iter*/)> = None;
   for iter in 1..=3 {
       let writer_msg = build_writer_user_message(cfg.topic, &digest, &voice, prev_revision.as_ref());
       let draft = runners.writer.execute(&writer_msg).await
           .map_err(|e| PipelineError::Agent { stage: format!("writer (iter {iter})"), source: e })?
           .result;
       let critic_msg = build_critic_user_message(&draft, &voice);
       let critic_raw = runners.style_critic.execute(&critic_msg).await
           .map_err(|e| PipelineError::Agent { stage: format!("style_critic (iter {iter})"), source: e })?
           .result;
       let verdict = parse_critic_verdict(&critic_raw)?;
       match verdict {
           StyleVerdict::Pass { score } => {
               final_state = Some((draft, score, iter));
               break;
           }
           StyleVerdict::Reject { reason, score } => {
               return Err(PipelineError::Rejected { reason, score });
           }
           StyleVerdict::Revise { reason, score: _ } => {
               prev_revision = Some((draft, reason));
               continue;
           }
       }
   }
   let (final_draft, score, iterations) = final_state.ok_or_else(|| {
       let (last_draft, last_reason) = prev_revision.unwrap_or_default();
       PipelineError::MaxRevisionsExceeded {
           iterations: 3, last_draft, last_reason, last_score: 0.0,
       }
   })?;

6. fact_check (non-blocking):
   let fact_msg = build_fact_user_message(&final_draft, &digest);
   let fact_raw = runners.fact_check.execute(&fact_msg).await
       .map_err(|e| PipelineError::Agent { stage: "fact_check".into(), source: e })?
       .result;
   let fact_verdict = parse_fact_verdict(&fact_raw)?;
   if let FactVerdict::Unverifiable { ref reason } = fact_verdict {
       progress(&format!("fact_check unverifiable: {reason}"));
       // Per umbrella spec §3 row "fact_check": "unverifiable may still pass with a flag"
       // P1.3b continues; the flag is the on_progress warning
   }

7. publish_gate (deterministic):
   check_publish_gate(&final_draft, &profile)?;

8. Print + return:
   println!("{}", final_draft);
   Ok(PipelineOutput {
       final_draft, research_digest: digest,
       style_match_score: score, revise_iterations: iterations,
       fact_check_verdict, usage_summary,
   })
```

**`on_progress` calls** (when set): `"Loading profile snapshot..."`, `"Researching topic..."`, `"Drafting (iter 1)..."`, `"Style-checking (iter 1)..."`, `"Drafting (iter 2)..."`, ..., `"Fact-checking..."`, `"Running publish_gate..."`, `"Done."` Each call is a brief status string; the CLI prints them with a `> ` prefix so they don't get confused with the final draft printed to stdout.

## 5. Style profile rendering

`render_style_profile_as_english(&StyleProfile) -> String` produces a structured-English block that goes into the writer's user message. Format:

```text
Voice guidelines:
- sentence length: short (40% short, 30% medium-short, 20% medium-long, 10% long)
- fragments: common
- opening patterns: claim_first (40%), number_first (20%), scene_first (20%), question_first (20%)
- formatting: lowercase, optional periods, em-dashes forbidden, double quotes, single line breaks
- emoji policy: rare punchline only
- hashtag policy: never
- specificity target: high
- voice traits: specific, contrarian_when_defensible, no_hedging
- ai tells to avoid: delve, in conclusion, balanced both-sides, as an AI
- thread rhythm: punchline_callbacks
- thread max length: 10 (opener must hook)
- topical obsessions: AI capabilities, engineering craftsmanship
- topical avoidances: politics, stock_picks
```

All 16 non-version `StyleProfile` fields are surfaced. Snake_case enum variants are converted to readable English where the difference matters (e.g., `RarePunchlineOnly` → "rare punchline only"). Empty `Vec<String>` fields are still shown ("voice traits: (none)") so the writer doesn't infer absence as flexibility.

**Why all 16 fields**: prompt size is small (~200 tokens) and the writer's quality benefits from full context. Skipping fields would force a "what did I miss?" guess.

## 6. Verdicts + parsing

```rust
// crates/heartbit-ghost/src/pipeline/verdicts.rs

#[derive(Debug, Clone, PartialEq)]
pub enum StyleVerdict {
    Pass { score: f64 },
    Revise { reason: String, score: f64 },
    Reject { reason: String, score: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FactVerdict {
    Verified,
    Unverifiable { reason: String },
}

pub fn parse_critic_verdict(raw: &str) -> Result<StyleVerdict, PipelineError>;
pub fn parse_fact_verdict(raw: &str) -> Result<FactVerdict, PipelineError>;
```

Both parsers:
1. Try `serde_json::from_str` on the trimmed `raw`.
2. If the LLM wrapped output in ```json fences, strip a single fence pair before parsing (defensive — same pattern as P1.2c's `StyleExtractor`).
3. Validate the schema's required fields (`verdict`, plus `style_match_score` for the critic, `reason` when applicable).
4. Return `PipelineError::CriticParseFailed { raw, source }` / `PipelineError::FactCheckParseFailed { raw, source }` on failure.

**Reason field semantics for the critic**:
- `Pass`: `reason` is optional (often absent); not surfaced.
- `Revise`: `reason` is required (used as feedback to the writer in the next iteration). If absent, treat as `"unspecified"`.
- `Reject`: `reason` is required. If absent, treat as `"unspecified"`.

**Score semantics**: `style_match_score` is required by the schema. Pass typically scores ≥ 0.9; Revise scores 0.4–0.9; Reject scores < 0.4. The pipeline doesn't enforce these bands; the verdict is the source of truth.

## 7. publish_gate

```rust
// crates/heartbit-ghost/src/pipeline/publish_gate.rs

#[derive(Debug, thiserror::Error)]
pub enum PublishGateError {
    #[error("tweet {index} exceeds 280 chars (got {len}); offending text: {text:?}")]
    TweetTooLong { index: usize, len: usize, text: String },

    #[error("thread length {actual} exceeds profile.thread_max_length {max}")]
    ThreadTooLong { actual: u32, max: u32 },

    #[error("draft is empty")]
    EmptyDraft,
}

pub fn check_publish_gate(draft: &str, profile: &StyleProfile) -> Result<(), PublishGateError>;
```

**Semantics**:
- Split `draft` on `\n\n` (two consecutive newlines) → `Vec<&str>` of trimmed tweets.
- Filter out empty tweets (handles trailing blank line).
- If 0 tweets remain → `EmptyDraft`.
- For each tweet, count Unicode chars (`tweet.chars().count()`). If > 280 → `TweetTooLong { index, len, text }`. (X uses "weighted" character counts that treat URLs as 23 chars and CJK chars as 2 — that's P1.4 territory; P1.3b uses a simple Unicode count.)
- If `tweets.len() > profile.thread_max_length` → `ThreadTooLong`.

**No PII / brand safety / harassment / electoral checks** in P1.3b. Per umbrella spec §7.3 those are P1.4 (composed pre-publisher; P1.3b doesn't have a publisher yet).

## 8. CLI wiring

`heartbit-cli/src/persona.rs::Run { name, once }` body:

```rust
PersonaCommand::Run { name, once } => {
    if registry.get(&name).is_none() {
        return Err(anyhow!(
            "persona '{name}' not found. {}",
            registry_suffix(registry)
        ));
    }

    let provider = build_provider_from_env(None)
        .map_err(|e| anyhow!("build llm provider: {e}"))?;
    let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
        .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
    let profiles_root = heartbit_ghost::voice::default_profiles_dir()
        .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;

    let on_progress: Arc<dyn Fn(&str) + Send + Sync> =
        Arc::new(|s: &str| eprintln!("> {s}"));

    let cfg = heartbit_ghost::pipeline::PipelineConfig {
        persona_name: &name,
        topic: &once,
        provider,
        corpora_root: &corpora_root,
        profiles_root: &profiles_root,
        on_progress: Some(on_progress),
    };

    let output = heartbit_ghost::pipeline::run_pipeline(cfg)
        .await
        .map_err(|e| anyhow!("pipeline: {e}"))?;

    // run_pipeline already prints the final draft to stdout.
    eprintln!(
        "> ok: revise iterations={}, style match={:.2}, fact check={:?}",
        output.revise_iterations, output.style_match_score, output.fact_check_verdict
    );
    Ok(())
}
```

Progress lines + the final summary go to **stderr** (with `> ` prefix); the final draft goes to **stdout** so the caller can pipe it (`heartbit persona run x --once "..." > draft.txt`).

## 9. Error handling, edge cases

**Missing profile snapshot** (no rebuild ever run): `NoProfileSnapshot { persona, profiles_dir }` with the canonical "run `profile rebuild` first" message. CLI surfaces verbatim.

**Researcher returns empty digest**: not an error — the pipeline continues. The writer will produce a short draft; style_critic may flag it as low-substance and revise. If that fails too, MaxRevisionsExceeded surfaces.

**style_critic output is not JSON**: `CriticParseFailed { raw, source }` carries the offending raw output. CLI prints both the source error and the raw output for debugging.

**style_critic returns Reject on iter 1**: pipeline aborts immediately with `PipelineError::Rejected`. No revise loop on Reject.

**Writer or critic agent execution fails (network, rate limit)**: `Agent { stage, source }` with the stage name. CLI surfaces.

**fact_check returns Unverifiable**: pipeline continues, prints a warning via `on_progress`, and includes the verdict in `PipelineOutput.fact_check_verdict`. P1.3d may decide to surface this in Telegram review.

**publish_gate rejects**: pipeline aborts with `PublishGate(PublishGateError)`. The CLI doesn't print the final draft to stdout in this case (the gate runs before the println). Caller must regenerate (no auto-retry in P1.3b).

**Token accumulation**: each agent's `AgentOutput.tokens_used` accumulates into `PipelineOutput.usage_summary`. On error mid-pipeline, the partial usage is dropped (no `..usage` field on errors in v0.1; could be added later).

## 10. Testing

**~22 tests, all in-tree:**

| File | Coverage | Tests |
|------|----------|-------|
| `style_render.rs` | render produces all 16 fields, snake_case enums become readable, empty Vec rendered as "(none)", lowercase/forbidden formatting reflected | 4 |
| `verdicts.rs` | parse_critic: Pass / Revise / Reject / malformed JSON / fence-stripped JSON | 5 |
| `verdicts.rs` | parse_fact: Verified / Unverifiable / malformed JSON | 3 |
| `publish_gate.rs` | single tweet OK / single tweet too long / thread OK / thread too long / individual tweet in thread too long / empty draft | 6 |
| `mod.rs` | run_pipeline integration with MockProvider returning canned text per stage: happy path single iter / revise once then pass / revise 3x then MaxRevisionsExceeded / Reject on iter 1 / no profile snapshot | 5 (the heartbit-cli dispatch error tests are part of P1.3b's CLI wiring task — small, covered separately) |

**MockProvider** mirrors the P1.2c/P1.2e pattern: a hand-rolled `LlmProvider` impl returning canned text. The pipeline integration tests construct a sequence-aware mock that returns different responses per call (researcher digest first, then writer draft, then critic verdict, etc.) — implemented via a `Vec<String>` queue popped once per call.

**TempDir-backed snapshot fixture**: integration tests construct a `SnapshotStore` over a `TempDir`, save a fixture profile, then call `run_pipeline` with that path. No env-var mutation.

**`heartbit-cli/persona.rs Run` tests**: 2 dispatch-level tests — `run_persona_not_found_returns_error` (registry + name mismatch) and `run_with_registered_persona_lists_available` (mirrors the existing P1.2e pattern). Body-level tests are covered by `pipeline::run_pipeline` library tests; the CLI body is just argument-passing + provider construction.

**Quality gate** (mirrors prior phases):

```bash
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
```

Workspace test count: 3944 → ~3966.

## 11. Architecture decisions (ADs)

**AD-1 — Manual orchestration, not `SequentialAgent`/`LoopAgent`.** The pipeline has custom per-stage input construction (style profile injection, structured verdict parsing, revision feedback) that doesn't fit the existing primitives' single-string-pipe contract. `LoopAgent` also can't take a `SequentialAgent` as its body. Manual `AgentRunner.execute().await?` calls in a Rust function gives full control with zero abstraction tax.

**AD-2 — Style profile injected into writer's USER message, not system message.** P1.3a's writer recipe already commits to this: its system prompt says "The user message contains: a topic or research digest, voice guidelines for the persona, and optionally a few exemplar posts to mirror." Honoring that contract is the cleanest path; the alternative (modifying the recipe to take a templated system prompt) violates P1.3a's "recipes are static data" principle.

**AD-3 — All 16 StyleProfile fields rendered, not a curated subset.** ~200 tokens isn't enough to justify the field-by-field judgment call. The writer's quality benefits from full context; skipping fields invites guesses about what's "important enough".

**AD-4 — Fail loud on missing profile snapshot.** No silent default profile. The persona's voice is non-trivial state and silently fabricating one risks producing posts that don't match the user's intent. Error message points to the canonical fix (`heartbit persona profile rebuild <name>`).

**AD-5 — Revise loop max 3, hardcoded for P1.3b.** Per umbrella spec §3 step row. Configurability lands in P1.3c or later when there's a real reason to vary it. YAGNI.

**AD-6 — fact_check is non-blocking.** Per umbrella spec §3: "unverifiable may still pass with a flag". P1.3b prints a warning via `on_progress` and continues. P1.3d's Telegram delivery can surface the warning to the user.

**AD-7 — publish_gate is char count + thread length only.** PII / brand safety / defamation / harassment / electoral guards are explicitly P1.4 (umbrella spec §7.3). P1.3b ships the deterministic gate scaffolding; P1.4 adds the LLM-based composed checks.

**AD-8 — Final draft to stdout, progress + summary to stderr.** Standard CLI convention. Lets the caller pipe the final draft cleanly: `heartbit persona run x --once "topic" > draft.txt`.

## 12. Acceptance criteria

P1.3b is done when:

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features` and `cargo check -p heartbit-cli --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~22 net new tests pass; coverage spans style rendering, both verdict parsers, publish_gate, run_pipeline integration (happy / revise-then-pass / max-revisions / Reject / no-snapshot), and the CLI body's dispatch error paths
- `heartbit_ghost::pipeline::{PipelineConfig, PipelineOutput, PipelineError, run_pipeline, render_style_profile_as_english, StyleVerdict, FactVerdict, parse_critic_verdict, parse_fact_verdict, PublishGateError, check_publish_gate}` are reachable as public surface
- `heartbit persona run <name> --once "<topic>"` runs end-to-end against a real profile snapshot + LLM provider, prints progress to stderr and the final draft to stdout
- `heartbit-cli/persona.rs::Run { name, once }` body wires `PipelineConfig` from CLI args + env-resolved paths and calls `run_pipeline`

## 13. Out of scope (re-stated)

- Multi-candidate generation (3-rotation + Levenshtein dedup) → P1.3c
- `judge` recipe usage → P1.3c
- `image_generator` recipe usage → P1.3c+
- `publisher` recipe usage / actual posting → P1.3d
- Telegram delivery → P1.3d
- Pick storage / few-shot exemplar retrieval into writer prompt → P1.3e
- Autonomy phase logic → P1.3d (Phase 0 only) + P1.4
- LLM-based content guardrails (PII, brand safety, harassment, electoral) → P1.4
- Audit log integration → P1.4
- Trigger specs (cron / sensors / mention polling) → P1.4
- X "weighted character" counting in publish_gate → P1.4
- Configurable revise-loop max (currently hardcoded to 3) → defer until a real need exists
- Per-tenant pipeline overrides via `PersonaParams::overrides` → P1.4
- Streaming the writer's output as it's generated (currently buffers the full draft) → defer

## 14. Reference

- Umbrella heartbit-ghost spec §3 (generation pipeline) + §6.1 (candidate generation context): `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md`
- P1.3a spec (recipes consumed by this phase): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3a-sub-agent-recipes-design.md`
- P1.2c spec (extractor pattern referenced for MockProvider + JSON parse + fence stripping): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2c-llm-style-extractor-design.md`
- P1.2e spec (`SnapshotStore::load_latest` + `default_profiles_dir`): `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.2e-cli-bodies-design.md`
- `AgentRunner` + `AgentRunnerBuilder`: `crates/heartbit-core/src/agent/runner.rs`
- `SequentialAgent` / `LoopAgent` (referenced for AD-1 rationale, NOT used): `crates/heartbit-core/src/agent/workflow.rs`
- Existing CLI scaffolding: `crates/heartbit-cli/src/persona.rs::PersonaCommand::Run { name, once }`
