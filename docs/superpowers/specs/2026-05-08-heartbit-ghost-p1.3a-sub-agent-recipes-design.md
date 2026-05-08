# heartbit-ghost P1.3a — sub-agent recipes design

**Status:** approved 2026-05-08
**Branch:** `feat/heartbit-ghost-p1.3` (P1.3a is the first sub-phase of the P1.3 decomposition)
**Predecessors:** P1.2a/b/c/d/e all merged. P1.0 scaffolding stubs `XGhostPersona::expand()` to return `PersonaExpansion::default()` (empty).
**Successors:** P1.3b (pipeline orchestrator), P1.3c (multi-candidate generation), P1.3d (Telegram review), P1.3e (pick storage).

## 1. Goal

Ship the 7 sub-agent recipes (per umbrella spec §5) and rewrite `XGhostPersona::expand()` to return a populated `PersonaExpansion` containing those recipes plus the 6 tool instances they reference. After P1.3a, the persona is no longer a stub — its `expand()` produces real `AgentConfig` values that downstream orchestration (P1.3b) will chain into the generation pipeline.

Out of scope for P1.3a: the pipeline orchestrator that chains the agents (P1.3b), runtime injection of the style profile into the writer prompt (P1.3b), few-shot exemplar selection (P1.3e), multi-candidate generation (P1.3c), Telegram delivery (P1.3d), pick storage (P1.3e).

## 2. Architecture

A new `agents/` directory inside `heartbit-ghost`, one file per recipe (mirrors heartbit-core's `tool/builtins/` per-tool layout):

```
crates/heartbit-ghost/src/agents/
├── mod.rs              # re-exports + tools_for_persona() helper
├── researcher.rs       # reusable: websearch + webfetch
├── writer.rs           # reusable: no tools (style-conditioned generation)
├── style_critic.rs     # partially reusable: no tools
├── judge.rs            # reusable: no tools (multi-candidate ranking)
├── fact_check.rs       # reusable: no tools
├── image_generator.rs  # reusable: image_generate
└── publisher.rs        # Twitter-specific: twitter_post + twitter_thread + twitter_reply
```

Each recipe file exposes:

```rust
pub const <NAME>_SYSTEM_PROMPT: &str = r#"..."#;
pub fn <name>_recipe() -> AgentConfig { ... }
```

`agents/mod.rs` re-exports the 7 `*_recipe` functions and the `tools_for_persona()` helper.

**Reusability boundary** (per umbrella spec §5):
- 3 cleanly reusable: `researcher`, `writer`, `judge` — system prompts are platform-agnostic; no `twitter` / `x ` / `(twitter)` substrings allowed (enforced by tests).
- 3 partially reusable: `style_critic`, `fact_check`, `image_generator` — generic prompts; might be tuned per-platform later.
- 1 Twitter-specific: `publisher` — explicitly references X conventions and uses X tools.

**No new dependencies and no new tools.** Tool sources:
- `WebSearchTool`, `WebFetchTool`, `ImageGenerateTool` from `heartbit_core::tool::builtins::` — no-arg `::new()` constructors that fit the persona-expansion pattern.
- `TwitterThreadTool`, `TwitterReplyTool` from `heartbit_ghost::tools::` (P1.1) — unit structs with no-arg `::new()`; resolve OAuth1 credentials from `ExecutionContext` at `execute()` time via `XClient::from_context`.

**Why not `twitter_post`** (a deliberate scope cut): the pre-existing `heartbit_core::tool::builtins::TwitterPostTool` requires `TwitterCredentials` at construction time (older P1.0 pattern, incompatible with persona expansion since credentials aren't known at startup). Adding a heartbit-ghost-native equivalent (~200 LOC of OAuth1 + media-upload code + ~12 tests adapted from the heartbit-core version) is out of scope for P1.3a. **Workaround**: the publisher recipe uses `twitter_thread` for single tweets too (thread of length 1; the X API treats a thread-of-1 as a regular tweet without `reply_to`). Media-attached single posts are NOT supported in P1.3a — when that becomes a real requirement (likely P1.3b or P1.4), a small follow-up adds the heartbit-ghost-native `TwitterPostTool`.

## 3. Recipes

Per-recipe knobs (the spec pins these contracts; exact prompt text lives in the implementation plan, not the spec, since prompt wording will iterate based on LLM behavior):

| Recipe | Tools | `max_turns` | `max_tokens` | `reasoning_effort` | `response_schema` | Reusability |
|--------|-------|-------------|--------------|--------------------|--------------------|-------------|
| `researcher` | `websearch`, `webfetch` | 8 | 4096 | `medium` | None (free-form digest) | reusable |
| `writer` | none | 1 | 1024 | `low` | None (free-form draft) | reusable |
| `style_critic` | none | 1 | 512 | `medium` | structured: `pass` / `revise: <reason>` / `reject` + `style_match_score: 0.0..=1.0` | partial |
| `judge` | none | 1 | 512 | `medium` | structured: `chosen_index: 0..N` + `reasoning: String` | reusable |
| `fact_check` | none | 1 | 1024 | `medium` | structured: `verdict: "verified"` \| `"unverifiable: <reason>"` | reusable |
| `image_generator` | `image_generate` | 2 | 1024 | `low` | None (image url + alt text via tool output) | reusable |
| `publisher` | `twitter_thread`, `twitter_reply` | 2 | 512 | `low` | None (final tool output is the tweet id) | Twitter-only |

**Why these knobs:**

- **`max_turns = 1`** for the LLM-only single-shot agents (writer, style_critic, judge, fact_check) — they're single-call agents, no iterative tool use. The pipeline (P1.3b) handles the revise loop via re-invocation, not by giving style_critic more turns.
- **`max_turns = 8`** for researcher — multiple `websearch` + `webfetch` calls per topic before producing the digest.
- **`max_turns = 2`** for image_generator and publisher — at most one tool call + one final response.
- **`reasoning_effort = medium`** for the agents whose quality matters most (researcher, style_critic, judge, fact_check). `low` for the agents where speed matters more than nuance (writer, image_generator, publisher).
- **`response_schema`** is set on style_critic, judge, and fact_check — these produce structured verdicts the orchestrator parses directly. The other 4 produce free-form text.

**`writer.rs` system prompt is style-profile-free.** Says something like:

> *"You are a social media writer. Produce one short, engaging post draft per call. Output the post text only — no preamble, no markdown fences, no commentary. The orchestrator will supply topic context and voice guidelines in the user message."*

P1.3b's orchestrator wraps this at runtime by appending the rendered style profile + few-shot exemplars + topic context to the writer's user message. The static system prompt stays clean and reusable across personas.

**`publisher.rs` system prompt is the only one that mentions X/Twitter:**

> *"You publish a finalized social post to X (Twitter). Choose the right tool: `twitter_thread` for any post that's not a reply (single tweet → pass a single-element array; chained sequence → pass the full thread), `twitter_reply` when replying to a specific tweet id. The post text is approved — do not modify it. Return the final tool output (the tweet id) without commentary."*

(Note: `twitter_post` for media-attached single posts is intentionally absent — see §2 scope cut. When media support becomes a requirement, a heartbit-ghost-native `TwitterPostTool` lands in a follow-up phase and the publisher prompt is updated.)

## 4. `XGhostPersona::expand()` rewrite

```rust
impl Persona for XGhostPersona {
    // name, description, version unchanged

    fn expand(&self, _params: &PersonaParams) -> Result<PersonaExpansion, heartbit_core::Error> {
        let agents = vec![
            agents::researcher_recipe(),
            agents::writer_recipe(),
            agents::style_critic_recipe(),
            agents::judge_recipe(),
            agents::fact_check_recipe(),
            agents::image_generator_recipe(),
            agents::publisher_recipe(),
        ];

        let tools = agents::tools_for_persona();

        Ok(PersonaExpansion {
            agents,
            tools,
            // P1.3b populates orchestrator
            // P1.4 populates triggers
            // P1.3d populates review
            ..PersonaExpansion::default()
        })
    }
}
```

`agents::tools_for_persona()`:

```rust
pub fn tools_for_persona() -> Vec<Arc<dyn Tool>> {
    use heartbit_core::tool::builtins::{ImageGenerateTool, WebFetchTool, WebSearchTool};
    use crate::tools::{TwitterReplyTool, TwitterThreadTool};

    vec![
        Arc::new(WebSearchTool::new()),
        Arc::new(WebFetchTool::new()),
        Arc::new(ImageGenerateTool::new()),
        Arc::new(TwitterThreadTool::new()),
        Arc::new(TwitterReplyTool::new()),
    ]
}
```

`heartbit_core::tool::builtins::{WebSearchTool, WebFetchTool, ImageGenerateTool}` all have no-arg `::new()` (verified — they internally panic on TLS init failure, not a runtime concern). 5 tools total in P1.3a's persona expansion. (`twitter_post` deferred — see §2.)

**`PersonaParams::credentials_env`** is unused at expand-time — credentials are resolved per-call by `ExecutionContext::credentials` (the `EnvResolver` pattern from the P1.1 smoke examples). The persona just declares the tools; execution-context wiring is a daemon/CLI concern.

**Same tool referenced from multiple recipes**: not a problem. The orchestrator (P1.3b) maps tool names to instances; `tools_for_persona()` returns each tool exactly once even though `twitter_post` would appear once in `publisher`'s tool set if recipes carried tool *instances* — they don't. Recipes reference tools by name; the actual `Arc<dyn Tool>` instances live in `expansion.tools`.

## 5. Error handling, edge cases, scope

**`*_recipe()` functions are infallible** — they construct `AgentConfig` from constants, no I/O, no async. If a recipe's prompt is malformed (e.g., a typo in the const), it's a compile error.

**`expand()` returns `Result<PersonaExpansion, Error>`** because the trait demands it (and P1.3b/P1.4 may add fallible loads — e.g., reading the persona's snapshot from disk). For P1.3a, every code path inside `expand()` is infallible — every `*_recipe()` is infallible and every tool constructor is currently infallible. Returns `Ok(...)` to preserve the trait signature.

**The 3 X tools NOT wired in P1.3a** (`twitter_search`, `twitter_mentions`, `twitter_user`): no P1.3a recipe uses them. They'll surface in:
- `twitter_search` → researcher's tool set in P1.4 (when researcher learns to look up X accounts)
- `twitter_mentions` → mention-trigger code in P1.4
- `twitter_user` → researcher in P1.4

Leaving them out of P1.3a's `tools_for_persona()` is deliberate. Including them would expose unused tools to the orchestrator and bloat the surface.

## 6. Testing

**~17 tests, all in-tree** in the new module's `#[cfg(test)] mod tests` blocks (one block per recipe file plus `mod.rs`). Pure unit tests; no async, no I/O.

**Per-recipe shape tests (7):** for each recipe, assert `name`, `description` non-empty, `system_prompt` non-empty, `max_turns`, `max_tokens`, `reasoning_effort`, presence-or-absence of `response_schema`. One test per recipe — locks the public contract that P1.3b's orchestrator depends on.

**Reusability boundary tests (3):** `{researcher,writer,judge}_prompt_is_platform_agnostic` — assert the lowercased system prompt does NOT contain `twitter`, `x `, or `(twitter)`. Cheap regression guard against future drift.

**Twitter-specific recipe test (1):** `publisher_prompt_mentions_x_and_uses_twitter_tools` — assert the lowercased system prompt DOES contain `twitter` or `x ` or `(twitter)`.

**`tools_for_persona()` test (1):** `tools_for_persona_returns_five_distinct_tools_in_declared_order` — assert `tools.len() == 5` and the names match the spec order: `websearch, webfetch, image_generate, twitter_thread, twitter_reply`.

**`expand()` integration test (1, replaces the deleted P1.0 stub test):** `expand_returns_seven_agents_and_five_tools_in_declared_order` — assert `exp.agents.len() == 7` and `exp.tools.len() == 5`, with exact agent names in declared order.

**Existing P1.0 test `stub_expand_returns_empty_expansion` is deleted** — it asserts the empty default that's no longer the behavior.

**No tests on system-prompt content beyond the substring boundary checks.** Prompt wording will iterate based on real LLM behavior; locking exact phrasing creates brittle assertions. The structural tests (name, max_turns, response_schema, etc.) are what callers actually depend on.

**Quality gate** (mirrors prior phases):

```bash
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features
```

Workspace test count: 3932 → ~3948 (net +16: 17 new tests minus 1 deleted P1.0 stub test).

## 7. Architecture decisions (ADs)

**AD-1 — Rust constants, not TOML files.** Each recipe is a `pub fn <name>_recipe() -> AgentConfig` constructed inline from raw-string constants. Mirrors P1.2c's `default_system_prompt`. No TOML parsing layer at runtime; uniform with the existing voice/extractor surface; faster to grep and edit.

**AD-2 — One file per recipe.** Mirrors heartbit-core's `tool/builtins/` per-tool layout (14 files for 14 builtins). Each file is self-contained — system prompt const + recipe function + tests. Easy to find, easy to test, easy to refactor a single recipe without touching the others.

**AD-3 — Style-profile injection deferred to P1.3b.** The writer's static system prompt is style-agnostic; the pipeline orchestrator (P1.3b) appends the rendered style profile + few-shot exemplars to the writer's user message at runtime. Keeps recipes pure data, defers per-invocation composition to where it belongs.

**AD-4 — Reusable recipes are platform-agnostic.** Per umbrella spec §5, researcher / writer / judge are designed for reuse beyond Twitter (LinkedIn, blog, newsletter). Their prompts must not mention X/Twitter; tests enforce this with substring assertions. Future personas can `pub use` these recipes directly.

**AD-5 — `max_turns = 1` for LLM-only single-shot agents.** writer / style_critic / judge / fact_check are single-call agents; the revise loop is the orchestrator's responsibility, not internal to the agent. Keeps each agent's contract simple.

**AD-6 — Structured `response_schema` for verdict-producing agents.** style_critic / judge / fact_check return parsed structured verdicts (verdict + reasoning), letting the orchestrator branch on them deterministically without custom parsing. Free-form output for the agents whose result is consumed downstream as text (researcher digest, writer draft).

**AD-7 — `tools_for_persona()` returns a flat `Vec`, deduplicated by construction.** Even if multiple recipes reference the same tool by name, the persona expansion exposes each tool instance once. The orchestrator (P1.3b) maps tool *names* (referenced in agent configs) to instances; the persona just provides the pool.

## 8. Acceptance criteria

P1.3a is done when:

- All public types compile cleanly under `cargo check -p heartbit-ghost --all-features`
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- ~16 net new tests pass; coverage spans every recipe's structural contract, the reusability boundary on the 3 platform-agnostic recipes, the Twitter-specific publisher, the `tools_for_persona()` order, and the `expand()` integration
- `heartbit_ghost::agents::{researcher_recipe, writer_recipe, style_critic_recipe, judge_recipe, fact_check_recipe, image_generator_recipe, publisher_recipe, tools_for_persona}` are reachable as public surface
- `XGhostPersona::expand(&PersonaParams::default())` returns a `PersonaExpansion` with 7 agents (in declared order) and 5 tools (in declared order)

## 9. Out of scope (re-stated)

- Pipeline orchestration / agent chaining (P1.3b)
- Style profile injection into writer prompt at runtime (P1.3b)
- Multi-candidate generation (3-rotation + Levenshtein dedup) (P1.3c)
- Telegram review delivery (P1.3d)
- Pick storage / few-shot exemplar retrieval (P1.3e)
- Autonomy phase logic (Phase 0 in P1.3d; rest in P1.4)
- Audit log integration (P1.4)
- Trigger specs (cron / sensors / mention polling) (P1.4)
- The 3 X tools not used by any recipe (`twitter_search`, `twitter_mentions`, `twitter_user`) — surfaced when a P1.4 consumer needs them
- Per-tenant tool overrides (P1.4 via `PersonaParams.overrides`)
- A `PersonaConfig` integration that loads the latest profile snapshot (the writer's prompt construction lives in P1.3b)

## 10. Reference

- Umbrella heartbit-ghost spec: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md` (§3 Generation pipeline, §5 Sub-agent recipes, §6 A/B feedback loop)
- P1.2 specs (predecessors, all merged): `docs/superpowers/specs/2026-05-0[78]-heartbit-ghost-p1.2*.md`
- `Persona` trait + `PersonaExpansion`: `crates/heartbit-core/src/persona/{mod,types}.rs`
- `AgentConfig`: `crates/heartbit-core/src/config/agent.rs`
- Existing X tools: `crates/heartbit-ghost/src/tools/` (P1.1)
- Existing builtin tools: `crates/heartbit-core/src/tool/builtins/`
