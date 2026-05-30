# heartbit-core domain-extraction — Design

**Goal:** Make `heartbit-core` releasable as a credible SOTA Rust agentic framework crate (in the company of `rig-core`, `swiftide-core`, `autoagents-core`) by extracting the heartbit-ghost domain leaks that have accumulated since v2026.507.2.

**Snapshot:** 2026-05-26. heartbit-core's good parts (`ExecutionContext`, `CredentialResolver`, `Tool::redact_for_history`, cascade `__respond__` fix) are ahead of Rig/Swiftide; the leaks are concentrated and well-localized.

## Non-goals

- Renaming, redesigning, or refactoring the agent runner / LLM provider trait / tool trait. These are the SOTA-grade parts; leave them alone.
- Publishing to crates.io. Just internal hygiene; release tagging is downstream.
- Touching `heartbit-gateway`. Verified: gateway depends on `heartbit` umbrella (with `features = ["daemon"]`) and references zero persona configs in its source. The doc-comment in `config/daemon.rs:13-16` justifying these configs as "kept here so they can be re-used by the gateway" is **stale and incorrect**.
- Adding a separate `heartbit-personas-config` crate. Since the gateway doesn't consume these types, `heartbit-ghost` is the natural home — it's already the X-domain crate.
- Vendor LLM client extraction (Anthropic, OpenRouter, etc. clients). They're already correctly placed in `heartbit-core::llm`; surveyed SOTA frameworks split these only at large scale and we don't need to.

## Architecture decision

**Move persona configs + the OpenverseImageSearchTool to `heartbit-ghost`. Delete the duplicated `TwitterPostTool` from heartbit-core outright.**

Rationale:
- `heartbit-ghost` is already the X/Twitter/blog domain crate. It owns the publishing flow, the persona pipelines, and the duplicated OAuth signing. The configs naturally belong with their consumers.
- A separate config-only crate (`heartbit-personas-config`) would be cleaner *if* multiple downstream crates needed the types without the runtime. Since only `heartbit` (umbrella) and `heartbit-cli` actually parse + consume these configs, and both already depend on `heartbit-ghost` directly or transitively, the extra crate is YAGNI.
- The `heartbit` umbrella crate already re-exports config types from `heartbit_core::config::*`. After the move, it will re-export from `heartbit_ghost::config::*` instead. CLI updates 1 import path.

## SOTA reference (from deep-research workflow `wx2r162az`)

| Framework | Pattern | What's in `*-core` | What's split out |
|---|---|---|---|
| **Swiftide** | Workspace crates | `swiftide-core` (primitives), `swiftide-agents` (orchestration) | `swiftide-integrations` (24 feature-gated vendor connectors — "No integrations enabled by default") |
| **Rig** | Workspace crates | `rig-core` (provider abstractions) | 12 companion crates: `rig-lancedb`, `rig-postgres`, `rig-qdrant`, etc. |
| **AutoAgents** | Workspace crates | `autoagents-core` (runtime), `autoagents-llm` (LLM trait), `autoagents-protocol` | `autoagents-toolkit` (downstream of umbrella, feature-gated tools), `autoagents-qdrant`, `autoagents-guardrails`, `autoagents-telemetry`, `autoagents-speech` |
| **langchain-rust** | Single crate | Everything | 17 feature flags, `default = []` — heavy/vendor deps opt-in |

**Verified 3-0**: No surveyed framework ships Twitter/X publishing, blog rendering, or persona-specific configuration in (or even adjacent to) the core crate. The only domain tools shipped anywhere — web search, Wolfram Alpha, document parsing, filesystem, Treesitter — live behind feature flags in dedicated tool/integration crates.

## Concrete moves

### 1. Delete `heartbit_core::tool::builtins::twitter_post`

- **File:** `crates/heartbit-core/src/tool/builtins/twitter_post.rs` (~358 lines + ~10 tests)
- **Re-exports to delete:** `tool/builtins/mod.rs:18` (`pub(crate) mod twitter_post;`), `mod.rs:151` (`pub use twitter_post::TwitterCredentials`), `lib.rs:155` (root re-export of `TwitterCredentials`)
- **Wiring to delete:** the credential-gated branch in `builtin_tools()` (`mod.rs:~334-336`)
- **Why safe:** `heartbit-ghost/src/tools/client.rs:331` doc-comments itself as `"port from heartbit-core::tool::builtins::twitter_post"` and reimplements the same OAuth 1.0a HMAC-SHA1 signing (`client.rs:343-438`). The core copy is dead duplication.

### 2. Move `heartbit_core::tool::builtins::openverse_image` → `heartbit_ghost::tools::openverse_image`

- **Source:** `crates/heartbit-core/src/tool/builtins/openverse_image.rs` (~460 lines + 8 tests)
- **Target:** `crates/heartbit-ghost/src/tools/openverse_image.rs`
- **Re-exports to update:** delete the line at `core/tool/builtins/mod.rs:148` (`pub use openverse_image::OpenverseImageSearchTool`); add to `ghost/src/tools/mod.rs` (or `ghost/src/lib.rs` if that's where `tools` is declared)
- **Consumer updates:** `crates/heartbit-ghost/src/review/mod.rs` import switches from `heartbit_core::tool::builtins::OpenverseImageSearchTool` to `crate::tools::OpenverseImageSearchTool`
- **Why safe:** sole consumer is ghost; the only thing in core that knew about it was the re-export

### 3. Feature-gate the persona configs

**Decision:** Cargo feature gate. The types stay in `heartbit-core` (avoiding a breaking TOML change for the operator) but are excluded from the default surface when SOTA users opt out.

**Why this and not a full move to ghost:** moving the persona config types to `heartbit-ghost` would require either
- inverting the core→ghost dependency direction (the `DaemonConfig` struct in core references the persona types), or
- breaking the operator's TOML shape by relocating `[[daemon.persona_posts]]` to `[[ghost.persona_posts]]`.

Both are worse than the leak we're trying to clean up. The feature-gate approach mirrors the pattern langchain-rust uses for vendor backends (single crate, `default = []`, heavy/vendor deps opt-in) and aligns with the SOTA principle "heavy/domain code must be opt-in" without forcing operator migrations.

**Concretely:**

1. Add a feature to `crates/heartbit-core/Cargo.toml`:
   ```toml
   [features]
   default = ["ghost-domain-config"]
   ghost-domain-config = []
   ```

2. Gate all persona-specific config types + `ImageSource` behind `#[cfg(feature = "ghost-domain-config")]`. Types affected (file: `crates/heartbit-core/src/config/daemon.rs`):
   - `PersonaMentionsConfig` (~50 lines)
   - `PersonaPostsConfig` (~80 lines)
   - `PersonaQuotesConfig` (~50 lines)
   - `PersonaBlogConfig` (~60 lines, includes `XAnnounceConfig` + `GithubReadmeConfig`)
   - `ImageSource` enum (~10 lines)

3. Gate the corresponding fields on `DaemonConfig`:
   ```rust
   pub struct DaemonConfig {
       // ... unchanged ...
       #[cfg(feature = "ghost-domain-config")]
       #[serde(default)]
       pub persona_mentions: Vec<PersonaMentionsConfig>,
       #[cfg(feature = "ghost-domain-config")]
       #[serde(default)]
       pub persona_posts: Vec<PersonaPostsConfig>,
       #[cfg(feature = "ghost-domain-config")]
       #[serde(default)]
       pub persona_quotes: Vec<PersonaQuotesConfig>,
       #[cfg(feature = "ghost-domain-config")]
       #[serde(default)]
       pub persona_blog: Option<PersonaBlogConfig>,
   }
   ```

4. Update `tool/builtins/mod.rs` and `config/mod.rs` re-exports + `lib.rs` to similarly `#[cfg]`-gate any re-export of `ImageSource` / persona config types.

5. Internal callers (`heartbit`, `heartbit-cli`, `heartbit-ghost`) require **no source changes** — they get `heartbit-core` with default features on. Each can later opt out via `heartbit-core = { default-features = false }` if they ever want to drop the ghost surface, but in practice they need these types.

6. SOTA-framework users who depend on `heartbit-core = { default-features = false }` get the clean surface: agent runner, LLM providers, generic tools (file/shell/search/edit/etc.), memory, eval, guardrails, workflow agents, ExecutionContext, cascade. No X publishing, no blog config, no ImageSource.

**Out of scope for this spec (future work):** if the project grows external SOTA-framework adopters, the natural v2 is a hard split — either move the gated types into `heartbit-ghost::config` with a TOML migration, or extract a `heartbit-domain-config` sibling crate. v1 keeps the door open without forcing it.

### 4. Restore the docs gate

Remove three `#![allow(missing_docs)]` inner attributes:
- `crates/heartbit-core/src/config/daemon.rs:1`
- `crates/heartbit-core/src/tool/builtins/twitter_post.rs:1` (deleted in move #1)
- `crates/heartbit-core/src/tool/builtins/mod.rs:3` (this one silently exempts the entire builtins subtree — the highest-leverage fix)

Fill missing rustdoc as compile errors surface. Estimate: ~50-100 doc additions; many are short one-liners.

### 5. Uniform builtin wiring

Today: `image_generate` + `tts` always-on, `twitter_post` cred-gated, `openverse_image` never wired.

After moves #1, #2: only `image_generate` + `tts` remain as cred-or-feature-ambiguous builtins in core.

Action: pick one pattern. Recommend opt-in via `BuiltinToolsConfig` flags (e.g. `enable_image_generate: bool`, `enable_tts: bool`, defaulting to false). Matches Swiftide's "No integrations enabled by default" precedent. Document in `BuiltinToolsConfig` rustdoc.

Out of scope if this expands the diff too much — drop to a follow-up in `tasks/lessons.md`.

### 6. Persona trait cleanup

In `crates/heartbit-core/src/persona/types.rs`:
- Remove the *"evangelism framing"* example from `PersonaExpansion::mode_addendum` doc (lines 36-39). Replace with a domain-neutral example or none.
- Delete `pub enum TriggerSpec {}` and `pub enum ReviewSpec {}` (empty `#[non_exhaustive]` enums, scaffolding noise). If any code references them, replace with concrete variants or `()`.

If `persona` is gated by `ghost-domain-config` too: keep the trait + `PersonaRegistry` in default-features (it's the plugin seam — generic enough), but gate `PersonaExpansion::mode_addendum` if it's domain-specific. Probably fine to keep the trait in default-features.

## Migration impact

Internal callers (the only callers):
- `heartbit/src/daemon/*.rs`: zero changes (everything still resolves via re-exports through the umbrella crate)
- `heartbit-cli/src/daemon/*.rs`: zero changes (uses umbrella re-exports)
- `heartbit-ghost/src/review/mod.rs`: 1 import update (OpenverseImageSearchTool path)
- `heartbit-ghost/src/tools/client.rs`: zero changes (TwitterPostTool removal is invisible — ghost has its own impl)

External users (none today, but future SOTA-framework adopters):
- Depending on `heartbit-core = { default-features = false }` gives them the clean surface
- The default features keep current behavior — no breaking change

## Test impact

All 2395 heartbit-core lib tests should still pass after extraction. Tests that get moved with their code: ~30 (twitter_post: ~10, openverse_image: 8, persona configs: ~12-15). They land in the destination crate's tests.

New tests:
- Smoke: `cargo build --package heartbit-core --no-default-features` builds clean (validates the feature gating)
- The full workspace test suite (4292 currently) stays green

## Out of scope

- Publishing heartbit-core to crates.io
- Renaming any types
- Changing the public LLM provider trait, agent runner, or tool trait signatures
- Reworking `ImageGenerateTool` or `tts` (decision deferred to a follow-up)
- Adding new vendor integrations
- Splitting heartbit-ghost itself
