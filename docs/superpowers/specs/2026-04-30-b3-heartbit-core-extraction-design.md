# B3 — Extract `heartbit-core` as the Official Rust Agentic Framework

**Date:** 2026-04-30
**Status:** Design — pending user approval before implementation plan
**Scope:** Workspace restructure. Move foundational modules from `crates/heartbit/` into a new `crates/heartbit-core/` member crate. The existing `crates/heartbit/` becomes a thin umbrella that re-exports core and adds platform-specific gated modules (daemon, sensor, channel adapters, postgres impls, etc.).
**Estimated effort:** 7–10 working days, executed as ~12 small independently-green commits.
**Public API breakage:** None. All existing imports (`use heartbit::AgentRunner`, etc.) continue to work via `pub use heartbit_core::*;` in the umbrella.

## Background

Following the B2 round (commits `b5e7cc4..f862159`), the `heartbit` crate is ~119k LOC across 209 files in a single workspace member. The architecture review commissioned during the project audit flagged this as approaching a maintenance threshold and explicitly recommended extracting a `heartbit-core` sub-crate before any major new subsystem lands.

Beyond raw LOC, the single-crate model conflates two distinct positioning stories:

1. **Library positioning.** Anyone who runs `cargo add heartbit` today gets the agent runtime *plus* Kafka, Axum, Postgres, fastembed/ONNX, Telegram bot, Discord WebSocket, and ~60 transitive crates gated behind features. The README leads with platform content (multi-tenant runtime, dashboard, daemon mode). A library user evaluating "Rust agentic frameworks" forms the impression of an internal platform that happens to expose a library — not a framework they can build on.

2. **Platform positioning.** The same repo also delivers a deployable multi-tenant runtime (`heartbit-cli` daemon mode, `heartbit-gateway`, the dashboard) which is the heartbit-cloud product's engine. This audience needs the platform docs and the operator-facing surface.

Splitting `heartbit-core` out as a workspace member crate (in the same repo, following the Tokio / Axum / Tracing / Serde precedent) lets us position the library cleanly via crates.io and docs.rs without paying the operational cost of a separate git repo. Library users `cargo add heartbit-core` and target a small, focused dependency tree; platform consumers continue to use `heartbit` (the umbrella) plus `heartbit-cli` exactly as today.

## Goals

1. Make `heartbit-core` the recommended `cargo add` target for any consumer building a library or service that uses the heartbit agent runtime.
2. Keep all existing public API paths working without source-level changes for downstream consumers (`heartbit-cli`, `heartbit-cloud`, third parties on crates.io if/when published). `use heartbit::AgentRunner;` must continue to compile.
3. Establish a clear, principled split between **framework** (heartbit-core) and **platform integrations** (heartbit umbrella's inline-gated modules: daemon, sensor, channel platform adapters, postgres impls, etc.).
4. Reposition the top-level `README.md` around the framework. Demote platform content to `crates/heartbit-cli/README.md` and `docs/platform.md`.
5. Preserve `git blame` / `git log --follow` history across the move via `git mv`.
6. Maintain the B2 CI gate at every commit — `cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` must pass on every commit on the branch.

## Non-Goals

- **Publishing to crates.io.** Out of scope for this round. The split lands on `main`; the actual `cargo publish` happens in a separate small release round after we verify heartbit-cloud against the new layout end-to-end. Prevents irreversible mistakes (you cannot unpublish a crate-name reservation; you can only yank versions).
- **Splitting any further satellites.** `heartbit-daemon`, `heartbit-sensor`, `heartbit-channel`, `heartbit-postgres`, `heartbit-embedding` could each be their own future crate, but they all stay inline in the umbrella for B3. Each can be its own ~half-day follow-up round.
- **Dropping any feature flag.** Every existing umbrella feature continues to work, including `core` (which becomes a back-compat no-op alias). No deprecations.
- **Refactoring inside core during the move.** No clippy fixes beyond the existing gate, no API renames, no module rationalization (e.g., `agent/orchestrator.rs` is 6.8k LOC and was flagged as a god-module — leave it for a separate round).
- **`heartbit-cloud` migration to depend on `heartbit-core` directly.** That's a heartbit-cloud repo decision. B3 only ensures `heartbit-cloud` *can* migrate, not that it must.
- **Toolchain pinning policy.** Flagged separately at the start of this round; needs its own tiny round.

## Design

### Architecture

After B3, the workspace shape is:

```
heartbit/                              workspace root
├── crates/
│   ├── heartbit-core/                 NEW. The framework. ~24-30k LOC.
│   │   ├── src/                       (foundational modules — see Module Placement)
│   │   ├── README.md                  (framework intro, docs.rs landing)
│   │   └── Cargo.toml                 (own [package], minimal feature graph)
│   ├── heartbit/                      EXISTING. Now the umbrella. ~5k LOC inline + glob re-export.
│   │   ├── src/lib.rs                 `pub use heartbit_core::*;` + platform-gated mods
│   │   └── src/{daemon,sensor,workflow,channel/{telegram,discord,slack},lsp,sandbox,
│   │             auth/{jwt,vault},...}/
│   ├── heartbit-cli/                  EXISTING. Unchanged at the dep level (depends on heartbit umbrella).
│   ├── heartbit-gateway/              EXISTING. Unchanged.
│   └── heartbit-macro/                EXISTING. Unchanged.
└── README.md                          REWRITTEN. Framework-first.
```

**Dependency graph:**

```
heartbit-cli ─────────► heartbit (umbrella) ─────────► heartbit-core
heartbit-gateway ─────► heartbit ──────────────┘
heartbit-cloud ───────► heartbit OR heartbit-core (their choice)
external lib user ────► heartbit-core (the recommended path)
```

The umbrella becomes a thin layer that adds the platform-specific gated modules on top of core. Library users target `heartbit-core` directly; runtime/platform users keep using `heartbit`.

### Module Placement

#### Lives in `heartbit-core/src/`

```
agent/                  AgentRunner, Orchestrator, all sub-agent kinds (audit, batch,
                        blackboard, cache, context, dag, debate, events, guardrail,
                        guardrails/{tool_policy, llm_judge, secret_scanner, injection,
                        pii, behavioral, action_budget, content_fence, sensor_security},
                        instructions, mixture, observability, permission, prompts,
                        pruner, routing, tool_filter, voting, workflow, runner.rs, mod.rs)
auth/                   ct.rs only. (mod.rs trims down: jwt + vault stay in umbrella)
channel/                bridge.rs, session.rs (traits + InMemorySessionStore), types.rs
config/                 all the data structs (HeartbitConfig, AgentConfig, ...)
error.rs
eval/                   EvalRunner, EvalCase, scorer configs
http.rs                 SafeUrl, IpPolicy, factories (B2 work)
knowledge/              chunker, BM25, Embedding trait, loader. (No LocalEmbeddingProvider.)
llm/                    LlmProvider trait + Anthropic / OpenRouter / Gemini / OpenAICompat,
                        RetryingProvider, CascadingProvider, BoxedProvider, types
memory/                 Memory trait + InMemoryStore + NamespacedMemory + memory tools
                        (No PostgresMemoryStore, no LocalEmbeddingProvider)
signal.rs               shutdown handler
store/                  TaskStore trait + in-memory impl + types (No PostgresTaskStore)
template/               Tera-based prompt templates
tool/                   Tool trait + MCP client + all in-process builtins (read, write,
                        edit, bash, patch, todo, web_search, web_fetch, twitter_post,
                        image_generate, tts, a2a (gated), ...)
workspace.rs            workspace path normalization
```

#### Stays in `heartbit/` umbrella (re-exported via `pub use heartbit_core::*;` plus inline-gated)

| Module | Reason |
|---|---|
| `daemon/` | Kafka, Axum, cron, SSE — heavy infra, gated behind `daemon`. |
| `sensor/` | Depends on daemon. |
| `workflow/` | Restate-specific, gated behind `restate`. |
| `channel/{telegram, discord, slack}/` | Platform-specific bot adapters, each with its own dep. |
| `memory::PostgresMemoryStore` | Needs sqlx; gated behind `postgres`. |
| `memory::embedding::local::LocalEmbeddingProvider` | Needs fastembed/ONNX; gated behind `local-embedding`. |
| `channel::session::PostgresSessionStore` | Needs sqlx. |
| `store::postgres::PostgresTaskStore` | Needs sqlx. |
| `lsp/` | Niche, depends on tokio process spawn — feels integration-shaped. Could be promoted later if a core feature requires it. |
| `sandbox.rs` | Linux landlock, feature-gated. |
| `auth/jwt.rs` | Specific protocol with `jsonwebtoken` + 6 transitive deps; treat as integration, not primitive. |
| `auth/vault.rs` | Specific algorithms (AES-GCM, Argon2 + 3 deps); treat as integration. |

The line drawn between core and umbrella: **primitives stay in core, integrations stay in umbrella**. `auth::ct` is a primitive. `auth::jwt` and `auth::vault` are integrations. LSP is an integration.

### Cargo.toml & Feature Graph

#### `heartbit-core/Cargo.toml`

```toml
[package]
name = "heartbit-core"
version.workspace = true
edition = "2024"
authors.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
documentation = "https://docs.rs/heartbit-core"
description = "The Rust agentic framework — agents, tools, LLM providers, memory, evaluation."
readme = "README.md"
keywords = ["agent", "llm", "mcp", "ai", "framework"]
categories = ["development-tools", "asynchronous"]

[dependencies]
# Core foundationals — unconditional. Inherited from workspace.
tokio.workspace = true
reqwest.workspace = true
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true
tracing.workspace = true
bytes.workspace = true
futures.workspace = true
toml.workspace = true
chrono.workspace = true
uuid.workspace = true
glob.workspace = true
jsonschema.workspace = true
walkdir.workspace = true
regex.workspace = true
subtle.workspace = true        # for auth::ct (B2)
# (Add any other workspace deps that the moved core modules actually import,
#  verified empirically during step 9 of the migration sequencing.)

# Optional, gated:
a2a-sdk = { workspace = true, optional = true }
heartbit-macro = { workspace = true, optional = true }

[features]
default = []
a2a   = ["dep:a2a-sdk"]        # turns on tool::a2a
macro = ["dep:heartbit-macro"] # re-exports proc-macros from heartbit-core
```

The `core` feature flag is dropped from `heartbit-core` itself (it was always-on by definition; making it a feature inside the framework crate has no point). The umbrella keeps `core` as a back-compat alias.

#### `heartbit/Cargo.toml` (the umbrella)

```toml
[dependencies]
heartbit-core = { path = "../heartbit-core" }

# Platform deps for inline-gated modules (unchanged from today, modulo
# what core absorbed):
rdkafka = { workspace = true, optional = true }
sqlx    = { workspace = true, optional = true }
pgvector = { workspace = true, optional = true }
restate-sdk = { workspace = true, optional = true }
teloxide = { workspace = true, optional = true }
tokio-tungstenite = { workspace = true, optional = true }
fastembed = { workspace = true, optional = true }
landlock = { workspace = true, optional = true }
cron = { workspace = true, optional = true }
prometheus = { workspace = true, optional = true }
jsonwebtoken = { workspace = true, optional = true }
aes-gcm = { workspace = true, optional = true }
argon2 = { workspace = true, optional = true }
rand = { workspace = true, optional = true }
axum = { workspace = true, optional = true }
hmac = { workspace = true, optional = true }
sha1 = { workspace = true, optional = true }
hex = { workspace = true, optional = true }

[features]
default = ["core"]
core    = []                                    # back-compat alias (does nothing now)
a2a     = ["heartbit-core/a2a"]                 # forward to core's a2a feature
macro   = ["heartbit-core/macro"]               # forward to core's macro feature
kafka   = ["dep:rdkafka"]
daemon  = ["kafka", "dep:cron", "dep:prometheus", "dep:jsonwebtoken", "dep:axum"]
sensor  = ["daemon", "dep:hmac", "dep:sha1", "dep:hex"]
restate = ["dep:restate-sdk"]
postgres = ["dep:sqlx", "dep:pgvector"]
telegram = ["dep:teloxide"]
discord = ["dep:tokio-tungstenite"]
slack   = ["dep:tokio-tungstenite"]
local-embedding = ["dep:fastembed"]
sandbox = ["dep:landlock"]
vault   = ["dep:aes-gcm", "dep:argon2", "dep:rand"]
full    = ["daemon", "sensor", "restate", "postgres", "a2a", "telegram", "discord", "slack", "vault"]
```

The umbrella's feature graph is essentially today's, minus the `dep:subtle` from `sensor` (already done in B2) and minus anything that became unconditional in core. `heartbit-cli` keeps `features = ["full"]` and works exactly as before.

**Workspace `Cargo.toml`:** add `members = [..., "crates/heartbit-core"]`. All workspace deps already declared — no other changes.

### Umbrella Re-Export Strategy

`crates/heartbit/src/lib.rs` becomes ~30 lines, structured as:

```rust
//! Heartbit umbrella crate. Re-exports `heartbit-core` (the framework) and
//! adds platform integrations behind feature gates.

pub use heartbit_core::*;                       // glob re-export

#[cfg(feature = "daemon")]   pub mod daemon;
#[cfg(feature = "sensor")]   pub mod sensor;
#[cfg(feature = "restate")]  pub mod workflow;
#[cfg(all(target_os = "linux", feature = "sandbox"))] pub mod sandbox;
pub mod lsp;                                    // unconditional (matches today)

#[cfg(feature = "daemon")]
pub use auth::jwt::{JwksClient, JwtValidator};

#[cfg(feature = "vault")]
pub mod vault;                                  // src moved here from auth::vault

// Platform-gated impls of core traits stay inline as direct submodules:
#[cfg(feature = "postgres")]
pub use { memory::PostgresMemoryStore, channel::PostgresSessionStore, store::PostgresTaskStore, ... };
#[cfg(feature = "local-embedding")]
pub use memory::embedding::LocalEmbeddingProvider;

// Channel platform adapters stay inline:
#[cfg(feature = "telegram")] pub use channel::telegram::*;
#[cfg(feature = "discord")]  pub use channel::discord::*;
#[cfg(feature = "slack")]    pub use channel::slack::*;
```

The current ~75 explicit `pub use agent::audit::{AuditMode, ...}` style lines in the umbrella's `lib.rs` all get deleted — they're maintenance debt that the glob covers wholesale. Any item that should be public in core but *not* re-exported by the umbrella gets demoted (`pub` → `pub(crate)`) in core itself.

### Migration Mechanics & Sequencing

**Constraint:** `cargo test --workspace` must pass at the end of every commit on the branch. The B2 work made this gate strictly enforced; B3 must respect it for review hygiene and bisectability.

**Strategy:** ~12 small commits, each green, ordered by dependency depth (leaves first):

1. **Scaffold `heartbit-core` crate.** Add `crates/heartbit-core/{Cargo.toml, src/lib.rs, README.md}` with placeholder contents (`pub mod placeholder { }` so it compiles). Add to workspace `members`. Verify gate green.

2. **Move `error.rs` and `signal.rs`.** `git mv` both files; fix umbrella's `pub mod` lines to `pub use heartbit_core::{error, signal};`. Leaf modules — easiest first. Verify gate green.

3. **Move `http.rs` and `auth/ct.rs`.** Both just-added in B2, both leaf modules. Migrate the imports in `tool/builtins/webfetch.rs`, `tool/a2a.rs`, sensor sources, etc., to `heartbit_core::http::*` paths. Verify gate green.

4. **Move `workspace.rs`, `template/`, `eval/`, `knowledge/`.** Mid-tier leaves. Verify gate.

5. **Move `llm/` and `config/`.** Slightly trickier: `config` references many module paths; update them as paths shift. Verify gate.

6. **Move `tool/`.** Largest single move. Includes MCP client, ~14 builtins, and the gated a2a tool. Verify gate.

7. **Move `memory/` (excluding `postgres.rs` and `embedding/local.rs`).** Foundational `Memory` trait + `InMemoryStore` + `NamespacedMemory`. Postgres impl and local-embedding impl stay in umbrella. Verify gate.

8. **Move `store/` (excluding `postgres.rs`).** Same shape — trait + InMem only. Verify gate.

9. **Move `agent/` and `channel/{bridge, session, types}`.** Largest commit by diffstat. The cross-references inside agent (orchestrator → guardrail → memory → tool) all need rewiring; most of the rewires are mechanical because everything sits inside `heartbit-core::` after this commit, so internal `use crate::tool::Tool` literally still works (just resolved within `heartbit_core` instead of `heartbit`). Verify gate.

10. **Convert umbrella to thin re-export.** `crates/heartbit/src/lib.rs` becomes ~30 lines (glob + platform-gated mods). Delete the now-redundant ~75 explicit `pub use` lines. Verify gate.

11. **Move `auth/{jwt, vault}` from core back into umbrella.** Per the placement rules. Done as a final sweep so step 9 isn't burdened with it. Verify gate.

12. **Test sweep.** Run all four gate commands plus `cargo build -p heartbit-core --no-default-features` (the "minimal core compiles standalone" assertion) and `cargo doc --workspace --no-deps`. Resolve any remaining doctest path issues. Final commit on the branch.

Each commit is independently green and reviewable. PR can be one squash-merge or 12 individual commits — your call at merge time.

### README & Positioning

**Top-level `README.md` (rewritten):** framework-first. Quickstart with `cargo add heartbit-core`, ≤30-line code example showing AgentRunner + provider + a tool, feature table, link to docs.rs. Platform content moves to `crates/heartbit-cli/README.md` and `docs/platform.md` and is linked under a "Want the full multi-tenant runtime / platform?" section.

**`crates/heartbit-core/README.md` (new):** focused framework intro, ~150 lines max. This is what docs.rs renders as the front matter when someone visits `https://docs.rs/heartbit-core`. No platform content.

**`crates/heartbit-cli/README.md` (new or rewritten):** operator/platform-focused. Pulls in the demoted content from today's top-level: daemon mode, Kafka config, dashboard, multi-tenant deployment.

**`docs/platform.md`:** architecture detail for the platform — what daemon mode is, how the gateway works, how multi-tenancy is enforced. Existing `docs/` content (memory.md, daemon.md, sensors.md, etc.) is largely fine and stays as-is; this file is the platform-overview entry point.

**`CHANGELOG.md`:** new `## Unreleased` entry: "B3 — `heartbit-core` extracted as the official Rust agentic framework. No public API changes; existing imports continue to work."

### Test Migration

- **Inline `#[cfg(test)] mod tests { use super::*; }` blocks**: move with their parent modules automatically via `git mv`. No path changes required.
- **Cross-module integration tests** (`crates/heartbit/tests/sensor_pipeline_e2e.rs`): reference `heartbit::*` which keeps working via the umbrella's glob re-export. No path rewrites expected.
- **Shell-based E2E tests** (`tests/*.sh` at repo root): exercise the CLI binary, not the lib. Unaffected.
- **Doctest path sweep**: at step 12, `cargo test --doc -p heartbit-core` and `cargo test --doc -p heartbit` must both succeed. If a doctest used a deep internal path (e.g., `heartbit::agent::orchestrator::SubAgentDef`) that shifts, fix the doctest path.

### Risks

- **Hidden cross-module references breaking the move.** Likeliest failure: `use crate::foo::bar` somewhere expects `foo` in the same crate, but `foo` moved to core while the using code stayed in umbrella. Mitigation: per-step gate verification (sequencing above).
- **Doctest paths in rustdoc comments.** `///` examples that use `heartbit::AgentRunner` continue to compile (umbrella re-exports). But a doctest specifically using an internal path may shift if module depth changes. Sweep at step 12.
- **Cargo lockfile noise.** Adding a workspace member produces a `Cargo.lock` diff. Expected.
- **`heartbit-cloud` external dep.** Lives in a separate repo; depends on `heartbit`. Since the umbrella's public API is unchanged, heartbit-cloud should keep working without code changes. Confirm by running heartbit-cloud's CI against a path-deps version pinned to this branch before merging B3 to main.
- **Unintended public API growth via the glob.** `pub use heartbit_core::*;` re-exports everything `pub` from core's `lib.rs`. Mitigation: audit core's `lib.rs` at step 10 — anything that should be `pub(crate)` gets demoted before the umbrella starts globbing.
- **Toolchain-version drift mid-flight.** B2 surfaced this (Rust 1.93→1.95). B3 doesn't introduce new exposure but the round is multi-week. Stay disciplined: `rustup update` before each clippy verification.

### Exit Criteria

1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace --no-run` passes.
4. `cargo test --workspace --lib` passes (all current tests, including those that ended up in heartbit-core).
5. `cargo build -p heartbit-core --no-default-features` succeeds — proves the minimal-core compile graph works.
6. `cargo doc --workspace --no-deps` succeeds — proves all rustdoc paths still resolve.
7. `heartbit-cli` continues to build; a smoke test (`heartbit run "say hi"` with a stub provider) returns sensible output.
8. Top-level `README.md` is the framework story; platform content has moved to `crates/heartbit-cli/README.md` and `docs/platform.md`.
9. `crates/heartbit-core/README.md` exists and is the docs.rs landing surface.
10. `CHANGELOG.md` has the `## Unreleased` entry.
11. Public API surface check: `cargo public-api` (or eyeball comparison of pre/post `crates/heartbit/src/lib.rs` `pub use` lines) shows no removed or renamed public items in the umbrella's surface.

### Out of Scope

- `cargo publish` to crates.io — separate small follow-up round.
- Splitting any further satellites — `heartbit-daemon`, `heartbit-sensor`, `heartbit-channel`, `heartbit-postgres`, `heartbit-embedding` all stay inline in the umbrella. Each can be its own future round.
- Dropping any feature flag — `core` becomes a back-compat alias; no deprecations.
- Refactoring inside core during the move.
- `heartbit-cloud` migration to depend on `heartbit-core` directly.
- Toolchain pinning policy.
