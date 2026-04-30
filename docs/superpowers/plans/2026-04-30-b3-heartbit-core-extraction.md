# B3 — `heartbit-core` Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `crates/heartbit-core/` as a workspace member from the existing `crates/heartbit/`, making `heartbit-core` the recommended `cargo add` target for library users while preserving every existing public API path through the `heartbit` umbrella crate.

**Architecture:** Move foundational modules (agent, llm, memory traits + InMem, tool + builtins, error, http, auth::ct, config, knowledge, eval, template, store traits + InMem, channel base traits + InMem session, workspace, signal) into the new crate. The existing `heartbit` crate becomes a thin umbrella that does `pub use heartbit_core::*;` and keeps the platform-specific gated modules inline (daemon, sensor, channel adapters, postgres impls, local-embedding, lsp, sandbox, auth::jwt, auth::vault).

**Tech Stack:** Rust 2024 workspace; `git mv` for blame preservation; per-step temporary forwarding pattern in the umbrella's `lib.rs`; existing CI gate from B2 (fmt + clippy `--workspace --all-targets -D warnings` + test `--no-run` + test `--lib`).

**Spec:** `docs/superpowers/specs/2026-04-30-b3-heartbit-core-extraction-design.md`

---

## File Structure

**Create:**
- `crates/heartbit-core/` (new workspace member)
  - `Cargo.toml`
  - `src/lib.rs` (initially placeholder; absorbs ~75 flat re-exports at task 9b)
  - `README.md` (framework intro for docs.rs)
- `crates/heartbit-cli/README.md` (operator/platform-focused; new in this round)
- `docs/platform.md` (architecture overview for the daemon / multi-tenant runtime)

**Modify:**
- Workspace `Cargo.toml`: add `crates/heartbit-core` to `members`.
- `crates/heartbit/Cargo.toml`: add `heartbit-core = { path = "../heartbit-core" }`; trim deps that became core's responsibility; rewire features.
- `crates/heartbit/src/lib.rs`: shrink from ~324 lines to ~30 — delete the ~75 flat re-exports (they migrate to core's lib.rs at task 9b); replace `pub mod` lines with `pub use heartbit_core::*;` glob plus the platform-gated module declarations.
- `README.md` (top-level): rewrite around the framework.
- `CHANGELOG.md`: add `## Unreleased` entry.

**Move (`git mv`):**
- `crates/heartbit/src/error.rs` → `crates/heartbit-core/src/error.rs`
- `crates/heartbit/src/signal.rs` → `crates/heartbit-core/src/signal.rs`
- `crates/heartbit/src/http.rs` → `crates/heartbit-core/src/http.rs`
- `crates/heartbit/src/auth/ct.rs` → `crates/heartbit-core/src/auth/ct.rs`
- `crates/heartbit/src/workspace.rs` → `crates/heartbit-core/src/workspace.rs`
- `crates/heartbit/src/template/` → `crates/heartbit-core/src/template/`
- `crates/heartbit/src/eval/` → `crates/heartbit-core/src/eval/`
- `crates/heartbit/src/knowledge/` → `crates/heartbit-core/src/knowledge/`
- `crates/heartbit/src/llm/` → `crates/heartbit-core/src/llm/`
- `crates/heartbit/src/config/` → `crates/heartbit-core/src/config/` (or `config.rs` if it's a single file — verify with `ls -la crates/heartbit/src/config*`)
- `crates/heartbit/src/tool/` → `crates/heartbit-core/src/tool/`
- `crates/heartbit/src/memory/` → `crates/heartbit-core/src/memory/` (then `git mv` `postgres.rs` and `embedding/local.rs` *back* to umbrella in task 11)
- `crates/heartbit/src/store/` → `crates/heartbit-core/src/store/` (then `git mv` `postgres.rs` *back* to umbrella in task 11)
- `crates/heartbit/src/agent/` → `crates/heartbit-core/src/agent/` (in three sub-tasks: 9a, 9b, 9c)
- `crates/heartbit/src/channel/{bridge.rs, session.rs, types.rs}` → `crates/heartbit-core/src/channel/{bridge.rs, session.rs, types.rs}` (the platform adapters under `channel/{telegram, discord, slack}` stay in the umbrella; the postgres session impl moves back at task 11)

---

## Per-Step Forwarding Pattern (read once, applies to every move-task)

**Why it's needed.** Between steps, the umbrella still has inline modules (e.g., `daemon`, `sensor`, `channel/telegram`) that reference moved modules via `crate::tool::Tool`, `crate::memory::Memory`, etc. Without forwarding, the umbrella stops compiling between steps.

**The pattern.** For every module `M` moved to core in a given step, the same step adds a forwarding line at the top of `crates/heartbit/src/lib.rs`:

```rust
pub use heartbit_core::M;
```

So `crate::M::*` in still-inline umbrella code resolves transparently. At task 10, all the per-step forwardings get deleted and replaced with one `pub use heartbit_core::*;` glob.

**Concretely:** after task 6 (move `tool/`), the umbrella's lib.rs contains a temporary `pub use heartbit_core::tool;` line. After task 9b (move `agent/`), it also has `pub use heartbit_core::agent;`. After task 9c (move `channel/{bridge,session,types}`), it has `pub use heartbit_core::channel;`. Etc. Task 10 removes all of these.

---

## Two-Commit Discipline for Move Tasks

Every move task ships **two commits**:

- **Commit A — mv-only.** Pure `git mv` of the relevant files. Zero content edits. Git's rename detection is most reliable when the diff is exactly a rename. After this commit, the build will likely be broken (modules are in the wrong crate); do not run the gate yet.
- **Commit B — edits.** Add the module declaration in `heartbit-core/src/lib.rs`, add the per-step forwarding line in `heartbit/src/lib.rs`, update Cargo.toml deps if needed, fix any import path adjustments. Run the gate. Commit.

**Why two commits.** `git blame` rename detection degrades when an "edit similarity" threshold is exceeded — co-locating moves and edits in one commit costs you line history. Splitting them keeps `git blame` and `git log -L` working perfectly across the move.

---

## Task 0: Worktree setup

**Files:** none modified yet.

- [ ] **Step 0.1: Confirm B2 has merged to main.**

```bash
cd /home/pleclech/projects/heartbit
git fetch origin
git log --oneline origin/main -3
```
Expected: top of origin/main is the squash-merge of B2 (commit message starts with the B2 PR title or is the merge commit). If B2 hasn't merged yet, **stop and merge B2 first** before starting B3 — B3 depends on the post-B2 codebase shape.

- [ ] **Step 0.2: Sync local main and create the B3 branch + worktree.**

```bash
git checkout main
git pull --rebase
git worktree add .worktrees/b3-heartbit-core -b feat/b3-heartbit-core
cd .worktrees/b3-heartbit-core
```

The branch `feat/b3-heartbit-core` is created off the latest main. Subsequent commits stay local on this branch.

- [ ] **Step 0.3: Verify baseline gate.**

```bash
cargo fmt -- --check && \
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3 && \
cargo test --workspace --no-run 2>&1 | tail -3 && \
cargo test --workspace --lib 2>&1 | tail -3
```

Expected: all four exit 0. If any fails, **stop** — main is not green; B3 cannot start until that is fixed.

---

## Task 1: Scaffold `heartbit-core`

**Files:**
- Create: `crates/heartbit-core/Cargo.toml`
- Create: `crates/heartbit-core/src/lib.rs`
- Create: `crates/heartbit-core/README.md` (placeholder; rewritten in task 13)
- Modify: workspace `Cargo.toml` (add member)

- [ ] **Step 1.1: Create `crates/heartbit-core/Cargo.toml`.**

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
# Empirical-deps pattern: add a dep here only when a subsequent move-task's
# `cargo check -p heartbit-core` fails with "unresolved import". Start minimal.
tokio = { workspace = true }
serde = { workspace = true }
thiserror = { workspace = true }
tracing = { workspace = true }

[features]
default = []
```

- [ ] **Step 1.2: Create `crates/heartbit-core/src/lib.rs` placeholder.**

```rust
//! # heartbit-core
//!
//! The Rust agentic framework — agents, tools, LLM providers, memory, evaluation.
//!
//! Documentation lands here as the crate's docs.rs preamble. The README
//! is rendered above this on docs.rs.

// Modules are added one at a time as subsequent tasks move them in.
```

- [ ] **Step 1.3: Create `crates/heartbit-core/README.md` placeholder.**

```markdown
# heartbit-core

The Rust agentic framework. Full intro lands here in task 13.
```

- [ ] **Step 1.4: Add `crates/heartbit-core` to workspace members.**

In root `Cargo.toml`, find the `[workspace]` section and update `members`:

```toml
[workspace]
resolver = "2"
members = ["crates/heartbit", "crates/heartbit-cli", "crates/heartbit-macro", "crates/heartbit-gateway", "crates/heartbit-core"]
```

- [ ] **Step 1.5: Verify the new crate compiles.**

```bash
cargo build -p heartbit-core 2>&1 | tail -3
```
Expected: `Finished` line, exit 0.

- [ ] **Step 1.6: Verify the workspace gate.**

```bash
cargo fmt -- --check && \
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3 && \
cargo test --workspace --no-run 2>&1 | tail -3
```
All three exit 0.

- [ ] **Step 1.7: Commit.**

```bash
git add Cargo.toml crates/heartbit-core/
git commit -m "feat(core): scaffold heartbit-core workspace member

Empty placeholder crate. Subsequent tasks move modules in one at a time
using git mv to preserve blame history."
```

---

## Task 2: Move `error.rs` and `signal.rs`

**Files:**
- Move: `crates/heartbit/src/error.rs` → `crates/heartbit-core/src/error.rs`
- Move: `crates/heartbit/src/signal.rs` → `crates/heartbit-core/src/signal.rs`
- Modify: `crates/heartbit-core/src/lib.rs` (add module declarations)
- Modify: `crates/heartbit/src/lib.rs` (replace `pub mod error;` and `pub mod signal;` with forwarding)
- Modify: `crates/heartbit-core/Cargo.toml` (add `thiserror` if not already)

- [ ] **Step 2.1: `git mv` both files.**

```bash
git mv crates/heartbit/src/error.rs crates/heartbit-core/src/error.rs
git mv crates/heartbit/src/signal.rs crates/heartbit-core/src/signal.rs
```

- [ ] **Step 2.2: Commit (mv-only).**

```bash
git commit -m "refactor(core): move error.rs and signal.rs (mv-only)"
```

(Build will be broken at this commit. That's expected. The next commit fixes it.)

- [ ] **Step 2.3: Add module declarations in `heartbit-core/src/lib.rs`.**

Edit `crates/heartbit-core/src/lib.rs` — add at the bottom:

```rust
pub mod error;
pub mod signal;
```

- [ ] **Step 2.4: Replace umbrella's module declarations with forwarding.**

In `crates/heartbit/src/lib.rs`, find the lines:

```rust
pub mod error;
pub mod signal;
```

Replace with:

```rust
pub use heartbit_core::error;
pub use heartbit_core::signal;
```

- [ ] **Step 2.5: Verify gate.**

```bash
cargo build -p heartbit-core 2>&1 | tail -3
cargo build --workspace 2>&1 | tail -3
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
```

If `cargo build -p heartbit-core` fails with "unresolved import" referring to a missing dep (e.g., `thiserror`), add the workspace dep to `crates/heartbit-core/Cargo.toml`:

```toml
thiserror = { workspace = true }
```

Repeat the build until clean.

- [ ] **Step 2.6: Commit (edits).**

```bash
git add crates/heartbit-core/src/lib.rs crates/heartbit-core/Cargo.toml crates/heartbit/src/lib.rs
git commit -m "refactor(core): wire error and signal modules

heartbit-core now declares pub mod error and pub mod signal. The
umbrella forwards via pub use heartbit_core::{error, signal} so existing
heartbit::error::Error and heartbit::signal::* imports keep working."
```

---

## Task 3: Move `http.rs` and `auth/ct.rs`

**Files:**
- Move: `crates/heartbit/src/http.rs` → `crates/heartbit-core/src/http.rs`
- Move: `crates/heartbit/src/auth/ct.rs` → `crates/heartbit-core/src/auth/ct.rs`
- Modify: `crates/heartbit-core/src/lib.rs`
- Modify: `crates/heartbit-core/src/auth/mod.rs` (create — minimal)
- Modify: `crates/heartbit/src/auth/mod.rs` (remove `pub mod ct`)
- Modify: `crates/heartbit/src/lib.rs` (add forwarding)
- Modify: `crates/heartbit-core/Cargo.toml` (add `reqwest`, `subtle`)

- [ ] **Step 3.1: `git mv` files.**

```bash
git mv crates/heartbit/src/http.rs crates/heartbit-core/src/http.rs
mkdir -p crates/heartbit-core/src/auth
git mv crates/heartbit/src/auth/ct.rs crates/heartbit-core/src/auth/ct.rs
```

- [ ] **Step 3.2: Commit (mv-only).**

```bash
git add crates/heartbit-core/src/auth/
git commit -m "refactor(core): move http.rs and auth/ct.rs (mv-only)"
```

- [ ] **Step 3.3: Create `crates/heartbit-core/src/auth/mod.rs`.**

```rust
//! Authentication primitives.

pub mod ct;
```

- [ ] **Step 3.4: Add module declarations in `heartbit-core/src/lib.rs`.**

Append:

```rust
pub mod auth;
pub mod http;
```

- [ ] **Step 3.5: Update `heartbit/src/auth/mod.rs`.**

Find `pub mod ct;` and remove it. Then re-export from core:

```rust
pub use heartbit_core::auth::ct;
```

- [ ] **Step 3.6: Add forwarding in `heartbit/src/lib.rs`.**

Find `pub mod http;` (or whatever the current declaration is) and replace with:

```rust
pub use heartbit_core::http;
```

The `auth` module stays declared in the umbrella as a normal `pub mod auth;` because it still has umbrella-side submodules (`jwt`, `vault`).

- [ ] **Step 3.7: Add `reqwest` and `subtle` to `heartbit-core/Cargo.toml`.**

```toml
reqwest = { workspace = true }
subtle = { workspace = true }
```

- [ ] **Step 3.8: Verify gate.**

```bash
cargo build --workspace 2>&1 | tail -3
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib auth::ct:: 2>&1 | tail -5
cargo test --workspace --lib http:: 2>&1 | tail -5
cargo test --workspace --no-run 2>&1 | tail -3
```

All gates green; the auth::ct and http unit tests must still pass.

- [ ] **Step 3.9: Commit (edits).**

```bash
git add -A
git commit -m "refactor(core): wire http and auth::ct modules

http.rs and auth/ct.rs (both added in B2) now live in heartbit-core. The
umbrella forwards via pub use heartbit_core::http and re-exports
heartbit_core::auth::ct from the umbrella's auth module."
```

---

## Task 4: Move `workspace.rs`, `template/`, `eval/`, `knowledge/`

**Files:**
- Move: `crates/heartbit/src/workspace.rs` → `crates/heartbit-core/src/workspace.rs`
- Move: `crates/heartbit/src/template/` → `crates/heartbit-core/src/template/`
- Move: `crates/heartbit/src/eval/` → `crates/heartbit-core/src/eval/`
- Move: `crates/heartbit/src/knowledge/` → `crates/heartbit-core/src/knowledge/`
- Modify: `crates/heartbit-core/src/lib.rs`, `crates/heartbit/src/lib.rs`, `crates/heartbit-core/Cargo.toml`

- [ ] **Step 4.1: `git mv` files.**

```bash
git mv crates/heartbit/src/workspace.rs crates/heartbit-core/src/workspace.rs
git mv crates/heartbit/src/template crates/heartbit-core/src/template
git mv crates/heartbit/src/eval crates/heartbit-core/src/eval
git mv crates/heartbit/src/knowledge crates/heartbit-core/src/knowledge
```

- [ ] **Step 4.2: Commit (mv-only).**

```bash
git commit -m "refactor(core): move workspace, template, eval, knowledge (mv-only)"
```

- [ ] **Step 4.3: Add module declarations in `heartbit-core/src/lib.rs`.**

```rust
pub mod eval;
pub mod knowledge;
pub mod template;
pub mod workspace;
```

- [ ] **Step 4.4: Replace umbrella's declarations with forwarding.**

In `crates/heartbit/src/lib.rs`, find:

```rust
pub mod eval;
pub mod knowledge;
pub mod template;
pub mod workspace;
```

Replace with:

```rust
pub use heartbit_core::{eval, knowledge, template, workspace};
```

- [ ] **Step 4.5: Run `cargo build`; add deps as needed.**

```bash
cargo build --workspace 2>&1 | tail -10
```

If it fails with `unresolved import`, identify which workspace dep is missing and add it to `crates/heartbit-core/Cargo.toml`. Likely additions: `tera`, `chrono`, `regex`, `walkdir`, `glob`, `bm25`, `pulldown-cmark`. Add the workspace dep, retry, repeat until green.

- [ ] **Step 4.6: Verify gate.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
cargo test --workspace --lib 2>&1 | tail -3
```

- [ ] **Step 4.7: Commit (edits).**

```bash
git add -A
git commit -m "refactor(core): wire workspace, template, eval, knowledge

Mid-tier leaf modules. heartbit-core gains the workspace path normalizer,
prompt template engine, eval framework, and knowledge (chunker + BM25 +
Embedding trait + in-memory KB). The local-embedding impl stays in the
umbrella per the spec; only the trait + chunker + scoring are in core."
```

---

## Task 5: Move `llm/` and `config/`

**Files:**
- Move: `crates/heartbit/src/llm/` → `crates/heartbit-core/src/llm/`
- Move: `crates/heartbit/src/config*` → `crates/heartbit-core/src/config*` (verify whether it's a single `.rs` or a `config/` dir)
- Modify: `crates/heartbit-core/src/lib.rs`, `crates/heartbit/src/lib.rs`, `crates/heartbit-core/Cargo.toml`

- [ ] **Step 5.1: Verify config layout, then `git mv`.**

```bash
ls -la crates/heartbit/src/config* 2>&1
```

If it's a directory:

```bash
git mv crates/heartbit/src/llm crates/heartbit-core/src/llm
git mv crates/heartbit/src/config crates/heartbit-core/src/config
```

If `config.rs` is a single file (alongside a `config/` dir or alone):

```bash
git mv crates/heartbit/src/llm crates/heartbit-core/src/llm
git mv crates/heartbit/src/config.rs crates/heartbit-core/src/config.rs
```

- [ ] **Step 5.2: Commit (mv-only).**

```bash
git commit -m "refactor(core): move llm and config (mv-only)"
```

- [ ] **Step 5.3: Wire in `heartbit-core/src/lib.rs`.**

```rust
pub mod config;
pub mod llm;
```

- [ ] **Step 5.4: Replace umbrella's declarations with forwarding.**

In `crates/heartbit/src/lib.rs`, find the `pub mod config;` and `pub mod llm;` lines. Replace with:

```rust
pub use heartbit_core::{config, llm};
```

- [ ] **Step 5.5: Build, add deps as needed.**

```bash
cargo build --workspace 2>&1 | tail -10
```

Likely additions to `crates/heartbit-core/Cargo.toml`: `serde_json`, `bytes`, `futures`, `toml`, `uuid`, `jsonschema`. Add as failures surface; retry until green.

- [ ] **Step 5.6: Verify gate.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib 2>&1 | tail -3
```

- [ ] **Step 5.7: Commit (edits).**

```bash
git add -A
git commit -m "refactor(core): wire llm and config

LLM provider trait + Anthropic/OpenRouter/Gemini/OpenAICompat clients,
RetryingProvider, CascadingProvider, BoxedProvider. Config data structs
(HeartbitConfig, AgentConfig, MemoryConfig, etc.) — TOML parsing only;
no infra. Umbrella forwards both."
```

---

## Task 6: Move `tool/`

**Files:**
- Move: `crates/heartbit/src/tool/` → `crates/heartbit-core/src/tool/`
- Modify: `crates/heartbit-core/src/lib.rs`, `crates/heartbit/src/lib.rs`, `crates/heartbit-core/Cargo.toml`

- [ ] **Step 6.1: `git mv`.**

```bash
git mv crates/heartbit/src/tool crates/heartbit-core/src/tool
```

- [ ] **Step 6.2: Commit (mv-only).**

```bash
git commit -m "refactor(core): move tool/ (mv-only)"
```

- [ ] **Step 6.3: Wire in `heartbit-core/src/lib.rs`.**

```rust
pub mod tool;
```

- [ ] **Step 6.4: Forward in umbrella.**

In `crates/heartbit/src/lib.rs`, replace `pub mod tool;` with:

```rust
pub use heartbit_core::tool;
```

- [ ] **Step 6.5: Forward `a2a` feature, build.**

The `tool::a2a` module is gated behind the `a2a` feature. The current umbrella `Cargo.toml` has:

```toml
a2a = ["dep:a2a-sdk"]
```

This must change to forward to core's a2a feature. In `crates/heartbit-core/Cargo.toml`, add:

```toml
[dependencies]
# ... existing
a2a-sdk = { workspace = true, optional = true }

[features]
default = []
a2a = ["dep:a2a-sdk"]
```

In `crates/heartbit/Cargo.toml`, change the `a2a` feature line to forward:

```toml
a2a = ["heartbit-core/a2a"]
```

And remove `a2a-sdk` from the umbrella's `[dependencies]` if it lives there (it probably does — check via `grep "a2a-sdk" crates/heartbit/Cargo.toml`).

```bash
cargo build --workspace 2>&1 | tail -10
cargo build --workspace --features full 2>&1 | tail -10
```

If failures surface other workspace deps, add them to `heartbit-core/Cargo.toml` as needed.

- [ ] **Step 6.6: Verify gate.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib tool:: 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
```

- [ ] **Step 6.7: Commit (edits).**

```bash
git add -A
git commit -m "refactor(core): wire tool module + forward a2a feature

Tool trait, MCP client, all in-process builtins (read, write, edit,
bash, patch, todo, web_search, web_fetch, twitter_post, image_generate,
tts, mcp_server, a2a (gated)) move to heartbit-core. The a2a-sdk dep
moves to core; the umbrella's a2a feature now forwards to
heartbit-core/a2a."
```

---

## Task 7: Move `memory/` (excluding `postgres.rs` and `embedding/local.rs`)

**Files:**
- Move: `crates/heartbit/src/memory/` → `crates/heartbit-core/src/memory/`
- Move-back: `crates/heartbit-core/src/memory/postgres.rs` → `crates/heartbit/src/memory/postgres.rs` (after the bulk move)
- Move-back: `crates/heartbit-core/src/memory/embedding/local.rs` → `crates/heartbit/src/memory/embedding/local.rs`
- Modify: `crates/heartbit-core/src/lib.rs`, `crates/heartbit-core/src/memory/mod.rs`, `crates/heartbit-core/src/memory/embedding/mod.rs`, `crates/heartbit/src/lib.rs`

- [ ] **Step 7.1: `git mv` the whole memory dir.**

```bash
git mv crates/heartbit/src/memory crates/heartbit-core/src/memory
```

- [ ] **Step 7.2: Move postgres.rs and local-embedding back to umbrella.**

```bash
mkdir -p crates/heartbit/src/memory crates/heartbit/src/memory/embedding
git mv crates/heartbit-core/src/memory/postgres.rs crates/heartbit/src/memory/postgres.rs
git mv crates/heartbit-core/src/memory/embedding/local.rs crates/heartbit/src/memory/embedding/local.rs
```

- [ ] **Step 7.3: Commit (mv-only).**

```bash
git commit -m "refactor(core): move memory/ (mv-only; postgres + local-embedding stay in umbrella)"
```

- [ ] **Step 7.4: Update memory module declarations.**

In `crates/heartbit-core/src/memory/mod.rs`, remove the line declaring `pub mod postgres;` (gated on `postgres` feature). The postgres module no longer lives in core.

In `crates/heartbit-core/src/memory/embedding/mod.rs`, remove the line declaring the local-embedding module. Same reason.

If those modules had `#[cfg(feature = "postgres")]` or `#[cfg(feature = "local-embedding")]` on them — note the feature names; you'll re-declare them on the umbrella side in the next step.

- [ ] **Step 7.5: Wire in `heartbit-core/src/lib.rs`.**

```rust
pub mod memory;
```

- [ ] **Step 7.6: Replicate the postgres + local-embedding modules in the umbrella.**

In `crates/heartbit/src/memory/mod.rs` (which the move-back created as an empty new file), declare:

```rust
//! Umbrella-side memory implementations.
//!
//! Trait + InMemory + NamespacedMemory live in heartbit_core::memory;
//! the platform-specific impls below stay here behind feature flags.

#[cfg(feature = "postgres")]
pub mod postgres;
#[cfg(feature = "postgres")]
pub use postgres::PostgresMemoryStore;

#[cfg(feature = "local-embedding")]
pub mod embedding;
```

Then create `crates/heartbit/src/memory/embedding/mod.rs`:

```rust
#[cfg(feature = "local-embedding")]
pub mod local;

#[cfg(feature = "local-embedding")]
pub use local::LocalEmbeddingProvider;
```

- [ ] **Step 7.7: Forward in umbrella.**

In `crates/heartbit/src/lib.rs`, the existing `pub mod memory;` line stays — but now it points to the umbrella's local `memory/mod.rs` (which holds postgres + local-embedding). Add a forwarding for the core module too:

```rust
pub use heartbit_core::memory as memory_core;  // for explicit re-references
```

Actually no — the umbrella's `memory` module needs to also re-export the core's memory items so that `heartbit::memory::Memory` (the trait) keeps working. In `crates/heartbit/src/memory/mod.rs`, add at the top:

```rust
pub use heartbit_core::memory::*;
```

This way `heartbit::memory::Memory`, `heartbit::memory::InMemoryStore`, etc. all resolve via the glob, while the postgres + local-embedding impls (declared in this file) sit alongside them.

- [ ] **Step 7.8: Verify build for both feature shapes.**

```bash
cargo build --workspace 2>&1 | tail -10
cargo build --workspace --features full 2>&1 | tail -10
cargo build -p heartbit --features postgres 2>&1 | tail -5
cargo build -p heartbit --features local-embedding 2>&1 | tail -5
```

If failures surface, identify the missing import. Postgres should still find `Memory` / `MemoryEntry` etc. through `super::*` (which now includes the heartbit_core re-exports).

- [ ] **Step 7.9: Verify gate.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib memory:: 2>&1 | tail -5
cargo test --workspace --no-run 2>&1 | tail -3
```

- [ ] **Step 7.10: Commit (edits).**

```bash
git add -A
git commit -m "refactor(core): wire memory module; postgres + local-embedding stay in umbrella

Memory trait + MemoryEntry + InMemoryStore + NamespacedMemory move to
heartbit-core::memory. PostgresMemoryStore (sqlx-backed) and
LocalEmbeddingProvider (fastembed/ONNX) stay in the umbrella behind
their respective features. heartbit::memory::* now re-exports the core's
items via glob plus exposes the platform impls."
```

---

## Task 8: Move `store/` (excluding `postgres.rs`)

**Files:**
- Move: `crates/heartbit/src/store/` → `crates/heartbit-core/src/store/`
- Move-back: `crates/heartbit-core/src/store/postgres.rs` → `crates/heartbit/src/store/postgres.rs`
- Modify: `crates/heartbit-core/src/lib.rs`, `crates/heartbit-core/src/store/mod.rs`, `crates/heartbit/src/lib.rs`, new `crates/heartbit/src/store/mod.rs`

- [ ] **Step 8.1: `git mv` the store dir.**

```bash
git mv crates/heartbit/src/store crates/heartbit-core/src/store
mkdir -p crates/heartbit/src/store
git mv crates/heartbit-core/src/store/postgres.rs crates/heartbit/src/store/postgres.rs
```

- [ ] **Step 8.2: Commit (mv-only).**

```bash
git commit -m "refactor(core): move store/ (mv-only; postgres stays in umbrella)"
```

- [ ] **Step 8.3: Update `heartbit-core/src/store/mod.rs`.**

Remove any `#[cfg(feature = "postgres")] pub mod postgres;` lines.

- [ ] **Step 8.4: Wire in `heartbit-core/src/lib.rs`.**

```rust
pub mod store;
```

- [ ] **Step 8.5: Create `crates/heartbit/src/store/mod.rs`.**

```rust
//! Umbrella-side store implementations.

pub use heartbit_core::store::*;

#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "postgres")]
pub use postgres::PostgresTaskStore;

#[cfg(feature = "postgres")]
pub use postgres::PostgresAuditTrail;
```

- [ ] **Step 8.6: Forward in umbrella's lib.rs.**

The umbrella already has `pub mod store;` (now pointing to the new `store/mod.rs` above) which re-exports core's store via glob. Confirm the line stays and **does not** become `pub use heartbit_core::store;` — because the umbrella has its own postgres impl to add.

If the umbrella's lib.rs has explicit `pub use store::PostgresStore;` lines (the current code has them — verify with `grep "pub use store" crates/heartbit/src/lib.rs`), they continue to work because store/mod.rs's glob re-exports core items and explicit declarations expose postgres types alongside.

- [ ] **Step 8.7: Verify build.**

```bash
cargo build --workspace 2>&1 | tail -5
cargo build --workspace --features full 2>&1 | tail -5
cargo build -p heartbit --features postgres 2>&1 | tail -5
```

- [ ] **Step 8.8: Verify gate.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib store:: 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
```

- [ ] **Step 8.9: Commit (edits).**

```bash
git add -A
git commit -m "refactor(core): wire store; PostgresTaskStore + PostgresAuditTrail stay in umbrella

TaskStore trait + types + in-memory impl move to heartbit-core::store.
Postgres-backed task store and audit trail (sqlx) stay in the umbrella
behind the postgres feature."
```

---

## Task 9a: Move agent leaf submodules

The agent module is large (~30k LOC across 20+ submodules). Sub-divide by dependency depth.

**Move-only-leaves (this task):** submodules that don't reference other agent submodules (other than `mod.rs`'s common types). Concretely: `audit`, `events`, `permission`, `observability`, `cache`, `prompts`, `pruner`, `instructions`, `routing`, `tool_filter`. These are reachable without dragging in the orchestrator/runner/guardrail core.

**Files:**
- Move: each leaf submodule → `crates/heartbit-core/src/agent/<leaf>/` or `<leaf>.rs`
- Modify: `crates/heartbit-core/src/agent/mod.rs` (create — minimal), `crates/heartbit/src/agent/mod.rs` (trim)

- [ ] **Step 9a.1: Verify the leaf list.**

```bash
ls crates/heartbit/src/agent/
```

If the directory listing shows submodules not in the planned leaf list, decide where they go (most likely 9b for orchestration-related code). Confirm before proceeding: only `audit`, `events`, `permission`, `observability`, `cache`, `prompts`, `pruner`, `instructions`, `routing`, `tool_filter` move in this task.

- [ ] **Step 9a.2: Create the agent dir in core.**

```bash
mkdir -p crates/heartbit-core/src/agent
```

- [ ] **Step 9a.3: `git mv` each leaf.**

For each leaf in {audit, events, permission, observability, cache, prompts, pruner, instructions, routing, tool_filter}, the leaf may be `<name>.rs` or `<name>/`. Discover and move:

```bash
for leaf in audit events permission observability cache prompts pruner instructions routing tool_filter; do
  if [ -d "crates/heartbit/src/agent/$leaf" ]; then
    git mv "crates/heartbit/src/agent/$leaf" "crates/heartbit-core/src/agent/$leaf"
  elif [ -f "crates/heartbit/src/agent/$leaf.rs" ]; then
    git mv "crates/heartbit/src/agent/$leaf.rs" "crates/heartbit-core/src/agent/$leaf.rs"
  fi
done
```

- [ ] **Step 9a.4: Commit (mv-only).**

```bash
git commit -m "refactor(core): move agent leaf submodules (mv-only)"
```

- [ ] **Step 9a.5: Create `crates/heartbit-core/src/agent/mod.rs`.**

This file re-declares the moved submodules (and nothing else; the rest of agent moves in 9b/9c).

```rust
//! Agent runtime — partial. Foundational submodules live here; the
//! orchestration core (orchestrator, runner, guardrails) moves in
//! task 9b.

pub mod audit;
pub mod cache;
pub mod events;
pub mod instructions;
pub mod observability;
pub mod permission;
pub mod prompts;
pub mod pruner;
pub mod routing;
pub mod tool_filter;
```

- [ ] **Step 9a.6: Trim `crates/heartbit/src/agent/mod.rs`.**

Find the `pub mod` lines for those 10 leaves and replace each with a re-export:

```rust
pub use heartbit_core::agent::audit;
pub use heartbit_core::agent::cache;
pub use heartbit_core::agent::events;
pub use heartbit_core::agent::instructions;
pub use heartbit_core::agent::observability;
pub use heartbit_core::agent::permission;
pub use heartbit_core::agent::prompts;
pub use heartbit_core::agent::pruner;
pub use heartbit_core::agent::routing;
pub use heartbit_core::agent::tool_filter;
```

The other `pub mod` lines (orchestrator, runner, guardrail, guardrails, batch, blackboard, context, dag, debate, mixture, voting, workflow, etc.) stay declared in the umbrella for now — they move in 9b.

- [ ] **Step 9a.7: Wire `pub mod agent;` in heartbit-core/src/lib.rs.**

Append:

```rust
pub mod agent;
```

(In the umbrella, the `pub mod agent;` line stays — it now points to the umbrella's `agent/mod.rs` which has the partial declarations + re-exports.)

- [ ] **Step 9a.8: Build and verify gate.**

```bash
cargo build --workspace 2>&1 | tail -10
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib agent::audit:: 2>&1 | tail -3
cargo test --workspace --lib agent::events:: 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
```

If failures surface (likely from inter-leaf references — e.g., `events.rs` imports something from a non-leaf agent submodule), fix the import to use the umbrella path (e.g., `use heartbit::agent::orchestrator::SubAgentConfig;`) — but this is a code smell and probably means that submodule is NOT actually a leaf and should move in 9b. Re-evaluate before patching.

- [ ] **Step 9a.9: Commit (edits).**

```bash
git add -A
git commit -m "refactor(core): wire agent leaf submodules

audit, cache, events, instructions, observability, permission, prompts,
pruner, routing, tool_filter — all the agent-side leaves with no
back-edge to the orchestrator/runner. Moves to heartbit-core::agent;
the umbrella's agent module re-exports each leaf for back-compat
(heartbit::agent::audit::* etc. continue to resolve)."
```

---

## Task 9b: Move agent core (orchestrator, runner, guardrails) + migrate flat re-exports

**Move:** the remaining agent submodules — `orchestrator`, `runner`, `guardrail`, `guardrails/*`, `batch`, `blackboard`, `context`, `dag`, `debate`, `mixture`, `voting`, `workflow`, plus `mod.rs` itself (to absorb the now-empty umbrella agent/mod.rs).

**The flat re-export migration happens in this task.** This is the load-bearing one — the ~75 `pub use agent::...`, `pub use llm::...`, `pub use memory::...`, etc. lines in `crates/heartbit/src/lib.rs` move to `crates/heartbit-core/src/lib.rs`.

**Files:**
- Move: each remaining agent submodule
- Modify: `crates/heartbit-core/src/agent/mod.rs` (replace with the original umbrella's), `crates/heartbit-core/src/lib.rs` (absorb the ~75 flat re-exports), `crates/heartbit/src/agent/mod.rs` (delete after empty), `crates/heartbit/src/lib.rs` (delete the ~75 flat re-exports — task 10 then handles the glob)

- [ ] **Step 9b.1: Inventory remaining agent submodules.**

```bash
ls crates/heartbit/src/agent/
```

Expected: `orchestrator.rs`, `runner.rs`, `guardrail.rs`, `guardrails/`, `batch.rs`, `blackboard.rs`, `context.rs`, `dag.rs`, `debate.rs`, `mixture.rs`, `voting.rs`, `workflow.rs`, `mod.rs`. The 10 leaves moved in 9a should be gone.

- [ ] **Step 9b.2: `git mv` each.**

```bash
for sub in orchestrator runner guardrail guardrails batch blackboard context dag debate mixture voting workflow; do
  if [ -d "crates/heartbit/src/agent/$sub" ]; then
    git mv "crates/heartbit/src/agent/$sub" "crates/heartbit-core/src/agent/$sub"
  elif [ -f "crates/heartbit/src/agent/$sub.rs" ]; then
    git mv "crates/heartbit/src/agent/$sub.rs" "crates/heartbit-core/src/agent/$sub.rs"
  fi
done
```

**Do not `git mv` the umbrella's `agent/mod.rs`.** The core already has its own `agent/mod.rs` from task 9a; instead, the next commit edits *both* mod.rs files (extending core's, deleting the umbrella's empty file).

- [ ] **Step 9b.3: Commit (mv-only).**

After the loop above, the umbrella's `crates/heartbit/src/agent/` directory contains only `mod.rs` (untouched) and is otherwise empty. Don't delete it yet; the next commit handles cleanup.

```bash
git commit -m "refactor(core): move agent core (orchestrator, runner, guardrails, …) (mv-only)

This is the heaviest single move-step. Flat re-exports migration follows
in the next commit."
```

- [ ] **Step 9b.4: Extend `heartbit-core/src/agent/mod.rs` with the remaining submodule declarations.**

The file from task 9a currently declares only the 10 leaf submodules. Add the remaining 12:

```rust
pub mod batch;
pub mod blackboard;
pub mod context;
pub mod dag;
pub mod debate;
pub mod guardrail;
pub mod guardrails;
pub mod mixture;
pub mod orchestrator;
pub mod runner;
pub mod voting;
pub mod workflow;
```

If the original umbrella's `crates/heartbit/src/agent/mod.rs` had additional content (re-exports of internal items, helper types, etc.), copy that content over too. To find it, look at the umbrella's mod.rs:

```bash
cat crates/heartbit/src/agent/mod.rs
```

Migrate any content that isn't a `pub mod` line for one of the leaves (already declared in core) — re-export blocks, type aliases, constants, helpers — into `crates/heartbit-core/src/agent/mod.rs`.

After the content migration, delete the umbrella's now-redundant mod.rs:

```bash
rm crates/heartbit/src/agent/mod.rs
rmdir crates/heartbit/src/agent  # if the directory is otherwise empty
```

In `crates/heartbit/src/lib.rs`, remove the line `pub mod agent;` (which previously pointed to the umbrella's now-deleted dir) — replace with the per-step forwarding `pub use heartbit_core::agent;`.

- [ ] **Step 9b.5: Migrate the flat re-exports.**

The current `crates/heartbit/src/lib.rs` has ~75 lines that look like:

```rust
pub use agent::audit::{AuditMode, AuditRecord, AuditTrail, InMemoryAuditTrail};
pub use agent::orchestrator::{Orchestrator, OrchestratorBuilder, SubAgentConfig};
pub use llm::{BoxedProvider, DynLlmProvider};
pub use memory::Confidentiality;
pub use tool::{Tool, ToolOutput, validate_tool_input};
// … etc
```

These lines are what makes `heartbit::AgentRunner`, `heartbit::Tool`, `heartbit::Orchestrator` reachable directly (instead of via `heartbit::agent::AgentRunner`). They must continue to work for downstream consumers.

Move each line to `crates/heartbit-core/src/lib.rs`. Concretely: cut each `pub use ...` line whose path begins with one of `agent::`, `llm::`, `memory::`, `tool::`, `eval::`, `knowledge::`, `config::`, `template::`, `error::`, `signal::`, `workspace::`, `store::`, `channel::bridge::`, `channel::session::` (excluding `PostgresSessionStore`), `channel::types::`, `channel::ChannelBridge`, `channel::ConsolidateSession`, `channel::MediaAttachment`, `channel::RunTask`, `channel::RunTaskInput`, `http::`, `auth::ct`, and paste them into core's lib.rs.

These are the lines that currently reference modules now in heartbit-core. Lines that reference *umbrella-only* items (postgres impls, telegram/discord/slack adapters, daemon, sensor, workflow (Restate), lsp, sandbox, vault, jwt) **stay in the umbrella's lib.rs**.

To be precise, run this command to surface the candidate lines from the current umbrella's lib.rs:

```bash
grep -nE "^pub use (agent|llm|memory|tool|eval|knowledge|config|template|error|signal|workspace|store|http)::" crates/heartbit/src/lib.rs
grep -nE "^pub use channel::(bridge|session|types|ChannelBridge|ConsolidateSession|MediaAttachment|RunTask|RunTaskInput)" crates/heartbit/src/lib.rs
grep -nE "^pub use auth::ct" crates/heartbit/src/lib.rs
```

These are the ~75 lines to migrate. Cut them from `crates/heartbit/src/lib.rs` and paste into `crates/heartbit-core/src/lib.rs`.

**Important:** lines that reference `PostgresSessionStore` or `LocalEmbeddingProvider` or `PostgresMemoryStore` etc. must STAY in the umbrella — those types are not in core.

- [ ] **Step 9b.6: Build and verify gate.**

```bash
cargo build --workspace 2>&1 | tail -10
cargo build --workspace --features full 2>&1 | tail -10
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
```

If a public API path regresses (e.g., `heartbit::AgentRunner` no longer resolves) — that's a sign the corresponding `pub use` line wasn't migrated correctly. The umbrella inherits via temporary `pub use heartbit_core::agent;` and then through the per-step forwarding pattern; re-check that core's lib.rs has the matching `pub use crate::agent::AgentRunner;` after the migration.

- [ ] **Step 9b.7: Commit (edits).**

```bash
git add -A
git commit -m "refactor(core): wire agent core + migrate flat re-exports

Orchestrator, AgentRunner, all guardrails (12), workflow agents (DAG,
voting, debate, mixture-of-agents, sequential, parallel, loop), batch,
blackboard, context, debate. heartbit-core's lib.rs absorbs the ~75
flat re-exports that today live in the umbrella, preserving every
heartbit::AgentRunner / heartbit::Tool / heartbit::Orchestrator flat
shortcut for downstream consumers via the umbrella's per-step
forwardings."
```

---

## Task 9c: Move `channel/{bridge, session, types}`

The channel module is partially in core (bridge, session traits + InMemSession, types) and partially in umbrella (telegram, discord, slack adapters, postgres session impl).

**Files:**
- Move: `crates/heartbit/src/channel/bridge.rs` → `crates/heartbit-core/src/channel/bridge.rs`
- Move: `crates/heartbit/src/channel/session.rs` → `crates/heartbit-core/src/channel/session.rs` (later: split out the postgres-flagged subset back to umbrella if it lives in this file)
- Move: `crates/heartbit/src/channel/types.rs` → `crates/heartbit-core/src/channel/types.rs`
- Modify: `crates/heartbit-core/src/channel/mod.rs` (create — partial), `crates/heartbit/src/channel/mod.rs` (trim)

- [ ] **Step 9c.1: Inspect channel layout.**

```bash
ls crates/heartbit/src/channel/
ls crates/heartbit/src/channel/session* 2>&1
```

Note: `session.rs` likely contains both the trait/InMem store AND the `PostgresSessionStore` (gated by `#[cfg(feature = "postgres")]`). If so, the postgres impl needs to be extracted back to the umbrella in step 9c.5.

- [ ] **Step 9c.2: `git mv` the three files.**

```bash
mkdir -p crates/heartbit-core/src/channel
git mv crates/heartbit/src/channel/bridge.rs crates/heartbit-core/src/channel/bridge.rs
git mv crates/heartbit/src/channel/session.rs crates/heartbit-core/src/channel/session.rs
git mv crates/heartbit/src/channel/types.rs crates/heartbit-core/src/channel/types.rs
```

- [ ] **Step 9c.3: Commit (mv-only).**

```bash
git commit -m "refactor(core): move channel/{bridge, session, types} (mv-only)"
```

- [ ] **Step 9c.4: Create `crates/heartbit-core/src/channel/mod.rs`.**

```rust
//! Channel base traits and in-process implementations.
//!
//! Platform-specific adapters (Telegram, Discord, Slack) and the
//! Postgres-backed session store live in the heartbit umbrella.

pub mod bridge;
pub mod session;
pub mod types;

pub use bridge::*;
pub use session::*;
pub use types::*;
```

- [ ] **Step 9c.5: Extract `PostgresSessionStore` to the umbrella.**

If `crates/heartbit-core/src/channel/session.rs` contains a `#[cfg(feature = "postgres")]` block defining `PostgresSessionStore`, move that block into a new file in the umbrella:

```bash
mkdir -p crates/heartbit/src/channel
# Use sed or your editor to cut the #[cfg(feature = "postgres")] PostgresSessionStore block
# from crates/heartbit-core/src/channel/session.rs
# and paste it into crates/heartbit/src/channel/session_postgres.rs
```

Concretely: open the file, locate the `PostgresSessionStore` definition (and any helper types it uses), cut to a new file `crates/heartbit/src/channel/session_postgres.rs`. The new file's first line should re-import what it needs from heartbit_core:

```rust
//! Postgres-backed implementation of the SessionStore trait.

use heartbit_core::channel::session::{Session, SessionMessage, SessionRole, SessionStore};
// ... rest of the implementation as before
```

- [ ] **Step 9c.6: Wire `pub mod channel;` in `heartbit-core/src/lib.rs`.**

```rust
pub mod channel;
```

- [ ] **Step 9c.7: Update umbrella's `crates/heartbit/src/channel/mod.rs`.**

Replace the existing content (which declared `pub mod bridge; pub mod session; pub mod types;` etc.) with:

```rust
//! Umbrella-side channel: re-export base traits from heartbit_core, add
//! platform adapters and Postgres-backed session impl.

pub use heartbit_core::channel::*;

#[cfg(feature = "postgres")]
mod session_postgres;
#[cfg(feature = "postgres")]
pub use session_postgres::PostgresSessionStore;

#[cfg(feature = "telegram")]
pub mod telegram;
#[cfg(feature = "discord")]
pub mod discord;
#[cfg(feature = "slack")]
pub mod slack;
```

- [ ] **Step 9c.8: Build and verify.**

```bash
cargo build --workspace 2>&1 | tail -5
cargo build --workspace --features full 2>&1 | tail -10
cargo build -p heartbit --features postgres 2>&1 | tail -5
cargo build -p heartbit --features telegram 2>&1 | tail -5
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib channel:: 2>&1 | tail -5
cargo test --workspace --no-run 2>&1 | tail -3
```

- [ ] **Step 9c.9: Commit (edits).**

```bash
git add -A
git commit -m "refactor(core): wire channel base traits; PostgresSessionStore + bot adapters stay in umbrella

Channel bridge trait, session/SessionStore trait + InMemSessionStore,
shared types (WsFrame). Postgres-backed session impl extracted to
umbrella's channel/session_postgres.rs. Telegram/Discord/Slack adapters
remain in the umbrella behind their respective features."
```

---

## Task 10: Convert umbrella to thin re-export

**Files:**
- Modify: `crates/heartbit/src/lib.rs` — drop the per-step forwardings, use `pub use heartbit_core::*;` instead.

- [ ] **Step 10.1: Read the current state of `crates/heartbit/src/lib.rs`.**

After tasks 2–9, the umbrella's lib.rs has ~10–15 `pub use heartbit_core::<module>;` forwarding lines (one per module moved) plus the inline `pub mod daemon;` etc. for platform-gated modules.

- [ ] **Step 10.2: Replace forwardings with the glob.**

Edit `crates/heartbit/src/lib.rs`. Remove every line that looks like:

```rust
pub use heartbit_core::error;
pub use heartbit_core::signal;
pub use heartbit_core::http;
pub use heartbit_core::{eval, knowledge, template, workspace};
pub use heartbit_core::{config, llm};
pub use heartbit_core::tool;
pub use heartbit_core::agent;
// ... all the per-step forwardings from tasks 2–9
```

Replace with one glob at the top of the file:

```rust
pub use heartbit_core::*;
```

The remaining content of the umbrella's lib.rs should be:
- The crate-level `//!` doc.
- `extern crate self as heartbit;` (if present today).
- `pub use heartbit_core::*;`
- The platform-gated `pub mod` lines: `daemon`, `sensor`, `workflow`, `lsp`, `sandbox`, etc.
- The inline platform-only re-exports (`PostgresSessionStore`, `LocalEmbeddingProvider`, telegram/discord/slack adapters, daemon types, lsp types, sandbox SandboxPolicy, etc.).
- The `auth` module declaration (which now contains umbrella-only `jwt` and `vault`; `ct` is re-exported from core via auth/mod.rs's `pub use heartbit_core::auth::ct;`).

The end result is a ~30–60 line file (down from ~324).

- [ ] **Step 10.3: Audit core's `lib.rs` for items that should NOT be globbed.**

```bash
grep -E "^pub" crates/heartbit-core/src/lib.rs | head
```

Anything publicly visible in core's lib.rs is now globbed into the umbrella. If any item is intended as core-internal (visible only to core itself, not heartbit consumers), demote `pub` → `pub(crate)` in core's lib.rs.

For this round, the audit should produce zero demotions — every item in core was previously public in the umbrella, so they should all stay `pub`.

- [ ] **Step 10.4: Verify gate.**

```bash
cargo build --workspace 2>&1 | tail -5
cargo build --workspace --features full 2>&1 | tail -10
cargo build -p heartbit-core --no-default-features 2>&1 | tail -5
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
cargo test --workspace --lib 2>&1 | tail -3
```

All exit 0. The `cargo build -p heartbit-core --no-default-features` is the new "minimal core compiles standalone" assertion — verifies that nothing in core accidentally references an umbrella-only feature.

- [ ] **Step 10.5: Spot-check key public paths still resolve.**

```bash
cat <<'EOF' > /tmp/api_check.rs
fn main() {
    // Flat shortcuts (the load-bearing ones)
    let _: Option<heartbit::AgentRunner<_>> = None::<heartbit::AgentRunner<heartbit::BoxedProvider>>;
    let _: Option<&dyn heartbit::Tool> = None;

    // Module paths
    use heartbit::error::Error as _;
    use heartbit::http::{IpPolicy, SafeUrl};
    use heartbit::auth::ct;

    // Umbrella-only platform paths still work
    #[cfg(feature = "postgres")]
    let _: Option<heartbit::PostgresMemoryStore> = None;
}
EOF
# Just verify the file compiles when added as an example. Cleanup after.
mkdir -p crates/heartbit/examples
cp /tmp/api_check.rs crates/heartbit/examples/api_check.rs
cargo build -p heartbit --example api_check --features full 2>&1 | tail -5
rm crates/heartbit/examples/api_check.rs
```

If this doesn't compile, the public API path it failed on needs to be re-exported correctly — debug before continuing.

- [ ] **Step 10.6: Commit.**

```bash
git add -A
git commit -m "refactor(umbrella): switch to glob re-export from heartbit-core

The umbrella's lib.rs now does pub use heartbit_core::*; subsuming all
the per-step forwardings added in tasks 2-9. Down from ~324 lines to
~50. Existing public paths (heartbit::AgentRunner, heartbit::Tool,
heartbit::error::Error, etc.) all resolve through the glob. Platform
integrations (daemon, sensor, workflow, channel adapters, postgres
impls, local-embedding, lsp, sandbox, auth::jwt, auth::vault) remain
inline behind their feature gates."
```

---

## Task 11: Move `auth/{jwt, vault}` from core back to umbrella

Per the spec, `auth::jwt` and `auth::vault` are integrations (specific protocols with specific deps), not foundational primitives. They belong in the umbrella, not core.

(They never landed in core during this plan — `task 3` only moved `auth/ct.rs`. So this task is verifying placement and moving any incidental files that ended up in core.)

**Files:**
- Verify: `crates/heartbit-core/src/auth/` only contains `mod.rs` and `ct.rs`.
- Move (if needed): any other auth files back to `crates/heartbit/src/auth/`.
- Modify: `crates/heartbit-core/Cargo.toml` (no jwt/vault deps), `crates/heartbit/Cargo.toml` (jwt + vault deps stay).

- [ ] **Step 11.1: Verify core's auth dir.**

```bash
ls crates/heartbit-core/src/auth/
ls crates/heartbit/src/auth/
```

Expected in core: `mod.rs`, `ct.rs`. Expected in umbrella: `mod.rs`, `jwt.rs`, `vault.rs`. If the layout doesn't match — particularly if `jwt.rs` or `vault.rs` somehow landed in core — `git mv` it back to the umbrella now.

- [ ] **Step 11.2: Verify umbrella's `crates/heartbit/src/auth/mod.rs`.**

The current content should be:

```rust
#[cfg(feature = "daemon")]
pub mod jwt;

#[cfg(feature = "daemon")]
pub use jwt::{JwksClient, JwtValidator};

#[cfg(feature = "vault")]
pub mod vault;

pub use heartbit_core::auth::ct;
```

If the `pub use heartbit_core::auth::ct;` line is missing (i.e., `pub mod ct;` was just removed in task 3 without adding the re-export back), add it now.

- [ ] **Step 11.3: Verify Cargo.toml deps.**

```bash
grep -E "jsonwebtoken|aes-gcm|argon2" crates/heartbit-core/Cargo.toml
grep -E "jsonwebtoken|aes-gcm|argon2" crates/heartbit/Cargo.toml
```

Expected: zero matches in core. All three deps in umbrella, gated `optional = true` and forwarded via the `daemon` and `vault` feature lists. If core has any of these as deps, remove them.

- [ ] **Step 11.4: Verify gate.**

```bash
cargo build --workspace --features full 2>&1 | tail -5
cargo build -p heartbit-core --no-default-features 2>&1 | tail -5  # must compile without jwt/vault
cargo build -p heartbit --features vault 2>&1 | tail -5
cargo build -p heartbit --features daemon 2>&1 | tail -5
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --lib auth:: 2>&1 | tail -5
```

- [ ] **Step 11.5: Commit (only if anything actually changed).**

```bash
git status
# If clean (i.e., placement was already correct), skip the commit step.
# If files changed:
git add -A
git commit -m "refactor(auth): confirm jwt + vault placement in umbrella, ct in core

Per the spec: auth::ct is a primitive (constant-time helpers, used by
HMAC and bearer-token compares); it belongs in core. auth::jwt and
auth::vault are integrations (specific protocols with their own dep
trees: jsonwebtoken / aes-gcm + argon2 + rand); they belong in the
umbrella behind the daemon and vault features."
```

If the only thing this task did was verify that previous tasks placed everything correctly (i.e., no diffs), close the task with no commit.

---

## Task 12: Test sweep + minimal-core verification + doctest sweep

**Files:** none modified (this is a verification task).

- [ ] **Step 12.1: Full gate.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
cargo test --workspace --lib 2>&1 | tail -3
```

All four exit 0.

- [ ] **Step 12.2: Minimal-core compile.**

```bash
cargo build -p heartbit-core --no-default-features 2>&1 | tail -3
```

Exit 0. This proves a library user running `cargo add heartbit-core` with zero features gets a working compile.

- [ ] **Step 12.3: Per-feature smoke compile.**

```bash
for f in core a2a macro daemon sensor restate postgres telegram discord slack vault local-embedding sandbox; do
  echo "=== feature: $f ==="
  cargo build -p heartbit --features "$f" --no-default-features 2>&1 | tail -2
done
```

Every feature combination must compile. If any fails, the feature graph wasn't migrated correctly — fix and re-run.

- [ ] **Step 12.4: Doctest sweep.**

```bash
cargo test --doc -p heartbit-core 2>&1 | tail -5
cargo test --doc -p heartbit 2>&1 | tail -5
```

If doctests fail with path-resolution errors (e.g., `///` examples that referenced `heartbit::agent::orchestrator::SubAgentDef` directly), update the doctest paths. Most should pass automatically because the umbrella's glob preserves them.

- [ ] **Step 12.5: Public API spot-check.**

```bash
cat <<'EOF' > /tmp/api_smoke.rs
//! Public API smoke test. Confirms the back-compat surface holds.
use heartbit::{
    AgentRunner, AgentRunnerBuilder, Orchestrator, OrchestratorBuilder,
    Tool, ToolOutput,
    BoxedProvider, AnthropicProvider,
    Memory, InMemoryStore, Confidentiality,
    error::Error,
    http::{SafeUrl, IpPolicy, safe_client_builder, vendor_client_builder},
    auth::ct::{ct_eq_str, contains as ct_contains},
};

fn _it_compiles() {}
EOF
mkdir -p crates/heartbit/examples
cp /tmp/api_smoke.rs crates/heartbit/examples/api_smoke.rs
cargo build -p heartbit --example api_smoke --features full 2>&1 | tail -3
rm crates/heartbit/examples/api_smoke.rs
```

Must compile. If a path doesn't resolve, the corresponding flat re-export wasn't migrated correctly in task 9b — debug and fix.

- [ ] **Step 12.6: heartbit-cli smoke (build only).**

```bash
cargo build -p heartbit-cli --features full 2>&1 | tail -3
```

The CLI must still build. If a runtime smoke is feasible (`HEARTBIT_LLM_PROVIDER=anthropic-stub heartbit run "hi"` style), run it; otherwise build success is sufficient.

- [ ] **Step 12.7: Commit only if any cleanup edits were made; otherwise skip.**

```bash
git status
# If non-empty: commit. Otherwise: no commit.
```

---

## Task 13: README rewrite + CHANGELOG entry

**Files:**
- Modify: `README.md` (top-level — rewrite around the framework)
- Create: `crates/heartbit-cli/README.md` (operator/platform-focused; absorbs the demoted content)
- Create: `docs/platform.md` (architecture overview for daemon mode, gateway, multi-tenancy)
- Modify: `crates/heartbit-core/README.md` (full intro, replacing the placeholder from task 1)
- Modify: `CHANGELOG.md`

- [ ] **Step 13.1: Read the current top-level `README.md` to extract content.**

```bash
wc -l README.md
head -200 README.md
```

Identify which sections are framework-relevant (quickstart, examples, tools list, providers, guardrails) vs platform-relevant (daemon mode, dashboard, multi-tenant deployment, Kafka, Postgres operator setup).

- [ ] **Step 13.2: Rewrite top-level `README.md`.**

Replace its contents with the framework-first structure:

```markdown
# Heartbit — the Rust agentic framework

[![crates.io](https://img.shields.io/crates/v/heartbit-core.svg)](https://crates.io/crates/heartbit-core)
[![docs.rs](https://docs.rs/heartbit-core/badge.svg)](https://docs.rs/heartbit-core)
[![CI](https://github.com/heartbit-ai/heartbit/workflows/CI/badge.svg)](https://github.com/heartbit-ai/heartbit/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A production-grade framework for building LLM-powered agents in Rust.
Type-safe, async-native, and runtime-agnostic.

## Quickstart

```bash
cargo add heartbit-core
```

```rust
use std::sync::Arc;
use heartbit_core::{AgentRunner, AnthropicProvider, BoxedProvider, RetryingProvider};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let provider = Arc::new(BoxedProvider::new(
        RetryingProvider::with_defaults(
            AnthropicProvider::new(
                std::env::var("ANTHROPIC_API_KEY").unwrap(),
                "claude-sonnet-4-20250514",
            ),
        ),
    ));

    let mut agent = AgentRunner::builder(provider)
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let output = agent.execute("What is Rust?").await?;
    println!("{}", output.content);
    Ok(())
}
```

## Features

- **ReAct agent loop** with parallel tool execution via `tokio::JoinSet`.
- **LLM providers**: Anthropic, OpenRouter, Gemini, OpenAI-compatible. Retry, cascade, prompt caching built in.
- **Built-in tools**: `web_fetch` (with SSRF defense), `web_search`, file `read`/`write`/`edit`, `bash`, `patch`, `todo`, `image_generate`, `tts`, `twitter_post`, MCP client, A2A.
- **Memory**: trait + `InMemoryStore` + `NamespacedMemory`. Postgres-backed impl in the [`heartbit`](crates/heartbit) umbrella.
- **Guardrails**: 12 of them — LLM judge, secret scanner, PII, content fence, action budget, behavioral monitor, tool policy, injection classifier, sensor security, …
- **Workflow agents**: Sequential / Parallel / Loop / DAG / Voting / Debate / Mixture-of-agents.
- **Eval framework**: `EvalRunner`, `EvalCase`, scorer configs.
- **Multi-tenant primitives**: workspace jails, namespaced memory, guardrail kill-switch, constant-time auth helpers.

## Crate layout

| Crate | What it is |
|---|---|
| [`heartbit-core`](crates/heartbit-core) | The framework. ← `cargo add` this. |
| [`heartbit`](crates/heartbit) | Umbrella + platform integrations: Postgres, Telegram/Discord/Slack adapters, Restate workflows, fastembed local embeddings, vault, JWT validator, daemon mode. |
| [`heartbit-cli`](crates/heartbit-cli) | The binary: `heartbit run`, `heartbit chat`, `heartbit serve`, `heartbit daemon`. |
| [`heartbit-gateway`](crates/heartbit-gateway) | Ingestion gateway — cron, sensors, webhooks to Kafka. |
| [`heartbit-macro`](crates/heartbit-macro) | Proc macros for tool definitions. |

## Want the full multi-tenant runtime / platform?

The platform side — daemon mode, Kafka-backed task queue, Axum HTTP API, multi-tenant JWT auth, sandboxed workspaces, dashboard — is documented in:

- [`crates/heartbit-cli/README.md`](crates/heartbit-cli/README.md) — operator-facing: how to run the daemon, configure Kafka, etc.
- [`docs/platform.md`](docs/platform.md) — architecture: how the platform pieces fit together.

## Documentation

- [API reference (heartbit-core)](https://docs.rs/heartbit-core)
- [`docs/`](docs/) — deep-dives on memory, sensors, daemon mode, etc.
- [Configuration reference](docs/configuration.md)

## Examples

See [`crates/heartbit/examples/`](crates/heartbit/examples/) for runnable examples.

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE), [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security-related reports: see [SECURITY.md](SECURITY.md).

## Acknowledgements

heartbit-core is the engine for [heartbit-cloud](https://github.com/heartbit-ai/heartbit-cloud), an Agents-as-a-Service platform. The framework is independently usable and licensed under MIT OR Apache-2.0.
```

If any of the linked doc files don't exist (e.g., `docs/configuration.md`), either remove the link from the README or create a stub for that file. Don't ship a README with broken internal links.

- [ ] **Step 13.3: Create `crates/heartbit-cli/README.md`.**

Migrate the platform/daemon content from the old top-level README. Skeleton:

```markdown
# heartbit-cli — the Heartbit binary

The `heartbit` CLI is the operator-facing entry point for the Heartbit
multi-tenant runtime. Library users should see [`heartbit-core`](../heartbit-core/README.md).

## Install

```bash
cargo install heartbit-cli
```

## Subcommands

- `heartbit run "prompt"` — single-shot agent run (env-config; uses `ANTHROPIC_API_KEY` etc.)
- `heartbit chat` — interactive REPL with the default agent
- `heartbit serve` — Restate worker for durable agent execution
- `heartbit daemon` — Kafka-backed multi-tenant runtime with HTTP API + SSE event streaming
- `heartbit submit` / `status` / `approve` — interact with a running daemon
- `heartbit eval` — run eval suites against agents

## Daemon mode

[Migrate the daemon content from the old top-level README — Kafka config, Postgres setup, dashboard URL, multi-tenant JWT setup. See docs/platform.md for architecture context.]

## Configuration

[Migrate config-file content. Reference `daemon-dev.toml.example` and `heartbit.toml`.]

## Operations

[Migrate operator-facing topics: logs, metrics, OpenTelemetry, deployment.]
```

Substitute `[Migrate ...]` placeholders with the actual content cut from the old top-level README. Use `git diff HEAD~1 README.md` (or refer to the README before this task's rewrite, available via `git show HEAD~1:README.md`) to source the content.

- [ ] **Step 13.4: Create `docs/platform.md`.**

```markdown
# Heartbit Platform Architecture

## What "the platform" means

Heartbit ships in two shapes:

1. **The framework** — `heartbit-core`, a library you `cargo add` and embed
   in your own application. Single-process, no infrastructure dependencies.
2. **The platform** — the daemon mode in `heartbit-cli` plus
   `heartbit-gateway`, providing a multi-tenant Agents-as-a-Service runtime.

This document covers the platform.

## Components

[Architecture diagram and component descriptions: daemon, gateway, Kafka topics,
 Postgres tables, Restate workflows, channel adapters, dashboard.]

## Multi-tenancy

[How tenant isolation works: JWT auth, namespaced memory, sandboxed workspaces,
 per-tenant Twitter credentials, guardrail policies, audit trails.]

## Running locally

[docker-compose, daemon-dev.toml, dashboard URL.]

## Production deployment

[Topology, scaling considerations, observability stack, Postgres schema migrations.]
```

This is a placeholder structure — actual content can be drafted from the old top-level README's platform sections + the existing `docs/daemon.md` content. Cross-link to existing docs (`memory.md`, `sensors.md`, etc.).

- [ ] **Step 13.5: Replace `crates/heartbit-core/README.md` (was placeholder from task 1).**

Same content shape as the new top-level README's framework sections, but with crates.io-rendering in mind (no relative repo links to platform docs since docs.rs renders this in isolation):

```markdown
# heartbit-core

The Rust agentic framework — agents, tools, LLM providers, memory, evaluation.

```bash
cargo add heartbit-core
```

```rust
use std::sync::Arc;
use heartbit_core::{AgentRunner, AnthropicProvider, BoxedProvider, RetryingProvider};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let provider = Arc::new(BoxedProvider::new(
        RetryingProvider::with_defaults(
            AnthropicProvider::new(
                std::env::var("ANTHROPIC_API_KEY").unwrap(),
                "claude-sonnet-4-20250514",
            ),
        ),
    ));

    let mut agent = AgentRunner::builder(provider)
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let output = agent.execute("What is Rust?").await?;
    println!("{}", output.content);
    Ok(())
}
```

## What's in the box

[Same feature list as the top-level README's "Features" section.]

## Optional integrations

For Postgres-backed memory, Telegram/Discord/Slack chat adapters, fastembed
local embeddings, sandboxed workspaces, JWT auth, vault, multi-tenant daemon
mode, etc., add the [`heartbit`](https://crates.io/crates/heartbit) umbrella crate:

```bash
cargo add heartbit --features postgres,telegram
```

## License

MIT OR Apache-2.0.

## Source

[https://github.com/heartbit-ai/heartbit](https://github.com/heartbit-ai/heartbit)
```

- [ ] **Step 13.6: Add `## Unreleased` to `CHANGELOG.md`.**

In `CHANGELOG.md`, find the existing `## [Unreleased]` section (added in B2) and append:

```markdown
### Refactor

- Workspace restructured: `heartbit-core` extracted as the official Rust agentic framework. The `heartbit` crate becomes a thin umbrella that re-exports `heartbit-core` and adds platform integrations (Postgres, Telegram/Discord/Slack adapters, Restate workflows, fastembed local embeddings, vault, JWT validator, daemon mode). **No public API changes** — every existing import (`use heartbit::AgentRunner;` etc.) continues to compile via the umbrella's `pub use heartbit_core::*;`. Library users should target `heartbit-core` directly; runtime/platform users keep using `heartbit`.
```

- [ ] **Step 13.7: Verify nothing breaks.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace --no-run 2>&1 | tail -3
```

(Documentation changes shouldn't affect the gate, but verify defensively.)

- [ ] **Step 13.8: Commit.**

```bash
git add README.md CHANGELOG.md docs/platform.md crates/heartbit-cli/README.md crates/heartbit-core/README.md
git commit -m "docs: reposition top-level README around heartbit-core framework

Top-level README now leads with the framework: 'cargo add heartbit-core',
quickstart, feature table. Platform content (daemon mode, Kafka,
multi-tenant deployment, dashboard) moves to crates/heartbit-cli/README.md
and docs/platform.md. crates/heartbit-core/README.md is the docs.rs
landing page. CHANGELOG entry recorded.

No code changes; entirely doc work."
```

---

## Self-Review

Run after Task 13 completes.

- [ ] **Step S.1: Spec coverage.**

Verify each spec section maps to a task:

- Architecture & dependency graph → tasks 1, 2–9, 10
- Module placement → tasks 2–9 (one task per module group), 11
- Cargo.toml & feature graph → tasks 1 (scaffold), 6 (a2a forwarding), 11 (verify auth deps)
- Umbrella re-export strategy → tasks 2–9 (per-step forwardings) + 10 (glob switch)
- Migration mechanics & sequencing → enforced via the two-commit-per-move discipline in tasks 2–9c
- README & positioning → task 13
- Test migration → tasks 12 (sweep) + relies on inline tests moving with `git mv` automatically
- Risks → mitigations baked into each task's gate verification
- Exit criteria → task 12

If a section has no corresponding task, add the task before exiting self-review.

- [ ] **Step S.2: Run the full exit-criteria battery.**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --no-run
cargo test --workspace --lib
cargo build -p heartbit-core --no-default-features
cargo doc --workspace --no-deps
```

All six must exit 0. If any fails, fix and re-commit before declaring done.

- [ ] **Step S.3: Verify CHANGELOG entry exists.**

```bash
grep -A5 "^### Refactor" CHANGELOG.md | head -10
```

Expected: a Refactor subsection mentioning `heartbit-core` extracted, no public API changes.

- [ ] **Step S.4: Verify the new crate is on crates.io-publishable shape (without publishing).**

```bash
cargo publish --dry-run -p heartbit-core 2>&1 | tail -10
```

This is a dry-run validation: catches missing fields in Cargo.toml, missing license, README path issues, etc. Does NOT publish. Any errors must be fixed in `crates/heartbit-core/Cargo.toml` before declaring B3 done. (Actual publication is a separate follow-up round per the spec.)

- [ ] **Step S.5: heartbit-cloud canary (manual / out-of-band).**

If you have access to the heartbit-cloud repository, point its `heartbit` dependency at this branch (path or git ref) and run heartbit-cloud's CI. The umbrella's public API is unchanged so it should build without modification. If anything regresses, that's a B3 bug to fix.

This step is informational — heartbit-cloud lives in a separate repo and isn't blocking on B3 merging to main; this is just a courtesy verification before claiming B3 is done.

---

## Out of Scope (per spec)

These are NOT part of this plan:

- `cargo publish` to crates.io — separate small follow-up round.
- Splitting any further satellites (`heartbit-daemon`, `heartbit-sensor`, `heartbit-channel`, `heartbit-postgres`, `heartbit-embedding`).
- Dropping any feature flag.
- Refactoring inside core during the move (no API renames; `agent/orchestrator.rs` stays at 6.8k LOC).
- `heartbit-cloud` migration to depend on `heartbit-core` directly.
- Toolchain pinning policy.

If any of these are tempting during execution: stop, note as a follow-up, proceed with the plan as written.
