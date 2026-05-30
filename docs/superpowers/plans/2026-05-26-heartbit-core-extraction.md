# heartbit-core domain extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `heartbit-core` releasable as a credible SOTA Rust agentic framework crate by extracting the heartbit-ghost domain leaks accumulated since v2026.507.2.

**Architecture:** Three mechanisms (per spec `docs/superpowers/specs/2026-05-26-heartbit-core-extraction-design.md`): (a) DELETE the duplicated `TwitterPostTool` from core; (b) MOVE `OpenverseImageSearchTool` to `heartbit-ghost`; (c) FEATURE-GATE the persona configs + `ImageSource` enum behind a default-on `ghost-domain-config` feature so SOTA users get a clean surface via `default-features = false`. Plus polish: drop three `#![allow(missing_docs)]` inner attrs, uniform builtin wiring, persona-trait doc cleanup.

**Tech Stack:** Rust workspace (heartbit-core, heartbit-ghost, heartbit, heartbit-cli). Cargo features for gating. Direct commits to `main` per CLAUDE.md.

---

## File Structure (changes)

| Path | Action | Responsibility after |
|---|---|---|
| `crates/heartbit-core/Cargo.toml` | Modify | Add `[features] default = ["ghost-domain-config"]` + `ghost-domain-config = []` |
| `crates/heartbit-core/src/config/daemon.rs` | Modify | `#[cfg(feature = "ghost-domain-config")]` on 5 persona configs + `ImageSource` + their `DaemonConfig` fields |
| `crates/heartbit-core/src/config/mod.rs` | Modify | `#[cfg]` on persona-config re-exports |
| `crates/heartbit-core/src/lib.rs` | Modify | Remove `pub use ... TwitterCredentials`; `#[cfg]` any ghost-only re-exports if at root |
| `crates/heartbit-core/src/tool/builtins/twitter_post.rs` | **Delete** | — |
| `crates/heartbit-core/src/tool/builtins/openverse_image.rs` | **Delete** (moved) | — |
| `crates/heartbit-core/src/tool/builtins/mod.rs` | Modify | Remove `twitter_post` + `openverse_image` `pub mod`/`pub use` + builtin_tools wiring; drop the file-level `#![allow(missing_docs)]` |
| `crates/heartbit-ghost/src/tools/openverse_image.rs` | **Create** | New home for OpenverseImageSearchTool |
| `crates/heartbit-ghost/src/tools/mod.rs` (or `lib.rs`) | Modify | Declare + re-export `openverse_image` |
| `crates/heartbit-ghost/src/review/mod.rs` | Modify | Import path: `heartbit_core::tool::builtins::OpenverseImageSearchTool` → `crate::tools::OpenverseImageSearchTool` |
| `crates/heartbit-core/src/persona/types.rs` | Modify | Remove "evangelism framing" doc; delete or implement empty `TriggerSpec`/`ReviewSpec` enums |

---

## Task 1: Add the `ghost-domain-config` Cargo feature

**Files:**
- Modify: `crates/heartbit-core/Cargo.toml`

- [ ] **Step 1: Read the existing `[features]` table**

```bash
grep -A 5 "^\[features\]" crates/heartbit-core/Cargo.toml
```

If no `[features]` table exists, you'll add one. If it does, the new key joins it.

- [ ] **Step 2: Add the feature**

Add (or modify) the `[features]` section in `crates/heartbit-core/Cargo.toml`:

```toml
[features]
default = ["ghost-domain-config"]
ghost-domain-config = []
```

If a `[features]` block already exists, MERGE these entries — don't duplicate. If `default = [...]` already exists with other features, append `"ghost-domain-config"` to it: `default = ["existing", "ghost-domain-config"]`.

- [ ] **Step 3: Verify nothing breaks**

```bash
cargo check --workspace --all-targets --features daemon
```

Expected: clean. At this point the feature is declared but no `#[cfg]` references it yet, so behavior is unchanged.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-core/Cargo.toml
git commit -m "feat(core): ghost-domain-config feature flag (default on)"
```

---

## Task 2: Cfg-gate the persona config types + `ImageSource`

**Files:**
- Modify: `crates/heartbit-core/src/config/daemon.rs`

Add `#[cfg(feature = "ghost-domain-config")]` to the definitions of each persona-specific config type + the `ImageSource` enum + each `default_*` helper fn that exists only to support those types.

- [ ] **Step 1: Find the exact line numbers**

```bash
grep -n "^pub struct PersonaMentionsConfig\|^pub struct PersonaPostsConfig\|^pub struct PersonaQuotesConfig\|^pub struct PersonaBlogConfig\|^pub struct XAnnounceConfig\|^pub struct GithubReadmeConfig\|^pub enum ImageSource" crates/heartbit-core/src/config/daemon.rs
```

Note the line of each.

- [ ] **Step 2: Add `#[cfg]` immediately above each struct/enum + each `impl` block + each `fn default_*` that only those types reference**

For each `pub struct PersonaXxxConfig` and `pub enum ImageSource`:

```rust
#[cfg(feature = "ghost-domain-config")]
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaPostsConfig {
    // ... unchanged ...
}
```

Apply the same `#[cfg]` to:
- Every `impl PersonaXxxConfig { ... }` block (search with `grep -n "^impl PersonaPostsConfig\|^impl PersonaBlogConfig\|^impl PersonaQuotesConfig\|^impl PersonaMentionsConfig\|^impl XAnnounceConfig\|^impl GithubReadmeConfig\|^impl ImageSource"`)
- Every `Default` impl on these types (if any)
- Every `fn default_*` helper used only by these types — search the file for the helper, see what types call it; if only the gated types use it, gate the helper too (e.g. `fn default_blog_poll_interval_seconds`, `fn default_bio_template_path`, `fn default_image_source` if present, etc.). If a helper is shared with non-gated types, leave it ungated.
- Every `#[cfg(test)] mod tests` test fn that constructs the gated types (or gate the entire test module if all tests reference gated types — use `#[cfg(all(test, feature = "ghost-domain-config"))]`).

**Critical pattern:** trait implementations cfg-gated MUST have the trait's referenced types also reachable under that cfg. If `impl Default for PersonaPostsConfig` uses `super::default_true`, that helper is generic and stays ungated. Read the impl bodies before gating.

- [ ] **Step 3: Build with default features (workspace stays unchanged)**

```bash
cargo check --workspace --all-targets --features daemon
```

Expected: clean. (Feature is on by default, so everything still compiles.)

- [ ] **Step 4: Smoke build with feature OFF**

```bash
cargo check --package heartbit-core --no-default-features
```

Expected: clean. If there are dangling references to gated types (e.g. a `pub use` re-export still pointing at them), the build will fail with "cannot find type X" — you'll fix them in Tasks 3 and 4. For NOW, if the type definitions are gated but their consumers in the SAME file (e.g. `impl` blocks, helpers) reference them, those must be co-gated.

The most common breakage: a `default_*` fn that returns `ImageSource::Online` — that fn must also be gated, OR change its return type if it's used elsewhere.

If the smoke build still fails after gating all definitions + impls + tests, identify the remaining dangling reference and gate it. Don't proceed to the next task until this smoke build is clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-core/src/config/daemon.rs
git commit -m "refactor(core): gate persona config types behind ghost-domain-config"
```

---

## Task 3: Cfg-gate the `DaemonConfig` fields that hold persona configs

**Files:**
- Modify: `crates/heartbit-core/src/config/daemon.rs`

`DaemonConfig` itself stays in core unconditionally. But its 4 fields that hold the gated types must themselves be gated.

- [ ] **Step 1: Locate the fields**

```bash
grep -n "persona_mentions:\|persona_posts:\|persona_quotes:\|persona_blog:" crates/heartbit-core/src/config/daemon.rs
```

You'll see lines like:
```rust
    #[serde(default)]
    pub persona_mentions: Vec<PersonaMentionsConfig>,
```

- [ ] **Step 2: Add `#[cfg]` above each of the 4 fields**

For each:

```rust
    #[cfg(feature = "ghost-domain-config")]
    #[serde(default)]
    pub persona_mentions: Vec<PersonaMentionsConfig>,
```

The `#[cfg]` goes BEFORE the `#[serde]` attribute.

- [ ] **Step 3: Build with default features**

```bash
cargo check --workspace --all-targets --features daemon
```

Expected: clean. Workspace consumers (heartbit-cli, heartbit umbrella) still see the field because default features are on.

- [ ] **Step 4: Smoke build with feature OFF**

```bash
cargo check --package heartbit-core --no-default-features
```

Expected: clean. `DaemonConfig` compiles without the gated fields.

If `DaemonConfig`'s tests (round-trip TOML parse, etc.) construct it with the gated fields, gate those tests too — or use `..Default::default()` patterns if a `Default` impl exists.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-core/src/config/daemon.rs
git commit -m "refactor(core): gate DaemonConfig persona fields behind ghost-domain-config"
```

---

## Task 4: Cfg-gate the re-exports

**Files:**
- Modify: `crates/heartbit-core/src/config/mod.rs`
- Modify: `crates/heartbit-core/src/lib.rs` (only if persona configs are re-exported at the crate root)

- [ ] **Step 1: Find every `pub use` of the gated types**

```bash
grep -n "pub use.*PersonaMentionsConfig\|pub use.*PersonaPostsConfig\|pub use.*PersonaQuotesConfig\|pub use.*PersonaBlogConfig\|pub use.*XAnnounceConfig\|pub use.*GithubReadmeConfig\|pub use.*ImageSource" crates/heartbit-core/src/config/mod.rs crates/heartbit-core/src/lib.rs
```

You'll likely see a multi-name `pub use daemon::{...};` statement.

- [ ] **Step 2: Split or gate the `pub use`**

If the `pub use` is multi-name (`pub use daemon::{KafkaConfig, PersonaPostsConfig, ImageSource, ...};`), split it into two statements: one with the always-on names, one with `#[cfg(feature = "ghost-domain-config")]` + the gated names.

Example:

```rust
// Before:
pub use daemon::{
    DaemonConfig, KafkaConfig, MetricsConfig, AuthConfig,
    PersonaMentionsConfig, PersonaPostsConfig, PersonaQuotesConfig,
    PersonaBlogConfig, XAnnounceConfig, GithubReadmeConfig, ImageSource,
};

// After:
pub use daemon::{DaemonConfig, KafkaConfig, MetricsConfig, AuthConfig};

#[cfg(feature = "ghost-domain-config")]
pub use daemon::{
    PersonaMentionsConfig, PersonaPostsConfig, PersonaQuotesConfig,
    PersonaBlogConfig, XAnnounceConfig, GithubReadmeConfig, ImageSource,
};
```

If `lib.rs` re-exports any of these at the crate root, apply the same split/gate there.

- [ ] **Step 3: Build with both feature configurations**

```bash
cargo check --workspace --all-targets --features daemon
cargo check --package heartbit-core --no-default-features
```

Both should be clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-core/src/config/mod.rs crates/heartbit-core/src/lib.rs
git commit -m "refactor(core): gate persona config re-exports behind ghost-domain-config"
```

---

## Task 5: Delete the duplicated `TwitterPostTool` from core

**Files:**
- Delete: `crates/heartbit-core/src/tool/builtins/twitter_post.rs`
- Modify: `crates/heartbit-core/src/tool/builtins/mod.rs`
- Modify: `crates/heartbit-core/src/lib.rs` (if `TwitterCredentials` is re-exported at root)

`heartbit-ghost/src/tools/client.rs:343-438` has its own OAuth 1.0a HMAC-SHA1 signing — the core version is dead duplication.

- [ ] **Step 1: Verify the ghost replacement is real**

```bash
grep -n "port from.*heartbit-core::tool::builtins::twitter_post\|fn sign_request\|HmacSha1::" crates/heartbit-ghost/src/tools/client.rs | head -3
```

Expected to find the "port from" comment and the HMAC implementation. If not, STOP and report BLOCKED.

- [ ] **Step 2: Confirm nothing outside ghost uses `TwitterPostTool` or `TwitterCredentials`**

```bash
grep -rn "TwitterPostTool\|TwitterCredentials" crates/ --include=*.rs | grep -v "heartbit-core\|heartbit-ghost/src/tools/client.rs"
```

Expected: empty. If non-empty, those callers must migrate to ghost's equivalents before the delete. If `heartbit-cli` uses `TwitterCredentials`, that's a real consumer — STOP and report BLOCKED (the spec said "ghost has its own" but didn't account for CLI).

- [ ] **Step 3: Delete the file**

```bash
rm crates/heartbit-core/src/tool/builtins/twitter_post.rs
```

- [ ] **Step 4: Remove from `mod.rs`**

In `crates/heartbit-core/src/tool/builtins/mod.rs`:
- Delete the line `pub(crate) mod twitter_post;` (or `pub mod twitter_post;`)
- Delete the line `pub use twitter_post::TwitterCredentials;` (or whatever the re-export is — grep for `twitter_post::`)
- Find the `builtin_tools()` function and delete the conditional branch that registered `TwitterPostTool` (look for `TwitterPostTool::new()` or `TwitterCredentials`)

- [ ] **Step 5: Remove from `lib.rs` if re-exported at root**

```bash
grep -n "TwitterCredentials\|TwitterPostTool" crates/heartbit-core/src/lib.rs
```

If anything matches, delete those re-exports.

- [ ] **Step 6: Build + test gate**

```bash
cargo check --workspace --all-targets --features daemon
cargo test --workspace --lib --features daemon 2>&1 | grep "^test result" | tail -6
```

Expected: clean compile, all tests pass (minus the ~10 twitter_post tests that lived in the deleted file — that's expected). Numbers go down by ~10 tests but no failures.

- [ ] **Step 7: Commit**

```bash
git add -A crates/heartbit-core/src/tool/builtins/ crates/heartbit-core/src/lib.rs
git commit -m "refactor(core): delete duplicated TwitterPostTool (ghost owns OAuth signing)"
```

---

## Task 6: Move `OpenverseImageSearchTool` to heartbit-ghost

**Files:**
- Delete: `crates/heartbit-core/src/tool/builtins/openverse_image.rs`
- Create: `crates/heartbit-ghost/src/tools/openverse_image.rs`
- Modify: `crates/heartbit-core/src/tool/builtins/mod.rs`
- Modify: `crates/heartbit-ghost/src/tools/mod.rs` (or wherever `tools` is declared in ghost)
- Modify: `crates/heartbit-ghost/src/review/mod.rs` (import path update)

- [ ] **Step 1: Locate the ghost tools module**

```bash
ls crates/heartbit-ghost/src/tools/ 2>&1
grep -n "pub mod tools\|^pub mod " crates/heartbit-ghost/src/lib.rs | head -10
```

The `tools` module exists (`client.rs` lives there). Confirm it has a `mod.rs` or `tools.rs` file.

- [ ] **Step 2: Verify imports the openverse tool needs are available in ghost**

```bash
grep -n "^use\|^pub" crates/heartbit-core/src/tool/builtins/openverse_image.rs | head -20
```

The file uses `crate::http::vendor_client_builder`, `crate::tool::{Tool, ToolOutput, ToolContext, ToolDefinition}`, `crate::Error`, `base64`. These are all in `heartbit-core`'s public API. Ghost depends on heartbit-core, so they resolve via `heartbit_core::*` instead of `crate::*`.

- [ ] **Step 3: Read the source + rewrite imports**

Read the original file:

```bash
wc -l crates/heartbit-core/src/tool/builtins/openverse_image.rs
```

Copy the file to `crates/heartbit-ghost/src/tools/openverse_image.rs` and update the imports:

```rust
// Original (in core):
use crate::http::vendor_client_builder;
use crate::tool::{Tool, ToolContext, ToolDefinition, ToolOutput};
use crate::Error;

// In ghost:
use heartbit_core::http::vendor_client_builder;
use heartbit_core::tool::{Tool, ToolContext, ToolDefinition, ToolOutput};
use heartbit_core::Error;
```

Adapt every `crate::` reference that points to a `heartbit-core` path. Leave intra-module references (`super::`, etc.) intact if they exist within the file.

- [ ] **Step 4: Verify the public API is exposed**

Check what's `pub use`'d from the original mod.rs:

```bash
grep "openverse" crates/heartbit-core/src/tool/builtins/mod.rs
```

Likely `pub use openverse_image::OpenverseImageSearchTool;`. Replicate in ghost's tools module (`crates/heartbit-ghost/src/tools/mod.rs` or wherever):

```rust
pub mod openverse_image;
pub use openverse_image::OpenverseImageSearchTool;
```

If the ghost tools module uses a different export pattern, follow the existing convention.

- [ ] **Step 5: Delete from core**

```bash
rm crates/heartbit-core/src/tool/builtins/openverse_image.rs
```

In `crates/heartbit-core/src/tool/builtins/mod.rs`, delete the `pub mod openverse_image;` and `pub use openverse_image::OpenverseImageSearchTool;` lines.

- [ ] **Step 6: Update the import in `review/mod.rs`**

```bash
grep -n "OpenverseImageSearchTool" crates/heartbit-ghost/src/review/mod.rs
```

Change `heartbit_core::tool::builtins::OpenverseImageSearchTool` (or wherever it's imported from) to `crate::tools::OpenverseImageSearchTool` (or the new ghost path).

- [ ] **Step 7: Build + test gate**

```bash
cargo check --workspace --all-targets --features daemon
cargo test --package heartbit-ghost --lib openverse 2>&1 | tail -10
cargo test --workspace --lib --features daemon 2>&1 | grep "^test result" | tail -6
```

Expected: clean. The 8 openverse tests now run as part of `heartbit-ghost` instead of `heartbit-core`. Workspace test count unchanged.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: move OpenverseImageSearchTool from core to heartbit-ghost"
```

---

## Task 7: Restore the `missing_docs` gate

**Files:**
- Modify: `crates/heartbit-core/src/config/daemon.rs` (remove `#![allow(missing_docs)]` at line 1)
- Modify: `crates/heartbit-core/src/tool/builtins/mod.rs` (remove `#![allow(missing_docs)]` if present at module top)
- Fix every resulting `missing_docs` compile error

The `twitter_post.rs` inner attr is gone by deletion (Task 5). The remaining two undermine the crate's stated docs gate.

- [ ] **Step 1: Confirm the inner attrs are present**

```bash
head -3 crates/heartbit-core/src/config/daemon.rs crates/heartbit-core/src/tool/builtins/mod.rs
```

Expected: `#![allow(missing_docs)]` near the top of each.

- [ ] **Step 2: Remove both inner attrs**

Delete the `#![allow(missing_docs)]` line from:
- `crates/heartbit-core/src/config/daemon.rs:1`
- `crates/heartbit-core/src/tool/builtins/mod.rs` (whichever line — likely 1, 2, or 3)

- [ ] **Step 3: Find the missing docs**

```bash
cargo build --package heartbit-core --features daemon 2>&1 | grep "missing_docs\|missing documentation" | head -30
```

You'll get a list of `pub` items without rustdoc.

- [ ] **Step 4: Add rustdoc**

For each missing-doc error, add a `/// ...` line above the offending item. Keep docs short and accurate. Examples:
- For a struct field: one-liner describing what the field controls
- For an enum variant: one-liner describing the case
- For a struct: one-liner describing its purpose

If a public item is genuinely undocumentable (e.g. a `Default::default()` impl), use `#[allow(missing_docs)]` on that specific item only — never inner-attr at module level.

Iterate `cargo build` + add docs until clean.

- [ ] **Step 5: Verify both feature configurations**

```bash
cargo build --workspace --all-targets --features daemon
cargo build --package heartbit-core --no-default-features
cargo clippy --workspace --all-targets --features daemon -- -D warnings
```

All clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-core/src/
git commit -m "refactor(core): restore deny(missing_docs) — drop module-level allows + fill rustdoc"
```

---

## Task 8: Persona trait cleanup

**Files:**
- Modify: `crates/heartbit-core/src/persona/types.rs`

- [ ] **Step 1: Remove "evangelism framing" from `PersonaExpansion::mode_addendum` doc**

Find lines 36-39 (approximately) of `crates/heartbit-core/src/persona/types.rs`:

```bash
grep -n -B2 -A4 "mode_addendum\|evangelism" crates/heartbit-core/src/persona/types.rs | head -20
```

Replace the doc comment that mentions "evangelism framing" with a domain-neutral one. Suggested:

```rust
    /// Optional addendum appended to the persona's system prompt at
    /// expansion time. Implementations use this to scope a single
    /// persona to multiple sub-modes (e.g. an X persona that posts
    /// generally vs. one that focuses on a specific topic cluster).
    /// `None` for personas without per-mode variation.
    pub mode_addendum: Option<&'static str>,
```

(Adjust to match the actual field type — `Option<&'static str>` or `Option<String>` per current source.)

- [ ] **Step 2: Decide the fate of empty `TriggerSpec`/`ReviewSpec` enums**

```bash
grep -n "pub enum TriggerSpec\|pub enum ReviewSpec\|#\[non_exhaustive\]" crates/heartbit-core/src/persona/types.rs
```

If both are `#[non_exhaustive] pub enum X {}` with no variants:

**Option A (recommended): Delete them.** Search for references; if zero, delete.

```bash
grep -rn "TriggerSpec\|ReviewSpec" crates/ --include=*.rs
```

If only their own definition matches: delete both `pub enum` blocks.

**Option B:** If they're referenced anywhere (e.g. in a trait method signature), don't delete — file a `tasks/lessons.md` note explaining they're scaffolding for future trigger/review variants and move on. Add a `/// Placeholder for future persona trigger types. Currently no variants — implementors should not match on this.` doc to keep the missing_docs gate happy.

Pick Option A if grep shows zero external references. Otherwise Option B.

- [ ] **Step 3: Build + test**

```bash
cargo build --workspace --all-targets --features daemon
cargo test --package heartbit-core --lib persona 2>&1 | tail -10
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-core/src/persona/types.rs
git commit -m "refactor(core): persona trait — drop evangelism-framing doc + empty trigger/review enums"
```

---

## Task 9: Final workspace gate + commit chain verification

- [ ] **Step 1: Full workspace gate**

```bash
cd ~/projects/heartbit
cargo fmt -- --check && \
cargo clippy --workspace --all-targets --features daemon -- -D warnings && \
cargo test --workspace --lib --features daemon 2>&1 | grep "^test result" | tail -6
```

Expected: all green. Test count should be roughly: previous total minus ~10 (deleted twitter_post tests) ≈ 4282.

- [ ] **Step 2: Smoke build with feature OFF (the whole point of this refactor)**

```bash
cargo build --package heartbit-core --no-default-features
cargo test --package heartbit-core --lib --no-default-features 2>&1 | tail -10
```

Expected: clean compile AND lib tests pass (most tests don't need the feature, but if any rely on it they should be `#[cfg(feature = "ghost-domain-config")]`-gated).

- [ ] **Step 3: Verify the public API surface with feature OFF**

```bash
cargo doc --package heartbit-core --no-default-features --no-deps 2>&1 | tail -5
```

Optional: open `target/doc/heartbit_core/index.html` in a browser to visually confirm no PersonaXxxConfig, ImageSource, TwitterCredentials, or OpenverseImageSearchTool appear in the public API.

- [ ] **Step 4: Verify validate-config still works**

```bash
HEARTBIT_GHOST_OPERATOR_USER_ID=999 target/release/heartbit --config daemon-dev.toml daemon --validate-config 2>&1 | tail -3
```

Expected: `✓ daemon-dev.toml validates clean`. This proves the operator TOML format is unchanged (no breaking config change).

If the release binary is stale, rebuild first: `cargo build --release --bin heartbit --features daemon`.

- [ ] **Step 5: Show the commit chain**

```bash
git log --oneline 8040b59..HEAD
```

Expected: ~8 commits, one per task above.

- [ ] **Step 6: Push to origin**

```bash
git push origin main
```

The CLAUDE.md push restriction is satisfied — the user explicitly authorized this extraction via "proceed with the implementation plan."

---

## Verification matrix

| Spec item | Covered by |
|---|---|
| Feature flag declared (`ghost-domain-config`, default on) | Task 1 |
| Persona config types cfg-gated | Task 2 |
| `DaemonConfig` fields cfg-gated | Task 3 |
| Re-exports cfg-gated | Task 4 |
| `TwitterPostTool` deleted | Task 5 |
| `OpenverseImageSearchTool` moved to ghost | Task 6 |
| `#![allow(missing_docs)]` x3 removed + rustdoc filled | Task 7 |
| `PersonaExpansion::mode_addendum` doc neutralized | Task 8 |
| Empty `TriggerSpec`/`ReviewSpec` enums resolved | Task 8 |
| Uniform builtin wiring (out of scope per spec) | — (deferred) |
| Workspace gate green | Task 9 |
| `--no-default-features` builds clean | Task 9 |
| Operator TOML still parses | Task 9 |
| Pushed to origin | Task 9 |

## Notes for the implementer

- **Direct commits to `main`** per project workflow (CLAUDE.md). No worktree.
- **No new crates.** The spec considered then rejected a `heartbit-personas-config` crate. Stay inside the existing 4 crates.
- **The point of this refactor is the `--no-default-features` smoke build passing.** That's the gate that proves SOTA users get a clean surface. If at any task the smoke build breaks and you can't fix it without changing the spec's architecture, STOP and report — don't paper over by gating more aggressively or weakening the default surface.
- **Tests for gated types** must themselves be gated (`#[cfg(all(test, feature = "ghost-domain-config"))]` on the `mod tests` block, or per-test if mixed). The full-feature workspace test count drops by ~10 (the deleted twitter_post tests); openverse tests move to ghost.
- **Bug-fixing inside the moved openverse code** is out of scope. Mechanical translation only. If the original had a bug, it still has it after the move; file a separate task.
- **No emoji in commit messages** unless the user explicitly requests them (CLAUDE.md).
