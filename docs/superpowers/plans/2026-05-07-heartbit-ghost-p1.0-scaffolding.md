# heartbit-ghost P1.0 — Crate Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the `heartbit-ghost` crate as a workspace member that registers a stub persona (`heartbit-ghost:x`) into `heartbit_core::PersonaRegistry` at CLI startup, so `heartbit persona list` shows the new persona and `heartbit persona show heartbit-ghost:x` no longer reports "persona not found".

**Architecture:** New downstream crate `heartbit-ghost` exports a `register(&mut PersonaRegistry)` function and a `XGhostPersona` stub implementing `heartbit_core::Persona`. `heartbit-cli` adds `heartbit-ghost` as a dependency and calls the registration function in `persona::run()` before dispatch. Stub `expand()` returns `PersonaExpansion::default()` — no agents, no tools, no triggers. All P1.1+ scope (X tool family, voice modeling, A/B feedback loop, autonomy phases) is **out of scope** for this plan.

**Tech Stack:** Rust 2024, Tokio, `heartbit-core` (`Persona`, `PersonaRegistry`, `PersonaParams`, `PersonaExpansion`), `heartbit-cli` for the user-facing surface.

**Spec:** `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md` §9 (P1.0 sub-phase, 5 bullets)

**Branch:** `feat/heartbit-ghost`

---

## File Structure

### New files
- `crates/heartbit-ghost/Cargo.toml` — workspace-member crate manifest
- `crates/heartbit-ghost/src/lib.rs` — `XGhostPersona` stub + `register()` function + unit tests

### Modified files
- `Cargo.toml` (root) — add `crates/heartbit-ghost` to `[workspace] members`
- `crates/heartbit-cli/Cargo.toml` — add `heartbit-ghost` dependency
- `crates/heartbit-cli/src/persona.rs` — `run()` calls `heartbit_ghost::register(&mut registry)` before dispatch; update one error string that referenced "Phase 0" so it stays accurate post-P1.0

### Out of scope
- All P1.1 X tool family work (`twitter_thread`, `twitter_search`, `twitter_mentions`, `twitter_reply`, `twitter_user`, `twitter_dm`, `twitter_schedule`, `twitter_metrics`, media support on `twitter_post`)
- All P1.2 voice modeling (corpus, profile schema, blend algorithm, runtime conditioning)
- All P1.3 generation pipeline (researcher / writer / style_critic / revise loop / fact_check / publish_gate / publisher) and Telegram review wiring
- All P1.4 autonomy phases, audit log, kill-switch, anti-coordination guard, content guardrails, dataset export
- The 4 documented `TODO(phase-1):` sites carrying empty `ExecutionContext`/`ToolCallRequest` identity — addressed in later sub-phases when their consumers land

---

## Task 1: Create the `heartbit-ghost` crate (scaffold + register stub)

**Files:**
- Create: `crates/heartbit-ghost/Cargo.toml`
- Create: `crates/heartbit-ghost/src/lib.rs`
- Modify: `Cargo.toml` (root)

- [ ] **Step 1: Create the crate manifest**

Create `crates/heartbit-ghost/Cargo.toml` with:

```toml
[package]
name = "heartbit-ghost"
version.workspace = true
edition = "2024"
authors.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
description = "Best-in-class autonomous X (Twitter) agent persona for the Heartbit runtime."
keywords = ["agent", "llm", "twitter", "x", "persona"]

[dependencies]
heartbit-core = { path = "../heartbit-core" }

[dev-dependencies]
tokio = { workspace = true }
```

(No tokio in main `[dependencies]` — the stub doesn't need an async runtime; future sub-phases will add it.)

- [ ] **Step 2: Create the lib.rs with `XGhostPersona` stub + `register()` + tests**

Create `crates/heartbit-ghost/src/lib.rs`:

```rust
//! `heartbit-ghost` — best-in-class autonomous X (Twitter) agent persona.
//!
//! P1.0 (this release) ships a scaffolding stub: the persona registers itself
//! into `heartbit_core::PersonaRegistry` so the CLI surface lights up, but
//! `expand()` returns an empty `PersonaExpansion` (no agents, no tools, no
//! triggers, no review channel). Real bodies land in P1.1 (X tool family),
//! P1.2 (voice modeling), P1.3 (generation pipeline + Telegram review), and
//! P1.4 (autonomy phases + audit + dataset export).

#![deny(missing_docs)]

use std::sync::Arc;

use heartbit_core::{Persona, PersonaExpansion, PersonaParams, PersonaRegistry};

/// Stable persona identifier — used as the registry key and as the
/// `recipe = "..."` value in `[[persona]]` config blocks.
pub const PERSONA_NAME: &str = "heartbit-ghost:x";

/// Scaffolding stub for the X (Twitter) ghost persona.
///
/// In P1.0 this expands to an empty `PersonaExpansion`. Real expansion
/// (sub-agents, tools, triggers, review spec) lands in P1.1+.
pub struct XGhostPersona {
    /// Persona version string, derived at compile time from the workspace
    /// `Cargo.toml`.
    version: &'static str,
}

impl XGhostPersona {
    /// Create a new stub instance.
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

impl Default for XGhostPersona {
    fn default() -> Self {
        Self::new()
    }
}

impl Persona for XGhostPersona {
    fn name(&self) -> &str {
        PERSONA_NAME
    }

    fn description(&self) -> &str {
        "Best-in-class autonomous X (Twitter) agent. Scaffolding stub — Phase 1 P1.0."
    }

    fn version(&self) -> &str {
        self.version
    }

    fn expand(
        &self,
        _params: &PersonaParams,
    ) -> Result<PersonaExpansion, heartbit_core::Error> {
        // P1.0 stub: empty expansion. P1.1+ fills this with the real persona
        // (sub-agent recipes, X tool family, triggers, Telegram review).
        Ok(PersonaExpansion::default())
    }
}

/// Register the X ghost persona into the supplied registry.
///
/// Callers (e.g. `heartbit-cli` at startup, the daemon at boot) build a
/// `PersonaRegistry`, call this function (and any other persona crates'
/// equivalent functions), then pass the populated registry to the CLI
/// dispatch / daemon dispatch / etc.
pub fn register(registry: &mut PersonaRegistry) {
    registry.register(Arc::new(XGhostPersona::new()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_name_is_stable() {
        let p = XGhostPersona::new();
        assert_eq!(p.name(), "heartbit-ghost:x");
        assert_eq!(p.name(), PERSONA_NAME);
    }

    #[test]
    fn stub_description_is_non_empty_and_marks_p1_0() {
        let p = XGhostPersona::new();
        let desc = p.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("P1.0") || desc.contains("Scaffolding") || desc.contains("stub"));
    }

    #[test]
    fn stub_version_matches_cargo_pkg_version() {
        let p = XGhostPersona::new();
        assert_eq!(p.version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn stub_expand_returns_empty_expansion() {
        let p = XGhostPersona::new();
        let params = PersonaParams::default();
        let exp = p.expand(&params).expect("expand returns Ok");
        assert!(exp.agents.is_empty());
        assert!(exp.tools.is_empty());
        assert!(exp.triggers.is_empty());
        assert!(exp.review.is_none());
    }

    #[test]
    fn register_adds_persona_to_empty_registry() {
        let mut r = PersonaRegistry::new();
        assert!(r.is_empty());
        register(&mut r);
        assert_eq!(r.len(), 1);
        assert!(r.get(PERSONA_NAME).is_some());
        assert_eq!(r.list(), vec!["heartbit-ghost:x"]);
    }

    #[test]
    fn register_twice_is_idempotent() {
        // PersonaRegistry::register is last-write-wins, so calling register()
        // twice should leave exactly one entry under the same key.
        let mut r = PersonaRegistry::new();
        register(&mut r);
        register(&mut r);
        assert_eq!(r.len(), 1);
        assert!(r.get(PERSONA_NAME).is_some());
    }
}
```

- [ ] **Step 3: Add the new crate to the workspace `members` list**

Edit `Cargo.toml` (root) to add `crates/heartbit-ghost` to the `members` array. Find:

```toml
members = ["crates/heartbit", "crates/heartbit-cli", "crates/heartbit-macro", "crates/heartbit-gateway", "crates/heartbit-core", "crates/heartbit-sensors", "crates/heartbit-telegram"]
```

Replace with:

```toml
members = ["crates/heartbit", "crates/heartbit-cli", "crates/heartbit-macro", "crates/heartbit-gateway", "crates/heartbit-core", "crates/heartbit-sensors", "crates/heartbit-telegram", "crates/heartbit-ghost"]
```

(Append `"crates/heartbit-ghost"` at the end. Match the existing single-line format.)

- [ ] **Step 4: Build the new crate in isolation**

Run from the repo root:

```bash
cargo build -p heartbit-ghost
```

Expected: `Finished dev profile [unoptimized + debuginfo] target(s)` with no errors. The crate compiles standalone since its only dependency is `heartbit-core`.

If you see `error: failed to find package heartbit-ghost`, the workspace `members` edit did not take effect — re-check Step 3.

- [ ] **Step 5: Run the new crate's tests**

```bash
cargo test -p heartbit-ghost
```

Expected: `test result: ok. 6 passed; 0 failed; 0 ignored` for the 6 unit tests defined in Step 2.

- [ ] **Step 6: Workspace quality gate (touched files only)**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost -- -D warnings
```

Both should be clean. (If `cargo fmt --check` fails on the lib.rs, run `cargo fmt -p heartbit-ghost` once to apply formatting and confirm clean.)

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-ghost/ Cargo.toml
git commit -m "$(cat <<'EOF'
feat(ghost): scaffold heartbit-ghost crate with P1.0 stub persona

XGhostPersona implements heartbit_core::Persona with a stub expand()
returning an empty PersonaExpansion. The crate exports a register()
helper for downstream callers (heartbit-cli, daemon) to add the
persona to their PersonaRegistry at startup.

This is P1.0 of the heartbit-ghost roadmap. Real persona contents
(sub-agents, X tool family, voice modeling, A/B feedback, autonomy
phases) land in P1.1-P1.4 as separate plans on this branch.

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md §9 P1.0
EOF
)"
```

---

## Task 2: Wire `heartbit-cli` to register the ghost persona at startup

**Files:**
- Modify: `crates/heartbit-cli/Cargo.toml`
- Modify: `crates/heartbit-cli/src/persona.rs`

- [ ] **Step 1: Add `heartbit-ghost` as a dependency of `heartbit-cli`**

Edit `crates/heartbit-cli/Cargo.toml`. In the `[dependencies]` section, after the existing `heartbit = { path = "../heartbit", features = ["full"] }` line, add:

```toml
heartbit-ghost = { path = "../heartbit-ghost" }
```

The full `[dependencies]` block (just the new line; rest unchanged) becomes:

```toml
heartbit = { path = "../heartbit", features = ["full"] }
heartbit-ghost = { path = "../heartbit-ghost" }
tokio = { workspace = true }
# ... rest unchanged
```

- [ ] **Step 2: Verify the new dep resolves cleanly**

```bash
cargo check -p heartbit-cli
```

Expected: `Finished dev profile`. No errors.

- [ ] **Step 3: Update `persona::run()` to register the ghost persona before dispatch**

In `crates/heartbit-cli/src/persona.rs`, locate the existing `run()` function (around line 121–125):

```rust
/// Dispatch a `persona` subcommand against the (Phase 0: empty) registry.
pub async fn run(cmd: PersonaCommand) -> Result<()> {
    let registry = PersonaRegistry::new();
    dispatch(cmd, &registry).await
}
```

Replace with:

```rust
/// Dispatch a `persona` subcommand against the registry populated by
/// linked persona crates (e.g. `heartbit-ghost`).
pub async fn run(cmd: PersonaCommand) -> Result<()> {
    let mut registry = PersonaRegistry::new();
    heartbit_ghost::register(&mut registry);
    dispatch(cmd, &registry).await
}
```

(Two changes: `let registry` → `let mut registry`, and the new line calling `heartbit_ghost::register`. Plus the docstring update so it doesn't lie about being empty.)

- [ ] **Step 4: Update the now-stale "Phase 0" wording in the dispatch error**

Same file, around line 152–155. Locate:

```rust
            // Bodies for non-empty registry land in Phase 1 alongside concrete persona crates.
            Err(anyhow!(
                "persona subcommand bodies are not implemented in Phase 0; this CLI shell ships with the foundation release."
            ))
```

Replace with:

```rust
            // P1.0 ships the registration shell; subcommand bodies land in
            // later sub-phases (P1.1–P1.4) alongside the persona's tools,
            // voice modeling, and pipeline.
            Err(anyhow!(
                "persona '{name}': subcommand body is not yet implemented (P1.0 scaffolding stub). The persona is registered; its tools, voice modeling, and pipeline land in later sub-phases."
            ))
```

(Two changes: the block comment, and the error string. The new error string interpolates `name` so the user sees which persona they tried to invoke.)

- [ ] **Step 5: Verify existing CLI tests still pass**

The existing `crates/heartbit-cli/src/persona.rs` test mod constructs its own empty `PersonaRegistry` and calls `dispatch()` directly, bypassing `run()`. Those tests should be unaffected by Task 2. Run:

```bash
cargo test -p heartbit-cli persona
```

Expected: `test result: ok. 4 passed; 0 failed` (same 4 tests as Phase 0 — `list_against_empty_registry_prints_message`, `show_against_empty_registry_returns_error`, `corpus_add_against_empty_registry_returns_error`, `profile_rebuild_against_empty_registry_returns_error`).

If any test fails because of the new error wording in Step 4 (e.g. a test asserts on the old "Phase 0" string), update the assertion to match the new wording. Looking at the existing test code, none of the 4 tests assert on the "Phase 0" string — they assert on `"No personas registered"` for empty-registry path, which is unchanged. So no test update should be needed.

- [ ] **Step 6: Workspace quality gate**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features 2>&1 | tail -5
```

All clean; same test count as before P1.0 plus the 6 new heartbit-ghost tests (totals: previous workspace count + 6).

- [ ] **Step 7: Build the release binary for end-to-end smoke**

```bash
cargo build -p heartbit-cli --release
```

Expected: `Finished release profile`.

- [ ] **Step 8: Smoke test — `persona list` shows the ghost persona**

```bash
./target/release/heartbit persona list
```

Expected output (one line, exit 0):

```
heartbit-ghost:x
```

(NOT "No personas registered." — that's the Phase 0 default that should no longer fire.)

- [ ] **Step 9: Smoke test — `persona show` for the registered persona returns the new "not yet implemented" error**

```bash
./target/release/heartbit persona show heartbit-ghost:x
echo "exit=$?"
```

Expected: stderr contains the new wording (`"persona 'heartbit-ghost:x': subcommand body is not yet implemented (P1.0 scaffolding stub)..."`), exit code is non-zero.

The error must NOT mention "Phase 0" anymore. The error must NOT say "not found" — that's the unknown-persona path, which we test next.

- [ ] **Step 10: Smoke test — `persona show` for an unknown persona still says "not found"**

```bash
./target/release/heartbit persona show foo
echo "exit=$?"
```

Expected: stderr contains `"persona 'foo' not found. No personas registered."` — wait, that's wrong because we DO have a persona registered now. The Phase 0 `NO_PERSONAS_REGISTERED` constant is now stale.

Verify the actual output:

```bash
./target/release/heartbit persona show foo 2>&1
```

If the message still says "No personas registered" but the registry now has `heartbit-ghost:x`, that's an inconsistency. Decide:

- (a) **Accept** — the constant says "No personas registered (heartbit-ghost or another persona crate must be linked into this build.)" — that hint is now obsolete since heartbit-ghost IS linked. Acceptable for P1.0 since the user CAN run `persona list` to see what's registered.

- (b) **Fix in this same step** — update `NO_PERSONAS_REGISTERED` constant or split the error path so unknown-persona errors say "persona 'foo' not found. Available: heartbit-ghost:x" (use `registry.list().join(", ")`).

Choose (b) for a better UX. Edit `crates/heartbit-cli/src/persona.rs`:

Find the constant definition (around line 119):

```rust
const NO_PERSONAS_REGISTERED: &str = "No personas registered. (heartbit-ghost or another persona crate must be linked into this build.)";
```

Find the corpus and profile sub-arms (around lines 158–166) — those legitimately describe the "no registered persona" path and may keep the constant. The `Show / Run / Phase / Pause / Resume / ExportPreferences / Audit` arm (around line 147) is the one that should change to surface available personas.

Update the unknown-persona arm to list available personas:

```rust
        PersonaCommand::Show { name }
        | PersonaCommand::Run { name, .. }
        | PersonaCommand::Phase { name, .. }
        | PersonaCommand::Pause { name }
        | PersonaCommand::Resume { name }
        | PersonaCommand::ExportPreferences { name, .. }
        | PersonaCommand::Audit { name, .. } => {
            if registry.get(&name).is_none() {
                let available = registry.list();
                let suffix = if available.is_empty() {
                    NO_PERSONAS_REGISTERED.to_string()
                } else {
                    format!("Available personas: {}.", available.join(", "))
                };
                return Err(anyhow!("persona '{name}' not found. {suffix}"));
            }
            // P1.0 ships the registration shell; subcommand bodies land in
            // later sub-phases (P1.1–P1.4) alongside the persona's tools,
            // voice modeling, and pipeline.
            Err(anyhow!(
                "persona '{name}': subcommand body is not yet implemented (P1.0 scaffolding stub). The persona is registered; its tools, voice modeling, and pipeline land in later sub-phases."
            ))
        }
```

(Two paths now: empty-registry → keep `NO_PERSONAS_REGISTERED`; non-empty registry but unknown name → list available personas.)

- [ ] **Step 11: Add a unit test that exercises the new "Available personas" path**

In the `crates/heartbit-cli/src/persona.rs` test mod, add a new test:

```rust
    #[tokio::test]
    async fn show_unknown_persona_with_registered_persona_lists_available() {
        // Manually populate the registry with the heartbit-ghost stub, then
        // ask for a name that isn't there; the error must surface the
        // available persona name(s).
        let mut r = PersonaRegistry::new();
        heartbit_ghost::register(&mut r);
        let result = dispatch(
            PersonaCommand::Show {
                name: "doesnotexist".into(),
            },
            &r,
        )
        .await;
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("persona 'doesnotexist' not found"));
        assert!(msg.contains("Available personas: heartbit-ghost:x"));
        // Must NOT regress to the empty-registry hint when one IS registered.
        assert!(!msg.contains("No personas registered"));
    }
```

Run it:

```bash
cargo test -p heartbit-cli show_unknown_persona_with_registered_persona_lists_available 2>&1 | tail -3
```

Expected: 1 passed.

- [ ] **Step 12: Re-run the smoke test from Step 10 to confirm the new wording**

```bash
cargo build -p heartbit-cli --release  # rebuild after Step 10/11 edits
./target/release/heartbit persona show foo 2>&1
echo "exit=$?"
```

Expected stderr: `Error: persona 'foo' not found. Available personas: heartbit-ghost:x.` (exit non-zero).

- [ ] **Step 13: Re-run all CLI tests**

```bash
cargo test -p heartbit-cli 2>&1 | tail -5
```

Expected: previously-passing CLI tests still pass + the 1 new test from Step 11 (so heartbit-cli's persona tests now total 5).

- [ ] **Step 14: Final workspace quality gate**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features 2>&1 | tail -5
```

All four signals green.

- [ ] **Step 15: Commit**

```bash
git add crates/heartbit-cli/Cargo.toml crates/heartbit-cli/src/persona.rs
git commit -m "$(cat <<'EOF'
feat(cli): register heartbit-ghost persona at startup; surface available personas

heartbit-cli depends on heartbit-ghost and calls
heartbit_ghost::register() in persona::run() before dispatch, so the
PersonaRegistry the CLI hands to dispatch() now contains the
heartbit-ghost:x stub.

Cleanups:
- The Phase 0-specific "subcommand bodies not implemented in Phase 0"
  error string referenced an outdated phase; reworded to be P1.0-aware
  and interpolate the persona name.
- The unknown-persona error now lists registered persona names instead
  of always claiming the registry is empty (which it no longer is once
  any persona crate is linked).
- New test asserts the "Available personas: ..." path fires correctly.

After this commit:
  heartbit persona list             → "heartbit-ghost:x"
  heartbit persona show heartbit-ghost:x → "subcommand body not yet implemented (P1.0 scaffolding stub)"
  heartbit persona show foo         → "persona 'foo' not found. Available personas: heartbit-ghost:x."

Refs: docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md §9 P1.0
EOF
)"
```

---

## Acceptance criteria

P1.0 is done when:

- `cargo build -p heartbit-ghost` clean
- `cargo test -p heartbit-ghost` shows 6 passing tests
- `cargo build -p heartbit-cli --release` clean
- `heartbit persona list` prints exactly `heartbit-ghost:x` and exits 0
- `heartbit persona show heartbit-ghost:x` exits non-zero with the new "not yet implemented (P1.0 scaffolding stub)" wording (NOT "Phase 0", NOT "not found")
- `heartbit persona show foo` exits non-zero with `"persona 'foo' not found. Available personas: heartbit-ghost:x."` wording
- `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green
- 2 commits on `feat/heartbit-ghost`: one for Task 1 (new crate), one for Task 2 (CLI wiring)

## Out of scope (explicit)

- Real persona expansion: agents, tools, triggers, review spec — all deferred to P1.1–P1.4
- X tool family beyond the existing `twitter_post` builtin
- Voice modeling (corpus, profile schema, blend algorithm)
- A/B feedback loop / Telegram review wiring / autonomy phases
- The 4 documented `TODO(phase-1):` sites — they're orthogonal multi-tenant integration work; addressed in a later P1.x or a dedicated tenant-integration plan
- Daemon-side registration: the daemon's `build_runner` callback constructs an `AgentRunner`, not a `PersonaRegistry`. Daemon persona dispatch (cron + sensor + Telegram review channels) lands with P1.3 when the persona has triggers and a review spec

## Reference

- Spec: `docs/superpowers/specs/2026-05-07-heartbit-ghost-x-agent-design.md` §9 P1.0
- Foundation: `docs/superpowers/specs/2026-05-07-heartbit-foundation-design.md` (just shipped as `2026.507.3`)
- Phase 0 plan: `docs/superpowers/plans/2026-05-07-heartbit-foundation.md`
