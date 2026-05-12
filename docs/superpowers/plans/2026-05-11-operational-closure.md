# Operational Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three operator-facing paper-cuts: graceful fallback for `HEARTBIT_GHOST_OPERATOR_USER_ID`, a `--validate-config` startup flag, and a consolidated `docs/operating-heartbit.md` knob reference.

**Architecture:**
1. Extract operator-user-id resolution into a unit-testable helper that falls back through `persona_mentions.user_id` → env var → skip-with-error.
2. Add a `--validate-config` flag on `heartbit daemon` that runs config validation + static cross-reference and path checks, then exits.
3. Document the runtime knobs (jitter, top_n, engagement window, active hours, kill switches) in a new operator-focused page.

Per-persona skips emit a loud `tracing::error!` banner *and* increment a new `heartbit_persona_posts_skipped_total{persona,reason}` counter so silent degradation has both log and metrics visibility.

**Tech Stack:** Rust 1.x, anyhow (CLI), thiserror (lib), clap derive, prometheus, tracing, tokio.

---

## File Structure

**Created:**
- `crates/heartbit-cli/src/daemon/operator_id.rs` — `resolve_operator_user_id()` helper + unit tests
- `crates/heartbit-cli/src/daemon/validate.rs` — `validate_daemon_config()` static-validation entry + unit tests
- `docs/operating-heartbit.md` — operator-facing knob reference

**Modified:**
- `crates/heartbit/src/daemon/metrics.rs` — add `persona_posts_skipped_total: IntCounterVec` + getter + tests
- `crates/heartbit-cli/src/main.rs` — extend `Commands::Daemon` with `--validate-config`; route to validator
- `crates/heartbit-cli/src/daemon/mod.rs` — declare new modules; replace inline env-var lookup with helper; wire metric on skip; expose `validate_config` entry point
- `crates/heartbit-cli/src/persona.rs` — add regression test pinning the strict-error contract for `persona post` (no fallback)

---

## Task 1: Add `persona_posts_skipped_total` metric

**Files:**
- Modify: `crates/heartbit/src/daemon/metrics.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module at the bottom of `crates/heartbit/src/daemon/metrics.rs`:

```rust
#[test]
fn persona_posts_skipped_metric_registers_and_increments() {
    let m = DaemonMetrics::new().unwrap();
    m.inc_persona_posts_skipped("heartbit-ghost:x", "missing_operator_user_id");
    m.inc_persona_posts_skipped("heartbit-ghost:x", "missing_operator_user_id");
    m.inc_persona_posts_skipped("other:x", "missing_post_history_path");

    let mut buf = Vec::new();
    TextEncoder::new().encode(&m.registry.gather(), &mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    assert!(
        text.contains(
            r#"heartbit_persona_posts_skipped_total{persona="heartbit-ghost:x",reason="missing_operator_user_id"} 2"#
        ),
        "missing or wrong counter line: {text}"
    );
    assert!(
        text.contains(
            r#"heartbit_persona_posts_skipped_total{persona="other:x",reason="missing_post_history_path"} 1"#
        ),
        "missing second-label counter line: {text}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package heartbit --lib daemon::metrics::tests::persona_posts_skipped_metric_registers_and_increments -- --nocapture`

Expected: FAIL with `no method named 'inc_persona_posts_skipped' found`.

- [ ] **Step 3: Add the field**

In `crates/heartbit/src/daemon/metrics.rs`, add after the `cascade_escalations_total` field (around line 82) inside `pub struct DaemonMetrics`:

```rust
    // Persona posts (prefix: heartbit_persona_)
    persona_posts_skipped_total: IntCounterVec,
```

- [ ] **Step 4: Construct the metric**

In `impl DaemonMetrics::new()`, after the `cascade_escalations_total = IntCounterVec::new(...)` block (around line 349-356), add:

```rust
        let persona_posts_skipped_total = IntCounterVec::new(
            Opts::new(
                "heartbit_persona_posts_skipped_total",
                "Number of [[daemon.persona_posts]] entries skipped at startup due to misconfiguration",
            ),
            &["persona", "reason"],
        )?;
```

- [ ] **Step 5: Register and store the metric**

After the `registry.register(Box::new(cascade_escalations_total.clone()))?;` line (around line 410), add:

```rust
        registry.register(Box::new(persona_posts_skipped_total.clone()))?;
```

In the `Ok(Self { ... })` block (around line 412-456), add `persona_posts_skipped_total,` after `cascade_escalations_total,` as a struct-init line.

- [ ] **Step 6: Add the getter**

Find the end of the `impl DaemonMetrics` block (just before `#[cfg(test)] mod tests` near the bottom of the file). Add this method inside the impl block:

```rust
    /// Increment the persona-posts skip counter for `persona` with `reason`.
    ///
    /// Used at daemon startup when a `[[daemon.persona_posts]]` entry cannot
    /// be activated (e.g. unresolved operator user-id). The counter exposes
    /// silent skips so dashboards can alert on `rate(...) > 0`.
    pub fn inc_persona_posts_skipped(&self, persona: &str, reason: &str) {
        self.persona_posts_skipped_total
            .with_label_values(&[persona, reason])
            .inc();
    }
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cargo test --package heartbit --lib daemon::metrics::tests::persona_posts_skipped_metric_registers_and_increments -- --nocapture`

Expected: PASS.

- [ ] **Step 8: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit -- -D warnings`

Expected: clean (no diff, no warnings).

- [ ] **Step 9: Commit**

```bash
git add crates/heartbit/src/daemon/metrics.rs
git commit -m "feat(daemon): heartbit_persona_posts_skipped_total counter"
```

---

## Task 2: Extract `resolve_operator_user_id` helper

**Files:**
- Create: `crates/heartbit-cli/src/daemon/operator_id.rs`
- Modify: `crates/heartbit-cli/src/daemon/mod.rs` (just to declare the module; wiring happens in Task 3)

- [ ] **Step 1: Write the failing tests**

Create `crates/heartbit-cli/src/daemon/operator_id.rs` with **only the tests** to start:

```rust
//! Resolves the X/Twitter operator user-id for a `[[daemon.persona_posts]]`
//! entry. Resolution order (most-specific to least-specific):
//!
//! 1. A matching `[[daemon.persona_mentions]]` entry with the same `persona`
//!    slug — uses its `user_id` field directly.
//! 2. The `HEARTBIT_GHOST_OPERATOR_USER_ID` environment variable.
//!
//! Both sources missing returns `Err(OperatorIdError::Unresolved)`; the
//! daemon caller then logs an error banner, increments the skip metric,
//! and continues past the entry (rather than crash-looping the process).

use heartbit_core::config::daemon::PersonaMentionsConfig;

/// Source from which an operator user-id was resolved. Used by callers for
/// logging — the resolved id itself is the primary return value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorIdSource {
    PersonaMentions,
    EnvVar,
}

#[derive(Debug, thiserror::Error)]
pub enum OperatorIdError {
    #[error(
        "no operator user-id for persona '{persona}': set HEARTBIT_GHOST_OPERATOR_USER_ID \
         or add a matching [[daemon.persona_mentions]] entry"
    )]
    Unresolved { persona: String },
}

/// Resolve the operator user-id for `persona_slug`, given the daemon's
/// `persona_mentions` config and the process environment.
///
/// `env_lookup` is an injectable hook so tests don't need to mutate
/// real `std::env`. Production callers pass `|k| std::env::var(k).ok()`.
pub fn resolve_operator_user_id(
    persona_slug: &str,
    persona_mentions: &[PersonaMentionsConfig],
    env_lookup: impl Fn(&str) -> Option<String>,
) -> Result<(String, OperatorIdSource), OperatorIdError> {
    if let Some(m) = persona_mentions
        .iter()
        .find(|m| m.persona == persona_slug && m.enabled)
    {
        return Ok((m.user_id.clone(), OperatorIdSource::PersonaMentions));
    }
    if let Some(v) = env_lookup("HEARTBIT_GHOST_OPERATOR_USER_ID") {
        if !v.trim().is_empty() {
            return Ok((v, OperatorIdSource::EnvVar));
        }
    }
    Err(OperatorIdError::Unresolved {
        persona: persona_slug.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::config::daemon::PersonaMentionsConfig;

    fn mention(persona: &str, user_id: &str) -> PersonaMentionsConfig {
        // Hand-roll a minimal PersonaMentionsConfig via TOML so we don't
        // depend on every default-fn here.
        let toml = format!(
            r#"
persona = "{persona}"
user_id = "{user_id}"
"#
        );
        toml::from_str(&toml).expect("valid PersonaMentionsConfig fixture")
    }

    #[test]
    fn persona_mentions_match_wins_over_env() {
        let mentions = vec![mention("heartbit-ghost:x", "111")];
        let env = |k: &str| match k {
            "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("999".into()),
            _ => None,
        };
        let (id, src) = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap();
        assert_eq!(id, "111");
        assert_eq!(src, OperatorIdSource::PersonaMentions);
    }

    #[test]
    fn env_used_when_no_mentions_match() {
        let mentions = vec![mention("other:x", "222")];
        let env = |k: &str| match k {
            "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("777".into()),
            _ => None,
        };
        let (id, src) = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap();
        assert_eq!(id, "777");
        assert_eq!(src, OperatorIdSource::EnvVar);
    }

    #[test]
    fn empty_env_value_falls_through_to_unresolved() {
        let mentions: Vec<PersonaMentionsConfig> = vec![];
        let env = |k: &str| match k {
            "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("   ".into()),
            _ => None,
        };
        let err = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap_err();
        match err {
            OperatorIdError::Unresolved { persona } => {
                assert_eq!(persona, "heartbit-ghost:x");
            }
        }
    }

    #[test]
    fn disabled_mentions_entry_is_ignored() {
        let mut m = mention("heartbit-ghost:x", "111");
        m.enabled = false;
        let mentions = vec![m];
        let env = |k: &str| match k {
            "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("777".into()),
            _ => None,
        };
        let (id, src) = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap();
        assert_eq!(id, "777");
        assert_eq!(src, OperatorIdSource::EnvVar);
    }

    #[test]
    fn no_sources_returns_unresolved() {
        let mentions: Vec<PersonaMentionsConfig> = vec![];
        let env = |_k: &str| None;
        let err = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("heartbit-ghost:x"), "msg: {msg}");
        assert!(msg.contains("HEARTBIT_GHOST_OPERATOR_USER_ID"), "msg: {msg}");
        assert!(msg.contains("persona_mentions"), "msg: {msg}");
    }
}
```

Then add the module declaration in `crates/heartbit-cli/src/daemon/mod.rs` at the top (around line 1-6) where other `mod` lines live:

```rust
mod auth;
mod eval;
mod execute;
mod handlers;
mod memory;
mod operator_id;
mod types;
```

- [ ] **Step 2: Run tests to verify they fail (or compile errors first)**

Run: `cargo test --package heartbit-cli --lib daemon::operator_id::tests -- --nocapture`

Expected: PASS on first run because the helper is fully implemented above. If a build error appears (e.g. missing `thiserror` in `heartbit-cli`), add `thiserror = { workspace = true }` to `crates/heartbit-cli/Cargo.toml` under `[dependencies]` and re-run.

> NOTE: If `cargo test` is green on first run, that's expected — the helper, its types, and the tests are all in this single file. If you prefer strict Red→Green, comment out the body of `resolve_operator_user_id`, watch the tests fail, then uncomment.

- [ ] **Step 3: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-cli -- -D warnings`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-cli/src/daemon/operator_id.rs crates/heartbit-cli/src/daemon/mod.rs crates/heartbit-cli/Cargo.toml
git commit -m "feat(daemon): resolve_operator_user_id helper with mentions→env fallback"
```

---

## Task 3: Wire the resolver into daemon startup (skip-with-banner on failure)

**Files:**
- Modify: `crates/heartbit-cli/src/daemon/mod.rs:265-273` — replace inline `env::var` with helper; add skip path

- [ ] **Step 1: Read the current site**

The current block lives at lines 265-273 of `crates/heartbit-cli/src/daemon/mod.rs`:

```rust
            // V1: operator user_id comes from env. Once P1.5 merges, this can be
            // cross-referenced from persona_mentions config.
            let operator_user_id =
                std::env::var("HEARTBIT_GHOST_OPERATOR_USER_ID").map_err(|_| {
                    anyhow::anyhow!(
                        "persona_posts persona='{}': HEARTBIT_GHOST_OPERATOR_USER_ID must be set",
                        cfg.persona
                    )
                })?;
```

This sits inside `for cfg in &daemon_config.persona_posts { ... }`, after the `post_history_store` branch that ends around line 264 and before the `entries.contains_key` duplicate check at line 274. The local `metrics: Option<Arc<DaemonMetrics>>` was created earlier (line 78) and is in scope.

- [ ] **Step 2: Replace the inline lookup with the helper + skip path**

Replace the lines above (265-273) with:

```rust
            let operator_user_id = match operator_id::resolve_operator_user_id(
                &cfg.persona,
                &daemon_config.persona_mentions,
                |k| std::env::var(k).ok(),
            ) {
                Ok((id, source)) => {
                    tracing::info!(
                        persona = %cfg.persona,
                        source = ?source,
                        "resolved operator_user_id for persona_posts entry"
                    );
                    id
                }
                Err(e) => {
                    // Loud banner: this is strictly worse for ops than a
                    // crash-loop if it goes unnoticed, so log at error level
                    // and bump a metric. Daemon stays up; other personas run.
                    tracing::error!(
                        persona = %cfg.persona,
                        "SKIPPING [[daemon.persona_posts]] entry: {e}"
                    );
                    if let Some(ref m) = metrics {
                        m.inc_persona_posts_skipped(&cfg.persona, "missing_operator_user_id");
                    }
                    continue;
                }
            };
```

Then, at the top of the file (around line 26), add `operator_id` to the `use self::` line. The existing line is:

```rust
use self::auth::{JwtMiddlewareState, auth_middleware, jwt_auth_middleware, resolve_auth_tokens};
```

Add a new line above or below it:

```rust
use self::operator_id::resolve_operator_user_id;
```

> Note: with this import the inner block can be simplified — the `operator_id::resolve_operator_user_id(...)` call may instead be written `resolve_operator_user_id(...)`. Either form is fine; pick whichever reads better with the surrounding imports.

- [ ] **Step 3: Add the integration test**

Direct unit-testing of `run_daemon` is out of scope (it's a long async startup). Instead, append a test to `crates/heartbit-cli/src/daemon/operator_id.rs` that locks in the *intended caller behavior* by exercising the helper end-to-end with both sources present, only env, and neither:

```rust
    #[test]
    fn caller_contract_three_paths() {
        // Path 1: mentions present → use it
        let mentions = vec![mention("p1", "from_mentions")];
        let (id, _) = resolve_operator_user_id(
            "p1",
            &mentions,
            |_| Some("from_env".into()),
        ).unwrap();
        assert_eq!(id, "from_mentions");

        // Path 2: mentions absent for this persona → use env
        let (id, _) = resolve_operator_user_id(
            "p2",
            &mentions,
            |_| Some("from_env".into()),
        ).unwrap();
        assert_eq!(id, "from_env");

        // Path 3: neither → caller must skip this persona
        let err = resolve_operator_user_id(
            "p2",
            &[],
            |_| None,
        ).unwrap_err();
        assert!(matches!(err, OperatorIdError::Unresolved { .. }));
    }
```

- [ ] **Step 4: Build + run all tests**

Run: `cargo test --package heartbit-cli --lib -- --nocapture`

Expected: all green.

- [ ] **Step 5: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-cli -- -D warnings`

Expected: clean.

- [ ] **Step 6: Smoke-test the skip path manually**

Run: `unset HEARTBIT_GHOST_OPERATOR_USER_ID && cargo run --bin heartbit -- daemon --config daemon-dev.toml 2>&1 | head -50`

(Or whichever local config file has `[[daemon.persona_posts]]` enabled.)

Expected: an `ERROR ... SKIPPING [[daemon.persona_posts]] entry` line for each affected persona, the daemon then continues binding HTTP. Kill the daemon with Ctrl-C after confirming.

> If the only persona is the now-skipped one, that's fine — the daemon will still serve `/healthz` and `/v1/tasks/execute`.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-cli/src/daemon/mod.rs crates/heartbit-cli/src/daemon/operator_id.rs
git commit -m "feat(daemon): skip persona_posts entry instead of crash-looping when operator_user_id is missing"
```

---

## Task 4: Regression test: `persona post` CLI stays strict

**Files:**
- Modify: `crates/heartbit-cli/src/persona.rs:402-405` — keep as-is, but add a test that pins the contract

The one-off CLI (`heartbit persona post <name>` without `--topic`) has a different UX contract from the supervised daemon: a typo in the env var should fail loudly so the operator notices immediately. We pin this by asserting that the env-var lookup site still uses `anyhow!` (not the fallback helper).

- [ ] **Step 1: Locate the existing call site**

The site at `crates/heartbit-cli/src/persona.rs:402-405` reads:

```rust
            let operator_user_id = std::env::var("HEARTBIT_GHOST_OPERATOR_USER_ID")
                .map_err(|_| anyhow!(
                    "HEARTBIT_GHOST_OPERATOR_USER_ID must be set for `persona post` without --topic"
                ))?;
```

This is the contract we want to preserve.

- [ ] **Step 2: Add a guard test**

Append to the `#[cfg(test)] mod tests` block in `crates/heartbit-cli/src/persona.rs` (find the existing `mod tests` block in that file; if there is no test module yet, create one at the bottom):

```rust
    /// The one-off `persona post` CLI must hard-error when
    /// HEARTBIT_GHOST_OPERATOR_USER_ID is missing — different contract from
    /// the supervised daemon. If a future refactor wires the fallback helper
    /// in here too, this test will need to be updated *and* the change
    /// reviewed against `docs/operating-heartbit.md`.
    #[test]
    fn persona_post_uses_strict_env_var_check_not_fallback_helper() {
        // Grep the source file for the canonical strict pattern. We assert
        // on a stable substring rather than the literal error message so
        // wording tweaks don't break the test.
        let src = include_str!("persona.rs");
        assert!(
            src.contains(r#"std::env::var("HEARTBIT_GHOST_OPERATOR_USER_ID")"#),
            "strict env-var check removed from persona.rs"
        );
        assert!(
            src.contains("must be set for `persona post` without --topic"),
            "strict error message changed; if intentional, update the doc"
        );
        // Negative: the fallback helper must NOT be used from this file —
        // it would silently substitute a persona_mentions value, masking
        // a config typo from the operator running the one-off command.
        assert!(
            !src.contains("resolve_operator_user_id"),
            "persona post must keep strict env-var contract — see Task 4 plan"
        );
    }
```

If no `#[cfg(test)] mod tests` exists in `persona.rs`, add at the very bottom:

```rust
#[cfg(test)]
mod tests {
    // (insert the test above here)
}
```

- [ ] **Step 3: Run test**

Run: `cargo test --package heartbit-cli --lib persona::tests::persona_post_uses_strict_env_var_check_not_fallback_helper -- --nocapture`

Expected: PASS.

- [ ] **Step 4: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-cli -- -D warnings`

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-cli/src/persona.rs
git commit -m "test(persona): pin strict env-var contract for `persona post` one-off"
```

---

## Task 5: Add `--validate-config` flag skeleton

**Files:**
- Create: `crates/heartbit-cli/src/daemon/validate.rs`
- Modify: `crates/heartbit-cli/src/main.rs:117-125` — extend the `Daemon` variant
- Modify: `crates/heartbit-cli/src/main.rs:404-416` — route to validator
- Modify: `crates/heartbit-cli/src/daemon/mod.rs` — declare `validate` module, export `validate_config_only`

- [ ] **Step 1: Write the validator with failing tests first**

Create `crates/heartbit-cli/src/daemon/validate.rs`:

```rust
//! Static validation for daemon configs.
//!
//! Performs only filesystem + cross-reference checks — no network calls.
//! Surfaces all findings (don't fail-fast) so the operator sees every
//! issue in one pass.
//!
//! Layered on top of `HeartbitConfig::validate()` (which is already invoked
//! by `HeartbitConfig::from_file`); this module covers the gap between
//! "parses + structural validation" and "daemon will actually start".

use std::path::Path;

use heartbit::HeartbitConfig;
use heartbit_core::config::daemon::DaemonConfig;

use super::operator_id::resolve_operator_user_id;

/// A single validation finding. Each finding maps to a single fixable
/// misconfiguration and includes the persona/entry it relates to so the
/// operator can locate the offending block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub kind: ValidationIssueKind,
    pub context: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationIssueKind {
    /// `[[daemon.persona_posts]]` entry's persona cannot resolve an
    /// operator user-id from either persona_mentions or the env.
    MissingOperatorUserId,
    /// `post_history_store = "jsonl"` but no `post_history_path` set.
    MissingPostHistoryPath,
    /// `budget_store = "jsonl"` but no `budget_path` set
    /// (mentions config).
    MissingBudgetPath,
    /// A JSONL store path's parent directory doesn't exist (and can't be
    /// created at validation time — we only check, don't mutate).
    NonexistentParentDir { path: String },
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ValidationIssueKind::MissingOperatorUserId => write!(
                f,
                "{}: cannot resolve operator user-id (set HEARTBIT_GHOST_OPERATOR_USER_ID \
                 or add a matching [[daemon.persona_mentions]] entry)",
                self.context
            ),
            ValidationIssueKind::MissingPostHistoryPath => write!(
                f,
                "{}: post_history_store = \"jsonl\" but post_history_path is not set",
                self.context
            ),
            ValidationIssueKind::MissingBudgetPath => write!(
                f,
                "{}: budget_store = \"jsonl\" but budget_path is not set",
                self.context
            ),
            ValidationIssueKind::NonexistentParentDir { path } => write!(
                f,
                "{}: parent directory of '{path}' does not exist",
                self.context
            ),
        }
    }
}

/// Static validation entry point. Returns the list of issues; empty list = OK.
pub fn validate_daemon_config(
    config: &HeartbitConfig,
    env_lookup: impl Fn(&str) -> Option<String>,
    path_exists: impl Fn(&Path) -> bool,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    let Some(daemon_config) = config.daemon.as_ref() else {
        return issues; // Nothing daemon-specific to validate.
    };

    validate_persona_posts(daemon_config, &env_lookup, &path_exists, &mut issues);
    validate_persona_mentions(daemon_config, &path_exists, &mut issues);

    issues
}

fn validate_persona_posts(
    daemon: &DaemonConfig,
    env_lookup: &impl Fn(&str) -> Option<String>,
    path_exists: &impl Fn(&Path) -> bool,
    issues: &mut Vec<ValidationIssue>,
) {
    for cfg in &daemon.persona_posts {
        if !cfg.enabled {
            continue;
        }
        let context = format!("[[daemon.persona_posts]] persona='{}'", cfg.persona);

        // 1. operator user-id resolvable?
        if resolve_operator_user_id(&cfg.persona, &daemon.persona_mentions, |k| env_lookup(k)).is_err() {
            issues.push(ValidationIssue {
                kind: ValidationIssueKind::MissingOperatorUserId,
                context: context.clone(),
            });
        }

        // 2. jsonl store needs a path
        if cfg.post_history_store == "jsonl" {
            match cfg.post_history_path.as_deref() {
                None => issues.push(ValidationIssue {
                    kind: ValidationIssueKind::MissingPostHistoryPath,
                    context: context.clone(),
                }),
                Some(p) => check_parent_dir(p, &context, path_exists, issues),
            }
        }
    }
}

fn validate_persona_mentions(
    daemon: &DaemonConfig,
    path_exists: &impl Fn(&Path) -> bool,
    issues: &mut Vec<ValidationIssue>,
) {
    for cfg in &daemon.persona_mentions {
        if !cfg.enabled {
            continue;
        }
        let context = format!("[[daemon.persona_mentions]] persona='{}'", cfg.persona);

        if cfg.budget_store == "jsonl" {
            match cfg.budget_path.as_deref() {
                None => issues.push(ValidationIssue {
                    kind: ValidationIssueKind::MissingBudgetPath,
                    context: context.clone(),
                }),
                Some(p) => check_parent_dir(p, &context, path_exists, issues),
            }
        }
        if cfg.mention_store == "jsonl" {
            if let Some(p) = cfg.mention_store_path.as_deref() {
                check_parent_dir(p, &context, path_exists, issues);
            }
        }
    }
}

fn check_parent_dir(
    raw_path: &str,
    context: &str,
    path_exists: &impl Fn(&Path) -> bool,
    issues: &mut Vec<ValidationIssue>,
) {
    // Tilde expansion — mirror what the daemon startup does. Plan keeps it
    // local so this module doesn't take a dependency on the CLI's
    // expand_tilde helper.
    let expanded: std::path::PathBuf = if let Some(stripped) = raw_path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            Path::new(&home).join(stripped)
        } else {
            Path::new(raw_path).to_path_buf()
        }
    } else {
        Path::new(raw_path).to_path_buf()
    };
    if let Some(parent) = expanded.parent() {
        if !parent.as_os_str().is_empty() && !path_exists(parent) {
            issues.push(ValidationIssue {
                kind: ValidationIssueKind::NonexistentParentDir {
                    path: expanded.display().to_string(),
                },
                context: context.to_string(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::config::daemon::{
        DaemonConfig, PersonaMentionsConfig, PersonaPostsConfig,
    };

    fn config_with_daemon(daemon: DaemonConfig) -> HeartbitConfig {
        // HeartbitConfig has many required fields; build via TOML and then
        // splice the daemon section in for test focus.
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let mut c: HeartbitConfig = toml::from_str(toml).expect("base config parses");
        c.daemon = Some(daemon);
        c
    }

    fn minimal_daemon() -> DaemonConfig {
        let toml = r#""#;
        toml::from_str(toml).expect("default DaemonConfig parses from empty TOML")
    }

    fn mention(persona: &str, user_id: &str) -> PersonaMentionsConfig {
        let toml = format!(
            r#"
persona = "{persona}"
user_id = "{user_id}"
"#
        );
        toml::from_str(&toml).expect("PersonaMentionsConfig fixture parses")
    }

    fn post(persona: &str) -> PersonaPostsConfig {
        let toml = format!(
            r#"
persona = "{persona}"
"#
        );
        toml::from_str(&toml).expect("PersonaPostsConfig fixture parses")
    }

    #[test]
    fn empty_daemon_config_has_no_issues() {
        let cfg = config_with_daemon(minimal_daemon());
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert!(issues.is_empty(), "issues: {issues:?}");
    }

    #[test]
    fn persona_posts_without_operator_user_id_is_flagged() {
        let mut d = minimal_daemon();
        d.persona_posts.push(post("heartbit-ghost:x"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert_eq!(issues.len(), 1, "issues: {issues:?}");
        assert_eq!(issues[0].kind, ValidationIssueKind::MissingOperatorUserId);
        assert!(issues[0].context.contains("heartbit-ghost:x"));
    }

    #[test]
    fn persona_posts_with_matching_mentions_passes() {
        let mut d = minimal_daemon();
        d.persona_posts.push(post("heartbit-ghost:x"));
        d.persona_mentions.push(mention("heartbit-ghost:x", "42"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert!(issues.is_empty(), "issues: {issues:?}");
    }

    #[test]
    fn persona_posts_with_env_var_passes() {
        let mut d = minimal_daemon();
        d.persona_posts.push(post("heartbit-ghost:x"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(
            &cfg,
            |k| match k {
                "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("42".into()),
                _ => None,
            },
            |_| true,
        );
        assert!(issues.is_empty(), "issues: {issues:?}");
    }

    #[test]
    fn jsonl_post_history_without_path_is_flagged() {
        let mut d = minimal_daemon();
        let mut p = post("heartbit-ghost:x");
        p.post_history_store = "jsonl".into();
        d.persona_posts.push(p);
        d.persona_mentions.push(mention("heartbit-ghost:x", "42"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert!(
            issues.iter().any(|i| i.kind == ValidationIssueKind::MissingPostHistoryPath),
            "issues: {issues:?}"
        );
    }

    #[test]
    fn jsonl_post_history_with_missing_parent_dir_is_flagged() {
        let mut d = minimal_daemon();
        let mut p = post("heartbit-ghost:x");
        p.post_history_store = "jsonl".into();
        p.post_history_path = Some("/definitely/not/a/real/dir/file.jsonl".into());
        d.persona_posts.push(p);
        d.persona_mentions.push(mention("heartbit-ghost:x", "42"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| false);
        assert!(
            issues.iter().any(|i| matches!(
                &i.kind,
                ValidationIssueKind::NonexistentParentDir { path } if path.contains("/definitely/not/a/real/dir")
            )),
            "issues: {issues:?}"
        );
    }

    #[test]
    fn no_daemon_section_returns_no_issues() {
        let toml = r#"
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
"#;
        let cfg: HeartbitConfig = toml::from_str(toml).expect("config parses");
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert!(issues.is_empty(), "issues: {issues:?}");
    }
}
```

- [ ] **Step 2: Declare the module**

In `crates/heartbit-cli/src/daemon/mod.rs`, add `mod validate;` near the other `mod` declarations at the top (around line 1-7). Final block:

```rust
mod auth;
mod eval;
mod execute;
mod handlers;
mod memory;
mod operator_id;
mod types;
mod validate;
```

- [ ] **Step 3: Run tests**

Run: `cargo test --package heartbit-cli --lib daemon::validate::tests -- --nocapture`

Expected: all pass. If `HeartbitConfig` deserialize fails on the minimal TOML, the test fixture may need additional required keys — examine the actual error and add the minimum needed fields.

- [ ] **Step 4: Add `validate_config_only` entry point**

Append to `crates/heartbit-cli/src/daemon/mod.rs` (immediately after `run_daemon`'s closing brace — find the matching `}` for the function that starts at line 39):

```rust
/// Static-validate the daemon config and print findings to stderr.
/// Returns Ok if no issues found, Err with a count summary otherwise.
///
/// Performs no network calls and no Kafka/DB initialization — safe to run
/// against a production config file from anywhere.
pub async fn validate_config_only(config_path: &std::path::Path) -> Result<()> {
    let config = HeartbitConfig::from_file(config_path)
        .with_context(|| format!("failed to load config from {}", config_path.display()))?;

    let issues = validate::validate_daemon_config(
        &config,
        |k| std::env::var(k).ok(),
        |p| p.exists(),
    );

    if issues.is_empty() {
        eprintln!("✓ {} validates clean", config_path.display());
        return Ok(());
    }

    eprintln!("✗ {} has {} issue(s):", config_path.display(), issues.len());
    for issue in &issues {
        eprintln!("  - {issue}");
    }
    anyhow::bail!("config validation found {} issue(s)", issues.len());
}
```

> If `eprintln!` is forbidden by an existing clippy lint in this repo, swap to `tracing::warn!` / `tracing::info!`. The `cargo clippy` gate at the end of this task will catch it.

- [ ] **Step 5: Extend the clap subcommand**

In `crates/heartbit-cli/src/main.rs`, replace the `Daemon` variant (lines 117-125):

```rust
    /// Run the daemon: long-running Kafka-backed task execution with HTTP API
    Daemon {
        /// Address to bind the HTTP API (overrides config)
        #[arg(long)]
        bind: Option<String>,
        /// Print structured agent events to stderr as one-line JSON
        #[arg(long, short)]
        verbose: bool,
        /// Validate config and exit (no Kafka, no HTTP bind, no DB connect)
        #[arg(long)]
        validate_config: bool,
    },
```

Then update the match arm (lines 404-416):

```rust
        Some(Commands::Daemon {
            bind,
            verbose,
            validate_config,
        }) => {
            let config_path = cli
                .config
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("heartbit.toml"));
            if validate_config {
                daemon::validate_config_only(config_path).await
            } else {
                daemon::run_daemon(
                    config_path,
                    bind.as_deref(),
                    verbose,
                    cli.observability.as_deref(),
                )
                .await
            }
        }
```

- [ ] **Step 6: Smoke-test from the CLI**

Run on a known-broken config first (no env, no mentions block):

```bash
unset HEARTBIT_GHOST_OPERATOR_USER_ID && cargo run --bin heartbit -- --config daemon-dev.toml daemon --validate-config
echo "exit: $?"
```

Expected: stderr lists each issue, exit code is non-zero. Then re-run with `HEARTBIT_GHOST_OPERATOR_USER_ID=999`:

```bash
HEARTBIT_GHOST_OPERATOR_USER_ID=999 cargo run --bin heartbit -- --config daemon-dev.toml daemon --validate-config
echo "exit: $?"
```

Expected: stderr says `validates clean`, exit code is 0 (assuming the other JSONL paths exist).

- [ ] **Step 7: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-cli -- -D warnings && cargo test --package heartbit-cli --lib`

Expected: clean across the board.

- [ ] **Step 8: Commit**

```bash
git add crates/heartbit-cli/src/daemon/validate.rs crates/heartbit-cli/src/daemon/mod.rs crates/heartbit-cli/src/main.rs
git commit -m "feat(daemon): heartbit daemon --validate-config flag"
```

---

## Task 6: Write `docs/operating-heartbit.md`

**Files:**
- Create: `docs/operating-heartbit.md`

The doc is operator-focused (SRE / on-call). Other docs cover the schema (`configuration.md`) and the daemon architecture overview (`daemon.md`); this one is "here are the knobs you'll touch in production and how to use them safely."

- [ ] **Step 1: Draft the doc**

Create `docs/operating-heartbit.md` with this content:

````markdown
# Operating Heartbit

Operational reference for running `heartbit daemon` in production. Focuses on the knobs an operator turns day-to-day (cadence, jitter, kill switches, observability) rather than the full TOML schema — see [`configuration.md`](configuration.md) for that.

For the daemon's overall architecture (Kafka, HTTP API, channels), see [`daemon.md`](daemon.md).

## Pre-flight

Validate a config without starting Kafka, HTTP, or the database:

```bash
heartbit daemon --config heartbit.toml --validate-config
```

Exit code is non-zero with a list of issues if anything would cause startup misbehavior — most importantly:
- A `[[daemon.persona_posts]]` entry whose operator user-id is unresolvable
- A JSONL store with a missing parent directory
- Required path/identifier fields left empty

Run this as part of CI or a deploy hook before rolling out config changes.

## Proactive posting knobs

`[[daemon.persona_posts]]` controls the proactive-post loop. Defaults are listed in [`configuration.md`](configuration.md#daemon).

| Knob | Default | When to change |
|---|---|---|
| `enabled` | `true` | Set to `false` to pause posting for this persona without removing the block. |
| `post_interval_seconds` | `14400` (4h) | Lower for higher-volume personas; minimum is `60`. |
| `interval_jitter_pct` | `25` (±25%) | Lower for stricter cadence (debugging only); higher (up to `50`) to look less bot-like. `0` disables jitter — test only. |
| `active_hours` | unset (24/7) | Set to e.g. `"08:00-22:00"` to restrict to local waking hours. |
| `candidates_per_draft` | `3` | Higher = more LLM cost per tick but better picks. |
| `post_history_store` | `"in_memory"` | Use `"jsonl"` for restart durability. |
| `post_history_path` | required for jsonl | Tilde-expanded; ensure the parent directory exists. |
| `post_history_lookback_days` | `30` | How far back duplicate-topic detection scans. |
| `topic_brief` | unset | Free-form prompt addendum for the topic generator. |

## Engagement-feedback loop

Engagement metrics are refreshed in the background and the top-N engaged posts are injected into the writer as few-shot exemplars.

| Knob | Default | Behavior |
|---|---|---|
| `engagement_refresh_seconds` | `21600` (6h) | Tick interval for the engagement collector. |
| `engagement_top_n` | `5` | Number of top-engaged posts injected as writer exemplars. **Set to `0` to disable injection** — fastest kill switch for the feedback loop. |
| `engagement_min_age_hours` | `24` | Tweets younger than this are skipped (algorithm hasn't fanned out yet). |
| `engagement_max_age_days` | `30` | Tweets older than this are dropped from refresh. |

Engagement metrics live alongside the post history in `.heartbit/engagement/{persona}.jsonl` (jsonl mode) — the file is operator-readable JSONL.

## Mention polling knobs

`[[daemon.persona_mentions]]` controls reactive replies. Two safety layers matter most for ops.

**Thread / bot guards** — leave on unless debugging:
- `enable_thread_depth_guard = true` skips threads this persona already replied to.
- `enable_bot_heuristic_guard = true` evaluates handle patterns, follower ratio, and account age.
- `bot_heuristic_threshold = 2` is the number of signals required to skip.

**Per-conversation cap** — `per_conversation_max_replies = 2` prevents back-and-forth lockup.

**Daily LLM budget** — `daily_token_budget = 100000` (set to `null` to disable) is the safest hard stop when the bot is over-engaging.

## Kill switches

In order of granularity:

1. **Disable one entry**: set `enabled = false` on the specific `[[daemon.persona_posts]]` or `[[daemon.persona_mentions]]` block and reload the daemon.
2. **Disable engagement injection**: `engagement_top_n = 0` keeps the bot posting but removes top-engaged few-shots — useful if the feedback loop is producing degenerate writing.
3. **Pause posting only**: set `active_hours = "00:00-00:01"` (a one-minute window in the past or future). Crude but it doesn't lose any history state.
4. **Stop the daemon**: graceful Ctrl-C / SIGTERM — in-flight ticks complete; queued ticks are dropped.

## Operator user-id resolution

For `[[daemon.persona_posts]]` to function, each enabled entry needs an X user-id. The daemon resolves it in this order at startup:

1. A matching `[[daemon.persona_mentions]]` entry's `user_id` field — preferred (single source of truth in config).
2. The `HEARTBIT_GHOST_OPERATOR_USER_ID` environment variable — kept for backward-compat and quick overrides.
3. **Skip this entry**: the daemon logs an `ERROR ... SKIPPING [[daemon.persona_posts]] entry: ...` banner and increments `heartbit_persona_posts_skipped_total{persona, reason}`. Other personas/entries continue to run; the daemon does **not** crash-loop.

Set an alert on `rate(heartbit_persona_posts_skipped_total[5m]) > 0` so silent skips don't go unnoticed.

The one-off `heartbit persona post <name>` CLI (without `--topic`) does **not** apply this fallback — it errors hard so a misconfigured one-shot run fails fast and visibly to the operator at the terminal.

## Environment variables operators care about

The full list is in [`configuration.md`](configuration.md#environment-variables). The subset commonly tuned at deploy time:

| Variable | Purpose |
|---|---|
| `HEARTBIT_GHOST_OPERATOR_USER_ID` | X user-id fallback for persona_posts (per the resolution order above). |
| `HEARTBIT_GHOST_PERSONAS` | Override persona-config directory (defaults to `~/.heartbit/personas`). |
| `HEARTBIT_GHOST_PROFILES` | Override voice-profile directory. |
| `HEARTBIT_GHOST_CORPORA` | Override corpus directory. |
| `HEARTBIT_TOOL_PROFILE` | Pre-filter tool definitions: `conversational` / `standard` / `full`. |
| `HEARTBIT_AUDIT_RETAIN_DAYS` | Days to keep audit-log rows before pruning. |
| `HEARTBIT_SESSION_PRUNE` | `1` to trim old tool results before each LLM call. |
| `HEARTBIT_TELEGRAM_TOKEN` | Telegram bot token for review-delivery / interactive channel. |

## Observability quick-reference

`/metrics` exposes Prometheus counters. Most useful for ops:

| Metric | Why it matters |
|---|---|
| `heartbit_persona_posts_skipped_total{persona, reason}` | Increments when a persona_posts entry is silently disabled at startup. Alert on `rate > 0`. |
| `heartbit_daemon_tasks_failed_total{tenant}` | Per-tenant task failure counter. |
| `heartbit_llm_cost_usd_total{agent, tenant}` | Running LLM-cost estimate. |
| `heartbit_reliability_doom_loops_detected_total` | Bumps when the doom-loop guard short-circuits a runaway agent. |
| `heartbit_cascade_escalations_total{from_tier, to_tier, reason}` | LLM cascade escalations. |

`/healthz` and `/readyz` return 200 when the daemon is up and ready. Use `/readyz` for load-balancer probes.

## Common operations

**Pause one persona without losing history**: edit the block, set `enabled = false`, restart the daemon. The JSONL store on disk is unchanged.

**Reset engagement few-shots cleanly**: stop the daemon, move `.heartbit/engagement/{persona}.jsonl` aside (don't delete — keep for postmortem), restart. Writer falls back to no exemplars on the next tick.

**Recover from a runaway bot**: bump `daily_token_budget` low (e.g. `10000`), restart. The bot will hit the cap quickly and stop replying; investigate logs without further engagement on X.

**Migrate JSONL stores**: the JSONL files are append-only; copying them between hosts preserves history. Ensure parent dirs exist on the destination, then `--validate-config` to confirm.
````

- [ ] **Step 2: Lint-check the doc rendering**

Optional but recommended:

```bash
# If markdownlint is installed:
markdownlint docs/operating-heartbit.md || true

# Or just eyeball the rendered file:
less docs/operating-heartbit.md
```

- [ ] **Step 3: Cross-reference check**

Confirm `docs/configuration.md` has anchors `#daemon` and `#environment-variables` that the new doc links to. They exist as Markdown auto-generated anchors from the `## Daemon` and `## Environment Variables` headings — no action needed unless those headings have been renamed.

- [ ] **Step 4: Commit**

```bash
git add docs/operating-heartbit.md
git commit -m "docs: add operating-heartbit.md with operator knob reference"
```

---

## Task 7: Final integration smoke + plan close-out

- [ ] **Step 1: Full quality gate**

Run from the repo root:

```bash
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
```

Expected: all green.

- [ ] **Step 2: End-to-end smoke**

In a separate terminal, with a real config that has `[[daemon.persona_posts]]`:

```bash
# Validator path
HEARTBIT_GHOST_OPERATOR_USER_ID=999 \
  cargo run --bin heartbit -- --config daemon-dev.toml daemon --validate-config

# Skip-with-banner path
unset HEARTBIT_GHOST_OPERATOR_USER_ID && \
  cargo run --bin heartbit -- --config daemon-dev.toml daemon 2>&1 | head -30
```

Expected:
- Validator prints `✓ ... validates clean` and exits 0.
- Daemon prints `ERROR ... SKIPPING [[daemon.persona_posts]] entry: ...` and stays up.

> Per CLAUDE.md: **do not pkill the daemon**. Stop with Ctrl-C in its own terminal.

- [ ] **Step 3: Check metric exposure**

While the daemon is running (in the skip-with-banner case), confirm the new metric is in the registry:

```bash
curl -s http://localhost:3000/metrics | grep persona_posts_skipped_total
```

Expected: one or more lines with `heartbit_persona_posts_skipped_total{persona="...",reason="missing_operator_user_id"} <N>`.

- [ ] **Step 4: Mark the plan complete in the operational closure tracker**

Update `tasks/lessons.md` if any non-obvious gotchas surfaced during implementation (e.g. clap derive conflict with `validate_config` field, deserialization fixture surprises). Keep the entries terse — one or two lines per lesson.

- [ ] **Step 5: Final commit (if step 4 produced changes)**

```bash
git add tasks/lessons.md
git commit -m "docs(lessons): operational-closure implementation gotchas"
```

---

## Verification matrix

Cross-checks each plan task against the spec the user provided:

| Spec item | Covered by |
|---|---|
| Task #47: HEARTBIT_GHOST_OPERATOR_USER_ID fallback so daemon won't crash-loop | Tasks 2, 3 (helper + wiring + skip path) |
| Operator user-id source from persona_mentions config (resolves existing TODO) | Task 2 (resolution order) |
| Visibility signal so skip is not silent | Task 1 (metric) + Task 3 (tracing::error banner) |
| `persona post` one-off stays strict | Task 4 (regression test) |
| `--validate-config` fails at startup not 5 minutes in | Task 5 |
| Validator scope: static (no external API calls) | Task 5 (env_lookup + path_exists injection only) |
| `docs/operating-heartbit.md` with jitter, top_n, engagement window, active hours, kill switches | Task 6 |

---

## Notes for the implementer

- Keep `heartbit::HeartbitConfig::from_file` as the *only* entrypoint for config loading — don't re-parse TOML in the validator. The helper takes an already-loaded `HeartbitConfig`.
- The `path_exists` and `env_lookup` parameters on `validate_daemon_config` exist *only* to keep unit tests pure. Production callers pass `Path::exists` and `std::env::var(...).ok()`.
- If `eprintln!` is restricted by the workspace lint config, swap the validator output to `tracing::warn!` and adjust the smoke test commands to redirect stderr.
- The `--validate-config` flag is deliberately a no-op when `[daemon]` is absent (returns OK with zero issues). That matches the daemon's behavior of bailing with "daemon section required" only at actual startup.
