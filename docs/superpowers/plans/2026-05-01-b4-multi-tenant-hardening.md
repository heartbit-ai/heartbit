# B4 — Multi-Tenant Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound tool-call concurrency, enforce filesystem path policy across all builtins, make tenant scope a load-bearing type on Memory + AuditTrail, add audit retention, and add tenant columns to Postgres — clearing the five multi-tenant gaps from the 2026-04-30 production-readiness review.

**Architecture:** New `TenantScope` (owned, `String` tenant_id) and `CorePathPolicy` types in `heartbit-core`. Memory + AuditTrail traits gain `&TenantScope` parameter. Filesystem builtins gain `with_path_policy(Arc<CorePathPolicy>)`. `SandboxPolicy` (umbrella) composes `CorePathPolicy`. Postgres migrations add the missing tenant columns and indexes. CLI/config wires it all.

**Tech Stack:** Rust 2021, `tokio`, `sqlx`, `Pin<Box<dyn Future>>` async traits, `glob` crate for path matching.

**Spec:** `docs/superpowers/specs/2026-05-01-b4-multi-tenant-hardening-design.md`

---

## Pre-flight (Task 0)

This is a process step, not a code task. Do not skip.

- [ ] **Step 0.1: Confirm PR #3 (user-docs) has merged to `main`**

The advisor flagged that branching B4 before PR #3 merges produces a heisen-merge state. Check:
```bash
gh pr view 3 --json state,mergedAt 2>/dev/null
git log origin/main --oneline -5
```
Expected: PR #3 state is `MERGED`. If not, **stop** and wait. Pick this plan back up after the merge lands on main.

- [ ] **Step 0.2: Create the B4 worktree**

Use the `superpowers:using-git-worktrees` skill. Branch name: `b4-multi-tenant-hardening`. Run baseline tests:
```bash
cd .worktrees/b4-multi-tenant-hardening
cargo test --workspace --lib
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt -- --check
```
Expected: all green. If anything fails, do not proceed — investigate and report.

---

## Task 1: TenantScope type

**Files:**
- Create: `crates/heartbit-core/src/auth/tenant.rs`
- Modify: `crates/heartbit-core/src/auth/mod.rs`
- Modify: `crates/heartbit/src/auth.rs` (or `crates/heartbit/src/lib.rs` re-exports)

- [ ] **Step 1.1: Write the failing test**

Create `crates/heartbit-core/src/auth/tenant.rs` with only the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_single_tenant_with_empty_id() {
        let scope = TenantScope::default();
        assert_eq!(scope.tenant_id, "");
        assert!(scope.user_id.is_none());
        assert!(scope.is_single_tenant());
    }

    #[test]
    fn new_with_real_tenant_is_not_single_tenant() {
        let scope = TenantScope::new("tenant-123");
        assert_eq!(scope.tenant_id, "tenant-123");
        assert!(!scope.is_single_tenant());
    }

    #[test]
    fn new_with_empty_string_collapses_to_single_tenant() {
        let scope = TenantScope::new("");
        assert!(scope.is_single_tenant());
    }

    #[test]
    fn with_user_attaches_identity() {
        let scope = TenantScope::new("acme").with_user("user-42");
        assert_eq!(scope.tenant_id, "acme");
        assert_eq!(scope.user_id.as_deref(), Some("user-42"));
    }

    #[test]
    fn equal_scopes_compare_equal() {
        let a = TenantScope::new("acme").with_user("u1");
        let b = TenantScope::new("acme").with_user("u1");
        assert_eq!(a, b);
    }
}
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cargo test -p heartbit-core auth::tenant 2>&1 | tail -10
```
Expected: compile error — `TenantScope` not defined.

- [ ] **Step 1.3: Implement the type**

Prepend to `crates/heartbit-core/src/auth/tenant.rs`:

```rust
//! Tenant + optional user identity for scoping memory, audit, and policy.

/// Tenant + optional user identity for scoping memory, audit, and policy
/// decisions. Owned (no lifetime parameter) so it composes cleanly into
/// async contexts and can be stored in `Arc`-shared state.
///
/// `tenant_id` is `String`, not `Uuid`, to match the existing
/// `UserContext.tenant_id: String` (deliberate: JWT `tid` claims from
/// Auth0 / Cognito / Okta etc. are not always UUIDs). The sentinel for
/// "single-tenant mode" is the empty string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TenantScope {
    pub tenant_id: String,
    pub user_id: Option<String>,
}

impl TenantScope {
    /// Multi-tenant scope from an externally-supplied tenant id (typically
    /// JWT `tid` claim). Empty strings collapse to `single_tenant()` so a
    /// dropped scope can never silently widen to all tenants.
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            user_id: None,
        }
    }

    /// Add a user identity (typically `sub` claim from JWT).
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Single-tenant default; `tenant_id == ""`.
    pub fn single_tenant() -> Self {
        Self {
            tenant_id: String::new(),
            user_id: None,
        }
    }

    pub fn is_single_tenant(&self) -> bool {
        self.tenant_id.is_empty()
    }
}

impl Default for TenantScope {
    fn default() -> Self {
        Self::single_tenant()
    }
}
```

Wire the module: edit `crates/heartbit-core/src/auth/mod.rs`, add:

```rust
pub mod tenant;

pub use tenant::TenantScope;
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
cargo test -p heartbit-core auth::tenant 2>&1 | tail -10
```
Expected: 5 tests pass.

- [ ] **Step 1.5: Re-export from umbrella**

Edit `crates/heartbit/src/lib.rs` (or wherever `heartbit::auth` is composed). Add to the existing `pub use heartbit_core::*;` re-export verification — it already covers it. Add a targeted re-export only if `heartbit_core::auth::TenantScope` is not already glob-reachable:

```rust
// crates/heartbit/src/lib.rs — verify reachability
pub use heartbit_core::auth::TenantScope;
```

Then add a smoke test in `crates/heartbit/src/lib.rs` test module:

```rust
#[cfg(test)]
mod umbrella_tests {
    #[test]
    fn tenant_scope_reachable_from_umbrella() {
        let _ = crate::TenantScope::default();
    }
}
```

Run:
```bash
cargo test -p heartbit umbrella_tests 2>&1 | tail -5
```
Expected: PASS.

- [ ] **Step 1.6: Commit**

```bash
git add crates/heartbit-core/src/auth/tenant.rs \
        crates/heartbit-core/src/auth/mod.rs \
        crates/heartbit/src/lib.rs
git commit -m "feat(core): TenantScope type in heartbit_core::auth::tenant

Owned (no lifetime), String tenant_id matching UserContext convention,
empty-string sentinel for single-tenant mode collapse-on-construction."
```

- [ ] **Step 1.7: Add `From<&UserContext>` (umbrella, gated)**

The conversion from JWT context lives in the umbrella because `UserContext` lives there. Edit `crates/heartbit/src/daemon/types.rs`, add at the bottom (or in a new file):

```rust
use heartbit_core::auth::TenantScope;

impl From<&UserContext> for TenantScope {
    fn from(ctx: &UserContext) -> Self {
        TenantScope::new(&ctx.tenant_id).with_user(&ctx.user_id)
    }
}
```

Add to the same file's `#[cfg(test)] mod tests`:

```rust
#[test]
fn user_context_to_tenant_scope() {
    let ctx = UserContext {
        user_id: "u1".into(),
        tenant_id: "acme".into(),
        roles: vec![],
        raw_token: None,
    };
    let scope: TenantScope = (&ctx).into();
    assert_eq!(scope.tenant_id, "acme");
    assert_eq!(scope.user_id.as_deref(), Some("u1"));
}
```

Run:
```bash
cargo test -p heartbit user_context_to_tenant_scope 2>&1 | tail -5
```
Expected: PASS.

- [ ] **Step 1.8: Commit**

```bash
git add crates/heartbit/src/daemon/types.rs
git commit -m "feat(daemon): From<&UserContext> for TenantScope"
```

---

## Task 2: CorePathPolicy

**Files:**
- Create: `crates/heartbit-core/src/sandbox.rs`
- Modify: `crates/heartbit-core/src/lib.rs`
- Modify: `crates/heartbit-core/Cargo.toml` (promote `glob` from transitive to direct)
- Modify: `crates/heartbit-core/src/error.rs` (add `Error::Sandbox` variant if not present)

- [ ] **Step 2.1: Verify `Error::Sandbox` exists, add if missing**

```bash
grep -n 'Sandbox' crates/heartbit-core/src/error.rs
```
If absent, edit `crates/heartbit-core/src/error.rs`:
```rust
#[error("sandbox violation: {0}")]
Sandbox(String),
```
(Place alphabetically inside the existing `enum Error` definition.)

- [ ] **Step 2.2: Promote `glob` to a direct dep**

Edit `crates/heartbit-core/Cargo.toml`. In the `[dependencies]` section add:
```toml
glob = "0.3"
```

Run:
```bash
cargo build -p heartbit-core 2>&1 | tail -5
```
Expected: builds (glob was already transitive, so no new lockfile change beyond promotion).

- [ ] **Step 2.3: Write the failing test**

Create `crates/heartbit-core/src/sandbox.rs`:

```rust
//! Path-level sandbox policy shared across filesystem-touching builtins.

use std::path::{Path, PathBuf};

use crate::error::Error;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("heartbit-sandbox-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn allows_path_under_allowed_dir() {
        let root = tmp();
        let file = root.join("ok.txt");
        fs::write(&file, b"x").unwrap();
        let policy = CorePathPolicy::builder().allow_dir(&root).build().unwrap();
        assert!(policy.check_path(&file).is_ok());
    }

    #[test]
    fn denies_path_outside_allowed_dirs() {
        let root = tmp();
        let policy = CorePathPolicy::builder().allow_dir(&root).build().unwrap();
        let bad = std::env::temp_dir().join("not-in-root");
        fs::write(&bad, b"x").ok();
        let err = policy.check_path(&bad).unwrap_err();
        assert!(matches!(err, Error::Sandbox(_)));
    }

    #[test]
    fn denies_glob_match_inside_allowed_dir() {
        let root = tmp();
        let dotenv = root.join(".env");
        fs::write(&dotenv, b"x").unwrap();
        let policy = CorePathPolicy::builder()
            .allow_dir(&root)
            .deny_glob("**/.env")
            .build()
            .unwrap();
        let err = policy.check_path(&dotenv).unwrap_err();
        assert!(matches!(err, Error::Sandbox(_)));
    }

    #[test]
    fn empty_allowlist_denies_everything() {
        let policy = CorePathPolicy::builder().build().unwrap();
        let err = policy.check_path(&PathBuf::from("/tmp")).unwrap_err();
        assert!(matches!(err, Error::Sandbox(_)));
    }
}
```

- [ ] **Step 2.4: Run test to verify it fails**

```bash
cargo test -p heartbit-core sandbox::tests 2>&1 | tail -15
```
Expected: compile error — `CorePathPolicy` not defined.

- [ ] **Step 2.5: Implement**

Prepend to `crates/heartbit-core/src/sandbox.rs` (above the test module):

```rust
use std::sync::Arc;

/// Path-level policy for filesystem-touching tools.
#[derive(Debug, Clone)]
pub struct CorePathPolicy {
    allowed_dirs: Vec<PathBuf>,
    deny_globs: Vec<glob::Pattern>,
}

impl CorePathPolicy {
    pub fn builder() -> CorePathPolicyBuilder {
        CorePathPolicyBuilder::default()
    }

    /// Returns Ok(()) if `path` is allowed, Err(Error::Sandbox(...)) otherwise.
    /// Canonicalizes the input so symlinks pointing outside `allowed_dirs`
    /// are rejected.
    pub fn check_path(&self, path: &Path) -> Result<(), Error> {
        let canonical = path
            .canonicalize()
            .map_err(|e| Error::Sandbox(format!("canonicalize {}: {e}", path.display())))?;

        let allowed = self
            .allowed_dirs
            .iter()
            .any(|root| canonical.starts_with(root));
        if !allowed {
            return Err(Error::Sandbox(format!(
                "path {} not under any allowed directory",
                canonical.display()
            )));
        }

        for pat in &self.deny_globs {
            if pat.matches_path(&canonical) {
                return Err(Error::Sandbox(format!(
                    "path {} matches deny pattern {}",
                    canonical.display(),
                    pat.as_str()
                )));
            }
        }

        Ok(())
    }
}

#[derive(Default, Debug)]
pub struct CorePathPolicyBuilder {
    allowed_dirs: Vec<PathBuf>,
    deny_globs: Vec<String>,
}

impl CorePathPolicyBuilder {
    pub fn allow_dir(mut self, dir: impl AsRef<Path>) -> Self {
        // Canonicalize at build time so check_path can do `starts_with` cleanly.
        if let Ok(canon) = dir.as_ref().canonicalize() {
            self.allowed_dirs.push(canon);
        } else {
            self.allowed_dirs.push(dir.as_ref().to_path_buf());
        }
        self
    }

    pub fn deny_glob(mut self, pat: impl Into<String>) -> Self {
        self.deny_globs.push(pat.into());
        self
    }

    pub fn build(self) -> Result<CorePathPolicy, Error> {
        let deny_globs = self
            .deny_globs
            .into_iter()
            .map(|p| {
                glob::Pattern::new(&p)
                    .map_err(|e| Error::Sandbox(format!("invalid deny glob {p}: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CorePathPolicy {
            allowed_dirs: self.allowed_dirs,
            deny_globs,
        })
    }
}
```

Edit `crates/heartbit-core/src/lib.rs`, add `pub mod sandbox;` and `pub use sandbox::{CorePathPolicy, CorePathPolicyBuilder};`.

- [ ] **Step 2.6: Run test to verify it passes**

```bash
cargo test -p heartbit-core sandbox::tests 2>&1 | tail -10
```
Expected: 4 tests pass.

- [ ] **Step 2.7: Commit**

```bash
git add crates/heartbit-core/src/sandbox.rs \
        crates/heartbit-core/src/lib.rs \
        crates/heartbit-core/src/error.rs \
        crates/heartbit-core/Cargo.toml \
        Cargo.lock
git commit -m "feat(core): CorePathPolicy in heartbit_core::sandbox

Path allowlist + glob denylist with canonicalize-first symlink
defense. Plain Rust (no landlock dep) so it compiles on every target."
```

---

## Task 3: max_tool_calls_per_turn

**Naming.** The existing `max_tools_per_turn: Option<usize>` (`builder.rs:49`) is a *pre-filter on the tool definition set* offered to the LLM. The new `max_tool_calls_per_turn` is a *cap on dispatched tool invocations after the LLM responds*. Different feature, similar name. Document both in rustdoc on the new builder method.

**Files:**
- Modify: `crates/heartbit-core/src/agent/builder.rs`
- Modify: `crates/heartbit-core/src/agent/runner.rs`

- [ ] **Step 3.1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block in `crates/heartbit-core/src/agent/mod.rs` (where the other builder tests live):

```rust
#[tokio::test]
async fn max_tool_calls_per_turn_caps_excess_dispatch() {
    use crate::agent::AgentRunnerBuilder;
    use crate::llm::testing::CannedProvider;

    // Canned response with 3 tool_use blocks
    let provider = CannedProvider::with_tool_calls(vec![
        ("a".into(), serde_json::json!({})),
        ("b".into(), serde_json::json!({})),
        ("c".into(), serde_json::json!({})),
    ]);

    let runner = AgentRunnerBuilder::new()
        .provider(std::sync::Arc::new(provider))
        .name("test")
        .system_prompt("hi")
        .max_tool_calls_per_turn(2)
        .build()
        .unwrap();

    let result = runner.run("go").await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("tool-call cap exceeded"), "got: {err}");
}

#[test]
fn max_tool_calls_per_turn_zero_is_rejected_at_build() {
    use crate::agent::AgentRunnerBuilder;
    use crate::llm::testing::CannedProvider;
    let provider = CannedProvider::with_text("hi");
    let err = AgentRunnerBuilder::new()
        .provider(std::sync::Arc::new(provider))
        .name("t")
        .system_prompt("p")
        .max_tool_calls_per_turn(0)
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("max_tool_calls_per_turn must be > 0"));
}
```

If `CannedProvider::with_tool_calls` does not exist yet, check `crates/heartbit-core/src/llm/testing.rs` for the closest helper and use it. If the test helper produces a single tool call only, *extend* it as part of this task (its own commit beforehand) rather than skipping the test.

- [ ] **Step 3.2: Run test to verify it fails**

```bash
cargo test -p heartbit-core max_tool_calls_per_turn 2>&1 | tail -10
```
Expected: compile error — method does not exist.

- [ ] **Step 3.3: Add the field to the builder**

Edit `crates/heartbit-core/src/agent/builder.rs`. Add after line 51 (`max_identical_tool_calls`):
```rust
    pub(super) max_tool_calls_per_turn: Option<u32>,
```
Add to the `default()` impl (around line 88 — wherever `max_identical_tool_calls: None,` lives):
```rust
            max_tool_calls_per_turn: None,
```
Add the builder method (place near `max_tools_per_turn` around line 257):
```rust
    /// Cap dispatched tool calls per LLM turn. Distinct from
    /// `max_tools_per_turn` (which limits the *tool definitions* offered
    /// to the LLM). This caps the *invocations* the LLM emits per turn.
    /// Excess calls return `Error::Agent` with partial usage attached.
    ///
    /// Default: None (unlimited). Recommended for production: 8.
    /// Zero is rejected at build time.
    pub fn max_tool_calls_per_turn(mut self, cap: u32) -> Self {
        self.max_tool_calls_per_turn = Some(cap);
        self
    }
```
Add the build-time validation near the existing `max_tools_per_turn == Some(0)` check (around line 476):
```rust
        if self.max_tool_calls_per_turn == Some(0) {
            return Err(Error::Config(
                "max_tool_calls_per_turn must be > 0 if set".into(),
            ));
        }
```
Add the field to the `AgentRunner` construction (around line 604):
```rust
            max_tool_calls_per_turn: self.max_tool_calls_per_turn,
```

- [ ] **Step 3.4: Add the field to the runner and enforce it**

Edit `crates/heartbit-core/src/agent/runner.rs`. Add after line 129 (`max_tools_per_turn: Option<usize>`):
```rust
    pub(super) max_tool_calls_per_turn: Option<u32>,
```
Add to the runner default near line 208:
```rust
            max_tool_calls_per_turn: None,
```
Insert the cap check immediately after `let tool_calls = response.tool_calls();` (line 790):
```rust
                if let Some(cap) = self.max_tool_calls_per_turn
                    && tool_calls.len() as u32 > cap
                {
                    let err = Error::Agent(format!(
                        "tool-call cap exceeded: turn produced {} calls, max is {cap}",
                        tool_calls.len()
                    ));
                    self.emit(AgentEvent::RunFailed {
                        agent: self.name.clone(),
                        error: err.to_string(),
                        partial_usage: total_usage,
                    });
                    return Err((err, total_usage));
                }
```

- [ ] **Step 3.5: Run test to verify it passes**

```bash
cargo test -p heartbit-core max_tool_calls_per_turn 2>&1 | tail -10
```
Expected: 2 tests pass.

- [ ] **Step 3.6: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings
```
Expected: clean.

- [ ] **Step 3.7: Commit**

```bash
git add crates/heartbit-core/src/agent/builder.rs \
        crates/heartbit-core/src/agent/runner.rs \
        crates/heartbit-core/src/agent/mod.rs
git commit -m "feat(core): max_tool_calls_per_turn on AgentRunner

Caps dispatched tool_use blocks per LLM turn. Excess returns
Error::Agent with partial usage. Distinct from existing
max_tools_per_turn (which pre-filters definitions)."
```

---

## Task 4: Memory trait takes &TenantScope

This is the largest commit in the plan — it touches every Memory call site. Use the test gate at every file.

**Files:**
- Modify: `crates/heartbit-core/src/memory/mod.rs` (trait)
- Modify: `crates/heartbit-core/src/memory/in_memory.rs` (impl)
- Modify: `crates/heartbit-core/src/memory/namespaced.rs` (impl)
- Modify: `crates/heartbit/src/memory/postgres.rs` (impl)
- Modify: every test that constructs a Memory and calls store/recall — search and update.

- [ ] **Step 4.1: Find every call site that needs updating**

```bash
grep -rn 'memory\.store\|memory\.recall\|memory\.update\|memory\.forget\|memory\.add_link\|memory\.prune' crates/ tests/ 2>/dev/null > /tmp/b4-memory-callsites.txt
wc -l /tmp/b4-memory-callsites.txt
```
Save the file; use it as a checklist when updating call sites.

- [ ] **Step 4.2: Write the failing isolation test**

Create `crates/heartbit-core/src/memory/tenant_isolation_tests.rs`:

```rust
//! Tenant isolation tests covering Memory impls.
//!
//! Stored entries under one tenant must not leak to recall under another.

#[cfg(test)]
mod tests {
    use crate::auth::TenantScope;
    use crate::memory::{InMemoryStore, Memory, MemoryEntry, MemoryQuery};

    fn entry(id: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: id.into(),
            agent: "a".into(),
            content: content.into(),
            // ... default the rest; copy whatever the existing test helper does
            ..MemoryEntry::default()
        }
    }

    #[tokio::test]
    async fn recall_does_not_leak_across_tenants() {
        let store = InMemoryStore::new();
        let acme = TenantScope::new("acme");
        let globex = TenantScope::new("globex");

        store.store(&acme, entry("a1", "acme-secret")).await.unwrap();
        store.store(&globex, entry("g1", "globex-secret")).await.unwrap();

        let acme_results = store
            .recall(&acme, MemoryQuery::default())
            .await
            .unwrap();
        assert_eq!(acme_results.len(), 1);
        assert_eq!(acme_results[0].id, "a1");

        let globex_results = store
            .recall(&globex, MemoryQuery::default())
            .await
            .unwrap();
        assert_eq!(globex_results.len(), 1);
        assert_eq!(globex_results[0].id, "g1");
    }

    #[tokio::test]
    async fn forget_does_not_delete_other_tenant() {
        let store = InMemoryStore::new();
        let acme = TenantScope::new("acme");
        let globex = TenantScope::new("globex");

        store.store(&acme, entry("a1", "x")).await.unwrap();
        store.store(&globex, entry("g1", "y")).await.unwrap();

        // Try to forget acme's id under globex's scope — should not delete acme's entry.
        let removed = store.forget(&globex, "a1").await.unwrap();
        assert!(!removed);

        let acme_results = store.recall(&acme, MemoryQuery::default()).await.unwrap();
        assert_eq!(acme_results.len(), 1);
    }

    #[tokio::test]
    async fn default_scope_is_single_tenant_namespace() {
        let store = InMemoryStore::new();
        let scope = TenantScope::default();
        store.store(&scope, entry("s1", "x")).await.unwrap();
        let results = store.recall(&scope, MemoryQuery::default()).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].author_tenant_id.as_deref(), Some(""));
    }
}
```

Wire the file in `crates/heartbit-core/src/memory/mod.rs`:
```rust
#[cfg(test)]
mod tenant_isolation_tests;
```

- [ ] **Step 4.3: Run test to verify it fails**

```bash
cargo test -p heartbit-core memory::tenant_isolation 2>&1 | tail -15
```
Expected: compile errors — methods don't take `&TenantScope` yet.

- [ ] **Step 4.4: Update the trait**

Edit `crates/heartbit-core/src/memory/mod.rs`. Replace the existing trait definition (around line 151) with:

```rust
pub trait Memory: Send + Sync {
    fn store(
        &self,
        scope: &TenantScope,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    fn recall(
        &self,
        scope: &TenantScope,
        query: MemoryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>>;

    fn update(
        &self,
        scope: &TenantScope,
        id: &str,
        content: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    fn forget(
        &self,
        scope: &TenantScope,
        id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>>;

    fn add_link(
        &self,
        _scope: &TenantScope,
        _id: &str,
        _related_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }

    fn prune(
        &self,
        _scope: &TenantScope,
        _min_strength: f64,
        _min_age: chrono::Duration,
        _agent_prefix: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async { Ok(0) })
    }
}
```

Add `use crate::auth::TenantScope;` near the existing `use` block at the top of the file.

- [ ] **Step 4.5: Update `InMemoryStore`**

Edit `crates/heartbit-core/src/memory/in_memory.rs`. For each method, accept `scope: &TenantScope` as the first parameter.

`store`:
```rust
fn store(
    &self,
    scope: &TenantScope,
    mut entry: MemoryEntry,
) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
    entry.author_tenant_id = Some(scope.tenant_id.clone());
    entry.author_user_id = scope.user_id.clone();
    let entries = self.entries.clone();
    Box::pin(async move {
        let mut guard = entries
            .write()
            .map_err(|_| Error::Memory("poisoned lock".into()))?;
        guard.insert(entry.id.clone(), entry);
        Ok(())
    })
}
```

`recall`:
```rust
fn recall(
    &self,
    scope: &TenantScope,
    query: MemoryQuery,
) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>> {
    let entries = self.entries.clone();
    let scope_tenant = scope.tenant_id.clone();
    Box::pin(async move {
        let guard = entries
            .read()
            .map_err(|_| Error::Memory("poisoned lock".into()))?;
        let candidates: Vec<MemoryEntry> = guard
            .values()
            .filter(|e| {
                e.author_tenant_id.as_deref().unwrap_or("") == scope_tenant
            })
            .cloned()
            .collect();
        // existing scoring/filter logic on `candidates` using `query`
        Ok(score_and_filter(candidates, &query))
    })
}
```

`forget`:
```rust
fn forget(
    &self,
    scope: &TenantScope,
    id: &str,
) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>> {
    let entries = self.entries.clone();
    let scope_tenant = scope.tenant_id.clone();
    let id = id.to_string();
    Box::pin(async move {
        let mut guard = entries
            .write()
            .map_err(|_| Error::Memory("poisoned lock".into()))?;
        match guard.get(&id) {
            Some(e) if e.author_tenant_id.as_deref().unwrap_or("") == scope_tenant => {
                guard.remove(&id);
                Ok(true)
            }
            _ => Ok(false),
        }
    })
}
```

`update` follows the same pattern as `forget` — load, check tenant matches, update content. `prune` filters by tenant before the existing strength/age logic.

- [ ] **Step 4.6: Update `NamespacedMemory`**

Edit `crates/heartbit-core/src/memory/namespaced.rs`. The wrapper currently captures a `prefix: String` at construction. Keep that prefix (it's intra-tenant namespacing — preserved) and forward `scope` through unchanged. For each method:

```rust
fn store(&self, scope: &TenantScope, entry: MemoryEntry) -> ... {
    let mut entry = entry;
    entry.agent = format!("{}{}", self.prefix, entry.agent);
    self.inner.store(scope, entry)
}

fn recall(&self, scope: &TenantScope, query: MemoryQuery) -> ... {
    let mut query = query;
    query.agent_prefix = Some(self.prefix.clone());
    self.inner.recall(scope, query)
}
```
(... and similar passthrough for `update`/`forget`/`prune`/`add_link`.)

- [ ] **Step 4.7: Update `PostgresMemoryStore`**

Edit `crates/heartbit/src/memory/postgres.rs`. **Defer schema migration to Task 8** — for now, just thread `scope` through and `WHERE author_tenant_id = $N` (the column will be added in Task 8; this code compiles now and tests pass once Task 8's migration runs). Bind `entry.author_tenant_id` after setting it from `scope.tenant_id`:

```rust
fn store(&self, scope: &TenantScope, mut entry: MemoryEntry) -> ... {
    entry.author_tenant_id = Some(scope.tenant_id.clone());
    entry.author_user_id = scope.user_id.clone();
    // existing INSERT, with two more bound params for the new columns
    // (Task 8 adds them to the schema; this code is forward-compatible)
    let pool = self.pool.clone();
    Box::pin(async move {
        sqlx::query(
            "INSERT INTO memories (..., author_tenant_id, author_user_id) \
             VALUES (..., $N, $N+1) ON CONFLICT (id) DO UPDATE SET ..."
        )
        // ... existing binds ...
        .bind(entry.author_tenant_id.as_deref().unwrap_or(""))
        .bind(entry.author_user_id.as_deref())
        .execute(&pool)
        .await
        .map_err(|e| Error::Memory(format!("store: {e}")))?;
        Ok(())
    })
}
```

`recall` adds the filter:
```rust
.bind(scope.tenant_id.as_str())
// SQL: WHERE author_tenant_id = $N AND ...
```

For Task 4 itself, **gate the Postgres impl behind an attribute** so unit tests (no DB) keep passing:
- The `postgres` cargo feature already gates this module.
- After Task 8 adds the columns, integration tests under `tests/postgres/` (already feature-gated) cover the full flow.

- [ ] **Step 4.8: Update every call site**

Walk `/tmp/b4-memory-callsites.txt` from Step 4.1. For each, prepend `&TenantScope::default()` (or the appropriate live scope if the caller has user context). Examples:

`crates/heartbit-core/src/agent/runner.rs` — wherever a Memory tool calls `memory.recall(...)`, the runner already carries `audit_user_id` and `audit_tenant_id`. Build a scope:
```rust
let scope = match &self.audit_tenant_id {
    Some(t) => TenantScope::new(t).with_user(self.audit_user_id.clone().unwrap_or_default()),
    None => TenantScope::default(),
};
memory.recall(&scope, query).await
```

`crates/heartbit/src/channel/telegram/*.rs` — Telegram derives a scope from the chat id. Use `TenantScope::new(format!("tg:{chat_id}"))` to match the existing prefix-based isolation that NamespacedMemory used.

`crates/heartbit/src/daemon/dispatch.rs` (or wherever `DaemonCommand::SubmitTask` lands) — build scope from `UserContext`:
```rust
let scope = TenantScope::from(&user_ctx);
memory.recall(&scope, query).await
```

Run tests file-by-file after each batch of edits:
```bash
cargo test -p heartbit-core --no-run 2>&1 | tail -20
cargo test -p heartbit --no-run 2>&1 | tail -20
```
Expected: compiles cleanly; remaining errors are call sites still on the old signature — fix them.

- [ ] **Step 4.9: Run isolation tests**

```bash
cargo test -p heartbit-core memory::tenant_isolation 2>&1 | tail -10
```
Expected: 3 tests pass.

- [ ] **Step 4.10: Full quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --lib
```
Expected: clean. If clippy flags new warnings (most likely: unused `scope` arg in default trait impls — silence with `_scope` prefix), fix in this commit.

- [ ] **Step 4.11: Commit**

```bash
git add crates/heartbit-core/src/memory/ \
        crates/heartbit-core/src/agent/runner.rs \
        crates/heartbit-core/src/agent/orchestrator.rs \
        crates/heartbit/src/memory/postgres.rs \
        crates/heartbit/src/channel/ \
        crates/heartbit/src/daemon/
git commit -m "refactor: Memory trait requires &TenantScope on every method

Tenant scoping is now a load-bearing parameter, not a NamespacedMemory
convention. InMemoryStore filters on author_tenant_id; NamespacedMemory
forwards scope through and keeps prefix for intra-tenant namespacing;
PostgresMemoryStore writes new columns (Task 8 migration adds them).
Call sites without explicit scope use TenantScope::default()."
```

---

## Task 5: SandboxPolicy composes CorePathPolicy

**Files:**
- Modify: `crates/heartbit/src/sandbox.rs`

- [ ] **Step 5.1: Write the failing test**

Append to `crates/heartbit/src/sandbox.rs` test module:

```rust
#[test]
fn sandbox_policy_exposes_inner_path_policy() {
    use heartbit_core::sandbox::CorePathPolicy;
    let path = std::sync::Arc::new(
        CorePathPolicy::builder()
            .allow_dir(std::env::temp_dir())
            .build()
            .unwrap(),
    );
    let sandbox = SandboxPolicy::from_path_policy(path.clone());
    assert!(std::sync::Arc::ptr_eq(&path, &sandbox.path_policy()));
}
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
cargo test -p heartbit sandbox::tests::sandbox_policy_exposes_inner_path_policy 2>&1 | tail -10
```
Expected: compile error — `from_path_policy` / `path_policy` methods don't exist.

- [ ] **Step 5.3: Compose**

Edit `crates/heartbit/src/sandbox.rs`. Add a field and methods:

```rust
use std::sync::Arc;
use heartbit_core::sandbox::CorePathPolicy;

pub struct SandboxPolicy {
    path_policy: Arc<CorePathPolicy>,
    // ... existing landlock-specific fields ...
}

impl SandboxPolicy {
    /// Build from an externally-constructed `CorePathPolicy`. Used when one
    /// path policy is shared across multiple tools (bash + write + edit + ...).
    pub fn from_path_policy(path_policy: Arc<CorePathPolicy>) -> Self {
        Self {
            path_policy,
            // ... default the rest of existing fields ...
        }
    }

    /// Expose the inner CorePathPolicy so callers can pass it to non-bash
    /// filesystem tools that take `Arc<CorePathPolicy>`.
    pub fn path_policy(&self) -> Arc<CorePathPolicy> {
        self.path_policy.clone()
    }
}
```

Update the existing `SandboxPolicyBuilder::build()` to construct a `CorePathPolicy` from its allow/deny configuration and store it as the shared inner type. The existing landlock setup code on Linux continues to reference `path_policy.allowed_dirs` / `path_policy.deny_globs` accessors (add them as `pub(crate)` if not already accessible).

- [ ] **Step 5.4: Run test to verify it passes**

```bash
cargo test -p heartbit sandbox 2>&1 | tail -10
```
Expected: PASS, plus all existing sandbox tests still pass.

- [ ] **Step 5.5: Commit**

```bash
git add crates/heartbit/src/sandbox.rs
git commit -m "refactor: SandboxPolicy composes Arc<CorePathPolicy>

Composition lets one path policy back both the bash sandbox (with
landlock syscall enforcement) and the other filesystem tools
(application-layer enforcement only)."
```

---

## Task 6: with_path_policy on filesystem builtins

**Files:**
- Modify: `crates/heartbit-core/src/tool/builtins/bash.rs`
- Modify: `crates/heartbit-core/src/tool/builtins/patch.rs`
- Modify: `crates/heartbit-core/src/tool/builtins/edit.rs`
- Modify: `crates/heartbit-core/src/tool/builtins/write.rs`
- Modify: `crates/heartbit-core/src/tool/builtins/read.rs`

The change is mechanical and identical across all five tools: add an `Option<Arc<CorePathPolicy>>` field, a `with_path_policy(...)` builder method, and a `check_path` call in `execute` before any I/O.

- [ ] **Step 6.1: Write a failing test for `WriteTool`**

Append to `crates/heartbit-core/src/tool/builtins/write.rs` test module:

```rust
#[tokio::test]
async fn write_tool_rejects_path_outside_policy() {
    use crate::sandbox::CorePathPolicy;
    use std::sync::Arc;

    let workspace = std::env::temp_dir().join(format!("hb-write-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&workspace).unwrap();
    let policy = Arc::new(CorePathPolicy::builder().allow_dir(&workspace).build().unwrap());

    let tool = WriteTool::new().with_path_policy(policy);

    let outside = std::env::temp_dir().join("hb-outside.txt");
    let result = tool
        .execute(serde_json::json!({"file_path": outside, "content": "x"}))
        .await
        .unwrap();
    assert!(result.is_error, "expected sandbox violation");
    assert!(result.content.contains("not under any allowed directory"));
}

#[tokio::test]
async fn write_tool_allows_path_inside_policy() {
    use crate::sandbox::CorePathPolicy;
    use std::sync::Arc;
    let workspace = std::env::temp_dir().join(format!("hb-write-ok-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&workspace).unwrap();
    let policy = Arc::new(CorePathPolicy::builder().allow_dir(&workspace).build().unwrap());

    let tool = WriteTool::new().with_path_policy(policy);
    let ok = workspace.join("ok.txt");
    let result = tool
        .execute(serde_json::json!({"file_path": ok, "content": "x"}))
        .await
        .unwrap();
    assert!(!result.is_error);
}
```

- [ ] **Step 6.2: Run test to verify it fails**

```bash
cargo test -p heartbit-core tool::builtins::write 2>&1 | tail -10
```
Expected: compile error — `with_path_policy` does not exist.

- [ ] **Step 6.3: Implement on `WriteTool`**

Edit `crates/heartbit-core/src/tool/builtins/write.rs`. Add the field and builder:

```rust
use std::sync::Arc;
use crate::sandbox::CorePathPolicy;

pub struct WriteTool {
    // ... existing fields ...
    path_policy: Option<Arc<CorePathPolicy>>,
}

impl WriteTool {
    pub fn new() -> Self {
        Self {
            // ... existing defaults ...
            path_policy: None,
        }
    }

    pub fn with_path_policy(mut self, policy: Arc<CorePathPolicy>) -> Self {
        self.path_policy = Some(policy);
        self
    }
}
```

In `execute`, before any I/O:
```rust
fn execute(&self, input: serde_json::Value) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
    let path_policy = self.path_policy.clone();
    let tracker = self.tracker.clone();
    Box::pin(async move {
        // ... parse args into `path` and `content` ...
        if let Some(p) = &path_policy {
            if let Err(e) = p.check_path(&path) {
                return Ok(ToolOutput::error(e.to_string()));
            }
        }
        // ... existing FileTracker mtime guard ...
        // ... existing UTF-8 boundary guard ...
        // ... existing write logic ...
    })
}
```

- [ ] **Step 6.4: Run test to verify it passes**

```bash
cargo test -p heartbit-core tool::builtins::write::tests 2>&1 | tail -10
```
Expected: 2 new tests pass; existing write tests still pass.

- [ ] **Step 6.5: Repeat for `EditTool`, `PatchTool`, `ReadTool`**

Apply the identical pattern to `edit.rs`, `patch.rs`, `read.rs`. For each tool, add the same two tests (renamed with the tool's name) before adding the field and check.

For `BashTool`: the existing `with_sandbox_policy(Arc<SandboxPolicy>)` already guards via landlock on Linux. Add `with_path_policy(Arc<CorePathPolicy>)` as well — bash callers can either pass a full SandboxPolicy or just a path policy. When `with_sandbox_policy` is set, path enforcement comes from the sandbox; when only `with_path_policy` is set, do an application-layer `check_path` on `working_dir` before spawning.

After each tool, run its tests:
```bash
cargo test -p heartbit-core tool::builtins::edit::tests 2>&1 | tail -5
cargo test -p heartbit-core tool::builtins::patch::tests 2>&1 | tail -5
cargo test -p heartbit-core tool::builtins::read::tests 2>&1 | tail -5
cargo test -p heartbit-core tool::builtins::bash::tests 2>&1 | tail -5
```
Each: PASS.

- [ ] **Step 6.6: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --lib
```
Expected: clean.

- [ ] **Step 6.7: Commit**

```bash
git add crates/heartbit-core/src/tool/builtins/{bash,patch,edit,write,read}.rs
git commit -m "feat(core): with_path_policy on bash/patch/edit/write/read

Each filesystem builtin now accepts Option<Arc<CorePathPolicy>> at
construction and rejects denied paths in execute() before any I/O.
Bash retains its existing with_sandbox_policy for landlock-backed
syscall enforcement; the two are complementary."
```

---

## Task 7: AuditTrail trait scoped reads + prune

**Files:**
- Modify: `crates/heartbit-core/src/agent/audit.rs`
- Modify: `crates/heartbit/src/store/postgres.rs` (PostgresAuditTrail or analogous)
- Modify: `crates/heartbit/src/daemon/mod.rs` (background prune task) — exact filename varies; grep for `DaemonCore::run_kafka` / `run_channel`.

- [ ] **Step 7.1: Write the failing test**

Append to `crates/heartbit-core/src/agent/audit.rs` test module:

```rust
#[tokio::test]
async fn entries_filters_by_scope() {
    use crate::auth::TenantScope;
    let trail = InMemoryAuditTrail::new();
    let acme = TenantScope::new("acme");
    let globex = TenantScope::new("globex");

    trail
        .record(AuditRecord {
            agent: "a".into(), turn: 0, event_type: "x".into(),
            payload: serde_json::Value::Null, usage: TokenUsage::default(),
            timestamp: chrono::Utc::now(),
            user_id: None, tenant_id: Some("acme".into()), delegation_chain: vec![],
        })
        .await
        .unwrap();
    trail
        .record(AuditRecord {
            agent: "a".into(), turn: 0, event_type: "x".into(),
            payload: serde_json::Value::Null, usage: TokenUsage::default(),
            timestamp: chrono::Utc::now(),
            user_id: None, tenant_id: Some("globex".into()), delegation_chain: vec![],
        })
        .await
        .unwrap();

    let acme_rows = trail.entries(&acme, 100).await.unwrap();
    assert_eq!(acme_rows.len(), 1);
    assert_eq!(acme_rows[0].tenant_id.as_deref(), Some("acme"));

    let unscoped = trail.entries_unscoped(100).await.unwrap();
    assert_eq!(unscoped.len(), 2);
}

#[tokio::test]
async fn prune_deletes_old_entries() {
    let trail = InMemoryAuditTrail::new();
    let now = chrono::Utc::now();
    trail
        .record(AuditRecord {
            agent: "a".into(), turn: 0, event_type: "old".into(),
            payload: serde_json::Value::Null, usage: TokenUsage::default(),
            timestamp: now - chrono::Duration::days(10),
            user_id: None, tenant_id: None, delegation_chain: vec![],
        })
        .await
        .unwrap();
    trail
        .record(AuditRecord {
            agent: "a".into(), turn: 0, event_type: "new".into(),
            payload: serde_json::Value::Null, usage: TokenUsage::default(),
            timestamp: now,
            user_id: None, tenant_id: None, delegation_chain: vec![],
        })
        .await
        .unwrap();

    let removed = trail.prune(chrono::Duration::days(7)).await.unwrap();
    assert_eq!(removed, 1);

    let rest = trail.entries_unscoped(100).await.unwrap();
    assert_eq!(rest.len(), 1);
    assert_eq!(rest[0].event_type, "new");
}
```

- [ ] **Step 7.2: Run test to verify it fails**

```bash
cargo test -p heartbit-core agent::audit 2>&1 | tail -15
```
Expected: compile error — methods don't exist with those signatures.

- [ ] **Step 7.3: Update the trait**

Edit `crates/heartbit-core/src/agent/audit.rs`. Replace the trait (around line 76) with:

```rust
use crate::auth::TenantScope;

pub trait AuditTrail: Send + Sync {
    fn record(
        &self,
        entry: AuditRecord,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    /// Tenant-scoped read.
    fn entries(
        &self,
        scope: &TenantScope,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Cross-tenant admin read. Renamed from the previous unscoped
    /// `entries()` so callers must explicitly opt into cross-tenant
    /// visibility.
    fn entries_unscoped(
        &self,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Time-windowed scoped read.
    fn entries_since(
        &self,
        scope: &TenantScope,
        since: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Delete entries older than `now - retain`. Returns count deleted.
    fn prune(
        &self,
        retain: chrono::Duration,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>>;

    /// GDPR right-to-erasure. Default no-op.
    fn erase_for_user(
        &self,
        _user_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async { Ok(0) })
    }
}
```

- [ ] **Step 7.4: Update `InMemoryAuditTrail`**

Edit the impl below the trait. Replace the existing `entries()` and `entries_for_tenant(...)` methods:

```rust
impl AuditTrail for InMemoryAuditTrail {
    fn record(&self, entry: AuditRecord) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        let records = self.records.clone();
        Box::pin(async move {
            records
                .write()
                .map_err(|_| Error::AuditTrail("poisoned".into()))?
                .push(entry);
            Ok(())
        })
    }

    fn entries(&self, scope: &TenantScope, limit: usize) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>> {
        let records = self.records.clone();
        let tenant = scope.tenant_id.clone();
        Box::pin(async move {
            let guard = records.read().map_err(|_| Error::AuditTrail("poisoned".into()))?;
            let out: Vec<AuditRecord> = guard
                .iter()
                .filter(|r| r.tenant_id.as_deref().unwrap_or("") == tenant)
                .rev()
                .take(limit)
                .cloned()
                .collect();
            Ok(out.into_iter().rev().collect())
        })
    }

    fn entries_unscoped(&self, limit: usize) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>> {
        let records = self.records.clone();
        Box::pin(async move {
            let guard = records.read().map_err(|_| Error::AuditTrail("poisoned".into()))?;
            let out: Vec<AuditRecord> = guard
                .iter()
                .rev()
                .take(limit)
                .cloned()
                .collect();
            Ok(out.into_iter().rev().collect())
        })
    }

    fn entries_since(&self, scope: &TenantScope, since: chrono::DateTime<chrono::Utc>, limit: usize) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>> {
        let records = self.records.clone();
        let tenant = scope.tenant_id.clone();
        Box::pin(async move {
            let guard = records.read().map_err(|_| Error::AuditTrail("poisoned".into()))?;
            let out: Vec<AuditRecord> = guard
                .iter()
                .filter(|r| {
                    r.tenant_id.as_deref().unwrap_or("") == tenant && r.timestamp >= since
                })
                .rev()
                .take(limit)
                .cloned()
                .collect();
            Ok(out.into_iter().rev().collect())
        })
    }

    fn prune(&self, retain: chrono::Duration) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        let records = self.records.clone();
        Box::pin(async move {
            let cutoff = chrono::Utc::now() - retain;
            let mut guard = records.write().map_err(|_| Error::AuditTrail("poisoned".into()))?;
            let before = guard.len();
            guard.retain(|r| r.timestamp >= cutoff);
            Ok(before - guard.len())
        })
    }
}
```

(Drop the old `entries_for_tenant`. Keep `erase_for_user` if it has a non-default implementation; otherwise the trait default applies.)

- [ ] **Step 7.5: Update every AuditTrail call site**

```bash
grep -rn 'audit.entries(\|audit.entries_for_tenant\|\.entries()\.await' crates/ tests/ 2>/dev/null
```

For each result:
- `.entries()` (no args) → `.entries_unscoped(N)` with an appropriate limit (e.g., 1000 for admin views; existing call sites that did `.entries()` for tests can use a generous limit).
- `.entries_for_tenant(Some(tid))` → `.entries(&TenantScope::new(tid), N)`.
- `.entries_for_tenant(None)` → `.entries_unscoped(N)`.

Run `cargo build --workspace` between batches to catch missed call sites.

- [ ] **Step 7.6: Add the daemon background prune task**

In whichever file holds `DaemonCore::run_kafka` / `run_channel`, add at the start of the run method:

```rust
if let Some(retain) = self.config.audit.retain_days.map(|d| chrono::Duration::days(d as i64)) {
    let trail = self.audit_trail.clone();
    let interval = std::time::Duration::from_secs(
        self.config.audit.prune_interval_minutes.unwrap_or(60) * 60,
    );
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        loop {
            tick.tick().await;
            if let Err(e) = trail.prune(retain).await {
                tracing::warn!(error = %e, "audit prune failed");
            }
        }
    });
}
```

(Add `audit: DaemonAuditConfig` to `DaemonConfig` and the corresponding fields to `DaemonAuditConfig` if not present. See Task 9 for the user-facing TOML.)

- [ ] **Step 7.7: Run tests**

```bash
cargo test -p heartbit-core agent::audit 2>&1 | tail -10
cargo test --workspace --lib 2>&1 | tail -10
```
Expected: 2 new tests pass; full lib suite green.

- [ ] **Step 7.8: Commit**

```bash
git add crates/heartbit-core/src/agent/audit.rs \
        crates/heartbit/src/daemon/ \
        crates/heartbit/src/store/postgres.rs
git commit -m "refactor: AuditTrail scoped reads + retention prune

Renames entries() → entries_unscoped() and entries_for_tenant(Option)
→ entries(&TenantScope) (typed). Adds entries_since() and prune().
Daemon spawns a background prune task when retain_days is set."
```

---

## Task 8: Postgres tenant columns + indexes

**Files:**
- Modify: `crates/heartbit/src/memory/postgres.rs` (memories table migration + binds)
- Modify: `crates/heartbit/src/store/postgres.rs` (audit_log table migration + binds)
- Test: `crates/heartbit/tests/postgres_tenant_columns.rs` (feature-gated integration test)

- [ ] **Step 8.1: Add the migrations**

Edit `crates/heartbit/src/memory/postgres.rs::run_migration`. Append to the `statements` array (line 175):

```rust
"ALTER TABLE memories ADD COLUMN IF NOT EXISTS author_tenant_id TEXT NOT NULL DEFAULT ''",
"ALTER TABLE memories ADD COLUMN IF NOT EXISTS author_user_id TEXT",
"CREATE INDEX IF NOT EXISTS idx_memories_tenant ON memories(author_tenant_id, agent)",
```

Edit `crates/heartbit/src/store/postgres.rs::run_migration`. Append to the statements array (line 89):

```rust
"ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''",
"ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS user_id TEXT",
"CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, created_at DESC)",
```

- [ ] **Step 8.2: Update INSERT/SELECT to bind the new columns**

In `memory/postgres.rs::store`, change the INSERT statement to include the two new columns and bind from `entry.author_tenant_id` / `entry.author_user_id`. In `recall`, add `WHERE author_tenant_id = $N` and select the columns into the row. In the `From<PgMemoryRow> for MemoryEntry` (or equivalent struct mapper) populate the new fields instead of always-`None`.

In `store/postgres.rs::write_audit`, take a `record: &AuditRecord` (or extend the function signature to accept tenant_id and user_id), bind them. In `audit_entries` queries, add the appropriate `WHERE tenant_id = $N` filter for scoped reads and select the new column for unmapped reads.

- [ ] **Step 8.3: Write the failing integration test**

Create `crates/heartbit/tests/postgres_tenant_columns.rs`:

```rust
//! Postgres integration test: tenant columns persist and isolate.
//!
//! Requires DATABASE_URL env var. Skipped via #[ignore] otherwise.

#![cfg(feature = "postgres")]

use heartbit::auth::TenantScope;
use heartbit::memory::{Memory, MemoryEntry, MemoryQuery, PostgresMemoryStore};

#[tokio::test]
#[ignore = "requires DATABASE_URL"]
async fn memory_tenant_columns_isolate_recall() {
    let url = std::env::var("DATABASE_URL").unwrap();
    let mut store = PostgresMemoryStore::connect(&url).await.unwrap();
    store.run_migration().await.unwrap();

    let acme = TenantScope::new("acme");
    let globex = TenantScope::new("globex");
    let entry_a = MemoryEntry { id: format!("a-{}", uuid::Uuid::new_v4()), ..MemoryEntry::default() };
    let entry_g = MemoryEntry { id: format!("g-{}", uuid::Uuid::new_v4()), ..MemoryEntry::default() };

    store.store(&acme, entry_a.clone()).await.unwrap();
    store.store(&globex, entry_g.clone()).await.unwrap();

    let mut q = MemoryQuery::default();
    q.agent = Some("a".into());
    let acme_rows = store.recall(&acme, q.clone()).await.unwrap();
    assert!(acme_rows.iter().any(|r| r.id == entry_a.id));
    assert!(!acme_rows.iter().any(|r| r.id == entry_g.id));
}
```

- [ ] **Step 8.4: Run with a live DB**

```bash
DATABASE_URL=postgres://... cargo test -p heartbit --test postgres_tenant_columns -- --ignored 2>&1 | tail -10
```
Expected: PASS. If no Postgres is available, skip this step locally and rely on CI's Postgres job; mention this in the commit body.

- [ ] **Step 8.5: Commit**

```bash
git add crates/heartbit/src/memory/postgres.rs \
        crates/heartbit/src/store/postgres.rs \
        crates/heartbit/tests/postgres_tenant_columns.rs
git commit -m "feat(postgres): add tenant_id/author_tenant_id columns + indexes

Idempotent ADD COLUMN IF NOT EXISTS migrations; default '' matches
TenantScope::single_tenant(). PostgresMemoryStore + PostgresAuditTrail
now persist and filter on tenant. Integration test gated on
DATABASE_URL via #[ignore]."
```

---

## Task 9: CLI / config wiring

**Files:**
- Modify: `crates/heartbit-core/src/types.rs` (or wherever `AgentConfig` / `OrchestratorConfig` live — grep `pub struct AgentConfig`)
- Modify: `crates/heartbit/src/daemon/types.rs` or `daemon/config.rs` for `DaemonAuditConfig`
- Modify: `crates/heartbit-cli/src/main.rs` (or `run.rs` / `serve.rs` — wherever the CLI builds the runner)

- [ ] **Step 9.1: Add config fields**

In `AgentConfig`:
```rust
#[serde(default)]
pub max_tool_calls_per_turn: Option<u32>,
```

In `OrchestratorConfig`:
```rust
#[serde(default)]
pub max_tool_calls_per_turn: Option<u32>,
```

New `SandboxConfig`:
```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SandboxConfig {
    #[serde(default)]
    pub allowed_dirs: Vec<std::path::PathBuf>,
    #[serde(default)]
    pub deny_globs: Vec<String>,
}
```

New `DaemonAuditConfig`:
```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DaemonAuditConfig {
    #[serde(default)]
    pub retain_days: Option<u32>,
    #[serde(default)]
    pub prune_interval_minutes: Option<u64>,
}
```

Add `sandbox: Option<SandboxConfig>` to `HeartbitConfig`. Add `audit: DaemonAuditConfig` to `DaemonConfig`.

- [ ] **Step 9.2: Wire env-var overrides**

In CLI provider/runner construction, after loading the TOML config, apply:
```rust
if let Ok(s) = std::env::var("HEARTBIT_MAX_TOOL_CALLS_PER_TURN") {
    if let Ok(v) = s.parse::<u32>() {
        cfg.orchestrator.max_tool_calls_per_turn = Some(v);
    }
}
if let Ok(s) = std::env::var("HEARTBIT_AUDIT_RETAIN_DAYS") {
    if let Ok(v) = s.parse::<u32>() {
        daemon_cfg.audit.retain_days = Some(v);
    }
}
```

- [ ] **Step 9.3: Wire the runner builder**

In whichever CLI helper builds `AgentRunnerBuilder`:
```rust
if let Some(cap) = agent_cfg.max_tool_calls_per_turn.or(orch_cfg.max_tool_calls_per_turn) {
    builder = builder.max_tool_calls_per_turn(cap);
}
```

In sandbox construction:
```rust
let core_path_policy = if let Some(sb) = &cfg.sandbox {
    let mut b = heartbit_core::CorePathPolicy::builder();
    for d in &sb.allowed_dirs { b = b.allow_dir(d); }
    for g in &sb.deny_globs { b = b.deny_glob(g); }
    Some(std::sync::Arc::new(b.build()?))
} else {
    None
};

if let Some(p) = &core_path_policy {
    write_tool = write_tool.with_path_policy(p.clone());
    edit_tool = edit_tool.with_path_policy(p.clone());
    patch_tool = patch_tool.with_path_policy(p.clone());
    read_tool = read_tool.with_path_policy(p.clone());
    bash_tool = bash_tool.with_sandbox_policy(std::sync::Arc::new(
        heartbit::sandbox::SandboxPolicy::from_path_policy(p.clone())
    ));
}
```

- [ ] **Step 9.4: Add a config validation test**

Append to `crates/heartbit-core/src/config_tests.rs` (or wherever the existing config tests live):

```rust
#[test]
fn max_tool_calls_per_turn_zero_rejected() {
    let toml = r#"
[provider]
kind = "anthropic"
model = "claude-3-5-sonnet-20241022"

[orchestrator]
max_tool_calls_per_turn = 0
"#;
    let cfg: Result<HeartbitConfig, _> = toml::from_str(toml);
    let cfg = cfg.unwrap();
    let err = cfg.validate().unwrap_err();
    assert!(err.to_string().contains("max_tool_calls_per_turn"));
}
```
Add the corresponding check inside `HeartbitConfig::validate`:
```rust
if let Some(0) = self.orchestrator.max_tool_calls_per_turn {
    return Err(Error::Config("max_tool_calls_per_turn must be > 0 if set".into()));
}
for a in &self.agents {
    if let Some(0) = a.max_tool_calls_per_turn {
        return Err(Error::Config(format!(
            "agent {}: max_tool_calls_per_turn must be > 0 if set", a.name
        )));
    }
}
```

- [ ] **Step 9.5: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --lib
```
Expected: clean.

- [ ] **Step 9.6: Commit**

```bash
git add crates/heartbit-core/src/types.rs \
        crates/heartbit/src/daemon/ \
        crates/heartbit-cli/src/
git commit -m "feat(cli): wire max_tool_calls_per_turn + sandbox + audit retention

TOML fields: orchestrator.max_tool_calls_per_turn,
[sandbox]allowed_dirs/deny_globs, [daemon.audit]retain_days /
prune_interval_minutes. Env vars: HEARTBIT_MAX_TOOL_CALLS_PER_TURN,
HEARTBIT_AUDIT_RETAIN_DAYS. Validates zero at parse time."
```

---

## Task 10: Docs + CHANGELOG

**Files:**
- Create: `book/src/recipes/multi-tenant.md`
- Modify: `book/src/SUMMARY.md`
- Modify: `CHANGELOG.md`
- Modify: `crates/heartbit-core/README.md` (brief multi-tenant section)
- Modify: `crates/heartbit-core/src/lib.rs` (rustdoc on `TenantScope`)

- [ ] **Step 10.1: Write the multi-tenant recipe chapter**

Create `book/src/recipes/multi-tenant.md`:

```markdown
# Multi-Tenant Hardening

Heartbit's multi-tenant story is grounded in three load-bearing types:

- `TenantScope` — required parameter on every `Memory` and `AuditTrail` method.
- `CorePathPolicy` — shared path allowlist/denylist for all filesystem builtins.
- `max_tool_calls_per_turn` — caps tool dispatches per LLM turn.

## TenantScope

```rust
use heartbit::TenantScope;

let scope = TenantScope::new("acme-corp")
    .with_user("user-42");
memory.recall(&scope, query).await?;
```

Empty-string tenant id is the single-tenant sentinel:

```rust
let scope = TenantScope::default();      // single-tenant
assert!(scope.is_single_tenant());
```

In daemon mode the scope is built from JWT claims via
`From<&UserContext>`:

```rust
let scope: TenantScope = (&user_context).into();
```

## CorePathPolicy

```rust
use heartbit::CorePathPolicy;
use std::sync::Arc;

let policy = Arc::new(
    CorePathPolicy::builder()
        .allow_dir("/workspace")
        .deny_glob("**/.env")
        .deny_glob("**/secrets/**")
        .build()?
);

let write_tool = WriteTool::new().with_path_policy(policy.clone());
let edit_tool  = EditTool::new().with_path_policy(policy.clone());
let read_tool  = ReadTool::new().with_path_policy(policy);
```

For bash, compose a full SandboxPolicy:

```rust
let sandbox = SandboxPolicy::from_path_policy(policy.clone());
let bash = BashTool::new().with_sandbox_policy(Arc::new(sandbox));
```

## max_tool_calls_per_turn

```rust
let runner = AgentRunnerBuilder::new()
    .max_tool_calls_per_turn(8)  // production recommendation
    .build()?;
```

Distinct from `max_tools_per_turn` (which limits the *tool definition
set* offered to the LLM). This caps the *invocations* the LLM produces
per turn. Excess returns `Error::Agent` with partial usage attached.

## Postgres tenant columns

Run `PostgresMemoryStore::run_migration` once on upgrade. It adds the
`author_tenant_id` and `author_user_id` columns to `memories` and the
analogous columns to `audit_log`. Default value is `''`, matching
`TenantScope::single_tenant()` — single-tenant deployments are
unaffected.

## Audit retention

```toml
[daemon.audit]
retain_days = 90
prune_interval_minutes = 60
```

The daemon spawns a background task that calls `audit.prune(retain)`
on the configured interval.
```

- [ ] **Step 10.2: Add to SUMMARY**

Edit `book/src/SUMMARY.md`. Under the Recipes section, add:
```markdown
- [Multi-tenant hardening](recipes/multi-tenant.md)
```

- [ ] **Step 10.3: CHANGELOG entry**

Edit `CHANGELOG.md`. At the top of the unreleased / next-version section:

```markdown
### Added
- `heartbit_core::auth::TenantScope` — owned tenant + user identity, required by Memory and AuditTrail.
- `heartbit_core::sandbox::CorePathPolicy` — path allowlist + glob denylist shared across filesystem builtins.
- `with_path_policy(Arc<CorePathPolicy>)` on `BashTool`, `PatchTool`, `EditTool`, `WriteTool`, `ReadTool`.
- `AgentRunnerBuilder::max_tool_calls_per_turn(u32)` — caps dispatched tool calls per LLM turn.
- `AuditTrail::entries_since`, `AuditTrail::prune` — windowed reads + retention.
- `[daemon.audit]` config: `retain_days`, `prune_interval_minutes`.
- `[sandbox]` config: `allowed_dirs`, `deny_globs`.
- Multi-tenant recipe chapter in the user docs.

### Changed (breaking; pre-release)
- `Memory` trait now requires `&TenantScope` as the first parameter on every method. Migrate by passing `TenantScope::default()` at single-tenant call sites.
- `AuditTrail::entries()` (no args) renamed to `entries_unscoped(limit)`.
- `AuditTrail::entries_for_tenant(Option<&str>)` replaced by `entries(&TenantScope, limit)`.
- Postgres schema gains `author_tenant_id` (memories) and `tenant_id` (audit_log) columns. `run_migration` is idempotent and safe to re-run.
```

- [ ] **Step 10.4: Build the book**

```bash
cargo install mdbook --version "^0.5" 2>/dev/null
cd book && mdbook build 2>&1 | tail -5
```
Expected: `Book built successfully` and the new chapter appears in `book/book/recipes/multi-tenant.html`.

- [ ] **Step 10.5: Commit**

```bash
git add book/src/recipes/multi-tenant.md book/src/SUMMARY.md CHANGELOG.md \
        crates/heartbit-core/README.md crates/heartbit-core/src/lib.rs
git commit -m "docs: B4 multi-tenant chapter + CHANGELOG breaking-change notes

User-facing recipe with three load-bearing types (TenantScope,
CorePathPolicy, max_tool_calls_per_turn), Postgres migration note,
audit retention config. CHANGELOG documents the pre-release breaking
changes for Memory + AuditTrail."
```

---

## Final verification

- [ ] **Step F.1: Full quality gate**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
All three must pass.

- [ ] **Step F.2: Spec coverage check**

Open `docs/superpowers/specs/2026-05-01-b4-multi-tenant-hardening-design.md` and walk each Goal (1-6). Confirm each has at least one task implementing it. Note any gaps in this checklist:

- Goal 1 (max_tool_calls_per_turn) — Task 3 ✓
- Goal 2 (filesystem builtins enforce path policy) — Tasks 2 + 5 + 6 ✓
- Goal 3 (TenantScope load-bearing) — Tasks 1 + 4 + 7 ✓
- Goal 4 (Postgres NOT NULL tenant) — Task 8 ✓
- Goal 5 (audit retention + scoped reads) — Task 7 ✓
- Goal 6 (no regressions for non-tenancy users) — covered by `TenantScope::default()` migration in Task 4 + the `Option<Arc<CorePathPolicy>>` opt-in in Task 6.

- [ ] **Step F.3: Hand off to finishing-a-development-branch**

Use `superpowers:finishing-a-development-branch` skill to merge the worktree to main and clean up.

---

## Self-review notes

**1. Spec coverage** — every Goal maps to at least one task; final-verification step F.2 is the explicit cross-reference.

**2. No placeholders** — every step shows actual code or actual commands. Step 4.7 (Postgres recall) is the closest to vague (uses `... existing INSERT, with two more bound params ...`); the actual SQL string is intentionally not duplicated because the existing INSERT in `memory/postgres.rs` is large and the implementer needs to read the surrounding context to integrate cleanly. Acceptable trade-off for this one step.

**3. Type consistency** — `TenantScope.tenant_id: String` (Task 1) is reused unchanged in Tasks 4, 7, 8, 9. `CorePathPolicy::check_path(&Path) -> Result<(), Error>` (Task 2) is reused in Tasks 6, 5, 9. `max_tool_calls_per_turn: Option<u32>` (Task 3) is reused in Task 9 config wiring. `From<&UserContext>` (Task 1.7) is reused in any daemon call site that builds a scope from a JWT.

**4. Sequencing dependencies** — Task 1 → Task 4, 7 (TenantScope used). Task 2 → Task 5, 6 (CorePathPolicy used). Task 5 must come before Task 6's bash test (which composes through SandboxPolicy). Task 8's Postgres migration is independent of Task 4's trait change; Task 4 binds the new fields to a *not-yet-migrated* schema, which works because `INSERT INTO memories (..., author_tenant_id, ...)` simply fails until Task 8 lands. Order Task 4 → Task 8 → integration test, OR run Task 8 first and Task 4 second. The plan above runs Task 4 before Task 8 because Task 4's compile is the larger surface area; the integration test in Task 8.3 is the gate that catches any binding mismatch.
