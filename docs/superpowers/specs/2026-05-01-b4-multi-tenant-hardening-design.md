# B4 — Multi-Tenant Hardening: Tool Budgets, Sandbox Enforcement, Tenant Scope, Audit Retention

**Date:** 2026-05-01
**Status:** Design — pending user approval before implementation plan
**Scope:** `crates/heartbit-core` (primary), `crates/heartbit` (umbrella, audit + sandbox composition), `crates/heartbit-cli` (carry-fix only)
**Estimated effort:** ~5–7 working days, executed as ~10 small independently-green commits.
**Public API breakage:** Pre-release breaking changes only. `heartbit-core` is not yet on crates.io. The umbrella `heartbit` is published, but the changed surface (`AuditTrail`, `Memory`, `NamespacedMemory`) was added during the B3 round and has had no external users. Documented in CHANGELOG.

## Background

The 2026-04-30 production-readiness review (5 parallel reviewers) graded the standalone library as **production-grade** but identified five gaps that block public release of the multi-tenant runtime path. Three are critical (data isolation / runaway compute / audit), two are important (audit retention / tenant-scoped APIs):

1. **No per-turn / per-task tool-call cap.** A single LLM turn can return arbitrarily many `tool_use` blocks; `AgentRunner::run` will dispatch them all into a `tokio::JoinSet`. A misbehaving model (or a prompt-injection attack) can spawn hundreds of bash/web-fetch tool calls in one turn and saturate per-task quotas before the existing `max_turns` budget triggers. The `DoomLoopTracker` only catches *identical* repeated calls.
2. **Sandbox path policy is bash-only.** `crates/heartbit/src/sandbox/landlock.rs` integrates with the bash builtin. The other filesystem-touching builtins (`patch`, `edit`, `write`, `read`) check only the `FileTracker` mtime guard and `floor_char_boundary` UTF-8 guard. An agent can write to `/etc/passwd` via `write` even when the bash sandbox would reject it.
3. **Tenant scoping is convention, not type.** `NamespacedMemory` wraps a `Memory` impl with a tenant prefix string. There is no compile-time guarantee that a memory query carries the right tenant — calling sites pass `&str` or `Option<String>` and an empty/wrong value silently scopes to the global namespace. The audit trail (`AuditTrail::entries(limit)`) returns *all* entries with no tenant filter.
4. **Postgres `tenant_id` columns are nullable with no constraint.** `crates/heartbit/src/store/postgres/{memory,audit}.rs` schemas allow `NULL` for `tenant_id`. A bug that drops the tenant scope writes a row visible cross-tenant. There is no `NOT NULL` constraint and no row-level isolation.
5. **Audit trail has no retention or unscoped reads.** `AuditTrail::entries(limit)` is unscoped. There is no time-window query, no "entries-for-this-tenant-only" path, no archival or deletion. Long-running deployments grow audit storage unbounded.

These five issues share a single architectural root cause: the multi-tenant story (`tenant_id` on `MemoryEntry`, `audit_user_id`/`audit_tenant_id` in agent prompts, `NamespacedMemory`, `shared_write_roles`) was added incrementally without a single load-bearing tenant-scope type. B4 introduces that type (`TenantScope`), converts memory + audit APIs to require it, adds `CorePathPolicy` for path enforcement across all filesystem builtins, and lands the per-turn tool budget.

## Goals

1. **Bound tool-call concurrency per turn.** Add `max_tool_calls_per_turn` to `AgentRunner` (default `None` = unlimited; recommended config `8`). Exceeding the cap returns `Error::Agent` with `WithPartialUsage` so token cost is preserved.
2. **Filesystem builtins enforce path policy uniformly.** Promote sandbox path checking from bash-only to a `CorePathPolicy` type in `heartbit-core`. All five filesystem builtins (`bash`, `patch`, `edit`, `write`, `read`) accept `Option<Arc<CorePathPolicy>>` and reject denied paths before any I/O. Bash retains its existing landlock-backed `SandboxPolicy` for syscall-level enforcement; `SandboxPolicy` composes with `CorePathPolicy`.
3. **`TenantScope` is a load-bearing type.** Introduce `TenantScope { tenant_id: String, user_id: Option<String> }` in `heartbit_core::auth::tenant` (string-typed to match the existing `UserContext.tenant_id` convention — see Component 1). Convert `Memory::{store,recall,update,forget,add_link,prune}`, `NamespacedMemory::*`, and `AuditTrail::entries` to require `&TenantScope`. Compile-time prevents a missing scope.
4. **Postgres rows always carry a non-null tenant.** Migrations add `NOT NULL DEFAULT ''` to the existing `tenant_id text` columns on `audit` and `author_tenant_id text` on `memory`. Existing `NULL` rows are backfilled to the empty-string sentinel (matching `TenantScope::single_tenant()`) so single-tenant deployments are unaffected. New composite indexes on `(tenant_id, created_at DESC)` (audit) and `(author_tenant_id, agent, created_at DESC)` (memory) make the new scoped query shapes efficient.
5. **Audit trail has retention + scoped reads.** `AuditTrail::entries(&TenantScope, limit)` returns rows for that tenant only. New methods: `entries_unscoped(limit)` (admin), `entries_since(&TenantScope, since: DateTime<Utc>, limit)` (windowed), `prune(retain: Duration)` (deletion). InMemory and Postgres impls both.
6. **No regressions to the framework path.** `heartbit-core` library users who never touch tenancy continue to work — `TenantScope::default()` exists, `CorePathPolicy` is opt-in, audit retention is opt-in. Default behavior matches today.

## Non-Goals

- **DNS-rebind defense for `WebFetchTool`.** B2 documented limitation. Still deferred.
- **Failure-mode hardening** — context-overflow accounting, idempotency keys for daemon `SubmitTask`, structured retry policy. Deferred to B5.
- **Row-level security in Postgres** (`policy` / `RLS` on tables). The `NOT NULL` + scoped-read API gives application-layer isolation; database-layer RLS is a B6+ topic that needs a tenancy-claim → `current_setting('app.tenant_id')` adapter and is invasive to test.
- **Audit log signing / tamper-evidence.** Out of scope. Audit storage is honest-but-not-cryptographic.
- **Token rate limiting per tenant.** The existing `BudgetService` (Restate path only) handles per-user budgets. Standalone path gets `max_tool_calls_per_turn`; per-tenant token budgets are a separate round.
- **Migration of `heartbit-cloud` to the new APIs.** The cloud repo follows the heartbit version bump on its own cadence. B4 only changes the heartbit side; cloud's adoption PR is separate.
- **Refactor of `agent/orchestrator.rs`.** Still flagged as a god-module; still deferred (B3 explicitly punted it).

## Design

### Architecture

Five components, in dependency order:

```
┌──────────────────────────────────────────────────────────────────┐
│ heartbit-core                                                     │
│                                                                    │
│  auth/tenant.rs (NEW)                                              │
│    TenantScope { tenant_id: String, user_id: Option<String> }     │
│      .is_single_tenant() — empty string sentinel                   │
│      From<&UserContext>                                            │
│                                                                    │
│  sandbox.rs (NEW in core)                                          │
│    CorePathPolicy { allowed_dirs, deny_globs }                     │
│      .check_path(&Path) -> Result<(), Error>                       │
│                                                                    │
│  memory/ (MODIFIED)                                                │
│    Memory trait gains &TenantScope as first param on every method  │
│    NamespacedMemory still wraps; tenant comes from scope, not str  │
│                                                                    │
│  agent/audit.rs (MODIFIED)                                         │
│    AuditTrail trait: typed scope + retention + windowed reads      │
│      entries(scope, limit) / entries_unscoped(limit)               │
│      entries_since / prune                                         │
│                                                                    │
│  agent/runner.rs (MODIFIED)                                        │
│    AgentRunnerBuilder::max_tool_calls_per_turn(u32)               │
│    Per-turn cap enforced before tool dispatch                      │
│                                                                    │
│  tool/builtins/{bash,patch,edit,write,read}.rs (MODIFIED)         │
│    each accepts Option<Arc<CorePathPolicy>> at construction        │
│    .check_path() called pre-I/O                                    │
└──────────────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────┴────────────────────────────────────────┐
│ heartbit (umbrella)                                                │
│                                                                    │
│  sandbox/landlock.rs (MODIFIED)                                    │
│    SandboxPolicy now composes Arc<CorePathPolicy>                  │
│    .from_path_policy(Arc<CorePathPolicy>) constructor              │
│    Existing bash integration unchanged at the call site            │
│                                                                    │
│  daemon/ (MODIFIED)                                                │
│    Background prune task spawned by DaemonCore::run_*              │
│    daemon.toml: [daemon.audit] retain_days, prune_interval_minutes │
│                                                                    │
│  store/postgres/{audit,memory}.rs (MODIFIED)                       │
│    Migrations: NOT NULL on tenant_id text column, backfill ''      │
│    Composite indexes for (tenant, ...) query shapes                │
└──────────────────────────────────────────────────────────────────┘
```

**One new public module in core** (`auth::tenant`). **One promoted module** (`sandbox` lives in core now; umbrella's `sandbox/landlock.rs` becomes a thin landlock-backed extension that composes the core type). **Two SQL migrations.** **Zero new deps** (`glob` is already a transitive dep of `heartbit-core` via `mcp-presets/`; promoted to a direct dep). The only behavioral change for existing library users: methods that previously took no scope now require one. Single-tenant call sites pass `TenantScope::default()` (empty-string tenant id) and storage rows get `''` instead of `NULL` after the migration — functionally equivalent at the query level since the new `entries(scope, ...)` filters by `tenant_id = ''` and matches the same rows.

### Component 1: `heartbit_core::auth::tenant`

New file `crates/heartbit-core/src/auth/tenant.rs`, ~120 LOC + tests.

```rust
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
        let tenant_id = tenant_id.into();
        Self { tenant_id, user_id: None }
    }

    /// Add a user identity (typically `sub` claim from JWT).
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Single-tenant default; `tenant_id == ""`.
    pub fn single_tenant() -> Self {
        Self { tenant_id: String::new(), user_id: None }
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

impl From<&UserContext> for TenantScope {
    fn from(ctx: &UserContext) -> Self {
        Self {
            tenant_id: ctx.tenant_id.clone(),
            user_id: Some(ctx.user_id.clone()),
        }
    }
}
```

**Why `String`, not `Uuid`.** During self-review I checked `crates/heartbit/src/daemon/types.rs:181`. `UserContext.tenant_id` is deliberately `String` with this comment: *"All fields are strings to avoid coupling to a specific identity provider's ID format."* Auth0 / Cognito / Okta tenant identifiers aren't always UUIDs. Existing `MemoryEntry.author_tenant_id: Option<String>` and `AuditRecord.tenant_id: Option<String>` follow the same convention. Switching `TenantScope.tenant_id` to `Uuid` would introduce a parse-fail layer at every JWT boundary and break every existing storage row. Accepting `String` keeps the new type compatible with current data and current auth.

**Why owned, not `<'a>`.** The advisor flagged a borrowed variant during review: `TenantScope<'a>` would force lifetimes onto `Memory::recall`, `AuditTrail::entries`, `AgentEvent`, the orchestrator's spawn paths, and every `tokio::spawn` call site that closes over a scope. The library is async-heavy; owning two short strings is cheap relative to the engineering tax of threading lifetimes through future-bearing methods.

**`user_id` source.** In daemon mode, `TenantScope` is constructed in `crates/heartbit/src/daemon/auth_layer.rs` from the validated JWT's `UserContext` via the `From<&UserContext>` impl above. Standalone library users who don't have JWTs construct it manually. Single-tenant CLI mode uses `TenantScope::default()` everywhere.

Re-exported from `heartbit_core::auth` and from the umbrella `heartbit::auth`.

### Component 2: `heartbit_core::sandbox` — `CorePathPolicy`

New file `crates/heartbit-core/src/sandbox.rs`, ~200 LOC + tests.

```rust
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Path-level policy for filesystem-touching tools. Lives in `heartbit-core`
/// so all five filesystem builtins (bash, patch, edit, write, read) can
/// share enforcement. The umbrella's `SandboxPolicy` (landlock-backed)
/// composes a `CorePathPolicy` for the path-allow-list piece and adds
/// kernel-level enforcement on Linux.
#[derive(Debug, Clone)]
pub struct CorePathPolicy {
    /// Canonicalized allowed directories. Any read/write under one of
    /// these paths is permitted.
    allowed_dirs: Vec<PathBuf>,

    /// Glob patterns that are denied even within allowed_dirs.
    /// Examples: `**/.env`, `**/.git/config`, `**/secrets/**`.
    deny_globs: Vec<String>,
}

impl CorePathPolicy {
    pub fn builder() -> CorePathPolicyBuilder { ... }

    /// Returns Ok(()) if `path` is allowed, Err with a descriptive
    /// `Error::Sandbox(...)` otherwise. Caller canonicalizes if it
    /// wants symlink-following enforcement.
    pub fn check_path(&self, path: &Path) -> Result<(), Error> {
        let canonical = path.canonicalize()
            .map_err(|e| Error::Sandbox(format!("canonicalize {}: {e}", path.display())))?;

        // Must fall under one of allowed_dirs.
        let allowed = self.allowed_dirs.iter().any(|root| canonical.starts_with(root));
        if !allowed {
            return Err(Error::Sandbox(format!(
                "path {} not under any allowed directory",
                canonical.display()
            )));
        }

        // Must not match any deny_globs.
        for pat in &self.deny_globs {
            if glob_match(pat, &canonical) {
                return Err(Error::Sandbox(format!(
                    "path {} matches deny pattern {pat}",
                    canonical.display()
                )));
            }
        }

        Ok(())
    }
}
```

`CorePathPolicy` is plain Rust (no `landlock` dep, no Linux gating) so it compiles on every supported target. The `glob` crate is already a transitive dep of `heartbit-core` (via `mcp-presets/`); we add it as a direct dep.

**Tool wiring.** Each filesystem builtin's constructor gains an `Option<Arc<CorePathPolicy>>` parameter:

```rust
// crates/heartbit-core/src/tool/builtins/write.rs
impl WriteTool {
    pub fn new() -> Self { ... }
    pub fn with_path_policy(mut self, policy: Arc<CorePathPolicy>) -> Self {
        self.path_policy = Some(policy);
        self
    }
}

impl Tool for WriteTool {
    fn definition(&self) -> ToolDefinition { ... }

    fn execute(
        &self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        let path_policy = self.path_policy.clone();
        let tracker = self.tracker.clone();
        Box::pin(async move {
            let path = parse_path(&input)?;
            if let Some(p) = &path_policy {
                p.check_path(&path)?;
            }
            // ... existing FileTracker mtime guard ...
            // ... existing UTF-8 boundary guard ...
            // ... write
            Ok(ToolOutput::success(...))
        })
    }
}
```

This matches the existing `BashTool::with_sandbox_policy(...)` shape, which is the API the user explicitly chose during brainstorm. No trait extension, no wrapper-type indirection, no per-call argument threading.

### Component 3: Sandbox umbrella — `SandboxPolicy` composes `CorePathPolicy`

`crates/heartbit/src/sandbox/landlock.rs` keeps the existing `SandboxPolicy` API (it's the published type) but its construction now wraps a `CorePathPolicy`:

```rust
// crates/heartbit/src/sandbox/landlock.rs
use heartbit_core::sandbox::CorePathPolicy;

pub struct SandboxPolicy {
    path_policy: Arc<CorePathPolicy>,
    // existing landlock-specific fields ...
}

impl SandboxPolicy {
    /// Build a SandboxPolicy from an already-constructed CorePathPolicy.
    /// Used by callers (typically the daemon) that share one path policy
    /// across multiple tools — bash gets the full Sandbox, others get
    /// just the path policy.
    pub fn from_path_policy(path_policy: Arc<CorePathPolicy>) -> Self { ... }

    /// Convenience: build path policy + sandbox in one call (existing API
    /// kept; now constructs CorePathPolicy internally).
    pub fn builder() -> SandboxPolicyBuilder { ... }

    /// Expose the inner CorePathPolicy so callers can pass it to
    /// non-bash filesystem tools.
    pub fn path_policy(&self) -> Arc<CorePathPolicy> {
        self.path_policy.clone()
    }
}
```

**Wiring pattern in the daemon and CLI:**

```rust
let path_policy = Arc::new(CorePathPolicy::builder()
    .allow_dir(workspace)
    .deny_glob("**/.env")
    .build()?);

let bash_sandbox = SandboxPolicy::from_path_policy(path_policy.clone());

let bash = BashTool::new().with_sandbox_policy(Arc::new(bash_sandbox));
let write = WriteTool::new().with_path_policy(path_policy.clone());
let edit = EditTool::new().with_path_policy(path_policy.clone());
let patch = PatchTool::new().with_path_policy(path_policy.clone());
let read = ReadTool::new().with_path_policy(path_policy);
```

This is the design the advisor's review converged on after flagging the original (which had `CorePathPolicy` "moved" out of the umbrella entirely — that would have left the umbrella's published `SandboxPolicy` API contradicting itself). The split keeps `SandboxPolicy` on the umbrella (where landlock lives, since landlock is Linux-gated and `cfg(linux)`), keeps `CorePathPolicy` portable in core, and uses composition rather than re-exposure.

### Component 4: `AgentRunner::max_tool_calls_per_turn`

Modify `crates/heartbit-core/src/agent/runner.rs`:

```rust
pub struct AgentRunnerBuilder {
    // ... existing fields ...
    max_tool_calls_per_turn: Option<u32>,
}

impl AgentRunnerBuilder {
    /// Cap the number of tool calls dispatched per LLM turn. Excess
    /// calls return `Error::Agent` with the partial usage attached;
    /// the caller can retry with a tighter system prompt or a smaller
    /// tool set.
    ///
    /// Default: None (unlimited). Recommended for production: 8.
    /// Zero is rejected at build time.
    pub fn max_tool_calls_per_turn(mut self, cap: u32) -> Self {
        self.max_tool_calls_per_turn = Some(cap);
        self
    }
}
```

In the run loop, after parsing the LLM response's `tool_use` blocks:

```rust
// crates/heartbit-core/src/agent/runner.rs (run loop, after LLM response parse)
let tool_calls: Vec<ToolCall> = response.tool_calls();

if let Some(cap) = self.max_tool_calls_per_turn
    && tool_calls.len() as u32 > cap
{
    return Err(Error::Agent(format!(
        "tool-call cap exceeded: turn produced {} calls, max is {}",
        tool_calls.len(), cap
    )).with_partial_usage(usage_so_far));
}
```

Build-time validation rejects zero (matching the existing `max_turns > 0` and `max_tokens > 0` rules in `HeartbitConfig::validate`):

```rust
// runner.rs build()
if let Some(0) = self.max_tool_calls_per_turn {
    return Err(Error::Config("max_tool_calls_per_turn must be > 0 if set".into()));
}
```

Config wiring: add `max_tool_calls_per_turn: Option<u32>` to `AgentConfig` and to `OrchestratorConfig` (defaults). Single-agent fast path in `build_orchestrator_from_config` propagates it. SubAgentConfig carries it.

### Component 5: Tenant-scoped Memory + Audit

The project uses `Pin<Box<dyn Future>>`-style async traits for object safety (matches `Tool` and existing `Memory` / `AuditTrail` impls). Both updated traits keep that style — no `#[async_trait]`, no `async fn` in trait. This is consistent with the existing codebase (verified in `crates/heartbit-core/src/memory/mod.rs:148` and `crates/heartbit-core/src/agent/audit.rs:74`).

**Memory trait change.** `crates/heartbit-core/src/memory/mod.rs`. Today's trait has six methods that take `&self` plus the entry / query / id directly. Tenant scoping is currently done by `NamespacedMemory` wrapping with an `agent_prefix` string convention. The change adds `&TenantScope` as an explicit first parameter, forcing every call site to commit to a scope at compile time:

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
        scope: &TenantScope,
        _id: &str,
        _related_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }

    fn prune(
        &self,
        scope: &TenantScope,
        _min_strength: f64,
        _min_age: chrono::Duration,
        _agent_prefix: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async { Ok(0) })
    }
}
```

Implementations populate `MemoryEntry.author_tenant_id` from `scope.tenant_id` on `store`, and filter on `scope.tenant_id` (along with the existing `agent_prefix`) on `recall` / `prune`. `InMemoryStore`, `PostgresMemoryStore`, `NamespacedMemory` all updated.

`NamespacedMemory` is now a thinner layer: it still rewrites `agent` field with a per-namespace prefix (so one base store can host many isolated namespaces without separate tables), but tenant identity comes from `TenantScope` rather than a string captured at construction. We keep `NamespacedMemory` for namespace partitioning *within* a tenant (e.g., per-thread or per-conversation scopes); cross-tenant isolation now happens at the scope layer.

**AuditTrail trait change.** `crates/heartbit-core/src/agent/audit.rs`. The current trait has:

```rust
fn record(&self, entry) -> ...;
fn entries(&self) -> ...;                              // ALL records
fn entries_for_tenant(&self, tenant_id: Option<&str>) -> ...;  // string-typed
fn erase_for_user(&self, user_id) -> ...;              // GDPR
```

After B4:

```rust
pub trait AuditTrail: Send + Sync {
    fn record(
        &self,
        entry: AuditRecord,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    /// Tenant-scoped read. `scope.is_single_tenant()` returns the records
    /// stored under the empty-string sentinel — i.e., single-tenant data.
    fn entries(
        &self,
        scope: &TenantScope,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Cross-tenant admin read. Renamed from the previous unscoped
    /// `entries()` so call sites must explicitly opt into cross-tenant
    /// visibility — loud, greppable, audit-able.
    fn entries_unscoped(
        &self,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Time-windowed scoped read.
    fn entries_since(
        &self,
        scope: &TenantScope,
        since: DateTime<Utc>,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Delete entries older than `now - retain`. Returns count deleted.
    fn prune(
        &self,
        retain: chrono::Duration,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>>;

    /// GDPR right-to-erasure. Default no-op; overridden by impls that
    /// support deletion.
    fn erase_for_user(
        &self,
        _user_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async { Ok(0) })
    }
}
```

Three explicit renames / signature changes:

- `entries(&self)` → `entries_unscoped(&self, limit)`. The unscoped path keeps existing semantics (return all rows) but is renamed and now requires a `limit` so an admin can't accidentally fetch a billion-row audit log. Call sites that did `.entries()` for read-only display purposes get a compile error pointing them at the choice between scoped and unscoped.
- `entries_for_tenant(Option<&str>)` → `entries(scope, limit)`. Typed instead of stringly. The `Option<&str> = None` previously meant "all"; that case now goes through `entries_unscoped`. The `Some` case maps to a typed `&TenantScope`.
- New: `entries_since` for windowed reads. New: `prune` for retention.

This is acceptable as a pre-release breaking change because (1) `AuditTrail` was added in the B3 round and has no published downstream consumers, (2) `heartbit-core` is not yet on crates.io. CHANGELOG documents the rename and the migration recipe.

**Storage retention.** `PostgresAuditTrail::prune` runs `DELETE FROM audit WHERE created_at < $1` chunked at 10k rows with a 50ms sleep between chunks (to avoid blocking concurrent writes on a large audit table). `InMemoryAuditTrail::prune` walks the `RwLock<Vec<AuditRecord>>` and retains rows newer than the threshold. Operators wire retention via daemon config:

```toml
[daemon.audit]
retain_days = 90  # default: unlimited (None) — matches today
prune_interval_minutes = 60  # how often the background task runs
```

A new background task spawned in `DaemonCore::run_*` calls `audit.prune(retain)` on the configured interval. The audit trait itself stays in `heartbit-core`; the retention background task is in `heartbit/daemon/` because it's daemon-mode-specific.

### Component 6: Postgres tenant columns

**Current state.** I checked the live schemas (`crates/heartbit/src/memory/postgres.rs:176` and `crates/heartbit/src/store/postgres.rs:101`). The `memories` table has no `author_tenant_id` column at all; the `audit_log` table has no `tenant_id` column at all. The Rust structs carry these fields, but they're **dropped on persistence** — `PostgresMemoryStore::recall` always returns `author_tenant_id: None`, and `write_audit` never persists tenant. Tenant scoping in Postgres is currently non-functional.

This makes Component 6 larger than originally drafted: we're adding columns, writing them on insert, and filtering them in queries — not just adding `NOT NULL` to existing columns. The migrations use idempotent `ADD COLUMN IF NOT EXISTS` (matching the existing migration pattern in `memory/postgres.rs:198`) so re-running is safe.

**Migrations** are added inline in the existing `run_migration` methods (the codebase already does idempotent migrations this way; no separate `migrations/` directory is in use today, and adding one would be a separate refactor):

- In `memory/postgres.rs::run_migration` — append:
  ```sql
  ALTER TABLE memories ADD COLUMN IF NOT EXISTS author_tenant_id TEXT NOT NULL DEFAULT '';
  ALTER TABLE memories ADD COLUMN IF NOT EXISTS author_user_id TEXT;
  CREATE INDEX IF NOT EXISTS idx_memories_tenant ON memories(author_tenant_id, agent);
  ```
- In `store/postgres.rs::run_migration` — append:
  ```sql
  ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT '';
  ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS user_id TEXT;
  CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, created_at DESC);
  ```

**Default `''`** matches `TenantScope::single_tenant()`. Existing rows get `''`; new rows under a real tenant get the actual tenant id. Single-tenant deployments are unaffected. The empty-string default is what makes `TenantScope::default()` queries match historical rows transparently.

**Code changes:**

- `PostgresMemoryStore::store` — bind `entry.author_tenant_id.unwrap_or_default()` and `entry.author_user_id`.
- `PostgresMemoryStore::recall` — `WHERE author_tenant_id = $N` filter; populate `author_tenant_id` and `author_user_id` on the returned `MemoryEntry`.
- `PostgresAuditTrail` (today: implicit via `AuditStore::write_audit`) — bind `record.tenant_id.as_deref().unwrap_or("")` and `record.user_id`. Add `audit_log.tenant_id` filter to scoped reads.

Migrations are gated behind the `postgres` feature; library-only consumers see no schema change. No `task_outcomes` table exists (the spec's earlier draft mentioned it; that was a stale reference — there is `tasks` but tenant scoping for tasks is out of scope and follows in B5).

**Pre-migration audit query** for operators (documented in upgrade notes), useful only on installations that ran a downstream patch adding the columns earlier:

```sql
SELECT count(*) FROM memories WHERE author_tenant_id IS NULL;
SELECT count(*) FROM audit_log WHERE tenant_id IS NULL;
```

On a stock installation both queries return `0` because the columns don't exist yet — `ADD COLUMN IF NOT EXISTS ... NOT NULL DEFAULT ''` creates them already filled.

### CLI / config surface

```toml
# heartbit.toml
[orchestrator]
# Cap tool dispatch per turn across the whole orchestrator.
max_tool_calls_per_turn = 8

[[agents]]
name = "researcher"
# Override per agent; None inherits orchestrator default.
max_tool_calls_per_turn = 16

[sandbox]
allowed_dirs = ["/workspace", "/tmp/agent"]
deny_globs = ["**/.env", "**/.git/config", "**/secrets/**"]

[daemon.audit]
retain_days = 90
prune_interval_minutes = 60
```

Env vars:

- `HEARTBIT_MAX_TOOL_CALLS_PER_TURN` — runtime override.
- `HEARTBIT_AUDIT_RETAIN_DAYS` — runtime override.

CLI gets no new subcommand; `serve`/`run`/`chat` pick up new config. A new `audit-prune` subcommand is **out of scope** for B4 — operators run the migration once and let the daemon's background task handle ongoing retention.

## Test plan

Per-task tests (red → green → refactor) per CLAUDE.md TDD rule. Highlights:

- `auth/tenant.rs` — `single_tenant()` returns `tenant_id = ""`; `is_single_tenant()` agrees; `From<&UserContext>` round-trip preserves both fields; `new("")` collapses to single-tenant; `with_user` adds user; PartialEq holds on equal scopes.
- `sandbox.rs` (core) — denies path outside `allowed_dirs`; denies `**/.env` even inside allowed; allows symlinked path that canonicalizes back into allowed; canonicalize-failure produces clear error.
- `runner.rs` — `max_tool_calls_per_turn = Some(2)` with a turn returning 3 tool_use blocks: error returned, partial usage attached, no tool dispatched.
- Each of `bash`, `patch`, `edit`, `write`, `read` — when constructed with a `CorePathPolicy` that denies `/etc/passwd`, calling the tool against `/etc/passwd` returns `Error::Sandbox(...)` and never opens the file. Existing tests (FileTracker, UTF-8 boundary) still pass.
- `Memory` — `recall` with scope `A` does not return entries stored under scope `B` (InMemoryStore + NamespacedMemory + PostgresMemoryStore). Existing single-tenant tests adopt `TenantScope::default()`.
- `AuditTrail::entries(scope)` — returns only matching tenant rows; `entries_unscoped(limit)` returns all; `prune(7 days)` removes rows older than now − 7 days.
- Postgres migrations — apply against a fresh DB and a DB populated with NULL tenant rows: no data loss, NULL rows backfilled to `''`, `NOT NULL DEFAULT ''` added, composite index created, scoped queries match the same rows as before.

Total estimated new tests: ~70.

## Sequencing

The advisor flagged that branching B4 from main *before* PR #3 merges produces a heisen-merge state where the user-docs PR includes spec edits for B4 but no implementation. Sequence:

0. **Wait for PR #3 (user-docs) to merge to main.** Spec lands on main first via a separate small commit (this document); implementation branches from post-#3-merge main.

Implementation order (~10 commits):

1. `feat(core): TenantScope type in heartbit_core::auth::tenant` — type + builder + tests; not yet wired anywhere. Independent commit.
2. `feat(core): CorePathPolicy in heartbit_core::sandbox` — type + builder + tests + glob dep added. Not yet wired. Independent commit.
3. `feat(core): max_tool_calls_per_turn on AgentRunner` — builder method, build-time validation, run-loop check, tests. Independent commit.
4. `refactor(core): Memory trait takes &TenantScope` — trait change, all impls updated, all tests updated to `TenantScope::default()` for single-tenant, then targeted tests for multi-tenant isolation. Migration commit (large diff).
5. `refactor(heartbit): SandboxPolicy composes CorePathPolicy` — landlock side composes the new core type; `from_path_policy` constructor; `path_policy()` accessor.
6. `feat(core): with_path_policy on bash/patch/edit/write/read` — builder method on each, `check_path` call in `execute`, tests. Five tools, one commit (mechanical).
7. `refactor(heartbit): AuditTrail trait scoped reads + prune` — trait + InMemory impl + Postgres impl + retention background task in DaemonCore.
8. `feat(postgres): tenant_id NOT NULL migrations` — two SQL migrations (audit + memory), sqlx-tested against a fresh DB and a DB populated with NULL-tenant rows; backfill confirms parity.
9. `feat(cli): wire config — max_tool_calls_per_turn + sandbox + audit retention` — TOML schema, env vars, single-agent fast path, sub-agent propagation.
10. `docs: B4 CHANGELOG, multi-tenant chapter in mdBook, breaking-change callouts` — book chapter under `book/src/recipes/multi-tenant.md`, CHANGELOG.md entry, README.md note pointing at the chapter.

Each commit must keep `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` green.

## Risks

1. **Tenant-scope migration is a big diff.** Commit 4 touches every `Memory::*` call site in heartbit-core, heartbit, and heartbit-cli. Likely 80–120 changed lines across ~25 files. Mitigated by:
   - Default to `TenantScope::default()` everywhere existing single-tenant code lives.
   - Run `cargo test --workspace` after every file's edits, not at the end.
   - Use `git mv`-friendly renames so blame survives.
2. **Postgres migration backfill on production data.** The memory migration backfills NULL `author_tenant_id` rows with `''` (single-tenant sentinel). If a deployment was in true multi-tenant mode and had nulls (a pre-existing bug), this collapses them into the single-tenant bucket and they become invisible to scoped queries. Mitigated by: pre-migration audit query (`SELECT count(*) FROM memory WHERE author_tenant_id IS NULL`) documented in the upgrade notes; operators check before running. Same treatment for audit.
3. **`CorePathPolicy::check_path` does `canonicalize()` per call.** That's a syscall. For tight loops (e.g., a `read` called 100 times per turn) this is measurable. Mitigated by: builtins typically open the file shortly after, so the canonicalize cost is on the same hot path as the I/O. If profiling later shows it's a bottleneck, add a path-cache as a follow-up.
4. **Audit retention background task can starve.** A `prune` query against a 100M-row Postgres audit table can block. Mitigated by: `DELETE ... WHERE created_at < $1 LIMIT 10000` chunked loop with `tokio::time::sleep(50ms)` between chunks. Documented in commit 7.
5. **Pre-release breaking changes still annoy any external consumer.** No external consumers are known; the umbrella's `heartbit::Memory` was stable but `&TenantScope` adds a parameter. CHANGELOG calls this out as a 0.x → 0.y breaking change with a `TenantScope::default()` migration recipe.

## Out-of-Scope (deferred to B5)

These were identified in the same production-readiness review but are *failure-mode* gaps, not multi-tenant gaps. They become B5 after B4 ships:

- **Idempotency keys on `DaemonCommand::SubmitTask`.** Today, a Kafka redelivery re-runs the agent.
- **Context-overflow accounting per tenant.** Auto-compaction is global; a noisy tenant can OOM the in-memory token tracker.
- **Structured retry policy with circuit breakers.** `RetryingProvider` is wired but uses fixed exponential backoff with no per-tenant tracking.

After B5: release prep (CHANGELOG → CHANGELOG.lock, version bump, `cargo publish` heartbit-core, GitHub release, docs.heartbit.ai DNS verification).
