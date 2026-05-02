# Multi-Tenant Hardening

Heartbit's multi-tenant story is grounded in three load-bearing types added in
the B4 round:

- `TenantScope` — required parameter on every `Memory` and `AuditTrail` method.
- `CorePathPolicy` — shared path allowlist + glob denylist for all filesystem
  builtins.
- `max_tool_calls_per_turn` — caps tool dispatches per LLM turn.

Together they let one `heartbit` runtime serve many tenants without code paths
having to remember which tenant they're acting on. The compiler and the type
system enforce the boundary.

## TenantScope

```rust
use heartbit_core::auth::TenantScope;

let scope = TenantScope::new("acme-corp")
    .with_user("user-42");

memory.recall(&scope, query).await?;
audit.entries(&scope, 100).await?;
```

The empty string is the single-tenant sentinel. `TenantScope::default()` gives
you a single-tenant scope:

```rust
let scope = TenantScope::default();
assert!(scope.is_single_tenant());
```

`TenantScope::new("")` collapses to `single_tenant()` and `with_user("")`
clears the user identity to `None` — both empty-string traps are normalized
in code, not just in prose.

In daemon mode the scope is built from JWT claims via `From<&UserContext>`:

```rust
use heartbit_core::auth::TenantScope;
use heartbit::daemon::UserContext;

let ctx = UserContext { /* extracted from JWT */ };
let scope: TenantScope = (&ctx).into();
```

If your code already has the tenant id and user id as separate `Option<String>`
fields, the `from_audit_fields` helper gives you a scope without writing a
match arm:

```rust
let scope = TenantScope::from_audit_fields(
    self.audit_tenant_id.as_deref(),
    self.audit_user_id.as_deref(),
);
```

## CorePathPolicy

```rust
use heartbit_core::CorePathPolicy;
use std::sync::Arc;

let policy = Arc::new(
    CorePathPolicy::builder()
        .allow_dir("/workspace")
        .deny_glob("**/.env")
        .deny_glob("**/secrets/**")
        .build()?
);
```

Every filesystem builtin accepts the same `Arc<CorePathPolicy>` via
`with_path_policy`:

```rust
let read_tool  = ReadTool::new(...).with_path_policy(policy.clone());
let write_tool = WriteTool::new(...).with_path_policy(policy.clone());
let edit_tool  = EditTool::new(...).with_path_policy(policy.clone());
let patch_tool = PatchTool::new(...).with_path_policy(policy.clone());
```

For bash, compose a full `SandboxPolicy` so you also get kernel-level Landlock
enforcement on Linux:

```rust
use heartbit::sandbox::SandboxPolicy;

let sandbox = SandboxPolicy::from_path_policy(policy.clone());
let bash = BashTool::new().with_sandbox_policy(Arc::new(sandbox));
```

`from_path_policy` derives the Landlock read/write paths from the policy's
allowed directories — bash subprocesses can read and write everything the
path policy allows, no extra wiring needed.

### Configuring via TOML

The CLI builds the policy automatically from `[sandbox]`:

```toml
[sandbox]
allowed_dirs = ["/workspace", "/tmp/agent"]
deny_globs = ["**/.env", "**/secrets/**"]
```

All five filesystem builtins inherit it. On Linux with the `sandbox` feature,
bash also gets the wrapped `SandboxPolicy`.

## max_tool_calls_per_turn

```rust
let runner = AgentRunnerBuilder::new()
    .max_tool_calls_per_turn(8)  // production recommendation
    .build()?;
```

This is **distinct from** `max_tools_per_turn`:

- `max_tools_per_turn` limits the *tool definitions* offered to the LLM
  (pre-filter). Use it to keep the prompt tight when an agent has many tools.
- `max_tool_calls_per_turn` caps the *invocations* the LLM produces per turn
  (post-response). Use it to bound concurrent dispatch and protect per-task
  budgets from a misbehaving model that emits dozens of tool calls in one turn.

Excess returns `Error::Agent` wrapped in `Error::WithPartialUsage` — the
caller can retry with a tighter system prompt or a smaller tool set, and the
token cost from the offending turn is preserved. Zero is rejected at
build time.

### Configuring via TOML

```toml
[orchestrator]
max_tool_calls_per_turn = 8

[[agents]]
name = "researcher"
max_tool_calls_per_turn = 16  # per-agent override
```

Or via env var: `HEARTBIT_MAX_TOOL_CALLS_PER_TURN=8`.

## Postgres tenant columns

The `memories` and `audit_log` tables now carry tenant identity:

| Table       | Column              | Type         | Notes                       |
|-------------|---------------------|--------------|-----------------------------|
| `memories`  | `author_tenant_id`  | TEXT NOT NULL DEFAULT '' | indexed                     |
| `memories`  | `author_user_id`    | TEXT NULL    |                             |
| `audit_log` | `tenant_id`         | TEXT NOT NULL DEFAULT '' | indexed                     |
| `audit_log` | `user_id`           | TEXT NULL    |                             |

The default is the empty string (single-tenant sentinel) so existing rows
remain visible to `TenantScope::default()`-scoped queries. New rows under a
real tenant store the actual tenant id.

Run `PostgresStore::run_migration` once on upgrade. It's idempotent and safe
to re-run. Before upgrading from a pre-B4 deployment, audit existing data:

```sql
SELECT count(*) FROM memories WHERE author_tenant_id IS NULL;
SELECT count(*) FROM audit_log WHERE tenant_id IS NULL;
```

Non-zero on a multi-tenant installation indicates rows that were written
without a scope — investigate before running the migration.

## Audit retention

```toml
[daemon.audit]
retain_days = 90
prune_interval_minutes = 60
```

The daemon spawns a background task that calls `audit.prune(retain)` on the
configured interval. The task observes the daemon's cancellation token and
exits cleanly on graceful shutdown.

`retain_days = 0` and `prune_interval_minutes = 0` are rejected at config-load
time (would either delete everything or cause a `tokio::time::interval` panic).

For one-off retention via env var: `HEARTBIT_AUDIT_RETAIN_DAYS=90` (the TOML
takes precedence when both are set).

## Programmatic API

```rust
use heartbit_core::auth::TenantScope;
use heartbit_core::AuditTrail;

// tenant-scoped read
let recent = audit.entries(&scope, 100).await?;

// time-windowed scoped read
let since = chrono::Utc::now() - chrono::Duration::hours(24);
let last_day = audit.entries_since(&scope, since, 1000).await?;

// admin / cross-tenant read (loud, greppable)
let all = audit.entries_unscoped(1000).await?;

// retention prune
let removed = audit.prune(chrono::Duration::days(90)).await?;
```

The rename from `entries()` (no args, returned all) to `entries_unscoped(limit)`
is a deliberate breaking change so call sites must opt into cross-tenant
visibility — a `grep entries_unscoped` shows every site that crosses the
tenant boundary.
