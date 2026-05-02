# B5b Failure-Mode Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land idempotency keys, per-tenant context-overflow accounting, and per-(tenant, provider) circuit breakers in the heartbit daemon.

**Architecture:** Three independent components composed over `TenantScope`. Idempotency adds a `daemon_tasks.idempotency_key` column with a partial unique index `(tenant_id, idempotency_key) WHERE NOT NULL`, plus a 24h TTL sweep. The token tracker is a process-local `HashMap<TenantId, State>` with RAII reservations (`Arc`-owning, async-safe). The circuit breaker is a per-(tenant, provider) state machine wrapped around `RetryingProvider` (outer-circuit, inner-retry).

**Tech Stack:** Rust 2024, `tokio`, `sqlx`, `parking_lot`, `chrono`, `axum`, `rdkafka`. New crate dep on `parking_lot` in `heartbit-core`.

**Spec:** `docs/superpowers/specs/2026-05-02-b5b-failure-mode-hardening-design.md`

**Working tree:** Set up an isolated worktree before starting (`superpowers:using-git-worktrees`). Suggested branch: `feat/b5b-failure-mode-hardening`.

---

## Task 1: Idempotency schema + `DaemonTask` field

**Goal:** Tighten `daemon_tasks.tenant_id` to `NOT NULL DEFAULT ''` (matches B4's audit_log pattern), add `idempotency_key TEXT` column + partial unique index, and add the field to `DaemonTask`.

**Files:**
- Modify: `crates/heartbit/src/daemon/types.rs:104-136` (`DaemonTask` struct + `new` / `new_with_user`)
- Modify: `crates/heartbit/src/daemon/store.rs:389-449` (`PostgresTaskStore::run_migration`)
- Modify: `crates/heartbit/src/daemon/store.rs:188-369` (`InMemoryTaskStore` row map preserves the new field)

**TDD steps:**

- [ ] **Step 1.1: Write failing unit test for `DaemonTask::idempotency_key`**

In `crates/heartbit/src/daemon/types.rs` (test module at end of file):

```rust
#[test]
fn daemon_task_default_idempotency_key_is_none() {
    let task = DaemonTask::new(Uuid::nil(), "hello", "test");
    assert!(task.idempotency_key.is_none());
}

#[test]
fn daemon_task_idempotency_key_round_trips_through_serde() {
    let mut task = DaemonTask::new(Uuid::nil(), "hello", "test");
    task.idempotency_key = Some("idem-abc-123".to_string());
    let json = serde_json::to_string(&task).unwrap();
    assert!(json.contains("idempotency_key"));
    let back: DaemonTask = serde_json::from_str(&json).unwrap();
    assert_eq!(back.idempotency_key.as_deref(), Some("idem-abc-123"));
}

#[test]
fn daemon_task_missing_idempotency_key_field_deserializes_as_none() {
    // Backward compat: payloads from before B5b have no `idempotency_key` field.
    let legacy = r#"{
        "id": "00000000-0000-0000-0000-000000000000",
        "task": "x",
        "state": "pending",
        "created_at": "2026-05-02T00:00:00Z",
        "tokens_used": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "reasoning_tokens": 0
        },
        "tool_calls_made": 0,
        "source": "test"
    }"#;
    let task: DaemonTask = serde_json::from_str(legacy).unwrap();
    assert!(task.idempotency_key.is_none());
}
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cargo test -p heartbit --lib daemon::types::tests::daemon_task -- --nocapture
```

Expected: 3 failures with "no field `idempotency_key`".

- [ ] **Step 1.3: Add the field to `DaemonTask`**

Add immediately before the `model_name` field (around `crates/heartbit/src/daemon/types.rs:135`):

```rust
    /// Idempotency key supplied by the client to dedup retries
    /// (e.g., Kafka redelivery, HTTP retry-on-timeout). Scoped to
    /// `(tenant_id, idempotency_key)` via a partial unique index.
    /// Cleared by the TTL sweep after `idempotency.ttl_hours`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
```

In `DaemonTask::new` (around line 140), add `idempotency_key: None,` to the struct literal. `new_with_user` reuses `new` via `..Self::new(...)` so it gets it for free.

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
cargo test -p heartbit --lib daemon::types::tests::daemon_task -- --nocapture
```

Expected: all 3 pass.

- [ ] **Step 1.5: Update `PostgresTaskStore::run_migration` for `daemon_tasks`**

Edit `crates/heartbit/src/daemon/store.rs` inside `run_migration` (around line 423, after the indexes-individually block). Add these statements **after** the existing column-add loop (around line 446):

```rust
            // B5b: tighten tenant_id to NOT NULL DEFAULT '' for symmetry with
            // audit_log (B4) and so the partial unique idempotency index has a
            // guaranteed-present column to scope on.
            for stmt in [
                "UPDATE daemon_tasks SET tenant_id = '' WHERE tenant_id IS NULL",
                "ALTER TABLE daemon_tasks ALTER COLUMN tenant_id SET DEFAULT ''",
                "ALTER TABLE daemon_tasks ALTER COLUMN tenant_id SET NOT NULL",
                "ALTER TABLE daemon_tasks ADD COLUMN IF NOT EXISTS idempotency_key TEXT",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_daemon_tasks_idem
                   ON daemon_tasks (tenant_id, idempotency_key)
                   WHERE idempotency_key IS NOT NULL",
                "CREATE INDEX IF NOT EXISTS idx_daemon_tasks_created_at_for_sweep
                   ON daemon_tasks (created_at)
                   WHERE idempotency_key IS NOT NULL",
            ] {
                sqlx::query(stmt)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| Error::Daemon(format!("idempotency migration failed: {e}")))?;
            }
```

- [ ] **Step 1.6: Update Postgres `insert` to bind `idempotency_key`**

Edit `crates/heartbit/src/daemon/store.rs` `insert` impl (around lines 458-487). Change the SQL from 20 columns/binds to 21:

```rust
                sqlx::query(
                    r#"INSERT INTO daemon_tasks
                        (id, task, state, created_at, started_at, completed_at, result, error,
                         input_tokens, output_tokens, cache_creation_input_tokens,
                         cache_read_input_tokens, reasoning_tokens, tool_calls_made,
                         estimated_cost_usd, source, agent_name, user_id, tenant_id, model_name,
                         idempotency_key)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                            $16, $17, $18, $19, $20, $21)"#,
                )
                // ... existing binds 1..20 unchanged ...
                .bind(&task.idempotency_key)
                .execute(&pool)
```

Also update the SELECT in `get` and `list` to include `idempotency_key`. Locate the `TaskRow` struct (search `struct TaskRow`) and add `idempotency_key: Option<String>` after `model_name`. Add `idempotency_key: row.idempotency_key` to every row→`DaemonTask` conversion site.

- [ ] **Step 1.7: Update `InMemoryTaskStore` to preserve `idempotency_key`**

`InMemoryTaskStore` stores `DaemonTask` directly (search "struct InMemoryTaskStore" around `crates/heartbit/src/daemon/store.rs:188`). The new field is preserved automatically; no code changes needed in the in-memory store for this step.

- [ ] **Step 1.8: Run unit tests to verify nothing broke**

```bash
cargo test -p heartbit --lib daemon::store -- --nocapture
```

Expected: all pre-existing in-memory tests pass (Postgres-only tests are `#[ignore]`-gated).

- [ ] **Step 1.9: Commit**

```bash
git add crates/heartbit/src/daemon/types.rs crates/heartbit/src/daemon/store.rs
git commit -m "$(cat <<'EOF'
feat(daemon): add idempotency_key column + tighten tenant_id

Migration:
- daemon_tasks.tenant_id: NOT NULL DEFAULT '' (matches audit_log B4 pattern)
- daemon_tasks.idempotency_key: TEXT (nullable; partial unique index scoped
  to (tenant_id, idempotency_key) where not null)
- idx_daemon_tasks_created_at_for_sweep for the TTL sweep

DaemonTask gains idempotency_key: Option<String> with serde default so
existing serialized payloads still deserialize.

This is the foundation for B5b Component 1 (Stripe-style idempotency keys
on SubmitTask).
EOF
)"
```

---

## Task 2: `TaskStore` trait extension + `is_unique_violation` helper

**Goal:** Add `find_by_idempotency_key` and `sweep_expired_idempotency_keys` methods to `TaskStore`. Both stores implement them. Add a `is_unique_violation` helper.

**Files:**
- Modify: `crates/heartbit/src/daemon/store.rs:14-186` (trait + InMemory impl)
- Modify: `crates/heartbit/src/daemon/store.rs:452-905` (Postgres impl)

**TDD steps:**

- [ ] **Step 2.1: Write failing tests in `mod tests` at end of `crates/heartbit/src/daemon/store.rs`**

```rust
#[test]
fn in_memory_find_by_idempotency_key_returns_match() {
    let store = InMemoryTaskStore::new();
    let id = Uuid::new_v4();
    let mut task = DaemonTask::new_with_user(id, "hello", "kafka", "u1", "tenant-A");
    task.idempotency_key = Some("idem-1".into());
    store.insert(task).unwrap();

    let found = store.find_by_idempotency_key("tenant-A", "idem-1").unwrap();
    assert_eq!(found.map(|t| t.id), Some(id));
}

#[test]
fn in_memory_find_by_idempotency_key_isolates_tenants() {
    let store = InMemoryTaskStore::new();
    let mut t1 = DaemonTask::new_with_user(Uuid::new_v4(), "x", "kafka", "u1", "tenant-A");
    t1.idempotency_key = Some("idem-shared".into());
    store.insert(t1).unwrap();

    let mut t2 = DaemonTask::new_with_user(Uuid::new_v4(), "y", "kafka", "u2", "tenant-B");
    t2.idempotency_key = Some("idem-shared".into());
    store.insert(t2).unwrap();

    let found_a = store.find_by_idempotency_key("tenant-A", "idem-shared").unwrap();
    let found_b = store.find_by_idempotency_key("tenant-B", "idem-shared").unwrap();
    assert!(found_a.is_some());
    assert!(found_b.is_some());
    assert_ne!(found_a.unwrap().id, found_b.unwrap().id);
}

#[test]
fn in_memory_find_by_idempotency_key_returns_none_when_missing() {
    let store = InMemoryTaskStore::new();
    let found = store.find_by_idempotency_key("tenant-A", "missing").unwrap();
    assert!(found.is_none());
}

#[test]
fn in_memory_sweep_clears_keys_older_than_cutoff() {
    let store = InMemoryTaskStore::new();
    let id_old = Uuid::new_v4();
    let mut old = DaemonTask::new_with_user(id_old, "x", "kafka", "u", "t");
    old.idempotency_key = Some("idem-old".into());
    old.created_at = chrono::Utc::now() - chrono::Duration::hours(48);
    store.insert(old).unwrap();

    let id_new = Uuid::new_v4();
    let mut fresh = DaemonTask::new_with_user(id_new, "x", "kafka", "u", "t");
    fresh.idempotency_key = Some("idem-fresh".into());
    store.insert(fresh).unwrap();

    let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
    let cleared = store.sweep_expired_idempotency_keys(cutoff).unwrap();
    assert_eq!(cleared, 1);
    assert!(store.find_by_idempotency_key("t", "idem-old").unwrap().is_none());
    assert!(store.find_by_idempotency_key("t", "idem-fresh").unwrap().is_some());
}

#[test]
fn is_unique_violation_recognizes_pg_23505_signature() {
    use crate::Error;
    let err = Error::Daemon("failed to insert task: error returned from database: duplicate key value violates unique constraint \"idx_daemon_tasks_idem\" (code: 23505)".into());
    assert!(super::is_unique_violation(&err));
    let err2 = Error::Daemon("connection refused".into());
    assert!(!super::is_unique_violation(&err2));
}
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
cargo test -p heartbit --lib daemon::store::tests::in_memory_find -- --nocapture
cargo test -p heartbit --lib daemon::store::tests::in_memory_sweep -- --nocapture
cargo test -p heartbit --lib daemon::store::tests::is_unique_violation -- --nocapture
```

Expected: 5 failures with "method not found" / "function not found".

- [ ] **Step 2.3: Add the trait methods**

Edit the `TaskStore` trait (`crates/heartbit/src/daemon/store.rs:14`). Add after the `update` method:

```rust
    /// Find a task by `(tenant_id, idempotency_key)`. Returns `None` if no
    /// matching live row exists, or if the key has been nulled out by the
    /// TTL sweep (the row may still exist; we just no longer dedup against it).
    fn find_by_idempotency_key(
        &self,
        tenant_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<DaemonTask>, Error>;

    /// Null out the `idempotency_key` field on rows older than `cutoff`.
    /// Returns the number of rows updated. The row itself is retained so
    /// existing primary-key lookups still hit; only the dedup contract
    /// expires.
    fn sweep_expired_idempotency_keys(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize, Error>;
```

- [ ] **Step 2.4: Implement on `InMemoryTaskStore`**

Edit `impl TaskStore for InMemoryTaskStore` (around line 208). Add at the end of the impl block:

```rust
    fn find_by_idempotency_key(
        &self,
        tenant_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<DaemonTask>, Error> {
        let guard = self.tasks.read().map_err(|_| Error::Daemon("task store poisoned".into()))?;
        Ok(guard.values().find(|t| {
            t.tenant_id.as_deref().unwrap_or("") == tenant_id
                && t.idempotency_key.as_deref() == Some(idempotency_key)
        }).cloned())
    }

    fn sweep_expired_idempotency_keys(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize, Error> {
        let mut guard = self.tasks.write().map_err(|_| Error::Daemon("task store poisoned".into()))?;
        let mut count = 0usize;
        for task in guard.values_mut() {
            if task.idempotency_key.is_some() && task.created_at < cutoff {
                task.idempotency_key = None;
                count += 1;
            }
        }
        Ok(count)
    }
```

- [ ] **Step 2.5: Implement on `PostgresTaskStore`**

Edit `impl TaskStore for PostgresTaskStore` (around line 452). Add at the end:

```rust
        fn find_by_idempotency_key(
            &self,
            tenant_id: &str,
            idempotency_key: &str,
        ) -> Result<Option<DaemonTask>, Error> {
            let pool = self.pool.clone();
            let tenant = tenant_id.to_string();
            let key = idempotency_key.to_string();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    let row: Option<TaskRow> = sqlx::query_as(
                        "SELECT id, task, state, created_at, started_at, completed_at, result, error, \
                         input_tokens, output_tokens, cache_creation_input_tokens, \
                         cache_read_input_tokens, reasoning_tokens, tool_calls_made, \
                         estimated_cost_usd, source, agent_name, user_id, tenant_id, model_name, \
                         idempotency_key \
                         FROM daemon_tasks \
                         WHERE tenant_id = $1 AND idempotency_key = $2 \
                         LIMIT 1",
                    )
                    .bind(&tenant)
                    .bind(&key)
                    .fetch_optional(&pool)
                    .await
                    .map_err(|e| Error::Daemon(format!("idempotency lookup failed: {e}")))?;
                    Ok(row.map(DaemonTask::from))
                })
            })
        }

        fn sweep_expired_idempotency_keys(
            &self,
            cutoff: chrono::DateTime<chrono::Utc>,
        ) -> Result<usize, Error> {
            let pool = self.pool.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    let result = sqlx::query(
                        "UPDATE daemon_tasks SET idempotency_key = NULL \
                         WHERE idempotency_key IS NOT NULL AND created_at < $1",
                    )
                    .bind(cutoff)
                    .execute(&pool)
                    .await
                    .map_err(|e| Error::Daemon(format!("idempotency sweep failed: {e}")))?;
                    Ok(result.rows_affected() as usize)
                })
            })
        }
```

- [ ] **Step 2.6: Add the `is_unique_violation` helper**

Add at the top of `crates/heartbit/src/daemon/store.rs` (before the trait definition):

```rust
/// Detect a Postgres unique-constraint violation by inspecting the error
/// message for the `23505` SQLSTATE code that sqlx surfaces. Used by the
/// idempotency-key insert flow to recover from concurrent inserts of the
/// same `(tenant_id, idempotency_key)` pair.
pub(crate) fn is_unique_violation(err: &Error) -> bool {
    let msg = err.to_string().to_lowercase();
    msg.contains("23505") || msg.contains("duplicate key value violates unique constraint")
}
```

- [ ] **Step 2.7: Run tests to verify they pass**

```bash
cargo test -p heartbit --lib daemon::store -- --nocapture
```

Expected: all tests pass.

- [ ] **Step 2.8: Run quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings
```

Expected: no warnings.

- [ ] **Step 2.9: Commit**

```bash
git add crates/heartbit/src/daemon/store.rs
git commit -m "$(cat <<'EOF'
feat(daemon): TaskStore::find_by_idempotency_key + sweep + unique-violation helper

- TaskStore trait gains find_by_idempotency_key (tenant-scoped lookup) and
  sweep_expired_idempotency_keys (NULLs out keys past TTL).
- InMemoryTaskStore: HashMap walk + std::sync::RwLock guard.
- PostgresTaskStore: tenant-scoped SELECT and a single bulk UPDATE for the sweep.
- is_unique_violation(&Error) recognizes SQLSTATE 23505 from sqlx error text;
  used by the lookup-then-insert-or-fallback idempotency flow.
EOF
)"
```

---

## Task 3: Wire idempotency into `DaemonHandle::submit_task_with_user`

**Goal:** Idempotency-aware submission flow. When a key is supplied, lookup-first; if missing, insert with key; on unique violation, fall back to lookup. The Kafka publish only fires for fresh inserts.

**Files:**
- Modify: `crates/heartbit/src/daemon/types.rs` (`DaemonCommand::SubmitTask` gains `idempotency_key: Option<String>`)
- Modify: `crates/heartbit/src/daemon/core.rs:118-165` (`submit_task_with_user`)

**TDD steps:**

- [ ] **Step 3.1: Write failing tests for the wire format change**

Add to the test module of `crates/heartbit/src/daemon/types.rs`:

```rust
#[test]
fn submit_task_serializes_idempotency_key_when_present() {
    let cmd = DaemonCommand::SubmitTask {
        id: Uuid::nil(),
        task: "x".into(),
        source: "test".into(),
        story_id: None,
        trust_level: None,
        user_id: None,
        tenant_id: None,
        roles: vec![],
        mcp_auth_tokens: None,
        idempotency_key: Some("idem-zzz".into()),
    };
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("idempotency_key"), "json was: {json}");
    assert!(json.contains("idem-zzz"));
}

#[test]
fn submit_task_omits_idempotency_key_when_absent() {
    let cmd = DaemonCommand::SubmitTask {
        id: Uuid::nil(),
        task: "x".into(),
        source: "test".into(),
        story_id: None,
        trust_level: None,
        user_id: None,
        tenant_id: None,
        roles: vec![],
        mcp_auth_tokens: None,
        idempotency_key: None,
    };
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(!json.contains("idempotency_key"), "json was: {json}");
}

#[test]
fn submit_task_legacy_payload_without_idempotency_key_deserializes() {
    let legacy = r#"{
        "type": "submit_task",
        "id": "00000000-0000-0000-0000-000000000000",
        "task": "x",
        "source": "test"
    }"#;
    let cmd: DaemonCommand = serde_json::from_str(legacy).unwrap();
    if let DaemonCommand::SubmitTask { idempotency_key, .. } = cmd {
        assert!(idempotency_key.is_none());
    } else {
        panic!("wrong variant");
    }
}
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
cargo test -p heartbit --lib daemon::types::tests::submit_task -- --nocapture
```

Expected: 3 failures with "no field `idempotency_key`".

- [ ] **Step 3.3: Add the field to `DaemonCommand::SubmitTask`**

Edit `crates/heartbit/src/daemon/types.rs:16-40`. Add at the end of the variant (after `mcp_auth_tokens`, before the closing brace):

```rust
        /// Stripe-style idempotency key, scoped to `(tenant_id, idempotency_key)`.
        /// Used by the daemon to dedup Kafka redeliveries and HTTP retries.
        /// `None` = no dedup (each submission creates a new task).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        idempotency_key: Option<String>,
```

Update **every** existing call site that constructs `DaemonCommand::SubmitTask { ... }`. Use grep to find them:

```bash
grep -rn "DaemonCommand::SubmitTask {" crates/ --include="*.rs"
```

Add `idempotency_key: None,` to each constructor literal. Expected sites:
- `crates/heartbit/src/daemon/core.rs:85` (in `submit_task`)
- `crates/heartbit/src/daemon/core.rs:140` (in `submit_task_with_user`)
- Various test fixtures in `crates/heartbit/src/daemon/types.rs` and `crates/heartbit/src/daemon/core.rs`
- Any CLI / openai_compat shim sites

- [ ] **Step 3.4: Update the Kafka match arm to read `idempotency_key`**

Edit `crates/heartbit/src/daemon/core.rs:472`. The existing arm destructures the variant for handling. Add `idempotency_key` to the destructure pattern (currently it's `id, task, source, story_id, trust_level, user_id, tenant_id, roles, mcp_auth_tokens`):

```rust
DaemonCommand::SubmitTask {
    id, task, source, story_id, trust_level, user_id, tenant_id, roles, mcp_auth_tokens,
    idempotency_key,
} => {
```

The body uses `idempotency_key` only to set it on the `DaemonTask` when re-creating. Locate the `DaemonTask::new_with_user(...)` and `DaemonTask::new(...)` constructions inside this match arm. After the construction, set the field:

```rust
                    let mut daemon_task = DaemonTask::new_with_user(
                        id, &task, &source, user_id_val, tenant_id_val,
                    );
                    daemon_task.idempotency_key = idempotency_key.clone();
                    let _ = self.store.insert(daemon_task);
```

(Apply the same to the `DaemonTask::new(id, &task, &source)` branch.)

- [ ] **Step 3.5: Run tests to verify wire-format tests pass**

```bash
cargo test -p heartbit --lib daemon::types -- --nocapture
```

Expected: all daemon::types tests pass.

- [ ] **Step 3.6: Write a failing integration test for the submit-with-idem flow**

Add to `crates/heartbit/src/daemon/core.rs` test module (search `mod tests` near the bottom):

```rust
    #[tokio::test(flavor = "multi_thread")]
    async fn submit_with_idem_returns_same_task_id_on_redelivery() {
        // Build a daemon handle with in-memory store + channel producer (no Kafka).
        let (handle, _core_task) = test_helper_make_handle().await;

        let user_ctx = super::super::types::UserContext {
            user_id: "user-1".into(),
            tenant_id: "tenant-A".into(),
            roles: vec![],
        };

        let id1 = handle
            .submit_task_with_user_idem("hello", "test", None, &user_ctx, Some("idem-xyz"))
            .await
            .expect("first submit");

        let id2 = handle
            .submit_task_with_user_idem("hello again", "test", None, &user_ctx, Some("idem-xyz"))
            .await
            .expect("redelivery");

        assert_eq!(id1, id2, "redelivery must dedup to same task id");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_with_idem_isolates_per_tenant() {
        let (handle, _core_task) = test_helper_make_handle().await;

        let user_a = super::super::types::UserContext {
            user_id: "u".into(), tenant_id: "tenant-A".into(), roles: vec![],
        };
        let user_b = super::super::types::UserContext {
            user_id: "u".into(), tenant_id: "tenant-B".into(), roles: vec![],
        };

        let id_a = handle
            .submit_task_with_user_idem("x", "test", None, &user_a, Some("shared-key"))
            .await.unwrap();
        let id_b = handle
            .submit_task_with_user_idem("x", "test", None, &user_b, Some("shared-key"))
            .await.unwrap();

        assert_ne!(id_a, id_b, "different tenants must NOT collide on same key");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_without_idem_creates_new_task_each_time() {
        let (handle, _core_task) = test_helper_make_handle().await;
        let user_ctx = super::super::types::UserContext {
            user_id: "u".into(), tenant_id: "tenant-A".into(), roles: vec![],
        };

        let id1 = handle
            .submit_task_with_user_idem("x", "test", None, &user_ctx, None)
            .await.unwrap();
        let id2 = handle
            .submit_task_with_user_idem("x", "test", None, &user_ctx, None)
            .await.unwrap();

        assert_ne!(id1, id2, "no idem key → no dedup");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_submits_same_key_resolve_to_same_id() {
        // Tests the unique-violation fallback path. Fires N concurrent submits
        // with the same (tenant, key); all must resolve to the same task id.
        let (handle, _core_task) = test_helper_make_handle().await;
        let user_ctx = std::sync::Arc::new(super::super::types::UserContext {
            user_id: "u".into(), tenant_id: "tenant-A".into(), roles: vec![],
        });
        let handle = std::sync::Arc::new(handle);

        let mut joinset = tokio::task::JoinSet::new();
        for _ in 0..10 {
            let h = handle.clone();
            let u = user_ctx.clone();
            joinset.spawn(async move {
                h.submit_task_with_user_idem("payload", "test", None, &u, Some("race-key"))
                    .await
            });
        }

        let mut ids = std::collections::HashSet::new();
        while let Some(res) = joinset.join_next().await {
            ids.insert(res.unwrap().unwrap());
        }
        assert_eq!(ids.len(), 1, "all concurrent submits must dedup to one id, got {ids:?}");
    }
```

You will need a `test_helper_make_handle()` function. Add at the top of the test module (or reuse an existing helper if one exists — search for `fn make_handle` or similar in the test module):

```rust
    async fn test_helper_make_handle() -> (super::DaemonHandle, tokio::task::JoinHandle<()>) {
        use crate::daemon::store::InMemoryTaskStore;
        let store: std::sync::Arc<dyn super::TaskStore> = std::sync::Arc::new(InMemoryTaskStore::new());
        let cancel = tokio_util::sync::CancellationToken::new();
        let config = crate::config::DaemonConfig::default();
        let (consumer, producer) = crate::daemon::kafka::channel_pair_for_tests();
        let (_core, handle) = super::DaemonCore::new(&config, consumer, producer, store, cancel);
        let core_task = tokio::spawn(async move { /* core not needed for submit-only tests */ });
        (handle, core_task)
    }
```

(If a helper already exists, reuse it instead of redefining. Search the file first.)

- [ ] **Step 3.7: Run failing tests**

```bash
cargo test -p heartbit --lib daemon::core::tests::submit_with_idem -- --nocapture
```

Expected: failures — `submit_task_with_user_idem` not yet implemented.

- [ ] **Step 3.8: Implement `submit_task_with_user_idem`**

Add to `crates/heartbit/src/daemon/core.rs` inside `impl DaemonHandle`, after `submit_task_with_user` (around line 165):

```rust
    /// Like `submit_task_with_user` but dedups on `idempotency_key`.
    /// When `idempotency_key` is supplied and an existing task matches,
    /// returns the existing task id without publishing a new Kafka message
    /// or creating a duplicate row. When omitted, behaves identically to
    /// `submit_task_with_user`.
    pub async fn submit_task_with_user_idem(
        &self,
        task: impl Into<String>,
        source: impl Into<String>,
        story_id: Option<String>,
        user_context: &super::types::UserContext,
        idempotency_key: Option<&str>,
    ) -> Result<uuid::Uuid, Error> {
        let task_str = task.into();
        let source_str = source.into();

        // Idempotency lookup-then-insert path
        if let Some(key) = idempotency_key {
            if let Some(existing) = self
                .store
                .find_by_idempotency_key(&user_context.tenant_id, key)?
            {
                return Ok(existing.id);
            }

            let id = uuid::Uuid::new_v4();
            let mut daemon_task = DaemonTask::new_with_user(
                id,
                &task_str,
                &source_str,
                &user_context.user_id,
                &user_context.tenant_id,
            );
            daemon_task.idempotency_key = Some(key.to_string());

            match self.store.insert(daemon_task) {
                Ok(()) => {
                    self.publish_submit(
                        id,
                        task_str,
                        source_str,
                        story_id,
                        user_context,
                        Some(key.to_string()),
                    )
                    .await?;
                    Ok(id)
                }
                Err(e) if super::store::is_unique_violation(&e) => {
                    // Concurrent inserter raced ahead. Resolve to their id.
                    self.store
                        .find_by_idempotency_key(&user_context.tenant_id, key)?
                        .map(|t| t.id)
                        .ok_or_else(|| {
                            Error::Daemon("unique violation but row not found".into())
                        })
                }
                Err(e) => Err(e),
            }
        } else {
            // Existing path: no key, no dedup
            self.submit_task_with_user(task_str, source_str, story_id, user_context).await
        }
    }

    async fn publish_submit(
        &self,
        id: uuid::Uuid,
        task: String,
        source: String,
        story_id: Option<String>,
        user_context: &super::types::UserContext,
        idempotency_key: Option<String>,
    ) -> Result<(), Error> {
        let (producer, commands_topic) = match self.require_kafka() {
            Ok(p) => p,
            // HTTP-only mode: no Kafka publish needed; the task is already in store.
            // Caller is responsible for triggering execution another way.
            Err(_) => return Ok(()),
        };
        let cmd = DaemonCommand::SubmitTask {
            id,
            task,
            source,
            story_id,
            trust_level: None,
            user_id: Some(user_context.user_id.clone()),
            tenant_id: Some(user_context.tenant_id.clone()),
            roles: user_context.roles.clone(),
            mcp_auth_tokens: None,
            idempotency_key,
        };
        let payload = serde_json::to_vec(&cmd)
            .map_err(|e| Error::Daemon(format!("failed to serialize command: {e}")))?;
        producer
            .send(
                FutureRecord::to(commands_topic)
                    .key(&id.to_string())
                    .payload(&payload),
                rdkafka::util::Timeout::Never,
            )
            .await
            .map_err(|(e, _)| Error::Daemon(format!("failed to produce command: {e}")))?;
        Ok(())
    }
```

- [ ] **Step 3.9: Run tests to verify they pass**

```bash
cargo test -p heartbit --lib daemon::core::tests::submit_with_idem -- --nocapture
```

Expected: both tests pass.

- [ ] **Step 3.10: Run quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --lib
```

Expected: green across the workspace.

- [ ] **Step 3.11: Commit**

```bash
git add crates/heartbit/src/daemon/types.rs crates/heartbit/src/daemon/core.rs
git commit -m "$(cat <<'EOF'
feat(daemon): submit_task_with_user_idem dedups on (tenant_id, key)

- DaemonCommand::SubmitTask gains idempotency_key: Option<String> with
  serde default + skip_serializing_if so existing payloads still parse.
- DaemonHandle::submit_task_with_user_idem: lookup-first; on miss, insert
  with key; on unique violation, fall back to lookup (concurrent inserter
  won the race).
- Tenant-scoped: same key under different tenants creates two separate tasks.
EOF
)"
```

---

## Task 4: HTTP API + TTL sweep background task

**Goal:** Wire the `Idempotency-Key` HTTP header through to `submit_task_with_user_idem`, and start the TTL sweep background task on daemon startup.

**Files:**
- Modify: `crates/heartbit/src/daemon/openai_compat.rs` (HTTP handler — search for the SubmitTask path)
- Modify: `crates/heartbit/src/daemon/core.rs` (`DaemonCore::run_kafka` and `run_channel` — start sweep task)
- Modify: `crates/heartbit/src/config.rs` (new `IdempotencyConfig` section)

**TDD steps:**

- [ ] **Step 4.1: Find the HTTP submit handler**

```bash
grep -n "submit_task_with_user\|fn submit_handler\|POST.*tasks\|axum::routing" crates/heartbit/src/daemon/openai_compat.rs | head -20
```

Note the function name and signature. The plan assumes the handler extracts a `UserContext` from JWT and constructs the body. If it lives in a different file, adjust the file path below.

- [ ] **Step 4.2: Add the config section**

Edit `crates/heartbit/src/config.rs`. Add a new struct:

```rust
/// Idempotency-key sweep settings. When `ttl_hours` is `Some`, the daemon
/// runs a background task that nulls out keys older than the TTL.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdempotencyConfig {
    /// Hours to retain idempotency keys before the sweep nulls them out.
    /// Default `Some(24)` matches the Stripe contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_hours: Option<u32>,
    /// How often the sweep runs, in minutes. Default 60.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sweep_interval_minutes: Option<u32>,
}
```

Add a field to `DaemonConfig`:

```rust
    #[serde(default)]
    pub idempotency: IdempotencyConfig,
```

Add validation in `DaemonConfig::validate` (or wherever zero-rejection happens for `prune_interval_minutes`):

```rust
    if let Some(0) = self.idempotency.ttl_hours {
        return Err(Error::Config("daemon.idempotency.ttl_hours must be > 0".into()));
    }
    if let Some(0) = self.idempotency.sweep_interval_minutes {
        return Err(Error::Config("daemon.idempotency.sweep_interval_minutes must be > 0".into()));
    }
```

- [ ] **Step 4.3: Write the failing config test**

Add to `crates/heartbit/src/config.rs` test module:

```rust
#[test]
fn idempotency_config_defaults_are_none() {
    let toml = "";
    let cfg: IdempotencyConfig = toml::from_str(toml).unwrap();
    assert!(cfg.ttl_hours.is_none());
    assert!(cfg.sweep_interval_minutes.is_none());
}

#[test]
fn idempotency_config_zero_rejected() {
    let toml = r#"
[idempotency]
ttl_hours = 0
"#;
    let cfg: DaemonConfig = toml::from_str(toml).unwrap();
    assert!(cfg.validate().is_err(), "zero ttl_hours must be rejected");
}
```

Run, verify it fails, implement, verify it passes.

- [ ] **Step 4.4: Add the sweep background task to `DaemonCore::run`**

Locate the place where existing background tasks are spawned (the audit-prune task from B4 Task 7 — search `prune_audit` or `prune_interval`):

```bash
grep -n "prune_interval\|prune_audit\|tokio::spawn.*prune" crates/heartbit/src/daemon/core.rs | head -10
```

Add a similar block right after the existing prune-task spawn, inside `DaemonCore::run_kafka` and `run_channel` (or whichever shared startup function exists — there should be a single one):

```rust
        // B5b: idempotency-key TTL sweep
        if let Some(ttl_hours) = self.config.idempotency.ttl_hours {
            let store = self.store.clone();
            let cancel = self.cancel.clone();
            let interval_min = self.config.idempotency.sweep_interval_minutes.unwrap_or(60);
            let interval = std::time::Duration::from_secs(u64::from(interval_min) * 60);
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(interval);
                tick.tick().await; // skip immediate fire
                loop {
                    tokio::select! {
                        _ = cancel.cancelled() => break,
                        _ = tick.tick() => {
                            let cutoff = chrono::Utc::now() - chrono::Duration::hours(i64::from(ttl_hours));
                            match store.sweep_expired_idempotency_keys(cutoff) {
                                Ok(n) if n > 0 => tracing::info!(swept = n, "idempotency keys expired"),
                                Ok(_) => {}
                                Err(e) => tracing::warn!(error = %e, "idempotency sweep failed"),
                            }
                        }
                    }
                }
            });
        }
```

- [ ] **Step 4.5: Wire `Idempotency-Key` HTTP header**

Edit `crates/heartbit/src/daemon/openai_compat.rs` at the submit-task handler. Read the header off the request, then pass it to `submit_task_with_user_idem`:

```rust
use axum::http::HeaderMap;

async fn submit_task_handler(
    headers: HeaderMap,
    /* existing params */,
) -> Result<...> {
    let idempotency_key = headers
        .get("Idempotency-Key")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    let id = handle
        .submit_task_with_user_idem(
            body.task,
            body.source.unwrap_or_else(|| "http".into()),
            body.story_id,
            &user_ctx,
            idempotency_key.as_deref(),
        )
        .await?;
    Ok(/* existing response */)
}
```

Use the existing handler's structure — only read the header and replace the call to `submit_task_with_user`. Don't restructure unrelated code.

- [ ] **Step 4.6: Write a failing HTTP-level test (optional — depends on existing patterns)**

If there's an existing axum `TestServer` or similar pattern in the file, mirror it:

```rust
#[tokio::test]
async fn http_idempotency_key_header_dedups() {
    let server = test_server();
    let resp1 = server
        .post("/v1/tasks")
        .add_header("Idempotency-Key", "test-1")
        .json(&serde_json::json!({"task": "hello"}))
        .await;
    let id1 = resp1.json::<serde_json::Value>()["id"].as_str().unwrap().to_string();

    let resp2 = server
        .post("/v1/tasks")
        .add_header("Idempotency-Key", "test-1")
        .json(&serde_json::json!({"task": "hello again"}))
        .await;
    let id2 = resp2.json::<serde_json::Value>()["id"].as_str().unwrap().to_string();

    assert_eq!(id1, id2);
}
```

If `test_server()` doesn't exist or the surrounding handler test infra is heavy to set up, skip this test and rely on the unit tests from Task 3. Note this in the commit message.

- [ ] **Step 4.7: Run quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --lib
```

Expected: green.

- [ ] **Step 4.8: Commit**

```bash
git add crates/heartbit/src/daemon/openai_compat.rs crates/heartbit/src/daemon/core.rs crates/heartbit/src/config.rs
git commit -m "$(cat <<'EOF'
feat(daemon): Idempotency-Key HTTP header + TTL sweep

- POST /v1/tasks reads `Idempotency-Key` header and threads it through
  submit_task_with_user_idem. Body field also accepted (HTTP convention).
- DaemonConfig.idempotency = IdempotencyConfig { ttl_hours, sweep_interval_minutes }
  Both Option<u32> with zero-value rejection at config validate.
- DaemonCore::run spawns a sweep task that nulls expired keys every
  `sweep_interval_minutes` (default 60). Cancellation-token aware.
EOF
)"
```

---

## Task 5: `TenantTokenTracker` + `Error::TenantOverloaded`

**Goal:** New tracker type with `Arc`-owning RAII reservation, per-tenant cap, and `adjust(delta: i64)` for per-turn reconciliation.

**Files:**
- Create: `crates/heartbit-core/src/agent/tenant_tracker.rs`
- Modify: `crates/heartbit-core/src/agent/mod.rs` (re-export)
- Modify: `crates/heartbit-core/src/error.rs` (new `Error::TenantOverloaded`)
- Modify: `crates/heartbit-core/src/lib.rs` (re-export)

**TDD steps:**

- [ ] **Step 5.1: Add the error variant**

Edit `crates/heartbit-core/src/error.rs`. Add to the `Error` enum (alphabetical or near other agent-state errors):

```rust
    #[error("tenant {tenant_id} overloaded: in_flight={in_flight}, cap={cap}")]
    TenantOverloaded {
        tenant_id: String,
        in_flight: usize,
        cap: usize,
    },
```

- [ ] **Step 5.2: Write the failing tracker tests**

Create `crates/heartbit-core/src/agent/tenant_tracker.rs`:

```rust
//! Per-tenant in-flight token tracker with Arc-owning RAII reservations.
//!
//! See `docs/superpowers/specs/2026-05-02-b5b-failure-mode-hardening-design.md`
//! Component 2 for design rationale.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::auth::TenantScope;
use crate::error::Error;

#[derive(Debug, Default, Clone)]
pub struct TenantTokenState {
    pub in_flight: usize,
    pub high_water: usize,
}

pub struct TenantTokenTracker {
    states: RwLock<HashMap<String, TenantTokenState>>,
    per_tenant_cap: usize,
}

pub struct TokenReservation {
    tracker: Arc<TenantTokenTracker>,
    tenant_id: String,
    tokens: usize,
}

impl Drop for TokenReservation {
    fn drop(&mut self) {
        self.tracker.release(&self.tenant_id, self.tokens);
    }
}

impl TenantTokenTracker {
    pub fn new(per_tenant_cap: usize) -> Self {
        Self {
            states: RwLock::new(HashMap::new()),
            per_tenant_cap,
        }
    }

    pub fn reserve(
        self: &Arc<Self>,
        scope: &TenantScope,
        tokens: usize,
    ) -> Result<TokenReservation, Error> {
        let tenant = scope.tenant_id.clone();
        let mut guard = self
            .states
            .write()
            .map_err(|_| Error::Agent("token tracker poisoned".into()))?;
        let state = guard.entry(tenant.clone()).or_default();
        if state.in_flight.saturating_add(tokens) > self.per_tenant_cap {
            return Err(Error::TenantOverloaded {
                tenant_id: tenant,
                in_flight: state.in_flight,
                cap: self.per_tenant_cap,
            });
        }
        state.in_flight += tokens;
        if state.in_flight > state.high_water {
            state.high_water = state.in_flight;
        }
        Ok(TokenReservation {
            tracker: Arc::clone(self),
            tenant_id: tenant,
            tokens,
        })
    }

    pub fn adjust(&self, scope: &TenantScope, delta: i64) {
        let Ok(mut guard) = self.states.write() else { return; };
        let Some(state) = guard.get_mut(&scope.tenant_id) else { return; };
        if delta >= 0 {
            state.in_flight = state
                .in_flight
                .saturating_add(delta as usize)
                .min(self.per_tenant_cap);
        } else {
            state.in_flight = state.in_flight.saturating_sub((-delta) as usize);
        }
        if state.in_flight > state.high_water {
            state.high_water = state.in_flight;
        }
    }

    fn release(&self, tenant_id: &str, tokens: usize) {
        if let Ok(mut guard) = self.states.write() {
            if let Some(state) = guard.get_mut(tenant_id) {
                state.in_flight = state.in_flight.saturating_sub(tokens);
            }
        }
    }

    pub fn snapshot(&self) -> Vec<(String, TenantTokenState)> {
        match self.states.read() {
            Ok(g) => g.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            Err(_) => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(t: &str) -> TenantScope {
        TenantScope::new(t)
    }

    #[test]
    fn reserve_within_cap_succeeds() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let r = t.reserve(&scope("a"), 500).unwrap();
        let snap = t.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].1.in_flight, 500);
        drop(r);
    }

    #[test]
    fn reserve_exceeding_cap_returns_tenant_overloaded() {
        let t = Arc::new(TenantTokenTracker::new(100));
        let _r = t.reserve(&scope("a"), 80).unwrap();
        let err = t.reserve(&scope("a"), 50).unwrap_err();
        match err {
            Error::TenantOverloaded { tenant_id, in_flight, cap } => {
                assert_eq!(tenant_id, "a");
                assert_eq!(in_flight, 80);
                assert_eq!(cap, 100);
            }
            other => panic!("expected TenantOverloaded, got {other:?}"),
        }
    }

    #[test]
    fn drop_releases_reservation() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        {
            let _r = t.reserve(&scope("a"), 700).unwrap();
            assert_eq!(t.snapshot()[0].1.in_flight, 700);
        }
        assert_eq!(t.snapshot()[0].1.in_flight, 0);
    }

    #[test]
    fn tenants_are_isolated() {
        let t = Arc::new(TenantTokenTracker::new(100));
        let _ra = t.reserve(&scope("a"), 90).unwrap();
        let _rb = t.reserve(&scope("b"), 90).unwrap();
        let snap: HashMap<_, _> = t.snapshot().into_iter().collect();
        assert_eq!(snap["a"].in_flight, 90);
        assert_eq!(snap["b"].in_flight, 90);
    }

    #[test]
    fn high_water_tracks_peak() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let r1 = t.reserve(&scope("a"), 400).unwrap();
        let r2 = t.reserve(&scope("a"), 300).unwrap();
        drop(r1);
        let snap = t.snapshot();
        assert_eq!(snap[0].1.in_flight, 300);
        assert_eq!(snap[0].1.high_water, 700);
        drop(r2);
    }

    #[test]
    fn adjust_positive_delta_clamps_at_cap() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let _r = t.reserve(&scope("a"), 500).unwrap();
        t.adjust(&scope("a"), 800);
        assert_eq!(t.snapshot()[0].1.in_flight, 1000); // clamped
    }

    #[test]
    fn adjust_negative_delta_decrements() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let _r = t.reserve(&scope("a"), 500).unwrap();
        t.adjust(&scope("a"), -200);
        assert_eq!(t.snapshot()[0].1.in_flight, 300);
    }

    #[test]
    fn reservation_owns_arc_and_outlives_borrow() {
        // Compile-time check: TokenReservation can be moved into a future.
        let t = Arc::new(TenantTokenTracker::new(1000));
        let r = t.reserve(&scope("a"), 500).unwrap();
        let _: tokio::task::JoinHandle<()> = tokio::task::spawn_blocking(move || {
            drop(r);
        });
    }

    #[test]
    fn default_scope_uses_empty_string_bucket() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        let _r = t.reserve(&TenantScope::default(), 500).unwrap();
        let snap = t.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].0, ""); // empty-string sentinel
    }

    #[test]
    fn adjust_on_unknown_tenant_is_noop() {
        let t = Arc::new(TenantTokenTracker::new(1000));
        t.adjust(&scope("unknown"), -100);
        assert!(t.snapshot().is_empty());
    }
}
```

- [ ] **Step 5.3: Wire the module**

Edit `crates/heartbit-core/src/agent/mod.rs`:

```rust
pub mod tenant_tracker;
```

Edit `crates/heartbit-core/src/lib.rs`:

```rust
pub use agent::tenant_tracker::{TenantTokenTracker, TenantTokenState, TokenReservation};
```

- [ ] **Step 5.4: Run tests to verify they pass**

```bash
cargo test -p heartbit-core --lib agent::tenant_tracker -- --nocapture
```

Expected: 10 tests pass.

- [ ] **Step 5.5: Run quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 5.6: Commit**

```bash
git add crates/heartbit-core/src/agent/tenant_tracker.rs crates/heartbit-core/src/agent/mod.rs crates/heartbit-core/src/lib.rs crates/heartbit-core/src/error.rs
git commit -m "$(cat <<'EOF'
feat(core): TenantTokenTracker with Arc-owning RAII reservation

- per-tenant in-flight cap with HashMap<TenantId, State>
- TokenReservation owns Arc<TenantTokenTracker> (not lifetime borrow) so
  it can be moved across .await points and tokio::spawn boundaries
- adjust(delta: i64) for per-turn reconciliation (positive saturates at cap,
  negative saturates at 0)
- Error::TenantOverloaded for overload signaling
EOF
)"
```

---

## Task 6: Wire `TenantTokenTracker` into `AgentRunner` per-turn loop

**Goal:** `AgentRunnerBuilder::tenant_tracker()` accepts an `Arc<TenantTokenTracker>`. When set, the runner calls `tracker.adjust(scope, delta)` after each LLM response. The runner does NOT call `reserve()` — the daemon owns submission gating.

**Files:**
- Modify: `crates/heartbit-core/src/agent/builder.rs` (new builder method + field)
- Modify: `crates/heartbit-core/src/agent/runner.rs` (per-turn adjust call)

**TDD steps:**

- [ ] **Step 6.1: Write a failing test in `crates/heartbit-core/src/agent/runner.rs`**

Add at the bottom of the existing test module (search `mod tests` near end of file). The test verifies that per-turn adjust is called by injecting a `TenantTokenTracker` and a mock provider that returns a known token usage:

```rust
    #[tokio::test(flavor = "multi_thread")]
    async fn agent_runner_adjusts_tenant_tracker_per_turn() {
        use std::sync::Arc;
        use crate::agent::tenant_tracker::TenantTokenTracker;
        use crate::auth::TenantScope;

        let tracker = Arc::new(TenantTokenTracker::new(1_000_000));
        let scope = TenantScope::new("acme");
        // Pre-reserve like the daemon would at submit time.
        let _initial = tracker.reserve(&scope, 5000).unwrap();
        assert_eq!(tracker.snapshot()[0].1.in_flight, 5000);

        // Build a fake provider that returns a known TokenUsage in one turn,
        // then stops with EndTurn.
        let provider = OneShotProvider::new(TokenUsage {
            input_tokens: 100,
            output_tokens: 200,
            ..Default::default()
        });
        let runner = AgentRunner::builder()
            .name("test")
            .system_prompt("test")
            .provider(provider)
            .max_turns(1)
            .audit_tenant_id("acme")
            .tenant_tracker(tracker.clone())
            .build()
            .unwrap();
        let _output = runner.run("hello").await.unwrap();

        // After one turn: in_flight should reconcile from 5000 (estimate)
        // to 300 (actual = 100 input + 200 output).
        let snap = tracker.snapshot();
        assert_eq!(snap[0].1.in_flight, 300);
    }
```

You will need a `OneShotProvider`. If a similar mock already exists in the same test module (search for `FakeProvider` or `CountingProvider` in `crates/heartbit-core/src/llm/mod.rs:255+`), reuse it. Otherwise add a minimal one to the test module:

```rust
    struct OneShotProvider {
        usage: TokenUsage,
        called: std::sync::atomic::AtomicBool,
    }

    impl OneShotProvider {
        fn new(usage: TokenUsage) -> Self {
            Self { usage, called: std::sync::atomic::AtomicBool::new(false) }
        }
    }

    impl LlmProvider for OneShotProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            self.called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text { text: "done".into() }],
                stop_reason: StopReason::EndTurn,
                usage: self.usage,
                model: None,
            })
        }
    }
```

- [ ] **Step 6.2: Run test to verify it fails**

```bash
cargo test -p heartbit-core --lib agent::runner::tests::agent_runner_adjusts_tenant_tracker_per_turn -- --nocapture
```

Expected: failure with "method `tenant_tracker` not found" or "method `audit_tenant_id` not found".

(If `audit_tenant_id` builder method is missing: search `audit_tenant_id` in `crates/heartbit-core/src/agent/builder.rs` and `runner.rs` to confirm. The B4 work added these fields directly on the runner; the builder method may already exist. Verify before adding.)

- [ ] **Step 6.3: Add the builder field**

Edit `crates/heartbit-core/src/agent/builder.rs`. Add a field on `AgentRunnerBuilder`:

```rust
    tenant_tracker: Option<Arc<crate::agent::tenant_tracker::TenantTokenTracker>>,
```

Add the builder method near the other tracker/observability methods (alphabetical or near `observability_mode`):

```rust
    /// Optional per-tenant in-flight token tracker. When set, the runner
    /// calls `tracker.adjust(&scope, delta)` after each LLM response,
    /// reconciling the per-tenant `in_flight` counter against the
    /// estimated reservation made at submit time. Has no effect when
    /// `audit_tenant_id` is unset.
    pub fn tenant_tracker(
        mut self,
        tracker: Arc<crate::agent::tenant_tracker::TenantTokenTracker>,
    ) -> Self {
        self.tenant_tracker = Some(tracker);
        self
    }
```

Also initialize the field in `Default` / `new` (search the existing default initialization block in `builder.rs`).

- [ ] **Step 6.4: Thread it through to `AgentRunner`**

Edit `crates/heartbit-core/src/agent/runner.rs`. Add a field on `AgentRunner`:

```rust
    tenant_tracker: Option<Arc<crate::agent::tenant_tracker::TenantTokenTracker>>,
    cumulative_actual_tokens: AtomicUsize,
```

Initialize from the builder in `build()`. Then, in the per-turn loop (search for the place where `total_usage += response.usage` happens — likely around `crates/heartbit-core/src/agent/runner.rs:680-700`), add the adjust call:

```rust
                    if let Some(ref tracker) = self.tenant_tracker {
                        if let Some(ref tid) = self.audit_tenant_id {
                            let scope = crate::auth::TenantScope::new(tid.clone());
                            let actual = (response.usage.input_tokens
                                + response.usage.output_tokens) as usize;
                            let prev = self
                                .cumulative_actual_tokens
                                .swap(actual, std::sync::atomic::Ordering::SeqCst);
                            let delta = actual as i64 - prev as i64;
                            tracker.adjust(&scope, delta);
                        }
                    }
```

`cumulative_actual_tokens` starts at 0; the first turn's `delta` is therefore `actual - 0 = actual`. The daemon's submit-time `reserve()` call (Task 7) drops its reservation **immediately** after the admission check (it does not hold during execution), so the per-turn `adjust(...)` calls are the *only* source of `in_flight` mutation during execution. At task end the Drop on the temporary submit-time reservation already fired (was zero net effect anyway); the per-turn deltas remain on the tracker until the runner exits, and a final `adjust(scope, -actual)` on `Drop` of the runner returns the tenant to baseline. Add an `impl Drop for AgentRunner` block that does this:

```rust
impl Drop for AgentRunner {
    fn drop(&mut self) {
        if let (Some(tracker), Some(tid)) = (
            self.tenant_tracker.as_ref(),
            self.audit_tenant_id.as_ref(),
        ) {
            let actual = self
                .cumulative_actual_tokens
                .load(std::sync::atomic::Ordering::SeqCst) as i64;
            if actual > 0 {
                let scope = crate::auth::TenantScope::new(tid.clone());
                tracker.adjust(&scope, -actual);
            }
        }
    }
}
```

This pattern assumes one runner per task — which matches the daemon's lifecycle (the daemon builds a fresh `AgentRunner` per `SubmitTask` command). If a runner is reused across multiple `run()` calls (e.g., a long-lived chat session), reset `cumulative_actual_tokens` at start of each `run()` and emit a per-call release before the reset. The plan only covers the per-task daemon case; the chat-session refinement can land later if/when needed.

- [ ] **Step 6.5: Update the test to match the deterministic math**

The test's pre-reserved `_initial = 5000` simulates the daemon's submit-time admission check, but Task 7 drops that reservation immediately. So the test needs to drop `_initial` before invoking the runner. Update the test:

```rust
        // Pre-reserve like the daemon's admission check would, then drop it
        // immediately (matching Task 7's "admission-only" pattern).
        drop(tracker.reserve(&scope, 5000).unwrap());

        // Build the runner...
        // After one turn: cumulative_actual_tokens went 0 → 300, so adjust(+300).
        let snap = tracker.snapshot();
        assert_eq!(snap[0].1.in_flight, 300);
```

Run:

```bash
cargo test -p heartbit-core --lib agent::runner::tests::agent_runner_adjusts_tenant_tracker_per_turn -- --nocapture
```

Expected: pass. After Drop fires (when `_output` goes out of scope), `in_flight` returns to 0 — verify in a follow-up test if desired:

```rust
        drop(_output);
        assert_eq!(tracker.snapshot()[0].1.in_flight, 0);
```

- [ ] **Step 6.6: Run quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --lib
```

- [ ] **Step 6.7: Commit**

```bash
git add crates/heartbit-core/src/agent/builder.rs crates/heartbit-core/src/agent/runner.rs
git commit -m "$(cat <<'EOF'
feat(core): AgentRunner per-turn adjusts TenantTokenTracker

- AgentRunnerBuilder::tenant_tracker(Arc<TenantTokenTracker>)
- Runner accumulates cumulative_actual_tokens across turns and emits
  signed deltas to the tracker after each LLM response.
- The runner does NOT call tracker.reserve() — the daemon owns submit-time
  reservation. The runner only reconciles actual usage.
EOF
)"
```

---

## Task 7: Daemon submission gate (Kafka NACK + HTTP 503)

**Goal:** `DaemonHandle::submit_task_with_user_idem` consults `tracker.reserve()` before insert. On `Error::TenantOverloaded`, the HTTP path returns 503 + `Retry-After: 5`. On the Kafka redelivery path, `DaemonCore::dispatch_command` likewise checks; on overload, it NACKs (returns without committing the offset) so Kafka redelivers later.

**Files:**
- Modify: `crates/heartbit/src/daemon/core.rs` (`DaemonHandle.tracker: Option<Arc<TenantTokenTracker>>`, `submit_task_with_user_idem`, kafka match arm)
- Modify: `crates/heartbit/src/daemon/openai_compat.rs` (HTTP 503 mapping)

**TDD steps:**

- [ ] **Step 7.1: Add the optional tracker field on `DaemonHandle`**

Edit `crates/heartbit/src/daemon/core.rs`. Add to `DaemonHandle`:

```rust
    pub(crate) tenant_tracker: Option<std::sync::Arc<heartbit_core::TenantTokenTracker>>,
```

Add a builder-style method to set it:

```rust
    pub fn with_tenant_tracker(
        mut self,
        tracker: std::sync::Arc<heartbit_core::TenantTokenTracker>,
    ) -> Self {
        self.tenant_tracker = Some(tracker);
        self
    }
```

Initialize the field to `None` wherever `DaemonHandle` is currently constructed (search `DaemonHandle {` constructor literals).

- [ ] **Step 7.2: Write the failing submit-with-overload test**

Add to `crates/heartbit/src/daemon/core.rs` test module:

```rust
    #[tokio::test(flavor = "multi_thread")]
    async fn submit_returns_tenant_overloaded_when_tracker_full() {
        let (handle, _core_task) = test_helper_make_handle().await;
        let tracker = std::sync::Arc::new(heartbit_core::TenantTokenTracker::new(100));
        let handle = handle.with_tenant_tracker(tracker.clone());

        // Pre-fill the tracker for tenant-A
        let scope = heartbit_core::TenantScope::new("tenant-A");
        let _hold = tracker.reserve(&scope, 99).unwrap();

        let user_ctx = super::super::types::UserContext {
            user_id: "u".into(), tenant_id: "tenant-A".into(), roles: vec![],
        };
        // Estimate for "hello" = 5 / 4 + 4096 ≈ 4097 → exceeds remaining cap of 1
        let err = handle
            .submit_task_with_user_idem("hello", "test", None, &user_ctx, None)
            .await
            .unwrap_err();
        match err {
            heartbit_core::Error::TenantOverloaded { tenant_id, .. } => {
                assert_eq!(tenant_id, "tenant-A");
            }
            other => panic!("expected TenantOverloaded, got {other:?}"),
        }
    }
```

- [ ] **Step 7.3: Run to verify it fails**

```bash
cargo test -p heartbit --lib daemon::core::tests::submit_returns_tenant_overloaded -- --nocapture
```

Expected: failure (no overload check yet).

- [ ] **Step 7.4: Implement the overload gate in `submit_task_with_user_idem`**

Edit `submit_task_with_user_idem` (Task 3). At the very top of the function (before any store or kafka work):

```rust
        // B5b: per-tenant overload gate
        if let Some(ref tracker) = self.tenant_tracker {
            let scope = heartbit_core::TenantScope::new(&user_context.tenant_id);
            let estimated = task_str.len() / 4 + 4096;
            // Reservation is dropped at end of this function on the happy path
            // because we don't currently propagate it into task execution.
            // For B5b that's acceptable: the runner re-reserves via adjust(...)
            // anyway, and submit-time enforcement is the primary goal.
            let _reservation = tracker.reserve(&scope, estimated)?;
            // Intentionally drop _reservation here. The tracker's adjust(...)
            // path used by AgentRunner is what tracks live usage during the run.
            // Submit-time gate is purely an admission decision.
        }
```

(Per-turn `adjust` is wired in Task 6. The submit-time `reserve` here is purely an admission check; we drop the reservation immediately after the check so the tracker's `in_flight` only reflects in-progress *actual* usage tracked via `adjust`.)

- [ ] **Step 7.5: Run to verify the test passes**

```bash
cargo test -p heartbit --lib daemon::core::tests::submit_returns_tenant_overloaded -- --nocapture
```

Expected: pass.

- [ ] **Step 7.6: Wire HTTP 503 mapping**

Edit `crates/heartbit/src/daemon/openai_compat.rs`. Search for the existing error→HTTP mapping (likely `impl IntoResponse for Error` or a helper inside the crate). Add a branch for `TenantOverloaded`:

```rust
            heartbit_core::Error::TenantOverloaded { tenant_id, in_flight, cap } => (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                [(axum::http::header::RETRY_AFTER, "5".parse().unwrap())],
                axum::Json(serde_json::json!({
                    "error": "tenant_overloaded",
                    "tenant_id": tenant_id,
                    "in_flight": in_flight,
                    "cap": cap,
                })),
            ).into_response(),
```

If the existing handler returns a custom error type (not `IntoResponse` directly on `Error`), find that wrapper and add the mapping there.

- [ ] **Step 7.7: Wire Kafka NACK**

Edit `crates/heartbit/src/daemon/core.rs:472` — the Kafka match arm for `DaemonCommand::SubmitTask`. Before calling `self.store.insert(...)`, add an overload check:

```rust
                            if let Some(ref tracker) = self.tenant_tracker {
                                let scope = heartbit_core::TenantScope::new(
                                    tenant_id.as_deref().unwrap_or(""),
                                );
                                let estimated = task.len() / 4 + 4096;
                                if let Err(e) = tracker.reserve(&scope, estimated) {
                                    tracing::warn!(
                                        error = %e,
                                        tenant_id = ?tenant_id,
                                        "submit overloaded; NACKing for redelivery"
                                    );
                                    // NACK: return early without committing the message.
                                    // The Kafka consumer's commit happens after this match arm
                                    // completes; returning without insert/work means the next
                                    // poll redelivers (with the consumer's at-least-once semantics).
                                    return;
                                }
                                // Drop the reservation after the check (admission only).
                            }
```

(Verify Kafka commit semantics in the consumer loop: if commit is automatic on every poll regardless of work done, NACK requires explicit `consumer.seek` to the previous offset. Inspect the existing kafka loop and adjust accordingly. If commits are manual after each message, simply not committing achieves NACK.)

- [ ] **Step 7.8: Run full test suite**

```bash
cargo test --workspace --lib
```

Expected: all tests pass, including the new overload test.

- [ ] **Step 7.9: Run quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 7.10: Commit**

```bash
git add crates/heartbit/src/daemon/core.rs crates/heartbit/src/daemon/openai_compat.rs
git commit -m "$(cat <<'EOF'
feat(daemon): submit-time per-tenant overload gate

- DaemonHandle.tenant_tracker (optional Arc<TenantTokenTracker>)
- submit_task_with_user_idem reserves at submit; drops the reservation
  immediately (admission-only check). AgentRunner adjusts in_flight
  per-turn during execution.
- HTTP 503 + Retry-After: 5 mapping for Error::TenantOverloaded.
- Kafka path returns early on overload (no commit), causing redelivery
  after the consumer's retry backoff.
EOF
)"
```

---

## Task 8: `ProviderCircuit` state machine + Arc-owning permit

**Goal:** Per-circuit state machine with `parking_lot::Mutex`. State enum: `Closed { consecutive_failures }`, `Open { until, prev_duration }`, `HalfOpen`. Returns Arc-owning `CircuitPermit`.

**Files:**
- Create: `crates/heartbit-core/src/llm/circuit.rs`
- Modify: `crates/heartbit-core/src/llm/mod.rs` (re-export)
- Modify: `crates/heartbit-core/src/error.rs` (`Error::CircuitOpen`)
- Modify: `crates/heartbit-core/Cargo.toml` (`parking_lot` dep)
- Modify: `crates/heartbit-core/src/lib.rs` (re-export)

**TDD steps:**

- [ ] **Step 8.1: Add `parking_lot` dep**

Edit `crates/heartbit-core/Cargo.toml`:

```toml
[dependencies]
# ... existing deps ...
parking_lot = "0.12"
```

If a workspace `parking_lot` already exists in the root `Cargo.toml`, use `parking_lot.workspace = true` instead. Verify with:

```bash
grep -n "parking_lot" Cargo.toml
```

- [ ] **Step 8.2: Add the `Error::CircuitOpen` variant**

Edit `crates/heartbit-core/src/error.rs`. Add:

```rust
    #[error("circuit breaker open: retry after {until:?} (prev open duration: {prev_duration:?})")]
    CircuitOpen {
        until: std::time::Instant,
        prev_duration: std::time::Duration,
    },
```

Note: `Instant` is not `Serialize`, so this variant cannot cross the wire — that's intentional (it's a runtime control-flow signal, not a persisted state).

- [ ] **Step 8.3: Write failing tests**

Create `crates/heartbit-core/src/llm/circuit.rs`:

```rust
//! Per-(tenant, provider) circuit breaker state machine.
//!
//! See `docs/superpowers/specs/2026-05-02-b5b-failure-mode-hardening-design.md`
//! Component 3 for design rationale.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::error::Error;

#[derive(Debug, Clone)]
pub struct CircuitConfig {
    pub failure_threshold: u32,
    pub initial_open_duration: Duration,
    pub max_open_duration: Duration,
    pub backoff_multiplier: f64,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            initial_open_duration: Duration::from_secs(30),
            max_open_duration: Duration::from_secs(300),
            backoff_multiplier: 2.0,
        }
    }
}

#[derive(Debug)]
enum CircuitState {
    Closed { consecutive_failures: u32 },
    Open { until: Instant, prev_duration: Duration },
    HalfOpen,
}

pub struct ProviderCircuit {
    state: Mutex<CircuitState>,
    config: CircuitConfig,
}

pub struct CircuitPermit {
    circuit: Arc<ProviderCircuit>,
}

impl CircuitPermit {
    pub fn record_success(self) {
        self.circuit.record_success();
    }
    pub fn record_failure(self) {
        self.circuit.record_failure();
    }
}

impl ProviderCircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            state: Mutex::new(CircuitState::Closed { consecutive_failures: 0 }),
            config,
        }
    }

    pub fn permit(self: &Arc<Self>) -> Result<CircuitPermit, Error> {
        let mut state = self.state.lock();
        match *state {
            CircuitState::Closed { .. } => Ok(CircuitPermit { circuit: Arc::clone(self) }),
            CircuitState::Open { until, prev_duration } => {
                if Instant::now() >= until {
                    *state = CircuitState::HalfOpen;
                    Ok(CircuitPermit { circuit: Arc::clone(self) })
                } else {
                    Err(Error::CircuitOpen { until, prev_duration })
                }
            }
            CircuitState::HalfOpen => Err(Error::CircuitOpen {
                until: Instant::now() + Duration::from_millis(50),
                prev_duration: Duration::ZERO,
            }),
        }
    }

    fn record_success(&self) {
        let mut state = self.state.lock();
        *state = CircuitState::Closed { consecutive_failures: 0 };
    }

    fn record_failure(&self) {
        let mut state = self.state.lock();
        match *state {
            CircuitState::Closed { consecutive_failures } => {
                let n = consecutive_failures + 1;
                *state = if n >= self.config.failure_threshold {
                    CircuitState::Open {
                        until: Instant::now() + self.config.initial_open_duration,
                        prev_duration: self.config.initial_open_duration,
                    }
                } else {
                    CircuitState::Closed { consecutive_failures: n }
                };
            }
            CircuitState::HalfOpen => {
                let new_dur_secs = (self.config.initial_open_duration.as_secs_f64()
                    * self.config.backoff_multiplier).max(1.0);
                let new_dur = Duration::from_secs_f64(new_dur_secs)
                    .min(self.config.max_open_duration);
                *state = CircuitState::Open {
                    until: Instant::now() + new_dur,
                    prev_duration: new_dur,
                };
            }
            CircuitState::Open { .. } => { /* already open; no-op */ }
        }
    }

    #[cfg(test)]
    fn force_state_for_test(&self, new_state: CircuitState) {
        *self.state.lock() = new_state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CircuitConfig {
        CircuitConfig {
            failure_threshold: 3,
            initial_open_duration: Duration::from_millis(50),
            max_open_duration: Duration::from_millis(500),
            backoff_multiplier: 2.0,
        }
    }

    #[test]
    fn closed_circuit_passes_requests() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        let p = c.permit().unwrap();
        p.record_success();
    }

    #[test]
    fn n_failures_open_circuit() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        for _ in 0..3 {
            let p = c.permit().unwrap();
            p.record_failure();
        }
        let err = c.permit().unwrap_err();
        assert!(matches!(err, Error::CircuitOpen { .. }));
    }

    #[test]
    fn success_resets_consecutive_failures() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        c.permit().unwrap().record_failure();
        c.permit().unwrap().record_failure();
        c.permit().unwrap().record_success();
        // Still under threshold after one more failure
        c.permit().unwrap().record_failure();
        assert!(c.permit().is_ok());
    }

    #[test]
    fn open_transitions_to_half_open_after_duration() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        for _ in 0..3 {
            c.permit().unwrap().record_failure();
        }
        std::thread::sleep(Duration::from_millis(60));
        assert!(c.permit().is_ok(), "should be HalfOpen permit");
    }

    #[test]
    fn half_open_success_closes_circuit() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        for _ in 0..3 {
            c.permit().unwrap().record_failure();
        }
        std::thread::sleep(Duration::from_millis(60));
        c.permit().unwrap().record_success();
        // Closed now: many permits in a row.
        for _ in 0..10 {
            assert!(c.permit().is_ok());
        }
    }

    #[test]
    fn half_open_failure_reopens_with_doubled_duration() {
        let c = Arc::new(ProviderCircuit::new(cfg()));
        for _ in 0..3 {
            c.permit().unwrap().record_failure();
        }
        std::thread::sleep(Duration::from_millis(60));
        c.permit().unwrap().record_failure();
        // Doubled duration: 100ms now. Try after 60ms — still open.
        std::thread::sleep(Duration::from_millis(60));
        assert!(c.permit().is_err());
        std::thread::sleep(Duration::from_millis(60));
        assert!(c.permit().is_ok());
    }

    #[test]
    fn repeated_half_open_failures_clamp_at_max() {
        let c = Arc::new(ProviderCircuit::new(CircuitConfig {
            failure_threshold: 1,
            initial_open_duration: Duration::from_millis(100),
            max_open_duration: Duration::from_millis(150),
            backoff_multiplier: 4.0,
        }));
        c.permit().unwrap().record_failure(); // → Open(100ms)
        std::thread::sleep(Duration::from_millis(110));
        c.permit().unwrap().record_failure(); // → Open(min(400, 150) = 150ms)
        std::thread::sleep(Duration::from_millis(160));
        assert!(c.permit().is_ok(), "should be openable again at clamped duration");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn permit_can_be_moved_across_await() {
        // Compile-time check for Arc-ownership.
        let c = Arc::new(ProviderCircuit::new(cfg()));
        let p = c.permit().unwrap();
        let task = tokio::spawn(async move {
            tokio::task::yield_now().await;
            p.record_success();
        });
        task.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_requests_during_half_open_only_one_probes() {
        // Once Open transitions to HalfOpen, only one permit is granted at a time.
        // Subsequent permit attempts get CircuitOpen until the probe resolves.
        let c = Arc::new(ProviderCircuit::new(CircuitConfig {
            failure_threshold: 1,
            initial_open_duration: Duration::from_millis(20),
            max_open_duration: Duration::from_millis(200),
            backoff_multiplier: 2.0,
        }));
        c.permit().unwrap().record_failure(); // Open
        tokio::time::sleep(Duration::from_millis(30)).await;

        // First permit transitions Open → HalfOpen and is granted.
        let probe = c.permit().expect("first probe granted");

        // Second concurrent attempt while HalfOpen: rejected with CircuitOpen.
        let second = c.permit();
        assert!(matches!(second, Err(Error::CircuitOpen { .. })));

        // Probe records success → Closed
        probe.record_success();
        assert!(c.permit().is_ok());
    }
}
```

- [ ] **Step 8.4: Wire the module**

Edit `crates/heartbit-core/src/llm/mod.rs`:

```rust
pub mod circuit;
```

Edit `crates/heartbit-core/src/lib.rs`:

```rust
pub use llm::circuit::{CircuitConfig, CircuitPermit, ProviderCircuit};
```

- [ ] **Step 8.5: Run tests**

```bash
cargo test -p heartbit-core --lib llm::circuit -- --nocapture
```

Expected: all 8 tests pass.

- [ ] **Step 8.6: Run quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 8.7: Commit**

```bash
git add crates/heartbit-core/src/llm/circuit.rs crates/heartbit-core/src/llm/mod.rs crates/heartbit-core/src/lib.rs crates/heartbit-core/src/error.rs crates/heartbit-core/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(core): ProviderCircuit state machine with parking_lot::Mutex

State machine: Closed → Open (after N consecutive failures) →
HalfOpen (after open duration) → Closed (on success) or Open
with doubled duration (on failure, clamped at max).

- parking_lot::Mutex over std::sync::Mutex: a fault-tolerance layer
  that disables itself permanently on a single panic defeats its purpose.
- CircuitPermit owns Arc<ProviderCircuit> for async-safety across .await
  and tokio::spawn.
- Error::CircuitOpen carries `until: Instant` (not serializable; intentional
  — runtime control-flow signal, not persisted state).
EOF
)"
```

---

## Task 9: `CircuitTracker` registry + `is_circuit_failure` classifier

**Goal:** `CircuitTracker` owns `HashMap<(tenant, provider), Arc<ProviderCircuit>>` and serves circuits on demand. `is_circuit_failure` reuses `error_class::classify` to decide whether a given error counts as a tripping failure.

**Files:**
- Modify: `crates/heartbit-core/src/llm/circuit.rs`

**TDD steps:**

- [ ] **Step 9.1: Write failing tests**

Append to `crates/heartbit-core/src/llm/circuit.rs`:

```rust
use std::collections::HashMap;
use std::sync::RwLock;

use crate::auth::TenantScope;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct CircuitKey {
    pub tenant_id: String,
    pub provider: String,
}

pub struct CircuitTracker {
    circuits: RwLock<HashMap<CircuitKey, Arc<ProviderCircuit>>>,
    config: CircuitConfig,
}

impl CircuitTracker {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            circuits: RwLock::new(HashMap::new()),
            config,
        }
    }

    pub fn circuit_for(&self, scope: &TenantScope, provider: &str) -> Arc<ProviderCircuit> {
        let key = CircuitKey {
            tenant_id: scope.tenant_id.clone(),
            provider: provider.to_string(),
        };
        // Fast path: read lock + clone.
        if let Ok(g) = self.circuits.read() {
            if let Some(c) = g.get(&key) {
                return Arc::clone(c);
            }
        }
        // Slow path: write lock, insert if still missing.
        let mut g = self.circuits.write().expect("circuit tracker poisoned");
        Arc::clone(g.entry(key).or_insert_with(|| {
            Arc::new(ProviderCircuit::new(self.config.clone()))
        }))
    }
}

pub fn is_circuit_failure(err: &Error) -> bool {
    use crate::llm::error_class::ErrorClass;
    matches!(
        crate::llm::error_class::classify(err),
        ErrorClass::ServerError | ErrorClass::RateLimited | ErrorClass::Network
    )
}
```

Append to the test module:

```rust
    #[test]
    fn tracker_returns_same_arc_for_same_key() {
        let t = CircuitTracker::new(cfg());
        let a = t.circuit_for(&TenantScope::new("acme"), "anthropic");
        let b = t.circuit_for(&TenantScope::new("acme"), "anthropic");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn tracker_isolates_tenants() {
        let t = CircuitTracker::new(cfg());
        let a = t.circuit_for(&TenantScope::new("acme"), "anthropic");
        let b = t.circuit_for(&TenantScope::new("globex"), "anthropic");
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn tracker_isolates_providers() {
        let t = CircuitTracker::new(cfg());
        let a = t.circuit_for(&TenantScope::new("acme"), "anthropic");
        let b = t.circuit_for(&TenantScope::new("acme"), "openai");
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn is_circuit_failure_classifies_correctly() {
        // Server error → trips
        let server = Error::Api { status: 503, message: "service unavailable".into() };
        assert!(is_circuit_failure(&server));

        // Rate limited → trips
        let rate = Error::Api { status: 429, message: "too many requests".into() };
        assert!(is_circuit_failure(&rate));

        // Auth error → does NOT trip (won't recover from retry)
        let auth = Error::Api { status: 401, message: "unauthorized".into() };
        assert!(!is_circuit_failure(&auth));

        // Bad request → does NOT trip
        let bad = Error::Api { status: 400, message: "bad json".into() };
        assert!(!is_circuit_failure(&bad));
    }
```

- [ ] **Step 9.2: Run, verify pass**

```bash
cargo test -p heartbit-core --lib llm::circuit -- --nocapture
```

Expected: all tests pass.

- [ ] **Step 9.3: Re-export from `lib.rs`**

```rust
pub use llm::circuit::{CircuitKey, CircuitTracker, is_circuit_failure};
```

- [ ] **Step 9.4: Quality gate + commit**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit-core/src/llm/circuit.rs crates/heartbit-core/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(core): CircuitTracker registry + is_circuit_failure classifier

- CircuitTracker: HashMap<(tenant, provider), Arc<ProviderCircuit>> with
  fast-path read lock and slow-path write lock for first-insert.
- is_circuit_failure reuses error_class::classify to map errors to
  trip/no-trip decisions: ServerError, RateLimited, Network → trip.
  AuthError, InvalidRequest, ContextOverflow → no-trip.
EOF
)"
```

---

## Task 10: `CircuitBreakerProvider<P>` wrapper with composition tests

**Goal:** Per-runner wrapper that reads tenant scope from construction, fetches circuit from shared tracker, requests permit before delegating to inner provider, records success/failure based on classifier.

**Files:**
- Modify: `crates/heartbit-core/src/llm/circuit.rs`

**TDD steps:**

- [ ] **Step 10.1: Write failing tests**

Append to the test module:

```rust
    use crate::llm::types::{Message, StopReason, TokenUsage, ContentBlock};
    use crate::llm::{LlmProvider, types::CompletionRequest, types::CompletionResponse};

    struct FailingProvider {
        error: Box<dyn Fn() -> Error + Send + Sync>,
    }

    impl LlmProvider for FailingProvider {
        async fn complete(&self, _r: CompletionRequest) -> Result<CompletionResponse, Error> {
            Err((self.error)())
        }
    }

    fn dummy_request() -> CompletionRequest {
        CompletionRequest {
            system: "test".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 10,
            tool_choice: None,
            reasoning_effort: None,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn circuit_opens_after_threshold_failures() {
        let tracker = Arc::new(CircuitTracker::new(CircuitConfig {
            failure_threshold: 3,
            initial_open_duration: Duration::from_secs(60),
            max_open_duration: Duration::from_secs(120),
            backoff_multiplier: 2.0,
        }));
        let inner = FailingProvider {
            error: Box::new(|| Error::Api { status: 503, message: "down".into() }),
        };
        let wrapper = CircuitBreakerProvider::new(
            inner,
            tracker.clone(),
            "anthropic",
            TenantScope::new("acme"),
        );

        // Three failing calls → circuit opens
        for _ in 0..3 {
            let _ = wrapper.complete(dummy_request()).await;
        }
        // Fourth call: short-circuits with CircuitOpen, no inner call
        let err = wrapper.complete(dummy_request()).await.unwrap_err();
        assert!(matches!(err, Error::CircuitOpen { .. }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn auth_errors_do_not_trip_circuit() {
        let tracker = Arc::new(CircuitTracker::new(cfg()));
        let inner = FailingProvider {
            error: Box::new(|| Error::Api { status: 401, message: "no key".into() }),
        };
        let wrapper = CircuitBreakerProvider::new(
            inner,
            tracker.clone(),
            "anthropic",
            TenantScope::new("acme"),
        );

        for _ in 0..10 {
            let _ = wrapper.complete(dummy_request()).await;
        }
        // Circuit still closed: 401s don't trip
        let circuit = tracker.circuit_for(&TenantScope::new("acme"), "anthropic");
        assert!(circuit.permit().is_ok());
    }
```

- [ ] **Step 10.2: Run to verify failure**

```bash
cargo test -p heartbit-core --lib llm::circuit::tests::circuit_opens -- --nocapture
```

Expected: failure (CircuitBreakerProvider not yet defined).

- [ ] **Step 10.3: Implement the wrapper**

Append to `crates/heartbit-core/src/llm/circuit.rs`:

```rust
use crate::llm::types::{CompletionRequest, CompletionResponse};

pub struct CircuitBreakerProvider<P: super::LlmProvider> {
    inner: P,
    tracker: Arc<CircuitTracker>,
    provider_name: String,
    scope: TenantScope,
}

impl<P: super::LlmProvider> CircuitBreakerProvider<P> {
    pub fn new(
        inner: P,
        tracker: Arc<CircuitTracker>,
        provider_name: impl Into<String>,
        scope: TenantScope,
    ) -> Self {
        Self {
            inner,
            tracker,
            provider_name: provider_name.into(),
            scope,
        }
    }
}

impl<P: super::LlmProvider> super::LlmProvider for CircuitBreakerProvider<P> {
    fn model_name(&self) -> Option<&str> {
        self.inner.model_name()
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let circuit = self.tracker.circuit_for(&self.scope, &self.provider_name);
        let permit = circuit.permit()?;
        let result = self.inner.complete(request).await;
        match &result {
            Ok(_) => permit.record_success(),
            Err(e) if is_circuit_failure(e) => permit.record_failure(),
            Err(_) => permit.record_success(),
        }
        result
    }

    async fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &super::OnText,
    ) -> Result<CompletionResponse, Error> {
        let circuit = self.tracker.circuit_for(&self.scope, &self.provider_name);
        let permit = circuit.permit()?;
        let result = self.inner.stream_complete(request, on_text).await;
        match &result {
            Ok(_) => permit.record_success(),
            Err(e) if is_circuit_failure(e) => permit.record_failure(),
            Err(_) => permit.record_success(),
        }
        result
    }
}
```

- [ ] **Step 10.4: Run tests, verify pass**

```bash
cargo test -p heartbit-core --lib llm::circuit -- --nocapture
```

Expected: all 13+ tests pass.

- [ ] **Step 10.5: Add a composition test (circuit + retry)**

Append to the test module:

```rust
    #[tokio::test(flavor = "multi_thread")]
    async fn circuit_outer_retry_inner_one_permit_per_outer_call() {
        // Wrapped: CircuitBreaker<Retrying<FailingProvider>>
        let tracker = Arc::new(CircuitTracker::new(CircuitConfig {
            failure_threshold: 2,
            initial_open_duration: Duration::from_secs(60),
            max_open_duration: Duration::from_secs(120),
            backoff_multiplier: 2.0,
        }));
        let inner = FailingProvider {
            error: Box::new(|| Error::Api { status: 503, message: "down".into() }),
        };
        // RetryingProvider with 3 retries: each outer call burns 4 attempts before failing.
        let retrying = crate::llm::retry::RetryingProvider::new(
            inner,
            crate::llm::retry::RetryConfig {
                max_retries: 3,
                base_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(10),
            },
        );
        let wrapper = CircuitBreakerProvider::new(
            retrying,
            tracker.clone(),
            "anthropic",
            TenantScope::new("acme"),
        );

        // Two outer calls (each burns 4 attempts internally) → 2 circuit failures → opens.
        let _ = wrapper.complete(dummy_request()).await;
        let _ = wrapper.complete(dummy_request()).await;
        let err = wrapper.complete(dummy_request()).await.unwrap_err();
        assert!(matches!(err, Error::CircuitOpen { .. }));
    }
```

- [ ] **Step 10.6: Re-export the wrapper**

Edit `crates/heartbit-core/src/lib.rs`:

```rust
pub use llm::circuit::CircuitBreakerProvider;
```

- [ ] **Step 10.7: Quality gate + commit**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --lib
git add crates/heartbit-core/src/llm/circuit.rs crates/heartbit-core/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(core): CircuitBreakerProvider wrapper composes with RetryingProvider

- Per-runner wrapper takes TenantScope at construction (CompletionRequest
  doesn't carry tenant identity; runner does).
- Composition order: CircuitBreaker<Retrying<P>>. One permit per outer call
  covers the full retry budget; failure_threshold = 5 means 5 retry-exhausted
  outer attempts.
- is_circuit_failure classifier excludes auth/4xx/context-overflow.
EOF
)"
```

---

## Task 11: CLI wiring + config + `parking_lot` workspace

**Goal:** CLI builds `Arc<TenantTokenTracker>` and `Arc<CircuitTracker>` from config and threads them into the daemon and per-runner provider construction. Configuration covers all three components.

**Files:**
- Modify: `Cargo.toml` (workspace dep on `parking_lot`)
- Modify: `crates/heartbit/src/config.rs` (orchestrator + provider sections)
- Modify: `crates/heartbit-cli/src/daemon/mod.rs` (build trackers, wire into handle + runner builders)

**TDD steps:**

- [ ] **Step 11.1: Add config sections**

Edit `crates/heartbit/src/config.rs`. Locate `OrchestratorConfig` (or equivalent) and add:

```rust
    /// Per-tenant in-flight token cap. None → effectively unbounded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_in_flight_per_tenant: Option<usize>,
```

Add a new struct for circuit config:

```rust
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCircuitConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_open_duration_seconds: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_open_duration_seconds: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_multiplier: Option<f64>,
}
```

Add to the `Provider` config block:

```rust
    #[serde(default)]
    pub circuit: ProviderCircuitConfig,
```

Add zero-rejection in `validate`:

```rust
    if let Some(0) = self.provider.circuit.failure_threshold {
        return Err(Error::Config("provider.circuit.failure_threshold must be > 0".into()));
    }
    if let Some(0) = self.provider.circuit.initial_open_duration_seconds {
        return Err(Error::Config("provider.circuit.initial_open_duration_seconds must be > 0".into()));
    }
    if let Some(0) = self.provider.circuit.max_open_duration_seconds {
        return Err(Error::Config("provider.circuit.max_open_duration_seconds must be > 0".into()));
    }
```

Provide a `From<&ProviderCircuitConfig> for CircuitConfig`:

```rust
impl From<&ProviderCircuitConfig> for heartbit_core::CircuitConfig {
    fn from(c: &ProviderCircuitConfig) -> Self {
        let default = heartbit_core::CircuitConfig::default();
        Self {
            failure_threshold: c.failure_threshold.unwrap_or(default.failure_threshold),
            initial_open_duration: c.initial_open_duration_seconds
                .map(|s| std::time::Duration::from_secs(u64::from(s)))
                .unwrap_or(default.initial_open_duration),
            max_open_duration: c.max_open_duration_seconds
                .map(|s| std::time::Duration::from_secs(u64::from(s)))
                .unwrap_or(default.max_open_duration),
            backoff_multiplier: c.backoff_multiplier.unwrap_or(default.backoff_multiplier),
        }
    }
}
```

- [ ] **Step 11.2: Build the trackers in CLI**

Edit `crates/heartbit-cli/src/daemon/mod.rs`. Find the place where `DaemonHandle` is constructed (search `DaemonCore::new` or `handle = `). Before that:

```rust
    // B5b: per-tenant token tracker
    let tenant_tracker = config
        .orchestrator
        .max_tokens_in_flight_per_tenant
        .map(|cap| std::sync::Arc::new(heartbit_core::TenantTokenTracker::new(cap)));

    // B5b: circuit tracker
    let circuit_tracker = std::sync::Arc::new(
        heartbit_core::CircuitTracker::new((&config.provider.circuit).into())
    );
```

After `let handle = ...`:

```rust
    let handle = if let Some(ref tracker) = tenant_tracker {
        handle.with_tenant_tracker(tracker.clone())
    } else {
        handle
    };
```

In the per-runner construction (search for `AgentRunnerBuilder::default()` or similar inside the CLI's runner setup), thread the trackers:

```rust
    let provider_name = match config.provider.kind {
        ProviderKind::Anthropic => "anthropic",
        ProviderKind::OpenAi => "openai",
        ProviderKind::OpenRouter => "openrouter",
        ProviderKind::Gemini => "gemini",
    };
    let scope = heartbit_core::TenantScope::from_audit_fields(
        agent_config.audit_tenant_id.as_deref(),
        agent_config.audit_user_id.as_deref(),
    );
    let provider = heartbit_core::CircuitBreakerProvider::new(
        retrying_provider,
        circuit_tracker.clone(),
        provider_name,
        scope,
    );
    let mut builder = AgentRunnerBuilder::default()
        .provider(provider);
    if let Some(ref tracker) = tenant_tracker {
        builder = builder.tenant_tracker(tracker.clone());
    }
```

(Adjust the exact integration to match the CLI's existing per-runner construction pattern.)

- [ ] **Step 11.3: Add a small smoke test for config parse**

Edit `crates/heartbit/src/config.rs` test module:

```rust
#[test]
fn b5b_full_config_parses() {
    let toml = r#"
[provider.anthropic]
api_key = "test"

[provider.circuit]
failure_threshold = 5
initial_open_duration_seconds = 30
max_open_duration_seconds = 300
backoff_multiplier = 2.0

[orchestrator]
max_tokens_in_flight_per_tenant = 1000000

[daemon]
[daemon.idempotency]
ttl_hours = 24
sweep_interval_minutes = 60
"#;
    let cfg: HeartbitConfig = toml::from_str(toml).unwrap();
    cfg.validate().unwrap();
    assert_eq!(cfg.provider.circuit.failure_threshold, Some(5));
    assert_eq!(cfg.orchestrator.max_tokens_in_flight_per_tenant, Some(1_000_000));
    assert_eq!(cfg.daemon.unwrap().idempotency.ttl_hours, Some(24));
}

#[test]
fn b5b_config_zero_failure_threshold_rejected() {
    let toml = r#"
[provider.anthropic]
api_key = "test"

[provider.circuit]
failure_threshold = 0
"#;
    let cfg: HeartbitConfig = toml::from_str(toml).unwrap();
    assert!(cfg.validate().is_err());
}
```

(Adjust the toml prologue to whatever the existing config tests require.)

- [ ] **Step 11.4: Run tests**

```bash
cargo test -p heartbit --lib config -- --nocapture
cargo test --workspace --lib
```

Expected: green.

- [ ] **Step 11.5: Build heartbit-cli to verify wiring compiles**

```bash
cargo build -p heartbit-cli
```

Expected: clean build.

- [ ] **Step 11.6: Quality gate + commit**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings
git add Cargo.toml crates/heartbit/src/config.rs crates/heartbit-cli/src/daemon/mod.rs
git commit -m "$(cat <<'EOF'
feat(cli): wire B5b — tenant tracker, circuit tracker, idempotency config

- Cargo.toml: parking_lot 0.12 workspace dep
- HeartbitConfig: provider.circuit + orchestrator.max_tokens_in_flight_per_tenant
  + daemon.idempotency. All optional with sensible defaults; zero rejected.
- CLI builds Arc<TenantTokenTracker> from orchestrator config and threads it
  into DaemonHandle and AgentRunnerBuilder.
- CLI builds Arc<CircuitTracker> from provider.circuit and constructs per-runner
  CircuitBreakerProvider<RetryingProvider<P>> with the runner's tenant scope.
EOF
)"
```

---

## Task 12: Docs — CHANGELOG + recipe + verification matrix

**Goal:** User-facing documentation. CHANGELOG entries for the three components. Recipe under `book/src/recipes/failure-modes.md` covering the three opt-in features.

**Files:**
- Modify: `CHANGELOG.md` (top of `[Unreleased]`)
- Create: `book/src/recipes/failure-modes.md` (or whatever path the existing recipes use)
- Modify: `book/src/SUMMARY.md` (link the recipe)

**Steps:**

- [ ] **Step 12.1: Verify the recipe location**

```bash
ls book/src/recipes/ 2>/dev/null || ls book/src/ 2>/dev/null
```

Use whichever path matches the project's existing layout.

- [ ] **Step 12.2: Write the recipe**

Create `book/src/recipes/failure-modes.md`:

```markdown
# Failure-Mode Hardening (B5b)

Three opt-in components that turn the daemon into a fault-tolerant
multi-tenant runtime.

## 1. Idempotency keys

Set the `Idempotency-Key` header on `POST /v1/tasks`:

    POST /v1/tasks
    Idempotency-Key: dedup-payment-12345
    {"task": "process payment 12345"}

Subsequent requests with the same `(tenant, key)` pair return the existing
task id without re-executing. Keys expire after 24h by default. Configure:

```toml
[daemon.idempotency]
ttl_hours = 24
sweep_interval_minutes = 60
```

## 2. Per-tenant token cap

Cap concurrent in-flight tokens per tenant:

```toml
[orchestrator]
max_tokens_in_flight_per_tenant = 1000000
```

Submissions estimated to push the tenant past the cap fail with
`Error::TenantOverloaded` → HTTP 503 + `Retry-After: 5`. Single-tenant
deployments leave this unset (effectively unbounded).

## 3. Per-(tenant, provider) circuit breaker

Trip the circuit after 5 consecutive retry-exhausted attempts. Compose with
existing `RetryingProvider`:

```toml
[provider.circuit]
failure_threshold = 5
initial_open_duration_seconds = 30
max_open_duration_seconds = 300
backoff_multiplier = 2.0
```

Composition order: `CircuitBreaker<Retrying<Provider>>`. One permit per
outer call covers a full retry budget. Open circuits return
`Error::CircuitOpen` immediately (no retries fire while open).
```

- [ ] **Step 12.3: Link from `SUMMARY.md`**

Edit `book/src/SUMMARY.md` and add (under Recipes or Cookbook):

```markdown
  - [Failure-Mode Hardening](recipes/failure-modes.md)
```

- [ ] **Step 12.4: Update CHANGELOG**

Edit `CHANGELOG.md`. Add at the top of `[Unreleased]`:

```markdown
### Added

- **Idempotency keys (B5b).** `DaemonCommand::SubmitTask` and `POST /v1/tasks`
  accept an `Idempotency-Key` (`Option<String>` field / HTTP header).
  Scoped to `(tenant_id, idempotency_key)` via a partial unique index on
  `daemon_tasks`. 24h TTL with background sweep.
- **Per-tenant token cap (B5b).** `TenantTokenTracker` with `Arc`-owning RAII
  reservation tracks in-flight tokens per tenant. Configurable cap via
  `orchestrator.max_tokens_in_flight_per_tenant`. Submissions exceeding the
  cap return `Error::TenantOverloaded` (HTTP 503 + `Retry-After: 5`).
- **Per-(tenant, provider) circuit breaker (B5b).** `CircuitBreakerProvider`
  wrapper around any `LlmProvider`. Composes outside `RetryingProvider`.
  State machine: Closed → Open (after 5 consecutive retry-exhausted
  failures) → HalfOpen → Closed/Open. Configurable via `[provider.circuit]`.

### Changed

- `daemon_tasks.tenant_id` tightened to `NOT NULL DEFAULT ''` (matches the
  B4 audit_log pattern). Existing rows backfilled to the empty-string
  sentinel. Migration is idempotent.
```

- [ ] **Step 12.5: Final verification matrix**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --lib
cargo test --workspace --lib --features postgres,local-embedding
```

(Add whatever feature flags the project uses for full coverage; mirror the project's CI.)

- [ ] **Step 12.6: Optional Postgres integration tests**

If `DATABASE_URL` is set in the environment, run the `#[ignore]`-gated Postgres tests:

```bash
DATABASE_URL=postgres://localhost/heartbit_test cargo test --workspace --lib -- --ignored
```

- [ ] **Step 12.7: Commit**

```bash
git add CHANGELOG.md book/src/recipes/failure-modes.md book/src/SUMMARY.md
git commit -m "$(cat <<'EOF'
docs: B5b CHANGELOG + failure-modes recipe

CHANGELOG entries for the three components and the daemon_tasks.tenant_id
migration. Recipe under book/src/recipes/ covers the opt-in HTTP header,
config sections, and composition order.
EOF
)"
```

---

## Task 13: Final code review + finishing the branch

- [ ] **Step 13.1: Run final verification**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --lib
```

- [ ] **Step 13.2: Use the finishing-a-development-branch skill**

Announce: "I'm using the finishing-a-development-branch skill to complete this work."

Follow that skill:
1. Verify tests pass (just ran).
2. Determine base branch (`main`).
3. Present the four options to the user.
4. Execute the user's choice.

---

## Test Coverage Summary

| Component | New unit tests | Postgres `#[ignore]` tests |
|-----------|----------------|----------------------------|
| Idempotency (Tasks 1–4) | ~14 | ~6 (covering find/insert/sweep) |
| Tenant tracker (Tasks 5–7) | ~11 | 0 (in-process only) |
| Circuit breaker (Tasks 8–10) | ~16 | 0 (in-process only) |
| Config (Task 11) | ~3 | 0 |
| **Total** | **~44 unit** | **~6 integration** |

Existing test count post-B5a: 432 (heartbit) + 2306 (heartbit-core) + 621 (heartbit-cli) + 113 (heartbit-telegram) = **3472 lib tests**. Target post-B5b: ~3516 lib tests.
