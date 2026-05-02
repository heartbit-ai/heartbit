use chrono::{DateTime, Utc};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::Error;
use crate::store::{AuditEntry, TaskRecord};

/// Internal PostgreSQL-specific TaskRecord row with FromRow derive.
#[derive(Debug, Clone, FromRow)]
struct PgTaskRecord {
    id: Uuid,
    status: String,
    task_input: String,
    config_name: Option<String>,
    result: Option<String>,
    error: Option<String>,
    token_usage: Option<serde_json::Value>,
    created_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
}

impl From<PgTaskRecord> for TaskRecord {
    fn from(record: PgTaskRecord) -> Self {
        TaskRecord {
            id: record.id,
            status: record.status,
            task_input: record.task_input,
            config_name: record.config_name,
            result: record.result,
            error: record.error,
            token_usage: record.token_usage,
            created_at: record.created_at,
            completed_at: record.completed_at,
        }
    }
}

/// Internal PostgreSQL-specific AuditEntry row with FromRow derive.
#[derive(Debug, Clone, FromRow)]
struct PgAuditEntry {
    id: i64,
    task_id: Uuid,
    agent_name: String,
    event_type: String,
    payload: serde_json::Value,
    tokens_in: Option<i32>,
    tokens_out: Option<i32>,
    created_at: DateTime<Utc>,
    tenant_id: Option<String>,
    user_id: Option<String>,
}

impl From<PgAuditEntry> for AuditEntry {
    fn from(entry: PgAuditEntry) -> Self {
        AuditEntry {
            id: entry.id,
            task_id: entry.task_id,
            agent_name: entry.agent_name,
            event_type: entry.event_type,
            payload: entry.payload,
            tokens_in: entry.tokens_in,
            tokens_out: entry.tokens_out,
            created_at: entry.created_at,
            tenant_id: entry.tenant_id,
            user_id: entry.user_id,
        }
    }
}

/// PostgreSQL store for task tracking and audit logging.
pub struct PostgresStore {
    pool: PgPool,
}

impl PostgresStore {
    /// Create a store from an existing connection pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Connect to PostgreSQL using the given URL.
    pub async fn connect(database_url: &str) -> Result<Self, Error> {
        let pool = PgPool::connect(database_url)
            .await
            .map_err(|e| Error::Store(format!("database connection failed: {e}")))?;
        Ok(Self { pool })
    }

    /// Run the initial schema migration. Idempotent — safe to re-run.
    ///
    /// **B4 multi-tenant upgrade notes:** before upgrading from a pre-B4
    /// deployment, audit existing data:
    ///
    /// ```sql
    /// SELECT count(*) FROM audit_log WHERE tenant_id IS NULL;
    /// ```
    ///
    /// Non-zero on a multi-tenant installation indicates rows that were
    /// written without a tenant scope (a pre-B4 bug). The migration backfills
    /// these rows with the empty-string sentinel (single-tenant), matching
    /// `TenantScope::default()`. After backfill, the column becomes NOT NULL.
    pub async fn run_migration(&self) -> Result<(), Error> {
        // Split into separate statements — sqlx doesn't support multiple
        // commands in a single prepared statement.
        let statements = [
            r#"CREATE TABLE IF NOT EXISTS tasks (
                id            UUID PRIMARY KEY,
                status        TEXT NOT NULL DEFAULT 'pending',
                task_input    TEXT NOT NULL,
                config_name   TEXT,
                result        TEXT,
                error         TEXT,
                token_usage   JSONB,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                completed_at  TIMESTAMPTZ
            )"#,
            r#"CREATE TABLE IF NOT EXISTS audit_log (
                id          BIGSERIAL PRIMARY KEY,
                task_id     UUID REFERENCES tasks(id),
                agent_name  TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                payload     JSONB NOT NULL,
                tokens_in   INT,
                tokens_out  INT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )"#,
            "CREATE INDEX IF NOT EXISTS idx_audit_task ON audit_log(task_id)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
            "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS tenant_id TEXT",
            "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS user_id TEXT",
            // B4 (Task 8): tighten tenant_id to NOT NULL DEFAULT '' for symmetry with
            // memories.author_tenant_id and to make tenant scoping a guaranteed
            // post-write invariant. Existing rows with NULL are backfilled to ''
            // (single-tenant sentinel) — matches TenantScope::default() so default-
            // scoped queries remain transparent against historical data.
            "UPDATE audit_log SET tenant_id = '' WHERE tenant_id IS NULL",
            "ALTER TABLE audit_log ALTER COLUMN tenant_id SET DEFAULT ''",
            "ALTER TABLE audit_log ALTER COLUMN tenant_id SET NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id)",
            // idx_audit_created_at is required by prune_audit's DELETE to avoid full table scans.
            "CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_log(created_at)",
            // Composite index for the most common scoped-retention query shape:
            // tenant-scoped recall ordered by recency.
            "CREATE INDEX IF NOT EXISTS idx_audit_tenant_created ON audit_log(tenant_id, created_at DESC)",
        ];
        for stmt in statements {
            sqlx::query(stmt)
                .execute(&self.pool)
                .await
                .map_err(|e| Error::Store(format!("migration failed: {e}")))?;
        }

        Ok(())
    }

    /// Create a new task record.
    pub async fn create_task(
        &self,
        id: Uuid,
        task_input: &str,
        config_name: Option<&str>,
    ) -> Result<TaskRecord, Error> {
        let record: PgTaskRecord = sqlx::query_as(
            r#"
            INSERT INTO tasks (id, task_input, config_name)
            VALUES ($1, $2, $3)
            RETURNING id, status, task_input, config_name, result, error,
                      token_usage, created_at, completed_at
            "#,
        )
        .bind(id)
        .bind(task_input)
        .bind(config_name)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| Error::Store(format!("failed to create task: {e}")))?;

        Ok(record.into())
    }

    /// Update task status and optionally set result/error.
    pub async fn complete_task(
        &self,
        id: Uuid,
        status: &str,
        result: Option<&str>,
        error: Option<&str>,
        token_usage: Option<serde_json::Value>,
    ) -> Result<(), Error> {
        let result_row = sqlx::query(
            r#"
            UPDATE tasks
            SET status = $2, result = $3, error = $4, token_usage = $5,
                completed_at = now()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(status)
        .bind(result)
        .bind(error)
        .bind(token_usage)
        .execute(&self.pool)
        .await
        .map_err(|e| Error::Store(format!("failed to update task: {e}")))?;

        if result_row.rows_affected() == 0 {
            return Err(Error::Store(format!("task not found: {id}")));
        }

        Ok(())
    }

    /// Get a task by ID.
    pub async fn get_task(&self, id: Uuid) -> Result<Option<TaskRecord>, Error> {
        let record: Option<PgTaskRecord> = sqlx::query_as(
            r#"
            SELECT id, status, task_input, config_name, result, error,
                   token_usage, created_at, completed_at
            FROM tasks WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Error::Store(format!("failed to fetch task: {e}")))?;

        Ok(record.map(|r| r.into()))
    }

    /// Write an audit log entry.
    #[allow(clippy::too_many_arguments)]
    pub async fn write_audit(
        &self,
        task_id: Uuid,
        agent_name: &str,
        event_type: &str,
        payload: serde_json::Value,
        tokens_in: Option<i32>,
        tokens_out: Option<i32>,
        tenant_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<(), Error> {
        sqlx::query(
            r#"
            INSERT INTO audit_log (task_id, agent_name, event_type, payload, tokens_in, tokens_out, tenant_id, user_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
        )
        .bind(task_id)
        .bind(agent_name)
        .bind(event_type)
        .bind(payload)
        .bind(tokens_in)
        .bind(tokens_out)
        // tenant_id is NOT NULL DEFAULT '' on the column; bind "" when caller
        // supplies None so single-tenant audits land in the sentinel namespace.
        .bind(tenant_id.unwrap_or(""))
        .bind(user_id)
        .execute(&self.pool)
        .await
        .map_err(|e| Error::Store(format!("failed to write audit log: {e}")))?;

        Ok(())
    }

    /// Get audit log entries for a task.
    pub async fn get_audit_log(&self, task_id: Uuid) -> Result<Vec<AuditEntry>, Error> {
        let entries: Vec<PgAuditEntry> = sqlx::query_as(
            r#"
            SELECT id, task_id, agent_name, event_type, payload,
                   tokens_in, tokens_out, created_at, tenant_id, user_id
            FROM audit_log WHERE task_id = $1
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Error::Store(format!("failed to fetch audit log: {e}")))?;

        Ok(entries.into_iter().map(|e| e.into()).collect())
    }

    /// Delete audit_log rows older than `now - retain`. Returns the total number of rows deleted.
    ///
    /// Rows are deleted in batches of 10 000 with a 50 ms yield between chunks so that
    /// the DELETE transactions stay short and avoid long-running exclusive locks on large
    /// audit tables.
    pub async fn prune_audit(&self, retain: chrono::Duration) -> Result<usize, Error> {
        let cutoff = chrono::Utc::now() - retain;
        let mut total_removed = 0usize;
        loop {
            let result = sqlx::query(
                "DELETE FROM audit_log WHERE id IN (
                     SELECT id FROM audit_log WHERE created_at < $1 LIMIT 10000
                 )",
            )
            .bind(cutoff)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Store(format!("failed to prune audit log: {e}")))?;

            let removed = result.rows_affected() as usize;
            total_removed += removed;
            if removed < 10000 {
                // Last batch was partial — no more rows to delete.
                break;
            }
            // Yield to other writers between chunks.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        Ok(total_removed)
    }
}

/// Audit trail backed by PostgreSQL, bridging to the existing `audit_log` table.
///
/// Each `PostgresAuditTrail` is scoped to a single task (via `task_id`).
pub struct PostgresAuditTrail {
    store: std::sync::Arc<PostgresStore>,
    task_id: Uuid,
}

impl PostgresAuditTrail {
    pub fn new(store: std::sync::Arc<PostgresStore>, task_id: Uuid) -> Self {
        Self { store, task_id }
    }
}

impl PostgresAuditTrail {
    /// Convert a stored [`AuditEntry`] row into the public [`crate::agent::audit::AuditRecord`].
    fn entry_to_record(row: AuditEntry) -> crate::agent::audit::AuditRecord {
        crate::agent::audit::AuditRecord {
            agent: row.agent_name,
            turn: 0,
            event_type: row.event_type,
            payload: row.payload,
            usage: crate::llm::types::TokenUsage {
                input_tokens: row.tokens_in.unwrap_or(0) as u32,
                output_tokens: row.tokens_out.unwrap_or(0) as u32,
                ..Default::default()
            },
            timestamp: row.created_at,
            user_id: row.user_id,
            tenant_id: row.tenant_id,
            delegation_chain: Vec::new(),
        }
    }
}

impl crate::agent::audit::AuditTrail for PostgresAuditTrail {
    fn record(
        &self,
        entry: crate::agent::audit::AuditRecord,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            self.store
                .write_audit(
                    self.task_id,
                    &entry.agent,
                    &entry.event_type,
                    entry.payload,
                    Some(entry.usage.input_tokens as i32),
                    Some(entry.usage.output_tokens as i32),
                    entry.tenant_id.as_deref(),
                    entry.user_id.as_deref(),
                )
                .await
        })
    }

    fn entries(
        &self,
        scope: &crate::auth::TenantScope,
        limit: usize,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Vec<crate::agent::audit::AuditRecord>, Error>>
                + Send
                + '_,
        >,
    > {
        let tid = scope.tenant_id.clone();
        Box::pin(async move {
            let rows = self.store.get_audit_log(self.task_id).await?;
            let matched: Vec<_> = rows
                .into_iter()
                .map(Self::entry_to_record)
                .filter(|r| r.tenant_id.as_deref().unwrap_or("") == tid.as_str())
                .collect();
            let start = matched.len().saturating_sub(limit);
            Ok(matched[start..].to_vec())
        })
    }

    fn entries_unscoped(
        &self,
        limit: usize,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Vec<crate::agent::audit::AuditRecord>, Error>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let rows = self.store.get_audit_log(self.task_id).await?;
            let all: Vec<_> = rows.into_iter().map(Self::entry_to_record).collect();
            let start = all.len().saturating_sub(limit);
            Ok(all[start..].to_vec())
        })
    }

    fn entries_since(
        &self,
        scope: &crate::auth::TenantScope,
        since: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Vec<crate::agent::audit::AuditRecord>, Error>>
                + Send
                + '_,
        >,
    > {
        let tid = scope.tenant_id.clone();
        Box::pin(async move {
            let rows = self.store.get_audit_log(self.task_id).await?;
            let matched: Vec<_> = rows
                .into_iter()
                .map(Self::entry_to_record)
                .filter(|r| {
                    r.tenant_id.as_deref().unwrap_or("") == tid.as_str() && r.timestamp >= since
                })
                .collect();
            let start = matched.len().saturating_sub(limit);
            Ok(matched[start..].to_vec())
        })
    }

    fn prune(
        &self,
        retain: chrono::Duration,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize, Error>> + Send + '_>>
    {
        let store = self.store.clone();
        Box::pin(async move { store.prune_audit(retain).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::audit::AuditTrail as _;

    #[test]
    fn task_record_serializes() {
        let record = TaskRecord {
            id: Uuid::new_v4(),
            status: "pending".into(),
            task_input: "Analyze data".into(),
            config_name: Some("default".into()),
            result: None,
            error: None,
            token_usage: None,
            created_at: Utc::now(),
            completed_at: None,
        };
        let json = serde_json::to_string(&record).unwrap();
        let parsed: TaskRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.status, "pending");
        assert_eq!(parsed.task_input, "Analyze data");
    }

    #[test]
    fn audit_entry_serializes() {
        let entry = AuditEntry {
            id: 1,
            task_id: Uuid::new_v4(),
            agent_name: "researcher".into(),
            event_type: "llm_call".into(),
            payload: serde_json::json!({"model": "claude-sonnet-4"}),
            tokens_in: Some(100),
            tokens_out: Some(50),
            created_at: Utc::now(),
            tenant_id: None,
            user_id: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: AuditEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.agent_name, "researcher");
        assert_eq!(parsed.event_type, "llm_call");
    }

    #[test]
    fn task_record_with_token_usage() {
        let record = TaskRecord {
            id: Uuid::new_v4(),
            status: "completed".into(),
            task_input: "test".into(),
            config_name: None,
            result: Some("done".into()),
            error: None,
            token_usage: Some(serde_json::json!({"input_tokens": 100, "output_tokens": 50})),
            created_at: Utc::now(),
            completed_at: Some(Utc::now()),
        };
        let json = serde_json::to_string(&record).unwrap();
        let parsed: TaskRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.status, "completed");
        assert!(parsed.token_usage.is_some());
    }

    /// Integration test: prune_audit deletes rows older than the retain window.
    ///
    /// Requires a live PostgreSQL instance (DATABASE_URL env var). Ignored in CI.
    #[tokio::test]
    #[ignore = "requires DATABASE_URL (live PostgreSQL)"]
    async fn prune_audit_deletes_old_rows() {
        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let store = PostgresStore::connect(&url).await.expect("connect");
        store.run_migration().await.expect("migration");

        let task_id = Uuid::new_v4();
        // Insert a task so the FK constraint on audit_log is satisfied
        sqlx::query(
            "INSERT INTO tasks (id, status, task_input) VALUES ($1, 'completed', 'prune test')",
        )
        .bind(task_id)
        .execute(&store.pool)
        .await
        .expect("insert task");

        // Insert a row with a created_at in the far past (30 days ago)
        let old_ts = chrono::Utc::now() - chrono::Duration::days(30);
        sqlx::query(
            "INSERT INTO audit_log (task_id, agent_name, event_type, payload, created_at) \
             VALUES ($1, 'test-agent', 'test', '{}', $2)",
        )
        .bind(task_id)
        .bind(old_ts)
        .execute(&store.pool)
        .await
        .expect("insert old audit row");

        // Insert a fresh row (should NOT be pruned)
        store
            .write_audit(
                task_id,
                "test-agent",
                "test",
                serde_json::json!({}),
                None,
                None,
                None,
                None,
            )
            .await
            .expect("insert fresh audit row");

        // Prune rows older than 7 days — only the 30-day-old row should be removed.
        // In an isolated test DB exactly 1 row is deleted; in a shared DB at least 1.
        let removed = store
            .prune_audit(chrono::Duration::days(7))
            .await
            .expect("prune_audit");
        assert!(
            removed >= 1,
            "expected at least 1 deleted row, got {removed}"
        );

        // Cleanup
        sqlx::query("DELETE FROM audit_log WHERE task_id = $1")
            .bind(task_id)
            .execute(&store.pool)
            .await
            .ok();
        sqlx::query("DELETE FROM tasks WHERE id = $1")
            .bind(task_id)
            .execute(&store.pool)
            .await
            .ok();
    }

    /// Integration test: PostgresAuditTrail::entries() returns only rows for the
    /// matching tenant. Without tenant_id in the schema this always returned empty.
    ///
    /// Requires a live PostgreSQL instance (DATABASE_URL env var). Ignored in CI.
    #[tokio::test]
    #[ignore = "requires DATABASE_URL (live PostgreSQL)"]
    async fn audit_entries_by_scope_returns_only_matching_tenant() {
        use std::sync::Arc;

        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let store = Arc::new(PostgresStore::connect(&url).await.expect("connect"));
        store.run_migration().await.expect("migration");

        let task_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO tasks (id, status, task_input) VALUES ($1, 'completed', 'scope test')",
        )
        .bind(task_id)
        .execute(&store.pool)
        .await
        .expect("insert task");

        // Write two rows under different tenants
        store
            .write_audit(
                task_id,
                "agent-acme",
                "llm_response",
                serde_json::json!({}),
                Some(10),
                Some(5),
                Some("acme"),
                Some("alice"),
            )
            .await
            .expect("write acme row");
        store
            .write_audit(
                task_id,
                "agent-globex",
                "llm_response",
                serde_json::json!({}),
                Some(20),
                Some(10),
                Some("globex"),
                Some("bob"),
            )
            .await
            .expect("write globex row");

        let trail = PostgresAuditTrail::new(store.clone(), task_id);
        let scope_acme = crate::auth::TenantScope::new("acme");
        let scope_globex = crate::auth::TenantScope::new("globex");

        let acme_rows = trail.entries(&scope_acme, 100).await.expect("entries acme");
        assert_eq!(acme_rows.len(), 1, "expected exactly 1 acme row");
        assert_eq!(acme_rows[0].agent, "agent-acme");
        assert_eq!(acme_rows[0].tenant_id.as_deref(), Some("acme"));

        let globex_rows = trail
            .entries(&scope_globex, 100)
            .await
            .expect("entries globex");
        assert_eq!(globex_rows.len(), 1, "expected exactly 1 globex row");
        assert_eq!(globex_rows[0].agent, "agent-globex");
        assert_eq!(globex_rows[0].tenant_id.as_deref(), Some("globex"));

        // Cleanup
        sqlx::query("DELETE FROM audit_log WHERE task_id = $1")
            .bind(task_id)
            .execute(&store.pool)
            .await
            .ok();
        sqlx::query("DELETE FROM tasks WHERE id = $1")
            .bind(task_id)
            .execute(&store.pool)
            .await
            .ok();
    }
}
