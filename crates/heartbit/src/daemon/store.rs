use std::collections::HashMap;
use std::sync::RwLock;

#[cfg(any(feature = "postgres", test))]
use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::types::{DaemonTask, TaskState, TaskStats, UsageGroupBy, UsageQuery, UsageRow};
use crate::Error;
#[cfg(any(feature = "postgres", test))]
use crate::llm::types::TokenUsage;

/// Detect a Postgres unique-constraint violation by inspecting the error
/// message for the `23505` SQLSTATE code that sqlx surfaces. Used by the
/// idempotency-key insert flow to recover from concurrent inserts of the
/// same `(tenant_id, idempotency_key)` pair.
#[allow(dead_code)]
pub(crate) fn is_unique_violation(err: &Error) -> bool {
    let msg = err.to_string().to_lowercase();
    msg.contains("23505") || msg.contains("duplicate key value violates unique constraint")
}

/// Trait for persisting daemon task state.
pub trait TaskStore: Send + Sync {
    /// Insert a new task. Returns an error if the task ID already exists.
    fn insert(&self, task: DaemonTask) -> Result<(), Error>;

    /// Get a task by ID.
    fn get(&self, id: Uuid) -> Result<Option<DaemonTask>, Error>;

    /// List tasks in insertion order. Returns `(tasks, total_count)`.
    fn list(&self, limit: usize, offset: usize) -> Result<(Vec<DaemonTask>, usize), Error>;

    /// Update a task via a closure. The closure receives a mutable reference
    /// to the task and may modify it in place. Returns an error if the task
    /// is not found.
    fn update(&self, id: Uuid, f: &dyn Fn(&mut DaemonTask)) -> Result<(), Error>;

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

    /// List tasks with optional source, state, and tenant filters. Returns `(tasks, total_matching)`.
    fn list_filtered(
        &self,
        limit: usize,
        offset: usize,
        source: Option<&str>,
        state: Option<TaskState>,
        tenant_id: Option<&str>,
    ) -> Result<(Vec<DaemonTask>, usize), Error> {
        let (all_tasks, _) = self.list(usize::MAX, 0)?;
        let filtered: Vec<DaemonTask> = all_tasks
            .into_iter()
            .filter(|t| source.is_none_or(|s| t.source == s))
            .filter(|t| state.is_none_or(|s| t.state == s))
            .filter(|t| tenant_id.is_none_or(|tid| t.tenant_id.as_deref() == Some(tid)))
            .collect();
        let total = filtered.len();
        let tasks = filtered.into_iter().skip(offset).take(limit).collect();
        Ok((tasks, total))
    }

    /// Compute aggregate statistics, optionally scoped to a tenant.
    fn stats(&self, tenant_id: Option<&str>) -> Result<TaskStats, Error> {
        let (all_tasks, _) = self.list(usize::MAX, 0)?;
        let mut stats = TaskStats::default();
        for task in &all_tasks {
            if let Some(tid) = tenant_id
                && task.tenant_id.as_deref() != Some(tid)
            {
                continue;
            }
            stats.total_tasks += 1;
            let state_key = task.state.as_str();
            *stats
                .tasks_by_state
                .entry(state_key.to_string())
                .or_default() += 1;
            *stats
                .tasks_by_source
                .entry(task.source.clone())
                .or_default() += 1;
            if task.state == TaskState::Running {
                stats.active_tasks += 1;
            }
            stats.total_input_tokens += task.tokens_used.input_tokens as u64;
            stats.total_output_tokens += task.tokens_used.output_tokens as u64;
            stats.total_cache_read_tokens += task.tokens_used.cache_read_input_tokens as u64;
            stats.total_cache_creation_tokens +=
                task.tokens_used.cache_creation_input_tokens as u64;
            if let Some(cost) = task.estimated_cost_usd {
                stats.total_estimated_cost_usd += cost;
            }
        }
        Ok(stats)
    }

    /// Query usage statistics with time-range, filters, and optional grouping.
    fn usage_stats(&self, query: &UsageQuery) -> Result<Vec<UsageRow>, Error> {
        let (all_tasks, _) = self.list(usize::MAX, 0)?;
        let filtered: Vec<&DaemonTask> = all_tasks
            .iter()
            .filter(|t| query.from.is_none_or(|from| t.created_at >= from))
            .filter(|t| query.to.is_none_or(|to| t.created_at < to))
            .filter(|t| {
                query
                    .tenant_id
                    .as_deref()
                    .is_none_or(|tid| t.tenant_id.as_deref() == Some(tid))
            })
            .filter(|t| {
                query
                    .user_id
                    .as_deref()
                    .is_none_or(|uid| t.user_id.as_deref() == Some(uid))
            })
            .filter(|t| {
                query
                    .agent_name
                    .as_deref()
                    .is_none_or(|a| t.agent_name.as_deref() == Some(a))
            })
            .filter(|t| {
                query
                    .model_name
                    .as_deref()
                    .is_none_or(|m| t.model_name.as_deref() == Some(m))
            })
            .filter(|t| query.source.as_deref().is_none_or(|s| t.source == s))
            .collect();

        // Group tasks
        let mut groups: HashMap<Option<String>, Vec<&DaemonTask>> = HashMap::new();
        for task in &filtered {
            let key = match query.group_by {
                None => None,
                Some(UsageGroupBy::Agent) => task.agent_name.clone(),
                Some(UsageGroupBy::Model) => task.model_name.clone(),
                Some(UsageGroupBy::User) => task.user_id.clone(),
                Some(UsageGroupBy::Tenant) => task.tenant_id.clone(),
                Some(UsageGroupBy::Source) => Some(task.source.clone()),
                Some(UsageGroupBy::Day) => Some(task.created_at.format("%Y-%m-%d").to_string()),
            };
            groups.entry(key).or_default().push(task);
        }

        // If no tasks matched and no grouping, return a single zero row
        if groups.is_empty() {
            return Ok(vec![UsageRow::default()]);
        }

        let mut rows: Vec<UsageRow> = groups
            .into_iter()
            .map(|(key, tasks)| {
                let mut row = UsageRow {
                    group_key: key,
                    task_count: tasks.len() as u64,
                    ..Default::default()
                };
                let mut duration_sum = 0.0f64;
                let mut duration_count = 0u64;
                for t in &tasks {
                    if t.state == TaskState::Completed {
                        row.completed_count += 1;
                    }
                    if t.state == TaskState::Failed {
                        row.failed_count += 1;
                    }
                    row.input_tokens += t.tokens_used.input_tokens as u64;
                    row.output_tokens += t.tokens_used.output_tokens as u64;
                    row.cache_read_tokens += t.tokens_used.cache_read_input_tokens as u64;
                    row.cache_creation_tokens += t.tokens_used.cache_creation_input_tokens as u64;
                    row.reasoning_tokens += t.tokens_used.reasoning_tokens as u64;
                    row.tool_calls += t.tool_calls_made as u64;
                    if let Some(cost) = t.estimated_cost_usd {
                        row.estimated_cost_usd += cost;
                    }
                    if let (Some(started), Some(completed)) = (t.started_at, t.completed_at) {
                        let dur = (completed - started).num_milliseconds() as f64 / 1000.0;
                        duration_sum += dur;
                        duration_count += 1;
                    }
                }
                if duration_count > 0 {
                    row.avg_duration_secs = Some(duration_sum / duration_count as f64);
                }
                row
            })
            .collect();

        // Sort by group_key for deterministic output
        rows.sort_by(|a, b| a.group_key.cmp(&b.group_key));
        Ok(rows)
    }
}

/// In-memory task store backed by `std::sync::RwLock`.
///
/// Uses `std::sync::RwLock` (not tokio) because locks are never held across
/// `.await` boundaries. A separate `Vec<Uuid>` tracks insertion order.
pub struct InMemoryTaskStore {
    tasks: RwLock<HashMap<Uuid, DaemonTask>>,
    order: RwLock<Vec<Uuid>>,
}

impl InMemoryTaskStore {
    pub fn new() -> Self {
        Self {
            tasks: RwLock::new(HashMap::new()),
            order: RwLock::new(Vec::new()),
        }
    }
}

impl Default for InMemoryTaskStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskStore for InMemoryTaskStore {
    fn insert(&self, task: DaemonTask) -> Result<(), Error> {
        let id = task.id;
        // Acquire both locks before mutating to keep tasks and order consistent.
        // Lock order: tasks → order (same as list/list_filtered to avoid deadlock).
        let mut tasks = self
            .tasks
            .write()
            .map_err(|e| Error::Daemon(e.to_string()))?;
        let mut order = self
            .order
            .write()
            .map_err(|e| Error::Daemon(e.to_string()))?;
        if tasks.contains_key(&id) {
            return Err(Error::Daemon(format!("task {id} already exists")));
        }
        tasks.insert(id, task);
        order.push(id);
        Ok(())
    }

    fn get(&self, id: Uuid) -> Result<Option<DaemonTask>, Error> {
        let tasks = self
            .tasks
            .read()
            .map_err(|e| Error::Daemon(e.to_string()))?;
        Ok(tasks.get(&id).cloned())
    }

    fn list(&self, limit: usize, offset: usize) -> Result<(Vec<DaemonTask>, usize), Error> {
        let tasks = self
            .tasks
            .read()
            .map_err(|e| Error::Daemon(e.to_string()))?;
        let order = self
            .order
            .read()
            .map_err(|e| Error::Daemon(e.to_string()))?;
        let total = order.len();
        let result: Vec<DaemonTask> = order
            .iter()
            .rev() // newest first
            .skip(offset)
            .take(limit)
            .filter_map(|id| tasks.get(id).cloned())
            .collect();
        Ok((result, total))
    }

    fn update(&self, id: Uuid, f: &dyn Fn(&mut DaemonTask)) -> Result<(), Error> {
        let mut tasks = self
            .tasks
            .write()
            .map_err(|e| Error::Daemon(e.to_string()))?;
        let task = tasks
            .get_mut(&id)
            .ok_or_else(|| Error::Daemon(format!("task {id} not found")))?;
        f(task);
        Ok(())
    }

    fn list_filtered(
        &self,
        limit: usize,
        offset: usize,
        source: Option<&str>,
        state: Option<TaskState>,
        tenant_id: Option<&str>,
    ) -> Result<(Vec<DaemonTask>, usize), Error> {
        let tasks = self
            .tasks
            .read()
            .map_err(|e| Error::Daemon(e.to_string()))?;
        let order = self
            .order
            .read()
            .map_err(|e| Error::Daemon(e.to_string()))?;
        let filtered: Vec<DaemonTask> = order
            .iter()
            .rev()
            .filter_map(|id| tasks.get(id))
            .filter(|t| source.is_none_or(|s| t.source == s))
            .filter(|t| state.is_none_or(|s| t.state == s))
            .filter(|t| tenant_id.is_none_or(|tid| t.tenant_id.as_deref() == Some(tid)))
            .cloned()
            .collect();
        let total = filtered.len();
        let result = filtered.into_iter().skip(offset).take(limit).collect();
        Ok((result, total))
    }

    fn find_by_idempotency_key(
        &self,
        tenant_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<DaemonTask>, Error> {
        let guard = self
            .tasks
            .read()
            .map_err(|_| Error::Daemon("task store poisoned".into()))?;
        Ok(guard
            .values()
            .find(|t| {
                t.tenant_id.as_deref().unwrap_or("") == tenant_id
                    && t.idempotency_key.as_deref() == Some(idempotency_key)
            })
            .cloned())
    }

    fn sweep_expired_idempotency_keys(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize, Error> {
        let mut guard = self
            .tasks
            .write()
            .map_err(|_| Error::Daemon("task store poisoned".into()))?;
        let mut count = 0usize;
        for task in guard.values_mut() {
            if task.idempotency_key.is_some() && task.created_at < cutoff {
                task.idempotency_key = None;
                count += 1;
            }
        }
        Ok(count)
    }
}

// --- PostgreSQL task store ---

#[cfg(feature = "postgres")]
mod postgres_store {
    use super::*;

    /// Row type for reading daemon tasks from PostgreSQL.
    #[derive(Debug, sqlx::FromRow)]
    pub(crate) struct TaskRow {
        pub(crate) id: Uuid,
        pub(crate) task: String,
        pub(crate) state: String,
        pub(crate) created_at: DateTime<Utc>,
        pub(crate) started_at: Option<DateTime<Utc>>,
        pub(crate) completed_at: Option<DateTime<Utc>>,
        pub(crate) result: Option<String>,
        pub(crate) error: Option<String>,
        pub(crate) input_tokens: i32,
        pub(crate) output_tokens: i32,
        pub(crate) cache_creation_input_tokens: i32,
        pub(crate) cache_read_input_tokens: i32,
        pub(crate) reasoning_tokens: i32,
        pub(crate) tool_calls_made: i32,
        pub(crate) estimated_cost_usd: Option<f64>,
        pub(crate) source: String,
        pub(crate) agent_name: Option<String>,
        pub(crate) user_id: Option<String>,
        pub(crate) tenant_id: Option<String>,
        pub(crate) idempotency_key: Option<String>,
        pub(crate) model_name: Option<String>,
    }

    /// Parse a DB task state string back to `TaskState`.
    /// Unknown strings fall back to `Pending` for forward-compatibility with future state additions.
    pub(crate) fn str_to_task_state(s: &str) -> TaskState {
        TaskState::from_db_str(s).unwrap_or(TaskState::Pending)
    }

    impl From<TaskRow> for DaemonTask {
        fn from(row: TaskRow) -> Self {
            Self {
                id: row.id,
                task: row.task,
                state: str_to_task_state(&row.state),
                created_at: row.created_at,
                started_at: row.started_at,
                completed_at: row.completed_at,
                result: row.result,
                error: row.error,
                tokens_used: TokenUsage {
                    input_tokens: row.input_tokens as u32,
                    output_tokens: row.output_tokens as u32,
                    cache_creation_input_tokens: row.cache_creation_input_tokens as u32,
                    cache_read_input_tokens: row.cache_read_input_tokens as u32,
                    reasoning_tokens: row.reasoning_tokens as u32,
                },
                tool_calls_made: row.tool_calls_made as usize,
                estimated_cost_usd: row.estimated_cost_usd,
                source: row.source,
                agent_name: row.agent_name,
                user_id: row.user_id,
                tenant_id: row.tenant_id,
                idempotency_key: row.idempotency_key,
                model_name: row.model_name,
            }
        }
    }

    /// PostgreSQL-backed daemon task store for durable task persistence.
    ///
    /// Uses `sqlx` runtime queries (no compile-time macros). Single table
    /// `daemon_tasks` with all lifecycle fields. Read-modify-write for `update()`.
    pub struct PostgresTaskStore {
        pool: sqlx::PgPool,
    }

    impl PostgresTaskStore {
        /// Create from an existing connection pool.
        pub fn new(pool: sqlx::PgPool) -> Self {
            Self { pool }
        }

        /// Connect to PostgreSQL using the given URL.
        pub async fn connect(database_url: &str) -> Result<Self, Error> {
            let pool = sqlx::PgPool::connect(database_url)
                .await
                .map_err(|e| Error::Daemon(format!("database connection failed: {e}")))?;
            Ok(Self { pool })
        }

        /// Run the daemon_tasks migration. Safe to call multiple times.
        pub async fn run_migration(&self) -> Result<(), Error> {
            // Split into separate statements — sqlx doesn't support multiple
            // commands in a single prepared statement.
            sqlx::query(
                r#"
            CREATE TABLE IF NOT EXISTS daemon_tasks (
                id                          UUID PRIMARY KEY,
                task                        TEXT NOT NULL,
                state                       TEXT NOT NULL DEFAULT 'pending',
                created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
                started_at                  TIMESTAMPTZ,
                completed_at                TIMESTAMPTZ,
                result                      TEXT,
                error                       TEXT,
                input_tokens                INTEGER NOT NULL DEFAULT 0,
                output_tokens               INTEGER NOT NULL DEFAULT 0,
                cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_input_tokens     INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens            INTEGER NOT NULL DEFAULT 0,
                tool_calls_made             INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd          DOUBLE PRECISION,
                source                      TEXT NOT NULL,
                agent_name                  TEXT,
                user_id                     TEXT,
                tenant_id                   TEXT,
                model_name                  TEXT
            )
            "#,
            )
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Daemon(format!("task table migration failed: {e}")))?;

            // Create indexes individually
            for index_sql in [
                "CREATE INDEX IF NOT EXISTS idx_daemon_tasks_created_at ON daemon_tasks(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_daemon_tasks_state ON daemon_tasks(state)",
                "CREATE INDEX IF NOT EXISTS idx_daemon_tasks_tenant_id ON daemon_tasks(tenant_id)",
                "CREATE INDEX IF NOT EXISTS idx_daemon_tasks_model_name ON daemon_tasks(model_name)",
                "CREATE INDEX IF NOT EXISTS idx_daemon_tasks_agent_name ON daemon_tasks(agent_name)",
                "CREATE INDEX IF NOT EXISTS idx_daemon_tasks_user_id ON daemon_tasks(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_daemon_tasks_source ON daemon_tasks(source)",
            ] {
                sqlx::query(index_sql)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| Error::Daemon(format!("index migration failed: {e}")))?;
            }

            // Add columns if not already present (for existing tables).
            for col in ["agent_name", "user_id", "tenant_id", "model_name"] {
                sqlx::query(&format!(
                    "ALTER TABLE daemon_tasks ADD COLUMN IF NOT EXISTS {col} TEXT"
                ))
                .execute(&self.pool)
                .await
                .map_err(|e| Error::Daemon(format!("{col} migration failed: {e}")))?;
            }

            // B5b: tighten tenant_id to NOT NULL DEFAULT '' for symmetry with
            // audit_log (B4) and so the partial unique idempotency index has a
            // guaranteed-present column to scope on.
            for stmt in [
                "UPDATE daemon_tasks SET tenant_id = '' WHERE tenant_id IS NULL",
                "ALTER TABLE daemon_tasks ALTER COLUMN tenant_id SET DEFAULT ''",
                "ALTER TABLE daemon_tasks ALTER COLUMN tenant_id SET NOT NULL",
                "ALTER TABLE daemon_tasks ADD COLUMN IF NOT EXISTS idempotency_key TEXT",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_daemon_tasks_idem \
                   ON daemon_tasks (tenant_id, idempotency_key) \
                   WHERE idempotency_key IS NOT NULL",
                "CREATE INDEX IF NOT EXISTS idx_daemon_tasks_created_at_for_sweep \
                   ON daemon_tasks (created_at) \
                   WHERE idempotency_key IS NOT NULL",
            ] {
                sqlx::query(stmt)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| Error::Daemon(format!("idempotency migration failed: {e}")))?;
            }

            Ok(())
        }
    }

    impl TaskStore for PostgresTaskStore {
        fn insert(&self, task: DaemonTask) -> Result<(), Error> {
            let pool = self.pool.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    sqlx::query(
                        r#"INSERT INTO daemon_tasks
                        (id, task, state, created_at, started_at, completed_at, result, error,
                         input_tokens, output_tokens, cache_creation_input_tokens,
                         cache_read_input_tokens, reasoning_tokens, tool_calls_made,
                         estimated_cost_usd, source, agent_name, user_id, tenant_id,
                         idempotency_key, model_name)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                            $16, $17, $18, $19, $20, $21)"#,
                    )
                    .bind(task.id)
                    .bind(&task.task)
                    .bind(task.state.as_str())
                    .bind(task.created_at)
                    .bind(task.started_at)
                    .bind(task.completed_at)
                    .bind(&task.result)
                    .bind(&task.error)
                    .bind(task.tokens_used.input_tokens as i32)
                    .bind(task.tokens_used.output_tokens as i32)
                    .bind(task.tokens_used.cache_creation_input_tokens as i32)
                    .bind(task.tokens_used.cache_read_input_tokens as i32)
                    .bind(task.tokens_used.reasoning_tokens as i32)
                    .bind(task.tool_calls_made as i32)
                    .bind(task.estimated_cost_usd)
                    .bind(&task.source)
                    .bind(&task.agent_name)
                    .bind(&task.user_id)
                    .bind(&task.tenant_id)
                    .bind(&task.idempotency_key)
                    .bind(&task.model_name)
                    .execute(&pool)
                    .await
                    .map_err(|e| Error::Daemon(format!("failed to insert task: {e}")))?;
                    Ok(())
                })
            })
        }

        fn get(&self, id: Uuid) -> Result<Option<DaemonTask>, Error> {
            let pool = self.pool.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    let row: Option<TaskRow> = sqlx::query_as(
                    "SELECT id, task, state, created_at, started_at, completed_at, result, error, \
                     input_tokens, output_tokens, cache_creation_input_tokens, \
                     cache_read_input_tokens, reasoning_tokens, tool_calls_made, \
                     estimated_cost_usd, source, agent_name, user_id, tenant_id, \
                     idempotency_key, model_name \
                     FROM daemon_tasks WHERE id = $1",
                )
                .bind(id)
                .fetch_optional(&pool)
                .await
                .map_err(|e| Error::Daemon(format!("failed to get task: {e}")))?;
                    Ok(row.map(DaemonTask::from))
                })
            })
        }

        fn list(&self, limit: usize, offset: usize) -> Result<(Vec<DaemonTask>, usize), Error> {
            let pool = self.pool.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM daemon_tasks")
                        .fetch_one(&pool)
                        .await
                        .map_err(|e| Error::Daemon(format!("failed to count tasks: {e}")))?;
                    let rows: Vec<TaskRow> = sqlx::query_as(
                    "SELECT id, task, state, created_at, started_at, completed_at, result, error, \
                     input_tokens, output_tokens, cache_creation_input_tokens, \
                     cache_read_input_tokens, reasoning_tokens, tool_calls_made, \
                     estimated_cost_usd, source, agent_name, user_id, tenant_id, \
                     idempotency_key, model_name \
                     FROM daemon_tasks ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                )
                .bind(limit as i64)
                .bind(offset as i64)
                .fetch_all(&pool)
                .await
                .map_err(|e| Error::Daemon(format!("failed to list tasks: {e}")))?;
                    let tasks = rows.into_iter().map(DaemonTask::from).collect();
                    Ok((tasks, total as usize))
                })
            })
        }

        fn update(&self, id: Uuid, f: &dyn Fn(&mut DaemonTask)) -> Result<(), Error> {
            let pool = self.pool.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    // Read the current task
                    let row: TaskRow = sqlx::query_as(
                    "SELECT id, task, state, created_at, started_at, completed_at, result, error, \
                     input_tokens, output_tokens, cache_creation_input_tokens, \
                     cache_read_input_tokens, reasoning_tokens, tool_calls_made, \
                     estimated_cost_usd, source, agent_name, user_id, tenant_id, \
                     idempotency_key, model_name \
                     FROM daemon_tasks WHERE id = $1",
                )
                .bind(id)
                .fetch_optional(&pool)
                .await
                .map_err(|e| Error::Daemon(format!("failed to read task for update: {e}")))?
                .ok_or_else(|| Error::Daemon(format!("task {id} not found")))?;

                    // Apply the mutation closure
                    let mut task = DaemonTask::from(row);
                    f(&mut task);

                    // Write back all fields
                    sqlx::query(
                        r#"UPDATE daemon_tasks SET
                        task = $2, state = $3, started_at = $4, completed_at = $5,
                        result = $6, error = $7, input_tokens = $8, output_tokens = $9,
                        cache_creation_input_tokens = $10, cache_read_input_tokens = $11,
                        reasoning_tokens = $12, tool_calls_made = $13,
                        estimated_cost_usd = $14, source = $15, agent_name = $16,
                        user_id = $17, tenant_id = $18, idempotency_key = $19, model_name = $20
                    WHERE id = $1"#,
                    )
                    .bind(task.id)
                    .bind(&task.task)
                    .bind(task.state.as_str())
                    .bind(task.started_at)
                    .bind(task.completed_at)
                    .bind(&task.result)
                    .bind(&task.error)
                    .bind(task.tokens_used.input_tokens as i32)
                    .bind(task.tokens_used.output_tokens as i32)
                    .bind(task.tokens_used.cache_creation_input_tokens as i32)
                    .bind(task.tokens_used.cache_read_input_tokens as i32)
                    .bind(task.tokens_used.reasoning_tokens as i32)
                    .bind(task.tool_calls_made as i32)
                    .bind(task.estimated_cost_usd)
                    .bind(&task.source)
                    .bind(&task.agent_name)
                    .bind(&task.user_id)
                    .bind(&task.tenant_id)
                    .bind(&task.idempotency_key)
                    .bind(&task.model_name)
                    .execute(&pool)
                    .await
                    .map_err(|e| Error::Daemon(format!("failed to update task: {e}")))?;
                    Ok(())
                })
            })
        }

        fn list_filtered(
            &self,
            limit: usize,
            offset: usize,
            source: Option<&str>,
            state: Option<TaskState>,
            tenant_id: Option<&str>,
        ) -> Result<(Vec<DaemonTask>, usize), Error> {
            let pool = self.pool.clone();
            let source_owned = source.map(String::from);
            let state_str = state.map(TaskState::as_str);
            let tenant_owned = tenant_id.map(String::from);
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                // Build dynamic WHERE clause
                let mut conditions = Vec::new();
                let mut param_idx = 1;

                if source_owned.is_some() {
                    conditions.push(format!("source = ${param_idx}"));
                    param_idx += 1;
                }
                if state_str.is_some() {
                    conditions.push(format!("state = ${param_idx}"));
                    param_idx += 1;
                }
                if tenant_owned.is_some() {
                    conditions.push(format!("tenant_id = ${param_idx}"));
                    param_idx += 1;
                }

                let where_clause = if conditions.is_empty() {
                    String::new()
                } else {
                    format!("WHERE {}", conditions.join(" AND "))
                };

                // Count query
                let count_sql = format!("SELECT COUNT(*) FROM daemon_tasks {where_clause}");
                let mut count_query = sqlx::query_scalar::<_, i64>(&count_sql);
                if let Some(ref s) = source_owned {
                    count_query = count_query.bind(s);
                }
                if let Some(st) = state_str {
                    count_query = count_query.bind(st);
                }
                if let Some(ref tid) = tenant_owned {
                    count_query = count_query.bind(tid);
                }
                let total: i64 = count_query
                    .fetch_one(&pool)
                    .await
                    .map_err(|e| Error::Daemon(format!("failed to count filtered tasks: {e}")))?;

                // Data query
                let data_sql = format!(
                    "SELECT id, task, state, created_at, started_at, completed_at, result, error, \
                     input_tokens, output_tokens, cache_creation_input_tokens, \
                     cache_read_input_tokens, reasoning_tokens, tool_calls_made, \
                     estimated_cost_usd, source, agent_name, user_id, tenant_id, \
                     idempotency_key, model_name \
                     FROM daemon_tasks {where_clause} ORDER BY created_at DESC \
                     LIMIT ${param_idx} OFFSET ${}",
                    param_idx + 1
                );
                let mut data_query = sqlx::query_as::<_, TaskRow>(&data_sql);
                if let Some(ref s) = source_owned {
                    data_query = data_query.bind(s);
                }
                if let Some(st) = state_str {
                    data_query = data_query.bind(st);
                }
                if let Some(ref tid) = tenant_owned {
                    data_query = data_query.bind(tid);
                }
                data_query = data_query.bind(limit as i64).bind(offset as i64);

                let rows: Vec<TaskRow> = data_query
                    .fetch_all(&pool)
                    .await
                    .map_err(|e| Error::Daemon(format!("failed to list filtered tasks: {e}")))?;
                let tasks = rows.into_iter().map(DaemonTask::from).collect();
                Ok((tasks, total as usize))
            })
            })
        }

        fn stats(&self, tenant_id: Option<&str>) -> Result<TaskStats, Error> {
            let pool = self.pool.clone();
            let tenant_owned = tenant_id.map(String::from);
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    // Single query with aggregation grouped by state and source
                    #[derive(sqlx::FromRow)]
                    struct StatsRow {
                        state: String,
                        source: String,
                        cnt: i64,
                        sum_input: i64,
                        sum_output: i64,
                        sum_cache_read: i64,
                        sum_cache_creation: i64,
                        sum_cost: f64,
                    }
                    let (sql, rows): (String, Vec<StatsRow>) = if let Some(ref tid) = tenant_owned {
                        let sql = "SELECT state, source, COUNT(*) AS cnt, \
                         COALESCE(SUM(input_tokens), 0) AS sum_input, \
                         COALESCE(SUM(output_tokens), 0) AS sum_output, \
                         COALESCE(SUM(cache_read_input_tokens), 0) AS sum_cache_read, \
                         COALESCE(SUM(cache_creation_input_tokens), 0) AS sum_cache_creation, \
                         COALESCE(SUM(estimated_cost_usd), 0.0) AS sum_cost \
                         FROM daemon_tasks WHERE tenant_id = $1 GROUP BY state, source"
                            .to_string();
                        let rows = sqlx::query_as(&sql)
                            .bind(tid)
                            .fetch_all(&pool)
                            .await
                            .map_err(|e| Error::Daemon(format!("failed to compute stats: {e}")))?;
                        (sql, rows)
                    } else {
                        let sql = "SELECT state, source, COUNT(*) AS cnt, \
                         COALESCE(SUM(input_tokens), 0) AS sum_input, \
                         COALESCE(SUM(output_tokens), 0) AS sum_output, \
                         COALESCE(SUM(cache_read_input_tokens), 0) AS sum_cache_read, \
                         COALESCE(SUM(cache_creation_input_tokens), 0) AS sum_cache_creation, \
                         COALESCE(SUM(estimated_cost_usd), 0.0) AS sum_cost \
                         FROM daemon_tasks GROUP BY state, source"
                            .to_string();
                        let rows = sqlx::query_as(&sql)
                            .fetch_all(&pool)
                            .await
                            .map_err(|e| Error::Daemon(format!("failed to compute stats: {e}")))?;
                        (sql, rows)
                    };
                    let _ = sql; // used for query lifetime

                    let mut stats = TaskStats::default();
                    for row in &rows {
                        let count = row.cnt as usize;
                        stats.total_tasks += count;
                        *stats.tasks_by_state.entry(row.state.clone()).or_default() += count;
                        *stats.tasks_by_source.entry(row.source.clone()).or_default() += count;
                        if row.state == TaskState::Running.as_str() {
                            stats.active_tasks += count;
                        }
                        stats.total_input_tokens += row.sum_input as u64;
                        stats.total_output_tokens += row.sum_output as u64;
                        stats.total_cache_read_tokens += row.sum_cache_read as u64;
                        stats.total_cache_creation_tokens += row.sum_cache_creation as u64;
                        stats.total_estimated_cost_usd += row.sum_cost;
                    }
                    Ok(stats)
                })
            })
        }

        fn usage_stats(&self, query: &UsageQuery) -> Result<Vec<UsageRow>, Error> {
            let pool = self.pool.clone();
            let query = query.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    // Build dynamic SQL
                    let mut conditions = Vec::new();
                    let mut param_idx = 1u32;

                    if query.from.is_some() {
                        conditions.push(format!("created_at >= ${param_idx}"));
                        param_idx += 1;
                    }
                    if query.to.is_some() {
                        conditions.push(format!("created_at < ${param_idx}"));
                        param_idx += 1;
                    }
                    if query.tenant_id.is_some() {
                        conditions.push(format!("tenant_id = ${param_idx}"));
                        param_idx += 1;
                    }
                    if query.user_id.is_some() {
                        conditions.push(format!("user_id = ${param_idx}"));
                        param_idx += 1;
                    }
                    if query.agent_name.is_some() {
                        conditions.push(format!("agent_name = ${param_idx}"));
                        param_idx += 1;
                    }
                    if query.model_name.is_some() {
                        conditions.push(format!("model_name = ${param_idx}"));
                        param_idx += 1;
                    }
                    if query.source.is_some() {
                        conditions.push(format!("source = ${param_idx}"));
                        param_idx += 1;
                    }
                    let _ = param_idx;

                    let where_clause = if conditions.is_empty() {
                        String::new()
                    } else {
                        format!("WHERE {}", conditions.join(" AND "))
                    };

                    let group_col = match query.group_by {
                        Some(UsageGroupBy::Agent) => Some("agent_name"),
                        Some(UsageGroupBy::Model) => Some("model_name"),
                        Some(UsageGroupBy::User) => Some("user_id"),
                        Some(UsageGroupBy::Tenant) => Some("tenant_id"),
                        Some(UsageGroupBy::Source) => Some("source"),
                        Some(UsageGroupBy::Day) => {
                            Some("DATE_TRUNC('day', created_at)::date::text")
                        }
                        None => None,
                    };

                    let (select_key, group_by_clause) = match group_col {
                        Some(col) => (format!("{col} AS group_key"), format!("GROUP BY {col}")),
                        None => ("NULL AS group_key".to_string(), String::new()),
                    };

                    let sql = format!(
                        "SELECT {select_key}, \
                         COUNT(*) AS task_count, \
                         COUNT(*) FILTER (WHERE state = 'completed') AS completed_count, \
                         COUNT(*) FILTER (WHERE state = 'failed') AS failed_count, \
                         COALESCE(SUM(input_tokens), 0) AS input_tokens, \
                         COALESCE(SUM(output_tokens), 0) AS output_tokens, \
                         COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_tokens, \
                         COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_tokens, \
                         COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens, \
                         COALESCE(SUM(tool_calls_made), 0) AS tool_calls, \
                         COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd, \
                         AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) \
                           FILTER (WHERE completed_at IS NOT NULL AND started_at IS NOT NULL) \
                           AS avg_duration_secs \
                         FROM daemon_tasks {where_clause} {group_by_clause} \
                         ORDER BY group_key NULLS FIRST"
                    );

                    #[derive(sqlx::FromRow)]
                    struct Row {
                        group_key: Option<String>,
                        task_count: i64,
                        completed_count: i64,
                        failed_count: i64,
                        input_tokens: i64,
                        output_tokens: i64,
                        cache_read_tokens: i64,
                        cache_creation_tokens: i64,
                        reasoning_tokens: i64,
                        tool_calls: i64,
                        estimated_cost_usd: f64,
                        avg_duration_secs: Option<f64>,
                    }

                    let mut qb = sqlx::query_as::<_, Row>(&sql);
                    // Bind parameters in the same order as conditions
                    if let Some(ref v) = query.from {
                        qb = qb.bind(v);
                    }
                    if let Some(ref v) = query.to {
                        qb = qb.bind(v);
                    }
                    if let Some(ref v) = query.tenant_id {
                        qb = qb.bind(v);
                    }
                    if let Some(ref v) = query.user_id {
                        qb = qb.bind(v);
                    }
                    if let Some(ref v) = query.agent_name {
                        qb = qb.bind(v);
                    }
                    if let Some(ref v) = query.model_name {
                        qb = qb.bind(v);
                    }
                    if let Some(ref v) = query.source {
                        qb = qb.bind(v);
                    }

                    let rows: Vec<Row> = qb
                        .fetch_all(&pool)
                        .await
                        .map_err(|e| Error::Daemon(format!("usage_stats query failed: {e}")))?;

                    if rows.is_empty() {
                        return Ok(vec![UsageRow::default()]);
                    }

                    Ok(rows
                        .into_iter()
                        .map(|r| UsageRow {
                            group_key: r.group_key,
                            task_count: r.task_count as u64,
                            completed_count: r.completed_count as u64,
                            failed_count: r.failed_count as u64,
                            input_tokens: r.input_tokens as u64,
                            output_tokens: r.output_tokens as u64,
                            cache_read_tokens: r.cache_read_tokens as u64,
                            cache_creation_tokens: r.cache_creation_tokens as u64,
                            reasoning_tokens: r.reasoning_tokens as u64,
                            tool_calls: r.tool_calls as u64,
                            estimated_cost_usd: r.estimated_cost_usd,
                            avg_duration_secs: r.avg_duration_secs,
                        })
                        .collect())
                })
            })
        }

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
                         estimated_cost_usd, source, agent_name, user_id, tenant_id, \
                         idempotency_key, model_name \
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
    }
} // mod postgres_store

#[cfg(feature = "postgres")]
pub use postgres_store::PostgresTaskStore;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let store = InMemoryTaskStore::new();
        let id = Uuid::new_v4();
        let task = DaemonTask::new(id, "test task", "api");
        store.insert(task).unwrap();

        let fetched = store.get(id).unwrap().unwrap();
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.task, "test task");
        assert_eq!(fetched.state, TaskState::Pending);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let store = InMemoryTaskStore::new();
        let result = store.get(Uuid::new_v4()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn insert_duplicate_rejected() {
        let store = InMemoryTaskStore::new();
        let id = Uuid::new_v4();
        store.insert(DaemonTask::new(id, "first", "api")).unwrap();
        let err = store
            .insert(DaemonTask::new(id, "second", "api"))
            .unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn list_newest_first() {
        let store = InMemoryTaskStore::new();
        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        for (i, &id) in ids.iter().enumerate() {
            store
                .insert(DaemonTask::new(id, format!("task {i}"), "api"))
                .unwrap();
        }

        let (tasks, total) = store.list(3, 0).unwrap();
        assert_eq!(total, 5);
        assert_eq!(tasks.len(), 3);
        // Newest first (reversed insertion order)
        assert_eq!(tasks[0].id, ids[4]);
        assert_eq!(tasks[1].id, ids[3]);
        assert_eq!(tasks[2].id, ids[2]);
    }

    #[test]
    fn list_with_offset() {
        let store = InMemoryTaskStore::new();
        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        for (i, &id) in ids.iter().enumerate() {
            store
                .insert(DaemonTask::new(id, format!("task {i}"), "api"))
                .unwrap();
        }

        let (tasks, total) = store.list(2, 2).unwrap();
        assert_eq!(total, 5);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].id, ids[2]);
        assert_eq!(tasks[1].id, ids[1]);
    }

    #[test]
    fn list_empty_store() {
        let store = InMemoryTaskStore::new();
        let (tasks, total) = store.list(10, 0).unwrap();
        assert_eq!(total, 0);
        assert!(tasks.is_empty());
    }

    #[test]
    fn update_modifies_task() {
        let store = InMemoryTaskStore::new();
        let id = Uuid::new_v4();
        store.insert(DaemonTask::new(id, "test", "api")).unwrap();

        store
            .update(id, &|t| {
                t.state = TaskState::Running;
                t.started_at = Some(chrono::Utc::now());
            })
            .unwrap();

        let task = store.get(id).unwrap().unwrap();
        assert_eq!(task.state, TaskState::Running);
        assert!(task.started_at.is_some());
    }

    #[test]
    fn update_nonexistent_returns_error() {
        let store = InMemoryTaskStore::new();
        let err = store.update(Uuid::new_v4(), &|_| {}).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn update_completion_with_tokens() {
        let store = InMemoryTaskStore::new();
        let id = Uuid::new_v4();
        store.insert(DaemonTask::new(id, "test", "api")).unwrap();

        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        store
            .update(id, &|t| {
                t.state = TaskState::Completed;
                t.completed_at = Some(chrono::Utc::now());
                t.result = Some("done".into());
                t.tokens_used = usage;
                t.tool_calls_made = 5;
                t.estimated_cost_usd = Some(0.001);
            })
            .unwrap();

        let task = store.get(id).unwrap().unwrap();
        assert_eq!(task.state, TaskState::Completed);
        assert_eq!(task.result.as_deref(), Some("done"));
        assert_eq!(task.tokens_used.input_tokens, 100);
        assert_eq!(task.tool_calls_made, 5);
        assert_eq!(task.estimated_cost_usd, Some(0.001));
    }

    #[test]
    fn concurrent_insert_and_read() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(InMemoryTaskStore::new());
        let mut handles = Vec::new();

        // Spawn 10 threads each inserting a task
        for i in 0..10 {
            let store = store.clone();
            handles.push(thread::spawn(move || {
                let id = Uuid::new_v4();
                store
                    .insert(DaemonTask::new(id, format!("task {i}"), "api"))
                    .unwrap();
                id
            }));
        }

        let ids: Vec<Uuid> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All tasks should be retrievable
        for id in &ids {
            assert!(store.get(*id).unwrap().is_some());
        }

        let (_, total) = store.list(100, 0).unwrap();
        assert_eq!(total, 10);
    }

    // --- PostgresTaskStore unit tests (row conversion, no DB needed) ---

    #[cfg(feature = "postgres")]
    use postgres_store::*;

    #[cfg(feature = "postgres")]
    #[test]
    fn task_state_str_roundtrip() {
        for state in [
            TaskState::Pending,
            TaskState::Running,
            TaskState::Completed,
            TaskState::Failed,
            TaskState::Cancelled,
            TaskState::InputRequired,
            TaskState::AuthRequired,
            TaskState::Rejected,
        ] {
            let s = state.as_str();
            let back = str_to_task_state(s);
            assert_eq!(back, state, "roundtrip failed for {s}");
        }
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn str_to_task_state_unknown_defaults_to_pending() {
        assert_eq!(str_to_task_state("bogus"), TaskState::Pending);
        assert_eq!(str_to_task_state(""), TaskState::Pending);
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn task_row_to_daemon_task_pending() {
        let id = Uuid::new_v4();
        let now = chrono::Utc::now();
        let row = TaskRow {
            id,
            task: "analyze code".into(),
            state: "pending".into(),
            created_at: now,
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            reasoning_tokens: 0,
            tool_calls_made: 0,
            estimated_cost_usd: None,
            source: "api".into(),
            agent_name: None,
            user_id: None,
            tenant_id: None,
            idempotency_key: None,
            model_name: None,
        };
        let task = DaemonTask::from(row);
        assert_eq!(task.id, id);
        assert_eq!(task.task, "analyze code");
        assert_eq!(task.state, TaskState::Pending);
        assert_eq!(task.created_at, now);
        assert!(task.started_at.is_none());
        assert!(task.result.is_none());
        assert_eq!(task.tokens_used, TokenUsage::default());
        assert_eq!(task.tool_calls_made, 0);
        assert_eq!(task.source, "api");
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn task_row_to_daemon_task_completed() {
        let id = Uuid::new_v4();
        let now = chrono::Utc::now();
        let row = TaskRow {
            id,
            task: "build report".into(),
            state: "completed".into(),
            created_at: now,
            started_at: Some(now),
            completed_at: Some(now),
            result: Some("report done".into()),
            error: None,
            input_tokens: 1000,
            output_tokens: 500,
            cache_creation_input_tokens: 200,
            cache_read_input_tokens: 300,
            reasoning_tokens: 150,
            tool_calls_made: 7,
            estimated_cost_usd: Some(0.042),
            source: "cron:daily".into(),
            agent_name: None,
            user_id: None,
            tenant_id: None,
            idempotency_key: None,
            model_name: None,
        };
        let task = DaemonTask::from(row);
        assert_eq!(task.state, TaskState::Completed);
        assert_eq!(task.result.as_deref(), Some("report done"));
        assert_eq!(task.tokens_used.input_tokens, 1000);
        assert_eq!(task.tokens_used.output_tokens, 500);
        assert_eq!(task.tokens_used.cache_creation_input_tokens, 200);
        assert_eq!(task.tokens_used.cache_read_input_tokens, 300);
        assert_eq!(task.tokens_used.reasoning_tokens, 150);
        assert_eq!(task.tool_calls_made, 7);
        assert_eq!(task.estimated_cost_usd, Some(0.042));
        assert_eq!(task.source, "cron:daily");
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn task_row_to_daemon_task_failed() {
        let now = chrono::Utc::now();
        let row = TaskRow {
            id: Uuid::new_v4(),
            task: "failing task".into(),
            state: "failed".into(),
            created_at: now,
            started_at: Some(now),
            completed_at: Some(now),
            result: None,
            error: Some("out of tokens".into()),
            input_tokens: 50,
            output_tokens: 10,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            reasoning_tokens: 0,
            tool_calls_made: 1,
            estimated_cost_usd: Some(0.001),
            source: "sensor:email".into(),
            agent_name: None,
            user_id: None,
            tenant_id: None,
            idempotency_key: None,
            model_name: None,
        };
        let task = DaemonTask::from(row);
        assert_eq!(task.state, TaskState::Failed);
        assert!(task.result.is_none());
        assert_eq!(task.error.as_deref(), Some("out of tokens"));
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn task_row_preserves_timestamps() {
        let created = chrono::Utc::now();
        let started = created + chrono::Duration::seconds(1);
        let completed = created + chrono::Duration::seconds(5);
        let row = TaskRow {
            id: Uuid::new_v4(),
            task: "timed task".into(),
            state: "completed".into(),
            created_at: created,
            started_at: Some(started),
            completed_at: Some(completed),
            result: None,
            error: None,
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            reasoning_tokens: 0,
            tool_calls_made: 0,
            estimated_cost_usd: None,
            source: "api".into(),
            agent_name: None,
            user_id: None,
            tenant_id: None,
            idempotency_key: None,
            model_name: None,
        };
        let task = DaemonTask::from(row);
        assert_eq!(task.created_at, created);
        assert_eq!(task.started_at, Some(started));
        assert_eq!(task.completed_at, Some(completed));
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn task_row_with_agent_name() {
        let row = TaskRow {
            id: Uuid::new_v4(),
            task: "named task".into(),
            state: "running".into(),
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            reasoning_tokens: 0,
            tool_calls_made: 0,
            estimated_cost_usd: None,
            source: "api".into(),
            agent_name: Some("security-bot".into()),
            user_id: Some("user-42".into()),
            tenant_id: Some("tenant-a".into()),
            idempotency_key: None,
            model_name: None,
        };
        let task = DaemonTask::from(row);
        assert_eq!(task.agent_name.as_deref(), Some("security-bot"));
    }

    // --- list_filtered tests ---

    #[test]
    fn list_filtered_by_source() {
        let store = InMemoryTaskStore::new();
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "t1", "api"))
            .unwrap();
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "t2", "cron"))
            .unwrap();
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "t3", "api"))
            .unwrap();

        let (tasks, total) = store.list_filtered(10, 0, Some("api"), None, None).unwrap();
        assert_eq!(total, 2);
        assert_eq!(tasks.len(), 2);
        for t in &tasks {
            assert_eq!(t.source, "api");
        }
    }

    #[test]
    fn list_filtered_by_state() {
        let store = InMemoryTaskStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        store.insert(DaemonTask::new(id1, "t1", "api")).unwrap();
        store.insert(DaemonTask::new(id2, "t2", "api")).unwrap();
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "t3", "api"))
            .unwrap();
        store
            .update(id1, &|t| t.state = TaskState::Running)
            .unwrap();
        store
            .update(id2, &|t| t.state = TaskState::Running)
            .unwrap();

        let (tasks, total) = store
            .list_filtered(10, 0, None, Some(TaskState::Running), None)
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(tasks.len(), 2);
        for t in &tasks {
            assert_eq!(t.state, TaskState::Running);
        }
    }

    #[test]
    fn list_filtered_by_both() {
        let store = InMemoryTaskStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        store.insert(DaemonTask::new(id1, "t1", "api")).unwrap();
        store.insert(DaemonTask::new(id2, "t2", "cron")).unwrap();
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "t3", "api"))
            .unwrap();
        store
            .update(id1, &|t| t.state = TaskState::Running)
            .unwrap();
        store
            .update(id2, &|t| t.state = TaskState::Running)
            .unwrap();

        let (tasks, total) = store
            .list_filtered(10, 0, Some("api"), Some(TaskState::Running), None)
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, id1);
    }

    #[test]
    fn list_filtered_none_returns_all() {
        let store = InMemoryTaskStore::new();
        for i in 0..4 {
            store
                .insert(DaemonTask::new(Uuid::new_v4(), format!("t{i}"), "api"))
                .unwrap();
        }

        let (tasks, total) = store.list_filtered(10, 0, None, None, None).unwrap();
        assert_eq!(total, 4);
        assert_eq!(tasks.len(), 4);
    }

    #[test]
    fn list_filtered_pagination() {
        let store = InMemoryTaskStore::new();
        for i in 0..5 {
            store
                .insert(DaemonTask::new(Uuid::new_v4(), format!("t{i}"), "api"))
                .unwrap();
        }

        let (tasks, total) = store.list_filtered(2, 1, Some("api"), None, None).unwrap();
        assert_eq!(total, 5);
        assert_eq!(tasks.len(), 2);
    }

    // --- stats tests ---

    #[test]
    fn stats_empty_store() {
        let store = InMemoryTaskStore::new();
        let stats = store.stats(None).unwrap();
        assert_eq!(stats.total_tasks, 0);
        assert!(stats.tasks_by_state.is_empty());
        assert!(stats.tasks_by_source.is_empty());
        assert_eq!(stats.active_tasks, 0);
        assert_eq!(stats.total_input_tokens, 0);
        assert_eq!(stats.total_output_tokens, 0);
        assert_eq!(stats.total_cache_read_tokens, 0);
        assert_eq!(stats.total_cache_creation_tokens, 0);
        assert_eq!(stats.total_estimated_cost_usd, 0.0);
    }

    #[test]
    fn stats_aggregates_correctly() {
        let store = InMemoryTaskStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        store.insert(DaemonTask::new(id1, "t1", "api")).unwrap();
        store.insert(DaemonTask::new(id2, "t2", "cron")).unwrap();
        store.insert(DaemonTask::new(id3, "t3", "api")).unwrap();

        store
            .update(id1, &|t| {
                t.state = TaskState::Running;
                t.tokens_used.input_tokens = 100;
                t.tokens_used.output_tokens = 50;
            })
            .unwrap();
        store
            .update(id2, &|t| {
                t.state = TaskState::Completed;
                t.tokens_used.input_tokens = 200;
                t.tokens_used.output_tokens = 80;
                t.tokens_used.cache_read_input_tokens = 30;
                t.tokens_used.cache_creation_input_tokens = 10;
                t.estimated_cost_usd = Some(0.05);
            })
            .unwrap();
        store
            .update(id3, &|t| {
                t.state = TaskState::Failed;
                t.tokens_used.input_tokens = 50;
                t.estimated_cost_usd = Some(0.01);
            })
            .unwrap();

        let stats = store.stats(None).unwrap();
        assert_eq!(stats.total_tasks, 3);
        assert_eq!(
            stats.tasks_by_state.get(TaskState::Running.as_str()),
            Some(&1)
        );
        assert_eq!(
            stats.tasks_by_state.get(TaskState::Completed.as_str()),
            Some(&1)
        );
        assert_eq!(
            stats.tasks_by_state.get(TaskState::Failed.as_str()),
            Some(&1)
        );
        assert_eq!(stats.active_tasks, 1); // only Running
        assert_eq!(stats.total_input_tokens, 350);
        assert_eq!(stats.total_output_tokens, 130);
        assert_eq!(stats.total_cache_read_tokens, 30);
        assert_eq!(stats.total_cache_creation_tokens, 10);
        assert!((stats.total_estimated_cost_usd - 0.06).abs() < 1e-9);
    }

    #[test]
    fn stats_by_source_breakdown() {
        let store = InMemoryTaskStore::new();
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "t1", "api"))
            .unwrap();
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "t2", "api"))
            .unwrap();
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "t3", "cron"))
            .unwrap();
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "t4", "sensor:rss"))
            .unwrap();

        let stats = store.stats(None).unwrap();
        assert_eq!(stats.tasks_by_source.get("api"), Some(&2));
        assert_eq!(stats.tasks_by_source.get("cron"), Some(&1));
        assert_eq!(stats.tasks_by_source.get("sensor:rss"), Some(&1));
    }

    #[test]
    fn stats_active_count() {
        let store = InMemoryTaskStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        store.insert(DaemonTask::new(id1, "t1", "api")).unwrap();
        store.insert(DaemonTask::new(id2, "t2", "api")).unwrap();
        store.insert(DaemonTask::new(id3, "t3", "api")).unwrap();

        store
            .update(id1, &|t| t.state = TaskState::Running)
            .unwrap();
        store
            .update(id2, &|t| t.state = TaskState::Running)
            .unwrap();
        // id3 stays Pending

        let stats = store.stats(None).unwrap();
        assert_eq!(stats.active_tasks, 2);
        assert_eq!(
            stats.tasks_by_state.get(TaskState::Running.as_str()),
            Some(&2)
        );
        assert_eq!(
            stats.tasks_by_state.get(TaskState::Pending.as_str()),
            Some(&1)
        );
    }

    // --- tenant filter tests ---

    #[test]
    fn list_filtered_by_tenant() {
        let store = InMemoryTaskStore::new();
        store
            .insert(DaemonTask::new_with_user(
                Uuid::new_v4(),
                "t1",
                "api",
                "alice",
                "acme",
            ))
            .unwrap();
        store
            .insert(DaemonTask::new_with_user(
                Uuid::new_v4(),
                "t2",
                "api",
                "bob",
                "acme",
            ))
            .unwrap();
        store
            .insert(DaemonTask::new_with_user(
                Uuid::new_v4(),
                "t3",
                "api",
                "carol",
                "globex",
            ))
            .unwrap();

        let (tasks, total) = store
            .list_filtered(10, 0, None, None, Some("acme"))
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().all(|t| t.tenant_id.as_deref() == Some("acme")));

        let (tasks, total) = store
            .list_filtered(10, 0, None, None, Some("globex"))
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].tenant_id.as_deref(), Some("globex"));
    }

    #[test]
    fn list_filtered_tenant_with_pagination() {
        let store = InMemoryTaskStore::new();
        for i in 0..5 {
            store
                .insert(DaemonTask::new_with_user(
                    Uuid::new_v4(),
                    format!("t{i}"),
                    "api",
                    "alice",
                    "acme",
                ))
                .unwrap();
        }
        store
            .insert(DaemonTask::new_with_user(
                Uuid::new_v4(),
                "other",
                "api",
                "dave",
                "globex",
            ))
            .unwrap();

        // Page 1 of acme tasks (limit 2)
        let (tasks, total) = store.list_filtered(2, 0, None, None, Some("acme")).unwrap();
        assert_eq!(total, 5);
        assert_eq!(tasks.len(), 2);

        // Page 3 of acme tasks (only 1 remaining)
        let (tasks, total) = store.list_filtered(2, 4, None, None, Some("acme")).unwrap();
        assert_eq!(total, 5);
        assert_eq!(tasks.len(), 1);
    }

    #[test]
    fn stats_filtered_by_tenant() {
        let store = InMemoryTaskStore::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        store
            .insert(DaemonTask::new_with_user(id1, "t1", "api", "alice", "acme"))
            .unwrap();
        store
            .insert(DaemonTask::new_with_user(
                id2, "t2", "telegram", "bob", "acme",
            ))
            .unwrap();
        store
            .insert(DaemonTask::new_with_user(
                id3, "t3", "api", "carol", "globex",
            ))
            .unwrap();

        store
            .update(id1, &|t| {
                t.state = TaskState::Running;
                t.tokens_used.input_tokens = 100;
            })
            .unwrap();
        store
            .update(id3, &|t| {
                t.state = TaskState::Completed;
                t.tokens_used.input_tokens = 200;
            })
            .unwrap();

        // acme-scoped stats
        let stats = store.stats(Some("acme")).unwrap();
        assert_eq!(stats.total_tasks, 2);
        assert_eq!(stats.active_tasks, 1);
        assert_eq!(stats.total_input_tokens, 100);
        assert_eq!(stats.tasks_by_source.get("api"), Some(&1));
        assert_eq!(stats.tasks_by_source.get("telegram"), Some(&1));

        // globex-scoped stats
        let stats = store.stats(Some("globex")).unwrap();
        assert_eq!(stats.total_tasks, 1);
        assert_eq!(stats.active_tasks, 0);
        assert_eq!(stats.total_input_tokens, 200);

        // unscoped stats (all tenants)
        let stats = store.stats(None).unwrap();
        assert_eq!(stats.total_tasks, 3);
        assert_eq!(stats.total_input_tokens, 300);
    }

    #[test]
    fn list_filtered_tenant_none_includes_tasks_without_tenant() {
        let store = InMemoryTaskStore::new();
        // Task without tenant (old-style)
        store
            .insert(DaemonTask::new(Uuid::new_v4(), "old", "api"))
            .unwrap();
        // Task with tenant
        store
            .insert(DaemonTask::new_with_user(
                Uuid::new_v4(),
                "new",
                "api",
                "alice",
                "acme",
            ))
            .unwrap();

        // No tenant filter: both visible
        let (_, total) = store.list_filtered(10, 0, None, None, None).unwrap();
        assert_eq!(total, 2);

        // Tenant filter: only matching
        let (_, total) = store
            .list_filtered(10, 0, None, None, Some("acme"))
            .unwrap();
        assert_eq!(total, 1);
    }

    // --- usage_stats tests ---

    fn make_task_with(
        source: &str,
        state: TaskState,
        agent: Option<&str>,
        model: Option<&str>,
        tokens: u32,
        cost: Option<f64>,
        created_at: DateTime<Utc>,
    ) -> DaemonTask {
        let mut t = DaemonTask::new(Uuid::new_v4(), "task", source);
        t.state = state;
        t.agent_name = agent.map(String::from);
        t.model_name = model.map(String::from);
        t.tokens_used.input_tokens = tokens;
        t.tokens_used.output_tokens = tokens / 2;
        t.estimated_cost_usd = cost;
        t.created_at = created_at;
        if state == TaskState::Completed || state == TaskState::Failed {
            t.started_at = Some(created_at);
            t.completed_at = Some(created_at + chrono::Duration::seconds(10));
        }
        t
    }

    #[test]
    fn usage_stats_empty_store() {
        let store = InMemoryTaskStore::new();
        let rows = store.usage_stats(&UsageQuery::default()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].task_count, 0);
        assert_eq!(rows[0].estimated_cost_usd, 0.0);
        assert!(rows[0].group_key.is_none());
    }

    #[test]
    fn usage_stats_no_group_single_total() {
        let store = InMemoryTaskStore::new();
        let now = Utc::now();
        store
            .insert(make_task_with(
                "api",
                TaskState::Completed,
                Some("bot-a"),
                Some("claude-sonnet-4-6-20250610"),
                100,
                Some(0.01),
                now,
            ))
            .unwrap();
        store
            .insert(make_task_with(
                "api",
                TaskState::Failed,
                Some("bot-b"),
                Some("claude-haiku-4-5-20251001"),
                50,
                Some(0.005),
                now,
            ))
            .unwrap();

        let rows = store.usage_stats(&UsageQuery::default()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].task_count, 2);
        assert_eq!(rows[0].completed_count, 1);
        assert_eq!(rows[0].failed_count, 1);
        assert_eq!(rows[0].input_tokens, 150);
        assert!((rows[0].estimated_cost_usd - 0.015).abs() < 1e-9);
        assert!(rows[0].avg_duration_secs.is_some());
    }

    #[test]
    fn usage_stats_time_range_filter() {
        let store = InMemoryTaskStore::new();
        let t1 = Utc::now() - chrono::Duration::hours(48);
        let t2 = Utc::now() - chrono::Duration::hours(1);
        store
            .insert(make_task_with(
                "api",
                TaskState::Completed,
                None,
                None,
                100,
                Some(0.01),
                t1,
            ))
            .unwrap();
        store
            .insert(make_task_with(
                "api",
                TaskState::Completed,
                None,
                None,
                200,
                Some(0.02),
                t2,
            ))
            .unwrap();

        // Only recent task in range
        let rows = store
            .usage_stats(&UsageQuery {
                from: Some(Utc::now() - chrono::Duration::hours(24)),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].task_count, 1);
        assert_eq!(rows[0].input_tokens, 200);
    }

    #[test]
    fn usage_stats_group_by_agent() {
        let store = InMemoryTaskStore::new();
        let now = Utc::now();
        for agent in &["bot-a", "bot-a", "bot-b"] {
            store
                .insert(make_task_with(
                    "api",
                    TaskState::Completed,
                    Some(agent),
                    None,
                    100,
                    Some(0.01),
                    now,
                ))
                .unwrap();
        }

        let rows = store
            .usage_stats(&UsageQuery {
                group_by: Some(UsageGroupBy::Agent),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 2);
        // Sorted: bot-a, bot-b
        assert_eq!(rows[0].group_key.as_deref(), Some("bot-a"));
        assert_eq!(rows[0].task_count, 2);
        assert_eq!(rows[1].group_key.as_deref(), Some("bot-b"));
        assert_eq!(rows[1].task_count, 1);
    }

    #[test]
    fn usage_stats_group_by_model() {
        let store = InMemoryTaskStore::new();
        let now = Utc::now();
        store
            .insert(make_task_with(
                "api",
                TaskState::Completed,
                None,
                Some("claude-sonnet-4-6-20250610"),
                100,
                Some(0.01),
                now,
            ))
            .unwrap();
        store
            .insert(make_task_with(
                "api",
                TaskState::Completed,
                None,
                Some("claude-haiku-4-5-20251001"),
                50,
                Some(0.002),
                now,
            ))
            .unwrap();

        let rows = store
            .usage_stats(&UsageQuery {
                group_by: Some(UsageGroupBy::Model),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0].group_key.as_deref(),
            Some("claude-haiku-4-5-20251001")
        );
        assert_eq!(
            rows[1].group_key.as_deref(),
            Some("claude-sonnet-4-6-20250610")
        );
    }

    #[test]
    fn usage_stats_group_by_day() {
        let store = InMemoryTaskStore::new();
        let day1 = DateTime::parse_from_rfc3339("2026-03-01T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let day2 = DateTime::parse_from_rfc3339("2026-03-02T15:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        store
            .insert(make_task_with(
                "api",
                TaskState::Completed,
                None,
                None,
                100,
                Some(0.01),
                day1,
            ))
            .unwrap();
        store
            .insert(make_task_with(
                "api",
                TaskState::Completed,
                None,
                None,
                200,
                Some(0.02),
                day2,
            ))
            .unwrap();

        let rows = store
            .usage_stats(&UsageQuery {
                group_by: Some(UsageGroupBy::Day),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].group_key.as_deref(), Some("2026-03-01"));
        assert_eq!(rows[1].group_key.as_deref(), Some("2026-03-02"));
    }

    #[test]
    fn usage_stats_combined_filters() {
        let store = InMemoryTaskStore::new();
        let now = Utc::now();
        let mut t = make_task_with(
            "api",
            TaskState::Completed,
            None,
            None,
            100,
            Some(0.01),
            now,
        );
        t.tenant_id = Some("acme".into());
        store.insert(t).unwrap();

        let mut t = make_task_with(
            "cron",
            TaskState::Completed,
            None,
            None,
            200,
            Some(0.02),
            now,
        );
        t.tenant_id = Some("acme".into());
        store.insert(t).unwrap();

        let mut t = make_task_with(
            "api",
            TaskState::Completed,
            None,
            None,
            300,
            Some(0.03),
            now,
        );
        t.tenant_id = Some("other".into());
        store.insert(t).unwrap();

        // Filter: tenant=acme, source=api
        let rows = store
            .usage_stats(&UsageQuery {
                tenant_id: Some("acme".into()),
                source: Some("api".into()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].task_count, 1);
        assert_eq!(rows[0].input_tokens, 100);
    }

    // --- idempotency key tests ---

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

        let found_a = store
            .find_by_idempotency_key("tenant-A", "idem-shared")
            .unwrap();
        let found_b = store
            .find_by_idempotency_key("tenant-B", "idem-shared")
            .unwrap();
        assert!(found_a.is_some());
        assert!(found_b.is_some());
        assert_ne!(found_a.unwrap().id, found_b.unwrap().id);
    }

    #[test]
    fn in_memory_find_by_idempotency_key_returns_none_when_missing() {
        let store = InMemoryTaskStore::new();
        let found = store
            .find_by_idempotency_key("tenant-A", "missing")
            .unwrap();
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
        assert!(
            store
                .find_by_idempotency_key("t", "idem-old")
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .find_by_idempotency_key("t", "idem-fresh")
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn is_unique_violation_recognizes_pg_23505_signature() {
        use crate::Error;
        let err = Error::Daemon("failed to insert task: error returned from database: duplicate key value violates unique constraint \"idx_daemon_tasks_idem\" (code: 23505)".into());
        assert!(super::is_unique_violation(&err));
        let err2 = Error::Daemon("connection refused".into());
        assert!(!super::is_unique_violation(&err2));
    }
}
