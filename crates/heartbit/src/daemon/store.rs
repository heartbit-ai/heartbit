use std::collections::HashMap;
use std::sync::RwLock;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::types::{DaemonTask, TaskState};
use crate::Error;
use crate::llm::types::TokenUsage;

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
        let mut tasks = self
            .tasks
            .write()
            .map_err(|e| Error::Daemon(e.to_string()))?;
        if tasks.contains_key(&id) {
            return Err(Error::Daemon(format!("task {id} already exists")));
        }
        tasks.insert(id, task);
        drop(tasks);
        let mut order = self
            .order
            .write()
            .map_err(|e| Error::Daemon(e.to_string()))?;
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
}

// --- PostgreSQL task store ---

/// Row type for reading daemon tasks from PostgreSQL.
#[derive(Debug, sqlx::FromRow)]
struct TaskRow {
    id: Uuid,
    task: String,
    state: String,
    created_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
    result: Option<String>,
    error: Option<String>,
    input_tokens: i32,
    output_tokens: i32,
    cache_creation_input_tokens: i32,
    cache_read_input_tokens: i32,
    reasoning_tokens: i32,
    tool_calls_made: i32,
    estimated_cost_usd: Option<f64>,
    source: String,
}

fn task_state_to_str(state: TaskState) -> &'static str {
    match state {
        TaskState::Pending => "pending",
        TaskState::Running => "running",
        TaskState::Completed => "completed",
        TaskState::Failed => "failed",
        TaskState::Cancelled => "cancelled",
    }
}

fn str_to_task_state(s: &str) -> TaskState {
    match s {
        "running" => TaskState::Running,
        "completed" => TaskState::Completed,
        "failed" => TaskState::Failed,
        "cancelled" => TaskState::Cancelled,
        _ => TaskState::Pending,
    }
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
                source                      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_daemon_tasks_created_at
                ON daemon_tasks(created_at);
            CREATE INDEX IF NOT EXISTS idx_daemon_tasks_state
                ON daemon_tasks(state);
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| Error::Daemon(format!("task migration failed: {e}")))?;
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
                         estimated_cost_usd, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)"#,
                )
                .bind(task.id)
                .bind(&task.task)
                .bind(task_state_to_str(task.state))
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
                     estimated_cost_usd, source \
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
                     estimated_cost_usd, source \
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
                     estimated_cost_usd, source \
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
                        estimated_cost_usd = $14, source = $15
                    WHERE id = $1"#,
                )
                .bind(task.id)
                .bind(&task.task)
                .bind(task_state_to_str(task.state))
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
                .execute(&pool)
                .await
                .map_err(|e| Error::Daemon(format!("failed to update task: {e}")))?;
                Ok(())
            })
        })
    }
}

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

    #[test]
    fn task_state_str_roundtrip() {
        for state in [
            TaskState::Pending,
            TaskState::Running,
            TaskState::Completed,
            TaskState::Failed,
            TaskState::Cancelled,
        ] {
            let s = task_state_to_str(state);
            let back = str_to_task_state(s);
            assert_eq!(back, state, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn str_to_task_state_unknown_defaults_to_pending() {
        assert_eq!(str_to_task_state("bogus"), TaskState::Pending);
        assert_eq!(str_to_task_state(""), TaskState::Pending);
    }

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
        };
        let task = DaemonTask::from(row);
        assert_eq!(task.state, TaskState::Failed);
        assert!(task.result.is_none());
        assert_eq!(task.error.as_deref(), Some("out of tokens"));
    }

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
        };
        let task = DaemonTask::from(row);
        assert_eq!(task.created_at, created);
        assert_eq!(task.started_at, Some(started));
        assert_eq!(task.completed_at, Some(completed));
    }
}
