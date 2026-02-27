use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::Error;

/// Task record stored in PostgreSQL.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TaskRecord {
    pub id: Uuid,
    pub status: String,
    pub task_input: String,
    pub config_name: Option<String>,
    pub result: Option<String>,
    pub error: Option<String>,
    pub token_usage: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Audit log entry stored in PostgreSQL.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AuditEntry {
    pub id: i64,
    pub task_id: Uuid,
    pub agent_name: String,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub tokens_in: Option<i32>,
    pub tokens_out: Option<i32>,
    pub created_at: DateTime<Utc>,
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

    /// Run the initial schema migration.
    pub async fn run_migration(&self) -> Result<(), Error> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS tasks (
                id            UUID PRIMARY KEY,
                status        TEXT NOT NULL DEFAULT 'pending',
                task_input    TEXT NOT NULL,
                config_name   TEXT,
                result        TEXT,
                error         TEXT,
                token_usage   JSONB,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                completed_at  TIMESTAMPTZ
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id          BIGSERIAL PRIMARY KEY,
                task_id     UUID REFERENCES tasks(id),
                agent_name  TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                payload     JSONB NOT NULL,
                tokens_in   INT,
                tokens_out  INT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS idx_audit_task ON audit_log(task_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| Error::Store(format!("migration failed: {e}")))?;
        Ok(())
    }

    /// Create a new task record.
    pub async fn create_task(
        &self,
        id: Uuid,
        task_input: &str,
        config_name: Option<&str>,
    ) -> Result<TaskRecord, Error> {
        let record: TaskRecord = sqlx::query_as(
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

        Ok(record)
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
        let record: Option<TaskRecord> = sqlx::query_as(
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

        Ok(record)
    }

    /// Write an audit log entry.
    pub async fn write_audit(
        &self,
        task_id: Uuid,
        agent_name: &str,
        event_type: &str,
        payload: serde_json::Value,
        tokens_in: Option<i32>,
        tokens_out: Option<i32>,
    ) -> Result<(), Error> {
        sqlx::query(
            r#"
            INSERT INTO audit_log (task_id, agent_name, event_type, payload, tokens_in, tokens_out)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
        )
        .bind(task_id)
        .bind(agent_name)
        .bind(event_type)
        .bind(payload)
        .bind(tokens_in)
        .bind(tokens_out)
        .execute(&self.pool)
        .await
        .map_err(|e| Error::Store(format!("failed to write audit log: {e}")))?;

        Ok(())
    }

    /// Get audit log entries for a task.
    pub async fn get_audit_log(&self, task_id: Uuid) -> Result<Vec<AuditEntry>, Error> {
        let entries: Vec<AuditEntry> = sqlx::query_as(
            r#"
            SELECT id, task_id, agent_name, event_type, payload,
                   tokens_in, tokens_out, created_at
            FROM audit_log WHERE task_id = $1
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Error::Store(format!("failed to fetch audit log: {e}")))?;

        Ok(entries)
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
                )
                .await
        })
    }

    fn entries(
        &self,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Vec<crate::agent::audit::AuditRecord>, Error>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let rows = self.store.get_audit_log(self.task_id).await?;
            Ok(rows
                .into_iter()
                .map(|row| crate::agent::audit::AuditRecord {
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
                })
                .collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
