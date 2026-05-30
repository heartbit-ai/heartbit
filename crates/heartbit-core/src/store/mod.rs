//! Task and audit record persistence (in-memory and PostgreSQL backends).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Task record stored in a task store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    /// Task identifier.
    pub id: Uuid,
    /// Lifecycle status (`pending`, `running`, `completed`, `failed`, etc.).
    pub status: String,
    /// Original user input that triggered the task.
    pub task_input: String,
    /// Name of the agent/orchestrator config that ran the task.
    pub config_name: Option<String>,
    /// Final agent output (`None` until completion).
    pub result: Option<String>,
    /// Error message when `status` indicates failure.
    pub error: Option<String>,
    /// Serialized `TokenUsage` from the run.
    pub token_usage: Option<serde_json::Value>,
    /// Task creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Task completion timestamp (`None` while running).
    pub completed_at: Option<DateTime<Utc>>,
}

/// Audit log entry stored in an audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Monotonic entry ID (database-assigned).
    pub id: i64,
    /// Task this entry belongs to.
    pub task_id: Uuid,
    /// Agent that emitted the event.
    pub agent_name: String,
    /// Event type discriminator (e.g. `tool_call`, `llm_request`).
    pub event_type: String,
    /// Event-specific JSON payload.
    pub payload: serde_json::Value,
    /// Input tokens consumed (when applicable).
    pub tokens_in: Option<i32>,
    /// Output tokens produced (when applicable).
    pub tokens_out: Option<i32>,
    /// Entry creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Tenant that owns this audit entry. `None` for single-tenant deployments.
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// User who triggered the action. `None` when identity is unavailable.
    #[serde(default)]
    pub user_id: Option<String>,
}
