use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Task record stored in a task store.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Audit log entry stored in an audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: i64,
    pub task_id: Uuid,
    pub agent_name: String,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub tokens_in: Option<i32>,
    pub tokens_out: Option<i32>,
    pub created_at: DateTime<Utc>,
    /// Tenant that owns this audit entry. `None` for single-tenant deployments.
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// User who triggered the action. `None` when identity is unavailable.
    #[serde(default)]
    pub user_id: Option<String>,
}
