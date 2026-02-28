use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::TrustLevel;
use crate::llm::types::TokenUsage;

/// Commands serialized to the `heartbit.commands` Kafka topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonCommand {
    SubmitTask {
        id: Uuid,
        task: String,
        source: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        story_id: Option<String>,
        /// Sender trust level for security guardrails (sensor tasks only).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trust_level: Option<TrustLevel>,
        /// Authenticated user ID for multi-tenant isolation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_id: Option<String>,
        /// Tenant ID for multi-tenant isolation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tenant_id: Option<String>,
    },
    CancelTask {
        id: Uuid,
    },
}

/// State machine for daemon task lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// A daemon task with its full lifecycle state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonTask {
    pub id: Uuid,
    pub task: String,
    pub state: TaskState,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub tokens_used: TokenUsage,
    #[serde(default)]
    pub tool_calls_made: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_cost_usd: Option<f64>,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    /// Authenticated user ID for multi-tenant isolation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Tenant ID for multi-tenant isolation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
}

impl DaemonTask {
    /// Create a new pending task.
    pub fn new(id: Uuid, task: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            id,
            task: task.into(),
            state: TaskState::Pending,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
            tokens_used: TokenUsage::default(),
            tool_calls_made: 0,
            estimated_cost_usd: None,
            source: source.into(),
            agent_name: None,
            user_id: None,
            tenant_id: None,
        }
    }

    /// Create a new pending task with user context for multi-tenant isolation.
    pub fn new_with_user(
        id: Uuid,
        task: impl Into<String>,
        source: impl Into<String>,
        user_id: impl Into<String>,
        tenant_id: impl Into<String>,
    ) -> Self {
        Self {
            user_id: Some(user_id.into()),
            tenant_id: Some(tenant_id.into()),
            ..Self::new(id, task, source)
        }
    }
}

/// Authenticated user context extracted from JWT or API token.
///
/// Propagated through the daemon to scope memory, workspace, audit, and MCP auth
/// to the authenticated user. All fields are strings to avoid coupling to a
/// specific identity provider's ID format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserContext {
    /// User identifier (typically JWT `sub` claim).
    pub user_id: String,
    /// Tenant identifier for multi-tenant isolation (typically JWT `tid` claim).
    pub tenant_id: String,
    /// Roles granted to this user (typically JWT `roles` claim).
    #[serde(default)]
    pub roles: Vec<String>,
}

/// Aggregated statistics across all daemon tasks.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskStats {
    pub total_tasks: usize,
    pub tasks_by_state: HashMap<String, usize>,
    pub tasks_by_source: HashMap<String, usize>,
    pub active_tasks: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cache_read_tokens: u64,
    pub total_cache_creation_tokens: u64,
    pub total_estimated_cost_usd: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_command_submit_roundtrip() {
        let id = Uuid::new_v4();
        let cmd = DaemonCommand::SubmitTask {
            id,
            task: "Do something".into(),
            source: "api".into(),
            story_id: None,
            trust_level: None,
            user_id: None,
            tenant_id: None,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: DaemonCommand = serde_json::from_str(&json).unwrap();
        match parsed {
            DaemonCommand::SubmitTask {
                id: parsed_id,
                task,
                source,
                story_id,
                ..
            } => {
                assert_eq!(parsed_id, id);
                assert_eq!(task, "Do something");
                assert_eq!(source, "api");
                assert!(story_id.is_none());
            }
            _ => panic!("expected SubmitTask"),
        }
    }

    #[test]
    fn daemon_command_cancel_roundtrip() {
        let id = Uuid::new_v4();
        let cmd = DaemonCommand::CancelTask { id };
        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: DaemonCommand = serde_json::from_str(&json).unwrap();
        match parsed {
            DaemonCommand::CancelTask { id: parsed_id } => {
                assert_eq!(parsed_id, id);
            }
            _ => panic!("expected CancelTask"),
        }
    }

    #[test]
    fn daemon_command_tagged_serialization() {
        let cmd = DaemonCommand::SubmitTask {
            id: Uuid::nil(),
            task: "test".into(),
            source: "api".into(),
            story_id: None,
            trust_level: None,
            user_id: None,
            tenant_id: None,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains(r#""type":"submit_task""#));
        // story_id=None should be omitted
        assert!(!json.contains("story_id"));

        let cmd = DaemonCommand::CancelTask { id: Uuid::nil() };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains(r#""type":"cancel_task""#));
    }

    #[test]
    fn submit_task_with_story_id_roundtrip() {
        let id = Uuid::new_v4();
        let cmd = DaemonCommand::SubmitTask {
            id,
            task: "Analyze CVE".into(),
            source: "sensor:rss".into(),
            story_id: Some("story-cve-2026-001".into()),
            trust_level: None,
            user_id: None,
            tenant_id: None,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("story_id"));
        let parsed: DaemonCommand = serde_json::from_str(&json).unwrap();
        match parsed {
            DaemonCommand::SubmitTask { story_id, .. } => {
                assert_eq!(story_id.as_deref(), Some("story-cve-2026-001"));
            }
            _ => panic!("expected SubmitTask"),
        }
    }

    #[test]
    fn submit_task_without_story_id_backward_compat() {
        // Old JSON without story_id field should deserialize with None
        let json = r#"{"type":"submit_task","id":"00000000-0000-0000-0000-000000000000","task":"test","source":"api"}"#;
        let parsed: DaemonCommand = serde_json::from_str(json).unwrap();
        match parsed {
            DaemonCommand::SubmitTask { story_id, .. } => {
                assert!(story_id.is_none());
            }
            _ => panic!("expected SubmitTask"),
        }
    }

    #[test]
    fn task_state_roundtrip() {
        for state in [
            TaskState::Pending,
            TaskState::Running,
            TaskState::Completed,
            TaskState::Failed,
            TaskState::Cancelled,
        ] {
            let json = serde_json::to_string(&state).unwrap();
            let parsed: TaskState = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, state);
        }
    }

    #[test]
    fn task_state_snake_case() {
        assert_eq!(
            serde_json::to_string(&TaskState::Pending).unwrap(),
            r#""pending""#
        );
        assert_eq!(
            serde_json::to_string(&TaskState::Running).unwrap(),
            r#""running""#
        );
        assert_eq!(
            serde_json::to_string(&TaskState::Completed).unwrap(),
            r#""completed""#
        );
        assert_eq!(
            serde_json::to_string(&TaskState::Failed).unwrap(),
            r#""failed""#
        );
        assert_eq!(
            serde_json::to_string(&TaskState::Cancelled).unwrap(),
            r#""cancelled""#
        );
    }

    #[test]
    fn daemon_task_new_defaults() {
        let id = Uuid::new_v4();
        let task = DaemonTask::new(id, "do something", "api");
        assert_eq!(task.id, id);
        assert_eq!(task.task, "do something");
        assert_eq!(task.state, TaskState::Pending);
        assert!(task.started_at.is_none());
        assert!(task.completed_at.is_none());
        assert!(task.result.is_none());
        assert!(task.error.is_none());
        assert_eq!(task.tokens_used, TokenUsage::default());
        assert_eq!(task.tool_calls_made, 0);
        assert!(task.estimated_cost_usd.is_none());
        assert_eq!(task.source, "api");
        assert!(task.agent_name.is_none());
        assert!(task.user_id.is_none());
        assert!(task.tenant_id.is_none());
    }

    #[test]
    fn daemon_task_roundtrip() {
        let id = Uuid::new_v4();
        let task = DaemonTask::new(id, "hello world", "cron:daily");
        let json = serde_json::to_string(&task).unwrap();
        let parsed: DaemonTask = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, id);
        assert_eq!(parsed.task, "hello world");
        assert_eq!(parsed.state, TaskState::Pending);
        assert_eq!(parsed.source, "cron:daily");
    }

    #[test]
    fn daemon_task_optional_fields_omitted() {
        let task = DaemonTask::new(Uuid::nil(), "test", "api");
        let json = serde_json::to_string(&task).unwrap();
        // Optional None fields should be omitted
        assert!(!json.contains("started_at"));
        assert!(!json.contains("completed_at"));
        assert!(!json.contains("result"));
        assert!(!json.contains("error"));
        assert!(!json.contains("estimated_cost_usd"));
        assert!(!json.contains("agent_name"));
    }

    #[test]
    fn agent_name_roundtrip_present() {
        let id = Uuid::new_v4();
        let mut task = DaemonTask::new(id, "analyze code", "api");
        task.agent_name = Some("security-agent".into());
        let json = serde_json::to_string(&task).unwrap();
        assert!(json.contains("agent_name"));
        assert!(json.contains("security-agent"));
        let parsed: DaemonTask = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.agent_name.as_deref(), Some("security-agent"));
    }

    #[test]
    fn agent_name_roundtrip_absent_backward_compat() {
        // Old JSON without agent_name field should deserialize with None
        let json = r#"{"id":"00000000-0000-0000-0000-000000000000","task":"test","state":"pending","created_at":"2026-01-01T00:00:00Z","tokens_used":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"reasoning_tokens":0},"tool_calls_made":0,"source":"api"}"#;
        let parsed: DaemonTask = serde_json::from_str(json).unwrap();
        assert!(parsed.agent_name.is_none());
        assert_eq!(parsed.task, "test");
        assert_eq!(parsed.source, "api");
    }

    #[test]
    fn task_stats_default_is_zero() {
        let stats = TaskStats::default();
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
    fn task_stats_serde_roundtrip() {
        let mut stats = TaskStats::default();
        stats.total_tasks = 10;
        stats.tasks_by_state.insert("running".into(), 3);
        stats.tasks_by_source.insert("api".into(), 7);
        stats.active_tasks = 3;
        stats.total_input_tokens = 5000;
        stats.total_output_tokens = 2000;
        stats.total_estimated_cost_usd = 1.23;

        let json = serde_json::to_string(&stats).unwrap();
        let parsed: TaskStats = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_tasks, 10);
        assert_eq!(parsed.tasks_by_state.get("running"), Some(&3));
        assert_eq!(parsed.tasks_by_source.get("api"), Some(&7));
        assert_eq!(parsed.active_tasks, 3);
        assert_eq!(parsed.total_input_tokens, 5000);
        assert_eq!(parsed.total_output_tokens, 2000);
        assert_eq!(parsed.total_estimated_cost_usd, 1.23);
    }

    #[test]
    fn task_stats_deserialize_partial_json() {
        // Older versions may not have all fields — #[serde(default)] ensures they default.
        let json = r#"{"total_tasks": 5}"#;
        let stats: TaskStats = serde_json::from_str(json).unwrap();
        assert_eq!(stats.total_tasks, 5);
        assert_eq!(stats.active_tasks, 0);
        assert!(stats.tasks_by_state.is_empty());
        assert_eq!(stats.total_input_tokens, 0);
        assert_eq!(stats.total_estimated_cost_usd, 0.0);
    }

    // --- Multi-tenant / UserContext tests ---

    #[test]
    fn user_context_serde_roundtrip() {
        let ctx = UserContext {
            user_id: "user-123".into(),
            tenant_id: "acme".into(),
            roles: vec!["sales".into(), "admin".into()],
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let parsed: UserContext = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.user_id, "user-123");
        assert_eq!(parsed.tenant_id, "acme");
        assert_eq!(parsed.roles, vec!["sales", "admin"]);
    }

    #[test]
    fn user_context_empty_roles_default() {
        let json = r#"{"user_id":"u1","tenant_id":"t1"}"#;
        let ctx: UserContext = serde_json::from_str(json).unwrap();
        assert_eq!(ctx.user_id, "u1");
        assert_eq!(ctx.tenant_id, "t1");
        assert!(ctx.roles.is_empty());
    }

    #[test]
    fn user_context_equality() {
        let a = UserContext {
            user_id: "u1".into(),
            tenant_id: "t1".into(),
            roles: vec!["r1".into()],
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn daemon_task_new_with_user() {
        let id = Uuid::new_v4();
        let task = DaemonTask::new_with_user(id, "crm query", "api", "user-42", "acme-corp");
        assert_eq!(task.id, id);
        assert_eq!(task.task, "crm query");
        assert_eq!(task.source, "api");
        assert_eq!(task.user_id.as_deref(), Some("user-42"));
        assert_eq!(task.tenant_id.as_deref(), Some("acme-corp"));
        assert_eq!(task.state, TaskState::Pending);
    }

    #[test]
    fn daemon_task_user_fields_omitted_when_none() {
        let task = DaemonTask::new(Uuid::nil(), "test", "api");
        let json = serde_json::to_string(&task).unwrap();
        assert!(!json.contains("user_id"));
        assert!(!json.contains("tenant_id"));
    }

    #[test]
    fn daemon_task_user_fields_present_when_set() {
        let task = DaemonTask::new_with_user(Uuid::nil(), "test", "api", "alice", "acme");
        let json = serde_json::to_string(&task).unwrap();
        assert!(json.contains(r#""user_id":"alice""#));
        assert!(json.contains(r#""tenant_id":"acme""#));
    }

    #[test]
    fn daemon_task_backward_compat_no_user_fields() {
        // Old JSON without user_id/tenant_id should deserialize with None
        let json = r#"{"id":"00000000-0000-0000-0000-000000000000","task":"test","state":"pending","created_at":"2026-01-01T00:00:00Z","tokens_used":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"reasoning_tokens":0},"tool_calls_made":0,"source":"api"}"#;
        let parsed: DaemonTask = serde_json::from_str(json).unwrap();
        assert!(parsed.user_id.is_none());
        assert!(parsed.tenant_id.is_none());
    }

    #[test]
    fn submit_command_with_user_context() {
        let id = Uuid::new_v4();
        let cmd = DaemonCommand::SubmitTask {
            id,
            task: "list deals".into(),
            source: "api".into(),
            story_id: None,
            trust_level: None,
            user_id: Some("alice".into()),
            tenant_id: Some("acme".into()),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains(r#""user_id":"alice""#));
        assert!(json.contains(r#""tenant_id":"acme""#));
        let parsed: DaemonCommand = serde_json::from_str(&json).unwrap();
        match parsed {
            DaemonCommand::SubmitTask {
                user_id, tenant_id, ..
            } => {
                assert_eq!(user_id.as_deref(), Some("alice"));
                assert_eq!(tenant_id.as_deref(), Some("acme"));
            }
            _ => panic!("expected SubmitTask"),
        }
    }

    #[test]
    fn submit_command_user_fields_omitted_when_none() {
        let cmd = DaemonCommand::SubmitTask {
            id: Uuid::nil(),
            task: "test".into(),
            source: "api".into(),
            story_id: None,
            trust_level: None,
            user_id: None,
            tenant_id: None,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(!json.contains("user_id"));
        assert!(!json.contains("tenant_id"));
    }

    #[test]
    fn submit_command_backward_compat_no_user_fields() {
        // Old JSON without user_id/tenant_id
        let json = r#"{"type":"submit_task","id":"00000000-0000-0000-0000-000000000000","task":"test","source":"api"}"#;
        let parsed: DaemonCommand = serde_json::from_str(json).unwrap();
        match parsed {
            DaemonCommand::SubmitTask {
                user_id, tenant_id, ..
            } => {
                assert!(user_id.is_none());
                assert!(tenant_id.is_none());
            }
            _ => panic!("expected SubmitTask"),
        }
    }
}
