use std::future::Future;
use std::pin::Pin;
use std::sync::RwLock;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::llm::types::TokenUsage;

/// One entry per decision point in an agent run.
///
/// Records LLM responses, tool calls, tool results, run completion/failure,
/// and guardrail denials with full (untruncated) payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub agent: String,
    pub turn: usize,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub usage: TokenUsage,
    pub timestamp: DateTime<Utc>,
    /// User ID of the authenticated user who triggered this action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Tenant ID for multi-tenant isolation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    /// RFC 8693 delegation chain: \[actor1, actor2, ...\] from outermost to innermost.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delegation_chain: Vec<String>,
}

/// Minimal 2-method interface for persisting audit records.
///
/// The trail instance is scoped to a single run. Callers hold `Arc<dyn AuditTrail>`
/// and read entries after `execute()` returns.
pub trait AuditTrail: Send + Sync {
    /// Record a single audit entry. Best-effort: failures are logged, never abort the agent.
    fn record(
        &self,
        entry: AuditRecord,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    /// Retrieve all recorded entries in insertion order.
    fn entries(&self)
    -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Retrieve entries filtered by tenant ID. Returns all entries if `tenant_id` is `None`.
    fn entries_for_tenant(
        &self,
        tenant_id: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;
}

/// In-memory audit trail backed by `std::sync::RwLock<Vec<AuditRecord>>`.
///
/// Lock is never held across `.await` — all operations are synchronous inside
/// the lock, then wrapped in `Box::pin(async { ... })`.
pub struct InMemoryAuditTrail {
    records: RwLock<Vec<AuditRecord>>,
}

impl InMemoryAuditTrail {
    pub fn new() -> Self {
        Self {
            records: RwLock::new(Vec::new()),
        }
    }
}

impl Default for InMemoryAuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditTrail for InMemoryAuditTrail {
    fn record(
        &self,
        entry: AuditRecord,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            let mut records = self
                .records
                .write()
                .map_err(|e| Error::Agent(format!("audit lock poisoned: {e}")))?;
            records.push(entry);
            Ok(())
        })
    }

    fn entries(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>> {
        Box::pin(async move {
            let records = self
                .records
                .read()
                .map_err(|e| Error::Agent(format!("audit lock poisoned: {e}")))?;
            Ok(records.clone())
        })
    }

    fn entries_for_tenant(
        &self,
        tenant_id: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>> {
        let tenant_id = tenant_id.map(String::from);
        Box::pin(async move {
            let records = self
                .records
                .read()
                .map_err(|e| Error::Agent(format!("audit lock poisoned: {e}")))?;
            match tenant_id {
                None => Ok(records.clone()),
                Some(tid) => Ok(records
                    .iter()
                    .filter(|r| r.tenant_id.as_deref() == Some(tid.as_str()))
                    .cloned()
                    .collect()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_record(agent: &str, event_type: &str, payload: serde_json::Value) -> AuditRecord {
        AuditRecord {
            agent: agent.into(),
            turn: 1,
            event_type: event_type.into(),
            payload,
            usage: TokenUsage::default(),
            timestamp: Utc::now(),
            user_id: None,
            tenant_id: None,
            delegation_chain: Vec::new(),
        }
    }

    #[test]
    fn audit_record_serializes() {
        let record = make_record("test-agent", "llm_response", json!({"text": "hello"}));
        let json = serde_json::to_string(&record).expect("serialize");
        let deserialized: AuditRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.agent, "test-agent");
        assert_eq!(deserialized.event_type, "llm_response");
        assert_eq!(deserialized.payload, json!({"text": "hello"}));
    }

    #[tokio::test]
    async fn in_memory_trail_stores_and_retrieves() {
        let trail = InMemoryAuditTrail::new();
        trail
            .record(make_record("a", "llm_response", json!({"turn": 1})))
            .await
            .unwrap();
        trail
            .record(make_record("a", "tool_call", json!({"name": "bash"})))
            .await
            .unwrap();
        trail
            .record(make_record("a", "tool_result", json!({"ok": true})))
            .await
            .unwrap();

        let entries = trail.entries().await.unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].event_type, "llm_response");
        assert_eq!(entries[1].event_type, "tool_call");
        assert_eq!(entries[2].event_type, "tool_result");
    }

    #[tokio::test]
    async fn in_memory_trail_empty_by_default() {
        let trail = InMemoryAuditTrail::new();
        let entries = trail.entries().await.unwrap();
        assert!(entries.is_empty());
    }

    fn make_record_with_context(
        agent: &str,
        event_type: &str,
        payload: serde_json::Value,
        user_id: Option<&str>,
        tenant_id: Option<&str>,
        delegation_chain: Vec<String>,
    ) -> AuditRecord {
        AuditRecord {
            agent: agent.into(),
            turn: 1,
            event_type: event_type.into(),
            payload,
            usage: TokenUsage::default(),
            timestamp: Utc::now(),
            user_id: user_id.map(String::from),
            tenant_id: tenant_id.map(String::from),
            delegation_chain,
        }
    }

    #[test]
    fn audit_record_with_user_context() {
        let record = make_record_with_context(
            "agent-1",
            "llm_response",
            json!({"text": "hi"}),
            Some("user-42"),
            Some("tenant-a"),
            vec!["actor-1".into(), "actor-2".into()],
        );
        let json_str = serde_json::to_string(&record).expect("serialize");
        let deserialized: AuditRecord = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(deserialized.user_id.as_deref(), Some("user-42"));
        assert_eq!(deserialized.tenant_id.as_deref(), Some("tenant-a"));
        assert_eq!(deserialized.delegation_chain, vec!["actor-1", "actor-2"]);
    }

    #[test]
    fn audit_record_backward_compat() {
        // Old JSON without the new fields must deserialize with defaults.
        let old_json = json!({
            "agent": "old-agent",
            "turn": 3,
            "event_type": "tool_call",
            "payload": {"name": "bash"},
            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            "timestamp": "2026-01-01T00:00:00Z"
        });
        let record: AuditRecord = serde_json::from_value(old_json).expect("deserialize old format");
        assert_eq!(record.user_id, None);
        assert_eq!(record.tenant_id, None);
        assert!(record.delegation_chain.is_empty());
    }

    #[test]
    fn audit_record_delegation_chain_omitted_when_empty() {
        let record =
            make_record_with_context("agent-1", "llm_response", json!({}), None, None, vec![]);
        let json_val = serde_json::to_value(&record).expect("serialize");
        assert!(
            !json_val
                .as_object()
                .unwrap()
                .contains_key("delegation_chain")
        );
    }

    #[test]
    fn audit_record_user_id_omitted_when_none() {
        let record = make_record_with_context(
            "agent-1",
            "llm_response",
            json!({}),
            None,
            Some("tenant-a"),
            vec![],
        );
        let json_val = serde_json::to_value(&record).expect("serialize");
        let obj = json_val.as_object().unwrap();
        assert!(!obj.contains_key("user_id"));
        assert!(obj.contains_key("tenant_id"));
    }

    #[tokio::test]
    async fn entries_for_tenant_filters_correctly() {
        let trail = InMemoryAuditTrail::new();
        trail
            .record(make_record_with_context(
                "a",
                "llm_response",
                json!({}),
                None,
                Some("tenant-a"),
                vec![],
            ))
            .await
            .unwrap();
        trail
            .record(make_record_with_context(
                "b",
                "tool_call",
                json!({}),
                None,
                Some("tenant-b"),
                vec![],
            ))
            .await
            .unwrap();
        trail
            .record(make_record_with_context(
                "c",
                "tool_result",
                json!({}),
                None,
                Some("tenant-a"),
                vec![],
            ))
            .await
            .unwrap();

        let filtered = trail.entries_for_tenant(Some("tenant-a")).await.unwrap();
        assert_eq!(filtered.len(), 2);
        assert!(
            filtered
                .iter()
                .all(|r| r.tenant_id.as_deref() == Some("tenant-a"))
        );
    }

    #[tokio::test]
    async fn entries_for_tenant_none_returns_all() {
        let trail = InMemoryAuditTrail::new();
        trail
            .record(make_record_with_context(
                "a",
                "llm_response",
                json!({}),
                None,
                Some("tenant-a"),
                vec![],
            ))
            .await
            .unwrap();
        trail
            .record(make_record_with_context(
                "b",
                "tool_call",
                json!({}),
                None,
                Some("tenant-b"),
                vec![],
            ))
            .await
            .unwrap();
        trail
            .record(make_record_with_context(
                "c",
                "tool_result",
                json!({}),
                None,
                None,
                vec![],
            ))
            .await
            .unwrap();

        let all = trail.entries_for_tenant(None).await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn audit_record_with_large_payload() {
        let trail = InMemoryAuditTrail::new();
        // 1MB payload — must survive untruncated
        let large = "x".repeat(1_000_000);
        let payload = json!({"data": large});
        trail
            .record(make_record("a", "tool_result", payload.clone()))
            .await
            .unwrap();

        let entries = trail.entries().await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].payload, payload);
    }
}
