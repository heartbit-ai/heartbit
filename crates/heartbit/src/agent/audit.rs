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
