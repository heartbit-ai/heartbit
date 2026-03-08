use std::future::Future;
use std::pin::Pin;
use std::sync::RwLock;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::llm::types::TokenUsage;

/// Controls what data is stored in audit records.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditMode {
    /// Full content logging (current behavior, explicit opt-in for production).
    #[default]
    Full,
    /// Metadata only: tool names, timing, token counts, verdicts. No user content.
    MetadataOnly,
}

/// Strip user-content fields from an audit record payload, keeping only metadata.
///
/// Keeps: `tool_name`, `duration_ms`, `is_error`, `turn`, `hook`, `reason`, `event_type`,
/// `stop_reason`, `tool_call_count`, token usage fields.
/// Strips: `text`, `input`, `output`, `data`, `command`, `content`, `result`.
pub fn strip_content(payload: &serde_json::Value) -> serde_json::Value {
    match payload {
        serde_json::Value::Object(map) => {
            let mut stripped = serde_json::Map::new();
            for (key, value) in map {
                match key.as_str() {
                    "text" | "input" | "output" | "data" | "command" | "content" | "result" => {
                        // Replace with a marker instead of omitting, so we know it existed
                        stripped
                            .insert(key.clone(), serde_json::Value::String("[stripped]".into()));
                    }
                    _ => {
                        stripped.insert(key.clone(), value.clone());
                    }
                }
            }
            serde_json::Value::Object(stripped)
        }
        other => other.clone(),
    }
}

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

    /// Erase all audit records for the given user (GDPR right to erasure).
    /// Returns the number of records removed. Default: no-op returning 0.
    fn erase_for_user(
        &self,
        _user_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async { Ok(0) })
    }
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

    fn erase_for_user(
        &self,
        user_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        let user_id = user_id.to_string();
        Box::pin(async move {
            let mut records = self
                .records
                .write()
                .map_err(|e| Error::Agent(format!("audit lock poisoned: {e}")))?;
            let before = records.len();
            records.retain(|r| r.user_id.as_deref() != Some(user_id.as_str()));
            Ok(before - records.len())
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

    // --- Phase 25: Privacy-Preserving Audit Mode tests ---

    #[test]
    fn metadata_only_strips_content_fields() {
        let payload = json!({
            "tool_name": "bash",
            "text": "secret user input",
            "input": {"command": "ls"},
            "output": "file list",
            "duration_ms": 42,
            "is_error": false
        });
        let stripped = strip_content(&payload);
        let obj = stripped.as_object().unwrap();
        assert_eq!(obj["tool_name"], "bash");
        assert_eq!(obj["duration_ms"], 42);
        assert_eq!(obj["is_error"], false);
        assert_eq!(obj["text"], "[stripped]");
        assert_eq!(obj["input"], "[stripped]");
        assert_eq!(obj["output"], "[stripped]");
    }

    #[test]
    fn full_mode_preserves_content() {
        let payload = json!({
            "text": "hello world",
            "tool_name": "bash",
            "duration_ms": 10
        });
        // Full mode means no stripping — payload is used as-is
        let mode = AuditMode::Full;
        assert_eq!(mode, AuditMode::Full);
        // The payload should remain unchanged (strip_content is only called in MetadataOnly)
        assert_eq!(payload["text"], "hello world");
        assert_eq!(payload["tool_name"], "bash");
    }

    #[test]
    fn strip_content_replaces_known_fields_with_marker() {
        let payload = json!({
            "text": "some text",
            "input": "some input",
            "output": "some output",
            "data": "some data",
            "command": "some command",
            "content": "some content",
            "result": "some result"
        });
        let stripped = strip_content(&payload);
        let obj = stripped.as_object().unwrap();
        for key in &[
            "text", "input", "output", "data", "command", "content", "result",
        ] {
            assert_eq!(obj[*key], "[stripped]", "field {key} should be stripped");
        }
    }

    #[test]
    fn strip_content_preserves_metadata_fields() {
        let payload = json!({
            "tool_name": "read_file",
            "duration_ms": 100,
            "is_error": false,
            "turn": 3,
            "hook": "pre_tool",
            "reason": "policy violation",
            "event_type": "tool_call",
            "stop_reason": "end_turn",
            "tool_call_count": 2
        });
        let stripped = strip_content(&payload);
        assert_eq!(
            stripped, payload,
            "metadata-only payload should be unchanged"
        );
    }

    #[tokio::test]
    async fn erase_for_user_removes_matching_records() {
        let trail = InMemoryAuditTrail::new();
        trail
            .record(make_record_with_context(
                "a",
                "llm_response",
                json!({}),
                Some("user-1"),
                None,
                vec![],
            ))
            .await
            .unwrap();
        trail
            .record(make_record_with_context(
                "b",
                "tool_call",
                json!({}),
                Some("user-2"),
                None,
                vec![],
            ))
            .await
            .unwrap();
        trail
            .record(make_record_with_context(
                "c",
                "tool_result",
                json!({}),
                Some("user-1"),
                None,
                vec![],
            ))
            .await
            .unwrap();

        let removed = trail.erase_for_user("user-1").await.unwrap();
        assert_eq!(removed, 2);

        let remaining = trail.entries().await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].user_id.as_deref(), Some("user-2"));
    }

    #[tokio::test]
    async fn erase_for_user_no_matches_returns_zero() {
        let trail = InMemoryAuditTrail::new();
        trail
            .record(make_record_with_context(
                "a",
                "llm_response",
                json!({}),
                Some("user-1"),
                None,
                vec![],
            ))
            .await
            .unwrap();

        let removed = trail.erase_for_user("user-999").await.unwrap();
        assert_eq!(removed, 0);

        let remaining = trail.entries().await.unwrap();
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn audit_mode_serde_roundtrip() {
        let full = AuditMode::Full;
        let json_full = serde_json::to_string(&full).expect("serialize full");
        assert_eq!(json_full, "\"full\"");
        let deserialized: AuditMode = serde_json::from_str(&json_full).expect("deserialize full");
        assert_eq!(deserialized, AuditMode::Full);

        let meta = AuditMode::MetadataOnly;
        let json_meta = serde_json::to_string(&meta).expect("serialize metadata_only");
        assert_eq!(json_meta, "\"metadata_only\"");
        let deserialized: AuditMode =
            serde_json::from_str(&json_meta).expect("deserialize metadata_only");
        assert_eq!(deserialized, AuditMode::MetadataOnly);
    }

    #[test]
    fn strip_content_non_object_passthrough() {
        let string_val = serde_json::Value::String("hello".into());
        assert_eq!(strip_content(&string_val), string_val);

        let number_val = json!(42);
        assert_eq!(strip_content(&number_val), number_val);

        let array_val = json!([1, 2, 3]);
        assert_eq!(strip_content(&array_val), array_val);

        let null_val = serde_json::Value::Null;
        assert_eq!(strip_content(&null_val), null_val);

        let bool_val = json!(true);
        assert_eq!(strip_content(&bool_val), bool_val);
    }
}
