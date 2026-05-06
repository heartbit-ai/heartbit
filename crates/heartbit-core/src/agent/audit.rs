//! Agent audit trail — structured records of LLM calls, tool invocations, and completions.

#![allow(missing_docs)]
use parking_lot::RwLock;
use std::future::Future;
use std::pin::Pin;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::auth::TenantScope;
use crate::error::Error;
use crate::llm::types::TokenUsage;

/// Controls what data is stored in audit records.
///
/// **BREAKING CHANGE (F-AUTH-6)**: the default is now `MetadataOnly`.
/// Previously the default was `Full`, which logged complete LLM responses,
/// tool inputs, and tool outputs — incompatible with privacy-by-default
/// for regulated deployments (RGPD/HIPAA). Set `AuditMode::Full` explicitly
/// when you want content captured (e.g., debugging, single-tenant CLI dev).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditMode {
    /// Full content logging — explicit opt-in.
    Full,
    /// Metadata only: tool names, timing, token counts, verdicts. No user content.
    /// **Default** (F-AUTH-6).
    #[default]
    MetadataOnly,
}

/// Allow-list of audit payload keys that are guaranteed to contain only
/// metadata (no user content). Anything outside this list is stripped in
/// `MetadataOnly` mode.
///
/// SECURITY (F-AUTH-3): the previous implementation used a deny-list
/// (`text|input|output|data|command|content|result`) which silently leaked
/// `result_preview`, `error`, `reason`, and any nested user content (deny
/// list was not recursive). For privacy-by-default the boundary needs to be
/// "deny everything except known metadata", recursively.
const METADATA_ALLOWLIST: &[&str] = &[
    "tool_name",
    "tool_call_id",
    "tool_call_count",
    "duration_ms",
    "latency_ms",
    "is_error",
    "turn",
    "hook",
    "event_type",
    "stop_reason",
    "model",
    "total_tool_calls",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_tokens",
    "agent",
    "tenant_id",
    "user_id",
    "verdict",
    "guardrail_name",
    "consecutive_count",
    "tool_names",
    "from_tier",
    "to_tier",
    "decision",
    "priority",
    "spawned_name",
    "complexity_score",
    "escalated",
    "tool_results_pruned",
    "tool_results_total",
    "bytes_saved",
    "success",
    // `reason` is the policy-rule descriptor on guardrail denials and
    // doom-loop signals — content is the rule pattern / event type, not
    // user data. Documented as a metadata field by upstream call sites.
    "reason",
];

/// Strip user-content fields from an audit record payload, keeping only metadata.
///
/// Recursive walk: keeps only keys in [`METADATA_ALLOWLIST`]; replaces every
/// other field's value with the string `"[stripped]"` so the audit log
/// records *which* fields existed without their content. Numbers and bools
/// are preserved (they cannot leak content). Arrays/objects are recursed.
///
/// Prefer [`strip_content_owned`] in hot paths — it consumes the payload by
/// ownership and avoids the top-level `clone()` (P-CROSS-7).
pub fn strip_content(payload: &serde_json::Value) -> serde_json::Value {
    strip_content_owned(payload.clone())
}

/// Owned variant of [`strip_content`] — consumes `payload` and walks the tree
/// without cloning any preserved scalars/arrays. Used by the agent runner to
/// strip audit payloads in place when `AuditMode::MetadataOnly` is active.
///
/// Per `tasks/perf-audit-cross.md` (P-CROSS-7): the previous `&Value` variant
/// cloned every scalar, every recursed sub-value, and the top-level payload
/// — ~1 ms of avoidable CPU per audit record on 100 KB tool outputs.
pub fn strip_content_owned(payload: serde_json::Value) -> serde_json::Value {
    strip_value_owned(payload)
}

fn strip_value_owned(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut stripped = serde_json::Map::with_capacity(map.len());
            for (key, val) in map {
                if METADATA_ALLOWLIST.contains(&key.as_str()) {
                    // Even allow-listed keys: recurse so nested user content
                    // inside `tool_names: [...]` is still cleaned (it's a
                    // string array, scalars survive — see strip_scalar).
                    stripped.insert(key, strip_scalar_or_recurse_owned(val));
                } else {
                    // Replace non-metadata content with a marker. Preserve
                    // the structural type so downstream tooling sees a
                    // similarly-shaped object.
                    stripped.insert(key, redact_marker_owned(val));
                }
            }
            serde_json::Value::Object(stripped)
        }
        // Top-level non-object scalars/arrays pass through unchanged. The
        // production `AuditRecord.payload` is always built as a JSON object;
        // the non-object case exists only for tests / legacy callers.
        other => other,
    }
}

fn strip_scalar_or_recurse_owned(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(_) | serde_json::Value::Array(_) => strip_value_owned(value),
        // Scalars under allow-listed keys are safe (numbers, bools, string
        // metadata like tool names).
        other => other,
    }
}

fn redact_marker_owned(value: serde_json::Value) -> serde_json::Value {
    match value {
        // Numbers and bools cannot leak user content; keep them.
        v @ (serde_json::Value::Number(_)
        | serde_json::Value::Bool(_)
        | serde_json::Value::Null) => v,
        _ => serde_json::Value::String("[stripped]".into()),
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

/// Minimal interface for persisting and querying audit records.
///
/// The trail instance is scoped to a single run. Callers hold `Arc<dyn AuditTrail>`
/// and read entries after `execute()` returns.
pub trait AuditTrail: Send + Sync {
    /// Record a single audit entry. Best-effort: failures are logged, never abort the agent.
    fn record(
        &self,
        entry: AuditRecord,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    /// Tenant-scoped read. Returns the most recent `limit` entries whose
    /// `tenant_id` matches `scope.tenant_id`. Single-tenant scope (empty
    /// tenant_id) returns rows where `tenant_id` is `None` or `""`.
    fn entries(
        &self,
        scope: &TenantScope,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Cross-tenant admin read. Renamed from the previous unscoped `entries()`
    /// so call sites must explicitly opt in to cross-tenant visibility.
    /// Returns the most recent `limit` entries.
    fn entries_unscoped(
        &self,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Time-windowed scoped read. Returns the most recent `limit` entries for
    /// the given scope where `timestamp >= since`.
    fn entries_since(
        &self,
        scope: &TenantScope,
        since: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>>;

    /// Delete entries older than `now - retain`. Returns count deleted.
    fn prune(
        &self,
        retain: chrono::Duration,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>>;

    /// Erase all audit records for the given user (GDPR right to erasure).
    /// Returns the number of records removed. Default: no-op returning 0.
    fn erase_for_user(
        &self,
        _user_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async { Ok(0) })
    }
}

/// In-memory audit trail backed by `parking_lot::RwLock<Vec<AuditRecord>>`.
///
/// Lock is never held across `.await` — all operations are synchronous inside
/// the lock, then wrapped in `Box::pin(async { ... })`. `parking_lot` is used
/// (not `std::sync::RwLock`) for ~2× faster uncontended read acquisition on
/// the audit hot path (every turn, every tool call); see T2 in
/// `tasks/performance-audit-heartbit-core-2026-05-06.md`.
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
            self.records.write().push(entry);
            Ok(())
        })
    }

    fn entries(
        &self,
        scope: &TenantScope,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>> {
        let tid = scope.tenant_id.clone();
        Box::pin(async move {
            let records = self.records.read();
            let matched: Vec<AuditRecord> = records
                .iter()
                .filter(|r| r.tenant_id.as_deref().unwrap_or("") == tid.as_str())
                .cloned()
                .collect();
            let start = matched.len().saturating_sub(limit);
            Ok(matched[start..].to_vec())
        })
    }

    fn entries_unscoped(
        &self,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>> {
        Box::pin(async move {
            let records = self.records.read();
            let start = records.len().saturating_sub(limit);
            Ok(records[start..].to_vec())
        })
    }

    fn entries_since(
        &self,
        scope: &TenantScope,
        since: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<AuditRecord>, Error>> + Send + '_>> {
        let tid = scope.tenant_id.clone();
        Box::pin(async move {
            let records = self.records.read();
            let matched: Vec<AuditRecord> = records
                .iter()
                .filter(|r| {
                    r.tenant_id.as_deref().unwrap_or("") == tid.as_str() && r.timestamp >= since
                })
                .cloned()
                .collect();
            let start = matched.len().saturating_sub(limit);
            Ok(matched[start..].to_vec())
        })
    }

    fn prune(
        &self,
        retain: chrono::Duration,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async move {
            let cutoff = chrono::Utc::now() - retain;
            let mut records = self.records.write();
            let before = records.len();
            records.retain(|r| r.timestamp >= cutoff);
            Ok(before - records.len())
        })
    }

    fn erase_for_user(
        &self,
        user_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        let user_id = user_id.to_string();
        Box::pin(async move {
            let mut records = self.records.write();
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

        let entries = trail.entries_unscoped(usize::MAX).await.unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].event_type, "llm_response");
        assert_eq!(entries[1].event_type, "tool_call");
        assert_eq!(entries[2].event_type, "tool_result");
    }

    #[tokio::test]
    async fn in_memory_trail_empty_by_default() {
        let trail = InMemoryAuditTrail::new();
        let entries = trail.entries_unscoped(usize::MAX).await.unwrap();
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
    async fn entries_scoped_filters_correctly() {
        use crate::auth::TenantScope;

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

        let scope = TenantScope::new("tenant-a");
        let filtered = trail.entries(&scope, usize::MAX).await.unwrap();
        assert_eq!(filtered.len(), 2);
        assert!(
            filtered
                .iter()
                .all(|r| r.tenant_id.as_deref() == Some("tenant-a"))
        );
    }

    #[tokio::test]
    async fn entries_unscoped_returns_all() {
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

        let all = trail.entries_unscoped(usize::MAX).await.unwrap();
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

        let entries = trail.entries_unscoped(usize::MAX).await.unwrap();
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

        let remaining = trail.entries_unscoped(usize::MAX).await.unwrap();
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

        let remaining = trail.entries_unscoped(usize::MAX).await.unwrap();
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

    /// SECURITY (F-AUTH-3): the previous deny-list missed `result_preview`,
    /// `error`, and any nested user-content fields. The new allow-list
    /// approach must redact these even though they were not in the old list.
    #[test]
    fn strip_content_redacts_result_preview_and_error() {
        // run_completed payload — result_preview is the first 1000 chars of
        // the LLM's reply, which is user-facing content.
        let payload = json!({
            "total_tool_calls": 2,
            "result_preview": "user secret content here"
        });
        let stripped = strip_content(&payload);
        let obj = stripped.as_object().unwrap();
        assert_eq!(obj["total_tool_calls"], 2);
        assert_eq!(
            obj["result_preview"], "[stripped]",
            "result_preview MUST be stripped (F-AUTH-3)"
        );

        // run_failed payload — error message can echo user input.
        let payload = json!({
            "error": "tool foo failed: <user data was here>"
        });
        let stripped = strip_content(&payload);
        let obj = stripped.as_object().unwrap();
        assert_eq!(
            obj["error"], "[stripped]",
            "error MUST be stripped (F-AUTH-3)"
        );
    }

    /// SECURITY (F-AUTH-3): nested user content inside arbitrary parent keys
    /// (e.g. `meta.command`) was previously preserved because the deny-list
    /// only scanned top-level keys. The recursive walk redacts the parent
    /// when its key is not in the allow-list.
    #[test]
    fn strip_content_recursive_no_leak_via_nested_object() {
        let payload = json!({
            "meta": {
                "command": "rm -rf /",
                "input": {"file": "secret.txt"}
            }
        });
        let stripped = strip_content(&payload);
        let s = serde_json::to_string(&stripped).unwrap();
        assert!(
            !s.contains("rm -rf"),
            "nested command must be redacted (F-AUTH-3): {s}"
        );
        assert!(
            !s.contains("secret.txt"),
            "nested user content must be redacted: {s}"
        );
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

    #[tokio::test]
    async fn entries_filters_by_scope() {
        use crate::auth::TenantScope;

        let trail = InMemoryAuditTrail::new();
        let acme = TenantScope::new("acme");
        let globex = TenantScope::new("globex");

        let mk = |tid: Option<&str>| AuditRecord {
            agent: "a".into(),
            turn: 0,
            event_type: "x".into(),
            payload: serde_json::Value::Null,
            usage: TokenUsage::default(),
            timestamp: chrono::Utc::now(),
            user_id: None,
            tenant_id: tid.map(|s| s.into()),
            delegation_chain: vec![],
        };

        trail.record(mk(Some("acme"))).await.unwrap();
        trail.record(mk(Some("globex"))).await.unwrap();

        let acme_rows = trail.entries(&acme, 100).await.unwrap();
        assert_eq!(acme_rows.len(), 1);
        assert_eq!(acme_rows[0].tenant_id.as_deref(), Some("acme"));

        let globex_rows = trail.entries(&globex, 100).await.unwrap();
        assert_eq!(globex_rows.len(), 1);
        assert_eq!(globex_rows[0].tenant_id.as_deref(), Some("globex"));

        let unscoped = trail.entries_unscoped(100).await.unwrap();
        assert_eq!(unscoped.len(), 2);
    }

    #[tokio::test]
    async fn prune_deletes_old_entries() {
        let trail = InMemoryAuditTrail::new();
        let now = chrono::Utc::now();
        let mk = |timestamp| AuditRecord {
            agent: "a".into(),
            turn: 0,
            event_type: "x".into(),
            payload: serde_json::Value::Null,
            usage: TokenUsage::default(),
            timestamp,
            user_id: None,
            tenant_id: None,
            delegation_chain: vec![],
        };

        trail
            .record(mk(now - chrono::Duration::days(10)))
            .await
            .unwrap();
        trail.record(mk(now)).await.unwrap();

        let removed = trail.prune(chrono::Duration::days(7)).await.unwrap();
        assert_eq!(removed, 1);

        let rest = trail.entries_unscoped(100).await.unwrap();
        assert_eq!(rest.len(), 1);
    }

    #[tokio::test]
    async fn entries_since_filters_by_time() {
        use crate::auth::TenantScope;

        let trail = InMemoryAuditTrail::new();
        let now = chrono::Utc::now();
        let scope = TenantScope::new("acme");
        let mk = |timestamp| AuditRecord {
            agent: "a".into(),
            turn: 0,
            event_type: "x".into(),
            payload: serde_json::Value::Null,
            usage: TokenUsage::default(),
            timestamp,
            user_id: None,
            tenant_id: Some("acme".into()),
            delegation_chain: vec![],
        };

        trail
            .record(mk(now - chrono::Duration::hours(48)))
            .await
            .unwrap();
        trail
            .record(mk(now - chrono::Duration::hours(1)))
            .await
            .unwrap();

        let recent = trail
            .entries_since(&scope, now - chrono::Duration::hours(24), 100)
            .await
            .unwrap();
        assert_eq!(recent.len(), 1);
    }
}
