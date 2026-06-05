//! Per-run "restore-on-demand" store: indexes every tool output by
//! `tool_call_id` so a pruned/compacted result can be restored exactly
//! (`get`) or found semantically (`recall`). Reuses `InMemoryStore`'s
//! BM25(+vector)->RRF retrieval.

use chrono::Utc;

use crate::auth::TenantScope;
use crate::memory::in_memory::InMemoryStore;
use crate::memory::{Confidentiality, Memory, MemoryEntry, MemoryQuery, MemoryType};

/// Default cap on stored tool outputs (bounded so the store can't leak).
const DEFAULT_MAX_ENTRIES: usize = 256;

/// Max characters of head-content returned per recall hit (generous, so the
/// snippet often answers the question without a follow-up fetch).
// 280 ≈ one paragraph — generous enough to answer most queries without a full fetch.
pub const SNIPPET_CHARS: usize = 280;

/// One ranked match from `recall`: the ref to fetch, which tool produced it,
/// and a head snippet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallHit {
    /// The `tool_call_id` used to index this entry; pass to `get` for full restore.
    pub r#ref: String,
    /// Name of the tool that produced the output (first tag on the entry).
    pub tool_name: String,
    /// Leading `SNIPPET_CHARS` characters of the stored content.
    pub snippet: String,
}

/// A per-run store of tool outputs, keyed by `tool_call_id`.
pub struct ContextRecallStore {
    inner: InMemoryStore,
    scope: TenantScope,
}

impl Default for ContextRecallStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextRecallStore {
    /// Create a bounded store with the default capacity.
    pub fn new() -> Self {
        Self {
            inner: InMemoryStore::new().with_max_entries(DEFAULT_MAX_ENTRIES),
            scope: TenantScope::default(),
        }
    }

    /// Index a tool result so it can later be restored by `id` or found by query.
    /// Best-effort: a store error is swallowed (recall is a convenience, not a
    /// correctness dependency).
    pub async fn index(&self, tool_call_id: &str, tool_name: &str, content: &str) {
        let entry = make_entry(tool_call_id, tool_name, content);
        let _ = self.inner.store(&self.scope, entry).await;
    }

    /// Exact restore of a stored tool output by its `tool_call_id`.
    pub async fn get(&self, tool_call_id: &str) -> Option<String> {
        self.inner.get(tool_call_id).map(|e| e.content)
    }

    /// Semantically find stored tool outputs by `query` (BM25, or BM25+vector
    /// when an embedder is configured). Returns ranked refs + head snippets;
    /// the caller restores the full body via `get`/`fetch_full_output`.
    pub async fn recall(&self, query: &str, limit: usize) -> Vec<RecallHit> {
        let q = MemoryQuery {
            text: Some(query.to_string()),
            limit,
            reinforce: false,
            ..Default::default()
        };
        let entries = self.inner.recall(&self.scope, q).await.unwrap_or_default();
        entries
            .into_iter()
            .map(|e| RecallHit {
                r#ref: e.id,
                tool_name: e.tags.first().cloned().unwrap_or_default(),
                snippet: e.content.chars().take(SNIPPET_CHARS).collect(),
            })
            .collect()
    }
}

/// Build a `MemoryEntry` for a tool output (defaults for the unused fields).
fn make_entry(id: &str, tool_name: &str, content: &str) -> MemoryEntry {
    let now = Utc::now();
    MemoryEntry {
        id: id.to_string(),
        agent: "context_recall".into(),
        content: content.to_string(),
        category: "tool_output".into(),
        tags: vec![tool_name.to_string()],
        created_at: now,
        last_accessed: now,
        access_count: 0,
        importance: 5,
        memory_type: MemoryType::Episodic,
        keywords: Vec::new(),
        summary: None,
        strength: 1.0,
        related_ids: Vec::new(),
        source_ids: Vec::new(),
        embedding: None,
        confidentiality: Confidentiality::Public,
        author_user_id: None,
        author_tenant_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn index_then_get_roundtrips_exact_content() {
        let store = ContextRecallStore::new();
        store
            .index("tc_1", "bash", "the full untruncated output")
            .await;
        assert_eq!(
            store.get("tc_1").await.as_deref(),
            Some("the full untruncated output")
        );
        assert_eq!(store.get("nope").await, None);
    }

    #[tokio::test]
    async fn recall_ranks_a_matching_output_above_noise_and_caps_snippet() {
        let store = ContextRecallStore::new();
        store
            .index(
                "tc_match",
                "bash",
                "cargo test failed: assertion error in parser module",
            )
            .await;
        store
            .index(
                "tc_noise",
                "read",
                "the quick brown fox jumps over the lazy dog",
            )
            .await;

        let hits = store.recall("test failure parser", 5).await;
        assert!(!hits.is_empty(), "expected at least one hit");
        assert_eq!(
            hits[0].r#ref, "tc_match",
            "the matching output must rank first"
        );
        assert_eq!(hits[0].tool_name, "bash");
        assert!(hits[0].snippet.chars().count() <= SNIPPET_CHARS);
    }

    #[tokio::test]
    async fn recall_on_empty_store_is_empty() {
        let store = ContextRecallStore::new();
        assert!(store.recall("anything", 5).await.is_empty());
    }

    #[tokio::test]
    async fn recall_snippet_is_truncated_to_the_cap() {
        let store = ContextRecallStore::new();
        let long = "alpha ".repeat(200); // ~1200 chars, all matching the query term
        store.index("tc_long", "bash", &long).await;
        let hits = store.recall("alpha", 1).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].snippet.chars().count(),
            SNIPPET_CHARS,
            "a long output's snippet must be capped at exactly SNIPPET_CHARS"
        );
    }

    #[tokio::test]
    async fn pruned_then_restored_by_marker_ref_roundtrips() {
        use crate::agent::pruner::{SessionPruneConfig, prune_old_tool_results};
        use crate::llm::types::{ContentBlock, Message, Role};

        // 1. A big tool output is produced and indexed under its id.
        let store = ContextRecallStore::new();
        let big = "RESTORE_ME ".repeat(500); // well above the default 200-byte prune cap
        store.index("tc_abc", "bash", &big).await;

        // 2. The pruner truncates it in a request view; the marker names the ref.
        //    Build a message list where the ToolResult(id "tc_abc") is OLD enough
        //    to be pruned (followed by enough recent messages to push it out of the
        //    kept tail). With default keep_recent_n=2, last 4 messages are kept;
        //    index 1 (ToolResult) falls outside the recent window and gets pruned.
        let messages = vec![
            Message::user("task"),
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "tc_abc".into(),
                    content: big.clone(),
                    is_error: false,
                }],
            },
            Message::user("r1"),
            Message::assistant("r2"),
            Message::user("r3"),
            Message::assistant("r4"),
        ];
        let (pruned, stats) = prune_old_tool_results(&messages, &SessionPruneConfig::default());
        assert!(stats.did_prune());
        let marker: String = pruned
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                ContentBlock::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            })
            .collect();
        assert!(
            marker.contains("tc_abc"),
            "the pruned marker must name the ref"
        );
        assert!(
            marker.len() < big.len(),
            "the result was actually shortened"
        );

        // 3. Restore the exact original content by the ref from the store.
        assert_eq!(
            store.get("tc_abc").await.as_deref(),
            Some(big.as_str()),
            "the full content is restorable by the marker's ref"
        );
    }
}
