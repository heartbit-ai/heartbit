//! Per-run "restore-on-demand" store: indexes every tool output by
//! `tool_call_id` so a pruned/compacted result can be restored exactly
//! (`get`) or found semantically (`recall`). Reuses `InMemoryStore`'s
//! BM25(+vector)->RRF retrieval.

use chrono::Utc;

use crate::auth::TenantScope;
use crate::memory::in_memory::InMemoryStore;
use crate::memory::{Confidentiality, Memory, MemoryEntry, MemoryType};

/// Default cap on stored tool outputs (bounded so the store can't leak).
const DEFAULT_MAX_ENTRIES: usize = 256;

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
}
