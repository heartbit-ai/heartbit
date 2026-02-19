use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Error;

use super::{Memory, MemoryEntry, MemoryQuery};

/// Wraps a `Memory` store with namespace prefixing for agent isolation.
///
/// Each agent's memory entries get IDs prefixed with `{agent_name}:` for provenance.
/// Recall can search within the agent's namespace or across all namespaces.
pub struct NamespacedMemory {
    inner: Arc<dyn Memory>,
    agent_name: String,
}

impl NamespacedMemory {
    pub fn new(inner: Arc<dyn Memory>, agent_name: impl Into<String>) -> Self {
        Self {
            inner,
            agent_name: agent_name.into(),
        }
    }

    fn prefix_id(&self, id: &str) -> String {
        format!("{}:{}", self.agent_name, id)
    }
}

impl Memory for NamespacedMemory {
    fn store(
        &self,
        mut entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        entry.id = self.prefix_id(&entry.id);
        entry.agent = self.agent_name.clone();
        Box::pin(async move { self.inner.store(entry).await })
    }

    fn recall(
        &self,
        query: MemoryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>> {
        // If no agent filter specified, default to this agent's namespace
        let query = if query.agent.is_none() {
            MemoryQuery {
                agent: Some(self.agent_name.clone()),
                ..query
            }
        } else {
            query
        };
        let prefix = format!("{}:", self.agent_name);
        Box::pin(async move {
            let mut entries = self.inner.recall(query).await?;
            // Strip namespace prefix from IDs so consumers see unprefixed IDs.
            // This ensures update/forget (which re-add the prefix) work correctly.
            for entry in &mut entries {
                if let Some(stripped) = entry.id.strip_prefix(&prefix) {
                    entry.id = stripped.to_string();
                }
            }
            Ok(entries)
        })
    }

    fn update(
        &self,
        id: &str,
        content: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        let prefixed = self.prefix_id(id);
        Box::pin(async move { self.inner.update(&prefixed, content).await })
    }

    fn forget(&self, id: &str) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>> {
        let prefixed = self.prefix_id(id);
        Box::pin(async move { self.inner.forget(&prefixed).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::in_memory::InMemoryStore;
    use chrono::Utc;

    fn make_entry(id: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: id.into(),
            agent: String::new(),
            content: content.into(),
            category: "fact".into(),
            tags: vec![],
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            importance: 5,
        }
    }

    #[tokio::test]
    async fn store_prefixes_id_and_agent() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "researcher");

        ns.store(make_entry("m1", "test data")).await.unwrap();

        // Raw store should have prefixed entry
        let all = inner
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "researcher:m1");
        assert_eq!(all[0].agent, "researcher");

        // Namespaced recall should return unprefixed IDs
        let ns_results = ns
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(ns_results[0].id, "m1"); // prefix stripped
    }

    #[tokio::test]
    async fn recall_filters_by_agent() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns_a = NamespacedMemory::new(inner.clone(), "agent_a");
        let ns_b = NamespacedMemory::new(inner.clone(), "agent_b");

        ns_a.store(make_entry("m1", "data from A")).await.unwrap();
        ns_b.store(make_entry("m2", "data from B")).await.unwrap();

        // Agent A should only see its own memories
        let results = ns_a
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "data from A");

        // Agent B should only see its own memories
        let results = ns_b
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "data from B");
    }

    #[tokio::test]
    async fn namespace_isolation_requires_raw_store_for_cross_agent() {
        // NamespacedMemory always filters by its own agent. To read across
        // all agents, callers must use the raw inner store directly
        // (e.g., via shared_memory_tools).
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns_a = NamespacedMemory::new(inner.clone(), "agent_a");
        let ns_b = NamespacedMemory::new(inner.clone(), "agent_b");

        ns_a.store(make_entry("m1", "from A")).await.unwrap();
        ns_b.store(make_entry("m2", "from B")).await.unwrap();

        // Empty agent filter matches nothing — each namespace only sees its own
        let results = ns_a
            .recall(MemoryQuery {
                agent: Some(String::new()),
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 0);

        // Cross-agent access requires the raw inner store
        let all = inner
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn recall_then_update_roundtrip() {
        // Critical: LLM sees unprefixed IDs from recall, uses them in update.
        // update must re-prefix correctly (no double-prefix).
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "agent_a");

        ns.store(make_entry("m1", "original")).await.unwrap();

        // Recall gives us unprefixed ID
        let results = ns
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results[0].id, "m1");

        // Update using the unprefixed ID from recall
        ns.update(&results[0].id, "updated via recall ID".into())
            .await
            .unwrap();

        // Verify the update worked
        let results = ns
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results[0].content, "updated via recall ID");
    }

    #[tokio::test]
    async fn update_uses_prefixed_id() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "agent_a");

        ns.store(make_entry("m1", "original")).await.unwrap();
        ns.update("m1", "updated".into()).await.unwrap();

        let results = ns
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results[0].content, "updated");
    }

    #[tokio::test]
    async fn forget_uses_prefixed_id() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "agent_a");

        ns.store(make_entry("m1", "to delete")).await.unwrap();
        assert!(ns.forget("m1").await.unwrap());

        let results = ns
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(results.is_empty());
    }
}
