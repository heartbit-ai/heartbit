use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Error;

use super::{Confidentiality, Memory, MemoryEntry, MemoryQuery};

/// Wraps a `Memory` store with namespace prefixing for agent isolation.
///
/// Each agent's memory entries get IDs prefixed with `{agent_name}:` for provenance.
/// Recall can search within the agent's namespace or across all namespaces.
///
/// When `max_confidentiality` is set, recall queries are capped at that level
/// regardless of what the caller requests. This is the enforcement point for
/// sensor security — even if the LLM is tricked into calling `memory_recall`,
/// the store-level filter prevents confidential data from being returned.
pub struct NamespacedMemory {
    inner: Arc<dyn Memory>,
    agent_name: String,
    max_confidentiality: Option<Confidentiality>,
    default_store_confidentiality: Confidentiality,
}

impl NamespacedMemory {
    pub fn new(inner: Arc<dyn Memory>, agent_name: impl Into<String>) -> Self {
        Self {
            inner,
            agent_name: agent_name.into(),
            max_confidentiality: None,
            default_store_confidentiality: Confidentiality::Public,
        }
    }

    /// Set the maximum confidentiality level for recall queries.
    ///
    /// When set, all recall queries through this namespace will be capped at this
    /// level — entries with higher confidentiality are filtered out at the store level.
    pub fn with_max_confidentiality(mut self, cap: Option<Confidentiality>) -> Self {
        self.max_confidentiality = cap;
        self
    }

    /// Set the minimum confidentiality level for new entries stored through this namespace.
    ///
    /// When an entry is stored with a confidentiality level below this floor, it
    /// will be upgraded to this level. Entries already at or above this level are
    /// left unchanged. This prevents LLM-driven downgrade attacks and ensures
    /// private conversations (e.g. Telegram DMs) are stored as `Confidential`
    /// by default without requiring the LLM to specify it.
    pub fn with_default_store_confidentiality(mut self, level: Confidentiality) -> Self {
        self.default_store_confidentiality = level;
        self
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
        // Enforce minimum confidentiality floor for this namespace.
        // If the entry's level is below the namespace default, upgrade it.
        // This prevents LLM-driven downgrade attacks (e.g. storing as Internal
        // when the namespace default is Confidential).
        if entry.confidentiality < self.default_store_confidentiality {
            entry.confidentiality = self.default_store_confidentiality;
        }
        Box::pin(async move { self.inner.store(entry).await })
    }

    fn recall(
        &self,
        query: MemoryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>> {
        // If no agent filter specified, default to this agent's namespace
        let mut query = if query.agent.is_none() {
            MemoryQuery {
                agent: Some(self.agent_name.clone()),
                ..query
            }
        } else {
            query
        };
        // Enforce max_confidentiality cap — use the stricter of the two
        if let Some(cap) = self.max_confidentiality {
            query.max_confidentiality = Some(match query.max_confidentiality {
                Some(existing) if existing < cap => existing,
                _ => cap,
            });
        }
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

    fn add_link(
        &self,
        id: &str,
        related_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        let prefixed_id = self.prefix_id(id);
        let prefixed_related = self.prefix_id(related_id);
        Box::pin(async move { self.inner.add_link(&prefixed_id, &prefixed_related).await })
    }

    fn prune(
        &self,
        min_strength: f64,
        min_age: chrono::Duration,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async move { self.inner.prune(min_strength, min_age).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::in_memory::InMemoryStore;
    use chrono::Utc;

    use super::super::{Confidentiality, MemoryType};

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
            memory_type: MemoryType::default(),
            keywords: vec![],
            summary: None,
            strength: 1.0,
            related_ids: vec![],
            source_ids: vec![],
            embedding: None,
            confidentiality: Confidentiality::default(),
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

    #[tokio::test]
    async fn add_link_delegates_with_prefix() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "agent_a");

        ns.store(make_entry("m1", "first")).await.unwrap();
        ns.store(make_entry("m2", "second")).await.unwrap();

        // Link via namespaced (unprefixed IDs)
        ns.add_link("m1", "m2").await.unwrap();

        // Verify in raw store that prefixed IDs are linked
        let all = inner
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        let m1 = all.iter().find(|e| e.id == "agent_a:m1").unwrap();
        let m2 = all.iter().find(|e| e.id == "agent_a:m2").unwrap();
        assert!(m1.related_ids.contains(&"agent_a:m2".to_string()));
        assert!(m2.related_ids.contains(&"agent_a:m1".to_string()));
    }

    #[tokio::test]
    async fn max_confidentiality_caps_recall() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "agent_a")
            .with_max_confidentiality(Some(Confidentiality::Public));

        // Store entries at different confidentiality levels
        let mut public_entry = make_entry("m1", "public data");
        public_entry.confidentiality = Confidentiality::Public;
        ns.store(public_entry).await.unwrap();

        let mut confidential_entry = make_entry("m2", "confidential data");
        confidential_entry.confidentiality = Confidentiality::Confidential;
        // Store via inner directly to bypass namespace (then prefix manually)
        confidential_entry.id = "agent_a:m2".into();
        confidential_entry.agent = "agent_a".into();
        inner.store(confidential_entry).await.unwrap();

        // Recall should only return the public entry
        let results = ns
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "public data");
    }

    #[tokio::test]
    async fn no_confidentiality_cap_returns_all() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "agent_a");

        let mut public_entry = make_entry("m1", "public data");
        public_entry.confidentiality = Confidentiality::Public;
        ns.store(public_entry).await.unwrap();

        let mut confidential_entry = make_entry("m2", "confidential data");
        confidential_entry.confidentiality = Confidentiality::Confidential;
        ns.store(confidential_entry).await.unwrap();

        let results = ns
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn confidentiality_cap_uses_stricter_of_two() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        // Namespace cap at Internal
        let ns = NamespacedMemory::new(inner.clone(), "agent_a")
            .with_max_confidentiality(Some(Confidentiality::Internal));

        let mut public_entry = make_entry("m1", "public data");
        public_entry.confidentiality = Confidentiality::Public;
        ns.store(public_entry).await.unwrap();

        let mut internal_entry = make_entry("m2", "internal data");
        internal_entry.confidentiality = Confidentiality::Internal;
        ns.store(internal_entry).await.unwrap();

        let mut confidential_entry = make_entry("m3", "confidential data");
        confidential_entry.confidentiality = Confidentiality::Confidential;
        // Store via inner directly (bypassing namespace)
        confidential_entry.id = "agent_a:m3".into();
        confidential_entry.agent = "agent_a".into();
        inner.store(confidential_entry).await.unwrap();

        // Even with query requesting Confidential cap, namespace cap (Internal) wins
        let results = ns
            .recall(MemoryQuery {
                limit: 10,
                max_confidentiality: Some(Confidentiality::Confidential),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 2); // Public + Internal, not Confidential

        // With query requesting Public (stricter than namespace Internal), query wins
        let results = ns
            .recall(MemoryQuery {
                limit: 10,
                max_confidentiality: Some(Confidentiality::Public),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1); // Only Public
    }

    #[tokio::test]
    async fn default_store_confidentiality_upgrades_public() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "tg_agent")
            .with_default_store_confidentiality(Confidentiality::Confidential);

        // Store with default (Public) → should be upgraded to Confidential
        let entry = make_entry("m1", "private chat data");
        ns.store(entry).await.unwrap();

        // Check raw store: entry should be stored as Confidential
        let all = inner
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].confidentiality, Confidentiality::Confidential);
    }

    #[tokio::test]
    async fn default_store_confidentiality_enforces_minimum_floor() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "tg_agent")
            .with_default_store_confidentiality(Confidentiality::Confidential);

        // Store with Internal (below Confidential floor) → should be upgraded
        let mut entry = make_entry("m1", "internal data");
        entry.confidentiality = Confidentiality::Internal;
        ns.store(entry).await.unwrap();

        let all = inner
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].confidentiality, Confidentiality::Confidential);
    }

    #[tokio::test]
    async fn default_store_confidentiality_preserves_higher_level() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "tg_agent")
            .with_default_store_confidentiality(Confidentiality::Confidential);

        // Store with Restricted (above Confidential floor) → should NOT be changed
        let mut entry = make_entry("m1", "secret data");
        entry.confidentiality = Confidentiality::Restricted;
        ns.store(entry).await.unwrap();

        let all = inner
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].confidentiality, Confidentiality::Restricted);
    }

    #[tokio::test]
    async fn prune_delegates_to_inner() {
        let inner: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let ns = NamespacedMemory::new(inner.clone(), "agent_a");

        let mut entry = make_entry("m1", "weak memory");
        entry.strength = 0.01;
        entry.created_at = Utc::now() - chrono::Duration::hours(48);
        entry.last_accessed = Utc::now() - chrono::Duration::hours(48);
        ns.store(entry).await.unwrap();

        let pruned = ns.prune(0.1, chrono::Duration::hours(1)).await.unwrap();
        assert_eq!(pruned, 1);

        // Verify entry is gone
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
