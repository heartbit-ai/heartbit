use std::sync::Arc;

use crate::error::Error;

use super::Memory;

/// Prune memory entries whose strength has decayed below a threshold
/// and are older than a minimum age.
///
/// Returns the number of entries removed.
///
/// This is a convenience wrapper around `Memory::prune()` with default
/// parameters suitable for periodic maintenance.
pub async fn prune_weak_entries(
    memory: &Arc<dyn Memory>,
    min_strength: f64,
    min_age: chrono::Duration,
) -> Result<usize, Error> {
    memory.prune(min_strength, min_age).await
}

/// Default minimum strength below which entries are prunable.
pub const DEFAULT_MIN_STRENGTH: f64 = 0.1;

/// Default minimum age before an entry can be pruned (24 hours).
pub fn default_min_age() -> chrono::Duration {
    chrono::Duration::hours(24)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::in_memory::InMemoryStore;
    use crate::memory::{MemoryEntry, MemoryQuery, MemoryType};
    use chrono::Utc;

    fn make_entry(id: &str, strength: f64, hours_ago: i64) -> MemoryEntry {
        let now = Utc::now();
        MemoryEntry {
            id: id.into(),
            agent: "test".into(),
            content: format!("content {id}"),
            category: "fact".into(),
            tags: vec![],
            created_at: now - chrono::Duration::hours(hours_ago),
            last_accessed: now,
            access_count: 0,
            importance: 5,
            memory_type: MemoryType::default(),
            keywords: vec![],
            summary: None,
            strength,
            related_ids: vec![],
            source_ids: vec![],
            embedding: None,
        }
    }

    #[tokio::test]
    async fn prune_removes_below_threshold() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        store.store(make_entry("m1", 0.05, 48)).await.unwrap(); // weak + old
        store.store(make_entry("m2", 0.8, 48)).await.unwrap(); // strong + old
        store.store(make_entry("m3", 0.05, 0)).await.unwrap(); // weak + recent

        let removed = prune_weak_entries(&store, 0.1, chrono::Duration::hours(24))
            .await
            .unwrap();
        assert_eq!(removed, 1, "only m1 should be pruned (weak + old)");

        let remaining = store
            .recall(MemoryQuery {
                limit: 0,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(remaining.len(), 2);
        let ids: Vec<&str> = remaining.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"m2"));
        assert!(ids.contains(&"m3"));
    }

    #[tokio::test]
    async fn prune_preserves_strong_entries() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        store.store(make_entry("m1", 0.9, 100)).await.unwrap();
        store.store(make_entry("m2", 0.5, 100)).await.unwrap();

        let removed = prune_weak_entries(&store, 0.1, chrono::Duration::hours(24))
            .await
            .unwrap();
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn prune_respects_min_age() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        store.store(make_entry("m1", 0.01, 1)).await.unwrap(); // weak but only 1h old

        let removed = prune_weak_entries(&store, 0.1, chrono::Duration::hours(24))
            .await
            .unwrap();
        assert_eq!(removed, 0, "entry too recent to prune");
    }

    #[tokio::test]
    async fn prune_empty_store() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let removed = prune_weak_entries(&store, 0.1, chrono::Duration::hours(24))
            .await
            .unwrap();
        assert_eq!(removed, 0);
    }
}
