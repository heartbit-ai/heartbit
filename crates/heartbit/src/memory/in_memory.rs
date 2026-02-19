use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::RwLock;

use chrono::Utc;

use crate::error::Error;

use super::scoring::{ScoringWeights, composite_score};
use super::{Memory, MemoryEntry, MemoryQuery};

/// Thread-safe in-memory store for agent memories.
///
/// Backed by `RwLock<HashMap>`. Suitable for tests and single-process use.
/// Uses composite scoring (recency + importance + relevance) for recall ordering.
pub struct InMemoryStore {
    entries: RwLock<HashMap<String, MemoryEntry>>,
    scoring_weights: ScoringWeights,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            scoring_weights: ScoringWeights::default(),
        }
    }

    pub fn with_scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.scoring_weights = weights;
        self
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory for InMemoryStore {
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            let mut entries = self
                .entries
                .write()
                .map_err(|e| Error::Memory(format!("lock poisoned: {e}")))?;
            entries.insert(entry.id.clone(), entry);
            Ok(())
        })
    }

    fn recall(
        &self,
        query: MemoryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>> {
        Box::pin(async move {
            let has_text_query = query.text.is_some();

            // Phase 1: Read lock — filter, clone, sort (read-only)
            let mut results = {
                let entries = self
                    .entries
                    .read()
                    .map_err(|e| Error::Memory(format!("lock poisoned: {e}")))?;

                let mut results: Vec<MemoryEntry> = entries
                    .values()
                    .filter(|e| {
                        if let Some(ref text) = query.text {
                            let lower = text.to_lowercase();
                            if !e.content.to_lowercase().contains(&lower) {
                                return false;
                            }
                        }
                        if let Some(ref cat) = query.category
                            && e.category != *cat
                        {
                            return false;
                        }
                        if !query.tags.is_empty() && !query.tags.iter().any(|t| e.tags.contains(t))
                        {
                            return false;
                        }
                        if let Some(ref agent) = query.agent
                            && e.agent != *agent
                        {
                            return false;
                        }
                        true
                    })
                    .cloned()
                    .collect();

                // Sort by composite score descending (recency + importance + relevance)
                let now = Utc::now();
                results.sort_by(|a, b| {
                    let relevance_a = if has_text_query { 1.0 } else { 0.0 };
                    let relevance_b = if has_text_query { 1.0 } else { 0.0 };
                    let score_a = composite_score(
                        &self.scoring_weights,
                        a.created_at,
                        now,
                        a.importance,
                        relevance_a,
                    );
                    let score_b = composite_score(
                        &self.scoring_weights,
                        b.created_at,
                        now,
                        b.importance,
                        relevance_b,
                    );
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                if query.limit > 0 {
                    results.truncate(query.limit);
                }

                results
            }; // Read lock dropped here

            // Phase 2: Write lock — update access counts only
            if !results.is_empty() {
                let mut entries = self
                    .entries
                    .write()
                    .map_err(|e| Error::Memory(format!("lock poisoned: {e}")))?;
                let now = Utc::now();
                for r in &mut results {
                    if let Some(e) = entries.get_mut(&r.id) {
                        e.access_count += 1;
                        e.last_accessed = now;
                        // Update the returned copy too
                        r.access_count = e.access_count;
                        r.last_accessed = now;
                    }
                }
            }

            Ok(results)
        })
    }

    fn update(
        &self,
        id: &str,
        content: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        let id = id.to_string();
        Box::pin(async move {
            let mut entries = self
                .entries
                .write()
                .map_err(|e| Error::Memory(format!("lock poisoned: {e}")))?;
            match entries.get_mut(&id) {
                Some(entry) => {
                    entry.content = content;
                    entry.last_accessed = Utc::now();
                    Ok(())
                }
                None => Err(Error::Memory(format!("memory entry not found: {id}"))),
            }
        })
    }

    fn forget(&self, id: &str) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>> {
        let id = id.to_string();
        Box::pin(async move {
            let mut entries = self
                .entries
                .write()
                .map_err(|e| Error::Memory(format!("lock poisoned: {e}")))?;
            Ok(entries.remove(&id).is_some())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_entry(id: &str, agent: &str, content: &str, category: &str) -> MemoryEntry {
        MemoryEntry {
            id: id.into(),
            agent: agent.into(),
            content: content.into(),
            category: category.into(),
            tags: vec![],
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            importance: 5,
        }
    }

    fn make_entry_with_tags(
        id: &str,
        agent: &str,
        content: &str,
        category: &str,
        tags: Vec<String>,
    ) -> MemoryEntry {
        MemoryEntry {
            id: id.into(),
            agent: agent.into(),
            content: content.into(),
            category: category.into(),
            tags,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            importance: 5,
        }
    }

    #[tokio::test]
    async fn store_and_recall() {
        let store = InMemoryStore::new();
        let entry = make_entry("m1", "agent1", "Rust is fast", "fact");
        store.store(entry).await.unwrap();

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Rust is fast");
    }

    #[tokio::test]
    async fn recall_by_text() {
        let store = InMemoryStore::new();
        store
            .store(make_entry("m1", "a", "Rust is fast", "fact"))
            .await
            .unwrap();
        store
            .store(make_entry("m2", "a", "Python is slow", "fact"))
            .await
            .unwrap();

        let results = store
            .recall(MemoryQuery {
                text: Some("rust".into()),
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn recall_by_category() {
        let store = InMemoryStore::new();
        store
            .store(make_entry("m1", "a", "remember this", "fact"))
            .await
            .unwrap();
        store
            .store(make_entry("m2", "a", "I saw something", "observation"))
            .await
            .unwrap();

        let results = store
            .recall(MemoryQuery {
                category: Some("observation".into()),
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m2");
    }

    #[tokio::test]
    async fn recall_by_tags() {
        let store = InMemoryStore::new();
        store
            .store(make_entry_with_tags(
                "m1",
                "a",
                "Rust memory safety",
                "fact",
                vec!["rust".into(), "safety".into()],
            ))
            .await
            .unwrap();
        store
            .store(make_entry_with_tags(
                "m2",
                "a",
                "Go is garbage collected",
                "fact",
                vec!["go".into()],
            ))
            .await
            .unwrap();

        let results = store
            .recall(MemoryQuery {
                tags: vec!["rust".into()],
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn recall_by_agent() {
        let store = InMemoryStore::new();
        store
            .store(make_entry("m1", "researcher", "data point", "fact"))
            .await
            .unwrap();
        store
            .store(make_entry("m2", "coder", "code snippet", "procedure"))
            .await
            .unwrap();

        let results = store
            .recall(MemoryQuery {
                agent: Some("researcher".into()),
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn recall_limit() {
        let store = InMemoryStore::new();
        for i in 0..10 {
            store
                .store(make_entry(
                    &format!("m{i}"),
                    "a",
                    &format!("entry {i}"),
                    "fact",
                ))
                .await
                .unwrap();
        }

        let results = store
            .recall(MemoryQuery {
                limit: 3,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn update_existing() {
        let store = InMemoryStore::new();
        store
            .store(make_entry("m1", "a", "original", "fact"))
            .await
            .unwrap();

        store.update("m1", "updated content".into()).await.unwrap();

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results[0].content, "updated content");
    }

    #[tokio::test]
    async fn update_nonexistent() {
        let store = InMemoryStore::new();
        let err = store.update("missing", "content".into()).await.unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn forget_existing() {
        let store = InMemoryStore::new();
        store
            .store(make_entry("m1", "a", "to delete", "fact"))
            .await
            .unwrap();

        assert!(store.forget("m1").await.unwrap());

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn forget_nonexistent() {
        let store = InMemoryStore::new();
        assert!(!store.forget("missing").await.unwrap());
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemoryStore>();
    }

    #[tokio::test]
    async fn recall_sorts_by_composite_score() {
        let store = InMemoryStore::new();

        // Old entry with high importance (2 days old, importance=10)
        let mut high_imp = make_entry("m1", "a", "high importance", "fact");
        high_imp.importance = 10;
        high_imp.created_at = Utc::now() - chrono::Duration::hours(48);
        store.store(high_imp).await.unwrap();

        // Recent entry with low importance (now, importance=1)
        let mut low_imp = make_entry("m2", "a", "low importance", "fact");
        low_imp.importance = 1;
        low_imp.created_at = Utc::now();
        store.store(low_imp).await.unwrap();

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        // With default weights (0.3 recency, 0.3 importance, 0.4 relevance=0):
        //   m1: 0.3*e^(-0.01*48) + 0.3*1.0 ≈ 0.3*0.619 + 0.3 ≈ 0.486
        //   m2: 0.3*1.0 + 0.3*0.0 ≈ 0.300
        // High-importance old entry beats recent low-importance entry
        assert_eq!(results[0].id, "m1");
        assert_eq!(results[1].id, "m2");
    }

    #[tokio::test]
    async fn recall_recent_high_importance_first() {
        let store = InMemoryStore::new();

        // Old, low importance
        let mut old_low = make_entry("m1", "a", "old low", "fact");
        old_low.importance = 1;
        old_low.created_at = Utc::now() - chrono::Duration::hours(1000);
        store.store(old_low).await.unwrap();

        // Recent, high importance — should definitely be first
        let mut recent_high = make_entry("m2", "a", "recent high", "fact");
        recent_high.importance = 10;
        recent_high.created_at = Utc::now();
        store.store(recent_high).await.unwrap();

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(results[0].id, "m2");
    }

    #[tokio::test]
    async fn recall_with_custom_weights() {
        // Pure importance sorting (alpha=0, beta=1, gamma=0)
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 1.0,
            gamma: 0.0,
            decay_rate: 0.01,
        });

        let mut low = make_entry("m1", "a", "recent but low", "fact");
        low.importance = 1;
        low.created_at = Utc::now();
        store.store(low).await.unwrap();

        let mut high = make_entry("m2", "a", "old but high", "fact");
        high.importance = 10;
        high.created_at = Utc::now() - chrono::Duration::hours(1000);
        store.store(high).await.unwrap();

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();

        // Pure importance: high importance entry comes first regardless of age
        assert_eq!(results[0].id, "m2");
    }

    #[tokio::test]
    async fn recall_text_query_affects_relevance() {
        // With gamma=1 (pure relevance), matching entries should score higher
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
            decay_rate: 0.01,
        });

        let mut e1 = make_entry("m1", "a", "Rust is fast", "fact");
        e1.importance = 5;
        store.store(e1).await.unwrap();

        // Text query means relevance=1.0 for matched entries
        let results = store
            .recall(MemoryQuery {
                text: Some("Rust".into()),
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }
}
