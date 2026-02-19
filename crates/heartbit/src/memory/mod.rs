pub mod in_memory;
pub mod namespaced;
pub mod postgres;
pub mod scoring;
pub mod shared_tools;
pub mod tools;

use std::future::Future;
use std::pin::Pin;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// A single memory entry stored by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub agent: String,
    pub content: String,
    pub category: String,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    /// Importance score (1-10). Default: 5. Set by agent at store time.
    #[serde(default = "default_importance")]
    pub importance: u8,
}

pub(crate) fn default_importance() -> u8 {
    5
}

pub(crate) fn default_category() -> String {
    "fact".into()
}

pub(crate) fn default_recall_limit() -> usize {
    10
}

/// Query parameters for recalling memories.
///
/// `limit` controls the maximum number of results returned. A value of `0`
/// means no limit (return all matching entries). This is the default.
#[derive(Debug, Clone, Default)]
pub struct MemoryQuery {
    pub text: Option<String>,
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub agent: Option<String>,
    /// Maximum number of results. `0` means unlimited.
    pub limit: usize,
}

/// Trait for persistent memory stores.
///
/// Uses `Pin<Box<dyn Future>>` for dyn-compatibility, matching the `Tool` trait pattern.
pub trait Memory: Send + Sync {
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    fn recall(
        &self,
        query: MemoryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>>;

    fn update(
        &self,
        id: &str,
        content: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    fn forget(&self, id: &str) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_entry_serializes() {
        let entry = MemoryEntry {
            id: "m1".into(),
            agent: "researcher".into(),
            content: "Rust is fast".into(),
            category: "fact".into(),
            tags: vec!["rust".into()],
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            importance: 7,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "m1");
        assert_eq!(parsed.agent, "researcher");
        assert_eq!(parsed.content, "Rust is fast");
        assert_eq!(parsed.importance, 7);
    }

    #[test]
    fn memory_entry_default_importance() {
        let entry = MemoryEntry {
            id: "m1".into(),
            agent: "a".into(),
            content: "test".into(),
            category: "fact".into(),
            tags: vec![],
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            importance: default_importance(),
        };
        assert_eq!(entry.importance, 5);
    }

    #[test]
    fn memory_entry_deserialize_without_importance() {
        let json = r#"{"id":"m1","agent":"a","content":"test","category":"fact","tags":[],"created_at":"2024-01-01T00:00:00Z","last_accessed":"2024-01-01T00:00:00Z","access_count":0}"#;
        let entry: MemoryEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.importance, 5); // default
    }

    #[test]
    fn memory_entry_deserialize_with_importance() {
        let json = r#"{"id":"m1","agent":"a","content":"test","category":"fact","tags":[],"created_at":"2024-01-01T00:00:00Z","last_accessed":"2024-01-01T00:00:00Z","access_count":0,"importance":9}"#;
        let entry: MemoryEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.importance, 9);
    }

    #[test]
    fn memory_query_default() {
        let q = MemoryQuery::default();
        assert!(q.text.is_none());
        assert!(q.category.is_none());
        assert!(q.tags.is_empty());
        assert!(q.agent.is_none());
        assert_eq!(q.limit, 0);
    }

    #[test]
    fn memory_trait_is_object_safe() {
        // Verify Memory can be used as dyn trait
        fn _accepts_dyn(_m: &dyn Memory) {}
    }
}
