use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use serde_json::Value;
use tokio::sync::RwLock;

use crate::error::Error;

/// Shared key-value store for multi-agent coordination.
///
/// Uses `Pin<Box<dyn Future>>` for dyn-compatibility, matching `Tool` and `Memory` traits.
pub trait Blackboard: Send + Sync {
    fn write(
        &self,
        key: &str,
        value: Value,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    fn read(
        &self,
        key: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>, Error>> + Send + '_>>;

    fn list_keys(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>, Error>> + Send + '_>>;

    fn clear(&self) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;
}

/// In-memory blackboard backed by a `tokio::sync::RwLock<HashMap>`.
///
/// Concurrent reads are allowed; writes take exclusive access.
/// Always used behind `Arc<dyn Blackboard>`, so no inner `Arc` needed.
pub struct InMemoryBlackboard {
    data: RwLock<HashMap<String, Value>>,
}

impl InMemoryBlackboard {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryBlackboard {
    fn default() -> Self {
        Self::new()
    }
}

impl Blackboard for InMemoryBlackboard {
    fn write(
        &self,
        key: &str,
        value: Value,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        let key = key.to_string();
        Box::pin(async move {
            let mut data = self.data.write().await;
            data.insert(key, value);
            Ok(())
        })
    }

    fn read(
        &self,
        key: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>, Error>> + Send + '_>> {
        let key = key.to_string();
        Box::pin(async move {
            let data = self.data.read().await;
            Ok(data.get(&key).cloned())
        })
    }

    fn list_keys(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>, Error>> + Send + '_>> {
        Box::pin(async move {
            let data = self.data.read().await;
            let mut keys: Vec<String> = data.keys().cloned().collect();
            keys.sort();
            Ok(keys)
        })
    }

    fn clear(&self) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            let mut data = self.data.write().await;
            data.clear();
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use serde_json::json;

    #[tokio::test]
    async fn write_and_read_key() {
        let bb = InMemoryBlackboard::new();
        bb.write("k1", json!({"answer": 42})).await.unwrap();
        let val = bb.read("k1").await.unwrap();
        assert_eq!(val, Some(json!({"answer": 42})));
    }

    #[tokio::test]
    async fn read_missing_key_returns_none() {
        let bb = InMemoryBlackboard::new();
        let val = bb.read("nonexistent").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn write_overwrites_existing() {
        let bb = InMemoryBlackboard::new();
        bb.write("k1", json!("first")).await.unwrap();
        bb.write("k1", json!("second")).await.unwrap();
        let val = bb.read("k1").await.unwrap();
        assert_eq!(val, Some(json!("second")));
    }

    #[tokio::test]
    async fn list_keys_empty() {
        let bb = InMemoryBlackboard::new();
        let keys = bb.list_keys().await.unwrap();
        assert!(keys.is_empty());
    }

    #[tokio::test]
    async fn list_keys_returns_sorted() {
        let bb = InMemoryBlackboard::new();
        bb.write("charlie", json!(3)).await.unwrap();
        bb.write("alpha", json!(1)).await.unwrap();
        bb.write("bravo", json!(2)).await.unwrap();
        let keys = bb.list_keys().await.unwrap();
        assert_eq!(keys, vec!["alpha", "bravo", "charlie"]);
    }

    #[tokio::test]
    async fn clear_removes_all() {
        let bb = InMemoryBlackboard::new();
        bb.write("a", json!(1)).await.unwrap();
        bb.write("b", json!(2)).await.unwrap();
        bb.clear().await.unwrap();
        let keys = bb.list_keys().await.unwrap();
        assert!(keys.is_empty());
        assert!(bb.read("a").await.unwrap().is_none());
    }

    #[test]
    fn blackboard_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemoryBlackboard>();
        // Also verify the trait object is Send + Sync
        fn _accepts_dyn(_b: &dyn Blackboard) {}
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_reads_and_writes() {
        let bb = Arc::new(InMemoryBlackboard::new());
        let mut handles = Vec::new();

        // Spawn writers
        for i in 0..10 {
            let bb = bb.clone();
            handles.push(tokio::spawn(async move {
                bb.write(&format!("key-{i}"), json!(i)).await.unwrap();
            }));
        }

        // Spawn readers concurrently
        for i in 0..10 {
            let bb = bb.clone();
            handles.push(tokio::spawn(async move {
                // May or may not see the value depending on timing
                let _ = bb.read(&format!("key-{i}")).await.unwrap();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // After all writers complete, all keys should be present
        let keys = bb.list_keys().await.unwrap();
        assert_eq!(keys.len(), 10);
    }
}
