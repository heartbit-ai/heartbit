//! LLM response caching for deterministic replay and cost reduction.

use std::sync::Mutex;

use crate::llm::types::{CompletionResponse, Message};
use crate::util::fnv1a_hash;

/// LRU cache for LLM completion responses.
///
/// Thread-safe via `std::sync::Mutex` (never held across `.await`).
/// Entries are keyed by FNV-1a hash of (system_prompt, messages, sorted tool names).
///
/// Uses a `Vec` with move-to-front on hit and eviction from back, giving O(n)
/// operations per access. This is efficient for typical capacities (10–100).
/// For very large caches (1000+), consider an alternative implementation.
pub struct ResponseCache {
    entries: Mutex<Vec<(u64, CompletionResponse)>>,
    capacity: usize,
}

impl std::fmt::Debug for ResponseCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.entries.lock().map(|e| e.len()).unwrap_or(0);
        f.debug_struct("ResponseCache")
            .field("capacity", &self.capacity)
            .field("len", &len)
            .finish()
    }
}

impl ResponseCache {
    /// Create a new cache with the given maximum number of entries.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(Vec::with_capacity(capacity)),
            capacity,
        }
    }

    /// Look up a cached response by key. On hit, moves the entry to the front (LRU).
    pub fn get(&self, key: u64) -> Option<CompletionResponse> {
        let mut entries = self.entries.lock().expect("cache lock poisoned");
        if let Some(pos) = entries.iter().position(|(k, _)| *k == key) {
            let entry = entries.remove(pos);
            let response = entry.1.clone();
            entries.insert(0, entry);
            Some(response)
        } else {
            None
        }
    }

    /// Insert a response into the cache. Evicts the least-recently-used entry
    /// if at capacity.
    pub fn put(&self, key: u64, response: CompletionResponse) {
        let mut entries = self.entries.lock().expect("cache lock poisoned");
        // Remove existing entry with same key (will be re-inserted at front)
        if let Some(pos) = entries.iter().position(|(k, _)| *k == key) {
            entries.remove(pos);
        }
        // Evict from back if at capacity
        if entries.len() >= self.capacity {
            entries.pop();
        }
        entries.insert(0, (key, response));
    }

    /// Compute a cache key from the request components.
    ///
    /// Uses FNV-1a hash of system prompt, serialized messages, and sorted tool names.
    pub fn compute_key(system_prompt: &str, messages: &[Message], tool_names: &[&str]) -> u64 {
        let mut sorted_names: Vec<&str> = tool_names.to_vec();
        sorted_names.sort();

        let messages_json = serde_json::to_string(messages).expect("messages serialize infallibly");

        let mut data = Vec::new();
        data.extend_from_slice(system_prompt.as_bytes());
        data.push(0); // separator
        data.extend_from_slice(messages_json.as_bytes());
        data.push(0); // separator
        for name in &sorted_names {
            data.extend_from_slice(name.as_bytes());
            data.push(0);
        }

        fnv1a_hash(&data)
    }

    /// Remove all entries from the cache.
    pub fn clear(&self) {
        let mut entries = self.entries.lock().expect("cache lock poisoned");
        entries.clear();
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.lock().expect("cache lock poisoned").len()
    }

    /// Returns true if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.lock().expect("cache lock poisoned").is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ContentBlock, Message, StopReason, TokenUsage};

    fn make_response(text: &str) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        }
    }

    #[test]
    fn cache_stores_and_retrieves() {
        let cache = ResponseCache::new(10);
        let response = make_response("hello");
        let key = 42;
        cache.put(key, response.clone());

        let cached = cache.get(key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().text(), "hello");
    }

    #[test]
    fn cache_miss_returns_none() {
        let cache = ResponseCache::new(10);
        assert!(cache.get(999).is_none());
    }

    #[test]
    fn lru_eviction() {
        let cache = ResponseCache::new(3);
        cache.put(1, make_response("one"));
        cache.put(2, make_response("two"));
        cache.put(3, make_response("three"));

        // Cache is full. Insert a 4th entry — should evict key 1 (oldest).
        cache.put(4, make_response("four"));
        assert_eq!(cache.len(), 3);
        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
        assert!(cache.get(4).is_some());
    }

    #[test]
    fn lru_access_refreshes_order() {
        let cache = ResponseCache::new(3);
        cache.put(1, make_response("one"));
        cache.put(2, make_response("two"));
        cache.put(3, make_response("three"));

        // Access key 1 to move it to front
        let _ = cache.get(1);

        // Insert key 4 — should evict key 2 (now oldest)
        cache.put(4, make_response("four"));
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
        assert!(cache.get(4).is_some());
    }

    #[test]
    fn compute_key_deterministic() {
        let msgs = vec![Message::user("hello")];
        let tools = vec!["search", "read"];

        let key1 = ResponseCache::compute_key("system", &msgs, &tools);
        let key2 = ResponseCache::compute_key("system", &msgs, &tools);
        assert_eq!(key1, key2);
    }

    #[test]
    fn compute_key_different_for_different_inputs() {
        let msgs_a = vec![Message::user("hello")];
        let msgs_b = vec![Message::user("world")];
        let tools = vec!["search"];

        let key_a = ResponseCache::compute_key("system", &msgs_a, &tools);
        let key_b = ResponseCache::compute_key("system", &msgs_b, &tools);
        assert_ne!(key_a, key_b);

        // Different system prompt
        let key_c = ResponseCache::compute_key("other", &msgs_a, &tools);
        assert_ne!(key_a, key_c);

        // Different tools
        let key_d = ResponseCache::compute_key("system", &msgs_a, &["write"]);
        assert_ne!(key_a, key_d);
    }

    #[test]
    fn compute_key_tool_order_independent() {
        let msgs = vec![Message::user("hi")];
        let key1 = ResponseCache::compute_key("sys", &msgs, &["a", "b", "c"]);
        let key2 = ResponseCache::compute_key("sys", &msgs, &["c", "a", "b"]);
        assert_eq!(key1, key2);
    }

    #[test]
    fn clear_empties_cache() {
        let cache = ResponseCache::new(10);
        cache.put(1, make_response("one"));
        cache.put(2, make_response("two"));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn put_overwrites_existing_key() {
        let cache = ResponseCache::new(10);
        cache.put(1, make_response("first"));
        cache.put(1, make_response("second"));

        assert_eq!(cache.len(), 1);
        let cached = cache.get(1).unwrap();
        assert_eq!(cached.text(), "second");
    }

    #[test]
    fn thread_safety() {
        use std::sync::Arc;

        let cache = Arc::new(ResponseCache::new(100));
        let mut handles = vec![];

        for i in 0..10 {
            let cache = cache.clone();
            handles.push(std::thread::spawn(move || {
                for j in 0..100 {
                    let key = (i * 100 + j) as u64;
                    cache.put(key, make_response(&format!("resp-{key}")));
                    let _ = cache.get(key);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert!(cache.len() <= 100);
    }
}
