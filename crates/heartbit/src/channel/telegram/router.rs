use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use uuid::Uuid;

use crate::channel::session::SessionStore;
use crate::error::Error;

/// Per-chat tracking state.
struct ChatState {
    session_id: Uuid,
    user_id: i64,
    last_activity: Instant,
    created_at: Instant,
}

/// Idle session info returned by `collect_idle`.
pub struct IdleSession {
    pub chat_id: i64,
    pub session_id: Uuid,
    pub user_id: i64,
}

/// Expired session info returned by `collect_expired`.
pub struct ExpiredSession {
    pub chat_id: i64,
    pub session_id: Uuid,
}

/// Maps Telegram chat IDs to session IDs with inactivity and expiry tracking.
///
/// Uses `std::sync::RwLock` (never held across `.await`) per codebase convention.
pub struct ChatSessionMap {
    map: RwLock<HashMap<i64, ChatState>>,
    inactivity_timeout: Duration,
    expiry_timeout: Duration,
}

impl ChatSessionMap {
    pub fn new(inactivity_timeout: Duration, expiry_timeout: Duration) -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
            inactivity_timeout,
            expiry_timeout,
        }
    }

    /// Get the session ID for a chat, or create a new session.
    /// Returns `(session_id, is_new)`.
    pub fn get_or_create(
        &self,
        chat_id: i64,
        user_id: i64,
        store: &dyn SessionStore,
    ) -> Result<(Uuid, bool), Error> {
        // Fast path: read lock to check existing
        {
            let map = self
                .map
                .read()
                .map_err(|e| Error::Telegram(format!("lock poisoned: {e}")))?;
            if let Some(state) = map.get(&chat_id) {
                return Ok((state.session_id, false));
            }
        }

        // Slow path: write lock to insert
        let mut map = self
            .map
            .write()
            .map_err(|e| Error::Telegram(format!("lock poisoned: {e}")))?;

        // Double-check after acquiring write lock
        if let Some(state) = map.get(&chat_id) {
            return Ok((state.session_id, false));
        }

        let session = store.create(Some(format!("tg:{chat_id}")))?;
        let session_id = session.id;
        map.insert(
            chat_id,
            ChatState {
                session_id,
                user_id,
                last_activity: Instant::now(),
                created_at: Instant::now(),
            },
        );
        Ok((session_id, true))
    }

    /// Update the last activity timestamp for a chat.
    pub fn touch(&self, chat_id: i64) {
        if let Ok(mut map) = self.map.write()
            && let Some(state) = map.get_mut(&chat_id)
        {
            state.last_activity = Instant::now();
        }
    }

    /// Collect sessions that have been inactive longer than `inactivity_timeout`.
    /// Does NOT remove them — the caller should consolidate, then optionally remove.
    pub fn collect_idle(&self) -> Vec<IdleSession> {
        let map = match self.map.read() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let now = Instant::now();
        map.iter()
            .filter(|(_, state)| now.duration_since(state.last_activity) >= self.inactivity_timeout)
            .map(|(&chat_id, state)| IdleSession {
                chat_id,
                session_id: state.session_id,
                user_id: state.user_id,
            })
            .collect()
    }

    /// Collect sessions that have exceeded the expiry timeout since creation.
    /// Does NOT remove them — caller should clean up via `remove`.
    pub fn collect_expired(&self) -> Vec<ExpiredSession> {
        let map = match self.map.read() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let now = Instant::now();
        map.iter()
            .filter(|(_, state)| now.duration_since(state.created_at) >= self.expiry_timeout)
            .map(|(&chat_id, state)| ExpiredSession {
                chat_id,
                session_id: state.session_id,
            })
            .collect()
    }

    /// Remove a chat from the map.
    pub fn remove(&self, chat_id: i64) {
        if let Ok(mut map) = self.map.write() {
            map.remove(&chat_id);
        }
    }

    /// Get the current number of tracked chats.
    pub fn len(&self) -> usize {
        self.map.read().map(|m| m.len()).unwrap_or(0)
    }

    /// Check if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::session::InMemorySessionStore;

    fn make_map() -> ChatSessionMap {
        ChatSessionMap::new(Duration::from_millis(50), Duration::from_millis(200))
    }

    #[test]
    fn create_new_session() {
        let map = make_map();
        let store = InMemorySessionStore::new();
        let (sid, is_new) = map.get_or_create(100, 1, &store).unwrap();
        assert!(is_new);
        // Session should exist in the store
        assert!(store.get(sid).unwrap().is_some());
    }

    #[test]
    fn resume_existing_session() {
        let map = make_map();
        let store = InMemorySessionStore::new();
        let (sid1, new1) = map.get_or_create(100, 1, &store).unwrap();
        assert!(new1);
        let (sid2, new2) = map.get_or_create(100, 1, &store).unwrap();
        assert!(!new2);
        assert_eq!(sid1, sid2);
    }

    #[test]
    fn different_chats_different_sessions() {
        let map = make_map();
        let store = InMemorySessionStore::new();
        let (sid1, _) = map.get_or_create(100, 1, &store).unwrap();
        let (sid2, _) = map.get_or_create(200, 2, &store).unwrap();
        assert_ne!(sid1, sid2);
    }

    #[test]
    fn touch_updates_activity() {
        let map = make_map();
        let store = InMemorySessionStore::new();
        map.get_or_create(100, 1, &store).unwrap();

        // Before touch, check idle
        std::thread::sleep(Duration::from_millis(60));
        let idle = map.collect_idle();
        assert_eq!(idle.len(), 1);

        // After touch, no longer idle
        map.touch(100);
        let idle = map.collect_idle();
        assert!(idle.is_empty());
    }

    #[test]
    fn collect_idle_returns_inactive() {
        let map = make_map();
        let store = InMemorySessionStore::new();
        map.get_or_create(100, 1, &store).unwrap();
        map.get_or_create(200, 2, &store).unwrap();

        // Neither should be idle yet
        let idle = map.collect_idle();
        assert!(idle.is_empty());

        // Wait for inactivity timeout
        std::thread::sleep(Duration::from_millis(60));
        let idle = map.collect_idle();
        assert_eq!(idle.len(), 2);
    }

    #[test]
    fn collect_expired_returns_old_sessions() {
        let map = make_map();
        let store = InMemorySessionStore::new();
        map.get_or_create(100, 1, &store).unwrap();

        // Not expired yet
        let expired = map.collect_expired();
        assert!(expired.is_empty());

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(210));
        let expired = map.collect_expired();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].chat_id, 100);
    }

    #[test]
    fn remove_clears_entry() {
        let map = make_map();
        let store = InMemorySessionStore::new();
        map.get_or_create(100, 1, &store).unwrap();
        assert_eq!(map.len(), 1);

        map.remove(100);
        assert!(map.is_empty());
    }

    #[test]
    fn remove_nonexistent_is_noop() {
        let map = make_map();
        map.remove(999);
        assert!(map.is_empty());
    }

    #[test]
    fn concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let map = Arc::new(make_map());
        let store = Arc::new(InMemorySessionStore::new());
        let mut handles = Vec::new();

        for i in 0..10 {
            let map = Arc::clone(&map);
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                map.get_or_create(i, i, store.as_ref()).unwrap();
                map.touch(i);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(map.len(), 10);
    }

    #[test]
    fn session_title_contains_chat_id() {
        let map = make_map();
        let store = InMemorySessionStore::new();
        let (sid, _) = map.get_or_create(42, 1, &store).unwrap();
        let session = store.get(sid).unwrap().unwrap();
        assert_eq!(session.title.as_deref(), Some("tg:42"));
    }

    #[test]
    fn len_and_is_empty() {
        let map = make_map();
        let store = InMemorySessionStore::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.get_or_create(100, 1, &store).unwrap();
        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);
    }
}
