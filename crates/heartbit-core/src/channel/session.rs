//! Session management for WebSocket-connected agent interactions.

#![allow(missing_docs)]
use parking_lot::RwLock;
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Error;

/// A conversation session containing message history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: Uuid,
    pub title: Option<String>,
    pub created_at: DateTime<Utc>,
    pub messages: Vec<SessionMessage>,
    /// User who owns this session (multi-tenant isolation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Tenant that owns this session (multi-tenant isolation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
}

/// A single message within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: SessionRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

/// Role of a session message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionRole {
    User,
    Assistant,
}

/// Format session history as context to prepend to a new message.
///
/// When there is prior conversation history, returns the new message prefixed
/// with a formatted history section. When history is empty, returns the message
/// unchanged.
pub fn format_session_context(history: &[SessionMessage], message: &str) -> String {
    if history.is_empty() {
        return message.to_string();
    }

    let mut ctx = String::from("## Conversation history\n");
    for msg in history {
        let role = match msg.role {
            SessionRole::User => "User",
            SessionRole::Assistant => "Assistant",
        };
        ctx.push_str(&format!("{role}: {}\n", msg.content));
    }
    ctx.push_str(&format!("\n## Current message\n{message}"));
    ctx
}

/// Trait for session persistence.
pub trait SessionStore: Send + Sync {
    /// Create a new session with an optional title.
    fn create(&self, title: Option<String>) -> Result<Session, Error>;
    /// Get a session by ID. Returns `None` if not found.
    fn get(&self, id: Uuid) -> Result<Option<Session>, Error>;
    /// List all sessions (most recent first).
    fn list(&self) -> Result<Vec<Session>, Error>;
    /// Delete a session. Returns true if found and deleted.
    fn delete(&self, id: Uuid) -> Result<bool, Error>;
    /// Append a message to an existing session.
    fn add_message(&self, id: Uuid, message: SessionMessage) -> Result<(), Error>;

    /// Create a session with user/tenant context for multi-tenant isolation.
    /// Default: delegates to `create()` and patches user/tenant fields.
    fn create_with_user(
        &self,
        title: Option<String>,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<Session, Error> {
        let mut session = self.create(title)?;
        session.user_id = Some(user_id.to_string());
        session.tenant_id = Some(tenant_id.to_string());
        Ok(session)
    }

    /// List sessions scoped to a tenant (most recent first).
    /// Default: calls `list()` and filters in-memory.
    fn list_for_tenant(&self, tenant_id: &str) -> Result<Vec<Session>, Error> {
        let all = self.list()?;
        Ok(all
            .into_iter()
            .filter(|s| s.tenant_id.as_deref() == Some(tenant_id))
            .collect())
    }
}

/// In-memory session store using `parking_lot::RwLock` (not tokio — matches
/// codebase pattern for locks never held across `.await`; `parking_lot` is
/// adopted on the channel hot path for ~2× faster uncontended reads, see T2
/// in `tasks/performance-audit-heartbit-core-2026-05-06.md`).
pub struct InMemorySessionStore {
    sessions: RwLock<HashMap<Uuid, Session>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore for InMemorySessionStore {
    fn create(&self, title: Option<String>) -> Result<Session, Error> {
        let session = Session {
            id: Uuid::new_v4(),
            title,
            created_at: Utc::now(),
            messages: Vec::new(),
            user_id: None,
            tenant_id: None,
        };
        self.sessions.write().insert(session.id, session.clone());
        Ok(session)
    }

    fn create_with_user(
        &self,
        title: Option<String>,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<Session, Error> {
        let session = Session {
            id: Uuid::new_v4(),
            title,
            created_at: Utc::now(),
            messages: Vec::new(),
            user_id: Some(user_id.to_string()),
            tenant_id: Some(tenant_id.to_string()),
        };
        self.sessions.write().insert(session.id, session.clone());
        Ok(session)
    }

    fn get(&self, id: Uuid) -> Result<Option<Session>, Error> {
        Ok(self.sessions.read().get(&id).cloned())
    }

    fn list(&self) -> Result<Vec<Session>, Error> {
        let mut list: Vec<Session> = self.sessions.read().values().cloned().collect();
        // Most recent first
        list.sort_by_key(|s| std::cmp::Reverse(s.created_at));
        Ok(list)
    }

    fn delete(&self, id: Uuid) -> Result<bool, Error> {
        Ok(self.sessions.write().remove(&id).is_some())
    }

    fn add_message(&self, id: Uuid, message: SessionMessage) -> Result<(), Error> {
        match self.sessions.write().get_mut(&id) {
            Some(session) => {
                session.messages.push(message);
                Ok(())
            }
            None => Err(Error::Channel(format!("session {id} not found"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(role: SessionRole, content: &str) -> SessionMessage {
        SessionMessage {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn create_session() {
        let store = InMemorySessionStore::new();
        let session = store.create(None).unwrap();
        assert!(session.title.is_none());
        assert!(session.messages.is_empty());
        assert!(session.created_at <= Utc::now());
    }

    #[test]
    fn create_session_with_title() {
        let store = InMemorySessionStore::new();
        let session = store.create(Some("My Chat".to_string())).unwrap();
        assert_eq!(session.title.as_deref(), Some("My Chat"));
        assert!(session.messages.is_empty());
    }

    #[test]
    fn get_existing_session() {
        let store = InMemorySessionStore::new();
        let created = store.create(Some("Test".to_string())).unwrap();
        let fetched = store
            .get(created.id)
            .unwrap()
            .expect("session should exist");
        assert_eq!(fetched.id, created.id);
        assert_eq!(fetched.title, created.title);
        assert_eq!(fetched.messages.len(), created.messages.len());
    }

    #[test]
    fn get_missing_session() {
        let store = InMemorySessionStore::new();
        let result = store.get(Uuid::new_v4()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn list_empty() {
        let store = InMemorySessionStore::new();
        let list = store.list().unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn list_multiple() {
        let store = InMemorySessionStore::new();
        store.create(None).unwrap();
        store.create(None).unwrap();
        store.create(None).unwrap();
        let list = store.list().unwrap();
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn list_ordered_by_created_at() {
        let store = InMemorySessionStore::new();
        // Create sessions — they get Utc::now() timestamps so ordering depends on
        // insertion order. To test sorting, manually insert with controlled timestamps.
        {
            let mut sessions = store.sessions.write();

            let old = Session {
                id: Uuid::new_v4(),
                title: Some("old".to_string()),
                created_at: Utc::now() - chrono::Duration::hours(2),
                messages: Vec::new(),
                user_id: None,
                tenant_id: None,
            };
            let mid = Session {
                id: Uuid::new_v4(),
                title: Some("mid".to_string()),
                created_at: Utc::now() - chrono::Duration::hours(1),
                messages: Vec::new(),
                user_id: None,
                tenant_id: None,
            };
            let new = Session {
                id: Uuid::new_v4(),
                title: Some("new".to_string()),
                created_at: Utc::now(),
                messages: Vec::new(),
                user_id: None,
                tenant_id: None,
            };

            // Insert in non-sorted order
            sessions.insert(mid.id, mid);
            sessions.insert(old.id, old);
            sessions.insert(new.id, new);
        }

        let list = store.list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].title.as_deref(), Some("new"));
        assert_eq!(list[1].title.as_deref(), Some("mid"));
        assert_eq!(list[2].title.as_deref(), Some("old"));
    }

    #[test]
    fn delete_existing() {
        let store = InMemorySessionStore::new();
        let session = store.create(None).unwrap();
        assert!(store.delete(session.id).unwrap());
        assert!(store.get(session.id).unwrap().is_none());
    }

    #[test]
    fn delete_missing() {
        let store = InMemorySessionStore::new();
        assert!(!store.delete(Uuid::new_v4()).unwrap());
    }

    #[test]
    fn add_message_to_existing() {
        let store = InMemorySessionStore::new();
        let session = store.create(None).unwrap();
        let msg = make_message(SessionRole::User, "hello");
        store.add_message(session.id, msg).unwrap();

        let fetched = store.get(session.id).unwrap().unwrap();
        assert_eq!(fetched.messages.len(), 1);
        assert_eq!(fetched.messages[0].content, "hello");
        assert_eq!(fetched.messages[0].role, SessionRole::User);
    }

    #[test]
    fn add_message_to_missing() {
        let store = InMemorySessionStore::new();
        let msg = make_message(SessionRole::User, "hello");
        let err = store.add_message(Uuid::new_v4(), msg).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn add_multiple_messages() {
        let store = InMemorySessionStore::new();
        let session = store.create(None).unwrap();

        store
            .add_message(session.id, make_message(SessionRole::User, "first"))
            .unwrap();
        store
            .add_message(session.id, make_message(SessionRole::Assistant, "second"))
            .unwrap();
        store
            .add_message(session.id, make_message(SessionRole::User, "third"))
            .unwrap();

        let fetched = store.get(session.id).unwrap().unwrap();
        assert_eq!(fetched.messages.len(), 3);
        assert_eq!(fetched.messages[0].content, "first");
        assert_eq!(fetched.messages[1].content, "second");
        assert_eq!(fetched.messages[2].content, "third");
        assert_eq!(fetched.messages[0].role, SessionRole::User);
        assert_eq!(fetched.messages[1].role, SessionRole::Assistant);
        assert_eq!(fetched.messages[2].role, SessionRole::User);
    }

    #[test]
    fn session_role_serde() {
        let user_json = serde_json::to_string(&SessionRole::User).unwrap();
        assert_eq!(user_json, "\"user\"");

        let assistant_json = serde_json::to_string(&SessionRole::Assistant).unwrap();
        assert_eq!(assistant_json, "\"assistant\"");

        let user: SessionRole = serde_json::from_str("\"user\"").unwrap();
        assert_eq!(user, SessionRole::User);

        let assistant: SessionRole = serde_json::from_str("\"assistant\"").unwrap();
        assert_eq!(assistant, SessionRole::Assistant);
    }

    #[test]
    fn session_message_roundtrip() {
        let msg = SessionMessage {
            role: SessionRole::Assistant,
            content: "Hello, world!".to_string(),
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: SessionMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.role, msg.role);
        assert_eq!(deserialized.content, msg.content);
        assert_eq!(deserialized.timestamp, msg.timestamp);
    }

    #[test]
    fn concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(InMemorySessionStore::new());
        let mut handles = Vec::new();

        // Spawn threads that create sessions
        for i in 0..10 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let session = store
                    .create(Some(format!("thread-{i}")))
                    .expect("create should succeed");
                // Add a message to the session we just created
                let msg = SessionMessage {
                    role: SessionRole::User,
                    content: format!("msg from thread {i}"),
                    timestamp: Utc::now(),
                };
                store
                    .add_message(session.id, msg)
                    .expect("add_message should succeed");
                session.id
            }));
        }

        let ids: Vec<Uuid> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All sessions should exist with one message each
        for id in &ids {
            let session = store.get(*id).unwrap().expect("session should exist");
            assert_eq!(session.messages.len(), 1);
        }

        let list = store.list().unwrap();
        assert_eq!(list.len(), 10);
    }

    // --- format_session_context tests ---

    #[test]
    fn format_context_no_history() {
        let result = format_session_context(&[], "Hello");
        assert_eq!(result, "Hello");
    }

    #[test]
    fn format_context_with_history() {
        let history = vec![
            make_message(SessionRole::User, "What is Rust?"),
            make_message(SessionRole::Assistant, "A systems programming language."),
        ];
        let result = format_session_context(&history, "Tell me more");
        assert!(result.contains("## Conversation history"));
        assert!(result.contains("User: What is Rust?"));
        assert!(result.contains("Assistant: A systems programming language."));
        assert!(result.contains("## Current message"));
        assert!(result.contains("Tell me more"));
    }

    #[test]
    fn format_context_preserves_message_order() {
        let history = vec![
            make_message(SessionRole::User, "First"),
            make_message(SessionRole::Assistant, "Second"),
            make_message(SessionRole::User, "Third"),
            make_message(SessionRole::Assistant, "Fourth"),
        ];
        let result = format_session_context(&history, "Fifth");
        let first_pos = result.find("First").unwrap();
        let second_pos = result.find("Second").unwrap();
        let third_pos = result.find("Third").unwrap();
        let fourth_pos = result.find("Fourth").unwrap();
        let fifth_pos = result.find("Fifth").unwrap();
        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
        assert!(third_pos < fourth_pos);
        assert!(fourth_pos < fifth_pos);
    }

    #[test]
    fn format_context_single_message_history() {
        let history = vec![make_message(SessionRole::User, "Prior question")];
        let result = format_session_context(&history, "Follow-up");
        assert!(result.contains("User: Prior question"));
        assert!(result.contains("Follow-up"));
    }

    // --- Multi-tenant session tests ---

    #[test]
    fn create_with_user_sets_fields() {
        let store = InMemorySessionStore::new();
        let session = store
            .create_with_user(Some("Test".into()), "alice", "acme")
            .unwrap();
        assert_eq!(session.user_id.as_deref(), Some("alice"));
        assert_eq!(session.tenant_id.as_deref(), Some("acme"));
        assert_eq!(session.title.as_deref(), Some("Test"));
    }

    #[test]
    fn create_without_user_has_none_fields() {
        let store = InMemorySessionStore::new();
        let session = store.create(None).unwrap();
        assert!(session.user_id.is_none());
        assert!(session.tenant_id.is_none());
    }

    #[test]
    fn list_for_tenant_filters_by_tenant() {
        let store = InMemorySessionStore::new();
        store
            .create_with_user(Some("acme-1".into()), "alice", "acme")
            .unwrap();
        store
            .create_with_user(Some("acme-2".into()), "bob", "acme")
            .unwrap();
        store
            .create_with_user(Some("globex-1".into()), "charlie", "globex")
            .unwrap();
        store.create(Some("legacy".into())).unwrap(); // no tenant

        let acme = store.list_for_tenant("acme").unwrap();
        assert_eq!(acme.len(), 2);
        assert!(acme.iter().all(|s| s.tenant_id.as_deref() == Some("acme")));

        let globex = store.list_for_tenant("globex").unwrap();
        assert_eq!(globex.len(), 1);
        assert_eq!(globex[0].tenant_id.as_deref(), Some("globex"));

        // Legacy sessions (no tenant) are not returned by list_for_tenant
        let all = store.list().unwrap();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn session_serde_backward_compat() {
        // Old JSON without user_id/tenant_id should deserialize with None
        let json = r#"{"id":"00000000-0000-0000-0000-000000000000","title":"old","created_at":"2026-01-01T00:00:00Z","messages":[]}"#;
        let session: Session = serde_json::from_str(json).unwrap();
        assert!(session.user_id.is_none());
        assert!(session.tenant_id.is_none());
        assert_eq!(session.title.as_deref(), Some("old"));
    }

    #[test]
    fn session_serde_with_tenant() {
        let session = Session {
            id: Uuid::nil(),
            title: None,
            created_at: Utc::now(),
            messages: Vec::new(),
            user_id: Some("alice".into()),
            tenant_id: Some("acme".into()),
        };
        let json = serde_json::to_string(&session).unwrap();
        assert!(json.contains(r#""user_id":"alice""#));
        assert!(json.contains(r#""tenant_id":"acme""#));

        let deserialized: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.user_id.as_deref(), Some("alice"));
        assert_eq!(deserialized.tenant_id.as_deref(), Some("acme"));
    }
}
