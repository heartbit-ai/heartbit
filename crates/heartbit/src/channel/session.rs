use std::collections::HashMap;
use std::sync::RwLock;

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
}

/// In-memory session store using `std::sync::RwLock` (not tokio — matches codebase pattern
/// for locks never held across `.await`).
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
        };
        let mut sessions = self
            .sessions
            .write()
            .map_err(|e| Error::Channel(format!("lock poisoned: {e}")))?;
        sessions.insert(session.id, session.clone());
        Ok(session)
    }

    fn get(&self, id: Uuid) -> Result<Option<Session>, Error> {
        let sessions = self
            .sessions
            .read()
            .map_err(|e| Error::Channel(format!("lock poisoned: {e}")))?;
        Ok(sessions.get(&id).cloned())
    }

    fn list(&self) -> Result<Vec<Session>, Error> {
        let sessions = self
            .sessions
            .read()
            .map_err(|e| Error::Channel(format!("lock poisoned: {e}")))?;
        let mut list: Vec<Session> = sessions.values().cloned().collect();
        // Most recent first
        list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(list)
    }

    fn delete(&self, id: Uuid) -> Result<bool, Error> {
        let mut sessions = self
            .sessions
            .write()
            .map_err(|e| Error::Channel(format!("lock poisoned: {e}")))?;
        Ok(sessions.remove(&id).is_some())
    }

    fn add_message(&self, id: Uuid, message: SessionMessage) -> Result<(), Error> {
        let mut sessions = self
            .sessions
            .write()
            .map_err(|e| Error::Channel(format!("lock poisoned: {e}")))?;
        match sessions.get_mut(&id) {
            Some(session) => {
                session.messages.push(message);
                Ok(())
            }
            None => Err(Error::Channel(format!("session {id} not found"))),
        }
    }
}

// --- PostgreSQL session store ---

#[cfg(feature = "postgres")]
mod postgres_session {
    use super::*;

    /// Row type for reading sessions from PostgreSQL.
    #[derive(Debug, sqlx::FromRow)]
    struct SessionRow {
        id: Uuid,
        title: Option<String>,
        created_at: DateTime<Utc>,
    }

    /// Row type for reading session messages from PostgreSQL.
    #[derive(Debug, sqlx::FromRow)]
    pub(crate) struct MessageRow {
        pub(crate) role: String,
        pub(crate) content: String,
        pub(crate) created_at: DateTime<Utc>,
    }

    impl From<MessageRow> for SessionMessage {
        fn from(row: MessageRow) -> Self {
            Self {
                role: match row.role.as_str() {
                    "assistant" => SessionRole::Assistant,
                    _ => SessionRole::User,
                },
                content: row.content,
                timestamp: row.created_at,
            }
        }
    }

    pub(crate) fn session_role_to_str(role: SessionRole) -> &'static str {
        match role {
            SessionRole::User => "user",
            SessionRole::Assistant => "assistant",
        }
    }

    /// PostgreSQL-backed session store for durable conversation persistence.
    ///
    /// Uses `sqlx` runtime queries (no compile-time macros). Two tables:
    /// `sessions` (id, title, created_at) and `session_messages` (session_id, role,
    /// content, created_at). Foreign key cascade on delete.
    pub struct PostgresSessionStore {
        pool: sqlx::PgPool,
    }

    impl PostgresSessionStore {
        /// Create from an existing connection pool.
        pub fn new(pool: sqlx::PgPool) -> Self {
            Self { pool }
        }

        /// Connect to PostgreSQL using the given URL.
        pub async fn connect(database_url: &str) -> Result<Self, Error> {
            let pool = sqlx::PgPool::connect(database_url)
                .await
                .map_err(|e| Error::Channel(format!("database connection failed: {e}")))?;
            Ok(Self { pool })
        }

        /// Run the session tables migration. Safe to call multiple times.
        pub async fn run_migration(&self) -> Result<(), Error> {
            sqlx::query(
                r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id          UUID PRIMARY KEY,
                title       TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS session_messages (
                id          BIGSERIAL PRIMARY KEY,
                session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
                ON session_messages(session_id);
            "#,
            )
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Channel(format!("session migration failed: {e}")))?;
            Ok(())
        }
    }

    impl SessionStore for PostgresSessionStore {
        fn create(&self, title: Option<String>) -> Result<Session, Error> {
            let pool = self.pool.clone();
            let session = Session {
                id: Uuid::new_v4(),
                title: title.clone(),
                created_at: Utc::now(),
                messages: Vec::new(),
            };
            let id = session.id;
            let created_at = session.created_at;
            // Use block_in_place since SessionStore trait methods are sync.
            // The pool operations are async but the trait requires sync returns.
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    sqlx::query("INSERT INTO sessions (id, title, created_at) VALUES ($1, $2, $3)")
                        .bind(id)
                        .bind(title)
                        .bind(created_at)
                        .execute(&pool)
                        .await
                        .map_err(|e| Error::Channel(format!("failed to create session: {e}")))
                })
            })?;
            Ok(session)
        }

        fn get(&self, id: Uuid) -> Result<Option<Session>, Error> {
            let pool = self.pool.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                let row: Option<SessionRow> =
                    sqlx::query_as("SELECT id, title, created_at FROM sessions WHERE id = $1")
                        .bind(id)
                        .fetch_optional(&pool)
                        .await
                        .map_err(|e| {
                            Error::Channel(format!("failed to get session: {e}"))
                        })?;
                match row {
                    Some(r) => {
                        let messages: Vec<MessageRow> = sqlx::query_as(
                            "SELECT role, content, created_at FROM session_messages WHERE session_id = $1 ORDER BY created_at, id",
                        )
                        .bind(id)
                        .fetch_all(&pool)
                        .await
                        .map_err(|e| Error::Channel(format!("failed to load messages: {e}")))?;
                        Ok(Some(Session {
                            id: r.id,
                            title: r.title,
                            created_at: r.created_at,
                            messages: messages.into_iter().map(SessionMessage::from).collect(),
                        }))
                    }
                    None => Ok(None),
                }
            })
            })
        }

        fn list(&self) -> Result<Vec<Session>, Error> {
            let pool = self.pool.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                let rows: Vec<SessionRow> = sqlx::query_as(
                    "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC",
                )
                .fetch_all(&pool)
                .await
                .map_err(|e| Error::Channel(format!("failed to list sessions: {e}")))?;
                let mut sessions = Vec::with_capacity(rows.len());
                for r in rows {
                    let messages: Vec<MessageRow> = sqlx::query_as(
                        "SELECT role, content, created_at FROM session_messages WHERE session_id = $1 ORDER BY created_at, id",
                    )
                    .bind(r.id)
                    .fetch_all(&pool)
                    .await
                    .map_err(|e| Error::Channel(format!("failed to load messages: {e}")))?;
                    sessions.push(Session {
                        id: r.id,
                        title: r.title,
                        created_at: r.created_at,
                        messages: messages.into_iter().map(SessionMessage::from).collect(),
                    });
                }
                Ok(sessions)
            })
            })
        }

        fn delete(&self, id: Uuid) -> Result<bool, Error> {
            let pool = self.pool.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                    let result = sqlx::query("DELETE FROM sessions WHERE id = $1")
                        .bind(id)
                        .execute(&pool)
                        .await
                        .map_err(|e| Error::Channel(format!("failed to delete session: {e}")))?;
                    Ok(result.rows_affected() > 0)
                })
            })
        }

        fn add_message(&self, id: Uuid, message: SessionMessage) -> Result<(), Error> {
            let pool = self.pool.clone();
            tokio::task::block_in_place(move || {
                tokio::runtime::Handle::current().block_on(async move {
                // Verify session exists
                let exists: bool = sqlx::query_scalar(
                    "SELECT EXISTS(SELECT 1 FROM sessions WHERE id = $1)",
                )
                .bind(id)
                .fetch_one(&pool)
                .await
                .map_err(|e| Error::Channel(format!("failed to check session: {e}")))?;
                if !exists {
                    return Err(Error::Channel(format!("session {id} not found")));
                }
                sqlx::query(
                    "INSERT INTO session_messages (session_id, role, content, created_at) VALUES ($1, $2, $3, $4)",
                )
                .bind(id)
                .bind(session_role_to_str(message.role))
                .bind(&message.content)
                .bind(message.timestamp)
                .execute(&pool)
                .await
                .map_err(|e| Error::Channel(format!("failed to add message: {e}")))?;
                Ok(())
            })
            })
        }
    }
} // mod postgres_session

#[cfg(feature = "postgres")]
pub use postgres_session::PostgresSessionStore;

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
            let mut sessions = store.sessions.write().unwrap();

            let old = Session {
                id: Uuid::new_v4(),
                title: Some("old".to_string()),
                created_at: Utc::now() - chrono::Duration::hours(2),
                messages: Vec::new(),
            };
            let mid = Session {
                id: Uuid::new_v4(),
                title: Some("mid".to_string()),
                created_at: Utc::now() - chrono::Duration::hours(1),
                messages: Vec::new(),
            };
            let new = Session {
                id: Uuid::new_v4(),
                title: Some("new".to_string()),
                created_at: Utc::now(),
                messages: Vec::new(),
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

    // --- PostgresSessionStore unit tests (row conversion, no DB needed) ---

    #[cfg(feature = "postgres")]
    use postgres_session::*;

    #[cfg(feature = "postgres")]
    #[test]
    fn message_row_to_session_message_user() {
        let row = MessageRow {
            role: "user".into(),
            content: "hello".into(),
            created_at: Utc::now(),
        };
        let msg = SessionMessage::from(row);
        assert_eq!(msg.role, SessionRole::User);
        assert_eq!(msg.content, "hello");
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn message_row_to_session_message_assistant() {
        let row = MessageRow {
            role: "assistant".into(),
            content: "hi there".into(),
            created_at: Utc::now(),
        };
        let msg = SessionMessage::from(row);
        assert_eq!(msg.role, SessionRole::Assistant);
        assert_eq!(msg.content, "hi there");
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn message_row_unknown_role_defaults_to_user() {
        let row = MessageRow {
            role: "system".into(),
            content: "test".into(),
            created_at: Utc::now(),
        };
        let msg = SessionMessage::from(row);
        assert_eq!(msg.role, SessionRole::User);
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn session_role_to_str_roundtrip() {
        assert_eq!(session_role_to_str(SessionRole::User), "user");
        assert_eq!(session_role_to_str(SessionRole::Assistant), "assistant");
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn message_row_preserves_timestamp() {
        let ts = Utc::now();
        let row = MessageRow {
            role: "user".into(),
            content: "test".into(),
            created_at: ts,
        };
        let msg = SessionMessage::from(row);
        assert_eq!(msg.timestamp, ts);
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
}
