//! Postgres-backed implementation of the [`SessionStore`] trait.
//!
//! Gated behind the `postgres` feature in the umbrella crate. The base
//! traits (`Session`, `SessionMessage`, `SessionRole`, `SessionStore`,
//! `InMemorySessionStore`, `format_session_context`) live in
//! [`heartbit_core::channel::session`].

use chrono::{DateTime, Utc};
use uuid::Uuid;

use heartbit_core::channel::session::{Session, SessionMessage, SessionRole, SessionStore};
use heartbit_core::error::Error;

/// Row type for reading sessions from PostgreSQL.
#[derive(Debug, sqlx::FromRow)]
struct SessionRow {
    id: Uuid,
    title: Option<String>,
    created_at: DateTime<Utc>,
    #[sqlx(default)]
    user_id: Option<String>,
    #[sqlx(default)]
    tenant_id: Option<String>,
}

/// Row type for reading session messages from PostgreSQL.
#[derive(Debug, sqlx::FromRow)]
pub(crate) struct MessageRow {
    pub(crate) role: String,
    pub(crate) content: String,
    pub(crate) created_at: DateTime<Utc>,
}

pub(crate) fn message_row_to_session_message(row: MessageRow) -> SessionMessage {
    SessionMessage {
        role: match row.role.as_str() {
            "assistant" => SessionRole::Assistant,
            _ => SessionRole::User,
        },
        content: row.content,
        timestamp: row.created_at,
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

    /// Internal helper: create session with optional user/tenant fields.
    fn create_with_fields(
        &self,
        title: Option<String>,
        user_id: Option<String>,
        tenant_id: Option<String>,
    ) -> Result<Session, Error> {
        let pool = self.pool.clone();
        let session = Session {
            id: Uuid::new_v4(),
            title: title.clone(),
            created_at: Utc::now(),
            messages: Vec::new(),
            user_id: user_id.clone(),
            tenant_id: tenant_id.clone(),
        };
        let id = session.id;
        let created_at = session.created_at;
        tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                sqlx::query(
                    "INSERT INTO sessions (id, title, created_at, user_id, tenant_id) VALUES ($1, $2, $3, $4, $5)",
                )
                .bind(id)
                .bind(title)
                .bind(created_at)
                .bind(user_id)
                .bind(tenant_id)
                .execute(&pool)
                .await
                .map_err(|e| Error::Channel(format!("failed to create session: {e}")))
            })
        })?;
        Ok(session)
    }

    /// Run the session tables migration. Safe to call multiple times.
    pub async fn run_migration(&self) -> Result<(), Error> {
        // Split into separate statements — sqlx doesn't support multiple
        // commands in a single prepared statement.
        let statements = [
            r#"CREATE TABLE IF NOT EXISTS sessions (
                id          UUID PRIMARY KEY,
                title       TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                user_id     TEXT,
                tenant_id   TEXT
            )"#,
            r#"CREATE TABLE IF NOT EXISTS session_messages (
                id          BIGSERIAL PRIMARY KEY,
                session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )"#,
            "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages(session_id)",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id TEXT",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tenant_id TEXT",
            "CREATE INDEX IF NOT EXISTS idx_sessions_tenant_id ON sessions(tenant_id)",
        ];
        for stmt in statements {
            sqlx::query(stmt)
                .execute(&self.pool)
                .await
                .map_err(|e| Error::Channel(format!("session migration failed: {e}")))?;
        }
        Ok(())
    }
}

impl SessionStore for PostgresSessionStore {
    fn create(&self, title: Option<String>) -> Result<Session, Error> {
        self.create_with_fields(title, None, None)
    }

    fn create_with_user(
        &self,
        title: Option<String>,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<Session, Error> {
        self.create_with_fields(
            title,
            Some(user_id.to_string()),
            Some(tenant_id.to_string()),
        )
    }

    fn get(&self, id: Uuid) -> Result<Option<Session>, Error> {
        let pool = self.pool.clone();
        tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
            let row: Option<SessionRow> =
                sqlx::query_as("SELECT id, title, created_at, user_id, tenant_id FROM sessions WHERE id = $1")
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
                        messages: messages.into_iter().map(message_row_to_session_message).collect(),
                        user_id: r.user_id,
                        tenant_id: r.tenant_id,
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
                "SELECT id, title, created_at, user_id, tenant_id FROM sessions ORDER BY created_at DESC",
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
                    messages: messages.into_iter().map(message_row_to_session_message).collect(),
                    user_id: r.user_id,
                    tenant_id: r.tenant_id,
                });
            }
            Ok(sessions)
        })
        })
    }

    fn list_for_tenant(&self, tenant_id: &str) -> Result<Vec<Session>, Error> {
        let pool = self.pool.clone();
        let tid = tenant_id.to_string();
        tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
            let rows: Vec<SessionRow> = sqlx::query_as(
                "SELECT id, title, created_at, user_id, tenant_id FROM sessions WHERE tenant_id = $1 ORDER BY created_at DESC",
            )
            .bind(&tid)
            .fetch_all(&pool)
            .await
            .map_err(|e| Error::Channel(format!("failed to list tenant sessions: {e}")))?;
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
                    messages: messages.into_iter().map(message_row_to_session_message).collect(),
                    user_id: r.user_id,
                    tenant_id: r.tenant_id,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_row_to_session_message_user() {
        let row = MessageRow {
            role: "user".into(),
            content: "hello".into(),
            created_at: Utc::now(),
        };
        let msg = message_row_to_session_message(row);
        assert_eq!(msg.role, SessionRole::User);
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn message_row_to_session_message_assistant() {
        let row = MessageRow {
            role: "assistant".into(),
            content: "hi there".into(),
            created_at: Utc::now(),
        };
        let msg = message_row_to_session_message(row);
        assert_eq!(msg.role, SessionRole::Assistant);
        assert_eq!(msg.content, "hi there");
    }

    #[test]
    fn message_row_unknown_role_defaults_to_user() {
        let row = MessageRow {
            role: "system".into(),
            content: "test".into(),
            created_at: Utc::now(),
        };
        let msg = message_row_to_session_message(row);
        assert_eq!(msg.role, SessionRole::User);
    }

    #[test]
    fn session_role_to_str_roundtrip() {
        assert_eq!(session_role_to_str(SessionRole::User), "user");
        assert_eq!(session_role_to_str(SessionRole::Assistant), "assistant");
    }

    #[test]
    fn message_row_preserves_timestamp() {
        let ts = Utc::now();
        let row = MessageRow {
            role: "user".into(),
            content: "test".into(),
            created_at: ts,
        };
        let msg = message_row_to_session_message(row);
        assert_eq!(msg.timestamp, ts);
    }
}
