use std::future::Future;
use std::pin::Pin;

use chrono::{DateTime, Utc};
use sqlx::{FromRow, PgPool, Row};

use crate::error::Error;

use super::scoring::{ScoringWeights, composite_score};
use super::{Memory, MemoryEntry, MemoryQuery};

/// Row type for reading memories from PostgreSQL.
#[derive(Debug, FromRow)]
struct MemoryRow {
    id: String,
    agent: String,
    content: String,
    category: String,
    tags: Vec<String>,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    access_count: i32,
    importance: i16,
}

impl From<MemoryRow> for MemoryEntry {
    fn from(row: MemoryRow) -> Self {
        Self {
            id: row.id,
            agent: row.agent,
            content: row.content,
            category: row.category,
            tags: row.tags,
            created_at: row.created_at,
            last_accessed: row.last_accessed,
            access_count: row.access_count as u32,
            importance: row.importance.clamp(1, 10) as u8,
        }
    }
}

/// PostgreSQL-backed memory store for durable agent memory persistence.
///
/// Uses `sqlx::query_as()` (runtime queries, no compile-time macros).
/// Recall filtering uses SQL WHERE clauses with `ILIKE` for text search.
/// All matching rows are fetched, then scored and truncated in-memory using
/// composite scoring (recency + importance + relevance) for consistency
/// with `InMemoryStore`.
pub struct PostgresMemoryStore {
    pool: PgPool,
    scoring_weights: ScoringWeights,
}

impl PostgresMemoryStore {
    /// Create from an existing connection pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            scoring_weights: ScoringWeights::default(),
        }
    }

    /// Connect to PostgreSQL using the given URL.
    pub async fn connect(database_url: &str) -> Result<Self, Error> {
        let pool = PgPool::connect(database_url)
            .await
            .map_err(|e| Error::Memory(format!("database connection failed: {e}")))?;
        Ok(Self {
            pool,
            scoring_weights: ScoringWeights::default(),
        })
    }

    /// Set custom scoring weights for recall ordering.
    pub fn with_scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.scoring_weights = weights;
        self
    }

    /// Run the memory table migration. Safe to call multiple times.
    pub async fn run_migration(&self) -> Result<(), Error> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id              TEXT PRIMARY KEY,
                agent           TEXT NOT NULL,
                content         TEXT NOT NULL,
                category        TEXT NOT NULL DEFAULT 'fact',
                tags            TEXT[] NOT NULL DEFAULT '{}',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_accessed   TIMESTAMPTZ NOT NULL DEFAULT now(),
                access_count    INT NOT NULL DEFAULT 0,
                importance      SMALLINT NOT NULL DEFAULT 5
            );
            CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent);
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| Error::Memory(format!("memory migration failed: {e}")))?;
        Ok(())
    }
}

impl Memory for PostgresMemoryStore {
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            sqlx::query(
                r#"
                INSERT INTO memories (id, agent, content, category, tags, created_at, last_accessed, access_count, importance)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    tags = EXCLUDED.tags,
                    importance = EXCLUDED.importance
                "#,
            )
            .bind(&entry.id)
            .bind(&entry.agent)
            .bind(&entry.content)
            .bind(&entry.category)
            .bind(&entry.tags)
            .bind(entry.created_at)
            .bind(entry.last_accessed)
            .bind(entry.access_count as i32)
            .bind(entry.importance as i16)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Memory(format!("failed to store memory: {e}")))?;
            Ok(())
        })
    }

    fn recall(
        &self,
        query: MemoryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>> {
        Box::pin(async move {
            // Build dynamic query with filters
            let mut sql = String::from(
                "SELECT id, agent, content, category, tags, created_at, last_accessed, access_count, importance FROM memories WHERE true",
            );
            let mut param_idx = 1u32;

            // We'll collect bind values as we build the query
            let mut text_filter: Option<String> = None;
            let mut category_filter: Option<String> = None;
            let mut tags_filter: Option<Vec<String>> = None;
            let mut agent_filter: Option<String> = None;

            if let Some(ref text) = query.text {
                sql.push_str(&format!(" AND content ILIKE '%' || ${param_idx} || '%'"));
                text_filter = Some(text.clone());
                param_idx += 1;
            }

            if let Some(ref category) = query.category {
                sql.push_str(&format!(" AND category = ${param_idx}"));
                category_filter = Some(category.clone());
                param_idx += 1;
            }

            if !query.tags.is_empty() {
                sql.push_str(&format!(" AND tags && ${param_idx}"));
                tags_filter = Some(query.tags.clone());
                param_idx += 1;
            }

            if let Some(ref agent) = query.agent {
                sql.push_str(&format!(" AND agent = ${param_idx}"));
                agent_filter = Some(agent.clone());
            }

            sql.push_str(" ORDER BY created_at DESC");

            // Build and bind
            let mut q = sqlx::query(&sql);

            if let Some(ref text) = text_filter {
                q = q.bind(text);
            }
            if let Some(ref category) = category_filter {
                q = q.bind(category);
            }
            if let Some(ref tags) = tags_filter {
                q = q.bind(tags);
            }
            if let Some(ref agent) = agent_filter {
                q = q.bind(agent);
            }

            let rows = q
                .fetch_all(&self.pool)
                .await
                .map_err(|e| Error::Memory(format!("failed to recall memories: {e}")))?;

            let has_text_query = query.text.is_some();
            let mut entries: Vec<MemoryEntry> = rows
                .into_iter()
                .map(|row| {
                    let r = MemoryRow {
                        id: row.get("id"),
                        agent: row.get("agent"),
                        content: row.get("content"),
                        category: row.get("category"),
                        tags: row.get("tags"),
                        created_at: row.get("created_at"),
                        last_accessed: row.get("last_accessed"),
                        access_count: row.get("access_count"),
                        importance: row.get("importance"),
                    };
                    MemoryEntry::from(r)
                })
                .collect();

            // Apply composite scoring for consistency with InMemoryStore
            let now = Utc::now();
            let weights = &self.scoring_weights;
            entries.sort_by(|a, b| {
                let relevance_a = if has_text_query { 1.0 } else { 0.0 };
                let relevance_b = if has_text_query { 1.0 } else { 0.0 };
                let score_a =
                    composite_score(weights, a.created_at, now, a.importance, relevance_a);
                let score_b =
                    composite_score(weights, b.created_at, now, b.importance, relevance_b);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Apply limit after scoring (not in SQL) so high-importance
            // old entries aren't excluded before scoring.
            if query.limit > 0 {
                entries.truncate(query.limit);
            }

            // Update access counts for returned entries
            if !entries.is_empty() {
                let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
                sqlx::query(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = now() WHERE id = ANY($1)",
                )
                .bind(&ids)
                .execute(&self.pool)
                .await
                .map_err(|e| Error::Memory(format!("failed to update access counts: {e}")))?;

                // Update local entries to reflect the new access count
                for entry in &mut entries {
                    entry.access_count += 1;
                    entry.last_accessed = Utc::now();
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
        let id = id.to_string();
        Box::pin(async move {
            let result = sqlx::query(
                "UPDATE memories SET content = $2, last_accessed = now() WHERE id = $1",
            )
            .bind(&id)
            .bind(&content)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Memory(format!("failed to update memory: {e}")))?;

            if result.rows_affected() == 0 {
                return Err(Error::Memory(format!("memory not found: {id}")));
            }
            Ok(())
        })
    }

    fn forget(&self, id: &str) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>> {
        let id = id.to_string();
        Box::pin(async move {
            let result = sqlx::query("DELETE FROM memories WHERE id = $1")
                .bind(&id)
                .execute(&self.pool)
                .await
                .map_err(|e| Error::Memory(format!("failed to delete memory: {e}")))?;

            Ok(result.rows_affected() > 0)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_row_to_entry() {
        let row = MemoryRow {
            id: "m1".into(),
            agent: "agent1".into(),
            content: "test content".into(),
            category: "fact".into(),
            tags: vec!["tag1".into()],
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 3,
            importance: 7,
        };
        let entry = MemoryEntry::from(row);
        assert_eq!(entry.id, "m1");
        assert_eq!(entry.agent, "agent1");
        assert_eq!(entry.access_count, 3);
        assert_eq!(entry.importance, 7);
    }

    #[test]
    fn memory_row_clamps_importance() {
        let row = MemoryRow {
            id: "m1".into(),
            agent: "a".into(),
            content: "t".into(),
            category: "fact".into(),
            tags: vec![],
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            importance: 15, // Out of range
        };
        let entry = MemoryEntry::from(row);
        assert_eq!(entry.importance, 10); // clamped
    }

    #[test]
    fn memory_row_clamps_importance_low() {
        let row = MemoryRow {
            id: "m1".into(),
            agent: "a".into(),
            content: "t".into(),
            category: "fact".into(),
            tags: vec![],
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            importance: 0, // Out of range (but i16 allows it)
        };
        let entry = MemoryEntry::from(row);
        assert_eq!(entry.importance, 1); // clamped
    }

    #[test]
    fn postgres_memory_store_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PostgresMemoryStore>();
    }
}
