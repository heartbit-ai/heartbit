use std::future::Future;
use std::pin::Pin;

use chrono::{DateTime, Utc};
use sqlx::{FromRow, PgPool, Row};

use crate::error::Error;

use super::bm25;
use super::hybrid;
use super::scoring::{STRENGTH_DECAY_RATE, ScoringWeights, composite_score, effective_strength};
use super::{Memory, MemoryEntry, MemoryQuery};

use super::MemoryType;

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
    memory_type: String,
    keywords: Vec<String>,
    summary: Option<String>,
    strength: f64,
    related_ids: Vec<String>,
    source_ids: Vec<String>,
}

fn memory_type_from_str(s: &str) -> MemoryType {
    match s {
        "semantic" => MemoryType::Semantic,
        "reflection" => MemoryType::Reflection,
        _ => MemoryType::Episodic,
    }
}

fn memory_type_to_str(mt: MemoryType) -> &'static str {
    match mt {
        MemoryType::Episodic => "episodic",
        MemoryType::Semantic => "semantic",
        MemoryType::Reflection => "reflection",
    }
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
            memory_type: memory_type_from_str(&row.memory_type),
            keywords: row.keywords,
            summary: row.summary,
            strength: row.strength,
            related_ids: row.related_ids,
            source_ids: row.source_ids,
            embedding: None, // loaded separately when pgvector is available
        }
    }
}

/// PostgreSQL-backed memory store for durable agent memory persistence.
///
/// Uses `sqlx::query_as()` (runtime queries, no compile-time macros).
/// Recall filtering uses SQL WHERE clauses with `ILIKE` for text search.
/// All matching rows are fetched, then scored in Rust using BM25 relevance
/// (with keyword boost), Ebbinghaus effective strength, and composite scoring
/// (recency + importance + relevance + strength) for consistency with
/// `InMemoryStore`.
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

    /// Query pgvector for the top-N entries ranked by cosine similarity
    /// to the query embedding. Returns `(id, cosine_similarity)` pairs.
    /// Entries without embeddings are excluded.
    async fn vector_ranked_ids(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(String, f64)>, Error> {
        let emb_vec = pgvector::Vector::from(query_embedding.to_vec());
        let rows = sqlx::query(
            "SELECT id, 1 - (embedding <=> $1) AS cosine_similarity FROM memories WHERE embedding IS NOT NULL ORDER BY embedding <=> $1 LIMIT $2",
        )
        .bind(emb_vec)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Error::Memory(format!("vector search failed: {e}")))?;

        Ok(rows
            .iter()
            .map(|row| {
                let id: String = row.get("id");
                let sim: f64 = row.get("cosine_similarity");
                (id, sim)
            })
            .collect())
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
                importance      SMALLINT NOT NULL DEFAULT 5,
                memory_type     TEXT NOT NULL DEFAULT 'episodic',
                keywords        TEXT[] NOT NULL DEFAULT '{}',
                summary         TEXT,
                strength        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                related_ids     TEXT[] NOT NULL DEFAULT '{}',
                source_ids      TEXT[] NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent);
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength);

            -- Add new columns to existing tables (safe to run multiple times)
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS memory_type TEXT NOT NULL DEFAULT 'episodic';
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS keywords TEXT[] NOT NULL DEFAULT '{}';
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS summary TEXT;
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS strength DOUBLE PRECISION NOT NULL DEFAULT 1.0;
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS related_ids TEXT[] NOT NULL DEFAULT '{}';
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_ids TEXT[] NOT NULL DEFAULT '{}';

            -- pgvector extension and embedding column for hybrid retrieval
            CREATE EXTENSION IF NOT EXISTS vector;
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding vector(1536);
            CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops);
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
            let embedding = entry
                .embedding
                .as_ref()
                .map(|v| pgvector::Vector::from(v.clone()));
            sqlx::query(
                r#"
                INSERT INTO memories (id, agent, content, category, tags, created_at, last_accessed, access_count, importance,
                    memory_type, keywords, summary, strength, related_ids, source_ids, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    tags = EXCLUDED.tags,
                    importance = EXCLUDED.importance,
                    memory_type = EXCLUDED.memory_type,
                    keywords = EXCLUDED.keywords,
                    summary = EXCLUDED.summary,
                    strength = EXCLUDED.strength,
                    related_ids = EXCLUDED.related_ids,
                    source_ids = EXCLUDED.source_ids,
                    embedding = EXCLUDED.embedding
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
            .bind(memory_type_to_str(entry.memory_type))
            .bind(&entry.keywords)
            .bind(&entry.summary)
            .bind(entry.strength)
            .bind(&entry.related_ids)
            .bind(&entry.source_ids)
            .bind(&embedding)
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
                "SELECT id, agent, content, category, tags, created_at, last_accessed, access_count, importance, memory_type, keywords, summary, strength, related_ids, source_ids FROM memories WHERE true",
            );
            let mut param_idx = 1u32;

            // We'll collect bind values as we build the query
            let mut text_filter: Option<Vec<String>> = None;
            let mut category_filter: Option<String> = None;
            let mut tags_filter: Option<Vec<String>> = None;
            let mut agent_filter: Option<String> = None;
            let mut memory_type_filter: Option<String> = None;
            let mut min_strength_filter: Option<f64> = None;

            // Split text query into unique tokens for per-token matching.
            // SQL filters with OR (any token present), Rust-side scoring
            // computes granular relevance = matched_tokens / total_tokens.
            let query_tokens: Vec<String> = query
                .text
                .as_deref()
                .map(|t| {
                    let mut seen = std::collections::HashSet::new();
                    t.to_lowercase()
                        .split_whitespace()
                        .filter(|tok| seen.insert(tok.to_string()))
                        .map(String::from)
                        .collect()
                })
                .unwrap_or_default();

            if !query_tokens.is_empty() {
                let conditions: Vec<String> = query_tokens
                    .iter()
                    .map(|_| {
                        let cond = format!("content ILIKE '%' || ${param_idx} || '%'");
                        param_idx += 1;
                        cond
                    })
                    .collect();
                sql.push_str(&format!(" AND ({})", conditions.join(" OR ")));
                text_filter = Some(query_tokens.clone());
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
                param_idx += 1;
            } else if let Some(ref prefix) = query.agent_prefix {
                sql.push_str(&format!(" AND agent LIKE ${param_idx}"));
                agent_filter = Some(format!("{prefix}%"));
                param_idx += 1;
            }

            if let Some(ref mt) = query.memory_type {
                sql.push_str(&format!(" AND memory_type = ${param_idx}"));
                memory_type_filter = Some(memory_type_to_str(*mt).to_string());
                param_idx += 1;
            }

            // Note: min_strength is filtered in Rust after fetch using effective_strength()
            // with Ebbinghaus decay, matching InMemoryStore behavior. We still apply a
            // loose SQL filter (raw strength >= min_s) to reduce row count.
            if let Some(min_s) = query.min_strength {
                sql.push_str(&format!(" AND strength >= ${param_idx}"));
                min_strength_filter = Some(min_s);
                param_idx += 1;
            }

            // param_idx is intentionally incremented after each filter to keep
            // placeholders consistent if filters are reordered or new ones added.
            let _ = param_idx;

            sql.push_str(" ORDER BY created_at DESC");

            // Build and bind
            let mut q = sqlx::query(&sql);

            if let Some(ref tokens) = text_filter {
                for token in tokens {
                    q = q.bind(token);
                }
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
            if let Some(ref mt) = memory_type_filter {
                q = q.bind(mt);
            }
            if let Some(min_s) = min_strength_filter {
                q = q.bind(min_s);
            }

            let rows = q
                .fetch_all(&self.pool)
                .await
                .map_err(|e| Error::Memory(format!("failed to recall memories: {e}")))?;

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
                        memory_type: row.get("memory_type"),
                        keywords: row.get("keywords"),
                        summary: row.get("summary"),
                        strength: row.get("strength"),
                        related_ids: row.get("related_ids"),
                        source_ids: row.get("source_ids"),
                    };
                    MemoryEntry::from(r)
                })
                .collect();

            // Apply effective_strength filter (Ebbinghaus decay) for min_strength,
            // matching InMemoryStore's approach. The SQL filter above uses raw strength
            // as a loose pre-filter; this applies the actual decayed value.
            let now = Utc::now();
            if let Some(min_s) = query.min_strength {
                entries.retain(|e| {
                    effective_strength(e.strength, e.last_accessed, now, STRENGTH_DECAY_RATE)
                        >= min_s
                });
            }

            // Apply BM25 scoring with keyword field boost, matching InMemoryStore.
            let weights = &self.scoring_weights;
            let avgdl = if entries.is_empty() {
                1.0
            } else {
                let total_words: usize = entries
                    .iter()
                    .map(|e| e.content.split_whitespace().count())
                    .sum();
                (total_words as f64 / entries.len() as f64).max(1.0)
            };

            let bm25_map: std::collections::HashMap<String, f64> = entries
                .iter()
                .map(|e| {
                    let score = bm25::bm25_score(
                        &e.content,
                        &e.keywords,
                        &query_tokens,
                        avgdl,
                        bm25::DEFAULT_K1,
                        bm25::DEFAULT_B,
                    );
                    (e.id.clone(), score)
                })
                .collect();

            // Compute relevance: hybrid (BM25 + pgvector cosine via RRF) when
            // query_embedding is available, otherwise pure BM25.
            let relevance_map: std::collections::HashMap<String, f64> =
                if let Some(ref q_emb) = query.query_embedding {
                    // BM25 ranked list (descending)
                    let mut bm25_ranked: Vec<(&str, f64)> =
                        bm25_map.iter().map(|(id, &s)| (id.as_str(), s)).collect();
                    bm25_ranked
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    // Query pgvector for vector-ranked entries (top 40).
                    // Uses cosine distance operator <=>. Entries without embeddings
                    // are excluded. Falls back to pure BM25 on error.
                    let vector_ranked_owned: Vec<(String, f64)> =
                        match self.vector_ranked_ids(q_emb, 40).await {
                            Ok(ranked) => ranked,
                            Err(e) => {
                                tracing::warn!("pgvector query failed, falling back to BM25: {e}");
                                vec![]
                            }
                        };
                    let vector_ranked: Vec<(&str, f64)> = vector_ranked_owned
                        .iter()
                        .map(|(id, score)| (id.as_str(), *score))
                        .collect();

                    if vector_ranked.is_empty() {
                        let max_bm25 = bm25_map
                            .values()
                            .copied()
                            .fold(f64::NEG_INFINITY, f64::max)
                            .max(1.0);
                        bm25_map
                            .iter()
                            .map(|(id, &s)| (id.clone(), s / max_bm25))
                            .collect()
                    } else {
                        let fused = hybrid::rrf_fuse(&bm25_ranked, &vector_ranked, 50);
                        let max_fused = fused
                            .iter()
                            .map(|(_, s)| *s)
                            .fold(f64::NEG_INFINITY, f64::max)
                            .max(f64::EPSILON);
                        fused
                            .into_iter()
                            .map(|(id, s)| (id, s / max_fused))
                            .collect()
                    }
                } else {
                    let max_bm25 = bm25_map
                        .values()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max)
                        .max(1.0);
                    bm25_map
                        .iter()
                        .map(|(id, &s)| (id.clone(), s / max_bm25))
                        .collect()
                };

            entries.sort_by(|a, b| {
                let relevance_a = relevance_map.get(&a.id).copied().unwrap_or(0.0);
                let relevance_b = relevance_map.get(&b.id).copied().unwrap_or(0.0);
                let eff_a =
                    effective_strength(a.strength, a.last_accessed, now, STRENGTH_DECAY_RATE);
                let eff_b =
                    effective_strength(b.strength, b.last_accessed, now, STRENGTH_DECAY_RATE);
                let score_a =
                    composite_score(weights, a.created_at, now, a.importance, relevance_a, eff_a);
                let score_b =
                    composite_score(weights, b.created_at, now, b.importance, relevance_b, eff_b);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Apply limit after scoring (not in SQL) so high-importance
            // old entries aren't excluded before scoring.
            if query.limit > 0 {
                entries.truncate(query.limit);
            }

            // Graph expansion: follow related_ids one hop to surface linked entries.
            let top_ids: std::collections::HashSet<String> =
                entries.iter().map(|e| e.id.clone()).collect();
            let mut related_to_fetch = Vec::new();
            let mut seen_expanded = std::collections::HashSet::new();
            for entry in &entries {
                for related_id in &entry.related_ids {
                    if !top_ids.contains(related_id) && seen_expanded.insert(related_id.clone()) {
                        related_to_fetch.push(related_id.clone());
                    }
                }
            }

            if !related_to_fetch.is_empty() {
                let expanded_rows = sqlx::query(
                    "SELECT id, agent, content, category, tags, created_at, last_accessed, access_count, importance, memory_type, keywords, summary, strength, related_ids, source_ids FROM memories WHERE id = ANY($1)",
                )
                .bind(&related_to_fetch)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| Error::Memory(format!("failed to fetch related entries: {e}")))?;

                let min_s = query.min_strength.unwrap_or(0.0);
                let mut added = false;
                for row in expanded_rows {
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
                        memory_type: row.get("memory_type"),
                        keywords: row.get("keywords"),
                        summary: row.get("summary"),
                        strength: row.get("strength"),
                        related_ids: row.get("related_ids"),
                        source_ids: row.get("source_ids"),
                    };
                    let related_entry = MemoryEntry::from(r);
                    let eff = effective_strength(
                        related_entry.strength,
                        related_entry.last_accessed,
                        now,
                        STRENGTH_DECAY_RATE,
                    );
                    if eff >= min_s {
                        entries.push(related_entry);
                        added = true;
                    }
                }

                // Re-score and re-sort if we expanded
                if added {
                    let new_avgdl = if entries.is_empty() {
                        1.0
                    } else {
                        let total_words: usize = entries
                            .iter()
                            .map(|e| e.content.split_whitespace().count())
                            .sum();
                        (total_words as f64 / entries.len() as f64).max(1.0)
                    };

                    let new_bm25_map: std::collections::HashMap<String, f64> = entries
                        .iter()
                        .filter(|e| !bm25_map.contains_key(&e.id))
                        .map(|e| {
                            let score = bm25::bm25_score(
                                &e.content,
                                &e.keywords,
                                &query_tokens,
                                new_avgdl,
                                bm25::DEFAULT_K1,
                                bm25::DEFAULT_B,
                            );
                            (e.id.clone(), score)
                        })
                        .collect();

                    let combined_max = bm25_map
                        .values()
                        .chain(new_bm25_map.values())
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max)
                        .max(1.0);

                    entries.sort_by(|a, b| {
                        let rel_a = bm25_map
                            .get(&a.id)
                            .or_else(|| new_bm25_map.get(&a.id))
                            .copied()
                            .unwrap_or(0.0)
                            / combined_max;
                        let rel_b = bm25_map
                            .get(&b.id)
                            .or_else(|| new_bm25_map.get(&b.id))
                            .copied()
                            .unwrap_or(0.0)
                            / combined_max;
                        let eff_a = effective_strength(
                            a.strength,
                            a.last_accessed,
                            now,
                            STRENGTH_DECAY_RATE,
                        );
                        let eff_b = effective_strength(
                            b.strength,
                            b.last_accessed,
                            now,
                            STRENGTH_DECAY_RATE,
                        );
                        let score_a =
                            composite_score(weights, a.created_at, now, a.importance, rel_a, eff_a);
                        let score_b =
                            composite_score(weights, b.created_at, now, b.importance, rel_b, eff_b);
                        score_b
                            .partial_cmp(&score_a)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    if query.limit > 0 {
                        entries.truncate(query.limit);
                    }
                }
            }

            // Update access counts and reinforce strength for returned entries
            if !entries.is_empty() {
                let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
                sqlx::query(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = now(), strength = LEAST(strength + 0.2, 1.0) WHERE id = ANY($1)",
                )
                .bind(&ids)
                .execute(&self.pool)
                .await
                .map_err(|e| Error::Memory(format!("failed to update access counts: {e}")))?;

                // Update local entries to reflect the new values
                for entry in &mut entries {
                    entry.access_count += 1;
                    entry.last_accessed = Utc::now();
                    entry.strength = (entry.strength + 0.2).min(1.0);
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

    fn add_link(
        &self,
        id: &str,
        related_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        let id = id.to_string();
        let related_id = related_id.to_string();
        Box::pin(async move {
            // Add related_id to id's related_ids (if not already present)
            sqlx::query(
                "UPDATE memories SET related_ids = array_append(related_ids, $2) WHERE id = $1 AND NOT ($2 = ANY(related_ids))",
            )
            .bind(&id)
            .bind(&related_id)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Memory(format!("failed to add link {id} -> {related_id}: {e}")))?;

            // Add id to related_id's related_ids (bidirectional)
            sqlx::query(
                "UPDATE memories SET related_ids = array_append(related_ids, $2) WHERE id = $1 AND NOT ($2 = ANY(related_ids))",
            )
            .bind(&related_id)
            .bind(&id)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Memory(format!("failed to add link {related_id} -> {id}: {e}")))?;

            Ok(())
        })
    }

    fn prune(
        &self,
        min_strength: f64,
        min_age: chrono::Duration,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async move {
            // Fetch candidates: entries with raw strength below threshold and old enough.
            // Then filter in Rust using effective_strength (Ebbinghaus decay).
            let min_age_secs = min_age.num_seconds().max(0);
            let rows = sqlx::query(
                "SELECT id, strength, last_accessed, created_at FROM memories WHERE strength < $1 AND created_at < now() - make_interval(secs => $2)",
            )
            .bind(min_strength)
            .bind(min_age_secs as f64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::Memory(format!("failed to query for pruning: {e}")))?;

            let now = Utc::now();
            let ids_to_remove: Vec<String> = rows
                .iter()
                .filter(|row| {
                    let strength: f64 = row.get("strength");
                    let last_accessed: DateTime<Utc> = row.get("last_accessed");
                    let eff = effective_strength(strength, last_accessed, now, STRENGTH_DECAY_RATE);
                    eff < min_strength
                })
                .map(|row| row.get::<String, _>("id"))
                .collect();

            if ids_to_remove.is_empty() {
                return Ok(0);
            }

            let result = sqlx::query("DELETE FROM memories WHERE id = ANY($1)")
                .bind(&ids_to_remove)
                .execute(&self.pool)
                .await
                .map_err(|e| Error::Memory(format!("failed to prune memories: {e}")))?;

            Ok(result.rows_affected() as usize)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(id: &str, importance: i16) -> MemoryRow {
        MemoryRow {
            id: id.into(),
            agent: "a".into(),
            content: "test".into(),
            category: "fact".into(),
            tags: vec![],
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            importance,
            memory_type: "episodic".into(),
            keywords: vec![],
            summary: None,
            strength: 1.0,
            related_ids: vec![],
            source_ids: vec![],
        }
    }

    #[test]
    fn memory_row_to_entry() {
        let mut row = make_row("m1", 7);
        row.agent = "agent1".into();
        row.content = "test content".into();
        row.tags = vec!["tag1".into()];
        row.access_count = 3;
        let entry = MemoryEntry::from(row);
        assert_eq!(entry.id, "m1");
        assert_eq!(entry.agent, "agent1");
        assert_eq!(entry.access_count, 3);
        assert_eq!(entry.importance, 7);
        assert_eq!(entry.memory_type, MemoryType::Episodic);
        assert!((entry.strength - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn memory_row_maps_memory_type() {
        let mut row = make_row("m1", 5);
        row.memory_type = "semantic".into();
        row.keywords = vec!["rust".into()];
        row.summary = Some("A summary".into());
        row.strength = 0.7;
        row.related_ids = vec!["m2".into()];
        row.source_ids = vec!["m0".into()];

        let entry = MemoryEntry::from(row);
        assert_eq!(entry.memory_type, MemoryType::Semantic);
        assert_eq!(entry.keywords, vec!["rust"]);
        assert_eq!(entry.summary.as_deref(), Some("A summary"));
        assert!((entry.strength - 0.7).abs() < f64::EPSILON);
        assert_eq!(entry.related_ids, vec!["m2"]);
        assert_eq!(entry.source_ids, vec!["m0"]);
    }

    #[test]
    fn memory_row_reflection_type() {
        let mut row = make_row("m1", 5);
        row.memory_type = "reflection".into();
        let entry = MemoryEntry::from(row);
        assert_eq!(entry.memory_type, MemoryType::Reflection);
    }

    #[test]
    fn memory_row_unknown_type_defaults_to_episodic() {
        let mut row = make_row("m1", 5);
        row.memory_type = "unknown".into();
        let entry = MemoryEntry::from(row);
        assert_eq!(entry.memory_type, MemoryType::Episodic);
    }

    #[test]
    fn memory_row_clamps_importance() {
        let row = make_row("m1", 15);
        let entry = MemoryEntry::from(row);
        assert_eq!(entry.importance, 10); // clamped
    }

    #[test]
    fn memory_row_clamps_importance_low() {
        let row = make_row("m1", 0);
        let entry = MemoryEntry::from(row);
        assert_eq!(entry.importance, 1); // clamped
    }

    #[test]
    fn memory_type_str_roundtrip() {
        for mt in [
            MemoryType::Episodic,
            MemoryType::Semantic,
            MemoryType::Reflection,
        ] {
            let s = memory_type_to_str(mt);
            let parsed = memory_type_from_str(s);
            assert_eq!(parsed, mt);
        }
    }

    #[test]
    fn postgres_memory_store_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PostgresMemoryStore>();
    }
}
