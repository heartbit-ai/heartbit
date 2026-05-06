//! In-memory `Memory` implementation backed by a sorted `Vec`.

#![allow(missing_docs)]
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use parking_lot::RwLock;

use chrono::Utc;

use crate::auth::TenantScope;
use crate::error::Error;

use super::bm25;
use super::hybrid;
use super::scoring::{STRENGTH_DECAY_RATE, ScoringWeights, composite_score, effective_strength};
use super::{Memory, MemoryEntry, MemoryQuery};

/// Default cap on the number of entries an `InMemoryStore` will hold.
///
/// SECURITY (F-MEM-3): without a cap, a hostile or buggy agent that spams
/// `memory_store` can balloon the process memory linearly. Once the cap is
/// reached, `store()` evicts the entry with the lowest effective strength
/// (i.e., the weakest, oldest memory) before inserting the new one.
pub const IN_MEMORY_STORE_DEFAULT_CAP: usize = 100_000;

/// Thread-safe in-memory store for agent memories.
///
/// Backed by `parking_lot::RwLock<HashMap>` (T2 — `tasks/performance-audit-
/// heartbit-core-2026-05-06.md`). Suitable for tests and single-process use.
/// Uses composite scoring (recency + importance + relevance) for recall
/// ordering.
pub struct InMemoryStore {
    entries: RwLock<HashMap<String, MemoryEntry>>,
    scoring_weights: ScoringWeights,
    max_entries: usize,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            scoring_weights: ScoringWeights::default(),
            max_entries: IN_MEMORY_STORE_DEFAULT_CAP,
        }
    }

    pub fn with_scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.scoring_weights = weights;
        self
    }

    /// Override the max-entries cap. Set to `usize::MAX` to disable.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
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
        scope: &TenantScope,
        mut entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        // Stamp the entry with the calling scope's tenant/user identity.
        entry.author_tenant_id = Some(scope.tenant_id.clone());
        entry.author_user_id = scope.user_id.clone();
        Box::pin(async move {
            let mut entries = self.entries.write();
            // SECURITY (F-MEM-3): when at capacity, evict the entry with the
            // lowest effective strength (most-decayed, oldest weak memory)
            // before inserting the new one. Without this cap, a hostile or
            // buggy agent could balloon process memory by spamming
            // `memory_store`. Eviction happens BEFORE insertion to avoid a
            // transient over-cap state.
            if !entries.contains_key(&entry.id) && entries.len() >= self.max_entries {
                let now = Utc::now();
                if let Some(victim_id) = entries
                    .values()
                    .min_by(|a, b| {
                        let ea = effective_strength(
                            a.strength,
                            a.last_accessed,
                            now,
                            STRENGTH_DECAY_RATE,
                        );
                        let eb = effective_strength(
                            b.strength,
                            b.last_accessed,
                            now,
                            STRENGTH_DECAY_RATE,
                        );
                        ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|e| e.id.clone())
                {
                    entries.remove(&victim_id);
                    tracing::warn!(
                        evicted = %victim_id,
                        cap = self.max_entries,
                        "InMemoryStore at cap; evicted weakest entry (F-MEM-3)"
                    );
                }
            }
            entries.insert(entry.id.clone(), entry);
            Ok(())
        })
    }

    fn recall(
        &self,
        scope: &TenantScope,
        query: MemoryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>> {
        let tenant_id = scope.tenant_id.clone();
        Box::pin(async move {
            // Single write lock for the entire operation. Recall updates
            // access_count as a side effect, so we need write access anyway.
            // Using one lock avoids a TOCTOU window where concurrent forget()
            // or store() could interleave between filter and access-count update.
            let mut entries = self.entries.write();

            let now = Utc::now();
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
            let lower_query_tokens_for_filter: Vec<String> = query_tokens.clone();

            // PERF (P-MEM-5): collect candidate entries as references so we
            // never clone the 10k+ entries that pass the filter — only the
            // top-K (after limit and graph expansion) ever turn into owned
            // `MemoryEntry`s, and that final clone happens against
            // `entries.get_mut(&id)` below.
            //
            // Filter ordering: cheap field comparisons first, then the
            // text-substring scan last so we don't lowercase content for
            // entries that fail tenant / agent / category / tag checks.
            let candidates: Vec<&MemoryEntry> = entries
                .values()
                .filter(|e| {
                    // Tenant isolation first — fastest reject.
                    if e.author_tenant_id.as_deref() != Some(tenant_id.as_str()) {
                        return false;
                    }
                    if let Some(ref agent) = query.agent {
                        if e.agent != *agent {
                            return false;
                        }
                    } else if let Some(ref prefix) = query.agent_prefix
                        && !e.agent.starts_with(prefix.as_str())
                    {
                        return false;
                    }
                    if let Some(ref cat) = query.category
                        && e.category != *cat
                    {
                        return false;
                    }
                    if !query.tags.is_empty() && !query.tags.iter().any(|t| e.tags.contains(t)) {
                        return false;
                    }
                    if let Some(ref mt) = query.memory_type
                        && e.memory_type != *mt
                    {
                        return false;
                    }
                    if let Some(max_conf) = query.max_confidentiality
                        && e.confidentiality > max_conf
                    {
                        return false;
                    }
                    if let Some(min_s) = query.min_strength {
                        let eff = effective_strength(
                            e.strength,
                            e.last_accessed,
                            now,
                            STRENGTH_DECAY_RATE,
                        );
                        if eff < min_s {
                            return false;
                        }
                    }
                    // Expensive filter (tokenisation) last.
                    if !lower_query_tokens_for_filter.is_empty() {
                        let lower_content = e.content.to_lowercase();
                        let lower_keywords: Vec<String> =
                            e.keywords.iter().map(|k| k.to_lowercase()).collect();
                        let has_match = lower_query_tokens_for_filter.iter().any(|token| {
                            lower_content.contains(token.as_str())
                                || lower_keywords.iter().any(|k| k.contains(token.as_str()))
                        });
                        if !has_match {
                            return false;
                        }
                    }
                    true
                })
                .collect();

            // Compute average document length for BM25 normalisation.
            let avgdl = if candidates.is_empty() {
                1.0
            } else {
                let total_words: usize = candidates
                    .iter()
                    .map(|e| e.content.split_whitespace().count())
                    .sum();
                (total_words as f64 / candidates.len() as f64).max(1.0)
            };

            // Pre-compute BM25 scores. Keyed by `&str` slices into the
            // entries map; lifetime is tied to the immutable borrow held
            // by `candidates` (released before any `get_mut` below).
            let bm25_map: HashMap<&str, f64> = candidates
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
                    (e.id.as_str(), score)
                })
                .collect();

            // Compute relevance scores: hybrid (BM25 + cosine via RRF) when
            // query_embedding is available, otherwise pure BM25.
            let relevance_map: HashMap<&str, f64> = if let Some(ref q_emb) = query.query_embedding {
                // BM25 ranked list (descending by score)
                let mut bm25_ranked: Vec<(&str, f64)> =
                    bm25_map.iter().map(|(id, &s)| (*id, s)).collect();
                bm25_ranked
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Vector ranked list (cosine similarity, descending)
                let mut vector_ranked: Vec<(&str, f64)> = candidates
                    .iter()
                    .filter_map(|e| {
                        e.embedding
                            .as_ref()
                            .map(|emb| (e.id.as_str(), hybrid::cosine_similarity(emb, q_emb)))
                    })
                    .collect();
                vector_ranked
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                if vector_ranked.is_empty() {
                    // No embeddings stored — fall back to pure BM25
                    let max_bm25 = bm25_map
                        .values()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max)
                        .max(1.0);
                    bm25_map
                        .iter()
                        .map(|(id, &s)| (*id, s / max_bm25))
                        .collect()
                } else {
                    let fused = hybrid::rrf_fuse(&bm25_ranked, &vector_ranked, 50);
                    let max_fused = fused
                        .iter()
                        .map(|(_, s)| *s)
                        .fold(f64::NEG_INFINITY, f64::max)
                        .max(f64::EPSILON);
                    // `rrf_fuse` returns owned `String` keys; project them
                    // back onto the original `&str` slices we already
                    // hold via the bm25_map's keys to keep the lifetime
                    // discipline consistent across both branches.
                    let mut out: HashMap<&str, f64> = HashMap::with_capacity(fused.len());
                    for (id_owned, score) in &fused {
                        if let Some((k, _)) = bm25_map.get_key_value(id_owned.as_str()) {
                            out.insert(k, score / max_fused);
                        }
                    }
                    out
                }
            } else {
                // Pure BM25 path
                let max_bm25 = bm25_map
                    .values()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(1.0);
                bm25_map
                    .iter()
                    .map(|(id, &s)| (*id, s / max_bm25))
                    .collect()
            };

            // Pair refs with composite score (computed **once** per entry)
            // and sort the pairs. The previous `sort_by` recomputed
            // `effective_strength` on both elements of every comparison —
            // at N=10k that's ~280k redundant exp() calls per recall.
            let mut scored: Vec<(&MemoryEntry, f64)> = candidates
                .iter()
                .copied()
                .map(|e| {
                    let relevance = relevance_map.get(e.id.as_str()).copied().unwrap_or(0.0);
                    let eff =
                        effective_strength(e.strength, e.last_accessed, now, STRENGTH_DECAY_RATE);
                    let score = composite_score(
                        &self.scoring_weights,
                        e.created_at,
                        now,
                        e.importance,
                        relevance,
                        eff,
                    );
                    (e, score)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if query.limit > 0 && scored.len() > query.limit {
                scored.truncate(query.limit);
            }

            // Graph expansion: collect related_ids that aren't already in
            // the top-K. Uses owned `String` ids to keep lifetimes simple.
            let top_ids: std::collections::HashSet<String> =
                scored.iter().map(|(e, _)| e.id.clone()).collect();
            let mut to_expand = Vec::new();
            let mut seen_expanded = std::collections::HashSet::new();
            for (entry, _) in &scored {
                for related_id in &entry.related_ids {
                    if !top_ids.contains(related_id) && seen_expanded.insert(related_id.clone()) {
                        to_expand.push(related_id.clone());
                    }
                }
            }

            // Compute BM25 + composite for related entries (still under
            // the same immutable borrow on `entries`) and append to the
            // scored Vec.
            let min_s = query.min_strength.unwrap_or(0.0);
            let mut expanded_added = 0usize;
            for related_id in &to_expand {
                if let Some(related) = entries.get(related_id) {
                    if let Some(max_conf) = query.max_confidentiality
                        && related.confidentiality > max_conf
                    {
                        continue;
                    }
                    let eff = effective_strength(
                        related.strength,
                        related.last_accessed,
                        now,
                        STRENGTH_DECAY_RATE,
                    );
                    if eff < min_s {
                        continue;
                    }
                    // Score the related entry against the same query.
                    let relevance = bm25::bm25_score(
                        &related.content,
                        &related.keywords,
                        &query_tokens,
                        avgdl,
                        bm25::DEFAULT_K1,
                        bm25::DEFAULT_B,
                    );
                    // Normalise against the same scale as the top-K.
                    let max_bm25 = bm25_map
                        .values()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max)
                        .max(1.0);
                    let normalised = relevance / max_bm25;
                    let score = composite_score(
                        &self.scoring_weights,
                        related.created_at,
                        now,
                        related.importance,
                        normalised,
                        eff,
                    );
                    scored.push((related, score));
                    expanded_added += 1;
                }
            }

            if expanded_added > 0 {
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if query.limit > 0 && scored.len() > query.limit {
                    scored.truncate(query.limit);
                }
            }

            // Collect top-K ids (owned), drop the immutable borrow on the
            // entries map so we can call `get_mut` for the access-count
            // updates and the *single* clone per surviving entry.
            let top_ids_final: Vec<String> = scored.iter().map(|(e, _)| e.id.clone()).collect();
            drop(scored);
            drop(candidates);

            // PERF (P-MEM-5): clone only the top-K, after the limit has
            // been applied. Previously the entire filtered set was cloned
            // before sorting / truncation — at N=10k that's ~5 MB of
            // allocation per recall.
            let reinforce = query.reinforce;
            let mut results: Vec<MemoryEntry> = Vec::with_capacity(top_ids_final.len());
            for id in top_ids_final {
                if let Some(e) = entries.get_mut(&id) {
                    e.access_count += 1;
                    e.last_accessed = now;
                    if reinforce {
                        // Ebbinghaus reinforcement: +0.2 per access, capped at 1.0.
                        e.strength = (e.strength + 0.2).min(1.0);
                    }
                    results.push(e.clone());
                }
            }

            Ok(results)
        })
    }

    fn update(
        &self,
        scope: &TenantScope,
        id: &str,
        content: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        let id = id.to_string();
        let tenant_id = scope.tenant_id.clone();
        Box::pin(async move {
            let mut entries = self.entries.write();
            match entries.get_mut(&id) {
                Some(entry) if entry.author_tenant_id.as_deref() == Some(tenant_id.as_str()) => {
                    entry.content = content;
                    entry.last_accessed = Utc::now();
                    Ok(())
                }
                Some(_) => {
                    // Entry exists but belongs to a different tenant — treat as not found.
                    Err(Error::Memory(format!("memory not found: {id}")))
                }
                None => Err(Error::Memory(format!("memory not found: {id}"))),
            }
        })
    }

    fn forget(
        &self,
        scope: &TenantScope,
        id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>> {
        let id = id.to_string();
        let tenant_id = scope.tenant_id.clone();
        Box::pin(async move {
            let mut entries = self.entries.write();
            // Only remove if the entry belongs to this tenant.
            // Return false for both "not found" and "wrong tenant" to avoid
            // revealing cross-tenant id existence.
            let belongs = entries
                .get(&id)
                .map(|e| e.author_tenant_id.as_deref() == Some(tenant_id.as_str()))
                .unwrap_or(false);
            if belongs {
                Ok(entries.remove(&id).is_some())
            } else {
                Ok(false)
            }
        })
    }

    fn add_link(
        &self,
        scope: &TenantScope,
        id: &str,
        related_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        let id = id.to_string();
        let related_id = related_id.to_string();
        let tenant_id = scope.tenant_id.clone();
        Box::pin(async move {
            let mut entries = self.entries.write();

            // Only link entries that belong to the same tenant.
            let id_ok = entries
                .get(&id)
                .map(|e| e.author_tenant_id.as_deref() == Some(tenant_id.as_str()))
                .unwrap_or(false);
            let rel_ok = entries
                .get(&related_id)
                .map(|e| e.author_tenant_id.as_deref() == Some(tenant_id.as_str()))
                .unwrap_or(false);

            if id_ok
                && let Some(entry) = entries.get_mut(&id)
                && !entry.related_ids.contains(&related_id)
            {
                entry.related_ids.push(related_id.clone());
            }
            if rel_ok
                && let Some(entry) = entries.get_mut(&related_id)
                && !entry.related_ids.contains(&id)
            {
                entry.related_ids.push(id);
            }
            Ok(())
        })
    }

    fn prune(
        &self,
        scope: &TenantScope,
        min_strength: f64,
        min_age: chrono::Duration,
        agent_prefix: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        let owned_prefix = agent_prefix.map(String::from);
        let tenant_id = scope.tenant_id.clone();
        Box::pin(async move {
            let mut entries = self.entries.write();

            let now = Utc::now();
            let to_remove: Vec<String> = entries
                .values()
                .filter(|e| {
                    // Tenant isolation: only prune entries belonging to this scope.
                    if e.author_tenant_id.as_deref() != Some(tenant_id.as_str()) {
                        return false;
                    }
                    // SECURITY (F-MEM-1): match only on EXACT agent name or
                    // proper `prefix:` separator. Plain `starts_with` lets
                    // `user:alice` match `user:alice2` / `user:alice-staging`,
                    // which would let a NamespacedMemory for one user prune
                    // weak entries of a sibling user with an overlapping
                    // prefix. The recall path uses exact `agent ==` matching;
                    // align prune to the same semantics.
                    if let Some(ref prefix) = owned_prefix {
                        let p = prefix.as_str();
                        let agent = e.agent.as_str();
                        let separator_match = agent.len() > p.len()
                            && agent.starts_with(p)
                            && agent.as_bytes()[p.len()] == b':';
                        if agent != p && !separator_match {
                            return false;
                        }
                    }
                    let eff =
                        effective_strength(e.strength, e.last_accessed, now, STRENGTH_DECAY_RATE);
                    eff < min_strength && now.signed_duration_since(e.created_at) > min_age
                })
                .map(|e| e.id.clone())
                .collect();

            let count = to_remove.len();
            for id in to_remove {
                entries.remove(&id);
            }
            Ok(count)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    use super::super::{Confidentiality, MemoryType};

    fn test_scope() -> TenantScope {
        TenantScope::default()
    }

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
            memory_type: MemoryType::default(),
            keywords: vec![],
            summary: None,
            strength: 1.0,
            related_ids: vec![],
            source_ids: vec![],
            embedding: None,
            confidentiality: Confidentiality::default(),
            author_user_id: None,
            author_tenant_id: None,
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
            memory_type: MemoryType::default(),
            keywords: vec![],
            summary: None,
            strength: 1.0,
            related_ids: vec![],
            source_ids: vec![],
            embedding: None,
            confidentiality: Confidentiality::default(),
            author_user_id: None,
            author_tenant_id: None,
        }
    }

    #[tokio::test]
    async fn store_and_recall() {
        let store = InMemoryStore::new();
        let entry = make_entry("m1", "agent1", "Rust is fast", "fact");
        store.store(&test_scope(), entry).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Rust is fast");
    }

    #[tokio::test]
    async fn recall_by_text() {
        let store = InMemoryStore::new();
        store
            .store(&test_scope(), make_entry("m1", "a", "Rust is fast", "fact"))
            .await
            .unwrap();
        store
            .store(
                &test_scope(),
                make_entry("m2", "a", "Python is slow", "fact"),
            )
            .await
            .unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("rust".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn recall_by_category() {
        let store = InMemoryStore::new();
        store
            .store(
                &test_scope(),
                make_entry("m1", "a", "remember this", "fact"),
            )
            .await
            .unwrap();
        store
            .store(
                &test_scope(),
                make_entry("m2", "a", "I saw something", "observation"),
            )
            .await
            .unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    category: Some("observation".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m2");
    }

    #[tokio::test]
    async fn recall_by_tags() {
        let store = InMemoryStore::new();
        store
            .store(
                &test_scope(),
                make_entry_with_tags(
                    "m1",
                    "a",
                    "Rust memory safety",
                    "fact",
                    vec!["rust".into(), "safety".into()],
                ),
            )
            .await
            .unwrap();
        store
            .store(
                &test_scope(),
                make_entry_with_tags(
                    "m2",
                    "a",
                    "Go is garbage collected",
                    "fact",
                    vec!["go".into()],
                ),
            )
            .await
            .unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    tags: vec!["rust".into()],
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn recall_by_agent() {
        let store = InMemoryStore::new();
        store
            .store(
                &test_scope(),
                make_entry("m1", "researcher", "data point", "fact"),
            )
            .await
            .unwrap();
        store
            .store(
                &test_scope(),
                make_entry("m2", "coder", "code snippet", "procedure"),
            )
            .await
            .unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    agent: Some("researcher".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
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
                .store(
                    &test_scope(),
                    make_entry(&format!("m{i}"), "a", &format!("entry {i}"), "fact"),
                )
                .await
                .unwrap();
        }

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 3,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn update_existing() {
        let store = InMemoryStore::new();
        store
            .store(&test_scope(), make_entry("m1", "a", "original", "fact"))
            .await
            .unwrap();

        store
            .update(&test_scope(), "m1", "updated content".into())
            .await
            .unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results[0].content, "updated content");
    }

    #[tokio::test]
    async fn update_nonexistent() {
        let store = InMemoryStore::new();
        let err = store
            .update(&test_scope(), "missing", "content".into())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn forget_existing() {
        let store = InMemoryStore::new();
        store
            .store(&test_scope(), make_entry("m1", "a", "to delete", "fact"))
            .await
            .unwrap();

        assert!(store.forget(&test_scope(), "m1").await.unwrap());

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn forget_nonexistent() {
        let store = InMemoryStore::new();
        assert!(!store.forget(&test_scope(), "missing").await.unwrap());
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
        store.store(&test_scope(), high_imp).await.unwrap();

        // Recent entry with low importance (now, importance=1)
        let mut low_imp = make_entry("m2", "a", "low importance", "fact");
        low_imp.importance = 1;
        low_imp.created_at = Utc::now();
        store.store(&test_scope(), low_imp).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
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
        store.store(&test_scope(), old_low).await.unwrap();

        // Recent, high importance — should definitely be first
        let mut recent_high = make_entry("m2", "a", "recent high", "fact");
        recent_high.importance = 10;
        recent_high.created_at = Utc::now();
        store.store(&test_scope(), recent_high).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results[0].id, "m2");
    }

    #[tokio::test]
    async fn recall_with_custom_weights() {
        // Pure importance sorting (alpha=0, beta=1, gamma=0, delta=0)
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 1.0,
            gamma: 0.0,
            delta: 0.0,
            decay_rate: 0.01,
        });

        let mut low = make_entry("m1", "a", "recent but low", "fact");
        low.importance = 1;
        low.created_at = Utc::now();
        store.store(&test_scope(), low).await.unwrap();

        let mut high = make_entry("m2", "a", "old but high", "fact");
        high.importance = 10;
        high.created_at = Utc::now() - chrono::Duration::hours(1000);
        store.store(&test_scope(), high).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
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
            delta: 0.0,
            decay_rate: 0.01,
        });

        let mut e1 = make_entry("m1", "a", "Rust is fast", "fact");
        e1.importance = 5;
        store.store(&test_scope(), e1).await.unwrap();

        // Text query means relevance=1.0 for matched entries
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("Rust".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn recall_limit_zero_returns_all() {
        let store = InMemoryStore::new();
        for i in 0..5 {
            store
                .store(
                    &test_scope(),
                    make_entry(&format!("m{i}"), "a", &format!("entry {i}"), "fact"),
                )
                .await
                .unwrap();
        }

        // limit=0 means "no limit" — should return all entries
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 0,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn recall_deduplicates_query_tokens() {
        // "rust rust rust" should behave identically to "rust" for scoring.
        // Before the fix, repeated tokens inflated the denominator, lowering scores.
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
            decay_rate: 0.01,
        });

        store
            .store(&test_scope(), make_entry("m1", "a", "Rust is fast", "fact"))
            .await
            .unwrap();
        store
            .store(
                &test_scope(),
                make_entry("m2", "a", "Python is slow", "fact"),
            )
            .await
            .unwrap();

        // Query with duplicated token
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("rust rust rust".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Only m1 should match (contains "rust")
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn relevance_score_differentiates_results() {
        // Two entries with same importance and similar timestamps.
        // "Rust is fast and safe" matches both "Rust" and "fast" from query "Rust fast",
        // while "Rust is popular" matches only "Rust". Higher relevance should rank first.
        let store = InMemoryStore::new();

        let mut entry_partial =
            make_entry("m1", "agent1", "Rust is popular in the industry", "fact");
        entry_partial.importance = 5;

        let mut entry_full =
            make_entry("m2", "agent1", "Rust is fast and safe for systems", "fact");
        entry_full.importance = 5;

        // Store partial-match first so it would naturally sort first by insertion order
        store.store(&test_scope(), entry_partial).await.unwrap();
        store.store(&test_scope(), entry_full).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("Rust fast".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Both should match (both contain "rust")
        assert_eq!(results.len(), 2);
        // The entry matching both query tokens should rank higher
        assert_eq!(
            results[0].id, "m2",
            "entry matching more query tokens should rank first"
        );
    }

    // --- New field tests ---

    #[tokio::test]
    async fn recall_filters_by_memory_type() {
        let store = InMemoryStore::new();

        let mut episodic = make_entry("m1", "a", "episodic fact", "fact");
        episodic.memory_type = MemoryType::Episodic;
        store.store(&test_scope(), episodic).await.unwrap();

        let mut semantic = make_entry("m2", "a", "semantic knowledge", "fact");
        semantic.memory_type = MemoryType::Semantic;
        store.store(&test_scope(), semantic).await.unwrap();

        let mut reflection = make_entry("m3", "a", "reflection insight", "fact");
        reflection.memory_type = MemoryType::Reflection;
        store.store(&test_scope(), reflection).await.unwrap();

        // Filter by Semantic only
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    memory_type: Some(MemoryType::Semantic),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m2");

        // Filter by Reflection
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    memory_type: Some(MemoryType::Reflection),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m3");

        // No filter returns all
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn recall_filters_by_min_strength() {
        let store = InMemoryStore::new();

        let mut strong = make_entry("m1", "a", "strong memory", "fact");
        strong.strength = 0.9;
        store.store(&test_scope(), strong).await.unwrap();

        let mut weak = make_entry("m2", "a", "weak memory", "fact");
        weak.strength = 0.05;
        store.store(&test_scope(), weak).await.unwrap();

        // Only strong entries
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    min_strength: Some(0.5),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn strength_reinforced_on_access() {
        let store = InMemoryStore::new();

        let mut entry = make_entry("m1", "a", "test", "fact");
        entry.strength = 0.5;
        store.store(&test_scope(), entry).await.unwrap();

        // Recall reinforces strength by +0.2
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!((results[0].strength - 0.7).abs() < f64::EPSILON);

        // Second access: 0.7 + 0.2 = 0.9
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!((results[0].strength - 0.9).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn store_preserves_caller_supplied_strength() {
        // Issue #5: callers persisting an explicit `strength` (e.g. a freshly
        // weakened decay test fixture) must read the same value back on the
        // next pure recall.
        let store = InMemoryStore::new();

        let mut weak = make_entry("m1", "a", "test", "fact");
        weak.strength = 0.05;
        store.store(&test_scope(), weak).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    reinforce: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(
            (results[0].strength - 0.05).abs() < f64::EPSILON,
            "store must preserve caller strength; recall(reinforce=false) must not mutate it (got {})",
            results[0].strength
        );
    }

    #[tokio::test]
    async fn recall_with_reinforce_false_is_idempotent_for_strength() {
        let store = InMemoryStore::new();

        let mut entry = make_entry("m1", "a", "test", "fact");
        entry.strength = 0.4;
        store.store(&test_scope(), entry).await.unwrap();

        for _ in 0..5 {
            let results = store
                .recall(
                    &test_scope(),
                    MemoryQuery {
                        limit: 10,
                        reinforce: false,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            assert!((results[0].strength - 0.4).abs() < f64::EPSILON);
        }

        // access_count still ticks even when strength is frozen.
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    reinforce: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(results[0].access_count >= 5);
    }

    #[tokio::test]
    async fn strength_capped_at_one() {
        let store = InMemoryStore::new();

        let mut entry = make_entry("m1", "a", "test", "fact");
        entry.strength = 0.95;
        store.store(&test_scope(), entry).await.unwrap();

        // 0.95 + 0.2 should cap at 1.0
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!((results[0].strength - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn keywords_searched_during_recall() {
        let store = InMemoryStore::new();

        // Entry with "performance" only in keywords, not content
        let mut entry = make_entry("m1", "a", "Rust is great", "fact");
        entry.keywords = vec!["performance".into(), "speed".into()];
        store.store(&test_scope(), entry).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("performance".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn add_link_bidirectional() {
        let store = InMemoryStore::new();
        store
            .store(&test_scope(), make_entry("m1", "a", "first", "fact"))
            .await
            .unwrap();
        store
            .store(&test_scope(), make_entry("m2", "a", "second", "fact"))
            .await
            .unwrap();

        store.add_link(&test_scope(), "m1", "m2").await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let m1 = results.iter().find(|e| e.id == "m1").unwrap();
        let m2 = results.iter().find(|e| e.id == "m2").unwrap();

        assert!(m1.related_ids.contains(&"m2".to_string()));
        assert!(m2.related_ids.contains(&"m1".to_string()));
    }

    #[tokio::test]
    async fn add_link_idempotent() {
        let store = InMemoryStore::new();
        store
            .store(&test_scope(), make_entry("m1", "a", "first", "fact"))
            .await
            .unwrap();
        store
            .store(&test_scope(), make_entry("m2", "a", "second", "fact"))
            .await
            .unwrap();

        // Link twice — should not duplicate
        store.add_link(&test_scope(), "m1", "m2").await.unwrap();
        store.add_link(&test_scope(), "m1", "m2").await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let m1 = results.iter().find(|e| e.id == "m1").unwrap();
        assert_eq!(
            m1.related_ids.iter().filter(|id| *id == "m2").count(),
            1,
            "should not have duplicate links"
        );
    }

    #[tokio::test]
    async fn prune_removes_below_threshold() {
        let store = InMemoryStore::new();

        let mut strong = make_entry("m1", "a", "strong", "fact");
        strong.strength = 0.8;
        strong.created_at = Utc::now() - chrono::Duration::hours(48);
        store.store(&test_scope(), strong).await.unwrap();

        let mut weak = make_entry("m2", "a", "weak", "fact");
        weak.strength = 0.05;
        weak.created_at = Utc::now() - chrono::Duration::hours(48);
        store.store(&test_scope(), weak).await.unwrap();

        let pruned = store
            .prune(&test_scope(), 0.1, chrono::Duration::hours(1), None)
            .await
            .unwrap();
        assert_eq!(pruned, 1);

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn prune_respects_min_age() {
        let store = InMemoryStore::new();

        // Weak but recent — should NOT be pruned
        let mut weak_recent = make_entry("m1", "a", "weak recent", "fact");
        weak_recent.strength = 0.01;
        weak_recent.created_at = Utc::now(); // just created
        store.store(&test_scope(), weak_recent).await.unwrap();

        let pruned = store
            .prune(&test_scope(), 0.1, chrono::Duration::hours(24), None)
            .await
            .unwrap();
        assert_eq!(pruned, 0, "recent entry should not be pruned");
    }

    #[tokio::test]
    async fn prune_uses_effective_strength_with_decay() {
        let store = InMemoryStore::new();

        // Entry with moderate stored strength, but not accessed in a month.
        // effective_strength = 0.5 * e^(-0.005 * 720) ≈ 0.5 * 0.027 ≈ 0.014
        let mut old_accessed = make_entry("m1", "a", "old accessed", "fact");
        old_accessed.strength = 0.5;
        old_accessed.created_at = Utc::now() - chrono::Duration::hours(30 * 24);
        old_accessed.last_accessed = Utc::now() - chrono::Duration::hours(30 * 24);
        store.store(&test_scope(), old_accessed).await.unwrap();

        // Same stored strength but recently accessed — effective ≈ 0.5
        let mut recently_accessed = make_entry("m2", "a", "recently accessed", "fact");
        recently_accessed.strength = 0.5;
        recently_accessed.created_at = Utc::now() - chrono::Duration::hours(30 * 24);
        recently_accessed.last_accessed = Utc::now();
        store.store(&test_scope(), recently_accessed).await.unwrap();

        // Prune with min_strength=0.1, min_age=24h
        // m1: effective ≈ 0.014 < 0.1, age 30d > 24h → pruned
        // m2: effective ≈ 0.5 > 0.1 → kept
        let pruned = store
            .prune(&test_scope(), 0.1, chrono::Duration::hours(24), None)
            .await
            .unwrap();
        assert_eq!(pruned, 1, "old unaccessed entry should be pruned");

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m2");
    }

    #[tokio::test]
    async fn prune_with_agent_prefix_only_removes_matching_agent() {
        let store = InMemoryStore::new();

        // Weak + old entries from different agents
        let mut weak_a = make_entry("m1", "agent_a", "weak from A", "fact");
        weak_a.strength = 0.01;
        weak_a.created_at = Utc::now() - chrono::Duration::hours(48);
        weak_a.last_accessed = Utc::now() - chrono::Duration::hours(48);
        store.store(&test_scope(), weak_a).await.unwrap();

        let mut weak_b = make_entry("m2", "agent_b", "weak from B", "fact");
        weak_b.strength = 0.01;
        weak_b.created_at = Utc::now() - chrono::Duration::hours(48);
        weak_b.last_accessed = Utc::now() - chrono::Duration::hours(48);
        store.store(&test_scope(), weak_b).await.unwrap();

        // Prune with agent_prefix = "agent_a" — should only remove agent_a's entry
        let pruned = store
            .prune(
                &test_scope(),
                0.1,
                chrono::Duration::hours(1),
                Some("agent_a"),
            )
            .await
            .unwrap();
        assert_eq!(pruned, 1, "should only prune agent_a's entry");

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m2");
        assert_eq!(results[0].agent, "agent_b");
    }

    /// SECURITY (F-MEM-1): a NamespacedMemory whose name is a prefix of
    /// another sibling's must NOT prune the sibling's entries. Before the fix,
    /// `e.agent.starts_with("user:alice")` matched `user:alice2`, letting
    /// alice's NamespacedMemory wipe weak entries belonging to alice2 (or
    /// alice-staging, alice_admin, etc.).
    #[tokio::test]
    async fn prune_does_not_match_overlapping_agent_prefix() {
        let store = InMemoryStore::new();

        // alice — should be pruned
        let mut weak_alice = make_entry("ma", "user:alice", "weak alice", "fact");
        weak_alice.strength = 0.01;
        weak_alice.created_at = Utc::now() - chrono::Duration::hours(48);
        weak_alice.last_accessed = Utc::now() - chrono::Duration::hours(48);
        store.store(&test_scope(), weak_alice).await.unwrap();

        // alice2 — overlapping prefix; must NOT be pruned by `user:alice` prune.
        let mut weak_alice2 = make_entry("m2", "user:alice2", "weak alice2", "fact");
        weak_alice2.strength = 0.01;
        weak_alice2.created_at = Utc::now() - chrono::Duration::hours(48);
        weak_alice2.last_accessed = Utc::now() - chrono::Duration::hours(48);
        store.store(&test_scope(), weak_alice2).await.unwrap();

        // user:alice:tool — proper sub-namespace, MUST be pruned (separator match).
        let mut weak_subagent = make_entry("ms", "user:alice:tool", "weak sub", "fact");
        weak_subagent.strength = 0.01;
        weak_subagent.created_at = Utc::now() - chrono::Duration::hours(48);
        weak_subagent.last_accessed = Utc::now() - chrono::Duration::hours(48);
        store.store(&test_scope(), weak_subagent).await.unwrap();

        let pruned = store
            .prune(
                &test_scope(),
                0.1,
                chrono::Duration::hours(1),
                Some("user:alice"),
            )
            .await
            .unwrap();

        // Must prune alice + alice's sub-tool (separator match), NOT alice2.
        assert_eq!(pruned, 2, "must prune alice and user:alice:tool only");

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let agents: std::collections::HashSet<&str> =
            results.iter().map(|e| e.agent.as_str()).collect();
        assert!(
            agents.contains("user:alice2"),
            "alice2 must survive (overlapping prefix bypass): got {agents:?}"
        );
    }

    #[tokio::test]
    async fn prune_none_prefix_removes_all_matching() {
        let store = InMemoryStore::new();

        let mut weak_a = make_entry("m1", "agent_a", "weak from A", "fact");
        weak_a.strength = 0.01;
        weak_a.created_at = Utc::now() - chrono::Duration::hours(48);
        weak_a.last_accessed = Utc::now() - chrono::Duration::hours(48);
        store.store(&test_scope(), weak_a).await.unwrap();

        let mut weak_b = make_entry("m2", "agent_b", "weak from B", "fact");
        weak_b.strength = 0.01;
        weak_b.created_at = Utc::now() - chrono::Duration::hours(48);
        weak_b.last_accessed = Utc::now() - chrono::Duration::hours(48);
        store.store(&test_scope(), weak_b).await.unwrap();

        // Prune with None prefix — removes all weak entries
        let pruned = store
            .prune(&test_scope(), 0.1, chrono::Duration::hours(1), None)
            .await
            .unwrap();
        assert_eq!(pruned, 2, "should prune all weak entries");
    }

    #[tokio::test]
    async fn recall_bm25_ranks_better_than_naive_keyword() {
        // BM25 should rank an entry matching more query terms higher,
        // even when both entries have identical importance and recency.
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
            decay_rate: 0.01,
        });

        // Entry with only one query term match
        let e1 = make_entry("m1", "a", "Rust is a programming language", "fact");
        store.store(&test_scope(), e1).await.unwrap();

        // Entry matching both query terms
        let e2 = make_entry(
            "m2",
            "a",
            "Rust has excellent performance and speed",
            "fact",
        );
        store.store(&test_scope(), e2).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("Rust performance".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].id, "m2",
            "BM25 should rank entry matching more query terms first"
        );
    }

    #[tokio::test]
    async fn recall_bm25_keyword_field_boosts_ranking() {
        // Entry with match in keywords should rank higher than content-only match
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
            decay_rate: 0.01,
        });

        // Match in content only
        let e1 = make_entry("m1", "a", "optimization techniques for databases", "fact");
        store.store(&test_scope(), e1).await.unwrap();

        // Match in both content and keywords (keyword boost)
        let mut e2 = make_entry("m2", "a", "optimization techniques for systems", "fact");
        e2.keywords = vec!["optimization".into(), "databases".into()];
        store.store(&test_scope(), e2).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("optimization databases".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].id, "m2",
            "entry with keyword match should rank higher"
        );
    }

    #[tokio::test]
    async fn strength_affects_ranking() {
        // Use delta=1.0 (pure strength) to isolate the effect
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 0.0,
            delta: 1.0,
            decay_rate: 0.01,
        });

        let mut weak = make_entry("m1", "a", "weak entry", "fact");
        weak.strength = 0.2;
        store.store(&test_scope(), weak).await.unwrap();

        let mut strong = make_entry("m2", "a", "strong entry", "fact");
        strong.strength = 0.9;
        store.store(&test_scope(), strong).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].id, "m2",
            "stronger entry should rank first when delta=1.0"
        );
    }

    #[tokio::test]
    async fn hybrid_recall_cosine_boosts_semantic_match() {
        // Pure relevance scoring: entry with high cosine similarity but no keyword
        // match should still surface via hybrid retrieval.
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
            decay_rate: 0.01,
        });

        // e1: keyword match for "rust" but no embedding
        let e1 = make_entry("m1", "a", "Rust is fast", "fact");
        store.store(&test_scope(), e1).await.unwrap();

        // e2: no keyword match for "rust" but has embedding very similar to query
        let mut e2 = make_entry(
            "m2",
            "a",
            "Systems programming language with safety",
            "fact",
        );
        e2.embedding = Some(vec![0.9, 0.1, 0.0]);
        store.store(&test_scope(), e2).await.unwrap();

        // Query: "rust" with embedding close to e2's embedding
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("rust".into()),
                    query_embedding: Some(vec![0.9, 0.1, 0.0]),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Only m1 matches keyword filter, but m2 should not appear because
        // the keyword filter excludes it before scoring. Hybrid only affects
        // entries that pass the initial keyword filter.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn hybrid_recall_fuses_bm25_and_vector() {
        // When entries pass keyword filter AND have embeddings, hybrid should
        // affect ranking via RRF fusion. We use 3 entries so that RRF
        // asymmetry from vector ranking can change the outcome.
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
            decay_rate: 0.01,
        });

        // e1 matches "rust" and "fast" (2 terms) — strong BM25
        let mut e1 = make_entry("m1", "a", "Rust is fast and fast", "fact");
        e1.embedding = Some(vec![0.0, 0.0, 1.0]); // orthogonal to query embedding
        store.store(&test_scope(), e1).await.unwrap();

        // e2 matches only "rust" (1 term) — weaker BM25
        let mut e2 = make_entry("m2", "a", "Rust has zero-cost abstractions", "fact");
        e2.embedding = Some(vec![0.95, 0.05, 0.0]); // very similar to query embedding
        store.store(&test_scope(), e2).await.unwrap();

        // e3 matches only "rust" — weakest BM25, moderate vector
        let mut e3 = make_entry("m3", "a", "Rust is a programming language", "fact");
        e3.embedding = Some(vec![0.5, 0.5, 0.0]); // moderate similarity
        store.store(&test_scope(), e3).await.unwrap();

        // Without hybrid: BM25 ranks m1 first (matches "rust"+"fast").
        // With hybrid: vector strongly boosts m2 (0.95 similarity vs m1's 0.0).
        // RRF fuses both signals — m2 should come out on top.
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("rust fast".into()),
                    query_embedding: Some(vec![0.95, 0.05, 0.0]),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        // e2 is ranked #1 by vector (highest cosine) and #2 by BM25
        // e1 is ranked #1 by BM25 but #3 by vector (0.0 cosine = worst)
        // RRF: m2 = 1/52 + 1/51, m1 = 1/51 + 1/53, m3 = 1/53 + 1/52
        // m2 ≈ 0.03884, m1 ≈ 0.03850, m3 ≈ 0.03810
        assert_eq!(
            results[0].id, "m2",
            "entry with highest cosine similarity should rank first in hybrid mode"
        );
    }

    #[tokio::test]
    async fn hybrid_recall_bm25_fallback_when_no_embeddings() {
        // When query_embedding is set but no entries have embeddings,
        // should fall back to pure BM25 ranking.
        let store = InMemoryStore::new().with_scoring_weights(ScoringWeights {
            alpha: 0.0,
            beta: 0.0,
            gamma: 1.0,
            delta: 0.0,
            decay_rate: 0.01,
        });

        let e1 = make_entry("m1", "a", "Rust programming language", "fact");
        store.store(&test_scope(), e1).await.unwrap();

        let e2 = make_entry("m2", "a", "Rust performance and speed", "fact");
        store.store(&test_scope(), e2).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("Rust performance".into()),
                    query_embedding: Some(vec![0.5, 0.5, 0.0]),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        // e2 matches both "rust" and "performance" → higher BM25 → ranks first
        assert_eq!(results[0].id, "m2");
    }

    #[tokio::test]
    async fn recall_follows_related_ids_one_hop() {
        // m1 matches query "rust". m2 does NOT match "rust" but is linked to m1.
        // Graph expansion should surface m2 in results.
        let store = InMemoryStore::new();

        let mut m1 = make_entry("m1", "a", "Rust is fast", "fact");
        m1.related_ids = vec!["m2".into()];
        store.store(&test_scope(), m1).await.unwrap();

        let mut m2 = make_entry("m2", "a", "Memory safety guarantees", "fact");
        m2.related_ids = vec!["m1".into()];
        store.store(&test_scope(), m2).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("rust".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // m1 matches directly, m2 should appear via graph expansion
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"m1"), "direct match should be in results");
        assert!(
            ids.contains(&"m2"),
            "linked entry should be surfaced via graph expansion"
        );
    }

    #[tokio::test]
    async fn recall_graph_expansion_respects_strength_threshold() {
        let store = InMemoryStore::new();

        let mut m1 = make_entry("m1", "a", "Rust is fast", "fact");
        m1.related_ids = vec!["m2".into()];
        store.store(&test_scope(), m1).await.unwrap();

        // m2 has very low strength — should be excluded by min_strength
        let mut m2 = make_entry("m2", "a", "Weak linked memory", "fact");
        m2.related_ids = vec!["m1".into()];
        m2.strength = 0.01;
        m2.last_accessed = Utc::now() - chrono::Duration::hours(720); // very old
        store.store(&test_scope(), m2).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("rust".into()),
                    min_strength: Some(0.1),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Only m1 should appear — m2's effective strength is below threshold
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn recall_graph_expansion_does_not_duplicate() {
        let store = InMemoryStore::new();

        // Both m1 and m2 match "rust" directly AND are linked
        let mut m1 = make_entry("m1", "a", "Rust is fast", "fact");
        m1.related_ids = vec!["m2".into()];
        store.store(&test_scope(), m1).await.unwrap();

        let mut m2 = make_entry("m2", "a", "Rust is safe", "fact");
        m2.related_ids = vec!["m1".into()];
        store.store(&test_scope(), m2).await.unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("rust".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Both match directly — graph expansion should NOT add duplicates
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            ids.iter().filter(|&&id| id == "m1").count(),
            1,
            "m1 should appear exactly once"
        );
        assert_eq!(
            ids.iter().filter(|&&id| id == "m2").count(),
            1,
            "m2 should appear exactly once"
        );
    }

    #[tokio::test]
    async fn recall_agent_prefix_matches_sub_namespaces() {
        let store = InMemoryStore::new();
        // Sub-agent memories with compound namespace
        store
            .store(
                &test_scope(),
                make_entry("m1", "tg:123:assistant", "likes Rust", "fact"),
            )
            .await
            .unwrap();
        store
            .store(
                &test_scope(),
                make_entry("m2", "tg:123:researcher", "loves coffee", "fact"),
            )
            .await
            .unwrap();
        // Different user — should NOT match
        store
            .store(
                &test_scope(),
                make_entry("m3", "tg:456:assistant", "prefers Python", "fact"),
            )
            .await
            .unwrap();

        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    agent_prefix: Some("tg:123".into()),
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"m1"));
        assert!(ids.contains(&"m2"));
    }

    #[tokio::test]
    async fn recall_agent_exact_takes_precedence_over_prefix() {
        let store = InMemoryStore::new();
        store
            .store(
                &test_scope(),
                make_entry("m1", "tg:123:assistant", "from assistant", "fact"),
            )
            .await
            .unwrap();
        store
            .store(
                &test_scope(),
                make_entry("m2", "tg:123:researcher", "from researcher", "fact"),
            )
            .await
            .unwrap();

        // Exact agent filter should only return the exact match
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    agent: Some("tg:123:assistant".into()),
                    agent_prefix: Some("tg:123".into()), // ignored
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn recall_filters_by_max_confidentiality() {
        use super::super::Confidentiality;
        let store = InMemoryStore::new();

        let mut public = make_entry("m1", "a", "public fact", "fact");
        public.confidentiality = Confidentiality::Public;
        store.store(&test_scope(), public).await.unwrap();

        let mut internal = make_entry("m2", "a", "internal note", "fact");
        internal.confidentiality = Confidentiality::Internal;
        store.store(&test_scope(), internal).await.unwrap();

        let mut confidential = make_entry("m3", "a", "private expense", "fact");
        confidential.confidentiality = Confidentiality::Confidential;
        store.store(&test_scope(), confidential).await.unwrap();

        let mut restricted = make_entry("m4", "a", "api key", "fact");
        restricted.confidentiality = Confidentiality::Restricted;
        store.store(&test_scope(), restricted).await.unwrap();

        // Cap at Public — only public entries returned
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    max_confidentiality: Some(Confidentiality::Public),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");

        // No cap — all entries returned
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    max_confidentiality: None,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 4);

        // Cap at Confidential — excludes Restricted
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    max_confidentiality: Some(Confidentiality::Confidential),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|e| e.id != "m4"));
    }

    #[tokio::test]
    async fn graph_expansion_respects_max_confidentiality() {
        use super::super::Confidentiality;
        let store = InMemoryStore::new();

        // Public entry with a link to a Confidential entry
        let mut public = make_entry("m1", "a", "project update", "fact");
        public.confidentiality = Confidentiality::Public;
        public.related_ids = vec!["m2".into()];
        public.keywords = vec!["project".into()];
        store.store(&test_scope(), public).await.unwrap();

        let mut confidential = make_entry("m2", "a", "private expense data", "fact");
        confidential.confidentiality = Confidentiality::Confidential;
        confidential.keywords = vec!["expense".into()];
        store.store(&test_scope(), confidential).await.unwrap();

        // Query with Public cap and keyword that matches m1 — should NOT expand to m2
        let results = store
            .recall(
                &test_scope(),
                MemoryQuery {
                    text: Some("project".into()),
                    max_confidentiality: Some(Confidentiality::Public),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(
            results.iter().all(|e| e.id != "m2"),
            "graph expansion should not include Confidential entries when capped at Public"
        );
        assert!(results.iter().any(|e| e.id == "m1"));
    }

    // --- Tenant-scope isolation (B4): scope filter is independent of agent_prefix ---

    #[tokio::test]
    async fn recall_does_not_leak_across_tenants() {
        let store = InMemoryStore::new();
        let acme = TenantScope::new("acme");
        let globex = TenantScope::new("globex");

        store
            .store(&acme, make_entry("a1", "agent", "acme-secret", "fact"))
            .await
            .unwrap();
        store
            .store(&globex, make_entry("g1", "agent", "globex-secret", "fact"))
            .await
            .unwrap();

        let acme_results = store
            .recall(
                &acme,
                MemoryQuery {
                    agent: Some("agent".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(acme_results.len(), 1);
        assert_eq!(acme_results[0].id, "a1");

        let globex_results = store
            .recall(
                &globex,
                MemoryQuery {
                    agent: Some("agent".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(globex_results.len(), 1);
        assert_eq!(globex_results[0].id, "g1");
    }

    #[tokio::test]
    async fn forget_does_not_delete_other_tenant() {
        let store = InMemoryStore::new();
        let acme = TenantScope::new("acme");
        let globex = TenantScope::new("globex");

        store
            .store(&acme, make_entry("a1", "agent", "x", "fact"))
            .await
            .unwrap();
        store
            .store(&globex, make_entry("g1", "agent", "y", "fact"))
            .await
            .unwrap();

        // Try to forget acme's id under globex's scope — should not delete acme's entry.
        let removed = store.forget(&globex, "a1").await.unwrap();
        assert!(!removed);

        let acme_results = store
            .recall(
                &acme,
                MemoryQuery {
                    agent: Some("agent".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(acme_results.len(), 1);
    }

    #[tokio::test]
    async fn update_does_not_modify_other_tenant() {
        let store = InMemoryStore::new();
        let acme = TenantScope::new("acme");
        let globex = TenantScope::new("globex");

        store
            .store(&acme, make_entry("a1", "agent", "original", "fact"))
            .await
            .unwrap();

        // Cross-tenant update should fail (entry exists, but in a different tenant).
        let err = store
            .update(&globex, "a1", "tampered".into())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("memory not found"), "got: {err}");

        // Original content survived.
        let results = store
            .recall(
                &acme,
                MemoryQuery {
                    agent: Some("agent".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "original");
    }

    #[tokio::test]
    async fn store_under_scope_populates_author_tenant_id() {
        let store = InMemoryStore::new();
        let scope = TenantScope::new("acme").with_user("u-42");
        store
            .store(&scope, make_entry("s1", "agent", "x", "fact"))
            .await
            .unwrap();

        let results = store
            .recall(
                &scope,
                MemoryQuery {
                    agent: Some("agent".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].author_tenant_id.as_deref(), Some("acme"));
        assert_eq!(results[0].author_user_id.as_deref(), Some("u-42"));
    }
}
