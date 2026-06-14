//! Reranking decorator for a [`KnowledgeBase`] — wires the rerank stage into
//! retrieval.
//!
//! Wraps any `KnowledgeBase`: a search over-fetches `fetch_multiplier × limit`
//! candidates from the inner base (BM25 / hybrid), reranks them by true query
//! relevance via a [`Reranker`](crate::memory::rerank::Reranker), and returns the
//! top `limit`. This is the "+ rerank" of BM25 + dense + **rerank** as a live
//! retrieval path, not a loose helper.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::{Chunk, KnowledgeBase, KnowledgeQuery, SearchResult};
use crate::auth::TenantScope;
use crate::error::Error;
use crate::memory::rerank::Reranker;

/// A [`KnowledgeBase`] that reranks the inner base's results by query relevance.
pub struct RerankingKnowledgeBase {
    inner: Arc<dyn KnowledgeBase>,
    reranker: Arc<dyn Reranker>,
    /// How many candidates to over-fetch from the inner base per requested
    /// result before reranking (≥ 1). Larger = better recall, more rerank cost.
    fetch_multiplier: usize,
}

impl RerankingKnowledgeBase {
    /// Wrap `inner`, reranking with `reranker`. Over-fetches 4× the requested
    /// limit before reranking by default.
    pub fn new(inner: Arc<dyn KnowledgeBase>, reranker: Arc<dyn Reranker>) -> Self {
        Self {
            inner,
            reranker,
            fetch_multiplier: 4,
        }
    }

    /// Set the over-fetch multiplier (clamped to ≥ 1).
    #[must_use]
    pub fn fetch_multiplier(mut self, m: usize) -> Self {
        self.fetch_multiplier = m.max(1);
        self
    }
}

impl KnowledgeBase for RerankingKnowledgeBase {
    fn index(
        &self,
        scope: &TenantScope,
        chunk: Chunk,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        self.inner.index(scope, chunk)
    }

    fn chunk_count(
        &self,
        scope: &TenantScope,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        self.inner.chunk_count(scope)
    }

    fn search(
        &self,
        scope: &TenantScope,
        query: KnowledgeQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<SearchResult>, Error>> + Send + '_>> {
        let scope = scope.clone();
        Box::pin(async move {
            let final_limit = query.limit;
            // Over-fetch candidates for the reranker to choose from.
            let fetch_query = KnowledgeQuery {
                text: query.text.clone(),
                source_filter: query.source_filter.clone(),
                limit: final_limit
                    .saturating_mul(self.fetch_multiplier)
                    .max(final_limit),
            };
            let candidates = self.inner.search(&scope, fetch_query).await?;
            if candidates.len() <= 1 {
                return Ok(candidates.into_iter().take(final_limit).collect());
            }

            let texts: Vec<String> = candidates.iter().map(|r| r.chunk.content.clone()).collect();
            let ranked = self.reranker.rerank(&query.text, &texts).await?;

            // Reorder candidates by the reranker's order, then truncate.
            let reordered: Vec<SearchResult> = ranked
                .into_iter()
                .filter_map(|r| candidates.get(r.index).cloned())
                .take(final_limit)
                .collect();
            Ok(reordered)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{DocumentSource, in_memory::InMemoryKnowledgeBase};
    use crate::memory::rerank::RankedCandidate;

    fn scope() -> TenantScope {
        TenantScope::default()
    }

    fn chunk(id: &str, content: &str) -> Chunk {
        Chunk {
            id: id.into(),
            content: content.into(),
            source: DocumentSource {
                uri: "doc".into(),
                title: "doc".into(),
            },
            chunk_index: 0,
            tenant_id: None,
        }
    }

    /// Reranker that promotes any candidate containing "RELEVANT" to the top.
    struct KeywordReranker;
    impl Reranker for KeywordReranker {
        fn rerank<'a>(
            &'a self,
            _query: &'a str,
            candidates: &'a [String],
        ) -> Pin<Box<dyn Future<Output = Result<Vec<RankedCandidate>, Error>> + Send + 'a>>
        {
            let mut ranked: Vec<RankedCandidate> = candidates
                .iter()
                .enumerate()
                .map(|(index, c)| RankedCandidate {
                    index,
                    score: if c.contains("RELEVANT") { 1.0 } else { 0.1 },
                })
                .collect();
            ranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.index.cmp(&b.index))
            });
            Box::pin(async move { Ok(ranked) })
        }
    }

    #[tokio::test]
    async fn reranks_inner_results_to_promote_relevant() {
        let inner = Arc::new(InMemoryKnowledgeBase::new());
        // Index three chunks all matching the query term "data" so BM25 returns
        // them; the genuinely relevant one is indexed LAST (so it's not naturally
        // first).
        inner
            .index(&scope(), chunk("a", "data data noise"))
            .await
            .unwrap();
        inner
            .index(&scope(), chunk("b", "data filler text"))
            .await
            .unwrap();
        inner
            .index(&scope(), chunk("c", "data RELEVANT answer"))
            .await
            .unwrap();

        let reranked = RerankingKnowledgeBase::new(inner, Arc::new(KeywordReranker));
        let results = reranked
            .search(
                &scope(),
                KnowledgeQuery {
                    text: "data".into(),
                    source_filter: None,
                    limit: 1,
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0].chunk.content.contains("RELEVANT"),
            "reranker should have promoted the relevant chunk to top-1, got: {}",
            results[0].chunk.content
        );
    }

    #[tokio::test]
    async fn delegates_index_and_count() {
        let inner = Arc::new(InMemoryKnowledgeBase::new());
        let reranked = RerankingKnowledgeBase::new(inner, Arc::new(KeywordReranker));
        reranked.index(&scope(), chunk("a", "hello")).await.unwrap();
        assert_eq!(reranked.chunk_count(&scope()).await.unwrap(), 1);
    }
}
