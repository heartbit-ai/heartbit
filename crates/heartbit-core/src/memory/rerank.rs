//! Reranking — the third stage of BM25 + dense + **rerank** retrieval.
//!
//! Hybrid fusion ([`rrf_fuse`](super::hybrid::rrf_fuse), already wired into
//! recall) merges lexical (BM25) and dense (embedding) candidate lists, but the
//! fused order is still only as good as the two cheap signals. The 2026 RAG
//! stack adds a **reranker** that re-scores the fused top-N by true query
//! relevance — the highest-precision stage, since it reads each candidate in full
//! against the query.
//!
//! [`Reranker`] is the seam; [`LlmReranker`] is a concrete reranker that asks a
//! (cheap) judge model to score each candidate's relevance. Plug a cross-encoder
//! behind the same trait when one is available.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{CompletionRequest, ContentBlock, Message, Role};

/// A candidate's position after reranking. Higher `score` = more relevant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankedCandidate {
    /// Index into the input candidate slice.
    pub index: usize,
    /// Relevance score in `[0, 1]`.
    pub score: f64,
}

/// Re-scores candidates by relevance to a query, returning them sorted best-first.
pub trait Reranker: Send + Sync {
    /// Rerank `candidates` against `query`. Returns indices+scores, best-first.
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        candidates: &'a [String],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RankedCandidate>, Error>> + Send + 'a>>;
}

const RERANK_SYSTEM: &str = "You are a search RELEVANCE rater. Given a QUERY and a \
DOCUMENT, rate how well the document answers the query. Reply with exactly one \
line: 'RELEVANCE: N' where N is an integer 0-100 (100 = directly and fully \
answers, 0 = irrelevant). Output only that line.";

/// A [`Reranker`] backed by an LLM judge: scores each candidate's relevance to
/// the query, then sorts descending. Use a cheap model.
pub struct LlmReranker<P: LlmProvider> {
    provider: Arc<P>,
    max_tokens: u32,
}

impl<P: LlmProvider> LlmReranker<P> {
    /// Build a reranker over `provider`.
    pub fn new(provider: Arc<P>) -> Self {
        Self {
            provider,
            max_tokens: 16,
        }
    }
}

/// Parse a `RELEVANCE: N` (0-100) line into a `[0, 1]` score; 0.0 if absent.
pub(crate) fn parse_relevance(text: &str) -> f64 {
    for line in text.lines() {
        let upper = line.to_uppercase();
        if let Some(rest) = upper.split("RELEVANCE:").nth(1) {
            let digits: String = rest
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = digits.parse::<u32>() {
                return (n.min(100) as f64) / 100.0;
            }
        }
    }
    0.0
}

impl<P: LlmProvider> Reranker for LlmReranker<P> {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        candidates: &'a [String],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RankedCandidate>, Error>> + Send + 'a>> {
        Box::pin(async move {
            let mut ranked = Vec::with_capacity(candidates.len());
            for (index, candidate) in candidates.iter().enumerate() {
                let user = format!("QUERY:\n{query}\n\nDOCUMENT:\n{candidate}");
                let request = CompletionRequest {
                    system: RERANK_SYSTEM.to_string(),
                    messages: vec![Message {
                        role: Role::User,
                        content: vec![ContentBlock::Text { text: user }],
                    }],
                    tools: Vec::new(),
                    max_tokens: self.max_tokens,
                    tool_choice: None,
                    reasoning_effort: None,
                };
                let score = match self.provider.complete(request).await {
                    Ok(resp) => parse_relevance(&resp.text()),
                    Err(e) => {
                        tracing::warn!(index, error = %e, "rerank scoring failed; treating as 0");
                        0.0
                    }
                };
                ranked.push(RankedCandidate { index, score });
            }
            // Sort by score desc; ties keep original order (stable on equal keys
            // via index as the tiebreaker).
            ranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.index.cmp(&b.index))
            });
            Ok(ranked)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_helpers::MockProvider;

    #[test]
    fn parse_relevance_variants() {
        assert!((parse_relevance("RELEVANCE: 90") - 0.90).abs() < 1e-9);
        assert!((parse_relevance("relevance: 100 yes") - 1.0).abs() < 1e-9);
        assert!((parse_relevance("RELEVANCE: 999") - 1.0).abs() < 1e-9);
        assert_eq!(parse_relevance("no rating"), 0.0);
    }

    #[tokio::test]
    async fn llm_reranker_sorts_by_relevance() {
        // Three candidates scored 20 / 90 / 50 → order [1, 2, 0].
        let provider = Arc::new(MockProvider::new(vec![
            MockProvider::text_response("RELEVANCE: 20", 5, 2),
            MockProvider::text_response("RELEVANCE: 90", 5, 2),
            MockProvider::text_response("RELEVANCE: 50", 5, 2),
        ]));
        let reranker = LlmReranker::new(provider);
        let candidates = vec![
            "doc zero".to_string(),
            "doc one".to_string(),
            "doc two".to_string(),
        ];
        let ranked = reranker.rerank("query", &candidates).await.unwrap();
        assert_eq!(
            ranked.iter().map(|r| r.index).collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
        assert!((ranked[0].score - 0.90).abs() < 1e-9);
    }

    #[tokio::test]
    async fn empty_candidates_yield_empty_ranking() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let reranker = LlmReranker::new(provider);
        let ranked = reranker.rerank("q", &[]).await.unwrap();
        assert!(ranked.is_empty());
    }
}
