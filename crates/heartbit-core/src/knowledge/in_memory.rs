use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use tokio::sync::RwLock;

use crate::error::Error;

use super::{Chunk, KnowledgeBase, KnowledgeQuery, SearchResult};

/// In-memory knowledge base backed by a `tokio::sync::RwLock<HashMap>`.
///
/// Search is keyword-based: tokenizes query into lowercase words, counts
/// matches per chunk, and sorts by match count descending.
///
/// Always used behind `Arc<dyn KnowledgeBase>`, so no inner `Arc` needed.
pub struct InMemoryKnowledgeBase {
    chunks: RwLock<HashMap<String, Chunk>>,
}

impl InMemoryKnowledgeBase {
    pub fn new() -> Self {
        Self {
            chunks: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

/// Tokenize text into deduplicated lowercase words for keyword matching.
fn tokenize(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty() && seen.insert(w.clone()))
        .collect()
}

/// Count how many query tokens appear in the chunk content.
fn count_matches(query_tokens: &[String], content: &str) -> usize {
    let lower = content.to_lowercase();
    query_tokens
        .iter()
        .filter(|t| lower.contains(t.as_str()))
        .count()
}

impl KnowledgeBase for InMemoryKnowledgeBase {
    fn index(&self, chunk: Chunk) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            let mut data = self.chunks.write().await;
            data.insert(chunk.id.clone(), chunk);
            Ok(())
        })
    }

    fn search(
        &self,
        query: KnowledgeQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<SearchResult>, Error>> + Send + '_>> {
        Box::pin(async move {
            let data = self.chunks.read().await;
            let tokens = tokenize(&query.text);

            if tokens.is_empty() {
                return Ok(vec![]);
            }

            let mut results: Vec<SearchResult> = data
                .values()
                .filter(|chunk| {
                    if let Some(ref filter) = query.source_filter {
                        chunk.source.uri.starts_with(filter)
                    } else {
                        true
                    }
                })
                .filter_map(|chunk| {
                    let matches = count_matches(&tokens, &chunk.content);
                    if matches > 0 {
                        Some(SearchResult {
                            chunk: chunk.clone(),
                            match_count: matches,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            // Sort by match count descending, then chunk_index, then source URI for full stability
            results.sort_by(|a, b| {
                b.match_count
                    .cmp(&a.match_count)
                    .then_with(|| a.chunk.chunk_index.cmp(&b.chunk.chunk_index))
                    .then_with(|| a.chunk.source.uri.cmp(&b.chunk.source.uri))
            });

            if query.limit > 0 {
                results.truncate(query.limit);
            }

            Ok(results)
        })
    }

    fn chunk_count(&self) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        Box::pin(async move {
            let data = self.chunks.read().await;
            Ok(data.len())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::DocumentSource;
    use std::sync::Arc;

    fn make_chunk(id: &str, content: &str, uri: &str, index: usize) -> Chunk {
        Chunk {
            id: id.into(),
            content: content.into(),
            source: DocumentSource {
                uri: uri.into(),
                title: uri.into(),
            },
            chunk_index: index,
        }
    }

    #[tokio::test]
    async fn index_and_search_roundtrip() {
        let kb = InMemoryKnowledgeBase::new();
        kb.index(make_chunk(
            "c1",
            "Rust is a systems programming language",
            "docs/rust.md",
            0,
        ))
        .await
        .unwrap();

        let results = kb
            .search(KnowledgeQuery {
                text: "rust programming".into(),
                source_filter: None,
                limit: 5,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "c1");
        assert_eq!(results[0].match_count, 2); // "rust" + "programming"
    }

    #[tokio::test]
    async fn search_is_case_insensitive() {
        let kb = InMemoryKnowledgeBase::new();
        kb.index(make_chunk("c1", "RUST is GREAT", "f.md", 0))
            .await
            .unwrap();

        let results = kb
            .search(KnowledgeQuery {
                text: "rust great".into(),
                source_filter: None,
                limit: 5,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].match_count, 2);
    }

    #[tokio::test]
    async fn source_filter_restricts_results() {
        let kb = InMemoryKnowledgeBase::new();
        kb.index(make_chunk("c1", "Rust language", "docs/rust.md", 0))
            .await
            .unwrap();
        kb.index(make_chunk("c2", "Rust compiler", "api/rust.md", 0))
            .await
            .unwrap();

        let results = kb
            .search(KnowledgeQuery {
                text: "rust".into(),
                source_filter: Some("docs/".into()),
                limit: 10,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.source.uri, "docs/rust.md");
    }

    #[tokio::test]
    async fn limit_truncates_results() {
        let kb = InMemoryKnowledgeBase::new();
        for i in 0..10 {
            kb.index(make_chunk(
                &format!("c{i}"),
                "rust programming language",
                "docs/rust.md",
                i,
            ))
            .await
            .unwrap();
        }

        let results = kb
            .search(KnowledgeQuery {
                text: "rust".into(),
                source_filter: None,
                limit: 3,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn sorted_by_match_count_descending() {
        let kb = InMemoryKnowledgeBase::new();
        kb.index(make_chunk("c1", "rust", "f.md", 0)).await.unwrap();
        kb.index(make_chunk("c2", "rust programming rust systems", "f.md", 1))
            .await
            .unwrap();
        kb.index(make_chunk("c3", "rust programming", "f.md", 2))
            .await
            .unwrap();

        let results = kb
            .search(KnowledgeQuery {
                text: "rust programming systems".into(),
                source_filter: None,
                limit: 10,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].chunk.id, "c2"); // 3 matches
        assert_eq!(results[1].chunk.id, "c3"); // 2 matches
        assert_eq!(results[2].chunk.id, "c1"); // 1 match
    }

    #[tokio::test]
    async fn reindex_replaces_chunk() {
        let kb = InMemoryKnowledgeBase::new();
        kb.index(make_chunk("c1", "old content", "f.md", 0))
            .await
            .unwrap();
        kb.index(make_chunk("c1", "new content about rust", "f.md", 0))
            .await
            .unwrap();

        assert_eq!(kb.chunk_count().await.unwrap(), 1);

        let results = kb
            .search(KnowledgeQuery {
                text: "rust".into(),
                source_filter: None,
                limit: 5,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].chunk.content.contains("new content"));
    }

    #[tokio::test]
    async fn empty_query_returns_no_results() {
        let kb = InMemoryKnowledgeBase::new();
        kb.index(make_chunk("c1", "some content", "f.md", 0))
            .await
            .unwrap();

        let results = kb
            .search(KnowledgeQuery {
                text: "".into(),
                source_filter: None,
                limit: 5,
            })
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn no_match_returns_empty() {
        let kb = InMemoryKnowledgeBase::new();
        kb.index(make_chunk("c1", "hello world", "f.md", 0))
            .await
            .unwrap();

        let results = kb
            .search(KnowledgeQuery {
                text: "zzzznotfound".into(),
                source_filter: None,
                limit: 5,
            })
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn chunk_count_tracks_size() {
        let kb = InMemoryKnowledgeBase::new();
        assert_eq!(kb.chunk_count().await.unwrap(), 0);

        kb.index(make_chunk("c1", "a", "f.md", 0)).await.unwrap();
        kb.index(make_chunk("c2", "b", "f.md", 1)).await.unwrap();
        assert_eq!(kb.chunk_count().await.unwrap(), 2);
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemoryKnowledgeBase>();
        fn _accepts_dyn(_kb: &dyn KnowledgeBase) {}
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_index_and_search() {
        let kb = Arc::new(InMemoryKnowledgeBase::new());
        let mut handles = Vec::new();

        // Spawn writers
        for i in 0..20 {
            let kb = kb.clone();
            handles.push(tokio::spawn(async move {
                kb.index(make_chunk(
                    &format!("c{i}"),
                    &format!("rust content item {i}"),
                    "f.md",
                    i,
                ))
                .await
                .unwrap();
            }));
        }

        // Spawn readers concurrently
        for _ in 0..10 {
            let kb = kb.clone();
            handles.push(tokio::spawn(async move {
                let _ = kb
                    .search(KnowledgeQuery {
                        text: "rust".into(),
                        source_filter: None,
                        limit: 5,
                    })
                    .await
                    .unwrap();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(kb.chunk_count().await.unwrap(), 20);
    }

    #[tokio::test]
    async fn duplicate_query_terms_not_inflated() {
        let kb = InMemoryKnowledgeBase::new();
        kb.index(make_chunk("c1", "rust is great", "f.md", 0))
            .await
            .unwrap();

        let results = kb
            .search(KnowledgeQuery {
                text: "rust rust rust".into(),
                source_filter: None,
                limit: 5,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].match_count, 1); // deduplicated, not 3
    }

    #[tokio::test]
    async fn sort_stable_across_sources() {
        let kb = InMemoryKnowledgeBase::new();
        kb.index(make_chunk("c1", "rust programming", "z_file.md", 0))
            .await
            .unwrap();
        kb.index(make_chunk("c2", "rust programming", "a_file.md", 0))
            .await
            .unwrap();

        let results = kb
            .search(KnowledgeQuery {
                text: "rust".into(),
                source_filter: None,
                limit: 10,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        // Same match_count and chunk_index → sorted by source URI
        assert_eq!(results[0].chunk.source.uri, "a_file.md");
        assert_eq!(results[1].chunk.source.uri, "z_file.md");
    }
}
