pub mod chunker;
pub mod in_memory;
pub mod loader;
pub mod tools;

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Provenance of a document chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentSource {
    /// File path or URL where the document was loaded from.
    pub uri: String,
    /// Human-readable title (filename, page title, etc.).
    pub title: String,
}

/// Atomic search unit: a slice of a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Deterministic ID derived from source URI + chunk index.
    pub id: String,
    /// The text content of this chunk.
    pub content: String,
    /// Where this chunk came from.
    pub source: DocumentSource,
    /// Position of this chunk within its source document (0-based).
    pub chunk_index: usize,
}

/// Query parameters for knowledge search.
#[derive(Debug, Clone)]
pub struct KnowledgeQuery {
    /// Free-text search query.
    pub text: String,
    /// Optional filter to restrict results to a specific source URI prefix.
    pub source_filter: Option<String>,
    /// Maximum number of results to return.
    pub limit: usize,
}

/// A single search result with relevance info.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matching chunk.
    pub chunk: Chunk,
    /// Number of query-term matches found in the chunk.
    pub match_count: usize,
}

/// Trait for knowledge base implementations.
///
/// Uses `Pin<Box<dyn Future>>` for dyn-compatibility, matching `Tool`, `Memory`,
/// and `Blackboard` traits.
pub trait KnowledgeBase: Send + Sync {
    /// Index a chunk into the knowledge base.
    fn index(&self, chunk: Chunk) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>>;

    /// Search the knowledge base.
    fn search(
        &self,
        query: KnowledgeQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<SearchResult>, Error>> + Send + '_>>;

    /// Return the total number of indexed chunks.
    fn chunk_count(&self) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_source_equality() {
        let a = DocumentSource {
            uri: "docs/readme.md".into(),
            title: "README".into(),
        };
        let b = DocumentSource {
            uri: "docs/readme.md".into(),
            title: "README".into(),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn chunk_serializes() {
        let chunk = Chunk {
            id: "abc-0".into(),
            content: "Hello world".into(),
            source: DocumentSource {
                uri: "test.md".into(),
                title: "Test".into(),
            },
            chunk_index: 0,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: Chunk = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "abc-0");
        assert_eq!(parsed.content, "Hello world");
        assert_eq!(parsed.source.uri, "test.md");
        assert_eq!(parsed.chunk_index, 0);
    }

    #[test]
    fn knowledge_query_with_filter() {
        let q = KnowledgeQuery {
            text: "rust async".into(),
            source_filter: Some("docs/".into()),
            limit: 5,
        };
        assert_eq!(q.text, "rust async");
        assert_eq!(q.source_filter.as_deref(), Some("docs/"));
        assert_eq!(q.limit, 5);
    }

    #[test]
    fn knowledge_query_without_filter() {
        let q = KnowledgeQuery {
            text: "search".into(),
            source_filter: None,
            limit: 10,
        };
        assert!(q.source_filter.is_none());
    }

    #[test]
    fn search_result_holds_chunk_and_count() {
        let result = SearchResult {
            chunk: Chunk {
                id: "x-0".into(),
                content: "test".into(),
                source: DocumentSource {
                    uri: "f.md".into(),
                    title: "F".into(),
                },
                chunk_index: 0,
            },
            match_count: 3,
        };
        assert_eq!(result.match_count, 3);
        assert_eq!(result.chunk.id, "x-0");
    }

    #[test]
    fn knowledge_base_is_object_safe() {
        fn _accepts_dyn(_kb: &dyn KnowledgeBase) {}
    }
}
