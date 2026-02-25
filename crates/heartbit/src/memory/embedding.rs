use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;

use crate::error::Error;

use super::{Memory, MemoryEntry};

/// Trait for generating text embeddings.
#[allow(clippy::type_complexity)]
pub trait EmbeddingProvider: Send + Sync {
    fn embed(
        &self,
        texts: &[&str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, Error>> + Send + '_>>;

    fn dimension(&self) -> usize;
}

/// No-op embedding provider — returns empty results.
/// Used when no embedding API is configured (graceful degradation).
pub struct NoopEmbedding;

impl EmbeddingProvider for NoopEmbedding {
    fn embed(
        &self,
        texts: &[&str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, Error>> + Send + '_>> {
        let len = texts.len();
        Box::pin(async move { Ok(vec![vec![]; len]) })
    }

    fn dimension(&self) -> usize {
        0
    }
}

/// OpenAI-compatible embedding provider.
///
/// Calls `POST /v1/embeddings` with the configured model.
/// Works with OpenAI API and compatible endpoints.
pub struct OpenAiEmbedding {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    dimension: usize,
}

impl OpenAiEmbedding {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        let dimension = match model.as_str() {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // default
        };
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model,
            base_url: "https://api.openai.com".into(),
            dimension,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingProvider for OpenAiEmbedding {
    fn embed(
        &self,
        texts: &[&str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, Error>> + Send + '_>> {
        let input: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        Box::pin(async move {
            if input.is_empty() {
                return Ok(vec![]);
            }

            let body = serde_json::json!({
                "model": self.model,
                "input": input,
            });

            let resp = self
                .client
                .post(format!("{}/v1/embeddings", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| Error::Memory(format!("embedding request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_else(|_| "unknown error".into());
                return Err(Error::Memory(format!(
                    "embedding API returned {status}: {text}"
                )));
            }

            let response: EmbeddingResponse = resp
                .json()
                .await
                .map_err(|e| Error::Memory(format!("failed to parse embedding response: {e}")))?;

            Ok(response.data.into_iter().map(|d| d.embedding).collect())
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Decorator that generates embeddings on store and passes through to inner Memory.
///
/// When storing a `MemoryEntry` without an embedding, this wrapper generates
/// one via the configured `EmbeddingProvider` before delegating to the inner store.
/// All other operations pass through unchanged.
pub struct EmbeddingMemory {
    inner: Arc<dyn Memory>,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl EmbeddingMemory {
    pub fn new(inner: Arc<dyn Memory>, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self { inner, embedder }
    }
}

impl Memory for EmbeddingMemory {
    fn store(
        &self,
        entry: MemoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            let mut entry = entry;
            // Only generate embedding if not already present and embedder is real (dimension > 0)
            if entry.embedding.is_none() && self.embedder.dimension() > 0 {
                match self.embedder.embed(&[&entry.content]).await {
                    Ok(mut embeddings) if !embeddings.is_empty() => {
                        let emb = embeddings.swap_remove(0);
                        if !emb.is_empty() {
                            entry.embedding = Some(emb);
                        }
                    }
                    Ok(_) => {} // empty result, skip
                    Err(e) => {
                        // Log but don't fail — embedding is optional
                        tracing::warn!("failed to generate embedding for memory {}: {e}", entry.id);
                    }
                }
            }
            self.inner.store(entry).await
        })
    }

    fn recall(
        &self,
        query: super::MemoryQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryEntry>, Error>> + Send + '_>> {
        Box::pin(async move {
            let mut query = query;
            // Generate query embedding for hybrid retrieval when text is present
            // and no embedding was already provided.
            if query.query_embedding.is_none()
                && query.text.is_some()
                && self.embedder.dimension() > 0
            {
                let text = query.text.as_deref().unwrap_or_default();
                match self.embedder.embed(&[text]).await {
                    Ok(mut embeddings) if !embeddings.is_empty() => {
                        let emb = embeddings.swap_remove(0);
                        if !emb.is_empty() {
                            query.query_embedding = Some(emb);
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        // Log but don't fail — fall back to BM25-only
                        tracing::warn!("failed to generate query embedding: {e}");
                    }
                }
            }
            self.inner.recall(query).await
        })
    }

    fn update(
        &self,
        id: &str,
        content: String,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        self.inner.update(id, content)
    }

    fn forget(&self, id: &str) -> Pin<Box<dyn Future<Output = Result<bool, Error>> + Send + '_>> {
        self.inner.forget(id)
    }

    fn add_link(
        &self,
        id: &str,
        related_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        self.inner.add_link(id, related_id)
    }

    fn prune(
        &self,
        min_strength: f64,
        min_age: chrono::Duration,
    ) -> Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + '_>> {
        self.inner.prune(min_strength, min_age)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::in_memory::InMemoryStore;
    use crate::memory::{Confidentiality, MemoryEntry, MemoryQuery, MemoryType};
    use chrono::Utc;

    fn make_entry(id: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: id.into(),
            agent: "test".into(),
            content: content.into(),
            category: "fact".into(),
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
        }
    }

    #[test]
    fn noop_embedding_returns_empty() {
        let noop = NoopEmbedding;
        assert_eq!(noop.dimension(), 0);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let result = rt.block_on(noop.embed(&["hello", "world"])).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].is_empty());
        assert!(result[1].is_empty());
    }

    #[test]
    fn embedding_provider_is_object_safe() {
        fn _accepts_dyn(_p: &dyn EmbeddingProvider) {}
    }

    #[test]
    fn embedding_memory_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmbeddingMemory>();
    }

    #[tokio::test]
    async fn noop_embedding_skips_embedding_on_store() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(NoopEmbedding);
        let em = EmbeddingMemory::new(store.clone(), embedder);

        em.store(make_entry("m1", "test content")).await.unwrap();

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].embedding.is_none());
    }

    /// Fake embedding provider for testing that returns deterministic vectors.
    struct FakeEmbedding;

    impl EmbeddingProvider for FakeEmbedding {
        fn embed(
            &self,
            texts: &[&str],
        ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, Error>> + Send + '_>> {
            let results: Vec<Vec<f32>> = texts
                .iter()
                .map(|t| {
                    // Simple deterministic embedding: first 4 bytes as f32 values
                    let bytes = t.as_bytes();
                    vec![
                        bytes.first().copied().unwrap_or(0) as f32 / 255.0,
                        bytes.get(1).copied().unwrap_or(0) as f32 / 255.0,
                        bytes.get(2).copied().unwrap_or(0) as f32 / 255.0,
                    ]
                })
                .collect();
            Box::pin(async move { Ok(results) })
        }

        fn dimension(&self) -> usize {
            3
        }
    }

    #[tokio::test]
    async fn embedding_memory_generates_embedding_on_store() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(FakeEmbedding);
        let em = EmbeddingMemory::new(store.clone(), embedder);

        em.store(make_entry("m1", "hello")).await.unwrap();

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        let emb = results[0]
            .embedding
            .as_ref()
            .expect("embedding should be set");
        assert_eq!(emb.len(), 3);
    }

    #[tokio::test]
    async fn embedding_memory_preserves_existing_embedding() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(FakeEmbedding);
        let em = EmbeddingMemory::new(store.clone(), embedder);

        let mut entry = make_entry("m1", "hello");
        entry.embedding = Some(vec![9.0, 8.0, 7.0]);
        em.store(entry).await.unwrap();

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        let emb = results[0].embedding.as_ref().unwrap();
        // Should keep original, not overwrite with FakeEmbedding output
        assert!((emb[0] - 9.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn embedding_memory_delegates_recall() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(NoopEmbedding);
        let em = EmbeddingMemory::new(store.clone(), embedder);

        store.store(make_entry("m1", "test")).await.unwrap();
        let results = em
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
    }

    #[tokio::test]
    async fn embedding_memory_generates_query_embedding_on_recall() {
        // When EmbeddingMemory wraps a store and query has text,
        // it should generate a query embedding for hybrid retrieval.
        use std::sync::atomic::{AtomicBool, Ordering};

        // Tracking embedding provider that records whether embed() was called
        struct TrackingEmbedding {
            called: Arc<AtomicBool>,
        }

        impl EmbeddingProvider for TrackingEmbedding {
            fn embed(
                &self,
                _texts: &[&str],
            ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, Error>> + Send + '_>>
            {
                self.called.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(vec![vec![0.5, 0.5, 0.5]]) })
            }

            fn dimension(&self) -> usize {
                3
            }
        }

        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let called = Arc::new(AtomicBool::new(false));
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(TrackingEmbedding {
            called: called.clone(),
        });
        let em = EmbeddingMemory::new(store.clone(), embedder);

        store.store(make_entry("m1", "hello world")).await.unwrap();

        // Recall with text query should trigger embedding generation
        let _results = em
            .recall(MemoryQuery {
                text: Some("hello".into()),
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(
            called.load(Ordering::SeqCst),
            "embed() should have been called for query text"
        );
    }

    #[tokio::test]
    async fn embedding_memory_skips_query_embedding_without_text() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct TrackingEmbedding {
            called: Arc<AtomicBool>,
        }

        impl EmbeddingProvider for TrackingEmbedding {
            fn embed(
                &self,
                _texts: &[&str],
            ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, Error>> + Send + '_>>
            {
                self.called.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(vec![vec![0.5, 0.5, 0.5]]) })
            }

            fn dimension(&self) -> usize {
                3
            }
        }

        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let called = Arc::new(AtomicBool::new(false));
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(TrackingEmbedding {
            called: called.clone(),
        });
        let em = EmbeddingMemory::new(store.clone(), embedder);

        store.store(make_entry("m1", "hello world")).await.unwrap();

        // Recall WITHOUT text query should NOT generate embedding
        let _results = em
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(
            !called.load(Ordering::SeqCst),
            "embed() should NOT be called when no text query"
        );
    }

    #[tokio::test]
    async fn embedding_memory_delegates_forget() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(NoopEmbedding);
        let em = EmbeddingMemory::new(store.clone(), embedder);

        store.store(make_entry("m1", "test")).await.unwrap();
        let removed = em.forget("m1").await.unwrap();
        assert!(removed);

        let results = store
            .recall(MemoryQuery {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(results.is_empty());
    }
}
