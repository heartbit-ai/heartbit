//! Local embedding provider using fastembed (ONNX Runtime).
//!
//! Available behind the `local-embedding` feature.
//!
//! Runs inference locally — no API keys, no network, zero cost per query.
//! Model is downloaded once on first use (~30MB for the default model).

use std::future::Future;
use std::pin::Pin;

use heartbit_core::error::Error;
use heartbit_core::memory::embedding::EmbeddingProvider;

pub struct LocalEmbeddingProvider {
    model: std::sync::Arc<std::sync::Mutex<fastembed::TextEmbedding>>,
    dim: usize,
}

impl LocalEmbeddingProvider {
    /// Create a new local embedding provider.
    ///
    /// - `model_name`: Model identifier (e.g. `"all-MiniLM-L6-v2"`). Defaults to `AllMiniLML6V2`.
    /// - `cache_dir`: Directory to cache downloaded model files. Defaults to fastembed's default.
    pub fn new(
        model_name: Option<&str>,
        cache_dir: Option<&std::path::Path>,
    ) -> Result<Self, Error> {
        let model_enum = model_name
            .map(Self::parse_model_name)
            .unwrap_or(Ok(fastembed::EmbeddingModel::AllMiniLML6V2))?;

        // Look up dimension from supported models list
        let dim = fastembed::TextEmbedding::list_supported_models()
            .into_iter()
            .find(|info| std::mem::discriminant(&info.model) == std::mem::discriminant(&model_enum))
            .map(|info| info.dim)
            .ok_or_else(|| {
                Error::Memory(format!(
                    "model {model_enum:?} not found in supported models list"
                ))
            })?;

        let mut options = fastembed::TextInitOptions::new(model_enum);
        if let Some(dir) = cache_dir {
            options.cache_dir = dir.to_path_buf();
        }
        options.show_download_progress = true;

        let embedding =
            fastembed::TextEmbedding::try_new(options).map_err(|e| Error::Memory(e.to_string()))?;

        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(embedding)),
            dim,
        })
    }

    fn parse_model_name(name: &str) -> Result<fastembed::EmbeddingModel, Error> {
        // Map common string names to enum variants
        match name {
            "all-MiniLM-L6-v2" | "AllMiniLML6V2" => Ok(fastembed::EmbeddingModel::AllMiniLML6V2),
            "all-MiniLM-L6-v2-q" | "AllMiniLML6V2Q" => {
                Ok(fastembed::EmbeddingModel::AllMiniLML6V2Q)
            }
            "all-MiniLM-L12-v2" | "AllMiniLML12V2" => Ok(fastembed::EmbeddingModel::AllMiniLML12V2),
            "all-MiniLM-L12-v2-q" | "AllMiniLML12V2Q" => {
                Ok(fastembed::EmbeddingModel::AllMiniLML12V2Q)
            }
            "BGE-small-en-v1.5" | "BGESmallENV15" => Ok(fastembed::EmbeddingModel::BGESmallENV15),
            "BGE-base-en-v1.5" | "BGEBaseENV15" => Ok(fastembed::EmbeddingModel::BGEBaseENV15),
            "BGE-large-en-v1.5" | "BGELargeENV15" => Ok(fastembed::EmbeddingModel::BGELargeENV15),
            "nomic-embed-text-v1" | "NomicEmbedTextV1" => {
                Ok(fastembed::EmbeddingModel::NomicEmbedTextV1)
            }
            "nomic-embed-text-v1.5" | "NomicEmbedTextV15" => {
                Ok(fastembed::EmbeddingModel::NomicEmbedTextV15)
            }
            other => Err(Error::Memory(format!(
                "unknown local embedding model: {other}. Use one of: \
                 all-MiniLM-L6-v2, all-MiniLM-L6-v2-q, all-MiniLM-L12-v2, \
                 all-MiniLM-L12-v2-q, BGE-small-en-v1.5, BGE-base-en-v1.5, \
                 BGE-large-en-v1.5, nomic-embed-text-v1, nomic-embed-text-v1.5"
            ))),
        }
    }
}

impl EmbeddingProvider for LocalEmbeddingProvider {
    fn embed(
        &self,
        texts: &[&str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, Error>> + Send + '_>> {
        let input: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let model = self.model.clone();
        Box::pin(async move {
            if input.is_empty() {
                return Ok(vec![]);
            }

            tokio::task::spawn_blocking(move || {
                let mut model = model
                    .lock()
                    .map_err(|e| Error::Memory(format!("embedding model lock poisoned: {e}")))?;
                model
                    .embed(input, None)
                    .map_err(|e| Error::Memory(format!("local embedding failed: {e}")))
            })
            .await
            .map_err(|e| Error::Memory(format!("embedding task panicked: {e}")))?
        })
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::auth::tenant::TenantScope;
    use heartbit_core::memory::embedding::EmbeddingMemory;
    use heartbit_core::memory::in_memory::InMemoryStore;
    use heartbit_core::memory::{Confidentiality, Memory, MemoryEntry, MemoryQuery, MemoryType};
    use std::sync::Arc;

    fn make_entry(id: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: id.into(),
            agent: "test".into(),
            content: content.into(),
            category: "fact".into(),
            tags: vec![],
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
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

    #[test]
    fn local_embedding_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LocalEmbeddingProvider>();
    }

    #[test]
    fn local_embedding_provider_dimension() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        assert!(provider.dimension() > 0, "dimension should be positive");
        // AllMiniLML6V2 produces 384-dim vectors
        assert_eq!(provider.dimension(), 384);
    }

    #[tokio::test]
    async fn local_embedding_generates_vectors() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let result = provider.embed(&["hello world"]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), provider.dimension());
    }

    #[tokio::test]
    async fn local_embedding_batch_embed() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let texts = &["first sentence", "second sentence", "third sentence"];
        let result = provider.embed(texts).await.unwrap();
        assert_eq!(result.len(), 3);
        for vec in &result {
            assert_eq!(vec.len(), provider.dimension());
        }
    }

    #[tokio::test]
    async fn local_embedding_empty_input() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let result = provider.embed(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn local_embedding_with_embedding_memory() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let em = EmbeddingMemory::new(store.clone(), Arc::new(provider));

        let scope = TenantScope::default();
        em.store(
            &scope,
            make_entry("m1", "the quick brown fox jumps over the lazy dog"),
        )
        .await
        .unwrap();

        let results = store
            .recall(
                &scope,
                MemoryQuery {
                    limit: 10,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        let emb = results[0]
            .embedding
            .as_ref()
            .expect("embedding should be populated");
        assert_eq!(emb.len(), 384);
    }

    #[test]
    fn local_embedding_unknown_model_errors() {
        let result = LocalEmbeddingProvider::new(Some("nonexistent"), None);
        assert!(result.is_err());
    }

    #[test]
    fn local_embedding_invalid_model_names_rejected() {
        let invalid = ["gpt-4", "text-embedding-3-small", "", "AllMiniLM"];
        for name in &invalid {
            let result = LocalEmbeddingProvider::new(Some(name), None);
            assert!(
                result.is_err(),
                "'{name}' should not be a valid local model name"
            );
        }
    }

    #[tokio::test]
    async fn local_embedding_unicode_input() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let texts = &[
            "日本語テスト",
            "中文测试",
            "한국어 테스트",
            "Ñoño café résumé",
        ];
        let result = provider.embed(texts).await.unwrap();
        assert_eq!(result.len(), 4);
        for (i, vec) in result.iter().enumerate() {
            assert_eq!(vec.len(), 384, "unicode text {i} wrong dim");
            assert!(
                vec.iter().any(|&v| v != 0.0),
                "unicode text {i} should not be all zeros"
            );
        }
    }

    #[tokio::test]
    async fn local_embedding_very_long_text() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(500);
        let result = provider.embed(&[&long_text]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);
    }

    #[tokio::test]
    async fn local_embedding_special_characters() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let texts = &[
            "!@#$%^&*()",
            "<script>alert('xss')</script>",
            "\u{1F980}\u{1F525}",
        ];
        let result = provider.embed(texts).await.unwrap();
        assert_eq!(result.len(), 3);
        for vec in &result {
            assert_eq!(vec.len(), 384);
        }
    }

    #[tokio::test]
    async fn local_embedding_identical_texts_produce_identical_vectors() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let result = provider
            .embed(&["hello world", "hello world"])
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
        let sim = heartbit_core::memory::hybrid::cosine_similarity(&result[0], &result[1]);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "identical texts should produce identical vectors, got sim={sim:.6}"
        );
    }

    #[tokio::test]
    async fn local_embedding_semantic_similarity() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let texts = &[
            "The cat sat on the mat",
            "A kitten rested on the rug",
            "Quantum physics is fascinating",
        ];
        let result = provider.embed(texts).await.unwrap();

        let sim_similar = heartbit_core::memory::hybrid::cosine_similarity(&result[0], &result[1]);
        let sim_different =
            heartbit_core::memory::hybrid::cosine_similarity(&result[0], &result[2]);

        assert!(
            sim_similar > sim_different,
            "cat/kitten ({sim_similar:.4}) should be more similar than cat/quantum ({sim_different:.4})"
        );
        assert!(
            sim_similar > 0.5,
            "cat/kitten similarity ({sim_similar:.4}) should be > 0.5"
        );
    }

    #[tokio::test]
    async fn local_embedding_vectors_are_normalized() {
        let provider = LocalEmbeddingProvider::new(None, None).unwrap();
        let result = provider.embed(&["test normalization"]).await.unwrap();
        let norm: f32 = result[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.05,
            "vectors should be approximately normalized, got norm={norm:.6}"
        );
    }

    #[tokio::test]
    async fn local_embedding_concurrent_embeds() {
        let provider = std::sync::Arc::new(LocalEmbeddingProvider::new(None, None).unwrap());
        let mut handles = Vec::new();
        for i in 0..5 {
            let p = provider.clone();
            handles.push(tokio::spawn(async move {
                let text = format!("concurrent embedding test number {i}");
                let result = p.embed(&[text.as_str()]).await.unwrap();
                assert_eq!(result.len(), 1);
                assert_eq!(result[0].len(), 384);
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }
    }
}
