//! Memory consolidation pipeline — clusters and merges episodic memories into semantic summaries.

use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use crate::auth::TenantScope;
use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{CompletionRequest, Message, StopReason, TokenUsage};

use super::{Confidentiality, Memory, MemoryEntry, MemoryQuery, MemoryType};

/// Outcome of a consolidation pass.
///
/// `clusters_merged + clusters_skipped == clusters_identified` (clusters whose
/// member count >= `min_cluster_size`). Skips happen when the per-cluster LLM
/// summary returns `StopReason::MaxTokens` (raise the budget via
/// [`ConsolidationPipeline::with_summary_max_tokens`]) or when the LLM call
/// itself errors.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsolidationResult {
    /// Clusters that were summarised, stored as Semantic, and whose source
    /// episodic entries were deleted.
    pub clusters_merged: usize,
    /// Clusters identified by the keyword-overlap pass but skipped during
    /// summarisation. Each skip is also surfaced via `tracing::warn!`.
    pub clusters_skipped: usize,
    /// Total source entries that contributed to merged clusters.
    pub entries_consolidated: usize,
    /// LLM token usage accumulated across all summary attempts (including
    /// the ones that ended up skipped).
    pub token_usage: TokenUsage,
}

/// Default per-cluster summary budget. Mirrors `summary_max_tokens` so callers
/// who want to raise / lower it know the baseline.
pub const DEFAULT_SUMMARY_MAX_TOKENS: u32 = 512;

/// Consolidation pipeline that clusters related memories and merges them.
///
/// At session end, finds clusters of related memories (by keyword overlap),
/// generates a consolidated summary for each cluster, stores the result as
/// `MemoryType::Semantic`, and deletes the originals.
pub struct ConsolidationPipeline<P: LlmProvider> {
    memory: Arc<dyn Memory>,
    provider: Arc<P>,
    agent_name: String,
    /// Minimum Jaccard similarity for clustering. Default: 0.3.
    similarity_threshold: f64,
    /// Minimum cluster size to consolidate. Default: 2.
    min_cluster_size: usize,
    /// Per-cluster summary token budget passed to the provider. Default:
    /// [`DEFAULT_SUMMARY_MAX_TOKENS`]. When the LLM trips this cap (i.e. it
    /// returns `StopReason::MaxTokens`) the cluster is skipped and reported
    /// in [`ConsolidationResult::clusters_skipped`].
    summary_max_tokens: u32,
}

impl<P: LlmProvider> ConsolidationPipeline<P> {
    /// Construct a pipeline operating on `memory` for `agent_name` via the given LLM `provider`.
    pub fn new(memory: Arc<dyn Memory>, provider: Arc<P>, agent_name: impl Into<String>) -> Self {
        Self {
            memory,
            provider,
            agent_name: agent_name.into(),
            similarity_threshold: 0.3,
            min_cluster_size: 2,
            summary_max_tokens: DEFAULT_SUMMARY_MAX_TOKENS,
        }
    }

    /// Override the Jaccard similarity threshold used to cluster entries (default: 0.3).
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Override the minimum cluster size considered for consolidation (default: 2).
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }

    /// Override the per-cluster summary token budget (default: 512).
    ///
    /// Verbose clusters can trip the default cap and end up silently skipped;
    /// raise this when consolidating long episodic transcripts. The cap maps
    /// directly to `CompletionRequest::max_tokens` for the summary LLM call.
    pub fn with_summary_max_tokens(mut self, max_tokens: u32) -> Self {
        self.summary_max_tokens = max_tokens;
        self
    }

    /// Run the consolidation pipeline within the given tenant scope.
    ///
    /// Returns `(clusters_merged, total_entries_consolidated, token_usage)`
    /// for backward compatibility. Use [`ConsolidationPipeline::run_detailed`]
    /// when callers need the `clusters_skipped` count to detect partial runs.
    pub async fn run(&self, scope: &TenantScope) -> Result<(usize, usize, TokenUsage), Error> {
        let result = self.run_detailed(scope).await?;
        Ok((
            result.clusters_merged,
            result.entries_consolidated,
            result.token_usage,
        ))
    }

    /// Run the consolidation pipeline and return a structured outcome,
    /// including the count of clusters that were identified but skipped
    /// during summarisation.
    pub async fn run_detailed(&self, scope: &TenantScope) -> Result<ConsolidationResult, Error> {
        // 1. Recall all episodic memories
        let entries = self
            .memory
            .recall(
                scope,
                MemoryQuery {
                    agent: Some(self.agent_name.clone()),
                    memory_type: Some(MemoryType::Episodic),
                    limit: 0,
                    ..Default::default()
                },
            )
            .await?;

        if entries.len() < self.min_cluster_size {
            return Ok(ConsolidationResult::default());
        }

        // 2. Cluster by keyword overlap (Jaccard similarity)
        let clusters =
            cluster_by_keywords(&entries, self.similarity_threshold, self.min_cluster_size);

        let mut total_usage = TokenUsage::default();
        let mut clusters_merged = 0;
        let mut clusters_skipped = 0;
        let mut entries_consolidated = 0;

        // 3. For each cluster, generate summary and consolidate
        for cluster in &clusters {
            let content_parts: Vec<String> =
                cluster.iter().map(|e| format!("- {}", e.content)).collect();
            let combined = content_parts.join("\n");

            // Generate consolidated summary via LLM
            let (summary, usage) = match self.summarize_cluster(&combined).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(
                        agent = %self.agent_name,
                        cluster_size = cluster.len(),
                        error = %e,
                        "failed to summarize cluster, skipping"
                    );
                    clusters_skipped += 1;
                    continue;
                }
            };
            total_usage += usage;

            let Some(summary_text) = summary else {
                let first_id = cluster.first().map(|e| e.id.as_str()).unwrap_or("");
                tracing::warn!(
                    agent = %self.agent_name,
                    cluster_size = cluster.len(),
                    first_entry_id = %first_id,
                    summary_max_tokens = self.summary_max_tokens,
                    "consolidation summary hit max_tokens, cluster skipped — \
                     raise the budget via with_summary_max_tokens"
                );
                clusters_skipped += 1;
                continue;
            };

            // Collect source IDs and merge keywords/tags
            let source_ids: Vec<String> = cluster.iter().map(|e| e.id.clone()).collect();
            let mut all_keywords: HashSet<String> = HashSet::new();
            let mut all_tags: HashSet<String> = HashSet::new();
            let mut max_importance: u8 = 1;
            for e in cluster {
                all_keywords.extend(e.keywords.iter().cloned());
                all_tags.extend(e.tags.iter().cloned());
                max_importance = max_importance.max(e.importance);
            }

            // Store consolidated entry
            let now = Utc::now();
            let new_id = Uuid::new_v4().to_string();
            let consolidated = MemoryEntry {
                id: new_id,
                agent: self.agent_name.clone(),
                content: summary_text,
                category: "fact".into(),
                tags: all_tags.into_iter().collect(),
                created_at: now,
                last_accessed: now,
                access_count: 0,
                importance: max_importance,
                memory_type: MemoryType::Semantic,
                keywords: all_keywords.into_iter().collect(),
                summary: None,
                strength: 1.0,
                related_ids: vec![],
                source_ids: source_ids.clone(),
                embedding: None,
                confidentiality: Confidentiality::default(),
                author_user_id: None,
                author_tenant_id: None,
            };

            self.memory.store(scope, consolidated).await?;

            // Delete originals
            for id in &source_ids {
                let _ = self.memory.forget(scope, id).await;
            }

            clusters_merged += 1;
            entries_consolidated += cluster.len();
        }

        Ok(ConsolidationResult {
            clusters_merged,
            clusters_skipped,
            entries_consolidated,
            token_usage: total_usage,
        })
    }

    async fn summarize_cluster(
        &self,
        content: &str,
    ) -> Result<(Option<String>, TokenUsage), Error> {
        let request = CompletionRequest {
            system: "You are a memory consolidation assistant. Combine the following related \
                      memory entries into a single concise summary that preserves all key facts. \
                      Be specific and factual."
                .into(),
            messages: vec![Message::user(content.to_string())],
            tools: vec![],
            max_tokens: self.summary_max_tokens,
            tool_choice: None,
            reasoning_effort: None,
        };

        let response = self.provider.complete(request).await?;
        let usage = response.usage;
        if response.stop_reason == StopReason::MaxTokens {
            return Ok((None, usage));
        }
        Ok((Some(response.text()), usage))
    }
}

/// Cluster entries by keyword overlap using greedy single-linkage clustering.
///
/// Two entries are considered related if the Jaccard similarity of their
/// keyword sets exceeds `threshold`. Returns clusters with at least
/// `min_size` members.
pub fn cluster_by_keywords(
    entries: &[MemoryEntry],
    threshold: f64,
    min_size: usize,
) -> Vec<Vec<&MemoryEntry>> {
    let n = entries.len();
    let mut assigned = vec![false; n];
    let mut clusters: Vec<Vec<&MemoryEntry>> = Vec::new();

    for i in 0..n {
        if assigned[i] {
            continue;
        }
        let mut cluster = vec![&entries[i]];
        assigned[i] = true;

        for j in (i + 1)..n {
            if assigned[j] {
                continue;
            }
            // Check if j is similar to any member of the current cluster
            let similar = cluster.iter().any(|member| {
                jaccard_similarity(&member.keywords, &entries[j].keywords) >= threshold
            });
            if similar {
                cluster.push(&entries[j]);
                assigned[j] = true;
            }
        }

        if cluster.len() >= min_size {
            clusters.push(cluster);
        }
    }

    clusters
}

/// Jaccard similarity between two keyword sets.
pub(crate) fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<&str> = a.iter().map(String::as_str).collect();
    let set_b: HashSet<&str> = b.iter().map(String::as_str).collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry_with_keywords(id: &str, keywords: Vec<String>) -> MemoryEntry {
        let now = Utc::now();
        MemoryEntry {
            id: id.into(),
            agent: "test".into(),
            content: format!("content for {id}"),
            category: "fact".into(),
            tags: vec![],
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance: 5,
            memory_type: MemoryType::Episodic,
            keywords,
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
    fn jaccard_identical_sets() {
        let a = vec!["rust".into(), "fast".into()];
        let b = vec!["fast".into(), "rust".into()];
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a = vec!["rust".into()];
        let b = vec!["python".into()];
        assert!((jaccard_similarity(&a, &b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a = vec!["rust".into(), "fast".into()];
        let b = vec!["rust".into(), "safe".into()];
        // intersection=1 (rust), union=3 (rust, fast, safe) → 1/3
        assert!((jaccard_similarity(&a, &b) - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_empty_sets() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec![];
        assert!((jaccard_similarity(&a, &b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cluster_by_keyword_overlap() {
        let entries = vec![
            make_entry_with_keywords("m1", vec!["rust".into(), "performance".into()]),
            make_entry_with_keywords("m2", vec!["rust".into(), "safety".into()]),
            make_entry_with_keywords("m3", vec!["python".into(), "ml".into()]),
            make_entry_with_keywords("m4", vec!["python".into(), "data".into()]),
        ];

        // threshold 0.3 → m1 & m2 share "rust" (jaccard=1/3 ≈ 0.33 ≥ 0.3),
        // m3 & m4 share "python" (jaccard=1/3 ≈ 0.33 ≥ 0.3)
        let clusters = cluster_by_keywords(&entries, 0.3, 2);
        assert_eq!(clusters.len(), 2, "should have 2 clusters");
    }

    #[test]
    fn cluster_no_overlap() {
        let entries = vec![
            make_entry_with_keywords("m1", vec!["a".into()]),
            make_entry_with_keywords("m2", vec!["b".into()]),
            make_entry_with_keywords("m3", vec!["c".into()]),
        ];

        let clusters = cluster_by_keywords(&entries, 0.3, 2);
        assert!(clusters.is_empty(), "no clusters when no overlap");
    }

    #[test]
    fn cluster_min_size_respected() {
        let entries = vec![
            make_entry_with_keywords("m1", vec!["rust".into()]),
            make_entry_with_keywords("m2", vec!["python".into()]),
        ];

        // Each entry is alone, min_size=2 → no clusters
        let clusters = cluster_by_keywords(&entries, 0.3, 2);
        assert!(clusters.is_empty());
    }

    #[test]
    fn cluster_preserves_source_ids() {
        let entries = vec![
            make_entry_with_keywords("m1", vec!["rust".into(), "perf".into()]),
            make_entry_with_keywords("m2", vec!["rust".into(), "speed".into()]),
        ];

        let clusters = cluster_by_keywords(&entries, 0.3, 2);
        assert_eq!(clusters.len(), 1);
        let ids: Vec<&str> = clusters[0].iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"m1"));
        assert!(ids.contains(&"m2"));
    }

    // -----------------------------------------------------------------------
    // run_detailed: skip accounting + summary token budget (Issue #8)
    // -----------------------------------------------------------------------

    use crate::llm::types::{CompletionRequest, CompletionResponse, ContentBlock};
    use crate::memory::in_memory::InMemoryStore;
    use std::sync::Mutex;

    /// Provider that always responds with `StopReason::MaxTokens` and records
    /// the `max_tokens` budget it received.
    struct MaxTokensProvider {
        observed_max_tokens: Mutex<Vec<u32>>,
    }

    impl LlmProvider for MaxTokensProvider {
        async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
            self.observed_max_tokens
                .lock()
                .expect("lock")
                .push(request.max_tokens);
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "partial".into(),
                }],
                stop_reason: StopReason::MaxTokens,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            })
        }
    }

    #[tokio::test]
    async fn run_detailed_reports_clusters_skipped_on_max_tokens() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let scope = TenantScope::default();

        for entry in [
            make_entry_with_keywords("a1", vec!["rust".into(), "perf".into()]),
            make_entry_with_keywords("a2", vec!["rust".into(), "speed".into()]),
            make_entry_with_keywords("b1", vec!["python".into(), "ml".into()]),
            make_entry_with_keywords("b2", vec!["python".into(), "data".into()]),
        ] {
            store.store(&scope, entry).await.unwrap();
        }

        let provider = Arc::new(MaxTokensProvider {
            observed_max_tokens: Mutex::new(Vec::new()),
        });

        let pipeline = ConsolidationPipeline::new(store.clone(), provider.clone(), "test")
            .with_min_cluster_size(2);

        let result = pipeline.run_detailed(&scope).await.unwrap();

        // Two clusters identified, both summaries trip MaxTokens, so both
        // should be reported as skipped instead of being silently dropped.
        assert_eq!(result.clusters_merged, 0);
        assert_eq!(result.clusters_skipped, 2);
        assert_eq!(result.entries_consolidated, 0);

        // Default budget is DEFAULT_SUMMARY_MAX_TOKENS.
        let observed = provider.observed_max_tokens.lock().expect("lock").clone();
        assert!(
            observed.iter().all(|m| *m == DEFAULT_SUMMARY_MAX_TOKENS),
            "default summary max_tokens = {DEFAULT_SUMMARY_MAX_TOKENS}, observed: {observed:?}"
        );
    }

    #[tokio::test]
    async fn run_uses_configured_summary_max_tokens() {
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let scope = TenantScope::default();

        for entry in [
            make_entry_with_keywords("a1", vec!["rust".into()]),
            make_entry_with_keywords("a2", vec!["rust".into()]),
        ] {
            store.store(&scope, entry).await.unwrap();
        }

        let provider = Arc::new(MaxTokensProvider {
            observed_max_tokens: Mutex::new(Vec::new()),
        });

        let pipeline = ConsolidationPipeline::new(store.clone(), provider.clone(), "test")
            .with_min_cluster_size(2)
            .with_summary_max_tokens(4096);

        let _ = pipeline.run_detailed(&scope).await.unwrap();

        let observed = provider.observed_max_tokens.lock().expect("lock").clone();
        assert_eq!(
            observed,
            vec![4096],
            "with_summary_max_tokens must be passed to the provider"
        );
    }

    #[tokio::test]
    async fn run_tuple_keeps_backward_compatible_shape() {
        // The original `run()` returns a 3-tuple — ensure we kept that
        // shape so existing callers don't break.
        let store: Arc<dyn Memory> = Arc::new(InMemoryStore::new());
        let scope = TenantScope::default();
        let provider = Arc::new(MaxTokensProvider {
            observed_max_tokens: Mutex::new(Vec::new()),
        });
        let pipeline = ConsolidationPipeline::new(store, provider, "test");

        let (merged, entries, _usage) = pipeline.run(&scope).await.unwrap();
        assert_eq!(merged, 0);
        assert_eq!(entries, 0);
    }
}
