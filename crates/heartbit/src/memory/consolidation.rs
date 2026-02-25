use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{CompletionRequest, Message, StopReason, TokenUsage};

use super::{Confidentiality, Memory, MemoryEntry, MemoryQuery, MemoryType};

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
}

impl<P: LlmProvider> ConsolidationPipeline<P> {
    pub fn new(memory: Arc<dyn Memory>, provider: Arc<P>, agent_name: impl Into<String>) -> Self {
        Self {
            memory,
            provider,
            agent_name: agent_name.into(),
            similarity_threshold: 0.3,
            min_cluster_size: 2,
        }
    }

    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }

    /// Run the consolidation pipeline.
    ///
    /// Returns `(clusters_merged, total_entries_consolidated, token_usage)`.
    pub async fn run(&self) -> Result<(usize, usize, TokenUsage), Error> {
        // 1. Recall all episodic memories
        let entries = self
            .memory
            .recall(MemoryQuery {
                agent: Some(self.agent_name.clone()),
                memory_type: Some(MemoryType::Episodic),
                limit: 0,
                ..Default::default()
            })
            .await?;

        if entries.len() < self.min_cluster_size {
            return Ok((0, 0, TokenUsage::default()));
        }

        // 2. Cluster by keyword overlap (Jaccard similarity)
        let clusters =
            cluster_by_keywords(&entries, self.similarity_threshold, self.min_cluster_size);

        let mut total_usage = TokenUsage::default();
        let mut clusters_merged = 0;
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
                        error = %e,
                        "failed to summarize cluster, skipping"
                    );
                    continue;
                }
            };
            total_usage += usage;

            let Some(summary_text) = summary else {
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
            };

            self.memory.store(consolidated).await?;

            // Delete originals
            for id in &source_ids {
                let _ = self.memory.forget(id).await;
            }

            clusters_merged += 1;
            entries_consolidated += cluster.len();
        }

        Ok((clusters_merged, entries_consolidated, total_usage))
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
            max_tokens: 512,
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
}
