use std::collections::HashMap;

/// Reciprocal Rank Fusion (RRF) combines ranked lists from multiple retrieval
/// strategies into a single fused ranking.
///
/// For each document appearing in any list:
///   `score(d) = Σ 1 / (k + rank_i(d))`
///
/// where `k` is a smoothing constant (default: 50) and `rank_i(d)` is the
/// 1-based rank of document `d` in list `i`. Documents not present in a list
/// are skipped for that list's contribution.
///
/// Returns `(id, fused_score)` pairs sorted by descending fused score.
pub fn rrf_fuse(
    bm25_ranked: &[(&str, f64)],
    vector_ranked: &[(&str, f64)],
    k: usize,
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (rank, (id, _score)) in bm25_ranked.iter().enumerate() {
        *scores.entry(id.to_string()).or_default() += 1.0 / (k as f64 + (rank + 1) as f64);
    }

    for (rank, (id, _score)) in vector_ranked.iter().enumerate() {
        *scores.entry(id.to_string()).or_default() += 1.0 / (k as f64 + (rank + 1) as f64);
    }

    let mut results: Vec<(String, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if either vector is empty or has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f64;
    let mut mag_a = 0.0_f64;
    let mut mag_b = 0.0_f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }

    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < f64::EPSILON {
        return 0.0;
    }

    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_fusion_combines_both_sources() {
        let bm25 = vec![("d1", 10.0), ("d2", 5.0), ("d3", 1.0)];
        let vector = vec![("d2", 0.95), ("d3", 0.8), ("d4", 0.7)];

        let fused = rrf_fuse(&bm25, &vector, 50);
        let ids: Vec<&str> = fused.iter().map(|(id, _)| id.as_str()).collect();

        // d2 appears in both lists → should rank highest
        assert_eq!(ids[0], "d2");
        // All 4 documents should appear
        assert_eq!(fused.len(), 4);
    }

    #[test]
    fn rrf_fusion_bm25_only_fallback() {
        let bm25 = vec![("d1", 10.0), ("d2", 5.0)];
        let vector: Vec<(&str, f64)> = vec![];

        let fused = rrf_fuse(&bm25, &vector, 50);
        assert_eq!(fused.len(), 2);
        // Ranking preserved from BM25
        assert_eq!(fused[0].0, "d1");
        assert_eq!(fused[1].0, "d2");
    }

    #[test]
    fn rrf_fusion_vector_only() {
        let bm25: Vec<(&str, f64)> = vec![];
        let vector = vec![("d1", 0.99), ("d2", 0.8)];

        let fused = rrf_fuse(&bm25, &vector, 50);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, "d1");
        assert_eq!(fused[1].0, "d2");
    }

    #[test]
    fn rrf_fusion_both_empty() {
        let fused = rrf_fuse(&[], &[], 50);
        assert!(fused.is_empty());
    }

    #[test]
    fn rrf_fusion_same_document_in_both() {
        let bm25 = vec![("d1", 5.0)];
        let vector = vec![("d1", 0.9)];

        let fused = rrf_fuse(&bm25, &vector, 50);
        assert_eq!(fused.len(), 1);
        // Score = 1/(50+1) + 1/(50+1) = 2/51
        let expected = 2.0 / 51.0;
        assert!((fused[0].1 - expected).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert!(sim.abs() < f64::EPSILON);
    }

    #[test]
    fn cosine_similarity_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < f64::EPSILON);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < f64::EPSILON);
    }
}
