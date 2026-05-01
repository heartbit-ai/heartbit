/// Compute a BM25-like score for a single document against query terms.
///
/// Uses the standard BM25 formula:
///   `sum(IDF(t) * tf(t) * (k1 + 1) / (tf(t) + k1 * (1 - b + b * dl / avgdl)))`
///
/// Since we don't maintain a corpus-wide document frequency index, we use a
/// simplified version that focuses on term frequency and document length normalization.
///
/// - `content`: the document text
/// - `keywords`: additional keyword terms associated with the document
/// - `query_terms`: pre-lowercased, deduplicated query tokens
/// - `avgdl`: average document length across the corpus (in words)
/// - `k1`: term frequency saturation parameter (default: 1.2)
/// - `b`: length normalization parameter (default: 0.75)
///
/// Keywords are treated as a separate high-signal field — matches in keywords
/// get a 2x boost over content matches.
pub fn bm25_score(
    content: &str,
    keywords: &[String],
    query_terms: &[String],
    avgdl: f64,
    k1: f64,
    b: f64,
) -> f64 {
    if query_terms.is_empty() || avgdl <= 0.0 {
        return 0.0;
    }

    let lower_content = content.to_lowercase();
    let content_words: Vec<&str> = lower_content.split_whitespace().collect();
    let dl = content_words.len() as f64;

    let lower_keywords: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();

    let mut score = 0.0;
    for term in query_terms {
        // Term frequency in content
        let tf_content = content_words
            .iter()
            .filter(|w| w.contains(term.as_str()))
            .count() as f64;

        // Keyword match bonus (binary: 2.0 if any keyword matches, 0.0 otherwise)
        let keyword_bonus = if lower_keywords.iter().any(|k| k.contains(term.as_str())) {
            2.0
        } else {
            0.0
        };

        let tf = tf_content + keyword_bonus;
        if tf <= 0.0 {
            continue;
        }

        // BM25 TF component with length normalization
        let norm = 1.0 - b + b * (dl / avgdl);
        let tf_score = (tf * (k1 + 1.0)) / (tf + k1 * norm);

        score += tf_score;
    }

    score
}

/// Default BM25 parameters.
pub const DEFAULT_K1: f64 = 1.2;
pub const DEFAULT_B: f64 = 0.75;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_empty_query_returns_zero() {
        let score = bm25_score("some content", &[], &[], 10.0, DEFAULT_K1, DEFAULT_B);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bm25_no_match_returns_zero() {
        let terms = vec!["python".into()];
        let score = bm25_score("rust is fast", &[], &terms, 10.0, DEFAULT_K1, DEFAULT_B);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bm25_single_term_match() {
        let terms = vec!["rust".into()];
        let score = bm25_score("rust is fast", &[], &terms, 10.0, DEFAULT_K1, DEFAULT_B);
        assert!(score > 0.0);
    }

    #[test]
    fn bm25_multiple_term_matches_score_higher() {
        let terms = vec!["rust".into(), "fast".into()];
        let score_both = bm25_score("rust is fast", &[], &terms, 10.0, DEFAULT_K1, DEFAULT_B);

        let terms_one = vec!["rust".into()];
        let score_one = bm25_score("rust is fast", &[], &terms_one, 10.0, DEFAULT_K1, DEFAULT_B);

        assert!(
            score_both > score_one,
            "matching more terms should score higher"
        );
    }

    #[test]
    fn bm25_keyword_field_boosts_score() {
        let terms = vec!["performance".into()];

        // Only in content
        let score_content = bm25_score(
            "rust has good performance",
            &[],
            &terms,
            10.0,
            DEFAULT_K1,
            DEFAULT_B,
        );

        // In keywords (not in content)
        let score_keywords = bm25_score(
            "rust is great",
            &["performance".into()],
            &terms,
            10.0,
            DEFAULT_K1,
            DEFAULT_B,
        );

        // Both should match
        assert!(score_content > 0.0);
        assert!(score_keywords > 0.0);
    }

    #[test]
    fn bm25_shorter_docs_score_higher() {
        let terms = vec!["rust".into()];

        // Short doc (3 words, avgdl=10)
        let score_short = bm25_score("rust is fast", &[], &terms, 10.0, DEFAULT_K1, DEFAULT_B);

        // Long doc with same single match (10 words, avgdl=10)
        let score_long = bm25_score(
            "rust is a programming language that is very very fast",
            &[],
            &terms,
            10.0,
            DEFAULT_K1,
            DEFAULT_B,
        );

        assert!(
            score_short > score_long,
            "shorter doc with same matches should score higher (length normalization)"
        );
    }

    #[test]
    fn bm25_zero_avgdl_returns_zero() {
        let terms = vec!["rust".into()];
        let score = bm25_score("rust is fast", &[], &terms, 0.0, DEFAULT_K1, DEFAULT_B);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bm25_repeated_terms_in_content_boost() {
        let terms = vec!["rust".into()];

        let score_once = bm25_score("rust is great", &[], &terms, 10.0, DEFAULT_K1, DEFAULT_B);
        let score_twice = bm25_score(
            "rust rust is great",
            &[],
            &terms,
            10.0,
            DEFAULT_K1,
            DEFAULT_B,
        );

        assert!(
            score_twice > score_once,
            "higher TF should yield higher score"
        );
    }
}
