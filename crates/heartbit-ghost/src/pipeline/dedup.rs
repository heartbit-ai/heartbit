//! Levenshtein-based candidate dedup. Pure functions; no I/O.

// Task 1 introduces these helpers; Task 3 wires them into run_pipeline.
// Remove this attribute once distinct_indices is called from mod.rs.
#![allow(dead_code)]

/// Drafts with Levenshtein ratio above this threshold are considered
/// near-duplicates and one of the pair is dropped per umbrella spec §6.1.
pub(crate) const LEVENSHTEIN_DUPLICATE_THRESHOLD: f64 = 0.85;

/// Levenshtein distance via standard O(m·n) DP. Distance is in characters
/// (not bytes), so unicode multi-byte sequences count as one each.
pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let m = av.len();
    let n = bv.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if av[i - 1] == bv[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Levenshtein ratio in [0.0, 1.0]. 1.0 = identical; 0.0 = completely different.
/// Defined as `1.0 - distance / max(len_a, len_b)`. Empty-vs-empty = 1.0.
pub(crate) fn levenshtein_ratio(a: &str, b: &str) -> f64 {
    let len_a = a.chars().count();
    let len_b = b.chars().count();
    let max_len = len_a.max(len_b);
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

/// Greedy distinct-set computation. Walks `drafts` in declaration order;
/// each index survives if its Levenshtein ratio is `<= threshold` against
/// every already-surviving index. The lower-indexed of any colliding pair
/// wins (variant 0 takes precedence over variant 1).
pub(crate) fn distinct_indices(drafts: &[&str], threshold: f64) -> Vec<usize> {
    let mut survivors: Vec<usize> = Vec::with_capacity(drafts.len());
    for (i, draft) in drafts.iter().enumerate() {
        let collides = survivors
            .iter()
            .any(|&j| levenshtein_ratio(draft, drafts[j]) > threshold);
        if !collides {
            survivors.push(i);
        }
    }
    survivors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_identical_strings_zero_distance() {
        assert_eq!(levenshtein("hello", "hello"), 0);
        assert_eq!(levenshtein_ratio("hello", "hello"), 1.0);
    }

    #[test]
    fn levenshtein_single_char_diff_one_distance() {
        assert_eq!(levenshtein("hello", "hallo"), 1);
        assert!((levenshtein_ratio("hello", "hallo") - 0.8).abs() < 1e-9);
    }

    #[test]
    fn levenshtein_empty_strings_ratio_is_one() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein_ratio("", ""), 1.0);
    }

    #[test]
    fn levenshtein_handles_unicode_as_chars_not_bytes() {
        // "é" is 2 bytes UTF-8 but 1 char.
        assert_eq!(levenshtein("café", "cafe"), 1);
    }

    #[test]
    fn distinct_indices_all_distinct_keeps_all() {
        let drafts = vec!["alpha is one", "beta is two", "gamma is three"];
        let out = distinct_indices(&drafts, 0.85);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn distinct_indices_two_near_duplicates_keeps_lower_index() {
        // 1 and 2 are identical (ratio = 1.0 > 0.85). 0 is distinct.
        let drafts = vec!["the first draft is long", "alpha", "alpha"];
        let out = distinct_indices(&drafts, 0.85);
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn distinct_indices_three_identical_collapse_to_one() {
        let drafts = vec!["same", "same", "same"];
        let out = distinct_indices(&drafts, 0.85);
        assert_eq!(out, vec![0]);
    }
}
