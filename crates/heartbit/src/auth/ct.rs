//! Constant-time string and set comparison helpers.
//!
//! Use these when comparing secrets (bearer tokens, signing keys, HMACs)
//! to avoid leaking information through timing side-channels. Comparing
//! two equal-length non-equal strings is constant-time. The length
//! difference itself is intentionally not constant-time: token length is
//! not the secret being protected.

use std::collections::HashSet;

use subtle::ConstantTimeEq;

/// Constant-time string equality.
///
/// Returns `false` immediately if lengths differ; this short-circuit is
/// intentional (length is not the secret).
pub fn ct_eq_str(a: &str, b: &str) -> bool {
    a.len() == b.len() && bool::from(a.as_bytes().ct_eq(b.as_bytes()))
}

/// Constant-time membership test against a `HashSet<String>`.
///
/// Iterates every entry on every call; O(n) by design. Use only for sets
/// small enough that the linear scan is acceptable (bearer-token allow-lists,
/// signing-key sets, etc.). For large sets, use a different approach
/// — constant-time HashSet lookup is not something this helper provides.
pub fn contains(set: &HashSet<String>, candidate: &str) -> bool {
    let mut hit = false;
    for known in set {
        // bitwise-or so every iteration runs regardless of an earlier hit
        hit |= ct_eq_str(known, candidate);
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ct_eq_str_equal_strings() {
        assert!(ct_eq_str("hunter2", "hunter2"));
    }

    #[test]
    fn ct_eq_str_different_same_length() {
        assert!(!ct_eq_str("hunter2", "Hunter2"));
    }

    #[test]
    fn ct_eq_str_different_lengths() {
        assert!(!ct_eq_str("hunter2", "hunter22"));
        assert!(!ct_eq_str("", "x"));
    }

    #[test]
    fn ct_eq_str_empty() {
        assert!(ct_eq_str("", ""));
    }

    #[test]
    fn contains_hits_when_present() {
        let mut s = HashSet::new();
        s.insert("alpha".to_string());
        s.insert("bravo".to_string());
        assert!(contains(&s, "alpha"));
        assert!(contains(&s, "bravo"));
    }

    #[test]
    fn contains_misses_when_absent() {
        let mut s = HashSet::new();
        s.insert("alpha".to_string());
        assert!(!contains(&s, "alphax"));
        assert!(!contains(&s, "alph"));
        assert!(!contains(&s, ""));
    }

    #[test]
    fn contains_empty_set_is_always_false() {
        let s: HashSet<String> = HashSet::new();
        assert!(!contains(&s, "anything"));
    }
}
