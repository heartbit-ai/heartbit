//! Convert a multi-tweet draft (single string with `\n\n` separators)
//! into the `Vec<String>` format the `twitter_thread` tool accepts.

/// Split a draft on `\n\n` boundaries, trim each tweet, drop empties.
///
/// Mirrors `pipeline/publish_gate.rs::check_publish_gate`'s splitting
/// rule so the two stay consistent — what passes the gate is what gets
/// posted.
pub fn parse_thread_tweets(draft: &str) -> Vec<String> {
    draft
        .split("\n\n")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_tweet_returns_one_element_vec() {
        let v = parse_thread_tweets("a single tweet");
        assert_eq!(v, vec!["a single tweet".to_string()]);
    }

    #[test]
    fn parse_thread_splits_on_double_newline() {
        let v = parse_thread_tweets("first\n\nsecond\n\nthird");
        assert_eq!(
            v,
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string(),
            ]
        );
    }

    #[test]
    fn parse_thread_trims_whitespace_around_each_tweet() {
        let v = parse_thread_tweets("  first  \n\n  second  ");
        assert_eq!(v, vec!["first".to_string(), "second".to_string()]);
    }

    #[test]
    fn parse_thread_filters_empty_segments() {
        // Triple newline produces an empty middle segment after split.
        let v = parse_thread_tweets("first\n\n\n\nsecond");
        assert_eq!(v, vec!["first".to_string(), "second".to_string()]);
    }
}
