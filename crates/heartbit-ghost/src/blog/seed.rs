//! X-derived blog topic seed selection.
//!
//! Picks the highest-engagement X post from the prior `lookback_days`
//! window as the seed for this week's blog essay. The seed text becomes
//! the topic context fed to the researcher; the essay then EXPANDS on
//! that topic rather than quoting the original (the blog is its own
//! publishing surface, not a quote-tweet aggregator).

use chrono::{DateTime, Duration, Utc};

use crate::posts::TopPostsProvider;

/// A topic seed for the blog pipeline.
#[derive(Debug, Clone)]
pub struct BlogSeed {
    /// Text of the source X post (first tweet for threads).
    pub text: String,
    /// Public URL of the source X post.
    pub source_url: String,
    /// Tweet ID for traceability.
    pub source_tweet_id: String,
    /// When the source X post was published.
    pub source_posted_at: DateTime<Utc>,
    /// Composite engagement score of the source post.
    pub engagement_score: f64,
    /// Operator-facing summary of why this seed was chosen.
    pub rationale: String,
}

/// Errors returned by [`select_blog_seed`].
#[derive(Debug, thiserror::Error)]
pub enum SeedError {
    /// No high-engagement post was found within the lookback window.
    /// Returned when `top_n` returns no results, the top result is
    /// outside the window, or the top result has zero engagement.
    /// The handler should record `BlogOutcome::NoSeed` and not invoke
    /// the writer pipeline.
    #[error("no eligible seed within {lookback_days} day window")]
    NoEligibleSeed {
        /// The lookback window (in days) that was searched.
        lookback_days: i64,
    },
}

/// Default `n` for `select_blog_seed` callers.
pub const DEFAULT_TOP_N: usize = 10;

/// Pick the highest-engagement post from the prior `lookback_days` as
/// the topic seed.
///
/// `n` is the number of top posts to consider from the provider. The
/// function walks them in descending engagement order and picks the
/// first one whose `posted_at` is within the window. `n=10` is a
/// reasonable default — see [`DEFAULT_TOP_N`].
pub async fn select_blog_seed(
    provider: &dyn TopPostsProvider,
    n: usize,
    lookback_days: i64,
    now: DateTime<Utc>,
) -> Result<BlogSeed, SeedError> {
    if lookback_days <= 0 {
        return Err(SeedError::NoEligibleSeed { lookback_days });
    }
    let cutoff = now - Duration::days(lookback_days);
    let candidates = provider
        .top_n(n)
        .await
        .map_err(|_| SeedError::NoEligibleSeed { lookback_days })?;

    for c in candidates {
        if c.posted_at < cutoff {
            continue;
        }
        if c.engagement_score <= 0.0 {
            continue;
        }
        let source_url = format!("https://twitter.com/i/web/status/{}", c.tweet_id);
        let rationale = format!(
            "Top-engagement X post in the last {lookback_days} days (score {:.2}, posted {}).",
            c.engagement_score,
            c.posted_at.format("%Y-%m-%d")
        );
        return Ok(BlogSeed {
            text: c.text.clone(),
            source_url,
            source_tweet_id: c.tweet_id.clone(),
            source_posted_at: c.posted_at,
            engagement_score: c.engagement_score,
            rationale,
        });
    }
    Err(SeedError::NoEligibleSeed { lookback_days })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posts::{TopPost, TopPostsFut};
    use chrono::TimeZone;

    struct MockProvider {
        posts: Vec<TopPost>,
    }

    impl TopPostsProvider for MockProvider {
        fn top_n<'a>(&'a self, n: usize) -> TopPostsFut<'a> {
            let posts: Vec<TopPost> = self.posts.iter().take(n).cloned().collect();
            Box::pin(async move { Ok(posts) })
        }
    }

    fn now_fixture() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 16, 12, 0, 0).unwrap()
    }

    fn post(id: &str, score: f64, days_ago: i64) -> TopPost {
        TopPost {
            tweet_id: id.into(),
            text: format!("Post body for {id}."),
            posted_at: now_fixture() - Duration::days(days_ago),
            engagement_score: score,
        }
    }

    #[tokio::test]
    async fn picks_highest_engagement_within_window() {
        let provider = MockProvider {
            posts: vec![
                post("100", 5.0, 2),
                post("101", 3.0, 5),
                post("102", 1.0, 1),
            ],
        };
        let seed = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap();
        assert_eq!(seed.source_tweet_id, "100");
        assert_eq!(seed.engagement_score, 5.0);
        assert!(seed.source_url.contains("100"));
        assert!(seed.rationale.contains("score 5.00"));
    }

    #[tokio::test]
    async fn skips_posts_outside_window() {
        let provider = MockProvider {
            posts: vec![
                post("100", 10.0, 30), // outside 7-day window
                post("101", 5.0, 3),   // inside
            ],
        };
        let seed = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap();
        assert_eq!(seed.source_tweet_id, "101");
    }

    #[tokio::test]
    async fn errors_when_no_eligible_posts() {
        let provider = MockProvider {
            posts: vec![post("100", 10.0, 30), post("101", 5.0, 21)],
        };
        let err = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            SeedError::NoEligibleSeed { lookback_days: 7 }
        ));
    }

    #[tokio::test]
    async fn skips_zero_engagement() {
        let provider = MockProvider {
            posts: vec![post("100", 0.0, 2), post("101", 2.5, 3)],
        };
        let seed = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap();
        assert_eq!(seed.source_tweet_id, "101");
    }

    #[tokio::test]
    async fn empty_provider_returns_no_eligible_seed() {
        let provider = MockProvider { posts: vec![] };
        let err = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap_err();
        assert!(matches!(err, SeedError::NoEligibleSeed { .. }));
    }

    #[tokio::test]
    async fn negative_lookback_returns_no_seed() {
        let provider = MockProvider {
            posts: vec![post("100", 5.0, 1)],
        };
        let err = select_blog_seed(&provider, 10, 0, now_fixture())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            SeedError::NoEligibleSeed { lookback_days: 0 }
        ));
    }
}
