//! Anti-spam guards for the reply pipeline. Each rule can short-circuit
//! mention processing before the LLM is consulted. See spec §8 for the
//! full rule set + thresholds.

use chrono::{DateTime, Duration, Utc};

use super::{Mention, MentionerContext};

/// Reason a mention was skipped by [`SpamGuard::should_skip`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// Mention authored by the operator's own user_id (no self-replies).
    SelfReply,
    /// Parent tweet is older than `stale_parent_after_days`.
    StaleParent,
    /// Mentioner has both low followers AND short text — likely spam.
    LowEffortSpam,
    /// We've already replied to this author the maximum allowed times in the rate-limit window.
    PerAuthorRateLimit,
    /// Mention text is too short (fewer than `min_engagement_chars` alphanumerics).
    TooShortToEngage,
}

/// Configuration thresholds for [`SpamGuard`].
#[derive(Debug, Clone)]
pub struct SpamGuardConfig {
    /// Operator's own user_id — replies from this id are skipped.
    pub operator_user_id: String,
    /// Skip when the parent tweet is older than this. Default 7 days.
    pub stale_parent_after_days: i64,
    /// Low-follower threshold for the spam guard.
    pub low_follower_threshold: u64,
    /// Short-text threshold (chars) coupled with low-follower for spam.
    pub low_effort_short_text_chars: usize,
    /// Per-author rate limit window, in hours.
    pub per_author_window_hours: i64,
    /// Max replies allowed to one author per window.
    pub per_author_max_replies: usize,
    /// Minimum non-whitespace alphanumeric chars to be worth engaging.
    pub min_engagement_chars: usize,
}

impl SpamGuardConfig {
    /// Sensible production defaults parameterized by the operator's user_id.
    pub fn defaults_for(operator_user_id: impl Into<String>) -> Self {
        Self {
            operator_user_id: operator_user_id.into(),
            stale_parent_after_days: 7,
            low_follower_threshold: 5,
            low_effort_short_text_chars: 30,
            per_author_window_hours: 24,
            per_author_max_replies: 3,
            min_engagement_chars: 3,
        }
    }
}

/// Stateless evaluator that decides whether a mention should be skipped.
pub struct SpamGuard {
    cfg: SpamGuardConfig,
}

impl SpamGuard {
    /// Construct from a config.
    pub fn new(cfg: SpamGuardConfig) -> Self {
        Self { cfg }
    }

    /// Returns `Some(reason)` if the mention should be skipped, `None`
    /// to proceed. Evaluates rules in fail-fast order: SelfReply,
    /// StaleParent, LowEffortSpam, PerAuthorRateLimit, TooShortToEngage.
    pub fn should_skip(
        &self,
        mention: &Mention,
        parent_posted_at: Option<DateTime<Utc>>,
        mentioner: Option<&MentionerContext>,
        replies_to_author_recent: usize,
        now: DateTime<Utc>,
    ) -> Option<SkipReason> {
        // 1. Self-reply.
        if mention.author_id == self.cfg.operator_user_id {
            return Some(SkipReason::SelfReply);
        }
        // 2. Stale parent.
        if let Some(p) = parent_posted_at
            && p < now - Duration::days(self.cfg.stale_parent_after_days)
        {
            return Some(SkipReason::StaleParent);
        }
        // 3. Low-follower spam (BOTH signals required).
        if let Some(ctx) = mentioner
            && let Some(fc) = ctx.follower_count
            && fc < self.cfg.low_follower_threshold
            && mention.text.len() < self.cfg.low_effort_short_text_chars
        {
            return Some(SkipReason::LowEffortSpam);
        }
        // 4. Per-author rate limit.
        if replies_to_author_recent >= self.cfg.per_author_max_replies {
            return Some(SkipReason::PerAuthorRateLimit);
        }
        // 5. Too short to engage.
        let alnum_count = mention.text.chars().filter(|c| c.is_alphanumeric()).count();
        if alnum_count < self.cfg.min_engagement_chars {
            return Some(SkipReason::TooShortToEngage);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixture_mention(text: &str, author_id: &str) -> Mention {
        Mention {
            id: "m1".into(),
            text: text.into(),
            author_id: author_id.into(),
            author_handle: "x".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: Some("p1".into()),
        }
    }

    fn fixture_ctx(followers: u64) -> MentionerContext {
        MentionerContext {
            handle: "x".into(),
            bio: None,
            recent_tweets: vec![],
            follower_count: Some(followers),
        }
    }

    #[test]
    fn self_reply_skips() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("12345"));
        let m = fixture_mention("hello", "12345");
        assert_eq!(
            guard.should_skip(&m, None, None, 0, Utc::now()),
            Some(SkipReason::SelfReply)
        );
    }

    #[test]
    fn stale_parent_skips() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("op"));
        let m = fixture_mention("hi there long enough", "other");
        let stale = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let now = Utc.with_ymd_and_hms(2026, 5, 1, 0, 0, 0).unwrap();
        assert_eq!(
            guard.should_skip(&m, Some(stale), None, 0, now),
            Some(SkipReason::StaleParent)
        );
    }

    #[test]
    fn low_effort_spam_requires_both_signals() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("op"));
        // Both signals: 0 followers + 5-char text → spam.
        let m = fixture_mention("hi!!!", "other");
        let ctx = fixture_ctx(0);
        assert_eq!(
            guard.should_skip(&m, None, Some(&ctx), 0, Utc::now()),
            Some(SkipReason::LowEffortSpam)
        );
        // Only one signal (followers low but text long enough) → not spam.
        let m_long = fixture_mention(
            "this is a substantive question about the framework, what do you think",
            "other",
        );
        let ctx_low = fixture_ctx(1);
        assert_eq!(
            guard.should_skip(&m_long, None, Some(&ctx_low), 0, Utc::now()),
            None
        );
    }

    #[test]
    fn per_author_rate_limit_skips_at_threshold() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("op"));
        let m = fixture_mention("real question with substance here", "spammer");
        // 3 prior replies to this author → at threshold → skip.
        assert_eq!(
            guard.should_skip(&m, None, None, 3, Utc::now()),
            Some(SkipReason::PerAuthorRateLimit)
        );
        // 2 prior replies → still under threshold.
        assert_eq!(guard.should_skip(&m, None, None, 2, Utc::now()), None);
    }

    #[test]
    fn too_short_to_engage_skips_emoji_only() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("op"));
        let m = fixture_mention("👍🔥", "other");
        assert_eq!(
            guard.should_skip(&m, None, None, 0, Utc::now()),
            Some(SkipReason::TooShortToEngage)
        );
    }
}
