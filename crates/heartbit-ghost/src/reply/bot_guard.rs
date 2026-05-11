//! Bot-heuristic guard — skips mentions when ≥ `threshold` of 3
//! signals match: suspicious handle pattern, low follower/following
//! ratio, recent account creation. Conservative (≥2 of 3 by default)
//! to avoid false positives on real humans.
//!
//! See P1.7 spec §5.

use chrono::{DateTime, Duration, Utc};

use super::{Mention, MentionerContext, spam_guard::SkipReason};

/// Configuration for [`BotHeuristicGuard`].
#[derive(Debug, Clone)]
pub struct BotHeuristicConfig {
    /// Substrings that strongly suggest a bot. Case-insensitive
    /// substring match on the author's handle.
    pub suspicious_handle_patterns: Vec<String>,
    /// Minimum follower/following ratio. Skip rule fires when
    /// `followers/following < threshold`. Disabled when
    /// `following_count` is None or 0.
    pub min_follower_following_ratio: f32,
    /// Minimum account age in days. Skip rule fires when the account
    /// is younger.
    pub min_account_age_days: i64,
    /// Number of signals required to trigger a skip (0 disables the
    /// guard entirely, 3 requires all signals).
    pub threshold: usize,
}

impl BotHeuristicConfig {
    /// Sensible defaults: 6 common bot-handle patterns, 0.05 ratio,
    /// 7-day account age, threshold 2.
    pub fn defaults() -> Self {
        Self {
            suspicious_handle_patterns: vec![
                "_bot".into(),
                "_gpt".into(),
                "_ai".into(),
                "ai_".into(),
                "gpt_".into(),
                "bot_".into(),
            ],
            min_follower_following_ratio: 0.05,
            min_account_age_days: 7,
            threshold: 2,
        }
    }
}

impl Default for BotHeuristicConfig {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Bot heuristic guard. Pure logic — no network, no I/O.
pub struct BotHeuristicGuard {
    cfg: BotHeuristicConfig,
}

impl BotHeuristicGuard {
    /// Construct from config.
    pub fn new(cfg: BotHeuristicConfig) -> Self {
        Self { cfg }
    }

    /// Returns `Some(SkipReason::BotSuspected { reasons })` when at
    /// least `threshold` signals match. `None` to proceed.
    /// Threshold = 0 disables the guard.
    pub fn should_skip(
        &self,
        mention: &Mention,
        mentioner: Option<&MentionerContext>,
        now: DateTime<Utc>,
    ) -> Option<SkipReason> {
        if self.cfg.threshold == 0 {
            return None;
        }
        let mut reasons: Vec<String> = Vec::new();

        // Signal 1: handle suffix/prefix match (always evaluable;
        // uses mention.author_handle even when mentioner is None).
        let handle_lower = mention.author_handle.to_lowercase();
        for pattern in &self.cfg.suspicious_handle_patterns {
            if handle_lower.contains(&pattern.to_lowercase()) {
                reasons.push(format!("handle pattern '{pattern}'"));
                break; // one match per signal
            }
        }

        if let Some(ctx) = mentioner {
            // Signal 2: follower/following ratio.
            if let (Some(followers), Some(following)) = (ctx.follower_count, ctx.following_count)
                && following > 0
            {
                let ratio = followers as f32 / following as f32;
                if ratio < self.cfg.min_follower_following_ratio {
                    reasons.push(format!(
                        "follower/following ratio {ratio:.3} < {:.3}",
                        self.cfg.min_follower_following_ratio
                    ));
                }
            }
            // Signal 3: account age.
            if let Some(created) = ctx.account_created_at {
                let age = now.signed_duration_since(created);
                if age < Duration::days(self.cfg.min_account_age_days) {
                    reasons.push(format!(
                        "account age {} days < {}",
                        age.num_days(),
                        self.cfg.min_account_age_days
                    ));
                }
            }
        }

        if reasons.len() >= self.cfg.threshold {
            Some(SkipReason::BotSuspected { reasons })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_mention(handle: &str) -> Mention {
        Mention {
            id: "m1".into(),
            text: "hi".into(),
            author_id: "1".into(),
            author_handle: handle.into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: None,
            conversation_id: None,
        }
    }

    fn fixture_ctx(
        followers: Option<u64>,
        following: Option<u64>,
        created_at: Option<DateTime<Utc>>,
    ) -> MentionerContext {
        MentionerContext {
            handle: "x".into(),
            bio: None,
            recent_tweets: vec![],
            follower_count: followers,
            following_count: following,
            account_created_at: created_at,
        }
    }

    #[test]
    fn handle_pattern_signal_matches_substring_case_insensitive() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 1,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("ChatGPT_BOT");
        let result = guard.should_skip(&m, None, Utc::now());
        assert!(result.is_some());
        if let Some(SkipReason::BotSuspected { reasons }) = result {
            assert_eq!(reasons.len(), 1);
            assert!(
                reasons[0].contains("_bot") || reasons[0].contains("_gpt"),
                "got: {reasons:?}"
            );
        }
    }

    #[test]
    fn follow_ratio_signal_matches_below_threshold() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 1,
            min_follower_following_ratio: 0.05,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("normal_user");
        // 10 followers / 1000 following = 0.01 < 0.05.
        let ctx = fixture_ctx(Some(10), Some(1000), None);
        let result = guard.should_skip(&m, Some(&ctx), Utc::now());
        assert!(matches!(result, Some(SkipReason::BotSuspected { .. })));
    }

    #[test]
    fn account_age_signal_matches_recent_account() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 1,
            min_account_age_days: 7,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("normal_user");
        let now = Utc::now();
        let recent = now - Duration::days(2); // 2-day-old account.
        let ctx = fixture_ctx(None, None, Some(recent));
        let result = guard.should_skip(&m, Some(&ctx), now);
        assert!(matches!(result, Some(SkipReason::BotSuspected { .. })));
    }

    #[test]
    fn threshold_2_requires_two_signals() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 2,
            ..BotHeuristicConfig::defaults()
        });
        // Handle "_bot" matches signal 1; 1-day-old account matches signal 3.
        let m = fixture_mention("normal_user_bot");
        let now = Utc::now();
        let ctx = fixture_ctx(None, None, Some(now - Duration::days(1)));
        let result = guard.should_skip(&m, Some(&ctx), now);
        assert!(
            matches!(&result, Some(SkipReason::BotSuspected { reasons }) if reasons.len() == 2),
            "got: {result:?}"
        );
    }

    #[test]
    fn single_signal_does_not_skip_at_threshold_2() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 2,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("normal_user_bot"); // 1 signal (handle).
        // No follow ratio, no account age — only 1 signal total.
        let result = guard.should_skip(&m, None, Utc::now());
        assert!(result.is_none(), "got: {result:?}");
    }

    #[test]
    fn threshold_zero_disables_guard() {
        let guard = BotHeuristicGuard::new(BotHeuristicConfig {
            threshold: 0,
            ..BotHeuristicConfig::defaults()
        });
        let m = fixture_mention("definitely_a_bot_gpt_ai");
        let now = Utc::now();
        let ctx = fixture_ctx(Some(0), Some(10000), Some(now - Duration::days(1)));
        let result = guard.should_skip(&m, Some(&ctx), now);
        assert!(
            result.is_none(),
            "threshold=0 should disable; got {result:?}"
        );
    }
}
