use super::config::DmPolicy;

/// Access control for Telegram DMs based on the configured policy.
pub struct AccessControl {
    policy: DmPolicy,
    allowed_users: Vec<i64>,
}

impl AccessControl {
    pub fn new(policy: DmPolicy, allowed_users: Vec<i64>) -> Self {
        Self {
            policy,
            allowed_users,
        }
    }

    /// Check if a user is permitted to interact via DM.
    /// Returns `true` if the message should be processed.
    pub fn check_dm(&self, user_id: i64) -> bool {
        match self.policy {
            DmPolicy::Open => true,
            DmPolicy::Disabled => false,
            DmPolicy::Allowlist => self.allowed_users.contains(&user_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_policy_allows_anyone() {
        let ac = AccessControl::new(DmPolicy::Open, vec![]);
        assert!(ac.check_dm(12345));
        assert!(ac.check_dm(99999));
    }

    #[test]
    fn disabled_policy_denies_everyone() {
        let ac = AccessControl::new(DmPolicy::Disabled, vec![111, 222]);
        assert!(!ac.check_dm(111));
        assert!(!ac.check_dm(222));
        assert!(!ac.check_dm(333));
    }

    #[test]
    fn allowlist_permits_listed_users() {
        let ac = AccessControl::new(DmPolicy::Allowlist, vec![111, 222, 333]);
        assert!(ac.check_dm(111));
        assert!(ac.check_dm(222));
        assert!(ac.check_dm(333));
    }

    #[test]
    fn allowlist_denies_unlisted_users() {
        let ac = AccessControl::new(DmPolicy::Allowlist, vec![111, 222]);
        assert!(!ac.check_dm(333));
        assert!(!ac.check_dm(0));
    }

    #[test]
    fn allowlist_empty_denies_all() {
        let ac = AccessControl::new(DmPolicy::Allowlist, vec![]);
        assert!(!ac.check_dm(111));
    }

    #[test]
    fn allowlist_negative_user_ids() {
        // Telegram user IDs are always positive, but we handle negative gracefully
        let ac = AccessControl::new(DmPolicy::Allowlist, vec![-1]);
        assert!(ac.check_dm(-1));
        assert!(!ac.check_dm(1));
    }
}
