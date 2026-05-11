//! Resolves the X/Twitter operator user-id for a `[[daemon.persona_posts]]`
//! entry. Resolution order (most-specific to least-specific):
//!
//! 1. A matching `[[daemon.persona_mentions]]` entry with the same `persona`
//!    slug — uses its `user_id` field directly.
//! 2. The `HEARTBIT_GHOST_OPERATOR_USER_ID` environment variable.
//!
//! Both sources missing returns `Err(OperatorIdError::Unresolved)`; the
//! daemon caller then logs an error banner, increments the skip metric,
//! and continues past the entry (rather than crash-looping the process).

use heartbit::PersonaMentionsConfig;

/// Source from which an operator user-id was resolved. Used by callers for
/// logging — the resolved id itself is the primary return value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorIdSource {
    PersonaMentions,
    EnvVar,
}

#[derive(Debug, thiserror::Error)]
pub enum OperatorIdError {
    #[error(
        "no operator user-id for persona '{persona}': set HEARTBIT_GHOST_OPERATOR_USER_ID \
         or add a matching [[daemon.persona_mentions]] entry"
    )]
    Unresolved { persona: String },
}

/// Resolve the operator user-id for `persona_slug`, given the daemon's
/// `persona_mentions` config and the process environment.
///
/// `env_lookup` is an injectable hook so tests don't need to mutate
/// real `std::env`. Production callers pass `|k| std::env::var(k).ok()`.
pub fn resolve_operator_user_id(
    persona_slug: &str,
    persona_mentions: &[PersonaMentionsConfig],
    env_lookup: impl Fn(&str) -> Option<String>,
) -> Result<(String, OperatorIdSource), OperatorIdError> {
    if let Some(m) = persona_mentions
        .iter()
        .find(|m| m.persona == persona_slug && m.enabled)
    {
        return Ok((m.user_id.clone(), OperatorIdSource::PersonaMentions));
    }
    if let Some(v) = env_lookup("HEARTBIT_GHOST_OPERATOR_USER_ID")
        && !v.trim().is_empty()
    {
        return Ok((v, OperatorIdSource::EnvVar));
    }
    Err(OperatorIdError::Unresolved {
        persona: persona_slug.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit::PersonaMentionsConfig;

    fn mention(persona: &str, user_id: &str) -> PersonaMentionsConfig {
        // Hand-roll a minimal PersonaMentionsConfig via TOML so we don't
        // depend on every default-fn here.
        let toml = format!(
            r#"
persona = "{persona}"
user_id = "{user_id}"
"#
        );
        toml::from_str(&toml).expect("valid PersonaMentionsConfig fixture")
    }

    #[test]
    fn persona_mentions_match_wins_over_env() {
        let mentions = vec![mention("heartbit-ghost:x", "111")];
        let env = |k: &str| match k {
            "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("999".into()),
            _ => None,
        };
        let (id, src) = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap();
        assert_eq!(id, "111");
        assert_eq!(src, OperatorIdSource::PersonaMentions);
    }

    #[test]
    fn env_used_when_no_mentions_match() {
        let mentions = vec![mention("other:x", "222")];
        let env = |k: &str| match k {
            "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("777".into()),
            _ => None,
        };
        let (id, src) = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap();
        assert_eq!(id, "777");
        assert_eq!(src, OperatorIdSource::EnvVar);
    }

    #[test]
    fn empty_env_value_falls_through_to_unresolved() {
        let mentions: Vec<PersonaMentionsConfig> = vec![];
        let env = |k: &str| match k {
            "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("   ".into()),
            _ => None,
        };
        let err = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap_err();
        match err {
            OperatorIdError::Unresolved { persona } => {
                assert_eq!(persona, "heartbit-ghost:x");
            }
        }
    }

    #[test]
    fn disabled_mentions_entry_is_ignored() {
        let mut m = mention("heartbit-ghost:x", "111");
        m.enabled = false;
        let mentions = vec![m];
        let env = |k: &str| match k {
            "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("777".into()),
            _ => None,
        };
        let (id, src) = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap();
        assert_eq!(id, "777");
        assert_eq!(src, OperatorIdSource::EnvVar);
    }

    #[test]
    fn no_sources_returns_unresolved() {
        let mentions: Vec<PersonaMentionsConfig> = vec![];
        let env = |_k: &str| None;
        let err = resolve_operator_user_id("heartbit-ghost:x", &mentions, env).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("heartbit-ghost:x"), "msg: {msg}");
        assert!(
            msg.contains("HEARTBIT_GHOST_OPERATOR_USER_ID"),
            "msg: {msg}"
        );
        assert!(msg.contains("persona_mentions"), "msg: {msg}");
    }

    #[test]
    fn caller_contract_three_paths() {
        // Path 1: mentions present → use it
        let mentions = vec![mention("p1", "from_mentions")];
        let (id, _) =
            resolve_operator_user_id("p1", &mentions, |_| Some("from_env".into())).unwrap();
        assert_eq!(id, "from_mentions");

        // Path 2: mentions absent for this persona → use env
        let (id, _) =
            resolve_operator_user_id("p2", &mentions, |_| Some("from_env".into())).unwrap();
        assert_eq!(id, "from_env");

        // Path 3: neither → caller must skip this persona
        let err = resolve_operator_user_id("p2", &[], |_| None).unwrap_err();
        assert!(matches!(err, OperatorIdError::Unresolved { .. }));
    }
}
