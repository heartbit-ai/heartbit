//! Static validation for daemon configs.
//!
//! Performs only filesystem + cross-reference checks — no network calls.
//! Surfaces all findings (don't fail-fast) so the operator sees every
//! issue in one pass.
//!
//! Layered on top of `HeartbitConfig::validate()` (which is already invoked
//! by `HeartbitConfig::from_file`); this module covers the gap between
//! "parses + structural validation" and "daemon will actually start".

use std::path::Path;

use heartbit::{DaemonConfig, HeartbitConfig};

use super::operator_id::resolve_operator_user_id;

/// A single validation finding. Each finding maps to a single fixable
/// misconfiguration and includes the persona/entry it relates to so the
/// operator can locate the offending block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub kind: ValidationIssueKind,
    pub context: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationIssueKind {
    /// `[[daemon.persona_posts]]` entry's persona cannot resolve an
    /// operator user-id from either persona_mentions or the env.
    MissingOperatorUserId,
    /// `post_history_store = "jsonl"` but no `post_history_path` set.
    MissingPostHistoryPath,
    /// `budget_store = "jsonl"` but no `budget_path` set
    /// (mentions config).
    MissingBudgetPath,
    /// A JSONL store path's parent directory doesn't exist (and can't be
    /// created at validation time — we only check, don't mutate).
    NonexistentParentDir { path: String },
    /// `[[daemon.persona_quotes]]` entry has empty `source_user_ids`.
    MissingSourceUserIds,
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ValidationIssueKind::MissingOperatorUserId => write!(
                f,
                "{}: cannot resolve operator user-id (set HEARTBIT_GHOST_OPERATOR_USER_ID \
                 or add a matching [[daemon.persona_mentions]] entry)",
                self.context
            ),
            ValidationIssueKind::MissingPostHistoryPath => write!(
                f,
                "{}: post_history_store = \"jsonl\" but post_history_path is not set",
                self.context
            ),
            ValidationIssueKind::MissingBudgetPath => write!(
                f,
                "{}: budget_store = \"jsonl\" but budget_path is not set",
                self.context
            ),
            ValidationIssueKind::NonexistentParentDir { path } => write!(
                f,
                "{}: parent directory of '{path}' does not exist",
                self.context
            ),
            ValidationIssueKind::MissingSourceUserIds => write!(
                f,
                "{}: source_user_ids is empty — at least one X user ID required",
                self.context
            ),
        }
    }
}

/// Static validation entry point. Returns the list of issues; empty list = OK.
pub fn validate_daemon_config(
    config: &HeartbitConfig,
    env_lookup: impl Fn(&str) -> Option<String>,
    path_exists: impl Fn(&Path) -> bool,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    let Some(daemon_config) = config.daemon.as_ref() else {
        return issues; // Nothing daemon-specific to validate.
    };

    validate_persona_posts(daemon_config, &env_lookup, &path_exists, &mut issues);
    validate_persona_mentions(daemon_config, &path_exists, &mut issues);
    validate_persona_quotes(daemon_config, &path_exists, &mut issues);

    issues
}

fn validate_persona_posts(
    daemon: &DaemonConfig,
    env_lookup: &impl Fn(&str) -> Option<String>,
    path_exists: &impl Fn(&Path) -> bool,
    issues: &mut Vec<ValidationIssue>,
) {
    for cfg in &daemon.persona_posts {
        if !cfg.enabled {
            continue;
        }
        let context = format!("[[daemon.persona_posts]] persona='{}'", cfg.persona);

        // 1. operator user-id resolvable?
        if resolve_operator_user_id(&cfg.persona, &daemon.persona_mentions, |k| env_lookup(k))
            .is_err()
        {
            issues.push(ValidationIssue {
                kind: ValidationIssueKind::MissingOperatorUserId,
                context: context.clone(),
            });
        }

        // 2. jsonl store needs a path
        if cfg.post_history_store == "jsonl" {
            match cfg.post_history_path.as_deref() {
                None => issues.push(ValidationIssue {
                    kind: ValidationIssueKind::MissingPostHistoryPath,
                    context: context.clone(),
                }),
                Some(p) => check_parent_dir(p, &context, path_exists, issues),
            }
        }
    }
}

fn validate_persona_mentions(
    daemon: &DaemonConfig,
    path_exists: &impl Fn(&Path) -> bool,
    issues: &mut Vec<ValidationIssue>,
) {
    for cfg in &daemon.persona_mentions {
        if !cfg.enabled {
            continue;
        }
        let context = format!("[[daemon.persona_mentions]] persona='{}'", cfg.persona);

        if cfg.budget_store == "jsonl" {
            match cfg.budget_path.as_deref() {
                None => issues.push(ValidationIssue {
                    kind: ValidationIssueKind::MissingBudgetPath,
                    context: context.clone(),
                }),
                Some(p) => check_parent_dir(p, &context, path_exists, issues),
            }
        }
        if cfg.mention_store == "jsonl"
            && let Some(p) = cfg.mention_store_path.as_deref()
        {
            check_parent_dir(p, &context, path_exists, issues);
        }
    }
}

fn validate_persona_quotes(
    daemon: &DaemonConfig,
    path_exists: &impl Fn(&Path) -> bool,
    issues: &mut Vec<ValidationIssue>,
) {
    for cfg in &daemon.persona_quotes {
        if !cfg.enabled {
            continue;
        }
        let context = format!("[[daemon.persona_quotes]] persona='{}'", cfg.persona);

        if cfg.source_user_ids.is_empty() {
            issues.push(ValidationIssue {
                kind: ValidationIssueKind::MissingSourceUserIds,
                context: context.clone(),
            });
        }

        if cfg.seen_store == "jsonl"
            && let Some(p) = cfg.seen_store_path.as_deref()
        {
            check_parent_dir(p, &context, path_exists, issues);
        }
    }
}

fn check_parent_dir(
    raw_path: &str,
    context: &str,
    path_exists: &impl Fn(&Path) -> bool,
    issues: &mut Vec<ValidationIssue>,
) {
    // Tilde expansion — mirror what the daemon startup does. Plan keeps it
    // local so this module doesn't take a dependency on the CLI's
    // expand_tilde helper.
    let expanded: std::path::PathBuf = if let Some(stripped) = raw_path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            Path::new(&home).join(stripped)
        } else {
            Path::new(raw_path).to_path_buf()
        }
    } else {
        Path::new(raw_path).to_path_buf()
    };
    if let Some(parent) = expanded.parent()
        && !parent.as_os_str().is_empty()
        && !path_exists(parent)
    {
        issues.push(ValidationIssue {
            kind: ValidationIssueKind::NonexistentParentDir {
                path: expanded.display().to_string(),
            },
            context: context.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit::{DaemonConfig, PersonaMentionsConfig, PersonaPostsConfig, PersonaQuotesConfig};

    fn config_with_daemon(daemon: DaemonConfig) -> HeartbitConfig {
        // HeartbitConfig has all fields #[serde(default)], so empty TOML works.
        let mut c: HeartbitConfig = toml::from_str("").expect("base config parses");
        c.daemon = Some(daemon);
        c
    }

    fn minimal_daemon() -> DaemonConfig {
        toml::from_str("").expect("default DaemonConfig parses from empty TOML")
    }

    fn mention(persona: &str, user_id: &str) -> PersonaMentionsConfig {
        let toml = format!(
            r#"
persona = "{persona}"
user_id = "{user_id}"
"#
        );
        toml::from_str(&toml).expect("PersonaMentionsConfig fixture parses")
    }

    fn post(persona: &str) -> PersonaPostsConfig {
        let toml = format!(
            r#"
persona = "{persona}"
"#
        );
        toml::from_str(&toml).expect("PersonaPostsConfig fixture parses")
    }

    fn quote(persona: &str) -> PersonaQuotesConfig {
        let toml = format!(
            r#"
persona = "{persona}"
source_user_ids = ["44196397"]
"#
        );
        toml::from_str(&toml).expect("PersonaQuotesConfig fixture parses")
    }

    #[test]
    fn empty_daemon_config_has_no_issues() {
        let cfg = config_with_daemon(minimal_daemon());
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert!(issues.is_empty(), "issues: {issues:?}");
    }

    #[test]
    fn persona_posts_without_operator_user_id_is_flagged() {
        let mut d = minimal_daemon();
        d.persona_posts.push(post("heartbit-ghost:x"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert_eq!(issues.len(), 1, "issues: {issues:?}");
        assert_eq!(issues[0].kind, ValidationIssueKind::MissingOperatorUserId);
        assert!(issues[0].context.contains("heartbit-ghost:x"));
    }

    #[test]
    fn persona_posts_with_matching_mentions_passes() {
        let mut d = minimal_daemon();
        d.persona_posts.push(post("heartbit-ghost:x"));
        d.persona_mentions.push(mention("heartbit-ghost:x", "42"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert!(issues.is_empty(), "issues: {issues:?}");
    }

    #[test]
    fn persona_posts_with_env_var_passes() {
        let mut d = minimal_daemon();
        d.persona_posts.push(post("heartbit-ghost:x"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(
            &cfg,
            |k| match k {
                "HEARTBIT_GHOST_OPERATOR_USER_ID" => Some("42".into()),
                _ => None,
            },
            |_| true,
        );
        assert!(issues.is_empty(), "issues: {issues:?}");
    }

    #[test]
    fn jsonl_post_history_without_path_is_flagged() {
        let mut d = minimal_daemon();
        let mut p = post("heartbit-ghost:x");
        p.post_history_store = "jsonl".into();
        d.persona_posts.push(p);
        d.persona_mentions.push(mention("heartbit-ghost:x", "42"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert!(
            issues
                .iter()
                .any(|i| i.kind == ValidationIssueKind::MissingPostHistoryPath),
            "issues: {issues:?}"
        );
    }

    #[test]
    fn jsonl_post_history_with_missing_parent_dir_is_flagged() {
        let mut d = minimal_daemon();
        let mut p = post("heartbit-ghost:x");
        p.post_history_store = "jsonl".into();
        p.post_history_path = Some("/definitely/not/a/real/dir/file.jsonl".into());
        d.persona_posts.push(p);
        d.persona_mentions.push(mention("heartbit-ghost:x", "42"));
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| false);
        assert!(
            issues.iter().any(|i| matches!(
                &i.kind,
                ValidationIssueKind::NonexistentParentDir { path } if path.contains("/definitely/not/a/real/dir")
            )),
            "issues: {issues:?}"
        );
    }

    #[test]
    fn persona_quotes_empty_source_user_ids_is_flagged() {
        let mut d = minimal_daemon();
        let mut q = quote("heartbit-ghost:x");
        q.source_user_ids = vec![]; // explicitly empty
        d.persona_quotes.push(q);
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert_eq!(issues.len(), 1, "issues: {issues:?}");
        assert_eq!(issues[0].kind, ValidationIssueKind::MissingSourceUserIds);
        assert!(issues[0].context.contains("heartbit-ghost:x"));
    }

    #[test]
    fn persona_quotes_jsonl_with_missing_parent_dir_is_flagged() {
        let mut d = minimal_daemon();
        let mut q = quote("heartbit-ghost:x");
        q.seen_store = "jsonl".into();
        q.seen_store_path = Some("/definitely/not/a/real/dir/seen.jsonl".into());
        d.persona_quotes.push(q);
        let cfg = config_with_daemon(d);
        let issues = validate_daemon_config(&cfg, |_| None, |_| false);
        assert!(
            issues.iter().any(|i| matches!(
                &i.kind,
                ValidationIssueKind::NonexistentParentDir { path } if path.contains("/definitely/not/a/real/dir")
            )),
            "issues: {issues:?}"
        );
    }

    #[test]
    fn no_daemon_section_returns_no_issues() {
        let cfg: HeartbitConfig = toml::from_str("").expect("config parses");
        let issues = validate_daemon_config(&cfg, |_| None, |_| true);
        assert!(issues.is_empty(), "issues: {issues:?}");
    }
}
