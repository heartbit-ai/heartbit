use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Action to take when a permission rule matches a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionAction {
    /// Execute without asking.
    Allow,
    /// Reject without asking.
    Deny,
    /// Ask the human-in-the-loop callback (or allow if no callback is set).
    Ask,
}

/// A single permission rule matching a tool name and input pattern.
///
/// Rules are evaluated in order — first match wins. The `tool` field matches
/// the tool name (`"*"` matches all tools). The `pattern` field is a glob
/// matched against all string values in the tool's JSON input (`"*"` matches
/// everything).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRule {
    /// Tool name to match. `"*"` matches all tools.
    pub tool: String,
    /// Glob pattern matched against string values in the tool input.
    /// `"*"` matches everything.
    #[serde(default = "default_pattern")]
    pub pattern: String,
    /// Action to take when the rule matches.
    pub action: PermissionAction,
}

fn default_pattern() -> String {
    "*".into()
}

impl PermissionRule {
    /// Check if this rule matches the given tool call.
    fn matches(&self, tool_name: &str, input: &serde_json::Value) -> bool {
        // Tool name: "*" matches all, otherwise exact match.
        if self.tool != "*" && self.tool != tool_name {
            return false;
        }

        // Pattern: "*" matches everything.
        if self.pattern == "*" {
            return true;
        }

        // Match pattern against all string values in the input object.
        match input {
            serde_json::Value::Object(map) => map.values().any(|v| match v {
                serde_json::Value::String(s) => glob_match(&self.pattern, s),
                _ => false,
            }),
            // Non-object input: only match if pattern is "*" (handled above).
            _ => false,
        }
    }
}

/// Ordered set of permission rules evaluated for each tool call.
///
/// # Evaluation order
///
/// Rules are checked in order — first match wins. If no rule matches,
/// `evaluate` returns `None` and the caller decides the default behavior.
#[derive(Debug, Clone, Default)]
pub struct PermissionRuleset {
    rules: Vec<PermissionRule>,
}

impl PermissionRuleset {
    pub fn new(rules: Vec<PermissionRule>) -> Self {
        Self { rules }
    }

    /// Create a ruleset that allows all tool calls unconditionally.
    ///
    /// This bypasses both permission rules and the `on_approval` callback —
    /// every tool call is executed without asking.
    pub fn allow_all() -> Self {
        Self {
            rules: vec![PermissionRule {
                tool: "*".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            }],
        }
    }

    /// Evaluate a tool call against the ruleset.
    ///
    /// Returns the action of the first matching rule, or `None` if no rule matches.
    pub fn evaluate(&self, tool_name: &str, input: &serde_json::Value) -> Option<PermissionAction> {
        self.rules
            .iter()
            .find(|r| r.matches(tool_name, input))
            .map(|r| r.action)
    }

    /// Append rules from learned permissions (or any source).
    ///
    /// Appended rules are evaluated after existing rules, so config rules
    /// retain priority (first match wins).
    pub fn append_rules(&mut self, rules: &[PermissionRule]) {
        self.rules.extend(rules.iter().cloned());
    }

    /// Returns `true` if the ruleset has no rules.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

// ---------------------------------------------------------------------------
// LearnedPermissions — user decisions persisted to disk
// ---------------------------------------------------------------------------

/// TOML wrapper for serialization.
#[derive(Debug, Serialize, Deserialize)]
struct LearnedPermissionsFile {
    #[serde(default)]
    rules: Vec<PermissionRule>,
}

/// Permission rules learned from user approval decisions, persisted to a TOML
/// file (typically `~/.config/heartbit/permissions.toml`).
///
/// Call [`add_rule`](Self::add_rule) to record a new decision and
/// [`save`](Self::save) to flush to disk.
#[derive(Debug)]
pub struct LearnedPermissions {
    path: PathBuf,
    rules: Vec<PermissionRule>,
}

impl LearnedPermissions {
    /// Load learned permissions from a TOML file. Creates an empty set if the
    /// file does not exist.
    pub fn load(path: &Path) -> Result<Self, Error> {
        if !path.exists() {
            return Ok(Self {
                path: path.to_path_buf(),
                rules: Vec::new(),
            });
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("failed to read {}: {e}", path.display())))?;
        let file: LearnedPermissionsFile = toml::from_str(&content).map_err(|e| {
            Error::Config(format!("invalid permissions file {}: {e}", path.display()))
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            rules: file.rules,
        })
    }

    /// Save learned permissions to disk. Creates parent directories as needed.
    pub fn save(&self) -> Result<(), Error> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                Error::Config(format!(
                    "failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let file = LearnedPermissionsFile {
            rules: self.rules.clone(),
        };
        let content = toml::to_string_pretty(&file)
            .map_err(|e| Error::Config(format!("failed to serialize permissions: {e}")))?;
        std::fs::write(&self.path, content)
            .map_err(|e| Error::Config(format!("failed to write {}: {e}", self.path.display())))?;
        Ok(())
    }

    /// Add a rule and immediately save to disk.
    ///
    /// Duplicate rules (same tool + action) are silently skipped.
    pub fn add_rule(&mut self, rule: PermissionRule) -> Result<(), Error> {
        let exists = self
            .rules
            .iter()
            .any(|r| r.tool == rule.tool && r.pattern == rule.pattern && r.action == rule.action);
        if !exists {
            self.rules.push(rule);
            self.save()?;
        }
        Ok(())
    }

    /// Get a reference to the learned rules.
    pub fn rules(&self) -> &[PermissionRule] {
        &self.rules
    }

    /// Default path: `~/.config/heartbit/permissions.toml`.
    pub fn default_path() -> Option<PathBuf> {
        dirs_config_dir().map(|d| d.join("heartbit").join("permissions.toml"))
    }
}

/// Get the platform-appropriate config directory.
/// Uses `$XDG_CONFIG_HOME` on Linux, `~/Library/Application Support` on macOS,
/// `%APPDATA%` on Windows. Falls back to `~/.config`.
fn dirs_config_dir() -> Option<PathBuf> {
    // Prefer XDG_CONFIG_HOME if set
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        return Some(PathBuf::from(xdg));
    }
    // Fall back to ~/.config
    home_dir().map(|h| h.join(".config"))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Simple glob matching: `*` matches zero or more of any character,
/// `?` matches exactly one character. All other characters match literally.
///
/// Uses an iterative two-pointer approach (O(n*m) worst case) to avoid
/// exponential blowup from recursive backtracking on pathological patterns.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();

    let (mut pi, mut ti) = (0usize, 0usize);
    // Position of the last '*' in pattern, and the text position to retry from.
    let (mut star_pi, mut star_ti) = (usize::MAX, 0usize);

    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            // Backtrack: advance the star's text match by one
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }

    // Consume trailing '*'s in pattern
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }

    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    // --- Glob matching ---

    #[test]
    fn glob_exact_match() {
        assert!(glob_match("hello", "hello"));
        assert!(!glob_match("hello", "world"));
    }

    #[test]
    fn glob_star_matches_any() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(glob_match("*.rs", "test.rs"));
        assert!(!glob_match("*.rs", "main.py"));
    }

    #[test]
    fn glob_star_matches_path_separator() {
        assert!(glob_match("src/*/*.rs", "src/agent/mod.rs"));
        assert!(glob_match("src/**/*.rs", "src/agent/mod.rs"));
    }

    #[test]
    fn glob_question_mark() {
        assert!(glob_match("test?.rs", "test1.rs"));
        assert!(!glob_match("test?.rs", "test12.rs"));
    }

    #[test]
    fn glob_star_matches_empty() {
        assert!(glob_match("*", ""));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn glob_complex_pattern() {
        assert!(glob_match("*.env*", ".env"));
        assert!(glob_match("*.env*", ".env.local"));
        assert!(glob_match("*.env*", "config.env.bak"));
    }

    #[test]
    fn glob_rm_pattern() {
        assert!(glob_match("rm *", "rm -rf /"));
        assert!(glob_match("rm *", "rm file.txt"));
        assert!(!glob_match("rm *", "ls -la"));
    }

    #[test]
    fn glob_no_exponential_blowup() {
        // Pathological pattern that would cause exponential time with naive recursion.
        // With the iterative approach, this completes instantly.
        assert!(!glob_match("*a*a*a*a*a*a*a*a*b", "aaaaaaaaaaaaaaaaaaaaaa"));
    }

    #[test]
    fn glob_empty_pattern_matches_empty_text() {
        assert!(glob_match("", ""));
        assert!(!glob_match("", "nonempty"));
    }

    #[test]
    fn glob_consecutive_stars() {
        // Multiple consecutive stars should behave like one
        assert!(glob_match("**", "anything"));
        assert!(glob_match("a**b", "aXYZb"));
    }

    // --- PermissionRule matching ---

    #[test]
    fn rule_matches_exact_tool_name() {
        let rule = PermissionRule {
            tool: "read_file".into(),
            pattern: "*".into(),
            action: PermissionAction::Allow,
        };
        assert!(rule.matches("read_file", &json!({"path": "foo.txt"})));
        assert!(!rule.matches("write_file", &json!({"path": "foo.txt"})));
    }

    #[test]
    fn rule_wildcard_tool_matches_all() {
        let rule = PermissionRule {
            tool: "*".into(),
            pattern: "*.env*".into(),
            action: PermissionAction::Deny,
        };
        assert!(rule.matches("read_file", &json!({"path": ".env"})));
        assert!(rule.matches("edit_file", &json!({"path": ".env.local"})));
    }

    #[test]
    fn rule_pattern_matches_any_string_value() {
        let rule = PermissionRule {
            tool: "bash".into(),
            pattern: "rm *".into(),
            action: PermissionAction::Deny,
        };
        assert!(rule.matches("bash", &json!({"command": "rm -rf /tmp"})));
        assert!(!rule.matches("bash", &json!({"command": "ls -la"})));
    }

    #[test]
    fn rule_pattern_ignores_non_string_values() {
        let rule = PermissionRule {
            tool: "search".into(),
            pattern: "*.secret*".into(),
            action: PermissionAction::Deny,
        };
        // Non-string values are ignored
        assert!(!rule.matches("search", &json!({"limit": 10})));
    }

    #[test]
    fn rule_non_object_input() {
        let rule = PermissionRule {
            tool: "test".into(),
            pattern: "*.rs".into(),
            action: PermissionAction::Allow,
        };
        // Non-object input: only "*" pattern matches (handled before this)
        assert!(!rule.matches("test", &json!("hello.rs")));
    }

    // --- PermissionRuleset evaluation ---

    #[test]
    fn ruleset_first_match_wins() {
        let ruleset = PermissionRuleset::new(vec![
            PermissionRule {
                tool: "bash".into(),
                pattern: "rm *".into(),
                action: PermissionAction::Deny,
            },
            PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            },
        ]);
        // "rm" command matches first rule → Deny
        assert_eq!(
            ruleset.evaluate("bash", &json!({"command": "rm file.txt"})),
            Some(PermissionAction::Deny)
        );
        // "ls" command matches second rule → Allow
        assert_eq!(
            ruleset.evaluate("bash", &json!({"command": "ls -la"})),
            Some(PermissionAction::Allow)
        );
    }

    #[test]
    fn ruleset_no_match_returns_none() {
        let ruleset = PermissionRuleset::new(vec![PermissionRule {
            tool: "bash".into(),
            pattern: "*".into(),
            action: PermissionAction::Allow,
        }]);
        assert_eq!(
            ruleset.evaluate("read_file", &json!({"path": "foo.txt"})),
            None
        );
    }

    #[test]
    fn ruleset_empty_returns_none() {
        let ruleset = PermissionRuleset::default();
        assert!(ruleset.is_empty());
        assert_eq!(ruleset.evaluate("any_tool", &json!({"key": "value"})), None);
    }

    #[test]
    fn ruleset_allow_all_permits_everything() {
        let ruleset = PermissionRuleset::allow_all();
        assert!(!ruleset.is_empty());
        assert_eq!(
            ruleset.evaluate("bash", &json!({"command": "rm -rf /"})),
            Some(PermissionAction::Allow)
        );
        assert_eq!(
            ruleset.evaluate("read_file", &json!({"path": ".env"})),
            Some(PermissionAction::Allow)
        );
        assert_eq!(
            ruleset.evaluate("any_tool", &json!({})),
            Some(PermissionAction::Allow)
        );
    }

    #[test]
    fn ruleset_deny_env_files_across_all_tools() {
        let ruleset = PermissionRuleset::new(vec![
            PermissionRule {
                tool: "*".into(),
                pattern: "*.env*".into(),
                action: PermissionAction::Deny,
            },
            PermissionRule {
                tool: "read_file".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            },
        ]);
        // .env file denied even for read_file
        assert_eq!(
            ruleset.evaluate("read_file", &json!({"path": ".env"})),
            Some(PermissionAction::Deny)
        );
        // Non-.env file allowed for read_file
        assert_eq!(
            ruleset.evaluate("read_file", &json!({"path": "src/main.rs"})),
            Some(PermissionAction::Allow)
        );
        // Other tool accessing .env → denied
        assert_eq!(
            ruleset.evaluate("edit_file", &json!({"path": ".env.local"})),
            Some(PermissionAction::Deny)
        );
    }

    #[test]
    fn ruleset_ask_for_unmatched() {
        let ruleset = PermissionRuleset::new(vec![
            PermissionRule {
                tool: "read_file".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            },
            // Everything else: ask
            PermissionRule {
                tool: "*".into(),
                pattern: "*".into(),
                action: PermissionAction::Ask,
            },
        ]);
        assert_eq!(
            ruleset.evaluate("read_file", &json!({"path": "foo"})),
            Some(PermissionAction::Allow)
        );
        assert_eq!(
            ruleset.evaluate("bash", &json!({"command": "cargo test"})),
            Some(PermissionAction::Ask)
        );
    }

    // --- Serde ---

    #[test]
    fn permission_rule_deserializes_from_toml() {
        let toml_str = r#"
            tool = "bash"
            pattern = "rm *"
            action = "deny"
        "#;
        let rule: PermissionRule = toml::from_str(toml_str).unwrap();
        assert_eq!(rule.tool, "bash");
        assert_eq!(rule.pattern, "rm *");
        assert_eq!(rule.action, PermissionAction::Deny);
    }

    #[test]
    fn permission_rule_default_pattern() {
        let toml_str = r#"
            tool = "read_file"
            action = "allow"
        "#;
        let rule: PermissionRule = toml::from_str(toml_str).unwrap();
        assert_eq!(rule.pattern, "*");
    }

    #[test]
    fn permission_action_serde_roundtrip() {
        assert_eq!(
            serde_json::from_str::<PermissionAction>("\"allow\"").unwrap(),
            PermissionAction::Allow
        );
        assert_eq!(
            serde_json::from_str::<PermissionAction>("\"deny\"").unwrap(),
            PermissionAction::Deny
        );
        assert_eq!(
            serde_json::from_str::<PermissionAction>("\"ask\"").unwrap(),
            PermissionAction::Ask
        );
    }

    // --- PermissionRuleset::append_rules ---

    #[test]
    fn ruleset_append_rules_adds_after_existing() {
        let mut ruleset = PermissionRuleset::new(vec![PermissionRule {
            tool: "read_file".into(),
            pattern: "*".into(),
            action: PermissionAction::Allow,
        }]);
        ruleset.append_rules(&[PermissionRule {
            tool: "bash".into(),
            pattern: "*".into(),
            action: PermissionAction::Allow,
        }]);
        // Both tools matched
        assert_eq!(
            ruleset.evaluate("read_file", &json!({"path": "f"})),
            Some(PermissionAction::Allow)
        );
        assert_eq!(
            ruleset.evaluate("bash", &json!({"cmd": "ls"})),
            Some(PermissionAction::Allow)
        );
    }

    #[test]
    fn ruleset_config_rules_have_priority_over_learned() {
        // Config says Deny bash, learned says Allow bash
        let mut ruleset = PermissionRuleset::new(vec![PermissionRule {
            tool: "bash".into(),
            pattern: "*".into(),
            action: PermissionAction::Deny,
        }]);
        ruleset.append_rules(&[PermissionRule {
            tool: "bash".into(),
            pattern: "*".into(),
            action: PermissionAction::Allow,
        }]);
        // First match wins → config Deny
        assert_eq!(
            ruleset.evaluate("bash", &json!({"cmd": "ls"})),
            Some(PermissionAction::Deny)
        );
    }

    // --- LearnedPermissions ---

    #[test]
    fn learned_load_nonexistent_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.toml");
        let learned = LearnedPermissions::load(&path).unwrap();
        assert!(learned.rules().is_empty());
    }

    #[test]
    fn learned_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("perms.toml");

        let mut learned = LearnedPermissions::load(&path).unwrap();
        learned
            .add_rule(PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            })
            .unwrap();

        // Reload from disk
        let reloaded = LearnedPermissions::load(&path).unwrap();
        assert_eq!(reloaded.rules().len(), 1);
        assert_eq!(reloaded.rules()[0].tool, "bash");
        assert_eq!(reloaded.rules()[0].action, PermissionAction::Allow);
    }

    #[test]
    fn learned_add_rule_deduplicates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("perms.toml");

        let mut learned = LearnedPermissions::load(&path).unwrap();
        learned
            .add_rule(PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            })
            .unwrap();
        // Same rule again — should not add
        learned
            .add_rule(PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            })
            .unwrap();
        assert_eq!(learned.rules().len(), 1);
    }

    #[test]
    fn learned_different_actions_not_deduplicated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("perms.toml");

        let mut learned = LearnedPermissions::load(&path).unwrap();
        learned
            .add_rule(PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            })
            .unwrap();
        // Same tool but Deny — separate rule
        learned
            .add_rule(PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: PermissionAction::Deny,
            })
            .unwrap();
        assert_eq!(learned.rules().len(), 2);
    }

    #[test]
    fn learned_load_existing_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("perms.toml");
        let content = r#"
[[rules]]
tool = "read_file"
action = "allow"

[[rules]]
tool = "bash"
pattern = "rm *"
action = "deny"
"#;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();

        let learned = LearnedPermissions::load(&path).unwrap();
        assert_eq!(learned.rules().len(), 2);
        assert_eq!(learned.rules()[0].tool, "read_file");
        assert_eq!(learned.rules()[0].pattern, "*"); // default
        assert_eq!(learned.rules()[1].tool, "bash");
        assert_eq!(learned.rules()[1].pattern, "rm *");
    }

    #[test]
    fn learned_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deep").join("nested").join("perms.toml");

        let mut learned = LearnedPermissions::load(&path).unwrap();
        learned
            .add_rule(PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            })
            .unwrap();

        assert!(path.exists());
    }

    #[test]
    fn learned_default_path_returns_some() {
        // This test assumes HOME is set (true in CI and dev)
        if std::env::var_os("HOME").is_some() {
            let path = LearnedPermissions::default_path();
            assert!(path.is_some());
            let p = path.unwrap();
            assert!(p.ends_with("heartbit/permissions.toml"));
        }
    }

    #[test]
    fn learned_load_invalid_toml_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "this is not valid toml {{{").unwrap();

        let err = LearnedPermissions::load(&path).unwrap_err();
        assert!(err.to_string().contains("invalid permissions file"));
    }

    #[test]
    fn learned_rules_integrated_with_ruleset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("perms.toml");

        let mut learned = LearnedPermissions::load(&path).unwrap();
        learned
            .add_rule(PermissionRule {
                tool: "bash".into(),
                pattern: "*".into(),
                action: PermissionAction::Allow,
            })
            .unwrap();

        // Config rules deny bash, learned allows
        let mut ruleset = PermissionRuleset::new(vec![PermissionRule {
            tool: "bash".into(),
            pattern: "rm *".into(),
            action: PermissionAction::Deny,
        }]);
        ruleset.append_rules(learned.rules());

        // "rm" matches config Deny (first match wins)
        assert_eq!(
            ruleset.evaluate("bash", &json!({"cmd": "rm foo"})),
            Some(PermissionAction::Deny)
        );
        // "ls" doesn't match config rule, falls to learned Allow
        assert_eq!(
            ruleset.evaluate("bash", &json!({"cmd": "ls"})),
            Some(PermissionAction::Allow)
        );
    }
}
