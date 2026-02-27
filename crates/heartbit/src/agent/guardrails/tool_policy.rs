//! Declarative tool access control guardrail.
//!
//! Define rules that match tool names (exact or glob) and their inputs
//! to allow, warn, or deny operations. First matching rule wins.

use std::future::Future;
use std::pin::Pin;

use regex::Regex;

use crate::agent::guardrail::{GuardAction, Guardrail, GuardrailMeta};
use crate::error::Error;
use crate::llm::types::ToolCall;

/// Input constraint for tool policy rules.
#[derive(Debug, Clone)]
pub enum InputConstraint {
    /// Deny if a JSON field value matches a regex pattern.
    FieldDenied { path: String, pattern: Regex },
    /// Deny if a JSON field's string value exceeds max_bytes.
    MaxFieldLength { path: String, max_bytes: usize },
}

impl InputConstraint {
    /// Evaluate this constraint against a tool call's input.
    /// Returns `Some(reason)` if the constraint is violated.
    fn evaluate(&self, input: &serde_json::Value) -> Option<String> {
        match self {
            InputConstraint::FieldDenied { path, pattern } => {
                let val = json_path(input, path)?;
                let s = val.as_str().unwrap_or_default();
                if pattern.is_match(s) {
                    Some(format!(
                        "field `{path}` matches denied pattern `{}`",
                        pattern.as_str()
                    ))
                } else {
                    None
                }
            }
            InputConstraint::MaxFieldLength { path, max_bytes } => {
                let val = json_path(input, path)?;
                let s = val.as_str().unwrap_or_default();
                if s.len() > *max_bytes {
                    Some(format!(
                        "field `{path}` exceeds max length ({} > {max_bytes})",
                        s.len()
                    ))
                } else {
                    None
                }
            }
        }
    }
}

/// A single tool policy rule.
#[derive(Debug, Clone)]
pub struct ToolRule {
    /// Tool name pattern (exact match or glob with `*`).
    pub tool_pattern: String,
    /// Action to take when this rule matches.
    pub action: GuardAction,
    /// Optional input constraints (all must pass for Allow).
    pub input_constraints: Vec<InputConstraint>,
}

impl ToolRule {
    /// Check if the tool name matches this rule's pattern.
    fn matches_tool(&self, name: &str) -> bool {
        glob_match(&self.tool_pattern, name)
    }
}

/// Declarative tool policy guardrail.
///
/// Rules are evaluated in order — first match wins. If no rule matches,
/// `default_action` is used.
pub struct ToolPolicyGuardrail {
    rules: Vec<ToolRule>,
    default_action: GuardAction,
}

impl ToolPolicyGuardrail {
    pub fn new(rules: Vec<ToolRule>, default_action: GuardAction) -> Self {
        Self {
            rules,
            default_action,
        }
    }
}

impl GuardrailMeta for ToolPolicyGuardrail {
    fn name(&self) -> &str {
        "tool_policy"
    }
}

impl Guardrail for ToolPolicyGuardrail {
    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        let name = &call.name;
        let input = &call.input;

        for rule in &self.rules {
            if rule.matches_tool(name) {
                // Check input constraints
                for constraint in &rule.input_constraints {
                    if let Some(reason) = constraint.evaluate(input) {
                        let action = match &rule.action {
                            GuardAction::Allow => {
                                // Allow rule with violated constraint → deny
                                GuardAction::deny(format!(
                                    "Tool `{name}` input constraint violated: {reason}"
                                ))
                            }
                            other => other.clone(),
                        };
                        return Box::pin(async move { Ok(action) });
                    }
                }
                // All constraints passed (or none) → return the rule's action
                let action = enrich_action(rule.action.clone(), name, &rule.tool_pattern);
                return Box::pin(async move { Ok(action) });
            }
        }

        // No rule matched → default
        let action = enrich_action(self.default_action.clone(), name, "*");
        Box::pin(async move { Ok(action) })
    }
}

/// Add tool context to actions with empty reasons (common for config-built policies).
fn enrich_action(action: GuardAction, tool_name: &str, pattern: &str) -> GuardAction {
    match action {
        GuardAction::Deny { ref reason } if reason.is_empty() => GuardAction::deny(format!(
            "Tool `{tool_name}` denied by policy rule `{pattern}`"
        )),
        GuardAction::Warn { ref reason } if reason.is_empty() => GuardAction::warn(format!(
            "Tool `{tool_name}` matched policy rule `{pattern}`"
        )),
        other => other,
    }
}

/// Simple glob matching: `*` matches any sequence, `?` matches single char.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    glob_match_inner(&pat, &txt, 0, 0)
}

fn glob_match_inner(pattern: &[char], text: &[char], mut pi: usize, mut ti: usize) -> bool {
    let mut star_pi = None;
    let mut star_ti = None;

    while ti < text.len() {
        if pi < pattern.len() && (pattern[pi] == '?' || pattern[pi] == text[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pattern.len() && pattern[pi] == '*' {
            star_pi = Some(pi);
            star_ti = Some(ti);
            pi += 1;
        } else if let Some(spi) = star_pi {
            pi = spi + 1;
            star_ti = Some(star_ti.unwrap() + 1);
            ti = star_ti.unwrap();
        } else {
            return false;
        }
    }

    while pi < pattern.len() && pattern[pi] == '*' {
        pi += 1;
    }

    pi == pattern.len()
}

/// Simple JSON path lookup: `"field"` for top-level, `"a.b"` for nested.
fn json_path<'a>(value: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut current = value;
    for key in path.split('.') {
        current = current.get(key)?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn exact_match_denies() {
        let g = ToolPolicyGuardrail::new(
            vec![ToolRule {
                tool_pattern: "bash".into(),
                action: GuardAction::deny("bash is blocked"),
                input_constraints: vec![],
            }],
            GuardAction::Allow,
        );
        let action = g
            .pre_tool(&test_call("bash", serde_json::json!({})))
            .await
            .unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn glob_match_denies() {
        let g = ToolPolicyGuardrail::new(
            vec![ToolRule {
                tool_pattern: "gmail_*".into(),
                action: GuardAction::deny("gmail blocked"),
                input_constraints: vec![],
            }],
            GuardAction::Allow,
        );
        let action = g
            .pre_tool(&test_call("gmail_send_email", serde_json::json!({})))
            .await
            .unwrap();
        assert!(action.is_denied());

        // Non-matching
        let action = g
            .pre_tool(&test_call("slack_send", serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn default_allow() {
        let g = ToolPolicyGuardrail::new(vec![], GuardAction::Allow);
        let action = g
            .pre_tool(&test_call("anything", serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn default_deny() {
        let g = ToolPolicyGuardrail::new(
            vec![ToolRule {
                tool_pattern: "read".into(),
                action: GuardAction::Allow,
                input_constraints: vec![],
            }],
            GuardAction::deny("not in allowlist"),
        );
        let action = g
            .pre_tool(&test_call("read", serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(action, GuardAction::Allow);

        let action = g
            .pre_tool(&test_call("bash", serde_json::json!({})))
            .await
            .unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn input_constraint_field_denied() {
        let g = ToolPolicyGuardrail::new(
            vec![ToolRule {
                tool_pattern: "bash".into(),
                action: GuardAction::Allow,
                input_constraints: vec![InputConstraint::FieldDenied {
                    path: "command".into(),
                    pattern: Regex::new(r"rm\s+-rf").unwrap(),
                }],
            }],
            GuardAction::Allow,
        );
        let action = g
            .pre_tool(&test_call(
                "bash",
                serde_json::json!({"command": "rm -rf /"}),
            ))
            .await
            .unwrap();
        assert!(action.is_denied());

        // Safe command
        let action = g
            .pre_tool(&test_call("bash", serde_json::json!({"command": "ls -la"})))
            .await
            .unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn input_constraint_max_length() {
        let g = ToolPolicyGuardrail::new(
            vec![ToolRule {
                tool_pattern: "*".into(),
                action: GuardAction::Allow,
                input_constraints: vec![InputConstraint::MaxFieldLength {
                    path: "content".into(),
                    max_bytes: 10,
                }],
            }],
            GuardAction::Allow,
        );
        let action = g
            .pre_tool(&test_call("write", serde_json::json!({"content": "short"})))
            .await
            .unwrap();
        assert_eq!(action, GuardAction::Allow);

        let action = g
            .pre_tool(&test_call(
                "write",
                serde_json::json!({"content": "this is way too long for the limit"}),
            ))
            .await
            .unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn rule_priority_first_match_wins() {
        let g = ToolPolicyGuardrail::new(
            vec![
                ToolRule {
                    tool_pattern: "bash".into(),
                    action: GuardAction::deny("bash blocked"),
                    input_constraints: vec![],
                },
                ToolRule {
                    tool_pattern: "*".into(),
                    action: GuardAction::Allow,
                    input_constraints: vec![],
                },
            ],
            GuardAction::Allow,
        );
        let action = g
            .pre_tool(&test_call("bash", serde_json::json!({})))
            .await
            .unwrap();
        assert!(action.is_denied());

        let action = g
            .pre_tool(&test_call("read", serde_json::json!({})))
            .await
            .unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn warn_action_works() {
        let g = ToolPolicyGuardrail::new(
            vec![ToolRule {
                tool_pattern: "gmail_send_*".into(),
                action: GuardAction::warn("monitoring send operations"),
                input_constraints: vec![],
            }],
            GuardAction::Allow,
        );
        let action = g
            .pre_tool(&test_call("gmail_send_message", serde_json::json!({})))
            .await
            .unwrap();
        assert!(matches!(action, GuardAction::Warn { .. }));
    }

    #[test]
    fn glob_match_exact() {
        assert!(glob_match("bash", "bash"));
        assert!(!glob_match("bash", "read"));
    }

    #[test]
    fn glob_match_star() {
        assert!(glob_match("gmail_*", "gmail_send"));
        assert!(glob_match("gmail_*", "gmail_"));
        assert!(!glob_match("gmail_*", "slack_send"));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn glob_match_question() {
        assert!(glob_match("rea?", "read"));
        assert!(!glob_match("rea?", "reading"));
    }

    #[test]
    fn json_path_nested() {
        let val = serde_json::json!({"a": {"b": "value"}});
        assert_eq!(json_path(&val, "a.b").unwrap(), "value");
    }

    #[test]
    fn json_path_missing() {
        let val = serde_json::json!({"a": 1});
        assert!(json_path(&val, "b").is_none());
    }

    #[test]
    fn meta_name() {
        let g = ToolPolicyGuardrail::new(vec![], GuardAction::Allow);
        assert_eq!(g.name(), "tool_policy");
    }
}
