//! Per-session action budget guardrail.
//!
//! Tracks how many times each tool (or tool pattern) has been called and
//! denies further calls once the budget is exhausted. First matching rule wins.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use crate::agent::guardrail::{GuardAction, Guardrail};
use crate::error::Error;
use crate::llm::types::ToolCall;

/// A budget rule that limits how many times matching tools can be called.
pub struct BudgetRule {
    /// Glob-like pattern: `"bash"` (exact), `"write*"` (prefix), `"*"` (any).
    pub tool_pattern: String,
    /// Maximum number of calls allowed for this pattern.
    pub max_calls: usize,
}

/// Guardrail that enforces per-pattern call budgets within a session.
///
/// Each `BudgetRule` tracks its own count. When a tool call matches a rule
/// and the count exceeds `max_calls`, the call is denied. If no rule matches,
/// `default_budget` is checked (if set). Counts never reset (per-session scope).
pub struct ActionBudgetGuardrail {
    budgets: Vec<BudgetRule>,
    default_budget: Option<usize>,
    counts: Mutex<HashMap<String, usize>>,
}

impl ActionBudgetGuardrail {
    /// Create a new builder for `ActionBudgetGuardrail`.
    pub fn builder() -> ActionBudgetGuardrailBuilder {
        ActionBudgetGuardrailBuilder {
            budgets: Vec::new(),
            default_budget: None,
        }
    }
}

/// Simple glob matching: exact, prefix with trailing `*`, or `*` for any.
///
/// Duplicated from `behavioral.rs` per project "no premature abstraction" rule.
fn pattern_matches(pattern: &str, name: &str) -> bool {
    if pattern == "*" {
        true
    } else if let Some(prefix) = pattern.strip_suffix('*') {
        name.starts_with(prefix)
    } else {
        pattern == name
    }
}

impl Guardrail for ActionBudgetGuardrail {
    fn name(&self) -> &str {
        "action_budget"
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        let tool_name = &call.name;

        // Find first matching rule
        let matched_pattern = self
            .budgets
            .iter()
            .find(|rule| pattern_matches(&rule.tool_pattern, tool_name));

        let (pattern_key, max_calls) = match matched_pattern {
            Some(rule) => (rule.tool_pattern.clone(), rule.max_calls),
            None => match self.default_budget {
                Some(max) => ("*".to_owned(), max),
                None => return Box::pin(async { Ok(GuardAction::Allow) }),
            },
        };

        // Lock is not held across .await — synchronous increment
        let count = {
            let mut counts = self
                .counts
                .lock()
                .expect("action_budget counts lock poisoned");
            let entry = counts.entry(pattern_key.clone()).or_insert(0);
            *entry += 1;
            *entry
        };

        let action = if count > max_calls {
            GuardAction::deny(format!(
                "Tool `{}` denied: budget exhausted for pattern `{}` ({}/{})",
                tool_name, pattern_key, count, max_calls
            ))
        } else {
            GuardAction::Allow
        };

        Box::pin(async move { Ok(action) })
    }
}

/// Builder for [`ActionBudgetGuardrail`].
pub struct ActionBudgetGuardrailBuilder {
    budgets: Vec<BudgetRule>,
    default_budget: Option<usize>,
}

impl ActionBudgetGuardrailBuilder {
    /// Add a budget rule for a tool pattern.
    pub fn rule(mut self, tool_pattern: impl Into<String>, max_calls: usize) -> Self {
        self.budgets.push(BudgetRule {
            tool_pattern: tool_pattern.into(),
            max_calls,
        });
        self
    }

    /// Set a default budget for tools that don't match any rule.
    pub fn default_budget(mut self, max_calls: usize) -> Self {
        self.default_budget = Some(max_calls);
        self
    }

    /// Build the guardrail.
    pub fn build(self) -> ActionBudgetGuardrail {
        ActionBudgetGuardrail {
            budgets: self.budgets,
            default_budget: self.default_budget,
            counts: Mutex::new(HashMap::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_call(name: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn five_bash_calls_allowed_sixth_denied() {
        let g = ActionBudgetGuardrail::builder().rule("bash", 5).build();

        for i in 1..=5 {
            let action = g.pre_tool(&test_call("bash")).await.unwrap();
            assert_eq!(action, GuardAction::Allow, "call {i} should be allowed");
        }

        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert!(action.is_denied(), "6th call should be denied");
    }

    #[tokio::test]
    async fn default_budget_applies_to_unconfigured_tools() {
        let g = ActionBudgetGuardrail::builder()
            .rule("bash", 10)
            .default_budget(2)
            .build();

        // "read" has no explicit rule, should use default budget of 2
        let action = g.pre_tool(&test_call("read")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        let action = g.pre_tool(&test_call("read")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        let action = g.pre_tool(&test_call("read")).await.unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn different_patterns_track_independently() {
        let g = ActionBudgetGuardrail::builder()
            .rule("bash", 1)
            .rule("read", 1)
            .build();

        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        let action = g.pre_tool(&test_call("read")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);

        // Both should now be exhausted
        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert!(action.is_denied());
        let action = g.pre_tool(&test_call("read")).await.unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn glob_matching_works() {
        let g = ActionBudgetGuardrail::builder().rule("write*", 2).build();

        let action = g.pre_tool(&test_call("write_file")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        let action = g.pre_tool(&test_call("write_config")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        // Both matched "write*", so the shared count is now 2
        let action = g.pre_tool(&test_call("write_other")).await.unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn first_matching_rule_wins() {
        let g = ActionBudgetGuardrail::builder()
            .rule("bash", 1) // specific rule: 1 call
            .rule("*", 100) // catch-all: 100 calls
            .build();

        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
        // bash matches first rule (max 1), so second call denied
        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert!(action.is_denied());

        // other tools match "*" rule and have plenty of budget
        let action = g.pre_tool(&test_call("read")).await.unwrap();
        assert_eq!(action, GuardAction::Allow);
    }

    #[tokio::test]
    async fn no_rules_no_default_allows_all() {
        let g = ActionBudgetGuardrail::builder().build();

        for _ in 0..100 {
            let action = g.pre_tool(&test_call("bash")).await.unwrap();
            assert_eq!(action, GuardAction::Allow);
        }
    }

    #[test]
    fn builder_pattern_works() {
        let g = ActionBudgetGuardrail::builder()
            .rule("bash", 5)
            .rule("write*", 10)
            .default_budget(20)
            .build();

        assert_eq!(g.budgets.len(), 2);
        assert_eq!(g.budgets[0].tool_pattern, "bash");
        assert_eq!(g.budgets[0].max_calls, 5);
        assert_eq!(g.budgets[1].tool_pattern, "write*");
        assert_eq!(g.budgets[1].max_calls, 10);
        assert_eq!(g.default_budget, Some(20));
    }

    #[tokio::test]
    async fn count_never_resets() {
        let g = ActionBudgetGuardrail::builder().rule("bash", 2).build();

        // Use up budget
        g.pre_tool(&test_call("bash")).await.unwrap();
        g.pre_tool(&test_call("bash")).await.unwrap();

        // Call other tools in between
        g.pre_tool(&test_call("read")).await.unwrap();
        g.pre_tool(&test_call("write")).await.unwrap();

        // bash should still be denied
        let action = g.pre_tool(&test_call("bash")).await.unwrap();
        assert!(action.is_denied());
    }

    #[test]
    fn meta_name() {
        let g = ActionBudgetGuardrail::builder().build();
        assert_eq!(g.name(), "action_budget");
    }

    #[test]
    fn pattern_matches_exact() {
        assert!(pattern_matches("bash", "bash"));
        assert!(!pattern_matches("bash", "read"));
    }

    #[test]
    fn pattern_matches_glob() {
        assert!(pattern_matches("write*", "write_file"));
        assert!(pattern_matches("write*", "write"));
        assert!(!pattern_matches("write*", "read"));
    }

    #[test]
    fn pattern_matches_wildcard() {
        assert!(pattern_matches("*", "anything"));
        assert!(pattern_matches("*", ""));
    }
}
