//! Handoff tool for agent-to-agent conversation transfer.
//!
//! When an agent calls `handoff`, it signals that conversation control should
//! transfer to a different agent. The `HandoffRunner` detects this sentinel
//! output and routes the conversation accordingly.

#![allow(missing_docs)]
use std::future::Future;
use std::pin::Pin;

use serde::Deserialize;
use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

/// How conversation context is transferred during a handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoffContextMode {
    /// Forward the full conversation history to the target agent.
    Full,
    /// Summarize the conversation and forward only the summary.
    Summary,
}

/// A target agent that can receive a handoff.
#[derive(Debug, Clone)]
pub struct HandoffTarget {
    pub name: String,
    pub description: String,
}

/// Sentinel prefix in tool output that signals a handoff.
pub(crate) const HANDOFF_SENTINEL: &str = "__handoff__:";

/// Tool that allows an agent to hand off conversation control to a peer agent.
///
/// When called, returns a sentinel `ToolOutput` that the `HandoffRunner`
/// detects to trigger the conversation transfer. The agent loop treats this
/// as a normal tool result, but `HandoffRunner` inspects the final output.
pub struct HandoffTool {
    targets: Vec<HandoffTarget>,
    cached_definition: ToolDefinition,
}

impl HandoffTool {
    /// Create a new handoff tool with the given target agents.
    pub fn new(targets: Vec<HandoffTarget>) -> Self {
        let target_descriptions: Vec<serde_json::Value> = targets
            .iter()
            .map(|t| json!({"name": t.name, "description": t.description}))
            .collect();

        let cached_definition = ToolDefinition {
            name: "handoff".into(),
            description: format!(
                "Transfer conversation control to another agent. Use this when the user's \
                 request is better handled by a different specialist. The target agent will \
                 receive the conversation context and continue where you left off.\n\n\
                 Available targets: {}",
                serde_json::to_string(&target_descriptions)
                    .expect("target serialization is infallible")
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Name of the agent to hand off to"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why you're handing off (forwarded to the target agent as context)"
                    },
                    "context_mode": {
                        "type": "string",
                        "enum": ["full", "summary"],
                        "default": "summary",
                        "description": "How to transfer conversation context: 'full' forwards the entire history, 'summary' sends a compact summary (default)"
                    }
                },
                "required": ["target", "reason"]
            }),
        };

        Self {
            targets,
            cached_definition,
        }
    }

    /// Returns the list of valid target agent names.
    pub fn target_names(&self) -> Vec<&str> {
        self.targets.iter().map(|t| t.name.as_str()).collect()
    }
}

#[derive(Deserialize)]
struct HandoffInput {
    target: String,
    reason: String,
    #[serde(default)]
    context_mode: HandoffContextModeInput,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum HandoffContextModeInput {
    Full,
    #[default]
    Summary,
}

impl Tool for HandoffTool {
    fn definition(&self) -> ToolDefinition {
        self.cached_definition.clone()
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let handoff: HandoffInput = serde_json::from_value(input)
                .map_err(|e| Error::Agent(format!("Invalid handoff input: {e}")))?;

            // Validate target
            if !self.targets.iter().any(|t| t.name == handoff.target) {
                return Ok(ToolOutput::error(format!(
                    "Unknown handoff target '{}'. Available: {}",
                    handoff.target,
                    self.targets
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }

            let mode = match handoff.context_mode {
                HandoffContextModeInput::Full => "full",
                HandoffContextModeInput::Summary => "summary",
            };

            // Return sentinel that HandoffRunner detects
            Ok(ToolOutput::success(format!(
                "{HANDOFF_SENTINEL}{target}:{mode}:{reason}",
                target = handoff.target,
                reason = handoff.reason,
            )))
        })
    }
}

/// Parse a handoff sentinel from agent output text.
///
/// Returns `(target_name, context_mode, reason)` if the text contains a handoff sentinel.
pub(crate) fn parse_handoff_sentinel(text: &str) -> Option<(String, HandoffContextMode, String)> {
    let sentinel_line = text
        .lines()
        .find(|line| line.starts_with(HANDOFF_SENTINEL))?;
    let payload = sentinel_line.strip_prefix(HANDOFF_SENTINEL)?;

    // Format: target:mode:reason
    let mut parts = payload.splitn(3, ':');
    let target = parts.next()?.to_string();
    let mode_str = parts.next().unwrap_or("summary");
    let reason = parts.next().unwrap_or("").to_string();

    let mode = match mode_str {
        "full" => HandoffContextMode::Full,
        _ => HandoffContextMode::Summary,
    };

    Some((target, mode, reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handoff_tool_definition() {
        let tool = HandoffTool::new(vec![
            HandoffTarget {
                name: "billing".into(),
                description: "Billing specialist".into(),
            },
            HandoffTarget {
                name: "support".into(),
                description: "General support".into(),
            },
        ]);

        let def = tool.definition();
        assert_eq!(def.name, "handoff");
        assert!(def.description.contains("billing"));
        assert!(def.description.contains("support"));
    }

    #[test]
    fn target_names() {
        let tool = HandoffTool::new(vec![
            HandoffTarget {
                name: "a".into(),
                description: "Agent A".into(),
            },
            HandoffTarget {
                name: "b".into(),
                description: "Agent B".into(),
            },
        ]);
        assert_eq!(tool.target_names(), vec!["a", "b"]);
    }

    #[tokio::test]
    async fn handoff_to_valid_target() {
        let tool = HandoffTool::new(vec![HandoffTarget {
            name: "billing".into(),
            description: "Billing".into(),
        }]);

        let result = tool
            .execute(json!({
                "target": "billing",
                "reason": "User has a billing question"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains(HANDOFF_SENTINEL));
        assert!(result.content.contains("billing"));
        assert!(result.content.contains("User has a billing question"));
    }

    #[tokio::test]
    async fn handoff_to_invalid_target() {
        let tool = HandoffTool::new(vec![HandoffTarget {
            name: "billing".into(),
            description: "Billing".into(),
        }]);

        let result = tool
            .execute(json!({
                "target": "nonexistent",
                "reason": "test"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("Unknown handoff target"));
    }

    #[tokio::test]
    async fn handoff_full_context_mode() {
        let tool = HandoffTool::new(vec![HandoffTarget {
            name: "support".into(),
            description: "Support".into(),
        }]);

        let result = tool
            .execute(json!({
                "target": "support",
                "reason": "needs help",
                "context_mode": "full"
            }))
            .await
            .unwrap();

        assert!(result.content.contains(":full:"));
    }

    #[test]
    fn parse_sentinel_valid() {
        let text = format!("{HANDOFF_SENTINEL}billing:summary:User wants billing help");
        let (target, mode, reason) = parse_handoff_sentinel(&text).unwrap();
        assert_eq!(target, "billing");
        assert_eq!(mode, HandoffContextMode::Summary);
        assert_eq!(reason, "User wants billing help");
    }

    #[test]
    fn parse_sentinel_full_mode() {
        let text = format!("{HANDOFF_SENTINEL}support:full:Complex issue");
        let (target, mode, reason) = parse_handoff_sentinel(&text).unwrap();
        assert_eq!(target, "support");
        assert_eq!(mode, HandoffContextMode::Full);
        assert_eq!(reason, "Complex issue");
    }

    #[test]
    fn parse_sentinel_missing() {
        assert!(parse_handoff_sentinel("normal output text").is_none());
    }

    #[test]
    fn parse_sentinel_embedded_in_output() {
        let text = format!(
            "I'll transfer you now.\n{HANDOFF_SENTINEL}billing:summary:billing question\nDone."
        );
        let (target, _, _) = parse_handoff_sentinel(&text).unwrap();
        assert_eq!(target, "billing");
    }

    #[tokio::test]
    async fn handoff_invalid_json() {
        let tool = HandoffTool::new(vec![]);
        let result = tool.execute(json!({"wrong": "fields"})).await;
        assert!(result.is_err());
    }

    #[test]
    fn parse_sentinel_reason_with_colons() {
        let text = format!("{HANDOFF_SENTINEL}agent:full:reason:with:colons");
        let (target, mode, reason) = parse_handoff_sentinel(&text).unwrap();
        assert_eq!(target, "agent");
        assert_eq!(mode, HandoffContextMode::Full);
        assert_eq!(reason, "reason:with:colons");
    }
}
