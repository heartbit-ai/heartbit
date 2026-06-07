//! `set_goal` — the entry agent installs (or replaces) its own completion
//! goal at runtime, e.g. with the acceptance criteria the `intake` recipe
//! produced. The runner shares the same [`GoalSlot`]: from the next natural
//! stop on, an INDEPENDENT judge gates completion until the objective is met
//! (or the continuation budget runs out), then the goal auto-clears.
//!
//! This is the runtime half of judge-gated completion: `entry_goal` seeds a
//! goal at build time; `set_goal` makes the criteria-extraction front half
//! actually drive the back half mid-session.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::json;

use crate::agent::goal::{GoalCondition, GoalSlot};
use crate::error::Error;
use crate::llm::BoxedProvider;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

/// Install/replace/clear the runner's completion goal at runtime.
pub struct SetGoalTool {
    slot: GoalSlot,
    judge: Arc<BoxedProvider>,
}

impl SetGoalTool {
    /// `slot` must be the SAME slot handed to the runner via
    /// `AgentRunnerBuilder::goal_slot` (the orchestrator entry path wires
    /// both). `judge` is the independent judge provider (a separate/cheaper
    /// model than the working agent).
    pub fn new(slot: GoalSlot, judge: Arc<BoxedProvider>) -> Self {
        Self { slot, judge }
    }
}

impl Tool for SetGoalTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "set_goal".into(),
            description: "Install (or replace) your completion goal: an independent judge \
                          will gate your completion until the objective's criteria are \
                          verifiably met. Use it right after planning substantive work — \
                          pass the acceptance criteria (one per line). Pass clear=true to \
                          remove the goal."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The acceptance criteria as observable behavior, one per line (e.g. '- GET /health returns 200\\n- cargo test green')."
                    },
                    "clear": {
                        "type": "boolean",
                        "description": "true removes the current goal (no judge gating)."
                    }
                }
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            if input.get("clear").and_then(|v| v.as_bool()) == Some(true) {
                *self
                    .slot
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
                return Ok(ToolOutput::success(
                    "goal cleared — completion is no longer judge-gated",
                ));
            }
            let objective = input
                .get("objective")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty());
            let Some(objective) = objective else {
                return Ok(ToolOutput::error(
                    "objective is required (or pass clear=true). Give the acceptance \
                     criteria as observable behavior, one per line.",
                ));
            };
            // Multi-line objectives are criteria lists → per-criterion judging
            // (the judge names the SPECIFIC unmet criterion).
            let multi = objective.lines().filter(|l| !l.trim().is_empty()).count() > 1;
            let goal = GoalCondition::new(objective, self.judge.clone()).with_per_criterion(multi);
            *self
                .slot
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(goal);
            let preview: String = objective.chars().take(200).collect();
            Ok(ToolOutput::success(format!(
                "goal installed — an independent judge now gates your completion until:\n{preview}"
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmProvider;
    use crate::llm::types::{CompletionRequest, CompletionResponse, StopReason, TokenUsage};

    struct NoopJudge;
    impl LlmProvider for NoopJudge {
        async fn complete(&self, _r: CompletionRequest) -> Result<CompletionResponse, Error> {
            Ok(CompletionResponse {
                content: vec![],
                stop_reason: StopReason::EndTurn,
                reasoning: None,
                usage: TokenUsage::default(),
                model: None,
            })
        }
    }

    fn tool_and_slot() -> (SetGoalTool, GoalSlot) {
        let slot: GoalSlot = Arc::new(std::sync::RwLock::new(None));
        let judge = Arc::new(BoxedProvider::from_arc(Arc::new(NoopJudge)));
        (SetGoalTool::new(slot.clone(), judge), slot)
    }

    #[tokio::test]
    async fn installs_goal_into_the_shared_slot() {
        let (tool, slot) = tool_and_slot();
        let out = tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"objective": "GET /health returns 200"}),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        let g = slot.read().unwrap();
        let goal = g.as_ref().expect("goal installed");
        assert_eq!(goal.objective(), "GET /health returns 200");
        assert!(!goal.per_criterion(), "single line → plain verdict");
    }

    #[tokio::test]
    async fn multi_line_objective_enables_per_criterion_judging() {
        let (tool, slot) = tool_and_slot();
        tool.execute(
            &crate::ExecutionContext::default(),
            json!({"objective": "- GET /health returns 200\n- cargo test green"}),
        )
        .await
        .unwrap();
        let g = slot.read().unwrap();
        assert!(g.as_ref().expect("installed").per_criterion());
    }

    #[tokio::test]
    async fn clear_removes_the_goal() {
        let (tool, slot) = tool_and_slot();
        tool.execute(
            &crate::ExecutionContext::default(),
            json!({"objective": "x"}),
        )
        .await
        .unwrap();
        assert!(slot.read().unwrap().is_some());
        let out = tool
            .execute(&crate::ExecutionContext::default(), json!({"clear": true}))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(slot.read().unwrap().is_none());
    }

    #[tokio::test]
    async fn missing_objective_is_an_error() {
        let (tool, slot) = tool_and_slot();
        let out = tool
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(slot.read().unwrap().is_none());
    }

    #[test]
    fn definition_name() {
        let (tool, _) = tool_and_slot();
        assert_eq!(tool.definition().name, "set_goal");
    }
}
