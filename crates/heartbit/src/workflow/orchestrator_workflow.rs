use restate_sdk::prelude::*;

use crate::agent::orchestrator::{build_delegate_tool_schema, build_system_prompt};
use crate::llm::types::{ContentBlock, Message, Role, StopReason, TokenUsage};

use super::agent_service::AgentServiceClient;
use super::agent_workflow::AgentWorkflowClient;
use super::blackboard::{BlackboardEntry, BlackboardObjectClient};
use super::types::{
    AgentDef, AgentStatus, AgentTask, LlmCallRequest, LlmCallResponse, OrchestratorResult,
    OrchestratorTask,
};

/// Durable orchestrator workflow — delegates tasks to child agent workflows.
///
/// The orchestrator uses the LLM to decide what to delegate, spawns child
/// `AgentWorkflow` instances via durable RPC, and synthesizes results.
#[restate_sdk::workflow]
pub trait OrchestratorWorkflow {
    async fn run(task: Json<OrchestratorTask>) -> Result<Json<OrchestratorResult>, HandlerError>;

    /// Query: get current orchestrator workflow status.
    #[shared]
    async fn status() -> Result<Json<AgentStatus>, HandlerError>;
}

pub struct OrchestratorWorkflowImpl;

impl OrchestratorWorkflow for OrchestratorWorkflowImpl {
    async fn run(
        &self,
        mut ctx: WorkflowContext<'_>,
        Json(task): Json<OrchestratorTask>,
    ) -> Result<Json<OrchestratorResult>, HandlerError> {
        let mut total_usage = TokenUsage::default();

        // Store initial status
        ctx.set("state", "running".to_string());
        ctx.set("max_turns", task.max_turns as u64);

        // Build orchestrator system prompt and delegate tool def using shared functions
        let pairs: Vec<(&str, &str)> = task
            .agents
            .iter()
            .map(|a| (a.name.as_str(), a.description.as_str()))
            .collect();
        let system = build_system_prompt(&pairs);
        let delegate_tool_def = build_delegate_tool_schema(&pairs);

        let mut messages = vec![Message::user(&task.input)];

        for turn in 0..task.max_turns {
            ctx.set("current_turn", turn as u64);

            // Step 1: Ask the orchestrator LLM what to do (durable activity)
            let llm_response: LlmCallResponse = ctx
                .service_client::<AgentServiceClient>()
                .llm_call(Json(LlmCallRequest {
                    system: system.clone(),
                    messages: messages.clone(),
                    tools: vec![delegate_tool_def.clone()],
                    max_tokens: task.max_tokens,
                }))
                .call()
                .await?
                .into_inner();

            total_usage.input_tokens += llm_response.usage.input_tokens;
            total_usage.output_tokens += llm_response.usage.output_tokens;

            messages.push(Message {
                role: Role::Assistant,
                content: llm_response.content.clone(),
            });

            // If no tool calls, return the direct response
            if !llm_response.has_tool_calls() {
                if llm_response.stop_reason == StopReason::MaxTokens {
                    return Err(
                        TerminalError::new("Response truncated (max_tokens reached)").into(),
                    );
                }
                ctx.set("state", "completed".to_string());
                return Ok(Json(OrchestratorResult {
                    text: llm_response.text(),
                    tokens: total_usage,
                }));
            }

            // Step 2: Execute delegated tasks via child agent workflows
            let mut tool_result_blocks = Vec::new();
            for block in &llm_response.content {
                if let ContentBlock::ToolUse { id, name, input } = block {
                    if name == "delegate_task" {
                        let (result, is_error, sub_tokens) = execute_delegation(
                            &mut ctx,
                            input,
                            &task.agents,
                            task.max_turns,
                            task.max_tokens,
                            task.approval_required,
                        )
                        .await;
                        total_usage.input_tokens += sub_tokens.input_tokens;
                        total_usage.output_tokens += sub_tokens.output_tokens;
                        tool_result_blocks.push(ContentBlock::ToolResult {
                            tool_use_id: id.clone(),
                            content: result,
                            is_error,
                        });
                    } else {
                        tool_result_blocks.push(ContentBlock::ToolResult {
                            tool_use_id: id.clone(),
                            content: format!("Unknown tool: {name}"),
                            is_error: true,
                        });
                    }
                }
            }

            messages.push(Message {
                role: Role::User,
                content: tool_result_blocks,
            });
        }

        ctx.set("state", "error".to_string());
        Err(TerminalError::new(format!("Max turns ({}) exceeded", task.max_turns)).into())
    }

    async fn status(
        &self,
        ctx: SharedWorkflowContext<'_>,
    ) -> Result<Json<AgentStatus>, HandlerError> {
        let current_turn = ctx.get::<u64>("current_turn").await?.unwrap_or(0);
        let max_turns = ctx.get::<u64>("max_turns").await?.unwrap_or(0);
        let state = ctx
            .get::<String>("state")
            .await?
            .unwrap_or_else(|| "pending".into());
        let child_workflows = ctx
            .get::<String>("child_workflows")
            .await?
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        Ok(Json(AgentStatus {
            current_turn: current_turn as usize,
            max_turns: max_turns as usize,
            state,
            child_workflows,
        }))
    }
}

/// Execute delegated tasks by spawning child agent workflows.
///
/// Returns `(result_text, is_error, sub_agent_tokens)` — `is_error` is true
/// if any agent failed or if the input could not be parsed.
async fn execute_delegation(
    ctx: &mut WorkflowContext<'_>,
    input: &serde_json::Value,
    agents: &[AgentDef],
    max_turns: usize,
    max_tokens: u32,
    approval_required: bool,
) -> (String, bool, TokenUsage) {
    #[derive(serde::Deserialize)]
    struct DelegateInput {
        #[serde(default)]
        tasks: Vec<DelegateTask>,
    }

    #[derive(serde::Deserialize)]
    struct DelegateTask {
        agent: String,
        task: String,
    }

    let delegate_input: DelegateInput = match serde_json::from_value(input.clone()) {
        Ok(v) => v,
        Err(e) => {
            return (
                format!("Error parsing delegate_task input: {e}"),
                true,
                TokenUsage::default(),
            );
        }
    };

    let mut results = Vec::new();
    let mut any_error = false;
    let mut sub_tokens = TokenUsage::default();
    for dt in &delegate_input.tasks {
        let agent_def = match agents.iter().find(|a| a.name == dt.agent) {
            Some(def) => def,
            None => {
                any_error = true;
                results.push(format!(
                    "=== Agent: {} ===\nError: unknown agent '{}'",
                    dt.agent, dt.agent
                ));
                continue;
            }
        };

        // Spawn child agent workflow via durable RPC
        // NOTE: Restate SDK requires sequential journal entries for deterministic
        // replay, so child workflows are dispatched sequentially here. True
        // parallelism happens at the Restate server level across workflow instances.
        let workflow_id = format!("{}-{}", dt.agent, ctx.rand_uuid());

        // Track child workflow ID for status queries and approval routing
        let child_ids_json = match ctx.get::<String>("child_workflows").await {
            Ok(v) => v.unwrap_or_else(|| "[]".to_string()),
            Err(e) => {
                tracing::warn!(error = %e, "failed to read child_workflows state");
                "[]".to_string()
            }
        };
        let mut ids: Vec<String> = serde_json::from_str(&child_ids_json).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "corrupt child_workflows state, resetting");
            vec![]
        });
        ids.push(workflow_id.clone());
        let updated = serde_json::to_string(&ids).expect("Vec<String> serialization is infallible");
        ctx.set("child_workflows", updated);

        let agent_result = ctx
            .workflow_client::<AgentWorkflowClient>(&workflow_id)
            .run(Json(AgentTask {
                input: dt.task.clone(),
                system_prompt: agent_def.system_prompt.clone(),
                tool_defs: agent_def.tool_defs.clone(),
                max_turns,
                max_tokens,
                approval_required,
                context_window_tokens: agent_def.context_window_tokens,
                summarize_threshold: None,
            }))
            .call()
            .await;

        match agent_result {
            Ok(Json(result)) => {
                sub_tokens.input_tokens += result.tokens.input_tokens;
                sub_tokens.output_tokens += result.tokens.output_tokens;

                // Store result on the shared blackboard for cross-agent visibility
                if let Err(e) = ctx
                    .object_client::<BlackboardObjectClient>(ctx.key())
                    .write(Json(BlackboardEntry {
                        key: format!("agent:{}", dt.agent),
                        value: serde_json::json!({
                            "text": result.text,
                            "tokens": {
                                "input": result.tokens.input_tokens,
                                "output": result.tokens.output_tokens,
                            },
                            "tool_calls": result.tool_calls_made,
                        }),
                    }))
                    .call()
                    .await
                {
                    tracing::warn!(
                        agent = %dt.agent,
                        error = %e,
                        "failed to write agent result to blackboard"
                    );
                }

                results.push(format!("=== Agent: {} ===\n{}", dt.agent, result.text));
            }
            Err(e) => {
                any_error = true;
                results.push(format!("=== Agent: {} ===\nError: {e}", dt.agent));
            }
        }
    }

    (results.join("\n\n"), any_error, sub_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orchestrator_prompt_includes_agents() {
        let agents = vec![
            ("researcher", "Research specialist"),
            ("coder", "Coding expert"),
        ];

        let prompt = build_system_prompt(&agents);
        assert!(prompt.contains("researcher"));
        assert!(prompt.contains("Research specialist"));
        assert!(prompt.contains("coder"));
    }

    #[test]
    fn delegate_tool_def_has_correct_schema() {
        let agents = vec![("researcher", "Research")];

        let def = build_delegate_tool_schema(&agents);
        assert_eq!(def.name, "delegate_task");
        assert!(def.description.contains("researcher"));
        assert!(
            def.input_schema["properties"]["tasks"].is_array()
                || def.input_schema["properties"]["tasks"]["type"] == "array"
        );
    }
}
