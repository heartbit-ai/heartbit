use restate_sdk::prelude::*;

use crate::agent::context::{apply_sliding_window, inject_summary_into_messages, messages_to_text};
use crate::agent::token_estimator::estimate_message_tokens;
use crate::llm::types::{ContentBlock, Message, Role, StopReason, TokenUsage};

use super::agent_service::AgentServiceClient;
use super::budget::TokenBudgetObjectClient;
use super::types::{
    AgentResult, AgentStatus, AgentTask, HumanDecision, LlmCallRequest, ToolCallRequest,
    ToolCallResponse,
};

/// Durable agent workflow — the ReAct loop as a Restate workflow.
///
/// Each LLM call and tool execution is a durable activity (via `AgentService`).
/// On crash, the workflow replays from its event journal, skipping completed
/// activities. The agent resumes exactly where it left off.
#[restate_sdk::workflow]
pub trait AgentWorkflow {
    /// The durable agent loop.
    async fn run(task: Json<AgentTask>) -> Result<Json<AgentResult>, HandlerError>;

    /// Signal: human-in-the-loop approval.
    #[shared]
    async fn approve(decision: Json<HumanDecision>) -> Result<(), HandlerError>;

    /// Query: get current workflow status.
    #[shared]
    async fn status() -> Result<Json<AgentStatus>, HandlerError>;
}

pub struct AgentWorkflowImpl;

impl AgentWorkflow for AgentWorkflowImpl {
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(task): Json<AgentTask>,
    ) -> Result<Json<AgentResult>, HandlerError> {
        // Auto-discover tools from AgentService if none were provided.
        // The list_tools() call is journaled by Restate, so replay is deterministic.
        let tool_defs = if task.tool_defs.is_empty() {
            match ctx
                .service_client::<AgentServiceClient>()
                .list_tools()
                .call()
                .await
            {
                Ok(Json(defs)) => defs,
                Err(e) => {
                    tracing::warn!(error = %e, "list_tools failed, agent will run without tools");
                    vec![]
                }
            }
        } else {
            task.tool_defs.clone()
        };

        let mut messages = vec![Message::user(&task.input)];
        let mut total_usage = TokenUsage::default();
        let mut total_tool_calls: usize = 0;

        // Store initial status
        ctx.set("state", "running".to_string());
        ctx.set("max_turns", task.max_turns as u64);

        for turn in 0..task.max_turns {
            ctx.set("current_turn", turn as u64);

            // Apply sliding window if configured
            let request_messages = if let Some(window_tokens) = task.context_window_tokens {
                apply_sliding_window(&messages, window_tokens)
            } else {
                messages.clone()
            };

            // LLM call — durable activity, result persisted in journal
            let llm_response = ctx
                .service_client::<AgentServiceClient>()
                .llm_call(Json(LlmCallRequest {
                    system: task.system_prompt.clone(),
                    messages: request_messages,
                    tools: tool_defs.clone(),
                    max_tokens: task.max_tokens,
                }))
                .call()
                .await?;

            let llm_response = llm_response.into_inner();
            total_usage.input_tokens += llm_response.usage.input_tokens;
            total_usage.output_tokens += llm_response.usage.output_tokens;

            // Record token usage with budget tracker (TerminalError if exceeded)
            ctx.object_client::<TokenBudgetObjectClient>(ctx.key())
                .record_usage(Json(super::budget::TokenUsageRecord {
                    input_tokens: llm_response.usage.input_tokens as u64,
                    output_tokens: llm_response.usage.output_tokens as u64,
                }))
                .call()
                .await?;

            // Add assistant message to conversation
            messages.push(Message {
                role: Role::Assistant,
                content: llm_response.content.clone(),
            });

            // If no tool calls, we're done
            if !llm_response.has_tool_calls() {
                if llm_response.stop_reason == StopReason::MaxTokens {
                    ctx.set("state", "error".to_string());
                    return Err(
                        TerminalError::new("Response truncated (max_tokens reached)").into(),
                    );
                }

                ctx.set("state", "completed".to_string());
                return Ok(Json(AgentResult {
                    text: llm_response.text(),
                    tokens: total_usage,
                    tool_calls_made: total_tool_calls,
                }));
            }

            // Execute tool calls — each is a durable activity
            let tool_calls = extract_tool_calls(&llm_response.content);
            total_tool_calls += tool_calls.len();

            // Human-in-the-loop gate: await approval before executing tools.
            // Each turn gets a unique promise key so that approvals are
            // consumed once and subsequent turns block on fresh promises.
            if task.approval_required && !tool_calls.is_empty() {
                let promise_key = format!("approval-turn-{turn}");
                ctx.set("state", "awaiting_approval".to_string());
                ctx.set("approval_turn", turn as u64);
                let Json(decision) = ctx.promise::<Json<HumanDecision>>(&promise_key).await?;
                if !decision.approved {
                    ctx.set("state", "rejected".to_string());
                    return Err(TerminalError::new(format!(
                        "Rejected by human: {}",
                        decision.reason.unwrap_or_default()
                    ))
                    .into());
                }
                ctx.set("state", "running".to_string());
            }

            // Tool calls run sequentially here (not parallel like standalone).
            // Restate SDK requires sequential journal entries for deterministic
            // replay. Parallelism happens at the Restate server level instead.
            let mut tool_result_blocks = Vec::with_capacity(tool_calls.len());
            for (id, name, input) in &tool_calls {
                let result: ToolCallResponse = ctx
                    .service_client::<AgentServiceClient>()
                    .tool_call(Json(ToolCallRequest {
                        tool_name: name.clone(),
                        input: input.clone(),
                    }))
                    .call()
                    .await?
                    .into_inner();

                tool_result_blocks.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: result.content,
                    is_error: result.is_error,
                });
            }

            // Add tool results as a user message
            messages.push(Message {
                role: Role::User,
                content: tool_result_blocks,
            });

            // Summarization: compress context if it exceeds the threshold.
            // Uses an LLM call through AgentService for durability.
            if let Some(threshold) = task.summarize_threshold {
                let total_tokens: u32 = messages.iter().map(estimate_message_tokens).sum();
                if total_tokens > threshold && messages.len() > 5 {
                    let summary_resp = ctx
                        .service_client::<AgentServiceClient>()
                        .llm_call(Json(LlmCallRequest {
                            system: "You are a summarization assistant. Summarize the following \
                                     conversation concisely, preserving key facts, decisions, \
                                     and tool results. Focus on information needed to continue \
                                     the conversation."
                                .into(),
                            messages: vec![Message::user(messages_to_text(&messages))],
                            tools: vec![],
                            max_tokens: 1024,
                        }))
                        .call()
                        .await?
                        .into_inner();

                    let summary = summary_resp.text();
                    total_usage.input_tokens += summary_resp.usage.input_tokens;
                    total_usage.output_tokens += summary_resp.usage.output_tokens;

                    // Merge summary into first message, keep last 4
                    inject_summary_into_messages(&mut messages, &task.input, &summary, 4);
                }
            }
        }

        ctx.set("state", "error".to_string());
        Err(TerminalError::new(format!("Max turns ({}) exceeded", task.max_turns)).into())
    }

    async fn approve(
        &self,
        ctx: SharedWorkflowContext<'_>,
        Json(decision): Json<HumanDecision>,
    ) -> Result<(), HandlerError> {
        // Use the explicit turn from the decision if provided (avoids TOCTOU race),
        // otherwise fall back to reading the current approval_turn from state.
        let turn = match decision.turn {
            Some(t) => t,
            None => ctx.get::<u64>("approval_turn").await?.unwrap_or(0),
        };
        let promise_key = format!("approval-turn-{turn}");
        ctx.resolve_promise::<Json<HumanDecision>>(&promise_key, Json(decision));
        Ok(())
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

        Ok(Json(AgentStatus {
            current_turn: current_turn as usize,
            max_turns: max_turns as usize,
            state,
            child_workflows: vec![],
        }))
    }
}

/// Extract tool call info from content blocks.
fn extract_tool_calls(content: &[ContentBlock]) -> Vec<(String, String, serde_json::Value)> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, input } => {
                Some((id.clone(), name.clone(), input.clone()))
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_tool_calls_from_mixed_content() {
        let content = vec![
            ContentBlock::Text {
                text: "Let me search.".into(),
            },
            ContentBlock::ToolUse {
                id: "tc_1".into(),
                name: "search".into(),
                input: serde_json::json!({"q": "rust"}),
            },
            ContentBlock::ToolUse {
                id: "tc_2".into(),
                name: "read".into(),
                input: serde_json::json!({"path": "/tmp"}),
            },
        ];

        let calls = extract_tool_calls(&content);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "tc_1");
        assert_eq!(calls[0].1, "search");
        assert_eq!(calls[1].0, "tc_2");
        assert_eq!(calls[1].1, "read");
    }

    #[test]
    fn extract_tool_calls_empty_when_no_tools() {
        let content = vec![ContentBlock::Text {
            text: "Just text.".into(),
        }];
        assert!(extract_tool_calls(&content).is_empty());
    }
}
