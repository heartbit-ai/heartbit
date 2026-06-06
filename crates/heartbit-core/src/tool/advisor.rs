//! The `advisor` tool: consult a stronger (frontier) reviewer mid-run.
//!
//! Mirrors Claude Code's advisor mode: the tool takes NO parameters — the full
//! conversation transcript is forwarded automatically (snapshotted into
//! [`ExecutionContext::transcript`](crate::ExecutionContext) at tool-dispatch
//! time by the runner). The frontier model reviews the transcript skeptically
//! and answers with what blocks, what doesn't, and the single most
//! discriminating next check.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::error::Error;
use crate::llm::types::{CompletionRequest, Message, ToolDefinition};
use crate::llm::{BoxedProvider, LlmProvider};
use crate::tool::{Tool, ToolOutput};

/// Upper bound on the advisor's reply.
const ADVISOR_MAX_TOKENS: u32 = 1024;

/// The advisor's operating contract — distilled from Claude Code's advisor.
const ADVISOR_SYSTEM_PROMPT: &str = "\
You are the ADVISOR: an independent, stronger reviewer consulted by a working \
agent mid-run. The full conversation transcript so far follows. Review it \
skeptically:\n\
- Verify the agent's claims against the evidence actually present in the \
transcript (tool outputs, test results) — never against its own narration.\n\
- Name what BLOCKS the current path versus what does not. Be explicit: \
'Does it block? …'.\n\
- Give the single most discriminating check or correction, not a list of vague \
suggestions and not a verdict without a path.\n\
- If the approach is sound, say so plainly and name the one risk most worth \
watching. No flattery, no padding.\n\
Reply in prose (no JSON), under 400 words.";

/// Consult a frontier-model reviewer over the full transcript. No parameters.
pub struct AdvisorTool {
    provider: Arc<BoxedProvider>,
}

impl AdvisorTool {
    /// Build over the frontier provider (resolve it via the host's model-role
    /// factory: `factory("frontier")`).
    pub fn new(provider: Arc<BoxedProvider>) -> Self {
        Self { provider }
    }
}

impl Tool for AdvisorTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "advisor".into(),
            description: "Consult a stronger independent reviewer (frontier model). \
                          Your FULL conversation transcript is forwarded automatically — \
                          no parameters. Call BEFORE substantive work (before committing \
                          to an interpretation or writing code), when you believe the \
                          task is COMPLETE (before declaring done), and whenever you are \
                          STUCK or about to change approach. Give the advice serious \
                          weight; if it conflicts with primary evidence you hold, say so \
                          and reconcile rather than silently picking one side."
                .into(),
            input_schema: json!({"type": "object", "properties": {}}),
        }
    }

    fn execute(
        &self,
        ctx: &crate::ExecutionContext,
        _input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        let transcript = ctx.transcript.clone();
        Box::pin(async move {
            let Some(messages) = transcript else {
                return Ok(ToolOutput::error(
                    "advisor unavailable: this runner does not forward the conversation \
                     transcript to tools"
                        .to_string(),
                ));
            };
            let rendered = crate::agent::context::messages_to_text(&messages);
            let request = CompletionRequest {
                system: ADVISOR_SYSTEM_PROMPT.to_string(),
                messages: vec![Message::user(format!(
                    "=== TRANSCRIPT (oldest first) ===\n{rendered}\n=== END TRANSCRIPT ===\n\
                     Advise the working agent now."
                ))],
                tools: vec![],
                max_tokens: ADVISOR_MAX_TOKENS,
                tool_choice: None,
                reasoning_effort: None,
            };
            match LlmProvider::complete(self.provider.as_ref(), request).await {
                Ok(resp) => {
                    let text: String = resp
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            crate::llm::types::ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect();
                    if text.trim().is_empty() {
                        Ok(ToolOutput::error(
                            "advisor returned an empty review".to_string(),
                        ))
                    } else {
                        Ok(ToolOutput::success(text))
                    }
                }
                // Fail open with an honest error — advice must never kill a run.
                Err(e) => Ok(ToolOutput::error(format!("advisor unavailable: {e}"))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutionContext;
    use crate::agent::test_helpers::MockProvider;

    fn ctx_with_transcript() -> ExecutionContext {
        ExecutionContext {
            transcript: Some(Arc::new(vec![
                Message::user("please fix the divide function"),
                Message::user("I ran the tests: 3 failed"),
            ])),
            ..ExecutionContext::default()
        }
    }

    #[tokio::test]
    async fn forwards_the_full_transcript_to_the_frontier_model() {
        let mock = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "Does it block? No. Check the divisor-zero path first.",
            10,
            5,
        )]));
        let tool = AdvisorTool::new(Arc::new(BoxedProvider::from_arc(Arc::clone(&mock))));
        let out = tool
            .execute(&ctx_with_transcript(), json!({}))
            .await
            .unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert!(out.content.contains("Does it block?"));

        let reqs = mock.captured_requests.lock().expect("lock");
        assert_eq!(reqs.len(), 1);
        let req = &reqs[0];
        assert!(
            req.system.contains("ADVISOR") && req.system.contains("skeptically"),
            "advisor contract in the system prompt: {}",
            req.system
        );
        let user: String = reqs[0]
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                crate::llm::types::ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            user.contains("fix the divide function") && user.contains("3 failed"),
            "full transcript forwarded: {user}"
        );
    }

    #[tokio::test]
    async fn without_a_transcript_the_error_is_honest() {
        let mock = Arc::new(MockProvider::new(vec![]));
        let tool = AdvisorTool::new(Arc::new(BoxedProvider::from_arc(mock)));
        let out = tool
            .execute(&ExecutionContext::default(), json!({}))
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("transcript"), "{}", out.content);
    }

    #[test]
    fn definition_encodes_the_calling_contract() {
        let mock = Arc::new(MockProvider::new(vec![]));
        let tool = AdvisorTool::new(Arc::new(BoxedProvider::from_arc(mock)));
        let def = tool.definition();
        assert_eq!(def.name, "advisor");
        for phrase in ["BEFORE", "COMPLETE", "STUCK", "no parameters"] {
            assert!(def.description.contains(phrase), "missing: {phrase}");
        }
    }
}
