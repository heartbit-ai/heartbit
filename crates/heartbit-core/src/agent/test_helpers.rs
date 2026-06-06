//! Shared test helpers for agent module tests.
//!
//! Provides [`MockProvider`] (returns pre-configured responses, captures all
//! requests for inspection) and a generic [`make_agent`] factory.

use std::sync::{Arc, Mutex};

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{
    CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
};

use super::AgentRunner;

/// Mock LLM provider that returns pre-configured responses in order and
/// captures every request for later inspection.
pub(crate) struct MockProvider {
    responses: Mutex<Vec<CompletionResponse>>,
    /// Every `CompletionRequest` received, in call order.
    pub captured_requests: Mutex<Vec<CompletionRequest>>,
}

impl MockProvider {
    pub fn new(responses: Vec<CompletionResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            captured_requests: Mutex::new(Vec::new()),
        }
    }

    pub fn text_response(text: &str, input_tokens: u32, output_tokens: u32) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            reasoning: None,
            usage: TokenUsage {
                input_tokens,
                output_tokens,
                ..Default::default()
            },
            model: None,
        }
    }
}

impl LlmProvider for MockProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        self.captured_requests
            .lock()
            .expect("capture lock poisoned")
            .push(request);
        let mut responses = self.responses.lock().expect("mock lock poisoned");
        if responses.is_empty() {
            return Err(Error::Agent("no more mock responses".into()));
        }
        Ok(responses.remove(0))
    }

    fn model_name(&self) -> Option<&str> {
        Some("mock-model")
    }
}

/// Build a minimal `AgentRunner` for tests (1 turn, fixed system prompt).
pub(crate) fn make_agent<P: LlmProvider>(provider: Arc<P>, name: &str) -> AgentRunner<P> {
    AgentRunner::builder(provider)
        .name(name)
        .system_prompt("test system prompt")
        .max_turns(1)
        .build()
        .expect("failed to build test agent")
}
