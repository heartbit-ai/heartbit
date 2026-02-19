use std::collections::HashMap;
use std::sync::Arc;

use restate_sdk::prelude::*;

use crate::llm::types::ToolDefinition;
use crate::tool::Tool;

use super::circuit_breaker::CircuitBreakerObjectClient;
use super::types::{
    DynLlmProvider, LlmCallRequest, LlmCallResponse, ToolCallRequest, ToolCallResponse,
};

/// Restate service for LLM and tool call activities.
///
/// Each handler is a durable activity: its result is persisted in the Restate
/// event journal. On replay, completed calls are short-circuited.
#[restate_sdk::service]
pub trait AgentService {
    async fn llm_call(request: Json<LlmCallRequest>)
    -> Result<Json<LlmCallResponse>, HandlerError>;
    async fn tool_call(
        request: Json<ToolCallRequest>,
    ) -> Result<Json<ToolCallResponse>, HandlerError>;
    /// List available tool definitions from the worker's tool registry.
    async fn list_tools() -> Result<Json<Vec<ToolDefinition>>, HandlerError>;
}

/// Implementation holding the LLM provider and registered tools.
///
/// The `provider_name` is used as the circuit breaker object key.
pub struct AgentServiceImpl {
    provider: Arc<dyn DynLlmProvider>,
    provider_name: String,
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl AgentServiceImpl {
    pub fn new(
        provider: Arc<dyn DynLlmProvider>,
        provider_name: impl Into<String>,
        tools: HashMap<String, Arc<dyn Tool>>,
    ) -> Self {
        Self {
            provider,
            provider_name: provider_name.into(),
            tools,
        }
    }
}

impl AgentService for AgentServiceImpl {
    async fn llm_call(
        &self,
        ctx: Context<'_>,
        Json(request): Json<LlmCallRequest>,
    ) -> Result<Json<LlmCallResponse>, HandlerError> {
        // Check circuit breaker before calling the LLM provider
        let is_open = ctx
            .object_client::<CircuitBreakerObjectClient>(&self.provider_name)
            .is_open()
            .call()
            .await?;
        if is_open {
            return Err(TerminalError::new(format!(
                "Circuit breaker open for provider '{}'",
                self.provider_name
            ))
            .into());
        }

        let completion_request = request.to_completion_request();
        let result = self.provider.complete(completion_request).await;

        match result {
            Ok(response) => {
                // Record success with circuit breaker
                ctx.object_client::<CircuitBreakerObjectClient>(&self.provider_name)
                    .record_success()
                    .call()
                    .await?;
                Ok(Json(LlmCallResponse::from(response)))
            }
            Err(e) => {
                // Record failure with circuit breaker
                ctx.object_client::<CircuitBreakerObjectClient>(&self.provider_name)
                    .record_failure()
                    .call()
                    .await?;
                Err(TerminalError::new(format!("LLM call failed: {e}")).into())
            }
        }
    }

    async fn tool_call(
        &self,
        _ctx: Context<'_>,
        Json(request): Json<ToolCallRequest>,
    ) -> Result<Json<ToolCallResponse>, HandlerError> {
        // Match standalone behavior: tool errors are recoverable (sent back to
        // LLM) rather than TerminalError (which aborts the workflow).
        let tool = match self.tools.get(&request.tool_name) {
            Some(t) => t,
            None => {
                return Ok(Json(ToolCallResponse {
                    content: format!("Tool not found: {}", request.tool_name),
                    is_error: true,
                }));
            }
        };

        let result = match request.timeout_seconds {
            Some(secs) => {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(secs),
                    tool.execute(request.input),
                )
                .await
                {
                    Ok(r) => r,
                    Err(_) => {
                        return Ok(Json(ToolCallResponse {
                            content: format!(
                                "Tool '{}' execution timed out after {secs}s",
                                request.tool_name
                            ),
                            is_error: true,
                        }));
                    }
                }
            }
            None => tool.execute(request.input).await,
        };

        match result {
            Ok(output) => Ok(Json(ToolCallResponse {
                content: output.content,
                is_error: output.is_error,
            })),
            Err(e) => Ok(Json(ToolCallResponse {
                content: format!("Tool '{}' error: {e}", request.tool_name),
                is_error: true,
            })),
        }
    }

    async fn list_tools(
        &self,
        _ctx: Context<'_>,
    ) -> Result<Json<Vec<ToolDefinition>>, HandlerError> {
        let mut defs: Vec<ToolDefinition> = self.tools.values().map(|t| t.definition()).collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Json(defs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage, ToolDefinition,
    };
    use crate::tool::ToolOutput;
    use std::future::Future;
    use std::pin::Pin;

    struct MockProvider {
        response: CompletionResponse,
    }

    impl crate::LlmProvider for MockProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, Error> {
            Ok(self.response.clone())
        }
    }

    struct MockTool {
        name: String,
        output: ToolOutput,
    }

    impl Tool for MockTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: "mock".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        fn execute(
            &self,
            _input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
            let output = self.output.clone();
            Box::pin(async move { Ok(output) })
        }
    }

    #[test]
    fn agent_service_impl_construction() {
        let provider = Arc::new(MockProvider {
            response: CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "test".into(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            },
        });

        let mut tools = HashMap::new();
        tools.insert(
            "search".into(),
            Arc::new(MockTool {
                name: "search".into(),
                output: ToolOutput::success("found it"),
            }) as Arc<dyn Tool>,
        );

        let service = AgentServiceImpl::new(provider, "test-provider", tools);
        assert_eq!(service.tools.len(), 1);
        assert!(service.tools.contains_key("search"));
    }
}
