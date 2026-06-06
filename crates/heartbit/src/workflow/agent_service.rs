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
    /// Execute an LLM completion as a Restate activity.
    ///
    /// Uses `DynLlmProvider::complete` (not `stream_complete`) because Restate
    /// activities return serialized responses — there is no streaming channel
    /// back to the caller. Streaming is only used in the standalone CLI path.
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
                // Record success with circuit breaker (best-effort — don't mask LLM result)
                if let Err(e) = ctx
                    .object_client::<CircuitBreakerObjectClient>(&self.provider_name)
                    .record_success()
                    .call()
                    .await
                {
                    tracing::warn!(error = %e, "failed to record success with circuit breaker");
                }
                Ok(Json(LlmCallResponse::from(response)))
            }
            Err(e) => {
                // Record failure with circuit breaker (best-effort — don't mask LLM error)
                if let Err(cb_err) = ctx
                    .object_client::<CircuitBreakerObjectClient>(&self.provider_name)
                    .record_failure()
                    .call()
                    .await
                {
                    tracing::warn!(error = %cb_err, "failed to record failure with circuit breaker");
                }
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

        // Validate input against the tool's declared schema before executing.
        let schema = &tool.definition().input_schema;
        if let Err(msg) = crate::tool::validate_tool_input(schema, &request.input) {
            return Ok(Json(ToolCallResponse {
                content: msg,
                is_error: true,
            }));
        }

        // Capture name and timeout up-front because the helper consumes `request`.
        let tool_name = request.tool_name.clone();
        let timeout_seconds = request.timeout_seconds;
        let max_output_bytes = request.max_output_bytes;

        let result = match timeout_seconds {
            Some(secs) => {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(secs),
                    tool_call_inner(&self.tools, request),
                )
                .await
                {
                    Ok(r) => r,
                    Err(_) => {
                        return Ok(Json(ToolCallResponse {
                            content: format!(
                                "Tool '{tool_name}' execution timed out after {secs}s"
                            ),
                            is_error: true,
                        }));
                    }
                }
            }
            None => tool_call_inner(&self.tools, request).await,
        };

        match result {
            Ok(output) => {
                let output = match max_output_bytes {
                    Some(max) => output.truncated(max),
                    None => output,
                };
                Ok(Json(ToolCallResponse {
                    content: output.content,
                    is_error: output.is_error,
                }))
            }
            Err(e) => Ok(Json(ToolCallResponse {
                content: format!("Tool '{tool_name}' error: {e}"),
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

/// Dispatch a tool call by constructing an `ExecutionContext` from the
/// invocation params and delegating to `Tool::execute`.
///
/// Looks up the tool in `tools` and returns `Error::Agent` if absent. The
/// caller (the Restate `tool_call` activity) handles graceful "tool not
/// found" responses up-front; a not-found here is defensive against drift
/// between caller checks and the helper.
async fn tool_call_inner(
    tools: &HashMap<String, Arc<dyn Tool>>,
    request: ToolCallRequest,
) -> Result<crate::tool::ToolOutput, crate::Error> {
    let tool = tools
        .get(&request.tool_name)
        .ok_or_else(|| crate::Error::Agent(format!("tool '{}' not found", request.tool_name)))?;
    let ctx = heartbit_core::ExecutionContext {
        tenant_id: request.tenant_id,
        user_id: request.user_id,
        ..Default::default()
    };
    tool.execute(&ctx, request.input).await
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
            _ctx: &heartbit_core::ExecutionContext,
            _input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
            let output = self.output.clone();
            Box::pin(async move { Ok(output) })
        }
    }

    #[tokio::test]
    async fn tool_call_activity_constructs_context_from_invocation_params() {
        use heartbit_core::ExecutionContext;
        use std::sync::Mutex;

        struct CtxCapture {
            captured_tenant: Arc<Mutex<Option<String>>>,
            captured_user: Arc<Mutex<Option<String>>>,
        }

        impl Tool for CtxCapture {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "ctx_capture".into(),
                    description: "captures tenant".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }
            }

            fn execute(
                &self,
                ctx: &ExecutionContext,
                _input: serde_json::Value,
            ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
                let cap_t = self.captured_tenant.clone();
                let cap_u = self.captured_user.clone();
                let t = ctx.tenant_id.clone();
                let u = ctx.user_id.clone();
                Box::pin(async move {
                    *cap_t.lock().unwrap() = t;
                    *cap_u.lock().unwrap() = u;
                    Ok(ToolOutput::success("ok"))
                })
            }
        }

        let captured_tenant = Arc::new(Mutex::new(None));
        let captured_user = Arc::new(Mutex::new(None));
        let tool = Arc::new(CtxCapture {
            captured_tenant: captured_tenant.clone(),
            captured_user: captured_user.clone(),
        });
        let mut tools: HashMap<String, Arc<dyn Tool>> = HashMap::new();
        tools.insert("ctx_capture".into(), tool as Arc<dyn Tool>);

        let request = ToolCallRequest {
            tool_name: "ctx_capture".into(),
            input: serde_json::json!({}),
            timeout_seconds: None,
            max_output_bytes: None,
            tenant_id: Some("restate-tenant".into()),
            user_id: Some("restate-user".into()),
        };

        let result = tool_call_inner(&tools, request).await.expect("ok result");
        assert!(!result.is_error);

        assert_eq!(
            captured_tenant.lock().unwrap().as_deref(),
            Some("restate-tenant")
        );
        assert_eq!(
            captured_user.lock().unwrap().as_deref(),
            Some("restate-user")
        );
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
                model: None,
                reasoning: None,
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
