pub mod anthropic;
pub mod types;

use crate::llm::types::{CompletionRequest, CompletionResponse};

/// Trait for LLM providers.
///
/// Implementors must be thread-safe (`Send + Sync`) to allow
/// shared usage across concurrent agent tasks.
pub trait LlmProvider: Send + Sync {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, crate::error::Error>> + Send;
}
