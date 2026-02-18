pub mod anthropic;
pub mod openrouter;
pub mod types;

use crate::llm::types::{CompletionRequest, CompletionResponse};

/// Trait for LLM providers.
///
/// Uses RPITIT (`impl Future`) which means this trait is NOT dyn-compatible.
/// All consumers are generic over `P: LlmProvider`. This is intentional:
/// one provider per process, no need for trait objects.
///
/// If dyn dispatch is needed later, wrap with `Pin<Box<dyn Future>>`.
pub trait LlmProvider: Send + Sync {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, crate::error::Error>> + Send;
}
