pub mod anthropic;
pub mod openrouter;
pub mod retry;
pub mod types;

use crate::llm::types::{CompletionRequest, CompletionResponse};

/// Callback invoked with each text delta during streaming.
pub type OnText = dyn Fn(&str) + Send + Sync;

/// Callback invoked before tool execution for human-in-the-loop approval.
///
/// Receives the list of tool calls the LLM wants to execute.
/// Returns `true` to proceed with execution, `false` to deny.
/// When denied, the agent sends an error result back to the LLM.
pub type OnApproval = dyn Fn(&[crate::llm::types::ToolCall]) -> bool + Send + Sync;

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

    /// Stream a completion, calling `on_text` for each text delta as it arrives.
    ///
    /// The returned `CompletionResponse` contains the full accumulated response
    /// (same as `complete()`), but text was emitted incrementally via the callback.
    ///
    /// Default: falls back to `complete()` (no incremental streaming).
    fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &OnText,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, crate::error::Error>> + Send
    {
        let _ = on_text;
        self.complete(request)
    }
}
