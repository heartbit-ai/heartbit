//! Trait surface for persona-specific topic-context pre-fetching.
//! Concrete impls live in persona crates (e.g., heartbit-ghost).
//! See heartbit-ghost P1.6 spec §5.

use std::future::Future;
use std::pin::Pin;

/// Builds a plain-text context block consumed by the proactive-post
/// topic generator. Implementation strategies vary by persona —
/// `heartbit-ghost:x` fetches own tweets + mentions; `heartbit-rs:x`
/// inspects the local repo.
///
/// Uses primitive types in the signature so heartbit-core stays free
/// of persona-crate value types. The `recent_history_json` argument
/// is a JSON-encoded `Vec<PostHistoryEntry>` (the persona crate
/// decodes it internally).
pub trait TopicContextProvider: Send + Sync {
    /// Returns a multi-line plain-text block. Empty string is allowed.
    fn build_context<'a>(
        &'a self,
        operator_user_id: &'a str,
        recent_history_json: &'a str,
        credentials: std::sync::Arc<dyn crate::execution_context::CredentialResolver>,
    ) -> Pin<Box<dyn Future<Output = Result<String, anyhow::Error>> + Send + 'a>>;
}
