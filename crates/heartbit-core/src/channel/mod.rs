//! Channel base traits and in-process implementations.
//!
//! Platform-specific adapters (Telegram, Discord, Slack) and the
//! Postgres-backed session store live in the heartbit umbrella crate.

pub mod bridge;
pub mod session;
pub mod types;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::events::OnEvent;
use crate::error::Error;
use crate::llm::{OnApproval, OnText};
use crate::memory::Memory;
use crate::tool::builtins::OnQuestion;

/// A media attachment from a messaging channel (photo, voice, document).
pub struct MediaAttachment {
    /// MIME type or platform-specific media kind (`image/jpeg`, `voice`, etc.).
    pub media_type: String,
    /// Raw attachment bytes.
    pub data: Vec<u8>,
    /// Optional caption supplied with the attachment.
    pub caption: Option<String>,
}

/// Trait for channel-specific bridges that produce agent callbacks.
///
/// Each messaging channel (Telegram, Discord, etc.) implements this trait
/// so the same `RunTask` closure can drive any channel without duplication.
pub trait ChannelBridge: Send + Sync {
    /// Produce the `OnText` callback that forwards streaming text to the client.
    fn make_on_text(self: Arc<Self>) -> Arc<OnText>;
    /// Produce the `OnEvent` callback that forwards `AgentEvent` records to the client.
    fn make_on_event(self: Arc<Self>) -> Arc<OnEvent>;
    /// Produce the `OnApproval` callback for human-in-the-loop tool gating.
    fn make_on_approval(self: Arc<Self>) -> Arc<OnApproval>;
    /// Produce the `OnQuestion` callback for structured agent-to-user questions.
    fn make_on_question(self: Arc<Self>) -> Arc<OnQuestion>;
}

/// Input for the `RunTask` callback.
pub struct RunTaskInput {
    /// User-typed task text.
    pub task_text: String,
    /// Channel bridge providing the callback set for this run.
    pub bridge: Arc<dyn ChannelBridge>,
    /// Pre-existing shared memory store so sub-agent memory tools persist
    /// across tasks. Passed as the raw (un-namespaced) store.
    pub memory: Option<Arc<dyn Memory>>,
    /// User-specific namespace prefix (e.g. `"tg:12345"`). Passed as `story_id`
    /// to `build_orchestrator_from_config` for per-user memory isolation.
    pub user_namespace: Option<String>,
    /// Media attachments (photos, documents). Empty for text-only messages.
    pub attachments: Vec<MediaAttachment>,
}

/// Callback type for running an agent task with bridge callbacks.
///
/// The CLI crate provides this closure to wire `build_orchestrator_from_config`
/// with the channel bridge callbacks. Returns the agent's final text output.
pub type RunTask = dyn Fn(RunTaskInput) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send>>
    + Send
    + Sync;

/// Callback type for memory consolidation on idle sessions.
pub type ConsolidateSession =
    dyn Fn(i64) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send>> + Send + Sync;

/// Split a message into chunks that fit a platform's message-length limit.
///
/// Tries to split at newlines for readability; falls back to char boundaries.
/// Shared by Discord, Slack, and other channel adapters.
pub fn chunk_message(text: &str, max_len: usize) -> Vec<&str> {
    if text.len() <= max_len {
        return vec![text];
    }
    let mut chunks = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        if remaining.len() <= max_len {
            chunks.push(remaining);
            break;
        }
        // Try to split at a newline
        let split_at = remaining[..max_len].rfind('\n').unwrap_or_else(|| {
            // Fall back to char boundary
            let mut pos = max_len;
            while pos > 0 && !remaining.is_char_boundary(pos) {
                pos -= 1;
            }
            pos
        });
        let split_at = if split_at == 0 {
            max_len.min(remaining.len())
        } else {
            split_at
        };
        chunks.push(&remaining[..split_at]);
        remaining = &remaining[split_at..];
        // Skip leading newline after split
        if remaining.starts_with('\n') {
            remaining = &remaining[1..];
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_bridge_is_object_safe() {
        // Compile-time check: ChannelBridge can be used as a trait object.
        fn _assert(_: &Arc<dyn ChannelBridge>) {}
    }

    #[test]
    fn run_task_input_accepts_dyn_bridge() {
        // Compile-time check: RunTaskInput.bridge is Arc<dyn ChannelBridge>.
        fn _assert(bridge: Arc<dyn ChannelBridge>) {
            let _input = RunTaskInput {
                task_text: String::new(),
                bridge,
                memory: None,
                user_namespace: None,
                attachments: Vec::new(),
            };
        }
    }
}
