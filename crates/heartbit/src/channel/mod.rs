pub mod bridge;
pub mod session;
#[cfg(feature = "telegram")]
pub mod telegram;
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
    pub media_type: String,
    pub data: Vec<u8>,
    pub caption: Option<String>,
}

/// Trait for channel-specific bridges that produce agent callbacks.
///
/// Each messaging channel (Telegram, Discord, etc.) implements this trait
/// so the same `RunTask` closure can drive any channel without duplication.
pub trait ChannelBridge: Send + Sync {
    fn make_on_text(self: Arc<Self>) -> Arc<OnText>;
    fn make_on_event(self: Arc<Self>) -> Arc<OnEvent>;
    fn make_on_approval(self: Arc<Self>) -> Arc<OnApproval>;
    fn make_on_question(self: Arc<Self>) -> Arc<OnQuestion>;
}

/// Input for the `RunTask` callback.
pub struct RunTaskInput {
    pub task_text: String,
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
