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
/// Tries to split at newlines for readability; falls back to the largest
/// UTF-8 char boundary that fits. When `max_len` is smaller than a single
/// character, that character is emitted whole (a codepoint cannot be split).
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
        // Largest char-boundary window that fits the limit — computed BEFORE
        // any slicing, so a multibyte character at `max_len` cannot panic.
        let window_end = crate::tool::builtins::floor_char_boundary(remaining, max_len);
        // Prefer splitting at the last newline inside the window.
        let split_at = match remaining[..window_end].rfind('\n') {
            Some(nl) if nl > 0 => nl,
            // No usable newline: take the whole window.
            _ if window_end > 0 => window_end,
            // Degenerate: max_len is smaller than the first character. Emit
            // it whole so the loop always advances.
            _ => remaining
                .chars()
                .next()
                .map(char::len_utf8)
                .unwrap_or(remaining.len()),
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

    #[test]
    fn chunk_message_short_text_passthrough() {
        assert_eq!(chunk_message("hello", 10), vec!["hello"]);
    }

    #[test]
    fn chunk_message_prefers_newline_split() {
        let chunks = chunk_message("aaa\nbbb", 5);
        assert_eq!(chunks, vec!["aaa", "bbb"]);
    }

    // Panic-safety: byte index max_len falling INSIDE a multibyte character
    // must not panic — the old code sliced `remaining[..max_len]` before the
    // char-boundary fallback ever ran. An LLM reply in French/emoji over the
    // platform limit is the live trigger (Discord/Slack bridges).
    #[test]
    fn chunk_message_multibyte_at_boundary_does_not_panic() {
        // "é" is 2 bytes; an odd max_len lands mid-codepoint.
        let text = "é".repeat(60); // 120 bytes
        let chunks = chunk_message(&text, 5);
        // No newlines in the input, so the chunks must reassemble losslessly.
        assert_eq!(chunks.concat(), text);
        for c in &chunks {
            assert!(c.len() <= 5, "chunk over limit: {} bytes", c.len());
        }
    }

    #[test]
    fn chunk_message_emoji_at_boundary_does_not_panic() {
        // 4-byte emoji; max_len = 10 lands mid-codepoint (10 % 4 != 0).
        let text = "🦀".repeat(30); // 120 bytes
        let chunks = chunk_message(&text, 10);
        assert_eq!(chunks.concat(), text);
        for c in &chunks {
            assert!(c.len() <= 10, "chunk over limit: {} bytes", c.len());
        }
    }

    // The `split_at == 0` fallback used to re-assign a raw `max_len` index,
    // reintroducing the non-boundary slice even after a safe one was computed:
    // a window whose ONLY newline is at position 0 takes that path.
    #[test]
    fn chunk_message_leading_newline_window_does_not_panic() {
        let mut text = String::from("\n");
        text.push_str(&"é".repeat(30)); // newline + 60 bytes of 2-byte chars
        let chunks = chunk_message(&text, 5);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.len() <= 5, "chunk over limit: {} bytes", c.len());
        }
    }

    // Degenerate limit: max_len smaller than the first character's width. A
    // codepoint cannot be split, so the chunk carries the whole char (the only
    // non-panicking option) and the loop must still terminate.
    #[test]
    fn chunk_message_max_len_smaller_than_char_still_terminates() {
        let chunks = chunk_message("日本語", 1);
        assert_eq!(chunks, vec!["日", "本", "語"]);
    }
}
