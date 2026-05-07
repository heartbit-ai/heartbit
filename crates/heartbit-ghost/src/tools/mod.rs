//! X (Twitter) tool family for the heartbit-ghost persona.
//!
//! Each tool implements `heartbit_core::Tool` and resolves credentials at
//! execute-time via `ExecutionContext::credentials`. Tools share `XClient`
//! for HTTP, OAuth1 signing, and error mapping.

pub mod client;
pub mod mentions;
pub mod reply;
pub mod search;
pub mod thread;
pub mod user;

pub use client::{XApiError, XClient, format_error};
pub use mentions::TwitterMentionsTool;
pub use reply::TwitterReplyTool;
pub use search::TwitterSearchTool;
pub use thread::TwitterThreadTool;
pub use user::TwitterUserTool;
