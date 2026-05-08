//! # heartbit-telegram
//!
//! Telegram bot adapter for the Heartbit multi-agent runtime.
//!
//! Provides a Telegram DM interface with streaming text, inline keyboard
//! approvals/questions, session management, and memory recall.

mod access;
mod adapter;
mod bridge;
mod config;
mod delivery;
mod extract;
mod keyboard;
mod router;

pub use access::AccessControl;
pub use adapter::TelegramAdapter;
pub use bridge::TelegramBridge;
pub use config::{DmPolicy, TelegramConfig};
pub use delivery::{RateLimiter, StreamBuffer, chunk_message, markdown_to_telegram_html};
pub use keyboard::{
    CallbackAction, PickChoice, approval_buttons, parse_callback_data, persona_pick_buttons,
    question_buttons,
};
pub use router::ChatSessionMap;
