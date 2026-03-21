//! Discord channel adapter.
//!
//! Connects to Discord Gateway v10 via WebSocket and forwards messages
//! to heartbit agents via the `RunTask` callback.

pub mod types;

use std::sync::Arc;

use serde_json::json;
use tokio::sync::Mutex;
use tracing::warn;

use crate::agent::events::OnEvent;
use crate::channel::ChannelBridge;
use crate::error::Error;
use crate::llm::{ApprovalDecision, OnApproval, OnText};
use crate::tool::builtins::OnQuestion;

use types::*;

/// Maximum message length for Discord (will be chunked if exceeded).
const MAX_MESSAGE_LEN: usize = 2000;

/// Discord Gateway API version.
const GATEWAY_VERSION: u8 = 10;

/// Discord bot intents for receiving messages.
const GUILD_MESSAGES: u64 = 1 << 9;
const DIRECT_MESSAGES: u64 = 1 << 12;
const MESSAGE_CONTENT: u64 = 1 << 15;

/// Configuration for the Discord adapter.
#[derive(Debug, Clone)]
pub struct DiscordConfig {
    /// Bot token (from Discord Developer Portal).
    pub token: String,
    /// Gateway intents. Default: GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT.
    pub intents: u64,
    /// Whether to ignore messages from other bots. Default: true.
    pub ignore_bots: bool,
    /// Whether the bot only responds when mentioned. Default: true (in guilds).
    pub require_mention: bool,
}

impl DiscordConfig {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            intents: GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT,
            ignore_bots: true,
            require_mention: true,
        }
    }
}

/// Bridge that translates Discord messages into heartbit callbacks.
pub struct DiscordBridge {
    channel_id: String,
    token: String,
    client: reqwest::Client,
}

impl DiscordBridge {
    pub fn new(channel_id: String, token: String) -> Self {
        Self {
            channel_id,
            token,
            client: reqwest::Client::new(),
        }
    }

    /// Send a message to the Discord channel (chunked if needed).
    async fn send_message(&self, text: &str) -> Result<(), Error> {
        for chunk in chunk_message(text, MAX_MESSAGE_LEN) {
            let url = format!(
                "https://discord.com/api/v10/channels/{}/messages",
                self.channel_id
            );
            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bot {}", self.token))
                .header("Content-Type", "application/json")
                .json(&json!({"content": chunk}))
                .send()
                .await
                .map_err(|e| Error::Channel(format!("Discord send failed: {e}")))?;

            if !response.status().is_success() {
                let body = response.text().await.unwrap_or_default();
                warn!(body = %body, "Discord message send failed");
            }
        }
        Ok(())
    }
}

impl ChannelBridge for DiscordBridge {
    fn make_on_text(self: Arc<Self>) -> Arc<OnText> {
        let bridge = self;
        let buffer = Arc::new(Mutex::new(String::new()));
        Arc::new(move |text: &str| {
            let bridge = Arc::clone(&bridge);
            let buffer = Arc::clone(&buffer);
            let text = text.to_string();
            tokio::spawn(async move {
                let mut buf = buffer.lock().await;
                buf.push_str(&text);
                // Flush on sentence boundaries or when buffer exceeds threshold.
                // Note: the final fragment may remain buffered if it doesn't hit
                // a boundary. The agent's ChatFinal message delivers the complete
                // result independently, so this streaming preview is best-effort.
                if buf.len() >= 200 || text.contains('\n') {
                    let msg = std::mem::take(&mut *buf);
                    drop(buf);
                    if let Err(e) = bridge.send_message(&msg).await {
                        warn!(error = %e, "failed to send Discord text");
                    }
                }
            });
        })
    }

    fn make_on_event(self: Arc<Self>) -> Arc<OnEvent> {
        Arc::new(|_event| {
            // Events not forwarded to Discord (too verbose)
        })
    }

    fn make_on_approval(self: Arc<Self>) -> Arc<OnApproval> {
        // Auto-approve in Discord (no interactive approval UI)
        Arc::new(|_tool_calls| ApprovalDecision::Allow)
    }

    fn make_on_question(self: Arc<Self>) -> Arc<OnQuestion> {
        Arc::new(|_request| {
            Box::pin(async {
                Err(Error::Channel(
                    "interactive questions not supported in Discord".into(),
                ))
            })
        })
    }
}

// Re-export the shared `chunk_message` from the parent channel module.
pub use super::chunk_message;

/// Fetch the Gateway WebSocket URL from Discord's REST API.
pub async fn get_gateway_url(token: &str) -> Result<String, Error> {
    let client = reqwest::Client::new();
    let response = client
        .get("https://discord.com/api/v10/gateway/bot")
        .header("Authorization", format!("Bot {token}"))
        .send()
        .await
        .map_err(|e| Error::Channel(format!("failed to get Discord gateway URL: {e}")))?;

    if !response.status().is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(Error::Channel(format!("Discord gateway API error: {body}")));
    }

    let data: GatewayUrlResponse = response
        .json()
        .await
        .map_err(|e| Error::Channel(format!("failed to parse gateway response: {e}")))?;

    Ok(format!("{}/?v={GATEWAY_VERSION}&encoding=json", data.url))
}

/// Send a typing indicator to a channel.
pub async fn send_typing(token: &str, channel_id: &str) -> Result<(), Error> {
    let client = reqwest::Client::new();
    let url = format!("https://discord.com/api/v10/channels/{channel_id}/typing");
    client
        .post(&url)
        .header("Authorization", format!("Bot {token}"))
        .send()
        .await
        .map_err(|e| Error::Channel(format!("Discord typing indicator failed: {e}")))?;
    Ok(())
}

/// Strip bot mention from message content.
pub fn strip_mention(content: &str, bot_id: &str) -> String {
    content
        .replace(&format!("<@{bot_id}>"), "")
        .replace(&format!("<@!{bot_id}>"), "")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_short_message() {
        let chunks = chunk_message("hello", 2000);
        assert_eq!(chunks, vec!["hello"]);
    }

    #[test]
    fn chunk_long_message() {
        let long = "a".repeat(3000);
        let chunks = chunk_message(&long, 2000);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.len() <= 2000);
        }
    }

    #[test]
    fn chunk_splits_at_newline() {
        let text = format!("{}\n{}", "a".repeat(1500), "b".repeat(1000));
        let chunks = chunk_message(&text, 2000);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 1500);
    }

    #[test]
    fn chunk_empty_message() {
        let chunks = chunk_message("", 2000);
        assert_eq!(chunks, vec![""]);
    }

    #[test]
    fn strip_mention_removes_bot_id() {
        let content = "<@1234567890> hello there";
        assert_eq!(strip_mention(content, "1234567890"), "hello there");
    }

    #[test]
    fn strip_mention_removes_nickname_mention() {
        let content = "<@!1234567890> hello";
        assert_eq!(strip_mention(content, "1234567890"), "hello");
    }

    #[test]
    fn strip_mention_no_mention() {
        let content = "hello world";
        assert_eq!(strip_mention(content, "123"), "hello world");
    }

    #[test]
    fn discord_config_defaults() {
        let config = DiscordConfig::new("test-token");
        assert!(config.ignore_bots);
        assert!(config.require_mention);
        assert_eq!(
            config.intents,
            GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT
        );
    }

    #[test]
    fn gateway_payload_deserialize() {
        let json = r#"{"op":10,"d":{"heartbeat_interval":41250}}"#;
        let payload: GatewayPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.op, opcode::HELLO);
    }

    #[test]
    fn discord_message_deserialize() {
        let json = r#"{"id":"1","channel_id":"2","content":"hello","author":{"id":"3","username":"user","bot":false}}"#;
        let msg: DiscordMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, "hello");
        assert_eq!(msg.author.id, "3");
        assert!(!msg.author.bot);
    }

    #[test]
    fn hello_data_deserialize() {
        let json = r#"{"heartbeat_interval":41250}"#;
        let hello: HelloData = serde_json::from_str(json).unwrap();
        assert_eq!(hello.heartbeat_interval, 41250);
    }

    #[test]
    fn ready_event_deserialize() {
        let json = r#"{"user":{"id":"bot1","username":"Bot","bot":true},"session_id":"sess1","resume_gateway_url":"wss://gateway.discord.gg"}"#;
        let ready: ReadyEvent = serde_json::from_str(json).unwrap();
        assert_eq!(ready.user.id, "bot1");
        assert!(ready.user.bot);
        assert_eq!(ready.session_id, "sess1");
    }

    #[test]
    fn discord_bridge_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DiscordBridge>();
    }

    #[test]
    fn channel_bridge_impl_exists() {
        // Compile-time check that DiscordBridge implements ChannelBridge
        fn assert_bridge<T: ChannelBridge>() {}
        assert_bridge::<DiscordBridge>();
    }
}
