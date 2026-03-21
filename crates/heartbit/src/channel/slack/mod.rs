//! Slack channel adapter.
//!
//! Uses Socket Mode (WebSocket) for receiving events and Web API for sending.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::warn;

use crate::agent::events::OnEvent;
use crate::channel::ChannelBridge;
use crate::error::Error;
use crate::llm::{ApprovalDecision, OnApproval, OnText};
use crate::tool::builtins::OnQuestion;

/// Maximum message length for Slack.
const MAX_MESSAGE_LEN: usize = 3000;

/// Configuration for the Slack adapter.
#[derive(Debug, Clone)]
pub struct SlackConfig {
    /// App-level token (xapp-...) for Socket Mode connections.
    pub app_token: String,
    /// Bot token (xoxb-...) for Web API calls.
    pub bot_token: String,
    /// Whether to auto-reply in threads. Default: true.
    pub auto_thread_reply: bool,
}

impl SlackConfig {
    pub fn new(app_token: impl Into<String>, bot_token: impl Into<String>) -> Self {
        Self {
            app_token: app_token.into(),
            bot_token: bot_token.into(),
            auto_thread_reply: true,
        }
    }
}

/// Bridge that translates Slack events into heartbit callbacks.
pub struct SlackBridge {
    channel_id: String,
    thread_ts: Option<String>,
    bot_token: String,
    client: reqwest::Client,
}

impl SlackBridge {
    pub fn new(channel_id: String, thread_ts: Option<String>, bot_token: String) -> Self {
        Self {
            channel_id,
            thread_ts,
            bot_token,
            client: reqwest::Client::new(),
        }
    }

    /// Send a message to Slack via Web API.
    async fn send_message(&self, text: &str) -> Result<(), Error> {
        for chunk in chunk_message(text, MAX_MESSAGE_LEN) {
            let mut body = serde_json::json!({
                "channel": self.channel_id,
                "text": chunk,
            });
            if let Some(ref ts) = self.thread_ts {
                body["thread_ts"] = serde_json::json!(ts);
            }

            let response = self
                .client
                .post("https://slack.com/api/chat.postMessage")
                .header("Authorization", format!("Bearer {}", self.bot_token))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| Error::Channel(format!("Slack send failed: {e}")))?;

            let resp_body: serde_json::Value = response
                .json()
                .await
                .map_err(|e| Error::Channel(format!("Slack response parse failed: {e}")))?;

            if resp_body.get("ok") != Some(&serde_json::Value::Bool(true)) {
                let error = resp_body
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                warn!(error = %error, "Slack chat.postMessage failed");
            }
        }
        Ok(())
    }
}

impl ChannelBridge for SlackBridge {
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
                        warn!(error = %e, "failed to send Slack text");
                    }
                }
            });
        })
    }

    fn make_on_event(self: Arc<Self>) -> Arc<OnEvent> {
        Arc::new(|_event| {})
    }

    fn make_on_approval(self: Arc<Self>) -> Arc<OnApproval> {
        Arc::new(|_tool_calls| ApprovalDecision::Allow)
    }

    fn make_on_question(self: Arc<Self>) -> Arc<OnQuestion> {
        Arc::new(|_request| {
            Box::pin(async {
                Err(Error::Channel(
                    "interactive questions not supported in Slack".into(),
                ))
            })
        })
    }
}

/// Slack Socket Mode connection URL response.
#[derive(Debug, Deserialize)]
pub struct SocketModeResponse {
    pub ok: bool,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Slack event envelope (Socket Mode).
#[derive(Debug, Deserialize)]
pub struct SlackEnvelope {
    #[serde(rename = "type")]
    pub envelope_type: String,
    #[serde(default)]
    pub envelope_id: Option<String>,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}

/// Slack event callback payload.
#[derive(Debug, Deserialize)]
pub struct SlackEventCallback {
    #[serde(default)]
    pub event: Option<SlackEvent>,
}

/// A Slack event (subset).
#[derive(Debug, Deserialize)]
pub struct SlackEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub thread_ts: Option<String>,
    #[serde(default)]
    pub ts: Option<String>,
    #[serde(default)]
    pub bot_id: Option<String>,
}

/// Acknowledge envelope for Socket Mode.
#[derive(Debug, Serialize)]
pub struct SocketModeAck {
    pub envelope_id: String,
}

/// Get a Socket Mode WebSocket URL.
pub async fn get_socket_url(app_token: &str) -> Result<String, Error> {
    let client = reqwest::Client::new();
    let response = client
        .post("https://slack.com/api/apps.connections.open")
        .header("Authorization", format!("Bearer {app_token}"))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .send()
        .await
        .map_err(|e| Error::Channel(format!("Slack Socket Mode connect failed: {e}")))?;

    let data: SocketModeResponse = response
        .json()
        .await
        .map_err(|e| Error::Channel(format!("failed to parse Slack response: {e}")))?;

    if !data.ok {
        return Err(Error::Channel(format!(
            "Slack Socket Mode error: {}",
            data.error.unwrap_or_else(|| "unknown".into())
        )));
    }

    data.url
        .ok_or_else(|| Error::Channel("Slack Socket Mode response missing URL".into()))
}

/// Validate a bot token by calling auth.test.
pub async fn validate_bot_token(bot_token: &str) -> Result<String, Error> {
    let client = reqwest::Client::new();
    let response = client
        .post("https://slack.com/api/auth.test")
        .header("Authorization", format!("Bearer {bot_token}"))
        .send()
        .await
        .map_err(|e| Error::Channel(format!("Slack auth.test failed: {e}")))?;

    #[derive(Deserialize)]
    struct AuthTestResponse {
        ok: bool,
        #[serde(default)]
        user_id: Option<String>,
        #[serde(default)]
        error: Option<String>,
    }

    let data: AuthTestResponse = response
        .json()
        .await
        .map_err(|e| Error::Channel(format!("failed to parse auth.test response: {e}")))?;

    if !data.ok {
        return Err(Error::Channel(format!(
            "Slack auth.test error: {}",
            data.error.unwrap_or_else(|| "unknown".into())
        )));
    }

    data.user_id
        .ok_or_else(|| Error::Channel("Slack auth.test missing user_id".into()))
}

// Re-export the shared `chunk_message` from the parent channel module.
pub use super::chunk_message;

/// Strip bot mention from Slack message text.
pub fn strip_mention(text: &str, bot_user_id: &str) -> String {
    text.replace(&format!("<@{bot_user_id}>"), "")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_short_message() {
        let chunks = chunk_message("hello", 3000);
        assert_eq!(chunks, vec!["hello"]);
    }

    #[test]
    fn chunk_long_message() {
        let long = "a".repeat(5000);
        let chunks = chunk_message(&long, 3000);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.len() <= 3000);
        }
    }

    #[test]
    fn chunk_splits_at_newline() {
        let text = format!("{}\n{}", "a".repeat(2000), "b".repeat(2000));
        let chunks = chunk_message(&text, 3000);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn strip_mention_removes_user_id() {
        assert_eq!(strip_mention("<@U123> hello", "U123"), "hello");
    }

    #[test]
    fn strip_mention_no_mention() {
        assert_eq!(strip_mention("hello world", "U123"), "hello world");
    }

    #[test]
    fn slack_config_defaults() {
        let config = SlackConfig::new("xapp-token", "xoxb-token");
        assert!(config.auto_thread_reply);
    }

    #[test]
    fn slack_bridge_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SlackBridge>();
    }

    #[test]
    fn channel_bridge_impl_exists() {
        fn assert_bridge<T: ChannelBridge>() {}
        assert_bridge::<SlackBridge>();
    }

    #[test]
    fn socket_mode_response_deserialize() {
        let json = r#"{"ok":true,"url":"wss://wss-primary.slack.com/link/"}"#;
        let resp: SocketModeResponse = serde_json::from_str(json).unwrap();
        assert!(resp.ok);
        assert_eq!(resp.url.unwrap(), "wss://wss-primary.slack.com/link/");
    }

    #[test]
    fn slack_envelope_deserialize() {
        let json = r#"{"type":"events_api","envelope_id":"env1","payload":{}}"#;
        let env: SlackEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(env.envelope_type, "events_api");
        assert_eq!(env.envelope_id.unwrap(), "env1");
    }

    #[test]
    fn slack_event_deserialize() {
        let json = r#"{"type":"message","text":"hello","channel":"C123","user":"U456"}"#;
        let event: SlackEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "message");
        assert_eq!(event.text.unwrap(), "hello");
        assert_eq!(event.channel.unwrap(), "C123");
    }

    #[test]
    fn socket_mode_ack_serializes() {
        let ack = SocketModeAck {
            envelope_id: "env1".into(),
        };
        let json = serde_json::to_string(&ack).unwrap();
        assert!(json.contains("env1"));
    }
}
