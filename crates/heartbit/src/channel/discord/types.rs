//! Discord API types.

use serde::{Deserialize, Serialize};

/// Discord Gateway opcodes.
pub mod opcode {
    pub const DISPATCH: u8 = 0;
    pub const HEARTBEAT: u8 = 1;
    pub const IDENTIFY: u8 = 2;
    pub const RESUME: u8 = 6;
    pub const RECONNECT: u8 = 7;
    pub const INVALID_SESSION: u8 = 9;
    pub const HELLO: u8 = 10;
    pub const HEARTBEAT_ACK: u8 = 11;
}

/// A Gateway payload.
#[derive(Debug, Deserialize, Serialize)]
pub struct GatewayPayload {
    pub op: u8,
    #[serde(default)]
    pub d: serde_json::Value,
    #[serde(default)]
    pub s: Option<u64>,
    #[serde(default)]
    pub t: Option<String>,
}

/// Discord message object (subset).
#[derive(Debug, Deserialize)]
pub struct DiscordMessage {
    pub id: String,
    pub channel_id: String,
    pub content: String,
    #[serde(default)]
    pub author: DiscordUser,
    #[serde(default)]
    pub guild_id: Option<String>,
    #[serde(default)]
    pub mentions: Vec<DiscordUser>,
}

/// Discord user object (subset).
#[derive(Debug, Default, Deserialize)]
pub struct DiscordUser {
    pub id: String,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub bot: bool,
}

/// Discord READY event data (subset).
#[derive(Debug, Deserialize)]
pub struct ReadyEvent {
    pub user: DiscordUser,
    pub session_id: String,
    pub resume_gateway_url: String,
}

/// Hello event data.
#[derive(Debug, Deserialize)]
pub struct HelloData {
    pub heartbeat_interval: u64,
}

/// Gateway URL response.
#[derive(Debug, Deserialize)]
pub struct GatewayUrlResponse {
    pub url: String,
}
