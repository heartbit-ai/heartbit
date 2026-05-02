//! Umbrella-side channel module.
//!
//! Re-exports the base traits and shared types from `heartbit_core::channel`
//! and adds the Postgres-backed [`SessionStore`] implementation plus the
//! platform-specific adapters (Discord, Slack) behind feature
//! flags.
//!
//! [`SessionStore`]: heartbit_core::channel::session::SessionStore

pub use heartbit_core::channel::*;

#[cfg(feature = "postgres")]
mod session_postgres;
#[cfg(feature = "postgres")]
pub use session_postgres::PostgresSessionStore;

#[cfg(feature = "discord")]
pub mod discord;
#[cfg(feature = "slack")]
pub mod slack;
