//! X (Twitter) tool family for the heartbit-ghost persona.
//!
//! Each tool implements `heartbit_core::Tool` and resolves credentials at
//! execute-time via `ExecutionContext::credentials`. Tools share `XClient`
//! for HTTP, OAuth1 signing, and error mapping.

pub mod client;

pub use client::{XApiError, XClient, format_error};
