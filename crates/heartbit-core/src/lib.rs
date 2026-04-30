//! # heartbit-core
//!
//! The Rust agentic framework — agents, tools, LLM providers, memory, evaluation.
//!
//! Documentation lands here as the crate's docs.rs preamble. The README
//! is rendered above this on docs.rs.

#![allow(unexpected_cfgs)]

// Modules are added one at a time as subsequent tasks move them in.
pub mod agent;
pub mod auth;
pub mod error;
pub mod http;
pub mod knowledge;
pub mod llm;
pub mod memory;
pub mod signal;
pub mod store;
pub mod tool;
pub mod types;
pub mod util;
pub mod workspace;
