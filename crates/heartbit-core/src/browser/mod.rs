//! SOTA browser-bot harness: drive a real Chrome via the `chrome-devtools` MCP
//! server, with the reliability and safety layer that makes a web agent actually
//! work (per `tasks/browser-bot-sota-2026-05-31.md`).
//!
//! The browser-control problem ("how to call Chrome") is already solved: the MCP
//! stdio client ([`crate::tool::mcp`]) turns the server's tools into
//! `Arc<dyn Tool>`, and the bundled `chrome-devtools` preset
//! ([`crate::connect_preset`]) spawns it with one call. This module adds the
//! *agent-layer* capabilities the research found decisive — starting with
//! observation distillation ([`distill`]) — built on the existing agent runner,
//! dynamic-workflow combinators, structured output, guardrails, and memory.
//!
//! Built incrementally (spec §9, cheap reliability wins first):
//! - **B1 (this):** [`distill`] — prune the a11y snapshot before it reaches the LLM.
//! - B2+: stale-uid retry, post-action verification, settle, replan-loop,
//!   completion judge, browser guardrails. (See the spec.)

pub mod distill;
pub mod harness;

pub use distill::{DistillConfig, distill_snapshot, interactive_uids};
pub use harness::ReliableInteractionTool;
