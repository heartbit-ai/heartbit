//! Code-generation harness — a domain-specific agent harness for writing and
//! **verifying** code, mirroring [`browser`](crate::browser) but for the code
//! domain.
//!
//! The browser harness needed an LLM judge because page state is not reducible
//! to a boolean. Code is different: a build/test command's **exit code is the
//! ground truth**. So this harness centers on one primitive the rest of the
//! framework lacked — a structured, *configured*, gate-able verification tool —
//! and otherwise composes existing machinery (`AgentRunner`, `GoalCondition`'s
//! independent-judge repair loop, builtins rooted at a workspace, guardrails).
//!
//! - [`VerifyCommandTool`] runs the harness-configured verify command(s) and
//!   emits a machine-greppable `VERIFY_RESULT:` sentinel (PASS/FAIL + exit code).
//!   Its two reasons to exist over raw `bash`: the command is *configured* (the
//!   agent never guesses it) and the result is *structured and gate-able*.
//! - [`parse_latest_verify`] reads that sentinel back out of a transcript — the
//!   seam for a deterministic completion gate.
//!
//! See `tasks/codegen-harness-2026-06-02.md` for the full design + non-duplication
//! rationale (what we deliberately do NOT build from `claw-code`).

pub mod builder;
pub mod verify;

#[cfg(test)]
mod live;

pub use builder::{CODE_SYSTEM_PROMPT, CodeAgentBuilder, code_goal, code_tools};
pub use verify::{VerifyCommandTool, VerifyOutcome, parse_latest_verify, render_verify_result};
