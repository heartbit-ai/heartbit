//! Re-exports `heartbit_core::sandbox::*` for back-compat.
//!
//! The `SandboxPolicy` type moved into `heartbit-core` in B4 so the
//! filesystem builtins (which live in core) can compose it with
//! `CorePathPolicy`. Callers that imported `heartbit::sandbox::SandboxPolicy`
//! continue to work via this re-export.

pub use heartbit_core::sandbox::*;
