//! Umbrella-side store module.
//!
//! Task records and audit entries live in [`heartbit_core::store`]. The
//! PostgreSQL-backed implementations below stay in the umbrella behind the
//! `postgres` feature so they don't pollute `heartbit-core`'s dep graph.

pub use heartbit_core::store::*;

#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "postgres")]
pub use postgres::{PostgresStore, PostgresAuditTrail};
