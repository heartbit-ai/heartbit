//! Umbrella-side memory implementations.
//!
//! Trait + InMemory + NamespacedMemory + memory tools live in
//! [`heartbit_core::memory`]; the platform-specific impls below stay here
//! behind feature flags so they don't pollute `heartbit-core`'s dep graph.

pub use heartbit_core::memory::*;

#[cfg(feature = "postgres")]
pub mod postgres;
#[cfg(feature = "postgres")]
pub use postgres::PostgresMemoryStore;

#[cfg(feature = "local-embedding")]
pub mod embedding_local;
#[cfg(feature = "local-embedding")]
pub use embedding_local::LocalEmbeddingProvider;
