//! `PostHistoryStore` — placeholder. Trait + impls land in Task 6.

/// Errors returned by post history store operations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// I/O failure reading or writing the history file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSONL parse failure on replay.
    #[error("parse: {0}")]
    Parse(String),
}

/// Persistent storage for proactive post outcomes. Stub — full trait + impls land in Task 6.
///
/// Defining the marker trait here so types in Tasks 3-5 can reference `dyn PostHistoryStore`.
pub trait PostHistoryStore: Send + Sync {}

/// Stub — full impl in Task 6.
#[allow(dead_code)]
pub struct InMemoryPostHistoryStore;

/// Stub — full impl in Task 6.
#[allow(dead_code)]
pub struct JsonlPostHistoryStore;
