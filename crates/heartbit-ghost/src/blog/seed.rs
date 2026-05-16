//! X-derived blog topic seed selection. Implemented in Task 6.
#![allow(dead_code)]

/// Topic seed derived from top-engaged X posts. Populated in Task 6.
#[derive(Debug, Clone)]
pub struct BlogSeed {
    /// Placeholder field — replaced in Task 6.
    pub _todo: (),
}

/// Errors returned by [`select_blog_seed`].
#[derive(Debug, thiserror::Error)]
pub enum SeedError {
    /// Implementation not yet available (Task 6 stub).
    #[error("not implemented yet")]
    NotImplemented,
}

/// Select a blog topic seed from X engagement history. Implemented in Task 6.
pub fn select_blog_seed() -> Result<BlogSeed, SeedError> {
    Err(SeedError::NotImplemented)
}
