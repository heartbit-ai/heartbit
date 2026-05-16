//! Static-site renderer. Implemented in Task 8.
#![allow(dead_code)]

/// Metadata for a rendered blog post. Populated in Task 8.
#[derive(Debug, Clone)]
pub struct RenderedPostMeta {
    /// Placeholder field — replaced in Task 8.
    pub _todo: (),
}

/// Errors returned by [`render_site`].
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// Implementation not yet available (Task 8 stub).
    #[error("not implemented yet")]
    NotImplemented,
}

/// Render all Markdown posts to static HTML. Implemented in Task 8.
pub fn render_site() -> Result<Vec<RenderedPostMeta>, RenderError> {
    Err(RenderError::NotImplemented)
}
