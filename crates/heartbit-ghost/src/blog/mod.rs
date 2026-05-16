//! Personal-blog pipeline — picks an X-derived topic seed, drafts a
//! long-form essay via `blog_writer`, routes through Telegram review,
//! commits Markdown to disk + renders the static site.

pub mod markdown;
pub mod prompts;
pub mod render;
pub mod seed;
pub mod templates;

pub use markdown::{BlogPostFrontmatter, WriteMarkdownError, write_post_markdown};
pub use render::{RenderError, RenderedPostMeta, render_site};
pub use seed::{BlogSeed, SeedError, select_blog_seed};
