//! Markdown writer — serializes a blog post (frontmatter + body) to a
//! file in the posts directory. Single source of truth for the on-disk
//! Markdown shape consumed by the renderer.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// YAML frontmatter for a single blog post. Persisted at the top of the
/// Markdown file between `---` delimiters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlogPostFrontmatter {
    /// Post title.
    pub title: String,
    /// RFC3339 UTC timestamp.
    pub date: DateTime<Utc>,
    /// URL slug (must be URL-safe — see `slug::slugify`).
    pub slug: String,
    /// Excerpt for `<meta description>`, OpenGraph, RSS, and the index
    /// page card. Recommended 120-160 chars; not enforced.
    pub excerpt: String,
    /// Lowercase tags. Empty Vec is OK.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Errors returned by [`write_post_markdown`].
#[derive(Debug, thiserror::Error)]
pub enum WriteMarkdownError {
    /// I/O failure (e.g., permission denied, disk full).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// YAML serialization failure.
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
    /// The slug is empty or contains non-URL-safe characters.
    #[error("invalid slug: must be non-empty and contain only [a-z0-9-]")]
    InvalidSlug,
}

/// Write a Markdown post to `<posts_dir>/<YYYY-MM-DD>-<slug>.md`.
///
/// Returns the absolute path of the written file. Fails if the slug is
/// empty or contains non-URL-safe characters (use `slug::slugify` first).
pub fn write_post_markdown(
    posts_dir: &Path,
    front: &BlogPostFrontmatter,
    body: &str,
) -> Result<PathBuf, WriteMarkdownError> {
    if front.slug.is_empty()
        || !front
            .slug
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(WriteMarkdownError::InvalidSlug);
    }

    std::fs::create_dir_all(posts_dir)?;
    let filename = format!("{}-{}.md", front.date.format("%Y-%m-%d"), front.slug);
    let path = posts_dir.join(filename);

    let yaml = serde_yaml::to_string(front)?;
    let mut content = String::with_capacity(yaml.len() + body.len() + 16);
    content.push_str("---\n");
    content.push_str(&yaml);
    content.push_str("---\n\n");
    content.push_str(body);
    if !body.ends_with('\n') {
        content.push('\n');
    }

    std::fs::write(&path, content)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixture_front() -> BlogPostFrontmatter {
        BlogPostFrontmatter {
            title: "Agent loops without guardrails".into(),
            date: Utc.with_ymd_and_hms(2026, 5, 16, 12, 0, 0).unwrap(),
            slug: "agent-loops-without-guardrails".into(),
            excerpt: "Why a 20-step tool loop with 1000 tokens per step costs you 210k tokens — and what to do about it.".into(),
            tags: vec!["agents".into(), "llm-cost".into()],
        }
    }

    #[test]
    fn write_post_creates_dated_filename() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_post_markdown(dir.path(), &fixture_front(), "Body content.\n").unwrap();
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()).unwrap(),
            "2026-05-16-agent-loops-without-guardrails.md"
        );
    }

    #[test]
    fn write_post_includes_frontmatter_and_body() {
        let dir = tempfile::tempdir().unwrap();
        let body = "Opening paragraph.\n\n## Section\n\nDetail.\n";
        let path = write_post_markdown(dir.path(), &fixture_front(), body).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("---\n"));
        assert!(content.contains("title: Agent loops without guardrails"));
        assert!(content.contains("slug: agent-loops-without-guardrails"));
        assert!(content.contains("\n---\n\nOpening paragraph."));
        assert!(content.contains("## Section"));
    }

    #[test]
    fn write_post_rejects_invalid_slug() {
        let dir = tempfile::tempdir().unwrap();
        let mut bad = fixture_front();
        bad.slug = "Has Spaces!".into();
        let err = write_post_markdown(dir.path(), &bad, "x").unwrap_err();
        assert!(matches!(err, WriteMarkdownError::InvalidSlug));
    }

    #[test]
    fn write_post_rejects_empty_slug() {
        let dir = tempfile::tempdir().unwrap();
        let mut bad = fixture_front();
        bad.slug = String::new();
        let err = write_post_markdown(dir.path(), &bad, "x").unwrap_err();
        assert!(matches!(err, WriteMarkdownError::InvalidSlug));
    }

    #[test]
    fn write_post_creates_posts_dir_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested").join("posts");
        let path = write_post_markdown(&nested, &fixture_front(), "x").unwrap();
        assert!(path.starts_with(&nested));
        assert!(path.exists());
    }
}
