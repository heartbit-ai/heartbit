use std::path::Path;

use crate::error::Error;

use super::chunker::{ChunkConfig, split_into_chunks};
use super::{DocumentSource, KnowledgeBase};

/// Load a single file and index its chunks into the knowledge base.
pub async fn load_file(
    kb: &dyn KnowledgeBase,
    path: &Path,
    config: &ChunkConfig,
) -> Result<usize, Error> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| Error::Knowledge(format!("failed to read {}: {e}", path.display())))?;

    let title = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());

    let source = DocumentSource {
        uri: path.display().to_string(),
        title,
    };

    let chunks = split_into_chunks(&content, &source, config);
    let count = chunks.len();
    for chunk in chunks {
        kb.index(chunk).await?;
    }
    Ok(count)
}

/// Load all files matching a glob pattern and index their chunks.
pub async fn load_glob(
    kb: &dyn KnowledgeBase,
    pattern: &str,
    config: &ChunkConfig,
) -> Result<usize, Error> {
    let paths = glob::glob(pattern)
        .map_err(|e| Error::Knowledge(format!("invalid glob pattern '{pattern}': {e}")))?;

    let mut total = 0;
    for entry in paths {
        let path = entry.map_err(|e| Error::Knowledge(format!("glob error: {e}")))?;
        if path.is_file() {
            match load_file(kb, &path, config).await {
                Ok(count) => total += count,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "skipping file");
                }
            }
        }
    }
    Ok(total)
}

/// Load a URL, strip HTML tags, and index chunks.
pub async fn load_url(
    kb: &dyn KnowledgeBase,
    url: &str,
    config: &ChunkConfig,
) -> Result<usize, Error> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| Error::Knowledge(format!("failed to fetch {url}: {e}")))?;

    if !response.status().is_success() {
        return Err(Error::Knowledge(format!(
            "HTTP {} fetching {url}",
            response.status()
        )));
    }

    let body = response
        .text()
        .await
        .map_err(|e| Error::Knowledge(format!("failed to read body from {url}: {e}")))?;

    let content = strip_html_tags(&body);

    let source = DocumentSource {
        uri: url.to_string(),
        title: url.to_string(),
    };

    let chunks = split_into_chunks(&content, &source, config);
    let count = chunks.len();
    for chunk in chunks {
        kb.index(chunk).await?;
    }
    Ok(count)
}

/// Strip HTML tags from text, replacing them with spaces.
///
/// This is a simple regex-free parser. For full HTML→markdown conversion
/// a dedicated crate would be appropriate, but for V1 tag stripping suffices.
pub fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut last_was_space = false;

    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
            if !last_was_space && !result.is_empty() {
                result.push(' ');
                last_was_space = true;
            }
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            if ch.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(ch);
                last_was_space = false;
            }
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::KnowledgeQuery;
    use crate::knowledge::in_memory::InMemoryKnowledgeBase;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn load_file_indexes_content() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "Rust is a systems programming language.").unwrap();
        writeln!(tmp).unwrap();
        writeln!(tmp, "It provides memory safety without garbage collection.").unwrap();

        let kb = InMemoryKnowledgeBase::new();
        let count = load_file(&kb, tmp.path(), &ChunkConfig::default())
            .await
            .unwrap();
        assert!(count >= 1);

        let results = kb
            .search(KnowledgeQuery {
                text: "rust memory".into(),
                source_filter: None,
                limit: 5,
            })
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn load_file_nonexistent_returns_error() {
        let kb = InMemoryKnowledgeBase::new();
        let err = load_file(
            &kb,
            Path::new("/nonexistent/file.md"),
            &ChunkConfig::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::Knowledge(_)));
        assert!(err.to_string().contains("failed to read"));
    }

    #[tokio::test]
    async fn load_glob_collects_files() {
        let dir = tempfile::tempdir().unwrap();
        let f1 = dir.path().join("doc1.md");
        let f2 = dir.path().join("doc2.md");
        std::fs::write(&f1, "First document about rust.").unwrap();
        std::fs::write(&f2, "Second document about async.").unwrap();

        let kb = InMemoryKnowledgeBase::new();
        let pattern = format!("{}/*.md", dir.path().display());
        let count = load_glob(&kb, &pattern, &ChunkConfig::default())
            .await
            .unwrap();
        assert!(count >= 2, "expected >= 2 chunks, got {count}");
        assert!(kb.chunk_count().await.unwrap() >= 2);
    }

    #[tokio::test]
    async fn load_glob_invalid_pattern_returns_error() {
        let kb = InMemoryKnowledgeBase::new();
        let err = load_glob(&kb, "[invalid", &ChunkConfig::default())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Knowledge(_)));
    }

    #[test]
    fn strip_html_basic() {
        let html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>";
        let text = strip_html_tags(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Hello world"));
        assert!(!text.contains('<'));
        assert!(!text.contains('>'));
    }

    #[test]
    fn strip_html_preserves_plain_text() {
        let text = "Just plain text, no HTML.";
        assert_eq!(strip_html_tags(text), text);
    }

    #[test]
    fn strip_html_collapses_whitespace() {
        let html = "<p>  lots   of    spaces  </p>";
        let text = strip_html_tags(html);
        assert_eq!(text, "lots of spaces");
    }

    #[test]
    fn strip_html_empty_input() {
        assert_eq!(strip_html_tags(""), "");
    }

    #[test]
    fn strip_html_nested_tags() {
        let html = "<div><span>nested</span> content</div>";
        let text = strip_html_tags(html);
        assert!(text.contains("nested"));
        assert!(text.contains("content"));
    }
}
