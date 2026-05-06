//! Document loader — reads plain text, Markdown, and PDF files into the knowledge base.

use std::path::Path;

use crate::auth::TenantScope;
use crate::error::Error;

use super::chunker::{ChunkConfig, split_into_chunks};
use super::{DocumentSource, KnowledgeBase};

/// Maximum bytes loaded from a single file by [`load_file`].
///
/// SECURITY (F-KB-3): a hostile filesystem (or accidental glob hit on a
/// large dump file) would otherwise OOM the process via
/// `read_to_string`. 50 MiB covers any reasonable Markdown / code file.
pub const KNOWLEDGE_LOAD_FILE_MAX_BYTES: u64 = 50 * 1024 * 1024;

/// Maximum bytes loaded from a single URL by [`load_url`].
///
/// SECURITY (F-KB-2): cap at 16 MiB to bound memory.
pub const KNOWLEDGE_LOAD_URL_MAX_BYTES: usize = 16 * 1024 * 1024;

/// Load a single file and index its chunks into the knowledge base under the given tenant.
///
/// SECURITY (F-KB-3): bounded by [`KNOWLEDGE_LOAD_FILE_MAX_BYTES`].
pub async fn load_file(
    kb: &dyn KnowledgeBase,
    scope: &TenantScope,
    path: &Path,
    config: &ChunkConfig,
) -> Result<usize, Error> {
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|e| Error::Knowledge(format!("failed to open {}: {e}", path.display())))?;
    let mut bytes: Vec<u8> = Vec::new();
    let read = (&mut file)
        .take(KNOWLEDGE_LOAD_FILE_MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|e| Error::Knowledge(format!("failed to read {}: {e}", path.display())))?;
    if read as u64 > KNOWLEDGE_LOAD_FILE_MAX_BYTES {
        return Err(Error::Knowledge(format!(
            "{} exceeds {KNOWLEDGE_LOAD_FILE_MAX_BYTES} bytes (F-KB-3)",
            path.display()
        )));
    }
    let content = String::from_utf8_lossy(&bytes).into_owned();

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
        kb.index(scope, chunk).await?;
    }
    Ok(count)
}

/// Load all files matching a glob pattern and index their chunks.
pub async fn load_glob(
    kb: &dyn KnowledgeBase,
    scope: &TenantScope,
    pattern: &str,
    config: &ChunkConfig,
) -> Result<usize, Error> {
    let paths = glob::glob(pattern)
        .map_err(|e| Error::Knowledge(format!("invalid glob pattern '{pattern}': {e}")))?;

    let mut total = 0;
    for entry in paths {
        let path = entry.map_err(|e| Error::Knowledge(format!("glob error: {e}")))?;
        if path.is_file() {
            match load_file(kb, scope, &path, config).await {
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
///
/// SECURITY (F-KB-2): the URL is validated via [`crate::http::SafeUrl::parse`]
/// (scheme allowlist + IP blocklist), the request uses
/// [`crate::http::safe_client_builder`] (redirect:none, no_proxy,
/// connect_timeout), and the body is capped at
/// [`KNOWLEDGE_LOAD_URL_MAX_BYTES`].
pub async fn load_url(
    kb: &dyn KnowledgeBase,
    scope: &TenantScope,
    url: &str,
    config: &ChunkConfig,
) -> Result<usize, Error> {
    // SECURITY (F-KB-2): SSRF blocklist + scheme allowlist.
    let safe = crate::http::SafeUrl::parse(url, crate::http::IpPolicy::default())
        .await
        .map_err(|e| Error::Knowledge(format!("URL refused for {url}: {e}")))?;

    let client = crate::http::safe_client_builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| Error::Knowledge(format!("client build error: {e}")))?;
    let response = client
        .get(safe.as_str())
        .send()
        .await
        .map_err(|e| Error::Knowledge(format!("failed to fetch {url}: {e}")))?;

    if !response.status().is_success() {
        return Err(Error::Knowledge(format!(
            "HTTP {} fetching {url}",
            response.status()
        )));
    }

    let body = crate::http::read_text_capped(response, KNOWLEDGE_LOAD_URL_MAX_BYTES)
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
        kb.index(scope, chunk).await?;
    }
    Ok(count)
}

/// Strip HTML tags from text, replacing them with spaces.
///
/// Skips content inside `<script>` and `<style>` tags. For full
/// HTML→markdown conversion a dedicated crate would be appropriate,
/// but for V1 tag stripping suffices.
pub fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut tag_name = String::new();
    let mut collecting_tag = false;
    let mut last_was_space = false;
    let mut skip_content = false; // true inside <script> or <style>

    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
            tag_name.clear();
            collecting_tag = true;
            if !skip_content && !last_was_space && !result.is_empty() {
                result.push(' ');
                last_was_space = true;
            }
        } else if ch == '>' && in_tag {
            in_tag = false;
            collecting_tag = false;
            let tag_lower = tag_name.to_lowercase();
            match tag_lower.as_str() {
                "script" | "style" => skip_content = true,
                "/script" | "/style" => skip_content = false,
                _ => {}
            }
        } else if in_tag && collecting_tag {
            if ch.is_whitespace() {
                collecting_tag = false;
            } else {
                tag_name.push(ch);
            }
        } else if !in_tag && !skip_content {
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

    fn s() -> TenantScope {
        TenantScope::default()
    }

    #[tokio::test]
    async fn load_file_indexes_content() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "Rust is a systems programming language.").unwrap();
        writeln!(tmp).unwrap();
        writeln!(tmp, "It provides memory safety without garbage collection.").unwrap();

        let kb = InMemoryKnowledgeBase::new();
        let count = load_file(&kb, &s(), tmp.path(), &ChunkConfig::default())
            .await
            .unwrap();
        assert!(count >= 1);

        let results = kb
            .search(
                &s(),
                KnowledgeQuery {
                    text: "rust memory".into(),
                    source_filter: None,
                    limit: 5,
                },
            )
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn load_file_nonexistent_returns_error() {
        let kb = InMemoryKnowledgeBase::new();
        let err = load_file(
            &kb,
            &s(),
            Path::new("/nonexistent/file.md"),
            &ChunkConfig::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::Knowledge(_)));
        // Either failed to open or failed to read — both are acceptable
        // (we now open then take(...).read_to_end, F-KB-3).
        let s = err.to_string();
        assert!(
            s.contains("failed to read") || s.contains("failed to open"),
            "expected open/read error, got: {s}"
        );
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
        let count = load_glob(&kb, &s(), &pattern, &ChunkConfig::default())
            .await
            .unwrap();
        assert!(count >= 2, "expected >= 2 chunks, got {count}");
        assert!(kb.chunk_count(&s()).await.unwrap() >= 2);
    }

    #[tokio::test]
    async fn load_glob_invalid_pattern_returns_error() {
        let kb = InMemoryKnowledgeBase::new();
        let err = load_glob(&kb, &s(), "[invalid", &ChunkConfig::default())
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
    fn strip_html_skips_script_content() {
        let html = "<p>Hello</p><script>var x = 1; alert('xss');</script><p>World</p>";
        let text = strip_html_tags(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("var x"));
    }

    #[test]
    fn strip_html_skips_style_content() {
        let html = "<p>Hello</p><style>body { color: red; }</style><p>World</p>";
        let text = strip_html_tags(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(!text.contains("color"));
        assert!(!text.contains("body"));
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
