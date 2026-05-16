//! Static-site renderer. Reads `posts_dir/*.md` (frontmatter + body),
//! renders each post into `out_dir/<slug>/index.html`, regenerates
//! `out_dir/index.html`, `out_dir/feed.xml`, `out_dir/sitemap.xml`,
//! and copies `style.css`.
//!
//! Pure I/O — no LLM calls, no network. Safe to run standalone via
//! the `heartbit_blog_render` binary or invoke after a successful
//! pipeline tick.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use minijinja::context;
use pulldown_cmark::{Options, Parser, html as md_html};
use serde::{Deserialize, Serialize};

use crate::blog::markdown::BlogPostFrontmatter;
use crate::blog::templates::build_env;

/// Metadata for a rendered blog post, returned by [`render_site`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderedPostMeta {
    /// URL slug (used as the output directory name).
    pub slug: String,
    /// Post title from frontmatter.
    pub title: String,
    /// Publication date from frontmatter.
    pub date: DateTime<Utc>,
    /// Excerpt from frontmatter (used in index, RSS, and meta tags).
    pub excerpt: String,
    /// Lowercase tags from frontmatter.
    pub tags: Vec<String>,
    /// Absolute path to the rendered `index.html` file.
    pub output_path: PathBuf,
}

/// Errors returned by [`render_site`].
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// I/O failure (e.g., permission denied, disk full).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// YAML frontmatter could not be deserialized.
    #[error("frontmatter parse error in {file}: {source}")]
    Frontmatter {
        /// Source file path.
        file: String,
        /// The underlying serde_yaml error.
        #[source]
        source: serde_yaml::Error,
    },
    /// File is missing the leading `---` frontmatter delimiter.
    #[error("missing frontmatter in {file} (no leading `---`)")]
    MissingFrontmatter {
        /// Source file path.
        file: String,
    },
    /// A minijinja template failed to render.
    #[error("template: {0}")]
    Template(#[from] minijinja::Error),
    /// The `style.css` source file does not exist.
    #[error("style.css not found at {0} — needed for copy to out_dir")]
    StyleNotFound(PathBuf),
}

/// Site-level config passed into every render call. Mirrors the
/// `PersonaBlogConfig` knobs but doesn't depend on the daemon crate
/// (so this module is testable in isolation).
#[derive(Debug, Clone)]
pub struct RenderConfig<'a> {
    /// Canonical base URL for the site (e.g. `https://pascal.heartbit.ai`).
    pub site_url: &'a str,
    /// Human-readable site name used in `<title>` and RSS channel title.
    pub site_title: &'a str,
    /// Path to `style.css` to copy into out_dir. Conventionally
    /// `blog-site/style.css`.
    pub style_css: &'a Path,
}

/// Walk `posts_dir`, render every `*.md`, regenerate the index +
/// feed + sitemap into `out_dir`, copy `style.css`. Returns metadata
/// for each rendered post (sorted newest-first).
pub fn render_site(
    posts_dir: &Path,
    out_dir: &Path,
    cfg: &RenderConfig<'_>,
) -> Result<Vec<RenderedPostMeta>, RenderError> {
    std::fs::create_dir_all(out_dir)?;

    let posts = read_posts(posts_dir)?;
    let env = build_env();
    let post_tmpl = env.get_template("post.html")?;
    let index_tmpl = env.get_template("index.html")?;

    let mut rendered: Vec<RenderedPostMeta> = Vec::new();
    for (front, body_md) in &posts {
        let body_html = markdown_to_html(body_md);
        let post_url = format!("{}/{}/", cfg.site_url.trim_end_matches('/'), front.slug);
        let post_html = post_tmpl.render(context! {
            site_title => cfg.site_title,
            site_url => cfg.site_url,
            post_url => &post_url,
            post => context! {
                title => &front.title,
                slug => &front.slug,
                date_iso => front.date.to_rfc3339(),
                date_human => front.date.format("%B %-d, %Y").to_string(),
                excerpt => &front.excerpt,
                tags => &front.tags,
                body_html => &body_html,
            },
        })?;

        let post_out_dir = out_dir.join(&front.slug);
        std::fs::create_dir_all(&post_out_dir)?;
        let out_path = post_out_dir.join("index.html");
        std::fs::write(&out_path, post_html)?;

        rendered.push(RenderedPostMeta {
            slug: front.slug.clone(),
            title: front.title.clone(),
            date: front.date,
            excerpt: front.excerpt.clone(),
            tags: front.tags.clone(),
            output_path: out_path,
        });
    }

    rendered.sort_by_key(|b| std::cmp::Reverse(b.date));

    let index_html = index_tmpl.render(context! {
        site_title => cfg.site_title,
        site_url => cfg.site_url,
        posts => rendered
            .iter()
            .map(|p| context! {
                slug => &p.slug,
                title => &p.title,
                date_iso => p.date.to_rfc3339(),
                date_human => p.date.format("%B %-d, %Y").to_string(),
                excerpt => &p.excerpt,
            })
            .collect::<Vec<_>>(),
    })?;
    std::fs::write(out_dir.join("index.html"), index_html)?;

    let feed_xml = render_rss(&rendered, cfg);
    std::fs::write(out_dir.join("feed.xml"), feed_xml)?;

    let sitemap_xml = render_sitemap(&rendered, cfg);
    std::fs::write(out_dir.join("sitemap.xml"), sitemap_xml)?;

    let robots = format!(
        "User-agent: *\nAllow: /\nSitemap: {}/sitemap.xml\n",
        cfg.site_url.trim_end_matches('/')
    );
    std::fs::write(out_dir.join("robots.txt"), robots)?;

    if !cfg.style_css.exists() {
        return Err(RenderError::StyleNotFound(cfg.style_css.to_path_buf()));
    }
    std::fs::copy(cfg.style_css, out_dir.join("style.css"))?;

    Ok(rendered)
}

fn read_posts(posts_dir: &Path) -> Result<Vec<(BlogPostFrontmatter, String)>, RenderError> {
    if !posts_dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(posts_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let content = std::fs::read_to_string(&path)?;
        let (front, body) = parse_post(&content, &path.display().to_string())?;
        out.push((front, body));
    }
    Ok(out)
}

fn parse_post(content: &str, file: &str) -> Result<(BlogPostFrontmatter, String), RenderError> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return Err(RenderError::MissingFrontmatter {
            file: file.to_string(),
        });
    }
    let after_first = &trimmed[3..];
    let end = after_first
        .find("\n---\n")
        .ok_or_else(|| RenderError::MissingFrontmatter {
            file: file.to_string(),
        })?;
    let yaml = &after_first[..end];
    let body = &after_first[end + 5..];
    let front: BlogPostFrontmatter =
        serde_yaml::from_str(yaml).map_err(|source| RenderError::Frontmatter {
            file: file.to_string(),
            source,
        })?;
    Ok((front, body.trim_start_matches('\n').to_string()))
}

fn markdown_to_html(md: &str) -> String {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_FOOTNOTES);
    let parser = Parser::new_ext(md, opts);
    let mut out = String::new();
    md_html::push_html(&mut out, parser);
    out
}

fn render_rss(rendered: &[RenderedPostMeta], cfg: &RenderConfig<'_>) -> String {
    let mut items = String::new();
    for p in rendered.iter().take(20) {
        let post_url = format!("{}/{}/", cfg.site_url.trim_end_matches('/'), p.slug);
        items.push_str(&format!(
            "    <item>\n      <title>{}</title>\n      <link>{}</link>\n      <guid>{}</guid>\n      <pubDate>{}</pubDate>\n      <description>{}</description>\n    </item>\n",
            xml_escape(&p.title),
            post_url,
            post_url,
            p.date.to_rfc2822(),
            xml_escape(&p.excerpt),
        ));
    }
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rss version=\"2.0\">\n  <channel>\n    <title>{}</title>\n    <link>{}</link>\n    <description>{}</description>\n{}  </channel>\n</rss>\n",
        xml_escape(cfg.site_title),
        cfg.site_url,
        xml_escape(cfg.site_title),
        items,
    )
}

fn render_sitemap(rendered: &[RenderedPostMeta], cfg: &RenderConfig<'_>) -> String {
    let mut urls = String::new();
    urls.push_str(&format!(
        "  <url><loc>{}/</loc></url>\n",
        cfg.site_url.trim_end_matches('/')
    ));
    for p in rendered {
        let post_url = format!("{}/{}/", cfg.site_url.trim_end_matches('/'), p.slug);
        urls.push_str(&format!(
            "  <url><loc>{}</loc><lastmod>{}</lastmod></url>\n",
            post_url,
            p.date.format("%Y-%m-%d"),
        ));
    }
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n{}</urlset>\n",
        urls,
    )
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blog::markdown::{BlogPostFrontmatter, write_post_markdown};
    use chrono::TimeZone;

    fn cfg<'a>(style_css: &'a Path) -> RenderConfig<'a> {
        RenderConfig {
            site_url: "https://pascal.heartbit.ai",
            site_title: "pascal.heartbit.ai",
            style_css,
        }
    }

    fn write_style(dir: &Path) -> PathBuf {
        let p = dir.join("style.css");
        std::fs::write(&p, "body{}").unwrap();
        p
    }

    fn fixture_post(slug: &str, days_ago: i64) -> (BlogPostFrontmatter, String) {
        (
            BlogPostFrontmatter {
                title: format!("Post {slug}"),
                date: Utc.with_ymd_and_hms(2026, 5, 16, 12, 0, 0).unwrap()
                    - chrono::Duration::days(days_ago),
                slug: slug.into(),
                excerpt: format!("Excerpt for {slug}."),
                tags: vec!["test".into()],
            },
            "# Heading\n\nParagraph with `code`.\n".into(),
        )
    }

    #[test]
    fn empty_posts_dir_renders_index_and_feed() {
        let tmp = tempfile::tempdir().unwrap();
        let posts_dir = tmp.path().join("posts");
        std::fs::create_dir_all(&posts_dir).unwrap();
        let out_dir = tmp.path().join("public");
        let style = write_style(tmp.path());
        let metas = render_site(&posts_dir, &out_dir, &cfg(&style)).unwrap();
        assert!(metas.is_empty());
        assert!(out_dir.join("index.html").exists());
        let idx = std::fs::read_to_string(out_dir.join("index.html")).unwrap();
        assert!(idx.contains("No posts yet."));
        assert!(out_dir.join("feed.xml").exists());
        assert!(out_dir.join("sitemap.xml").exists());
        assert!(out_dir.join("robots.txt").exists());
        assert!(out_dir.join("style.css").exists());
    }

    #[test]
    fn single_post_renders_to_slug_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style = write_style(tmp.path());

        let (front, body) = fixture_post("agent-loops", 1);
        write_post_markdown(&posts_dir, &front, &body).unwrap();
        let metas = render_site(&posts_dir, &out_dir, &cfg(&style)).unwrap();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].slug, "agent-loops");
        let post_html = out_dir.join("agent-loops").join("index.html");
        assert!(post_html.exists());
        let html = std::fs::read_to_string(&post_html).unwrap();
        assert!(html.contains("Post agent-loops"));
        assert!(html.contains("<p>Paragraph with <code>code</code>.</p>"));
    }

    #[test]
    fn multiple_posts_sorted_newest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style = write_style(tmp.path());

        let (f1, b1) = fixture_post("old", 10);
        let (f2, b2) = fixture_post("new", 1);
        let (f3, b3) = fixture_post("middle", 5);
        write_post_markdown(&posts_dir, &f1, &b1).unwrap();
        write_post_markdown(&posts_dir, &f2, &b2).unwrap();
        write_post_markdown(&posts_dir, &f3, &b3).unwrap();

        let metas = render_site(&posts_dir, &out_dir, &cfg(&style)).unwrap();
        assert_eq!(metas.len(), 3);
        assert_eq!(metas[0].slug, "new");
        assert_eq!(metas[1].slug, "middle");
        assert_eq!(metas[2].slug, "old");
    }

    #[test]
    fn index_lists_posts_with_excerpts_and_links() {
        let tmp = tempfile::tempdir().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style = write_style(tmp.path());

        let (f, b) = fixture_post("agent-loops", 1);
        write_post_markdown(&posts_dir, &f, &b).unwrap();
        render_site(&posts_dir, &out_dir, &cfg(&style)).unwrap();
        let idx = std::fs::read_to_string(out_dir.join("index.html")).unwrap();
        assert!(idx.contains("Excerpt for agent-loops."));
        assert!(idx.contains("/agent-loops/"));
    }

    #[test]
    fn malformed_frontmatter_is_reported_with_filename() {
        let tmp = tempfile::tempdir().unwrap();
        let posts_dir = tmp.path().join("posts");
        std::fs::create_dir_all(&posts_dir).unwrap();
        std::fs::write(posts_dir.join("bad.md"), "no frontmatter here\n").unwrap();
        let out_dir = tmp.path().join("public");
        let style = write_style(tmp.path());
        let err = render_site(&posts_dir, &out_dir, &cfg(&style)).unwrap_err();
        match err {
            RenderError::MissingFrontmatter { file } => assert!(file.contains("bad.md")),
            other => panic!("expected MissingFrontmatter, got {other:?}"),
        }
    }

    #[test]
    fn rss_feed_includes_post_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style = write_style(tmp.path());

        let (f, b) = fixture_post("agent-loops", 1);
        write_post_markdown(&posts_dir, &f, &b).unwrap();
        render_site(&posts_dir, &out_dir, &cfg(&style)).unwrap();
        let feed = std::fs::read_to_string(out_dir.join("feed.xml")).unwrap();
        assert!(feed.contains("<rss version=\"2.0\">"));
        assert!(feed.contains("https://pascal.heartbit.ai/agent-loops/"));
        assert!(feed.contains("Post agent-loops"));
    }

    #[test]
    fn sitemap_includes_homepage_and_posts() {
        let tmp = tempfile::tempdir().unwrap();
        let posts_dir = tmp.path().join("posts");
        let out_dir = tmp.path().join("public");
        let style = write_style(tmp.path());

        let (f, b) = fixture_post("agent-loops", 1);
        write_post_markdown(&posts_dir, &f, &b).unwrap();
        render_site(&posts_dir, &out_dir, &cfg(&style)).unwrap();
        let sitemap = std::fs::read_to_string(out_dir.join("sitemap.xml")).unwrap();
        assert!(sitemap.contains("https://pascal.heartbit.ai/"));
        assert!(sitemap.contains("https://pascal.heartbit.ai/agent-loops/"));
    }

    #[test]
    fn xml_escape_handles_special_chars() {
        let s = xml_escape("a&b<c>d\"e'f");
        assert_eq!(s, "a&amp;b&lt;c&gt;d&quot;e&apos;f");
    }
}
