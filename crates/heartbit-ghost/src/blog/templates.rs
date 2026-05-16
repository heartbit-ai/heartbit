//! minijinja templates for the blog SSG — embedded into the binary via
//! `include_str!` so we don't depend on a filesystem layout at runtime.
//! The source-of-truth files live in `blog-site/templates/`; the SSG
//! tests assert the embedded strings match.

const BASE_HTML: &str = include_str!("../../../../blog-site/templates/base.html");
const POST_HTML: &str = include_str!("../../../../blog-site/templates/post.html");
const INDEX_HTML: &str = include_str!("../../../../blog-site/templates/index.html");

/// Build a `minijinja::Environment` pre-loaded with the 3 templates.
pub fn build_env() -> minijinja::Environment<'static> {
    let mut env = minijinja::Environment::new();
    env.add_template("base.html", BASE_HTML)
        .expect("base.html parses");
    env.add_template("post.html", POST_HTML)
        .expect("post.html parses");
    env.add_template("index.html", INDEX_HTML)
        .expect("index.html parses");
    env
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::context;

    #[test]
    fn templates_parse() {
        let _env = build_env();
    }

    #[test]
    fn base_template_renders_minimum_context() {
        let env = build_env();
        let tmpl = env.get_template("base.html").unwrap();
        let out = tmpl
            .render(context! {
                site_title => "test.example.com",
                site_url => "https://test.example.com",
            })
            .unwrap();
        assert!(out.contains("<!DOCTYPE html>"));
        assert!(out.contains("test.example.com"));
        assert!(out.contains("/feed.xml"));
        assert!(out.contains("/style.css"));
    }

    #[test]
    fn post_template_renders_with_post_context() {
        let env = build_env();
        let tmpl = env.get_template("post.html").unwrap();
        let out = tmpl
            .render(context! {
                site_title => "test.example.com",
                site_url => "https://test.example.com",
                post_url => "https://test.example.com/agent-loops/",
                post => context! {
                    title => "Agent loops",
                    slug => "agent-loops",
                    date_iso => "2026-05-16T12:00:00Z",
                    date_human => "May 16, 2026",
                    excerpt => "Why agent loops compound costs.",
                    body_html => "<p>Body content.</p>",
                    tags => vec!["agents", "cost"],
                },
            })
            .unwrap();
        assert!(out.contains("Agent loops"));
        assert!(out.contains("May 16, 2026"));
        assert!(out.contains("<p>Body content.</p>"));
        assert!(out.contains("BlogPosting"));
        assert!(out.contains("\"agents\""));
        assert!(out.contains("\"cost\""));
    }

    #[test]
    fn index_template_renders_empty_state() {
        let env = build_env();
        let tmpl = env.get_template("index.html").unwrap();
        let out = tmpl
            .render(context! {
                site_title => "test",
                site_url => "https://test.example.com",
                posts => Vec::<minijinja::Value>::new(),
            })
            .unwrap();
        assert!(out.contains("No posts yet."));
    }

    #[test]
    fn index_template_renders_post_list() {
        let env = build_env();
        let tmpl = env.get_template("index.html").unwrap();
        let out = tmpl
            .render(context! {
                site_title => "test",
                site_url => "https://test.example.com",
                posts => vec![
                    context! {
                        slug => "first",
                        title => "First Post",
                        date_iso => "2026-05-16T12:00:00Z",
                        date_human => "May 16, 2026",
                        excerpt => "First excerpt.",
                    },
                ],
            })
            .unwrap();
        assert!(out.contains("First Post"));
        assert!(out.contains("/first/"));
        assert!(out.contains("First excerpt."));
        assert!(!out.contains("No posts yet."));
    }
}
