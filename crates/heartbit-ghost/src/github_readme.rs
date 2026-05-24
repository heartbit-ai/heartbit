//! GitHub Profile README auto-update — pure rendering + an I/O helper
//! that writes + commits + pushes the result. Triggered by the
//! `handle_persona_blog` handler after a successful blog publish.

use crate::blog::markdown::BlogPostFrontmatter;

/// Marker that separates operator-authored content (above) from the
/// auto-generated section (below). The render preserves everything
/// above and replaces everything from the marker downward.
pub const AUTO_GENERATED_MARKER: &str = "<!-- AUTO-GENERATED: do not edit below this line -->";

/// Render the README given an operator bio + recent posts. Returns
/// the full README content (including the bio above the marker).
///
/// `recent_posts` should be the 3 most recent posts, sorted newest
/// first (caller's responsibility — this function does not sort).
pub fn render_readme(
    bio_template: &str,
    recent_posts: &[BlogPostFrontmatter],
    site_url: &str,
) -> String {
    let trimmed_url = site_url.trim_end_matches('/');
    let bio = bio_template.trim_end();
    let mut out = String::new();
    out.push_str(bio);
    out.push_str("\n\n");
    out.push_str(AUTO_GENERATED_MARKER);
    out.push_str("\n## Recent essays\n\n");
    if recent_posts.is_empty() {
        out.push_str("_No essays yet._\n");
    } else {
        for p in recent_posts.iter().take(3) {
            out.push_str(&format!(
                "- [{}]({}/{}/) — *{}* ({})\n",
                p.title,
                trimmed_url,
                p.slug,
                p.excerpt,
                p.date.format("%Y-%m-%d")
            ));
        }
    }
    out.push_str("\n<sub>Auto-updated on each new essay. Source: ");
    out.push_str(trimmed_url);
    out.push_str("</sub>\n");
    out
}

/// Merge an auto-generated section into an existing README, preserving
/// any content the operator wrote above [`AUTO_GENERATED_MARKER`]. If
/// the README has no marker (first run), the auto-generated section is
/// appended after a blank line.
pub fn merge_readme(existing: &str, auto_section: &str) -> String {
    if let Some(idx) = existing.find(AUTO_GENERATED_MARKER) {
        let preserved = &existing[..idx];
        let mut out = preserved.trim_end_matches('\n').to_string();
        out.push_str("\n\n");
        out.push_str(auto_section);
        out
    } else {
        // First run — append marker + auto section.
        let mut out = existing.trim_end_matches('\n').to_string();
        out.push_str("\n\n");
        out.push_str(auto_section);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn post(slug: &str, title: &str, excerpt: &str, days_ago: i64) -> BlogPostFrontmatter {
        let date =
            Utc.with_ymd_and_hms(2026, 5, 24, 12, 0, 0).unwrap() - chrono::Duration::days(days_ago);
        BlogPostFrontmatter {
            title: title.into(),
            date,
            slug: slug.into(),
            excerpt: excerpt.into(),
            tags: vec![],
        }
    }

    #[test]
    fn renders_empty_state_when_no_posts() {
        let out = render_readme("# Hello\n\nBio.", &[], "https://pascal.heartbit.ai");
        assert!(out.contains("_No essays yet._"));
        assert!(out.contains(AUTO_GENERATED_MARKER));
        assert!(out.contains("# Hello"));
        assert!(out.contains("Bio."));
    }

    #[test]
    fn renders_single_post_with_link() {
        let posts = vec![post(
            "agent-loops",
            "Agent loops",
            "Why loops compound costs.",
            1,
        )];
        let out = render_readme("# Pascal", &posts, "https://pascal.heartbit.ai");
        assert!(out.contains("[Agent loops](https://pascal.heartbit.ai/agent-loops/)"));
        assert!(out.contains("*Why loops compound costs.*"));
    }

    #[test]
    fn caps_at_3_posts_even_if_more_provided() {
        let posts = vec![
            post("p1", "Post 1", "Excerpt 1", 1),
            post("p2", "Post 2", "Excerpt 2", 2),
            post("p3", "Post 3", "Excerpt 3", 3),
            post("p4", "Post 4", "Excerpt 4", 4),
            post("p5", "Post 5", "Excerpt 5", 5),
        ];
        let out = render_readme("# Pascal", &posts, "https://pascal.heartbit.ai");
        assert!(out.contains("Post 1"));
        assert!(out.contains("Post 3"));
        assert!(!out.contains("Post 4"));
        assert!(!out.contains("Post 5"));
    }

    #[test]
    fn trims_trailing_slash_from_site_url() {
        let posts = vec![post("hello", "Hello", "Body.", 0)];
        let out = render_readme("# Bio", &posts, "https://pascal.heartbit.ai/");
        assert!(out.contains("https://pascal.heartbit.ai/hello/"));
        assert!(!out.contains("//hello/"));
    }

    #[test]
    fn bio_preserved_verbatim_above_marker() {
        let bio = "# Custom Header\n\nMulti-paragraph\n\nbio content.";
        let out = render_readme(bio, &[], "https://pascal.heartbit.ai");
        let marker_idx = out.find(AUTO_GENERATED_MARKER).unwrap();
        let before_marker = &out[..marker_idx];
        assert!(before_marker.contains("# Custom Header"));
        assert!(before_marker.contains("Multi-paragraph"));
        assert!(before_marker.contains("bio content."));
    }

    #[test]
    fn merge_preserves_content_above_marker() {
        let existing = "# Pascal Le Clech\n\nHello.\n\n<!-- AUTO-GENERATED: do not edit below this line -->\n## Old stale section\n";
        let new_section = "<!-- AUTO-GENERATED: do not edit below this line -->\n## Recent essays\n\n- New entry\n";
        let merged = merge_readme(existing, new_section);
        assert!(merged.contains("# Pascal Le Clech"));
        assert!(merged.contains("Hello."));
        assert!(merged.contains("New entry"));
        assert!(!merged.contains("Old stale section"));
    }

    #[test]
    fn merge_appends_marker_on_first_run() {
        let existing = "# Pascal Le Clech\n\nBio.";
        let new_section =
            "<!-- AUTO-GENERATED: do not edit below this line -->\n## Recent essays\n\n- New\n";
        let merged = merge_readme(existing, new_section);
        assert!(merged.starts_with("# Pascal Le Clech"));
        assert!(merged.contains(AUTO_GENERATED_MARKER));
        assert!(merged.contains("New"));
    }
}
