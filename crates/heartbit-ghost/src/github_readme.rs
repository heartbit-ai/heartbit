//! GitHub Profile README auto-update — pure rendering + an I/O helper
//! that writes + commits + pushes the result. Triggered by the
//! `handle_persona_blog` handler after a successful blog publish.

use std::path::{Path, PathBuf};

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

/// Inputs to one README update.
#[derive(Debug, Clone)]
pub struct UpdateReadmeParams<'a> {
    /// Absolute path to the local clone of the profile repo.
    pub local_repo_path: &'a Path,
    /// Path to the operator-authored bio template (absolute, OR relative
    /// to `local_repo_path`).
    pub bio_template_path: &'a Path,
    /// Path to the directory holding blog post `*.md` files.
    pub blog_posts_dir: &'a Path,
    /// Public site URL.
    pub site_url: &'a str,
    /// Commit author name.
    pub git_author_name: &'a str,
    /// Commit author email.
    pub git_author_email: &'a str,
    /// Slug of the new post (used in the commit message). May be empty
    /// if the trigger isn't a specific post.
    pub new_post_slug: &'a str,
}

/// Default minimal bio used when `bio_template_path` doesn't exist.
const DEFAULT_BIO: &str = "# Pascal Le Clech\n\nMulti-agent runtime, Rust, AI infra.\n";

/// Errors emitted by [`update_github_readme`].
#[derive(Debug, thiserror::Error)]
pub enum UpdateReadmeError {
    /// `local_repo_path` doesn't exist or isn't a directory.
    #[error("local_repo_path missing or not a directory: {0}")]
    RepoPathMissing(PathBuf),
    /// Filesystem I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Frontmatter parse failure on a post file (rare).
    #[error("read posts: {0}")]
    ReadPosts(String),
    /// Git command failed (returned non-zero or timed out).
    #[error("git: {0}")]
    Git(String),
}

/// Read recent blog posts from `posts_dir`, render the README, write it
/// into `local_repo_path/README.md`, then `git add + commit + push`.
/// Bounded at 5 minutes for the git push.
pub async fn update_github_readme(p: UpdateReadmeParams<'_>) -> Result<(), UpdateReadmeError> {
    if !p.local_repo_path.exists() || !p.local_repo_path.is_dir() {
        return Err(UpdateReadmeError::RepoPathMissing(
            p.local_repo_path.to_path_buf(),
        ));
    }

    // 1) Bio: load template (absolute path, or relative to repo).
    let bio_abs = if p.bio_template_path.is_absolute() {
        p.bio_template_path.to_path_buf()
    } else {
        p.local_repo_path.join(p.bio_template_path)
    };
    let bio = match std::fs::read_to_string(&bio_abs) {
        Ok(s) => s,
        Err(_) => DEFAULT_BIO.to_string(),
    };

    // 2) Recent posts: read every *.md in posts_dir, parse frontmatter,
    //    sort newest-first, take top 3.
    let recent = read_recent_posts(p.blog_posts_dir, 3)?;

    // 3) Render the auto section + merge with whatever's currently in
    //    the README (preserves operator edits above the marker).
    //    On first run (no existing marker), use the full render so the
    //    bio template is seeded into the README; on subsequent runs only
    //    the auto section (below the marker) is replaced.
    let readme_path = p.local_repo_path.join("README.md");
    let existing = std::fs::read_to_string(&readme_path).unwrap_or_default();
    let rendered = render_readme(&bio, &recent, p.site_url);
    let merged = if existing.contains(AUTO_GENERATED_MARKER) {
        let auto_section = extract_auto_section(&rendered);
        merge_readme(&existing, auto_section)
    } else {
        // First run: seed the full render (bio + auto section).
        rendered
    };
    std::fs::write(&readme_path, &merged)?;

    // 4) Git: add + commit + push.
    let commit_msg = if p.new_post_slug.is_empty() {
        "profile: refresh recent essays".to_string()
    } else {
        format!("profile: feature {}", p.new_post_slug)
    };
    let sh = format!(
        "git add README.md && \
         git -c user.name='{name}' -c user.email='{email}' \
             commit -m '{msg}' && \
         git push origin HEAD",
        name = shell_escape(p.git_author_name),
        email = shell_escape(p.git_author_email),
        msg = shell_escape(&commit_msg),
    );
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(300),
        tokio::process::Command::new("sh")
            .current_dir(p.local_repo_path)
            .arg("-c")
            .arg(&sh)
            .output(),
    )
    .await;
    match result {
        Ok(Ok(out)) => {
            if out.status.success() {
                Ok(())
            } else {
                Err(UpdateReadmeError::Git(format!(
                    "exit {}: stdout={} stderr={}",
                    out.status,
                    String::from_utf8_lossy(&out.stdout).trim(),
                    String::from_utf8_lossy(&out.stderr).trim()
                )))
            }
        }
        Ok(Err(e)) => Err(UpdateReadmeError::Git(format!("spawn: {e}"))),
        Err(_) => Err(UpdateReadmeError::Git("timeout after 300s".into())),
    }
}

fn read_recent_posts(
    posts_dir: &Path,
    n: usize,
) -> Result<Vec<BlogPostFrontmatter>, UpdateReadmeError> {
    if !posts_dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries: Vec<BlogPostFrontmatter> = Vec::new();
    for entry in std::fs::read_dir(posts_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let content = std::fs::read_to_string(&path)?;
        if let Some(front) = parse_frontmatter(&content) {
            entries.push(front);
        }
    }
    entries.sort_by_key(|p| std::cmp::Reverse(p.date));
    entries.truncate(n);
    Ok(entries)
}

fn parse_frontmatter(content: &str) -> Option<BlogPostFrontmatter> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return None;
    }
    let after_first = &trimmed[3..];
    let end = after_first.find("\n---\n")?;
    let yaml = &after_first[..end];
    serde_yaml::from_str::<BlogPostFrontmatter>(yaml).ok()
}

fn extract_auto_section(rendered: &str) -> &str {
    if let Some(idx) = rendered.find(AUTO_GENERATED_MARKER) {
        &rendered[idx..]
    } else {
        rendered
    }
}

fn shell_escape(s: &str) -> String {
    // Simple single-quote escape: replace `'` with `'\''`. Sufficient for
    // author names/emails which don't contain control characters.
    s.replace('\'', "'\\''")
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

    use crate::blog::markdown::write_post_markdown;
    use tempfile::TempDir;

    fn run_git(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?} failed");
    }

    fn setup_local_repo() -> (TempDir, std::path::PathBuf) {
        let tmp = TempDir::new().unwrap();
        let repo = tmp.path().join("repo");
        let upstream = tmp.path().join("upstream");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&upstream).unwrap();
        run_git(&upstream, &["init", "--bare", "-b", "main"]);
        run_git(&repo, &["init", "-b", "main"]);
        run_git(&repo, &["config", "user.email", "test@test"]);
        run_git(&repo, &["config", "user.name", "test"]);
        run_git(&repo, &["config", "commit.gpgsign", "false"]);
        run_git(
            &repo,
            &[
                "remote",
                "add",
                "origin",
                &upstream.to_string_lossy().into_owned(),
            ],
        );
        std::fs::write(repo.join("README.md"), "# Original\n").unwrap();
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "-m", "init"]);
        run_git(&repo, &["push", "-u", "origin", "main"]);
        (tmp, repo)
    }

    #[tokio::test]
    async fn update_returns_repo_missing_when_path_doesnt_exist() {
        let bogus = std::path::PathBuf::from("/nope/does/not/exist");
        let err = update_github_readme(UpdateReadmeParams {
            local_repo_path: &bogus,
            bio_template_path: Path::new("bio.md"),
            blog_posts_dir: Path::new("/tmp"),
            site_url: "https://pascal.heartbit.ai",
            git_author_name: "t",
            git_author_email: "t@t",
            new_post_slug: "x",
        })
        .await
        .unwrap_err();
        assert!(matches!(err, UpdateReadmeError::RepoPathMissing(_)));
    }

    #[tokio::test]
    async fn update_writes_and_pushes_readme() {
        let (_tmp, repo) = setup_local_repo();
        let posts_dir = repo.parent().unwrap().join("posts");
        std::fs::create_dir_all(&posts_dir).unwrap();
        let front = post("agent-loops", "Agent loops", "Why loops compound costs.", 0);
        write_post_markdown(&posts_dir, &front, "Body").unwrap();
        update_github_readme(UpdateReadmeParams {
            local_repo_path: &repo,
            bio_template_path: Path::new("bio.md"),
            blog_posts_dir: &posts_dir,
            site_url: "https://pascal.heartbit.ai",
            git_author_name: "test",
            git_author_email: "test@test",
            new_post_slug: "agent-loops",
        })
        .await
        .unwrap();
        let readme = std::fs::read_to_string(repo.join("README.md")).unwrap();
        assert!(readme.contains("Agent loops"));
        assert!(readme.contains(AUTO_GENERATED_MARKER));
    }

    #[tokio::test]
    async fn update_uses_default_bio_when_template_missing() {
        let (_tmp, repo) = setup_local_repo();
        // The repo starts with "# Original\n" (no AUTO_GENERATED_MARKER),
        // so this is a first-run — the full render (DEFAULT_BIO + auto
        // section) is written, seeding "# Pascal Le Clech" into the README.
        let posts_dir = repo.parent().unwrap().join("posts");
        std::fs::create_dir_all(&posts_dir).unwrap();
        update_github_readme(UpdateReadmeParams {
            local_repo_path: &repo,
            bio_template_path: Path::new("does-not-exist.md"),
            blog_posts_dir: &posts_dir,
            site_url: "https://pascal.heartbit.ai",
            git_author_name: "test",
            git_author_email: "test@test",
            new_post_slug: "",
        })
        .await
        .unwrap();
        let readme = std::fs::read_to_string(repo.join("README.md")).unwrap();
        // Default bio header is "# Pascal Le Clech" (from DEFAULT_BIO)
        assert!(readme.contains("# Pascal Le Clech"));
        assert!(readme.contains(AUTO_GENERATED_MARKER));
    }
}
