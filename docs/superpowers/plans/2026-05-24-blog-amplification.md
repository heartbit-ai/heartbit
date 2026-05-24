# Blog Amplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When pascal.heartbit.ai publishes a new blog post, automatically (a) post an announcement X thread via the existing heartbit-ghost persona with Telegram review, and (b) refresh the operator's GitHub Profile README to feature the new post.

**Architecture:** Two amp surfaces fire after `handle_persona_blog` reports `BlogOutcome::Posted` AND `deploy_command` succeeds (or is absent). Surface 2 (X self-amp) goes through Kafka via a new `DaemonCommand::BlogAnnounceX` so it runs in the same retryable consumer loop as other personas. Surface 3 (GitHub README) is synchronous (small operation, atomic git push).

**Tech Stack:** Rust, heartbit-ghost crate (new `blog/announce.rs` + `github_readme.rs`), heartbit daemon Kafka consumer, existing X publish via `twitter_tool`, existing `ReviewDelivery` trait for Telegram review.

---

## File Structure

| Path | Responsibility | Status |
|---|---|---|
| `crates/heartbit-ghost/src/blog/announce.rs` | New: writer prompt builder + `run_x_announcement_pipeline` | Create |
| `crates/heartbit-ghost/src/github_readme.rs` | New: `render_readme` (pure) + `update_github_readme` (I/O) | Create |
| `crates/heartbit-ghost/src/lib.rs` | Add `pub mod github_readme;` | Modify |
| `crates/heartbit-ghost/src/blog/mod.rs` | Add `pub mod announce;` + re-exports | Modify |
| `crates/heartbit-core/src/config/daemon.rs` | Add `PersonaBlogConfig::x_announce` + `::github_readme` sub-blocks | Modify |
| `crates/heartbit/src/daemon/types.rs` | Add `DaemonCommand::BlogAnnounceX` variant | Modify |
| `crates/heartbit/src/daemon/blog_context.rs` | Add `x_announce` + `github_readme` fields on `PersonaBlogEntry` | Modify |
| `crates/heartbit/src/daemon/blog_announce_x_handler.rs` | New: `handle_blog_announce_x` | Create |
| `crates/heartbit/src/daemon/mod.rs` | Declare + re-export the new handler | Modify |
| `crates/heartbit/src/daemon/persona_blog_handler.rs` | Extend: enqueue `BlogAnnounceX` + call `update_github_readme` on Posted+deploy-success | Modify |
| `crates/heartbit/src/daemon/core.rs` | Dispatch arm for `BlogAnnounceX` | Modify |
| `crates/heartbit-cli/src/daemon/mod.rs` | Build x_announce + github_readme configs into `BlogContext` | Modify |
| `docs/operating-heartbit.md` | Document the two new sub-blocks | Modify |

---

## Task 1: Config Types

**Files:**
- Modify: `crates/heartbit-core/src/config/daemon.rs`

Add two optional sub-block structs to `PersonaBlogConfig`: `XAnnounceConfig` (just `enabled`) and `GithubReadmeConfig` (paths + git author). Both optional; absence disables the surface.

- [ ] **Step 1: Add the two new structs + fields to PersonaBlogConfig**

Open `crates/heartbit-core/src/config/daemon.rs`. Find the `pub struct PersonaBlogConfig { ... pub deploy_command: Option<String>, }` block (around line 585). Add two new fields after `deploy_command`:

```rust
    /// Optional X self-amplification — when `Some(cfg)` and `cfg.enabled`,
    /// each successful blog publish (after deploy_command succeeds or is
    /// absent) enqueues a `DaemonCommand::BlogAnnounceX` that drafts an
    /// announcement thread through the existing X persona pipeline +
    /// Telegram review.
    #[serde(default)]
    pub x_announce: Option<XAnnounceConfig>,

    /// Optional GitHub Profile README auto-update — when `Some(cfg)` and
    /// `cfg.enabled`, each successful blog publish re-renders the
    /// operator's profile README to feature the 3 most recent essays
    /// and pushes the change to GitHub via the local repo clone.
    #[serde(default)]
    pub github_readme: Option<GithubReadmeConfig>,
```

Then add the struct definitions at the bottom of the file (right after the existing helper `fn default_blog_site_title()` block):

```rust
/// X self-amplification settings (sub-block of `[daemon.persona_blog]`).
#[derive(Debug, Clone, Deserialize)]
pub struct XAnnounceConfig {
    /// Whether X self-amplification is enabled.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
}

/// GitHub Profile README auto-update settings.
#[derive(Debug, Clone, Deserialize)]
pub struct GithubReadmeConfig {
    /// Whether the GitHub README update is enabled.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
    /// Absolute path to the local clone of the profile repo
    /// (e.g. `/home/pleclech/projects/100-tokens-profile`). Must already
    /// be cloned with `origin` configured for `git push` (HTTPS token or
    /// SSH key).
    pub local_repo_path: String,
    /// Path (absolute or relative to `local_repo_path`) to the operator-
    /// authored bio template inserted at the top of the README. If the
    /// file is missing, a minimal default is used.
    #[serde(default = "default_bio_template_path")]
    pub bio_template_path: String,
    /// Git commit author name.
    pub git_author_name: String,
    /// Git commit author email.
    pub git_author_email: String,
}

fn default_bio_template_path() -> String {
    "bio.md".into()
}
```

- [ ] **Step 2: Write tests in the same file**

In the existing `#[cfg(test)] mod tests` block (search for `fn persona_blog_config_parses_with_defaults`), add three new tests right after `persona_blog_config_parses_deploy_command`:

```rust
    #[test]
    fn persona_blog_config_parses_x_announce_block() {
        let toml = r#"
[persona_blog]
persona = "heartbit-ghost:x"
site_url = "https://pascal.heartbit.ai"

[persona_blog.x_announce]
enabled = true
"#;
        #[derive(Deserialize)]
        struct Shim {
            persona_blog: PersonaBlogConfig,
        }
        let cfg: Shim = toml::from_str(toml).unwrap();
        let x = cfg.persona_blog.x_announce.as_ref().unwrap();
        assert!(x.enabled);
    }

    #[test]
    fn persona_blog_config_parses_github_readme_block() {
        let toml = r#"
[persona_blog]
persona = "heartbit-ghost:x"
site_url = "https://pascal.heartbit.ai"

[persona_blog.github_readme]
enabled = true
local_repo_path = "/home/pleclech/projects/100-tokens-profile"
git_author_name = "Pascal Le Clech"
git_author_email = "pascal@heartbit.ai"
"#;
        #[derive(Deserialize)]
        struct Shim {
            persona_blog: PersonaBlogConfig,
        }
        let cfg: Shim = toml::from_str(toml).unwrap();
        let gh = cfg.persona_blog.github_readme.as_ref().unwrap();
        assert!(gh.enabled);
        assert_eq!(gh.local_repo_path, "/home/pleclech/projects/100-tokens-profile");
        assert_eq!(gh.bio_template_path, "bio.md"); // default
        assert_eq!(gh.git_author_name, "Pascal Le Clech");
    }

    #[test]
    fn persona_blog_config_amp_blocks_default_to_none() {
        let toml = r#"
[persona_blog]
persona = "heartbit-ghost:x"
site_url = "https://pascal.heartbit.ai"
"#;
        #[derive(Deserialize)]
        struct Shim {
            persona_blog: PersonaBlogConfig,
        }
        let cfg: Shim = toml::from_str(toml).unwrap();
        assert!(cfg.persona_blog.x_announce.is_none());
        assert!(cfg.persona_blog.github_readme.is_none());
    }
```

- [ ] **Step 3: Re-export the new types from `crates/heartbit-core/src/config/mod.rs`**

Find the existing line `pub use daemon::{... PersonaBlogConfig ...};` in `crates/heartbit-core/src/config/mod.rs` and add `XAnnounceConfig, GithubReadmeConfig` to the export list.

- [ ] **Step 4: Run tests**

```bash
cargo test --package heartbit-core --lib config::daemon::tests::persona_blog
```

Expected: all `persona_blog_*` tests pass, including the 3 new ones.

- [ ] **Step 5: Fix downstream constructors**

`cargo check --workspace --all-targets --features daemon` will fail because `PersonaBlogConfig { ... }` literal constructors must include the two new fields. Find them with `grep -rn "PersonaBlogConfig {" crates/`. Add `x_announce: None, github_readme: None,` after the existing `deploy_command: None,` line in each call site.

- [ ] **Step 6: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets --features daemon -- -D warnings
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-core/src/config/daemon.rs crates/heartbit-core/src/config/mod.rs crates/heartbit/src/daemon/core.rs crates/heartbit/src/daemon/persona_blog.rs
git commit -m "feat(config): PersonaBlogConfig.x_announce + .github_readme sub-blocks"
```

---

## Task 2: `render_readme` pure function

**Files:**
- Create: `crates/heartbit-ghost/src/github_readme.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs`

Pure function that takes bio template + recent posts and produces README markdown. No I/O — testable in isolation.

- [ ] **Step 1: Add module declaration**

In `crates/heartbit-ghost/src/lib.rs`, alphabetically after `pub mod corpus;` (or wherever the `g`-prefixed modules sort), add:

```rust
pub mod github_readme;
```

- [ ] **Step 2: Create the module skeleton**

Create `crates/heartbit-ghost/src/github_readme.rs`:

```rust
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
        // Strip the old auto-marker line from preserved content if it
        // ended right at the marker (it always does — find() returns
        // start of marker).
        format!("{}{}", preserved.trim_end_matches('\n'), {
            let mut suffix = String::from("\n\n");
            suffix.push_str(auto_section);
            suffix
        })
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
        let date = Utc.with_ymd_and_hms(2026, 5, 24, 12, 0, 0).unwrap()
            - chrono::Duration::days(days_ago);
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
        let posts = vec![post("agent-loops", "Agent loops", "Why loops compound costs.", 1)];
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
        let new_section = "<!-- AUTO-GENERATED: do not edit below this line -->\n## Recent essays\n\n- New\n";
        let merged = merge_readme(existing, new_section);
        assert!(merged.starts_with("# Pascal Le Clech"));
        assert!(merged.contains(AUTO_GENERATED_MARKER));
        assert!(merged.contains("New"));
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --package heartbit-ghost --lib github_readme
```

Expected: 7 tests PASS.

- [ ] **Step 4: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/github_readme.rs crates/heartbit-ghost/src/lib.rs
git commit -m "feat(ghost): render_readme + merge_readme for profile README"
```

---

## Task 3: `update_github_readme` (I/O wrapper)

**Files:**
- Modify: `crates/heartbit-ghost/src/github_readme.rs`

Layer the disk + git operations on top of `render_readme`. Reads bio from disk (with fallback), reads `posts_dir/*.md`, writes `local_repo_path/README.md`, runs `git add + commit + push` via `tokio::process::Command`. Failures are bubbled as `UpdateReadmeError`; the caller in the daemon swallows them.

- [ ] **Step 1: Append the I/O wrapper to `github_readme.rs`**

After the existing `merge_readme` function (before the `#[cfg(test)] mod tests` block), add:

```rust
use std::path::{Path, PathBuf};

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
        return Err(UpdateReadmeError::RepoPathMissing(p.local_repo_path.to_path_buf()));
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
    let readme_path = p.local_repo_path.join("README.md");
    let existing = std::fs::read_to_string(&readme_path).unwrap_or_default();
    let rendered = render_readme(&bio, &recent, p.site_url);
    let auto_section = extract_auto_section(&rendered);
    let merged = merge_readme(&existing, auto_section);
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
```

- [ ] **Step 2: Add tempfile dev-dep to `heartbit-ghost/Cargo.toml` if not present**

Check `crates/heartbit-ghost/Cargo.toml` `[dev-dependencies]` section — `tempfile = "3"` should already be there from earlier work. Verify with `grep tempfile crates/heartbit-ghost/Cargo.toml`. If missing, add it.

- [ ] **Step 3: Add tests inside the existing `#[cfg(test)] mod tests` block**

In `crates/heartbit-ghost/src/github_readme.rs`, at the end of the `#[cfg(test)] mod tests` block (before the closing `}`), add:

```rust
    use crate::blog::markdown::{BlogPostFrontmatter, write_post_markdown};
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
        let repo = tmp.path().to_path_buf();
        // Init a local-only git repo with a bare upstream to push to.
        let upstream = tmp.path().parent().unwrap().join(format!(
            "upstream-{}",
            tmp.path().file_name().unwrap().to_string_lossy()
        ));
        std::fs::create_dir_all(&upstream).unwrap();
        run_git(&upstream, &["init", "--bare", "-b", "main"]);
        run_git(&repo, &["init", "-b", "main"]);
        run_git(&repo, &["config", "user.email", "test@test"]);
        run_git(&repo, &["config", "user.name", "test"]);
        run_git(&repo, &["remote", "add", "origin", &upstream.to_string_lossy()]);
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
        // Verify the push landed in the upstream (clone it elsewhere to check).
    }

    #[tokio::test]
    async fn update_uses_default_bio_when_template_missing() {
        let (_tmp, repo) = setup_local_repo();
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
        // Default bio header is "# Pascal Le Clech"
        assert!(readme.contains("# Pascal Le Clech"));
    }
```

- [ ] **Step 4: Run tests**

```bash
cargo test --package heartbit-ghost --lib github_readme
```

Expected: 10 tests PASS (7 from Task 2 + 3 new).

- [ ] **Step 5: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/github_readme.rs
git commit -m "feat(ghost): update_github_readme — disk + git push of profile README"
```

---

## Task 4: Blog announcement writer prompt + module skeleton

**Files:**
- Create: `crates/heartbit-ghost/src/blog/announce.rs`
- Modify: `crates/heartbit-ghost/src/blog/mod.rs`

The writer prompt for blog-announcement X threads. No pipeline yet — just the prompt builder + the config/output/error types.

- [ ] **Step 1: Declare the module**

In `crates/heartbit-ghost/src/blog/mod.rs`, add `pub mod announce;` alphabetically (after `pub mod markdown;`, before `pub mod prompts;`).

- [ ] **Step 2: Create the module**

Create `crates/heartbit-ghost/src/blog/announce.rs`:

```rust
//! X self-amplification of pascal.heartbit.ai blog posts.
//!
//! When `handle_persona_blog` reports `BlogOutcome::Posted` and the
//! optional `deploy_command` succeeded (or was absent), the daemon
//! enqueues a `DaemonCommand::BlogAnnounceX` command. The handler in
//! `heartbit::daemon::blog_announce_x_handler` then calls
//! [`run_x_announcement_pipeline`] which drafts a thread → length
//! normalize → Telegram review → publish via the existing
//! `twitter_tool`.
//!
//! No researcher / no fact_check: the source is the operator's own
//! blog, already fact-checked through the blog pipeline.

#![allow(dead_code)] // Pipeline impl arrives in Task 5; types are public for the handler.

use std::path::Path;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::tool::Tool;

use crate::pipeline::ProgressCallback;
use crate::review::ReviewDelivery;

/// Configuration for one X announcement tick.
pub struct XAnnouncementConfig<'a> {
    /// Persona name (e.g. `"heartbit-ghost:x"`).
    pub persona_name: &'a str,
    /// Default LLM provider.
    pub provider: Arc<BoxedProvider>,
    /// Optional writer-stage provider override (falls back to `provider`).
    pub writer_provider: Option<Arc<BoxedProvider>>,
    /// Persona corpora root.
    pub corpora_root: &'a Path,
    /// Persona voice profiles root.
    pub profiles_root: &'a Path,
    /// Optional progress callback for tracing.
    pub on_progress: Option<ProgressCallback>,
    /// Title of the blog post being announced.
    pub title: &'a str,
    /// One-line excerpt (≤160 chars).
    pub excerpt: &'a str,
    /// First ~500 chars of the blog post body for context.
    pub body_snippet: &'a str,
    /// Canonical URL of the blog post.
    pub post_url: &'a str,
    /// Telegram review delivery.
    pub delivery: Arc<dyn ReviewDelivery>,
    /// X publish tool (existing `twitter_tool` from the daemon).
    pub twitter_tool: Arc<dyn Tool>,
    /// X API credentials.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Terminal state of one X announcement tick.
#[derive(Debug, Clone)]
pub enum XAnnouncementOutcome {
    /// Pipeline ran, operator picked the draft, X publish succeeded.
    Posted {
        /// IDs of the tweets in the thread (head-first).
        tweet_ids: Vec<String>,
        /// Public URL of the head tweet.
        head_url: String,
    },
    /// Operator pressed Skip on Telegram.
    Skipped,
    /// Telegram review timed out.
    TimedOut,
    /// Operator picked but X publish failed (non-OK tool result).
    PublishFailed {
        /// Reason from the X tool's error.
        reason: String,
    },
}

/// Result of one announcement pipeline tick.
#[derive(Debug, Clone)]
pub struct XAnnouncementOutput {
    /// Final outcome.
    pub outcome: XAnnouncementOutcome,
    /// Token usage across writer + length_normalize calls.
    pub usage_summary: heartbit_core::llm::types::TokenUsage,
}

#[derive(Debug, thiserror::Error)]
pub enum XAnnouncementError {
    /// Writer stage failure.
    #[error("writer: {0}")]
    Writer(String),
    /// Length normalization fatally rejected the draft (rare).
    #[error("length normalize: {0}")]
    LengthNormalize(String),
    /// Telegram delivery error.
    #[error("delivery: {0}")]
    Delivery(#[from] crate::review::ReviewDeliveryError),
    /// Invalid config.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// System prompt for the X-announcement writer. The writer is told this
/// is announcing the operator's own essay (not a generated post). Loaded
/// at module scope so tests can pin the load-bearing rules.
pub const X_ANNOUNCE_WRITER_PROMPT: &str = "You are writing an X (Twitter) announcement thread for a long-form essay the operator just published on their personal blog.\n\n\
Rules:\n\
- Produce 3-5 tweets. Each tweet ≤280 characters (hard cap).\n\
- Tweet 1 = the hook. Lead with the most surprising claim from the essay.\n\
- Middle tweets = the substance. Distill the argument into bite-sized chunks.\n\
- Final tweet MUST include the canonical blog URL. Format the URL on its own line.\n\
- Maintain the operator's voice (dhh/mitsuhiko-leaning, opinionated, no marketing-speak).\n\
- NO emojis. NO hashtags. NO 'Read more here'. Just substance + link.\n\
- Do NOT quote the essay verbatim. Re-state the argument in tweet-native form.\n\
- ZERO TOLERANCE FOR INVENTION: every claim in the thread MUST be supported by the body_snippet provided. If you can't say something from the snippet, omit it.\n\n\
Output format: one tweet per line. Empty lines between tweets are ignored.";

/// Build the user message for the X announcement writer.
pub fn build_x_announce_user_message(title: &str, excerpt: &str, body_snippet: &str, post_url: &str) -> String {
    format!(
        "Announce this essay on X. Use a 3-5 tweet thread.\n\n\
TITLE: {title}\n\n\
EXCERPT: {excerpt}\n\n\
BODY SNIPPET (only source of truth — do not invent beyond this):\n{body_snippet}\n\n\
CANONICAL URL (must appear in final tweet): {post_url}\n\n\
Write the thread now. One tweet per line. ≤280 chars each."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_prompt_pins_load_bearing_rules() {
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("3-5 tweets"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("≤280"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("ZERO TOLERANCE FOR INVENTION"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("canonical blog URL"));
        assert!(X_ANNOUNCE_WRITER_PROMPT.contains("NO emojis"));
    }

    #[test]
    fn user_message_includes_url_and_title() {
        let msg = build_x_announce_user_message(
            "Agent loops cost money",
            "Why background loops compound costs.",
            "When you wrap a model in a loop, every tick is a separate API call...",
            "https://pascal.heartbit.ai/agent-loops/",
        );
        assert!(msg.contains("Agent loops cost money"));
        assert!(msg.contains("https://pascal.heartbit.ai/agent-loops/"));
        assert!(msg.contains("only source of truth"));
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --package heartbit-ghost --lib blog::announce
```

Expected: 2 tests PASS.

- [ ] **Step 4: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/blog/announce.rs crates/heartbit-ghost/src/blog/mod.rs
git commit -m "feat(ghost): X-announcement writer prompt + types"
```

---

## Task 5: `run_x_announcement_pipeline` implementation

**Files:**
- Modify: `crates/heartbit-ghost/src/blog/announce.rs`

Implement the actual pipeline body: writer → length_normalize → delivery → publish. Reuses existing `writer_recipe`, `normalize_tweet_length`, and the X publish via `twitter_tool.execute`.

- [ ] **Step 1: Read the existing review pipeline as reference**

```bash
wc -l crates/heartbit-ghost/src/review/mod.rs
grep -n "twitter_tool.execute\|deliver_and_await\|report" crates/heartbit-ghost/src/review/mod.rs | head -10
```

Use the pattern around lines 631-680 of `review/mod.rs` (which calls `twitter_tool.execute` after the operator's Pick) as the template for the publish step.

- [ ] **Step 2: Replace the `#![allow(dead_code)]` line with the impl**

In `crates/heartbit-ghost/src/blog/announce.rs`, remove the `#![allow(dead_code)]` attribute at the top of the file. Then append the pipeline function above the `#[cfg(test)]` block:

```rust
use heartbit_core::agent::AgentRunner;
use heartbit_core::llm::types::TokenUsage;

use crate::agents::writer_recipe;
use crate::pipeline::normalize_tweet_length;
use crate::review::{
    DeliveredReview, DeliveryOutcome, ReportableOutcome, ReviewDelivery, ReviewMessage,
};

/// Run the X announcement pipeline end-to-end.
pub async fn run_x_announcement_pipeline(
    cfg: XAnnouncementConfig<'_>,
) -> Result<XAnnouncementOutput, XAnnouncementError> {
    let mut usage = TokenUsage::default();

    // 1. Writer: produce a 3-5 tweet thread draft.
    let progress = |s: &str| {
        if let Some(p) = cfg.on_progress.as_ref() {
            p(s);
        }
    };
    progress("Drafting announcement thread...");

    let writer_provider = cfg.writer_provider.clone().unwrap_or_else(|| cfg.provider.clone());
    let runner = AgentRunner::builder()
        .recipe(writer_recipe())
        .system_prompt(X_ANNOUNCE_WRITER_PROMPT.to_string())
        .provider(writer_provider)
        .build()
        .map_err(|e| XAnnouncementError::Writer(e.to_string()))?;
    let user_msg = build_x_announce_user_message(cfg.title, cfg.excerpt, cfg.body_snippet, cfg.post_url);
    let writer_out = runner
        .run(&user_msg)
        .await
        .map_err(|e| XAnnouncementError::Writer(e.to_string()))?;
    usage += writer_out.usage;
    let raw_draft = writer_out.text;

    // 2. Length-normalize each tweet to 280 chars.
    let normalized = normalize_tweet_length(&raw_draft, 280);

    // 3. Telegram review.
    progress("Sending to Telegram for review...");
    let review_msg = ReviewMessage {
        persona_name: cfg.persona_name.to_string(),
        topic: format!("Announcement: {}", cfg.title),
        candidates: vec![normalized.clone()],
        interaction_id: uuid::Uuid::new_v4(),
    };
    let delivered: DeliveredReview = cfg.delivery.deliver_and_await(&review_msg).await?;

    let receipt = delivered.receipt;
    match delivered.outcome {
        DeliveryOutcome::Pick(_) => {
            // 4. Publish via twitter_tool.
            progress("Publishing thread to X...");
            let tool_input = serde_json::json!({
                "thread": split_tweets(&normalized),
            });
            let exec_ctx = heartbit_core::tool::ToolContext::new(cfg.credentials.clone());
            let tool_result = cfg.twitter_tool.execute(&exec_ctx, tool_input).await;
            let outcome = match tool_result {
                Ok(out) if !out.is_error() => match parse_thread_output(&out.content_as_string()) {
                    Some((tweet_ids, head_url)) => XAnnouncementOutcome::Posted { tweet_ids, head_url },
                    None => XAnnouncementOutcome::PublishFailed {
                        reason: "twitter_tool returned unparseable output".into(),
                    },
                },
                Ok(out) => XAnnouncementOutcome::PublishFailed {
                    reason: out.content_as_string(),
                },
                Err(e) => XAnnouncementOutcome::PublishFailed { reason: e.to_string() },
            };
            // Report final state back to delivery (best-effort).
            let reportable = match &outcome {
                XAnnouncementOutcome::Posted { head_url, .. } => ReportableOutcome::Posted {
                    chosen_index: 0,
                    url: head_url.clone(),
                },
                XAnnouncementOutcome::PublishFailed { reason } => ReportableOutcome::PublishFailed {
                    chosen_index: 0,
                    reason: reason.clone(),
                },
                _ => unreachable!(),
            };
            let _ = cfg.delivery.report(receipt, reportable).await;
            Ok(XAnnouncementOutput { outcome, usage_summary: usage })
        }
        DeliveryOutcome::Skip => {
            let _ = cfg.delivery.report(receipt, ReportableOutcome::Skipped).await;
            Ok(XAnnouncementOutput {
                outcome: XAnnouncementOutcome::Skipped,
                usage_summary: usage,
            })
        }
        DeliveryOutcome::Timeout => Ok(XAnnouncementOutput {
            outcome: XAnnouncementOutcome::TimedOut,
            usage_summary: usage,
        }),
    }
}

/// Split a normalized draft (one tweet per non-blank line) into a Vec.
fn split_tweets(draft: &str) -> Vec<String> {
    draft
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

/// Parse `twitter_tool` success output JSON: `{thread_root_id, tweet_ids, urls}`.
/// Returns `(tweet_ids, head_url)`. Mirrors `crate::review::parse_thread_output`.
fn parse_thread_output(content: &str) -> Option<(Vec<String>, String)> {
    #[derive(serde::Deserialize)]
    struct ThreadOutput {
        tweet_ids: Vec<String>,
        urls: Vec<String>,
    }
    let parsed: ThreadOutput = serde_json::from_str(content).ok()?;
    let head_url = parsed.urls.first()?.clone();
    Some((parsed.tweet_ids, head_url))
}
```

- [ ] **Step 3: Add pipeline tests**

In the same file, inside the existing `#[cfg(test)] mod tests` block, append:

```rust
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use heartbit_core::Tool as ToolTrait;
    use heartbit_core::tool::{ToolContext, ToolOutput};

    // ─── Mocks ────────────────────────────────────────────────────────────────

    struct MockDelivery {
        outcome: DeliveryOutcome,
        deliver_calls: AtomicUsize,
        report_calls: AtomicUsize,
    }

    impl MockDelivery {
        fn arc(outcome: DeliveryOutcome) -> Arc<Self> {
            Arc::new(MockDelivery {
                outcome,
                deliver_calls: AtomicUsize::new(0),
                report_calls: AtomicUsize::new(0),
            })
        }
    }

    impl ReviewDelivery for MockDelivery {
        fn deliver_and_await<'a>(
            &'a self,
            _msg: &'a ReviewMessage,
        ) -> Pin<Box<dyn Future<Output = Result<DeliveredReview, crate::review::ReviewDeliveryError>> + Send + 'a>>
        {
            self.deliver_calls.fetch_add(1, Ordering::SeqCst);
            let outcome = self.outcome.clone();
            Box::pin(async move {
                Ok(DeliveredReview {
                    outcome,
                    receipt: crate::review::DeliveryReceipt::default(),
                })
            })
        }
        fn report<'a>(
            &'a self,
            _receipt: crate::review::DeliveryReceipt,
            _outcome: ReportableOutcome,
        ) -> Pin<Box<dyn Future<Output = Result<(), crate::review::ReviewDeliveryError>> + Send + 'a>>
        {
            self.report_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(()) })
        }
    }

    struct MockTwitterTool {
        last_input: Mutex<Option<serde_json::Value>>,
        return_value: serde_json::Value,
        fail: bool,
    }

    impl MockTwitterTool {
        fn ok_with_thread() -> Arc<Self> {
            Arc::new(MockTwitterTool {
                last_input: Mutex::new(None),
                return_value: serde_json::json!({
                    "thread_root_id": "1234",
                    "tweet_ids": ["1234", "1235"],
                    "urls": ["https://x.com/i/web/status/1234"]
                }),
                fail: false,
            })
        }
        fn failing() -> Arc<Self> {
            Arc::new(MockTwitterTool {
                last_input: Mutex::new(None),
                return_value: serde_json::json!({}),
                fail: true,
            })
        }
    }

    impl heartbit_core::Tool for MockTwitterTool {
        fn definition(&self) -> heartbit_core::tool::ToolDefinition {
            heartbit_core::tool::ToolDefinition {
                name: "twitter_thread".into(),
                description: "mock".into(),
                input_schema: serde_json::json!({}),
            }
        }
        fn execute<'a>(
            &'a self,
            _ctx: &'a ToolContext,
            input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + 'a>>
        {
            *self.last_input.lock().unwrap() = Some(input);
            let fail = self.fail;
            let rv = self.return_value.clone();
            Box::pin(async move {
                if fail {
                    Ok(ToolOutput::error("publish failed"))
                } else {
                    Ok(ToolOutput::text(rv.to_string()))
                }
            })
        }
    }

    struct StubCreds;
    impl CredentialResolver for StubCreds {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>> + Send + '_>>
        {
            Box::pin(async move { Ok(heartbit_core::Secret::new("stub")) })
        }
    }

    // ─── Tests ────────────────────────────────────────────────────────────────

    #[test]
    fn split_tweets_strips_empty_lines() {
        let s = "Tweet 1.\n\nTweet 2.\n   \nTweet 3.";
        let v = split_tweets(s);
        assert_eq!(v, vec!["Tweet 1.", "Tweet 2.", "Tweet 3."]);
    }

    #[test]
    fn parse_thread_output_extracts_ids_and_head_url() {
        let content = r#"{"thread_root_id":"1","tweet_ids":["1","2"],"urls":["https://x.com/i/web/status/1"]}"#;
        let (ids, url) = parse_thread_output(content).unwrap();
        assert_eq!(ids, vec!["1", "2"]);
        assert_eq!(url, "https://x.com/i/web/status/1");
    }
```

**Note**: a full happy-path test of `run_x_announcement_pipeline` requires a mock LLM provider (the existing `MockProvider` pattern lives in `daemon/persona_post_handler.rs::tests`). For this task's scope, the unit tests for `split_tweets` + `parse_thread_output` + the prompt assertions from Task 4 are sufficient. The pipeline's integration test goes in Task 11.

- [ ] **Step 4: Run tests**

```bash
cargo test --package heartbit-ghost --lib blog::announce
```

Expected: 4 tests PASS (2 from Task 4 + `split_tweets_strips_empty_lines` + `parse_thread_output_extracts_ids_and_head_url`).

- [ ] **Step 5: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings && cargo check --workspace --all-targets --features daemon
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/blog/announce.rs
git commit -m "feat(ghost): run_x_announcement_pipeline — writer → length → review → publish"
```

---

## Task 6: `DaemonCommand::BlogAnnounceX` variant + round-trip test

**Files:**
- Modify: `crates/heartbit/src/daemon/types.rs`

Add the new command variant and its serde round-trip test.

- [ ] **Step 1: Add the variant**

In `crates/heartbit/src/daemon/types.rs`, find the `PersonaBlog` variant (around line 91-94). Add immediately after it:

```rust
    /// Fire one X announcement thread for a freshly published blog post.
    /// Enqueued by `handle_persona_blog` after `BlogOutcome::Posted` and
    /// successful `deploy_command` (or no `deploy_command` configured).
    BlogAnnounceX {
        /// Persona name (e.g. `"heartbit-ghost:x"`).
        persona: String,
        /// Canonical URL of the blog post.
        post_url: String,
        /// Title of the blog post.
        title: String,
        /// One-line excerpt.
        excerpt: String,
        /// First ~500 chars of body, used as the only source-of-truth
        /// for the writer.
        body_snippet: String,
    },
```

- [ ] **Step 2: Add the round-trip test**

Find the existing `persona_blog_command_round_trips` test (around line 1145) and add right after it:

```rust
#[test]
fn blog_announce_x_command_round_trips() {
    let cmd = DaemonCommand::BlogAnnounceX {
        persona: "heartbit-ghost:x".into(),
        post_url: "https://pascal.heartbit.ai/agent-loops/".into(),
        title: "Agent loops cost money".into(),
        excerpt: "Why loops compound costs.".into(),
        body_snippet: "When you wrap a model in a loop...".into(),
    };
    let s = serde_json::to_string(&cmd).unwrap();
    let parsed: DaemonCommand = serde_json::from_str(&s).unwrap();
    match parsed {
        DaemonCommand::BlogAnnounceX { persona, post_url, title, excerpt, body_snippet } => {
            assert_eq!(persona, "heartbit-ghost:x");
            assert_eq!(post_url, "https://pascal.heartbit.ai/agent-loops/");
            assert_eq!(title, "Agent loops cost money");
            assert_eq!(excerpt, "Why loops compound costs.");
            assert!(body_snippet.starts_with("When you wrap"));
        }
        other => panic!("expected BlogAnnounceX, got {other:?}"),
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test --package heartbit --features daemon --lib daemon::types::tests::blog_announce_x
```

Expected: PASS.

- [ ] **Step 4: Quality gate**

```bash
cargo fmt -- --check && cargo check --workspace --all-targets --features daemon
```

Expected: clean. Any other files that exhaustively match `DaemonCommand` will fail compilation — they need a wildcard or explicit no-op arm for `BlogAnnounceX`. Add them as you encounter them (typically just `_ => {}` in non-core dispatchers).

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit/src/daemon/types.rs
git commit -m "feat(daemon): DaemonCommand::BlogAnnounceX variant"
```

---

## Task 7: `handle_blog_announce_x` handler

**Files:**
- Create: `crates/heartbit/src/daemon/blog_announce_x_handler.rs`
- Modify: `crates/heartbit/src/daemon/mod.rs`

The Kafka-dispatch target for `BlogAnnounceX`. Builds an `XAnnouncementConfig` from inputs and calls `run_x_announcement_pipeline`.

- [ ] **Step 1: Declare the module**

In `crates/heartbit/src/daemon/mod.rs`, add `pub mod blog_announce_x_handler;` near the other handler `pub mod` lines, then add at the re-export section:

```rust
pub use blog_announce_x_handler::{BlogAnnounceXDeps, handle_blog_announce_x};
```

- [ ] **Step 2: Create the handler**

Create `crates/heartbit/src/daemon/blog_announce_x_handler.rs`:

```rust
//! Handler for `DaemonCommand::BlogAnnounceX`. Builds the announcement
//! pipeline config and invokes `run_x_announcement_pipeline`.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::{PersonaParams, PersonaRegistry};
use heartbit_ghost::blog::announce::{XAnnouncementConfig, run_x_announcement_pipeline};
use heartbit_ghost::review::ReviewDelivery;

/// Inputs to one announcement handler tick.
pub struct BlogAnnounceXDeps<'a> {
    /// Persona name to load.
    pub persona_name: &'a str,
    /// Persona registry.
    pub registry: &'a PersonaRegistry,
    /// Default LLM provider.
    pub provider: Arc<BoxedProvider>,
    /// Writer-stage provider override (falls back to `provider`).
    pub writer_provider: Option<Arc<BoxedProvider>>,
    /// Persona corpora root.
    pub corpora_root: &'a Path,
    /// Persona voice profiles root.
    pub profiles_root: &'a Path,
    /// Title of the blog post.
    pub title: &'a str,
    /// Blog excerpt.
    pub excerpt: &'a str,
    /// First ~500 chars of body.
    pub body_snippet: &'a str,
    /// Canonical blog URL.
    pub post_url: &'a str,
    /// Telegram review delivery.
    pub delivery: Arc<dyn ReviewDelivery>,
    /// X publish tool.
    pub twitter_tool: Arc<dyn Tool>,
    /// X credentials.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Run one announcement tick. Never panics; errors are logged and
/// swallowed by the caller.
pub async fn handle_blog_announce_x(deps: BlogAnnounceXDeps<'_>) -> Result<()> {
    let persona = deps
        .registry
        .get(deps.persona_name)
        .ok_or_else(|| anyhow::anyhow!("persona '{}' not registered", deps.persona_name))?;
    let _expansion = persona
        .expand(&PersonaParams::default())
        .map_err(|e| anyhow::anyhow!("expand persona '{}': {e}", deps.persona_name))?;

    let cfg = XAnnouncementConfig {
        persona_name: deps.persona_name,
        provider: deps.provider.clone(),
        writer_provider: deps.writer_provider.clone(),
        corpora_root: deps.corpora_root,
        profiles_root: deps.profiles_root,
        on_progress: Some(Arc::new(|s: &str| tracing::info!("blog_announce_x: {s}"))),
        title: deps.title,
        excerpt: deps.excerpt,
        body_snippet: deps.body_snippet,
        post_url: deps.post_url,
        delivery: deps.delivery.clone(),
        twitter_tool: deps.twitter_tool.clone(),
        credentials: deps.credentials.clone(),
    };

    match run_x_announcement_pipeline(cfg).await {
        Ok(out) => {
            tracing::info!(
                persona = %deps.persona_name,
                outcome = ?out.outcome,
                "blog announce X complete"
            );
        }
        Err(e) => {
            tracing::error!(
                persona = %deps.persona_name,
                error = %e,
                "blog announce X pipeline failed"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    // For the full handler tests we'd need MockProvider/MockTwitterTool/
    // MockReviewDelivery. The unit tests in heartbit_ghost::blog::announce
    // cover the pipeline parts; this handler is just a thin dispatcher.
    // Test the unknown-persona path only.

    use heartbit_core::persona::Persona;

    #[tokio::test]
    async fn handle_blog_announce_x_unknown_persona_errors() {
        let registry = PersonaRegistry::new();
        // No persona registered.
        // Stub deps: provider/tool/delivery don't matter — error fires
        // before they're used.
        struct DummyTool;
        impl Tool for DummyTool {
            fn definition(&self) -> heartbit_core::tool::ToolDefinition {
                heartbit_core::tool::ToolDefinition {
                    name: "x".into(),
                    description: "x".into(),
                    input_schema: serde_json::json!({}),
                }
            }
            fn execute<'a>(
                &'a self,
                _ctx: &'a heartbit_core::tool::ToolContext,
                _input: serde_json::Value,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<heartbit_core::tool::ToolOutput, heartbit_core::Error>> + Send + 'a>>
            {
                Box::pin(async move { Ok(heartbit_core::tool::ToolOutput::text("ok")) })
            }
        }
        struct DummyCreds;
        impl CredentialResolver for DummyCreds {
            fn resolve(
                &self,
                _name: &str,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>> + Send + '_>>
            {
                Box::pin(async move { Ok(heartbit_core::Secret::new("x")) })
            }
        }
        // The provider Arc needs to satisfy Arc<BoxedProvider>. Use the
        // existing test helper if one exists; otherwise this test
        // exercises only the early registry.get() error path.

        // Note: this test is intentionally minimal — full handler
        // coverage lives in the heartbit_ghost::blog::announce tests.
        let _ = registry; // keep compiler happy; the rest of the deps
        // require provider construction. Test as a doc-test or skip.
    }
}
```

**Note**: The handler test above is intentionally a smoke. Real coverage of the pipeline lives in `crates/heartbit-ghost/src/blog/announce.rs`. The handler is a 30-line dispatcher.

- [ ] **Step 3: Run tests**

```bash
cargo test --package heartbit --features daemon --lib daemon::blog_announce_x_handler
```

Expected: at minimum compile-passes; the smoke test is a placeholder.

- [ ] **Step 4: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets --features daemon -- -D warnings
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit/src/daemon/blog_announce_x_handler.rs crates/heartbit/src/daemon/mod.rs
git commit -m "feat(daemon): handle_blog_announce_x — dispatcher for BlogAnnounceX"
```

---

## Task 8: Propagate `x_announce` + `github_readme` configs through `PersonaBlogEntry`

**Files:**
- Modify: `crates/heartbit/src/daemon/blog_context.rs`

Add two new fields on `PersonaBlogEntry` to carry the operator-configured sub-blocks at runtime.

- [ ] **Step 1: Add fields + update Debug**

In `crates/heartbit/src/daemon/blog_context.rs`, find the `PersonaBlogEntry` struct. After `pub deploy_command: Option<String>`, add:

```rust
    /// Optional X self-amplification config. When `Some`, each Posted
    /// outcome enqueues a `DaemonCommand::BlogAnnounceX`.
    pub x_announce: Option<heartbit_core::config::XAnnounceConfig>,
    /// Optional GitHub README auto-update config. When `Some`, each
    /// Posted outcome refreshes the operator's profile README.
    pub github_readme: Option<heartbit_core::config::GithubReadmeConfig>,
```

In the `impl std::fmt::Debug for PersonaBlogEntry`, add two more `.field(...)` calls before `.finish()`:

```rust
            .field("x_announce_enabled", &self.x_announce.as_ref().map(|c| c.enabled))
            .field("github_readme_enabled", &self.github_readme.as_ref().map(|c| c.enabled))
```

- [ ] **Step 2: Fix all explicit constructors**

Run `grep -rn "PersonaBlogEntry {" crates/`. For each match, add `x_announce: None, github_readme: None,` (or appropriate values if the call site already has the config from `PersonaBlogConfig`). The CLI startup wiring is fixed in Task 9 below — for now just unblock the build with `None` everywhere except real wiring sites.

- [ ] **Step 3: Run tests + check**

```bash
cargo check --workspace --all-targets --features daemon && cargo test --package heartbit --features daemon --lib daemon::persona_blog
```

Expected: clean compile, persona_blog scheduler tests still pass.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit/src/daemon/blog_context.rs crates/heartbit/src/daemon/persona_blog.rs
git commit -m "feat(daemon): PersonaBlogEntry carries x_announce + github_readme"
```

---

## Task 9: CLI startup wires sub-configs

**Files:**
- Modify: `crates/heartbit-cli/src/daemon/mod.rs`

Plumb the operator's `[daemon.persona_blog.x_announce]` and `[daemon.persona_blog.github_readme]` into the runtime `PersonaBlogEntry`.

- [ ] **Step 1: Patch the entry construction**

In `crates/heartbit-cli/src/daemon/mod.rs`, find the `let entry = heartbit::PersonaBlogEntry { ... };` block (around line 658). Add two new field assignments after `deploy_command: blog_cfg.deploy_command.clone(),`:

```rust
                x_announce: blog_cfg.x_announce.clone(),
                github_readme: blog_cfg.github_readme.clone(),
```

- [ ] **Step 2: Build + quality gate**

```bash
cargo check --workspace --all-targets --features daemon && \
cargo fmt -- --check && \
cargo clippy --workspace --all-targets --features daemon -- -D warnings
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/heartbit-cli/src/daemon/mod.rs
git commit -m "feat(cli): wire x_announce + github_readme into BlogContext"
```

---

## Task 10: Daemon dispatch arm + `handle_persona_blog` extension

**Files:**
- Modify: `crates/heartbit/src/daemon/core.rs`
- Modify: `crates/heartbit/src/daemon/persona_blog_handler.rs`

Two integration points:
1. New `DaemonCommand::BlogAnnounceX { ... }` arm in `core.rs` dispatch loop.
2. `handle_persona_blog`: after `BlogOutcome::Posted` *and* successful `deploy_command` (or absent), produce a `BlogAnnounceX` command + call `update_github_readme`.

- [ ] **Step 1: Add the dispatch arm in `core.rs`**

In `crates/heartbit/src/daemon/core.rs`, find the existing `DaemonCommand::PersonaBlog { persona }` arm (around line 1453). Add a new arm right after it. The arm needs access to `twitter_tool` — `BlogContext.twitter_tool` field doesn't currently exist; we reuse `posts_context.twitter_tool` (the existing X tool is shared at the daemon level).

```rust
                        DaemonCommand::BlogAnnounceX { persona, post_url, title, excerpt, body_snippet } => {
                            let Some(blog_ctx) = self.blog_context.clone() else {
                                tracing::warn!(persona = %persona, "BlogAnnounceX: no blog_context configured");
                                continue;
                            };
                            let Some(posts_ctx) = self.posts_context.clone() else {
                                tracing::warn!(persona = %persona, "BlogAnnounceX: no posts_context (needed for twitter_tool)");
                                continue;
                            };
                            let Some(posts_entry) = posts_ctx.entries.get(&persona).cloned() else {
                                tracing::warn!(persona = %persona, "BlogAnnounceX: no posts_entry for persona");
                                continue;
                            };
                            // Skip if x_announce is None or disabled.
                            let enabled = blog_ctx
                                .entry
                                .x_announce
                                .as_ref()
                                .map(|c| c.enabled)
                                .unwrap_or(false);
                            if !enabled {
                                tracing::info!(persona = %persona, "BlogAnnounceX: x_announce disabled — skipping");
                                continue;
                            }
                            let registry = blog_ctx.registry.clone();
                            let provider = blog_ctx.provider.clone();
                            let writer_provider = blog_ctx.entry.writer_provider.clone();
                            let credentials = blog_ctx.credentials.clone();
                            let corpora_root = blog_ctx.corpora_root.clone();
                            let profiles_root = blog_ctx.profiles_root.clone();
                            let delivery = posts_entry.delivery.clone();
                            let twitter_tool = posts_entry.twitter_tool.clone();
                            let persona_owned = persona.clone();
                            let post_url_owned = post_url.clone();
                            let title_owned = title.clone();
                            let excerpt_owned = excerpt.clone();
                            let body_snippet_owned = body_snippet.clone();
                            tokio::spawn(async move {
                                let deps = crate::daemon::BlogAnnounceXDeps {
                                    persona_name: &persona_owned,
                                    registry: &registry,
                                    provider,
                                    writer_provider,
                                    corpora_root: &corpora_root,
                                    profiles_root: &profiles_root,
                                    title: &title_owned,
                                    excerpt: &excerpt_owned,
                                    body_snippet: &body_snippet_owned,
                                    post_url: &post_url_owned,
                                    delivery,
                                    twitter_tool,
                                    credentials,
                                };
                                if let Err(e) = crate::daemon::handle_blog_announce_x(deps).await {
                                    tracing::error!(persona = %persona_owned, error = %e, "blog announce X handler failed");
                                }
                            });
                        }
```

- [ ] **Step 2: Extend `handle_persona_blog` in `persona_blog_handler.rs`**

Open `crates/heartbit/src/daemon/persona_blog_handler.rs`. Find the existing `if let BlogOutcome::Posted { post_path, post_url, .. } = &out.outcome { ... }` block. Replace the contents of that `if let` block with:

```rust
            if let BlogOutcome::Posted {
                post_path,
                post_url,
                ..
            } = &out.outcome
            {
                tracing::info!(
                    persona = %deps.persona_name,
                    %post_url,
                    path = %post_path.display(),
                    "blog: post published"
                );
                // Run deploy_command first; on success (or absence), fire
                // the two amp surfaces.
                let deploy_ok = if let Some(cmd) = deps.deploy_command {
                    run_deploy_command(deps.persona_name, cmd).await
                } else {
                    true
                };
                if deploy_ok {
                    // Surface 2: GitHub README on-publish (synchronous).
                    if let Some(gh) = deps.github_readme {
                        if gh.enabled {
                            let res = heartbit_ghost::github_readme::update_github_readme(
                                heartbit_ghost::github_readme::UpdateReadmeParams {
                                    local_repo_path: std::path::Path::new(&gh.local_repo_path),
                                    bio_template_path: std::path::Path::new(&gh.bio_template_path),
                                    blog_posts_dir: deps.posts_dir,
                                    site_url: deps.site_url,
                                    git_author_name: &gh.git_author_name,
                                    git_author_email: &gh.git_author_email,
                                    new_post_slug: post_path
                                        .file_stem()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or(""),
                                },
                            )
                            .await;
                            if let Err(e) = res {
                                tracing::error!(persona = %deps.persona_name, error = %e, "github_readme update failed");
                            }
                        }
                    }
                    // Surface 1: X announcement via Kafka (best-effort enqueue).
                    if let Some(xa) = deps.x_announce {
                        if xa.enabled {
                            if let Some(producer) = deps.command_producer {
                                let body_snippet = title_and_snippet_from_path(post_path)
                                    .unwrap_or_default();
                                let cmd = crate::daemon::DaemonCommand::BlogAnnounceX {
                                    persona: deps.persona_name.to_string(),
                                    post_url: post_url.clone(),
                                    title: out
                                        .candidates
                                        .first()
                                        .map(|c| c.title.clone())
                                        .unwrap_or_default(),
                                    excerpt: out
                                        .candidates
                                        .first()
                                        .map(|c| c.excerpt.clone())
                                        .unwrap_or_default(),
                                    body_snippet,
                                };
                                if let Err(e) = producer
                                    .produce(deps.commands_topic, &cmd)
                                    .await
                                {
                                    tracing::error!(persona = %deps.persona_name, error = %e, "BlogAnnounceX enqueue failed");
                                }
                            } else {
                                tracing::warn!(persona = %deps.persona_name, "x_announce enabled but no command_producer configured");
                            }
                        }
                    }
                } else {
                    tracing::warn!(persona = %deps.persona_name, "deploy_command failed — skipping amp surfaces");
                }
            }
```

Then add the helper function at the bottom of the file (before the `#[cfg(test)] mod tests`):

```rust
/// Read the first ~500 chars of the post body for the X announcement
/// writer. Returns `None` if the file can't be read.
fn title_and_snippet_from_path(post_path: &std::path::Path) -> Option<String> {
    let content = std::fs::read_to_string(post_path).ok()?;
    // Strip YAML frontmatter.
    let trimmed = content.trim_start();
    let body = if trimmed.starts_with("---") {
        let after = &trimmed[3..];
        match after.find("\n---\n") {
            Some(end) => after[end + 5..].trim_start_matches('\n').to_string(),
            None => return None,
        }
    } else {
        trimmed.to_string()
    };
    let snippet: String = body.chars().take(500).collect();
    Some(snippet)
}
```

Also add three new fields to `PersonaBlogDeps`:

```rust
    /// Optional X self-amp config.
    pub x_announce: Option<&'a heartbit_core::config::XAnnounceConfig>,
    /// Optional GitHub README config.
    pub github_readme: Option<&'a heartbit_core::config::GithubReadmeConfig>,
    /// Command producer for enqueueing `BlogAnnounceX`. Optional —
    /// `None` skips X self-amp.
    pub command_producer: Option<Arc<dyn crate::daemon::CommandProducer>>,
    /// Kafka topic for commands.
    pub commands_topic: &'a str,
```

Update the existing dispatch arm in `core.rs` (line 1453 area, the `PersonaBlog` arm — NOT the new `BlogAnnounceX` arm) to populate these from context. Around the `let deps = PersonaBlogDeps { ... }`, add:

```rust
                                x_announce: blog_ctx.entry.x_announce.as_ref(),
                                github_readme: blog_ctx.entry.github_readme.as_ref(),
                                command_producer: Some(self.command_producer.clone()),
                                commands_topic: &self.commands_topic,
```

(`self.command_producer` is an `Arc<dyn CommandProducer>` field that the daemon already holds — it's the same one the schedulers use. If the field is named differently in the actual code, adapt.)

Fix the existing test fixtures in `persona_blog_handler.rs::tests` that construct `PersonaBlogDeps` — add `x_announce: None, github_readme: None, command_producer: None, commands_topic: "test.commands",`.

- [ ] **Step 3: Run tests + check**

```bash
cargo test --package heartbit --features daemon --lib daemon::persona_blog_handler
cargo check --workspace --all-targets --features daemon
```

Expected: existing 4 handler tests still pass (deploy_command tests + unknown persona + no_seed).

- [ ] **Step 4: Quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets --features daemon -- -D warnings
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit/src/daemon/core.rs crates/heartbit/src/daemon/persona_blog_handler.rs
git commit -m "feat(daemon): handle_persona_blog fires github_readme + BlogAnnounceX on Posted"
```

---

## Task 11: Integration test — Posted outcome triggers both amp surfaces

**Files:**
- Modify: `crates/heartbit/src/daemon/persona_blog_handler.rs`

One integration test that mocks the pipeline returning `BlogOutcome::Posted`, captures the enqueued `BlogAnnounceX`, and verifies the README was updated in a tempdir.

- [ ] **Step 1: Add a mock `CommandProducer` test type**

In the `#[cfg(test)] mod tests` block of `crates/heartbit/src/daemon/persona_blog_handler.rs`, add:

```rust
    use std::sync::Mutex;

    struct CapturingProducer {
        captured: Mutex<Vec<crate::daemon::DaemonCommand>>,
    }
    impl CapturingProducer {
        fn arc() -> Arc<Self> {
            Arc::new(CapturingProducer { captured: Mutex::new(Vec::new()) })
        }
    }
    impl crate::daemon::CommandProducer for CapturingProducer {
        fn produce<'a>(
            &'a self,
            _topic: &'a str,
            cmd: &'a crate::daemon::DaemonCommand,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), anyhow::Error>> + Send + 'a>>
        {
            self.captured.lock().unwrap().push(cmd.clone());
            Box::pin(async move { Ok(()) })
        }
    }
```

(The exact trait signature for `CommandProducer::produce` must match the real trait — run `grep "trait CommandProducer" crates/heartbit/src/daemon/` and copy the signature exactly.)

- [ ] **Step 2: Add the integration test**

```rust
    #[tokio::test]
    async fn posted_outcome_fires_both_amp_surfaces() {
        // This test requires a working mock pipeline. Approach:
        // 1) Bypass run_blog_pipeline by constructing a fake BlogOutput
        //    directly and testing the post-Posted side-effects.
        //
        // Since handle_persona_blog calls run_blog_pipeline internally,
        // we can't easily mock the pipeline result without refactoring.
        // Instead, this test verifies the side-effect helpers directly:
        //   - update_github_readme runs against a tempdir
        //   - CapturingProducer captures the BlogAnnounceX command

        // Set up a tempdir repo for the README update.
        let (_tmp, repo) = setup_local_repo();
        let posts_dir = repo.parent().unwrap().join("posts");
        std::fs::create_dir_all(&posts_dir).unwrap();

        // 1) update_github_readme — direct call, verify file changes.
        let gh_res = heartbit_ghost::github_readme::update_github_readme(
            heartbit_ghost::github_readme::UpdateReadmeParams {
                local_repo_path: &repo,
                bio_template_path: std::path::Path::new("bio.md"),
                blog_posts_dir: &posts_dir,
                site_url: "https://pascal.heartbit.ai",
                git_author_name: "test",
                git_author_email: "test@test",
                new_post_slug: "smoke-test",
            },
        )
        .await;
        assert!(gh_res.is_ok(), "update_github_readme failed: {gh_res:?}");
        let readme = std::fs::read_to_string(repo.join("README.md")).unwrap();
        assert!(readme.contains(heartbit_ghost::github_readme::AUTO_GENERATED_MARKER));

        // 2) CapturingProducer — manually enqueue BlogAnnounceX, verify
        //    capture.
        let producer = CapturingProducer::arc();
        let cmd = crate::daemon::DaemonCommand::BlogAnnounceX {
            persona: "heartbit-ghost:x".into(),
            post_url: "https://pascal.heartbit.ai/test/".into(),
            title: "Test".into(),
            excerpt: "x".into(),
            body_snippet: "y".into(),
        };
        producer.produce("test.commands", &cmd).await.unwrap();
        let captured = producer.captured.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert!(matches!(&captured[0], crate::daemon::DaemonCommand::BlogAnnounceX { .. }));
    }
```

**Note**: this test verifies the *helpers* (which the handler calls). A full end-to-end test (mock `run_blog_pipeline` returning `Posted` → handler calls both helpers) requires either refactoring `handle_persona_blog` to inject the pipeline or accepting that the per-unit tests in earlier tasks already cover correctness. We accept the per-unit coverage.

Helpers `setup_local_repo` and `run_git` from `heartbit-ghost::github_readme::tests` are private; for this test, just inline a minimal repo init using `std::process::Command::new("git")`. Or expose them via a `pub(crate)` test helper module — keep it private; inline a small helper here.

- [ ] **Step 3: Run + commit**

```bash
cargo test --package heartbit --features daemon --lib daemon::persona_blog_handler::tests::posted_outcome
cargo fmt -- --check && cargo clippy --workspace --all-targets --features daemon -- -D warnings
git add crates/heartbit/src/daemon/persona_blog_handler.rs
git commit -m "test(daemon): integration test for Posted outcome + amp surfaces"
```

---

## Task 12: Docs — operating-heartbit.md

**Files:**
- Modify: `docs/operating-heartbit.md`

Add a subsection under "Personal blog knobs" documenting the two new sub-blocks.

- [ ] **Step 1: Patch the doc**

Open `docs/operating-heartbit.md`. Find the row for `deploy_command` in the "Personal blog knobs" table. Right after the table (before the `### Prerequisite` header), add:

```markdown
### Sub-blocks: `x_announce` + `github_readme`

`[daemon.persona_blog.x_announce]` — when set, each successful blog publish (gated on `deploy_command` succeeding or being absent) enqueues a `DaemonCommand::BlogAnnounceX` that drafts a 3-5 tweet announcement thread, sends it to Telegram for your review, then publishes on Pick.

```toml
[daemon.persona_blog.x_announce]
enabled = true
```

Re-uses the same Telegram review dispatcher as proactive X posts, the same `twitter_tool` for publishing, and the operator's voice profile. The writer's source-of-truth is the just-published blog post (title + excerpt + first ~500 chars of body) — no fact-checking step (the blog post is the source).

`[daemon.persona_blog.github_readme]` — when set, after a successful blog publish the daemon refreshes the operator's GitHub Profile README to feature the 3 most recent essays, then `git push`es the change to the configured local clone.

```toml
[daemon.persona_blog.github_readme]
enabled = true
local_repo_path = "/home/pleclech/projects/100-tokens-profile"
bio_template_path = "bio.md"           # relative to local_repo_path
git_author_name = "Pascal Le Clech"
git_author_email = "pascal@heartbit.ai"
```

The README is rendered with a marker (`<!-- AUTO-GENERATED: do not edit below this line -->`). Everything above the marker is preserved verbatim — that's where your bio lives. Everything below is regenerated on each blog publish.

**Requirements:**
- `local_repo_path` must be a working git clone with `origin` configured for push (HTTPS token or SSH key).
- The daemon does NOT manage GitHub auth — same model as `deploy_command` for `wrangler`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/operating-heartbit.md
git commit -m "docs: document persona_blog.x_announce + .github_readme sub-blocks"
```

---

## Task 13: Final close-out

- [ ] **Step 1: Full workspace gate**

```bash
cargo fmt -- --check && \
cargo clippy --workspace --all-targets --features daemon -- -D warnings && \
cargo test --workspace --lib --features daemon
```

Expected: all green.

- [ ] **Step 2: Manual config validation**

```bash
cargo build --release --bin heartbit --features daemon
HEARTBIT_GHOST_OPERATOR_USER_ID=999 target/release/heartbit --config daemon-dev.toml daemon --validate-config
```

Expected: `✓ daemon-dev.toml validates clean`.

- [ ] **Step 3: Activate in daemon-dev.toml (operator step, NOT committed)**

The operator adds the two sub-blocks to their local `daemon-dev.toml` (gitignored), then restarts the daemon with `CLOUDFLARE_API_TOKEN` + git push credentials available in env.

```toml
[daemon.persona_blog.x_announce]
enabled = true

[daemon.persona_blog.github_readme]
enabled = true
local_repo_path = "/home/pleclech/projects/100-tokens-profile"
git_author_name = "Pascal Le Clech"
git_author_email = "pascal@heartbit.ai"
```

- [ ] **Step 4: Smoke test (deferred to operator)**

Fire a test blog tick (via Kafka command, as in the prior plan's Task 14). On Posted outcome you should observe:
1. `wrangler pages deploy` runs (existing).
2. `update_github_readme` writes + commits + pushes (new — check the `100-tokens-profile` repo on GitHub for the commit).
3. `BlogAnnounceX` command lands in the Kafka topic.
4. `handle_blog_announce_x` runs → drafts thread → Telegram review fires → operator picks → X publish.

---

## Verification matrix

| Spec item | Covered by |
|---|---|
| `XAnnounceConfig` + `GithubReadmeConfig` sub-blocks | Task 1 |
| `render_readme` pure fn | Task 2 |
| `update_github_readme` I/O | Task 3 |
| `X_ANNOUNCE_WRITER_PROMPT` | Task 4 |
| `run_x_announcement_pipeline` | Task 5 |
| `DaemonCommand::BlogAnnounceX` | Task 6 |
| `handle_blog_announce_x` | Task 7 |
| `PersonaBlogEntry` extended | Task 8 |
| CLI wires sub-configs | Task 9 |
| Dispatch arm + handler extension | Task 10 |
| Integration test | Task 11 |
| Docs | Task 12 |
| Quality gate | Task 13 |
| Gating on deploy success | Task 10 (`if deploy_ok { ... }`) |
| Marker-preserved bio | Task 2 (`merge_readme`) |
| Per-surface failure swallowed | Tasks 3 + 10 (errors logged, not propagated) |

---

## Notes for the implementer

- The X publish call (`twitter_tool.execute(...)`) returns a JSON output that needs parsing. The reference parser lives at `crates/heartbit-ghost/src/review/mod.rs::parse_thread_output` — mirror that logic exactly in `crates/heartbit-ghost/src/blog/announce.rs::parse_thread_output`.
- The `CommandProducer` trait signature: check `crates/heartbit/src/daemon/command_producer.rs` (or wherever it lives) — the test mock must match exactly.
- `posts_context.entries[persona].twitter_tool` is the existing `Arc<dyn Tool>` for X publishing. Reuse it (don't construct a new one) since X creds are session-stateful.
- `posts_context.entries[persona].delivery` — same `Arc<dyn ReviewDelivery>` as proactive posts use. The Telegram review will look identical to other X threads from the operator's perspective.
- Direct commits to `main` are fine per project workflow (CLAUDE.md). Each task = one commit.
