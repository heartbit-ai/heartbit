# Personal Blog (pascal.heartbit.ai) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth content surface to heartbit-ghost: weekly long-form blog posts seeded by the highest-engagement X post from the prior 7 days, rendered as a static site at `pascal.heartbit.ai` via a minimal Rust SSG, deployed to Cloudflare Pages.

**Architecture:** Mirrors the existing `persona_posts` pipeline pattern: scheduler → handler → pipeline (researcher → blog_writer → critic → fact_check → strict-sourcing pre-filter) → publish. The publish step writes Markdown to a posts directory + invokes a deterministic Rust SSG that renders the full static site. Telegram review optional but recommended for v1 (operator-in-the-loop). Static site files committed to a `blog-site/` directory at repo root; deployment is git-push-driven via Cloudflare Pages (zero-config).

**Tech Stack:** Rust 1.x, `minijinja` for templating, `pulldown-cmark` for Markdown→HTML, `serde_yaml` for frontmatter, existing X-Cient + Telegram delivery shared with persona_posts, Cloudflare Pages for hosting.

---

## File Structure

**Created:**

- `crates/heartbit-ghost/src/agents/blog_writer.rs` — `blog_writer_recipe()` + system prompt (long-form, 800-1500 word output, Markdown body, section structure, strict-sourcing rule reused)
- `crates/heartbit-ghost/src/blog/mod.rs` — pipeline runtime (`run_blog_pipeline`, `BlogConfig`, `BlogOutput`, `BlogOutcome`, `BlogError`, `BlogReviewDelivery` trait)
- `crates/heartbit-ghost/src/blog/prompts.rs` — user-message builders for blog_research + blog_writer + blog_critic + blog_fact
- `crates/heartbit-ghost/src/blog/seed.rs` — X-derived seed selection (`select_blog_seed` — wraps existing `JoinedTopPostsProvider::top_n` over a 7-day window, returns the highest-engagement post text + URL as the seed topic)
- `crates/heartbit-ghost/src/blog/markdown.rs` — Markdown writer (`write_post_markdown` — writes `posts/{date}-{slug}.md` with YAML frontmatter)
- `crates/heartbit-ghost/src/blog/render.rs` — SSG render entry (`render_site` — reads `posts/*.md`, emits `public/`)
- `crates/heartbit-ghost/src/blog/templates.rs` — embedded `minijinja` templates (3 templates: `base.html`, `post.html`, `index.html`) compiled-in via `include_str!`
- `crates/heartbit/src/daemon/persona_blog.rs` — `PersonaBlogScheduler` (weekly cron-style trigger with jitter, same shape as `persona_post.rs`)
- `crates/heartbit/src/daemon/persona_blog_handler.rs` — `handle_persona_blog` + `PersonaBlogDeps` (selects seed → invokes pipeline → renders site → commits markdown)
- `crates/heartbit/src/daemon/blog_context.rs` — `BlogContext` + `PersonaBlogEntry` (shared state, mirrors `QuotesContext`)
- `crates/heartbit-cli/src/bin/heartbit_blog_render.rs` — standalone binary for ad-hoc renders (`heartbit_blog_render --posts-dir posts/ --out-dir public/`)
- `blog-site/posts/.gitkeep` — placeholder so the directory exists in the repo
- `blog-site/public/.gitkeep` — placeholder so the directory exists in the repo
- `blog-site/style.css` — single stylesheet (~80 lines, dark-mode friendly)

**Modified:**

- `crates/heartbit-core/src/config/daemon.rs` — add `PersonaBlogConfig` + `DaemonConfig.persona_blog: Option<PersonaBlogConfig>`
- `crates/heartbit-core/src/config/mod.rs` — re-export `PersonaBlogConfig`
- `crates/heartbit/src/lib.rs` — re-export `BlogContext`, `PersonaBlogEntry`, blog command type
- `crates/heartbit-ghost/src/agents/mod.rs` — `pub mod blog_writer; pub use blog_writer::{BLOG_WRITER_SYSTEM_PROMPT, blog_writer_recipe};`
- `crates/heartbit-ghost/src/lib.rs` — `pub mod blog;`
- `crates/heartbit-ghost/Cargo.toml` — add `pulldown-cmark`, `minijinja`, `serde_yaml`, `slug` as deps
- `Cargo.toml` (workspace) — add the new deps to `[workspace.dependencies]`
- `crates/heartbit/src/daemon/types.rs` — add `DaemonCommand::PersonaBlog { persona: String }`
- `crates/heartbit/src/daemon/mod.rs` — declare new modules + re-exports
- `crates/heartbit/src/daemon/core.rs` — spawn `PersonaBlogScheduler` when configured, dispatch `PersonaBlog` to handler
- `crates/heartbit-cli/src/daemon/mod.rs` — build `BlogContext` from `[daemon.persona_blog]` at startup
- `crates/heartbit-cli/src/daemon/validate.rs` — validate the blog block (posts_dir + out_dir parent dirs exist, etc.)
- `docs/operating-heartbit.md` — document the new `[daemon.persona_blog]` knobs + render workflow + Cloudflare Pages setup

**Important: NOT modified:**

- Existing X writer / reply_writer / quote_writer recipes (blog has its own writer; the long-form voice deviates from 280-char short-form tweets)
- Existing post pipeline / reply pipeline / quote pipeline (parallel, not modified)

---

## Voice + Disposition Decisions (Locked from Brainstorm)

- **Voice profile**: same v5 dhh/mitsuhiko-flavored profile (no new persona). The voice translates well to long-form — opinionated, dry, technical, specific.
- **Disposition**: blog posts are **opinion-with-evidence**, not the caritas-in-veritate quote disposition. Standard zero-tolerance-for-invention sourcing rule applies (every quantitative claim has a URL).
- **Format**: 800-1500 words. Optional sections via Markdown `##`. Code blocks allowed (rendered via `pulldown-cmark`'s default syntax-aware renderer). No frontmatter beyond `title`, `date`, `slug`, `excerpt`, `tags`.
- **Tone latitude**: blog posts can use longer sentences, multi-clause structure, and footnote-style asides that X's 280-char cap forbids. The writer prompt explicitly authorizes this.

---

## Task 1: Workspace Deps + Crate Setup

**Files:**
- Modify: `Cargo.toml` (workspace dependencies)
- Modify: `crates/heartbit-ghost/Cargo.toml`

- [ ] **Step 1: Write the failing test**

Append to `crates/heartbit-ghost/src/lib.rs` (or any existing test file in heartbit-ghost) the following compile-only smoke test:

```rust
#[cfg(test)]
mod blog_deps_smoke {
    #[test]
    fn blog_deps_compile() {
        let _md = pulldown_cmark::Parser::new("# hello");
        let env = minijinja::Environment::new();
        let _ = env.render_str("{{ x }}", minijinja::context! { x => "y" });
        let _yaml: serde_yaml::Value = serde_yaml::from_str("a: 1").unwrap();
        let _slug = slug::slugify("Hello World!");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package heartbit-ghost --lib blog_deps_smoke`

Expected: compile error — `unresolved import \`pulldown_cmark\`` (and friends).

- [ ] **Step 3: Add the deps to the workspace Cargo.toml**

In `/home/pleclech/projects/heartbit/Cargo.toml`, find the `[workspace.dependencies]` block and add after the existing `whatlang` line:

```toml
pulldown-cmark = "0.12"
minijinja = "2"
serde_yaml = "0.9"
slug = "0.1"
```

- [ ] **Step 4: Add the deps to heartbit-ghost**

In `crates/heartbit-ghost/Cargo.toml`, add to `[dependencies]` (after the existing `whatlang` line):

```toml
pulldown-cmark = { workspace = true }
minijinja = { workspace = true }
serde_yaml = { workspace = true }
slug = { workspace = true }
```

- [ ] **Step 5: Run the smoke test**

Run: `cargo test --package heartbit-ghost --lib blog_deps_smoke -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/heartbit-ghost/Cargo.toml crates/heartbit-ghost/src/lib.rs
git commit -m "chore(deps): add pulldown-cmark + minijinja + serde_yaml + slug for blog SSG"
```

---

## Task 2: `blog_writer` Agent Recipe

**Files:**
- Create: `crates/heartbit-ghost/src/agents/blog_writer.rs`
- Modify: `crates/heartbit-ghost/src/agents/mod.rs` (declare + re-export)

- [ ] **Step 1: Create the recipe file with prompt + tests**

Create `crates/heartbit-ghost/src/agents/blog_writer.rs`:

```rust
//! Blog writer sub-agent — long-form (800-1500 words) opinionated essay
//! based on a topic + research digest. Used ONLY in the blog pipeline;
//! the short-form `writer` recipe is unchanged.

use heartbit_core::config::AgentConfig;

/// System prompt for the blog_writer.
///
/// Long-form latitude: multi-paragraph structure, sections, code blocks
/// where appropriate, multi-clause sentences. The zero-tolerance-for-
/// invention rule from the short-form writers carries over verbatim —
/// every quantitative claim must trace to the research digest.
pub const BLOG_WRITER_SYSTEM_PROMPT: &str = r#"You are a long-form essayist writing for a personal technical blog. Output ONE complete essay (800-1500 words) on the topic provided.

INPUT (from the user message)
- The TOPIC + framing (often derived from a high-engagement X post — your job is to expand on it with substance).
- A research digest with sourced facts and URLs.
- Voice guidelines for the persona.

OUTPUT
The essay text only, in Markdown. No preamble, no commentary, no surrounding quotation marks. Start with the first line of the essay (NOT a title — the renderer adds that from frontmatter). Multi-paragraph. Optional `## Section` headers when the structure warrants. Code blocks (triple-backtick) allowed when discussing code.

FORMAT
- 800-1500 words. Aim for the middle (~1100) unless the topic genuinely needs the extremes.
- Multi-paragraph. Use `## Section` headers when the essay has 3+ natural sections.
- Code blocks for code, not for emphasis.
- Footnote-style asides via parenthetical clauses or em-dashes are allowed (em-dashes are forbidden by short-form voice guidelines; the long-form variant relaxes this for asides ONLY, not for sentence breaks).
- One link per ~200 words, anchored in the prose. Use Markdown links: `[text](https://url)`.

VOICE
Honor the voice guidelines exactly — same opinionated/dry/technical/never-aggressive disposition that drives the X writer. Long-form lets you sustain an argument across paragraphs that wouldn't fit in 280 chars; use that latitude. A weak essay is one that reads like 4 tweets pasted together.

SOURCING — ZERO TOLERANCE FOR INVENTION
- Every quantitative claim (number, percentage, dollar amount, date, version) MUST trace to the research digest. Copy figures exactly; never paraphrase or approximate.
- "Plausible-sounding" is NOT verified. If the digest gives a range, do not collapse it to a point estimate.
- Attribution claims ("X said Y", "the paper showed Z") must trace verbatim to the research digest.
- If you don't have a sourced number for a claim, reframe qualitatively ("noticeably more", "in practice") OR drop the claim. Never invent precision.
- Every external URL in your essay must appear in the research digest. No invented URLs.

STRUCTURE GUIDANCE (advisory, not mandatory)
1. Open with a specific observation or claim — not a throat-clear ("In this post we will...").
2. Develop with 2-4 sections, each anchored on a sourced specific.
3. Close with something that's load-bearing for the reader — a tradeoff, a prediction, a question that compounds.

OUTPUT THE ESSAY ONLY. No frontmatter, no title line — the renderer handles those.
"#;

/// Construct the blog_writer [`AgentConfig`].
pub fn blog_writer_recipe() -> AgentConfig {
    AgentConfig {
        name: "blog_writer".to_string(),
        description: "Long-form opinionated essay (800-1500 words) from a topic + research digest."
            .to_string(),
        system_prompt: BLOG_WRITER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(4096),
        reasoning_effort: Some("medium".to_string()),
        ..super::stub_recipe("blog_writer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blog_writer_recipe_has_expected_shape() {
        let cfg = blog_writer_recipe();
        assert_eq!(cfg.name, "blog_writer");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(4096));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            cfg.response_schema.is_none(),
            "blog_writer produces free-form Markdown, no schema"
        );
    }

    /// Regression: long-form prompt must state the 800-1500 word range
    /// so the writer doesn't produce a 300-word stub OR a 5000-word
    /// dissertation.
    #[test]
    fn blog_writer_prompt_states_word_range() {
        let p = BLOG_WRITER_SYSTEM_PROMPT;
        assert!(p.contains("800-1500 words"), "must state word range");
        assert!(
            p.contains("Markdown"),
            "must specify Markdown output format"
        );
    }

    /// Regression: zero-tolerance sourcing rule must carry over from the
    /// short-form writer. The blog's strict-sourcing chain depends on it.
    #[test]
    fn blog_writer_prompt_states_zero_tolerance_for_invention() {
        let p = BLOG_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("ZERO TOLERANCE FOR INVENTION"),
            "must state zero-tolerance for invented quantities"
        );
        assert!(
            p.contains("research digest"),
            "must anchor sourcing in the research digest"
        );
        assert!(
            p.contains("invented URLs") || p.contains("invent precision"),
            "must explicitly forbid invented URLs or precision"
        );
    }

    /// Regression: the prompt must instruct the writer NOT to emit a
    /// title line — the renderer pulls the title from YAML frontmatter
    /// written separately by the markdown writer step.
    #[test]
    fn blog_writer_prompt_forbids_emitting_title() {
        let p = BLOG_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("NOT a title") || p.contains("no title line"),
            "writer must not emit a title — frontmatter handles it"
        );
    }
}
```

In `crates/heartbit-ghost/src/agents/mod.rs`, find the existing `pub mod quote_writer;` line and add alongside (alphabetically — `blog_writer` goes before `fact_check`):

```rust
pub mod blog_writer;
pub use blog_writer::{BLOG_WRITER_SYSTEM_PROMPT, blog_writer_recipe};
```

- [ ] **Step 2: Run tests**

Run: `cargo test --package heartbit-ghost --lib agents::blog_writer`

Expected: 4 tests PASS.

- [ ] **Step 3: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-ghost/src/agents/blog_writer.rs crates/heartbit-ghost/src/agents/mod.rs
git commit -m "feat(ghost): blog_writer recipe — long-form 800-1500 word essays"
```

---

## Task 3: Config — `PersonaBlogConfig`

**Files:**
- Modify: `crates/heartbit-core/src/config/daemon.rs`
- Modify: `crates/heartbit-core/src/config/mod.rs` (re-export)
- Modify: `crates/heartbit/src/lib.rs` (umbrella re-export)

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block in `crates/heartbit-core/src/config/daemon.rs`:

```rust
#[test]
fn persona_blog_config_parses_with_defaults() {
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
    let b = &cfg.persona_blog;
    assert_eq!(b.persona, "heartbit-ghost:x");
    assert!(b.enabled);
    assert_eq!(b.poll_interval_seconds, 604_800); // 7 days
    assert_eq!(b.interval_jitter_pct, 10);
    assert_eq!(b.posts_dir, "blog-site/posts");
    assert_eq!(b.out_dir, "blog-site/public");
    assert_eq!(b.seed_lookback_days, 7);
    assert_eq!(b.site_url, "https://pascal.heartbit.ai");
    assert_eq!(b.site_title, "pascal.heartbit.ai");
    assert!(b.writer_provider.is_none());
}

#[test]
fn persona_blog_config_rejects_missing_required_fields() {
    let toml = r#"
[persona_blog]
persona = "heartbit-ghost:x"
"#;
    // site_url is required (no default). Parse must fail.
    #[derive(Deserialize, Debug)]
    struct Shim {
        #[allow(dead_code)]
        persona_blog: PersonaBlogConfig,
    }
    let err = toml::from_str::<Shim>(toml).unwrap_err();
    assert!(
        err.to_string().contains("site_url"),
        "expected missing-field error for site_url; got: {err}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package heartbit-core --lib config::daemon::tests::persona_blog`

Expected: compile error — `PersonaBlogConfig` not defined.

- [ ] **Step 3: Add the struct + defaults**

In `crates/heartbit-core/src/config/daemon.rs`, append after the `PersonaQuotesConfig` block + its default-fn helpers:

```rust
/// Personal-blog publishing configuration.
///
/// When present, the daemon registers a `PersonaBlogScheduler` that
/// fires `DaemonCommand::PersonaBlog` on a weekly cadence (with jitter).
/// The handler picks the highest-engagement X post from the prior
/// `seed_lookback_days` (default 7) as the topic seed, drafts a
/// long-form essay via the `blog_writer` agent, routes through
/// Telegram for review, writes the picked draft as Markdown to
/// `posts_dir`, and re-renders the static site into `out_dir`.
///
/// Configured under `[daemon.persona_blog]` (single block, not a
/// list — one blog per daemon).
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaBlogConfig {
    /// Persona registry name (e.g. `"heartbit-ghost:x"`). Must match an
    /// existing persona whose post history + engagement store are
    /// already configured under `[[daemon.persona_posts]]` for the same
    /// slug — the blog reuses those stores for seed selection.
    pub persona: String,
    /// Whether the blog scheduler is enabled.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
    /// Polling interval in seconds. Default 604_800 (7 days = weekly).
    /// Validation: must be ≥3600 (1 hour) — anything tighter is almost
    /// certainly a misconfig and produces thin posts.
    #[serde(default = "default_blog_poll_interval_seconds")]
    pub poll_interval_seconds: u64,
    /// Jitter percentage applied to the cadence. Default 10 — tighter
    /// than X posts because weekly is already coarse and operators
    /// usually want a predictable day-of-week.
    #[serde(default = "default_blog_interval_jitter_pct")]
    pub interval_jitter_pct: u32,
    /// Optional active-hours window for the scheduler.
    #[serde(default)]
    pub active_hours: Option<ActiveHoursConfig>,
    /// Directory holding the generated Markdown post files. Relative to
    /// the daemon's CWD; tilde-expanded.
    /// Default: `"blog-site/posts"`.
    #[serde(default = "default_blog_posts_dir")]
    pub posts_dir: String,
    /// Directory where the rendered static site is written. Relative to
    /// CWD; tilde-expanded. This is what gets deployed to Cloudflare
    /// Pages (commit + push triggers deploy).
    /// Default: `"blog-site/public"`.
    #[serde(default = "default_blog_out_dir")]
    pub out_dir: String,
    /// How many days back to look for the highest-engagement X post
    /// used as the topic seed. Default 7. Set to 0 to disable
    /// X-derived seeding (forces the operator to set `topic_brief`).
    #[serde(default = "default_blog_seed_lookback_days")]
    pub seed_lookback_days: i64,
    /// Number of candidate essay drafts per tick. Default 2 — long-form
    /// drafts are expensive (~3-5k tokens each); 2 is enough for
    /// meaningful comparison without blowing the budget.
    #[serde(default = "default_blog_candidates_per_draft")]
    pub candidates_per_draft: usize,
    /// Public site URL (required) — used to build canonical URLs,
    /// OpenGraph tags, the sitemap, and the RSS feed.
    pub site_url: String,
    /// Site title rendered in `<title>` and the index page header.
    /// Default: `"pascal.heartbit.ai"`.
    #[serde(default = "default_blog_site_title")]
    pub site_title: String,
    /// Optional override LLM provider for the blog_writer + critic.
    /// `None` falls back to the global `[provider]`. Same shape as
    /// `persona_posts.writer_provider`.
    #[serde(default)]
    pub writer_provider: Option<super::agent::AgentProviderConfig>,
}

fn default_blog_poll_interval_seconds() -> u64 {
    604_800 // 7 days
}

fn default_blog_interval_jitter_pct() -> u32 {
    10
}

fn default_blog_posts_dir() -> String {
    "blog-site/posts".into()
}

fn default_blog_out_dir() -> String {
    "blog-site/public".into()
}

fn default_blog_seed_lookback_days() -> i64 {
    7
}

fn default_blog_candidates_per_draft() -> usize {
    2
}

fn default_blog_site_title() -> String {
    "pascal.heartbit.ai".into()
}
```

Add to the `DaemonConfig` struct (alongside `persona_quotes`):

```rust
    /// Personal blog configuration. Single block (one blog per daemon).
    /// When absent, the daemon does not spawn a blog scheduler.
    #[serde(default)]
    pub persona_blog: Option<PersonaBlogConfig>,
```

In `crates/heartbit-core/src/config/mod.rs`, add `PersonaBlogConfig` to the existing `pub use daemon::{...}` block.

In `crates/heartbit/src/lib.rs`, add `PersonaBlogConfig` to the umbrella re-export.

- [ ] **Step 4: Run tests**

Run: `cargo test --package heartbit-core --lib config::daemon::tests::persona_blog`

Expected: both tests PASS.

- [ ] **Step 5: Update existing DaemonConfig test fixtures**

Find all `DaemonConfig { ... }` struct-init sites that don't use `..Default::default()` and add `persona_blog: None,`. Use `grep -rn "DaemonConfig {" --include='*.rs' crates/` to enumerate. Typically the test fixtures in `crates/heartbit/src/daemon/{kafka,core}.rs`.

For each, add `persona_blog: None,` after the `persona_quotes` line.

- [ ] **Step 6: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --workspace -- -D warnings && cargo check --workspace --all-targets`

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-core/src/config/daemon.rs crates/heartbit-core/src/config/mod.rs \
        crates/heartbit/src/lib.rs crates/heartbit/src/daemon/core.rs crates/heartbit/src/daemon/kafka.rs
git commit -m "feat(config): PersonaBlogConfig + DaemonConfig.persona_blog"
```

---

## Task 4: Markdown Writer (`write_post_markdown`)

**Files:**
- Create: `crates/heartbit-ghost/src/blog/mod.rs` (scaffold)
- Create: `crates/heartbit-ghost/src/blog/markdown.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs` (declare `pub mod blog;`)

- [ ] **Step 1: Create scaffold + Markdown writer**

Create `crates/heartbit-ghost/src/blog/mod.rs`:

```rust
//! Personal-blog pipeline — picks an X-derived topic seed, drafts a
//! long-form essay via `blog_writer`, routes through Telegram review,
//! commits Markdown to disk + renders the static site.

pub mod markdown;
pub mod seed;
pub mod templates;
pub mod render;
pub mod prompts;

pub use markdown::{BlogPostFrontmatter, write_post_markdown, WriteMarkdownError};
pub use seed::{BlogSeed, SeedError, select_blog_seed};
pub use render::{render_site, RenderError, RenderedPostMeta};
```

In `crates/heartbit-ghost/src/lib.rs`, add `pub mod blog;` next to `pub mod posts;` and `pub mod quote;`.

Create `crates/heartbit-ghost/src/blog/markdown.rs`:

```rust
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

#[derive(Debug, thiserror::Error)]
pub enum WriteMarkdownError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
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
        let path =
            write_post_markdown(dir.path(), &fixture_front(), "Body content.\n").unwrap();
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
```

The other module files (`seed.rs`, `templates.rs`, `render.rs`, `prompts.rs`) will be created in later tasks; create them as empty stubs now to satisfy the `pub mod` declarations:

```bash
touch crates/heartbit-ghost/src/blog/{seed,templates,render,prompts}.rs
```

Then add minimal contents to each so they compile:

```rust
// seed.rs
//! X-derived blog topic seed selection. Implemented in Task 6.
#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct BlogSeed { pub _todo: () }

#[derive(Debug, thiserror::Error)]
pub enum SeedError {
    #[error("not implemented yet")]
    NotImplemented,
}

pub fn select_blog_seed() -> Result<BlogSeed, SeedError> {
    Err(SeedError::NotImplemented)
}
```

```rust
// templates.rs
//! minijinja templates for the blog SSG. Implemented in Task 7.
#![allow(dead_code)]
```

```rust
// render.rs
//! Static-site renderer. Implemented in Task 8.
#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct RenderedPostMeta { pub _todo: () }

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("not implemented yet")]
    NotImplemented,
}

pub fn render_site() -> Result<Vec<RenderedPostMeta>, RenderError> {
    Err(RenderError::NotImplemented)
}
```

```rust
// prompts.rs
//! User-message builders for the blog pipeline. Implemented in Task 5.
```

> Implementer note: these stubs exist so the `pub use` lines in `blog/mod.rs` resolve. They'll be filled in by the following tasks. Don't leave stub items in the public re-exports past their owning task.

- [ ] **Step 2: Run Markdown writer tests**

Run: `cargo test --package heartbit-ghost --lib blog::markdown`

Expected: 5 tests PASS.

- [ ] **Step 3: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-ghost/src/blog/ crates/heartbit-ghost/src/lib.rs
git commit -m "feat(ghost): blog module scaffold + write_post_markdown"
```

---

## Task 5: Prompt Builders

**Files:**
- Modify: `crates/heartbit-ghost/src/blog/prompts.rs`

- [ ] **Step 1: Write the prompts module with tests**

Replace `crates/heartbit-ghost/src/blog/prompts.rs` with:

```rust
//! User-message builders for each blog-pipeline stage. Pure string
//! composition — same shape as `reply/prompts.rs` and `quote/prompts.rs`.

use crate::blog::seed::BlogSeed;

/// Build the blog researcher's user message. The seed gives the topic;
/// the researcher's job is to surface sourced specifics that the writer
/// will weave into the essay.
pub(crate) fn build_blog_research_user_message(seed: &BlogSeedInput<'_>) -> String {
    let mut out = String::new();
    out.push_str("TOPIC SEED\n");
    out.push_str(&format!("Seed text: {}\n", seed.text));
    if let Some(url) = seed.source_url {
        out.push_str(&format!("Originally posted at: {url}\n"));
    }
    if let Some(rationale) = seed.rationale {
        out.push_str(&format!("Why this topic: {rationale}\n"));
    }
    out.push_str("\nResearch this topic. Find 4-6 substantive sources with sourced specifics (numbers, dates, citations, attributions). Output the structured digest per your system prompt. Do NOT compose the essay — the blog_writer composes it next.\n");
    out
}

/// Build the blog writer's user message. Includes the research digest,
/// the topic seed for framing context, and voice guidelines.
pub(crate) fn build_blog_writer_user_message(
    digest: &str,
    seed: &BlogSeedInput<'_>,
    voice_guidelines: &str,
) -> String {
    let mut out = String::new();
    out.push_str("Research digest (sourced facts to anchor the essay):\n");
    out.push_str(digest);
    out.push_str("\n\n");
    out.push_str(&format!("TOPIC SEED: {}\n", seed.text));
    if let Some(url) = seed.source_url {
        out.push_str(&format!(
            "(This topic was derived from a high-engagement X post: {url}. The essay expands on the idea; it does NOT quote or re-tweet the original.)\n"
        ));
    }
    out.push('\n');
    out.push_str(voice_guidelines);
    out.push('\n');
    out.push_str("\nWrite ONE complete essay (800-1500 words) in Markdown. Output the essay text only — no title line, no frontmatter.\n");
    out
}

/// Build the blog style critic's user message.
pub(crate) fn build_blog_critic_user_message(draft: &str, voice_guidelines: &str) -> String {
    format!(
        "Essay draft to evaluate:\n\n{draft}\n\n{voice_guidelines}\n\nScore the draft and return your verdict as JSON per the schema.\n"
    )
}

/// Build the blog fact-check's user message.
pub(crate) fn build_blog_fact_user_message(draft: &str, digest: &str) -> String {
    format!(
        "Essay draft to verify:\n\n{draft}\n\nResearch digest (only source of truth):\n\n{digest}\n\nVerify and return your verdict as JSON per the schema.\n"
    )
}

/// Lightweight seed projection — only the fields the prompts need. Lets
/// the prompts module be testable without depending on the full
/// `BlogSeed` runtime type.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlogSeedInput<'a> {
    pub text: &'a str,
    pub source_url: Option<&'a str>,
    pub rationale: Option<&'a str>,
}

impl<'a> From<&'a BlogSeed> for BlogSeedInput<'a> {
    fn from(seed: &'a BlogSeed) -> Self {
        // Implementer note: this conversion is filled in once BlogSeed
        // gains its real fields in Task 6. For now the stub seed has
        // a single `_todo` field; this impl returns placeholder strings
        // to keep prompts compileable. Replace with the real projection
        // when BlogSeed is fleshed out.
        let _ = seed;
        Self {
            text: "<seed-todo>",
            source_url: None,
            rationale: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_seed_input<'a>() -> BlogSeedInput<'a> {
        BlogSeedInput {
            text: "Every tool response your LLM agent consumes is a potential attack vector.",
            source_url: Some("https://twitter.com/i/web/status/2054484107212538042"),
            rationale: Some("Highest engagement post in the last 7 days (score 4.2)."),
        }
    }

    #[test]
    fn writer_message_includes_seed_and_digest() {
        let seed = fixture_seed_input();
        let s = build_blog_writer_user_message("digest text", &seed, "VOICE GUIDELINES");
        assert!(s.contains("digest text"));
        assert!(s.contains("Every tool response"));
        assert!(s.contains("VOICE GUIDELINES"));
        assert!(s.contains("800-1500 words"));
    }

    #[test]
    fn writer_message_mentions_no_title_line() {
        let seed = fixture_seed_input();
        let s = build_blog_writer_user_message("d", &seed, "v");
        assert!(
            s.contains("no title line") || s.contains("no frontmatter"),
            "writer must be told to skip title/frontmatter (renderer handles them): {s}"
        );
    }

    #[test]
    fn writer_message_clarifies_essay_is_not_a_quote_of_the_source() {
        let seed = fixture_seed_input();
        let s = build_blog_writer_user_message("d", &seed, "v");
        assert!(
            s.contains("does NOT quote or re-tweet"),
            "writer must be told the essay EXPANDS on the seed, not quotes it"
        );
    }

    #[test]
    fn research_message_includes_topic_seed() {
        let seed = fixture_seed_input();
        let s = build_blog_research_user_message(&seed);
        assert!(s.contains("Every tool response"));
        assert!(s.contains("Originally posted at"));
        assert!(s.contains("4-6 substantive sources"));
        assert!(s.contains("Do NOT compose the essay"));
    }

    #[test]
    fn critic_message_includes_draft_and_voice() {
        let s = build_blog_critic_user_message("DRAFT", "VOICE");
        assert!(s.contains("DRAFT"));
        assert!(s.contains("VOICE"));
        assert!(s.contains("JSON"));
    }

    #[test]
    fn fact_message_includes_draft_and_digest() {
        let s = build_blog_fact_user_message("DRAFT", "DIGEST");
        assert!(s.contains("DRAFT"));
        assert!(s.contains("DIGEST"));
        assert!(s.contains("only source of truth"));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --package heartbit-ghost --lib blog::prompts`

Expected: 6 tests PASS.

- [ ] **Step 3: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-ghost/src/blog/prompts.rs
git commit -m "feat(ghost): blog pipeline prompt builders"
```

---

## Task 6: X-Derived Seed Selection (`select_blog_seed`)

**Files:**
- Modify: `crates/heartbit-ghost/src/blog/seed.rs`
- Modify: `crates/heartbit-ghost/src/blog/prompts.rs` (replace the stub `From<&BlogSeed>` impl with the real one)

- [ ] **Step 1: Write the real seed module**

Replace `crates/heartbit-ghost/src/blog/seed.rs` with:

```rust
//! X-derived blog topic seed selection.
//!
//! Picks the highest-engagement X post from the prior `lookback_days`
//! window as the seed for this week's blog essay. The seed text becomes
//! the topic context fed to the researcher; the essay then EXPANDS on
//! that topic rather than quoting the original (the blog is its own
//! publishing surface, not a quote-tweet aggregator).

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};

use crate::posts::TopPostsProvider;

/// A topic seed for the blog pipeline.
#[derive(Debug, Clone)]
pub struct BlogSeed {
    /// Text of the source X post (first tweet for threads).
    pub text: String,
    /// Public URL of the source X post.
    pub source_url: String,
    /// Tweet ID for traceability.
    pub source_tweet_id: String,
    /// When the source X post was published.
    pub source_posted_at: DateTime<Utc>,
    /// Composite engagement score of the source post.
    pub engagement_score: f64,
    /// Operator-facing summary of why this seed was chosen.
    pub rationale: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SeedError {
    /// No high-engagement post was found within the lookback window.
    /// Returned when `top_n` returns no results, the top result is
    /// outside the window, or the top result has zero engagement.
    /// The handler should record `BlogOutcome::NoSeed` and not invoke
    /// the writer pipeline.
    #[error("no eligible seed within {lookback_days} day window")]
    NoEligibleSeed { lookback_days: i64 },
}

/// Pick the highest-engagement post from the prior `lookback_days` as
/// the topic seed.
///
/// `n` is the number of top posts to consider from the provider. The
/// function walks them in descending engagement order and picks the
/// first one whose `posted_at` is within the window. `n=10` is a
/// reasonable default — see `select_blog_seed_default_n`.
pub async fn select_blog_seed(
    provider: &dyn TopPostsProvider,
    n: usize,
    lookback_days: i64,
    now: DateTime<Utc>,
) -> Result<BlogSeed, SeedError> {
    if lookback_days <= 0 {
        return Err(SeedError::NoEligibleSeed { lookback_days });
    }
    let cutoff = now - Duration::days(lookback_days);
    let candidates = provider
        .top_n(n)
        .await
        .map_err(|_| SeedError::NoEligibleSeed { lookback_days })?;

    for c in candidates {
        if c.posted_at < cutoff {
            continue;
        }
        if c.engagement_score <= 0.0 {
            continue;
        }
        let source_url = format!("https://twitter.com/i/web/status/{}", c.tweet_id);
        let rationale = format!(
            "Top-engagement X post in the last {lookback_days} days (score {:.2}, posted {}).",
            c.engagement_score,
            c.posted_at.format("%Y-%m-%d")
        );
        return Ok(BlogSeed {
            text: c.text.clone(),
            source_url,
            source_tweet_id: c.tweet_id.clone(),
            source_posted_at: c.posted_at,
            engagement_score: c.engagement_score,
            rationale,
        });
    }
    Err(SeedError::NoEligibleSeed { lookback_days })
}

/// Default `n` for `select_blog_seed` callers.
pub const DEFAULT_TOP_N: usize = 10;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posts::{TopPost, TopPostsFut};
    use chrono::TimeZone;
    use std::pin::Pin;

    struct MockProvider {
        posts: Vec<TopPost>,
    }

    impl TopPostsProvider for MockProvider {
        fn top_n<'a>(&'a self, n: usize) -> TopPostsFut<'a> {
            let posts: Vec<TopPost> = self.posts.iter().take(n).cloned().collect();
            Box::pin(async move { Ok(posts) })
        }
    }

    fn now_fixture() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 16, 12, 0, 0).unwrap()
    }

    fn post(id: &str, score: f64, days_ago: i64) -> TopPost {
        TopPost {
            tweet_id: id.into(),
            text: format!("Post body for {id}."),
            posted_at: now_fixture() - Duration::days(days_ago),
            engagement_score: score,
        }
    }

    #[tokio::test]
    async fn picks_highest_engagement_within_window() {
        let provider = MockProvider {
            posts: vec![
                post("100", 5.0, 2),
                post("101", 3.0, 5),
                post("102", 1.0, 1),
            ],
        };
        let seed = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap();
        assert_eq!(seed.source_tweet_id, "100");
        assert_eq!(seed.engagement_score, 5.0);
        assert!(seed.source_url.contains("100"));
        assert!(seed.rationale.contains("score 5.00"));
    }

    #[tokio::test]
    async fn skips_posts_outside_window() {
        let provider = MockProvider {
            posts: vec![
                post("100", 10.0, 30), // outside 7-day window
                post("101", 5.0, 3),   // inside
            ],
        };
        let seed = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap();
        assert_eq!(seed.source_tweet_id, "101");
    }

    #[tokio::test]
    async fn errors_when_no_eligible_posts() {
        let provider = MockProvider {
            posts: vec![post("100", 10.0, 30), post("101", 5.0, 21)],
        };
        let err = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap_err();
        assert!(matches!(err, SeedError::NoEligibleSeed { lookback_days: 7 }));
    }

    #[tokio::test]
    async fn skips_zero_engagement() {
        let provider = MockProvider {
            posts: vec![post("100", 0.0, 2), post("101", 2.5, 3)],
        };
        let seed = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap();
        assert_eq!(seed.source_tweet_id, "101");
    }

    #[tokio::test]
    async fn empty_provider_returns_no_eligible_seed() {
        let provider = MockProvider { posts: vec![] };
        let err = select_blog_seed(&provider, 10, 7, now_fixture())
            .await
            .unwrap_err();
        assert!(matches!(err, SeedError::NoEligibleSeed { .. }));
    }

    #[tokio::test]
    async fn negative_lookback_returns_no_seed() {
        let provider = MockProvider {
            posts: vec![post("100", 5.0, 1)],
        };
        let err = select_blog_seed(&provider, 10, 0, now_fixture())
            .await
            .unwrap_err();
        assert!(matches!(err, SeedError::NoEligibleSeed { lookback_days: 0 }));
    }
}

/// Re-exported alias for `TopPostsFut`. The provider's future type is
/// declared in `crate::posts::engagement`. This module doesn't define
/// its own — we just need `TopPost` and `TopPostsProvider`.
#[doc(hidden)]
pub use crate::posts::TopPostsFut as _Ensure;
```

Now update the `From<&BlogSeed>` impl in `crates/heartbit-ghost/src/blog/prompts.rs` to use the real fields. Replace the stub block with:

```rust
impl<'a> From<&'a BlogSeed> for BlogSeedInput<'a> {
    fn from(seed: &'a BlogSeed) -> Self {
        Self {
            text: &seed.text,
            source_url: Some(&seed.source_url),
            rationale: Some(&seed.rationale),
        }
    }
}
```

Also update `crates/heartbit-ghost/src/blog/mod.rs` to export `DEFAULT_TOP_N`:

```rust
pub use seed::{BlogSeed, SeedError, select_blog_seed, DEFAULT_TOP_N};
```

- [ ] **Step 2: Confirm `TopPostsFut` is exported from `posts`**

Check that `crate::posts::TopPostsFut` is exposed. If not, add:

```rust
// in crates/heartbit-ghost/src/posts/mod.rs
pub use engagement::TopPostsFut;
```

(Skip if already exported.)

- [ ] **Step 3: Run tests**

Run: `cargo test --package heartbit-ghost --lib blog::seed`

Expected: 6 tests PASS.

- [ ] **Step 4: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/blog/seed.rs crates/heartbit-ghost/src/blog/prompts.rs \
        crates/heartbit-ghost/src/blog/mod.rs crates/heartbit-ghost/src/posts/mod.rs
git commit -m "feat(ghost): select_blog_seed — X-derived weekly topic from top engagement"
```

---

## Task 7: Templates + CSS

**Files:**
- Create: `blog-site/templates/base.html`
- Create: `blog-site/templates/post.html`
- Create: `blog-site/templates/index.html`
- Create: `blog-site/style.css`
- Modify: `crates/heartbit-ghost/src/blog/templates.rs`
- Create: `blog-site/posts/.gitkeep`
- Create: `blog-site/public/.gitkeep`

- [ ] **Step 1: Create the directory layout + .gitkeeps**

```bash
mkdir -p blog-site/templates blog-site/posts blog-site/public
touch blog-site/posts/.gitkeep blog-site/public/.gitkeep
```

- [ ] **Step 2: Write `blog-site/templates/base.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{% block title %}{{ site_title }}{% endblock %}</title>
  <meta name="description" content="{% block description %}{{ site_description | default(value=site_title) }}{% endblock %}">
  <link rel="canonical" href="{% block canonical %}{{ site_url }}{% endblock %}">
  <link rel="alternate" type="application/rss+xml" title="{{ site_title }}" href="{{ site_url }}/feed.xml">
  <link rel="stylesheet" href="/style.css">
  {# OpenGraph + Twitter card — minimal but complete. #}
  <meta property="og:title" content="{% block og_title %}{{ site_title }}{% endblock %}">
  <meta property="og:description" content="{% block og_description %}{{ site_description | default(value=site_title) }}{% endblock %}">
  <meta property="og:type" content="{% block og_type %}website{% endblock %}">
  <meta property="og:url" content="{% block og_url %}{{ site_url }}{% endblock %}">
  <meta name="twitter:card" content="summary_large_image">
  {% block extra_head %}{% endblock %}
</head>
<body>
  <header class="site-header">
    <a class="site-title" href="/">{{ site_title }}</a>
    <nav class="site-nav">
      <a href="/">posts</a>
      <a href="/feed.xml">rss</a>
    </nav>
  </header>
  <main class="site-main">
    {% block content %}{% endblock %}
  </main>
  <footer class="site-footer">
    <p>{{ site_title }} · <a href="/feed.xml">RSS</a></p>
  </footer>
</body>
</html>
```

- [ ] **Step 3: Write `blog-site/templates/post.html`**

```html
{% extends "base.html" %}
{% block title %}{{ post.title }} · {{ site_title }}{% endblock %}
{% block description %}{{ post.excerpt }}{% endblock %}
{% block canonical %}{{ site_url }}/{{ post.slug }}/{% endblock %}
{% block og_title %}{{ post.title }}{% endblock %}
{% block og_description %}{{ post.excerpt }}{% endblock %}
{% block og_type %}article{% endblock %}
{% block og_url %}{{ site_url }}/{{ post.slug }}/{% endblock %}
{% block extra_head %}
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": {{ post.title | tojson }},
  "datePublished": {{ post.date_iso | tojson }},
  "description": {{ post.excerpt | tojson }},
  "mainEntityOfPage": {{ post_url | tojson }}
}
</script>
{% endblock %}
{% block content %}
<article class="post">
  <header class="post-header">
    <h1 class="post-title">{{ post.title }}</h1>
    <time class="post-date" datetime="{{ post.date_iso }}">{{ post.date_human }}</time>
  </header>
  <div class="post-body">
    {{ post.body_html | safe }}
  </div>
  {% if post.tags %}
  <footer class="post-tags">
    {% for tag in post.tags %}<span class="tag">{{ tag }}</span>{% endfor %}
  </footer>
  {% endif %}
</article>
{% endblock %}
```

- [ ] **Step 4: Write `blog-site/templates/index.html`**

```html
{% extends "base.html" %}
{% block content %}
<section class="post-list">
  {% for post in posts %}
  <article class="post-card">
    <h2 class="post-card-title"><a href="/{{ post.slug }}/">{{ post.title }}</a></h2>
    <time class="post-card-date" datetime="{{ post.date_iso }}">{{ post.date_human }}</time>
    <p class="post-card-excerpt">{{ post.excerpt }}</p>
  </article>
  {% endfor %}
  {% if not posts %}
  <p class="empty">No posts yet.</p>
  {% endif %}
</section>
{% endblock %}
```

- [ ] **Step 5: Write `blog-site/style.css`**

```css
:root {
  --bg: #0e0e10;
  --fg: #e6e6e6;
  --muted: #9a9a9a;
  --accent: #f1c40f;
  --border: #2a2a30;
  --max-w: 720px;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; line-height: 1.6; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.site-header, .site-main, .site-footer { max-width: var(--max-w); margin: 0 auto; padding: 1.5rem 1rem; }
.site-header { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid var(--border); }
.site-title { font-weight: 700; font-size: 1.1rem; }
.site-nav a { margin-left: 1rem; color: var(--muted); }
.site-footer { color: var(--muted); border-top: 1px solid var(--border); font-size: 0.85rem; text-align: center; }
.post-card { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border); }
.post-card-title { margin: 0 0 0.25rem 0; font-size: 1.4rem; }
.post-card-date { color: var(--muted); font-size: 0.85rem; }
.post-card-excerpt { margin: 0.5rem 0 0 0; color: var(--fg); }
.post-header { margin-bottom: 2rem; }
.post-title { margin: 0 0 0.25rem 0; font-size: 2rem; line-height: 1.2; }
.post-date { color: var(--muted); font-size: 0.9rem; }
.post-body h2 { margin-top: 2rem; font-size: 1.4rem; }
.post-body h3 { margin-top: 1.5rem; font-size: 1.15rem; }
.post-body pre { background: #1a1a1f; padding: 1rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; }
.post-body code { background: #1a1a1f; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.9em; }
.post-body pre code { background: none; padding: 0; }
.post-body blockquote { border-left: 3px solid var(--accent); padding-left: 1rem; color: var(--muted); margin: 1rem 0; }
.post-tags { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); }
.tag { display: inline-block; background: var(--border); color: var(--muted); padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem; margin-right: 0.3rem; }
.empty { color: var(--muted); text-align: center; padding: 2rem 0; }
```

- [ ] **Step 6: Replace `crates/heartbit-ghost/src/blog/templates.rs` with the embedded loader**

```rust
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
```

- [ ] **Step 7: Run tests**

Run: `cargo test --package heartbit-ghost --lib blog::templates`

Expected: 5 tests PASS.

- [ ] **Step 8: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add blog-site/ crates/heartbit-ghost/src/blog/templates.rs
git commit -m "feat(ghost): blog templates (base + post + index) + dark-mode CSS"
```

---

## Task 8: Static Site Renderer (`render_site`)

**Files:**
- Modify: `crates/heartbit-ghost/src/blog/render.rs`

- [ ] **Step 1: Write the renderer with tests**

Replace `crates/heartbit-ghost/src/blog/render.rs` with:

```rust
//! Static-site renderer. Reads `posts_dir/*.md` (frontmatter + body),
//! renders each post into `out_dir/<slug>/index.html`, regenerates
//! `out_dir/index.html`, `out_dir/feed.xml`, `out_dir/sitemap.xml`,
//! and copies `style.css`.
//!
//! Pure I/O — no LLM calls, no network. Safe to run standalone via
//! the `heartbit_blog_render` binary or invoke after a successful
//! pipeline tick.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use minijinja::context;
use pulldown_cmark::{Options, Parser, html as md_html};
use serde::{Deserialize, Serialize};

use crate::blog::markdown::BlogPostFrontmatter;
use crate::blog::templates::build_env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderedPostMeta {
    pub slug: String,
    pub title: String,
    pub date: DateTime<Utc>,
    pub excerpt: String,
    pub tags: Vec<String>,
    pub output_path: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("frontmatter parse error in {file}: {source}")]
    Frontmatter {
        file: String,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("missing frontmatter in {file} (no leading `---`)")]
    MissingFrontmatter { file: String },
    #[error("template: {0}")]
    Template(#[from] minijinja::Error),
    #[error("style.css not found at {0} — needed for copy to out_dir")]
    StyleNotFound(PathBuf),
}

/// Site-level config passed into every render call. Mirrors the
/// `PersonaBlogConfig` knobs but doesn't depend on the daemon crate
/// (so this module is testable in isolation).
#[derive(Debug, Clone)]
pub struct RenderConfig<'a> {
    pub site_url: &'a str,
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

    rendered.sort_by(|a, b| b.date.cmp(&a.date));

    // index.html
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

    // feed.xml (RSS 2.0)
    let feed_xml = render_rss(&rendered, cfg);
    std::fs::write(out_dir.join("feed.xml"), feed_xml)?;

    // sitemap.xml
    let sitemap_xml = render_sitemap(&rendered, cfg);
    std::fs::write(out_dir.join("sitemap.xml"), sitemap_xml)?;

    // robots.txt
    let robots = format!(
        "User-agent: *\nAllow: /\nSitemap: {}/sitemap.xml\n",
        cfg.site_url.trim_end_matches('/')
    );
    std::fs::write(out_dir.join("robots.txt"), robots)?;

    // style.css
    if !cfg.style_css.exists() {
        return Err(RenderError::StyleNotFound(cfg.style_css.to_path_buf()));
    }
    std::fs::copy(cfg.style_css, out_dir.join("style.css"))?;

    Ok(rendered)
}

fn read_posts(
    posts_dir: &Path,
) -> Result<Vec<(BlogPostFrontmatter, String)>, RenderError> {
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

fn parse_post(
    content: &str,
    file: &str,
) -> Result<(BlogPostFrontmatter, String), RenderError> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return Err(RenderError::MissingFrontmatter {
            file: file.to_string(),
        });
    }
    let after_first = &trimmed[3..]; // skip first ---
    let end = after_first.find("\n---\n").ok_or_else(|| {
        RenderError::MissingFrontmatter {
            file: file.to_string(),
        }
    })?;
    let yaml = &after_first[..end];
    let body = &after_first[end + 5..]; // skip "\n---\n"
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
```

- [ ] **Step 2: Run renderer tests**

Run: `cargo test --package heartbit-ghost --lib blog::render`

Expected: 8 tests PASS.

- [ ] **Step 3: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-ghost/src/blog/render.rs crates/heartbit-ghost/src/blog/mod.rs
git commit -m "feat(ghost): render_site — markdown → HTML + RSS + sitemap + robots"
```

---

## Task 9: Blog Pipeline (`run_blog_pipeline`)

**Files:**
- Modify: `crates/heartbit-ghost/src/blog/mod.rs` (add the pipeline)

The pipeline mirrors `quote/mod.rs::run_quote_pipeline` and `pipeline::generate_candidate` with three differences:
1. Input seed type is `BlogSeed` (not a tweet to quote / not an open-form topic from generator)
2. Writer is `blog_writer` (long-form)
3. Publish action is `write_post_markdown` + `render_site` (not a tweet POST)

Read `crates/heartbit-ghost/src/quote/mod.rs` end-to-end as the template before writing. Translate as:
- `QuoteCandidate` (source X tweet) → `BlogSeed`
- `QuoteConfig` → `BlogConfig`
- `quote_writer_recipe` → `blog_writer_recipe`
- twitter publish → `write_post_markdown` + `render_site`
- `QuoteOutcome::Posted { chosen_index, tweet_id, url }` → `BlogOutcome::Posted { chosen_index, post_path, post_url }`
- New variant: `BlogOutcome::NoSeed` (no eligible X post in lookback window)

Required types in `crates/heartbit-ghost/src/blog/mod.rs` (add to the existing scaffold):

```rust
use std::path::PathBuf;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::llm::types::TokenUsage;
use thiserror::Error;

use crate::pipeline::{PipelineError, ProgressCallback};

pub struct BlogConfig<'a> {
    pub persona_name: &'a str,
    pub provider: Arc<BoxedProvider>,
    pub writer_provider: Option<Arc<BoxedProvider>>,
    pub corpora_root: &'a std::path::Path,
    pub profiles_root: &'a std::path::Path,
    pub on_progress: Option<ProgressCallback>,
    pub seed: BlogSeed,
    pub candidates_per_draft: usize,
    pub delivery: Arc<dyn BlogReviewDelivery>,
    pub credentials: Arc<dyn CredentialResolver>,
    pub posts_dir: &'a std::path::Path,
    pub out_dir: &'a std::path::Path,
    pub style_css: &'a std::path::Path,
    pub site_url: &'a str,
    pub site_title: &'a str,
}

#[derive(Debug, Clone)]
pub struct BlogOutput {
    pub seed: BlogSeed,
    pub candidates: Vec<BlogCandidateRecord>,
    pub usage_summary: TokenUsage,
    pub outcome: BlogOutcome,
}

#[derive(Debug, Clone)]
pub struct BlogCandidateRecord {
    pub draft: String,
    pub style_match_score: f32,
    pub fact_check_verdict: crate::pipeline::FactVerdict,
    pub title: String,
    pub slug: String,
    pub excerpt: String,
}

#[derive(Debug, Clone)]
pub enum BlogOutcome {
    Posted {
        chosen_index: usize,
        post_path: PathBuf,
        post_url: String,
    },
    Skipped,
    TimedOut,
    AllCandidatesGateRejected {
        reasons: Vec<String>,
    },
    NoSeed,
}

#[derive(Debug, Error)]
pub enum BlogError {
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("delivery: {0}")]
    Delivery(#[from] crate::review::ReviewDeliveryError),
    #[error("markdown: {0}")]
    Markdown(#[from] markdown::WriteMarkdownError),
    #[error("render: {0}")]
    Render(#[from] render::RenderError),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

pub trait BlogReviewDelivery: Send + Sync {
    fn deliver_and_await<'a>(
        &'a self,
        message: &'a BlogReviewMessage,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        crate::review::DeliveredReview,
                        crate::review::ReviewDeliveryError,
                    >,
                > + Send
                + 'a,
        >,
    >;

    fn report<'a>(
        &'a self,
        receipt: crate::review::DeliveryReceipt,
        outcome: BlogOutcome,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<(), crate::review::ReviewDeliveryError>>
                + Send
                + 'a,
        >,
    >;
}

#[derive(Debug, Clone)]
pub struct BlogReviewMessage {
    pub persona_name: String,
    pub seed_text: String,
    pub seed_url: String,
    /// Each candidate is the full essay markdown — operator reads the
    /// full thing in Telegram and picks one (or skips).
    pub candidates: Vec<String>,
    pub interaction_id: uuid::Uuid,
}

pub async fn run_blog_pipeline(cfg: BlogConfig<'_>) -> Result<BlogOutput, BlogError> {
    // Implementer note: this function is ~400-500 lines. Read
    // `crates/heartbit-ghost/src/quote/mod.rs::run_quote_pipeline` end
    // to end first; the structure is nearly identical. Key differences:
    //
    // 1. NO topic generator — the topic IS the seed (already chosen).
    // 2. Researcher's user message = build_blog_research_user_message
    // 3. Writer = blog_writer_recipe (max_tokens 4096)
    //    NO length_normalize step (we want long-form output verbatim)
    // 4. Critic + fact_check as usual (mirror quote pipeline)
    // 5. Pre-filter: drop FactVerdict::Unverifiable; NO 280-char check
    //    (blog posts don't have the X length cap)
    // 6. Build BlogReviewMessage with full essay drafts
    // 7. delivery.deliver_and_await
    // 8. On Pick(idx):
    //    a. Extract title + slug + excerpt from the chosen draft:
    //       - title: first non-blank line, stripped of leading `# `
    //         if present (writer may or may not emit one — the prompt
    //         says no, but be permissive)
    //       - slug: slug::slugify(title)
    //       - excerpt: first paragraph, trimmed to ~160 chars
    //    b. write_post_markdown(posts_dir, frontmatter, body)
    //    c. render_site(posts_dir, out_dir, RenderConfig{...})
    //    d. Return BlogOutcome::Posted { chosen_index, post_path,
    //       post_url: format!("{site_url}/{slug}/") }
    todo!("implement following the quote pipeline pattern — see comment above")
}
```

This is genuinely a large task. For execution, the implementer subagent should:

1. Open `crates/heartbit-ghost/src/quote/mod.rs` and read it end-to-end.
2. Copy the `run_quote_pipeline` function structure into `run_blog_pipeline`, mechanically applying the translation table above.
3. Write **7 tests** mirroring the quote pipeline tests, with names:
   - `run_blog_pipeline_pick_index_0_writes_markdown_and_renders_site`
   - `run_blog_pipeline_skip_returns_skipped_no_write`
   - `run_blog_pipeline_timed_out_returns_timed_out_no_write`
   - `run_blog_pipeline_all_candidates_gate_rejected_skips_delivery`
   - `run_blog_pipeline_all_unverifiable_returns_all_candidates_gate_rejected`
   - `run_blog_pipeline_writes_post_to_slugged_subdir`
   - `run_blog_pipeline_render_failure_is_reported`

Use `MockProvider`, `MockBlogReviewDelivery`, and a real `posts_dir` + `out_dir` via `tempfile::tempdir()`. The renderer is real (no mock) — its own tests already cover correctness.

- [ ] **Step 1: Read the quote pipeline template**

Run: `wc -l crates/heartbit-ghost/src/quote/mod.rs`
Expected output: ~1100 lines.

Open the file and read it. The translation should produce a similar-size file (~1000-1200 lines).

- [ ] **Step 2: Implement `run_blog_pipeline`**

Replace the `todo!()` with the translated pipeline body. The structure follows quote/mod.rs section-by-section.

- [ ] **Step 3: Implement the 7 tests**

Each test follows the quote pipeline test layout. Reference `crates/heartbit-ghost/src/quote/mod.rs` test module.

- [ ] **Step 4: Run all blog tests**

Run: `cargo test --package heartbit-ghost --lib blog::`

Expected: ~25 tests PASS (5 markdown + 6 prompts + 6 seed + 5 templates + 8 render + 7 pipeline).

- [ ] **Step 5: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/blog/mod.rs
git commit -m "feat(ghost): run_blog_pipeline — seed → research → essay → critic → fact → render"
```

---

## Task 10: Standalone Render Binary

**Files:**
- Create: `crates/heartbit-cli/src/bin/heartbit_blog_render.rs`
- Modify: `crates/heartbit-cli/Cargo.toml` (declare the bin)

This binary lets the operator regenerate the site after editing a template or post manually, without running the daemon.

- [ ] **Step 1: Create the binary**

```rust
//! `heartbit_blog_render` — regenerate the blog static site from
//! Markdown posts. Operator tool for when templates change or
//! posts are edited manually.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use heartbit_ghost::blog::render::{RenderConfig, render_site};

#[derive(Debug, Parser)]
#[command(version, about = "Regenerate the blog static site from Markdown posts.")]
struct Args {
    /// Directory holding `*.md` posts.
    #[arg(long, default_value = "blog-site/posts")]
    posts_dir: PathBuf,

    /// Output directory for rendered HTML (this is what gets deployed).
    #[arg(long, default_value = "blog-site/public")]
    out_dir: PathBuf,

    /// Path to style.css (copied verbatim into out_dir).
    #[arg(long, default_value = "blog-site/style.css")]
    style_css: PathBuf,

    /// Public site URL (canonical, sitemap, RSS).
    #[arg(long)]
    site_url: String,

    /// Site title (`<title>` and index header).
    #[arg(long)]
    site_title: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = RenderConfig {
        site_url: &args.site_url,
        site_title: &args.site_title,
        style_css: &args.style_css,
    };
    let metas = render_site(&args.posts_dir, &args.out_dir, &cfg)
        .context("render_site failed")?;
    eprintln!(
        "✓ rendered {} post(s) into {}",
        metas.len(),
        args.out_dir.display()
    );
    for m in &metas {
        eprintln!("  - {} ({})", m.slug, m.date.format("%Y-%m-%d"));
    }
    Ok(())
}
```

- [ ] **Step 2: Declare the bin in Cargo.toml**

In `crates/heartbit-cli/Cargo.toml`, add to the existing `[[bin]]` entries or alongside the `heartbit` binary:

```toml
[[bin]]
name = "heartbit_blog_render"
path = "src/bin/heartbit_blog_render.rs"
```

- [ ] **Step 3: Smoke test**

```bash
# Should compile cleanly:
cargo build --release --bin heartbit_blog_render

# Should render an empty site (no posts):
target/release/heartbit_blog_render \
  --site-url https://pascal.heartbit.ai \
  --site-title "pascal.heartbit.ai"

# Inspect:
ls -la blog-site/public/
```

Expected: `index.html`, `feed.xml`, `sitemap.xml`, `robots.txt`, `style.css` in `blog-site/public/`. Open `index.html` in a browser — should render "No posts yet."

- [ ] **Step 4: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings`

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-cli/src/bin/heartbit_blog_render.rs crates/heartbit-cli/Cargo.toml
git commit -m "feat(cli): heartbit_blog_render standalone binary for site regen"
```

---

## Task 11: Daemon Wiring — Scheduler + Handler + Command

**Files:**
- Create: `crates/heartbit/src/daemon/persona_blog.rs` — `PersonaBlogScheduler`
- Create: `crates/heartbit/src/daemon/persona_blog_handler.rs` — `handle_persona_blog` + `PersonaBlogDeps`
- Create: `crates/heartbit/src/daemon/blog_context.rs` — `BlogContext`
- Modify: `crates/heartbit/src/daemon/types.rs` — `DaemonCommand::PersonaBlog`
- Modify: `crates/heartbit/src/daemon/mod.rs` — declare + re-export
- Modify: `crates/heartbit/src/daemon/core.rs` — spawn + dispatch

Mirrors the quote-tweet daemon wiring exactly. Read `crates/heartbit/src/daemon/persona_quote.rs`, `persona_quote_handler.rs`, `quotes_context.rs` first.

- [ ] **Step 1: Add `DaemonCommand::PersonaBlog` variant + serde test**

In `crates/heartbit/src/daemon/types.rs`, add:

```rust
    /// Fire one blog-pipeline tick. Selects the highest-engagement X
    /// post from the prior `seed_lookback_days` as the topic seed,
    /// drafts an essay, routes through Telegram, writes Markdown +
    /// renders the static site.
    PersonaBlog {
        /// Persona name (e.g. `"heartbit-ghost:x"`).
        persona: String,
    },
```

And the round-trip test:

```rust
#[test]
fn persona_blog_command_round_trips() {
    let cmd = DaemonCommand::PersonaBlog {
        persona: "heartbit-ghost:x".into(),
    };
    let s = serde_json::to_string(&cmd).unwrap();
    let parsed: DaemonCommand = serde_json::from_str(&s).unwrap();
    match parsed {
        DaemonCommand::PersonaBlog { persona } => {
            assert_eq!(persona, "heartbit-ghost:x");
        }
        other => panic!("expected PersonaBlog, got {other:?}"),
    }
}
```

- [ ] **Step 2: Create `blog_context.rs`**

Create `crates/heartbit/src/daemon/blog_context.rs`:

```rust
//! Daemon-wide shared state for the blog pipeline.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use heartbit_core::CredentialResolver;
use heartbit_core::config::daemon::ActiveHoursConfig;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::PersonaRegistry;
use heartbit_ghost::blog::BlogReviewDelivery;
use heartbit_ghost::posts::TopPostsProvider;

pub struct PersonaBlogEntry {
    pub top_posts_provider: Arc<dyn TopPostsProvider>,
    pub interval: Duration,
    pub interval_jitter_pct: u32,
    pub active_hours: Option<ActiveHoursConfig>,
    pub seed_lookback_days: i64,
    pub candidates_per_draft: usize,
    pub posts_dir: PathBuf,
    pub out_dir: PathBuf,
    pub style_css: PathBuf,
    pub site_url: String,
    pub site_title: String,
    pub writer_provider: Option<Arc<BoxedProvider>>,
}

impl std::fmt::Debug for PersonaBlogEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaBlogEntry")
            .field("interval", &self.interval)
            .field("interval_jitter_pct", &self.interval_jitter_pct)
            .field("seed_lookback_days", &self.seed_lookback_days)
            .field("candidates_per_draft", &self.candidates_per_draft)
            .field("posts_dir", &self.posts_dir)
            .field("out_dir", &self.out_dir)
            .field("site_url", &self.site_url)
            .field("writer_provider_set", &self.writer_provider.is_some())
            .finish()
    }
}

pub struct BlogContext {
    pub registry: Arc<PersonaRegistry>,
    pub provider: Arc<BoxedProvider>,
    pub delivery: Arc<dyn BlogReviewDelivery>,
    pub credentials: Arc<dyn CredentialResolver>,
    pub corpora_root: PathBuf,
    pub profiles_root: PathBuf,
    /// Single entry — keyed by persona name for consistency with other
    /// contexts even though there's only one blog per daemon.
    pub entry: PersonaBlogEntry,
    pub persona_name: String,
}
```

- [ ] **Step 3: Create `persona_blog.rs` (scheduler)**

Mirror `crates/heartbit/src/daemon/persona_quote.rs` end-to-end. Replace:
- `PersonaQuoteScheduler` → `PersonaBlogScheduler`
- `&PersonaQuotesConfig` → `&PersonaBlogConfig`
- `poll_interval_seconds` (same field name) + `interval_jitter_pct` + `active_hours`
- `DaemonCommand::PersonaQuote` → `DaemonCommand::PersonaBlog`
- Test: `fires_persona_blog_after_interval` (mirror the quote test exactly, just with `PersonaBlog` and the blog config)

- [ ] **Step 4: Create `persona_blog_handler.rs` (handler)**

```rust
//! Handler for `DaemonCommand::PersonaBlog`. Selects the X-derived
//! seed, runs `run_blog_pipeline`, handles outcomes.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use heartbit_core::CredentialResolver;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::{PersonaParams, PersonaRegistry};
use heartbit_ghost::blog::{
    BlogConfig, BlogOutcome, BlogReviewDelivery, DEFAULT_TOP_N, run_blog_pipeline,
    select_blog_seed,
};
use heartbit_ghost::posts::TopPostsProvider;

pub struct PersonaBlogDeps<'a> {
    pub persona_name: &'a str,
    pub registry: &'a PersonaRegistry,
    pub top_posts_provider: &'a dyn TopPostsProvider,
    pub seed_lookback_days: i64,
    pub provider: Arc<BoxedProvider>,
    pub writer_provider: Option<Arc<BoxedProvider>>,
    pub delivery: Arc<dyn BlogReviewDelivery>,
    pub credentials: Arc<dyn CredentialResolver>,
    pub candidates_per_draft: usize,
    pub corpora_root: &'a Path,
    pub profiles_root: &'a Path,
    pub posts_dir: &'a Path,
    pub out_dir: &'a Path,
    pub style_css: &'a Path,
    pub site_url: &'a str,
    pub site_title: &'a str,
}

pub async fn handle_persona_blog(deps: PersonaBlogDeps<'_>) -> Result<()> {
    let persona = deps
        .registry
        .get(deps.persona_name)
        .ok_or_else(|| anyhow::anyhow!("persona '{}' not registered", deps.persona_name))?;
    let _expansion = persona
        .expand(&PersonaParams::default())
        .map_err(|e| anyhow::anyhow!("expand persona '{}': {e}", deps.persona_name))?;

    let seed = match select_blog_seed(
        deps.top_posts_provider,
        DEFAULT_TOP_N,
        deps.seed_lookback_days,
        Utc::now(),
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            tracing::info!(
                persona = %deps.persona_name,
                error = %e,
                "blog: no eligible seed this week — skipping tick"
            );
            return Ok(());
        }
    };

    tracing::info!(
        persona = %deps.persona_name,
        seed_tweet_id = %seed.source_tweet_id,
        seed_score = seed.engagement_score,
        "blog: selected seed"
    );

    let cfg = BlogConfig {
        persona_name: deps.persona_name,
        provider: deps.provider.clone(),
        writer_provider: deps.writer_provider.clone(),
        corpora_root: deps.corpora_root,
        profiles_root: deps.profiles_root,
        on_progress: Some(Arc::new(|s: &str| tracing::info!("blog: {s}"))),
        seed,
        candidates_per_draft: deps.candidates_per_draft,
        delivery: deps.delivery.clone(),
        credentials: deps.credentials.clone(),
        posts_dir: deps.posts_dir,
        out_dir: deps.out_dir,
        style_css: deps.style_css,
        site_url: deps.site_url,
        site_title: deps.site_title,
    };

    match run_blog_pipeline(cfg).await {
        Ok(out) => {
            tracing::info!(
                persona = %deps.persona_name,
                outcome = ?out.outcome,
                "blog pipeline complete"
            );
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
            }
        }
        Err(e) => {
            tracing::error!(
                persona = %deps.persona_name,
                error = %e,
                "blog pipeline failed"
            );
        }
    }
    Ok(())
}
```

Add a handler test mirroring the quote handler tests (unknown persona, happy path, no-seed path).

- [ ] **Step 5: Wire into `core.rs`**

In `crates/heartbit/src/daemon/core.rs`:

1. Add `blog_context: Option<Arc<BlogContext>>` field on `DaemonCore`.
2. Add `with_blog_context` builder method.
3. In the spawn block, after the quote scheduler block, add an analogous block for `PersonaBlogScheduler` (just one — single block, not a Vec).
4. In the command-dispatch match, add an arm for `DaemonCommand::PersonaBlog { persona }` that looks up the context and calls `handle_persona_blog`.

- [ ] **Step 6: Module declarations + re-exports**

In `crates/heartbit/src/daemon/mod.rs`:

```rust
pub mod blog_context;
pub mod persona_blog;
pub mod persona_blog_handler;

pub use blog_context::{BlogContext, PersonaBlogEntry};
pub use persona_blog::PersonaBlogScheduler;
pub use persona_blog_handler::{PersonaBlogDeps, handle_persona_blog};
```

In `crates/heartbit/src/lib.rs`, re-export `BlogContext` and `PersonaBlogEntry` alongside `QuotesContext`.

- [ ] **Step 7: Run tests**

Run: `cargo test --package heartbit --features daemon --lib daemon::persona_blog daemon::persona_blog_handler daemon::types::tests::persona_blog_command_round_trips`

Expected: PASS.

- [ ] **Step 8: Workspace quality gate**

Run: `cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`

Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add crates/heartbit/src/daemon/persona_blog.rs \
        crates/heartbit/src/daemon/persona_blog_handler.rs \
        crates/heartbit/src/daemon/blog_context.rs \
        crates/heartbit/src/daemon/types.rs \
        crates/heartbit/src/daemon/mod.rs \
        crates/heartbit/src/daemon/core.rs \
        crates/heartbit/src/lib.rs
git commit -m "feat(daemon): PersonaBlogScheduler + handle_persona_blog + DaemonCommand::PersonaBlog"
```

---

## Task 12: CLI Wiring — Build BlogContext at Startup

**Files:**
- Modify: `crates/heartbit-cli/src/daemon/mod.rs` — build `BlogContext`
- Create: `crates/heartbit-cli/src/persona_review.rs` (extend) — `TelegramBlogReviewDelivery`
- Modify: `crates/heartbit-cli/src/daemon/validate.rs` — validate the blog block
- Modify: `daemon-dev.toml` — example block (operator-local, gitignored)

- [ ] **Step 1: Build BlogContext at daemon startup**

In `crates/heartbit-cli/src/daemon/mod.rs`, after the QuotesContext block, add:

```rust
// --- Build BlogContext from daemon_config.persona_blog ---
let core = if let Some(ref blog_cfg) = daemon_config.persona_blog {
    if !blog_cfg.enabled {
        tracing::info!("persona_blog config present but disabled — skipping");
        core
    } else {
        // Validate the persona has post history + engagement stores
        // configured (we reuse them for seed selection). If not, bail.
        let posts_entry = posts_ctx
            .as_ref()
            .and_then(|c| c.entries.get(&blog_cfg.persona))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "persona_blog references persona '{}' but no matching \
                     [[daemon.persona_posts]] entry exists — the blog reuses \
                     the post history + engagement stores for seed selection",
                    blog_cfg.persona
                )
            })?;

        let posts_dir = expand_tilde(&blog_cfg.posts_dir)?;
        let out_dir = expand_tilde(&blog_cfg.out_dir)?;
        let style_css = std::path::PathBuf::from("blog-site/style.css");

        std::fs::create_dir_all(&posts_dir)
            .with_context(|| format!("create posts_dir at {}", posts_dir.display()))?;
        std::fs::create_dir_all(&out_dir)
            .with_context(|| format!("create out_dir at {}", out_dir.display()))?;

        let blog_delivery: Arc<dyn heartbit_ghost::blog::BlogReviewDelivery> =
            Arc::new(crate::persona_review::TelegramBlogReviewDelivery::from_env()?);

        let writer_provider: Option<Arc<heartbit::BoxedProvider>> = blog_cfg
            .writer_provider
            .as_ref()
            .map(|p| crate::build_agent_provider(p, None, None))
            .transpose()
            .with_context(|| {
                format!(
                    "build writer_provider for persona_blog '{}'",
                    blog_cfg.persona
                )
            })?;

        let entry = heartbit::PersonaBlogEntry {
            top_posts_provider: posts_entry
                .top_posts_provider
                .clone()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "persona_blog requires the matching persona_posts entry to have \
                         top_posts_provider configured (it's auto-built when \
                         post_history_store='jsonl' — set that on the posts entry)"
                    )
                })?,
            interval: std::time::Duration::from_secs(blog_cfg.poll_interval_seconds),
            interval_jitter_pct: blog_cfg.interval_jitter_pct,
            active_hours: blog_cfg.active_hours.clone(),
            seed_lookback_days: blog_cfg.seed_lookback_days,
            candidates_per_draft: blog_cfg.candidates_per_draft,
            posts_dir,
            out_dir,
            style_css,
            site_url: blog_cfg.site_url.clone(),
            site_title: blog_cfg.site_title.clone(),
            writer_provider,
        };

        let blog_ctx = Arc::new(heartbit::BlogContext {
            registry: posts_ctx.as_ref().unwrap().registry.clone(),
            provider: posts_ctx.as_ref().unwrap().provider.clone(),
            delivery: blog_delivery,
            credentials: posts_ctx.as_ref().unwrap().credentials.clone(),
            corpora_root: corpora_root.clone(),
            profiles_root: profiles_root.clone(),
            entry,
            persona_name: blog_cfg.persona.clone(),
        });

        core.map(|c| c.with_blog_context(blog_ctx))
    }
} else {
    core
};
```

- [ ] **Step 2: Add `TelegramBlogReviewDelivery` adapter**

In `crates/heartbit-cli/src/persona_review.rs`, add a parallel struct to `TelegramQuoteReviewDelivery`. The shape mirrors it — same shared dispatcher, different message rendering (full essay markdown vs. quote-tweet), different outcome reporting.

Key differences:
- Render the review message as: title-line preview + first ~300 chars of each candidate (essays are too long to ship the full body in Telegram; the operator clicks through to the rendered preview)
- Outcome report on `BlogOutcome::Posted` includes the URL of the published post
- Outcome report on `BlogOutcome::AllCandidatesGateRejected` lists the reasons

- [ ] **Step 3: Validator extension**

In `crates/heartbit-cli/src/daemon/validate.rs`, add a `validate_persona_blog` fn that checks:
- If `persona_blog` is present and enabled, the referenced `persona` slug has a matching `[[daemon.persona_posts]]` entry
- The `posts_dir` and `out_dir` parents exist (or both can be created)
- `site_url` is non-empty and starts with `https://`

Add 2-3 tests.

- [ ] **Step 4: daemon-dev.toml example block (operator-local)**

Add an example `[daemon.persona_blog]` block to `daemon-dev.toml` (gitignored). The block should be commented out by default; the operator un-comments to activate.

```toml
# Personal blog at pascal.heartbit.ai. Weekly cadence, picks the highest-
# engagement X post from the prior 7 days as the topic seed. Drafts a
# long-form essay (800-1500 words), routes through Telegram for review,
# writes Markdown + renders the static site for Cloudflare Pages.
#
# Uncomment to enable.
#
# [daemon.persona_blog]
# persona = "heartbit-ghost:x"
# enabled = true
# poll_interval_seconds = 604800           # 7 days = weekly
# interval_jitter_pct = 10
# posts_dir = "blog-site/posts"
# out_dir = "blog-site/public"
# seed_lookback_days = 7
# candidates_per_draft = 2
# site_url = "https://pascal.heartbit.ai"
# site_title = "pascal.heartbit.ai"
#
# [daemon.persona_blog.active_hours]
# start = "10:00"
# end = "12:00"
#
# # Same Grok writer override as proactive posts (optional):
# [daemon.persona_blog.writer_provider]
# name = "openrouter"
# model = "x-ai/grok-4.3"
# prompt_caching = false
```

- [ ] **Step 5: Validate live**

```bash
HEARTBIT_GHOST_OPERATOR_USER_ID=999 target/release/heartbit \
  --config daemon-dev.toml daemon --validate-config
```

Expected: clean (since the block is commented out, no validation issues).

- [ ] **Step 6: Workspace quality gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
```

Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-cli/src/daemon/mod.rs \
        crates/heartbit-cli/src/persona_review.rs \
        crates/heartbit-cli/src/daemon/validate.rs
git commit -m "feat(daemon): CLI wiring + validator + TelegramBlogReviewDelivery for persona_blog"
```

---

## Task 13: Docs + Deploy Notes

**Files:**
- Modify: `docs/operating-heartbit.md`

- [ ] **Step 1: Add a new section to operating-heartbit.md**

After the existing "Quote-tweet knobs" section, add:

````markdown
## Personal blog knobs

`[daemon.persona_blog]` controls the weekly blog pipeline. The blog reuses the X persona's post history + engagement store to seed each week's topic from the highest-engagement post in the prior 7 days.

| Knob | Default | When to change |
|---|---|---|
| `enabled` | `true` | Set `false` to pause without removing the block. |
| `poll_interval_seconds` | `604800` (7 days) | Weekly is the recommended cadence — long-form posts need accumulated X signal to seed from. Don't go shorter than 3 days. |
| `interval_jitter_pct` | `10` (±10%) | Tighter than X posts because weekly is already coarse. |
| `active_hours` | unset | Set to a narrow window (e.g. `10:00-12:00`) for predictable publish times. |
| `posts_dir` | `blog-site/posts` | Where Markdown files are written. |
| `out_dir` | `blog-site/public` | Where the rendered static site is written. This is what gets deployed. |
| `seed_lookback_days` | `7` | How far back to look for the X-derived seed. Set to `0` to disable X-seeding (rarely useful — disables the main feature). |
| `candidates_per_draft` | `2` | Long-form drafts are expensive; 2 is enough for meaningful comparison. |
| `site_url` | required | Public URL for canonical tags, RSS, sitemap. |
| `site_title` | `pascal.heartbit.ai` | Site title in `<title>` and the index header. |
| `writer_provider` | unset | Same shape as `persona_posts.writer_provider`. Falls back to global `[provider]`. |

### Prerequisite: matching `[[daemon.persona_posts]]` entry

The blog requires a matching `[[daemon.persona_posts]]` entry for the same persona slug — it reuses that entry's post history + engagement store for seed selection. If you've configured proactive posts, this is automatic. The daemon fails fast at startup with a clear error if missing.

### Deployment to Cloudflare Pages

1. Create a Cloudflare Pages project pointed at the repository.
2. Set the build directory to `blog-site/public/`. No build command needed (the daemon pre-renders).
3. Add the custom domain `pascal.heartbit.ai` in the Pages project's domain settings.
4. Each successful blog tick:
   a. The daemon writes a new Markdown file to `blog-site/posts/`.
   b. The renderer regenerates `blog-site/public/` from all posts.
   c. The daemon (or you, on next git push) commits + pushes.
   d. Cloudflare Pages auto-deploys from the push.

For now the daemon does NOT auto-commit. After a successful tick, the operator runs `git add blog-site/ && git commit -m "blog: <slug>" && git push` to trigger deploy. (Future enhancement: optional auto-commit hook.)

### Manual regen

Edit a template or fix a typo in a post? Regenerate the whole site:

```bash
target/release/heartbit_blog_render \
  --site-url https://pascal.heartbit.ai \
  --site-title pascal.heartbit.ai
```

Output lands in `blog-site/public/`. Commit + push to deploy.
````

- [ ] **Step 2: Commit**

```bash
git add docs/operating-heartbit.md
git commit -m "docs: add persona_blog section to operating-heartbit.md"
```

---

## Task 14: Final Integration Smoke + Close-out

- [ ] **Step 1: Full workspace gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
```

Expected: all green.

- [ ] **Step 2: Manual end-to-end smoke**

In the operator's terminal (NOT via the daemon):

```bash
# 1. Regenerate the empty site:
target/release/heartbit_blog_render \
  --site-url https://pascal.heartbit.ai \
  --site-title pascal.heartbit.ai

# 2. Inspect output:
ls -la blog-site/public/
# Expected: index.html, feed.xml, sitemap.xml, robots.txt, style.css

# 3. Open in browser:
xdg-open blog-site/public/index.html
# Or: cat blog-site/public/index.html
# Expected: "No posts yet." rendered cleanly with the dark theme.

# 4. Drop a hand-written Markdown post to test render:
mkdir -p blog-site/posts
cat > blog-site/posts/2026-05-16-hello.md <<'EOF'
---
title: Hello
date: 2026-05-16T12:00:00Z
slug: hello
excerpt: First post — testing the renderer end to end.
tags: [meta]
---
# This should NOT show

Just kidding — the renderer pulls the title from frontmatter, so this `# Hello` heading WILL show up. The blog_writer prompt instructs the LLM not to emit a title line, but if it does we render it harmlessly.

Body paragraph with a [link](https://example.com).
EOF

# 5. Regenerate:
target/release/heartbit_blog_render \
  --site-url https://pascal.heartbit.ai \
  --site-title pascal.heartbit.ai

# 6. Inspect:
ls blog-site/public/hello/
# Expected: index.html
cat blog-site/public/hello/index.html | grep -E "<title>|<h1>|<p>"
# Expected: title contains "Hello · pascal.heartbit.ai", h1 contains "Hello"

# 7. Validate XML:
xmllint --noout blog-site/public/feed.xml
xmllint --noout blog-site/public/sitemap.xml
# Expected: no errors

# 8. Clean up the smoke post:
rm blog-site/posts/2026-05-16-hello.md
rm -rf blog-site/public/hello
```

Manual smoke verified. Daemon integration smoke is deferred to operator (requires configuring `[daemon.persona_blog]` + restart, which requires the X engagement loop to have data — needs a few days of warm engagement-collector state).

- [ ] **Step 3: Update lessons.md if any non-obvious gotchas surfaced**

Edit `tasks/lessons.md` if applicable. Keep entries terse — one or two lines per lesson.

- [ ] **Step 4: Final commit (if step 3 produced changes)**

```bash
git add tasks/lessons.md
git commit -m "docs(lessons): personal-blog implementation gotchas"
```

---

## Verification matrix

| Spec item | Covered by |
|---|---|
| Static site generator (no platform) | Tasks 4, 7, 8 (markdown writer + templates + renderer) |
| Weekly cadence | Task 11 (scheduler default `poll_interval_seconds = 604800`) |
| X-derived topic seed (highest-engagement post in prior 7 days) | Task 6 (`select_blog_seed`) |
| Long-form writer (800-1500 words) | Task 2 (`blog_writer_recipe`, `max_tokens=4096`, word-range pinned in prompt) |
| Same voice profile as X (v5 dhh/mitsuhiko) | Inherits from existing voice rendering — no new persona, no new profile |
| Strict-sourcing chain reused | Task 2 (prompt) + Task 9 (pre-filter drops `FactVerdict::Unverifiable`) |
| Telegram review path | Task 9 (`BlogReviewDelivery` trait) + Task 12 (`TelegramBlogReviewDelivery`) |
| 3 templates (base, post, index) factored | Task 7 |
| Navigation via base.html | Task 7 (`<nav>` in `base.html`) |
| RSS + sitemap + robots.txt for SEO | Task 8 (`render_rss`, `render_sitemap`, robots written by `render_site`) |
| Standalone render binary | Task 10 |
| Cloudflare Pages deploy | Task 13 docs (no code — Pages auto-deploys from git push) |
| Operator validation | Task 12 (validator extension) |
| Backward compat with existing post/quote/reply pipelines | Strict file isolation: new modules, no edits to existing writer/reply/quote recipes |

---

## Notes for the implementer

- Tasks 6 (seed), 7 (templates), 8 (render) are independent of one another within the blog module — could be parallelized but the plan sequences them linearly for clarity.
- Task 9 (pipeline) is the largest single task — read `quote/mod.rs` end-to-end before starting. Mechanical translation produces ~1000 lines.
- Task 11 (daemon wiring) requires Task 9 to compile (depends on `BlogConfig`, `BlogReviewDelivery`, etc.).
- Task 12 (CLI) requires Task 11 (depends on `BlogContext`, `PersonaBlogEntry`).
- All commits should pass `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`. No exceptions.
- Don't bundle "while I'm here" refactors. Each task = one logical commit.
- The blog reuses the existing post history + engagement store. Don't create parallel storage.
- `topic_brief` from the proactive posts setup is unrelated — that's a separate input to the topic_generator and only affects X posts. The blog's "topic" is always the seed.
