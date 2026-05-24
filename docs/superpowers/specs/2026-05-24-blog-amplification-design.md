# Blog Amplification — Design

**Goal**: When pascal.heartbit.ai publishes a new essay, automatically (a) post an announcement X thread via the existing heartbit-ghost persona with Telegram review, and (b) refresh the operator's GitHub Profile README to feature the new essay.

**Non-goals**: LinkedIn integration (operator wants manual control); Bluesky / Mastodon (excluded by operator); newsletter (no existing list); podcast / video (out of scope this iteration); daily cron refresh of the GitHub README (on-publish is enough at weekly blog cadence).

**Surface count**: 2 (X self-amp thread, GitHub README on-publish).

---

## Architecture

```
handle_persona_blog
  └── run_blog_pipeline → BlogOutcome::Posted { post_path, post_url, … }
        ├── run_deploy_command (existing — wrangler pages deploy)
        ├── enqueue DaemonCommand::BlogAnnounceX { persona, post_url, title, excerpt, body_snippet }
        │     └── handle_blog_announce_x
        │           ├── generate thread via writer (no researcher, no fact_check)
        │           ├── Telegram review (reuse PostReviewDelivery shape)
        │           └── on Pick → publish thread via existing X client
        └── update_github_readme(handle, post_path, post_url, title, excerpt)
              ├── render README.md from template (intro + top-3 recent posts)
              ├── write to local repo working tree
              └── git add + commit + push (shell-out)
```

Failures in either amp surface are logged and swallowed — they never crash the daemon and never roll back the blog publish.

---

## Surface 2 — X self-amplification thread

### Trigger

`handle_persona_blog` extended: after `run_deploy_command` succeeds (or there is no `deploy_command` configured — i.e. the blog is reachable at `post_url`), enqueue a new Kafka command. If `deploy_command` is configured and fails, the X announcement is **skipped** — we don't want to publish a thread linking to a stale build.

```rust
DaemonCommand::BlogAnnounceX {
    persona: String,
    post_url: String,
    title: String,
    excerpt: String,
    body_snippet: String,  // first ~500 chars of body for context
}
```

The command lands in the daemon's existing Kafka consumer loop and dispatches to a new handler `handle_blog_announce_x` (mirroring `handle_persona_post` in shape).

### Thread generation

A new minimal pipeline in `crates/heartbit-ghost/src/blog/announce.rs`:

```rust
pub async fn run_x_announcement_pipeline(
    cfg: XAnnouncementConfig<'_>,
) -> Result<XAnnouncementOutput, XAnnouncementError>
```

Pipeline stages:
1. **Writer only** (no researcher, no fact_check — the source is the operator's own blog, already fact-checked through the blog pipeline). Reuses the existing `x_writer_recipe` with a specialized user-message builder.
2. **Length normalize** (reuses `length_normalize`) to enforce 280 chars/tweet across the thread.
3. **Telegram review** via the existing `XReviewDelivery` trait (no new adapter needed — the blog announcement is just another X thread from the operator's perspective).
4. **On Pick** → publish thread via existing X publish client.

Writer system prompt addendum: the writer is told this is an **announcement thread** for the operator's own essay. Format: 3-5 tweets, last tweet must include the canonical blog URL.

### Telegram review

Reuses `XReviewDelivery::deliver_and_await`. Same shared dispatcher as proactive posts / quote-tweets / replies. From the operator's perspective on Telegram, the announcement thread shows up the same way other X threads do — same Pick/Skip buttons.

### Failure modes

- Writer rejects schema 2x in a row → `AllCandidatesGateRejected`, log + skip
- Telegram review times out → log + skip, no publish
- X API publish fails → log error + send Telegram "publish failed" message
- Each failure mode is *independent of the blog publish* — the blog post stays live on pascal.heartbit.ai regardless

---

## Surface 3 — GitHub Profile README on-publish

### Trigger

Inside `handle_persona_blog`, after `run_deploy_command` succeeds (or there is no `deploy_command`), **synchronously** (not via Kafka) call:

```rust
update_github_readme(&GitHubReadmeConfig {
    local_repo_path: Path,        // /home/pleclech/projects/100-tokens-profile or similar
    blog_posts_dir: Path,         // blog-site/posts/
    site_url: &str,               // https://pascal.heartbit.ai
    bio_template_path: Path,      // path to operator-authored bio.md template
    git_author_name: &str,
    git_author_email: &str,
}).await
```

### README structure

A new module `crates/heartbit-ghost/src/github_readme.rs`:

```rust
pub async fn render_readme(
    bio_template: &str,
    recent_posts: &[BlogPostFrontmatter],  // top 3, sorted newest-first
    site_url: &str,
) -> String
```

Output structure:

```markdown
<!-- intro section: from bio_template, verbatim -->
{{ bio_template_content }}

<!-- AUTO-GENERATED: do not edit below this line -->
## Recent essays

- [{{ post.title }}]({{ site_url }}/{{ post.slug }}/) — *{{ post.excerpt }}* ({{ post.date | format_date }})
- [{{ post.title }}]({{ site_url }}/{{ post.slug }}/) — *{{ post.excerpt }}* ({{ post.date | format_date }})
- [{{ post.title }}]({{ site_url }}/{{ post.slug }}/) — *{{ post.excerpt }}* ({{ post.date | format_date }})

<sub>Auto-updated on each new essay. Source: pascal.heartbit.ai</sub>
```

The `<!-- AUTO-GENERATED: do not edit below this line -->` marker lets the operator edit the bio above it freely without losing changes — the regeneration logic preserves everything before the marker and replaces everything after.

### Reading the blog posts

`render_readme` reads `blog_posts_dir/*.md`, parses the YAML frontmatter using the existing `BlogPostFrontmatter` struct, sorts newest-first, takes the top 3.

### Git operations

After `render_readme` writes to `<local_repo_path>/README.md`, shell out:

```rust
tokio::process::Command::new("sh")
    .current_dir(&local_repo_path)
    .arg("-c")
    .arg(format!(
        "git add README.md && \
         git -c user.name='{name}' -c user.email='{email}' \
             commit -m 'profile: feature {slug}' && \
         git push origin main",
        name = git_author_name,
        email = git_author_email,
        slug = new_post_slug,
    ))
    .output()
    .await
```

5-min timeout (mirroring `run_deploy_command`). Failures logged + swallowed.

GH credentials: the operator's existing local clone must have its `origin` configured with credentials (HTTPS token or SSH key). The daemon does not manage GH auth — the user's git config does. This is intentional — same model as `deploy_command` for `wrangler`.

### Edge cases

- `local_repo_path` doesn't exist → log error + skip (don't crash)
- `bio_template_path` doesn't exist → fall back to a minimal hard-coded intro (`# {handle}\n\nMulti-agent runtime, Rust, AI infra.\n`)
- README has no `<!-- AUTO-GENERATED -->` marker (first run) → append the marker + auto-section, preserving existing content above
- No blog posts yet → render section as `_No essays yet._`

---

## Code organization

### New files

- `crates/heartbit-ghost/src/blog/announce.rs` — `run_x_announcement_pipeline` + `XAnnouncementConfig`/`Output`/`Error` types
- `crates/heartbit-ghost/src/github_readme.rs` — `render_readme` (pure fn) + `update_github_readme` (does the git shell-out)
- `crates/heartbit/src/daemon/blog_announce_x_handler.rs` — `handle_blog_announce_x` (Kafka command dispatch target)

### Modified files

- `crates/heartbit/src/daemon/persona_blog_handler.rs` — after `run_deploy_command` succeeds, enqueue `BlogAnnounceX` Kafka message + call `update_github_readme`
- `crates/heartbit/src/daemon/types.rs` — add `DaemonCommand::BlogAnnounceX` variant + serde round-trip test
- `crates/heartbit/src/daemon/core.rs` — dispatch arm for `BlogAnnounceX`
- `crates/heartbit-core/src/config/daemon.rs` — add `PersonaBlogConfig::x_announce` and `PersonaBlogConfig::github_readme` optional sub-blocks
- `crates/heartbit/src/daemon/blog_context.rs` — propagate the two new sub-configs onto `PersonaBlogEntry`
- `crates/heartbit-cli/src/daemon/mod.rs` — wire the sub-configs through `BlogContext` build
- `docs/operating-heartbit.md` — document the two new sub-blocks under "Personal blog knobs"

### No-touch

- `crates/heartbit-ghost/src/blog/mod.rs::run_blog_pipeline` — the pipeline itself doesn't change. Amplification is strictly a *post-Posted* concern handled by the daemon handler, not the pipeline.
- All existing X post / quote / reply paths — unchanged.

---

## Config

New TOML sub-blocks under `[daemon.persona_blog]`:

```toml
[daemon.persona_blog.x_announce]
enabled = true
# review_via_telegram is always true in v1 — auto-publish would be a v2 toggle
# Reuses the parent persona_blog.writer_provider (or global [provider]) — no separate provider override needed v1.

[daemon.persona_blog.github_readme]
enabled = true
local_repo_path = "/home/pleclech/projects/100-tokens-profile"
bio_template_path = "/home/pleclech/projects/100-tokens-profile/bio.md"
git_author_name = "Pascal Le Clech"
git_author_email = "pascal@heartbit.ai"
```

Both blocks are `Option<...>` — omitting either disables that surface. Default for both: not present → disabled.

---

## Testing

### Unit tests

- `render_readme` — 5 tests:
  - empty `recent_posts` → "_No essays yet._" section
  - 1 post → 1-item list
  - 5 posts → top-3 only, newest-first
  - bio template preserved verbatim above marker
  - first run (no marker in README) → appends marker + section
- `run_x_announcement_pipeline`:
  - happy path: writer produces 4-tweet thread, Telegram returns Pick(0), publish succeeds
  - writer returns >280-char tweet → `length_normalize` truncates, test asserts boundary
  - Telegram returns Skip → outcome `Skipped`, no publish
  - Telegram returns TimedOut → outcome `TimedOut`
- `handle_blog_announce_x`:
  - unknown persona → Err
  - delivery missing → log + Ok (no crash)
- `update_github_readme`:
  - missing `local_repo_path` → log + Ok (no crash)
  - missing `bio_template_path` → falls back to default bio, succeeds
  - git command failure → log + Ok

### Integration test

One end-to-end test in `crates/heartbit/src/daemon/persona_blog_handler.rs` tests:
1. Mock pipeline returns `BlogOutcome::Posted`
2. Mock Kafka producer (in-channel) captures the `BlogAnnounceX` command
3. Mock GitHub repo (tempdir) verifies README is updated
4. Assert both side-effects fire after Posted

### Quality gate

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets --features daemon -- -D warnings && cargo test --workspace --lib --features daemon
```

---

## Estimated effort

- X announce module + writer prompt + pipeline tests: 1.5d
- GitHub README module + render tests: 1d
- Handler integration + Kafka command + dispatch arm: 1d
- Config wiring + CLI startup: 0.5d
- Docs update: 0.5d

**Total**: ~4.5 days.

---

## Open questions for the operator (resolved at brainstorm time)

| Question | Answer |
|---|---|
| Telegram review on X self-amp? | Yes (mirror existing flow) |
| GitHub README daily cron? | No — on-publish only |
| GitHub repo handle | `100-tokens` (existing) |
| LinkedIn? | Out — operator wants manual control |
| Bluesky / Mastodon? | Out — excluded by operator |

---

## What this design intentionally does NOT do

- **No new `ChannelPublisher` trait abstraction**. Only 2 surfaces with different mechanics — premature abstraction.
- **No retry logic** beyond what `tokio::process::Command` + 5min timeout provides. Both amp surfaces are best-effort.
- **No deferred queue / retry on failure**. If GitHub push fails, next blog publish overwrites the README anyway — self-healing.
- **No auto-publish path for X self-amp** in v1. Always Telegram-reviewed. A v2 toggle could enable auto-publish later.
- **No analytics** on what published / engagement metrics. The existing `PostHistoryStore` already records X post outcomes.
- **No portrait / hero image generation** for posts. Visual content layer is out of scope this iteration.
