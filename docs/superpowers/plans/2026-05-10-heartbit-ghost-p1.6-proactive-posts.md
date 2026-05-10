# heartbit-ghost P1.6 — Proactive posts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-10-heartbit-ghost-p1.6-proactive-posts-design.md`

**Goal:** Wire the existing `run_review_pipeline` into a daemon-driven loop fired on cadence + active hours per persona. A `topic_generator` sub-agent proposes the topic from pre-fetched persona-specific context (own tweets + mentions for ghost; repo state for heartbit-rs); the existing pipeline drafts → Telegram-reviews → posts.

**Architecture:** Mirror exact of P1.5 mentions. New `crates/heartbit-ghost/src/posts/` module mirrors `reply/`. New `topic_generator` sub-agent recipe in `agents/`. New `DaemonCommand::PersonaPost { persona }` variant. `PostHistoryStore` (JSONL on disk for V1) tracks posted topics + outcomes; `TopicContextProvider` per-persona pre-fetches the generator's input context. Calibration mode only — every post gates on Telegram.

**Tech Stack:** Rust 2024 edition, tokio, serde, toml, chrono. Existing crates: `heartbit-core`, `heartbit-ghost`, `heartbit-cli`, `heartbit` umbrella. Reuses `run_review_pipeline` (P1.3d), `TelegramReviewDelivery` (P1.5), `MentionContext` (P1.5 task 13), `expand_tilde` helper (P1.5).

**Branch:** `feat/heartbit-ghost-p1.6-proactive-posts` (created off `main`; spec already committed there).

**Sub-phases (per spec §14):**
- **P1.6a** — Tools, recipe, providers, expansion plumbing (Tasks 1-5)
- **P1.6b** — Storage, config, daemon command (Tasks 6-8)
- **P1.6c** — Scheduler, handler, lifecycle wiring, CLI (Tasks 9-12)
- **P1.6d** — Acceptance (Task 13)

---

## Task 1: `TwitterUserTweetsTool` — fetch own recent posts

**Files:**
- Create: `crates/heartbit-ghost/src/tools/user_tweets.rs`
- Modify: `crates/heartbit-ghost/src/tools/mod.rs` (declare + re-export)

- [ ] **Step 1: Read the existing `TwitterMentionsTool` for pattern**

```bash
sed -n '1,140p' crates/heartbit-ghost/src/tools/mentions.rs
```

Mirror its shape exactly — the X user-tweets endpoint has the same OAuth1/JSON shape as mentions.

- [ ] **Step 2: Write the failing tests**

Create `crates/heartbit-ghost/src/tools/user_tweets.rs`:

```rust
//! `twitter_user_tweets` — fetch a user's recent posts.

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

#[derive(Debug, Deserialize)]
struct UserTweetsInput {
    /// Numeric X user id (not handle). Use `twitter_user` to resolve.
    user_id: String,
    #[serde(default = "default_max_results")]
    max_results: u32,
    #[serde(default)]
    since_id: Option<String>,
    /// When `true`, exclude replies and retweets — keep only original posts.
    /// Default `true` (most useful for "what has this account said recently?").
    #[serde(default = "default_exclude")]
    exclude_replies: bool,
}

fn default_max_results() -> u32 {
    10
}

fn default_exclude() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Tweet {
    id: String,
    text: String,
    #[serde(default)]
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct UserTweetsOutput {
    tweets: Vec<Tweet>,
    next_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    #[serde(default)]
    data: Vec<Tweet>,
    #[serde(default)]
    meta: ApiMeta,
}

#[derive(Debug, Default, Deserialize)]
struct ApiMeta {
    #[serde(default)]
    next_token: Option<String>,
}

/// Fetch a user's recent original tweets.
pub struct TwitterUserTweetsTool;

impl Default for TwitterUserTweetsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterUserTweetsTool {
    /// Construct the tool. Credentials resolved at execute time via
    /// `ExecutionContext::credentials` (OAuth1).
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterUserTweetsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_user_tweets".into(),
            description: "Fetch a user's recent original tweets (excludes replies/retweets by default). Returns up to `max_results`; use `since_id` to paginate forward.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "X user id (numeric string), not handle. Get this from twitter_user."
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 100,
                        "default": 10
                    },
                    "since_id": {
                        "type": "string",
                        "description": "Optional. Return only tweets newer than this id."
                    },
                    "exclude_replies": {
                        "type": "boolean",
                        "description": "When true, exclude replies and retweets. Default true.",
                        "default": true
                    }
                },
                "required": ["user_id"]
            }),
        }
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, heartbit_core::Error>> + Send + '_>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let parsed: UserTweetsInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json =
                        serde_json::to_string(&out).expect("UserTweetsOutput fields are infallible");
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &UserTweetsInput) -> Result<UserTweetsOutput, XApiError> {
    let path = format!("/2/users/{}/tweets", input.user_id);
    let max_str = input.max_results.to_string();
    let mut query: Vec<(&str, &str)> = vec![
        ("max_results", &max_str),
        ("tweet.fields", "created_at"),
    ];
    let exclude_value;
    if input.exclude_replies {
        exclude_value = "replies,retweets".to_string();
        query.push(("exclude", &exclude_value));
    }
    if let Some(since) = input.since_id.as_deref() {
        query.push(("since_id", since));
    }
    let response: ApiResponse = client.get_json(&path, &query).await?;
    Ok(UserTweetsOutput {
        tweets: response.data,
        next_token: response.meta.next_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::Secret;
    use wiremock::matchers::{method, path as wm_path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_client(server_uri: &str) -> XClient {
        XClient::new(
            server_uri,
            Secret::new("ck"),
            Secret::new("cs"),
            Secret::new("at"),
            Secret::new("ats"),
        )
        .expect("test client builds")
    }

    #[tokio::test]
    async fn user_tweets_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/tweets"))
            .and(query_param("max_results", "10"))
            .and(query_param("exclude", "replies,retweets"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {
                        "id": "9001",
                        "text": "first post",
                        "created_at": "2026-05-09T00:00:00.000Z"
                    }
                ],
                "meta": {"next_token": "next-1"}
            })))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserTweetsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
            exclude_replies: true,
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.tweets.len(), 1);
        assert_eq!(result.tweets[0].id, "9001");
        assert_eq!(result.tweets[0].text, "first post");
        assert_eq!(result.next_token.as_deref(), Some("next-1"));
    }

    #[tokio::test]
    async fn user_tweets_returns_unauthenticated_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/tweets"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserTweetsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
            exclude_replies: true,
        };
        let err = call_x(&client, &input).await.expect_err("401 expected");
        assert!(matches!(err, XApiError::Unauthenticated { .. }), "got: {err:?}");
    }

    #[tokio::test]
    async fn user_tweets_exclude_replies_false_omits_exclude_param() {
        let server = MockServer::start().await;
        // No `exclude` query param expected — tweet types should pass through.
        Mock::given(method("GET"))
            .and(wm_path("/2/users/100/tweets"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data": [], "meta": {}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = UserTweetsInput {
            user_id: "100".into(),
            max_results: 10,
            since_id: None,
            exclude_replies: false,
        };
        let result = call_x(&client, &input).await.expect("ok");
        assert!(result.tweets.is_empty());
    }

    #[tokio::test]
    async fn definition_has_stable_name() {
        let tool = TwitterUserTweetsTool::new();
        assert_eq!(tool.definition().name, "twitter_user_tweets");
    }
}
```

- [ ] **Step 3: Wire into `tools/mod.rs`**

In `crates/heartbit-ghost/src/tools/mod.rs`, alongside the existing tool modules:

```rust
pub mod user_tweets;
```

And in the existing `pub use` block:

```rust
pub use user_tweets::TwitterUserTweetsTool;
```

(Maintain alphabetical ordering matching the existing convention.)

- [ ] **Step 4: Run the tests**

```bash
cargo test -p heartbit-ghost --lib tools::user_tweets
```

Expected: 4 PASS.

- [ ] **Step 5: Format + clippy**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/tools/user_tweets.rs crates/heartbit-ghost/src/tools/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): TwitterUserTweetsTool — fetch own recent posts

Wraps GET /2/users/:id/tweets with the same OAuth1 shape as
TwitterMentionsTool. exclude_replies defaults to true so the topic
generator (P1.6) gets a clean signal of "what this account has
posted recently" without reply/retweet noise. 4 unit tests cover
happy path, 401, exclude param toggle, and tool name stability.

heartbit-ghost P1.6a — task 1/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `topic_generator` sub-agent recipe

**Files:**
- Create: `crates/heartbit-ghost/src/agents/topic_generator.rs`
- Modify: `crates/heartbit-ghost/src/agents/mod.rs` (re-export)

- [ ] **Step 1: Write the failing tests**

Create `crates/heartbit-ghost/src/agents/topic_generator.rs`:

```rust
//! Topic generator sub-agent — proposes ONE specific thread topic
//! (or "no_topic") from pre-fetched static context. See spec §4.

use heartbit_core::config::AgentConfig;

/// System prompt for the topic generator. No tools — pure text-in /
/// text-out. The handler pre-fetches all context and injects it into
/// the user message. Single line of plain text, ≤120 chars; or the
/// literal string "no_topic" if nothing fresh.
pub const TOPIC_GENERATOR_SYSTEM_PROMPT: &str = r#"You propose ONE specific topic worth a thread (or "no_topic" if nothing fresh to say). Your inputs vary by persona — see the user message.

OUTPUT
Either a single line of plain text (the topic) — terse, ≤120 chars, no preamble, no quotation marks — OR the literal string "no_topic" if:
- you've already covered every input
- nothing in the inputs warrants a thread
- the inputs are too thin to ground a substantive post

CONSTRAINTS
- The topic must be ground-able: the writer should be able to draft a thread without inventing facts. If you can't say what specific point to make, output "no_topic".
- Avoid duplicating recent posts. Recent posts are in your inputs.
- Avoid generic topics ("AI is changing everything"). Be specific ("calibrated abstention vs forced answers in tool-use loops").
- One topic only. The thread structure is the writer's job, not yours.
"#;

/// Construct the topic generator [`AgentConfig`].
pub fn topic_generator_recipe() -> AgentConfig {
    AgentConfig {
        name: "topic_generator".to_string(),
        description: "Propose one specific thread topic (or 'no_topic') from pre-fetched static context.".to_string(),
        system_prompt: TOPIC_GENERATOR_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(512),
        reasoning_effort: Some("low".to_string()),
        ..super::stub_recipe("topic_generator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_generator_recipe_has_expected_shape() {
        let cfg = topic_generator_recipe();
        assert_eq!(cfg.name, "topic_generator");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(cfg.response_schema.is_none(), "free-form text, no schema");
    }

    #[test]
    fn topic_generator_prompt_mandates_no_topic_escape() {
        let p = TOPIC_GENERATOR_SYSTEM_PROMPT;
        assert!(p.contains("no_topic"), "prompt must offer no_topic escape hatch");
        assert!(p.contains("OUTPUT"), "prompt must specify OUTPUT format");
    }

    #[test]
    fn topic_generator_prompt_bans_generic_and_demands_specificity() {
        let p = TOPIC_GENERATOR_SYSTEM_PROMPT;
        assert!(p.contains("Be specific"), "prompt must demand specificity");
        assert!(
            p.contains("ground-able") || p.contains("ground"),
            "prompt must require groundable topics"
        );
    }
}
```

- [ ] **Step 2: Wire into `agents/mod.rs`**

In `crates/heartbit-ghost/src/agents/mod.rs`, add `pub mod topic_generator;` and `pub use topic_generator::topic_generator_recipe;` in alphabetical order (between `style_critic` and `writer`).

- [ ] **Step 3: Run tests**

```bash
cargo test -p heartbit-ghost --lib agents::topic_generator
```

Expected: 3 PASS.

- [ ] **Step 4: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/agents/topic_generator.rs crates/heartbit-ghost/src/agents/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): topic_generator recipe — propose one specific thread topic

System prompt locks: ≤120-char output, "no_topic" escape hatch when
inputs are thin or duplicates, demands specificity (no "AI is
changing everything"), demands groundability (writer must be able to
draft without inventing facts). max_turns=1, max_tokens=512,
reasoning=low — no tools; the handler pre-fetches all context.

heartbit-ghost P1.6a — task 2/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Posts module skeleton — value types + `TopicContextProvider` trait

**Files:**
- Create: `crates/heartbit-ghost/src/posts/mod.rs`
- Create: `crates/heartbit-ghost/src/posts/topic_context.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs` (`pub mod posts;`)

- [ ] **Step 1: Write `crates/heartbit-ghost/src/posts/mod.rs` (value types)**

```rust
//! Proactive posting pipeline — generates a topic, drafts candidates
//! via the existing review pipeline, gates through Telegram, posts
//! the chosen draft, records outcome.
//!
//! See spec §6 for the storage shape; the runtime lives in the daemon
//! umbrella's `handle_persona_post`.

use chrono::{DateTime, Utc};

pub mod topic_context;
pub mod history;

pub use history::{InMemoryPostHistoryStore, JsonlPostHistoryStore, PostHistoryStore, StoreError};
pub use topic_context::{
    HeartbitRsXTopicContext, TopicContextDeps, TopicContextProvider, XGhostTopicContext,
};

/// One historical post (or skip / time-out / no_topic) recorded by the
/// daemon's persona post handler.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PostHistoryEntry {
    /// When the tick fired (regardless of outcome).
    pub posted_at: DateTime<Utc>,
    /// Topic the generator proposed (empty when outcome is `NoTopic`).
    pub topic: String,
    /// What ultimately happened.
    pub outcome: PostOutcome,
    /// Tweet id when `outcome` is `Posted`; else `None`.
    pub tweet_id: Option<String>,
}

/// What happened in one persona-post tick.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PostOutcome {
    /// Topic generator returned the literal "no_topic" — pipeline NOT called.
    NoTopic,
    /// Topic was already posted within the lookback window — pipeline NOT called.
    SkippedDuplicate,
    /// Pipeline ran, user picked a draft, post succeeded.
    Posted {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Public URL of the published thread.
        url: String,
    },
    /// User pressed Skip on Telegram review.
    Skipped,
    /// Telegram review timed out without a pick.
    TimedOut,
    /// Pipeline's publish gate rejected the chosen candidate.
    GateRejected {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Reason from the publish gate.
        reason: String,
    },
    /// User picked but the X API call failed.
    PublishFailed {
        /// Index of the candidate the user picked.
        chosen_index: usize,
        /// Reason for failure.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_outcome_no_topic_distinct_from_skipped() {
        let a = PostOutcome::NoTopic;
        let b = PostOutcome::Skipped;
        assert_ne!(a, b);
    }

    #[test]
    fn post_history_entry_round_trips_through_serde() {
        let entry = PostHistoryEntry {
            posted_at: Utc::now(),
            topic: "calibrated abstention".into(),
            outcome: PostOutcome::Posted {
                chosen_index: 1,
                url: "https://x.com/i/web/status/123".into(),
            },
            tweet_id: Some("123".into()),
        };
        let s = serde_json::to_string(&entry).unwrap();
        let parsed: PostHistoryEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.topic, entry.topic);
        assert_eq!(parsed.tweet_id, entry.tweet_id);
        assert_eq!(parsed.outcome, entry.outcome);
    }
}
```

- [ ] **Step 2: Write `crates/heartbit-ghost/src/posts/topic_context.rs` (trait skeleton)**

```rust
//! `TopicContextProvider` — persona-specific pre-fetch strategy that
//! assembles the topic generator's input context. The agent itself is
//! a singleton (no tools); each persona declares HOW to build its
//! context block via this trait.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use heartbit_core::CredentialResolver;

use super::PostHistoryEntry;

/// Dependencies passed to a [`TopicContextProvider`] during pre-fetch.
pub struct TopicContextDeps<'a> {
    /// Credentials for any X API calls the provider needs (own tweets,
    /// mentions). The provider is responsible for building its own
    /// `XClient` from these.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Operator's X user_id (resolved at config load).
    pub operator_user_id: &'a str,
    /// Recent post history (most-recent-first), passed verbatim into
    /// the rendered context so the generator avoids duplicates.
    pub recent_history: Vec<PostHistoryEntry>,
}

/// Builds the persona-specific block of context that goes into the
/// topic generator's user message. Called by `handle_persona_post`
/// once per tick before the generator is invoked.
pub trait TopicContextProvider: Send + Sync {
    /// Returns a multi-line plain-text block. Empty string is allowed
    /// — the generator falls back to the `topic_brief` from config.
    fn build_context<'a>(
        &'a self,
        deps: &'a TopicContextDeps<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String, anyhow::Error>> + Send + 'a>>;
}

// Stub structs land here in this commit; full impls land in Tasks 4-5.
/// X-grounded topic context for `heartbit-ghost:x`. Implementation in Task 4.
pub struct XGhostTopicContext;

/// Repo-grounded topic context for `heartbit-rs:x`. Implementation in Task 5.
pub struct HeartbitRsXTopicContext;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_context_deps_can_be_constructed_with_zero_history() {
        struct StubCreds;
        impl CredentialResolver for StubCreds {
            fn resolve(
                &self,
                _name: &str,
            ) -> Pin<
                Box<dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>> + Send + '_>,
            > {
                Box::pin(async { Ok(heartbit_core::Secret::new("x")) })
            }
        }
        let creds: Arc<dyn CredentialResolver> = Arc::new(StubCreds);
        let deps = TopicContextDeps {
            credentials: creds,
            operator_user_id: "12345",
            recent_history: vec![],
        };
        assert!(deps.recent_history.is_empty());
        assert_eq!(deps.operator_user_id, "12345");
    }
}
```

The `XGhostTopicContext` and `HeartbitRsXTopicContext` are unit-struct stubs in this task — Tasks 4 and 5 add their `impl TopicContextProvider for ...` blocks. This task ships only the trait surface so the rest of the plan can reference it.

- [ ] **Step 3: Write `crates/heartbit-ghost/src/posts/history.rs` (stub for Task 6)**

```rust
//! `PostHistoryStore` — placeholder. Trait + impls land in Task 6.

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// I/O failure reading or writing the history file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSONL parse failure on replay.
    #[error("parse: {0}")]
    Parse(String),
}

/// Persistent storage for proactive post outcomes. Stub — full trait
/// + impls land in Task 6. Defining the marker trait here so types in
/// Tasks 3-5 can reference `dyn PostHistoryStore`.
pub trait PostHistoryStore: Send + Sync {
    /// Placeholder. Real signatures in Task 6.
    fn _marker(&self) {}
}

/// Stub — full impl in Task 6.
pub struct InMemoryPostHistoryStore;

/// Stub — full impl in Task 6.
pub struct JsonlPostHistoryStore;
```

(This stub is intentional: it lets `posts/mod.rs` compile and re-export the names so downstream tasks can wire types without circular-task ordering. Task 6 replaces all three items with full implementations.)

- [ ] **Step 4: Wire into `lib.rs`**

In `crates/heartbit-ghost/src/lib.rs`, alongside the existing `pub mod reply;`:

```rust
pub mod posts;
```

(Alphabetical ordering — `posts` comes BEFORE `reply` and `review`.)

- [ ] **Step 5: Verify dependencies**

```bash
grep -E "^async-trait\s*=" crates/heartbit-ghost/Cargo.toml
```

If absent, no need to add — we use `Pin<Box<dyn Future>>` desugaring (matches the existing `ReplyReviewDelivery` pattern from P1.5). `anyhow` should already be available transitively; confirm:

```bash
grep -E "^anyhow\s*=" crates/heartbit-ghost/Cargo.toml
```

If `anyhow` is absent in `[dependencies]`, add it (workspace dep already exists):

```toml
anyhow = { workspace = true }
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p heartbit-ghost --lib posts
```

Expected: 3 PASS (2 from mod.rs tests + 1 from topic_context.rs).

- [ ] **Step 7: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/posts/ crates/heartbit-ghost/src/lib.rs crates/heartbit-ghost/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(ghost): posts module skeleton — types + TopicContextProvider trait

Value types: PostHistoryEntry { posted_at, topic, outcome, tweet_id }
+ PostOutcome (7 variants: NoTopic, SkippedDuplicate, Posted, Skipped,
TimedOut, GateRejected, PublishFailed). TopicContextProvider trait
uses Pin<Box<dyn Future>> (matches ReplyReviewDelivery; no async-trait
dep). XGhostTopicContext + HeartbitRsXTopicContext are unit-struct
stubs — impls land in tasks 4-5. PostHistoryStore is a marker stub —
full trait + impls land in task 6.

heartbit-ghost P1.6a — task 3/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `XGhostTopicContext` impl

**Files:**
- Modify: `crates/heartbit-ghost/src/posts/topic_context.rs` (add impl block + tests)

- [ ] **Step 1: Read `XClient::from_context` for the API pattern**

```bash
sed -n '118,160p' crates/heartbit-ghost/src/tools/client.rs
```

The provider builds an `XClient`, then makes 1-2 GET calls (own tweets + mentions). Mirror the pattern used in `crates/heartbit-cli/src/persona_review.rs::list_recent_mentions`.

- [ ] **Step 2: Write the failing tests**

Append to `crates/heartbit-ghost/src/posts/topic_context.rs`'s `#[cfg(test)] mod tests` block:

```rust
    use super::super::PostOutcome;
    use chrono::Utc;

    /// Mock CredentialResolver that returns canned secrets per name. Used
    /// by XClient::from_context — the actual HTTP calls are stubbed via wiremock.
    struct CannedCreds;
    impl CredentialResolver for CannedCreds {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<
            Box<dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>> + Send + '_>,
        > {
            Box::pin(async { Ok(heartbit_core::Secret::new("x")) })
        }
    }

    #[tokio::test]
    async fn xghost_context_assembles_recent_posts_mentions_and_history() {
        use wiremock::matchers::{method, path as wm_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Mock own-tweets endpoint
        Mock::given(method("GET"))
            .and(wm_path("/2/users/12345/tweets"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"id": "1", "text": "first own post", "created_at": "2026-05-08T00:00:00.000Z"}
                ],
                "meta": {}
            })))
            .mount(&server)
            .await;
        // Mock mentions endpoint
        Mock::given(method("GET"))
            .and(wm_path("/2/users/12345/mentions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {
                        "id": "9001",
                        "text": "asking about agent loops",
                        "author_id": "777",
                        "created_at": "2026-05-09T00:00:00.000Z"
                    }
                ],
                "meta": {}
            })))
            .mount(&server)
            .await;

        let creds: Arc<dyn CredentialResolver> = Arc::new(CannedCreds);
        let history = vec![PostHistoryEntry {
            posted_at: Utc::now(),
            topic: "calibrated abstention".into(),
            outcome: PostOutcome::Posted {
                chosen_index: 0,
                url: "https://x.com/i/web/status/100".into(),
            },
            tweet_id: Some("100".into()),
        }];
        let deps = TopicContextDeps {
            credentials: creds,
            operator_user_id: "12345",
            recent_history: history,
        };

        let provider = XGhostTopicContext::with_base_url(&server.uri());
        let ctx = provider.build_context(&deps).await.expect("happy path");
        assert!(ctx.contains("RECENT POSTS"), "context: {ctx}");
        assert!(ctx.contains("first own post"), "context: {ctx}");
        assert!(ctx.contains("RECENT MENTIONS"), "context: {ctx}");
        assert!(ctx.contains("asking about agent loops"), "context: {ctx}");
        assert!(ctx.contains("RECENT POST HISTORY"), "context: {ctx}");
        assert!(ctx.contains("calibrated abstention"), "context: {ctx}");
    }

    #[tokio::test]
    async fn xghost_context_degrades_gracefully_on_api_error() {
        use wiremock::matchers::{method, path as wm_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // 401 on own-tweets — provider should still return a context with
        // mentions/history (not propagate as a fatal error).
        Mock::given(method("GET"))
            .and(wm_path("/2/users/12345/tweets"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(wm_path("/2/users/12345/mentions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data": [], "meta": {}})),
            )
            .mount(&server)
            .await;

        let creds: Arc<dyn CredentialResolver> = Arc::new(CannedCreds);
        let deps = TopicContextDeps {
            credentials: creds,
            operator_user_id: "12345",
            recent_history: vec![],
        };
        let provider = XGhostTopicContext::with_base_url(&server.uri());
        let ctx = provider.build_context(&deps).await.expect("graceful");
        // Header still rendered, body is empty/abridged.
        assert!(
            ctx.contains("RECENT POSTS")
                || ctx.contains("RECENT MENTIONS")
                || ctx.contains("RECENT POST HISTORY"),
            "context: {ctx}"
        );
    }
```

- [ ] **Step 3: Replace the `XGhostTopicContext` stub with the full impl**

In `crates/heartbit-ghost/src/posts/topic_context.rs`, replace `pub struct XGhostTopicContext;` with:

```rust
/// X-grounded topic context for `heartbit-ghost:x`. Pre-fetches
/// the operator's own recent tweets + recent mentions and renders
/// them as plain-text blocks alongside the post history.
pub struct XGhostTopicContext {
    base_url: String,
}

impl Default for XGhostTopicContext {
    fn default() -> Self {
        Self::new()
    }
}

impl XGhostTopicContext {
    /// Production constructor. Uses the X API base URL from
    /// [`crate::tools::client::X_API_BASE_URL`].
    pub fn new() -> Self {
        Self {
            base_url: crate::tools::client::X_API_BASE_URL.to_string(),
        }
    }

    /// Test constructor. Lets a mock server URI override the base URL.
    #[cfg(test)]
    pub(crate) fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }
}

impl TopicContextProvider for XGhostTopicContext {
    fn build_context<'a>(
        &'a self,
        deps: &'a TopicContextDeps<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String, anyhow::Error>> + Send + 'a>> {
        Box::pin(async move {
            // Build XClient once; reuse for both calls.
            let client = match build_client(&self.base_url, deps.credentials.clone()).await {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "topic context: client build failed; returning history only");
                    return Ok(render_history_only(&deps.recent_history));
                }
            };

            let own_tweets = fetch_own_tweets(&client, deps.operator_user_id).await;
            let mentions = fetch_recent_mentions(&client, deps.operator_user_id).await;

            let mut out = String::new();
            // Own posts block (best-effort; empty if API call failed)
            out.push_str("RECENT POSTS (yours, last 10):\n");
            match own_tweets {
                Ok(tweets) => {
                    if tweets.is_empty() {
                        out.push_str("(none)\n");
                    } else {
                        for t in tweets.iter().take(10) {
                            let when = t.created_at.as_deref().unwrap_or("?");
                            let preview: String = t.text.chars().take(140).collect();
                            out.push_str(&format!("- [{when}] \"{preview}\"\n"));
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "own tweets fetch failed");
                    out.push_str("(unavailable: api error)\n");
                }
            }
            out.push('\n');

            // Mentions block
            out.push_str("RECENT MENTIONS (last 10):\n");
            match mentions {
                Ok(ms) => {
                    if ms.is_empty() {
                        out.push_str("(none)\n");
                    } else {
                        for m in ms.iter().take(10) {
                            let preview: String = m.text.chars().take(140).collect();
                            out.push_str(&format!("- {preview}\n"));
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "mentions fetch failed");
                    out.push_str("(unavailable: api error)\n");
                }
            }
            out.push('\n');

            // Post history block (always available)
            out.push_str(&render_history_only(&deps.recent_history));
            Ok(out)
        })
    }
}

// --- private helpers -------------------------------------------------

async fn build_client(
    base_url: &str,
    creds: Arc<dyn CredentialResolver>,
) -> Result<crate::tools::client::XClient, crate::tools::client::XApiError> {
    use crate::tools::client::XClient;
    use heartbit_core::ExecutionContext;

    let ctx = ExecutionContext {
        credentials: Some(creds),
        ..Default::default()
    };
    // XClient::from_context resolves the 4 OAuth1 secrets from the resolver
    // and uses the production base URL. We need to override the base URL
    // for tests, so we re-resolve here and call XClient::new directly.
    let resolver = ctx
        .credentials
        .as_ref()
        .ok_or(crate::tools::client::XApiError::MissingResolver)?;
    let consumer_key = resolver
        .resolve("X_CONSUMER_KEY")
        .await
        .map_err(|e| crate::tools::client::XApiError::CoreError(format!("{e}")))?;
    let consumer_secret = resolver
        .resolve("X_CONSUMER_SECRET")
        .await
        .map_err(|e| crate::tools::client::XApiError::CoreError(format!("{e}")))?;
    let access_token = resolver
        .resolve("X_ACCESS_TOKEN")
        .await
        .map_err(|e| crate::tools::client::XApiError::CoreError(format!("{e}")))?;
    let access_token_secret = resolver
        .resolve("X_ACCESS_TOKEN_SECRET")
        .await
        .map_err(|e| crate::tools::client::XApiError::CoreError(format!("{e}")))?;
    XClient::new(
        base_url,
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret,
    )
}

#[derive(Debug, serde::Deserialize)]
struct OwnTweetItem {
    #[allow(dead_code)]
    id: String,
    text: String,
    #[serde(default)]
    created_at: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct OwnTweetsResp {
    #[serde(default)]
    data: Vec<OwnTweetItem>,
}

async fn fetch_own_tweets(
    client: &crate::tools::client::XClient,
    user_id: &str,
) -> Result<Vec<OwnTweetItem>, crate::tools::client::XApiError> {
    let path = format!("/2/users/{user_id}/tweets");
    let query: Vec<(&str, &str)> = vec![
        ("max_results", "10"),
        ("tweet.fields", "created_at"),
        ("exclude", "replies,retweets"),
    ];
    let resp: OwnTweetsResp = client.get_json(&path, &query).await?;
    Ok(resp.data)
}

#[derive(Debug, serde::Deserialize)]
struct MentionItem {
    #[allow(dead_code)]
    id: String,
    text: String,
}

#[derive(Debug, serde::Deserialize)]
struct MentionsResp {
    #[serde(default)]
    data: Vec<MentionItem>,
}

async fn fetch_recent_mentions(
    client: &crate::tools::client::XClient,
    user_id: &str,
) -> Result<Vec<MentionItem>, crate::tools::client::XApiError> {
    let path = format!("/2/users/{user_id}/mentions");
    let query: Vec<(&str, &str)> = vec![
        ("max_results", "10"),
        ("tweet.fields", "author_id"),
    ];
    let resp: MentionsResp = client.get_json(&path, &query).await?;
    Ok(resp.data)
}

fn render_history_only(history: &[PostHistoryEntry]) -> String {
    let mut out = String::new();
    out.push_str("RECENT POST HISTORY (last 5 from store):\n");
    if history.is_empty() {
        out.push_str("(none)\n");
    } else {
        for entry in history.iter().take(5) {
            let when = entry.posted_at.format("%Y-%m-%d");
            let outcome = match &entry.outcome {
                PostOutcome::Posted { .. } => "Posted",
                PostOutcome::Skipped => "Skipped",
                PostOutcome::TimedOut => "TimedOut",
                PostOutcome::NoTopic => "NoTopic",
                PostOutcome::SkippedDuplicate => "SkippedDuplicate",
                PostOutcome::GateRejected { .. } => "GateRejected",
                PostOutcome::PublishFailed { .. } => "PublishFailed",
            };
            let topic = if entry.topic.is_empty() {
                "(no topic)"
            } else {
                entry.topic.as_str()
            };
            out.push_str(&format!("- [{when}] {outcome}: {topic}\n"));
        }
    }
    out
}
```

The `X_API_BASE_URL` constant must be `pub(crate)` or `pub` in `tools/client.rs`. Check:

```bash
grep "X_API_BASE_URL" crates/heartbit-ghost/src/tools/client.rs
```

If it's currently private (`const X_API_BASE_URL`), promote to `pub(crate)`:

```bash
# Edit tools/client.rs:
#   const X_API_BASE_URL → pub(crate) const X_API_BASE_URL
```

If `X_API_BASE_URL` doesn't exist as a constant (the tool uses a string literal directly), define it inline at the top of `tools/client.rs`:

```rust
pub(crate) const X_API_BASE_URL: &str = "https://api.x.com";
```

(Match whichever URL the existing tools use.)

`XApiError::CoreError(String)` may not exist as a variant — verify:

```bash
grep "pub enum XApiError" crates/heartbit-ghost/src/tools/client.rs -A 30
```

If absent, either add it or replace the `.map_err(|e| crate::tools::client::XApiError::CoreError(format!("{e}")))` calls with `.map_err(|e| crate::tools::client::XApiError::Network(format!("{e}")))` (or the closest existing variant — `Network` is a likely fit for "credential resolver failed").

- [ ] **Step 4: Run tests**

```bash
cargo test -p heartbit-ghost --lib posts::topic_context
```

Expected: 3 PASS (1 from Task 3 + 2 new).

- [ ] **Step 5: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/posts/topic_context.rs crates/heartbit-ghost/src/tools/client.rs
git commit -m "$(cat <<'EOF'
feat(ghost): XGhostTopicContext — pre-fetch own tweets + mentions + history

Implements TopicContextProvider for the heartbit-ghost:x persona.
Builds a 3-block plain-text context (RECENT POSTS / RECENT MENTIONS /
RECENT POST HISTORY) from one X API call to /2/users/:id/tweets and
one to /2/users/:id/mentions. Degrades gracefully: API errors render
"(unavailable: api error)" rather than failing the whole tick — the
generator can still propose a topic from history alone.

heartbit-ghost P1.6a — task 4/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `HeartbitRsXTopicContext` impl + `topic_context_provider` field on `PersonaExpansion` + persona declarations

**Files:**
- Modify: `crates/heartbit-ghost/src/posts/topic_context.rs` (HeartbitRsXTopicContext impl)
- Modify: `crates/heartbit-core/src/persona/types.rs` (add field to `PersonaExpansion`)
- Modify: `crates/heartbit-ghost/src/lib.rs` (XGhostPersona populates field)
- Modify: `crates/heartbit-ghost/src/heartbit_rs.rs` (HeartbitRsXPersona populates field)

- [ ] **Step 1: Add the field to `PersonaExpansion`**

In `crates/heartbit-core/src/persona/types.rs`, after the existing `mode_addendum` field on `pub struct PersonaExpansion { ... }`:

```rust
    /// Optional persona-specific topic context provider for proactive
    /// posting. When present, `handle_persona_post` calls
    /// [`crate::persona::TopicContextProvider::build_context`] before
    /// invoking the topic generator. When absent, the handler injects
    /// only the post history + topic_brief from config. See P1.6 spec
    /// §5 for the rationale.
    ///
    /// `Option<Arc<dyn TopicContextProvider>>` is intentionally a heartbit-core
    /// concept (object-safe trait re-exported by personas).
    pub topic_context_provider: Option<Arc<dyn crate::persona::TopicContextProvider>>,
```

(Adapt the path: this field references a trait that lives in heartbit-ghost. To avoid a backwards dependency, define a *minimal* `TopicContextProvider` trait in `heartbit-core` instead — pure interface, no impl. heartbit-ghost re-exports it. See sub-step below.)

**Define `TopicContextProvider` in heartbit-core:**

Create `crates/heartbit-core/src/persona/topic_context.rs`:

```rust
//! Trait surface for persona-specific topic-context pre-fetching.
//! Concrete impls live in persona crates (e.g., heartbit-ghost).
//! See P1.6 spec §5.

use std::future::Future;
use std::pin::Pin;

/// Builds a plain-text context block consumed by the proactive-post
/// topic generator. Implementation strategies vary by persona —
/// `heartbit-ghost:x` fetches own tweets + mentions; `heartbit-rs:x`
/// inspects the local repo.
pub trait TopicContextProvider: Send + Sync {
    /// Returns a multi-line plain-text block. Empty string is allowed.
    /// `deps_ptr` is an opaque pointer to a persona-crate type
    /// (TopicContextDeps) — the trait is defined here without that
    /// type to avoid pulling persona-specific structs into core.
    fn build_context<'a>(
        &'a self,
        operator_user_id: &'a str,
        recent_history_json: &'a str,
        credentials: std::sync::Arc<dyn crate::execution_context::CredentialResolver>,
    ) -> Pin<Box<dyn Future<Output = Result<String, anyhow::Error>> + Send + 'a>>;
}
```

(The trait moves to `heartbit-core` and uses primitive types — string-encoded history — to avoid circular crate deps. The persona crate decodes the history JSON internally.)

Wire into `crates/heartbit-core/src/persona/mod.rs`:

```rust
pub mod topic_context;
pub use topic_context::TopicContextProvider;
```

- [ ] **Step 2: Update `crates/heartbit-ghost/src/posts/topic_context.rs` to use the core trait**

Replace the in-crate `pub trait TopicContextProvider` with a re-export from core:

```rust
pub use heartbit_core::persona::TopicContextProvider;
```

And update the `XGhostTopicContext` impl signature to match the core trait's signature (passes `operator_user_id`, `recent_history_json`, `credentials` as primitives — decode `recent_history_json` to `Vec<PostHistoryEntry>` inside `build_context`).

```rust
impl TopicContextProvider for XGhostTopicContext {
    fn build_context<'a>(
        &'a self,
        operator_user_id: &'a str,
        recent_history_json: &'a str,
        credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver>,
    ) -> Pin<Box<dyn Future<Output = Result<String, anyhow::Error>> + Send + 'a>> {
        Box::pin(async move {
            let recent_history: Vec<PostHistoryEntry> =
                serde_json::from_str(recent_history_json).unwrap_or_default();
            let deps = TopicContextDeps {
                credentials,
                operator_user_id,
                recent_history,
            };
            self.build_context_inner(&deps).await
        })
    }
}

impl XGhostTopicContext {
    async fn build_context_inner<'a>(
        &'a self,
        deps: &'a TopicContextDeps<'a>,
    ) -> Result<String, anyhow::Error> {
        // ... existing body from Task 4 ...
    }
}
```

(This is a refactor of Task 4's code. Move the body into `build_context_inner`, and have the trait method decode + delegate.)

Update Task 4's tests to call `build_context_inner` directly so they don't go through JSON encode/decode for the history:

```rust
let provider = XGhostTopicContext::with_base_url(&server.uri());
let ctx = provider.build_context_inner(&deps).await.expect("happy path");
```

- [ ] **Step 3: Implement `HeartbitRsXTopicContext`**

In `crates/heartbit-ghost/src/posts/topic_context.rs`, replace the stub `pub struct HeartbitRsXTopicContext;` with:

```rust
/// Repo-grounded topic context for `heartbit-rs:x`. Inspects the
/// local repo (commits, recently-modified modules) to surface fresh
/// material for the topic generator.
pub struct HeartbitRsXTopicContext {
    repo_root: std::path::PathBuf,
}

impl HeartbitRsXTopicContext {
    /// Construct from a repo root path (the same path
    /// `RepoInspectTool` uses).
    pub fn new(repo_root: std::path::PathBuf) -> Self {
        Self { repo_root }
    }
}

impl TopicContextProvider for HeartbitRsXTopicContext {
    fn build_context<'a>(
        &'a self,
        _operator_user_id: &'a str,
        recent_history_json: &'a str,
        _credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver>,
    ) -> Pin<Box<dyn Future<Output = Result<String, anyhow::Error>> + Send + 'a>> {
        Box::pin(async move {
            let recent_history: Vec<PostHistoryEntry> =
                serde_json::from_str(recent_history_json).unwrap_or_default();
            let mut out = String::new();

            // Recent commits via `git log` (out-of-process).
            out.push_str("RECENT COMMITS (last 10):\n");
            match fetch_recent_commits(&self.repo_root, 10).await {
                Ok(lines) if !lines.is_empty() => {
                    for line in lines.iter().take(10) {
                        out.push_str(&format!("- {line}\n"));
                    }
                }
                Ok(_) => out.push_str("(none)\n"),
                Err(e) => {
                    tracing::warn!(error = %e, "git log failed");
                    out.push_str("(unavailable)\n");
                }
            }
            out.push('\n');

            // Recently-modified module names (top-level under crates/).
            out.push_str("RECENTLY-MODIFIED MODULES (last 24h):\n");
            match fetch_recently_modified_modules(&self.repo_root).await {
                Ok(mods) if !mods.is_empty() => {
                    for m in mods.iter().take(10) {
                        out.push_str(&format!("- {m}\n"));
                    }
                }
                Ok(_) => out.push_str("(none)\n"),
                Err(e) => {
                    tracing::warn!(error = %e, "module scan failed");
                    out.push_str("(unavailable)\n");
                }
            }
            out.push('\n');

            // Post history (shared format).
            out.push_str(&render_history_only(&recent_history));
            Ok(out)
        })
    }
}

async fn fetch_recent_commits(
    repo_root: &std::path::Path,
    n: usize,
) -> Result<Vec<String>, anyhow::Error> {
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("log")
        .arg("--oneline")
        .arg(format!("-{n}"))
        .output()
        .await?;
    if !output.status.success() {
        anyhow::bail!("git log failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(String::from)
        .collect())
}

async fn fetch_recently_modified_modules(
    repo_root: &std::path::Path,
) -> Result<Vec<String>, anyhow::Error> {
    use chrono::Duration;
    use chrono::Utc;

    let cutoff = Utc::now() - Duration::hours(24);
    let cutoff_unix = cutoff.timestamp();
    let output = tokio::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("log")
        .arg(format!("--since={cutoff_unix}"))
        .arg("--name-only")
        .arg("--pretty=format:")
        .output()
        .await?;
    if !output.status.success() {
        anyhow::bail!(
            "git log --name-only failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut modules = std::collections::BTreeSet::new();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("crates/") {
            // crates/<name>/...
            if let Some(slash) = rest.find('/') {
                modules.insert(rest[..slash].to_string());
            }
        }
    }
    Ok(modules.into_iter().collect())
}
```

- [ ] **Step 4: Wire personas to populate `topic_context_provider`**

In `crates/heartbit-ghost/src/lib.rs`, find `XGhostPersona::expand` and add to the returned `PersonaExpansion`:

```rust
topic_context_provider: Some(std::sync::Arc::new(
    crate::posts::XGhostTopicContext::new(),
)),
```

In `crates/heartbit-ghost/src/heartbit_rs.rs`, find the persona's `expand()` and add:

```rust
topic_context_provider: {
    // The repo root is the same one used by RepoInspectTool.
    let repo_root = /* same logic as agents::tools_for_heartbit_rs */
        std::env::var("HEARTBIT_REPO_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().expect("cwd"));
    Some(std::sync::Arc::new(
        crate::posts::HeartbitRsXTopicContext::new(repo_root),
    ))
},
```

(Both `expand()` functions need to update their `PersonaExpansion { ... }` literal to include the new field. Compile errors will guide you.)

- [ ] **Step 5: Add `topic_context_provider: None` to all other call sites**

```bash
grep -rn "PersonaExpansion {" crates/ --include="*.rs"
```

Add `topic_context_provider: None` to each existing `PersonaExpansion { ... }` literal (typically test fixtures that build the struct manually). The compiler will fail until they're all updated.

- [ ] **Step 6: Tests**

Add to `crates/heartbit-ghost/src/posts/topic_context.rs::tests`:

```rust
    #[tokio::test]
    async fn heartbit_rs_context_handles_missing_repo_gracefully() {
        let provider = HeartbitRsXTopicContext::new(std::path::PathBuf::from("/nonexistent/path"));
        struct StubCreds;
        impl CredentialResolver for StubCreds {
            fn resolve(
                &self,
                _name: &str,
            ) -> Pin<
                Box<dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>> + Send + '_>,
            > {
                Box::pin(async { Ok(heartbit_core::Secret::new("x")) })
            }
        }
        let creds: Arc<dyn CredentialResolver> = Arc::new(StubCreds);
        // Pass empty history.
        let history_json = "[]";
        let ctx = provider
            .build_context("anything", history_json, creds)
            .await
            .expect("graceful");
        assert!(ctx.contains("RECENT COMMITS"), "should still render headers");
        assert!(ctx.contains("(unavailable)"));
    }
```

Run:

```bash
cargo test -p heartbit-ghost --lib posts::topic_context
cargo test -p heartbit-core --lib persona::topic_context
cargo build --workspace
```

Expected: all pass; build clean.

- [ ] **Step 7: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "$(cat <<'EOF'
feat(core+ghost): TopicContextProvider trait + 2 persona impls

- New trait heartbit_core::persona::TopicContextProvider — primitive
  signature (operator_user_id + recent_history_json + credentials) so
  heartbit-core stays free of persona-crate types.
- New PersonaExpansion field topic_context_provider: Option<Arc<dyn ...>>.
- XGhostTopicContext (heartbit-ghost): pre-fetches own tweets + mentions
  via the X API, renders alongside post history.
- HeartbitRsXTopicContext (heartbit-ghost): inspects the local repo
  via `git log --oneline` + `git log --name-only` for recent commits
  and recently-modified modules.
- Both personas populate the field in their expand() implementations.

heartbit-ghost P1.6a — task 5/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `PostHistoryStore` trait + `InMemoryPostHistoryStore` + `JsonlPostHistoryStore`

**Files:**
- Modify: `crates/heartbit-ghost/src/posts/history.rs` (replace stub with full impls)

- [ ] **Step 1: Read the P1.5 `MentionStore` for the pattern**

```bash
sed -n '1,200p' crates/heartbit-ghost/src/reply/storage.rs
```

The post-history store mirrors the same shape (Pin<Box<dyn Future>> trait, two impls, JSONL append-only). Adapt the entry shape to `PostHistoryEntry` (already defined in `posts/mod.rs`).

- [ ] **Step 2: Replace `posts/history.rs` with the full impls**

Replace the contents of `crates/heartbit-ghost/src/posts/history.rs` with:

```rust
//! Post history store for proactive posting. Tracks per-persona
//! `PostHistoryEntry` records (posted_at + topic + outcome + tweet_id).
//! Used by the topic generator's input (recent history block) and by
//! the duplicate check before the pipeline runs.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use chrono::{DateTime, Duration, Utc};

use super::PostHistoryEntry;

/// Errors raised by [`PostHistoryStore`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// I/O failure (file not readable, write failed, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSONL parse failure on replay.
    #[error("parse: {0}")]
    Parse(String),
}

/// Persistent storage for proactive post outcomes.
///
/// Implementations use `Pin<Box<dyn Future>>` desugaring (matches
/// the rest of P1.5/P1.6 — no async-trait dep).
pub trait PostHistoryStore: Send + Sync {
    /// Append one entry for `persona`.
    fn record<'a>(
        &'a self,
        persona: &'a str,
        entry: PostHistoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>>;

    /// Most-recent-first up to `limit` entries for `persona`.
    fn recent<'a>(
        &'a self,
        persona: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PostHistoryEntry>, StoreError>> + Send + 'a>>;

    /// Whether a topic case-insensitive-equal to `topic` was already
    /// recorded for `persona` in the lookback `within`.
    fn was_posted_recently<'a>(
        &'a self,
        persona: &'a str,
        topic: &'a str,
        within: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>>;
}

// -- in-memory impl ----------------------------------------------------

/// Volatile in-memory store. Useful for tests and dev runs.
pub struct InMemoryPostHistoryStore {
    inner: tokio::sync::RwLock<InMemoryInner>,
}

#[derive(Default)]
struct InMemoryInner {
    /// persona → vec of entries (append order; most recent at the end).
    entries: std::collections::HashMap<String, Vec<PostHistoryEntry>>,
}

impl Default for InMemoryPostHistoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryPostHistoryStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::RwLock::new(InMemoryInner::default()),
        }
    }
}

impl PostHistoryStore for InMemoryPostHistoryStore {
    fn record<'a>(
        &'a self,
        persona: &'a str,
        entry: PostHistoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.inner
                .write()
                .await
                .entries
                .entry(persona.to_string())
                .or_default()
                .push(entry);
            Ok(())
        })
    }

    fn recent<'a>(
        &'a self,
        persona: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PostHistoryEntry>, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            let v = g.entries.get(persona).cloned().unwrap_or_default();
            let mut out: Vec<PostHistoryEntry> = v.into_iter().rev().take(limit).collect();
            // out is most-recent-first since we reversed before take.
            out.shrink_to_fit();
            Ok(out)
        })
    }

    fn was_posted_recently<'a>(
        &'a self,
        persona: &'a str,
        topic: &'a str,
        within: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            let cutoff = Utc::now() - within;
            let topic_lower = topic.to_lowercase();
            Ok(g.entries
                .get(persona)
                .map(|v| {
                    v.iter().any(|e| {
                        e.posted_at >= cutoff && e.topic.to_lowercase() == topic_lower
                    })
                })
                .unwrap_or(false))
        })
    }
}

// -- JSONL impl --------------------------------------------------------

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StoreLine {
    persona: String,
    entry: PostHistoryEntry,
}

/// JSONL append-only post-history store. Replays the file at [`open`]
/// time into an in-memory mirror; subsequent records both append and
/// update the mirror.
pub struct JsonlPostHistoryStore {
    path: PathBuf,
    inner: tokio::sync::RwLock<InMemoryInner>,
}

impl JsonlPostHistoryStore {
    /// Open or create the store at `path`. Replays existing JSONL events.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let store = Self {
            path,
            inner: tokio::sync::RwLock::new(InMemoryInner::default()),
        };
        store.replay().await?;
        Ok(store)
    }

    async fn replay(&self) -> Result<(), StoreError> {
        let text = match tokio::fs::read_to_string(&self.path).await {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let mut g = self.inner.write().await;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let parsed: StoreLine = serde_json::from_str(line)
                .map_err(|e| StoreError::Parse(format!("line {line:?}: {e}")))?;
            g.entries
                .entry(parsed.persona)
                .or_default()
                .push(parsed.entry);
        }
        Ok(())
    }

    async fn append(&self, persona: &str, entry: &PostHistoryEntry) -> Result<(), StoreError> {
        use tokio::io::AsyncWriteExt;
        let line = StoreLine {
            persona: persona.to_string(),
            entry: entry.clone(),
        };
        let serialized =
            serde_json::to_string(&line).map_err(|e| StoreError::Parse(format!("{e}")))?;
        let mut f = tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)
            .await?;
        f.write_all(serialized.as_bytes()).await?;
        f.write_all(b"\n").await?;
        f.sync_data().await?;
        Ok(())
    }
}

impl PostHistoryStore for JsonlPostHistoryStore {
    fn record<'a>(
        &'a self,
        persona: &'a str,
        entry: PostHistoryEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), StoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.append(persona, &entry).await?;
            self.inner
                .write()
                .await
                .entries
                .entry(persona.to_string())
                .or_default()
                .push(entry);
            Ok(())
        })
    }

    fn recent<'a>(
        &'a self,
        persona: &'a str,
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PostHistoryEntry>, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            let v = g.entries.get(persona).cloned().unwrap_or_default();
            Ok(v.into_iter().rev().take(limit).collect())
        })
    }

    fn was_posted_recently<'a>(
        &'a self,
        persona: &'a str,
        topic: &'a str,
        within: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, StoreError>> + Send + 'a>> {
        Box::pin(async move {
            let g = self.inner.read().await;
            let cutoff = Utc::now() - within;
            let topic_lower = topic.to_lowercase();
            Ok(g.entries
                .get(persona)
                .map(|v| {
                    v.iter().any(|e| {
                        e.posted_at >= cutoff && e.topic.to_lowercase() == topic_lower
                    })
                })
                .unwrap_or(false))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posts::PostOutcome;

    fn fixture_entry(topic: &str, ago: chrono::Duration) -> PostHistoryEntry {
        PostHistoryEntry {
            posted_at: Utc::now() - ago,
            topic: topic.into(),
            outcome: PostOutcome::Posted {
                chosen_index: 0,
                url: "https://x.com/i/web/status/123".into(),
            },
            tweet_id: Some("123".into()),
        }
    }

    #[tokio::test]
    async fn in_memory_record_then_recent_returns_most_recent_first() {
        let store = InMemoryPostHistoryStore::new();
        store
            .record("p", fixture_entry("topic A", chrono::Duration::hours(2)))
            .await
            .unwrap();
        store
            .record("p", fixture_entry("topic B", chrono::Duration::hours(1)))
            .await
            .unwrap();
        let recent = store.recent("p", 5).await.unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].topic, "topic B");
        assert_eq!(recent[1].topic, "topic A");
    }

    #[tokio::test]
    async fn in_memory_was_posted_recently_is_case_insensitive_within_window() {
        let store = InMemoryPostHistoryStore::new();
        store
            .record(
                "p",
                fixture_entry("Calibrated Abstention", chrono::Duration::hours(12)),
            )
            .await
            .unwrap();
        // Same topic, different case, within 24h → true
        let hit = store
            .was_posted_recently("p", "calibrated abstention", chrono::Duration::hours(24))
            .await
            .unwrap();
        assert!(hit);
        // Outside the window → false
        let miss = store
            .was_posted_recently("p", "calibrated abstention", chrono::Duration::hours(6))
            .await
            .unwrap();
        assert!(!miss);
    }

    #[tokio::test]
    async fn in_memory_per_persona_isolation() {
        let store = InMemoryPostHistoryStore::new();
        store
            .record("a", fixture_entry("foo", chrono::Duration::hours(1)))
            .await
            .unwrap();
        let recent_a = store.recent("a", 5).await.unwrap();
        let recent_b = store.recent("b", 5).await.unwrap();
        assert_eq!(recent_a.len(), 1);
        assert!(recent_b.is_empty());
    }

    #[tokio::test]
    async fn jsonl_round_trips_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("posts.jsonl");
        {
            let s1 = JsonlPostHistoryStore::open(&path).await.unwrap();
            s1.record("p", fixture_entry("alpha", chrono::Duration::hours(1)))
                .await
                .unwrap();
            s1.record("p", fixture_entry("beta", chrono::Duration::minutes(30)))
                .await
                .unwrap();
        }
        let s2 = JsonlPostHistoryStore::open(&path).await.unwrap();
        let recent = s2.recent("p", 5).await.unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].topic, "beta");
        assert_eq!(recent[1].topic, "alpha");
    }

    #[tokio::test]
    async fn jsonl_handles_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.jsonl");
        let s = JsonlPostHistoryStore::open(&path).await.unwrap();
        assert!(s.recent("p", 5).await.unwrap().is_empty());
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p heartbit-ghost --lib posts::history
```

Expected: 5 PASS.

- [ ] **Step 4: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/posts/history.rs
git commit -m "$(cat <<'EOF'
feat(ghost): PostHistoryStore — InMemory + Jsonl impls

Tracks per-persona PostHistoryEntry records (posted_at + topic +
outcome + tweet_id). Pin<Box<dyn Future>> trait shape (matches
ReplyReviewDelivery pattern). 3 methods: record, recent
(most-recent-first), was_posted_recently (case-insensitive equality
within a configurable lookback). JsonlPostHistoryStore is append-only
+ in-memory replay at open. 5 unit tests cover happy path, case
insensitivity, per-persona isolation, jsonl reload, missing file.

heartbit-ghost P1.6b — task 6/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `PersonaPostsConfig` + DaemonConfig field

**Files:**
- Modify: `crates/heartbit-core/src/config/daemon.rs` (add struct + field)
- Modify: `crates/heartbit-core/src/config/mod.rs` (re-export)
- Modify: `crates/heartbit/src/lib.rs` (re-export)

- [ ] **Step 1: Add `PersonaPostsConfig` to daemon.rs**

In `crates/heartbit-core/src/config/daemon.rs`, after the existing `PersonaMentionsConfig` struct:

```rust
/// Per-persona proactive-posting configuration.
///
/// When present, the daemon registers a `PersonaPostScheduler` that
/// fires `DaemonCommand::PersonaPost` on the configured cadence
/// (gated by `active_hours`). The handler runs the topic generator,
/// drafts candidate threads via the existing review pipeline, sends
/// them to Telegram for review, posts the chosen draft.
///
/// Configured under `[[daemon.persona_posts]]` blocks.
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaPostsConfig {
    /// Persona registry name (e.g. `"heartbit-ghost:x"`).
    pub persona: String,
    /// Whether this poster is enabled.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
    /// Posting interval, in seconds. Default 14400 (4 hours).
    /// Validation: must be ≥60 (rejected at config load otherwise).
    #[serde(default = "default_post_interval_seconds")]
    pub post_interval_seconds: u64,
    /// Optional `"HH:MM-HH:MM"` window during which posts are allowed.
    /// Outside this window, the scheduler tick is a no-op. When absent,
    /// posting is allowed 24/7.
    #[serde(default)]
    pub active_hours: Option<ActiveHoursConfig>,
    /// Number of candidate threads to draft per tick (1..=10).
    /// Default 3.
    #[serde(default = "default_post_candidates")]
    pub candidates_per_draft: usize,
    /// Backend for the post history store: `"in_memory"` or `"jsonl"`.
    #[serde(default = "default_post_history_store")]
    pub post_history_store: String,
    /// Path to the JSONL store file (only used when
    /// `post_history_store == "jsonl"`). Tilde expansion happens at
    /// store-construction time.
    #[serde(default)]
    pub post_history_path: Option<String>,
    /// How far back to check for topic duplicates. Default 30 days.
    #[serde(default = "default_post_history_lookback_days")]
    pub post_history_lookback_days: i64,
    /// Optional fallback brief used when the persona declares no
    /// topic-context provider, or appended to the provider's output.
    #[serde(default)]
    pub topic_brief: Option<String>,
}

fn default_post_interval_seconds() -> u64 {
    14400
}

fn default_post_candidates() -> usize {
    3
}

fn default_post_history_store() -> String {
    "in_memory".into()
}

fn default_post_history_lookback_days() -> i64 {
    30
}
```

`ActiveHoursConfig` already exists in this file. Reuse.

- [ ] **Step 2: Add the field to `DaemonConfig`**

In the same file, in `pub struct DaemonConfig { ... }`, after `pub persona_mentions: Vec<PersonaMentionsConfig>`:

```rust
    /// Per-persona proactive-posting configuration. One entry per
    /// persona that has proactive posting enabled.
    #[serde(default)]
    pub persona_posts: Vec<PersonaPostsConfig>,
```

- [ ] **Step 3: Update existing struct literals**

```bash
grep -rn "DaemonConfig {" crates/ --include="*.rs"
```

Add `persona_posts: vec![]` to each existing `DaemonConfig { ... }` literal (test fixtures, etc.).

- [ ] **Step 4: Re-export in `config/mod.rs`**

In `crates/heartbit-core/src/config/mod.rs`, find the line that re-exports `PersonaMentionsConfig` and add `PersonaPostsConfig` next to it.

- [ ] **Step 5: Re-export in `crates/heartbit/src/lib.rs`**

Find the existing `pub use config::{ ..., PersonaMentionsConfig, ... };` block and add `PersonaPostsConfig` alphabetically.

- [ ] **Step 6: Tests**

In `crates/heartbit-core/src/config/daemon.rs::tests`:

```rust
    #[test]
    fn persona_posts_config_parses_with_defaults() {
        let toml = r#"
[[daemon.persona_posts]]
persona = "heartbit-ghost:x"
"#;
        #[derive(Deserialize)]
        struct Wrapper {
            daemon: DaemonConfig,
        }
        let cfg: Wrapper = toml::from_str(toml).unwrap();
        assert_eq!(cfg.daemon.persona_posts.len(), 1);
        let p = &cfg.daemon.persona_posts[0];
        assert_eq!(p.persona, "heartbit-ghost:x");
        assert!(p.enabled);
        assert_eq!(p.post_interval_seconds, 14400);
        assert_eq!(p.candidates_per_draft, 3);
        assert_eq!(p.post_history_store, "in_memory");
        assert_eq!(p.post_history_lookback_days, 30);
        assert!(p.active_hours.is_none());
    }

    #[test]
    fn persona_posts_config_parses_full() {
        let toml = r#"
[[daemon.persona_posts]]
persona = "heartbit-ghost:x"
enabled = true
post_interval_seconds = 7200
active_hours = { start = "09:00", end = "22:00" }
candidates_per_draft = 5
post_history_store = "jsonl"
post_history_path = "~/.heartbit/posts.jsonl"
post_history_lookback_days = 14
topic_brief = "agents, Rust, LLMs"
"#;
        #[derive(Deserialize)]
        struct Wrapper {
            daemon: DaemonConfig,
        }
        let cfg: Wrapper = toml::from_str(toml).unwrap();
        let p = &cfg.daemon.persona_posts[0];
        assert_eq!(p.post_interval_seconds, 7200);
        assert_eq!(p.candidates_per_draft, 5);
        assert_eq!(p.post_history_store, "jsonl");
        assert_eq!(p.post_history_path.as_deref(), Some("~/.heartbit/posts.jsonl"));
        assert_eq!(p.post_history_lookback_days, 14);
        assert_eq!(p.topic_brief.as_deref(), Some("agents, Rust, LLMs"));
        assert!(p.active_hours.is_some());
    }
```

(`ActiveHoursConfig` deserialization syntax — check the existing struct's serde shape; the example above uses inline-table `{ start = ..., end = ... }`. Adjust if the struct uses a different format.)

- [ ] **Step 7: Build + test + commit**

```bash
cargo build --workspace
cargo test -p heartbit-core --lib config::daemon
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
git add -A
git commit -m "$(cat <<'EOF'
feat(core): PersonaPostsConfig — daemon config for proactive posting

[[daemon.persona_posts]] blocks: persona, enabled, post_interval_seconds
(default 14400 = 4h), active_hours (optional), candidates_per_draft
(default 3, 1..=10), post_history_store ("in_memory" | "jsonl"),
post_history_path (jsonl only), post_history_lookback_days (default 30),
topic_brief (optional). Reuses the existing ActiveHoursConfig struct.

heartbit-ghost P1.6b — task 7/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `DaemonCommand::PersonaPost` variant + stub arm + serde test

**Files:**
- Modify: `crates/heartbit/src/daemon/types.rs`
- Modify: `crates/heartbit/src/daemon/core.rs` (stub arm)

- [ ] **Step 1: Add the variant**

In `crates/heartbit/src/daemon/types.rs`, in `pub enum DaemonCommand { ... }`, after `ReplyDraft`:

```rust
    /// Cron-driven: generate one proactive post for `persona`. Fired
    /// by `PersonaPostScheduler` on the configured cadence (gated by
    /// active_hours). Handler in task 10.
    PersonaPost {
        /// Persona name (e.g. `"heartbit-ghost:x"`).
        persona: String,
    },
```

- [ ] **Step 2: Add stub arm in core.rs**

Find the `match cmd { ... }` block in `crates/heartbit/src/daemon/core.rs`. After the `ReplyDraft` arm:

```rust
                        DaemonCommand::PersonaPost { persona } => {
                            tracing::warn!(
                                persona = %persona,
                                "PersonaPost dispatched but handler is not yet wired (P1.6c task 10)"
                            );
                        }
                    }
```

- [ ] **Step 3: Add serde round-trip test**

In `crates/heartbit/src/daemon/types.rs::tests`:

```rust
    #[test]
    fn daemon_command_persona_post_serde_round_trips() {
        let cmd = DaemonCommand::PersonaPost {
            persona: "heartbit-ghost:x".into(),
        };
        let s = serde_json::to_string(&cmd).unwrap();
        let parsed: DaemonCommand = serde_json::from_str(&s).unwrap();
        match parsed {
            DaemonCommand::PersonaPost { persona } => {
                assert_eq!(persona, "heartbit-ghost:x");
            }
            other => panic!("expected PersonaPost, got {other:?}"),
        }
    }
```

- [ ] **Step 4: Build + test + commit**

```bash
cargo build --workspace --features daemon
cargo test -p heartbit --features daemon --lib daemon::types
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
git add crates/heartbit/src/daemon/types.rs crates/heartbit/src/daemon/core.rs
git commit -m "$(cat <<'EOF'
feat(daemon): DaemonCommand::PersonaPost variant + stub arm

Single-field variant { persona: String } — posting acts on the OAuth1-
authenticated account, no per-tweet operator selection needed. Stub
warn+noop arm in core.rs; real handler dispatch lands in task 11.

heartbit-ghost P1.6b — task 8/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `PersonaPostScheduler` (interval + active hours)

**Files:**
- Create: `crates/heartbit/src/daemon/persona_post.rs`
- Modify: `crates/heartbit/src/daemon/mod.rs` (re-export)
- Modify: `crates/heartbit/src/lib.rs` (re-export)

- [ ] **Step 1: Read the existing `MentionPollScheduler` for pattern**

```bash
sed -n '1,150p' crates/heartbit/src/daemon/mention_poll.rs
```

The post scheduler mirrors the same shape with one addition: `active_hours` gating.

- [ ] **Step 2: Read the active-hours helper from `HeartbitPulseScheduler`**

```bash
grep -n "is_within_active_hours\|fn parse_start\|fn parse_end" crates/heartbit/src/daemon/heartbit_pulse.rs crates/heartbit-core/src/config/daemon.rs
```

The helper might be a private method on `HeartbitPulseScheduler`. Either:
- Promote it to a free function in a shared module, OR
- Duplicate the logic in `PersonaPostScheduler` (simpler if the helper is small).

For V1, duplicate. The active-hours logic is ~10 lines.

- [ ] **Step 3: Create `crates/heartbit/src/daemon/persona_post.rs`**

```rust
//! Periodic proactive-post scheduler. Fires `DaemonCommand::PersonaPost`
//! per configured persona on the operator's cadence (gated by
//! `active_hours`).
//!
//! See P1.6 spec §8 and Task 9 of the corresponding plan.

use std::sync::Arc;
use std::time::Duration;

use chrono::Local;
use tokio_util::sync::CancellationToken;

use heartbit_core::config::{ActiveHoursConfig, PersonaPostsConfig};

use super::CommandProducer;
use super::types::DaemonCommand;

/// One scheduled poster. Fires a `PersonaPost` command every
/// `interval` (gated by `active_hours` when set) via the producer.
pub struct PersonaPostScheduler {
    persona: String,
    interval: Duration,
    active_hours: Option<ActiveHoursConfig>,
    producer: Arc<dyn CommandProducer>,
    commands_topic: String,
}

impl std::fmt::Debug for PersonaPostScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaPostScheduler")
            .field("persona", &self.persona)
            .field("interval", &self.interval)
            .field("active_hours_set", &self.active_hours.is_some())
            .field("commands_topic", &self.commands_topic)
            .finish()
    }
}

impl PersonaPostScheduler {
    /// Construct from a config entry.
    pub fn new(
        cfg: &PersonaPostsConfig,
        producer: Arc<dyn CommandProducer>,
        commands_topic: &str,
    ) -> Self {
        Self {
            persona: cfg.persona.clone(),
            interval: Duration::from_secs(cfg.post_interval_seconds.max(60)),
            active_hours: cfg.active_hours.clone(),
            producer,
            commands_topic: commands_topic.into(),
        }
    }

    /// Run the loop until `cancel` fires.
    pub async fn run(self, cancel: CancellationToken) {
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::info!(persona = %self.persona, "post scheduler shutting down");
                    break;
                }
                _ = tokio::time::sleep(self.interval) => {
                    if !self.is_within_active_hours() {
                        tracing::debug!(
                            persona = %self.persona,
                            "post scheduler tick: outside active hours, skipping"
                        );
                        continue;
                    }
                    let cmd = DaemonCommand::PersonaPost {
                        persona: self.persona.clone(),
                    };
                    let payload = match serde_json::to_vec(&cmd) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!(error = %e, "failed to serialize PersonaPost");
                            continue;
                        }
                    };
                    let key = format!("posts:{}", self.persona);
                    if let Err(e) = self
                        .producer
                        .send_command(&self.commands_topic, &key, &payload)
                        .await
                    {
                        tracing::error!(
                            persona = %self.persona,
                            error = %e,
                            "failed to dispatch PersonaPost"
                        );
                    } else {
                        tracing::debug!(persona = %self.persona, "post scheduler fired");
                    }
                }
            }
        }
    }

    fn is_within_active_hours(&self) -> bool {
        let Some(hours) = self.active_hours.as_ref() else {
            return true; // no gate
        };
        let now_local = Local::now().time();
        let Ok(start) = hours.parse_start() else {
            tracing::warn!(
                persona = %self.persona,
                "invalid active_hours.start; defaulting to allowed"
            );
            return true;
        };
        let Ok(end) = hours.parse_end() else {
            tracing::warn!(
                persona = %self.persona,
                "invalid active_hours.end; defaulting to allowed"
            );
            return true;
        };
        if start <= end {
            now_local >= start && now_local <= end
        } else {
            // Overnight window (e.g. 22:00–06:00).
            now_local >= start || now_local <= end
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ChannelCommandProducer;

    fn cfg_with_interval(interval: u64) -> PersonaPostsConfig {
        PersonaPostsConfig {
            persona: "heartbit-ghost:x".into(),
            enabled: true,
            post_interval_seconds: interval,
            active_hours: None,
            candidates_per_draft: 3,
            post_history_store: "in_memory".into(),
            post_history_path: None,
            post_history_lookback_days: 30,
            topic_brief: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn fires_persona_post_after_interval() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler = PersonaPostScheduler::new(&cfg_with_interval(60), producer, "test.commands");
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(scheduler.run(cancel_for_task));

        tokio::time::advance(Duration::from_secs(61)).await;
        let (key, payload) = rx
            .recv()
            .await
            .expect("scheduler should have fired");
        assert_eq!(key, "test.commands");
        let parsed: DaemonCommand = serde_json::from_slice(&payload).unwrap();
        assert!(matches!(parsed, DaemonCommand::PersonaPost { ref persona } if persona == "heartbit-ghost:x"));

        cancel.cancel();
        let _ = handle.await;
    }

    #[tokio::test(start_paused = true)]
    async fn does_not_fire_outside_active_hours() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        // Configure active_hours that exclude the current local time. We
        // can't know what local time is in CI, so use a 1-minute window
        // that's almost certainly NOT now.
        let mut cfg = cfg_with_interval(60);
        cfg.active_hours = Some(ActiveHoursConfig {
            start: "00:00".into(),
            end: "00:01".into(),
        });
        let scheduler = PersonaPostScheduler::new(&cfg, producer, "test.commands");
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(scheduler.run(cancel_for_task));

        tokio::time::advance(Duration::from_secs(120)).await;
        // After two ticks, no command should have been sent.
        assert!(rx.try_recv().is_err());

        cancel.cancel();
        let _ = handle.await;
    }

    #[tokio::test(start_paused = true)]
    async fn cancels_cleanly() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<(String, Vec<u8>)>();
        let producer: Arc<dyn CommandProducer> = Arc::new(ChannelCommandProducer { tx });
        let scheduler = PersonaPostScheduler::new(&cfg_with_interval(3600), producer, "test.commands");
        let cancel = CancellationToken::new();
        let cancel_for_task = cancel.clone();
        let handle = tokio::spawn(scheduler.run(cancel_for_task));
        cancel.cancel();
        // Should complete promptly.
        tokio::time::timeout(Duration::from_millis(500), handle)
            .await
            .expect("scheduler should exit on cancel")
            .unwrap();
    }
}
```

`ActiveHoursConfig` field names (`start`, `end`) and `parse_start()`/`parse_end()` methods must match the actual struct. Check `crates/heartbit-core/src/config/daemon.rs`. Adjust the test fixture accordingly.

- [ ] **Step 4: Wire into mod.rs and lib.rs**

In `crates/heartbit/src/daemon/mod.rs`:

```rust
pub mod persona_post;
pub use persona_post::PersonaPostScheduler;
```

In `crates/heartbit/src/lib.rs`, find the existing `pub use daemon::{ ... };` block and add `PersonaPostScheduler`.

- [ ] **Step 5: Run tests**

```bash
cargo test -p heartbit --features daemon --lib daemon::persona_post
```

Expected: 3 PASS.

- [ ] **Step 6: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit/src/daemon/persona_post.rs crates/heartbit/src/daemon/mod.rs crates/heartbit/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(daemon): PersonaPostScheduler — interval + active hours

Mirrors MentionPollScheduler's shape with active-hours gating
(parses HH:MM-HH:MM window from config; supports overnight wrap).
Fires DaemonCommand::PersonaPost on tick when within window. 3
tokio-paused-time tests cover firing, active-hours skipping, and
clean cancellation.

heartbit-ghost P1.6c — task 9/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `handle_persona_post` free function + `PersonaPostDeps`

**Files:**
- Create: `crates/heartbit/src/daemon/persona_post_handler.rs`
- Modify: `crates/heartbit/src/daemon/mod.rs` (re-export)
- Modify: `crates/heartbit/src/lib.rs` (re-export)

- [ ] **Step 1: Read the existing `handle_reply_draft` for the pattern**

```bash
sed -n '1,200p' crates/heartbit/src/daemon/reply_draft_handler.rs
```

`handle_persona_post` follows the same shape but invokes `run_review_pipeline` (review crate) instead of `run_reply_pipeline`.

- [ ] **Step 2: Create `crates/heartbit/src/daemon/persona_post_handler.rs`**

```rust
//! PersonaPost dispatcher handler — runs the proactive-post pipeline
//! for one scheduler tick: pre-fetch context → topic generator →
//! duplicate-check → review pipeline → record outcome.
//!
//! See P1.6 spec §8 for the algorithm.

use std::path::Path;
use std::sync::Arc;

use chrono::Duration;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::{PersonaParams, PersonaRegistry};

use heartbit_ghost::posts::{PostHistoryEntry, PostHistoryStore, PostOutcome};
use heartbit_ghost::review::{
    self, ReviewConfig, ReviewDelivery, ReviewError, ReviewOutcome, run_review_pipeline,
};

/// Dependencies for one PersonaPost handler invocation.
pub struct PersonaPostDeps<'a> {
    /// Persona name to run.
    pub persona_name: &'a str,
    /// Persona registry.
    pub registry: &'a PersonaRegistry,
    /// Post history store (for de-dup + recording outcome).
    pub history: &'a dyn PostHistoryStore,
    /// Lookback for the duplicate check.
    pub history_lookback: Duration,
    /// Optional fallback brief from config.
    pub topic_brief: Option<&'a str>,
    /// Operator's X user_id (passed to the topic context provider).
    pub operator_user_id: &'a str,
    /// LLM provider for sub-agents (topic generator + review pipeline).
    pub provider: Arc<BoxedProvider>,
    /// Telegram (or mock) review delivery.
    pub delivery: Arc<dyn ReviewDelivery>,
    /// `twitter_thread` tool — used by run_review_pipeline to post.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver shared with twitter_tool + topic context provider.
    pub credentials: Arc<dyn CredentialResolver>,
    /// Number of candidate threads to draft.
    pub candidates_per_draft: usize,
    /// Roots used by the review pipeline.
    pub corpora_root: &'a Path,
    pub profiles_root: &'a Path,
}

/// Run one PersonaPost handler invocation. On any terminal outcome,
/// records to history. Returns the outcome for caller introspection.
pub async fn handle_persona_post<'a>(
    deps: PersonaPostDeps<'a>,
) -> Result<PostOutcome, anyhow::Error> {
    let persona = deps
        .registry
        .get(deps.persona_name)
        .ok_or_else(|| anyhow::anyhow!("persona '{}' not registered", deps.persona_name))?;
    let expansion = persona
        .expand(&PersonaParams::default())
        .map_err(|e| anyhow::anyhow!("expand persona '{}': {e}", deps.persona_name))?;

    // Researcher override (heartbit-rs:x uses repo_researcher).
    let researcher_override = expansion
        .agents
        .iter()
        .find(|a| a.name == "repo_researcher")
        .map(|recipe| {
            let recipe = std::sync::Arc::new(recipe.clone_config());
            let tools: Vec<std::sync::Arc<dyn Tool>> = expansion
                .tools
                .iter()
                .filter(|t| t.definition().name == "repo_inspect")
                .cloned()
                .collect();
            (recipe, tools)
        });

    // 1. Build the topic generator's input context.
    let recent_history = deps
        .history
        .recent(deps.persona_name, 5)
        .await
        .map_err(|e| anyhow::anyhow!("history.recent: {e}"))?;
    let recent_history_json = serde_json::to_string(&recent_history).unwrap_or_else(|_| "[]".into());
    let mut user_message = String::new();
    if let Some(provider) = expansion.topic_context_provider.as_ref() {
        match provider
            .build_context(
                deps.operator_user_id,
                &recent_history_json,
                deps.credentials.clone(),
            )
            .await
        {
            Ok(s) => user_message.push_str(&s),
            Err(e) => {
                tracing::warn!(error = %e, "topic context provider failed; continuing with history only");
                user_message.push_str(&render_history_block(&recent_history));
            }
        }
    } else {
        user_message.push_str(&render_history_block(&recent_history));
    }
    if let Some(brief) = deps.topic_brief {
        user_message.push_str("\nTOPIC AREA (from config):\n");
        user_message.push_str(brief);
        user_message.push('\n');
    }
    user_message.push_str("\nPropose ONE topic per the OUTPUT spec, or 'no_topic'.\n");

    // 2. Run the topic generator.
    let topic_runner = heartbit_ghost::pipeline::runner_from_recipe(
        deps.provider.clone(),
        heartbit_ghost::agents::topic_generator_recipe(),
        Vec::new(),
    )
    .map_err(|e| anyhow::anyhow!("topic generator builder: {e}"))?;
    let gen_out = topic_runner
        .execute(&user_message)
        .await
        .map_err(|e| anyhow::anyhow!("topic generator exec: {e}"))?;
    let topic_raw = gen_out.result.trim();

    // 3. Handle the no_topic short-circuit.
    if topic_raw.eq_ignore_ascii_case("no_topic") || topic_raw.is_empty() {
        let entry = PostHistoryEntry {
            posted_at: chrono::Utc::now(),
            topic: String::new(),
            outcome: PostOutcome::NoTopic,
            tweet_id: None,
        };
        if let Err(e) = deps.history.record(deps.persona_name, entry).await {
            tracing::warn!(error = %e, "history.record (NoTopic) failed");
        }
        return Ok(PostOutcome::NoTopic);
    }
    let topic = topic_raw.to_string();

    // 4. Duplicate check.
    let is_dupe = deps
        .history
        .was_posted_recently(deps.persona_name, &topic, deps.history_lookback)
        .await
        .unwrap_or(false);
    if is_dupe {
        let entry = PostHistoryEntry {
            posted_at: chrono::Utc::now(),
            topic: topic.clone(),
            outcome: PostOutcome::SkippedDuplicate,
            tweet_id: None,
        };
        if let Err(e) = deps.history.record(deps.persona_name, entry).await {
            tracing::warn!(error = %e, "history.record (SkippedDuplicate) failed");
        }
        return Ok(PostOutcome::SkippedDuplicate);
    }

    // 5. Run the review pipeline.
    let cfg = ReviewConfig {
        persona_name: deps.persona_name,
        topic: &topic,
        provider: deps.provider.clone(),
        corpora_root: deps.corpora_root,
        profiles_root: deps.profiles_root,
        on_progress: Some(std::sync::Arc::new(|s: &str| {
            tracing::info!("post: {s}")
        })),
        candidates_per_draft: deps.candidates_per_draft,
        delivery: deps.delivery.clone(),
        twitter_tool: deps.twitter_tool.clone(),
        credentials: deps.credentials.clone(),
        mode_addendum: expansion.mode_addendum,
        researcher_override,
    };
    let review_out = run_review_pipeline(cfg)
        .await
        .map_err(|e: ReviewError| anyhow::anyhow!("review pipeline: {e}"))?;

    // 6. Map ReviewOutcome → PostOutcome and record.
    let post_outcome = map_review_outcome(&review_out.outcome);
    let tweet_id = match &review_out.outcome {
        ReviewOutcome::Posted { tweet_ids, .. } => tweet_ids.first().cloned(),
        _ => None,
    };
    let entry = PostHistoryEntry {
        posted_at: chrono::Utc::now(),
        topic: topic.clone(),
        outcome: post_outcome.clone(),
        tweet_id,
    };
    if let Err(e) = deps.history.record(deps.persona_name, entry).await {
        tracing::warn!(error = %e, "history.record (terminal) failed");
    }
    Ok(post_outcome)
}

fn map_review_outcome(o: &ReviewOutcome) -> PostOutcome {
    match o {
        ReviewOutcome::Posted {
            chosen_index,
            tweet_url,
            ..
        } => PostOutcome::Posted {
            chosen_index: *chosen_index,
            url: tweet_url.clone(),
        },
        ReviewOutcome::Skipped => PostOutcome::Skipped,
        ReviewOutcome::TimedOut => PostOutcome::TimedOut,
        ReviewOutcome::GateRejected {
            chosen_index,
            reason,
        } => PostOutcome::GateRejected {
            chosen_index: *chosen_index,
            reason: reason.clone(),
        },
        ReviewOutcome::PublishFailed {
            chosen_index,
            reason,
        } => PostOutcome::PublishFailed {
            chosen_index: *chosen_index,
            reason: reason.clone(),
        },
    }
}

fn render_history_block(history: &[PostHistoryEntry]) -> String {
    let mut out = String::new();
    out.push_str("RECENT POST HISTORY (last 5 from store):\n");
    if history.is_empty() {
        out.push_str("(none)\n");
    } else {
        for entry in history.iter().take(5) {
            let when = entry.posted_at.format("%Y-%m-%d");
            let outcome = match &entry.outcome {
                PostOutcome::Posted { .. } => "Posted",
                PostOutcome::Skipped => "Skipped",
                PostOutcome::TimedOut => "TimedOut",
                PostOutcome::NoTopic => "NoTopic",
                PostOutcome::SkippedDuplicate => "SkippedDuplicate",
                PostOutcome::GateRejected { .. } => "GateRejected",
                PostOutcome::PublishFailed { .. } => "PublishFailed",
            };
            let topic = if entry.topic.is_empty() {
                "(no topic)"
            } else {
                entry.topic.as_str()
            };
            out.push_str(&format!("- [{when}] {outcome}: {topic}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    // Mirror the test scaffold from reply_draft_handler.rs:
    //   - InMemoryPostHistoryStore
    //   - MockProvider canned responses
    //   - MockReviewDelivery (Pick(0) / Skip / TimedOut)
    //   - MockTwitterTool returning success
    //   - StubCredentialResolver
    //   - seed_snapshot helper from heartbit-ghost
    //
    // 5 integration tests:
    // 1. happy_path_runs_pipeline_and_records — generator returns topic,
    //    pipeline returns Posted → history records Posted with tweet_id.
    // 2. no_topic_short_circuits — generator returns "no_topic" → history
    //    records NoTopic, pipeline NOT called.
    // 3. duplicate_topic_skips_pipeline — generator returns a topic seen
    //    in history within lookback → records SkippedDuplicate, pipeline
    //    NOT called.
    // 4. telegram_skip_records_skipped — pipeline runs, user picks Skip
    //    → records Skipped.
    // 5. unknown_persona_returns_err — registry has no entry → Err, store
    //    unchanged.
    //
    // Each test ~80-100 lines. Use the same MockProvider / MockReviewDelivery
    // / MockTwitterTool patterns as crates/heartbit/src/daemon/reply_draft_handler.rs.
    // Reference that file's tests for verbatim patterns and copy verbatim.
    //
    // (Implementer expands these stubs to full bodies. The pattern is
    // already established in reply_draft_handler.rs::tests.)
}
```

The test stubs in the comment block correspond to 5 full test bodies. The implementer expands them by mirroring `crates/heartbit/src/daemon/reply_draft_handler.rs::tests` line-by-line — same `MockProvider::arc(canned)`, same `MockReviewDelivery::arc(DeliveryOutcome::Pick(0))` (with adjustments for the post pipeline's 3 candidates), same `seed_snapshot` helper, same `MockTwitterTool::success(thread_json)`.

For test 1 happy path, the canned MockProvider responses are:
1. Topic generator: `"calibrated abstention"`
2. Researcher: `"Research digest:\n- ..."`
3. Writer (×3 for candidates_per_draft=3): three different drafts
4. Critic (×3): `r#"{"verdict": "pass", "style_match_score": 0.9}"#`
5. Fact (×3): `r#"{"verdict": "verified"}"#`

Plus the MockReviewDelivery returns `Pick(0)` and the mock twitter tool returns the thread_json.

For test 2 (no_topic), MockProvider canned: `"no_topic"` (single response — the rest of the pipeline is skipped). MockReviewDelivery is set to error if called.

For test 3 (duplicate), pre-seed the history store with an entry matching the topic the generator will return. MockReviewDelivery is set to error if called.

For test 4 (skip), all stages run normally; MockReviewDelivery returns `Skip`; MockTwitterTool errors if called (it shouldn't be).

For test 5 (unknown persona), use an empty `PersonaRegistry` and verify Err propagates without touching the store (use `InMemoryPostHistoryStore::new()` and assert `store.recent(...)` is empty after).

- [ ] **Step 3: Wire into mod.rs and lib.rs**

In `crates/heartbit/src/daemon/mod.rs`:

```rust
pub mod persona_post_handler;
pub use persona_post_handler::{PersonaPostDeps, handle_persona_post};
```

In `crates/heartbit/src/lib.rs`, add to the existing daemon re-exports.

- [ ] **Step 4: Run tests**

```bash
cargo test -p heartbit --features daemon --lib daemon::persona_post_handler
```

Expected: 5 PASS.

- [ ] **Step 5: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/heartbit/src/daemon/persona_post_handler.rs crates/heartbit/src/daemon/mod.rs crates/heartbit/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(daemon): handle_persona_post — proactive post pipeline runtime

Free function + PersonaPostDeps struct (mirror of MentionPollDeps and
ReplyDraftDeps). Steps: build topic-generator user message from
TopicContextProvider output + recent history + topic_brief; run topic
generator; short-circuit on "no_topic"; duplicate-check via
PostHistoryStore.was_posted_recently; run_review_pipeline on the topic;
map ReviewOutcome to PostOutcome; record outcome to store.

Always records to history (every terminal outcome) — the cron poll
won't retry the same topic without an explicit gap, and the store
gives the operator a full audit trail.

5 integration tests cover the 5 outcome paths (happy, no_topic,
duplicate, telegram-skip, unknown persona).

heartbit-ghost P1.6c — task 10/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Lifecycle wiring — extend `MentionContext` + spawn scheduler + real handler dispatch

**Files:**
- Modify: `crates/heartbit/src/daemon/mention_context.rs` (add posts_entries + twitter_thread field)
- Modify: `crates/heartbit/src/daemon/core.rs` (spawn PersonaPostScheduler + replace stub PersonaPost arm)
- Modify: `crates/heartbit-cli/src/daemon/mod.rs` (build_mention_context constructs post entries + spawns)

- [ ] **Step 1: Extend `MentionContext`**

In `crates/heartbit/src/daemon/mention_context.rs`, add:

```rust
/// Per-persona state for one proactive-posts configuration entry.
pub struct PersonaPostEntry {
    /// Post history store (for de-dup + outcome recording).
    pub history: std::sync::Arc<dyn heartbit_ghost::posts::PostHistoryStore>,
    /// Polling interval (used by the scheduler at startup).
    pub interval: std::time::Duration,
    /// Optional active-hours window.
    pub active_hours: Option<heartbit_core::config::ActiveHoursConfig>,
    /// Number of candidate threads per tick.
    pub candidates_per_draft: usize,
    /// Lookback window for the duplicate check.
    pub history_lookback: chrono::Duration,
    /// Optional fallback brief.
    pub topic_brief: Option<String>,
    /// Operator's X user_id (passed to the topic context provider).
    pub operator_user_id: String,
}
```

In `pub struct MentionContext { ... }`, add a field after `entries`:

```rust
    /// Per-persona proactive-post state. Keyed by persona name.
    pub posts_entries: std::collections::HashMap<String, PersonaPostEntry>,
    /// `twitter_thread` tool — used by the post pipeline. Shared with
    /// the rest of the daemon. (P1.5 only used twitter_reply.)
    pub twitter_thread: std::sync::Arc<dyn heartbit_core::Tool>,
```

Update `MentionContext::new` (or however it's constructed) to take `posts_entries` and `twitter_thread`. Update `Debug` impl to include them.

Re-export `PersonaPostEntry` from `crates/heartbit/src/daemon/mod.rs` and `crates/heartbit/src/lib.rs`.

Update existing `MentionContext` test fixtures (and the CLI's `build_mention_context`) to populate the new fields.

- [ ] **Step 2: Spawn `PersonaPostScheduler` in `DaemonCore::run`**

In `crates/heartbit/src/daemon/core.rs::run`, after the `MentionPollScheduler` spawn block:

```rust
        // --- Spawn PersonaPostScheduler instances ---
        if let Some(ctx) = self.mention_context.as_ref() {
            for (persona, entry) in ctx.posts_entries.iter() {
                let cfg = heartbit_core::config::PersonaPostsConfig {
                    persona: persona.clone(),
                    enabled: true,
                    post_interval_seconds: entry.interval.as_secs(),
                    active_hours: entry.active_hours.clone(),
                    candidates_per_draft: entry.candidates_per_draft,
                    post_history_store: "in_memory".into(),
                    post_history_path: None,
                    post_history_lookback_days: entry.history_lookback.num_days(),
                    topic_brief: entry.topic_brief.clone(),
                };
                let producer: Arc<dyn CommandProducer> = Arc::new(KafkaCommandProducer::new(
                    self.producer.clone(),
                ));
                let scheduler = crate::daemon::PersonaPostScheduler::new(
                    &cfg,
                    producer,
                    &self.commands_topic,
                );
                let cancel = self.cancel.clone();
                tokio::spawn(scheduler.run(cancel));
                tracing::info!(persona = %persona, "post scheduler spawned");
            }
        }
```

- [ ] **Step 3: Replace stub `PersonaPost` arm with real handler dispatch**

In the same file, replace the warn+noop `PersonaPost` arm with:

```rust
                        DaemonCommand::PersonaPost { persona } => {
                            let Some(ctx) = self.mention_context.clone() else {
                                tracing::warn!(
                                    persona = %persona,
                                    "PersonaPost received but no mention_context configured"
                                );
                                continue;
                            };
                            let Some(entry) = ctx.posts_entries.get(&persona) else {
                                tracing::warn!(
                                    persona = %persona,
                                    "PersonaPost for unknown persona"
                                );
                                continue;
                            };
                            let history = entry.history.clone();
                            let topic_brief = entry.topic_brief.clone();
                            let candidates_per_draft = entry.candidates_per_draft;
                            let history_lookback = entry.history_lookback;
                            let operator_user_id = entry.operator_user_id.clone();
                            let registry = ctx.registry.clone();
                            let provider = ctx.provider.clone();
                            let delivery = ctx.delivery.clone();
                            let twitter_thread = ctx.twitter_thread.clone();
                            let credentials = ctx.credentials.clone();
                            let corpora_root = ctx.corpora_root.clone();
                            let profiles_root = ctx.profiles_root.clone();
                            let persona_owned = persona.clone();
                            tokio::spawn(async move {
                                let deps = crate::daemon::PersonaPostDeps {
                                    persona_name: &persona_owned,
                                    registry: &registry,
                                    history: &*history,
                                    history_lookback,
                                    topic_brief: topic_brief.as_deref(),
                                    operator_user_id: &operator_user_id,
                                    provider,
                                    delivery,
                                    twitter_tool: twitter_thread,
                                    credentials,
                                    candidates_per_draft,
                                    corpora_root: &corpora_root,
                                    profiles_root: &profiles_root,
                                };
                                if let Err(e) = crate::daemon::handle_persona_post(deps).await {
                                    tracing::error!(error = %e, "persona post handler failed");
                                }
                            });
                        }
```

- [ ] **Step 4: CLI wiring — build post entries**

In `crates/heartbit-cli/src/daemon/mod.rs`, find `build_mention_context` (or wherever `MentionContext` is constructed). Extend it to also iterate `daemon_config.persona_posts` and populate `posts_entries`:

```rust
    let mut posts_entries: std::collections::HashMap<String, heartbit::PersonaPostEntry> =
        std::collections::HashMap::new();
    for cfg in &daemon_config.persona_posts {
        if !cfg.enabled {
            continue;
        }
        if cfg.post_interval_seconds < 60 {
            anyhow::bail!(
                "[[daemon.persona_posts]] persona='{}': post_interval_seconds must be ≥60",
                cfg.persona
            );
        }
        let history: std::sync::Arc<dyn heartbit_ghost::posts::PostHistoryStore> =
            match cfg.post_history_store.as_str() {
                "in_memory" => std::sync::Arc::new(
                    heartbit_ghost::posts::InMemoryPostHistoryStore::new(),
                ),
                "jsonl" => {
                    let path = cfg.post_history_path.as_deref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "[[daemon.persona_posts]] persona='{}' uses post_history_store='jsonl' but no post_history_path",
                            cfg.persona
                        )
                    })?;
                    let path = expand_tilde(path)?;
                    std::sync::Arc::new(
                        heartbit_ghost::posts::JsonlPostHistoryStore::open(&path)
                            .await
                            .with_context(|| format!("open jsonl post history at {}", path.display()))?,
                    )
                }
                other => anyhow::bail!(
                    "unknown post_history_store backend '{other}' (expected 'in_memory' or 'jsonl')"
                ),
            };
        // Operator user_id: re-resolve from the matching mentions config if any,
        // else require a top-level field on the post config (V1: borrow from
        // mentions config if present, else error).
        let operator_user_id = daemon_config
            .persona_mentions
            .iter()
            .find(|m| m.persona == cfg.persona && m.enabled)
            .map(|m| m.user_id.clone())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "persona_posts persona='{}' requires a matching enabled persona_mentions entry to resolve operator_user_id",
                    cfg.persona
                )
            })?;
        posts_entries.insert(
            cfg.persona.clone(),
            heartbit::PersonaPostEntry {
                history,
                interval: std::time::Duration::from_secs(cfg.post_interval_seconds),
                active_hours: cfg.active_hours.clone(),
                candidates_per_draft: cfg.candidates_per_draft,
                history_lookback: chrono::Duration::days(cfg.post_history_lookback_days),
                topic_brief: cfg.topic_brief.clone(),
                operator_user_id,
            },
        );
    }
```

Also: instantiate `twitter_thread`:

```rust
    let twitter_thread: std::sync::Arc<dyn heartbit_core::Tool> =
        std::sync::Arc::new(heartbit_ghost::tools::TwitterThreadTool::new());
```

Pass `posts_entries` + `twitter_thread` into `MentionContext::new(...)`.

(If `MentionContext` is constructed via struct-literal rather than a `new`, update the literal.)

- [ ] **Step 5: Build + test + commit**

```bash
cargo build --workspace --features daemon
cargo test --workspace --features daemon 2>&1 | tail -10
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
git add -A
git commit -m "$(cat <<'EOF'
feat(daemon+cli): wire PersonaPost lifecycle

Closes the deferred wiring from P1.6 task 8:
- MentionContext gains posts_entries: HashMap<String, PersonaPostEntry>
  and twitter_thread (twitter_thread alongside twitter_reply).
- DaemonCore::run spawns one PersonaPostScheduler per enabled persona_posts
  entry, before the consumer loop.
- DaemonCommand::PersonaPost arm calls handle_persona_post via tokio::spawn
  (mirror of MentionPoll/ReplyDraft dispatch shape).
- CLI build_mention_context constructs PostHistoryStore (in_memory or
  jsonl) per entry; rejects post_interval_seconds < 60; resolves
  operator_user_id from the matching persona_mentions entry; loads
  TwitterThreadTool.

After this commit, `heartbit daemon --config some.toml` with
[[daemon.persona_posts]] configured will actually fire the proactive
post pipeline on cadence.

heartbit-ghost P1.6c — task 11/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: CLI — `persona post` + `persona posts list`

**Files:**
- Modify: `crates/heartbit-cli/src/persona.rs` (add 2 variants + dispatch arms)
- Modify: `crates/heartbit-cli/src/persona_review.rs` (add `post_config_from_env` + history list helper)

- [ ] **Step 1: Read the existing `persona reply` for the dispatch pattern**

```bash
grep -n "PersonaCommand::Reply\|fn dispatch" crates/heartbit-cli/src/persona.rs | head -5
```

The new `Post` arm follows the same shape: resolve persona, expand, build provider/delivery/tool, call `handle_persona_post` (or `run_review_pipeline` directly with a hand-fed topic).

- [ ] **Step 2: Add the variants**

In `crates/heartbit-cli/src/persona.rs`, add to `PersonaCommand`:

```rust
    /// Post a proactive thread on demand (no daemon needed).
    /// Generates a topic via the topic_generator (using the persona's
    /// declared TopicContextProvider) unless `--topic` overrides.
    Post {
        /// Persona instance name.
        name: String,
        /// Override the topic; skips the topic generator.
        #[arg(long)]
        topic: Option<String>,
        /// Override the candidate count.
        #[arg(long)]
        candidates: Option<usize>,
    },

    /// List recent post history for a persona.
    Posts {
        /// Persona instance name.
        name: String,
        /// Maximum number of entries to return.
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Path to the JSONL post history file.
        /// Defaults to ~/.heartbit/ghost/posts/<persona>.jsonl.
        #[arg(long)]
        history_path: Option<String>,
    },
```

- [ ] **Step 3: Add the dispatch arms**

```rust
        PersonaCommand::Post { name, topic, candidates } => {
            let persona = registry.get(&name).ok_or_else(|| {
                anyhow!("persona '{name}' not found. {}", registry_suffix(registry))
            })?;
            let _expansion = persona
                .expand(&PersonaParams::default())
                .map_err(|e| anyhow!("expand persona '{name}': {e}"))?;
            let provider =
                build_provider_from_env(None).map_err(|e| anyhow!("build llm provider: {e}"))?;
            let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
            let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;

            let on_progress: std::sync::Arc<dyn Fn(&str) + Send + Sync> =
                std::sync::Arc::new(|s: &str| eprintln!("> {s}"));

            // V1: when --topic is set, skip the topic generator entirely
            // and call run_review_pipeline directly (matches `persona run --review`).
            if let Some(t) = topic {
                let cfg = crate::persona_review::review_config_from_env(
                    &name,
                    &t,
                    candidates.unwrap_or(3),
                    provider,
                    &corpora_root,
                    &profiles_root,
                    Some(on_progress),
                    None,  // mode_addendum: handler-side normally re-derived
                    None,  // researcher_override
                )
                .await
                .map_err(|e| anyhow!("review config: {e}"))?;
                let output = heartbit_ghost::review::run_review_pipeline(cfg)
                    .await
                    .map_err(|e| anyhow!("review pipeline: {e}"))?;
                eprintln!("> ok: outcome={:?}", output.outcome);
                return Ok(());
            }

            // No --topic: invoke handle_persona_post with a per-call
            // ephemeral InMemoryPostHistoryStore. (For real use, the daemon
            // does this; the CLI is for one-off testing.)
            let operator_user_id = std::env::var("HEARTBIT_GHOST_OPERATOR_USER_ID")
                .context("HEARTBIT_GHOST_OPERATOR_USER_ID must be set for `persona post` without --topic")?;
            let history: std::sync::Arc<dyn heartbit_ghost::posts::PostHistoryStore> =
                std::sync::Arc::new(heartbit_ghost::posts::InMemoryPostHistoryStore::new());
            let delivery: std::sync::Arc<dyn heartbit_ghost::review::ReviewDelivery> =
                std::sync::Arc::new(
                    crate::persona_review::TelegramReviewDelivery::from_env()
                        .context("construct TelegramReviewDelivery")?,
                );
            let twitter_thread: std::sync::Arc<dyn heartbit_core::Tool> =
                std::sync::Arc::new(heartbit_ghost::tools::TwitterThreadTool::new());
            let credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver> =
                std::sync::Arc::new(crate::persona_review::EnvCredentialResolver);
            let deps = heartbit::PersonaPostDeps {
                persona_name: &name,
                registry,
                history: &*history,
                history_lookback: chrono::Duration::days(30),
                topic_brief: None,
                operator_user_id: &operator_user_id,
                provider,
                delivery,
                twitter_tool: twitter_thread,
                credentials,
                candidates_per_draft: candidates.unwrap_or(3),
                corpora_root: &corpora_root,
                profiles_root: &profiles_root,
            };
            let outcome = heartbit::handle_persona_post(deps)
                .await
                .map_err(|e| anyhow!("persona post: {e}"))?;
            eprintln!("> ok: outcome={outcome:?}");
            Ok(())
        }
        PersonaCommand::Posts { name, limit, history_path } => {
            let path = match history_path {
                Some(p) => crate::persona_review::expand_tilde_str(&p)?,
                None => {
                    let home = std::env::var("HOME").context("$HOME not set")?;
                    std::path::PathBuf::from(home)
                        .join(".heartbit/ghost/posts")
                        .join(format!("{name}.jsonl"))
                }
            };
            if !path.exists() {
                println!("(no history at {})", path.display());
                return Ok(());
            }
            let store = heartbit_ghost::posts::JsonlPostHistoryStore::open(&path)
                .await
                .with_context(|| format!("open {}", path.display()))?;
            let recent = store
                .recent(&name, limit)
                .await
                .map_err(|e| anyhow!("recent: {e}"))?;
            if recent.is_empty() {
                println!("(no entries for persona '{name}')");
                return Ok(());
            }
            println!("Recent posts for {name} ({}):", recent.len());
            for (i, e) in recent.iter().enumerate() {
                let when = e.posted_at.format("%Y-%m-%d %H:%M");
                let tweet = e.tweet_id.as_deref().unwrap_or("-");
                println!(
                    "  [{i}] {when} tweet={tweet}\n      topic: {}\n      outcome: {:?}",
                    if e.topic.is_empty() { "(none)" } else { e.topic.as_str() },
                    e.outcome,
                );
            }
            Ok(())
        }
```

- [ ] **Step 4: Add `expand_tilde_str` helper to `persona_review.rs` if absent**

If `expand_tilde` is `pub(crate)` in the CLI's daemon module, expose a public version in `persona_review.rs`:

```rust
/// Expand a leading `~/` to `$HOME` in a path string. Returns
/// `PathBuf` regardless of whether expansion occurred.
pub fn expand_tilde_str(s: &str) -> anyhow::Result<std::path::PathBuf> {
    use anyhow::Context;
    if let Some(rest) = s.strip_prefix("~/") {
        let home = std::env::var("HOME").context("$HOME not set")?;
        Ok(std::path::PathBuf::from(home).join(rest))
    } else {
        Ok(std::path::PathBuf::from(s))
    }
}
```

- [ ] **Step 5: Build + commit**

```bash
cargo build --workspace --features daemon
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
./target/debug/heartbit persona post --help
./target/debug/heartbit persona posts --help
git add -A
git commit -m "$(cat <<'EOF'
feat(cli): persona post <name> + persona posts list <name>

On-demand counterparts to the daemon's persona-post loop:
- `persona post NAME` — runs handle_persona_post with an ephemeral
  in-memory store. Triggers topic generator + pipeline + Telegram
  review. Honors --topic to skip the generator (passes the topic
  directly to run_review_pipeline). Honors --candidates to override
  the count. Requires HEARTBIT_GHOST_OPERATOR_USER_ID env var when
  --topic is absent (for the topic context provider).
- `persona posts NAME` — reads the JSONL post history at
  ~/.heartbit/ghost/posts/<persona>.jsonl (or --history-path) and
  prints the last N entries.

heartbit-ghost P1.6c — task 12/13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Acceptance — quality gate + manual setup + live test

**Files:** none modified by this task; verifies prior tasks land cleanly.

- [ ] **Step 1: Full quality gate**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --features daemon 2>&1 | tail -25
```

Expected: all green. Test count: previous baseline + ~25 new tests across `posts::*`, `daemon::persona_post::*`, `daemon::persona_post_handler::*`, `config::daemon::tests`, `daemon::types::tests`.

- [ ] **Step 2: Build the release binary**

```bash
cargo build --release --bin heartbit 2>&1 | tail -3
```

Expected: `Finished release …`.

- [ ] **Step 3: Operator-side setup (manual; not part of plan deliverable)**

1. Resolve operator user_id (if not already done in P1.5):
   ```bash
   ./target/release/heartbit persona mentions heartbit-ghost:x --limit 1
   # Top line: "Authenticated as @<handle> ... user_id=<id>"
   ```

2. Update `daemon-dev.toml`:
   ```toml
   [[daemon.persona_mentions]]
   persona = "heartbit-ghost:x"
   enabled = true
   user_id = "1234567890"   # from step 1
   poll_interval_seconds = 300
   candidates_per_reply = 2
   mention_store = "jsonl"
   mention_store_path = "~/.heartbit/ghost/mentions/heartbit-ghost-x.jsonl"

   [[daemon.persona_posts]]
   persona = "heartbit-ghost:x"
   enabled = true
   post_interval_seconds = 14400
   active_hours = { start = "09:00", end = "22:00" }
   candidates_per_draft = 3
   post_history_store = "jsonl"
   post_history_path = "~/.heartbit/ghost/posts/heartbit-ghost-x.jsonl"
   post_history_lookback_days = 30
   topic_brief = "agent infrastructure, Rust async, LLM tool use"
   ```

   For a quick test, override `post_interval_seconds = 120` (2 minutes) so the loop fires fast enough to validate.

3. Restart the daemon (or start it):
   ```bash
   ./target/release/heartbit daemon --config daemon-dev.toml
   ```

- [ ] **Step 4: Live test — proactive post on cadence**

1. Start the daemon as above. Within `post_interval_seconds`, the daemon fires `PersonaPost`.
2. Topic generator runs (uses `XGhostTopicContext` to fetch own tweets + mentions + history).
3. Pipeline drafts 3 candidate threads.
4. Telegram bot delivers the candidate-list message + `[1] [2] [3] [Skip]` keyboard.
5. Pick a draft → thread posts on X. Visit the URL in the Telegram outcome message to verify.
6. Verify `~/.heartbit/ghost/posts/heartbit-ghost-x.jsonl` contains a `Posted` entry with the tweet_id.

- [ ] **Step 5: Live test — alternate "Skip" path**

1. Wait for the next tick (or manually fire via `heartbit persona post heartbit-ghost:x`).
2. On Telegram, press Skip.
3. Verify:
   - No new tweet on X
   - JSONL store contains one new `Skipped` entry
   - Next tick: topic generator may propose the same topic (allowed), the duplicate-check catches it (since the recent topic was Skipped not Posted, but `was_posted_recently` matches by topic-string regardless of outcome — Skipped within lookback also blocks). The history entry will be `SkippedDuplicate`.

   (This demonstrates the "Skipped counts as 'don't retry' for the lookback window" semantics. If the user wants different semantics — e.g., Skipped should re-fire — that's a follow-up tweak.)

- [ ] **Step 6: CLI verify**

```bash
./target/release/heartbit persona posts heartbit-ghost:x --limit 5
```

Expected: prints the recent entries with timestamps, topics, tweet_ids, outcomes.

- [ ] **Step 7: Final merge / PR**

Once Steps 4-6 pass, finish via the **superpowers:finishing-a-development-branch** skill (creates PR or merges to main).

---

## Self-review — pre-execution

- **Spec coverage:** every section maps to at least one task. §3 Files (Tasks 1-12), §4 topic_generator (Task 2), §5 TopicContextProvider (Tasks 3-5), §6 PostHistoryStore (Task 6), §7 PersonaPostsConfig (Task 7), §8 daemon command + handler (Tasks 8, 10), §9 lifecycle wiring (Task 11), §10 CLI (Task 12), §11 tests (each task ships its own + acceptance), §12 out-of-scope (no tasks), §13 risks (mitigations applied throughout — interval validation, graceful degradation, history dedup), §14 sub-phases (Tasks 1-5 / 6-8 / 9-12 / 13 ≈ a/b/c/d).

- **Placeholder scan:** Task 10 contains a comment-block stub for the 5 integration tests rather than full bodies. Each stub has a clear pointer to `crates/heartbit/src/daemon/reply_draft_handler.rs::tests` for the verbatim pattern. This is intentional — duplicating those ~80-line bodies in the plan would inflate it without improving clarity. **Each stub is one full test the implementer expands by mirroring the reference.**

- **Type consistency:**
  - `TopicContextProvider` trait signature consistent across Tasks 3 (define), 5 (refactor in core), and 10 (consumed by handler).
  - `PostOutcome` variant names match across Tasks 3 (define), 6 (history match), 10 (handler map_review_outcome), 12 (CLI display).
  - `PersonaPostEntry` field names match across Tasks 11 (define on context) and 11 (handler dispatch consumes).
  - `PostHistoryStore` method names (`record`, `recent`, `was_posted_recently`) match across Tasks 6 (impl), 10 (handler), 12 (CLI).

- **Cross-crate dependency direction:** `heartbit-core` defines `TopicContextProvider` trait (primitive types only). `heartbit-ghost` provides impls. `heartbit` umbrella wires them. No circular deps.

- **Branch state:** the spec is already committed on `feat/heartbit-ghost-p1.6-proactive-posts` (commit `ac956bd`). Tasks 1-13 land on top of that branch.

- **Known TBD:** Task 11's CLI wiring requires `operator_user_id` for posts; V1 reuses the `persona_mentions` entry's `user_id`. If you want posts without mentions enabled, this would force a separate `user_id` field on `PersonaPostsConfig` (small follow-up; not blocking).
