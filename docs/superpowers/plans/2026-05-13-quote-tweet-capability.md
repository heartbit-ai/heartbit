# Quote-Tweet Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third content vector to heartbit-ghost: poll a curated list of X accounts, draft an opinionated-but-charitable quote-tweet for selected tweets, route through the existing Telegram review pipeline, post via `POST /2/tweets` with `quote_tweet_id`.

**Architecture:** Mirrors the existing `persona_mentions` + `persona_posts` patterns end-to-end. New `[[daemon.persona_quotes]]` config block → `PersonaQuoteScheduler` fires `DaemonCommand::PersonaQuote` on a jittered cadence → `handle_persona_quote` polls source accounts via X v2 `/2/users/{id}/tweets`, picks one un-quoted candidate, runs `research → quote_writer → critic → fact_check → pre-filter → Telegram review → POST /2/tweets`. The quote_writer agent has a distinct Catholic-charity disposition (opinionated, good faith, never aggressive) layered ONLY here — proactive posts and replies keep their current voice.

**Tech Stack:** Rust 1.x, existing X v2 client (`crates/heartbit-ghost/src/tools/client.rs`), tokio, axum, sqlx (optional), thiserror, anyhow, wiremock for tests.

---

## File Structure

**Created:**
- `crates/heartbit-ghost/src/tools/quote.rs` — `TwitterQuoteTool` (POST /2/tweets with quote_tweet_id)
- `crates/heartbit-ghost/src/agents/quote_writer.rs` — quote_writer recipe with Catholic-charity prompt
- `crates/heartbit-ghost/src/quote/mod.rs` — quote pipeline (`run_quote_pipeline`, `QuoteConfig`, outcomes)
- `crates/heartbit-ghost/src/quote/sources.rs` — `QuoteSource` trait + `XUserTimelineSource` + `InMemoryQuoteSeenStore` + `JsonlQuoteSeenStore` (dedup of already-quoted tweet IDs)
- `crates/heartbit-ghost/src/quote/prompts.rs` — user-message builders for quote_research + quote_writer + quote_critic
- `crates/heartbit/src/daemon/persona_quote.rs` — `PersonaQuoteScheduler` (jittered cron-style trigger)
- `crates/heartbit/src/daemon/persona_quote_handler.rs` — `handle_persona_quote` + `PersonaQuoteDeps` + `QuoteContext`
- `crates/heartbit/src/daemon/quotes_context.rs` — `PersonaQuoteEntry` + `QuotesContext` (shared across handler invocations)

**Modified:**
- `crates/heartbit-core/src/config/daemon.rs` — add `PersonaQuotesConfig` + `DaemonConfig.persona_quotes: Vec<...>`
- `crates/heartbit-core/src/config/mod.rs` — re-export `PersonaQuotesConfig`
- `crates/heartbit/src/lib.rs` — re-export from `heartbit-core`
- `crates/heartbit/src/daemon/types.rs` — add `DaemonCommand::PersonaQuote { persona: String }`
- `crates/heartbit/src/daemon/mod.rs` — declare modules, re-exports
- `crates/heartbit/src/daemon/core.rs` — spawn `PersonaQuoteScheduler` per enabled entry, dispatch `PersonaQuote` to handler
- `crates/heartbit-cli/src/daemon/mod.rs` — parse `[[daemon.persona_quotes]]`, build `QuotesContext`, wire writer_provider override
- `docs/operating-heartbit.md` — document `[[daemon.persona_quotes]]` knobs
- `daemon-dev.toml` (operator-local, gitignored) — commented example block

**Important: NOT modified:**
- `crates/heartbit-ghost/src/agents/writer.rs` (proactive writer) — voice unchanged
- `crates/heartbit-ghost/src/agents/reply_writer.rs` — voice unchanged
- The "strict Catholic charity" disposition lands ONLY in `quote_writer.rs`. Per the brainstorm decision, this scope is intentional.

---

## Catholic Charity Disposition — Prompt Design

The quote_writer's system prompt must enforce a specific behavioral disposition: **opinionated, good-faith disagreement, never aggressive, grounded in Catholic charity (caritas in veritate — truth in love).** This is the operator's chosen voice for the bot's quote-tweet surface. The plan locks specific phrasing into Task 3 with regression tests so future edits don't soften it.

Key principles (operator-specified, implementation-faithful):
- **Opinionated**: never a content-free "interesting take". Take a clear position.
- **Good faith**: charitably interpret the source author's strongest version of their argument.
- **Disagree clearly**: when wrong, say so plainly. Never hedge into mush.
- **Never aggressive**: no sneering, no contempt, no mockery, no insults, no profanity, no ad hominem. Engage the argument, not the person.
- **Catholic charity**: respect the human dignity of every interlocutor including those you disagree with. Truth in love, love in truth.
- **Concrete reasoning**: ground stances in specific reasons, not slogans.

---

## Task 1: Config schema — `PersonaQuotesConfig`

**Files:**
- Modify: `crates/heartbit-core/src/config/daemon.rs`
- Modify: `crates/heartbit-core/src/config/mod.rs` (re-export)
- Modify: `crates/heartbit/src/lib.rs` (umbrella re-export)

- [ ] **Step 1: Write the failing test**

Append to the existing `#[cfg(test)] mod tests` block in `crates/heartbit-core/src/config/daemon.rs`:

```rust
#[test]
fn persona_quotes_config_parses_with_defaults() {
    let toml = r#"
[[persona_quotes]]
persona = "heartbit-ghost:x"
source_user_ids = ["44196397", "16884623"]
"#;
    #[derive(Deserialize)]
    struct Shim {
        persona_quotes: Vec<PersonaQuotesConfig>,
    }
    let cfg: Shim = toml::from_str(toml).unwrap();
    assert_eq!(cfg.persona_quotes.len(), 1);
    let q = &cfg.persona_quotes[0];
    assert_eq!(q.persona, "heartbit-ghost:x");
    assert!(q.enabled);
    assert_eq!(q.poll_interval_seconds, 5400); // default 90 min
    assert_eq!(q.interval_jitter_pct, 25);
    assert!(q.active_hours.is_none());
    assert_eq!(q.candidates_per_draft, 3);
    assert_eq!(q.seen_store, "in_memory");
    assert!(q.seen_store_path.is_none());
    assert_eq!(q.max_age_hours, 12);
    assert_eq!(q.max_candidates_per_tick, 1);
    assert_eq!(q.source_user_ids, vec!["44196397", "16884623"]);
    assert!(q.writer_provider.is_none());
}

#[test]
fn persona_quotes_config_rejects_missing_required_fields() {
    let toml = r#"
[[persona_quotes]]
persona = "heartbit-ghost:x"
"#;
    // source_user_ids is required (no default). Parse must fail.
    #[derive(Deserialize)]
    struct Shim {
        #[allow(dead_code)]
        persona_quotes: Vec<PersonaQuotesConfig>,
    }
    let err = toml::from_str::<Shim>(toml).unwrap_err();
    assert!(
        err.to_string().contains("source_user_ids"),
        "expected missing-field error for source_user_ids; got: {err}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package heartbit-core --lib config::daemon::tests::persona_quotes_config_parses_with_defaults`

Expected: compile error — `PersonaQuotesConfig` not defined.

- [ ] **Step 3: Add the struct + defaults**

In `crates/heartbit-core/src/config/daemon.rs`, append after the `PersonaPostsConfig` block (after its default-fn helpers):

```rust
/// Per-persona quote-tweet configuration.
///
/// When present, the daemon registers a `PersonaQuoteScheduler` that
/// fires `DaemonCommand::PersonaQuote` on the configured cadence. The
/// handler polls each `source_user_ids` entry via X v2
/// `/2/users/{id}/tweets`, filters by `max_age_hours` + not-yet-quoted,
/// drafts an opinionated-but-charitable quote-tweet via the
/// `quote_writer` agent, sends to Telegram for review, posts the
/// chosen draft via `POST /2/tweets` with `quote_tweet_id`.
///
/// Configured under `[[daemon.persona_quotes]]` blocks.
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaQuotesConfig {
    /// Persona registry name (e.g. `"heartbit-ghost:x"`).
    pub persona: String,
    /// Whether this quoter is enabled.
    #[serde(default = "super::default_true")]
    pub enabled: bool,
    /// Polling interval, in seconds. Default 5400 (90 min — medium per
    /// brainstorm).
    /// Validation: must be ≥60 (rejected at config load otherwise).
    #[serde(default = "default_quote_poll_interval_seconds")]
    pub poll_interval_seconds: u64,
    /// Randomness applied to each `poll_interval_seconds` tick, as a
    /// percentage (`0`–`50`). Default `25` = ±25%. Same anti-pattern
    /// rationale as `interval_jitter_pct` on `persona_posts`.
    #[serde(default = "default_quote_interval_jitter_pct")]
    pub interval_jitter_pct: u32,
    /// Optional `"HH:MM-HH:MM"` window during which quote-polls run.
    /// Outside this window, the scheduler tick is a no-op.
    #[serde(default)]
    pub active_hours: Option<ActiveHoursConfig>,
    /// X user IDs (numeric strings) of accounts to poll. Curated list.
    /// Required — must contain at least one entry.
    pub source_user_ids: Vec<String>,
    /// Number of candidate quote-tweets to draft per chosen source
    /// tweet (1..=5). Default 3.
    #[serde(default = "default_quote_candidates_per_draft")]
    pub candidates_per_draft: usize,
    /// Backend for the "already quoted" dedup store: `"in_memory"` or
    /// `"jsonl"`. Default `"in_memory"`.
    #[serde(default = "default_quote_seen_store")]
    pub seen_store: String,
    /// Path to the JSONL store file (only used when
    /// `seen_store == "jsonl"`). Tilde expansion at construction time.
    #[serde(default)]
    pub seen_store_path: Option<String>,
    /// Maximum age in hours of a source tweet for it to be quote-able.
    /// Default 12 — beyond that the discourse has moved on. Set to 0
    /// to disable the age filter.
    #[serde(default = "default_quote_max_age_hours")]
    pub max_age_hours: i64,
    /// Maximum number of quote-tweets to draft+review per scheduler
    /// tick. Default 1 — pick the best candidate from sources and
    /// stop. Set higher to draft multiple quote-tweets per tick
    /// (one Telegram review per draft).
    #[serde(default = "default_quote_max_candidates_per_tick")]
    pub max_candidates_per_tick: usize,
    /// Optional override LLM provider for the quote_writer + critic.
    /// `None` falls back to the global `[provider]`. Same shape as
    /// `persona_posts.writer_provider`.
    #[serde(default)]
    pub writer_provider: Option<super::agent::AgentProviderConfig>,
}

fn default_quote_poll_interval_seconds() -> u64 {
    5400 // 90 minutes
}

fn default_quote_interval_jitter_pct() -> u32 {
    25
}

fn default_quote_candidates_per_draft() -> usize {
    3
}

fn default_quote_seen_store() -> String {
    "in_memory".into()
}

fn default_quote_max_age_hours() -> i64 {
    12
}

fn default_quote_max_candidates_per_tick() -> usize {
    1
}
```

Add to the `DaemonConfig` struct (find it near the top of the file, around line 18):

```rust
    /// Per-persona quote-tweet configurations. Each entry launches a
    /// `PersonaQuoteScheduler`.
    #[serde(default)]
    pub persona_quotes: Vec<PersonaQuotesConfig>,
```

In `crates/heartbit-core/src/config/mod.rs`, find the `pub use daemon::{...}` re-export line and add `PersonaQuotesConfig`:

```rust
pub use daemon::{
    // ... existing entries ...
    PersonaQuotesConfig,
};
```

In `crates/heartbit/src/lib.rs`, find the same pattern for the umbrella re-export and add `PersonaQuotesConfig`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --package heartbit-core --lib config::daemon::tests::persona_quotes`

Expected: both tests PASS.

- [ ] **Step 5: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-core -- -D warnings && cargo test --package heartbit-core --lib`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-core/src/config/daemon.rs crates/heartbit-core/src/config/mod.rs crates/heartbit/src/lib.rs
git commit -m "feat(config): PersonaQuotesConfig + DaemonConfig.persona_quotes"
```

---

## Task 2: `TwitterQuoteTool` — X v2 quote-tweet posting

**Files:**
- Create: `crates/heartbit-ghost/src/tools/quote.rs`
- Modify: `crates/heartbit-ghost/src/tools/mod.rs` (declare + re-export)

- [ ] **Step 1: Write the failing test**

Create `crates/heartbit-ghost/src/tools/quote.rs` with the full implementation block + tests below. (TDD purists: comment out the function body, watch tests fail, uncomment. The single-file pattern matches `tools/reply.rs`.)

```rust
//! `twitter_quote` — post a quote-tweet (a new tweet that references
//! an existing tweet via the X v2 `quote_tweet_id` field).

use std::future::Future;
use std::pin::Pin;

use heartbit_core::llm::types::ToolDefinition;
use heartbit_core::{ExecutionContext, Tool, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::tools::client::{XApiError, XClient, format_error};

const MAX_TWEET_LENGTH: usize = 280;

#[derive(Debug, Deserialize)]
struct QuoteInput {
    text: String,
    quote_tweet_id: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct QuoteOutput {
    tweet_id: String,
    url: String,
}

/// X v2 POST /2/tweets request body for a quote-tweet.
#[derive(Debug, Serialize)]
struct QuoteRequest<'a> {
    text: &'a str,
    quote_tweet_id: &'a str,
}

#[derive(Debug, Deserialize)]
struct QuoteApiResponse {
    data: QuoteApiData,
}

#[derive(Debug, Deserialize)]
struct QuoteApiData {
    id: String,
}

/// Post a quote-tweet by `text` + `quote_tweet_id`.
pub struct TwitterQuoteTool;

impl Default for TwitterQuoteTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TwitterQuoteTool {
    /// Create the tool. Credentials are resolved at execute-time via
    /// `ExecutionContext::credentials`.
    pub fn new() -> Self {
        Self
    }
}

impl Tool for TwitterQuoteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "twitter_quote".into(),
            description: "Post a quote-tweet on X. Wraps POST /2/tweets with quote_tweet_id. Max 280 chars of comment text.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Quote-tweet comment text. Max 280 characters.",
                        "maxLength": 280
                    },
                    "quote_tweet_id": {
                        "type": "string",
                        "description": "Numeric tweet id of the tweet being quoted."
                    }
                },
                "required": ["text", "quote_tweet_id"]
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
            let parsed: QuoteInput = match serde_json::from_value(input) {
                Ok(v) => v,
                Err(e) => return Ok(ToolOutput::error(format!("invalid input: {e}"))),
            };
            let char_count = parsed.text.chars().count();
            if parsed.text.is_empty() {
                return Ok(ToolOutput::error("text must not be empty"));
            }
            if char_count > MAX_TWEET_LENGTH {
                return Ok(ToolOutput::error(format!(
                    "Quote text exceeds {MAX_TWEET_LENGTH} characters (got {char_count})."
                )));
            }
            if parsed.quote_tweet_id.is_empty()
                || !parsed.quote_tweet_id.chars().all(|c| c.is_ascii_digit())
            {
                return Ok(ToolOutput::error(
                    "quote_tweet_id must be a non-empty numeric string",
                ));
            }
            let client = match XClient::from_context(&ctx).await {
                Ok(c) => c,
                Err(e) => return Ok(ToolOutput::error(format_error(&e))),
            };
            match call_x(&client, &parsed).await {
                Ok(out) => {
                    let json =
                        serde_json::to_string(&out).expect("QuoteOutput fields are infallible");
                    Ok(ToolOutput::success(json))
                }
                Err(e) => Ok(ToolOutput::error(format_error(&e))),
            }
        })
    }
}

async fn call_x(client: &XClient, input: &QuoteInput) -> Result<QuoteOutput, XApiError> {
    let body = QuoteRequest {
        text: &input.text,
        quote_tweet_id: &input.quote_tweet_id,
    };
    let response: QuoteApiResponse = client.post_json("/2/tweets", &body).await?;
    let tweet_id = response.data.id;
    let url = format!("https://twitter.com/i/web/status/{}", tweet_id);
    Ok(QuoteOutput { tweet_id, url })
}

#[cfg(test)]
mod tests {
    use super::*;
    use heartbit_core::Secret;
    use wiremock::matchers::{body_json, method, path as wm_path};
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
    async fn quote_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wm_path("/2/tweets"))
            .and(body_json(serde_json::json!({
                "text": "thoughtful agreement",
                "quote_tweet_id": "9001"
            })))
            .respond_with(
                ResponseTemplate::new(201)
                    .set_body_json(serde_json::json!({"data": {"id": "9999"}})),
            )
            .mount(&server)
            .await;

        let client = test_client(&server.uri());
        let input = QuoteInput {
            text: "thoughtful agreement".into(),
            quote_tweet_id: "9001".into(),
        };
        let result = call_x(&client, &input).await.expect("happy path");
        assert_eq!(result.tweet_id, "9999");
        assert!(result.url.contains("9999"));
    }

    #[tokio::test]
    async fn quote_rejects_text_over_280_chars() {
        // Validation happens BEFORE the HTTP call — no server needed.
        let tool = TwitterQuoteTool::new();
        let input = serde_json::json!({
            "text": "x".repeat(281),
            "quote_tweet_id": "9001"
        });
        let ctx = ExecutionContext::default();
        let out = tool
            .execute(&ctx, input)
            .await
            .expect("validation returns Ok(ToolOutput::error)");
        assert!(
            out.content.contains("exceeds 280"),
            "expected length-rejection; got: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn quote_rejects_non_numeric_id() {
        let tool = TwitterQuoteTool::new();
        let input = serde_json::json!({
            "text": "ok",
            "quote_tweet_id": "not-a-number"
        });
        let ctx = ExecutionContext::default();
        let out = tool.execute(&ctx, input).await.expect("Ok");
        assert!(
            out.content.contains("non-empty numeric string"),
            "got: {}",
            out.content
        );
    }
}
```

Then in `crates/heartbit-ghost/src/tools/mod.rs`, add the declaration + re-export. Find the existing pattern (search for `pub mod reply;` and `pub use reply::TwitterReplyTool;`) and add alongside:

```rust
pub mod quote;
pub use quote::TwitterQuoteTool;
```

- [ ] **Step 2: Run tests**

Run: `cargo test --package heartbit-ghost --lib tools::quote`

Expected: 3 tests PASS (happy path, length rejection, non-numeric id rejection).

- [ ] **Step 3: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-ghost/src/tools/quote.rs crates/heartbit-ghost/src/tools/mod.rs
git commit -m "feat(ghost): TwitterQuoteTool — POST /2/tweets with quote_tweet_id"
```

---

## Task 3: `quote_writer` agent recipe — Catholic-charity disposition

**Files:**
- Create: `crates/heartbit-ghost/src/agents/quote_writer.rs`
- Modify: `crates/heartbit-ghost/src/agents/mod.rs` (declare + re-export)

- [ ] **Step 1: Create the recipe file with prompt + tests**

Create `crates/heartbit-ghost/src/agents/quote_writer.rs`:

```rust
//! Quote-writer sub-agent — composes a single ≤280-char quote-tweet
//! comment that engages opinionatedly but charitably with the quoted
//! tweet. Used ONLY in the quote-tweet pipeline; proactive posts and
//! replies use the separate `writer` and `reply_writer` recipes.

use heartbit_core::config::AgentConfig;

/// System prompt for the quote_writer.
///
/// The disposition: **opinionated, good-faith, never aggressive,
/// grounded in Catholic charity (caritas in veritate — truth in love).**
/// This is a deliberate persona choice for the quote-tweet surface.
/// Future edits that soften "never aggressive" or "never sneering" must
/// also update the regression tests below.
pub const QUOTE_WRITER_SYSTEM_PROMPT: &str = r#"You compose a single short comment (≤280 characters) that quote-tweets an existing X post. Your job is to engage with the quoted post's claim — agreeing, disagreeing, or refining — in a way that lands an opinion clearly and charitably.

INPUT (from the user message)
- The QUOTED tweet (its text + author handle).
- Optional: the original author's bio + 2-3 recent tweets for tone calibration.
- A research digest the researcher built on the topic the quoted tweet raises.
- Voice guidelines for the persona.
- Target language (the language of the quoted tweet — mirror it exactly).

OUTPUT
The comment text only. No preamble, no quotation marks around it, no markdown. ≤280 characters HARD CAP — count includes spaces and emoji. Aim for 80-200 characters; brevity reads as confidence.

DISPOSITION — NON-NEGOTIABLE
You are opinionated and you take a position. But you do so in the spirit of caritas in veritate — truth in love. This means:

1. CHARITABLE INTERPRETATION. Engage with the strongest version of the author's argument, not a weak caricature. If their claim is ambiguous, pick the most generous reading and respond to that.

2. CLEAR DISAGREEMENT WHEN WARRANTED. When you disagree, say so plainly. "I disagree because…" or "this misses…" beats hedging into mush. Truth-seeking is itself an act of respect for the interlocutor.

3. NEVER AGGRESSIVE. No sneering, no mockery, no contempt, no insults, no profanity, no blasphemy, no ad hominem attacks. Engage the argument; never the person. Dismissive one-liners ("lol no", "this is stupid", "what an idiot") are forbidden regardless of how wrong the original is.

4. RESPECT FOR HUMAN DIGNITY. The interlocutor is a person made in the image of God, with whom you may disagree but whose dignity you never demean. This applies even when responding to bad-faith content.

5. CONCRETE REASONING. Ground stances in specific reasons, not slogans. "Because X" beats "obviously wrong". When you cite a fact, cite it specifically; when you reason from principle, name the principle.

6. AGREEMENT IS ALSO VALID. If the quoted post is right, say so and add to it — a useful corollary, a relevant case, a sharpening. Not every quote-tweet needs to be a disagreement.

TONE LADDER (in order of preference)
1. Substantive agreement + extension ("yes, and here's why this matters more than people realize")
2. Substantive disagreement with a clear reason ("I disagree — here's the case that's stronger")
3. Refinement ("the claim is roughly right but the framing buries the key trade-off, which is…")
4. Honest acknowledgement of uncertainty ("I'm not sure — the data I've seen on this is X, and that cuts both ways")
5. "no_quote" if no substantive engagement is possible

If the quoted tweet is hostile, dehumanizing, blasphemous, or in obviously bad faith, output the literal string "no_quote" and stop. Do not engage with content whose engagement would itself violate the disposition above.

FORMAT — HARD CONSTRAINTS
- ≤280 characters HARD CAP. Counts include spaces and emoji.
- Never use exclamation marks unless the persona voice explicitly allows them.
- Never @-mention the original author. X auto-attributes quote-tweets.
- Never start with "Thanks for…", "Great point…", "Interesting…" — these are AI tells.
- Voice MUST match the persona's voice guidelines (no em-dashes if forbidden, formatting rules, AI-tells to avoid).

SOURCING — ZERO TOLERANCE FOR INVENTION
- Every quantitative claim (number, percentage, dollar amount, date, version) you make MUST trace to the research digest. Never paraphrase or approximate.
- If you don't have a sourced number for a point, reframe qualitatively or drop the point. Never invent precision to sound sharper.
"#;

/// Construct the quote_writer [`AgentConfig`].
pub fn quote_writer_recipe() -> AgentConfig {
    AgentConfig {
        name: "quote_writer".to_string(),
        description: "Compose a single ≤280-char quote-tweet comment, opinionated but charitable.".to_string(),
        system_prompt: QUOTE_WRITER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(512),
        reasoning_effort: Some("medium".to_string()),
        ..super::stub_recipe("quote_writer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_writer_recipe_has_expected_shape() {
        let cfg = quote_writer_recipe();
        assert_eq!(cfg.name, "quote_writer");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            cfg.response_schema.is_none(),
            "quote_writer produces free-form text, no schema"
        );
    }

    /// Regression: the disposition phrasing is the load-bearing part of
    /// this prompt. Soften it and the bot starts sneering on X.
    #[test]
    fn quote_writer_prompt_states_charity_disposition() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(p.contains("caritas in veritate"), "must cite caritas in veritate");
        assert!(p.contains("NEVER AGGRESSIVE"), "must forbid aggression");
        assert!(p.contains("CHARITABLE INTERPRETATION"), "must require charitable interpretation");
        assert!(p.contains("ad hominem"), "must forbid ad hominem");
        assert!(p.contains("human dignity") || p.contains("HUMAN DIGNITY"), "must invoke human dignity");
        assert!(
            p.contains("no sneering") || p.contains("No sneering"),
            "must forbid sneering"
        );
        assert!(
            p.contains("mockery"),
            "must forbid mockery"
        );
    }

    #[test]
    fn quote_writer_prompt_allows_disagreement() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("CLEAR DISAGREEMENT") || p.contains("clear disagreement"),
            "the disposition is opinionated, NOT mealy-mouthed; clear disagreement must be allowed"
        );
        assert!(
            p.contains("AGREEMENT IS ALSO VALID") || p.contains("agreement is also valid"),
            "must permit agreement+extension; not every quote is a disagreement"
        );
    }

    #[test]
    fn quote_writer_prompt_has_no_quote_escape_hatch() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("no_quote"),
            "must define a no_quote escape hatch for bad-faith content"
        );
    }

    #[test]
    fn quote_writer_prompt_enforces_280_char_cap() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(p.contains("280 characters HARD CAP"));
    }

    #[test]
    fn quote_writer_prompt_enforces_zero_tolerance_sourcing() {
        let p = QUOTE_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("ZERO TOLERANCE FOR INVENTION"),
            "consistency with proactive writer + reply_writer chain"
        );
    }
}
```

Then in `crates/heartbit-ghost/src/agents/mod.rs`, add the declaration + re-export. Find the existing pattern (e.g. `pub mod writer;` and `pub use writer::writer_recipe;`) and add:

```rust
pub mod quote_writer;
pub use quote_writer::{QUOTE_WRITER_SYSTEM_PROMPT, quote_writer_recipe};
```

- [ ] **Step 2: Run tests**

Run: `cargo test --package heartbit-ghost --lib agents::quote_writer`

Expected: 6 tests PASS.

- [ ] **Step 3: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/heartbit-ghost/src/agents/quote_writer.rs crates/heartbit-ghost/src/agents/mod.rs
git commit -m "feat(ghost): quote_writer recipe with caritas-in-veritate disposition"
```

---

## Task 4: Quote source polling + `QuoteSeenStore`

**Files:**
- Create: `crates/heartbit-ghost/src/quote/mod.rs` (empty module scaffolding for now)
- Create: `crates/heartbit-ghost/src/quote/sources.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs` (declare `pub mod quote;`)

- [ ] **Step 1: Create scaffolding + source module**

Create `crates/heartbit-ghost/src/quote/mod.rs` with just the module declaration (later tasks fill it in):

```rust
//! Quote-tweet pipeline — polls curated source accounts, drafts
//! opinionated-but-charitable quote-tweets via the `quote_writer`
//! agent, routes through Telegram review, posts via `twitter_quote`.

pub mod sources;

pub use sources::{
    InMemoryQuoteSeenStore, JsonlQuoteSeenStore, QuoteCandidate, QuoteSeenStore,
    XUserTimelineSource,
};
```

In `crates/heartbit-ghost/src/lib.rs`, find the existing `pub mod reply;` line and add alongside:

```rust
pub mod quote;
```

- [ ] **Step 2: Write failing tests for the source module**

Create `crates/heartbit-ghost/src/quote/sources.rs`:

```rust
//! Source-tweet polling + already-quoted dedup store.
//!
//! `XUserTimelineSource` fetches recent tweets from a curated X user's
//! timeline via `GET /2/users/{id}/tweets` (the same endpoint
//! `posts::topic_context::fetch_own_tweets` uses). `QuoteSeenStore`
//! tracks which source tweet IDs the persona has already quoted so we
//! don't double-quote.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::tools::client::{XApiError, XClient};

/// A candidate source tweet returned from `XUserTimelineSource::recent`.
///
/// `id`, `text`, and `author_handle` are the load-bearing fields for
/// quote drafting. `posted_at` is used by the age filter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuoteCandidate {
    pub id: String,
    pub text: String,
    pub author_id: String,
    pub author_handle: String,
    pub posted_at: DateTime<Utc>,
}

/// Object-safe async trait for fetching recent tweets from a source.
/// Production wires `XUserTimelineSource`; tests wire a mock.
pub trait QuoteSource: Send + Sync {
    fn recent<'a>(
        &'a self,
        user_id: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Vec<QuoteCandidate>, XApiError>> + Send + 'a,
        >,
    >;
}

/// Production source: X v2 `/2/users/{id}/tweets`.
pub struct XUserTimelineSource {
    client: Arc<XClient>,
}

impl XUserTimelineSource {
    pub fn new(client: Arc<XClient>) -> Self {
        Self { client }
    }
}

#[derive(Debug, Deserialize)]
struct TimelineResp {
    #[serde(default)]
    data: Vec<TimelineItem>,
    #[serde(default)]
    includes: Option<TimelineIncludes>,
}

#[derive(Debug, Deserialize)]
struct TimelineItem {
    id: String,
    text: String,
    author_id: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
struct TimelineIncludes {
    #[serde(default)]
    users: Vec<TimelineUser>,
}

#[derive(Debug, Deserialize)]
struct TimelineUser {
    id: String,
    username: String,
}

impl QuoteSource for XUserTimelineSource {
    fn recent<'a>(
        &'a self,
        user_id: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<Vec<QuoteCandidate>, XApiError>> + Send + 'a,
        >,
    > {
        let client = self.client.clone();
        Box::pin(async move {
            let path = format!("/2/users/{user_id}/tweets");
            let query: Vec<(&str, &str)> = vec![
                ("max_results", "10"),
                ("tweet.fields", "author_id,created_at"),
                ("expansions", "author_id"),
                ("user.fields", "username"),
                ("exclude", "replies,retweets"),
            ];
            let resp: TimelineResp = client.get_json(&path, &query).await?;
            let users = resp
                .includes
                .map(|i| i.users)
                .unwrap_or_default();
            let candidates: Vec<QuoteCandidate> = resp
                .data
                .into_iter()
                .map(|t| {
                    let author_handle = users
                        .iter()
                        .find(|u| u.id == t.author_id)
                        .map(|u| u.username.clone())
                        .unwrap_or_else(|| t.author_id.clone());
                    QuoteCandidate {
                        id: t.id,
                        text: t.text,
                        author_id: t.author_id,
                        author_handle,
                        posted_at: t.created_at,
                    }
                })
                .collect();
            Ok(candidates)
        })
    }
}

/// Object-safe async trait for the already-quoted dedup store.
pub trait QuoteSeenStore: Send + Sync {
    /// Record that we've drafted/quoted the given source tweet ID.
    fn record<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), QuoteStoreError>> + Send + 'a>>;

    /// Return true if we've already quoted this source tweet.
    fn was_seen<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool, QuoteStoreError>> + Send + 'a>>;
}

#[derive(Debug, thiserror::Error)]
pub enum QuoteStoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

/// In-memory store (tests + ephemeral runs).
pub struct InMemoryQuoteSeenStore {
    seen: Mutex<std::collections::HashSet<String>>,
}

impl InMemoryQuoteSeenStore {
    pub fn new() -> Self {
        Self {
            seen: Mutex::new(std::collections::HashSet::new()),
        }
    }
}

impl Default for InMemoryQuoteSeenStore {
    fn default() -> Self {
        Self::new()
    }
}

impl QuoteSeenStore for InMemoryQuoteSeenStore {
    fn record<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), QuoteStoreError>> + Send + 'a>>
    {
        Box::pin(async move {
            let key = format!("{persona}:{tweet_id}");
            self.seen.lock().await.insert(key);
            Ok(())
        })
    }

    fn was_seen<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool, QuoteStoreError>> + Send + 'a>>
    {
        Box::pin(async move {
            let key = format!("{persona}:{tweet_id}");
            Ok(self.seen.lock().await.contains(&key))
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SeenEntry {
    persona: String,
    tweet_id: String,
    seen_at: DateTime<Utc>,
}

/// JSONL-backed store for restart durability.
pub struct JsonlQuoteSeenStore {
    path: PathBuf,
    cache: Mutex<std::collections::HashSet<String>>,
}

impl JsonlQuoteSeenStore {
    /// Open (creating if absent) the JSONL store and warm-load the cache.
    pub async fn open(path: &std::path::Path) -> Result<Self, QuoteStoreError> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut cache = std::collections::HashSet::new();
        if path.exists() {
            let content = tokio::fs::read_to_string(path).await?;
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let entry: SeenEntry = serde_json::from_str(line)?;
                cache.insert(format!("{}:{}", entry.persona, entry.tweet_id));
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            cache: Mutex::new(cache),
        })
    }
}

impl QuoteSeenStore for JsonlQuoteSeenStore {
    fn record<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), QuoteStoreError>> + Send + 'a>>
    {
        Box::pin(async move {
            let key = format!("{persona}:{tweet_id}");
            let mut cache = self.cache.lock().await;
            if cache.contains(&key) {
                return Ok(());
            }
            let entry = SeenEntry {
                persona: persona.to_string(),
                tweet_id: tweet_id.to_string(),
                seen_at: Utc::now(),
            };
            let line = format!("{}\n", serde_json::to_string(&entry)?);
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .await?
                .write_all_async(line.as_bytes())
                .await?;
            cache.insert(key);
            Ok(())
        })
    }

    fn was_seen<'a>(
        &'a self,
        persona: &'a str,
        tweet_id: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool, QuoteStoreError>> + Send + 'a>>
    {
        Box::pin(async move {
            let key = format!("{persona}:{tweet_id}");
            Ok(self.cache.lock().await.contains(&key))
        })
    }
}

// `tokio::fs::File` doesn't have a `write_all` async helper directly named
// `write_all_async` in stable tokio. Use the `tokio::io::AsyncWriteExt` extension
// trait via a thin wrapper above. (Implementer note: replace the
// `write_all_async` call with `use tokio::io::AsyncWriteExt;` then
// `file.write_all(line.as_bytes()).await?` — the prose above is to flag
// that the trait import is required.)

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_store_records_and_recalls() {
        let store = InMemoryQuoteSeenStore::new();
        assert!(!store.was_seen("p", "123").await.unwrap());
        store.record("p", "123").await.unwrap();
        assert!(store.was_seen("p", "123").await.unwrap());
        // Different persona = different key.
        assert!(!store.was_seen("other", "123").await.unwrap());
        // Different tweet = different key.
        assert!(!store.was_seen("p", "999").await.unwrap());
    }

    #[tokio::test]
    async fn jsonl_store_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seen.jsonl");
        {
            let store = JsonlQuoteSeenStore::open(&path).await.unwrap();
            store.record("p", "111").await.unwrap();
            store.record("p", "222").await.unwrap();
        }
        // Reopen and check the cache warm-loaded.
        let store = JsonlQuoteSeenStore::open(&path).await.unwrap();
        assert!(store.was_seen("p", "111").await.unwrap());
        assert!(store.was_seen("p", "222").await.unwrap());
        assert!(!store.was_seen("p", "333").await.unwrap());
    }

    #[tokio::test]
    async fn jsonl_record_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seen.jsonl");
        let store = JsonlQuoteSeenStore::open(&path).await.unwrap();
        store.record("p", "111").await.unwrap();
        store.record("p", "111").await.unwrap();
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(
            content.lines().count(),
            1,
            "duplicate record() calls must not write twice; got:\n{content}"
        );
    }
}
```

> Implementer note for the JSONL write: replace the `write_all_async` call with the standard `tokio::io::AsyncWriteExt::write_all` pattern. At the top of the file add `use tokio::io::AsyncWriteExt;` and change the offending line to:
> ```rust
> let mut file = tokio::fs::OpenOptions::new()
>     .create(true)
>     .append(true)
>     .open(&self.path)
>     .await?;
> file.write_all(line.as_bytes()).await?;
> ```

- [ ] **Step 3: Run tests**

Run: `cargo test --package heartbit-ghost --lib quote::sources`

Expected: 3 tests PASS.

- [ ] **Step 4: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings`

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/heartbit-ghost/src/quote/ crates/heartbit-ghost/src/lib.rs
git commit -m "feat(ghost): QuoteSource + QuoteSeenStore (in-memory + jsonl) + XUserTimelineSource"
```

---

## Task 5: Quote pipeline — `run_quote_pipeline`

**Files:**
- Create: `crates/heartbit-ghost/src/quote/prompts.rs`
- Modify: `crates/heartbit-ghost/src/quote/mod.rs` (add the pipeline)

This task is large. It mirrors `reply/mod.rs::run_reply_pipeline` almost exactly with three key differences: (a) the writer is `quote_writer` not `reply_writer`, (b) the publish tool is `twitter_quote` not `twitter_reply`, (c) the writer's user message includes the target language detected from the source tweet text.

- [ ] **Step 1: Create the prompt builders**

Create `crates/heartbit-ghost/src/quote/prompts.rs`:

```rust
//! User-message builders for each quote-pipeline stage. Pure string
//! composition — same shape as `reply/prompts.rs`.

use super::sources::QuoteCandidate;
use crate::reply::language::ReplyLanguage;

/// Build the mini-researcher's user message for a quote-tweet target.
pub(crate) fn build_quote_research_user_message(source: &QuoteCandidate) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "SOURCE TWEET (from @{}, posted {}):\n> {}\n\n",
        source.author_handle,
        source.posted_at.to_rfc3339(),
        source.text,
    ));
    out.push_str(
        "Identify the SPECIFIC claim or framing to engage with in 1-3 sentences. \
         Surface any quantitative claims (numbers, percentages, dates, citations) and \
         whether they are supported by reputable sources. Do NOT compose the quote-tweet — \
         the quote_writer composes it next.\n",
    );
    out
}

/// Build the quote_writer's user message: digest, source tweet, voice
/// guidelines, target language.
pub(crate) fn build_quote_writer_user_message(
    digest: &str,
    source: &QuoteCandidate,
    voice_guidelines: &str,
    language: &ReplyLanguage,
) -> String {
    let mut out = String::new();
    out.push_str("Research digest (claims to verify + framing to engage with):\n");
    out.push_str(digest);
    out.push_str("\n\n");
    out.push_str(&format!(
        "QUOTED TWEET (the post you are quoting; from @{}):\n> {}\n\n",
        source.author_handle, source.text,
    ));
    out.push_str(voice_guidelines);
    out.push('\n');
    out.push_str(&format!(
        "\nRESPOND IN {}. Mirror the quoted tweet's language exactly — do not switch to English just because the voice guidelines are English-described.\n",
        language.english_name
    ));
    out.push_str("\nCompose ONE quote-tweet comment (≤280 chars). Output the comment text only.\n");
    out
}

/// Build the style critic's user message for a quote-tweet candidate.
pub(crate) fn build_quote_critic_user_message(draft: &str, voice_guidelines: &str) -> String {
    format!(
        "Quote-tweet comment draft to evaluate:\n{draft}\n\n{voice_guidelines}\n\
         Score the draft and return your verdict as JSON per the schema.\n"
    )
}

/// Build the fact-check's user message for a quote-tweet draft.
pub(crate) fn build_quote_fact_user_message(draft: &str, digest: &str) -> String {
    format!(
        "Quote-tweet comment draft to verify:\n{draft}\n\nResearch digest (only source of truth):\n{digest}\n\
         Verify and return your verdict as JSON per the schema.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixture_source() -> QuoteCandidate {
        QuoteCandidate {
            id: "1".into(),
            text: "Microservices solve every problem".into(),
            author_id: "42".into(),
            author_handle: "shipit".into(),
            posted_at: Utc.with_ymd_and_hms(2026, 5, 13, 9, 0, 0).unwrap(),
        }
    }

    #[test]
    fn writer_message_injects_language_directive() {
        let french = ReplyLanguage {
            code: "fra".to_string(),
            english_name: "French".to_string(),
        };
        let s = build_quote_writer_user_message("digest", &fixture_source(), "VOICE", &french);
        assert!(s.contains("RESPOND IN French."));
        assert!(s.contains("QUOTED TWEET"));
        assert!(s.contains("@shipit"));
    }

    #[test]
    fn writer_message_includes_source_tweet_text() {
        let s = build_quote_writer_user_message(
            "digest",
            &fixture_source(),
            "VOICE",
            &ReplyLanguage::english(),
        );
        assert!(s.contains("Microservices solve every problem"));
    }

    #[test]
    fn research_message_quotes_source() {
        let s = build_quote_research_user_message(&fixture_source());
        assert!(s.contains("@shipit"));
        assert!(s.contains("Microservices solve every problem"));
        assert!(s.contains("SPECIFIC claim"));
    }
}
```

You'll need to import `Utc` in the test module — add `use chrono::Utc;` above the `fixture_source` definition.

- [ ] **Step 2: Run prompt tests**

Run: `cargo test --package heartbit-ghost --lib quote::prompts`

Expected: 3 tests PASS.

- [ ] **Step 3: Create the pipeline runtime**

Now extend `crates/heartbit-ghost/src/quote/mod.rs` (currently just module declarations) with the full pipeline. Read `crates/heartbit-ghost/src/reply/mod.rs` for the exact pattern, then mirror it. Specifically:

- Add `pub mod prompts;` declaration
- Add types: `QuoteConfig<'a>`, `QuoteOutput`, `QuoteCandidateRecord`, `QuoteOutcome`, `QuoteError`, `QuoteReviewDelivery` trait (same shape as `ReplyReviewDelivery`)
- Add `pub async fn run_quote_pipeline(cfg: QuoteConfig<'_>) -> Result<QuoteOutput, QuoteError>`:
  1. Validate `candidates_per_draft in 1..=5`
  2. Load StyleProfile snapshot (same as reply pipeline)
  3. Render voice guidelines via `pipeline::render_style_profile_as_english`
  4. Detect language of `source.text` via `reply::language::detect_reply_language`
  5. Build the researcher recipe + tools — quote pipeline uses the standard `researcher_recipe` + websearch/webfetch tools
  6. Run researcher to produce a digest
  7. Spawn N parallel writer/critic/fact_check chains via `tokio::JoinSet`, each calling:
     - `build_quote_writer_user_message(digest, source, voice, language)`
     - `length_normalize::normalize_tweet_length(writer_output, MAX_TWEET_CHARS)` (deterministic length normalizer — proven necessary in production)
     - Style critic on the normalized draft
     - Fact_check on the normalized draft
     - Short-circuit on writer output `"no_quote"` (return special sentinel)
  8. Pre-filter: drop drafts that fail `check_publish_gate` OR have `FactVerdict::Unverifiable` (same pattern as `review/mod.rs:431-470`)
  9. If all candidates pre-filtered out → return `QuoteOutcome::AllCandidatesGateRejected { reasons }`; if all "no_quote" → return `QuoteOutcome::NoQuote`
  10. Build `QuoteReviewMessage`, call `delivery.deliver_and_await`
  11. On `Pick(idx)`: run `check_publish_gate` post-pick (defensive), then call `twitter_quote.execute({ text, quote_tweet_id: source.id })`
  12. Record outcome and return

Because of the size, **the implementer is allowed to copy `crates/heartbit-ghost/src/reply/mod.rs` end-to-end as a starting template**, then:
- Replace every `Reply` identifier with `Quote`
- Replace `Mention` with `QuoteCandidate`
- Replace `parent: Option<TweetSnapshot>` with `source: QuoteCandidate`
- Replace `MentionerContext` with `Option<()>` (the quote pipeline doesn't enrich the source author — YAGNI for v1; can add later)
- Replace `reply_writer_recipe` with `quote_writer_recipe`
- Replace `twitter_reply` references with `twitter_quote`
- Replace `build_reply_*_user_message` with `build_quote_*_user_message`
- Set the publish-tool input to `{ "text": draft, "quote_tweet_id": source.id }` (NOT `in_reply_to`)
- Detect language of `cfg.source.text` (NOT `cfg.mention.text`)
- Tests must use mocks for `QuoteSource`, `QuoteReviewDelivery`, and the twitter tool

This is mechanical translation. The plan does NOT inline the full ~1500 lines because they're directly derivable from the existing pattern; it does require the implementer to read `reply/mod.rs` thoroughly before starting (acknowledged in the implementer prompt).

- [ ] **Step 4: Run all quote pipeline tests**

Run: `cargo test --package heartbit-ghost --lib quote::`

Expected: prompt tests (3) + pipeline tests (~5-7 happy/sad path tests) all PASS. If the implementer wrote a happy-path test analogous to `run_reply_pipeline_pick_index_0_posts_to_twitter`, that's the load-bearing one.

- [ ] **Step 5: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --package heartbit-ghost -- -D warnings && cargo test --package heartbit-ghost --lib`

Expected: clean across the board.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/quote/
git commit -m "feat(ghost): run_quote_pipeline — research → quote_writer → critic → fact → review → publish"
```

---

## Task 6: Daemon scheduler + handler + command + context

**Files:**
- Create: `crates/heartbit/src/daemon/persona_quote.rs` — `PersonaQuoteScheduler`
- Create: `crates/heartbit/src/daemon/persona_quote_handler.rs` — `handle_persona_quote` + `PersonaQuoteDeps`
- Create: `crates/heartbit/src/daemon/quotes_context.rs` — `QuotesContext` + `PersonaQuoteEntry`
- Modify: `crates/heartbit/src/daemon/types.rs` — add `DaemonCommand::PersonaQuote`
- Modify: `crates/heartbit/src/daemon/mod.rs` — declare modules + re-exports
- Modify: `crates/heartbit/src/daemon/core.rs` — spawn `PersonaQuoteScheduler` per enabled entry, dispatch `PersonaQuote` to handler

This task mirrors three existing patterns exactly:
- `persona_post.rs` → `PersonaQuoteScheduler` (replace "post" with "quote", same jitter/active-hours logic)
- `persona_post_handler.rs` → `persona_quote_handler.rs` (replace `PostsContext` with `QuotesContext`, replace topic generator with source poller — pick one un-seen + under-age tweet from the curated list)
- `posts_context.rs` → `quotes_context.rs` (replace post fields with quote fields)

- [ ] **Step 1: Add the command variant**

In `crates/heartbit/src/daemon/types.rs`, add a new variant to the `DaemonCommand` enum (find the existing `PersonaPost { persona: String }` and add alongside):

```rust
    /// Fire one quote-tweet handler invocation. Picks an un-quoted source
    /// tweet from the configured `persona_quotes.source_user_ids`, drafts
    /// via `run_quote_pipeline`, routes through Telegram review.
    PersonaQuote {
        /// Persona name (e.g. `"heartbit-ghost:x"`).
        persona: String,
    },
```

Also append a serde round-trip test inside the existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn persona_quote_command_round_trips() {
    let cmd = DaemonCommand::PersonaQuote {
        persona: "heartbit-ghost:x".into(),
    };
    let s = serde_json::to_string(&cmd).unwrap();
    let parsed: DaemonCommand = serde_json::from_str(&s).unwrap();
    match parsed {
        DaemonCommand::PersonaQuote { persona } => {
            assert_eq!(persona, "heartbit-ghost:x");
        }
        other => panic!("expected PersonaQuote, got {other:?}"),
    }
}
```

Run: `cargo test --package heartbit --features daemon --lib daemon::types::tests::persona_quote_command_round_trips`

Expected: PASS.

- [ ] **Step 2: Create `QuotesContext` + `PersonaQuoteEntry`**

Create `crates/heartbit/src/daemon/quotes_context.rs`:

```rust
//! Daemon-wide shared state for the quote-tweet pipeline. Constructed
//! once at startup by `heartbit-cli` from `[[daemon.persona_quotes]]`
//! and shared via `Arc` across handler invocations.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::config::daemon::ActiveHoursConfig;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::PersonaRegistry;
use heartbit_ghost::quote::sources::{QuoteSeenStore, QuoteSource};

/// One persona's quote-tweet runtime config.
pub struct PersonaQuoteEntry {
    pub source: Arc<dyn QuoteSource>,
    pub seen_store: Arc<dyn QuoteSeenStore>,
    pub interval: Duration,
    pub interval_jitter_pct: u32,
    pub active_hours: Option<ActiveHoursConfig>,
    pub source_user_ids: Vec<String>,
    pub candidates_per_draft: usize,
    pub max_age_hours: i64,
    pub max_candidates_per_tick: usize,
    /// Optional override LLM provider for quote_writer + critic.
    pub writer_provider: Option<Arc<BoxedProvider>>,
}

impl std::fmt::Debug for PersonaQuoteEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonaQuoteEntry")
            .field("interval", &self.interval)
            .field("interval_jitter_pct", &self.interval_jitter_pct)
            .field("source_user_ids", &self.source_user_ids)
            .field("candidates_per_draft", &self.candidates_per_draft)
            .field("max_age_hours", &self.max_age_hours)
            .field("max_candidates_per_tick", &self.max_candidates_per_tick)
            .field("writer_provider_set", &self.writer_provider.is_some())
            .finish()
    }
}

/// Daemon-wide context for the quote-tweet pipeline.
pub struct QuotesContext {
    pub registry: Arc<PersonaRegistry>,
    pub provider: Arc<BoxedProvider>,
    pub delivery: Arc<dyn heartbit_ghost::quote::QuoteReviewDelivery>,
    pub twitter_quote_tool: Arc<dyn Tool>,
    pub credentials: Arc<dyn CredentialResolver>,
    pub corpora_root: PathBuf,
    pub profiles_root: PathBuf,
    pub entries: std::collections::HashMap<String, PersonaQuoteEntry>,
}
```

(Note: `heartbit_ghost::quote::QuoteReviewDelivery` is created in Task 5 as part of the pipeline. If the implementer is reading this task before Task 5 is done, the trait will not exist yet — block on Task 5 first.)

- [ ] **Step 3: Create the scheduler**

Create `crates/heartbit/src/daemon/persona_quote.rs`. Read `crates/heartbit/src/daemon/persona_post.rs` end-to-end as the template, then:

- Rename `PersonaPostScheduler` → `PersonaQuoteScheduler`
- Replace `DaemonCommand::PersonaPost { persona }` with `DaemonCommand::PersonaQuote { persona }`
- Keep all jitter / active-hours / interval logic identical
- Constructor takes a `&PersonaQuotesConfig` and pulls the same fields (`persona`, `post_interval_seconds` → `poll_interval_seconds`, `interval_jitter_pct`, `active_hours`)

Add a test analogous to `fires_persona_post_after_interval` but for `PersonaQuote`.

- [ ] **Step 4: Create the handler**

Create `crates/heartbit/src/daemon/persona_quote_handler.rs`:

```rust
//! Handler for `DaemonCommand::PersonaQuote` — picks an un-seen source
//! tweet from one of the configured `source_user_ids`, drafts a
//! quote-tweet via `run_quote_pipeline`, records the outcome.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};
use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::persona::{PersonaParams, PersonaRegistry};
use heartbit_ghost::quote::sources::{QuoteCandidate, QuoteSeenStore, QuoteSource};
use heartbit_ghost::quote::{
    QuoteConfig, QuoteOutcome, QuoteReviewDelivery, run_quote_pipeline,
};

pub struct PersonaQuoteDeps<'a> {
    pub persona_name: &'a str,
    pub registry: &'a PersonaRegistry,
    pub source: &'a dyn QuoteSource,
    pub seen_store: &'a dyn QuoteSeenStore,
    pub source_user_ids: &'a [String],
    pub max_age_hours: i64,
    pub max_candidates_per_tick: usize,
    pub provider: Arc<BoxedProvider>,
    pub writer_provider: Option<Arc<BoxedProvider>>,
    pub delivery: Arc<dyn QuoteReviewDelivery>,
    pub twitter_quote_tool: Arc<dyn Tool>,
    pub credentials: Arc<dyn CredentialResolver>,
    pub candidates_per_draft: usize,
    pub corpora_root: &'a Path,
    pub profiles_root: &'a Path,
}

/// Run one `PersonaQuote` handler invocation.
///
/// Steps:
/// 1. For each source_user_id, fetch recent tweets via `source.recent`.
/// 2. Filter out already-seen + over-age tweets.
/// 3. Pick the first `max_candidates_per_tick` candidates (by source order;
///    consider scoring later).
/// 4. For each, run `run_quote_pipeline` and record `seen_store.record`
///    BEFORE the pipeline so we don't double-draft if it fails part-way.
/// 5. Return the outcome of the first successful pipeline run (or
///    aggregate when `max_candidates_per_tick > 1`).
pub async fn handle_persona_quote(deps: PersonaQuoteDeps<'_>) -> Result<()> {
    let persona = deps
        .registry
        .get(deps.persona_name)
        .ok_or_else(|| anyhow::anyhow!("persona '{}' not registered", deps.persona_name))?;
    let _expansion = persona
        .expand(&PersonaParams::default())
        .map_err(|e| anyhow::anyhow!("expand persona '{}': {e}", deps.persona_name))?;

    let now = Utc::now();
    let max_age = ChronoDuration::hours(deps.max_age_hours);

    // 1+2: collect candidates across all sources, filter age + seen.
    let mut candidates: Vec<QuoteCandidate> = Vec::new();
    for user_id in deps.source_user_ids {
        let fetched = match deps.source.recent(user_id).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    user_id = %user_id,
                    error = %e,
                    "failed to fetch source timeline; skipping"
                );
                continue;
            }
        };
        for c in fetched {
            if deps.max_age_hours > 0 && (now - c.posted_at) > max_age {
                continue;
            }
            match deps.seen_store.was_seen(deps.persona_name, &c.id).await {
                Ok(true) => continue,
                Ok(false) => {}
                Err(e) => {
                    tracing::warn!(
                        tweet_id = %c.id,
                        error = %e,
                        "seen_store.was_seen failed; treating as un-seen"
                    );
                }
            }
            candidates.push(c);
        }
    }

    if candidates.is_empty() {
        tracing::info!(persona = %deps.persona_name, "no quote candidates available this tick");
        return Ok(());
    }

    // 3+4: pick first N, draft each.
    let to_process = candidates.into_iter().take(deps.max_candidates_per_tick);
    for source in to_process {
        // Record BEFORE running so a panic/restart doesn't re-draft.
        if let Err(e) = deps.seen_store.record(deps.persona_name, &source.id).await {
            tracing::warn!(error = %e, "seen_store.record failed before pipeline");
        }
        let cfg = QuoteConfig {
            persona_name: deps.persona_name,
            provider: deps.provider.clone(),
            corpora_root: deps.corpora_root,
            profiles_root: deps.profiles_root,
            on_progress: Some(Arc::new(|s: &str| tracing::info!("quote: {s}"))),
            source: source.clone(),
            candidates_per_draft: deps.candidates_per_draft,
            delivery: deps.delivery.clone(),
            twitter_tool: deps.twitter_quote_tool.clone(),
            credentials: deps.credentials.clone(),
            writer_provider: deps.writer_provider.clone(),
        };
        match run_quote_pipeline(cfg).await {
            Ok(out) => {
                tracing::info!(
                    persona = %deps.persona_name,
                    source_id = %source.id,
                    outcome = ?out.outcome,
                    "quote pipeline complete"
                );
                // Stop on first Posted; continue if any other outcome
                // so we try the next candidate this tick.
                if matches!(out.outcome, QuoteOutcome::Posted { .. }) {
                    break;
                }
            }
            Err(e) => {
                tracing::error!(
                    persona = %deps.persona_name,
                    source_id = %source.id,
                    error = %e,
                    "quote pipeline failed"
                );
            }
        }
    }
    Ok(())
}
```

- [ ] **Step 5: Wire into `core.rs`**

In `crates/heartbit/src/daemon/core.rs`:

1. Add a `quotes_context: Option<Arc<QuotesContext>>` field on `DaemonCore` (mirror the existing `posts_context` field).
2. Add a `with_quotes_context` builder method.
3. In the per-tick spawn block (find the `// --- Spawn PersonaPostScheduler instances per configured persona ---` comment), add an analogous block for quote schedulers — iterate `ctx.entries`, build a `PersonaQuoteScheduler` per persona, spawn.
4. In the command-dispatch match (find the `DaemonCommand::PersonaPost { persona }` arm), add an arm for `DaemonCommand::PersonaQuote { persona }` that builds `PersonaQuoteDeps` from the matched entry and calls `handle_persona_quote`.

- [ ] **Step 6: Module declarations + re-exports**

In `crates/heartbit/src/daemon/mod.rs`, find the existing `pub mod persona_post;` etc. and add:

```rust
pub mod persona_quote;
pub mod persona_quote_handler;
pub mod quotes_context;

pub use persona_quote::PersonaQuoteScheduler;
pub use persona_quote_handler::{PersonaQuoteDeps, handle_persona_quote};
pub use quotes_context::{PersonaQuoteEntry, QuotesContext};
```

- [ ] **Step 7: Run tests**

Run: `cargo test --package heartbit --features daemon --lib daemon::persona_quote && cargo test --package heartbit --features daemon --lib daemon::persona_quote_handler`

Expected: PASS.

- [ ] **Step 8: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`

Expected: clean across the board.

- [ ] **Step 9: Commit**

```bash
git add crates/heartbit/src/daemon/persona_quote.rs \
        crates/heartbit/src/daemon/persona_quote_handler.rs \
        crates/heartbit/src/daemon/quotes_context.rs \
        crates/heartbit/src/daemon/types.rs \
        crates/heartbit/src/daemon/mod.rs \
        crates/heartbit/src/daemon/core.rs
git commit -m "feat(daemon): PersonaQuoteScheduler + handle_persona_quote + DaemonCommand::PersonaQuote"
```

---

## Task 7: CLI wiring + docs + example

**Files:**
- Modify: `crates/heartbit-cli/src/daemon/mod.rs` — build `QuotesContext` from `[[daemon.persona_quotes]]`
- Modify: `crates/heartbit-cli/src/daemon/validate.rs` — extend `validate_daemon_config` to cross-check quotes
- Modify: `docs/operating-heartbit.md` — document the new knobs
- Modify: `daemon-dev.toml` — commented-out example block (operator-local; not committed unless you're me)

- [ ] **Step 1: Build QuotesContext at daemon startup**

In `crates/heartbit-cli/src/daemon/mod.rs`, find the existing `// --- Build PostsContext from daemon_config.persona_posts ---` block. Add a parallel `// --- Build QuotesContext from daemon_config.persona_quotes ---` block after it.

The block must:
1. Iterate `daemon_config.persona_quotes` (only `enabled` entries)
2. Validate `poll_interval_seconds >= 60` (anyhow::bail otherwise)
3. For each entry:
   - Build `seen_store: Arc<dyn QuoteSeenStore>` from `seen_store` + `seen_store_path` (in_memory or jsonl)
   - Build `source: Arc<dyn QuoteSource>` as `XUserTimelineSource::new(xclient.clone())`
   - Build `writer_provider: Option<Arc<BoxedProvider>>` via `crate::build_agent_provider` (same pattern as `persona_posts.writer_provider`)
4. Build the shared `QuoteReviewDelivery` (likely a Telegram impl alongside `TelegramReviewDelivery` — for v1 the implementer can reuse the same struct if shapes match, or stub a minimal `QuoteTelegramDelivery`)
5. Build `QuotesContext` with all entries and wire into `DaemonCore` via `with_quotes_context`

Reuse the existing `xclient` (X v2 client) and `credentials` already constructed for the posts context.

- [ ] **Step 2: Extend `--validate-config`**

In `crates/heartbit-cli/src/daemon/validate.rs`, add validation for `persona_quotes`:

```rust
fn validate_persona_quotes(
    daemon: &DaemonConfig,
    path_exists: &impl Fn(&Path) -> bool,
    issues: &mut Vec<ValidationIssue>,
) {
    for cfg in &daemon.persona_quotes {
        if !cfg.enabled {
            continue;
        }
        let context = format!("[[daemon.persona_quotes]] persona='{}'", cfg.persona);

        if cfg.source_user_ids.is_empty() {
            issues.push(ValidationIssue {
                kind: ValidationIssueKind::MissingPostHistoryPath, // TODO: add a dedicated variant
                context: format!("{context}: source_user_ids is empty"),
            });
        }

        if cfg.seen_store == "jsonl" {
            if let Some(p) = cfg.seen_store_path.as_deref() {
                check_parent_dir(p, &context, path_exists, issues);
            }
        }
    }
}
```

> Implementer note: extend `ValidationIssueKind` with a new variant `MissingSourceUserIds` instead of reusing `MissingPostHistoryPath`. The TODO above is a placeholder for that.

Then call `validate_persona_quotes(daemon_config, &path_exists, &mut issues);` from `validate_daemon_config`.

Add unit tests mirroring the existing `persona_posts_without_operator_user_id_is_flagged` shape: one for empty `source_user_ids`, one for jsonl with missing parent dir.

- [ ] **Step 3: Update operating docs**

In `docs/operating-heartbit.md`, add a new section `## Quote-tweet knobs` between `## Mention polling knobs` and `## Kill switches`:

```markdown
## Quote-tweet knobs

`[[daemon.persona_quotes]]` controls the proactive quote-tweet loop. The daemon polls each `source_user_ids` entry and quote-tweets the most engaging un-quoted tweet on a jittered cadence.

| Knob | Default | When to change |
|---|---|---|
| `enabled` | `true` | Set to `false` to pause this persona's quote loop. |
| `poll_interval_seconds` | `5400` (90 min) | Lower for higher quote volume; minimum is `60`. |
| `interval_jitter_pct` | `25` (±25%) | Same anti-bot rationale as proactive posts. |
| `active_hours` | unset (24/7) | E.g. `"08:00-22:00"` to restrict to waking hours. |
| `source_user_ids` | required | List of X user IDs (numeric strings) to poll. Curated voices you want to engage with. |
| `candidates_per_draft` | `3` | Higher = more LLM cost per tick but better picks. |
| `seen_store` | `"in_memory"` | Use `"jsonl"` for restart durability — recommended for production. |
| `seen_store_path` | required for jsonl | Tilde-expanded; ensure parent dir exists. |
| `max_age_hours` | `12` | Tweets older than this are skipped; the discourse has moved on. |
| `max_candidates_per_tick` | `1` | How many quote-drafts to attempt per scheduler tick. |
| `writer_provider` | unset | Same shape as `persona_posts.writer_provider`. Falls back to global `[provider]`. |

**Voice note**: the quote_writer uses a distinct disposition (opinionated but charitable — caritas in veritate). Proactive posts and replies keep their existing voice. To audit the disposition see `crates/heartbit-ghost/src/agents/quote_writer.rs::QUOTE_WRITER_SYSTEM_PROMPT`.
```

- [ ] **Step 4: Add example to daemon-dev.toml**

In `daemon-dev.toml` (gitignored, operator-local), add a commented-out example block after the existing `[[daemon.persona_posts]]` block. Each operator activates by uncommenting.

```toml
# Quote-tweet curated voices. Daemon polls these accounts every ~90 min
# (with jitter), drafts opinionated-but-charitable quote-tweets, routes
# through Telegram for review.
# Uncomment to enable. Replace the user IDs with the X accounts you want
# to engage with.
#
# [[daemon.persona_quotes]]
# persona = "heartbit-ghost:x"
# enabled = true
# poll_interval_seconds = 5400
# source_user_ids = ["44196397", "16884623"]   # example: musk, dhh
# seen_store = "jsonl"
# seen_store_path = ".heartbit/quotes/heartbit-ghost-x.seen.jsonl"
# max_age_hours = 12
# max_candidates_per_tick = 1
#
# [daemon.persona_quotes.active_hours]
# start = "09:00"
# end = "22:00"
```

- [ ] **Step 5: Run the validator on the live config**

```bash
HEARTBIT_GHOST_OPERATOR_USER_ID=999 target/release/heartbit --config daemon-dev.toml daemon --validate-config
```

Expected: `✓ daemon-dev.toml validates clean` (the quotes block is commented out so the validator doesn't see it).

Then uncomment the example block in daemon-dev.toml and re-run; the validator should still pass because the example uses safe defaults.

- [ ] **Step 6: Quality gate**

Run: `cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`

Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-cli/src/daemon/mod.rs \
        crates/heartbit-cli/src/daemon/validate.rs \
        docs/operating-heartbit.md
git commit -m "feat(daemon): CLI wiring + --validate-config support + docs for persona_quotes"
```

---

## Task 8: Final integration smoke + plan close-out

- [ ] **Step 1: Full workspace gate**

```bash
cargo fmt -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
```

Expected: all green.

- [ ] **Step 2: Manual end-to-end smoke**

In a separate terminal (not via me — see CLAUDE.md):
1. Stop the running daemon
2. `cargo build --release --bin heartbit`
3. Uncomment the `[[daemon.persona_quotes]]` block in your `daemon-dev.toml`, set `source_user_ids` to 2-3 real X user IDs you want to engage with
4. Restart the daemon
5. Wait up to `poll_interval_seconds` for the first scheduler tick, or fire `{"type":"persona_quote","persona":"heartbit-ghost:x"}` into the Kafka commands topic via `docker exec ... kafka-console-producer.sh`
6. Watch logs for `quote: Researching topic...`, candidate generation, pre-filter, Telegram review delivery
7. Pick a candidate via Telegram, verify the quote-tweet posts to X with the right `quote_tweet_id` URL

- [ ] **Step 3: Update `tasks/lessons.md` if applicable**

If any non-obvious gotcha surfaced (e.g. Grok handling of strict-schema critic, an X API edge case, a config-validation surprise), record one or two lines.

---

## Verification matrix

| Spec item | Covered by |
|---|---|
| `[[daemon.persona_quotes]]` config block | Task 1 |
| Curated account list as source (vs search) | Task 1 (source_user_ids), Task 4 (XUserTimelineSource) |
| Medium cadence (1-2h default) | Task 1 (5400 sec = 90 min default) |
| New scheduler / command / handler | Task 6 |
| New X tool: `twitter_quote` (POST /2/tweets w/ quote_tweet_id) | Task 2 |
| New writer recipe with Catholic-charity disposition | Task 3 |
| Disposition: opinionated, good faith, no aggression | Task 3 prompt + 4 regression tests |
| Disposition scoped ONLY to quote_writer (not writer or reply_writer) | Task 3 ("Files modified" section asserts this) |
| Telegram review (same pattern as posts/replies) | Task 5 (delivery layer in pipeline) |
| Language detection on source tweet | Task 5 (build_quote_writer_user_message uses ReplyLanguage) |
| Already-quoted dedup | Task 4 (QuoteSeenStore + in-memory + jsonl) |
| 280-char length enforcement | Task 2 (tool validation), Task 5 (length_normalize in pipeline) |
| Strict-sourcing chain (no invented numbers) | Task 3 prompt + Task 5 pipeline (fact_check pre-filter) |
| Operator config validation | Task 7 (--validate-config extension) |
| Operating docs | Task 7 |

---

## Notes for the implementer

- The order of tasks matters: Task 5 (pipeline) depends on Task 3 (recipe) + Task 4 (sources). Task 6 (daemon) depends on Task 5. Task 7 (CLI) depends on Task 6.
- The Catholic-charity disposition is THE distinguishing feature. The 4 regression tests in Task 3 are load-bearing — do not soften the prompt without updating them.
- Task 5 explicitly authorizes copying `reply/mod.rs` as a starting template because the structural mirror is tight. Read both files end-to-end before starting, then mechanical-translate.
- The `length_normalize` module (shipped in commit d4816ed) lives in `crates/heartbit-ghost/src/pipeline/length_normalize.rs` and is re-exported as `pipeline::normalize_tweet_length` + `pipeline::MAX_TWEET_CHARS`. Use it.
- The `reply::language::detect_reply_language` function (shipped in commit 746006a) is the canonical language detector. Reuse it; do not write a new one.
- `parse_thread_tweets` (in `review::tweet_split`) is reused by the publish_gate; if a quote-tweet is single-tweet only (the v1 spec — no quote-tweet threads), the publish_gate's existing logic works without changes.
- Don't bundle "while I'm here" refactors. Each task = one logical commit.
