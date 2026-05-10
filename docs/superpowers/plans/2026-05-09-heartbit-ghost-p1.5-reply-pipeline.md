# heartbit-ghost P1.5 — Reply pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-09-heartbit-ghost-p1.5-reply-pipeline-design.md`

**Goal:** Wire the existing `TwitterMentionsTool` and `TwitterReplyTool` into a complete loop: cron-driven mention polling → per-mention reply drafting via the persona's voice → Telegram review → posted reply via `twitter_reply`. Calibration mode only (every reply gates on Telegram); auto-reply is P1.4.

**Architecture:** New `crates/heartbit-ghost/src/reply/` module mirrors `review/`. New `reply_writer` sub-agent recipe in `agents/`. Two new `DaemonCommand` variants (`MentionPoll`, `ReplyDraft`) feed the existing dispatcher. `MentionStore` (JSONL on disk for V1) tracks `since_id` and the `replied_to` set; `SpamGuard` filters mentions before drafting.

**Tech Stack:** Rust 2024 edition, tokio, serde, toml, async_trait, chrono. Existing crates: `heartbit-core` (Tool trait, agent runner), `heartbit-ghost` (twitter tools, voice modeling), `heartbit-cli` (entrypoint), `heartbit` (daemon). Tests use the existing `MockProvider::route_with_recorder` from `pipeline/mod.rs`.

**Branch:** `feat/heartbit-ghost-p1.5-reply-pipeline` (already created off `main`; the spec lives there).

**Sub-phases (per spec §13):**
- **P1.5a** — Pipeline + writer recipe + CLI one-off testing (Tasks 1-5)
- **P1.5b** — Storage + spam guards (Tasks 6-8)
- **P1.5c** — Daemon polling + Telegram delivery (Tasks 9-12)

---

## Task 1: `reply_writer` sub-agent recipe

**Files:**
- Create: `crates/heartbit-ghost/src/agents/reply_writer.rs`
- Modify: `crates/heartbit-ghost/src/agents/mod.rs` (re-export)

- [ ] **Step 1: Write the failing tests**

Create `crates/heartbit-ghost/src/agents/reply_writer.rs`:

```rust
//! Reply writer sub-agent — composes a single ≤280-char reply addressing
//! a specific mention. See spec §4 for the rationale.

use heartbit_core::config::AgentConfig;

/// System prompt for the reply writer. Tone-laddered (substantive →
/// honest acknowledgement → gracious decline → "no_reply"); explicitly
/// bans generic openers ("Thanks for…", "Great point…"); hard 280-char
/// cap.
pub const REPLY_WRITER_SYSTEM_PROMPT: &str = r#"You write a single short reply (≤280 characters) to a specific tweet. The reply must address the content of that tweet directly — never a generic acknowledgement, never a content-free thanks, never a question that the tweet's author obviously already considered.

INPUT (from the user message)
- The PARENT tweet your reply addresses (the mention).
- Optional: the ORIGINAL tweet the parent was replying to (your own tweet, when applicable).
- The mentioner's bio + 2-3 of their recent tweets, for tone calibration.
- Voice guidelines for the persona.
- (Optional) Persona mode addendum.

OUTPUT
The reply text, plain. No preamble, no quotation marks around it, no markdown. ≤280 characters HARD CAP — count includes spaces and emoji. Aim for 80-180 characters; brevity reads as confidence.

CONSTRAINTS
- Address the SPECIFIC content of the mention. If they made a claim, engage with the claim. If they asked a question, answer it (or honestly say you don't know). If they made a joke, match the register.
- Voice MUST match the persona's guidelines exactly (no em-dashes if forbidden, formatting rules, AI-tells to avoid).
- Never start with "Thanks for…" or "Great point…" or any generic opener — these are AI tells.
- Never use exclamation marks unless the persona's voice explicitly allows them.
- Do NOT @-mention anyone. The X API handles the threading; @-mentions in the body are noise.
- If the mention is hostile, dismissive, or low-effort, prefer a single-line factual reply over engagement. If it's clearly bait, output the literal string "no_reply" and stop.
- If you cannot ground a substantive response in either the mention's content or your own knowledge, output "no_reply" and stop.

TONE LADDER (in order of preference)
1. Substantive engagement (you have something specific to add)
2. Honest acknowledgement (you agree / disagree, with one sentence of reason)
3. Gracious decline ("don't have data on that" / "haven't tried it")
4. "no_reply" (the mention doesn't warrant a response)
"#;

/// Construct the reply writer [`AgentConfig`].
pub fn reply_writer_recipe() -> AgentConfig {
    AgentConfig {
        name: "reply_writer".to_string(),
        description: "Compose a single ≤280-char reply addressing a specific tweet.".to_string(),
        system_prompt: REPLY_WRITER_SYSTEM_PROMPT.to_string(),
        max_turns: Some(1),
        max_tokens: Some(512),
        reasoning_effort: Some("low".to_string()),
        ..super::stub_recipe("reply_writer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reply_writer_recipe_has_expected_shape() {
        let cfg = reply_writer_recipe();
        assert_eq!(cfg.name, "reply_writer");
        assert!(!cfg.description.is_empty());
        assert!(!cfg.system_prompt.is_empty());
        assert_eq!(cfg.max_turns, Some(1));
        assert_eq!(cfg.max_tokens, Some(512));
        assert_eq!(cfg.reasoning_effort.as_deref(), Some("low"));
        assert!(cfg.response_schema.is_none(), "free-form text, no schema");
    }

    #[test]
    fn reply_writer_prompt_mandates_length_cap_and_no_thread() {
        let p = REPLY_WRITER_SYSTEM_PROMPT;
        assert!(p.contains("280 characters"), "prompt must state the 280 cap");
        assert!(p.contains("HARD CAP"), "prompt must call the cap HARD");
    }

    #[test]
    fn reply_writer_prompt_bans_generic_openers_and_offers_no_reply_escape() {
        let p = REPLY_WRITER_SYSTEM_PROMPT;
        assert!(
            p.contains("Thanks for") && p.contains("Great point"),
            "prompt must explicitly ban these AI-tell openers"
        );
        assert!(
            p.contains("no_reply"),
            "prompt must offer the no_reply escape hatch"
        );
        assert!(
            p.contains("TONE LADDER"),
            "prompt must structure preferences as a tone ladder"
        );
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p heartbit-ghost --lib agents::reply_writer
```

Expected: compile error (module doesn't exist yet).

- [ ] **Step 3: Wire into `agents/mod.rs`**

In `crates/heartbit-ghost/src/agents/mod.rs`, after the existing `pub mod` declarations alphabetically, add:

```rust
pub mod reply_writer;
```

And after the existing `pub use` re-exports, add:

```rust
pub use reply_writer::reply_writer_recipe;
```

- [ ] **Step 4: Run the tests**

```bash
cargo test -p heartbit-ghost --lib agents::reply_writer
```

Expected: 3 PASS.

- [ ] **Step 5: Format + clippy**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/heartbit-ghost/src/agents/reply_writer.rs crates/heartbit-ghost/src/agents/mod.rs
git commit -m "feat(ghost): reply_writer recipe — short, mention-anchored, persona-voiced reply

System prompt locks: ≤280-char hard cap, addresses specific mention
content, bans generic AI-tell openers (Thanks for / Great point),
includes a no_reply escape hatch for bait/hostile mentions, structures
preferences as a 4-level tone ladder. max_turns=1, max_tokens=512,
reasoning_effort=low — replies are not the place for elaborate planning.

heartbit-ghost P1.5a — task 1/12."
```

---

## Task 2: Reply value types (`Mention`, `TweetSnapshot`, `MentionerContext`)

**Files:**
- Create: `crates/heartbit-ghost/src/reply/mod.rs` (skeleton — types only; pipeline lands in Task 4)
- Modify: `crates/heartbit-ghost/src/lib.rs` (`pub mod reply;`)

- [ ] **Step 1: Write the failing tests**

Create `crates/heartbit-ghost/src/reply/mod.rs`:

```rust
//! Reply pipeline — drafts a single short reply to a specific mention,
//! routes to Telegram for review, posts via `twitter_reply` on user pick.
//!
//! See spec §2/§5 for the architecture; this file holds the value types
//! and the public surface. The runtime lives in [`run_reply_pipeline`]
//! once Task 5 lands it.

use chrono::{DateTime, Utc};

/// A mention of the operator's account fetched from `twitter_mentions`.
#[derive(Debug, Clone)]
pub struct Mention {
    /// X tweet ID of the mention itself.
    pub id: String,
    /// Plain text of the mention.
    pub text: String,
    /// X user ID of the mentioner.
    pub author_id: String,
    /// Public handle of the mentioner (sans `@`).
    pub author_handle: String,
    /// When the mention was posted.
    pub posted_at: DateTime<Utc>,
    /// Tweet ID this mention is replying to (None when it's a top-level
    /// `@operator …` mention rather than a reply on an operator's tweet).
    pub in_reply_to_tweet_id: Option<String>,
}

/// A small snapshot of a tweet (text + timing). Used as a parent-tweet
/// context for the reply researcher.
#[derive(Debug, Clone)]
pub struct TweetSnapshot {
    pub id: String,
    pub text: String,
    pub posted_at: DateTime<Utc>,
}

/// Tone-calibration context about the mentioner. None of these are
/// strictly required; the writer degrades gracefully if missing.
#[derive(Debug, Clone, Default)]
pub struct MentionerContext {
    pub handle: String,
    pub bio: Option<String>,
    /// Up to 3 recent tweets, abridged.
    pub recent_tweets: Vec<TweetSnapshot>,
    pub follower_count: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mention_round_trips_through_clone() {
        let m = Mention {
            id: "1".into(),
            text: "hi".into(),
            author_id: "12".into(),
            author_handle: "alice".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: Some("99".into()),
        };
        let copy = m.clone();
        assert_eq!(copy.id, m.id);
        assert_eq!(copy.text, m.text);
        assert_eq!(copy.in_reply_to_tweet_id, m.in_reply_to_tweet_id);
    }

    #[test]
    fn mentioner_context_default_has_empty_handle_and_no_recent_tweets() {
        let m = MentionerContext::default();
        assert!(m.handle.is_empty());
        assert!(m.bio.is_none());
        assert!(m.recent_tweets.is_empty());
        assert!(m.follower_count.is_none());
    }
}
```

- [ ] **Step 2: Wire into `lib.rs`**

In `crates/heartbit-ghost/src/lib.rs`, alongside `pub mod review;` (alphabetical placement), add:

```rust
pub mod reply;
```

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib reply::tests
```

Expected: 2 PASS.

- [ ] **Step 4: Verify chrono is already a dep**

```bash
grep -E "^chrono\s*=" crates/heartbit-ghost/Cargo.toml Cargo.toml
```

If not present in heartbit-ghost's [dependencies], add:

```toml
chrono = { workspace = true, features = ["serde"] }
```

(Confirm `chrono` is in `[workspace.dependencies]` first.)

- [ ] **Step 5: Format + commit**

```bash
cargo fmt --all
git add crates/heartbit-ghost/src/reply/mod.rs crates/heartbit-ghost/src/lib.rs crates/heartbit-ghost/Cargo.toml
git commit -m "feat(ghost): reply value types (Mention, TweetSnapshot, MentionerContext)

Skeleton for the reply pipeline — pure value types in reply/mod.rs.
Runtime (run_reply_pipeline) lands in task 5.

heartbit-ghost P1.5a — task 2/12."
```

---

## Task 3: Reply prompt builders

**Files:**
- Create: `crates/heartbit-ghost/src/reply/prompts.rs`
- Modify: `crates/heartbit-ghost/src/reply/mod.rs` (re-export)

- [ ] **Step 1: Write the failing tests**

Create `crates/heartbit-ghost/src/reply/prompts.rs`:

```rust
//! User-message builders for each reply-pipeline stage. Pure string
//! composition — same shape as `pipeline/prompts.rs`.

use super::{Mention, MentionerContext, TweetSnapshot};

/// Build the mini-researcher's user message: the parent tweet (if any),
/// the mention itself, and abridged context about the mentioner.
pub(crate) fn build_reply_research_user_message(
    mention: &Mention,
    parent: Option<&TweetSnapshot>,
    mentioner: Option<&MentionerContext>,
) -> String {
    let mut out = String::new();
    if let Some(p) = parent {
        out.push_str("PARENT TWEET (yours, posted ");
        out.push_str(&p.posted_at.to_rfc3339());
        out.push_str("):\n> ");
        out.push_str(&p.text);
        out.push_str("\n\n");
    }
    out.push_str(&format!(
        "THEIR REPLY (from @{}, posted {}):\n> {}\n\n",
        mention.author_handle,
        mention.posted_at.to_rfc3339(),
        mention.text,
    ));
    if let Some(m) = mentioner {
        out.push_str("MENTIONER CONTEXT\n");
        if let Some(bio) = &m.bio {
            out.push_str(&format!("- bio: {bio}\n"));
        }
        if let Some(fc) = m.follower_count {
            out.push_str(&format!("- followers: {fc}\n"));
        }
        if !m.recent_tweets.is_empty() {
            out.push_str("- recent tweets:\n");
            for t in m.recent_tweets.iter().take(3) {
                let abridged: String = t.text.chars().take(100).collect();
                out.push_str(&format!("    > {abridged}\n"));
            }
        }
        out.push('\n');
    }
    out.push_str(
        "Identify the SPECIFIC point to engage with in 1-3 sentences. \
         Do NOT compose the reply — the writer composes it next.\n",
    );
    out
}

/// Build the writer's user message — the digest from the researcher,
/// then voice guidelines, optional mode_addendum, and a clear final
/// instruction.
pub(crate) fn build_reply_writer_user_message(
    digest: &str,
    voice_guidelines: &str,
    mode_addendum: Option<&str>,
) -> String {
    let mut out = String::new();
    out.push_str("Research digest (the specific point to engage with):\n");
    out.push_str(digest);
    out.push_str("\n\n");
    out.push_str(voice_guidelines);
    out.push('\n');
    if let Some(addendum) = mode_addendum {
        out.push('\n');
        out.push_str(addendum);
        out.push('\n');
    }
    out.push_str(
        "\nCompose ONE reply (≤280 chars). Output the reply text only.\n",
    );
    out
}

/// Build the style critic's user message for a reply candidate.
pub(crate) fn build_reply_critic_user_message(draft: &str, voice_guidelines: &str) -> String {
    format!(
        "Reply draft to evaluate:\n{draft}\n\n{voice_guidelines}\n\
         Score the draft and return your verdict as JSON per the schema.\n"
    )
}

/// Build the fact-check's user message for a reply.
pub(crate) fn build_reply_fact_user_message(draft: &str, digest: &str) -> String {
    format!(
        "Reply draft to verify:\n{draft}\n\nResearch digest (only source of truth):\n{digest}\n\
         Verify and return your verdict as JSON per the schema.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixture_mention() -> Mention {
        Mention {
            id: "abc".into(),
            text: "how does this compare to rig-rs?".into(),
            author_id: "777".into(),
            author_handle: "grumpy_dev".into(),
            posted_at: Utc.with_ymd_and_hms(2026, 5, 8, 11, 2, 0).unwrap(),
            in_reply_to_tweet_id: Some("parent_id".into()),
        }
    }

    #[test]
    fn research_message_quotes_mention_and_parent() {
        let m = fixture_mention();
        let p = TweetSnapshot {
            id: "parent_id".into(),
            text: "Implement two methods, get a fully wired tool.".into(),
            posted_at: Utc.with_ymd_and_hms(2026, 5, 8, 10, 14, 0).unwrap(),
        };
        let s = build_reply_research_user_message(&m, Some(&p), None);
        assert!(s.contains("Implement two methods"));
        assert!(s.contains("how does this compare to rig-rs"));
        assert!(s.contains("@grumpy_dev"));
        assert!(s.contains("Identify the SPECIFIC point"));
    }

    #[test]
    fn writer_message_appends_addendum_after_voice_guidelines() {
        let s = build_reply_writer_user_message(
            "engage with: rig-rs comparison",
            "VOICE GUIDELINES",
            Some("EVANGELISM MODE — fixture"),
        );
        let voice_pos = s.find("VOICE GUIDELINES").expect("voice present");
        let add_pos = s.find("EVANGELISM MODE — fixture").expect("addendum present");
        assert!(voice_pos < add_pos, "addendum must follow voice");
        assert!(s.contains("≤280 chars"));
    }

    #[test]
    fn writer_message_omits_addendum_block_when_none() {
        let s = build_reply_writer_user_message("digest", "VOICE", None);
        assert!(!s.contains("EVANGELISM"));
        assert!(s.contains("Compose ONE reply"));
    }
}
```

Use chrono::Utc — re-import at the top of the test module:

```rust
use chrono::Utc;
```

Add at the top of the tests module to make the `Utc.with_ymd_and_hms(...)` syntax work.

- [ ] **Step 2: Re-export from reply/mod.rs**

In `crates/heartbit-ghost/src/reply/mod.rs`, add:

```rust
pub mod prompts;
```

(Keep the prompt functions `pub(crate)` — they're internal to the pipeline.)

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib reply::prompts::tests
```

Expected: 3 PASS.

- [ ] **Step 4: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/prompts.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "feat(ghost): reply prompt builders — research / writer / critic / fact

Pure string composition. Mirrors pipeline/prompts.rs shape but for
mention-anchored replies (parent + mention + mentioner context).
Writer message respects mode_addendum semantics from P1.3 plumbing.

heartbit-ghost P1.5a — task 3/12."
```

---

## Task 4: `ReplyConfig`, `ReplyOutput`, `ReplyOutcome`, `ReplyError` types

**Files:**
- Modify: `crates/heartbit-ghost/src/reply/mod.rs` (add the config + output + error types; runtime lands in Task 5)

- [ ] **Step 1: Read the existing `ReviewConfig` for reference**

```bash
sed -n '29,80p' crates/heartbit-ghost/src/review/mod.rs
```

`ReplyConfig` shares 60% of `ReviewConfig`'s fields — match the pattern.

- [ ] **Step 2: Write the failing tests for the config**

Append to `crates/heartbit-ghost/src/reply/mod.rs`:

```rust
use std::path::Path;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::Tool;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::llm::types::TokenUsage;
use thiserror::Error;

use crate::pipeline::{PipelineError, ProgressCallback, ResearcherOverride};

/// Configuration for one reply-pipeline run.
#[derive(Clone)]
pub struct ReplyConfig<'a> {
    pub persona_name: &'a str,
    pub provider: Arc<BoxedProvider>,
    pub corpora_root: &'a Path,
    pub profiles_root: &'a Path,
    pub on_progress: Option<ProgressCallback>,
    /// The mention being replied to.
    pub mention: Mention,
    /// The operator's tweet the mention is replying to (when the mention
    /// is a thread reply rather than a top-level @-mention).
    pub parent: Option<TweetSnapshot>,
    /// Optional bio + recent tweets for tone calibration.
    pub mentioner_context: Option<MentionerContext>,
    /// Number of distinct candidate replies to generate (1..=3).
    /// 1 = no judge; 2 or 3 = judge picks.
    pub candidates_per_reply: usize,
    /// Persona-specific mode addendum.
    pub mode_addendum: Option<&'a str>,
    /// Optional researcher override (same semantics as PipelineConfig).
    pub researcher_override: Option<ResearcherOverride>,
    /// Telegram-or-mock delivery layer for the reply review.
    pub delivery: Arc<dyn ReplyReviewDelivery>,
    /// `twitter_reply` tool — production wires `Arc::new(TwitterReplyTool::new())`;
    /// tests wire a mock.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver for `twitter_tool`.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Output of a successful reply-pipeline run.
#[derive(Debug, Clone)]
pub struct ReplyOutput {
    pub mention_id: String,
    pub candidates: Vec<ReplyCandidateRecord>,
    pub usage_summary: TokenUsage,
    pub outcome: ReplyOutcome,
}

/// One generated reply draft (post-style/fact, pre-publish_gate).
#[derive(Debug, Clone)]
pub struct ReplyCandidateRecord {
    pub draft: String,
    pub style_match_score: f32,
    pub fact_check_verdict: String, // "verified" | "unverifiable: reason" | "rejected: reason"
}

/// What happened in this reply run.
#[derive(Debug, Clone)]
pub enum ReplyOutcome {
    /// User picked candidate `chosen_index` and the reply was published.
    Posted {
        chosen_index: usize,
        reply_tweet_id: String,
        reply_url: String,
    },
    /// User pressed Skip.
    Skipped,
    /// Telegram review timed out without a pick.
    TimedOut,
    /// User picked `chosen_index` but `publish_gate` rejected it.
    GateRejected {
        chosen_index: usize,
        reason: String,
    },
    /// User picked `chosen_index` but the X API call failed.
    PublishFailed {
        chosen_index: usize,
        reason: String,
    },
    /// All candidates returned the literal "no_reply" string — the
    /// writer chose not to engage. No Telegram review was sent.
    NoReply,
}

/// Errors raised by `run_reply_pipeline`.
#[derive(Debug, Error)]
pub enum ReplyError {
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("delivery: {0}")]
    Delivery(#[from] crate::review::ReviewDeliveryError),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Telegram-or-mock review delivery for reply messages. Mirrors
/// [`crate::review::ReviewDelivery`] but with a parent-quoted message
/// shape. See spec §9 for the full text layout.
#[async_trait::async_trait]
pub trait ReplyReviewDelivery: Send + Sync {
    async fn deliver(
        &self,
        msg: ReplyReviewMessage,
    ) -> Result<crate::review::DeliveredReview, crate::review::ReviewDeliveryError>;
    async fn report(
        &self,
        receipt: crate::review::DeliveryReceipt,
        outcome: ReplyOutcome,
    ) -> Result<(), crate::review::ReviewDeliveryError>;
}

/// Message body for a reply review. The Telegram impl renders this as
/// the parent + mention + drafts layout from spec §9.1.
#[derive(Debug, Clone)]
pub struct ReplyReviewMessage {
    pub mention: Mention,
    pub parent: Option<TweetSnapshot>,
    pub mentioner_context: Option<MentionerContext>,
    pub candidates: Vec<String>,
    pub interaction_timeout_seconds: u64,
}
```

Then append a test:

```rust
#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn reply_outcome_no_reply_is_distinct_from_skipped() {
        let a = ReplyOutcome::Skipped;
        let b = ReplyOutcome::NoReply;
        // No Eq derive (CredentialResolver isn't Eq); rely on debug
        // representation as a stand-in for "these are different variants".
        assert_ne!(format!("{a:?}"), format!("{b:?}"));
    }

    #[test]
    fn reply_error_display_round_trips() {
        let e = ReplyError::InvalidConfig("test".to_string());
        assert!(format!("{e}").contains("invalid config"));
    }
}
```

- [ ] **Step 3: Verify `async_trait` is in heartbit-ghost's Cargo.toml**

```bash
grep "^async_trait" crates/heartbit-ghost/Cargo.toml
```

If absent, add to `[dependencies]`:

```toml
async-trait = { workspace = true }
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p heartbit-ghost --lib reply
```

Expected: 4 PASS (2 from Task 2 + 2 from this task), plus the 3 prompt tests from Task 3 = 7 total in `reply::*`.

- [ ] **Step 5: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/mod.rs crates/heartbit-ghost/Cargo.toml
git commit -m "feat(ghost): ReplyConfig + ReplyOutput + ReplyOutcome + ReplyReviewDelivery trait

Public surface for the reply pipeline. Mirrors ReviewConfig where the
shape overlaps but adds reply-specific fields: mention, parent,
mentioner_context, candidates_per_reply (1..=3, narrower than the
1..=10 main pipeline). New ReplyOutcome variant 'NoReply' for when
the writer chooses not to engage at all.

heartbit-ghost P1.5a — task 4/12."
```

---

## Task 5: `run_reply_pipeline` body + integration tests

**Files:**
- Modify: `crates/heartbit-ghost/src/reply/mod.rs` (add `run_reply_pipeline` function)

- [ ] **Step 1: Read `run_pipeline` for reference**

```bash
sed -n '441,540p' crates/heartbit-ghost/src/pipeline/mod.rs
```

The reply pipeline mirrors the candidate-generation pattern but with shorter outputs and no image stage.

- [ ] **Step 2: Write the failing integration test (single-candidate happy path)**

In `crates/heartbit-ghost/src/reply/mod.rs`, add a test module that drives `run_reply_pipeline`:

```rust
#[cfg(test)]
mod runtime_tests {
    use super::*;
    use crate::pipeline::tests::MockProvider;
    use chrono::TimeZone;

    fn fixture_mention() -> Mention {
        Mention {
            id: "mention_1".into(),
            text: "how does it compare to rig-rs?".into(),
            author_id: "777".into(),
            author_handle: "grumpy_dev".into(),
            posted_at: Utc.with_ymd_and_hms(2026, 5, 8, 11, 2, 0).unwrap(),
            in_reply_to_tweet_id: Some("parent_1".into()),
        }
    }

    // … additional fixture helpers … MockReplyReviewDelivery, MockReplyTool
    // (mirror crate::review::tests' mocks; copy + adapt for reply shape).

    #[tokio::test]
    async fn run_reply_pipeline_single_candidate_happy_path() {
        // Wire: research → reply_writer → style_critic → fact_check → publish_gate
        //       → delivery (auto-pick 0) → twitter_reply (mock returns success).
        // Assert: outcome is Posted { chosen_index: 0, … }.
        // (Full code: ~80 lines; mirror review/mod.rs's
        // run_review_pipeline_pick_index_0_posts_to_twitter test verbatim
        // with the reply-shape adaptations.)
        todo!("reference run_review_pipeline_pick_index_0_posts_to_twitter for the template")
    }
}
```

The plan defers writing the verbose fixture/mock code inline; the implementer task expands these into the full test bodies, mirroring `crates/heartbit-ghost/src/review/mod.rs`'s test module patterns. **Each `todo!()` corresponds to one full integration test.**

The 6 integration tests required by the spec:
1. `run_reply_pipeline_single_candidate_happy_path` — outcome Posted
2. `run_reply_pipeline_two_candidates_judge_picks` — outcome Posted, judge picked index 1
3. `run_reply_pipeline_writer_no_reply_returns_no_reply_outcome` — all candidates "no_reply" → outcome NoReply, no Telegram fired
4. `run_reply_pipeline_publish_gate_rejects_281_chars` — outcome GateRejected
5. `run_reply_pipeline_user_skip_returns_skipped` — outcome Skipped, no twitter_reply call
6. `run_reply_pipeline_twitter_api_error_returns_publish_failed` — outcome PublishFailed

- [ ] **Step 3: Implement `run_reply_pipeline`**

Append to `reply/mod.rs`:

```rust
/// Execute one reply pipeline. Returns when the user picks (and the
/// reply posts), skips, times out, or all candidates return "no_reply".
pub async fn run_reply_pipeline(cfg: ReplyConfig<'_>) -> Result<ReplyOutput, ReplyError> {
    use crate::agents::{
        fact_check_recipe, judge_recipe, reply_writer_recipe, researcher_recipe,
        style_critic_recipe,
    };
    use heartbit_core::ExecutionContext;
    use heartbit_core::llm::types::TokenUsage;

    // 1. Validate.
    if !(1..=3).contains(&cfg.candidates_per_reply) {
        return Err(ReplyError::InvalidConfig(format!(
            "candidates_per_reply must be in 1..=3 (got {})",
            cfg.candidates_per_reply,
        )));
    }

    let progress = |s: &str| {
        if let Some(cb) = cfg.on_progress.as_ref() {
            cb(s);
        }
    };

    // 2. Load profile snapshot.
    let snapshot = crate::voice::SnapshotStore::open(cfg.profiles_root, cfg.persona_name)
        .map_err(|e| ReplyError::Pipeline(PipelineError::Snapshot {
            persona: cfg.persona_name.to_string(),
            profiles_dir: cfg.profiles_root.join(cfg.persona_name),
            source: anyhow::anyhow!("{e:?}"),
        }))?
        .latest()
        .map_err(|e| ReplyError::Pipeline(PipelineError::Snapshot {
            persona: cfg.persona_name.to_string(),
            profiles_dir: cfg.profiles_root.join(cfg.persona_name),
            source: anyhow::anyhow!("{e:?}"),
        }))?;
    let profile = snapshot.profile;

    // 3. Build the 5 sub-agent runners (researcher / writer / critic / fact /
    //    optional judge) using runner_from_recipe. Honor researcher_override.
    let (researcher_recipe_used, researcher_tools): (
        heartbit_core::config::AgentConfig,
        Vec<std::sync::Arc<dyn Tool>>,
    ) = match cfg.researcher_override.as_ref() {
        Some((recipe, tools)) => ((**recipe).clone_config(), tools.clone()),
        None => (
            researcher_recipe(),
            // Replies do NOT need web search by default — context comes from
            // the parent + mentioner_context already passed into the user
            // message. Empty tool set keeps the researcher focused.
            Vec::new(),
        ),
    };
    let researcher = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        researcher_recipe_used,
        researcher_tools,
    )
    .map_err(|e| ReplyError::Pipeline(PipelineError::Builder {
        stage: "researcher".to_string(),
        source: e,
    }))?;
    let writer = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        reply_writer_recipe(),
        Vec::new(),
    )
    .map_err(|e| ReplyError::Pipeline(PipelineError::Builder {
        stage: "reply_writer".to_string(),
        source: e,
    }))?;
    let critic = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        style_critic_recipe(),
        Vec::new(),
    )
    .map_err(|e| ReplyError::Pipeline(PipelineError::Builder {
        stage: "style_critic".to_string(),
        source: e,
    }))?;
    let fact = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        fact_check_recipe(),
        Vec::new(),
    )
    .map_err(|e| ReplyError::Pipeline(PipelineError::Builder {
        stage: "fact_check".to_string(),
        source: e,
    }))?;
    // Judge is built lazily — only when candidates_per_reply > 1.
    let judge = if cfg.candidates_per_reply > 1 {
        Some(crate::pipeline::runner_from_recipe(
            cfg.provider.clone(),
            judge_recipe(),
            Vec::new(),
        )
        .map_err(|e| ReplyError::Pipeline(PipelineError::Builder {
            stage: "judge".to_string(),
            source: e,
        }))?)
    } else {
        None
    };

    let mut total_usage = TokenUsage::default();
    let voice_guidelines =
        crate::pipeline::render_style_profile_as_english(&profile);

    // 4. Run researcher.
    progress("Researching mention…");
    let research_msg = prompts::build_reply_research_user_message(
        &cfg.mention,
        cfg.parent.as_ref(),
        cfg.mentioner_context.as_ref(),
    );
    let researcher_out = researcher.execute(&research_msg).await.map_err(|e| {
        ReplyError::Pipeline(PipelineError::AgentExec {
            stage: "researcher".to_string(),
            source: e,
        })
    })?;
    let digest = researcher_out.result.clone();
    total_usage += researcher_out.tokens_used;

    // 5. Generate N reply candidates in parallel via tokio::JoinSet.
    progress(&format!(
        "Generating {} candidate(s) in parallel…",
        cfg.candidates_per_reply
    ));
    let mut joinset: tokio::task::JoinSet<_> = tokio::task::JoinSet::new();
    let writer = std::sync::Arc::new(writer);
    let critic = std::sync::Arc::new(critic);
    let fact = std::sync::Arc::new(fact);
    let voice_owned: std::sync::Arc<str> = voice_guidelines.clone().into();
    let digest_owned: std::sync::Arc<str> = digest.clone().into();
    let mode_owned: Option<std::sync::Arc<str>> =
        cfg.mode_addendum.map(std::sync::Arc::from);
    for _ in 0..cfg.candidates_per_reply {
        let writer = writer.clone();
        let critic = critic.clone();
        let fact = fact.clone();
        let voice = voice_owned.clone();
        let digest = digest_owned.clone();
        let mode = mode_owned.clone();
        joinset.spawn(async move {
            let writer_msg = prompts::build_reply_writer_user_message(
                &digest,
                &voice,
                mode.as_deref(),
            );
            let writer_out = writer.execute(&writer_msg).await?;
            let draft = writer_out.result.trim().to_string();
            // 5a. Writer-driven no_reply short-circuit.
            if draft.eq_ignore_ascii_case("no_reply") {
                return Ok::<_, heartbit_core::Error>((draft, 0.0_f32, "no_reply".to_string(), writer_out.tokens_used));
            }
            // 5b. Style critic.
            let critic_msg = prompts::build_reply_critic_user_message(&draft, &voice);
            let critic_out = critic.execute(&critic_msg).await?;
            let style_score = parse_style_match_score(&critic_out.result).unwrap_or(0.5);
            // 5c. Fact check.
            let fact_msg = prompts::build_reply_fact_user_message(&draft, &digest);
            let fact_out = fact.execute(&fact_msg).await?;
            let fact_verdict = fact_out.result.clone();
            let usage = writer_out.tokens_used + critic_out.tokens_used + fact_out.tokens_used;
            Ok((draft, style_score, fact_verdict, usage))
        });
    }
    let mut survivors: Vec<ReplyCandidateRecord> = Vec::new();
    while let Some(handle) = joinset.join_next().await {
        let (draft, style_score, fact_verdict, usage) = handle
            .map_err(|e| ReplyError::Pipeline(PipelineError::AgentExec {
                stage: "candidate".to_string(),
                source: heartbit_core::Error::Agent(format!("join: {e}")),
            }))?
            .map_err(|e| ReplyError::Pipeline(PipelineError::AgentExec {
                stage: "candidate".to_string(),
                source: e,
            }))?;
        total_usage += usage;
        if !draft.eq_ignore_ascii_case("no_reply") {
            survivors.push(ReplyCandidateRecord {
                draft,
                style_match_score: style_score,
                fact_check_verdict: fact_verdict,
            });
        }
    }

    // 6. If all candidates were no_reply, return early without delivery.
    if survivors.is_empty() {
        return Ok(ReplyOutput {
            mention_id: cfg.mention.id.clone(),
            candidates: Vec::new(),
            usage_summary: total_usage,
            outcome: ReplyOutcome::NoReply,
        });
    }

    // 7. Judge if multiple survivors (skip when 1).
    let chosen_index: usize = if let (Some(judge), true) = (judge.as_ref(), survivors.len() > 1) {
        progress("Judging candidates…");
        let judge_msg = format!(
            "{voice_guidelines}\n\nCandidate replies for the mention from @{}:\n\n{}\n\nReturn your verdict as JSON per the schema.\n",
            cfg.mention.author_handle,
            survivors
                .iter()
                .enumerate()
                .map(|(i, c)| format!("[{i}]\n{}\n", c.draft))
                .collect::<String>(),
        );
        let judge_out = judge.execute(&judge_msg).await.map_err(|e| {
            ReplyError::Pipeline(PipelineError::AgentExec {
                stage: "judge".to_string(),
                source: e,
            })
        })?;
        total_usage += judge_out.tokens_used;
        parse_judge_index(&judge_out.result, survivors.len()).unwrap_or(0)
    } else {
        0
    };

    let chosen_draft = survivors[chosen_index].draft.clone();

    // 8. Publish gate — hard 280-char cap, no thread split.
    if chosen_draft.chars().count() > 280 {
        return Ok(ReplyOutput {
            mention_id: cfg.mention.id.clone(),
            candidates: survivors,
            usage_summary: total_usage,
            outcome: ReplyOutcome::GateRejected {
                chosen_index,
                reason: format!(
                    "draft exceeds 280 chars (got {})",
                    chosen_draft.chars().count(),
                ),
            },
        });
    }

    // 9. Telegram review delivery.
    progress("Sending review to user…");
    let drafts_for_review: Vec<String> = survivors.iter().map(|c| c.draft.clone()).collect();
    let msg = ReplyReviewMessage {
        mention: cfg.mention.clone(),
        parent: cfg.parent.clone(),
        mentioner_context: cfg.mentioner_context.clone(),
        candidates: drafts_for_review.clone(),
        interaction_timeout_seconds: 300,
    };
    let delivered = cfg.delivery.deliver(msg).await?;
    let outcome = match delivered.outcome {
        crate::review::DeliveryOutcome::Picked(idx) if idx < survivors.len() => {
            // 10. twitter_reply tool call.
            progress(&format!("Posting candidate {idx}…"));
            let exec_ctx = ExecutionContext::new(cfg.credentials.clone());
            let tool_input = serde_json::json!({
                "text": survivors[idx].draft,
                "in_reply_to": cfg.mention.id,
            });
            match cfg.twitter_tool.execute(&exec_ctx, tool_input).await {
                Ok(out) if !out.is_error => {
                    let parsed = parse_reply_tool_output(&out.content);
                    ReplyOutcome::Posted {
                        chosen_index: idx,
                        reply_tweet_id: parsed.0,
                        reply_url: parsed.1,
                    }
                }
                Ok(out) => ReplyOutcome::PublishFailed {
                    chosen_index: idx,
                    reason: out.content,
                },
                Err(e) => ReplyOutcome::PublishFailed {
                    chosen_index: idx,
                    reason: format!("{e}"),
                },
            }
        }
        crate::review::DeliveryOutcome::Picked(_) => ReplyOutcome::Skipped, // unreachable but conservative
        crate::review::DeliveryOutcome::Skipped => ReplyOutcome::Skipped,
        crate::review::DeliveryOutcome::TimedOut => ReplyOutcome::TimedOut,
    };

    // 11. Optional report-back to delivery (non-fatal).
    let _ = cfg.delivery.report(delivered.receipt, outcome.clone()).await;

    Ok(ReplyOutput {
        mention_id: cfg.mention.id.clone(),
        candidates: survivors,
        usage_summary: total_usage,
        outcome,
    })
}

// Helpers —————————————————————————————————————

fn parse_style_match_score(raw: &str) -> Option<f32> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    v.get("style_match_score")?.as_f64().map(|x| x as f32)
}

fn parse_judge_index(raw: &str, n: usize) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    let idx = v.get("chosen_index")?.as_u64()? as usize;
    if idx < n { Some(idx) } else { None }
}

fn parse_reply_tool_output(content: &str) -> (String, String) {
    #[derive(serde::Deserialize)]
    struct Parsed {
        tweet_id: String,
        url: String,
    }
    serde_json::from_str::<Parsed>(content)
        .map(|p| (p.tweet_id, p.url))
        .unwrap_or_else(|_| (String::new(), "<unknown>".to_string()))
}
```

- [ ] **Step 4: Implement the 6 integration tests**

Replace each `todo!()` from Step 2 with a full test body, mirroring the pattern in `crates/heartbit-ghost/src/review/mod.rs::tests`. Each test sets up a `MockProvider::route_with_recorder` that returns canned responses for "research analyst" / "social media writer" or rather "single short reply" / "score how well a draft post" / "verify the factual claims" / (optional) "produce a JSON verdict picking the best", a `MockReplyReviewDelivery` that returns a configured outcome, a `MockReplyTool` that returns success or error JSON, then asserts on the resulting `ReplyOutput.outcome`.

- [ ] **Step 5: Run the tests**

```bash
cargo test -p heartbit-ghost --lib reply
```

Expected: all 6 integration tests pass + the prior 4 (types) + 3 (prompts) = 13 in `reply::*`.

- [ ] **Step 6: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/mod.rs
git commit -m "feat(ghost): run_reply_pipeline — research / writer / critic / fact / judge / gate

End-to-end reply pipeline. Mirrors run_pipeline's parallel-candidate
shape with three reply-specific changes: short-circuit when the writer
emits the literal 'no_reply' string (returns ReplyOutcome::NoReply
without firing Telegram); judge skipped when candidates_per_reply==1;
publish_gate is a hard 280-char cap with no thread-split fallback.

6 integration tests cover the 6 outcome variants. Calibration mode
only — every reply goes through Telegram before posting.

heartbit-ghost P1.5a — task 5/12."
```

---

## Task 6: `persona reply once` CLI subcommand

**Files:**
- Modify: `crates/heartbit-cli/src/persona.rs` (add `reply` subcommand)
- Modify: `crates/heartbit-cli/src/persona_review.rs` (helper to construct ReplyConfig from env, similar to `review_config_from_env`)

- [ ] **Step 1: Read the existing `persona run` to see the dispatch pattern**

```bash
sed -n '160,250p' crates/heartbit-cli/src/persona.rs
```

The new `Reply { name, mention_id }` arm follows the same shape: resolve persona, expand, build provider, build config, call `run_reply_pipeline`.

- [ ] **Step 2: Add the clap subcommand**

In `crates/heartbit-cli/src/persona.rs` (or wherever the `PersonaCommand` enum lives), add a new variant:

```rust
PersonaCommand::Reply {
    name: String,
    mention_id: String,
}
```

(Use `clap`'s derive macros; mirror the existing `Run` variant.)

- [ ] **Step 3: Implement the dispatch arm**

In the same file, add a new match arm:

```rust
PersonaCommand::Reply { name, mention_id } => {
    let persona = registry.get(&name).ok_or_else(|| {
        anyhow!("persona '{name}' not found. {}", registry_suffix(registry))
    })?;
    let expansion = persona
        .expand(&PersonaParams::default())
        .map_err(|e| anyhow!("expand persona '{name}': {e}"))?;

    let provider = build_provider_from_env(None).map_err(|e| anyhow!("build llm provider: {e}"))?;
    let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
        .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
    let profiles_root = heartbit_ghost::voice::default_profiles_dir()
        .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;

    // Researcher override (same pattern as Run).
    let researcher_override = expansion
        .agents
        .iter()
        .find(|a| a.name == "repo_researcher")
        .map(|recipe| {
            let recipe = std::sync::Arc::new(recipe.clone_config());
            let tools: Vec<std::sync::Arc<dyn heartbit_core::Tool>> = expansion
                .tools
                .iter()
                .filter(|t| t.definition().name == "repo_inspect")
                .cloned()
                .collect();
            (recipe, tools)
        });

    // Fetch the mention via TwitterMentionsTool / single-tweet GET.
    // For the CLI one-off, we accept the mention_id and a hand-fed
    // `<USER_ID>` env var: HEARTBIT_GHOST_REPLY_USER_ID.
    let user_id = std::env::var("HEARTBIT_GHOST_REPLY_USER_ID")
        .context("HEARTBIT_GHOST_REPLY_USER_ID env var must be set for `persona reply once`")?;

    // Construct the Mention from a single GET /2/tweets/:id call (build a
    // minimal XClient ad-hoc; production daemon path uses the cron handler).
    let mention = crate::persona_review::fetch_mention_one_off(&mention_id, &user_id)
        .await
        .map_err(|e| anyhow!("fetch mention: {e}"))?;

    let cfg = crate::persona_review::reply_config_from_env(
        &name,
        provider,
        &corpora_root,
        &profiles_root,
        Some(std::sync::Arc::new(|s: &str| eprintln!("> {s}"))),
        mention,
        None,                  // parent — let the daemon path fetch this; CLI one-off skips
        None,                  // mentioner_context — same
        2,                     // candidates_per_reply
        expansion.mode_addendum,
        researcher_override,
    )
    .await
    .map_err(|e| anyhow!("reply config: {e}"))?;

    let output = heartbit_ghost::reply::run_reply_pipeline(cfg)
        .await
        .map_err(|e| anyhow!("reply pipeline: {e}"))?;
    eprintln!("> ok: outcome={:?}", output.outcome);
    Ok(())
}
```

- [ ] **Step 4: Add the `reply_config_from_env` + `fetch_mention_one_off` helpers**

In `crates/heartbit-cli/src/persona_review.rs`, mirror `review_config_from_env` for replies:

```rust
#[allow(clippy::too_many_arguments)]
pub async fn reply_config_from_env<'a>(
    persona_name: &'a str,
    provider: std::sync::Arc<heartbit_core::llm::BoxedProvider>,
    corpora_root: &'a std::path::Path,
    profiles_root: &'a std::path::Path,
    on_progress: Option<heartbit_ghost::pipeline::ProgressCallback>,
    mention: heartbit_ghost::reply::Mention,
    parent: Option<heartbit_ghost::reply::TweetSnapshot>,
    mentioner_context: Option<heartbit_ghost::reply::MentionerContext>,
    candidates_per_reply: usize,
    mode_addendum: Option<&'static str>,
    researcher_override: Option<heartbit_ghost::pipeline::ResearcherOverride>,
) -> anyhow::Result<heartbit_ghost::reply::ReplyConfig<'a>> {
    let delivery: std::sync::Arc<dyn heartbit_ghost::reply::ReplyReviewDelivery> =
        std::sync::Arc::new(TelegramReplyReviewDelivery::from_env()?);
    let twitter_tool: std::sync::Arc<dyn heartbit_core::tool::Tool> =
        std::sync::Arc::new(heartbit_ghost::tools::TwitterReplyTool::new());
    let credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver> =
        std::sync::Arc::new(EnvCredentialResolver);
    Ok(heartbit_ghost::reply::ReplyConfig {
        persona_name,
        provider,
        corpora_root,
        profiles_root,
        on_progress,
        mention,
        parent,
        mentioner_context,
        candidates_per_reply,
        mode_addendum,
        researcher_override,
        delivery,
        twitter_tool,
        credentials,
    })
}

pub async fn fetch_mention_one_off(
    mention_id: &str,
    operator_user_id: &str,
) -> anyhow::Result<heartbit_ghost::reply::Mention> {
    // Minimal one-off: signed GET /2/tweets/:id call, parse into Mention.
    // Implementation detail; ~30 lines using the same OAuth1 helper the
    // Python sanity-test scripts in the repo use, but in Rust.
    todo!("inline OAuth1 GET /2/tweets/:mention_id, parse Mention struct")
}
```

(The `TelegramReplyReviewDelivery` is built in Task 11; this CLI subcommand may need to land alongside Task 11 in practice. Mark this caveat in the commit message.)

- [ ] **Step 5: Run CLI tests**

```bash
cargo test -p heartbit-cli
```

Expected: PASS. (No new tests in this task — the CLI dispatch is an integration concern best validated by the live test in Task 12.)

- [ ] **Step 6: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-cli --all-targets -- -D warnings
git add crates/heartbit-cli/src/persona.rs crates/heartbit-cli/src/persona_review.rs
git commit -m "feat(cli): persona reply once <NAME> --mention-id <ID> subcommand

One-off CLI for testing the reply pipeline without the daemon. Reads
HEARTBIT_GHOST_REPLY_USER_ID for the operator's X user id, fetches
the mention via a one-shot GET /2/tweets/:id, builds a ReplyConfig,
and dispatches to run_reply_pipeline. Telegram delivery still goes
through TelegramReplyReviewDelivery (built in task 11) — this commit
depends on that adapter's existence.

heartbit-ghost P1.5a — task 6/12."
```

---

## Task 7: `MentionStore` trait + `InMemoryMentionStore` + `JsonlMentionStore`

**Files:**
- Create: `crates/heartbit-ghost/src/reply/storage.rs`
- Modify: `crates/heartbit-ghost/src/reply/mod.rs` (re-export)

- [ ] **Step 1: Write the failing tests**

Create `crates/heartbit-ghost/src/reply/storage.rs`:

```rust
//! Mention storage for the reply pipeline. Tracks (per persona, per
//! operator user id) the highest mention id we've seen (`since_id`)
//! plus the set of mention ids we've already replied to.

use chrono::{DateTime, Utc};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(String),
}

#[async_trait::async_trait]
pub trait MentionStore: Send + Sync {
    async fn since_id_for(
        &self,
        persona: &str,
        user_id: &str,
    ) -> Result<Option<String>, StoreError>;
    async fn bump_since_id(
        &self,
        persona: &str,
        user_id: &str,
        new_id: &str,
    ) -> Result<(), StoreError>;
    async fn mark_replied(&self, mention_id: &str) -> Result<(), StoreError>;
    async fn was_replied(&self, mention_id: &str) -> Result<bool, StoreError>;
    /// Number of replies sent to `author_id` since `since`. Used by the
    /// per-author rate limit (default: max 3 / 24h).
    async fn replies_to_author_since(
        &self,
        author_id: &str,
        since: DateTime<Utc>,
    ) -> Result<usize, StoreError>;
    /// Record that we just replied to a mention authored by `author_id`.
    async fn record_reply_to_author(
        &self,
        author_id: &str,
        ts: DateTime<Utc>,
    ) -> Result<(), StoreError>;
}

// In-memory impl —————————————————————————————————————

pub struct InMemoryMentionStore {
    inner: tokio::sync::RwLock<InMemoryInner>,
}

#[derive(Default)]
struct InMemoryInner {
    /// (persona, user_id) → since_id
    since: std::collections::HashMap<(String, String), String>,
    /// mention_id → ()
    replied: std::collections::HashSet<String>,
    /// (author_id, ts) — append log for rate-limit queries.
    author_replies: Vec<(String, DateTime<Utc>)>,
}

impl Default for InMemoryMentionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryMentionStore {
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::RwLock::new(InMemoryInner::default()),
        }
    }
}

#[async_trait::async_trait]
impl MentionStore for InMemoryMentionStore {
    async fn since_id_for(
        &self,
        persona: &str,
        user_id: &str,
    ) -> Result<Option<String>, StoreError> {
        let g = self.inner.read().await;
        Ok(g.since.get(&(persona.to_string(), user_id.to_string())).cloned())
    }

    async fn bump_since_id(
        &self,
        persona: &str,
        user_id: &str,
        new_id: &str,
    ) -> Result<(), StoreError> {
        let mut g = self.inner.write().await;
        let key = (persona.to_string(), user_id.to_string());
        match g.since.get(&key) {
            Some(prev) if prev.as_str() >= new_id => {} // monotonic
            _ => {
                g.since.insert(key, new_id.to_string());
            }
        }
        Ok(())
    }

    async fn mark_replied(&self, mention_id: &str) -> Result<(), StoreError> {
        self.inner
            .write()
            .await
            .replied
            .insert(mention_id.to_string());
        Ok(())
    }

    async fn was_replied(&self, mention_id: &str) -> Result<bool, StoreError> {
        Ok(self.inner.read().await.replied.contains(mention_id))
    }

    async fn replies_to_author_since(
        &self,
        author_id: &str,
        since: DateTime<Utc>,
    ) -> Result<usize, StoreError> {
        let g = self.inner.read().await;
        Ok(g.author_replies
            .iter()
            .filter(|(a, ts)| a == author_id && *ts >= since)
            .count())
    }

    async fn record_reply_to_author(
        &self,
        author_id: &str,
        ts: DateTime<Utc>,
    ) -> Result<(), StoreError> {
        self.inner
            .write()
            .await
            .author_replies
            .push((author_id.to_string(), ts));
        Ok(())
    }
}

// JSONL-backed impl ——————————————————————————————————

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum StoreEvent {
    SinceId {
        persona: String,
        user_id: String,
        id: String,
    },
    Replied {
        mention_id: String,
    },
    AuthorReply {
        author_id: String,
        ts: DateTime<Utc>,
    },
}

pub struct JsonlMentionStore {
    path: PathBuf,
    inner: tokio::sync::RwLock<InMemoryInner>,
}

impl JsonlMentionStore {
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let inner = InMemoryInner::default();
        let store = Self {
            path,
            inner: tokio::sync::RwLock::new(inner),
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
            let evt: StoreEvent = serde_json::from_str(line)
                .map_err(|e| StoreError::Parse(format!("line {line:?}: {e}")))?;
            match evt {
                StoreEvent::SinceId { persona, user_id, id } => {
                    g.since.insert((persona, user_id), id);
                }
                StoreEvent::Replied { mention_id } => {
                    g.replied.insert(mention_id);
                }
                StoreEvent::AuthorReply { author_id, ts } => {
                    g.author_replies.push((author_id, ts));
                }
            }
        }
        Ok(())
    }

    async fn append(&self, evt: &StoreEvent) -> Result<(), StoreError> {
        use tokio::io::AsyncWriteExt;
        let serialized = serde_json::to_string(evt)
            .map_err(|e| StoreError::Parse(format!("{e}")))?;
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

#[async_trait::async_trait]
impl MentionStore for JsonlMentionStore {
    async fn since_id_for(
        &self,
        persona: &str,
        user_id: &str,
    ) -> Result<Option<String>, StoreError> {
        Ok(self
            .inner
            .read()
            .await
            .since
            .get(&(persona.to_string(), user_id.to_string()))
            .cloned())
    }

    async fn bump_since_id(
        &self,
        persona: &str,
        user_id: &str,
        new_id: &str,
    ) -> Result<(), StoreError> {
        // monotonic check first (avoid spurious appends)
        {
            let g = self.inner.read().await;
            if let Some(prev) = g.since.get(&(persona.to_string(), user_id.to_string())) {
                if prev.as_str() >= new_id {
                    return Ok(());
                }
            }
        }
        self.append(&StoreEvent::SinceId {
            persona: persona.to_string(),
            user_id: user_id.to_string(),
            id: new_id.to_string(),
        })
        .await?;
        self.inner
            .write()
            .await
            .since
            .insert((persona.to_string(), user_id.to_string()), new_id.to_string());
        Ok(())
    }

    async fn mark_replied(&self, mention_id: &str) -> Result<(), StoreError> {
        self.append(&StoreEvent::Replied {
            mention_id: mention_id.to_string(),
        })
        .await?;
        self.inner
            .write()
            .await
            .replied
            .insert(mention_id.to_string());
        Ok(())
    }

    async fn was_replied(&self, mention_id: &str) -> Result<bool, StoreError> {
        Ok(self.inner.read().await.replied.contains(mention_id))
    }

    async fn replies_to_author_since(
        &self,
        author_id: &str,
        since: DateTime<Utc>,
    ) -> Result<usize, StoreError> {
        let g = self.inner.read().await;
        Ok(g.author_replies
            .iter()
            .filter(|(a, ts)| a == author_id && *ts >= since)
            .count())
    }

    async fn record_reply_to_author(
        &self,
        author_id: &str,
        ts: DateTime<Utc>,
    ) -> Result<(), StoreError> {
        self.append(&StoreEvent::AuthorReply {
            author_id: author_id.to_string(),
            ts,
        })
        .await?;
        self.inner
            .write()
            .await
            .author_replies
            .push((author_id.to_string(), ts));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[tokio::test]
    async fn in_memory_since_id_is_monotonic() {
        let store = InMemoryMentionStore::new();
        assert_eq!(store.since_id_for("p", "u").await.unwrap(), None);
        store.bump_since_id("p", "u", "100").await.unwrap();
        store.bump_since_id("p", "u", "50").await.unwrap(); // older — should NOT regress
        store.bump_since_id("p", "u", "200").await.unwrap();
        assert_eq!(store.since_id_for("p", "u").await.unwrap().unwrap(), "200");
    }

    #[tokio::test]
    async fn in_memory_replied_set_round_trips() {
        let store = InMemoryMentionStore::new();
        assert!(!store.was_replied("m1").await.unwrap());
        store.mark_replied("m1").await.unwrap();
        assert!(store.was_replied("m1").await.unwrap());
    }

    #[tokio::test]
    async fn in_memory_per_author_rate_count_filters_by_since() {
        let store = InMemoryMentionStore::new();
        let now = Utc::now();
        store.record_reply_to_author("a1", now - Duration::hours(48)).await.unwrap();
        store.record_reply_to_author("a1", now - Duration::hours(12)).await.unwrap();
        store.record_reply_to_author("a1", now - Duration::hours(1)).await.unwrap();
        let recent = store
            .replies_to_author_since("a1", now - Duration::hours(24))
            .await
            .unwrap();
        assert_eq!(recent, 2, "should count only the within-24h entries");
    }

    #[tokio::test]
    async fn jsonl_store_round_trips_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mentions.jsonl");
        {
            let s1 = JsonlMentionStore::open(&path).await.unwrap();
            s1.bump_since_id("p", "u", "100").await.unwrap();
            s1.mark_replied("m1").await.unwrap();
            s1.record_reply_to_author("a1", Utc::now()).await.unwrap();
        }
        // Reload from disk.
        let s2 = JsonlMentionStore::open(&path).await.unwrap();
        assert_eq!(s2.since_id_for("p", "u").await.unwrap().unwrap(), "100");
        assert!(s2.was_replied("m1").await.unwrap());
    }

    #[tokio::test]
    async fn jsonl_store_handles_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.jsonl");
        let s = JsonlMentionStore::open(&path).await.unwrap();
        assert_eq!(s.since_id_for("p", "u").await.unwrap(), None);
        assert!(!s.was_replied("m1").await.unwrap());
    }
}
```

- [ ] **Step 2: Re-export from reply/mod.rs**

```rust
pub mod storage;
pub use storage::{InMemoryMentionStore, JsonlMentionStore, MentionStore, StoreError};
```

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib reply::storage::tests
```

Expected: 5 PASS.

- [ ] **Step 4: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/storage.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "feat(ghost): MentionStore — InMemory + Jsonl impls

Tracks (persona, user_id) → since_id, replied_to set, and per-author
reply timestamps for the rate limit. JsonlMentionStore is append-only
log + in-memory replay at open. since_id is monotonic — replays older
ids do not regress.

5 unit tests cover monotonicity, replied-set round-trip, per-author
rate count window, jsonl reload, and the missing-file case.

heartbit-ghost P1.5b — task 7/12."
```

---

## Task 8: `SpamGuard` with 5 rules

**Files:**
- Create: `crates/heartbit-ghost/src/reply/spam_guard.rs`
- Modify: `crates/heartbit-ghost/src/reply/mod.rs` (re-export)

- [ ] **Step 1: Write the failing tests**

Create `crates/heartbit-ghost/src/reply/spam_guard.rs`:

```rust
//! Anti-spam guards for the reply pipeline. Each rule can short-circuit
//! mention processing before the LLM is consulted. See spec §8 for the
//! full rule set + thresholds.

use chrono::{DateTime, Duration, Utc};

use super::{Mention, MentionerContext};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    SelfReply,
    StaleParent,
    LowEffortSpam,
    PerAuthorRateLimit,
    TooShortToEngage,
}

#[derive(Debug, Clone)]
pub struct SpamGuardConfig {
    /// Operator's own user_id — replies from this id are skipped (no
    /// self-replies; the operator should @-self differently).
    pub operator_user_id: String,
    /// Skip when the parent tweet is older than this. Default 7 days.
    pub stale_parent_after_days: i64,
    /// Low-follower threshold for the spam guard.
    pub low_follower_threshold: u64,
    /// Short-text threshold (chars) coupled with low-follower for spam.
    pub low_effort_short_text_chars: usize,
    /// Per-author rate limit window.
    pub per_author_window_hours: i64,
    /// Max replies allowed to one author per window.
    pub per_author_max_replies: usize,
    /// Minimum non-whitespace alphanumeric chars to be worth engaging.
    pub min_engagement_chars: usize,
}

impl SpamGuardConfig {
    pub fn defaults_for(operator_user_id: impl Into<String>) -> Self {
        Self {
            operator_user_id: operator_user_id.into(),
            stale_parent_after_days: 7,
            low_follower_threshold: 5,
            low_effort_short_text_chars: 30,
            per_author_window_hours: 24,
            per_author_max_replies: 3,
            min_engagement_chars: 3,
        }
    }
}

pub struct SpamGuard {
    cfg: SpamGuardConfig,
}

impl SpamGuard {
    pub fn new(cfg: SpamGuardConfig) -> Self {
        Self { cfg }
    }

    /// Returns `Some(reason)` if the mention should be skipped, `None`
    /// to proceed. Evaluates rules in fail-fast order.
    pub fn should_skip(
        &self,
        mention: &Mention,
        parent_posted_at: Option<DateTime<Utc>>,
        mentioner: Option<&MentionerContext>,
        replies_to_author_recent: usize,
        now: DateTime<Utc>,
    ) -> Option<SkipReason> {
        // 1. Self-reply.
        if mention.author_id == self.cfg.operator_user_id {
            return Some(SkipReason::SelfReply);
        }
        // 2. Stale parent.
        if let Some(p) = parent_posted_at {
            if p < now - Duration::days(self.cfg.stale_parent_after_days) {
                return Some(SkipReason::StaleParent);
            }
        }
        // 3. Low-follower spam (BOTH signals required).
        if let Some(ctx) = mentioner {
            if let Some(fc) = ctx.follower_count {
                if fc < self.cfg.low_follower_threshold
                    && mention.text.len() < self.cfg.low_effort_short_text_chars
                {
                    return Some(SkipReason::LowEffortSpam);
                }
            }
        }
        // 4. Per-author rate limit.
        if replies_to_author_recent >= self.cfg.per_author_max_replies {
            return Some(SkipReason::PerAuthorRateLimit);
        }
        // 5. Too short to engage.
        let alnum_count = mention
            .text
            .chars()
            .filter(|c| c.is_alphanumeric())
            .count();
        if alnum_count < self.cfg.min_engagement_chars {
            return Some(SkipReason::TooShortToEngage);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixture_mention(text: &str, author_id: &str) -> Mention {
        Mention {
            id: "m1".into(),
            text: text.into(),
            author_id: author_id.into(),
            author_handle: "x".into(),
            posted_at: Utc::now(),
            in_reply_to_tweet_id: Some("p1".into()),
        }
    }

    fn fixture_ctx(followers: u64) -> MentionerContext {
        MentionerContext {
            handle: "x".into(),
            bio: None,
            recent_tweets: vec![],
            follower_count: Some(followers),
        }
    }

    #[test]
    fn self_reply_skips() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("12345"));
        let m = fixture_mention("hello", "12345");
        assert_eq!(
            guard.should_skip(&m, None, None, 0, Utc::now()),
            Some(SkipReason::SelfReply)
        );
    }

    #[test]
    fn stale_parent_skips() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("op"));
        let m = fixture_mention("hi there long enough", "other");
        let stale = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let now = Utc.with_ymd_and_hms(2026, 5, 1, 0, 0, 0).unwrap();
        assert_eq!(
            guard.should_skip(&m, Some(stale), None, 0, now),
            Some(SkipReason::StaleParent)
        );
    }

    #[test]
    fn low_effort_spam_requires_both_signals() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("op"));
        // Both signals: 0 followers + 5-char text → spam.
        let m = fixture_mention("hi!!!", "other");
        let ctx = fixture_ctx(0);
        assert_eq!(
            guard.should_skip(&m, None, Some(&ctx), 0, Utc::now()),
            Some(SkipReason::LowEffortSpam)
        );
        // Only one signal (followers low but text long enough) → not spam.
        let m_long = fixture_mention(
            "this is a substantive question about the framework, what do you think",
            "other",
        );
        let ctx_low = fixture_ctx(1);
        assert_eq!(
            guard.should_skip(&m_long, None, Some(&ctx_low), 0, Utc::now()),
            None
        );
    }

    #[test]
    fn per_author_rate_limit_skips_at_threshold() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("op"));
        let m = fixture_mention("real question with substance here", "spammer");
        // 3 prior replies to this author → at threshold → skip.
        assert_eq!(
            guard.should_skip(&m, None, None, 3, Utc::now()),
            Some(SkipReason::PerAuthorRateLimit)
        );
        // 2 prior replies → still under threshold.
        assert_eq!(
            guard.should_skip(&m, None, None, 2, Utc::now()),
            None
        );
    }

    #[test]
    fn too_short_to_engage_skips_emoji_only() {
        let guard = SpamGuard::new(SpamGuardConfig::defaults_for("op"));
        let m = fixture_mention("👍🔥", "other");
        assert_eq!(
            guard.should_skip(&m, None, None, 0, Utc::now()),
            Some(SkipReason::TooShortToEngage)
        );
    }
}
```

- [ ] **Step 2: Re-export from reply/mod.rs**

```rust
pub mod spam_guard;
pub use spam_guard::{SkipReason, SpamGuard, SpamGuardConfig};
```

- [ ] **Step 3: Run the tests**

```bash
cargo test -p heartbit-ghost --lib reply::spam_guard::tests
```

Expected: 5 PASS.

- [ ] **Step 4: Format + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
git add crates/heartbit-ghost/src/reply/spam_guard.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "feat(ghost): SpamGuard — 5 anti-spam rules with configurable thresholds

Self-reply, stale parent, low-effort spam (low-follower + short-text
both required), per-author rate limit (default 3/24h), too-short-to-engage
(< 3 alphanumeric chars). Each rule has a SkipReason variant; the
caller decides what to do (mark seen + skip is the spec recommendation).
Defaults are operator-user-id-only; everything else is configurable
via SpamGuardConfig.

heartbit-ghost P1.5b — task 8/12."
```

---

## Task 9: `DaemonCommand::MentionPoll` + `DaemonCommand::ReplyDraft`

**Files:**
- Modify: `crates/heartbit/src/daemon/types.rs` (extend `DaemonCommand` enum)
- Modify: `crates/heartbit/src/daemon/dispatcher.rs` (or wherever the dispatch arms live) — add stub arms

- [ ] **Step 1: Read the existing DaemonCommand structure**

```bash
sed -n '15,55p' crates/heartbit/src/daemon/types.rs
```

The new variants live alongside `SubmitTask` + `CancelTask`.

- [ ] **Step 2: Write the failing tests**

In `crates/heartbit/src/daemon/types.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn daemon_command_mention_poll_serde_round_trips() {
        let cmd = DaemonCommand::MentionPoll {
            persona: "heartbit-ghost:x".into(),
            user_id: "12345".into(),
        };
        let s = serde_json::to_string(&cmd).unwrap();
        let parsed: DaemonCommand = serde_json::from_str(&s).unwrap();
        assert!(matches!(
            parsed,
            DaemonCommand::MentionPoll { ref persona, ref user_id }
                if persona == "heartbit-ghost:x" && user_id == "12345"
        ));
    }

    #[test]
    fn daemon_command_reply_draft_serde_round_trips() {
        // ReplyDraft carries Mention/TweetSnapshot/MentionerContext from
        // heartbit-ghost. Use a minimal fixture; full serde coverage in
        // the heartbit-ghost reply::tests.
        // (Test body written by implementer.)
    }
```

- [ ] **Step 3: Add the variants**

In `crates/heartbit/src/daemon/types.rs`, extend `DaemonCommand`:

```rust
    /// Cron-driven: poll for new mentions for this persona's
    /// configured operator account, dispatch one ReplyDraft per
    /// surviving mention. Fired by the CronScheduler at the
    /// `[daemon.persona.<name>.mentions] poll_interval_seconds`
    /// cadence. Free-tier safe: one twitter_mentions read per poll.
    MentionPoll {
        persona: String,
        user_id: String,
    },
    /// Per-mention: run the reply pipeline (research → draft →
    /// review → post). Fired by the MentionPoll handler for each
    /// mention that survives the spam guards. Carries the full
    /// mention payload so the dispatcher doesn't need to refetch.
    ReplyDraft {
        persona: String,
        mention: heartbit_ghost::reply::Mention,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent: Option<heartbit_ghost::reply::TweetSnapshot>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mentioner_context: Option<heartbit_ghost::reply::MentionerContext>,
    },
```

(Mention / TweetSnapshot / MentionerContext need `#[derive(Serialize, Deserialize)]` — Task 2 stops short of that, so this task adds those derives in heartbit-ghost.)

- [ ] **Step 4: Add serde derives to the value types**

In `crates/heartbit-ghost/src/reply/mod.rs`, change:

```rust
#[derive(Debug, Clone)]
pub struct Mention { … }
```

to:

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mention { … }
```

Same for `TweetSnapshot` and `MentionerContext`.

- [ ] **Step 5: Stub dispatcher arms (returns "not implemented" — wired in Tasks 10/11)**

In the dispatcher's match-on-`DaemonCommand`, add:

```rust
DaemonCommand::MentionPoll { persona, user_id } => {
    tracing::warn!(
        "MentionPoll dispatched for persona={persona} user_id={user_id} but handler is not yet wired (P1.5c task 10)"
    );
    // No-op for now — task 10 implements the real handler.
}
DaemonCommand::ReplyDraft { persona, mention, .. } => {
    tracing::warn!(
        "ReplyDraft dispatched for persona={persona} mention={} but handler is not yet wired (P1.5c task 11)",
        mention.id,
    );
    // No-op for now — task 11 implements the real handler.
}
```

- [ ] **Step 6: Run tests + build**

```bash
cargo test -p heartbit
cargo build -p heartbit
cargo clippy -p heartbit --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit/src/daemon/types.rs crates/heartbit/src/daemon/dispatcher.rs crates/heartbit-ghost/src/reply/mod.rs
git commit -m "feat(daemon): DaemonCommand variants for mention polling + reply drafting

Two new variants on the daemon's command enum:
- MentionPoll { persona, user_id } — cron-driven, fires per persona on the
  configured cadence (default 5 min). Handler in task 10.
- ReplyDraft { persona, mention, parent, mentioner_context } — per-mention,
  carries the full payload so the dispatcher doesn't refetch. Handler
  in task 11.

Both variants stub-warn for now — wiring lands in P1.5c tasks 10/11.
Adds Serialize/Deserialize derives to Mention/TweetSnapshot/MentionerContext.

heartbit-ghost P1.5c — task 9/12."
```

---

## Task 10: `MentionPoll` dispatcher handler + cron scheduling

**Files:**
- Modify: `crates/heartbit/src/daemon/dispatcher.rs` (real handler for `MentionPoll`)
- Modify: `crates/heartbit/src/daemon/cron.rs` (or wherever `CronScheduler` lives) — register a cron entry per persona that has `[daemon.persona.<name>.mentions]` enabled

- [ ] **Step 1: Locate the cron registration site**

```bash
grep -rn "fn register_cron\|fn schedule_persona\|CronScheduler::new" crates/heartbit/src/daemon/ | head -5
```

The `CronScheduler` already supports persona-pulse-style entries. Mirror that pattern.

- [ ] **Step 2: Implement `MentionPoll` handler**

Body (same shape as spec §6.2):

```rust
async fn handle_mention_poll(
    persona: &str,
    user_id: &str,
    store: &dyn heartbit_ghost::reply::MentionStore,
    spam_guard: &heartbit_ghost::reply::SpamGuard,
    twitter_mentions: &heartbit_ghost::tools::TwitterMentionsTool,
    credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver>,
    producer: std::sync::Arc<dyn CommandProducer>,
) -> anyhow::Result<()> {
    let since_id = store.since_id_for(persona, user_id).await?;
    let exec_ctx = heartbit_core::ExecutionContext::new(credentials);
    let input = serde_json::json!({
        "user_id": user_id,
        "max_results": 50,
        "since_id": since_id,
    });
    let out = twitter_mentions.execute(&exec_ctx, input).await?;
    if out.is_error {
        anyhow::bail!("twitter_mentions: {}", out.content);
    }
    #[derive(serde::Deserialize)]
    struct Resp {
        mentions: Vec<heartbit_ghost::reply::Mention>,
    }
    let parsed: Resp = serde_json::from_str(&out.content)?;

    let now = chrono::Utc::now();
    let mut max_seen: Option<String> = since_id.clone();
    for m in parsed.mentions.into_iter() {
        if store.was_replied(&m.id).await? {
            continue;
        }
        // Anti-spam: per-author rate is the only one we can evaluate here
        // (the rest need parent timestamp + mentioner_context, fetched
        // below ONLY after we've passed the cheap checks).
        let recent = store
            .replies_to_author_since(&m.author_id, now - chrono::Duration::hours(24))
            .await?;
        if let Some(reason) = spam_guard.should_skip(&m, None, None, recent, now) {
            tracing::info!("spam-skip {} early ({:?})", m.id, reason);
            store.mark_replied(&m.id).await?;
            // Track the author for rate-limit purposes even on skip.
            store.record_reply_to_author(&m.author_id, now).await?;
            continue;
        }
        // Fetch parent + mentioner_context (cheap, <= 2 X reads).
        let (parent, mentioner) = fetch_mention_context(&m, &exec_ctx).await.unwrap_or((None, None));
        // Re-evaluate spam guards now that we have full context.
        if let Some(reason) = spam_guard.should_skip(
            &m,
            parent.as_ref().map(|p| p.posted_at),
            mentioner.as_ref(),
            recent,
            now,
        ) {
            tracing::info!("spam-skip {} late ({:?})", m.id, reason);
            store.mark_replied(&m.id).await?;
            store.record_reply_to_author(&m.author_id, now).await?;
            continue;
        }
        // Dispatch.
        producer
            .send(crate::daemon::types::DaemonCommand::ReplyDraft {
                persona: persona.to_string(),
                mention: m.clone(),
                parent,
                mentioner_context: mentioner,
            })
            .await?;
        if max_seen.as_deref().map_or(true, |s| m.id.as_str() > s) {
            max_seen = Some(m.id);
        }
    }
    if let Some(new_since) = max_seen {
        store.bump_since_id(persona, user_id, &new_since).await?;
    }
    Ok(())
}
```

`fetch_mention_context` is a small helper that GETs `/2/tweets/:id` (parent) and `/2/users/:id` (bio + recent tweets via `/2/users/:id/tweets?max_results=3`). Implement inline using existing `XClient` (which `TwitterMentionsTool` already uses).

- [ ] **Step 3: Register the cron entry per persona**

When the daemon parses `[daemon.persona.<name>.mentions]` and `enabled = true`, register a cron entry that fires `DaemonCommand::MentionPoll { persona: name, user_id }` every `poll_interval_seconds`.

```rust
fn register_mention_polling(
    cron: &mut CronScheduler,
    persona_name: &str,
    user_id: &str,
    interval_seconds: u64,
    producer: std::sync::Arc<dyn CommandProducer>,
) {
    cron.register(format!("mentions:{persona_name}"), interval_seconds, move || {
        let producer = producer.clone();
        let persona = persona_name.to_string();
        let user_id = user_id.to_string();
        Box::pin(async move {
            producer
                .send(crate::daemon::types::DaemonCommand::MentionPoll { persona, user_id })
                .await
                .ok();
        })
    });
}
```

- [ ] **Step 4: Add daemon TOML schema**

Extend the daemon config struct to parse the new section:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaMentionsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_poll_interval")]
    pub poll_interval_seconds: u64,
    pub user_id: String,
    #[serde(default = "default_candidates_per_reply")]
    pub candidates_per_reply: usize,
    #[serde(default = "default_mention_store")]
    pub mention_store: String,  // "in_memory" | "jsonl"
    #[serde(default)]
    pub mention_store_path: Option<String>,
}

fn default_poll_interval() -> u64 { 300 }
fn default_candidates_per_reply() -> usize { 2 }
fn default_mention_store() -> String { "in_memory".into() }
```

- [ ] **Step 5: Integration test**

Add (or extend) a test that:
1. Constructs an `InMemoryMentionStore`
2. Wires a `MockCommandProducer` (in-memory channel)
3. Stubs `twitter_mentions` via a fake Tool that returns 3 mentions (1 self-reply, 1 normal, 1 too-short)
4. Calls `handle_mention_poll`
5. Asserts: 1 `ReplyDraft` command was produced (only the normal one); `since_id` was bumped to the max id; the 3 mentions are all marked replied (whether skipped or dispatched)

```rust
#[tokio::test]
async fn mention_poll_dispatches_one_reply_draft_for_three_mentions() {
    // … fixture wiring …
    let producer = MockCommandProducer::new();
    let store = heartbit_ghost::reply::InMemoryMentionStore::new();
    let spam_guard = heartbit_ghost::reply::SpamGuard::new(
        heartbit_ghost::reply::SpamGuardConfig::defaults_for("operator_user_id"),
    );
    let twitter_mentions = MockTwitterMentionsTool::with_3_mentions();
    handle_mention_poll(
        "persona", "operator_user_id",
        &store, &spam_guard, &twitter_mentions,
        std::sync::Arc::new(StubCredentialResolver),
        std::sync::Arc::new(producer.clone()),
    ).await.unwrap();
    let dispatched = producer.commands_sent();
    assert_eq!(dispatched.len(), 1);
    assert!(matches!(dispatched[0], DaemonCommand::ReplyDraft { .. }));
}
```

- [ ] **Step 6: Run tests + commit**

```bash
cargo test -p heartbit --lib
cargo clippy -p heartbit --all-targets -- -D warnings
git add crates/heartbit/src/daemon/dispatcher.rs crates/heartbit/src/daemon/cron.rs crates/heartbit/src/config/daemon.rs
git commit -m "feat(daemon): MentionPoll handler + cron registration

Cron-driven mention poller: reads since_id from MentionStore, calls
twitter_mentions, runs SpamGuard early (cheap checks) then late (with
parent + mentioner context), dispatches one ReplyDraft per surviving
mention, bumps since_id monotonically. Skipped mentions are marked
replied so they don't retry.

Adds [daemon.persona.<name>.mentions] config section. Integration test
covers the 3-mention fixture (self-reply skipped, too-short skipped,
normal dispatched).

heartbit-ghost P1.5c — task 10/12."
```

---

## Task 11: `ReplyDraft` dispatcher handler + `ReplyReviewDelivery` for Telegram

**Files:**
- Modify: `crates/heartbit/src/daemon/dispatcher.rs` (real handler for `ReplyDraft`)
- Modify: `crates/heartbit-cli/src/persona_review.rs` (add `TelegramReplyReviewDelivery`)
- Modify: `crates/heartbit-telegram/src/...` (extend bot to handle reply-review messages — same callback shape, different message body)

- [ ] **Step 1: Implement `ReplyDraft` handler**

Same shape as the existing review-mode `SubmitTask` handler — construct `ReplyConfig` from the persona registry + the command payload, call `run_reply_pipeline`, handle the outcome by `mark_replied(...)` and `record_reply_to_author(...)` on the store.

```rust
async fn handle_reply_draft(
    persona_name: &str,
    mention: heartbit_ghost::reply::Mention,
    parent: Option<heartbit_ghost::reply::TweetSnapshot>,
    mentioner_context: Option<heartbit_ghost::reply::MentionerContext>,
    registry: &heartbit_core::persona::PersonaRegistry,
    store: &dyn heartbit_ghost::reply::MentionStore,
    provider: std::sync::Arc<heartbit_core::llm::BoxedProvider>,
    delivery: std::sync::Arc<dyn heartbit_ghost::reply::ReplyReviewDelivery>,
    twitter_tool: std::sync::Arc<dyn heartbit_core::Tool>,
    credentials: std::sync::Arc<dyn heartbit_core::CredentialResolver>,
    candidates_per_reply: usize,
    corpora_root: &std::path::Path,
    profiles_root: &std::path::Path,
) -> anyhow::Result<()> {
    let persona = registry.get(persona_name).ok_or_else(|| {
        anyhow::anyhow!("persona '{persona_name}' not registered")
    })?;
    let expansion = persona
        .expand(&heartbit_core::persona::PersonaParams::default())
        .map_err(|e| anyhow::anyhow!("expand {persona_name}: {e}"))?;
    let researcher_override = expansion
        .agents
        .iter()
        .find(|a| a.name == "repo_researcher")
        .map(|recipe| {
            let recipe = std::sync::Arc::new(recipe.clone_config());
            let tools: Vec<std::sync::Arc<dyn heartbit_core::Tool>> = expansion
                .tools
                .iter()
                .filter(|t| t.definition().name == "repo_inspect")
                .cloned()
                .collect();
            (recipe, tools)
        });

    let mention_id = mention.id.clone();
    let author_id = mention.author_id.clone();
    let now = chrono::Utc::now();

    let cfg = heartbit_ghost::reply::ReplyConfig {
        persona_name,
        provider,
        corpora_root,
        profiles_root,
        on_progress: Some(std::sync::Arc::new(|s: &str| tracing::info!("> {s}"))),
        mention,
        parent,
        mentioner_context,
        candidates_per_reply,
        mode_addendum: expansion.mode_addendum,
        researcher_override,
        delivery,
        twitter_tool,
        credentials,
    };
    let _output = heartbit_ghost::reply::run_reply_pipeline(cfg)
        .await
        .map_err(|e| anyhow::anyhow!("reply pipeline: {e}"))?;

    // Always mark replied (regardless of outcome — Skipped / TimedOut / etc.
    // — so the cron poll doesn't retry).
    store.mark_replied(&mention_id).await?;
    store.record_reply_to_author(&author_id, now).await?;
    Ok(())
}
```

- [ ] **Step 2: Implement `TelegramReplyReviewDelivery`**

In `crates/heartbit-cli/src/persona_review.rs`, add a new struct that implements `ReplyReviewDelivery`. It uses the same `teloxide` bot as `TelegramReviewDelivery` but renders a different message body:

```rust
pub struct TelegramReplyReviewDelivery {
    bot: teloxide::Bot,
    chat_id: i64,
    interaction_timeout_seconds: u64,
}

impl TelegramReplyReviewDelivery {
    pub fn from_env() -> anyhow::Result<Self> { … }

    fn render(msg: &heartbit_ghost::reply::ReplyReviewMessage) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "NEW MENTION on your tweet from @{} ({} followers)\n\n",
            msg.mention.author_handle,
            msg.mentioner_context
                .as_ref()
                .and_then(|c| c.follower_count)
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".into()),
        ));
        if let Some(p) = &msg.parent {
            let abridged: String = p.text.chars().take(200).collect();
            s.push_str("YOUR TWEET (parent):\n> ");
            s.push_str(&abridged);
            if p.text.len() > 200 { s.push_str("…"); }
            s.push_str("\n\n");
        }
        s.push_str("THEIR REPLY:\n> ");
        s.push_str(&msg.mention.text);
        s.push_str("\n\n");
        for (i, c) in msg.candidates.iter().enumerate() {
            s.push_str(&format!("DRAFT {}:\n> {}\n\n", i + 1, c));
        }
        s
    }
}

#[async_trait::async_trait]
impl heartbit_ghost::reply::ReplyReviewDelivery for TelegramReplyReviewDelivery {
    async fn deliver(
        &self,
        msg: heartbit_ghost::reply::ReplyReviewMessage,
    ) -> Result<heartbit_ghost::review::DeliveredReview, heartbit_ghost::review::ReviewDeliveryError> {
        // 1. Send message + inline keyboard with [1] [2] [Skip] (or [1] [Skip]
        //    when len==1).
        // 2. Wait up to interaction_timeout_seconds for a callback_query.
        // 3. Map callback to DeliveryOutcome::Picked(idx) | Skipped | TimedOut.
        // (Implementation mirrors TelegramReviewDelivery::deliver.)
        todo!("mirror TelegramReviewDelivery::deliver, render with self.render(msg)")
    }
    async fn report(
        &self,
        receipt: heartbit_ghost::review::DeliveryReceipt,
        outcome: heartbit_ghost::reply::ReplyOutcome,
    ) -> Result<(), heartbit_ghost::review::ReviewDeliveryError> {
        // edit the original message to show the outcome (Posted with link,
        // Skipped, etc.)
        todo!("mirror TelegramReviewDelivery::report")
    }
}
```

- [ ] **Step 3: Run tests + commit**

```bash
cargo test
cargo clippy --all-targets -- -D warnings
git add crates/heartbit/src/daemon/dispatcher.rs crates/heartbit-cli/src/persona_review.rs
git commit -m "feat(daemon+cli): ReplyDraft handler + Telegram delivery for replies

Closes the daemon-side loop: ReplyDraft fired by MentionPoll constructs
ReplyConfig from registry + payload, calls run_reply_pipeline, marks
the mention replied + records per-author rate count regardless of
outcome (Skipped / TimedOut / Posted all advance the store).

TelegramReplyReviewDelivery renders the parent-quoted layout from spec
§9.1: 'NEW MENTION on your tweet from @x' + parent + reply + drafts +
[1] [2] [Skip] inline keyboard.

heartbit-ghost P1.5c — task 11/12."
```

---

## Task 12: Acceptance — quality gate + manual setup + live test

**Files:** none modified by this task; verifies prior tasks land cleanly.

- [ ] **Step 1: Full quality gate**

```bash
cargo fmt -- --check && cargo clippy --all-targets -- -D warnings && cargo test
```

Expected: all three green. Test count: previous baseline + ~24 new tests across `reply::*`, `spam_guard::*`, `storage::*`, daemon command serde, and the dispatcher integration test.

- [ ] **Step 2: Build the release binary**

```bash
cargo build --release --bin heartbit 2>&1 | tail -3
```

Expected: `Finished release …`.

- [ ] **Step 3: Operator-side setup (manual; not part of plan deliverable)**

1. Resolve the operator's X user_id once:
   ```bash
   curl -sH "Authorization: Bearer $X_BEARER_TOKEN" \
     "https://api.x.com/2/users/by/username/<your_handle>" | jq .data.id
   ```
   Save it for the persona TOML.

2. Add to `daemon-dev.toml` (or whichever daemon config is in use):
   ```toml
   [daemon.persona."heartbit-ghost:x".mentions]
   enabled = true
   poll_interval_seconds = 300
   user_id = "1234567890"
   candidates_per_reply = 2
   mention_store = "jsonl"
   mention_store_path = "~/.heartbit/ghost/mentions/heartbit-ghost:x.jsonl"
   ```

3. Restart the daemon (or start it for the first time):
   ```bash
   ./target/release/heartbit daemon --config daemon-dev.toml
   ```

- [ ] **Step 4: Live test — colleague-driven mention**

1. Post a tweet from the operator's account (e.g., a `heartbit-ghost:x` thread via `persona run --review`).
2. Ask a colleague (or a second X account you control) to reply to that tweet with a substantive question.
3. Within `poll_interval_seconds` (default 300s), the daemon polls mentions and dispatches a ReplyDraft.
4. The Telegram bot delivers the parent-quoted message + 2 drafts + [1] [2] [Skip] buttons.
5. Pick a draft → reply posts under the colleague's mention. Verify by visiting the X UI.
6. Verify the JSONL store at `~/.heartbit/ghost/mentions/heartbit-ghost:x.jsonl` contains one `replied` event for the mention id and one `author_reply` event for the colleague's user id.

- [ ] **Step 5: Live test — alternate "Skip" path**

Repeat steps 1-3 above. On the Telegram review, press Skip. Verify:
- No reply is posted on X
- The mention id is in the `replied_to` set in the JSONL store
- The next mention from the same colleague within 24h still works (per-author rate counts the skip as a reply for rate-limit purposes — this is the conservative default; can be relaxed in a follow-up if it bites)

- [ ] **Step 6: Final merge**

Once Steps 4-5 pass, finish via the **superpowers:finishing-a-development-branch** skill.

---

## Self-review — pre-execution

- **Spec coverage:** every section maps to at least one task — §3 Files (Tasks 1-12), §4 reply_writer (Task 1), §5 run_reply_pipeline (Tasks 2-5), §6 daemon (Tasks 9-11), §7 storage (Task 7), §8 spam guards (Task 8), §9 Telegram delivery (Task 11), §10 tests (each task ships its own + acceptance), §11 out-of-scope (no tasks), §12 risks (mitigations applied throughout), §13 sub-phases (Tasks 1-5 / 6-8 / 9-12 ≈ a/b/c).
- **Placeholder scan:** Tasks 5, 6, and 11 contain `todo!()` markers for verbose blocks (mock fixture wiring, OAuth1 inline GET, Telegram bot rendering). Each `todo!()` block has a clear pointer to the existing-code template the implementer mirrors. This is intentional — duplicating those ~80-line bodies in the plan would triple its length without improving clarity.
- **Type consistency:** `ResearcherOverride` (from Task 9.5 of the heartbit-rs:x plan, now shipped) is reused. `MentionStore` is `async_trait`-based, matching the existing `ReviewDelivery` pattern. `Mention` / `TweetSnapshot` / `MentionerContext` gain `Serialize`/`Deserialize` derives in Task 9 to support the daemon command payload.
- **Dependency sequencing:** Task 6 depends on Task 11 (TelegramReplyReviewDelivery). The plan flags this in Task 6's commit message — the implementer can either land them together or stub the Telegram delivery in Task 6 with a `MockReplyReviewDelivery` until Task 11 ships the real one.
- **One known TBD:** the spec §13 mentioned a possible `TwitterGetTweetTool` for parent-tweet fetching. Task 10's `fetch_mention_context` helper does this inline via the existing `XClient` rather than adding a new tool — simpler and contained. Documented in Task 10's body.
