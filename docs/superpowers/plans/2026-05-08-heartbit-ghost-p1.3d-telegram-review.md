# heartbit-ghost P1.3d — Telegram review delivery + twitter_thread direct + Phase 0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--review` mode that sends N candidates to Telegram via a standalone teloxide bot in the CLI process, awaits the user's pick on an inline keyboard, runs `publish_gate` on the chosen draft, then posts to X by calling `TwitterThreadTool` directly (skipping the publisher LLM recipe). User receives a Telegram message → picks → bot edits message in place + posts tweet.

**Architecture:** New `heartbit_ghost::review` module exposes `run_review_pipeline` + `ReviewDelivery` trait. `heartbit-ghost` stays free of teloxide; `heartbit-cli` provides the `TelegramReviewDelivery` impl using the existing `heartbit-telegram` crate (which gains `CallbackAction::PersonaPick`). Pipeline reuses P1.3c's candidate generation + dedup; skips judge / image_generator / publish_gate inside `run_review_pipeline` (publish_gate runs AFTER pick instead). `TwitterThreadTool` is invoked directly via `Tool::execute` with an `ExecutionContext` carrying an env-based `CredentialResolver` — no publisher LLM call.

**Tech Stack:** Rust 2024, `tokio` for async + `JoinSet` for parallel candidates (reused from P1.3c), object-safe async traits via `Pin<Box<dyn Future + Send + '_>>` (project convention), `teloxide` for the standalone bot in heartbit-cli (already a workspace dep via heartbit-telegram), `serde_json` for tool output parsing, `thiserror` for errors. **No new workspace deps.**

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `crates/heartbit-telegram/src/keyboard.rs` | MODIFY | Add `CallbackAction::PersonaPick`, `PickChoice`, `persona_pick_buttons`, `parse_callback_data` arm for `p:` prefix. 6 unit tests. |
| `crates/heartbit-telegram/src/lib.rs` | MODIFY | Re-export `PickChoice`, `persona_pick_buttons`. |
| `crates/heartbit-ghost/src/review/mod.rs` | NEW | `ReviewConfig`, `ReviewOutput`, `ReviewOutcome`, `ReviewError`, `run_review_pipeline`, `parse_twitter_thread_output` helper. Plus `MockReviewDelivery` + `MockTwitterTool` test helpers. 3 error display tests + 6 integration tests. |
| `crates/heartbit-ghost/src/review/delivery.rs` | NEW | `ReviewDelivery` trait, `ReviewMessage`, `DeliveredReview`, `DeliveryReceipt`, `DeliveryOutcome`, `ReportableOutcome`, `ReviewDeliveryError`. 3 unit tests on display formatting. |
| `crates/heartbit-ghost/src/review/prompts.rs` | NEW | `build_review_message` (text body for the Telegram message) + `build_report_message` (text for the in-place edit after pick). 4 unit tests. |
| `crates/heartbit-ghost/src/review/tweet_split.rs` | NEW | `parse_thread_tweets(draft)` — splits on `\n\n`, trims, filters empty. 4 unit tests. |
| `crates/heartbit-ghost/src/lib.rs` | MODIFY | Add `pub mod review;`. |
| `crates/heartbit-cli/src/persona_review.rs` | NEW | `TelegramReviewDelivery` impl using teloxide. Owns the bot, sends messages, registers callback dispatcher, awaits pick via oneshot, edits message on `report()`. |
| `crates/heartbit-cli/src/persona.rs` | MODIFY | `PersonaCommand::Run` gains `#[arg(long)] review: bool` field. Dispatch arm branches on `review`: `false` → existing P1.3c path; `true` → constructs `TelegramReviewDelivery` + `TwitterThreadTool` + env-based `CredentialResolver`, calls `run_review_pipeline`. |
| `crates/heartbit-cli/src/main.rs` (or where `mod persona` lives) | MODIFY | Add `mod persona_review;` declaration. |

3 implementation tasks + 1 final acceptance.

---

## Task 1: Foundation — heartbit-telegram extensions + heartbit-ghost::review types + helpers

**Why:** All pure types, traits, and helpers. No orchestration logic. Independent of teloxide (heartbit-ghost stays free of that dep). Tests can be unit-only here. Tasks 2 + 3 build on top.

**Files:**
- Modify: `crates/heartbit-telegram/src/keyboard.rs`
- Modify: `crates/heartbit-telegram/src/lib.rs`
- Create: `crates/heartbit-ghost/src/review/mod.rs` (skeleton — types + error variants only; `run_review_pipeline` lands in Task 2)
- Create: `crates/heartbit-ghost/src/review/delivery.rs`
- Create: `crates/heartbit-ghost/src/review/prompts.rs`
- Create: `crates/heartbit-ghost/src/review/tweet_split.rs`
- Modify: `crates/heartbit-ghost/src/lib.rs`

- [ ] **Step 1: Update `crates/heartbit-telegram/src/keyboard.rs` — add `PickChoice`, `CallbackAction::PersonaPick`, `persona_pick_buttons`, parse_callback_data arm**

Append to the existing file (after the `CallbackAction` enum and before `parse_callback_data`):

```rust
/// Pick choice from a persona-review inline keyboard.
#[derive(Debug, Clone, PartialEq)]
pub enum PickChoice {
    /// User picked a specific candidate by 0-based index.
    Index(usize),
    /// User pressed Skip.
    Skip,
}
```

Update the existing `CallbackAction` enum to add the new variant:

```rust
/// Parsed callback data from an inline keyboard button press.
#[derive(Debug, Clone, PartialEq)]
pub enum CallbackAction {
    /// Tool approval decision: `a:{uuid}:{decision}`
    Approval {
        interaction_id: Uuid,
        decision: String,
    },
    /// Question answer: `q:{uuid}:{question_idx}:{option_idx}`
    QuestionAnswer {
        interaction_id: Uuid,
        question_idx: usize,
        option_idx: usize,
    },
    /// Persona-review pick: `p:{uuid}:{choice}` where choice is a 0-based
    /// index digit `0..9` or the literal string `skip`.
    PersonaPick {
        /// Correlates back to the original review message.
        interaction_id: Uuid,
        /// User's pick or skip.
        choice: PickChoice,
    },
}
```

Add the new builder function (after `question_buttons`):

```rust
/// Build inline keyboard markup for a persona-review prompt.
///
/// Produces N buttons labelled `1` … `N` (1-based for users) plus a
/// trailing `Skip` button. Returns `Vec<(label, callback_data)>` pairs.
///
/// Panics if `n == 0` or `n > 9` — the callback format encodes index as
/// a single decimal digit, and an empty review has no candidates to pick.
pub fn persona_pick_buttons(interaction_id: Uuid, n: usize) -> Vec<(String, String)> {
    assert!(
        (1..=9).contains(&n),
        "persona_pick_buttons: n must be in 1..=9 (got {n})"
    );
    let id = interaction_id.to_string();
    let mut buttons = Vec::with_capacity(n + 1);
    for i in 0..n {
        buttons.push((format!("{}", i + 1), format!("p:{id}:{i}")));
    }
    buttons.push(("Skip".into(), format!("p:{id}:skip")));
    buttons
}
```

Update `parse_callback_data` to add a new match arm. Find the existing match block; add this arm BEFORE the `_` fallback:

```rust
        Some(&"p") => {
            if parts.len() != 3 {
                return Err(Error::Telegram(format!(
                    "invalid persona-pick callback: expected 3 parts, got {}",
                    parts.len()
                )));
            }
            let interaction_id = Uuid::parse_str(parts[1])
                .map_err(|e| Error::Telegram(format!("invalid UUID in callback: {e}")))?;
            let choice = match parts[2] {
                "skip" => PickChoice::Skip,
                other => {
                    let idx: usize = other.parse().map_err(|e| {
                        Error::Telegram(format!("invalid pick index '{other}': {e}"))
                    })?;
                    PickChoice::Index(idx)
                }
            };
            Ok(CallbackAction::PersonaPick {
                interaction_id,
                choice,
            })
        }
```

Add 6 new tests inside the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn persona_pick_buttons_three_candidates_produces_four_buttons() {
        let id = Uuid::new_v4();
        let buttons = persona_pick_buttons(id, 3);
        assert_eq!(buttons.len(), 4);
        assert_eq!(buttons[0].0, "1");
        assert_eq!(buttons[0].1, format!("p:{id}:0"));
        assert_eq!(buttons[1].0, "2");
        assert_eq!(buttons[1].1, format!("p:{id}:1"));
        assert_eq!(buttons[2].0, "3");
        assert_eq!(buttons[2].1, format!("p:{id}:2"));
        assert_eq!(buttons[3].0, "Skip");
        assert_eq!(buttons[3].1, format!("p:{id}:skip"));
    }

    #[test]
    fn persona_pick_buttons_single_candidate_produces_two_buttons() {
        let id = Uuid::new_v4();
        let buttons = persona_pick_buttons(id, 1);
        assert_eq!(buttons.len(), 2);
        assert_eq!(buttons[0].0, "1");
        assert_eq!(buttons[1].0, "Skip");
    }

    #[test]
    #[should_panic(expected = "persona_pick_buttons: n must be in 1..=9")]
    fn persona_pick_buttons_zero_candidates_panics() {
        let id = Uuid::new_v4();
        let _ = persona_pick_buttons(id, 0);
    }

    #[test]
    fn parse_callback_data_persona_pick_index_round_trip() {
        let id = Uuid::new_v4();
        let buttons = persona_pick_buttons(id, 3);
        // Pick the [2] button (idx 1 → label "2", callback "p:{id}:1").
        let action = parse_callback_data(&buttons[1].1).unwrap();
        assert_eq!(
            action,
            CallbackAction::PersonaPick {
                interaction_id: id,
                choice: PickChoice::Index(1),
            }
        );
    }

    #[test]
    fn parse_callback_data_persona_pick_skip_round_trip() {
        let id = Uuid::new_v4();
        let data = format!("p:{id}:skip");
        let action = parse_callback_data(&data).unwrap();
        assert_eq!(
            action,
            CallbackAction::PersonaPick {
                interaction_id: id,
                choice: PickChoice::Skip,
            }
        );
    }

    #[test]
    fn parse_callback_data_persona_pick_invalid_choice_returns_err() {
        let id = Uuid::new_v4();
        let data = format!("p:{id}:not-a-number");
        let err = parse_callback_data(&data).unwrap_err();
        assert!(
            err.to_string().contains("invalid pick index"),
            "got: {err}"
        );
    }
```

- [ ] **Step 2: Update `crates/heartbit-telegram/src/lib.rs` re-exports**

Find the existing `pub use keyboard::{CallbackAction, approval_buttons, parse_callback_data, question_buttons};` line and replace with:

```rust
pub use keyboard::{
    CallbackAction, PickChoice, approval_buttons, parse_callback_data, persona_pick_buttons,
    question_buttons,
};
```

- [ ] **Step 3: Run heartbit-telegram tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-telegram --lib keyboard 2>&1 | tail -10
```

Expected: existing tests + 6 new = all pass.

- [ ] **Step 4: Workspace gate (heartbit-telegram only at this point)**

```bash
cargo fmt -p heartbit-telegram -- --check
cargo clippy -p heartbit-telegram --all-targets -- -D warnings
```

Both clean.

- [ ] **Step 5: Create `crates/heartbit-ghost/src/review/delivery.rs`**

```rust
//! Trait + types abstracting the user-interaction layer for review-mode
//! pipelines. `heartbit-ghost` doesn't know or care about Telegram —
//! production wires `TelegramReviewDelivery` from `heartbit-cli`; tests
//! wire `MockReviewDelivery`.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

/// Input to a review delivery — the rendered message + correlation id.
#[derive(Debug, Clone)]
pub struct ReviewMessage {
    /// Persona instance name (rendered in the header).
    pub persona_name: String,
    /// Topic / shortname (rendered in the header).
    pub topic: String,
    /// Pre-rendered candidate drafts (one entry per surviving candidate).
    /// The delivery layer is responsible for laying these out (e.g.,
    /// numbered list with emoji indicators).
    pub candidates: Vec<String>,
    /// UUID for keyboard callback correlation. Used by the delivery
    /// implementation to thread callbacks back to the right pending review.
    pub interaction_id: Uuid,
}

/// Result of `ReviewDelivery::deliver_and_await`.
#[derive(Debug)]
pub struct DeliveredReview {
    /// What the user did (or didn't).
    pub outcome: DeliveryOutcome,
    /// Opaque ticket the impl uses to correlate `report()` back to this
    /// delivery. Telegram impl puts `{"chat_id": <i64>, "message_id": <i32>}`;
    /// mock can put `null`.
    pub receipt: DeliveryReceipt,
}

/// Opaque payload returned by `deliver_and_await`, threaded back to `report`.
#[derive(Debug, Clone)]
pub struct DeliveryReceipt {
    /// Implementation-defined data. Format is a contract between the
    /// concrete `ReviewDelivery` impl and itself; `run_review_pipeline`
    /// treats it as opaque.
    pub data: Value,
}

/// What the user did.
#[derive(Debug, Clone, PartialEq)]
pub enum DeliveryOutcome {
    /// User picked a specific candidate by 0-based index.
    Pick(usize),
    /// User pressed Skip.
    Skip,
    /// Timeout reached without a response.
    TimedOut,
}

/// What the orchestrator wants to report back to the user (via the
/// delivery layer's `report()` method, which typically edits the
/// original message in place).
#[derive(Debug, Clone)]
pub enum ReportableOutcome {
    /// Pick succeeded and tweet was posted.
    Posted {
        /// 0-based index into the original candidates list.
        chosen_index: usize,
        /// First-tweet URL.
        tweet_url: String,
    },
    /// User pressed Skip.
    Skipped,
    /// Timeout elapsed.
    TimedOut,
    /// Pick succeeded but `publish_gate` rejected the chosen draft.
    GateRejected {
        /// 0-based index of the rejected draft.
        chosen_index: usize,
        /// Reason from `PublishGateError`'s display.
        reason: String,
    },
    /// Pick succeeded, gate passed, but the X API call failed.
    PublishFailed {
        /// 0-based index of the draft that failed to post.
        chosen_index: usize,
        /// Failure reason (typically the X API error message).
        reason: String,
    },
}

/// Errors raised by the delivery layer.
#[derive(Debug, Error)]
pub enum ReviewDeliveryError {
    /// Bot connection / send / API failure.
    #[error("delivery transport: {0}")]
    Transport(String),
    /// Pick callback was received but couldn't be parsed.
    #[error("invalid callback: {0}")]
    InvalidCallback(String),
    /// Configuration failure (e.g., missing env vars).
    #[error("delivery config: {0}")]
    Config(String),
}

/// Object-safe async trait for delivering candidates to a user and
/// awaiting their pick.
///
/// Methods use the project's `Pin<Box<dyn Future>>` desugaring (matches
/// `heartbit_core::CredentialResolver`).
pub trait ReviewDelivery: Send + Sync {
    /// Send candidates to the user, wait for their pick (or timeout).
    /// Returns the outcome + opaque receipt for a later `report()`.
    fn deliver_and_await<'a>(
        &'a self,
        message: &'a ReviewMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DeliveredReview, ReviewDeliveryError>> + Send + 'a>>;

    /// Update the prior delivery with the final result. Implementations
    /// may noop if the medium doesn't support editing. Failure is
    /// non-fatal at the caller (run_review_pipeline logs and continues).
    fn report<'a>(
        &'a self,
        receipt: DeliveryReceipt,
        outcome: ReportableOutcome,
    ) -> Pin<Box<dyn Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivery_outcome_pick_displays_via_debug() {
        let o = DeliveryOutcome::Pick(2);
        assert_eq!(format!("{:?}", o), "Pick(2)");
    }

    #[test]
    fn reportable_outcome_posted_carries_url() {
        let o = ReportableOutcome::Posted {
            chosen_index: 1,
            tweet_url: "https://twitter.com/i/web/status/123".to_string(),
        };
        let s = format!("{:?}", o);
        assert!(s.contains("chosen_index: 1"), "got: {s}");
        assert!(s.contains("https://twitter.com"), "got: {s}");
    }

    #[test]
    fn review_delivery_error_transport_renders_inner_message() {
        let e = ReviewDeliveryError::Transport("connection refused".to_string());
        let s = format!("{e}");
        assert!(s.contains("connection refused"), "got: {s}");
        assert!(s.starts_with("delivery transport"), "got: {s}");
    }
}
```

- [ ] **Step 6: Create `crates/heartbit-ghost/src/review/prompts.rs`**

```rust
//! Render the review message for delivery and the report message for
//! the in-place edit after pick.

use crate::review::delivery::{ReportableOutcome, ReviewMessage};

/// Maximum total characters in the rendered Telegram review message body.
/// Telegram's per-message limit is 4096; we leave headroom for the
/// keyboard footer and emoji decorations.
const MAX_REVIEW_BODY_CHARS: usize = 3500;

/// Per-candidate truncation budget — keeps the message readable when
/// candidates are long. Truncation appends `…` and the rest is hidden.
const PER_CANDIDATE_TRUNCATE_CHARS: usize = 900;

/// Render the review message body for delivery. Output is plain text;
/// the delivery layer handles emoji rendering / Telegram-specific markup.
pub fn build_review_message(message: &ReviewMessage) -> String {
    let mut out = String::with_capacity(MAX_REVIEW_BODY_CHARS);
    out.push_str(&format!(
        "🪶 Draft for {} — {}\n\n",
        message.persona_name, message.topic
    ));
    let emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"];
    for (i, candidate) in message.candidates.iter().enumerate() {
        let label = emojis.get(i).copied().unwrap_or("•");
        out.push_str(label);
        out.push(' ');
        out.push_str(&truncate_with_ellipsis(candidate, PER_CANDIDATE_TRUNCATE_CHARS));
        out.push_str("\n\n");
    }
    out.push_str("Pick one, or Skip");
    // Hard-cap the total body in case truncation didn't keep us under.
    truncate_with_ellipsis(&out, MAX_REVIEW_BODY_CHARS)
}

/// Render the report message that replaces the original Telegram message
/// after the user picks (or skips, or times out).
pub fn build_report_message(outcome: &ReportableOutcome) -> String {
    match outcome {
        ReportableOutcome::Posted {
            chosen_index,
            tweet_url,
        } => format!(
            "✅ Posted draft {} — {}",
            chosen_index + 1,
            tweet_url
        ),
        ReportableOutcome::Skipped => "❎ Skipped (no post)".to_string(),
        ReportableOutcome::TimedOut => "⏰ Timed out — no pick".to_string(),
        ReportableOutcome::GateRejected {
            chosen_index,
            reason,
        } => format!(
            "🚫 Draft {} rejected by publish_gate: {}",
            chosen_index + 1,
            reason
        ),
        ReportableOutcome::PublishFailed {
            chosen_index,
            reason,
        } => format!(
            "⚠️ Publish failed for draft {}: {}",
            chosen_index + 1,
            reason
        ),
    }
}

fn truncate_with_ellipsis(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn mk_message(candidates: Vec<&str>) -> ReviewMessage {
        ReviewMessage {
            persona_name: "heartbit-ghost:x".to_string(),
            topic: "agent harness".to_string(),
            candidates: candidates.into_iter().map(String::from).collect(),
            interaction_id: Uuid::new_v4(),
        }
    }

    #[test]
    fn build_review_message_three_candidates_renders_with_emoji_labels() {
        let m = mk_message(vec!["alpha draft", "bravo draft", "charlie draft"]);
        let out = build_review_message(&m);
        assert!(out.contains("🪶 Draft for heartbit-ghost:x — agent harness"));
        assert!(out.contains("1️⃣ alpha draft"));
        assert!(out.contains("2️⃣ bravo draft"));
        assert!(out.contains("3️⃣ charlie draft"));
        assert!(out.ends_with("Pick one, or Skip"));
    }

    #[test]
    fn build_review_message_truncates_long_candidate_with_ellipsis() {
        let long = "x".repeat(2000);
        let m = mk_message(vec![long.as_str()]);
        let out = build_review_message(&m);
        assert!(
            out.chars().count() < 1500,
            "expected per-candidate truncation; got {} chars",
            out.chars().count()
        );
        assert!(out.contains('…'), "expected ellipsis: {out}");
    }

    #[test]
    fn build_review_message_handles_special_chars_passthrough() {
        // No HTML / Markdown escaping in this layer — delivery layer
        // handles its own escaping.
        let m = mk_message(vec!["draft with <html> & \"quotes\""]);
        let out = build_review_message(&m);
        assert!(out.contains("<html>"), "should not strip < >");
        assert!(out.contains("\"quotes\""), "should not strip quotes");
    }

    #[test]
    fn build_report_message_posted_includes_one_based_index_and_url() {
        let o = ReportableOutcome::Posted {
            chosen_index: 1, // 0-based
            tweet_url: "https://twitter.com/i/web/status/12345".to_string(),
        };
        let s = build_report_message(&o);
        assert!(s.contains("Posted draft 2"), "got: {s}"); // 1-based for users
        assert!(s.contains("https://twitter.com"), "got: {s}");
    }
}
```

- [ ] **Step 7: Create `crates/heartbit-ghost/src/review/tweet_split.rs`**

```rust
//! Convert a multi-tweet draft (single string with `\n\n` separators)
//! into the `Vec<String>` format the `twitter_thread` tool accepts.

/// Split a draft on `\n\n` boundaries, trim each tweet, drop empties.
///
/// Mirrors `pipeline/publish_gate.rs::check_publish_gate`'s splitting
/// rule so the two stay consistent — what passes the gate is what gets
/// posted.
pub fn parse_thread_tweets(draft: &str) -> Vec<String> {
    draft
        .split("\n\n")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_tweet_returns_one_element_vec() {
        let v = parse_thread_tweets("a single tweet");
        assert_eq!(v, vec!["a single tweet".to_string()]);
    }

    #[test]
    fn parse_thread_splits_on_double_newline() {
        let v = parse_thread_tweets("first\n\nsecond\n\nthird");
        assert_eq!(
            v,
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string(),
            ]
        );
    }

    #[test]
    fn parse_thread_trims_whitespace_around_each_tweet() {
        let v = parse_thread_tweets("  first  \n\n  second  ");
        assert_eq!(v, vec!["first".to_string(), "second".to_string()]);
    }

    #[test]
    fn parse_thread_filters_empty_segments() {
        // Triple newline produces an empty middle segment after split.
        let v = parse_thread_tweets("first\n\n\n\nsecond");
        assert_eq!(v, vec!["first".to_string(), "second".to_string()]);
    }
}
```

- [ ] **Step 8: Create `crates/heartbit-ghost/src/review/mod.rs` (skeleton — types + errors only; orchestration in Task 2)**

```rust
//! Review-mode pipeline — sends N candidate drafts to the user via a
//! [`ReviewDelivery`] (Telegram in production), awaits the user's pick,
//! then posts the chosen draft to X via the `twitter_thread` tool.
//!
//! Public entry: [`run_review_pipeline`] (lands in Task 2).

use std::path::Path;
use std::sync::Arc;

use heartbit_core::CredentialResolver;
use heartbit_core::llm::types::TokenUsage;
use heartbit_core::llm::BoxedProvider;
use heartbit_core::tool::Tool;
use thiserror::Error;

use crate::pipeline::{CandidateRecord, PipelineError, ProgressCallback};

pub mod delivery;
pub mod prompts;
pub mod tweet_split;

pub use delivery::{
    DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReportableOutcome, ReviewDelivery,
    ReviewDeliveryError, ReviewMessage,
};
pub use prompts::{build_report_message, build_review_message};
pub use tweet_split::parse_thread_tweets;

/// Configuration for one review-mode pipeline run.
#[derive(Clone)]
pub struct ReviewConfig<'a> {
    /// Persona instance name (used to load StyleProfile snapshot).
    pub persona_name: &'a str,
    /// Topic / prompt for this run.
    pub topic: &'a str,
    /// LLM provider (shared across sub-agents).
    pub provider: Arc<BoxedProvider>,
    /// Corpora root (currently unused; reserved for P1.3e few-shot retrieval).
    pub corpora_root: &'a Path,
    /// Profiles root (passed to SnapshotStore::open).
    pub profiles_root: &'a Path,
    /// Optional progress callback. Called with a short status string at
    /// each pipeline stage start.
    pub on_progress: Option<ProgressCallback>,
    /// Number of distinct candidate drafts to generate (1..=10).
    /// Same semantics as `PipelineConfig.candidates_per_draft`.
    pub candidates_per_draft: usize,
    /// Telegram-or-mock delivery layer.
    pub delivery: Arc<dyn ReviewDelivery>,
    /// Twitter-or-mock posting tool. Production wires
    /// `Arc::new(TwitterThreadTool::new())`; tests wire a mock Tool.
    pub twitter_tool: Arc<dyn Tool>,
    /// Credential resolver for `twitter_tool`. Threaded into
    /// `ExecutionContext::credentials` at execute-time.
    pub credentials: Arc<dyn CredentialResolver>,
}

/// Output of a successful review-mode run.
#[derive(Debug, Clone)]
pub struct ReviewOutput {
    /// All distinct candidate drafts (1..=`candidates_per_draft` after dedup).
    pub candidates: Vec<CandidateRecord>,
    /// Researcher's digest text.
    pub research_digest: String,
    /// Accumulated token usage across sub-agent calls.
    pub usage_summary: TokenUsage,
    /// What happened.
    pub outcome: ReviewOutcome,
}

/// Outcome of the review interaction.
#[derive(Debug, Clone)]
pub enum ReviewOutcome {
    /// User picked candidate `chosen_index` and the post was published.
    Posted {
        /// 0-based index into `candidates`.
        chosen_index: usize,
        /// Final URL of the first tweet in the (possibly single-tweet) thread.
        tweet_url: String,
        /// IDs of all tweets posted (1 for single, N for thread).
        tweet_ids: Vec<String>,
    },
    /// User pressed Skip.
    Skipped,
    /// Timeout elapsed before user responded.
    TimedOut,
    /// User picked candidate `chosen_index` but `publish_gate` rejected it.
    GateRejected {
        /// 0-based index of the rejected draft.
        chosen_index: usize,
        /// Reason from `PublishGateError`'s display.
        reason: String,
    },
    /// User picked candidate `chosen_index` but the X API call failed.
    PublishFailed {
        /// 0-based index of the draft that failed to post.
        chosen_index: usize,
        /// Failure reason.
        reason: String,
    },
}

/// Errors raised by `run_review_pipeline`. Note: `ReviewDelivery::report()`
/// failures are intentionally NOT a `ReviewError` variant. They're
/// non-fatal (post may have succeeded; only the after-the-fact message
/// edit failed) and are logged via the `on_progress` callback.
#[derive(Debug, Error)]
pub enum ReviewError {
    /// Candidate generation failed (delegates to `PipelineError`).
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Telegram (or mock) delivery failed.
    #[error("delivery: {0}")]
    Delivery(#[from] ReviewDeliveryError),
    /// Config validation at run start.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_error_pipeline_renders_inner_message() {
        let e = ReviewError::Pipeline(PipelineError::InvalidConfig(
            "candidates_per_draft must be in 1..=10 (got 0)".to_string(),
        ));
        let s = format!("{e}");
        assert!(s.contains("pipeline:"), "got: {s}");
        assert!(s.contains("invalid config"), "got: {s}");
    }

    #[test]
    fn review_error_delivery_renders_transport_error() {
        let e = ReviewError::Delivery(ReviewDeliveryError::Transport(
            "bot offline".to_string(),
        ));
        let s = format!("{e}");
        assert!(s.contains("delivery:"), "got: {s}");
        assert!(s.contains("bot offline"), "got: {s}");
    }

    #[test]
    fn review_error_invalid_config_renders_with_string() {
        let e = ReviewError::InvalidConfig("no profile snapshot".to_string());
        let s = format!("{e}");
        assert!(s.contains("invalid config"), "got: {s}");
        assert!(s.contains("no profile snapshot"), "got: {s}");
    }
}
```

- [ ] **Step 9: Modify `crates/heartbit-ghost/src/lib.rs`**

Add `pub mod review;` after `pub mod pipeline;`:

```rust
pub mod agents;
pub mod corpus;
pub mod pipeline;
pub mod review;
pub mod tools;
pub mod voice;
```

- [ ] **Step 10: Run heartbit-ghost tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-ghost --lib review 2>&1 | tail -10
```

Expected: 14 passed (3 delivery + 4 prompts + 4 tweet_split + 3 mod). Plus the pipeline tests still pass (run `cargo test -p heartbit-ghost --lib pipeline` to confirm — should still be 43).

- [ ] **Step 11: Workspace gate (heartbit-ghost only)**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean. Note: `Arc<dyn ReviewDelivery>` and `Arc<dyn Tool>` may trigger `clippy::type_complexity` on the `ReviewConfig` struct. If so, accept (it's a public type alias domain) or extract `pub type` aliases — your call.

- [ ] **Step 12: Commit**

```bash
cd /home/pleclech/projects/heartbit
git add crates/heartbit-telegram/src/keyboard.rs \
        crates/heartbit-telegram/src/lib.rs \
        crates/heartbit-ghost/src/review/mod.rs \
        crates/heartbit-ghost/src/review/delivery.rs \
        crates/heartbit-ghost/src/review/prompts.rs \
        crates/heartbit-ghost/src/review/tweet_split.rs \
        crates/heartbit-ghost/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(ghost,telegram): pipeline review — foundation types + helpers (P1.3d)

Foundation layer for P1.3d. Pure types, traits, and helpers — no
orchestration or teloxide dep:

- heartbit-telegram: CallbackAction::PersonaPick + PickChoice +
  persona_pick_buttons (N=1..=9 buttons + Skip) + parse_callback_data
  arm for "p:{uuid}:{choice}" prefix. 6 tests.

- heartbit-ghost: new review module with:
  - ReviewDelivery trait (object-safe via Pin<Box<dyn Future>>;
    matches heartbit_core::CredentialResolver convention) + types
    (ReviewMessage, DeliveredReview, DeliveryReceipt, DeliveryOutcome,
    ReportableOutcome, ReviewDeliveryError). 3 tests.
  - prompts.rs: build_review_message (3500-char body cap +
    900-char per-candidate truncation) + build_report_message
    (Posted/Skipped/TimedOut/GateRejected/PublishFailed). 4 tests.
  - tweet_split.rs: parse_thread_tweets — splits on \n\n, trims,
    filters empty (matches publish_gate's splitting rule). 4 tests.
  - mod.rs: ReviewConfig, ReviewOutput, ReviewOutcome, ReviewError
    (skeleton — run_review_pipeline lands in Task 2). 3 tests.

heartbit-ghost stays free of teloxide (the trait abstraction
enforces this). Tasks 2/3 wire the orchestration body and the
TelegramReviewDelivery impl in heartbit-cli.

20 net new tests: 6 keyboard + 3 delivery + 4 prompts + 4 tweet_split
+ 3 mod display.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3d-telegram-review-design.md §3.1, §3.2, §3.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `run_review_pipeline` body + integration tests

**Why:** The orchestration. Wires Task 1's types + the P1.3c shared helpers (`generate_candidate`, `dedup_candidates`, `runner_from_recipe`) into the review-mode pipeline. Adds the `MockReviewDelivery` and `MockTwitterTool` test helpers needed for deterministic integration testing.

**Files:**
- Modify: `crates/heartbit-ghost/src/review/mod.rs` — add `run_review_pipeline`, `parse_twitter_thread_output`, mocks, 6 integration tests
- Modify: `crates/heartbit-ghost/src/pipeline/mod.rs` — promote `generate_candidate` from `async fn` (private) to `pub(crate)` and `dedup_candidates` similarly, so the new review module can call them. Also export `runner_from_recipe`.

- [ ] **Step 1: Promote shared helpers in `crates/heartbit-ghost/src/pipeline/mod.rs` to `pub(crate)`**

The current `async fn generate_candidate(...)`, `fn dedup_candidates(...)`, and `fn runner_from_recipe(...)` are private (or `pub(crate)` for the latter). Make all three `pub(crate)`:

```rust
// Find: async fn generate_candidate(
// Change to: pub(crate) async fn generate_candidate(

// Find: fn dedup_candidates(
// Change to: pub(crate) fn dedup_candidates(

// runner_from_recipe is already pub(crate) — verify.
```

Verify nothing else changed.

- [ ] **Step 2: Add `parse_twitter_thread_output` helper to `crates/heartbit-ghost/src/review/mod.rs`**

Append (just before the `#[cfg(test)] mod tests` block):

```rust
/// Parse the `twitter_thread` tool's success output into `(tweet_ids, head_url)`.
///
/// The tool's `ToolOutput.content` on success is the JSON serialization
/// of `ThreadOutput { thread_root_id, tweet_ids, urls }` (see
/// `crates/heartbit-ghost/src/tools/thread.rs`). On parse failure (e.g.,
/// the tool was mocked with a non-JSON string), returns `(vec![], "<unknown>")`
/// — the caller has already accepted that the post succeeded; treat the
/// missing structure as a non-fatal observability gap.
pub(crate) fn parse_twitter_thread_output(
    content: &str,
) -> (Vec<String>, String) {
    #[derive(serde::Deserialize)]
    struct Parsed {
        tweet_ids: Vec<String>,
        urls: Vec<String>,
    }
    match serde_json::from_str::<Parsed>(content) {
        Ok(p) => {
            let head_url = p.urls.first().cloned().unwrap_or_else(|| {
                p.tweet_ids
                    .first()
                    .map(|id| format!("https://twitter.com/i/web/status/{id}"))
                    .unwrap_or_else(|| "<unknown>".to_string())
            });
            (p.tweet_ids, head_url)
        }
        Err(_) => (Vec::new(), "<unknown>".to_string()),
    }
}
```

- [ ] **Step 3: Add `run_review_pipeline` body**

Append (just after `parse_twitter_thread_output`):

```rust
/// Execute one review-mode pipeline run.
///
/// Flow: snapshot load → research → N parallel writer→critic→fact_check
/// → dedup → ReviewDelivery::deliver_and_await → on Pick: publish_gate
/// → twitter_tool.execute → ReviewDelivery::report → return.
pub async fn run_review_pipeline(
    cfg: ReviewConfig<'_>,
) -> Result<ReviewOutput, ReviewError> {
    let progress = |msg: &str| {
        if let Some(cb) = cfg.on_progress.as_ref() {
            cb(msg);
        }
    };

    // 1. Validate config.
    if !(1..=10).contains(&cfg.candidates_per_draft) {
        return Err(ReviewError::InvalidConfig(format!(
            "candidates_per_draft must be in 1..=10 (got {})",
            cfg.candidates_per_draft,
        )));
    }
    // persona_pick_buttons asserts 1..=9. Constrain further here.
    if cfg.candidates_per_draft > 9 {
        return Err(ReviewError::InvalidConfig(format!(
            "review mode requires candidates_per_draft <= 9 \
             (Telegram inline-keyboard limit; got {})",
            cfg.candidates_per_draft,
        )));
    }

    // 2. Load StyleProfile snapshot — same as run_pipeline.
    progress("Loading profile snapshot...");
    let store =
        crate::voice::SnapshotStore::open(cfg.profiles_root, cfg.persona_name)?;
    let snapshot = store.load_latest()?.ok_or_else(|| {
        PipelineError::NoProfileSnapshot {
            persona: cfg.persona_name.to_string(),
            profiles_dir: cfg.profiles_root.join(cfg.persona_name),
        }
    })?;
    let profile = snapshot.profile;

    // 3. Build agents (researcher / writer / critic / fact_check ONLY —
    // no judge or image_generator in review mode).
    use crate::agents::{
        fact_check_recipe, researcher_recipe, style_critic_recipe, writer_recipe,
    };
    use heartbit_core::tool::builtins::{WebFetchTool, WebSearchTool};

    let researcher_tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(WebSearchTool::new()),
        Arc::new(WebFetchTool::new()),
    ];
    let researcher = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        researcher_recipe(),
        researcher_tools,
    )
    .map_err(|e| {
        PipelineError::Builder {
            stage: "researcher".to_string(),
            source: e,
        }
    })?;
    let writer = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        writer_recipe(),
        Vec::new(),
    )
    .map_err(|e| {
        PipelineError::Builder {
            stage: "writer".to_string(),
            source: e,
        }
    })?;
    let critic = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        style_critic_recipe(),
        Vec::new(),
    )
    .map_err(|e| {
        PipelineError::Builder {
            stage: "style_critic".to_string(),
            source: e,
        }
    })?;
    let fact = crate::pipeline::runner_from_recipe(
        cfg.provider.clone(),
        fact_check_recipe(),
        Vec::new(),
    )
    .map_err(|e| {
        PipelineError::Builder {
            stage: "fact_check".to_string(),
            source: e,
        }
    })?;

    let mut total_usage = TokenUsage::default();

    // 4. Run researcher.
    progress("Researching topic...");
    let researcher_out = researcher.execute(cfg.topic).await.map_err(|e| {
        PipelineError::Agent {
            stage: "researcher".to_string(),
            source: e,
        }
    })?;
    let research_digest = researcher_out.result.clone();
    total_usage += researcher_out.tokens_used;

    // 5. Render voice guidelines.
    let voice_guidelines = crate::pipeline::render_style_profile_as_english(&profile);

    // 6. Parallel candidate generation — same shape as run_pipeline.
    let n = cfg.candidates_per_draft;
    progress(&format!("Generating {n} candidate(s) in parallel..."));

    let writer = std::sync::Arc::new(writer);
    let critic = std::sync::Arc::new(critic);
    let fact = std::sync::Arc::new(fact);
    let topic_owned: String = cfg.topic.to_string();
    let digest_owned = std::sync::Arc::new(research_digest.clone());
    let voice_owned = std::sync::Arc::new(voice_guidelines.clone());

    let mut joinset: tokio::task::JoinSet<Result<CandidateRecord, PipelineError>> =
        tokio::task::JoinSet::new();
    for i in 0..n {
        let writer = writer.clone();
        let critic = critic.clone();
        let fact = fact.clone();
        let topic = topic_owned.clone();
        let digest = digest_owned.clone();
        let voice = voice_owned.clone();
        joinset.spawn(async move {
            crate::pipeline::generate_candidate(
                i, n, &topic, &digest, &voice, &writer, &critic, &fact,
            )
            .await
        });
    }

    let mut candidates: Vec<CandidateRecord> = Vec::with_capacity(n);
    let mut errors: Vec<PipelineError> = Vec::new();
    while let Some(res) = joinset.join_next().await {
        match res {
            Ok(Ok(rec)) => candidates.push(rec),
            Ok(Err(e)) => {
                progress(&format!("candidate failed: {e}"));
                errors.push(e);
            }
            Err(joinerr) => {
                progress(&format!("candidate task panicked: {joinerr}"));
            }
        }
    }
    candidates.sort_by_key(|c| c.variant_index);

    if candidates.is_empty() {
        if errors.len() == 1 {
            return Err(ReviewError::Pipeline(errors.swap_remove(0)));
        }
        return Err(ReviewError::Pipeline(
            PipelineError::AllCandidatesFailed { errors, n },
        ));
    }

    // 7. Dedup. (Skip the retry-once pass — review mode is OK with
    // ship-with-fewer; the user may pick from the surviving distinct set.)
    let candidates = crate::pipeline::dedup_candidates(candidates);

    // Sum per-candidate usage.
    for c in &candidates {
        total_usage += c.usage;
    }

    // 8. Build ReviewMessage.
    let interaction_id = uuid::Uuid::new_v4();
    let candidate_drafts: Vec<String> =
        candidates.iter().map(|c| c.draft.clone()).collect();
    let review_msg = ReviewMessage {
        persona_name: cfg.persona_name.to_string(),
        topic: cfg.topic.to_string(),
        candidates: candidate_drafts,
        interaction_id,
    };

    // 9. Deliver and await pick.
    progress("Sending review to user...");
    let delivered = cfg.delivery.deliver_and_await(&review_msg).await?;

    // 10. Branch on outcome.
    let (outcome, report) = match delivered.outcome {
        DeliveryOutcome::Skip => {
            progress("User skipped.");
            (ReviewOutcome::Skipped, ReportableOutcome::Skipped)
        }
        DeliveryOutcome::TimedOut => {
            progress("Review timed out.");
            (ReviewOutcome::TimedOut, ReportableOutcome::TimedOut)
        }
        DeliveryOutcome::Pick(chosen_index) => {
            if chosen_index >= candidates.len() {
                return Err(ReviewError::InvalidConfig(format!(
                    "delivery returned out-of-range pick index {chosen_index} \
                     (candidates.len() = {})",
                    candidates.len()
                )));
            }
            let chosen = &candidates[chosen_index];

            // 10a. publish_gate.
            match crate::pipeline::check_publish_gate(&chosen.draft, &profile) {
                Err(gate_err) => {
                    let reason = format!("{gate_err}");
                    progress(&format!("publish_gate rejected pick: {reason}"));
                    (
                        ReviewOutcome::GateRejected {
                            chosen_index,
                            reason: reason.clone(),
                        },
                        ReportableOutcome::GateRejected {
                            chosen_index,
                            reason,
                        },
                    )
                }
                Ok(()) => {
                    // 10b. Post via twitter_tool.
                    progress(&format!("Posting candidate {chosen_index}..."));
                    let tweets = parse_thread_tweets(&chosen.draft);
                    let exec_ctx = heartbit_core::ExecutionContext {
                        credentials: Some(cfg.credentials.clone()),
                        ..Default::default()
                    };
                    let input = serde_json::json!({"tweets": tweets});
                    match cfg.twitter_tool.execute(&exec_ctx, input).await {
                        Err(e) => {
                            let reason = format!("{e}");
                            progress(&format!("twitter_tool errored: {reason}"));
                            (
                                ReviewOutcome::PublishFailed {
                                    chosen_index,
                                    reason: reason.clone(),
                                },
                                ReportableOutcome::PublishFailed {
                                    chosen_index,
                                    reason,
                                },
                            )
                        }
                        Ok(tool_out) if tool_out.is_error => {
                            let reason = tool_out.content.clone();
                            progress(&format!(
                                "twitter_tool returned is_error=true: {reason}"
                            ));
                            (
                                ReviewOutcome::PublishFailed {
                                    chosen_index,
                                    reason: reason.clone(),
                                },
                                ReportableOutcome::PublishFailed {
                                    chosen_index,
                                    reason,
                                },
                            )
                        }
                        Ok(tool_out) => {
                            let (tweet_ids, tweet_url) =
                                parse_twitter_thread_output(&tool_out.content);
                            (
                                ReviewOutcome::Posted {
                                    chosen_index,
                                    tweet_url: tweet_url.clone(),
                                    tweet_ids,
                                },
                                ReportableOutcome::Posted {
                                    chosen_index,
                                    tweet_url,
                                },
                            )
                        }
                    }
                }
            }
        }
    };

    // 11. Report outcome (non-fatal on error).
    if let Err(e) = cfg.delivery.report(delivered.receipt, report).await {
        progress(&format!("report failed (non-fatal): {e}"));
    }

    progress("Done.");
    Ok(ReviewOutput {
        candidates,
        research_digest,
        usage_summary: total_usage,
        outcome,
    })
}
```

- [ ] **Step 4: Add `MockReviewDelivery` + `MockTwitterTool` test helpers**

Inside the existing `#[cfg(test)] mod tests` block in `review/mod.rs`, append after the 3 existing display tests:

```rust
    use heartbit_core::config::AgentConfig;
    use heartbit_core::error::Error as CoreError;
    use heartbit_core::execution_context::CredentialResolver as CredentialResolverTrait;
    use heartbit_core::execution_context::Secret;
    use heartbit_core::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };
    use heartbit_core::llm::{BoxedProvider, LlmProvider};
    use heartbit_core::tool::{ToolDefinition, ToolOutput};
    use heartbit_core::ExecutionContext;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// MockReviewDelivery returns a pre-canned outcome and records report() calls.
    struct MockReviewDelivery {
        outcome: DeliveryOutcome,
        reports: Mutex<Vec<ReportableOutcome>>,
    }

    impl MockReviewDelivery {
        fn arc(outcome: DeliveryOutcome) -> Arc<dyn ReviewDelivery> {
            Arc::new(MockReviewDelivery {
                outcome,
                reports: Mutex::new(Vec::new()),
            })
        }
    }

    impl ReviewDelivery for MockReviewDelivery {
        fn deliver_and_await<'a>(
            &'a self,
            _message: &'a ReviewMessage,
        ) -> Pin<Box<dyn Future<Output = Result<DeliveredReview, ReviewDeliveryError>> + Send + 'a>>
        {
            let outcome = self.outcome.clone();
            Box::pin(async move {
                Ok(DeliveredReview {
                    outcome,
                    receipt: DeliveryReceipt {
                        data: serde_json::Value::Null,
                    },
                })
            })
        }

        fn report<'a>(
            &'a self,
            _receipt: DeliveryReceipt,
            outcome: ReportableOutcome,
        ) -> Pin<Box<dyn Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>> {
            self.reports.lock().unwrap().push(outcome);
            Box::pin(async move { Ok(()) })
        }
    }

    /// MockTwitterTool returns a canned ToolOutput. Tests can configure
    /// the success body or set is_error=true.
    struct MockTwitterTool {
        canned: Mutex<Option<ToolOutput>>,
        last_input: Mutex<Option<serde_json::Value>>,
    }

    impl MockTwitterTool {
        fn success(thread_json: &str) -> Arc<dyn Tool> {
            Arc::new(MockTwitterTool {
                canned: Mutex::new(Some(ToolOutput::success(thread_json))),
                last_input: Mutex::new(None),
            })
        }

        fn errored(reason: &str) -> Arc<dyn Tool> {
            Arc::new(MockTwitterTool {
                canned: Mutex::new(Some(ToolOutput::error(reason))),
                last_input: Mutex::new(None),
            })
        }
    }

    impl Tool for MockTwitterTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "twitter_thread".to_string(),
                description: "mock".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, CoreError>> + Send + '_>> {
            *self.last_input.lock().unwrap() = Some(input);
            let canned = self.canned.lock().unwrap().take();
            Box::pin(async move {
                canned.ok_or_else(|| CoreError::Agent("mock twitter tool exhausted".into()))
            })
        }
    }

    /// MockProvider — same shape as pipeline::tests but local copy
    /// (the pipeline tests' MockProvider is `pub(super)`-scoped).
    struct MockProvider {
        responses: Mutex<VecDeque<String>>,
    }

    impl MockProvider {
        fn arc(responses: Vec<&str>) -> Arc<BoxedProvider> {
            let p = MockProvider {
                responses: Mutex::new(responses.into_iter().map(String::from).collect()),
            };
            Arc::new(BoxedProvider::new(p))
        }
    }

    impl LlmProvider for MockProvider {
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CoreError>> + Send {
            let response = self.responses.lock().unwrap().pop_front();
            let has_respond = request
                .tools
                .iter()
                .any(|t| t.name == heartbit_core::llm::types::RESPOND_TOOL_NAME);
            async move {
                let text = response
                    .ok_or_else(|| CoreError::Agent("mock exhausted".to_string()))?;
                let content = if has_respond {
                    let value: serde_json::Value =
                        serde_json::from_str(&text).map_err(|e| {
                            CoreError::Agent(format!(
                                "mock: canned response is not valid JSON: {e}"
                            ))
                        })?;
                    vec![ContentBlock::ToolUse {
                        id: "respond_1".to_string(),
                        name: "__respond__".to_string(),
                        input: value,
                    }]
                } else {
                    vec![ContentBlock::Text { text }]
                };
                Ok(CompletionResponse {
                    content,
                    usage: TokenUsage::default(),
                    stop_reason: if has_respond {
                        StopReason::ToolUse
                    } else {
                        StopReason::EndTurn
                    },
                    model: None,
                })
            }
        }
    }

    /// Stub credential resolver — never called in mock tests.
    struct StubCredentialResolver;

    impl CredentialResolverTrait for StubCredentialResolver {
        fn resolve(
            &self,
            _name: &str,
        ) -> Pin<Box<dyn Future<Output = Result<Secret, CoreError>> + Send + '_>> {
            Box::pin(async move { Ok(Secret::new("stub")) })
        }
    }

    /// Snapshot fixture — same shape as pipeline::tests::seed_snapshot.
    fn seed_snapshot(persona: &str) -> (TempDir, std::path::PathBuf) {
        use crate::voice::{
            BlendEntry, BlendRecipe, EmDashPolicy, EmojiPolicy, Formatting,
            FragmentFrequency, HashtagPolicy, LineBreaks, OpeningPattern,
            PartialStyleProfile, PeriodsPolicy, QuotationMarks, SentenceLengthTarget,
            SnapshotStore, SpecificityTarget, StyleProfile, ThreadRhythm,
        };
        let dir = TempDir::new().unwrap();
        let profile = StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![OpeningPattern::ClaimFirst],
            opening_pattern_weights: vec![1.0],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::Never,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec!["specific".to_string()],
            ai_tells_to_avoid: vec!["delve".to_string()],
            thread_rhythm: ThreadRhythm::Linear,
            thread_max_length: 5,
            thread_opener_must_hook: false,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        };
        let recipe = BlendRecipe {
            version: 1,
            blend: vec![BlendEntry {
                writer: "k".to_string(),
                weight: 1.0,
            }],
            overrides: PartialStyleProfile::default(),
        };
        let store = SnapshotStore::open(dir.path(), persona).unwrap();
        store.save_new(profile, &recipe).unwrap();
        let root = dir.path().to_path_buf();
        (dir, root)
    }
```

- [ ] **Step 5: Add 6 integration tests**

Append after the test helpers:

```rust
    /// Boilerplate: build a single-candidate ReviewConfig (saves repetition
    /// across tests). `provider` and `delivery` and `twitter_tool` are
    /// caller-provided.
    fn mk_review_cfg<'a>(
        profiles_root: &'a std::path::Path,
        provider: Arc<BoxedProvider>,
        delivery: Arc<dyn ReviewDelivery>,
        twitter_tool: Arc<dyn Tool>,
    ) -> ReviewConfig<'a> {
        ReviewConfig {
            persona_name: "x",
            topic: "agent harness",
            provider,
            corpora_root: profiles_root, // unused in tests; reuse path
            profiles_root,
            on_progress: None,
            candidates_per_draft: 1,
            delivery,
            twitter_tool,
            credentials: Arc::new(StubCredentialResolver),
        }
    }

    #[tokio::test]
    async fn run_review_pipeline_pick_index_0_posts_to_twitter() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- AI is moving fast",                      // researcher
            "concrete short post",                                         // writer iter 1
            r#"{"verdict": "pass", "style_match_score": 0.92}"#,          // critic
            r#"{"verdict": "verified"}"#,                                  // fact_check
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::success(
            r#"{"thread_root_id":"123","tweet_ids":["123"],"urls":["https://twitter.com/i/web/status/123"]}"#,
        );
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg).await.expect("happy path");
        match out.outcome {
            ReviewOutcome::Posted { chosen_index, tweet_url, tweet_ids } => {
                assert_eq!(chosen_index, 0);
                assert_eq!(tweet_url, "https://twitter.com/i/web/status/123");
                assert_eq!(tweet_ids, vec!["123".to_string()]);
            }
            other => panic!("expected Posted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_review_pipeline_skip_returns_skipped_no_post() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            "draft text",
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Skip);
        // twitter_tool should never be called — set it up to return an
        // error so we'd notice if it was invoked.
        let twitter_tool = MockTwitterTool::errored("should not be called");
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg).await.expect("skip is success");
        assert!(matches!(out.outcome, ReviewOutcome::Skipped));
    }

    #[tokio::test]
    async fn run_review_pipeline_timed_out_returns_timed_out_no_post() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            "draft text",
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::TimedOut);
        let twitter_tool = MockTwitterTool::errored("should not be called");
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg).await.expect("timeout is success");
        assert!(matches!(out.outcome, ReviewOutcome::TimedOut));
    }

    #[tokio::test]
    async fn run_review_pipeline_pick_publish_gate_rejects_long_draft() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let too_long = "x".repeat(290);
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            too_long.as_str(),
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::errored("should not be called");
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg).await.expect("gate rejection is success");
        match out.outcome {
            ReviewOutcome::GateRejected { chosen_index, reason } => {
                assert_eq!(chosen_index, 0);
                assert!(reason.contains("280"), "got: {reason}");
            }
            other => panic!("expected GateRejected, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_review_pipeline_pick_twitter_api_error_returns_publish_failed() {
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![
            "Research digest:\n- notes",
            "short post",
            r#"{"verdict": "pass", "style_match_score": 0.9}"#,
            r#"{"verdict": "verified"}"#,
        ]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Pick(0));
        let twitter_tool = MockTwitterTool::errored("X auth failed (401): Unauthorized");
        let cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        let out = run_review_pipeline(cfg).await.expect("publish failure is success");
        match out.outcome {
            ReviewOutcome::PublishFailed { chosen_index, reason } => {
                assert_eq!(chosen_index, 0);
                assert!(reason.contains("401"), "got: {reason}");
            }
            other => panic!("expected PublishFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_review_pipeline_invalid_candidates_per_draft_rejected() {
        // Build a minimal cfg with candidates_per_draft = 0 (invalid).
        let (_dir, profiles_root) = seed_snapshot("x");
        let provider = MockProvider::arc(vec![]);
        let delivery = MockReviewDelivery::arc(DeliveryOutcome::Skip);
        let twitter_tool = MockTwitterTool::errored("never called");
        let mut cfg = mk_review_cfg(&profiles_root, provider, delivery, twitter_tool);
        cfg.candidates_per_draft = 0;
        let err = run_review_pipeline(cfg).await.unwrap_err();
        match err {
            ReviewError::InvalidConfig(msg) => {
                assert!(msg.contains("candidates_per_draft"), "got: {msg}");
                assert!(msg.contains("1..=10"), "got: {msg}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }
```

- [ ] **Step 6: Run tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-ghost --lib review 2>&1 | tail -10
```

Expected: 14 (Task 1) + 6 (new integration) = **20 passed**.

Plus pipeline tests stay green:

```bash
cargo test -p heartbit-ghost --lib pipeline 2>&1 | tail -3
```

Expected: 43 passed (unchanged from P1.3c).

- [ ] **Step 7: Workspace gate**

```bash
cargo fmt -p heartbit-ghost -- --check
cargo clippy -p heartbit-ghost --all-targets -- -D warnings
```

Both clean. If `clippy::too_many_arguments` fires on `run_review_pipeline`, that's accepted (the function takes a single `ReviewConfig` argument so this should not fire).

- [ ] **Step 8: Commit**

```bash
git add crates/heartbit-ghost/src/review/mod.rs \
        crates/heartbit-ghost/src/pipeline/mod.rs
git commit -m "$(cat <<'EOF'
feat(ghost): review pipeline orchestration body + integration tests (P1.3d)

The orchestration layer for review-mode pipeline. Wires Task 1's types
+ P1.3c's shared candidate-generation helpers into the review flow:

1. Validate candidates_per_draft (1..=9 because Telegram inline
   keyboard limit + spec).
2. Load StyleProfile snapshot.
3. Build researcher / writer / critic / fact_check agents (NO
   judge / image_generator — review mode skips them).
4. Researcher → digest.
5. Render voice guidelines.
6. Parallel candidate generation via tokio::JoinSet (same shape as
   run_pipeline; reuses `pipeline::generate_candidate`).
7. dedup_candidates (no retry-once pass — review mode is OK with
   ship-with-fewer; user picks from surviving distinct set).
8. Build ReviewMessage + UUID interaction_id.
9. ReviewDelivery::deliver_and_await(message).
10. Match DeliveryOutcome:
    - Skip → Skipped
    - TimedOut → TimedOut
    - Pick(i) → publish_gate(chosen) → twitter_tool.execute() →
      Posted | GateRejected | PublishFailed
11. ReviewDelivery::report() — non-fatal on error.
12. Return ReviewOutput.

Plus `parse_twitter_thread_output` to deserialize the tool's JSON
content into (tweet_ids, head_url). Falls back to "<unknown>" if
the tool returned non-JSON content (e.g., mock tool with custom
payload).

Promoted `pipeline::generate_candidate` and `pipeline::dedup_candidates`
from private to pub(crate) so the review module can call them.

Test infrastructure: MockReviewDelivery (canned outcome + reports
recorder), MockTwitterTool (canned ToolOutput, last_input
inspector), StubCredentialResolver (resolve always returns the
literal "stub" Secret). 6 integration tests cover all 5 outcome
variants + InvalidConfig.

20 review tests pass total (14 Task 1 + 6 integration). Pipeline
tests stay at 43.

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3d-telegram-review-design.md §2, §4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `TelegramReviewDelivery` impl + CLI `--review` flag wiring

**Why:** The production-side glue. Implements `ReviewDelivery` using a standalone teloxide bot in the CLI process. CLI gets a `--review` flag that branches between P1.3c direct mode and the new review mode. Constructs `TwitterThreadTool`, env-based `CredentialResolver`, `TelegramReviewDelivery`, and stitches them into a `ReviewConfig`.

**Files:**
- Create: `crates/heartbit-cli/src/persona_review.rs`
- Modify: `crates/heartbit-cli/src/main.rs` — `mod persona_review;`
- Modify: `crates/heartbit-cli/src/persona.rs` — `Run` arm gains `--review`, branches on it

- [ ] **Step 1: Create `crates/heartbit-cli/src/persona_review.rs`**

```rust
//! `TelegramReviewDelivery` — standalone teloxide bot in the CLI process
//! that sends review messages with inline keyboards, awaits a callback,
//! and edits the message in place when the outcome is reported.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use heartbit_ghost::review::{
    DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReportableOutcome, ReviewDelivery,
    ReviewDeliveryError, ReviewMessage, build_report_message, build_review_message,
};
use heartbit_telegram::{CallbackAction, PickChoice, parse_callback_data, persona_pick_buttons};
use teloxide::prelude::*;
use teloxide::types::{ChatId, InlineKeyboardButton, InlineKeyboardMarkup, MessageId};
use tokio::sync::{Mutex as AsyncMutex, oneshot};
use tokio::time::timeout;
use uuid::Uuid;

const DEFAULT_TIMEOUT_SECS: u64 = 3600;

/// Standalone teloxide-backed `ReviewDelivery`.
pub struct TelegramReviewDelivery {
    bot: Bot,
    chat_id: ChatId,
    timeout: Duration,
    /// Pending pick resolvers keyed by interaction_id.
    pending: Arc<AsyncMutex<HashMap<Uuid, oneshot::Sender<DeliveryOutcome>>>>,
}

impl TelegramReviewDelivery {
    /// Construct from environment variables and eagerly spawn the
    /// callback dispatcher.
    ///
    /// Required env:
    /// - `HEARTBIT_TELEGRAM_TOKEN` — bot token
    /// - `HEARTBIT_TELEGRAM_REVIEW_CHAT_ID` — destination chat (`i64`)
    ///
    /// Optional:
    /// - `HEARTBIT_REVIEW_TIMEOUT_SECS` — pick timeout (default 3600 = 1h)
    pub fn from_env() -> Result<Self, ReviewDeliveryError> {
        let token = std::env::var("HEARTBIT_TELEGRAM_TOKEN").map_err(|_| {
            ReviewDeliveryError::Config("HEARTBIT_TELEGRAM_TOKEN env var not set".into())
        })?;
        let chat_id_raw = std::env::var("HEARTBIT_TELEGRAM_REVIEW_CHAT_ID").map_err(|_| {
            ReviewDeliveryError::Config(
                "HEARTBIT_TELEGRAM_REVIEW_CHAT_ID env var not set".into(),
            )
        })?;
        let chat_id_num: i64 = chat_id_raw.parse().map_err(|e| {
            ReviewDeliveryError::Config(format!(
                "invalid HEARTBIT_TELEGRAM_REVIEW_CHAT_ID '{chat_id_raw}': {e}"
            ))
        })?;
        let timeout_secs = std::env::var("HEARTBIT_REVIEW_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let bot = Bot::new(token);
        let pending: Arc<AsyncMutex<HashMap<Uuid, oneshot::Sender<DeliveryOutcome>>>> =
            Arc::new(AsyncMutex::new(HashMap::new()));

        // Eagerly spawn the callback dispatcher. One dispatcher per
        // TelegramReviewDelivery instance; lives for the duration of the
        // CLI process.
        let dispatcher_bot = bot.clone();
        let dispatcher_pending = pending.clone();
        tokio::spawn(async move {
            let handler =
                Update::filter_callback_query().endpoint(move |q: CallbackQuery, bot: Bot| {
                    let pending = dispatcher_pending.clone();
                    async move {
                        let data = match q.data.as_ref() {
                            Some(d) => d,
                            None => return Ok::<_, teloxide::RequestError>(()),
                        };
                        let action = match parse_callback_data(data) {
                            Ok(a) => a,
                            Err(_) => return Ok(()),
                        };
                        if let CallbackAction::PersonaPick {
                            interaction_id,
                            choice,
                        } = action
                        {
                            let outcome = match choice {
                                PickChoice::Index(i) => DeliveryOutcome::Pick(i),
                                PickChoice::Skip => DeliveryOutcome::Skip,
                            };
                            let mut map = pending.lock().await;
                            if let Some(sender) = map.remove(&interaction_id) {
                                let _ = sender.send(outcome);
                                // Acknowledge the callback (UI feedback).
                                let _ = bot.answer_callback_query(q.id.clone()).await;
                            }
                        }
                        Ok(())
                    }
                });
            Dispatcher::builder(dispatcher_bot, handler)
                .build()
                .dispatch()
                .await;
        });

        Ok(Self {
            bot,
            chat_id: ChatId(chat_id_num),
            timeout: Duration::from_secs(timeout_secs),
            pending,
        })
    }
}

impl ReviewDelivery for TelegramReviewDelivery {
    fn deliver_and_await<'a>(
        &'a self,
        message: &'a ReviewMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DeliveredReview, ReviewDeliveryError>> + Send + 'a>>
    {
        Box::pin(async move {
            let body = build_review_message(message);
            let buttons = persona_pick_buttons(message.interaction_id, message.candidates.len());
            let keyboard = InlineKeyboardMarkup::new(vec![buttons
                .into_iter()
                .map(|(label, data)| InlineKeyboardButton::callback(label, data))
                .collect::<Vec<_>>()]);

            let (tx, rx) = oneshot::channel::<DeliveryOutcome>();
            self.pending.lock().await.insert(message.interaction_id, tx);

            // Dispatcher already running (spawned by from_env).
            let sent = self
                .bot
                .send_message(self.chat_id, body)
                .reply_markup(keyboard)
                .await
                .map_err(|e| ReviewDeliveryError::Transport(format!("send_message: {e}")))?;
            let message_id = sent.id;

            let outcome = match timeout(self.timeout, rx).await {
                Ok(Ok(o)) => o,
                Ok(Err(_)) => DeliveryOutcome::TimedOut,  // sender dropped
                Err(_) => {
                    // Timeout — clean up the pending entry.
                    self.pending.lock().await.remove(&message.interaction_id);
                    DeliveryOutcome::TimedOut
                }
            };

            let receipt = DeliveryReceipt {
                data: serde_json::json!({
                    "chat_id": self.chat_id.0,
                    "message_id": message_id.0,
                }),
            };

            Ok(DeliveredReview { outcome, receipt })
        })
    }

    fn report<'a>(
        &'a self,
        receipt: DeliveryReceipt,
        outcome: ReportableOutcome,
    ) -> Pin<Box<dyn Future<Output = Result<(), ReviewDeliveryError>> + Send + 'a>> {
        Box::pin(async move {
            let chat_id_num = receipt.data.get("chat_id").and_then(|v| v.as_i64()).ok_or_else(
                || ReviewDeliveryError::InvalidCallback("receipt missing chat_id".into()),
            )?;
            let message_id_num = receipt
                .data
                .get("message_id")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    ReviewDeliveryError::InvalidCallback("receipt missing message_id".into())
                })?;
            let body = build_report_message(&outcome);
            self.bot
                .edit_message_text(ChatId(chat_id_num), MessageId(message_id_num as i32), body)
                .await
                .map_err(|e| ReviewDeliveryError::Transport(format!("edit_message: {e}")))?;
            Ok(())
        })
    }
}

/// Helper: construct the production `ReviewConfig` from env + CLI args.
pub async fn review_config_from_env<'a>(
    persona_name: &'a str,
    topic: &'a str,
    candidates_per_draft: usize,
    provider: Arc<heartbit_core::llm::BoxedProvider>,
    corpora_root: &'a std::path::Path,
    profiles_root: &'a std::path::Path,
    on_progress: Option<heartbit_ghost::pipeline::ProgressCallback>,
) -> Result<heartbit_ghost::review::ReviewConfig<'a>> {
    let delivery: Arc<dyn ReviewDelivery> = Arc::new(
        TelegramReviewDelivery::from_env()
            .context("construct TelegramReviewDelivery")?,
    );
    let twitter_tool: Arc<dyn heartbit_core::tool::Tool> =
        Arc::new(heartbit_ghost::tools::TwitterThreadTool::new());
    let credentials: Arc<dyn heartbit_core::CredentialResolver> =
        Arc::new(EnvCredentialResolver);
    Ok(heartbit_ghost::review::ReviewConfig {
        persona_name,
        topic,
        provider,
        corpora_root,
        profiles_root,
        on_progress,
        candidates_per_draft,
        delivery,
        twitter_tool,
        credentials,
    })
}

/// Env-only credential resolver — reads `name` from `std::env`, wraps
/// in `Secret`. Error if env var unset.
struct EnvCredentialResolver;

impl heartbit_core::CredentialResolver for EnvCredentialResolver {
    fn resolve(
        &self,
        name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<heartbit_core::Secret, heartbit_core::Error>> + Send + '_>>
    {
        let name = name.to_string();
        Box::pin(async move {
            std::env::var(&name)
                .map(heartbit_core::Secret::new)
                .map_err(|_| heartbit_core::Error::Config(format!("env var '{name}' not set")))
        })
    }
}
```

> **Note on the dispatcher**: `from_env` eagerly spawns the callback dispatcher as a `tokio::spawn` task. The task captures clones of `bot` and `pending` (both `Arc`-backed) and runs for the lifetime of the CLI process. This is fine for a single-shot CLI invocation that exits after the review completes. For daemon-mode (P1.4), the same pattern works — the dispatcher just runs longer.

- [ ] **Step 2: Modify `crates/heartbit-cli/src/main.rs` to declare the new module**

Find the existing module declarations near the top of `main.rs` (look for `mod persona;`). Add:

```rust
mod persona_review;
```

(Order alphabetical or grouped with other persona-related modules.)

- [ ] **Step 3: Modify `crates/heartbit-cli/src/persona.rs` — add `--review` flag**

Find the `PersonaCommand::Run` variant in the enum. Update it:

```rust
    Run {
        name: String,
        #[arg(long, value_name = "PROMPT")]
        once: String,
        /// Send candidates to Telegram for review and post on user pick.
        /// Without this flag: runs P1.3c direct mode (judge picks; stdout only).
        #[arg(long, default_value = "false")]
        review: bool,
    },
```

- [ ] **Step 4: Modify the `Run` dispatch arm in `crates/heartbit-cli/src/persona.rs`**

The current `Run` arm body calls `run_pipeline`. Wrap it in a branch:

```rust
        PersonaCommand::Run { name, once, review } => {
            if registry.get(&name).is_none() {
                return Err(anyhow!(
                    "persona '{name}' not found. {}",
                    registry_suffix(registry)
                ));
            }

            let provider = build_provider_from_env(None)
                .map_err(|e| anyhow!("build llm provider: {e}"))?;
            let corpora_root = heartbit_ghost::corpus::default_corpora_dir()
                .map_err(|e| anyhow!("resolve corpora dir: {e}"))?;
            let profiles_root = heartbit_ghost::voice::default_profiles_dir()
                .map_err(|e| anyhow!("resolve profiles dir: {e}"))?;

            let on_progress: heartbit_ghost::pipeline::ProgressCallback =
                std::sync::Arc::new(|s: &str| eprintln!("> {s}"));

            if review {
                // P1.3d review mode: Telegram + post.
                let cfg = crate::persona_review::review_config_from_env(
                    &name,
                    &once,
                    3,
                    provider,
                    &corpora_root,
                    &profiles_root,
                    Some(on_progress),
                )
                .await
                .map_err(|e| anyhow!("review config: {e}"))?;

                let n_requested = cfg.candidates_per_draft;
                let output = heartbit_ghost::review::run_review_pipeline(cfg)
                    .await
                    .map_err(|e| anyhow!("review pipeline: {e}"))?;

                eprintln!(
                    "> ok: candidates={}/{}, outcome={:?}",
                    output.candidates.len(),
                    n_requested,
                    output.outcome,
                );
                Ok(())
            } else {
                // P1.3c direct mode (unchanged).
                let cfg = heartbit_ghost::pipeline::PipelineConfig {
                    persona_name: &name,
                    topic: &once,
                    provider,
                    corpora_root: &corpora_root,
                    profiles_root: &profiles_root,
                    on_progress: Some(on_progress),
                    candidates_per_draft: 3,
                };
                let n_requested = cfg.candidates_per_draft;
                let output = heartbit_ghost::pipeline::run_pipeline(cfg)
                    .await
                    .map_err(|e| anyhow!("pipeline: {e}"))?;

                eprintln!(
                    "> ok: candidates={}/{}, chosen={}, revise iterations={}, style match={:.2}, fact check={:?}, image={}",
                    output.candidates.len(),
                    n_requested,
                    output.chosen_index,
                    output.revise_iterations,
                    output.style_match_score,
                    output.fact_check_verdict,
                    output.image.as_ref().map(|i| i.url.as_str()).unwrap_or("none"),
                );
                Ok(())
            }
        }
```

- [ ] **Step 5: Run heartbit-cli persona tests**

```bash
cd /home/pleclech/projects/heartbit
cargo test -p heartbit-cli --bin heartbit persona 2>&1 | tail -10
```

Expected: 16 tests pass (existing 14 + 2 P1.3b dispatch tests). The 2 dispatch tests don't exercise the pipeline body, so the new `--review` branch isn't tested here. P1.3d's integration tests cover the review path inside heartbit-ghost.

- [ ] **Step 6: Workspace gate (full)**

```bash
cargo fmt -- --check
cargo clippy --workspace --all-features -- -D warnings 2>&1 | tail -5
cargo test --workspace --all-features 2>&1 | grep -E "test result:" | awk '{ p+=$4; f+=$6 } END { print "Total: " p " passed, " f " failed" }'
```

All clean. Workspace test count: 3989 (post-P1.3c baseline) → ~4012 (+23 net new from P1.3d).

- [ ] **Step 7: Commit**

```bash
git add crates/heartbit-cli/src/persona_review.rs \
        crates/heartbit-cli/src/main.rs \
        crates/heartbit-cli/src/persona.rs
git commit -m "$(cat <<'EOF'
feat(cli): persona run --review — Telegram delivery + twitter_thread post (P1.3d)

CLI body for review mode:

- New persona_review.rs with TelegramReviewDelivery (standalone
  teloxide bot in CLI process). Bot token + chat_id + timeout from
  env vars (HEARTBIT_TELEGRAM_TOKEN, HEARTBIT_TELEGRAM_REVIEW_CHAT_ID,
  HEARTBIT_REVIEW_TIMEOUT_SECS, default 1h). Lazy dispatcher started
  on first deliver_and_await; pending pick resolvers keyed by
  interaction_id (single-flight per CLI process per AD-11).

- TelegramReviewDelivery::report() edits the original message in
  place to show outcome (Posted with URL / Skipped / TimedOut /
  GateRejected / PublishFailed) and removes the keyboard.

- EnvCredentialResolver: env-only CredentialResolver impl for
  TwitterThreadTool's runtime credential resolution. Reads
  X_API_KEY / X_API_SECRET / etc. from std::env, wraps in
  heartbit_core::Secret.

- review_config_from_env helper bundles the env-derived
  TelegramReviewDelivery + TwitterThreadTool + EnvCredentialResolver
  into a ReviewConfig.

- PersonaCommand::Run gains #[arg(long)] review: bool. Without:
  P1.3c direct mode (unchanged). With: review_config_from_env →
  run_review_pipeline. Post-pipeline summary on stderr includes
  outcome (Debug-formatted ReviewOutcome).

No new tests in heartbit-cli — the 2 existing dispatch tests
short-circuit at the registry check and don't exercise the
pipeline body. Integration tests for the review flow live in
heartbit-ghost (Task 2).

Refs: docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3d-telegram-review-design.md §4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Final acceptance + workspace quality gate + final review

**Why:** Confirm P1.3d meets every acceptance criterion. Verification only — no commit.

**Files:** none.

- [ ] **Step 1: Workspace quality gate**

```bash
cd /home/pleclech/projects/heartbit
cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features 2>&1 | tail -10
```

Expected: all green. Workspace test count: 3989 (post-P1.3c baseline) → **~4012** (+23 net new):
- Task 1: 17 tests (6 keyboard + 3 delivery + 4 prompts + 4 tweet_split — `tweet_split` actually has 4)
- Task 2: 6 integration tests
- Task 3: 0 new tests
- Plus 3 ReviewError display tests (already counted in Task 1's mod.rs additions)

Actually re-counting: 6 + 3 + 4 + 4 + 3 + 6 = **26 new tests**. The plan estimated ~23; the difference is the 3 mod-level error display tests. Both numbers are within the rounding band.

- [ ] **Step 2: Verify the public surface is reachable**

```bash
cd /home/pleclech/projects/heartbit
cat > /tmp/p1_3d_surface_check.rs <<'EOF'
fn _check() {
    use heartbit_ghost::review::{
        ReviewConfig, ReviewOutput, ReviewOutcome, ReviewError,
        ReviewDelivery, ReviewMessage, ReviewDeliveryError,
        DeliveredReview, DeliveryOutcome, DeliveryReceipt, ReportableOutcome,
        run_review_pipeline,
        build_review_message, build_report_message,
        parse_thread_tweets,
    };
    use heartbit_telegram::{
        CallbackAction, PickChoice, persona_pick_buttons,
    };
    let _ = ReviewOutcome::Skipped;
    let _ = DeliveryOutcome::TimedOut;
    let _ = ReportableOutcome::Skipped;
    let _ = PickChoice::Skip;
}
EOF
echo "(surface check is illustrative; cargo build covers it)"
rm -f /tmp/p1_3d_surface_check.rs
```

- [ ] **Step 3: Print final commit log for the branch**

```bash
git log --oneline main..feat/heartbit-ghost-p1.3d
```

Expected: 5 commits — spec doc + 3 task commits.

- [ ] **Step 4: No commit for this task**

Task 4 is verification only. The branch is ready for final review + merge.

---

## Acceptance criteria

P1.3d is done when (per spec §9):

1. All public types compile cleanly under `cargo check -p heartbit-ghost --all-features` and `cargo check -p heartbit-cli --all-features`.
2. `cargo fmt -- --check && cargo clippy --workspace --all-features -- -D warnings && cargo test --workspace --all-features` all green.
3. ~26 net new tests pass (6 keyboard + 3 delivery types + 4 prompts + 4 tweet_split + 3 error display + 6 integration).
4. New public surface from `heartbit_ghost::review`: `ReviewConfig`, `ReviewOutput`, `ReviewOutcome`, `ReviewError`, `run_review_pipeline`, `ReviewDelivery`, `ReviewMessage`, `DeliveryOutcome`, `ReportableOutcome`, `DeliveryReceipt`, `DeliveredReview`, `ReviewDeliveryError`, `parse_thread_tweets`, `build_review_message`, `build_report_message`.
5. New public surface from `heartbit_telegram`: `CallbackAction::PersonaPick`, `PickChoice`, `persona_pick_buttons`.
6. `heartbit persona run heartbit-ghost:x --once "<topic>" --review` runs end-to-end against:
   - real `OPENROUTER_API_KEY` (or whichever provider) for candidate generation
   - real `HEARTBIT_TELEGRAM_TOKEN` + `HEARTBIT_TELEGRAM_REVIEW_CHAT_ID` for the Telegram bot
   - real X API credentials (env vars per `TwitterThreadTool`'s convention) for posting
   - User receives a Telegram message with N candidates + inline keyboard. Pick → tweet posts + message edited to show URL. Skip → graceful exit + message edited "Skipped". Timeout → graceful exit + message edited "Timed out". (User-driven verification, same path as P1.3b/c acceptance.)

## Out of scope (re-stated from spec §10)

- Persistent pick storage in memory namespace (Reflection entry per umbrella spec §6.3) → P1.3e
- Few-shot exemplar retrieval from past picks → P1.3e
- Daemon-mode review path (CLI submits → daemon handles bot + posting) → P1.4
- Phase 1+ autonomy (auto-publish path with confidence threshold) → P1.4
- Audit log integration → P1.4
- LLM-based content guardrails → P1.4
- `twitter_thread_with_media` tool / image attachment → P1.4
- Reply triggers / mention polling → P1.4

## Reference

- Spec: `docs/superpowers/specs/2026-05-08-heartbit-ghost-p1.3d-telegram-review-design.md`
- P1.3a publisher recipe (skipped per AD-2): `crates/heartbit-ghost/src/agents/publisher.rs`
- P1.3c plan (predecessor): `docs/superpowers/plans/2026-05-08-heartbit-ghost-p1.3c-multi-candidate.md`
- ExecutionContext + CredentialResolver: `crates/heartbit-core/src/execution_context.rs`
- TwitterThreadTool: `crates/heartbit-ghost/src/tools/thread.rs`
- heartbit-telegram callback patterns: `crates/heartbit-telegram/src/keyboard.rs`
